"""
CS2 Valve Regional Standings (VRS) Simulator  ·  v2.0
======================================================
Corrected implementation based on the official Valve VRS specification.
Source: https://github.com/ValveSoftware/counter-strike_regional_standings

Architecture
────────────
  Final Score = Factor Score (400–2000) + Glicko H2H Adjustments

Factor Score Factors  (Top-10 "bucket" results over a 6-month window)
  1. Bounty Offered   — scaled prize winnings ÷ 5th-highest normalization
  2. Bounty Collected — sum of opponents' Bounty Offered scores
  3. Opponent Network — distinct opponents beaten × time modifier (top-10)
  4. LAN Wins         — count of LAN match victories

H2H Phase
  Glicko-derived system, fixed RD = 75 (≡ Elo), processed chronologically.
  Information content (K) is scaled by the match's time modifier × event stakes.

Time Decay (corrected)
  modifier = remapValueClamped(date, WindowStart, WindowEnd, 0, 1)
  → Oldest match in window → 0,  most recent match → 1

Curve Function
  f(x) = 1 / (1 + |log₁₀(x)|)   used for event stakes & bounty normalization

Eligibility
  A team is excluded from final standings if:
    · They have 0 wins in the window, OR
    · They played fewer than 5 total matches in the window
"""

import math
import os
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
_logo_path = "VRS_Sim_Icon.png"
if os.path.exists(_logo_path):
    _page_icon = Image.open(_logo_path)
else:
    _page_icon = "🔮"

st.set_page_config(
    page_title="CS2 VRS Simulator",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS  —  CS2-dark aesthetic
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stSidebar"] { background:#0d1117; }

.rank-badge {
    display:inline-block; width:30px; height:30px;
    border-radius:50%; font-weight:700; text-align:center;
    line-height:30px; font-size:13px;
}
.rank-badge.top1  { background:#FFD700; color:#000; }
.rank-badge.top3  { background:#C0C0C0; color:#000; }
.rank-badge.top10 { background:#CD7F32; color:#fff; }
.rank-badge.rest  { background:#21262d; color:#8b949e; }

.pill-eu  { background:#1565c0; color:#fff; padding:2px 8px; border-radius:12px; font-size:11px; }
.pill-am  { background:#b71c1c; color:#fff; padding:2px 8px; border-radius:12px; font-size:11px; }
.pill-as  { background:#1b5e20; color:#fff; padding:2px 8px; border-radius:12px; font-size:11px; }

.change-up   { color:#3fb950; font-weight:600; }
.change-down { color:#f85149; font-weight:600; }
.change-same { color:#8b949e; }

.explainer-box {
    background:#161b22; border:1px solid #30363d;
    border-radius:10px; padding:18px 22px; margin:10px 0;
}
.formula-block {
    background:#0d1117; border-left:3px solid #58a6ff;
    border-radius:4px; padding:12px 16px; font-family:monospace;
    font-size:14px; margin:8px 0; color:#79c0ff;
}
.step-badge {
    display:inline-block; background:#58a6ff; color:#0d1117;
    font-weight:700; width:26px; height:26px; border-radius:50%;
    text-align:center; line-height:26px; font-size:13px;
    margin-right:8px;
}
.factor-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:16px; margin:8px 0;
}
.factor-title { font-size:16px; font-weight:700; color:#c9d1d9; }
.factor-subtitle { font-size:12px; color:#8b949e; margin-bottom:8px; }

div[data-testid="metric-container"] {
    background:#161b22; border:1px solid #30363d; border-radius:10px; padding:10px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# IMPORTS FROM VRS MODULES
# ══════════════════════════════════════════════════════════════════
from vrs_engine import (
    compute_vrs,
    DECAY_DAYS, FLAT_DAYS, DECAY_RAMP, RD_FIXED, Q_GLICKO,
    BASE_K, PRIZE_CAP, TOP_N, ON_ITERS, SEED_MIN, SEED_MAX,
    curve, event_stakes, age_weight, lerp,
    g_rd, expected_win, top_n_sum, G_FIXED
)
from data_loaders import load_valve_github_data
from data_loaders.github_loader import _find_all_dates
from data_loaders import (
    fetch_hltv_matches, load_from_cache,
    cache_exists, cache_mtime, clear_cache,
)
from data_loaders import (
    fetch_liquipedia_matches,
    load_liquipedia_from_cache,
    liquipedia_cache_exists,
    liquipedia_cache_mtime,
    clear_liquipedia_cache,
)
from utils import get_team_meta, KNOWN_META, COLOR_CYCLE
TEAM_META = KNOWN_META   # alias used throughout app


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_find_all_dates() -> list[tuple[str, str]]:
    """Cached wrapper — GitHub file list, refreshed at most once per hour."""
    return _find_all_dates()


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load_github_data(date_str: str | None,
                              year: str | None) -> dict:
    """Cached wrapper — full snapshot fetch, keyed by (date_str, year).
    Historical snapshots never change; the 1-hour TTL only matters for
    the latest date where Valve occasionally pushes updates."""
    return load_valve_github_data(date_str, year)


from utils.ui_helpers import (
    region_pill, rank_badge, change_arrow, add_meta, add_regional_rank
)



# ══════════════════════════════════════════════════════════════════
# SIDEBAR  +  DATA LOADING
# ══════════════════════════════════════════════════════════════════

# ── Discover all available dates first (lightweight, cached 1h) ────
_all_dates = _cached_find_all_dates()   # [(date_str, year), ...]

# ── Query-param navigation: must run BEFORE sidebar widgets ────────
# Team links in the dashboard use href="?team=TeamName".
# We intercept here, before the radio is instantiated, so we can
# safely pre-set session_state["main_nav"] and rerun.
_qp_team = st.query_params.get("team", "")
if _qp_team:
    _preserve_sim = st.session_state.get("sim_toggle", False)
    st.query_params.clear()
    st.session_state["main_nav"] = "🔍 Team Breakdown"
    st.session_state["bd_team"]  = _qp_team
    st.session_state["sim_toggle"] = _preserve_sim
    st.rerun()

with st.sidebar:
    st.markdown("## 🎯 CS2 VRS Simulator")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📖 How VRS Works",
        "📊 Ranking Dashboard",
        "🔍 Team Breakdown",
        "🔮 What-If Predictor",
        "⚔️ Team Comparison",
    ], label_visibility="collapsed", key="main_nav")
    st.markdown("---")

    # ── Standings Mode selector ───────────────────────────────────
    standings_mode = st.radio(
        "Standings Mode",
        ["🏛️ As Published", "📡 Updated to Today"],
        index=0,
        key="standings_mode",
        help="As Published: Official Valve standings. Updated to Today: includes all results since publication (auto-fetched).",
    )

    # ── Historical date selector (only for "As Published" mode) ────
    if _all_dates:
        _date_labels = []
        for ds, yr in _all_dates:
            try:
                d_obj = datetime.strptime(ds, "%Y_%m_%d")
                label = d_obj.strftime("%B %d, %Y")
            except Exception:
                label = ds
            _date_labels.append(label)
        _date_labels[0] += "  ← latest"

        _sel_label = st.selectbox(
            "📅 VRS Snapshot",
            _date_labels,
            index=0,
            disabled=(standings_mode == "📡 Updated to Today"),
            help=("Latest snapshot is used for Updated to Today mode."
                  if standings_mode == "📡 Updated to Today" else
                  "Valve publishes new standings monthly. Select any historical snapshot."),
        )
        if standings_mode == "📡 Updated to Today":
            _sel_date, _sel_year = _all_dates[0]
        else:
            _sel_idx   = _date_labels.index(_sel_label)
            _sel_date, _sel_year = _all_dates[_sel_idx]
    else:
        # GitHub unreachable — will use fallback
        _sel_date, _sel_year = None, None
        st.warning("⚠️ Could not reach GitHub")

    st.markdown("---")
    st.markdown(
        "**Engine v3.0** · Live data · "
        "[Valve Repo](https://github.com/ValveSoftware/counter-strike_regional_standings)"
    )

# ── Load selected snapshot (each date cached independently) ────────
_load_ph = st.empty()
with _load_ph.container():
    with st.spinner("⏳ Loading VRS data from Valve GitHub…"):
        _gd = _cached_load_github_data(_sel_date, _sel_year)

_load_ph.empty()

if _gd["error"]:
    st.warning(f"⚠️ {_gd['error']}")

cutoff_dt   = _gd["cutoff_datetime"]
cutoff_date = cutoff_dt   # alias used in page display

# ══════════════════════════════════════════════════════════════════
# BASE COMPUTATION
# ══════════════════════════════════════════════════════════════════

base_standings:      pd.DataFrame         = _gd["standings"]
base_matches:        pd.DataFrame         = _gd["matches"]
team_match_history:  dict                 = _gd.get("team_match_history", {})
bo_prizes_map:       dict                 = _gd.get("bo_prizes_map", {})

# If GitHub load failed, base_standings will be empty
if base_standings.empty:
    st.error(
        "No standings data available. "
        "Check your internet connection and reload the app."
    )
    st.stop()

# Ensure required columns present (add defaults for any missing)
for _col, _default in [
    ("region","Global"), ("flag","🌍"), ("color","#58a6ff"),
    ("regional_rank", 0),
]:
    if _col not in base_standings.columns:
        base_standings[_col] = _default

if "regional_rank" not in base_standings.columns or base_standings["regional_rank"].eq(0).all():
    base_standings = add_regional_rank(base_standings)


def compute_standings(
    extra_matches: pd.DataFrame = None,
    extra_prizes: list = None,
    cutoff: datetime = None,
) -> pd.DataFrame:
    """
    For the scenario simulator.
    Starts from Valve's published scores and applies:
      1. BO adjustments from extra_prizes (recalculates seed for affected teams)
      2. H2H deltas from extra_matches
    """
    if cutoff is None:
        cutoff = datetime.now()

    result = base_standings.copy()

    # ── Step 1: BO adjustment from new prize earnings ─────────────
    if extra_prizes:
        from collections import defaultdict
        new_contribs: dict[str, float] = defaultdict(float)
        for p in extra_prizes:
            aw = age_weight(p["date"] if isinstance(p["date"], datetime)
                            else datetime.combine(p["date"], datetime.min.time()),
                            cutoff)
            new_contribs[p["team"]] += p["prize"] * aw

        # Update bo_sum for teams with new prizes
        for team, extra_bo in new_contribs.items():
            mask = result["team"] == team
            if mask.any():
                result.loc[mask, "bo_sum"] = result.loc[mask, "bo_sum"] + extra_bo

        # Recompute bo_factor for all teams
        all_bo = result["bo_sum"].values
        sorted_bo = sorted(all_bo, reverse=True)
        ref_5th = max(sorted_bo[4] if len(sorted_bo) >= 5 else sorted_bo[-1], 1e-9)
        result["bo_factor"] = result["bo_sum"].apply(
            lambda s: curve(min(1.0, s / ref_5th))
        )

        # Recompute seed from updated factors
        result["seed_combined"] = (
            result["bo_factor"] + result["bc_factor"] +
            result["on_factor"] + result["lan_factor"]
        ) / 4.0
        min_c = result["seed_combined"].min()
        max_c = result["seed_combined"].max()
        span  = max(max_c - min_c, 1e-9)
        result["seed"] = result["seed_combined"].apply(
            lambda c: lerp(SEED_MIN, SEED_MAX, (c - min_c) / span)
        )
        result["total_points"] = result["seed"] + result["h2h_delta"]

    # ── Step 2: H2H deltas from new match results ─────────────────
    scores:    dict[str, float] = result.set_index("team")["total_points"].to_dict()
    h2h_extra: dict[str, float] = {t: 0.0 for t in scores}

    if extra_matches is not None and not extra_matches.empty:
        _series = extra_matches[extra_matches["loser"] != ""] if "loser" in extra_matches.columns else extra_matches
        for _, row in _series.sort_values("date").iterrows():
            w, l = str(row["winner"]), str(row["loser"])
            if w not in scores or l not in scores:
                continue
            E_w = expected_win(scores[w], scores[l])
            K   = BASE_K * 1.0
            d_w =  K * (1.0 - E_w)
            d_l = -K * E_w
            scores[w]    += d_w;  scores[l]    += d_l
            h2h_extra[w] += d_w;  h2h_extra[l] += d_l

    result["total_points"] = result["team"].map(scores).fillna(result["total_points"])
    result["h2h_delta"]    = result["team"].map(
        lambda t: result.loc[result["team"] == t, "h2h_delta"].iloc[0] + h2h_extra.get(t, 0.0)
    )
    result = result.sort_values("total_points", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


# ══════════════════════════════════════════════════════════════════
# AUTO-FETCH "UPDATED TO TODAY" ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def _identify_active_rosters(standings: pd.DataFrame) -> dict:
    """
    Dynamically identify which roster version is 'active' for each team.

    Returns: dict[team_name] -> original_row_index_of_active_roster

    Active roster = the one with the lowest rank (best ranking).
    This is determined at runtime, no hard-coded lists.
    For future-proofing: as rosters change next month, the function
    automatically identifies the new active one.
    """
    active_map = {}

    for team_name in standings["team"].unique():
        team_rows = standings[standings["team"] == team_name]

        if len(team_rows) == 1:
            # No duplicates, trivially active
            active_map[team_name] = team_rows.index[0]
        else:
            # Multiple rosters: pick the best-ranked one
            best_idx = team_rows["rank"].idxmin()
            active_map[team_name] = best_idx

    return active_map


def _auto_fetch_updated_standings(
    snapshot_cutoff: datetime,
    snapshot_standings: pd.DataFrame,
    team_match_history: dict,
    bo_prizes_map: dict,
    progress_callback=None,
) -> dict:
    """
    Auto-fetch all tournaments from Liquipedia since snapshot_cutoff,
    inject them, and recompute standings with shifted cutoff to today.

    progress_callback: optional function(step: int, total: int, status: str) for UI updates

    Returns: {
        "standings": updated DataFrame or empty on error,
        "cutoff": datetime.now(),
        "source": "liquipedia" or "fallback",
        "match_count": int,
        "error": str or None,
    }
    """
    from data_loaders import (
        discover_liquipedia_from_portal,
        fetch_liquipedia_matches,
        liquipedia_cache_exists,
        liquipedia_cache_mtime,
        load_liquipedia_from_cache,
    )

    today = datetime.now()
    start_date_str = snapshot_cutoff.strftime("%Y-%m-%d")
    end_date_str = today.strftime("%Y-%m-%d")

    # ── Step 1: Check cache ────────────────────────────────────────
    cache_key_base = (start_date_str, end_date_str)
    try:
        if progress_callback:
            progress_callback(1, 3, "Discovering tournaments from Liquipedia Portal…")

        # Attempt auto-discover to get tournament list for cache check
        discovered = discover_liquipedia_from_portal(
            start_date_str, end_date_str, min_tier="B-Tier", include_qualifiers=True
        )
        if not discovered:
            # No tournaments found in date range
            return {
                "standings": pd.DataFrame(),
                "cutoff": today,
                "source": "fallback",
                "match_count": 0,
                "error": "No tournaments found in Liquipedia for this date range."
            }

        tournament_slugs = [d["slug"] for d in discovered]

        if progress_callback:
            progress_callback(2, 3, f"Fetching {len(tournament_slugs)} tournaments…")

        # Check if cache exists and is fresh (< 2 hours old)
        cache_fresh = False
        if liquipedia_cache_exists(start_date_str, end_date_str, tournament_slugs):
            mtime = liquipedia_cache_mtime(start_date_str, end_date_str, tournament_slugs)
            age_minutes = (datetime.now() - mtime).total_seconds() / 60
            if age_minutes < 120:  # 2 hours
                cache_fresh = True
                if progress_callback:
                    progress_callback(2, 3, f"Loading {len(tournament_slugs)} tournaments from cache…")
                matches_df = load_liquipedia_from_cache(start_date_str, end_date_str, tournament_slugs)
            else:
                # Cache is stale, re-fetch
                matches_df = fetch_liquipedia_matches(
                    start_date_str, end_date_str,
                    tournament_slugs=tournament_slugs,
                    force_refresh=True,
                    progress_callback=progress_callback,
                )
        else:
            # No cache, fetch fresh
            matches_df = fetch_liquipedia_matches(
                start_date_str, end_date_str,
                tournament_slugs=tournament_slugs,
                force_refresh=False,
                progress_callback=progress_callback,
            )

        if matches_df.empty:
            return {
                "standings": pd.DataFrame(),
                "cutoff": today,
                "source": "fallback",
                "match_count": 0,
                "error": "Liquipedia fetch returned no matches."
            }

        # ── Step 2: Filter to active rosters only (Layer 1) ──────────
        # Identify active rosters from the SNAPSHOT standings BEFORE injection
        active_rosters = _identify_active_rosters(snapshot_standings)

        # Build a set of active team names for filtering
        active_team_names = set(
            snapshot_standings.loc[active_rosters[team], "team"]
            for team in active_rosters
        )

        # Separate series from prize-only rows
        series_df = matches_df[matches_df["loser"] != ""].copy()
        prize_df = matches_df[matches_df["loser"] == ""].copy()

        # Filter: keep matches where BOTH teams are either:
        #   - Active rosters (if they have duplicates), OR
        #   - Not in snapshot at all (unknown teams)
        series_df_filtered = []
        for _, row in series_df.iterrows():
            winner = row["winner"]
            loser = row["loser"]

            # Check if this match should be kept
            winner_ok = winner not in snapshot_standings["team"].values or winner in active_team_names
            loser_ok = loser not in snapshot_standings["team"].values or loser in active_team_names

            if winner_ok and loser_ok:
                series_df_filtered.append(row)

        series_df = pd.DataFrame(series_df_filtered) if series_df_filtered else pd.DataFrame()

        # Same for prizes: only apply to active rosters
        prize_df_filtered = []
        for _, row in prize_df.iterrows():
            team = row["winner"]
            if team not in snapshot_standings["team"].values or team in active_team_names:
                prize_df_filtered.append(row)

        prize_df = pd.DataFrame(prize_df_filtered) if prize_df_filtered else pd.DataFrame()

        # ── Step 3: Extend tmh + bpm with Liquipedia data ──────────────
        if progress_callback:
            progress_callback(3, 3, "Merging new matches and recalculating all factors…")

        # Build opponent factor lookup for new match entries
        bo_lookup = snapshot_standings.set_index("team")["bo_factor"].to_dict()
        on_lookup = snapshot_standings.set_index("team")["on_factor"].to_dict()

        # Convert Liquipedia series results → tmh match-dict format.
        # ev_w uses the same event_stakes(prize_pool) formula as Valve.
        # opp_bo/opp_on are fallbacks; _simulate_time_decay looks up live values first.
        new_tmh: dict = {}
        match_id_offset = 10_000_000  # well above Valve's match IDs
        for i, row in (series_df.iterrows() if not series_df.empty else iter([])):
            winner     = str(row["winner"])
            loser      = str(row["loser"])
            match_date = row["date"] if isinstance(row["date"], datetime) \
                         else datetime.combine(row["date"], datetime.min.time())
            prize_pool = float(row.get("prize_pool", 0.0))
            is_lan     = bool(row.get("is_lan", False))
            ev_w       = event_stakes(prize_pool)
            mid        = match_id_offset + i

            for team, opp, result in [(winner, loser, "W"), (loser, winner, "L")]:
                entry = {
                    "match_id":  mid,
                    "date":      match_date,
                    "opponent":  opp,
                    "result":    result,
                    "age_w":     0.0,   # re-computed inside _simulate_time_decay
                    "ev_w":      ev_w,
                    "prize_pool": prize_pool,
                    "is_lan":    is_lan,
                    "h2h_adj":   0.0,
                    "opp_bo":    bo_lookup.get(opp, 0.0),
                    "opp_on":    on_lookup.get(opp, 0.0),
                }
                new_tmh.setdefault(team, []).append(entry)

        # Merge: original tmh + new Liquipedia match entries
        extended_tmh = {t: list(v) for t, v in team_match_history.items()}
        for team, entries in new_tmh.items():
            extended_tmh.setdefault(team, [])
            extended_tmh[team] = extended_tmh[team] + entries

        # Convert prize-only rows → bpm format for BO recalculation
        new_bpm: dict = {}
        for _, row in (prize_df.iterrows() if not prize_df.empty else iter([])):
            team          = str(row["winner"])
            prize_amount  = float(row.get("winner_prize", 0.0))
            prize_date    = row["date"]
            if prize_amount <= 0:
                continue
            date_str = prize_date.strftime("%Y-%m-%d") \
                       if hasattr(prize_date, "strftime") else str(prize_date)
            new_bpm.setdefault(team, []).append({
                "event_date":  date_str,
                "age_weight":  0.0,   # re-computed inside _simulate_time_decay
                "prize_won":   prize_amount,
                "scaled_prize": 0.0,
            })

        # Merge: original bpm + new prize entries
        extended_bpm = {t: list(v) for t, v in bo_prizes_map.items()}
        for team, prizes in new_bpm.items():
            extended_bpm.setdefault(team, [])
            extended_bpm[team] = extended_bpm[team] + prizes

        # ── Step 4: Full VRS recomputation (all 4 factors + H2H) ───────
        # This is the same pipeline Valve runs:
        #   – All matches in the 180-day window, age-weighted to today
        #   – BO, BC, ON, LAN recomputed from scratch
        #   – Glicko H2H applied chronologically
        # New Liquipedia matches are already in extended_tmh, so they
        # participate in every factor calculation at their correct age weight.
        sim_result = _simulate_time_decay(
            extended_tmh, extended_bpm,
            snapshot_standings, snapshot_cutoff, today,
        )

        if sim_result["standings"].empty:
            return {
                "standings": pd.DataFrame(),
                "cutoff": today,
                "source": "fallback",
                "match_count": len(series_df) if not series_df.empty else 0,
                "error": "Full recomputation returned no eligible teams."
            }

        return {
            "standings": sim_result["standings"],
            "match_h2h": sim_result.get("match_h2h", {}),
            "cutoff": today,
            "source": "liquipedia",
            "match_count": len(series_df) if not series_df.empty else 0,
            "error": None,
        }

    except Exception as e:
        import traceback as tb
        return {
            "standings": pd.DataFrame(),
            "cutoff": today,
            "source": "fallback",
            "match_count": 0,
            "error": f"Liquipedia fetch failed: {str(e)[:100]}"
        }


# ══════════════════════════════════════════════════════════════════
# ████████████  TIME-DECAY SIMULATION  █████████████████████████████
# ══════════════════════════════════════════════════════════════════
#
# Recomputes ALL four factors + H2H with a shifted cutoff date.
# The only thing that changes is the age_weight of each match —
# no new matches are added.  This shows the pure decay effect.
# ══════════════════════════════════════════════════════════════════

def _simulate_time_decay(
    tmh: dict,                   # team_match_history
    bpm: dict,                   # bo_prizes_map
    standings: pd.DataFrame,     # current base_standings (for meta)
    old_cutoff: datetime,
    new_cutoff: datetime,
) -> pd.DataFrame:
    """
    Full VRS recomputation with a shifted cutoff date.

    Returns a new standings DataFrame with recalculated factor scores,
    seeds, and H2H adjustments reflecting the new age weights.
    """
    # ── Layer 2: Identify and filter to active rosters only ────────
    # Instead of excluding duplicates entirely, filter to keep only the
    # active roster for each team (lowest rank = active), then proceed
    # with normal factor calculations.
    active_rosters = _identify_active_rosters(standings)

    # Filter standings to only active rosters
    standings_active_only = []
    for idx, row in standings.iterrows():
        team_name = row["team"]
        if active_rosters.get(team_name) == idx:
            standings_active_only.append(row)

    standings = pd.DataFrame(standings_active_only).reset_index(drop=True)

    # Now work with the consolidated standings (no duplicates)
    _clean_std = standings.sort_values("rank").reset_index(drop=True)
    all_teams  = _clean_std["team"].tolist()

    # Keep reference for region/flag/color data
    _ref_std = standings.sort_values("rank").drop_duplicates("team", keep="first")
    _dup_names = set()  # No duplicates anymore, but keep for compatibility

    new_window_start = new_cutoff - timedelta(days=DECAY_DAYS)

    # ── Eligibility in the new window ─────────────────────────────
    wins_ct:  dict[str, int] = {}
    match_ct: dict[str, int] = {}
    for t in all_teams:
        ms = tmh.get(t, [])
        in_w = [m for m in ms if new_window_start <= m["date"] <= new_cutoff]
        match_ct[t] = len(in_w)
        wins_ct[t]  = sum(1 for m in in_w if m["result"] == "W")

    eligible = [
        t for t in all_teams
        if wins_ct.get(t, 0) > 0 and match_ct.get(t, 0) >= 5
    ]
    if not eligible:
        return {"standings": pd.DataFrame(), "match_h2h": {}, "dup_teams": _dup_names}

    # ── Factor 1: Bounty Offered ──────────────────────────────────
    # Compute from actual prize data (bpm) with new age weights.
    # Teams with no prize data in bpm fall back to their Valve bo_sum.
    bo_sum_new: dict[str, float] = {}
    for t in all_teams:
        prizes = bpm.get(t, [])
        contribs = []
        for bp in prizes:
            try:
                dt = datetime.strptime(str(bp["event_date"]).strip(), "%Y-%m-%d")
            except Exception:
                continue
            aw = age_weight(dt, new_cutoff)
            if aw > 0:
                contribs.append(bp["prize_won"] * aw)
        if contribs:
            bo_sum_new[t] = top_n_sum(contribs, TOP_N)
        else:
            bo_sum_new[t] = float(
                _ref_std.loc[_ref_std["team"] == t, "bo_sum"].iloc[0]
                if t in _ref_std["team"].values else 0.0
            )

    # Normalise bo_sum → bo_factor (same formula as core.py)
    sorted_bo = sorted(bo_sum_new.values(), reverse=True)
    ref_5th_bo = sorted_bo[4] if len(sorted_bo) >= 5 else (sorted_bo[-1] if sorted_bo else 1.0)
    ref_5th_bo = max(ref_5th_bo, 1e-9)
    bo_f: dict[str, float] = {
        t: curve(min(1.0, bo_sum_new[t] / ref_5th_bo))
        for t in all_teams
    }

    # ── Factor 2: Bounty Collected ────────────────────────────────
    # Use bo_f for opponents in standings; fall back to the per-match
    # opp_bo parsed from Valve's detail page (covers small teams not
    # in the published standings).
    bc_f:   dict[str, float] = {}
    bc_pre: dict[str, float] = {}
    for t in eligible:
        entries = []
        for m in tmh.get(t, []):
            if m["result"] != "W" or m.get("ev_w", 0) <= 0:
                continue
            if not (new_window_start <= m["date"] <= new_cutoff):
                continue
            aw = age_weight(m["date"], new_cutoff)
            opp_bo_val = bo_f.get(m["opponent"], m.get("opp_bo", 0.0))
            entries.append(opp_bo_val * aw * m["ev_w"])
        s = top_n_sum(entries, TOP_N) / TOP_N
        bc_pre[t] = s
        bc_f[t]   = curve(s)

    # ── Factor 3: Opponent Network ─────────────────────────────────
    on_f: dict[str, float] = dict(bo_f)   # seed from BO factors (same as core.py)
    for _iter in range(ON_ITERS):
        new_on: dict[str, float] = {}
        for t in eligible:
            entries = []
            for m in tmh.get(t, []):
                if m["result"] != "W" or m.get("ev_w", 0) <= 0:
                    continue
                if not (new_window_start <= m["date"] <= new_cutoff):
                    continue
                aw = age_weight(m["date"], new_cutoff)
                opp_on_val = on_f.get(m["opponent"], m.get("opp_on", 0.0))
                entries.append(opp_on_val * aw * m["ev_w"])
            new_on[t] = top_n_sum(entries, TOP_N) / TOP_N
        on_f.update(new_on)
    on_final: dict[str, float] = on_f

    # ── Factor 4: LAN Wins ────────────────────────────────────────
    lan_f:  dict[str, float] = {}
    lan_ct: dict[str, int]   = {}
    for t in eligible:
        lw = [m for m in tmh.get(t, [])
              if m["result"] == "W" and m.get("is_lan")
              and new_window_start <= m["date"] <= new_cutoff]
        lan_ct[t] = len(lw)
        entries   = [age_weight(m["date"], new_cutoff) for m in lw]
        lan_f[t]  = top_n_sum(entries, TOP_N) / TOP_N

    # ── Combine → Seed ────────────────────────────────────────────
    combined: dict[str, float] = {
        t: (bo_f.get(t, 0) + bc_f.get(t, 0)
            + on_final.get(t, 0) + lan_f.get(t, 0)) / 4.0
        for t in eligible
    }
    avg_vals = list(combined.values())
    min_avg  = min(avg_vals)
    max_avg  = max(avg_vals)
    span     = max(max_avg - min_avg, 1e-9)
    seeds: dict[str, float] = {
        t: lerp(SEED_MIN, SEED_MAX, (combined[t] - min_avg) / span)
        for t in eligible
    }

    # ── H2H (chronological) ──────────────────────────────────────
    ratings: dict[str, float] = {t: seeds[t] for t in eligible}
    h2h_d:   dict[str, float] = {t: 0.0     for t in eligible}

    seen_ids: set[int] = set()
    all_ms: list[tuple] = []
    for t in eligible:
        for m in tmh.get(t, []):
            if m["match_id"] in seen_ids or m["result"] != "W":
                continue
            if not (new_window_start <= m["date"] <= new_cutoff):
                continue
            seen_ids.add(m["match_id"])
            all_ms.append((m["date"], t, m["opponent"], m["match_id"]))
    all_ms.sort()

    # Per-match H2H tracking for team breakdown display
    match_h2h: dict[int, dict] = {}   # match_id → {winner, loser, w_delta, l_delta}

    for dt, w, l, mid in all_ms:
        if w not in ratings or l not in ratings:
            continue
        E_w = expected_win(ratings[w], ratings[l])
        K   = BASE_K * age_weight(dt, new_cutoff)
        d_w =  K * (1.0 - E_w)
        d_l = -K * E_w
        ratings[w] += d_w;  ratings[l] += d_l
        h2h_d[w]   += d_w;  h2h_d[l]   += d_l
        match_h2h[mid] = {"winner": w, "loser": l, "w_delta": d_w, "l_delta": d_l}

    # ── Build DataFrame ───────────────────────────────────────────
    region_map = _ref_std.set_index("team")["region"].to_dict()
    flag_map   = _ref_std.set_index("team")["flag"].to_dict()
    color_map  = _ref_std.set_index("team")["color"].to_dict()

    records = []
    for t in eligible:
        seed = seeds[t]
        h2h  = h2h_d[t]
        records.append({
            "team":          t,
            "total_points":  round(seed + h2h, 1),
            "seed":          round(seed, 1),
            "h2h_delta":     round(h2h, 1),
            "bo_sum":        round(bo_sum_new.get(t, 0), 2),
            "bo_factor":     round(bo_f.get(t, 0), 4),
            "bc_pre_curve":  round(bc_pre.get(t, 0), 4),
            "bc_factor":     round(bc_f.get(t, 0), 4),
            "on_factor":     round(on_final.get(t, 0), 4),
            "on_raw":        round(on_final.get(t, 0), 4),
            "lan_factor":    round(lan_f.get(t, 0), 4),
            "lan_wins":      lan_ct.get(t, 0),
            "seed_combined": round(combined.get(t, 0), 4),
            "wins":          wins_ct.get(t, 0),
            "losses":        match_ct.get(t, 0) - wins_ct.get(t, 0),
            "total_matches": match_ct.get(t, 0),
            "region":        region_map.get(t, "Global"),
            "flag":          flag_map.get(t, "🌍"),
            "color":         color_map.get(t, "#58a6ff"),
        })

    result = pd.DataFrame(records)

    # ── Sort and rank ─────────────────────────────────────────────
    # (No duplicate merging needed — we filtered to active rosters only)
    result = (result.sort_values("total_points", ascending=False)
              .reset_index(drop=True))
    result["rank"] = result.index + 1
    result = add_regional_rank(result)
    return {"standings": result, "match_h2h": match_h2h,
            "dup_teams": set()}


# ══════════════════════════════════════════════════════════════════
# STANDINGS MODE EXECUTION
# ══════════════════════════════════════════════════════════════════

mode_active          = "published"  # "published", "updated", or "fallback"
mode_fetch_error     = None
updated_match_count  = 0
original_standings   = base_standings.copy()
sim_match_h2h        = {}  # match_id → {winner, loser, w_delta, l_delta} from simulation
sim_cutoff_dt        = datetime.now()  # cutoff used for simulation (for age_weight calc)

if standings_mode == "📡 Updated to Today":
    _prog_bar = st.progress(0, text="📡 Fetching live tournaments…")
    _prog_text = st.empty()

    def _fetch_progress(step: int, total: int, status: str):
        _prog_bar.progress(min(step / max(total, 1), 0.95), text=status)
        _prog_text.caption(f"{status} ({step}/{total})")

    try:
        _fetch_result = _auto_fetch_updated_standings(
            cutoff_dt, base_standings, team_match_history, bo_prizes_map,
            progress_callback=_fetch_progress,
        )
        _prog_bar.progress(1.0, text="✅ Done")
        _prog_text.empty()
    except Exception as e:
        _prog_bar.empty()
        _prog_text.empty()
        _fetch_result = {
            "standings": pd.DataFrame(),
            "cutoff": datetime.now(),
            "source": "fallback",
            "match_count": 0,
            "error": f"Fetch error: {str(e)[:100]}"
        }

    if _fetch_result["standings"].empty:
        # Fetch failed, fallback to As Published
        mode_active = "fallback"
        mode_fetch_error = _fetch_result.get("error", "Unknown error during fetch")
        st.warning(f"⚠️ Could not fetch live data: {mode_fetch_error}")
    else:
        # Success: update standings, cutoff, and state
        original_standings = base_standings.copy()
        base_standings = _fetch_result["standings"]
        updated_match_count = _fetch_result.get("match_count", 0)
        cutoff_dt = _fetch_result["cutoff"]
        cutoff_date = _fetch_result["cutoff"]
        sim_cutoff_dt = _fetch_result["cutoff"]  # use simulation cutoff for age_weight calcs
        sim_match_h2h = _fetch_result.get("match_h2h", {})
        mode_active = "updated"

        # Compute rank deltas
        _orig_rank_map = original_standings.sort_values("rank").drop_duplicates("team", keep="first").set_index("team")["rank"].to_dict()
        base_standings["rank_delta"] = base_standings.apply(
            lambda r: (int(_orig_rank_map.get(r["team"], r["rank"])) - int(r["rank"])),
            axis=1,
        ).astype(int)
else:
    # As Published mode: use snapshot as-is
    mode_active = "published"
    if "rank_delta" not in base_standings.columns:
        base_standings["rank_delta"] = 0

# ── Mode banner (shown on all pages) ────────────────────────────────
if mode_active == "updated":
    _days_fwd = (cutoff_dt - _gd["cutoff_datetime"]).days
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d3d1a,#0d1117);border:1px solid #3fb950;
                border-radius:10px;padding:16px 20px;margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="font-size:22px">📡</span>
        <span style="font-size:16px;font-weight:700;color:#3fb950;">
          Updated to {cutoff_dt.strftime('%B %d, %Y')}</span>
        <span style="background:#3fb950;color:#000;border-radius:12px;padding:2px 10px;
                     font-size:11px;font-weight:600">+{updated_match_count} tournaments</span>
      </div>
      <div style="font-size:12px;color:#56d364;line-height:1.6;">
        Fetched all official tournaments and recomputed standings with latest results.
        All four factors (BO, BC, ON, LAN) and H2H recalculated as of {cutoff_dt.strftime('%b %d, %H:%M UTC')}.
      </div>
    </div>""", unsafe_allow_html=True)

elif mode_active == "fallback":
    st.warning(f"⚠️ Using As Published standings (fetch failed: {mode_fetch_error[:80]})")

# ── Sidebar status (after mode state is resolved) ───────────────────
with st.sidebar:
    if mode_active == "updated":
        st.success(f"📡 Updated  ·  {cutoff_dt.strftime('%Y_%m_%d')}")
    elif mode_active == "published":
        st.success(f"🏛️ Official  ·  {_gd['cutoff_date']}")
    else:  # fallback
        st.warning(f"⚠️ Fallback  ·  {_gd['cutoff_date']}")


# ══════════════════════════════════════════════════════════════════
# PAGE 1  ·  RANKING DASHBOARD
# ══════════════════════════════════════════════════════════════════

if page == "📊 Ranking Dashboard":
    st.title("📊 CS2 Valve Regional Standings")

    # ── Mode banner ─────────────────────────────────────────────────
    if mode_active == "updated":
        st.info(f"📡 **Updated to {cutoff_dt.strftime('%B %d, %Y')}** — includes {updated_match_count} tournaments since publication")
    elif mode_active == "published":
        st.info(f"🏛️ **Official Valve standings** — as of {cutoff_dt.strftime('%B %d, %Y')}")
    elif mode_active == "fallback":
        st.warning(f"⚠️ **Fallback to official standings** — {mode_fetch_error}")

    st.caption(
        f"Cutoff **{cutoff_date.strftime('%B %d, %Y')}** · "
        "Two-phase VRS engine (Factor Score + Glicko H2H) · "
        "Source: [Valve GitHub](https://github.com/ValveSoftware/counter-strike_regional_standings)"
    )

    if base_standings.empty:
        st.error("No eligible teams for the selected cutoff window.")
        st.stop()

    # ── KPI row ───────────────────────────────────────────────────
    top  = base_standings.iloc[0]
    eu   = base_standings[base_standings["region"] == "Europe"].iloc[0]
    am   = base_standings[base_standings["region"] == "Americas"].iloc[0]
    asia = base_standings[base_standings["region"] == "Asia"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌍 Global #1",   top["team"],  f'{top["total_points"]:,.0f} pts')
    c2.metric("🇪🇺 Europe #1",  eu["team"],   f'{eu["total_points"]:,.0f} pts')
    c3.metric("🌎 Americas #1", am["team"],   f'{am["total_points"]:,.0f} pts')
    c4.metric("🌏 Asia #1",     asia["team"], f'{asia["total_points"]:,.0f} pts')

    # ── Mode impact summary (only when updated) ──────────────────────
    if mode_active == "updated":
        st.markdown("---")
        st.markdown("### 📉 Rankings Impact (vs Official)")
        _orig_map  = original_standings.sort_values("rank").drop_duplicates("team", keep="first").set_index("team")
        _sim_map   = base_standings.sort_values("rank").drop_duplicates("team", keep="first").set_index("team")
        # Compare all teams that appear in both
        _both      = sorted(set(_orig_map.index) & set(_sim_map.index))

        _deltas = []
        for t in _both:
            old_pts  = float(_orig_map.at[t, "total_points"])
            new_pts  = float(_sim_map.at[t, "total_points"])
            old_rank = int(_orig_map.at[t, "rank"])
            new_rank = int(_sim_map.at[t, "rank"])
            flag_v   = _sim_map.at[t, "flag"] if "flag" in _sim_map.columns else "🌍"
            _deltas.append({
                "team": t,
                "old_rank": old_rank, "new_rank": new_rank,
                "rank_delta": old_rank - new_rank,
                "old_pts": old_pts, "new_pts": new_pts,
                "pts_delta": new_pts - old_pts,
                "flag": flag_v,
            })
        _deltas_df = pd.DataFrame(_deltas).sort_values("pts_delta")

        # KPI: average decay, biggest winner/loser
        avg_decay   = _deltas_df["pts_delta"].mean()
        biggest_drop = _deltas_df.iloc[0]
        biggest_rise = _deltas_df.iloc[-1]
        teams_dropped = int((_deltas_df["rank_delta"] < 0).sum())
        teams_rose    = int((_deltas_df["rank_delta"] > 0).sum())

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Ø Pts Change", f"{avg_decay:+.1f}")
        s2.metric("Biggest Drop",
                  f'{biggest_drop["team"]}',
                  f'{biggest_drop["pts_delta"]:+.1f} pts')
        s3.metric("Biggest Rise",
                  f'{biggest_rise["team"]}',
                  f'{biggest_rise["pts_delta"]:+.1f} pts')
        s4.metric("Rank Changes", f"↑{teams_rose}  ↓{teams_dropped}")

        # Top movers bar chart
        movers = pd.concat([_deltas_df.head(10), _deltas_df.tail(10)]).drop_duplicates("team")
        movers = movers.sort_values("pts_delta")
        fig_decay = go.Figure(go.Bar(
            x=movers["pts_delta"], y=movers["team"], orientation="h",
            marker_color=["#f85149" if v < 0 else "#3fb950" for v in movers["pts_delta"]],
            text=[f"{v:+.1f}" for v in movers["pts_delta"]],
            textposition="outside",
        ))
        fig_decay.update_layout(
            title="Points Change (Top Movers — Simulation vs Actual)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            xaxis=dict(gridcolor="#21262d", title="Δ Points"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=10, r=80, t=40, b=10),
            height=max(300, len(movers) * 26),
        )
        st.plotly_chart(fig_decay, use_container_width=True, config={"staticPlot": True})

    st.markdown("---")

    # ── Table renderer ────────────────────────────────────────────
    # Pre-compute per-factor maximums across all teams for relative bar scaling
    _bo_max  = max(base_standings["bo_factor"].max(),  1e-9)
    _bc_max  = max(base_standings["bc_factor"].max(),  1e-9)
    _on_max  = max(base_standings["on_factor"].max(),  1e-9)
    _lan_max = max(base_standings["lan_factor"].max(), 1e-9)

    def factor_bar(value: float, color: str = "#58a6ff", width: int = 60,
                   max_val: float = 1.0) -> str:
        """Mini progress bar scaled to max_val across all teams."""
        pct = max(0.0, min(1.0, float(value) / max(float(max_val), 1e-9))) * 100
        return (
            f'<div style="display:flex;align-items:center;gap:5px;">' +
            f'<div style="width:{width}px;height:7px;background:#21262d;border-radius:4px;overflow:hidden;">' +
            f'<div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:4px;"></div></div>' +
            f'<span style="font-size:11px;color:#8b949e;min-width:32px">{float(value):.3f}</span>' +
            '</div>'
        )

    def render_table(df: pd.DataFrame, show_all: bool = False):
        display = df.copy() if show_all else df.head(50)
        rows = []
        for _, row in display.iterrows():
            rk   = int(row["rank"])
            rd   = int(row.get("rank_delta", 0)) if mode_active == "updated" else 0
            rd_s = (f' <span class="change-up" style="font-size:10px">▲{rd}</span>' if rd > 0 else
                    f' <span class="change-down" style="font-size:10px">▼{abs(rd)}</span>' if rd < 0 else
                    '') if rd != 0 else ''
            flag = row.get("flag", "🌍")
            team = row["team"]
            pts  = f'{row["total_points"]:,.1f}'
            seed = f'{row["seed"]:,.1f}'
            h2h  = float(row["h2h_delta"])
            h2h_s = (f'<span class="change-up">+{h2h:.1f}</span>' if h2h > 0 else
                     f'<span class="change-down">{h2h:.1f}</span>'  if h2h < 0 else
                     '<span class="change-same">0.0</span>')
            region_str = row.get("region", "Global")
            rrk  = int(row.get("regional_rank", 0))
            reg  = region_pill(region_str, rrk)
            w    = int(row.get("wins", 0))
            l    = int(row.get("losses", 0))
            bo   = factor_bar(row.get("bo_factor", 0), "#f0b429", max_val=_bo_max)
            bc   = factor_bar(row.get("bc_factor", 0), "#3fb950", max_val=_bc_max)
            on   = factor_bar(row.get("on_factor", 0), "#79c0ff", max_val=_on_max)
            lan  = factor_bar(row.get("lan_factor", 0), "#f85149", max_val=_lan_max)
            rows.append(f"""
            <tr style="border-bottom:1px solid #21262d;">
              <td style="padding:6px 4px;text-align:center">{rank_badge(rk)}{rd_s}</td>
              <td style="padding:6px 4px">{flag} <a href="?team={team}" target="_self" style="color:#e6edf3;text-decoration:none;font-weight:700;cursor:pointer" title="Open Team Breakdown">{team}</a></td>
              <td style="padding:6px 4px;text-align:center">{reg}</td>
              <td style="padding:6px 8px;text-align:right;color:#58a6ff;font-weight:700;font-size:14px">{pts}</td>
              <td style="padding:6px 8px;text-align:right;color:#8b949e">{seed}</td>
              <td style="padding:6px 8px;text-align:right">{h2h_s}</td>
              <td style="padding:6px 8px">{bo}</td>
              <td style="padding:6px 8px">{bc}</td>
              <td style="padding:6px 8px">{on}</td>
              <td style="padding:6px 8px">{lan}</td>
              <td style="padding:6px 4px;text-align:center;color:#8b949e;font-size:12px">{w}W/{l}L</td>
            </tr>""")
        st.markdown(f"""
        <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
          <thead>
            <tr style="background:#161b22;color:#8b949e;font-size:10px;text-transform:uppercase;letter-spacing:.05em;">
              <th style="padding:9px 4px;text-align:center">Rank</th>
              <th style="padding:9px 4px;text-align:left">Team</th>
              <th style="padding:9px 4px;text-align:center">Region</th>
              <th style="padding:9px 8px;text-align:right" title="Factor Score + H2H adjustment">Total</th>
              <th style="padding:9px 8px;text-align:right" title="Factor Score (400–2000)">Factor Score</th>
              <th style="padding:9px 8px;text-align:right" title="Glicko H2H adjustment">H2H Δ</th>
              <th style="padding:9px 8px;text-align:left" title="Bounty Offered (prize money)">
                <span style="color:#f0b429">■</span> BO</th>
              <th style="padding:9px 8px;text-align:left" title="Bounty Collected (opp strength)">
                <span style="color:#3fb950">■</span> BC</th>
              <th style="padding:9px 8px;text-align:left" title="Opponent Network">
                <span style="color:#79c0ff">■</span> ON</th>
              <th style="padding:9px 8px;text-align:left" title="LAN Wins">
                <span style="color:#f85149">■</span> LAN</th>
              <th style="padding:9px 4px;text-align:center">W/L</th>
            </tr>
          </thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        </div>
        <div style="font-size:11px;color:#8b949e;margin-top:6px;padding:4px 0">
          <span style="color:#f0b429">■ BO</span> Bounty Offered &nbsp;·&nbsp;
          <span style="color:#3fb950">■ BC</span> Bounty Collected &nbsp;·&nbsp;
          <span style="color:#79c0ff">■ ON</span> Opponent Network &nbsp;·&nbsp;
          <span style="color:#f85149">■ LAN</span> LAN Wins &nbsp;·&nbsp;
          Each bar = 0.000 → 1.000, averaged to produce Factor Score
        </div>
        """, unsafe_allow_html=True)

    tab_gl, tab_eu, tab_am, tab_as = st.tabs(
        ["🌍 Global", "🇪🇺 Europe", "🌎 Americas", "🌏 Asia"])

    with tab_gl:
        show_all = st.checkbox("Show all teams", key="chk_global")
        render_table(base_standings, show_all=show_all)

    with tab_eu:
        eu_df = base_standings[base_standings["region"] == "Europe"].reset_index(drop=True)
        eu_df["rank"] = eu_df.index + 1
        if mode_active == "updated":
            _orig_eu = original_standings[original_standings["region"] == "Europe"].sort_values("total_points", ascending=False).reset_index(drop=True)
            _orig_eu["rank"] = _orig_eu.index + 1
            _oem = _orig_eu.set_index("team")["rank"].to_dict()
            eu_df["rank_delta"] = eu_df.apply(lambda r: int(_oem.get(r["team"], r["rank"])) - int(r["rank"]), axis=1)
        render_table(eu_df, show_all=True)

    with tab_am:
        am_df = base_standings[base_standings["region"] == "Americas"].reset_index(drop=True)
        am_df["rank"] = am_df.index + 1
        if mode_active == "updated":
            _orig_am = original_standings[original_standings["region"] == "Americas"].sort_values("total_points", ascending=False).reset_index(drop=True)
            _orig_am["rank"] = _orig_am.index + 1
            _oam = _orig_am.set_index("team")["rank"].to_dict()
            am_df["rank_delta"] = am_df.apply(lambda r: int(_oam.get(r["team"], r["rank"])) - int(r["rank"]), axis=1)
        render_table(am_df, show_all=True)

    with tab_as:
        as_df = base_standings[base_standings["region"] == "Asia"].reset_index(drop=True)
        as_df["rank"] = as_df.index + 1
        if mode_active == "updated":
            _orig_as = original_standings[original_standings["region"] == "Asia"].sort_values("total_points", ascending=False).reset_index(drop=True)
            _orig_as["rank"] = _orig_as.index + 1
            _oas = _orig_as.set_index("team")["rank"].to_dict()
            as_df["rank_delta"] = as_df.apply(lambda r: int(_oas.get(r["team"], r["rank"])) - int(r["rank"]), axis=1)
        render_table(as_df, show_all=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 15 — Factor Score Breakdown")
        top15 = base_standings.head(15).copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Factor Score (400–2000)", x=top15["team"], y=top15["seed"],
            marker_color="#79c0ff",
        ))
        fig.add_trace(go.Bar(
            name="H2H Δ",
            x=top15["team"],
            y=top15["h2h_delta"],
            marker_color=[("#3fb950" if v >= 0 else "#f85149") for v in top15["h2h_delta"]],
        ))
        fig.update_layout(
            barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"), xaxis=dict(tickangle=-45, gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=80), height=400,
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    with col2:
        st.subheader("🕸️ Top 10 — Factor Radar")
        top10 = base_standings.head(10)
        pillar_cols = ["bo_factor", "bc_factor", "on_factor", "lan_factor"]
        pillar_labels = ["Bounty Offered", "Bounty Collected", "Opp. Network", "LAN Wins"]
        fig2 = go.Figure()
        for _, row in top10.iterrows():
            vals = [row[c] for c in pillar_cols] + [row[pillar_cols[0]]]
            fig2.add_trace(go.Scatterpolar(
                r=vals, theta=pillar_labels + [pillar_labels[0]],
                name=row["team"], mode="lines",
                line=dict(color=TEAM_META.get(row["team"], {}).get("color", "#58a6ff"), width=2),
            ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, gridcolor="#30363d", color="#8b949e", range=[0, 1]),
                angularaxis=dict(gridcolor="#30363d", color="#8b949e"),
                bgcolor="#161b22",
            ),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"),
            showlegend=True,
            legend=dict(orientation="v", x=1.05, y=0.5, font=dict(size=10)),
            margin=dict(l=10, r=120, t=10, b=10), height=400,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"staticPlot": True})


# ══════════════════════════════════════════════════════════════════
# PAGE 2  ·  SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════════════

elif page == "🔮 What-If Predictor":
    st.title("🔮 What-If Predictor")
    st.markdown(
        "Build hypothetical scenarios on top of the current baseline standings. "
        "Add match results and/or prize earnings to instantly see how standings would shift. "
        "**Note:** Baseline automatically reflects your current mode selection (As Published or Updated to Today)."
    )

    all_teams = sorted(base_standings["team"].tolist()) if not base_standings.empty else sorted(TEAM_META.keys())

    if "hyp_matches" not in st.session_state:
        st.session_state.hyp_matches = []
    if "hyp_prizes" not in st.session_state:
        st.session_state.hyp_prizes = []

    st.markdown("---")

    # ── Input forms: matches (left) and prizes (right) ────────────
    _form_left, _form_right = st.columns(2)

    with _form_left:
        with st.form("add_match"):
            st.subheader("➕ Add Hypothetical Match")
            c1, c2 = st.columns(2)
            winner = c1.selectbox("🏆 Winner", all_teams, index=all_teams.index("G2"))
            loser  = c2.selectbox("❌ Loser",
                                  [t for t in all_teams if t != winner],
                                  index=0)
            c3, c4 = st.columns(2)
            event_name = c3.text_input("🏟️ Event", value="Hypothetical Event")
            prize_pool = c4.number_input("💵 Prize Pool (USD)", 10_000, 2_000_000, 250_000, 10_000)
            c5, c6 = st.columns(2)
            is_lan     = c5.toggle("🖥️ LAN?", value=True)
            match_date = c6.date_input("📅 Match Date", value=datetime(2026, 3, 20))

            if st.form_submit_button("➕ Queue Match", use_container_width=True, type="primary"):
                st.session_state.hyp_matches.append({
                    "date":         datetime.combine(match_date, datetime.min.time()),
                    "winner":       winner,
                    "loser":        loser,
                    "event":        event_name,
                    "prize_pool":   float(prize_pool),
                    "winner_prize": 0.0,
                    "loser_prize":  0.0,
                    "is_lan":       is_lan,
                })
                st.success(f"✅ Queued: **{winner}** def. **{loser}**")

    with _form_right:
        with st.form("add_prize"):
            st.subheader("🏅 Add Hypothetical Prize Earning")
            p1, p2 = st.columns(2)
            prize_team   = p1.selectbox("🏆 Team", all_teams, index=0)
            prize_amount = p2.number_input("💵 Prize (USD)", 1_000, 2_000_000, 50_000, 1_000)
            p3, p4 = st.columns(2)
            prize_event  = p3.text_input("🏟️ Event", value="Hypothetical Event")
            prize_date   = p4.date_input("📅 Tournament End Date", value=datetime(2026, 3, 20))

            if st.form_submit_button("➕ Queue Prize", use_container_width=True, type="primary"):
                st.session_state.hyp_prizes.append({
                    "team":  prize_team,
                    "prize": float(prize_amount),
                    "event": prize_event,
                    "date":  datetime.combine(prize_date, datetime.min.time()),
                })
                st.success(f"✅ Queued: **{prize_team}** earns **${prize_amount:,.0f}**")

    # ── Queue display: matches (left) and prizes (right) ──────────
    _has_matches = bool(st.session_state.hyp_matches)
    _has_prizes  = bool(st.session_state.hyp_prizes)

    if _has_matches or _has_prizes:
        _ql, _qr = st.columns(2)

        with _ql:
            st.caption(f"Match queue — {len(st.session_state.hyp_matches)} series")
            if _has_matches:
                _qm = pd.DataFrame(st.session_state.hyp_matches)[
                    ["winner", "loser", "event", "is_lan", "date"]]
                _qm["date"] = pd.to_datetime(_qm["date"]).dt.strftime("%Y-%m-%d")
                _qm.columns = ["Winner", "Loser", "Event", "LAN", "Date"]
                st.dataframe(_qm, use_container_width=True, hide_index=True, height=200)

        with _qr:
            st.caption(f"Prize queue — {len(st.session_state.hyp_prizes)} entries")
            if _has_prizes:
                _qp = pd.DataFrame(st.session_state.hyp_prizes)[["team", "prize", "event", "date"]]
                _qp["date"]  = pd.to_datetime(_qp["date"]).dt.strftime("%Y-%m-%d")
                _qp["prize"] = _qp["prize"].apply(lambda x: f"${x:,.0f}")
                _qp.columns = ["Team", "Prize", "Event", "Date"]
                st.dataframe(_qp, use_container_width=True, hide_index=True, height=200)

        _btn_row_l, _btn_row_r = st.columns([3, 1])
        if _btn_row_r.button("🗑️ Clear All Queues", use_container_width=True):
            st.session_state.hyp_matches = []
            st.session_state.hyp_prizes  = []
            st.rerun()

        if _btn_row_l.button("🚀 Simulate & Compare", use_container_width=True, type="primary"):
            with st.spinner("Recalculating full VRS…"):
                hyp_df        = pd.DataFrame(st.session_state.hyp_matches) if _has_matches else pd.DataFrame()
                hyp_prizes    = st.session_state.hyp_prizes
                new_standings = compute_standings(extra_matches=hyp_df, extra_prizes=hyp_prizes)

            if new_standings.empty:
                st.warning("Not enough data to compute standings with the queued matches.")
                st.stop()

            st.markdown("---")
            st.subheader("📊 Impact: New Standings vs Baseline")

            old = base_standings[["team", "rank", "total_points", "seed", "h2h_delta"]].copy()
            new = new_standings[["team", "rank", "total_points", "seed",
                                  "h2h_delta", "region", "flag"]].copy()
            merged = old.merge(new, on="team", suffixes=("_old", "_new"))
            merged["rank_delta"]   = merged["rank_old"]  - merged["rank_new"]
            merged["points_delta"] = merged["total_points_new"] - merged["total_points_old"]
            merged = merged.sort_values("rank_new")

            affected = set(hyp_df["winner"].tolist() + hyp_df["loser"].tolist()) if not hyp_df.empty else set()

            rows = []
            for _, row in merged.iterrows():
                hl    = "background:#1c2128;" if row["team"] in affected else ""
                star  = "⭐ " if row["team"] in affected else ""
                d_pts = row["points_delta"]
                pts_s = (f'<span class="change-up">+{d_pts:.1f}</span>'   if d_pts > 0 else
                         f'<span class="change-down">{d_pts:.1f}</span>'   if d_pts < 0 else
                         '<span class="change-same">—</span>')
                rows.append(f"""
                <tr style="{hl}">
                  <td style="text-align:center">{rank_badge(int(row["rank_new"]))}</td>
                  <td>{row["flag"]} {star}<strong>{row["team"]}</strong></td>
                  <td style="text-align:right; color:#58a6ff; font-weight:700">
                      {row["total_points_new"]:,.1f}</td>
                  <td style="text-align:center">{change_arrow(int(row["rank_delta"]))}</td>
                  <td style="text-align:center; color:#8b949e">{int(row["rank_old"])}</td>
                  <td style="text-align:right">{pts_s}</td>
                </tr>""")

            st.markdown(f"""
            <table style="width:100%; border-collapse:collapse; font-size:13px;">
              <thead>
                <tr style="background:#21262d; color:#8b949e; font-size:11px; text-transform:uppercase;">
                  <th style="padding:10px 5px; text-align:center">New Rank</th>
                  <th style="padding:10px 5px; text-align:left">Team</th>
                  <th style="padding:10px 5px; text-align:right">New Points</th>
                  <th style="padding:10px 5px; text-align:center">Rank Δ</th>
                  <th style="padding:10px 5px; text-align:center">Old Rank</th>
                  <th style="padding:10px 5px; text-align:right">Points Δ</th>
                </tr>
              </thead>
              <tbody>{"".join(rows)}</tbody>
            </table>
            <p style="color:#8b949e;font-size:12px;margin-top:6px">⭐ = directly involved</p>
            """, unsafe_allow_html=True)

            # Points delta chart
            st.markdown("---")
            plot_df = merged[merged["points_delta"].abs() > 0.01].sort_values("points_delta")
            if not plot_df.empty:
                fig = go.Figure(go.Bar(
                    x=plot_df["points_delta"], y=plot_df["team"], orientation="h",
                    marker_color=["#3fb950" if v > 0 else "#f85149"
                                  for v in plot_df["points_delta"]],
                    text=[f"{v:+.1f}" for v in plot_df["points_delta"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Points Change by Team",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9d1d9"),
                    xaxis=dict(gridcolor="#21262d", title="Δ Points"),
                    yaxis=dict(gridcolor="#21262d"),
                    margin=dict(l=10, r=80, t=40, b=10),
                    height=max(300, len(plot_df) * 28),
                )
                st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
    else:
        st.info("👆 Add at least one match, then click **Simulate & Compare**.")


# ══════════════════════════════════════════════════════════════════
# PAGE 3  ·  TEAM COMPARISON
# ══════════════════════════════════════════════════════════════════

elif page == "⚔️ Team Comparison":
    st.title("⚔️ Team Comparison")

    all_ranked = base_standings["team"].tolist()
    default_a  = all_ranked.index("Vitality") if "Vitality" in all_ranked else 0
    default_b  = all_ranked.index("NAVI")     if "NAVI"     in all_ranked else 1

    ca, cb = st.columns(2)
    team_a = ca.selectbox("Team A", all_ranked, index=default_a)
    team_b = cb.selectbox("Team B", all_ranked, index=default_b)

    def team_row(name):
        r = base_standings[base_standings["team"] == name]
        return r.iloc[0].to_dict() if not r.empty else {}

    a, b = team_row(team_a), team_row(team_b)
    if not a or not b:
        st.warning("One or both teams not found in standings.")
        st.stop()

    st.markdown("---")

    # ── Score cards ───────────────────────────────────────────────
    def score_card(data, name, border_color, meta):
        return f"""
        <div style="text-align:center; padding:20px; background:#161b22;
                    border-radius:12px; border:2px solid {border_color};">
          <div style="font-size:36px">{meta.get('flag','🌍')}</div>
          <div style="font-size:22px; font-weight:700; color:#c9d1d9; margin:8px 0">{name}</div>
          <div style="font-size:12px; color:#8b949e">{meta.get('region','')}</div>
          <div style="font-size:34px; font-weight:700; color:{border_color}; margin-top:10px">
              {data['total_points']:,.1f}
          </div>
          <div style="color:#8b949e; font-size:12px">Total VRS Points</div>
          <div style="font-size:15px; color:#8b949e; margin-top:4px">
              Seed: <strong style="color:#79c0ff">{data['seed']:,.0f}</strong> &nbsp;+&nbsp;
              H2H: <strong style="color:{'#3fb950' if data['h2h_delta']>=0 else '#f85149'}">
              {data['h2h_delta']:+.0f}</strong>
          </div>
          <div style="font-size:18px; font-weight:600; color:#f0b429; margin-top:10px">
              #{int(data['rank'])} Global &nbsp;·&nbsp;
              #{int(data['regional_rank'])} {meta.get('region','')}
          </div>
        </div>"""

    cc1, cc2, cc3 = st.columns([5, 1, 5])
    with cc1:
        st.markdown(score_card(a, team_a, "#58a6ff", TEAM_META.get(team_a, {})),
                    unsafe_allow_html=True)
    with cc2:
        st.markdown("<div style='text-align:center;font-size:26px;font-weight:700;"
                    "color:#8b949e;padding-top:60px'>VS</div>", unsafe_allow_html=True)
    with cc3:
        st.markdown(score_card(b, team_b, "#f85149", TEAM_META.get(team_b, {})),
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Factor breakdown ──────────────────────────────────────────
    pillars = ["bo_factor", "bc_factor", "on_factor", "lan_factor"]
    labels  = ["Bounty Offered", "Bounty Collected", "Opp. Network", "LAN Wins"]
    a_vals  = [a[p] for p in pillars]
    b_vals  = [b[p] for p in pillars]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🕸️ Factor Radar")
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=a_vals + [a_vals[0]], theta=labels + [labels[0]],
            fill="toself", name=team_a,
            line_color="#58a6ff", fillcolor="rgba(88,166,255,0.12)",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=b_vals + [b_vals[0]], theta=labels + [labels[0]],
            fill="toself", name=team_b,
            line_color="#f85149", fillcolor="rgba(248,81,73,0.12)",
        ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d", color="#8b949e"),
                angularaxis=dict(gridcolor="#30363d", color="#8b949e"),
                bgcolor="#161b22",
            ),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=400, margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig_r, use_container_width=True, config={"staticPlot": True})

    with col2:
        st.subheader("📊 Raw Factor Scores")
        raw_labels = ["Bounty Offered\n(raw USD×stakes)",
                      "Bounty Collected\n(sum opp. BO)",
                      "Opp. Network\n(sum t_mod)",
                      "LAN Wins\n(count)"]
        raw_cols = ["bo_sum", "bc_pre_curve", "on_factor", "lan_wins"]
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(
            name=team_a, x=labels,
            y=[a[c] for c in raw_cols],
            marker_color="#58a6ff",
            text=[f"{a[c]:.2f}" for c in raw_cols], textposition="outside",
        ))
        fig_b.add_trace(go.Bar(
            name=team_b, x=labels,
            y=[b[c] for c in raw_cols],
            marker_color="#f85149",
            text=[f"{b[c]:.2f}" for c in raw_cols], textposition="outside",
        ))
        fig_b.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_b, use_container_width=True, config={"staticPlot": True})

    # ── H2H record ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤝 Head-to-Head Record (in dataset)")
    h2h_df = base_matches[
        ((base_matches["winner"] == team_a) & (base_matches["loser"] == team_b)) |
        ((base_matches["winner"] == team_b) & (base_matches["loser"] == team_a))
    ].sort_values("date", ascending=False)

    if h2h_df.empty:
        st.info(f"No direct matches between **{team_a}** and **{team_b}** in the dataset.")
    else:
        a_wins = int((h2h_df["winner"] == team_a).sum())
        b_wins = int((h2h_df["winner"] == team_b).sum())
        h1, h2, h3 = st.columns(3)
        h1.metric(f"{team_a} Wins", a_wins)
        h2.metric("Matches", len(h2h_df))
        h3.metric(f"{team_b} Wins", b_wins)

        disp = h2h_df[["date", "winner", "loser", "event", "prize_pool", "is_lan"]].copy()
        disp["date"]       = disp["date"].dt.strftime("%Y-%m-%d")
        disp["prize_pool"] = disp["prize_pool"].apply(lambda x: f"${x:,.0f}")
        disp.columns = ["Date", "Winner", "Loser", "Event", "Prize Pool", "LAN"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Stat comparison ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Stats Comparison")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(f"{team_a} Wins",    int(a["wins"]))
    s2.metric(f"{team_a} Losses",  int(a["losses"]))
    s3.metric(f"{team_b} Wins",    int(b["wins"]))
    s4.metric(f"{team_b} Losses",  int(b["losses"]))



# ══════════════════════════════════════════════════════════════════
# PAGE 4  ·  HOW VRS WORKS  (Explainer — v3, formula-verified)
# ══════════════════════════════════════════════════════════════════

elif page == "📖 How VRS Works":

    st.markdown("""
    <style>
    .vrs-box {
        background:#0d1117; border:1px solid #58a6ff;
        border-left:4px solid #58a6ff; border-radius:6px;
        padding:14px 18px; margin:10px 0; font-size:13px; color:#c9d1d9;
    }
    .verified-box {
        background:#0d1a0d; border:1px solid #3fb950;
        border-left:4px solid #3fb950; border-radius:6px;
        padding:12px 16px; margin:8px 0; font-size:13px; color:#c9d1d9;
    }
    .assumption-box {
        background:#1c1a00; border:1px solid #f0b429;
        border-left:4px solid #f0b429; border-radius:6px;
        padding:12px 16px; margin:8px 0; font-size:13px; color:#c9d1d9;
    }
    .formula-mono {
        font-family:monospace; background:#0d1117; color:#79c0ff;
        padding:10px 14px; border-radius:5px; border-left:3px solid #30363d;
        font-size:13px; white-space:pre; display:block; margin:8px 0;
    }
    .tag-v { background:#1f4a1f; color:#3fb950; border-radius:3px;
              padding:2px 7px; font-size:11px; font-weight:700; }
    .tag-a { background:#3d2e00; color:#f0b429; border-radius:3px;
              padding:2px 7px; font-size:11px; font-weight:700; }
    .factor-pill {
        display:inline-block; border-radius:20px; padding:4px 14px;
        font-weight:700; font-size:13px; margin:4px 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📖 How the VRS Works")

    # ── Tab categories ────────────────────────────────────────────
    # Welcome | Architecture | Background (Age, Event, Curve) | Factors (BO, BC, ON, LAN → Seed) | Final (H2H)
    (tab_welcome, tab_arch,
     tab_age, tab_evw, tab_curve,
     tab_bo2, tab_bc2, tab_on2, tab_lan2, tab_seed,
     tab_h2h) = st.tabs([
        "👋 Welcome",
        "🏗️ Architecture",
        #  — Background —
        "⏳ Age Weight",
        "🎪 Event Weight",
        "📐 Curve f(x)",
        #  — Four Factors —
        "🏆 Bounty Offered",
        "💰 Bounty Collected",
        "🕸️ Opp. Network",
        "🖥️ LAN Wins",
        "🌱 → Factor Score",
        #  — Final —
        "⚔️ H2H → Final",
    ])

    # ══════════════════════════════════════════════════════════════
    with tab_welcome:
    # ══════════════════════════════════════════════════════════════

        st.markdown("""
### What is the VRS?

The **Valve Regional Standings** (VRS) is Valve's official ranking system for professional
Counter-Strike 2 teams. Introduced in October 2023, it replaced the previous RMR
(Regional Major Rankings) system and is now the **sole factor** for determining invitations
to Major Championships and seedings at Valve-sanctioned events.

> *"Teams play meaningful matches in third-party events throughout the year.
> To reduce the burden on Major participants and streamline the Major qualification process,
> we're going to leverage those match results to identify teams that should be invited
> to later qualification stages."*
> — **Valve** (2023 announcement)

### Why does it matter?

Unlike community-driven rankings (e.g. HLTV), the VRS directly determines which teams
get to play at Majors — the most prestigious tournaments in CS2. Tournament organizers
are also increasingly using VRS for their own invitations, making it the backbone of the
entire competitive CS2 ecosystem.

### What does this tool offer?

This site fetches live data from [Valve's GitHub repository](https://github.com/ValveSoftware/counter-strike_regional_standings)
and provides three things:
""")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-top:3px solid #58a6ff;
            border-radius:8px;padding:16px;height:200px;">
  <div style="font-size:22px;margin-bottom:8px">📊</div>
  <div style="font-size:15px;font-weight:700;color:#58a6ff;margin-bottom:6px">Live Rankings</div>
  <div style="font-size:12px;color:#8b949e;line-height:1.6">
    Current and historical VRS standings with full factor breakdowns per team.
    Explore global, European, Americas, and Asia rankings.
  </div>
</div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-top:3px solid #3fb950;
            border-radius:8px;padding:16px;height:200px;">
  <div style="font-size:22px;margin-bottom:8px">🔍</div>
  <div style="font-size:15px;font-weight:700;color:#3fb950;margin-bottom:6px">Deep Team Analysis</div>
  <div style="font-size:12px;color:#8b949e;line-height:1.6">
    Detailed breakdowns per team: how each of the four factors and every single match
    contributed to their final score.
  </div>
</div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-top:3px solid #7c3aed;
            border-radius:8px;padding:16px;height:200px;">
  <div style="font-size:22px;margin-bottom:8px">🔮</div>
  <div style="font-size:15px;font-weight:700;color:#7c3aed;margin-bottom:6px">Simulation</div>
  <div style="font-size:12px;color:#8b949e;line-height:1.6">
    Simulate time decay: see how rankings will shift next month as matches age out
    of the 180-day window — before Valve publishes the update.
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
### How is the VRS calculated?

The remaining tabs on this page walk through the complete calculation, verified formula-by-formula
against Valve's published data. Here's the short version:
""")
        st.latex(
            r"\text{Final Score} = "
            r"\underbrace{\text{Factor Score}_{[400,\;2000]}}_{\text{Phase 1}} "
            r"+ \;\underbrace{\Delta_{\text{H2H}}}_{\text{Phase 2}}"
        )
        st.markdown("""
**Phase 1** averages four factors (Bounty Offered, Bounty Collected, Opponent Network, LAN Wins)
and maps the result to a score between 400 and 2000.

**Phase 2** replays every match chronologically using a Glicko/Elo system, adjusting the seed up or down.

Only the last **180 days** of matches count, with recent matches weighted more heavily.

Use the tabs above to explore each component in detail. →
""")
        st.markdown(
            '<span class="tag-v">✓ VERIFIED</span>&nbsp; Formula confirmed from official data &nbsp;&nbsp;'
            '<span class="tag-a">~ ASSUMED</span>&nbsp; Not explicitly stated in public spec',
            unsafe_allow_html=True,
        )


    # ══════════════════════════════════════════════════════════════
    with tab_arch:
    # ══════════════════════════════════════════════════════════════
        st.subheader("Two-Phase Architecture")

        st.markdown("""
The VRS is computed in **two sequential phases** that are simply added together.
The flowchart below shows how match data flows through the system.
Click any factor to jump to its detailed tab.
""")

        # ── Interactive HTML flowchart ─────────────────────────────
        _fc = (
            '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:24px;margin:12px 0;">'
            '<div style="text-align:center;margin-bottom:20px;">'
            '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:10px 28px;font-size:14px;font-weight:700;color:#8b949e;">📅 180-Day Match History</div>'
            '<div style="color:#30363d;font-size:20px;margin:6px 0;">▼</div>'
            '</div>'
            '<div style="display:flex;gap:20px;margin-bottom:20px;">'
            # Phase 1
            '<div style="flex:1;background:#0d1a2e;border:2px solid #58a6ff;border-radius:10px;padding:16px;">'
            '<div style="font-size:14px;font-weight:700;color:#58a6ff;text-align:center;margin-bottom:14px;">Phase 1 — Factor Score</div>'
            '<div style="display:flex;gap:6px;margin-bottom:12px;justify-content:center;">'
            '<span style="background:#21262d;border:1px solid #484f58;border-radius:6px;padding:4px 10px;font-size:11px;color:#8b949e;">⏳ Age Weight</span>'
            '<span style="background:#21262d;border:1px solid #484f58;border-radius:6px;padding:4px 10px;font-size:11px;color:#8b949e;">🎪 Event Weight</span>'
            '<span style="background:#21262d;border:1px solid #484f58;border-radius:6px;padding:4px 10px;font-size:11px;color:#8b949e;">📐 Curve</span>'
            '</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px;">'
            '<div style="background:#161b22;border-left:3px solid #f0b429;border-radius:6px;padding:10px;text-align:center;"><div style="font-size:12px;font-weight:700;color:#f0b429;">🏆 Bounty Offered</div><div style="font-size:10px;color:#8b949e;margin-top:2px;">Prize money earned</div></div>'
            '<div style="background:#161b22;border-left:3px solid #3fb950;border-radius:6px;padding:10px;text-align:center;"><div style="font-size:12px;font-weight:700;color:#3fb950;">💰 Bounty Collected</div><div style="font-size:10px;color:#8b949e;margin-top:2px;">Quality of opponents beaten</div></div>'
            '<div style="background:#161b22;border-left:3px solid #79c0ff;border-radius:6px;padding:10px;text-align:center;"><div style="font-size:12px;font-weight:700;color:#79c0ff;">🕸️ Opp. Network</div><div style="font-size:10px;color:#8b949e;margin-top:2px;">Network depth (PageRank)</div></div>'
            '<div style="background:#161b22;border-left:3px solid #f85149;border-radius:6px;padding:10px;text-align:center;"><div style="font-size:12px;font-weight:700;color:#f85149;">🖥️ LAN Wins</div><div style="font-size:10px;color:#8b949e;margin-top:2px;">Offline wins count</div></div>'
            '</div>'
            '<div style="text-align:center;color:#30363d;font-size:16px;margin-bottom:8px;">▼ avg (25% each) ▼</div>'
            '<div style="background:#161b22;border:2px solid #58a6ff;border-radius:8px;padding:10px;text-align:center;"><div style="font-size:13px;font-weight:700;color:#58a6ff;">🌱 Factor Score</div><div style="font-size:10px;color:#8b949e;">lerp → [400, 2000]</div></div>'
            '</div>'
            # Phase 2
            '<div style="flex:1;background:#0d1a0d;border:2px solid #3fb950;border-radius:10px;padding:16px;display:flex;flex-direction:column;justify-content:space-between;">'
            '<div>'
            '<div style="font-size:14px;font-weight:700;color:#3fb950;text-align:center;margin-bottom:14px;">Phase 2 — Head-to-Head</div>'
            '<div style="font-size:12px;color:#c9d1d9;line-height:1.6;text-align:center;padding:0 10px;">'
            'Starting from the Factor Score, every match is replayed <b>chronologically</b> using a Glicko/Elo system.<br><br>'
            'Upsets (beating higher-rated teams) gain more points; expected wins gain fewer.<br><br>'
            "Each match's K-factor is scaled by <b>Age Weight only</b> — recent matches have more impact."
            '</div></div>'
            '<div style="background:#161b22;border:2px solid #3fb950;border-radius:8px;padding:10px;text-align:center;margin-top:14px;"><div style="font-size:13px;font-weight:700;color:#3fb950;">⚔️ H2H Adjustment (Δ)</div><div style="font-size:10px;color:#8b949e;">Can be positive or negative</div></div>'
            '</div>'
            '</div>'
            # Final
            '<div style="text-align:center;">'
            '<div style="color:#30363d;font-size:20px;margin-bottom:6px;">▼ + ▼</div>'
            '<div style="display:inline-block;background:#2d1f00;border:2px solid #f0b429;border-radius:10px;padding:12px 36px;">'
            '<div style="font-size:16px;font-weight:700;color:#f0b429;">🎯 Final Score</div>'
            '<div style="font-size:11px;color:#c9d1d9;margin-top:2px;">Factor Score + H2H Δ</div>'
            '</div></div>'
            '</div>'
        )
        st.markdown(_fc, unsafe_allow_html=True)

        st.markdown('<span class="tag-v">✓ VERIFIED</span> — from official team detail sheets', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
**Data window:** Only matches within the last **180 days** contribute.
Within the window, **Age Weight** scales every contribution continuously —
the most recent 30 days count at full strength, then decay linearly to zero at 180 days.

**Eligibility:** A team must have ≥1 win and ≥5 total matches in the window to be ranked.

**Explore each component** using the tabs above. The tabs are ordered left-to-right
following the calculation flow:

| Category | Tabs | Purpose |
|---|---|---|
| **Background** | Age Weight, Event Weight, Curve | Modifiers applied within the factors |
| **Four Factors** | BO, BC, ON, LAN → Factor Score | Phase 1 computation |
| **Final Score** | H2H → Final | Phase 2 Glicko adjustment |
""")

    # ══════════════════════════════════════════════════════════════
    with tab_evw:
    # ══════════════════════════════════════════════════════════════
        st.subheader("🎪 Event Weight (Event Stakes)")
        col_r, col_l = st.columns([2, 3])
        with col_l:
            st.markdown(r"""
Event Weight measures the **prestige of a tournament** based on its total prize pool.
It is used as a multiplier in the **Bounty Collected** and **Opponent Network** calculations
(but NOT in Bounty Offered or LAN Wins).

$$\text{Event Weight} = f\!\left(\frac{\text{prize\_pool}}{\$1{,}000{,}000}\right)$$

where $f(x) = \frac{1}{1 + |\log_{10}(x)|}$ is the Curve function.

**Key idea:** A $1M tournament gets weight 1.000. Smaller events get proportionally less weight,
but the log-scale means a $100k event still gets 0.500 — not 0.100. This prevents small
events from being completely worthless while ensuring large tournaments carry the most importance.

**Where Event Weight is used:**
- ✅ **Bounty Collected:** `entry = opp_BO × age × event_weight`
- ✅ **Opponent Network:** `entry = opp_ON × age × event_weight`
- ❌ **Bounty Offered:** uses age weight only (no event weight)
- ❌ **LAN Wins:** uses age weight only (no event weight)
- ❌ **H2H (Glicko):** K-factor uses age weight only

**Matches without a prize pool** (e.g., online qualifiers) have Event Weight = 0
and therefore do **not contribute** to BC or ON at all.
""")
            st.markdown('<span class="tag-v">✓ VERIFIED — derived from curve function applied to prize pool</span>', unsafe_allow_html=True)
        with col_r:
            st.markdown(
                '<div style="background:#1a0f2e;border:1px solid #7c3aed;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px;font-weight:600;color:#c4b5fd;text-align:center;">✏️ <strong>Try it yourself!</strong>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown("#### 🎛️ Event Weight Calculator")
            _ew_pool = st.slider("Prize pool (USD)", 0, 1_200_000, 250_000, 10_000, key="ew_pool")
            _ew_val = event_stakes(max(_ew_pool, 1))
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                        padding:16px;text-align:center;margin-bottom:12px;">
              <div style="font-size:12px;color:#8b949e">${_ew_pool:,.0f} / $1,000,000</div>
              <div style="font-size:48px;font-weight:700;color:#3fb950;margin:4px 0">{_ew_val:.4f}</div>
              <div style="font-size:12px;color:#8b949e">Event Weight</div>
              <div style="font-size:11px;color:#484f58;margin-top:4px">
                  {'✅ Full weight (≥ $1M)' if _ew_pool >= 1_000_000 else
                   f'→ {_ew_val*100:.0f}% of a $1M tournament' if _ew_pool > 0 else
                   '→ 14.3% (minimum floor)'}</div>
            </div>""", unsafe_allow_html=True)

            # Event Weight curve chart
            _ew_xs = list(range(0, 1_200_001, 10_000))
            _ew_ys = [event_stakes(max(x, 1)) for x in _ew_xs]
            fig_ew = go.Figure()
            fig_ew.add_trace(go.Scatter(
                x=[x/1e6 for x in _ew_xs], y=_ew_ys, mode="lines",
                line=dict(color="#58a6ff", width=2.5),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"))
            fig_ew.add_trace(go.Scatter(
                x=[_ew_pool/1e6], y=[_ew_val], mode="markers",
                marker=dict(color="#f0b429", size=12, symbol="diamond",
                            line=dict(color="#c9d1d9", width=1.5)),
                showlegend=False))
            fig_ew.add_vrect(x0=1.0, x1=1.2, fillcolor="#3fb950", opacity=0.06,
                line_width=0, annotation_text="Capped at $1M",
                annotation_position="top right",
                annotation=dict(font_color="#3fb950", font_size=10))
            fig_ew.add_vline(x=1.0, line_dash="dot", line_color="#3fb950",
                annotation_text="$1M cap", annotation_position="top left",
                annotation=dict(font_color="#3fb950", font_size=10))
            for pp, lbl, ax_off, ay_off in [
                (1e6,  "$1M → 1.000",  -55, -25),
                (5e5,  "$500k → 0.769", 50, -30),
                (1e5,  "$100k → 0.500", 50, -20),
                (2.5e4,"$25k → 0.286",  50,  20),
            ]:
                ev = event_stakes(max(pp, 1))
                fig_ew.add_annotation(x=pp/1e6, y=ev, text=lbl,
                    showarrow=True, arrowhead=2, arrowwidth=1,
                    arrowcolor="#484f58", ax=ax_off, ay=ay_off,
                    font=dict(size=9, color="#8b949e"),
                    bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d", borderwidth=1)
            fig_ew.update_layout(
                xaxis=dict(title="Prize Pool ($M)", gridcolor="#21262d",
                           tickvals=[0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.2],
                           ticktext=["$0","$100k","$250k","$500k","$750k","$1M","$1.2M"]),
                yaxis=dict(title="Event Weight", gridcolor="#21262d",
                           tickformat=".0%", range=[-0.05, 1.15]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), showlegend=False,
                margin=dict(l=10,r=10,t=20,b=10), height=300)
            st.plotly_chart(fig_ew, use_container_width=True, config={"staticPlot": True})
            st.caption(
                "Source: [Reddit — How Valve's CSGO Team Ranking System Works]"
                "(https://www.reddit.com/r/GlobalOffensive/comments/15j0t5e/) · "
                '"$1M tournaments are scaled 100%, $100k tournaments 50%, $0 tournaments 14.3%"'
            )

    # ══════════════════════════════════════════════════════════════
    with tab_age:
        st.subheader("⏳ Age Weight (Time Decay)")
        col_r, col_l = st.columns([2, 3])
        with col_l:
            st.markdown(r"""
Every match in the 180-day window is assigned an **age weight** based on how many days
before the cutoff it was played.

$$\text{age\_weight} = \begin{cases} 1.0 & \text{if days\_ago} \leq 30 \\ 1 - \dfrac{\text{days\_ago} - 30}{150} & \text{if } 30 < \text{days\_ago} \leq 180 \\ 0 \text{ (excluded)} & \text{if days\_ago} > 180 \end{cases}$$

The structure is **two regions**:
- **Last 30 days:** flat weight of 1.000 — everything very recent counts fully
- **Days 31–180:** linear decay from 1.000 → 0.000 over 150 days

This is NOT a simple 0→1 remap over 180 days. The flat period is critical.
""")
            st.markdown("**Verification against official data (cutoff 2026-03-02):**")
            vdata = [
                ("2026-01-31", 30, 1.000),
                ("2026-01-24", 37, 0.953),
                ("2025-12-14", 78, 0.681),
                ("2025-11-09",113, 0.445),
                ("2025-10-12",141, 0.260),
                ("2025-09-07",176, 0.027),
            ]
            vrows = []
            for date_s, days, official in vdata:
                calc = age_weight(datetime.strptime(date_s, "%Y-%m-%d"), datetime(2026, 3, 2))
                match = "✅" if abs(calc - official) < 0.002 else "❌"
                vrows.append({"Date": date_s, "Days ago": days,
                               "Official": official, "Calculated": round(calc,3), "": match})
            st.dataframe(pd.DataFrame(vrows), use_container_width=True, hide_index=True)
            st.markdown('<span class="tag-v">✓ VERIFIED — exact match to Vitality detail sheet</span>', unsafe_allow_html=True)

        with col_r:
            st.markdown(
                '<div style="background:#1a0f2e;border:1px solid #7c3aed;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px;font-weight:600;color:#c4b5fd;text-align:center;">✏️ <strong>Try it yourself!</strong>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown("#### 🎛️ Age Weight Calculator")
            days_ago_val = st.slider("Days before cutoff", 0, 200, 60, 5, key="aw_slider")
            dummy_cutoff = datetime(2026, 3, 2)
            dummy_date   = dummy_cutoff - timedelta(days=days_ago_val)
            aw = age_weight(dummy_date, dummy_cutoff)
            color = "#3fb950" if aw > 0.6 else ("#f0b429" if aw > 0.2 else "#f85149")
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                        padding:20px;text-align:center;margin-bottom:12px;">
              <div style="font-size:13px;color:#8b949e">Age weight for match played</div>
              <div style="font-size:22px;font-weight:700;color:#c9d1d9;margin:4px 0">
                  {days_ago_val} days ago</div>
              <div style="font-size:48px;font-weight:700;color:{color}">{aw:.4f}</div>
              <div style="font-size:12px;color:#8b949e;margin-top:6px">
                  {'✅ In window' if days_ago_val <= 180 else '❌ Outside window'}</div>
            </div>""", unsafe_allow_html=True)

            # Curve chart
            xs = list(range(0, 201, 2))
            ys = [age_weight(dummy_cutoff - timedelta(days=d), dummy_cutoff) for d in xs]
            fig_aw = go.Figure()
            fig_aw.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                line=dict(color="#58a6ff", width=2.5),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"))
            fig_aw.add_trace(go.Scatter(x=[days_ago_val], y=[aw], mode="markers",
                marker=dict(color="#f0b429", size=11), showlegend=False))
            fig_aw.add_vrect(x0=180, x1=200, fillcolor="#f85149", opacity=0.06,
                line_width=0, annotation_text="Excluded", annotation_position="top right",
                annotation=dict(font_color="#f85149", font_size=10))
            fig_aw.add_vline(x=30, line_dash="dot", line_color="#f0b429",
                annotation_text="30-day flat", annotation_position="top right",
                annotation=dict(font_color="#f0b429", font_size=10))
            fig_aw.update_layout(
                xaxis=dict(title="Days before cutoff", gridcolor="#21262d",
                           tickvals=[0,30,60,90,120,150,180]),
                yaxis=dict(title="Age Weight", gridcolor="#21262d",
                           tickformat=".0%", range=[-0.05, 1.1]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10), height=280)
            st.plotly_chart(fig_aw, use_container_width=True, config={"staticPlot": True})

    # ══════════════════════════════════════════════════════════════
    with tab_curve:
        st.subheader("📐 The Curve Function  f(x)")
        col_r, col_l = st.columns([2, 3])
        with col_l:
            st.markdown(r"""
The curve function compresses any positive value into $(0,\,1]$ using a log scale:

$$f(x) = \frac{1}{1 + |\log_{10}(x)|}$$

**Where is it applied?** The curve is used as a **normalisation step** on two factors:

| Factor | Input x | Why |
|---|---|---|
| **Bounty Offered** | `min(1.0, bo_sum / 5th_ref)` | Compress prize money ratio into [0, 1] |
| **Bounty Collected** | `Σ_top10_adjusted / 10` | Compress opponent-strength sum into [0, 1] |

It is **not** applied to Opponent Network or LAN Wins.

**What does it do for BC?** Without the curve, a team that beats 10 opponents with
avg BO=0.05 would get BC_pre = 0.05. The curve maps this to `f(0.05) = 0.435` — a
significant boost. This prevents small teams from having near-zero BC scores when
they beat opponents with modest BO values.

**What does it do for BO?** Teams with less prize money than the top 5 get a ratio < 1.
The curve compresses this: a team at 10% of the ref gets `f(0.1) = 0.500` instead of 0.1.
This keeps smaller teams competitive in the ranking.

**Key properties:**
- `f(1.0) = 1.000` — at the reference point, full value
- `f(0.5) ≈ 0.769` — half of the reference
- `f(0.1) = 0.500` — one-tenth still gets 50%
- `f(0.01) = 0.333` — one-hundredth still gets 33%
- Peaks at x = 1; symmetric in log-space

**In short:** The curve prevents the rich-get-richer effect by giving meaningful
scores to teams that are orders of magnitude below the top.
""")
            st.markdown('<span class="tag-v">✓ VERIFIED — explicitly stated in official data footnote</span>', unsafe_allow_html=True)
        with col_r:
            st.markdown(
                '<div style="background:#1a0f2e;border:1px solid #7c3aed;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px;font-weight:600;color:#c4b5fd;text-align:center;">✏️ <strong>Try it yourself!</strong>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown("#### 🎛️ Curve Calculator")
            st.caption("Input: a factor value between 0 and 1 (typical BC/BO pre-curve range)")
            _cv_input = st.slider("Pre-curve value (x)", 0.01, 1.00, 0.25, 0.01, key="cv_x")
            _cv_output = curve(_cv_input)
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                        padding:16px;text-align:center;margin-bottom:12px;">
              <div style="font-size:13px;color:#8b949e">f({_cv_input:.2f}) = 1 / (1 + |log₁₀({_cv_input:.2f})|)</div>
              <div style="font-size:48px;font-weight:700;color:#3fb950;margin:4px 0">{_cv_output:.4f}</div>
              <div style="font-size:12px;color:#8b949e">After curve normalization</div>
            </div>""", unsafe_allow_html=True)

            # Curve chart with x from 0 to 1
            xs_c = [i/100 for i in range(1, 101)]
            ys_c = [curve(x) for x in xs_c]
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=xs_c, y=ys_c,
                mode="lines", line=dict(color="#bc8cff", width=2.5),
                fill="tozeroy", fillcolor="rgba(188,140,255,0.07)"))
            fig_c.add_trace(go.Scatter(x=[_cv_input], y=[_cv_output], mode="markers",
                marker=dict(color="#f0b429", size=12, symbol="diamond",
                            line=dict(color="#c9d1d9", width=1.5)),
                showlegend=False))
            for xr, lbl, ax_off, ay_off in [
                (1.00, "f(1.0) = 1.000", -65, -25),
                (0.50, "f(0.5) = 0.769", -65, -25),
                (0.10, "f(0.1) = 0.500",  50, -25),
                (0.01, "f(0.01) = 0.333", 50,  20),
            ]:
                fig_c.add_annotation(x=xr, y=curve(xr), text=lbl,
                    showarrow=True, arrowhead=2, arrowwidth=1,
                    arrowcolor="#484f58", ax=ax_off, ay=ay_off,
                    font=dict(size=9, color="#8b949e"),
                    bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d", borderwidth=1)
            fig_c.update_layout(
                xaxis=dict(title="Pre-curve value (x)", gridcolor="#21262d",
                           tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                yaxis=dict(title="f(x) — after curve", gridcolor="#21262d",
                           tickformat=".0%", range=[-0.05, 1.15]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10), height=280)
            st.plotly_chart(fig_c, use_container_width=True, config={"staticPlot": True})
            st.caption(
                "The curve is also used for Event Weight (see Event Weight tab), "
                "but with prize_pool/$1M as input instead of factor values."
            )

    # ══════════════════════════════════════════════════════════════
    with tab_bo2:
        st.subheader("🏆 Factor 1 — Bounty Offered")

        if "bo_sim_entries" not in st.session_state:
            st.session_state.bo_sim_entries = [
                {"event": "BLAST Fall 2025",    "prize": 500_000, "days_ago":  77},
                {"event": "ESL Pro League S21", "prize":  90_000, "days_ago":  49},
            ]
        if "bo_sim_result" not in st.session_state:
            st.session_state.bo_sim_result = None

        col_r, col_l = st.columns([2, 3])
        with col_l:
            _bo_fc = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:8px 0;">'
                '<div style="text-align:center;margin-bottom:16px;">'
                '<span style="font-size:13px;font-weight:700;color:#f0b429;">🏆 Bounty Offered — How is it calculated?</span>'
                '</div>'
                # Step 0: window
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:8px 22px;">'
                '<div style="font-size:13px;font-weight:700;color:#c9d1d9;">Look back at the last 180 days of play</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:2px;">📅 All tournament wins in this window are considered</div>'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                # Step 1
                '<div style="background:#0d1a2e;border:1px solid #58a6ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:6px;">Step 1 — Score each win</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">Each win is worth the prize money you collected, discounted for age — recent wins count full, older wins count less.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#0a0e1a;border-radius:4px;padding:5px 8px;">'
                'contribution = prize_won × age_weight &nbsp;(no event-stakes — only time matters here)'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                # Step 2
                '<div style="background:#0d1a2e;border:1px solid #58a6ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:6px;">Step 2 — Keep only your best 10 wins</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">Only your top 10 highest-scoring wins are summed. Wins beyond #10 are ignored — quality over quantity.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#0a0e1a;border-radius:4px;padding:5px 8px;">'
                'BO_sum = Σ top-10 contributions'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                # Step 3
                '<div style="background:#2d1f00;border:1px solid #f0b429;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f0b429;margin-bottom:6px;">Step 3 — Compare to the field</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">Your prize total is measured against the 5th-richest team in the field. If you match them, you score maximum. The top 5 earners all tie at 1.0.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#1a1000;border-radius:4px;padding:5px 8px;">'
                'ref₅ = 5th-highest BO_sum in the field &nbsp;·&nbsp; ratio = min(1.0, BO_sum / ref₅)'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                # Step 4
                '<div style="background:#2d1f00;border:1px solid #f0b429;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f0b429;margin-bottom:6px;">Step 4 — Smooth with the curve</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">A log-based curve is applied so smaller teams aren\'t crushed. A team at 10% of the benchmark still scores 0.500 — not 0.1.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#1a1000;border-radius:4px;padding:5px 8px;">'
                'BO = f(ratio) = 1 / (1 + |log₁₀(ratio)|) &nbsp;·&nbsp; f(1.0)=1.000, f(0.5)≈0.770, f(0.1)=0.500'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#2d1f00;border:2px solid #f0b429;border-radius:8px;padding:10px 28px;">'
                '<div style="font-size:14px;font-weight:700;color:#f0b429;">🏆 BO Factor &nbsp; [0 → 1]</div>'
                '</div></div>'
                '</div>'
            )
            st.markdown(_bo_fc, unsafe_allow_html=True)
            st.markdown('<span class="tag-v">✓ VERIFIED — sum / 5th-ref / curve(min(1,x)) confirmed</span>', unsafe_allow_html=True)

        with col_r:
            st.markdown(
                '<div style="background:#1a0f2e;border:1px solid #7c3aed;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px;font-weight:600;color:#c4b5fd;text-align:center;">✏️ <strong>Try it yourself!</strong>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown("#### Build a prize history")
            st.caption("Add fictive wins, set a ref₅ value, then click **Calculate** for a step-by-step breakdown.")

            with st.form("bo_add_form", clear_on_submit=True):
                _c1, _c2, _c3 = st.columns([3, 2, 1])
                _ne = _c1.text_input("Event name", placeholder="IEM Major 2026")
                _np = _c2.number_input("Prize ($)", min_value=0, max_value=2_000_000,
                                       value=250_000, step=10_000)
                _nd = _c3.number_input("Days ago", min_value=0, max_value=179, value=20)
                if st.form_submit_button("➕ Add win", use_container_width=True) and _ne:
                    st.session_state.bo_sim_entries.append(
                        {"event": _ne, "prize": int(_np), "days_ago": int(_nd)})
                    st.session_state.bo_sim_result = None

            if st.session_state.bo_sim_entries:
                for _i, _e in enumerate(st.session_state.bo_sim_entries):
                    _ec1, _ec2, _ec3 = st.columns([4, 3, 0.7])
                    _ec1.markdown(f"<span style='font-size:12px'>{_e['event']}</span>",
                                  unsafe_allow_html=True)
                    _ec2.markdown(
                        f"<span style='font-size:12px;color:#8b949e'>${_e['prize']:,} · {_e['days_ago']}d</span>",
                        unsafe_allow_html=True)
                    if _ec3.button("✕", key=f"bo_rm_{_i}"):
                        st.session_state.bo_sim_entries.pop(_i)
                        st.session_state.bo_sim_result = None
                        st.rerun()

                st.markdown("---")
                _ref5 = st.number_input(
                    "🏆 Benchmark — top-5 prize earner in the field ($)",
                    min_value=1_000, max_value=5_000_000, value=334_320, step=10_000,
                    help="This is ref₅: the 5th-highest BO_sum across all ranked teams. "
                         "Your score is compared against this value — reach it and you "
                         "hit the top-5 level (ratio = 1.0). Real Valve value ≈ $334,320 at cutoff 2026-03-02.")

                if st.button("🧮 Calculate Bounty Offered", use_container_width=True,
                             type="primary"):
                    _now = datetime.now()
                    _calcs = []
                    for _e in st.session_state.bo_sim_entries:
                        _md = _now - timedelta(days=_e["days_ago"])
                        _aw = age_weight(_md, _now)
                        _calcs.append({"event": _e["event"], "prize": _e["prize"],
                                       "aw": _aw, "contrib": _e["prize"] * _aw})
                    _calcs.sort(key=lambda x: x["contrib"], reverse=True)
                    _top10 = _calcs[:10]
                    _bo_sum = sum(x["contrib"] for x in _top10)
                    _ratio  = min(1.0, _bo_sum / _ref5)
                    _bo_val = curve(_ratio)
                    st.session_state.bo_sim_result = {
                        "calcs": _calcs, "top10": _top10,
                        "bo_sum": _bo_sum, "ref5": _ref5,
                        "ratio": _ratio, "bo_val": _bo_val,
                    }

            if st.session_state.bo_sim_result:
                _r = st.session_state.bo_sim_result
                st.markdown("##### 📋 Step-by-step result")
                _rows = ""
                for _idx, _x in enumerate(_r["calcs"]):
                    _top = _idx < 10
                    _bg  = "background:#0d2818" if _top else "background:#161b22;opacity:0.55"
                    _badge = ('<span style="background:#3fb950;color:#000;border-radius:3px;'
                              'padding:0 4px;font-size:9px;font-weight:700;margin-left:4px;">TOP10</span>'
                              if _top else '')
                    _rows += (
                        f'<tr style="{_bg}">'
                        f'<td style="padding:5px 6px;font-size:11px;color:#c9d1d9">{_x["event"]}{_badge}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#c9d1d9;text-align:right">${_x["prize"]:,}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#79c0ff;text-align:right">{_x["aw"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#f0b429;font-weight:700;text-align:right">${_x["contrib"]:,.0f}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:10px;">'
                    f'<table style="width:100%;border-collapse:collapse">'
                    f'<thead><tr style="border-bottom:1px solid #30363d">'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:left">Event</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">Prize</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">AgeW</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">Contribution</th>'
                    f'</tr></thead><tbody>{_rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                _clamp_note = ("✅ Clamped to 1.0 — top-5 level"
                               if _r["ratio"] >= 1.0
                               else f"→ {_r['bo_val']*100:.1f}% of a top-5 team")
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;">'
                    f'<div style="display:flex;flex-direction:column;gap:8px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:12px;color:#8b949e;">① BO_sum (top {min(10, len(_r["calcs"]))})</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#f0b429;">${_r["bo_sum"]:,.0f}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:12px;color:#8b949e;">② ref₅</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#c9d1d9;">${_r["ref5"]:,.0f}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;border-top:1px solid #30363d;padding-top:8px;">'
                    f'<span style="font-size:12px;color:#8b949e;">③ ratio = min(1.0, {_r["bo_sum"]/1000:.1f}K / {_r["ref5"]/1000:.1f}K)</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#c9d1d9;">{_r["ratio"]:.4f}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid #30363d;padding-top:8px;">'
                    f'<span style="font-size:12px;color:#8b949e;">④ BO = curve({_r["ratio"]:.4f})</span>'
                    f'<span style="font-size:30px;font-weight:700;color:#f0b429;">{_r["bo_val"]:.4f}</span></div>'
                    f'<div style="font-size:10px;color:#8b949e;text-align:center;">{_clamp_note}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
            elif st.session_state.bo_sim_entries:
                st.info("Click **Calculate Bounty Offered** to see results.", icon="💡")
            else:
                st.info("Add at least one win to start.", icon="💡")

    # ══════════════════════════════════════════════════════════════
    with tab_bc2:
        st.subheader("💰 Factor 2 — Bounty Collected")

        if "bc_sim_entries" not in st.session_state:
            st.session_state.bc_sim_entries = [
                {"opponent": "NAVI",      "opp_bo": 1.000, "pool": 1_000_000, "days_ago":  5},
                {"opponent": "G2 Esports","opp_bo": 0.843, "pool":   500_000, "days_ago": 20},
            ]
        if "bc_sim_result" not in st.session_state:
            st.session_state.bc_sim_result = None

        col_r, col_l = st.columns([2, 3])
        with col_l:
            _bc_fc = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:8px 0;">'
                '<div style="text-align:center;margin-bottom:16px;">'
                '<span style="font-size:13px;font-weight:700;color:#3fb950;">💰 Bounty Collected — How is it calculated?</span>'
                '</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:8px 22px;">'
                '<div style="font-size:13px;font-weight:700;color:#c9d1d9;">All wins in the last 180 days</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:2px;">📅 Only wins at events with a prize pool count</div>'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0d0d;border:1px solid #f85149;border-radius:8px;padding:8px 14px;margin:0 24px 4px;">'
                '<div style="font-size:12px;font-weight:700;color:#f85149;">⚠️ No prize pool? Win doesn\'t count.</div>'
                '<div style="font-size:12px;color:#c9d1d9;margin-top:3px;">Online qualifiers and events with $0 pool are excluded entirely.</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d1a1a;border:1px solid #3fb950;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#3fb950;margin-bottom:6px;">Step 1 — Score each win</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">Each win is scored by three things: <strong>how strong the opponent was</strong> (their BO factor), <strong>how recent the win was</strong> (age weight), and <strong>how big the tournament was</strong> (event weight).</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#061010;border-radius:4px;padding:5px 8px;">'
                'entry = opp_BO × age_weight × event_weight'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d1a1a;border:1px solid #3fb950;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#3fb950;margin-bottom:6px;">Step 2 — Average your best 10 wins</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">Take your 10 highest-scoring wins and average them. This rewards consistency — you need to beat strong opponents repeatedly, not just once.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#061010;border-radius:4px;padding:5px 8px;">'
                'BC_pre = Σ top-10 entries / 10'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d1a1a;border:1px solid #3fb950;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#3fb950;margin-bottom:6px;">Step 3 — Smooth with the curve</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">The same log-scale curve used in BO is applied — so even a team that only beats modest opponents still gets a meaningful score.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#061010;border-radius:4px;padding:5px 8px;">'
                'BC = f(BC_pre) = 1 / (1 + |log₁₀(BC_pre)|)'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#0d1a1a;border:2px solid #3fb950;border-radius:8px;padding:10px 28px;">'
                '<div style="font-size:14px;font-weight:700;color:#3fb950;">💰 BC Factor &nbsp; [0 → 1]</div>'
                '</div></div>'
                # WHY BC section moved under flowchart
                '<div style="border-top:1px solid #21262d;margin-top:20px;padding-top:16px;">'
                '<div style="font-size:13px;font-weight:700;color:#3fb950;margin-bottom:8px;">Why does beating one strong team beat beating ten weak ones?</div>'
                '<div style="font-size:12px;color:#c9d1d9;margin-bottom:10px;">'
                'It comes down to <strong>averages</strong>. Your BC score is the average entry value of your best 10 wins. '
                'One win against a top team (BO=1.0) contributes 1.0 to that average. '
                'Ten wins against weak teams (BO=0.1) each contribute 0.1 — the average stays at 0.1. '
                'After the curve: <em>f(1.0) = 1.000</em> vs <em>f(0.1) = 0.500</em>. '
                'The strong-team route scores double — and the curve even helps the weak route not hit zero.'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(_bc_fc, unsafe_allow_html=True)
            st.markdown('<span class="tag-v">✓ VERIFIED — BC = curve(Σ_top10_adjusted / 10) confirmed numerically</span>', unsafe_allow_html=True)

        with col_r:
            st.markdown(
                '<div style="background:#1a0f2e;border:1px solid #7c3aed;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px;font-weight:600;color:#c4b5fd;text-align:center;">✏️ <strong>Try it yourself!</strong>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown("#### Build a win record")
            st.caption("Set opponent BO and event prize pool — both scale the entry value.")

            with st.form("bc_add_form", clear_on_submit=True):
                _bc1, _bc2 = st.columns(2)
                _bopp    = _bc1.text_input("Opponent", placeholder="NAVI")
                _boppbo  = _bc2.slider("Opp. BO factor", 0.0, 1.0, 0.70, 0.01)
                _bc4, _bc3 = st.columns(2)
                _bdays   = _bc4.number_input("Days ago", min_value=0, max_value=179, value=20)
                _bpool   = _bc3.number_input("Prize pool ($)", min_value=0,
                                             max_value=2_000_000, value=500_000, step=50_000,
                                             help="💡 Bigger events give a higher event_weight multiplier — "
                                                  "a win at a $1M tournament counts roughly twice as much "
                                                  "as the same win at a $100k event. Events with $0 prize "
                                                  "pool are fully excluded from BC.")
                if st.form_submit_button("➕ Add win", use_container_width=True) and _bopp:
                    st.session_state.bc_sim_entries.append(
                        {"opponent": _bopp, "opp_bo": float(_boppbo),
                         "pool": int(_bpool), "days_ago": int(_bdays)})
                    st.session_state.bc_sim_result = None

            if st.session_state.bc_sim_entries:
                for _i, _e in enumerate(st.session_state.bc_sim_entries):
                    _ec1, _ec2, _ec3 = st.columns([3, 4, 0.7])
                    _ec1.markdown(f"<span style='font-size:12px'>vs {_e['opponent']}</span>",
                                  unsafe_allow_html=True)
                    _ec2.markdown(
                        f"<span style='font-size:11px;color:#8b949e'>BO={_e['opp_bo']:.2f} · ${_e['pool']:,} · {_e['days_ago']}d</span>",
                        unsafe_allow_html=True)
                    if _ec3.button("✕", key=f"bc_rm_{_i}"):
                        st.session_state.bc_sim_entries.pop(_i)
                        st.session_state.bc_sim_result = None
                        st.rerun()

                st.markdown("---")
                if st.button("🧮 Calculate Bounty Collected", use_container_width=True,
                             type="primary"):
                    _now = datetime.now()
                    _calcs = []
                    for _e in st.session_state.bc_sim_entries:
                        _md   = _now - timedelta(days=_e["days_ago"])
                        _aw   = age_weight(_md, _now)
                        _ew   = event_stakes(max(_e["pool"], 1)) if _e["pool"] > 0 else 0.0
                        _entry = _e["opp_bo"] * _aw * _ew
                        _calcs.append({"opponent": _e["opponent"], "opp_bo": _e["opp_bo"],
                                       "aw": _aw, "ew": _ew, "entry": _entry})
                    _calcs.sort(key=lambda x: x["entry"], reverse=True)
                    _top10   = _calcs[:10]
                    _bc_pre  = sum(x["entry"] for x in _top10) / 10
                    _bc_val  = curve(_bc_pre) if _bc_pre > 0 else 0.0
                    st.session_state.bc_sim_result = {
                        "calcs": _calcs, "top10": _top10,
                        "bc_pre": _bc_pre, "bc_val": _bc_val,
                    }

            if st.session_state.bc_sim_result:
                _r = st.session_state.bc_sim_result
                st.markdown("##### 📋 Step-by-step result")
                _rows = ""
                for _idx, _x in enumerate(_r["calcs"]):
                    _top   = _idx < 10
                    _bg    = "background:#0d2818" if _top else "background:#161b22;opacity:0.55"
                    _badge = ('<span style="background:#3fb950;color:#000;border-radius:3px;'
                              'padding:0 4px;font-size:9px;font-weight:700;margin-left:4px;">TOP10</span>'
                              if _top else '')
                    _rows += (
                        f'<tr style="{_bg}">'
                        f'<td style="padding:5px 6px;font-size:11px;color:#c9d1d9">vs {_x["opponent"]}{_badge}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#f0b429;text-align:right">{_x["opp_bo"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#79c0ff;text-align:right">{_x["aw"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">{_x["ew"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#3fb950;font-weight:700;text-align:right">{_x["entry"]:.4f}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:10px;">'
                    f'<table style="width:100%;border-collapse:collapse">'
                    f'<thead><tr style="border-bottom:1px solid #30363d">'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:left">vs Opponent</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">opp_BO</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">AgeW</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">EvW</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">Entry</th>'
                    f'</tr></thead><tbody>{_rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;">'
                    f'<div style="display:flex;flex-direction:column;gap:8px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:12px;color:#8b949e;">① Sum top {min(10, len(_r["calcs"]))}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#c9d1d9;">{sum(x["entry"] for x in _r["top10"]):.4f}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:12px;color:#8b949e;">② BC_pre = &#x3A3; / 10</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#c9d1d9;">{_r["bc_pre"]:.4f}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid #30363d;padding-top:8px;">'
                    f'<span style="font-size:12px;color:#8b949e;">③ BC = curve({_r["bc_pre"]:.4f})</span>'
                    f'<span style="font-size:30px;font-weight:700;color:#3fb950;">{_r["bc_val"]:.4f}</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
            elif st.session_state.bc_sim_entries:
                st.info("Click **Calculate Bounty Collected** to see results.", icon="💡")
            else:
                st.info("Add at least one win to start.", icon="💡")

    # ══════════════════════════════════════════════════════════════
    with tab_on2:
        st.subheader("🕸️ Factor 3 — Opponent Network")

        if "on_sim_entries" not in st.session_state:
            st.session_state.on_sim_entries = [
                {"opponent": "NAVI",       "opp_on": 0.622, "pool": 1_000_000, "days_ago":  5},
                {"opponent": "MOUZ",       "opp_on": 0.460, "pool":   500_000, "days_ago": 20},
            ]
        if "on_sim_result" not in st.session_state:
            st.session_state.on_sim_result = None

        col_r, col_l = st.columns([2, 3])
        with col_l:
            st.markdown(
                '<div style="background:#0d0d1a;border:1px solid #484f58;border-radius:8px;'
                'padding:10px 16px;margin-bottom:12px;font-size:12px;color:#c9d1d9;">'
                '<strong style="color:#79c0ff;">Why does this factor exist?</strong> '
                'Valve designed the Opponent Network to measure the <em>depth</em> of a team\'s competitive activity — '
                'not just who you beat, but how well-connected those opponents are within the broader scene. '
                'Teams that consistently compete against each other converge toward shared, accurate ratings. '
                'The PageRank-style iteration ensures the scores reflect the whole ecosystem, not just isolated results.'
                '</div>',
                unsafe_allow_html=True)
            _on_fc = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:8px 0;">'
                '<div style="text-align:center;margin-bottom:16px;">'
                '<span style="font-size:13px;font-weight:700;color:#79c0ff;">🕸️ Opponent Network — How is it calculated?</span>'
                '</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:8px 22px;">'
                '<div style="font-size:13px;font-weight:700;color:#c9d1d9;">All wins in the last 180 days</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:2px;">📅 Only wins at events with a prize pool count</div>'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d0d1a;border:1px solid #79c0ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#79c0ff;margin-bottom:6px;">Initialise — start from Bounty Offered</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">The algorithm starts with each team\'s BO value as a placeholder. Since ON depends on other teams\' ON scores (a circular problem), it needs a starting point.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#060610;border-radius:4px;padding:5px 8px;">'
                'ON₀[team] = BO_factor[team]  ∀ teams'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0d0d;border:1px solid #f85149;border-radius:8px;padding:8px 14px;margin:0 24px 4px;">'
                '<div style="font-size:12px;font-weight:700;color:#f85149;">⚠️ No prize pool? Win doesn\'t count.</div>'
                '<div style="font-size:12px;color:#c9d1d9;margin-top:3px;">Same rule as BC — events without prize money are excluded.</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d0d1a;border:1px solid #79c0ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#79c0ff;margin-bottom:6px;">Step 1 — Score each win (using opponent\'s network score)</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">Unlike BC (which uses the opponent\'s prize earnings), ON uses the opponent\'s <em>network score</em>. Beating a well-connected team is worth more than beating a rich one.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#060610;border-radius:4px;padding:5px 8px;">'
                'entry = opp_ON × age_weight × event_weight'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d0d1a;border:1px solid #79c0ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#79c0ff;margin-bottom:6px;">Step 2 — Average your best 10 wins (no curve)</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">Same averaging as BC — but ON skips the final curve step. The raw average IS the ON score. This means ON can be lower than BC for equivalent win quality.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#060610;border-radius:4px;padding:5px 8px;">'
                'ON_new = Σ top-10 entries / 10  &nbsp;⚠️ no curve applied'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a1a0a;border:2px dashed #f0b429;border-radius:8px;padding:8px 14px;margin:0 24px 4px;">'
                '<div style="font-size:12px;font-weight:700;color:#f0b429;">↺ Repeat 6 times — like PageRank</div>'
                '<div style="font-size:12px;color:#c9d1d9;margin-top:3px;">The whole calculation runs 6 times. Each round uses the previous round\'s ON scores as inputs, refining the values until they stabilise. This mirrors how Google PageRank works for web pages.</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#0d0d1a;border:2px solid #79c0ff;border-radius:8px;padding:10px 28px;">'
                '<div style="font-size:14px;font-weight:700;color:#79c0ff;">🕸️ ON Factor &nbsp; [0 → 1]</div>'
                '</div></div>'
                '</div>'
            )
            st.markdown(_on_fc, unsafe_allow_html=True)
            st.markdown('<span class="tag-v">✓ VERIFIED — ON = Σ_top10_adjusted / 10, NO curve, confirmed numerically</span>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### 🔀 ON vs BC — what each rewards")
            st.markdown("""
| Scenario | BC rewards | ON rewards |
|---|---|---|
| Beat a rich team at a big event | ✅ High BC | Depends on their network |
| Beat a team with many wins vs good opponents | Neutral | ✅ High ON |
| Beat the same team 3× | Counted 3× | **Counted 3× too** |
| Beat obscure teams | Low BC | Low ON |

**Key insight:** ON propagates *connectivity*. A team that has beaten many opponents who have themselves beaten many opponents scores highly — regardless of prize money.
""")
            st.markdown("#### 📈 PageRank iterations — how values converge")
            st.caption("Watch how all five teams' ON scores stabilise over 6 rounds of the algorithm.")
            # Show how ON converges iteratively using real-ish values
            on_init  = [1.0, 0.8, 0.6, 0.4, 0.2]  # fake 5-team initial (BO)
            teams_pg = ["Team A", "Team B", "Team C", "Team D", "Team E"]
            # Simulate 6 rounds of updates (toy example)
            iters_data = {"Iter 0 (=BO)": on_init[:]}
            vals = on_init[:]
            for it in range(1, 7):
                new_vals = [
                    min(1.0, (vals[1]*0.9 + vals[2]*0.8) / 2),
                    min(1.0, (vals[0]*0.5 + vals[3]*0.7) / 2),
                    min(1.0, (vals[1]*0.6 + vals[4]*0.9) / 2),
                    min(1.0, (vals[2]*0.8 + vals[0]*0.3) / 2),
                    min(1.0, (vals[3]*0.7 + vals[2]*0.5) / 2),
                ]
                iters_data[f"Iter {it}"] = new_vals[:]
                vals = new_vals
            fig_pg = go.Figure()
            for i, team in enumerate(teams_pg):
                fig_pg.add_trace(go.Scatter(
                    x=list(iters_data.keys()),
                    y=[iters_data[k][i] for k in iters_data],
                    mode="lines+markers", name=team,
                ))
            fig_pg.update_layout(
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d", title="ON value"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
                legend=dict(orientation="h", y=1.15),
                margin=dict(l=10,r=10,t=30,b=10), height=280)
            st.plotly_chart(fig_pg, use_container_width=True, config={"staticPlot": True})
            st.caption(
                "Each line shows one team's ON value across iterations 0–6. "
                "Iteration 0 is simply the teams' BO values (the starting guess). "
                "By round 3–4 the values barely move — this is convergence. "
                "The real engine runs exactly 6 iterations. "
                "Teams whose opponents have high ON scores see their own ON rise with each pass."
            )

        with col_r:
            st.markdown(
                '<div style="background:#1a0f2e;border:1px solid #7c3aed;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px;font-weight:600;color:#c4b5fd;text-align:center;">✏️ <strong>Try it yourself!</strong>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown("#### Build an opponent network")
            st.caption("The opp. ON values you enter represent those opponents' converged ON scores.")

            with st.form("on_add_form", clear_on_submit=True):
                _oc1, _oc2 = st.columns(2)
                _oopp   = _oc1.text_input("Opponent", placeholder="MOUZ")
                _ooppon = _oc2.slider("Opp. ON factor", 0.0, 1.0, 0.45, 0.01)
                _oc3, _oc4 = st.columns(2)
                _opool  = _oc3.number_input("Prize pool ($)", min_value=0,
                                            max_value=2_000_000, value=500_000, step=50_000)
                _odays  = _oc4.number_input("Days ago", min_value=0, max_value=179, value=20)
                if st.form_submit_button("➕ Add win", use_container_width=True) and _oopp:
                    st.session_state.on_sim_entries.append(
                        {"opponent": _oopp, "opp_on": float(_ooppon),
                         "pool": int(_opool), "days_ago": int(_odays)})
                    st.session_state.on_sim_result = None

            if st.session_state.on_sim_entries:
                for _i, _e in enumerate(st.session_state.on_sim_entries):
                    _ec1, _ec2, _ec3 = st.columns([3, 4, 0.7])
                    _ec1.markdown(f"<span style='font-size:12px'>vs {_e['opponent']}</span>",
                                  unsafe_allow_html=True)
                    _ec2.markdown(
                        f"<span style='font-size:11px;color:#8b949e'>ON={_e['opp_on']:.2f} · ${_e['pool']:,} · {_e['days_ago']}d</span>",
                        unsafe_allow_html=True)
                    if _ec3.button("✕", key=f"on_rm_{_i}"):
                        st.session_state.on_sim_entries.pop(_i)
                        st.session_state.on_sim_result = None
                        st.rerun()

                st.markdown("---")
                if st.button("🧮 Calculate Opponent Network", use_container_width=True,
                             type="primary"):
                    _now = datetime.now()
                    _calcs = []
                    for _e in st.session_state.on_sim_entries:
                        _md    = _now - timedelta(days=_e["days_ago"])
                        _aw    = age_weight(_md, _now)
                        _ew    = event_stakes(max(_e["pool"], 1)) if _e["pool"] > 0 else 0.0
                        _entry = _e["opp_on"] * _aw * _ew
                        _calcs.append({"opponent": _e["opponent"], "opp_on": _e["opp_on"],
                                       "aw": _aw, "ew": _ew, "entry": _entry})
                    _calcs.sort(key=lambda x: x["entry"], reverse=True)
                    _top10  = _calcs[:10]
                    _on_val = sum(x["entry"] for x in _top10) / 10
                    st.session_state.on_sim_result = {
                        "calcs": _calcs, "top10": _top10, "on_val": _on_val,
                    }

            if st.session_state.on_sim_result:
                _r = st.session_state.on_sim_result
                st.markdown("##### 📋 Step-by-step result")
                _rows = ""
                for _idx, _x in enumerate(_r["calcs"]):
                    _top   = _idx < 10
                    _bg    = "background:#0d1626" if _top else "background:#161b22;opacity:0.55"
                    _badge = ('<span style="background:#79c0ff;color:#000;border-radius:3px;'
                              'padding:0 4px;font-size:9px;font-weight:700;margin-left:4px;">TOP10</span>'
                              if _top else '')
                    _rows += (
                        f'<tr style="{_bg}">'
                        f'<td style="padding:5px 6px;font-size:11px;color:#c9d1d9">vs {_x["opponent"]}{_badge}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#79c0ff;text-align:right">{_x["opp_on"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#79c0ff;text-align:right">{_x["aw"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">{_x["ew"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#79c0ff;font-weight:700;text-align:right">{_x["entry"]:.4f}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:10px;">'
                    f'<table style="width:100%;border-collapse:collapse">'
                    f'<thead><tr style="border-bottom:1px solid #30363d">'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:left">vs Opponent</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">opp_ON</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">AgeW</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">EvW</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">Entry</th>'
                    f'</tr></thead><tbody>{_rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;">'
                    f'<div style="display:flex;flex-direction:column;gap:8px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:12px;color:#8b949e;">① Sum top {min(10, len(_r["calcs"]))}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#c9d1d9;">{sum(x["entry"] for x in _r["top10"]):.4f}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid #30363d;padding-top:8px;">'
                    f'<span style="font-size:12px;color:#8b949e;">② ON = &#x3A3; / 10 &nbsp;(no curve)</span>'
                    f'<span style="font-size:30px;font-weight:700;color:#79c0ff;">{_r["on_val"]:.4f}</span></div>'
                    f'<div style="font-size:10px;color:#8b949e;text-align:center;">In the real engine opp_ON values are themselves updated over 6 PageRank iterations</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
            elif st.session_state.on_sim_entries:
                st.info("Click **Calculate Opponent Network** to see results.", icon="💡")
            else:
                st.info("Add at least one win to start.", icon="💡")

    # ══════════════════════════════════════════════════════════════
    with tab_lan2:
        st.subheader("🖥️ Factor 4 — LAN Wins")

        if "lan_sim_events" not in st.session_state:
            st.session_state.lan_sim_events = [
                {"event": "IEM Major 2026",   "default_days": 10, "is_lan": True, "custom_timing": False,
                 "matches": [{"days_ago": 10}, {"days_ago": 10}, {"days_ago": 10}]},
                {"event": "BLAST Fall 2025",  "default_days": 90, "is_lan": True, "custom_timing": False,
                 "matches": [{"days_ago": 90}, {"days_ago": 90}]},
            ]
        if "lan_sim_result" not in st.session_state:
            st.session_state.lan_sim_result = None

        col_r, col_l = st.columns([2, 3])
        with col_l:
            _lan_fc = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:8px 0;">'
                '<div style="text-align:center;margin-bottom:16px;">'
                '<span style="font-size:13px;font-weight:700;color:#f85149;">🖥️ LAN Wins — How is it calculated?</span>'
                '</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:8px 22px;">'
                '<div style="font-size:13px;font-weight:700;color:#c9d1d9;">All wins in the last 180 days</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:2px;">📅 But only wins played in person — at LAN events</div>'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0a0d;border:1px solid #f85149;border-radius:8px;padding:8px 14px;margin:0 24px 4px;">'
                '<div style="font-size:12px;font-weight:700;color:#f85149;">⚠️ Online? Doesn\'t count. No exceptions.</div>'
                '<div style="font-size:12px;color:#c9d1d9;margin-top:3px;">Unlike BC/ON, there\'s no prize pool requirement — any LAN win counts. But online wins are fully excluded.</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0a0d;border:1px solid #f85149;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f85149;margin-bottom:6px;">Step 1 — Each LAN win gets an age score</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">Every LAN win starts at 1.0 and is scaled down by how long ago it happened. A win yesterday = 1.0. A win 5 months ago ≈ 0.2. No prize money or opponent strength involved.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#0e0508;border-radius:4px;padding:5px 8px;">'
                'entry = 1.0 × age_weight &nbsp;(no event stakes, no curve, no opp. strength)'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0a0d;border:1px solid #f85149;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f85149;margin-bottom:6px;">Step 2 — Average your best 10 LAN wins</div>'
                '<div style="font-size:13px;color:#c9d1d9;margin-bottom:6px;">Take the 10 highest-scoring LAN wins and average them. You can win multiple matches at the same event — each counts separately. A perfect score requires 10 recent LAN wins.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#0e0508;border-radius:4px;padding:5px 8px;">'
                'LAN = Σ top-10 entries / 10 &nbsp;·&nbsp; max = 1.000 (10 wins all within 30 days)'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#1a0a0d;border:2px solid #f85149;border-radius:8px;padding:10px 28px;">'
                '<div style="font-size:14px;font-weight:700;color:#f85149;">🖥️ LAN Factor &nbsp; [0 → 1]</div>'
                '</div></div>'
                '</div>'
            )
            st.markdown(_lan_fc, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### Effect of age on LAN score")
            st.caption("Recent LAN wins are worth much more than old ones. 10 wins 3 months ago scores the same as 5 wins last month.")
            scenarios_lan = [
                ("10 wins last month\n(age=1.0 each)",  [1.0]*10),
                ("5 fresh + 5 old\n(age 1.0 & 0.5)",   [1.0]*5+[0.5]*5),
                ("10 wins 3 months ago\n(age≈0.5 each)",[0.5]*10),
                ("5 wins last month\n(age=1.0 each)",   [1.0]*5),
            ]
            fig_lan = go.Figure()
            for label, entries in scenarios_lan:
                lan_val = top_n_sum(entries) / TOP_N
                fig_lan.add_trace(go.Bar(
                    x=[label.replace("\n","<br>")], y=[lan_val],
                    text=[f"{lan_val:.3f}"], textposition="outside",
                    marker_color="#f0b429" if lan_val > 0.8 else
                                  "#79c0ff" if lan_val > 0.5 else "#f85149",
                    showlegend=False,
                ))
            fig_lan.update_layout(
                yaxis=dict(title="LAN factor", gridcolor="#21262d", range=[0,1.2]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), xaxis=dict(gridcolor="#21262d"),
                margin=dict(l=10,r=10,t=10,b=80), height=340)
            st.plotly_chart(fig_lan, use_container_width=True, config={"staticPlot": True})
            st.markdown('<span class="tag-v">✓ VERIFIED — 1.0 × age_weight, top-10, /10, no curve, no event stakes</span>', unsafe_allow_html=True)

        with col_r:
            st.markdown(
                '<div style="background:#1a0f2e;border:1px solid #7c3aed;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px;font-weight:600;color:#c4b5fd;text-align:center;">✏️ <strong>Try it yourself!</strong>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown("#### Build a LAN event record")
            st.caption(
                "Add events and set how many matches you played there. "
                "Each match is a separate win. Use the timing toggle to set individual days per match."
            )

            with st.form("lan_add_event_form", clear_on_submit=True):
                _le1, _le2 = st.columns([3, 2])
                _levent   = _le1.text_input("Event name", placeholder="IEM Major 2026")
                _llan     = _le1.checkbox("LAN event", value=True)
                if not _llan:
                    _le1.caption("ℹ️ Non-LAN wins count as 0 for this factor")
                _ldays    = _le2.number_input("Days ago (all matches)", min_value=0, max_value=179, value=20)
                _lmatches = _le2.number_input("Matches won", min_value=1, max_value=10, value=1)
                if st.form_submit_button("➕ Add event", use_container_width=True) and _levent:
                    st.session_state.lan_sim_events.append({
                        "event": _levent,
                        "default_days": int(_ldays),
                        "is_lan": bool(_llan),
                        "custom_timing": False,
                        "matches": [{"days_ago": int(_ldays)} for _ in range(int(_lmatches))]
                    })
                    st.session_state.lan_sim_result = None

            for _ei, _ev in enumerate(st.session_state.lan_sim_events):
                # Ensure legacy entries have custom_timing key
                if "custom_timing" not in _ev:
                    st.session_state.lan_sim_events[_ei]["custom_timing"] = False
                _lan_badge = (
                    '<span style="background:#f85149;color:#fff;border-radius:3px;'
                    'padding:1px 5px;font-size:9px;font-weight:700;">LAN</span>'
                    if _ev["is_lan"] else
                    '<span style="background:#484f58;color:#c9d1d9;border-radius:3px;'
                    'padding:1px 5px;font-size:9px;">online</span>'
                )
                _hdr1, _hdr3 = st.columns([5, 0.7])
                _hdr1.markdown(
                    f"<span style='font-size:16px;font-weight:700;color:#c9d1d9'>{_ev['event']}</span> "
                    f"{_lan_badge} &nbsp; <span style='font-size:11px;color:#8b949e'>"
                    f"{len(_ev['matches'])} match(es) · {_ev['default_days']}d ago</span>",
                    unsafe_allow_html=True)
                if _hdr3.button("✕", key=f"lan_rm_ev_{_ei}"):
                    st.session_state.lan_sim_events.pop(_ei)
                    st.session_state.lan_sim_result = None
                    st.rerun()
                # One toggle per event for per-match timing
                _custom = st.toggle(
                    f"Set individual timing per match",
                    value=_ev["custom_timing"],
                    key=f"lan_ct_{_ei}")
                if _custom != _ev["custom_timing"]:
                    st.session_state.lan_sim_events[_ei]["custom_timing"] = _custom
                    if not _custom:
                        # Reset all matches to default
                        for _mi in range(len(_ev["matches"])):
                            st.session_state.lan_sim_events[_ei]["matches"][_mi]["days_ago"] = _ev["default_days"]
                    st.session_state.lan_sim_result = None
                    st.rerun()
                if _ev["custom_timing"]:
                    for _mi, _m in enumerate(_ev["matches"]):
                        _cur_days = _m["days_ago"] if _m["days_ago"] is not None else _ev["default_days"]
                        _new_days = st.slider(
                            f"Match {_mi+1} — days ago",
                            min_value=0, max_value=179,
                            value=int(_cur_days),
                            key=f"lan_slider_{_ei}_{_mi}")
                        if _new_days != _m.get("days_ago"):
                            st.session_state.lan_sim_events[_ei]["matches"][_mi]["days_ago"] = _new_days
                            st.session_state.lan_sim_result = None

            st.markdown("---")
            if st.button("🧮 Calculate LAN Wins", use_container_width=True, type="primary"):
                _now = datetime.now()
                _calcs_all = []
                for _ev in st.session_state.lan_sim_events:
                    for _mi, _m in enumerate(_ev["matches"]):
                        _days = _m["days_ago"] if _m["days_ago"] is not None else _ev["default_days"]
                        _md   = _now - timedelta(days=_days)
                        _aw   = age_weight(_md, _now)
                        _calcs_all.append({
                            "event": _ev["event"],
                            "match_label": f"Match {_mi+1}",
                            "is_lan": _ev["is_lan"],
                            "days": _days,
                            "aw": _aw,
                            "entry": _aw if _ev["is_lan"] else 0.0,
                        })
                _lan_only = sorted(
                    [x for x in _calcs_all if x["is_lan"]],
                    key=lambda x: x["entry"], reverse=True)
                _excl     = [x for x in _calcs_all if not x["is_lan"]]
                _top10    = _lan_only[:10]
                _lan_val  = sum(x["entry"] for x in _top10) / 10 if _top10 else 0.0
                st.session_state.lan_sim_result = {
                    "lan_only": _lan_only, "excl": _excl,
                    "top10": _top10, "lan_val": _lan_val,
                }

            if st.session_state.lan_sim_result:
                _r = st.session_state.lan_sim_result
                st.markdown("##### 📋 Step-by-step result")
                _rows = ""
                for _idx, _x in enumerate(_r["lan_only"]):
                    _top   = _idx < 10
                    _bg    = "background:#1a0d0d" if _top else "background:#161b22;opacity:0.55"
                    _badge = ('<span style="background:#f85149;color:#fff;border-radius:3px;'
                              'padding:0 4px;font-size:9px;font-weight:700;margin-left:4px;">TOP10</span>'
                              if _top else '')
                    _rows += (
                        f'<tr style="{_bg}">'
                        f'<td style="padding:5px 6px;font-size:11px;color:#c9d1d9">'
                        f'{_x["event"]} — {_x["match_label"]}{_badge}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#f85149;text-align:center">🖥️</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">{_x["days"]}d</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#79c0ff;text-align:right">{_x["aw"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#f85149;font-weight:700;text-align:right">{_x["entry"]:.4f}</td>'
                        f'</tr>'
                    )
                for _x in _r["excl"]:
                    _rows += (
                        f'<tr style="background:#161b22;opacity:0.4">'
                        f'<td style="padding:5px 6px;font-size:11px;color:#8b949e">{_x["event"]} — {_x["match_label"]}'
                        f'<span style="background:#484f58;color:#c9d1d9;border-radius:3px;padding:0 4px;font-size:9px;margin-left:4px;">online — excluded</span></td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#484f58;text-align:center">🌐</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#484f58;text-align:right">{_x["days"]}d</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#484f58;text-align:right">{_x["aw"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#484f58;text-align:right">—</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:10px;">'
                    f'<table style="width:100%;border-collapse:collapse">'
                    f'<thead><tr style="border-bottom:1px solid #30363d">'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:left">Event / Match</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:center">Type</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">Days</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">AgeW</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">Entry</th>'
                    f'</tr></thead><tbody>{_rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;">'
                    f'<div style="display:flex;flex-direction:column;gap:8px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:12px;color:#8b949e;">LAN wins (total)</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#c9d1d9;">{len(_r["lan_only"])} · top {min(10, len(_r["lan_only"]))} used</span></div>'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:12px;color:#8b949e;">① Sum top {min(10, len(_r["lan_only"]))}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#c9d1d9;">{sum(x["entry"] for x in _r["top10"]):.4f}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid #30363d;padding-top:8px;">'
                    f'<span style="font-size:12px;color:#8b949e;">② LAN = Σ / 10</span>'
                    f'<span style="font-size:30px;font-weight:700;color:#f85149;">{_r["lan_val"]:.4f}</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
            elif st.session_state.lan_sim_events:
                st.info("Click **Calculate LAN Wins** to see results.", icon="💡")
            else:
                st.info("Add at least one event to start.", icon="💡")

    # ══════════════════════════════════════════════════════════════
    with tab_h2h:
        st.subheader("⚔️ Phase 2 — Head-to-Head (Glicko/Elo)")
        col_r, col_l = st.columns([2, 3])
        with col_l:
            st.markdown(r"""
**What is H2H (Head-to-Head)?** After the four factor scores are computed, every match in the 180-day window updates team ratings in a second pass — like an Elo/Glicko chess system. Beating a highly-rated team earns you more points; losing to them costs less.

**How it works in plain English:**
- Every team starts with a **Factor Score** (400–2000) as their rating
- Matches are replayed chronologically, oldest first
- Each match shifts ratings up for the winner, down for the loser
- An upset (weak team beats strong team) moves ratings more than an expected win

**The maths (for the curious):**
$$g(\text{RD}) \approx 0.9728 \qquad E(A\text{ vs }B) = \frac{1}{1 + 10^{-g \cdot (r_A - r_B)/400}}$$
$$K = 32 \times \text{age\_weight} \qquad \Delta_{\text{winner}} = K \cdot (1 - E)$$

**Key facts:**
- K-factor uses **age_weight only** — recent matches move ratings more
- No event stakes, no LAN multiplier — a win is a win
- BASE_K = **32** · Processing is **chronological** (oldest first), which matters: a win in month 1 raises the winner's rating, making their later wins worth more

**Why oldest first?** Reversing the order would give different results. Processing in order means each match is evaluated against the ratings at *that point in time*, not retroactively.
""")
            st.markdown('<span class="tag-v">✓ VERIFIED — K = BASE_K × age_weight only; no event stakes; no LAN mult</span>', unsafe_allow_html=True)
        with col_r:
            st.markdown("#### 🎛️ K-Factor & Rating Shift Calculator")
            k_days  = st.slider("Match days ago", 0, 180, 45, 5, key="h2h_days")
            k_rdiff = st.slider("Winner rating − Loser rating", -600, 600, 0, 50, key="h2h_diff")

            dummy_cut = datetime(2026, 3, 2)
            aw_h2h = age_weight(dummy_cut - timedelta(days=k_days), dummy_cut)
            K_eff  = BASE_K * aw_h2h
            r_w2   = 1200 + k_rdiff / 2
            r_l2   = 1200 - k_rdiff / 2
            E_w2   = expected_win(r_w2, r_l2)
            dw2    = K_eff * (1 - E_w2)
            dl2    = -K_eff * E_w2

            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                        padding:16px;margin-top:8px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:12px;">
                <div style="text-align:center;flex:1;">
                  <div style="font-size:12px;color:#8b949e">Age Weight</div>
                  <div style="font-size:22px;font-weight:700;color:#79c0ff">{aw_h2h:.3f}</div>
                </div>
                <div style="text-align:center;flex:1;">
                  <div style="font-size:12px;color:#8b949e">K = 32 × {aw_h2h:.3f}</div>
                  <div style="font-size:22px;font-weight:700;color:#58a6ff">{K_eff:.2f}</div>
                </div>
                <div style="text-align:center;flex:1;">
                  <div style="font-size:12px;color:#8b949e">E(winner)</div>
                  <div style="font-size:22px;font-weight:700;color:#c9d1d9">{E_w2:.3f}</div>
                </div>
              </div>
              <div style="display:flex;justify-content:space-around;border-top:1px solid #30363d;padding-top:12px;">
                <div style="text-align:center;">
                  <div style="font-size:12px;color:#8b949e">Winner gains</div>
                  <div style="font-size:26px;font-weight:700;color:#3fb950">+{dw2:.2f}</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:12px;color:#8b949e">Loser loses</div>
                  <div style="font-size:26px;font-weight:700;color:#f85149">{dl2:.2f}</div>
                </div>
              </div>
              <div style="font-size:11px;color:#8b949e;text-align:center;margin-top:8px">
                  {'Upset (underdog wins → bigger gain)' if k_rdiff < -100 else
                   'Favourite wins → smaller gain' if k_rdiff > 100 else
                   'Even match'}
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Winner's rating gain vs. rating difference:**")
            st.caption("The curve shifts up/down as K changes (set by 'Match days ago' slider above). The ◆ symbol shows your current slider position.")
            _h2h_diffs_c = list(range(-600, 601, 10))
            _h2h_gains_c = [K_eff * (1 - expected_win(1200+d/2, 1200-d/2)) for d in _h2h_diffs_c]
            fig_sh = go.Figure()
            fig_sh.add_trace(go.Scatter(
                x=_h2h_diffs_c, y=_h2h_gains_c,
                mode="lines",
                line=dict(color="#58a6ff", width=2.5),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
                name="Winner gain curve",
            ))
            fig_sh.add_trace(go.Scatter(
                x=[k_rdiff], y=[dw2],
                mode="markers",
                marker=dict(color="#f0b429", size=13, symbol="diamond",
                            line=dict(color="#c9d1d9", width=1.5)),
                name="Current setting",
                showlegend=False,
            ))
            fig_sh.add_vline(x=0, line_dash="dot", line_color="#484f58")
            fig_sh.add_annotation(
                x=k_rdiff, y=dw2,
                text=f"+{dw2:.1f}",
                showarrow=True, arrowhead=2, arrowwidth=1,
                arrowcolor="#f0b429", ax=30, ay=-25,
                font=dict(size=10, color="#f0b429"),
                bgcolor="rgba(22,27,34,0.85)", bordercolor="#f0b429", borderwidth=1)
            fig_sh.update_layout(
                xaxis=dict(title="Winner rating − Loser rating", gridcolor="#21262d",
                           tickvals=[-600,-400,-200,0,200,400,600]),
                yaxis=dict(title="Winner rating gain (pts)", gridcolor="#21262d", range=[0, 35]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10), height=260)
            st.plotly_chart(fig_sh, use_container_width=True, config={"staticPlot": False})
            st.caption(
                "Left of 0 = underdog wins (big gain). Right of 0 = favourite wins (small gain). "
                "Move the 'Match days ago' slider to see the whole curve scale up (fresh match) or down (old match)."
            )

    # ══════════════════════════════════════════════════════════════
    with tab_seed:
        st.subheader("🌱 Phase 1 — Combining Factors → Factor Score")
        col_r, col_l = st.columns([2, 3])
        with col_l:
            st.markdown(r"""
After computing the four factor values, they are combined into a single seed.

**Step 1.** Simple unweighted average (25% each):
$$\text{average} = \frac{\text{BO} + \text{BC} + \text{ON} + \text{LAN}}{4}$$

**Step 2.** Find the minimum and maximum average across all eligible teams.

**Step 3.** Linearly interpolate (lerp) to [400, 2000]:
$$\text{Factor Score} = 400 + \frac{\text{average} - \text{avg}_{\min}}{\text{avg}_{\max} - \text{avg}_{\min}} \times 1600$$

The **worst eligible team** always gets seed = **400**.
The **best eligible team** always gets seed = **2000**.
All others are spaced proportionally between them.

**Verification — Vitality:**
$$\frac{1.000 + 0.923 + 0.460 + 1.000}{4} = 0.846$$

Since Vitality has the highest average, they are the reference maximum:
$$\text{Factor Score} = 400 + \frac{0.846 - 0.000}{0.846 - 0.000} \times 1600 = \mathbf{2000.0} \checkmark$$
""")
            st.markdown('<span class="tag-v">✓ VERIFIED — simple average (25% each) confirmed from Vitality data</span>', unsafe_allow_html=True)
        with col_r:
            st.markdown(
                '<div style="background:#1a0f2e;border:1px solid #7c3aed;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:18px;font-weight:600;color:#c4b5fd;text-align:center;">✏️ <strong>Try it yourself!</strong>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown("#### Factor Score Sandbox")
            st.caption("Each slider controls one factor. The BO slider uses prize money (age-adjusted) — the curve and ref₅ are applied automatically.")

            # BO as prize money (ref5 ≈ $334,320 — top-5 benchmark)
            _REF5_DEMO = 334_320
            st.markdown("**🏆 Bounty Offered** — prize money earned (age-adjusted $)")
            # Hint shown ABOVE slider: read current value from session state (updates on every rerun)
            _bo_prev    = st.session_state.get("seed_bo_money", 220_000)
            _bo_p_ratio = min(1.0, _bo_prev / _REF5_DEMO)
            _bo_p_score = curve(_bo_p_ratio) if _bo_p_ratio > 0 else 0.0
            _bo_p_cap   = _bo_prev >= _REF5_DEMO
            _bo_p_pct   = min(100, int(_bo_prev / _REF5_DEMO * 100))
            st.markdown(
                f'<div style="font-size:12px;padding:7px 12px;margin-bottom:4px;'
                f'background:#161b22;border-radius:6px;border-left:3px solid '
                f'{"#3fb950" if _bo_p_cap else "#f0b429"};">'
                f'{"✅ At or above top-5 benchmark — BO capped at <strong style=\'color:#f0b429\'>1.000</strong>" if _bo_p_cap else f"<strong style=\'color:#c9d1d9\'>${_bo_prev:,}</strong> = {_bo_p_pct}% of the top-5 benchmark (${_REF5_DEMO:,})"}'
                f' &nbsp;→&nbsp; BO = <strong style="color:#f0b429">{_bo_p_score:.4f}</strong>'
                f'{"" if _bo_p_cap else f" &nbsp;·&nbsp; <span style=\'color:#8b949e;\'>Slide past ${_REF5_DEMO:,} to reach 1.000</span>"}'
                f'</div>',
                unsafe_allow_html=True)
            _s_bo_money = st.slider(
                "Prize money ($)", 0, 600_000, 220_000, 5_000, key="seed_bo_money",
                label_visibility="collapsed")
            _s_bo_ratio = min(1.0, _s_bo_money / _REF5_DEMO)
            _s_bo = curve(_s_bo_ratio) if _s_bo_ratio > 0 else 0.0

            _s_bc  = st.slider("💰 Bounty Collected", 0.00, 1.00, 0.55, 0.01, key="seed_bc")
            _s_on  = st.slider("🕸️ Opponent Network", 0.00, 1.00, 0.45, 0.01, key="seed_on")
            _s_lan = st.slider("🖥️ LAN Wins",         0.00, 1.00, 0.40, 0.01, key="seed_lan")

            _s_avg = (_s_bo + _s_bc + _s_on + _s_lan) / 4
            # Benchmark: Vitality-level values (confirmed)
            _ref_bo, _ref_bc, _ref_on, _ref_lan = 1.000, 0.923, 0.460, 1.000
            _ref_avg = (_ref_bo + _ref_bc + _ref_on + _ref_lan) / 4  # = 0.846
            _min_avg = 0.0
            _s_seed = 400 + (_s_avg - _min_avg) / max(_ref_avg - _min_avg, 1e-9) * 1600
            _s_seed = min(2000.0, _s_seed)

            # Factor breakdown card
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-top:8px;">'
                f'<div style="font-size:12px;color:#8b949e;margin-bottom:10px;">Factor breakdown</div>'
                f'{"".join([f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;"><span style="font-size:12px;color:{c};min-width:130px;">{lbl}</span><div style="flex:1;height:8px;background:#21262d;border-radius:4px;overflow:hidden;"><div style="width:{v*100:.0f}%;height:100%;background:{c};border-radius:4px;"></div></div><span style="font-size:12px;font-weight:700;color:{c};min-width:42px;text-align:right">{v:.3f}</span></div>""" for lbl, v, c in [("🏆 BO", _s_bo, "#f0b429"), ("💰 BC", _s_bc, "#3fb950"), ("🕸️ ON", _s_on, "#79c0ff"), ("🖥️ LAN", _s_lan, "#f85149")]])}'
                f'<div style="border-top:1px solid #30363d;margin:10px 0;padding-top:10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:13px;color:#8b949e;">Average (25% each)</span>'
                f'<span style="font-size:18px;font-weight:700;color:#c9d1d9;">{_s_avg:.4f}</span></div>'
                f'</div>'
                f'<div style="text-align:center;margin-top:12px;">'
                f'<div style="font-size:11px;color:#8b949e;">→ Factor Score (lerped to 400–2000)</div>'
                f'<div style="font-size:42px;font-weight:700;color:{"#3fb950" if _s_seed >= 1600 else "#f0b429" if _s_seed >= 1000 else "#f85149"}">{_s_seed:.0f}</div>'
                f'<div style="font-size:11px;color:#8b949e;">{"🏆 Top-tier range" if _s_seed >= 1600 else "🥈 Mid-tier range" if _s_seed >= 1000 else "🔴 Lower tier"}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True)

            st.markdown("**vs. benchmark (Vitality-level):**")
            _bench_labels = ["BO", "BC", "ON", "LAN", "Avg", "Score"]
            _bench_yours  = [_s_bo, _s_bc, _s_on, _s_lan, _s_avg, _s_seed/2000]
            _bench_ref    = [_ref_bo, _ref_bc, _ref_on, _ref_lan, _ref_avg, 1.0]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=_bench_yours, theta=_bench_labels, fill="toself",
                name="Your team", line_color="#58a6ff", fillcolor="rgba(88,166,255,0.15)"))
            fig_radar.add_trace(go.Scatterpolar(
                r=_bench_ref, theta=_bench_labels, fill="toself",
                name="Benchmark (Vitality)", line_color="#f0b429", fillcolor="rgba(240,180,41,0.08)"))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d", linecolor="#30363d"),
                    angularaxis=dict(gridcolor="#30363d", linecolor="#30363d"),
                    bgcolor="rgba(0,0,0,0)"),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"),
                legend=dict(orientation="h", y=-0.15),
                margin=dict(l=30,r=30,t=20,b=40), height=300)
            st.plotly_chart(fig_radar, use_container_width=True, config={"staticPlot": False})
            st.caption(
                f"BO benchmark: slide past ${_REF5_DEMO:,} to reach BO = 1.000 (top-5 level). "
                "To match Vitality's Factor Score of 2000, you'd need BO=1.0, BC=0.923, ON=0.460, LAN=1.0."
            )

    # ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# PAGE 5  ·  TEAM BREAKDOWN
# ══════════════════════════════════════════════════════════════════


elif page == "🔍 Team Breakdown":

    st.title("🔍 Team Breakdown")

    if base_standings.empty:
        st.warning("No data loaded. Check sidebar.")
        st.stop()

    # ── Team selector ─────────────────────────────────────────────
    team_list = base_standings["team"].tolist()
    sel_team  = st.selectbox("Select team:", team_list, key="bd_team")
    ex        = base_standings[base_standings["team"] == sel_team].iloc[0]
    raw_ms    = team_match_history.get(sel_team, [])
    bo_prizes = bo_prizes_map.get(sel_team, [])

    # ── Original values for delta display (sim mode) ──────────────
    _orig_ex = None
    if (mode_active == "updated") and sel_team in original_standings["team"].values:
        _orig_matches = original_standings[original_standings["team"] == sel_team].sort_values("rank")
        _orig_ex = _orig_matches.iloc[0]

    if (mode_active == "updated"):
        st.markdown(
            '<div style="background:#0d3d1a;border:1px solid #3fb950;border-radius:8px;'
            'padding:10px 14px;font-size:12px;color:#56d364;margin-bottom:12px;">'
            '📡 <strong>Updated to today</strong> — All values reflect standings as of '
            f'{cutoff_dt.strftime("%b %d, %Y")}. '
            'Age weights, H2H adjustments, and factor scores are fully recalculated with live data.'
            '</div>',
            unsafe_allow_html=True,
        )

    def _delta_str(new_val, old_val, fmt="+.1f", small=True):
        """Return HTML span with +/- delta, or empty string if no sim."""
        if _orig_ex is None:
            return ""
        d = new_val - old_val
        if abs(d) < 0.01:
            return ""
        c = "#3fb950" if d > 0 else "#f85149"
        fs = "10px" if small else "12px"
        return f' <span style="color:{c};font-size:{fs};font-weight:600">({d:{fmt}})</span>'

    def _eff_age(m):
        """Return the effective age weight for a match (mode-aware)."""
        if (mode_active == "updated"):
            return age_weight(m["date"], sim_cutoff_dt)
        return m["age_w"]

    def _eff_h2h(m, team):
        """Return the effective H2H adjustment for a match (sim-aware)."""
        if mode_active == "updated":
            mid = m.get("match_id")
            if mid in sim_match_h2h:
                entry = sim_match_h2h[mid]
                if m["result"] == "W":
                    return entry["w_delta"]
                else:
                    return entry["l_delta"]
            return 0.0
        return m.get("h2h_adj", 0.0)

    # Pre-compute globals needed for calculation boxes
    all_avgs       = base_standings["seed_combined"].tolist()
    max_avg        = max(all_avgs) if all_avgs else 1.0
    min_avg        = min(all_avgs) if all_avgs else 0.0
    bo_sums_sorted = sorted(base_standings["bo_sum"].tolist(), reverse=True)
    ref5_bo        = bo_sums_sorted[4] if len(bo_sums_sorted) >= 5 else (bo_sums_sorted[-1] if bo_sums_sorted else 1.0)
    opp_bo_map     = base_standings.drop_duplicates("team").set_index("team")["bo_factor"].to_dict()
    opp_on_map     = base_standings.drop_duplicates("team").set_index("team")["on_factor"].to_dict()

    # ── Helpers ───────────────────────────────────────────────────
    # Per-factor max for relative bar scaling in breakdown
    _bd_bo_max  = max(base_standings["bo_factor"].max(),  1e-9)
    _bd_bc_max  = max(base_standings["bc_factor"].max(),  1e-9)
    _bd_on_max  = max(base_standings["on_factor"].max(),  1e-9)
    _bd_lan_max = max(base_standings["lan_factor"].max(), 1e-9)

    def _bar(val, color, w=60, max_val=1.0):
        pct = max(0.0, min(1.0, float(val) / max(float(max_val), 1e-9))) * 100
        return (
            f'<span style="display:inline-flex;align-items:center;gap:4px;">' +
            f'<span style="display:inline-block;width:{w}px;height:7px;background:#21262d;border-radius:4px;overflow:hidden;">' +
            f'<span style="display:block;width:{pct:.0f}%;height:100%;background:{color};border-radius:4px;"></span></span>' +
            f'<span style="font-size:11px;color:#8b949e">{float(val):.3f}</span></span>'
        )

    def _calc_box(lines_html: str) -> str:
        return (
            '<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;' +
            'padding:10px 14px;font-size:12px;font-family:monospace;line-height:1.8;">' +
            lines_html + '</div>'
        )

    def _factor_band(emoji, name, value, color, max_val=1.0, delta_html=""):
        pct = max(0.0, min(1.0, float(value) / max(float(max_val), 1e-9))) * 100
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;' +
            f'background:#161b22;border-left:4px solid {color};' +
            f'border-radius:6px;padding:10px 14px;margin-bottom:6px;">' +
            f'<span style="font-size:14px;font-weight:700;color:{color}">{emoji} {name}</span>' +
            f'<span style="display:flex;align-items:center;gap:8px;">' +
            f'<span style="display:inline-block;width:100px;height:8px;background:#21262d;border-radius:4px;overflow:hidden;">' +
            f'<span style="display:block;width:{pct:.0f}%;height:100%;background:{color};border-radius:4px;"></span></span>' +
            f'<span style="font-size:20px;font-weight:700;color:{color};min-width:50px;text-align:right">{float(value):.4f}</span>' +
            f'{delta_html}' +
            f'</span></div>'
        )

    # ═══════════════════════════════════════════════════════════════
    # TWO MAIN TABS
    # ═══════════════════════════════════════════════════════════════
    tab_snap, tab_hist = st.tabs(["📊 Current Snapshot", "📈 Historical Development"])

    # ═══════════════════════════════════════════════════════════════
    with tab_snap:
    # ═══════════════════════════════════════════════════════════════

        # ── KPI header ────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        _rk_d = f" ({int(_orig_ex['rank']) - int(ex['rank']):+d})" if _orig_ex is not None and int(ex['rank']) != int(_orig_ex['rank']) else ""
        k1.metric("🌍 Global Rank",   f"#{int(ex['rank'])}{_rk_d}")
        k2.metric("🎯 Final Score",   f"{ex['total_points']:,.1f}",
                  delta=f"{ex['total_points'] - _orig_ex['total_points']:+.1f}" if _orig_ex is not None else None)
        k3.metric("🌱 Factor Score",  f"{ex['seed']:,.1f}",
                  delta=f"{ex['seed'] - _orig_ex['seed']:+.1f}" if _orig_ex is not None else None)
        k4.metric("⚔️ H2H Adj.",      f"{ex['h2h_delta']:+.1f}",
                  delta=f"{ex['h2h_delta'] - _orig_ex['h2h_delta']:+.1f}" if _orig_ex is not None else None)
        k5.metric("📝 Record",        f"{int(ex['wins'])}W / {int(ex['losses'])}L")

        # Global context (compact)
        pct_rank = (ex["seed_combined"] - min_avg) / max(max_avg - min_avg, 1e-9) * 100
        st.caption(
            f"Pool: **{len(base_standings)} teams** · "
            f"5th BO ref: **${ref5_bo:,.0f}** · "
            f"Factor avg: **{ex['seed_combined']:.4f}** · "
            f"Percentile: **{pct_rank:.1f}th**"
        )

        # ════════════════════════════════════════════════════════
        # SIMULATION WATERFALL — "Why did the score change?"
        # ════════════════════════════════════════════════════════
        if _orig_ex is not None:
            st.markdown("---")
            st.markdown("### 🔬 Score Change Breakdown")

            # ── Compute deltas ─────────────────────────────────
            _d_total = ex["total_points"] - _orig_ex["total_points"]
            _d_seed  = ex["seed"]         - _orig_ex["seed"]
            _d_h2h   = ex["h2h_delta"]    - _orig_ex["h2h_delta"]

            # Factor-level deltas (each contributes 25% to the avg)
            _d_bo  = (ex["bo_factor"]  - _orig_ex["bo_factor"])  / 4.0
            _d_bc  = (ex["bc_factor"]  - _orig_ex["bc_factor"])  / 4.0
            _d_on  = (ex["on_factor"]  - _orig_ex["on_factor"])  / 4.0
            _d_lan = (ex["lan_factor"] - _orig_ex["lan_factor"]) / 4.0
            _d_avg = _d_bo + _d_bc + _d_on + _d_lan

            # Lerp scale: convert avg-units → seed points
            _lerp_scale = 1600.0 / max(max_avg - min_avg, 1e-9)

            # Factor contributions in seed points
            _pts_bo  = _d_bo  * _lerp_scale
            _pts_bc  = _d_bc  * _lerp_scale
            _pts_on  = _d_on  * _lerp_scale
            _pts_lan = _d_lan * _lerp_scale
            _pts_factors_sum = _pts_bo + _pts_bc + _pts_on + _pts_lan

            # Lerp shift = residual (min/max change effect)
            _pts_lerp = _d_seed - _pts_factors_sum

            # ── Waterfall chart ────────────────────────────────

            _old_label = _gd["cutoff_datetime"].strftime("%b")
            _new_label = cutoff_dt.strftime("%b")
            _wf_labels = [
                f"{_old_label} Score<br><b>{_orig_ex['total_points']:,.0f}</b>",
                "🏆 BO",
                "💰 BC",
                "🕸️ ON",
                "🖥️ LAN",
                "📐 Lerp Shift",
                "⚔️ H2H",
                f"{_new_label} Score<br><b>{ex['total_points']:,.0f}</b>",
            ]
            _wf_values = [_orig_ex["total_points"], _pts_bo, _pts_bc, _pts_on, _pts_lan, _pts_lerp, _d_h2h, ex["total_points"]]
            _wf_measures = ["absolute", "relative", "relative", "relative", "relative", "relative", "relative", "total"]

            fig_wf = go.Figure(go.Waterfall(
                x=_wf_labels,
                y=_wf_values,
                measure=_wf_measures,
                text=[
                    f"{_orig_ex['total_points']:,.0f}",
                    f"{_pts_bo:+.1f}",
                    f"{_pts_bc:+.1f}",
                    f"{_pts_on:+.1f}",
                    f"{_pts_lan:+.1f}",
                    f"{_pts_lerp:+.1f}",
                    f"{_d_h2h:+.1f}",
                    f"{ex['total_points']:,.0f}",
                ],
                textposition="outside",
                textfont=dict(size=12, color="#c9d1d9"),
                connector=dict(line=dict(color="#30363d", width=1, dash="dot")),
                increasing=dict(marker_color="#3fb950"),
                decreasing=dict(marker_color="#f85149"),
                totals=dict(marker_color="#58a6ff"),
            ))
            fig_wf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
                yaxis=dict(gridcolor="#21262d", title="Points"),
                xaxis=dict(gridcolor="#21262d"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True, config={"staticPlot": True})

            # ── Insight text (directly under waterfall) ───────
            _drivers = [
                ("BC decay", _pts_bc), ("ON decay", _pts_on),
                ("LAN decay", _pts_lan), ("Lerp shift", _pts_lerp),
                ("H2H shift", _d_h2h),
            ]
            _drivers.sort(key=lambda x: abs(x[1]), reverse=True)
            _top_driver = _drivers[0]

            if abs(_pts_lerp) > abs(_pts_bc) + abs(_pts_on) + abs(_pts_lan):
                _insight = (
                    f"⚠️ The **Lerp normalization** ({_pts_lerp:+.0f} pts) is the dominant driver. "
                    "This happens when teams drop out of the eligible pool, shifting the min/max baseline. "
                    "The actual factor decay is modest."
                )
            elif _d_total > 10:
                _insight = (
                    f"📈 This team **gains** points mainly from **{_top_driver[0]}** ({_top_driver[1]:+.0f} pts). "
                    "Likely the lerp normalization is inflating their score as weaker teams fall out."
                )
            elif _d_total < -10:
                _biggest_neg = min(_drivers, key=lambda x: x[1])
                _insight = (
                    f"📉 This team **loses** points mainly from **{_biggest_neg[0]}** ({_biggest_neg[1]:+.0f} pts). "
                    "Older matches are losing weight, reducing their factor scores."
                )
            else:
                _insight = "➡️ This team is relatively stable — factor decay and normalization largely cancel out."

            # Lerp context data
            _orig_pool = original_standings["team"].nunique()
            _sim_pool  = base_standings["team"].nunique()
            _dropped   = max(0, _orig_pool - _sim_pool)
            _orig_min_avg = original_standings["seed_combined"].min()
            _orig_max_avg = original_standings["seed_combined"].max()
            _lerp_color = "#3fb950" if _pts_lerp > 0 else "#f85149"

            st.markdown(
                f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
                f'padding:12px 16px;font-size:12px;color:#c9d1d9;line-height:1.6">'
                f'{_insight}</div>',
                unsafe_allow_html=True,
            )

            # Lerp Shift expander (collapsible, doesn't take space by default)
            if abs(_pts_lerp) > 1.0:
                with st.expander(f"📐 What is the Lerp Shift? ({_pts_lerp:+.0f} pts)"):
                    st.markdown(f"""
The VRS maps each team's factor average to a score between **400** and **2000** using min-max normalization.
The worst eligible team always gets 400, the best always 2000.

**This is not a simulation artifact** — Valve does this every month with the then-current pool.
But it means that score changes don't just come from a team's own performance.
""")
                    st.code(
                        "Factor Score = 400 + (team_avg − min_avg) / (max_avg − min_avg) × 1600",
                        language=None,
                    )
                    st.markdown(f"""
When the eligible pool changes (teams drop out due to inactivity or window expiry),
`min_avg` and `max_avg` shift — and **every team's score shifts with them**.
""")
                    lc1, lc2, lc3, lc4 = st.columns(4)
                    lc1.metric("Pool size", f"{_sim_pool}", f"{-_dropped} teams")
                    lc2.metric("min avg", f"{min_avg:.4f}", f"{min_avg - _orig_min_avg:+.4f}")
                    lc3.metric("max avg", f"{max_avg:.4f}", f"{max_avg - _orig_max_avg:+.4f}")
                    lc4.metric("Lerp effect", f"{_pts_lerp:+.0f} pts")

        st.markdown("---")

        # ════════════════════════════════════════════════════════
        # PHASE 1 — FOUR FACTORS (vertical stack, compact)
        # ════════════════════════════════════════════════════════
        st.markdown("### 🌱 Phase 1 — Four Factors")

        # ── BOUNTY OFFERED ────────────────────────────────────────
        st.markdown(_factor_band("🏆", "Bounty Offered", ex["bo_factor"], "#f0b429", _bd_bo_max,
            _delta_str(ex["bo_factor"], _orig_ex["bo_factor"], "+.4f") if _orig_ex is not None else ""), unsafe_allow_html=True)
        col_bo_l, col_bo_r = st.columns([1, 1])
        with col_bo_l:
            ratio_bo = ex["bo_sum"] / max(ref5_bo, 1.0)
            st.markdown(_calc_box(
                f'<div>BO Sum (top-10 scaled wins) = <strong style="color:#f0b429">${ex["bo_sum"]:,.0f}</strong></div>' +
                f'<div>÷ 5th ref = ${ref5_bo:,.0f} → ratio = {ratio_bo:.4f}</div>' +
                f'<div>curve(min(1.0, {min(1.0,ratio_bo):.4f})) = <strong style="color:#f0b429">{ex["bo_factor"]:.4f}</strong></div>'
            ), unsafe_allow_html=True)
        with col_bo_r:
            if bo_prizes:
                # Recalculate age weights for simulation if active
                _disp_bo = []
                for bp in bo_prizes:
                    bp_copy = dict(bp)
                    if mode_active == "updated":
                        try:
                            dt = datetime.strptime(str(bp["event_date"]).strip(), "%Y-%m-%d")
                            bp_copy["age_weight"] = age_weight(dt, cutoff_dt)
                            bp_copy["scaled_prize"] = bp["prize_won"] * bp_copy["age_weight"]
                        except Exception:
                            pass
                    _disp_bo.append(bp_copy)
                bp_html = "".join(
                    f'<tr style="border-bottom:1px solid #21262d;">' +
                    f'<td style="padding:4px 6px;color:#8b949e;font-size:11px">{bp["event_date"]}</td>' +
                    f'<td style="padding:4px 6px;text-align:right;color:#c9d1d9;font-size:11px">${bp["prize_won"]:,.0f}</td>' +
                    f'<td style="padding:4px 6px">{_bar(bp["age_weight"],"#f0b429",40)}</td>' +
                    f'<td style="padding:4px 6px;text-align:right;color:#f0b429;font-size:11px;font-weight:600">${bp["scaled_prize"]:,.0f}</td>' +
                    f'</tr>'
                    for bp in _disp_bo
                )
                total_sc = sum(b["scaled_prize"] for b in _disp_bo)
                st.markdown(
                    '<div style="overflow-y:auto;max-height:180px;border:1px solid #30363d;border-radius:6px;">' +
                    '<table style="width:100%;border-collapse:collapse;">' +
                    '<thead><tr style="background:#161b22;color:#8b949e;font-size:9px;text-transform:uppercase;">' +
                    '<th style="padding:6px">Date</th><th style="padding:6px;text-align:right">Prize</th>' +
                    '<th style="padding:6px">Age Wt</th><th style="padding:6px;text-align:right">Scaled</th>' +
                    '</tr></thead><tbody>' + bp_html + '</tbody>' +
                    f'<tfoot><tr style="background:#161b22;border-top:1px solid #30363d;">' +
                    f'<td colspan="3" style="padding:5px 6px;color:#8b949e;font-size:10px">BO Sum</td>' +
                    f'<td style="padding:5px 6px;text-align:right;color:#f0b429;font-weight:700">${total_sc:,.0f}</td>' +
                    '</tr></tfoot></table></div>',
                    unsafe_allow_html=True
                )
            else:
                st.caption("No prize data available.")

        st.markdown("<div style='margin:10px 0'>", unsafe_allow_html=True)

        # ── BOUNTY COLLECTED ──────────────────────────────────────
        st.markdown(_factor_band("💰", "Bounty Collected", ex["bc_factor"], "#3fb950", _bd_bc_max,
            _delta_str(ex["bc_factor"], _orig_ex["bc_factor"], "+.4f") if _orig_ex is not None else ""), unsafe_allow_html=True)
        col_bc_l, col_bc_r = st.columns([1, 1])
        bc_matches = [m for m in raw_ms if m["result"] == "W" and m.get("ev_w", 0) > 0]
        bc_entries = sorted(
            [((opp_bo_map.get(m["opponent"], m.get("opp_bo", 0.0)))*_eff_age(m)*m["ev_w"],
              m,
              opp_bo_map.get(m["opponent"], m.get("opp_bo", 0.0)))
             for m in bc_matches],
            key=lambda x: x[0], reverse=True
        )
        sum10_bc = sum(e for e,_,_ in bc_entries[:10])
        with col_bc_l:
            st.markdown(_calc_box(
                f'<div>Σ top-10 entries = <strong style="color:#3fb950">{sum10_bc:.4f}</strong></div>' +
                f'<div>BC_pre = {sum10_bc:.4f} / 10 = {sum10_bc/10:.4f}</div>' +
                f'<div>curve({sum10_bc/10:.4f}) = <strong style="color:#3fb950">{ex["bc_factor"]:.4f}</strong></div>'
            ), unsafe_allow_html=True)
        with col_bc_r:
            if bc_entries:
                bc_html = "".join(
                    f'<tr style="border-bottom:1px solid #21262d;{" " if i<10 else "opacity:0.35;"}">' +
                    f'<td style="padding:4px 5px;color:#8b949e;font-size:10px">{m["date"].strftime("%m-%d")}</td>' +
                    f'<td style="padding:4px 5px;color:#c9d1d9;font-size:10px">{m["opponent"][:14]}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(ob,"#f0b429",30,_bd_bo_max)}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(_eff_age(m),"#e6b430",30)}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(m["ev_w"],"#79c0ff",30)}</td>' +
                    f'<td style="padding:4px 5px;text-align:right;color:#3fb950;font-size:10px;font-weight:600">{e:.3f}</td>' +
                    f'</tr>'
                    for i,(e,m,ob) in enumerate(bc_entries)
                )
                st.markdown(
                    '<div style="overflow-y:auto;max-height:180px;border:1px solid #30363d;border-radius:6px;">' +
                    '<table style="width:100%;border-collapse:collapse;">' +
                    '<thead><tr style="background:#161b22;color:#8b949e;font-size:9px;text-transform:uppercase;">' +
                    '<th style="padding:5px">Date</th><th style="padding:5px">Opp</th>' +
                    '<th style="padding:5px">BO</th><th style="padding:5px">Age</th>' +
                    '<th style="padding:5px">Ev</th><th style="padding:5px;text-align:right">Entry</th>' +
                    '</tr></thead><tbody>' + bc_html + '</tbody>' +
                    f'<tfoot><tr style="background:#161b22;border-top:1px solid #30363d;">' +
                    f'<td colspan="5" style="padding:5px;color:#8b949e;font-size:10px">Top-10 sum/10</td>' +
                    f'<td style="padding:5px;text-align:right;color:#3fb950;font-weight:700">{sum10_bc/10:.4f}</td>' +
                    '</tr></tfoot></table></div>' +
                    '<div style="font-size:9px;color:#484f58;margin-top:2px">Faded = outside top-10</div>',
                    unsafe_allow_html=True
                )
            else:
                st.caption("No event wins found.")

        st.markdown("<div style='margin:10px 0'>", unsafe_allow_html=True)

        # ── OPPONENT NETWORK ──────────────────────────────────────
        st.markdown(_factor_band("🕸️", "Opponent Network", ex["on_factor"], "#79c0ff", _bd_on_max,
            _delta_str(ex["on_factor"], _orig_ex["on_factor"], "+.4f") if _orig_ex is not None else ""), unsafe_allow_html=True)
        col_on_l, col_on_r = st.columns([1, 1])
        on_matches = [m for m in raw_ms if m["result"] == "W" and m.get("ev_w", 0) > 0]
        on_entries = sorted(
            [((opp_on_map.get(m["opponent"], m.get("opp_on", 0.0)))*_eff_age(m)*m["ev_w"],
              m,
              opp_on_map.get(m["opponent"], m.get("opp_on", 0.0)))
             for m in on_matches],
            key=lambda x: x[0], reverse=True
        )
        sum10_on = sum(e for e,_,_ in on_entries[:10])
        with col_on_l:
            st.markdown(_calc_box(
                f'<div>Like BC but uses opponent ON score (PageRank, 6 iterations)</div>' +
                f'<div style="margin-top:6px">Σ top-10 entries = <strong style="color:#79c0ff">{sum10_on:.4f}</strong></div>' +
                f'<div>ON = {sum10_on:.4f} / 10 = <strong style="color:#79c0ff">{ex["on_factor"]:.4f}</strong> (no curve)</div>'
            ), unsafe_allow_html=True)
        with col_on_r:
            if on_entries:
                on_html = "".join(
                    f'<tr style="border-bottom:1px solid #21262d;{" " if i<10 else "opacity:0.35;"}">' +
                    f'<td style="padding:4px 5px;color:#8b949e;font-size:10px">{m["date"].strftime("%m-%d")}</td>' +
                    f'<td style="padding:4px 5px;color:#c9d1d9;font-size:10px">{m["opponent"][:14]}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(on_v,"#79c0ff",30,_bd_on_max)}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(_eff_age(m),"#e6b430",30)}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(m["ev_w"],"#58a6ff",30)}</td>' +
                    f'<td style="padding:4px 5px;text-align:right;color:#79c0ff;font-size:10px;font-weight:600">{e:.3f}</td>' +
                    f'</tr>'
                    for i,(e,m,on_v) in enumerate(on_entries)
                )
                st.markdown(
                    '<div style="overflow-y:auto;max-height:180px;border:1px solid #30363d;border-radius:6px;">' +
                    '<table style="width:100%;border-collapse:collapse;">' +
                    '<thead><tr style="background:#161b22;color:#8b949e;font-size:9px;text-transform:uppercase;">' +
                    '<th style="padding:5px">Date</th><th style="padding:5px">Opp</th>' +
                    '<th style="padding:5px">ON</th><th style="padding:5px">Age</th>' +
                    '<th style="padding:5px">Ev</th><th style="padding:5px;text-align:right">Entry</th>' +
                    '</tr></thead><tbody>' + on_html + '</tbody>' +
                    f'<tfoot><tr style="background:#161b22;border-top:1px solid #30363d;">' +
                    f'<td colspan="5" style="padding:5px;color:#8b949e;font-size:10px">Top-10 sum/10</td>' +
                    f'<td style="padding:5px;text-align:right;color:#79c0ff;font-weight:700">{sum10_on/10:.4f}</td>' +
                    '</tr></tfoot></table></div>' +
                    '<div style="font-size:9px;color:#484f58;margin-top:2px">Faded = outside top-10 · ON from current snapshot</div>',
                    unsafe_allow_html=True
                )
            else:
                st.caption("No event wins found.")

        st.markdown("<div style='margin:10px 0'>", unsafe_allow_html=True)

        # ── LAN WINS ──────────────────────────────────────────────
        st.markdown(_factor_band("🖥️", "LAN Wins", ex["lan_factor"], "#f85149", _bd_lan_max,
            _delta_str(ex["lan_factor"], _orig_ex["lan_factor"], "+.4f") if _orig_ex is not None else ""), unsafe_allow_html=True)
        col_lan_l, col_lan_r = st.columns([1, 1])
        lan_wins_list = sorted(
            [m for m in raw_ms if m["result"] == "W" and m.get("is_lan")],
            key=lambda m: _eff_age(m), reverse=True
        )
        sum10_lan = sum(_eff_age(m) for m in lan_wins_list[:10])
        with col_lan_l:
            st.markdown(_calc_box(
                f'<div>LAN wins in window: <strong style="color:#f85149">{len(lan_wins_list)}</strong></div>' +
                f'<div>Top-10 age weights sum = <strong>{sum10_lan:.4f}</strong></div>' +
                f'<div>LAN = {sum10_lan:.4f} / 10 = <strong style="color:#f85149">{ex["lan_factor"]:.4f}</strong> (no curve)</div>'
            ), unsafe_allow_html=True)
        with col_lan_r:
            if lan_wins_list:
                lan_html = "".join(
                    f'<tr style="border-bottom:1px solid #21262d;{" " if i<10 else "opacity:0.35;"}">' +
                    f'<td style="padding:4px 6px;color:#8b949e;font-size:11px">{m["date"].strftime("%Y-%m-%d")}</td>' +
                    f'<td style="padding:4px 6px;color:#c9d1d9;font-size:11px">{m["opponent"][:16]}</td>' +
                    f'<td style="padding:4px 6px">{_bar(_eff_age(m),"#f85149",55)}</td>' +
                    f'</tr>'
                    for i,m in enumerate(lan_wins_list)
                )
                st.markdown(
                    '<div style="overflow-y:auto;max-height:180px;border:1px solid #30363d;border-radius:6px;">' +
                    '<table style="width:100%;border-collapse:collapse;">' +
                    '<thead><tr style="background:#161b22;color:#8b949e;font-size:9px;text-transform:uppercase;">' +
                    '<th style="padding:6px">Date</th><th style="padding:6px">Opponent</th><th style="padding:6px">Age Wt</th>' +
                    '</tr></thead><tbody>' + lan_html + '</tbody>' +
                    f'<tfoot><tr style="background:#161b22;border-top:1px solid #30363d;">' +
                    f'<td colspan="2" style="padding:5px 6px;color:#8b949e;font-size:10px">Top-10 sum/10</td>' +
                    f'<td style="padding:5px 6px;color:#f85149;font-weight:700">{sum10_lan/10:.4f}</td>' +
                    '</tr></tfoot></table></div>' +
                    '<div style="font-size:9px;color:#484f58;margin-top:2px">Faded = outside top-10 · sorted by recency</div>',
                    unsafe_allow_html=True
                )
            else:
                st.caption("No LAN wins in 6-month window.")

        # ── PHASE 1 RESULT ────────────────────────────────────────
        st.markdown("---")
        combined = ex["seed_combined"]
        st.markdown(f"""
        <div style="background:#1c2128;border:2px solid #58a6ff;border-radius:10px;padding:16px 20px;">
          <div style="font-size:13px;color:#8b949e;margin-bottom:10px">🌱 Phase 1 Result</div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;">
            <div style="text-align:center;background:#161b22;border-radius:6px;padding:8px;">
              <div style="font-size:10px;color:#f0b429">🏆 BO</div>
              <div style="font-size:16px;font-weight:700;color:#f0b429">{ex["bo_factor"]:.4f}</div>
              <div style="font-size:10px;color:#484f58">×0.25 = {ex["bo_factor"]*0.25:.4f}</div>
            </div>
            <div style="text-align:center;background:#161b22;border-radius:6px;padding:8px;">
              <div style="font-size:10px;color:#3fb950">💰 BC</div>
              <div style="font-size:16px;font-weight:700;color:#3fb950">{ex["bc_factor"]:.4f}</div>
              <div style="font-size:10px;color:#484f58">×0.25 = {ex["bc_factor"]*0.25:.4f}</div>
            </div>
            <div style="text-align:center;background:#161b22;border-radius:6px;padding:8px;">
              <div style="font-size:10px;color:#79c0ff">🕸️ ON</div>
              <div style="font-size:16px;font-weight:700;color:#79c0ff">{ex["on_factor"]:.4f}</div>
              <div style="font-size:10px;color:#484f58">×0.25 = {ex["on_factor"]*0.25:.4f}</div>
            </div>
            <div style="text-align:center;background:#161b22;border-radius:6px;padding:8px;">
              <div style="font-size:10px;color:#f85149">🖥️ LAN</div>
              <div style="font-size:16px;font-weight:700;color:#f85149">{ex["lan_factor"]:.4f}</div>
              <div style="font-size:10px;color:#484f58">×0.25 = {ex["lan_factor"]*0.25:.4f}</div>
            </div>
          </div>
          <div style="color:#8b949e;font-size:12px">avg = {combined:.4f} → 400 + ({combined:.4f} − {min_avg:.4f}) / ({max_avg:.4f} − {min_avg:.4f}) × 1600</div>
          <div style="font-size:28px;font-weight:700;color:#58a6ff;margin-top:4px">Factor Score = {ex["seed"]:,.1f}</div>
        </div>""", unsafe_allow_html=True)

        # ── PHASE 2 ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚔️ Phase 2 — Head-to-Head (Glicko)")

        p2c1, p2c2, p2c3 = st.columns(3)
        p2c1.metric("🌱 Factor Score",    f"{ex['seed']:,.1f}")
        p2c2.metric("⚔️ H2H Adj.",     f"{ex['h2h_delta']:+.1f}")
        p2c3.metric("🎯 Final Score",   f"{ex['total_points']:,.1f}")

        import plotly.graph_objects as _go
        fig_p2 = _go.Figure()
        fig_p2.add_trace(_go.Bar(
            name="Factor Score", x=[sel_team], y=[ex["seed"]],
            marker_color="#79c0ff", text=[f"{ex['seed']:.0f}"], textposition="auto",
        ))
        h2h_v = ex["h2h_delta"]
        fig_p2.add_trace(_go.Bar(
            name="H2H Adj.", x=[sel_team], y=[h2h_v],
            marker_color="#3fb950" if h2h_v >= 0 else "#f85149",
            text=[f"{h2h_v:+.1f}"], textposition="auto",
        ))
        fig_p2.update_layout(
            barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"), yaxis=dict(gridcolor="#21262d"),
            legend=dict(orientation="h", y=1.18),
            margin=dict(l=10,r=10,t=40,b=10), height=200,
        )
        st.plotly_chart(fig_p2, use_container_width=True, config={"staticPlot": True})

        # All matches for H2H (newest first)
        if raw_ms:
            # Filter to matches in the sim window if active
            _h2h_ms = raw_ms
            if mode_active == "updated":
                _new_ws = sim_cutoff_dt - timedelta(days=DECAY_DAYS)
                _h2h_ms = [m for m in raw_ms if _new_ws <= m["date"] <= sim_cutoff_dt]

            m_rows = []
            total_h2h = 0.0
            h2h_match_ids = []  # DEBUG: track which matches contribute to total
            for m in sorted(_h2h_ms, key=lambda x: x["date"], reverse=True):
                h2h      = _eff_h2h(m, sel_team)
                total_h2h += h2h
                h2h_match_ids.append((m.get("match_id"), h2h))  # DEBUG
                is_win   = m["result"] == "W"
                aw_disp  = _eff_age(m)
                res_b    = (
                    '<span style="background:#1f4a1f;color:#3fb950;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700">W</span>'
                    if is_win else
                    '<span style="background:#4a1f1f;color:#f85149;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700">L</span>'
                )
                h2h_c    = (
                    f'<span style="color:#3fb950;font-weight:700">+{h2h:.1f}</span>' if h2h > 0 else
                    f'<span style="color:#f85149;font-weight:700">{h2h:.1f}</span>' if h2h < 0 else
                    '<span style="color:#8b949e">0.0</span>'
                )
                m_rows.append(
                    f'<tr style="border-bottom:1px solid #21262d;">' +
                    f'<td style="padding:4px 7px;color:#8b949e;font-size:11px">{m["date"].strftime("%Y-%m-%d")}</td>' +
                    f'<td style="padding:4px 7px">{res_b}</td>' +
                    f'<td style="padding:4px 7px;color:#c9d1d9;font-size:11px">{m["opponent"]}</td>' +
                    f'<td style="padding:4px 7px;text-align:center;font-size:11px">{"🖥️" if (is_win and m.get("is_lan")) else "🌐" if is_win else ""}</td>' +
                    f'<td style="padding:4px 7px">{_bar(aw_disp,"#f0b429",45)}</td>' +
                    f'<td style="padding:4px 7px;text-align:right">{h2h_c}</td>' +
                    f'</tr>'
                )
            n_w = sum(1 for m in _h2h_ms if m["result"]=="W")
            n_l = len(_h2h_ms) - n_w
            h2h_col = "#3fb950" if total_h2h >= 0 else "#f85149"
            st.markdown(
                '<div style="overflow-y:auto;max-height:340px;border:1px solid #30363d;border-radius:8px;">' +
                '<table style="width:100%;border-collapse:collapse;">' +
                '<thead style="position:sticky;top:0;background:#161b22;">' +
                '<tr style="color:#8b949e;font-size:9px;text-transform:uppercase;">' +
                '<th style="padding:7px">Date</th><th style="padding:7px">W/L</th>' +
                '<th style="padding:7px;text-align:left">Opponent</th>' +
                '<th style="padding:7px">Type</th><th style="padding:7px">Age Wt</th>' +
                '<th style="padding:7px;text-align:right">H2H Δ</th>' +
                '</tr></thead><tbody>' + "".join(m_rows) + '</tbody></table></div>' +
                f'<div style="display:flex;justify-content:space-between;margin-top:5px;font-size:11px;color:#8b949e;">' +
                f'<span>{len(_h2h_ms)} matches · {n_w}W {n_l}L</span>' +
                f'<span>Total H2H: <strong style="color:{h2h_col}">{total_h2h:+.1f} pts</strong></span></div>',
                unsafe_allow_html=True
            )

            # DEBUG: Compare standings vs table
            if mode_active == "updated":
                standings_h2h = base_standings.loc[base_standings["team"] == sel_team, "h2h_delta"].values
                if len(standings_h2h) > 0:
                    standings_h2h_val = standings_h2h[0]
                    with st.expander("🔧 DEBUG: H2H Diagnostics"):
                        st.metric("Standings h2h_delta", f"{standings_h2h_val:.1f}")
                        st.metric("Table Total H2H", f"{total_h2h:.1f}")
                        st.metric("Difference", f"{standings_h2h_val - total_h2h:.1f}")
                        st.metric("Matches displayed", len(_h2h_ms))
                        st.metric("sim_match_h2h size", len(sim_match_h2h))

                        # Show which matches have 0 H2H
                        zero_h2h = [mid for mid, h2h in h2h_match_ids if h2h == 0.0]
                        if zero_h2h:
                            st.warning(f"⚠️ {len(zero_h2h)} matches have 0 H2H (not in sim_match_h2h): {zero_h2h[:10]}")
        else:
            st.info("No match history available.")

    # ═══════════════════════════════════════════════════════════════
    with tab_hist:
    # ═══════════════════════════════════════════════════════════════

        @st.cache_data(ttl=3600, show_spinner=False)
        def _load_team_history(team_name: str, all_dates) -> pd.DataFrame:
            rows = []
            for ds, yr in all_dates:
                gd = _cached_load_github_data(ds, yr)
                s  = gd.get("standings", pd.DataFrame())
                if not s.empty and team_name in s["team"].values:
                    row = s[s["team"] == team_name].iloc[0]
                    try:
                        dt = datetime.strptime(ds, "%Y_%m_%d")
                    except Exception:
                        continue
                    rows.append({
                        "date":         dt,
                        "date_label":   dt.strftime("%b %d, %Y"),
                        "rank":         int(row["rank"]),
                        "total_points": float(row["total_points"]),
                        "seed":         float(row["seed"]),
                        "h2h_delta":    float(row["h2h_delta"]),
                        "bo_factor":    float(row.get("bo_factor", 0)),
                        "bc_factor":    float(row.get("bc_factor", 0)),
                        "on_factor":    float(row.get("on_factor", 0)),
                        "lan_factor":   float(row.get("lan_factor", 0)),
                        "wins":         int(row.get("wins", 0)),
                        "losses":       int(row.get("losses", 0)),
                    })
            return pd.DataFrame(sorted(rows, key=lambda r: r["date"])) if rows else pd.DataFrame()

        # Only include snapshots up to and including the currently selected date
        _hist_dates = [(ds, yr) for ds, yr in _all_dates if ds <= _sel_date]
        with st.spinner(f"Loading history for {sel_team}…"):
            hist_df = _load_team_history(sel_team, tuple(_hist_dates))

        # ── Append simulated data point if sim is active ─────────
        if (mode_active == "updated") and not hist_df.empty and sel_team in base_standings["team"].values:
            _sim_row = base_standings[base_standings["team"] == sel_team].iloc[0]
            sim_point = {
                "date":         sim_cutoff_dt,
                "date_label":   "🔮 " + sim_cutoff_dt.strftime("%b %d, %Y"),
                "rank":         int(_sim_row["rank"]),
                "total_points": float(_sim_row["total_points"]),
                "seed":         float(_sim_row["seed"]),
                "h2h_delta":    float(_sim_row["h2h_delta"]),
                "bo_factor":    float(_sim_row.get("bo_factor", 0)),
                "bc_factor":    float(_sim_row.get("bc_factor", 0)),
                "on_factor":    float(_sim_row.get("on_factor", 0)),
                "lan_factor":   float(_sim_row.get("lan_factor", 0)),
                "wins":         int(_sim_row.get("wins", 0)),
                "losses":       int(_sim_row.get("losses", 0)),
                "is_sim":       True,
            }
            hist_df["is_sim"] = False
            hist_df = pd.concat([hist_df, pd.DataFrame([sim_point])], ignore_index=True)
            hist_df = hist_df.sort_values("date").reset_index(drop=True)

        if hist_df.empty:
            st.info(f"No historical data found for {sel_team}.")
        else:
            import plotly.graph_objects as _go2

            # ── Rank + Score dual-axis chart ─────────────────────────
            st.markdown("#### 📊 Rank & Factor Score over time")
            fig_dual = _go2.Figure()

            _bar_colors = ["#7c3aed" if hist_df.iloc[i].get("is_sim", False)
                           else "#79c0ff" for i in range(len(hist_df))]
            fig_dual.add_trace(_go2.Bar(
                x=hist_df["date_label"], y=hist_df["total_points"],
                name="Final Score", marker_color=_bar_colors,
                opacity=0.75, yaxis="y1",
            ))
            fig_dual.add_trace(_go2.Scatter(
                x=hist_df["date_label"], y=hist_df["rank"],
                name="Global Rank", mode="lines+markers",
                line=dict(color="#f0b429", width=2.5),
                marker=dict(size=8, color="#f0b429"),
                yaxis="y2",
            ))

            fig_dual.update_layout(
                yaxis=dict(
                    title="Final Factor Score",
                    gridcolor="#21262d", color="#79c0ff",
                    side="left",
                ),
                yaxis2=dict(
                    title="Global Rank",
                    gridcolor="#21262d", color="#f0b429",
                    side="right", overlaying="y",
                    autorange="reversed",
                    showgrid=False,
                ),
                xaxis=dict(gridcolor="#21262d"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
                legend=dict(orientation="h", y=1.12),
                margin=dict(l=10,r=10,t=40,b=10), height=350,
                barmode="group",
            )
            st.plotly_chart(fig_dual, use_container_width=True, config={"staticPlot": True})

            # Delta vs previous snapshot
            if len(hist_df) >= 2:
                latest = hist_df.iloc[-1]
                prev   = hist_df.iloc[-2]
                d1,d2,d3,d4 = st.columns(4)
                rank_d  = int(latest["rank"]) - int(prev["rank"])
                score_d = latest["total_points"] - prev["total_points"]
                d1.metric("Current Rank",  f"#{latest['rank']}",
                          f"{'+' if rank_d>0 else ''}{rank_d}", delta_color="inverse")
                d2.metric("Final Score",   f"{latest['total_points']:,.0f}", f"{score_d:+.0f}")
                wl = latest["wins"] + latest["losses"]
                d3.metric("Win Rate", f"{latest['wins']/wl*100:.0f}%" if wl>0 else "—")
                d4.metric("Snapshots", str(len(hist_df)))

            st.markdown("---")

            # ── Factor trend ─────────────────────────────────────────
            st.markdown("#### 🔬 Factor Trends")
            fig_f = _go2.Figure()
            for label, col, color in [
                ("BO","bo_factor","#f0b429"), ("BC","bc_factor","#3fb950"),
                ("ON","on_factor","#79c0ff"), ("LAN","lan_factor","#f85149"),
            ]:
                if col in hist_df.columns:
                    fig_f.add_trace(_go2.Scatter(
                        x=hist_df["date_label"], y=hist_df[col],
                        name=label, mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=6, color=color),
                    ))
            fig_f.update_layout(
                yaxis=dict(title="Factor (0–1)", gridcolor="#21262d", range=[0,1.05]),
                xaxis=dict(gridcolor="#21262d"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
                legend=dict(orientation="h", y=1.12),
                margin=dict(l=10,r=10,t=40,b=10), height=280,
            )
            st.plotly_chart(fig_f, use_container_width=True, config={"staticPlot": True})

            st.markdown("---")

            # ── Summary table (sorted by date descending) ─────────────
            st.markdown("#### 📋 All snapshots")
            _tbl = hist_df.copy().sort_values("date", ascending=False)
            disp = _tbl[[
                "date_label","rank","total_points","seed","h2h_delta",
                "bo_factor","bc_factor","on_factor","lan_factor","wins","losses"
            ]].rename(columns={
                "date_label":"Date","rank":"Rank","total_points":"Final Score",
                "seed":"Factor Score","h2h_delta":"H2H Δ",
                "bo_factor":"BO","bc_factor":"BC","on_factor":"ON","lan_factor":"LAN",
                "wins":"W","losses":"L",
            })
            st.dataframe(disp, use_container_width=True, hide_index=True)
