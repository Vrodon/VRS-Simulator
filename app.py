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
import hashlib
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

/* Force CTA card text white — overrides Streamlit's `a * { color: inherit }` rule */
a .vrs-cta-title,
p a .vrs-cta-title,
div a .vrs-cta-title,
[data-testid="stMarkdownContainer"] a .vrs-cta-title {
    color: #ffffff !important;
}
a .vrs-cta-sub,
p a .vrs-cta-sub,
div a .vrs-cta-sub,
[data-testid="stMarkdownContainer"] a .vrs-cta-sub {
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# IMPORTS FROM VRS MODULES
# ══════════════════════════════════════════════════════════════════
from vrs_engine import (
    Store, run_vrs,
    DECAY_DAYS, FLAT_DAYS, RD_FIXED, Q_GLICKO,
    BASE_K, PRIZE_CAP, TOP_N, SEED_MIN, SEED_MAX,
    curve, event_stakes, age_weight, lerp,
    g_rd, expected_win, top_n_sum, G_FIXED,
    next_valve_publication, prev_valve_publication,
)
from data_loaders import load_valve_github_data
from data_loaders.github_loader import _find_all_dates
from data_loaders import (
    fetch_liquipedia_matches,
    load_liquipedia_from_cache,
    liquipedia_cache_exists,
    liquipedia_cache_mtime,
    clear_liquipedia_cache,
    parse_tournament_brackets,
)
from data_loaders.liquipedia_loader import _fetch_page as _fetch_liquipedia_page, _bs4 as _liquipedia_bs4
from utils import get_team_meta, KNOWN_META, COLOR_CYCLE
TEAM_META = KNOWN_META   # alias used throughout app


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_find_all_dates() -> list[tuple[str, str]]:
    """Cached wrapper — GitHub file list, refreshed at most once per hour."""
    return _find_all_dates()


# ── Predicted-mode helpers (Task 6.4) ───────────────────────────────────────
# Walk any active bracket-state picks across events, emit engine-ready
# extra_matches / extra_prizes rows, then run_vrs with them appended. Results
# replace `base_standings` so every page (Ranking Dashboard, Team Breakdown)
# auto-reflects the predicted future.

def _ordinal(n: int) -> str:
    """Convert 1→'1st', 2→'2nd', 3→'3rd', 4→'4th', 21→'21st', etc."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{('th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th')[n % 10]}"


def _format_place(lo: int, hi: int) -> str:
    """Format a placement bucket; lo==hi → single ('3rd'), else range ('5th-8th')."""
    if lo == hi:
        return _ordinal(lo)
    return f"{_ordinal(lo)}-{_ordinal(hi)}"


def _snapshot_standings(date_str: str | None,
                         year: str | None) -> dict[str, int]:
    """
    Load a Valve VRS snapshot and return ``{team_name: rank}``.

    Returns an empty dict when ``date_str`` / ``year`` is missing or the
    snapshot can't be loaded; callers treat empty as "no snapshot picked
    → Swiss synth disabled" per spec lock 10.Q5.
    """
    if not date_str or not year:
        return {}
    try:
        from data_loaders.github_loader import load_valve_github_data
        data = load_valve_github_data(date_str=date_str, year=year)
    except Exception:
        return {}
    standings = data.get("standings")
    if standings is None or len(standings) == 0:
        return {}
    return {row["team"]: int(row["rank"]) for _, row in standings.iterrows()}


# Phase 5 §5.1 — Swiss override computation lives in stage_graph as a pure
# function (callable from any thread / non-Streamlit context). The leading-
# underscore alias keeps the prior call sites (``_resolve``, autofill loop,
# engine emit) source-stable.
from data_loaders.stage_graph import compute_swiss_overrides as _compute_swiss_overrides


def _lookup_prize_for_place(place_label: str, prize_distribution: dict) -> float:
    """
    Resolve a placement label against Liquipedia's prize_distribution.

    First tries an exact label match. Falls back to any published label whose
    range contains our entire range (Liquipedia sometimes publishes coarser
    buckets than the bracket can resolve — e.g. ``"5th-8th"`` covers
    ``"5th-6th"`` derived from a non-bronze SE).
    """
    if place_label in prize_distribution:
        return float(prize_distribution[place_label])

    import re as _re
    m = _re.match(r"(\d+)(?:st|nd|rd|th)(?:-(\d+)(?:st|nd|rd|th))?", place_label)
    if not m:
        return 0.0
    lo = int(m.group(1)); hi = int(m.group(2)) if m.group(2) else lo
    for label, amount in prize_distribution.items():
        m2 = _re.match(r"(\d+)(?:st|nd|rd|th)(?:-(\d+)(?:st|nd|rd|th))?", label)
        if not m2:
            continue
        plo = int(m2.group(1)); phi = int(m2.group(2)) if m2.group(2) else plo
        if plo <= lo and phi >= hi:
            return float(amount)
    return 0.0


def _emit_event_simulation_rows(
    slug: str,
    slug_state: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Convert one event's bracket picks → engine-ready ``(extra_matches, extra_prizes)``.

    Phase 5 §5.1: shares its post-realised body with the fragment render
    via ``stage_graph.resolve_for_render`` + ``vrs_engine.event_simulation
    .emit_simulation_rows``. Same input → same output across both call
    sites; cascade-resolved downstream R1 seeds + Swiss synth bake reach
    the engine identically to the UI.

    Skips played matches (those live in ``matches_df`` via the
    Updated-to-Today fetch — re-emitting would double-count, see
    architecture decision #7). Returns ``([], [])`` when the event has
    no picks and no parseable bracket.
    """
    picks = slug_state["scenarios"][slug_state["active_scenario"]]["picks"]
    ev    = slug_state.get("event_meta") or {}

    if not picks:
        return [], []

    html = _fetch_liquipedia_page(slug)
    if html is None:
        return [], []
    soup    = _liquipedia_bs4(html)
    seeded  = ev.get("seeded_teams") or []
    from data_loaders.liquipedia_loader import (
        _parse_format as _lp_parse_format,
        _parse_per_stage_invites as _lp_per_stage_invites,
    )
    from data_loaders.stage_graph import (
        collect_sub_page_rosters,
        resolve_for_render,
    )
    from vrs_engine.event_simulation import emit_simulation_rows
    prose      = _lp_parse_format(soup)
    per_stage  = _lp_per_stage_invites(soup)
    sub_rost   = collect_sub_page_rosters(slug, soup)
    parsed_stages = parse_tournament_brackets(
        soup, seeded, slug=slug,
        format_prose=prose,
        direct_invitees_by_stage=per_stage,
        sub_page_rosters=sub_rost,
    )

    manual_seeds = slug_state["scenarios"][slug_state["active_scenario"]].get(
        "manual_seeds", {}) or {}
    snapshot = _snapshot_standings(slug_state.get("snapshot_date"),
                                    slug_state.get("snapshot_year"))

    realised = resolve_for_render(parsed_stages, picks,
                                   manual_seeds=manual_seeds,
                                   snapshot_standings=snapshot)

    return emit_simulation_rows(
        realised, picks,
        event_name=ev.get("name", slug),
        prize_pool=float(ev.get("prize_pool", 0) or 0),
        is_lan=bool(ev.get("is_lan", False)),
        event_end=ev.get("end_date") or datetime.now(),
        prize_distribution=ev.get("prize_distribution") or {},
    )


def _collect_predicted_simulation_rows(
    bracket_state: dict,
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Walk every event in ``bracket_state`` that has at least one pick and
    concatenate their simulation rows in chronological order (by event end
    date). Returns ``(extra_matches, extra_prizes, included_slugs)``.

    Architecture decision #13: events with picks auto-include; the explicit
    ``include_in_chain`` toggle is reserved for Day 7.4 multi-event UX.
    """
    candidates: list[tuple[datetime, str, dict]] = []
    for slug, slug_state in bracket_state.items():
        active = slug_state.get("active_scenario", "current")
        scen   = slug_state.get("scenarios", {}).get(active, {})
        picks  = scen.get("picks", {})
        if not picks:
            continue
        ev = slug_state.get("event_meta") or {}
        end = ev.get("end_date") or datetime.now()
        candidates.append((end, slug, slug_state))

    candidates.sort(key=lambda x: x[0])

    all_matches: list[dict] = []
    all_prizes:  list[dict] = []
    included:    list[str]  = []
    for _, slug, slug_state in candidates:
        em, ep = _emit_event_simulation_rows(slug, slug_state)
        if em or ep:
            all_matches.extend(em)
            all_prizes.extend(ep)
            included.append(slug)

    return all_matches, all_prizes, included


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

# Factor-learn deep link: Team Breakdown factor bands use href="?learn=bo|bc|on|lan".
# We route to the "How VRS Works" page and stash the target factor so that page
# can highlight which tab the user should click.
_qp_learn = st.query_params.get("learn", "")
if _qp_learn in ("bo", "bc", "on", "lan"):
    st.query_params.clear()
    st.session_state["main_nav"]    = "📖 How VRS Works"
    st.session_state["learn_focus"] = _qp_learn
    st.rerun()

with st.sidebar:
    st.markdown("## 🎯 CS2 VRS Simulator")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📖 How VRS Works",
        "📊 Ranking Dashboard",
        "🔍 Team Breakdown",
        "🏆 Tournament Predictor",
    ], label_visibility="collapsed", key="main_nav")
    st.markdown("---")

    # ── Standings Mode selector ───────────────────────────────────
    standings_mode = st.radio(
        "Standings Mode",
        ["🏛️ As Published", "📡 Updated to Today", "🔮 Predicted"],
        index=0,
        key="standings_mode",
        help=(
            "As Published: Official Valve standings.\n"
            "Updated to Today: includes all real results since publication (auto-fetched).\n"
            "Predicted: Updated to Today + your bracket picks from the Tournament Predictor."
        ),
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
            disabled=(standings_mode in ("📡 Updated to Today", "🔮 Predicted")),
            help=("Latest snapshot is used for Updated to Today / Predicted modes."
                  if standings_mode in ("📡 Updated to Today", "🔮 Predicted") else
                  "Valve publishes new standings monthly. Select any historical snapshot."),
        )
        if standings_mode in ("📡 Updated to Today", "🔮 Predicted"):
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


# ══════════════════════════════════════════════════════════════════
# AUTO-FETCH "UPDATED TO TODAY" ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def _auto_fetch_updated_standings(
    snapshot_cutoff: datetime,
    snapshot_standings: pd.DataFrame,
    team_match_history: dict,
    bo_prizes_map: dict,
    progress_callback=None,
) -> dict:
    """
    Fetch all CS2 tournaments from Liquipedia since snapshot_cutoff,
    build a clean Store, and run a full VRS recomputation with cutoff = today.

    Returns:
        standings   pd.DataFrame  — updated rankings (empty on error)
        match_h2h   dict          — per-match H2H detail for team breakdown
        cutoff      datetime
        source      "liquipedia" or "fallback"
        match_count int           — number of new series results added
        error       str or None
    """
    from data_loaders import (
        discover_liquipedia_from_portal,
        fetch_liquipedia_matches,
        liquipedia_cache_exists,
        liquipedia_cache_mtime,
        load_liquipedia_from_cache,
    )

    today = datetime.now()

    # Widen discovery backward so events that started before the Valve snapshot
    # but were still "in progress" at snapshot time (Valve drops these whole
    # via their `finished` flag — e.g. PGL Bucharest matches dated days before
    # the snapshot) are rediscovered and backfilled. 30 days comfortably covers
    # typical event durations.
    from datetime import timedelta as _td
    discovery_start_dt = snapshot_cutoff - _td(days=30)
    discovery_start_str = discovery_start_dt.strftime("%Y-%m-%d")
    start_date_str      = snapshot_cutoff.strftime("%Y-%m-%d")
    end_date_str        = today.strftime("%Y-%m-%d")

    try:
        # ── Step 1: Discover + fetch Liquipedia tournaments ───────────
        if progress_callback:
            progress_callback(1, 3, "Discovering tournaments from Liquipedia Portal…")

        discovered = discover_liquipedia_from_portal(
            discovery_start_str, end_date_str,
            min_tier="B-Tier", include_qualifiers=True,
        )
        if not discovered:
            return {
                "standings": pd.DataFrame(), "match_h2h": {},
                "cutoff": today, "source": "fallback", "match_count": 0,
                "error": "No tournaments found in Liquipedia for this date range.",
            }

        tournament_slugs = [d["slug"] for d in discovered]

        if progress_callback:
            progress_callback(2, 3, f"Fetching {len(tournament_slugs)} tournaments…")

        # Cache is keyed on (window, slugs, snapshot_cutoff) — the cutoff is
        # part of the key because it changes per-event fetch behaviour
        # (unfinished events are fetched whole vs. windowed).
        cache_args = (start_date_str, end_date_str, tournament_slugs)

        # Serve from cache when fresh (< 2 hours old)
        if liquipedia_cache_exists(*cache_args, snapshot_cutoff=snapshot_cutoff):
            mtime = liquipedia_cache_mtime(*cache_args, snapshot_cutoff=snapshot_cutoff)
            if (datetime.now() - mtime).total_seconds() / 60 < 120:
                if progress_callback:
                    progress_callback(2, 3, f"Loading {len(tournament_slugs)} tournaments from cache…")
                liq_df = load_liquipedia_from_cache(
                    *cache_args, snapshot_cutoff=snapshot_cutoff,
                )
            else:
                liq_df = fetch_liquipedia_matches(
                    start_date_str, end_date_str,
                    tournament_slugs=tournament_slugs, force_refresh=True,
                    progress_callback=progress_callback,
                    snapshot_cutoff=snapshot_cutoff,
                )
        else:
            liq_df = fetch_liquipedia_matches(
                start_date_str, end_date_str,
                tournament_slugs=tournament_slugs, force_refresh=False,
                progress_callback=progress_callback,
                snapshot_cutoff=snapshot_cutoff,
            )

        if liq_df.empty:
            return {
                "standings": pd.DataFrame(), "match_h2h": {},
                "cutoff": today, "source": "fallback", "match_count": 0,
                "error": "Liquipedia fetch returned no matches.",
            }

        # Roster-split dedup happens upstream in data_loaders/github_loader.py
        # (`_team_best` keeps only the best-ranked roster per team name), so the
        # tmh/bpm dicts arriving here already target the active roster. Liquipedia
        # rows reference team names only (no roster version), so they land on the
        # active roster naturally.
        new_match_count = int((liq_df["loser"].astype(str) != "").sum())

        # ── Step 2: Build Store and run engine ────────────────────────
        if progress_callback:
            progress_callback(3, 3, "Running full VRS computation…")

        store = Store.from_valve(team_match_history, bo_prizes_map)
        store.append_liquipedia(liq_df)

        engine_result = run_vrs(store, cutoff=today)
        new_standings = engine_result["standings"]
        match_h2h     = engine_result["match_h2h"]

        # Reconstruct combined team_match_history from store.matches_df so the
        # Team Breakdown page can display Valve + Liquipedia matches together.
        combined_tmh: dict = {}
        for _, row in store.matches_df.iterrows():
            ev_w = event_stakes(float(row["prize_pool"]))
            for perspective, team, opponent in [
                ("W", row["winner"], row["loser"]),
                ("L", row["loser"],  row["winner"]),
            ]:
                combined_tmh.setdefault(str(team), []).append({
                    "result":     perspective,
                    "opponent":   str(opponent),
                    "date":       row["date"],
                    "match_id":   int(row["match_id"]),
                    "ev_w":       ev_w,
                    "age_w":      0.0,    # recomputed dynamically in updated mode
                    "prize_pool": float(row["prize_pool"]),
                    "is_lan":     bool(row["is_lan"]),
                    "opp_bo":     0.0,
                    "opp_on":     0.0,
                    "h2h_adj":    0.0,    # recomputed dynamically via sim_match_h2h
                })

        # Reconstruct combined bo_prizes_map from store.prizes_df.
        combined_bpm: dict = {}
        for _, row in store.prizes_df.iterrows():
            combined_bpm.setdefault(str(row["team"]), []).append({
                "event_date":   row["date"].strftime("%Y-%m-%d"),
                "prize_won":    float(row["amount"]),
                "age_weight":   0.0,   # recalculated at display time in updated mode
                "scaled_prize": 0.0,
            })

        if new_standings.empty:
            return {
                "standings": pd.DataFrame(), "match_h2h": {},
                "cutoff": today, "source": "fallback",
                "match_count": new_match_count,
                "error": "Full recomputation returned no eligible teams.",
            }

        # ── Step 4: Attach display metadata from snapshot ─────────────
        # region, flag, color are not calculated — they come from Valve's
        # snapshot or team_meta.  Merge here so the rest of app.py works.
        meta_cols = [c for c in ["team", "region", "flag", "color"]
                     if c in snapshot_standings.columns]
        if len(meta_cols) > 1:
            meta_df = (snapshot_standings[meta_cols]
                       .drop_duplicates("team", keep="first"))
            new_standings = new_standings.merge(meta_df, on="team", how="left")
            new_standings["region"] = new_standings["region"].fillna("Global")
            new_standings["flag"]   = new_standings["flag"].fillna("🌍")
            new_standings["color"]  = new_standings["color"].fillna("#58a6ff")

        new_standings = add_regional_rank(new_standings)

        return {
            "standings":          new_standings,
            "match_h2h":          match_h2h,
            "team_match_history": combined_tmh,
            "bo_prizes_map":      combined_bpm,
            "cutoff":             today,
            "source":             "liquipedia",
            "match_count":        new_match_count,
            "error":              None,
        }

    except Exception as e:
        return {
            "standings": pd.DataFrame(), "match_h2h": {},
            "cutoff": today, "source": "fallback", "match_count": 0,
            "error": f"Liquipedia fetch failed: {str(e)[:100]}",
        }

# ══════════════════════════════════════════════════════════════════
# STANDINGS MODE EXECUTION
# ══════════════════════════════════════════════════════════════════

mode_active          = "published"  # "published", "updated", or "fallback"
mode_fetch_error     = None
updated_match_count  = 0
original_standings   = base_standings.copy()
sim_match_h2h        = {}  # match_id → {winner, loser, w_delta, l_delta} from simulation
sim_cutoff_dt        = datetime.now()  # cutoff used for simulation (for age_weight calc)


# ── Engine-computed H2H replay for the selected Valve snapshot ─────
# We re-run our engine on the published snapshot so the Team Breakdown
# H2H table can show per-match Opp Rating / Win % / H2H Δ in *Published*
# mode too.  Cached by snapshot date so it only runs once per snapshot.
@st.cache_data(ttl=3600, show_spinner=False)
def _compute_pub_match_h2h(date_str: str | None,
                           year:     str | None) -> dict:
    """Return `match_h2h` dict for the given Valve snapshot (cached)."""
    if not date_str:
        return {}
    gd = _cached_load_github_data(date_str, year)
    tmh = gd.get("team_match_history", {})
    bpm = gd.get("bo_prizes_map", {})
    cutoff = gd.get("cutoff_datetime")
    if not tmh or cutoff is None:
        return {}
    try:
        _pub_store = Store.from_valve(tmh, bpm)
        _pub_res   = run_vrs(_pub_store, cutoff=cutoff)
        return _pub_res.get("match_h2h", {}) or {}
    except Exception:
        return {}

pub_match_h2h: dict = _compute_pub_match_h2h(_sel_date, _sel_year)

if standings_mode in ("📡 Updated to Today", "🔮 Predicted"):
    # Cache key includes the snapshot date so switching snapshots re-fetches.
    # Predicted mode rides on the same Updated-to-Today fetch (architecture
    # decision #5 — single baseline) and then layers bracket picks on top.
    _cache_key = f"updated_result_{_sel_date}_{_sel_year}"

    if _cache_key not in st.session_state:
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

        st.session_state[_cache_key] = _fetch_result
    else:
        _fetch_result = st.session_state[_cache_key]

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
        sim_match_h2h      = _fetch_result.get("match_h2h", {})
        team_match_history = _fetch_result.get("team_match_history", team_match_history)
        bo_prizes_map      = _fetch_result.get("bo_prizes_map",      bo_prizes_map)
        mode_active = "updated"

        # Compute rank deltas
        _orig_rank_map = original_standings.sort_values("rank").drop_duplicates("team", keep="first").set_index("team")["rank"].to_dict()
        base_standings["rank_delta"] = base_standings.apply(
            lambda r: (int(_orig_rank_map.get(r["team"], r["rank"])) - int(r["rank"])),
            axis=1,
        ).astype(int)

        # ── Predicted layer (Task 6.4) ──────────────────────────────
        # When `🔮 Predicted` is active, append every event's bracket picks to
        # the Updated-to-Today engine input and re-run. The result replaces
        # `base_standings` so every page reflects the simulated future.
        # Architecture decisions #4 (live recompute, memoised) and #13
        # (multi-event chaining via picked-anywhere events).
        predicted_meta = None     # populated when layering succeeds
        if standings_mode == "🔮 Predicted":
            _b_state = st.session_state.get("bracket_state", {}) or {}
            _em, _ep, _included = _collect_predicted_simulation_rows(_b_state)

            if not _em and not _ep:
                st.info(
                    "🔮 **Predicted mode is active but no bracket picks have "
                    "been queued yet.** Open the **🏆 Tournament Predictor** "
                    "page and fill at least one bracket to see standings shift."
                )
                predicted_meta = {"included": [], "matches": 0, "prizes": 0}
            else:
                # Memo: keyed by frozenset of pick tuples + cutoff date.
                # Architecture decision #4 (live recompute + memoise).
                _pred_key = (
                    tuple(sorted(_included)),
                    cutoff_dt.strftime("%Y-%m-%d"),
                    hashlib.md5(
                        repr(sorted(
                            (slug,
                             tuple(sorted(
                                 _b_state[slug]["scenarios"][_b_state[slug]["active_scenario"]]["picks"].items()
                             )))
                            for slug in _included
                        )).encode()
                    ).hexdigest()[:12],
                )
                if "_predicted_engine_cache" not in st.session_state:
                    st.session_state._predicted_engine_cache = {}
                _pcache = st.session_state._predicted_engine_cache

                if _pred_key in _pcache:
                    _pred_result = _pcache[_pred_key]
                else:
                    # Honour the user's Pause toggle on each contributing event.
                    # If ANY contributing event is paused, suppress the recompute
                    # — picks save but standings don't update until unpause.
                    _any_paused = any(
                        _b_state[s].get("paused", False) for s in _included
                    )
                    if _any_paused:
                        _pred_result = None
                        st.info(
                            "⏸️ Predicted recompute paused on at least one "
                            "queued event. Untoggle **Pause** in the Tournament "
                            "Predictor to recompute standings."
                        )
                    else:
                        try:
                            with st.spinner(f"🔮 Recomputing standings with "
                                            f"{len(_em)} match{'es' if len(_em)!=1 else ''} + "
                                            f"{len(_ep)} prize{'s' if len(_ep)!=1 else ''} layered…"):
                                _sim_store = Store.from_valve(team_match_history, bo_prizes_map)
                                _sim_store.append_simulation(
                                    extra_matches=pd.DataFrame(_em) if _em else None,
                                    extra_prizes=_ep if _ep else None,
                                )
                                _pred_result = run_vrs(_sim_store, cutoff=cutoff_dt)
                                _pcache[_pred_key] = _pred_result
                                # LRU trim — keep only 20 most recent
                                if len(_pcache) > 20:
                                    _pcache.pop(next(iter(_pcache)))
                        except Exception as exc:
                            st.error(f"🔴 Predicted recompute failed: {exc}")
                            _pred_result = None

                if _pred_result is not None:
                    _pred_standings = _pred_result["standings"]
                    if not _pred_standings.empty:
                        # Re-attach region/flag/color metadata for display.
                        _meta_cols = [c for c in ("team", "region", "flag", "color")
                                      if c in base_standings.columns]
                        if len(_meta_cols) > 1:
                            _meta_df = (base_standings[_meta_cols]
                                        .drop_duplicates("team", keep="first"))
                            _pred_standings = _pred_standings.merge(
                                _meta_df, on="team", how="left",
                            )
                            _pred_standings["region"] = _pred_standings["region"].fillna("Global")
                            _pred_standings["flag"]   = _pred_standings["flag"].fillna("🌍")
                            _pred_standings["color"]  = _pred_standings["color"].fillna("#58a6ff")

                        # Re-compute rank_delta vs the Updated-to-Today baseline
                        # (not vs Valve published) so the delta isolates the
                        # bracket-pick effect.
                        _upd_rank_map = (base_standings
                                         .sort_values("rank")
                                         .drop_duplicates("team", keep="first")
                                         .set_index("team")["rank"].to_dict())
                        _pred_standings["rank_delta"] = _pred_standings.apply(
                            lambda r: (int(_upd_rank_map.get(r["team"], r["rank"])) - int(r["rank"])),
                            axis=1,
                        ).astype(int)

                        base_standings = _pred_standings
                        sim_match_h2h  = _pred_result.get("match_h2h", {})
                        mode_active    = "predicted"
                        predicted_meta = {
                            "included": _included,
                            "matches":  len(_em),
                            "prizes":   len(_ep),
                            "key":      _pred_key,
                        }
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

elif mode_active == "predicted":
    _pm = predicted_meta or {}
    _events_label = (
        f"{len(_pm.get('included', []))} event{'s' if len(_pm.get('included', [])) != 1 else ''}"
    )
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#3d2d0d,#0d1117);border:1px solid #f0883e;
                border-radius:10px;padding:16px 20px;margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="font-size:22px">🔮</span>
        <span style="font-size:16px;font-weight:700;color:#f0883e;">
          Predicted future · cutoff {cutoff_dt.strftime('%B %d, %Y')}</span>
        <span style="background:#f0883e;color:#000;border-radius:12px;padding:2px 10px;
                     font-size:11px;font-weight:600">{_events_label} queued</span>
      </div>
      <div style="font-size:12px;color:#f0c987;line-height:1.6;">
        Updated-to-Today baseline + your Tournament Predictor picks
        ({_pm.get("matches", 0)} match{'es' if _pm.get("matches", 0) != 1 else ''},
        {_pm.get("prizes", 0)} prize{'s' if _pm.get("prizes", 0) != 1 else ''} layered).
        Edit picks via <strong>🏆 Tournament Predictor</strong> in the sidebar.
      </div>
    </div>""", unsafe_allow_html=True)

elif mode_active == "fallback":
    st.warning(f"⚠️ Using As Published standings (fetch failed: {mode_fetch_error[:80]})")

# ── Sidebar status (after mode state is resolved) ───────────────────
with st.sidebar:
    if mode_active == "updated":
        st.success(f"📡 Updated  ·  {cutoff_dt.strftime('%Y_%m_%d')}")
    elif mode_active == "predicted":
        st.success(f"🔮 Predicted  ·  {cutoff_dt.strftime('%Y_%m_%d')}")
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
    elif mode_active == "predicted":
        _pm = predicted_meta or {}
        st.info(
            f"🔮 **Predicted future** (cutoff {cutoff_dt.strftime('%B %d, %Y')}) — "
            f"baseline + {_pm.get('matches', 0)} layered matches across "
            f"{len(_pm.get('included', []))} event(s)"
        )
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
        st.subheader("⚔️ H2H Adjustment — Best & Worst")

        _h2h_top50 = st.toggle("Limit to Top 50 teams", value=False, key="h2h_top50_filter")
        _h2h_pool = base_standings.head(50) if _h2h_top50 else base_standings
        _h2h_sorted = _h2h_pool.sort_values("h2h_delta", ascending=False)
        _h2h_top5    = _h2h_sorted.head(5)
        _h2h_bottom5 = _h2h_sorted.tail(5).iloc[::-1]  # worst first

        def _h2h_row(rank, team, val):
            color = "#3fb950" if val >= 0 else "#f85149"
            flag  = TEAM_META.get(team, {}).get("flag", "")
            return (
                f'<tr style="border-bottom:1px solid #21262d;">'
                f'<td style="padding:5px 8px;color:#8b949e;font-size:11px">#{rank}</td>'
                f'<td style="padding:5px 8px;font-size:12px">{flag} {team}</td>'
                f'<td style="padding:5px 8px;text-align:right;font-weight:700;color:{color};font-size:13px">{val:+.1f}</td>'
                f'</tr>'
            )

        _top_rows = "".join(
            _h2h_row(int(r["rank"]), r["team"], r["h2h_delta"])
            for _, r in _h2h_top5.iterrows()
        )
        _bot_rows = "".join(
            _h2h_row(int(r["rank"]), r["team"], r["h2h_delta"])
            for _, r in _h2h_bottom5.iterrows()
        )

        _tbl_header = (
            '<thead><tr style="color:#8b949e;font-size:9px;text-transform:uppercase;background:#161b22;">'
            '<th style="padding:6px 8px">Rank</th>'
            '<th style="padding:6px 8px;text-align:left">Team</th>'
            '<th style="padding:6px 8px;text-align:right">H2H Δ</th>'
            '</tr></thead>'
        )

        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:#3fb950;margin-bottom:4px;">🔝 Top 5 gainers</div>'
            '<div style="border:1px solid #30363d;border-radius:8px;overflow:hidden;margin-bottom:14px;">'
            f'<table style="width:100%;border-collapse:collapse;">{_tbl_header}<tbody>{_top_rows}</tbody></table>'
            '</div>'
            '<div style="font-size:11px;font-weight:700;color:#f85149;margin-bottom:4px;">📉 Bottom 5 losers</div>'
            '<div style="border:1px solid #30363d;border-radius:8px;overflow:hidden;">'
            f'<table style="width:100%;border-collapse:collapse;">{_tbl_header}<tbody>{_bot_rows}</tbody></table>'
            '</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# PAGE 2  ·  TOURNAMENT PREDICTOR  (Pillar 3 — bracket simulator)
# ══════════════════════════════════════════════════════════════════
#
# Architectural decisions live in NEXT_STEPS.md > "Pillar 3 — Architecture
# Decisions". Implementation rolls out in three steps:
#   • 6.2 (this block) — event picker + metadata banner.
#   • 6.3              — click-through bracket UI (SE+bronze, DE, Swiss, groups).
#   • 6.4              — placement → engine wiring; exposes 🔮 Predicted mode.
#
# The What-If Predictor (Mode 3 manual hypotheticals) was retired in favour of
# this surface — Tournament Predictor uses the same `Store.append_simulation`
# plumbing under the hood, just with a structured bracket UI instead of free-form
# match entry.

elif page == "🏆 Tournament Predictor":
    from data_loaders import discover_liquipedia_upcoming_events

    st.title("🏆 Tournament Predictor")
    st.markdown(
        "Pick an upcoming tournament, fill the bracket, see how standings would "
        "shift if results played out the way you predict. Once filled, the prediction "
        "propagates app-wide via the **🔮 Predicted** standings mode (lands in step 6.4)."
    )

    # ── Discovery filters ─────────────────────────────────────────
    _now = datetime.now()
    # Default end of range = day before predicted next Valve release. Valve
    # publishes roughly monthly, so we estimate next-release ≈ latest known
    # snapshot + 30 days. Once a new official snapshot lands the prediction
    # itself becomes obsolete, so framing the discovery window as "until the
    # next refresh" matches the user's mental model.
    if _all_dates:
        try:
            _latest_snap = datetime.strptime(_all_dates[0][0], "%Y_%m_%d")
            _predicted_next_release = _latest_snap + timedelta(days=30)
            _default_end = (_predicted_next_release - timedelta(days=1)).date()
        except Exception:
            _default_end = (_now + timedelta(days=30)).date()
    else:
        _default_end = (_now + timedelta(days=30)).date()

    f_left, f_mid, f_right, f_btn = st.columns([1.2, 1.2, 1.2, 1])
    discovery_start = f_left.date_input(
        "🔎 Range start", value=_now.date(),
        key="tp_disc_start",
        help="Earliest event date to surface in the picker.",
    )
    discovery_end = f_mid.date_input(
        "🔎 Range end", value=_default_end,
        key="tp_disc_end",
        help=(
            f"Latest event date to surface. Default = day before predicted "
            f"next Valve release ({_default_end:%b %d, %Y})."
        ),
    )
    min_tier = f_right.selectbox(
        "Min tier", ["S-Tier", "A-Tier", "B-Tier"],
        index=1, key="tp_min_tier",
        help="S = Majors and equivalents · A = top regional · B = mid-tier.",
    )
    f_btn.write("")  # vertical spacer to align the button with date inputs
    refresh = f_btn.button(
        "🔄 Refresh", use_container_width=True,
        help="Bypass the 6h discovery cache and re-walk Liquipedia.",
    )

    # ── Discover events (uses 6.1's 6h JSON cache + Streamlit memo) ───
    @st.cache_data(ttl=21600, show_spinner=False)
    def _cached_upcoming_events(start_iso: str, end_iso: str, tier: str, force_token: int):
        # `force_token` forces a Streamlit cache miss when the user hits Refresh;
        # 6.1's file cache is bypassed via force_refresh=True on the same trigger.
        return discover_liquipedia_upcoming_events(
            start_date=start_iso,
            end_date=end_iso,
            min_tier=tier,
            today=datetime.now(),
            fetch_details=True,
            force_refresh=(force_token > 0),
        )

    if "_tp_force_token" not in st.session_state:
        st.session_state._tp_force_token = 0
    if refresh:
        st.session_state._tp_force_token += 1

    with st.spinner("Loading upcoming events from Liquipedia…"):
        try:
            events = _cached_upcoming_events(
                discovery_start.isoformat(),
                discovery_end.isoformat(),
                min_tier,
                st.session_state._tp_force_token,
            )
        except Exception as exc:
            st.error(f"🔴 Could not load upcoming events: {exc}")
            events = []

    if not events:
        st.warning(
            f"🟡 No upcoming events found between **{discovery_start}** and "
            f"**{discovery_end}** at **{min_tier}** or higher. Try widening the "
            f"range or lowering the tier."
        )
        st.stop()

    # ── Event picker (sorted by start date asc; default = next event) ─
    def _label(ev: dict) -> str:
        sd = ev.get("start_date")
        ed = ev.get("end_date")
        sd_s = sd.strftime("%b %d") if sd else "?"
        ed_s = ed.strftime("%b %d, %Y") if ed else "?"
        pp   = ev.get("prize_pool", 0) or 0
        if pp >= 1_000_000:
            pp_s = f"${pp / 1_000_000:.2f}M"
        elif pp > 0:
            pp_s = f"${pp / 1000:.0f}K"
        else:
            pp_s = "TBA"
        return f"{ev['tier']} · {ev['name']}  ({sd_s} – {ed_s}, {pp_s})"

    event_labels = [_label(e) for e in events]
    sel_idx = st.selectbox(
        "Choose an event",
        list(range(len(events))),
        format_func=lambda i: event_labels[i],
        key="tp_sel_event",
    )
    ev = events[sel_idx]

    st.markdown("---")

    # ── Metadata banner ───────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    pp = ev.get("prize_pool", 0) or 0
    m1.metric("💰 Prize Pool", f"${pp:,.0f}" if pp else "TBA")
    sd, ed = ev.get("start_date"), ev.get("end_date")
    if sd and ed:
        days = (ed - sd).days + 1
        m2.metric("📅 Dates", f"{sd:%b %d} – {ed:%b %d}",
                  delta=f"{days} day{'s' if days != 1 else ''}",
                  delta_color="off")
    else:
        m2.metric("📅 Dates", "TBA")
    m3.metric("🌐 Format", "🖥️ LAN" if ev.get("is_lan") else "🌍 Online")
    m4.metric("🏅 Tier", ev["tier"])

    # ── Format prose (from <h3>Format</h3>) ───────────────────────
    fmt_text = (ev.get("format") or "").strip()
    if fmt_text:
        with st.expander("📋 Format details", expanded=False):
            st.markdown(fmt_text)
    else:
        st.caption("Format details not yet published on Liquipedia.")

    # ── Seeded teams chip cloud ───────────────────────────────────
    seeded = ev.get("seeded_teams", []) or []
    if seeded:
        st.markdown(f"**🎟️ Seeded teams ({len(seeded)})**")
        chips_html = " ".join(
            f"<span style='display:inline-block; padding:4px 10px; margin:3px; "
            f"background:#1f2937; border:1px solid #374151; border-radius:14px; "
            f"font-size:13px; color:#e5e7eb;'>"
            f"{TEAM_META.get(t, {}).get('flag', '')} {t}</span>"
            for t in seeded
        )
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.warning(
            "🟡 No seeded teams listed yet. Bracket fill will be unavailable "
            "until seeding is announced on Liquipedia."
        )

    # ── Prize distribution table ──────────────────────────────────
    pd_dist = ev.get("prize_distribution", {}) or {}
    if pd_dist:
        with st.expander(f"💰 Prize distribution ({len(pd_dist)} placements)", expanded=False):
            pd_rows = [
                {"Placement": place, "Prize (USD)": f"${amt:,.0f}"}
                for place, amt in pd_dist.items()
            ]
            st.dataframe(
                pd.DataFrame(pd_rows),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.caption("Prize breakdown not yet published.")

    # ── Bracket parsing + render ──────────────────────────────────
    # Slice 1 of Task 6.3: SE / SE_with_bronze interactive bracket. Other
    # formats (DE, Swiss, GSL groups, RR groups) surface as 🚧 placeholder
    # cards until later slices land their parsers.
    _fetch_page = _fetch_liquipedia_page  # local alias to keep slice-1 code stable
    _bs4        = _liquipedia_bs4

    st.markdown("---")
    st.markdown("### 🧱 Bracket")

    # Bump this when the parser logic changes — every existing memoised
    # Stage from a prior session is then invalidated automatically without
    # requiring users to hit the Refresh button.
    #   v1 = slice 1 (SE+bronze)
    #   v2 = + slice 2 (DE) and slice 3 (Swiss)
    #   v3 = + slice 4 (RR Groups, with cascade RR completion)
    #   v4 = + Major sub-page traversal (Stage_N)
    #   v5 = + GSL-lite / Full-GSL group format detection (sub-section headers)
    #   v6 = + bye-slot backfill for non-power-of-two SE brackets (bronze fix)
    #   v7 = + Phase 3 stage-graph hard-cut: per-stage roster drives R1 seeds
    #   v8 = + Phase 3 fix: no auto pair-order R1 (Liquipedia DOM or manual only)
    #   v9 = + h4 sub-stage heading (Group A/B distinct labels in DE groups)
    #   v10 = + Phase 4 cross-stage cascade (advance_from / direct_invitees
    #          drive downstream rosters at render time)
    #   v11 = + §10 Swiss rework: VRS-snapshot picker, auto-seed R1 button,
    #          Buchholz pairer with rematch swap (vrs_engine/swiss_pairer.py)
    #   v12 = + Phase 5.3 per-format prize emission: vrs_engine/placement_labels.py
    #          dispatches Swiss/DE/GSL/RR/SE bucket walkers, compute_place_offsets
    #          composes chain offsets, exit-stage tracking for partial-fill
    _PARSER_VERSION = "v12"

    @st.cache_data(ttl=21600, show_spinner=False)
    def _cached_parsed_stages(slug: str, seed_tuple: tuple, force_token: int, parser_version: str):
        # L2 cache (architecture decision #12). Re-fetches the tournament
        # page and parses brackets; 6h TTL aligns with L1 discovery cache.
        # `force_token` lets the user invalidate via the page-level Refresh.
        # `parser_version` invalidates ALL prior memoised stages whenever the
        # parser logic ships a new behaviour (RR cascade, sub-page traversal,
        # …) — users no longer need to know about the Refresh button to get
        # the latest improvements.
        html = _fetch_page(slug)
        if html is None:
            return None
        soup = _bs4(html)
        # Phase 3 wiring: pull the format prose, per-stage invitee partition
        # and (for Major-tier events) sub-page teamcard rosters off the same
        # soup so parse_tournament_brackets can build the cross-stage graph
        # and drive R1 seeds from each stage's resolved roster.
        from data_loaders.liquipedia_loader import (
            _parse_format as _lp_parse_format,
            _parse_per_stage_invites as _lp_per_stage_invites,
        )
        from data_loaders.stage_graph import collect_sub_page_rosters
        prose       = _lp_parse_format(soup)
        per_stage   = _lp_per_stage_invites(soup)
        sub_rosters = collect_sub_page_rosters(slug, soup)
        stages = parse_tournament_brackets(
            soup, list(seed_tuple), slug=slug,
            format_prose=prose,
            direct_invitees_by_stage=per_stage,
            sub_page_rosters=sub_rosters,
        )
        return stages

    with st.spinner("Parsing brackets…"):
        parsed_stages = _cached_parsed_stages(
            ev["slug"], tuple(seeded),
            st.session_state._tp_force_token,
            _PARSER_VERSION,
        )

    if parsed_stages is None:
        st.error("🔴 Could not fetch the tournament page from Liquipedia.")
        st.stop()
    if not parsed_stages:
        st.warning(
            "🟡 No bracket structure found on the Liquipedia page yet. "
            "Bracket fill becomes available once seeding is published."
        )
        st.stop()

    # ── Build a quick rank lookup for Glicko E(win) (architecture #6) ─
    # Auto-fill favorites uses `expected_win(seed_a, seed_b)`; teams not in
    # `base_standings` fall back to SEED_MIN (=400) per architecture #6.
    if not base_standings.empty and "seed" in base_standings.columns:
        _seed_lookup = dict(zip(base_standings["team"], base_standings["seed"]))
    else:
        _seed_lookup = {}

    def _team_seed(team: str | None) -> float:
        if not team:
            return float(SEED_MIN)
        return float(_seed_lookup.get(team, SEED_MIN))

    # ── Session state init (architecture #3 — scenario-aware shape) ───
    if "bracket_state" not in st.session_state:
        st.session_state.bracket_state = {}
    bs_root = st.session_state.bracket_state
    if ev["slug"] not in bs_root:
        bs_root[ev["slug"]] = {
            "scenarios":         {"current": {
                "picks":         {},
                "manual_seeds":  {},   # {match_id: {"a": team, "b": team}}
                "created":       datetime.now(),
            }},
            "active_scenario":   "current",
            "seed_signature":    hashlib.md5(",".join(seeded).encode()).hexdigest()[:12]
                                  if seeded else "",
            "include_in_chain":  False,
            "event_meta":        ev,   # stashed for Predicted-mode lookup from other pages
        }
    else:
        # Refresh event metadata in case Liquipedia data changed (prize pool
        # announcement, seeds finalised, end-date adjusted).
        bs_root[ev["slug"]]["event_meta"] = ev
        # Backfill `manual_seeds` for scenarios created before Phase 3.
        for scen in bs_root[ev["slug"]]["scenarios"].values():
            scen.setdefault("manual_seeds", {})
    slug_state    = bs_root[ev["slug"]]
    active_name   = slug_state["active_scenario"]
    picks: dict   = slug_state["scenarios"][active_name]["picks"]
    manual_seeds: dict = slug_state["scenarios"][active_name]["manual_seeds"]

    # ── VRS-snapshot picker (Swiss seeding authority — §10 spec) ──────
    # User must pick a snapshot before Swiss synth fires. No default
    # (10.Q5). State persists in slug_state["snapshot_date"] +
    # slug_state["snapshot_year"]. Snapshot scope = Swiss seed-tiebreak
    # only; engine factors and auto-fill E(win) keep using the global
    # mode (10.Q9).
    #
    # Date discovery: try the live GitHub API first, fall back to the
    # local pickle cache directory when the API is unreachable / rate-
    # limited (60 reqs/h unauthenticated). Without this fallback the
    # selectbox shows only the placeholder option in offline / capped
    # sessions and the user can't pick at all.
    from data_loaders.github_loader import (
        _find_all_dates as _gh_dates, GH_CACHE_DIR as _GH_CACHE_DIR,
    )
    import os as _os, re as _re_dates
    _snap_dates = _gh_dates()
    if not _snap_dates and _os.path.isdir(_GH_CACHE_DIR):
        seen: set[tuple[str, str]] = set()
        for fn in _os.listdir(_GH_CACHE_DIR):
            m = _re_dates.match(r"vrs_(\d{4})_(\d{4}_\d{2}_\d{2})\.pkl$", fn)
            if m:
                year, date_str = m.group(1), m.group(2)
                seen.add((date_str, year))
        _snap_dates = sorted(seen, key=lambda x: x[0], reverse=True)
    _snap_options = ["— Pick a VRS snapshot —"] + [
        f"{d} ({y})" for (d, y) in _snap_dates
    ]
    _cur_date = slug_state.get("snapshot_date")
    _cur_year = slug_state.get("snapshot_year")
    _cur_label = (f"{_cur_date} ({_cur_year})"
                  if _cur_date and _cur_year else _snap_options[0])
    _idx = _snap_options.index(_cur_label) if _cur_label in _snap_options else 0
    _picked = st.selectbox(
        "VRS snapshot for Swiss seeding",
        options=_snap_options,
        index=_idx,
        key=f"tp_snapshot_{ev['slug']}",
        help="Drives seed numbers used for Buchholz tiebreak and the "
             "🌱 auto-seed R1 button on Swiss stages. Affects Swiss "
             "seeding only — engine factors / E(win) come from the "
             "global mode.",
    )
    if _picked == _snap_options[0]:
        slug_state["snapshot_date"] = None
        slug_state["snapshot_year"] = None
    else:
        # "YYYY_MM_DD (YYYY)" → split date + year out
        _d, _y = _picked.rsplit(" (", 1)
        slug_state["snapshot_date"] = _d
        slug_state["snapshot_year"] = _y.rstrip(")")

    snapshot_standings = _snapshot_standings(
        slug_state.get("snapshot_date"),
        slug_state.get("snapshot_year"),
    )
    if not snapshot_standings:
        st.warning(
            "🟠 Pick a VRS snapshot to enable Swiss seeding "
            "(🌱 auto-seed R1 button + R>1 cascade)."
        )
    else:
        # Identify participants missing from the snapshot — they'll get
        # bottom-rank tail seeds (10.Q6) so the user knows.
        all_participants: set[str] = set(seeded)
        for s in parsed_stages:
            all_participants.update(s.roster or [])
            all_participants.update(s.direct_invitees or [])
        _missing = sorted(t for t in all_participants
                          if t and t not in snapshot_standings)
        if _missing:
            st.info(
                f"ℹ️ {len(_missing)} team(s) not in snapshot — "
                f"placed at bottom-rank tail alphabetically: "
                f"{', '.join(_missing[:6])}"
                + (f" + {len(_missing) - 6} more" if len(_missing) > 6 else "")
            )

    # ── Phase 4 fragment wrapper ──────────────────────────────────
    # Wrap cascade + render + toolbar in @st.fragment so per-pick
    # button clicks rerun ONLY this panel, not the whole predictor
    # page (event discovery, metadata banner, format expander, …).
    # Massively cuts per-click latency on Cologne-sized brackets.
    @st.fragment
    def _bracket_panel(parsed_stages=parsed_stages):
        # ── Phase 4 cross-stage cascade (render-time roster propagation) ──
        # Cached `parsed_stages` is shared across reruns and across sessions
        # via the L2 cache, so it must stay immutable. Deep-copy before
        # mutating per-stage `roster` / R1 bye seeds so each render's cascade
        # output doesn't leak into the next user's view.
        #
        # Steps per render:
        #   1. Overlay `manual_seeds` onto match seed slots (so the cascade's
        #      ``compute_stage_advancers`` sees user-placed teams as if
        #      Liquipedia had them).
        #   2. Walk the stage graph topologically — each stage with
        #      ``advance_from`` entrants computes its roster from upstream
        #      picks + concrete `direct_invitees`.
        #   3. Write the computed roster back to each stage and re-fill any
        #      non-power-of-two bracket bye slots so SF byes resolve.
        # Phase 5 §5.1 — single seam. resolve_for_render does the deep-copy +
        # manual_seeds overlay + Swiss-synth bake + cross-stage cascade +
        # downstream R1 seat + bye fill. Same call site that engine emission
        # uses, guaranteeing identical state across UI and prize emission.
        from data_loaders.stage_graph import (
            apply_cross_stage_cascade,
            seat_cross_stage_r1,
            resolve_for_render,
        )
        from data_loaders.bracket_parser import _apply_roster_seeds
        parsed_stages = resolve_for_render(
            parsed_stages, picks,
            manual_seeds=manual_seeds,
            snapshot_standings=snapshot_standings,
        )

        # ── Compact bracket styles (Phase 3 polish) ───────────────────────
        # Shrink the manual-seed selectbox, tighten button typography, pull
        # within-match teams together, and breathe between rounds / matches.
        # Lines must be dedented — Streamlit's markdown parser treats
        # leading-whitespace multi-line content as a code block, which would
        # leak the raw <style> block onto the page as visible text.
        st.markdown(
            "<style>"
            "div[data-testid=\"stSelectbox\"]{min-height:0;}"
            "div[data-testid=\"stSelectbox\"]>div{min-height:1.6rem;font-size:0.78rem;padding:1px 6px;}"
            "div[data-testid=\"stSelectbox\"] svg{height:12px;width:12px;}"
            "div[data-testid=\"stButton\"]>button{padding:2px 8px;font-size:0.78rem;min-height:1.7rem;line-height:1.2;}"
            ".tp-match-pair-anchor+div [data-testid=\"stHorizontalBlock\"]{gap:0.15rem !important;}"
            ".tp-match-pair-anchor{margin-bottom:14px;}"
            "</style>",
            unsafe_allow_html=True,
        )

        # Stage-signature staleness banner (architecture #12 part b).
        sig_now = ":".join(s.signature() for s in parsed_stages)
        sig_key = "_bracket_sig_" + ev["slug"]
        if sig_key in st.session_state and st.session_state[sig_key] != sig_now:
            st.warning(
                "🟠 Liquipedia's bracket structure has changed since you last "
                "edited this prediction. Picks for altered slots are kept but "
                "may now reference different match-ups. Review highlighted slots."
            )
        st.session_state[sig_key] = sig_now

        # ── Resolve teams in any match given current picks ────────────────
        # Pre-compute Swiss cascade pairings once per render — keyed by stage_id.
        _swiss_overrides_cache: dict[str, dict[str, tuple[str, str]]] = {}

        def _resolve(stage: "Stage", m: "BracketMatch") -> tuple[str | None, str | None]:
            """Return (team_a, team_b). For R2+ slots, follow feeders through picks."""
            by_id = {x.match_id: x for x in stage.matches}

            # Swiss cascade: synthesised R2+ pairings for empty matchlist seeds.
            if m.sub == "SW":
                ov = _swiss_overrides_cache.get(stage.stage_id)
                if ov is None:
                    ov = _compute_swiss_overrides(stage, picks, snapshot_standings)
                    _swiss_overrides_cache[stage.stage_id] = ov
                if m.match_id in ov:
                    return ov[m.match_id]

            def _winner_of(mid: str) -> str | None:
                if mid is None:
                    return None
                up = by_id.get(mid)
                if up is None:
                    return None
                if up.played_winner:
                    return up.played_winner
                return picks.get(mid)

            def _loser_of(mid: str) -> str | None:
                if mid is None:
                    return None
                up = by_id.get(mid)
                if up is None:
                    return None
                up_a, up_b = _resolve(stage, up)
                up_winner = _winner_of(mid)
                if up_winner == up_a:
                    return up_b
                if up_winner == up_b:
                    return up_a
                return None

            # R1 slots in SE / UB carry seeds; LB R1 has no seeds — always feeders.
            if m.round_idx == 1 and not m.is_bronze and m.sub in ("", "UB"):
                return m.seed_a, m.seed_b

            # Matchlist-based formats: R1 (and Swiss in general) carry pairings
            # in ``seed_a/b``. GSL-lite / Full-GSL groups encode R2+ Winners' /
            # Elimination / Decider matches as feeder-driven rows just like
            # SE+bronze / DE LB; we let those fall through to the generic
            # feeder-kind walker below.
            if m.sub == "SW":
                return m.seed_a, m.seed_b
            if m.sub == "GR" and m.round_idx == 1:
                return m.seed_a, m.seed_b

            # Generic resolution by feeder kind. SE+bronze, DE LB and DE UB-vs-LB
            # cross-feeders all flow through here.
            def _team(feeder_id: str | None, kind: str) -> str | None:
                if not feeder_id:
                    return None
                return _loser_of(feeder_id) if kind == "loser" else _winner_of(feeder_id)

            # Fall back to direct seed when no feeder (R2+ bye slot for non-
            # power-of-two SE / SE+bronze brackets). The cross-stage cascade
            # `seat_cross_stage_r1` writes top-of-roster teams into SF byes
            # for fan-in 6-team layouts (Atlanta) — the resolver must honour
            # them, otherwise byes render as TBD even when seeded.
            return (
                _team(m.feeder_a, m.feeder_a_kind) or m.seed_a,
                _team(m.feeder_b, m.feeder_b_kind) or m.seed_b,
            )

        # Wrap `_resolve` with the manual-seed overlay (Phase 3 Q5). Manual seeds
        # take priority over upstream-derived teams so the user can override any
        # slot when Liquipedia / cascade hasn't placed one.
        _resolve_inner = _resolve
        def _resolve(stage: "Stage", m: "BracketMatch") -> tuple[str | None, str | None]:
            a, b = _resolve_inner(stage, m)
            ms = manual_seeds.get(m.match_id) or {}
            return ms.get("a") or a, ms.get("b") or b

        # ── Top toolbar: auto-fill / clear / pause / chain flag ───────────
        tb1, tb2, tb3, tb4, tb5 = st.columns([1, 1, 1, 1.4, 1.2])

        def _autofill_stage(stage: "Stage") -> int:
            """Greedy auto-fill in round order. Returns count of picks set."""
            n = 0
            # Sort so dependencies resolve first:
            #   1. UB before LB (LB R1 feeds from UB R1 losers).
            #   2. Round asc within sub-bracket.
            #   3. Slot asc, bronze/GF last.
            _sub_order = {"UB": 0, "": 0, "LB": 1, "GF": 2, "RST": 3}
            ordered = sorted(stage.matches,
                             key=lambda x: (
                                 x.round_idx if x.sub != "LB" else x.round_idx + 0.5,
                                 _sub_order.get(x.sub, 9),
                                 x.is_bronze,
                                 x.slot_idx,
                             ))
            for m in ordered:
                if m.played_winner:
                    continue
                a, b = _resolve(stage, m)
                if a is None and b is None:
                    continue
                if a is None:
                    picks[m.match_id] = b; n += 1; continue
                if b is None:
                    picks[m.match_id] = a; n += 1; continue
                ew = expected_win(_team_seed(a), _team_seed(b))
                if ew == 0.5:
                    picks[m.match_id] = a if a < b else b   # alphabetical tie-break
                else:
                    picks[m.match_id] = a if ew >= 0.5 else b
                n += 1
            return n

        if tb1.button("⚡ Auto-fill favorites", use_container_width=True,
                      help="Fill every empty slot with the higher-rated team (Glicko E(win))."):
            total = 0
            for s in parsed_stages:
                if s.format in ("SE", "SE_with_bronze", "DE", "Swiss", "Groups"):
                    total += _autofill_stage(s)
                    # After this stage's picks land:
                    #   1. Re-bake Swiss synth pairings into seed slots (the
                    #      Buchholz pairer derives R2-R5 from the just-set
                    #      R1 picks; cascade's top_by_wins tally needs them
                    #      to count wins from later rounds).
                    #   2. Re-run cascade so downstream R1 slots seat from
                    #      the new advancer list BEFORE autofill reaches the
                    #      next stage. Without (1)+(2) the Playoffs stays
                    #      TBD even after a fully-picked Swiss.
                    for _stg in parsed_stages:
                        if _stg.format == "Swiss":
                            _ov = _compute_swiss_overrides(
                                _stg, picks, snapshot_standings,
                            )
                            for _m in _stg.matches:
                                if _m.match_id in _ov and not _m.seed_a:
                                    _m.seed_a, _m.seed_b = _ov[_m.match_id]
                    _re = apply_cross_stage_cascade(parsed_stages, picks)
                    for _stg in parsed_stages:
                        _r = _re.get(_stg.def_name)
                        if not _r:
                            continue
                        # Partial-roster guard: when a downstream stage
                        # has multiple upstream sources (Atlanta fan-in:
                        # 2 groups × 3 advancers), a single group's
                        # autofill produces a partial roster. Seating a
                        # partial roster bakes wrong teams into byes
                        # (e.g. SF1.seed_b = top of Group A roster) that
                        # the next pass's fan-in seeding can't overwrite.
                        # Skip until cascade returns the full expected
                        # team count.
                        _expected = sum(en.count for en in _stg.entrants
                                        if en.source in ("advance_from",
                                                         "direct_invite"))
                        if _expected and len(_r) < _expected:
                            continue
                        _stg.roster = _r
                        seat_cross_stage_r1(_stg, _r)
                        _apply_roster_seeds(_stg.matches, _r)
            st.toast(f"⚡ Auto-filled {total} pick{'s' if total != 1 else ''}.")
            st.rerun(scope="fragment")

        if tb2.button("🗑️ Clear picks", use_container_width=True):
            picks.clear()
            manual_seeds.clear()
            st.rerun(scope="fragment")

        paused = tb3.toggle(
            "⏸️ Pause", value=slug_state.get("paused", False),
            key=f"tp_pause_{ev['slug']}",
            help="When on, picks save but the engine doesn't auto-recompute on every click. (6.4)",
        )
        slug_state["paused"] = paused

        chain_flag = tb4.toggle(
            "🔗 Include in multi-event chain",
            value=slug_state.get("include_in_chain", False),
            key=f"tp_chain_{ev['slug']}",
            help="(7.4) Stack this event's picks with other chain-flagged events when computing standings.",
        )
        slug_state["include_in_chain"] = chain_flag

        _RENDERED_FMTS = ("SE", "SE_with_bronze", "DE", "Swiss", "Groups")
        n_total  = sum(len(s.matches) for s in parsed_stages if s.format in _RENDERED_FMTS)
        n_picked = sum(1 for s in parsed_stages
                       if s.format in _RENDERED_FMTS
                       for m in s.matches if (m.played_winner or picks.get(m.match_id)))
        tb5.metric("Picks", f"{n_picked} / {n_total}", label_visibility="visible")

        # ── Render each stage ─────────────────────────────────────────────
        def _render_se_stage(stage: "Stage"):
            st.markdown(f"#### {stage.display_heading or 'Bracket'} "
                        f"<span style='color:#8b949e; font-size:13px'>"
                        f"({stage.format.replace('_', ' ').replace('SE with bronze', 'SE + bronze')}, "
                        f"{stage.rounds} rounds, {len(stage.matches)} matches)</span>",
                        unsafe_allow_html=True)

            rounds_max = stage.rounds
            round_titles = {
                1: "Round 1", rounds_max: "Final",
                rounds_max - 1: "Semifinals", rounds_max - 2: "Quarterfinals",
            }
            # Re-label edge cases
            if rounds_max == 2:
                round_titles = {1: "Semifinals", 2: "Final"}
            elif rounds_max == 1:
                round_titles = {1: "Final"}

            col_count = rounds_max + (1 if any(m.is_bronze for m in stage.matches) else 0)
            cols = st.columns(col_count, gap="large")

            # Group matches by render column
            for r in range(1, rounds_max + 1):
                with cols[r - 1]:
                    st.caption(round_titles.get(r, f"Round {r}"))
                    round_matches = sorted(
                        [m for m in stage.matches if m.round_idx == r and not m.is_bronze],
                        key=lambda x: x.slot_idx,
                    )
                    for m in round_matches:
                        _render_match(stage, m)

            bronze = [m for m in stage.matches if m.is_bronze]
            if bronze:
                with cols[-1]:
                    st.caption("🥉 Bronze")
                    # Bronze sits visually below the final — pad with a spacer
                    st.markdown("<div style='height: 36px'></div>", unsafe_allow_html=True)
                    for m in bronze:
                        _render_match(stage, m)

        # Pool of teams a user can manually drop into a TBD slot. Phase 3 Q5
        # says "available teams = upstream advancers not yet placed + direct
        # invitees + 'Other…' escape hatch". Until the cascade lands (Phase 4)
        # we don't know advancers — so we offer the union of every stage's
        # roster + direct_invitees + the event-level seeded list. Teams already
        # placed (via played_winner / picks / manual_seeds) are still allowed
        # so the user can correct mistakes; sorting is alphabetic for findability.
        def _manual_seed_pool(stage: "Stage | None" = None) -> list[str]:
            """Available teams for a manual-seed dropdown.

            Per-stage when ``stage`` given: cascade-resolved roster + concrete
            direct invitees. This is what Phase 3 Q5 actually meant — for a
            cross-stage Playoffs R1 TBD, the dropdown should show the 8
            upstream advancers, not every team in the tournament.

            Falls back to the global union when the per-stage pool is empty
            (e.g. snapshot not yet picked, upstream not yet decided), so the
            user is never locked out of placing a team manually.
            """
            if stage is not None:
                local: set[str] = set()
                local.update(stage.roster or [])
                local.update(stage.direct_invitees or [])
                if local:
                    return sorted(local)
            pool: set[str] = set(seeded)
            for s in parsed_stages:
                pool.update(s.roster)
                pool.update(s.direct_invitees)
            return sorted(pool)

        def _is_manual_seedable(m: "BracketMatch", side: str = "a") -> bool:
            """Per-side check: should the dropdown render for THIS side of m?
            Yes when:
              * R1 leaf in a single-tree bracket (whole match is seedable).
              * R2+ bye slot — *this* side has no upstream feeder (the other
                side typically does in non-power-of-two SE brackets).
            Bronze and feeder-driven slots stay disabled."""
            if m.is_bronze:
                return False
            if m.sub not in ("", "UB", "SW", "GR"):
                return False
            if m.round_idx == 1:
                return True
            # R2+: only seedable if THIS side has no feeder (true bye).
            return not (m.feeder_a if side == "a" else m.feeder_b)

        def _render_manual_seed_box(stage: "Stage", m: "BracketMatch", side: str) -> None:
            """Render a tiny selectbox letting the user place a team into a
            TBD slot. Selection persists in `manual_seeds[match_id][side]`."""
            pool = _manual_seed_pool(stage)
            if not pool:
                st.caption("TBD")
                return
            cur = (manual_seeds.get(m.match_id) or {}).get(side, "")
            opts = ["— TBD —"] + pool
            idx = opts.index(cur) if cur in opts else 0
            sel = st.selectbox(
                label="seed",
                options=opts,
                index=idx,
                key=f"tp_seed_{ev['slug']}_{m.match_id}_{side}",
                label_visibility="collapsed",
            )
            new_val = "" if sel == "— TBD —" else sel
            if new_val != cur:
                manual_seeds.setdefault(m.match_id, {})[side] = new_val
                if not new_val:
                    manual_seeds[m.match_id].pop(side, None)
                    if not manual_seeds[m.match_id]:
                        manual_seeds.pop(m.match_id, None)
                st.rerun(scope="fragment")

        def _render_match(stage: "Stage", m: "BracketMatch"):
            a, b = _resolve(stage, m)
            played = m.played_winner is not None
            chosen = m.played_winner if played else picks.get(m.match_id)

            def _btn_label(team: str | None, is_winner: bool) -> str:
                # Phase 3 polish: country flag dropped from in-bracket labels —
                # the bracket grid is already dense and the flag adds noise
                # without information (region is usually obvious from the team).
                if team is None:
                    return "TBD"
                return team + (" ✓" if is_winner else "")

            # Played: read-only side-by-side chips with "actual result" badge.
            if played:
                st.markdown(
                    f"<div style='border:1px solid #374151; border-radius:6px; "
                    f"padding:6px 8px; margin-bottom:14px; background:#161b22; "
                    f"display:flex; gap:8px; align-items:center;'>"
                    f"<div style='flex:1; color:{'#3fb950' if chosen == a else '#8b949e'}; "
                    f"font-size:12px;'>{_btn_label(a, chosen == a)}</div>"
                    f"<div style='color:#6b7280; font-size:11px;'>vs</div>"
                    f"<div style='flex:1; color:{'#3fb950' if chosen == b else '#8b949e'}; "
                    f"font-size:12px; text-align:right;'>"
                    f"{_btn_label(b, chosen == b)}</div>"
                    f"<div style='color:#6b7280; font-size:10px;'>📜</div></div>",
                    unsafe_allow_html=True,
                )
                return

            # Unplayed: two clickable buttons side-by-side via st.columns. When
            # a side is TBD AND that side accepts a manual seed (R1 leaf or R2+
            # bye), render a dropdown instead so the user can place a team
            # directly (Phase 3 Q5). Feeder-driven sides stay disabled — fix
            # the upstream pick.
            # Anchor div lets the bracket CSS target this pair's column gap +
            # add breathing room below the match.
            st.markdown(
                "<div class='tp-match-pair-anchor'></div>",
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2, gap="small")

            with col_a:
                if a is None and _is_manual_seedable(m, "a"):
                    _render_manual_seed_box(stage, m, "a")
                    b_a = False
                else:
                    b_a = st.button(
                        _btn_label(a, chosen == a),
                        key=f"tp_btn_a_{ev['slug']}_{m.match_id}",
                        use_container_width=True,
                        disabled=(a is None),
                        type="primary" if chosen == a else "secondary",
                    )
            with col_b:
                if b is None and _is_manual_seedable(m, "b"):
                    _render_manual_seed_box(stage, m, "b")
                    b_b = False
                else:
                    b_b = st.button(
                        _btn_label(b, chosen == b),
                        key=f"tp_btn_b_{ev['slug']}_{m.match_id}",
                        use_container_width=True,
                        disabled=(b is None),
                        type="primary" if chosen == b else "secondary",
                    )

            if b_a and a:
                picks[m.match_id] = a
                st.rerun(scope="fragment")
            if b_b and b:
                picks[m.match_id] = b
                st.rerun(scope="fragment")

        def _render_swiss_stage(stage: "Stage"):
            """
            Swiss stage render: one column per round, matches stacked vertically.
            Pool labels (High/Low/Mid) are implicit in match ordering — Liquipedia
            groups by record bucket within each round automatically.
            """
            rounds = sorted({m.round_idx for m in stage.matches})
            n_rounds = max(rounds) if rounds else 0
            st.markdown(f"#### {stage.display_heading or 'Swiss Stage'} "
                        f"<span style='color:#8b949e; font-size:13px'>"
                        f"(Swiss, {len(stage.matches)} matches across {n_rounds} rounds)</span>",
                        unsafe_allow_html=True)

            # 🌱 auto-seed R1 button (spec lock 10.Q1). Greyed when:
            #   * no snapshot picked, OR
            #   * Liquipedia has already placed any R1 seed (DOM truth wins).
            r1_matches = sorted(
                [m for m in stage.matches if m.round_idx == 1 and m.sub == "SW"],
                key=lambda x: x.slot_idx,
            )
            liqui_has_r1 = any(m.seed_a or m.seed_b for m in r1_matches)
            roster = list(stage.roster) if stage.roster else []
            can_seed = (bool(snapshot_standings) and not liqui_has_r1
                        and len(roster) >= 2)
            seed_help = (
                "Synthesizes R1 pairings as #1v#9, #2v#10, …, #N/2v#N using the "
                "picked VRS snapshot rank. Disabled when Liquipedia has already "
                "placed R1 or when no snapshot is selected."
            )
            if st.button("🌱 auto-seed R1", key=f"tp_seed_r1_{stage.stage_id}",
                         disabled=not can_seed, help=seed_help):
                from vrs_engine.swiss_pairer import seed_table, r1_split_bracket
                seeds = seed_table(roster, snapshot_standings)
                r1_pairs = r1_split_bracket(seeds)
                for m, (a, b) in zip(r1_matches, r1_pairs):
                    manual_seeds[m.match_id] = {"a": a, "b": b}
                st.toast(f"🌱 Auto-seeded {len(r1_pairs)} R1 pairings for "
                         f"{stage.display_heading or 'Swiss stage'}.")
                st.rerun(scope="fragment")

            cols = st.columns(max(n_rounds, 1), gap="large")
            for r in rounds:
                with cols[r - 1]:
                    round_matches = sorted(
                        [m for m in stage.matches if m.round_idx == r],
                        key=lambda x: x.slot_idx,
                    )
                    played = sum(1 for m in round_matches if m.played_winner)
                    st.caption(f"Round {r}  ·  {len(round_matches)} match"
                               f"{'es' if len(round_matches) != 1 else ''}"
                               f"{f' ({played} played)' if played else ''}")
                    for m in round_matches:
                        _render_match(stage, m)

        def _render_groups_stage(stage: "Stage"):
            """
            Round-robin group stage render: vertical match list per group. Stage
            already represents one group (display_heading carries the label like
            "Group A"). Multiple groups produce multiple Stage objects.
            """
            st.markdown(f"#### {stage.display_heading or 'Group'} "
                        f"<span style='color:#8b949e; font-size:13px'>"
                        f"(Round-robin, {len(stage.matches)} matches)</span>",
                        unsafe_allow_html=True)

            if not stage.matches:
                st.caption("Schedule not yet published.")
                return

            # Compute current standings within this group from picks + played.
            wins:   dict[str, int] = {}
            played: dict[str, int] = {}
            for m in stage.matches:
                chosen = m.played_winner or picks.get(m.match_id)
                a, b   = m.seed_a, m.seed_b
                if not chosen or not a or not b:
                    continue
                wins[chosen] = wins.get(chosen, 0) + 1
                played[a]    = played.get(a, 0) + 1
                played[b]    = played.get(b, 0) + 1
                loser = b if chosen == a else a
                wins.setdefault(loser, 0)

            # Two-column layout: matches on the left, current standings on the right.
            col_m, col_s = st.columns([2, 1])
            with col_m:
                for m in stage.matches:
                    _render_match(stage, m)

            with col_s:
                if wins:
                    st.caption("Group standings")
                    ranks = sorted(wins.items(), key=lambda kv: (-kv[1], kv[0]))
                    std_html = "<table style='width:100%; font-size:12px;'>"
                    for i, (team, w) in enumerate(ranks, start=1):
                        pl = played.get(team, 0)
                        flag = TEAM_META.get(team, {}).get("flag", "")
                        std_html += (
                            f"<tr><td style='padding:2px 4px; color:#8b949e;'>{i}</td>"
                            f"<td style='padding:2px 4px;'>{flag} {team}</td>"
                            f"<td style='padding:2px 4px; text-align:right; "
                            f"color:#3fb950;'>{w}-{pl - w}</td></tr>"
                        )
                    std_html += "</table>"
                    st.markdown(std_html, unsafe_allow_html=True)
                else:
                    st.caption("(Pick matches to see standings)")

        def _render_de_stage(stage: "Stage"):
            """
            Render a double-elimination bracket. Compact 8-team group format
            (used by IEM Atlanta / CS Asia Champs group stage) renders as two
            stacked rows: UB tree on top, LB tree below. Full DE-with-GF +
            bracket-reset extension lives in slice 2b.
            """
            st.markdown(f"#### {stage.display_heading or 'Bracket'} "
                        f"<span style='color:#8b949e; font-size:13px'>"
                        f"(Double-elim, {len(stage.matches)} matches "
                        f"— UB {sum(1 for m in stage.matches if m.sub == 'UB')}, "
                        f"LB {sum(1 for m in stage.matches if m.sub == 'LB')})</span>",
                        unsafe_allow_html=True)

            ub_matches = [m for m in stage.matches if m.sub == "UB"]
            lb_matches = [m for m in stage.matches if m.sub == "LB"]

            # Upper bracket
            if ub_matches:
                ub_max = max(m.round_idx for m in ub_matches)
                st.markdown(
                    "<div style='color:#3fb950; font-size:12px; font-weight:600; "
                    "margin:6px 0;'>⬆️ Upper Bracket</div>",
                    unsafe_allow_html=True,
                )
                cols_ub = st.columns(ub_max, gap="large")
                for r in range(1, ub_max + 1):
                    with cols_ub[r - 1]:
                        title = "UB Final" if r == ub_max else (
                            "UB Semifinals" if r == ub_max - 1 else f"UB R{r}"
                        )
                        st.caption(title)
                        rms = sorted([m for m in ub_matches if m.round_idx == r],
                                     key=lambda x: x.slot_idx)
                        for m in rms:
                            _render_match(stage, m)

            # Lower bracket
            if lb_matches:
                lb_max = max(m.round_idx for m in lb_matches)
                st.markdown(
                    "<div style='color:#f85149; font-size:12px; font-weight:600; "
                    "margin:12px 0 6px 0;'>⬇️ Lower Bracket</div>",
                    unsafe_allow_html=True,
                )
                cols_lb = st.columns(lb_max, gap="large")
                for r in range(1, lb_max + 1):
                    with cols_lb[r - 1]:
                        title = "LB Final" if r == lb_max else (
                            "LB Semifinals" if r == lb_max - 1 else f"LB R{r}"
                        )
                        st.caption(title)
                        rms = sorted([m for m in lb_matches if m.round_idx == r],
                                     key=lambda x: x.slot_idx)
                        for m in rms:
                            _render_match(stage, m)

        # ── Iterate stages in document order ──────────────────────────────
        rendered_any = False
        for stage in parsed_stages:
            st.markdown("---")
            if stage.format in ("SE", "SE_with_bronze"):
                _render_se_stage(stage)
                rendered_any = True
            elif stage.format == "DE":
                _render_de_stage(stage)
                rendered_any = True
            elif stage.format == "Swiss":
                _render_swiss_stage(stage)
                rendered_any = True
            elif stage.format == "Groups":
                _render_groups_stage(stage)
                rendered_any = True
            else:
                st.info(
                    f"**{stage.display_heading or stage.format}** — "
                    f"🚧 Format `{stage.format}` not yet supported. "
                    f"This event's other stages still contribute picks normally."
                )

        if not rendered_any:
            st.warning(
                "🟡 This event has only unsupported stages. Bracket support for "
                "this format lands in a future slice."
            )

        # Liquipedia source link (kept at the bottom of the page).
        st.caption(f"📡 Source: [Liquipedia / {ev['slug']}]({ev['url']})")

    _bracket_panel()


# ══════════════════════════════════════════════════════════════════
# PAGE 3  ·  HOW VRS WORKS  (Explainer — v3, formula-verified)
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

    # ── Section selector — 4 grouped columns ─────────────────────
    # Grouped into logical categories, rendered as columns of buttons.
    # Clicking any button sets vrs_section in session_state (persists
    # across reruns so interactive widgets don't lose your place).
    _vrs_groups = [
        ("🔖 General", [
            ("👋 Welcome",         "welcome"),
            ("🏗️ Architecture",    "arch"),
        ]),
        ("📐 Math Support", [
            ("⏳ Age Weight",      "age"),
            ("🎪 Event Weight",    "evw"),
            ("📐 Curve f(x)",      "curve"),
        ]),
        ("🎯 4 VRS Factors", [
            ("🏆 Bounty Offered",  "bo2"),
            ("💰 Bounty Collected","bc2"),
            ("🕸️ Opp. Network",    "on2"),
            ("🖥️ LAN Wins",        "lan2"),
        ]),
        ("🏁 Final Calculation", [
            ("🌱 → Factor Score",  "seed"),
            ("⚔️ H2H → Final",    "h2h"),
        ]),
    ]
    # Flat maps for key ↔ label lookup
    _vrs_all_keys   = [k for _, grp in _vrs_groups for _, k in grp]
    _vrs_all_labels = [l for _, grp in _vrs_groups for l, _ in grp]
    _key_to_label   = dict(zip(_vrs_all_keys, _vrs_all_labels))

    # Deep-link from factor CTAs in Team Breakdown
    _lf = st.session_state.pop("learn_focus", None)
    _lf_target = {"bo": "bo2", "bc": "bc2", "on": "on2", "lan": "lan2"}.get(_lf)
    if _lf_target and _lf_target in _vrs_all_keys:
        st.session_state["vrs_section"] = _lf_target

    # Default / validate active key
    active_tab = st.session_state.get("vrs_section", "welcome")
    if active_tab not in _vrs_all_keys:
        active_tab = "welcome"

    # Render 4 group columns
    _g_cols = st.columns(4)
    for _gi, (_g_title, _g_items) in enumerate(_vrs_groups):
        with _g_cols[_gi]:
            st.markdown(
                f'<div style="font-size:10px;font-weight:700;color:#8b949e;'
                f'text-transform:uppercase;letter-spacing:0.05em;'
                f'margin-bottom:4px;padding-left:2px;">{_g_title}</div>',
                unsafe_allow_html=True,
            )
            for _lbl, _key in _g_items:
                _is_active = (active_tab == _key)
                if st.button(
                    _lbl,
                    key=f"vrs_nav_{_key}",
                    use_container_width=True,
                    type="primary" if _is_active else "secondary",
                ):
                    st.session_state["vrs_section"] = _key
                    st.rerun()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════
    if active_tab == "welcome":
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
    if active_tab == "arch":
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
            '<div style="background:#161b22;border-left:3px solid #79c0ff;border-radius:6px;padding:10px;text-align:center;"><div style="font-size:12px;font-weight:700;color:#79c0ff;">🕸️ Opp. Network</div><div style="font-size:10px;color:#8b949e;margin-top:2px;">Top-10 opponents\' network depth</div></div>'
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
    if active_tab == "evw":
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
    if active_tab == "age":
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
    if active_tab == "curve":
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
    if active_tab == "bo2":
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

        @st.fragment
        def _bo_sandbox():
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
                        # no st.rerun() — fragment reruns automatically on button click

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

        with col_r:
            _bo_sandbox()

    # ══════════════════════════════════════════════════════════════
    if active_tab == "bc2":
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

        @st.fragment
        def _bc_sandbox():
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
                        # no st.rerun() — fragment reruns automatically on button click

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
                    _bc_pre  = sum(x["entry"] for x in _top10) / len(_top10) if _top10 else 0.0
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
                    f'<span style="font-size:12px;color:#8b949e;">② BC_pre = &#x3A3; / {len(_r["top10"])}</span>'
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

        with col_r:
            _bc_sandbox()

    # ══════════════════════════════════════════════════════════════
    if active_tab == "on2":
        st.subheader("🕸️ Factor 3 — Opponent Network")

        # ── Sandbox preset + session state ──────────────────────
        _ON_PRESET_TEAMS = ["Vitality", "FaZe", "MOUZ", "Mongolz", "Spirit"]
        _ON_PRESET_WINS  = [
            ("Vitality","FaZe"),("Vitality","MOUZ"),("Vitality","Mongolz"),("Vitality","Spirit"),
            ("FaZe","MOUZ"),("FaZe","Mongolz"),("FaZe","Spirit"),
            ("MOUZ","Mongolz"),("MOUZ","Spirit"),
            ("Mongolz","Spirit"),
            ("Spirit","Mongolz"),   # one upset → every team has ≥1 win
        ]
        _ON_STRENGTHS = [1.00, 0.85, 0.70, 0.55, 0.40]   # used only by Randomise

        if "on_net_teams" not in st.session_state:
            st.session_state.on_net_teams = _ON_PRESET_TEAMS[:]
        if "on_net_wins" not in st.session_state:
            st.session_state.on_net_wins = list(_ON_PRESET_WINS)
        if "on_net_focus" not in st.session_state:
            st.session_state.on_net_focus = "Vitality"
        if "on_net_outlier_rank" not in st.session_state:
            st.session_state.on_net_outlier_rank = 2

        # ── Valve's two-pass math, specialised for the sandbox ──
        #   sandbox fixes age_w = ev_w = 1, so:
        #     • distinct_def = count of distinct opponents beaten
        #     • each Pass-2 entry = opp.ownNetwork
        def _on_sandbox_compute(teams, wins, outlier_rank):
            distinct_opps = {t: set() for t in teams}
            for (w, l) in wins:
                if w in distinct_opps and l in distinct_opps:
                    distinct_opps[w].add(l)
            distinct_def = {t: len(s) for t, s in distinct_opps.items()}
            sorted_vals = sorted(distinct_def.values(), reverse=True)
            rk = min(max(1, outlier_rank), max(1, len(sorted_vals))) - 1
            ref_raw = sorted_vals[rk] if sorted_vals else 0
            ref = max(ref_raw, 1)
            own = {t: min(distinct_def[t] / ref, 1.0) for t in teams}
            return distinct_def, own, ref

        def _on_sandbox_entries(focus, wins, own):
            entries = [(l, own.get(l, 0.0)) for (w, l) in wins if w == focus]
            entries.sort(key=lambda x: -x[1])
            return entries

        col_r, col_l = st.columns([3, 2])

        # ═════════════════════════════════════════════════════════
        # RIGHT COLUMN — Explanation + Flowchart
        # ═════════════════════════════════════════════════════════
        with col_l:
            st.markdown(
                '<div style="background:#0d0d1a;border:1px solid #484f58;border-radius:8px;'
                'padding:10px 16px;margin-bottom:12px;font-size:12px;color:#c9d1d9;">'
                '<strong style="color:#79c0ff;">Why does this factor exist?</strong> '
                'Opponent Network rewards teams for beating <em>well-connected</em> opponents — '
                'teams who themselves have recent wins against a deep pool of distinct rivals. '
                'Credit flows entirely from the match graph, not prize money, so prestige emerges '
                'from who you beat and who your opponents beat.'
                '</div>',
                unsafe_allow_html=True)

            # ── Key insight + deeper explainer (above flowchart) ──
            st.markdown(
                '<div style="background:#12111f;border:1px solid #7c3aed;border-left:4px solid #d2a8ff;'
                'border-radius:8px;padding:14px 18px;margin-bottom:14px;">'

                '<div style="font-size:13px;font-weight:700;color:#d2a8ff;margin-bottom:10px;">'
                '🎯 What does it take to maximize your ON factor?'
                '</div>'

                '<div style="font-size:12px;color:#c9d1d9;line-height:1.7;margin-bottom:10px;">'
                'Each win contributes <code>opp.ownNetwork × age_w × ev_w</code> to your Pass 2 pool. '
                'Recency and event prestige help, but the critical lever is '
                '<strong style="color:#79c0ff;">opp.ownNetwork</strong> — how broad your opponent\'s '
                'own win portfolio is. That score is set entirely in Pass 1, before you come into the picture.'
                '</div>'

                '<div style="font-size:12px;color:#c9d1d9;line-height:1.7;margin-bottom:12px;">'
                'The chain reaction: '
                '<span style="color:#58a6ff;">your opponents beat many distinct rivals</span> → '
                'their <code>distinct_def</code> is high → their <code>ownNetwork</code> → 1 → '
                '<span style="color:#d2a8ff;">your wins over them score high</span> → '
                'your <code>on_factor</code> rises.'
                '</div>'

                '<div style="background:#0d1117;border-radius:6px;padding:10px 14px;margin-bottom:10px;'
                'font-size:12px;color:#8b949e;line-height:1.65;">'
                '<strong style="color:#c9d1d9;">Example —</strong> '
                'Team A beats 8 inactive teams, all with <code>ownNetwork ≈ 0</code>. '
                'Their <code>on_factor ≈ 0</code> despite 8 wins.<br>'
                'Team B beats FaZe, Vitality and MOUZ — each with <code>ownNetwork = 1.0</code>. '
                'Their <code>on_factor = 3.0 / 10 = 0.30</code> from just 3 wins.<br>'
                '<strong style="color:#79c0ff;">Wins against well-connected opponents count far more than volume against weak ones.</strong>'
                '</div>'

                '<div style="font-size:12px;color:#c9d1d9;line-height:1.6;">'
                '🔑 <strong>Key insight:</strong> '
                'Pass 1 rewards <strong>breadth</strong> — how many distinct opponents a team has beaten recently. '
                'Pass 2 rewards <strong>depth</strong> — the top-10 mean of your opponents\' breadth. '
                'Prize money only enters via event stakes in Pass 2, never in Pass 1.'
                '</div>'

                '</div>',
                unsafe_allow_html=True)

            _on_fc = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:8px 0;">'
                '<div style="text-align:center;margin-bottom:14px;">'
                '<span style="font-size:13px;font-weight:700;color:#79c0ff;">🕸️ Opponent Network — How is it calculated?</span>'
                '</div>'
                # Input
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:8px 22px;">'
                '<div style="font-size:13px;font-weight:700;color:#c9d1d9;">All wins in the last 180 days</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:2px;">📅 Each match weighted by recency + event stakes</div>'
                '</div></div>'
                # ── Pass 1 banner ──
                '<div style="margin:14px 12px 6px;background:#0d1a2e;border-left:3px solid #58a6ff;border-radius:6px;padding:6px 12px;">'
                '<span style="font-size:11px;font-weight:700;color:#58a6ff;letter-spacing:1px;">PASS 1</span>'
                '<span style="font-size:11px;color:#8b949e;margin-left:8px;">run for EVERY team with ≥1 win</span>'
                '</div>'
                # Step 1: distinct_def
                '<div style="background:#0d0d1a;border:1px solid #79c0ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#79c0ff;margin-bottom:6px;">Step 1 — Sum max(age_w) over distinct opponents defeated</div>'
                '<div style="font-size:12px;color:#c9d1d9;margin-bottom:6px;">Each distinct opponent contributes ONE entry: the age_w of your <em>most recent</em> win against them. Beating the same team 3× does NOT triple-count here — only the freshest date matters.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#060610;border-radius:4px;padding:5px 8px;">'
                'distinct_def[T] = Σ over distinct opponents of max(age_w)'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                # Step 2: normalise + clamp
                '<div style="background:#0d0d1a;border:1px solid #79c0ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#79c0ff;margin-bottom:6px;">Step 2 — Divide by the 5th-highest across ALL teams, clamp to 1</div>'
                '<div style="font-size:12px;color:#c9d1d9;margin-bottom:6px;">The 5th-highest <code>distinct_def</code> across every team is the reference (<code>setOutlierCount(5)</code>). Your <code>ownNetwork</code> is your share of that reference, capped at 1.0.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#060610;border-radius:4px;padding:5px 8px;">'
                'ownNetwork[T] = min(distinct_def[T] / ref, 1.0)'
                '</div></div>'
                # ── Pass 2 banner ──
                '<div style="margin:18px 12px 6px;background:#1a0d2e;border-left:3px solid #d2a8ff;border-radius:6px;padding:6px 12px;">'
                '<span style="font-size:11px;font-weight:700;color:#d2a8ff;letter-spacing:1px;">PASS 2</span>'
                '<span style="font-size:11px;color:#8b949e;margin-left:8px;">run only for ELIGIBLE teams (≥5 matches)</span>'
                '</div>'
                # Step 3: score wins
                '<div style="background:#1a0d2e;border:1px solid #d2a8ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#d2a8ff;margin-bottom:6px;">Step 3 — Score each win using the opponent\'s ownNetwork</div>'
                '<div style="font-size:12px;color:#c9d1d9;margin-bottom:6px;">For every win (not deduped — all of them), the entry equals the <strong>opponent\'s Pass-1 ownNetwork</strong> scaled by the match\'s age weight and event stakes.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#0a0610;border-radius:4px;padding:5px 8px;">'
                'entry = opp.ownNetwork × age_w × ev_w'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                # Step 4: top-10 mean, no curve
                '<div style="background:#1a0d2e;border:1px solid #d2a8ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#d2a8ff;margin-bottom:6px;">Step 4 — Average the top 10 entries (no curve)</div>'
                '<div style="font-size:12px;color:#c9d1d9;margin-bottom:6px;">Take the 10 highest entries, sum them, divide by 10. Fewer than 10 wins? Still divide by 10 — missing slots contribute 0.</div>'
                '<div style="font-family:monospace;font-size:10px;color:#8b949e;background:#0a0610;border-radius:4px;padding:5px 8px;">'
                'on_factor[T] = Σ top-10 entries / 10'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                # Final output
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#0d0d1a;border:2px solid #79c0ff;border-radius:8px;padding:10px 28px;">'
                '<div style="font-size:14px;font-weight:700;color:#79c0ff;">🕸️ ON Factor &nbsp; [0 → 1]</div>'
                '</div></div>'
                '</div>'
            )
            st.markdown(_on_fc, unsafe_allow_html=True)
            st.markdown(
                '<span class="tag-v">✓ VERIFIED — two-pass scalar computation (NOT iterative PageRank), '
                'reverse-engineered from '
                '<a href="https://github.com/ValveSoftware/counter-strike_regional_standings/blob/main/model/team.js" target="_blank" style="color:#79c0ff">team.js</a>. '
                'Engine MAE ≤ 0.01 against Valve\'s published values across 21 snapshots.</span>',
                unsafe_allow_html=True)


        # ═════════════════════════════════════════════════════════
        # LEFT COLUMN — Interactive Network Sandbox
        # ═════════════════════════════════════════════════════════
        @st.fragment
        def _on_network_sandbox():
            st.markdown(
                '<div style="background:#1a0f2e;border:1px solid #7c3aed;border-radius:8px;'
                'padding:12px 16px;margin-bottom:12px;font-size:18px;font-weight:600;'
                'color:#c4b5fd;text-align:center;">🕸️ <strong>Network Sandbox</strong></div>',
                unsafe_allow_html=True)
            st.caption(
                "Configure match results and watch both passes compute live. "
                "For pedagogical clarity, age_w and ev_w are fixed at 1.0 here — so "
                "every Pass-2 entry equals the opponent's ownNetwork directly."
            )

            # ── ① Match results UI ──────────────────────────────
            st.markdown("**① Match results** — who beat whom?")

            if st.button("🎲 Randomise Results", key="preset_rng", type="primary"):
                _rng_teams    = _ON_PRESET_TEAMS[:]
                _rng_strength = _ON_STRENGTHS[:]
                _rng_wins     = []
                for _ri in range(len(_rng_teams)):
                    for _rj in range(_ri + 1, len(_rng_teams)):
                        _t1, _t2   = _rng_teams[_ri], _rng_teams[_rj]
                        _s_a, _s_b = _rng_strength[_ri], _rng_strength[_rj]
                        _p_a = _s_a**3 / (_s_a**3 + _s_b**3)
                        for _ in range(2):
                            if np.random.random() < _p_a:
                                _rng_wins.append((_t1, _t2))
                            else:
                                _rng_wins.append((_t2, _t1))
                _win_counts = {t: 0 for t in _rng_teams}
                for _w, _l in _rng_wins:
                    _win_counts[_w] += 1
                for _t in _rng_teams:
                    while _win_counts[_t] < 1:
                        _opp = np.random.choice([x for x in _rng_teams if x != _t])
                        _rng_wins.append((_t, _opp))
                        _win_counts[_t] += 1
                st.session_state.on_net_teams = _rng_teams
                st.session_state.on_net_wins  = _rng_wins
                # no st.rerun() — fragment reruns automatically on button click

            # Add-win row
            _wa, _wb, _wc, _wd = st.columns([2.8, 0.7, 2.8, 1.3])
            _win_from = _wa.selectbox(
                "Winner", st.session_state.on_net_teams,
                key="on_win_from", label_visibility="collapsed")
            _wb.markdown(
                '<div style="padding-top:8px;text-align:center;font-size:12px;'
                'color:#8b949e;">beat</div>',
                unsafe_allow_html=True)
            _win_to = _wc.selectbox(
                "Loser", st.session_state.on_net_teams,
                key="on_win_to", label_visibility="collapsed")
            _wd.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
            if _wd.button("➕ Add", use_container_width=True, type="primary"):
                if _win_from != _win_to:
                    st.session_state.on_net_wins.append((_win_from, _win_to))
                    # no st.rerun() — fragment reruns automatically on button click

            # Compact delete-button CSS — scoped to buttons inside NESTED columns
            # (the × delete buttons live in st.columns([5,1]) inside the outer match
            # grid st.columns(n_teams), so they have two column ancestors). The top-
            # level nav buttons have only one column ancestor and are NOT affected.
            st.markdown(
                "<style>"
                "div[data-testid='column'] div[data-testid='column'] div[data-testid='stButton'] button[kind='secondaryFormSubmit'],"
                "div[data-testid='column'] div[data-testid='column'] div[data-testid='stButton'] button[kind='secondary']{"
                "padding:1px 4px!important;"
                "min-height:22px!important;"
                "height:22px!important;"
                "font-size:10px!important;"
                "line-height:1!important;"
                "border-radius:4px!important;"
                "}</style>",
                unsafe_allow_html=True,
            )

            # Match grid — one column per team, wins with inline ✕
            _wins_by_team: dict = {}
            for _widx, (_wf, _wl) in enumerate(st.session_state.on_net_wins):
                _wins_by_team.setdefault(_wf, []).append((_widx, _wl))

            _match_cols = st.columns(len(st.session_state.on_net_teams))
            for _ci, _cteam in enumerate(st.session_state.on_net_teams):
                with _match_cols[_ci]:
                    st.markdown(
                        f'<div style="font-size:11px;font-weight:600;color:#79c0ff;'
                        f'padding-bottom:4px;border-bottom:1px solid #21262d;'
                        f'margin-bottom:4px;">{_cteam}</div>',
                        unsafe_allow_html=True)
                    _team_wins = _wins_by_team.get(_cteam, [])
                    if _team_wins:
                        for _widx2, _wopp in _team_wins:
                            _wname_col, _wdel_col = st.columns([5, 1])
                            _wname_col.markdown(
                                f'<div style="font-size:11px;color:#c9d1d9;'
                                f'padding:1px 0;white-space:nowrap;overflow:hidden;'
                                f'text-overflow:ellipsis;">'
                                f'<span style="color:#484f58;margin-right:3px;">›</span>'
                                f'{_wopp}</div>',
                                unsafe_allow_html=True)
                            if _wdel_col.button("×", key=f"on_wdel_{_widx2}"):
                                st.session_state.on_net_wins.pop(_widx2)
                                # no st.rerun() — fragment reruns automatically on button click
                    else:
                        st.markdown(
                            '<div style="font-size:12px;color:#30363d;padding:2px 0;">—</div>',
                            unsafe_allow_html=True)

            st.markdown("---")

            # ── ② Pass 1 — ownNetwork for every team ────────────
            st.markdown(
                '<div style="background:#0d1a2e;border-left:4px solid #58a6ff;'
                'border-radius:6px;padding:8px 14px;margin:6px 0;">'
                '<div style="font-size:14px;font-weight:700;color:#58a6ff;">'
                '② Pass 1 — ownNetwork for every team</div>'
                '<div style="font-size:11px;color:#8b949e;">'
                'distinct_def = # distinct opponents beaten (age_w=1 in sandbox). '
                'ownNetwork = min(distinct_def / reference, 1.0).'
                '</div></div>',
                unsafe_allow_html=True,
            )

            _rank_help = (
                f"Real VRS uses 5th-highest across ~200 teams. With our "
                f"{len(st.session_state.on_net_teams)}-team demo, try rank 2 (default) "
                f"to watch ownNetwork clamp at 1.0 for the top team and scale "
                f"linearly below it."
            )
            st.slider(
                "Reference rank (which highest distinct_def is used as the denominator?)",
                1, len(st.session_state.on_net_teams),
                key="on_net_outlier_rank",
                help=_rank_help,
            )

            # Live compute — after the slider so it picks up the latest rank
            _distinct_def, _own_net, _ref = _on_sandbox_compute(
                st.session_state.on_net_teams,
                st.session_state.on_net_wins,
                st.session_state.on_net_outlier_rank,
            )

            _p1_teams  = sorted(st.session_state.on_net_teams,
                                key=lambda t: -_distinct_def.get(t, 0))
            _p1_dd     = [_distinct_def.get(t, 0)   for t in _p1_teams]
            _p1_own    = [_own_net.get(t, 0.0)      for t in _p1_teams]
            _p1_colors = [
                "#f0b429" if o >= 0.999 else
                "#58a6ff" if o >= 0.5   else
                "#6e7681"
                for o in _p1_own
            ]
            _fig_p1 = go.Figure()
            _fig_p1.add_trace(go.Bar(
                x=_p1_dd, y=_p1_teams,
                orientation="h",
                marker=dict(color=_p1_colors, line=dict(color="#0d1117", width=1)),
                text=[f" dd={d} → own={o:.2f}" for d, o in zip(_p1_dd, _p1_own)],
                textposition="outside", textfont=dict(size=11, color="#c9d1d9"),
                hovertemplate="%{y}<br>distinct_def = %{x}<extra></extra>",
                showlegend=False,
            ))
            _fig_p1.add_vline(
                x=_ref, line_color="#f0883e", line_width=2, line_dash="dash",
                annotation_text=f"ref (rank {st.session_state.on_net_outlier_rank}) = {_ref}",
                annotation_position="top",
                annotation_font=dict(color="#f0883e", size=11),
            )
            _p1_max = max(_p1_dd + [_ref]) if _p1_dd else _ref
            _fig_p1.update_layout(
                xaxis=dict(title="distinct_def", gridcolor="#21262d",
                           range=[0, _p1_max * 1.35 + 0.5]),
                yaxis=dict(autorange="reversed", gridcolor="#21262d"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9", size=11),
                margin=dict(l=10, r=40, t=20, b=30), height=220,
            )
            st.plotly_chart(_fig_p1, use_container_width=True, config={"staticPlot": True})

            # ── ③ Pass 2 — on_factor breakdown per team ─────────
            st.markdown(
                '<div style="background:#1a0d2e;border-left:4px solid #d2a8ff;'
                'border-radius:6px;padding:8px 14px;margin:14px 0 6px;">'
                '<div style="font-size:14px;font-weight:700;color:#d2a8ff;">'
                '③ Pass 2 — on_factor breakdown per team</div>'
                '<div style="font-size:11px;color:#8b949e;">'
                'Y-axis = on_factor contribution. '
                'Segment colour = opponent beaten. '
                'Number inside = wins against that team. '
                'Label above bar = final on_factor.'
                '</div></div>',
                unsafe_allow_html=True,
            )

            # One pastel colour per team (the loser's identity colour)
            _pastel = ["#A2D2FF", "#BDB2FF", "#B9FBC0", "#FFCCB6", "#FFADAD"]
            _team_color = {
                t: _pastel[i % len(_pastel)]
                for i, t in enumerate(st.session_state.on_net_teams)
            }

            # Total wins per (winner, loser) pair — used for text labels
            _pair_counts: dict = {}
            for (_sw, _sl) in st.session_state.on_net_wins:
                _pair_counts[(_sw, _sl)] = _pair_counts.get((_sw, _sl), 0) + 1

            # Per-winner, per-opponent contribution to on_factor
            # entries are sorted by opp.ownNetwork desc; top-10 are counted
            _all_on_factors: dict = {}
            _top10_contrib: dict = {}   # {winner: {loser: on_factor contribution}}
            for _t in st.session_state.on_net_teams:
                _t_entries = _on_sandbox_entries(_t, st.session_state.on_net_wins, _own_net)
                _top10 = _t_entries[:TOP_N]
                _contrib: dict = {}
                for (_opp, _val) in _top10:
                    _contrib[_opp] = _contrib.get(_opp, 0.0) + _val / TOP_N
                _top10_contrib[_t] = _contrib
                _all_on_factors[_t] = sum(_contrib.values())

            # One Bar trace per loser team (= one colour segment)
            _fig_stack = go.Figure()
            for _loser in st.session_state.on_net_teams:
                _y_vals, _txt_vals, _htxt = [], [], []
                for _winner in st.session_state.on_net_teams:
                    _c = _top10_contrib[_winner].get(_loser, 0.0)
                    _n = _pair_counts.get((_winner, _loser), 0)
                    _y_vals.append(_c)
                    _txt_vals.append(f"{_n} win{'s' if _n != 1 else ''}" if _n > 0 else "")
                    _htxt.append(
                        f"<b>{_winner}</b> beat <b>{_loser}</b>: {_n}× "
                        f"(contributes {_c:.3f} to on_factor)"
                    )
                _fig_stack.add_trace(go.Bar(
                    name=_loser,
                    x=st.session_state.on_net_teams,
                    y=_y_vals,
                    text=_txt_vals,
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=13, color="#1a1a2e", family="monospace"),
                    marker=dict(
                        color=_team_color[_loser],
                        line=dict(color="#0d1117", width=1),
                    ),
                    hovertemplate=[h + "<extra></extra>" for h in _htxt],
                ))

            # Annotate final on_factor above each bar
            for _t in st.session_state.on_net_teams:
                _fig_stack.add_annotation(
                    x=_t,
                    y=_all_on_factors[_t],
                    text=f"<b>{_all_on_factors[_t]:.3f}</b>",
                    showarrow=False,
                    yanchor="bottom",
                    yshift=5,
                    font=dict(size=12, color="#d2a8ff"),
                )

            _fig_stack.update_layout(
                barmode="stack",
                xaxis=dict(title="", gridcolor="#21262d",
                           tickfont=dict(size=11, color="#c9d1d9")),
                yaxis=dict(title="on_factor contribution", gridcolor="#21262d"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9", size=11),
                legend=dict(
                    title=dict(text="Opponent beaten", font=dict(size=11)),
                    orientation="h", y=-0.22,
                    font=dict(size=10),
                ),
                margin=dict(l=10, r=10, t=30, b=60), height=310,
            )
            st.plotly_chart(_fig_stack, use_container_width=True, config={"staticPlot": True})

        with col_r:
            _on_network_sandbox()

    # ══════════════════════════════════════════════════════════════
    if active_tab == "lan2":
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

        @st.fragment
        def _lan_sandbox():
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
                    # no st.rerun() — fragment reruns automatically on button click
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
                    # no st.rerun() — fragment reruns automatically on widget change
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
                _lan_val  = sum(x["entry"] for x in _top10) / len(_top10) if _top10 else 0.0
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
                    f'<span style="font-size:12px;color:#8b949e;">② LAN = Σ / {len(_r["top10"])}</span>'
                    f'<span style="font-size:30px;font-weight:700;color:#f85149;">{_r["lan_val"]:.4f}</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
            elif st.session_state.lan_sim_events:
                st.info("Click **Calculate LAN Wins** to see results.", icon="💡")
            else:
                st.info("Add at least one event to start.", icon="💡")

        with col_r:
            _lan_sandbox()

    # ══════════════════════════════════════════════════════════════
    if active_tab == "h2h":
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
    if active_tab == "seed":
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
# PAGE 4  ·  TEAM BREAKDOWN
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

    def _delta_chip(delta: float, fmt: str = "+.1f",
                    unit: str = "", threshold: float = 0.05) -> str:
        """Small inline pill showing the Pub→Today shift on a cell."""
        if abs(delta) < threshold:
            return ""
        arrow = "↑" if delta > 0 else "↓"
        col   = "#3fb950" if delta > 0 else "#f85149"
        bg    = "rgba(63,185,80,0.18)" if delta > 0 else "rgba(248,81,73,0.18)"
        return (
            f'<span style="display:inline-block;margin-left:4px;'
            f'padding:0 4px;border-radius:3px;background:{bg};'
            f'color:{col};font-size:9px;font-weight:600;'
            f'vertical-align:middle">{arrow} {delta:{fmt}}{unit}</span>'
        )

    # Pre-compute globals needed for calculation boxes
    all_avgs       = base_standings["seed_combined"].tolist()
    max_avg        = max(all_avgs) if all_avgs else 1.0
    min_avg        = min(all_avgs) if all_avgs else 0.0
    # VRS-point sensitivity: 1 unit of any factor → this many points in the final seed.
    # Derivation: each factor contributes factor/4 to avg;
    #   Δseed = Δavg / (max_avg − min_avg) × (SEED_MAX − SEED_MIN)
    #         = (factor/4) / _avg_range × 1600
    #   → pts per full factor unit = 400 / _avg_range
    _avg_range      = max(max_avg - min_avg, 1e-9)
    _pts_per_factor = (SEED_MAX - SEED_MIN) / (4.0 * _avg_range)
    bo_sums_sorted = sorted(base_standings["bo_sum"].tolist(), reverse=True)
    ref5_bo        = bo_sums_sorted[4] if len(bo_sums_sorted) >= 5 else (bo_sums_sorted[-1] if bo_sums_sorted else 1.0)
    opp_bo_map     = base_standings.drop_duplicates("team").set_index("team")["bo_factor"].to_dict()
    opp_on_map     = base_standings.drop_duplicates("team").set_index("team")["on_factor"].to_dict()
    # bo_ratio = raw normalised ratio (bo_sum / ref5, capped at 1.0) — what BC actually uses
    bo_ratio_map   = {
        team: min(1.0, float(bs) / max(ref5_bo, 1e-9))
        for team, bs in base_standings.drop_duplicates("team").set_index("team")["bo_sum"].to_dict().items()
    }

    # ── ownNetwork replay — per-opponent weights for breakdown display ────────
    # Valve's on_factor uses opp.own_network (from team.js Phase 2) as the
    # per-match multiplier.  For "As Published" mode these match m["opp_on"]
    # (Valve's pre-stored value); for "Updated"/"Sim" modes we recompute
    # own_network directly from team_match_history:
    #   distinct_def[T] = Σ over distinct opponents of max(age_w) in T's wins
    #   ref             = 5th-highest distinct_def
    #   own_network[T]  = min(distinct_def[T] / ref, 1)
    _max_age_by_pair: dict = {}
    for _wteam, _tms in team_match_history.items():
        for _m in _tms:
            if _m["result"] != "W":
                continue
            _pair = (_wteam, _m.get("opponent", ""))
            _aw   = _eff_age(_m)
            if _aw > _max_age_by_pair.get(_pair, 0.0):
                _max_age_by_pair[_pair] = _aw

    _distinct_def: dict = {}
    for (_wteam, _opp), _aw in _max_age_by_pair.items():
        _distinct_def[_wteam] = _distinct_def.get(_wteam, 0.0) + _aw

    _sorted_dd  = sorted(_distinct_def.values(), reverse=True)
    _ref_own    = _sorted_dd[4] if len(_sorted_dd) >= 5 else (
                  _sorted_dd[-1] if _sorted_dd else 1.0)
    _ref_own    = max(_ref_own, 1e-9)
    _on_prev_replay: dict = {
        _t: min(_d / _ref_own, 1.0) for _t, _d in _distinct_def.items()
    }

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

    def _learn_cta(slug: str, name: str, color: str) -> str:
        """CTA card linking a factor band to the matching section in 'How VRS Works'.
        Rendered inside the left column under the calc box so it occupies the
        empty space there and is visible enough for users to notice it's clickable."""
        return (
            f'<a href="?learn={slug}" target="_self" style="text-decoration:none;">'
            f'<div style="margin-top:10px;background:#161b22;'
            f'border:1px solid {color};border-left:4px solid {color};'
            f'border-radius:8px;padding:12px 16px;'
            f'display:flex;align-items:center;justify-content:space-between;gap:12px;'
            f'transition:background 0.15s;cursor:pointer;">'
            f'<div>'
            f'<div class="vrs-cta-title" style="font-size:14px;font-weight:700;margin-bottom:2px;">'
            f'💡 Learn how {name} works</div>'
            f'<div class="vrs-cta-sub" style="font-size:12px;">'
            f'Jump to the formula &amp; try it yourself with interactive sliders</div>'
            f'</div>'
            f'<span style="font-size:20px;color:{color};font-weight:700;">→</span>'
            f'</div></a>'
        )

    def _count_caption(counting: int, total: int, below_sum: float, top_sum: float,
                        item: str = "wins", color: str = "#8b949e") -> str:
        """Render a compact "N of M counting · X% of pool is below-cut" line."""
        below_pct = (below_sum / max(top_sum, 1e-9) * 100) if top_sum > 0 else 0.0
        below_bit = (
            f' · <span style="color:{color}">{total - counting}</span> below cut '
            f'(≈ {below_pct:.0f}% of top-10 pool)'
        ) if total > counting else ''
        return (
            f'<div style="margin:-2px 0 8px 14px;font-size:11px;color:#8b949e;">'
            f'🔎 <strong style="color:#c9d1d9">{counting}</strong> of '
            f'<strong style="color:#c9d1d9">{total}</strong> {item} counting'
            f'{below_bit}</div>'
        )

    def _factor_band(emoji, name, value, color, max_val=1.0, delta_html="", pts=None):
        pct = max(0.0, min(1.0, float(value) / max(float(max_val), 1e-9))) * 100
        pts_html = (
            f'<span style="font-size:11px;color:#8b949e;font-weight:400;margin-left:8px;">'
            f'≈ <strong style="color:#c9d1d9">{pts:,.0f}</strong> pts</span>'
        ) if pts is not None else ""
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;' +
            f'background:#161b22;border-left:4px solid {color};' +
            f'border-radius:6px;padding:10px 14px;margin-bottom:6px;">' +
            f'<span style="font-size:14px;font-weight:700;color:{color}">{emoji} {name}</span>' +
            f'<span style="display:flex;align-items:center;gap:8px;">' +
            f'<span style="display:inline-block;width:100px;height:8px;background:#21262d;border-radius:4px;overflow:hidden;">' +
            f'<span style="display:block;width:{pct:.0f}%;height:100%;background:{color};border-radius:4px;"></span></span>' +
            f'<span style="font-size:20px;font-weight:700;color:{color};min-width:50px;text-align:right">{float(value):.4f}</span>' +
            f'{pts_html}' +
            f'{delta_html}' +
            f'</span></div>'
        )

    # ═══════════════════════════════════════════════════════════════
    # THREE MAIN VIEWS
    # ═══════════════════════════════════════════════════════════════
    # Insights        — KPIs, donut, time window, why X above Y, climb plan
    # Score Breakdown — Score Change Breakdown, Phase 1 factors, Phase 2 H2H
    # Historical      — Long-running rank/score chart across snapshots
    #
    # Use a radio (not st.tabs) so only the active section's Python runs.
    # st.tabs executes ALL tab bodies every rerun, causing heavy renders
    # (history spinner, chart builds) to appear appended below the page.
    _bd_tab_sel = st.radio(
        "View",
        ["📊 Insights", "🔬 Score Breakdown", "📈 Historical Development"],
        horizontal=True,
        label_visibility="collapsed",
        key="bd_outer_tab",
    )

    # ═══════════════════════════════════════════════════════════════
    if _bd_tab_sel == "📊 Insights":
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
        # WHERE THE POINTS COME FROM — factor-share donut
        # ════════════════════════════════════════════════════════
        # Each factor contributes factor_value × _pts_per_factor points to `seed`.
        # H2H sits alongside seed as a ±shift on top. Hovering a slice tells
        # the user what their rank would be if that factor were zeroed out.
        _share_bo  = float(ex["bo_factor"])  * _pts_per_factor
        _share_bc  = float(ex["bc_factor"])  * _pts_per_factor
        _share_on  = float(ex["on_factor"])  * _pts_per_factor
        _share_lan = float(ex["lan_factor"]) * _pts_per_factor
        _share_h2h = float(ex["h2h_delta"])

        def _rank_without(factor_key: str) -> int:
            """Rank this team would hold if `factor_key` went to zero for them
            (others held constant).  Cheap recompute — no engine re-run."""
            _points = base_standings.set_index("team")["total_points"].to_dict()
            if factor_key == "bo":
                _points[sel_team] -= _share_bo
            elif factor_key == "bc":
                _points[sel_team] -= _share_bc
            elif factor_key == "on":
                _points[sel_team] -= _share_on
            elif factor_key == "lan":
                _points[sel_team] -= _share_lan
            elif factor_key == "h2h":
                _points[sel_team] -= _share_h2h
            sorted_teams = sorted(_points.items(), key=lambda x: x[1], reverse=True)
            for i, (t, _) in enumerate(sorted_teams, 1):
                if t == sel_team:
                    return i
            return -1

        _donut_labels = ["🏆 BO", "💰 BC", "🕸️ ON", "🖥️ LAN"]
        _donut_values = [max(0.0, _share_bo), max(0.0, _share_bc),
                         max(0.0, _share_on), max(0.0, _share_lan)]
        _donut_colors = ["#f0b429", "#3fb950", "#79c0ff", "#f85149"]
        _donut_keys   = ["bo", "bc", "on", "lan"]
        _donut_ranks  = [_rank_without(k) for k in _donut_keys]

        _dcol1, _dcol2 = st.columns([2, 3])
        with _dcol1:
            fig_donut = go.Figure(go.Pie(
                labels=_donut_labels,
                values=_donut_values,
                hole=0.58,
                marker=dict(colors=_donut_colors, line=dict(color="#0d1117", width=2)),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "%{value:.0f} pts · %{percent}<br>"
                    "Rank without this factor: #%{customdata}<extra></extra>"
                ),
                customdata=_donut_ranks,
                textinfo="label+percent",
                textfont=dict(size=11, color="#c9d1d9"),
                sort=False,
            ))
            # Total seed in the middle
            fig_donut.add_annotation(
                text=f"<b>{ex['seed']:,.0f}</b><br><span style='font-size:10px;color:#8b949e'>Factor Score</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color="#c9d1d9"),
            )
            fig_donut.update_layout(
                title=dict(text="Where the points come from", x=0.0, font=dict(size=13)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=10),
                height=260,
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        with _dcol2:
            # Remove-this-factor cards — what rank would this team hold if a factor disappeared?
            current_rank = int(ex["rank"])
            h2h_rank = _rank_without("h2h")

            # Pub-side contributions for the inline delta chips. We use the same
            # `_pts_per_factor` scale as the sim side so the chip reads as a
            # clean "points shift on this contribution" number.
            if _orig_ex is not None:
                _pub_share_bo  = float(_orig_ex["bo_factor"])  * _pts_per_factor
                _pub_share_bc  = float(_orig_ex["bc_factor"])  * _pts_per_factor
                _pub_share_on  = float(_orig_ex["on_factor"])  * _pts_per_factor
                _pub_share_lan = float(_orig_ex["lan_factor"]) * _pts_per_factor
                _pub_share_h2h = float(_orig_ex["h2h_delta"])
                _pub_shares = [_pub_share_bo, _pub_share_bc, _pub_share_on, _pub_share_lan, _pub_share_h2h]
            else:
                _pub_shares = [None] * 5

            _rm_rows = []
            for label, pts, pub_pts, color, without_rank in zip(
                ["🏆 BO", "💰 BC", "🕸️ ON", "🖥️ LAN", "⚔️ H2H"],
                [_share_bo, _share_bc, _share_on, _share_lan, _share_h2h],
                _pub_shares,
                ["#f0b429", "#3fb950", "#79c0ff", "#f85149", "#d2a8ff"],
                _donut_ranks + [h2h_rank],
            ):
                _contrib_chip = ""
                if pub_pts is not None and (mode_active == "updated" or _orig_ex is not None):
                    _contrib_chip = _delta_chip(pts - pub_pts, fmt="+.1f", threshold=0.5)
                _drop = without_rank - current_rank  # positive = would fall
                _arrow = (
                    f'<span style="color:#f85149;font-weight:700">▼ {abs(_drop)}</span>' if _drop > 0 else
                    f'<span style="color:#3fb950;font-weight:700">▲ {abs(_drop)}</span>' if _drop < 0 else
                    '<span style="color:#8b949e">—</span>'
                )
                _rm_rows.append(
                    f'<tr style="border-bottom:1px solid #21262d;">'
                    f'<td style="padding:5px 8px;color:{color};font-weight:600;font-size:12px">{label}</td>'
                    f'<td style="padding:5px 8px;text-align:right;color:#c9d1d9;font-size:12px;white-space:nowrap">{pts:+,.0f} pts{_contrib_chip}</td>'
                    f'<td style="padding:5px 8px;text-align:center;color:#8b949e;font-size:11px">#{current_rank} → #{without_rank}</td>'
                    f'<td style="padding:5px 8px;text-align:right">{_arrow}</td>'
                    f'</tr>'
                )
            st.markdown(
                '<div style="font-size:12px;color:#8b949e;margin-bottom:6px;">'
                '🧪 <strong>What if this factor were zero for this team?</strong> '
                '(other teams held constant — a quick sensitivity check, not a re-run)'
                '</div>'
                '<div style="border:1px solid #30363d;border-radius:8px;overflow:hidden;">'
                '<table style="width:100%;border-collapse:collapse;">'
                '<thead style="background:#161b22;color:#8b949e;font-size:10px;text-transform:uppercase;">'
                '<tr><th style="padding:7px 8px;text-align:left">Factor</th>'
                '<th style="padding:7px 8px;text-align:right">Contribution</th>'
                '<th style="padding:7px 8px;text-align:center">Rank change</th>'
                '<th style="padding:7px 8px;text-align:right">Δ</th></tr>'
                '</thead><tbody>' + "".join(_rm_rows) + '</tbody></table></div>',
                unsafe_allow_html=True,
            )

        # ════════════════════════════════════════════════════════
        # 4.1 — TIME WINDOW: what's expiring & what just landed
        # ════════════════════════════════════════════════════════
        # Both modes are forward-looking from a fixed anchor:
        #
        #   Today mode      → anchor = today (sim_cutoff_dt)
        #   As-published    → anchor = the snapshot's published date
        #
        # The reference R is always the next Valve publication after the
        # anchor (first Monday of next calendar month — empirically
        # verified Dec 2024 → present, see cache/github_vrs).
        #
        # "At risk" = points lost between anchor and R:
        #   loss = pts × (1 − age_w_at_R / age_w_at_anchor)
        #
        # Today mode is RED (the next pub is imminent, so the red band
        # is narrow — only the days that will fully drop in the few
        # days remaining); As-published is YELLOW (a full cycle out, so
        # a wider band of items is at risk but it's not yet urgent).
        #
        # X-axis = days before anchor.  Day 0 = "now" from the user's
        # perspective.  The green flat-zone band straddles day 0 — the
        # right half is the past portion (data points exist there) and
        # the left half is the future portion (where data will appear
        # as time passes — the green zone "grows day by day" toward R).
        st.markdown("---")
        st.markdown("### ⏳ Time Window — what's expiring, what's fresh")

        # ── anchor / reference / mode-specific framing ─────────
        if mode_active == "updated":
            _anchor_dt   = sim_cutoff_dt
            _exp_color   = "#f85149"           # red — urgent
            _exp_emoji   = "⏳"
            _exp_title   = "At risk by next update"
            _at_risk_label = "at risk"
        else:
            _anchor_dt   = cutoff_dt
            _exp_color   = "#e3b341"           # amber / yellow — at risk next cycle
            _exp_emoji   = "📉"
            _exp_title   = "At risk next cycle"
            _at_risk_label = "at risk"

        _ref_pub      = next_valve_publication(_anchor_dt)
        _other_anchor = prev_valve_publication(_anchor_dt)
        _gap_days     = max(0, (_ref_pub - _anchor_dt).days)
        _cycle_days   = max(0, (_anchor_dt - _other_anchor).days)

        _ref_cutoff = sim_cutoff_dt if mode_active == "updated" else cutoff_dt

        def _aw_anchor(d):
            return age_weight(d, _anchor_dt)
        def _aw_ref(d):
            return age_weight(d, _ref_pub)

        _window_items: list[dict] = []

        # ── BO: each top-10 prize's share of bo_factor ─────────
        # Sort by the contribution at the *viewing* cutoff (matches the
        # per-factor table below).  The expiring math then re-projects.
        _bo_local = []
        for bp in bo_prizes:
            try:
                _dt = datetime.strptime(str(bp["event_date"]).strip(), "%Y-%m-%d")
            except Exception:
                continue
            if mode_active == "updated":
                _aw = age_weight(_dt, _ref_cutoff)
            else:
                _aw = float(bp.get("age_weight", 0.0))
            if _aw <= 0:
                continue
            _scaled = float(bp["prize_won"]) * _aw
            _bo_local.append({"date": _dt, "age_w": _aw,
                              "scaled": _scaled, "amount": float(bp["prize_won"])})
        _bo_local.sort(key=lambda x: x["scaled"], reverse=True)
        _bo_top_sum = sum(b["scaled"] for b in _bo_local[:10])
        for b in _bo_local[:10]:
            _share = b["scaled"] / max(_bo_top_sum, 1e-9)
            _pts   = _share * float(ex["bo_factor"]) * _pts_per_factor
            _window_items.append({
                "date": b["date"], "age_w": b["age_w"], "pts": _pts,
                "age_w_anchor": _aw_anchor(b["date"]),
                "age_w_ref":    _aw_ref(b["date"]),
                "kind": "BO", "color": "#f0b429",
                "label": f"${b['amount']:,.0f} prize",
                "match_id": None,
            })

        # ── BC + ON: top-10 winning entries ─────────────────────
        _wins = [m for m in raw_ms if m["result"] == "W"]

        _bc_e = sorted(
            [(bo_ratio_map.get(m["opponent"], 0.0) * _eff_age(m) * m.get("ev_w", 0.0), m)
             for m in _wins],
            key=lambda x: x[0], reverse=True,
        )
        for e, m in _bc_e[:10]:
            if e <= 0:
                continue
            _window_items.append({
                "date": m["date"], "age_w": _eff_age(m),
                "pts": e / 10 * _pts_per_factor,
                "age_w_anchor": _aw_anchor(m["date"]),
                "age_w_ref":    _aw_ref(m["date"]),
                "kind": "BC", "color": "#3fb950",
                "label": f"vs {m['opponent']}",
                "match_id": m.get("match_id"),
            })

        _on_e = sorted(
            [(((m.get("opp_on", 0.0) if m.get("opp_on", 0.0) > 0
                 else _on_prev_replay.get(m["opponent"], 0.0))
                * _eff_age(m) * m.get("ev_w", 0.0)), m)
             for m in _wins],
            key=lambda x: x[0], reverse=True,
        )
        for e, m in _on_e[:10]:
            if e <= 0:
                continue
            _window_items.append({
                "date": m["date"], "age_w": _eff_age(m),
                "pts": e / 10 * _pts_per_factor,
                "age_w_anchor": _aw_anchor(m["date"]),
                "age_w_ref":    _aw_ref(m["date"]),
                "kind": "ON", "color": "#79c0ff",
                "label": f"vs {m['opponent']}",
                "match_id": m.get("match_id"),
            })

        # ── LAN top-10 ──────────────────────────────────────────
        _lan_l = sorted([m for m in raw_ms if m["result"] == "W" and m.get("is_lan")],
                        key=lambda mm: _eff_age(mm), reverse=True)
        for m in _lan_l[:10]:
            _aw = _eff_age(m)
            if _aw <= 0:
                continue
            _window_items.append({
                "date": m["date"], "age_w": _aw,
                "pts": _aw / 10 * _pts_per_factor,
                "age_w_anchor": _aw_anchor(m["date"]),
                "age_w_ref":    _aw_ref(m["date"]),
                "kind": "LAN", "color": "#f85149",
                "label": f"vs {m['opponent']}",
                "match_id": m.get("match_id"),
            })

        # ── Per-item loss between anchor and R ──────────────────
        # `pts` is the contribution at the viewing cutoff which now
        # equals the anchor in BOTH modes (sim_cutoff_dt for updated,
        # cutoff_dt for published), so pts_anchor = pts.
        # loss = pts × (1 − age_w_at_R / age_w_at_anchor)
        for it in _window_items:
            awa, awr = it["age_w_anchor"], it["age_w_ref"]
            it["pts_anchor"] = it["pts"]
            if awa <= 0:
                it["loss"] = 0.0
                continue
            ratio = (awr / awa) if awa > 0 else 0.0
            it["loss"] = max(0.0, it["pts"] * (1.0 - ratio))

        # Split loss into "fully drops out" vs "decays but stays in window"
        # so the card can spell out both halves clearly.
        _full_drops = [it for it in _window_items
                       if it["age_w_anchor"] > 0 and it["age_w_ref"] <= 1e-9]
        _decay_only = [it for it in _window_items
                       if it["age_w_anchor"] > 0 and it["age_w_ref"] > 1e-9
                       and it["loss"] > 0]
        _pts_full   = sum(it["loss"] for it in _full_drops)
        _pts_decay  = sum(it["loss"] for it in _decay_only)
        _pts_at_risk = _pts_full + _pts_decay

        # ── Newly gained: matches/prizes dated since the previous
        # publication (i.e. landed inside this most recent cycle).  No
        # green-at-R filter — in the published view almost nothing
        # would survive that filter (R is a full cycle out), and the
        # user-facing question is "what landed since we last looked?".
        _green = [it for it in _window_items
                  if it["date"] >= _other_anchor]
        _pts_recent = sum(it["pts"] for it in _green)

        # Pre-format date strings for the cards.
        _anchor_str  = _anchor_dt.strftime("%b %d")
        _ref_str     = _ref_pub.strftime("%b %d, %Y")
        _ref_short   = _ref_pub.strftime("%b %d")
        _other_str   = _other_anchor.strftime("%b %d")

        # Expiring subtitle — concise, both halves spelled out.
        if _full_drops or _decay_only:
            _parts = []
            if _full_drops:
                _parts.append(
                    f"<strong>{len(_full_drops)}</strong> "
                    f"{'item' if len(_full_drops) == 1 else 'items'} "
                    f"fully drop out (~{_pts_full:.1f} pts)"
                )
            if _decay_only:
                _parts.append(
                    f"<strong>{len(_decay_only)}</strong> more decay "
                    f"(~{_pts_decay:.1f} pts)"
                )
            _gap_phrase = (
                f"in {_gap_days} day{'s' if _gap_days != 1 else ''}"
                if _gap_days > 0 else "today"
            )
            _exp_subtitle = (
                f"By <strong>{_ref_short}</strong> ({_gap_phrase}): "
                + " · ".join(_parts) + "."
            )
        else:
            _exp_subtitle = (
                f"Nothing in the top-10 buckets is set to drop or decay "
                f"meaningfully before <strong>{_ref_short}</strong>."
            )

        # Newly-gained subtitle.
        if _green:
            _boost_subtitle = (
                f"<strong>{len(_green)}</strong> "
                f"{'match/prize' if len(_green) == 1 else 'matches/prizes'} "
                f"landed since <strong>{_other_str}</strong> "
                f"(last publication, {_cycle_days} day"
                f"{'s' if _cycle_days != 1 else ''} ago)."
            )
        else:
            _boost_subtitle = (
                f"No new top-10 contributors since <strong>{_other_str}</strong>."
            )

        _tw_l, _tw_r = st.columns([2, 3])
        with _tw_l:
            st.markdown(
                f'<div style="background:#161b22;border:1px solid {_exp_color};'
                f'border-left:4px solid {_exp_color};border-radius:8px;'
                f'padding:12px 14px;margin-bottom:8px;">'
                f'<div style="font-size:11px;color:#8b949e;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:4px;">{_exp_emoji} {_exp_title}</div>'
                f'<div style="font-size:22px;font-weight:700;color:{_exp_color};">'
                f'~{_pts_at_risk:,.1f} pts'
                f'<span style="font-size:12px;color:#8b949e;font-weight:400;'
                f'margin-left:6px;">{_at_risk_label}</span></div>'
                f'<div style="font-size:11px;color:#8b949e;margin-top:4px;'
                f'line-height:1.5;">{_exp_subtitle}</div></div>'
                f'<div style="background:#161b22;border:1px solid #3fb950;'
                f'border-left:4px solid #3fb950;border-radius:8px;'
                f'padding:12px 14px;">'
                f'<div style="font-size:11px;color:#8b949e;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:4px;">✨ Newly gained</div>'
                f'<div style="font-size:22px;font-weight:700;color:#3fb950;">'
                f'~{_pts_recent:,.1f} pts'
                f'<span style="font-size:12px;color:#8b949e;font-weight:400;'
                f'margin-left:6px;">newly gained</span></div>'
                f'<div style="font-size:11px;color:#8b949e;margin-top:4px;'
                f'line-height:1.5;">{_boost_subtitle}</div></div>',
                unsafe_allow_html=True,
            )

        with _tw_r:
            if _window_items:
                import plotly.graph_objects as _go
                fig_tl = _go.Figure()
                _kind_y = {"BO": 4, "BC": 3, "ON": 2, "LAN": 1}
                _kind_color = {"BO": "#f0b429", "BC": "#3fb950",
                               "ON": "#79c0ff", "LAN": "#f85149"}

                # x-axis = days before anchor (today / snapshot date).
                # Day 0 = anchor.  All actual data points have x ≥ 0
                # (no future matches in the snapshot).  Bands extend
                # into negative x to visualise the future portion of
                # the green flat-zone-at-R that hasn't been "filled" yet.
                for kind in ["BO", "BC", "ON", "LAN"]:
                    items = [it for it in _window_items if it["kind"] == kind]
                    if not items:
                        continue
                    fig_tl.add_trace(_go.Scatter(
                        x=[(_anchor_dt - it["date"]).days for it in items],
                        y=[_kind_y[kind]] * len(items),
                        mode="markers",
                        marker=dict(
                            color=_kind_color[kind],
                            size=[max(7, min(28, 7 + abs(it["pts"]) / 4))
                                  for it in items],
                            opacity=[max(0.35, min(1.0, it["age_w_anchor"]))
                                     for it in items],
                            line=dict(color="#0d1117", width=1),
                        ),
                        name=kind,
                        customdata=[
                            (it["label"], it["age_w_anchor"], it["age_w_ref"],
                             it["pts"], it["loss"],
                             it["date"].strftime("%Y-%m-%d"))
                            for it in items
                        ],
                        hovertemplate=(
                            f"<b>{kind}</b> · %{{customdata[5]}}<br>"
                            "%{customdata[0]}<br>"
                            "age weight: %{customdata[1]:.2f} → %{customdata[2]:.2f}<br>"
                            "≈ %{customdata[3]:+,.1f} pts · "
                            "loss %{customdata[4]:+,.1f}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    ))

                # Bands in days-before-anchor coordinates.
                #
                # TODAY mode (forward-looking from today):
                #   GREEN  [-gap, 30 - gap]   = items that are (or will be)
                #                               in the 30-day flat zone at R.
                #                               Right edge sits on day 0
                #                               only when anchor == R.
                #   RED    [180 - gap, 180]   = items currently in window
                #                               that will fully drop by R.
                #
                # PUBLISHED mode (chart starts at the snapshot, no future):
                #   GREEN  [0, 30]            = items dated in the 30 days
                #                               before the snapshot — the
                #                               snapshot's own flat zone
                #                               (i.e. "newly added").
                #   YELLOW [180 - gap, 180]   = items currently in window
                #                               that are at risk of fully
                #                               dropping by next pub R.
                if mode_active == "updated":
                    _green_left  = -_gap_days
                    _green_right = 30 - _gap_days
                    _x_min       = min(-5, _green_left - 3)
                else:
                    _green_left  = 0
                    _green_right = 30
                    _x_min       = -3
                _risk_left   = 180 - _gap_days
                _risk_right  = 180
                _x_max = 185

                fig_tl.add_vrect(x0=_green_left, x1=_green_right,
                                 fillcolor="#3fb950",
                                 opacity=0.10, line_width=0)
                if _gap_days > 0 and _risk_right > _risk_left:
                    fig_tl.add_vrect(x0=_risk_left, x1=_risk_right,
                                     fillcolor=_exp_color,
                                     opacity=0.20, line_width=0)
                # Solid line at day 0 = "now" from the user's view.
                fig_tl.add_vline(x=0, line_color="#8b949e",
                                 line_dash="solid", line_width=1)
                # Dotted line at the 180-day window edge.
                fig_tl.add_vline(x=180, line_color="#30363d",
                                 line_dash="dot", line_width=1)
                fig_tl.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b949e", size=10),
                    xaxis=dict(
                        title=dict(
                            text=f"Days before {_anchor_str} "
                                 f"(0 = "
                                 f"{'today' if mode_active == 'updated' else 'snapshot'}, "
                                 f"next pub {_ref_short})",
                            font=dict(size=10),
                        ),
                        range=[_x_min, _x_max], gridcolor="#21262d",
                        tickvals=[0, 30, 60, 90, 120, 150, 180],
                    ),
                    yaxis=dict(
                        tickvals=[1, 2, 3, 4],
                        ticktext=["LAN", "ON", "BC", "BO"],
                        range=[0.4, 4.6], gridcolor="#21262d", zeroline=False,
                    ),
                    margin=dict(l=10, r=10, t=10, b=30), height=210,
                )
                st.plotly_chart(fig_tl, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.caption("No counting items in window.")

        if mode_active == "updated":
            st.caption(
                f"Day 0 = **{_anchor_str}** (today). "
                f"Green band = items that are (or will be) in the 30-day "
                f"flat zone at the next publication on **{_ref_str}** "
                f"(first Monday of {_ref_pub.strftime('%B')}). "
                f"Red band = items currently in window that will fully "
                f"drop by then ({_gap_days}-day sliver)."
            )
        else:
            st.caption(
                f"Day 0 = **{_anchor_str}** (this snapshot). "
                f"Green band = items added in the last 30 days "
                f"(currently in the flat zone). "
                f"Yellow band = items at risk of fully dropping by the "
                f"next publication on **{_ref_str}** "
                f"({_gap_days}-day window after the snapshot)."
            )

        # ════════════════════════════════════════════════════════
        # 4.2 + 4.3 — PEER COMPARATOR + MATCH IMPACT EXPLORER
        # ════════════════════════════════════════════════════════
        # Side-by-side panels: peer comparator (left) breaks the gap
        # to a chosen peer down into per-factor contributions; match
        # impact explorer (right) re-runs the engine without a chosen
        # match to surface that match's own contribution. Within each
        # half-width panel everything stacks vertically to avoid
        # cramped nested columns.
        st.markdown("---")
        _peer_panel, _impact_panel = st.columns(2)

        # ── LEFT: 🆚 Why is X above Y? ─────────────────────────
        with _peer_panel:
            st.markdown("### 🆚 Why is X above Y? — Peer comparator")

            _team_rank = int(ex["rank"])
            _peer_options: list[tuple[str, str]] = []
            if _team_rank > 1:
                _above = base_standings[base_standings["rank"] == _team_rank - 1]
                if not _above.empty:
                    _peer_options.append(
                        (f"⬆️ Rank above (#{_team_rank - 1})", str(_above.iloc[0]["team"]))
                    )
            if _team_rank > 2:
                _top = base_standings[base_standings["rank"] == 1]
                if not _top.empty:
                    _peer_options.append(("🥇 Rank #1", str(_top.iloc[0]["team"])))
            if _team_rank != 10 and 10 in base_standings["rank"].values:
                _bot10 = base_standings[base_standings["rank"] == 10]
                if not _bot10.empty:
                    _peer_options.append(
                        ("🔟 Bottom of top-10 (#10)", str(_bot10.iloc[0]["team"]))
                    )
            _other_teams = [t for t in base_standings["team"].tolist() if t != sel_team]

            _peer_choices = [opt[0] for opt in _peer_options] + ["✏️ Pick a team…"]
            _peer_mode = st.radio(
                "Compare against",
                _peer_choices,
                key=f"cmp_mode_{sel_team}",
                horizontal=False,
                label_visibility="collapsed",
            )
            if _peer_mode == "✏️ Pick a team…":
                _peer_team = st.selectbox(
                    "Peer",
                    _other_teams,
                    key=f"cmp_peer_{sel_team}",
                    label_visibility="collapsed",
                )
            else:
                _peer_team = next(t for label, t in _peer_options if label == _peer_mode)

            _peer_row = base_standings[base_standings["team"] == _peer_team].iloc[0]
            _gap = float(ex["total_points"]) - float(_peer_row["total_points"])
            _ahead = _gap > 0

            _factor_specs = [
                ("🏆 BO",  "bo_factor",  "#f0b429"),
                ("💰 BC",  "bc_factor",  "#3fb950"),
                ("🕸️ ON",  "on_factor",  "#79c0ff"),
                ("🖥️ LAN", "lan_factor", "#f85149"),
            ]
            _gap_rows: list[str] = []
            _factor_drivers: list[tuple[str, float]] = []
            for label, key, color in _factor_specs:
                _df  = float(ex[key]) - float(_peer_row[key])
                _dpt = _df * _pts_per_factor
                _factor_drivers.append((label, _dpt))
                _arrow = (
                    f'<span style="color:#3fb950;font-weight:700">▲ {_dpt:+,.1f}</span>'
                    if _dpt > 0 else
                    f'<span style="color:#f85149;font-weight:700">▼ {_dpt:+,.1f}</span>'
                    if _dpt < 0 else
                    '<span style="color:#8b949e">—</span>'
                )
                _gap_rows.append(
                    f'<tr style="border-bottom:1px solid #21262d;">'
                    f'<td style="padding:6px 8px;color:{color};font-weight:600;font-size:12px">{label}</td>'
                    f'<td style="padding:6px 8px;text-align:right;color:#c9d1d9;font-size:11px">'
                    f'{float(ex[key]):.4f}</td>'
                    f'<td style="padding:6px 8px;text-align:right;color:#8b949e;font-size:11px">'
                    f'{float(_peer_row[key]):.4f}</td>'
                    f'<td style="padding:6px 8px;text-align:right;font-size:11px">{_arrow}</td>'
                    f'</tr>'
                )

            _h2h_dpt = float(ex["h2h_delta"]) - float(_peer_row["h2h_delta"])
            _factor_drivers.append(("⚔️ H2H", _h2h_dpt))
            _h2h_arrow = (
                f'<span style="color:#3fb950;font-weight:700">▲ {_h2h_dpt:+,.1f}</span>'
                if _h2h_dpt > 0 else
                f'<span style="color:#f85149;font-weight:700">▼ {_h2h_dpt:+,.1f}</span>'
                if _h2h_dpt < 0 else
                '<span style="color:#8b949e">—</span>'
            )
            _gap_rows.append(
                f'<tr style="border-bottom:1px solid #21262d;">'
                f'<td style="padding:6px 8px;color:#d2a8ff;font-weight:600;font-size:12px">⚔️ H2H</td>'
                f'<td style="padding:6px 8px;text-align:right;color:#c9d1d9;font-size:11px">'
                f'{float(ex["h2h_delta"]):+.1f}</td>'
                f'<td style="padding:6px 8px;text-align:right;color:#8b949e;font-size:11px">'
                f'{float(_peer_row["h2h_delta"]):+.1f}</td>'
                f'<td style="padding:6px 8px;text-align:right;font-size:11px">{_h2h_arrow}</td>'
                f'</tr>'
            )

            _drivers_sorted = sorted(_factor_drivers, key=lambda x: abs(x[1]), reverse=True)[:2]
            _verb = "leads" if _ahead else "trails"
            _driver_phrase = " and ".join(
                f"<strong>{lbl}</strong> ({pts:+,.1f} pts)" for lbl, pts in _drivers_sorted
            )
            _summary = (
                f"<strong>{sel_team}</strong> {_verb} <strong>{_peer_team}</strong> by "
                f"<strong>{abs(_gap):,.1f} pts</strong>. "
                f"Biggest contributors to the gap: {_driver_phrase}."
            )

            _gap_color = "#3fb950" if _ahead else "#f85149"
            st.markdown(
                f'<div style="background:#161b22;border:1px solid {_gap_color};'
                f'border-radius:8px;padding:10px 14px;font-size:13px;color:#c9d1d9;'
                f'line-height:1.5;margin:8px 0;">{_summary}</div>'
                f'<div style="border:1px solid #30363d;border-radius:8px;overflow:hidden;">'
                f'<table style="width:100%;border-collapse:collapse;">'
                f'<thead style="background:#161b22;color:#8b949e;font-size:10px;text-transform:uppercase;">'
                f'<tr><th style="padding:7px 8px;text-align:left">Factor</th>'
                f'<th style="padding:7px 8px;text-align:right">{sel_team[:14]}</th>'
                f'<th style="padding:7px 8px;text-align:right">{_peer_team[:14]}</th>'
                f'<th style="padding:7px 8px;text-align:right">Gap (pts)</th></tr>'
                f'</thead><tbody>' + "".join(_gap_rows) + '</tbody></table></div>',
                unsafe_allow_html=True,
            )

        # ── RIGHT: 🔬 Match Impact Explorer ────────────────────
        with _impact_panel:
            st.markdown("### 🔬 Match Impact Explorer — what if a match never happened?")
            st.caption(
                "Removes the chosen match from the entire dataset and re-runs "
                "the full engine. First click takes ~1 s; later picks of the "
                "same match are instant for this session."
            )

            # Build candidate list: top-5 from BC, ON and LAN buckets,
            # plus this team's recent wins. Dedup by mid.
            _mi_cand: dict[int, dict] = {}

            def _mi_add(m, kind, e=None):
                mid = m.get("match_id")
                if mid is None:
                    return
                _mi_cand.setdefault(int(mid), {
                    "match_id": int(mid),
                    "date":     m["date"],
                    "opponent": m["opponent"],
                    "result":   m["result"],
                    "kinds":    [],
                    "is_lan":   bool(m.get("is_lan", False)),
                })
                _mi_cand[int(mid)]["kinds"].append(kind if e is None else f"{kind} #{e}")

            for _i, (_, _m) in enumerate(_bc_e[:5], start=1):
                _mi_add(_m, "BC", _i)
            for _i, (_, _m) in enumerate(_on_e[:5], start=1):
                _mi_add(_m, "ON", _i)
            for _i, _m in enumerate(_lan_l[:5], start=1):
                _mi_add(_m, "LAN", _i)
            _mi_recent = sorted(
                [m for m in raw_ms if m.get("match_id") is not None],
                key=lambda mm: mm["date"], reverse=True,
            )[:5]
            for _m in _mi_recent:
                _mi_add(_m, "recent")

            if not _mi_cand:
                st.caption("No matches with engine match-IDs available for this team.")
            else:
                _mi_list = sorted(_mi_cand.values(),
                                  key=lambda c: c["date"], reverse=True)

                def _mi_label(c):
                    tag = "·".join(sorted(set(c["kinds"])))
                    lan = " 🖥" if c["is_lan"] else ""
                    return (f"{c['date'].strftime('%Y-%m-%d')} · "
                            f"{c['result']} vs {c['opponent']}{lan} — {tag}")

                _mi_label_to_id = {_mi_label(c): c["match_id"] for c in _mi_list}
                _mi_picked_label = st.selectbox(
                    "Pick a match to remove",
                    list(_mi_label_to_id.keys()),
                    key=f"impact_pick_{sel_team}",
                )
                _mi_remove_id = _mi_label_to_id[_mi_picked_label]

                _mi_go = st.button(
                    "▶ Compute impact",
                    key=f"impact_btn_{sel_team}_{_mi_remove_id}",
                    type="primary",
                )

                # Session-scoped cache: avoids re-running the engine for the
                # same match more than once per session.
                if "_match_impact_cache" not in st.session_state:
                    st.session_state._match_impact_cache = {}
                _mi_cache = st.session_state._match_impact_cache
                _mi_key = (mode_active,
                           _ref_cutoff.strftime("%Y%m%d"),
                           int(_mi_remove_id))

                if _mi_go and _mi_key not in _mi_cache:
                    with st.spinner("Re-running engine without this match…"):
                        _mi_store = Store.from_valve(team_match_history, bo_prizes_map)
                        _mi_store.matches_df = (
                            _mi_store.matches_df[
                                _mi_store.matches_df["match_id"] != _mi_remove_id
                            ].reset_index(drop=True)
                        )
                        _mi_res = run_vrs(_mi_store, cutoff=_ref_cutoff)
                        _mi_cache[_mi_key] = _mi_res

                if _mi_key in _mi_cache:
                    _mi_res = _mi_cache[_mi_key]
                    _mi_std = _mi_res.get("standings", pd.DataFrame())
                    if _mi_std.empty or sel_team not in _mi_std["team"].values:
                        st.warning(
                            "After removing this match, this team is no longer "
                            "eligible (≥ 5 matches required)."
                        )
                    else:
                        _mi_row = _mi_std[_mi_std["team"] == sel_team].iloc[0]
                        # Sign convention: positive = this match HELPED the team
                        # (added pts / improved rank), negative = it hurt them.
                        # All deltas are computed as (current_with_match − without_match)
                        # so removing a contributing win shows as a positive contribution.
                        _mi_drank = int(_mi_row["rank"]) - int(ex["rank"])  # ranks gained from this match
                        _mi_dpts  = float(ex["total_points"]) - float(_mi_row["total_points"])
                        # Per-factor pts contribution: BO/BC/ON/LAN factors are
                        # converted into seed-points via `_pts_per_factor`
                        # (same scaling used by the donut + peer comparator);
                        # Seed and H2H deltas are already on the points scale.
                        _mi_dseed = float(ex["seed"])      - float(_mi_row["seed"])
                        _mi_dh2h  = float(ex["h2h_delta"]) - float(_mi_row["h2h_delta"])
                        _mi_dbo_pts  = (float(ex["bo_factor"])  - float(_mi_row["bo_factor"]))  * _pts_per_factor
                        _mi_dbc_pts  = (float(ex["bc_factor"])  - float(_mi_row["bc_factor"]))  * _pts_per_factor
                        _mi_don_pts  = (float(ex["on_factor"])  - float(_mi_row["on_factor"]))  * _pts_per_factor
                        _mi_dlan_pts = (float(ex["lan_factor"]) - float(_mi_row["lan_factor"])) * _pts_per_factor

                        _mi_rk_arrow = (
                            f'<span style="color:#3fb950;font-weight:700">▲ {_mi_drank}</span>'
                            if _mi_drank > 0 else
                            f'<span style="color:#f85149;font-weight:700">▼ {-_mi_drank}</span>'
                            if _mi_drank < 0 else
                            '<span style="color:#8b949e">—</span>'
                        )

                        st.markdown(
                            f'<div style="background:#161b22;border:1px solid #30363d;'
                            f'border-radius:8px;padding:12px 14px;margin:8px 0;">'
                            f'<div style="font-size:11px;color:#8b949e;text-transform:uppercase;">'
                            f'Impact of this match</div>'
                            f'<div style="font-size:22px;font-weight:700;color:#c9d1d9;'
                            f'margin-top:4px;">'
                            f'#{int(ex["rank"])} '
                            f'<span style="font-size:13px;color:#8b949e;font-weight:400">'
                            f'(would be #{int(_mi_row["rank"])} without · {_mi_rk_arrow})</span></div>'
                            f'<div style="font-size:13px;color:#c9d1d9;margin-top:6px;">'
                            f'{float(ex["total_points"]):,.1f} pts '
                            f'<span style="color:#8b949e;font-size:11px">'
                            f'({_mi_dpts:+,.1f} from this match)</span></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        def _mi_pts_row(label, color, dpts):
                            col = ("#3fb950" if dpts > 0 else
                                   "#f85149" if dpts < 0 else "#8b949e")
                            return (
                                f'<tr style="border-bottom:1px solid #21262d;">'
                                f'<td style="padding:5px 8px;color:{color};'
                                f'font-weight:600;font-size:12px">{label}</td>'
                                f'<td style="padding:5px 8px;text-align:right;'
                                f'color:{col};font-size:11px">{dpts:+,.1f}</td>'
                                f'</tr>'
                            )
                        _mi_html = (
                            _mi_pts_row("🏆 BO",    "#f0b429", _mi_dbo_pts)  +
                            _mi_pts_row("💰 BC",    "#3fb950", _mi_dbc_pts)  +
                            _mi_pts_row("🕸️ ON",    "#79c0ff", _mi_don_pts)  +
                            _mi_pts_row("🖥️ LAN",   "#f85149", _mi_dlan_pts) +
                            _mi_pts_row("🌱 Seed",  "#58a6ff", _mi_dseed)    +
                            _mi_pts_row("⚔️ H2H Δ", "#d2a8ff", _mi_dh2h)
                        )
                        st.markdown(
                            f'<div style="border:1px solid #30363d;border-radius:8px;'
                            f'overflow:hidden;">'
                            f'<table style="width:100%;border-collapse:collapse;">'
                            f'<thead style="background:#161b22;color:#8b949e;'
                            f'font-size:10px;text-transform:uppercase;">'
                            f'<tr><th style="padding:7px 8px;text-align:left">Factor</th>'
                            f'<th style="padding:7px 8px;text-align:right">'
                            f'Δ pts (from this match)</th></tr></thead>'
                            f'<tbody>{_mi_html}</tbody></table></div>',
                            unsafe_allow_html=True,
                        )

        # ════════════════════════════════════════════════════════
        # DAY 5 — CLIMB PLAN  (🧗 Pillar 2: Improvement Path)
        # ════════════════════════════════════════════════════════
        # Four stacked panels:
        #   5.1  Bottleneck detector  — weakest factor vs ±10-rank peers
        #   5.4  Expiring-points alert — quantified 14-day decay risk
        #   5.2  Target-opponent sim   — engine run per candidate (button)
        #   5.3  Path-to-rank checklist — greedy combo from 5.2 results
        # 5.1 and 5.4 are cheap and always visible; 5.2 is gated behind a
        # button (≈ 10× engine runs); 5.3 only renders once 5.2 is cached.
        st.markdown("---")
        st.markdown("### 🧗 Climb Plan — what holds this team back, and how to move up")

        # ── 5.1 Bottleneck detector ────────────────────────────
        _cp_rank = int(ex["rank"])
        _cp_peers = base_standings[
            base_standings["rank"].between(max(1, _cp_rank - 10), _cp_rank + 10) &
            (base_standings["team"] != sel_team)
        ]
        _cp_specs = [
            ("🏆 BO",  "bo_factor",  "#f0b429", "BO"),
            ("💰 BC",  "bc_factor",  "#3fb950", "BC"),
            ("🕸️ ON",  "on_factor",  "#79c0ff", "ON"),
            ("🖥️ LAN", "lan_factor", "#f85149", "LAN"),
        ]
        _cp_rows_data: list[tuple] = []
        _cp_worst = None       # (label, key, color, short, self, peer_mean, gap_pts)
        for lbl, key, color, short in _cp_specs:
            sv = float(ex[key])
            pm = float(_cp_peers[key].mean()) if not _cp_peers.empty else sv
            gap_pts = (sv - pm) * _pts_per_factor
            _cp_rows_data.append((lbl, key, color, short, sv, pm, gap_pts))
            if _cp_worst is None or gap_pts < _cp_worst[6]:
                _cp_worst = (lbl, key, color, short, sv, pm, gap_pts)

        # Root-cause diagnosis for the weakest factor.
        _cp_diag = ""
        if _cp_worst is not None and _cp_worst[6] < -0.5:
            lbl, key, color, short, sv, pm, gap_pts = _cp_worst
            if short == "ON":
                _my_opps  = len({m["opponent"] for m in raw_ms
                                  if m["result"] == "W" and _eff_age(m) > 0})
                _peer_opp_counts: list[int] = []
                for _pt in _cp_peers["team"].tolist():
                    _pms = team_match_history.get(_pt, [])
                    _peer_opp_counts.append(
                        len({m["opponent"] for m in _pms
                             if m["result"] == "W" and _eff_age(m) > 0})
                    )
                _peer_opp_mean = (sum(_peer_opp_counts) / len(_peer_opp_counts)
                                   if _peer_opp_counts else float(_my_opps))
                _cp_diag = (
                    f"<strong>{_my_opps}</strong> distinct opponents beaten vs "
                    f"peer average <strong>{_peer_opp_mean:.0f}</strong>. "
                    "Face new teams — repeat wins over the same opponent don't compound "
                    "in ownNetwork."
                )
            elif short == "BC":
                _opp_bos = [bo_ratio_map.get(m["opponent"], 0.0)
                            for m in raw_ms
                            if m["result"] == "W" and _eff_age(m) > 0]
                _avg_obo = (sum(_opp_bos) / len(_opp_bos)) if _opp_bos else 0.0
                _cp_diag = (
                    f"Average opponent BO ratio is <strong>{_avg_obo:.3f}</strong>. "
                    "BC rewards wins over teams with large prize earnings — "
                    "target opponents near the top of the rankings."
                )
            elif short == "LAN":
                _my_lan = int(ex["lan_wins"])
                _peer_lan = (float(_cp_peers["lan_wins"].mean())
                              if not _cp_peers.empty else 0.0)
                _cp_diag = (
                    f"<strong>{_my_lan}</strong> LAN win(s) in the 180-day window — "
                    f"peers average <strong>{_peer_lan:.1f}</strong>. "
                    "Only wins at LAN events count; online results don't move this factor."
                )
            else:  # BO
                _top_prize = max((float(bp["prize_won"]) for bp in bo_prizes),
                                  default=0.0)
                _cp_diag = (
                    f"Top prize in window: <strong>${_top_prize:,.0f}</strong>. "
                    "BO is driven by prize money — deep runs at large events compound "
                    "over the top-10 bucket."
                )

        _bn_l, _bn_r = st.columns([2, 3])
        with _bn_l:
            if _cp_worst is not None and _cp_worst[6] < -0.5:
                _wl = _cp_worst
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid {_wl[2]};'
                    f'border-left:4px solid {_wl[2]};border-radius:8px;'
                    f'padding:12px 14px;">'
                    f'<div style="font-size:11px;color:#8b949e;'
                    f'text-transform:uppercase;letter-spacing:0.5px;">'
                    f'Weakest factor vs peers</div>'
                    f'<div style="font-size:22px;font-weight:700;color:{_wl[2]};'
                    f'margin-top:4px;">{_wl[0]}</div>'
                    f'<div style="font-size:12px;color:#c9d1d9;margin-top:6px;">'
                    f'{_wl[4]:.3f} vs peer avg {_wl[5]:.3f} · '
                    f'<strong style="color:#f85149">{_wl[6]:+,.1f} pts</strong>'
                    f'</div>'
                    f'<div style="font-size:11px;color:#8b949e;margin-top:10px;'
                    f'line-height:1.55;">{_cp_diag}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:#161b22;border:1px solid #3fb950;'
                    'border-left:4px solid #3fb950;border-radius:8px;'
                    'padding:12px 14px;">'
                    '<div style="font-size:11px;color:#8b949e;'
                    'text-transform:uppercase;letter-spacing:0.5px;">Bottleneck</div>'
                    '<div style="font-size:18px;font-weight:700;color:#3fb950;'
                    'margin-top:4px;">No factor lags its peer group</div>'
                    '<div style="font-size:11px;color:#8b949e;margin-top:6px;'
                    'line-height:1.55;">'
                    "This team matches or exceeds the ±10-rank peer group on every "
                    "factor — climb will require lifting the whole bracket, "
                    "not fixing one weak spot.</div></div>",
                    unsafe_allow_html=True,
                )
        with _bn_r:
            _br_html = []
            for lbl, key, color, short, sv, pm, gap_pts in _cp_rows_data:
                _ar = (
                    f'<span style="color:#3fb950;font-weight:700">▲ {gap_pts:+,.1f}</span>'
                    if gap_pts > 0.5 else
                    f'<span style="color:#f85149;font-weight:700">▼ {gap_pts:+,.1f}</span>'
                    if gap_pts < -0.5 else
                    '<span style="color:#8b949e">—</span>'
                )
                _br_html.append(
                    f'<tr style="border-bottom:1px solid #21262d;">'
                    f'<td style="padding:6px 8px;color:{color};font-weight:600;'
                    f'font-size:12px">{lbl}</td>'
                    f'<td style="padding:6px 8px;text-align:right;color:#c9d1d9;'
                    f'font-size:11px">{sv:.4f}</td>'
                    f'<td style="padding:6px 8px;text-align:right;color:#8b949e;'
                    f'font-size:11px">{pm:.4f}</td>'
                    f'<td style="padding:6px 8px;text-align:right;font-size:11px">'
                    f'{_ar}</td></tr>'
                )
            st.markdown(
                '<div style="border:1px solid #30363d;border-radius:8px;overflow:hidden;">'
                '<table style="width:100%;border-collapse:collapse;">'
                '<thead style="background:#161b22;color:#8b949e;font-size:10px;'
                'text-transform:uppercase;">'
                '<tr><th style="padding:7px 8px;text-align:left">Factor</th>'
                '<th style="padding:7px 8px;text-align:right">Self</th>'
                f'<th style="padding:7px 8px;text-align:right">'
                f'Peer avg (±10 ranks, n={len(_cp_peers)})</th>'
                '<th style="padding:7px 8px;text-align:right">Gap (pts)</th></tr>'
                '</thead><tbody>' + "".join(_br_html) + '</tbody></table></div>',
                unsafe_allow_html=True,
            )

        # ── 5.4 Expiring-points alert ────────────────────────────
        # Mirrors the Time Window section's "at risk" math: every counting
        # item loses pts × (1 − age_w_at_R / age_w_at_anchor) by the next
        # reference cutoff R (next Valve publication).  This captures BOTH
        # items that fully drop out of the 180-day window AND items that
        # merely decay further (still in window, lower age weight).
        # _full_drops / _decay_only / _pts_full / _pts_decay / _pts_at_risk
        # / _ref_pub / _gap_days are computed in the Time Window section.
        _cp_n_full   = len(_full_drops)
        _cp_n_decay  = len(_decay_only)
        _cp_pts_risk = _pts_at_risk
        # Rough "wins worth +X pts" estimate: a single top-tier LAN win lifts
        # BC by ≈ opp_bo_ratio/10, ON by ≈ opp_ownNetwork/10, plus a LAN slot
        # (1/10 of lan_factor) at ev_w ≈ curve(0.25)=0.602. This is a guidance
        # number — the Target Opponents panel below gives the actual engine delta.
        _cp_lan_boost_est = (
            (1.0 * 0.602 / 10.0) +   # BC max contribution (opp_bo=1.0, ev_w=0.602)
            (1.0 * 0.602 / 10.0) +   # ON max contribution (opp_on≈1.0, ev_w=0.602)
            (1.0 / 10.0)             # LAN slot (age_w=1.0)
        ) / 4.0 * (SEED_MAX - SEED_MIN) / max(max_avg - min_avg, 1e-9)
        _cp_mid_boost_est = _cp_lan_boost_est * 0.55  # mid-tier ≈ 55% of a top-LAN win

        _cp_when_phrase = (
            f"by <strong>{_ref_short}</strong> "
            f"({'in ' + str(_gap_days) + ' day' + ('s' if _gap_days != 1 else '') if _gap_days > 0 else 'today'})"
        )

        st.markdown("#### ⚠️ Expiring alert — hold your rank")
        _ex_color = "#f85149" if _cp_pts_risk > 5.0 else "#d2a8ff"
        if _cp_pts_risk > 0:
            _n_top = max(1, int(round(_cp_pts_risk / max(_cp_lan_boost_est, 1e-9))))
            _n_mid = max(1, int(round(_cp_pts_risk / max(_cp_mid_boost_est, 1e-9))))
            _breakdown_parts = []
            if _cp_n_full:
                _breakdown_parts.append(
                    f'<strong style="color:{_ex_color}">{_cp_n_full}</strong> '
                    f'{"item" if _cp_n_full == 1 else "items"} fully drop out '
                    f'(~<strong>−{_pts_full:,.1f} pts</strong>)'
                )
            if _cp_n_decay:
                _breakdown_parts.append(
                    f'<strong style="color:{_ex_color}">{_cp_n_decay}</strong> more '
                    f'decay further (~<strong>−{_pts_decay:,.1f} pts</strong>)'
                )
            _breakdown_html = " · ".join(_breakdown_parts)
            st.markdown(
                f'<div style="background:#161b22;border:1px solid {_ex_color};'
                f'border-left:4px solid {_ex_color};border-radius:8px;'
                f'padding:12px 14px;">'
                f'<div style="font-size:13px;color:#c9d1d9;line-height:1.55;">'
                f'⏳ {_cp_when_phrase.capitalize()}: {_breakdown_html} — '
                f'about <strong style="color:{_ex_color}">−{_cp_pts_risk:,.1f} pts</strong> '
                f'total to the Factor Score.<br>'
                f'To <strong>hold rank</strong> you need wins worth '
                f'<strong>+{_cp_pts_risk:,.1f} pts</strong>. Rough equivalents: '
                f'≈ <strong>{_n_top}</strong> top-tier LAN win(s) at $250K+, or '
                f'≈ <strong>{_n_mid}</strong> mid-tier win(s). '
                f'<span style="color:#8b949e;font-size:11px">'
                f'(Aggregates age-weight decay across all top-10 BO/BC/ON/LAN '
                f'items between now and {_ref_short}. '
                f'Use the Target Opponents panel below for exact engine deltas.)'
                f'</span></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #3fb950;'
                f'border-left:4px solid #3fb950;border-radius:8px;'
                f'padding:12px 14px;font-size:13px;color:#c9d1d9;">'
                f"✅ No meaningful age-weight decay {_cp_when_phrase} — "
                f"this team's top-10 buckets are stable through the next reference cutoff.</div>",
                unsafe_allow_html=True,
            )

        # ── 5.2 Target opponents — engine sim per candidate ────
        st.markdown("#### 🎯 Target opponents — which wins move the needle?")
        # Candidates = the 10 teams immediately above this team by rank.
        # "Immediately above" maximises achievable Δrank per win; including
        # the very top teams adds compute for little marginal insight because
        # opp_bo_ratio already caps at 1.0 for any top-5 opponent.
        _cp_above = (base_standings[base_standings["rank"] < _cp_rank]
                     .sort_values("rank", ascending=False)
                     .head(10))

        if _cp_above.empty:
            st.info("This team is already rank #1 — no teams above to target.")
        else:
            if "_climb_targets_cache" not in st.session_state:
                st.session_state._climb_targets_cache = {}
            _cp_cache = st.session_state._climb_targets_cache
            _cp_ref_cut = _ref_cutoff  # defined earlier in Time Window section
            _cp_key = (mode_active, _cp_ref_cut.strftime("%Y%m%d"), sel_team)

            _btn_l, _btn_r = st.columns([1, 3])
            _run_clicked = _btn_l.button(
                "▶ Simulate climb plan",
                key=f"climb_btn_{sel_team}_{_cp_rank}",
                type="primary",
                help=(
                    f"Runs the engine {len(_cp_above)}× — once per candidate — with "
                    "a hypothetical BO3 win at a $250K LAN event on the cutoff date. "
                    "Takes ~10s; cached for the session."
                ),
            )
            _btn_r.caption(
                f"Evaluates {len(_cp_above)} candidate opponent(s) "
                f"(closest ranked above {sel_team})."
            )

            if _run_clicked and _cp_key not in _cp_cache:
                _cp_results: list[dict] = []
                _cp_sim_date = _cp_ref_cut  # same-day sim → no age-weight drift
                _cp_pool = 250_000.0
                # NOTE: named `_cp_prog` (not `_bar`) — the `_bar(...)` helper
                # defined above renders inline factor bars, reusing that name
                # would shadow it for the rest of the page.
                _cp_prog = st.progress(0.0, text="Simulating hypothetical LAN wins…")
                for i, opp_row in enumerate(
                    _cp_above.itertuples(index=False), start=1
                ):
                    _opp = str(opp_row.team)
                    _store_c = Store.from_valve(team_match_history, bo_prizes_map)
                    _store_c.append_simulation(
                        extra_matches=[{
                            "date":       _cp_sim_date,
                            "winner":     sel_team,
                            "loser":      _opp,
                            "prize_pool": _cp_pool,
                            "is_lan":     True,
                        }],
                        extra_prizes=None,
                    )
                    _res_c = run_vrs(_store_c, cutoff=_cp_sim_date)
                    _std_c = _res_c.get("standings", pd.DataFrame())
                    if (not _std_c.empty
                            and sel_team in _std_c["team"].values):
                        _row_c = _std_c[_std_c["team"] == sel_team].iloc[0]
                        _d_rank = int(_row_c["rank"]) - _cp_rank
                        _d_pts  = (float(_row_c["total_points"])
                                    - float(ex["total_points"]))
                        _ew = expected_win(float(ex["seed"]),
                                            float(opp_row.seed))
                        _cp_results.append({
                            "opponent":   _opp,
                            "opp_rank":   int(opp_row.rank),
                            "opp_points": float(opp_row.total_points),
                            "d_rank":     _d_rank,
                            "d_pts":      _d_pts,
                            "ew":         _ew,
                            "new_rank":   int(_row_c["rank"]),
                        })
                    _cp_prog.progress(i / len(_cp_above),
                                      text=f"Simulating {i}/{len(_cp_above)}…")
                _cp_prog.empty()
                _cp_cache[_cp_key] = _cp_results

            if _cp_key in _cp_cache:
                _cp_results = _cp_cache[_cp_key]
                if not _cp_results:
                    st.caption("No simulation results — engine returned no standings.")
                else:
                    # Sort: most negative Δrank first (biggest climb), then Δpts.
                    _cp_sorted = sorted(
                        _cp_results,
                        key=lambda r: (r["d_rank"], -r["d_pts"]),
                    )
                    _tg_rows = []
                    for r in _cp_sorted:
                        _rgain = -r["d_rank"]
                        _arrow = (
                            f'<span style="color:#3fb950;font-weight:700">'
                            f'▲ {_rgain}</span>'
                            if _rgain > 0 else
                            '<span style="color:#8b949e">—</span>'
                        )
                        _ewc = ("#3fb950" if r["ew"] >= 0.50 else
                                "#d2a8ff" if r["ew"] >= 0.30 else "#f85149")
                        _pts_c = "#3fb950" if r["d_pts"] > 0 else "#8b949e"
                        _tg_rows.append(
                            f'<tr style="border-bottom:1px solid #21262d;">'
                            f'<td style="padding:5px 8px;color:#c9d1d9;'
                            f'font-size:12px">'
                            f'<span style="color:#8b949e">#{r["opp_rank"]}</span> '
                            f'<strong>{r["opponent"]}</strong></td>'
                            f'<td style="padding:5px 8px;text-align:right;'
                            f'color:{_ewc};font-size:11px;font-weight:600">'
                            f'{r["ew"]*100:.0f}%</td>'
                            f'<td style="padding:5px 8px;text-align:right;'
                            f'color:{_pts_c};font-size:11px">'
                            f'{r["d_pts"]:+,.1f}</td>'
                            f'<td style="padding:5px 8px;text-align:right;'
                            f'color:#8b949e;font-size:11px">'
                            f'#{_cp_rank} → #{r["new_rank"]}</td>'
                            f'<td style="padding:5px 8px;text-align:right;'
                            f'font-size:11px">{_arrow}</td></tr>'
                        )
                    st.markdown(
                        '<div style="border:1px solid #30363d;border-radius:8px;'
                        'overflow:hidden;">'
                        '<table style="width:100%;border-collapse:collapse;">'
                        '<thead style="background:#161b22;color:#8b949e;'
                        'font-size:10px;text-transform:uppercase;">'
                        '<tr><th style="padding:7px 8px;text-align:left">Opponent</th>'
                        '<th style="padding:7px 8px;text-align:right">E(win)</th>'
                        '<th style="padding:7px 8px;text-align:right">Δ pts</th>'
                        '<th style="padding:7px 8px;text-align:right">Rank change</th>'
                        '<th style="padding:7px 8px;text-align:right">Δ rank</th></tr>'
                        '</thead><tbody>' + "".join(_tg_rows) + '</tbody></table></div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Each row simulates a single BO3 win at a $250K LAN event "
                        f"on {_cp_ref_cut.strftime('%b %d, %Y')}. E(win) is the "
                        "Glicko expected score from current seeds — "
                        "<span style='color:#3fb950'>green ≥ 50%</span>, "
                        "<span style='color:#d2a8ff'>purple ≥ 30%</span>, "
                        "<span style='color:#f85149'>red &lt; 30%</span>.",
                        unsafe_allow_html=True,
                    )

                    # ── 5.3 Path-to-rank checklist ────────────
                    if _cp_rank > 1:
                        _above_team_row = base_standings[
                            base_standings["rank"] == _cp_rank - 1
                        ]
                        if not _above_team_row.empty:
                            _above_name = str(_above_team_row.iloc[0]["team"])
                            _gap_pts = (
                                float(_above_team_row.iloc[0]["total_points"])
                                - float(ex["total_points"])
                                + 0.5  # tiny margin so we actually pass, not tie
                            )
                            st.markdown(
                                f"#### 🧗 Path to rank #{_cp_rank - 1} "
                                f"({_above_name})"
                            )
                            if _gap_pts <= 0:
                                st.success(
                                    f"Already level with rank #{_cp_rank - 1} — "
                                    "next simulation tick should promote this team."
                                )
                            else:
                                # Plausible = E(win) ≥ 30% AND Δpts > 0.
                                _plaus = sorted(
                                    [r for r in _cp_sorted
                                     if r["ew"] >= 0.30 and r["d_pts"] > 0],
                                    key=lambda r: r["d_pts"], reverse=True,
                                )
                                # Greedy: pick highest-yielding until sum ≥ gap.
                                _acc, _path = 0.0, []
                                for r in _plaus:
                                    _path.append(r)
                                    _acc += r["d_pts"]
                                    if _acc >= _gap_pts:
                                        break
                                if _acc >= _gap_pts and _path:
                                    _chk = "".join(
                                        f'<li style="margin-bottom:5px;color:#c9d1d9;">'
                                        f'Beat <strong>{r["opponent"]}</strong> '
                                        f'<span style="color:#8b949e">(#{r["opp_rank"]}, '
                                        f'E(win) {r["ew"]*100:.0f}%)</span> '
                                        f'→ <strong style="color:#3fb950">'
                                        f'+{r["d_pts"]:.1f} pts</strong></li>'
                                        for r in _path
                                    )
                                    st.markdown(
                                        f'<div style="background:#0d3d1a;'
                                        f'border:1px solid #3fb950;'
                                        f'border-radius:8px;padding:12px 16px;">'
                                        f'<div style="font-size:12px;color:#56d364;'
                                        f'margin-bottom:8px;">'
                                        f'✅ Minimum plausible path — '
                                        f'<strong>{len(_path)} win(s)</strong>, '
                                        f'total <strong>+{_acc:,.1f} pts</strong> '
                                        f'(need +{_gap_pts:,.1f}):</div>'
                                        f'<ul style="font-size:13px;margin:0;'
                                        f'padding-left:22px;line-height:1.6;">'
                                        f'{_chk}</ul></div>',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    _best_sum = sum(r["d_pts"] for r in _plaus)
                                    st.markdown(
                                        f'<div style="background:#161b22;'
                                        f'border:1px solid #f0b429;'
                                        f'border-left:4px solid #f0b429;'
                                        f'border-radius:8px;padding:12px 16px;'
                                        f'font-size:12px;color:#c9d1d9;line-height:1.55;">'
                                        f'⚠️ No single-LAN plausible path closes the '
                                        f'<strong>{_gap_pts:,.1f}-pt</strong> gap. '
                                        f'Best case from the candidate pool '
                                        f'(≥30% E(win)): <strong>+{_best_sum:,.1f} pts</strong>. '
                                        f'A bigger event ($1M LAN), running the table '
                                        f'at a major, or multiple-event stacking would '
                                        f'be needed.</div>',
                                        unsafe_allow_html=True,
                                    )

    # ═══════════════════════════════════════════════════════════════
    if _bd_tab_sel == "🔬 Score Breakdown":
    # ═══════════════════════════════════════════════════════════════

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
            _delta_str(ex["bo_factor"], _orig_ex["bo_factor"], "+.4f") if _orig_ex is not None else "",
            pts=ex["bo_factor"] * _pts_per_factor), unsafe_allow_html=True)
        # Counting summary (BO bucket is top-10 scaled prizes)
        _bo_scaled = []
        for bp in bo_prizes:
            if mode_active == "updated":
                try:
                    _dt = datetime.strptime(str(bp["event_date"]).strip(), "%Y-%m-%d")
                    _bo_scaled.append(bp["prize_won"] * age_weight(_dt, cutoff_dt))
                except Exception:
                    pass
            else:
                _bo_scaled.append(bp.get("scaled_prize", 0.0))
        _bo_scaled_sorted = sorted(_bo_scaled, reverse=True)
        _bo_top10_sum = sum(_bo_scaled_sorted[:10])
        _bo_below_sum = sum(_bo_scaled_sorted[10:])
        _bo_counting  = min(TOP_N, len([v for v in _bo_scaled_sorted if v > 0]))
        st.markdown(_count_caption(
            _bo_counting, len(bo_prizes), _bo_below_sum, _bo_top10_sum,
            item="prizes", color="#f0b429",
        ), unsafe_allow_html=True)
        col_bo_l, col_bo_r = st.columns([1, 1])
        with col_bo_l:
            ratio_bo = ex["bo_sum"] / max(ref5_bo, 1.0)
            st.markdown(_calc_box(
                f'<div>BO Sum (top-10 scaled wins) = <strong style="color:#f0b429">${ex["bo_sum"]:,.0f}</strong></div>' +
                f'<div>÷ 5th ref = ${ref5_bo:,.0f} → ratio = {ratio_bo:.4f}</div>' +
                f'<div>curve(min(1.0, {min(1.0,ratio_bo):.4f})) = <strong style="color:#f0b429">{ex["bo_factor"]:.4f}</strong></div>'
            ), unsafe_allow_html=True)
            st.markdown(_learn_cta("bo", "Bounty Offered", "#f0b429"), unsafe_allow_html=True)
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
                # Sort: scaled prize desc, then date desc as tiebreaker
                _disp_bo.sort(key=lambda bp: (bp["scaled_prize"], bp["event_date"]), reverse=True)
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
            _delta_str(ex["bc_factor"], _orig_ex["bc_factor"], "+.4f") if _orig_ex is not None else "",
            pts=ex["bc_factor"] * _pts_per_factor), unsafe_allow_html=True)
        # BC formula: bo_ratio (raw, not curve'd) × age_w × ev_w — prized events only.
        # Show ALL wins; zero ev_w entries appear faded at the bottom.
        bc_matches = [m for m in raw_ms if m["result"] == "W"]
        bc_entries = sorted(
            [(bo_ratio_map.get(m["opponent"], m.get("opp_bo", 0.0)) * _eff_age(m) * m.get("ev_w", 0.0),
              m,
              bo_ratio_map.get(m["opponent"], m.get("opp_bo", 0.0)))
             for m in bc_matches],
            key=lambda x: x[0], reverse=True
        )
        sum10_bc    = sum(e for e,_,_ in bc_entries[:10])
        _bc_below   = sum(e for e,_,_ in bc_entries[10:])
        _bc_counting = min(TOP_N, len([e for e,_,_ in bc_entries if e > 0]))
        st.markdown(_count_caption(
            _bc_counting, len(bc_matches), _bc_below, sum10_bc,
            item="wins", color="#3fb950",
        ), unsafe_allow_html=True)
        col_bc_l, col_bc_r = st.columns([1, 1])
        with col_bc_l:
            st.markdown(_calc_box(
                f'<div>Σ top-10 entries = <strong style="color:#3fb950">{sum10_bc:.4f}</strong></div>' +
                f'<div>BC_pre = {sum10_bc:.4f} / 10 = {sum10_bc/10:.4f}</div>' +
                f'<div>curve({sum10_bc/10:.4f}) = <strong style="color:#3fb950">{ex["bc_factor"]:.4f}</strong></div>'
            ), unsafe_allow_html=True)
            st.markdown(_learn_cta("bc", "Bounty Collected", "#3fb950"), unsafe_allow_html=True)
        with col_bc_r:
            if bc_entries:
                # Sort: entry desc, then date desc as tiebreaker
                bc_entries = sorted(bc_entries, key=lambda x: (x[0], x[1]["date"]), reverse=True)
                bc_html = "".join(
                    f'<tr style="border-bottom:1px solid #21262d;{" " if i<10 else "opacity:0.35;"}">' +
                    f'<td style="padding:4px 5px;color:#8b949e;font-size:10px">{m["date"].strftime("%m-%d")}</td>' +
                    f'<td style="padding:4px 5px;color:#c9d1d9;font-size:10px">{m["opponent"][:14]}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(ob,"#f0b429",30,_bd_bo_max)}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(_eff_age(m),"#e6b430",30)}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(m["ev_w"],"#79c0ff",30)}</td>' +
                    f'<td style="padding:4px 5px;text-align:right;color:#3fb950;font-size:10px;font-weight:600">{e:.3f}</td>' +
                    (f'<td style="padding:4px 5px;text-align:right;color:#8b949e;font-size:10px">{round(e/10*_pts_per_factor):+,d}</td>' if i<10 else
                     f'<td style="padding:4px 5px"></td>') +
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
                    '<th style="padding:5px;text-align:right">Pts</th>' +
                    '</tr></thead><tbody>' + bc_html + '</tbody>' +
                    f'<tfoot><tr style="background:#161b22;border-top:1px solid #30363d;">' +
                    f'<td colspan="5" style="padding:5px;color:#8b949e;font-size:10px">Top-10 sum/10</td>' +
                    f'<td style="padding:5px;text-align:right;color:#3fb950;font-weight:700">{sum10_bc/10:.4f}</td>' +
                    f'<td style="padding:5px;text-align:right;color:#8b949e;font-size:10px">≈ {round(ex["bc_factor"]*_pts_per_factor):,d}</td>' +
                    '</tr></tfoot></table></div>' +
                    '<div style="font-size:9px;color:#484f58;margin-top:2px">Faded = outside top-10 · Pts = linear contribution to final VRS score</div>',
                    unsafe_allow_html=True
                )
            else:
                st.caption("No event wins found.")

        st.markdown("<div style='margin:10px 0'>", unsafe_allow_html=True)

        # ── OPPONENT NETWORK ──────────────────────────────────────
        st.markdown(_factor_band("🕸️", "Opponent Network", ex["on_factor"], "#79c0ff", _bd_on_max,
            _delta_str(ex["on_factor"], _orig_ex["on_factor"], "+.4f") if _orig_ex is not None else "",
            pts=ex["on_factor"] * _pts_per_factor), unsafe_allow_html=True)
        # ON formula: opp.own_network × age_w × ev_w (from Valve's team.js).
        # Primary source: m["opp_on"] — Valve's pre-stored own_network (As Published).
        # Fallback: _on_prev_replay — recomputed own_network (Updated/Sim mode).
        on_matches = [m for m in raw_ms if m["result"] == "W"]
        on_entries = sorted(
            [
                (
                    (m.get("opp_on", 0.0)
                     if m.get("opp_on", 0.0) > 0
                     else _on_prev_replay.get(m["opponent"], 0.0))
                    * _eff_age(m) * m.get("ev_w", 0.0),
                    m,
                    m.get("opp_on", 0.0)
                    if m.get("opp_on", 0.0) > 0
                    else _on_prev_replay.get(m["opponent"], 0.0),
                )
                for m in on_matches
            ],
            key=lambda x: (x[0], x[1]["date"]), reverse=True
        )
        sum10_on    = sum(e for e,_,_ in on_entries[:10])
        _on_below   = sum(e for e,_,_ in on_entries[10:])
        _on_counting = min(TOP_N, len([e for e,_,_ in on_entries if e > 0]))
        st.markdown(_count_caption(
            _on_counting, len(on_matches), _on_below, sum10_on,
            item="wins", color="#79c0ff",
        ), unsafe_allow_html=True)
        col_on_l, col_on_r = st.columns([1, 1])
        with col_on_l:
            st.markdown(_calc_box(
                f'<div>opp.own_network × age_w × ev_w  (top-10 / 10)</div>' +
                f'<div style="margin-top:6px">Σ top-10 entries = <strong style="color:#79c0ff">{sum10_on:.4f}</strong></div>' +
                f'<div>ON = {sum10_on:.4f} / 10 = <strong style="color:#79c0ff">{ex["on_factor"]:.4f}</strong> (no curve)</div>'
            ), unsafe_allow_html=True)
            st.markdown(_learn_cta("on", "Opponent Network", "#79c0ff"), unsafe_allow_html=True)
        with col_on_r:
            if on_entries:
                on_html = "".join(
                    f'<tr style="border-bottom:1px solid #21262d;{" " if i<10 else "opacity:0.35;"}">' +
                    f'<td style="padding:4px 5px;color:#8b949e;font-size:10px">{m["date"].strftime("%m-%d")}</td>' +
                    f'<td style="padding:4px 5px;color:#c9d1d9;font-size:10px">{m["opponent"][:14]}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(on_v,"#79c0ff",30,_bd_on_max)}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(_eff_age(m),"#e6b430",30)}</td>' +
                    f'<td style="padding:4px 5px;font-size:10px">{_bar(m.get("ev_w",0.0),"#79c0ff",30)}</td>' +
                    f'<td style="padding:4px 5px;text-align:right;color:#79c0ff;font-size:10px;font-weight:600">{e:.3f}</td>' +
                    (f'<td style="padding:4px 5px;text-align:right;color:#8b949e;font-size:10px">{round(e/10*_pts_per_factor):+,d}</td>' if i<10 else
                     f'<td style="padding:4px 5px"></td>') +
                    f'</tr>'
                    for i,(e,m,on_v) in enumerate(on_entries)
                )
                st.markdown(
                    '<div style="overflow-y:auto;max-height:180px;border:1px solid #30363d;border-radius:6px;">' +
                    '<table style="width:100%;border-collapse:collapse;">' +
                    '<thead><tr style="background:#161b22;color:#8b949e;font-size:9px;text-transform:uppercase;">' +
                    '<th style="padding:5px">Date</th><th style="padding:5px">Opp</th>' +
                    '<th style="padding:5px">ON</th><th style="padding:5px">Age</th>' +
                    '<th style="padding:5px">Ev</th>' +
                    '<th style="padding:5px;text-align:right">Entry</th>' +
                    '<th style="padding:5px;text-align:right">Pts</th>' +
                    '</tr></thead><tbody>' + on_html + '</tbody>' +
                    f'<tfoot><tr style="background:#161b22;border-top:1px solid #30363d;">' +
                    f'<td colspan="5" style="padding:5px;color:#8b949e;font-size:10px">Top-10 sum/10</td>' +
                    f'<td style="padding:5px;text-align:right;color:#79c0ff;font-weight:700">{sum10_on/10:.4f}</td>' +
                    f'<td style="padding:5px;text-align:right;color:#8b949e;font-size:10px">≈ {round(ex["on_factor"]*_pts_per_factor):,d}</td>' +
                    '</tr></tfoot></table></div>' +
                    '<div style="font-size:9px;color:#484f58;margin-top:2px">Faded = outside top-10 · Pts = linear contribution to final VRS score</div>',
                    unsafe_allow_html=True
                )
            else:
                st.caption("No wins found.")

        st.markdown("<div style='margin:10px 0'>", unsafe_allow_html=True)

        # ── LAN WINS ──────────────────────────────────────────────
        st.markdown(_factor_band("🖥️", "LAN Wins", ex["lan_factor"], "#f85149", _bd_lan_max,
            _delta_str(ex["lan_factor"], _orig_ex["lan_factor"], "+.4f") if _orig_ex is not None else "",
            pts=ex["lan_factor"] * _pts_per_factor), unsafe_allow_html=True)
        lan_wins_list = sorted(
            [m for m in raw_ms if m["result"] == "W" and m.get("is_lan")],
            key=lambda m: _eff_age(m), reverse=True
        )
        sum10_lan = sum(_eff_age(m) for m in lan_wins_list[:10])
        _lan_below  = sum(_eff_age(m) for m in lan_wins_list[10:])
        _lan_counting = min(TOP_N, len([m for m in lan_wins_list if _eff_age(m) > 0]))
        st.markdown(_count_caption(
            _lan_counting, len(lan_wins_list), _lan_below, sum10_lan,
            item="LAN wins", color="#f85149",
        ), unsafe_allow_html=True)
        col_lan_l, col_lan_r = st.columns([1, 1])
        with col_lan_l:
            st.markdown(_calc_box(
                f'<div>LAN wins in window: <strong style="color:#f85149">{len(lan_wins_list)}</strong></div>' +
                f'<div>Top-10 age weights sum = <strong>{sum10_lan:.4f}</strong></div>' +
                f'<div>LAN = {sum10_lan:.4f} / 10 = <strong style="color:#f85149">{ex["lan_factor"]:.4f}</strong> (no curve)</div>'
            ), unsafe_allow_html=True)
            st.markdown(_learn_cta("lan", "LAN Wins", "#f85149"), unsafe_allow_html=True)
        with col_lan_r:
            if lan_wins_list:
                # Sort: age weight desc, then date desc as tiebreaker
                lan_wins_list = sorted(
                    lan_wins_list,
                    key=lambda m: (_eff_age(m), m["date"]),
                    reverse=True
                )
                lan_html = "".join(
                    f'<tr style="border-bottom:1px solid #21262d;{" " if i<10 else "opacity:0.35;"}">' +
                    f'<td style="padding:4px 6px;color:#8b949e;font-size:11px">{m["date"].strftime("%Y-%m-%d")}</td>' +
                    f'<td style="padding:4px 6px;color:#c9d1d9;font-size:11px">{m["opponent"][:16]}</td>' +
                    f'<td style="padding:4px 6px">{_bar(_eff_age(m),"#f85149",55)}</td>' +
                    (f'<td style="padding:4px 6px;text-align:right;color:#8b949e;font-size:10px">{round(_eff_age(m)/10*_pts_per_factor):+,d}</td>' if i<10 else
                     f'<td style="padding:4px 6px"></td>') +
                    f'</tr>'
                    for i,m in enumerate(lan_wins_list)
                )
                st.markdown(
                    '<div style="overflow-y:auto;max-height:180px;border:1px solid #30363d;border-radius:6px;">' +
                    '<table style="width:100%;border-collapse:collapse;">' +
                    '<thead><tr style="background:#161b22;color:#8b949e;font-size:9px;text-transform:uppercase;">' +
                    '<th style="padding:6px">Date</th><th style="padding:6px">Opponent</th>' +
                    '<th style="padding:6px">Age Wt</th><th style="padding:6px;text-align:right">Pts</th>' +
                    '</tr></thead><tbody>' + lan_html + '</tbody>' +
                    f'<tfoot><tr style="background:#161b22;border-top:1px solid #30363d;">' +
                    f'<td colspan="2" style="padding:5px 6px;color:#8b949e;font-size:10px">Top-10 sum/10</td>' +
                    f'<td style="padding:5px 6px;color:#f85149;font-weight:700">{sum10_lan/10:.4f}</td>' +
                    f'<td style="padding:5px 6px;text-align:right;color:#8b949e;font-size:10px">≈ {round(ex["lan_factor"]*_pts_per_factor):,d}</td>' +
                    '</tr></tfoot></table></div>' +
                    '<div style="font-size:9px;color:#484f58;margin-top:2px">Faded = outside top-10 · Pts = linear contribution to final VRS score</div>',
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

        # Two-column layout: narrow chart on the left, wide match table on the right.
        col_h2h_chart, col_h2h_table = st.columns([1, 3])

        with col_h2h_chart:
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
                margin=dict(l=10,r=10,t=40,b=10), height=280,
            )
            st.plotly_chart(fig_p2, use_container_width=True, config={"staticPlot": True})

        # All matches for H2H (sorted by date desc) — rendered in the right column.
        with col_h2h_table:
            if raw_ms:
                # Filter to matches in the sim window if active
                _h2h_ms = raw_ms
                if mode_active == "updated":
                    _new_ws = sim_cutoff_dt - timedelta(days=DECAY_DAYS)
                    _h2h_ms = [m for m in raw_ms if _new_ws <= m["date"] <= sim_cutoff_dt]

                # Pre-compute h2h values so we can sort before rendering
                total_h2h = 0.0
                _h2h_computed = []
                for m in _h2h_ms:
                    h2h     = _eff_h2h(m, sel_team)
                    aw_disp = _eff_age(m)
                    total_h2h += h2h
                    _h2h_computed.append((m, h2h, aw_disp))

                # Sort: date desc (most recent first) for the table render
                _h2h_computed.sort(key=lambda x: x[0]["date"], reverse=True)

                # Opp Rating + Win % columns render in both modes (sourced from
                # pub_match_h2h in Published mode, from sim_match_h2h in Updated
                # mode). The Pub Δ column goes away in favour of inline delta
                # chips on the three columns that actually shifted: Opp Rating,
                # Win %, and H2H Δ.
                _is_updated     = (mode_active == "updated")
                _show_deltas    = _is_updated and bool(pub_match_h2h) and bool(sim_match_h2h)
                _show_glicko_cols = bool(pub_match_h2h) or bool(sim_match_h2h)

                # ── Cumulative H2H delta sparkline (chronological ascending) ─
                if _h2h_computed:
                    _asc = sorted(_h2h_computed, key=lambda x: x[0]["date"])
                    _dates_asc = [x[0]["date"] for x in _asc]
                    _cum = []
                    _running = 0.0
                    for _, _h, _ in _asc:
                        _running += _h
                        _cum.append(_running)
                    _end_color = "#3fb950" if (_cum and _cum[-1] >= 0) else "#f85149"
                    fig_cum = _go.Figure()
                    fig_cum.add_trace(_go.Scatter(
                        x=_dates_asc, y=_cum, mode="lines",
                        line=dict(color=_end_color, width=2),
                        fill="tozeroy",
                        fillcolor=("rgba(63,185,80,0.15)" if _cum and _cum[-1] >= 0 else "rgba(248,81,73,0.15)"),
                        hovertemplate="%{x|%Y-%m-%d}<br>cum Δ: %{y:+.1f}<extra></extra>",
                    ))
                    fig_cum.add_hline(y=0, line_color="#30363d", line_width=1)
                    fig_cum.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#8b949e", size=10),
                        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=9)),
                        yaxis=dict(gridcolor="#21262d", zeroline=False, tickfont=dict(size=9),
                                   title=dict(text="cum H2H Δ", font=dict(size=10, color="#8b949e"))),
                        margin=dict(l=10, r=10, t=8, b=20), height=110,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})

                # ── Match table ─────────────────────────────────────────────
                _empty_cell = '<td style="padding:4px 7px;text-align:right;color:#484f58;font-size:10px">—</td>'

                m_rows = []
                for m, h2h, aw_disp in _h2h_computed:
                    is_win  = m["result"] == "W"
                    res_b   = (
                        '<span style="background:#1f4a1f;color:#3fb950;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700">W</span>'
                        if is_win else
                        '<span style="background:#4a1f1f;color:#f85149;border-radius:3px;padding:1px 6px;font-size:10px;font-weight:700">L</span>'
                    )

                    # ── Pull Glicko replay state (pub + sim as available) ──
                    mid       = m.get("match_id")
                    pub_entry = pub_match_h2h.get(mid) if pub_match_h2h else None
                    sim_entry = sim_match_h2h.get(mid) if (_is_updated and sim_match_h2h) else None
                    # Choose the "primary" entry shown in the cell: Updated mode
                    # prefers sim, Published mode prefers pub.
                    primary = sim_entry if _is_updated and sim_entry is not None else pub_entry

                    def _opp_rating(entry):
                        return entry["w_rating"] if not is_win else entry["l_rating"]

                    def _e_self(entry):
                        e_w = float(entry.get("e_w", 0.5))
                        return e_w if is_win else (1.0 - e_w)

                    # Opp Rating cell
                    if primary is not None:
                        opp_rating = _opp_rating(primary)
                        rating_chip = ""
                        if _show_deltas and pub_entry is not None and sim_entry is not None:
                            rating_chip = _delta_chip(
                                _opp_rating(sim_entry) - _opp_rating(pub_entry),
                                fmt="+.0f", threshold=1.0,
                            )
                        rating_td = (
                            f'<td style="padding:4px 7px;text-align:right;color:#c9d1d9;font-size:10px;white-space:nowrap">'
                            f'{opp_rating:,.0f}{rating_chip}</td>'
                        )
                    else:
                        rating_td = _empty_cell

                    # Win % cell
                    if primary is not None:
                        e_self = _e_self(primary)
                        if is_win:
                            _e_color = "#3fb950" if e_self < 0.40 else ("#c9d1d9" if e_self < 0.70 else "#8b949e")
                        else:
                            _e_color = "#f85149" if e_self > 0.60 else ("#c9d1d9" if e_self > 0.30 else "#8b949e")
                        win_chip = ""
                        if _show_deltas and pub_entry is not None and sim_entry is not None:
                            win_chip = _delta_chip(
                                (_e_self(sim_entry) - _e_self(pub_entry)) * 100.0,
                                fmt="+.1f", unit="%", threshold=0.5,
                            )
                        win_td = (
                            f'<td style="padding:4px 7px;text-align:right;color:{_e_color};font-size:10px;white-space:nowrap">'
                            f'{e_self*100:.0f}%{win_chip}</td>'
                        )
                    else:
                        win_td = _empty_cell

                    # H2H Δ cell (effective value + delta chip vs Valve's stored pub value)
                    h2h_c = (
                        f'<span style="color:#3fb950;font-weight:700">+{h2h:.1f}</span>' if h2h > 0 else
                        f'<span style="color:#f85149;font-weight:700">{h2h:.1f}</span>' if h2h < 0 else
                        '<span style="color:#8b949e">0.0</span>'
                    )
                    h2h_chip = ""
                    if _show_deltas:
                        # Prefer our engine's pub value (match_h2h) for a like-for-like delta;
                        # fall back to Valve's stored per-match h2h_adj for pre-cutoff matches
                        # that we couldn't recompute.
                        if pub_entry is not None:
                            prev_h2h = pub_entry["w_delta"] if is_win else pub_entry["l_delta"]
                        else:
                            prev_h2h = float(m.get("h2h_adj", 0.0) or 0.0)
                        h2h_chip = _delta_chip(h2h - prev_h2h, fmt="+.1f", threshold=0.1)
                    h2h_td = (
                        f'<td style="padding:4px 7px;text-align:right;white-space:nowrap">'
                        f'{h2h_c}{h2h_chip}</td>'
                    )

                    m_rows.append(
                        f'<tr style="border-bottom:1px solid #21262d;">' +
                        f'<td style="padding:4px 7px;color:#8b949e;font-size:11px">{m["date"].strftime("%Y-%m-%d")}</td>' +
                        f'<td style="padding:4px 7px">{res_b}</td>' +
                        f'<td style="padding:4px 7px;color:#c9d1d9;font-size:11px">{m["opponent"]}</td>' +
                        f'<td style="padding:4px 7px;text-align:center;font-size:11px">{"🖥️" if (is_win and m.get("is_lan")) else "🌐" if is_win else ""}</td>' +
                        f'<td style="padding:4px 7px">{_bar(aw_disp,"#f0b429",45)}</td>' +
                        (rating_td if _show_glicko_cols else "") +
                        (win_td    if _show_glicko_cols else "") +
                        h2h_td +
                        f'</tr>'
                    )
                n_w = sum(1 for m in _h2h_ms if m["result"] == "W")
                n_l = len(_h2h_ms) - n_w
                h2h_col = "#3fb950" if total_h2h >= 0 else "#f85149"
                _glicko_th = (
                    '<th style="padding:7px;text-align:right" title="Opponent Glicko rating at the time of this match">Opp Rating</th>' +
                    '<th style="padding:7px;text-align:right" title="Expected win probability at the time of the match">Win %</th>'
                ) if _show_glicko_cols else ""
                st.markdown(
                    '<div style="overflow-y:auto;max-height:340px;border:1px solid #30363d;border-radius:8px;">' +
                    '<table style="width:100%;border-collapse:collapse;">' +
                    '<thead style="position:sticky;top:0;background:#161b22;">' +
                    '<tr style="color:#8b949e;font-size:9px;text-transform:uppercase;">' +
                    '<th style="padding:7px">Date</th><th style="padding:7px">W/L</th>' +
                    '<th style="padding:7px;text-align:left">Opponent</th>' +
                    '<th style="padding:7px">Type</th><th style="padding:7px">Age Wt</th>' +
                    _glicko_th +
                    '<th style="padding:7px;text-align:right">H2H Δ</th>' +
                    '</tr></thead><tbody>' + "".join(m_rows) + '</tbody></table></div>' +
                    f'<div style="display:flex;justify-content:space-between;margin-top:5px;font-size:11px;color:#8b949e;">' +
                    f'<span>{len(_h2h_ms)} matches · {n_w}W {n_l}L</span>' +
                    f'<span>Total H2H: <strong style="color:{h2h_col}">{total_h2h:+.1f} pts</strong></span></div>',
                    unsafe_allow_html=True
                )

            else:
                st.info("No match history available.")

    # ═══════════════════════════════════════════════════════════════
    if _bd_tab_sel == "📈 Historical Development":
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
