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
# ████████████  VRS CALCULATION ENGINE  v3  ████████████████████████
# ══════════════════════════════════════════════════════════════════
#
# All formulas verified against the official Vitality March-2026
# breakdown published in the Valve repository.
#
# KEY CORRECTIONS vs v2:
#   1. Time decay: flat 1.0 for first 30 days, linear 1.0→0.0 over
#      the following 150 days  (NOT a simple 0→1 remap over 180 days)
#   2. BO: prize × age_weight (no event stakes); normalised with
#      curve(min(1.0, sum / 5th_team_sum))
#   3. BC: uses opponent's BO factor × age × event_stakes, then
#      curve(sum_top10 / 10)  — curve IS applied
#   4. ON: uses opponent's ON factor × age × event_stakes, then
#      sum_top10 / 10  — NO curve; iterative (PageRank-like)
#   5. LAN: 1.0 × age_weight per LAN win; sum_top10 / 10; NO curve,
#      NO event stakes; no LAN multiplier anywhere in H2H
#   6. Seeding: simple unweighted average of the 4 factors (25% each)
#   7. H2H K = BASE_K × age_weight ONLY (no event stakes, no LAN mult)
# ══════════════════════════════════════════════════════════════════

# ── Constants ──────────────────────────────────────────────────────
DECAY_DAYS  = 180           # Total lookback window (days)
FLAT_DAYS   = 30            # Days at the start that hold 1.0 age weight
DECAY_RAMP  = 150           # Days over which weight ramps 1.0 → 0.0
RD_FIXED    = 75            # Fixed Glicko RD → Elo-equivalent behaviour
Q_GLICKO    = math.log(10) / 400   # ≈ 0.005756
BASE_K      = 32            # H2H base K-factor (scaled by age only)
PRIZE_CAP   = 1_000_000     # Prize pool cap for event stakes (USD)
TOP_N       = 10            # Bucket size — top-N results per factor
ON_ITERS    = 6             # PageRank iterations for Opponent Network

SEED_MIN = 400
SEED_MAX = 2000


# ── Core maths ────────────────────────────────────────────────────

def curve(x: float) -> float:
    """
    Valve's normalisation curve:  f(x) = 1 / (1 + |log₁₀(x)|)

    · f(1.0)  = 1.000  — exactly at the reference point
    · f(0.1)  = 0.500  — one order of magnitude below
    · f(0.01) = 0.333  — two orders of magnitude below
    · f(10)   = 0.500  — one order of magnitude above
    · Always in (0, 1] for x > 0; peaks at x = 1
    """
    if x <= 0:
        return 0.0
    return 1.0 / (1.0 + abs(math.log10(x)))


def event_stakes(prize_pool: float) -> float:
    """
    Event weight applied to BC and ON calculations:
      stakes = curve(pool / $1,000,000)

    A $1M event  → stakes = curve(1.0) = 1.000
    A $250k event→ stakes = curve(0.25) ≈ 0.602
    A $100k event→ stakes = curve(0.1)  = 0.500
    Applied to:  BC, ON   (NOT to BO or LAN)
    """
    ratio = min(max(prize_pool, 1.0), PRIZE_CAP) / PRIZE_CAP
    return curve(ratio)


def age_weight(match_date: datetime, cutoff: datetime) -> float:
    """
    Age Weight (Time Modifier) — verified against official Vitality data.

    · Days 0–30 before cutoff: weight = 1.000 (flat)
    · Days 31–180:             weight = 1.0 – (days_ago – 30) / 150
    · Beyond 180 days:         excluded (weight = 0.0)

    Examples at cutoff 2026-03-02:
      2026-01-31 (30 days):  1.000  ✓
      2026-01-24 (37 days):  0.953  ✓
      2025-12-14 (78 days):  0.680  ✓
      2025-11-09 (113 days): 0.447  ✓
      2025-10-12 (141 days): 0.260  ✓
      2025-09-07 (176 days): 0.027  ✓
    """
    days_ago = (cutoff - match_date).days
    if days_ago < 0:
        return 1.0          # future match (should not occur)
    if days_ago <= FLAT_DAYS:
        return 1.0
    if days_ago >= DECAY_DAYS:
        return 0.0
    return 1.0 - (days_ago - FLAT_DAYS) / DECAY_RAMP


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + (b–a)×t, t clamped to [0, 1]."""
    return a + (b - a) * max(0.0, min(1.0, t))


# ── Glicko helpers ─────────────────────────────────────────────────

def g_rd(rd: float = RD_FIXED) -> float:
    """
    Glicko g(RD) dampening factor.
    With RD fixed at 75 this is constant ≈ 0.9728.
    """
    return 1.0 / math.sqrt(1.0 + 3.0 * Q_GLICKO**2 * rd**2 / math.pi**2)


G_FIXED = g_rd(RD_FIXED)   # pre-computed; constant for the entire run


def expected_win(r_self: float, r_opp: float) -> float:
    """
    Glicko expected score for r_self against r_opp:
      E = 1 / (1 + 10^(−g(RD) × (r_self − r_opp) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** (-G_FIXED * (r_self - r_opp) / 400.0))


# ── Helper: top-N sum ───────────────────────────────────────────────

def top_n_sum(values: list, n: int = TOP_N) -> float:
    """Return the sum of the n largest values in the list."""
    return float(sum(sorted(values, reverse=True)[:n]))


# ══════════════════════════════════════════════════════════════════
# MAIN VRS COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_vrs(matches_df: pd.DataFrame, cutoff: datetime = None) -> pd.DataFrame:
    """
    Full two-phase VRS computation (v3, formula-verified).

    Input DataFrame columns:
        date         – datetime
        winner       – str
        loser        – str
        prize_pool   – float  (USD, event total prize pool; 0 if none)
        winner_prize – float  (USD earned by the winner)
        loser_prize  – float  (USD earned by the loser)
        is_lan       – bool

    Returns one row per eligible team with full score breakdown.
    Ineligible teams (0 wins OR <5 total matches) are excluded.
    """
    if cutoff is None:
        cutoff = datetime.now()

    window_start = cutoff - timedelta(days=DECAY_DAYS)

    # ── Filter window, sort chronologically ──────────────────────
    df = matches_df[
        (matches_df["date"] >= window_start) &
        (matches_df["date"] <= cutoff)
    ].copy().sort_values("date").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    # ── Per-match derived values ──────────────────────────────────
    df["age_w"]    = df["date"].apply(lambda d: age_weight(d, cutoff))
    df["ev_w"]     = df["prize_pool"].apply(event_stakes)   # for BC + ON only
    df["has_prize"] = df["prize_pool"] > 0                  # event-weight gate

    all_teams = sorted(set(df["winner"].tolist() + df["loser"].tolist()))

    # ── Eligibility (pre-filter) ──────────────────────────────────
    match_counts = {
        t: int(len(df[(df["winner"] == t) | (df["loser"] == t)]))
        for t in all_teams
    }
    wins_counts = {
        t: int((df["winner"] == t).sum())
        for t in all_teams
    }
    eligible = [
        t for t in all_teams
        if wins_counts.get(t, 0) > 0          # ≥1 win in window
        and match_counts.get(t, 0) >= 5       # ≥5 matches in window
    ]
    if not eligible:
        return pd.DataFrame()

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 — SEEDING
    # ══════════════════════════════════════════════════════════════

    # ── Factor 1: Bounty Offered ───────────────────────────────────
    # Each win contributes:  winner_prize × age_weight
    # (NO event stakes — BO does not use event weight)
    # Top-10 such contributions are summed into bo_sum.
    # Normalised: curve(min(1.0, bo_sum / 5th_team_bo_sum))
    # Teams ranked 1-5 by prize money all receive BO = 1.000.

    bo_sum: dict[str, float] = {}
    for team in all_teams:
        wins = df[df["winner"] == team]
        if wins.empty:
            bo_sum[team] = 0.0
        else:
            contribs = wins["winner_prize"] * wins["age_w"]
            bo_sum[team] = float(contribs.nlargest(TOP_N).sum())

    # 5th-highest BO sum as the normalisation reference
    sorted_bo_sums = sorted(bo_sum.values(), reverse=True)
    ref_5th = sorted_bo_sums[4] if len(sorted_bo_sums) >= 5 else (
              sorted_bo_sums[-1] if sorted_bo_sums else 1.0)
    ref_5th = max(ref_5th, 1e-9)

    # BO factor for every team (eligible and ineligible — needed for BC/ON)
    bo_factor: dict[str, float] = {
        t: curve(min(1.0, bo_sum[t] / ref_5th))
        for t in all_teams
    }

    # ── Factor 2: Bounty Collected ─────────────────────────────────
    # Each win (at an event with prize pool) contributes:
    #   opponent_BO_factor × age_weight × event_weight
    # Take top-10 such values, sum them, divide by 10, apply curve.
    # BC = curve( Σ_top10(opp_BO × age × ev_stakes) / 10 )

    bc_factor: dict[str, float] = {}
    bc_sum_raw: dict[str, float] = {}   # pre-curve sum/10, for display
    for team in eligible:
        wins_ev = df[(df["winner"] == team) & df["has_prize"]]
        if wins_ev.empty:
            bc_factor[team] = 0.0
            bc_sum_raw[team] = 0.0
            continue
        entries = [
            bo_factor.get(row["loser"], 0.0) * row["age_w"] * row["ev_w"]
            for _, row in wins_ev.iterrows()
        ]
        s = top_n_sum(entries) / TOP_N
        bc_sum_raw[team] = s
        bc_factor[team]  = curve(s)

    # ── Factor 3: Opponent Network ──────────────────────────────────
    # Each win (at an event with prize pool) contributes:
    #   opponent_ON_factor × age_weight × event_weight
    # Take top-10, sum, divide by 10.  NO curve applied.
    # ON = Σ_top10(opp_ON × age × ev_stakes) / 10
    #
    # Circular dependency → iterate (PageRank-style).
    # Initialise with BO factors as the first estimate of each team's
    # "network value", then update ON_ITERS times.

    on_factor: dict[str, float] = dict(bo_factor)   # seed estimate

    for _iter in range(ON_ITERS):
        new_on: dict[str, float] = {}
        for team in eligible:
            wins_ev = df[(df["winner"] == team) & df["has_prize"]]
            if wins_ev.empty:
                new_on[team] = 0.0
                continue
            entries = [
                on_factor.get(row["loser"], 0.0) * row["age_w"] * row["ev_w"]
                for _, row in wins_ev.iterrows()
            ]
            new_on[team] = top_n_sum(entries) / TOP_N
        # Update only eligible teams; ineligible keep their BO seed
        on_factor.update(new_on)

    on_factor_final: dict[str, float] = {t: on_factor.get(t, 0.0) for t in eligible}

    # ── Factor 4: LAN Wins ─────────────────────────────────────────
    # Each LAN win contributes: 1.0 × age_weight
    # (no event stakes, no curve — pure time-weighted count)
    # LAN = Σ_top10(1.0 × age) / 10

    lan_factor: dict[str, float] = {}
    lan_wins_ct: dict[str, int] = {}
    for team in eligible:
        lan_wins = df[(df["winner"] == team) & df["is_lan"]]
        lan_wins_ct[team] = len(lan_wins)
        if lan_wins.empty:
            lan_factor[team] = 0.0
        else:
            entries = lan_wins["age_w"].tolist()
            lan_factor[team] = top_n_sum(entries) / TOP_N

    # ── Combine → Seed ─────────────────────────────────────────────
    # Simple unweighted average of the four factor values (25% each).
    # Then lerp to [400, 2000] via min-max across eligible teams.

    combined: dict[str, float] = {
        t: (bo_factor[t] + bc_factor[t] + on_factor_final[t] + lan_factor[t]) / 4.0
        for t in eligible
    }

    avg_vals = list(combined.values())
    min_avg  = min(avg_vals)
    max_avg  = max(avg_vals)
    span_avg = max(max_avg - min_avg, 1e-9)

    seeds: dict[str, float] = {
        t: lerp(SEED_MIN, SEED_MAX, (combined[t] - min_avg) / span_avg)
        for t in eligible
    }

    # ══════════════════════════════════════════════════════════════
    # PHASE 2 — HEAD-TO-HEAD  (Glicko / Elo, chronological)
    # ══════════════════════════════════════════════════════════════
    # K = BASE_K × age_weight  ONLY.
    # No event stakes, no LAN multiplier — confirmed from official data.
    # Matches are processed oldest-first so ratings evolve naturally.

    ratings:   dict[str, float] = {t: seeds[t] for t in eligible}
    h2h_delta: dict[str, float] = {t: 0.0      for t in eligible}

    for _, row in df.iterrows():
        w, l = str(row["winner"]), str(row["loser"])
        if w not in ratings or l not in ratings:
            continue

        r_w, r_l = ratings[w], ratings[l]
        E_w = expected_win(r_w, r_l)   # E_l = 1 – E_w implicitly

        K = BASE_K * float(row["age_w"])

        d_w =  K * (1.0 - E_w)          # winner always gains
        d_l =  K * (0.0 - (1.0 - E_w))  # loser always loses (= –K × E_w)

        ratings[w]   += d_w
        ratings[l]   += d_l
        h2h_delta[w] += d_w
        h2h_delta[l] += d_l

    # ── Build output DataFrame ─────────────────────────────────────
    records = []
    for t in eligible:
        seed  = seeds[t]
        h2h   = h2h_delta[t]
        total = seed + h2h

        records.append({
            "team":             t,
            "total_points":     round(total, 1),
            "seed":             round(seed,  1),
            "h2h_delta":        round(h2h,   1),
            # Raw factor scores (pre-combination)
            "bo_sum":           round(bo_sum.get(t, 0.0),        2),
            "bo_factor":        round(bo_factor.get(t, 0.0),     4),
            "bc_pre_curve":     round(bc_sum_raw.get(t, 0.0),    4),
            "bc_factor":        round(bc_factor.get(t, 0.0),     4),
            "on_factor":        round(on_factor_final.get(t,0.0),4),
            "lan_factor":       round(lan_factor.get(t, 0.0),    4),
            "lan_wins":         int(lan_wins_ct.get(t, 0)),
            # Combined average (input to lerp)
            "seed_combined":    round(combined.get(t, 0.0),      4),
            # Match counts
            "wins":             int(wins_counts.get(t, 0)),
            "losses":           int(match_counts.get(t, 0) - wins_counts.get(t, 0)),
            "total_matches":    int(match_counts.get(t, 0)),
        })

    result = (pd.DataFrame(records)
              .sort_values("total_points", ascending=False)
              .reset_index(drop=True))
    result["rank"] = result.index + 1
    return result


# ══════════════════════════════════════════════════════════════════
# ████████████  GITHUB DATA LOADER  ████████████████████████████████
# ══════════════════════════════════════════════════════════════════
#
# Fetches live data from Valve's public GitHub repo.
# All 300-400 team detail files are fetched in parallel (20 workers)
# and cached for 1 hour via @st.cache_data.
#
# Data flow:
#   1. GitHub API  → discover latest standings date
#   2. standings_global_*.md → get team list + detail file paths  
#   3. Parallel fetch all detail *.md files (ThreadPoolExecutor)
#   4. Parse each file:
#      · Pre-computed factor values (Valve's exact numbers)
#      · Match history (for H2H simulation)
# ══════════════════════════════════════════════════════════════════

import json
import re as _re
from concurrent.futures import ThreadPoolExecutor, as_completed

GITHUB_RAW  = ("https://raw.githubusercontent.com/ValveSoftware/"
               "counter-strike_regional_standings/refs/heads/main")
GITHUB_API  = ("https://api.github.com/repos/ValveSoftware/"
               "counter-strike_regional_standings/contents")
GH_FOLDER   = "invitation"   # "invitation" = ranked; "live" = continuous
GH_WORKERS  = 20


# ── Internal helpers ───────────────────────────────────────────────

def _gh_get(url: str, timeout: int = 12) -> str | None:
    """Fetch a URL; return text on 200, else None."""
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "VRS-Simulator/1.0"})
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _find_all_dates() -> list[tuple[str, str]]:
    """
    Discover all published VRS standing dates from the GitHub repo.
    Returns list of (date_str, year) sorted newest-first.
    e.g. [("2026_03_02","2026"), ("2026_02_03","2026"), ...]
    """
    result = []
    for year in ("2026", "2025", "2024"):
        text = _gh_get(f"{GITHUB_API}/{GH_FOLDER}/{year}")
        if not text:
            continue
        try:
            files = json.loads(text)
            for f in files:
                if not isinstance(f, dict):
                    continue
                if "standings_global" not in f.get("name", ""):
                    continue
                m = _re.search(r"(\d{4}_\d{2}_\d{2})", f.get("name", ""))
                if m:
                    result.append((m.group(1), year))
        except Exception:
            pass
    result.sort(key=lambda x: x[0], reverse=True)
    return result


def _find_latest_date() -> tuple[str | None, str | None]:
    """Return the most recent (date_str, year) pair."""
    dates = _find_all_dates()
    return (dates[0][0], dates[0][1]) if dates else (None, None)


def _parse_standings_index(text: str) -> list[dict]:
    """
    Parse standings_global_*.md.
    Returns list of {rank, points, team, detail_path}.
    """
    rows = []
    for line in text.splitlines():
        m = _re.match(
            r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*"
            r"\|[^|]*\|\s*\[details\]\(([^)]+)\)",
            line,
        )
        if m:
            rows.append({
                "rank":        int(m.group(1)),
                "points":      int(m.group(2)),
                "team":        m.group(3).strip(),
                "detail_path": m.group(4).strip(),
            })
    return rows


def _parse_detail_md(team: str, rank: int, valve_pts: int, text: str) -> dict:
    """
    Parse one team detail markdown file.
    Extracts pre-computed factor values and match history.
    """
    d: dict = {
        "team": team, "rank": rank, "valve_points": valve_pts,
        "total_points": float(valve_pts), "seed": float(valve_pts),
        "h2h_delta": 0.0,
        "bo_factor": 0.0, "bc_factor": 0.0, "on_factor": 0.0, "lan_factor": 0.0,
        "seed_combined": 0.0, "bo_sum": 0.0, "bc_pre_curve": 0.0,
        "wins": 0, "losses": 0, "lan_wins": 0, "total_matches": 0,
        "matches": [],
    }

    # ── Score breakdown ─────────────────────────────────────────────
    m = _re.search(
        r"Final Rank Value[^(]*\(([-\d.]+)\)[^=]*=.*?"
        r"Starting Rank Value[^(]*\(([-\d.]+)\).*?"
        r"Head To Head Adjustments[^(]*\(([-\d.]+)\)",
        text, _re.DOTALL,
    )
    if m:
        d["total_points"] = float(m.group(1))
        d["seed"]         = float(m.group(2))
        d["h2h_delta"]    = float(m.group(3))

    # ── Four factors ────────────────────────────────────────────────
    for key, label in [
        ("bo_factor", "Bounty Offered"), ("bc_factor", "Bounty Collected"),
        ("on_factor", "Opponent Network"), ("lan_factor", "LAN Wins"),
    ]:
        m = _re.search(rf"- {label}:\s*([\d.]+)", text)
        if m:
            d[key] = float(m.group(1))

    # ── Average (seed_combined) ──────────────────────────────────────
    m = _re.search(r"average of these factors is ([\d.]+)", text)
    if m:
        d["seed_combined"] = float(m.group(1))

    # ── BO sum + prize table ────────────────────────────────────────
    m = _re.search(r"sum of their top 10 scaled winnings \(\$([\d,]+\.\d+)\)", text)
    if m:
        d["bo_sum"] = float(m.group(1).replace(",", ""))

    # Parse the BO prize rows (table below "Top ten winnings for this roster:")
    bo_prizes: list[dict] = []
    in_bo_table = False
    for line in text.splitlines():
        if "Top ten winnings for this roster" in line:
            in_bo_table = True
            continue
        if in_bo_table:
            if not line.strip().startswith("|"):
                if bo_prizes:   # non-table line after we've collected rows → done
                    break
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 4:
                continue
            if "---" in cells[0] or "Event Date" in cells[0]:
                continue
            try:
                ev_date = cells[0]
                age_w_p = float(cells[1])
                prize_s = cells[2].replace("$", "").replace(",", "")
                scaled_s = cells[3].replace("$", "").replace(",", "")
                bo_prizes.append({
                    "event_date":     ev_date,
                    "age_weight":     age_w_p,
                    "prize_won":      float(prize_s),
                    "scaled_prize":   float(scaled_s),
                })
            except (ValueError, IndexError):
                continue
    d["bo_prizes"] = bo_prizes

    # ── bc_pre_curve (back-calculate from bc_factor via curve inverse) ─
    bf = d["bc_factor"]
    if 0 < bf < 1.0:
        try:
            d["bc_pre_curve"] = round(10 ** (1.0 - 1.0 / bf), 4)
        except Exception:
            d["bc_pre_curve"] = bf
    else:
        d["bc_pre_curve"] = 1.0 if bf >= 1.0 else 0.0

    # ── Match table ─────────────────────────────────────────────────
    wins = losses = lan_wins = 0
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 11:
            continue
        # Must start with a sequence integer
        if not _re.match(r"^\d+$", cells[0].lstrip("-").strip()):
            continue
        try:
            match_id = int(cells[1])
            date_s   = cells[2]
            opponent = cells[3]
            result   = cells[4]
            age_w_s  = cells[5]
            ev_w_s   = cells[6]
            lan_s    = cells[9]
            h2h_s    = cells[10]

            if not _re.match(r"\d{4}-\d{2}-\d{2}", date_s):
                continue

            age_w  = float(age_w_s)
            ev_w   = float(ev_w_s) if ev_w_s not in ("-", "") else 0.0
            is_lan = (lan_s not in ("-", "") and not lan_s.startswith("0"))
            h2h_a  = float(h2h_s) if h2h_s not in ("-", "") else 0.0

            # Extract opponent BO from Bounty Collected column: "0.017 (0.002)"
            bc_s = cells[7] if len(cells) > 7 else "-"
            opp_bo = 0.0
            if bc_s not in ("-", ""):
                _obo_m = _re.match(r"([\d.]+)", bc_s)
                if _obo_m:
                    opp_bo = float(_obo_m.group(1))

            # Extract opponent ON from Opponent Network column: "0.349 (0.050)"
            on_s = cells[8] if len(cells) > 8 else "-"
            opp_on = 0.0
            if on_s not in ("-", ""):
                _oon_m = _re.match(r"([\d.]+)", on_s)
                if _oon_m:
                    opp_on = float(_oon_m.group(1))

            # Approximate prize_pool from event_weight (curve inverse)
            if 0 < ev_w <= 1.0:
                pool = round(10 ** (1.0 - 1.0 / ev_w) * 1_000_000)
            else:
                pool = 0

            dt = datetime.strptime(date_s, "%Y-%m-%d")

            if result == "W":
                wins += 1
                if is_lan:
                    lan_wins += 1
            else:
                losses += 1

            d["matches"].append({
                "match_id":  match_id,
                "date":      dt,
                "opponent":  opponent,
                "result":    result,
                "age_w":     age_w,
                "ev_w":      ev_w,
                "prize_pool": float(pool),
                "is_lan":    is_lan,
                "h2h_adj":   h2h_a,
                "opp_bo":    opp_bo,
                "opp_on":    opp_on,
            })
        except (ValueError, IndexError):
            continue

    d["wins"]          = wins
    d["losses"]        = losses
    d["lan_wins"]      = lan_wins
    d["total_matches"] = wins + losses
    return d


@st.cache_data(ttl=3600, show_spinner=False)
def load_valve_github_data(date_str: str | None = None,
                            year: str | None = None,
                            _cache_v: int = 3) -> dict:
    """
    Fetch a VRS snapshot from GitHub.  If date_str is None, use the latest.

    Returns
    -------
    dict with keys:
        standings           pd.DataFrame  — one row per team, Valve exact numbers
        matches             pd.DataFrame  — win-level rows (for simulator)
        team_match_history  dict[str, list[dict]]  — all matches per team incl losses
        cutoff_date         str           — e.g. "2026_03_02"
        cutoff_datetime     datetime
        total_teams         int
        source              str           — "github" or "fallback"
        error               str | None
    """
    result = {
        "standings":          pd.DataFrame(),
        "matches":            pd.DataFrame(),
        "team_match_history": {},
        "cutoff_date":        "2026_03_02",
        "cutoff_datetime":    datetime(2026, 3, 2),
        "total_teams":        0,
        "source":             "fallback",
        "error":              None,
    }

    # ── Step 1: Resolve date ─────────────────────────────────────────
    if not date_str:
        date_str, year = _find_latest_date()
    if not date_str:
        result["error"] = "GitHub unreachable — using fallback data"
        return result

    # ── Step 2: Fetch standings index ───────────────────────────────
    idx_url = f"{GITHUB_RAW}/{GH_FOLDER}/{year}/standings_global_{date_str}.md"
    idx_text = _gh_get(idx_url)
    if not idx_text:
        result["error"] = f"Could not fetch standings index for {date_str}"
        return result

    teams = _parse_standings_index(idx_text)
    if not teams:
        result["error"] = "Standings index parsed 0 teams"
        return result

    # Also fetch regional standings to get region mapping
    region_map: dict[str, str] = {}
    for region_tag, region_label in [
        ("europe", "Europe"), ("americas", "Americas"),
        ("asia", "Asia"), ("asia-pacific", "Asia"),
        ("middle-east", "Middle East"),
    ]:
        r_url = f"{GITHUB_RAW}/{GH_FOLDER}/{year}/standings_{region_tag}_{date_str}.md"
        r_text = _gh_get(r_url)
        if r_text:
            for row in _parse_standings_index(r_text):
                region_map.setdefault(row["team"], region_label)

    # ── Step 3: Parallel-fetch all detail files ───────────────────
    detail_base = f"{GITHUB_RAW}/{GH_FOLDER}/{year}/"

    def _fetch_one(t: dict) -> tuple[dict, str | None]:
        return t, _gh_get(detail_base + t["detail_path"])

    parsed: list[dict] = []
    with ThreadPoolExecutor(max_workers=GH_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in teams}
        for fut in as_completed(futs):
            t, text = fut.result()
            if text:
                d = _parse_detail_md(t["team"], t["rank"], t["points"], text)
                d["region"] = region_map.get(t["team"], "Global")
                parsed.append(d)

    if not parsed:
        result["error"] = "Could not parse any detail files"
        return result

    # ── Step 4: Build standings DataFrame ───────────────────────────
    rows = [{
        "team":          d["team"],
        "rank":          d["rank"],
        "total_points":  d["total_points"],
        "seed":          d["seed"],
        "h2h_delta":     d["h2h_delta"],
        "bo_factor":     d["bo_factor"],
        "bc_factor":     d["bc_factor"],
        "on_factor":     d["on_factor"],
        "lan_factor":    d["lan_factor"],
        "bo_sum":        d["bo_sum"],
        "bc_pre_curve":  d["bc_pre_curve"],
        "on_raw":        d["on_factor"],
        "lan_wins":      d["lan_wins"],
        "seed_combined": d["seed_combined"],
        "wins":          d["wins"],
        "losses":        d["losses"],
        "total_matches": d["total_matches"],
        "region":        d["region"],
        "flag":          "🌍",
        "color":         "#58a6ff",
    } for d in parsed]
    # store bo_prizes per team for explainer page
    _bo_prizes_map: dict[str, list] = {d["team"]: d.get("bo_prizes", []) for d in parsed}

    standings_df = (pd.DataFrame(rows)
                    .sort_values("rank")
                    .reset_index(drop=True))

    # ── Step 5: Build match DataFrame (deduplicated) ─────────────────
    seen: set[int] = set()
    match_rows = []
    for d in parsed:
        for m in d.get("matches", []):
            mid = m["match_id"]
            if m["result"] == "W" and mid not in seen:
                seen.add(mid)
                match_rows.append({
                    "date":         m["date"],
                    "winner":       d["team"],
                    "loser":        m["opponent"],
                    "prize_pool":   m["prize_pool"],
                    "winner_prize": 0.0,
                    "loser_prize":  0.0,
                    "is_lan":       m["is_lan"],
                    "event":        "",
                })

    matches_df = pd.DataFrame(match_rows) if match_rows else pd.DataFrame()

    try:
        cutoff_dt_p = datetime.strptime(date_str, "%Y_%m_%d")
    except Exception:
        cutoff_dt_p = datetime(2026, 3, 2)

    # ── Build per-team match history (wins + losses, for explainer) ─
    team_match_history: dict[str, list[dict]] = {
        d["team"]: d.get("matches", []) for d in parsed
    }

    result.update({
        "standings":          standings_df,
        "matches":            matches_df,
        "team_match_history": team_match_history,
        "bo_prizes_map":      _bo_prizes_map,
        "cutoff_date":        date_str,
        "cutoff_datetime":    cutoff_dt_p,
        "total_teams":        len(standings_df),
        "source":             "github",
        "error":              None,
    })
    return result


# ══════════════════════════════════════════════════════════════════
# TEAM META  —  region / flag / colour (overrides for known teams)
# ══════════════════════════════════════════════════════════════════

_KNOWN_META: dict[str, dict] = {
    "Vitality":       {"flag": "🇫🇷", "color": "#F5A623"},
    "PARIVISION":     {"flag": "🇷🇺", "color": "#E91E63"},
    "Natus Vincere":  {"flag": "🇺🇦", "color": "#FFD600"},
    "Spirit":         {"flag": "🇷🇺", "color": "#7B68EE"},
    "MOUZ":           {"flag": "🇩🇪", "color": "#E53935"},
    "FaZe":           {"flag": "🌍",  "color": "#EC407A"},
    "Aurora":         {"flag": "🇷🇺", "color": "#00BCD4"},
    "G2":             {"flag": "🇪🇸", "color": "#F44336"},
    "3DMAX":          {"flag": "🇫🇷", "color": "#607D8B"},
    "Astralis":       {"flag": "🇩🇰", "color": "#1565C0"},
    "Falcons":        {"flag": "🇸🇦", "color": "#0288D1"},
    "FURIA":          {"flag": "🇧🇷", "color": "#F5A623"},
    "Liquid":         {"flag": "🇺🇸", "color": "#00B0FF"},
    "NRG":            {"flag": "🇺🇸", "color": "#FF5722"},
    "paiN":           {"flag": "🇧🇷", "color": "#D32F2F"},
    "The MongolZ":    {"flag": "🇲🇳", "color": "#FF6F00"},
    "TYLOO":          {"flag": "🇨🇳", "color": "#C62828"},
    "Rare Atom":      {"flag": "🇨🇳", "color": "#00897B"},
    "GamerLegion":    {"flag": "🇩🇰", "color": "#9C27B0"},
    "FUT":            {"flag": "🇹🇷", "color": "#FF9800"},
    "B8":             {"flag": "🇺🇦", "color": "#4CAF50"},
    "Gentle Mates":   {"flag": "🇪🇸", "color": "#AB47BC"},
    "HEROIC":         {"flag": "🇩🇰", "color": "#FF7043"},
    "Monte":          {"flag": "🇺🇦", "color": "#26C6DA"},
    "BetBoom":        {"flag": "🇷🇺", "color": "#EF5350"},
    "MIBR":           {"flag": "🇧🇷", "color": "#43A047"},
    "9z":             {"flag": "🇦🇷", "color": "#1976D2"},
    "BC.Game":        {"flag": "🌍",  "color": "#7E57C2"},
    "Virtus.pro":     {"flag": "🇷🇺", "color": "#FF8F00"},
}

# Colour palette for teams not in the known list (cycles)
_COLOR_CYCLE = [
    "#58a6ff","#3fb950","#f0b429","#f85149","#79c0ff",
    "#56d364","#e3b341","#ff7b72","#bc8cff","#39c5cf",
]
_color_idx = 0


def get_team_meta(team: str, region: str = "Global") -> dict:
    """Return {flag, color, region} for any team name."""
    global _color_idx
    known = _KNOWN_META.get(team, {})
    flag  = known.get("flag", "🌍")
    color = known.get("color", _COLOR_CYCLE[_color_idx % len(_COLOR_CYCLE)])
    if team not in _KNOWN_META:
        _color_idx += 1
    return {"flag": flag, "color": color, "region": region}


# ── Kept for backward-compatibility with display code ─────────────
TEAM_META: dict[str, dict] = {
    t: {**v, "region": "Global"}
    for t, v in _KNOWN_META.items()
}


# ══════════════════════════════════════════════════════════════════
# HELPERS  —  formatting utilities
# ══════════════════════════════════════════════════════════════════

def region_pill(region: str, regional_rank: int = 0) -> str:
    cls = {"Europe": "pill-eu", "Americas": "pill-am", "Asia": "pill-as"}.get(region, "pill-eu")
    rank_str = f' #{regional_rank}' if regional_rank > 0 else ''
    return f'<span class="{cls}">{region}{rank_str}</span>'


def rank_badge(rank: int) -> str:
    cls = "top1" if rank == 1 else "top3" if rank <= 3 else "top10" if rank <= 10 else "rest"
    return f'<span class="rank-badge {cls}">{rank}</span>'


def change_arrow(delta: int) -> str:
    if delta > 0:
        return f'<span class="change-up">▲ {delta}</span>'
    elif delta < 0:
        return f'<span class="change-down">▼ {abs(delta)}</span>'
    return '<span class="change-same">—</span>'


def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "region" not in df.columns:
        df["region"] = df["team"].map(lambda t: TEAM_META.get(t, {}).get("region", "Global"))
    df["flag"] = df["team"].map(
        lambda t: _KNOWN_META.get(t, {}).get("flag",
                  df.loc[df["team"]==t, "flag"].iloc[0]
                  if "flag" in df.columns and (df["team"]==t).any() else "🌍")
    )
    return df


def add_regional_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regional_rank"] = (df.groupby("region")["total_points"]
                             .rank(ascending=False, method="first")
                             .astype(int))
    return df


# ══════════════════════════════════════════════════════════════════
# SIDEBAR  +  DATA LOADING
# ══════════════════════════════════════════════════════════════════

# ── Discover all available dates first (lightweight, cached 1h) ────
_all_dates = _find_all_dates()   # [(date_str, year), ...]

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
        "🔮 Scenario Simulator",
        "⚔️ Team Comparison",
    ], label_visibility="collapsed", key="main_nav")
    st.markdown("---")

    # ── Simulation toggle ─────────────────────────────────────────
    sim_enabled = st.toggle(
        "🔮 Simulate next month",
        value=False,
        key="sim_toggle",
        help="Shift the cutoff forward ~1 month (no new matches). Shows pure time-decay effect on standings.",
    )

    # ── Historical date selector ──────────────────────────────────
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
            disabled=sim_enabled,
            help=("⚠️ Fixiert auf letzten Snapshot während Simulation."
                  if sim_enabled else
                  "Valve publishes new standings monthly. Select any historical snapshot."),
        )
        if sim_enabled:
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
        _gd = load_valve_github_data(_sel_date, _sel_year)

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


def compute_standings(extra_matches: pd.DataFrame = None, cutoff: datetime = None) -> pd.DataFrame:
    """
    For the scenario simulator.
    Starts from Valve's published final scores (exact) and applies
    H2H deltas for any hypothetical extra_matches.
    Factor Score factors (BO/BC/ON/LAN) are held constant from Valve's snapshot.
    """
    result = base_standings.copy()
    if extra_matches is None or extra_matches.empty:
        return result

    # Build mutable score dict starting from Valve's published totals
    scores: dict[str, float] = result.set_index("team")["total_points"].to_dict()
    h2h_extra: dict[str, float] = {t: 0.0 for t in scores}

    # Process hypothetical matches chronologically
    for _, row in extra_matches.sort_values("date").iterrows():
        w, l = str(row["winner"]), str(row["loser"])
        if w not in scores or l not in scores:
            continue
        E_w = expected_win(scores[w], scores[l])
        K   = BASE_K * 1.0   # new match → age_weight = 1.0
        d_w =  K * (1.0 - E_w)
        d_l = -K * E_w
        scores[w]     += d_w
        scores[l]     += d_l
        h2h_extra[w]  += d_w
        h2h_extra[l]  += d_l

    result["total_points"] = result["team"].map(scores).fillna(result["total_points"])
    result["h2h_delta"]    = result["team"].map(
        lambda t: result.loc[result["team"]==t, "h2h_delta"].iloc[0] + h2h_extra.get(t, 0.0)
    )
    result = result.sort_values("total_points", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


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
    # ── Detect duplicate team names (roster changes → 2 entries) ──
    # These teams have unreliable match history (dict key collision)
    # so we exclude them from simulation and keep their original values.
    _name_counts = standings["team"].value_counts()
    _dup_names   = set(_name_counts[_name_counts > 1].index)

    # Work with deduplicated, non-duplicate teams only
    _clean_std = standings[~standings["team"].isin(_dup_names)].copy()
    _clean_std = _clean_std.sort_values("rank").reset_index(drop=True)
    all_teams  = _clean_std["team"].tolist()

    # Keep ALL original rows for reference data (BO/ON factors)
    _ref_std = standings.sort_values("rank").drop_duplicates("team", keep="first")

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

    # ── Factor 1: Bounty Offered (all teams, needed for BC/ON) ────
    # USE VALVE'S PUBLISHED BO FACTORS FOR ALL TEAMS.
    #
    # Rationale: We don't have complete prize-pool data for every team
    # (the "Top ten winnings" table is only parseable for ~40 teams).
    # Mixing recalculated BO (with decay) for some teams with original
    # BO for others creates an apples-to-oranges comparison.
    #
    # BO changes minimally over 30 days anyway (big prizes are recent).
    # The real time-decay effect comes from BC, ON, LAN, and H2H where
    # we DO have full match-level data and CAN properly shift age weights.
    #
    # This keeps the "opponent strength" baseline consistent for all teams.
    orig_bo_map = _ref_std.set_index("team")["bo_factor"].to_dict()
    bo_f: dict[str, float] = {t: orig_bo_map.get(t, 0.0) for t in all_teams}

    # Still compute bo_sum for the selected team's display (where prize data exists)
    bo_sum_new: dict[str, float] = {}
    _bo_from_prizes = 0
    _bo_from_fallback = 0
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
            _bo_from_prizes += 1
        else:
            # Use original bo_sum for display purposes
            bo_sum_new[t] = float(_ref_std.loc[_ref_std["team"] == t, "bo_sum"].iloc[0]
                                  if t in _ref_std["team"].values else 0.0)
            _bo_from_fallback += 1

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
    # Use Valve's published ON factors as the network baseline, then
    # run a SINGLE iteration with new age weights to capture the decay.
    # Fall back to per-match opp_on for opponents not in standings.
    orig_on_map = _ref_std.set_index("team")["on_factor"].to_dict()
    on_seed: dict[str, float] = {t: orig_on_map.get(t, 0.0) for t in all_teams}

    on_final: dict[str, float] = {}
    for t in eligible:
        entries = []
        for m in tmh.get(t, []):
            if m["result"] != "W" or m.get("ev_w", 0) <= 0:
                continue
            if not (new_window_start <= m["date"] <= new_cutoff):
                continue
            aw = age_weight(m["date"], new_cutoff)
            opp_on_val = on_seed.get(m["opponent"], m.get("opp_on", 0.0))
            entries.append(opp_on_val * aw * m["ev_w"])
        on_final[t] = top_n_sum(entries, TOP_N) / TOP_N

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

    # ── Merge back duplicate teams (unchanged original values) ────
    if _dup_names:
        dup_rows = standings[standings["team"].isin(_dup_names)].copy()
        # Ensure dup_rows have all required columns with defaults
        for col in result.columns:
            if col not in dup_rows.columns:
                dup_rows[col] = 0
        result = pd.concat([result, dup_rows[result.columns]], ignore_index=True)

    result = (result.sort_values("total_points", ascending=False)
              .reset_index(drop=True))
    result["rank"] = result.index + 1
    result = add_regional_rank(result)
    return {"standings": result, "match_h2h": match_h2h,
            "diag_bo_prizes": _bo_from_prizes, "diag_bo_fallback": _bo_from_fallback,
            "dup_teams": _dup_names}


# ══════════════════════════════════════════════════════════════════
# SIMULATION EXECUTION
# ══════════════════════════════════════════════════════════════════

sim_active       = False
sim_cutoff_dt    = None
sim_match_h2h    = {}           # match_id → {winner, loser, w_delta, l_delta}
_diag_prizes     = 0
_diag_fallback   = 0
_sim_dup_teams   = set()
original_standings = base_standings.copy()

if sim_enabled:
    # Compute simulated cutoff: 1st of the month AFTER the snapshot
    _snap_dt = _gd["cutoff_datetime"]
    if _snap_dt.month == 12:
        sim_cutoff_dt = datetime(_snap_dt.year + 1, 1, 1)
    else:
        sim_cutoff_dt = datetime(_snap_dt.year, _snap_dt.month + 1, 1)

    with st.spinner("🔮 Recalculating all factors with shifted cutoff…"):
        _sim_out = _simulate_time_decay(
            team_match_history, bo_prizes_map,
            base_standings, _snap_dt, sim_cutoff_dt,
        )

    _sim_result   = _sim_out["standings"]
    sim_match_h2h = _sim_out["match_h2h"]
    _diag_prizes  = _sim_out.get("diag_bo_prizes", 0)
    _diag_fallback = _sim_out.get("diag_bo_fallback", 0)
    _sim_dup_teams = _sim_out.get("dup_teams", set())

    if not _sim_result.empty:
        original_standings = base_standings.copy()
        base_standings     = _sim_result
        # Compute rank delta (positive = climbed, negative = dropped)
        _orig_rank_map = original_standings.sort_values("rank").drop_duplicates("team", keep="first").set_index("team")["rank"].to_dict()
        base_standings["rank_delta"] = base_standings.apply(
            lambda r: (int(_orig_rank_map.get(r["team"], r["rank"])) - int(r["rank"])
                       if r["team"] not in _sim_dup_teams else 0),
            axis=1,
        ).astype(int)
        cutoff_dt          = sim_cutoff_dt
        cutoff_date        = sim_cutoff_dt
        sim_active         = True

# ── Simulation banner (shown on all pages) ────────────────────────
if sim_active:
    _days_fwd = (sim_cutoff_dt - _gd["cutoff_datetime"]).days
    # Merge original ranks for quick delta display
    _orig_rank = original_standings.set_index("team")["rank"].to_dict()
    _new_top5  = base_standings.head(5)
    _top5_info = " · ".join(
        f'**#{int(r["rank"])}** {r["team"]} ({r["total_points"]:,.0f})'
        for _, r in _new_top5.iterrows()
    )
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1040,#0d1117);border:1px solid #7c3aed;
                border-radius:10px;padding:16px 20px;margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="font-size:22px">🔮</span>
        <span style="font-size:16px;font-weight:700;color:#c4b5fd;">
          Simulation: Cutoff → {sim_cutoff_dt.strftime('%B %d, %Y')}</span>
        <span style="background:#7c3aed;color:#fff;border-radius:12px;padding:2px 10px;
                     font-size:11px;font-weight:600">+{_days_fwd} days</span>
      </div>
      <div style="font-size:12px;color:#a78bfa;line-height:1.6;">
        BC, ON, LAN, and H2H fully recomputed with shifted age weights — no new matches assumed.<br>
        BO uses Valve's published factors (consistent baseline; BO changes minimally over 30 days).
        {f'<br><span style="color:#8b949e;font-size:11px">{len(_sim_dup_teams)} teams with roster splits excluded from simulation (kept original values): {", ".join(sorted(_sim_dup_teams)[:8])}{"…" if len(_sim_dup_teams) > 8 else ""}</span>' if _sim_dup_teams else ''}
      </div>
    </div>""", unsafe_allow_html=True)

# ── Sidebar status (after simulation state is resolved) ───────────
with st.sidebar:
    if sim_active:
        st.info(f"🔮 Simulation  ·  {sim_cutoff_dt.strftime('%Y_%m_%d')}")
    elif _gd["source"] == "github":
        st.success(f"✅ {_gd['total_teams']} teams  ·  {_gd['cutoff_date']}")
    else:
        st.error("❌ GitHub offline — limited data")


# ══════════════════════════════════════════════════════════════════
# PAGE 1  ·  RANKING DASHBOARD
# ══════════════════════════════════════════════════════════════════

if page == "📊 Ranking Dashboard":
    st.title("📊 CS2 Valve Regional Standings")
    st.caption(
        f"Simulated standings · cutoff **{cutoff_date.strftime('%B %d, %Y')}** · "
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

    # ── Simulation impact summary (only when sim is active) ────────
    if sim_active:
        st.markdown("---")
        st.markdown("### 📉 Time-Decay Impact")
        _orig_map  = original_standings.sort_values("rank").drop_duplicates("team", keep="first").set_index("team")
        _sim_map   = base_standings.sort_values("rank").drop_duplicates("team", keep="first").set_index("team")
        # Only compare teams that were actually simulated (exclude roster-split teams)
        _both      = sorted((set(_orig_map.index) & set(_sim_map.index)) - _sim_dup_teams)

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
        st.plotly_chart(fig_decay, use_container_width=True)

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
            rd   = int(row.get("rank_delta", 0)) if sim_active else 0
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
        if sim_active:
            _orig_eu = original_standings[original_standings["region"] == "Europe"].sort_values("total_points", ascending=False).reset_index(drop=True)
            _orig_eu["rank"] = _orig_eu.index + 1
            _oem = _orig_eu.set_index("team")["rank"].to_dict()
            eu_df["rank_delta"] = eu_df.apply(lambda r: int(_oem.get(r["team"], r["rank"])) - int(r["rank"]), axis=1)
        render_table(eu_df, show_all=True)

    with tab_am:
        am_df = base_standings[base_standings["region"] == "Americas"].reset_index(drop=True)
        am_df["rank"] = am_df.index + 1
        if sim_active:
            _orig_am = original_standings[original_standings["region"] == "Americas"].sort_values("total_points", ascending=False).reset_index(drop=True)
            _orig_am["rank"] = _orig_am.index + 1
            _oam = _orig_am.set_index("team")["rank"].to_dict()
            am_df["rank_delta"] = am_df.apply(lambda r: int(_oam.get(r["team"], r["rank"])) - int(r["rank"]), axis=1)
        render_table(am_df, show_all=True)

    with tab_as:
        as_df = base_standings[base_standings["region"] == "Asia"].reset_index(drop=True)
        as_df["rank"] = as_df.index + 1
        if sim_active:
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2  ·  SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════════════

elif page == "🔮 Scenario Simulator":
    st.title("🔮 What-If Scenario Simulator")
    st.markdown(
        "Queue hypothetical match results and instantly see how the VRS standings shift. "
        "Uses the full two-phase engine (seed recalculation + H2H) on every simulation."
    )

    all_teams = sorted(base_standings["team"].tolist()) if not base_standings.empty else sorted(TEAM_META.keys())

    if "hyp_matches" not in st.session_state:
        st.session_state.hyp_matches = []

    # ── Input form ────────────────────────────────────────────────
    with st.form("add_match"):
        st.subheader("➕ Add a Hypothetical Match")
        c1, c2, c3 = st.columns(3)
        winner = c1.selectbox("🏆 Winner", all_teams, index=all_teams.index("G2"))
        loser  = c2.selectbox("❌ Loser",
                              [t for t in all_teams if t != winner],
                              index=0)
        event_name = c3.text_input("🏟️ Event", value="Hypothetical Event")

        c4, c5, c6, c7 = st.columns(4)
        prize_pool   = c4.number_input("💵 Prize Pool (USD)", 10_000, 2_000_000, 250_000, 10_000)
        winner_pct   = c5.slider("🥇 Winner %", 10, 60, 35)
        loser_pct    = c6.slider("🥈 Loser %",  0, 30, 20)
        is_lan       = c7.toggle("🖥️ LAN?", value=True)
        match_date   = st.date_input("📅 Match Date", value=datetime(2026, 3, 20))

        if st.form_submit_button("➕ Queue Match", use_container_width=True, type="primary"):
            pool = float(prize_pool)
            st.session_state.hyp_matches.append({
                "date":         datetime.combine(match_date, datetime.min.time()),
                "winner":       winner,
                "loser":        loser,
                "event":        event_name,
                "prize_pool":   pool,
                "winner_prize": pool * winner_pct / 100,
                "loser_prize":  pool * loser_pct  / 100,
                "is_lan":       is_lan,
            })
            st.success(f"✅ Added: **{winner}** def. **{loser}**")

    # ── Queue display ─────────────────────────────────────────────
    if st.session_state.hyp_matches:
        q = pd.DataFrame(st.session_state.hyp_matches)[
            ["winner", "loser", "event", "prize_pool", "is_lan", "date"]]
        q.columns = ["Winner", "Loser", "Event", "Prize Pool", "LAN", "Date"]
        q["Prize Pool"] = q["Prize Pool"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(q, use_container_width=True, hide_index=True)

        ca, cb = st.columns([3, 1])
        if cb.button("🗑️ Clear Queue", use_container_width=True):
            st.session_state.hyp_matches = []
            st.rerun()

        if ca.button("🚀 Simulate & Compare", use_container_width=True, type="primary"):
            with st.spinner("Recalculating full VRS…"):
                hyp_df       = pd.DataFrame(st.session_state.hyp_matches)
                new_standings = compute_standings(extra_matches=hyp_df)

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

            affected = set(hyp_df["winner"].tolist() + hyp_df["loser"].tolist())

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
                st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(fig_r, use_container_width=True)

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
        st.plotly_chart(fig_b, use_container_width=True)

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
        st.markdown('<span class="tag-v">✓ VERIFIED — derived from curve function applied to prize pool</span>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2])
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
        with col_r:
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
                line=dict(color="#3fb950", width=2.5),
                fill="tozeroy", fillcolor="rgba(63,185,80,0.07)"))
            fig_ew.add_trace(go.Scatter(
                x=[_ew_pool/1e6], y=[_ew_val], mode="markers",
                marker=dict(color="#3fb950", size=12, symbol="diamond"), showlegend=False))
            for pp, lbl in [(1e6, "$1M → 100%"), (1e5, "$100k → 50%")]:
                ev = event_stakes(max(pp, 1))
                fig_ew.add_annotation(x=pp/1e6, y=ev, text=lbl,
                    showarrow=True, arrowhead=0, ax=40, ay=-20,
                    font=dict(size=9, color="#8b949e"))
            fig_ew.update_layout(
                xaxis=dict(title="Prize Pool ($M)", gridcolor="#21262d",
                           tickvals=[0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.2],
                           ticktext=["$0","$100k","$250k","$500k","$750k","$1M","$1.2M"]),
                yaxis=dict(title="Event Weight", gridcolor="#21262d", range=[0, 1.1]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10), height=280)
            st.plotly_chart(fig_ew, use_container_width=True)
            st.caption(
                "Source: [Reddit — How Valve's CSGO Team Ranking System Works]"
                "(https://www.reddit.com/r/GlobalOffensive/comments/15j0t5e/) · "
                '"$1M tournaments are scaled 100%, $100k tournaments 50%, $0 tournaments 14.3%"'
            )

    # ══════════════════════════════════════════════════════════════
    with tab_age:
        st.subheader("⏳ Age Weight (Time Decay)")
        st.markdown('<span class="tag-v">✓ VERIFIED — exact match to Vitality detail sheet</span>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2])
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

        with col_r:
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
            st.plotly_chart(fig_aw, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    with tab_curve:
        st.subheader("📐 The Curve Function  f(x)")
        st.markdown('<span class="tag-v">✓ VERIFIED — explicitly stated in official data footnote</span>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2])
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
        with col_r:
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
                mode="lines", line=dict(color="#3fb950", width=2.5),
                fill="tozeroy", fillcolor="rgba(63,185,80,0.07)"))
            fig_c.add_trace(go.Scatter(x=[_cv_input], y=[_cv_output], mode="markers",
                marker=dict(color="#3fb950", size=12, symbol="diamond"), showlegend=False))
            # Reference annotations
            for xr, lbl in [(1.0, "f(1.0) = 1.000"), (0.5, "f(0.5) = 0.769"),
                            (0.1, "f(0.1) = 0.500"), (0.01, "f(0.01) = 0.333")]:
                fig_c.add_annotation(x=xr, y=curve(xr), text=lbl,
                    showarrow=True, arrowhead=0, ax=50, ay=-15,
                    font=dict(size=9, color="#8b949e"))
            fig_c.update_layout(
                xaxis=dict(title="Pre-curve value (x)", gridcolor="#21262d"),
                yaxis=dict(title="f(x) — after curve", gridcolor="#21262d", range=[0, 1.1]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10), height=280)
            st.plotly_chart(fig_c, use_container_width=True)
            st.caption(
                "The curve is also used for Event Weight (see Event Weight tab), "
                "but with prize_pool/$1M as input instead of factor values."
            )

    # ══════════════════════════════════════════════════════════════
    with tab_bo2:
        st.subheader("🏆 Factor 1 — Bounty Offered")
        st.markdown('<span class="tag-v">✓ VERIFIED — sum / 5th-ref / curve(min(1,x)) confirmed</span>', unsafe_allow_html=True)

        if "bo_sim_entries" not in st.session_state:
            st.session_state.bo_sim_entries = [
                {"event": "IEM Krakow 2026",   "prize": 350_000, "days_ago":  14},
                {"event": "PGL Cluj 2026",      "prize": 460_000, "days_ago":   7},
                {"event": "BLAST Fall 2025",    "prize": 500_000, "days_ago":  77},
                {"event": "ESL Pro League S21", "prize":  90_000, "days_ago":  49},
                {"event": "Small LAN Oct 2025", "prize":  65_000, "days_ago": 168},
            ]
        if "bo_sim_result" not in st.session_state:
            st.session_state.bo_sim_result = None

        col_l, col_r = st.columns([3, 2])
        with col_l:
            _bo_fc = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:8px 0;">'
                '<div style="text-align:center;margin-bottom:16px;">'
                '<span style="font-size:13px;font-weight:700;color:#f0b429;">🏆 Bounty Offered — Calculation Flow</span>'
                '</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:8px 22px;">'
                '<div style="font-size:12px;font-weight:700;color:#8b949e;">📅 All wins · 180-day window</div>'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d1a2e;border:1px solid #58a6ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:4px;">Step 1 — Contribution per win</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#0a0e1a;border-radius:4px;padding:6px 8px;">'
                'contribution = prize_won × age_weight'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">⚠️ No event stakes — only time decay applied</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d1a2e;border:1px solid #58a6ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#58a6ff;margin-bottom:4px;">Step 2 — Keep top 10 · sum</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#0a0e1a;border-radius:4px;padding:6px 8px;">'
                'BO_sum = Σ top-10 (contributions)'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">Wins beyond the top-10 are discarded</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#2d1f00;border:1px solid #f0b429;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f0b429;margin-bottom:4px;">Step 3 — Normalise against the field</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#1a1000;border-radius:4px;padding:6px 8px;">'
                'ref&#8325; = 5th-highest BO_sum across all teams<br>'
                'ratio&#160;&#160; = min(1.0,  BO_sum / ref&#8325;)'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">Top-5 teams by prize money all clamp to ratio = 1.0</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#2d1f00;border:1px solid #f0b429;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f0b429;margin-bottom:4px;">Step 4 — Apply curve function</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#1a1000;border-radius:4px;padding:6px 8px;">'
                'BO = f(ratio) = 1 / (1 + |log&#8321;&#8320;(ratio)|)'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">f(1.0)=1.000 · f(0.5)≈0.770 · f(0.1)=0.500 · f(0.01)=0.333</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#2d1f00;border:2px solid #f0b429;border-radius:8px;padding:10px 28px;">'
                '<div style="font-size:14px;font-weight:700;color:#f0b429;">🏆 BO Factor &nbsp; [0 → 1]</div>'
                '</div></div>'
                '</div>'
            )
            st.markdown(_bo_fc, unsafe_allow_html=True)

        with col_r:
            st.markdown("#### 🎮 Build a prize history")
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
                    "ref₅ — 5th-highest BO_sum in the field ($)",
                    min_value=1_000, max_value=5_000_000, value=334_320, step=10_000,
                    help="Real Valve value ≈ $334,320 at cutoff 2026-03-02")

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
        st.markdown('<span class="tag-v">✓ VERIFIED — BC = curve(Σ_top10_adjusted / 10) confirmed numerically</span>', unsafe_allow_html=True)

        if "bc_sim_entries" not in st.session_state:
            st.session_state.bc_sim_entries = [
                {"opponent": "NAVI",      "opp_bo": 1.000, "pool": 1_000_000, "days_ago":  5},
                {"opponent": "G2 Esports","opp_bo": 0.843, "pool":   500_000, "days_ago": 20},
                {"opponent": "MOUZ",      "opp_bo": 0.672, "pool":   500_000, "days_ago": 45},
                {"opponent": "Liquid",    "opp_bo": 0.446, "pool":   250_000, "days_ago": 80},
                {"opponent": "Cloud9",    "opp_bo": 0.250, "pool":   100_000, "days_ago": 10},
            ]
        if "bc_sim_result" not in st.session_state:
            st.session_state.bc_sim_result = None

        col_l, col_r = st.columns([3, 2])
        with col_l:
            _bc_fc = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:8px 0;">'
                '<div style="text-align:center;margin-bottom:16px;">'
                '<span style="font-size:13px;font-weight:700;color:#3fb950;">💰 Bounty Collected — Calculation Flow</span>'
                '</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:8px 22px;">'
                '<div style="font-size:12px;font-weight:700;color:#8b949e;">📅 All wins · 180-day window</div>'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0d0d;border:1px solid #f85149;border-radius:8px;padding:8px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f85149;margin-bottom:2px;">⚠️ Prize pool required</div>'
                '<div style="font-size:10px;color:#c9d1d9;">Wins at events with no prize pool are excluded (event_weight = 0)</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d1a1a;border:1px solid #3fb950;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#3fb950;margin-bottom:4px;">Step 1 — Entry per win</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#061010;border-radius:4px;padding:6px 8px;">'
                'entry = opp_BO × age_weight × event_weight'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">opp_BO = opponent\'s Bounty Offered · event_weight = curve(pool / $1M)</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d1a1a;border:1px solid #3fb950;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#3fb950;margin-bottom:4px;">Step 2 — Top 10 · average</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#061010;border-radius:4px;padding:6px 8px;">'
                'BC_pre = &#x3A3; top-10 (entries) / 10'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d1a1a;border:1px solid #3fb950;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#3fb950;margin-bottom:4px;">Step 3 — Apply curve</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#061010;border-radius:4px;padding:6px 8px;">'
                'BC = f(BC_pre) = 1 / (1 + |log&#8321;&#8320;(BC_pre)|)'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">Unlike ON, BC does apply the curve function</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#0d1a1a;border:2px solid #3fb950;border-radius:8px;padding:10px 28px;">'
                '<div style="font-size:14px;font-weight:700;color:#3fb950;">💰 BC Factor &nbsp; [0 → 1]</div>'
                '</div></div>'
                '</div>'
            )
            st.markdown(_bc_fc, unsafe_allow_html=True)

        with col_r:
            st.markdown("#### 🎮 Build a win record")
            st.caption("Set opponent BO and event prize pool — both scale the entry value.")

            with st.form("bc_add_form", clear_on_submit=True):
                _bc1, _bc2 = st.columns(2)
                _bopp    = _bc1.text_input("Opponent", placeholder="NAVI")
                _boppbo  = _bc2.slider("Opp. BO factor", 0.0, 1.0, 0.70, 0.01)
                _bc3, _bc4 = st.columns(2)
                _bpool   = _bc3.number_input("Prize pool ($)", min_value=0,
                                             max_value=2_000_000, value=500_000, step=50_000)
                _bdays   = _bc4.number_input("Days ago", min_value=0, max_value=179, value=20)
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
        st.markdown('<span class="tag-v">✓ VERIFIED — ON = Σ_top10_adjusted / 10, NO curve, confirmed numerically</span>', unsafe_allow_html=True)

        if "on_sim_entries" not in st.session_state:
            st.session_state.on_sim_entries = [
                {"opponent": "NAVI",       "opp_on": 0.622, "pool": 1_000_000, "days_ago":  5},
                {"opponent": "MOUZ",       "opp_on": 0.460, "pool":   500_000, "days_ago": 20},
                {"opponent": "G2 Esports", "opp_on": 0.380, "pool":   500_000, "days_ago": 45},
                {"opponent": "Liquid",     "opp_on": 0.280, "pool":   250_000, "days_ago": 80},
                {"opponent": "BIG",        "opp_on": 0.195, "pool":   100_000, "days_ago": 30},
            ]
        if "on_sim_result" not in st.session_state:
            st.session_state.on_sim_result = None

        col_l, col_r = st.columns([3, 2])
        with col_l:
            _on_fc = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:8px 0;">'
                '<div style="text-align:center;margin-bottom:16px;">'
                '<span style="font-size:13px;font-weight:700;color:#79c0ff;">🕸️ Opponent Network — Calculation Flow</span>'
                '</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:8px 22px;">'
                '<div style="font-size:12px;font-weight:700;color:#8b949e;">📅 All wins · 180-day window</div>'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d0d1a;border:1px solid #79c0ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#79c0ff;margin-bottom:4px;">Initialise — seed with BO</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#060610;border-radius:4px;padding:6px 8px;">'
                'ON&#8320;[team] = BO_factor[team]  &#x2200; teams'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">Bootstraps the circular dependency with a known prior</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0d0d;border:1px solid #f85149;border-radius:8px;padding:8px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f85149;margin-bottom:2px;">⚠️ Prize pool required</div>'
                '<div style="font-size:10px;color:#c9d1d9;">Only wins at events with a prize pool count</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d0d1a;border:1px solid #79c0ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#79c0ff;margin-bottom:4px;">Step 1 — Entry per win (uses opp ON, not BO)</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#060610;border-radius:4px;padding:6px 8px;">'
                'entry = opp_ON × age_weight × event_weight'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">Key difference from BC: uses opponent\'s ON value, not their BO</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#0d0d1a;border:1px solid #79c0ff;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#79c0ff;margin-bottom:4px;">Step 2 — Top 10 · average &nbsp;(no curve!)</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#060610;border-radius:4px;padding:6px 8px;">'
                'ON_new = &#x3A3; top-10 (entries) / 10'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">⚠️ No curve applied — output is a direct average (unlike BC)</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a1a0a;border:2px dashed #f0b429;border-radius:8px;padding:8px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f0b429;margin-bottom:2px;">↺ Repeat 6 iterations (PageRank)</div>'
                '<div style="font-size:10px;color:#c9d1d9;">Each pass refines ON for all teams simultaneously — converges in ~5 rounds</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#0d0d1a;border:2px solid #79c0ff;border-radius:8px;padding:10px 28px;">'
                '<div style="font-size:14px;font-weight:700;color:#79c0ff;">🕸️ ON Factor &nbsp; [0 → 1]</div>'
                '</div></div>'
                '</div>'
            )
            st.markdown(_on_fc, unsafe_allow_html=True)

        with col_r:
            st.markdown("#### 🎮 Build an opponent network")
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
        st.markdown('<span class="tag-v">✓ VERIFIED — 1.0 × age_weight, top-10, /10, no curve, no event stakes</span>', unsafe_allow_html=True)

        if "lan_sim_entries" not in st.session_state:
            st.session_state.lan_sim_entries = [
                {"event": "IEM Major 2026",        "days_ago":  10, "is_lan": True},
                {"event": "PGL Antwerp 2026",       "days_ago":  40, "is_lan": True},
                {"event": "BLAST Fall 2025",        "days_ago":  90, "is_lan": True},
                {"event": "Online Qualifier (Q1)",  "days_ago":  15, "is_lan": False},
                {"event": "IEM Katowice 2025",      "days_ago": 130, "is_lan": True},
            ]
        if "lan_sim_result" not in st.session_state:
            st.session_state.lan_sim_result = None

        col_l, col_r = st.columns([3, 2])
        with col_l:
            _lan_fc = (
                '<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:8px 0;">'
                '<div style="text-align:center;margin-bottom:16px;">'
                '<span style="font-size:13px;font-weight:700;color:#f85149;">🖥️ LAN Wins — Calculation Flow</span>'
                '</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#161b22;border:2px solid #8b949e;border-radius:8px;padding:8px 22px;">'
                '<div style="font-size:12px;font-weight:700;color:#8b949e;">📅 All wins · 180-day window</div>'
                '</div></div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0a0d;border:1px solid #f85149;border-radius:8px;padding:8px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f85149;margin-bottom:2px;">⚠️ LAN event only</div>'
                '<div style="font-size:10px;color:#c9d1d9;">Online wins are excluded. No prize pool requirement (unlike BC / ON).</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0a0d;border:1px solid #f85149;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f85149;margin-bottom:4px;">Step 1 — Fixed value per LAN win</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#0e0508;border-radius:4px;padding:6px 8px;">'
                'entry = 1.0 × age_weight'
                '</div>'
                '<div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;">'
                '<span style="background:#21262d;border-radius:4px;padding:3px 8px;font-size:10px;color:#8b949e;">❌ No event stakes</span>'
                '<span style="background:#21262d;border-radius:4px;padding:3px 8px;font-size:10px;color:#8b949e;">❌ No prize requirement</span>'
                '<span style="background:#21262d;border-radius:4px;padding:3px 8px;font-size:10px;color:#8b949e;">❌ No curve</span>'
                '</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="background:#1a0a0d;border:1px solid #f85149;border-radius:8px;padding:10px 14px;margin:0 24px 4px;">'
                '<div style="font-size:11px;font-weight:700;color:#f85149;margin-bottom:4px;">Step 2 — Top 10 · average</div>'
                '<div style="font-family:monospace;font-size:11px;color:#c9d1d9;background:#0e0508;border-radius:4px;padding:6px 8px;">'
                'LAN = &#x3A3; top-10 (entries) / 10'
                '</div>'
                '<div style="font-size:10px;color:#8b949e;margin-top:4px;">Max = 1.000 — 10 LAN wins all within last 30 days</div>'
                '</div>'
                '<div style="text-align:center;color:#484f58;font-size:20px;margin:3px 0;">▼</div>'
                '<div style="text-align:center;">'
                '<div style="display:inline-block;background:#1a0a0d;border:2px solid #f85149;border-radius:8px;padding:10px 28px;">'
                '<div style="font-size:14px;font-weight:700;color:#f85149;">🖥️ LAN Factor &nbsp; [0 → 1]</div>'
                '</div></div>'
                '</div>'
            )
            st.markdown(_lan_fc, unsafe_allow_html=True)

        with col_r:
            st.markdown("#### 🎮 Build a LAN win record")
            st.caption("Toggle LAN/online for each entry — only LAN wins count toward this factor.")

            with st.form("lan_add_form", clear_on_submit=True):
                _lc1, _lc2, _lc3 = st.columns([4, 2, 1])
                _levent = _lc1.text_input("Event name", placeholder="IEM Major 2026")
                _ldays  = _lc2.number_input("Days ago", min_value=0, max_value=179, value=20)
                _llan   = _lc3.checkbox("LAN", value=True)
                if st.form_submit_button("➕ Add win", use_container_width=True) and _levent:
                    st.session_state.lan_sim_entries.append(
                        {"event": _levent, "days_ago": int(_ldays), "is_lan": bool(_llan)})
                    st.session_state.lan_sim_result = None

            if st.session_state.lan_sim_entries:
                for _i, _e in enumerate(st.session_state.lan_sim_entries):
                    _ec1, _ec2, _ec3 = st.columns([4, 3, 0.7])
                    _lan_badge = (
                        '<span style="background:#f85149;color:#fff;border-radius:3px;'
                        'padding:1px 5px;font-size:9px;font-weight:700;">LAN</span>'
                        if _e["is_lan"] else
                        '<span style="background:#484f58;color:#c9d1d9;border-radius:3px;'
                        'padding:1px 5px;font-size:9px;">online</span>'
                    )
                    _ec1.markdown(f"<span style='font-size:12px'>{_e['event']}</span>",
                                  unsafe_allow_html=True)
                    _ec2.markdown(
                        f"<span style='font-size:11px;color:#8b949e'>{_lan_badge} · {_e['days_ago']}d ago</span>",
                        unsafe_allow_html=True)
                    if _ec3.button("✕", key=f"lan_rm_{_i}"):
                        st.session_state.lan_sim_entries.pop(_i)
                        st.session_state.lan_sim_result = None
                        st.rerun()

                st.markdown("---")
                if st.button("🧮 Calculate LAN Wins", use_container_width=True, type="primary"):
                    _now = datetime.now()
                    _calcs_all = []
                    for _e in st.session_state.lan_sim_entries:
                        _md = _now - timedelta(days=_e["days_ago"])
                        _aw = age_weight(_md, _now)
                        _calcs_all.append({
                            "event": _e["event"], "is_lan": _e["is_lan"],
                            "aw": _aw, "entry": _aw if _e["is_lan"] else 0.0,
                        })
                    _lan_only = sorted(
                        [x for x in _calcs_all if x["is_lan"]],
                        key=lambda x: x["entry"], reverse=True)
                    _excl     = [x for x in _calcs_all if not x["is_lan"]]
                    _top10    = _lan_only[:10]
                    _lan_val  = sum(x["entry"] for x in _top10) / 10
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
                        f'<td style="padding:5px 6px;font-size:11px;color:#c9d1d9">{_x["event"]}{_badge}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#f85149;text-align:center">🖥️</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#79c0ff;text-align:right">{_x["aw"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#f85149;font-weight:700;text-align:right">{_x["entry"]:.4f}</td>'
                        f'</tr>'
                    )
                for _x in _r["excl"]:
                    _rows += (
                        f'<tr style="background:#161b22;opacity:0.4">'
                        f'<td style="padding:5px 6px;font-size:11px;color:#8b949e">{_x["event"]}'
                        f'<span style="background:#484f58;color:#c9d1d9;border-radius:3px;padding:0 4px;font-size:9px;margin-left:4px;">online — excluded</span></td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#484f58;text-align:center">🌐</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#484f58;text-align:right">{_x["aw"]:.3f}</td>'
                        f'<td style="padding:5px 6px;font-size:11px;color:#484f58;text-align:right">—</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:10px;">'
                    f'<table style="width:100%;border-collapse:collapse">'
                    f'<thead><tr style="border-bottom:1px solid #30363d">'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:left">Event</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:center">Type</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">AgeW</th>'
                    f'<th style="padding:5px 6px;font-size:11px;color:#8b949e;text-align:right">Entry</th>'
                    f'</tr></thead><tbody>{_rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;">'
                    f'<div style="display:flex;flex-direction:column;gap:8px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:12px;color:#8b949e;">LAN wins</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#c9d1d9;">{len(_r["lan_only"])} total · top {min(10, len(_r["lan_only"]))} used</span></div>'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:12px;color:#8b949e;">① Sum top {min(10, len(_r["lan_only"]))}</span>'
                    f'<span style="font-size:13px;font-weight:700;color:#c9d1d9;">{sum(x["entry"] for x in _r["top10"]):.4f}</span></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;border-top:1px solid #30363d;padding-top:8px;">'
                    f'<span style="font-size:12px;color:#8b949e;">② LAN = &#x3A3; / 10</span>'
                    f'<span style="font-size:30px;font-weight:700;color:#f85149;">{_r["lan_val"]:.4f}</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
            elif st.session_state.lan_sim_entries:
                st.info("Click **Calculate LAN Wins** to see results.", icon="💡")
            else:
                st.info("Add at least one win to start.", icon="💡")

    # ══════════════════════════════════════════════════════════════
    with tab_h2h:
        st.subheader("⚔️ Phase 2 — Head-to-Head (Glicko/Elo)")
        st.markdown(
            '<span class="tag-v">✓ VERIFIED — K = BASE_K × age_weight only; no event stakes; no LAN mult</span>',
            unsafe_allow_html=True,
        )
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown(r"""
Starting from their seed, each team's rating is updated by every match in the window,
processed **chronologically (oldest first)**.

**Glicko constants (from spec):**
$$Q = \frac{\ln 10}{400} \approx 0.005756 \qquad \text{RD} = 75 \text{ (fixed)}$$
$$g(\text{RD}) = \frac{1}{\sqrt{1 + \frac{3Q^2 \cdot \text{RD}^2}{\pi^2}}} \approx 0.9728$$

**Expected win probability:**
$$E(A\text{ vs }B) = \frac{1}{1 + 10^{-g(\text{RD})\,\cdot\,(r_A - r_B)\,/\,400}}$$

**Rating update after each match:**
$$K = 32 \times \text{age\_weight}$$
$$\Delta r_{\text{winner}} = K \cdot (1 - E_{\text{winner}})$$
$$\Delta r_{\text{loser}}  = K \cdot (0 - E_{\text{winner}}) = -K \cdot E_{\text{winner}}$$

**⚠️ Corrections vs v2:**
- K uses **age_weight only** — event stakes removed (confirmed: "adjusted by Age Weight and not Event Weight")
- **No LAN multiplier** — LAN only affects the LAN Wins seeding factor
- BASE_K = **32** (not 28)

**Why chronological order matters:**
A win in month 1 raises the winner's rating, making their later wins more
valuable (beating a higher-rated opponent). Processing in reverse order would
produce different, incorrect results.
""")
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

            st.markdown("**Gain by rating difference (same K):**")
            diffs  = [-400, -200, -100, 0, 100, 200, 400]
            gains  = [K_eff * (1 - expected_win(1200+d/2, 1200-d/2)) for d in diffs]
            fig_sh = go.Figure(go.Bar(
                x=[f"{d:+d}" for d in diffs], y=gains,
                marker_color=["#f85149" if d < 0 else "#3fb950" if d > 0 else "#f0b429"
                              for d in diffs],
                text=[f"+{v:.1f}" for v in gains], textposition="outside",
            ))
            fig_sh.update_layout(
                xaxis=dict(title="Winner − Loser rating", gridcolor="#21262d"),
                yaxis=dict(title="Winner rating gain", gridcolor="#21262d"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
                margin=dict(l=10,r=10,t=10,b=10), height=240)
            st.plotly_chart(fig_sh, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    with tab_seed:
        st.subheader("🌱 Phase 1 — Combining Factors → Factor Score")
        st.markdown(
            '<span class="tag-v">✓ VERIFIED — simple average (25% each) confirmed from Vitality data</span>',
            unsafe_allow_html=True,
        )
        col_l, col_r = st.columns([3, 2])
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
        with col_r:
            st.markdown("#### Factor importance — equal weights confirmed")
            fig_weights = go.Figure(go.Pie(
                labels=["Bounty Offered (25%)", "Bounty Collected (25%)",
                        "Opp. Network (25%)", "LAN Wins (25%)"],
                values=[25, 25, 25, 25],
                marker_colors=["#f0b429", "#3fb950", "#79c0ff", "#f85149"],
                textinfo="label+percent", hole=0.45,
            ))
            fig_weights.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"),
                showlegend=False, height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_weights, use_container_width=True)
            st.markdown("""
**Why lerp (not just sort)?**

The lerp preserves the *proportional gaps* between teams.
A team at average=0.6 is not just ranked below 0.7 — they receive a specific
seed score of `400 + (0.6/max)×1600` that reflects how far behind they are.
This proportional gap then influences the H2H expected scores in Phase 2.
""")

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
    if sim_active and sel_team in original_standings["team"].values:
        _orig_matches = original_standings[original_standings["team"] == sel_team].sort_values("rank")
        _orig_ex = _orig_matches.iloc[0]

    if sim_active:
        st.markdown(
            '<div style="background:#1a1040;border:1px solid #7c3aed;border-radius:8px;'
            'padding:10px 14px;font-size:12px;color:#c4b5fd;margin-bottom:12px;">'
            '🔮 <strong>Simulation active</strong> — All values reflect the '
            f'simulated cutoff ({sim_cutoff_dt.strftime("%b %d, %Y")}). '
            'Age weights, H2H adjustments, and factor scores are fully recalculated.'
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
        """Return the effective age weight for a match (sim-aware)."""
        if sim_active:
            return age_weight(m["date"], sim_cutoff_dt)
        return m["age_w"]

    def _eff_h2h(m, team):
        """Return the effective H2H adjustment for a match (sim-aware)."""
        if sim_active:
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
            _new_label = sim_cutoff_dt.strftime("%b")
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
            st.plotly_chart(fig_wf, use_container_width=True)

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
            _sim_pool  = base_standings[~base_standings["team"].isin(_sim_dup_teams)]["team"].nunique()
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
                    if sim_active:
                        try:
                            dt = datetime.strptime(str(bp["event_date"]).strip(), "%Y-%m-%d")
                            bp_copy["age_weight"] = age_weight(dt, sim_cutoff_dt)
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
        st.plotly_chart(fig_p2, use_container_width=True)

        # All matches for H2H (newest first)
        if raw_ms:
            # Filter to matches in the sim window if active
            _h2h_ms = raw_ms
            if sim_active:
                _new_ws = sim_cutoff_dt - timedelta(days=DECAY_DAYS)
                _h2h_ms = [m for m in raw_ms if _new_ws <= m["date"] <= sim_cutoff_dt]

            m_rows = []
            total_h2h = 0.0
            for m in sorted(_h2h_ms, key=lambda x: x["date"], reverse=True):
                h2h      = _eff_h2h(m, sel_team)
                total_h2h += h2h
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
        else:
            st.info("No match history available.")

    # ═══════════════════════════════════════════════════════════════
    with tab_hist:
    # ═══════════════════════════════════════════════════════════════

        @st.cache_data(ttl=3600, show_spinner=False)
        def _load_team_history(team_name: str, all_dates) -> pd.DataFrame:
            rows = []
            for ds, yr in all_dates:
                gd = load_valve_github_data(ds, yr)
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
        if sim_active and not hist_df.empty and sel_team in base_standings["team"].values:
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
            st.plotly_chart(fig_dual, use_container_width=True)

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
            st.plotly_chart(fig_f, use_container_width=True)

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
