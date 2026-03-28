"""
CS2 Valve Regional Standings (VRS) Simulator  ·  v2.0
======================================================
Corrected implementation based on the official Valve VRS specification.
Source: https://github.com/ValveSoftware/counter-strike_regional_standings

Architecture
────────────
  Final Score = Seeding Score (400–2000) + Glicko H2H Adjustments

Seeding Factors  (Top-10 "bucket" results over a 6-month window)
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
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CS2 VRS Simulator",
    page_icon="🎯",
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
# ████████████  SEED DATA  ██████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════

TEAM_META: dict[str, dict] = {
    # Europe
    "Vitality":    {"region": "Europe",   "flag": "🇫🇷", "color": "#F5A623"},
    "PARIVISION":  {"region": "Europe",   "flag": "🇷🇺", "color": "#E91E63"},
    "NAVI":        {"region": "Europe",   "flag": "🇺🇦", "color": "#FFD600"},
    "Spirit":      {"region": "Europe",   "flag": "🇷🇺", "color": "#7B68EE"},
    "MOUZ":        {"region": "Europe",   "flag": "🇩🇪", "color": "#E53935"},
    "FaZe":        {"region": "Europe",   "flag": "🌍",  "color": "#EC407A"},
    "Aurora":      {"region": "Europe",   "flag": "🇷🇺", "color": "#00BCD4"},
    "G2":          {"region": "Europe",   "flag": "🇪🇸", "color": "#F44336"},
    "3DMAX":       {"region": "Europe",   "flag": "🇫🇷", "color": "#607D8B"},
    "Astralis":    {"region": "Europe",   "flag": "🇩🇰", "color": "#1565C0"},
    "NIP":         {"region": "Europe",   "flag": "🇸🇪", "color": "#1A237E"},
    "B8":          {"region": "Europe",   "flag": "🇺🇦", "color": "#4CAF50"},
    "GamerLegion": {"region": "Europe",   "flag": "🇩🇰", "color": "#9C27B0"},
    "Falcons":     {"region": "Europe",   "flag": "🇸🇦", "color": "#0288D1"},
    "FUT":         {"region": "Europe",   "flag": "🇹🇷", "color": "#FF9800"},
    # Americas
    "FURIA":       {"region": "Americas", "flag": "🇧🇷", "color": "#F5A623"},
    "Liquid":      {"region": "Americas", "flag": "🇺🇸", "color": "#00B0FF"},
    "NRG":         {"region": "Americas", "flag": "🇺🇸", "color": "#FF5722"},
    "M80":         {"region": "Americas", "flag": "🇺🇸", "color": "#78909C"},
    "Imperial":    {"region": "Americas", "flag": "🇧🇷", "color": "#7B1FA2"},
    "paiN":        {"region": "Americas", "flag": "🇧🇷", "color": "#D32F2F"},
    "9z":          {"region": "Americas", "flag": "🇦🇷", "color": "#1976D2"},
    "Cloud9":      {"region": "Americas", "flag": "🇺🇸", "color": "#29B6F6"},
    # Asia
    "MongolZ":     {"region": "Asia",     "flag": "🇲🇳", "color": "#FF6F00"},
    "TYLOO":       {"region": "Asia",     "flag": "🇨🇳", "color": "#C62828"},
    "Rare Atom":   {"region": "Asia",     "flag": "🇨🇳", "color": "#00897B"},
    "ThunderPick": {"region": "Asia",     "flag": "🇦🇺", "color": "#5E35B1"},
    "Grayhound":   {"region": "Asia",     "flag": "🇦🇺", "color": "#546E7A"},
}

# Prize placement splits (winner's share of total pool)
PRIZE_SPLITS = {1: 0.35, 2: 0.20, 3: 0.12, 4: 0.08,
                5: 0.05, 6: 0.04, 7: 0.03, 8: 0.02}


def placement_prize(pool: float, place: int) -> float:
    return pool * PRIZE_SPLITS.get(place, 0.01)


# Event catalogue  —  (name, prize_pool_usd, is_lan, date_str)
EVENT_LIST = [
    ("IEM Krakow 2026",           1_000_000, True,  "2026-01-22"),
    ("PGL Cluj-Napoca 2026",      1_000_000, True,  "2026-02-16"),
    ("ESL Pro League S21",          750_000, True,  "2026-01-10"),
    ("BLAST Bounty S1 Finals",      500_000, True,  "2025-12-18"),
    ("BLAST Premier Fall 2025",     500_000, True,  "2025-11-08"),
    ("IEM Dallas 2025",           1_000_000, True,  "2025-10-12"),
    ("PGL Astana 2025",           1_000_000, True,  "2025-09-20"),
    ("BLAST Open Rotterdam 2026",   250_000, True,  "2026-03-14"),
    ("CCT Season 3 EU Finals",      200_000, False, "2026-02-02"),
    ("ESL Challenger S47",          100_000, False, "2026-01-05"),
    ("Fragadelphia Miami 2026",      50_000, True,  "2026-01-12"),
    ("ESEA Premier S52 AM",          30_000, False, "2025-12-05"),
    ("ESL Challenger AM S47",        80_000, False, "2025-11-20"),
    ("PGL Wallachia S4",            200_000, True,  "2025-10-28"),
    ("IEM Sydney 2025",             250_000, True,  "2025-09-05"),
]

_EVENT_LOOKUP = {name: (pool, lan, date) for name, pool, lan, date in EVENT_LIST}


def _match(winner, loser, event_name, w_place, l_place):
    """Helper to build a single match record dict."""
    pool, lan, date_str = _EVENT_LOOKUP[event_name]
    base = datetime.strptime(date_str, "%Y-%m-%d")
    # Stagger by place so final rounds are a few days after group stage
    return dict(
        date=base + timedelta(days=max(w_place, l_place)),
        winner=winner, loser=loser, event=event_name,
        prize_pool=float(pool),
        winner_prize=placement_prize(pool, w_place),
        loser_prize=placement_prize(pool, l_place),
        is_lan=lan,
    )


@st.cache_data
def generate_seed_matches() -> pd.DataFrame:
    """
    Realistic match history producing VRS standings close to the
    real-world March 2026 snapshot.
    """
    M = _match  # alias for brevity

    blueprints = [
        # ─── IEM Krakow 2026  (Jan 2026, $1M LAN) ──────────────────
        M("Vitality",   "PARIVISION", "IEM Krakow 2026", 1, 2),
        M("Vitality",   "NAVI",       "IEM Krakow 2026", 1, 3),
        M("PARIVISION", "NAVI",       "IEM Krakow 2026", 2, 3),
        M("PARIVISION", "Spirit",     "IEM Krakow 2026", 2, 4),
        M("NAVI",       "Falcons",    "IEM Krakow 2026", 3, 5),
        M("Spirit",     "Falcons",    "IEM Krakow 2026", 4, 5),
        M("Spirit",     "MOUZ",       "IEM Krakow 2026", 4, 6),
        M("Falcons",    "MOUZ",       "IEM Krakow 2026", 5, 6),
        M("MOUZ",       "G2",         "IEM Krakow 2026", 6, 7),
        M("G2",         "Aurora",     "IEM Krakow 2026", 7, 8),
        M("FURIA",      "Liquid",     "IEM Krakow 2026", 7, 8),

        # ─── PGL Cluj-Napoca 2026  (Feb 2026, $1M LAN) ─────────────
        M("Vitality",   "PARIVISION", "PGL Cluj-Napoca 2026", 1, 2),
        M("Vitality",   "Falcons",    "PGL Cluj-Napoca 2026", 1, 4),
        M("PARIVISION", "NAVI",       "PGL Cluj-Napoca 2026", 2, 3),
        M("PARIVISION", "Falcons",    "PGL Cluj-Napoca 2026", 2, 4),
        M("NAVI",       "Spirit",     "PGL Cluj-Napoca 2026", 3, 5),
        M("Falcons",    "Aurora",     "PGL Cluj-Napoca 2026", 4, 6),
        M("Spirit",     "FaZe",       "PGL Cluj-Napoca 2026", 5, 7),
        M("Aurora",     "B8",         "PGL Cluj-Napoca 2026", 6, 8),
        M("FURIA",      "Liquid",     "PGL Cluj-Napoca 2026", 7, 8),

        # ─── ESL Pro League S21  (Jan 2026, $750k LAN) ─────────────
        M("NAVI",      "G2",          "ESL Pro League S21", 1, 2),
        M("G2",        "MOUZ",        "ESL Pro League S21", 2, 3),
        M("MOUZ",      "FaZe",        "ESL Pro League S21", 3, 4),
        M("FaZe",      "Astralis",    "ESL Pro League S21", 4, 5),
        M("Astralis",  "3DMAX",       "ESL Pro League S21", 5, 6),
        M("3DMAX",     "NIP",         "ESL Pro League S21", 6, 7),
        M("FURIA",     "NRG",         "ESL Pro League S21", 1, 2),
        M("NRG",       "Liquid",      "ESL Pro League S21", 2, 3),
        M("Liquid",    "M80",         "ESL Pro League S21", 3, 4),
        M("MongolZ",   "TYLOO",       "ESL Pro League S21", 1, 2),
        M("TYLOO",     "Rare Atom",   "ESL Pro League S21", 2, 3),

        # ─── BLAST Bounty S1 Finals  (Dec 2025, $500k LAN) ─────────
        M("PARIVISION","Vitality",    "BLAST Bounty S1 Finals", 1, 2),
        M("Vitality",  "Falcons",     "BLAST Bounty S1 Finals", 2, 3),
        M("Falcons",   "NAVI",        "BLAST Bounty S1 Finals", 3, 4),
        M("NAVI",      "FaZe",        "BLAST Bounty S1 Finals", 4, 5),
        M("FaZe",      "Spirit",      "BLAST Bounty S1 Finals", 5, 6),
        M("Spirit",    "MOUZ",        "BLAST Bounty S1 Finals", 6, 7),
        M("FURIA",     "Liquid",      "BLAST Bounty S1 Finals", 5, 6),
        M("MongolZ",   "TYLOO",       "BLAST Bounty S1 Finals", 3, 4),

        # ─── BLAST Premier Fall 2025  (Nov 2025, $500k LAN) ────────
        M("Spirit",    "MOUZ",        "BLAST Premier Fall 2025", 1, 2),
        M("MOUZ",      "FaZe",        "BLAST Premier Fall 2025", 2, 3),
        M("FaZe",      "G2",          "BLAST Premier Fall 2025", 3, 4),
        M("G2",        "Astralis",    "BLAST Premier Fall 2025", 4, 5),
        M("Astralis",  "NAVI",        "BLAST Premier Fall 2025", 5, 6),
        M("NAVI",      "Vitality",    "BLAST Premier Fall 2025", 6, 7),
        M("Vitality",  "3DMAX",       "BLAST Premier Fall 2025", 7, 8),
        M("FURIA",     "Liquid",      "BLAST Premier Fall 2025", 1, 2),
        M("Liquid",    "NRG",         "BLAST Premier Fall 2025", 2, 3),
        M("MongolZ",   "TYLOO",       "BLAST Premier Fall 2025", 1, 2),
        M("TYLOO",     "Rare Atom",   "BLAST Premier Fall 2025", 2, 3),

        # ─── IEM Dallas 2025  (Oct 2025, $1M LAN) ──────────────────
        M("Spirit",    "NAVI",        "IEM Dallas 2025", 1, 2),
        M("NAVI",      "Vitality",    "IEM Dallas 2025", 2, 3),
        M("Vitality",  "FaZe",        "IEM Dallas 2025", 3, 4),
        M("FaZe",      "MOUZ",        "IEM Dallas 2025", 4, 5),
        M("MOUZ",      "G2",          "IEM Dallas 2025", 5, 6),
        M("G2",        "Aurora",      "IEM Dallas 2025", 6, 7),
        M("FURIA",     "Liquid",      "IEM Dallas 2025", 1, 2),
        M("Liquid",    "NRG",         "IEM Dallas 2025", 2, 3),
        M("NRG",       "M80",         "IEM Dallas 2025", 3, 4),
        M("MongolZ",   "Rare Atom",   "IEM Dallas 2025", 1, 2),
        M("TYLOO",     "ThunderPick", "IEM Dallas 2025", 3, 4),

        # ─── PGL Astana 2025  (Sep 2025, $1M LAN – near decay edge) ─
        M("MOUZ",      "Spirit",      "PGL Astana 2025", 1, 2),
        M("Spirit",    "NAVI",        "PGL Astana 2025", 2, 3),
        M("NAVI",      "G2",          "PGL Astana 2025", 3, 4),
        M("G2",        "FaZe",        "PGL Astana 2025", 4, 5),
        M("FaZe",      "Falcons",     "PGL Astana 2025", 5, 6),
        M("FURIA",     "NRG",         "PGL Astana 2025", 1, 2),
        M("NRG",       "Liquid",      "PGL Astana 2025", 2, 3),
        M("MongolZ",   "TYLOO",       "PGL Astana 2025", 1, 2),
        M("Rare Atom", "ThunderPick", "PGL Astana 2025", 3, 4),

        # ─── BLAST Open Rotterdam 2026  (Mar 2026, $250k LAN) ──────
        M("Vitality",   "PARIVISION", "BLAST Open Rotterdam 2026", 1, 4),
        M("NAVI",       "Spirit",     "BLAST Open Rotterdam 2026", 2, 3),
        M("Spirit",     "Aurora",     "BLAST Open Rotterdam 2026", 3, 5),
        M("MongolZ",    "TYLOO",      "BLAST Open Rotterdam 2026", 1, 6),
        M("FURIA",      "NRG",        "BLAST Open Rotterdam 2026", 2, 7),
        M("Falcons",    "B8",         "BLAST Open Rotterdam 2026", 4, 8),

        # ─── CCT Season 3 EU Finals  (Feb 2026, $200k online) ──────
        M("B8",         "GamerLegion","CCT Season 3 EU Finals", 1, 2),
        M("GamerLegion","3DMAX",      "CCT Season 3 EU Finals", 2, 3),
        M("3DMAX",      "FUT",        "CCT Season 3 EU Finals", 3, 4),
        M("FUT",        "NIP",        "CCT Season 3 EU Finals", 4, 5),
        M("NIP",        "Astralis",   "CCT Season 3 EU Finals", 5, 6),
        M("Aurora",     "FaZe",       "CCT Season 3 EU Finals", 1, 2),

        # ─── ESL Challenger S47  (Jan 2026, $100k online) ──────────
        M("3DMAX",      "GamerLegion","ESL Challenger S47", 1, 2),
        M("GamerLegion","B8",         "ESL Challenger S47", 2, 3),
        M("B8",         "NIP",        "ESL Challenger S47", 3, 4),
        M("NIP",        "FUT",        "ESL Challenger S47", 4, 5),
        M("Grayhound",  "ThunderPick","ESL Challenger S47", 1, 2),
        M("TYLOO",      "Grayhound",  "ESL Challenger S47", 1, 2),

        # ─── Fragadelphia Miami 2026  (Jan 2026, $50k LAN) ─────────
        M("NRG",       "M80",         "Fragadelphia Miami 2026", 1, 2),
        M("M80",       "Cloud9",      "Fragadelphia Miami 2026", 2, 3),
        M("Cloud9",    "9z",          "Fragadelphia Miami 2026", 3, 4),
        M("9z",        "Imperial",    "Fragadelphia Miami 2026", 4, 5),
        M("Imperial",  "paiN",        "Fragadelphia Miami 2026", 5, 6),

        # ─── ESEA Premier S52 AM  (Dec 2025, $30k online) ──────────
        M("Liquid",    "NRG",         "ESEA Premier S52 AM", 1, 2),
        M("NRG",       "M80",         "ESEA Premier S52 AM", 2, 3),
        M("M80",       "Cloud9",      "ESEA Premier S52 AM", 3, 4),
        M("Cloud9",    "paiN",        "ESEA Premier S52 AM", 4, 5),
        M("paiN",      "9z",          "ESEA Premier S52 AM", 5, 6),
        M("9z",        "Imperial",    "ESEA Premier S52 AM", 6, 7),

        # ─── ESL Challenger AM S47  (Nov 2025, $80k online) ────────
        M("FURIA",     "NRG",         "ESL Challenger AM S47", 1, 2),
        M("NRG",       "Liquid",      "ESL Challenger AM S47", 2, 3),
        M("Liquid",    "M80",         "ESL Challenger AM S47", 3, 4),
        M("M80",       "Cloud9",      "ESL Challenger AM S47", 4, 5),
        M("Imperial",  "paiN",        "ESL Challenger AM S47", 5, 6),

        # ─── PGL Wallachia S4  (Oct 2025, $200k LAN) ───────────────
        M("Spirit",    "Falcons",     "PGL Wallachia S4", 1, 2),
        M("Falcons",   "MOUZ",        "PGL Wallachia S4", 2, 3),
        M("MOUZ",      "Aurora",      "PGL Wallachia S4", 3, 4),
        M("Aurora",    "B8",          "PGL Wallachia S4", 4, 5),
        M("B8",        "GamerLegion", "PGL Wallachia S4", 5, 6),
        M("GamerLegion","3DMAX",      "PGL Wallachia S4", 6, 7),

        # ─── IEM Sydney 2025  (Sep 2025, $250k LAN) ─────────────────
        M("MongolZ",   "Rare Atom",   "IEM Sydney 2025", 1, 2),
        M("Rare Atom", "TYLOO",       "IEM Sydney 2025", 2, 3),
        M("TYLOO",     "ThunderPick", "IEM Sydney 2025", 3, 4),
        M("ThunderPick","Grayhound",  "IEM Sydney 2025", 4, 5),
        M("Grayhound", "ThunderPick", "IEM Sydney 2025", 5, 6),
    ]

    return pd.DataFrame(blueprints)


# ══════════════════════════════════════════════════════════════════
# HELPERS  —  formatting utilities
# ══════════════════════════════════════════════════════════════════

def region_pill(region: str) -> str:
    cls = {"Europe": "pill-eu", "Americas": "pill-am", "Asia": "pill-as"}.get(region, "pill-eu")
    return f'<span class="{cls}">{region}</span>'


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
    df["region"] = df["team"].map(lambda t: TEAM_META.get(t, {}).get("region", "Europe"))
    df["flag"]   = df["team"].map(lambda t: TEAM_META.get(t, {}).get("flag", "🌍"))
    return df


def add_regional_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regional_rank"] = (df.groupby("region")["total_points"]
                             .rank(ascending=False, method="first")
                             .astype(int))
    return df


@st.cache_data(ttl=3600)
def fetch_live_standings(region: str) -> pd.DataFrame | None:
    rmap = {"Europe": "standings_europe.md",
            "Americas": "standings_americas.md",
            "Asia": "standings_asia.md"}
    url = (f"https://raw.githubusercontent.com/ValveSoftware/"
           f"counter-strike_regional_standings/main/{rmap.get(region,'')}")
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        lines = [l for l in r.text.splitlines() if l.startswith("|") and "---" not in l]
        if len(lines) < 2:
            return None
        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
        records = []
        for line in lines[1:]:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) == len(headers):
                records.append(dict(zip(headers, cells)))
        return pd.DataFrame(records) if records else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎯 CS2 VRS Simulator")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Ranking Dashboard",
        "🔮 Scenario Simulator",
        "⚔️ Team Comparison",
        "📖 How VRS Works",
    ], label_visibility="collapsed")
    st.markdown("---")
    cutoff_date = st.date_input(
        "📅 Standings Cutoff Date",
        value=datetime(2026, 3, 2),
        help="View how standings looked at any point in time.",
    )
    cutoff_dt = datetime.combine(cutoff_date, datetime.min.time())
    st.markdown("---")
    st.markdown(
        """
**Engine v2.0**
- ✅ Two-phase: Seed + H2H
- ✅ Correct time decay (0→1)
- ✅ Curve function f(x)
- ✅ Top-10 buckets per factor
- ✅ Glicko H2H (RD=75)
- ✅ Event stakes
- ✅ Eligibility filter

📄 [Valve Repo](https://github.com/ValveSoftware/counter-strike_regional_standings)
""")


# ══════════════════════════════════════════════════════════════════
# BASE COMPUTATION
# ══════════════════════════════════════════════════════════════════

base_matches = generate_seed_matches()


def compute_standings(extra_matches: pd.DataFrame = None, cutoff: datetime = None):
    if cutoff is None:
        cutoff = cutoff_dt
    matches = base_matches.copy()
    if extra_matches is not None and not extra_matches.empty:
        matches = pd.concat([matches, extra_matches], ignore_index=True)
    result = compute_vrs(matches, cutoff=cutoff)
    if result.empty:
        return result
    result = add_meta(result)
    result = add_regional_rank(result)
    return result


base_standings = compute_standings()


# ══════════════════════════════════════════════════════════════════
# PAGE 1  ·  RANKING DASHBOARD
# ══════════════════════════════════════════════════════════════════

if page == "📊 Ranking Dashboard":
    st.title("📊 CS2 Valve Regional Standings")
    st.caption(
        f"Simulated standings · cutoff **{cutoff_date.strftime('%B %d, %Y')}** · "
        "Two-phase VRS engine (Seed + Glicko H2H) · "
        "Source: [Valve GitHub](https://github.com/ValveSoftware/counter-strike_regional_standings)"
    )

    if base_standings.empty:
        st.error("No eligible teams for the selected cutoff window.")
        st.stop()

    # ── Live data fetch attempt ───────────────────────────────────
    with st.expander("🔴 Live Valve Data — direct from GitHub", expanded=False):
        live_eu = fetch_live_standings("Europe")
        if live_eu is not None:
            st.success("✅ Live Europe standings fetched from Valve's GitHub repo")
            st.dataframe(live_eu, use_container_width=True)
        else:
            st.info("Could not fetch live data (network or rate-limit). Simulator uses built-in dataset below.")

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

    st.markdown("---")

    # ── Table renderer ────────────────────────────────────────────
    def render_table(df: pd.DataFrame, show_all: bool = False):
        display = df.copy() if show_all else df.head(25)
        rows = []
        for _, row in display.iterrows():
            rk   = int(row["rank"])
            flag = row.get("flag", "🌍")
            team = row["team"]
            pts  = f'{row["total_points"]:,.1f}'
            seed = f'{row["seed"]:,.1f}'
            h2h  = row["h2h_delta"]
            h2h_s = (f'<span class="change-up">+{h2h:.1f}</span>' if h2h > 0 else
                     f'<span class="change-down">{h2h:.1f}</span>'  if h2h < 0 else
                     '<span class="change-same">0.0</span>')
            reg  = region_pill(row["region"])
            rrk  = int(row["regional_rank"])
            w    = int(row["wins"])
            l    = int(row["losses"])
            rows.append(f"""
            <tr>
              <td style="text-align:center">{rank_badge(rk)}</td>
              <td>{flag} <strong>{team}</strong></td>
              <td style="text-align:center">{reg}</td>
              <td style="text-align:right; color:#58a6ff; font-weight:700">{pts}</td>
              <td style="text-align:right; color:#79c0ff">{seed}</td>
              <td style="text-align:right">{h2h_s}</td>
              <td style="text-align:center; color:#8b949e">{w}W / {l}L</td>
              <td style="text-align:center; color:#8b949e">{rrk}</td>
            </tr>""")
        st.markdown(f"""
        <table style="width:100%; border-collapse:collapse; font-size:13px;">
          <thead>
            <tr style="background:#21262d; color:#8b949e; font-size:11px; text-transform:uppercase;">
              <th style="padding:10px 5px; text-align:center">Rank</th>
              <th style="padding:10px 5px; text-align:left">Team</th>
              <th style="padding:10px 5px; text-align:center">Region</th>
              <th style="padding:10px 5px; text-align:right" title="Seed + H2H">Total Pts</th>
              <th style="padding:10px 5px; text-align:right" title="Seeding score (400-2000)">Seed</th>
              <th style="padding:10px 5px; text-align:right" title="Glicko H2H adjustment">H2H Δ</th>
              <th style="padding:10px 5px; text-align:center">Record</th>
              <th style="padding:10px 5px; text-align:center">Reg. Rank</th>
            </tr>
          </thead>
          <tbody>{"".join(rows)}</tbody>
        </table>""", unsafe_allow_html=True)

    tab_gl, tab_eu, tab_am, tab_as = st.tabs(
        ["🌍 Global", "🇪🇺 Europe", "🌎 Americas", "🌏 Asia"])

    with tab_gl:
        show_all = st.checkbox("Show all teams", key="chk_global")
        render_table(base_standings, show_all=show_all)

    with tab_eu:
        eu_df = base_standings[base_standings["region"] == "Europe"].reset_index(drop=True)
        eu_df["rank"] = eu_df.index + 1
        render_table(eu_df, show_all=True)

    with tab_am:
        am_df = base_standings[base_standings["region"] == "Americas"].reset_index(drop=True)
        am_df["rank"] = am_df.index + 1
        render_table(am_df, show_all=True)

    with tab_as:
        as_df = base_standings[base_standings["region"] == "Asia"].reset_index(drop=True)
        as_df["rank"] = as_df.index + 1
        render_table(as_df, show_all=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 15 — Seed vs H2H Breakdown")
        top15 = base_standings.head(15).copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Seed (400–2000)", x=top15["team"], y=top15["seed"],
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
        st.subheader("🕸️ Top 10 — Seeding Factor Radar")
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

    all_teams = sorted(TEAM_META.keys())

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
        st.subheader("🕸️ Seeding Factor Radar")
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

    st.title("📖 How the VRS is Calculated")
    st.markdown(
        "Complete technical breakdown — every formula verified against "
        "Valve's published Vitality detail sheet (March 2026 cutoff)."
    )
    st.markdown(
        '<span class="tag-v">✓ VERIFIED</span>&nbsp; Formula confirmed from official data &nbsp;&nbsp;'
        '<span class="tag-a">~ ASSUMED</span>&nbsp; Not explicitly stated in public spec',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tab_arch, tab_age, tab_curve, tab_bo, tab_bc, tab_on, tab_lan, tab_h2h, tab_seed, tab_worked = st.tabs([
        "🏗️ Architecture",
        "⏳ Age Weight",
        "📐 Curve f(x)",
        "🏆 Bounty Offered",
        "💰 Bounty Collected",
        "🕸️ Opp. Network",
        "🖥️ LAN Wins",
        "⚔️ H2H (Glicko)",
        "🌱 Seeding",
        "🔍 Worked Example",
    ])

    # ══════════════════════════════════════════════════════════════
    with tab_arch:
        st.subheader("Two-Phase Architecture")
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown("""
The VRS is computed in **two sequential phases** that are simply added together:
""")
            st.latex(
                r"\text{Final Score} = "
                r"\underbrace{\text{Seed}_{[400,\;2000]}}_{\text{Phase 1 — Seeding}} "
                r"+ \;\underbrace{\Delta_{\text{H2H}}}_{\text{Phase 2 — Head-to-Head}}"
            )
            st.markdown("""
**Phase 1 — Seeding** asks: *"What is this team's overall quality over the last 6 months?"*

Four factors are computed, averaged (25% each), and mapped to [400, 2000]. This is the
team's **starting rank value** for Phase 2.

**Phase 2 — Head-to-Head** asks: *"How did this team perform against specific opponents?"*

Starting from their seed, every match in the window is replayed chronologically using a
Glicko/Elo system. The cumulative rating change is the H2H adjustment.

**Data window:** Only matches within the last **180 days** contribute. Older matches are
excluded entirely. Within the window, **age weight** scales contributions continuously.
""")
            st.markdown('<span class="tag-v">✓ VERIFIED</span> — from official team detail sheets', unsafe_allow_html=True)

        with col_r:
            # Simple flow diagram
            fig = go.Figure()
            nodes = [
                (0.5, 0.88, "180-day match history", "#21262d", "#8b949e", 13),
                (0.25, 0.65, "Phase 1\nSeeding\n(4 factors)", "#0d1a2e", "#58a6ff", 14),
                (0.75, 0.65, "Phase 2\nHead-to-Head\n(Glicko, chronological)", "#0d1a0d", "#3fb950", 13),
                (0.5,  0.40, "Seed + H2H Δ", "#1c1c1c", "#c9d1d9", 13),
                (0.5,  0.17, "Final VRS Score", "#2d1f00", "#f0b429", 15),
            ]
            for x, y, text, bg, fc, fs in nodes:
                fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                    bgcolor=bg, bordercolor=fc, borderwidth=1.5, borderpad=9,
                    font=dict(color=fc, size=fs), xref="paper", yref="paper", align="center")
            for x0, y0, x1, y1 in [
                (0.5,0.84,0.3,0.76),(0.5,0.84,0.7,0.76),
                (0.3,0.55,0.44,0.47),(0.7,0.55,0.56,0.47),(0.5,0.33,0.5,0.24)]:
                fig.add_annotation(ax=x0, ay=y0, x=x1, y=y1,
                    axref="paper", ayref="paper", xref="paper", yref="paper",
                    arrowhead=2, arrowwidth=1.5, arrowcolor="#8b949e", showarrow=True, text="")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False, range=[0,1]), yaxis=dict(visible=False, range=[0,1]),
                height=360, margin=dict(l=5,r=5,t=5,b=5))
            st.plotly_chart(fig, use_container_width=True)

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

It is used in **three places:**

| Usage | Input x | Applied to |
|---|---|---|
| **Event stakes** | `pool / $1,000,000` | BC, ON (not BO or LAN) |
| **BO normalisation** | `min(1.0, bo_sum / 5th_ref)` | Final BO factor |
| **BC normalisation** | `Σ_top10_adjusted / 10` | Final BC factor |

**Key properties:**
- `f(1.0) = 1.000` — at the reference, full value
- `f(0.5) ≈ 0.768` — half the reference
- `f(0.1) = 0.500` — one-tenth of reference
- `f(0.01) = 0.333` — one-hundredth
- Peaks at x = 1; symmetric in log-space (f(0.1) = f(10))

**Why log-scale?** A $100k event is roughly as different from $10k as $1M is from $100k.
The curve distributes "importance" evenly across orders of magnitude.
""")
            st.markdown("**Event stakes reference table:**")
            ref = [(1_000_000,"$1M (Major)",),(500_000,"$500k"),(250_000,"$250k"),
                   (100_000,"$100k"),(50_000,"$50k"),(10_000,"$10k")]
            st.dataframe(pd.DataFrame([{
                "Prize Pool": lbl, "x = pool/$1M": f"{p/1e6:.3f}",
                "Event Stakes f(x)": f"{event_stakes(p):.4f}",
            } for p,lbl in ref]), use_container_width=True, hide_index=True)

        with col_r:
            st.markdown("#### 🎛️ Curve Calculator")
            pool_calc = st.number_input("Prize pool (USD)", 1000, 2_000_000, 250_000, 10_000, key="cv_pool")
            x_c = min(float(pool_calc), PRIZE_CAP) / PRIZE_CAP
            fx_c = curve(x_c)
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                        padding:16px;text-align:center;">
              <div style="font-size:12px;color:#8b949e">x = ${min(pool_calc,PRIZE_CAP):,.0f} / $1,000,000</div>
              <div style="font-size:28px;font-weight:700;color:#79c0ff;margin:4px 0">x = {x_c:.4f}</div>
              <div style="font-size:12px;color:#8b949e">f(x) = 1 / (1 + |log₁₀({x_c:.4f})|)</div>
              <div style="font-size:38px;font-weight:700;color:#f0b429;margin-top:6px">{fx_c:.4f}</div>
              <div style="font-size:12px;color:#8b949e">Event Stakes</div>
            </div>""", unsafe_allow_html=True)

            xs_c = [i/200 for i in range(1, 201)]
            ys_c = [curve(x) for x in xs_c]
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=[xi*1e6 for xi in xs_c], y=ys_c,
                mode="lines", line=dict(color="#f0b429", width=2.5)))
            fig_c.add_trace(go.Scatter(x=[pool_calc], y=[fx_c], mode="markers",
                marker=dict(color="#58a6ff", size=12, symbol="diamond"), showlegend=False))
            fig_c.update_layout(
                xaxis=dict(title="Prize Pool (USD)", gridcolor="#21262d",
                    tickvals=[0,100_000,250_000,500_000,750_000,1_000_000],
                    ticktext=["$0","$100k","$250k","$500k","$750k","$1M"]),
                yaxis=dict(title="f(x)", gridcolor="#21262d", range=[0,1.1]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10), height=280)
            st.plotly_chart(fig_c, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    with tab_bo:
        st.subheader("🏆 Factor 1 — Bounty Offered")
        st.markdown('<span class="tag-v">✓ VERIFIED — sum / 5th-ref / curve(min(1,x)) confirmed</span>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown(r"""
Bounty Offered (BO) measures how much **prize money a team has earned**, scaled for recency.

**Step-by-step:**

**Step 1.** For each win, compute the scaled prize:
$$\text{contribution}_i = \text{prize\_won}_i \;\times\; \text{age\_weight}_i$$

⚠️ **No event stakes.** BO uses only age weight — confirmed from the official data where
"Scaled Winnings = Prize × Age Weight" with no event-stakes column.

**Step 2.** Sum the **top 10** contributions across all wins:
$$\text{BO\_sum} = \sum_{i \in \text{top-10}} \text{prize\_won}_i \times \text{age\_weight}_i$$

**Step 3.** Find the **5th-highest BO\_sum** across all teams in the dataset. Call it `ref₅`.
Teams ranked 1–5 by prize money all have `ratio ≥ 1.0`.

**Step 4.** Clamp and apply the curve:
$$\text{BO} = f\!\left(\min\!\left(1.0,\;\frac{\text{BO\_sum}}{\text{ref}_5}\right)\right)$$

Since `curve(1.0) = 1.000`, all teams with `BO_sum ≥ ref₅` receive **BO = 1.000**.
Teams below get `BO = curve(ratio)` where ratio < 1 → BO < 1.000.
""")
            st.markdown("""
**Verification — Vitality:**
- BO_sum = $1,472,719.98  ✓ (sum of 8 entries × age weights)
- ref₅ = $334,320.24
- ratio = 4.406 → min(1.0, 4.406) = 1.0 → curve(1.0) = **1.000** ✓
""")
        with col_r:
            st.markdown("#### Example contributions")
            ex_bo = [
                ("IEM Krakow (W, Feb)", datetime(2026,2,16), 350_000),
                ("PGL Cluj (W, Feb)",   datetime(2026,2,22), 460_000),
                ("BLAST Fall (W, Dec)", datetime(2025,12,14),500_000),
                ("ESL PL (3rd, Jan)",   datetime(2026,1,10),  90_000),
                ("Small LAN (W, Oct)", datetime(2025,10,12), 65_000),
            ]
            cutoff_ex = datetime(2026, 3, 2)
            ex_rows = []
            for ename, edate, prize in ex_bo:
                aw = age_weight(edate, cutoff_ex)
                contrib = prize * aw
                ex_rows.append({"Event": ename, "Prize Won": f"${prize:,.0f}",
                                 "Age Weight": f"{aw:.3f}",
                                 "Scaled (contribution)": f"${contrib:,.0f}"})
            st.dataframe(pd.DataFrame(ex_rows), use_container_width=True, hide_index=True)
            total_ex = sum(p * age_weight(d, cutoff_ex) for _,d,p in ex_bo)
            st.caption(f"BO_sum (top 5 shown) = ${total_ex:,.0f}. If ref₅ = $334,320 → ratio = {total_ex/334320:.2f} → clamped to 1.0 → BO = 1.000")

    # ══════════════════════════════════════════════════════════════
    with tab_bc:
        st.subheader("💰 Factor 2 — Bounty Collected")
        st.markdown('<span class="tag-v">✓ VERIFIED — BC = curve(Σ_top10_adjusted / 10) confirmed numerically</span>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown(r"""
Bounty Collected (BC) measures the **quality of opponents defeated**, using each
opponent's own Bounty Offered score as the proxy for their strength.

**Step-by-step:**

**Step 1.** For each win **at an event with a prize pool**, compute the adjusted entry:
$$\text{entry}_i = \text{BO\_factor}[\text{opponent}] \;\times\; \text{age\_weight}_i \;\times\; \text{event\_stakes}(\text{pool})$$

Note: matches at events **without** a prize pool (e.g. online qualifiers) do NOT
contribute to BC (event stakes = 0 for prize_pool = 0).

**Step 2.** Take the **top 10** entries, sum them, divide by 10:
$$\text{BC\_pre\_curve} = \frac{\sum_{i \in \text{top-10}} \text{entry}_i}{10}$$

**Step 3.** Apply the curve function:
$$\text{BC} = f(\text{BC\_pre\_curve})$$

**Verification — Vitality (cutoff 2026-03-02):**

| Opponent | opp BO | Age | Event | Entry |
|---|---|---|---|---|
| PARIVISION | 1.000 | 1.000 | 1.000 | 1.000 |
| FURIA | 1.000 | 1.000 | 1.000 | 1.000 |
| Aurora (×2) | 1.000 | 1.000 | 1.000 | 1.000 |
| MOUZ (×2) | 0.949 | 1.000 | 1.000 | 0.949 |
| MongolZ (×2) | 0.672 | 1.000 | 1.000 | 0.672 |
| Spirit | 0.843 | 0.674 | 1.000 | 0.568 |
| G2 | 0.446 | 1.000 | 1.000 | 0.446 |

Sum of top 10 = **8.256**, /10 = **0.8256**, curve(0.8256) = **0.923** ✓
""")
        with col_r:
            st.markdown("#### Why BC rewards beating strong teams")
            scenarios = [
                ("1 win vs #1 (BO=1.0)", [1.0]),
                ("5 wins vs mid (BO=0.5)", [0.5]*5),
                ("10 wins vs weak (BO=0.1)", [0.1]*10),
                ("Mix: 2 top + 6 mid", [1.0,1.0]+[0.5]*6),
            ]
            fig_bc = go.Figure()
            for label, entries in scenarios:
                bc_pre = sum(sorted(entries, reverse=True)[:10]) / 10
                bc_val = curve(bc_pre)
                fig_bc.add_trace(go.Bar(
                    name=label, x=[label.split("(")[0].strip()], y=[bc_val],
                    text=[f"{bc_val:.3f}"], textposition="outside",
                    marker_color="#3fb950" if bc_val > 0.8 else
                                 "#f0b429" if bc_val > 0.5 else "#f85149",
                    showlegend=False,
                ))
            fig_bc.update_layout(
                yaxis=dict(title="BC factor (after curve)", gridcolor="#21262d", range=[0,1.15]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), xaxis=dict(gridcolor="#21262d"),
                margin=dict(l=10,r=10,t=10,b=10), height=320)
            st.plotly_chart(fig_bc, use_container_width=True)
            st.caption("Beating one strong opponent can be worth more than beating 10 weak ones.")

    # ══════════════════════════════════════════════════════════════
    with tab_on:
        st.subheader("🕸️ Factor 3 — Opponent Network")
        st.markdown('<span class="tag-v">✓ VERIFIED — ON = Σ_top10_adjusted / 10, NO curve, confirmed numerically</span>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown(r"""
Opponent Network (ON) measures the **depth of the network** of opponents beaten.
The key difference from BC: ON uses the opponent's **own ON factor** (not their BO),
creating a recursive PageRank-like structure.

**Step-by-step:**

**Step 1.** For each win **at an event with a prize pool**, compute:
$$\text{entry}_i = \text{ON\_factor}[\text{opponent}] \;\times\; \text{age\_weight}_i \;\times\; \text{event\_stakes}(\text{pool})$$

**Step 2.** Take the **top 10** entries, sum, divide by 10:
$$\text{ON} = \frac{\sum_{i \in \text{top-10}} \text{entry}_i}{10}$$

**⚠️ No curve applied to ON** (unlike BC).

**The circular dependency:**

ON for team A uses ON of all opponents of A. But those opponents' ON values use ON
of *their* opponents. This is circular — no closed-form solution exists.

**Solution: iterative computation (like PageRank):**
1. Initialise every team's ON with their BO factor (a reasonable prior)
2. Compute new ON values for all teams simultaneously using current ON estimates
3. Repeat 6 times — values converge quickly

**Verification — Vitality:**

| Opponent | opp ON | Age | Event | Entry |
|---|---|---|---|---|
| PARIVISION | 0.622 | 1.000 | 1.000 | 0.622 |
| EYEBALLERS | 0.707 | 0.914 | 0.895 | **0.578** |
| GamerLegion | 0.663 | 0.947 | 0.899 | **0.565** |
| FURIA | 0.536 | 1.000 | 1.000 | 0.536 |
| BC.Game | 0.521 | 1.000 | 1.000 | 0.521 |
| MOUZ (×2) | 0.385 | 1.000 | 1.000 | 0.385 |
| Aurora (×2) | 0.351 | 1.000 | 1.000 | 0.351 |
| G2 | 0.305 | 1.000 | 1.000 | 0.305 |

Sum of top 10 = **4.599**, /10 = **0.460** ✓
""")
        with col_r:
            st.markdown("#### ON vs BC — what each rewards")
            st.markdown("""
| Scenario | BO rewards | ON rewards |
|---|---|---|
| Beat a rich team | ✅ High BC | Depends on their network |
| Beat a team with many wins | Neutral | ✅ High ON |
| Beat the same team 3× | Counted 3× in BC | **Counted 3× in ON too** |
| Beat obscure teams | Low BC | Low ON |

The key insight: ON propagates **connectivity**. A team that has beaten many opponents
who have themselves beaten many opponents scores highly, regardless of prize money.
""")
            st.markdown("#### PageRank iterations (convergence demo)")
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
            st.plotly_chart(fig_pg, use_container_width=True)
            st.caption("Values converge within 5–6 iterations. We use 6 in the engine.")

    # ══════════════════════════════════════════════════════════════
    with tab_lan:
        st.subheader("🖥️ Factor 4 — LAN Wins")
        st.markdown('<span class="tag-v">✓ VERIFIED — 1.0 × age_weight, top-10, /10, no curve, no event stakes</span>', unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown(r"""
LAN Wins is the simplest factor: a time-weighted count of wins at **LAN (offline) events**.

**Step-by-step:**

**Step 1.** For each win at a LAN event, compute:
$$\text{entry}_i = 1.0 \;\times\; \text{age\_weight}_i$$

**Step 2.** Take the **top 10** entries, sum, divide by 10:
$$\text{LAN} = \frac{\sum_{i \in \text{top-10}} 1.0 \times \text{age\_weight}_i}{10}$$

**Key confirmed behaviours:**
- **No event stakes** — a LAN win at a small $30k event counts equally to a LAN win at a
  $1M event for this factor (unlike BC/ON which do use event stakes)
- **No curve** — the output is a direct time-weighted count, not curve-adjusted
- Matches with **no prize pool** (Event Weight = "−") still count for LAN Wins if played on LAN

**LAN vs Online: where the distinction matters:**

LAN affects **only this factor**. There is **no LAN multiplier in Phase 2 (H2H)**.
This was a v2 engine error that is now corrected.

**Verification — Vitality (10 February LAN wins, all age=1.000):**
$$\text{LAN} = \frac{10 \times 1.000}{10} = \mathbf{1.000} \checkmark$$

The maximum LAN factor is 1.000 (10 or more recent LAN wins all at age=1.000).
""")
        with col_r:
            st.markdown("#### Effect of age on LAN score")
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
            st.plotly_chart(fig_lan, use_container_width=True)
            st.caption("Recent LAN wins are worth much more than old ones. 10 wins 3 months ago scores the same as 5 wins last month.")

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
        st.subheader("🌱 Phase 1 — Combining Factors → Seed Score")
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
$$\text{Seed} = 400 + \frac{\text{average} - \text{avg}_{\min}}{\text{avg}_{\max} - \text{avg}_{\min}} \times 1600$$

The **worst eligible team** always gets seed = **400**.
The **best eligible team** always gets seed = **2000**.
All others are spaced proportionally between them.

**Verification — Vitality:**
$$\frac{1.000 + 0.923 + 0.460 + 1.000}{4} = 0.846$$

Since Vitality has the highest average, they are the reference maximum:
$$\text{Seed} = 400 + \frac{0.846 - 0.000}{0.846 - 0.000} \times 1600 = \mathbf{2000.0} \checkmark$$
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
    with tab_worked:
        st.subheader("🔍 Full Worked Example")
        st.markdown("Trace any team's complete score calculation step-by-step.")

        if base_standings.empty:
            st.warning("No standings data for the selected cutoff.")
            st.stop()

        ex_team = st.selectbox("Select team:", base_standings["team"].tolist(), key="wex_team")
        ex = base_standings[base_standings["team"] == ex_team].iloc[0]
        meta = TEAM_META.get(ex_team, {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🏆 Rank",       f"#{int(ex['rank'])}")
        c2.metric("🎯 Final Score", f"{ex['total_points']:,.1f}")
        c3.metric("🌱 Seed",        f"{ex['seed']:,.1f}")
        c4.metric("⚔️ H2H Δ",      f"{ex['h2h_delta']:+.1f}")
        c5.metric("📝 Record",      f"{int(ex['wins'])}W / {int(ex['losses'])}L")

        st.markdown("---")
        st.markdown("### 🌱 Phase 1: Factor Scores")

        col_l, col_r = st.columns([2, 3])
        with col_l:
            factor_rows = [
                ("🏆 Bounty Offered", "BO",
                 f"BO_sum=${ex['bo_sum']:,.2f}", f"{ex['bo_factor']:.4f}", "#f0b429"),
                ("💰 Bounty Collected", "BC",
                 f"Σ_top10/10={ex['bc_pre_curve']:.4f}", f"{ex['bc_factor']:.4f}", "#3fb950"),
                ("🕸️ Opp. Network", "ON",
                 f"Σ_top10/10={ex['on_factor']:.4f}", f"{ex['on_factor']:.4f}", "#79c0ff"),
                ("🖥️ LAN Wins", "LAN",
                 f"{int(ex['lan_wins'])} LAN wins", f"{ex['lan_factor']:.4f}", "#f85149"),
            ]
            for fname, fcode, raw_info, factor_val, color in factor_rows:
                note = " (curve applied)" if fcode in ("BO", "BC") else " (no curve)"
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                            padding:12px 14px;margin:5px 0;">
                  <div style="color:{color};font-weight:700;margin-bottom:5px">{fname}</div>
                  <div style="font-size:12px;color:#8b949e">{raw_info}{note}</div>
                  <div style="display:flex;justify-content:space-between;margin-top:6px;">
                    <span style="font-size:12px;color:#8b949e">Factor value (×0.25)</span>
                    <span style="font-weight:700;color:{color}">{factor_val}
                      → <span style="color:#c9d1d9">{float(factor_val)*0.25:.4f}</span>
                    </span>
                  </div>
                </div>""", unsafe_allow_html=True)

            combined_val = ex["seed_combined"]
            seed_val     = ex["seed"]
            st.markdown(f"""
            <div style="background:#1c2128;border:2px solid #58a6ff;border-radius:10px;
                        padding:16px;margin-top:10px;">
              <div style="font-size:13px;color:#8b949e">Average = (BO+BC+ON+LAN)/4</div>
              <div style="font-size:26px;font-weight:700;color:#c9d1d9">{combined_val:.4f}</div>
              <div style="font-size:12px;color:#8b949e;margin-top:6px">
                400 + ({combined_val:.4f} − min) / (max − min) × 1600</div>
              <div style="font-size:30px;font-weight:700;color:#58a6ff;margin-top:4px">
                Seed = {seed_val:,.1f}</div>
            </div>""", unsafe_allow_html=True)

        with col_r:
            wex_cutoff = cutoff_dt
            wex_ws     = wex_cutoff - timedelta(days=DECAY_DAYS)
            team_m = base_matches[
                ((base_matches["winner"] == ex_team) |
                 (base_matches["loser"]  == ex_team)) &
                (base_matches["date"] >= wex_ws) &
                (base_matches["date"] <= wex_cutoff)
            ].copy().sort_values("date", ascending=False)

            if not team_m.empty:
                team_m["Result"] = team_m["winner"].apply(
                    lambda w: "✅ Win" if w == ex_team else "❌ Loss")
                team_m["Age Wt"] = team_m["date"].apply(
                    lambda d: age_weight(d, wex_cutoff))
                team_m["Ev. Stk"] = team_m["prize_pool"].apply(event_stakes)
                team_m["K_eff"] = team_m.apply(
                    lambda r: BASE_K * r["Age Wt"], axis=1)
                team_m["BO contrib"] = team_m.apply(
                    lambda r: r["winner_prize"] * r["Age Wt"]
                    if r["winner"] == ex_team else 0.0, axis=1)

                disp = team_m.rename(columns={"date":"Date","event":"Event",
                                               "prize_pool":"Pool"})[
                    ["Date","Result","winner","loser","Event","Pool","is_lan",
                     "Age Wt","Ev. Stk","K_eff","BO contrib"]].copy()
                disp["Date"]       = disp["Date"].dt.strftime("%Y-%m-%d")
                disp["Pool"]       = disp["Pool"].apply(lambda x: f"${float(x):,.0f}")
                disp["Age Wt"]     = disp["Age Wt"].apply(lambda x: f"{x:.3f}")
                disp["Ev. Stk"]    = disp["Ev. Stk"].apply(lambda x: f"{x:.3f}")
                disp["K_eff"]      = disp["K_eff"].apply(lambda x: f"{x:.2f}")
                disp["BO contrib"] = disp["BO contrib"].apply(
                    lambda x: f"${float(x):,.0f}" if float(x) > 0 else "—")
                disp.columns = ["Date","Result","Winner","Loser","Event","Pool",
                                 "LAN","Age Wt","Ev.Stakes","H2H K","BO Contrib"]
                st.dataframe(disp, use_container_width=True, hide_index=True, height=310)
                st.caption(
                    "**Age Wt**: flat 1.0 for ≤30d, then linear · "
                    "**Ev.Stakes**: curve(pool/$1M), used by BC+ON · "
                    "**H2H K**: K=32×Age Wt · "
                    "**BO Contrib**: prize×age (wins only, no event stakes)"
                )
            else:
                st.info("No matches in current window.")

        st.markdown("---")
        st.markdown("### ⚔️ Phase 2 Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Seed",        f"{ex['seed']:,.1f}")
        c2.metric("H2H Δ",       f"{ex['h2h_delta']:+.1f}")
        c3.metric("Final Score", f"{ex['total_points']:,.1f}")

        fig_fin = go.Figure()
        fig_fin.add_trace(go.Bar(name="Seed", x=[ex_team], y=[ex["seed"]],
            marker_color="#79c0ff", text=[f"{ex['seed']:.0f}"], textposition="auto"))
        h2h_v = ex["h2h_delta"]
        fig_fin.add_trace(go.Bar(name="H2H Δ", x=[ex_team], y=[h2h_v],
            marker_color="#3fb950" if h2h_v >= 0 else "#f85149",
            text=[f"{h2h_v:+.1f}"], textposition="auto"))
        fig_fin.update_layout(
            barmode="relative",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            yaxis=dict(gridcolor="#21262d"),
            legend=dict(orientation="h", y=1.15),
            margin=dict(l=10,r=10,t=30,b=10), height=200)
        st.plotly_chart(fig_fin, use_container_width=True)
