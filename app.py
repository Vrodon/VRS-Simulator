"""
CS2 Valve Regional Standings (VRS) Simulator
=============================================
Replicates and simulates the official Valve Regional Standings system.
Based on: https://github.com/ValveSoftware/counter-strike_regional_standings

VRS Pillars:
  1. Bounty Offered   – Points from team's own prize earnings
  2. Bounty Collected – Points from opponents' prize money
  3. Opponent Network – Points from breadth of opponents' victories
  4. LAN / H2H        – Recency-decayed match results with LAN bonus
"""

import math
import json
import requests
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CS2 VRS Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Dark CS2-inspired theme tweaks */
[data-testid="stSidebar"] { background: #0d1117; }
.rank-badge {
    display:inline-block; width:32px; height:32px;
    border-radius:50%; background:#f0b429;
    color:#000; font-weight:700; text-align:center;
    line-height:32px; font-size:13px;
}
.rank-badge.top3  { background:#f0b429; }
.rank-badge.top10 { background:#6c757d; color:#fff; }
.rank-badge.rest  { background:#2d333b; color:#adb5bd; }

.pill-eu  { background:#1565c0; color:#fff; padding:2px 8px; border-radius:12px; font-size:12px; }
.pill-am  { background:#c62828; color:#fff; padding:2px 8px; border-radius:12px; font-size:12px; }
.pill-as  { background:#2e7d32; color:#fff; padding:2px 8px; border-radius:12px; font-size:12px; }

.metric-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:12px; padding:16px 20px; text-align:center;
}
.metric-value { font-size:28px; font-weight:700; color:#58a6ff; }
.metric-label { font-size:13px; color:#8b949e; margin-top:4px; }

.change-up   { color:#3fb950; font-weight:600; }
.change-down { color:#f85149; font-weight:600; }
.change-same { color:#8b949e; }

div[data-testid="metric-container"] {
    background:#161b22; border:1px solid #30363d;
    border-radius:12px; padding:12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# ═══════════════════  VRS CALCULATION ENGINE  ════════════════
# ─────────────────────────────────────────────────────────────

DECAY_WINDOW_DAYS = 180  # 6 months
LAN_MULTIPLIER    = 1.5
ONLINE_MULTIPLIER = 1.0

# Pillar weight constants (tuned to approximate official VRS scores ~1000-2000)
W_BOUNTY_OFFERED   = 420.0
W_BOUNTY_COLLECTED = 280.0
W_OPPONENT_NETWORK = 180.0


def recency_decay(match_date: datetime, cutoff: datetime = None) -> float:
    """Linear decay: 1.0 at cutoff → 0.0 at cutoff-180 days."""
    if cutoff is None:
        cutoff = datetime.now()
    days_ago = (cutoff - match_date).days
    if days_ago < 0:
        return 1.0  # future-dated (shouldn't happen)
    if days_ago >= DECAY_WINDOW_DAYS:
        return 0.0
    return 1.0 - (days_ago / DECAY_WINDOW_DAYS)


def prize_to_points(prize_usd: float, weight: float) -> float:
    """Log-scale conversion of prize money to VRS points."""
    if prize_usd <= 0:
        return 0.0
    return weight * math.log10(1.0 + prize_usd / 1_000.0)


def compute_vrs(matches_df: pd.DataFrame, cutoff: datetime = None) -> pd.DataFrame:
    """
    Full VRS computation from a match-history DataFrame.

    Expected columns:
        date        – datetime
        winner      – str (team name)
        loser       – str (team name)
        prize_pool  – float (USD, total event prize pool)
        winner_prize– float (USD won by the winner)
        loser_prize – float (USD won by the loser)
        is_lan      – bool

    Returns a DataFrame with one row per team and VRS breakdown columns.
    """
    if cutoff is None:
        cutoff = datetime.now()

    # Only consider matches within the decay window
    window_start = cutoff - timedelta(days=DECAY_WINDOW_DAYS)
    df = matches_df[matches_df["date"] >= window_start].copy()
    df["decay"]  = df["date"].apply(lambda d: recency_decay(d, cutoff))
    df["lan_mult"] = df["is_lan"].apply(lambda x: LAN_MULTIPLIER if x else ONLINE_MULTIPLIER)
    df["weight"] = df["decay"] * df["lan_mult"]

    teams = sorted(set(df["winner"].tolist() + df["loser"].tolist()))
    records = []

    for team in teams:
        wins_df  = df[df["winner"] == team]
        losses_df = df[df["loser"] == team]

        # ── Pillar 1: Bounty Offered ─────────────────────────────────
        # Prize money the team itself earned (weighted)
        bo_score = 0.0
        for _, row in wins_df.iterrows():
            bo_score += prize_to_points(row["winner_prize"], W_BOUNTY_OFFERED) * row["weight"]
        for _, row in losses_df.iterrows():
            bo_score += prize_to_points(row["loser_prize"], W_BOUNTY_OFFERED) * row["weight"] * 0.35

        # ── Pillar 2: Bounty Collected ───────────────────────────────
        # Prize money earned by each defeated opponent
        bc_score = 0.0
        for _, row in wins_df.iterrows():
            opp = row["loser"]
            opp_wins = df[df["winner"] == opp]
            opp_prize = opp_wins["winner_prize"].sum()
            bc_score += prize_to_points(opp_prize, W_BOUNTY_COLLECTED) * row["weight"]

        # ── Pillar 3: Opponent Network ───────────────────────────────
        # How many teams each opponent has beaten
        on_score = 0.0
        for _, row in wins_df.iterrows():
            opp = row["loser"]
            opp_wins_count = len(df[df["winner"] == opp])
            on_score += W_OPPONENT_NETWORK * math.log10(1.0 + opp_wins_count) * row["weight"]

        total_score = bo_score + bc_score + on_score
        wins_total  = len(wins_df)
        losses_total = len(losses_df)

        records.append({
            "team":             team,
            "total_points":     round(total_score, 1),
            "bounty_offered":   round(bo_score, 1),
            "bounty_collected": round(bc_score, 1),
            "opponent_network": round(on_score, 1),
            "wins":             wins_total,
            "losses":           losses_total,
        })

    result = pd.DataFrame(records)
    result = result.sort_values("total_points", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


# ─────────────────────────────────────────────────────────────
# ═══════════════════  SEED DATA  ═════════════════════════════
# ─────────────────────────────────────────────────────────────

TEAM_META = {
    # Europe
    "Vitality":     {"region": "Europe",   "flag": "🇫🇷", "color": "#F5A623"},
    "PARIVISION":   {"region": "Europe",   "flag": "🇷🇺", "color": "#E91E63"},
    "NAVI":         {"region": "Europe",   "flag": "🇺🇦", "color": "#FFD600"},
    "Spirit":       {"region": "Europe",   "flag": "🇷🇺", "color": "#7B68EE"},
    "MOUZ":         {"region": "Europe",   "flag": "🇩🇪", "color": "#E53935"},
    "FaZe":         {"region": "Europe",   "flag": "🌍",   "color": "#EC407A"},
    "Aurora":       {"region": "Europe",   "flag": "🇷🇺", "color": "#00BCD4"},
    "G2":           {"region": "Europe",   "flag": "🇪🇸", "color": "#F44336"},
    "3DMAX":        {"region": "Europe",   "flag": "🇫🇷", "color": "#607D8B"},
    "Astralis":     {"region": "Europe",   "flag": "🇩🇰", "color": "#1565C0"},
    "NIP":          {"region": "Europe",   "flag": "🇸🇪", "color": "#1A237E"},
    "B8":           {"region": "Europe",   "flag": "🇺🇦", "color": "#4CAF50"},
    "GamerLegion":  {"region": "Europe",   "flag": "🇩🇰", "color": "#9C27B0"},
    "Falcons":      {"region": "Europe",   "flag": "🇸🇦", "color": "#0288D1"},
    "FUT":          {"region": "Europe",   "flag": "🇹🇷", "color": "#FF9800"},
    # Americas
    "FURIA":        {"region": "Americas", "flag": "🇧🇷", "color": "#F5A623"},
    "Liquid":       {"region": "Americas", "flag": "🇺🇸", "color": "#00B0FF"},
    "NRG":          {"region": "Americas", "flag": "🇺🇸", "color": "#FF5722"},
    "M80":          {"region": "Americas", "flag": "🇺🇸", "color": "#78909C"},
    "Imperial":     {"region": "Americas", "flag": "🇧🇷", "color": "#7B1FA2"},
    "paiN":         {"region": "Americas", "flag": "🇧🇷", "color": "#D32F2F"},
    "9z":           {"region": "Americas", "flag": "🇦🇷", "color": "#1976D2"},
    "Cloud9":       {"region": "Americas", "flag": "🇺🇸", "color": "#29B6F6"},
    # Asia
    "MongolZ":      {"region": "Asia",     "flag": "🇲🇳", "color": "#FF6F00"},
    "TYLOO":        {"region": "Asia",     "flag": "🇨🇳", "color": "#C62828"},
    "Rare Atom":    {"region": "Asia",     "flag": "🇨🇳", "color": "#00897B"},
    "ThunderPick":  {"region": "Asia",     "flag": "🇦🇺", "color": "#5E35B1"},
    "XTREME":       {"region": "Asia",     "flag": "🇯🇵", "color": "#F57F17"},
    "Grayhound":    {"region": "Asia",     "flag": "🇦🇺", "color": "#546E7A"},
}

# Curated events (prize pool in USD, is_lan)
EVENTS = [
    {"name": "IEM Krakow 2026",          "prize_pool": 1_000_000, "is_lan": True,  "date": "2026-01-15"},
    {"name": "PGL Cluj-Napoca 2026",     "prize_pool": 1_000_000, "is_lan": True,  "date": "2026-02-10"},
    {"name": "ESL Pro League S21",       "prize_pool":   800_000, "is_lan": True,  "date": "2026-01-25"},
    {"name": "BLAST Bounty S1 Finals",   "prize_pool":   500_000, "is_lan": True,  "date": "2025-12-20"},
    {"name": "CCT Season 3 EU Playoffs", "prize_pool":   200_000, "is_lan": False, "date": "2026-02-28"},
    {"name": "IEM Dallas 2025",          "prize_pool": 1_000_000, "is_lan": True,  "date": "2025-10-10"},
    {"name": "BLAST Premier Fall 2025",  "prize_pool": 1_000_000, "is_lan": True,  "date": "2025-11-05"},
    {"name": "ESL Challenger League S47","prize_pool":   100_000, "is_lan": False, "date": "2026-01-10"},
    {"name": "BLAST Open Rotterdam",     "prize_pool":   250_000, "is_lan": True,  "date": "2026-03-15"},
    {"name": "Fragadelphia Miami",       "prize_pool":    50_000, "is_lan": True,  "date": "2026-01-05"},
    {"name": "fl0m Mythic LAN",          "prize_pool":    30_000, "is_lan": True,  "date": "2025-12-05"},
    {"name": "PGL Astana 2025",          "prize_pool": 1_000_000, "is_lan": True,  "date": "2025-09-15"},
]

def _build_placement_prize(prize_pool: float, place: int) -> float:
    """Distribute prize money based on placement (approximate typical split)."""
    splits = {1: 0.40, 2: 0.22, 3: 0.12, 4: 0.08, 5: 0.05, 6: 0.03, 7: 0.02, 8: 0.01}
    return prize_pool * splits.get(place, 0.005)


def generate_seed_matches() -> pd.DataFrame:
    """
    Generate a realistic match history that produces VRS-like standings
    matching roughly the March 2026 real-world snapshot.
    """
    rows = []

    # (winner, loser, event_name, place_winner, place_loser)
    match_blueprints = [
        # IEM Krakow 2026 (Jan 2026) — Vitality wins
        ("Vitality",  "PARIVISION", "IEM Krakow 2026",         1, 2),
        ("Vitality",  "NAVI",       "IEM Krakow 2026",         1, 4),
        ("PARIVISION","NAVI",       "IEM Krakow 2026",         2, 4),
        ("PARIVISION","Spirit",     "IEM Krakow 2026",         2, 3),
        ("Spirit",    "MOUZ",       "IEM Krakow 2026",         3, 5),
        ("NAVI",      "FaZe",       "IEM Krakow 2026",         4, 6),
        ("MOUZ",      "G2",         "IEM Krakow 2026",         5, 7),
        ("FaZe",      "Aurora",     "IEM Krakow 2026",         6, 8),
        ("FURIA",     "Liquid",     "IEM Krakow 2026",         7, 8),
        # PGL Cluj-Napoca 2026 (Feb 2026) — Vitality wins again
        ("Vitality",  "PARIVISION", "PGL Cluj-Napoca 2026",    1, 2),
        ("Vitality",  "Falcons",    "PGL Cluj-Napoca 2026",    1, 4),
        ("PARIVISION","Falcons",    "PGL Cluj-Napoca 2026",    2, 4),
        ("PARIVISION","NAVI",       "PGL Cluj-Napoca 2026",    2, 3),
        ("NAVI",      "Spirit",     "PGL Cluj-Napoca 2026",    3, 5),
        ("Falcons",   "Aurora",     "PGL Cluj-Napoca 2026",    4, 6),
        ("Spirit",    "FaZe",       "PGL Cluj-Napoca 2026",    5, 7),
        ("Aurora",    "B8",         "PGL Cluj-Napoca 2026",    6, 8),
        ("FURIA",     "Liquid",     "PGL Cluj-Napoca 2026",    7, 8),
        # ESL Pro League S21 (Jan 2026)
        ("NAVI",      "G2",         "ESL Pro League S21",      1, 2),
        ("G2",        "MOUZ",       "ESL Pro League S21",      2, 3),
        ("MOUZ",      "Astralis",   "ESL Pro League S21",      3, 4),
        ("Astralis",  "3DMAX",      "ESL Pro League S21",      4, 5),
        ("FURIA",     "NRG",        "ESL Pro League S21",      1, 2),
        ("Liquid",    "M80",        "ESL Pro League S21",      3, 4),
        ("MongolZ",   "TYLOO",      "ESL Pro League S21",      1, 2),
        ("TYLOO",     "Rare Atom",  "ESL Pro League S21",      2, 3),
        # BLAST Bounty S1 Finals (Dec 2025)
        ("PARIVISION","Vitality",   "BLAST Bounty S1 Finals",  1, 2),
        ("Vitality",  "Falcons",    "BLAST Bounty S1 Finals",  2, 3),
        ("Falcons",   "NAVI",       "BLAST Bounty S1 Finals",  3, 4),
        ("NAVI",      "FaZe",       "BLAST Bounty S1 Finals",  4, 5),
        ("FURIA",     "Liquid",     "BLAST Bounty S1 Finals",  5, 6),
        # BLAST Premier Fall 2025 (Nov 2025)
        ("Spirit",    "MOUZ",       "BLAST Premier Fall 2025", 1, 2),
        ("MOUZ",      "FaZe",       "BLAST Premier Fall 2025", 2, 3),
        ("FaZe",      "G2",         "BLAST Premier Fall 2025", 3, 4),
        ("NAVI",      "Vitality",   "BLAST Premier Fall 2025", 4, 5),
        ("FURIA",     "Liquid",     "BLAST Premier Fall 2025", 1, 2),
        ("MongolZ",   "TYLOO",      "BLAST Premier Fall 2025", 1, 2),
        # IEM Dallas 2025 (Oct 2025)
        ("Spirit",    "NAVI",       "IEM Dallas 2025",         1, 2),
        ("NAVI",      "Vitality",   "IEM Dallas 2025",         2, 3),
        ("Vitality",  "FaZe",       "IEM Dallas 2025",         3, 4),
        ("FaZe",      "MOUZ",       "IEM Dallas 2025",         4, 5),
        ("FURIA",     "Liquid",     "IEM Dallas 2025",         1, 2),
        ("MongolZ",   "Rare Atom",  "IEM Dallas 2025",         1, 2),
        # PGL Astana 2025 (Sep 2025) — near decay boundary
        ("MOUZ",      "Spirit",     "PGL Astana 2025",         1, 2),
        ("Spirit",    "NAVI",       "PGL Astana 2025",         2, 3),
        ("NAVI",      "G2",         "PGL Astana 2025",         3, 4),
        ("FURIA",     "NRG",        "PGL Astana 2025",         1, 2),
        ("MongolZ",   "TYLOO",      "PGL Astana 2025",         1, 2),
        # BLAST Open Rotterdam (Mar 2026, live)
        ("Vitality",  "PARIVISION", "BLAST Open Rotterdam",    1, 4),
        ("NAVI",      "Aurora",     "BLAST Open Rotterdam",    2, 5),
        ("Falcons",   "FURIA",      "BLAST Open Rotterdam",    3, 6),
        ("MongolZ",   "Spirit",     "BLAST Open Rotterdam",    1, 7),
        ("NRG",       "B8",         "BLAST Open Rotterdam",    4, 8),
        # CCT / lower-tier
        ("B8",        "GamerLegion","CCT Season 3 EU Playoffs",1, 2),
        ("GamerLegion","3DMAX",     "CCT Season 3 EU Playoffs",2, 3),
        ("3DMAX",     "FUT",        "CCT Season 3 EU Playoffs",3, 4),
        ("FUT",       "NIP",        "CCT Season 3 EU Playoffs",4, 5),
        # NA lower tier
        ("NRG",       "M80",        "Fragadelphia Miami",      1, 2),
        ("M80",       "Cloud9",     "Fragadelphia Miami",      2, 3),
        ("Cloud9",    "9z",         "Fragadelphia Miami",      3, 4),
        # Asia lower tier
        ("Grayhound", "ThunderPick","ESL Challenger League S47",1, 2),
        ("TYLOO",     "XTREME",     "ESL Challenger League S47",1, 2),
    ]

    # Build event lookup
    event_lookup = {e["name"]: e for e in EVENTS}

    for (winner, loser, event_name, winner_place, loser_place) in match_blueprints:
        ev = event_lookup.get(event_name)
        if ev is None:
            continue
        prize_pool = ev["prize_pool"]
        is_lan     = ev["is_lan"]
        date       = datetime.strptime(ev["date"], "%Y-%m-%d")
        # Add slight jitter within the event window so round progression feels real
        date += timedelta(days=winner_place)

        rows.append({
            "date":         date,
            "winner":       winner,
            "loser":        loser,
            "event":        event_name,
            "prize_pool":   prize_pool,
            "winner_prize": _build_placement_prize(prize_pool, winner_place),
            "loser_prize":  _build_placement_prize(prize_pool, loser_place),
            "is_lan":       is_lan,
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def fetch_live_standings(region: str) -> pd.DataFrame | None:
    """
    Attempt to fetch current standings from the Valve GitHub repo (Markdown file).
    Falls back to None if unavailable.
    """
    region_map = {
        "Europe":   "standings_europe.md",
        "Americas": "standings_americas.md",
        "Asia":     "standings_asia.md",
    }
    filename = region_map.get(region)
    if not filename:
        return None

    url = (
        f"https://raw.githubusercontent.com/ValveSoftware/"
        f"counter-strike_regional_standings/main/{filename}"
    )
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            lines = resp.text.strip().splitlines()
            # Parse the Markdown table
            data_lines = [l for l in lines if l.startswith("|") and "---" not in l]
            if len(data_lines) < 2:
                return None
            headers = [h.strip() for h in data_lines[0].split("|") if h.strip()]
            records = []
            for line in data_lines[1:]:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) == len(headers):
                    records.append(dict(zip(headers, cells)))
            return pd.DataFrame(records) if records else None
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# ═══════════════════  HELPERS / FORMATTERS  ══════════════════
# ─────────────────────────────────────────────────────────────

def region_pill(region: str) -> str:
    cls = {"Europe": "pill-eu", "Americas": "pill-am", "Asia": "pill-as"}.get(region, "pill-eu")
    return f'<span class="{cls}">{region}</span>'


def rank_badge(rank: int) -> str:
    if rank <= 3:
        cls = "top3"
    elif rank <= 10:
        cls = "top10"
    else:
        cls = "rest"
    return f'<span class="rank-badge {cls}">{rank}</span>'


def change_arrow(delta: int) -> str:
    if delta > 0:
        return f'<span class="change-up">▲ {delta}</span>'
    elif delta < 0:
        return f'<span class="change-down">▼ {abs(delta)}</span>'
    else:
        return '<span class="change-same">—</span>'


def add_region_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["region"] = df["team"].map(lambda t: TEAM_META.get(t, {}).get("region", "Europe"))
    df["flag"]   = df["team"].map(lambda t: TEAM_META.get(t, {}).get("flag", "🌍"))
    return df


def regional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add a regional_rank column."""
    df = df.copy()
    df["regional_rank"] = df.groupby("region")["total_points"].rank(
        ascending=False, method="first"
    ).astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# ════════════════════  SIDEBAR  ══════════════════════════════
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 CS2 VRS Simulator")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Ranking Dashboard", "🔮 Scenario Simulator", "⚔️ Team Comparison"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
**About VRS**
The Valve Regional Standings use four pillars:
1. **Bounty Offered** – own prize money
2. **Bounty Collected** – opponents' prize money
3. **Opponent Network** – breadth of opponents' wins
4. **LAN Bonus** – ×1.5 multiplier for LAN matches

All points decay linearly to **zero** at **6 months**.

📄 [Valve GitHub Repo](https://github.com/ValveSoftware/counter-strike_regional_standings)
"""
    )
    st.markdown("---")
    cutoff_date = st.date_input(
        "📅 Standings Cutoff Date",
        value=datetime(2026, 3, 2),
        help="Change the date to see how standings looked at that point in time.",
    )
    cutoff_dt = datetime.combine(cutoff_date, datetime.min.time())


# ─────────────────────────────────────────────────────────────
# ════════════════  COMPUTE BASE STANDINGS  ════════════════════
# ─────────────────────────────────────────────────────────────

@st.cache_data
def get_base_matches():
    return generate_seed_matches()


base_matches = get_base_matches()


def compute_standings(extra_matches: pd.DataFrame = None, cutoff: datetime = None):
    if cutoff is None:
        cutoff = cutoff_dt
    matches = base_matches.copy()
    if extra_matches is not None and not extra_matches.empty:
        matches = pd.concat([matches, extra_matches], ignore_index=True)
    result = compute_vrs(matches, cutoff=cutoff)
    result = add_region_meta(result)
    result = regional_ranks(result)
    return result


base_standings = compute_standings()


# ─────────────────────────────────────────────────────────────
# ════════════════  PAGE 1: RANKING DASHBOARD  ════════════════
# ─────────────────────────────────────────────────────────────

if page == "📊 Ranking Dashboard":
    st.title("📊 CS2 Valve Regional Standings")
    st.caption(
        f"Simulated standings as of **{cutoff_date.strftime('%B %d, %Y')}** · "
        f"6-month decay window · Based on [Valve's VRS model]"
        f"(https://github.com/ValveSoftware/counter-strike_regional_standings)"
    )

    # ── Attempt live data fetch ──────────────────────────────
    with st.expander("🔴 Live Valve Data (GitHub)", expanded=False):
        live_eu = fetch_live_standings("Europe")
        if live_eu is not None:
            st.success("✅ Fetched live Europe standings from Valve's GitHub!")
            st.dataframe(live_eu, use_container_width=True)
        else:
            st.info(
                "Could not fetch live data from Valve's GitHub (rate limit or network). "
                "The simulator below uses the built-in seeded dataset."
            )

    # ── Top-level KPIs ───────────────────────────────────────
    top_team  = base_standings.iloc[0]
    eu_top    = base_standings[base_standings["region"] == "Europe"].iloc[0]
    am_top    = base_standings[base_standings["region"] == "Americas"].iloc[0]
    as_top    = base_standings[base_standings["region"] == "Asia"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌍 Global #1", top_team["team"], f'{top_team["total_points"]:,.0f} pts')
    with col2:
        st.metric("🇪🇺 Europe #1", eu_top["team"], f'{eu_top["total_points"]:,.0f} pts')
    with col3:
        st.metric("🌎 Americas #1", am_top["team"], f'{am_top["total_points"]:,.0f} pts')
    with col4:
        st.metric("🌏 Asia #1", as_top["team"], f'{as_top["total_points"]:,.0f} pts')

    st.markdown("---")

    # ── Region filter tabs ───────────────────────────────────
    tab_global, tab_eu, tab_am, tab_as = st.tabs(
        ["🌍 Global", "🇪🇺 Europe", "🌎 Americas", "🌏 Asia"]
    )

    def render_standings_table(df: pd.DataFrame, show_all: bool = False):
        display = df.copy()
        if not show_all:
            display = display.head(25)

        rows_html = []
        for _, row in display.iterrows():
            rk   = int(row["rank"])
            flag = row.get("flag", "🌍")
            team = row["team"]
            pts  = f'{row["total_points"]:,.1f}'
            bo   = f'{row["bounty_offered"]:,.1f}'
            bc   = f'{row["bounty_collected"]:,.1f}'
            on   = f'{row["opponent_network"]:,.1f}'
            reg  = region_pill(row["region"])
            rrk  = int(row["regional_rank"])

            row_html = f"""
            <tr>
                <td style="text-align:center">{rank_badge(rk)}</td>
                <td>{flag} <strong>{team}</strong></td>
                <td style="text-align:center">{reg}</td>
                <td style="text-align:right; color:#58a6ff; font-weight:700">{pts}</td>
                <td style="text-align:right; color:#f0883e">{bo}</td>
                <td style="text-align:right; color:#79c0ff">{bc}</td>
                <td style="text-align:right; color:#7ee787">{on}</td>
                <td style="text-align:center; color:#8b949e">{rrk}</td>
            </tr>"""
            rows_html.append(row_html)

        table_html = f"""
        <table style="width:100%; border-collapse:collapse; font-size:14px;">
          <thead>
            <tr style="background:#21262d; color:#8b949e; font-size:12px; text-transform:uppercase;">
              <th style="padding:10px 6px; text-align:center">Rank</th>
              <th style="padding:10px 6px; text-align:left">Team</th>
              <th style="padding:10px 6px; text-align:center">Region</th>
              <th style="padding:10px 6px; text-align:right">Total Pts</th>
              <th style="padding:10px 6px; text-align:right" title="Bounty Offered">Bounty Off.</th>
              <th style="padding:10px 6px; text-align:right" title="Bounty Collected">Bounty Col.</th>
              <th style="padding:10px 6px; text-align:right" title="Opponent Network">Opp. Net.</th>
              <th style="padding:10px 6px; text-align:center">Reg. Rank</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows_html)}
          </tbody>
        </table>"""
        st.markdown(table_html, unsafe_allow_html=True)

    with tab_global:
        show_all_global = st.checkbox("Show all teams", key="show_all_global")
        render_standings_table(base_standings, show_all=show_all_global)

    with tab_eu:
        eu_df = base_standings[base_standings["region"] == "Europe"].reset_index(drop=True)
        eu_df["rank"] = eu_df.index + 1
        render_standings_table(eu_df, show_all=True)

    with tab_am:
        am_df = base_standings[base_standings["region"] == "Americas"].reset_index(drop=True)
        am_df["rank"] = am_df.index + 1
        render_standings_table(am_df, show_all=True)

    with tab_as:
        as_df = base_standings[base_standings["region"] == "Asia"].reset_index(drop=True)
        as_df["rank"] = as_df.index + 1
        render_standings_table(as_df, show_all=True)

    st.markdown("---")

    # ── Visualisations ───────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🏆 Top 15 — Points Breakdown")
        top15 = base_standings.head(15).copy()
        fig = go.Figure()
        colors = {"bounty_offered": "#f0883e", "bounty_collected": "#79c0ff", "opponent_network": "#7ee787"}
        labels = {"bounty_offered": "Bounty Offered", "bounty_collected": "Bounty Collected", "opponent_network": "Opponent Network"}
        for col_name, color in colors.items():
            fig.add_trace(go.Bar(
                name=labels[col_name],
                x=top15["team"],
                y=top15[col_name],
                marker_color=color,
            ))
        fig.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            xaxis=dict(tickangle=-45, gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=80),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🌍 Regional Distribution")
        reg_counts = base_standings["region"].value_counts().reset_index()
        reg_counts.columns = ["Region", "Teams"]
        fig2 = px.pie(
            reg_counts, values="Teams", names="Region",
            color="Region",
            color_discrete_map={"Europe": "#1565c0", "Americas": "#c62828", "Asia": "#2e7d32"},
            hole=0.45,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Decay explainer ─────────────────────────────────────
    with st.expander("📉 Recency Decay Visualisation"):
        days = list(range(0, 181, 10))
        decay_vals = [round(1.0 - d / 180, 3) for d in days]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=days, y=decay_vals, mode="lines+markers",
            line=dict(color="#58a6ff", width=3),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
            name="Decay factor",
        ))
        fig3.update_layout(
            title="VRS Recency Decay (linear, 6-month window)",
            xaxis_title="Days before cutoff",
            yaxis_title="Point multiplier",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", tickformat=".0%"),
            height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "A match played today contributes 100% of its points. "
            "A match from 3 months ago contributes 50%. "
            "A match from 6+ months ago contributes 0%."
        )


# ─────────────────────────────────────────────────────────────
# ════════════════  PAGE 2: SCENARIO SIMULATOR  ═══════════════
# ─────────────────────────────────────────────────────────────

elif page == "🔮 Scenario Simulator":
    st.title("🔮 What-If Scenario Simulator")
    st.markdown(
        "Add hypothetical match results and see how the VRS standings would change. "
        "You can queue up multiple matches and simulate them all at once."
    )

    all_teams = sorted(TEAM_META.keys())

    # Session-state for queued hypothetical matches
    if "hyp_matches" not in st.session_state:
        st.session_state.hyp_matches = []

    # ── Match input form ─────────────────────────────────────
    with st.form("add_match_form"):
        st.subheader("➕ Add a Hypothetical Match")
        c1, c2, c3 = st.columns(3)
        with c1:
            winner = st.selectbox("🏆 Winner", all_teams, index=all_teams.index("G2"))
        with c2:
            losers_list = [t for t in all_teams if t != winner]
            loser = st.selectbox("❌ Loser", losers_list, index=0)
        with c3:
            event_name_input = st.text_input("🏟️ Event Name", value="Hypothetical Event")

        c4, c5, c6 = st.columns(3)
        with c4:
            prize_pool = st.number_input(
                "💵 Prize Pool (USD)", min_value=10_000, max_value=2_000_000,
                value=250_000, step=10_000, format="%d"
            )
        with c5:
            winner_prize_pct = st.slider(
                "🥇 Winner Prize %", min_value=10, max_value=60, value=40
            )
        with c6:
            loser_prize_pct = st.slider(
                "🥈 Loser Prize %", min_value=0, max_value=30, value=22
            )

        c7, c8 = st.columns(2)
        with c7:
            match_date = st.date_input("📅 Match Date", value=datetime(2026, 3, 20))
        with c8:
            is_lan = st.toggle("🖥️ LAN Event?", value=True)

        submitted = st.form_submit_button("➕ Add to Queue", use_container_width=True, type="primary")
        if submitted:
            st.session_state.hyp_matches.append({
                "date":         datetime.combine(match_date, datetime.min.time()),
                "winner":       winner,
                "loser":        loser,
                "event":        event_name_input,
                "prize_pool":   float(prize_pool),
                "winner_prize": float(prize_pool) * winner_prize_pct / 100,
                "loser_prize":  float(prize_pool) * loser_prize_pct / 100,
                "is_lan":       is_lan,
            })
            st.success(f"✅ Added: **{winner}** def. **{loser}** @ {event_name_input}")

    # ── Match Queue ──────────────────────────────────────────
    if st.session_state.hyp_matches:
        st.markdown("#### 📋 Match Queue")
        queue_df = pd.DataFrame(st.session_state.hyp_matches)[
            ["winner", "loser", "event", "prize_pool", "is_lan", "date"]
        ]
        queue_df.columns = ["Winner", "Loser", "Event", "Prize Pool", "LAN", "Date"]
        queue_df["Prize Pool"] = queue_df["Prize Pool"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(queue_df, use_container_width=True, hide_index=True)

        col_sim, col_clr = st.columns([3, 1])
        with col_clr:
            if st.button("🗑️ Clear Queue", use_container_width=True):
                st.session_state.hyp_matches = []
                st.rerun()
        with col_sim:
            simulate_btn = st.button(
                "🚀 Simulate & Show Impact",
                use_container_width=True,
                type="primary",
            )

        if simulate_btn:
            with st.spinner("Recalculating VRS standings…"):
                hyp_df = pd.DataFrame(st.session_state.hyp_matches)
                new_standings = compute_standings(extra_matches=hyp_df, cutoff=cutoff_dt)

            st.markdown("---")
            st.subheader("📊 Predicted New Standings vs Current")

            # Merge old and new
            merged = base_standings[["team", "rank", "total_points"]].merge(
                new_standings[["team", "rank", "total_points", "region", "flag"]],
                on="team", suffixes=("_old", "_new"),
            )
            merged["rank_delta"]  = merged["rank_old"]  - merged["rank_new"]   # positive = moved up
            merged["points_delta"] = merged["total_points_new"] - merged["total_points_old"]
            merged = merged.sort_values("rank_new")

            # Highlight changed teams
            affected = set(hyp_df["winner"].tolist() + hyp_df["loser"].tolist())

            rows_html = []
            for _, row in merged.iterrows():
                is_affected = row["team"] in affected
                highlight   = "background:#1c2128;" if is_affected else ""
                rk_new = int(row["rank_new"])
                rk_old = int(row["rank_old"])
                delta_pts = row["points_delta"]
                delta_rk  = int(row["rank_delta"])

                arrow = change_arrow(delta_rk)
                pts_change = (
                    f'<span class="change-up">+{delta_pts:.1f}</span>'
                    if delta_pts > 0
                    else (
                        f'<span class="change-down">{delta_pts:.1f}</span>'
                        if delta_pts < 0
                        else '<span class="change-same">—</span>'
                    )
                )
                star = "⭐ " if is_affected else ""
                rows_html.append(f"""
                <tr style="{highlight}">
                  <td style="text-align:center">{rank_badge(rk_new)}</td>
                  <td>{row['flag']} {star}<strong>{row['team']}</strong></td>
                  <td style="text-align:right; color:#58a6ff; font-weight:700">
                      {row['total_points_new']:,.1f}
                  </td>
                  <td style="text-align:center">{arrow}</td>
                  <td style="text-align:center; color:#8b949e">{rk_old}</td>
                  <td style="text-align:right">{pts_change}</td>
                </tr>""")

            table_html = f"""
            <table style="width:100%; border-collapse:collapse; font-size:14px;">
              <thead>
                <tr style="background:#21262d; color:#8b949e; font-size:12px; text-transform:uppercase;">
                  <th style="padding:10px 6px; text-align:center">New Rank</th>
                  <th style="padding:10px 6px; text-align:left">Team</th>
                  <th style="padding:10px 6px; text-align:right">New Points</th>
                  <th style="padding:10px 6px; text-align:center">Rank Change</th>
                  <th style="padding:10px 6px; text-align:center">Old Rank</th>
                  <th style="padding:10px 6px; text-align:right">Points Δ</th>
                </tr>
              </thead>
              <tbody>{"".join(rows_html)}</tbody>
            </table>
            <p style="color:#8b949e; font-size:12px; margin-top:8px">⭐ = directly involved in simulated match(es)</p>"""
            st.markdown(table_html, unsafe_allow_html=True)

            # ── Points delta bar chart ───────────────────────
            st.markdown("---")
            st.subheader("📈 Points Change by Team")
            plot_df = merged[merged["points_delta"].abs() > 0.1].sort_values(
                "points_delta", ascending=True
            )
            fig = go.Figure(go.Bar(
                x=plot_df["points_delta"],
                y=plot_df["team"],
                orientation="h",
                marker_color=[
                    "#3fb950" if v > 0 else "#f85149"
                    for v in plot_df["points_delta"]
                ],
                text=[f"{v:+.1f}" for v in plot_df["points_delta"]],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
                xaxis=dict(gridcolor="#21262d", title="Points Change"),
                yaxis=dict(gridcolor="#21262d"),
                margin=dict(l=10, r=80, t=10, b=10),
                height=max(300, len(plot_df) * 28),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Add at least one match to the queue, then click **Simulate**.")


# ─────────────────────────────────────────────────────────────
# ════════════════  PAGE 3: TEAM COMPARISON  ══════════════════
# ─────────────────────────────────────────────────────────────

elif page == "⚔️ Team Comparison":
    st.title("⚔️ Team Comparison")
    st.markdown("Select any two teams to see a side-by-side breakdown of their VRS score.")

    all_teams_sorted = base_standings["team"].tolist()

    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.selectbox(
            "Team A",
            all_teams_sorted,
            index=all_teams_sorted.index("Vitality") if "Vitality" in all_teams_sorted else 0,
        )
    with col_b:
        team_b = st.selectbox(
            "Team B",
            all_teams_sorted,
            index=all_teams_sorted.index("NAVI") if "NAVI" in all_teams_sorted else 1,
        )

    def get_team_row(name: str) -> dict:
        row = base_standings[base_standings["team"] == name]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    a = get_team_row(team_a)
    b = get_team_row(team_b)

    if not a or not b:
        st.warning("Could not find one or both teams in the standings.")
        st.stop()

    st.markdown("---")

    # ── Score cards ──────────────────────────────────────────
    col1, col_mid, col2 = st.columns([5, 1, 5])
    with col1:
        meta_a = TEAM_META.get(team_a, {})
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background:#161b22;
                        border-radius:12px; border:2px solid #58a6ff;">
                <div style="font-size:40px">{meta_a.get('flag','🌍')}</div>
                <div style="font-size:24px; font-weight:700; color:#c9d1d9; margin:8px 0">{team_a}</div>
                <div style="font-size:13px; color:#8b949e">{meta_a.get('region','')}</div>
                <div style="font-size:36px; font-weight:700; color:#58a6ff; margin-top:12px">
                    {a['total_points']:,.1f}
                </div>
                <div style="color:#8b949e; font-size:13px">VRS Points</div>
                <div style="font-size:20px; font-weight:600; color:#f0b429; margin-top:8px">
                    #{int(a['rank'])} Global &nbsp;|&nbsp; #{int(a['regional_rank'])} {meta_a.get('region','')}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_mid:
        st.markdown(
            "<div style='text-align:center; font-size:28px; font-weight:700;"
            " color:#8b949e; padding-top:60px'>VS</div>",
            unsafe_allow_html=True,
        )
    with col2:
        meta_b = TEAM_META.get(team_b, {})
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background:#161b22;
                        border-radius:12px; border:2px solid #f85149;">
                <div style="font-size:40px">{meta_b.get('flag','🌍')}</div>
                <div style="font-size:24px; font-weight:700; color:#c9d1d9; margin:8px 0">{team_b}</div>
                <div style="font-size:13px; color:#8b949e">{meta_b.get('region','')}</div>
                <div style="font-size:36px; font-weight:700; color:#f85149; margin-top:12px">
                    {b['total_points']:,.1f}
                </div>
                <div style="color:#8b949e; font-size:13px">VRS Points</div>
                <div style="font-size:20px; font-weight:600; color:#f0b429; margin-top:8px">
                    #{int(b['rank'])} Global &nbsp;|&nbsp; #{int(b['regional_rank'])} {meta_b.get('region','')}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Radar / Spider chart ─────────────────────────────────
    pillars = ["Bounty Offered", "Bounty Collected", "Opponent Network"]
    vals_a  = [a["bounty_offered"], a["bounty_collected"], a["opponent_network"]]
    vals_b  = [b["bounty_offered"], b["bounty_collected"], b["opponent_network"]]
    # Close the loop
    pillars_r = pillars + [pillars[0]]
    vals_a_r  = vals_a + [vals_a[0]]
    vals_b_r  = vals_b + [vals_b[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_a_r, theta=pillars_r, fill="toself",
        name=team_a, line_color="#58a6ff", fillcolor="rgba(88,166,255,0.15)",
        mode="lines+markers",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_b_r, theta=pillars_r, fill="toself",
        name=team_b, line_color="#f85149", fillcolor="rgba(248,81,73,0.15)",
        mode="lines+markers",
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#30363d", color="#8b949e"),
            angularaxis=dict(gridcolor="#30363d", color="#8b949e"),
            bgcolor="#161b22",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=420,
    )

    col_radar, col_bars = st.columns(2)
    with col_radar:
        st.subheader("🕸️ VRS Pillar Radar")
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bars:
        st.subheader("📊 Side-by-Side Breakdown")
        fig_bar = go.Figure()
        pillar_names = ["Bounty Offered", "Bounty Collected", "Opponent Network"]
        a_vals = [a["bounty_offered"], a["bounty_collected"], a["opponent_network"]]
        b_vals = [b["bounty_offered"], b["bounty_collected"], b["opponent_network"]]

        fig_bar.add_trace(go.Bar(
            name=team_a, x=pillar_names, y=a_vals,
            marker_color="#58a6ff",
            text=[f"{v:.0f}" for v in a_vals], textposition="outside",
        ))
        fig_bar.add_trace(go.Bar(
            name=team_b, x=pillar_names, y=b_vals,
            marker_color="#f85149",
            text=[f"{v:.0f}" for v in b_vals], textposition="outside",
        ))
        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
            height=420,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Head-to-Head records ─────────────────────────────────
    st.markdown("---")
    st.subheader("🤝 Head-to-Head Record (in dataset)")

    h2h = base_matches[
        ((base_matches["winner"] == team_a) & (base_matches["loser"] == team_b)) |
        ((base_matches["winner"] == team_b) & (base_matches["loser"] == team_a))
    ].sort_values("date", ascending=False)

    if h2h.empty:
        st.info(f"No head-to-head matches found between **{team_a}** and **{team_b}** in the dataset.")
    else:
        a_wins = len(h2h[h2h["winner"] == team_a])
        b_wins = len(h2h[h2h["winner"] == team_b])
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            st.metric(f"{team_a} Wins", a_wins)
        with hc2:
            st.metric("Total Matches", len(h2h))
        with hc3:
            st.metric(f"{team_b} Wins", b_wins)

        h2h_display = h2h[["date", "winner", "loser", "event", "prize_pool", "is_lan"]].copy()
        h2h_display["date"] = h2h_display["date"].dt.strftime("%Y-%m-%d")
        h2h_display["prize_pool"] = h2h_display["prize_pool"].apply(lambda x: f"${x:,.0f}")
        h2h_display.columns = ["Date", "Winner", "Loser", "Event", "Prize Pool", "LAN"]
        st.dataframe(h2h_display, use_container_width=True, hide_index=True)

    # ── Wins / Losses stats ──────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Match Stats Comparison")
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric(f"{team_a} Wins",   int(a["wins"]))
    with stat_cols[1]:
        st.metric(f"{team_a} Losses", int(a["losses"]))
    with stat_cols[2]:
        st.metric(f"{team_b} Wins",   int(b["wins"]))
    with stat_cols[3]:
        st.metric(f"{team_b} Losses", int(b["losses"]))

    # ── Timeline: points contribution per month ───────────────
    st.markdown("---")
    st.subheader("📅 Monthly Points Contribution (last 6 months)")

    cutoff_for_timeline = cutoff_dt
    window_start = cutoff_for_timeline - timedelta(days=DECAY_WINDOW_DAYS)

    def monthly_points(team_name: str) -> pd.DataFrame:
        """Compute per-month undecayed points for a team."""
        wins = base_matches[base_matches["winner"] == team_name].copy()
        wins = wins[wins["date"] >= window_start]
        wins["month"] = wins["date"].dt.to_period("M").dt.to_timestamp()
        wins["raw_pts"] = wins.apply(
            lambda r: (
                prize_to_points(r["winner_prize"], W_BOUNTY_OFFERED) +
                prize_to_points(r["loser_prize"], W_BOUNTY_COLLECTED) * 0.5
            ) * (LAN_MULTIPLIER if r["is_lan"] else ONLINE_MULTIPLIER),
            axis=1
        )
        monthly = wins.groupby("month")["raw_pts"].sum().reset_index()
        monthly.columns = ["Month", "Points"]
        monthly["Team"] = team_name
        return monthly

    df_timeline = pd.concat([monthly_points(team_a), monthly_points(team_b)], ignore_index=True)

    if not df_timeline.empty:
        fig_tl = px.bar(
            df_timeline, x="Month", y="Points", color="Team", barmode="group",
            color_discrete_map={team_a: "#58a6ff", team_b: "#f85149"},
        )
        fig_tl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10),
            height=350,
        )
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("Not enough monthly data to display timeline.")

    # ── Pillar description ───────────────────────────────────
    with st.expander("ℹ️ Understanding the VRS Pillars"):
        st.markdown(
            """
| Pillar | Description | Weight |
|--------|-------------|--------|
| **Bounty Offered** | Points the team earns from their own prize winnings. Higher prize events = more points. Losses give a small fraction (~35%). | High |
| **Bounty Collected** | Points from the prize money won by opponents they have beaten. Beat a rich opponent = more points. | Medium |
| **Opponent Network** | Points based on how many teams each defeated opponent has beaten. Beating a team that has beaten many others = more points. | Medium |
| **Recency Decay** | All points are multiplied by a linear decay from 1.0 (today) to 0.0 (6 months ago). Old results matter less. | — |
| **LAN Bonus** | Matches played on LAN get a ×1.5 multiplier vs ×1.0 for online. | — |
"""
        )
