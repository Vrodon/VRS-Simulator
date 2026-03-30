"""
VRS Engine Core

Full two-phase VRS computation: Factor Scoring (Phase 1) + Head-to-Head (Phase 2).
"""

import pandas as pd
from datetime import datetime, timedelta

from .constants import (
    DECAY_DAYS, SEED_MIN, SEED_MAX, TOP_N, ON_ITERS, BASE_K
)
from .math_helpers import (
    curve, event_stakes, age_weight, lerp, expected_win, top_n_sum
)


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
    bo_sum: dict[str, float] = {}
    for team in all_teams:
        wins = df[df["winner"] == team]
        if wins.empty:
            bo_sum[team] = 0.0
        else:
            contribs = wins["winner_prize"] * wins["age_w"]
            bo_sum[team] = float(contribs.nlargest(TOP_N).sum())

    sorted_bo_sums = sorted(bo_sum.values(), reverse=True)
    ref_5th = sorted_bo_sums[4] if len(sorted_bo_sums) >= 5 else (
              sorted_bo_sums[-1] if sorted_bo_sums else 1.0)
    ref_5th = max(ref_5th, 1e-9)

    bo_factor: dict[str, float] = {
        t: curve(min(1.0, bo_sum[t] / ref_5th))
        for t in all_teams
    }

    # ── Factor 2: Bounty Collected ─────────────────────────────────
    bc_factor: dict[str, float] = {}
    bc_sum_raw: dict[str, float] = {}
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
        on_factor.update(new_on)

    on_factor_final: dict[str, float] = {t: on_factor.get(t, 0.0) for t in eligible}

    # ── Factor 4: LAN Wins ─────────────────────────────────────────
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

    ratings:   dict[str, float] = {t: seeds[t] for t in eligible}
    h2h_delta: dict[str, float] = {t: 0.0      for t in eligible}

    for _, row in df.iterrows():
        w, l = str(row["winner"]), str(row["loser"])
        if w not in ratings or l not in ratings:
            continue

        r_w, r_l = ratings[w], ratings[l]
        E_w = expected_win(r_w, r_l)

        K = BASE_K * float(row["age_w"])

        d_w =  K * (1.0 - E_w)
        d_l =  K * (0.0 - (1.0 - E_w))

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
            "bo_sum":           round(bo_sum.get(t, 0.0),        2),
            "bo_factor":        round(bo_factor.get(t, 0.0),     4),
            "bc_pre_curve":     round(bc_sum_raw.get(t, 0.0),    4),
            "bc_factor":        round(bc_factor.get(t, 0.0),     4),
            "on_factor":        round(on_factor_final.get(t,0.0),4),
            "lan_factor":       round(lan_factor.get(t, 0.0),    4),
            "lan_wins":         int(lan_wins_ct.get(t, 0)),
            "seed_combined":    round(combined.get(t, 0.0),      4),
            "wins":             int(wins_counts.get(t, 0)),
            "losses":           int(match_counts.get(t, 0) - wins_counts.get(t, 0)),
            "total_matches":    int(match_counts.get(t, 0)),
        })

    result = (pd.DataFrame(records)
              .sort_values("total_points", ascending=False)
              .reset_index(drop=True))
    result["rank"] = result.index + 1
    return result
