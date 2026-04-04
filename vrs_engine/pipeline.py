"""
VRS Engine — Pipeline

The script: knows the order, calls the right calculator functions, and
assembles the final standings DataFrame.

Single public entry point:

    result = run(store, cutoff=datetime)

    result["standings"]  — pd.DataFrame, ranked by total_points
    result["match_h2h"]  — dict[match_id → per-match H2H detail] for UI
"""

import pandas as pd
from datetime import datetime, timedelta

from .constants import DECAY_DAYS
from .math_helpers import age_weight, event_stakes
from .store import Store
from .calculator import (
    compute_bo,
    compute_bc,
    compute_on,
    compute_lan,
    compute_seed,
    compute_h2h,
)


def run(store: Store, cutoff: datetime = None) -> dict:
    """
    Full two-phase VRS computation.

    Parameters
    ----------
    store   Store  populated with matches_df and prizes_df
    cutoff  datetime  reference date; age weights are relative to this.
                      Defaults to datetime.now().

    Returns
    -------
    dict with keys:
        "standings"  pd.DataFrame   one row per eligible team, sorted by
                                    total_points descending, with columns:
                                    rank, team, total_points, seed, h2h_delta,
                                    bo_sum, bo_factor, bc_pre_curve, bc_factor,
                                    on_factor, lan_factor, lan_wins,
                                    seed_combined, wins, losses, total_matches

        "match_h2h"  dict[int → dict]   per-match H2H detail for team
                                         breakdown display; keyed by match_id.
                                         Each value: {winner, loser,
                                                      w_delta, l_delta}
    """
    if cutoff is None:
        cutoff = datetime.now()

    window_start = cutoff - timedelta(days=DECAY_DAYS)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1 — Filter both tables to the active 180-day window
    # ═══════════════════════════════════════════════════════════════════════

    matches = store.matches_df[
        (store.matches_df["date"] >= window_start) &
        (store.matches_df["date"] <= cutoff)
    ].copy()

    prizes = store.prizes_df[
        (store.prizes_df["date"] >= window_start) &
        (store.prizes_df["date"] <= cutoff)
    ].copy()

    if matches.empty:
        return {"standings": pd.DataFrame(), "match_h2h": {}}

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2 — Compute age_w and ev_w once; attach to both tables
    #
    # age_w  — time decay modifier per row  [0, 1]
    # ev_w   — event stakes per match       [0, 1]  (from prize_pool only)
    #
    # These are computed here and passed into every calculator function so
    # the math layer never needs to know about the cutoff date.
    # ═══════════════════════════════════════════════════════════════════════

    matches["age_w"] = matches["date"].apply(lambda d: age_weight(d, cutoff))
    matches["ev_w"]  = matches["prize_pool"].apply(event_stakes)
    prizes["age_w"]  = prizes["date"].apply(lambda d: age_weight(d, cutoff))

    # Drop zero-weight rows (age exactly 0 means at the boundary; safe to exclude)
    matches = matches[matches["age_w"] > 0].reset_index(drop=True)
    prizes  = prizes[prizes["age_w"]   > 0].reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3 — Build team list and apply eligibility criteria
    #
    # Eligible teams must have:
    #   ≥ 1 win   in the window
    #   ≥ 5 total matches (wins + losses) in the window
    # ═══════════════════════════════════════════════════════════════════════

    all_teams = sorted(
        set(matches["winner"].tolist() + matches["loser"].tolist())
    )

    wins_ct  = {t: int((matches["winner"] == t).sum()) for t in all_teams}
    match_ct = {
        t: int(((matches["winner"] == t) | (matches["loser"] == t)).sum())
        for t in all_teams
    }

    eligible = [
        t for t in all_teams
        if wins_ct.get(t, 0) >= 1 and match_ct.get(t, 0) >= 5
    ]

    if not eligible:
        return {"standings": pd.DataFrame(), "match_h2h": {}}

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4 — Bounty Offered (BO)
    #
    # Uses prizes_df.  all_teams (not just eligible) because ineligible
    # teams still appear as opponents in BC and ON calculations.
    # ═══════════════════════════════════════════════════════════════════════

    bo_sum, bo_factor = compute_bo(prizes, all_teams)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5 — Bounty Collected (BC)
    # ═══════════════════════════════════════════════════════════════════════

    bc_pre, bc_factor = compute_bc(matches, bo_factor, eligible)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6 — Opponent Network (ON)  — 6 PageRank iterations
    # ═══════════════════════════════════════════════════════════════════════

    on_factor = compute_on(matches, bo_factor, eligible)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7 — LAN Wins
    # ═══════════════════════════════════════════════════════════════════════

    lan_ct, lan_factor = compute_lan(matches, eligible)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8 — Factor Score (seed)
    #
    # Average the four factors, lerp into [SEED_MIN, SEED_MAX].
    # ═══════════════════════════════════════════════════════════════════════

    combined, seeds = compute_seed(bo_factor, bc_factor, on_factor, lan_factor, eligible)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 9 — Head-to-Head (H2H)
    #
    # Glicko/Elo replay, chronological from seed rating.
    # ═══════════════════════════════════════════════════════════════════════

    h2h_delta, match_h2h = compute_h2h(matches, seeds, eligible)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 10 — Assemble output DataFrame
    # ═══════════════════════════════════════════════════════════════════════

    records = []
    for t in eligible:
        seed = seeds[t]
        h2h  = h2h_delta[t]
        records.append({
            "team":          t,
            "total_points":  round(seed + h2h,              1),
            "seed":          round(seed,                    1),
            "h2h_delta":     round(h2h,                    1),
            "bo_sum":        round(bo_sum.get(t, 0.0),      2),
            "bo_factor":     round(bo_factor.get(t, 0.0),   4),
            "bc_pre_curve":  round(bc_pre.get(t, 0.0),      4),
            "bc_factor":     round(bc_factor.get(t, 0.0),   4),
            "on_factor":     round(on_factor.get(t, 0.0),   4),
            "lan_factor":    round(lan_factor.get(t, 0.0),  4),
            "lan_wins":      int(lan_ct.get(t, 0)),
            "seed_combined": round(combined.get(t, 0.0),    4),
            "wins":          int(wins_ct.get(t, 0)),
            "losses":        int(match_ct.get(t, 0) - wins_ct.get(t, 0)),
            "total_matches": int(match_ct.get(t, 0)),
        })

    standings = (
        pd.DataFrame(records)
        .sort_values("total_points", ascending=False)
        .reset_index(drop=True)
    )
    standings["rank"] = standings.index + 1

    return {"standings": standings, "match_h2h": match_h2h}
