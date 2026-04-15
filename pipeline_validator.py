"""
Pipeline Validator
==================
Loads the latest Valve snapshot from cache, runs the VRS engine on
Valve-only data, and compares every factor team-by-team against
Valve's official standings.

Usage
-----
    python pipeline_validator.py [--date 2026_04_06] [--year 2026] [--top N]

Output
------
    Per-factor RMSE / MAE / MaxErr table
    Per-team discrepancy table sorted by |Δ total_points|
    (Optionally --factor bo_factor|bc_factor|on_factor|... for drill-down)
"""

import os
import sys
import pickle
import argparse
import math
from datetime import datetime

import pandas as pd

# ── Project root on the path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from vrs_engine import Store, run_vrs

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR     = "cache/github_vrs"
DEFAULT_DATE  = "2026_04_06"
DEFAULT_YEAR  = "2026"

FACTORS = [
    "bo_factor",
    "bc_factor",
    "on_factor",
    "lan_factor",
    "seed_combined",
    "seed",
    "h2h_delta",
    "total_points",
]

FACTOR_LABELS = {
    "bo_factor":     "Bounty Offered",
    "bc_factor":     "Bounty Collected",
    "on_factor":     "Opponent Network",
    "lan_factor":    "LAN Wins",
    "seed_combined": "Factor Average",
    "seed":          "Seed (SRV)",
    "h2h_delta":     "H2H Delta",
    "total_points":  "Total Points (FRV)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cache loader — ignores TTL (historical snapshots are immutable)
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache_force(date_str: str, year: str) -> dict | None:
    """Load cache pickle regardless of age."""
    path = os.path.join(CACHE_DIR, f"vrs_{year}_{date_str}.pkl")
    if not os.path.exists(path):
        print(f"[ERROR] Cache not found: {path}")
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        print(f"[OK] Loaded cache: {path}")
        return data
    except Exception as e:
        print(f"[ERROR] Could not read cache: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def rmse(errors: list[float]) -> float:
    if not errors:
        return 0.0
    return math.sqrt(sum(e**2 for e in errors) / len(errors))


def mae(errors: list[float]) -> float:
    if not errors:
        return 0.0
    return sum(abs(e) for e in errors) / len(errors)


def max_abs(errors: list[float]) -> float:
    if not errors:
        return 0.0
    return max(abs(e) for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VRS Pipeline Validator")
    parser.add_argument("--date",   default=DEFAULT_DATE, help="Valve snapshot date (YYYY_MM_DD)")
    parser.add_argument("--year",   default=DEFAULT_YEAR, help="Valve snapshot year")
    parser.add_argument("--top",    type=int, default=30,  help="Rows to show in per-team table")
    parser.add_argument("--factor", default=None,          help="Drill-down: show sorted by this factor")
    parser.add_argument("--all",    action="store_true",   help="Show all teams (ignore --top)")
    args = parser.parse_args()

    # ── 1. Load Valve snapshot ────────────────────────────────────────────────
    valve_data = _load_cache_force(args.date, args.year)
    if valve_data is None:
        sys.exit(1)

    tmh        = valve_data["team_match_history"]   # dict[team → list[match]]
    bpm        = valve_data["bo_prizes_map"]         # dict[team → list[prize]]
    valve_df   = valve_data["standings"]             # official Valve standings
    cutoff     = valve_data["cutoff_datetime"]       # datetime (e.g. 2026-04-06)

    print(f"\n{'='*70}")
    print(f"  Snapshot date  : {args.date}")
    print(f"  Cutoff used    : {cutoff.date()}")
    print(f"  Valve teams    : {len(valve_df)}")
    print(f"  TMH entries    : {len(tmh)}")
    print(f"{'='*70}\n")

    # ── 2. Build Store and run pipeline ───────────────────────────────────────
    print("Building Store from Valve data...")
    store = Store.from_valve(tmh, bpm)
    print(f"  matches_df rows : {len(store.matches_df)}")
    print(f"  prizes_df rows  : {len(store.prizes_df)}")

    print(f"\nRunning pipeline with cutoff={cutoff.date()} ...")
    result     = run_vrs(store, cutoff=cutoff)
    our_df     = result["standings"]

    print(f"  Our eligible teams : {len(our_df)}")

    # ── 3. Join on team name ──────────────────────────────────────────────────
    # Roster-split teams appear TWICE in valve_df (active + inactive roster).
    # Deduplicate by keeping the best-ranked (lowest rank number) entry per team.
    valve_df_dedup = (
        valve_df
        .sort_values("rank")
        .drop_duplicates("team", keep="first")
    )
    n_dupes = len(valve_df) - len(valve_df_dedup)
    if n_dupes:
        print(f"  (Roster-split dedup: dropped {n_dupes} duplicate team rows from Valve standings)")

    valve_indexed = valve_df_dedup.set_index("team")
    our_indexed   = our_df.set_index("team")

    common_teams = sorted(
        set(valve_indexed.index) & set(our_indexed.index)
    )
    only_valve   = sorted(set(valve_indexed.index) - set(our_indexed.index))
    only_ours    = sorted(set(our_indexed.index)   - set(valve_indexed.index))

    print(f"\n  Teams in both           : {len(common_teams)}")
    if only_valve:
        print(f"  Teams only in Valve     : {len(only_valve)}")
        if len(only_valve) <= 10:
            print(f"    -> {', '.join(only_valve)}")
    if only_ours:
        print(f"  Teams only in ours      : {len(only_ours)}")
        if len(only_ours) <= 10:
            print(f"    -> {', '.join(only_ours)}")

    if not common_teams:
        print("\n[ERROR] No teams in common — cannot compare!")
        sys.exit(1)

    # ── 4. Compute per-team deltas ────────────────────────────────────────────
    rows = []
    for team in common_teams:
        v = valve_indexed.loc[team]   # guaranteed unique after dedup → Series
        o = our_indexed.loc[team]     # guaranteed unique (pipeline) → Series

        row = {"team": team}
        for f in FACTORS:
            try:
                v_val = float(v[f]) if f in v.index else 0.0
            except (TypeError, ValueError):
                v_val = 0.0
            try:
                o_val = float(o[f]) if f in o.index else 0.0
            except (TypeError, ValueError):
                o_val = 0.0
            row[f"valve_{f}"] = v_val
            row[f"our_{f}"]   = o_val
            row[f"delta_{f}"] = o_val - v_val   # positive = we're higher

        rows.append(row)

    cmp_df = pd.DataFrame(rows)

    # ── 5. Per-factor summary table ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FACTOR ACCURACY SUMMARY  ({len(common_teams)} common teams)")
    print(f"{'='*70}")
    print(f"  {'Factor':<22}  {'RMSE':>8}  {'MAE':>8}  {'MaxErr':>8}  {'MeanDelta':>10}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    for f in FACTORS:
        col   = f"delta_{f}"
        errs  = cmp_df[col].tolist()
        label = FACTOR_LABELS[f]
        print(
            f"  {label:<22}  "
            f"{rmse(errs):>8.4f}  "
            f"{mae(errs):>8.4f}  "
            f"{max_abs(errs):>8.4f}  "
            f"{sum(errs)/len(errs):>+10.4f}"
        )

    # ── 6. Per-team comparison table ──────────────────────────────────────────
    sort_col = f"delta_{args.factor}" if args.factor and f"delta_{args.factor}" in cmp_df.columns \
               else "delta_total_points"

    cmp_df["_abs_sort"] = cmp_df[sort_col].abs()
    cmp_df = cmp_df.sort_values("_abs_sort", ascending=False).drop(columns="_abs_sort")

    n_show = len(cmp_df) if args.all else min(args.top, len(cmp_df))

    # Determine which factors to show columns for
    show_factors = [args.factor] if args.factor and f"delta_{args.factor}" in cmp_df.columns \
                   else FACTORS

    print(f"\n{'='*70}")
    sort_label = FACTOR_LABELS.get(args.factor, args.factor) if args.factor else "Total Points"
    print(f"  PER-TEAM DISCREPANCIES  (sorted by |diff {sort_label}|, top {n_show})")
    print(f"{'='*70}")

    # Choose compact or full view
    if args.factor:
        # Single factor drill-down: show valve, ours, delta side-by-side
        f = args.factor
        hdr = f"  {'Team':<25}  {'Valve':>10}  {'Ours':>10}  {'Delta':>10}"
        print(hdr)
        print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}")
        for _, r in cmp_df.head(n_show).iterrows():
            print(
                f"  {r['team']:<25}  "
                f"{r[f'valve_{f}']:>10.4f}  "
                f"{r[f'our_{f}']:>10.4f}  "
                f"{r[f'delta_{f}']:>+10.4f}"
            )
    else:
        # All factors: two lines per team (valve / ours)
        for _, r in cmp_df.head(n_show).iterrows():
            team = r["team"]
            print(f"\n  {team}")
            print(f"    {'Factor':<18}  {'Valve':>8}  {'Ours':>8}  {'Delta':>8}")
            print(f"    {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}")
            for f in show_factors:
                label = FACTOR_LABELS[f]
                v_val = r[f"valve_{f}"]
                o_val = r[f"our_{f}"]
                d_val = r[f"delta_{f}"]
                flag  = "  <<< BIG" if abs(d_val) > 0.05 * max(abs(v_val), 0.01) else "        "
                print(f"    {label:<18}  {v_val:>8.4f}  {o_val:>8.4f}  {d_val:>+8.4f}{flag}")

    # ── 7. Eligibility mismatch report ───────────────────────────────────────
    if only_valve or only_ours:
        print(f"\n{'='*70}")
        print("  ELIGIBILITY MISMATCHES")
        print(f"{'='*70}")
        if only_valve:
            print(f"\n  In Valve standings but NOT in our pipeline ({len(only_valve)} teams):")
            for t in only_valve[:20]:
                v = valve_indexed.loc[t]
                pts = v.get("total_points", "?")
                rank = v.get("rank", "?")
                print(f"    Rank {rank:>4}  {t:<30}  {pts} pts")
            if len(only_valve) > 20:
                print(f"    ... and {len(only_valve)-20} more")
        if only_ours:
            print(f"\n  In our pipeline but NOT in Valve standings ({len(only_ours)} teams):")
            for t in only_ours[:20]:
                o = our_indexed.loc[t]
                pts = o.get("total_points", "?")
                print(f"    {t:<30}  {pts} pts")
            if len(only_ours) > 20:
                print(f"    ... and {len(only_ours)-20} more")

    # ── 8. Quick rank correlation ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RANK COMPARISON  (our rank vs Valve rank, common teams)")
    print(f"{'='*70}")

    # Re-rank our results among common teams only
    our_common = our_indexed.loc[common_teams].sort_values("total_points", ascending=False)
    our_common["our_rank"] = range(1, len(our_common) + 1)

    rank_cmp = []
    for team in common_teams:
        v_rank = int(valve_indexed.loc[team]["rank"])
        o_rank = int(our_common.loc[team]["our_rank"])
        rank_cmp.append({"team": team, "valve_rank": v_rank, "our_rank": o_rank,
                          "rank_delta": o_rank - v_rank})

    rank_df = pd.DataFrame(rank_cmp).sort_values("rank_delta", key=abs, ascending=False)

    rank_errors = rank_df["rank_delta"].tolist()
    print(f"\n  Rank RMSE    : {rmse(rank_errors):.2f}")
    print(f"  Rank MAE     : {mae(rank_errors):.2f}")
    print(f"  Rank MaxErr  : {max_abs(rank_errors):.0f}")

    top_rank_n = min(20, len(rank_df))
    print(f"\n  Largest rank mismatches (top {top_rank_n}):")
    print(f"  {'Team':<28}  {'Valve':>6}  {'Ours':>6}  {'Delta':>7}")
    print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*7}")
    for _, r in rank_df.head(top_rank_n).iterrows():
        print(
            f"  {r['team']:<28}  "
            f"{int(r['valve_rank']):>6}  "
            f"{int(r['our_rank']):>6}  "
            f"{int(r['rank_delta']):>+7}"
        )

    print(f"\n{'='*70}")
    print("  Validation complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
