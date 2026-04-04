"""
VRS Engine — Calculator

All mathematical operations applied to the data.

Each function receives pre-filtered, age-weighted DataFrames (produced and
annotated by pipeline.py) plus any factor dicts already computed, and returns
new factor dicts.  No I/O, no state, no ordering decisions — pure computation.

Expected columns on the DataFrames received from pipeline.py
-------------------------------------------------------------
matches_df  match_id, date, winner, loser, prize_pool, is_lan,
            age_w  (float 0–1, computed by pipeline),
            ev_w   (float 0–1, computed by pipeline)

prizes_df   team, date, amount,
            age_w  (float 0–1, computed by pipeline)
"""

from .constants import TOP_N, ON_ITERS, SEED_MIN, SEED_MAX, BASE_K
from .math_helpers import curve, lerp, expected_win, top_n_sum


# ── Factor 1: Bounty Offered ──────────────────────────────────────────────────

def compute_bo(
    prizes_df: "pd.DataFrame",
    all_teams: list[str],
) -> tuple[dict, dict]:
    """
    For each team: sum of top-10 (prize_amount × age_w).
    Normalised using the 5th-highest bo_sum as reference → curve().

    Parameters
    ----------
    prizes_df   team, date, amount, age_w  (window-filtered + weighted)
    all_teams   every team seen in the match window (eligible + ineligible)

    Returns
    -------
    bo_sum    dict[team → float]   raw weighted prize total
    bo_factor dict[team → float]   normalised via curve()
    """
    bo_sum: dict[str, float] = {}

    for team in all_teams:
        rows = prizes_df[prizes_df["team"] == team]
        if rows.empty:
            bo_sum[team] = 0.0
        else:
            contribs = (rows["amount"] * rows["age_w"]).tolist()
            bo_sum[team] = top_n_sum(contribs)

    # 5th-highest bo_sum is the normalisation reference
    sorted_sums = sorted(bo_sum.values(), reverse=True)
    ref_5th = sorted_sums[4] if len(sorted_sums) >= 5 else (
              sorted_sums[-1] if sorted_sums else 1.0)
    ref_5th = max(ref_5th, 1e-9)

    bo_factor: dict[str, float] = {
        t: curve(min(1.0, bo_sum[t] / ref_5th))
        for t in all_teams
    }

    return bo_sum, bo_factor


# ── Factor 2: Bounty Collected ────────────────────────────────────────────────

def compute_bc(
    matches_df: "pd.DataFrame",
    bo_factor: dict,
    eligible: list[str],
) -> tuple[dict, dict]:
    """
    For each eligible team: top-10 wins at prized events,
    each scored as  bo_factor[opponent] × age_w × ev_w,
    averaged (÷ TOP_N) then normalised via curve().

    Only wins at events with prize_pool > 0 (ev_w > 0) count.

    Parameters
    ----------
    matches_df  window-filtered matches with age_w and ev_w columns
    bo_factor   dict[team → float] from compute_bo()
    eligible    teams that pass the eligibility check

    Returns
    -------
    bc_pre    dict[team → float]   pre-curve average
    bc_factor dict[team → float]   after curve()
    """
    bc_pre:    dict[str, float] = {}
    bc_factor: dict[str, float] = {}

    wins_ev = matches_df[matches_df["ev_w"] > 0]

    for team in eligible:
        team_wins = wins_ev[wins_ev["winner"] == team]
        if team_wins.empty:
            bc_pre[team]    = 0.0
            bc_factor[team] = 0.0
            continue

        entries = [
            bo_factor.get(row.loser, 0.0) * row.age_w * row.ev_w
            for row in team_wins.itertuples(index=False)
        ]
        s = top_n_sum(entries) / TOP_N
        bc_pre[team]    = s
        bc_factor[team] = curve(s)

    return bc_pre, bc_factor


# ── Factor 3: Opponent Network ────────────────────────────────────────────────

def compute_on(
    matches_df: "pd.DataFrame",
    bo_factor: dict,
    eligible: list[str],
) -> dict:
    """
    PageRank-style iteration (ON_ITERS = 6 rounds).

    Initialised with bo_factor values for every team.
    Each iteration: ON_new[team] = sum_top10(ON[opp] × age_w × ev_w) / TOP_N
    Only wins at prized events count (ev_w > 0).

    ✓ VERIFIED: NO curve() applied — the raw average IS the ON score.
    The iterative convergence keeps values in a comparable range.

    Parameters
    ----------
    matches_df  window-filtered matches with age_w and ev_w columns
    bo_factor   dict[team → float] — starting seed for all teams
    eligible    teams that pass the eligibility check

    Returns
    -------
    on_factor  dict[team → float]  (eligible teams only, after 6 iterations)
    """
    # Seed from bo_factor for ALL teams (ineligible teams act as opponents)
    on_factor: dict[str, float] = dict(bo_factor)

    wins_ev = matches_df[matches_df["ev_w"] > 0]

    for _ in range(ON_ITERS):
        new_on: dict[str, float] = {}
        for team in eligible:
            team_wins = wins_ev[wins_ev["winner"] == team]
            if team_wins.empty:
                new_on[team] = 0.0
                continue
            entries = [
                on_factor.get(row.loser, 0.0) * row.age_w * row.ev_w
                for row in team_wins.itertuples(index=False)
            ]
            new_on[team] = top_n_sum(entries) / TOP_N
        on_factor.update(new_on)

    return {t: on_factor.get(t, 0.0) for t in eligible}


# ── Factor 4: LAN Wins ────────────────────────────────────────────────────────

def compute_lan(
    matches_df: "pd.DataFrame",
    eligible: list[str],
) -> tuple[dict, dict]:
    """
    For each eligible team: sum of top-10 age_w values for LAN wins ÷ TOP_N.
    No curve() applied — max possible value is 1.0 (10 × full-weight LAN wins).

    Parameters
    ----------
    matches_df  window-filtered matches with age_w and is_lan columns
    eligible    teams that pass the eligibility check

    Returns
    -------
    lan_wins_ct  dict[team → int]    count of LAN wins in the window
    lan_factor   dict[team → float]  factor score
    """
    lan_wins_ct: dict[str, int]   = {}
    lan_factor:  dict[str, float] = {}

    lan_wins = matches_df[matches_df["is_lan"] == True]

    for team in eligible:
        team_lw = lan_wins[lan_wins["winner"] == team]
        lan_wins_ct[team] = len(team_lw)
        if team_lw.empty:
            lan_factor[team] = 0.0
        else:
            lan_factor[team] = top_n_sum(team_lw["age_w"].tolist()) / TOP_N

    return lan_wins_ct, lan_factor


# ── Seeding ───────────────────────────────────────────────────────────────────

def compute_seed(
    bo_factor:  dict,
    bc_factor:  dict,
    on_factor:  dict,
    lan_factor: dict,
    eligible:   list[str],
) -> tuple[dict, dict]:
    """
    Average the four factors (equal 25% weight each) then lerp into [SEED_MIN, SEED_MAX].

    combined = (bo + bc + on + lan) / 4
    seed     = lerp(SEED_MIN, SEED_MAX,  (combined − min) / (max − min))

    Parameters
    ----------
    bo_factor, bc_factor, on_factor, lan_factor   dicts keyed by team
    eligible    teams to include

    Returns
    -------
    combined  dict[team → float]   pre-lerp average
    seeds     dict[team → float]   final seed in [SEED_MIN, SEED_MAX]
    """
    combined: dict[str, float] = {
        t: (bo_factor.get(t, 0.0) + bc_factor.get(t, 0.0)
            + on_factor.get(t, 0.0) + lan_factor.get(t, 0.0)) / 4.0
        for t in eligible
    }

    vals = list(combined.values())
    lo   = min(vals)
    hi   = max(vals)
    span = max(hi - lo, 1e-9)

    seeds: dict[str, float] = {
        t: lerp(SEED_MIN, SEED_MAX, (combined[t] - lo) / span)
        for t in eligible
    }

    return combined, seeds


# ── Phase 2: Head-to-Head ─────────────────────────────────────────────────────

def compute_h2h(
    matches_df: "pd.DataFrame",
    seeds: dict,
    eligible: list[str],
) -> tuple[dict, dict]:
    """
    Glicko/Elo replay — chronological, starting from each team's seed.

    K = BASE_K × age_w  (recent matches carry more weight).
    Only matches where both winner and loser are eligible teams are processed.

    Parameters
    ----------
    matches_df  window-filtered matches with age_w column, sorted by date
    seeds       dict[team → float] from compute_seed()
    eligible    teams that pass the eligibility check

    Returns
    -------
    h2h_delta  dict[team → float]
        Cumulative Glicko adjustment per team (positive or negative).

    match_h2h  dict[match_id → dict]
        Per-match detail for UI display:
        { "winner": str, "loser": str, "w_delta": float, "l_delta": float }
    """
    ratings:   dict[str, float] = {t: seeds[t] for t in eligible}
    h2h_delta: dict[str, float] = {t: 0.0      for t in eligible}
    match_h2h: dict[int, dict]  = {}

    eligible_set = set(eligible)

    for row in matches_df.sort_values("date").itertuples(index=False):
        w = str(row.winner)
        l = str(row.loser)
        if w not in eligible_set or l not in eligible_set:
            continue

        E_w = expected_win(ratings[w], ratings[l])
        K   = BASE_K * float(row.age_w)

        d_w =  K * (1.0 - E_w)
        d_l = -K * E_w

        ratings[w]   += d_w
        ratings[l]   += d_l
        h2h_delta[w] += d_w
        h2h_delta[l] += d_l

        match_h2h[int(row.match_id)] = {
            "winner":  w,
            "loser":   l,
            "w_delta": d_w,
            "l_delta": d_l,
        }

    return h2h_delta, match_h2h
