"""
Phase 5 §5.2 — per-format placement labellers.

These pure functions emit per-team within-stage placement info that the
engine emission path combines with ``stage_graph.compute_place_offsets``
to produce absolute Liquipedia-compatible place labels (e.g. ``"25th-27th"``,
``"7th-8th"``).

Per-format outputs all share the same shape:

    list[tuple[bucket_label, list[teams]]]

ordered best-first (rank 1 at index 0). Caller composes with offset:

    place_range_for_bucket(idx) =
        offset + sum_of_better_bucket_sizes + 1
        ..
        offset + sum_of_better_bucket_sizes + this_bucket_size

Tied bucket sizes ≥ 2 → range label like ``"7th-8th"`` (single shared
prize per Q3 lock).
"""

from __future__ import annotations

from data_loaders.bracket_parser import Stage


def compute_swiss_exit_buckets(
    stage: Stage,
    picks: dict[str, str],
) -> list[tuple[str, list[str]]]:
    """
    Walk Swiss matchlist, return eliminated teams grouped by their final
    W-L record. Best-eliminated (most wins at exit) first.

    A team is "eliminated" when it reaches the loss threshold for the
    Swiss config — derived from observed max round and total team count.
    For CS2 standard 16-team-3-3 Swiss the threshold is 3L; for smaller
    test configs we approximate via ``ceil(rounds/2)``.

    Bucket label format: ``"W-L"`` (e.g. ``"2-3"`` for the 2-3 bucket).
    Teams within a bucket are alphabetically sorted (no Buchholz tiebreak
    needed at prize-emission time — they all share the same prize).
    """
    if not stage.matches:
        return []

    sw = [m for m in stage.matches if m.sub == "SW"]
    if not sw:
        return []

    max_round = max((m.round_idx for m in sw), default=0)
    if max_round == 0:
        return []

    # Threshold derivation mirrors `compute_stage_advancers` Swiss gate.
    team_count_proxy = (
        len(stage.roster)
        or sum(en.count for en in stage.entrants
               if en.source != "advance_from")
    )
    if team_count_proxy >= 16:
        loss_threshold = 3   # CS2 3-3 Swiss
    else:
        loss_threshold = max(1, (max_round + 1) // 2)

    # Walk picks, accumulate W-L per team.
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for m in sw:
        a, b = m.seed_a, m.seed_b
        if not a or not b:
            continue
        w = m.played_winner or picks.get(m.match_id)
        if w is None:
            continue
        wins.setdefault(a, 0); wins.setdefault(b, 0)
        losses.setdefault(a, 0); losses.setdefault(b, 0)
        if w == a:
            wins[a]   += 1
            losses[b] += 1
        elif w == b:
            wins[b]   += 1
            losses[a] += 1

    # Eliminated = teams with losses ≥ threshold.
    eliminated = [t for t in wins if losses.get(t, 0) >= loss_threshold]

    # Group by W-L; bucket label = f"{W}-{L}".
    buckets: dict[tuple[int, int], list[str]] = {}
    for t in eliminated:
        key = (wins[t], losses[t])
        buckets.setdefault(key, []).append(t)
    for teams in buckets.values():
        teams.sort()  # alphabetical within bucket

    # Sort buckets best-first (highest W desc, then lowest L asc).
    sorted_keys = sorted(buckets, key=lambda k: (-k[0], k[1]))
    return [(f"{w}-{l}", buckets[(w, l)]) for (w, l) in sorted_keys]
