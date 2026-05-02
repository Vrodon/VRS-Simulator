"""
Swiss Pairer
============
CS2 Swiss-system pairing logic with VRS-snapshot seeding and Buchholz
tiebreakers. See ``NEXT_STEPS_BRACKETS.md`` §10 + ``CLAUDE.md`` "Swiss
Tournament Architecture" for the full locked spec.

This module is pure — no Streamlit, no Liquipedia. Inputs are
participants, snapshot standings, records, and opponent histories.
Outputs are seed tables and pairings.
"""

from __future__ import annotations


def seed_table(participants: list[str],
               snapshot_standings: dict[str, int]) -> list[str]:
    """
    Sort participants by snapshot rank ascending. Result indexes 0..N-1
    correspond to seeds #1..#N.

    Teams missing from the snapshot are appended at the tail in
    alphabetical order (spec lock 10.Q6).
    """
    def key(t: str) -> tuple[int, int | str]:
        rank = snapshot_standings.get(t)
        if rank is None:
            return (1, t)            # missing → tail bucket, alphabetical
        return (0, rank)             # ranked → head bucket, rank asc
    return sorted(participants, key=key)


def r1_split_bracket(seeds: list[str]) -> list[tuple[str, str]]:
    """
    CS2 Swiss R1 pairings: top-half-vs-bottom-half split bracket.

    For N seeds, ``seeds[i]`` plays ``seeds[i + N//2]`` for i in 0..N//2-1.
    Concretely (16-team): #1v#9, #2v#10, ..., #8v#16.

    Spec correction §10.3 — NOT outer-pair (1v16). CS2 Major / S-Tier
    organisers use the split-bracket variant.
    """
    half = len(seeds) // 2
    return [(seeds[i], seeds[i + half]) for i in range(half)]


def compute_buchholz(team: str,
                      opponents_played: dict[str, list[str]],
                      records: dict[str, tuple[int, int]]) -> int:
    """
    Buchholz score = sum of ``team``'s opponents' CURRENT wins.

    No opponent-of-opponent. Recalculated every round (caller passes
    fresh ``records`` after each round update); never stored.
    """
    opps = opponents_played.get(team, [])
    return sum(records.get(o, (0, 0))[0] for o in opps)


def pair_round(records: dict[str, tuple[int, int]],
                opponents_played: dict[str, list[str]],
                seeds: list[str]) -> list[tuple[str, str]]:
    """
    R>1 Swiss pairer.

    Steps (spec lock §10.3):
      1. Filter to active teams (wins<3 AND losses<3).
      2. Group by exact W-L record. Pair only within a bucket.
      3. Sort each bucket by Buchholz desc, then seed asc.
      4. Outer-pair within bucket (sorted A,B,C,D → A-D, B-C).
      5. Greedy single-swap on rematch: if (top, bottom) replays, try
         the next-bottom-up; accept rematch only when no swap is clean.

    Bucket emit order: highest wins first, then fewest losses (so the
    1-0 group is paired before 0-1, etc.). Caller can rely on this for
    slot assignment.
    """
    from collections import defaultdict

    seed_idx: dict[str, int] = {t: i for i, t in enumerate(seeds)}
    active: dict[str, tuple[int, int]] = {
        t: r for t, r in records.items() if r[0] < 3 and r[1] < 3
    }

    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for t, r in active.items():
        buckets[r].append(t)

    pairs: list[tuple[str, str]] = []
    sorted_keys = sorted(buckets.keys(), key=lambda k: (-k[0], k[1]))

    for k in sorted_keys:
        teams = buckets[k]
        teams.sort(key=lambda t: (
            -compute_buchholz(t, opponents_played, records),
            seed_idx.get(t, len(seeds)),
        ))
        pairs.extend(_pair_within_bucket(teams, opponents_played))

    return pairs


def _pair_within_bucket(sorted_teams: list[str],
                         opponents_played: dict[str, list[str]]
                         ) -> list[tuple[str, str]]:
    """
    Outer-pair the bucket with greedy single-swap on rematches.

    Algorithm: pop the top, scan candidates from the bottom upward.
    First non-rematch wins. If every remaining candidate is a rematch,
    accept the bottom (minimum-rematch fallback). Repeat.
    """
    unpaired = list(sorted_teams)
    pairs: list[tuple[str, str]] = []

    while len(unpaired) >= 2:
        top = unpaired[0]
        partner_idx: int | None = None
        for j in range(len(unpaired) - 1, 0, -1):
            cand = unpaired[j]
            if cand not in opponents_played.get(top, []):
                partner_idx = j
                break
        if partner_idx is None:
            partner_idx = len(unpaired) - 1   # forced rematch
        pairs.append((top, unpaired[partner_idx]))
        # Drop top + partner. Remove partner first (higher index) so the
        # top index stays valid.
        unpaired.pop(partner_idx)
        unpaired.pop(0)

    return pairs
