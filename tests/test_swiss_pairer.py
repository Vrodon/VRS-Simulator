"""
Swiss pairer tests — Buchholz pairer with VRS-snapshot seeding.

See NEXT_STEPS_BRACKETS.md §10 for the locked spec; CLAUDE.md "Swiss
Tournament Architecture" for the user-facing contract.

The pairer is a pure module — no Streamlit, no Liquipedia. Inputs are
participant lists, snapshot standings, current records, and opponent
histories; outputs are seed tables and pairings.
"""

from __future__ import annotations

from vrs_engine.swiss_pairer import (
    seed_table,
    r1_split_bracket,
    compute_buchholz,
    pair_round,
)


def test_pair_round_drops_advanced_and_eliminated_teams():
    """
    Teams at 3 wins (advanced) or 3 losses (eliminated) drop out of
    pairings. Step §10.3 spec: filter to ``wins < 3 AND losses < 3``.

    Setup: 6 teams. A,B,C,D in (2,1) — still active. ADV=3-0
    (advanced), ELIM=0-3 (eliminated). Pair_round should emit only the
    (2,1) bucket — outer-pair the four active teams.
    """
    seeds   = ["A", "B", "C", "D", "ADV", "ELIM"]
    records = {
        "A": (2, 1), "B": (2, 1), "C": (2, 1), "D": (2, 1),
        "ADV": (3, 0), "ELIM": (0, 3),
    }
    opps = {"A": ["X"], "B": ["Y"], "C": ["Z"], "D": ["W"],
            "ADV": ["X", "Y", "Z"], "ELIM": ["A", "B", "C"]}

    pairs = pair_round(records, opps, seeds)

    # Only the four active (2,1) teams pair; ADV / ELIM never appear.
    flat = {t for p in pairs for t in p}
    assert flat == {"A", "B", "C", "D"}
    assert pairs == [("A", "D"), ("B", "C")]


def test_pair_round_avoids_rematch_via_greedy_swap():
    """
    Greedy single-swap: when the outer pair would replay an earlier
    match, the partner is bumped up the bucket until a clean pair is
    found. Spec lock 10.Q4.

    Setup: R3 with four teams in the (1,1) bucket. A-D played each
    other earlier (rematch on outer-pair); B-C also played each other.
    Greedy swap on (A, D) → (A, C); the leftover (B, D) is clean.
    Each team also played a (1,0) filler opponent so Buchholz is tied.
    """
    seeds   = ["A", "B", "C", "D"]
    records = {
        "A": (1, 1), "B": (1, 1), "C": (1, 1), "D": (1, 1),
        "FA": (1, 0), "FB": (1, 0), "FC": (1, 0), "FD": (1, 0),
    }
    opps = {
        "A": ["D", "FA"], "B": ["C", "FB"],
        "C": ["B", "FC"], "D": ["A", "FD"],
        "FA": ["A"], "FB": ["B"], "FC": ["C"], "FD": ["D"],
    }

    pairs = pair_round(records, opps, seeds)

    one_one = [(a, b) for a, b in pairs
                if records[a] == (1, 1) and records[b] == (1, 1)]
    assert one_one == [("A", "C"), ("B", "D")]


def test_pair_round_buchholz_desc_beats_seed_asc():
    """
    Within a single record bucket, Buchholz desc breaks ties before seed
    asc. Set up an 8-team R3 scenario where four teams sit in the (1,1)
    bucket with distinct Buchholz scores; the others occupy (2,0) and
    (0,2) buckets and pair separately.
    """
    seeds = ["A", "B", "C", "D", "X", "Y", "Z", "W"]
    records = {
        "A": (1, 1), "B": (1, 1), "C": (1, 1), "D": (1, 1),
        "X": (2, 0), "Y": (2, 0),
        "Z": (0, 2), "W": (0, 2),
    }
    # Buchholz calculation on the (1,1) bucket:
    #   A played [X(2W), Z(0W)] → Buch=2
    #   B played [X(2W), Y(2W)] → Buch=4
    #   C played [Y(2W), Z(0W)] → Buch=2
    #   D played [W(0W), Y(2W)] → Buch=2
    # Ranked: B(4) > A(2,seed=0) > C(2,seed=2) > D(2,seed=3).
    # Outer-pair within the (1,1) bucket: (B, D), (A, C).
    opps = {
        "A": ["X", "Z"], "B": ["X", "Y"], "C": ["Y", "Z"], "D": ["W", "Y"],
        "X": ["A", "B"], "Y": ["B", "C", "D"],
        "Z": ["A", "C"], "W": ["D"],
    }

    pairs = pair_round(records, opps, seeds)

    # Filter to the (1,1) bucket pairs only — other buckets pair on their
    # own and are not the focus of this test.
    one_one = [(a, b) for a, b in pairs
                if records[a] == (1, 1) and records[b] == (1, 1)]
    assert one_one == [("B", "D"), ("A", "C")]


def test_pair_round_single_bucket_outer_pair_seed_tiebreak():
    """
    Smallest R>1 case: one record bucket, all teams tied on Buchholz.
    Sort by seed asc, outer-pair (A vs D, B vs C).

    Setup: 4 teams all 1-0 after R1; each played a different external
    opponent so Buchholz is tied at 0 for everyone.
    """
    seeds   = ["A", "B", "C", "D"]
    records = {"A": (1, 0), "B": (1, 0), "C": (1, 0), "D": (1, 0)}
    opps    = {"A": ["X1"], "B": ["X2"], "C": ["X3"], "D": ["X4"]}

    pairs = pair_round(records, opps, seeds)

    assert pairs == [("A", "D"), ("B", "C")]


def test_compute_buchholz_sums_opponents_current_wins():
    """
    Buchholz = sum of opponents' CURRENT wins. No opp-of-opp; recompute
    every round; never store. Spec §10.3.

    Scenario: team T played A and B. A is currently 2-1, B is currently
    1-2. Buchholz(T) = 2 + 1 = 3.
    """
    opponents_played = {"T": ["A", "B"], "A": ["T", "C"], "B": ["T", "C"],
                        "C": ["A", "B"]}
    records = {
        "T": (1, 1), "A": (2, 1), "B": (1, 2), "C": (2, 1),
    }

    assert compute_buchholz("T", opponents_played, records) == 3


def test_r1_split_bracket_pairs_top_half_with_bottom_half_in_order():
    """
    CS2 Swiss R1 = top-half-vs-bottom-half split bracket. For 16 teams
    seeded 1..16, the pairings are:
      #1 vs #9, #2 vs #10, ..., #8 vs #16.
    User correction in §10.3: NOT outer-pair (1v16) — split-bracket.
    """
    seeds = [f"T{i}" for i in range(1, 17)]   # T1..T16, idx 0=#1

    pairs = r1_split_bracket(seeds)

    assert pairs == [
        ("T1", "T9"), ("T2", "T10"), ("T3", "T11"), ("T4", "T12"),
        ("T5", "T13"), ("T6", "T14"), ("T7", "T15"), ("T8", "T16"),
    ]


def test_seed_table_appends_missing_teams_at_tail_alphabetically():
    """
    Teams not present in the chosen snapshot (open-qualifier winners,
    brand-new orgs) get appended after every ranked team, in
    alphabetical order. Ranked teams keep their rank-asc order.
    Spec lock 10.Q6.
    """
    participants = ["NewOrg", "FaZe", "Brand", "Vitality"]
    standings    = {"FaZe": 5, "Vitality": 1}   # NewOrg, Brand missing

    seeded = seed_table(participants, standings)

    assert seeded == ["Vitality", "FaZe", "Brand", "NewOrg"]


def test_seed_table_orders_participants_by_snapshot_rank_ascending():
    """
    Snapshot rank is the seeding authority. seed_table(...) returns the
    participants sorted by rank ascending, so callers can index into the
    result to recover seed numbers (idx+1 = seed).
    """
    participants = ["FaZe", "Vitality", "G2"]
    standings    = {"FaZe": 5, "Vitality": 1, "G2": 12}

    assert seed_table(participants, standings) == ["Vitality", "FaZe", "G2"]
