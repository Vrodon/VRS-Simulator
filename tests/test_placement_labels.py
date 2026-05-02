"""
Phase 5 §5.2 — per-format placement labellers.

Emit (team, within-stage-bucket-position, bucket-size) per stage so the
caller can combine with compute_place_offsets() to produce absolute
Liquipedia-compatible place labels (e.g. "25th-27th", "7th-8th").
"""

from __future__ import annotations

from data_loaders.bracket_parser import Stage, BracketMatch
from data_loaders.format_parser import StageEdge


def test_swiss_exit_buckets_groups_eliminated_teams_by_loss_round():
    """
    For a 4-team Swiss / 3-round / 2W-advance / 2L-eliminate config, the
    two eliminated teams fall into separate buckets:

      1-2 bucket: hit 2L in R3 (best eliminated)
      0-2 bucket: hit 2L in R2 (worst eliminated)

    Walker returns ordered list[bucket_label, list[teams]] best-first.
    """
    from vrs_engine.placement_labels import compute_swiss_exit_buckets

    matches = [
        # R1: A>B, C>D
        BracketMatch(match_id="m::SW-R1M0", round_idx=1, slot_idx=0, sub="SW",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="m::SW-R1M1", round_idx=1, slot_idx=1, sub="SW",
                     seed_a="C", seed_b="D"),
        # R2: 1-0 bucket A vs C; 0-1 bucket B vs D. A>C, B>D.
        BracketMatch(match_id="m::SW-R2M0", round_idx=2, slot_idx=0, sub="SW",
                     seed_a="A", seed_b="C"),
        BracketMatch(match_id="m::SW-R2M1", round_idx=2, slot_idx=1, sub="SW",
                     seed_a="B", seed_b="D"),
        # After R2: A=2-0 advance. D=0-2 eliminated. B=1-1, C=1-1.
        # R3: 1-1 vs 1-1 (B vs C). C wins → C=2-1 advance, B=1-2 eliminate.
        BracketMatch(match_id="m::SW-R3M0", round_idx=3, slot_idx=0, sub="SW",
                     seed_a="B", seed_b="C"),
    ]
    stage = Stage(stage_id="m", format="Swiss", display_heading="Swiss",
                  matches=matches, roster=["A", "B", "C", "D"])
    picks = {
        "m::SW-R1M0": "A", "m::SW-R1M1": "C",
        "m::SW-R2M0": "A", "m::SW-R2M1": "B",
        "m::SW-R3M0": "C",
    }

    buckets = compute_swiss_exit_buckets(stage, picks)

    # Best eliminated (1-2) first; worst (0-2) last.
    assert buckets == [("1-2", ["B"]), ("0-2", ["D"])]


def test_gsl_lite_exit_buckets_returns_em_winner_3rd_em_loser_4th():
    """
    GSL-lite (4-team, 4 matches): top 2 advance silent (handled by the
    next stage's prize emission). Dropouts are EM winner = 3rd-in-group,
    EM loser = 4th-in-group. Walker emits within-stage rank buckets only.

    Layout: A>D + B>C in opening; A>B in WM (A advance 1st, B advance 2nd);
    C>D in EM (C is 3rd-in-group, D is 4th-in-group).
    """
    from vrs_engine.placement_labels import compute_gsl_exit_buckets

    matches = [
        BracketMatch(match_id="g::GR-O0", round_idx=1, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="D"),
        BracketMatch(match_id="g::GR-O1", round_idx=1, slot_idx=1, sub="GR",
                     seed_a="B", seed_b="C"),
        BracketMatch(match_id="g::GR-WM", round_idx=2, slot_idx=0, sub="GR",
                     feeder_a="g::GR-O0", feeder_b="g::GR-O1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        BracketMatch(match_id="g::GR-EM", round_idx=2, slot_idx=1, sub="GR",
                     feeder_a="g::GR-O0", feeder_b="g::GR-O1",
                     feeder_a_kind="loser", feeder_b_kind="loser"),
    ]
    stage = Stage(stage_id="g", format="Groups",
                  display_heading="Group A (GSL-lite)",
                  matches=matches, roster=["A", "B", "C", "D"])
    picks = {
        "g::GR-O0": "A", "g::GR-O1": "B",
        "g::GR-WM": "A", "g::GR-EM": "C",
    }

    buckets = compute_gsl_exit_buckets(stage, picks)

    assert buckets == [("3", ["C"]), ("4", ["D"])]


def test_gsl_full_with_decider_returns_decider_loser_3rd_em_loser_4th():
    """
    Full GSL (5 matches): WM winner = 1st advance silent, Decider winner =
    2nd advance silent, Decider loser = 3rd-in-group, EM loser = 4th-in-group.

    Layout: A>D + B>C in opening; A>B in WM (A advance 1st);
    D>C in EM (C is 4th, D goes to Decider);
    B>D in DM (B advance 2nd, D is 3rd-in-group).
    """
    from vrs_engine.placement_labels import compute_gsl_exit_buckets

    matches = [
        BracketMatch(match_id="g::GR-O0", round_idx=1, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="D"),
        BracketMatch(match_id="g::GR-O1", round_idx=1, slot_idx=1, sub="GR",
                     seed_a="B", seed_b="C"),
        BracketMatch(match_id="g::GR-WM", round_idx=2, slot_idx=0, sub="GR",
                     feeder_a="g::GR-O0", feeder_b="g::GR-O1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        BracketMatch(match_id="g::GR-EM", round_idx=2, slot_idx=1, sub="GR",
                     feeder_a="g::GR-O0", feeder_b="g::GR-O1",
                     feeder_a_kind="loser", feeder_b_kind="loser"),
        BracketMatch(match_id="g::GR-DM", round_idx=3, slot_idx=0, sub="GR",
                     feeder_a="g::GR-WM", feeder_b="g::GR-EM",
                     feeder_a_kind="loser", feeder_b_kind="winner"),
    ]
    stage = Stage(stage_id="g", format="Groups",
                  display_heading="Group A (GSL)",
                  matches=matches, roster=["A", "B", "C", "D"])
    picks = {
        "g::GR-O0": "A", "g::GR-O1": "B",
        "g::GR-WM": "A",
        "g::GR-EM": "D",        # D survives (C is 4th)
        "g::GR-DM": "B",        # B beats D in Decider (B = 2nd advance, D = 3rd dropout)
    }

    buckets = compute_gsl_exit_buckets(stage, picks)

    assert buckets == [("3", ["D"]), ("4", ["C"])]


def test_de_compact_8team_returns_lb_final_4th_lb_sf_5_6_lb_r1_7_8():
    """
    Atlanta-style 8-team DE_compact: top 3 advance silent (UB winner = 1st,
    UB-final loser = 2nd, LB-final winner = 3rd). Dropouts:

      "4"   = LB-final loser
      "5-6" = LB-SF (R2) losers
      "7-8" = LB-R1 losers (alphabetical within bucket).

    Bracket walked: A>B, C>D, E>F, G>H in UB R1 (losers B,D,F,H).
    UB R2: A>C, E>G (losers C,G drop to LB R2). UB R3: A>E (final).
    LB R1: B>D, F>H (losers D,H = 7-8 bucket).
    LB R2: B>C, F>G (losers C,G = 5-6 bucket; LB R1 winners go on).
    LB R3 (LB final): B>F → F is 4th, B advances as 3rd.
    """
    from vrs_engine.placement_labels import compute_de_compact_exit_buckets

    matches = []
    # UB R1 (4 matches)
    ub_r1_pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    ub_r1_ids = []
    for i, (a, b) in enumerate(ub_r1_pairs):
        mid = f"d::UB-R1m{i}"
        ub_r1_ids.append(mid)
        matches.append(BracketMatch(match_id=mid, round_idx=1, slot_idx=i,
                                     sub="UB", seed_a=a, seed_b=b))
    # UB R2 (2 matches)
    ub_r2_ids = []
    for i in range(2):
        mid = f"d::UB-R2m{i}"
        ub_r2_ids.append(mid)
        matches.append(BracketMatch(
            match_id=mid, round_idx=2, slot_idx=i, sub="UB",
            feeder_a=ub_r1_ids[i * 2], feeder_b=ub_r1_ids[i * 2 + 1],
            feeder_a_kind="winner", feeder_b_kind="winner",
        ))
    # UB R3 (final)
    matches.append(BracketMatch(
        match_id="d::UB-R3m0", round_idx=3, slot_idx=0, sub="UB",
        feeder_a=ub_r2_ids[0], feeder_b=ub_r2_ids[1],
        feeder_a_kind="winner", feeder_b_kind="winner",
    ))
    # LB R1 (2 matches — UB R1 losers paired)
    lb_r1_ids = []
    for i in range(2):
        mid = f"d::LB-R1m{i}"
        lb_r1_ids.append(mid)
        matches.append(BracketMatch(
            match_id=mid, round_idx=1, slot_idx=i, sub="LB",
            feeder_a=ub_r1_ids[i * 2], feeder_b=ub_r1_ids[i * 2 + 1],
            feeder_a_kind="loser", feeder_b_kind="loser",
        ))
    # LB R2 (2 matches — LB R1 winners + UB R2 losers cross)
    lb_r2_ids = []
    for i in range(2):
        mid = f"d::LB-R2m{i}"
        lb_r2_ids.append(mid)
        matches.append(BracketMatch(
            match_id=mid, round_idx=2, slot_idx=i, sub="LB",
            feeder_a=lb_r1_ids[i], feeder_b=ub_r2_ids[i],
            feeder_a_kind="winner", feeder_b_kind="loser",
        ))
    # LB R3 (LB final — LB R2 winners; compact: no UB drop)
    matches.append(BracketMatch(
        match_id="d::LB-R3m0", round_idx=3, slot_idx=0, sub="LB",
        feeder_a=lb_r2_ids[0], feeder_b=lb_r2_ids[1],
        feeder_a_kind="winner", feeder_b_kind="winner",
    ))

    stage = Stage(stage_id="d", format="DE", display_heading="Group A",
                  matches=matches,
                  roster=["A", "B", "C", "D", "E", "F", "G", "H"])

    picks = {
        "d::UB-R1m0": "A", "d::UB-R1m1": "C",
        "d::UB-R1m2": "E", "d::UB-R1m3": "G",
        "d::UB-R2m0": "A", "d::UB-R2m1": "E",
        "d::UB-R3m0": "A",
        "d::LB-R1m0": "B", "d::LB-R1m1": "F",
        "d::LB-R2m0": "B", "d::LB-R2m1": "F",
        "d::LB-R3m0": "B",
    }

    buckets = compute_de_compact_exit_buckets(stage, picks)

    assert buckets == [
        ("4",   ["F"]),
        ("5-6", ["C", "G"]),
        ("7-8", ["D", "H"]),
    ]


def test_groups_rr_returns_dropouts_ranked_by_wins_skipping_advancers():
    """
    Round-robin group: 4 teams, 6 matches, ``advance_to.count = 2``.
    A goes 3-0, B 2-1, C 1-2, D 0-3 → top 2 (A,B) advance silent;
    dropouts are C (3rd-in-group) and D (4th-in-group).
    """
    from vrs_engine.placement_labels import compute_rr_exit_buckets

    matches = [
        BracketMatch(match_id="r::GR-M0", round_idx=1, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="r::GR-M1", round_idx=1, slot_idx=1, sub="GR",
                     seed_a="A", seed_b="C"),
        BracketMatch(match_id="r::GR-M2", round_idx=1, slot_idx=2, sub="GR",
                     seed_a="A", seed_b="D"),
        BracketMatch(match_id="r::GR-M3", round_idx=1, slot_idx=3, sub="GR",
                     seed_a="B", seed_b="C"),
        BracketMatch(match_id="r::GR-M4", round_idx=1, slot_idx=4, sub="GR",
                     seed_a="B", seed_b="D"),
        BracketMatch(match_id="r::GR-M5", round_idx=1, slot_idx=5, sub="GR",
                     seed_a="C", seed_b="D"),
    ]
    stage = Stage(
        stage_id="r", format="Groups",
        display_heading="Group A (Round-robin)",
        matches=matches, roster=["A", "B", "C", "D"],
        advance_to=[StageEdge("Playoffs", 2, "group_rank")],
    )
    picks = {
        "r::GR-M0": "A", "r::GR-M1": "A", "r::GR-M2": "A",  # A 3-0
        "r::GR-M3": "B", "r::GR-M4": "B",                    # B 2-1
        "r::GR-M5": "C",                                      # C 1-2; D 0-3
    }

    buckets = compute_rr_exit_buckets(stage, picks)

    assert buckets == [("3", ["C"]), ("4", ["D"])]


def test_se_with_bronze_8team_returns_full_placement_chain():
    """
    8-team SE+bronze playoff: every team is a "dropout" (terminal stage).
    Buckets emitted in placement order:

      "1"   = final winner
      "2"   = final loser
      "3"   = bronze winner
      "4"   = bronze loser
      "5-8" = QF losers (4 teams alphabetical within bucket)
    """
    from vrs_engine.placement_labels import compute_se_exit_buckets

    r1_pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    r1_ids = []
    matches = []
    for i, (a, b) in enumerate(r1_pairs):
        mid = f"s::R1m{i}"
        r1_ids.append(mid)
        matches.append(BracketMatch(match_id=mid, round_idx=1, slot_idx=i,
                                     seed_a=a, seed_b=b))
    r2_ids = []
    for i in range(2):
        mid = f"s::R2m{i}"
        r2_ids.append(mid)
        matches.append(BracketMatch(
            match_id=mid, round_idx=2, slot_idx=i,
            feeder_a=r1_ids[i * 2], feeder_b=r1_ids[i * 2 + 1],
            feeder_a_kind="winner", feeder_b_kind="winner",
        ))
    matches.append(BracketMatch(
        match_id="s::R3m0", round_idx=3, slot_idx=0,
        feeder_a=r2_ids[0], feeder_b=r2_ids[1],
        feeder_a_kind="winner", feeder_b_kind="winner",
    ))
    matches.append(BracketMatch(
        match_id="s::Bm0", round_idx=3, slot_idx=1, is_bronze=True,
        feeder_a=r2_ids[0], feeder_b=r2_ids[1],
        feeder_a_kind="loser", feeder_b_kind="loser",
    ))

    stage = Stage(stage_id="s", format="SE_with_bronze",
                  display_heading="Playoffs", matches=matches)

    picks = {
        "s::R1m0": "A", "s::R1m1": "C", "s::R1m2": "E", "s::R1m3": "G",
        "s::R2m0": "A", "s::R2m1": "E",
        "s::R3m0": "A",
        "s::Bm0":  "C",
    }

    buckets = compute_se_exit_buckets(stage, picks)

    assert buckets == [
        ("1",   ["A"]),
        ("2",   ["E"]),
        ("3",   ["C"]),
        ("4",   ["G"]),
        ("5-8", ["B", "D", "F", "H"]),
    ]
