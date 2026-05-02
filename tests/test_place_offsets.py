"""
Phase 5 §5.3 — ``compute_place_offsets`` maps each stage to the absolute
place where its dropouts begin. Caller composes per-stage buckets +
offset to produce Liquipedia-compatible absolute labels.

Linear chain: offset(terminal) = 0; offset(N) = team_count(N+1) +
dropout_count(N+1).
Fan-in: sibling stages share an offset (aggregation across siblings
happens at integration time, not at offset time).
"""

from __future__ import annotations

from data_loaders.bracket_parser import Stage
from data_loaders.format_parser import StageDef, StageEdge, StageEntrant
from data_loaders.stage_graph import build_stage_graph


def test_compute_place_offsets_two_stage_linear_chain():
    """
    Group(4 teams, top 2 advance) → Playoffs(2 teams).

    Playoffs is terminal — offset 0. Group's 2 dropouts get places 3..4,
    so Group offset = team_count(Playoffs) = 2.
    """
    from vrs_engine.placement_labels import compute_place_offsets

    grp_def = StageDef(name="Group", team_count=4, format="Groups",
                       advance_to=[StageEdge("Playoffs", 2, "group_rank")])
    play_def = StageDef(name="Playoffs", team_count=2, format="SE")

    grp_stage = Stage(stage_id="g", format="Groups",
                      display_heading="Group", matches=[])
    play_stage = Stage(stage_id="p", format="SE",
                       display_heading="Playoffs", matches=[])
    build_stage_graph([grp_stage, play_stage], [grp_def, play_def],
                      seeded_teams=["A", "B", "C", "D"])

    offsets = compute_place_offsets([grp_stage, play_stage])

    assert offsets == {"Playoffs": 0, "Group": 2}


def test_compute_place_offsets_fan_in_siblings_share_offset():
    """
    Atlanta-style: 2 parallel groups (DE, 8 teams each) → SE+bronze
    Playoffs (6 teams). Both groups share def_name "Group Stage" and
    therefore share offset = team_count(Playoffs) = 6.
    """
    from vrs_engine.placement_labels import compute_place_offsets

    grp_def = StageDef(
        name="Group Stage", team_count=8, format="DE",
        n_groups=2, teams_per_group=8,
        advance_to=[StageEdge("Playoffs", 3, "group_rank")],
        entrants=[StageEntrant("initial_roster", count=8)],
    )
    play_def = StageDef(
        name="Playoffs", team_count=6, format="SE_with_bronze",
        entrants=[StageEntrant("advance_from",
                                upstream_stage="Group Stage", count=6)],
    )

    grp_a = Stage(stage_id="ga", format="DE", display_heading="Group A",
                  matches=[])
    grp_b = Stage(stage_id="gb", format="DE", display_heading="Group B",
                  matches=[])
    play  = Stage(stage_id="p",  format="SE_with_bronze",
                  display_heading="Playoffs", matches=[])
    build_stage_graph([grp_a, grp_b, play], [grp_def, play_def],
                      seeded_teams=[f"T{i}" for i in range(16)])

    offsets = compute_place_offsets([grp_a, grp_b, play])

    assert offsets == {"Playoffs": 0, "Group Stage": 6}


def test_compute_absolute_placements_simple_rr_to_se_chain():
    """
    End-to-end: 4-team RR Group (top 2 advance) → 2-team SE Playoffs.

    Expected per-team absolute labels:

      A → "1st"       (Playoffs final winner)
      B → "2nd"       (Playoffs final loser)
      C → "3rd"       (Group rank 3, offset 2 → absolute place 3)
      D → "4th"       (Group rank 4, offset 2 → absolute place 4)
    """
    from data_loaders.bracket_parser import BracketMatch
    from vrs_engine.placement_labels import compute_absolute_placements

    grp_def = StageDef(
        name="Group", team_count=4, format="Groups",
        advance_to=[StageEdge("Playoffs", 2, "group_rank")],
        entrants=[StageEntrant("initial_roster", count=4)],
    )
    play_def = StageDef(
        name="Playoffs", team_count=2, format="SE",
        entrants=[StageEntrant("advance_from",
                                upstream_stage="Group", count=2)],
    )

    rr_matches = [
        BracketMatch(match_id="g::GR-M0", round_idx=1, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="g::GR-M1", round_idx=1, slot_idx=1, sub="GR",
                     seed_a="A", seed_b="C"),
        BracketMatch(match_id="g::GR-M2", round_idx=1, slot_idx=2, sub="GR",
                     seed_a="A", seed_b="D"),
        BracketMatch(match_id="g::GR-M3", round_idx=1, slot_idx=3, sub="GR",
                     seed_a="B", seed_b="C"),
        BracketMatch(match_id="g::GR-M4", round_idx=1, slot_idx=4, sub="GR",
                     seed_a="B", seed_b="D"),
        BracketMatch(match_id="g::GR-M5", round_idx=1, slot_idx=5, sub="GR",
                     seed_a="C", seed_b="D"),
    ]
    play_matches = [
        BracketMatch(match_id="p::R1m0", round_idx=1, slot_idx=0,
                     seed_a="A", seed_b="B"),
    ]
    grp_stage = Stage(stage_id="g", format="Groups",
                      display_heading="Group", matches=rr_matches,
                      roster=["A", "B", "C", "D"])
    play_stage = Stage(stage_id="p", format="SE",
                       display_heading="Playoffs", matches=play_matches,
                       roster=["A", "B"])
    build_stage_graph([grp_stage, play_stage], [grp_def, play_def],
                      seeded_teams=["A", "B", "C", "D"])

    picks = {
        "g::GR-M0": "A", "g::GR-M1": "A", "g::GR-M2": "A",  # A 3-0
        "g::GR-M3": "B", "g::GR-M4": "B",                    # B 2-1
        "g::GR-M5": "C",                                      # C 1-2; D 0-3
        "p::R1m0":  "A",                                      # Playoffs A>B
    }

    placements = compute_absolute_placements([grp_stage, play_stage], picks)

    assert placements == {
        "A": "1st",
        "B": "2nd",
        "C": "3rd",
        "D": "4th",
    }
