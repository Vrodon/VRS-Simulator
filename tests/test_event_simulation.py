"""
Phase 5 §5.1 close-out — engine emit consumes resolve_for_render output.

The fragment render path already calls ``stage_graph.resolve_for_render``;
``_emit_event_simulation_rows`` in app.py currently re-parses + manually
overlays manual_seeds + recomputes Swiss synth, but never runs cross-stage
cascade or seats downstream R1. So picks on downstream R1 produce nothing
because ``seed_a/seed_b`` stay None.

The fix is to share the post-realised emit body via a pure helper that
both call sites can hit.
"""

from __future__ import annotations

from datetime import datetime

from data_loaders.bracket_parser import BracketMatch, Stage
from data_loaders.format_parser import StageDef, StageEdge
from data_loaders.stage_graph import (
    build_stage_graph,
    resolve_for_render,
)


def test_emit_simulation_rows_emits_downstream_r1_after_cascade_seats():
    """
    SE → SE chain; cascade resolves Playoffs roster from upstream picks and
    seats Playoffs R1 via Option B (1v4 / 2v3). A pick on Playoffs R1m0
    (A > D) must surface as an emitted match row — proving the helper sees
    cascade-seated seeds rather than the original-empty parsed_stages.
    """
    from vrs_engine.event_simulation import emit_simulation_rows

    grp_def = StageDef(name="Group", team_count=4, format="SE",
                       advance_to=[StageEdge("Playoffs", 4, "placement")])
    play_def = StageDef(name="Playoffs", team_count=4, format="SE")

    grp_matches = [
        BracketMatch(match_id="g::R1m0", round_idx=1, slot_idx=0,
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="g::R1m1", round_idx=1, slot_idx=1,
                     seed_a="C", seed_b="D"),
        BracketMatch(match_id="g::R2m0", round_idx=2, slot_idx=0,
                     feeder_a="g::R1m0", feeder_b="g::R1m1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
    ]
    play_matches = [
        BracketMatch(match_id="p::R1m0", round_idx=1, slot_idx=0),
        BracketMatch(match_id="p::R1m1", round_idx=1, slot_idx=1),
        BracketMatch(match_id="p::R2m0", round_idx=2, slot_idx=0,
                     feeder_a="p::R1m0", feeder_b="p::R1m1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
    ]
    grp_stage = Stage(stage_id="g", format="SE", display_heading="Group",
                      matches=grp_matches)
    play_stage = Stage(stage_id="p", format="SE", display_heading="Playoffs",
                       matches=play_matches)

    build_stage_graph([grp_stage, play_stage], [grp_def, play_def],
                      seeded_teams=["A", "B", "C", "D"])

    picks = {
        # Upstream — full picks so cascade fires:  A>B, C>D, A>C.
        "g::R1m0": "A", "g::R1m1": "C", "g::R2m0": "A",
        # Downstream R1 pick we want to lock in.
        "p::R1m0": "A",
    }

    realised = resolve_for_render([grp_stage, play_stage], picks)

    extra_matches, _extra_prizes = emit_simulation_rows(
        realised, picks,
        event_name="ChainTest",
        prize_pool=100_000.0,
        is_lan=True,
        event_end=datetime(2026, 6, 1),
    )

    p_r1_rows = [r for r in extra_matches
                 if r["winner"] == "A" and r["loser"] == "D"
                 and r["event"] == "ChainTest"]
    assert len(p_r1_rows) == 1, (
        f"expected one A>D row from cascade-seated Playoffs R1; got "
        f"{[(r['winner'], r['loser']) for r in extra_matches]}"
    )


def test_emit_simulation_rows_emits_prizes_for_upstream_group_dropouts():
    """
    Phase 5 §5.3 — engine emit prize rows now span all formats, not just
    SE/SE+bronze. Group RR (4 teams, top 2 advance) → SE Playoffs (2 teams),
    with full picks + prize_distribution covering every place. Expect prize
    rows for all 4 teams (A=1st, B=2nd, C=3rd, D=4th) — previously C and D
    silently got nothing because the inline emit only handled SE buckets.
    """
    from data_loaders.format_parser import StageEntrant
    from vrs_engine.event_simulation import emit_simulation_rows

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
        "g::GR-M0": "A", "g::GR-M1": "A", "g::GR-M2": "A",
        "g::GR-M3": "B", "g::GR-M4": "B",
        "g::GR-M5": "C",
        "p::R1m0":  "A",
    }

    _matches, prizes = emit_simulation_rows(
        [grp_stage, play_stage], picks,
        event_name="ChainPrizeTest",
        prize_pool=185_000.0,
        is_lan=True,
        event_end=datetime(2026, 6, 1),
        prize_distribution={"1st": 100_000, "2nd": 50_000,
                              "3rd": 25_000, "4th": 10_000},
    )

    by_team = {p["team"]: p["prize"] for p in prizes}
    assert by_team == {
        "A": 100_000.0,
        "B":  50_000.0,
        "C":  25_000.0,
        "D":  10_000.0,
    }
