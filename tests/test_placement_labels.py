"""
Phase 5 §5.2 — per-format placement labellers.

Emit (team, within-stage-bucket-position, bucket-size) per stage so the
caller can combine with compute_place_offsets() to produce absolute
Liquipedia-compatible place labels (e.g. "25th-27th", "7th-8th").
"""

from __future__ import annotations

from data_loaders.bracket_parser import Stage, BracketMatch


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
