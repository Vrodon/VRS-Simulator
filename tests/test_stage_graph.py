"""
Stage graph construction tests (Phase 2 of NEXT_STEPS_BRACKETS.md).

The builder takes:
  * `parsed_stages`  — list[Stage] from bracket_parser (DOM walker)
  * `stage_defs`     — list[StageDef] from format_parser (prose walker)
  * `seeded_teams`   — initial roster of canonical team names

…and populates each Stage's cross-stage fields (`def_name`, `entrants`,
`advance_to`, `eliminations`) by linking DOM stages to prose StageDefs by
heading.

We test through the public `build_stage_graph(...)` interface only — the
heading-matching helper is an implementation detail, so we don't poke at it
directly.
"""

from __future__ import annotations

from data_loaders.bracket_parser import Stage, BracketMatch, _apply_roster_seeds
from data_loaders.format_parser import StageDef, StageEdge, StageEntrant  # noqa: F401
from data_loaders.stage_graph import (
    build_stage_graph,
    compute_stage_advancers,
    apply_cross_stage_cascade,
    seat_cross_stage_r1,
    resolve_for_render,
)


# ── Heading matching ────────────────────────────────────────────────────────────

def test_exact_heading_match_links_def_to_stage():
    """A Stage whose heading equals a StageDef.name picks up that def."""
    stages = [Stage(stage_id="s0", format="SE", display_heading="Playoffs")]
    defs = [StageDef(name="Playoffs", team_count=8, format="SE")]

    build_stage_graph(stages, defs, seeded_teams=[])

    assert stages[0].def_name == "Playoffs"


def test_two_dom_groups_sharing_one_def_emit_one_aggregate_advance_edge():
    """
    Atlanta / CSAC ship two DOM stages "Group A" + "Group B" both linking to
    a single multi-group StageDef whose `advance_to.count` (6 = top-3 ×
    2 groups) is already the aggregate. Playoffs must see exactly ONE
    `advance_from` entrant of count=6, not two-of-six (which would imply 12
    teams entering an 8-team SE bracket).
    """
    grp_def = StageDef(
        name="Group Stage", team_count=16, format="DE",
        n_groups=2, teams_per_group=8,
        advance_to=[StageEdge("Playoffs", 6, "group_rank")],
    )
    play_def = StageDef(name="Playoffs", team_count=6, format="SE_with_bronze")
    stages = [
        Stage(stage_id="s0", format="DE", display_heading="Group A"),
        Stage(stage_id="s1", format="DE", display_heading="Group B"),
        Stage(stage_id="s2", format="SE_with_bronze", display_heading="Playoffs"),
    ]

    build_stage_graph(stages, [grp_def, play_def], seeded_teams=[])

    advance_entrants = [en for en in stages[2].entrants if en.source == "advance_from"]
    assert len(advance_entrants) == 1
    assert advance_entrants[0].count == 6


def test_group_heading_with_parenthetical_suffix_still_links_to_multigroup_def():
    """
    BLAST Rivals tags groups as ``"Group A (GSL-lite)"`` / ``"Group B (GSL-lite)"``.
    The heading matcher should link both to a multi-group StageDef.
    """
    stages = [
        Stage(stage_id="s0", format="Groups", display_heading="Group A (GSL-lite)"),
        Stage(stage_id="s1", format="Groups", display_heading="Group B (GSL-lite)"),
    ]
    defs = [StageDef(name="Group Stage", team_count=8, format="GSL",
                     n_groups=2, teams_per_group=4)]

    build_stage_graph(stages, defs, seeded_teams=[])

    assert stages[0].def_name == "Group Stage"
    assert stages[1].def_name == "Group Stage"


def test_group_a_heading_links_to_multigroup_stage_def():
    """
    BLAST/Atlanta/CSAC: each parallel group is its own DOM Stage with heading
    "Group A" / "Group B", but prose declares a single multi-group StageDef
    "Group Stage" with ``n_groups = 2``. Both DOM stages must link to it.
    """
    stages = [
        Stage(stage_id="s0", format="Groups", display_heading="Group A"),
        Stage(stage_id="s1", format="Groups", display_heading="Group B"),
    ]
    defs = [
        StageDef(name="Group Stage", team_count=8, format="GSL",
                 n_groups=2, teams_per_group=4),
    ]

    build_stage_graph(stages, defs, seeded_teams=[])

    assert stages[0].def_name == "Group Stage"
    assert stages[1].def_name == "Group Stage"


def test_stage_inherits_advance_to_and_eliminations_from_def():
    """
    Once linked, the Stage's `advance_to` and `eliminations` mirror the
    StageDef's so the cascade walker (Phase 4) can follow edges via Stage
    objects without dipping back into prose.
    """
    edge = StageEdge(target_stage="Playoffs", count=8, criterion="top_by_wins")
    sd = StageDef(
        name="Group Stage",
        team_count=16,
        format="Swiss",
        advance_to=[edge],
        eliminations=[("9th-11th", 3), ("12th-14th", 3), ("15th-16th", 2)],
    )
    stages = [Stage(stage_id="s0", format="Swiss", display_heading="Group Stage")]

    build_stage_graph(stages, [sd], seeded_teams=[])

    assert [(e.target_stage, e.count) for e in stages[0].advance_to] == [("Playoffs", 8)]
    assert stages[0].eliminations == [("9th-11th", 3), ("12th-14th", 3), ("15th-16th", 2)]


def test_first_stage_roster_seeded_from_initial_roster():
    """
    The first stage's roster is the discovery list (`seeded_teams`). Until the
    teamcard scraper lands (Phase 2.3) this is the only entrant source we have.
    """
    sd = StageDef(name="Group Stage", team_count=16, format="Swiss")
    stages = [Stage(stage_id="s0", format="Swiss", display_heading="Group Stage")]
    seeds = [f"Team{i}" for i in range(16)]

    build_stage_graph(stages, [sd], seeded_teams=seeds)

    assert stages[0].roster == seeds
    assert any(en.source == "initial_roster" for en in stages[0].entrants)
    assert sum(en.count for en in stages[0].entrants) == 16


def test_downstream_stage_entrant_traces_back_to_upstream_advancers():
    """
    A linear two-stage chain (Group Stage → Playoffs): Playoffs' entrants
    list should include an `advance_from("Group Stage", 8)` source whose
    count matches the upstream advance edge.
    """
    grp = StageDef(
        name="Group Stage",
        team_count=16,
        format="Swiss",
        advance_to=[StageEdge("Playoffs", 8, "top_by_wins")],
    )
    play = StageDef(name="Playoffs", team_count=8, format="SE")
    stages = [
        Stage(stage_id="s0", format="Swiss", display_heading="Group Stage"),
        Stage(stage_id="s1", format="SE",    display_heading="Playoffs"),
    ]

    build_stage_graph(stages, [grp, play], seeded_teams=[f"T{i}" for i in range(16)])

    sources = [(en.source, en.upstream_stage, en.count) for en in stages[1].entrants]
    assert ("advance_from", "Group Stage", 8) in sources
    assert sum(en.count for en in stages[1].entrants) == 8


def test_gap_between_advancers_and_team_count_flagged_as_direct_invite():
    """
    Cologne S2: advancers from S1 = 8, but stage holds 16 teams. Until the
    sub-page teamcard scraper lands (Phase 2.3) the gap is a placeholder
    `direct_invite` source whose count = team_count - advancers.
    """
    s1 = StageDef(name="Stage 1", team_count=16, format="Swiss",
                  advance_to=[StageEdge("Stage 2", 8, "top_by_wins")])
    s2 = StageDef(name="Stage 2", team_count=16, format="Swiss",
                  advance_to=[StageEdge("Playoffs", 8, "top_by_wins")])
    stages = [
        Stage(stage_id="s0", format="Swiss", display_heading="Stage 1"),
        Stage(stage_id="s1", format="Swiss", display_heading="Stage 2"),
    ]

    build_stage_graph(stages, [s1, s2], seeded_teams=[f"T{i}" for i in range(16)])

    sources = [(en.source, en.count) for en in stages[1].entrants]
    assert ("advance_from", 8) in sources
    assert ("direct_invite", 8) in sources


def test_sub_page_roster_resolves_direct_invitees_with_concrete_names():
    """
    When the caller supplies a sub-page roster for a downstream stage, the
    placeholder `direct_invite` source becomes a concrete one — invitees are
    everyone in the sub-page roster who wasn't already known upstream
    (initial roster ∪ prior stages' rosters).
    """
    s1 = StageDef(name="Stage 1", team_count=4, format="Swiss",
                  advance_to=[StageEdge("Stage 2", 2, "top_by_wins")])
    s2 = StageDef(name="Stage 2", team_count=4, format="Swiss")
    stages = [
        Stage(stage_id="s0", format="Swiss", display_heading="Stage 1"),
        Stage(stage_id="s1", format="Swiss", display_heading="Stage 2"),
    ]
    sub_rosters = {
        "Stage 2": ["A", "B", "X", "Y"],   # A,B advanced from S1; X,Y invited
    }

    build_stage_graph(
        stages, [s1, s2],
        seeded_teams=["A", "B", "C", "D"],
        sub_page_rosters=sub_rosters,
    )

    invitees = [en for en in stages[1].entrants if en.source == "direct_invite"]
    assert len(invitees) == 1
    assert invitees[0].count == 2
    assert sorted(invitees[0].notes.split(",")) == ["X", "Y"]
    # Stage 2's roster reflects the sub-page teamcards.
    assert stages[1].roster == ["A", "B", "X", "Y"]


def test_collect_sub_page_rosters_walks_stage_n_links_and_calls_fetcher():
    """
    `collect_sub_page_rosters` should discover ``/Stage_N`` anchor hrefs in
    the main page's soup, call the injected fetcher for each, and key the
    output by canonical StageDef name (``"Stage N"``).
    """
    from bs4 import BeautifulSoup
    from data_loaders.stage_graph import collect_sub_page_rosters

    main_slug = "Intel_Extreme_Masters/2026/Cologne"
    html = (
        '<a href="/counterstrike/Intel_Extreme_Masters/2026/Cologne/Stage_1">S1</a>'
        '<a href="/counterstrike/Intel_Extreme_Masters/2026/Cologne/Stage_2">S2</a>'
        '<a href="/counterstrike/Intel_Extreme_Masters/2026/Cologne/Stage_3">S3</a>'
    )
    soup = BeautifulSoup(html, "html.parser")

    fake_rosters = {
        f"{main_slug}/Stage_1": ["A", "B"],
        f"{main_slug}/Stage_2": ["A", "C"],
        f"{main_slug}/Stage_3": ["C", "D"],
    }
    rosters = collect_sub_page_rosters(
        main_slug, soup,
        fetch_roster=lambda slug: fake_rosters.get(slug, []),
    )

    assert rosters == {
        "Stage 1": ["A", "B"],
        "Stage 2": ["A", "C"],
        "Stage 3": ["C", "D"],
    }


def test_per_stage_invitees_override_seeded_teams_and_fill_direct_invitees():
    """
    Major-tier events (IEM Cologne) partition the main-page Participants
    section into "Stage N Invites" h3 blocks. When the caller supplies
    `direct_invitees_by_stage`, the first stage's roster is the Stage 1
    invites (16 teams, NOT the full 32 listed across all sub-headings),
    and downstream stages get concrete invitees (8 each) instead of the
    placeholder fallback.
    """
    s1 = StageDef(name="Stage 1", team_count=16, format="Swiss",
                  advance_to=[StageEdge("Stage 2", 8, "top_by_wins")])
    s2 = StageDef(name="Stage 2", team_count=16, format="Swiss",
                  advance_to=[StageEdge("Stage 3", 8, "top_by_wins")])
    s3 = StageDef(name="Stage 3", team_count=16, format="Swiss")
    stages = [
        Stage(stage_id="s0", format="Swiss", display_heading="Stage 1"),
        Stage(stage_id="s1", format="Swiss", display_heading="Stage 2"),
        Stage(stage_id="s2", format="Swiss", display_heading="Stage 3"),
    ]
    seeded = [f"T{i}" for i in range(32)]   # full 32-team listing across stages
    invites = {
        "Stage 1": [f"T{i}" for i in range(16)],
        "Stage 2": [f"T{i}" for i in range(16, 24)],
        "Stage 3": [f"T{i}" for i in range(24, 32)],
    }

    build_stage_graph(stages, [s1, s2, s3], seeded_teams=seeded,
                      direct_invitees_by_stage=invites)

    # Stage 1 picks the 16 invites, not the 32 full-page list.
    assert stages[0].roster == invites["Stage 1"]
    # Stage 2/3 get concrete invitee names on the direct_invite entrant.
    s2_dinv = [en for en in stages[1].entrants if en.source == "direct_invite"]
    s3_dinv = [en for en in stages[2].entrants if en.source == "direct_invite"]
    assert s2_dinv and s2_dinv[0].count == 8
    assert sorted(s2_dinv[0].notes.split(",")) == sorted(invites["Stage 2"])
    assert s3_dinv and s3_dinv[0].count == 8
    assert sorted(s3_dinv[0].notes.split(",")) == sorted(invites["Stage 3"])
    assert stages[1].direct_invitees == invites["Stage 2"]
    assert stages[2].direct_invitees == invites["Stage 3"]


def test_compute_stage_advancers_swiss_top_by_wins():
    """
    For a Swiss stage, advancers = teams ranked by wins descending. The
    full picks across all rounds drive the W-L count. Ties broken
    alphabetically (Buchholz proxy — refined later).
    """
    # 4-team mini Swiss: 3 rounds. Pairings already placed in R1; we don't
    # exercise the cascade pairing here, just the advancer computation
    # over a fully-resolved match set.
    matches = [
        # R1: A beats B, C beats D
        BracketMatch(match_id="m::SW-R1M0", round_idx=1, slot_idx=0, sub="SW",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="m::SW-R1M1", round_idx=1, slot_idx=1, sub="SW",
                     seed_a="C", seed_b="D"),
        # R2: A beats C, B beats D
        BracketMatch(match_id="m::SW-R2M0", round_idx=2, slot_idx=0, sub="SW",
                     seed_a="A", seed_b="C"),
        BracketMatch(match_id="m::SW-R2M1", round_idx=2, slot_idx=1, sub="SW",
                     seed_a="B", seed_b="D"),
    ]
    stage = Stage(stage_id="m", format="Swiss", display_heading="Swiss",
                  matches=matches)
    picks = {
        "m::SW-R1M0": "A", "m::SW-R1M1": "C",
        "m::SW-R2M0": "A", "m::SW-R2M1": "B",
    }

    # Records:  A=2-0, B=1-1, C=1-1, D=0-2
    # Top 2 by wins → [A, then one of B/C alphabetically → B]
    advancers = compute_stage_advancers(stage, picks, count=2,
                                        criterion="top_by_wins")
    assert advancers == ["A", "B"]


def test_compute_stage_advancers_swiss_gate_blocks_until_threshold_reached():
    """
    CS2 Swiss: a team only ADVANCES when it reaches the win threshold
    (= ceil(rounds/2)). For a 5-round 16-team Swiss with top-8 advance,
    threshold = 3 wins. The cascade must NOT seat downstream Playoffs
    after just R1 picks — too early to know who's in.

    Caller observes: returns [] until ≥count teams have ≥threshold wins.
    """
    matches = []
    teams = [chr(ord("A") + i) for i in range(16)]
    # 5 rounds × 8 matches each = 40 placeholder Swiss matches in DOM.
    for r in range(1, 6):
        for s in range(8):
            matches.append(BracketMatch(
                match_id=f"m::SW-R{r}M{s}", round_idx=r, slot_idx=s, sub="SW",
                seed_a=teams[2 * s] if r == 1 else None,
                seed_b=teams[2 * s + 1] if r == 1 else None,
            ))
    stage = Stage(stage_id="m", format="Swiss", display_heading="Stage 1",
                  matches=matches)
    # Only R1 picked: top-half (even-index teams) wins. No team has 3W yet.
    picks = {f"m::SW-R1M{s}": teams[2 * s] for s in range(8)}

    advancers = compute_stage_advancers(stage, picks, count=8,
                                        criterion="top_by_wins")
    assert advancers == []


def test_compute_stage_advancers_swiss_gate_uses_roster_for_16team_cologne():
    """
    Liquipedia's Swiss matchlist publishes rounds incrementally — Cologne S1
    after R1 picks has DOM `max(round_idx) == 1`. A formula threshold of
    `ceil(1/2) = 1` would misqualify all 8 R1 winners and fire the cascade
    after R1 alone. The gate must look at the stage's roster size (16) and
    apply the CS2 3-3 Swiss threshold of 3W regardless of DOM round count.
    """
    teams = [chr(ord("A") + i) for i in range(16)]
    matches = [
        BracketMatch(match_id=f"m::SW-R1M{s}", round_idx=1, slot_idx=s, sub="SW",
                     seed_a=teams[2 * s], seed_b=teams[2 * s + 1])
        for s in range(8)
    ]
    stage = Stage(stage_id="m", format="Swiss", display_heading="Stage 1",
                  matches=matches, roster=teams)
    picks = {f"m::SW-R1M{s}": teams[2 * s] for s in range(8)}

    advancers = compute_stage_advancers(stage, picks, count=8,
                                        criterion="top_by_wins")
    assert advancers == [], "16-team Swiss must wait for 8 teams at 3W"


def test_compute_stage_advancers_swiss_returns_empty_when_no_picks():
    """No picks yet → no advancers. Caller can short-circuit cascade."""
    stage = Stage(
        stage_id="m", format="Swiss", display_heading="Swiss",
        matches=[BracketMatch(match_id="m::SW-R1M0", round_idx=1,
                              slot_idx=0, sub="SW",
                              seed_a="A", seed_b="B")],
    )
    assert compute_stage_advancers(stage, picks={}, count=1,
                                   criterion="top_by_wins") == []


def test_compute_stage_advancers_se_placement():
    """
    SE bracket: advancers = top finishers by placement order.
      placement(1) = final winner
      placement(2) = final loser
      placement(3-4) = SF losers (position immaterial — both go through)
    """
    # 4-team SE: SF1, SF2, F.
    matches = [
        BracketMatch(match_id="m::R1m1", round_idx=1, slot_idx=0,
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="m::R1m2", round_idx=1, slot_idx=1,
                     seed_a="C", seed_b="D"),
        BracketMatch(match_id="m::R2m1", round_idx=2, slot_idx=0,
                     feeder_a="m::R1m1", feeder_b="m::R1m2"),
    ]
    stage = Stage(stage_id="m", format="SE", display_heading="Bracket",
                  matches=matches)
    # A beats B; C beats D; A beats C → A=1st, C=2nd, [B,D]=3rd-4th
    picks = {"m::R1m1": "A", "m::R1m2": "C", "m::R2m1": "A"}

    top1 = compute_stage_advancers(stage, picks, count=1, criterion="placement")
    top4 = compute_stage_advancers(stage, picks, count=4, criterion="placement")

    assert top1 == ["A"]
    assert top4[0] == "A"
    assert top4[1] == "C"
    assert set(top4[2:]) == {"B", "D"}


def test_seat_cross_stage_r1_8team_se_option_b_seeding():
    """
    8-team SE Playoffs fed by Swiss top-8: R1 pairings = Option B (CS2
    standard) — 1v8 (slot 0), 4v5 (slot 1), 2v7 (slot 2), 3v6 (slot 3).
    Top half (slots 0-1) and bottom half (slots 2-3) avoid #1+#2 meeting
    before the grand final.
    """
    matches = [
        BracketMatch(match_id=f"m::R1m{i}", round_idx=1, slot_idx=i)
        for i in range(4)
    ] + [
        BracketMatch(match_id="m::R2m0", round_idx=2, slot_idx=0,
                     feeder_a="m::R1m0", feeder_b="m::R1m1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        BracketMatch(match_id="m::R2m1", round_idx=2, slot_idx=1,
                     feeder_a="m::R1m2", feeder_b="m::R1m3",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        BracketMatch(match_id="m::R3m0", round_idx=3, slot_idx=0,
                     feeder_a="m::R2m0", feeder_b="m::R2m1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
    ]
    stage = Stage(stage_id="m", format="SE", display_heading="Playoffs",
                  matches=matches,
                  entrants=[StageEntrant(source="advance_from", count=8)])

    seat_cross_stage_r1(stage, roster=list("ABCDEFGH"))

    by_id = {m.match_id: m for m in stage.matches}
    assert (by_id["m::R1m0"].seed_a, by_id["m::R1m0"].seed_b) == ("A", "H")
    assert (by_id["m::R1m1"].seed_a, by_id["m::R1m1"].seed_b) == ("D", "E")
    assert (by_id["m::R1m2"].seed_a, by_id["m::R1m2"].seed_b) == ("B", "G")
    assert (by_id["m::R1m3"].seed_a, by_id["m::R1m3"].seed_b) == ("C", "F")


def test_seat_cross_stage_r1_6team_se_bronze_fan_in_atlanta_seeding():
    """
    Atlanta-style: 2 DE groups × 3 advancers → 6-team SE+bronze. Roster
    arrives concatenated [1A,2A,3A,1B,2B,3B]. Fan-in seeding: top group
    winners (1A, 1B) get SF byes; QFs cross-pair seeds 2A vs 3B and 2B
    vs 3A so group winners can't meet co-group teams before SF.

    Layout (slot-indexed):
      QF1 (R1 slot 0): 2A vs 3B
      QF2 (R1 slot 1): 2B vs 3A
      SF1 (R2 slot 0): bye = 1A   (feeder_a = QF1.winner)
      SF2 (R2 slot 1): bye = 1B   (feeder_a = QF2.winner)
    """
    matches = [
        # QFs
        BracketMatch(match_id="m::R1m0", round_idx=1, slot_idx=0),
        BracketMatch(match_id="m::R1m1", round_idx=1, slot_idx=1),
        # SFs (top-half SF gets bye on side b — feeder_a is QF winner).
        BracketMatch(match_id="m::R2m0", round_idx=2, slot_idx=0,
                     feeder_a="m::R1m0", feeder_a_kind="winner"),
        BracketMatch(match_id="m::R2m1", round_idx=2, slot_idx=1,
                     feeder_a="m::R1m1", feeder_a_kind="winner"),
        # Final
        BracketMatch(match_id="m::R3m0", round_idx=3, slot_idx=0,
                     feeder_a="m::R2m0", feeder_b="m::R2m1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        # Bronze
        BracketMatch(match_id="m::B", round_idx=3, slot_idx=0, is_bronze=True,
                     feeder_a="m::R2m0", feeder_b="m::R2m1",
                     feeder_a_kind="loser", feeder_b_kind="loser"),
    ]
    stage = Stage(stage_id="m", format="SE_with_bronze", display_heading="Playoffs",
                  matches=matches,
                  entrants=[StageEntrant(source="advance_from", count=6)])

    seat_cross_stage_r1(stage, roster=["1A", "2A", "3A", "1B", "2B", "3B"])

    by_id = {m.match_id: m for m in stage.matches}
    assert (by_id["m::R1m0"].seed_a, by_id["m::R1m0"].seed_b) == ("2A", "3B")
    assert (by_id["m::R1m1"].seed_a, by_id["m::R1m1"].seed_b) == ("2B", "3A")
    # SF byes — top-of-bracket gets group-A winner, bottom-of-bracket B.
    assert by_id["m::R2m0"].seed_b == "1A"
    assert by_id["m::R2m1"].seed_b == "1B"


def test_seat_cross_stage_r1_4team_se_uses_1v4_2v3():
    """
    4-team SE Playoffs fed by Swiss top-4 (or DE top-4): R1 pairings should
    be 1v4 (slot 0) and 2v3 (slot 1). Roster order = seed order.
    """
    matches = [
        BracketMatch(match_id="m::R1m0", round_idx=1, slot_idx=0),
        BracketMatch(match_id="m::R1m1", round_idx=1, slot_idx=1),
        BracketMatch(match_id="m::R2m0", round_idx=2, slot_idx=0,
                     feeder_a="m::R1m0", feeder_b="m::R1m1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
    ]
    stage = Stage(stage_id="m", format="SE", display_heading="Playoffs",
                  matches=matches,
                  entrants=[StageEntrant(source="advance_from", count=4)])

    seat_cross_stage_r1(stage, roster=["A", "B", "C", "D"])

    r1m0 = next(m for m in stage.matches if m.match_id == "m::R1m0")
    r1m1 = next(m for m in stage.matches if m.match_id == "m::R1m1")
    assert (r1m0.seed_a, r1m0.seed_b) == ("A", "D")
    assert (r1m1.seed_a, r1m1.seed_b) == ("B", "C")


def test_compute_stage_advancers_group_rank_falls_through_to_placement_for_de():
    """
    Atlanta-style: format_parser emits criterion=`group_rank` for the
    "DE Group Stage" (because the prose calls it a group). The DOM
    matches use sub="UB"/"LB" though — so the walker must dispatch on
    *stage format*, not just criterion string. group_rank + DE matches
    → walk the bracket via placement; group_rank + RR matchlist → use
    sub="GR" wins/losses.
    """
    matches = [
        BracketMatch(match_id="m::UB-R1m0", round_idx=1, slot_idx=0, sub="UB",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="m::UB-R1m1", round_idx=1, slot_idx=1, sub="UB",
                     seed_a="C", seed_b="D"),
        BracketMatch(match_id="m::UB-R2m0", round_idx=2, slot_idx=0, sub="UB",
                     feeder_a="m::UB-R1m0", feeder_b="m::UB-R1m1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        BracketMatch(match_id="m::LB-R1m0", round_idx=1, slot_idx=0, sub="LB",
                     feeder_a="m::UB-R1m0", feeder_b="m::UB-R1m1",
                     feeder_a_kind="loser", feeder_b_kind="loser"),
    ]
    stage = Stage(stage_id="m", format="DE", display_heading="Group A",
                  matches=matches)
    picks = {"m::UB-R1m0": "A", "m::UB-R1m1": "C", "m::UB-R2m0": "A",
             "m::LB-R1m0": "D"}

    top3 = compute_stage_advancers(stage, picks, count=3, criterion="group_rank")
    assert top3 == ["A", "C", "D"]


def test_compute_stage_advancers_de_placement_walks_lb_for_third():
    """
    DE_compact (Atlanta-style 8-team group, no GF): UB-final winner = 1st,
    UB-final loser = 2nd, LB-final winner = 3rd. UB-final loser does NOT
    drop to LB. Walker must reach LB tree to identify 3rd, not return a
    UB-R1 loser as the third advancer.

    Minimal 4-team DE shape (UB-R1×2 → UB-final, LB-R1 absorbs UB-R1
    losers → LB-final winner = 3rd):
    """
    matches = [
        # UB
        BracketMatch(match_id="m::UB-R1m0", round_idx=1, slot_idx=0, sub="UB",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="m::UB-R1m1", round_idx=1, slot_idx=1, sub="UB",
                     seed_a="C", seed_b="D"),
        BracketMatch(match_id="m::UB-R2m0", round_idx=2, slot_idx=0, sub="UB",
                     feeder_a="m::UB-R1m0", feeder_b="m::UB-R1m1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        # LB-R1 = UB-R1 losers paired
        BracketMatch(match_id="m::LB-R1m0", round_idx=1, slot_idx=0, sub="LB",
                     feeder_a="m::UB-R1m0", feeder_b="m::UB-R1m1",
                     feeder_a_kind="loser", feeder_b_kind="loser"),
    ]
    stage = Stage(stage_id="m", format="DE", display_heading="Group A",
                  matches=matches)
    # A>B, C>D, A>C → A=1st, C=2nd. UB-R1 losers = B, D.
    # LB pick: D>B → D=3rd, B=4th.
    picks = {
        "m::UB-R1m0": "A",
        "m::UB-R1m1": "C",
        "m::UB-R2m0": "A",
        "m::LB-R1m0": "D",
    }

    top3 = compute_stage_advancers(stage, picks, count=3, criterion="placement")
    assert top3 == ["A", "C", "D"]


def test_compute_stage_advancers_group_rank_round_robin():
    """
    Round-robin group: advancers = top-N by W-L. 4-team RR has 6 matches.
    Each team plays every other team once.
    """
    # Matches are matchlist-style with `sub == "GR"` and seeds populated.
    matches = [
        BracketMatch(match_id="m::GR-r1m0", round_idx=1, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="m::GR-r1m1", round_idx=1, slot_idx=1, sub="GR",
                     seed_a="C", seed_b="D"),
        BracketMatch(match_id="m::GR-r2m0", round_idx=2, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="C"),
        BracketMatch(match_id="m::GR-r2m1", round_idx=2, slot_idx=1, sub="GR",
                     seed_a="B", seed_b="D"),
        BracketMatch(match_id="m::GR-r3m0", round_idx=3, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="D"),
        BracketMatch(match_id="m::GR-r3m1", round_idx=3, slot_idx=1, sub="GR",
                     seed_a="B", seed_b="C"),
    ]
    stage = Stage(stage_id="m", format="Groups", display_heading="Group A",
                  matches=matches)
    # A wins all 3 (3-0). B beats D, loses to A and C (1-2).
    # C beats B, loses to A and D (1-2). D beats C, loses to A and B (1-2).
    # Wait — let me design so a clear top-2 emerges.
    # A wins all 3. B wins 2 (beats C, D). C wins 1 (beats D). D wins 0.
    picks = {
        "m::GR-r1m0": "A", "m::GR-r1m1": "C",
        "m::GR-r2m0": "A", "m::GR-r2m1": "B",
        "m::GR-r3m0": "A", "m::GR-r3m1": "B",
    }
    # Records: A=3-0, B=2-1, C=1-2, D=0-3 → top-2 = [A, B]

    advancers = compute_stage_advancers(stage, picks, count=2,
                                        criterion="group_rank")
    assert advancers == ["A", "B"]


def test_apply_cross_stage_cascade_linear_swiss_to_se():
    """
    Smallest cascade: 4-team Swiss → 2-team SE. With every Swiss match
    picked, the cascade computes top-2 by wins and writes them onto the
    Playoffs stage's roster so the manual-pick dropdown can place them.
    """
    grp = Stage(
        stage_id="s0", format="Swiss", display_heading="Group Stage",
        matches=[
            BracketMatch(match_id="s0::SW-R1M0", round_idx=1, slot_idx=0,
                         sub="SW", seed_a="A", seed_b="B"),
            BracketMatch(match_id="s0::SW-R1M1", round_idx=1, slot_idx=1,
                         sub="SW", seed_a="C", seed_b="D"),
            BracketMatch(match_id="s0::SW-R2M0", round_idx=2, slot_idx=0,
                         sub="SW", seed_a="A", seed_b="C"),
            BracketMatch(match_id="s0::SW-R2M1", round_idx=2, slot_idx=1,
                         sub="SW", seed_a="B", seed_b="D"),
        ],
        def_name="Group Stage",
        roster=["A", "B", "C", "D"],
        entrants=[StageEntrant(source="initial_roster", count=4)],
        advance_to=[StageEdge("Playoffs", 2, "top_by_wins")],
    )
    play = Stage(
        stage_id="s1", format="SE", display_heading="Playoffs",
        def_name="Playoffs",
        entrants=[StageEntrant(source="advance_from",
                                upstream_stage="Group Stage",
                                count=2, criterion="top_by_wins")],
    )
    picks = {
        "s0::SW-R1M0": "A", "s0::SW-R1M1": "C",
        "s0::SW-R2M0": "A", "s0::SW-R2M1": "B",
    }

    rosters = apply_cross_stage_cascade([grp, play], picks)

    assert rosters["Playoffs"] == ["A", "B"]


def test_apply_cross_stage_cascade_empty_picks_keeps_invitees_only():
    """
    With no picks made yet, upstream advancers can't be ranked, so the
    downstream roster contains only the concrete direct invitees (if any).
    Stages with neither picks nor invitees come back empty — caller leaves
    every R1 slot TBD via the manual-pick dropdown.
    """
    s1 = Stage(
        stage_id="s0", format="Swiss", display_heading="Stage 1",
        matches=[BracketMatch(match_id="s0::SW-R1M0", round_idx=1,
                              slot_idx=0, sub="SW",
                              seed_a="A", seed_b="B")],
        def_name="Stage 1", roster=["A", "B"],
        entrants=[StageEntrant(source="initial_roster", count=2)],
        advance_to=[StageEdge("Stage 2", 1, "top_by_wins")],
    )
    s2 = Stage(
        stage_id="s1", format="Swiss", display_heading="Stage 2",
        def_name="Stage 2",
        entrants=[
            StageEntrant(source="advance_from", upstream_stage="Stage 1",
                         count=1, criterion="top_by_wins"),
            StageEntrant(source="direct_invite", count=1, notes="X"),
        ],
        direct_invitees=["X"],
    )

    rosters = apply_cross_stage_cascade([s1, s2], picks={})

    # Empty picks → no advancers. Only direct invitees survive.
    assert rosters["Stage 2"] == ["X"]


def test_apply_cross_stage_cascade_concatenates_direct_invitees():
    """
    Cologne S2 shape: 8 advance from S1 + 8 direct invites entering at S2.
    Cascade returns S2 roster = upstream advancers (in rank order) + concrete
    invitee names (in given order). Invitees never duplicate advancers.
    """
    s1_matches = []
    teams = [f"T{i}" for i in range(4)]
    # 4-team Swiss with 2 rounds → top-2 advance.
    s1_matches.append(BracketMatch(match_id="s0::SW-R1M0", round_idx=1,
                                    slot_idx=0, sub="SW",
                                    seed_a="T0", seed_b="T1"))
    s1_matches.append(BracketMatch(match_id="s0::SW-R1M1", round_idx=1,
                                    slot_idx=1, sub="SW",
                                    seed_a="T2", seed_b="T3"))
    s1_matches.append(BracketMatch(match_id="s0::SW-R2M0", round_idx=2,
                                    slot_idx=0, sub="SW",
                                    seed_a="T0", seed_b="T2"))
    s1_matches.append(BracketMatch(match_id="s0::SW-R2M1", round_idx=2,
                                    slot_idx=1, sub="SW",
                                    seed_a="T1", seed_b="T3"))
    s1 = Stage(
        stage_id="s0", format="Swiss", display_heading="Stage 1",
        matches=s1_matches, def_name="Stage 1", roster=teams,
        entrants=[StageEntrant(source="initial_roster", count=4)],
        advance_to=[StageEdge("Stage 2", 2, "top_by_wins")],
    )
    s2 = Stage(
        stage_id="s1", format="Swiss", display_heading="Stage 2",
        def_name="Stage 2",
        entrants=[
            StageEntrant(source="advance_from", upstream_stage="Stage 1",
                         count=2, criterion="top_by_wins"),
            StageEntrant(source="direct_invite", count=2, notes="X,Y"),
        ],
        direct_invitees=["X", "Y"],
    )
    picks = {
        "s0::SW-R1M0": "T0", "s0::SW-R1M1": "T2",
        "s0::SW-R2M0": "T0", "s0::SW-R2M1": "T1",
    }

    rosters = apply_cross_stage_cascade([s1, s2], picks)

    # T0 wins both → 1st; T1 + T2 each 1-1 → alphabetical breaks to T1.
    assert rosters["Stage 2"] == ["T0", "T1", "X", "Y"]


def test_apply_cross_stage_cascade_fan_in_concatenates_groups():
    """
    BLAST Rivals shape: two GSL-lite groups each contributing 2 teams to a
    4-team SE Playoffs. The cascade concatenates Group A's advancers
    followed by Group B's advancers (DOM order). Slot-placement (1st-A vs
    2nd-B etc.) is deferred to the manual-pick dropdown.
    """
    def gsl_group(stage_id: str, heading: str, teams: tuple[str, str, str, str]):
        a, b, c, d = teams
        return Stage(
            stage_id=stage_id, format="Groups", display_heading=heading,
            matches=[
                BracketMatch(match_id=f"{stage_id}::GR-O0", round_idx=1,
                             slot_idx=0, sub="GR", seed_a=a, seed_b=b),
                BracketMatch(match_id=f"{stage_id}::GR-O1", round_idx=1,
                             slot_idx=1, sub="GR", seed_a=c, seed_b=d),
                BracketMatch(match_id=f"{stage_id}::GR-WM", round_idx=2,
                             slot_idx=0, sub="GR",
                             feeder_a=f"{stage_id}::GR-O0",
                             feeder_b=f"{stage_id}::GR-O1",
                             feeder_a_kind="winner", feeder_b_kind="winner"),
                BracketMatch(match_id=f"{stage_id}::GR-EM", round_idx=2,
                             slot_idx=1, sub="GR",
                             feeder_a=f"{stage_id}::GR-O0",
                             feeder_b=f"{stage_id}::GR-O1",
                             feeder_a_kind="loser", feeder_b_kind="loser"),
            ],
            def_name="Group Stage",
            roster=list(teams),
            entrants=[StageEntrant(source="initial_roster", count=4)],
            advance_to=[StageEdge("Playoffs", 4, "gsl_rank")],
        )

    grp_a = gsl_group("s0", "Group A (GSL-lite)", ("A1", "A2", "A3", "A4"))
    grp_b = gsl_group("s1", "Group B (GSL-lite)", ("B1", "B2", "B3", "B4"))
    # build_stage_graph emits ONE aggregate advance_from for parallel
    # groups sharing a multi-group StageDef (count = top-N × n_groups).
    play = Stage(
        stage_id="s2", format="SE", display_heading="Playoffs",
        def_name="Playoffs",
        entrants=[
            StageEntrant(source="advance_from", upstream_stage="Group Stage",
                         count=4, criterion="gsl_rank"),
        ],
    )
    picks = {
        # Group A: A1 beats A2; A3 beats A4; A1 beats A3 (WM); A2 beats A4 (EM)
        "s0::GR-O0": "A1", "s0::GR-O1": "A3",
        "s0::GR-WM": "A1", "s0::GR-EM": "A2",
        # Group B: B1 beats B2; B3 beats B4; B3 beats B1 (WM); B2 beats B4 (EM)
        "s1::GR-O0": "B1", "s1::GR-O1": "B3",
        "s1::GR-WM": "B3", "s1::GR-EM": "B2",
    }

    rosters = apply_cross_stage_cascade([grp_a, grp_b, play], picks)

    # A's top-2 = [A1 (WM winner), A3 (WM loser)]; B's top-2 = [B3, B1].
    # Cascade walks DOM order so Playoffs roster = A-first then B.
    assert rosters["Playoffs"][:2] == ["A1", "A3"]
    assert rosters["Playoffs"][2:] == ["B3", "B1"]


def test_compute_stage_advancers_gsl_lite_rank():
    """
    GSL-lite (BLAST Rivals): 4 teams, 4 matches. Top 2 advance via Opening
    Match wins; the Winners' Match decides 1st-vs-2nd; the Elimination Match
    decides 3rd-vs-4th. Match shape mirrors ``_parse_groups_stage``'s
    GSL-lite branch: opening matches with seeds, WM/EM with feeders.

      1st-in-group = WM winner
      2nd-in-group = WM loser
      3rd-in-group = EM winner
      4th-in-group = EM loser
    """
    matches = [
        BracketMatch(match_id="m::GR-O0", round_idx=1, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="m::GR-O1", round_idx=1, slot_idx=1, sub="GR",
                     seed_a="C", seed_b="D"),
        BracketMatch(match_id="m::GR-WM", round_idx=2, slot_idx=0, sub="GR",
                     feeder_a="m::GR-O0", feeder_b="m::GR-O1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        BracketMatch(match_id="m::GR-EM", round_idx=2, slot_idx=1, sub="GR",
                     feeder_a="m::GR-O0", feeder_b="m::GR-O1",
                     feeder_a_kind="loser", feeder_b_kind="loser"),
    ]
    stage = Stage(stage_id="m", format="Groups",
                  display_heading="Group A (GSL-lite)", matches=matches)
    # A beats B; C beats D → openings settle. WM: A beats C. EM: B beats D.
    picks = {
        "m::GR-O0": "A", "m::GR-O1": "C",
        "m::GR-WM": "A", "m::GR-EM": "B",
    }

    top2 = compute_stage_advancers(stage, picks, count=2, criterion="gsl_rank")
    top4 = compute_stage_advancers(stage, picks, count=4, criterion="gsl_rank")

    assert top2 == ["A", "C"]
    assert top4 == ["A", "C", "B", "D"]


def test_compute_stage_advancers_gsl_full_rank_uses_decider():
    """
    Full GSL: 5 matches. Decider settles 2nd-vs-3rd between WM-loser and
    EM-winner. Top-2 advancers = [WM winner, Decider winner].
    """
    matches = [
        BracketMatch(match_id="m::GR-O0", round_idx=1, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="m::GR-O1", round_idx=1, slot_idx=1, sub="GR",
                     seed_a="C", seed_b="D"),
        BracketMatch(match_id="m::GR-WM", round_idx=2, slot_idx=0, sub="GR",
                     feeder_a="m::GR-O0", feeder_b="m::GR-O1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        BracketMatch(match_id="m::GR-EM", round_idx=2, slot_idx=1, sub="GR",
                     feeder_a="m::GR-O0", feeder_b="m::GR-O1",
                     feeder_a_kind="loser", feeder_b_kind="loser"),
        BracketMatch(match_id="m::GR-DM", round_idx=3, slot_idx=0, sub="GR",
                     feeder_a="m::GR-WM", feeder_b="m::GR-EM",
                     feeder_a_kind="loser", feeder_b_kind="winner"),
    ]
    stage = Stage(stage_id="m", format="Groups",
                  display_heading="Group A (GSL)", matches=matches)
    # A beats B; C beats D; A beats C (WM); B beats D (EM); B beats C (DM).
    # Ranks: 1st=A (WM winner), 2nd=B (DM winner), 3rd=C (DM loser),
    #        4th=D (EM loser).
    picks = {
        "m::GR-O0": "A", "m::GR-O1": "C",
        "m::GR-WM": "A", "m::GR-EM": "B",
        "m::GR-DM": "B",
    }

    top2 = compute_stage_advancers(stage, picks, count=2, criterion="gsl_rank")
    assert top2 == ["A", "B"]


def test_compute_stage_advancers_gsl_returns_empty_until_wm_picked():
    """
    Until the Winners' Match is decided we have no 1st-place answer, so the
    cascade should treat the group as not-yet-resolved and leave downstream
    slots TBD. Returns ``[]`` (caller short-circuits) — same contract as the
    Swiss criterion when no picks exist.
    """
    matches = [
        BracketMatch(match_id="m::GR-O0", round_idx=1, slot_idx=0, sub="GR",
                     seed_a="A", seed_b="B"),
        BracketMatch(match_id="m::GR-O1", round_idx=1, slot_idx=1, sub="GR",
                     seed_a="C", seed_b="D"),
        BracketMatch(match_id="m::GR-WM", round_idx=2, slot_idx=0, sub="GR",
                     feeder_a="m::GR-O0", feeder_b="m::GR-O1",
                     feeder_a_kind="winner", feeder_b_kind="winner"),
        BracketMatch(match_id="m::GR-EM", round_idx=2, slot_idx=1, sub="GR",
                     feeder_a="m::GR-O0", feeder_b="m::GR-O1",
                     feeder_a_kind="loser", feeder_b_kind="loser"),
    ]
    stage = Stage(stage_id="m", format="Groups",
                  display_heading="Group A (GSL-lite)", matches=matches)
    # Openings picked but WM still TBD.
    picks = {"m::GR-O0": "A", "m::GR-O1": "C"}

    assert compute_stage_advancers(stage, picks, count=2,
                                   criterion="gsl_rank") == []


def test_apply_roster_seeds_does_not_pair_order_r1_from_roster():
    """
    Phase 3 design (per user feedback on Atlanta Group A): when Liquipedia
    publishes a bracket *structure* but hasn't placed teams yet, R1 must
    stay TBD so the manual-pick dropdown can drive each slot. Auto pair-
    ordering by roster order is misleading because the real first-round
    pairings will come from the tournament organisers, not from seed order.
    """
    matches = [
        BracketMatch(match_id="s0::R1m1", round_idx=1, slot_idx=0),
        BracketMatch(match_id="s0::R1m2", round_idx=1, slot_idx=1),
    ]
    _apply_roster_seeds(matches, ["A", "B", "C", "D"])

    assert matches[0].seed_a is None and matches[0].seed_b is None
    assert matches[1].seed_a is None and matches[1].seed_b is None


def test_apply_roster_seeds_still_fills_bye_slots_in_non_power_of_two_brackets():
    """
    Bye-slot fill stays in scope: 6-team SE+bronze has 2 R1 matches and 2
    R2 (SF) bye slots that no upstream R1 feeds. Without bye-fill the SF
    is unreachable. Top seeds populate the byes; the manual dropdown can
    still override.
    """
    matches = [
        BracketMatch(match_id="s::R1m1", round_idx=1, slot_idx=0),
        BracketMatch(match_id="s::R1m2", round_idx=1, slot_idx=1),
        # SF1: bye on side a, fed by R1m1 on side b
        BracketMatch(match_id="s::R2m1", round_idx=2, slot_idx=0,
                     feeder_b="s::R1m1"),
        # SF2: bye on side a, fed by R1m2 on side b
        BracketMatch(match_id="s::R2m2", round_idx=2, slot_idx=1,
                     feeder_b="s::R1m2"),
    ]
    _apply_roster_seeds(matches, ["TopA", "TopB", "T3", "T4", "T5", "T6"])

    # R1 still TBD — manual-pick dropdown owns these.
    assert matches[0].seed_a is None and matches[0].seed_b is None
    # Byes filled in seed order on the open side (a).
    assert matches[2].seed_a == "TopA"
    assert matches[3].seed_a == "TopB"


def test_subpage_heading_with_emdash_suffix_matches_stage_def():
    """
    Cologne-style sub-page tagging — bracket_parser prefixes ``Stage N`` and
    appends ``— <heading>`` for sub-page stages — should still resolve to the
    matching StageDef by stripping the suffix.
    """
    stages = [Stage(stage_id="s0", format="Swiss",
                    display_heading="Stage 1 — Swiss Stage")]
    defs = [StageDef(name="Stage 1", team_count=16, format="Swiss")]

    build_stage_graph(stages, defs, seeded_teams=[])

    assert stages[0].def_name == "Stage 1"



def test_resolve_for_render_overlays_manual_seeds_and_does_not_mutate_input():
    """
    Phase 5 §5.1 tracer bullet: `resolve_for_render` is the single seam
    that takes cached `parsed_stages` + current picks/manual_seeds and
    returns a deep-copied "realised" stages list with overlays applied.
    Both the fragment render and engine emission call this seam — same
    input must produce same output regardless of caller.

    Manual seeds overlay onto match seed_a/seed_b. Inputs must not
    mutate (cached parsed_stages is shared across reruns + sessions).
    """
    matches = [
        BracketMatch(match_id="m::R1m0", round_idx=1, slot_idx=0),
    ]
    stage = Stage(stage_id="m", format="SE", display_heading="Playoffs",
                  matches=matches)

    realised = resolve_for_render(
        [stage],
        picks={},
        manual_seeds={"m::R1m0": {"a": "FaZe", "b": "Spirit"}},
        snapshot_standings=None,
    )

    assert realised[0].matches[0].seed_a == "FaZe"
    assert realised[0].matches[0].seed_b == "Spirit"
    # Caller's parsed_stages must not mutate.
    assert stage.matches[0].seed_a is None
    assert stage.matches[0].seed_b is None



def test_resolve_for_render_seats_downstream_r1_from_cascade_roster():
    """
    Phase 5 §5.1 — second tracer slice: cascade output flows through
    resolve_for_render and seats downstream R1 via Option B map.
    Linear SE → SE chain (count=4 placement). After full upstream picks,
    the helper computes cascade roster and seats Playoffs R1 1v4 / 2v3.
    """
    grp = StageDef(name="Group", team_count=4, format="SE",
                   advance_to=[StageEdge("Playoffs", 4, "placement")])
    play = StageDef(name="Playoffs", team_count=4, format="SE")

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

    build_stage_graph([grp_stage, play_stage], [grp, play],
                      seeded_teams=["A", "B", "C", "D"])

    # Full picks: A>B, C>D, A>C. Placement order = [A, C, B, D].
    picks = {"g::R1m0": "A", "g::R1m1": "C", "g::R2m0": "A"}

    realised = resolve_for_render(
        [grp_stage, play_stage], picks,
        manual_seeds={}, snapshot_standings=None,
    )

    p_r1 = sorted([m for m in realised[1].matches if m.round_idx == 1],
                  key=lambda x: x.slot_idx)
    # 4-team SE Option B map: (1,4), (2,3) → A vs D, C vs B.
    assert (p_r1[0].seed_a, p_r1[0].seed_b) == ("A", "D")
    assert (p_r1[1].seed_a, p_r1[1].seed_b) == ("C", "B")



def test_compute_place_offsets_linear_swiss_chain_cologne():
    """
    Phase 5 §5.4 — place_offset via topological walk of the stage graph.
    Cologne linear chain: S1 → S2 → S3 → Playoffs (each Swiss advances 8;
    Playoffs is terminal SE 8-team).

    Expected absolute place ranges per stage's dropouts:
      Playoffs: places 1-8     (offset = 0,  terminal)
      S3:       places 9-16    (offset = 8)
      S2:       places 17-24   (offset = 16)
      S1:       places 25-32   (offset = 24)

    Verified against Liquipedia's published Cologne prize_distribution.
    """
    from data_loaders.stage_graph import compute_place_offsets

    s1 = StageDef(name="Stage 1", team_count=16, format="Swiss",
                  advance_to=[StageEdge("Stage 2", 8, "top_by_wins")])
    s2 = StageDef(name="Stage 2", team_count=16, format="Swiss",
                  advance_to=[StageEdge("Stage 3", 8, "top_by_wins")])
    s3 = StageDef(name="Stage 3", team_count=16, format="Swiss",
                  advance_to=[StageEdge("Playoffs", 8, "top_by_wins")])
    playoffs = StageDef(name="Playoffs", team_count=8, format="SE")

    offsets = compute_place_offsets([s1, s2, s3, playoffs])

    assert offsets["Playoffs"] == 0
    assert offsets["Stage 3"] == 8
    assert offsets["Stage 2"] == 16
    assert offsets["Stage 1"] == 24



def test_compute_place_offsets_fan_in_atlanta():
    """
    Atlanta fan-in: 2 parallel DE groups (def_name="Group Stage",
    team_count=16 aggregate) feed Playoffs (6-team SE+bronze).

    Expected:
      Playoffs:    offset=0,  dropouts span places 1-6
      Group Stage: offset=6,  dropouts (10 teams across 2 groups) → places 7-16

    Verified against Liquipedia's published Atlanta prize_distribution
    (1st-6th in Playoffs; 7th-8th / 9th-12th / 13th-16th aggregated
    across both groups).
    """
    from data_loaders.stage_graph import compute_place_offsets

    grp = StageDef(name="Group Stage", team_count=16, format="DE",
                   n_groups=2, teams_per_group=8,
                   advance_to=[StageEdge("Playoffs", 6, "group_rank")])
    playoffs = StageDef(name="Playoffs", team_count=6, format="SE_with_bronze")

    offsets = compute_place_offsets([grp, playoffs])

    assert offsets["Playoffs"] == 0
    assert offsets["Group Stage"] == 6
