"""
Stage Graph Construction
========================
Phase 2 of ``NEXT_STEPS_BRACKETS.md``: link DOM-derived ``Stage`` objects
(from ``bracket_parser``) to prose-derived ``StageDef`` entries (from
``format_parser``) by heading, then populate each ``Stage`` with
cross-stage flow fields (``entrants``, ``advance_to``, ``eliminations``)
and per-stage rosters.

Public entry point
------------------
    build_stage_graph(parsed_stages, stage_defs, seeded_teams) -> None

Mutates each ``Stage`` in-place.
"""

from __future__ import annotations

import re

from typing import Callable

from .bracket_parser import Stage, BracketMatch, _discover_sub_stage_slugs
from .format_parser import StageDef, StageEntrant


# Bracket parser tags Major sub-page stages as ``"Stage N — <inner heading>"``.
# Strip the em-dash suffix so the leading ``Stage N`` part can match a StageDef.
_SUBPAGE_SUFFIX_RE = re.compile(r"\s*[—–-]\s*.+$")
_GROUP_HEADING_RE  = re.compile(r"^\s*Group\s+\S+", re.IGNORECASE)


def _candidate_keys(display_heading: str) -> list[str]:
    """Return progressively-stripped lookup keys for matching a StageDef."""
    raw = (display_heading or "").strip()
    keys: list[str] = []
    if raw:
        keys.append(raw)
    stripped = _SUBPAGE_SUFFIX_RE.sub("", raw).strip()
    if stripped and stripped != raw:
        keys.append(stripped)
    return keys


def _resolve_def(stage: Stage, defs_by_name: dict[str, StageDef],
                 stage_defs: list[StageDef]) -> StageDef | None:
    """Match a parsed Stage to its StageDef. Returns None if nothing fits."""
    for key in _candidate_keys(stage.display_heading):
        d = defs_by_name.get(key)
        if d is not None:
            return d

    # "Group A" / "Group B" headings → first multi-group StageDef in the prose.
    if _GROUP_HEADING_RE.match(stage.display_heading or ""):
        for d in stage_defs:
            if d.n_groups > 1:
                return d
    return None


def _placement_advancers(stage: Stage, picks: dict[str, str],
                          count: int) -> list[str]:
    """
    Placement-ordered advancers for SE / SE+bronze / DE main tree.

    Walks bracket matches deepest-round first. Round losers contribute
    placement bands: final loser = 2nd; SF losers = 3rd-4th; QF losers =
    5th-8th; etc. Bronze match (SE+bronze only) splits 3rd-4th when picked.
    """
    if not stage.matches:
        return []

    by_id = {m.match_id: m for m in stage.matches}

    def _winner(m):
        return m.played_winner or picks.get(m.match_id)

    def _team_a(m):
        # R1 leaves carry seeds; later rounds resolve via feeders.
        if m.round_idx == 1 and not m.is_bronze and m.sub in ("", "UB"):
            return m.seed_a
        if m.feeder_a:
            up = by_id.get(m.feeder_a)
            if up is None:
                return m.seed_a
            return _resolve_match_team(up, picks, by_id, kind=m.feeder_a_kind)
        return m.seed_a

    def _team_b(m):
        if m.round_idx == 1 and not m.is_bronze and m.sub in ("", "UB"):
            return m.seed_b
        if m.feeder_b:
            up = by_id.get(m.feeder_b)
            if up is None:
                return m.seed_b
            return _resolve_match_team(up, picks, by_id, kind=m.feeder_b_kind)
        return m.seed_b

    main = [m for m in stage.matches if not m.is_bronze
            and m.sub in ("", "UB", "GF")]
    if not main:
        return []
    final_round = max(m.round_idx for m in main)
    final = next((m for m in main if m.round_idx == final_round), None)
    if final is None:
        return []

    winner = _winner(final)
    if winner is None:
        return []

    a, b = _team_a(final), _team_b(final)
    runner = b if winner == a else (a if winner == b else None)

    placed: list[str] = [winner]
    if runner is not None and len(placed) < count:
        placed.append(runner)

    # DE: earlier-round UB losers drop to LB — their final placement is set by
    # the LB walk, not by their UB exit round. Walk LB deepest-first for 3rd
    # onward. Pure SE (no LB) falls through to the existing UB-loser logic.
    lb_matches = [m for m in stage.matches if m.sub == "LB" and not m.is_bronze]
    if lb_matches:
        lb_max = max(m.round_idx for m in lb_matches)
        for r in range(lb_max, 0, -1):
            if len(placed) >= count:
                break
            round_matches = sorted(
                [m for m in lb_matches if m.round_idx == r],
                key=lambda x: x.slot_idx,
            )
            for m in round_matches:
                if len(placed) >= count:
                    break
                w = _winner(m)
                if w is None:
                    continue
                ma, mb = _team_a(m), _team_b(m)
                loser = mb if w == ma else (ma if w == mb else None)
                # Deepest LB round: winner = next placement (e.g. 3rd), loser
                # = the one after. Earlier LB rounds: only losers contribute
                # (winners advance to deeper LB and get placed there).
                if r == lb_max:
                    if w not in placed and len(placed) < count:
                        placed.append(w)
                if loser is not None and loser not in placed and len(placed) < count:
                    placed.append(loser)
        return placed[:count]

    # Earlier rounds, deepest-first → losers go into next placement band.
    for r in range(final_round - 1, 0, -1):
        if len(placed) >= count:
            break
        round_matches = sorted(
            [m for m in main if m.round_idx == r],
            key=lambda x: x.slot_idx,
        )
        losers: list[str] = []
        for m in round_matches:
            w = _winner(m)
            if w is None:
                continue
            ma, mb = _team_a(m), _team_b(m)
            loser = mb if w == ma else (ma if w == mb else None)
            if loser is not None and loser not in placed:
                losers.append(loser)
        # Bronze match splits SF losers if it's been picked.
        if r == final_round - 1:
            bronze = next((m for m in stage.matches if m.is_bronze), None)
            if bronze is not None:
                bw = _winner(bronze)
                if bw is not None and bw in losers:
                    bl = next((t for t in losers if t != bw), None)
                    losers = [bw] + ([bl] if bl else [])
        for t in losers:
            if t not in placed and len(placed) < count:
                placed.append(t)

    return placed[:count]


def _resolve_match_team(m, picks: dict[str, str],
                        by_id: dict[str, "BracketMatch"],
                        kind: str) -> str | None:
    """Recursively resolve a feeder's winner (or loser) through prior
    picks. Mirrors the UI's `_resolve` but standalone for engine use."""
    w = m.played_winner or picks.get(m.match_id)
    if w is None:
        return None
    if kind == "winner":
        return w
    # loser side: need both teams + winner.
    if m.round_idx == 1 and not m.is_bronze and m.sub in ("", "UB", "GR"):
        a, b = m.seed_a, m.seed_b
    else:
        a = _resolve_match_team(by_id[m.feeder_a], picks, by_id, m.feeder_a_kind) \
            if m.feeder_a in by_id else m.seed_a
        b = _resolve_match_team(by_id[m.feeder_b], picks, by_id, m.feeder_b_kind) \
            if m.feeder_b in by_id else m.seed_b
    if w == a:
        return b
    if w == b:
        return a
    return None


def _gsl_advancers(stage: Stage, picks: dict[str, str], count: int) -> list[str]:
    """
    Rank GSL / GSL-lite groups by within-group placement.

    GSL-lite (4 matches): 1st = WM winner, 2nd = WM loser, 3rd = EM winner,
    4th = EM loser. Both Opening winners advance regardless of WM outcome.

    Full GSL (5 matches): 1st = WM winner, 2nd = Decider winner, 3rd =
    Decider loser, 4th = EM loser.

    Returns an empty list when the Winners' Match is undecided — the
    cascade leaves downstream slots TBD.
    """
    if count <= 0 or not stage.matches:
        return []
    by_id = {m.match_id: m for m in stage.matches}

    def find(suffix: str):
        # GSL match IDs are ``<stage_id>::<suffix>`` (e.g. ``s0::GR-WM``);
        # split on the last ``::`` so we don't accidentally match a stage
        # whose id happens to contain the suffix substring.
        for m in stage.matches:
            tail = m.match_id.rsplit("::", 1)[-1]
            if tail == suffix:
                return m
        return None

    wm, em, dm = find("GR-WM"), find("GR-EM"), find("GR-DM")
    if wm is None:
        return []

    def team_on(m, side: str) -> str | None:
        seed = m.seed_a if side == "a" else m.seed_b
        if seed:
            return seed
        fid = m.feeder_a if side == "a" else m.feeder_b
        kind = m.feeder_a_kind if side == "a" else m.feeder_b_kind
        if fid and fid in by_id:
            return _resolve_match_team(by_id[fid], picks, by_id, kind)
        return None

    def winner_loser(m):
        w = m.played_winner or picks.get(m.match_id)
        if w is None:
            return None, None
        a, b = team_on(m, "a"), team_on(m, "b")
        loser = b if w == a else (a if w == b else None)
        return w, loser

    wm_w, wm_l = winner_loser(wm)
    if wm_w is None:
        return []

    placed: list[str] = [wm_w]

    if dm is not None:
        dm_w, dm_l = winner_loser(dm)
        if dm_w is not None and len(placed) < count:
            placed.append(dm_w)
        if dm_l is not None and len(placed) < count and dm_l not in placed:
            placed.append(dm_l)
    else:
        if wm_l is not None and len(placed) < count:
            placed.append(wm_l)
        if em is not None and len(placed) < count:
            em_w, _ = winner_loser(em)
            if em_w is not None and em_w not in placed:
                placed.append(em_w)

    if em is not None and len(placed) < count:
        em_w, em_l = winner_loser(em)
        if em_l is not None and em_l not in placed:
            placed.append(em_l)

    return placed[:count]


def compute_stage_advancers(
    stage: Stage,
    picks: dict[str, str],
    count: int,
    criterion: str = "top_by_wins",
) -> list[str]:
    """
    Compute the top ``count`` teams advancing from this stage based on the
    edge ``criterion``. Returns an ordered list (1st → Nth). Returns an
    empty list when too few picks have been made to rank — the cascade
    caller short-circuits and leaves downstream slots TBD.

    Phase 4 slice 1 implements ``top_by_wins`` (Swiss). Other criteria
    (``placement``, ``group_rank``) follow in subsequent slices.
    """
    if count <= 0 or not picks:
        return []

    if criterion == "placement":
        # SE / DE / SE_with_bronze: walk matches in round order, track
        # winners + losers. Final winner = 1st; final loser = 2nd; round
        # losers contribute to subsequent placement bands. Operates over
        # main-tree matches (SE) or UB/GF (DE).
        return _placement_advancers(stage, picks, count)

    if criterion == "gsl_rank":
        # GSL / GSL-lite mini-bracket placements.
        return _gsl_advancers(stage, picks, count)

    if criterion == "group_rank" and any(
        m.sub in ("UB", "LB") for m in stage.matches
    ):
        # DE-shaped groups (Atlanta / CSAC): format_parser emits
        # criterion=group_rank because Liquipedia prose calls them
        # "groups", but the DOM matches are bracket-shaped (UB/LB), not
        # round-robin. Dispatch on actual match shape, not criterion.
        return _placement_advancers(stage, picks, count)

    if criterion in ("top_by_wins", "group_rank"):
        # Walk Swiss / RR-group matches, accumulate W-L from picks + played.
        # Both criteria rank by wins descending; the difference is which
        # match-rows we count (Swiss matchlists for "top_by_wins", group
        # matchlists for "group_rank").
        target_subs = {"SW"} if criterion == "top_by_wins" else {"GR"}
        wins:   dict[str, int] = {}
        losses: dict[str, int] = {}
        seen_any_pick = False
        max_round = 0
        for m in stage.matches:
            if m.sub not in target_subs:
                continue
            if m.round_idx > max_round:
                max_round = m.round_idx
            a, b = m.seed_a, m.seed_b
            if not a or not b:
                continue
            w = m.played_winner or picks.get(m.match_id)
            if w is None:
                continue
            seen_any_pick = True
            wins.setdefault(a, 0); wins.setdefault(b, 0)
            losses.setdefault(a, 0); losses.setdefault(b, 0)
            if w == a:
                wins[a]   += 1
                losses[b] += 1
            elif w == b:
                wins[b]   += 1
                losses[a] += 1
        if not seen_any_pick:
            return []
        # CS2 Swiss/RR-group advancement gate: a team is "advanced" only when
        # it reaches the win threshold. CS2 standard is the 3-3 Swiss
        # (16 teams, 5 rounds, 3W advance / 3L eliminate). For 16-team Swiss
        # the threshold is hardcoded to 3 — Liquipedia's matchlist DOM only
        # publishes pairings round-by-round, so deriving threshold from
        # `max(round_idx)` would underestimate during early rounds and let
        # the cascade fire prematurely after R1. Smaller Swiss configurations
        # (test fixtures, hypothetical mini-Swiss) fall back to
        # ceil(rounds/2) using max observed round.
        team_count_proxy = (len(stage.roster) or
                            sum(en.count for en in stage.entrants
                                if en.source != "advance_from"))
        if team_count_proxy >= 16:
            threshold = 3
        else:
            threshold = max(1, (max_round + 1) // 2)
        qualified = [t for t in wins if wins[t] >= threshold]
        if len(qualified) < count:
            return []
        # Rank by (wins desc, losses asc, alphabetical). Alphabetical is a
        # placeholder Buchholz tiebreak; refine when we wire real Buchholz.
        qualified.sort(key=lambda t: (-wins[t], losses[t], t))
        return qualified[:count]

    return []


def apply_cross_stage_cascade(
    stages: list[Stage],
    picks: dict[str, str],
) -> dict[str, list[str]]:
    """
    Walk the stage graph and resolve downstream rosters from upstream picks.

    For each stage that has ``advance_from`` entrants, look up the upstream
    stage by ``def_name`` and call ``compute_stage_advancers`` with the
    entrant's count + criterion. Concatenate all such advancer lists, then
    append the stage's pre-computed ``direct_invitees`` (Major-tier stages
    that ship with concrete invitee names).

    Returns a dict keyed by ``Stage.def_name`` whose values are the
    computed downstream rosters. Stages whose entrants are *only*
    ``initial_roster`` are skipped — their roster was already populated
    by ``build_stage_graph``.

    Multi-group fan-in (e.g. Atlanta's two DE groups → 6-team Playoffs)
    falls out naturally: each upstream stage contributes its own advancer
    list, concatenated in DOM order. Slot-level seeding (1st-Group-A vs
    2nd-Group-B, etc.) is deferred to the manual-pick dropdown for now.
    """
    # A multi-group StageDef (Atlanta / BLAST Rivals: two parallel "Group A"
    # + "Group B" DOM stages both linked to a single ``Group Stage`` def)
    # collapses to ONE upstream key but multiple DOM stages. Index them
    # all so the cascade can fan over each group and concatenate.
    by_def: dict[str, list[Stage]] = {}
    for s in stages:
        if s.def_name:
            by_def.setdefault(s.def_name, []).append(s)

    rosters: dict[str, list[str]] = {}

    for stage in stages:
        if not stage.def_name:
            continue
        adv_entrants = [en for en in stage.entrants
                        if en.source == "advance_from"]
        if not adv_entrants:
            continue

        roster: list[str] = []
        for en in adv_entrants:
            upstreams = by_def.get(en.upstream_stage or "", [])
            if not upstreams:
                continue
            # Aggregate count = per-group count × n_groups (set by
            # build_stage_graph from the StageDef's edge.count).
            per_group = en.count // len(upstreams) if len(upstreams) > 1 else en.count
            for up in upstreams:
                advancers = compute_stage_advancers(
                    up, picks,
                    count=per_group,
                    criterion=en.criterion or "top_by_wins",
                )
                roster.extend(advancers)

        for t in stage.direct_invitees:
            if t not in roster:
                roster.append(t)

        rosters[stage.def_name] = roster

    return rosters


# Standard SE seeding maps. Index = slot_idx of R1 match; value = (a_seed,
# b_seed) where seed is 1-based into the cascade-resolved roster.
# Layout convention: slot 0 = top-of-bracket; consecutive slots feed SFs
# from outer-to-inner so seeds 1 and 2 don't meet before the final.
_SE_SEEDING_MAPS: dict[int, list[tuple[int, int]]] = {
    2: [(1, 2)],
    4: [(1, 4), (2, 3)],
    8: [(1, 8), (4, 5), (2, 7), (3, 6)],
    16: [(1, 16), (8, 9), (4, 13), (5, 12),
         (2, 15), (7, 10), (3, 14), (6, 11)],
}


def seat_cross_stage_r1(stage: Stage, roster: list[str]) -> None:
    """
    Seat a cascade-resolved roster into ``stage``'s R1 slots using the
    standard CS2 seeding map (1v8, 4v5, 2v7, 3v6 for 8-team SE; 1v4, 2v3
    for 4-team; etc.). Mutates ``stage.matches`` in place.

    Fan-in special case: 6-team SE+bronze fed by 2 groups × 3 advancers
    (Atlanta layout). Roster arrives [1A,2A,3A,1B,2B,3B]. QF1 = 2A vs 3B,
    QF2 = 2B vs 3A; SF byes = 1A (top half) and 1B (bottom half). This
    keeps group winners on opposite halves of the bracket.

    No-ops when:
      * any R1 slot already has a seed (Liquipedia or manual_seeds wins),
      * roster size doesn't match a known seeding map,
      * stage has no R1 matches.
    """
    if not stage.matches or not roster:
        return
    main = [m for m in stage.matches if not m.is_bronze
            and m.sub in ("", "UB")]
    r1 = sorted([m for m in main if m.round_idx == 1], key=lambda x: x.slot_idx)
    if not r1:
        return
    if any(m.seed_a or m.seed_b for m in r1):
        return  # Liquipedia / manual seeds own this stage.

    # Fan-in 6-team SE+bronze: cross-pair QFs + seat SF byes.
    if len(roster) == 6 and len(r1) == 2:
        sfs = sorted([m for m in main if m.round_idx == 2],
                     key=lambda x: x.slot_idx)
        if len(sfs) == 2:
            r1[0].seed_a, r1[0].seed_b = roster[1], roster[5]   # 2A vs 3B
            r1[1].seed_a, r1[1].seed_b = roster[4], roster[2]   # 2B vs 3A
            for sf, top_seed in zip(sfs, (roster[0], roster[3])):
                # SF bye side = whichever side has no feeder.
                if not sf.feeder_a and not sf.seed_a:
                    sf.seed_a = top_seed
                elif not sf.feeder_b and not sf.seed_b:
                    sf.seed_b = top_seed
            return

    seeding = _SE_SEEDING_MAPS.get(len(roster))
    if seeding is None or len(seeding) != len(r1):
        return  # Unknown shape — leave TBD for manual dropdown.

    for m, (a_idx, b_idx) in zip(r1, seeding):
        m.seed_a = roster[a_idx - 1]
        m.seed_b = roster[b_idx - 1]


def compute_place_offsets(stage_defs: list[StageDef]) -> dict[str, int]:
    """
    Phase 5 §5.4 — compute absolute-place offset per stage_def via
    topological walk.

    For each stage_def S, ``offset(S) = number of teams placed BETTER
    than S's dropouts``. S's dropouts then get places (offset+1) through
    (offset + dropout_count).

    Algorithm: walk topologically from terminal backward.
      offset(terminal)        = 0
      offset(upstream)        = offset(downstream) + dropout_count(downstream)
      dropout_count(terminal) = team_count            (all placed at terminal)
      dropout_count(other)    = team_count - sum(advance_to[i].count)

    Linear chains (Cologne S1→S2→S3→Playoffs): each upstream's offset
    accumulates downstream dropouts. Fan-in (Atlanta A+B → Playoffs):
    sibling stages share an offset since they aggregate at the def
    level. Caller's per-stage placement walker emits within-stage rank,
    bucketed across sibling stages → tied label.
    """
    if not stage_defs:
        return {}

    by_name = {d.name: d for d in stage_defs}

    # Find terminal stages (no advance_to). Multiple terminals possible
    # for fan-out events; for current S-Tier corpus there's always one.
    targets: set[str] = set()
    for d in stage_defs:
        for e in d.advance_to:
            targets.add(e.target_stage)
    # A stage is "downstream of S" if S.advance_to lists it. Build the
    # reverse map: downstream → list[upstream].
    upstream_of: dict[str, list[str]] = {}
    for d in stage_defs:
        for e in d.advance_to:
            upstream_of.setdefault(e.target_stage, []).append(d.name)

    def _dropout_count(d: StageDef) -> int:
        adv = sum(e.count for e in d.advance_to)
        return d.team_count - adv if adv else d.team_count

    # Topo walk: process terminal first (offset=0), then BFS upstream.
    offsets: dict[str, int] = {}
    queue: list[str] = [d.name for d in stage_defs if not d.advance_to]
    for n in queue:
        offsets[n] = 0
    visited: set[str] = set(queue)
    while queue:
        next_queue: list[str] = []
        for downstream_name in queue:
            for upstream_name in upstream_of.get(downstream_name, []):
                if upstream_name in visited:
                    continue
                downstream_def = by_name[downstream_name]
                offsets[upstream_name] = (
                    offsets[downstream_name] + _dropout_count(downstream_def)
                )
                visited.add(upstream_name)
                next_queue.append(upstream_name)
        queue = next_queue

    return offsets


def resolve_for_render(
    parsed_stages: list[Stage],
    picks: dict[str, str],
    manual_seeds: dict[str, dict[str, str]] | None = None,
    snapshot_standings: dict[str, int] | None = None,
) -> list[Stage]:
    """
    Phase 5 §5.1 — single seam between cached parsed_stages and any caller
    that needs realised stages (UI fragment, engine emission).

    Steps (deep-copies first; never mutates inputs):
      1. Overlay ``manual_seeds`` onto match ``seed_a/seed_b``.
      2. Bake Swiss synth pairings (Buchholz pairer) into seed slots so
         downstream cascade tally / placement walkers see all rounds.
      3. Apply cross-stage cascade — populates each downstream stage's
         ``roster`` from upstream advancers + concrete direct invitees.
      4. Seat cross-stage R1 from cascade roster (Option B for power-of-
         two SE; fan-in for 6-team SE+bronze; bye fill for everything
         else via ``_apply_roster_seeds``).

    Pure function. Same input → same output. Safe to call from any
    thread or non-Streamlit context.
    """
    import copy
    from .bracket_parser import _apply_roster_seeds

    manual_seeds = manual_seeds or {}
    stages = copy.deepcopy(parsed_stages)

    # Step 1 — manual_seeds overlay.
    for stage in stages:
        for m in stage.matches:
            ms = manual_seeds.get(m.match_id) or {}
            if ms.get("a") and not m.seed_a:
                m.seed_a = ms["a"]
            if ms.get("b") and not m.seed_b:
                m.seed_b = ms["b"]

    # Step 2 — bake Swiss synth pairings into seed slots.
    for stage in stages:
        if stage.format == "Swiss":
            ov = compute_swiss_overrides(stage, picks, snapshot_standings)
            for m in stage.matches:
                if m.match_id in ov and not m.seed_a:
                    m.seed_a, m.seed_b = ov[m.match_id]

    # Step 3 — cross-stage cascade.
    cascade_rosters = apply_cross_stage_cascade(stages, picks)

    # Step 4 — seat downstream R1 from cascade roster + bye fill.
    for stage in stages:
        roster = cascade_rosters.get(stage.def_name)
        if not roster:
            continue
        # Partial-roster guard (matches autofill loop guard in app.py):
        # only seat downstream stages when cascade has the full expected
        # team count. Otherwise wait — partial seat would bake wrong
        # teams into byes that fan-in can't overwrite later.
        expected = sum(en.count for en in stage.entrants
                       if en.source in ("advance_from", "direct_invite"))
        if expected and len(roster) < expected:
            continue
        stage.roster = roster
        seat_cross_stage_r1(stage, roster)
        _apply_roster_seeds(stage.matches, roster)

    return stages


def compute_swiss_overrides(
    stage: Stage,
    picks: dict[str, str],
    snapshot_standings: dict[str, int] | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Compute round-2+ Swiss pairings dynamically from earlier-round picks
    using ``vrs_engine.swiss_pairer``.

    Behaviour (mirrors prior app.py implementation; see CLAUDE.md /
    NEXT_STEPS_BRACKETS.md §10):
      * R1: trust ``seed_a/seed_b`` (Liquipedia DOM truth); accumulate
        W-L + opponent history from picks.
      * R>1: if Liquipedia (or a prior bake) populated any seed, trust
        the round and ingest its results. Without picks for the round,
        stop cascading deeper (subsequent rounds need this round's
        results to compute Buchholz buckets).
      * Synth path: with snapshot AND ≥1 earlier pick, call
        ``pair_round`` for the next round's pairings.
      * No snapshot → return empty (synth disabled).

    Returns ``{match_id: (team_a, team_b)}`` for synthesised matches.
    """
    from collections import defaultdict
    from vrs_engine.swiss_pairer import seed_table, pair_round

    overrides: dict[str, tuple[str, str]] = {}
    snapshot = snapshot_standings or {}

    sw = [m for m in stage.matches if m.sub == "SW"]
    if not sw:
        return overrides

    by_round: dict[int, list] = defaultdict(list)
    for m in sw:
        by_round[m.round_idx].append(m)
    for r in by_round:
        by_round[r].sort(key=lambda x: x.slot_idx)

    records: dict[str, tuple[int, int]] = {}
    opps: dict[str, list[str]] = {}

    def _ingest_match(m, a, b):
        if not a or not b:
            return
        w = m.played_winner or picks.get(m.match_id)
        records.setdefault(a, (0, 0)); records.setdefault(b, (0, 0))
        opps.setdefault(a, []);        opps.setdefault(b, [])
        if b not in opps[a]:
            opps[a].append(b)
        if a not in opps[b]:
            opps[b].append(a)
        if w == a:
            wa, la = records[a]; wb, lb = records[b]
            records[a] = (wa + 1, la); records[b] = (wb, lb + 1)
        elif w == b:
            wa, la = records[a]; wb, lb = records[b]
            records[a] = (wa, la + 1); records[b] = (wb + 1, lb)

    for r in sorted(by_round):
        round_matches = by_round[r]

        if r == 1:
            for m in round_matches:
                _ingest_match(m, m.seed_a, m.seed_b)
            continue

        any_seeded = any(m.seed_a or m.seed_b for m in round_matches)
        if any_seeded:
            any_pick_in_round = False
            for m in round_matches:
                _ingest_match(m, m.seed_a, m.seed_b)
                if m.played_winner or picks.get(m.match_id):
                    any_pick_in_round = True
            if not any_pick_in_round:
                break
            continue

        if not snapshot:
            break
        if not any(w > 0 or l > 0 for (w, l) in records.values()):
            break

        roster = list(stage.roster) if stage.roster else list(records.keys())
        seeds = seed_table(roster, snapshot)

        pairs = pair_round(records, opps, seeds)
        any_pick_in_round = False
        for m, (a, b) in zip(round_matches, pairs):
            overrides[m.match_id] = (a, b)
            _ingest_match(m, a, b)
            if m.played_winner or picks.get(m.match_id):
                any_pick_in_round = True

        if not any_pick_in_round:
            break

    return overrides


def _default_fetch_roster(slug: str) -> list[str]:
    """Default sub-page roster fetcher — HTTPs the slug, parses teamcards."""
    from .liquipedia_loader import _fetch_page, _bs4, _parse_seeded_teams
    html = _fetch_page(slug)
    if html is None:
        return []
    return _parse_seeded_teams(_bs4(html))


def collect_sub_page_rosters(
    main_slug: str,
    main_soup,
    *,
    fetch_roster: Callable[[str], list[str]] | None = None,
) -> dict[str, list[str]]:
    """
    For Major-tier events, discover ``/Stage_N`` sub-pages from anchor hrefs
    on the main page and fetch each one's teamcards roster.

    Returns ``{StageDef.name: list[canonical team names]}`` keyed by
    ``"Stage N"`` so the result drops straight into ``build_stage_graph``'s
    ``sub_page_rosters`` argument.
    """
    fetch = fetch_roster or _default_fetch_roster
    rosters: dict[str, list[str]] = {}
    for n, sub_slug in _discover_sub_stage_slugs(main_soup, main_slug):
        roster = fetch(sub_slug)
        if roster:
            rosters[f"Stage {n}"] = roster
    return rosters


def build_stage_graph(
    parsed_stages: list[Stage],
    stage_defs: list[StageDef],
    seeded_teams: list[str],
    sub_page_rosters: dict[str, list[str]] | None = None,
    direct_invitees_by_stage: dict[str, list[str]] | None = None,
) -> None:
    """Link parsed Stages to StageDefs by heading and populate cross-stage fields.

    The first DOM stage that links to a def is treated as the *initial* stage
    of the chain — its roster is the tournament's `seeded_teams`. Mid-chain
    stages get rosters from `sub_page_rosters` when present (Major-tier
    events ship per-stage sub-pages with their own teamcards section).

    `direct_invitees_by_stage` is the alternate Major source: when the main
    page partitions the Participants section into ``"Stage N Invites"`` h3
    blocks (parsed via ``liquipedia_loader._parse_per_stage_invites``), the
    first stage's roster comes from that block (instead of the full 32-team
    cross-stage listing) and downstream stages pre-fill `direct_invitees`
    with concrete names.
    """
    sub_page_rosters = sub_page_rosters or {}
    direct_invitees_by_stage = direct_invitees_by_stage or {}
    defs_by_name = {d.name: d for d in stage_defs}

    initial_stage_seeded = False
    for stage in parsed_stages:
        d = _resolve_def(stage, defs_by_name, stage_defs)
        if d is None:
            continue

        stage.def_name        = d.name
        stage.advance_to      = list(d.advance_to)
        stage.eliminations    = list(d.eliminations)

        # Per-stage invites (Major main-page layout) win over the full
        # cross-stage seeded_teams list for the *initial* stage.
        invitees = direct_invitees_by_stage.get(d.name, [])

        if not initial_stage_seeded:
            initial_roster = invitees if invitees else list(seeded_teams)
            if initial_roster:
                stage.roster   = list(initial_roster)
                stage.entrants = [StageEntrant(
                    source="initial_roster",
                    count=len(initial_roster),
                )]
                initial_stage_seeded = True

    # Backfill downstream `advance_from` entrants by walking upstream edges.
    # Stages that already have an `initial_roster` entrant keep it. When
    # multiple DOM stages share a single multi-group StageDef (e.g. "Group A"
    # and "Group B" both link to "Group Stage"), the StageDef's advance_to
    # count is already the aggregate — emit one entrant per (upstream_def,
    # target) pair, not one per DOM stage.
    stage_by_def_name = {s.def_name: s for s in parsed_stages if s.def_name}
    emitted: set[tuple[str, str]] = set()
    for upstream in parsed_stages:
        if not upstream.def_name:
            continue
        for edge in upstream.advance_to:
            target = stage_by_def_name.get(edge.target_stage)
            if target is None or any(en.source == "initial_roster"
                                     for en in target.entrants):
                continue
            key = (upstream.def_name, edge.target_stage)
            if key in emitted:
                continue
            emitted.add(key)
            target.entrants.append(StageEntrant(
                source="advance_from",
                upstream_stage=upstream.def_name,
                count=edge.count,
                criterion=edge.criterion,
                seeding=edge.seeding,
            ))

    # Gap fill: any stage whose entrant counts underflow its def's team_count
    # gets a `direct_invite` source. Three resolution paths, in priority:
    #   1. `direct_invitees_by_stage` (Major main-page "Stage N Invites" h3s)
    #   2. `sub_page_rosters` (Major /Stage_N sub-page teamcards)
    #   3. placeholder count-only entrant
    known_so_far: list[str] = list(seeded_teams)
    for stage in parsed_stages:
        if not stage.def_name:
            continue
        d = defs_by_name[stage.def_name]
        is_initial = any(en.source == "initial_roster" for en in stage.entrants)

        invitees_explicit = direct_invitees_by_stage.get(stage.def_name)
        sub_roster        = sub_page_rosters.get(stage.def_name)

        if invitees_explicit and not is_initial:
            stage.direct_invitees = list(invitees_explicit)
            stage.entrants.append(StageEntrant(
                source="direct_invite",
                count=len(invitees_explicit),
                notes=",".join(invitees_explicit),
            ))
        elif sub_roster is not None and not is_initial:
            stage.roster = list(sub_roster)
            invitees = [t for t in sub_roster if t not in set(known_so_far)]
            stage.direct_invitees = invitees
            if invitees:
                stage.entrants.append(StageEntrant(
                    source="direct_invite",
                    count=len(invitees),
                    notes=",".join(invitees),
                ))
        else:
            have = sum(en.count for en in stage.entrants)
            gap = d.team_count - have
            if gap > 0:
                stage.entrants.append(StageEntrant(
                    source="direct_invite",
                    count=gap,
                    notes="placeholder — sub-page teamcard scrape not yet wired",
                ))

        known_so_far.extend(stage.roster)
