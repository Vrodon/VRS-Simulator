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


def compute_gsl_exit_buckets(
    stage: Stage,
    picks: dict[str, str],
) -> list[tuple[str, list[str]]]:
    """
    GSL / GSL-lite within-group dropout buckets, best-first.

    Top 2 always advance silent (their prize comes from the downstream
    stage). Dropouts:

    * GSL-lite (no Decider): 3rd = Elimination-Match winner, 4th = EM loser.
    * Full GSL (with Decider): 3rd = Decider loser, 4th = EM loser.

    Bucket label = within-group rank as string ("3" / "4"). Caller
    composes with cross-group aggregation + ``compute_place_offsets`` to
    produce absolute labels.

    Returns ``[]`` until enough picks resolve who occupies each bucket.
    """
    if not stage.matches:
        return []

    by_id = {m.match_id: m for m in stage.matches}

    def find_suffix(suffix: str):
        for m in stage.matches:
            tail = m.match_id.rsplit("::", 1)[-1]
            if tail == suffix:
                return m
        return None

    wm = find_suffix("GR-WM")
    em = find_suffix("GR-EM")
    dm = find_suffix("GR-DM")
    if wm is None or em is None:
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

    em_w, em_l = winner_loser(em)

    buckets: list[tuple[str, list[str]]] = []
    if dm is not None:
        # Full GSL: 3rd = Decider loser, 4th = EM loser.
        _, dm_l = winner_loser(dm)
        if dm_l is not None:
            buckets.append(("3", [dm_l]))
        if em_l is not None and em_l != (dm_l if dm is not None else None):
            buckets.append(("4", [em_l]))
    else:
        # GSL-lite: 3rd = EM winner, 4th = EM loser.
        if em_w is not None:
            buckets.append(("3", [em_w]))
        if em_l is not None:
            buckets.append(("4", [em_l]))

    return buckets


def compute_de_compact_exit_buckets(
    stage: Stage,
    picks: dict[str, str],
) -> list[tuple[str, list[str]]]:
    """
    DE compact (8-team group format) within-stage dropout buckets.

    Top 3 advance silent (UB winner = 1st-in-group; UB-final loser =
    2nd-in-group; LB-final winner = 3rd-in-group). Dropouts:

      "4"   = LB-final loser
      "5-6" = LB semifinal (round-before-final) losers
      "7-8" = LB R1 losers
      "9-12", … = generalised: each earlier LB round's losers form a
                  doubling bucket.

    Empty list when LB final hasn't been picked yet (we can't yet
    distinguish 3rd from 4th).

    Returns ``[(label, [team, …]), …]`` best-first; teams within a
    bucket are sorted alphabetically.
    """
    if not stage.matches:
        return []

    by_id = {m.match_id: m for m in stage.matches}
    lb = [m for m in stage.matches if m.sub == "LB"]
    if not lb:
        return []

    lb_max = max(m.round_idx for m in lb)

    def winner(m):
        return m.played_winner or picks.get(m.match_id)

    def team_on(m, side: str) -> str | None:
        seed = m.seed_a if side == "a" else m.seed_b
        if seed:
            return seed
        fid = m.feeder_a if side == "a" else m.feeder_b
        kind = m.feeder_a_kind if side == "a" else m.feeder_b_kind
        if fid and fid in by_id:
            return _resolve_match_team(by_id[fid], picks, by_id, kind)
        return None

    def loser_of(m):
        w = winner(m)
        if w is None:
            return None
        a, b = team_on(m, "a"), team_on(m, "b")
        return b if w == a else (a if w == b else None)

    # Need LB final pick to slot the 4th bucket.
    lb_final = next((m for m in lb if m.round_idx == lb_max), None)
    if lb_final is None or winner(lb_final) is None:
        return []

    buckets: list[tuple[str, list[str]]] = []

    # 4th = LB final loser.
    lf_loser = loser_of(lb_final)
    if lf_loser is not None:
        buckets.append(("4", [lf_loser]))

    # Earlier LB rounds shed one loser per match. Bucket size for round r
    # = match count in that round. Standard 8-team DE: LB R2 has 2 matches
    # ("5-6"), LB R1 has 2 matches ("7-8"). Larger DEs scale: LB R3 has 1
    # match ("4"), LB R2 has 2 ("5-6"), LB R1 has 4 ("7-10"), etc.
    base = 4
    for r in range(lb_max - 1, 0, -1):
        round_matches = [m for m in lb if m.round_idx == r]
        size = len(round_matches)
        if size == 0:
            continue
        lo = base + 1
        hi = base + size
        label = f"{lo}-{hi}" if lo != hi else str(lo)
        losers: list[str] = []
        for m in round_matches:
            w = winner(m)
            if w is None:
                continue
            l = loser_of(m)
            if l is not None:
                losers.append(l)
        if not losers:
            base = hi
            continue
        losers.sort()
        buckets.append((label, losers))
        base = hi

    return buckets


def compute_rr_exit_buckets(
    stage: Stage,
    picks: dict[str, str],
) -> list[tuple[str, list[str]]]:
    """
    Round-robin group dropout buckets.

    Top ``stage.advance_to[0].count`` teams advance silent (their prize
    comes from the downstream stage). Remaining teams form one bucket
    per within-group rank — bucket label is the rank as a string ("3"
    / "4" / etc.). Tied wins → tied bucket (single label, multiple
    teams alphabetical).

    RR matches use ``sub == "GR"`` and IDs ending in ``GR-M{i}``. We
    walk those, tally wins, and rank.

    Returns ``[]`` when not all RR matches are resolved (caller waits
    until the group is fully picked before emitting prizes — matches
    Q6 strict partial-fill rule).
    """
    if not stage.matches:
        return []

    rr = [m for m in stage.matches if m.sub == "GR"
          and m.match_id.rsplit("::", 1)[-1].startswith("GR-M")]
    if not rr:
        return []

    # Strict: every RR match must be picked or played for the rank to lock.
    wins: dict[str, int] = {t: 0 for t in stage.roster}
    for m in rr:
        a, b = m.seed_a, m.seed_b
        w = m.played_winner or picks.get(m.match_id)
        if not a or not b or w is None:
            return []
        wins[a] = wins.get(a, 0)
        wins[b] = wins.get(b, 0)
        if w == a:
            wins[a] += 1
        elif w == b:
            wins[b] += 1

    advance_count = (stage.advance_to[0].count
                     if stage.advance_to else 0)

    # Sort by wins desc; group tied teams alphabetically within bucket.
    teams_by_wins: dict[int, list[str]] = {}
    for t, w in wins.items():
        teams_by_wins.setdefault(w, []).append(t)
    for tied in teams_by_wins.values():
        tied.sort()

    # Walk wins desc, accumulate rank position; emit only buckets whose
    # rank-range starts above advance_count.
    buckets: list[tuple[str, list[str]]] = []
    rank = 1
    for w in sorted(teams_by_wins, reverse=True):
        tied = teams_by_wins[w]
        size = len(tied)
        rank_lo = rank
        rank_hi = rank + size - 1
        if rank_lo > advance_count:
            label = (f"{rank_lo}-{rank_hi}" if size > 1 else str(rank_lo))
            buckets.append((label, tied))
        rank += size

    return buckets


def compute_se_exit_buckets(
    stage: Stage,
    picks: dict[str, str],
) -> list[tuple[str, list[str]]]:
    """
    SE / SE+bronze playoff bucket walker (terminal stage — every team
    receives a placement).

    Buckets best-first:

      "1"   = final winner
      "2"   = final loser
      "3-4" = SF losers when there's no bronze match
      "3"   = bronze winner       \\__ when SE+bronze
      "4"   = bronze loser        /
      "5-8" = QF losers (4 teams)
      "9-16" = R1 losers in 16-team SE (8 teams)
      …

    Earlier-round losers form doubling buckets per the standard SE
    placement convention: round r contributes losers at place range
    [2^(N-r)+1 .. 2^(N-r+1)].

    Skipped if the round's losers can't be resolved (partial picks).
    """
    if not stage.matches:
        return []

    by_id = {m.match_id: m for m in stage.matches}

    def winner(m):
        return m.played_winner or picks.get(m.match_id)

    def team_on(m, side: str) -> str | None:
        seed = m.seed_a if side == "a" else m.seed_b
        if seed:
            return seed
        fid = m.feeder_a if side == "a" else m.feeder_b
        kind = m.feeder_a_kind if side == "a" else m.feeder_b_kind
        if fid and fid in by_id:
            return _resolve_match_team(by_id[fid], picks, by_id, kind)
        return None

    def loser_of(m):
        w = winner(m)
        if w is None:
            return None
        a, b = team_on(m, "a"), team_on(m, "b")
        return b if w == a else (a if w == b else None)

    main = [m for m in stage.matches if not m.is_bronze]
    if not main:
        return []
    rounds_max = max(m.round_idx for m in main)
    final_match = next((m for m in main if m.round_idx == rounds_max), None)
    bronze = next((m for m in stage.matches if m.is_bronze), None)

    buckets: list[tuple[str, list[str]]] = []
    placed: set[str] = set()

    if final_match is not None:
        f_w = winner(final_match)
        if f_w is not None:
            buckets.append(("1", [f_w]))
            placed.add(f_w)
            f_l = loser_of(final_match)
            if f_l is not None:
                buckets.append(("2", [f_l]))
                placed.add(f_l)

    # SF losers — bronze splits 3 / 4 if picked; otherwise tied "3-4".
    sf_losers: list[str] = []
    for m in main:
        if m.round_idx != rounds_max - 1:
            continue
        l = loser_of(m)
        if l is not None and l not in placed:
            sf_losers.append(l)
    if sf_losers:
        sf_losers.sort()
        if bronze is not None:
            bw = winner(bronze)
            if bw is not None:
                bl = loser_of(bronze)
                if bw not in placed:
                    buckets.append(("3", [bw]))
                    placed.add(bw)
                if bl is not None and bl not in placed:
                    buckets.append(("4", [bl]))
                    placed.add(bl)
            else:
                # Bronze unpicked → leave SF losers unbucketed for now.
                pass
        else:
            buckets.append(("3-4", sf_losers))
            placed.update(sf_losers)

    # Earlier-round losers — doubling buckets.
    for r in range(rounds_max - 2, 0, -1):
        losers_so_far = 2 ** (rounds_max - r)
        place_lo = losers_so_far + 1
        place_hi = losers_so_far * 2
        label = f"{place_lo}-{place_hi}" if place_lo != place_hi else str(place_lo)
        round_losers: list[str] = []
        for m in main:
            if m.round_idx != r:
                continue
            l = loser_of(m)
            if l is not None and l not in placed:
                round_losers.append(l)
        if round_losers:
            round_losers.sort()
            buckets.append((label, round_losers))
            placed.update(round_losers)

    return buckets


def compute_place_offsets(stages: list[Stage]) -> dict[str, int]:
    """
    Per-stage absolute place offset.

    ``offset(stage)`` = number of placements that lie *above* this stage's
    dropouts in the absolute tournament ranking.

      Terminal stage (no advance_to): offset = 0 — its 1st place is the
      tournament's 1st place.
      Linear chain: offset(N) = team_count(N+1) + offset(N+1).
        Equivalently: cumulative team_count of all downstream stages.
      Fan-in (multiple sibling stages share def_name): siblings share
        offset; cross-sibling aggregation happens at integration time.

    Returns ``{def_name: offset}``. Stages without a ``def_name`` are
    skipped (engine emit only emits prizes for stages tied to the graph).
    """
    by_def: dict[str, list[Stage]] = {}
    for s in stages:
        if not s.def_name:
            continue
        by_def.setdefault(s.def_name, []).append(s)

    def team_count_of(def_name: str) -> int:
        sib = by_def.get(def_name, [])
        if not sib:
            return 0
        # Use the first sibling — siblings share entrant shape.
        return sum(en.count for en in sib[0].entrants)

    offsets: dict[str, int] = {}

    def offset_of(def_name: str) -> int:
        if def_name in offsets:
            return offsets[def_name]
        sib = by_def.get(def_name, [])
        if not sib:
            offsets[def_name] = 0
            return 0
        edges = sib[0].advance_to
        if not edges:
            offsets[def_name] = 0
            return 0
        target = edges[0].target_stage
        offsets[def_name] = team_count_of(target) + offset_of(target)
        return offsets[def_name]

    for def_name in by_def:
        offset_of(def_name)

    return offsets


_ORDINAL_TENS = ('th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th')


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{_ORDINAL_TENS[n % 10]}"


def _format_place(lo: int, hi: int) -> str:
    if lo == hi:
        return _ordinal(lo)
    return f"{_ordinal(lo)}-{_ordinal(hi)}"


def _detect_groups_subformat(stage: Stage) -> str:
    """Distinguish GSL/GSL-lite vs RR for ``format == "Groups"`` stages."""
    for m in stage.matches:
        tail = m.match_id.rsplit("::", 1)[-1]
        if tail in ("GR-WM", "GR-EM", "GR-DM") or tail.startswith("GR-O"):
            return "gsl"
        if tail.startswith("GR-M"):
            return "rr"
    return ""


def _exit_buckets_for(stage: Stage,
                      picks: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Dispatch the right format walker for ``stage``."""
    fmt = stage.format
    if fmt == "Swiss":
        return compute_swiss_exit_buckets(stage, picks)
    if fmt == "DE":
        return compute_de_compact_exit_buckets(stage, picks)
    if fmt in ("SE", "SE_with_bronze"):
        return compute_se_exit_buckets(stage, picks)
    if fmt == "Groups":
        sub = _detect_groups_subformat(stage)
        if sub == "gsl":
            return compute_gsl_exit_buckets(stage, picks)
        if sub == "rr":
            return compute_rr_exit_buckets(stage, picks)
    return []


def compute_absolute_placements(
    stages: list[Stage],
    picks: dict[str, str],
) -> dict[str, str]:
    """
    Map each team to its absolute Liquipedia placement label.

    Composes per-stage exit buckets (via per-format walkers) with
    per-stage place offsets (via ``compute_place_offsets``) and fan-in
    sibling aggregation to produce labels like ``"5th-8th"`` /
    ``"9th-12th"`` matching Liquipedia's ``prize_distribution`` keys.

    Sibling stages (same ``def_name``) are aggregated per bucket label —
    e.g. Atlanta's two DE groups both emit a ``"4"`` bucket with one
    team each; they merge into a single 2-team bucket → ``"7th-8th"``.
    Cumulative position per def_name advances across merged buckets.

    Returns ``{team: absolute_label}``. Teams whose path includes any
    unresolved bucket are omitted (Q6 strict partial-fill).
    """
    offsets = compute_place_offsets(stages)

    by_def: dict[str, list[Stage]] = {}
    for s in stages:
        if not s.def_name:
            continue
        by_def.setdefault(s.def_name, []).append(s)

    # Each team's exit stage = the deepest stage whose roster contains them
    # (Phase 5 §5.4). Walk stages in parse order (upstream → downstream);
    # last-write-wins captures the deepest stage. Teams that advance to a
    # downstream stage get filtered out of upstream walker output so their
    # prize comes from the downstream stage's bucket only.
    team_exit_def: dict[str, str] = {}
    for stage in stages:
        if not stage.def_name:
            continue
        for team in stage.roster:
            team_exit_def[team] = stage.def_name

    placements: dict[str, str] = {}

    for def_name, sibling_stages in by_def.items():
        offset = offsets.get(def_name, 0)
        per_sibling_buckets = [_exit_buckets_for(s, picks) for s in sibling_stages]

        order: list[str] = []
        merged: dict[str, list[str]] = {}
        for buckets in per_sibling_buckets:
            for label, teams in buckets:
                kept = [t for t in teams if team_exit_def.get(t) == def_name]
                if not kept:
                    continue
                if label not in merged:
                    merged[label] = []
                    order.append(label)
                merged[label].extend(kept)
        for label in merged:
            merged[label].sort()

        cum = 0
        for label in order:
            teams = merged[label]
            size = len(teams)
            lo = offset + cum + 1
            hi = offset + cum + size
            absolute_label = _format_place(lo, hi)
            for t in teams:
                placements[t] = absolute_label
            cum += size

    return placements


def _resolve_match_team(m, picks, by_id, kind):
    """Recursive feeder walk — mirrors stage_graph helper.

    Resolves a match's winner / loser by walking back through feeders
    via picks. Used when a downstream match's seeds are populated only
    via feeder edges.
    """
    w = m.played_winner or picks.get(m.match_id)
    if w is None:
        return None
    if kind == "winner":
        return w
    if m.round_idx == 1 and not m.is_bronze and m.sub in ("", "UB", "GR"):
        a, b = m.seed_a, m.seed_b
    else:
        a = (_resolve_match_team(by_id[m.feeder_a], picks, by_id, m.feeder_a_kind)
             if m.feeder_a in by_id else m.seed_a)
        b = (_resolve_match_team(by_id[m.feeder_b], picks, by_id, m.feeder_b_kind)
             if m.feeder_b in by_id else m.seed_b)
    if w == a:
        return b
    if w == b:
        return a
    return None
