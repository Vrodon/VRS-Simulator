"""
Bracket Parser
==============
Turns Liquipedia tournament-page bracket markup into a structured tree the
Tournament Predictor UI can render and the engine can consume.

Contract
--------
    parse_tournament_brackets(soup, seeded_teams) -> list[Stage]

Each ``Stage`` is one stage container (Swiss group, GSL group, SE playoff, …)
in document order. ``stage.matches`` is a flat list of ``BracketMatch`` rows,
each carrying:

  - ``match_id`` — ``"<stage_id>::<round><slot>"`` (architecture decision #2/#10).
  - ``feeder_a`` / ``feeder_b`` — ids of upstream matches whose winners feed
    this slot. R1 matches have feeders = None and ``seed_a/seed_b`` populated.
  - ``is_bronze`` — True for the optional bronze match in ``SE_with_bronze``
    brackets; feeders point to the SFs, but the semantic is "**loser** of the
    feeder advances", not winner.
  - ``played_winner`` / ``played_date`` — set when Liquipedia has already
    resolved the slot.

Format-detection is the hybrid structural-first / heading-tiebreaker algorithm
from architecture decision #11. Day 6.3 *slice 1* implements the SE and
SE_with_bronze parsers only; double-elim, Swiss and groups follow in later
slices and currently classify as ``Stage(format="Unsupported")`` placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import re

import time

from .liquipedia_loader import _norm, _fetch_page, _bs4, _REQ_DELAY
from .format_parser import StageEntrant, StageEdge

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────────

@dataclass
class BracketMatch:
    """A single slot in a stage. R1 matches have seeds; R2+ have feeders."""
    match_id:    str
    round_idx:   int                   # 1 = first round; max round_idx = final
    slot_idx:    int                   # position within round (0-based)
    is_bronze:   bool = False          # only true for the optional 3rd-place match

    # Sub-bracket identity for DE / multi-tree formats:
    #   "" (empty)  → SE (single tree)
    #   "UB"        → upper bracket (DE)
    #   "LB"        → lower bracket (DE)
    #   "GF"        → grand final
    #   "RST"       → reset (only when LB winner takes GF1)
    sub:         str = ""

    # Round-1 seeds (canonical team names; None when Liquipedia hasn't placed them yet)
    seed_a:      str | None = None
    seed_b:      str | None = None

    # Round-2+ feeders. Each feeder slot has a kind:
    #   "winner" — the feeder's winner advances into this slot (default).
    #   "loser"  — the feeder's loser drops into this slot. Used for the
    #              SE-with-bronze 3rd-place match and for LB rounds whose
    #              "external" feeder is the UB-this-round loser.
    feeder_a:       str | None = None
    feeder_b:       str | None = None
    feeder_a_kind:  str = "winner"
    feeder_b_kind:  str = "winner"

    # Already-resolved slots (Liquipedia has the result)
    played_winner: str | None = None
    played_date:   datetime | None = None


@dataclass
class Stage:
    """One bracket stage on a tournament page."""
    stage_id:        str                       # "s0", "s1", …
    format:          str                       # "SE" | "SE_with_bronze" | "DE" | "Swiss" | "Groups" | "Unsupported"
    # Heading distinguishes context: "Group A" → DE used as group format
    # (group placements); "Playoffs" → DE used as playoff format (tournament
    # placements). Same parser, different placement-mapping in the engine layer.
    display_heading: str
    matches:         list[BracketMatch] = field(default_factory=list)

    # ── Cross-stage fields (populated by stage_graph.build_stage_graph) ──
    # Empty until Phase 2 of the bracket-system rewrite wires them; the
    # in-stage rendering / engine emission paths don't read these fields
    # so the existing bracket walkers stay clean (Q2 locked decision).
    def_name:         str = ""                 # back-ref: format_parser StageDef.name this stage came from
    entrants:         list[StageEntrant] = field(default_factory=list)
    advance_to:       list[StageEdge]    = field(default_factory=list)
    eliminations:     list[tuple[str, int]] = field(default_factory=list)
    roster:           list[str]          = field(default_factory=list)   # canonical names of teams playing this stage
    direct_invitees:  list[str]          = field(default_factory=list)   # subset of roster that entered at this stage

    @property
    def rounds(self) -> int:
        """Number of rounds in the stage. -1 when empty / unsupported."""
        if not self.matches:
            return -1
        return max(m.round_idx for m in self.matches if not m.is_bronze)

    def signature(self) -> str:
        """
        Hash of the stage's structural shape — round count, match ids, seeded
        teams, format. Used by the L2 cache invalidation rule (architecture
        decision #12 part b): if the signature flips between user sessions,
        their picks may reference altered slots → banner without auto-clear.
        """
        payload = "|".join([
            self.format,
            str(self.rounds),
            ",".join(m.match_id for m in self.matches),
            ",".join(f"{m.seed_a or '?'}-{m.seed_b or '?'}"
                     for m in self.matches if m.round_idx == 1),
        ])
        return hashlib.md5(payload.encode()).hexdigest()[:12]


# ── Format detection (architecture decision #11) ────────────────────────────────

_DE_HINT_PATTERNS = re.compile(r"lower bracket|grand final", re.IGNORECASE)
_SWISS_HINT       = re.compile(r"\bswiss\b", re.IGNORECASE)
_GROUP_HINT       = re.compile(r"\bgroup", re.IGNORECASE)


_SUB_STAGE_HEADING_RE = re.compile(
    r"^\s*(?:Group|Pool|Bracket)\s+\S+", re.IGNORECASE,
)


def _nearest_preceding_heading(el) -> str:
    """Walk back through preceding headings in proximity order. The first
    ``h2``/``h3`` is the canonical stage label. An ``h4`` only wins if it's
    the *closest* preceding heading AND matches a sub-stage discriminator
    pattern (``Group A`` / ``Pool 1`` / ``Bracket X``). Generic h4s
    (``Detailed Results``, ``Round 1``) are skipped so they don't override
    the parent h3.
    """
    cur = el.find_previous(["h2", "h3", "h4"])
    first_seen = cur
    while cur is not None:
        text = cur.get_text(" ", strip=True)
        if cur.name in ("h2", "h3"):
            return text
        # h4: only honour if it's the very first preceding heading AND it
        # matches the sub-stage discriminator pattern.
        if cur is first_seen and _SUB_STAGE_HEADING_RE.match(text or ""):
            return text
        cur = cur.find_previous(["h2", "h3", "h4"])
    return ""


def _classify_bracket_container(bracket_el, heading: str) -> str:
    """
    Decide what format a single ``.brkts-bracket`` element represents.

    Structural signals (hybrid algorithm — architecture decision #11):

    - **Two or more top-level ``.brkts-round-body`` direct children** →
      ``DE`` (one round-body for each of the upper / lower trees). This holds
      whether the DE is at playoff stage or group stage; the surrounding
      heading distinguishes the use case for placement mapping later.
    - **One round-body, top round-center has ≥2 matches** → ``SE_with_bronze``.
    - **One round-body, top round-center has 1 match** → ``SE``.
    """
    outer_bodies = bracket_el.find_all("div", class_="brkts-round-body", recursive=False)
    if not outer_bodies:
        return "Unsupported"

    if len(outer_bodies) >= 2:
        return "DE"

    outer = outer_bodies[0]
    center = outer.find("div", class_="brkts-round-center", recursive=False)
    top_matches = center.find_all("div", class_="brkts-match", recursive=False) if center else []

    if _DE_HINT_PATTERNS.search(heading):
        # Heading-driven override for DE pages where the markup happens to be
        # rendered as a single tree (rare, defensive).
        return "DE"

    if len(top_matches) >= 2:
        return "SE_with_bronze"

    return "SE"


def _matchlist_title(matchlist_el) -> str:
    """
    Pull the user-facing title from a ``.brkts-matchlist`` element. Liquipedia
    appends a hidden ``ShowHide`` toggle suffix that we strip.
    """
    title_el = matchlist_el.find("div", class_="brkts-matchlist-title")
    if title_el is None:
        return ""
    raw = title_el.get_text(" ", strip=True)
    return raw.replace("ShowHide", "").replace("Show Hide", "").strip()


def _detect_stages(soup) -> list[tuple[str, str, object]]:
    """
    Walk the page in document order, returning ``(stage_id, format, payload)``
    triples — one per recognised stage container.

    Payload shape depends on format:

    - Bracket-based stages (``SE`` / ``SE_with_bronze`` / ``DE``) → the
      ``.brkts-bracket`` element itself.
    - Matchlist-based stages (``Swiss`` / ``Groups``) → a list of
      ``.brkts-matchlist`` elements grouped by title pattern:

        * "Round N …" titles → one Swiss stage per consecutive run of such
          matchlists (PGL Astana = 1 Swiss stage spanning 9 matchlists).
        * "Group X …" titles → one Groups stage per matchlist (BLAST Rivals'
          Group A and Group B become two distinct stages).
    """
    stages: list[tuple[str, str, object]] = []
    seen: set[int] = set()
    pending_swiss: list = []  # accumulating Swiss matchlists to flush on context change

    def flush_swiss():
        if pending_swiss:
            stages.append((f"s{len(stages)}", "Swiss", list(pending_swiss)))
            pending_swiss.clear()

    for el in soup.select(".brkts-bracket, .brkts-matchlist"):
        if id(el) in seen:
            continue
        cls = el.get("class", [])

        if "brkts-bracket" in cls:
            flush_swiss()
            heading = _nearest_preceding_heading(el)
            fmt = _classify_bracket_container(el, heading)
            stages.append((f"s{len(stages)}", fmt, el))
            seen.add(id(el))
            continue

        # Matchlist branch
        title = _matchlist_title(el)
        title_lc = title.lower()
        seen.add(id(el))

        if title_lc.startswith("group "):
            # Each group is its own RR stage. Flush any pending Swiss block first.
            flush_swiss()
            stages.append((f"s{len(stages)}", "Groups", [el]))
        elif title_lc.startswith("round") or title_lc.startswith("swiss") \
                or _SWISS_HINT.search(_nearest_preceding_heading(el)):
            pending_swiss.append(el)
        else:
            # Unrecognised title — fall back to heading-based classification,
            # treating as Unsupported placeholder.
            flush_swiss()
            heading = _nearest_preceding_heading(el)
            fmt = "Groups" if _GROUP_HINT.search(heading) else (
                "Swiss" if _SWISS_HINT.search(heading) else "Unsupported"
            )
            stages.append((f"s{len(stages)}", fmt, [el]))

    flush_swiss()
    return stages


# ── SE / SE_with_bronze parser ──────────────────────────────────────────────────

def _team_from_aria(label: str) -> str | None:
    """Liquipedia aria-label is the display-name; normalise to canonical name."""
    label = (label or "").strip()
    if not label:
        return None
    return _norm(label)


def _played_result(match_el) -> tuple[str | None, datetime | None]:
    """
    If a Liquipedia match has a recorded winner, return (winner_canonical, date).
    Detection: a `.brkts-opponent-entry` carrying the ``brkts-opponent-win`` class.
    """
    winner = None
    for ent in match_el.select(".brkts-opponent-entry"):
        cls = ent.get("class", []) or []
        if any("opponent-win" in c for c in cls):
            winner = _team_from_aria(ent.get("aria-label", ""))
            break
    # Date: not always present for unplayed; try standard date extraction
    date_attr = match_el.get("data-timestamp")
    played_date = None
    if date_attr:
        try:
            played_date = datetime.fromtimestamp(int(date_attr))
        except (TypeError, ValueError):
            pass
    return winner, played_date


def _walk_se_subtree(
    outer_round_body,
    bracket_el,
    *,
    id_prefix: str,
    sub_label: str,
    is_bronze_capable: bool,
    out: list[BracketMatch],
) -> tuple[int, list[str]]:
    """
    Walk an SE-style subtree rooted at ``outer_round_body``. Used for the
    single-bracket SE/SE_with_bronze parser as well as the upper / lower
    halves of the DE parser. Append matches to ``out`` and return
    ``(top_round_idx, top_match_ids)``.

    Match-id format: ``{id_prefix}{round_idx}M{slot_idx}`` — caller supplies
    ``id_prefix`` as e.g. ``"s1::R"`` or ``"s1::UB-R"``.
    """
    round_counter: dict[int, int] = {}

    def _next_id(round_idx: int, is_bronze: bool = False) -> str:
        if is_bronze:
            return f"{id_prefix.replace('R', '')}B0"  # SE bronze marker
        slot = round_counter.get(round_idx, 0)
        round_counter[round_idx] = slot + 1
        return f"{id_prefix}{round_idx}M{slot}"

    def walk(round_body) -> tuple[int, list[str]]:
        lower = round_body.find("div", class_="brkts-round-lower", recursive=False)
        feeder_match_ids: list[str] = []
        feeder_round_idx = 0
        if lower is not None:
            feeder_bodies = lower.find_all("div", class_="brkts-round-body", recursive=False)
            for fb in feeder_bodies:
                f_round, f_ids = walk(fb)
                feeder_match_ids.extend(f_ids)
                feeder_round_idx = max(feeder_round_idx, f_round)

        round_idx = feeder_round_idx + 1
        center = round_body.find("div", class_="brkts-round-center", recursive=False)
        if center is None:
            return round_idx, []

        center_matches = center.find_all("div", class_="brkts-match", recursive=False)
        emitted_ids: list[str] = []

        for i, mel in enumerate(center_matches):
            is_top_level = (round_body is outer_round_body)
            is_bronze = (is_top_level and i > 0 and is_bronze_capable)
            mid = _next_id(round_idx, is_bronze=is_bronze)

            ent = mel.select(".brkts-opponent-entry")
            seed_a = _team_from_aria(ent[0].get("aria-label", "")) if len(ent) > 0 else None
            seed_b = _team_from_aria(ent[1].get("aria-label", "")) if len(ent) > 1 else None

            feeder_a = feeder_b = None
            kind_a = kind_b = "winner"
            if round_idx > 1 and not is_bronze:
                base = i * 2
                if base < len(feeder_match_ids):
                    feeder_a = feeder_match_ids[base]
                if base + 1 < len(feeder_match_ids):
                    feeder_b = feeder_match_ids[base + 1]
            elif is_bronze:
                if len(feeder_match_ids) >= 2:
                    feeder_a, feeder_b = feeder_match_ids[0], feeder_match_ids[1]
                    kind_a = kind_b = "loser"

            played_winner, played_date = _played_result(mel)

            out.append(BracketMatch(
                match_id=mid,
                round_idx=round_idx,
                slot_idx=i if not is_bronze else 0,
                is_bronze=is_bronze,
                sub=sub_label,
                seed_a=seed_a,
                seed_b=seed_b,
                feeder_a=feeder_a,
                feeder_b=feeder_b,
                feeder_a_kind=kind_a,
                feeder_b_kind=kind_b,
                played_winner=played_winner,
                played_date=played_date,
            ))
            if not is_bronze:
                emitted_ids.append(mid)

        return round_idx, emitted_ids

    return walk(outer_round_body)


def _apply_roster_seeds(matches: list[BracketMatch], roster: list[str]) -> None:
    """
    Phase 3 stage-graph-aware seed filler.

    Liquipedia is the source of truth for *who plays whom* in R1 — when it
    publishes seedings into the bracket DOM, ``_parse_se_bracket`` /
    ``_parse_de_bracket`` capture them. When the DOM is structurally
    rendered but slots are blank (pre-tournament), we leave R1 as TBD so
    the manual-pick dropdown owns the placement (Phase 3 Q5). Auto pair-
    ordering by roster index is misleading.

    The one place we *do* fill from roster: **bye slots** in non-power-of-
    two brackets (e.g. 6-team SE+bronze where SFs have one bye + one R1
    feeder). Without bye-fill those slots are unreachable. Top seeds get
    the byes; the manual dropdown can override.

    Skips everything when ``roster`` is empty or any R1 match already has a
    seed (Liquipedia knows better than us when it has placed teams).
    """
    main = [m for m in matches if not m.is_bronze and m.sub in ("", "UB")]
    r1 = [m for m in main if m.round_idx == 1]
    if not r1 or not roster:
        return
    if any(m.seed_a or m.seed_b for m in r1):
        return  # Liquipedia knows

    # Bye slots = R2+ slots that have no feeder *and* no seed yet, paired
    # with at least one upstream feeder on the *other* side (true byes).
    bye_slots: list[tuple[BracketMatch, str]] = []
    for m in main:
        if m.round_idx == 1:
            continue
        if not m.feeder_a and not m.seed_a and (m.feeder_b or m.seed_b):
            bye_slots.append((m, "a"))
        if not m.feeder_b and not m.seed_b and (m.feeder_a or m.seed_a):
            bye_slots.append((m, "b"))

    # Top seeds → byes (highest seeds get byes by tournament convention).
    for (m, side), seed in zip(bye_slots, roster):
        if side == "a":
            m.seed_a = seed
        else:
            m.seed_b = seed


def _parse_se_bracket(stage_id: str, fmt: str, bracket_el, seeded_teams: list[str]) -> Stage:
    """
    Single-elimination parser (with optional bronze match). See
    :func:`_walk_se_subtree` for the recursion.
    """
    heading = _nearest_preceding_heading(bracket_el)
    stage = Stage(stage_id=stage_id, format=fmt, display_heading=heading)

    outer = bracket_el.find("div", class_="brkts-round-body", recursive=False)
    if outer is None:
        return stage

    _walk_se_subtree(
        outer, bracket_el,
        id_prefix=f"{stage_id}::R",
        sub_label="",
        is_bronze_capable=(fmt == "SE_with_bronze"),
        out=stage.matches,
    )
    # Phase 3 hard-cut: seed backfill is now applied per-stage in
    # `parse_tournament_brackets` via `stage_graph.build_stage_graph` →
    # `_apply_roster_seeds(stage.matches, stage.roster)`. Per-format parsers
    # only emit raw DOM-derived matches.
    return stage


def _parse_de_bracket(stage_id: str, fmt: str, bracket_el, seeded_teams: list[str]) -> Stage:
    """
    Double-elimination parser. The bracket has two top-level round-bodies:
    the upper bracket (UB) and the lower bracket (LB). Both are walked with
    the SE subtree walker — but LB matches need explicit cross-bracket
    feeder wiring because the "external" feeder of each LB round is the
    UB-this-round loser, which the LB DOM doesn't encode itself.

    Standard 8-team DE pairing convention (used by IEM / BLAST groups):

    - LB-R1 takes UB-R1 losers in pair order (no LB feeder yet).
    - LB-R(k>1) inner side = LB-R(k-1) winner; outer side = UB-R(k) loser.
    - LB-final pairs the two LB-R(N-1) winners (no UB drop in the final
      LB round when the bracket is "compact" — like the 8-team group format
      with no Grand Final node).

    This implementation handles the compact 8-team group format used by
    IEM Atlanta / CS Asia Champs (12-match brackets, 7 UB + 5 LB, no GF).
    The full DE-with-GF + bracket-reset extension is left for slice 2b.
    """
    heading = _nearest_preceding_heading(bracket_el)
    stage = Stage(stage_id=stage_id, format=fmt, display_heading=heading)

    bodies = bracket_el.find_all("div", class_="brkts-round-body", recursive=False)
    if len(bodies) < 2:
        # Defensive: shouldn't happen — _classify_bracket_container only emits
        # "DE" when ≥2 round-bodies are present.
        return stage

    ub_tree, lb_tree = bodies[0], bodies[1]

    # Walk UB first.
    _walk_se_subtree(
        ub_tree, bracket_el,
        id_prefix=f"{stage_id}::UB-R",
        sub_label="UB",
        is_bronze_capable=False,
        out=stage.matches,
    )

    # Walk LB; treat as a smaller SE-shaped tree. LB feeders that point to
    # other LB matches are correct as-is; LB feeders that should point to UB
    # losers will be re-wired below.
    _walk_se_subtree(
        lb_tree, bracket_el,
        id_prefix=f"{stage_id}::LB-R",
        sub_label="LB",
        is_bronze_capable=False,
        out=stage.matches,
    )

    # Cross-bracket feeder wiring for LB.
    ub_by_round: dict[int, list[BracketMatch]] = {}
    lb_by_round: dict[int, list[BracketMatch]] = {}
    for m in stage.matches:
        (ub_by_round if m.sub == "UB" else lb_by_round).setdefault(m.round_idx, []).append(m)
    for r in ub_by_round:
        ub_by_round[r].sort(key=lambda x: x.slot_idx)
    for r in lb_by_round:
        lb_by_round[r].sort(key=lambda x: x.slot_idx)

    ub_max = max(ub_by_round) if ub_by_round else 0
    lb_max = max(lb_by_round) if lb_by_round else 0

    for r, lb_matches in lb_by_round.items():
        if r == 1:
            # LB-R1 inputs are the 2 UB-R1 losers per LB match (4 UB-R1 losers
            # split into 2 matches). Wire feeder_a/b → UB-R1 with kind="loser".
            ub_r1 = ub_by_round.get(1, [])
            for i, m in enumerate(lb_matches):
                base = i * 2
                if base < len(ub_r1):
                    m.feeder_a = ub_r1[base].match_id
                    m.feeder_a_kind = "loser"
                if base + 1 < len(ub_r1):
                    m.feeder_b = ub_r1[base + 1].match_id
                    m.feeder_b_kind = "loser"
                # Ensure no stale R1 seeds linger from the SE walker
                m.seed_a = m.seed_b = None
            continue

        # LB-R(k>1): inner side = LB-R(k-1) winner (already wired by SE walker
        # via the round-lower tree). Outer side = UB-R(k) loser (or, for the
        # LB-final in compact 8-team groups, the SE walker produced two LB
        # feeders since LB-final pairs two LB-SF winners — which is correct;
        # no UB drop in that case).
        prev_lb = lb_by_round.get(r - 1, [])
        ub_drop = ub_by_round.get(r, [])
        for i, m in enumerate(lb_matches):
            # Trust feeder_a (set by SE walker — points at LB-R(r-1) inner).
            # Re-wire feeder_b to UB-R(r) loser unless we're in the LB-final
            # of a "compact" group (no UB drop available at this round).
            if ub_drop and i < len(ub_drop):
                m.feeder_b = ub_drop[i].match_id
                m.feeder_b_kind = "loser"
            # Ensure feeder_a stays "winner" semantics
            m.feeder_a_kind = "winner"

    # Seed backfill happens later in `parse_tournament_brackets` via
    # the stage graph — see SE parser for the rationale.
    return stage


# ── Matchlist parser (Swiss / round-robin Groups) ──────────────────────────────

def _parse_matchlist_match(match_el):
    """
    Extract (seed_a, seed_b, played_winner) from a ``.brkts-matchlist-match``.

    The matchlist DOM uses ``.brkts-matchlist-opponent`` rows (not
    ``.brkts-opponent-entry`` like the bracket DOM) — different selectors but
    same semantics: aria-label = team display name; winning row carries an
    ``opponent-win`` class.
    """
    ents = match_el.select(".brkts-matchlist-opponent")
    seed_a = _team_from_aria(ents[0].get("aria-label", "")) if len(ents) > 0 else None
    seed_b = _team_from_aria(ents[1].get("aria-label", "")) if len(ents) > 1 else None
    winner = None
    for ent in ents:
        cls = ent.get("class", []) or []
        if any("opponent-win" in c for c in cls):
            winner = _team_from_aria(ent.get("aria-label", ""))
            break
    return seed_a, seed_b, winner


_ROUND_RE = re.compile(r"round\s*(\d+)", re.IGNORECASE)


def _parse_swiss_stage(stage_id: str, matchlist_els: list, seeded_teams: list[str]) -> Stage:
    """
    Swiss-format stage. Each matchlist title is "Round N <Pool>", e.g.
    ``"Round 2 High"``, ``"Round 2 Low"``, ``"Round 3 Mid"``. Matches within a
    pool are independent; the round number drives display ordering.

    Match IDs: ``s0::SW-R{round}M{global_slot}`` — global slot is unique
    across the stage. ``sub`` = ``"SW"``.
    """
    stage = Stage(stage_id=stage_id, format="Swiss",
                  display_heading=_nearest_preceding_heading(matchlist_els[0]) if matchlist_els else "")
    global_slot_counter: dict[int, int] = {}

    for ml in matchlist_els:
        title = _matchlist_title(ml)
        m_round = _ROUND_RE.search(title)
        round_idx = int(m_round.group(1)) if m_round else 1
        # "High" / "Low" / "Mid" / "Decider" — preserved in title for display
        pool = ""
        for tok in ("High", "Low", "Mid", "Decider", "Initial", "Elimination", "Advancement"):
            if tok in title:
                pool = tok
                break

        for mel in ml.select(".brkts-matchlist-match"):
            seed_a, seed_b, played_winner = _parse_matchlist_match(mel)
            slot = global_slot_counter.get(round_idx, 0)
            global_slot_counter[round_idx] = slot + 1

            stage.matches.append(BracketMatch(
                match_id=f"{stage_id}::SW-R{round_idx}M{slot}",
                round_idx=round_idx,
                slot_idx=slot,
                sub="SW",
                seed_a=seed_a,
                seed_b=seed_b,
                played_winner=played_winner,
            ))
            # Stash pool label in display_heading suffix for one match — non-essential
            if pool and slot == 0 and round_idx not in {m.round_idx for m in stage.matches[:-1]}:
                pass  # pool tracked implicitly via slot order

    # Phase 3: per-tournament-organiser pairings are the source of truth. When
    # Liquipedia hasn't placed Swiss R1 teams yet, slots stay TBD so the
    # manual-pick dropdown / Phase 4 cascade owns them. (Old "Standard 1v9 /
    # 2v10 / …" pair-order was misleading — real first-round seeding is
    # rarely strict high-vs-low.)
    return stage


def _parse_groups_stage(stage_id: str, matchlist_els: list, seeded_teams: list[str]) -> Stage:
    """
    Group-stage matchlist. Liquipedia uses three layouts inside ``Group X``
    matchlists; we detect via the ``.brkts-matchlist-header`` sub-section
    labels:

    * **GSL-lite** (BLAST Rivals): ``Opening Matches`` × 2 → ``Winners Match``
      (R1-winners pairing) + ``Elimination Match`` (R1-losers pairing). 4
      matches total. Top 2 advance via Opening-Match wins; Winners' Match
      seeds the playoff bracket, Elimination breaks the 3rd-place tie.
    * **Full GSL**: adds a ``Decider Match`` between Winners-Match-loser
      and Elimination-Match-winner. 5 matches total.
    * **Round-robin** (no sub-headers): every team plays every other; we
      fall through to the RR cascade that fills missing pairings.

    Match IDs / round numbers depend on the detected layout:
    * Opening = ``s0::GR-O{i}`` (round 1).
    * Winners = ``s0::GR-WM`` (round 2, feeder_kind=winner).
    * Elimination = ``s0::GR-EM`` (round 2, feeder_kind=loser).
    * Decider = ``s0::GR-DM`` (round 3, mixed feeders).
    * RR fallback = ``s0::GR-M{i}`` (round 1).

    All match flavours land in the same ``Stage`` with ``sub="GR"``; the
    UI's generic ``_resolve`` walker handles them via ``feeder_a_kind`` /
    ``feeder_b_kind`` like LB cross-feeds in DE.
    """
    if not matchlist_els:
        return Stage(stage_id=stage_id, format="Groups", display_heading="")

    ml = matchlist_els[0]
    title = _matchlist_title(ml)
    base_heading = title.replace("Matches", "").strip() or _nearest_preceding_heading(ml)

    # ── Walk matchlist children, tagging each match with its sub-header ─
    container = ml.find("div", class_="should-collapse") or ml
    sections: list[tuple[str, object]] = []  # [(section_label, match_el), …]
    current_label = ""
    for child in container.find_all("div", recursive=False):
        cls = child.get("class", []) or []
        if "brkts-matchlist-header" in cls:
            current_label = child.get_text(" ", strip=True)
        elif "brkts-matchlist-match" in cls:
            sections.append((current_label, child))

    # Group by section label (case-insensitive)
    by_label: dict[str, list] = {}
    for label, mel in sections:
        by_label.setdefault(label.lower(), []).append(mel)

    opening = by_label.get("opening matches", []) or by_label.get("opening match", [])
    winners = by_label.get("winners match", [])  or by_label.get("winners' match", [])
    elim    = by_label.get("elimination match", [])
    decider = by_label.get("decider match", [])

    has_gsl = bool(opening) and bool(winners or elim or decider)

    # ── GSL-lite / Full GSL branch ──────────────────────────────────────
    if has_gsl:
        subformat = "GSL-lite" if not decider else "GSL"
        stage = Stage(
            stage_id=stage_id, format="Groups",
            display_heading=f"{base_heading} ({subformat})",
        )

        # Opening matches — round 1, seeds populated from aria-label.
        opening_ids: list[str] = []
        for i, mel in enumerate(opening[:2]):
            seed_a, seed_b, played_winner = _parse_matchlist_match(mel)
            mid = f"{stage_id}::GR-O{i}"
            opening_ids.append(mid)
            stage.matches.append(BracketMatch(
                match_id=mid,
                round_idx=1,
                slot_idx=i,
                sub="GR",
                seed_a=seed_a,
                seed_b=seed_b,
                played_winner=played_winner,
            ))

        # Winners Match — round 2, feeders = both opening winners.
        wm_id = None
        if winners:
            mel = winners[0]
            _, _, played_winner = _parse_matchlist_match(mel)
            wm_id = f"{stage_id}::GR-WM"
            stage.matches.append(BracketMatch(
                match_id=wm_id,
                round_idx=2,
                slot_idx=0,
                sub="GR",
                feeder_a=opening_ids[0] if len(opening_ids) > 0 else None,
                feeder_b=opening_ids[1] if len(opening_ids) > 1 else None,
                feeder_a_kind="winner",
                feeder_b_kind="winner",
                played_winner=played_winner,
            ))

        # Elimination Match — round 2, feeders = both opening losers.
        em_id = None
        if elim:
            mel = elim[0]
            _, _, played_winner = _parse_matchlist_match(mel)
            em_id = f"{stage_id}::GR-EM"
            stage.matches.append(BracketMatch(
                match_id=em_id,
                round_idx=2,
                slot_idx=1,
                sub="GR",
                feeder_a=opening_ids[0] if len(opening_ids) > 0 else None,
                feeder_b=opening_ids[1] if len(opening_ids) > 1 else None,
                feeder_a_kind="loser",
                feeder_b_kind="loser",
                played_winner=played_winner,
            ))

        # Decider Match (full-GSL only) — round 3, WM-loser vs EM-winner.
        if decider and wm_id and em_id:
            mel = decider[0]
            _, _, played_winner = _parse_matchlist_match(mel)
            stage.matches.append(BracketMatch(
                match_id=f"{stage_id}::GR-DM",
                round_idx=3,
                slot_idx=0,
                sub="GR",
                feeder_a=wm_id,
                feeder_b=em_id,
                feeder_a_kind="loser",
                feeder_b_kind="winner",
                played_winner=played_winner,
            ))

        return stage

    # ── Round-robin branch (no sub-headers) ─────────────────────────────
    stage = Stage(
        stage_id=stage_id, format="Groups",
        display_heading=f"{base_heading} (Round-robin)",
    )
    for i, mel in enumerate(ml.select(".brkts-matchlist-match")):
        seed_a, seed_b, played_winner = _parse_matchlist_match(mel)
        stage.matches.append(BracketMatch(
            match_id=f"{stage_id}::GR-M{i}",
            round_idx=1,
            slot_idx=i,
            sub="GR",
            seed_a=seed_a,
            seed_b=seed_b,
            played_winner=played_winner,
        ))

    teams_seen: list[str] = []
    seen_set: set[str] = set()
    for m in stage.matches:
        for t in (m.seed_a, m.seed_b):
            if t and t not in seen_set:
                seen_set.add(t)
                teams_seen.append(t)

    if len(teams_seen) >= 3:
        expected_pairs: set[frozenset] = set()
        for i in range(len(teams_seen)):
            for j in range(i + 1, len(teams_seen)):
                expected_pairs.add(frozenset((teams_seen[i], teams_seen[j])))
        existing_pairs: set[frozenset] = set()
        for m in stage.matches:
            if m.seed_a and m.seed_b:
                existing_pairs.add(frozenset((m.seed_a, m.seed_b)))
        missing_pairs_sorted = sorted(
            (sorted(p) for p in (expected_pairs - existing_pairs)),
            key=lambda p: (teams_seen.index(p[0]), teams_seen.index(p[1])),
        )

        empty_slots = [m for m in stage.matches if not m.seed_a or not m.seed_b]
        for m in empty_slots:
            if not missing_pairs_sorted:
                break
            pair = missing_pairs_sorted.pop(0)
            m.seed_a, m.seed_b = pair[0], pair[1]

        next_slot = len(stage.matches)
        for pair in missing_pairs_sorted:
            stage.matches.append(BracketMatch(
                match_id=f"{stage_id}::GR-M{next_slot}",
                round_idx=1,
                slot_idx=next_slot,
                sub="GR",
                seed_a=pair[0],
                seed_b=pair[1],
            ))
            next_slot += 1

    return stage


# ── Stage builder ───────────────────────────────────────────────────────────────

def _build_stage(stage_id: str, fmt: str, container, seeded_teams: list[str]) -> Stage:
    """Dispatch on format → the right per-format parser. Used by the main page
    walker and the sub-page walker so both share the same code path."""
    if fmt in ("SE", "SE_with_bronze"):
        return _parse_se_bracket(stage_id, fmt, container, seeded_teams)
    if fmt == "DE":
        return _parse_de_bracket(stage_id, fmt, container, seeded_teams)
    if fmt == "Swiss":
        return _parse_swiss_stage(stage_id, container, seeded_teams)
    if fmt == "Groups":
        return _parse_groups_stage(stage_id, container, seeded_teams)
    # Unsupported — empty placeholder.
    ref = container[0] if isinstance(container, list) and container else container
    heading = _nearest_preceding_heading(ref) if ref is not None else ""
    return Stage(stage_id=stage_id, format=fmt, display_heading=heading)


# ── Public entry point ──────────────────────────────────────────────────────────

_SUB_STAGE_HREF_RE = re.compile(r"^/counterstrike/(.+?)/Stage_(\d+)$")


def _discover_sub_stage_slugs(soup, base_slug: str | None) -> list[tuple[int, str]]:
    """
    Liquipedia splits very large events (Majors) across sub-pages — Stage_1 /
    Stage_2 / Stage_3 host the Swiss / group stages while the main page only
    carries the SE playoff bracket. Find the ``/Stage_N`` links that share
    ``base_slug`` and return them sorted by stage number.

    Returns ``[(N, full_slug)]`` — e.g. ``[(1, "Intel_Extreme_Masters/2026/
    Cologne/Stage_1"), (2, "…/Stage_2"), …]``.
    """
    if not base_slug:
        return []
    seen: dict[int, str] = {}
    for a in soup.select('a[href]'):
        href = a.get("href", "")
        m = _SUB_STAGE_HREF_RE.match(href)
        if not m:
            continue
        if m.group(1) != base_slug:
            continue
        try:
            n = int(m.group(2))
        except ValueError:
            continue
        seen[n] = f"{m.group(1)}/Stage_{n}"
    return sorted(seen.items())


def parse_tournament_brackets(
    soup,
    seeded_teams: list[str] | None = None,
    slug: str | None = None,
    *,
    format_prose: str | None = None,
    direct_invitees_by_stage: dict[str, list[str]] | None = None,
    sub_page_rosters: dict[str, list[str]] | None = None,
) -> list[Stage]:
    """
    Parse all bracket stages on a Liquipedia tournament page.

    Parameters
    ----------
    soup
        BeautifulSoup of the tournament page (already fetched via
        ``liquipedia_loader._fetch_page`` + ``_bs4``).
    seeded_teams
        Canonical team names of the tournament's participant list. Used as
        the initial-stage roster when no per-stage invitee partition exists
        (single-stage / non-Major events).
    slug
        The page's slug — when supplied, the parser also discovers and fetches
        ``/Stage_N`` sub-pages (Liquipedia's convention for Major-tier events
        where Stage 1/2/3 live as separate pages and only the SE playoff
        bracket lives on the main page). Sub-page stages are *prepended* to
        the main page's stages so the timeline reads in execution order.
    format_prose
        Liquipedia ``<h3>Format</h3>`` prose. When given, parsed into
        ``StageDef``s and used to populate each ``Stage``'s cross-stage
        fields (``def_name``, ``entrants``, ``advance_to``, ``eliminations``)
        and to drive R1 seed backfill from the resolved per-stage roster.
        When ``None``, R1 seeds remain whatever Liquipedia placed in the DOM
        (Phase 3 hard-cut: no blanket backfill).
    direct_invitees_by_stage
        Optional ``{stage_name: [teams]}`` from
        ``liquipedia_loader._parse_per_stage_invites`` — partitions Major
        events' main-page Participants section into per-stage invitee lists.
    sub_page_rosters
        Optional ``{stage_name: [teams]}`` from
        ``stage_graph.collect_sub_page_rosters`` — full-roster teamcards
        scraped from each ``/Stage_N`` sub-page.

    Returns
    -------
    list[Stage]
        One ``Stage`` per recognised stage container, in execution order,
        with cross-stage fields populated when ``format_prose`` is supplied.
    """
    seeded_teams = seeded_teams or []
    stages: list[Stage] = []

    # ── Sub-page discovery (Major-tier events only) ──────────────────────
    sub_slugs: list[tuple[int, str]] = _discover_sub_stage_slugs(soup, slug)
    for i, (n, sub_slug) in enumerate(sub_slugs):
        try:
            sub_html = _fetch_page(sub_slug)
        except Exception as exc:
            logger.warning("Sub-page fetch failed for %s: %s", sub_slug, exc)
            continue
        if sub_html is None:
            logger.warning("Sub-page returned None: %s", sub_slug)
            continue
        sub_soup = _bs4(sub_html)
        sub_triples = _detect_stages(sub_soup)
        for stage_id, fmt, container in sub_triples:
            sub_stage = _build_stage(stage_id, fmt, container, seeded_teams)
            # Tag the display heading with the stage number for UI clarity.
            if not sub_stage.display_heading.lower().startswith("stage"):
                sub_stage.display_heading = (
                    f"Stage {n}" + (f" — {sub_stage.display_heading}"
                                    if sub_stage.display_heading else "")
                )
            stages.append(sub_stage)
        # Be polite between sub-page requests.
        if i < len(sub_slugs) - 1:
            time.sleep(_REQ_DELAY)

    # ── Main page stages ─────────────────────────────────────────────────
    triples = _detect_stages(soup)
    for stage_id, fmt, container in triples:
        stages.append(_build_stage(stage_id, fmt, container, seeded_teams))

    # ── Stage graph + roster-driven seed backfill (Phase 3 hard-cut) ─────
    # Only when format_prose is supplied. Without prose, we can't link DOM
    # stages to a cross-stage model — Stages keep raw seeds (whatever
    # Liquipedia placed) and R1 leaves unset slots as TBD.
    if format_prose:
        # Local import keeps `bracket_parser` independent of `stage_graph`
        # at module-load time and dodges any future circular-import risk.
        from .format_parser import parse_format_prose
        from .stage_graph import build_stage_graph

        stage_defs, _prose_warnings = parse_format_prose(format_prose)
        build_stage_graph(
            stages, stage_defs, seeded_teams,
            sub_page_rosters=sub_page_rosters,
            direct_invitees_by_stage=direct_invitees_by_stage,
        )
        for stage in stages:
            _apply_roster_seeds(stage.matches, stage.roster)

    # Merge consecutive matchlist-derived placeholders with the same heading.
    # A Swiss page emits one container per round (10 rounds = 10 matchlists);
    # those should collapse into one placeholder card. Bracket-derived stages
    # (SE / SE_with_bronze / DE) stay distinct even with shared headings —
    # IEM Atlanta's "Group Stage" produces two separate DE groups, both must
    # render and contribute to picks independently.
    merged: list[Stage] = []
    _BRACKET_FORMATS = {"SE", "SE_with_bronze", "DE"}
    for s in stages:
        if (
            merged
            and s.format not in _BRACKET_FORMATS
            and merged[-1].format == s.format
            and merged[-1].display_heading == s.display_heading
        ):
            continue
        merged.append(s)
    stages = merged

    # Re-number stage_ids contiguously after the merge so downstream pick keys
    # stay clean (no gaps if the parser ever surfaces a stage we then drop).
    for i, s in enumerate(stages):
        s.stage_id = f"s{i}"
        for m in s.matches:
            # Rewrite match_ids to keep the new stage_id prefix consistent.
            _, _, suffix = m.match_id.partition("::")
            m.match_id = f"{s.stage_id}::{suffix}"
            if m.feeder_a:
                _, _, fa_suffix = m.feeder_a.partition("::")
                m.feeder_a = f"{s.stage_id}::{fa_suffix}"
            if m.feeder_b:
                _, _, fb_suffix = m.feeder_b.partition("::")
                m.feeder_b = f"{s.stage_id}::{fb_suffix}"

    logger.info(
        "Parsed %d stage(s): %s",
        len(stages),
        ", ".join(f"{s.stage_id}={s.format}({len(s.matches)}m)" for s in stages),
    )
    return stages
