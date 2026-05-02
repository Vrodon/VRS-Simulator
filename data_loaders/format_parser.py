"""
Format Prose Parser
===================
Extracts structured stage definitions from Liquipedia tournament-page
``<h3>Format</h3>`` prose. The parser feeds Phase 2's stage-graph
construction (``NEXT_STEPS_BRACKETS.md`` Q2/Q3) and ultimately drives
cross-stage R1 seed population, advancement cascade, and prize emission.

Strategy (locked, Q1)
---------------------
Template-based regex with **flag-on-deviation**. We match Liquipedia's
standard format-section phrasing (`"Top N proceed to X"`, `"Bottom M
eliminated"`, `"N Team Swiss System Format"`, `"Single-Elimination
bracket"`, `"The Grand Final is Bo5"`, `"Each group has K teams"`, …).
Anything that doesn't match a template lands in ``StageDef.warnings``;
the UI surfaces those as 🟠 deviation banners so the user can manually
mark direct invitees / advancement edges (Q3 fallback).

Cross-stage shape (Q2)
----------------------
``StageDef`` carries *cross-stage* flow (entrants, advance_to,
eliminations). ``BracketMatch`` (in ``bracket_parser``) keeps its
*within-stage* feeders untouched. The two are linked at render time
when the stage graph resolves entrants for each stage's R1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


# ── Dataclasses ─────────────────────────────────────────────────────────────────

@dataclass
class StageEntrant:
    """One entrant source contributing to a stage's R1 pool."""
    source: str                       # "initial_roster" | "advance_from" | "direct_invite"
                                      # | "eliminate_from" | "manual_seed"
    upstream_stage: str | None = None
    count:    int = 0
    criterion: str = ""               # "top_by_wins" | "placement" | "group_rank" | "all"
    seeding:  str | None = None       # optional positional overlay name
    notes:    str = ""


@dataclass
class StageEdge:
    """One advancement edge from this stage to a downstream stage."""
    target_stage: str
    count:     int
    criterion: str                    # "top_by_wins" | "placement" | "group_rank" | "all"
    seeding:   str | None = None
    notes:     str = ""


@dataclass
class StageDef:
    """
    Structured definition of a single tournament stage as extracted from
    Liquipedia format prose.

    ``format`` enums: ``"Swiss" | "SE" | "SE_with_bronze" | "DE" | "GSL" | "RR" | "Unknown"``.

    For multi-group formats (e.g. "Two double-elimination format (GSL) Groups
    Each group has 8 teams"), ``n_groups`` and ``teams_per_group`` populate
    independently. Total teams = ``n_groups * teams_per_group`` when set.
    """
    name:      str
    team_count: int = 0
    format:    str = "Unknown"
    n_groups:  int = 1
    teams_per_group: int = 0
    bo:        str = ""               # display string: e.g. "Bo3 + GF Bo5"
    entrants:    list[StageEntrant]   = field(default_factory=list)
    advance_to:  list[StageEdge]      = field(default_factory=list)
    eliminations: list[tuple[str, int]] = field(default_factory=list)
    warnings:    list[str]            = field(default_factory=list)


# ── Regex templates ─────────────────────────────────────────────────────────────
#
# Each template captures one phrase pattern Liquipedia uses. New phrasings get
# added here as they're encountered (template + flag-on-deviation, Q1).

# Stage delimiters: "·" splits stage blocks; "Stage N:" / "Group Stage:" /
# "Playoffs:" / "Quarterfinals:" / "Semifinals:" mark stage starts.
_RE_STAGE_DELIM = re.compile(r"\s*·\s*")
_STAGE_HEADER_KEYWORDS = (
    r"Stage\s*\d+",
    r"Group\s*Stage",
    r"Play-?[Ii]ns?",
    r"Playoffs",
    r"Open\s*Qualifier",
    r"Closed\s*Qualifier",
    r"Last[-\s]Chance\s*Qualifier",
    r"Repechage",
    r"Main\s*Event",
    r"Final\s*Stage",
)
_RE_STAGE_HEADER = re.compile(
    r"^\s*(" + r"|".join(_STAGE_HEADER_KEYWORDS) + r")\s*[:\.]",
    re.IGNORECASE,
)

# Format identification within a stage.
_RE_TEAM_FORMAT = re.compile(
    r"(\d+)\s*Team\s*"
    r"(Swiss|Single[-\s]?Elimination|Double[-\s]?Elimination|Round[-\s]?Robin|GSL)",
    re.IGNORECASE,
)
_RE_SE_BRACKET = re.compile(r"Single[-\s]?Elimination\s*bracket", re.IGNORECASE)
_RE_DE_BRACKET = re.compile(r"Double[-\s]?Elimination\s*bracket", re.IGNORECASE)
_RE_BRONZE     = re.compile(r"\bbronze\b|\b3rd\s+place\s+match\b", re.IGNORECASE)

# Multi-group hints. "Two double-elimination format (GSL) Groups" / "Each
# group has 4 teams". Plural-only ``Groups`` (capital G) — singular
# ``group`` is used in distractor phrases like "top three teams from each
# group advance" and would over-match. The 0-6 modifier-token bound stops
# the regex from spanning sentence boundaries.
_RE_GROUP_COUNT = re.compile(
    # Single-digit only — multi-digit would match "2026" in dates like
    # "April 29 - 30, 2026 Two ... Groups". CS2 tournaments never have
    # more than 8 groups in practice.
    r"\b(One|Two|Three|Four|Five|Six|Seven|Eight|[1-9])\s+"
    r"(?:\S+\s+){0,6}?"
    r"Groups\b",
)
_RE_GROUP_TEAMS = re.compile(
    r"Each\s+group\s+has\s+(\w+)\s+teams?",
    re.IGNORECASE,
)
_RE_GSL_HINT = re.compile(r"\(GSL\)|\bGSL\b", re.IGNORECASE)
_RE_DE_HINT  = re.compile(r"double[-\s]?elimination", re.IGNORECASE)

# Advancement / elimination counts.
#   "Top 8 Teams proceed to Stage 2"
#   "Top 8 Teams proceed to the Playoffs"
#   "Top 8 teams proceed to Playoffs"
# Target-stage capture: a non-greedy run of CapitalisedWord (+ digit |
# + CapitalisedWord) terminated by a known sentence-boundary lookahead.
# The capitalised-letter requirement is **case-sensitive** (wrapped in
# `(?-i:…)`) so the parent regex's IGNORECASE doesn't dilute it — that
# would cause lowercase "the" to be captured as a target stage.
_TARGET_STAGE = (
    r"(?-i:([A-Z][A-Za-z]+(?:\s+(?:[A-Z][A-Za-z]+|\d+))*?))"
    r"(?=\s+(?:Bottom|Top|Click|Group|All|The|Each|Opening|"
    r"Elimination|Advancement|Matches?|Round)\b"
    r"|\s*[.,:;]|\s*$)"
)

_RE_TOP_PROCEED = re.compile(
    r"Top\s+(\d+|\w+)\s+[Tt]eams?\s+(?:proceed|advance)\s+to\s+"
    r"(?:the\s+)?" + _TARGET_STAGE,
    re.IGNORECASE,
)
#   "The top three teams from each group advance to the Playoffs"
_RE_TOP_PER_GROUP = re.compile(
    r"(?:[Tt]he\s+)?top\s+(\d+|\w+)\s+teams?\s+from\s+each\s+group\s+"
    r"(?:advance|proceed)\s+to\s+(?:the\s+)?" + _TARGET_STAGE,
    re.IGNORECASE,
)
#   "Bottom 8 Teams are eliminated"
_RE_BOTTOM_ELIM = re.compile(
    r"Bottom\s+(\d+|\w+)\s+[Tt]eams?\s+(?:are\s+)?eliminated",
    re.IGNORECASE,
)
# Group-stage placement-specific routing (BLAST/IEM convention):
#   "Group stage winners advance to the Semifinals"
#   "Group stage runners-up advance to the Quarterfinals as the High Seeds"
#   "Group stage 3 rd place teams advance to the Quarterfinals as the Low Seeds"
_RE_GROUP_PLACEMENT_TO = re.compile(
    r"Group\s+stage\s+"
    r"(winners?|runners?[-\s]?up|\d+\s*(?:st|nd|rd|th)?\s*place(?:\s+teams?)?)\s+"
    r"advance\s+to\s+(?:the\s+)?" + _TARGET_STAGE +
    r"(?:\s+as\s+(?:the\s+)?(High|Low|First|Second|Third|Fourth)\s+Seeds?)?",
    re.IGNORECASE,
)

# Bo / GF Bo
_RE_BO    = re.compile(r"\bBo([135])\b")
_RE_GF_BO = re.compile(r"Grand\s*Final\s+is\s+Bo([135])", re.IGNORECASE)


# Number-word table (Liquipedia uses both digits and words)
_NUM_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "twenty": 20, "twenty-four": 24, "thirty-two": 32,
}


def _to_int(token: str) -> int | None:
    """Parse an integer from a digit string or English number word."""
    if token is None:
        return None
    t = token.strip().lower().rstrip(".,:")
    if t.isdigit():
        return int(t)
    return _NUM_WORDS.get(t)


# ── Stage-name canonicalisation ─────────────────────────────────────────────────

def _canon_stage_name(raw: str) -> str:
    """
    Strip leading articles, trailing punctuation, and collapse whitespace so
    cross-references between stages line up.

    "the Playoffs" → "Playoffs"
    "Stage 2 "     → "Stage 2"
    """
    s = re.sub(r"^\s*the\s+", "", raw, flags=re.IGNORECASE)
    s = s.strip().rstrip(",.:")
    return re.sub(r"\s+", " ", s)


# ── Per-stage parser ────────────────────────────────────────────────────────────

def _parse_stage_chunk(chunk: str) -> StageDef | None:
    """
    Parse one stage chunk (text between two ``·`` delimiters) into a StageDef.

    Returns None when no recognisable stage header is present at the start
    of the chunk — the caller logs that as a top-level warning.
    """
    chunk = chunk.strip()
    if not chunk:
        return None

    m = _RE_STAGE_HEADER.match(chunk)
    if m is None:
        return None
    name = _canon_stage_name(m.group(1))

    body = chunk[m.end():].strip()
    stage = StageDef(name=name)

    # ── Format identification ───────────────────────────────────────────
    m_fmt = _RE_TEAM_FORMAT.search(body)
    if m_fmt:
        stage.team_count = int(m_fmt.group(1))
        kw = m_fmt.group(2).lower().replace("-", "").replace(" ", "")
        if kw.startswith("swiss"):
            stage.format = "Swiss"
        elif kw.startswith("singleelim"):
            stage.format = "SE"
        elif kw.startswith("doubleelim"):
            stage.format = "DE"
        elif kw.startswith("roundrobin"):
            stage.format = "RR"
        elif kw == "gsl":
            stage.format = "GSL"

    # Multi-group formats override a top-level "N Team …" if present.
    m_grps = _RE_GROUP_COUNT.search(body)
    m_per  = _RE_GROUP_TEAMS.search(body)
    if m_grps and m_per:
        n_groups = _to_int(m_grps.group(1))
        per_grp  = _to_int(m_per.group(1))
        if n_groups and per_grp:
            stage.n_groups = n_groups
            stage.teams_per_group = per_grp
            stage.team_count = n_groups * per_grp
            # Decide format from the keywords inside the multi-group phrase.
            if _RE_GSL_HINT.search(body) or _RE_DE_HINT.search(body):
                # GSL keyword + 4 teams/group ⇒ GSL (DOM decides GSL vs GSL-lite).
                # 8 teams/group ⇒ compact 8-team DE-as-group.
                stage.format = "GSL" if per_grp == 4 else "DE"
            elif "single-elimination" in body.lower():
                stage.format = "SE"
            elif "round-robin" in body.lower() or "round robin" in body.lower():
                stage.format = "RR"

    # SE / SE+bronze fallback when no "N Team" prefix.
    if stage.format == "Unknown":
        if _RE_SE_BRACKET.search(body):
            stage.format = "SE_with_bronze" if _RE_BRONZE.search(body) else "SE"
        elif _RE_DE_BRACKET.search(body):
            stage.format = "DE"

    # Bronze upgrade (if format is SE and bronze keyword exists).
    if stage.format == "SE" and _RE_BRONZE.search(body):
        stage.format = "SE_with_bronze"

    # ── Advancement edges ───────────────────────────────────────────────
    # Generic "Top N → STAGE" (single-group flow)
    for m in _RE_TOP_PROCEED.finditer(body):
        n = _to_int(m.group(1))
        target = _canon_stage_name(m.group(2))
        if n and target:
            criterion = "top_by_wins" if stage.format == "Swiss" else "placement"
            stage.advance_to.append(StageEdge(
                target_stage=target,
                count=n,
                criterion=criterion,
            ))

    # "Top N from each group → STAGE" (multi-group flow). Translates to a single
    # edge with count = N * n_groups (architecturally simpler — the seeding
    # overlay carries the per-group placement breakdown).
    for m in _RE_TOP_PER_GROUP.finditer(body):
        n_per = _to_int(m.group(1))
        target = _canon_stage_name(m.group(2))
        if n_per and target and stage.n_groups:
            stage.advance_to.append(StageEdge(
                target_stage=target,
                count=n_per * stage.n_groups,
                criterion="group_rank",
                notes=f"top {n_per} per group × {stage.n_groups} groups",
            ))

    # Group-placement-specific routing (overlay on top of the multi-group edge).
    # We attach these as notes on the most recently added group-rank edge so the
    # downstream seeding resolver can consume them.
    placement_overlays: list[str] = []
    for m in _RE_GROUP_PLACEMENT_TO.finditer(body):
        place_raw = m.group(1).strip().lower()
        target    = _canon_stage_name(m.group(2))
        seed_lbl  = (m.group(3) or "").strip().lower() or None
        if "winner" in place_raw:
            place_norm = "1st"
        elif "runner" in place_raw:
            place_norm = "2nd"
        else:
            num_match = re.match(r"(\d+)", place_raw)
            place_norm = f"{num_match.group(1)}th" if num_match else place_raw
        seed_str = f" as {seed_lbl}-seed" if seed_lbl else ""
        placement_overlays.append(f"{place_norm}-in-group -> {target}{seed_str}")

    # Annotate the latest group-rank edge with the overlays.
    if placement_overlays and stage.advance_to:
        last_grp_edge = next(
            (e for e in reversed(stage.advance_to) if e.criterion == "group_rank"),
            None,
        )
        if last_grp_edge is not None:
            existing = last_grp_edge.notes or ""
            last_grp_edge.notes = (existing + "; " if existing else "") + ", ".join(placement_overlays)
            last_grp_edge.seeding = "group_placement_overlay"

    # ── Eliminations ────────────────────────────────────────────────────
    for m in _RE_BOTTOM_ELIM.finditer(body):
        n = _to_int(m.group(1))
        if n:
            stage.eliminations.append((f"bottom-{n}", n))

    # ── Bo display string ───────────────────────────────────────────────
    bo_set = sorted({m.group(1) for m in _RE_BO.finditer(body)})
    gf = _RE_GF_BO.search(body)
    bo_parts: list[str] = []
    if bo_set:
        bo_parts.append("Bo" + "/".join(bo_set))
    if gf:
        bo_parts.append(f"GF Bo{gf.group(1)}")
    stage.bo = " · ".join(bo_parts)

    # ── Sanity warnings ─────────────────────────────────────────────────
    if stage.format == "Unknown":
        stage.warnings.append("could not identify format from prose")
    if stage.team_count == 0:
        stage.warnings.append("could not identify team count from prose")
    if not stage.advance_to and not stage.eliminations and "playoffs" not in name.lower():
        # Terminal stage (Playoffs / Grand Final) is allowed to lack
        # advance_to. Anything else missing both signals is suspicious.
        stage.warnings.append("no advancement edge and no elimination found")

    return stage


# ── Public entry point ──────────────────────────────────────────────────────────

def parse_format_prose(prose: str) -> tuple[list[StageDef], list[str]]:
    """
    Parse Liquipedia format-section prose into structured stage definitions.

    Parameters
    ----------
    prose : str
        Full untruncated text of the ``<h3>Format</h3>`` section, with stage
        chunks delimited by `` · `` (the joiner used in
        ``liquipedia_loader._parse_format``).

    Returns
    -------
    (stages, warnings)
        ``stages`` is a list of ``StageDef`` in execution order.
        ``warnings`` lists global parser issues (chunks without a stage
        header, unresolved cross-stage references, etc.). Per-stage parser
        notes live in ``StageDef.warnings``.
    """
    if not prose or not prose.strip():
        return [], ["empty format prose"]

    warnings: list[str] = []
    # Liquipedia normally separates stage blocks with `·`, but for some events
    # (IEM Atlanta, CS Asia Championships) it omits the separator and the
    # next stage's header runs into the previous block. Insert a synthetic
    # `·` before every stage-header keyword **followed by a colon** — that
    # `[:\.]` requirement is what distinguishes a real stage header (`Stage 1:`)
    # from a cross-reference (`proceed to Stage 2 Bottom...`). Without it
    # we'd split on every distractor mention.
    #
    # Negative lookbehinds skip the descriptive use ("...advance to the
    # Playoffs:" — Liquipedia uses the colon there as an in-line clause
    # marker, not as a stage header). "to" + "the" cover the common
    # phrasings; new ones get added here as we encounter them.
    boundary = re.compile(
        r"(?<!to)(?<!the)\s+"
        r"(?=(?:" + r"|".join(_STAGE_HEADER_KEYWORDS) + r")\s*[:\.])",
        re.IGNORECASE,
    )
    prose_norm = boundary.sub(" · ", prose)
    chunks = _RE_STAGE_DELIM.split(prose_norm)
    stages: list[StageDef] = []

    for chunk in chunks:
        stage = _parse_stage_chunk(chunk)
        if stage is None:
            if chunk.strip():
                warnings.append(f"unrecognised chunk (no stage header): {chunk.strip()[:80]}")
            continue
        stages.append(stage)

    # Same-name dedup. Some Liquipedia format prose uses a stage name as a
    # descriptive colon-clause ("...advance to the Playoffs: Group winners
    # advance to the Semifinals...") in addition to the actual stage header.
    # Both look like stage starts to the boundary regex. Pick the entry
    # with the most information (real format, populated team count, real
    # advancement edges) and drop the rest. Preserves first-seen order.
    seen_order: list[str] = []
    best: dict[str, StageDef] = {}

    def _score(s: StageDef) -> tuple:
        return (
            s.format != "Unknown",
            s.team_count,
            len(s.advance_to) + len(s.eliminations),
            bool(s.bo),
        )

    for s in stages:
        if s.name not in best:
            best[s.name] = s
            seen_order.append(s.name)
        elif _score(s) > _score(best[s.name]):
            best[s.name] = s
    stages = [best[n] for n in seen_order]

    # Cross-stage reference validation: every advance_to.target_stage should
    # match a parsed stage's name.
    by_name = {s.name for s in stages}
    for s in stages:
        for edge in s.advance_to:
            if edge.target_stage not in by_name:
                warnings.append(
                    f"stage {s.name!r} advances to unknown stage {edge.target_stage!r}"
                )

    return stages, warnings
