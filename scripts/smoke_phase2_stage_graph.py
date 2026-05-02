"""
Phase 2 smoke script — stage graph construction for the 5 live S-Tier events.

For each event, this script:
  1. Fetches the main Liquipedia page.
  2. Parses format prose -> StageDefs (Phase 1).
  3. Parses brackets -> Stages (existing parser).
  4. Collects per-stage sub-page rosters where Major-tier events expose them.
  5. Builds the stage graph and prints a per-stage breakdown.

Run:  python scripts/smoke_phase2_stage_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.liquipedia_loader import (
    _fetch_page, _bs4, _parse_format, _parse_per_stage_invites, _parse_seeded_teams,
)
from data_loaders.format_parser import parse_format_prose
from data_loaders.bracket_parser import parse_tournament_brackets
from data_loaders.stage_graph import build_stage_graph, collect_sub_page_rosters


import glob, json

S_TIER_SLUGS = [
    "BLAST/Rivals/2026/Spring",
    "PGL/2026/Astana",
    "Intel_Extreme_Masters/2026/Atlanta",
    "CS_Asia_Championships/2026",
    "Intel_Extreme_Masters/2026/Cologne",
]


def _load_seeded_teams_from_cache() -> dict[str, list[str]]:
    seeds: dict[str, list[str]] = {}
    for path in sorted(glob.glob(str(REPO_ROOT / "cache" / "upcoming*.json"))):
        try:
            for ev in json.load(open(path, encoding="utf-8")):
                if ev["slug"] in S_TIER_SLUGS:
                    seeds.setdefault(ev["slug"], ev.get("seeded_teams") or [])
        except Exception:
            continue
    return seeds


_SEEDS_BY_SLUG = _load_seeded_teams_from_cache()
EVENTS = [(slug, _SEEDS_BY_SLUG.get(slug, [])) for slug in S_TIER_SLUGS]


def _summarise_stage(stage) -> str:
    r1 = [m for m in stage.matches if m.round_idx == 1 and not m.is_bronze
          and m.sub in ("", "UB")]
    r1_pairs = ", ".join(f"({m.seed_a or 'TBD'} vs {m.seed_b or 'TBD'})" for m in r1)
    rows = [
        f"  Stage  : {stage.display_heading!r}  (id={stage.stage_id}, fmt={stage.format})",
        f"    def_name      : {stage.def_name!r}",
        f"    roster ({len(stage.roster):>2}) : {stage.roster}",
        f"    direct invitees: {stage.direct_invitees}",
        f"    R1 pairs ({len(r1):>2}) : {r1_pairs or '-'}",
    ]
    if stage.entrants:
        ent_lines = []
        for en in stage.entrants:
            ent_lines.append(
                f"      - {en.source}  count={en.count}"
                + (f"  upstream={en.upstream_stage}" if en.upstream_stage else "")
                + (f"  notes={en.notes!r}" if en.notes else "")
            )
        rows.append("    entrants:")
        rows.extend(ent_lines)
    if stage.advance_to:
        edge_lines = [
            f"      -> {e.target_stage}  count={e.count}  criterion={e.criterion}"
            + (f"  seeding={e.seeding}" if e.seeding else "")
            + (f"  notes={e.notes!r}" if e.notes else "")
            for e in stage.advance_to
        ]
        rows.append("    advance_to:")
        rows.extend(edge_lines)
    if stage.eliminations:
        rows.append(f"    eliminations: {stage.eliminations}")
    return "\n".join(rows)


def smoke_one(slug: str, seeded_teams: list[str]) -> None:
    print("=" * 88)
    print(f"EVENT: {slug}")
    print("=" * 88)

    html = _fetch_page(slug)
    if html is None:
        print("  [skip] failed to fetch main page")
        return
    soup = _bs4(html)

    prose = _parse_format(soup)
    stage_defs, prose_warnings = parse_format_prose(prose)

    print(f"  format prose : {len(prose)} chars")
    print(f"  StageDefs    : {len(stage_defs)}")
    for d in stage_defs:
        edges = ", ".join(f"{e.target_stage}({e.count})" for e in d.advance_to) or "-"
        print(f"    * {d.name}  fmt={d.format}  team_count={d.team_count}"
              f"  groups={d.n_groups}  advance={edges}")
        for w in d.warnings:
            print(f"        [!]  {w}")
    for w in prose_warnings:
        print(f"    [!] prose-warning: {w}")

    # Re-parse seeded teams from the live page so we get withdrawal-filtered list
    # (the upstream cache may pre-date the withdrawal note).
    live_seeded = _parse_seeded_teams(soup) or seeded_teams

    sub_rosters = collect_sub_page_rosters(slug, soup)
    invites = _parse_per_stage_invites(soup)

    # Phase 3 hard-cut: parse_tournament_brackets now orchestrates
    # build_stage_graph internally and applies per-stage roster seeds.
    parsed_stages = parse_tournament_brackets(
        soup, seeded_teams=live_seeded, slug=slug,
        format_prose=prose,
        direct_invitees_by_stage=invites,
        sub_page_rosters=sub_rosters,
    )
    print(f"  parsed Stages: {len(parsed_stages)}  (seeded teams: {len(live_seeded)})")

    if sub_rosters:
        print(f"  sub-page rosters discovered: {list(sub_rosters)}")
        for k, v in sub_rosters.items():
            print(f"    {k} ({len(v)}): {v}")

    if invites:
        print(f"  per-stage invites discovered: {list(invites)}")
        for k, v in invites.items():
            print(f"    {k} ({len(v)}): {v}")

    print()
    for stage in parsed_stages:
        print(_summarise_stage(stage))
        print()


def main() -> int:
    for slug, seeded in EVENTS:
        try:
            smoke_one(slug, seeded)
        except Exception as exc:
            print(f"  [error] {slug}: {exc!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
