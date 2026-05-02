# Bracket System — Functional Completion

> **⚠️ EXECUTION ORDER — READ FIRST**
>
> Before making *any* code changes from this plan, **invoke the `grill-me`
> skill** (`skills/grill-me`) on the current state of bracket support.
> Walk every architectural decision below — Liquipedia format-prose
> parsing strategy, cross-stage feeder representation, direct-invite
> handling, prize emission across stage chains — and ask the user to
> confirm the path before writing code.
>
> Reason: previous bracket slices (1–4) shipped per-format parsers
> without first nailing the *cross-stage* model, which is the central
> piece this plan addresses. Skipping the grill risks another round of
> visualised-but-not-functional brackets.

## Top-level checklist

- [x] grill-me session complete (8 questions locked — see §8.1)
- [x] taxonomy + general cross-stage model documented (§2)
- [x] format-prose contract specified (§3)
- [x] phase plan ordered with hard-cut at Phase 3 (§4, §8.3)
- [x] validation protocol agreed: user-driven per phase (§8.2)
- [x] **Phase 1** — Format-prose parser (§4)
- [x] **Phase 2** — Stage graph construction
- [x] **Phase 3** — Cross-stage R1 seed population *(hard-cut applied)*
- [x] **Phase 4** — Cross-stage advancement cascade + manual TBD dropdowns *(user-validated 2026-05-02 across PGL Astana / Atlanta / Cologne; all 3 scenarios green end-to-end)*
- [x] **Swiss Rework** — Buchholz pairer + VRS-snapshot seeding (§10) — user-confirmed working on Cologne S1
- [~] **Phase 5** — Per-format prize emission *(5.1–5.4 code-complete; awaiting user-validation in 5.6 and fixture tests in 5.5)*
- [ ] **Phase 6** — UI: cross-stage visibility + manual seed override
- [ ] **Phase 7** — Final validation pass across all 5 S-Tier events

---

## 0. Reframe — what "functional" means

The current bracket implementation **renders** every format (SE / SE+bronze /
DE / Swiss / RR Groups / GSL-lite) but doesn't make them **work** as
predictors. Specifically, the missing piece is **cross-stage qualification
flow**: how a team progresses from one stage to the next.

A bracket system is functional when **all four** of these hold:

1. **Every clickable slot resolves to two specific teams** before the user
   picks. Either (a) Liquipedia placed them, (b) cascade from upstream
   picks fills them, or (c) a manual seed-edit affordance lets the user
   place teams when neither (a) nor (b) is possible.
2. **Picks propagate across stages.** Picking S1 winners populates S2's
   entrant pool; picking S2 winners populates S3's; picking S3 winners
   populates Playoffs R1 seeds. No "TBD" left after the user has
   committed enough picks.
3. **Final placements emit the correct prize-label per Liquipedia's
   `prize_distribution`.** Group-stage exits get their group-stage
   prizes; Swiss exits get their record-bucket prizes; playoff finishers
   get their standard SE/DE placement prizes. Engine sees the right BO
   contributions across the entire bracket.
4. **Direct invites at intermediate stages are honoured.** A Major's
   Stage 2 has 16 teams: 8 from Stage 1 + 8 invited at Stage 2. The
   parser must know which 8 are the upstream advancers and which 8 are
   the new entrants.

If any of these fail, the bracket "works visually" but doesn't actually
predict standings.

---

## 1. Current state — what's broken

### What works (slices 1–4 + minor fixes)
- Per-format parsers parse DOM into `Stage` + `BracketMatch` objects.
- Single-stage events (just SE, just SE+bronze) work end-to-end.
- Within-stage feeder cascade works for SE+bronze (R1 → R2 → R3 → bronze).
- Within-stage Swiss cascade synthesises R2+ pairings via Buchholz buckets.
- DE within-stage feeders: UB tree + LB tree with cross-bracket loser feeds.

### What's broken
- **No cross-stage feeders.** `seeded_teams` (the full event roster from
  the discovery list) blindly backfills every stage's R1, regardless of
  whether that stage is downstream of group/Swiss stages.
  - IEM Atlanta SE Playoffs prefilled with `seeded_teams[0..5]` even
    though actual playoff entrants are top-2-per-group of the upstream
    DE groups.
  - PGL Astana SE Playoffs prefilled with `seeded_teams[0..7]` even
    though actual entrants are Swiss top-8.
  - IEM Cologne Stage 2/3 prefilled identically with `seeded_teams[0..15]`
    even though S2 has 8 from S1 + 8 new invites, S3 has 8 from S2 + 8
    new invites. Stage 1 also gets the same teams which is even more wrong.
- **No direct-invite awareness.** Cologne's Stage 2/3 inject 8 fresh teams
  per stage; we have no concept of "team entered at stage N".
- **No format-prose parsing.** Liquipedia's `<h3>Format</h3>` section
  contains the truth ("Top 8 proceed to Stage 2", "Bottom 8 eliminated",
  "Bo3 Elimination matches", "Grand Final Bo5") and we ignore it
  beyond rendering it as an info expander.
- **Prize emission only handles SE/SE+bronze.** Swiss/DE/Groups/GSL-lite
  emit match rows but no prize rows — group-eliminated teams who would
  have earned `9th-12th: $5K`-style Liquipedia prizes are uncounted.
- **Bracket signature drift causes silent state corruption.** When a
  parser version bump changes match IDs, prior picks reference dead
  IDs. The signature guard (architecture #12) shows a banner but
  doesn't migrate or invalidate selectively.

---

## 2. Tournament format taxonomy

To fix bracket flow we have to model what each format actually *does*.
Below is the canonical taxonomy across CS2 events; every upcoming
S-Tier event we've parsed maps to one of these patterns.

### 2.1 Single-stage formats

#### Single-Elimination (SE)
- N teams, log2(N) rounds. Final winner = 1st.
- **Optional bronze match**: extra slot in top round-center, fed by SF
  losers. Bronze winner = 3rd, loser = 4th.
- Placements: 1st, 2nd, 3rd-4th (or 3rd + 4th if bronze), 5th-8th, …
- **Bracket-reset** variant (rare in SE): not currently in scope.

#### Double-Elimination (DE)
- Upper Bracket (UB) tree + Lower Bracket (LB) tree.
- LB-R1 fed by UB-R1 losers; LB-R(k>1) inner = LB-R(k−1) winner, outer
  = UB-R(k) loser.
- **Grand Final** combines UB winner vs LB winner.
- **Bracket reset** (when LB winner wins GF1, GF2 plays for the title).
- 8-team **compact DE** (no GF — group format used by IEM Atlanta /
  CS Asia): UB-final winner = "1st in group", UB-final loser = "2nd
  in group", LB resolves 3rd–8th.

#### Swiss System
- N teams, fixed number of rounds (typically 5 for 16-team).
- Round 1: paired by initial seed (typically 1v9, 2v10, …).
- Round R+1: teams paired by current W-L record (Buchholz: top-half
  vs bottom-half within each record bucket; secondary tiebreak =
  opponents' record).
- **Bo3 distinction**: matches that *advance or eliminate* a team are
  Bo3 (e.g. when reaching 3 wins or 3 losses); other matches Bo1.
- Top X advance (typically half the field).
- Placements within Swiss: by record at exit.
  - 3-0: best record
  - 3-1: 2nd-tier
  - 3-2: 3rd-tier
  - 2-3: 4th-tier (eliminated)
  - 1-3: 5th-tier
  - 0-3: bottom

#### Round-Robin (RR)
- Every team plays every other team once (or twice for double RR).
- Standings by W-L; tiebreaker by head-to-head, then map difference.
- Top X advance.

#### GSL (4-team mini-DE)
- Opening Match × 2: 1v4, 2v3.
- Winners' Match: R1 winners → 1st-in-group advances.
- Elimination Match: R1 losers → loser eliminated.
- Decider: Winners' loser vs Elimination winner → 2nd-in-group.
- 5 matches total.

#### GSL-lite (4-team, 4 matches)
- Opening Match × 2 (same).
- Winners' Match: R1 winners (seeds the playoff bracket).
- Elimination Match: R1 losers (resolves 3rd vs 4th in group).
- **No decider.** Top 2 advance via R1 wins regardless of Winners'
  Match outcome.
- Used by BLAST Rivals.

### 2.2 Multi-stage chains — the general model

A tournament is **a directed graph of stages**. The graph is usually linear
(Stage 1 → Stage 2 → Playoffs) but can fan-in (multiple parallel groups →
one playoff), fan-out (one Swiss → two separate playoff brackets), and any
combination of the section-2.1 formats may appear at any node. The fix
must handle **arbitrary** combinations, not a hand-curated list of patterns.

#### 2.2.1 Stage entrant model

Every stage's R1 entrant pool is the union of one or more **entrant
sources**. A source is one of:

| Source type | Meaning | Example |
|---|---|---|
| `initial_roster` | Direct entry into the tournament's first stage. The teams listed in `seeded_teams` from event discovery. | Cologne S1 (16 RMR teams) |
| `advance_from(stage_X, count, criterion)` | Top-N teams from an upstream stage by some criterion (record, placement). | Cologne S2 takes "top 8 from S1 by record" |
| `direct_invite(stage_X, count)` | New teams entering directly at this stage, bypassing earlier stages. | Cologne S2 has 8 invitees joining S1's 8 advancers |
| `eliminate_from(stage_X, count, criterion)` | Bottom-N teams from upstream — used by "last chance" / repechage stages where eliminated teams get a second path. | Some Majors had a "Showmatch / consolation bracket" |
| `manual_seed` | User-placed via the manual seed override UX (Phase 6). Catch-all when prose parsing is incomplete or the user wants a counterfactual. | "What if FaZe were directly invited to Playoffs?" |

A stage's entrant pool = a list of `(source, count)` pairs that **must
sum to the stage's `team_count`**. Validation: if the sources don't add
up, the prose parsing or sub-page roster scraping is incomplete →
surface 🟠 banner with the gap.

#### 2.2.2 Stage advancement model

Every stage emits zero or more **advancement edges** to downstream stages,
plus an **elimination tail** for teams that exit at this stage:

| Edge type | Meaning |
|---|---|
| `advance_to(stage_Y, count, criterion, seeding?)` | Top-N (by record/placement) flow to a downstream stage; optional `seeding` overlay decides positional placement (1v8 / 2v7 / etc.) |
| `eliminate_at(this_stage, count, placement_label)` | Teams that exit here get a placement label (e.g. "9th-16th", "5th-8th-in-group") for prize lookup |

An edge's `criterion` can be:
- **By record** (Swiss): `top_by_wins(8)` — highest W-L count, tiebreak Buchholz.
- **By placement** (SE/DE): `placement(1)` = winner, `placement(3)` = bronze winner, `placement(3-4)` = SF losers.
- **By group rank** (RR/GSL): `group_rank(1)` = group winner, `group_rank(2)` = group runner-up, …
- **All** (rare — a stage that exists only to seed): every team advances.

Validation: sum of `advance_to.count` + `eliminate_at.count` per stage = `team_count`. Same surface-on-mismatch rule.

#### 2.2.3 Worked examples (showing the model is general)

The same dataclasses describe every event we've encountered.

**PGL Astana (Swiss → SE+bronze)**
```
Stage("Group Stage", format=Swiss, team_count=16,
  entrants=[initial_roster(16)],
  advance_to=[advance_to("Playoffs", count=8, criterion=top_by_wins, seeding=swiss_record_seed)],
  eliminate=[eliminate_at("9th-11th", 3), eliminate_at("12th-14th", 3), eliminate_at("15th-16th", 2)])

Stage("Playoffs", format=SE_with_bronze, team_count=8,
  entrants=[advance_from("Group Stage", 8)],
  advance_to=[],   # terminal
  eliminate=[eliminate_at("1st", 1), eliminate_at("2nd", 1), eliminate_at("3rd", 1),
             eliminate_at("4th", 1), eliminate_at("5th-8th", 4)])
```

**IEM Cologne Major (3× Swiss + direct invites → SE)**
```
Stage("Stage 1", format=Swiss, team_count=16,
  entrants=[initial_roster(16)],          # RMR low-tier
  advance_to=[advance_to("Stage 2", 8, top_by_wins)],
  eliminate=[eliminate_at("17th-19th", 3), eliminate_at("20th-22nd", 3), eliminate_at("23rd-24th", 2)])

Stage("Stage 2", format=Swiss, team_count=16,
  entrants=[advance_from("Stage 1", 8), direct_invite(8)],   # 8 mid-tier invitees
  advance_to=[advance_to("Stage 3", 8, top_by_wins)],
  eliminate=[eliminate_at("9th-11th", 3), eliminate_at("12th-14th", 3), eliminate_at("15th-16th", 2)])

Stage("Stage 3", format=Swiss, team_count=16,
  entrants=[advance_from("Stage 2", 8), direct_invite(8)],   # 8 top-tier invitees
  advance_to=[advance_to("Playoffs", 8, top_by_wins, seeding=swiss_record_seed)],
  eliminate=[eliminate_at("9th-11th", 3), eliminate_at("12th-14th", 3), eliminate_at("15th-16th", 2)])

Stage("Playoffs", format=SE, team_count=8,
  entrants=[advance_from("Stage 3", 8)],
  advance_to=[],
  eliminate=[eliminate_at("1st", 1), eliminate_at("2nd", 1), eliminate_at("3rd-4th", 2),
             eliminate_at("5th-8th", 4)])
```

**IEM Atlanta (2 parallel DE groups → SE+bronze) — fan-in**
```
Stage("Group A", format=DE_compact, team_count=8,
  entrants=[initial_roster(group="A", 8)],
  advance_to=[advance_to("Playoffs", 3, group_rank, seeding=group_a_seed)],   # 1st-A, 2nd-A, 3rd-A
  eliminate=[eliminate_at("4th-in-group", 1), eliminate_at("5th-6th-in-group", 2), eliminate_at("7th-8th-in-group", 2)])

Stage("Group B", format=DE_compact, team_count=8,
  entrants=[initial_roster(group="B", 8)],
  advance_to=[advance_to("Playoffs", 3, group_rank, seeding=group_b_seed)],
  eliminate=[eliminate_at("4th-in-group", 1), eliminate_at("5th-6th-in-group", 2), eliminate_at("7th-8th-in-group", 2)])

Stage("Playoffs", format=SE_with_bronze, team_count=6,
  entrants=[advance_from("Group A", 3), advance_from("Group B", 3)],   # fan-in
  advance_to=[],
  eliminate=[eliminate_at("1st", 1), eliminate_at("2nd", 1), eliminate_at("3rd", 1),
             eliminate_at("4th", 1), eliminate_at("5th-6th", 2)])
```

**BLAST Rivals (2× GSL-lite → 4-team SE) — fan-in, simpler**
```
Stage("Group A", format=GSL_lite, team_count=4,
  entrants=[initial_roster(group="A", 4)],
  advance_to=[advance_to("Playoffs", 2, group_rank)])

Stage("Group B", format=GSL_lite, team_count=4,
  entrants=[initial_roster(group="B", 4)],
  advance_to=[advance_to("Playoffs", 2, group_rank)])

Stage("Playoffs", format=SE, team_count=4,
  entrants=[advance_from("Group A", 2), advance_from("Group B", 2)])
```

#### 2.2.4 Edge cases the model must handle

These all fall out of the same dataclasses without special-casing:

- **Carryover** (same team in multiple stages): naturally expressed —
  if Team T advances from S1 to S2, it appears in both stages' rosters.
  The engine sees both stages' matches independently.
- **Last-chance / repechage** (eliminated team gets second path):
  modelled as `eliminate_from(stage_X, count, criterion)` source for
  the consolation stage. E.g. "S1 9th-11th get one more shot in
  Repechage Bracket".
- **Wildcards / open qualifiers feeding into closed**: the open qual
  is just another stage with `advance_to(closed_stage, count)`.
- **Asymmetric advancement** (1st-Group-A goes to UB R1, 2nd-Group-A
  goes to LB R1): the `seeding` overlay on `advance_to` handles this.
  The seeding function maps `(source_stage, source_rank)` → playoff
  bracket position.
- **Fan-out** (one stage feeds two downstream paths): a stage can have
  multiple `advance_to` edges. E.g. "Top 4 → SE Playoffs, 5th-8th →
  Consolation".
- **Bracket-mid invites** (a team enters at S2 having skipped S1): the
  `direct_invite` source on S2 captures this.
- **Last-stage Grand Final Bo5 vs other matches Bo3**: stored on
  `StageDef` for display; engine ignores Bo (works series-level).

#### 2.2.5 Implementation implications

- The `StageGraph` of section 3 is rewritten around `StageDef` with
  these source/edge lists.
- Phase 1's prose parser must extract source/edge structure, not just
  `advance_count`. Concretely: every "Top N proceed to X" → `advance_to`
  edge; every "Bottom M are eliminated" → `eliminate_at` row(s) with
  placement labels per Liquipedia's prize_distribution; absent source
  for stage-N means we infer `advance_from(stage_(N-1))` and need
  per-stage sub-page rosters to reconcile direct invites.
- Phase 3 (cross-stage R1 seed population) does graph traversal: walk
  every stage in topological order, fill its entrant pool from
  upstream advancers + direct invites + initial roster.
- Phase 4 cascade re-runs the traversal whenever any pick changes,
  re-computing top-N advancers per stage and propagating downstream.

---

## 3. The central missing piece — Format Prose Parser

Liquipedia exposes the truth in the `<h3>Format</h3>` section of every
tournament page (and per stage on Major sub-pages). We currently dump
the prose into a UI expander but don't extract structure from it.

Example (IEM Cologne, exact text from Liquipedia):
```
Stage 1: June 2nd - 5th, 2026
  16 Team Swiss System Format
  Elimination and Advancement matches are Bo3
  All other matches are Bo1
  Top 8 Teams proceed to Stage 2
  Bottom 8 Teams are eliminated
Stage 2: June 6th - 9th, 2026
  16 Team Swiss System Format
  …
  Top 8 Teams proceed to Stage 3
  Bottom 8 Teams are eliminated
Stage 3: June 11th - 15th, 2026
  16 Team Swiss System Format
  All matches are Bo3
  Top 8 Teams proceed to the Playoffs
  Bottom 8 Teams are eliminated
Playoffs: June 18th - 21st, 2026
  Single-Elimination bracket
  All matches (excl. Grand Final) are Bo3
  The Grand Final is Bo5
```

### What we need to extract per stage
| Field | Example | Used for |
|---|---|---|
| `name` | `"Stage 1"` / `"Playoffs"` | matching parsed `Stage.display_heading` |
| `team_count` | 16 | sanity-check against parsed match count |
| `format_keyword` | `"Swiss"` / `"Single-Elimination"` / `"GSL"` / `"Round-Robin"` / `"Double-Elimination"` | confirming `Stage.format` |
| `advance_count` | 8 (`"Top 8 proceed to …"`) | how many teams flow to next stage |
| `advance_to` | `"Stage 2"` / `"Playoffs"` | which stage receives them |
| `eliminate_count` | 8 (`"Bottom 8 eliminated"`) | placement labels for losers |
| `bo` | Bo3 / Bo1 / "Bo3 elim only" | for prize-pool deltas (engine ignores Bo) |
| `gf_bo` | Bo5 (Playoffs only) | display only |

### Parsing strategy options

| Option | Pros | Cons |
|---|---|---|
| **Regex** on prose | Deterministic, no extra deps | Brittle to phrasing variations |
| **Liquipedia template fetch** (mediawiki API) | Source of truth, structured | Extra HTTP, schema may vary per template version |
| **LLM call** with extraction schema | Robust to phrasing | External call cost, non-determinism |
| **Hand-coded heuristics + regex fallback** | Cheap, can be iteratively hardened | Coverage gaps until tested on many events |

The grill must decide. My recommendation in the grill: **regex with a
small heuristic library** — these prose blocks follow a near-rigid
template across Liquipedia CS2 pages. Start regex-only; flag any prose
that doesn't match cleanly and surface it for manual stage definition.

### Cross-stage feeder model

Once parsed, build a `StageGraph`:

```python
@dataclass
class StageDef:
    name: str
    team_count: int
    format: str               # "Swiss", "SE", "DE", "GSL", "GSL_lite", "RR"
    advance_count: int        # 0 for terminal stage (Grand Final)
    advance_to: str | None    # name of next stage; None for Playoffs

@dataclass
class StageGraph:
    stages: list[StageDef]
    # Per-stage entrant breakdown:
    #   Stage[0].entrants = direct_invites_at_stage_0  (= initial roster)
    #   Stage[i>0].entrants = (Stage[i-1].advance_count from prev) + direct_invites_at_stage_i
    direct_invites: dict[str, list[str]]   # stage_name → invitees entering at that stage
```

The graph drives:
1. **R1 seeding** for each stage: union of (upstream advancers' identities, picks-derived) + direct invites.
2. **Stage transitions**: when all of stage N's matches are picked, top N
   teams (by record / placement) populate stage N+1's entrant pool.
3. **Prize emission**: each stage exit = a placement label (`"9th-16th"`
   etc.), looked up in `prize_distribution`.

### Direct-invite source

Where do we get the per-stage direct invites?

- For Major sub-pages (`/Stage_2`, `/Stage_3`): each sub-page has its
  own teamcards/participants section listing the 16 teams of that stage.
  We can compute `direct_invites_2 = stage_2_teamcards − stage_1_advancers_from_main_seeded_teams`.
- For non-Major events with a single group stage: no direct invites at
  intermediate stages (Pattern A); skip.

This is doable per event but format-specific. Plan needs to detail per
upcoming event how to source the direct-invite list.

---

## 4. Implementation phases

### Phase 1 — Format prose parser (core)  `[x]`
- [x] New module `data_loaders/format_parser.py` with `StageDef`, `StageEntrant`, `StageEdge` dataclasses.
- [x] Regex template library matching Liquipedia's standard phrasing
      (Top-N proceed, Bottom-N eliminated, N Team SWiss/SE/DE/RR/GSL,
      Single/Double-Elimination bracket, "top N from each group",
      "Group stage winners/runners-up/3rd advance to X as Y Seeds",
      "Grand Final is Bo5") plus boundary chunker that handles both
      `·`-delimited and run-on prose layouts.
- [x] Public function `parse_format_prose(prose) -> (stages, warnings)`.
- [x] Returns warnings list so UI can surface 🟠 deviation banners (Q1).
- [x] Smoke output for all 5 live S-Tier events — clean, ready for Phase 2.
- [x] Removed 280-char truncation from `liquipedia_loader._parse_format`
      (full prose now reaches the parser); discovery cache version
      bumped to `v2` so prior cached entries auto-invalidate.
- [x] 26/26 regression tests pass.

### Phase 2 — Stage graph construction  `[ ]`
- [ ] Match parsed `Stage` objects (DOM walker) to `StageDef` entries by heading.
- [ ] For Major events, follow `/Stage_N` sub-pages; collect per-stage teamcards.
- [ ] Compute direct invites per stage = sub-page roster − upstream advancers.
- [ ] Persist on `Stage` as new fields `entrants` / `advance_to` / `eliminations` (per Q2).
- [ ] Smoke output: per-stage roster + direct-invite count for all 5 events; user sanity-checks.

### Phase 3 — Cross-stage R1 seed population  `[ ]` *(hard-cut: delete `seeded_teams` blanket backfill — Q4)*
- [ ] Delete the existing `_backfill_seeds` blanket logic entirely.
- [ ] Replace with stage-graph-aware resolver: first stage uses
      initial roster; mid-stages use upstream advancers + direct invites.
- [ ] TBD slots that lack a resolved entrant render as a manual-pick
      dropdown (Q5: per-slot selectbox with available teams + "Other…").
- [ ] User validation: each event renders correctly per checklist
      (some TBDs expected until Phase 4 cascade lands).

### Phase 4 — Cross-stage advancement cascade  `[ ]`
- [ ] When all of stage N's matches are picked, compute top
      `advance_count` teams by edge `criterion` → populate stage N+1's
      entrant pool automatically.
- [ ] Handle direct invites: stage N+1's roster = upstream advancers
      + invitees in their respective slots.
- [ ] Re-runs on every pick (cheap graph walk, no memoisation needed
      per Q7 — render-time resolution is microseconds).
- [ ] User validation: picks propagate cross-stage on all 5 events.

### Phase 5 — Per-format prize emission  `[ ]`
- [ ] SE/SE+bronze: already works (verify still works post-Phase 3).
- [ ] DE: GF winner = 1st, GF loser = 2nd, LB-final loser = 3rd, etc.
- [ ] Swiss: by W-L bucket → prize label. `3-0` = 1st-tier prize,
      `3-1`-tier = 2nd-tier, etc.
- [ ] RR Groups: by within-group rank → overall placement label.
- [ ] GSL/GSL-lite: 1st-in-group, 2nd-in-group, … → overall placement.
- [ ] Cross-stage: each team's prize comes from the stage at which they exited.
- [ ] Q6 invariant: prize row emitted only when the team's full upstream
      chain is picked; partial-fill = no prize for that team.
- [ ] User validation: engine deltas reflect real prize money on each
      eliminated team for all 5 events.

### Phase 6 — UI: cross-stage visibility + manual seed override  `[ ]`
- [ ] Each Stage card shows roster source caption: "16 from Stage 1" /
      "8 from Stage 1 + 8 direct invites" / "Initial roster".
- [ ] TBD slots that depend on upstream picks render an "(awaits
      Stage X results)" hint with a scroll-to-that-stage link.
- [ ] Per-Q5 dropdown lives on every TBD slot.
- [ ] User validation: clarity check + override flow exercised on a
      counterfactual scenario.

### Phase 7 — Validation against all 5 live S-Tier events  `[ ]`
End-to-end smoke test, user-driven:
- BLAST Rivals: 8 teams in 2 GSL-lite groups → 4 in SE. Top-2-per-group
  feeds SE R1 (1st-Group-A vs 2nd-Group-B, 1st-Group-B vs 2nd-Group-A
  per typical seeding).
- PGL Astana: 16 in Swiss → 8 in SE+bronze. Swiss top 8 by record
  (1-seed plays 8-seed in SE R1).
- IEM Atlanta: 16 in 2 DE groups → 6 in SE+bronze (3 from each group:
  UB winner = 1st-group, UB-final loser = 2nd-group, LB-final winner =
  3rd-group). Playoff seeds 1-2 (UB winners) get byes; play-in 3-6.
- CS Asia Champs: same as Atlanta.
- IEM Cologne Major:
  - S1: 16 teams. Top 8 → S2. Bottom 8 = 17th-32nd placements.
  - S2: 16 = 8 (from S1) + 8 (direct invites). Top 8 → S3. Bottom 8 = 9th-16th.
  - S3: 16 = 8 (from S2) + 8 (direct invites). Top 8 → Playoffs. Bottom 8 = 9th-16th (overlapping bucket — Liquipedia uses 9th-11th / 12th-14th sub-buckets per Swiss record).
  - Playoffs: 8 → SE w/ Grand Final Bo5. 1st, 2nd, 3rd-4th, 5th-8th placements.

For each event:
- Auto-fill all stages → engine emits `extra_matches` for every match
  + `extra_prizes` for every team's final placement.
- Compare predicted standings to baseline, assert non-zero deltas across
  every team that appears in any stage roster.
- No "TBD" left in UI after auto-fill.

---

## 5. Acceptance criteria

A bracket is "complete" iff:

- [ ] Every match across every stage shows two non-TBD teams after the
  user fills upstream stages (or auto-fills the entire chain).
- [ ] Picking a single match never leaves a downstream slot inconsistent
  (e.g. a team appears in two slots simultaneously, or vanishes between
  rounds).
- [ ] Each picked match emits one engine match row.
- [ ] Each team's exit placement maps to a Liquipedia `prize_distribution`
  label, emitting the corresponding prize row.
- [ ] All 5 live S-Tier events smoke-test cleanly.
- [ ] 26+ regression tests pass (no Valve baseline drift > existing
  budgets).

---

## 6. Out of scope (for this plan)

- Bracket-reset detection in DE GF.
- Liquipedia template/API integration (we keep HTML scraping).
- Per-pick "what-if multiple paths" branching scenarios (single-scenario
  per event).
- Map-score-level picks (engine works series-level).
- Mobile UI tweaks for cross-stage visibility.

---

## 7. References

- `data_loaders/bracket_parser.py` — current parser per format
- `app.py` — Tournament Predictor page block (~ line 1300+); session
  state model; engine emission helpers (`_emit_event_simulation_rows`,
  `_collect_predicted_simulation_rows`, `_compute_swiss_overrides`)
- `NEXT_STEPS.md` — prior Pillar 3 architecture decisions (16 locked
  decisions; this plan refines #9 (format scope) + introduces a new
  cross-stage qualification model not present in the original sheet)
- Liquipedia CS2 portal: `liquipedia.net/counterstrike/Portal:Tournaments`
- Format prose lives at: `liquipedia.net/counterstrike/<slug>` →
  `<h3 id="Format">` and on `/Stage_N` sub-pages.

---

## 8. EXECUTE THIS PLAN

The grill-me session is **complete**; decisions are locked below. Begin
Phase 1 directly, but re-invoke the grill-me skill on any new
architectural fork that surfaces during implementation.

After every Phase: run `pytest tests/` (26 regression tests must stay
green) **and** request user-driven validation per the protocol in §8.2.
Halt on any regression or failed validation step.

### 8.1 Locked grill decisions

| # | Question | Locked answer |
|---|---|---|
| Q1 | Format-prose parsing strategy | **Template-based with flag-on-deviation.** Build a small regex template library matching Liquipedia's standard format-prose phrasing. Prose that doesn't match emits a 🟠 stage banner so the user can manually mark direct invitees / advancement edges. |
| Q2 | Cross-stage feeder representation | **Separate `StageEntrant` / `StageEdge` dataclasses; `BracketMatch` stays clean.** Within-stage feeders (LB cross-bracket, SE+bronze loser feeds) and cross-stage flow are different concerns — mixing them in `BracketMatch` would force every match-level walker to filter cross-stage feeds and duplicates the source spec across the N entrants of a stage. New shape: `Stage.entrants: list[StageEntrant]` + `Stage.advance_to: list[StageEdge]` + `Stage.eliminations: list[(label, count)]`. |
| Q3 | Direct-invite source | **Parse per-stage sub-page teamcards as primary source.** Compare against upstream advance count; gap = direct invitees. UI manual-mark fallback when teamcards are absent or the gap doesn't reconcile. |
| Q4 | Transition strategy | **Hard-cut at start of Phase 1.** Delete the existing `seeded_teams` blanket backfill immediately. App will be partially broken for some events mid-development — that's the point. Forces real fixes, no masking, no obsolete code paths to maintain. |
| Q5 | Manual seed override UX | **Dropdown per TBD slot.** Each unresolved slot shows a small selectbox of "available teams" (= upstream advancers not yet placed + direct invitees + an "Other…" escape hatch). |
| Q6 | Prize emission for partial fills | **Only when unambiguous.** Emit a prize row for a team only when every stage on its path is fully picked. Partial-fill = no prize row for that team. |
| Q7 | Cache strategy | **Cache parser output (structure only); resolve cross-stage entrants at render time.** L2 cache key remains `(slug, seeded_teams_tuple, force_token, parser_version)` — does not include picks_hash. Render path: `parsed_stages → resolve_cross_stage(parsed_stages, picks) → realised Stage objects`. Resolution is cheap (graph walk over ≤5 stages × ≤32 teams). Caching post-cascade state would explode the cache key combinatorially across pick changes. |
| Q8 | Validation strategy | **User-driven manual validation per phase.** No fixture files. After each phase I report what changed and instruct the user step-by-step on which event(s) to open, what to click, and what the expected outcome is. User confirms (or rejects with details) before next phase starts. `pytest tests/` (existing 26-test suite) remains the automated regression tier. |

### 8.2 Validation protocol (per phase)

After each Phase lands:

1. **Automated regression**: run `pytest tests/`. Must show 26+ passed, no fails.
2. **User validation request**: I post a numbered checklist of the form:
   > **Phase N validation — please test:**
   > 1. Open `🏆 Tournament Predictor`, hit 🔄 Refresh, select **PGL Astana 2026**.
   > 2. Expected: Group Stage shows 16 teams in 8 R1 matches; Playoffs shows 8 TBD slots.
   > 3. Click the first 8 R1 matches in any order. Expected: Round 2 pairings populate with 1-0 winners paired by Buchholz buckets.
   > 4. Reply ✅ or describe what you saw instead.
3. **Halt on rejection**: if the user reports any deviation, fix that before moving on.
4. **Phase done** when user replies ✅.

Phases 1 and 2 (parser + graph construction, no UI) validate via small
Python smoke scripts I'll run myself and paste output for the user to
sanity-check. Phases 3+ require the user to drive the UI.

### 8.3 Phase ordering (re-affirmed)

1. **Phase 1** — Format-prose parser → `[StageDef, …]` per event.
   *(I run smoke scripts on all 5 events; user sanity-checks the output.)*
2. **Phase 2** — Stage graph construction with sub-page teamcard
   reconciliation. Direct invitees identified.
   *(I run smoke scripts; user sanity-checks per-stage rosters.)*
3. **Phase 3** — Cross-stage R1 seed population. **At this point
   `seeded_teams` blanket backfill is deleted (Q4 hard-cut).** Some
   events will show TBD where the cascade hasn't taken over yet.
   *(User opens app, validates each event per checklist.)*
4. **Phase 4** — Advancement cascade + manual TBD-slot dropdowns.
   *(User validates picks propagate cross-stage on all 5 events.)*
5. **Phase 5** — Per-format prize emission (Swiss/DE/Groups/GSL).
   *(User validates engine deltas reflect prize money for each
   eliminated team's correct placement.)*
6. **Phase 6** — UI: cross-stage visibility (entrant-source captions
   on each Stage card, "awaits Stage X" hints on TBD slots,
   click-to-edit dropdowns).
   *(User validates clarity + manual override flow.)*
7. **Phase 7** — Final validation pass across all 5 S-Tier events.
   No TBD left after auto-fill; engine deltas non-zero across every
   team that appears in any stage roster; placements map to
   prize_distribution labels correctly.

---

## 9. Status snapshot — RESUME HERE

> **Tomorrow's prompt:** *"continue the next steps brackets md file"* — read
> this section, run the failing test (none currently), and pick up at the
> next unchecked checkbox under **Phase 4 work plan**.

### 9.1 Where we landed today (sessions 1–3)

**Phase 1 — DONE.** `data_loaders/format_parser.py` parses Liquipedia
`<h3>Format</h3>` prose into `StageDef` / `StageEntrant` / `StageEdge`.

**Phase 2 — DONE.** `data_loaders/stage_graph.py` (new module) links DOM
stages to StageDefs by heading, populates `Stage.entrants` /
`advance_to` / `eliminations` / `roster` / `direct_invitees`.
Sub-page `/Stage_N` teamcard scrape via `collect_sub_page_rosters`.
Per-stage Major-event invitee partition via
`liquipedia_loader._parse_per_stage_invites`. Withdrawn teams filtered
via `_parse_withdrawn_teams` (parses `div.inotes-inner` notes).

**Phase 3 — DONE (hard-cut applied + UI validated by user).**
- `_backfill_seeds` renamed `_apply_roster_seeds`, signature now
  `(matches, roster)`, called per-stage from `parse_tournament_brackets`
  via the stage graph. R1 pair-ordering removed entirely (was
  misleading — Liquipedia DOM or manual dropdown only). Bye-slot fill
  retained for non-power-of-two SE+bronze.
- `_parse_swiss_stage` no longer pair-orders R1 (the "1v9 / 2v10 / …"
  fallback is gone).
- `_compute_swiss_overrides` skips synthesising R>1 pairings until at
  least one team has a non-zero W or L count (was alphabet-pairing
  every team from all-0-0 records, looking like the cascade had run).
- `_nearest_preceding_heading` upgraded to consider h4 sub-stage
  discriminators ("Group A" / "Pool 1") so two DE groups inside one
  h3 "Group Stage" get distinct labels in the UI. Generic h4s like
  "Detailed Results" or "Round 1" stay ignored.
- New session-state field `manual_seeds: dict[match_id, {a, b}]` on
  each scenario; `_resolve` overlays manual seeds on computed teams;
  TBD slots render a compact selectbox via `_render_manual_seed_box`.
- 🗑️ Clear picks now also clears `manual_seeds`.
- Cosmetic: country flag dropped from in-bracket button labels;
  compact CSS for selectbox + button + within-match column gap;
  `gap="large"` on round columns; `margin-bottom: 14px` between
  matches in a column.

`_PARSER_VERSION = "v9"` — bump it again if you change parser shape.

### 9.2 Phase 4 — code done, pending user validation

#### What's done

- **`compute_stage_advancers(stage, picks, count, criterion)`** in
  `data_loaders/stage_graph.py` — four criteria covered:
  - `"top_by_wins"` (Swiss): rank by W desc, L asc, alphabetical.
  - `"placement"` (SE / SE+bronze / DE main tree): final winner →
    final loser → SF losers → QF losers, deepest-round-first.
    Bronze splits 3rd/4th when picked.
  - `"group_rank"` (RR groups, 4-team round-robin): same code path
    as `top_by_wins` but counts `sub == "GR"` matches instead of
    `sub == "SW"`.
  - **`"gsl_rank"`** (GSL / GSL-lite mini-bracket): 1st = WM winner;
    2nd = WM loser (lite) / Decider winner (full); 3rd = EM winner
    (lite) / Decider loser (full); 4th = EM loser. Returns `[]` until
    Winners' Match decided.
- **`apply_cross_stage_cascade(stages, picks)`** in `stage_graph.py`:
  topological walk; stages with `advance_from` entrants pull advancers
  via the right criterion + concatenate `direct_invitees`. Multi-group
  fan-in handled by indexing all DOM stages sharing a `def_name` and
  splitting the entrant's aggregate count evenly across them.
- **App.py wiring** — after the cache hit + session-state init, the
  render loop deep-copies cached stages, overlays `manual_seeds` onto
  match seed slots, runs the cascade, writes resolved rosters back to
  each stage and re-runs `_apply_roster_seeds` for non-power-of-two
  SE+bronze bye slots. `_PARSER_VERSION` bumped to `v10` so existing
  L2-cached entries auto-invalidate.
- 53/53 regression tests pass (5 new: gsl_rank × 3, cascade × 3,
  -2 obsolete… actually +5 net). No skips, no xfails.

#### Known scope deferred to later phases

- **Engine emission** (`_emit_event_simulation_rows`) re-parses brackets
  independently and walks `m.seed_a/b` directly — manual seeds + cascade
  output don't flow into engine match rows yet. That's Phase 5's job
  ("per-format prize emission") since prize labels also depend on a
  unified seed-resolution path.
- **Auto-fill chaining** through multi-stage Major-style chains (Cologne
  S1→S2→S3): auto-fill walks each stage's matches and picks via
  `_resolve`. R1 slots in S2/S3 stay TBD until the user manually places
  advancers → autofill skips them. Cross-stage auto-placement (auto-fill
  also seats cascade advancers into downstream R1 in deterministic
  order) is a Phase 4.4 enhancement if the user reports pain.

#### Phase 4 work plan (status)

- [x] **4.1.a** — gsl_rank criterion with Winners' / Decider / Elimination
      cascade. 3 TDD cycles, all green.
- [x] **4.2** — `apply_cross_stage_cascade(stages, picks)` walker.
      Multi-group fan-in handled (shared def_name → fan over each DOM
      stage). 3 TDD cycles (linear chain, fan-in, direct-invite
      concatenation, empty-picks short-circuit).
- [x] **4.3** — Render-time cascade in `app.py`. Option (a) chosen:
      `copy.deepcopy(parsed_stages)` after cache hit, mutate roster +
      apply bye seeds. Cleaner than session-state overlays for the
      fields we mutate (just `roster` + a couple of bye seed slots).
- [ ] **4.4** — User-driven UI validation per §8.2 protocol:
      1. Open PGL Astana → pick all Swiss R1 winners → expect
         downstream Playoffs roster pool (visible in the manual-seed
         dropdown on the SE Playoffs R1 TBD slots) populates from the
         top-N by wins.
      2. Open IEM Atlanta → pick UB winners in Group A/B + LB to
         decide 3rd-place → expect Playoffs SF byes pre-fill with the
         top-2 group winners; manual dropdown for the play-in QFs sees
         all 6 group advancers.
      3. Open IEM Cologne → pick S1 R1-R5 winners → expect S2 manual
         dropdown shows S1 top-8 + the 8 "Stage 2 Invites".
      4. Auto-fill toolbar button: PGL Astana / Atlanta should
         propagate cleanly. Cologne expected to halt at S1 boundary
         (R1 of S2 stays TBD until user manually places). Report
         pain if this is too clunky.

### 9.3 Files touched (cumulative through Phase 4 code)

| File | Status | Notes |
|---|---|---|
| `data_loaders/format_parser.py` | new | Phase 1 |
| `data_loaders/stage_graph.py` | new | Phase 2 + 4.1 advancer logic + 4.2 cascade |
| `data_loaders/bracket_parser.py` | edited | `_apply_roster_seeds`; `_nearest_preceding_heading` upgrade; `parse_tournament_brackets` orchestrates graph build; per-format parsers no longer call backfill |
| `data_loaders/liquipedia_loader.py` | edited | `_parse_withdrawn_teams`, `_parse_per_stage_invites`, `_parse_seeded_teams` filters withdrawn |
| `app.py` | edited | `_cached_parsed_stages` wiring; `_collect_predicted_simulation_rows` wiring; `_PARSER_VERSION = "v10"`; manual-seed dropdown; `_resolve` overlay; 🗑️ Clear picks clears manual_seeds; compact-bracket CSS; `_compute_swiss_overrides` no-pick guard; **Phase 4 cascade applied at top of render** |
| `tests/test_stage_graph.py` | new | 24 tests covering Phase 2 + 4.1 + 4.2 |
| `tests/test_seeded_teams_filter.py` | new | 2 tests for withdrawal + Stage N Invites |
| `scripts/smoke_phase2_stage_graph.py` | new | Live-Liquipedia smoke for all 5 S-Tier events |
| `smoke_phase2.out`, `smoke_phase3.out`, `smoke_phase3_v2.out` | output artefacts | last-run smoke captures |

### 9.4 Test status

```
$ pytest tests/
tests/test_engine_regression.py        21 passed
tests/test_seeded_teams_filter.py       2 passed
tests/test_stage_graph.py              24 passed
tests/test_unfinished_event_backfill.py 5 passed
=========================== 53 passed ===========================
```

No skipped or xfailed tests. No untested regressions known.

### 9.5 Resume command

Next session:

> **continue the next steps brackets md file**

**Phase 4 ✅ done** (user-validated 2026-05-02 across PGL Astana /
Atlanta / Cologne). Resume at **Phase 5 — per-format prize emission**
(§4 / §11). Engine currently emits match rows + prize rows for SE /
SE+bronze only. Phase 5 unifies the seed-resolution path between UI
render and engine emission so picks-driven match rows + prize rows
reach the engine for all formats (Swiss / DE / RR Groups / GSL /
GSL-lite), and maps each team's exit placement to its
`prize_distribution` label.

> **Insertion 2026-04-28** — Phase 4.4 user validation surfaced two
> Swiss-specific bugs (Cologne S1 R2 wrong pairings, broken auto-fill
> chain). Root cause: the existing `_compute_swiss_overrides` pairer
> in app.py uses alphabetical sort + top-half-vs-bottom-half pairing,
> which violates the Swiss spec on every count. Phase 4.4 ✅ is on
> hold pending the **Swiss Rework** in §10 below. Once §10 lands,
> resume Phase 4.4 validation against the corrected pairer.

> **Insertion 2026-05-01** — Phase 4.4 re-validation surfaced three
> cross-stage bugs the user hit immediately:
>
> 1. **Dropdown showed every team in the tournament** instead of the
>    cascade-resolved per-stage pool. Fixed: `_manual_seed_pool(stage)`
>    now scopes to `stage.roster + stage.direct_invitees` first; falls
>    back to global only when stage-local pool is empty.
> 2. **DE_compact placement criterion returned UB-R1 losers as 3rd-Nth**
>    instead of walking the LB tree. Atlanta scenario broke because
>    Group Stage's `placement` advancers picked wrong teams for
>    Playoffs cascade. Fixed: `_placement_advancers` now walks LB
>    deepest-first when LB matches exist on the stage; UB earlier-round
>    losers are no longer treated as placement winners (they drop to
>    LB and resolve there).
> 3. **Cascade roster wrote to `Stage.roster` but R1 stayed TBD.**
>    Phase 3 Q5 had deliberately disabled auto-pair-ordering, but for
>    cross-stage cascade output the entrant pool IS deterministic.
>    Added `seat_cross_stage_r1(stage, roster)` with Option B seeding
>    map (1v8/4v5/2v7/3v6 for 8-team SE, 1v4/2v3 for 4-team, 16-team
>    map included). Wired into the render-time cascade block in app.py.
>    Fan-in shapes (Atlanta 6-team SE+bronze) are not in the seeding
>    map — they fall through to `_apply_roster_seeds` bye-fill +
>    manual-dropdown override (which is now correctly scoped per
>    Fix 1).
>
> Tests: 64/64 green (+2 new — `test_compute_stage_advancers_de_placement_walks_lb_for_third`,
> `test_seat_cross_stage_r1_*`). `_PARSER_VERSION` unchanged (cascade
> + seat run at render time on deepcopies — no DOM-shape change).
>
> User to re-walk §9.2 four scenarios. If Atlanta SF byes still seat
> wrong (top-of-roster instead of 1A/1B), add fan-in seeding map as
> Fix 4. If Cologne S2/S3 R1 stays TBD without 🌱 click, that's the
> documented Phase 4.4 R1-synthesis-on-snapshot-change behaviour —
> not a bug.

> **Insertion 2026-05-01 (continued)** — Phase 4.4 second-pass user
> validation surfaced three more bugs, all fixed. Tests now 67/67.
>
> 4. **Cascade fired prematurely after Swiss R1 alone** (PGL Astana,
>    Cologne). Root cause: `compute_stage_advancers` for `top_by_wins`
>    sorted by W desc and returned top-N regardless of whether teams
>    had actually advanced. After R1 every winner has 1W ≥ trivial
>    threshold. Fix: gate on a CS2 3-3 Swiss threshold of 3 wins for
>    16-team Swiss (`team_count_proxy >= 16`); smaller test Swiss
>    configurations infer threshold from `ceil(max_round/2)`. Cascade
>    now returns `[]` until ≥count teams have ≥3W. Also ensures
>    Playoffs only auto-seats after Swiss is fully decided.
> 5. **Atlanta cascade returned no advancers** (Group Stage → Playoffs
>    silent). Root cause: `format_parser` emits criterion=`group_rank`
>    for DE-formatted "groups", but `group_rank` walked sub="GR"
>    matches only — Atlanta's matches are sub="UB"/"LB". Fix:
>    `compute_stage_advancers` now dispatches on actual match shape —
>    `group_rank` + any UB/LB match → delegate to
>    `_placement_advancers`. RR matchlist-style groups still use the
>    sub="GR" wins/losses path.
> 6. **(Carried)** DE LB-final wiring: `_parse_de_bracket` rewires
>    LB-R3 feeder_b to UB-final loser unconditionally, even for
>    8-team DE_compact groups where the spec says UB-final loser
>    stays at 2nd (no drop). Behaviourally `_placement_advancers`
>    handles both paths (UB-final loser already in placed; the LB-R3
>    non-UB-loser is the 3rd-place winner) so cascade output is
>    correct either way. Parser-side cleanup deferred.
>
> Tests: 67/67 green (+3 new — `test_compute_stage_advancers_swiss_gate_*`,
> `test_compute_stage_advancers_group_rank_falls_through_to_placement_for_de`).

> **Insertion 2026-05-02** — Phase 4.4 third-pass (and final) user
> validation surfaced four more bugs + one performance gap. All fixed,
> all 3 scenarios green end-to-end. Tests now 68/68.
>
> 7. **Per-click 1-2s lag in As-published mode.** Root cause: every
>    pick triggered a full Streamlit script rerun — discovery filters,
>    metadata banner, format expander, prize table, seeded chips,
>    snapshot picker, *and* the bracket render all re-executed for one
>    pick. **Fix:** wrap the bracket render block (cascade + toolbar +
>    all stage rendering, ~600 lines) in a nested `@st.fragment def
>    _bracket_panel()`. Pick clicks now rerun ONLY the fragment, not
>    the surrounding page. `st.rerun()` calls inside replaced with
>    `st.rerun(scope="fragment")` (Streamlit 1.55+). Default-arg trick
>    `def _bracket_panel(parsed_stages=parsed_stages)` avoids Python
>    closure rebind on the in-fragment `parsed_stages = deepcopy(...)`.
> 8. **Astana Playoffs stayed TBD after fully-picked Swiss.** Root
>    cause: `_compute_swiss_overrides` synthesises R2-R5 pairings for
>    display only; `m.seed_a/b` stays `None` for unpublished rounds.
>    `compute_stage_advancers` reads `m.seed_a/b` to tally W-L → only
>    R1 wins counted → no team ever reaches 3W → cascade returns `[]`.
>    **Fix:** at fragment top (and between stages in autofill loop),
>    bake synth pairings into `m.seed_a/b` BEFORE running cascade.
>    Now top_by_wins sees all rounds.
> 9. **Auto-fill walked Playoffs before cascade re-fired.** Auto-fill
>    iterates stages sequentially; cascade only ran at fragment start
>    with empty picks → Playoffs R1 still TBD when autofill loop
>    reached it → autofill skipped. **Fix:** re-bake Swiss synth +
>    re-run cascade between stages in autofill loop, so each
>    downstream stage sees fresh upstream advancers before being
>    walked.
> 10. **Cologne R3 populated eagerly after R1 picks alone.** Root
>     cause: bake step (Fix 8) writes R2 seeds → next render's
>     `_compute_swiss_overrides` sees `any_seeded=True` for R2 and
>     loops to R3 without checking whether R2 has actual *picks* (the
>     break-on-no-picks only existed in the synth branch, not the
>     trusted-seed branch). **Fix:** mirror the no-picks break in the
>     trusted-seed branch too. R3 now waits for R2 picks.
> 11. **Atlanta Playoffs partially seated then stuck wrong.** Two
>     sub-bugs:
>     a. **No fan-in seeding map for 6-team SE+bronze.** Cascade
>        returned `[1A,2A,3A,1B,2B,3B]` but `seat_cross_stage_r1` only
>        had power-of-two maps. Fall-through to `_apply_roster_seeds`
>        bye-fill seated `[1A, 2A]` (top-of-roster) into SFs — wrong;
>        should be `[1A, 1B]` cross-half. **Fix:** added 6-team
>        fan-in branch — QF1 = 2A vs 3B, QF2 = 2B vs 3A, SF byes =
>        1A and 1B (split-bracket so co-group teams meet only at SF).
>     b. **Autofill seated partial roster which fan-in then couldn't
>        overwrite.** When Group A autofilled before Group B, cascade
>        returned 3-team partial roster; `_apply_roster_seeds` baked
>        `[V, PaiN]` into SF byes; Group B's autofill produced full
>        6-team roster but fan-in branch's "skip if seed already set"
>        guard left the wrong byes. **Fix:** partial-roster guard in
>        autofill loop — only seat downstream stages when cascade
>        roster has reached the expected entrant count.
> 12. **SF byes seated by cascade but rendered as TBD.** `_resolve`
>     for R2+ feeder-driven matches called `_team(feeder_id, kind)`
>     and returned `None` when feeder was absent — never consulting
>     `m.seed_a/b`. Fan-in seat wrote `seed_b="1A"` for SF byes; UI
>     showed TBD with manual-seed dropdown. **Fix:** `_team(...)
>     or m.seed_a/b` fallback at the resolver's final return —
>     seeded byes now render correctly.
>
> Tests: 68/68 green (+1 new —
> `test_seat_cross_stage_r1_6team_se_bronze_fan_in_atlanta_seeding`).
> No skips, no xfails. `_PARSER_VERSION` unchanged (all fixes are
> render-time + cascade-time, no DOM-shape change).
>
> User validation log:
> - **PGL Astana**: ✅ end-to-end. Snapshot pick → 🌱 R1 → ⚡ Auto-fill
>   fills Swiss + Playoffs (1v8/4v5/2v7/3v6) cleanly.
> - **IEM Cologne**: ✅. Snapshot pick → 🌱 S1 R1 → manual or autofill
>   propagates S1 → S2 → S3 → Playoffs. R3 stays TBD until R2 picked
>   (no eager-cascade). Single-stage transitions fully gated by 3W.
> - **IEM Atlanta**: ✅. Auto-fill chains both DE groups → Playoffs
>   with QF cross-pair (2A-3B, 2B-3A) + SF byes (1A, 1B) + Final +
>   Bronze.

---

## 10. Swiss Rework — Buchholz pairer with VRS-snapshot seeding

### 10.1 Why

Phase 3/4 quietly inherited a broken Swiss synthesiser from the original
slice 3 implementation:

- **Tiebreak**: alphabetical-by-team-name, called a "Buchholz proxy" in
  the docstring. Spec demands real Buchholz (sum of opponents' current
  wins) desc, then seed-asc.
- **Pairing scheme**: top-half-vs-bottom-half (`pairs.append((teams[i],
  teams[half + i]))`). Spec demands outer-pair (sorted A,B,C,D → A-D,
  B-C) within each W-L bucket.
- **Rematch handling**: none. Spec demands greedy single-swap.
- **Seed concept**: `seeded_teams` insertion order proxied as seed.
  Spec demands an authoritative seed — supplied by the user via the
  VRS-snapshot picker we're adding.
- **R1 synthesis**: deliberately disabled in Phase 3 (Q4 hard-cut)
  because we couldn't seed. Now opt-in via "🌱 auto-seed R1" button
  using snapshot rank for the top-half-vs-bottom-half split-bracket
  (#1v#9, #2v#10, …, #8v#16).

Bugs visible to the user (Cologne S1):
- R1 published correctly by Liquipedia, R2 alphabet-paired the 1-0
  bucket: Astralis-FlyQuest, Aurora-Gaimin Gladiators, BIG-MIBR,
  Falcons-PaiN Gaming. Should be outer-pair by Buchholz/seed.
- R2 renders 9 boxes instead of 8 (extra TBD slots) — separate parser
  glitch in `_parse_swiss_stage` matchlist iteration.
- Auto-fill toolbar can't propagate cross-stage in Cologne because
  S2/S3 R1 pairs are never synthesised.

### 10.2 Locked grill decisions (2026-04-28)

| # | Question | Locked answer |
|---|---|---|
| 10.Q1 | R1 synthesis policy | **Opt-in.** Per-Swiss-stage "🌱 auto-seed R1" button. Liquipedia DOM still wins when present. R1 stays TBD if button not clicked + Liqui silent. |
| 10.Q2 | R>1 synthesis trigger | **Liqui truth first; synth from records when Liqui silent.** Mid-flight events: trust whatever Liqui has published; only fill the holes. |
| 10.Q3 | Seed source | **VRS-snapshot rank.** Per-event header `selectbox`; no default. Snapshot's `rank` column → seed #1..#N. |
| 10.Q4 | Rematch handling | **Greedy single-swap.** Pair top-vs-bottom in sorted bucket; on rematch, swap j with j-1, j-2, …. Accept minimum rematches as fallback. |
| 10.Q5 | No-snapshot fallback | **Synth disabled.** Auto-seed-R1 button + R>1 cascade greyed; 🟠 banner "Pick a VRS snapshot to enable Swiss seeding". |
| 10.Q6 | Missing teams in snapshot | **Bottom-rank tail.** Append to seed list at snapshot_size+1, +2, … alphabetically. ℹ️ banner identifies them. |
| 10.Q7 | UI placement | **Per-event header.** One selectbox per tournament; persists in `slug_state["snapshot_date"]`. |
| 10.Q8 | Snapshot scope | **Swiss seed-tiebreak only.** Engine factors (BO/BC/ON/LAN/H2H) and auto-fill E(win) keep using the global mode. |

### 10.3 Spec corrections (user-supplied)

The original spec the user pasted was generic Swiss; CS2 reality
deviates in two places that we MUST encode:

- **R1 = top-half-vs-bottom-half split-bracket.** #1 vs #9, #2 vs #10, …, #8 vs #16. (Generic spec said #1 vs #16 outer-pair — wrong for CS2.)
- **R2+ = outer-pair within sub-bracket.** Sorted A,B,C,D → A-D, B-C. (Generic spec is right here.)

Buchholz convention: opponents' **current** wins, recalculated every
round. No opp-of-opp.

### 10.4 Implementation plan

- [x] **10.4.a** — `vrs_engine/swiss_pairer.py` (new module, pure
      functions). Public API: `seed_table`, `r1_split_bracket`,
      `compute_buchholz`, `pair_round`. 8 TDD cycles green.
- [x] **10.4.b** — Snapshot loader (`_snapshot_standings`) wired to
      `github_loader.load_valve_github_data` with pickle cache.
- [x] **10.4.c** — Per-event snapshot picker UI in `app.py`'s
      Tournament Predictor block. State persists in
      `slug_state["snapshot_date"]` + `slug_state["snapshot_year"]`.
      Date discovery falls back to scanning the local pickle cache
      when the GitHub API is rate-limited (60/hr unauthenticated)
      so users in offline / capped sessions can still pick from the
      21 cached snapshots.
- [x] **10.4.d** — "🌱 auto-seed R1" button per Swiss stage. Greyed
      when no snapshot picked OR when Liqui (or a prior auto-seed)
      already placed R1. Writes split-bracket pairings into
      `manual_seeds`.
- [x] **10.4.e** — `_compute_swiss_overrides` rewritten on top of
      `swiss_pairer.pair_round`. Snapshot-gated; "trust Liqui first;
      synth from records when silent" preserved.
- [x] **10.4.f** — Engine emit (`_emit_event_simulation_rows`) now
      overlays `manual_seeds` onto match seeds so auto-seed-R1 picks
      flow into engine match rows. Calls `_compute_swiss_overrides`
      with the per-event snapshot from `slug_state`.
- [ ] **10.4.g** — R2-shows-9-boxes parser bug (deferred). Re-test on
      live Cologne / PGL Astana DOM after Phase 5 work; address if
      still reproducible.
- [x] **10.4.h** — `_PARSER_VERSION` bumped to `v11`. No engine
      regression baseline drift (61/61 tests green).
- [x] **10.4.i** — User validation: Cologne Stage 1 confirmed
      working end-to-end (snapshot picker → auto-seed R1 → R1 picks →
      R2 cascade pairings render correctly via Buchholz pairer).
      1. Open Cologne. Pick a snapshot. Click 🌱 auto-seed R1 on
         Stage 1. Expect 8 R1 pairings rendered in 1v9, 2v10, … order.
      2. Pick R1 winners (any order). Expect R2 pairings:
         - 1-0 bucket sorted by snapshot rank asc → outer-pair (#1
           plays the 4th-best 1-0 team, #2 plays the 3rd, etc.).
         - 0-1 bucket same, separately.
      3. Continue picks through R5. Expect 8 advancers, 8 eliminated,
         no rematches across the run.
      4. Open PGL Astana. Same drill, single Swiss stage.
      5. Auto-fill toolbar: should now propagate Cologne S1→S2→S3
         when the user has clicked 🌱 auto-seed R1 on each Swiss stage
         beforehand.

### 10.5 Files in scope

| File | Action |
|---|---|
| `vrs_engine/swiss_pairer.py` | **new** — pure pairing module |
| `tests/test_swiss_pairer.py` | **new** — TDD tests, one per behaviour |
| `app.py` | edit — snapshot picker, auto-seed button, replace `_compute_swiss_overrides`, mirror in `_emit_event_simulation_rows` |
| `data_loaders/bracket_parser.py` | edit — fix `_parse_swiss_stage` slot-count bug |
| `CLAUDE.md` | edited — Swiss Tournament Architecture section landed in this session |
| `NEXT_STEPS.md` | edit — supersede #16 ("auto-fill default") with reference to §10 |
| `NEXT_STEPS_BRACKETS.md` | this section + status updates as work lands |

### 10.6 Resume command

> **continue the swiss rework**

Reads §10. Confirms tests still green, picks up at first unchecked
**10.4.x** item, resumes TDD cycle for `swiss_pairer.py`.

---

## 11. Phase 5 — Per-format prize emission

> **⚠️ EXECUTION ORDER — READ FIRST**
>
> Before making *any* code changes for Phase 5, **invoke the
> `grill-me` skill** on the open architectural questions in §11.2.
> Several decisions are not yet locked (engine seam, per-format
> placement → prize_distribution mapping, partial-fill emission rule
> from §8.1 Q6 vs cascade behaviour). Skipping the grill risks a
> partial Phase 5 that emits matches but misses prizes (or vice
> versa) for whole formats.

### 11.1 What's missing

Phase 4 cascade ends at *match cells*: every clickable slot resolves
to two specific teams once the user has picked enough upstream. But
the engine emission path (`_emit_event_simulation_rows`) still:

- **Re-parses brackets independently of the fragment-side cascade.**
  Cascade output (manual_seeds overlay + Swiss synth bake + roster
  population + R1 seat) doesn't flow into engine match rows. Picks
  for stages with cascade-resolved seeds are *invisible* to the
  Predicted-mode standings recompute.
- **Emits prize rows for SE / SE_with_bronze only.** Swiss, DE, RR
  Groups, GSL, GSL-lite all emit no prize rows — a team eliminated
  in Swiss with a `9th-11th: $5K` prize bucket gets no engine
  contribution from that placement. The picked match rows go
  through (so BO factor reflects them), but the BC factor that
  prizes drive does not.

### 11.2 Locked grill decisions (2026-05-02)

| # | Question | Locked answer |
|---|---|---|
| 11.Q1 | Engine seam (shared helper vs session-state pass) | **(a) Shared helper.** Hoist cascade + bake + seat into `stage_graph.resolve_for_render(parsed_stages, picks, manual_seeds, snapshot_standings) -> realised_stages`. Both fragment render AND `_emit_event_simulation_rows` call it. Pure function, testable, runs in any thread. The session-state alternative was rejected for breaking deterministic engine emission when user lands on a non-Predictor page first. |
| 11.Q2 | Prize label resolution order | **Already coded.** `_lookup_prize_for_place` (app.py:317-341): exact label match first, fall back to smallest containing range. Keep as-is. |
| 11.Q3 | Placement label mapping (all formats) | **Stage-graph derived absolute place.** Compute `place_offset(stage)` via topo walk of stage graph (linear chains: cumulative team_count from terminal backward; fan-in: aggregate sibling stages). Per-format bucketer emits within-stage rank; bucket × offset → absolute place range like `"25th-27th"`. Verified against Liquipedia data: Cologne publishes 9th-11th/12th-14th/15th-16th/17th-19th/.../31st-32nd matching this exactly; Atlanta publishes 7th-8th/9th-12th/13th-16th matching the fan-in aggregation. Tied bucket = single label, shared prize, no tiebreak. |
| 11.Q4 | DE 8-team compact labels | **Walker rule:** Advanced (3 per group) silent until Playoffs. Dropouts: 4th = LB-final loser, 5th-6th = LB-SF losers, 7th-8th = LB-R1 losers. Fan-in aggregation across parallel groups → tied bucket. |
| 11.Q5 | GSL / GSL-lite labels | **Same algorithm.** Top 2 advance to Playoffs (silent until exit). Dropouts: 3rd = EM winner, 4th = EM loser. Fan-in aggregates per-rank. |
| 11.Q6 | Partial-fill rule | **Strict.** Emit prize row only when every stage on team's path is fully picked. Phase 4's cascade gates (Swiss 3W threshold, placement walker requiring final picked) already enforce this naturally. No ergonomic relaxation. |
| 11.Q7 | `_PARSER_VERSION` bump | **Yes — v11 → v12.** Cheap insurance, invalidates any stale L2 cache from prior emission paths. |
| 11.Q8 | Test strategy | **Fixture-based.** Capture one HTML snapshot per format (PGL Astana for Swiss→SE+bronze; Atlanta for DE-fan-in→SE+bronze; Cologne for chained-Swiss→SE) under `tests/fixtures/`. Unit tests for placement walkers + offset computation. Integration test walks fixture event end-to-end, asserts engine emits expected `(extra_matches, extra_prizes)` rows. Live-Liquipedia smoke kept separate (`scripts/smoke_phase5.py`) for sanity but never blocks CI. |

### 11.3 Phase 5 work plan (skeleton, refine after grill)

- [x] **5.1** — Hoist cascade + bake + seat into a shared helper.
      `stage_graph.resolve_for_render(parsed_stages, picks,
      manual_seeds, snapshot_standings)` returns realised stages.
      Fragment render already wired in §10. **2026-05-02 close-out:**
      `vrs_engine/event_simulation.emit_simulation_rows` extracted as
      pure post-realised body; `_emit_event_simulation_rows` in app.py
      now: fetch → parse → `resolve_for_render` → `emit_simulation_rows`.
      Both call sites share identical seed-resolution path. New test
      `tests/test_event_simulation.py` locks downstream-R1-after-cascade
      behaviour (74/74 green).
- [x] **5.2** — Per-format placement labellers in
      `vrs_engine/placement_labels.py`. Pure functions emitting
      `list[tuple[bucket_label, list[teams]]]` best-first:
      `compute_swiss_exit_buckets`, `compute_gsl_exit_buckets`
      (covers GSL-lite + full GSL via Decider detection),
      `compute_de_compact_exit_buckets`, `compute_rr_exit_buckets`,
      `compute_se_exit_buckets`. Walkers emit dropouts only for
      mid-chain stages (Swiss/DE/GSL/RR); SE walker emits all teams
      (terminal-stage convention). 6 unit tests.
- [x] **5.3** — Engine emit rewrite. New
      `compute_place_offsets(stages)` (linear chain + fan-in siblings)
      and `compute_absolute_placements(stages, picks)` (per-format
      walker dispatch + cross-sibling aggregation by bucket label +
      offset composition + exit-stage filter so advancers take their
      downstream prize, not their upstream one). Wired into
      `vrs_engine/event_simulation.emit_simulation_rows` —
      replaced the old SE-only inline placement loop. **3 new tests**
      (`tests/test_place_offsets.py`) plus **1 integration test**
      (`tests/test_event_simulation.py::…upstream_group_dropouts`)
      locking RR-group dropouts emit prizes through the new path.
      83/83 green. `_PARSER_VERSION` bumped to `v12`.
- [x] **5.4** — Cross-stage exit tracking — landed inside
      `compute_absolute_placements`: ``team_exit_def`` map captures the
      deepest stage whose ``roster`` contains each team (last-write-wins
      across parsed-stage order which is upstream→downstream); buckets
      drop teams whose exit-stage isn't this stage's def_name, so
      advancers' prizes come from downstream only. Q6 strict
      partial-fill enforced naturally — RR walker returns ``[]`` when
      any group match is unpicked; Swiss/DE/GSL walkers gate on the
      decisive match (LB final / WM / loss-threshold).
- [ ] **5.5** — Smoke + integration tests (fixture-based per Q8).
- [ ] **5.6** — User validation across all 5 S-Tier events: predicted
      standings must show non-zero deltas for every team appearing in
      any stage roster after auto-fill.

### 11.4 Resume command

> **continue the next steps brackets md file**

Reads §11. Runs `pytest tests/` (must show **83/83** green from
end-of-5.4). Picks up at the first unchecked **5.x** item — currently
**5.5** fixture-based integration tests per Q8 (one HTML snapshot per
format under `tests/fixtures/` exercising end-to-end engine emission)
followed by **5.6** user validation across all 5 S-Tier events.
