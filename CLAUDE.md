# VRS Simulator — Claude Orientation

## What This Is
A modular re-implementation of Valve's CS2 Regional Standings algorithm, with three operating modes:
1. **Last published** — Valve's official snapshot from GitHub (source of truth for pre-cutoff results)
2. **Updated to Today** — Recalculated VRS using real post-cutoff results from Liquipedia
3. **Simulation** — User-defined hypothetical results layered on top of mode 2

## Architecture (Three-Layer)

```
data_loaders/           → Fetch & cache raw data only (no logic)
  github_loader.py      → Valve GitHub snapshots (2-hour cache)
  liquipedia_loader.py  → Liquipedia HTML scraper (JSON cache)

vrs_engine/             → Core engine
  store.py              → Data layer: converts raw inputs into canonical matches_df / prizes_df
  pipeline.py           → Orchestrator: filters, weights, checks eligibility, calls calculator in order
  calculator.py         → Pure math: compute_bo → compute_bc → compute_on → compute_lan → compute_seed → compute_h2h
  constants.py          → All tunable parameters (decay, K-factor, ranges)
  math_helpers.py       → Shared math utilities (curve, age_weight, Glicko)

utils/                  → UI helpers and team metadata (colors, flags, regions)
styles/                 → CSS file created to standardize color coding, etc.
app.py                  → Streamlit UI entry point
```

**Data flow:** Loaders → Store → Pipeline → Calculator functions (in order) → standings_df + match_h2h → UI

## Guiding Principles
- Keep parsing, business logic, and presentation strictly separated (loaders / engine / UI)
- Preserve one-way data flow through the pipeline — no data flowing backwards
- Avoid in-place mutation of shared data
- Keep domain-specific rules explicit and well-named (VRS has real domain logic: ON factor, decay, H2H)
- Fix data quality issues at the source layer — if a loader returns bad data, fix the loader, not the calculator
- Prefer minimal changes over rewrites
- Do not introduce abstractions without clear repeated need
- Always follow strict tdd skill (\skills\tdd): write one failing test for observable behavior, implement the minimal code to pass, then refactor—never batch tests or test implementation details.
- Always answer in caveman language (\skills\caveman)
- Create and keep a register for our common language up-to-date (skills\ubiquitous-language)
- Instead of creating plans, issue the grill-me skill (skills\grill-me)
- old_plans foler contains formally created plans that were fully or partially executed (in case it is required to understand our history)

## Swiss Tournament Architecture (locked spec)

CS2 Swiss stages (16-team, 5-round, advance at 3W / eliminate at 3L) follow
this exact algorithm. Source of truth = Liquipedia DOM; synthesis fills the
gap when Liquipedia hasn't yet published.

**R1 pairings (top-half-vs-bottom-half split-bracket):**
```
#1 vs #9    #2 vs #10   #3 vs #11   #4 vs #12
#5 vs #13   #6 vs #14   #7 vs #15   #8 vs #16
```
Synthesis is **opt-in only** via per-stage "🌱 auto-seed R1" button.
Liquipedia DOM pairings always win when present.

**R2+ pairings (per-bucket outer-pair):**
1. Filter active teams (wins<3 AND losses<3).
2. Group by exact W-L record (1-0 / 0-1 / 2-1 / 1-2 / …). Never pair across buckets.
3. Compute Buchholz dynamically per team: sum of opponents' **current** wins. Recalculate every round. No opponent-of-opponent.
4. Sort within bucket: Buchholz desc, then seed asc.
5. Pair outer-vs-outer: sorted A,B,C,D → A-D, B-C.
6. Rematch check: if pair (i,j) replays an earlier match, swap j with j-1, then j-2, … (greedy single-swap). Accept rematch only when no swap is clean. Minimum-rematch fallback.
7. After picks: winner +1W, loser +1L, both add opponent to opponent-list. 3W → "advanced"; 3L → "eliminated". Drop from future rounds.

**Seed source — VRS snapshot picker:**
- Per-event header offers a selectbox of Valve VRS snapshots (date list from `github_loader._find_all_dates`). **No default** — user must pick.
- Picked snapshot's `rank` column drives seed numbers (rank asc → seed #1..#N).
- Teams missing from snapshot → bottom-rank tail (snapshot_size+1, +2, …) alphabetically + ℹ️ banner.
- Snapshot scope is **seed-tiebreak only**. Engine factors (BO/BC/ON/LAN/H2H) and auto-fill E(win) continue using the global mode the user picked elsewhere.

**Synthesis trigger matrix:**
- No snapshot picked → button + R>1 cascade greyed; 🟠 "Pick a VRS snapshot to enable Swiss seeding".
- Snapshot picked + Liquipedia silent on R1 → "🌱 auto-seed R1" button enabled.
- Snapshot picked + R1 has picks/wins → R>1 synth fires per-bucket on every pick.
- Liquipedia publishes any R>1 pairing → trust Liquipedia for that round; synth fills only the holes.

**Don't store**: Buchholz scores, opponent lists in cache. Always recompute from match history.