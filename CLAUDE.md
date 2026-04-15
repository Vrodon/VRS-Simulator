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
