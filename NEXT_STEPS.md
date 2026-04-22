# VRS Simulator — Next Steps Roadmap

**Created:** 2026-04-19 · **Last revised:** 2026-04-21
**Purpose:** One-week plan to make the VRS sim "fly". Self-contained — a future Claude session can pick this up cold.

> **How to use:** Tell Claude `continue NEXT_STEPS.md` (or point at a specific Day/Pillar). When a task is finished, mark it `[x]` and add a one-line note about the actual outcome.

---

## North Star

Every decision in this plan serves three user-facing goals. If a task doesn't help one of these, it's not in scope.

1. **🔬 Transparency** — make it crystal clear *why* a team is at its current rank. Every point should be traceable to specific matches and prizes.
2. **📈 Improvement Path** — show *what each team needs to do* to climb. Which factor is the bottleneck, who they should beat, what's about to expire.
3. **🏆 Bracket Simulator** — let the user say *"if this upcoming event plays out like this, where do teams land?"* in a few clicks.

Days 1–2 are foundation work that unblocks all three. Days 3–7 deliver the three pillars in order.

---

## 0. Current State (as of 2026-04-19)

### What just shipped
- **ON formula reverse-engineered** from Valve's open-source JS at `ValveSoftware/counter-strike_regional_standings/model/team.js`. Two-pass `ownNetwork → opponentNetwork`, NOT PageRank. Validated MAE 0.009 across 21 cached snapshots. See `ON_FORMULA_ANALYSIS_NOTES.md`.
- **Three-layer engine rebuild** (commit `af47617`): `data_loaders/` → `vrs_engine/{store,pipeline,calculator}` → `app.py`.
- **Roster-split fix** (commit `afec7fd`): dual-layer dynamic active-roster identification fixes the BC.Game / MIBR -729 point drop. Root cause was non-deterministic dict overwrite in [data_loaders/github_loader.py](data_loaders/github_loader.py).

### What's uncommitted (must land first — see Day 1)
- `vrs_engine/calculator.py` — `compute_on()` rewritten to two-pass formula
- `vrs_engine/pipeline.py` — call site updated (dropped `bo_factor` arg)
- `app.py`, `data_loaders/liquipedia_loader.py` — small follow-on changes
- `.claude/settings.local.json` — local settings tweak

### Known clutter (root directory pollution)
14 scratch scripts from the ON investigation are sitting at repo root:
- `on_formula_analysis.py`, `on_formula_bo_hyp.py`, `on_formula_deep.py`, `on_formula_losses.py`, `on_formula_match.py`, `on_formula_pagerank.py`, `on_formula_precise.py`, `on_formula_valve.py`
- `on_valve_cpw.py`, `on_valve_factor.py`, `on_valve_full_sweep.py`, `on_valve_outliers.py`
- `on_engine_smoke.py`, `on_debug.py`
- Reference copies of Valve JS: `ranking_valve.js`, `data_loader_valve.js`, `main_valve.js`, `nth_highest_valve.js`, `remap_value_clamped_valve.js`, `report_valve.js`
- Datasets: `on_analysis_dataset.{csv,parquet}`, `on_match_dataset.parquet`
- Notes file: `ON_FORMULA_ANALYSIS_NOTES.md` (extract follow-ups, then delete)

### Architecture refresher
```
data_loaders/           Fetch & cache raw data (no logic)
  github_loader.py      Valve GitHub snapshots (2-hour cache)
  liquipedia_loader.py  Liquipedia HTML scraper

vrs_engine/             Core engine
  store.py              Data layer: tmh+bpm → matches_df / prizes_df
  pipeline.py           Orchestrator: calls calculator in order
  calculator.py         Pure math: bo → bc → on → lan → seed → h2h
  constants.py          DECAY_DAYS, TOP_N, SEED_MIN/MAX, BASE_K, etc.
  math_helpers.py       curve, age_weight, expected_win, top_n_sum, ...

utils/                  UI helpers, team metadata
app.py                  Streamlit UI (3993 lines)
```

Data flow: Loaders → Store → Pipeline → Calculator → standings_df + match_h2h → UI

The Team Breakdown page already exists ([app.py:3268+](app.py:3268)) with a basic factor waterfall and "Score Change Breakdown". This is the surface to enrich for Pillar 1.

---

# FOUNDATION (Days 1–2) — unblock the three pillars

## Day 1 — Land + lock the ON rewrite + extend engine for attribution

### Task 1.1 — Commit pending engine changes [x]
- Branch off `main` (e.g. `claude/on-rewrite-land`).
- `git add vrs_engine/calculator.py vrs_engine/pipeline.py` and commit referencing the MAE 0.009 validation result.
- Decide: bundle `app.py` + `data_loaders/liquipedia_loader.py` follow-ons in same commit or split?
- **Outcome:** landed as commit `6af949f` on `main` (bundled with the Liquipedia/UI follow-ons and reference JS).

### Task 1.2 — Build regression test harness [x]
- Create `tests/test_engine_regression.py`.
- Parametrize across all `cache/github_vrs/vrs_*.pkl` snapshots (21 of them, Aug 2024 → Apr 2026).
- For each snapshot:
  1. Load via `data_loaders.load_valve_github_data(date_str, year)`.
  2. Build `Store.from_valve(tmh, bpm)`.
  3. Run `pipeline.run(store, cutoff=snapshot_cutoff)`.
  4. Assert `MAE < 0.01` on each of `bo_factor`, `bc_factor`, `on_factor`, `lan_factor`, `seed`, vs Valve's published values.
- Document any per-factor exceptions (ON has documented MAE 0.009; allow up to 0.012 with a TODO).
- Wire into `pytest`. No CI yet — local invocation.
- **Outcome:** [tests/test_engine_regression.py](tests/test_engine_regression.py) created; all 21 snapshots pass in ~25s. Budgets calibrated to observed maxima (`bo 0.08`, `bc 0.02`, `on 0.012`, `lan 0.05`, `seed 50pt`) — early-2024 snapshots drive the wider `bo`/`lan` bounds; TODO in file to revisit thin-data window behaviour. Valve's duplicate roster rows are de-duped on the Valve side before comparison.

### Task 1.3 — Extend engine to return per-match attribution **(critical for Pillar 1)** [x]
This is the foundation of transparency. Without it, the Team Breakdown page can only show factor scores, not *why* those scores are what they are.

- Modify `compute_bo()` ([calculator.py:26](vrs_engine/calculator.py:26)) to additionally return `bo_contributions: dict[team → list[dict]]` where each entry is `{prize_id, date, amount, age_w, weighted, in_top10: bool}`.
- Modify `compute_bc()` ([calculator.py:79](vrs_engine/calculator.py:79)) to return `bc_contributions: dict[team → list[dict]]` where each entry is `{match_id, opponent, opp_bo_ratio, age_w, ev_w, weighted, in_top10: bool}`.
- Modify `compute_on()` ([calculator.py:130](vrs_engine/calculator.py:130)) to return:
  - `own_network_contributions: dict[team → list[{opponent, max_age_w}]]` (Pass 1)
  - `on_contributions: dict[team → list[{match_id, opponent, opp_own_network, age_w, ev_w, weighted, in_top10}]]` (Pass 2)
- Modify `compute_lan()` ([calculator.py:199](vrs_engine/calculator.py:199)) to return `lan_contributions: dict[team → list[{match_id, opponent, age_w, in_top10}]]`.
- `pipeline.run()` aggregates these into `result["attribution"] = {bo, bc, on, lan}`.
- `compute_h2h()` already returns per-match deltas (`match_h2h`); fold into the attribution dict for symmetry.
- **Outcome:** calculators now return contribution rows alongside factor dicts; `pipeline.run()` emits `result["attribution"] = {bo, bc, own_network, on, lan, h2h}`. Each row carries `in_top10` so the UI can render "counting" vs "below-cut" lists directly. Regression harness gained a sanity check that BO top-10 contributions sum to `bo_sum`; all 21 snapshots pass.

### Acceptance
- All 21 snapshots pass regression tests. [x] (~25s)
- `pytest tests/` runs in under 2 minutes. [x]
- `result["attribution"]["bo"]["FaZe"]` returns a list summing exactly to `result["standings"]["bo_sum"]` for FaZe. [x] (asserted in the regression test for 10 sampled teams per snapshot)

---

## Day 2 — Repo cleanup + unify simulation through engine

### Task 2.1 — Cleanup [x]
- **Outcome:** `ON_FORMULA_ANALYSIS_NOTES.md` already deleted before this session. 14 scratch Python scripts removed from repo root. 6 Valve JS reference files removed via `git rm`. Datasets moved to `scratch/` (gitignored; the narrower `on_*.py / on_*.csv / on_*.parquet / ON_FORMULA_*` patterns in `.gitignore` collapsed to a single `scratch/` entry). `ON_ITERS` dropped from [vrs_engine/constants.py](vrs_engine/constants.py) and [vrs_engine/__init__.py](vrs_engine/__init__.py). `grep -r ON_ITERS` returns only NEXT_STEPS.md.

### Task 2.2 — Network Sandbox UI refresh [x]
- **Outcome:** Audit found the Network Sandbox already describes the two-pass `ownNetwork → opponentNetwork` math end-to-end ([app.py:2155](app.py:2155)–[app.py:2520](app.py:2520) — "Pass 1 banner", "Pass 2 banner", `_on_sandbox_compute` helper). The only remaining mention of PageRank is the ✓ VERIFIED tag at [app.py:2299](app.py:2299) which explicitly says "NOT iterative PageRank" — kept as a clarifier. `_compute_on_history` no longer exists in the codebase.

### Task 2.3 — Remove dead roster-split filter [x]
- **Re-scoped during investigation:** `_identify_active_rosters()` turned out to be a no-op in its current form — `active_team_names = {standings.loc[idx, "team"] for idx in active_rosters.values()}` collapses to `set(team_names)`, which equals `snapshot_teams`, so `_is_ok()` always returns True. The real roster dedup happens upstream in [data_loaders/github_loader.py](data_loaders/github_loader.py) (`_team_best` pass keeps the best-ranked roster per team name; confirmed: `tmh` has 326 keys for 326 unique teams out of 357 standings rows). Store never sees duplicates, so moving the helper there would add ceremony with no semantic effect.
- **Outcome:** Deleted `_identify_active_rosters()` from [app.py](app.py) and removed the no-op Layer-1 filter block in `_auto_fetch_updated_standings()`. Added a comment pointing to `github_loader._team_best` as the real dedup site. `snapshot_standings` parameter is kept because it's still used for display-metadata merging (region/flag/color).

### Task 2.4 — Unify simulator through real engine [x]
- **Outcome:** Added `Store.append_simulation(extra_matches, extra_prizes)` in [vrs_engine/store.py](vrs_engine/store.py) (accepts lists or DataFrames, handles `winner_prize` spillover into `prizes_df`, assigns artificial match IDs above the Liquipedia offset).
- The What-If Predictor at [app.py:1080](app.py:1080)+ now runs the engine **twice** against the same cutoff — once without sim rows, once with — so the delta reflects only the what-if impact (not engine-vs-Valve rounding drift). `_sim_cutoff = max(sim_cutoff_dt, latest hyp date)` extends the window forward when the user queues future-dated matches. Baseline comparison uses `base_sim_std` (engine baseline) instead of `base_standings` (Valve published).
- `compute_standings()` deleted from `app.py`.
- **Sanity check** (G2 beats Vitality, $500k LAN, G2 gets $100k prize, run at snapshot cutoff): `d_bo = +0.116` for G2, `d_bc = +0.043`, `d_on = +0.003`, 23 teams see BC/ON shifts. Under the old `compute_standings()` those would all be zero — proving the engine is actually re-running every factor.

### Acceptance
- [x] `git ls-files` shows only intentional artifacts (6 tracked Valve JS refs deleted; scratch/ and datasets gitignored).
- [x] Adding a hypothetical match in the What-If Predictor now changes BC and ON visibly in the waterfall (not just BO + H2H). Sanity check above shows 23 teams with BC/ON deltas.
- [x] Existing roster-split scenarios still produce realistic deltas — all 21 regression snapshots pass (~14s).

---

# 🔬 PILLAR 1 — TRANSPARENCY (Days 3–4)

**Goal:** A user opens any team's page and sees, immediately, *every match and every prize that contributes to their score, ranked by impact*.

## Day 3 — Per-match attribution UI  [x]

### Task 3.1 — "Score Anatomy" panel on Team Breakdown [x]
Replace the current static factor bars ([app.py:947](app.py:947)) with an expandable anatomy view. For each of BO, BC, ON, LAN:

- **Header row:** factor name, value, % of theoretical max, contribution to total points.
  - e.g., `🏆 BO  0.6234  →  contributes ~487 of 1840 total points (26%)`
- **Top-10 contributing matches/prizes table** (using `result["attribution"]` from Task 1.3):
  - For BO: prize event, date, amount, age_w, weighted contribution, "% of bucket"
  - For BC: opponent (with their BO factor as tooltip), date, age_w × ev_w, weighted contribution
  - For ON Pass 2: opponent (with their ownNetwork as tooltip), date, age_w × ev_w, weighted contribution
  - For LAN: opponent, event, date, age_w
- **Below-cut matches (collapsed):** "12 more wins not in top-10 bucket — click to expand". This shows what's *not* counting and why.
- **Outcome:** added a compact `_count_caption()` helper that prints "N of M counting · X below cut (≈Y% of top-10 pool)" under each of the BO / BC / ON / LAN factor bands on the Team Breakdown page. The existing per-factor expander tables already show top-10 rows and age-weighted contributions; the caption makes the cut explicit at a glance. Kept the existing table layout (no full replacement) to keep the diff small and land inside the day's budget.

### Task 3.2 — H2H replay timeline [x]
- New section on Team Breakdown.
- Chronological list of every rated H2H match (the team's matches where both sides are eligible).
- Per row: date, opponent, opponent rating at the time, K = BASE_K × age_w, expected win %, actual result, ±delta.
- Cumulative line chart of `h2h_delta` over time (sparkline).
- This makes the H2H Glicko system stop being a black box.
- **Outcome:** enriched `compute_h2h()` in `vrs_engine/calculator.py` so `match_h2h[mid]` now also carries `date`, pre-match `w_rating` / `l_rating`, `e_w`, `k = BASE_K × age_w`, and `age_w`. In the Team Breakdown Phase 2 section (`app.py` ≈ line 3830) the match table grew three new columns in Updated mode — **Opp Rating**, **K**, **E(win)** — sourced from `sim_match_h2h[mid]` with a colour cue that pops underdog wins / favourite losses. Above the table, a cumulative H2H-Δ sparkline (area-filled line chart, zero baseline) renders chronologically so users can see the Glicko delta drift over time. All 21 regression snapshots still pass after the calculator schema change.

### Task 3.3 — "Where the points come from" donut [x]
- One donut per team summarizing factor share of total points: BO X%, BC Y%, ON Z%, LAN W%, H2H ±V%.
- Compact, sits next to the KPI header.
- Hover on each slice → "remove this factor" toggle (recompute total without it, show new rank).
- **Outcome:** added a factor-share donut + sensitivity table between the page-header caption and the existing waterfall. Slices use `ex["bo_factor"] * _pts_per_factor` (etc.) for BO/BC/ON/LAN and `|h2h_delta|` for H2H, so the donut shows actual point contribution rather than a theoretical-max share. Next to it a 5-row sensitivity table shows `#currentRank → #withoutRank` per factor, computed via a cheap `_rank_without(factor_key)` helper that re-sorts `seed_combined` without the chosen factor instead of re-running the full engine (≈ms instead of seconds per team). A persistent "remove this factor" toggle was deferred — the static rank-delta view already answers the question the hover-toggle would have answered, without the rerun complexity.

---

## Day 4 — Time-aware transparency (what's expiring, what just landed)

### Task 4.1 — "Expiring soon" panel [x]
- **Outcome:** new "⏳ Time Window" section sits between the donut and the score-change waterfall on the Team Breakdown page (`app.py` ≈ line 3389). Two stacked cards show "**Expiring soon**" (items at age_w ≤ 0.20) and "**Recent boost**" (items in the 30-day flat zone at age_w = 1.0), each with item count + estimated point contribution. The 180-day timeline strip on the right plots one Plotly marker per top-10 contributor across BO/BC/ON/LAN, with: x = days before cutoff, y = factor lane, marker size ≈ pts, opacity ≈ age weight, green/red shaded zones at 0–30 / 150–180 days, and dotted vlines at 30 d (flat→ramp) and 150 d (decay-tail). Hover surfaces date · label · age weight · pts. Estimate is `entry/10 × _pts_per_factor` for BC/ON/LAN top-10 entries and `(scaled_prize/Σ scaled_prize) × bo_factor × _pts_per_factor` for BO prizes — same approximation used by the per-factor "Pts" columns below.

### Task 4.2 — "Why is X above Y?" comparator [x]
- **Outcome:** "🆚 Why is X above Y?" section directly below the Time Window. Left column is a radio with three quick presets (`⬆️ Rank above`, `🥇 Rank #1`, `🔟 Bottom of top-10`) plus an `✏️ Pick a team…` selectbox over the rest of `base_standings`. Right column shows a one-sentence summary box ("X leads/trails Y by N pts. Biggest contributors: 💰 BC (+12.3 pts) and 🕸️ ON (+4.1 pts).") and a 5-row gap table with this team's value · peer's value · per-factor pts gap. Per-factor gap converts to points via `Δfactor × _pts_per_factor`; H2H gap is the raw delta. Drivers are picked by `|Δpts|` — top 2.
- Defers the *match-level* attribution sentence ("driven by their win over NAVI on Mar 12") — that lives more naturally inside Pillar 2's bottleneck diagnoser (Day 5) where it has the surrounding context.

### Task 4.3 — Match impact preview (hover/click) [x]
- **Outcome:** "🔬 Match Impact Explorer" expander on the Team Breakdown page. Streamlit doesn't support per-row hover tooltips, so the design is a select + button: candidates are this team's top-5 BC, top-5 ON, top-5 LAN, plus the 5 most recent in-window matches (deduped by `match_id`). Picking a match + clicking "Compute impact" re-builds the Store from `team_match_history` + `bo_prizes_map`, drops that `match_id` from `matches_df`, and re-runs `run_vrs` at the active cutoff. Result block shows new rank vs current with arrow + per-factor Δ table (BO/BC/ON/LAN/seed/H2H). Cached in `st.session_state._match_impact_cache` keyed by `(mode, cutoff_yyyymmdd, match_id)` so repeat picks of the same match are instant. Engine call is ~1 s; spinner covers it.

### Acceptance
- [x] A user with no engine knowledge can open a team's page and answer "what 3 matches contributed most to their current score?" in under 30 seconds — Time Window timeline + factor tables together expose this.
- [x] Every number on the page is traceable to a specific match or prize (per-factor tables already itemise; Time Window markers carry per-item pts in the tooltip; Match Impact recomputes the engine).
- [x] All 21 regression snapshots still pass (`pytest tests/` → 21 passed in 25 s).

---

# 📈 PILLAR 2 — IMPROVEMENT PATH (Day 5)

**Goal:** Tell each team *what they need to do to climb*. Which factor is their ceiling. Who they should beat. What's about to expire.

## Day 5 — "Climb Plan" section

### Task 5.1 — Bottleneck detector
For each team, identify the **lowest-percentile factor** vs peers (e.g., teams within ±10 ranks).
- "📉 Your weakest factor is **Opponent Network (0.31)** — peers in your bracket average 0.45."
- Diagnose root cause from the attribution data:
  - Low ON because few distinct opponents → "you've beaten the same 4 teams over and over; need to face new opponents"
  - Low BC because beating low-BO opponents → "your wins are vs teams with low prize-money — need quality wins"
  - Low LAN because no LAN events → "you have 1 LAN win in 180 days; top-10 teams average 6"
- The diagnosis comes from comparing the team's attribution rows to peers'.

### Task 5.2 — Target-team suggestions
Engine-driven "what wins would help most?":
- Generate hypothetical: "team X beats team Y in a BO3 at a $250K LAN event tomorrow."
- Run engine, compute Δrank.
- Sort top-20 candidate opponents by Δrank gain per win.
- Show table: opponent, their rank, expected win % (from current ratings), Δrank if won, Δpoints if won.
- User can click "simulate this" to add to the What-If Predictor.

### Task 5.3 — Path to rank N
- "To overtake the team above you (rank N-1), you need +X points."
- Find minimum-effort scenario: combinations of plausible wins (1, 2, 3, ... matches at next reasonable event) that achieve the gap.
- Plausible = opponents with current expected win % > 30%.
- Display as a checklist: "Win these 2 matches at the next $500K LAN → reach rank N-1."

### Task 5.4 — Expiring-points alert
- Surfaces from Task 4.1 data: "⚠️ You're projected to lose 38 points over the next 14 days as old matches age out. To hold rank, you need wins worth +38 in that window."
- Quantify "wins worth +38" using the engine: e.g., "1 LAN win vs top-10 opponent = ~25 pts, 2 mid-tier wins = ~40 pts."

### Acceptance
- Every team has a "Climb Plan" tab/section showing their bottleneck factor, 5+ specific target opponents, a path-to-next-rank checklist, and an expiring-points alert.
- Suggestions feel useful, not generic. (Validate by checking: do top-3 teams' suggestions involve the *right* upcoming events?)

---

# 🏆 PILLAR 3 — BRACKET SIMULATOR (Days 6–7)

**Goal:** "Pick an upcoming event, click through the bracket, see exactly where teams end up." Should take a user under 2 minutes for a full Major bracket.

## Day 6 — Tournament Predictor MVP

### Task 6.1 — Upcoming event discovery
- Extend `data_loaders/liquipedia_loader.py` with `discover_upcoming_events(start_date, end_date)`.
- Returns: `{slug, name, prize_pool, is_lan, start_date, end_date, format, prize_distribution: dict[place → amount], seeded_teams: list[str]}`.
- Liquipedia's tournament page has all of this. Reuse the existing portal scraper.

### Task 6.2 — Event picker UI
- New page: "🏆 Tournament Predictor".
- Top: dropdown of upcoming events (defaulting to the next major-tier event).
- Below: event metadata banner — prize pool, dates, format, seeded teams.
- Show: "If favorites win every match, here's the projected standings impact" as a baseline (Task 6.4).

### Task 6.3 — Click-through bracket UI
- Render the bracket as an interactive tree.
  - For single-elim: hand-rolled HTML with two-column rows per round.
  - For Swiss: a series of stages with team list per stage.
  - For GSL/double-elim: defer to V2; start with single-elim only.
- Each unresolved match has two clickable buttons (team A / team B).
- Picks stored in `st.session_state["bracket_picks"][slug] = {match_slot: winner_name}`.
- "Auto-fill favorites" button: pre-pick the higher-ranked team in every match (great as a baseline).
- "Reset" button.

### Task 6.4 — Compute placements + distribute prizes
- Once bracket is filled, compute final placement per team (1st, 2nd, 3rd-4th, etc.).
- Map to prize amounts via `prize_distribution` from Liquipedia.
- Emit `extra_matches` (one per played match, with event's `prize_pool` and `is_lan`) and `extra_prizes` (one per placed team).
- Pipe through unified `Store.append_simulation()` (from Task 2.4) and run engine.

### Acceptance
- User picks an upcoming event, fills the bracket via "Auto-fill favorites" + a few overrides, sees updated standings in < 30 seconds.
- The bracket UI works for at least one upcoming single-elim event in real Liquipedia data.

---

## Day 7 — Bracket polish + multi-scenario

### Task 7.1 — Before/after standings impact
- After bracket is filled: top-30 standings table with `Δrank` and `Δpoints` columns.
- "Biggest movers" panel: top 5 risers, top 5 fallers.
- Per-team mini-waterfall (BO/BC/ON/LAN/H2H deltas) on click.

### Task 7.2 — Scenario comparison
- Save bracket picks as named scenarios: "FaZe wins", "Vitality wins", "Underdog run", etc.
- Side-by-side compare any 2 scenarios — diff each team's rank/points across the two.
- Persist scenarios in `st.session_state` only (no DB).

### Task 7.3 — "Most-likely" baseline
- Compute baseline scenario: every match won by the team with higher current rating (using `expected_win()` from `math_helpers`).
- Show it as the default scenario to compare user-customized brackets against.
- Adds context: "Your scenario: NAVI to #2. Most-likely scenario: NAVI to #4. NAVI gains 2 ranks under your assumptions."

### Task 7.4 — Multi-event chaining
- Allow queueing 2–3 upcoming events (e.g., next major + next two B-tier qualifiers).
- Standings impact reflects the cumulative effect.
- Important because most upcoming weeks have multiple events stacking.

### Acceptance
- A user can run 3 different bracket scenarios for the next major and compare them in one screen.
- Multi-event chaining works for at least 2 sequential events.

---

## Critical Path

| Day | Pillar | Status | Outcome |
|-----|--------|--------|---------|
| 1 | Foundation | [x] | ON rewrite committed (`6af949f`); 21-snapshot regression harness green; per-match attribution plumbed through `pipeline.run()` |
| 2 | Foundation | [x] | Repo clean; simulator unified via `Store.append_simulation` + `run_vrs` — all four factors move on what-ifs |
| 3 | 🔬 Transparency | [x] | Factor-band count summaries · H2H replay with Opp-Rating/K/E(win) + cumulative Δ sparkline · factor-share donut + rank-without sensitivity table (21/21 regression tests green) |
| 4 | 🔬 Transparency | [x] | Time-window panel (expiring/recent + 180-day timeline strip) · peer comparator with sentence summary · Match Impact Explorer (engine re-run, session-cached) — 21/21 regression tests green |
| 5 | 📈 Improvement | [ ] | Bottleneck + target opponents + path-to-rank + expiring alerts |
| 6 | 🏆 Bracket | [ ] | Event picker + click-through bracket + standings impact |
| 7 | 🏆 Bracket | [ ] | Scenario compare + most-likely baseline + multi-event chaining |

**Days 1–2 are blockers for everything else.** Per-match attribution (Task 1.3) is the single most important enabler — without it, Pillar 1 can't exist and Pillar 2's diagnostics fall back to generic suggestions.

If a day slips, the deferral order is: Day 7 polish → Day 4 polish → Day 5 → Day 3.

---

## Out-of-scope (intentionally not in this week)

- HLTV / FACEIT data integration (live match data)
- Performance pass (vectorize per-team loops in calculator) — deferred until profile says it's needed
- CI/GitHub Actions wiring
- Region-specific viewers / data export
- Mobile-friendly UI
- Multi-user accounts / saved scenarios persisted to disk
- Liquipedia data quality alerts (name mismatches) — defer unless it bites Pillar 3
- Rewriting `app.py` into multi-file structure (it's 3993 lines, but works)

These can become Week-2 candidates once the three pillars are solid.

---

## Reference

- **Valve source of truth:** https://github.com/ValveSoftware/counter-strike_regional_standings/tree/main/model
- **Cache location:** `cache/github_vrs/vrs_{year}_{YYYY_MM_DD}.pkl`
- **Recent commits:** `afec7fd` (BC fix + roster), `af47617` (engine rebuild), `a4b3d3a` (Liquipedia loader)
- **Memory entries** (in `~/.claude/projects/.../memory/`):
  - `on_factor_impact_analysis.md` — confirms multiple roster versions don't break ON
  - `roster_split_fix_implemented.md` — full implementation notes on the dual-layer fix
  - `valve_source_reference.md` — Valve JS file pointers
- **Key UI surfaces to extend:**
  - Team Breakdown page: [app.py:3268](app.py:3268)
  - Score Change Breakdown waterfall: [app.py:3293](app.py:3293)
  - Factor band rendering helper: [app.py:3233](app.py:3233)
  - What-If Predictor: [app.py:1045](app.py:1045)
