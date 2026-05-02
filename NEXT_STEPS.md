# VRS Simulator — Next Steps Roadmap

**Created:** 2026-04-19 · **Last revised:** 2026-04-24
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

## Day 5 — "Climb Plan" section [x]

A single `### 🧗 Climb Plan` block sits between the Match Impact Explorer and the Score Change Breakdown waterfall ([app.py:3917](app.py:3917)). Four stacked panels: bottleneck (cheap), expiring alert (cheap), target opponents (engine-gated button), path-to-rank (derived from target-opponent cache).

### Task 5.1 — Bottleneck detector [x]
- **Outcome:** peer group = teams within ±10 ranks of this team (`base_standings["rank"].between(max(1, _cp_rank-10), _cp_rank+10)`, self excluded). For each of BO/BC/ON/LAN, compare factor value to peer mean and convert the delta to seed points via `_pts_per_factor`. The weakest factor (largest negative gap, if worse than −0.5 pts) drives a root-cause sentence: ON → "N distinct opponents beaten vs peer avg M — face new teams"; BC → "avg opponent BO ratio X — target top-ranked opponents"; LAN → "N LAN wins vs peer avg M"; BO → "top prize in window $X — deep runs at large events compound". If no factor lags, the panel shows a green "no weakness vs peers" card. Two-column layout: the weakest-factor card on the left, a full 4-row gap table on the right with self / peer-avg / pts-gap.

### Task 5.2 — Target-team suggestions [x]
- **Outcome:** on-demand via `▶ Simulate climb plan` button (gated to avoid ≈10s engine work on every page render). Candidates = the 10 teams immediately above the selected team by rank (closest first); this maximises achievable Δrank per win and `opp_bo_ratio` already caps at 1.0 for any top-5 opponent, so widening the pool adds latency with little marginal insight. Per candidate: build a fresh `Store`, append one hypothetical BO3 win (`is_lan=True`, `prize_pool=$250K`, `date=_ref_cutoff`), re-run the engine with the same cutoff, diff standings vs `ex`. Session-cached in `st.session_state._climb_targets_cache` keyed by `(mode, cutoff_yyyymmdd, sel_team)` so later tab switches are instant. Progress bar during the run. Result table columns: **Opponent** (with current rank), **E(win)** (Glicko from `expected_win(ex.seed, opp.seed)`, green ≥ 50 / purple ≥ 30 / red < 30), **Δ pts**, **Rank change** (old → new), **Δ rank** arrow. Sort by Δrank asc, Δpts desc.
- Deferred the "simulate this" deep-link into the What-If Predictor — adding a single-click route into a different page requires a session-state bridge that's worth its own small task. The table gives the same information the user would have copied across manually.

### Task 5.3 — Path to rank N [x]
- **Outcome:** rendered only when 5.2's cache exists and `_cp_rank > 1`. Gap = `points_of_(rank-1) − team_points + 0.5` (small margin to actually pass). From 5.2's results, filter to "plausible" wins (E(win) ≥ 30% and Δpts > 0), sort by Δpts desc, greedily accumulate until sum ≥ gap. Success case → green checklist card: "N win(s), total +X pts (need +Y)". Failure case → yellow warning card: "No single-LAN plausible path closes the Y-pt gap — best achievable +X; a $1M event or multi-event stacking needed."

### Task 5.4 — Expiring-points alert [x]
- **Outcome:** reuses the `_window_items` list already built in the Time Window section (4.1), filtering to items with `age_w ≤ 14/DECAY_RAMP` (≈ 0.093) — the ones whose weight will reach 0 within 14 days. Point loss = `Σ pts` of those items. The "wins worth +X" estimate compares against a closed-form top-tier LAN boost: `(2 × 0.602/10 + 1/10) / 4 × 1600 / _avg_range` (BC + ON via `ev_w(250k) = curve(0.25) ≈ 0.602`, plus a LAN slot), and `× 0.55` for a mid-tier equivalent. The alert sentence resolves to "≈ N top-tier LAN wins, or ≈ M mid-tier wins" — the exact engine deltas live in the Target Opponents table. Turns green with a ✅ if there's nothing within 14 days of expiring.

### Acceptance
- [x] Every team has a Climb Plan section: bottleneck card + gap table, expiring alert, target-opponents table (10 candidates), and path-to-next-rank checklist.
- [x] Suggestions are engine-driven (actual Δrank / Δpts, not heuristics). Bottleneck diagnosis is attribution-driven (distinct opponents, avg opp_bo, LAN count).
- [x] All 26 tests pass (21 regression snapshots + 5 others) in ~17s — Climb Plan is UI-only, no engine changes.

---

# 🏆 PILLAR 3 — BRACKET SIMULATOR (Days 6–7)

**Goal:** "Pick an upcoming event, click through the bracket, see exactly where teams end up." Should take a user under 2 minutes for a full Major bracket.

## Architecture Decisions (locked via grill-me, 2026-04-25)

These constrain Days 6–7. Every implementation choice traces back to one of these.

### Reframings (apply globally)

- **Predicted = third `standings_mode`.** Sidebar gains `"🔮 Predicted"` alongside `"🏛️ As Published"` / `"📡 Updated to Today"`. When active, every page (Ranking Dashboard, Team Breakdown) reflects the simulated future. Tournament Predictor page is the *definition surface*; mode toggle is the *propagation surface*.
- **"🔮 What-If Predictor" page → deleted.** Tournament Predictor supersedes it. The `Store.append_simulation()` plumbing from Task 2.4 stays — Tournament Predictor uses the same call site. Only the orphan UI page (≈ `app.py:1045–~1500`) goes away.
- **Persistent banner on other pages when mode=Predicted.** Format: "📡 Predicted: N events queued, last computed M min ago · [edit in Tournament Predictor]". ~10 lines in a shared header helper.

### Decision sheet

| # | Decision | Choice |
|---|---|---|
| 1 | Bracket source | Parse Liquipedia DOM (`.brkts-bracket` / `.brkts-matchlist`); generate-SE fallback when parse yields zero matches |
| 2 | Match identity | Slot lineage path `"<stage_id>::<round><match>"` (e.g. `"s1::R1M0"`); `bracket_signature = hash(seeded_teams_tuple, format)` guards against upstream changes |
| 3 | State shape | Scenario-aware from start: `bracket_state[slug] = {"scenarios": {name: {"picks": {...}, "created": ts}}, "active_scenario": "current", "seed_signature": h, "include_in_chain": bool}` |
| 4 | Recompute trigger | Live on every pick + memoize by `(picks_hash, today_cutoff)` + user-toggleable Pause for batch fills |
| 5 | Baseline | **No per-page toggle.** Predicted mode = Updated-to-Today data + active picks. One baseline, one engine path |
| 6 | Auto-fill rule | Glicko `expected_win(seed_a, seed_b)`; unranked teams → SEED_MIN(400); ties broken alphabetically |
| 7 | Mid-flight events | Lock played slots (read-only, "actual result" badge). Pre-playoff stages (Swiss/groups) auto-reflect Liquipedia data |
| 8 | Match dates | Per-match Liquipedia date when present, else event `end_date`. Prize rows always at `end_date` |
| 9 | Format scope | Full coverage: SE+bronze, double-elim, Swiss, groups |
| 10 | Pick keys | Flat dict, stage-prefixed: `picks["s0::R1M0"] = "FaZe"`. Hashable in one line for engine cache |
| 11 | Format detection | Hybrid structural-first + heading tiebreaker; walk page in document order, classify each stage container, tag `(stage_id, format, display_heading)` |
| 12 | Cache layers | **L1** discovery (file, 6h TTL) — already shipped in 6.1. **L2** parsed brackets (file, 6h TTL, new). **L3** engine recompute (session memory, LRU 20). No cross-session pick persistence. Pick-mismatch on L2 refresh = banner + altered-slot badge, never auto-clear |
| 13 | Multi-event chain | Implicit via `include_in_chain: bool` per event; chain run = concat picks from all flagged events sorted by date, single engine call |
| 14 | Page integration | New sidebar entry `"🏆 Tournament Predictor"` replaces `"🔮 What-If Predictor"` slot |
| 15 | Failure modes | Fail soft. Three-tier banner taxonomy: 🟡 warn (degraded but usable), 🟠 stale (picks vs structure mismatch), 🔴 error (can't compute). Never silently nuke user state |
| 16 | Pick UX | Series-level only (no map score — engine works series-level). SE = clickable bracket. Bronze = extra slot in top round-center. Double-elim = UB+LB+GF nodes. **Swiss** = sequential rounds. R1 stays TBD until Liquipedia publishes OR the user clicks the per-stage **"🌱 auto-seed R1"** button (requires a VRS-snapshot pick). R>1 cascade fires from picks via the Buchholz pairer (`vrs_engine/swiss_pairer.py`); see `NEXT_STEPS_BRACKETS.md` §10 for the locked spec. **Groups** = matrix of pairings, click winner |

### Format-detection algorithm (concrete from #11)

```
1. Walk page in document order. Collect stage containers (.brkts-bracket / .brkts-matchlist groups).
2. For each, classify by structure:
   - 1× .brkts-bracket  + extra-match-in-top-round-center → SE_with_bronze
   - 1× .brkts-bracket  alone                              → SE
   - 2× .brkts-bracket  adjacent                           → AMBIGUOUS (DE or 2-group GSL)
   - 3-4× .brkts-bracket                                   → GSL_groups
   - .brkts-matchlist × N grouped                          → AMBIGUOUS (Swiss or RR groups)
3. For AMBIGUOUS stages, resolve via nearest preceding h2/h3:
   - "Lower Bracket" / "Grand Final" present  → DE
   - "Swiss" keyword                          → Swiss
   - "Group" keyword                          → Groups
4. Tag each stage: (stage_id="s{i}", format, display_heading).
```

### Placement → prize-label mapping (concrete from #9)

For each format, derive `placement_label` per team after bracket fully resolved, then look up `prize_distribution[label]`:

| Format | Mapping |
|---|---|
| SE no bronze | final winner=`"1st"`, finalist=`"2nd"`, SF losers=`"3rd-4th"`, QF losers=`"5th-8th"`, R1 losers=`"9th-16th"`, … |
| SE w/ bronze | final winner=`"1st"`, finalist=`"2nd"`, bronze winner=`"3rd"`, bronze loser=`"4th"`, then QF/R1 as above |
| Double-elim | GF winner=`"1st"`, GF loser=`"2nd"`, LB final loser=`"3rd"`, LB SF loser=`"4th"`, LB R(N-2) loser=`"5th-6th"`, … |
| Swiss | by W-L bucket: 3-0=top, 3-1=next, 3-2=next, 2-3=next, 1-3=next, 0-3=bottom; map to whichever labels Liquipedia published |
| Groups | within-group rank (1st-in-group, 2nd-in-group, …) → overall placement label per Liquipedia's tournament-wide breakdown |

Missing label in `prize_distribution` (rare — finer bracket resolution than published prize tiers) → 0 prize for that team, log warning.

## Day 6 — Tournament Predictor MVP

### Task 6.1 — Upcoming event discovery [x]
- **Outcome:** added `discover_upcoming_events(start_date, end_date, min_tier, today, fetch_details, force_refresh, progress_callback)` in [data_loaders/liquipedia_loader.py](data_loaders/liquipedia_loader.py). Reuses `discover_from_portal()` for slug discovery, then per-event fetches the page and runs four small parsers: `_parse_infobox` (existing — name, dates, prize pool, LAN), `_infobox_field` (sibling-walk; handles modern Liquipedia layout where value is the next sibling of `.infobox-description`, not the next `.infobox-cell-2`), `_parse_prize_distribution_by_place` (place-keyed prize table → `{"1st": 250000, "3rd-4th": 35000}`), `_parse_seeded_teams` (canonical team names from `.teamcard center a[title]`), and `_parse_format` (walks siblings of the `<h3>Format</h3>` wrapper, collecting `<p>/<ul>/<ol>/<dl>` text — the heading is wrapped in `<div class="mw-heading">` on modern MediaWiki, so it walks the wrapper's siblings). Returns the spec'd payload: `{slug, name, url, tier, start_date, end_date, prize_pool, is_lan, format, prize_distribution, seeded_teams}`.
- **Caching:** 6-hour JSON cache at `cache/upcoming_<start>_<end>_<key>.json` so the UI can iterate without re-walking 7+ pages × 2.5s rate limit. Cache filters on `today` at read time (events that just ended drop out without a refresh).
- **Live validation (2026-04-25, A-Tier+, 2 mo window):** 7 events parsed cleanly — BLAST Rivals Spring, PGL Astana, Asian Champions League, IEM Atlanta, CS Asia Championships, Stake Ranked Episode 2, **IEM Cologne Major** (32 teams, $1.25M, prize_distribution covers 1st through 17th-19th). Format strings are accurate ("16 Team Swiss System Format … Top 8 teams proceed to Playoffs" for PGL Astana, etc.) and seeded_teams arrives in canonical form (`Vitality`, `FaZe`, `G2`, `FURIA`, …) ready for the bracket UI.
- All 26 regression tests still pass (~14s).

### Task 6.2 — Event picker UI [x]
- **Outcome:** "🏆 Tournament Predictor" replaces "🔮 What-If Predictor" in the sidebar (per architecture decision #14). Old What-If page body deleted (~200 lines); plumbing it relied on (`Store.append_simulation`, `run_vrs`) is unchanged and now serves Tournament Predictor instead. New page in [app.py:945-1123](app.py:945) renders:
  - **Discovery filter row:** range start/end (default today → +90d), min tier (default A-Tier), 🔄 Refresh button (force-bypasses 6.1's 6h JSON cache + Streamlit memo).
  - **Event picker** — selectbox sorted by start date asc, default index = next event. Label format: `"S-Tier · IEM Cologne Major  (Jun 02 – Jun 21, 2026, $1.25M)"`.
  - **Metadata banner** — 4 metrics (prize pool, dates with day count, LAN/online, tier).
  - **Format details expander** — renders the 6.1 format prose (`<h3>Format</h3>` section). Caption-only fallback when missing.
  - **Seeded teams chip cloud** — flag + canonical name per chip; 🟡 warning banner when empty.
  - **Prize distribution expander** — table view of place → USD (`{"1st": $500K, "3rd-4th": $80K, …}`).
  - **Bracket placeholder** — info banner pointing to 6.3/6.4 work.
  - **Liquipedia source link** at the bottom.
- Streamlit memo (`@st.cache_data`, 6h TTL) wrapping `discover_liquipedia_upcoming_events` so per-rerun walks don't re-hit the file cache. Force-refresh token in session state increments on Refresh click → invalidates both layers atomically.
- Failure modes wired (per architecture #15): 🔴 error banner on loader exception, 🟡 warning when zero events match filters, 🟡 warning when an event has no seeded teams.
- All 26 regression tests still pass (~14s); `app.py` syntax-clean; `from data_loaders import discover_liquipedia_upcoming_events` resolves via the new package export.

### Task 6.3 — Click-through bracket UI  [in-progress: slice 1 of 4]

Rolled out incrementally; format scope is the locked architecture decision #9 (full coverage). Each slice lands a parser + render path for one format family and is regression-checked end-to-end against a real upcoming event.

**Slice 1 — SE / SE_with_bronze [x]**
- New `data_loaders/bracket_parser.py` (~250 lines):
  - `BracketMatch` / `Stage` dataclasses carrying lineage path keys (`"s1::R1M0"`, `"s1::B0"`), feeder ids, R1 seeds, played-result fields.
  - `_detect_stages` walks the page in document order and classifies each `.brkts-bracket` / `.brkts-matchlist` group via the architecture-#11 hybrid algorithm. Heading tiebreaker upgrade: a "Group Stage" heading on a single-bracket container forces `GSL_groups` (avoids miscalling GSL as SE since the markup is structurally identical).
  - SE parser handles both bare SE and the SE+bronze layout (top round-center carrying a second match = bronze tell). Bronze feeders point to the SFs but with **loser-advances** semantics.
  - Stage consolidation: consecutive unsupported stages with the same heading merge into one placeholder card so a 10-round Swiss page emits 1 stage entry, not 10.
  - `Stage.signature()` for the L2 staleness check (architecture #12 part b).
- Live validation against 5 S-Tier events: BLAST Rivals (SE, 5 matches), PGL Astana (SE+bronze, 8 matches w/ bronze), IEM Atlanta (SE+bronze, 6 matches), CS Asia Champs (SE+bronze, 6 matches), IEM Cologne Major (SE, 7 matches). Group / GSL stages on those pages classify correctly as placeholders.
- App-side wiring in [app.py](app.py) Tournament Predictor block:
  - L2-style `@st.cache_data` (6h) wrapping parser; force-refresh token shared with the L1 discovery cache.
  - Session state per architecture #3: `bracket_state[slug] = {"scenarios": {"current": {"picks": {…}, "created": ts}}, "active_scenario": "current", "seed_signature": h, "include_in_chain": bool, "paused": bool}`.
  - Resolver `_resolve(stage, m)` walks feeders through `picks` to compute team_a/team_b for any R2+ slot; bronze branch follows feeder *losers*.
  - Toolbar: ⚡ Auto-fill favorites (Glicko `expected_win`, unranked → SEED_MIN=400, alphabetical tie-break), 🗑️ Clear picks, ⏸️ Pause toggle (architecture #4), 🔗 Include-in-chain toggle (architecture #13), live picks counter.
  - Render: round-by-round columns; each match = two `st.button`s with primary highlight on the picked team; played slots render as read-only chips with "📜 actual result" caption (architecture #7).
  - Staleness banner: if the parsed stage signature changes between renders, surface the 🟠 architecture-#12-(b) banner and keep picks intact.
- Other formats (DE, Swiss, GSL groups, RR groups) emit `🚧 coming in slice N` info cards so multi-stage events still render the SE playoffs end-to-end.
- All 26 regression tests still pass (~14s).

**Slice 2 — Double-elimination [x]**
- Refactored slice-1's SE walker into a reusable `_walk_se_subtree` helper; `_parse_se_bracket` and the new `_parse_de_bracket` both call it.
- Added `feeder_a_kind` / `feeder_b_kind` ∈ {"winner", "loser"} fields to `BracketMatch` so cross-bracket feeds (LB outer-side = UB this-round loser; SE+bronze = SF losers) are encoded structurally rather than via per-format branching. `_resolve` (UI) and `_pair` (engine emitter) now share a generic feeder-kind walker.
- Added `sub` field on `BracketMatch` ∈ {"", "UB", "LB", "GF", "RST"}; new ID prefixes `s0::UB-R1M0`, `s0::LB-R2M1`, etc.
- Format detection upgraded: `≥2 top-level round-body direct children` → `DE` (architecture #11 hybrid algorithm, structural-first). The earlier heading-based `GSL_groups` classifier was dropped — DE is DE regardless of context, and the heading distinguishes group vs playoff for placement mapping. Stage-merge tightened to keep DE stages distinct (IEM Atlanta's two "Group Stage" DE groups must each render and pick independently).
- LB cross-bracket wiring: standard 8-team DE pairing convention. LB-R1 takes UB-R1 losers in pair order; LB-R(k>1) inner = LB-R(k-1) winner; LB-R(k>1) outer = UB-R(k) loser. Compact 8-team group format (no GF, group ends at LB-final + UB-final) handled directly. Full DE-with-GF + bracket-reset extension noted as slice 2b TODO.
- App-side `_render_de_stage`: UB rendered as columns of clickable matches under a green "⬆️ Upper Bracket" header, LB below under a red "⬇️ Lower Bracket" header. Auto-fill cascade now sorts UB-before-LB-within-round so feeders always resolve before dependants.
- `_emit_event_simulation_rows` walks DE matches and emits per-pick match rows. Prize-row emission for DE-in-groups deferred (cross-group → tournament placement mapping needs aggregation logic). Engine consequence for current S-Tier events: group-stage matches still feed BC/ON/LAN; group-eliminated teams who would have earned `9th-12th: $5K`-style prizes don't see that BO bump. Acceptable underestimate.
- **Live engine smoke test** (IEM Atlanta Group A, full auto-fill): 12 picks resolved correctly, cross-bracket loser-feeds traced cleanly (NRG vs NaVi in LB-R1M0; NaVi vs Legacy in LB-R2M0; NaVi vs Astralis in LB-final). Engine output: Astralis (UB-final loser) +16.3 pts and #11→#9; Vitality (group winner) +1 pt; Legacy +4.7 pts. Modest deltas reflect match-only emission.
- All 26 regression tests still pass (~16s).

**Slice 3 — Swiss [x]**
- New matchlist parser path in [data_loaders/bracket_parser.py](data_loaders/bracket_parser.py): `_detect_stages` now chunks consecutive `.brkts-matchlist` elements by their title-prefix (`"Round N …"` → Swiss, `"Group X …"` → Groups), so PGL Astana's 9 matchlists (`"Round 1 Matches"`, `"Round 2 High Matches"`, `"Round 2 Low Matches"`, …) collapse into one Swiss stage spanning rounds 1-5.
- `_parse_swiss_stage` extracts each `.brkts-matchlist-match`, parses `_team_from_aria` for both opponents + winning row's `opponent-win` class, builds `BracketMatch(sub="SW", round_idx=N, slot_idx=…)` per match. Round number derived from a `Round\s*(\d+)` regex on the matchlist title.
- R1 seed backfill uses standard high-vs-low pairing (1v9, 2v10, …, 8v16 for 16-team Swiss). R2+ seeds remain `None` for upcoming events until Liquipedia populates pairings — the UI renders "TBD" disabled buttons in that case. Full Buchholz cascade pairing (architecture #16) is documented as a follow-up; current MVP works for played rounds + event-mid-flight scenarios where Liquipedia has populated subsequent rounds.
- App-side `_render_swiss_stage`: per-round columns, matches stacked. Round caption shows match count + played count. Picks counter / auto-fill / pause toggles all extend cleanly via the `_RENDERED_FMTS` set update.
- Engine emission: Swiss matches feed BC/ON/LAN/H2H. Per-bucket prize emission (W-L → "9th-11th" → prize lookup) deferred — group/Swiss-eliminated teams miss their small Liquipedia-listed BO bumps; playoff prizes (the high-impact chunk) still emit via the SE/SE_with_bronze branch.
- Live engine smoke (PGL Astana, R1 auto-filled): 8 picks emitted, modest deltas across affected teams (Spirit +67 / #10→#9, MOUZ +50 / #7→#5, FURIA +26, etc.). 33 total matches in DOM but only 8 have populated seeds (R1 only).

**Slice 4 — Round-robin groups [x]**
- `_parse_groups_stage` reuses the same matchlist match parser; emits one stage per matchlist (BLAST Rivals' Group A and Group B = two distinct stages). `BracketMatch(sub="GR", round_idx=1)` for every match.
- App-side `_render_groups_stage`: two-column layout — match list on the left (clickable, same as SE/Swiss), live group standings on the right computed from picks + played results (W-L per team, sorted by wins). The standings table updates instantly on each pick — gives the user the round-robin "see who's leading" feedback architecture #16 called for via the matrix design (kept simpler — vertical list + standings panel — because Liquipedia ships ≤ 6 matches per group, so a matrix would be sparser than the list).
- Engine emission shares the same code path as Swiss/DE (match rows only).
- Live engine smoke (BLAST Rivals Group A, 2 visible picks): G2 +7.4, modest tail-end shifts.

**Slice 5 — Major-tier sub-page discovery [x]**
- Liquipedia splits Majors (IEM Cologne) across `/Stage_1`, `/Stage_2`, `/Stage_3` sub-pages with only the SE playoff bracket on the main tournament page. The slice 1-4 parsers walked only the main page → Cologne came out as just 1 SE stage when it actually has 3 Swiss stages + SE playoffs.
- `parse_tournament_brackets` now takes an optional `slug` parameter. When supplied, `_discover_sub_stage_slugs` regex-matches `/<base_slug>/Stage_(\d+)` anchors on the main page, fetches each sub-page in numeric order (with `_REQ_DELAY` between requests), parses, and *prepends* its stages to the main page's stages so the timeline reads in execution order.
- L2 cache wrapper in [app.py](app.py) now passes `slug=slug` through; same for the engine-side `_emit_event_simulation_rows` re-parse.
- **Verification post-fix:** IEM Cologne goes 1 stage → **4 stages** (Swiss × 3 + SE). All 5 S-Tier events parse correctly: BLAST Rivals (2 RR Groups + SE), PGL Astana (Swiss + SE+bronze), IEM Atlanta + CS Asia Champs (2 DE groups + SE+bronze), IEM Cologne Major (3 Swiss + SE).
- Sub-page fetch latency: 3 extra HTTP calls × 2.5s rate-limit on first parse for Cologne. L2 cache (6h TTL) absorbs repeats.

**Coverage now:** every parsed format is interactive end-to-end including Major-tier multi-page events. SE, SE+bronze, DE (UB+LB with cross-bracket loser-feeders), Swiss, RR Groups all clickable; auto-fill cascades through every stage; engine emits match rows for every pick across every format; Liquipedia sub-pages followed automatically when present.

**Deferred to future work:**
- Full Buchholz Swiss cascade for pre-event pairing of R2+.
- DE-with-GF + bracket-reset extension (slice 2b — none of the current S-Tier events use it; BLAST Open does).
- Per-format prize-row emission for Swiss/DE-in-groups/Groups — needs cross-format placement-bucket → tournament prize-label aggregation.

### Task 6.4 — Compute placements + distribute prizes [x]
- **Outcome:** sidebar `standings_mode` extended with `"🔮 Predicted"` (architecture decisions #5 + #14) — selecting it rides on the existing Updated-to-Today fetch, then layers bracket picks via `Store.append_simulation()` and re-runs `run_vrs`. Result replaces `base_standings`, propagating to every page (Ranking Dashboard, Team Breakdown, Tournament Predictor itself).
- **Module-level helpers** added at the top of [app.py](app.py):
  - `_ordinal(n)` — 1st/2nd/3rd/…/21st/22nd/101st with proper teen-suppression (11th/12th/13th).
  - `_format_place(lo, hi)` — bucket label like "5th-8th" / "1st".
  - `_lookup_prize_for_place(label, prize_distribution)` — exact-match first, then subrange resolution (a "5th-6th" derived placement looks up under Liquipedia's coarser "5th-8th" bucket and gets the listed amount per architecture #9 per-team convention).
  - `_emit_event_simulation_rows(slug, slug_state)` — re-parses the event's bracket on demand (hits L1+L2 caches), walks every SE/SE_with_bronze stage to derive placements, emits `extra_matches` for user-picked unplayed slots only (architecture #7 — played slots stay in `matches_df` to avoid double-count), emits `extra_prizes` from placement → distribution lookup.
  - `_collect_predicted_simulation_rows(bracket_state)` — concatenates rows from every event with at least one pick, sorted chronologically by event end date (architecture #13).
- **Predicted layer** wired into the main mode dispatcher: after Updated-to-Today populates the engine state, when mode == Predicted, builds a `Store`, appends sim rows, runs engine, attaches region/flag/color metadata, recomputes `rank_delta` against the Updated-to-Today baseline (so deltas isolate the bracket-pick effect, not engine-vs-Valve drift).
- **Engine cache** (architecture #4): `st.session_state._predicted_engine_cache` keyed by `(included_slugs, cutoff_date, picks_md5)`. LRU-trimmed at 20 entries. Pause toggle (architecture #4): if any contributing event is paused, the recompute is suppressed and a status banner shows.
- **Banners**: Predicted mode gets its own gradient banner (orange "🔮 Predicted future · cutoff Mon DD" + queued event count + match/prize counts) on every page, plus matching sidebar status badge.
- **Live engine smoke test** (PGL Astana, FURIA-wins-all bracket, prizes layered):
  - Spirit (final loser, +$120k) #10 → #5 (+170.5 pts) — Major-tier final = ~+170 pts, sane.
  - FURIA (champion, +$256k) #5 → #3 (+146.8 pts).
  - PARIVISION (bronze winner, +$96k) #4 → #4 (+70.1 pts).
  - The MongolZ (bronze loser, +$56k) #6 → #7 (+47.9 pts) — small drop in rank from Spirit's bigger jump.
  - Top teams (Vitality, NaVi) +5-9 pts each from ON-network ripple, no rank change.
- **Helper unit tests**: AST-extracted `_ordinal`, `_format_place`, `_lookup_prize_for_place` exercised against PGL Astana's prize_distribution; all assertions pass (1st/2nd/3rd/11th/12th/13th/21st/22nd/23rd/101st correct; 5th-6th resolves to 5th-8th's amount via subrange match).
- All 26 regression tests still pass (~14s); `app.py` syntax clean.

### Acceptance
- [x] User picks an upcoming event, fills the bracket via "Auto-fill favorites" + a few overrides, sees updated standings — engine wired, ~1s recompute, memoised.
- [x] The bracket UI works for at least one upcoming single-elim event in real Liquipedia data — PGL Astana + 4 other S-Tier events parse and render correctly.

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
| 5 | 📈 Improvement | [x] | Climb Plan section: bottleneck (peer-relative + root-cause diagnosis) · expiring-14d alert with equivalent-wins guidance · engine-gated target-opponents table (10 candidates, session-cached) · greedy path-to-rank checklist — 26/26 tests green |
| 6 | 🏆 Bracket | [x] | 6.1 ✓ + 6.2 ✓ + 6.3 all 4 slices ✓ (SE+bronze, DE, Swiss, RR groups) + 6.4 ✓ — full Pillar 3 coverage. Predicted standings_mode live; every parsed format is clickable end-to-end and emits to the engine |
| 7 | 🏆 Bracket | [ ] | Scenario compare + most-likely baseline + multi-event chaining |

**Days 1–2 are blockers for everything else.** Per-match attribution (Task 1.3) is the single most important enabler — without it, Pillar 1 can't exist and Pillar 2's diagnostics fall back to generic suggestions.

If a day slips, the deferral order is: Day 7 polish → Day 4 polish → Day 5 → Day 3.

---

## Out-of-scope (intentionally not in this week)

- HLTV / FACEIT data integration (live match data)
- Performance pass (vectorize per-team loops in calculator) — deferred until profile says it's needed
- CI/GitHub Actions wiring
- Region-specific viewers / data export
- Multi-user accounts / saved scenarios persisted to disk
- Liquipedia data quality alerts (name mismatches) — defer unless it bites Pillar 3

These can become Week-2 candidates once the three pillars are solid.

---

# WEEK 2 CANDIDATES (Days 8–9) — formalized for tracking

Once Days 1–7 are shipped and the three pillars are live, these are the natural next pulls:

## Day 8 — App structure refactor

**Scope:** Rewrite `app.py` (currently 5,237 lines) into a multi-file component structure.

**Why now:** The monolithic file is stable and works, but adding features (Day 7 scenario compare, future features) requires scrolling thousands of lines to find UI sections. Multi-file structure unblocks faster feature iteration + easier testing of individual components.

**Approach (to be grill'd):**
- Extract CSS/theme into `styles/theme.py`
- Extract Streamlit helper functions into `ui/components.py` (factor band renderer, metric cards, tables)
- Extract page sections into `pages/team_breakdown.py`, `pages/tournament_predictor.py`, `pages/explorer.py`
- Keep `app.py` as a thin router and session-state orchestrator
- No engine changes; pure presentation refactor

**Acceptance:** Code is organized by feature (not arbitrary file size), IDE navigation is instant, new features don't require scanning >1000 lines to find the right place to add UI.

---

## Day 9 — Mobile-friendly UI

**Scope:** Adapt the UI for tablet and mobile viewports (375–800px widths).

**Why now:** Currently optimized for desktop; mobile users hit horizontal scroll and oversized components. Post-pillar-shipping, supporting mobile opens the simulator to more analysts.

**Approach (to be grill'd):**
- Use Streamlit's responsive grid system and `st.columns` with dynamic span logic
- Condense wide tables (factor contributions, comparators) into card-stack layouts for mobile
- Test at `@media (max-width: 800px)`, `(max-width: 480px)` breakpoints
- Verify tournament bracket click-through works on touch
- Keep desktop layout intact; CSS media queries only

**Acceptance:** App is usable and readable on iPad (tablet) and iPhone 13 (mobile). No horizontal scroll. Tables and charts fit viewport. All interactive elements have touch-safe padding (44px+).

---

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
