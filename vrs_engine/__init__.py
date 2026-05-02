"""
VRS Engine — Public Interface

Primary entry point
-------------------
    from vrs_engine import Store, run_vrs

    store = Store.from_valve(team_match_history, bo_prizes_map)
    store.append_liquipedia(liq_df)          # optional
    result = run_vrs(store, cutoff=datetime.now())

    standings  = result["standings"]         # pd.DataFrame
    match_h2h  = result["match_h2h"]         # dict[match_id → detail]

Constants and math helpers are also exported for use in app.py.
"""

# ── Core pipeline ─────────────────────────────────────────────────────────────
from .store    import Store
from .pipeline import run as run_vrs

# ── Constants ────────────────────────────────────────────────────────────────
from .constants import (
    DECAY_DAYS, FLAT_DAYS, DECAY_RAMP,
    RD_FIXED, Q_GLICKO, BASE_K, PRIZE_CAP,
    TOP_N, SEED_MIN, SEED_MAX,
)

# ── Math helpers (used directly in app.py for display / what-if logic) ────────
from .math_helpers import (
    curve, event_stakes, age_weight, lerp,
    g_rd, G_FIXED, expected_win, top_n_sum,
    first_monday_of_month, next_valve_publication, prev_valve_publication,
)

__all__ = [
    # Pipeline
    "Store",
    "run_vrs",
    # Constants
    "DECAY_DAYS", "FLAT_DAYS", "DECAY_RAMP",
    "RD_FIXED", "Q_GLICKO", "BASE_K", "PRIZE_CAP",
    "TOP_N", "SEED_MIN", "SEED_MAX",
    # Math helpers
    "curve", "event_stakes", "age_weight", "lerp",
    "g_rd", "G_FIXED", "expected_win", "top_n_sum",
    # Valve publication schedule helpers
    "first_monday_of_month", "next_valve_publication", "prev_valve_publication",
]
