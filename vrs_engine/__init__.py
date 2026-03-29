"""
VRS Engine

Core Valve Regional Standings (VRS) calculation engine.
"""

from .core import compute_vrs
from .constants import (
    DECAY_DAYS, FLAT_DAYS, DECAY_RAMP, RD_FIXED, Q_GLICKO,
    BASE_K, PRIZE_CAP, TOP_N, ON_ITERS, SEED_MIN, SEED_MAX
)
from .math_helpers import (
    curve, event_stakes, age_weight, lerp,
    g_rd, expected_win, top_n_sum, G_FIXED
)

__all__ = [
    "compute_vrs",
    "DECAY_DAYS", "FLAT_DAYS", "DECAY_RAMP",
    "RD_FIXED", "Q_GLICKO", "BASE_K", "PRIZE_CAP", "TOP_N", "ON_ITERS",
    "SEED_MIN", "SEED_MAX",
    "curve", "event_stakes", "age_weight", "lerp",
    "g_rd", "expected_win", "top_n_sum", "G_FIXED",
]
