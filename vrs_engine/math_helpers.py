"""
VRS Engine Math Helpers

Core mathematical functions: normalization curve, time decay, interpolation, Glicko.
"""

import math
from datetime import datetime
from .constants import FLAT_DAYS, DECAY_RAMP, DECAY_DAYS, Q_GLICKO, PRIZE_CAP, RD_FIXED


def curve(x: float) -> float:
    """
    Valve's normalisation curve: f(x) = 1 / (1 + |log₁₀(x)|)

    · f(1.0)  = 1.000  — exactly at the reference point
    · f(0.1)  = 0.500  — one order of magnitude below
    · f(0.01) = 0.333  — two orders of magnitude below
    · f(10)   = 0.500  — one order of magnitude above
    · Always in (0, 1] for x > 0; peaks at x = 1
    """
    if x <= 0:
        return 0.0
    return 1.0 / (1.0 + abs(math.log10(x)))


def event_stakes(prize_pool: float) -> float:
    """
    Event weight applied to BC and ON calculations:
    stakes = curve(pool / $1,000,000)

    A $1M event   → stakes = curve(1.0) = 1.000
    A $250k event → stakes = curve(0.25) ≈ 0.602
    A $100k event → stakes = curve(0.1)  = 0.500
    Applied to: BC, ON (NOT to BO or LAN)
    """
    ratio = min(max(prize_pool, 1.0), PRIZE_CAP) / PRIZE_CAP
    return curve(ratio)


def age_weight(match_date: datetime, cutoff: datetime) -> float:
    """
    Age Weight (Time Modifier) — verified against official Vitality data.

    · Days 0–30 before cutoff: weight = 1.000 (flat)
    · Days 31–180:             weight = 1.0 – (days_ago – 30) / 150
    · Beyond 180 days:         excluded (weight = 0.0)

    Examples at cutoff 2026-03-02:
      2026-01-31 (30 days):  1.000  ✓
      2026-01-24 (37 days):  0.953  ✓
      2025-12-14 (78 days):  0.680  ✓
      2025-11-09 (113 days): 0.447  ✓
      2025-10-12 (141 days): 0.260  ✓
      2025-09-07 (176 days): 0.027  ✓
    """
    days_ago = (cutoff - match_date).days
    if days_ago < 0:
        return 1.0          # future match (should not occur)
    if days_ago <= FLAT_DAYS:
        return 1.0
    if days_ago >= DECAY_DAYS:
        return 0.0
    return 1.0 - (days_ago - FLAT_DAYS) / DECAY_RAMP


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + (b–a)×t, t clamped to [0, 1]."""
    return a + (b - a) * max(0.0, min(1.0, t))


def g_rd(rd: float = RD_FIXED) -> float:
    """
    Glicko g(RD) dampening factor.
    With RD fixed at 75 this is constant ≈ 0.9728.
    """
    return 1.0 / math.sqrt(1.0 + 3.0 * Q_GLICKO**2 * rd**2 / math.pi**2)


G_FIXED = g_rd(RD_FIXED)   # pre-computed; constant for the entire run


def expected_win(r_self: float, r_opp: float) -> float:
    """
    Glicko expected score for r_self against r_opp:
    E = 1 / (1 + 10^(−g(RD) × (r_self − r_opp) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** (-G_FIXED * (r_self - r_opp) / 400.0))


def top_n_sum(values: list, n: int = 10) -> float:
    """Return the sum of the n largest values in the list."""
    return float(sum(sorted(values, reverse=True)[:n]))
