"""
VRS Engine Constants

All numerical parameters for the CS2 Valve Regional Standings calculation.
"""

import math

# Time Decay Parameters
DECAY_DAYS = 180           # Total lookback window (days)
FLAT_DAYS = 30             # Days at the start that hold 1.0 age weight
DECAY_RAMP = 150           # Days over which weight ramps 1.0 → 0.0

# Glicko Parameters
RD_FIXED = 75              # Fixed Glicko RD → Elo-equivalent behaviour
Q_GLICKO = math.log(10) / 400  # ≈ 0.005756

# H2H Phase Parameters
BASE_K = 32                # H2H base K-factor (scaled by age only)

# Prize Pool & Scoring
PRIZE_CAP = 1_000_000      # Prize pool cap for event stakes (USD)

# Factor Calculation
TOP_N = 10                 # Bucket size — top-N results per factor

# Seeding Range
SEED_MIN = 400
SEED_MAX = 2000
