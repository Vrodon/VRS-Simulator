"""
UI Helpers

Display formatting utilities for Streamlit.
"""

import pandas as pd
from utils.team_meta import KNOWN_META


def region_pill(region: str, regional_rank: int = 0) -> str:
    """Format region as a styled pill."""
    cls = {"Europe": "pill-eu", "Americas": "pill-am", "Asia": "pill-as"}.get(region, "pill-eu")
    rank_str = f' #{regional_rank}' if regional_rank > 0 else ''
    return f'<span class="{cls}">{region}{rank_str}</span>'


def rank_badge(rank: int) -> str:
    """Format rank as a colored badge."""
    cls = "top1" if rank == 1 else "top3" if rank <= 3 else "top10" if rank <= 10 else "rest"
    return f'<span class="rank-badge {cls}">{rank}</span>'


def change_arrow(delta: int) -> str:
    """Format rank change as an arrow."""
    if delta > 0:
        return f'<span class="change-up">▲ {delta}</span>'
    elif delta < 0:
        return f'<span class="change-down">▼ {abs(delta)}</span>'
    return '<span class="change-same">—</span>'


def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Add region and flag metadata to standings DataFrame."""
    df = df.copy()
    if "region" not in df.columns:
        df["region"] = "Global"
    df["flag"] = df["team"].map(
        lambda t: KNOWN_META.get(t, {}).get("flag",
                  df.loc[df["team"]==t, "flag"].iloc[0]
                  if "flag" in df.columns and (df["team"]==t).any() else "🌍")
    )
    return df


def add_regional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Add ranking within each region."""
    df = df.copy()
    df["regional_rank"] = (df.groupby("region")["total_points"]
                             .rank(ascending=False, method="first")
                             .astype(int))
    return df
