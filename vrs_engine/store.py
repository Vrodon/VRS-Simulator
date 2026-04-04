"""
VRS Engine — Data Store

Owns the two canonical raw tables that feed every calculation:

    matches_df  — one row per series result (winner, loser, event info)
    prizes_df   — one row per prize earned  (team, date, amount)

Responsibilities:
    - Convert Valve's tmh + bpm dicts into clean, stripped tables
      (discards all pre-computed values: age_w, ev_w, opp_bo, opp_on, h2h_adj,
       scaled_prize — these are recomputed fresh by the pipeline)
    - Append Liquipedia rows with non-colliding artificial match IDs
    - Nothing else: no math, no ordering, no factor logic
"""

import pandas as pd
from datetime import datetime

# ── Canonical column sets ─────────────────────────────────────────────────────
MATCH_COLS = ["match_id", "date", "winner", "loser", "prize_pool", "is_lan"]
PRIZE_COLS = ["team", "date", "amount"]

# Artificial match IDs for Liquipedia rows start here (well above Valve IDs)
_LIQ_ID_OFFSET = 10_000_000


class Store:
    """
    Holds the two canonical raw tables for a single VRS computation run.

    matches_df columns
    ------------------
    match_id   int       unique per series
    date       datetime
    winner     str
    loser      str
    prize_pool float     event total prize pool (USD); 0 if none
    is_lan     bool

    prizes_df columns
    -----------------
    team       str
    date       datetime
    amount     float     USD earned by this team at this event
    """

    def __init__(self) -> None:
        self.matches_df: pd.DataFrame = pd.DataFrame(columns=MATCH_COLS)
        self.prizes_df:  pd.DataFrame = pd.DataFrame(columns=PRIZE_COLS)

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_valve(cls, tmh: dict, bpm: dict) -> "Store":
        """
        Build a Store from Valve's GitHub data structures.

        tmh  (team_match_history)
            dict[team_name → list of match-entry dicts]
            Fields used:    match_id, date, opponent, result ("W"/"L"),
                            prize_pool, is_lan
            Fields dropped: age_w, ev_w, h2h_adj, opp_bo, opp_on
                            (all pre-computed by Valve for their own cutoff)

        bpm  (bo_prizes_map)
            dict[team_name → list of prize-entry dicts]
            Fields used:    event_date (str "YYYY-MM-DD"), prize_won
            Fields dropped: age_weight, scaled_prize
                            (pre-computed by Valve for their own cutoff)
        """
        store = cls()

        # ── Matches ───────────────────────────────────────────────────
        # Take winner-perspective entries only → each match appears once,
        # giving us canonical (winner, loser) without any duplication.
        seen_ids: set[int] = set()
        match_rows: list[dict] = []

        for team, entries in tmh.items():
            for e in entries:
                if e.get("result") != "W":
                    continue
                mid = int(e["match_id"])
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                match_rows.append({
                    "match_id":  mid,
                    "date":      e["date"],
                    "winner":    str(team),
                    "loser":     str(e["opponent"]),
                    "prize_pool": float(e.get("prize_pool", 0.0) or 0.0),
                    "is_lan":    bool(e.get("is_lan", False)),
                })

        if match_rows:
            store.matches_df = pd.DataFrame(match_rows, columns=MATCH_COLS)

        # ── Prizes ────────────────────────────────────────────────────
        prize_rows: list[dict] = []

        for team, prizes in bpm.items():
            for p in prizes:
                try:
                    dt = datetime.strptime(str(p["event_date"]).strip(), "%Y-%m-%d")
                except (KeyError, ValueError):
                    continue
                amount = float(p.get("prize_won", 0.0) or 0.0)
                if amount > 0:
                    prize_rows.append({
                        "team":   str(team),
                        "date":   dt,
                        "amount": amount,
                    })

        if prize_rows:
            store.prizes_df = pd.DataFrame(prize_rows, columns=PRIZE_COLS)

        return store

    # ── Liquipedia append ─────────────────────────────────────────────────────

    def append_liquipedia(self, liq_df: pd.DataFrame) -> None:
        """
        Append Liquipedia match results and prize entries to the store.

        liq_df is the raw output from fetch_liquipedia_matches().
        Expected columns: date, winner, loser, prize_pool, is_lan, winner_prize

        Row types
        ---------
        loser != ""   →  series result  → added to matches_df
                          if winner_prize > 0, also added to prizes_df
        loser == ""   →  prize-only row → added to prizes_df only

        Artificial match IDs are assigned starting at _LIQ_ID_OFFSET + current
        match count, guaranteeing no collision with Valve's integer match IDs.
        """
        if liq_df is None or liq_df.empty:
            return

        loser_col = liq_df["loser"].astype(str)
        series_df = liq_df[loser_col != ""].copy()
        prize_df  = liq_df[loser_col == ""].copy()

        id_start = _LIQ_ID_OFFSET + len(self.matches_df)

        new_matches: list[dict] = []
        new_prizes:  list[dict] = []

        # ── Series results ────────────────────────────────────────────
        for i, row in enumerate(series_df.itertuples(index=False)):
            date = row.date
            if not isinstance(date, datetime):
                date = datetime.combine(date, datetime.min.time())

            new_matches.append({
                "match_id":  id_start + i,
                "date":      date,
                "winner":    str(row.winner),
                "loser":     str(row.loser),
                "prize_pool": float(getattr(row, "prize_pool", 0.0) or 0.0),
                "is_lan":    bool(getattr(row, "is_lan", False)),
            })

            # Capture winner prize if available
            winner_prize = float(getattr(row, "winner_prize", 0.0) or 0.0)
            if winner_prize > 0:
                new_prizes.append({
                    "team":   str(row.winner),
                    "date":   date,
                    "amount": winner_prize,
                })

        # ── Prize-only rows ───────────────────────────────────────────
        for row in prize_df.itertuples(index=False):
            amount = float(getattr(row, "winner_prize", 0.0) or 0.0)
            if amount <= 0:
                continue
            date = row.date
            if not isinstance(date, datetime):
                date = datetime.combine(date, datetime.min.time())
            new_prizes.append({
                "team":   str(row.winner),
                "date":   date,
                "amount": amount,
            })

        # ── Append to canonical tables ────────────────────────────────
        if new_matches:
            self.matches_df = pd.concat(
                [self.matches_df,
                 pd.DataFrame(new_matches, columns=MATCH_COLS)],
                ignore_index=True,
            )

        if new_prizes:
            self.prizes_df = pd.concat(
                [self.prizes_df,
                 pd.DataFrame(new_prizes, columns=PRIZE_COLS)],
                ignore_index=True,
            )
