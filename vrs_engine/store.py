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

        # Dedup guard: build a set of (date-of-day, winner, loser) keys already
        # present from Valve's snapshot so we don't double-count a match that
        # somehow appears in both sources. In normal operation Valve drops
        # unfinished events whole, so overlap is zero — but the guard is cheap
        # and protects against edge cases (e.g. an event Valve did include but
        # that we re-fetch from Liquipedia during backfill).
        existing_keys: set[tuple] = set()
        if not self.matches_df.empty:
            for m in self.matches_df.itertuples(index=False):
                m_date = m.date
                if isinstance(m_date, datetime):
                    day = m_date.date()
                else:
                    day = pd.Timestamp(m_date).date()
                existing_keys.add((day, str(m.winner), str(m.loser)))

        new_matches: list[dict] = []
        new_prizes:  list[dict] = []
        skipped_dupes = 0

        # ── Series results ────────────────────────────────────────────
        for i, row in enumerate(series_df.itertuples(index=False)):
            date = row.date
            if not isinstance(date, datetime):
                date = datetime.combine(date, datetime.min.time())

            winner = str(row.winner)
            loser  = str(row.loser)
            key = (date.date(), winner, loser)
            if key in existing_keys:
                skipped_dupes += 1
                continue
            existing_keys.add(key)

            new_matches.append({
                "match_id":  id_start + len(new_matches),
                "date":      date,
                "winner":    winner,
                "loser":     loser,
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

        # Dedup guard for prizes: avoid double-crediting an event whose prize
        # rows are already in the store (Valve bpm produces one row per team
        # per event, dated at the event end). Key is (team, date-of-day).
        existing_prize_keys: set[tuple] = set()
        if not self.prizes_df.empty:
            for p in self.prizes_df.itertuples(index=False):
                p_date = p.date
                if isinstance(p_date, datetime):
                    day = p_date.date()
                else:
                    day = pd.Timestamp(p_date).date()
                existing_prize_keys.add((str(p.team), day))
        # Also exclude prize rows we just added from the match loop above.
        for np_row in new_prizes:
            existing_prize_keys.add((np_row["team"], np_row["date"].date()))

        skipped_prize_dupes = 0

        # ── Prize-only rows ───────────────────────────────────────────
        for row in prize_df.itertuples(index=False):
            amount = float(getattr(row, "winner_prize", 0.0) or 0.0)
            if amount <= 0:
                continue
            date = row.date
            if not isinstance(date, datetime):
                date = datetime.combine(date, datetime.min.time())
            team = str(row.winner)
            prize_key = (team, date.date())
            if prize_key in existing_prize_keys:
                skipped_prize_dupes += 1
                continue
            existing_prize_keys.add(prize_key)
            new_prizes.append({
                "team":   team,
                "date":   date,
                "amount": amount,
            })

        if skipped_dupes or skipped_prize_dupes:
            import logging
            logging.getLogger(__name__).info(
                "append_liquipedia: skipped %d duplicate match(es) and %d "
                "duplicate prize row(s) already in store",
                skipped_dupes, skipped_prize_dupes,
            )

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

    # ── Simulation append ─────────────────────────────────────────────────────

    def append_simulation(
        self,
        extra_matches: list[dict] | pd.DataFrame | None = None,
        extra_prizes:  list[dict] | None               = None,
    ) -> None:
        """
        Append hypothetical sim rows to the store, so the full engine can be
        re-run against a what-if scenario.

        extra_matches  list of dicts or DataFrame; each entry has:
                         date, winner, loser, prize_pool, is_lan
                       (optional: winner_prize — treated as a BO prize if > 0)

        extra_prizes   list of dicts; each entry has:
                         team, date, prize
                       (keys "amount" or "prize_won" are also accepted)

        Artificial match IDs pick up where append_liquipedia left off, so
        simulation rows never collide with Valve or Liquipedia match IDs.
        """
        # ── Matches ───────────────────────────────────────────────────
        if extra_matches is not None:
            if isinstance(extra_matches, pd.DataFrame):
                match_iter = extra_matches.to_dict(orient="records")
            else:
                match_iter = list(extra_matches)

            id_start = _LIQ_ID_OFFSET + len(self.matches_df)
            new_matches: list[dict] = []
            new_prizes_from_matches: list[dict] = []

            for i, m in enumerate(match_iter):
                loser = str(m.get("loser", "") or "")
                if not loser:
                    continue  # prize-only rows handled via extra_prizes
                date = m["date"]
                if not isinstance(date, datetime):
                    date = datetime.combine(date, datetime.min.time())

                new_matches.append({
                    "match_id":  id_start + i,
                    "date":      date,
                    "winner":    str(m["winner"]),
                    "loser":     loser,
                    "prize_pool": float(m.get("prize_pool", 0.0) or 0.0),
                    "is_lan":    bool(m.get("is_lan", False)),
                })

                winner_prize = float(m.get("winner_prize", 0.0) or 0.0)
                if winner_prize > 0:
                    new_prizes_from_matches.append({
                        "team":   str(m["winner"]),
                        "date":   date,
                        "amount": winner_prize,
                    })

            if new_matches:
                self.matches_df = pd.concat(
                    [self.matches_df,
                     pd.DataFrame(new_matches, columns=MATCH_COLS)],
                    ignore_index=True,
                )
            if new_prizes_from_matches:
                self.prizes_df = pd.concat(
                    [self.prizes_df,
                     pd.DataFrame(new_prizes_from_matches, columns=PRIZE_COLS)],
                    ignore_index=True,
                )

        # ── Prizes ────────────────────────────────────────────────────
        if extra_prizes:
            new_prizes: list[dict] = []
            for p in extra_prizes:
                amount = float(
                    p.get("amount", p.get("prize", p.get("prize_won", 0.0))) or 0.0
                )
                if amount <= 0:
                    continue
                date = p["date"]
                if not isinstance(date, datetime):
                    date = datetime.combine(date, datetime.min.time())
                new_prizes.append({
                    "team":   str(p["team"]),
                    "date":   date,
                    "amount": amount,
                })

            if new_prizes:
                self.prizes_df = pd.concat(
                    [self.prizes_df,
                     pd.DataFrame(new_prizes, columns=PRIZE_COLS)],
                    ignore_index=True,
                )
