"""
Tests for the unfinished-event backfill behaviour.

Context
-------
Valve's ranking drops every match from any event whose ``finished`` flag is
false at snapshot generation time (see
https://github.com/ValveSoftware/counter-strike_regional_standings/blob/main/model/data_loader.js
``filterInProgressEvents``). In practice: if an event's end date falls on or
after the snapshot release, ALL its matches are excluded — even matches
played days earlier.

The fix:
  * ``fetch_tournament_page`` accepts a ``snapshot_cutoff`` and, when an
    event's end date is on/after that cutoff, ignores the per-match
    ``start_dt`` filter (fetches the entire event).
  * ``Store.append_liquipedia`` deduplicates against matches / prize rows
    already present (belt-and-braces in case Valve did include the event).

These tests exercise those two guarantees with no network I/O.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_loaders import liquipedia_loader  # noqa: E402
from vrs_engine.store import Store  # noqa: E402


# ── Fetch behaviour: unfinished event ignores start_dt filter ──────────────

class _StubSoup:
    """Minimal stand-in for BeautifulSoup used by fetch_tournament_page."""

    def __init__(self, matches):
        self._matches = matches

    def find(self, *_a, **_kw):
        return None

    def select_one(self, *_a, **_kw):
        return None

    def find_all(self, *_a, **_kw):
        return []

    def select(self, selector):
        # fetch_tournament_page only selects ".brkts-match" from the root soup
        if selector == ".brkts-match":
            return list(self._matches)
        return []


def _fake_match(date_str: str, winner: str, loser: str) -> dict:
    """A dict that _parse_match will return when given this stub element."""
    return {
        "date":         datetime.strptime(date_str, "%Y-%m-%d"),
        "winner":       winner,
        "loser":        loser,
        "event":        "Stub Event",
        "prize_pool":   0.0,
        "winner_prize": 0.0,
        "loser_prize":  0.0,
        "is_lan":       False,
    }


def test_unfinished_event_fetches_early_matches(monkeypatch):
    """
    Simulate PGL Bucharest (end 2026-04-11) at snapshot 2026-04-10:
    matches from Apr 4 and Apr 5 (before snapshot) should still be returned
    because the event is unfinished at the cutoff.
    """
    april04 = _fake_match("2026-04-04", "Vitality", "NAVI")
    april05 = _fake_match("2026-04-05", "Spirit",   "G2")

    raw_matches = [object(), object()]  # opaque stand-ins; _parse_match stubbed
    stub_soup = _StubSoup(raw_matches)

    # Stub the HTTP, HTML-parse, infobox, prize, and per-match parse helpers.
    monkeypatch.setattr(liquipedia_loader, "_fetch_page", lambda slug: "<html/>")
    monkeypatch.setattr(liquipedia_loader, "_bs4", lambda html, *a, **kw: stub_soup)
    monkeypatch.setattr(liquipedia_loader, "_parse_infobox", lambda soup: {
        "prize_pool": 300000.0, "is_lan": True, "event_name": "PGL Bucharest",
        "start_date": datetime(2026, 4, 4),
        "end_date":   datetime(2026, 4, 11),
    })
    monkeypatch.setattr(liquipedia_loader, "_parse_prize_pool", lambda soup: {})
    parsed = iter([april04, april05])
    monkeypatch.setattr(
        liquipedia_loader, "_parse_match",
        lambda el, *a, **kw: next(parsed),
    )

    # snapshot_cutoff = 2026-04-10 (event ends Apr 11 → unfinished)
    out = liquipedia_loader.fetch_tournament_page(
        "PGL/2026/Bucharest",
        start_dt=datetime(2026, 4, 10),           # would normally exclude Apr 4 & 5
        end_dt=datetime(2026, 4, 22, 23, 59, 59),
        snapshot_cutoff=datetime(2026, 4, 10),
    )

    dates = sorted(str(m["date"].date()) for m in out if m["loser"] != "")
    assert dates == ["2026-04-04", "2026-04-05"], (
        f"Expected early matches included for unfinished event, got {dates}"
    )


def test_finished_event_honours_start_dt_filter(monkeypatch):
    """
    Event ends BEFORE the snapshot cutoff → Valve would have included it, so
    we should NOT re-fetch its pre-cutoff matches (they already live in the
    Valve snapshot). The per-match start_dt filter must still apply.
    """
    pre_cutoff  = _fake_match("2026-03-20", "Vitality", "FaZe")
    post_cutoff = _fake_match("2026-03-29", "Spirit",   "MOUZ")

    stub_soup = _StubSoup([object(), object()])

    monkeypatch.setattr(liquipedia_loader, "_fetch_page", lambda slug: "<html/>")
    monkeypatch.setattr(liquipedia_loader, "_bs4", lambda html, *a, **kw: stub_soup)
    monkeypatch.setattr(liquipedia_loader, "_parse_infobox", lambda soup: {
        "prize_pool": 0.0, "is_lan": False, "event_name": "Old Event",
        "start_date": datetime(2026, 3, 15),
        "end_date":   datetime(2026, 3, 25),   # ends BEFORE snapshot
    })
    monkeypatch.setattr(liquipedia_loader, "_parse_prize_pool", lambda soup: {})
    parsed = iter([pre_cutoff, post_cutoff])
    monkeypatch.setattr(
        liquipedia_loader, "_parse_match",
        lambda el, *a, **kw: next(parsed),
    )

    out = liquipedia_loader.fetch_tournament_page(
        "Old/Event",
        start_dt=datetime(2026, 3, 27),          # filters out Mar 20
        end_dt=datetime(2026, 4, 22, 23, 59, 59),
        snapshot_cutoff=datetime(2026, 3, 27),   # event finished before cutoff
    )

    # Mar 20 match is before start_dt; Mar 29 match is after both start_dt and
    # end_date of the event — _parse_match still yields it, start_dt filter
    # keeps it. The Mar 20 one must be dropped.
    dates = [str(m["date"].date()) for m in out if m["loser"] != ""]
    assert "2026-03-20" not in dates
    assert "2026-03-29" in dates


# ── Dedup behaviour in Store.append_liquipedia ─────────────────────────────

def _valve_tmh_one_match():
    """Return tmh + bpm with a single Valve-snapshot match + prize."""
    tmh = {
        "Vitality": [{
            "match_id": 123, "date": datetime(2026, 4, 4),
            "opponent": "NAVI", "result": "W",
            "prize_pool": 300000.0, "is_lan": True,
        }],
        "NAVI": [{
            "match_id": 123, "date": datetime(2026, 4, 4),
            "opponent": "Vitality", "result": "L",
            "prize_pool": 300000.0, "is_lan": True,
        }],
    }
    bpm = {"Vitality": [{"event_date": "2026-04-11", "prize_won": 100000.0}]}
    return tmh, bpm


def test_append_liquipedia_deduplicates_matches():
    """
    If Liquipedia returns a match that Valve already has, it should be
    dropped — no double-counting.
    """
    tmh, bpm = _valve_tmh_one_match()
    store = Store.from_valve(tmh, bpm)
    assert len(store.matches_df) == 1

    # Liquipedia returns the same match (date, winner, loser) plus a new one.
    liq_df = pd.DataFrame([
        {
            "date": datetime(2026, 4, 4),
            "winner": "Vitality", "loser": "NAVI",
            "event": "Dup", "prize_pool": 300000.0,
            "winner_prize": 0.0, "loser_prize": 0.0, "is_lan": True,
        },
        {
            "date": datetime(2026, 4, 5),
            "winner": "Spirit", "loser": "G2",
            "event": "New", "prize_pool": 300000.0,
            "winner_prize": 0.0, "loser_prize": 0.0, "is_lan": True,
        },
    ])

    store.append_liquipedia(liq_df)

    assert len(store.matches_df) == 2, (
        f"Expected 1 Valve + 1 new Liquipedia match (dup dropped), "
        f"got {len(store.matches_df)}"
    )
    # Apr 4 Vitality vs NAVI appears exactly once
    april04 = store.matches_df[store.matches_df["date"] == datetime(2026, 4, 4)]
    assert len(april04) == 1


def test_append_liquipedia_deduplicates_prizes():
    """
    If Liquipedia emits a prize row for a team+date already present in the
    Valve bpm, it should be dropped.
    """
    tmh, bpm = _valve_tmh_one_match()
    store = Store.from_valve(tmh, bpm)
    assert len(store.prizes_df) == 1  # Vitality 2026-04-11 $100k

    liq_df = pd.DataFrame([
        {   # prize-only row (loser=="") for same (team, date) as Valve bpm
            "date": datetime(2026, 4, 11),
            "winner": "Vitality", "loser": "",
            "event": "PGL Bucharest", "prize_pool": 300000.0,
            "winner_prize": 100000.0, "loser_prize": 0.0, "is_lan": False,
        },
        {   # prize-only row for a brand-new team+date
            "date": datetime(2026, 4, 11),
            "winner": "Spirit", "loser": "",
            "event": "PGL Bucharest", "prize_pool": 300000.0,
            "winner_prize": 50000.0, "loser_prize": 0.0, "is_lan": False,
        },
    ])

    store.append_liquipedia(liq_df)

    # One existing + one new = 2; the dup must have been skipped.
    assert len(store.prizes_df) == 2
    teams = sorted(store.prizes_df["team"].tolist())
    assert teams == ["Spirit", "Vitality"]


# ── Matchlist (Swiss / group-stage) parser ─────────────────────────────────

def test_parse_matchlist_match_swiss_style():
    """
    Parse a Swiss-stage match element (``.brkts-matchlist-match``) — these use
    different classes than bracket matches and were previously being dropped,
    causing PGL Bucharest Round 1-3 matches (Apr 4-6) to never appear.
    """
    from bs4 import BeautifulSoup
    from data_loaders.liquipedia_loader import _parse_matchlist_match

    # Minimal version of the real Liquipedia matchlist-match HTML.
    html = '''
    <div class="brkts-matchlist-match">
      <div aria-label="FUT Esports"
           class="brkts-matchlist-cell brkts-matchlist-opponent brkts-matchlist-slot-winner">
        <span class="name">FUT</span>
      </div>
      <div aria-label="FUT Esports"
           class="brkts-matchlist-cell brkts-matchlist-score">
        <div class="brkts-matchlist-cell-content">2</div>
      </div>
      <div aria-label="IC Esports"
           class="brkts-matchlist-cell brkts-matchlist-score">
        <div class="brkts-matchlist-cell-content">1</div>
      </div>
      <div aria-label="IC Esports"
           class="brkts-matchlist-cell brkts-matchlist-opponent">
        <span class="name">IC</span>
      </div>
      <span class="timer-object">April 4, 2026 - 10:00 EEST</span>
    </div>
    '''
    el = BeautifulSoup(html, "html.parser").select_one(".brkts-matchlist-match")
    m = _parse_matchlist_match(el, "PGL Bucharest 2026", 300000.0, True)

    assert m is not None, "matchlist parser returned None on valid input"
    assert m["winner"] == "FUT"           # normalised from "FUT Esports"
    assert m["loser"]  == "IC Esports"    # no mapping → passes through
    assert m["date"]   == datetime(2026, 4, 4, 10, 0)
    assert m["is_lan"] is True
    assert m["prize_pool"] == 300000.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
