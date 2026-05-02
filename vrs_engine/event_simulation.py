"""
Phase 5 §5.1 — pure engine-emit helper.

``emit_simulation_rows`` walks already-realised stages (output of
``stage_graph.resolve_for_render``) and produces engine-ready
``(extra_matches, extra_prizes)`` row lists. Both the Streamlit
fragment-level call site and the page-level engine emit path share
this single body — guaranteeing UI predictions and the recomputed
standings stay in sync.

Pure function: no Streamlit, no HTTP, no caching. Same input → same
output. Safe to call from any thread.
"""

from __future__ import annotations

from datetime import datetime
import re

from data_loaders.bracket_parser import Stage
from .placement_labels import compute_absolute_placements


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{('th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th')[n % 10]}"


def _format_place(lo: int, hi: int) -> str:
    if lo == hi:
        return _ordinal(lo)
    return f"{_ordinal(lo)}-{_ordinal(hi)}"


def _lookup_prize_for_place(place_label: str, prize_distribution: dict) -> float:
    """Exact label match, then containing-range fallback."""
    if place_label in prize_distribution:
        return float(prize_distribution[place_label])

    m = re.match(r"(\d+)(?:st|nd|rd|th)(?:-(\d+)(?:st|nd|rd|th))?", place_label)
    if not m:
        return 0.0
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    for label, amount in prize_distribution.items():
        m2 = re.match(r"(\d+)(?:st|nd|rd|th)(?:-(\d+)(?:st|nd|rd|th))?", label)
        if not m2:
            continue
        plo = int(m2.group(1))
        phi = int(m2.group(2)) if m2.group(2) else plo
        if plo <= lo and phi >= hi:
            return float(amount)
    return 0.0


def emit_simulation_rows(
    stages: list[Stage],
    picks: dict[str, str],
    *,
    event_name: str,
    prize_pool: float = 0.0,
    is_lan: bool = False,
    event_end: datetime | None = None,
    prize_distribution: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Walk realised stages → ``(extra_matches, extra_prizes)``.

    Caller is responsible for running ``resolve_for_render`` first; this
    helper trusts each match's ``seed_a/seed_b`` (Swiss synth already
    baked) and walks feeders for downstream rounds via ``picks``.

    Played slots (``played_winner`` set) are skipped — those live in
    ``matches_df`` already. Match rows reflect only user picks.
    """
    extra_matches: list[dict] = []
    end_dt = event_end or datetime.now()
    pp = float(prize_pool or 0.0)

    for stage in stages:
        if stage.format not in ("SE", "SE_with_bronze", "DE", "Swiss", "Groups"):
            continue

        by_id = {m.match_id: m for m in stage.matches}

        def _pair(m, _picks=picks, _by_id=by_id):
            """(a, b) for a match — direct seeds for R1 / Swiss / GR R1, else
            feeder walk through picks."""
            if m.round_idx == 1 and not m.is_bronze and m.sub in ("", "UB"):
                return m.seed_a, m.seed_b
            if m.sub == "SW":
                return m.seed_a, m.seed_b
            if m.sub == "GR" and m.round_idx == 1:
                return m.seed_a, m.seed_b

            def winner_of(mid):
                if not mid:
                    return None
                up = _by_id.get(mid)
                if not up:
                    return None
                return up.played_winner or _picks.get(mid)

            def loser_of(mid):
                if not mid:
                    return None
                up = _by_id.get(mid)
                if not up:
                    return None
                up_a, up_b = _pair(up)
                w = up.played_winner or _picks.get(mid)
                if w == up_a:
                    return up_b
                if w == up_b:
                    return up_a
                return None

            def team_for(feeder_id, kind):
                if not feeder_id:
                    return None
                return loser_of(feeder_id) if kind == "loser" else winner_of(feeder_id)

            return (team_for(m.feeder_a, m.feeder_a_kind),
                    team_for(m.feeder_b, m.feeder_b_kind))

        # Match rows for picked-but-unplayed slots.
        for m in stage.matches:
            if m.played_winner:
                continue
            picked = picks.get(m.match_id)
            if not picked:
                continue
            a, b = _pair(m)
            loser = b if picked == a else a if picked == b else None
            if not loser:
                continue
            extra_matches.append({
                "date":         m.played_date or end_dt,
                "winner":       picked,
                "loser":        loser,
                "event":        event_name,
                "prize_pool":   pp,
                "winner_prize": 0.0,
                "loser_prize":  0.0,
                "is_lan":       is_lan,
            })

    # Phase 5 §5.3: prize emission per-format via shared placement walkers.
    # ``compute_absolute_placements`` dispatches to the right format walker
    # per stage, aggregates fan-in siblings, applies ``compute_place_offsets``
    # for chain composition, and tracks each team's exit stage so advancers
    # take their downstream prize (Q6 strict partial-fill).
    final_placements = compute_absolute_placements(stages, picks)

    extra_prizes: list[dict] = []
    pd_dist = prize_distribution or {}
    for team, place in final_placements.items():
        amount = _lookup_prize_for_place(place, pd_dist)
        if amount <= 0:
            continue
        extra_prizes.append({
            "team":  team,
            "prize": amount,
            "event": event_name,
            "date":  end_dt,
        })

    return extra_matches, extra_prizes
