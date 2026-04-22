"""
VRS Data Viewer
===============
Standalone Streamlit app for manual verification of the combined
Valve + Liquipedia data tables used in "Updated to Today" mode.

Run with:  streamlit run data_viewer.py

Pages
-----
  Tab 1 — Match Table   : all matches (per-team perspective), filterable + downloadable
  Tab 2 — Winnings Table: all prize entries (per-team), filterable + downloadable

Design
------
Kept from Valve's repo (time-invariant):
    Match ID, Date, Opponent, W/L, Event Weight (ev_w), is_lan, Roster

Recomputed with cutoff = today:
    Age Weight, Bounty Collected, Opponent Network, H2H Adj., Scaled Winnings

Liquipedia entries use the same recomputed values; Event Weight is computed
fresh from prize_pool via event_stakes().
"""

import os
import pickle
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from data_loaders import (
    load_valve_github_data,
    discover_liquipedia_from_portal,
    fetch_liquipedia_matches,
    liquipedia_cache_exists,
    liquipedia_cache_mtime,
    load_liquipedia_from_cache,
)
from data_loaders.liquipedia_loader import LIQUIPEDIA_TO_VALVE
from vrs_engine import Store, run_vrs, age_weight, event_stakes

# ── Configuration ─────────────────────────────────────────────────────────────

VALVE_DATE = "2026_04_06"
VALVE_YEAR = "2026"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_valve_cache_force(date_str: str, year: str) -> dict | None:
    """
    Load a Valve GitHub cache pickle ignoring the TTL.

    Historical VRS snapshots are immutable — the data for a past date will
    never change on Valve's repo, so a stale cache is always valid.
    Used as a fallback when the live GitHub fetch fails.
    """
    path = os.path.join("cache", "github_vrs", f"vrs_{year}_{date_str}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        data["source"] = "github_cache"
        return data
    except Exception:
        return None


def _determine_cutoff(tmh: dict, bpm: dict) -> datetime:
    """
    Return the latest date mentioned anywhere in Valve's match and prize data.
    This is the true data cutoff — typically a day or two before the snapshot
    publication date.
    """
    latest = datetime(2000, 1, 1)

    for matches in tmh.values():
        for m in matches:
            d = m.get("date")
            if isinstance(d, datetime) and d > latest:
                latest = d

    for prizes in bpm.values():
        for p in prizes:
            try:
                dt = datetime.strptime(str(p["event_date"]).strip(), "%Y-%m-%d")
                if dt > latest:
                    latest = dt
            except (ValueError, KeyError):
                continue

    return latest


def _identify_active_rosters(standings: pd.DataFrame) -> dict:
    """
    For each team name in the snapshot standings, return the DataFrame index
    of the best-ranked (lowest rank number) row. Handles roster splits.
    """
    active_map = {}
    for team_name in standings["team"].unique():
        rows = standings[standings["team"] == team_name]
        active_map[team_name] = (
            rows.index[0] if len(rows) == 1 else rows["rank"].idxmin()
        )
    return active_map


def _as_date(d):
    """Coerce datetime / date to a plain date object for display."""
    if isinstance(d, datetime):
        return d.date()
    return d


# ── Data loading (cached for 1 h) ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data() -> dict:
    """
    Full pipeline:
        1. Load Valve GitHub snapshot
        2. Determine data cutoff (latest date in Valve data)
        3. Fetch Liquipedia data from cutoff+1 → today
        4. Filter by active rosters
        5. Build Store + run VRS pipeline (cutoff = today)
        6. Build Match DataFrame (per-team rows)
        7. Build Winnings DataFrame (per-team rows)

    Returns a dict with match_df, winnings_df, and metadata.
    On any failure returns {"error": <message>}.
    """
    today = datetime.now()

    # ── 1. Valve snapshot ─────────────────────────────────────────────────────
    valve_data = load_valve_github_data(VALVE_DATE, VALVE_YEAR)
    if valve_data.get("error"):
        # GitHub fetch failed — fall back to stale cache if available.
        # Historical snapshots are immutable, so stale == still correct.
        fallback = _load_valve_cache_force(VALVE_DATE, VALVE_YEAR)
        if fallback:
            valve_data = fallback
        else:
            return {"error": f"Valve load failed: {valve_data['error']}"}

    tmh               = valve_data["team_match_history"]   # dict[team → [match dicts]]
    bpm               = valve_data["bo_prizes_map"]        # dict[team → [prize dicts]]
    snapshot_standings = valve_data["standings"]

    if not tmh:
        return {"error": "Valve data loaded but team_match_history is empty."}

    # ── 2. Determine data cutoff ──────────────────────────────────────────────
    cutoff           = _determine_cutoff(tmh, bpm)
    start_date_str   = (cutoff + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date_str     = today.strftime("%Y-%m-%d")

    # Snapshot cutoff for unfinished-event backfill. Events whose end date is
    # on/after this datetime were in-progress in Valve's snapshot and must be
    # fetched whole (Valve drops them via the `finished` flag).
    # ``_determine_cutoff`` already returns a datetime — use it directly.
    snapshot_cutoff_dt = cutoff

    # Widen discovery backward so events in-progress at the snapshot (e.g. a
    # Swiss stage that started days before the snapshot) are rediscovered.
    discovery_start_str = (cutoff - timedelta(days=30)).strftime("%Y-%m-%d")

    # ── 3. Fetch Liquipedia data ──────────────────────────────────────────────
    try:
        discovered = discover_liquipedia_from_portal(
            discovery_start_str, end_date_str,
            min_tier="B-Tier", include_qualifiers=True,
        )
        if not discovered:
            return {
                "error": (
                    f"No Liquipedia tournaments found between "
                    f"{discovery_start_str} and {end_date_str}."
                )
            }

        slugs = [d["slug"] for d in discovered]

        # Serve from cache if fresh (< 2 hours old)
        if liquipedia_cache_exists(
            start_date_str, end_date_str, slugs,
            snapshot_cutoff=snapshot_cutoff_dt,
        ):
            mtime = liquipedia_cache_mtime(
                start_date_str, end_date_str, slugs,
                snapshot_cutoff=snapshot_cutoff_dt,
            )
            fresh = (datetime.now() - mtime).total_seconds() < 7200
        else:
            fresh = False

        if fresh:
            liq_df = load_liquipedia_from_cache(
                start_date_str, end_date_str, slugs,
                snapshot_cutoff=snapshot_cutoff_dt,
            )
        else:
            liq_df = fetch_liquipedia_matches(
                start_date_str, end_date_str,
                tournament_slugs=slugs,
                force_refresh=not fresh,
                snapshot_cutoff=snapshot_cutoff_dt,
            )

        if liq_df is None or liq_df.empty:
            return {"error": "Liquipedia fetch returned no data."}

        # Re-apply name normalisation on top of the cached data.
        # Caches built before a mapping entry was added still contain the
        # old Liquipedia display name — this pass corrects them without
        # requiring a full re-fetch.
        def _renorm(name: str) -> str:
            s = str(name).strip()
            return LIQUIPEDIA_TO_VALVE.get(s, s)

        liq_df["winner"] = liq_df["winner"].apply(_renorm)
        liq_df["loser"]  = liq_df["loser"].apply(_renorm)

    except Exception as exc:
        return {"error": f"Liquipedia fetch failed: {exc}"}

    # ── 4. Active roster filtering ────────────────────────────────────────────
    active_rosters    = _identify_active_rosters(snapshot_standings)
    active_team_names = {
        snapshot_standings.loc[active_rosters[t], "team"]
        for t in active_rosters
    }
    snapshot_teams = set(snapshot_standings["team"].values)

    def _is_ok(name: str) -> bool:
        return name not in snapshot_teams or name in active_team_names

    is_series = liq_df["loser"].astype(str) != ""
    series_df = liq_df[is_series].copy()
    prize_df  = liq_df[~is_series].copy()

    series_df = series_df[
        series_df["winner"].apply(_is_ok) & series_df["loser"].apply(_is_ok)
    ].copy()
    prize_df  = prize_df[prize_df["winner"].apply(_is_ok)].copy()

    liq_filtered = pd.concat([series_df, prize_df], ignore_index=True)
    liq_series   = series_df.reset_index(drop=True)   # series matches only

    # ── 5. Build Store + run pipeline ─────────────────────────────────────────
    store              = Store.from_valve(tmh, bpm)
    valve_match_count  = len(store.matches_df)         # needed for LIQ ID offset
    store.append_liquipedia(liq_filtered)

    engine      = run_vrs(store, cutoff=today)
    standings   = engine["standings"]
    match_h2h   = engine["match_h2h"]                 # dict[int(match_id) → {w_delta,l_delta}]

    bo_map = dict(zip(standings["team"], standings["bo_factor"]))
    on_map = dict(zip(standings["team"], standings["on_factor"]))

    # ── 6. Build Match DataFrame ──────────────────────────────────────────────
    rows: list[dict] = []

    # -- Valve matches (per-team perspective already in tmh) --
    for team, matches in tmh.items():
        for m in matches:
            dt      = m["date"]               # datetime
            opp     = m["opponent"]
            result  = m["result"]             # "W" or "L"
            mid     = int(m["match_id"])
            ev_w    = float(m.get("ev_w", 0.0) or 0.0)   # KEEP Valve's value
            is_lan  = bool(m.get("is_lan", False))

            aw       = age_weight(dt, today)
            h2h_info = match_h2h.get(mid, {})
            h2h_val  = (
                h2h_info.get("w_delta", 0.0)
                if result == "W"
                else h2h_info.get("l_delta", 0.0)
            )

            if is_lan and result == "W":
                lan_col = f"{aw:.3f}"
            elif is_lan:
                lan_col = "0"
            else:
                lan_col = "-"

            # Use Valve's own pre-computed values for BC and ON.
            # opp_bo = raw BO ratio (bo_sum/ref_5th) as stored in Valve's markdown.
            # opp_on = intermediate PageRank ON value as stored in Valve's markdown.
            # These cannot be reproduced by our pipeline (different dataset/cutoff),
            # so we take them straight from the source.
            rows.append({
                "_sort_key": (team, dt, mid),
                "Team":               team,
                "Match Played":       0,            # assigned after sort
                "Match ID":           mid,
                "Date":               _as_date(dt),
                "Opponent":           opp,
                "W/L":                result,
                "Age Weight":         round(aw,   4),
                "Event Weight":       round(ev_w, 4),
                "Bounty Collected":   round(m.get("opp_bo", 0.0), 4),
                "Opponent Network":   round(m.get("opp_on", 0.0), 4),
                "LAN Wins":           lan_col,
                "H2H Adj.":           round(h2h_val, 2),
                "Event":              "",
                "Source":             "Valve",
            })

    # -- Liquipedia matches (expand each series into winner + loser row) --
    # IDs mirror exactly what Store.append_liquipedia() assigns
    liq_id_start = 10_000_000 + valve_match_count

    for i, row in enumerate(liq_series.itertuples(index=False)):
        mid    = liq_id_start + i
        dt     = row.date
        if not isinstance(dt, datetime):
            dt = datetime.combine(dt, datetime.min.time())

        pp      = float(getattr(row, "prize_pool",  0.0) or 0.0)
        is_lan  = bool(getattr(row,  "is_lan",      False))
        event   = str(getattr(row,   "event",       ""))
        winner  = str(row.winner)
        loser   = str(row.loser)

        aw       = age_weight(dt, today)
        ew       = event_stakes(pp) if pp > 0 else 0.0
        h2h_info = match_h2h.get(mid, {})

        # Winner row
        rows.append({
            "_sort_key": (winner, dt, mid),
            "Team":               winner,
            "Match Played":       0,
            "Match ID":           mid,
            "Date":               _as_date(dt),
            "Opponent":           loser,
            "W/L":                "W",
            "Age Weight":         round(aw,  4),
            "Event Weight":       round(ew,  4),
            "Bounty Collected":   round(bo_map.get(loser,   0.0), 4),
            "Opponent Network":   round(on_map.get(loser,   0.0), 4),
            "LAN Wins":           f"{aw:.3f}" if is_lan else "-",
            "H2H Adj.":           round(h2h_info.get("w_delta", 0.0), 2),
            "Event":              event,
            "Source":             "Liquipedia",
        })

        # Loser row
        rows.append({
            "_sort_key": (loser, dt, mid),
            "Team":               loser,
            "Match Played":       0,
            "Match ID":           mid,
            "Date":               _as_date(dt),
            "Opponent":           winner,
            "W/L":                "L",
            "Age Weight":         round(aw,  4),
            "Event Weight":       round(ew,  4),
            "Bounty Collected":   round(bo_map.get(winner,  0.0), 4),
            "Opponent Network":   round(on_map.get(winner,  0.0), 4),
            "LAN Wins":           "0" if is_lan else "-",
            "H2H Adj.":           round(h2h_info.get("l_delta", 0.0), 2),
            "Event":              event,
            "Source":             "Liquipedia",
        })

    # Sort + assign "Match Played" sequential per-team counter
    match_df = pd.DataFrame(rows)
    match_df.sort_values("_sort_key", inplace=True)
    match_df["Match Played"] = match_df.groupby("Team").cumcount() + 1
    match_df.drop(columns=["_sort_key"], inplace=True)
    match_df.reset_index(drop=True, inplace=True)

    # Final column order
    match_df = match_df[[
        "Team", "Match Played", "Match ID", "Date", "Opponent",
        "W/L", "Age Weight", "Event Weight", "Bounty Collected",
        "Opponent Network", "LAN Wins", "H2H Adj.", "Event", "Source",
    ]]

    # ── 7. Build Winnings DataFrame ───────────────────────────────────────────
    prize_rows: list[dict] = []

    # -- Valve prizes (top-10 per team from bo_prizes_map) --
    for team, prizes in bpm.items():
        for p in prizes:
            try:
                dt = datetime.strptime(str(p["event_date"]).strip(), "%Y-%m-%d")
            except (ValueError, KeyError):
                continue
            amount = float(p.get("prize_won", 0.0) or 0.0)
            if amount <= 0:
                continue
            aw = age_weight(dt, today)
            prize_rows.append({
                "Team":            team,
                "Event Date":      _as_date(dt),
                "Age Weight":      round(aw, 4),
                "Prize Winnings":  amount,
                "Scaled Winnings": round(amount * aw, 2),
                "Source":          "Valve",
            })

    # -- Liquipedia prizes: sentinel rows (loser == "") --
    liq_prize_only = liq_filtered[liq_filtered["loser"].astype(str) == ""]
    for row in liq_prize_only.itertuples(index=False):
        amount = float(getattr(row, "winner_prize", 0.0) or 0.0)
        if amount <= 0:
            continue
        dt = row.date
        if not isinstance(dt, datetime):
            dt = datetime.combine(dt, datetime.min.time())
        aw = age_weight(dt, today)
        prize_rows.append({
            "Team":            str(row.winner),
            "Event Date":      _as_date(dt),
            "Age Weight":      round(aw, 4),
            "Prize Winnings":  amount,
            "Scaled Winnings": round(amount * aw, 2),
            "Source":          "Liquipedia",
        })

    # -- Liquipedia prizes: winner_prize embedded in series rows --
    for row in liq_series.itertuples(index=False):
        amount = float(getattr(row, "winner_prize", 0.0) or 0.0)
        if amount <= 0:
            continue
        dt = row.date
        if not isinstance(dt, datetime):
            dt = datetime.combine(dt, datetime.min.time())
        aw = age_weight(dt, today)
        prize_rows.append({
            "Team":            str(row.winner),
            "Event Date":      _as_date(dt),
            "Age Weight":      round(aw, 4),
            "Prize Winnings":  amount,
            "Scaled Winnings": round(amount * aw, 2),
            "Source":          "Liquipedia",
        })

    winnings_df = pd.DataFrame(prize_rows)
    # Deduplicate: same team + date + prize amount can appear in both sentinel
    # and series rows for the same event
    winnings_df.drop_duplicates(
        subset=["Team", "Event Date", "Prize Winnings"],
        inplace=True,
    )
    winnings_df.sort_values(["Team", "Event Date"], inplace=True)
    winnings_df.reset_index(drop=True, inplace=True)

    # ── Metadata ──────────────────────────────────────────────────────────────
    all_liq_teams = set(liq_series["winner"].tolist() + liq_series["loser"].tolist())
    known_teams   = set(standings["team"].tolist())
    unmatched     = sorted(all_liq_teams - known_teams)

    return {
        "match_df":          match_df,
        "winnings_df":       winnings_df,
        "cutoff":            cutoff,
        "start_date_str":    start_date_str,
        "end_date_str":      end_date_str,
        "today":             today,
        "n_valve_match_rows": sum(len(ms) for ms in tmh.values()),
        "n_liq_series":       len(liq_series),
        "n_tournaments":      len(discovered),
        "n_eligible_teams":   len(standings),
        "unmatched_teams":    unmatched,
        "error":              None,
    }


# ── UI ────────────────────────────────────────────────────────────────────────

def _apply_filters(
    df: pd.DataFrame,
    team_col: str,
    source_col: str,
    date_col: str,
    selected_teams,
    selected_sources,
    date_range,
) -> pd.DataFrame:
    """Apply sidebar filter selections to a DataFrame."""
    out = df.copy()
    if selected_teams:
        out = out[out[team_col].isin(selected_teams)]
    if selected_sources:
        out = out[out[source_col].isin(selected_sources)]
    if date_range and len(date_range) == 2:
        start, end = date_range
        out = out[(out[date_col] >= start) & (out[date_col] <= end)]
    return out


def main():
    st.set_page_config(
        page_title="VRS Data Viewer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("VRS Data Viewer — Updated to Today")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading Valve snapshot and Liquipedia data…"):
        data = load_all_data()

    if data.get("error"):
        st.error(f"**Data load failed:** {data['error']}")
        st.stop()

    match_df    = data["match_df"]
    winnings_df = data["winnings_df"]

    # ── Header metadata ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valve snapshot",    VALVE_DATE.replace("_", "-"))
    col2.metric("Data cutoff",       data["cutoff"].strftime("%Y-%m-%d"))
    col3.metric("Liquipedia window", f"{data['start_date_str']} → {data['end_date_str']}")
    col4.metric("Today",             data["today"].strftime("%Y-%m-%d"))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Valve match rows",       data["n_valve_match_rows"])
    col6.metric("Liquipedia series",      data["n_liq_series"])
    col7.metric("Tournaments fetched",    data["n_tournaments"])
    col8.metric("Eligible teams (pipeline)", data["n_eligible_teams"])

    if data["unmatched_teams"]:
        st.warning(
            f"**{len(data['unmatched_teams'])} Liquipedia team(s) not in pipeline standings** "
            f"(BC/ON/H2H will be 0 for these): "
            + ", ".join(data["unmatched_teams"])
        )

    st.caption(
        "⚠️ Valve period shows **top-10 prizes only** per team (Valve's markdown limitation). "
        "Liquipedia period shows all tracked prize placements."
    )

    st.divider()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        all_teams = sorted(set(match_df["Team"].unique()) | set(winnings_df["Team"].unique()))
        sel_teams = st.multiselect("Team", all_teams, placeholder="All teams")

        sel_sources = st.multiselect(
            "Source", ["Valve", "Liquipedia"], placeholder="Both sources"
        )

        min_date = match_df["Date"].min()
        max_date = match_df["Date"].max()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        st.divider()
        st.caption(
            "Filters apply to both tabs.\n\n"
            "Use the table's built-in column headers to sort.\n\n"
            "Select cells → Ctrl+C to copy to clipboard."
        )

        if st.button("🔄 Refresh data"):
            st.cache_data.clear()
            st.rerun()

    # Normalise date_range to a 2-tuple (handles single-date edge case)
    if not isinstance(date_range, (list, tuple)) or len(date_range) < 2:
        date_range = (min_date, max_date)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_matches, tab_winnings = st.tabs(["📋 Match Table", "💰 Winnings Table"])

    # ─── Tab 1: Match Table ───────────────────────────────────────────────────
    with tab_matches:
        filtered_matches = _apply_filters(
            match_df, "Team", "Source", "Date",
            sel_teams, sel_sources, date_range,
        )

        n_valve = (filtered_matches["Source"] == "Valve").sum()
        n_liq   = (filtered_matches["Source"] == "Liquipedia").sum()
        st.caption(
            f"**{len(filtered_matches)} rows** — "
            f"{n_valve} from Valve / {n_liq} from Liquipedia"
        )

        st.dataframe(
            filtered_matches,
            use_container_width=True,
            height=600,
            hide_index=True,
        )

        st.download_button(
            label="📥 Download matches CSV",
            data=filtered_matches.to_csv(index=False),
            file_name="vrs_matches.csv",
            mime="text/csv",
        )

    # ─── Tab 2: Winnings Table ────────────────────────────────────────────────
    with tab_winnings:
        # Filter winnings by team + source; date column is "Event Date"
        filtered_winnings = _apply_filters(
            winnings_df, "Team", "Source", "Event Date",
            sel_teams, sel_sources, date_range,
        )

        n_valve_w = (filtered_winnings["Source"] == "Valve").sum()
        n_liq_w   = (filtered_winnings["Source"] == "Liquipedia").sum()
        st.caption(
            f"**{len(filtered_winnings)} rows** — "
            f"{n_valve_w} from Valve (top-10 per team) / {n_liq_w} from Liquipedia"
        )

        st.dataframe(
            filtered_winnings,
            use_container_width=True,
            height=600,
            hide_index=True,
        )

        st.download_button(
            label="📥 Download winnings CSV",
            data=filtered_winnings.to_csv(index=False),
            file_name="vrs_winnings.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
