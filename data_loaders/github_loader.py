"""
GitHub Data Loader

Fetches live VRS data from Valve's public GitHub repository.
Includes disk caching to avoid repeated GitHub fetches.
"""

import requests
import json
import pickle
import re as _re
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

GITHUB_RAW = ("https://raw.githubusercontent.com/ValveSoftware/"
              "counter-strike_regional_standings/refs/heads/main")
GITHUB_API = ("https://api.github.com/repos/ValveSoftware/"
              "counter-strike_regional_standings/contents")
GH_FOLDER = "invitation"
GH_WORKERS = 20
GH_CACHE_DIR = "cache/github_vrs"
GH_CACHE_TTL_HOURS = 2  # Keep cache for 2 hours before re-fetching


# ─────────────────────────────────────────────────────────────────────────────
# Cache Management
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_cache_dir():
    """Ensure cache directory exists."""
    os.makedirs(GH_CACHE_DIR, exist_ok=True)


def _get_cache_path(date_str: str, year: str) -> str:
    """Get cache file path for a given date/year snapshot."""
    _ensure_cache_dir()
    return os.path.join(GH_CACHE_DIR, f"vrs_{year}_{date_str}.json")


def _load_from_cache(date_str: str, year: str) -> dict | None:
    """Load cached VRS data if it exists and is fresh."""
    cache_path = _get_cache_path(date_str, year).replace('.json', '.pkl')

    if not os.path.exists(cache_path):
        return None

    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age_hours = (datetime.now() - mtime).total_seconds() / 3600

        if age_hours > GH_CACHE_TTL_HOURS:
            return None  # Cache is stale

        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_to_cache(date_str: str, year: str, data: dict):
    """Save VRS data to cache using pickle (handles DataFrames)."""
    cache_path = _get_cache_path(date_str, year).replace('.json', '.pkl')
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass  # Silently fail cache write


def github_cache_exists(date_str: str, year: str) -> bool:
    """Check if fresh cache exists."""
    return _load_from_cache(date_str, year) is not None


def github_cache_clear(date_str: str | None = None, year: str | None = None):
    """Clear GitHub cache. If date/year provided, clear only that entry. Otherwise clear all."""
    if date_str and year:
        cache_path = _get_cache_path(date_str, year)
        try:
            os.remove(cache_path)
        except Exception:
            pass
    else:
        # Clear entire cache directory
        try:
            import shutil
            if os.path.exists(GH_CACHE_DIR):
                shutil.rmtree(GH_CACHE_DIR)
            _ensure_cache_dir()
        except Exception:
            pass


def _gh_get(url: str, timeout: int = 12) -> str | None:
    """Fetch a URL; return text on 200, else None."""
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "VRS-Simulator/1.0"})
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def _find_all_dates() -> list[tuple[str, str]]:
    """Discover all published VRS standing dates from GitHub.
    Returns list of (date_str, year) sorted newest-first."""
    result = []
    for year in ("2026", "2025", "2024"):
        text = _gh_get(f"{GITHUB_API}/{GH_FOLDER}/{year}")
        if not text:
            continue
        try:
            files = json.loads(text)
            for f in files:
                if not isinstance(f, dict):
                    continue
                if "standings_global" not in f.get("name", ""):
                    continue
                m = _re.search(r"(\d{4}_\d{2}_\d{2})", f.get("name", ""))
                if m:
                    result.append((m.group(1), year))
        except Exception:
            pass
    result.sort(key=lambda x: x[0], reverse=True)
    return result


def find_latest_date() -> tuple[str | None, str | None]:
    """Return the most recent (date_str, year) pair."""
    dates = _find_all_dates()
    return (dates[0][0], dates[0][1]) if dates else (None, None)


def _parse_standings_index(text: str) -> list[dict]:
    """Parse standings_global_*.md. Returns list of {rank, points, team, detail_path}."""
    rows = []
    for line in text.splitlines():
        m = _re.match(
            r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*"
            r"\|[^|]*\|\s*\[details\]\(([^)]+)\)",
            line,
        )
        if m:
            rows.append({
                "rank":        int(m.group(1)),
                "points":      int(m.group(2)),
                "team":        m.group(3).strip(),
                "detail_path": m.group(4).strip(),
            })
    return rows


def _parse_detail_md(team: str, rank: int, valve_pts: int, text: str) -> dict:
    """Parse one team detail markdown file."""
    d: dict = {
        "team": team, "rank": rank, "valve_points": valve_pts,
        "total_points": float(valve_pts), "seed": float(valve_pts),
        "h2h_delta": 0.0,
        "bo_factor": 0.0, "bc_factor": 0.0, "on_factor": 0.0, "lan_factor": 0.0,
        "seed_combined": 0.0, "bo_sum": 0.0, "bc_pre_curve": 0.0,
        "wins": 0, "losses": 0, "lan_wins": 0, "total_matches": 0,
        "matches": [],
    }

    m = _re.search(
        r"Final Rank Value[^(]*\(([-\d.]+)\)[^=]*=.*?"
        r"Starting Rank Value[^(]*\(([-\d.]+)\).*?"
        r"Head To Head Adjustments[^(]*\(([-\d.]+)\)",
        text, _re.DOTALL,
    )
    if m:
        d["total_points"] = float(m.group(1))
        d["seed"]         = float(m.group(2))
        d["h2h_delta"]    = float(m.group(3))

    for key, label in [
        ("bo_factor", "Bounty Offered"), ("bc_factor", "Bounty Collected"),
        ("on_factor", "Opponent Network"), ("lan_factor", "LAN Wins"),
    ]:
        m = _re.search(rf"- {label}:\s*([\d.]+)", text)
        if m:
            d[key] = float(m.group(1))

    m = _re.search(r"average of these factors is ([\d.]+)", text)
    if m:
        d["seed_combined"] = float(m.group(1))

    m = _re.search(r"sum of their top 10 scaled winnings \(\$([\d,]+\.\d+)\)", text)
    if m:
        d["bo_sum"] = float(m.group(1).replace(",", ""))

    bo_prizes: list[dict] = []
    in_bo_table = False
    for line in text.splitlines():
        if "Top ten winnings for this roster" in line:
            in_bo_table = True
            continue
        if in_bo_table:
            if not line.strip().startswith("|"):
                if bo_prizes:
                    break
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 4:
                continue
            if "---" in cells[0] or "Event Date" in cells[0]:
                continue
            try:
                ev_date = cells[0]
                age_w_p = float(cells[1])
                prize_s = cells[2].replace("$", "").replace(",", "")
                scaled_s = cells[3].replace("$", "").replace(",", "")
                bo_prizes.append({
                    "event_date":     ev_date,
                    "age_weight":     age_w_p,
                    "prize_won":      float(prize_s),
                    "scaled_prize":   float(scaled_s),
                })
            except (ValueError, IndexError):
                continue
    d["bo_prizes"] = bo_prizes

    bf = d["bc_factor"]
    if 0 < bf < 1.0:
        try:
            d["bc_pre_curve"] = round(10 ** (1.0 - 1.0 / bf), 4)
        except Exception:
            d["bc_pre_curve"] = bf
    else:
        d["bc_pre_curve"] = 1.0 if bf >= 1.0 else 0.0

    wins = losses = lan_wins = 0
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 11:
            continue
        if not _re.match(r"^\d+$", cells[0].lstrip("-").strip()):
            continue
        try:
            match_id = int(cells[1])
            date_s   = cells[2]
            opponent = cells[3]
            result   = cells[4]
            age_w_s  = cells[5]
            ev_w_s   = cells[6]
            lan_s    = cells[9]
            h2h_s    = cells[10]

            if not _re.match(r"\d{4}-\d{2}-\d{2}", date_s):
                continue

            age_w  = float(age_w_s)
            ev_w   = float(ev_w_s) if ev_w_s not in ("-", "") else 0.0
            is_lan = (lan_s not in ("-", "") and not lan_s.startswith("0"))
            h2h_a  = float(h2h_s) if h2h_s not in ("-", "") else 0.0

            bc_s = cells[7] if len(cells) > 7 else "-"
            opp_bo = 0.0
            if bc_s not in ("-", ""):
                _obo_m = _re.match(r"([\d.]+)", bc_s)
                if _obo_m:
                    opp_bo = float(_obo_m.group(1))

            on_s = cells[8] if len(cells) > 8 else "-"
            opp_on = 0.0
            if on_s not in ("-", ""):
                _oon_m = _re.match(r"([\d.]+)", on_s)
                if _oon_m:
                    opp_on = float(_oon_m.group(1))

            if 0 < ev_w <= 1.0:
                pool = round(10 ** (1.0 - 1.0 / ev_w) * 1_000_000)
            else:
                pool = 0

            dt = datetime.strptime(date_s, "%Y-%m-%d")

            if result == "W":
                wins += 1
                if is_lan:
                    lan_wins += 1
            else:
                losses += 1

            d["matches"].append({
                "match_id":  match_id,
                "date":      dt,
                "opponent":  opponent,
                "result":    result,
                "age_w":     age_w,
                "ev_w":      ev_w,
                "prize_pool": float(pool),
                "is_lan":    is_lan,
                "h2h_adj":   h2h_a,
                "opp_bo":    opp_bo,
                "opp_on":    opp_on,
            })
        except (ValueError, IndexError):
            continue

    d["wins"]          = wins
    d["losses"]        = losses
    d["lan_wins"]      = lan_wins
    d["total_matches"] = wins + losses
    return d


def load_valve_github_data(date_str: str | None = None,
                           year: str | None = None) -> dict:
    """
    Fetch a VRS snapshot from GitHub.

    Returns dict with:
        standings, matches, team_match_history, cutoff_date, cutoff_datetime,
        total_teams, source, error
    """
    result = {
        "standings":          pd.DataFrame(),
        "matches":            pd.DataFrame(),
        "team_match_history": {},
        "cutoff_date":        "2026_03_02",
        "cutoff_datetime":    datetime(2026, 3, 2),
        "total_teams":        0,
        "source":             "fallback",
        "error":              None,
    }

    if not date_str:
        date_str, year = find_latest_date()
    if not date_str:
        result["error"] = "GitHub unreachable — using fallback data"
        return result

    # Check cache first (before GitHub fetch)
    cached = _load_from_cache(date_str, year)
    if cached:
        cached["source"] = "github_cache"
        return cached

    idx_url = f"{GITHUB_RAW}/{GH_FOLDER}/{year}/standings_global_{date_str}.md"
    idx_text = _gh_get(idx_url)
    if not idx_text:
        result["error"] = f"Could not fetch standings index for {date_str}"
        return result

    teams = _parse_standings_index(idx_text)
    if not teams:
        result["error"] = "Standings index parsed 0 teams"
        return result

    region_map: dict[str, str] = {}
    for region_tag, region_label in [
        ("europe", "Europe"), ("americas", "Americas"),
        ("asia", "Asia"), ("asia-pacific", "Asia"),
        ("middle-east", "Middle East"),
    ]:
        r_url = f"{GITHUB_RAW}/{GH_FOLDER}/{year}/standings_{region_tag}_{date_str}.md"
        r_text = _gh_get(r_url)
        if r_text:
            for row in _parse_standings_index(r_text):
                region_map.setdefault(row["team"], region_label)

    detail_base = f"{GITHUB_RAW}/{GH_FOLDER}/{year}/"

    def _fetch_one(t: dict) -> tuple[dict, str | None]:
        return t, _gh_get(detail_base + t["detail_path"])

    parsed: list[dict] = []
    with ThreadPoolExecutor(max_workers=GH_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in teams}
        for fut in as_completed(futs):
            t, text = fut.result()
            if text:
                d = _parse_detail_md(t["team"], t["rank"], t["points"], text)
                d["region"] = region_map.get(t["team"], "Global")
                parsed.append(d)

    if not parsed:
        result["error"] = "Could not parse any detail files"
        return result

    rows = [{
        "team":          d["team"],
        "rank":          d["rank"],
        "total_points":  d["total_points"],
        "seed":          d["seed"],
        "h2h_delta":     d["h2h_delta"],
        "bo_factor":     d["bo_factor"],
        "bc_factor":     d["bc_factor"],
        "on_factor":     d["on_factor"],
        "lan_factor":    d["lan_factor"],
        "bo_sum":        d["bo_sum"],
        "bc_pre_curve":  d["bc_pre_curve"],
        "on_raw":        d["on_factor"],
        "lan_wins":      d["lan_wins"],
        "seed_combined": d["seed_combined"],
        "wins":          d["wins"],
        "losses":        d["losses"],
        "total_matches": d["total_matches"],
        "region":        d["region"],
        "flag":          "🌍",
        "color":         "#58a6ff",
    } for d in parsed]
    # For teams with multiple roster versions (roster splits), always use the
    # best-ranked (active) roster's match and prize data.  The simple dict
    # comprehension would otherwise keep whichever entry was fetched last,
    # which is non-deterministic because of ThreadPoolExecutor — and picking
    # the inactive roster's match history causes massive wrong point drops in
    # the "Updated to Today" simulation.
    _team_best: dict[str, tuple] = {}  # team_name → (rank, matches, bo_prizes)
    for d in parsed:
        name, rank = d["team"], d["rank"]
        if name not in _team_best or rank < _team_best[name][0]:
            _team_best[name] = (rank, d.get("matches", []), d.get("bo_prizes", []))

    _bo_prizes_map: dict[str, list] = {
        name: v[2] for name, v in _team_best.items()
    }

    standings_df = (pd.DataFrame(rows)
                    .sort_values("rank")
                    .reset_index(drop=True))

    seen: set[int] = set()
    match_rows = []
    for d in parsed:
        for m in d.get("matches", []):
            mid = m["match_id"]
            if m["result"] == "W" and mid not in seen:
                seen.add(mid)
                match_rows.append({
                    "date":         m["date"],
                    "winner":       d["team"],
                    "loser":        m["opponent"],
                    "prize_pool":   m["prize_pool"],
                    "winner_prize": 0.0,
                    "loser_prize":  0.0,
                    "is_lan":       m["is_lan"],
                    "event":        "",
                })

    matches_df = pd.DataFrame(match_rows) if match_rows else pd.DataFrame()

    try:
        cutoff_dt_p = datetime.strptime(date_str, "%Y_%m_%d")
    except Exception:
        cutoff_dt_p = datetime(2026, 3, 2)

    team_match_history: dict[str, list[dict]] = {
        name: v[1] for name, v in _team_best.items()
    }

    result.update({
        "standings":          standings_df,
        "matches":            matches_df,
        "team_match_history": team_match_history,
        "bo_prizes_map":      _bo_prizes_map,
        "cutoff_date":        date_str,
        "cutoff_datetime":    cutoff_dt_p,
        "total_teams":        len(standings_df),
        "source":             "github",
        "error":              None,
    })

    # Cache the result for next time
    _save_to_cache(date_str, year, result)

    return result
