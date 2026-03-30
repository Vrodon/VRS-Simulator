"""
HLTV Match Loader
=================
Fetches live CS2 series results from HLTV.org for a given date range
and converts them into the schema expected by compute_vrs() / compute_standings().

Pipeline
--------
  1. Paginate /stats/matches?startDate=&endDate=  (50 map-rows / page)
  2. Deduplicate map-level rows  →  series-level wins
  3. Fetch one event page per unique event_id  →  prize_pool, is_lan
  4. Return a DataFrame ready for injection into the VRS engine
  5. Persist result as a JSON file in cache/  (keyed by date range)

HTTP approach
-------------
  Uses requests + a timezone cookie trick (SocksPls method) that bypasses
  HLTV's basic Cloudflare protection from normal residential IPs.
  Detects and raises on Cloudflare challenge pages so the UI can surface
  a clear error instead of silently returning empty data.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Callable

import pandas as pd
import requests

# BeautifulSoup is imported lazily inside functions so the rest of the app
# can start even if beautifulsoup4 is not yet installed.  A clear error is
# raised at fetch-time rather than at import-time.
def _bs4(html: str, parser: str = "html.parser"):
    """Lazy BeautifulSoup constructor — imports bs4 on first call."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "beautifulsoup4 is required for the HLTV loader. "
            "Install it with:  pip install beautifulsoup4"
        ) from exc
    return BeautifulSoup(html, parser)

logger = logging.getLogger(__name__)

# ── HTTP constants ─────────────────────────────────────────────────────────────
_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer":         "https://www.hltv.org/stats",
}
_COOKIES:    dict[str, str] = {"hltvTimeZone": "Europe/Copenhagen"}
_PAGE_SIZE:  int            = 50
_REQ_DELAY:  float          = 2.0   # seconds between requests — be polite to HLTV
_TIMEOUT:    int            = 15
_MAX_PAGES:  int            = 25    # safety cap: 25 × 50 = 1 250 map rows max

HLTV_BASE:     str = "https://www.hltv.org"
STATS_MATCHES: str = f"{HLTV_BASE}/stats/matches"
CACHE_DIR:     str = "cache"


# ── Team-name normalisation ────────────────────────────────────────────────────
# Maps HLTV display names  →  Valve / KNOWN_META keys.
# Add entries here whenever a mismatch is discovered.
HLTV_TO_VALVE: dict[str, str] = {
    # Europe
    "Team Vitality":    "Vitality",
    "Team Spirit":      "Spirit",
    "FaZe Clan":        "FaZe",
    "G2 Esports":       "G2",
    "Heroic":           "HEROIC",
    "GamerLegion":      "GamerLegion",
    "3DMAX":            "3DMAX",
    "Falcons":          "Falcons",
    "Aurora":           "Aurora",
    "Astralis":         "Astralis",
    "PARIVISION":       "PARIVISION",
    "FUT Esports":      "FUT",
    "BetBoom Team":     "BetBoom",
    "Team BetBoom":     "BetBoom",
    "Monte":            "Monte",
    "Gentle Mates":     "Gentle Mates",
    "B8":               "B8",
    "Virtus.pro":       "Virtus.pro",
    "MOUZ":             "MOUZ",
    # CIS / NAVI
    "Natus Vincere":    "Natus Vincere",
    "NAVI":             "Natus Vincere",
    # Americas
    "Team Liquid":      "Liquid",
    "FURIA Esports":    "FURIA",
    "FURIA":            "FURIA",
    "paiN Gaming":      "paiN",
    "MIBR":             "MIBR",
    "NRG":              "NRG",
    "9z Team":          "9z",
    "9 to 5":           "9z",
    # Asia / Pacific
    "The MongolZ":      "The MongolZ",
    "TYLOO":            "TYLOO",
    "Rare Atom":        "Rare Atom",
    "BC.Game Esports":  "BC.Game",
    "BC.Game":          "BC.Game",
}


def _norm(name: str) -> str:
    """Return the canonical (Valve) team name for a given HLTV display name."""
    clean = name.strip()
    return HLTV_TO_VALVE.get(clean, clean)


# ── HTTP helpers ───────────────────────────────────────────────────────────────
def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    s.cookies.update(_COOKIES)
    return s


def _get(session: requests.Session, url: str) -> str | None:
    """
    Fetch *url* and return its HTML.
    Returns None on HTTP error or Cloudflare challenge.
    Raises RuntimeError when a Cloudflare block is detected on the very
    first request so the UI can surface a meaningful error message.
    """
    try:
        r = session.get(url, timeout=_TIMEOUT)
    except Exception as exc:
        logger.warning("Request failed (%s): %s", url, exc)
        return None

    if r.status_code != 200:
        logger.warning("HTTP %s for %s", r.status_code, url)
        return None

    # Detect Cloudflare JS challenge / bot block
    cf_signals = ("Just a moment", "cf-browser-verification", "cf_chl_opt", "Ray ID")
    if any(sig in r.text for sig in cf_signals):
        raise RuntimeError(
            "HLTV.org is blocking the request (Cloudflare challenge). "
            "Try again in a few minutes, or from a different network."
        )

    return r.text


# ── Match-page parsing ─────────────────────────────────────────────────────────
def _parse_matches_page(html: str) -> list[dict]:
    """
    Parse one page of /stats/matches.
    Returns a list of raw **map-level** dicts:
        team1, team1_id, team2, team2_id,
        team1_rounds, team2_rounds,
        date_str, event_id, event_slug, event_name,
        map_stats_id
    """
    soup = _bs4(html, "html.parser")
    rows: list[dict] = []

    tbody = soup.find("tbody")
    if not tbody:
        return rows

    for tr in tbody.find_all("tr"):
        try:
            row = _parse_row(tr)
            if row:
                rows.append(row)
        except Exception as exc:
            logger.debug("Row parse error: %s", exc)

    return rows


def _parse_row(tr) -> dict | None:
    """Extract all relevant fields from a single stats-table <tr>."""

    # ── Date ──────────────────────────────────────────────────────────
    date_cell = tr.find("td", class_=lambda c: c and "date" in c)
    if not date_cell:
        return None
    date_str = date_cell.get_text(strip=True)
    if not date_str:
        return None

    # ── Teams (two td.team-col cells) ─────────────────────────────────
    def _team_from_cell(cell) -> tuple[str, int]:
        a = cell.find("a", href=True)
        if not a:
            return "", 0
        m = re.search(r"/team/(\d+)/", a["href"])
        tid = int(m.group(1)) if m else 0
        return _norm(a.get_text(strip=True)), tid

    team_cells = tr.find_all("td", class_=lambda c: c and "team" in c)
    if len(team_cells) < 2:
        return None
    team1, t1_id = _team_from_cell(team_cells[0])
    team2, t2_id = _team_from_cell(team_cells[1])
    if not team1 or not team2 or team1 == team2:
        return None

    # ── Map score (rounds per team) ────────────────────────────────────
    t1_rounds = t2_rounds = 0
    score_cell = tr.find("td", class_=lambda c: c and "score" in c)
    if score_cell:
        m = re.search(r"(\d+)\s*[:\-]\s*(\d+)", score_cell.get_text(strip=True))
        if m:
            t1_rounds, t2_rounds = int(m.group(1)), int(m.group(2))

    # ── Event (id + slug extracted from href) ─────────────────────────
    event_id, event_slug, event_name = 0, "", ""
    event_cell = tr.find("td", class_=lambda c: c and "event" in c)
    if event_cell:
        a = event_cell.find("a", href=True)
        if a:
            m = re.search(r"/events/(\d+)/([^/?#]+)", a["href"])
            if m:
                event_id   = int(m.group(1))
                event_slug = m.group(2)
            event_name = a.get_text(strip=True)

    # ── Map-stats ID (used as deduplication fallback) ──────────────────
    map_stats_id = 0
    for a in tr.find_all("a", href=True):
        m = re.search(r"/stats/matches/mapstatsid/(\d+)/", a["href"])
        if m:
            map_stats_id = int(m.group(1))
            break

    return {
        "team1":        team1,
        "team1_id":     t1_id,
        "team2":        team2,
        "team2_id":     t2_id,
        "team1_rounds": t1_rounds,
        "team2_rounds": t2_rounds,
        "date_str":     date_str,
        "event_id":     event_id,
        "event_slug":   event_slug,
        "event_name":   event_name,
        "map_stats_id": map_stats_id,
    }


def _total_results(html: str) -> int:
    """Try to read the total result count from the pagination element."""
    soup = _bs4(html, "html.parser")
    for txt in soup.find_all(string=re.compile(r"of\s+\d+")):
        m = re.search(r"of\s+(\d+)", txt)
        if m:
            return int(m.group(1))
    return 0


# ── Date parsing ───────────────────────────────────────────────────────────────
_DATE_FMTS = ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%b %d %Y", "%d %b %Y",
              "%B %d %Y", "%d %B %Y")


def _parse_date(raw: str) -> datetime | None:
    clean = raw.strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            pass
    # Fallback: numeric regex DD/MM/YYYY or MM/DD/YYYY
    m = re.search(r"(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})", clean)
    if m:
        a, b, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(yr, b, a)   # assume DD/MM/YYYY
        except ValueError:
            pass
    return None


# ── Series deduplication ───────────────────────────────────────────────────────
def _to_series(map_rows: list[dict]) -> list[dict]:
    """
    Collapse map-level rows into series-level (BO1 / BO3 / BO5) results.

    Grouping key: (sorted team names, date_str, event_id).
    This holds because CS2 teams don't play the *same opponent* twice on
    the same calendar day at the same event.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in map_rows:
        key = (
            tuple(sorted([row["team1"], row["team2"]])),
            row["date_str"],
            row["event_id"],
        )
        groups[key].append(row)

    series: list[dict] = []
    for (sorted_teams, date_str, event_id), maps in groups.items():
        ref = maps[0]
        t1  = ref["team1"]
        t2  = ref["team2"]

        # Count maps won by each team across all maps in this series
        maps_t1 = sum(
            1 for mp in maps
            if (mp["team1"] == t1 and mp["team1_rounds"] > mp["team2_rounds"])
            or (mp["team2"] == t1 and mp["team2_rounds"] > mp["team1_rounds"])
        )
        maps_t2 = sum(
            1 for mp in maps
            if (mp["team1"] == t2 and mp["team1_rounds"] > mp["team2_rounds"])
            or (mp["team2"] == t2 and mp["team2_rounds"] > mp["team1_rounds"])
        )

        if maps_t1 > maps_t2:
            winner, loser = t1, t2
        elif maps_t2 > maps_t1:
            winner, loser = t2, t1
        else:
            # True tie — no series winner determinable, skip
            logger.debug("Tie in series %s vs %s on %s — skipped", t1, t2, date_str)
            continue

        dt = _parse_date(date_str)
        if dt is None:
            logger.debug("Could not parse date '%s' — skipped", date_str)
            continue

        series.append({
            "winner":       winner,
            "loser":        loser,
            "date":         dt,
            "date_str":     date_str,
            "event_id":     event_id,
            "event_slug":   ref["event_slug"],
            "event_name":   ref["event_name"],
            "maps_played":  len(maps),
        })

    return series


# ── Event-page scraping ────────────────────────────────────────────────────────
def _fetch_event_meta(
    session: requests.Session,
    event_id: int,
    slug: str,
) -> dict:
    """
    Scrape /events/<event_id>/<slug> to extract:
      prize_pool  – total event prize pool in USD (float; 0.0 if unknown)
      is_lan      – True when a physical venue is shown (bool)
      event_name  – cleaned event name from the <h1>

    Always returns a safe dict even on network/parse failures.
    """
    default = {"prize_pool": 0.0, "is_lan": False, "event_name": slug}
    if not event_id:
        return default

    url  = f"{HLTV_BASE}/events/{event_id}/{slug}"
    html = _get(session, url)
    if not html:
        return default

    soup   = _bs4(html, "html.parser")
    result = dict(default)

    # ── Prize pool ─────────────────────────────────────────────────────
    # HLTV events list prize pool as a labelled value pair.
    # Look for "$NNN,NNN" or "€NNN,NNN" near a "Prize" label.
    prize_pool = 0.0

    # Strategy A: scan text nodes near any element whose text contains "prize"
    for el in soup.find_all(string=re.compile(r"prize", re.I)):
        container = el.parent.parent if el.parent else None
        if container is None:
            continue
        raw_text = container.get_text(" ", strip=True)
        m = re.search(r"[\$€£]\s*([\d,]+)", raw_text)
        if m:
            try:
                prize_pool = float(m.group(1).replace(",", ""))
                break
            except ValueError:
                pass

    # Strategy B: find any large dollar-amount on the entire page
    if not prize_pool:
        all_amounts = re.findall(r"[\$€£]\s*([\d,]+)", soup.get_text(" "))
        for raw in all_amounts:
            try:
                val = float(raw.replace(",", ""))
                if val >= 10_000:
                    prize_pool = val
                    break
            except ValueError:
                continue

    result["prize_pool"] = prize_pool

    # ── LAN vs Online ──────────────────────────────────────────────────
    # Online events have "Online" or "TBD" in their location field.
    # LAN events show a city / venue / country.
    is_lan = False

    # Strategy A: look for the "Location" labelled value
    loc_label = soup.find(string=re.compile(r"^location$", re.I))
    if loc_label:
        loc_container = loc_label.parent
        if loc_container:
            loc_text = loc_container.get_text(" ", strip=True).lower()
            # If the label and value are siblings, check parent too
            if loc_text in ("location", ""):
                sib = loc_container.find_next_sibling()
                loc_text = sib.get_text(strip=True).lower() if sib else ""
            is_lan = "online" not in loc_text and "tbd" not in loc_text and bool(loc_text)

    else:
        # Strategy B: check event "type" badge (Online / LAN / Regional)
        type_el = soup.find(class_=re.compile(r"event.?type|eventtype", re.I))
        if type_el:
            type_text = type_el.get_text(strip=True).lower()
            is_lan = "online" not in type_text
        else:
            # Strategy C: scan opening portion of the page text
            head_text = soup.get_text(" ")[:1500].lower()
            is_lan = "online" not in head_text

    result["is_lan"] = is_lan

    # ── Event name ─────────────────────────────────────────────────────
    h1 = soup.find("h1")
    if h1:
        result["event_name"] = h1.get_text(strip=True)

    time.sleep(_REQ_DELAY)
    return result


# ── Disk cache ─────────────────────────────────────────────────────────────────
def _cache_path(start_date: str, end_date: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"hltv_{start_date}_{end_date}.json")


def load_from_cache(start_date: str, end_date: str) -> pd.DataFrame | None:
    """Return cached DataFrame if the cache file exists, else None."""
    path = _cache_path(start_date, end_date)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["is_lan"] = df["is_lan"].astype(bool)
        return df
    except Exception as exc:
        logger.warning("Cache read failed (%s): %s", path, exc)
        return None


def _save_cache(start_date: str, end_date: str, df: pd.DataFrame) -> None:
    path = _cache_path(start_date, end_date)
    try:
        copy = df.copy()
        copy["date"] = copy["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(copy.to_dict(orient="records"), f, indent=2)
    except Exception as exc:
        logger.warning("Cache write failed (%s): %s", path, exc)


def cache_exists(start_date: str, end_date: str) -> bool:
    return os.path.exists(_cache_path(start_date, end_date))


def cache_mtime(start_date: str, end_date: str) -> datetime | None:
    path = _cache_path(start_date, end_date)
    if not os.path.exists(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path))


def clear_cache(start_date: str, end_date: str) -> None:
    path = _cache_path(start_date, end_date)
    if os.path.exists(path):
        os.remove(path)


# ── Main entry point ───────────────────────────────────────────────────────────
def fetch_hltv_matches(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """
    Fetch CS2 series results from HLTV.org for *start_date* → *end_date*.

    Parameters
    ----------
    start_date:
        ISO date string (YYYY-MM-DD), inclusive.
    end_date:
        ISO date string (YYYY-MM-DD), inclusive.
    force_refresh:
        When True, skip the disk cache and fetch fresh data.
    progress_callback:
        Optional callable(page_num: int, total_rows: int, status: str).
        Called after each network request so the UI can show progress.

    Returns
    -------
    pd.DataFrame with columns:
        date          – datetime
        winner        – str  (canonical team name)
        loser         – str
        event         – str  (event name)
        prize_pool    – float (USD; 0.0 when unknown)
        winner_prize  – float (always 0.0 — HLTV doesn't expose split earnings)
        loser_prize   – float (always 0.0)
        is_lan        – bool

    Raises
    ------
    RuntimeError
        When HLTV is reachable but returns a Cloudflare challenge page.
    """
    # ── Cache hit ──────────────────────────────────────────────────────
    if not force_refresh:
        cached = load_from_cache(start_date, end_date)
        if cached is not None:
            return cached

    session  = _make_session()
    all_maps: list[dict] = []

    # ── Phase 1: paginate /stats/matches ──────────────────────────────
    offset   = 0
    page_num = 0

    while page_num < _MAX_PAGES:
        url  = f"{STATS_MATCHES}?startDate={start_date}&endDate={end_date}&offset={offset}"

        if progress_callback:
            progress_callback(page_num + 1, len(all_maps),
                              f"Fetching match results (page {page_num + 1})…")

        html = _get(session, url)      # may raise RuntimeError on CF block

        if html is None:
            if page_num == 0:
                raise RuntimeError(
                    "HLTV.org returned no data on the first request. "
                    "The site may be temporarily unavailable."
                )
            break   # later pages silently stop pagination

        page_maps = _parse_matches_page(html)
        all_maps.extend(page_maps)
        page_num += 1

        if len(page_maps) < _PAGE_SIZE:
            break   # final page — fewer than PAGE_SIZE rows → done

        offset += _PAGE_SIZE
        time.sleep(_REQ_DELAY)

    if not all_maps:
        return pd.DataFrame()

    # ── Phase 2: deduplicate map rows → series ─────────────────────────
    series = _to_series(all_maps)
    if not series:
        return pd.DataFrame()

    # ── Phase 3: event metadata (one page per unique event) ────────────
    event_ids = sorted({s["event_id"] for s in series if s["event_id"]})
    event_meta: dict[int, dict] = {}

    for i, eid in enumerate(event_ids):
        slug = next(s["event_slug"] for s in series if s["event_id"] == eid)

        if progress_callback:
            progress_callback(
                page_num + i + 1,
                len(series),
                f"Fetching event details ({i + 1}/{len(event_ids)}): {slug}…",
            )

        event_meta[eid] = _fetch_event_meta(session, eid, slug)
        # _fetch_event_meta already sleeps _REQ_DELAY after each fetch

    # ── Phase 4: build output DataFrame ───────────────────────────────
    default_meta = {"prize_pool": 0.0, "is_lan": False, "event_name": ""}
    output: list[dict] = []

    for s in series:
        meta = event_meta.get(s["event_id"], default_meta)
        output.append({
            "date":         s["date"],
            "winner":       s["winner"],
            "loser":        s["loser"],
            "event":        meta.get("event_name", s["event_name"]) or s["event_name"],
            "prize_pool":   float(meta.get("prize_pool", 0.0)),
            "winner_prize": 0.0,
            "loser_prize":  0.0,
            "is_lan":       bool(meta.get("is_lan", False)),
        })

    df = (pd.DataFrame(output)
            .sort_values("date")
            .reset_index(drop=True))

    _save_cache(start_date, end_date, df)
    return df
