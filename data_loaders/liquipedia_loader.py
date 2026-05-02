"""
Liquipedia Match Loader
=======================
Fetches CS2 tournament results from Liquipedia tournament pages for a given
set of tournament slugs and converts them into the schema expected by
compute_vrs() / compute_standings().

Pipeline
--------
  1. For each slug, fetch the tournament page at liquipedia.net/counterstrike/<slug>
  2. Parse the .fo-nttax-infobox for prize pool, LAN status, event name, dates
  3. Parse all .brkts-match elements for individual series results
  4. Filter by date range, normalise team names, build output DataFrame
  5. Persist result as a JSON file in cache/  (keyed by date range + slug hash)

HTTP approach
-------------
  Uses plain requests — Liquipedia pages are accessible without JavaScript
  rendering. A small delay between requests respects Liquipedia's rate limits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Callable

import pandas as pd
import requests

# BeautifulSoup is imported lazily so the rest of the app can start even if
# beautifulsoup4 is not yet installed.
def _bs4(html: str, parser: str = "html.parser"):
    """Lazy BeautifulSoup constructor — imports bs4 on first call."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "beautifulsoup4 is required for the Liquipedia loader. "
            "Install it with:  pip install beautifulsoup4"
        ) from exc
    return BeautifulSoup(html, parser)


logger = logging.getLogger(__name__)

# ── HTTP constants ──────────────────────────────────────────────────────────────
_BASE_URL = "https://liquipedia.net/counterstrike"
_HEADERS = {
    "User-Agent": "VRS-Simulator/1.0 (CS2 tournament data loader)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip",
}
_REQ_DELAY = 2.5
_TIMEOUT = 20
CACHE_DIR = "cache"


# ── Team-name normalisation ─────────────────────────────────────────────────────
# Maps Liquipedia display names → Valve / KNOWN_META canonical keys.
LIQUIPEDIA_TO_VALVE: dict[str, str] = {
    "Team Vitality": "Vitality",
    "Team Spirit": "Spirit",
    "FaZe Clan": "FaZe",
    "G2 Esports": "G2",
    "Team Liquid": "Liquid",
    "Natus Vincere": "Natus Vincere",
    "Team Falcons": "Falcons",
    "paiN Gaming": "paiN",
    "FURIA Esports": "FURIA",
    "FURIA": "FURIA",
    "BIG": "BIG",
    # pass-through (name same in both systems):
    "HEROIC": "HEROIC",
    "3DMAX": "3DMAX",
    "GamerLegion": "GamerLegion",
    "MOUZ": "MOUZ",
    "Virtus.pro": "Virtus.pro",
    "Astralis": "Astralis",
    "MIBR": "MIBR",
    "Eternal Fire": "Eternal Fire",
    "The MongolZ": "The MongolZ",
    "Wildcard": "Wildcard",
    "FlyQuest": "FlyQuest",
    "Complexity": "Complexity",
    "SAW": "SAW",
    "NRG": "NRG",
    "Cloud9": "Cloud9",
    "Aurora": "Aurora",
    "Aurora Gaming": "Aurora",
    "Falcons": "Falcons",
    "Monte": "Monte",
    "Rare Atom": "Rare Atom",
    "TYLOO": "TYLOO",
    "paiN": "paiN",
    # ── Additional mappings found from live Liquipedia data ───────────────
    "FUT Esports":          "FUT",
    "ASTRAL Esports":       "ASTRAL",
    "ESC Gaming":           "ESC",
    "JUMBO TEAM":           "JUMBO",
    "Leo Team":             "Leo",
    "MANA eSports":         "MANA",
    "Players (Russian team)": "Players",
    "Rune Eaters Esports":  "Rune Eaters",
    "Team Nemesis":         "Nemesis",
    "TNC Esport":           "TNC",
    "Tricked Esport":       "Tricked",
    "UNiTY esports":        "UNiTY",
}


def _norm(name: str) -> str:
    """Return the canonical (Valve) team name for a given Liquipedia display name."""
    clean = name.strip()
    return LIQUIPEDIA_TO_VALVE.get(clean, clean)


# ── HTTP helpers ────────────────────────────────────────────────────────────────

def search_tournaments(query: str) -> list[dict]:
    """
    Search Liquipedia for CS2 tournament pages matching *query*.

    Returns a list of dicts:
        title  – human-readable page title
        slug   – the Liquipedia slug (paste into the slug text area)
        url    – full URL for reference
    """
    try:
        r = requests.get(
            "https://liquipedia.net/counterstrike/index.php",
            params={"search": query, "ns0": "1"},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
    except Exception as exc:
        logger.warning("Search request failed: %s", exc)
        return []

    if r.status_code != 200:
        logger.warning("Search HTTP %s", r.status_code)
        return []

    soup = _bs4(r.text)
    results = []
    for a in soup.select(".mw-search-result-heading a"):
        href  = a.get("href", "")
        title = a.get_text(strip=True)
        # href looks like /counterstrike/Intel_Extreme_Masters/2025/Katowice
        prefix = "/counterstrike/"
        if href.startswith(prefix):
            slug = href[len(prefix):]
            results.append({
                "title": title,
                "slug":  slug,
                "url":   f"https://liquipedia.net{href}",
            })
    return results


# Tier ordering for filtering (higher index = lower tier)
_TIER_ORDER = ["S-Tier", "A-Tier", "B-Tier", "C-Tier"]

# Slugs that point to index/meta pages rather than actual tournaments
_PORTAL_SKIP_SLUGS = {
    "C-Tier_Tournaments", "B-Tier_Tournaments", "A-Tier_Tournaments",
    "S-Tier_Tournaments", "Qualifier_Tournaments", "Valve_Tournaments",
    "Main_Page", "Recent_Tournament_Results",
}


def _parse_portal_date_range(raw: str):
    """
    Parse Portal:Tournaments date strings into (start, end) datetime tuples.

    Handles formats:
      "Jun 20, 2026"               → single day
      "Apr 02–03, 2026"            → same month range (en-dash)
      "Mar 22 – Apr 03, 2026"      → cross-month range
    Returns (None, None) when unparseable.
    """
    # Normalise em/en-dashes → hyphen, collapse surrounding spaces
    s = raw.replace("\u2013", "-").replace("\u2014", "-").strip()
    s = re.sub(r"\s*-\s*", " - ", s)

    # Single day: "Jun 20, 2026"
    m = re.fullmatch(r"([A-Za-z]+ \d{1,2}), (\d{4})", s)
    if m:
        try:
            d = datetime.strptime(f"{m.group(1)}, {m.group(2)}", "%b %d, %Y")
            return d, d
        except ValueError:
            pass

    # Same month: "Apr 02 - 03, 2026"
    m = re.fullmatch(r"([A-Za-z]+) (\d{1,2}) - (\d{1,2}), (\d{4})", s)
    if m:
        try:
            start = datetime.strptime(f"{m.group(1)} {m.group(2)}, {m.group(4)}", "%b %d, %Y")
            end   = datetime.strptime(f"{m.group(1)} {m.group(3)}, {m.group(4)}", "%b %d, %Y")
            return start, end
        except ValueError:
            pass

    # Cross-month: "Mar 22 - Apr 03, 2026"
    m = re.fullmatch(r"([A-Za-z]+ \d{1,2}) - ([A-Za-z]+ \d{1,2}), (\d{4})", s)
    if m:
        try:
            start = datetime.strptime(f"{m.group(1)}, {m.group(3)}", "%b %d, %Y")
            end   = datetime.strptime(f"{m.group(2)}, {m.group(3)}", "%b %d, %Y")
            return start, end
        except ValueError:
            pass

    return None, None


def discover_from_portal(
    start_date: str,
    end_date: str,
    min_tier: str = "B-Tier",
    include_qualifiers: bool = False,
) -> list[dict]:
    """
    Discover CS2 tournaments from the Liquipedia Portal:Tournaments page,
    filtered to those overlapping [start_date, end_date].

    Parameters
    ----------
    start_date : str
        ISO date string (YYYY-MM-DD).
    end_date : str
        ISO date string (YYYY-MM-DD).
    min_tier : str
        Minimum tier to include. One of "S-Tier", "A-Tier", "B-Tier", "C-Tier".
        Defaults to "B-Tier" (includes S, A, B — excludes C-tier).
    include_qualifiers : bool
        When True, also include qualifier tournaments (rows whose tier starts
        with "Qual."). These are listed in a separate table on the portal page.
        Defaults to False.

    Returns
    -------
    list of dicts with keys: title, slug, url, tier, start_date, end_date
        Sorted by tournament start date (ascending), then title.
    """
    try:
        filter_start = datetime.strptime(start_date, "%Y-%m-%d")
        filter_end   = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )
    except ValueError:
        logger.warning("Invalid date range: %s – %s", start_date, end_date)
        return []

    # Determine allowed tiers
    try:
        max_tier_idx = _TIER_ORDER.index(min_tier)
    except ValueError:
        max_tier_idx = _TIER_ORDER.index("B-Tier")
    allowed_tiers = set(_TIER_ORDER[:max_tier_idx + 1])

    html = _fetch_page("Portal:Tournaments")
    if html is None:
        logger.warning("Could not fetch Portal:Tournaments")
        return []

    soup = _bs4(html)
    tables = soup.select("table.table2__table")
    if not tables:
        logger.warning("No .table2__table found on Portal:Tournaments")
        return []

    seen_slugs: set[str] = set()
    results: list[dict] = []

    for table in tables:
        for row in table.find_all("tr")[1:]:  # skip header row
            cells = row.find_all(["td", "th"])
            if len(cells) < 5:
                continue

            tier_raw  = cells[0].get_text(strip=True)
            date_raw  = cells[4].get_text(strip=True)

            # Qualifier rows: tier looks like "Qual.(C-Tier)" or "Qual.(B-Tier)"
            is_qualifier = tier_raw.startswith("Qual.")
            if is_qualifier:
                if not include_qualifiers:
                    continue
                tier_norm = "Qualifier"
            else:
                tier_norm = next(
                    (t for t in _TIER_ORDER if tier_raw.startswith(t)), None
                )
                if tier_norm not in allowed_tiers:
                    continue

            # Tournament link in cells[3]
            link = cells[3].find("a", href=True)
            if not link:
                continue
            href = link["href"]
            if not href.startswith("/counterstrike/"):
                continue
            slug = href.replace("/counterstrike/", "")
            if slug in _PORTAL_SKIP_SLUGS or ":" in slug:
                continue
            if slug in seen_slugs:
                continue

            # Date range filter
            t_start, t_end = _parse_portal_date_range(date_raw)
            if t_start is None or t_end is None:
                continue
            if t_end < filter_start or t_start > filter_end:
                continue

            seen_slugs.add(slug)
            title = link.get_text(strip=True)
            results.append({
                "title":      title,
                "slug":       slug,
                "url":        f"https://liquipedia.net{href}",
                "tier":       tier_norm,
                "start_date": t_start,
                "end_date":   t_end,
            })

    # Sort by start date ascending, then title
    results.sort(key=lambda x: (x["start_date"], x["title"]))
    logger.info("Portal discovery: %d tournaments in range %s – %s",
                len(results), start_date, end_date)
    return results


def _fetch_page(slug: str) -> str | None:
    """
    HTTP GET liquipedia.net/counterstrike/<slug>.
    Returns HTML string on success, or None on any failure.
    Logs the status code on non-200 responses.
    """
    url = f"{_BASE_URL}/{slug}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    except Exception as exc:
        logger.warning("Request failed (%s): %s", url, exc)
        return None

    if r.status_code != 200:
        logger.warning("HTTP %s for %s", r.status_code, url)
        return None

    return r.text


# ── Infobox parsing ─────────────────────────────────────────────────────────────

def _parse_infobox(soup) -> dict:
    """
    Extract event metadata from the Liquipedia infobox.

    Returns a dict with keys:
      prize_pool  float
      is_lan      bool
      event_name  str
      start_date  datetime | None
      end_date    datetime | None
    """
    result = {
        "prize_pool": 0.0,
        "is_lan": False,
        "event_name": "",
        "start_date": None,
        "end_date": None,
    }

    # Event name from <h1>
    h1 = soup.find("h1")
    if h1:
        result["event_name"] = h1.get_text(strip=True)

    # Try .fo-nttax-infobox first, fall back to any table containing "Prize"
    infobox = soup.select_one(".fo-nttax-infobox")
    if infobox is None:
        for tbl in soup.find_all("table"):
            if "Prize" in tbl.get_text():
                infobox = tbl
                break

    if infobox is None:
        return result

    text = infobox.get_text(" ", strip=True)

    # Prize pool: find $N,NNN,NNN pattern
    m = re.search(r"\$([\d,]+)", text)
    if m:
        try:
            result["prize_pool"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # LAN status: "Offline" → LAN, "Online" → online
    if "Offline" in text:
        result["is_lan"] = True
    elif "Online" in text:
        result["is_lan"] = False

    # Start date
    m = re.search(r"Start Date:\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        try:
            result["start_date"] = datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass

    # End date
    m = re.search(r"End Date:\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        try:
            result["end_date"] = datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            pass

    return result


def _infobox_field(soup, label: str) -> str:
    """
    Look up a labelled value in the .fo-nttax-infobox.

    Liquipedia infobox rows are ``<div><div class="infobox-cell-2
    infobox-description">Label:</div><div style="width:50%">Value</div></div>``
    — i.e. the value is the **next sibling** of the description cell within a
    wrapping div, not another ``.infobox-cell-2``. This helper finds the
    description whose text matches ``label`` (case-insensitive, trailing colon
    optional) and returns the sibling's text.

    Returns "" when the label is not present.
    """
    infobox = soup.select_one(".fo-nttax-infobox")
    if infobox is None:
        return ""
    needle = label.strip().rstrip(":").lower()
    for desc in infobox.select(".infobox-description"):
        text = desc.get_text(" ", strip=True).rstrip(":").strip().lower()
        if text == needle:
            value = desc.find_next_sibling()
            if value is not None:
                return value.get_text(" ", strip=True)
            return ""
    return ""


def _parse_format(soup) -> str:
    """
    Extract a short, human-readable description of the tournament format.

    Liquipedia tournament pages don't carry "Format" as an infobox field —
    instead they have a ``<h3>Format</h3>`` section whose body lists bullet
    points ("Single-elimination playoffs", "Swiss group stage", …). On modern
    MediaWiki the heading is wrapped in ``<div class="mw-heading mw-heading3">``
    so the prose paragraphs are siblings of the *wrapper*, not the bare h3.
    This helper handles both the wrapped and bare layouts.

    Walks forward from the Format heading collecting the text of every
    ``<p>``, ``<ul>``, ``<ol>`` and ``<dl>`` until the next heading. Returns
    "" when no Format section is present.

    The full untruncated prose is returned — downstream consumers (the
    structured ``format_parser.parse_format_prose`` parser, plus UI
    rendering) handle their own truncation if they want a short version.
    """
    start = None
    # Modern: <div class="mw-heading"> wraps the h2/h3
    for wrap in soup.select("div.mw-heading"):
        heading = wrap.find(["h2", "h3"])
        if heading and heading.get_text(strip=True).lower() == "format":
            start = wrap
            break
    # Fallback: bare h2/h3 (older Liquipedia / non-wrapped layout)
    if start is None:
        for h in soup.select("h2, h3"):
            if h.get_text(strip=True).lower() == "format":
                start = h
                break
    if start is None:
        return ""

    parts: list[str] = []
    for sib in start.next_siblings:
        name = getattr(sib, "name", None)
        if name is None:
            continue
        # Stop at the next section heading (wrapper or bare)
        if name in ("h2", "h3"):
            break
        if name == "div" and any(c.startswith("mw-heading") for c in sib.get("class", [])):
            break
        if name in ("p", "ul", "ol", "dl"):
            chunk = sib.get_text(" ", strip=True)
            if chunk:
                parts.append(chunk)

    return " · ".join(parts).strip()


# ── Date string parsing ─────────────────────────────────────────────────────────

def _parse_date_str(raw: str) -> datetime | None:
    """
    Parse a Liquipedia date string such as "January 29, 2025 - 18:55CET".

    Steps:
      1. Strip trailing timezone abbreviation (e.g. CET, UTC, EST)
      2. Try several common date/datetime formats
    """
    # Strip trailing timezone abbreviation
    clean = re.sub(r'\s*[A-Z]{2,5}\s*$', '', raw).strip()

    formats = [
        "%B %d, %Y - %H:%M",
        "%B %d, %Y",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            pass

    return None


# ── Prize pool parsing ──────────────────────────────────────────────────────────

def _parse_prize_pool(soup) -> dict[str, float]:
    """
    Parse the prize pool placement table from a tournament page.

    Looks for ``div.prizepool-section-wrapper`` which contains
    ``div.csstable-widget-row`` entries, each with a place cell and one or more
    team names obtained from ``.block-team a`` anchors.

    Returns a dict mapping canonical team name → prize (USD float).
    Teams sharing a placement (e.g. 3rd–4th) each receive the listed prize.
    """
    team_prizes: dict[str, float] = {}

    wrapper = soup.select_one(".prizepool-section-wrapper")
    if wrapper is None:
        return team_prizes

    for row in wrapper.select(".csstable-widget-row"):
        # Skip header row and toggle/expand rows
        if "prizepooltable-header" in row.get("class", []):
            continue
        if "ppt-toggle-expand" in row.get("class", []):
            continue

        # Prize amount is in the second cell (index 1 after the place cell)
        cells = row.select(".csstable-widget-cell")
        if len(cells) < 2:
            continue

        prize_text = cells[1].get_text(strip=True)
        prize_text = prize_text.replace("$", "").replace(",", "").strip()
        try:
            prize = float(prize_text)
        except ValueError:
            continue  # row has no numeric prize (e.g. tokens-only rows)

        if prize <= 0:
            continue

        # All teams in this row share the listed prize amount
        for a in row.select(".block-team a[href]"):
            team_raw = a.get("title", "") or a.get_text(strip=True)
            team_raw = team_raw.strip()
            if team_raw:
                team_prizes[_norm(team_raw)] = prize

    return team_prizes


def _parse_prize_distribution_by_place(soup) -> dict[str, float]:
    """
    Parse the prize pool placement table keyed by placement label rather than
    team name. Useful for upcoming events where the placement column carries a
    label (``"1st"``, ``"2nd"``, ``"3rd-4th"``, …) but the team cells are still
    empty.

    Returns a dict like ``{"1st": 250000.0, "2nd": 100000.0, "3rd-4th": 35000.0}``.
    Shared placements keep the merged label so the consumer can decide how to
    distribute among slot-holders.
    """
    placements: dict[str, float] = {}
    wrapper = soup.select_one(".prizepool-section-wrapper")
    if wrapper is None:
        return placements

    for row in wrapper.select(".csstable-widget-row"):
        if "prizepooltable-header" in row.get("class", []):
            continue
        if "ppt-toggle-expand" in row.get("class", []):
            continue

        cells = row.select(".csstable-widget-cell")
        if len(cells) < 2:
            continue

        place_text = cells[0].get_text(" ", strip=True)
        # Normalise dashes in shared placements ("3rd – 4th" → "3rd-4th")
        place_norm = re.sub(
            r"\s*[\u2013\u2014-]\s*", "-",
            place_text.replace("\xa0", " ").strip(),
        )
        if not place_norm:
            continue

        prize_text = cells[1].get_text(strip=True).replace("$", "").replace(",", "").strip()
        try:
            prize = float(prize_text)
        except ValueError:
            continue
        if prize <= 0:
            continue

        # Last entry wins on duplicate place labels (Liquipedia occasionally
        # repeats rows for tokens / region pools — we already filtered to
        # rows with a USD prize, so collisions are rare).
        placements[place_norm] = prize

    return placements


# Withdrawal note pattern. Liquipedia phrases these as
#   "April 4th - FUT Esports withdraw from the event; ..."
#   "March 31st - 33 withdraw from the event due to visa issues; ..."
# Anchor the team name on a space-flanked dash separator so the leading
# date prefix isn't sucked into the capture by a non-greedy `.*?`.
_WITHDRAW_NOTE_RE = re.compile(
    r"\s[—–\-]\s+"
    r"([A-Z0-9][\w .']*?)\s+"
    r"(?:withdraws?|withdrew|has\s+withdrawn)\b",
    re.IGNORECASE,
)


def _parse_withdrawn_teams(soup) -> set[str]:
    """
    Pull canonical names of teams flagged as withdrawn / replaced in the
    page's ``div.inotes-inner`` blocks.

    Liquipedia phrases withdrawal notes as e.g.
    ``"FUT Esports withdraw from the event; they are replaced by K27 ."``
    The first capture group of ``_WITHDRAW_NOTE_RE`` is the team that left.
    """
    out: set[str] = set()
    for note_box in soup.select("div.inotes-inner"):
        text = note_box.get_text(" ", strip=True)
        for m in _WITHDRAW_NOTE_RE.finditer(text):
            raw = m.group(1).strip().rstrip(".,;:")
            canon = _norm(raw)
            if canon:
                out.add(canon)
    return out


_STAGE_INVITE_HEADING_RE = re.compile(
    r"^\s*(Stage\s*\d+)\s+Invites?\s*$", re.IGNORECASE,
)


def _parse_per_stage_invites(soup) -> dict[str, list[str]]:
    """
    For Major-tier events whose Participants section carries h3 sub-headings
    of the form ``"Stage N Invites"``, return ``{stage_name: [teams]}``
    grouping each h3 block's teamcards under its stage label.

    Returns ``{}`` when the page doesn't use this layout (i.e. all single-
    stage events). Withdrawn teams (per ``_parse_withdrawn_teams``) are
    excluded just like in ``_parse_seeded_teams``.
    """
    withdrawn = _parse_withdrawn_teams(soup)
    out: dict[str, list[str]] = {}

    for wrap in soup.select("div.mw-heading"):
        h = wrap.find(["h2", "h3", "h4"])
        if h is None:
            continue
        m = _STAGE_INVITE_HEADING_RE.match(h.get_text(" ", strip=True))
        if m is None:
            continue
        stage_name = re.sub(r"\s+", " ", m.group(1).strip())
        teams: list[str] = []
        seen: set[str] = set()
        for sib in wrap.next_siblings:
            name = getattr(sib, "name", None)
            if name is None:
                continue
            # Stop at the next mw-heading wrapper of any level.
            if name == "div" and any(c.startswith("mw-heading")
                                     for c in sib.get("class", [])):
                break
            if name in ("h2", "h3", "h4"):
                break
            # The sibling itself may *be* a teamcard, or contain teamcards as
            # descendants (Liquipedia sometimes wraps cards in a row div).
            cards = []
            if hasattr(sib, "get") and "teamcard" in (sib.get("class") or []):
                cards.append(sib)
            if hasattr(sib, "select"):
                cards.extend(sib.select(".teamcard"))
            for card in cards:
                head = card.find("center") or card
                a = head.select_one("a[title][href^='/counterstrike/']")
                if a is None:
                    continue
                tname = (a.get("title") or "").strip()
                if not tname or ":" in tname or tname.upper() in {"TBD", "TBA"}:
                    continue
                canon = _norm(tname)
                if canon in seen or canon in withdrawn:
                    continue
                seen.add(canon)
                teams.append(canon)
        if teams:
            out[stage_name] = teams

    return out


def _parse_seeded_teams(soup) -> list[str]:
    """
    Parse participating / seeded team names from a tournament page.

    Liquipedia renders each participant as a ``.teamcard`` block. The team's
    name lives in the heading link (``<a title="Team Vitality" …>``) inside the
    centred title row. Player rows under the same teamcard link to player
    pages and are filtered out via the ``title`` attribute (player titles
    typically include the team name as a parenthetical).

    Teams flagged as withdrawn in ``div.inotes-inner`` notes (e.g. *"FUT
    Esports withdraw from the event; they are replaced by K27"*) are
    excluded — their replacement is already listed as its own teamcard.

    Returns canonical (Valve-normalised) names with duplicates removed and
    insertion order preserved.
    """
    withdrawn = _parse_withdrawn_teams(soup)

    teams: list[str] = []
    seen: set[str] = set()

    for card in soup.select(".teamcard"):
        # The first <center> in the teamcard holds the team title row; fall
        # back to the card itself if the markup ever changes.
        head = card.find("center") or card
        a = head.select_one("a[title][href^='/counterstrike/']")
        if a is None:
            continue
        name = (a.get("title") or "").strip()
        if not name or ":" in name:
            continue
        # Skip TBD / placeholder cards
        if name.upper() in {"TBD", "TBA"}:
            continue
        canon = _norm(name)
        if canon in seen or canon in withdrawn:
            continue
        seen.add(canon)
        teams.append(canon)

    return teams


# ── Match parsing ───────────────────────────────────────────────────────────────

def _parse_match(
    el,
    event_name: str,
    prize_pool: float,
    is_lan: bool,
) -> dict | None:
    """
    Parse a single .brkts-match element into a result dict.

    winner_prize and loser_prize are always 0.0 here; prize data is added
    separately as prize rows (one per team per tournament, dated at the
    tournament end date) so that BO is attributed to the correct date.

    Returns None if the match is incomplete / undecided / unparseable.
    """
    # Teams via aria-label on .brkts-opponent-entry divs
    opponent_entries = el.select(".brkts-opponent-entry")
    if len(opponent_entries) < 2:
        return None

    team1_raw = opponent_entries[0].get("aria-label", "").strip()
    team2_raw = opponent_entries[1].get("aria-label", "").strip()

    if not team1_raw or not team2_raw:
        return None

    # Scores from .brkts-opponent-score-inner
    score_els = el.select(".brkts-opponent-score-inner")
    score_texts = [s.get_text(strip=True) for s in score_els]
    scores = [s for s in score_texts if s.isdigit()]

    if len(scores) < 2:
        return None

    score1, score2 = int(scores[0]), int(scores[1])

    # Skip undecided (equal scores) or matches with no data
    if score1 == score2:
        return None

    # Determine winner: .brkts-opponent-win is on .brkts-opponent-entry-left,
    # but aria-label lives on its PARENT .brkts-opponent-entry.
    winner_raw: str | None = None
    for entry_left in el.select(".brkts-opponent-entry-left"):
        if "brkts-opponent-win" in entry_left.get("class", []):
            winner_raw = entry_left.parent.get("aria-label", "").strip() or None
            break

    # Fallback: derive winner from series scores
    if not winner_raw:
        winner_raw = team1_raw if score1 > score2 else team2_raw

    loser_raw = team2_raw if winner_raw == team1_raw else team1_raw

    # Date from .timer-object
    date_el = el.select_one(".timer-object")
    match_date: datetime | None = None
    if date_el:
        match_date = _parse_date_str(date_el.get_text(strip=True))

    winner = _norm(winner_raw)
    loser  = _norm(loser_raw)

    return {
        "date":         match_date,
        "winner":       winner,
        "loser":        loser,
        "event":        event_name,
        "prize_pool":   prize_pool,
        "winner_prize": 0.0,
        "loser_prize":  0.0,
        "is_lan":       is_lan,
    }


def _parse_matchlist_match(
    el,
    event_name: str,
    prize_pool: float,
    is_lan: bool,
) -> dict | None:
    """
    Parse a single ``.brkts-matchlist-match`` element (Swiss / round-robin /
    group-stage list-style matches).

    Structure differs from bracket matches:
      * Opponents live on ``.brkts-matchlist-opponent`` (not ``.brkts-opponent-entry``).
      * Winner slot is marked with ``brkts-matchlist-slot-winner``.
      * Scores live in ``.brkts-matchlist-score`` cells, whose direct child
        ``.brkts-matchlist-cell-content`` holds the digit.
      * Timer lives in a ``.timer-object`` inside the match popup, same as
        bracket matches.
    """
    opponent_cells = el.select(".brkts-matchlist-opponent")
    if len(opponent_cells) < 2:
        return None

    team1_raw = opponent_cells[0].get("aria-label", "").strip()
    team2_raw = opponent_cells[1].get("aria-label", "").strip()
    if not team1_raw or not team2_raw:
        return None

    # Scores: look inside .brkts-matchlist-score cells for a digit. Fall back
    # to iterating cell-content children if the score markup varies.
    score_cells = el.select(".brkts-matchlist-score")
    scores: list[int] = []
    for sc in score_cells:
        txt = sc.get_text(strip=True)
        if txt.isdigit():
            scores.append(int(txt))
    if len(scores) < 2:
        return None

    score1, score2 = scores[0], scores[1]
    if score1 == score2:
        return None  # undecided / forfeit shown as equal scores

    # Winner: opponent cell with 'brkts-matchlist-slot-winner' class.
    winner_raw: str | None = None
    for op in opponent_cells:
        if "brkts-matchlist-slot-winner" in op.get("class", []):
            winner_raw = op.get("aria-label", "").strip() or None
            break
    if not winner_raw:
        winner_raw = team1_raw if score1 > score2 else team2_raw
    loser_raw = team2_raw if winner_raw == team1_raw else team1_raw

    date_el = el.select_one(".timer-object")
    match_date: datetime | None = None
    if date_el:
        match_date = _parse_date_str(date_el.get_text(strip=True))

    return {
        "date":         match_date,
        "winner":       _norm(winner_raw),
        "loser":        _norm(loser_raw),
        "event":        event_name,
        "prize_pool":   prize_pool,
        "winner_prize": 0.0,
        "loser_prize":  0.0,
        "is_lan":       is_lan,
    }


# ── Tournament page fetcher ─────────────────────────────────────────────────────

def fetch_tournament_page(
    slug: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    snapshot_cutoff: datetime | None = None,
) -> list[dict]:
    """
    Fetch a single Liquipedia tournament page, parse the infobox and all
    .brkts-match elements. Optionally filter matches by date range.

    Unfinished-event backfill
    -------------------------
    Valve's ranking algorithm drops every match from an event whose
    ``finished`` flag is false at snapshot time (see Valve's data_loader.js,
    ``filterInProgressEvents``). In practice: if an event's end date falls on
    or after the snapshot release, the event is dropped whole — even matches
    played days earlier are excluded.

    To backfill those missing matches we need to fetch the *entire* event from
    Liquipedia, not just matches after the snapshot cutoff. When
    ``snapshot_cutoff`` is provided and the event's end date is on/after it,
    this function bypasses the per-match ``start_dt`` filter. The ``end_dt``
    filter is still applied (we never want matches beyond the current cutoff).

    Returns a list of match dicts (conforming to the 8-column output schema).
    """
    html = _fetch_page(slug)
    if html is None:
        logger.warning("Could not fetch tournament page: %s", slug)
        return []

    soup = _bs4(html)
    meta = _parse_infobox(soup)

    event_name  = meta["event_name"] or slug
    prize_pool  = meta["prize_pool"]
    is_lan      = meta["is_lan"]
    event_end   = meta["end_date"]

    # Detect unfinished-at-snapshot events (Valve excludes these entirely).
    # For these, we fetch the whole event regardless of per-match date, so the
    # early-round matches aren't lost between Valve's "finished" drop and our
    # own post-cutoff window.
    is_unfinished_at_snapshot = (
        snapshot_cutoff is not None
        and event_end is not None
        and event_end >= snapshot_cutoff
    )
    effective_start_dt = None if is_unfinished_at_snapshot else start_dt

    # Prize distribution: team → amount they earned for their final placement
    team_prizes = _parse_prize_pool(soup)
    logger.debug("Prize pool entries for %s: %s", slug, team_prizes)

    matches: list[dict] = []

    # Liquipedia uses two different match element types on the same page:
    #   * ``.brkts-match``          — bracket (playoff) matches
    #   * ``.brkts-matchlist-match`` — Swiss / round-robin / group-stage matches
    # Both must be parsed; each uses a distinct aria/class layout, so a
    # dedicated parser handles each.
    for selector, parser in (
        (".brkts-match",          _parse_match),
        (".brkts-matchlist-match", _parse_matchlist_match),
    ):
        for el in soup.select(selector):
            try:
                m = parser(el, event_name, prize_pool, is_lan)
                if m is None:
                    continue
                # Date range filter (individual series use their own match date).
                # effective_start_dt is None for events that were unfinished at
                # the Valve snapshot cutoff → we want the full event.
                if m["date"] is not None:
                    if effective_start_dt is not None and m["date"] < effective_start_dt:
                        continue
                    if end_dt is not None and m["date"] > end_dt:
                        continue
                matches.append(m)
            except Exception as exc:
                logger.debug("Match parse error in %s: %s", slug, exc)

    # ── Prize rows ────────────────────────────────────────────────────────────
    # One row per team per tournament, dated at the tournament END date.
    # These rows carry winner_prize for BO calculation only:
    #   - loser = "" → bo_factor[""] = 0  → contributes 0 to BC / ON
    #   - is_lan = False → excluded from LAN wins
    #   - "" not in ratings → skipped in H2H
    # Prize rows are only added when the tournament end date falls within the
    # requested date range (i.e. the prize has been "awarded" in the window).
    if team_prizes and event_end is not None:
        prize_in_range = True
        if start_dt is not None and event_end < start_dt:
            prize_in_range = False
        if end_dt is not None and event_end > end_dt:
            prize_in_range = False

        if prize_in_range:
            for team, prize_amount in team_prizes.items():
                matches.append({
                    "date":         event_end,
                    "winner":       team,
                    "loser":        "",           # sentinel — no real opponent
                    "event":        event_name,
                    "prize_pool":   prize_pool,
                    "winner_prize": prize_amount,
                    "loser_prize":  0.0,
                    "is_lan":       False,        # must not inflate LAN wins
                })

    n_series = len([m for m in matches if m["loser"] != ""])
    n_prizes = len([m for m in matches if m["loser"] == ""])
    if is_unfinished_at_snapshot:
        logger.info(
            "[backfill] %s end=%s strategy=full-event series=%d prizes=%d "
            "(event was in-progress at snapshot cutoff %s)",
            slug,
            event_end.strftime("%Y-%m-%d") if event_end else "?",
            n_series, n_prizes,
            snapshot_cutoff.strftime("%Y-%m-%d"),
        )
    else:
        logger.info("Parsed %d series + %d prize rows from %s",
                    n_series, n_prizes, slug)
    return matches


# ── Disk cache ──────────────────────────────────────────────────────────────────

def _cache_key(
    start_date: str,
    end_date: str,
    tournament_slugs: list[str],
    snapshot_cutoff: datetime | None = None,
) -> str:
    """
    Return an 8-character md5 hash over the sorted slug list plus the snapshot
    cutoff date.

    ``snapshot_cutoff`` is included because it changes per-event fetch
    behaviour: events unfinished at the cutoff are fetched whole, while other
    events honour the start-date filter. Two runs with different cutoffs would
    otherwise produce different data under the same cache file.
    """
    # ``v`` is a parser-schema version; bump it when the set of matches
    # returned for the same (slugs, cutoff) changes (e.g. new parser added).
    # This forces a one-time cache invalidation without requiring users to
    # manually delete cache files.
    #   v1 = pre-matchlist (bracket-only parsing)
    #   v2 = bracket + matchlist (Swiss / group-stage) parsing
    slugs_str = "|".join(sorted(tournament_slugs))
    cutoff_str = snapshot_cutoff.strftime("%Y-%m-%d") if snapshot_cutoff else ""
    payload = f"v2||{slugs_str}||cutoff={cutoff_str}"
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def _cache_path(start_date: str, end_date: str, key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"liquipedia_{start_date}_{end_date}_{key}.json")


def load_from_cache(
    start_date: str,
    end_date: str,
    tournament_slugs: list[str],
    snapshot_cutoff: datetime | None = None,
) -> pd.DataFrame | None:
    """Return cached DataFrame if the cache file exists, else None."""
    key = _cache_key(start_date, end_date, tournament_slugs, snapshot_cutoff)
    path = _cache_path(start_date, end_date, key)
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


def _save_cache(
    start_date: str,
    end_date: str,
    tournament_slugs: list[str],
    df: pd.DataFrame,
    snapshot_cutoff: datetime | None = None,
) -> None:
    key = _cache_key(start_date, end_date, tournament_slugs, snapshot_cutoff)
    path = _cache_path(start_date, end_date, key)
    try:
        copy = df.copy()
        copy["date"] = copy["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(copy.to_dict(orient="records"), f, indent=2)
    except Exception as exc:
        logger.warning("Cache write failed (%s): %s", path, exc)


def cache_exists(
    start_date: str,
    end_date: str,
    tournament_slugs: list[str],
    snapshot_cutoff: datetime | None = None,
) -> bool:
    key = _cache_key(start_date, end_date, tournament_slugs, snapshot_cutoff)
    return os.path.exists(_cache_path(start_date, end_date, key))


def cache_mtime(
    start_date: str,
    end_date: str,
    tournament_slugs: list[str],
    snapshot_cutoff: datetime | None = None,
) -> datetime | None:
    key = _cache_key(start_date, end_date, tournament_slugs, snapshot_cutoff)
    path = _cache_path(start_date, end_date, key)
    if not os.path.exists(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path))


def clear_cache(
    start_date: str,
    end_date: str,
    tournament_slugs: list[str],
    snapshot_cutoff: datetime | None = None,
) -> None:
    key = _cache_key(start_date, end_date, tournament_slugs, snapshot_cutoff)
    path = _cache_path(start_date, end_date, key)
    if os.path.exists(path):
        os.remove(path)


# ── Main entry point ────────────────────────────────────────────────────────────

def fetch_liquipedia_matches(
    start_date: str,
    end_date: str,
    tournament_slugs: list[str],
    force_refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    snapshot_cutoff: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch CS2 series results from Liquipedia for the given tournament slugs,
    filtered to the [start_date, end_date] range.

    Parameters
    ----------
    start_date:
        ISO date string (YYYY-MM-DD), inclusive.
    end_date:
        ISO date string (YYYY-MM-DD), inclusive.
    tournament_slugs:
        List of Liquipedia page slugs, e.g.
        ["Intel_Extreme_Masters/2025/Katowice"].
    force_refresh:
        When True, skip the disk cache and fetch fresh data.
    progress_callback:
        Optional callable(step: int, total: int, status: str).
        Called before each network request so the UI can show progress.
    snapshot_cutoff:
        Datetime of the Valve snapshot this fetch is backfilling. Events whose
        end date is on/after this cutoff were in-progress ("not finished") in
        Valve's snapshot and are excluded whole — so the fetcher pulls them
        completely (not just matches after ``start_date``). Omit for a pure
        windowed fetch.

    Returns
    -------
    pd.DataFrame with columns:
        date          – datetime
        winner        – str  (canonical team name)
        loser         – str
        event         – str  (event name)
        prize_pool    – float (USD; 0.0 when unknown)
        winner_prize  – float (always 0.0)
        loser_prize   – float (always 0.0)
        is_lan        – bool
    """
    if not tournament_slugs:
        return pd.DataFrame(columns=[
            "date", "winner", "loser", "event",
            "prize_pool", "winner_prize", "loser_prize", "is_lan",
        ])

    # ── Cache hit ──────────────────────────────────────────────────────
    if not force_refresh:
        cached = load_from_cache(
            start_date, end_date, tournament_slugs, snapshot_cutoff
        )
        if cached is not None:
            return cached

    # Parse date bounds for filtering
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        start_dt = None
    try:
        # Make end_dt inclusive by using end of day
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )
    except ValueError:
        end_dt = None

    all_matches: list[dict] = []
    total = len(tournament_slugs)

    for i, slug in enumerate(tournament_slugs):
        if progress_callback:
            progress_callback(i + 1, total, f"Fetching {slug} ({i + 1}/{total})…")

        page_matches = fetch_tournament_page(
            slug,
            start_dt=start_dt,
            end_dt=end_dt,
            snapshot_cutoff=snapshot_cutoff,
        )
        all_matches.extend(page_matches)

        # Be polite — sleep between requests (skip after last slug)
        if i < total - 1:
            time.sleep(_REQ_DELAY)

    if progress_callback:
        progress_callback(total, total, "Building results…")

    if not all_matches:
        return pd.DataFrame(columns=[
            "date", "winner", "loser", "event",
            "prize_pool", "winner_prize", "loser_prize", "is_lan",
        ])

    df = pd.DataFrame(all_matches)

    # Ensure date column is datetime; drop rows with no parseable date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Sort by date and reset index
    df = df.sort_values("date").reset_index(drop=True)

    # Enforce column types
    df["prize_pool"] = df["prize_pool"].astype(float)
    df["winner_prize"] = df["winner_prize"].astype(float)
    df["loser_prize"] = df["loser_prize"].astype(float)
    df["is_lan"] = df["is_lan"].astype(bool)

    _save_cache(start_date, end_date, tournament_slugs, df, snapshot_cutoff)
    return df


# ── Upcoming-event discovery ────────────────────────────────────────────────────

_UPCOMING_CACHE_TTL_SEC = 6 * 3600  # 6 hours: brackets / seeds shift daily near events


# Bump when the discovery payload schema changes — invalidates all
# previously cached upcoming-event JSON files.
#   v1 = initial 6.1 payload
#   v2 = full untruncated format prose (Phase 1 of bracket-system rewrite)
_UPCOMING_CACHE_VERSION = "v2"


def _upcoming_cache_path(start_date: str, end_date: str, min_tier: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(
        f"{_UPCOMING_CACHE_VERSION}|{start_date}|{end_date}|{min_tier}".encode()
    ).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"upcoming_{start_date}_{end_date}_{key}.json")


def _load_upcoming_cache(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) > _UPCOMING_CACHE_TTL_SEC:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as exc:
        logger.warning("Upcoming cache read failed (%s): %s", path, exc)
        return None
    for rec in records:
        for k in ("start_date", "end_date"):
            if rec.get(k):
                try:
                    rec[k] = datetime.fromisoformat(rec[k])
                except ValueError:
                    rec[k] = None
    return records


def _save_upcoming_cache(path: str, records: list[dict]) -> None:
    try:
        out = []
        for rec in records:
            copy = dict(rec)
            for k in ("start_date", "end_date"):
                if isinstance(copy.get(k), datetime):
                    copy[k] = copy[k].isoformat()
            out.append(copy)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception as exc:
        logger.warning("Upcoming cache write failed (%s): %s", path, exc)


def discover_upcoming_events(
    start_date: str,
    end_date: str,
    min_tier: str = "B-Tier",
    today: datetime | None = None,
    fetch_details: bool = True,
    force_refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    """
    Discover CS2 events whose end date is on/after ``today`` and which overlap
    the ``[start_date, end_date]`` window. For each event the per-page infobox,
    prize-pool table and participant teamcards are parsed to assemble the
    payload required by the Tournament Predictor (Pillar 3).

    Parameters
    ----------
    start_date, end_date : str
        ISO date strings (YYYY-MM-DD) defining the discovery window.
    min_tier : str
        Lowest tier to include — see :func:`discover_from_portal`.
    today : datetime | None
        Reference "now". Events with ``end_date < today`` are filtered out
        (they're in the past). Defaults to ``datetime.now()``.
    fetch_details : bool
        When False, skip per-event page fetches and return only the metadata
        already known from the portal page (slug, name, tier, dates, url).
        ``prize_pool``, ``is_lan``, ``format``, ``prize_distribution`` and
        ``seeded_teams`` will be empty/zero. Useful for cheap UI previews.
    force_refresh : bool
        Bypass the on-disk cache.
    progress_callback : Callable | None
        ``(step, total, status)`` — invoked before each per-event fetch.

    Returns
    -------
    list of dicts. Each dict has keys::

        slug                str
        name                str         (event_name from infobox; falls back to slug title)
        url                 str
        tier                str         ("S-Tier" | "A-Tier" | "B-Tier" | …)
        start_date          datetime
        end_date            datetime
        prize_pool          float       (USD; 0.0 when unknown / not yet announced)
        is_lan              bool
        format              str         (e.g. "Single Elimination", "Swiss"; "" if unknown)
        prize_distribution  dict[str, float]   (place label → USD; e.g. {"1st": 250000})
        seeded_teams        list[str]   (canonical team names, in page order)

    Sorted by start date ascending, then title.
    """
    if today is None:
        today = datetime.now()
    today_floor = today.replace(hour=0, minute=0, second=0, microsecond=0)

    cache_path = _upcoming_cache_path(start_date, end_date, min_tier)
    if not force_refresh:
        cached = _load_upcoming_cache(cache_path)
        if cached is not None:
            return [
                rec for rec in cached
                if rec.get("end_date") and rec["end_date"] >= today_floor
            ]

    portal_events = discover_from_portal(
        start_date=start_date,
        end_date=end_date,
        min_tier=min_tier,
        include_qualifiers=False,
    )

    upcoming = [
        ev for ev in portal_events
        if ev["end_date"] is not None and ev["end_date"] >= today_floor
    ]

    if not fetch_details:
        results = [{
            **ev,
            "name":               ev["title"],
            "prize_pool":         0.0,
            "is_lan":             False,
            "format":             "",
            "prize_distribution": {},
            "seeded_teams":       [],
        } for ev in upcoming]
        _save_upcoming_cache(cache_path, results)
        return results

    results: list[dict] = []
    total = len(upcoming)
    for i, ev in enumerate(upcoming):
        if progress_callback:
            progress_callback(i + 1, total, f"Inspecting {ev['slug']} ({i + 1}/{total})…")

        html = _fetch_page(ev["slug"])
        if html is None:
            logger.warning("Skipping upcoming event (page fetch failed): %s", ev["slug"])
        else:
            soup = _bs4(html)
            meta = _parse_infobox(soup)
            fmt  = _parse_format(soup)
            prize_dist   = _parse_prize_distribution_by_place(soup)
            seeded_teams = _parse_seeded_teams(soup)

            # Prefer infobox dates when present (more precise than portal range)
            start_dt = meta["start_date"] or ev["start_date"]
            end_dt   = meta["end_date"]   or ev["end_date"]
            name     = meta["event_name"] or ev["title"]
            prize_pool = meta["prize_pool"]
            is_lan     = meta["is_lan"]

            results.append({
                "slug":               ev["slug"],
                "name":               name,
                "url":                ev["url"],
                "tier":               ev["tier"],
                "start_date":         start_dt,
                "end_date":           end_dt,
                "prize_pool":         prize_pool,
                "is_lan":             is_lan,
                "format":             fmt,
                "prize_distribution": prize_dist,
                "seeded_teams":       seeded_teams,
            })

        if i < total - 1:
            time.sleep(_REQ_DELAY)

    if progress_callback:
        progress_callback(total, total, "Done.")

    results.sort(key=lambda x: (x["start_date"] or datetime.max, x["name"]))
    _save_upcoming_cache(cache_path, results)
    logger.info(
        "Upcoming discovery: %d events in %s – %s (today=%s)",
        len(results), start_date, end_date, today_floor.strftime("%Y-%m-%d"),
    )
    return results
