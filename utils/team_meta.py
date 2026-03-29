"""
Team Metadata

Known team flags, colors, and regional information.
"""

KNOWN_META: dict[str, dict] = {
    "Vitality":       {"flag": "🇫🇷", "color": "#F5A623"},
    "PARIVISION":     {"flag": "🇷🇺", "color": "#E91E63"},
    "Natus Vincere":  {"flag": "🇺🇦", "color": "#FFD600"},
    "Spirit":         {"flag": "🇷🇺", "color": "#7B68EE"},
    "MOUZ":           {"flag": "🇩🇪", "color": "#E53935"},
    "FaZe":           {"flag": "🌍",  "color": "#EC407A"},
    "Aurora":         {"flag": "🇷🇺", "color": "#00BCD4"},
    "G2":             {"flag": "🇪🇸", "color": "#F44336"},
    "3DMAX":          {"flag": "🇫🇷", "color": "#607D8B"},
    "Astralis":       {"flag": "🇩🇰", "color": "#1565C0"},
    "Falcons":        {"flag": "🇸🇦", "color": "#0288D1"},
    "FURIA":          {"flag": "🇧🇷", "color": "#F5A623"},
    "Liquid":         {"flag": "🇺🇸", "color": "#00B0FF"},
    "NRG":            {"flag": "🇺🇸", "color": "#FF5722"},
    "paiN":           {"flag": "🇧🇷", "color": "#D32F2F"},
    "The MongolZ":    {"flag": "🇲🇳", "color": "#FF6F00"},
    "TYLOO":          {"flag": "🇨🇳", "color": "#C62828"},
    "Rare Atom":      {"flag": "🇨🇳", "color": "#00897B"},
    "GamerLegion":    {"flag": "🇩🇰", "color": "#9C27B0"},
    "FUT":            {"flag": "🇹🇷", "color": "#FF9800"},
    "B8":             {"flag": "🇺🇦", "color": "#4CAF50"},
    "Gentle Mates":   {"flag": "🇪🇸", "color": "#AB47BC"},
    "HEROIC":         {"flag": "🇩🇰", "color": "#FF7043"},
    "Monte":          {"flag": "🇺🇦", "color": "#26C6DA"},
    "BetBoom":        {"flag": "🇷🇺", "color": "#EF5350"},
    "MIBR":           {"flag": "🇧🇷", "color": "#43A047"},
    "9z":             {"flag": "🇦🇷", "color": "#1976D2"},
    "BC.Game":        {"flag": "🌍",  "color": "#7E57C2"},
    "Virtus.pro":     {"flag": "🇷🇺", "color": "#FF8F00"},
}

COLOR_CYCLE = [
    "#58a6ff", "#3fb950", "#f0b429", "#f85149", "#79c0ff",
    "#56d364", "#e3b341", "#ff7b72", "#bc8cff", "#39c5cf",
]


class TeamMetaManager:
    """Manages team metadata with fallback colors."""

    def __init__(self):
        self._color_idx = 0

    def get_team_meta(self, team: str, region: str = "Global") -> dict:
        """Return {flag, color, region} for any team name."""
        known = KNOWN_META.get(team, {})
        flag  = known.get("flag", "🌍")
        color = known.get("color", COLOR_CYCLE[self._color_idx % len(COLOR_CYCLE)])
        if team not in KNOWN_META:
            self._color_idx += 1
        return {"flag": flag, "color": color, "region": region}


_manager = TeamMetaManager()


def get_team_meta(team: str, region: str = "Global") -> dict:
    """Return {flag, color, region} for any team name."""
    return _manager.get_team_meta(team, region)
