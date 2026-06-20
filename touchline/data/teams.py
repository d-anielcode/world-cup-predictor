from __future__ import annotations

# Cross-source team-name normalization. openfootball and worldcupjson spell some
# national teams differently (e.g. "USA" vs "United States"). We map every known
# variant to a single canonical spelling so the same physical match produces the
# same natural_key from both sources and is not double-counted.

def _norm(name: str) -> str:
    # Normalize "&" -> "and" so ampersand spellings (openfootball's
    # "Bosnia & Herzegovina") collapse onto the same alias as the hyphen/word forms.
    return " ".join(name.replace("&", " and ").strip().lower().split())


# variant (normalized) -> canonical display name
_ALIASES: dict[str, str] = {
    "united states": "USA",
    "usa": "USA",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "korea dpr": "North Korea",
    "north korea": "North Korea",
    "ir iran": "Iran",
    "iran": "Iran",
    "china pr": "China",
    "china": "China",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "bosnia and herzegovina": "Bosnia-Herzegovina",
    "bosnia-herzegovina": "Bosnia-Herzegovina",
    # Kalshi spellings that differ from the openfootball/results data:
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "curacao": "Curaçao",
}


def canonical_team(name: str) -> str:
    """Return the canonical spelling of a team name, or the trimmed input if unknown."""
    return _ALIASES.get(_norm(name), name.strip())
