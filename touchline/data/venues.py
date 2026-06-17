from __future__ import annotations

import math
from dataclasses import dataclass

HOST_COUNTRIES = {"USA", "Canada", "Mexico"}


@dataclass(frozen=True)
class Venue:
    name: str
    city: str
    country: str
    lat: float
    lon: float
    altitude_m: int
    has_roof_ac: bool
    tz: str


# 2026 FIFA World Cup venues (16 stadiums). Altitude/roof verified from public data.
VENUES: dict[str, Venue] = {
    "MetLife Stadium": Venue("MetLife Stadium", "New York/NJ", "USA", 40.813, -74.074, 7, False, "America/New_York"),
    "AT&T Stadium": Venue("AT&T Stadium", "Dallas", "USA", 32.747, -97.093, 150, True, "America/Chicago"),
    "NRG Stadium": Venue("NRG Stadium", "Houston", "USA", 29.685, -95.411, 15, True, "America/Chicago"),
    "Mercedes-Benz Stadium": Venue("Mercedes-Benz Stadium", "Atlanta", "USA", 33.755, -84.401, 320, True, "America/New_York"),
    "Hard Rock Stadium": Venue("Hard Rock Stadium", "Miami", "USA", 25.958, -80.239, 2, False, "America/New_York"),
    "Arrowhead Stadium": Venue("Arrowhead Stadium", "Kansas City", "USA", 39.049, -94.484, 270, False, "America/Chicago"),
    "Lincoln Financial Field": Venue("Lincoln Financial Field", "Philadelphia", "USA", 39.901, -75.168, 12, False, "America/New_York"),
    "Levi's Stadium": Venue("Levi's Stadium", "San Francisco Bay", "USA", 37.403, -121.970, 5, False, "America/Los_Angeles"),
    "SoFi Stadium": Venue("SoFi Stadium", "Los Angeles", "USA", 33.953, -118.339, 30, True, "America/Los_Angeles"),
    "Lumen Field": Venue("Lumen Field", "Seattle", "USA", 47.595, -122.332, 3, False, "America/Los_Angeles"),
    "Gillette Stadium": Venue("Gillette Stadium", "Boston", "USA", 42.091, -71.264, 90, False, "America/New_York"),
    "BMO Field": Venue("BMO Field", "Toronto", "Canada", 43.633, -79.418, 80, False, "America/Toronto"),
    "BC Place": Venue("BC Place", "Vancouver", "Canada", 49.277, -123.112, 3, True, "America/Vancouver"),
    "Estadio Azteca": Venue("Estadio Azteca", "Mexico City", "Mexico", 19.303, -99.150, 2240, False, "America/Mexico_City"),
    "Estadio Akron": Venue("Estadio Akron", "Guadalajara", "Mexico", 20.681, -103.463, 1560, False, "America/Mexico_City"),
    "Estadio BBVA": Venue("Estadio BBVA", "Monterrey", "Mexico", 25.669, -100.244, 500, False, "America/Monterrey"),
}


def get_venue(name: str) -> Venue:
    """Look up a venue by exact or substring match on the canonical name."""
    if name in VENUES:
        return VENUES[name]
    for key, v in VENUES.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return v
    raise KeyError(f"Unknown venue: {name!r}")


def is_host_country(country: str) -> bool:
    return country in HOST_COUNTRIES


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
