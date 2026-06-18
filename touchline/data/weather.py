from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def estimate_wbgt(temp_c: float, rh_pct: float) -> float:
    """Outdoor WBGT (°C) from air temperature and relative humidity (BoM approximation)."""
    e = (rh_pct / 100.0) * 6.105 * math.exp(17.27 * temp_c / (237.7 + temp_c))
    return 0.567 * temp_c + 0.393 * e + 3.94


def wbgt_from_payload(payload: dict, kickoff: datetime) -> float | None:
    """Estimate WBGT at the hour nearest kickoff from an Open-Meteo hourly payload."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    rhs = hourly.get("relative_humidity_2m", [])
    if not times:
        return None
    parsed = [datetime.fromisoformat(t) for t in times]
    i = min(range(len(parsed)), key=lambda k: abs((parsed[k] - kickoff).total_seconds()))
    return estimate_wbgt(float(temps[i]), float(rhs[i]))


def fetch_wbgt(
    lat: float, lon: float, kickoff: datetime, cache_dir: Path,
    client: httpx.Client | None = None,
) -> float | None:
    """Fetch the Open-Meteo forecast for the kickoff day and estimate WBGT (cached)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    day = kickoff.date().isoformat()
    cache_file = cache_dir / f"openmeteo_{lat:.2f}_{lon:.2f}_{day}.json"
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        owns = client is None
        client = client or httpx.Client(timeout=15.0)
        try:
            resp = client.get(OPEN_METEO_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m",
                "start_date": day, "end_date": day,
            })
            resp.raise_for_status()
            payload = resp.json()
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        finally:
            if owns:
                client.close()
    return wbgt_from_payload(payload, kickoff)
