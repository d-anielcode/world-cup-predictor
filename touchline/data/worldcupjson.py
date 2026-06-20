from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import httpx
from dateutil import parser as dateparser

from touchline.data.teams import canonical_team
from touchline.models import Match

BASE_URL = "https://worldcupjson.net"
_COMPLETED = {"completed", "finished", "full-time"}


def parse_matches(payload: list[dict], competition: str | None = None) -> list[Match]:
    """Parse worldcupjson /matches payload into Match records.

    Team names are normalized to canonical spellings so they align with
    openfootball (e.g. "United States" -> "USA"). The competition label is
    derived from each match's own year unless an explicit label is given —
    this guards against the API serving a prior tournament's data (it has been
    observed returning 2022 results) being mislabeled as the current one.
    """
    out: list[Match] = []
    for m in payload:
        home = m["home_team"]
        away = m["away_team"]
        played = str(m.get("status", "")).lower() in _COMPLETED
        match_date = dateparser.parse(m["datetime"]).date()
        out.append(
            Match(
                match_date=match_date,
                home_team=canonical_team(home["name"]),
                away_team=canonical_team(away["name"]),
                home_goals=home["goals"] if played else None,
                away_goals=away["goals"] if played else None,
                competition=competition or f"World Cup {match_date.year}",
                stage=m.get("stage_name"),
                venue=m.get("venue"),
                played=played,
                source="worldcupjson",
            )
        )
    return out


def fetch_matches(
    cache_dir: Path,
    competition: str | None = None,
    ttl_seconds: int = 300,
    client: httpx.Client | None = None,
) -> list[Match]:
    """Fetch /matches with a simple on-disk TTL cache (respects 10 req/60s limit)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "worldcupjson_matches.json"

    fresh = cache_file.is_file() and (time.time() - cache_file.stat().st_mtime) < ttl_seconds
    if fresh:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        owns = client is None
        client = client or httpx.Client(timeout=15.0)
        try:
            resp = client.get(f"{BASE_URL}/matches")
            resp.raise_for_status()
            payload = resp.json()
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except (httpx.HTTPError, ValueError) as e:
            # worldcupjson.net is an optional, sometimes-dead feed (404s observed in
            # 2026). It must never crash the pipeline — fall back to a stale cache if
            # present, else return no live matches (openfootball is the primary source).
            warnings.warn(f"worldcupjson fetch failed ({e!r}); skipping live feed")
            if cache_file.is_file():
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                return []
        finally:
            if owns:
                client.close()
    return parse_matches(payload, competition=competition)
