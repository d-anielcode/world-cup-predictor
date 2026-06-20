from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import httpx

from touchline.models import Match

# football-data.co.uk publishes one CSV per league-season with results AND bookmaker
# closing odds — ideal for a model-vs-market backtest on CLUB leagues (our Dixon-Coles
# engine is team-agnostic; it just needs results). E0 = English Premier League.
BASE_URL = "https://www.football-data.co.uk/mmz4281"


def _f(value: str | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_csv(text: str, competition: str) -> list[tuple[Match, tuple[float, float, float]]]:
    """Parse a football-data.co.uk league CSV into (Match, closing-odds) pairs.

    Uses the market-average closing odds (AvgCH/CD/CA), falling back to Bet365 closing
    (B365CH/CD/CA). Rows without a full set of closing odds or final goals are skipped
    (unplayed/postponed). Odds are decimal (home, draw, away)."""
    out: list[tuple[Match, tuple[float, float, float]]] = []
    for r in csv.DictReader(io.StringIO(text)):
        hg, ag = _f(r.get("FTHG")), _f(r.get("FTAG"))
        oh = _f(r.get("AvgCH")) or _f(r.get("B365CH"))
        od = _f(r.get("AvgCD")) or _f(r.get("B365CD"))
        oa = _f(r.get("AvgCA")) or _f(r.get("B365CA"))
        home, away = (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
        if hg is None or ag is None or not home or not away:
            continue
        if not (oh and od and oa):
            continue
        try:
            d = datetime.strptime(r["Date"].strip(), "%d/%m/%Y").date()
        except (KeyError, ValueError):
            try:
                d = datetime.strptime(r["Date"].strip(), "%d/%m/%y").date()
            except (KeyError, ValueError):
                continue
        out.append((
            Match(match_date=d, home_team=home, away_team=away,
                  home_goals=int(hg), away_goals=int(ag), competition=competition,
                  stage=None, venue=None, played=True, source="footballdata"),
            (oh, od, oa),
        ))
    return out


def fetch_season(
    league: str, season: str, cache_dir: Path, ttl_seconds: int = 86400,
    client: httpx.Client | None = None,
) -> list[tuple[Match, tuple[float, float, float]]]:
    """Fetch one league-season CSV (e.g. league='E0', season='2526') with a TTL cache.

    `season` is football-data's 4-digit code: '2526' = 2025/26."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"fd_{league}_{season}.csv"
    import time
    fresh = cache_file.is_file() and (time.time() - cache_file.stat().st_mtime) < ttl_seconds
    if fresh:
        text = cache_file.read_text(encoding="utf-8-sig")
    else:
        owns = client is None
        client = client or httpx.Client(timeout=30.0, follow_redirects=True)
        try:
            resp = client.get(f"{BASE_URL}/{season}/{league}.csv")
            resp.raise_for_status()
            text = resp.text
            cache_file.write_text(text, encoding="utf-8")
        finally:
            if owns:
                client.close()
    return parse_csv(text, competition=league)
