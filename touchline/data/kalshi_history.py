from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from touchline.data.kalshi_quotes import (
    GAME_SERIES, parse_game_market, parse_total_market, parse_spread_market,
    parse_btts_market, _suffix, _GAME_TITLE,
)
from touchline.data.teams import canonical_team

# Binary per-match series and their parsers (1X2/GAME is handled separately as the
# 3-way market and as the source of the per-suffix team names). Correct-score is
# intentionally excluded: 200+ illiquid cells per match, low value for vs-market.
_BINARY_PARSERS = {
    "KXWCTOTAL": parse_total_market,
    "KXWCSPREAD": parse_spread_market,
    "KXWCBTTS": parse_btts_market,
}

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_SUFFIX_DATE = re.compile(r"^(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})")


def decode_suffix_date(suffix: str) -> date | None:
    """Date encoded at the front of a Kalshi WC event suffix, e.g. '26JUN19...' ->
    2026-06-19. None if it doesn't match."""
    m = _SUFFIX_DATE.match(suffix or "")
    if not m or m.group("mon") not in _MONTHS:
        return None
    try:
        return date(2000 + int(m.group("yy")), _MONTHS[m.group("mon")], int(m.group("dd")))
    except ValueError:
        return None


def _close(candle: dict, side: str) -> float:
    v = (candle.get(side) or {}).get("close_dollars")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _candle_mid(candle: dict) -> float | None:
    bid, ask = _close(candle, "yes_bid"), _close(candle, "yes_ask")
    sides = [p for p in (bid, ask) if p > 0.0]
    return round(sum(sides) / len(sides), 4) if sides else None


def pre_kickoff_price(candles: list[dict], kickoff_ts: int) -> float | None:
    """Mid price of the last candle that closed at or before kickoff (the market's
    pre-match closing line). None if no pre-kickoff candle carries a quote."""
    before = [c for c in candles if int(c.get("end_period_ts", 0)) <= kickoff_ts]
    for c in reversed(before):
        mid = _candle_mid(c)
        if mid is not None:
            return mid
    return None


def capture_settled_history(
    client, kickoff_lookup: dict, lookback_days: int = 45, period_interval: int = 60,
) -> list[dict]:
    """Recover every settled per-match contract's pre-kickoff price + result.

    Covers 1X2 (KXWCGAME) plus totals/handicap/BTTS. `kickoff_lookup` maps
    (date, frozenset({home, away})) -> kickoff datetime (UTC), from the match DB.
    For each contract we decode the fixture, find its kickoff, pull candlesticks up
    to kickoff, and take the closing mid. Binary markets need the per-suffix team
    names, which come from the settled GAME titles. One record per contract."""
    game_settled = client.get_markets(GAME_SERIES, status="settled")
    teams_by_suffix: dict[str, tuple[str, str]] = {}
    for m in game_settled:
        gm = _GAME_TITLE.match(m.get("title", "") or "")
        if gm:
            teams_by_suffix[_suffix(m.get("event_ticker", ""))] = (
                canonical_team(gm.group("home")), canonical_team(gm.group("away")))

    out: list[dict] = []

    def emit(series: str, m: dict, q) -> None:
        d = decode_suffix_date(_suffix(m.get("event_ticker", "")))
        ko = kickoff_lookup.get((d, frozenset((q.home, q.away)))) if d else None
        if ko is None:
            return
        ko_ts = int(ko.timestamp())
        candles = client.get_candlesticks(
            series, m["ticker"], ko_ts - lookback_days * 86400, ko_ts, period_interval)
        price = pre_kickoff_price(candles, ko_ts)
        if price is None or price <= 0:
            return
        out.append({
            "date": d.isoformat(), "home": q.home, "away": q.away,
            "market_type": q.market_type, "side": q.side, "line": q.line,
            "kickoff_ts": ko_ts, "market_price": price,
            "result": m.get("result"), "ticker": m.get("ticker", ""),
        })

    for m in game_settled:
        q = parse_game_market(m)
        if q is not None:
            emit(GAME_SERIES, m, q)
    for series, parser in _BINARY_PARSERS.items():
        for m in client.get_markets(series, status="settled"):
            teams = teams_by_suffix.get(_suffix(m.get("event_ticker", "")))
            if teams is None:
                continue
            q = parser(m, *teams)
            if q is not None:
                emit(series, m, q)
    return out


def write_jsonl(records: list[dict], path: Path) -> int:
    """Append records to a JSONL price-history store, de-duplicated by ticker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {r["ticker"]: r for r in read_jsonl(path)}
    for r in records:
        existing[r["ticker"]] = r
    with path.open("w", encoding="utf-8") as f:
        for r in existing.values():
            f.write(json.dumps(r) + "\n")
    return len(existing)


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
