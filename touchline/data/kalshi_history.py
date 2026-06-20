from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from touchline.data.kalshi_quotes import GAME_SERIES, parse_game_market, _suffix

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


def capture_1x2_history(
    client, kickoff_lookup: dict, lookback_days: int = 45, period_interval: int = 60,
) -> list[dict]:
    """Recover each settled KXWCGAME (match-winner) contract's pre-kickoff price.

    `kickoff_lookup` maps (date, frozenset({home, away})) -> kickoff datetime (UTC),
    built from the match DB. For every settled contract we decode the fixture, find
    its kickoff, pull candlesticks up to kickoff, and take the closing mid. Returns
    one record per contract: home, away, side, market_price, result, kickoff_ts."""
    out: list[dict] = []
    for m in client.get_markets(GAME_SERIES, status="settled"):
        q = parse_game_market(m)
        if q is None:
            continue
        suffix = _suffix(m.get("event_ticker", ""))
        d = decode_suffix_date(suffix)
        ko = kickoff_lookup.get((d, frozenset((q.home, q.away)))) if d else None
        if ko is None:
            continue
        ko_ts = int(ko.timestamp())
        candles = client.get_candlesticks(
            GAME_SERIES, m["ticker"], ko_ts - lookback_days * 86400, ko_ts,
            period_interval)
        price = pre_kickoff_price(candles, ko_ts)
        if price is None or price <= 0:
            continue
        out.append({
            "date": d.isoformat(), "home": q.home, "away": q.away,
            "market_type": "1x2", "side": q.side, "line": None,
            "kickoff_ts": ko_ts, "market_price": price,
            "result": m.get("result"), "ticker": m.get("ticker", ""),
        })
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
