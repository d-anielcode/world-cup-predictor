from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MarketQuoteRow:
    home: str
    away: str
    market_type: str          # "1x2" | "total" | "btts" | "handicap"
    side: str                 # see conventions
    line: float | None
    price: float              # market-implied probability of `side`, in [0,1]
    ticker: str = ""


def _parse_line(value: str) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def load_quotes(path: Path) -> list[MarketQuoteRow]:
    rows: list[MarketQuoteRow] = []
    with Path(path).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(MarketQuoteRow(
                home=r["home"].strip(), away=r["away"].strip(),
                market_type=r["market_type"].strip(), side=r["side"].strip(),
                line=_parse_line(r.get("line", "")), price=float(r["price"]),
                ticker=r.get("ticker", "").strip(),
            ))
    return rows


def fixture_lines(
    rows: list[MarketQuoteRow], home: str, away: str
) -> tuple[list[float], list[float]]:
    """Distinct total and handicap lines quoted for a fixture (sorted)."""
    totals, handicaps = set(), set()
    for r in rows:
        if r.home == home and r.away == away and r.line is not None:
            if r.market_type == "total":
                totals.add(r.line)
            elif r.market_type == "handicap":
                handicaps.add(r.line)
    return sorted(totals), sorted(handicaps)
