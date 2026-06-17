from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class Match:
    match_date: date
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    competition: str
    stage: str | None
    venue: str | None
    played: bool
    source: str

    def natural_key(self) -> str:
        """Dedup key: date + both teams (home listing as recorded)."""
        return f"{self.match_date.isoformat()}|{self.home_team}|{self.away_team}"


@dataclass
class MarketQuote:
    ticker: str
    series_ticker: str
    title: str
    yes_price: float  # dollars 0..1
    no_price: float
    status: str
    raw: dict
