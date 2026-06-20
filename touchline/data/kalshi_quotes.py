from __future__ import annotations

import re

from touchline.data.kalshi_read import KalshiReadClient
from touchline.data.teams import canonical_team
from touchline.edge.quotes import MarketQuoteRow

# Live Kalshi World Cup per-match series. Markets for one match share an event
# ticker suffix (e.g. "26JUN19BRAHTI"), so the team names parsed from the GAME
# (match-winner) titles are joined to the total/spread/BTTS markets by suffix.
# These four series are intentionally hardcoded (not a caller argument): they are
# the confirmed WC match series, and supplying a non-WC series would silently
# break the suffix-join. Adding a market type is a deliberate code change here.
GAME_SERIES = "KXWCGAME"

_GAME_TITLE = re.compile(r"^(?P<home>.+?)\s+vs\.?\s+(?P<away>.+?)\s+Winner", re.IGNORECASE)
_OVER = re.compile(r"over\s+(?P<line>\d+(?:\.\d+)?)\s+goals", re.IGNORECASE)
_SPREAD = re.compile(r"^(?P<team>.+?)\s+wins by more than\s+(?P<n>\d+(?:\.\d+)?)", re.IGNORECASE)
_DRAW = {"tie", "draw"}


def _mid(raw: dict) -> float:
    """Midpoint of Kalshi yes bid/ask (fixed-point dollars) as a probability in [0,1]."""
    b = float(raw.get("yes_bid_dollars") or 0)
    a = float(raw.get("yes_ask_dollars") or 0)
    return round((b + a) / 2, 4) if (a or b) else 0.0


def _suffix(event_ticker: str) -> str:
    """The match key shared across a match's markets, e.g. '26JUN19BRAHTI'."""
    return event_ticker.split("-", 1)[1] if "-" in event_ticker else event_ticker


def parse_game_market(raw: dict) -> MarketQuoteRow | None:
    """KXWCGAME market -> 1X2 MarketQuoteRow (teams from the title)."""
    m = _GAME_TITLE.match(raw.get("title", "") or "")
    sub = (raw.get("yes_sub_title") or "").strip()
    if not m or not sub:
        return None
    home, away = canonical_team(m.group("home")), canonical_team(m.group("away"))
    if sub.lower() in _DRAW:
        side = "draw"
    elif canonical_team(sub) == home:
        side = "home"
    elif canonical_team(sub) == away:
        side = "away"
    else:
        return None
    return MarketQuoteRow(home, away, "1x2", side, None, _mid(raw), raw.get("ticker", ""))


def parse_total_market(raw: dict, home: str, away: str) -> MarketQuoteRow | None:
    """KXWCTOTAL 'Will over X.5 goals be scored?' -> total/over MarketQuoteRow."""
    m = _OVER.search(raw.get("title", "") or "")
    if not m:
        return None
    return MarketQuoteRow(home, away, "total", "over", float(m.group("line")),
                          _mid(raw), raw.get("ticker", ""))


def parse_spread_market(raw: dict, home: str, away: str) -> MarketQuoteRow | None:
    """KXWCSPREAD '<team> wins by more than N goals?' -> handicap MarketQuoteRow.

    Home team: P(margin > N) == home_handicap[-N] -> (side='home', line=-N).
    Away team: P(margin < -N) == 1 - home_handicap[N] -> (side='away', line=+N)."""
    m = _SPREAD.match(raw.get("title", "") or "")
    if not m:
        return None
    team = canonical_team(m.group("team"))
    n = float(m.group("n"))
    if team == home:
        side, line = "home", -n
    elif team == away:
        side, line = "away", n
    else:
        return None
    return MarketQuoteRow(home, away, "handicap", side, line, _mid(raw), raw.get("ticker", ""))


def parse_btts_market(raw: dict, home: str, away: str) -> MarketQuoteRow | None:
    """KXWCBTTS 'Will both teams score?' -> btts/yes MarketQuoteRow."""
    if "both teams" not in (raw.get("title", "") or "").lower():
        return None
    return MarketQuoteRow(home, away, "btts", "yes", None, _mid(raw), raw.get("ticker", ""))


def fetch_quotes(client: KalshiReadClient | None = None) -> list[MarketQuoteRow]:
    """Fetch live Kalshi WC markets across the match series and parse to MarketQuoteRows.

    Markets that don't parse, lack a known match, or have no price are skipped."""
    owns = client is None
    client = client or KalshiReadClient()
    rows: list[MarketQuoteRow] = []
    teams_by_suffix: dict[str, tuple[str, str]] = {}
    try:
        for raw in client.get_markets(GAME_SERIES):
            m = _GAME_TITLE.match(raw.get("title", "") or "")
            if m:
                teams_by_suffix[_suffix(raw.get("event_ticker", ""))] = (
                    canonical_team(m.group("home")), canonical_team(m.group("away")))
            q = parse_game_market(raw)
            if q is not None and q.price > 0:
                rows.append(q)
        parsers = {"KXWCTOTAL": parse_total_market, "KXWCSPREAD": parse_spread_market,
                   "KXWCBTTS": parse_btts_market}
        for series, parser in parsers.items():
            for raw in client.get_markets(series):
                teams = teams_by_suffix.get(_suffix(raw.get("event_ticker", "")))
                if teams is None:
                    continue
                q = parser(raw, *teams)
                if q is not None and q.price > 0:
                    rows.append(q)
    finally:
        if owns:
            client.close()
    return rows
