import json
from pathlib import Path

from touchline.data.kalshi_quotes import (
    _mid, _suffix, parse_game_market, parse_total_market, parse_spread_market,
    parse_btts_market,
)
from touchline.edge.quotes import MarketQuoteRow

SAMPLE = json.loads(
    (Path(__file__).parent / "fixtures" / "kalshi_markets_sample.json").read_text(encoding="utf-8")
)
GAME = [m for m in SAMPLE if m["ticker"].startswith("KXWCGAME") or "Winner" in m["title"]]
TOTAL = [m for m in SAMPLE if "goals be scored" in m["title"]][0]
SPREADS = [m for m in SAMPLE if "wins by more than" in m["title"]]
BTTS = [m for m in SAMPLE if m["title"] == "Will both teams score?"][0]


def test_mid_and_suffix():
    assert _mid({"yes_bid_dollars": "0.50", "yes_ask_dollars": "0.52"}) == 0.51
    assert _mid({"yes_bid_dollars": None, "yes_ask_dollars": None}) == 0.0
    assert _suffix("KXWCGAME-26JUN19BRAHTI") == "26JUN19BRAHTI"


def test_parse_game_1x2():
    brazil = next(m for m in GAME if m["yes_sub_title"] == "Brazil")
    tie = next(m for m in GAME if m["yes_sub_title"] == "Tie")
    haiti = next(m for m in GAME if m["yes_sub_title"] == "Haiti")
    qb = parse_game_market(brazil)
    assert isinstance(qb, MarketQuoteRow)
    assert qb.home == "Brazil" and qb.away == "Haiti"
    assert qb.market_type == "1x2" and qb.side == "home" and qb.price > 0.9
    assert parse_game_market(tie).side == "draw"
    assert parse_game_market(haiti).side == "away"


def test_parse_total_over():
    q = parse_total_market(TOTAL, "Brazil", "Haiti")
    assert q.market_type == "total" and q.side == "over" and q.line == 2.5
    assert q.home == "Brazil" and q.price > 0.5


def test_parse_spread_home_and_away():
    # "Brazil wins by more than 3.5" (home) -> P(margin > 3.5) = home_handicap[-3.5]
    home_spread = next(m for m in SPREADS if m["yes_sub_title"].startswith("Brazil wins by more than 3.5"))
    qh = parse_spread_market(home_spread, "Brazil", "Haiti")
    assert qh.market_type == "handicap" and qh.side == "home" and qh.line == -3.5
    # "Haiti wins by more than 1.5" (away) -> away side, line +1.5
    away_spread = next(m for m in SPREADS if m["yes_sub_title"].startswith("Haiti wins by more than 1.5"))
    qa = parse_spread_market(away_spread, "Brazil", "Haiti")
    assert qa.side == "away" and qa.line == 1.5


def test_parse_btts():
    q = parse_btts_market(BTTS, "Brazil", "Haiti")
    assert q.market_type == "btts" and q.side == "yes" and q.home == "Brazil"


def test_unparseable_returns_none():
    assert parse_game_market({"title": "Top scorer", "yes_sub_title": "Mbappe"}) is None
    assert parse_spread_market({"title": "weird"}, "A", "B") is None
