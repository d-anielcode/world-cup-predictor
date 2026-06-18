from __future__ import annotations

from touchline.data.kalshi_read import KalshiReadClient
from touchline.edge.quotes import MarketQuoteRow


def fetch_quotes(series_ticker: str) -> list[MarketQuoteRow]:
    """Fetch live Kalshi World Cup markets and map them to MarketQuoteRows.

    NOTE: the exact Kalshi WC market ticker/title schema is not yet confirmed. This
    adapter is intentionally thin and isolated; until the live series is verified the
    CSV quotes path is authoritative. Parsing real markets into (home, away,
    market_type, side, line) requires reading the live `title`/`subtitle` fields and
    must be finalized against real data before relying on this path.
    """
    client = KalshiReadClient()
    try:
        markets = client.get_markets(series_ticker)
    finally:
        client.close()
    raise NotImplementedError(
        f"Kalshi WC market parsing not yet verified ({len(markets)} markets "
        f"fetched). Use the --quotes CSV path until the live schema is confirmed."
    )
