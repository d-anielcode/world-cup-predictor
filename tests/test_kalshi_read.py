from touchline.data.kalshi_read import parse_price, market_to_quote, KalshiReadClient


def test_parse_fixed_point_price_string():
    assert parse_price("0.6500") == 0.65
    assert parse_price("0") == 0.0
    assert parse_price(None) == 0.0


def test_market_to_quote_maps_fields():
    raw = {
        "ticker": "KXWC-26-USAWAL-USA",
        "title": "Will USA beat Wales?",
        "yes_bid": "0.5500", "yes_ask": "0.6100",
        "status": "active",
    }
    q = market_to_quote(raw, series_ticker="KXWC")
    assert q.ticker == "KXWC-26-USAWAL-USA"
    assert q.series_ticker == "KXWC"
    assert 0.5 < q.yes_price < 0.65   # midpoint of bid/ask
    assert q.no_price == round(1 - q.yes_price, 4)
    assert q.status == "active"


def test_market_to_quote_uses_available_side_when_one_sided():
    raw = {"ticker": "T", "yes_bid": "0", "yes_ask": "0.5500", "status": "active"}
    q = market_to_quote(raw, series_ticker="KXWC")
    assert q.yes_price == 0.55          # not 0.275 (no phantom zero bid)
    assert q.no_price == 0.45


def test_market_to_quote_zero_when_no_quotes():
    raw = {"ticker": "T", "status": "active"}
    q = market_to_quote(raw, series_ticker="KXWC")
    assert q.yes_price == 0.0


def test_headers_without_key_raises_clear_error():
    client = KalshiReadClient()  # default config has empty key path -> no key loaded
    try:
        import pytest
        with pytest.raises(RuntimeError, match="private key not loaded"):
            client._headers("GET", "/markets")
    finally:
        client.close()
