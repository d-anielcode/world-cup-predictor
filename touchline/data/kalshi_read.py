from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from touchline import config
from touchline.models import MarketQuote


def parse_price(value: str | None) -> float:
    """Kalshi prices are fixed-point dollar strings like '0.6500'."""
    if not value:
        return 0.0
    return float(value)


def market_to_quote(raw: dict, series_ticker: str) -> MarketQuote:
    yes_bid = parse_price(raw.get("yes_bid"))
    yes_ask = parse_price(raw.get("yes_ask"))
    sides = [p for p in (yes_bid, yes_ask) if p > 0.0]
    yes_price = round(sum(sides) / len(sides), 4) if sides else 0.0
    return MarketQuote(
        ticker=raw["ticker"],
        series_ticker=series_ticker,
        title=raw.get("title", ""),
        yes_price=yes_price,
        no_price=round(1 - yes_price, 4),
        status=raw.get("status", ""),
        raw=raw,
    )


class KalshiReadClient:
    """Read-only Kalshi REST client. Ported from EdgeRunner (sync). No order methods."""

    def __init__(self) -> None:
        self._base_url = config.KALSHI_BASE_URL.rstrip("/")
        self._api_key_id = config.KALSHI_API_KEY_ID
        self._key_path = Path(config.KALSHI_PRIVATE_KEY_PATH)
        self._client = httpx.Client(timeout=15.0)
        self._private_key = None
        if self._key_path.is_file():
            self._private_key = serialization.load_pem_private_key(
                self._key_path.read_bytes(), password=None
            )

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        path_without_query = path.split("?", 1)[0]
        base_path = urlparse(self._base_url).path
        message = (timestamp_ms + method.upper() + base_path + path_without_query).encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str) -> dict:
        if self._private_key is None:
            raise RuntimeError(
                "Kalshi private key not loaded — set KALSHI_PRIVATE_KEY_PATH."
            )
        ts = str(int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
            "Accept": "application/json",
        }

    def _get(self, path: str) -> dict:
        resp = self._client.get(
            f"{self._base_url}{path}", headers=self._headers("GET", path)
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}").get("market", {})

    def get_candlesticks(
        self, series_ticker: str, ticker: str, start_ts: int, end_ts: int,
        period_interval: int = 60,
    ) -> list[dict]:
        """Historical OHLC candlesticks for a market (read-only). Each candle has
        end_period_ts plus yes_bid/yes_ask {open,high,low,close}_dollars. Used to
        recover a settled market's pre-kickoff price for the vs-market backtest."""
        path = (f"/series/{series_ticker}/markets/{ticker}/candlesticks"
                f"?start_ts={start_ts}&end_ts={end_ts}&period_interval={period_interval}")
        return self._get(path).get("candlesticks", [])

    def get_markets(self, series_ticker: str, status: str = "open", limit: int = 100) -> list[dict]:
        """Page through all markets for a series; returns raw market dicts."""
        markets: list[dict] = []
        cursor: str | None = None
        while True:
            path = f"/markets?status={status}&limit={limit}&series_ticker={series_ticker}"
            if cursor:
                path += f"&cursor={cursor}"
            data = self._get(path)
            markets.extend(data.get("markets", []))
            cursor = data.get("cursor") or None
            if not cursor:
                break
        return markets

    def quotes_for_series(self, series_ticker: str) -> list[MarketQuote]:
        return [market_to_quote(m, series_ticker) for m in self.get_markets(series_ticker)]

    def __enter__(self) -> "KalshiReadClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()
