from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("TOUCHLINE_DATA_DIR", "touchline_data")).resolve()
DB_PATH = DATA_DIR / "touchline.db"
CACHE_DIR = DATA_DIR / "cache"

# Kalshi (read-only). Reused from EdgeRunner env conventions.
KALSHI_BASE_URL = os.environ.get(
    "KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"
)
KALSHI_API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = Path(
    os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
)
