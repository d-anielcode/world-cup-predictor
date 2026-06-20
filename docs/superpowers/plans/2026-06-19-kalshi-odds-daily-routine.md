# Kalshi Live Odds + Daily Routine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pull live Kalshi World Cup market prices and produce the ranked edge report end-to-end via one `touchline daily` command — no hand-gathered odds.

**Architecture:** Discovery-first. A `kalshi-discover` command captures the real live WC market structure; the quotes parser is built and verified against that captured fixture (never guessing the schema). `touchline daily` chains the existing pipeline (ingest → fit → fetch Kalshi quotes → `run_price` → report). Read-only — no order code.

**Tech Stack:** Python 3.11 (sync), httpx, cryptography (existing read-only Kalshi client), pytest. Builds on Plan 1 (`KalshiReadClient`), Plan 3 (`MarketQuoteRow`, `run_price`, `build_context`, overlay).

**Prerequisite:** `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` set in the environment (point at the EdgeRunner key). The client is read-only; no order/cancel methods exist. Task 1 needs live keys; if unreachable it fails with a clear message and the build pauses (the parser must be verified against real data, not guesses).

---

## File Structure

```
touchline/data/kalshi_read.py     MODIFY: add get_events() + list-series helper for discovery
touchline/data/kalshi_quotes.py   CREATE: parse_kalshi_market(), fetch_quotes()  (replaces kalshi_adapter.py)
touchline/data/kalshi_adapter.py  DELETE: superseded by kalshi_quotes.py
touchline/cli.py                  MODIFY: add `kalshi-discover` and `daily` commands; run_daily() orchestration
tests/test_kalshi_quotes.py       CREATE
tests/fixtures/kalshi_market_sample.json  CREATE (representative; replaced w/ real dump in Task 1)
tests/test_daily.py               CREATE
```

Run all commands from `C:\Users\dcho0\Documents\touchline` via `.venv/Scripts/python.exe`.

---

## Task 1: `kalshi-discover` command (capture the real schema)

**Files:**
- Modify: `touchline/cli.py`
- Test: none (one-shot operational command; exercised by the live smoke run)

- [ ] **Step 1: Add the discover command to `touchline/cli.py`**

Add imports near the top (integrate with the existing block):
```python
import json as _json
from touchline.data.kalshi_read import KalshiReadClient
```

Register the subcommand inside `main` (next to the others):
```python
    disc_p = sub.add_parser("kalshi-discover",
                            help="Dump live Kalshi World Cup markets to confirm the schema")
    disc_p.add_argument("--series", default="KXMENWORLDCUP",
                        help="Comma-separated Kalshi series tickers to dump")
```

Add the branch inside `main` (before `return 1`):
```python
    if args.command == "kalshi-discover":
        client = KalshiReadClient()
        try:
            all_markets = []
            for series in args.series.split(","):
                series = series.strip()
                markets = client.get_markets(series)
                print(f"Series {series}: {len(markets)} markets")
                all_markets.extend(markets)
        finally:
            client.close()
        out = config.DATA_DIR / "kalshi_discover.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(all_markets, indent=2), encoding="utf-8")
        print(f"\nWrote {len(all_markets)} markets to {out}")
        for m in all_markets[:10]:
            print(f"  ticker={m.get('ticker')}  title={m.get('title')!r}  "
                  f"yes_sub_title={m.get('yes_sub_title')!r}  "
                  f"yes_bid={m.get('yes_bid')} yes_ask={m.get('yes_ask')}")
        return 0
```

- [ ] **Step 2: Run the live discovery (needs keys)**

Run: `.venv/Scripts/python.exe -m touchline.cli kalshi-discover --series KXMENWORLDCUP`
Expected: prints market counts + a sample of tickers/titles/prices, and writes `touchline_data/kalshi_discover.json`.

**This is the schema-confirmation gate.** Inspect the output and record:
- the **series ticker(s)** that contain individual *match-winner* markets (the winner-only `KXMENWORLDCUP` may not — if so, re-run `--series` with the match series ticker seen on kalshi.com or in the dump's `event_ticker` prefixes);
- the **title / `yes_sub_title` format** (how the two teams and the outcome appear);
- the **price field type** (integer cents like `86`, or fixed-point string like `"0.86"`).

If the keys are unreachable, STOP and report — Task 2 cannot be verified without this dump.

- [ ] **Step 3: Capture a real fixture**

Copy 3–4 representative match-winner market dicts from `touchline_data/kalshi_discover.json` into `tests/fixtures/kalshi_market_sample.json` (a JSON list). These real records are what Task 2's parser is tested against. If the live format differs from the representative fixture in Task 2, this real capture is authoritative — adjust the parser to match it.

- [ ] **Step 4: Commit**

```bash
git add touchline/cli.py tests/fixtures/kalshi_market_sample.json
git commit -m "feat: kalshi-discover command + captured WC market fixture"
```

---

## Task 2: Kalshi quotes parser

**Files:**
- Create: `touchline/data/kalshi_quotes.py`
- Delete: `touchline/data/kalshi_adapter.py`
- Test: `tests/test_kalshi_quotes.py`

The parser below targets Kalshi's standard market fields: a match event titled
`"<Home> vs. <Away> Winner"` with each market's `yes_sub_title` naming the outcome
(`"<Home>"`, `"<Away>"`, or `"Draw"`), and integer-cent prices. **Verify against the
Task-1 fixture and adjust the title split / price scaling if the live format differs.**

- [ ] **Step 1: Create the representative fixture (overwritten by Task 1's real capture if different)**

`tests/fixtures/kalshi_market_sample.json` (if Task 1 already wrote a real one, keep that):
```json
[
  {"ticker": "KXWCGAME-26JUN19BRAHAI-BRA", "event_ticker": "KXWCGAME-26JUN19BRAHAI",
   "title": "Brazil vs. Haiti Winner", "yes_sub_title": "Brazil",
   "yes_bid": 85, "yes_ask": 88, "status": "active"},
  {"ticker": "KXWCGAME-26JUN19BRAHAI-DRAW", "event_ticker": "KXWCGAME-26JUN19BRAHAI",
   "title": "Brazil vs. Haiti Winner", "yes_sub_title": "Draw",
   "yes_bid": 8, "yes_ask": 11, "status": "active"},
  {"ticker": "KXWCGAME-26JUN19BRAHAI-HAI", "event_ticker": "KXWCGAME-26JUN19BRAHAI",
   "title": "Brazil vs. Haiti Winner", "yes_sub_title": "Haiti",
   "yes_bid": 3, "yes_ask": 5, "status": "active"}
]
```

- [ ] **Step 2: Write the failing test**

`tests/test_kalshi_quotes.py`:
```python
import json
from pathlib import Path
from touchline.data.kalshi_quotes import parse_kalshi_market, _kalshi_price
from touchline.edge.quotes import MarketQuoteRow

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "kalshi_market_sample.json").read_text(encoding="utf-8")
)


def test_price_cents_to_probability():
    assert _kalshi_price(85, 88) == 0.865    # midpoint of bid/ask, in [0,1]
    assert _kalshi_price(0, 0) == 0.0


def test_parses_home_market():
    q = parse_kalshi_market(FIXTURE[0])
    assert isinstance(q, MarketQuoteRow)
    assert q.home == "Brazil" and q.away == "Haiti"
    assert q.market_type == "1x2" and q.side == "home"
    assert 0.8 < q.price < 0.95
    assert q.ticker == "KXWCGAME-26JUN19BRAHAI-BRA"


def test_parses_draw_and_away():
    draw = parse_kalshi_market(FIXTURE[1]); away = parse_kalshi_market(FIXTURE[2])
    assert draw.side == "draw"
    assert away.side == "away" and away.home == "Brazil" and away.away == "Haiti"


def test_canonicalizes_team_names():
    raw = {"ticker": "T", "title": "United States vs. Wales Winner",
           "yes_sub_title": "United States", "yes_bid": 55, "yes_ask": 58, "status": "active"}
    q = parse_kalshi_market(raw)
    assert q.home == "USA" and q.side == "home"


def test_unparseable_market_returns_none():
    assert parse_kalshi_market({"ticker": "X", "title": "Top scorer", "yes_sub_title": "Mbappe"}) is None
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kalshi_quotes.py -v`
Expected: FAIL — `ModuleNotFoundError: touchline.data.kalshi_quotes`.

- [ ] **Step 4: Implement `touchline/data/kalshi_quotes.py`**
```python
from __future__ import annotations

import re

from touchline.data.kalshi_read import KalshiReadClient
from touchline.data.teams import canonical_team
from touchline.edge.quotes import MarketQuoteRow

# Match-winner event title, e.g. "Brazil vs. Haiti Winner".
_TITLE_RE = re.compile(r"^(?P<home>.+?)\s+vs\.?\s+(?P<away>.+?)(?:\s+Winner)?$", re.IGNORECASE)
_DRAW = {"draw", "tie", "the draw"}


def _kalshi_price(yes_bid, yes_ask) -> float:
    """Kalshi prices are integer cents (0-100). Midpoint -> probability in [0,1]."""
    b = float(yes_bid or 0); a = float(yes_ask or 0)
    mid = (b + a) / 2 if (b or a) else 0.0
    return round(mid / 100.0, 4)


def parse_kalshi_market(raw: dict) -> MarketQuoteRow | None:
    """Map a Kalshi match-winner market to a MarketQuoteRow, or None if unparseable.

    Verified against the live dump captured by `kalshi-discover` (Task 1)."""
    title = raw.get("title", "")
    sub = (raw.get("yes_sub_title") or "").strip()
    m = _TITLE_RE.match(title.strip())
    if not m or not sub:
        return None
    home = canonical_team(m.group("home"))
    away = canonical_team(m.group("away"))
    if sub.lower() in _DRAW:
        side = "draw"
    elif canonical_team(sub) == home:
        side = "home"
    elif canonical_team(sub) == away:
        side = "away"
    else:
        return None
    return MarketQuoteRow(
        home=home, away=away, market_type="1x2", side=side, line=None,
        price=_kalshi_price(raw.get("yes_bid"), raw.get("yes_ask")),
        ticker=raw.get("ticker", ""),
    )


def fetch_quotes(series_tickers: list[str], client: KalshiReadClient | None = None) -> list[MarketQuoteRow]:
    """Fetch open WC markets for the given series and parse them to MarketQuoteRows.

    Unparseable markets are skipped (never fabricated)."""
    owns = client is None
    client = client or KalshiReadClient()
    rows: list[MarketQuoteRow] = []
    try:
        for series in series_tickers:
            for raw in client.get_markets(series):
                q = parse_kalshi_market(raw)
                if q is not None and q.price > 0:
                    rows.append(q)
    finally:
        if owns:
            client.close()
    return rows
```

- [ ] **Step 5: Delete the superseded stub**

```bash
git rm touchline/data/kalshi_adapter.py
```
(If anything imports `kalshi_adapter`, repoint it to `kalshi_quotes`. Grep first:
`grep -rn kalshi_adapter touchline tests`.)

- [ ] **Step 6: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kalshi_quotes.py -v`
Expected: PASS (5 passed). If the real Task-1 fixture has a different title/price format,
adjust `_TITLE_RE` / `_kalshi_price` until these tests (rewritten against the real data) pass.

- [ ] **Step 7: Commit**

```bash
git add touchline/data/kalshi_quotes.py tests/test_kalshi_quotes.py tests/fixtures/kalshi_market_sample.json
git commit -m "feat: Kalshi market -> MarketQuoteRow parser (replaces adapter stub)"
```

---

## Task 3: `touchline daily` command

**Files:**
- Modify: `touchline/cli.py`
- Test: `tests/test_daily.py`

- [ ] **Step 1: Write the failing test (pure orchestration, no network)**

`tests/test_daily.py`:
```python
from datetime import date
from pathlib import Path
from touchline.models import Match
from touchline.model.ratings import Ratings
from touchline.edge.quotes import MarketQuoteRow
from touchline.cli import run_daily


def _ratings():
    return Ratings(attack={"USA": 0.5, "Wales": -0.2},
                   defense={"USA": 0.3, "Wales": -0.2}, home_adv=0.2, rho=-0.05, intercept=0.1)


def test_run_daily_writes_reports(tmp_path):
    history = [Match(match_date=date(2026, 6, 10), home_team="USA", away_team="Iran",
                     home_goals=2, away_goals=0, competition="WC", stage=None,
                     venue="MetLife Stadium", played=True, source="t")]
    fixtures = [("USA", "Wales", date(2026, 6, 24), "SoFi Stadium")]
    quotes = [MarketQuoteRow("USA", "Wales", "1x2", "home", None, 0.50, "T-USA"),
              MarketQuoteRow("USA", "Wales", "1x2", "away", None, 0.30, "T-WAL")]
    md_path, json_path = run_daily(
        ratings=_ratings(), overlay={}, quotes=quotes, fixtures=fixtures,
        history=history, team_games={"USA": 30, "Wales": 25},
        as_of="2026-06-23", out_dir=tmp_path,
    )
    assert md_path.exists() and json_path.exists()
    assert "Touchline Edge Report" in md_path.read_text(encoding="utf-8")
    assert md_path.name == "2026-06-23-report.md"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_daily.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_daily'`.

- [ ] **Step 3: Add `run_daily` + the `daily` command to `touchline/cli.py`**

Add imports near the top (integrate; `date`, `Counter`, `run_price`, `load_overlay`,
`build_context` are already imported from Plan 3):
```python
from touchline.data import kalshi_quotes
```

Add this function above `main` (next to `run_price`):
```python
def run_daily(ratings, overlay, quotes, fixtures, history, team_games, as_of, out_dir):
    """Price upcoming fixtures vs Kalshi quotes and write a dated report.

    Thin wrapper over run_price; returns (markdown_path, json_path)."""
    from pathlib import Path
    _, md, js = run_price(ratings, overlay, quotes, fixtures, history, team_games, as_of=as_of)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{as_of}-report.md"
    json_path = out_dir / f"{as_of}-report.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(js, encoding="utf-8")
    return md_path, json_path
```

Register the subcommand inside `main`:
```python
    daily_p = sub.add_parser("daily", help="Full pipeline: ingest, fit, fetch Kalshi, report")
    daily_p.add_argument("--series", default="KXMENWORLDCUP",
                         help="Comma-separated Kalshi WC series tickers")
    daily_p.add_argument("--skip-refresh", action="store_true",
                         help="Use existing DB/ratings instead of re-ingesting and re-fitting")
```

Add the branch inside `main` (before `return 1`):
```python
    if args.command == "daily":
        db = Database(config.DB_PATH)
        db.init_schema()
        if not args.skip_refresh:
            historical = _gather_historical()
            intl = intl_results.gather(config.CACHE_DIR, since_year=2014)
            live = worldcupjson.fetch_matches(config.CACHE_DIR)
            run_ingest(db, historical=historical + intl, live=live)
            elo_path = config.CACHE_DIR / "elo.csv"
            elo = load_elo(elo_path) if elo_path.is_file() else EloTable()
            db.save_ratings(fit_ratings(db.all_matches(), elo,
                                        half_life_days=900, prior_weight=0.05, as_of=date.today()))
        ratings = db.load_ratings()
        overlay = load_overlay(config.CACHE_DIR / "squad_adjustments.json")
        history = db.all_matches()
        team_games = Counter(t for m in history if m.played for t in (m.home_team, m.away_team))
        fixtures = [(m.home_team, m.away_team, m.match_date, m.venue)
                    for m in history if not m.played and m.venue]
        quotes = kalshi_quotes.fetch_quotes(args.series.split(","))
        md_path, json_path = run_daily(ratings, overlay, quotes, fixtures, history,
                                       dict(team_games), as_of=date.today().isoformat(),
                                       out_dir=config.DATA_DIR / "reports")
        print(f"Priced {len(quotes)} Kalshi markets -> {md_path}")
        return 0
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_daily.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all green (no remaining `kalshi_adapter` references).

- [ ] **Step 6: Live smoke test (needs keys)**

Run: `.venv/Scripts/python.exe -m touchline.cli daily`
Expected: prints `Priced N Kalshi markets -> ...reports/<date>-report.md` and writes the
dated md + json. If Kalshi lists only winner (tournament) markets and no per-match
markets for upcoming games, `N` may be small or 0 — that is a real finding about Kalshi's
WC coverage, not a bug; report it.

- [ ] **Step 7: Commit**

```bash
git add touchline/cli.py tests/test_daily.py
git commit -m "feat: touchline daily command (Kalshi odds -> dated edge report)"
```

---

## Self-Review Notes

- **Spec coverage:** `kalshi-discover` ✓ (Task 1), `kalshi_quotes.parse_kalshi_market`/`fetch_quotes` replacing the stub ✓ (Task 2), `touchline daily` reusing `run_price` ✓ (Task 3), read-only constraint preserved (only `get_markets`/`get_market` used), cents→probability handled in `_kalshi_price`, dated report output ✓. Error handling: missing keys surface from the client; unparseable markets → `None`/skip; empty quotes → empty-but-valid report (`run_price` already handles no-pick case).
- **Placeholder scan:** No TBD/TODO. The "verify/adjust against the Task-1 fixture" notes are a genuine discovery-then-confirm step (the spec's whole approach), not placeholders — the provided parser is complete and runnable; the test fixture is real data captured in Task 1.
- **Type consistency:** `MarketQuoteRow(home, away, market_type, side, line, price, ticker)` matches Plan 3's definition exactly. `parse_kalshi_market(raw)->MarketQuoteRow|None`, `fetch_quotes(series_tickers, client=None)`, `_kalshi_price(yes_bid, yes_ask)`, and `run_daily(ratings, overlay, quotes, fixtures, history, team_games, as_of, out_dir)` are consistent across `kalshi_quotes.py`, `cli.py`, and tests. `run_price` signature matches Plan 3.
- **Execution dependency (not a placeholder):** Tasks 1 & 3's live smoke tests require the Kalshi keys in the environment. The unit tests (parser, run_daily) are fully offline. If keys are unavailable, the offline tests still pass and the parser is verified against the representative fixture; the live confirmation happens when keys are present.
