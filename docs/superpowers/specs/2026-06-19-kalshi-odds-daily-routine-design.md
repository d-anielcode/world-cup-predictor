# Kalshi Live Odds + Daily Routine — Design

**Date:** 2026-06-19
**Status:** Design approved, pending spec review

## Purpose

Automate the market side of Touchline so the edge report runs end-to-end without
hand-gathering odds. Two pieces:

1. **Live Kalshi odds** — fetch real World Cup market prices from Kalshi (the venue
   the project targets) via the existing read-only client.
2. **`touchline daily` routine** — one command that runs the whole pipeline
   (ingest → fit → fetch Kalshi odds → price → ranked edge report).

Kalshi prices are quoted in cents = implied probability, a clean match for the
model's probabilities (no vig removal needed). **Read-only — no order-placement code.**

## Approach: discovery-first

The exact Kalshi WC event/market schema (tickers, titles) is unconfirmed. Rather than
guess it (the mistake that bit the openfootball parser), the first step **captures the
real live structure** and the parser is built and tested against that captured fixture.

## Prerequisite

Touchline's `config.py` already reads `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`
from the environment. Point them at the existing EdgeRunner key (env vars or a local
`.env`). The client is read-only by construction; no order/cancel methods exist.

If the keys are unreachable at implementation time, the discovery step fails with a clear
message and the build pauses there (the parser cannot be written against guessed data).

## Components

### 1. `kalshi-discover` CLI command
One-shot. Uses the read-only client to fetch the World Cup series' events and markets,
and writes:
- `touchline_data/kalshi_discover.json` — raw market dicts (tickers, titles, yes/no
  prices, status) for parser development.
- A stdout summary: series ticker(s), market count, and 5–10 sample titles.

Purpose: confirm the live schema and serve as a "is the data there?" check. The WC
winner series is `KXMENWORLDCUP`; match/group series tickers are discovered here.

### 2. `touchline/data/kalshi_quotes.py`
Replaces the `kalshi_adapter.py` stub.

- `parse_kalshi_market(raw: dict) -> MarketQuoteRow | None` — maps one Kalshi market to
  the normalized `MarketQuoteRow(home, away, market_type, side, line, price, ticker)`:
  extracts the two teams and the outcome from the market/event title, canonicalizes team
  names via `canonical_team`, sets `price = yes_price` (cents → 0–1), and `market_type`
  (`"1x2"`; `"total"`/`"handicap"` only if Kalshi lists them). Returns `None` for any
  market it cannot confidently parse (logged, never fabricated).
- `fetch_quotes(series_tickers: list[str]) -> list[MarketQuoteRow]` — pulls open markets
  via the read-only client's `get_markets`, parses each, drops `None`s.

Built and unit-tested against a fixture captured from the `kalshi-discover` dump.

### 3. `touchline daily` CLI command
Runs the full pipeline and writes a dated report:
- `ingest` (refresh matches), `fit-ratings` (refresh ratings),
- `fetch_quotes` for the WC series,
- build `FactorContext` per upcoming fixture + apply the squad overlay,
- `run_price` (the existing Plan-3 orchestration) → ranked edges,
- write `touchline_data/reports/<date>-report.md` and `.json`.

`run_price` is reused unchanged; Kalshi `MarketQuoteRow`s simply replace the CSV ones.

## Data flow

```
Kalshi market (binary Yes, cents)
   -> parse_kalshi_market -> MarketQuoteRow(price = cents/100)
   -> run_price (ratings + factors + overlay vs quotes)
   -> ranked edge report (md + json)
```

Because Kalshi prices are already implied probabilities, the edge is
`model_prob − kalshi_price` directly — no vig adjustment.

## Error handling

- Missing/invalid Kalshi keys → clear message, non-zero exit; `daily` aborts before
  writing a misleading report.
- Unparseable market → skipped with a logged reason; never guessed.
- No upcoming fixtures / no matching Kalshi markets → empty-but-valid report, no crash.
- Kalshi likely lists mainly match-winner (1X2) markets — the model's strongest,
  best-calibrated market — so limited market types is acceptable, not a blocker.

## Testing

- `parse_kalshi_market`: unit tests against the captured real fixture (correct teams,
  side, price; `None` on unparseable).
- `fetch_quotes`: parsing + skip logic with mocked client responses (no live calls in CI).
- `daily` orchestration: pure path with injected quotes (like the existing `run_price`
  test), asserting a report is produced and picks are ranked.
- Live `kalshi-discover` + `daily` are manual smoke tests (need keys, not run in CI).

## Out of scope (YAGNI)

Order placement (read-only forever), scheduling infrastructure (the user runs the
command), totals/spreads if Kalshi doesn't list them, and a UI.

## Open items for implementation (not guesses — confirmed against live data)

- Exact WC match/group **series ticker(s)** and **market-title format** — captured by
  `kalshi-discover` in task 1, then the parser is written against it.
- Whether Kalshi offers a **Draw** contract per match (3-way) or only two-way
  (team-to-advance / team-win) — determined from the dump.
