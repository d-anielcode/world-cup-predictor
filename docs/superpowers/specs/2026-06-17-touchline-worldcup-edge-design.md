# Touchline — World Cup Kalshi Edge-Finder

**Date:** 2026-06-17
**Status:** Design approved, pending spec review
**Context:** Built during the live 2026 FIFA World Cup (USA/Canada/Mexico).

## 1. Purpose

Find advantageous Kalshi World Cup markets. A deterministic statistical engine prices every
relevant Kalshi market (its own estimate of the true probability), compares that to the market
price, and flags where the market is **wrong / overvaluing** an outcome. It outputs per-bet
percentage chances and a ranked list of **top predictions**.

**This is an analysis tool, not an auto-bettor.** There is no order-placement code anywhere in
this project. Claude Code (in-session) is the human-in-the-loop review layer: runs the engine,
applies current lineup/injury judgment via the squad overlay, sanity-checks edges, and surfaces
top picks.

## 2. Core decisions

- **No LLM / no Anthropic API in production.** The engine is fully deterministic and
  backtestable. Reasoning/curation is done by Claude Code in-session, not wired into the app.
  (Mirrors the Compass deterministic-core pattern.)
- **Goal model:** Dixon-Coles bivariate Poisson with Elo-seeded attack/defense ratings.
- **Markets priced:** match winner (1X2), total goals (O/U), both-teams-to-score (BTTS),
  spread/handicap — all derived from one scoreline probability matrix.
- **Corners:** OUT OF SCOPE / stretch — no corner data in available free sources.
- **Pre-match only.** No in-match live trading logic.

## 3. Data sources

| Source | Use | Notes |
|---|---|---|
| openfootball/worldcup (Football.TXT, CC0) | Historical results 1930–2026, WC + qualifiers; lineups where present | Clone + parse. Also pull openfootball national-team repos (friendlies / Nations League) for max international form history. |
| worldcupjson.net (estiens/world_cup_json) | Live 2026 tournament: results, events, lineups, standings | Rate-limited 10 req/60s → cache with TTL. |
| International Elo (e.g. eloratings.net) | Rating prior for thin-sample teams | Scraped + cached. |
| Kalshi REST (read-only) | World Cup series markets, prices, orderbooks | Extracted/copied from `edgerunner/execution/kalshi_client.py`, **read methods only**. Keys exist locally. No order path. |
| Open-Meteo (free, no key) | Per-match weather forecast for heat/WBGT | Reuse EdgeRunner weather-edge code. |
| Static venue table (built once) | lat/long, altitude, roof/AC, time zone per 2026 venue | Powers travel, altitude, heat, host detection. |

**Storage:** local SQLite for normalized matches + ratings + report history; raw API responses
cached as JSON files on disk.

## 4. Rating engine

Dixon-Coles bivariate Poisson. Per team: **attack** and **defense** strength. Global terms:
**home advantage** (applied conditionally — see §5.1) and **rho** (low-score correlation
correction). Fit by weighted maximum-likelihood (scipy) over international results with
**exponential time-decay** (recent matches weighted more; half-life is a tunable parameter).

Teams with thin recent samples are **Elo-seeded**: the published international Elo provides a
prior on strength, updated by observed goals. The Elo-prior weight is a tunable parameter.

Output: a ratings table (attack, defense per team; global home-adv, rho).

## 5. Adjustment factors (deterministic modifiers)

Applied to the base ratings / expected goals before building the scoreline matrix. All computed
from schedule + venue + forecast data; all logged for audit.

### 5.1 Tier 1 — structural (high value)
1. **Conditional host home-advantage.** Home advantage applies ONLY when a host nation (USA,
   Canada, Mexico) plays in its own country; all other matches are neutral. (Fixes the structural
   error of a single global home term in a neutral-site tournament.)
2. **Travel distance / trip leg.** Per team, distance from previous match venue to current venue
   (haversine from the venue table). Top predictor in the literature; large in a continent-spanning
   tournament. Penalty scales with distance + short rest.
3. **Altitude.** Venue altitude (Mexico City ~2,240m, Guadalajara, Monterrey) adjusts expected
   goals / fatigue; acclimatized teams (Mexico) penalized less.

### 5.2 Tier 2 — computed context modifiers
4. **Heat / WBGT.** From venue + kickoff time + Open-Meteo forecast. High WBGT suppresses
   goal-scoring (lean Under on totals) and amplifies late-game fatigue. Reuse EdgeRunner weather code.
5. **Rest-days / congestion differential.** Days since each team's previous match, from schedule.
6. **Match-importance / rotation flag.** Dead rubbers (already-qualified or eliminated teams in
   final group game) → feeds the squad overlay as a rotation signal.

### 5.3 Tier 3 — market sanity layer
7. **Favorite-longshot bias.** Longshots are systematically overpriced, favorites underpriced.
   Used to **weight edge confidence**, not to alter the model probability: a fade-the-longshot
   edge is trusted more than a back-the-longshot edge of equal magnitude.

### 5.4 Squad-adjustment overlay
Versioned `squad_adjustments.json`: per team, explicit attack/defense **multipliers** with
`reason`, `source`, `timestamp` (e.g. `{"Brazil": {"attack": 0.90, "reason": "Neymar out",
"source": "...", "timestamp": "..."}}`). Applied after fitting, before pricing. Claude Code
curates it each matchday from current lineup/injury news, informed by all available lineup feeds
(openfootball + live API). Every adjustment is logged and auditable. This is how individual
players / lineup changes enter the otherwise team-level model.

### Skipped (YAGNI)
Head-to-head history (~6% importance), confederation familiarity, set-piece/style granularity.

## 6. Market pricing

For each fixture: build a **scoreline probability matrix** (0–0 … ~10–10) via Dixon-Coles from the
adjusted expected goals. Derive each market by summing the relevant cells:
- **1X2:** sum upper/lower/diagonal triangles.
- **Totals O/U:** sum by total goals at each line.
- **BTTS:** sum cells where both scores ≥ 1.
- **Spread/handicap:** sum by goal margin at each line.

## 7. Edge computation & ranking

Match model probabilities to Kalshi market tickers. For each market:
`edge = model_prob − implied_prob` (handling Kalshi fixed-point yes/no prices), plus EV. Flag
over/undervalued. **Rank top picks by edge, qualified by a confidence tag** combining sample size,
reliance on the Elo prior vs observed games, and the favorite-longshot direction — so thin-data
edges don't masquerade as strong.

## 8. Report output

- **Markdown report** per run: grouped by match, every market with model prob / Kalshi price /
  edge / confidence, plus a ranked **Top Predictions** section. This is what Claude Code reviews
  with the user.
- **JSON snapshot** per run: full machine-readable output for history + backtesting.

## 9. Calibration / backtest harness

Replays the engine **as-of each completed 2026 WC match** using only pre-kickoff data. Scores
**Brier + log-loss** vs actual outcomes and compares against closing Kalshi prices (did we flag the
correct side?). Used to tune the time-decay half-life and Elo-prior weight before trusting upcoming
fixtures. This is the compressed trial-and-error substitute for slow live iteration.

## 10. CLI

- `ingest` — clone/refresh openfootball, pull live API, scrape Elo, cache.
- `fit-ratings` — fit Dixon-Coles ratings from cached data.
- `overlay` — edit/validate squad_adjustments.json.
- `price` — fetch Kalshi WC markets, apply factors + overlay, produce md + json report.
- `backtest` — run the calibration harness over completed matches.

## 11. Tech stack & layout

Python 3.11; numpy/scipy (Dixon-Coles MLE), pandas, httpx, SQLite. New repo at
`C:\Users\dcho0\Documents\touchline`.

```
touchline/
├── data/        # ingest: openfootball parser, worldcupjson client, elo scraper,
│                #   kalshi_read.py (read-only), weather (open-meteo), cache, venues.py
├── model/       # ratings.py (DC MLE), dixon_coles.py, factors.py (travel/altitude/heat/host),
│                #   pricing.py (scoreline matrix → markets)
├── edge/        # compare.py (model vs Kalshi), rank.py (top picks + confidence)
├── overlay/     # squad_adjustments.json + loader/validator
├── report/      # markdown + json writers
├── backtest/    # calibration harness, scoring (brier/log-loss)
├── storage/     # sqlite schema + helpers
└── cli.py
```

## 12. Error handling & constraints

- Rate-limit + cache for worldcupjson (10/60s) and Kalshi; never hammer.
- Team with no/low data → fall back to Elo prior and flag **low confidence** (never silently
  emit a confident edge on thin data).
- Stale forecast/price → flag, don't fail the whole run.
- **No order-placement code exists in this repo.** Kalshi client is read-only by construction.

## 13. Out of scope (YAGNI)

Corners, auto-betting / any order path, web UI / dashboard, in-match live updating,
multi-tournament generalization.
