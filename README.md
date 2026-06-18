# Touchline

A deterministic statistical engine that finds **advantageous Kalshi World Cup markets**. It prices every market with its own probability estimate, compares that to the market price, and reports where the market looks **wrong** — ranked by edge. It is an **analysis tool, not an auto-bettor**: there is no order-placement code anywhere in the project.

Built during the 2026 FIFA World Cup (USA / Canada / Mexico).

## What it does

For each match it estimates a true probability for:

- **Match winner** (home / draw / away)
- **Total goals** (over/under any line)
- **Both teams to score**
- **Spread / handicap**

…then computes `edge = model probability − market price`, scores confidence, and ranks the **top predictions**. A human (or Claude Code) reviews the report, layers in current lineup/injury news, and decides what to act on.

## How it works

```
ingest ──→ fit-ratings ──→ price ──→ ranked edge report (Markdown + JSON)
                              ↑
                       backtest / calibrate  (validates & tunes the model)
```

- **Goal model:** Dixon-Coles bivariate Poisson. Each team has attack/defense ratings fit by **time-decay-weighted maximum likelihood**, seeded with an **international Elo prior** so thin-sample teams stay sensible. One scoreline-probability matrix prices every market.
- **Adjustment factors (deterministic):** conditional host home-advantage (only USA/Canada/Mexico at home), travel distance, altitude (Mexico City), heat (WBGT from Open-Meteo), rest days, plus a curated squad-injury overlay.
- **Edge & confidence:** confidence combines sample depth (how much a rating leans on the Elo prior) and the favorite-longshot bias (fading an overpriced longshot is trusted more than backing one).
- **Validation:** a walk-forward backtest replays completed matches using only pre-kickoff data and scores predictions with Brier + log-loss; a grid search calibrates the rating hyperparameters.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip

python -m touchline.cli ingest         # pull historical + live matches into SQLite
python -m touchline.cli fit-ratings    # fit Dixon-Coles ratings (calibrated defaults)
python -m touchline.cli price --quotes quotes.csv   # write report.md + report.json
python -m touchline.cli backtest --eval-start 2018-01-01            # validate
python -m touchline.cli backtest --calibrate                       # tune hyperparameters
```

Market prices are read from a quotes CSV (`home,away,market_type,side,line,price,ticker`), which keeps the whole pipeline runnable offline. A read-only Kalshi client is included for live market data.

## What the validation says (honestly)

On 128 historical World Cup matches (2018+), the model **beats a uniform baseline** but by a modest margin — Brier 0.66 vs 0.667, log-loss 1.073 vs 1.099 after calibration. World Cup outcomes are genuinely hard to predict on a small sample. The calibration did surface a robust, interpretable result: **longer memory wins** (national-team form is stable across years), so the default rating half-life is 900 days.

## Architecture

```
touchline/
├── data/        ingest: openfootball parser, worldcupjson client, Elo loader,
│                read-only Kalshi client, venue table, Open-Meteo weather
├── model/       Dixon-Coles matrix, ratings, weighted MLE fit, factors, pricing
├── overlay/     curated squad-injury multipliers
├── edge/        quotes, model lookup, edge/EV/confidence, context, ranking
├── report/      Markdown + JSON renderers
├── backtest/    walk-forward harness, scoring, hyperparameter calibration
└── cli.py       ingest / fit-ratings / price / backtest commands
```

Stack: Python 3.11 (sync), numpy/scipy, SQLite, httpx. No LLM or API key required at runtime. 107 tests, all network-free. Design specs and phase-by-phase implementation plans live in [`docs/superpowers/`](docs/superpowers).

## Data sources

- [openfootball/worldcup](https://github.com/openfootball/worldcup) — historical results (CC0)
- [worldcupjson.net](https://worldcupjson.net) — live tournament data
- International Elo ratings (prior) · Open-Meteo (weather, no key)

## Scope & limits

Out of scope by design: automated betting, corners (no source data), and any order path. Factor coefficients are expert-set (no historical venue/weather data exists to fit them). Going live needs the 2026 data feed populating and the Kalshi World Cup market schema confirmed.
