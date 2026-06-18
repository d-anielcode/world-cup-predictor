# Touchline Backtest, Calibration & Weather Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and tune the model — replay completed matches to score the model's predictions (Brier + log-loss), grid-search the rating hyperparameters to minimize error, and wire real Open-Meteo heat into the live pricing path.

**Architecture:** A new `touchline/backtest/` package (walk-forward harness + hyperparameter grid search) and a `touchline/data/weather.py` module (pure WBGT estimate + cached Open-Meteo fetch). The backtest re-fits ratings as-of each evaluated match's date (cached per date), prices the 1X2 outcome, and scores it against the realized result. A `backtest` CLI runs scoring and `--calibrate`. Weather is folded into the existing `build_context`.

**Tech Stack:** Python 3.11 (sync), numpy, scipy, httpx, pytest. Builds on Plans 1–3 (`Database`, `Match`, `EloTable`, `fit_ratings`, `Ratings`, `scoreline_matrix`, `prob_1x2`, `build_context`).

**Prerequisite:** Plans 1–3 merged to master.

---

## Conventions

**Backtest pricing convention.** `fit_ratings` adds `home_adv` to *every* match's home side. To score consistently with how the model was trained, the backtest prices each replayed match with `apply_home_adv=True` and **no** environmental factors (historical venue coordinates aren't available). This measures the rating model's predictive accuracy on the openfootball home/away designation.

**Scoring (1X2, three outcomes home/draw/away):**
- Realized one-hot `y`: home if `hg>ag`, draw if `hg==ag`, away if `hg<ag`.
- **Brier** = mean over matches of `Σ_o (p_o − y_o)²` (range 0–2; lower better).
- **Log-loss** = mean over matches of `−log(p_actual)`, with `p_actual` clipped to `[1e-12, 1]` (lower better).
- Baselines for sanity: uniform `(1/3,1/3,1/3)` gives Brier `2/3 ≈ 0.667` and log-loss `ln 3 ≈ 1.0986`. A useful model beats both.

**WBGT estimate** (Australian BoM outdoor approximation) from air temperature `T` (°C) and relative humidity `RH` (%):
```
e    = (RH/100) * 6.105 * exp(17.27*T / (237.7 + T))   # vapour pressure, hPa
WBGT = 0.567*T + 0.393*e + 3.94
```

---

## File Structure

```
touchline/
├── backtest/
│   ├── __init__.py
│   ├── scoring.py     # brier_score, log_loss, outcome_index
│   ├── harness.py     # walk-forward backtest -> BacktestResult
│   └── calibrate.py   # grid search -> CalibrationResult
├── data/
│   └── weather.py     # estimate_wbgt (pure) + fetch_wbgt (Open-Meteo, cached)
├── edge/context.py    # MODIFY: build_context gains optional wbgt_c
└── cli.py             # MODIFY: add `backtest` command
tests/
├── test_backtest_scoring.py
├── test_backtest_harness.py
├── test_backtest_calibrate.py
├── test_weather.py
└── test_context_weather.py
```

Run all commands from `C:\Users\dcho0\Documents\touchline` via `.venv/Scripts/python.exe -m pytest ...`.

---

## Task 1: Scoring functions

**Files:**
- Create: `touchline/backtest/__init__.py`, `touchline/backtest/scoring.py`
- Test: `tests/test_backtest_scoring.py`

- [ ] **Step 1: Write the failing test**

`tests/test_backtest_scoring.py`:
```python
import math
from touchline.backtest.scoring import outcome_index, brier_score, log_loss


def test_outcome_index_maps_result():
    assert outcome_index(2, 0) == 0   # home
    assert outcome_index(1, 1) == 1   # draw
    assert outcome_index(0, 3) == 2   # away


def test_brier_perfect_prediction_is_zero():
    assert brier_score([(1.0, 0.0, 0.0)], [0]) == 0.0


def test_brier_uniform_is_two_thirds():
    b = brier_score([(1/3, 1/3, 1/3)], [0])
    assert abs(b - 2/3) < 1e-9


def test_log_loss_uniform_is_ln3():
    ll = log_loss([(1/3, 1/3, 1/3)], [2])
    assert abs(ll - math.log(3)) < 1e-9


def test_log_loss_clips_zero_probability():
    # Predicting 0 for the actual outcome must not produce -inf.
    ll = log_loss([(1.0, 0.0, 0.0)], [2])
    assert ll > 0 and math.isfinite(ll)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`touchline/backtest/__init__.py`: empty file.

`touchline/backtest/scoring.py`:
```python
from __future__ import annotations

import math

_EPS = 1e-12


def outcome_index(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def brier_score(probs: list[tuple[float, float, float]], outcomes: list[int]) -> float:
    """Mean multiclass Brier score over (home, draw, away) predictions."""
    total = 0.0
    for (ph, pd, pa), o in zip(probs, outcomes):
        y = [0.0, 0.0, 0.0]
        y[o] = 1.0
        total += (ph - y[0]) ** 2 + (pd - y[1]) ** 2 + (pa - y[2]) ** 2
    return total / len(probs) if probs else 0.0


def log_loss(probs: list[tuple[float, float, float]], outcomes: list[int]) -> float:
    """Mean negative log-likelihood of the realized outcomes (clipped)."""
    total = 0.0
    for triple, o in zip(probs, outcomes):
        p = min(1.0, max(_EPS, triple[o]))
        total += -math.log(p)
    return total / len(probs) if probs else 0.0
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_scoring.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/backtest/__init__.py touchline/backtest/scoring.py tests/test_backtest_scoring.py
git commit -m "feat: backtest scoring (Brier, log-loss, outcome index)"
```

---

## Task 2: Walk-forward backtest harness

**Files:**
- Create: `touchline/backtest/harness.py`
- Test: `tests/test_backtest_harness.py`

- [ ] **Step 1: Write the failing test**

`tests/test_backtest_harness.py`:
```python
import math
from datetime import date, timedelta

import numpy as np

from touchline.models import Match
from touchline.data.elo import EloTable
from touchline.backtest.harness import backtest, BacktestResult


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    attack = {"Strong": 0.7, "Mid": 0.0, "Weak": -0.7}
    defense = {"Strong": 0.6, "Mid": 0.0, "Weak": -0.6}
    teams = list(attack)
    base = date(2025, 1, 1)
    out = []
    for i in range(400):
        h, a = rng.choice(teams, size=2, replace=False)
        lam = math.exp(attack[h] - defense[a])
        mu = math.exp(attack[a] - defense[h])
        out.append(Match(match_date=base + timedelta(days=i * 2),
                         home_team=str(h), away_team=str(a),
                         home_goals=int(rng.poisson(lam)), away_goals=int(rng.poisson(mu)),
                         competition="Syn", stage=None, venue=None, played=True, source="t"))
    return out


def test_backtest_returns_result_with_counts_and_scores():
    matches = _synthetic()
    eval_start = date(2026, 1, 1)   # evaluate the back half
    r = backtest(matches, eval_start=eval_start, half_life_days=400,
                 prior_weight=0.05, elo=EloTable())
    assert isinstance(r, BacktestResult)
    assert r.n_matches > 0
    assert 0.0 <= r.brier <= 2.0
    assert r.log_loss > 0


def test_model_beats_uniform_baseline_on_separable_data():
    matches = _synthetic()
    r = backtest(matches, eval_start=date(2026, 1, 1), half_life_days=400,
                 prior_weight=0.05, elo=EloTable())
    assert r.log_loss < math.log(3)     # better than uniform
    assert r.brier < 2/3


def test_no_eval_matches_returns_zero_count():
    matches = _synthetic()
    r = backtest(matches, eval_start=date(2099, 1, 1), half_life_days=400,
                 prior_weight=0.05, elo=EloTable())
    assert r.n_matches == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_harness.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`touchline/backtest/harness.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from touchline.data.elo import EloTable
from touchline.models import Match
from touchline.model.fit import fit_ratings
from touchline.model.dixon_coles import scoreline_matrix
from touchline.model.pricing import prob_1x2
from touchline.backtest.scoring import brier_score, log_loss, outcome_index


@dataclass
class BacktestResult:
    n_matches: int
    brier: float
    log_loss: float


def backtest(
    matches: list[Match],
    eval_start: date,
    half_life_days: float,
    prior_weight: float,
    elo: EloTable,
) -> BacktestResult:
    """Walk-forward: for each played match on/after eval_start, fit ratings using only
    prior matches and score the 1X2 prediction against the realized outcome.

    Ratings are re-fit per distinct as-of date (cached) to bound cost. Prices use
    apply_home_adv=True to match the fitting convention; no environmental factors."""
    played = sorted([m for m in matches if m.played and m.home_goals is not None
                     and m.away_goals is not None], key=lambda m: m.match_date)
    fits: dict[date, object] = {}
    probs: list[tuple[float, float, float]] = []
    outcomes: list[int] = []
    for m in played:
        if m.match_date < eval_start:
            continue
        if m.match_date not in fits:
            prior = [p for p in played if p.match_date < m.match_date]
            if not prior:
                continue
            fits[m.match_date] = fit_ratings(
                prior, elo, half_life_days=half_life_days,
                prior_weight=prior_weight, as_of=m.match_date,
            )
        ratings = fits.get(m.match_date)
        if ratings is None:
            continue
        lam, mu = ratings.expected_goals(m.home_team, m.away_team, apply_home_adv=True)
        matrix = scoreline_matrix(lam, mu, ratings.rho)
        home, draw, away = prob_1x2(matrix)
        probs.append((home, draw, away))
        outcomes.append(outcome_index(m.home_goals, m.away_goals))
    return BacktestResult(
        n_matches=len(probs),
        brier=brier_score(probs, outcomes),
        log_loss=log_loss(probs, outcomes),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_harness.py -v`
Expected: PASS (3 passed). (Fitting runs per distinct date; the synthetic set has ~200 eval dates, so this test takes a few seconds — acceptable.)

- [ ] **Step 5: Commit**

```bash
git add touchline/backtest/harness.py tests/test_backtest_harness.py
git commit -m "feat: walk-forward backtest harness scoring 1X2 predictions"
```

---

## Task 3: Hyperparameter calibration

**Files:**
- Create: `touchline/backtest/calibrate.py`
- Test: `tests/test_backtest_calibrate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_backtest_calibrate.py`:
```python
from datetime import date, timedelta
import math
import numpy as np
from touchline.models import Match
from touchline.data.elo import EloTable
from touchline.backtest.calibrate import calibrate, CalibrationResult


def _synthetic(seed=1):
    rng = np.random.default_rng(seed)
    attack = {"Strong": 0.7, "Mid": 0.0, "Weak": -0.7}
    defense = {"Strong": 0.6, "Mid": 0.0, "Weak": -0.6}
    teams = list(attack)
    base = date(2025, 1, 1)
    out = []
    for i in range(300):
        h, a = rng.choice(teams, size=2, replace=False)
        lam = math.exp(attack[h] - defense[a]); mu = math.exp(attack[a] - defense[h])
        out.append(Match(match_date=base + timedelta(days=i * 2),
                         home_team=str(h), away_team=str(a),
                         home_goals=int(rng.poisson(lam)), away_goals=int(rng.poisson(mu)),
                         competition="Syn", stage=None, venue=None, played=True, source="t"))
    return out


def test_calibrate_returns_best_params_from_grid():
    matches = _synthetic()
    res = calibrate(matches, eval_start=date(2026, 1, 1), elo=EloTable(),
                    half_lifes=[180, 540], prior_weights=[0.01, 0.5])
    assert isinstance(res, CalibrationResult)
    assert res.best_half_life in (180, 540)
    assert res.best_prior_weight in (0.01, 0.5)
    # best log-loss must be the minimum over the grid
    assert res.best_log_loss == min(row[2] for row in res.grid)
    assert len(res.grid) == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_calibrate.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`touchline/backtest/calibrate.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from touchline.data.elo import EloTable
from touchline.models import Match
from touchline.backtest.harness import backtest


@dataclass
class CalibrationResult:
    best_half_life: float
    best_prior_weight: float
    best_log_loss: float
    grid: list[tuple[float, float, float]]   # (half_life, prior_weight, log_loss)


def calibrate(
    matches: list[Match],
    eval_start: date,
    elo: EloTable,
    half_lifes: list[float],
    prior_weights: list[float],
) -> CalibrationResult:
    """Grid-search (half_life, prior_weight) minimizing backtest log-loss."""
    grid: list[tuple[float, float, float]] = []
    for hl in half_lifes:
        for pw in prior_weights:
            result = backtest(matches, eval_start=eval_start, half_life_days=hl,
                              prior_weight=pw, elo=elo)
            grid.append((hl, pw, result.log_loss))
    best = min(grid, key=lambda row: row[2])
    return CalibrationResult(
        best_half_life=best[0], best_prior_weight=best[1], best_log_loss=best[2],
        grid=grid,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_calibrate.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/backtest/calibrate.py tests/test_backtest_calibrate.py
git commit -m "feat: hyperparameter grid-search calibration over backtest log-loss"
```

---

## Task 4: Weather — WBGT estimate + Open-Meteo fetch

**Files:**
- Create: `touchline/data/weather.py`, `tests/fixtures/openmeteo_sample.json`
- Test: `tests/test_weather.py`

- [ ] **Step 1: Create the fixture**

`tests/fixtures/openmeteo_sample.json`:
```json
{
  "latitude": 25.96,
  "longitude": -80.24,
  "hourly": {
    "time": ["2026-06-24T18:00", "2026-06-24T19:00", "2026-06-24T20:00"],
    "temperature_2m": [33.0, 32.0, 30.0],
    "relative_humidity_2m": [60, 65, 70]
  }
}
```

- [ ] **Step 2: Write the failing test**

`tests/test_weather.py`:
```python
import json
from datetime import datetime
from pathlib import Path
from touchline.data.weather import estimate_wbgt, wbgt_from_payload

FIXTURE = Path(__file__).parent / "fixtures" / "openmeteo_sample.json"


def test_estimate_wbgt_hot_humid_is_high():
    w = estimate_wbgt(temp_c=33.0, rh_pct=60.0)
    assert 28 < w < 36           # dangerous-heat range


def test_estimate_wbgt_mild_is_lower():
    hot = estimate_wbgt(33.0, 60.0)
    mild = estimate_wbgt(18.0, 50.0)
    assert mild < hot


def test_wbgt_from_payload_picks_nearest_hour():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # kickoff 19:10 -> nearest hour is the 19:00 sample (T=32, RH=65)
    w = wbgt_from_payload(payload, datetime(2026, 6, 24, 19, 10))
    expected = estimate_wbgt(32.0, 65.0)
    assert abs(w - expected) < 1e-9


def test_wbgt_from_payload_returns_none_when_no_hours():
    assert wbgt_from_payload({"hourly": {"time": [], "temperature_2m": [],
                                         "relative_humidity_2m": []}},
                             datetime(2026, 6, 24, 19, 0)) is None
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_weather.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

`touchline/data/weather.py`:
```python
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def estimate_wbgt(temp_c: float, rh_pct: float) -> float:
    """Outdoor WBGT (°C) from air temperature and relative humidity (BoM approximation)."""
    e = (rh_pct / 100.0) * 6.105 * math.exp(17.27 * temp_c / (237.7 + temp_c))
    return 0.567 * temp_c + 0.393 * e + 3.94


def wbgt_from_payload(payload: dict, kickoff: datetime) -> float | None:
    """Estimate WBGT at the hour nearest kickoff from an Open-Meteo hourly payload."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    rhs = hourly.get("relative_humidity_2m", [])
    if not times:
        return None
    parsed = [datetime.fromisoformat(t) for t in times]
    i = min(range(len(parsed)), key=lambda k: abs((parsed[k] - kickoff).total_seconds()))
    return estimate_wbgt(float(temps[i]), float(rhs[i]))


def fetch_wbgt(
    lat: float, lon: float, kickoff: datetime, cache_dir: Path,
    client: httpx.Client | None = None,
) -> float | None:
    """Fetch the Open-Meteo forecast for the kickoff day and estimate WBGT (cached)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    day = kickoff.date().isoformat()
    cache_file = cache_dir / f"openmeteo_{lat:.2f}_{lon:.2f}_{day}.json"
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        owns = client is None
        client = client or httpx.Client(timeout=15.0)
        try:
            resp = client.get(OPEN_METEO_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m",
                "start_date": day, "end_date": day,
            })
            resp.raise_for_status()
            payload = resp.json()
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        finally:
            if owns:
                client.close()
    return wbgt_from_payload(payload, kickoff)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_weather.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add touchline/data/weather.py tests/test_weather.py tests/fixtures/openmeteo_sample.json
git commit -m "feat: WBGT estimate and cached Open-Meteo fetch"
```

---

## Task 5: Wire weather into build_context + `backtest` CLI

**Files:**
- Modify: `touchline/edge/context.py`
- Modify: `touchline/cli.py`
- Test: `tests/test_context_weather.py`

- [ ] **Step 1: Write the failing test**

`tests/test_context_weather.py`:
```python
from datetime import date
from touchline.edge.context import build_context


def test_wbgt_passes_through_to_factor_context():
    ctx = build_context("Mexico", "USA", date(2026, 6, 24),
                        venue_name="Estadio Azteca", history=[], wbgt_c=29.5)
    assert ctx.wbgt_c == 29.5


def test_wbgt_defaults_to_none():
    ctx = build_context("Mexico", "USA", date(2026, 6, 24),
                        venue_name="Estadio Azteca", history=[])
    assert ctx.wbgt_c is None


def test_unknown_venue_returns_neutral_context():
    # Historical venues are not in the 2026 table; must not crash.
    ctx = build_context("Brazil", "Chile", date(2014, 6, 28),
                        venue_name="Estadio Mineirao, Belo Horizonte", history=[])
    assert ctx.altitude_m == 0
    assert ctx.wbgt_c is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_context_weather.py -v`
Expected: FAIL — `test_wbgt_passes_through_to_factor_context` fails (unexpected `wbgt_c` kwarg) and `test_unknown_venue_returns_neutral_context` raises `KeyError`.

- [ ] **Step 3: Modify build_context in `touchline/edge/context.py`**

Replace the `build_context` function with:
```python
def build_context(
    home: str, away: str, when: date, venue_name: str, history: list[Match],
    wbgt_c: float | None = None,
) -> FactorContext:
    """Build a FactorContext for an upcoming fixture.

    `wbgt_c` is the heat estimate (None if unavailable). Unknown venue names (e.g.
    historical stadiums absent from the 2026 table) yield a neutral context rather
    than raising, so the same code path works for backtests over historical data."""
    try:
        venue = get_venue(venue_name)
    except KeyError:
        return FactorContext(wbgt_c=wbgt_c)
    travel_h, rest_h = _travel_and_rest(home, when, venue, history)
    travel_a, rest_a = _travel_and_rest(away, when, venue, history)
    return FactorContext(
        travel_km_home=travel_h,
        travel_km_away=travel_a,
        altitude_m=venue.altitude_m,
        home_altitude_acclimatized=is_host_country(home) and venue.country == home,
        away_altitude_acclimatized=is_host_country(away) and venue.country == away,
        wbgt_c=wbgt_c,
        rest_days_home=rest_h,
        rest_days_away=rest_a,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_context_weather.py tests/test_edge_context.py -v`
Expected: PASS (the 3 new + the 3 existing context tests).

- [ ] **Step 5: Add the `backtest` CLI command to `touchline/cli.py`**

Add imports near the top (integrate with existing import block):
```python
from touchline.backtest.harness import backtest as run_backtest
from touchline.backtest.calibrate import calibrate
```

Register the subcommand inside `main`:
```python
    bt_p = sub.add_parser("backtest", help="Score the model on completed matches")
    bt_p.add_argument("--eval-start", default="2018-01-01",
                      help="Only score matches on/after this ISO date")
    bt_p.add_argument("--half-life-days", type=float, default=540.0)
    bt_p.add_argument("--prior-weight", type=float, default=0.05)
    bt_p.add_argument("--calibrate", action="store_true",
                      help="Grid-search half-life x prior-weight")
```

Add the branch inside `main` (after `price`, before `return 1`):
```python
    if args.command == "backtest":
        db = Database(config.DB_PATH)
        db.init_schema()
        matches = db.all_matches()
        elo_path = config.CACHE_DIR / "elo.csv"
        elo = load_elo(elo_path) if elo_path.is_file() else EloTable()
        eval_start = date.fromisoformat(args.eval_start)
        if args.calibrate:
            res = calibrate(matches, eval_start=eval_start, elo=elo,
                            half_lifes=[180.0, 360.0, 540.0, 900.0],
                            prior_weights=[0.01, 0.05, 0.2, 0.5])
            print("Calibration grid (half_life, prior_weight, log_loss):")
            for hl, pw, ll in sorted(res.grid, key=lambda r: r[2]):
                print(f"  hl={hl:>6} pw={pw:<5} log_loss={ll:.4f}")
            print(f"Best: half_life={res.best_half_life}, "
                  f"prior_weight={res.best_prior_weight}, log_loss={res.best_log_loss:.4f}")
        else:
            res = run_backtest(matches, eval_start=eval_start,
                               half_life_days=args.half_life_days,
                               prior_weight=args.prior_weight, elo=elo)
            print(f"Backtest on {res.n_matches} matches "
                  f"(>= {args.eval_start}): Brier={res.brier:.4f}, "
                  f"log_loss={res.log_loss:.4f} "
                  f"(uniform baseline: Brier=0.6667, log_loss=1.0986).")
        return 0
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all green.

- [ ] **Step 7: Live smoke test**

```
.venv/Scripts/python.exe -m touchline.cli ingest
.venv/Scripts/python.exe -m touchline.cli backtest --eval-start 2018-01-01
```
Expected: prints `Backtest on N matches (>= 2018-01-01): Brier=..., log_loss=... (uniform baseline ...)`. Report the exact line. A healthy model has `log_loss < 1.0986` and `Brier < 0.6667` (beats uniform). If it does not beat the baseline, that is a real finding to report (not something to hide) — note it for review. Then optionally run `backtest --calibrate` and report the best params (this re-fits across a 4×4 grid and may take a minute).

- [ ] **Step 8: Commit**

```bash
git add touchline/edge/context.py touchline/cli.py tests/test_context_weather.py
git commit -m "feat: wire weather into build_context and add backtest CLI"
```

---

## Self-Review Notes

- **Spec coverage:** Brier + log-loss scoring ✓ (Task 1), walk-forward backtest replaying completed matches with pre-kickoff-only data ✓ (Task 2), hyperparameter calibration tuning half_life & prior_weight ✓ (Task 3), Open-Meteo heat estimate + fetch ✓ (Task 4), weather wired into the live `FactorContext` path ✓ (Task 5), plus the deferred hardening of `build_context` for unknown historical venues ✓ (Task 5). Comparison vs *closing Kalshi prices* (from the original spec) is intentionally NOT built: we have no historical Kalshi WC price series, so the harness scores against realized outcomes only — the meaningful, available signal. Factor-coefficient calibration (travel/heat/altitude/rest) is not included because we lack historical venue/weather data to fit it; those coefficients remain expert-set defaults from Plan 2, tuned manually if data becomes available.
- **Placeholder scan:** No TBD/TODO. Every code step has complete code. The unknown-venue neutral-context branch is intentional behavior, not a stub.
- **Type consistency:** `BacktestResult(n_matches, brier, log_loss)` consistent across `harness.py`, `calibrate.py` (via `result.log_loss`), `cli.py`, tests. `CalibrationResult(best_half_life, best_prior_weight, best_log_loss, grid)` consistent across `calibrate.py`, `cli.py`, tests. `backtest(matches, eval_start, half_life_days, prior_weight, elo)` and `calibrate(matches, eval_start, elo, half_lifes, prior_weights)` signatures match all call sites. `estimate_wbgt(temp_c, rh_pct)` / `wbgt_from_payload(payload, kickoff)` / `fetch_wbgt(lat, lon, kickoff, cache_dir, client)` consistent across `weather.py` and tests. `build_context(..., wbgt_c=None)` matches the Task 5 modification and the Plan 3 callers (which pass no `wbgt_c`, defaulting to None).
- **Open items flagged for execution (not placeholders):** the backtest re-fits per distinct eval date — on the full real dataset with `--calibrate` (16 grid points) this is the slowest operation in the project; the default `--eval-start 2018-01-01` bounds it to recent tournaments. If too slow, raise `--eval-start`. The live `fetch_wbgt` requires network (Open-Meteo, no key); it is not exercised in unit tests (only `estimate_wbgt`/`wbgt_from_payload` pure functions are). Wiring `fetch_wbgt` into the `price` CLI per-fixture is left as a one-line call the operator adds once the live 2026 schedule with venues + kickoff times is available (the venue lat/lon come from `venues.get_venue`).
```
