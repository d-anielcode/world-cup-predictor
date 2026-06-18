# Touchline Goal-Level Intercept + Multi-Market Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the model's systematic low-goals bias by fitting a global scoring intercept, and make the `backtest` harness permanently score all four market types (1X2, totals, BTTS, spread) with calibration — so improvements are measured, not eyeballed.

**Architecture:** Add a fitted `intercept` (log baseline goal rate) to the `Ratings` model so the average match's expected goals matches the observed level instead of being pinned by attack/defense centering. Extend `backtest/scoring.py` with binary-market scoring and the `backtest` harness to price the full `MarketProbs` and report per-market Brier vs base-rate plus calibration gap.

**Tech Stack:** Python 3.11 (sync), numpy, scipy, pytest. Builds on the existing `model/`, `backtest/`, `storage/`, and `cli.py` modules.

**Prerequisite:** Plans 1–4 merged to master (current `master` has the full pipeline + backtest CLI).

**Diagnosis driving this work:** real WC matches average **2.52** goals; the current model implies **2.37** (`exp(home_adv) + exp(0)`), because the centering penalty forces mean(attack)=mean(defense)=0 with no free level parameter. A global intercept absorbs the level.

---

## Conventions

- The model's expected goals become `log λ = intercept + attack[H] − defense[A] + (home_adv if host-home) ` and `log μ = intercept + attack[A] − defense[H]`. `intercept` defaults to `0.0` (so all existing behaviour/tests are unchanged until a fit sets it).
- Binary-market scoring uses, per market, the model probability of "yes" vs the realized 0/1 outcome:
  - **Over 2.5:** yes = `home_goals + away_goals > 2.5`
  - **BTTS:** yes = `home_goals ≥ 1 and away_goals ≥ 1`
  - **Home −1.5:** yes = `home_goals − away_goals > 1.5`
  - **Home win:** yes = `home_goals > away_goals`
- A market "beats base" when its Brier is below the base-rate predictor's Brier `p̄(1−p̄)`. The **calibration gap** = `mean(model_pred) − actual_rate` (near 0 = well calibrated; the bias fix should shrink the negative gap on totals/BTTS).

---

## File Structure

```
touchline/model/ratings.py    MODIFY: Ratings gains `intercept`; expected_goals uses it
touchline/model/fit.py        MODIFY: fit the intercept as a free parameter
touchline/storage/db.py       MODIFY: persist/load intercept in model_params
touchline/backtest/scoring.py MODIFY: binary_brier, base_rate_brier, calibration_gap
touchline/backtest/harness.py MODIFY: MarketScore + per-market scoring in backtest()
touchline/cli.py              MODIFY: backtest prints the multi-market table
tests/test_ratings.py         MODIFY: intercept behaviour
tests/test_fit.py             MODIFY: intercept recovers goal level
tests/test_ratings_storage.py MODIFY: intercept round-trips
tests/test_backtest_scoring.py MODIFY: binary scoring
tests/test_backtest_harness.py MODIFY: per-market results
```

Run all commands from `C:\Users\dcho0\Documents\touchline` via `.venv/Scripts/python.exe -m pytest ...`.

---

## Task 1: Global scoring intercept in the Ratings model

**Files:**
- Modify: `touchline/model/ratings.py`
- Test: `tests/test_ratings.py`

- [ ] **Step 1: Add the failing test** — append to `tests/test_ratings.py`:
```python
def test_intercept_lifts_both_expected_goals():
    import math
    from touchline.model.ratings import Ratings
    r0 = Ratings(attack={"A": 0.0}, defense={"A": 0.0}, home_adv=0.0, rho=-0.05)
    r1 = Ratings(attack={"A": 0.0}, defense={"A": 0.0}, home_adv=0.0, rho=-0.05,
                 intercept=0.2)
    lam0, mu0 = r0.expected_goals("A", "A", apply_home_adv=False)
    lam1, mu1 = r1.expected_goals("A", "A", apply_home_adv=False)
    assert lam1 > lam0 and mu1 > mu0
    assert abs(math.log(lam1) - math.log(lam0) - 0.2) < 1e-9


def test_intercept_defaults_to_zero():
    from touchline.model.ratings import Ratings
    r = Ratings(attack={"A": 0.0}, defense={"A": 0.0}, home_adv=0.0, rho=0.0)
    assert r.intercept == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ratings.py -v`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'intercept'`.

- [ ] **Step 3: Modify `touchline/model/ratings.py`**

Replace the dataclass body and `expected_goals` with:
```python
@dataclass
class Ratings:
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    intercept: float = 0.0   # log baseline goal rate (fitted; absorbs the overall level)

    def expected_goals(
        self, home: str, away: str, apply_home_adv: bool
    ) -> tuple[float, float]:
        """Return (lambda_home, mu_away) expected goals for a fixture.

        Unknown teams default to 0.0 attack/defense (league-average)."""
        ah, dh = self.attack.get(home, 0.0), self.defense.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defense.get(away, 0.0)
        log_lam = self.intercept + ah - da + (self.home_adv if apply_home_adv else 0.0)
        log_mu = self.intercept + aa - dh
        return math.exp(log_lam), math.exp(log_mu)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ratings.py -v`
Expected: PASS (existing 3 + new 2 = 5 passed). The existing tests pass because `intercept` defaults to 0.

- [ ] **Step 5: Commit**

```bash
git add touchline/model/ratings.py tests/test_ratings.py
git commit -m "feat: add fitted goal-level intercept to Ratings model"
```

---

## Task 2: Fit the intercept

**Files:**
- Modify: `touchline/model/fit.py`
- Test: `tests/test_fit.py`

- [ ] **Step 1: Add the failing test** — append to `tests/test_fit.py`:
```python
def test_fit_recovers_goal_level_via_intercept():
    # Generate high-scoring matches; the fitted model's average total should match.
    import numpy as np
    from datetime import date, timedelta
    from touchline.model.ratings import Ratings
    rng = np.random.default_rng(3)
    teams = ["A", "B", "C"]
    base = date(2026, 1, 1)
    matches = []
    base_rate = 1.8   # goals per side -> avg total ~3.6
    for i in range(400):
        h, a = rng.choice(teams, size=2, replace=False)
        matches.append(Match(
            match_date=base + timedelta(days=i % 120), home_team=str(h),
            away_team=str(a), home_goals=int(rng.poisson(base_rate)),
            away_goals=int(rng.poisson(base_rate)), competition="Syn", stage=None,
            venue=None, played=True, source="t"))
    r = fit_ratings(matches, EloTable(), half_life_days=400, prior_weight=0.01,
                    as_of=date(2026, 6, 1))
    # average matchup expected total under the fitted model
    lam, mu = r.expected_goals("A", "B", apply_home_adv=False)
    assert 3.0 < lam + mu < 4.2     # recovers the ~3.6 level (was pinned near 2.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit.py::test_fit_recovers_goal_level_via_intercept -v`
Expected: FAIL — the fitted total is pinned near 2.0 (no intercept), so `lam + mu` is well below 3.0.

- [ ] **Step 3: Modify `touchline/model/fit.py`**

Make these four edits inside `fit_ratings`:

(a) Replace the `unpack` helper:
```python
    def unpack(p):
        attack = p[:n]
        defense = p[n:2 * n]
        home_adv = p[2 * n]
        rho = p[2 * n + 1]
        intercept = p[2 * n + 2]
        return attack, defense, home_adv, rho, intercept
```

(b) Replace the body of `neg_log_lik` down to the `weighted =` line:
```python
    def neg_log_lik(p):
        attack, defense, home_adv, rho, intercept = unpack(p)
        log_lam = intercept + attack[hi] - defense[ai] + home_adv
        log_mu = intercept + attack[ai] - defense[hi]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)
        ll = hg * log_lam - lam + ag * log_mu - mu  # Poisson (drop constant log k!)
```
(Leave the tau correction, ridge, center, and `return -weighted + ridge + center` lines unchanged below this.)

(c) Replace the `x0` / `bounds` / `minimize` block:
```python
    mean_goals = float((hg.mean() + ag.mean()) / 2) if len(played) else 1.3
    intercept0 = math.log(max(mean_goals, 0.1))
    x0 = np.concatenate([prior, prior, [0.25], [-0.05], [intercept0]])
    bounds = ([(None, None)] * (2 * n + 1) + [(-_RHO_BOUND, _RHO_BOUND)]
              + [(None, None)])
    res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        warnings.warn(f"fit_ratings: optimizer did not converge: {res.message}")
    attack, defense, home_adv, rho, intercept = unpack(res.x)
```

(d) Replace the returned `Ratings(...)`:
```python
    return Ratings(
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        home_adv=float(home_adv),
        rho=float(rho),
        intercept=float(intercept),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit.py -v`
Expected: PASS (existing fit tests + the new goal-level test). The existing strength-ordering and Elo-prior tests still pass (the intercept is orthogonal to relative ratings).

- [ ] **Step 5: Commit**

```bash
git add touchline/model/fit.py tests/test_fit.py
git commit -m "feat: fit global goal-level intercept (corrects low-goals bias)"
```

---

## Task 3: Persist the intercept

**Files:**
- Modify: `touchline/storage/db.py`
- Test: `tests/test_ratings_storage.py`

- [ ] **Step 1: Add the failing test** — append to `tests/test_ratings_storage.py`:
```python
def test_intercept_roundtrips(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    db.save_ratings(Ratings(attack={"USA": 0.3}, defense={"USA": 0.2},
                            home_adv=0.27, rho=-0.06, intercept=0.15))
    loaded = db.load_ratings()
    assert abs(loaded.intercept - 0.15) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ratings_storage.py::test_intercept_roundtrips -v`
Expected: FAIL — `loaded.intercept` is 0.0 (not persisted).

- [ ] **Step 3: Modify `touchline/storage/db.py`**

In `save_ratings`, add `intercept` to the params written:
```python
            conn.executemany(
                "INSERT INTO model_params (key, value) VALUES (?,?)",
                [("home_adv", ratings.home_adv), ("rho", ratings.rho),
                 ("intercept", ratings.intercept)],
            )
```

In `load_ratings`, add the intercept to the returned `Ratings`:
```python
        return Ratings(
            attack={r["team"]: r["attack"] for r in rows},
            defense={r["team"]: r["defense"] for r in rows},
            home_adv=float(params.get("home_adv", 0.0)),
            rho=float(params.get("rho", 0.0)),
            intercept=float(params.get("intercept", 0.0)),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ratings_storage.py -v`
Expected: PASS (existing storage tests + the new intercept test).

- [ ] **Step 5: Commit**

```bash
git add touchline/storage/db.py tests/test_ratings_storage.py
git commit -m "feat: persist the goal-level intercept with ratings"
```

---

## Task 4: Binary-market scoring functions

**Files:**
- Modify: `touchline/backtest/scoring.py`
- Test: `tests/test_backtest_scoring.py`

- [ ] **Step 1: Add the failing test** — append to `tests/test_backtest_scoring.py`:
```python
def test_binary_brier_and_base():
    from touchline.backtest.scoring import binary_brier, base_rate_brier
    preds = [0.6, 0.6, 0.6, 0.6]
    actuals = [1.0, 1.0, 0.0, 0.0]   # base rate 0.5
    assert abs(binary_brier(preds, actuals) - 0.26) < 1e-9   # mean((0.6-y)^2)
    assert abs(base_rate_brier(actuals) - 0.25) < 1e-9       # 0.5*0.5


def test_calibration_gap_sign():
    from touchline.backtest.scoring import calibration_gap
    # model predicts higher than reality -> positive gap
    assert abs(calibration_gap([0.6, 0.6], [0.0, 1.0]) - 0.1) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_scoring.py -v`
Expected: FAIL — `ImportError` for `binary_brier`.

- [ ] **Step 3: Append to `touchline/backtest/scoring.py`**
```python
def binary_brier(preds: list[float], actuals: list[float]) -> float:
    """Mean squared error of binary probability predictions."""
    if not preds:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(preds, actuals)) / len(preds)


def base_rate_brier(actuals: list[float]) -> float:
    """Brier of the constant base-rate predictor (the bar a useful model must beat)."""
    if not actuals:
        return 0.0
    base = sum(actuals) / len(actuals)
    return base * (1 - base)


def calibration_gap(preds: list[float], actuals: list[float]) -> float:
    """mean(prediction) - actual_rate. Near 0 = well calibrated."""
    if not preds:
        return 0.0
    return sum(preds) / len(preds) - sum(actuals) / len(actuals)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_scoring.py -v`
Expected: PASS (existing 5 + new 2 = 7 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/backtest/scoring.py tests/test_backtest_scoring.py
git commit -m "feat: binary-market scoring (Brier, base-rate, calibration gap)"
```

---

## Task 5: Multi-market backtest harness + CLI

**Files:**
- Modify: `touchline/backtest/harness.py`, `touchline/cli.py`
- Test: `tests/test_backtest_harness.py`

- [ ] **Step 1: Add the failing test** — append to `tests/test_backtest_harness.py`:
```python
def test_backtest_reports_per_market_scores():
    from touchline.backtest.harness import MarketScore
    matches = _synthetic()
    r = backtest(matches, eval_start=date(2026, 1, 1), half_life_days=400,
                 prior_weight=0.05, elo=EloTable())
    assert set(r.markets) == {"over2.5", "btts", "home_-1.5", "home_win"}
    for ms in r.markets.values():
        assert isinstance(ms, MarketScore)
        assert ms.n == r.n_matches
        assert 0.0 <= ms.brier <= 1.0
        assert 0.0 <= ms.accuracy <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_harness.py::test_backtest_reports_per_market_scores -v`
Expected: FAIL — `BacktestResult` has no `markets` / no `MarketScore`.

- [ ] **Step 3: Modify `touchline/backtest/harness.py`**

Replace the imports, dataclass, and the scoring section of `backtest`. New top of file:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from touchline.data.elo import EloTable
from touchline.models import Match
from touchline.model.fit import fit_ratings
from touchline.model.dixon_coles import scoreline_matrix
from touchline.model.pricing import price_matrix
from touchline.backtest.scoring import (
    brier_score, log_loss, outcome_index, binary_brier, base_rate_brier, calibration_gap,
)


@dataclass
class MarketScore:
    n: int
    brier: float
    base_brier: float
    calibration_gap: float
    accuracy: float


@dataclass
class BacktestResult:
    n_matches: int
    brier: float
    log_loss: float
    markets: dict[str, MarketScore] = field(default_factory=dict)
```

Replace the body of `backtest` from `fits: dict...` to the end with:
```python
    fits: dict[date, object] = {}
    probs: list[tuple[float, float, float]] = []
    outcomes: list[int] = []
    # per-market (prediction, realized 0/1)
    bins: dict[str, tuple[list[float], list[float]]] = {
        "over2.5": ([], []), "btts": ([], []), "home_-1.5": ([], []), "home_win": ([], []),
    }
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
        p = price_matrix(matrix, total_lines=[2.5], handicap_lines=[-1.5])
        probs.append((p.home, p.draw, p.away))
        outcomes.append(outcome_index(m.home_goals, m.away_goals))
        bins["over2.5"][0].append(p.over[2.5])
        bins["over2.5"][1].append(1.0 if m.home_goals + m.away_goals > 2.5 else 0.0)
        bins["btts"][0].append(p.btts_yes)
        bins["btts"][1].append(1.0 if m.home_goals >= 1 and m.away_goals >= 1 else 0.0)
        bins["home_-1.5"][0].append(p.home_handicap[-1.5])
        bins["home_-1.5"][1].append(1.0 if m.home_goals - m.away_goals > 1.5 else 0.0)
        bins["home_win"][0].append(p.home)
        bins["home_win"][1].append(1.0 if m.home_goals > m.away_goals else 0.0)

    markets: dict[str, MarketScore] = {}
    for name, (preds, acts) in bins.items():
        acc = (sum((pr > 0.5) == bool(y) for pr, y in zip(preds, acts)) / len(preds)
               if preds else 0.0)
        markets[name] = MarketScore(
            n=len(preds), brier=binary_brier(preds, acts),
            base_brier=base_rate_brier(acts), calibration_gap=calibration_gap(preds, acts),
            accuracy=acc,
        )
    return BacktestResult(
        n_matches=len(probs),
        brier=brier_score(probs, outcomes),
        log_loss=log_loss(probs, outcomes),
        markets=markets,
    )
```
(Delete the now-unused `prob_1x2` import if present — `price_matrix` replaces it.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_backtest_harness.py -v`
Expected: PASS (existing 3 + new 1 = 4 passed).

- [ ] **Step 5: Extend the `backtest` CLI print in `touchline/cli.py`**

In the `else` branch of the `backtest` command (the non-calibrate path), after the existing `print(f"Backtest on ...")` line, add:
```python
            print("  Market calibration (model_avg vs actual; Brier vs base):")
            for name, ms in res.markets.items():
                verdict = "beats" if ms.brier < ms.base_brier else "below"
                print(f"    {name:10s} acc={ms.accuracy:.0%} "
                      f"calib_gap={ms.calibration_gap:+.3f} "
                      f"Brier={ms.brier:.4f} (base {ms.base_brier:.4f}) {verdict}")
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all green.

- [ ] **Step 7: Live smoke test — confirm the intercept fix shrank the bias**

```
.venv/Scripts/python.exe -m touchline.cli ingest
.venv/Scripts/python.exe -m touchline.cli fit-ratings
.venv/Scripts/python.exe -m touchline.cli backtest --eval-start 2018-01-01
```
Report the full output. SUCCESS CRITERIA: the `over2.5` and `btts` calibration gaps should be **closer to 0** than the pre-intercept ~−0.03 (the model previously predicted ~3 points under actual). Report the actual gaps. If the 1X2 log-loss regressed materially, report it (the intercept should not hurt 1X2 — it is orthogonal to relative strength).

- [ ] **Step 8: Commit**

```bash
git add touchline/backtest/harness.py touchline/cli.py tests/test_backtest_harness.py
git commit -m "feat: multi-market backtest scoring + calibration in harness and CLI"
```

---

## Self-Review Notes

- **Spec coverage:** goal-level intercept fixing the low-goals bias ✓ (Tasks 1–3), multi-market backtest scoring permanently in the harness + CLI ✓ (Tasks 4–5), calibration reporting (the gap that revealed the bias) ✓ (Task 5). Validation on the current 320-match dataset is the Task 5 smoke test; data expansion (more internationals) is intentionally NOT in scope per the user's decision.
- **Placeholder scan:** No TBD/TODO; every code step is complete. The smoke test's success criterion is a concrete numeric expectation (gaps closer to 0), not a vague instruction.
- **Type consistency:** `Ratings(..., intercept=0.0)` is threaded through `ratings.py`, `fit.py` (param vector index `2n+2`), `db.py` (`model_params`), and tests consistently. `MarketScore(n, brier, base_brier, calibration_gap, accuracy)` and `BacktestResult(n_matches, brier, log_loss, markets)` are consistent across `harness.py`, `cli.py`, and tests. Binary scoring signatures `binary_brier(preds, actuals)`, `base_rate_brier(actuals)`, `calibration_gap(preds, actuals)` match all call sites. The harness market keys (`over2.5`, `btts`, `home_-1.5`, `home_win`) match the test assertion exactly.
- **Open items flagged (not placeholders):** the intercept slightly changes every fitted rating's absolute level (relative orderings unchanged), so the live `fit-ratings` output will now show a non-zero intercept; re-running `backtest --calibrate` afterwards is optional (the calibrated half_life=900 still holds — the intercept is orthogonal to the decay/prior hyperparameters).
```
