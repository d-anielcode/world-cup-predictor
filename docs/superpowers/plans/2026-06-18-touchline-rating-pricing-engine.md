# Touchline Rating & Pricing Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the ingested match data (Plan 1) into per-fixture market probabilities — match winner (1X2), totals (O/U), both-teams-to-score, and spread/handicap — using a Dixon-Coles bivariate-Poisson goal model with Elo-seeded, time-decayed ratings plus deterministic adjustment factors.

**Architecture:** A `touchline/model/` package. Ratings are fit once by weighted MLE (`fit.py`) and stored in SQLite. For any fixture, `price_fixture.py` turns ratings → expected goals → adjusted expected goals (`factors.py`) → a Dixon-Coles scoreline probability matrix (`dixon_coles.py`) → market probabilities (`pricing.py`). Everything is pure/deterministic and unit-tested; no network and no LLM.

**Tech Stack:** Python 3.11 (sync), numpy, scipy (Poisson pmf + L-BFGS-B optimizer), pytest. Builds on Plan 1 modules (`touchline.models.Match`, `touchline.storage.db.Database`, `touchline.data.elo`, `touchline.data.venues`).

**Prerequisite:** Plan 1 (`plan-1-data-foundation`) is merged to the working branch. This plan assumes `touchline/models.py`, `touchline/storage/db.py`, `touchline/data/venues.py`, and `touchline/data/elo.py` exist as delivered in Plan 1.

---

## Conventions (read once before starting)

**Goal-model parameterization.** Each team `t` has an **attack** rating `a[t]` and a **defense** rating `d[t]` (both on a log scale, centered near 0; higher attack = scores more, higher defense = concedes fewer). A global **home_adv** `h` and Dixon-Coles low-score correlation **rho** `ρ`. For a fixture (home `H` vs away `A`):

```
log λ_home = a[H] - d[A] + (h if apply_home_adv else 0)
log μ_away = a[A] - d[H]
```

**Dixon-Coles correction** `τ(x, y)` adjusts the four lowest scorelines:

```
τ(0,0) = 1 - λ·μ·ρ
τ(0,1) = 1 + λ·ρ
τ(1,0) = 1 + μ·ρ
τ(1,1) = 1 - ρ
τ(x,y) = 1   otherwise
```

**Scoreline matrix** `M[x][y] = τ(x,y) · Poisson(x; λ) · Poisson(y; μ)`, for `x, y` in `0..max_goals` (default 10), then normalized so it sums to 1.

**Time decay.** A match `d` days before the as-of date gets weight `exp(-ln2/half_life · d)` (recent ≈ 1, old → 0).

**Elo prior (ridge).** Map each team's Elo to a strength `s[t] = (elo[t] - mean_elo)/400`. The fitter adds `prior_weight · Σ((a[t]-s[t])² + (d[t]-s[t])²)` to the loss, pulling thin-data teams toward their Elo strength.

---

## File Structure

```
touchline/model/
├── __init__.py
├── dixon_coles.py     # tau(), scoreline_matrix()
├── ratings.py         # Ratings dataclass + expected_goals()
├── fit.py             # fit_ratings(): weighted MLE + Elo ridge
├── factors.py         # FactorContext + adjust_expected_goals()
├── pricing.py         # MarketProbs + market prob functions from a matrix
└── price_fixture.py   # orchestrate ratings + factors -> MarketProbs
tests/
├── test_dixon_coles.py
├── test_ratings.py
├── test_fit.py
├── test_factors.py
├── test_pricing.py
└── test_price_fixture.py
```
Also extends `touchline/storage/db.py` (ratings persistence) and `touchline/cli.py` (`fit-ratings` command).

Run all commands from `C:\Users\dcho0\Documents\touchline` using the venv: `.venv/Scripts/python.exe -m pytest ...`.

---

## Task 0: Add numpy/scipy dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `touchline/model/__init__.py`

- [ ] **Step 1: Append to requirements.txt**

Add these two lines (keep existing lines):
```
numpy==2.1.1
scipy==1.14.1
```

- [ ] **Step 2: Install**

Run: `.venv/Scripts/python.exe -m pip install -r requirements.txt`
Expected: numpy and scipy install successfully.

- [ ] **Step 3: Create empty package marker**

Create `touchline/model/__init__.py` as an empty file.

- [ ] **Step 4: Verify imports**

Run: `.venv/Scripts/python.exe -c "import numpy, scipy.optimize, scipy.stats; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt touchline/model/__init__.py
git commit -m "chore: add numpy/scipy for the model package"
```

---

## Task 1: Dixon-Coles core (tau + scoreline matrix)

**Files:**
- Create: `touchline/model/dixon_coles.py`
- Test: `tests/test_dixon_coles.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dixon_coles.py`:
```python
import numpy as np
from touchline.model.dixon_coles import tau, scoreline_matrix


def test_tau_low_score_corrections():
    lam, mu, rho = 1.3, 1.1, -0.05
    assert tau(0, 0, lam, mu, rho) == 1 - lam * mu * rho
    assert tau(0, 1, lam, mu, rho) == 1 + lam * rho
    assert tau(1, 0, lam, mu, rho) == 1 + mu * rho
    assert tau(1, 1, lam, mu, rho) == 1 - rho


def test_tau_is_one_outside_low_scores():
    assert tau(2, 3, 1.0, 1.0, -0.05) == 1.0
    assert tau(0, 2, 1.0, 1.0, -0.05) == 1.0


def test_scoreline_matrix_sums_to_one():
    m = scoreline_matrix(1.5, 1.2, -0.05, max_goals=10)
    assert m.shape == (11, 11)
    assert abs(m.sum() - 1.0) < 1e-9


def test_scoreline_matrix_rho_zero_is_independent_poisson():
    # With rho=0, P(x,y) factorizes into the Poisson marginals.
    from scipy.stats import poisson
    lam, mu = 1.4, 0.9
    m = scoreline_matrix(lam, mu, 0.0, max_goals=15)
    # P(home scores 1) should match Poisson(1; lam) closely
    assert abs(m[1, :].sum() - poisson.pmf(1, lam)) < 1e-4
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dixon_coles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'touchline.model.dixon_coles'`.

- [ ] **Step 3: Implement dixon_coles.py**

`touchline/model/dixon_coles.py`:
```python
from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score correlation correction."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def scoreline_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Return a normalized (max_goals+1) x (max_goals+1) matrix of P(home=x, away=y)."""
    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals, lam)   # P(home scores x)
    away_pmf = poisson.pmf(goals, mu)    # P(away scores y)
    m = np.outer(home_pmf, away_pmf)     # independent part
    # Apply the four low-score corrections.
    m[0, 0] *= tau(0, 0, lam, mu, rho)
    m[0, 1] *= tau(0, 1, lam, mu, rho)
    m[1, 0] *= tau(1, 0, lam, mu, rho)
    m[1, 1] *= tau(1, 1, lam, mu, rho)
    total = m.sum()
    return m / total if total > 0 else m
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dixon_coles.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/model/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat: Dixon-Coles tau and scoreline probability matrix"
```

---

## Task 2: Ratings dataclass + expected goals

**Files:**
- Create: `touchline/model/ratings.py`
- Test: `tests/test_ratings.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ratings.py`:
```python
import math
from touchline.model.ratings import Ratings


def _ratings():
    return Ratings(
        attack={"Brazil": 0.5, "Bolivia": -0.4},
        defense={"Brazil": 0.3, "Bolivia": -0.2},
        home_adv=0.25,
        rho=-0.05,
    )


def test_expected_goals_neutral_site():
    r = _ratings()
    lam, mu = r.expected_goals("Brazil", "Bolivia", apply_home_adv=False)
    assert abs(lam - math.exp(0.5 - (-0.2))) < 1e-9   # a[H] - d[A]
    assert abs(mu - math.exp(-0.4 - 0.3)) < 1e-9       # a[A] - d[H]


def test_home_advantage_increases_home_lambda():
    r = _ratings()
    lam_n, _ = r.expected_goals("Brazil", "Bolivia", apply_home_adv=False)
    lam_h, _ = r.expected_goals("Brazil", "Bolivia", apply_home_adv=True)
    assert lam_h > lam_n
    assert abs(math.log(lam_h) - math.log(lam_n) - 0.25) < 1e-9


def test_unknown_team_uses_default_zero_rating():
    r = _ratings()
    lam, mu = r.expected_goals("Brazil", "Atlantis", apply_home_adv=False)
    # Atlantis attack/defense default to 0.0
    assert abs(lam - math.exp(0.5 - 0.0)) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ratings.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement ratings.py**

`touchline/model/ratings.py`:
```python
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Ratings:
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0

    def expected_goals(
        self, home: str, away: str, apply_home_adv: bool
    ) -> tuple[float, float]:
        """Return (lambda_home, mu_away) expected goals for a fixture.

        Unknown teams default to 0.0 attack/defense (league-average).
        """
        ah, dh = self.attack.get(home, 0.0), self.defense.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defense.get(away, 0.0)
        log_lam = ah - da + (self.home_adv if apply_home_adv else 0.0)
        log_mu = aa - dh
        return math.exp(log_lam), math.exp(log_mu)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ratings.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/model/ratings.py tests/test_ratings.py
git commit -m "feat: Ratings dataclass with expected-goals model"
```

---

## Task 3: Market pricing from a scoreline matrix

**Files:**
- Create: `touchline/model/pricing.py`
- Test: `tests/test_pricing.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pricing.py`:
```python
import numpy as np
from touchline.model.pricing import (
    prob_1x2, prob_over, prob_btts, prob_home_handicap, price_matrix, MarketProbs,
)


def _diagonal_matrix():
    # 4x4 matrix: only three outcomes with known probabilities.
    m = np.zeros((4, 4))
    m[2, 0] = 0.5   # home 2-0  -> home win, over 1.5, no BTTS, margin +2
    m[1, 1] = 0.3   # 1-1       -> draw, over 1.5, BTTS yes, margin 0
    m[0, 1] = 0.2   # 0-1       -> away win, under 1.5, no BTTS, margin -1
    return m


def test_prob_1x2_partitions_outcomes():
    m = _diagonal_matrix()
    home, draw, away = prob_1x2(m)
    assert abs(home - 0.5) < 1e-9
    assert abs(draw - 0.3) < 1e-9
    assert abs(away - 0.2) < 1e-9


def test_prob_over_line():
    m = _diagonal_matrix()
    # totals: 2-0=2 (>1.5), 1-1=2 (>1.5), 0-1=1 (<1.5)
    assert abs(prob_over(m, 1.5) - 0.8) < 1e-9
    assert abs(prob_over(m, 2.5) - 0.0) < 1e-9   # no scoreline totals > 2.5


def test_prob_btts():
    m = _diagonal_matrix()
    assert abs(prob_btts(m) - 0.3) < 1e-9   # only 1-1 has both scoring


def test_prob_home_handicap():
    m = _diagonal_matrix()
    # home covers -1.5 if margin > 1.5: only 2-0 (margin 2)
    assert abs(prob_home_handicap(m, -1.5) - 0.5) < 1e-9
    # home covers +0.5 if margin > -0.5: 2-0 and 1-1
    assert abs(prob_home_handicap(m, 0.5) - 0.8) < 1e-9


def test_price_matrix_returns_marketprobs():
    m = _diagonal_matrix()
    probs = price_matrix(m, total_lines=[1.5, 2.5], handicap_lines=[-1.5, 0.5])
    assert isinstance(probs, MarketProbs)
    assert abs(probs.home - 0.5) < 1e-9
    assert abs(probs.over[1.5] - 0.8) < 1e-9
    assert abs(probs.btts_yes - 0.3) < 1e-9
    assert abs(probs.home_handicap[-1.5] - 0.5) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pricing.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement pricing.py**

`touchline/model/pricing.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_TOTAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
DEFAULT_HANDICAP_LINES = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]


@dataclass
class MarketProbs:
    home: float
    draw: float
    away: float
    btts_yes: float
    over: dict[float, float]            # total line -> P(total goals > line)
    home_handicap: dict[float, float]   # line -> P(home_goals - away_goals > line)


def _margin_grid(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    x = np.arange(n)[:, None]
    y = np.arange(n)[None, :]
    return x - y   # home margin per cell


def _total_grid(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    x = np.arange(n)[:, None]
    y = np.arange(n)[None, :]
    return x + y


def prob_1x2(m: np.ndarray) -> tuple[float, float, float]:
    margin = _margin_grid(m)
    home = float(m[margin > 0].sum())
    draw = float(m[margin == 0].sum())
    away = float(m[margin < 0].sum())
    return home, draw, away


def prob_over(m: np.ndarray, line: float) -> float:
    return float(m[_total_grid(m) > line].sum())


def prob_btts(m: np.ndarray) -> float:
    return float(m[1:, 1:].sum())


def prob_home_handicap(m: np.ndarray, line: float) -> float:
    """P(home_goals - away_goals > line). Negative line => home giving a handicap."""
    return float(m[_margin_grid(m) > line].sum())


def price_matrix(
    m: np.ndarray,
    total_lines: list[float] | None = None,
    handicap_lines: list[float] | None = None,
) -> MarketProbs:
    total_lines = total_lines if total_lines is not None else DEFAULT_TOTAL_LINES
    handicap_lines = handicap_lines if handicap_lines is not None else DEFAULT_HANDICAP_LINES
    home, draw, away = prob_1x2(m)
    return MarketProbs(
        home=home,
        draw=draw,
        away=away,
        btts_yes=prob_btts(m),
        over={ln: prob_over(m, ln) for ln in total_lines},
        home_handicap={ln: prob_home_handicap(m, ln) for ln in handicap_lines},
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pricing.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/model/pricing.py tests/test_pricing.py
git commit -m "feat: market probabilities (1X2, totals, BTTS, handicap) from matrix"
```

---

## Task 4: Adjustment factors

**Files:**
- Create: `touchline/model/factors.py`
- Test: `tests/test_factors.py`

The model's `home_adv` is applied separately (only when a host plays at home — see Task 6). These factors are multiplicative adjustments to expected goals for travel, altitude, heat, and rest. Coefficients are module constants with conservative defaults; they are tuned in Plan 3's backtest.

- [ ] **Step 1: Write the failing test**

`tests/test_factors.py`:
```python
from touchline.model.factors import FactorContext, adjust_expected_goals


def test_no_context_is_identity():
    lam, mu = adjust_expected_goals(1.5, 1.2, FactorContext())
    assert (round(lam, 9), round(mu, 9)) == (1.5, 1.2)


def test_travel_reduces_travelling_team_goals():
    ctx = FactorContext(travel_km_home=3000.0)
    lam, mu = adjust_expected_goals(1.5, 1.2, ctx)
    assert lam < 1.5      # home travelled far -> fewer home goals
    assert mu == 1.2      # away unaffected


def test_heat_reduces_both_teams_goals():
    hot = FactorContext(wbgt_c=30.0)
    lam, mu = adjust_expected_goals(1.5, 1.2, hot)
    assert lam < 1.5 and mu < 1.2
    mild = FactorContext(wbgt_c=20.0)
    lam2, mu2 = adjust_expected_goals(1.5, 1.2, mild)
    assert (round(lam2, 9), round(mu2, 9)) == (1.5, 1.2)   # below threshold: no effect


def test_altitude_penalizes_unacclimatized_team_only():
    ctx = FactorContext(altitude_m=2240, away_altitude_acclimatized=False,
                        home_altitude_acclimatized=True)
    lam, mu = adjust_expected_goals(1.5, 1.2, ctx)
    assert lam == 1.5     # home acclimatized -> unaffected
    assert mu < 1.2       # away not acclimatized -> reduced


def test_rest_disadvantage_reduces_more_tired_team():
    ctx = FactorContext(rest_days_home=2, rest_days_away=6)
    lam, mu = adjust_expected_goals(1.5, 1.2, ctx)
    assert lam < 1.5      # home more tired
    assert mu == 1.2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement factors.py**

`touchline/model/factors.py`:
```python
from __future__ import annotations

import math
from dataclasses import dataclass

# Tunable coefficients (defaults; calibrated in Plan 3 backtest).
TRAVEL_PER_1000KM = 0.03     # fractional goal reduction per 1000 km travelled
HEAT_WBGT_THRESHOLD = 26.0   # WBGT (C) above which scoring is suppressed
HEAT_PER_DEG = 0.02          # fractional reduction per degree above threshold
ALTITUDE_THRESHOLD_M = 1500  # altitude above which unacclimatized teams tire
ALTITUDE_PER_1000M = 0.06    # fractional reduction per 1000 m above threshold
REST_PER_DAY = 0.02          # fractional reduction per day of rest deficit vs opponent


@dataclass
class FactorContext:
    travel_km_home: float = 0.0
    travel_km_away: float = 0.0
    altitude_m: int = 0
    home_altitude_acclimatized: bool = False
    away_altitude_acclimatized: bool = False
    wbgt_c: float | None = None
    rest_days_home: int | None = None
    rest_days_away: int | None = None


def _travel_mult(km: float) -> float:
    return math.exp(-TRAVEL_PER_1000KM * km / 1000.0)


def _altitude_mult(altitude_m: int, acclimatized: bool) -> float:
    if acclimatized or altitude_m <= ALTITUDE_THRESHOLD_M:
        return 1.0
    excess = (altitude_m - ALTITUDE_THRESHOLD_M) / 1000.0
    return math.exp(-ALTITUDE_PER_1000M * excess)


def _heat_mult(wbgt_c: float | None) -> float:
    if wbgt_c is None or wbgt_c <= HEAT_WBGT_THRESHOLD:
        return 1.0
    return math.exp(-HEAT_PER_DEG * (wbgt_c - HEAT_WBGT_THRESHOLD))


def _rest_deficit_mult(own: int | None, other: int | None) -> float:
    if own is None or other is None or own >= other:
        return 1.0
    return math.exp(-REST_PER_DAY * (other - own))


def adjust_expected_goals(
    lam: float, mu: float, ctx: FactorContext
) -> tuple[float, float]:
    """Apply travel/altitude/heat/rest multipliers to expected goals."""
    heat = _heat_mult(ctx.wbgt_c)  # affects both sides
    lam *= _travel_mult(ctx.travel_km_home)
    lam *= _altitude_mult(ctx.altitude_m, ctx.home_altitude_acclimatized)
    lam *= _rest_deficit_mult(ctx.rest_days_home, ctx.rest_days_away)
    lam *= heat
    mu *= _travel_mult(ctx.travel_km_away)
    mu *= _altitude_mult(ctx.altitude_m, ctx.away_altitude_acclimatized)
    mu *= _rest_deficit_mult(ctx.rest_days_away, ctx.rest_days_home)
    mu *= heat
    return lam, mu
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_factors.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/model/factors.py tests/test_factors.py
git commit -m "feat: deterministic adjustment factors (travel/altitude/heat/rest)"
```

---

## Task 5: Fit ratings by weighted MLE with Elo prior

**Files:**
- Create: `touchline/model/fit.py`
- Test: `tests/test_fit.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fit.py`:
```python
from datetime import date, timedelta

import numpy as np

from touchline.models import Match
from touchline.data.elo import EloTable
from touchline.model.fit import fit_ratings


def _synthetic_matches(seed: int = 0) -> list[Match]:
    """Generate matches from known strengths: Strong >> Mid >> Weak."""
    rng = np.random.default_rng(seed)
    true_attack = {"Strong": 0.6, "Mid": 0.0, "Weak": -0.6}
    true_defense = {"Strong": 0.5, "Mid": 0.0, "Weak": -0.5}
    teams = list(true_attack)
    base = date(2026, 1, 1)
    matches: list[Match] = []
    for i in range(300):
        h, a = rng.choice(teams, size=2, replace=False)
        lam = np.exp(true_attack[h] - true_defense[a])
        mu = np.exp(true_attack[a] - true_defense[h])
        matches.append(Match(
            match_date=base + timedelta(days=i % 120),
            home_team=str(h), away_team=str(a),
            home_goals=int(rng.poisson(lam)), away_goals=int(rng.poisson(mu)),
            competition="Synthetic", stage=None, venue=None,
            played=True, source="test",
        ))
    return matches


def test_fit_recovers_strength_ordering():
    matches = _synthetic_matches()
    elo = EloTable()  # empty -> all teams default 1500, prior is neutral
    r = fit_ratings(matches, elo, half_life_days=400, prior_weight=0.01,
                    as_of=date(2026, 6, 1))
    assert r.attack["Strong"] > r.attack["Mid"] > r.attack["Weak"]
    assert r.defense["Strong"] > r.defense["Weak"]


def test_unplayed_matches_are_ignored():
    matches = _synthetic_matches()
    matches.append(Match(
        match_date=date(2026, 6, 1), home_team="Strong", away_team="Weak",
        home_goals=None, away_goals=None, competition="Synthetic", stage=None,
        venue=None, played=False, source="test",
    ))
    r = fit_ratings(matches, EloTable(), half_life_days=400, prior_weight=0.01,
                    as_of=date(2026, 6, 1))
    # Should still fit without crashing on the None-goal row.
    assert "Strong" in r.attack


def test_elo_prior_dominates_for_team_with_no_games():
    # "Ghost" plays no matches; a strong Elo prior should give it high ratings.
    matches = _synthetic_matches()
    elo = EloTable(by_norm={"strong": 1500.0, "mid": 1500.0, "weak": 1500.0,
                            "ghost": 2300.0})
    # Register Ghost in the team universe via one unplayed fixture.
    r = fit_ratings(matches, elo, half_life_days=400, prior_weight=5.0,
                    as_of=date(2026, 6, 1), extra_teams=["Ghost"])
    assert r.attack["Ghost"] > r.attack["Weak"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement fit.py**

`touchline/model/fit.py`:
```python
from __future__ import annotations

import math
from datetime import date

import numpy as np
from scipy.optimize import minimize

from touchline.data.elo import EloTable
from touchline.models import Match
from touchline.model.dixon_coles import tau
from touchline.model.ratings import Ratings

_ELO_SCALE = 400.0
_CENTER_PENALTY = 100.0   # soft identifiability constraint (mean attack/defense = 0)


def _decay_weight(match_day: date, as_of: date, half_life_days: float) -> float:
    age = max((as_of - match_day).days, 0)
    return math.exp(-math.log(2) / half_life_days * age)


def fit_ratings(
    matches: list[Match],
    elo: EloTable,
    half_life_days: float,
    prior_weight: float,
    as_of: date,
    extra_teams: list[str] | None = None,
    max_goals: int = 10,
) -> Ratings:
    """Fit Dixon-Coles attack/defense ratings by time-weighted MLE with an Elo ridge.

    Only played matches contribute to the likelihood. `extra_teams` lets callers
    include teams that have no played matches yet (priced purely from the Elo prior).
    """
    played = [m for m in matches if m.played and m.home_goals is not None
              and m.away_goals is not None]
    team_set = {m.home_team for m in played} | {m.away_team for m in played}
    team_set |= set(extra_teams or [])
    teams = sorted(team_set)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Elo-derived strength prior per team (centered).
    elos = np.array([elo.get(t) for t in teams])
    prior = (elos - elos.mean()) / _ELO_SCALE if n else elos

    # Precompute per-match arrays.
    hi = np.array([idx[m.home_team] for m in played], dtype=int)
    ai = np.array([idx[m.away_team] for m in played], dtype=int)
    hg = np.array([m.home_goals for m in played], dtype=int)
    ag = np.array([m.away_goals for m in played], dtype=int)
    w = np.array([_decay_weight(m.match_date, as_of, half_life_days) for m in played])

    def unpack(p):
        attack = p[:n]
        defense = p[n:2 * n]
        home_adv = p[2 * n]
        rho = p[2 * n + 1]
        return attack, defense, home_adv, rho

    def neg_log_lik(p):
        attack, defense, home_adv, rho = unpack(p)
        log_lam = attack[hi] - defense[ai] + home_adv
        log_mu = attack[ai] - defense[hi]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)
        # Poisson log-pmf (drop constant log(k!) — irrelevant to the optimum).
        ll = hg * log_lam - lam + ag * log_mu - mu
        # Dixon-Coles correction only affects the four lowest scorelines.
        tau_vals = np.ones(len(played))
        for k in range(len(played)):
            tau_vals[k] = tau(int(hg[k]), int(ag[k]), float(lam[k]), float(mu[k]), float(rho))
        tau_vals = np.clip(tau_vals, 1e-9, None)
        ll = ll + np.log(tau_vals)
        weighted = np.sum(w * ll)
        # Elo ridge prior + soft centering.
        ridge = prior_weight * np.sum((attack - prior) ** 2 + (defense - prior) ** 2)
        center = _CENTER_PENALTY * (attack.mean() ** 2 + defense.mean() ** 2)
        return -weighted + ridge + center

    x0 = np.concatenate([prior, prior, [0.25], [-0.05]])
    res = minimize(neg_log_lik, x0, method="L-BFGS-B")
    attack, defense, home_adv, rho = unpack(res.x)
    return Ratings(
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        home_adv=float(home_adv),
        rho=float(rho),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit.py -v`
Expected: PASS (3 passed). The optimizer is deterministic for fixed input; if `test_fit_recovers_strength_ordering` is flaky, increase the synthetic match count, not the assertions.

- [ ] **Step 5: Commit**

```bash
git add touchline/model/fit.py tests/test_fit.py
git commit -m "feat: weighted Dixon-Coles MLE fit with Elo ridge prior"
```

---

## Task 6: Price a fixture (orchestration)

**Files:**
- Create: `touchline/model/price_fixture.py`
- Test: `tests/test_price_fixture.py`

- [ ] **Step 1: Write the failing test**

`tests/test_price_fixture.py`:
```python
from touchline.model.ratings import Ratings
from touchline.model.factors import FactorContext
from touchline.model.pricing import MarketProbs
from touchline.model.price_fixture import price_fixture


def _ratings():
    return Ratings(
        attack={"USA": 0.3, "Wales": -0.1},
        defense={"USA": 0.2, "Wales": -0.1},
        home_adv=0.3, rho=-0.05,
    )


def test_price_fixture_returns_normalized_1x2():
    probs = price_fixture(_ratings(), "USA", "Wales",
                          apply_home_adv=True, ctx=FactorContext())
    assert isinstance(probs, MarketProbs)
    assert abs((probs.home + probs.draw + probs.away) - 1.0) < 1e-6
    assert probs.home > probs.away   # stronger + home advantage


def test_home_advantage_flag_raises_home_prob():
    r = _ratings()
    with_adv = price_fixture(r, "USA", "Wales", apply_home_adv=True, ctx=FactorContext())
    neutral = price_fixture(r, "USA", "Wales", apply_home_adv=False, ctx=FactorContext())
    assert with_adv.home > neutral.home


def test_heat_lowers_over_probability():
    r = _ratings()
    mild = price_fixture(r, "USA", "Wales", apply_home_adv=False, ctx=FactorContext())
    hot = price_fixture(r, "USA", "Wales", apply_home_adv=False,
                        ctx=FactorContext(wbgt_c=31.0))
    assert hot.over[2.5] < mild.over[2.5]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_price_fixture.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement price_fixture.py**

`touchline/model/price_fixture.py`:
```python
from __future__ import annotations

from touchline.model.dixon_coles import scoreline_matrix
from touchline.model.factors import FactorContext, adjust_expected_goals
from touchline.model.pricing import MarketProbs, price_matrix
from touchline.model.ratings import Ratings


def price_fixture(
    ratings: Ratings,
    home: str,
    away: str,
    apply_home_adv: bool,
    ctx: FactorContext,
    max_goals: int = 10,
) -> MarketProbs:
    """Full pipeline: ratings -> expected goals -> factor adjustment ->
    Dixon-Coles scoreline matrix -> market probabilities."""
    lam, mu = ratings.expected_goals(home, away, apply_home_adv=apply_home_adv)
    lam, mu = adjust_expected_goals(lam, mu, ctx)
    matrix = scoreline_matrix(lam, mu, ratings.rho, max_goals=max_goals)
    return price_matrix(matrix)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_price_fixture.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/model/price_fixture.py tests/test_price_fixture.py
git commit -m "feat: price_fixture orchestration (ratings+factors+matrix->markets)"
```

---

## Task 7: Persist ratings + `fit-ratings` CLI

**Files:**
- Modify: `touchline/storage/db.py` (add ratings persistence)
- Modify: `touchline/cli.py` (add `fit-ratings` subcommand)
- Test: `tests/test_ratings_storage.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ratings_storage.py`:
```python
from touchline.storage.db import Database
from touchline.model.ratings import Ratings


def test_ratings_roundtrip(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    r = Ratings(attack={"USA": 0.3, "Wales": -0.1},
                defense={"USA": 0.2, "Wales": -0.1},
                home_adv=0.27, rho=-0.06)
    db.save_ratings(r)
    loaded = db.load_ratings()
    assert loaded.attack["USA"] == 0.3
    assert loaded.defense["Wales"] == -0.1
    assert abs(loaded.home_adv - 0.27) < 1e-9
    assert abs(loaded.rho - (-0.06)) < 1e-9


def test_save_ratings_replaces_previous(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    db.save_ratings(Ratings(attack={"USA": 0.1}, defense={"USA": 0.1},
                            home_adv=0.2, rho=-0.05))
    db.save_ratings(Ratings(attack={"USA": 0.9}, defense={"USA": 0.9},
                            home_adv=0.3, rho=-0.04))
    loaded = db.load_ratings()
    assert loaded.attack["USA"] == 0.9
    assert abs(loaded.home_adv - 0.3) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ratings_storage.py -v`
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'save_ratings'`.

- [ ] **Step 3: Add ratings persistence to db.py**

In `touchline/storage/db.py`, add to the `_SCHEMA` string (before the closing `"""`):
```sql
CREATE TABLE IF NOT EXISTS team_ratings (
    team    TEXT PRIMARY KEY,
    attack  REAL NOT NULL,
    defense REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_params (
    key   TEXT PRIMARY KEY,
    value REAL NOT NULL
);
```

Add `from touchline.model.ratings import Ratings` to the imports at the top of `db.py`.

Add these two methods to the `Database` class:
```python
    def save_ratings(self, ratings: Ratings) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM team_ratings")
            conn.execute("DELETE FROM model_params")
            conn.executemany(
                "INSERT INTO team_ratings (team, attack, defense) VALUES (?,?,?)",
                [(t, ratings.attack[t], ratings.defense.get(t, 0.0))
                 for t in ratings.attack],
            )
            conn.executemany(
                "INSERT INTO model_params (key, value) VALUES (?,?)",
                [("home_adv", ratings.home_adv), ("rho", ratings.rho)],
            )

    def load_ratings(self) -> Ratings:
        with self._connect() as conn:
            rows = conn.execute("SELECT team, attack, defense FROM team_ratings").fetchall()
            params = dict(conn.execute("SELECT key, value FROM model_params").fetchall())
        return Ratings(
            attack={r["team"]: r["attack"] for r in rows},
            defense={r["team"]: r["defense"] for r in rows},
            home_adv=float(params.get("home_adv", 0.0)),
            rho=float(params.get("rho", 0.0)),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ratings_storage.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the `fit-ratings` CLI command**

In `touchline/cli.py`, add imports near the top:
```python
from datetime import date

from touchline.data.elo import load_elo
from touchline.model.fit import fit_ratings
```

Register the subcommand inside `main` (next to the existing `ingest` parser):
```python
    fit_p = sub.add_parser("fit-ratings", help="Fit Dixon-Coles ratings from stored matches")
    fit_p.add_argument("--half-life-days", type=float, default=540.0)
    fit_p.add_argument("--prior-weight", type=float, default=0.05)
```

Add this branch inside `main` after the `ingest` branch:
```python
    if args.command == "fit-ratings":
        db = Database(config.DB_PATH)
        db.init_schema()
        matches = db.all_matches()
        elo_path = config.CACHE_DIR / "elo.csv"
        elo = load_elo(elo_path) if elo_path.is_file() else EloTable()
        ratings = fit_ratings(
            matches, elo,
            half_life_days=args.half_life_days,
            prior_weight=args.prior_weight,
            as_of=date.today(),
        )
        db.save_ratings(ratings)
        print(f"Fit ratings for {len(ratings.attack)} teams "
              f"(home_adv={ratings.home_adv:.3f}, rho={ratings.rho:.3f}).")
        return 0
```

Add the needed import for the empty-Elo fallback at the top of `cli.py`:
```python
from touchline.data.elo import EloTable
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all green (Plan 1 tests + the new model tests).

- [ ] **Step 7: Live smoke test (uses the DB produced by Plan 1's `ingest`)**

Run: `.venv/Scripts/python.exe -m touchline.cli ingest` then `.venv/Scripts/python.exe -m touchline.cli fit-ratings`
Expected: prints `Fit ratings for N teams (home_adv=..., rho=...).` with `N` in the dozens, `home_adv` positive (~0.2-0.4), `rho` small negative. Then verify a price:
```
.venv/Scripts/python.exe -c "from touchline.storage.db import Database; from touchline import config; from touchline.model.price_fixture import price_fixture; from touchline.model.factors import FactorContext; r=Database(config.DB_PATH).load_ratings(); ts=sorted(r.attack)[:2]; print(ts, price_fixture(r, ts[0], ts[1], apply_home_adv=False, ctx=FactorContext()))"
```
Expected: prints two team names and a `MarketProbs(...)` whose home+draw+away ≈ 1.

- [ ] **Step 8: Commit**

```bash
git add touchline/storage/db.py touchline/cli.py tests/test_ratings_storage.py
git commit -m "feat: persist ratings and add fit-ratings CLI command"
```

---

## Self-Review Notes

- **Spec coverage (model layer):** Dixon-Coles goal model ✓ (Task 1), Elo-seeded time-decayed ratings ✓ (Task 5), expected-goals + conditional home advantage ✓ (Tasks 2, 6), Tier-1/Tier-2 factors travel/altitude/heat/rest ✓ (Task 4), market pricing 1X2/totals/BTTS/spread ✓ (Task 3), `fit-ratings` + persistence ✓ (Task 7). Tier-3 favorite-longshot bias is an edge-*confidence* weight and belongs to Plan 3 (edge ranking), not pricing — intentionally deferred. Heat/rest/travel *data resolution* (Open-Meteo fetch, per-team schedule lookup, venue→FactorContext) is deferred to Plan 3, which builds real `FactorContext`s; Plan 2 delivers the pure factor math validated by unit tests.
- **Placeholder scan:** No TBD/TODO; every code step contains full code. Coefficients in `factors.py` are explicit named constants with stated defaults (calibrated in Plan 3), not placeholders.
- **Type consistency:** `Ratings(attack, defense, home_adv, rho)` used identically across `ratings.py`, `fit.py`, `db.py`, `price_fixture.py`, tests. `MarketProbs(home, draw, away, btts_yes, over, home_handicap)` consistent across `pricing.py`, `price_fixture.py`, tests. `FactorContext` field names consistent across `factors.py` and `price_fixture.py` tests. `scoreline_matrix(lam, mu, rho, max_goals)` and `price_matrix(m, total_lines, handicap_lines)` signatures match all call sites.
- **Open items flagged for execution (not placeholders):** the synthetic-MLE ordering test is behavioral — if the optimizer needs help, raise the synthetic sample size rather than weakening assertions; the live Elo prior needs an operator-supplied `touchline_data/cache/elo.csv` (falls back to neutral 1500 for all teams if absent, per Plan 1's loader).
```
