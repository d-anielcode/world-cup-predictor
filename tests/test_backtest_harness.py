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
    eval_start = date(2026, 1, 1)
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
    assert r.log_loss < math.log(3)
    assert r.brier < 2/3


def test_no_eval_matches_returns_zero_count():
    matches = _synthetic()
    r = backtest(matches, eval_start=date(2099, 1, 1), half_life_days=400,
                 prior_weight=0.05, elo=EloTable())
    assert r.n_matches == 0
