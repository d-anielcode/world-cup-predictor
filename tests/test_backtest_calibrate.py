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
    assert res.best_log_loss == min(row[2] for row in res.grid)
    assert len(res.grid) == 4
