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
_CENTER_PENALTY = 100.0


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

    elos = np.array([elo.get(t) for t in teams])
    prior = (elos - elos.mean()) / _ELO_SCALE if n else elos

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
        ll = hg * log_lam - lam + ag * log_mu - mu
        tau_vals = np.ones(len(played))
        for k in range(len(played)):
            tau_vals[k] = tau(int(hg[k]), int(ag[k]), float(lam[k]), float(mu[k]), float(rho))
        tau_vals = np.clip(tau_vals, 1e-9, None)
        ll = ll + np.log(tau_vals)
        weighted = np.sum(w * ll)
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
