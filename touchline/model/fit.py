from __future__ import annotations

import math
import warnings
from datetime import date

import numpy as np
from scipy.optimize import minimize

from touchline.data.elo import EloTable
from touchline.models import Match
from touchline.model.ratings import Ratings

_ELO_SCALE = 400.0
# Regularizer strengths are per-effective-match (scaled by the sum of decay weights
# inside the objective) so they stay comparable as the match count grows.
_CENTER_PENALTY = 100.0  # soft identifiability constraint: mean attack/defense -> 0
_RHO_BOUND = 0.20  # DC rho lives in roughly [-0.15, 0.05] for football; generous headroom


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
) -> Ratings:
    """Fit Dixon-Coles attack/defense ratings by time-weighted MLE with an Elo ridge.

    Only played matches contribute to the likelihood. `extra_teams` lets callers
    include teams that have no played matches yet (priced purely from the Elo prior).

    The negative log-likelihood is a decay-weighted sum; the Elo ridge and the
    identifiability centering penalty are scaled by the sum of decay weights so that
    `prior_weight` and `_CENTER_PENALTY` are expressed per effective match and remain
    stable as the dataset grows (so a calibrated `prior_weight` ports across windows).
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
    wsum = float(w.sum()) if len(played) else 1.0
    # Per-team effective (decay-weighted) match count. The Elo ridge is scaled by
    # THIS, not the global wsum, so the prior acts as `prior_weight` pseudo-matches
    # per team — a team with lots of games is driven by its data, not crushed toward
    # the prior just because the overall dataset is large.
    team_w = np.zeros(len(prior))
    if len(played):
        np.add.at(team_w, hi, w)
        np.add.at(team_w, ai, w)

    # Boolean masks for the four Dixon-Coles low-score corrections (vectorized).
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def unpack(p):
        attack = p[:n]
        defense = p[n:2 * n]
        home_adv = p[2 * n]
        rho = p[2 * n + 1]
        intercept = p[2 * n + 2]
        return attack, defense, home_adv, rho, intercept

    def neg_log_lik(p):
        attack, defense, home_adv, rho, intercept = unpack(p)
        log_lam = intercept + attack[hi] - defense[ai] + home_adv
        log_mu = intercept + attack[ai] - defense[hi]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)
        ll = hg * log_lam - lam + ag * log_mu - mu  # Poisson (drop constant log k!)
        tau_vals = np.ones(len(played))
        tau_vals[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau_vals[m01] = 1.0 + lam[m01] * rho
        tau_vals[m10] = 1.0 + mu[m10] * rho
        tau_vals[m11] = 1.0 - rho
        tau_vals = np.clip(tau_vals, 1e-9, None)
        ll = ll + np.log(tau_vals)
        weighted = np.sum(w * ll)
        ridge = prior_weight * np.sum(team_w * ((attack - prior) ** 2 + (defense - prior) ** 2))
        center = _CENTER_PENALTY * wsum * (attack.mean() ** 2 + defense.mean() ** 2)
        return -weighted + ridge + center

    mean_goals = float((hg.mean() + ag.mean()) / 2) if len(played) else 1.3
    intercept0 = math.log(max(mean_goals, 0.1))
    x0 = np.concatenate([prior, prior, [0.25], [-0.05], [intercept0]])
    bounds = ([(None, None)] * (2 * n + 1) + [(-_RHO_BOUND, _RHO_BOUND)]
              + [(None, None)])
    res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        warnings.warn(f"fit_ratings: optimizer did not converge: {res.message}")
    attack, defense, home_adv, rho, intercept = unpack(res.x)
    return Ratings(
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        home_adv=float(home_adv),
        rho=float(rho),
        intercept=float(intercept),
    )
