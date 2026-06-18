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
