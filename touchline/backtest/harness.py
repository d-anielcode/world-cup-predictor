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
