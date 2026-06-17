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
