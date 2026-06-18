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
    total_lines: list[float] | None = None,
    handicap_lines: list[float] | None = None,
    lam_mult: float = 1.0,
    mu_mult: float = 1.0,
) -> MarketProbs:
    """Full pipeline: ratings -> expected goals -> factor adjustment ->
    overlay multipliers -> Dixon-Coles scoreline matrix -> market probabilities.

    `lam_mult`/`mu_mult` are squad-overlay goal multipliers (home/away). They are
    applied after the environmental factors. `total_lines`/`handicap_lines` override
    the default market lines so callers can price the exact lines a market offers.
    `max_goals=10` is ample for football (truncation mass beyond it is ~1e-10)."""
    lam, mu = ratings.expected_goals(home, away, apply_home_adv=apply_home_adv)
    lam, mu = adjust_expected_goals(lam, mu, ctx)
    lam, mu = lam * lam_mult, mu * mu_mult
    matrix = scoreline_matrix(lam, mu, ratings.rho, max_goals=max_goals)
    return price_matrix(matrix, total_lines=total_lines, handicap_lines=handicap_lines)
