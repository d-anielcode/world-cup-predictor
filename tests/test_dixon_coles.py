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
    from scipy.stats import poisson
    lam, mu = 1.4, 0.9
    m = scoreline_matrix(lam, mu, 0.0, max_goals=15)
    assert abs(m[1, :].sum() - poisson.pmf(1, lam)) < 1e-4
