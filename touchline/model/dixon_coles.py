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
    home_pmf = poisson.pmf(goals, lam)
    away_pmf = poisson.pmf(goals, mu)
    m = np.outer(home_pmf, away_pmf)
    m[0, 0] *= tau(0, 0, lam, mu, rho)
    m[0, 1] *= tau(0, 1, lam, mu, rho)
    m[1, 0] *= tau(1, 0, lam, mu, rho)
    m[1, 1] *= tau(1, 1, lam, mu, rho)
    # Guard: at extreme rho a tau correction can drive a low-score cell negative.
    m = np.clip(m, 0.0, None)
    total = m.sum()
    return m / total if total > 0 else m
