from __future__ import annotations

import math

_EPS = 1e-12


def outcome_index(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def brier_score(probs: list[tuple[float, float, float]], outcomes: list[int]) -> float:
    """Mean multiclass Brier score over (home, draw, away) predictions."""
    total = 0.0
    for (ph, pd, pa), o in zip(probs, outcomes):
        y = [0.0, 0.0, 0.0]
        y[o] = 1.0
        total += (ph - y[0]) ** 2 + (pd - y[1]) ** 2 + (pa - y[2]) ** 2
    return total / len(probs) if probs else 0.0


def log_loss(probs: list[tuple[float, float, float]], outcomes: list[int]) -> float:
    """Mean negative log-likelihood of the realized outcomes (clipped)."""
    total = 0.0
    for triple, o in zip(probs, outcomes):
        p = min(1.0, max(_EPS, triple[o]))
        total += -math.log(p)
    return total / len(probs) if probs else 0.0


def binary_brier(preds: list[float], actuals: list[float]) -> float:
    """Mean squared error of binary probability predictions."""
    if not preds:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(preds, actuals)) / len(preds)


def base_rate_brier(actuals: list[float]) -> float:
    """Brier of the constant base-rate predictor (the bar a useful model must beat)."""
    if not actuals:
        return 0.0
    base = sum(actuals) / len(actuals)
    return base * (1 - base)


def calibration_gap(preds: list[float], actuals: list[float]) -> float:
    """mean(prediction) - actual_rate. Near 0 = well calibrated."""
    if not preds:
        return 0.0
    return sum(preds) / len(preds) - sum(actuals) / len(actuals)
