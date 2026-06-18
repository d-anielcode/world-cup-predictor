from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from touchline.data.elo import EloTable
from touchline.models import Match
from touchline.backtest.harness import backtest


@dataclass
class CalibrationResult:
    best_half_life: float
    best_prior_weight: float
    best_log_loss: float
    grid: list[tuple[float, float, float]]   # (half_life, prior_weight, log_loss)


def calibrate(
    matches: list[Match],
    eval_start: date,
    elo: EloTable,
    half_lifes: list[float],
    prior_weights: list[float],
) -> CalibrationResult:
    """Grid-search (half_life, prior_weight) minimizing backtest log-loss."""
    grid: list[tuple[float, float, float]] = []
    for hl in half_lifes:
        for pw in prior_weights:
            result = backtest(matches, eval_start=eval_start, half_life_days=hl,
                              prior_weight=pw, elo=elo)
            grid.append((hl, pw, result.log_loss))
    best = min(grid, key=lambda row: row[2])
    return CalibrationResult(
        best_half_life=best[0], best_prior_weight=best[1], best_log_loss=best[2],
        grid=grid,
    )
