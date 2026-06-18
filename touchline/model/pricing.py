from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_TOTAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
DEFAULT_HANDICAP_LINES = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]


@dataclass
class MarketProbs:
    home: float
    draw: float
    away: float
    btts_yes: float
    over: dict[float, float]            # total line -> P(total goals > line)
    # handicap line applied to home -> P(home covers). Negative line = home giving
    # goals (P(margin > -line)). See prob_home_handicap. Plan 3 must confirm the
    # Kalshi spread field's sign before mapping these keys to contracts.
    home_handicap: dict[float, float]


def _margin_grid(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    x = np.arange(n)[:, None]
    y = np.arange(n)[None, :]
    return x - y


def _total_grid(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    x = np.arange(n)[:, None]
    y = np.arange(n)[None, :]
    return x + y


def prob_1x2(m: np.ndarray) -> tuple[float, float, float]:
    margin = _margin_grid(m)
    home = float(m[margin > 0].sum())
    draw = float(m[margin == 0].sum())
    away = float(m[margin < 0].sum())
    return home, draw, away


def prob_over(m: np.ndarray, line: float) -> float:
    return float(m[_total_grid(m) > line].sum())


def prob_btts(m: np.ndarray) -> float:
    return float(m[1:, 1:].sum())


def prob_home_handicap(m: np.ndarray, line: float) -> float:
    """P(home_goals + line > away_goals), i.e. P(margin > -line).
    Negative line means home is giving goals (e.g. -1.5 => home must win by 2+)."""
    return float(m[_margin_grid(m) > -line].sum())


def price_matrix(
    m: np.ndarray,
    total_lines: list[float] | None = None,
    handicap_lines: list[float] | None = None,
) -> MarketProbs:
    total_lines = total_lines if total_lines is not None else DEFAULT_TOTAL_LINES
    handicap_lines = handicap_lines if handicap_lines is not None else DEFAULT_HANDICAP_LINES
    home, draw, away = prob_1x2(m)
    return MarketProbs(
        home=home,
        draw=draw,
        away=away,
        btts_yes=prob_btts(m),
        over={ln: prob_over(m, ln) for ln in total_lines},
        home_handicap={ln: prob_home_handicap(m, ln) for ln in handicap_lines},
    )
