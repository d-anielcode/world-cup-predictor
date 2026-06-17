import numpy as np
from touchline.model.pricing import (
    prob_1x2, prob_over, prob_btts, prob_home_handicap, price_matrix, MarketProbs,
)


def _diagonal_matrix():
    m = np.zeros((4, 4))
    m[2, 0] = 0.5   # home 2-0
    m[1, 1] = 0.3   # 1-1
    m[0, 1] = 0.2   # 0-1
    return m


def test_prob_1x2_partitions_outcomes():
    m = _diagonal_matrix()
    home, draw, away = prob_1x2(m)
    assert abs(home - 0.5) < 1e-9
    assert abs(draw - 0.3) < 1e-9
    assert abs(away - 0.2) < 1e-9


def test_prob_over_line():
    m = _diagonal_matrix()
    assert abs(prob_over(m, 1.5) - 0.8) < 1e-9
    assert abs(prob_over(m, 2.5) - 0.0) < 1e-9


def test_prob_btts():
    m = _diagonal_matrix()
    assert abs(prob_btts(m) - 0.3) < 1e-9


def test_prob_home_handicap():
    m = _diagonal_matrix()
    assert abs(prob_home_handicap(m, -1.5) - 0.5) < 1e-9
    assert abs(prob_home_handicap(m, 0.5) - 0.8) < 1e-9


def test_price_matrix_returns_marketprobs():
    m = _diagonal_matrix()
    probs = price_matrix(m, total_lines=[1.5, 2.5], handicap_lines=[-1.5, 0.5])
    assert isinstance(probs, MarketProbs)
    assert abs(probs.home - 0.5) < 1e-9
    assert abs(probs.over[1.5] - 0.8) < 1e-9
    assert abs(probs.btts_yes - 0.3) < 1e-9
    assert abs(probs.home_handicap[-1.5] - 0.5) < 1e-9
