import math
from touchline.backtest.scoring import outcome_index, brier_score, log_loss


def test_outcome_index_maps_result():
    assert outcome_index(2, 0) == 0
    assert outcome_index(1, 1) == 1
    assert outcome_index(0, 3) == 2


def test_brier_perfect_prediction_is_zero():
    assert brier_score([(1.0, 0.0, 0.0)], [0]) == 0.0


def test_brier_uniform_is_two_thirds():
    b = brier_score([(1/3, 1/3, 1/3)], [0])
    assert abs(b - 2/3) < 1e-9


def test_log_loss_uniform_is_ln3():
    ll = log_loss([(1/3, 1/3, 1/3)], [2])
    assert abs(ll - math.log(3)) < 1e-9


def test_log_loss_clips_zero_probability():
    ll = log_loss([(1.0, 0.0, 0.0)], [2])
    assert ll > 0 and math.isfinite(ll)
