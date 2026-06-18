import pytest
from touchline.model.pricing import MarketProbs
from touchline.edge.model_lookup import model_prob


def _probs():
    return MarketProbs(home=0.5, draw=0.3, away=0.2, btts_yes=0.55,
                       over={2.5: 0.48}, home_handicap={-1.5: 0.3})


def test_1x2_lookup():
    p = _probs()
    assert model_prob(p, "1x2", "home", None) == 0.5
    assert model_prob(p, "1x2", "away", None) == 0.2


def test_total_over_and_under():
    p = _probs()
    assert model_prob(p, "total", "over", 2.5) == 0.48
    assert abs(model_prob(p, "total", "under", 2.5) - 0.52) < 1e-12


def test_btts_yes_no():
    p = _probs()
    assert model_prob(p, "btts", "yes", None) == 0.55
    assert abs(model_prob(p, "btts", "no", None) - 0.45) < 1e-12


def test_handicap_home_and_away():
    p = _probs()
    assert model_prob(p, "handicap", "home", -1.5) == 0.3
    assert abs(model_prob(p, "handicap", "away", -1.5) - 0.7) < 1e-12


def test_missing_line_raises():
    p = _probs()
    with pytest.raises(KeyError):
        model_prob(p, "total", "over", 3.5)
