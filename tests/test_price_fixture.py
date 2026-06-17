from touchline.model.ratings import Ratings
from touchline.model.factors import FactorContext
from touchline.model.pricing import MarketProbs
from touchline.model.price_fixture import price_fixture


def _ratings():
    return Ratings(
        attack={"USA": 0.3, "Wales": -0.1},
        defense={"USA": 0.2, "Wales": -0.1},
        home_adv=0.3, rho=-0.05,
    )


def test_price_fixture_returns_normalized_1x2():
    probs = price_fixture(_ratings(), "USA", "Wales",
                          apply_home_adv=True, ctx=FactorContext())
    assert isinstance(probs, MarketProbs)
    assert abs((probs.home + probs.draw + probs.away) - 1.0) < 1e-6
    assert probs.home > probs.away


def test_home_advantage_flag_raises_home_prob():
    r = _ratings()
    with_adv = price_fixture(r, "USA", "Wales", apply_home_adv=True, ctx=FactorContext())
    neutral = price_fixture(r, "USA", "Wales", apply_home_adv=False, ctx=FactorContext())
    assert with_adv.home > neutral.home


def test_heat_lowers_over_probability():
    r = _ratings()
    mild = price_fixture(r, "USA", "Wales", apply_home_adv=False, ctx=FactorContext())
    hot = price_fixture(r, "USA", "Wales", apply_home_adv=False,
                        ctx=FactorContext(wbgt_c=31.0))
    assert hot.over[2.5] < mild.over[2.5]
