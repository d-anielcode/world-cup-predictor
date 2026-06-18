from touchline.model.ratings import Ratings
from touchline.model.factors import FactorContext
from touchline.model.price_fixture import price_fixture


def _ratings():
    return Ratings(attack={"A": 0.4, "B": -0.1}, defense={"A": 0.2, "B": -0.1},
                   home_adv=0.0, rho=-0.05)


def test_lam_mult_below_one_lowers_home_win_prob():
    r = _ratings()
    base = price_fixture(r, "A", "B", apply_home_adv=False, ctx=FactorContext())
    weakened = price_fixture(r, "A", "B", apply_home_adv=False, ctx=FactorContext(),
                             lam_mult=0.7)
    assert weakened.home < base.home


def test_default_mults_are_identity():
    r = _ratings()
    base = price_fixture(r, "A", "B", apply_home_adv=False, ctx=FactorContext())
    same = price_fixture(r, "A", "B", apply_home_adv=False, ctx=FactorContext(),
                         lam_mult=1.0, mu_mult=1.0)
    assert abs(base.home - same.home) < 1e-12
