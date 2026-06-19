import math
from touchline.model.ratings import Ratings


def _ratings():
    return Ratings(
        attack={"Brazil": 0.5, "Bolivia": -0.4},
        defense={"Brazil": 0.3, "Bolivia": -0.2},
        home_adv=0.25,
        rho=-0.05,
    )


def test_expected_goals_neutral_site():
    r = _ratings()
    lam, mu = r.expected_goals("Brazil", "Bolivia", apply_home_adv=False)
    assert abs(lam - math.exp(0.5 - (-0.2))) < 1e-9
    assert abs(mu - math.exp(-0.4 - 0.3)) < 1e-9


def test_home_advantage_increases_home_lambda():
    r = _ratings()
    lam_n, _ = r.expected_goals("Brazil", "Bolivia", apply_home_adv=False)
    lam_h, _ = r.expected_goals("Brazil", "Bolivia", apply_home_adv=True)
    assert lam_h > lam_n
    assert abs(math.log(lam_h) - math.log(lam_n) - 0.25) < 1e-9


def test_unknown_team_uses_default_zero_rating():
    r = _ratings()
    lam, mu = r.expected_goals("Brazil", "Atlantis", apply_home_adv=False)
    assert abs(lam - math.exp(0.5 - 0.0)) < 1e-9


def test_intercept_lifts_both_expected_goals():
    import math as _math
    from touchline.model.ratings import Ratings
    r0 = Ratings(attack={"A": 0.0}, defense={"A": 0.0}, home_adv=0.0, rho=-0.05)
    r1 = Ratings(attack={"A": 0.0}, defense={"A": 0.0}, home_adv=0.0, rho=-0.05,
                 intercept=0.2)
    lam0, mu0 = r0.expected_goals("A", "A", apply_home_adv=False)
    lam1, mu1 = r1.expected_goals("A", "A", apply_home_adv=False)
    assert lam1 > lam0 and mu1 > mu0
    assert abs(_math.log(lam1) - _math.log(lam0) - 0.2) < 1e-9


def test_intercept_defaults_to_zero():
    from touchline.model.ratings import Ratings
    r = Ratings(attack={"A": 0.0}, defense={"A": 0.0}, home_adv=0.0, rho=0.0)
    assert r.intercept == 0.0
