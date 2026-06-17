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
