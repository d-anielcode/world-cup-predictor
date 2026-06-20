from touchline.model.factors import FactorContext, adjust_expected_goals


def test_no_context_is_identity():
    lam, mu = adjust_expected_goals(1.5, 1.2, FactorContext())
    assert (round(lam, 9), round(mu, 9)) == (1.5, 1.2)


def test_travel_reduces_travelling_team_goals():
    ctx = FactorContext(travel_km_home=3000.0)
    lam, mu = adjust_expected_goals(1.5, 1.2, ctx)
    assert lam < 1.5
    assert mu == 1.2


def test_heat_reduces_both_teams_goals():
    hot = FactorContext(wbgt_c=30.0)
    lam, mu = adjust_expected_goals(1.5, 1.2, hot)
    assert lam < 1.5 and mu < 1.2
    mild = FactorContext(wbgt_c=20.0)
    lam2, mu2 = adjust_expected_goals(1.5, 1.2, mild)
    assert (round(lam2, 9), round(mu2, 9)) == (1.5, 1.2)


def test_altitude_penalizes_unacclimatized_team_only():
    ctx = FactorContext(altitude_m=2240, away_altitude_acclimatized=False,
                        home_altitude_acclimatized=True)
    lam, mu = adjust_expected_goals(1.5, 1.2, ctx)
    assert lam == 1.5
    assert mu < 1.2


def test_rest_disadvantage_reduces_more_tired_team():
    ctx = FactorContext(rest_days_home=2, rest_days_away=6)
    lam, mu = adjust_expected_goals(1.5, 1.2, ctx)
    assert lam < 1.5
    assert mu == 1.2


def test_host_advantage_lifts_host_supremacy_when_home():
    # A genuine host (listed home) gets a modest goal-supremacy boost beyond the
    # rating's generic home_adv: its expected goals rise and the opponent's fall.
    ctx = FactorContext(home_is_host=True)
    lam, mu = adjust_expected_goals(1.5, 1.2, ctx)
    assert lam > 1.5 and mu < 1.2
    assert (lam - mu) > (1.5 - 1.2)


def test_host_advantage_applies_to_host_listed_as_away():
    # Host-as-away: the away side is the host, so the boost goes to the away goals.
    ctx = FactorContext(away_is_host=True)
    lam, mu = adjust_expected_goals(1.5, 1.2, ctx)
    assert mu > 1.2 and lam < 1.5


def test_non_host_game_has_no_host_boost():
    lam, mu = adjust_expected_goals(1.5, 1.2, FactorContext())
    assert (round(lam, 9), round(mu, 9)) == (1.5, 1.2)
