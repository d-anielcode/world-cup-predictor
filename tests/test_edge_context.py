from datetime import date
from touchline.models import Match
from touchline.edge.context import build_context


def _played(home, away, d, venue):
    return Match(match_date=d, home_team=home, away_team=away, home_goals=1,
                 away_goals=0, competition="WC", stage=None, venue=venue,
                 played=True, source="t")


def test_host_and_altitude_from_venue():
    history = []
    ctx = build_context("Mexico", "USA", date(2026, 6, 24),
                        venue_name="Estadio Azteca", history=history)
    assert ctx.altitude_m > 2000
    assert ctx.home_altitude_acclimatized is True
    assert ctx.away_altitude_acclimatized is False


def test_host_flags_set_for_home_host_at_own_venue():
    # USA hosting at a US venue: home is the host, away (Australia) is not.
    ctx = build_context("USA", "Australia", date(2026, 6, 19),
                        venue_name="MetLife Stadium", history=[])
    assert ctx.home_is_host is True
    assert ctx.away_is_host is False


def test_host_flag_not_set_for_neutral_fixture():
    # Neither team is the host of the US venue.
    ctx = build_context("Turkey", "Paraguay", date(2026, 6, 20),
                        venue_name="MetLife Stadium", history=[])
    assert ctx.home_is_host is False
    assert ctx.away_is_host is False


def test_rest_days_from_prior_match():
    history = [_played("USA", "Wales", date(2026, 6, 20), "MetLife Stadium")]
    ctx = build_context("USA", "Mexico", date(2026, 6, 24),
                        venue_name="SoFi Stadium", history=history)
    assert ctx.rest_days_home == 4


def test_travel_distance_positive_for_cross_country_move():
    history = [_played("USA", "Wales", date(2026, 6, 20), "MetLife Stadium")]
    ctx = build_context("USA", "Mexico", date(2026, 6, 24),
                        venue_name="SoFi Stadium", history=history)
    assert ctx.travel_km_home > 3000
