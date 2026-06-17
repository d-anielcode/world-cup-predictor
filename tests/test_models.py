from datetime import date
from touchline.models import Match


def test_match_natural_key_is_stable_and_order_independent_fields():
    m = Match(
        match_date=date(2022, 11, 20),
        home_team="Qatar",
        away_team="Ecuador",
        home_goals=0,
        away_goals=2,
        competition="World Cup 2022",
        stage="Group A",
        venue="Al Bayt Stadium, Al Khor",
        played=True,
        source="openfootball",
    )
    assert m.natural_key() == "2022-11-20|Qatar|Ecuador"


def test_unplayed_match_has_none_goals():
    m = Match(
        match_date=date(2026, 6, 20),
        home_team="USA",
        away_team="Wales",
        home_goals=None,
        away_goals=None,
        competition="World Cup 2026",
        stage="Group A",
        venue=None,
        played=False,
        source="worldcupjson",
    )
    assert m.played is False
    assert m.home_goals is None
