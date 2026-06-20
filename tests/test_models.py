from datetime import date
from touchline.models import Match


def test_match_natural_key_ignores_non_identifying_fields():
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
    # Teams are sorted so the key is independent of which feed listed the nominal
    # home team (feeds disagree on neutral-site orientation -> would double-count).
    assert m.natural_key() == "2022-11-20|Ecuador|Qatar"


def test_match_natural_key_is_home_away_order_independent():
    a = Match(date(2014, 6, 23), "Cameroon", "Brazil", 1, 4,
              "World Cup 2014", None, None, True, "openfootball")
    b = Match(date(2014, 6, 23), "Brazil", "Cameroon", 4, 1,
              "World Cup 2014", None, None, True, "intl_results")
    assert a.natural_key() == b.natural_key()


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
