from datetime import date
from touchline.data.intl_results import parse_results_csv

SAMPLE = """date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
1990-06-10,Brazil,Sweden,1,1,FIFA World Cup,Turin,Italy,TRUE
2022-03-29,Bolivia,Brazil,0,4,FIFA World Cup qualification,La Paz,Bolivia,FALSE
2026-06-27,Panama,England,NA,NA,FIFA World Cup,East Rutherford,United States,TRUE
2023-09-12,United States,Oman,4,0,Friendly,St. Paul,United States,FALSE
"""


def test_filters_by_year_and_parses_played():
    ms = parse_results_csv(SAMPLE, since_year=2014)
    # 1990 row filtered out; 3 remain (2022, 2026, 2023)
    assert len(ms) == 3
    bol = next(m for m in ms if m.home_team == "Bolivia")
    assert bol.match_date == date(2022, 3, 29)
    assert bol.away_team == "Brazil"
    assert (bol.home_goals, bol.away_goals) == (0, 4)
    assert bol.played is True
    assert bol.competition == "FIFA World Cup qualification"
    assert bol.source == "intl_results"


def test_na_scores_are_unplayed():
    ms = parse_results_csv(SAMPLE, since_year=2014)
    fut = next(m for m in ms if m.home_team == "Panama")
    assert fut.played is False
    assert fut.home_goals is None and fut.away_goals is None


def test_team_names_are_canonicalized():
    ms = parse_results_csv(SAMPLE, since_year=2014)
    usa = next(m for m in ms if m.away_team == "Oman")
    assert usa.home_team == "USA"   # normalized from "United States"
