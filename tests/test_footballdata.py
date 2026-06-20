from datetime import date
from pathlib import Path

from touchline.data.footballdata import parse_csv

CSV = (Path(__file__).parent / "fixtures" / "footballdata_E0_sample.csv").read_text(
    encoding="utf-8")


def test_parse_csv_builds_played_matches_with_odds():
    rows = parse_csv(CSV, "EPL")
    # row 4 has no closing odds -> skipped
    assert len(rows) == 3
    m, odds = rows[0]
    assert m.match_date == date(2025, 8, 15)
    assert (m.home_team, m.away_team) == ("Liverpool", "Bournemouth")
    assert (m.home_goals, m.away_goals) == (4, 2)
    assert m.played is True
    assert m.competition == "EPL"
    # decimal closing odds (market-average preferred)
    assert odds == (1.29, 6.02, 8.68)


def test_parse_csv_falls_back_to_b365_when_avg_missing(tmp_path):
    csv = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365CH,B365CD,B365CA,AvgCH,AvgCD,AvgCA\n"
           "E0,16/08/2025,Arsenal,Chelsea,1,0,H,2.00,3.50,4.00,,,\n")
    rows = parse_csv(csv, "EPL")
    assert len(rows) == 1
    assert rows[0][1] == (2.00, 3.50, 4.00)


def test_parse_csv_skips_unplayed_or_malformed():
    csv = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgCH,AvgCD,AvgCA\n"
           "E0,16/08/2025,A,B,,,,,2.0,3.0\n"          # no goals
           "E0,16/08/2025,C,D,1,1,D,2.0,3.0,3.5\n")   # ok
    rows = parse_csv(csv, "EPL")
    assert len(rows) == 1
    assert (rows[0][0].home_team, rows[0][0].away_team) == ("C", "D")
