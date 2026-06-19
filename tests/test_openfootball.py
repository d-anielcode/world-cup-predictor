from datetime import date
from pathlib import Path
from touchline.data.openfootball import parse_cup_txt

FIXTURE = Path(__file__).parent / "fixtures" / "openfootball_sample.txt"


def test_parses_all_played_matches():
    matches = parse_cup_txt(FIXTURE.read_text(encoding="utf-8"), competition="World Cup 2022")
    assert len(matches) == 3


def test_first_match_fields():
    m = parse_cup_txt(FIXTURE.read_text(encoding="utf-8"), competition="World Cup 2022")[0]
    assert m.match_date == date(2022, 11, 20)
    assert m.home_team == "Qatar"
    assert m.away_team == "Ecuador"
    assert m.home_goals == 0
    assert m.away_goals == 2
    assert m.stage == "Group A"
    assert m.venue == "Al Bayt Stadium, Al Khor"
    assert m.played is True
    assert m.source == "openfootball"


def test_year_inferred_for_december_rollover():
    m = parse_cup_txt(FIXTURE.read_text(encoding="utf-8"), competition="World Cup 2022")[2]
    assert m.match_date == date(2022, 11, 25)


def test_parses_2014_2018_home_v_away_format():
    # 2014/2018 files: date on its own line, then "<time> UTC-N  Home v Away  score".
    text = """= World Cup 2014

▪ Group A
Thu Jun 12
  17:00 UTC-3  Brazil v Croatia   3-1 (1-1)       @ Arena de São Paulo, São Paulo
"""
    matches = parse_cup_txt(text, competition="World Cup 2014")
    assert len(matches) == 1
    m = matches[0]
    assert m.match_date == date(2014, 6, 12)
    assert m.home_team == "Brazil"
    assert m.away_team == "Croatia"
    assert m.home_goals == 3
    assert m.away_goals == 1
    assert m.venue == "Arena de São Paulo, São Paulo"


def test_parses_finals_with_extra_time_and_penalties():
    # cup_finals.txt: regulation/ET score is the first N-N; penalties decide the
    # winner but goals scored = the first N-N. Trailing "# seeding" comment ignored.
    text = """= World Cup 2014

▪ Round of 16
Sat Jun 28
 13:00 UTC-3  Brazil      1-1 a.e.t. (1-1, 1-1), 3-2 pen.   Chile    @ Estádio Mineirão, Belo Horizonte   # 1A - 2B
"""
    matches = parse_cup_txt(text, competition="World Cup 2014")
    assert len(matches) == 1
    m = matches[0]
    assert m.home_team == "Brazil"
    assert m.away_team == "Chile"
    assert m.home_goals == 1
    assert m.away_goals == 1
    assert m.venue == "Estádio Mineirão, Belo Horizonte"


def test_parses_finals_aet_without_trailing_dot():
    # Some files write "a.e.t" without the trailing period.
    text = """= World Cup 2018

▪ Round of 16
Mon Jul 2
 17:00  Japan      1-1 a.e.t (1-1, 1-0), 1-3 pen.   Croatia   @ Rostov Arena
"""
    m = parse_cup_txt(text, competition="World Cup 2018")[0]
    assert m.home_team == "Japan"
    assert m.away_team == "Croatia"
    assert (m.home_goals, m.away_goals) == (1, 1)


def test_normalizes_team_names_across_sources():
    text = """= World Cup 2022

▪ Group B
Mon Nov 21
  19:00     United States  1-1 (0-1)  Wales   @ Ahmad Bin Ali Stadium
"""
    m = parse_cup_txt(text, competition="World Cup 2022")[0]
    assert m.home_team == "USA"   # normalized from "United States"


def test_parses_pre_2014_inline_date_format():
    # 1930-2010 openfootball files put the weekday/date/time inline on the
    # match line (no separate date line, no halftime parenthetical).
    text = """= World Cup 2010          # in South Africa

▪ Group A

Fri Jun 11 16:00    South Africa  1-1  Mexico      @ Soccer City, Johannesburg
Fri Jun 11 20:30    Uruguay       0-0  France      @ Cape Town Stadium, Cape Town
"""
    matches = parse_cup_txt(text, competition="World Cup 2010")
    assert len(matches) == 2
    first = matches[0]
    assert first.match_date == date(2010, 6, 11)
    assert first.home_team == "South Africa"
    assert first.away_team == "Mexico"
    assert first.home_goals == 1
    assert first.away_goals == 1
    assert first.stage == "Group A"
    assert first.venue == "Soccer City, Johannesburg"
    assert first.played is True


def test_parses_2002_2006_inline_date_without_time():
    # 2002-2006 files use weekday+month+day but no kickoff time.
    text = """= World Cup 2006

▪ Group A

Fri Jun 9     Germany     4-2 (2-1)  Costa Rica   @ Allianz Arena, München
"""
    matches = parse_cup_txt(text, competition="World Cup 2006")
    assert len(matches) == 1
    m = matches[0]
    assert m.match_date == date(2006, 6, 9)
    assert m.home_team == "Germany"
    assert m.away_team == "Costa Rica"
    assert m.home_goals == 4
    assert m.away_goals == 2
    assert m.venue == "Allianz Arena, München"


def test_find_cup_files_filters_to_cup_txt(tmp_path):
    from touchline.data.openfootball import find_cup_files
    (tmp_path / "2022--qatar").mkdir()
    (tmp_path / "2022--qatar" / "cup.txt").write_text("x", encoding="utf-8")
    (tmp_path / "2022--qatar" / "cup_finals.txt").write_text("x", encoding="utf-8")
    (tmp_path / "2022--qatar" / "NOTES.md").write_text("x", encoding="utf-8")
    (tmp_path / "2022--qatar" / "cup_stadiums.csv").write_text("x", encoding="utf-8")
    files = find_cup_files(tmp_path)
    assert {f.name for f in files} == {"cup.txt", "cup_finals.txt"}


def test_parses_2026_full_month_played_and_upcoming():
    # 2026 format: full month name on the date line, time+UTC prefix, played
    # results as "Home N-N (HT) Away" and upcoming fixtures as "Home v Away".
    text = """= World Cup 2026

▪ Group D
Fri June 12
  18:00 UTC-7    USA  4-1 (3-0)  Paraguay    @ Los Angeles (Inglewood)
Fri June 19
  12:00 UTC-7    USA      v Australia   @ Seattle
"""
    matches = parse_cup_txt(text, competition="World Cup 2026")
    assert len(matches) == 2
    played = matches[0]
    assert played.match_date == date(2026, 6, 12)
    assert played.home_team == "USA" and played.away_team == "Paraguay"
    assert (played.home_goals, played.away_goals) == (4, 1)
    assert played.played is True
    upcoming = matches[1]
    assert upcoming.match_date == date(2026, 6, 19)
    assert upcoming.home_team == "USA" and upcoming.away_team == "Australia"
    assert upcoming.played is False
    assert upcoming.home_goals is None
    assert upcoming.venue == "Seattle"
