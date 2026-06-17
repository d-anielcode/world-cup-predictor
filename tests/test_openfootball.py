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


def test_find_cup_files_filters_to_cup_txt(tmp_path):
    from touchline.data.openfootball import find_cup_files
    (tmp_path / "2022--qatar").mkdir()
    (tmp_path / "2022--qatar" / "cup.txt").write_text("x", encoding="utf-8")
    (tmp_path / "2022--qatar" / "cup_finals.txt").write_text("x", encoding="utf-8")
    (tmp_path / "2022--qatar" / "NOTES.md").write_text("x", encoding="utf-8")
    (tmp_path / "2022--qatar" / "cup_stadiums.csv").write_text("x", encoding="utf-8")
    files = find_cup_files(tmp_path)
    assert {f.name for f in files} == {"cup.txt", "cup_finals.txt"}
