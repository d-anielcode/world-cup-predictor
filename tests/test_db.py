from datetime import date
from touchline.models import Match
from touchline.storage.db import Database


def _match(home="Qatar", away="Ecuador", hg=0, ag=2):
    return Match(
        match_date=date(2022, 11, 20), home_team=home, away_team=away,
        home_goals=hg, away_goals=ag, competition="World Cup 2022",
        stage="Group A", venue="Al Bayt Stadium", played=True, source="openfootball",
    )


def test_upsert_then_query_roundtrip(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    db.upsert_matches([_match()])
    rows = db.all_matches()
    assert len(rows) == 1
    assert rows[0].home_team == "Qatar"
    assert rows[0].away_goals == 2


def test_upsert_is_idempotent_on_natural_key(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    db.upsert_matches([_match(hg=0, ag=2)])
    db.upsert_matches([_match(hg=1, ag=1)])  # same teams+date, corrected score
    rows = db.all_matches()
    assert len(rows) == 1
    assert (rows[0].home_goals, rows[0].away_goals) == (1, 1)
