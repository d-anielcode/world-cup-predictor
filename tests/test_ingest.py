from datetime import date
from touchline.models import Match
from touchline.storage.db import Database
from touchline.cli import run_ingest


def _match(src, home="A", away="B"):
    return Match(
        match_date=date(2026, 6, 12), home_team=home, away_team=away,
        home_goals=1, away_goals=0, competition="World Cup 2026",
        stage="Group A", venue="MetLife Stadium", played=True, source=src,
    )


def test_run_ingest_stores_matches_from_all_sources(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    historical = [_match("openfootball", "Qatar", "Ecuador")]
    live = [_match("worldcupjson", "USA", "Wales")]
    count = run_ingest(db, historical=historical, live=live)
    assert count == 2
    assert len(db.all_matches()) == 2
