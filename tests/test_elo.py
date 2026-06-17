from pathlib import Path
from touchline.data.elo import load_elo, EloTable

FIXTURE = Path(__file__).parent / "fixtures" / "elo_sample.csv"


def test_load_and_lookup_exact():
    table = load_elo(FIXTURE)
    assert table.get("Brazil") == 2120.0
    assert table.get("United States") == 1790.0


def test_lookup_is_case_and_space_insensitive():
    table = load_elo(FIXTURE)
    assert table.get("  brazil ") == 2120.0


def test_missing_team_returns_default_prior():
    table = load_elo(FIXTURE)
    assert table.get("Atlantis", default=1500.0) == 1500.0
