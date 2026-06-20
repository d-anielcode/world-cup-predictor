from pathlib import Path

import httpx
import pytest

from touchline.data import elo_scrape
from touchline.data.elo import load_elo

FIXTURES = Path(__file__).parent / "fixtures"
WORLD = (FIXTURES / "eloratings_world_sample.tsv").read_text(encoding="utf-8")
TEAMS = (FIXTURES / "eloratings_teams_sample.tsv").read_text(encoding="utf-8")


def test_parse_joins_code_to_name_and_elo():
    table = elo_scrape.parse_elo(WORLD, TEAMS)
    assert table["Spain"] == 2129.0
    assert table["Brazil"] == 1978.0
    assert table["Turkey"] == 1849.0


def test_parse_canonicalizes_team_names():
    table = elo_scrape.parse_elo(WORLD, TEAMS)
    # "United States" must land under the canonical "USA"
    assert "USA" in table
    assert "United States" not in table
    assert table["USA"] == 1820.0


def test_parse_skips_unknown_code_and_malformed_elo():
    table = elo_scrape.parse_elo(WORLD, TEAMS)
    assert "ZZ" not in table  # code absent from teams file
    assert len(table) == 5     # ES, AR, BR, USA, TR (ZZ skipped, XX malformed)


def test_write_elo_csv_roundtrips_through_load_elo(tmp_path):
    table = elo_scrape.parse_elo(WORLD, TEAMS)
    out = tmp_path / "elo.csv"
    elo_scrape.write_elo_csv(table, out)
    loaded = load_elo(out)
    assert loaded.get("USA") == 1820.0
    assert loaded.get("Spain") == 2129.0


def test_fetch_elo_uses_injected_client(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("World.tsv"):
            return httpx.Response(200, text=WORLD)
        if request.url.path.endswith("en.teams.tsv"):
            return httpx.Response(200, text=TEAMS)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    table = elo_scrape.fetch_elo(client=client)
    assert table["USA"] == 1820.0
    assert len(table) == 5


def test_fetch_elo_raises_on_http_error():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(httpx.HTTPError):
        elo_scrape.fetch_elo(client=client)
