import json
from datetime import date
from pathlib import Path
from touchline.data.worldcupjson import parse_matches

FIXTURE = Path(__file__).parent / "fixtures" / "worldcupjson_sample.json"


def test_parses_completed_match():
    matches = parse_matches(json.loads(FIXTURE.read_text(encoding="utf-8")))
    done = matches[0]
    assert done.match_date == date(2026, 6, 12)
    assert done.home_team == "USA"      # normalized from "United States"
    assert done.away_team == "Wales"
    assert done.home_goals == 2
    assert done.away_goals == 1
    assert done.played is True
    assert done.venue == "MetLife Stadium"
    assert done.source == "worldcupjson"
    assert done.competition == "World Cup 2026"  # derived from match year


def test_competition_derived_from_match_year_not_assumed():
    # Guard: if the API serves a prior tournament's data, it must not be
    # mislabeled as the current World Cup.
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload[0]["datetime"] = "2022-11-21T19:00:00Z"
    m = parse_matches(payload)[0]
    assert m.competition == "World Cup 2022"


def test_future_match_is_unplayed_with_none_goals():
    matches = parse_matches(json.loads(FIXTURE.read_text(encoding="utf-8")))
    fut = matches[1]
    assert fut.played is False
    assert fut.home_goals is None
    assert fut.away_goals is None


def test_fetch_uses_cache_when_fresh(tmp_path):
    from touchline.data.worldcupjson import fetch_matches
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "worldcupjson_matches.json").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # No client passed; if it tried to hit the network with ttl high it'd fail offline.
    matches = fetch_matches(cache_dir, ttl_seconds=10_000)
    assert len(matches) == 2


def test_dead_feed_does_not_crash(tmp_path):
    # worldcupjson.net 404s in 2026; a dead optional feed must return [] not raise.
    import httpx
    from touchline.data.worldcupjson import fetch_matches

    def _raise(*a, **k):
        raise httpx.ConnectError("boom")

    client = httpx.Client()
    client.get = _raise  # type: ignore[assignment]
    assert fetch_matches(tmp_path / "cache", client=client) == []
