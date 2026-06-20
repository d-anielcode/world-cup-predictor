from datetime import date, datetime, timezone

from touchline.data.openfootball import parse_cup_txt
from touchline.models import Match, dedupe_matches
from touchline.storage.db import Database


def test_parse_captures_kickoff_as_utc():
    txt = "\n".join([
        "▪ Group A",
        "Thu June 11",
        "  19:00 UTC-6    Mexico  v  South Korea   @ Guadalajara (Zapopan)",
    ])
    m = parse_cup_txt(txt, "World Cup 2026")[0]
    # 19:00 at UTC-6 == 01:00 UTC the next day.
    assert m.kickoff == datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc)
    assert m.played is False


def test_parse_kickoff_none_without_utc_offset():
    txt = "\n".join([
        "Fri Jun 11",
        "  16:00 Brazil  2-0  Croatia   @ Somewhere",
    ])
    m = parse_cup_txt(txt, "World Cup 2014")[0]
    assert m.kickoff is None  # no UTC offset -> cannot resolve absolute time


def test_dedupe_prefers_a_known_kickoff():
    ko = datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc)
    a = Match(date(2026, 6, 11), "Mexico", "South Korea", None, None, "WC", None,
              "Guadalajara", False, "intl_results", kickoff=None)
    b = Match(date(2026, 6, 11), "Mexico", "South Korea", None, None, "WC", None,
              "Guadalajara", False, "openfootball", kickoff=ko)
    out = dedupe_matches([a, b])
    assert len(out) == 1
    assert out[0].kickoff == ko


def test_match_kickoff_defaults_to_none():
    m = Match(date(2026, 6, 11), "A", "B", None, None, "WC", None, None, False, "t")
    assert m.kickoff is None


def test_kickoff_roundtrips_through_db(tmp_path):
    ko = datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc)
    db = Database(tmp_path / "t.db")
    db.init_schema()
    db.upsert_matches([
        Match(date(2026, 6, 11), "Mexico", "South Korea", None, None, "WC", None,
              "Guadalajara", False, "openfootball", kickoff=ko),
    ])
    got = db.all_matches()[0]
    assert got.kickoff == ko
