import json
from datetime import datetime
from pathlib import Path
from touchline.data.weather import estimate_wbgt, wbgt_from_payload

FIXTURE = Path(__file__).parent / "fixtures" / "openmeteo_sample.json"


def test_estimate_wbgt_hot_humid_is_high():
    w = estimate_wbgt(temp_c=33.0, rh_pct=60.0)
    assert 28 < w < 36


def test_estimate_wbgt_mild_is_lower():
    hot = estimate_wbgt(33.0, 60.0)
    mild = estimate_wbgt(18.0, 50.0)
    assert mild < hot


def test_wbgt_from_payload_picks_nearest_hour():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    w = wbgt_from_payload(payload, datetime(2026, 6, 24, 19, 10))
    expected = estimate_wbgt(32.0, 65.0)
    assert abs(w - expected) < 1e-9


def test_wbgt_from_payload_returns_none_when_no_hours():
    assert wbgt_from_payload({"hourly": {"time": [], "temperature_2m": [],
                                         "relative_humidity_2m": []}},
                             datetime(2026, 6, 24, 19, 0)) is None
