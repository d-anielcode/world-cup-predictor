import json
import pytest
from touchline.overlay.squad import load_overlay, fixture_multipliers, TeamAdjustment


def _write(tmp_path, data):
    p = tmp_path / "squad.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_parses_team_adjustments(tmp_path):
    p = _write(tmp_path, {"Brazil": {"attack_mult": 0.9, "defense_mult": 1.0,
                                     "reason": "Neymar out", "source": "news"}})
    ov = load_overlay(p)
    assert isinstance(ov["Brazil"], TeamAdjustment)
    assert ov["Brazil"].attack_mult == 0.9
    assert ov["Brazil"].reason == "Neymar out"


def test_missing_path_returns_empty_overlay(tmp_path):
    assert load_overlay(tmp_path / "nope.json") == {}


def test_rejects_out_of_range_multiplier(tmp_path):
    p = _write(tmp_path, {"Brazil": {"attack_mult": 5.0, "defense_mult": 1.0,
                                     "reason": "x", "source": "y"}})
    with pytest.raises(ValueError):
        load_overlay(p)


def test_fixture_multipliers_compose_home_and_away():
    ov = {
        "Brazil": TeamAdjustment(0.9, 1.0, "Neymar out", "news"),
        "Chile": TeamAdjustment(1.0, 0.8, "leaky defense", "news"),
    }
    lam_mult, mu_mult = fixture_multipliers("Brazil", "Chile", ov)
    assert lam_mult == 0.9 * 0.8
    assert mu_mult == 1.0 * 1.0


def test_unknown_teams_are_identity():
    assert fixture_multipliers("X", "Y", {}) == (1.0, 1.0)
