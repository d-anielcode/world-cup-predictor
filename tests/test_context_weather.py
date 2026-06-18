from datetime import date
from touchline.edge.context import build_context


def test_wbgt_passes_through_to_factor_context():
    ctx = build_context("Mexico", "USA", date(2026, 6, 24),
                        venue_name="Estadio Azteca", history=[], wbgt_c=29.5)
    assert ctx.wbgt_c == 29.5


def test_wbgt_defaults_to_none():
    ctx = build_context("Mexico", "USA", date(2026, 6, 24),
                        venue_name="Estadio Azteca", history=[])
    assert ctx.wbgt_c is None


def test_unknown_venue_returns_neutral_context():
    ctx = build_context("Brazil", "Chile", date(2014, 6, 28),
                        venue_name="Estadio Mineirao, Belo Horizonte", history=[])
    assert ctx.altitude_m == 0
    assert ctx.wbgt_c is None
