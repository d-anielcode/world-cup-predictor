from pathlib import Path
from touchline.edge.quotes import load_quotes, MarketQuoteRow, fixture_lines

FIXTURE = Path(__file__).parent / "fixtures" / "quotes_sample.csv"


def test_load_quotes_parses_rows():
    rows = load_quotes(FIXTURE)
    assert len(rows) == 5
    assert isinstance(rows[0], MarketQuoteRow)
    assert rows[0].home == "USA" and rows[0].market_type == "1x2"
    assert rows[0].side == "home" and rows[0].line is None
    assert rows[0].price == 0.55


def test_total_and_handicap_lines_are_floats():
    rows = load_quotes(FIXTURE)
    total = next(r for r in rows if r.market_type == "total")
    hcap = next(r for r in rows if r.market_type == "handicap")
    assert total.line == 2.5
    assert hcap.line == -1.5


def test_fixture_lines_collects_distinct_lines_per_fixture():
    rows = load_quotes(FIXTURE)
    totals, handicaps = fixture_lines(rows, "USA", "Wales")
    assert totals == [2.5]
    assert handicaps == [-1.5]
