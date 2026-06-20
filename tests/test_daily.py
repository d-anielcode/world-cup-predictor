from datetime import date

from touchline.models import Match
from touchline.model.ratings import Ratings
from touchline.edge.quotes import MarketQuoteRow
from touchline.cli import run_daily


def _ratings():
    return Ratings(attack={"USA": 0.5, "Wales": -0.2},
                   defense={"USA": 0.3, "Wales": -0.2}, home_adv=0.2, rho=-0.05, intercept=0.1)


def test_run_daily_writes_dated_reports(tmp_path):
    history = [Match(match_date=date(2026, 6, 10), home_team="USA", away_team="Iran",
                     home_goals=2, away_goals=0, competition="WC", stage=None,
                     venue="MetLife Stadium", played=True, source="t")]
    fixtures = [("USA", "Wales", date(2026, 6, 24), "SoFi Stadium")]
    quotes = [MarketQuoteRow("USA", "Wales", "1x2", "home", None, 0.50, "T-USA"),
              MarketQuoteRow("USA", "Wales", "1x2", "away", None, 0.30, "T-WAL")]
    md_path, json_path = run_daily(
        ratings=_ratings(), overlay={}, quotes=quotes, fixtures=fixtures,
        history=history, team_games={"USA": 30, "Wales": 25},
        as_of="2026-06-23", out_dir=tmp_path,
    )
    assert md_path.exists() and json_path.exists()
    assert md_path.name == "2026-06-23-report.md"
    assert "Touchline Edge Report" in md_path.read_text(encoding="utf-8")


def test_run_daily_empty_quotes_still_writes(tmp_path):
    md_path, json_path = run_daily(
        ratings=_ratings(), overlay={}, quotes=[], fixtures=[], history=[],
        team_games={}, as_of="2026-06-23", out_dir=tmp_path,
    )
    assert md_path.exists() and json_path.exists()
