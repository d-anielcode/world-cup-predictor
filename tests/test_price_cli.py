from datetime import date
from pathlib import Path

from touchline.models import Match
from touchline.model.ratings import Ratings
from touchline.edge.quotes import load_quotes
from touchline.cli import run_price


def _ratings():
    return Ratings(attack={"USA": 0.5, "Wales": -0.2},
                   defense={"USA": 0.3, "Wales": -0.2}, home_adv=0.2, rho=-0.05)


def test_run_price_produces_ranked_picks(tmp_path):
    quotes = load_quotes(Path("tests/fixtures/quotes_sample.csv"))
    history = [Match(match_date=date(2026, 6, 20), home_team="USA", away_team="Iran",
                     home_goals=2, away_goals=0, competition="WC", stage=None,
                     venue="MetLife Stadium", played=True, source="t")]
    fixtures = [("USA", "Wales", date(2026, 6, 24), "SoFi Stadium")]
    picks, md, js = run_price(
        ratings=_ratings(), overlay={}, quotes=quotes, fixtures=fixtures,
        history=history, team_games={"USA": 30, "Wales": 25}, as_of="2026-06-24",
    )
    assert isinstance(md, str) and "Touchline Edge Report" in md
    assert '"as_of": "2026-06-24"' in js
    assert all(p.edge.recommendation == "BUY" for p in picks)
