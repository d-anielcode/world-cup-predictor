import numpy as np

from touchline.model.pricing import price_matrix, prob_correct_score
from touchline.model.dixon_coles import scoreline_matrix
from touchline.edge.model_lookup import model_prob
from touchline.edge.quotes import MarketQuoteRow, fixture_score_lines
from touchline.data.kalshi_quotes import parse_score_market


def _matrix():
    return scoreline_matrix(1.4, 1.0, -0.05, max_goals=8)


def test_prob_correct_score_reads_cell():
    m = _matrix()
    assert abs(prob_correct_score(m, 2, 1) - float(m[2, 1])) < 1e-12


def test_price_matrix_fills_requested_scorelines():
    probs = price_matrix(_matrix(), score_lines=[(2, 1), (0, 0), (1, 1)])
    assert set(probs.correct_score) == {(2, 1), (0, 0), (1, 1)}
    assert all(0 <= p <= 1 for p in probs.correct_score.values())


def test_model_lookup_correct_score():
    probs = price_matrix(_matrix(), score_lines=[(2, 1)])
    got = model_prob(probs, "correct_score", "2-1", None)
    assert abs(got - probs.correct_score[(2, 1)]) < 1e-12


def test_parse_score_market_win_orientation():
    raw = {"title": "Will the final score be New Zealand wins 3-2?",
           "yes_sub_title": "New Zealand wins 3-2", "yes_ask_dollars": "0.0400",
           "yes_bid_dollars": "0.0000", "ticker": "KXWCSCORE-26JUN21NZLEGY-NZL3EGY2"}
    q = parse_score_market(raw, "New Zealand", "Egypt")
    assert q.market_type == "correct_score"
    assert q.side == "3-2"          # home 3, away 2
    assert q.price == 0.04          # ask-priced (no bid on these markets)


def test_parse_score_market_away_winner_flips():
    raw = {"title": "Will the final score be Egypt wins 4-3?",
           "yes_sub_title": "Egypt wins 4-3", "yes_ask_dollars": "0.0100",
           "yes_bid_dollars": "0.0000", "ticker": "KXWCSCORE-26JUN21NZLEGY-NZL3EGY4"}
    q = parse_score_market(raw, "New Zealand", "Egypt")
    assert q.side == "3-4"          # home (NZ) 3, away (Egypt) 4


def test_parse_score_market_draw():
    raw = {"title": "Will the final score be Draw 2-2?",
           "yes_sub_title": "Draw 2-2", "yes_ask_dollars": "0.0400",
           "yes_bid_dollars": "0.0000", "ticker": "KXWCSCORE-26JUN21NZLEGY-NZL2EGY2"}
    q = parse_score_market(raw, "New Zealand", "Egypt")
    assert q.side == "2-2"


def test_fixture_score_lines_collects_tuples():
    rows = [
        MarketQuoteRow("A", "B", "correct_score", "2-1", None, 0.08),
        MarketQuoteRow("A", "B", "correct_score", "1-1", None, 0.10),
        MarketQuoteRow("A", "B", "1x2", "home", None, 0.5),
        MarketQuoteRow("C", "D", "correct_score", "0-0", None, 0.09),
    ]
    assert fixture_score_lines(rows, "A", "B") == [(1, 1), (2, 1)]
