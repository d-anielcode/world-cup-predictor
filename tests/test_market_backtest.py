from datetime import date

from touchline.data.kalshi_history import pre_kickoff_price, decode_suffix_date
from touchline.backtest.market import score_vs_market, score_binary_vs_market


def _candle(ts, bid, ask):
    return {"end_period_ts": ts,
            "yes_bid": {"close_dollars": bid},
            "yes_ask": {"close_dollars": ask}}


def test_pre_kickoff_price_takes_last_candle_before_kickoff():
    candles = [_candle(100, "0.40", "0.60"), _candle(200, "0.50", "0.54"),
               _candle(300, "0.70", "0.80")]
    assert abs(pre_kickoff_price(candles, 250) - 0.52) < 1e-9  # ts=200 mid


def test_pre_kickoff_price_none_when_no_candle_before():
    assert pre_kickoff_price([_candle(300, "0.5", "0.6")], 100) is None


def test_pre_kickoff_price_uses_one_sided_quote():
    # No bid -> fall back to the ask (the candle still informs the level).
    assert abs(pre_kickoff_price([_candle(100, "0", "0.30")], 200) - 0.30) < 1e-9


def test_decode_suffix_date():
    assert decode_suffix_date("26JUN19TURPAR") == date(2026, 6, 19)
    assert decode_suffix_date("26JAN05ABCDEF") == date(2026, 1, 5)
    assert decode_suffix_date("garbage") is None


def test_score_vs_market_model_beats_when_more_accurate():
    games = [
        {"outcome": 0, "model": (0.70, 0.20, 0.10), "market": (0.50, 0.30, 0.20)},
        {"outcome": 0, "model": (0.60, 0.25, 0.15), "market": (0.45, 0.30, 0.25)},
    ]
    r = score_vs_market(games)
    assert r.n == 2
    assert r.model_brier < r.market_brier
    assert r.model_log_loss < r.market_log_loss


def test_score_vs_market_pnl_positive_when_value_pick_wins():
    # Model likes home 0.70 vs market raw 0.50; home wins -> profit (1-0.50)=+0.50.
    games = [{"outcome": 0, "model": (0.70, 0.20, 0.10), "market": (0.50, 0.30, 0.20)}]
    r = score_vs_market(games)
    assert r.n_bets == 1
    assert abs(r.model_pnl - 0.50) < 1e-9


def test_score_vs_market_pnl_negative_when_value_pick_loses():
    games = [{"outcome": 2, "model": (0.70, 0.20, 0.10), "market": (0.50, 0.30, 0.20)}]
    r = score_vs_market(games)
    assert r.n_bets == 1
    assert abs(r.model_pnl - (-0.50)) < 1e-9  # lost the stake (price 0.50)


def test_score_binary_vs_market_brier_and_pnl():
    recs = [
        {"model": 0.70, "market": 0.55, "outcome": 1},  # value YES, hits -> +0.45
        {"model": 0.30, "market": 0.50, "outcome": 0},  # no bet (model below market)
    ]
    r = score_binary_vs_market("total", recs)
    assert r.market_type == "total" and r.n == 2
    assert r.n_bets == 1
    assert abs(r.model_pnl - 0.45) < 1e-9
    # model is more accurate on both rows -> lower Brier
    assert r.model_brier < r.market_brier
