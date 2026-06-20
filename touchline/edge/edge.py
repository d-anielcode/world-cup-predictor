from __future__ import annotations

from dataclasses import dataclass

_LONGSHOT_PRICE = 0.25     # value bets priced below this are down-weighted
_SAMPLE_TARGET = 20.0      # games-per-team at which sample confidence saturates
_EDGE_BUY_THRESHOLD = 0.0  # positive edge => BUY (not a calibration target: any
                           # positive-EV bet qualifies; tuning happens on confidence)
# How much to trust each market type, from the backtest: 1X2 and spreads beat the
# base-rate predictor; totals and BTTS are well-calibrated but don't out-discriminate
# it (near-random per-match goal counts), so their edges are down-weighted.
# correct_score is the lowest-trust market: exact scorelines are high-variance and
# the goal model is calibrated for aggregates (1X2/totals), not individual cells.
_MARKET_TRUST = {"1x2": 1.0, "handicap": 1.0, "total": 0.6, "btts": 0.6,
                 "correct_score": 0.4}
_DEFAULT_TRUST = 1.0


@dataclass
class Edge:
    model_prob: float
    market_price: float
    edge: float
    ev_per_dollar: float
    confidence: float
    recommendation: str


def _sample_confidence(min_games: int) -> float:
    return min(1.0, max(0, min_games) / _SAMPLE_TARGET)


def _longshot_confidence(market_price: float) -> float:
    """Down-weight value bets that sit on a longshot price (overpriced tail)."""
    if market_price >= _LONGSHOT_PRICE:
        return 1.0
    return max(0.3, market_price / _LONGSHOT_PRICE)


def compute_edge(
    model_prob: float, market_price: float, min_games: int, market_type: str = "1x2"
) -> Edge:
    """Compare a model probability to a market price.

    `min_games` is the smaller of the two teams' played-match counts (proxy for how
    much the rating leans on the Elo prior). Confidence combines sample depth, the
    favorite-longshot direction of the value bet, and how trustworthy the market type
    is (per the backtest: spreads/1X2 > totals/BTTS).
    """
    edge = model_prob - market_price
    ev = (model_prob - market_price) / market_price if market_price > 0 else 0.0
    trust = _MARKET_TRUST.get(market_type, _DEFAULT_TRUST)
    confidence = _sample_confidence(min_games) * _longshot_confidence(market_price) * trust
    recommendation = "BUY" if edge > _EDGE_BUY_THRESHOLD else "PASS"
    return Edge(
        model_prob=model_prob, market_price=market_price, edge=edge,
        ev_per_dollar=ev, confidence=confidence, recommendation=recommendation,
    )
