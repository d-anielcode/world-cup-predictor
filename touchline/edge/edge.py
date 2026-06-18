from __future__ import annotations

from dataclasses import dataclass

_LONGSHOT_PRICE = 0.25     # value bets priced below this are down-weighted
_SAMPLE_TARGET = 20.0      # games-per-team at which sample confidence saturates
_EDGE_BUY_THRESHOLD = 0.0  # positive edge => BUY


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


def compute_edge(model_prob: float, market_price: float, min_games: int) -> Edge:
    """Compare a model probability to a market price.

    `min_games` is the smaller of the two teams' played-match counts (proxy for how
    much the rating leans on the Elo prior). Confidence combines sample depth and the
    favorite-longshot direction of the value bet.
    """
    edge = model_prob - market_price
    ev = (model_prob - market_price) / market_price if market_price > 0 else 0.0
    confidence = _sample_confidence(min_games) * _longshot_confidence(market_price)
    recommendation = "BUY" if edge > _EDGE_BUY_THRESHOLD else "PASS"
    return Edge(
        model_prob=model_prob, market_price=market_price, edge=edge,
        ev_per_dollar=ev, confidence=confidence, recommendation=recommendation,
    )
