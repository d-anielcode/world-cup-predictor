from __future__ import annotations

from dataclasses import dataclass

from touchline.backtest.scoring import brier_score, log_loss


@dataclass
class MarketComparison:
    n: int
    model_brier: float
    market_brier: float
    model_log_loss: float
    market_log_loss: float
    model_pnl: float       # realized profit (per $1 flat stake) summed over value bets
    n_bets: int
    model_top_acc: float
    market_top_acc: float


def _normalize(triple: tuple[float, float, float]) -> tuple[float, float, float]:
    s = sum(triple)
    return tuple(x / s for x in triple) if s > 0 else triple


def score_vs_market(games: list[dict]) -> MarketComparison:
    """Compare the model's 1X2 probabilities to the market's pre-kickoff prices.

    Each game is {"outcome": 0/1/2, "model": (h,d,a), "market": (h,d,a) raw prices}.
    The market prices are normalized (vig removed) for a fair Brier/log-loss vs the
    realized outcome — the key question is whether the model out-predicts the market.
    The P&L is what you'd realize flat-staking $1 on every positive-edge side at its
    raw Kalshi price: profit (1 - price) if it wins, -price if it loses.
    """
    model_probs, market_probs, outcomes = [], [], []
    pnl, n_bets, model_hits, market_hits = 0.0, 0, 0, 0
    for g in games:
        o = g["outcome"]
        model = tuple(g["model"])
        raw = tuple(g["market"])
        norm = _normalize(raw)
        model_probs.append(model)
        market_probs.append(norm)
        outcomes.append(o)
        if max(range(3), key=lambda i: model[i]) == o:
            model_hits += 1
        if max(range(3), key=lambda i: norm[i]) == o:
            market_hits += 1
        for s in range(3):
            if model[s] > raw[s] and raw[s] > 0:  # positive-edge value bet
                pnl += (1.0 - raw[s]) if o == s else -raw[s]
                n_bets += 1
    n = len(games)
    return MarketComparison(
        n=n,
        model_brier=brier_score(model_probs, outcomes),
        market_brier=brier_score(market_probs, outcomes),
        model_log_loss=log_loss(model_probs, outcomes),
        market_log_loss=log_loss(market_probs, outcomes),
        model_pnl=round(pnl, 4),
        n_bets=n_bets,
        model_top_acc=model_hits / n if n else 0.0,
        market_top_acc=market_hits / n if n else 0.0,
    )
