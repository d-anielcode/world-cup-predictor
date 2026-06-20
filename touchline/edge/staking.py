from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stake:
    fraction: float  # fraction of bankroll to wager
    amount: float    # dollars to wager


def kelly_fraction(prob: float, price: float) -> float:
    """Full-Kelly bankroll fraction for a Kalshi YES contract.

    Buying YES at `price` (dollars, 0..1) risks `price` to win `1 - price`. With
    model win-probability `prob`, Kelly is f* = (prob - price) / (1 - price). Returns
    0 for a non-value bet or a degenerate price (<=0 or >=1)."""
    if price <= 0.0 or price >= 1.0:
        return 0.0
    return max(0.0, (prob - price) / (1.0 - price))


def size_stakes(
    picks,
    bankroll: float,
    kelly_multiplier: float = 0.25,
    max_bet_fraction: float = 0.05,
    max_game_fraction: float = 0.10,
) -> list[Stake]:
    """Recommend a wager for each ranked pick, in pick order.

    Discipline layered to control variance:
      * Fractional Kelly (`kelly_multiplier`, default quarter-Kelly) — Kelly assumes
        the true probability is known; our prob is an estimate, so we bet a fraction.
      * Confidence shrink — multiply by the pick's own confidence (sample depth,
        longshot direction, market trust), so thin/uncertain edges stake less.
      * Per-bet cap (`max_bet_fraction`) — no single wager dominates the bankroll.
      * Per-game cap (`max_game_fraction`) — markets on the same match are positively
        correlated (a blowout hits the winner, the spread and the over together), so
        independent Kelly over-bets the cluster; the per-game total is capped and the
        game's wagers scaled down proportionally.

    Each pick must expose `.home`, `.away`, and `.edge` with `.model_prob`,
    `.market_price`, `.confidence`.
    """
    raw: list[float] = []
    for p in picks:
        e = p.edge
        kelly = kelly_fraction(e.model_prob, e.market_price)
        frac = kelly_multiplier * e.confidence * kelly
        raw.append(min(frac, max_bet_fraction))

    # Per-game correlation cap: scale a game's wagers so their total <= the cap.
    game_totals: dict[tuple, float] = {}
    for p, frac in zip(picks, raw):
        game_totals[(p.home, p.away)] = game_totals.get((p.home, p.away), 0.0) + frac

    stakes: list[Stake] = []
    for p, frac in zip(picks, raw):
        total = game_totals[(p.home, p.away)]
        if total > max_game_fraction and total > 0:
            frac *= max_game_fraction / total
        stakes.append(Stake(fraction=frac, amount=round(frac * bankroll, 2)))
    return stakes
