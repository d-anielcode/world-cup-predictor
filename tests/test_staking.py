from dataclasses import dataclass

from touchline.edge.staking import kelly_fraction, size_stakes


@dataclass
class _Edge:
    model_prob: float
    market_price: float
    confidence: float = 1.0


@dataclass
class _Pick:
    home: str
    away: str
    edge: _Edge


def _pick(home, away, prob, price, conf=1.0):
    return _Pick(home, away, _Edge(prob, price, conf))


def test_kelly_fraction_matches_formula():
    # YES at 0.50, model 0.60 -> (0.60-0.50)/(1-0.50) = 0.20
    assert abs(kelly_fraction(0.60, 0.50) - 0.20) < 1e-9


def test_kelly_zero_for_non_value_or_degenerate_price():
    assert kelly_fraction(0.40, 0.50) == 0.0   # model below price
    assert kelly_fraction(0.9, 1.0) == 0.0     # price 1.0
    assert kelly_fraction(0.9, 0.0) == 0.0     # price 0.0


def test_fractional_kelly_and_confidence_shrink_stake():
    # full kelly 0.20 of $1000 = $200; quarter-kelly = $50; conf 0.5 halves -> $25.
    picks = [_pick("A", "B", 0.60, 0.50, conf=0.5)]
    stakes = size_stakes(picks, bankroll=1000.0, kelly_multiplier=0.25,
                         max_bet_fraction=1.0, max_game_fraction=1.0)
    assert abs(stakes[0].amount - 25.0) < 1e-6


def test_per_bet_cap_limits_a_single_stake():
    picks = [_pick("A", "B", 0.95, 0.10, conf=1.0)]  # huge kelly
    stakes = size_stakes(picks, bankroll=1000.0, kelly_multiplier=1.0,
                         max_bet_fraction=0.05, max_game_fraction=1.0)
    assert abs(stakes[0].amount - 50.0) < 1e-6  # capped at 5% of 1000


def test_per_game_cap_scales_correlated_bets_down():
    # Two markets on the SAME game, each wanting 8% -> 16% > 10% game cap -> scaled
    # to total 10% ($100), split proportionally (equal here -> $50 each).
    picks = [_pick("A", "B", 0.58, 0.50, conf=1.0),
             _pick("A", "B", 0.58, 0.50, conf=1.0)]
    stakes = size_stakes(picks, bankroll=1000.0, kelly_multiplier=1.0,
                         max_bet_fraction=1.0, max_game_fraction=0.10)
    total = stakes[0].amount + stakes[1].amount
    assert abs(total - 100.0) < 1e-6
    assert abs(stakes[0].amount - stakes[1].amount) < 1e-6


def test_stakes_align_one_to_one_with_picks():
    picks = [_pick("A", "B", 0.60, 0.50), _pick("C", "D", 0.40, 0.50)]
    stakes = size_stakes(picks, bankroll=100.0)
    assert len(stakes) == 2
    assert stakes[1].amount == 0.0  # second is a non-value bet
