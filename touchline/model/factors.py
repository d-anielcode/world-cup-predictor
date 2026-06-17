from __future__ import annotations

import math
from dataclasses import dataclass

# Tunable coefficients (defaults; calibrated in Plan 3 backtest).
TRAVEL_PER_1000KM = 0.03
HEAT_WBGT_THRESHOLD = 26.0
HEAT_PER_DEG = 0.02
ALTITUDE_THRESHOLD_M = 1500
ALTITUDE_PER_1000M = 0.06
REST_PER_DAY = 0.02


@dataclass
class FactorContext:
    travel_km_home: float = 0.0
    travel_km_away: float = 0.0
    altitude_m: int = 0
    home_altitude_acclimatized: bool = False
    away_altitude_acclimatized: bool = False
    wbgt_c: float | None = None
    rest_days_home: int | None = None
    rest_days_away: int | None = None


def _travel_mult(km: float) -> float:
    return math.exp(-TRAVEL_PER_1000KM * km / 1000.0)


def _altitude_mult(altitude_m: int, acclimatized: bool) -> float:
    if acclimatized or altitude_m <= ALTITUDE_THRESHOLD_M:
        return 1.0
    excess = (altitude_m - ALTITUDE_THRESHOLD_M) / 1000.0
    return math.exp(-ALTITUDE_PER_1000M * excess)


def _heat_mult(wbgt_c: float | None) -> float:
    if wbgt_c is None or wbgt_c <= HEAT_WBGT_THRESHOLD:
        return 1.0
    return math.exp(-HEAT_PER_DEG * (wbgt_c - HEAT_WBGT_THRESHOLD))


def _rest_deficit_mult(own: int | None, other: int | None) -> float:
    if own is None or other is None or own >= other:
        return 1.0
    return math.exp(-REST_PER_DAY * (other - own))


def adjust_expected_goals(
    lam: float, mu: float, ctx: FactorContext
) -> tuple[float, float]:
    """Apply travel/altitude/heat/rest multipliers to expected goals."""
    heat = _heat_mult(ctx.wbgt_c)
    lam *= _travel_mult(ctx.travel_km_home)
    lam *= _altitude_mult(ctx.altitude_m, ctx.home_altitude_acclimatized)
    lam *= _rest_deficit_mult(ctx.rest_days_home, ctx.rest_days_away)
    lam *= heat
    mu *= _travel_mult(ctx.travel_km_away)
    mu *= _altitude_mult(ctx.altitude_m, ctx.away_altitude_acclimatized)
    mu *= _rest_deficit_mult(ctx.rest_days_away, ctx.rest_days_home)
    mu *= heat
    return lam, mu
