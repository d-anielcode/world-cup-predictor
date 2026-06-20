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
# Extra goal-supremacy boost for a genuine tournament host, ON TOP of the rating's
# generic home_adv. Measured residual of host overperformance vs the model
# (Brazil'14/Russia'18/Qatar'22/2026 hosts, N=17) was ~+0.07 goals/game and within
# noise, so this is deliberately modest. Operator-tunable per tournament; set to 0
# to disable. It does NOT close a large model-vs-market gap on a host — that is a
# base-rating (Elo prior) issue, not a host-edge one.
HOST_ADVANTAGE = 0.08


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
    # The side (if any) that is the genuine tournament host playing in its own
    # country. Decoupled from home/away listing so a host listed as the away team
    # still receives its host edge.
    home_is_host: bool = False
    away_is_host: bool = False


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


def _host_mults(ctx: FactorContext) -> tuple[float, float]:
    """(lam_mult, mu_mult) for the host edge: a symmetric log-supremacy split that
    lifts the host's goals and trims the opponent's, mirroring how home_adv splits."""
    if ctx.home_is_host:
        return math.exp(HOST_ADVANTAGE / 2), math.exp(-HOST_ADVANTAGE / 2)
    if ctx.away_is_host:
        return math.exp(-HOST_ADVANTAGE / 2), math.exp(HOST_ADVANTAGE / 2)
    return 1.0, 1.0


def adjust_expected_goals(
    lam: float, mu: float, ctx: FactorContext
) -> tuple[float, float]:
    """Apply travel/altitude/heat/rest/host multipliers to expected goals."""
    heat = _heat_mult(ctx.wbgt_c)
    host_lam, host_mu = _host_mults(ctx)
    lam *= _travel_mult(ctx.travel_km_home)
    lam *= _altitude_mult(ctx.altitude_m, ctx.home_altitude_acclimatized)
    lam *= _rest_deficit_mult(ctx.rest_days_home, ctx.rest_days_away)
    lam *= heat * host_lam
    mu *= _travel_mult(ctx.travel_km_away)
    mu *= _altitude_mult(ctx.altitude_m, ctx.away_altitude_acclimatized)
    mu *= _rest_deficit_mult(ctx.rest_days_away, ctx.rest_days_home)
    mu *= heat * host_mu
    return lam, mu
