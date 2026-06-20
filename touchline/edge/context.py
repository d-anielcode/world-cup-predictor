from __future__ import annotations

from datetime import date

from touchline.data.venues import get_venue, haversine_km, is_host_country
from touchline.model.factors import FactorContext
from touchline.models import Match


def _last_match_before(team: str, when: date, history: list[Match]) -> Match | None:
    prior = [m for m in history
             if (m.home_team == team or m.away_team == team) and m.match_date < when]
    return max(prior, key=lambda m: m.match_date) if prior else None


def _travel_and_rest(team, when, venue, history):
    last = _last_match_before(team, when, history)
    if last is None:
        return 0.0, None
    rest = (when - last.match_date).days
    travel = 0.0
    if last.venue:
        try:
            prev, cur = get_venue(last.venue), venue
            travel = haversine_km(prev.lat, prev.lon, cur.lat, cur.lon)
        except KeyError:
            travel = 0.0
    return travel, rest


def build_context(
    home: str, away: str, when: date, venue_name: str, history: list[Match],
    wbgt_c: float | None = None,
) -> FactorContext:
    """Build a FactorContext for an upcoming fixture.

    `wbgt_c` is the heat estimate (None if unavailable). Unknown venue names (e.g.
    historical stadiums absent from the 2026 table) yield a neutral context rather
    than raising, so the same code path works for backtests over historical data."""
    try:
        venue = get_venue(venue_name)
    except KeyError:
        return FactorContext(wbgt_c=wbgt_c)
    travel_h, rest_h = _travel_and_rest(home, when, venue, history)
    travel_a, rest_a = _travel_and_rest(away, when, venue, history)
    home_host = is_host_country(home) and venue.country == home
    away_host = is_host_country(away) and venue.country == away
    return FactorContext(
        travel_km_home=travel_h,
        travel_km_away=travel_a,
        altitude_m=venue.altitude_m,
        home_altitude_acclimatized=home_host,
        away_altitude_acclimatized=away_host,
        wbgt_c=wbgt_c,
        rest_days_home=rest_h,
        rest_days_away=rest_a,
        home_is_host=home_host,
        away_is_host=away_host,
    )
