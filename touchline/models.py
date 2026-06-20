from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

# Orientation authority for de-duplication: when two feeds list the same fixture
# with home/away swapped, we keep the orientation from the highest-priority
# (lowest-number) source. openfootball carries the official FIFA fixture order.
_SOURCE_PRIORITY = {"openfootball": 0, "worldcupjson": 1, "intl_results": 2, "result": 3}


@dataclass
class Match:
    match_date: date
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    competition: str
    stage: str | None
    venue: str | None
    played: bool
    source: str

    def natural_key(self) -> str:
        """Dedup key: date + both teams, order-independent.

        The teams are sorted so a fixture counts once regardless of which feed
        listed the nominal home team. Feeds disagree on home/away for neutral-site
        (e.g. most World Cup) games, which would otherwise create duplicate rows.
        Orientation is preserved in the home_team/away_team fields, not the key.
        """
        a, b = sorted((self.home_team, self.away_team))
        return f"{self.match_date.isoformat()}|{a}|{b}"


def dedupe_matches(matches: list[Match]) -> list[Match]:
    """Collapse duplicate fixtures (same date + team pair) into one record each.

    Two feeds can list the same match with the teams swapped, each recording
    goals relative to its own orientation. We pick the orientation from the
    most authoritative source (``_SOURCE_PRIORITY``) and re-align the goals from
    whichever row carries the played result, flipping them when that row used the
    opposite orientation. Non-identifying fields (competition, venue) are taken
    from the authoritative row, falling back to any non-null value in the group.
    """
    from touchline.data.teams import canonical_team

    groups: dict[tuple, list[tuple[Match, str, str]]] = {}
    for m in matches:
        home, away = canonical_team(m.home_team), canonical_team(m.away_team)
        key = (m.match_date, frozenset((home, away)))
        groups.setdefault(key, []).append((m, home, away))

    out: list[Match] = []
    for items in groups.values():
        primary, home_k, away_k = min(
            items, key=lambda t: _SOURCE_PRIORITY.get(t[0].source, 99))

        played = next((t for t in items
                       if t[0].played and t[0].home_goals is not None), None)
        if played is not None:
            pm, ph, _ = played
            if (ph, _) == (home_k, away_k):
                home_goals, away_goals = pm.home_goals, pm.away_goals
            else:  # played row used the opposite orientation -> flip goals
                home_goals, away_goals = pm.away_goals, pm.home_goals
            is_played = True
        else:
            home_goals = away_goals = None
            is_played = False

        venue = primary.venue or next(
            (t[0].venue for t in items if t[0].venue), None)
        stage = primary.stage or next(
            (t[0].stage for t in items if t[0].stage), None)
        out.append(replace(
            primary,
            home_team=home_k, away_team=away_k,
            home_goals=home_goals, away_goals=away_goals,
            stage=stage, venue=venue, played=is_played,
        ))
    return out


@dataclass
class MarketQuote:
    ticker: str
    series_ticker: str
    title: str
    yes_price: float  # dollars 0..1
    no_price: float
    status: str
    raw: dict
