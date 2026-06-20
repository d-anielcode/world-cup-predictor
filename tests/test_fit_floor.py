"""The min-games ridge floor keeps a fringe team (few matches, lopsided results)
from overfitting to extreme ratings."""
from datetime import date

from touchline.data.elo import EloTable
from touchline.models import Match
from touchline.model.fit import fit_ratings


def _m(home, away, hg, ag, day):
    return Match(date(2026, 1, day), home, away, hg, ag, "T", None, None, True, "t")


def _dataset():
    # Four well-sampled teams playing a round-robin twice, plus a "Minnow" that
    # appears in a single 9-0 thrashing — exactly the thin-sample blowup case.
    teams = ["A", "B", "C", "D"]
    matches, day = [], 1
    for _ in range(3):
        for i in range(len(teams)):
            for j in range(len(teams)):
                if i != j:
                    matches.append(_m(teams[i], teams[j], 1, 1, day))
                    day = day % 27 + 1
    matches.append(_m("Minnow", "A", 9, 0, 28))  # single lopsided result
    return matches


def _minnow_attack(prior_min_games):
    r = fit_ratings(_dataset(), EloTable(), half_life_days=900.0, prior_weight=0.05,
                    as_of=date(2026, 2, 1), prior_min_games=prior_min_games)
    return r.attack["Minnow"]


def test_floor_shrinks_thin_sample_team_toward_prior():
    no_floor = _minnow_attack(0.0)
    floored = _minnow_attack(8.0)
    # Without a floor the single 9-0 inflates Minnow's attack; the floor pulls it
    # back toward the (zero) prior, so the floored estimate is markedly smaller.
    assert no_floor > 0.5
    assert floored < no_floor
    assert abs(floored) < abs(no_floor)


def test_floor_leaves_data_rich_team_essentially_unchanged():
    # A well-sampled team's team_w is far above the floor, so its rating barely moves.
    base = fit_ratings(_dataset(), EloTable(), half_life_days=900.0, prior_weight=0.05,
                       as_of=date(2026, 2, 1), prior_min_games=0.0)
    floored = fit_ratings(_dataset(), EloTable(), half_life_days=900.0, prior_weight=0.05,
                          as_of=date(2026, 2, 1), prior_min_games=8.0)
    assert abs(base.attack["A"] - floored.attack["A"]) < 0.05
