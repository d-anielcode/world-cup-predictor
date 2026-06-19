from datetime import date, timedelta

import numpy as np

from touchline.models import Match
from touchline.data.elo import EloTable
from touchline.model.fit import fit_ratings


def _synthetic_matches(seed: int = 0) -> list[Match]:
    rng = np.random.default_rng(seed)
    true_attack = {"Strong": 0.6, "Mid": 0.0, "Weak": -0.6}
    true_defense = {"Strong": 0.5, "Mid": 0.0, "Weak": -0.5}
    teams = list(true_attack)
    base = date(2026, 1, 1)
    matches: list[Match] = []
    for i in range(300):
        h, a = rng.choice(teams, size=2, replace=False)
        lam = np.exp(true_attack[h] - true_defense[a])
        mu = np.exp(true_attack[a] - true_defense[h])
        matches.append(Match(
            match_date=base + timedelta(days=i % 120),
            home_team=str(h), away_team=str(a),
            home_goals=int(rng.poisson(lam)), away_goals=int(rng.poisson(mu)),
            competition="Synthetic", stage=None, venue=None,
            played=True, source="test",
        ))
    return matches


def test_fit_recovers_strength_ordering():
    matches = _synthetic_matches()
    elo = EloTable()
    r = fit_ratings(matches, elo, half_life_days=400, prior_weight=0.01,
                    as_of=date(2026, 6, 1))
    assert r.attack["Strong"] > r.attack["Mid"] > r.attack["Weak"]
    assert r.defense["Strong"] > r.defense["Mid"] > r.defense["Weak"]


def test_unplayed_matches_are_ignored():
    matches = _synthetic_matches()
    matches.append(Match(
        match_date=date(2026, 6, 1), home_team="Strong", away_team="Weak",
        home_goals=None, away_goals=None, competition="Synthetic", stage=None,
        venue=None, played=False, source="test",
    ))
    r = fit_ratings(matches, EloTable(), half_life_days=400, prior_weight=0.01,
                    as_of=date(2026, 6, 1))
    assert "Strong" in r.attack


def test_elo_prior_dominates_for_team_with_no_games():
    # Two no-game teams: a high-Elo Ghost and a neutral-Elo NeutralGhost. The
    # prior must actually move Ghost above NeutralGhost (not just above data teams).
    matches = _synthetic_matches()
    elo = EloTable(by_norm={"strong": 1500.0, "mid": 1500.0, "weak": 1500.0,
                            "ghost": 2300.0, "neutralghost": 1500.0})
    r = fit_ratings(matches, elo, half_life_days=400, prior_weight=5.0,
                    as_of=date(2026, 6, 1), extra_teams=["Ghost", "NeutralGhost"])
    assert r.attack["Ghost"] > r.attack["Weak"]
    assert r.attack["Ghost"] > r.attack["NeutralGhost"]


def test_fit_recovers_goal_level_via_intercept():
    import numpy as np
    from datetime import date, timedelta
    rng = np.random.default_rng(3)
    teams = ["A", "B", "C"]
    base = date(2026, 1, 1)
    matches = []
    base_rate = 1.8
    for i in range(400):
        h, a = rng.choice(teams, size=2, replace=False)
        matches.append(Match(
            match_date=base + timedelta(days=i % 120), home_team=str(h),
            away_team=str(a), home_goals=int(rng.poisson(base_rate)),
            away_goals=int(rng.poisson(base_rate)), competition="Syn", stage=None,
            venue=None, played=True, source="t"))
    r = fit_ratings(matches, EloTable(), half_life_days=400, prior_weight=0.01,
                    as_of=date(2026, 6, 1))
    lam, mu = r.expected_goals("A", "B", apply_home_adv=False)
    assert 3.0 < lam + mu < 4.2


def test_team_rating_not_crushed_by_unrelated_dataset_size():
    # A team's rating must depend on ITS games, not the global dataset size.
    # The old global-wsum ridge over-shrinks every team toward the prior as
    # unrelated matches inflate wsum; the per-team ridge keeps it stable.
    import math
    from datetime import date, timedelta

    def make(padding_pairs):
        base = date(2026, 1, 1)
        ms = []
        for i in range(30):                       # Strong hammers Weak 4-0
            ms.append(Match(match_date=base + timedelta(days=i), home_team="Strong",
                            away_team="Weak", home_goals=4, away_goals=0,
                            competition="x", stage=None, venue=None, played=True, source="t"))
        for i in range(padding_pairs):            # unrelated teams, all 1-1 draws
            ms.append(Match(match_date=base + timedelta(days=i), home_team=f"P{i % 40}",
                            away_team=f"Q{i % 40}", home_goals=1, away_goals=1,
                            competition="x", stage=None, venue=None, played=True, source="t"))
        return ms

    # Long half-life so the padding fully inflates the global weight sum.
    r_big = fit_ratings(make(2000), EloTable(), half_life_days=5000,
                        prior_weight=0.05, as_of=date(2026, 6, 1))
    # With an uninformative (empty) prior, Strong must stay clearly above Weak.
    assert r_big.attack["Strong"] - r_big.attack["Weak"] > 1.0


def test_fit_converges_without_warning():
    # The analytic gradient must let the optimizer actually converge (no warning).
    import warnings
    matches = _synthetic_matches()
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # convergence warning -> error
        fit_ratings(matches, EloTable(), half_life_days=400, prior_weight=0.05,
                    as_of=date(2026, 6, 1))
