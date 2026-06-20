from datetime import date

from touchline.models import Match, dedupe_matches


def _m(home, away, hg, ag, source, played=True, when=date(2026, 6, 12),
       comp="World Cup 2026", venue=None):
    return Match(when, home, away, hg, ag, comp, None, venue, played, source)


def test_swapped_orientation_collapses_and_aligns_goals():
    # openfootball lists the official fixture orientation (Cameroon home) as an
    # unplayed fixture; intl_results has the played result in the OPPOSITE
    # orientation (Brazil home, Brazil 4-1). The merge must keep openfootball's
    # orientation AND flip the goals so Cameroon=1, Brazil=4.
    of = _m("Cameroon", "Brazil", None, None, "openfootball", played=False)
    intl = _m("Brazil", "Cameroon", 4, 1, "intl_results")
    out = dedupe_matches([of, intl])
    assert len(out) == 1
    r = out[0]
    assert (r.home_team, r.away_team) == ("Cameroon", "Brazil")
    assert (r.home_goals, r.away_goals) == (1, 4)
    assert r.played is True


def test_spelling_variant_collapses():
    of = _m("Canada", "Bosnia & Herzegovina", 2, 1, "openfootball")
    intl = _m("Canada", "Bosnia-Herzegovina", 2, 1, "intl_results")
    out = dedupe_matches([of, intl])
    assert len(out) == 1
    assert out[0].away_team == "Bosnia-Herzegovina"
    assert (out[0].home_goals, out[0].away_goals) == (2, 1)


def test_distinct_matches_not_merged():
    a = _m("Brazil", "Serbia", 2, 0, "intl_results")
    b = _m("Brazil", "Serbia", 1, 1, "intl_results", when=date(2026, 6, 15))
    out = dedupe_matches([a, b])
    assert len(out) == 2


def test_single_match_orientation_preserved():
    a = _m("Spain", "France", 1, 0, "intl_results")
    out = dedupe_matches([a])
    assert (out[0].home_team, out[0].away_team) == ("Spain", "France")
    assert (out[0].home_goals, out[0].away_goals) == (1, 0)


def test_unified_competition_label_from_authoritative_source():
    of = _m("USA", "Turkey", None, None, "openfootball", played=False,
            comp="World Cup 2026", venue="MetLife Stadium")
    intl = _m("Turkey", "USA", 0, 2, "intl_results", comp="FIFA World Cup")
    out = dedupe_matches([of, intl])
    assert len(out) == 1
    assert out[0].competition == "World Cup 2026"
    assert out[0].venue == "MetLife Stadium"
    # USA home (openfootball), goals aligned: USA 2, Turkey 0
    assert (out[0].home_team, out[0].away_team) == ("USA", "Turkey")
    assert (out[0].home_goals, out[0].away_goals) == (2, 0)
