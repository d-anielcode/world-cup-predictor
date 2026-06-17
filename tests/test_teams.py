from touchline.data.teams import canonical_team


def test_maps_known_cross_source_variants():
    assert canonical_team("United States") == "USA"
    assert canonical_team("USA") == "USA"
    assert canonical_team("Korea Republic") == "South Korea"
    assert canonical_team("IR Iran") == "Iran"


def test_is_case_and_space_insensitive():
    assert canonical_team("  united states ") == "USA"


def test_unknown_team_passes_through_trimmed():
    assert canonical_team("  Brazil ") == "Brazil"
