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


def test_ampersand_variant_normalizes_to_canonical():
    # openfootball uses "Bosnia & Herzegovina"; intl_results uses the hyphen form.
    # The "&" must normalize to "and" so both map to one canonical spelling.
    assert canonical_team("Bosnia & Herzegovina") == "Bosnia-Herzegovina"
    assert canonical_team("Bosnia and Herzegovina") == "Bosnia-Herzegovina"
