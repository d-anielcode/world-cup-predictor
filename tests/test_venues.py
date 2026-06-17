from touchline.data.venues import VENUES, get_venue, haversine_km, is_host_country


def test_known_venue_has_full_metadata():
    v = get_venue("Estadio Azteca")
    assert v.city == "Mexico City"
    assert v.country == "Mexico"
    assert v.altitude_m > 2000          # high-altitude venue
    assert v.has_roof_ac is False


def test_host_country_detection():
    assert is_host_country("Mexico") is True
    assert is_host_country("USA") is True
    assert is_host_country("Canada") is True
    assert is_host_country("Brazil") is False


def test_haversine_between_known_cities():
    # New York to Los Angeles ~ 3936 km
    nyc = (40.81, -74.07)
    la = (33.95, -118.34)
    d = haversine_km(nyc[0], nyc[1], la[0], la[1])
    assert 3800 < d < 4100


def test_all_venues_have_coordinates():
    for v in VENUES.values():
        assert -90 <= v.lat <= 90
        assert -180 <= v.lon <= 180
