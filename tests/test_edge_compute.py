from touchline.edge.edge import compute_edge, Edge


def test_positive_edge_when_model_above_market():
    e = compute_edge(model_prob=0.60, market_price=0.50, min_games=20)
    assert isinstance(e, Edge)
    assert abs(e.edge - 0.10) < 1e-12
    assert e.recommendation == "BUY"


def test_negative_edge_recommends_pass():
    e = compute_edge(model_prob=0.40, market_price=0.50, min_games=20)
    assert e.edge < 0
    assert e.recommendation == "PASS"


def test_ev_per_dollar_is_edge_over_price():
    e = compute_edge(model_prob=0.60, market_price=0.50, min_games=20)
    assert abs(e.ev_per_dollar - (0.60 - 0.50) / 0.50) < 1e-12


def test_longshot_value_bet_has_lower_confidence():
    favorite = compute_edge(model_prob=0.70, market_price=0.60, min_games=50)
    longshot = compute_edge(model_prob=0.20, market_price=0.10, min_games=50)
    assert longshot.confidence < favorite.confidence


def test_thin_sample_lowers_confidence():
    deep = compute_edge(model_prob=0.60, market_price=0.50, min_games=50)
    thin = compute_edge(model_prob=0.60, market_price=0.50, min_games=2)
    assert thin.confidence < deep.confidence


def test_market_trust_downweights_weak_markets():
    # Same edge magnitude + sample, but totals/BTTS are weakly-discriminating per the
    # backtest, so their confidence must be below spreads/1X2.
    strong = compute_edge(0.60, 0.50, min_games=50, market_type="handicap")
    one_x2 = compute_edge(0.60, 0.50, min_games=50, market_type="1x2")
    total = compute_edge(0.60, 0.50, min_games=50, market_type="total")
    btts = compute_edge(0.60, 0.50, min_games=50, market_type="btts")
    assert one_x2.confidence == strong.confidence          # both fully trusted
    assert total.confidence < one_x2.confidence            # totals down-weighted
    assert btts.confidence < one_x2.confidence


def test_market_type_defaults_to_full_trust():
    # Back-compat: omitting market_type behaves like a fully-trusted market.
    default = compute_edge(0.60, 0.50, min_games=50)
    one_x2 = compute_edge(0.60, 0.50, min_games=50, market_type="1x2")
    assert default.confidence == one_x2.confidence
