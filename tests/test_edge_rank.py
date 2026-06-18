from touchline.edge.edge import Edge
from touchline.edge.rank import RankedPick, rank_picks


def _edge(edge, conf, rec="BUY"):
    return Edge(model_prob=0.5 + edge, market_price=0.5, edge=edge,
                ev_per_dollar=edge / 0.5, confidence=conf, recommendation=rec)


def test_only_buys_with_positive_edge_are_ranked():
    picks = rank_picks([
        ("USA", "Wales", "1x2", "home", None, _edge(0.10, 1.0)),
        ("USA", "Wales", "1x2", "away", None, _edge(-0.05, 1.0, "PASS")),
    ])
    assert len(picks) == 1
    assert picks[0].side == "home"


def test_ranked_by_edge_times_confidence_descending():
    picks = rank_picks([
        ("A", "B", "1x2", "home", None, _edge(0.20, 0.3)),
        ("C", "D", "btts", "yes", None, _edge(0.10, 1.0)),
    ])
    assert [p.market_type for p in picks] == ["btts", "1x2"]
    assert isinstance(picks[0], RankedPick)
    assert picks[0].score > picks[1].score


def test_top_n_limits_results():
    edges = [("A", "B", "1x2", "home", None, _edge(0.10 + i / 100, 1.0)) for i in range(5)]
    picks = rank_picks(edges, top_n=2)
    assert len(picks) == 2
