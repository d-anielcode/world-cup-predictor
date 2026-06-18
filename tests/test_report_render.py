import json
from touchline.edge.edge import Edge
from touchline.edge.rank import RankedPick
from touchline.report.render import render_markdown, render_json


def _pick(home, away, mt, side, edge_val, conf):
    e = Edge(model_prob=0.6, market_price=0.5, edge=edge_val,
             ev_per_dollar=edge_val / 0.5, confidence=conf, recommendation="BUY")
    return RankedPick(home=home, away=away, market_type=mt, side=side, line=None,
                      edge=e, score=edge_val * conf)


def test_markdown_has_top_predictions_section_and_rows():
    picks = [_pick("USA", "Wales", "1x2", "home", 0.10, 0.9)]
    md = render_markdown(picks, as_of="2026-06-24")
    assert "# Touchline Edge Report" in md
    assert "Top Predictions" in md
    assert "USA" in md and "Wales" in md
    assert "10.0%" in md or "0.10" in md


def test_markdown_handles_empty_picks():
    md = render_markdown([], as_of="2026-06-24")
    assert "No positive-edge" in md


def test_json_roundtrips_pick_fields():
    picks = [_pick("USA", "Wales", "total", "over", 0.08, 0.7)]
    payload = json.loads(render_json(picks, as_of="2026-06-24"))
    assert payload["as_of"] == "2026-06-24"
    row = payload["picks"][0]
    assert row["home"] == "USA"
    assert row["market_type"] == "total"
    assert abs(row["edge"] - 0.08) < 1e-9
    assert abs(row["confidence"] - 0.7) < 1e-9
