from __future__ import annotations

import json

from touchline.edge.rank import RankedPick


def _label(p: RankedPick) -> str:
    line = "" if p.line is None else f" {p.line}"
    return f"{p.market_type}:{p.side}{line}"


def render_markdown(picks: list[RankedPick], as_of: str) -> str:
    lines = [f"# Touchline Edge Report", f"_As of {as_of}_", ""]
    lines.append("## Top Predictions")
    if not picks:
        lines.append("")
        lines.append("No positive-edge predictions found.")
        return "\n".join(lines)
    lines.append("")
    lines.append("| # | Match | Market | Model | Market | Edge | Conf |")
    lines.append("|---|-------|--------|-------|--------|------|------|")
    for i, p in enumerate(picks, 1):
        lines.append(
            f"| {i} | {p.home} v {p.away} | {_label(p)} | "
            f"{p.edge.model_prob:.0%} | {p.edge.market_price:.0%} | "
            f"{p.edge.edge*100:.1f}% | {p.edge.confidence:.2f} |"
        )
    return "\n".join(lines)


def render_json(picks: list[RankedPick], as_of: str) -> str:
    return json.dumps({
        "as_of": as_of,
        "picks": [
            {
                "home": p.home, "away": p.away, "market_type": p.market_type,
                "side": p.side, "line": p.line,
                "model_prob": p.edge.model_prob, "market_price": p.edge.market_price,
                "edge": p.edge.edge, "ev_per_dollar": p.edge.ev_per_dollar,
                "confidence": p.edge.confidence, "score": p.score,
            }
            for p in picks
        ],
    }, indent=2)
