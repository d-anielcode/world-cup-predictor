from __future__ import annotations

import json

from touchline.edge.rank import RankedPick
from touchline.edge.staking import Stake


def _label(p: RankedPick) -> str:
    line = "" if p.line is None else f" {p.line}"
    return f"{p.market_type}:{p.side}{line}"


def render_markdown(
    picks: list[RankedPick], as_of: str, stakes: list[Stake] | None = None
) -> str:
    lines = [f"# Touchline Edge Report", f"_As of {as_of}_", ""]
    lines.append("## Top Predictions")
    if not picks:
        lines.append("")
        lines.append("No positive-edge predictions found.")
        return "\n".join(lines)
    lines.append("")
    stake_h = " Stake |" if stakes is not None else ""
    stake_sep = "-------|" if stakes is not None else ""
    lines.append(f"| # | Match | Market | Model | Market | Edge | Conf |{stake_h}")
    lines.append(f"|---|-------|--------|-------|--------|------|------|{stake_sep}")
    for i, p in enumerate(picks, 1):
        stake_c = f" ${stakes[i-1].amount:.2f} |" if stakes is not None else ""
        lines.append(
            f"| {i} | {p.home} v {p.away} | {_label(p)} | "
            f"{p.edge.model_prob:.0%} | {p.edge.market_price:.0%} | "
            f"{p.edge.edge*100:.1f}% | {p.edge.confidence:.2f} |{stake_c}"
        )
    return "\n".join(lines)


def render_json(
    picks: list[RankedPick], as_of: str, stakes: list[Stake] | None = None
) -> str:
    out = []
    for i, p in enumerate(picks):
        row = {
            "home": p.home, "away": p.away, "market_type": p.market_type,
            "side": p.side, "line": p.line,
            "model_prob": p.edge.model_prob, "market_price": p.edge.market_price,
            "edge": p.edge.edge, "ev_per_dollar": p.edge.ev_per_dollar,
            "confidence": p.edge.confidence, "score": p.score,
        }
        if stakes is not None:
            row["stake"] = stakes[i].amount
            row["stake_fraction"] = stakes[i].fraction
        out.append(row)
    return json.dumps({"as_of": as_of, "picks": out}, indent=2)
