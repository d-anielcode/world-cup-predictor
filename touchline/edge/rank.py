from __future__ import annotations

from dataclasses import dataclass

from touchline.edge.edge import Edge


@dataclass
class RankedPick:
    home: str
    away: str
    market_type: str
    side: str
    line: float | None
    edge: Edge
    score: float


def rank_picks(
    edges: list[tuple[str, str, str, str, float | None, Edge]],
    top_n: int | None = None,
) -> list[RankedPick]:
    """Keep BUY recommendations with positive edge, score by edge*confidence, sort desc.

    Each input tuple is (home, away, market_type, side, line, Edge).
    """
    picks: list[RankedPick] = []
    for home, away, market_type, side, line, e in edges:
        # Drop non-buys, non-positive edges, and zero-confidence picks (a team with
        # no played matches yields confidence 0 and is pure-prior noise).
        if e.recommendation != "BUY" or e.edge <= 0 or e.confidence <= 0:
            continue
        picks.append(RankedPick(
            home=home, away=away, market_type=market_type, side=side, line=line,
            edge=e, score=e.edge * e.confidence,
        ))
    picks.sort(key=lambda p: p.score, reverse=True)
    return picks[:top_n] if top_n is not None else picks
