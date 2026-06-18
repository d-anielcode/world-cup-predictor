from __future__ import annotations

from touchline.model.pricing import MarketProbs


def model_prob(
    probs: MarketProbs, market_type: str, side: str, line: float | None
) -> float:
    """Return the model probability for a normalized (market_type, side, line).

    Raises KeyError if a total/handicap line was not priced, or ValueError for an
    unknown market_type/side."""
    if market_type == "1x2":
        return {"home": probs.home, "draw": probs.draw, "away": probs.away}[side]
    if market_type == "total":
        over = probs.over[line]
        return over if side == "over" else 1.0 - over
    if market_type == "btts":
        return probs.btts_yes if side == "yes" else 1.0 - probs.btts_yes
    if market_type == "handicap":
        home = probs.home_handicap[line]
        return home if side == "home" else 1.0 - home
    raise ValueError(f"Unknown market_type/side: {market_type}/{side}")
