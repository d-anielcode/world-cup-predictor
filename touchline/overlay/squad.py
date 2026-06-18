from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_MIN_MULT = 0.5
_MAX_MULT = 1.5


@dataclass
class TeamAdjustment:
    attack_mult: float
    defense_mult: float
    reason: str
    source: str


def load_overlay(path: Path) -> dict[str, TeamAdjustment]:
    """Load squad_adjustments.json. Missing file -> empty overlay.

    Multipliers must be within [0.5, 1.5]; anything outside is a likely typo and
    raises ValueError (a 5x swing would silently dominate the model)."""
    path = Path(path)
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    overlay: dict[str, TeamAdjustment] = {}
    for team, adj in raw.items():
        attack = float(adj.get("attack_mult", 1.0))
        defense = float(adj.get("defense_mult", 1.0))
        for m in (attack, defense):
            if not _MIN_MULT <= m <= _MAX_MULT:
                raise ValueError(
                    f"{team}: multiplier {m} outside [{_MIN_MULT}, {_MAX_MULT}]"
                )
        overlay[team] = TeamAdjustment(
            attack_mult=attack, defense_mult=defense,
            reason=str(adj.get("reason", "")), source=str(adj.get("source", "")),
        )
    return overlay


def fixture_multipliers(
    home: str, away: str, overlay: dict[str, TeamAdjustment]
) -> tuple[float, float]:
    """Return (lam_mult, mu_mult) goal multipliers for a fixture.

    Home goals scale with home attack and away defense; away goals scale with
    away attack and home defense. Unknown teams contribute 1.0.
    """
    h = overlay.get(home)
    a = overlay.get(away)
    h_att, h_def = (h.attack_mult, h.defense_mult) if h else (1.0, 1.0)
    a_att, a_def = (a.attack_mult, a.defense_mult) if a else (1.0, 1.0)
    return h_att * a_def, a_att * h_def
