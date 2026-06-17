from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


@dataclass
class EloTable:
    by_norm: dict[str, float] = field(default_factory=dict)

    def get(self, team: str, default: float = 1500.0) -> float:
        return self.by_norm.get(_norm(team), default)


def load_elo(path: Path) -> EloTable:
    table = EloTable()
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            table.by_norm[_norm(row["team"])] = float(row["elo"])
    return table
