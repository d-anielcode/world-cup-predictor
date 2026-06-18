from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Ratings:
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0

    def expected_goals(
        self, home: str, away: str, apply_home_adv: bool
    ) -> tuple[float, float]:
        """Return (lambda_home, mu_away) expected goals for a fixture.

        Unknown teams default to 0.0 attack/defense (league-average).
        """
        ah, dh = self.attack.get(home, 0.0), self.defense.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defense.get(away, 0.0)
        log_lam = ah - da + (self.home_adv if apply_home_adv else 0.0)
        log_mu = aa - dh
        return math.exp(log_lam), math.exp(log_mu)
