from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Ratings:
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    intercept: float = 0.0   # log baseline goal rate (fitted; absorbs the overall level)

    def expected_goals(
        self, home: str, away: str, apply_home_adv: bool
    ) -> tuple[float, float]:
        """Return (lambda_home, mu_away) expected goals for a fixture.

        `intercept` is the neutral-site goal level; `home_adv` is applied
        symmetrically (home +half, away -half) so a neutral game keeps the correct
        level instead of collapsing to the away baseline. Unknown teams default to
        0.0 attack/defense (league-average).
        """
        ah, dh = self.attack.get(home, 0.0), self.defense.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defense.get(away, 0.0)
        half = self.home_adv / 2 if apply_home_adv else 0.0
        log_lam = self.intercept + ah - da + half
        log_mu = self.intercept + aa - dh - half
        return math.exp(log_lam), math.exp(log_mu)
