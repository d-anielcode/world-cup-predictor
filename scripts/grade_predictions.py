"""Grade a locked predictions file against actual results.

Usage:
    python -m touchline.cli ingest           # refresh results first
    python scripts/grade_predictions.py docs/predictions/2026-06-19.json

Reads the pre-game predictions, looks up each game's actual score from the DB,
grades every BUY pick (win/loss/push), and scores the model's 1X2 calls
(Brier + log-loss vs the realized outcomes).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Allow running from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touchline import config
from touchline.storage.db import Database


def _result(hg: int, ag: int) -> str:
    return "home" if hg > ag else "away" if ag > hg else "draw"


def _grade_pick(pick: dict, hg: int, ag: int) -> str | None:
    """Return 'WIN' | 'LOSS' | 'PUSH' for a pick given the final score."""
    mt, side, line = pick["market_type"], pick["side"], pick["line"]
    total, margin = hg + ag, hg - ag
    if mt == "1x2":
        return "WIN" if _result(hg, ag) == side else "LOSS"
    if mt == "total":
        over = total > line
        return "WIN" if (over == (side == "over")) else "LOSS"
    if mt == "btts":
        yes = hg >= 1 and ag >= 1
        return "WIN" if (yes == (side == "yes")) else "LOSS"
    if mt == "handicap":  # line is the home handicap; away side is -line for away
        covered = margin > -line if side == "home" else margin < -line
        if margin == -line:
            return "PUSH"
        return "WIN" if covered else "LOSS"
    return None


def grade(path: Path) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    matchday = data["matchday"]
    # Key on (date, home, away) — matching on the team pair alone would grab a
    # historical fixture with the same matchup (e.g. Brazil 7-1 Haiti, 2016 Copa).
    matches = {(m.match_date.isoformat(), m.home_team, m.away_team): m
               for m in Database(config.DB_PATH).all_matches()
               if m.played and m.home_goals is not None}

    wins = losses = pushes = 0
    probs: list[tuple[float, float, float]] = []
    outcomes: list[int] = []
    print(f"=== Grading {matchday} ===\n")
    for g in data["games"]:
        m = matches.get((matchday, g["home"], g["away"]))
        if m is None or m.home_goals is None:
            print(f"{g['home']} v {g['away']}: result not available yet — skipping")
            continue
        hg, ag = m.home_goals, m.away_goals
        print(f"{g['home']} {hg}-{ag} {g['away']}  (model: "
              f"{g['model_1x2']['home']:.0%}/{g['model_1x2']['draw']:.0%}/{g['model_1x2']['away']:.0%})")
        probs.append((g['model_1x2']['home'], g['model_1x2']['draw'], g['model_1x2']['away']))
        outcomes.append({"home": 0, "draw": 1, "away": 2}[_result(hg, ag)])
        # Grade only the BUY recommendations (sorted best-edge first).
        buys = sorted((m for m in g["markets"] if m["recommendation"] == "BUY"),
                      key=lambda m: -m["edge"])
        for pk in buys:
            res = _grade_pick(pk, hg, ag)
            ln = "" if pk["line"] is None else f" {pk['line']}"
            mark = {"WIN": "[WIN] ", "LOSS": "[LOSS]", "PUSH": "[PUSH]"}.get(res, "[????]")
            print(f"    {mark} {pk['market_type']}:{pk['side']}{ln}  "
                  f"(model {pk['model_prob']:.0%}, edge {pk['edge']*100:+.1f}%, conf {pk['confidence']})  -> {res}")
            wins += res == "WIN"; losses += res == "LOSS"; pushes += res == "PUSH"
        print()

    graded = wins + losses
    if graded:
        print(f"BUY record: {wins}-{losses}" + (f"-{pushes} push" if pushes else "")
              + f"  ({wins/graded:.0%} hit rate)")
    if probs:
        brier = sum(sum((p[k] - (1.0 if o == k else 0.0)) ** 2 for k in range(3))
                    for p, o in zip(probs, outcomes)) / len(probs)
        ll = sum(-math.log(max(1e-9, p[o])) for p, o in zip(probs, outcomes)) / len(probs)
        print(f"Model 1X2: Brier={brier:.3f} (uniform 0.667), log_loss={ll:.3f} (uniform 1.099)"
              f"  over {len(probs)} games")


if __name__ == "__main__":
    grade(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/predictions/2026-06-19.json"))
