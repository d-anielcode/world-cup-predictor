from __future__ import annotations

import csv
import subprocess
from datetime import date
from pathlib import Path

from touchline.data.teams import canonical_team
from touchline.models import Match

# Comprehensive international results (friendlies, qualifiers, Nations League,
# tournaments) 1872-present. Provides the giants-vs-minnows form that the
# World-Cup-finals-only data lacks, de-compressing team ratings.
INTL_RESULTS_REPO = "https://github.com/martj42/international_results.git"


def refresh_repo(dest: Path) -> Path:
    """Clone or pull the martj42/international_results repo into dest."""
    dest = Path(dest)
    if (dest / ".git").is_dir():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", INTL_RESULTS_REPO, str(dest)], check=True)
    return dest


def parse_results_csv(text: str, since_year: int = 2014) -> list[Match]:
    """Parse results.csv into Match records, keeping matches on/after since_year.

    Rows with 'NA'/empty scores are unplayed future fixtures (played=False).
    Team names are canonicalized so they align with the openfootball/worldcup data.
    """
    out: list[Match] = []
    for row in csv.DictReader(text.splitlines()):
        try:
            d = date.fromisoformat(row["date"])
        except (ValueError, KeyError):
            continue
        if d.year < since_year:
            continue
        hs, as_ = row.get("home_score", ""), row.get("away_score", "")
        played = hs not in ("", "NA") and as_ not in ("", "NA")
        out.append(
            Match(
                match_date=d,
                home_team=canonical_team(row["home_team"]),
                away_team=canonical_team(row["away_team"]),
                home_goals=int(hs) if played else None,
                away_goals=int(as_) if played else None,
                competition=row.get("tournament") or "International",
                stage=None,
                venue=row.get("city") or None,
                played=played,
                source="intl_results",
            )
        )
    return out


def gather(cache_dir: Path, since_year: int = 2014) -> list[Match]:
    """Refresh the repo and parse results.csv. Returns Match records since since_year."""
    repo = refresh_repo(Path(cache_dir) / "intl_results")
    text = (repo / "results.csv").read_text(encoding="utf-8")
    return parse_results_csv(text, since_year=since_year)
