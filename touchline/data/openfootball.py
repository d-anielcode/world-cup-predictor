from __future__ import annotations

import re
import subprocess
from datetime import date
from pathlib import Path

from dateutil import parser as dateparser

from touchline.models import Match

# "▪ Group A"  /  "▪ Round of 16" etc.
_STAGE_RE = re.compile(r"^▪\s+(.*\S)\s*$")
# "Sun Nov 20"  (weekday month day, no year)
_DATE_RE = re.compile(r"^[A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}\s*$")
# "  19:00   Qatar  0-2 (0-2)  Ecuador  @ Al Bayt Stadium, Al Khor"
# (2014+ format: kickoff time only; the date is on a separate preceding line)
_MATCH_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}\s+"            # kickoff time
    r"(?P<home>.+?)\s+"               # home team (non-greedy)
    r"(?P<hg>\d+)-(?P<ag>\d+)"        # full-time score
    r"(?:\s+\([\d\-]+\))?\s+"          # optional (halftime)
    r"(?P<away>.+?)"                  # away team
    r"(?:\s+@\s+(?P<venue>.+?))?\s*$"  # optional @ venue
)
# "Fri Jun 11 16:00   South Africa  1-1  Mexico  @ Soccer City, Johannesburg" (2010)
# "Fri Jun 9    Germany  4-2 (2-1)  Costa Rica  @ ..."  (2002-2006: no kickoff time)
# (pre-2014 format: weekday + date [+ optional time] inline on the match line)
_MATCH_INLINE_RE = re.compile(
    r"^\s*[A-Z][a-z]{2}\s+"           # weekday
    r"(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+"  # month + day
    r"(?:\d{1,2}:\d{2}\s+)?"          # optional kickoff time
    r"(?P<home>.+?)\s+"
    r"(?P<hg>\d+)-(?P<ag>\d+)"
    r"(?:\s+\([\d\-]+\))?\s+"
    r"(?P<away>.+?)"
    r"(?:\s+@\s+(?P<venue>.+?))?\s*$"
)
_YEAR_RE = re.compile(r"(\d{4})")

OPENFOOTBALL_REPO = "https://github.com/openfootball/worldcup.git"


def _competition_year(competition: str) -> int:
    m = _YEAR_RE.search(competition)
    return int(m.group(1)) if m else date.today().year


def parse_cup_txt(text: str, competition: str) -> list[Match]:
    """Parse an openfootball cup.txt body into played Match records.

    Unplayed fixtures (no 'N-N' score token) are skipped — they are picked up
    live from worldcupjson during the tournament.
    """
    year = _competition_year(competition)
    matches: list[Match] = []
    stage: str | None = None
    current_date: date | None = None

    for line in text.splitlines():
        stage_m = _STAGE_RE.match(line)
        if stage_m:
            stage = stage_m.group(1)
            continue
        if _DATE_RE.match(line.strip()):
            current_date = dateparser.parse(
                f"{line.strip()} {year}", dayfirst=False
            ).date()
            continue
        inline = _MATCH_INLINE_RE.match(line)
        if inline:
            matches.append(
                Match(
                    match_date=dateparser.parse(
                        f"{inline.group('mon')} {inline.group('day')} {year}",
                        dayfirst=False,
                    ).date(),
                    home_team=inline.group("home").strip(),
                    away_team=inline.group("away").strip(),
                    home_goals=int(inline.group("hg")),
                    away_goals=int(inline.group("ag")),
                    competition=competition,
                    stage=stage,
                    venue=(inline.group("venue") or "").strip() or None,
                    played=True,
                    source="openfootball",
                )
            )
            continue
        mm = _MATCH_RE.match(line)
        if mm and current_date is not None:
            matches.append(
                Match(
                    match_date=current_date,
                    home_team=mm.group("home").strip(),
                    away_team=mm.group("away").strip(),
                    home_goals=int(mm.group("hg")),
                    away_goals=int(mm.group("ag")),
                    competition=competition,
                    stage=stage,
                    venue=(mm.group("venue") or "").strip() or None,
                    played=True,
                    source="openfootball",
                )
            )
    return matches


def refresh_repo(dest: Path) -> Path:
    """Clone or pull the openfootball/worldcup repo into dest. Returns the path."""
    dest = Path(dest)
    if (dest / ".git").is_dir():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", OPENFOOTBALL_REPO, str(dest)], check=True)
    return dest


def find_cup_files(repo: Path) -> list[Path]:
    """Return every tournament cup.txt / cup_finals.txt under the repo."""
    repo = Path(repo)
    return sorted(
        p for p in repo.rglob("*.txt")
        if p.name in {"cup.txt", "cup_finals.txt"}
    )
