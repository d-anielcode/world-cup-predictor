from __future__ import annotations

import re
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dateutil import parser as dateparser

from touchline.data.teams import canonical_team
from touchline.models import Match

# "▪ Group A" / "▪ Round of 16". Real headers are never commented with '#'.
_STAGE_RE = re.compile(r"^\s*▪\s+(?P<stage>(?:Group|Round|Quarter|Semi|Final|Third)[^|]*?)\s*$")
# Standalone date line, e.g. "Sun Nov 20" / "Thu Jun 12" / "Thu June 11" (full or
# abbreviated month; weekday month day, no year).
_DATE_RE = re.compile(r"^[A-Z][a-z]{2}\s+[A-Z][a-z]{2,}\s+\d{1,2}\s*$")
# Leading inline date on a match line, e.g. "Fri Jun 11 16:00  ..." (pre-2014).
_INLINE_DATE_RE = re.compile(
    r"^[A-Z][a-z]{2}\s+(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\b"
)
# Leading kickoff time + optional timezone offset, e.g. "17:00 UTC-3 " / "19:00 ".
_TIME_RE = re.compile(
    r"^\s*(?P<h>\d{1,2}):(?P<min>\d{2})(?:\s+UTC(?P<off>[+-]?\d+))?\s+")


def _parse_kickoff(rest: str, match_date: date) -> datetime | None:
    """Build a UTC kickoff datetime from a leading 'HH:MM UTC±N' on the line.

    Returns None when there is no time or no UTC offset (older feeds omit the
    offset, so an absolute instant can't be resolved)."""
    m = _TIME_RE.match(rest)
    if not m or m.group("off") is None:
        return None
    local = datetime(
        match_date.year, match_date.month, match_date.day,
        int(m.group("h")), int(m.group("min")),
        tzinfo=timezone(timedelta(hours=int(m.group("off")))),
    )
    return local.astimezone(timezone.utc)
_SCORE_RE = re.compile(r"(?P<hg>\d+)-(?P<ag>\d+)")
# Junk between the score and the away team in "Home Score [junk] Away" lines:
# halftime "(0-2)", extra time "a.e.t.", penalties ", 3-2 pen.", stray scores.
_RESULT_JUNK_RE = re.compile(
    r"^(?:\s*(?:a\.e\.t\.?|pen\.?|\([^)]*\)|,|\d+-\d+))+\s*", re.IGNORECASE
)
_YEAR_RE = re.compile(r"(\d{4})")

OPENFOOTBALL_REPO = "https://github.com/openfootball/worldcup.git"


def _competition_year(competition: str) -> int:
    m = _YEAR_RE.search(competition)
    return int(m.group(1)) if m else date.today().year


def _split_teams(body: str) -> tuple[str, str, int, int] | None:
    """Extract (home, away, home_goals, away_goals) from a match body.

    Handles both openfootball orderings:
      - "Home v Away  S-S ..."      (2014/2018)
      - "Home  S-S [junk]  Away"    (2002-2012, 2022)
    The first 'N-N' token is the result; extra-time/penalty suffixes are stripped.
    """
    score = _SCORE_RE.search(body)
    if not score:
        return None
    hg, ag = int(score.group("hg")), int(score.group("ag"))

    if " v " in body[: score.start()]:
        # "Home v Away  score" — teams precede the score.
        home, rest = body[: score.start()].split(" v ", 1)
        away = rest
    else:
        # "Home  score [junk]  Away" — away follows the score (and any et/pen junk).
        home = body[: score.start()]
        away = _RESULT_JUNK_RE.sub("", body[score.end():])

    home, away = home.strip(), away.strip()
    if not home or not away:
        return None
    return home, away, hg, ag


def parse_cup_txt(text: str, competition: str) -> list[Match]:
    """Parse an openfootball cup.txt / cup_finals.txt body into played Match records.

    Unplayed fixtures (no 'N-N' score token) are skipped — they are picked up
    live from worldcupjson during the tournament. The pre-2002 bare-line format
    (no per-match date, only matchday ranges) is intentionally not parsed; those
    matches are decades old and irrelevant under the model's recency weighting.
    """
    year = _competition_year(competition)
    matches: list[Match] = []
    stage: str | None = None
    current_date: date | None = None

    for raw_line in text.splitlines():
        stage_m = _STAGE_RE.match(raw_line)
        if stage_m:
            stage = stage_m.group("stage").strip()
            continue
        line = raw_line.split("#", 1)[0].rstrip()  # drop trailing "# seeding" comments
        if not line.strip():
            continue
        if _DATE_RE.match(line.strip()):
            current_date = dateparser.parse(f"{line.strip()} {year}").date()
            continue

        match_date = current_date
        rest = line
        inline = _INLINE_DATE_RE.match(line.strip())
        if inline:
            match_date = dateparser.parse(
                f"{inline.group('mon')} {inline.group('day')} {year}"
            ).date()
            rest = line.strip()[inline.end():]
        kickoff = _parse_kickoff(rest, match_date) if match_date else None
        rest = _TIME_RE.sub("", rest)

        if match_date is None:
            continue
        venue = None
        if " @ " in rest:
            rest, venue = rest.split(" @ ", 1)
            venue = venue.strip() or None
        parsed = _split_teams(rest)
        if parsed is not None:
            home, away, hg, ag = parsed
            played, hg, ag = True, hg, ag
        elif " v " in rest:
            # Upcoming fixture: "Home v Away" with no score (live/future schedule).
            home, away = (s.strip() for s in rest.split(" v ", 1))
            if not home or not away:
                continue
            played, hg, ag = False, None, None
        else:
            continue
        matches.append(
            Match(
                match_date=match_date,
                home_team=canonical_team(home),
                away_team=canonical_team(away),
                home_goals=hg,
                away_goals=ag,
                competition=competition,
                stage=stage,
                venue=venue,
                played=played,
                source="openfootball",
                kickoff=kickoff,
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
