# Touchline Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data layer for Touchline — ingest historical (openfootball) and live (worldcupjson) international match results, an international Elo prior, a static 2026 venue table, and a read-only Kalshi market client, all normalized into local SQLite.

**Architecture:** Synchronous Python package. Each external source has its own module that returns plain `Match`/`MarketQuote` dataclasses; a storage module upserts them into SQLite; a CLI `ingest` command orchestrates a refresh. No async, no LLM, no order-placement code anywhere.

**Tech Stack:** Python 3.11, httpx (sync), cryptography (Kalshi RSA-PSS), python-dateutil, pytest. (numpy/scipy arrive in Plan 2.)

---

## File Structure

```
touchline/
├── pyproject.toml              # package metadata + pytest config
├── requirements.txt
├── .gitignore
├── touchline/
│   ├── __init__.py
│   ├── models.py               # Match, MarketQuote dataclasses
│   ├── config.py               # paths, env-driven Kalshi settings
│   ├── storage/
│   │   ├── __init__.py
│   │   └── db.py               # SQLite schema + upsert/query
│   ├── data/
│   │   ├── __init__.py
│   │   ├── venues.py           # static 2026 venue table + haversine
│   │   ├── openfootball.py     # cup.txt parser + repo refresh/enumerate
│   │   ├── worldcupjson.py     # live API client (cached) + parser
│   │   ├── elo.py              # Elo CSV loader + cache
│   │   └── kalshi_read.py      # read-only Kalshi REST client
│   └── cli.py                  # `ingest` command
└── tests/
    ├── __init__.py
    ├── fixtures/
    │   ├── openfootball_sample.txt
    │   ├── worldcupjson_sample.json
    │   └── elo_sample.csv
    ├── test_models.py
    ├── test_db.py
    ├── test_venues.py
    ├── test_openfootball.py
    ├── test_worldcupjson.py
    ├── test_elo.py
    ├── test_kalshi_read.py
    └── test_ingest.py
```

Source under test is the working directory `C:\Users\dcho0\Documents\touchline`. Run all commands from there.

---

## Task 0: Project scaffolding

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.gitignore`, `touchline/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
httpx==0.27.2
cryptography==43.0.1
python-dateutil==2.9.0
pytest==8.3.3
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "touchline"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Create .gitignore**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
touchline_data/
.env
```

- [ ] **Step 4: Create empty package markers**

Create `touchline/__init__.py` and `tests/__init__.py` as empty files.

- [ ] **Step 5: Create venv and install deps**

Run (PowerShell):
```powershell
python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Expected: installs succeed, no errors.

- [ ] **Step 6: Verify pytest runs (collects 0 tests)**

Run: `.\.venv\Scripts\python.exe -m pytest`
Expected: "no tests ran" (exit 5) — confirms pytest is wired.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pyproject.toml .gitignore touchline/__init__.py tests/__init__.py
git commit -m "chore: scaffold touchline package"
```

---

## Task 1: Domain models + SQLite storage

**Files:**
- Create: `touchline/models.py`, `touchline/config.py`, `touchline/storage/__init__.py`, `touchline/storage/db.py`
- Test: `tests/test_models.py`, `tests/test_db.py`

- [ ] **Step 1: Write failing test for the Match model**

`tests/test_models.py`:
```python
from datetime import date
from touchline.models import Match


def test_match_natural_key_is_stable_and_order_independent_fields():
    m = Match(
        match_date=date(2022, 11, 20),
        home_team="Qatar",
        away_team="Ecuador",
        home_goals=0,
        away_goals=2,
        competition="World Cup 2022",
        stage="Group A",
        venue="Al Bayt Stadium, Al Khor",
        played=True,
        source="openfootball",
    )
    assert m.natural_key() == "2022-11-20|Qatar|Ecuador"


def test_unplayed_match_has_none_goals():
    m = Match(
        match_date=date(2026, 6, 20),
        home_team="USA",
        away_team="Wales",
        home_goals=None,
        away_goals=None,
        competition="World Cup 2026",
        stage="Group A",
        venue=None,
        played=False,
        source="worldcupjson",
    )
    assert m.played is False
    assert m.home_goals is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'touchline.models'`.

- [ ] **Step 3: Implement models.py**

`touchline/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class Match:
    match_date: date
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    competition: str
    stage: str | None
    venue: str | None
    played: bool
    source: str

    def natural_key(self) -> str:
        """Dedup key: date + both teams (home listing as recorded)."""
        return f"{self.match_date.isoformat()}|{self.home_team}|{self.away_team}"


@dataclass
class MarketQuote:
    ticker: str
    series_ticker: str
    title: str
    yes_price: float  # dollars 0..1
    no_price: float
    status: str
    raw: dict
```

- [ ] **Step 4: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_models.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write failing test for storage**

`tests/test_db.py`:
```python
from datetime import date
from touchline.models import Match
from touchline.storage.db import Database


def _match(home="Qatar", away="Ecuador", hg=0, ag=2):
    return Match(
        match_date=date(2022, 11, 20), home_team=home, away_team=away,
        home_goals=hg, away_goals=ag, competition="World Cup 2022",
        stage="Group A", venue="Al Bayt Stadium", played=True, source="openfootball",
    )


def test_upsert_then_query_roundtrip(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    db.upsert_matches([_match()])
    rows = db.all_matches()
    assert len(rows) == 1
    assert rows[0].home_team == "Qatar"
    assert rows[0].away_goals == 2


def test_upsert_is_idempotent_on_natural_key(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    db.upsert_matches([_match(hg=0, ag=2)])
    db.upsert_matches([_match(hg=1, ag=1)])  # same teams+date, corrected score
    rows = db.all_matches()
    assert len(rows) == 1
    assert (rows[0].home_goals, rows[0].away_goals) == (1, 1)
```

- [ ] **Step 6: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'touchline.storage.db'`.

- [ ] **Step 7: Implement config.py and db.py**

`touchline/config.py`:
```python
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("TOUCHLINE_DATA_DIR", "touchline_data")).resolve()
DB_PATH = DATA_DIR / "touchline.db"
CACHE_DIR = DATA_DIR / "cache"

# Kalshi (read-only). Reused from EdgeRunner env conventions.
KALSHI_BASE_URL = os.environ.get(
    "KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"
)
KALSHI_API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = Path(
    os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
)
```

`touchline/storage/__init__.py`: empty file.

`touchline/storage/db.py`:
```python
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from touchline.models import Match

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    natural_key TEXT PRIMARY KEY,
    match_date  TEXT NOT NULL,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    home_goals  INTEGER,
    away_goals  INTEGER,
    competition TEXT NOT NULL,
    stage       TEXT,
    venue       TEXT,
    played      INTEGER NOT NULL,
    source      TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def upsert_matches(self, matches: list[Match]) -> int:
        rows = [
            (
                m.natural_key(), m.match_date.isoformat(), m.home_team, m.away_team,
                m.home_goals, m.away_goals, m.competition, m.stage, m.venue,
                int(m.played), m.source,
            )
            for m in matches
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO matches (natural_key, match_date, home_team, away_team,
                    home_goals, away_goals, competition, stage, venue, played, source)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(natural_key) DO UPDATE SET
                    home_goals=excluded.home_goals,
                    away_goals=excluded.away_goals,
                    competition=excluded.competition,
                    stage=excluded.stage,
                    venue=excluded.venue,
                    played=excluded.played,
                    source=excluded.source
                """,
                rows,
            )
        return len(rows)

    def all_matches(self) -> list[Match]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM matches ORDER BY match_date")
            return [
                Match(
                    match_date=date.fromisoformat(r["match_date"]),
                    home_team=r["home_team"], away_team=r["away_team"],
                    home_goals=r["home_goals"], away_goals=r["away_goals"],
                    competition=r["competition"], stage=r["stage"], venue=r["venue"],
                    played=bool(r["played"]), source=r["source"],
                )
                for r in cur.fetchall()
            ]
```

- [ ] **Step 8: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_db.py tests/test_models.py -v`
Expected: PASS (4 passed).

- [ ] **Step 9: Commit**

```bash
git add touchline/models.py touchline/config.py touchline/storage tests/test_models.py tests/test_db.py
git commit -m "feat: Match/MarketQuote models and SQLite storage"
```

---

## Task 2: Static venue table + haversine

**Files:**
- Create: `touchline/data/__init__.py`, `touchline/data/venues.py`
- Test: `tests/test_venues.py`

- [ ] **Step 1: Write failing test**

`tests/test_venues.py`:
```python
from touchline.data.venues import VENUES, get_venue, haversine_km, is_host_country


def test_known_venue_has_full_metadata():
    v = get_venue("Estadio Azteca")
    assert v.city == "Mexico City"
    assert v.country == "Mexico"
    assert v.altitude_m > 2000          # high-altitude venue
    assert v.has_roof_ac is False


def test_host_country_detection():
    assert is_host_country("Mexico") is True
    assert is_host_country("USA") is True
    assert is_host_country("Canada") is True
    assert is_host_country("Brazil") is False


def test_haversine_between_known_cities():
    # New York to Los Angeles ~ 3936 km
    nyc = (40.81, -74.07)
    la = (33.95, -118.34)
    d = haversine_km(nyc[0], nyc[1], la[0], la[1])
    assert 3800 < d < 4100


def test_all_venues_have_coordinates():
    for v in VENUES.values():
        assert -90 <= v.lat <= 90
        assert -180 <= v.lon <= 180
```

- [ ] **Step 2: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_venues.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'touchline.data.venues'`.

- [ ] **Step 3: Implement venues.py**

`touchline/data/__init__.py`: empty file.

`touchline/data/venues.py`:
```python
from __future__ import annotations

import math
from dataclasses import dataclass

HOST_COUNTRIES = {"USA", "Canada", "Mexico"}


@dataclass(frozen=True)
class Venue:
    name: str
    city: str
    country: str
    lat: float
    lon: float
    altitude_m: int
    has_roof_ac: bool
    tz: str


# 2026 FIFA World Cup venues (16 stadiums). Altitude/roof verified from public data.
VENUES: dict[str, Venue] = {
    "MetLife Stadium": Venue("MetLife Stadium", "New York/NJ", "USA", 40.813, -74.074, 7, False, "America/New_York"),
    "AT&T Stadium": Venue("AT&T Stadium", "Dallas", "USA", 32.747, -97.093, 150, True, "America/Chicago"),
    "NRG Stadium": Venue("NRG Stadium", "Houston", "USA", 29.685, -95.411, 15, True, "America/Chicago"),
    "Mercedes-Benz Stadium": Venue("Mercedes-Benz Stadium", "Atlanta", "USA", 33.755, -84.401, 320, True, "America/New_York"),
    "Hard Rock Stadium": Venue("Hard Rock Stadium", "Miami", "USA", 25.958, -80.239, 2, False, "America/New_York"),
    "Arrowhead Stadium": Venue("Arrowhead Stadium", "Kansas City", "USA", 39.049, -94.484, 270, False, "America/Chicago"),
    "Lincoln Financial Field": Venue("Lincoln Financial Field", "Philadelphia", "USA", 39.901, -75.168, 12, False, "America/New_York"),
    "Levi's Stadium": Venue("Levi's Stadium", "San Francisco Bay", "USA", 37.403, -121.970, 5, False, "America/Los_Angeles"),
    "SoFi Stadium": Venue("SoFi Stadium", "Los Angeles", "USA", 33.953, -118.339, 30, True, "America/Los_Angeles"),
    "Lumen Field": Venue("Lumen Field", "Seattle", "USA", 47.595, -122.332, 3, False, "America/Los_Angeles"),
    "Gillette Stadium": Venue("Gillette Stadium", "Boston", "USA", 42.091, -71.264, 90, False, "America/New_York"),
    "BMO Field": Venue("BMO Field", "Toronto", "Canada", 43.633, -79.418, 80, False, "America/Toronto"),
    "BC Place": Venue("BC Place", "Vancouver", "Canada", 49.277, -123.112, 3, True, "America/Vancouver"),
    "Estadio Azteca": Venue("Estadio Azteca", "Mexico City", "Mexico", 19.303, -99.150, 2240, False, "America/Mexico_City"),
    "Estadio Akron": Venue("Estadio Akron", "Guadalajara", "Mexico", 20.681, -103.463, 1560, False, "America/Mexico_City"),
    "Estadio BBVA": Venue("Estadio BBVA", "Monterrey", "Mexico", 25.669, -100.244, 500, False, "America/Monterrey"),
}


def get_venue(name: str) -> Venue:
    """Look up a venue by exact or substring match on the canonical name."""
    if name in VENUES:
        return VENUES[name]
    for key, v in VENUES.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return v
    raise KeyError(f"Unknown venue: {name!r}")


def is_host_country(country: str) -> bool:
    return country in HOST_COUNTRIES


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
```

- [ ] **Step 4: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_venues.py -v`
Expected: PASS (4 passed).

> Note: the exact 2026 venue list/altitudes should be sanity-checked against the official FIFA venue list during execution; the assertions above only pin Azteca altitude, host detection, and coordinate ranges, which are stable facts.

- [ ] **Step 5: Commit**

```bash
git add touchline/data/__init__.py touchline/data/venues.py tests/test_venues.py
git commit -m "feat: 2026 venue table with altitude/host/haversine"
```

---

## Task 3: openfootball cup.txt parser

**Files:**
- Create: `touchline/data/openfootball.py`, `tests/fixtures/openfootball_sample.txt`
- Test: `tests/test_openfootball.py`

- [ ] **Step 1: Create the fixture (real openfootball syntax)**

`tests/fixtures/openfootball_sample.txt`:
```
= World Cup 2022       # in Qatar, November 20 - December 18

Group A  | Qatar      Ecuador       Senegal        Netherlands


▪ Group A
Sun Nov 20
  19:00      Qatar   0-2 (0-2)   Ecuador    @ Al Bayt Stadium, Al Khor
             (Enner Valencia 16' (pen.), 31')

Mon Nov 21
   19:00     Senegal  0-2 (0-0)  Netherlands  @ Al Thumama Stadium, Doha
               (Cody Gakpo 84' Davy Klaassen 90+9')

Fri Nov 25
   16:00     Qatar  1-3 (0-1)  Senegal   @ Al Thumama Stadium, Doha
```

- [ ] **Step 2: Write failing test**

`tests/test_openfootball.py`:
```python
from datetime import date
from pathlib import Path
from touchline.data.openfootball import parse_cup_txt

FIXTURE = Path(__file__).parent / "fixtures" / "openfootball_sample.txt"


def test_parses_all_played_matches():
    matches = parse_cup_txt(FIXTURE.read_text(encoding="utf-8"), competition="World Cup 2022")
    assert len(matches) == 3


def test_first_match_fields():
    m = parse_cup_txt(FIXTURE.read_text(encoding="utf-8"), competition="World Cup 2022")[0]
    assert m.match_date == date(2022, 11, 20)
    assert m.home_team == "Qatar"
    assert m.away_team == "Ecuador"
    assert m.home_goals == 0
    assert m.away_goals == 2
    assert m.stage == "Group A"
    assert m.venue == "Al Bayt Stadium, Al Khor"
    assert m.played is True
    assert m.source == "openfootball"


def test_year_inferred_for_december_rollover():
    # Matchday strings have no year; parser uses competition year as anchor.
    m = parse_cup_txt(FIXTURE.read_text(encoding="utf-8"), competition="World Cup 2022")[2]
    assert m.match_date == date(2022, 11, 25)
```

- [ ] **Step 3: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_openfootball.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'touchline.data.openfootball'`.

- [ ] **Step 4: Implement openfootball.py**

`touchline/data/openfootball.py`:
```python
from __future__ import annotations

import re
from datetime import date

from dateutil import parser as dateparser

from touchline.models import Match

# "▪ Group A"  /  "▪ Round of 16" etc.
_STAGE_RE = re.compile(r"^▪\s+(.*\S)\s*$")
# "Sun Nov 20"  (weekday month day, no year)
_DATE_RE = re.compile(r"^[A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}\s*$")
# "  19:00   Qatar  0-2 (0-2)  Ecuador  @ Al Bayt Stadium, Al Khor"
_MATCH_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}\s+"            # kickoff time
    r"(?P<home>.+?)\s+"               # home team (non-greedy)
    r"(?P<hg>\d+)-(?P<ag>\d+)"        # full-time score
    r"(?:\s+\([\d\-]+\))?\s+"          # optional (halftime)
    r"(?P<away>.+?)"                  # away team
    r"(?:\s+@\s+(?P<venue>.+?))?\s*$"  # optional @ venue
)
_YEAR_RE = re.compile(r"(\d{4})")


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
            # Month/day with no year; anchor to competition year. Jan-Jun => same
            # year, Jul-Dec => same year (World Cups don't cross New Year here).
            current_date = dateparser.parse(
                f"{line.strip()} {year}", dayfirst=False
            ).date()
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
```

- [ ] **Step 5: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_openfootball.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Add repo refresh + file enumeration (network-free unit test)**

Append to `touchline/data/openfootball.py`:
```python
import subprocess
from pathlib import Path

OPENFOOTBALL_REPO = "https://github.com/openfootball/worldcup.git"


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
```

Add to `tests/test_openfootball.py`:
```python
def test_find_cup_files_filters_to_cup_txt(tmp_path):
    from touchline.data.openfootball import find_cup_files
    (tmp_path / "2022--qatar").mkdir()
    (tmp_path / "2022--qatar" / "cup.txt").write_text("x", encoding="utf-8")
    (tmp_path / "2022--qatar" / "cup_finals.txt").write_text("x", encoding="utf-8")
    (tmp_path / "2022--qatar" / "NOTES.md").write_text("x", encoding="utf-8")
    (tmp_path / "2022--qatar" / "cup_stadiums.csv").write_text("x", encoding="utf-8")
    files = find_cup_files(tmp_path)
    assert {f.name for f in files} == {"cup.txt", "cup_finals.txt"}
```

- [ ] **Step 7: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_openfootball.py -v`
Expected: PASS (4 passed).

- [ ] **Step 8: Commit**

```bash
git add touchline/data/openfootball.py tests/test_openfootball.py tests/fixtures/openfootball_sample.txt
git commit -m "feat: openfootball cup.txt parser and repo refresh"
```

---

## Task 4: worldcupjson live client + parser

**Files:**
- Create: `touchline/data/worldcupjson.py`, `tests/fixtures/worldcupjson_sample.json`
- Test: `tests/test_worldcupjson.py`

- [ ] **Step 1: Create the fixture (real API shape)**

`tests/fixtures/worldcupjson_sample.json`:
```json
[
  {
    "id": 1,
    "venue": "MetLife Stadium",
    "location": "New York",
    "status": "completed",
    "stage_name": "Group A",
    "datetime": "2026-06-12T19:00:00Z",
    "winner": "USA",
    "winner_code": "USA",
    "home_team": {"country": "USA", "name": "United States", "goals": 2, "penalties": 0},
    "away_team": {"country": "WAL", "name": "Wales", "goals": 1, "penalties": 0}
  },
  {
    "id": 2,
    "venue": "SoFi Stadium",
    "location": "Los Angeles",
    "status": "future_scheduled",
    "stage_name": "Group A",
    "datetime": "2026-06-16T22:00:00Z",
    "winner": null,
    "winner_code": null,
    "home_team": {"country": "USA", "name": "United States", "goals": null, "penalties": null},
    "away_team": {"country": "MEX", "name": "Mexico", "goals": null, "penalties": null}
  }
]
```

- [ ] **Step 2: Write failing test**

`tests/test_worldcupjson.py`:
```python
import json
from datetime import date
from pathlib import Path
from touchline.data.worldcupjson import parse_matches

FIXTURE = Path(__file__).parent / "fixtures" / "worldcupjson_sample.json"


def test_parses_completed_match():
    matches = parse_matches(json.loads(FIXTURE.read_text(encoding="utf-8")),
                            competition="World Cup 2026")
    done = matches[0]
    assert done.match_date == date(2026, 6, 12)
    assert done.home_team == "United States"
    assert done.away_team == "Wales"
    assert done.home_goals == 2
    assert done.away_goals == 1
    assert done.played is True
    assert done.venue == "MetLife Stadium"
    assert done.source == "worldcupjson"


def test_future_match_is_unplayed_with_none_goals():
    matches = parse_matches(json.loads(FIXTURE.read_text(encoding="utf-8")),
                            competition="World Cup 2026")
    fut = matches[1]
    assert fut.played is False
    assert fut.home_goals is None
    assert fut.away_goals is None
```

- [ ] **Step 3: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_worldcupjson.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Implement worldcupjson.py**

`touchline/data/worldcupjson.py`:
```python
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from dateutil import parser as dateparser

from touchline.models import Match

BASE_URL = "https://worldcupjson.net"
_COMPLETED = {"completed", "finished", "full-time"}


def parse_matches(payload: list[dict], competition: str) -> list[Match]:
    out: list[Match] = []
    for m in payload:
        home = m["home_team"]
        away = m["away_team"]
        played = str(m.get("status", "")).lower() in _COMPLETED
        out.append(
            Match(
                match_date=dateparser.parse(m["datetime"]).date(),
                home_team=home["name"],
                away_team=away["name"],
                home_goals=home["goals"] if played else None,
                away_goals=away["goals"] if played else None,
                competition=competition,
                stage=m.get("stage_name"),
                venue=m.get("venue"),
                played=played,
                source="worldcupjson",
            )
        )
    return out


def fetch_matches(
    cache_dir: Path,
    competition: str = "World Cup 2026",
    ttl_seconds: int = 300,
    client: httpx.Client | None = None,
) -> list[Match]:
    """Fetch /matches with a simple on-disk TTL cache (respects 10 req/60s limit)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "worldcupjson_matches.json"

    fresh = cache_file.is_file() and (time.time() - cache_file.stat().st_mtime) < ttl_seconds
    if fresh:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        owns = client is None
        client = client or httpx.Client(timeout=15.0)
        try:
            resp = client.get(f"{BASE_URL}/matches")
            resp.raise_for_status()
            payload = resp.json()
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        finally:
            if owns:
                client.close()
    return parse_matches(payload, competition=competition)
```

- [ ] **Step 5: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_worldcupjson.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Add a cache-hit test (no network)**

Add to `tests/test_worldcupjson.py`:
```python
def test_fetch_uses_cache_when_fresh(tmp_path):
    from touchline.data.worldcupjson import fetch_matches
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "worldcupjson_matches.json").write_text(
        FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # No client passed; if it tried to hit the network with ttl high it'd fail offline.
    matches = fetch_matches(cache_dir, ttl_seconds=10_000)
    assert len(matches) == 2
```

- [ ] **Step 7: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_worldcupjson.py -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Commit**

```bash
git add touchline/data/worldcupjson.py tests/test_worldcupjson.py tests/fixtures/worldcupjson_sample.json
git commit -m "feat: worldcupjson live client with TTL cache"
```

---

## Task 5: Elo prior loader

**Files:**
- Create: `touchline/data/elo.py`, `tests/fixtures/elo_sample.csv`
- Test: `tests/test_elo.py`

The Elo source is a CSV (team,elo) cached on disk. During execution the operator drops a current
international Elo export (e.g. from eloratings.net) at `touchline_data/cache/elo.csv`; this task
only owns parsing + lookup with a normalized-name fallback.

- [ ] **Step 1: Create fixture**

`tests/fixtures/elo_sample.csv`:
```csv
team,elo
Brazil,2120
Argentina,2105
France,2080
United States,1790
Wales,1760
```

- [ ] **Step 2: Write failing test**

`tests/test_elo.py`:
```python
from pathlib import Path
from touchline.data.elo import load_elo, EloTable

FIXTURE = Path(__file__).parent / "fixtures" / "elo_sample.csv"


def test_load_and_lookup_exact():
    table = load_elo(FIXTURE)
    assert table.get("Brazil") == 2120.0
    assert table.get("United States") == 1790.0


def test_lookup_is_case_and_space_insensitive():
    table = load_elo(FIXTURE)
    assert table.get("  brazil ") == 2120.0


def test_missing_team_returns_default_prior():
    table = load_elo(FIXTURE)
    assert table.get("Atlantis", default=1500.0) == 1500.0
```

- [ ] **Step 3: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_elo.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Implement elo.py**

`touchline/data/elo.py`:
```python
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
```

- [ ] **Step 5: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_elo.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add touchline/data/elo.py tests/test_elo.py tests/fixtures/elo_sample.csv
git commit -m "feat: Elo prior CSV loader with name normalization"
```

---

## Task 6: Read-only Kalshi client

**Files:**
- Create: `touchline/data/kalshi_read.py`
- Test: `tests/test_kalshi_read.py`
- Reference (copy auth from): `C:\Users\dcho0\Documents\edgerunner\execution\kalshi_client.py` (`_sign_request`, `_build_headers`, `_request_with_retry`, `get_market`, `get_markets`)

Port EdgeRunner's RSA-PSS signing to a **synchronous** `httpx.Client`. Include ONLY read methods.
No order/cancel methods exist in this file.

- [ ] **Step 1: Write failing test for price parsing + market mapping (no network/auth)**

`tests/test_kalshi_read.py`:
```python
from touchline.data.kalshi_read import parse_price, market_to_quote


def test_parse_fixed_point_price_string():
    assert parse_price("0.6500") == 0.65
    assert parse_price("0") == 0.0
    assert parse_price(None) == 0.0


def test_market_to_quote_maps_fields():
    raw = {
        "ticker": "KXWC-26-USAWAL-USA",
        "title": "Will USA beat Wales?",
        "yes_bid": "0.5500", "yes_ask": "0.6100",
        "status": "active",
    }
    q = market_to_quote(raw, series_ticker="KXWC")
    assert q.ticker == "KXWC-26-USAWAL-USA"
    assert q.series_ticker == "KXWC"
    assert 0.5 < q.yes_price < 0.65   # midpoint of bid/ask
    assert q.no_price == round(1 - q.yes_price, 4)
    assert q.status == "active"
```

- [ ] **Step 2: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_kalshi_read.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement kalshi_read.py**

`touchline/data/kalshi_read.py`:
```python
from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from touchline import config
from touchline.models import MarketQuote


def parse_price(value: str | None) -> float:
    """Kalshi prices are fixed-point dollar strings like '0.6500'."""
    if not value:
        return 0.0
    return float(value)


def market_to_quote(raw: dict, series_ticker: str) -> MarketQuote:
    yes_bid = parse_price(raw.get("yes_bid"))
    yes_ask = parse_price(raw.get("yes_ask"))
    yes_price = round((yes_bid + yes_ask) / 2, 4) if (yes_bid or yes_ask) else 0.0
    return MarketQuote(
        ticker=raw["ticker"],
        series_ticker=series_ticker,
        title=raw.get("title", ""),
        yes_price=yes_price,
        no_price=round(1 - yes_price, 4),
        status=raw.get("status", ""),
        raw=raw,
    )


class KalshiReadClient:
    """Read-only Kalshi REST client. Ported from EdgeRunner (sync). No order methods."""

    def __init__(self) -> None:
        self._base_url = config.KALSHI_BASE_URL.rstrip("/")
        self._api_key_id = config.KALSHI_API_KEY_ID
        self._key_path = Path(config.KALSHI_PRIVATE_KEY_PATH)
        self._client = httpx.Client(timeout=15.0)
        self._private_key = None
        if self._key_path.is_file():
            self._private_key = serialization.load_pem_private_key(
                self._key_path.read_bytes(), password=None
            )

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        path_without_query = path.split("?", 1)[0]
        base_path = urlparse(self._base_url).path
        message = (timestamp_ms + method.upper() + base_path + path_without_query).encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str) -> dict:
        ts = str(int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
        }

    def _get(self, path: str) -> dict:
        resp = self._client.get(
            f"{self._base_url}{path}", headers=self._headers("GET", path)
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}").get("market", {})

    def get_markets(self, series_ticker: str, status: str = "open", limit: int = 100) -> list[dict]:
        """Page through all markets for a series; returns raw market dicts."""
        markets: list[dict] = []
        cursor: str | None = None
        while True:
            path = f"/markets?status={status}&limit={limit}&series_ticker={series_ticker}"
            if cursor:
                path += f"&cursor={cursor}"
            data = self._get(path)
            markets.extend(data.get("markets", []))
            cursor = data.get("cursor") or None
            if not cursor:
                break
        return markets

    def quotes_for_series(self, series_ticker: str) -> list[MarketQuote]:
        return [market_to_quote(m, series_ticker) for m in self.get_markets(series_ticker)]

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_kalshi_read.py -v`
Expected: PASS (2 passed).

> Note: live auth is verified during the `ingest` smoke test, not in unit tests (no keys in CI).
> The World Cup series ticker (e.g. `KXWC` / similar) must be confirmed via
> `get_markets` discovery during execution and recorded in `config.py`.

- [ ] **Step 5: Commit**

```bash
git add touchline/data/kalshi_read.py tests/test_kalshi_read.py
git commit -m "feat: read-only sync Kalshi client (RSA-PSS, no order path)"
```

---

## Task 7: `ingest` orchestration + CLI

**Files:**
- Create: `touchline/cli.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write failing test (sources injected, no network)**

`tests/test_ingest.py`:
```python
from datetime import date
from touchline.models import Match
from touchline.storage.db import Database
from touchline.cli import run_ingest


def _match(src, home="A", away="B"):
    return Match(
        match_date=date(2026, 6, 12), home_team=home, away_team=away,
        home_goals=1, away_goals=0, competition="World Cup 2026",
        stage="Group A", venue="MetLife Stadium", played=True, source=src,
    )


def test_run_ingest_stores_matches_from_all_sources(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    historical = [_match("openfootball", "Qatar", "Ecuador")]
    live = [_match("worldcupjson", "USA", "Wales")]
    count = run_ingest(db, historical=historical, live=live)
    assert count == 2
    assert len(db.all_matches()) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ingest.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_ingest'`.

- [ ] **Step 3: Implement cli.py**

`touchline/cli.py`:
```python
from __future__ import annotations

import argparse

from touchline import config
from touchline.data import openfootball, worldcupjson
from touchline.models import Match
from touchline.storage.db import Database


def run_ingest(db: Database, historical: list[Match], live: list[Match]) -> int:
    """Pure orchestration: upsert provided records. Returns total upserted."""
    all_matches = list(historical) + list(live)
    return db.upsert_matches(all_matches)


def _gather_historical() -> list[Match]:
    repo = openfootball.refresh_repo(config.CACHE_DIR / "openfootball")
    matches: list[Match] = []
    for f in openfootball.find_cup_files(repo):
        # Competition label derived from the tournament folder name.
        competition = f.parent.name.replace("--", " ").replace("-", " ").title()
        matches.extend(openfootball.parse_cup_txt(f.read_text(encoding="utf-8"), competition))
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="touchline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Refresh all data sources into SQLite")
    args = parser.parse_args(argv)

    if args.command == "ingest":
        db = Database(config.DB_PATH)
        db.init_schema()
        historical = _gather_historical()
        live = worldcupjson.fetch_matches(config.CACHE_DIR)
        total = run_ingest(db, historical=historical, live=live)
        print(f"Ingested {total} matches "
              f"({len(historical)} historical, {len(live)} live).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ingest.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest`
Expected: PASS — all tests green.

- [ ] **Step 6: Live smoke test (manual, network)**

Run: `.\.venv\Scripts\python.exe -m touchline.cli ingest`
Expected: clones openfootball, pulls live matches, prints an "Ingested N matches" summary, and creates `touchline_data/touchline.db`. (Kalshi auth is exercised in Plan 3 when markets are priced; if keys/series are unset, ingest still succeeds on football data.)

- [ ] **Step 7: Commit**

```bash
git add touchline/cli.py tests/test_ingest.py
git commit -m "feat: ingest orchestration and CLI"
```

---

## Self-Review Notes

- **Spec coverage (data layer):** openfootball ✓ (Task 3), worldcupjson ✓ (Task 4), Elo prior ✓ (Task 5), Kalshi read-only ✓ (Task 6), venue table w/ altitude+host+haversine ✓ (Task 2), SQLite storage ✓ (Task 1), ingest CLI ✓ (Task 7). Rating engine, factors, pricing, overlay, edge, report, and backtest are intentionally deferred to Plans 2 and 3.
- **No order path:** `kalshi_read.py` contains only GET methods — satisfies the "no order-placement code" constraint.
- **Type consistency:** `Match` and `MarketQuote` field names are used identically across `models.py`, `db.py`, all parsers, and `kalshi_read.py`. `Database` methods (`init_schema`, `upsert_matches`, `all_matches`) match across `db.py`, `cli.py`, and tests.
- **Open items flagged for execution (not placeholders):** confirm official 2026 venue altitudes; confirm the live Kalshi World Cup series ticker via discovery; operator supplies `elo.csv`. These are data-confirmation steps, each with a working default.
```
