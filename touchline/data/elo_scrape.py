from __future__ import annotations

import csv
import io
from pathlib import Path

import httpx

from touchline.data.teams import canonical_team

# eloratings.net is a JS front-end over flat TSV data files. The current World
# ranking is World.tsv (its currentPage() defaults to 'World'); en.teams.tsv maps
# each 2-letter team code to its English name(s).
BASE_URL = "https://www.eloratings.net"
WORLD_FILE = "World.tsv"
TEAMS_FILE = "en.teams.tsv"

_USER_AGENT = "touchline/0.1 (+worldcup-edge research)"


def parse_elo(world_tsv: str, teams_tsv: str) -> dict[str, float]:
    """Join eloratings World.tsv with en.teams.tsv into {canonical_team: elo}.

    World.tsv columns: rank, id, CODE, ELO, ... (cols 2 and 3 are used).
    en.teams.tsv columns: CODE, full_name, [aliases...] (cols 0 and 1 are used).
    Names are canonicalized so they align with the match data's spellings. Rows
    whose code is unknown or whose Elo is non-numeric are skipped. If two codes
    canonicalize to the same name, the higher rating wins.
    """
    names: dict[str, str] = {}
    for row in csv.reader(io.StringIO(teams_tsv), delimiter="\t"):
        if len(row) >= 2 and row[0]:
            names[row[0]] = row[1]

    table: dict[str, float] = {}
    for row in csv.reader(io.StringIO(world_tsv), delimiter="\t"):
        if len(row) < 4:
            continue
        code, elo_s = row[2], row[3]
        if code not in names or not elo_s.lstrip("-").isdigit():
            continue
        name = canonical_team(names[code])
        elo = float(elo_s)
        if name not in table or elo > table[name]:
            table[name] = elo
    return table


def fetch_elo(client: httpx.Client | None = None) -> dict[str, float]:
    """Fetch and parse the current eloratings.net World ratings.

    Raises httpx.HTTPError on a network/HTTP failure so callers can keep any
    existing elo.csv rather than overwriting it with a partial scrape."""
    owns = client is None
    client = client or httpx.Client(timeout=30.0, headers={"User-Agent": _USER_AGENT})
    try:
        world = client.get(f"{BASE_URL}/{WORLD_FILE}")
        world.raise_for_status()
        teams = client.get(f"{BASE_URL}/{TEAMS_FILE}")
        teams.raise_for_status()
        return parse_elo(world.text, teams.text)
    finally:
        if owns:
            client.close()


def write_elo_csv(table: dict[str, float], path: Path) -> int:
    """Write {team: elo} as the `team,elo` CSV that load_elo expects, ranked by
    Elo descending. Returns the number of teams written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(table.items(), key=lambda kv: kv[1], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "elo"])
        for team, elo in rows:
            w.writerow([team, int(elo)])
    return len(rows)
