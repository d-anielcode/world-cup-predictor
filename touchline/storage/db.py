from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from touchline.model.ratings import Ratings
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
CREATE TABLE IF NOT EXISTS team_ratings (
    team    TEXT PRIMARY KEY,
    attack  REAL NOT NULL,
    defense REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_params (
    key   TEXT PRIMARY KEY,
    value REAL NOT NULL
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

    def save_ratings(self, ratings: Ratings) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM team_ratings")
            conn.execute("DELETE FROM model_params")
            all_teams = ratings.attack.keys() | ratings.defense.keys()
            conn.executemany(
                "INSERT INTO team_ratings (team, attack, defense) VALUES (?,?,?)",
                [(t, ratings.attack.get(t, 0.0), ratings.defense.get(t, 0.0))
                 for t in all_teams],
            )
            conn.executemany(
                "INSERT INTO model_params (key, value) VALUES (?,?)",
                [("home_adv", ratings.home_adv), ("rho", ratings.rho)],
            )

    def load_ratings(self) -> Ratings:
        with self._connect() as conn:
            rows = conn.execute("SELECT team, attack, defense FROM team_ratings").fetchall()
            params = dict(conn.execute("SELECT key, value FROM model_params").fetchall())
        return Ratings(
            attack={r["team"]: r["attack"] for r in rows},
            defense={r["team"]: r["defense"] for r in rows},
            home_adv=float(params.get("home_adv", 0.0)),
            rho=float(params.get("rho", 0.0)),
        )

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
