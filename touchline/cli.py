from __future__ import annotations

import argparse
import re

from touchline import config
from touchline.data import openfootball, worldcupjson
from touchline.models import Match
from touchline.storage.db import Database

_YEAR_RE = re.compile(r"(\d{4})")


def run_ingest(db: Database, historical: list[Match], live: list[Match]) -> int:
    """Pure orchestration: upsert provided records. Returns total upserted."""
    all_matches = list(historical) + list(live)
    return db.upsert_matches(all_matches)


def _gather_historical() -> list[Match]:
    repo = openfootball.refresh_repo(config.CACHE_DIR / "openfootball")
    matches: list[Match] = []
    for f in openfootball.find_cup_files(repo):
        # Uniform "World Cup YYYY" label (matches worldcupjson) from the folder year.
        year_m = _YEAR_RE.search(f.parent.name)
        competition = f"World Cup {year_m.group(1)}" if year_m else f.parent.name
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
