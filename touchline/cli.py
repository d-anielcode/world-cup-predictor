from __future__ import annotations

import argparse
import re
from datetime import date

from touchline import config
from touchline.data import openfootball, worldcupjson
from touchline.data.elo import EloTable, load_elo
from touchline.model.fit import fit_ratings
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
    fit_p = sub.add_parser("fit-ratings", help="Fit Dixon-Coles ratings from stored matches")
    fit_p.add_argument("--half-life-days", type=float, default=540.0)
    fit_p.add_argument("--prior-weight", type=float, default=0.05)
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

    if args.command == "fit-ratings":
        db = Database(config.DB_PATH)
        db.init_schema()
        matches = db.all_matches()
        if not matches:
            print("No matches in the database. Run `touchline ingest` first.")
            return 1
        elo_path = config.CACHE_DIR / "elo.csv"
        elo = load_elo(elo_path) if elo_path.is_file() else EloTable()
        ratings = fit_ratings(
            matches, elo,
            half_life_days=args.half_life_days,
            prior_weight=args.prior_weight,
            as_of=date.today(),
        )
        db.save_ratings(ratings)
        print(f"Fit ratings for {len(ratings.attack)} teams "
              f"(home_adv={ratings.home_adv:.3f}, rho={ratings.rho:.3f}).")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
