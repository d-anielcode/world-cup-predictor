from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import date

from touchline import config
from touchline.data import openfootball, worldcupjson
from touchline.data.elo import EloTable, load_elo
from touchline.edge.context import build_context
from touchline.edge.edge import compute_edge
from touchline.edge.model_lookup import model_prob
from touchline.edge.quotes import load_quotes, fixture_lines, MarketQuoteRow
from touchline.edge.rank import rank_picks, RankedPick
from touchline.model.fit import fit_ratings
from touchline.model.price_fixture import price_fixture
from touchline.model.ratings import Ratings
from touchline.models import Match
from touchline.overlay.squad import load_overlay, fixture_multipliers
from touchline.report.render import render_markdown, render_json
from touchline.backtest.harness import backtest as run_backtest
from touchline.backtest.calibrate import calibrate
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


def run_price(
    ratings: Ratings,
    overlay: dict,
    quotes: list[MarketQuoteRow],
    fixtures: list[tuple],
    history: list[Match],
    team_games: dict[str, int],
    as_of: str,
    top_n: int | None = None,
) -> tuple[list[RankedPick], str, str]:
    """Price fixtures, compute edges vs quotes, rank, and render the report.

    `fixtures` is a list of (home, away, date, venue_name). `team_games` maps team ->
    played-match count (Elo-prior reliance proxy). Returns (picks, markdown, json)."""
    edges: list[tuple] = []
    for home, away, when, venue in fixtures:
        fx_quotes = [q for q in quotes if q.home == home and q.away == away]
        if not fx_quotes:
            continue
        totals, handicaps = fixture_lines(quotes, home, away)
        ctx = build_context(home, away, when, venue, history)
        lam_mult, mu_mult = fixture_multipliers(home, away, overlay)
        # Apply the learned home advantage only when the nominal home team is the
        # venue's host nation. ASSUMPTION: a host listed as the *away* team would
        # miss its home edge here (price_fixture only boosts the home side). No
        # current 2026 fixture triggers this; Plan 4 must handle host-as-away when
        # the real venue-assigned schedule lands.
        apply_home_adv = ctx.home_altitude_acclimatized
        probs = price_fixture(
            ratings, home, away, apply_home_adv=apply_home_adv, ctx=ctx,
            total_lines=totals or None, handicap_lines=handicaps or None,
            lam_mult=lam_mult, mu_mult=mu_mult,
        )
        min_games = min(team_games.get(home, 0), team_games.get(away, 0))
        for q in fx_quotes:
            try:
                mp = model_prob(probs, q.market_type, q.side, q.line)
            except (KeyError, ValueError):
                continue
            e = compute_edge(mp, q.price, min_games)
            edges.append((home, away, q.market_type, q.side, q.line, e))
    picks = rank_picks(edges, top_n=top_n)
    return picks, render_markdown(picks, as_of), render_json(picks, as_of)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="touchline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Refresh all data sources into SQLite")
    fit_p = sub.add_parser("fit-ratings", help="Fit Dixon-Coles ratings from stored matches")
    # Defaults calibrated via `backtest --calibrate` on 2018+ WC matches
    # (best log-loss at half_life=900, prior_weight=0.05; longer memory wins for
    # stable national teams). Re-run calibration as the dataset grows.
    fit_p.add_argument("--half-life-days", type=float, default=900.0)
    fit_p.add_argument("--prior-weight", type=float, default=0.05)
    price_p = sub.add_parser("price", help="Compute edges vs a quotes CSV and write a report")
    price_p.add_argument("--quotes", required=True, help="Path to a market quotes CSV")
    price_p.add_argument("--top", type=int, default=None)
    bt_p = sub.add_parser("backtest", help="Score the model on completed matches")
    bt_p.add_argument("--eval-start", default="2018-01-01",
                      help="Only score matches on/after this ISO date")
    bt_p.add_argument("--half-life-days", type=float, default=900.0)
    bt_p.add_argument("--prior-weight", type=float, default=0.05)
    bt_p.add_argument("--calibrate", action="store_true",
                      help="Grid-search half-life x prior-weight")
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

    if args.command == "price":
        db = Database(config.DB_PATH)
        db.init_schema()
        ratings = db.load_ratings()
        overlay_path = config.CACHE_DIR / "squad_adjustments.json"
        overlay = load_overlay(overlay_path)
        quotes = load_quotes(args.quotes)
        history = db.all_matches()
        team_games = Counter(t for m in history if m.played
                             for t in (m.home_team, m.away_team))
        fixtures = [(m.home_team, m.away_team, m.match_date, m.venue)
                    for m in history if not m.played and m.venue]
        picks, md, js = run_price(
            ratings, overlay, quotes, fixtures, history, dict(team_games),
            as_of=date.today().isoformat(), top_n=args.top,
        )
        out_dir = config.DATA_DIR
        (out_dir / "report.md").write_text(md, encoding="utf-8")
        (out_dir / "report.json").write_text(js, encoding="utf-8")
        print(f"Wrote {len(picks)} ranked picks to {out_dir / 'report.md'}.")
        return 0

    if args.command == "backtest":
        db = Database(config.DB_PATH)
        db.init_schema()
        matches = db.all_matches()
        elo_path = config.CACHE_DIR / "elo.csv"
        elo = load_elo(elo_path) if elo_path.is_file() else EloTable()
        eval_start = date.fromisoformat(args.eval_start)
        if args.calibrate:
            res = calibrate(matches, eval_start=eval_start, elo=elo,
                            half_lifes=[180.0, 360.0, 540.0, 900.0],
                            prior_weights=[0.01, 0.05, 0.2, 0.5])
            print("Calibration grid (half_life, prior_weight, log_loss):")
            for hl, pw, ll in sorted(res.grid, key=lambda r: r[2]):
                print(f"  hl={hl:>6} pw={pw:<5} log_loss={ll:.4f}")
            print(f"Best: half_life={res.best_half_life}, "
                  f"prior_weight={res.best_prior_weight}, log_loss={res.best_log_loss:.4f}")
        else:
            res = run_backtest(matches, eval_start=eval_start,
                               half_life_days=args.half_life_days,
                               prior_weight=args.prior_weight, elo=elo)
            print(f"Backtest on {res.n_matches} matches "
                  f"(>= {args.eval_start}): Brier={res.brier:.4f}, "
                  f"log_loss={res.log_loss:.4f} "
                  f"(uniform baseline: Brier=0.6667, log_loss=1.0986).")
            print("  Market calibration (model_avg vs actual; Brier vs base):")
            for name, ms in res.markets.items():
                verdict = "beats" if ms.brier < ms.base_brier else "below"
                print(f"    {name:10s} acc={ms.accuracy:.0%} "
                      f"calib_gap={ms.calibration_gap:+.3f} "
                      f"Brier={ms.brier:.4f} (base {ms.base_brier:.4f}) {verdict}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
