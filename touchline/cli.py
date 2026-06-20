from __future__ import annotations

import argparse
import json as _json
import re
from collections import Counter
from datetime import date, datetime, timezone

from touchline import config
import httpx

from touchline.data import openfootball, worldcupjson, intl_results
from touchline.data import elo_scrape
from touchline.data.kalshi_read import KalshiReadClient
from touchline.data import kalshi_quotes
from touchline.data import kalshi_history
from touchline.data.venues import is_host_country
from touchline.data.elo import EloTable, load_elo
from touchline.edge.context import build_context
from touchline.edge.edge import compute_edge
from touchline.edge.model_lookup import model_prob
from touchline.edge.quotes import load_quotes, fixture_lines, fixture_score_lines, MarketQuoteRow
from touchline.edge.rank import rank_picks, RankedPick
from touchline.edge.staking import size_stakes
from touchline.model.fit import fit_ratings
from touchline.model.price_fixture import price_fixture
from touchline.model.factors import FactorContext
from touchline.model.ratings import Ratings
from touchline.backtest.market import score_vs_market
from touchline.models import Match, dedupe_matches
from touchline.overlay.squad import load_overlay, fixture_multipliers
from touchline.report.render import render_markdown, render_json
from touchline.backtest.harness import backtest as run_backtest
from touchline.backtest.calibrate import calibrate
from touchline.storage.db import Database

_YEAR_RE = re.compile(r"(\d{4})")


def run_ingest(db: Database, historical: list[Match], live: list[Match]) -> int:
    """Pure orchestration: de-duplicate then upsert provided records.

    Returns the number of distinct fixtures upserted. De-duplication collapses
    the same match arriving from multiple feeds (with team spellings or home/away
    orientation differing) into one row — see `dedupe_matches`."""
    all_matches = dedupe_matches(list(historical) + list(live))
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
    bankroll: float | None = None,
) -> tuple[list[RankedPick], str, str]:
    """Price fixtures, compute edges vs quotes, rank, size stakes, and render.

    `fixtures` is a list of (home, away, date, venue_name). `team_games` maps team ->
    played-match count (Elo-prior reliance proxy). `bankroll` (defaults to
    config.BANKROLL) sizes the recommended stakes. Returns (picks, markdown, json)."""
    edges: list[tuple] = []
    for home, away, when, venue in fixtures:
        fx_quotes = [q for q in quotes if q.home == home and q.away == away]
        if not fx_quotes:
            continue
        totals, handicaps = fixture_lines(quotes, home, away)
        scores = fixture_score_lines(quotes, home, away)
        ctx = build_context(home, away, when, venue, history)
        lam_mult, mu_mult = fixture_multipliers(home, away, overlay)
        # Apply the rating's learned home advantage only when the nominal home team
        # is the venue's host nation. A host listed as the *away* team still gets its
        # host edge via the host factor (ctx.away_is_host), but not this home_adv term
        # (price_fixture's home_adv only lifts the home side).
        apply_home_adv = ctx.home_is_host
        probs = price_fixture(
            ratings, home, away, apply_home_adv=apply_home_adv, ctx=ctx,
            total_lines=totals or None, handicap_lines=handicaps or None,
            score_lines=scores or None,
            lam_mult=lam_mult, mu_mult=mu_mult,
        )
        min_games = min(team_games.get(home, 0), team_games.get(away, 0))
        for q in fx_quotes:
            try:
                mp = model_prob(probs, q.market_type, q.side, q.line)
            except (KeyError, ValueError):
                continue
            e = compute_edge(mp, q.price, min_games, market_type=q.market_type)
            edges.append((home, away, q.market_type, q.side, q.line, e))
    picks = rank_picks(edges, top_n=top_n)
    bank = config.BANKROLL if bankroll is None else bankroll
    stakes = size_stakes(picks, bank)
    return picks, render_markdown(picks, as_of, stakes), render_json(picks, as_of, stakes)


def run_daily(ratings, overlay, quotes, fixtures, history, team_games, as_of, out_dir):
    """Price upcoming fixtures vs quotes and write a dated report.

    Thin wrapper over run_price; returns (markdown_path, json_path)."""
    from pathlib import Path
    _, md, js = run_price(ratings, overlay, quotes, fixtures, history, team_games, as_of=as_of)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{as_of}-report.md"
    json_path = out_dir / f"{as_of}-report.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(js, encoding="utf-8")
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="touchline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Refresh all data sources into SQLite")
    sub.add_parser("fetch-elo",
                   help="Scrape current eloratings.net ratings into cache/elo.csv")
    fit_p = sub.add_parser("fit-ratings", help="Fit Dixon-Coles ratings from stored matches")
    # Defaults calibrated via `backtest --calibrate` on 2018+ WC matches
    # (best log-loss at half_life=900, prior_weight=0.05; longer memory wins for
    # stable national teams). Re-run calibration as the dataset grows.
    fit_p.add_argument("--half-life-days", type=float, default=900.0)
    fit_p.add_argument("--prior-weight", type=float, default=0.05)
    price_p = sub.add_parser("price", help="Compute edges vs a quotes CSV and write a report")
    price_p.add_argument("--quotes", required=True, help="Path to a market quotes CSV")
    price_p.add_argument("--top", type=int, default=None)
    disc_p = sub.add_parser("kalshi-discover",
                            help="Dump live Kalshi World Cup markets to confirm the schema")
    disc_p.add_argument("--series", default="KXWCGAME",
                        help="Comma-separated Kalshi series tickers to dump "
                             "(default KXWCGAME, the per-match winner series that `daily` prices)")
    daily_p = sub.add_parser("daily", help="Full pipeline: ingest, fit, fetch Kalshi, write report")
    daily_p.add_argument("--skip-refresh", action="store_true",
                         help="Use the existing DB/ratings instead of re-ingesting and re-fitting")
    sub.add_parser("capture-prices",
                   help="Backfill settled Kalshi 1X2 pre-kickoff prices into the history store")
    sub.add_parser("backtest-market",
                   help="Score the model against the captured market prices (vs realized outcomes)")
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
        intl = intl_results.gather(config.CACHE_DIR, since_year=2014)
        live = worldcupjson.fetch_matches(config.CACHE_DIR)
        total = run_ingest(db, historical=historical + intl, live=live)
        print(f"Ingested {total} matches "
              f"({len(historical)} WC, {len(intl)} intl, {len(live)} live).")
        return 0

    if args.command == "fetch-elo":
        path = config.CACHE_DIR / "elo.csv"
        try:
            table = elo_scrape.fetch_elo()
        except (httpx.HTTPError, ValueError) as e:
            # Resilient like the other feeds: never overwrite a good elo.csv with a
            # failed scrape. The fit falls back to the existing file (or 1500 prior).
            print(f"fetch-elo failed ({e!r}); keeping existing {path}")
            return 1
        n = elo_scrape.write_elo_csv(table, path)
        print(f"Wrote {n} team Elo ratings to {path} "
              f"(top: {', '.join(t for t, _ in sorted(table.items(), key=lambda kv: -kv[1])[:5])}).")
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

    if args.command == "kalshi-discover":
        client = KalshiReadClient()
        try:
            all_markets = []
            for series in args.series.split(","):
                series = series.strip()
                markets = client.get_markets(series)
                print(f"Series {series}: {len(markets)} markets")
                all_markets.extend(markets)
        finally:
            client.close()
        out = config.DATA_DIR / "kalshi_discover.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(all_markets, indent=2), encoding="utf-8")
        print(f"\nWrote {len(all_markets)} markets to {out}")
        priced = [m for m in all_markets if m.get("yes_bid_dollars")]
        for m in priced[:12]:
            print(f"  {m.get('yes_sub_title'):16s} yes_bid={m.get('yes_bid_dollars')} "
                  f"yes_ask={m.get('yes_ask_dollars')}  ticker={m.get('ticker')}")
        return 0

    if args.command == "capture-prices":
        db = Database(config.DB_PATH)
        db.init_schema()
        kickoff_lookup = {
            (m.match_date, frozenset((m.home_team, m.away_team))): m.kickoff
            for m in db.all_matches() if m.kickoff
        }
        client = KalshiReadClient()
        try:
            records = kalshi_history.capture_1x2_history(client, kickoff_lookup)
        finally:
            client.close()
        store = config.DATA_DIR / "market_history.jsonl"
        total = kalshi_history.write_jsonl(records, store)
        print(f"Captured {len(records)} settled 1X2 contracts; "
              f"store now holds {total} -> {store}")
        return 0

    if args.command == "backtest-market":
        store = config.DATA_DIR / "market_history.jsonl"
        records = kalshi_history.read_jsonl(store)
        if not records:
            print("No captured prices. Run `touchline capture-prices` first.")
            return 1
        by_game: dict[tuple, dict] = {}
        for r in records:
            by_game.setdefault((r["date"], r["home"], r["away"]), {})[r["side"]] = r
        db = Database(config.DB_PATH)
        db.init_schema()
        elo_path = config.CACHE_DIR / "elo.csv"
        elo = load_elo(elo_path) if elo_path.is_file() else EloTable()
        played = sorted([m for m in db.all_matches()
                         if m.played and m.home_goals is not None],
                        key=lambda m: m.match_date)
        fits: dict = {}
        games = []
        for (d, home, away), sides in by_game.items():
            if not {"home", "draw", "away"} <= set(sides):
                continue
            winner = next((s for s in ("home", "draw", "away")
                           if sides[s].get("result") == "yes"), None)
            if winner is None:
                continue
            gd = date.fromisoformat(d)
            if gd not in fits:
                prior = [p for p in played if p.match_date < gd]
                if not prior:
                    continue
                fits[gd] = fit_ratings(prior, elo, half_life_days=900.0,
                                       prior_weight=0.05, as_of=gd)
            r = fits[gd]
            host = is_host_country(home)
            probs = price_fixture(r, home, away, apply_home_adv=host,
                                  ctx=FactorContext(home_is_host=host))
            games.append({
                "outcome": {"home": 0, "draw": 1, "away": 2}[winner],
                "model": (probs.home, probs.draw, probs.away),
                "market": (sides["home"]["market_price"],
                           sides["draw"]["market_price"],
                           sides["away"]["market_price"]),
            })
        if not games:
            print("No scoreable games (need all 3 sides, a result, and prior matches).")
            return 1
        res = score_vs_market(games)
        verdict = "model beats" if res.model_brier < res.market_brier else "market beats"
        print(f"Model vs Market on {res.n} settled games:")
        print(f"  Brier:        model {res.model_brier:.4f}  vs  market "
              f"{res.market_brier:.4f}   ({verdict})")
        print(f"  log_loss:     model {res.model_log_loss:.4f}  vs  market "
              f"{res.market_log_loss:.4f}")
        print(f"  top-pick acc: model {res.model_top_acc:.0%}  vs  market "
              f"{res.market_top_acc:.0%}")
        if res.n_bets:
            roi = res.model_pnl / res.n_bets * 100
            print(f"  value bets:   {res.n_bets} placed, realized P&L "
                  f"${res.model_pnl:+.2f}/$1-flat  ({roi:+.1f}% ROI)")
        else:
            print("  value bets:   none (model never priced above the market)")
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

    if args.command == "daily":
        db = Database(config.DB_PATH)
        db.init_schema()
        if not args.skip_refresh:
            historical = _gather_historical()
            intl = intl_results.gather(config.CACHE_DIR, since_year=2014)
            live = worldcupjson.fetch_matches(config.CACHE_DIR)
            run_ingest(db, historical=historical + intl, live=live)
            elo_path = config.CACHE_DIR / "elo.csv"
            elo = load_elo(elo_path) if elo_path.is_file() else EloTable()
            db.save_ratings(fit_ratings(db.all_matches(), elo, half_life_days=900,
                                        prior_weight=0.05, as_of=date.today()))
        ratings = db.load_ratings()
        overlay = load_overlay(config.CACHE_DIR / "squad_adjustments.json")
        history = db.all_matches()
        team_games = Counter(t for m in history if m.played for t in (m.home_team, m.away_team))
        # Upcoming fixtures with a known venue, excluding games that have already
        # kicked off: Kalshi quotes live in-play odds once a match starts, and the
        # pre-match model isn't calibrated for in-play state. Kickoff times come from
        # openfootball (UTC); a fixture with no known kickoff falls back to included.
        now = datetime.now(timezone.utc)
        fixtures = [(m.home_team, m.away_team, m.match_date, m.venue)
                    for m in history if not m.played and m.venue
                    and (m.kickoff is None or m.kickoff > now)]
        quotes = kalshi_quotes.fetch_quotes()
        md_path, _ = run_daily(ratings, overlay, quotes, fixtures, history,
                               dict(team_games), as_of=date.today().isoformat(),
                               out_dir=config.DATA_DIR / "reports")
        print(f"Priced {len(quotes)} Kalshi markets across "
              f"{len({(q.home, q.away) for q in quotes})} matches -> {md_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
