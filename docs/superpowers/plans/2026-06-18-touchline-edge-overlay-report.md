# Touchline Edge, Overlay & Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce Touchline's headline deliverable — compare the model's probabilities (Plan 2) against market prices, flag where the market is wrong, and emit a ranked Markdown + JSON report of the top predictions, with a curated squad-injury overlay layered in.

**Architecture:** A new `touchline/edge/`, `touchline/overlay/`, and `touchline/report/` set of pure modules plus a `price` CLI command. The pipeline: load ratings + squad overlay → for each upcoming fixture build a `FactorContext` (travel/rest/altitude/host from the schedule & venue table) → price the exact market lines the venue offers (`price_fixture`) with overlay multipliers → compare model probabilities to market prices → score edge, EV, and confidence (sample-size + favorite-longshot direction) → rank → render report. Market prices come from a quotes CSV (works offline, fully testable) or, when available, the live read-only Kalshi client via a thin isolated adapter.

**Tech Stack:** Python 3.11 (sync), numpy, pytest. Builds on Plan 1 (`Database`, `Match`, `venues`, `kalshi_read`) and Plan 2 (`Ratings`, `price_fixture`, `MarketProbs`, `FactorContext`).

**Prerequisite:** Plans 1 & 2 are merged to master. This plan assumes `touchline/model/price_fixture.py` accepts `total_lines`/`handicap_lines` (delivered in Plan 2).

**Deferred to Plan 4 (do NOT build here):** the Brier/log-loss backtest & hyperparameter calibration harness, and Open-Meteo heat wiring (`FactorContext.wbgt_c` stays `None` here — `adjust_expected_goals` already treats `None` as a no-op).

---

## Conventions

**Normalized market identity** is a `(market_type, side, line)` triple:
- `market_type="1x2"`, side ∈ {`home`,`draw`,`away`}, line `None`
- `market_type="total"`, side ∈ {`over`,`under`}, line = the goals line (e.g. `2.5`)
- `market_type="btts"`, side ∈ {`yes`,`no`}, line `None`
- `market_type="handicap"`, side ∈ {`home`,`away`}, line = the home handicap (e.g. `-1.5`)

**Market price** is the implied probability of the named side winning, as a dollar value in `[0,1]` (Kalshi YES price for that contract). **Edge** = `model_prob - market_price`. A positive edge means the model thinks the side is underpriced (a value bet on that side).

**Favorite-longshot confidence:** longshots are systematically overpriced. So an edge that *fades* a longshot (model says a low-priced side is even less likely — i.e. positive edge on the *opposite* side) is more trustworthy than an edge that *backs* a longshot. We down-weight confidence when the value bet is itself a longshot (market_price < 0.25).

---

## File Structure

```
touchline/
├── overlay/
│   ├── __init__.py
│   └── squad.py            # load/validate overlay, fixture multipliers
├── edge/
│   ├── __init__.py
│   ├── quotes.py           # MarketQuoteRow + CSV loader; lines-per-fixture helper
│   ├── kalshi_adapter.py   # thin live-Kalshi -> MarketQuoteRow adapter (isolated)
│   ├── context.py          # build FactorContext from schedule + venues
│   ├── model_lookup.py     # MarketProbs + (type,side,line) -> probability
│   ├── edge.py             # Edge dataclass + compute_edge (edge, EV, confidence)
│   └── rank.py             # rank edges, top predictions
├── report/
│   ├── __init__.py
│   └── render.py           # Markdown + JSON writers
└── cli.py                  # add `price` command
overlay/
└── squad_adjustments.example.json
tests/
├── test_overlay_squad.py
├── test_price_fixture_overlay.py
├── test_edge_quotes.py
├── test_edge_context.py
├── test_model_lookup.py
├── test_edge_compute.py
├── test_edge_rank.py
├── test_report_render.py
└── test_price_cli.py
```

Run all commands from `C:\Users\dcho0\Documents\touchline` via `.venv/Scripts/python.exe -m pytest ...`.

---

## Task 1: Squad overlay (load, validate, fixture multipliers)

**Files:**
- Create: `touchline/overlay/__init__.py`, `touchline/overlay/squad.py`, `overlay/squad_adjustments.example.json`
- Test: `tests/test_overlay_squad.py`

- [ ] **Step 1: Write the failing test**

`tests/test_overlay_squad.py`:
```python
import json
import pytest
from touchline.overlay.squad import load_overlay, fixture_multipliers, TeamAdjustment


def _write(tmp_path, data):
    p = tmp_path / "squad.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_parses_team_adjustments(tmp_path):
    p = _write(tmp_path, {"Brazil": {"attack_mult": 0.9, "defense_mult": 1.0,
                                     "reason": "Neymar out", "source": "news"}})
    ov = load_overlay(p)
    assert isinstance(ov["Brazil"], TeamAdjustment)
    assert ov["Brazil"].attack_mult == 0.9
    assert ov["Brazil"].reason == "Neymar out"


def test_missing_path_returns_empty_overlay(tmp_path):
    assert load_overlay(tmp_path / "nope.json") == {}


def test_rejects_out_of_range_multiplier(tmp_path):
    p = _write(tmp_path, {"Brazil": {"attack_mult": 5.0, "defense_mult": 1.0,
                                     "reason": "x", "source": "y"}})
    with pytest.raises(ValueError):
        load_overlay(p)


def test_fixture_multipliers_compose_home_and_away():
    ov = {
        "Brazil": TeamAdjustment(0.9, 1.0, "Neymar out", "news"),
        "Chile": TeamAdjustment(1.0, 0.8, "leaky defense", "news"),
    }
    lam_mult, mu_mult = fixture_multipliers("Brazil", "Chile", ov)
    # home goals scaled by Brazil attack * Chile defense
    assert lam_mult == 0.9 * 0.8
    # away goals scaled by Chile attack * Brazil defense
    assert mu_mult == 1.0 * 1.0


def test_unknown_teams_are_identity():
    assert fixture_multipliers("X", "Y", {}) == (1.0, 1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_overlay_squad.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`touchline/overlay/__init__.py`: empty file.

`touchline/overlay/squad.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_MIN_MULT = 0.5
_MAX_MULT = 1.5


@dataclass
class TeamAdjustment:
    attack_mult: float
    defense_mult: float
    reason: str
    source: str


def load_overlay(path: Path) -> dict[str, TeamAdjustment]:
    """Load squad_adjustments.json. Missing file -> empty overlay.

    Multipliers must be within [0.5, 1.5]; anything outside is a likely typo and
    raises ValueError (a 5x swing would silently dominate the model)."""
    path = Path(path)
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    overlay: dict[str, TeamAdjustment] = {}
    for team, adj in raw.items():
        attack = float(adj.get("attack_mult", 1.0))
        defense = float(adj.get("defense_mult", 1.0))
        for m in (attack, defense):
            if not _MIN_MULT <= m <= _MAX_MULT:
                raise ValueError(
                    f"{team}: multiplier {m} outside [{_MIN_MULT}, {_MAX_MULT}]"
                )
        overlay[team] = TeamAdjustment(
            attack_mult=attack, defense_mult=defense,
            reason=str(adj.get("reason", "")), source=str(adj.get("source", "")),
        )
    return overlay


def fixture_multipliers(
    home: str, away: str, overlay: dict[str, TeamAdjustment]
) -> tuple[float, float]:
    """Return (lam_mult, mu_mult) goal multipliers for a fixture.

    Home goals scale with home attack and away defense; away goals scale with
    away attack and home defense. Unknown teams contribute 1.0.
    """
    h = overlay.get(home)
    a = overlay.get(away)
    h_att, h_def = (h.attack_mult, h.defense_mult) if h else (1.0, 1.0)
    a_att, a_def = (a.attack_mult, a.defense_mult) if a else (1.0, 1.0)
    return h_att * a_def, a_att * h_def
```

`overlay/squad_adjustments.example.json` (must stay loadable — every key is a real team, no comment keys; copy to `touchline_data/cache/squad_adjustments.json` and edit each matchday; multipliers in [0.5, 1.5], >1 strengthens, <1 weakens):
```json
{
  "Brazil": {"attack_mult": 0.90, "defense_mult": 1.00, "reason": "Star striker out (example)", "source": "https://example.com/injury-news"}
}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_overlay_squad.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/overlay tests/test_overlay_squad.py overlay/squad_adjustments.example.json
git commit -m "feat: squad overlay loader, validation, and fixture multipliers"
```

---

## Task 2: Apply overlay multipliers in price_fixture

**Files:**
- Modify: `touchline/model/price_fixture.py`
- Test: `tests/test_price_fixture_overlay.py`

- [ ] **Step 1: Write the failing test**

`tests/test_price_fixture_overlay.py`:
```python
from touchline.model.ratings import Ratings
from touchline.model.factors import FactorContext
from touchline.model.price_fixture import price_fixture


def _ratings():
    return Ratings(attack={"A": 0.4, "B": -0.1}, defense={"A": 0.2, "B": -0.1},
                   home_adv=0.0, rho=-0.05)


def test_lam_mult_below_one_lowers_home_win_prob():
    r = _ratings()
    base = price_fixture(r, "A", "B", apply_home_adv=False, ctx=FactorContext())
    weakened = price_fixture(r, "A", "B", apply_home_adv=False, ctx=FactorContext(),
                             lam_mult=0.7)
    assert weakened.home < base.home


def test_default_mults_are_identity():
    r = _ratings()
    base = price_fixture(r, "A", "B", apply_home_adv=False, ctx=FactorContext())
    same = price_fixture(r, "A", "B", apply_home_adv=False, ctx=FactorContext(),
                         lam_mult=1.0, mu_mult=1.0)
    assert abs(base.home - same.home) < 1e-12
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_price_fixture_overlay.py -v`
Expected: FAIL — `TypeError: price_fixture() got an unexpected keyword argument 'lam_mult'`.

- [ ] **Step 3: Modify price_fixture.py**

Replace the `price_fixture` signature and body in `touchline/model/price_fixture.py` with:
```python
def price_fixture(
    ratings: Ratings,
    home: str,
    away: str,
    apply_home_adv: bool,
    ctx: FactorContext,
    max_goals: int = 10,
    total_lines: list[float] | None = None,
    handicap_lines: list[float] | None = None,
    lam_mult: float = 1.0,
    mu_mult: float = 1.0,
) -> MarketProbs:
    """Full pipeline: ratings -> expected goals -> factor adjustment ->
    overlay multipliers -> Dixon-Coles scoreline matrix -> market probabilities.

    `lam_mult`/`mu_mult` are squad-overlay goal multipliers (home/away). They are
    applied after the environmental factors. `total_lines`/`handicap_lines` override
    the default market lines so callers can price the exact lines a market offers.
    `max_goals=10` is ample for football (truncation mass beyond it is ~1e-10)."""
    lam, mu = ratings.expected_goals(home, away, apply_home_adv=apply_home_adv)
    lam, mu = adjust_expected_goals(lam, mu, ctx)
    lam, mu = lam * lam_mult, mu * mu_mult
    matrix = scoreline_matrix(lam, mu, ratings.rho, max_goals=max_goals)
    return price_matrix(matrix, total_lines=total_lines, handicap_lines=handicap_lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_price_fixture_overlay.py -v`
Expected: PASS (2 passed). Also run `.venv/Scripts/python.exe -m pytest tests/test_price_fixture.py` to confirm no regression.

- [ ] **Step 5: Commit**

```bash
git add touchline/model/price_fixture.py tests/test_price_fixture_overlay.py
git commit -m "feat: apply squad-overlay goal multipliers in price_fixture"
```

---

## Task 3: Quotes (CSV loader + lines-per-fixture)

**Files:**
- Create: `touchline/edge/__init__.py`, `touchline/edge/quotes.py`
- Test: `tests/test_edge_quotes.py`, `tests/fixtures/quotes_sample.csv`

- [ ] **Step 1: Create the fixture**

`tests/fixtures/quotes_sample.csv`:
```csv
home,away,market_type,side,line,price,ticker
USA,Wales,1x2,home,,0.55,KXWC-USAWAL-USA
USA,Wales,1x2,draw,,0.27,KXWC-USAWAL-DRAW
USA,Wales,total,over,2.5,0.48,KXWC-USAWAL-OV25
USA,Wales,btts,yes,,0.50,KXWC-USAWAL-BTTS
USA,Wales,handicap,home,-1.5,0.30,KXWC-USAWAL-H15
```

- [ ] **Step 2: Write the failing test**

`tests/test_edge_quotes.py`:
```python
from pathlib import Path
from touchline.edge.quotes import load_quotes, MarketQuoteRow, fixture_lines

FIXTURE = Path(__file__).parent / "fixtures" / "quotes_sample.csv"


def test_load_quotes_parses_rows():
    rows = load_quotes(FIXTURE)
    assert len(rows) == 5
    assert isinstance(rows[0], MarketQuoteRow)
    assert rows[0].home == "USA" and rows[0].market_type == "1x2"
    assert rows[0].side == "home" and rows[0].line is None
    assert rows[0].price == 0.55


def test_total_and_handicap_lines_are_floats():
    rows = load_quotes(FIXTURE)
    total = next(r for r in rows if r.market_type == "total")
    hcap = next(r for r in rows if r.market_type == "handicap")
    assert total.line == 2.5
    assert hcap.line == -1.5


def test_fixture_lines_collects_distinct_lines_per_fixture():
    rows = load_quotes(FIXTURE)
    totals, handicaps = fixture_lines(rows, "USA", "Wales")
    assert totals == [2.5]
    assert handicaps == [-1.5]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_edge_quotes.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

`touchline/edge/__init__.py`: empty file.

`touchline/edge/quotes.py`:
```python
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MarketQuoteRow:
    home: str
    away: str
    market_type: str          # "1x2" | "total" | "btts" | "handicap"
    side: str                 # see conventions
    line: float | None
    price: float              # market-implied probability of `side`, in [0,1]
    ticker: str = ""


def _parse_line(value: str) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def load_quotes(path: Path) -> list[MarketQuoteRow]:
    rows: list[MarketQuoteRow] = []
    with Path(path).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(MarketQuoteRow(
                home=r["home"].strip(), away=r["away"].strip(),
                market_type=r["market_type"].strip(), side=r["side"].strip(),
                line=_parse_line(r.get("line", "")), price=float(r["price"]),
                ticker=r.get("ticker", "").strip(),
            ))
    return rows


def fixture_lines(
    rows: list[MarketQuoteRow], home: str, away: str
) -> tuple[list[float], list[float]]:
    """Distinct total and handicap lines quoted for a fixture (sorted)."""
    totals, handicaps = set(), set()
    for r in rows:
        if r.home == home and r.away == away and r.line is not None:
            if r.market_type == "total":
                totals.add(r.line)
            elif r.market_type == "handicap":
                handicaps.add(r.line)
    return sorted(totals), sorted(handicaps)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_edge_quotes.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add touchline/edge/__init__.py touchline/edge/quotes.py tests/test_edge_quotes.py tests/fixtures/quotes_sample.csv
git commit -m "feat: market quotes CSV loader and per-fixture line collection"
```

---

## Task 4: Model probability lookup

**Files:**
- Create: `touchline/edge/model_lookup.py`
- Test: `tests/test_model_lookup.py`

- [ ] **Step 1: Write the failing test**

`tests/test_model_lookup.py`:
```python
import pytest
from touchline.model.pricing import MarketProbs
from touchline.edge.model_lookup import model_prob


def _probs():
    return MarketProbs(home=0.5, draw=0.3, away=0.2, btts_yes=0.55,
                       over={2.5: 0.48}, home_handicap={-1.5: 0.3})


def test_1x2_lookup():
    p = _probs()
    assert model_prob(p, "1x2", "home", None) == 0.5
    assert model_prob(p, "1x2", "away", None) == 0.2


def test_total_over_and_under():
    p = _probs()
    assert model_prob(p, "total", "over", 2.5) == 0.48
    assert abs(model_prob(p, "total", "under", 2.5) - 0.52) < 1e-12


def test_btts_yes_no():
    p = _probs()
    assert model_prob(p, "btts", "yes", None) == 0.55
    assert abs(model_prob(p, "btts", "no", None) - 0.45) < 1e-12


def test_handicap_home_and_away():
    p = _probs()
    assert model_prob(p, "handicap", "home", -1.5) == 0.3
    assert abs(model_prob(p, "handicap", "away", -1.5) - 0.7) < 1e-12


def test_missing_line_raises():
    p = _probs()
    with pytest.raises(KeyError):
        model_prob(p, "total", "over", 3.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_lookup.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`touchline/edge/model_lookup.py`:
```python
from __future__ import annotations

from touchline.model.pricing import MarketProbs


def model_prob(
    probs: MarketProbs, market_type: str, side: str, line: float | None
) -> float:
    """Return the model probability for a normalized (market_type, side, line).

    Raises KeyError if a total/handicap line was not priced, or ValueError for an
    unknown market_type/side."""
    if market_type == "1x2":
        return {"home": probs.home, "draw": probs.draw, "away": probs.away}[side]
    if market_type == "total":
        over = probs.over[line]
        return over if side == "over" else 1.0 - over
    if market_type == "btts":
        return probs.btts_yes if side == "yes" else 1.0 - probs.btts_yes
    if market_type == "handicap":
        home = probs.home_handicap[line]
        return home if side == "home" else 1.0 - home
    raise ValueError(f"Unknown market_type/side: {market_type}/{side}")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_lookup.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/edge/model_lookup.py tests/test_model_lookup.py
git commit -m "feat: model probability lookup by normalized market identity"
```

---

## Task 5: Edge computation (edge, EV, confidence)

**Files:**
- Create: `touchline/edge/edge.py`
- Test: `tests/test_edge_compute.py`

- [ ] **Step 1: Write the failing test**

`tests/test_edge_compute.py`:
```python
from touchline.edge.edge import compute_edge, Edge


def test_positive_edge_when_model_above_market():
    e = compute_edge(model_prob=0.60, market_price=0.50, min_games=20)
    assert isinstance(e, Edge)
    assert abs(e.edge - 0.10) < 1e-12
    assert e.recommendation == "BUY"


def test_negative_edge_recommends_pass():
    e = compute_edge(model_prob=0.40, market_price=0.50, min_games=20)
    assert e.edge < 0
    assert e.recommendation == "PASS"


def test_ev_per_dollar_is_edge_over_price():
    e = compute_edge(model_prob=0.60, market_price=0.50, min_games=20)
    assert abs(e.ev_per_dollar - (0.60 - 0.50) / 0.50) < 1e-12


def test_longshot_value_bet_has_lower_confidence():
    # Same edge magnitude, but one value bet sits on a longshot price.
    favorite = compute_edge(model_prob=0.70, market_price=0.60, min_games=50)
    longshot = compute_edge(model_prob=0.20, market_price=0.10, min_games=50)
    assert longshot.confidence < favorite.confidence


def test_thin_sample_lowers_confidence():
    deep = compute_edge(model_prob=0.60, market_price=0.50, min_games=50)
    thin = compute_edge(model_prob=0.60, market_price=0.50, min_games=2)
    assert thin.confidence < deep.confidence
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_edge_compute.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`touchline/edge/edge.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

_LONGSHOT_PRICE = 0.25     # value bets priced below this are down-weighted
_SAMPLE_TARGET = 20.0      # games-per-team at which sample confidence saturates
_EDGE_BUY_THRESHOLD = 0.0  # positive edge => BUY


@dataclass
class Edge:
    model_prob: float
    market_price: float
    edge: float
    ev_per_dollar: float
    confidence: float
    recommendation: str


def _sample_confidence(min_games: int) -> float:
    return min(1.0, max(0, min_games) / _SAMPLE_TARGET)


def _longshot_confidence(market_price: float) -> float:
    """Down-weight value bets that sit on a longshot price (overpriced tail)."""
    if market_price >= _LONGSHOT_PRICE:
        return 1.0
    return max(0.3, market_price / _LONGSHOT_PRICE)


def compute_edge(model_prob: float, market_price: float, min_games: int) -> Edge:
    """Compare a model probability to a market price.

    `min_games` is the smaller of the two teams' played-match counts (proxy for how
    much the rating leans on the Elo prior). Confidence combines sample depth and the
    favorite-longshot direction of the value bet.
    """
    edge = model_prob - market_price
    ev = (model_prob - market_price) / market_price if market_price > 0 else 0.0
    confidence = _sample_confidence(min_games) * _longshot_confidence(market_price)
    recommendation = "BUY" if edge > _EDGE_BUY_THRESHOLD else "PASS"
    return Edge(
        model_prob=model_prob, market_price=market_price, edge=edge,
        ev_per_dollar=ev, confidence=confidence, recommendation=recommendation,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_edge_compute.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/edge/edge.py tests/test_edge_compute.py
git commit -m "feat: edge/EV/confidence computation with favorite-longshot weighting"
```

---

## Task 6: FactorContext from schedule + venues

**Files:**
- Create: `touchline/edge/context.py`
- Test: `tests/test_edge_context.py`

Builds a `FactorContext` for an upcoming fixture from match history and the venue table.
Heat (`wbgt_c`) stays `None` (Plan 4 wires Open-Meteo). Travel uses the haversine from a
team's most-recent prior venue to the fixture venue; rest uses days since that prior match.

- [ ] **Step 1: Write the failing test**

`tests/test_edge_context.py`:
```python
from datetime import date
from touchline.models import Match
from touchline.edge.context import build_context


def _played(home, away, d, venue):
    return Match(match_date=d, home_team=home, away_team=away, home_goals=1,
                 away_goals=0, competition="WC", stage=None, venue=venue,
                 played=True, source="t")


def test_host_and_altitude_from_venue():
    history = []
    ctx = build_context("Mexico", "USA", date(2026, 6, 24),
                        venue_name="Estadio Azteca", history=history)
    assert ctx.altitude_m > 2000
    assert ctx.home_altitude_acclimatized is True   # Mexico hosts at altitude
    assert ctx.away_altitude_acclimatized is False


def test_rest_days_from_prior_match():
    history = [_played("USA", "Wales", date(2026, 6, 20), "MetLife Stadium")]
    ctx = build_context("USA", "Mexico", date(2026, 6, 24),
                        venue_name="SoFi Stadium", history=history)
    assert ctx.rest_days_home == 4


def test_travel_distance_positive_for_cross_country_move():
    history = [_played("USA", "Wales", date(2026, 6, 20), "MetLife Stadium")]  # NJ
    ctx = build_context("USA", "Mexico", date(2026, 6, 24),
                        venue_name="SoFi Stadium", history=history)  # LA
    assert ctx.travel_km_home > 3000
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_edge_context.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`touchline/edge/context.py`:
```python
from __future__ import annotations

from datetime import date

from touchline.data.venues import get_venue, haversine_km, is_host_country
from touchline.model.factors import FactorContext
from touchline.models import Match


def _last_match_before(team: str, when: date, history: list[Match]) -> Match | None:
    prior = [m for m in history
             if (m.home_team == team or m.away_team == team) and m.match_date < when]
    return max(prior, key=lambda m: m.match_date) if prior else None


def _travel_and_rest(team, when, venue, history):
    last = _last_match_before(team, when, history)
    if last is None:
        return 0.0, None
    rest = (when - last.match_date).days
    travel = 0.0
    if last.venue:
        try:
            prev, cur = get_venue(last.venue), venue
            travel = haversine_km(prev.lat, prev.lon, cur.lat, cur.lon)
        except KeyError:
            travel = 0.0
    return travel, rest


def build_context(
    home: str, away: str, when: date, venue_name: str, history: list[Match]
) -> FactorContext:
    """Build a FactorContext for an upcoming fixture. wbgt_c is left None (Plan 4)."""
    venue = get_venue(venue_name)
    travel_h, rest_h = _travel_and_rest(home, when, venue, history)
    travel_a, rest_a = _travel_and_rest(away, when, venue, history)
    return FactorContext(
        travel_km_home=travel_h,
        travel_km_away=travel_a,
        altitude_m=venue.altitude_m,
        home_altitude_acclimatized=is_host_country(home) and venue.country == home,
        away_altitude_acclimatized=is_host_country(away) and venue.country == away,
        wbgt_c=None,
        rest_days_home=rest_h,
        rest_days_away=rest_a,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_edge_context.py -v`
Expected: PASS (3 passed). (Note: `home_altitude_acclimatized` uses host-country as an acclimatization proxy; "Mexico" is both a host country and the country of Estadio Azteca.)

- [ ] **Step 5: Commit**

```bash
git add touchline/edge/context.py tests/test_edge_context.py
git commit -m "feat: build FactorContext from schedule history and venue table"
```

---

## Task 7: Rank edges into top predictions

**Files:**
- Create: `touchline/edge/rank.py`
- Test: `tests/test_edge_rank.py`

- [ ] **Step 1: Write the failing test**

`tests/test_edge_rank.py`:
```python
from touchline.edge.edge import Edge
from touchline.edge.rank import RankedPick, rank_picks


def _edge(edge, conf, rec="BUY"):
    return Edge(model_prob=0.5 + edge, market_price=0.5, edge=edge,
                ev_per_dollar=edge / 0.5, confidence=conf, recommendation=rec)


def test_only_buys_with_positive_edge_are_ranked():
    picks = rank_picks([
        ("USA", "Wales", "1x2", "home", None, _edge(0.10, 1.0)),
        ("USA", "Wales", "1x2", "away", None, _edge(-0.05, 1.0, "PASS")),
    ])
    assert len(picks) == 1
    assert picks[0].side == "home"


def test_ranked_by_edge_times_confidence_descending():
    picks = rank_picks([
        ("A", "B", "1x2", "home", None, _edge(0.20, 0.3)),   # score 0.06
        ("C", "D", "btts", "yes", None, _edge(0.10, 1.0)),    # score 0.10
    ])
    assert [p.market_type for p in picks] == ["btts", "1x2"]
    assert isinstance(picks[0], RankedPick)
    assert picks[0].score > picks[1].score


def test_top_n_limits_results():
    edges = [("A", "B", "1x2", "home", None, _edge(0.10 + i / 100, 1.0)) for i in range(5)]
    picks = rank_picks(edges, top_n=2)
    assert len(picks) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_edge_rank.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`touchline/edge/rank.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from touchline.edge.edge import Edge


@dataclass
class RankedPick:
    home: str
    away: str
    market_type: str
    side: str
    line: float | None
    edge: Edge
    score: float


def rank_picks(
    edges: list[tuple[str, str, str, str, float | None, Edge]],
    top_n: int | None = None,
) -> list[RankedPick]:
    """Keep BUY recommendations with positive edge, score by edge*confidence, sort desc.

    Each input tuple is (home, away, market_type, side, line, Edge).
    """
    picks: list[RankedPick] = []
    for home, away, market_type, side, line, e in edges:
        if e.recommendation != "BUY" or e.edge <= 0:
            continue
        picks.append(RankedPick(
            home=home, away=away, market_type=market_type, side=side, line=line,
            edge=e, score=e.edge * e.confidence,
        ))
    picks.sort(key=lambda p: p.score, reverse=True)
    return picks[:top_n] if top_n is not None else picks
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_edge_rank.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/edge/rank.py tests/test_edge_rank.py
git commit -m "feat: rank edges into confidence-weighted top predictions"
```

---

## Task 8: Report rendering (Markdown + JSON)

**Files:**
- Create: `touchline/report/__init__.py`, `touchline/report/render.py`
- Test: `tests/test_report_render.py`

- [ ] **Step 1: Write the failing test**

`tests/test_report_render.py`:
```python
import json
from touchline.edge.edge import Edge
from touchline.edge.rank import RankedPick
from touchline.report.render import render_markdown, render_json


def _pick(home, away, mt, side, edge_val, conf):
    e = Edge(model_prob=0.6, market_price=0.5, edge=edge_val,
             ev_per_dollar=edge_val / 0.5, confidence=conf, recommendation="BUY")
    return RankedPick(home=home, away=away, market_type=mt, side=side, line=None,
                      edge=e, score=edge_val * conf)


def test_markdown_has_top_predictions_section_and_rows():
    picks = [_pick("USA", "Wales", "1x2", "home", 0.10, 0.9)]
    md = render_markdown(picks, as_of="2026-06-24")
    assert "# Touchline Edge Report" in md
    assert "Top Predictions" in md
    assert "USA" in md and "Wales" in md
    assert "10.0%" in md or "0.10" in md   # edge shown


def test_markdown_handles_empty_picks():
    md = render_markdown([], as_of="2026-06-24")
    assert "No positive-edge" in md


def test_json_roundtrips_pick_fields():
    picks = [_pick("USA", "Wales", "total", "over", 0.08, 0.7)]
    payload = json.loads(render_json(picks, as_of="2026-06-24"))
    assert payload["as_of"] == "2026-06-24"
    row = payload["picks"][0]
    assert row["home"] == "USA"
    assert row["market_type"] == "total"
    assert abs(row["edge"] - 0.08) < 1e-9
    assert abs(row["confidence"] - 0.7) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_report_render.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`touchline/report/__init__.py`: empty file.

`touchline/report/render.py`:
```python
from __future__ import annotations

import json

from touchline.edge.rank import RankedPick


def _label(p: RankedPick) -> str:
    line = "" if p.line is None else f" {p.line}"
    return f"{p.market_type}:{p.side}{line}"


def render_markdown(picks: list[RankedPick], as_of: str) -> str:
    lines = [f"# Touchline Edge Report", f"_As of {as_of}_", ""]
    lines.append("## Top Predictions")
    if not picks:
        lines.append("")
        lines.append("No positive-edge predictions found.")
        return "\n".join(lines)
    lines.append("")
    lines.append("| # | Match | Market | Model | Market | Edge | Conf |")
    lines.append("|---|-------|--------|-------|--------|------|------|")
    for i, p in enumerate(picks, 1):
        lines.append(
            f"| {i} | {p.home} v {p.away} | {_label(p)} | "
            f"{p.edge.model_prob:.0%} | {p.edge.market_price:.0%} | "
            f"{p.edge.edge*100:.1f}% | {p.edge.confidence:.2f} |"
        )
    return "\n".join(lines)


def render_json(picks: list[RankedPick], as_of: str) -> str:
    return json.dumps({
        "as_of": as_of,
        "picks": [
            {
                "home": p.home, "away": p.away, "market_type": p.market_type,
                "side": p.side, "line": p.line,
                "model_prob": p.edge.model_prob, "market_price": p.edge.market_price,
                "edge": p.edge.edge, "ev_per_dollar": p.edge.ev_per_dollar,
                "confidence": p.edge.confidence, "score": p.score,
            }
            for p in picks
        ],
    }, indent=2)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_report_render.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add touchline/report tests/test_report_render.py
git commit -m "feat: Markdown + JSON edge report rendering"
```

---

## Task 9: `price` CLI orchestration

**Files:**
- Create: `touchline/edge/kalshi_adapter.py`
- Modify: `touchline/cli.py`
- Test: `tests/test_price_cli.py`

The `price` command: load ratings + overlay, read a quotes CSV, group quotes by fixture,
price each fixture's exact lines (with factors + overlay), compute edges (using each fixture's
smaller team game-count), rank, and write `report.md` + `report.json`. The live-Kalshi path is
a thin adapter (`kalshi_adapter.py`) that converts read-only Kalshi market dicts to
`MarketQuoteRow`s; it is isolated and not exercised in unit tests (no keys/series in CI), and
the CSV path is the default.

- [ ] **Step 1: Write the failing test (pure orchestration, no network)**

`tests/test_price_cli.py`:
```python
from datetime import date
from pathlib import Path

from touchline.models import Match
from touchline.model.ratings import Ratings
from touchline.edge.quotes import load_quotes
from touchline.cli import run_price


def _ratings():
    return Ratings(attack={"USA": 0.5, "Wales": -0.2},
                   defense={"USA": 0.3, "Wales": -0.2}, home_adv=0.2, rho=-0.05)


def test_run_price_produces_ranked_picks(tmp_path):
    quotes = load_quotes(Path("tests/fixtures/quotes_sample.csv"))
    history = [Match(match_date=date(2026, 6, 20), home_team="USA", away_team="Iran",
                     home_goals=2, away_goals=0, competition="WC", stage=None,
                     venue="MetLife Stadium", played=True, source="t")]
    fixtures = [("USA", "Wales", date(2026, 6, 24), "SoFi Stadium")]
    picks, md, js = run_price(
        ratings=_ratings(), overlay={}, quotes=quotes, fixtures=fixtures,
        history=history, team_games={"USA": 30, "Wales": 25}, as_of="2026-06-24",
    )
    assert isinstance(md, str) and "Touchline Edge Report" in md
    assert '"as_of": "2026-06-24"' in js
    # Strong USA priced into a 0.55 home market should surface as a pick or be evaluated.
    assert all(p.edge.recommendation == "BUY" for p in picks)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_price_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_price'`.

- [ ] **Step 3: Implement kalshi_adapter.py**

`touchline/edge/kalshi_adapter.py`:
```python
from __future__ import annotations

from touchline.data.kalshi_read import KalshiReadClient
from touchline.edge.quotes import MarketQuoteRow


def fetch_quotes(series_ticker: str) -> list[MarketQuoteRow]:
    """Fetch live Kalshi World Cup markets and map them to MarketQuoteRows.

    NOTE: the exact Kalshi WC market ticker/title schema is not yet confirmed. This
    adapter is intentionally thin and isolated; until the live series is verified the
    CSV quotes path is authoritative. Parsing real markets into (home, away,
    market_type, side, line) requires reading the live `title`/`subtitle` fields and
    must be finalized against real data before relying on this path.
    """
    client = KalshiReadClient()
    try:
        markets = client.get_markets(series_ticker)
    finally:
        client.close()
    rows: list[MarketQuoteRow] = []
    for m in markets:
        # Placeholder mapping is deliberately omitted — see NOTE above. Returning the
        # raw markets count via an exception keeps callers from silently trusting an
        # unverified parse.
        raise NotImplementedError(
            f"Kalshi WC market parsing not yet verified ({len(markets)} markets "
            f"fetched). Use the --quotes CSV path until the live schema is confirmed."
        )
    return rows
```

- [ ] **Step 4: Add `run_price` + `price` command to cli.py**

In `touchline/cli.py`, add imports near the top:
```python
from touchline.model.price_fixture import price_fixture
from touchline.model.factors import FactorContext
from touchline.overlay.squad import load_overlay, fixture_multipliers
from touchline.edge.quotes import load_quotes, fixture_lines, MarketQuoteRow
from touchline.edge.context import build_context
from touchline.edge.model_lookup import model_prob
from touchline.edge.edge import compute_edge, Edge
from touchline.edge.rank import rank_picks, RankedPick
from touchline.report.render import render_markdown, render_json
from touchline.model.ratings import Ratings
from collections import Counter
```

Add this pure orchestration function (above `main`):
```python
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
        apply_home_adv = ctx.home_altitude_acclimatized  # host plays at home
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
```

Register the subcommand inside `main` (next to the others):
```python
    price_p = sub.add_parser("price", help="Compute edges vs a quotes CSV and write a report")
    price_p.add_argument("--quotes", required=True, help="Path to a market quotes CSV")
    price_p.add_argument("--top", type=int, default=None)
```

Add the branch inside `main` (after `fit-ratings`, before `return 1`):
```python
    if args.command == "price":
        from datetime import date as _date
        db = Database(config.DB_PATH)
        db.init_schema()
        ratings = db.load_ratings()
        overlay_path = config.CACHE_DIR / "squad_adjustments.json"
        overlay = load_overlay(overlay_path)
        quotes = load_quotes(args.quotes)
        history = db.all_matches()
        team_games = Counter(t for m in history if m.played
                             for t in (m.home_team, m.away_team))
        # Upcoming fixtures = unplayed matches with a known venue.
        fixtures = [(m.home_team, m.away_team, m.match_date, m.venue)
                    for m in history if not m.played and m.venue]
        picks, md, js = run_price(
            ratings, overlay, quotes, fixtures, history, dict(team_games),
            as_of=_date.today().isoformat(), top_n=args.top,
        )
        out_dir = config.DATA_DIR
        (out_dir / "report.md").write_text(md, encoding="utf-8")
        (out_dir / "report.json").write_text(js, encoding="utf-8")
        print(f"Wrote {len(picks)} ranked picks to {out_dir / 'report.md'}.")
        return 0
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_price_cli.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all green.

- [ ] **Step 7: Live smoke test (offline, CSV path)**

```
.venv/Scripts/python.exe -m touchline.cli ingest
.venv/Scripts/python.exe -m touchline.cli fit-ratings
.venv/Scripts/python.exe -m touchline.cli price --quotes tests/fixtures/quotes_sample.csv
```
Expected: prints `Wrote N ranked picks to ...report.md.` and creates `touchline_data/report.md` + `report.json`. (Picks may be 0 if the live DB has no unplayed USA/Wales fixture matching the sample quotes — that is acceptable for the smoke test; the unit test exercises the populated path. Confirm the report files are created and the Markdown has a "Top Predictions" section.)

- [ ] **Step 8: Commit**

```bash
git add touchline/edge/kalshi_adapter.py touchline/cli.py tests/test_price_cli.py
git commit -m "feat: price CLI — edges vs quotes, ranked report (md + json)"
```

---

## Self-Review Notes

- **Spec coverage:** squad overlay ✓ (Task 1–2), edge = model − implied + EV ✓ (Task 5), favorite-longshot confidence weight ✓ (Task 5), sample-size/prior-reliance confidence ✓ (Task 5 via `min_games`), ranked top predictions ✓ (Task 7), Markdown + JSON report ✓ (Task 8), `price` CLI fetching markets + applying factors + overlay ✓ (Task 9), real `FactorContext` from schedule/venue ✓ (Task 6). Corners remain out of scope (no data). The backtest/calibration harness and Open-Meteo heat are explicitly deferred to Plan 4.
- **Placeholder scan:** No TBD/TODO in code steps. The `kalshi_adapter.fetch_quotes` deliberately raises `NotImplementedError` with an explanatory message rather than silently mis-parsing an unverified live schema — this is an intentional, documented safety stop, not a placeholder, and the CSV path is the authoritative, tested one. Flagged clearly for finalization against real Kalshi data.
- **Type consistency:** `MarketQuoteRow(home, away, market_type, side, line, price, ticker)` used identically across `quotes.py`, `kalshi_adapter.py`, `cli.run_price`, tests. `Edge(model_prob, market_price, edge, ev_per_dollar, confidence, recommendation)` consistent across `edge.py`, `rank.py`, `render.py`, tests. `RankedPick(home, away, market_type, side, line, edge, score)` consistent across `rank.py`, `render.py`, `cli.run_price`, tests. `model_prob(probs, market_type, side, line)` and `compute_edge(model_prob, market_price, min_games)` signatures match all call sites. `price_fixture(..., total_lines, handicap_lines, lam_mult, mu_mult)` matches the Task 2 modification and the Plan 2 delivery.
- **Open items flagged for execution (not placeholders):** the live Kalshi WC series ticker and market-title schema must be confirmed against real data before `kalshi_adapter` is completed; until then the CSV quotes path is authoritative. The `home_altitude_acclimatized` host-country acclimatization proxy is a deliberate simplification (a host nation is treated as acclimatized at its own venues). Handicap sign convention follows Plan 2: `home_handicap[line] = P(margin > -line)` — confirm against the Kalshi spread field when wiring the adapter.
```
