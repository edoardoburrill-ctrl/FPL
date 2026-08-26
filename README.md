# FPL Advisor

A quantitative decision engine for Fantasy Premier League. It projects a
probability distribution of points for every player in every upcoming gameweek
and turns that into concrete weekly decisions: transfers, captain, bench, chips.

Documentation:

- [`docs/PLAN.md`](docs/PLAN.md) — the build plan and the mathematics behind each
  component, with the equations.
- [`docs/DESIGN.md`](docs/DESIGN.md) — the system as built: module contracts,
  invariants, and where the seams are. Read this before changing anything.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — the non-obvious choices and why they
  went the way they did, including the ones that were wrong the first time.
- [`docs/TODO.md`](docs/TODO.md) — the short list of things only you can do, and
  when.

## Status

| Phase | Contents | State |
|---|---|---|
| 0 | Data clients, snapshot store, squad ledger, CLI | done |
| 1 | Projection engine: minutes, team goals, player rates, points | done |
| 2 | Multi-gameweek transfer optimiser | done |
| 3 | Monte Carlo engine, ranked bonus, real backtest | first slice done — see below |
| 4 | Chips, scheduled automation | done (price prediction pending) |

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Configure

```bash
export FPL_ENTRY_ID=1234567        # the number in fantasy.premierleague.com/entry/<id>/event/2
export ODDS_API_KEY=...            # optional, free tier at the-odds-api.com
export FPL_DATA_DIR=./data         # optional, where snapshots and state live
```

Your **entry id** can also live in `fpl.toml` at the repository root, which is
where this checkout keeps it, so no shell export is needed. It is the number in
the URL of your Points page — not your premierleague.com account id (a UUID) —
and FPL reissues entry ids every season, so `fpl.toml` needs updating each
August. The environment variable overrides the file when both are set.

Secrets do not go in `fpl.toml`; `ODDS_API_KEY` is read from the environment.

You do not actually need it. `fpl sync` snapshots the player pool without one,
and `fpl set-squad` records your squad by hand:

```bash
fpl sync                                    # no entry id needed for player data
fpl set-squad "Raya, Sels, Gabriel@6.1, Gvardiol, ..." \
    --bank 0.5 --free-transfers 2 --gw 3
```

Append `@price` to record what you actually **paid** for a player. FPL returns
only half of any rise when you sell, so the selling price depends on the purchase
price — and no endpoint reports that retrospectively, which is why the ledger has
to be seeded once and then maintained.

## Use

```bash
fpl sync --entry 1234567 --deep    # pull data, snapshot prices, refresh squad state
fpl projections --gw 3 --top 25    # expected points table
fpl projections --position DEF --max-cost 5.5
fpl captain --gw 3                 # captaincy ranking with the margin to the next option
fpl squad                          # your squad, selling prices, bank, chips, free transfers
fpl set-squad "..." --bank 0.5     # record your squad by hand, no entry id required
fpl explain haaland                # term-by-term breakdown of a projection
fpl transfers --horizon 5          # multi-gameweek transfer plan
fpl brief                          # the one command: what to do before this deadline
fpl brief --markdown               # same, as markdown for a report or job summary
fpl chips --horizon 8              # when to play each chip, measured in points
fpl chips --restructuring          # also evaluate Wildcard and Free Hit (slow)
fpl backtest --season 2024-25      # replay a real season, no lookahead, score it
fpl prices                         # overnight price changes and transfer momentum
fpl verify                         # replay history to prove the scoring rules are right
```

`sync` needs network access to `fantasy.premierleague.com`. Everything else runs
from the local cache, so projections and analysis work offline.

Solve times on a full player pool: `brief` and `transfers` take around twenty
seconds, `chips` around two minutes over six gameweeks. `--restructuring` adds a
full re-solve per candidate gameweek and takes considerably longer, which is why
it is opt-in.

## Backtest result — read before trusting a captaincy pick

`fpl backtest` replayed 2024-25 (GW6–38, no lookahead) through the real
projection engine. The bonus tie-break rule is verified exactly against every
real match in two seasons (760/760). The projection engine itself is more
mixed: it beats a do-nothing baseline, but currently **loses to FPL's own
displayed `xP`** on ranking, and badly on picking the single best player each
week — it picked Mohamed Salah as the single best player in roughly 20 of 33
gameweeks, while FPL's `xP` correctly rotated to whoever had the standout
fixture that week. The likely cause: with no odds data in this backtest (the
market-anchored path was never exercised), the fallback team-strength model
doesn't swing a fixture's goal expectation sharply enough to let a good-fixture
week overtake a highly-rated player's season-long baseline. Full evidence and
what that means for trusting `fpl brief`'s captain pick right now:
[`docs/DECISIONS.md`](docs/DECISIONS.md) and [`docs/TODO.md`](docs/TODO.md).

## Automation

Two GitHub Actions workflows are included:

- **`sync.yml`** runs twice daily to snapshot prices and transfer momentum.
  Price changes land overnight and are never reported retrospectively, so
  catching them requires a snapshot either side.
- **`brief.yml`** runs before the usual deadlines and writes the gameweek brief
  into the run summary, readable from a phone.

Both need repository configuration: a variable `FPL_ENTRY_ID` and, if you use
odds, a secret `ODDS_API_KEY`. The snapshot database persists between runs via
the Actions cache.

## Tests

```bash
.venv/bin/python -m pytest
```

Tests run against a generated league with known latent parameters
(`tests/fixtures/synthetic.py`) rather than live data, so they neither depend on
the network nor break when the season moves on.
