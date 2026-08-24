# FPL Advisor

A quantitative decision engine for Fantasy Premier League. It projects a
probability distribution of points for every player in every upcoming gameweek
and turns that into concrete weekly decisions: transfers, captain, bench, chips.

The full design and the mathematics behind each component are in
[`docs/PLAN.md`](docs/PLAN.md).

## Status

| Phase | Contents | State |
|---|---|---|
| 0 | Data clients, snapshot store, squad ledger, CLI | done |
| 1 | Projection engine: minutes, team goals, player rates, points | done |
| 2 | Multi-gameweek transfer optimiser | done |
| 3 | Monte Carlo engine, ranked bonus, calibration, backtest | planned |
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

## Use

```bash
fpl sync --entry 1234567 --deep    # pull data, snapshot prices, refresh squad state
fpl projections --gw 3 --top 25    # expected points table
fpl projections --position DEF --max-cost 5.5
fpl captain --gw 3                 # captaincy ranking with the margin to the next option
fpl squad                          # your squad, selling prices, bank, chips, free transfers
fpl explain haaland                # term-by-term breakdown of a projection
fpl transfers --horizon 5          # multi-gameweek transfer plan
fpl brief                          # the one command: what to do before this deadline
fpl brief --markdown               # same, as markdown for a report or job summary
fpl chips                          # when to play each chip, measured in points
fpl prices                         # overnight price changes and transfer momentum
fpl verify                         # replay history to prove the scoring rules are right
```

`sync` needs network access to `fantasy.premierleague.com`. Everything else runs
from the local cache, so projections and analysis work offline.

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
