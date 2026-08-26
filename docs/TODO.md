# Your to-do list

Things only you can do. Everything else is mine.

Ordered by when, not by size. **Bold** items are time-critical — they get harder
or impossible to do later.

---

## Now — before you can use it (Phases 0–2, built)

- [ ] **Clone and install locally.** My build sandbox is blocked from
      `fantasy.premierleague.com`, so every live pull has to run on your machine.
      `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- [ ] **Run `fpl sync --deep` and tell me what it says.** This is the only
      confirmation that entry id 1807966 is right — I could not verify it. If it
      reports no such entry, grab the number from your Points page URL.
- [ ] **Record what you actually paid for each player.**
      `fpl set-squad "Raya, Gabriel@6.1, ..." --bank 0.5 --free-transfers 2`
      FPL returns only half of a player's price rise when you sell, and no
      endpoint reports purchase prices retrospectively. Every gameweek this waits
      is drift you cannot recover.
- [ ] Get a free odds API key at the-odds-api.com and
      `export ODDS_API_KEY=...`. Optional, but it is the single largest accuracy
      gain available.
- [ ] Run `fpl verify` once. It proves the scoring rules reproduce real history.
      If it fails, stop and tell me — every projection depends on it.

## Before Phase 3 lands (calibration and backtest)

- [ ] Tell me any chips you have already played, and in which gameweek.
      Otherwise the chip planner will offer you chips you no longer have.
- [ ] Read the backtest result when I deliver it, and decide whether you trust
      the model over your own judgement on close calls. Until then the numbers
      are plausible, not proven.

## When automation goes live (Phase 4)

- [ ] In GitHub → Settings: add repository **variable** `FPL_ENTRY_ID` and
      **secret** `ODDS_API_KEY`, and enable Actions.

## Every gameweek

- [ ] `fpl sync && fpl brief` before the deadline. That is the whole routine.
- [ ] Re-run `fpl sync` soon after making a transfer, so the ledger records the
      purchase price before it moves.

## Every August

- [ ] Update `entry_id` in `fpl.toml`. FPL reissues entry ids every season.
