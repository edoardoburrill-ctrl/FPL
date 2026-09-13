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
- [ ] **Get a free odds API key at the-odds-api.com and
      `export ODDS_API_KEY=...`.** No longer just "nice to have": the first real
      backtest (below) showed the model losing to FPL's own displayed
      expected-points on captaincy while running *without* odds data. Getting
      the market-anchored path actually exercised is the most likely fix.
- [ ] Run `fpl verify` once. It proves the scoring rules reproduce real history.
      If it fails, stop and tell me — every projection depends on it.

## Read this before trusting a captaincy pick

`fpl backtest --season 2024-25` replayed a real season with no lookahead. It
beats guessing, but it currently **loses to FPL's own expected-points number**
on ranking quality, and badly on picking the single best player each week (197
points captured over a season vs 360 for trusting FPL's own figure) — it picked
Mohamed Salah as the top pick in roughly 20 of 33 gameweeks regardless of that
week's specific fixture, while FPL's number correctly rotated to the week's
actual standout. Most likely cause: no odds data means the fallback
team-strength model doesn't swing sharply enough fixture to fixture. Until this
closes — most likely by validating with real odds data — **for captaincy,
weigh the model's pick against FPL's own displayed `xP`, don't defer to it
blindly.** Full evidence: `docs/DECISIONS.md`.

- [ ] Tell me any chips you have already played, and in which gameweek.
      Otherwise the chip planner will offer you chips you no longer have. This
      now matters even more for Free Hit specifically: it is modelled properly
      (the squad reverts next gameweek, as it does in real FPL) rather than as
      a permanent Wildcard-style upper bound, so its reported value is real,
      not inflated — the chip calendar's advice is only as good as knowing
      which chips are actually still available.

## When automation goes live (Phase 4)

- [ ] In GitHub → Settings: add repository **variable** `FPL_ENTRY_ID` and
      **secret** `ODDS_API_KEY`, and enable Actions.

## Every gameweek

- [ ] `fpl sync && fpl brief` before the deadline. That is the whole routine.
- [ ] Re-run `fpl sync` soon after making a transfer, so the ledger records the
      purchase price before it moves.

## Every August

- [ ] Update `entry_id` in `fpl.toml`. FPL reissues entry ids every season.
