# Design — the system as built

[`PLAN.md`](PLAN.md) describes what was intended and why, with the mathematics
in full. This document describes what exists: what each module owns, what it
guarantees, and where the seams are. [`DECISIONS.md`](DECISIONS.md) records the
choices that changed along the way and the reasoning behind them.

Read this before changing anything: most of the subtle behaviour here is load
bearing, and several pieces look redundant until you know what they are
defending against.

---

## 1. Shape of the system

Four layers, each usable and testable without the ones above it.

```
  data      fetch and remember          fpl/data/
    ↓       raw payloads, snapshots, no interpretation
  models    turn data into expectations fpl/models/
    ↓       a points distribution per player per gameweek
  optimize  turn expectations into a plan  fpl/optimize/
    ↓       squad, eleven, captain, transfers, chips
  report    turn a plan into instructions  fpl/report/
```

`fpl/app.py` wires them together; `fpl/cli.py` is the only user-facing surface.
`fpl/state/` cuts across all four: it holds what is true about *your* team.

The layering is strict in one direction — models never import from `optimize`,
`optimize` never imports from `report` — with one deliberate exception noted in
§6.

---

## 2. The data layer

### `data/fpl_client.py`

Fetches from the official API and **interprets nothing**. Returns decoded JSON.

Guarantees:

- Every successful response is written to a disk cache keyed by endpoint.
- A fresh cache entry (within `ttl_seconds`) is served without a network call.
- A **transport failure** falls back to a stale cache entry if one exists, so a
  flaky connection degrades to slightly old data rather than an error.
- A **404 never falls back to cache**, and raises `FPLNotFound`. This asymmetry
  is the point: a network problem should retry later, whereas a 404 on an entry
  means the id is wrong and cached data would hide that behind stale results.

`FPLNotFound` subclasses `FPLUnavailable`, so callers that do not care about the
distinction can still catch one exception.

### `data/odds_client.py`

Fetches bookmaker odds, takes a **median consensus per outcome across
bookmakers** before de-vigging (one stale book cannot drag a fixture), and
de-vigs with Shin's method.

`match_to_fixtures` returns `(matched, unmatched)`. Unmatched events are
**reported, never guessed at**: attaching the wrong match's odds to a fixture is
worse than having no odds, because the projection then looks confident and is
wrong. Team names are folded through the shared normaliser (§5) plus an alias
table for the cases folding cannot reach (`"Wolverhampton Wanderers"` →
`"wolves"`).

### `data/store.py`

Time-stamped snapshots in SQLite. Every row carries `observed_at`.

This exists for two things the API cannot give you. Price changes happen
overnight and are never reported retrospectively, so detecting them requires
having looked before and after. And an honest backtest must only read rows
observed before the deadline it is replaying — the timestamp column is what will
make that enforceable.

---

## 3. The model layer

The contract of this layer: given the raw payloads, produce an expected-points
number per player per gameweek, with a breakdown that can be explained.

### `models/scoring.py`

The scoring rules, and the guard that keeps them honest.

FPL publishes no machine-readable point table, so `ScoringRules` is hand-encoded.
Hand-encoded rules rot silently when the game changes. `verify_rules` replays
historical stat lines through `score_stat_line` and compares against the
`total_points` the API reported; `fpl verify` surfaces the mismatch rate. A
tolerance of 0.5% absorbs retrospective dubious-goal adjustments; anything above
that means the table is wrong. `load_rules` accepts a JSON override so a
mid-season change is a config edit, not a release.

**Invariant:** every points computation in the system goes through
`score_stat_line` or the same `ScoringRules` instance. Nothing hardcodes a 4 or a 6.

### `models/minutes.py`

Three-outcome hurdle model: `p_out`, `p_bench`, `p_start`, summing to 1.

What FPL actually pays for is the **60-minute threshold**, so `p_60plus` is
modelled explicitly rather than derived from expected minutes. Rates are shrunk
Beta-Binomial toward a prior estimated empirically by position and price tier.

Two details that look wrong and are not:

- Recency weights are **normalised to sum to `n`**, so decay reshapes the
  evidence without shrinking it. Without this the prior swamps ten matches of
  genuine evidence.
- `synthetic_history` reconstructs a plausible match log from season aggregates
  for players not worth a per-player API call, and **interleaves** the match
  types rather than blocking them, because aggregates carry no ordering
  information and blocked rows would let the recency weighting read a trend that
  the data does not contain.

### `models/team_strength.py`

`(lam_home, lam_away)` per fixture, from two sources blended by horizon.

`solve_from_market` inverts a Dixon-Coles bivariate Poisson against de-vigged
market probabilities. `fit_ratings` is the fallback where odds do not exist:
time-decayed weighted MLE with a ridge penalty toward league average, which is
what stops three August matches producing absurd ratings.

`score_matrix` is the joint distribution everything defensive is computed from —
clean sheets and goals conceded come from it, never from an independent Poisson.

### `models/player_rates.py`

Empirical-Bayes shrinkage. `fit_rate_prior` estimates the prior by method of
moments, subtracting the Poisson sampling term from the observed cross-sectional
variance — that subtraction is what stops small-sample noise being mistaken for
real spread between players, which would cause systematic under-shrinking.

`output_share` converts an absolute per-90 rate into a share of team output, so
multiplying by a fixture's goal expectation scales a player automatically. **There
is no fixture-difficulty multiplier anywhere in this codebase**; that is the
mechanism that replaces it.

### `models/points.py`

Assembles the breakdown. Every term is an expectation **over a distribution**,
never a threshold applied to an average, because FPL scoring is full of
thresholds: 60 minutes, every 2 conceded, every 3 saves, 10 or 12 defensive
actions.

Clean sheets and goals conceded are evaluated over the minutes the player is
actually on the pitch, via binomial thinning of the match concession
distribution — FPL only counts goals conceded while he is playing.

**Known weakness, stated in the module docstring:** bonus uses a normal
approximation on expected BPS. Bonus is a within-match ranking and the honest
treatment needs simulation. It is monotone and roughly calibrated, and it is the
first thing Phase 3 replaces.

Every term also carries a **variance**, alongside its mean, in
`PlayerProjection.variance` — derived from the same distributional assumption
the mean already makes for that term (Poisson for goals/assists/cards,
Bernoulli for clean sheets and defensive contributions, the on-pitch
concession pmf for goals conceded, the normal-threshold approximation for
bonus), not a fitted or guessed constant. `points_variance` sums the terms
as independent, which is a simplification in the same spirit as the mean
breakdown's own term-by-term independence. This exists for
`optimize/robustness.py`'s bootstrap resampling, not for the mean projection
itself — nothing in the analytic points assembly reads it back.

### `models/projection.py`

The pipeline. `build_context` fits everything that does not depend on which
gameweek is being projected (ratings, priors, team contexts); `project_gameweek`
sums over a player's fixtures in that gameweek — so a double gameweek sums two
and a blank sums none, with no special-casing.

### `models/bps.py`

Two things at very different confidence levels, and the module says so.
`award_bonus` implements FPL's exact tie rule (standard competition ranking —
a tie for 1st takes rank 1 twice and the next player is rank 3, not rank 2) and
is **verified against every real match in two full seasons: 760/760**. The BPS
point table itself is not verifiable this way — FPL publishes no machine-
readable table, and the one historical archive available lacks the raw
defensive-action counts needed even in principle. `partial_bps` was originally
built as a strict lower bound on the theory that every omitted term (CBI,
goalline clearances) only adds to the true figure; checking it against real
data disproved that (fouls conceded is also unobserved, and subtracts), so
`verify_plausibility` reports a residual distribution instead of asserting a
direction. See `DECISIONS.md` for both stories.

### `models/simulate.py`

The Monte Carlo match engine. Where the analytic path (`points.py`) treats
every scoring term as an independent expectation, this module draws whole
match scenarios — a scoreline, then minutes, then goals and assists allocated
to scorers by a minutes-weighted multinomial, then defensive actions, saves and
cards — and computes BPS and ranks the *whole match* for bonus, using
`bps.award_bonus`. This is what makes bonus a genuine rank rather than a normal
approximation, and it is also what makes haul probabilities and teammate
correlations available at all: an independent-expectation model cannot produce
either honestly.

Not wired into the optimiser's pruning path — it is too slow for that by
design (≈2s for one 22-player fixture at 10k draws; the per-draw bonus
allocation is a pure-Python loop and is the bottleneck). It exists for
enriching a shortlist: `fpl explain`-style output, not `fpl transfers`.

---

## 3b. Historical data and the backtest

### `data/historical.py`

Real per-gameweek FPL data for past seasons, from a public archive
(github.com/vaastav/Fantasy-Premier-League) — not the live API, and the only
network dependency in this codebase that isn't `fantasy.premierleague.com`.
Cached indefinitely, since a finished season never changes. `iter_match_bps`
groups player-rows by fixture to reconstruct real per-match BPS and bonus
without needing to parse the fixtures file's embedded stats blob.

### `backtest/adapter.py`

Converts historical rows into exactly the shapes `build_context` and
`project_gameweek` already consume from the live API, so the backtest exercises
the real pipeline rather than a parallel one. `snapshot_as_of(data, gameweek)`
is the load-bearing function: it aggregates only rows from gameweeks strictly
before the target. A snapshot that saw the target gameweek's own results would
not be predicting anything.

One real consequence of this design, not a bug: a player is only known to a
snapshot once he has appeared in an already-played gameweek, since the archive
carries no separate roster listing the way `bootstrap-static` does live. The
gameweek-1 snapshot is therefore empty, which is why `replay_season` defaults
to starting at gameweek 6.

### `backtest/replay.py`, `backtest/metrics.py`

Walks a season forward gameweek by gameweek, no lookahead, scoring the real
pipeline's projections against real `total_points` and against FPL's own `xP`
— the baseline this project's own plan named as the bar to beat. Three
comparisons: MAE against actual, per-gameweek Spearman rank correlation, and a
concrete captaincy stand-in (`top_pick_points`: sum of real points from each
gameweek's single highest-projected player).

**Deliberately does not replay the optimiser.** No transfers, no squad, no
budget — this validates the projection engine specifically, which is the layer
everything else depends on. See `DECISIONS.md` for why that scope was chosen
and what the first real run found.

---

## 4. The optimiser

### `optimize/milp.py`

One integer program over a rolling horizon deciding squad, eleven, captain,
transfers, hits and chips together.

FPL's nonlinear rules are linearised **exactly**:

| Rule | Encoding | Why it is tight |
|---|---|---|
| `hits = max(0, transfers − free)` | two inequalities | objective penalises hits |
| `free = min(5, remaining + 1)` | two upper bounds | token objective weight on holding transfers |
| Bench Boost / Triple Captain | product of two binaries | counting constraints, see below |

The free-transfer epsilon (`FREE_TRANSFER_EPSILON = 0.01`) exists because the
`min` is expressed as upper bounds, which the solver only pushes against when a
hit is at stake. Without it the *reported* balance is arbitrary even though the
decisions are right. It is two orders of magnitude below the value of a transfer,
so it cannot buy a decision.

Chips use **counting constraints** (`sum(boosted) == 4 * bench_boost`), not big-M
lower bounds. Both are exact; the big-M form is catastrophically weak here. See
[`DECISIONS.md`](DECISIONS.md#chip-linearisation).

`prune_pool` cuts the pool before solving, **banding by price** so cheap enablers
survive. Ranking by projection alone leaves a pool of premiums that no £100m
squad can afford — that produced a genuinely infeasible program during
development.

**Safety invariant:** a time-limited solve that stops unproven is returned as
`Feasible` only after `_is_legal` re-checks the extracted squad against size,
shape, club limits and budget. An infeasible solve also leaves values on the
variables — relaxation values that violate the budget — and accepting anything
that merely looks like a squad would emit illegal advice. This was a real bug
before the guard existed.

### `optimize/chips.py`

Bench Boost and Triple Captain are decided *inside* one solve — the program picks
the gameweek, and will build toward it. Wildcard and Free Hit restructure the
problem rather than adding a term, so they are evaluated by re-solving per
candidate gameweek; that is much slower and therefore opt-in.

Chip evaluation uses `CHIP_DISCOUNT = 0.97`, not the planner's 0.85. See
[`DECISIONS.md`](DECISIONS.md#chip-timing-discount).

Free Hit is currently evaluated with wildcard semantics, which is an **upper
bound** on its value because the squad revert is not modelled. Stated in the
code, not glossed.

### `optimize/transfers.py`

`TransferMove` and `pair_transfers`, pairing raw buy/sell decisions into
"sell X, buy Y" moves by position. Lives here rather than in `report/`
because `optimize/robustness.py` needs it too, and `optimize` must not
depend on `report` (§1) — a report-layer utility two optimiser-layer callers
both need belongs to the lower layer, not the higher one.

### `optimize/robustness.py`

The MILP optimises against a single point estimate per player-gameweek.
`transfer_stability` asks a cheaper question instead: re-solve the horizon
against `n_resamples` bootstrap draws of the projections — each player's
point estimate perturbed by a normal draw from its own projection variance
(`PlayerProjection.points_variance`, `models/points.py`) — and report what
fraction of resamples chose each transfer move. A move chosen in 95% of
resamples is a real edge; one chosen in 40% is a coin-flip that should not
cost a hit.

This deliberately does not touch `models/simulate.py`'s Monte Carlo engine:
that draws whole joint match scenarios and is too slow to re-run per
resample (≈2s per fixture at 10k draws, and a resample re-solves the *whole
horizon's* MILP on top of that). Resampling the analytic path's point
estimates against their own analytic variance is a different, cheaper
approximation — it captures projection uncertainty, not the correlation
structure between teammates the Monte Carlo engine would add. Opt-in via
`stability_resamples` on `build_brief` (`--stability` on `fpl brief` /
`fpl transfers`) because it multiplies solve time by roughly that count; 0
by default.

Each term in `PlayerProjection.variance` is derived analytically from the
same distributional assumptions the mean already uses — Poisson for goals,
assists and cards; Bernoulli for clean sheets and defensive contributions;
the on-pitch concession pmf directly for goals conceded; the same
normal-threshold approximation as `expected_bonus` for bonus — rather than
an arbitrary noise constant. Terms are then summed as if independent, which
understates the true variance somewhat (a player who scores is also more
likely to pick up bonus), but is a large improvement over treating every
projection as certain, which was the only alternative.

---

## 5. State

### `state/ledger.py`

`Holding.selling_price` implements FPL's rule: you get back the purchase price
plus half of any rise, rounded down. This is why the purchase-price ledger
matters and why it must be seeded early — **no endpoint reports historical
purchase prices**, so this is the one piece of state that cannot be
reconstructed after the fact.

`reconstruct_free_transfers` replays the entry's history forward, because the API
reports transfers made but never the free-transfer balance. Wildcard and Free Hit
gameweeks do not consume a free transfer, and are skipped.

`chips_available` enforces the two-set, half-season structure: a chip used before
the GW19 deadline is available again from GW20, and never carries across.

### `state/resolve.py`

Name resolution for manual squad entry, and the shared `normalise` used by the
odds client's team matching.

`normalise` carries an explicit transliteration table for letters that are **not**
accented ASCII — `ø`, `ł`, `đ`, `ß`, `æ`. NFKD leaves these intact and the
subsequent ASCII encode drops them entirely, so "Højlund" folded to "hjlund" and
matched nothing. This was a real bug, caught by a test.

`resolve_one` **refuses to guess**. An ambiguous surname raises an error listing
candidates with club, position and price. Silently picking the wrong Silva would
corrupt the purchase-price ledger — the one thing that cannot be rebuilt.

---

## 6. Reporting

`report/brief.py` quantifies a transfer by solving the horizon twice: once
freely, once with `transfer_caps={gameweek: 0}`, and differencing. The value of
acting is measured against **holding, over the whole horizon** — not against this
gameweek alone, which would ignore what a transfer sets up or forecloses.

Rotation warnings are raised only for the starting eleven. A £4.2m bench player
who never starts is doing exactly the job he was bought for; warning about him
weekly trains you to ignore the warnings that matter.

The layering exception: `report/brief.py` imports from `optimize/`, because
building a brief means running the optimiser twice. Reporting is downstream of
optimisation, so this is the direction of the arrow, not a violation.

---

## 7. Testing strategy

Tests run against `tests/fixtures/synthetic.py` — a generated 20-team, 500-player
league in the exact shape of the real API payloads, with **known latent
parameters**.

Two reasons. The build environment cannot reach `fantasy.premierleague.com`, and
tests that depend on live data fail on a Tuesday for reasons unrelated to the
code. But the real payoff is that tests can assert the models *recover* the
parameters the league was generated from, rather than merely that they run:
`fit_ratings` must correlate above 0.8 with the true attack ratings;
`fit_rate_prior` must recover the true prior mean.

Optimiser tests use deliberately tiny pools. Whether the rules are enforced is
proved as well by a 42-player league as a 700-player one, and proved in a second
rather than a minute.

**One deliberate exception.** `data/historical.py` and the backtest it feeds are
tested offline like everything else — `test_backtest.py` uses small embedded
fixtures, not the live archive. But the claims that actually matter here (the
tie rule against 760 real matches; the season backtest's real MAE and Spearman
numbers) come from interactive runs against the real archive, recorded in
`DECISIONS.md`, not re-derived by the test suite on every run. The archive is
more stable than the live FPL API but still an external dependency; baking a
network call into `pytest` would violate the same policy this section opened
with, for a much larger and slower fetch.

---

## 8. Where the seams are

If you are extending this, these are the intended insertion points.

| Want to | Change |
|---|---|
| Add a data source | a client in `data/`, then wire it in `app.py` |
| Change how points are computed | `models/points.py`, one term at a time |
| Replace the bonus model | `expected_bonus` in `models/points.py`, or wire `models/simulate.py`'s ranked bonus into the fast path |
| Change what the optimiser maximises | the objective block in `milp.py` |
| Add a chip | `milp.py` if it adds a term, `chips.py` if it restructures |
| Add an output format | `report/`, reusing `build_brief` |
| Improve the backtest | `backtest/adapter.py` for a new data source, `backtest/metrics.py` for a new comparison |
| Change a term's variance model | the matching `*_variance` helper in `models/points.py`, next to the mean it corresponds to |
| Change how transfer stability is judged | `optimize/robustness.py`, `resample_players` for the noise model or `transfer_stability` for the solve loop |

The one thing not to do is compute points anywhere except through
`ScoringRules`. That is what keeps `fpl verify` meaningful.
