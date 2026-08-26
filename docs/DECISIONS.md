# Decisions

The choices in this build that were not obvious, why they went the way they did,
and — where it applies — what going the other way actually cost. Several of these
were made *after* being wrong the first time; those are marked, because the
failure is usually more instructive than the fix.

Format: what was decided, what the alternative was, and why.

---

## Shin de-vigging over proportional normalisation

**Decided:** remove the bookmaker margin with Shin's method.

**Alternative:** divide each implied probability by their sum.

Proportional normalisation assumes the margin is spread evenly across outcomes.
It is not — bookmakers load more margin onto longshots. So proportional
de-vigging systematically **overstates unlikely outcomes**, and that bias lands
directly on low-scoring and clean-sheet estimates, which are exactly what FPL
defenders live on.

On a typical home-favourite book, the difference is around 0.4 percentage points
on the away win. Small, but it is a *bias*, not noise: it points the same way
every fixture, all season.

Both alternatives are kept in `models/devig.py` as fallbacks and for comparison.

---

## Team strength anchored on the market, not fitted alone

**Decided:** invert Dixon-Coles against de-vigged odds; fall back to fitted
ratings only where odds do not exist, blending toward the market as a fixture
approaches.

**Alternative:** fit a ratings model and trust it.

Bookmakers price in team news, lineup leaks, motivation and rotation, and they do
it faster and with far more information than a model fitted on 380 matches a
season can. Competing with that is not a good use of the modelling budget; using
it as an input is close to free.

The fitted model is not wasted — it covers gameweeks with no odds posted, and
acts as an independent check.

---

## Dixon-Coles correction rather than independent Poisson

**Decided:** apply the low-score correction.

Real football is not independent Poisson at low scores: 0-0 and 1-1 occur more
often than independence implies, 1-0 and 0-1 less. Since FPL pays 4 points for a
clean sheet, the 0-0 and 1-0 corner of the distribution is worth getting right.

---

## Rates as shares of team output, not absolute per-90

**Decided:** express a player's rate as a fraction of his team's expected goals,
then multiply by the fixture's goal expectation.

**Alternative:** an absolute per-90 rate, adjusted by a fixture-difficulty
multiplier.

The share formulation makes fixture scaling fall out of the arithmetic. **There is
no fixture-difficulty fudge factor anywhere in this codebase**, and there is
nothing to hand-tune, which means nothing to overfit.

---

## Expected goals rather than goals

**Decided:** shrink xG and xA, not actual goals and assists.

Over a full season the two converge. In September they do not, and xG stabilises
several times faster. Early season is precisely when the model has to be usable
and precisely when actual goals are noise.

---

## Empirical Bayes with an estimated prior strength

**Decided:** estimate the Gamma prior by method of moments from the player pool,
subtracting the Poisson sampling variance from the observed spread of rates.

**Alternative:** pick a shrinkage constant by feel.

The subtraction is the part that matters. Observed rate variance is genuine talent
spread *plus* sampling noise; not removing the noise term inflates the apparent
spread, which under-shrinks, which lets three goals in two starts read as elite.

Measured on generated data with a known prior: recovers a true mean of 0.300 as
0.279, shrinks "3 goals in 2 nineties" from a raw 1.500 to 0.549, and leaves a
well-sampled player at 0.248 against a raw 0.240. That asymmetry is the whole
point.

---

## Hand-encoded scoring rules plus an empirical verifier

**Decided:** encode the rules in `ScoringRules`, and prove them with
`verify_rules`, which replays historical stat lines and compares against the
`total_points` the API reported.

**This corrects an error in the original plan**, which claimed the rules would be
read from the API's `game_settings` at runtime. FPL does not publish a
machine-readable point table, so that was not possible. Rather than quietly
hardcoding and hoping, the guard makes the encoding *falsifiable*: if FPL changes
the game, the mismatch rate says so immediately, and a JSON override fixes it
without a release.

Verification beats good intentions. The rules will change again.

---

## Chip linearisation: counting constraints, not big-M

**Decided:** encode "Bench Boost boosts exactly four players" and "Triple Captain
triples exactly one" as equalities.

**Alternative, tried first:** the textbook big-M form,
`boosted ≥ (squad − start) − (1 − bench_boost)`.

Both are exact. The big-M form is disastrously weak: with a fractional chip
binary, the relaxation can spread a fraction of a chip across every gameweek at
once, and the branch tree explodes.

**Measured:** a four-gameweek Bench Boost solve ran past **fifteen minutes** with
big-M, and **twelve seconds** with counting constraints. Same answer, same model,
1000× the usability.

This was found by running it, not by reading it. It would not have shown up in
any test that used a small pool.

---

## Chip timing discount: 0.97, not the planner's 0.85

**Decided:** evaluate chips at a near-neutral discount.

**This was wrong the first time.** Chip evaluation inherited the transfer
planner's 0.85-per-gameweek discount, and the reason that discount exists does
not transfer.

The planner discounts the future because a transfer plan for gameweek 8 will be
revised long before it is executed. But a chip decision is a *comparison between
playing it now and playing it in gameweek 8*, and discounting the later branch at
0.85⁵ shrinks a 12-point double-gameweek Bench Boost to 5. The model therefore
advised burning chips immediately — the exact mistake chip strategy exists to
avoid.

The fix visibly changed the answer: on the same data, chips went from "play both
this week" to Triple Captain in GW4 and Bench Boost in GW6. A regression test now
asserts a chip is saved for a clearly better later gameweek.

A small residual discount remains, because a distant chip week can still be
disrupted by a fixture change.

---

## Price-banded pruning

**Decided:** when cutting the pool before solving, reserve part of the budget for
the best option in each price band.

**This was wrong the first time.** Ranking by projection alone kept only premiums,
and the resulting program was **genuinely infeasible** — no £100m squad could be
built from the pool. Cheap enablers are what make a premium affordable, so they
have to survive pruning.

---

## Legality re-check on unproven solves

**Decided:** a time-limited solve is promoted to `Feasible` only after the
extracted squad is re-checked against size, shape, club limits and budget.

**This was wrong the first time.** The first version accepted anything that
*looked* like a 15-player squad. But an infeasible solve also leaves values on the
variables — LP relaxation values that violate the budget — and the optimiser duly
produced a £75m squad on a £60m budget and called it feasible.

A solver quirk must never become illegal advice. The check is cheap and it is the
only thing standing between the two.

---

## Free transfers get a token objective weight

**Decided:** add `0.01 × free_transfers` to the objective.

`free = min(5, remaining + 1)` is expressed as upper bounds, which the solver only
pushes against when a hit is actually at stake. Without the token weight the
decisions are still right but the *reported* free-transfer balance is arbitrary —
and a brief that misreports your balance is worse than one that says nothing.

At 0.01 it is two orders of magnitude below the value of a transfer, so it cannot
buy a decision.

---

## Transfer value measured against holding, over the horizon

**Decided:** solve twice — freely, and with this gameweek's transfers capped at
zero — and difference the objectives.

**Alternative:** report the projected-points difference in the current gameweek.

A transfer is rarely about this week. It sets up a fixture swing, or forecloses
one. Measuring only the immediate gain systematically undervalues good transfers
and overvalues bad ones, and it cannot explain why the optimiser sometimes wants
to hold three weeks and then move three players at once.

---

## Refusing to guess, twice over

**Decided:** unmatched odds events are reported rather than force-matched, and
ambiguous player names raise an error listing the candidates.

The failure modes are asymmetric. Having no odds for a fixture costs a little
accuracy and the model falls back gracefully. Attaching *the wrong match's* odds
produces a projection that is confidently wrong, with nothing to indicate it.
Likewise, picking the wrong Silva corrupts the purchase-price ledger — the one
piece of state no endpoint can rebuild.

Where a wrong answer is much worse than no answer, refuse.

---

## Snapshot-based architecture

**Decided:** every sync writes a timestamped snapshot; models read from cache.

Originally forced by the build environment, which cannot reach the FPL API. But
it was required anyway: price changes are never reported retrospectively, so
detecting them needs before-and-after observations, and an honest backtest needs
to know what was knowable at each deadline. The constraint and the requirement
happened to agree.

---

## Testing against a generated league

**Decided:** tests run against a synthetic league with known latent parameters.

**Alternative:** captured real API payloads.

Captured fixtures prove the code runs. A generated league proves the models
**recover the parameters they were generated from** — that `fit_ratings`
correlates above 0.8 with true attack strength, that `fit_rate_prior` finds the
true prior mean. That is a much stronger claim, and it is the one worth making
about a statistical model.

---

## Still open

- **Bonus** is a normal approximation on expected BPS. Bonus is a within-match
  ranking; the honest treatment needs the Monte Carlo engine. This is the weakest
  term in the projection and the first thing Phase 3 replaces.
- **Free Hit** is evaluated with wildcard semantics — an upper bound, because the
  squad revert is not modelled.
- **Bench weights** are constants by position rather than the auto-sub
  probability the minutes model could supply.
- **Prices are static** across the horizon. Price-change prediction is Phase 4
  and is worth far less than getting the points right.
- **Nothing is calibrated against reality yet.** Every number here is validated
  against generated data or internal consistency. The backtest is Phase 3, and
  until it exists these models are plausible rather than proven.
