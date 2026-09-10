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

## Bonus tie-break rule: verified against every real match in two seasons

**Decided:** standard competition ranking ("1224" — a tie for 1st means both
take rank 1 and the next player is rank 3, not rank 2), sourced from FPL's own
published worked examples, not inferred.

This one did not have to stay a guess. `data/historical.py` groups two seasons
of real per-fixture BPS and bonus data (760 real matches, 2023-24 and 2024-25).
`award_bonus` reproduces the actual bonus award in **760 / 760 — 100%**. This is
the strongest verification any component in this codebase has: not three
documented examples, every real match played across two seasons.

**Caught by testing, not review.** The first implementation had the rank→bonus
payout table inverted — `{3: 3, 2: 2, 1: 1}` (a coincidental pass-through)
instead of `{1: 3, 2: 2, 3: 1}` (rank 1 pays 3). It silently handed bonus to the
*worst* BPS performer in every ungrouped case. The three worked-example tests
written immediately after caught it before it reached the real-data check at
all — which is the point of writing the narrow test before reaching for the big
one.

---

## The BPS table's lower bound turned out not to be a bound

**Decided, then undone:** `partial_bps` was designed as a strict lower bound —
every constant included was meant to be one no larger than the true
contribution, with CBI and goalline clearances the only omissions, both
additive, so the partial figure could only undershoot the real `bps`.

Checking it against two real seasons disproved the premise: **23-29% of real
stat lines had the partial estimate *exceed* the real BPS**, including plain
appearance-only rows with no goals, assists, or cards. The residuals ran as low
as −20. The lower-bound property assumed the only unobserved terms were
additive; it missed that **fouls conceded** — and, in these older archive
seasons specifically, the since-removed tackled-penalty — are also unobserved
and subtract.

The fix was not to tune the constants against this signal. Tuning against a
confounded target (goals could include unflagged penalties at old rates; "zero
other stats" rows still have invisible CBI content that can only ever push the
residual in the *other* direction) would trade one unfounded number for
another. Instead `verify_lower_bound` became `verify_plausibility`: it reports
the residual distribution and flags only extreme outliers, rather than
asserting a direction the evidence does not support. The underlying BPS
constants are unchanged and remain exactly as uncertain as before — this
correction is about not overclaiming what was checked, not about having fixed
the table.

---

## The backtest validates the projection engine, not the optimiser

**Decided:** `fpl backtest` replays a real season's projections against real
outcomes with a strict no-lookahead cutoff, and scores them against FPL's own
`xP` — but it does not replay transfers, budget, or squad selection.

**Why the smaller scope was the right call.** A full replay — the optimiser
making real transfer decisions against a real starting squad and budget across
a season — is what PLAN.md originally described, and it is meaningfully bigger:
it needs the price-history-aware selling-price ledger reconstructed for a whole
season, a decision at every deadline, and a much larger validation surface where
a bug anywhere in the optimiser can masquerade as a modelling problem. The
projection engine is the layer everything else is built on, so it is the
correct thing to validate *first* and in isolation. A full optimiser replay
remains open, tracked below.

**A real result, not a projected one — and not a flattering one.** Run against
2024-25, GW6 through GW38 (33 gameweeks, 9,858 player-gameweeks):

```
MAE          ours=1.842  FPL xP=1.785  season-mean baseline=2.094
Spearman rho ours=0.345  FPL xP=0.570  (per-gameweek, averaged)
Top-pick pts ours=197    FPL xP=360    (sum of real points from each week's #1 pick)
```

The honest reading: this beats doing nothing (MAE below the season-mean
baseline), but it does **not** currently beat FPL's own `xP` — close on MAE,
clearly behind on ranking quality, and badly behind on the metric that matters
most for captaincy (picking the single best player each week returned about
half the points FPL's own number would have).

**The first explanation reached for was wrong, and checking it found the real
one.** Before looking at a single pick, the natural guess was "no odds data,
no injury signal, both structural gaps in this backtest, not necessarily the
live tool." That is a reasonable-sounding story and it is not what the evidence
shows. Printing the actual top pick each week (see the table this session's
diagnostic produced) shows no pattern of picking an injured or suspended player
who then scores zero — the pattern is completely different and much more
specific:

```
GW  our pick              pts   xP pick               pts
 9  Erling Haaland          6   Chris Wood            13
10  Cole Palmer             3   Mohamed Salah          9
17  Matheus Cunha           9   Mohamed Salah         21
18–28  Mohamed Salah      (9 of 11 weeks)   Mohamed Salah      (matches ours 7 of 11)
29  Chris Wood              2   Bruno Fernandes       17
30  Mohamed Salah           3   Anthony Elanga        11
31  Mohamed Salah           2   Jacob Murphy          16
33  Mohamed Salah           3   Leandro Trossard       8
35  Mohamed Salah           2   Yoane Wissa            7
36  Bryan Mbeumo             6   Kevin Schade          11
38  Mohamed Salah          10   Jarrod Bowen          13
```

**The model picked Mohamed Salah as the single best player in roughly 20 of 33
gameweeks.** FPL's `xP` did not — it rotated to whoever had the standout
fixture that specific week (Wood, Elanga, Murphy, Trossard, Wissa, Bowen — a
different name most weeks). This is not a selection-logic bug; each gameweek is
projected independently with no memory of prior picks. It is the projection
engine itself lacking enough **week-to-week fixture reactivity**: once a
player's season-long attacking share is fitted high enough, he dominates the
projection almost every week regardless of whether that particular fixture is
good or poor, because the fallback fitted-ratings team-strength model — the
only one exercised here, since no historical odds exist to anchor it — updates
attack/defence ratings too slowly and too smoothly to swing a specific
fixture's goal expectation as sharply as it should.

This sharpens, rather than replaces, the original hypothesis. The root cause is
still "no odds data" — but the mechanism is now evidenced, not assumed: it is
specifically the fallback model's *lack of fixture-to-fixture swing*, not a
missing injury flag, that is starving the top pick of the signal it needs to
rotate. The injury-signal gap likely still costs some accuracy at the margins,
but the diagnostic gives it no support as the dominant factor here, and it is
downgraded to secondary rather than co-equal.

Confirming this needs a backtest run with real historical odds anchoring the
team-strength model, which is future work — not something to claim done now.
What is not in question is the number itself: on this test, as it stands today,
**the model should not be trusted over FPL's own displayed expected points for
captaincy**, and `docs/TODO.md` says so.

---

## Bench weight scaled by the player's own appearance chance

**Decided:** the objective's bench-value term multiplies the position-based
base rate (`BENCH_WEIGHT`, calibrated at `REFERENCE_APPEARANCE = 0.75`, a
fairly nailed player) by `player's own P(plays any minutes) / REFERENCE_APPEARANCE`
for that gameweek, clipped to `[0, 1]`.

**Alternative:** the flat per-position constant this replaced.

An auto-sub only pays off if the bench player himself gets minutes when
called upon — a flat constant credits a rotation-risk differential on the
bench the same as a nailed one, which overvalues the gamble. The minutes
model already produces `p_appears` per player-gameweek for the analytic
points path, so threading it through to the bench term was free: no new
model, no new data. This does not yet account for auto-sub *order* (GK →
outfield by bench slot, itself dependent on which teammate blanks), which
stays future work; the fix is specifically to the "does this bench player
convert the chance he gets" half, not the "does a chance arise" half.

---

## Bootstrap-robust transfer recommendations

**Decided:** give every breakdown term in `models/points.py` an analytic
variance alongside its mean, sum them into `PlayerProjection.point_std`
(terms treated as independent, the same simplification the means already
make of each other), thread it through to the optimiser's `Player`, and add
`optimize/robustness.py`: re-solve the horizon `n_resamples` times against
projections perturbed by `Normal(0, point_std)`, and report what fraction of
resamples chose the same `(sell, buy)` pair as the point-estimate plan. This
is exactly the mechanism PLAN.md's §4 "Robustness" section describes; it was
written into the plan but not built until now. `fpl transfers --bootstrap N`
and `fpl brief --bootstrap N` surface it (default off — it is `N` extra full
re-solves, the same cost class as `fpl chips --restructuring`).

**Alternative — draw the resamples from `models/simulate.py` instead.** The
Monte Carlo engine already produces a proper per-player sample and would give
a more honest variance (real correlations, the ranked-BPS bonus, no
independent-terms assumption). Rejected for this use *specifically* because
`simulate.py`'s own docstring is explicit that it is "not what the optimiser
prunes hundreds of players against" — it is priced for a shortlist, not a
few-hundred-player pool repeated `n_resamples` times over a multi-gameweek
horizon. The analytic variance is cheap enough to attach to every projection
by default, which the simulation is not. If the fast-path bonus rewrite
below ever happens, revisiting this with simulation-drawn resamples for a
shortlist-sized robustness check is worth doing then.

**Why sum term variances as independent when the terms are not.** Goals and
assists both scale with the same fixture goal expectation, so they are
positively correlated in reality; treating them as independent understates
the true variance somewhat. This is the same simplification `project_player`
already makes for the *means* (each term computed as its own expectation,
not as a joint draw), so it is not a new source of approximation, and the
resulting `point_std` only needs to rank "how much noise is on this pick"
well enough to separate an obvious transfer from a coin-flip — not to be a
precise number in its own right.

---

## Free Hit's squad revert, modelled exactly rather than bounded by Wildcard

**Decided:** let the persistent squad chain skip the Free Hit gameweek. The
gameweek after a Free Hit computes its "previous squad" from two gameweeks
back rather than one, so the Free Hit's own transient squad never enters the
chain; every later budget check likewise excludes the Free Hit gameweek's
spend from the running "money already committed" total; and the reported
bank for each gameweek does the same when walking the solved plan back out
in Python. Free Hit itself reuses the wildcard branch's unlimited-transfers,
no-hit, free-transfer-preserving logic for its own one gameweek — only the
"what happens next" is different.

**Alternative — the previous behaviour:** evaluate Free Hit by re-solving
with `wildcard_gameweek` set to the candidate week, i.e. treat it as a
permanent restructure. Since a Free Hit's squad can never be worth *more*
than the same squad kept permanently, this gave a correct upper bound, and
was documented as such rather than passed off as the real number. It has
been replaced now that the exact mechanism is cheap to express: no new
solver variables, just changing which past squad a handful of constraints
point at. `test_freehit_reverts_while_wildcard_persists` in `test_chips.py`
is the regression test that would have caught the old approximation: a
fixture swing that lasts past one gameweek now correctly favours Wildcard's
persistence over Free Hit's revert, where the old code scored them
identically.

---

## Still open

- **Bonus in the fast analytic path** (`points.py`, used by the optimiser) is
  still a normal approximation on expected BPS. `models/simulate.py` now
  computes bonus properly — by simulating the match and ranking real BPS
  draws with the verified tie rule — but it is not yet wired into the
  optimiser's pruning path, which needs the fast path's speed. It is available
  today for `fpl explain`-style enrichment of a shortlist.
- **Prices are static** across the horizon. Price-change prediction is Phase 4
  and is worth far less than getting the points right.
- **No calibration reliability plots yet.** The backtest (above) validates
  the projection engine's ranking and error against real 2024-25 outcomes,
  but there is no per-output reliability curve ("of the player-gameweeks we
  said were 20% to reach a haul, did 20% actually do it?") for clean sheets,
  defensive contributions, or bonus specifically. PLAN.md §6 calls for one
  per probabilistic output; only the headline MAE/Spearman/top-pick numbers
  exist today.
