# FPL Advisor — Build Plan

A quantitative decision engine for Fantasy Premier League: it projects a **full
probability distribution of points** for every player in every upcoming gameweek,
then solves a **multi-gameweek constrained optimisation** to tell you exactly which
transfers to make, who to captain, who to bench, and when to play each chip.

**Season context:** 2026/27, written at GW2 planning time. The build is sequenced so
the maths is usable for a real deadline as early as possible, then deepened.

---

## 0. Decisions and constraints

**Decisions taken (agreed up front):**

| Decision | Choice |
|---|---|
| Runtime | Local Python CLI **plus** scheduled GitHub Actions |
| Objective | **Pure expected points**, net of transfer hits, over a rolling horizon |
| Data | Official FPL API + **bookmaker odds** (the-odds-api.com free tier) |
| Squad state | Public `entry/{id}` endpoints + a local ledger file. No credentials. |

**Hard environment constraint.** The cloud sandbox this is being built in has an
egress policy that blocks `fantasy.premierleague.com`, `fbref.com`, `understat.com`
and similar. Consequences, designed around rather than fought:

1. All network access sits behind a thin client layer with an on-disk cache. Every
   model and optimiser is developed and tested against **cached JSON fixtures**, so
   the engine is fully testable without live network.
2. Live pulls run where egress is open: your machine (`fpl sync`) or the GitHub
   Action. Snapshots are committed, so a blocked environment can still reason about
   the latest committed data.
3. This is a feature, not just a workaround — snapshotting is required anyway for
   price-change tracking and for no-lookahead backtesting.

**Non-goals (deliberately):** no rank/mini-league win-probability objective (pure EV
chosen), no xG scraping from Understat/FBref in v1 (the FPL API now exposes xG, xA,
xGI, and defensive-contribution stats natively), no web dashboard.

---

## 1. Architecture

```
                  ┌────────────────────────────────────────────┐
   FPL API  ─────▶│  ingest    snapshot → SQLite + JSON cache   │
   Odds API ─────▶│            (runs where egress is open)      │
                  └───────────────────┬────────────────────────┘
                                      │
                  ┌───────────────────▼────────────────────────┐
                  │  model layer                               │
                  │   ① minutes    ② team goals (odds-anchored)│
                  │   ③ player rates  ④ points distribution    │
                  │   ⑤ BPS/bonus  ⑥ Monte Carlo match engine  │
                  └───────────────────┬────────────────────────┘
                                      │  joint samples of points
                  ┌───────────────────▼────────────────────────┐
                  │  optimiser  multi-GW MILP (HiGHS/CBC)      │
                  │   squad · XI · captain · transfers · chips │
                  └───────────────────┬────────────────────────┘
                                      │
                  ┌───────────────────▼────────────────────────┐
                  │  state ledger  ──▶  weekly brief (CLI/MD)  │
                  └────────────────────────────────────────────┘
```

Each layer is independently testable and independently useful. The model layer can
be interrogated on its own ("who are the top 20 by xPts for GW5?") before the
optimiser exists — which is what makes Phase 1 usable within days.

---

## 2. Data layer

### 2.1 Official FPL endpoints (all public, no auth)

| Endpoint | Gives us |
|---|---|
| `bootstrap-static/` | players, teams, positions, prices, ownership, form, xG/xA/xGI, DefCon stats, `game_settings` |
| `fixtures/` | full fixture list, kickoff times, FDR, blanks/doubles |
| `element-summary/{id}/` | per-player per-fixture history, minutes log, past seasons |
| `event/{gw}/live/` | live points and BPS during a gameweek |
| `entry/{id}/` | your total points, rank, bank, squad value |
| `entry/{id}/event/{gw}/picks/` | your XI, captain, chip used (public after deadline) |
| `entry/{id}/history/` | your chip usage history and per-GW record |
| `leagues-classic/{id}/standings/` | mini-league standings |

**Scoring rules are encoded once and then verified empirically.** The API does not
publish a machine-readable table of point values, so the rules have to be written
down by hand — and hand-written rules rot silently when FPL tweaks the game. The
guard is `fpl verify`, which replays every historical stat line through our scoring
function and compares the result with the `total_points` the API reported. If our
table is wrong, the mismatch rate says so immediately, and a JSON override file
corrects it without a code change.

### 2.2 Bookmaker odds

The-odds-api.com (free tier, ~500 req/month — ample at one pull per day). We take
1X2, Asian handicap, and Over/Under totals for every upcoming Premier League fixture,
plus anytime-goalscorer where available.

**De-vigging** (removing the bookmaker margin) uses **Shin's method** rather than
naive proportional normalisation — proportional de-vigging systematically overstates
longshots, which matters enormously when you're pricing a 12% clean-sheet chance.

### 2.3 Storage

SQLite (`data/fpl.db`) with time-stamped snapshot tables, plus raw JSON kept in
`data/snapshots/{ts}/`. Every row carries the timestamp it was observed at. This is
what makes strict no-lookahead backtesting possible: the backtester may only read
rows observed before the deadline it is simulating.

---

## 3. The mathematical core

This is the part that matters. Everything else is plumbing.

The goal is, for each player *p* and gameweek *t*, a **distribution** over FPL points
`P_{p,t}`, and critically the **joint** distribution across players (because
teammates' points are correlated through their team's goals — this is what makes
captaincy and bench decisions correct rather than merely plausible).

### 3.1 Minutes model

Nothing else matters if a player doesn't play. Three-outcome model per player-fixture:

  `π_out`, `π_bench`, `π_start`,  summing to 1.

- `π_out` is driven by FPL's `chance_of_playing_next_round` (null → 100%, else the
  stated %) combined with the news flag and a decay on how stale the news is.
- `π_start` from a logistic regression on exponentially-weighted recent start history,
  with the weight `w_k = exp(-λ·k)` over the last *k* appearances (λ tuned by CV),
  plus features: minutes trend, price tier, whether the club plays midweek
  (congestion → rotation), and whether the player started the most recent match.
- Shrinkage: a Beta–Binomial prior over start rate, prior strength `k₀` tuned by
  cross-validation, prior mean from the player's position and price decile. This
  stops a single start from a £4.0m squad player producing a 100% start estimate.

From this, the appearance-point distribution:

```
P(2 pts, i.e. ≥60 min) = π_start · P(60+ | start) + π_bench · P(60+ | bench)
P(1 pt,  i.e. 1–59)    = π_start · P(<60 | start) + π_bench · P(1–59 | bench)
E[minutes]             = π_start·E[m|start] + π_bench·E[m|bench]
```

`P(60+|start)` is itself a shrunk per-player rate — it is the single biggest
differentiator between a nailed-on 90-minute player and a 65-minute sub-magnet.

### 3.2 Team match model (odds-anchored)

For each fixture we need `(λ_home, λ_away)`: expected goals for each side.

**Primary source — the market.** Bookmakers aggregate injury news, lineup leaks and
motivation better than any model we could fit on 380 matches a season. Given de-vigged
`P(H), P(D), P(A)` and Over/Under 2.5 probabilities, solve for `(λ_H, λ_A)` under a
**Dixon–Coles bivariate Poisson**:

```
P(x, y) = τ_ρ(x, y) · Pois(x; λ_H) · Pois(y; λ_A)
```

where `τ_ρ` is the Dixon–Coles low-score correction (inflates 0-0/1-1, deflates
1-0/0-1 — real football is not independent Poisson at low scores). We fit `(λ_H, λ_A)`
by minimising KL divergence between model-implied and market-implied outcome
probabilities. Solved per fixture with `scipy.optimize`.

**Fallback for future gameweeks (odds not yet posted).** Weighted-MLE Dixon–Coles on
match results:

```
λ_home = exp(μ + α_i + δ_j + η_home)     λ_away = exp(μ + α_j + δ_i)
```

with attack `α`, defence `δ`, home advantage `η`, fitted by maximising a
time-decayed log-likelihood `Σ w_m · ℓ_m`, `w_m = exp(-ξ·Δt_m)`, half-life ξ ≈ 6 months.
Early season the ratings are shrunk toward last season's final ratings plus a
promoted-team prior — otherwise three matches of noise dominate.

**Blend.** `λ = ω·λ_market + (1-ω)·λ_model`, with `ω → 1` for the imminent gameweek
(odds are sharp and current) and `ω` decaying for gameweeks further out.

Team-level xG from the FPL API is used as a third, independent check and a
regulariser — sustained xG over/underperformance is informative about the *model*
side, not the market side.

### 3.3 Player attacking rates

For each player, a shrunk per-90 rate of goals and assists. Using expected rather
than actual involvement (xG/xA) because it stabilises far faster:

```
r̂_p = (n_p · r_p^obs + k · r_p^prior) / (n_p + k)        n_p = minutes / 90
```

Empirical-Bayes: `k` estimated from the cross-sectional variance of rates within a
position × price tier; `r_p^prior` is the position/tier mean blended with the
player's own prior-season rate (weighted by prior-season minutes). This is what stops
the model chasing a hat-trick from GW1.

**Fixture scaling.** Rather than using an absolute rate, convert to a *share* of the
team's output:

```
ρ_p^G = r̂_p^xG / (team xG per 90)          E[goals_{p,t}] = ρ_p^G · λ_team,t · (E[min]/90)
ρ_p^A = r̂_p^xA / (team xG per 90)          E[assists_{p,t}] = ρ_p^A · λ_team,t · (E[min]/90)
```

so a striker's projection automatically rises against Burnley and falls against
Arsenal, with no hand-tuned "fixture difficulty" fudge factor. Where anytime-scorer
odds exist, they override `ρ_p^G` directly (again the market beats us).

Penalty duties are handled explicitly: `E[pens] = P(team wins pen) × P(p is taker) ×
0.79` (the historical conversion rate), added on top of open-play xG.

### 3.4 Defensive points

- **Clean sheet** (GK/DEF 4 pts, MID 1 pt): requires 60+ minutes and zero conceded
  *while on the pitch*.
  `P(CS_p) = P(60+ min) · P(0 conceded during their minutes | λ_opp)`, taken from the
  Dixon–Coles joint distribution, not from an independent Poisson.
- **Goals conceded** (GK/DEF, −1 per 2 conceded):
  `E[penalty] = −Σ_{n≥2} floor(n/2) · P(N = n)` over the conceded distribution,
  scaled by expected share of the match played.
- **Saves** (GK, 1 pt per 3): `S ~ Poisson(shots_on_target_faced × (1 − conversion))`,
  with SoT faced modelled from opponent λ and the team's historical shot-suppression
  profile; `E[pts] = Σ floor(s/3)·P(S=s)`.
- **Defensive contribution** (2 pts at the positional threshold): a per-90 CBIT/CBIRT
  rate with the same empirical-Bayes shrinkage, converted through a Poisson (or
  negative-binomial where overdispersed) to `P(threshold reached | minutes)`. The
  thresholds and point values are **read from `game_settings`**, not hardcoded.

### 3.5 Bonus points

Bonus is where naive models leak the most accuracy, because BPS is *ranked within a
match* — it is inherently a joint, not marginal, quantity. So we model it jointly:

1. Encode the BPS-per-event table for the current season (2026/27 changed it: CBI at
   1 per 3, GK saves at 2 plus 1 for a big chance saved, no tackled-penalty), and
   hold it to the same empirical verification as the scoring table.
2. In the Monte Carlo (§3.6), simulate each player's event counts, compute their BPS,
   rank all 22+ players in the fixture, and award 3/2/1 with FPL's tie rules.
   (Until then, v1 uses a normal-approximation threshold model — monotone and
   roughly calibrated, and the weakest term in the projection.)
3. `E[bonus_p]` and the full bonus distribution fall out of the simulation.

### 3.6 Monte Carlo match engine

The unifying step. For each fixture, draw `N = 25,000` scenarios:

1. Draw the scoreline from the Dixon–Coles joint distribution.
2. Draw each player's minutes from the §3.1 model.
3. Allocate the team's goals to scorers and assisters via a multinomial over
   minutes-weighted `ρ_p` shares.
4. Draw saves, CBIT counts, cards (Poisson from per-90 rates and referee strictness).
5. Compute BPS for all players, rank, award bonus.
6. Apply the `game_settings` scoring table → a points sample for every player.

This yields **joint** samples, giving for free:

- `E[P_p]` — the projection used by the optimiser.
- `Var[P_p]`, `P(P_p ≥ 10)` (haul probability) — captaincy under the right criterion.
- Correlations between teammates — so the optimiser understands that stacking three
  Arsenal players is not three independent bets.
- The distribution of your **squad total**, not just its mean.

**Captaincy.** With a pure-EV objective the captain is `argmax E[P_p]`, and the
armband doubles it — but the simulation also reports `P(captain haul)` and the EV
gap to the second choice, so a 0.1-point gap is visibly a coin-flip rather than a
false precision.

---

## 4. The optimiser

A **multi-gameweek mixed-integer linear program**, horizon `H = 5–8` gameweeks,
solved with HiGHS (via PuLP or `highspy`) — free, and fast enough for this at a few
seconds per solve.

**Variables** for player `p`, gameweek `t`:
`x_{p,t}` in squad · `y_{p,t}` in XI · `c_{p,t}` captain · `v_{p,t}` vice ·
`b_{p,t}` bought · `s_{p,t}` sold.

**Objective:**

```
max  Σ_t γ^t [ Σ_p EP_{p,t}·(y_{p,t} + c_{p,t}) + Σ_p β_{p,t}·EP_{p,t}·(x_{p,t} − y_{p,t}) ]
     − 4 · Σ_t h_t
```

- `γ ≈ 0.85` per gameweek: future projections are less reliable and future plans get
  revised, so near-term points are worth more than distant ones. `γ` is a tuned
  hyperparameter, validated in backtest — not a guess.
- `β_{p,t}` is the bench weight: `P(this bench player is auto-subbed in)`, computed
  from the minutes model, rather than an arbitrary constant.

**Constraints:**

- Squad: 15 players, exactly 2 GK / 5 DEF / 5 MID / 3 FWD.
- Max 3 players per club.
- XI: 11 players, exactly 1 GK, ≥3 DEF, ≥1 FWD, `y_{p,t} ≤ x_{p,t}`.
- Captain/vice: `Σ_p c_{p,t} = 1`, `c_{p,t} ≤ y_{p,t}`.
- Transfer accounting: `x_{p,t} = x_{p,t−1} + b_{p,t} − s_{p,t}`.
- Budget: `bank_t = bank_{t−1} + Σ_p sell_p·s_{p,t} − Σ_p buy_p·b_{p,t} ≥ 0`, using
  true **selling prices** (FPL's sell-at-half-the-rise rule) from the ledger.
- Free transfers: `f_t = min(5, f_{t−1} − T_{t−1} + 1)`, linearised with big-M;
  hits `h_t ≥ Σ_p b_{p,t} − f_t`, `h_t ≥ 0`.
- Blanks/doubles handled naturally: `EP_{p,t}` sums over that player's fixtures in
  gameweek `t`, and is zero for a blank.

**Chips** as binary decisions inside the same program:

- **Bench Boost**: `BB_t = 1` adds the bench's full EP for that gameweek.
- **Triple Captain**: `TC_t = 1` adds a further `EP` for the captain.
- **Free Hit**: a one-gameweek unconstrained squad with `x` reverting at `t+1`.
- **Wildcard**: relaxes the hit penalty for one gameweek.
- One chip per gameweek; first set must be used by GW19; each chip once per half.

Because chips are *inside* the optimisation rather than decided by rules of thumb,
the answer to "should I bench boost this week?" is a direct comparison of two
optimised trajectories, in points.

**Chip timing uses a near-neutral discount (0.97), not the transfer planner's 0.85.**
The horizon discount exists because a transfer plan for gameweek 8 will be revised
before it is executed. That argument does not apply to a chip: the decision is a
comparison between playing it now and playing it in gameweek 8, and discounting the
later branch at 0.85⁵ would shrink a 12-point double-gameweek Bench Boost to 5,
advising you to burn chips early — exactly the mistake chip strategy exists to
avoid. A small residual discount remains, because a distant chip week can still be
disrupted by a fixture change.

**Formulation note.** Bench Boost and Triple Captain are products of a chip binary
with a selection binary. The textbook big-M linearisation is correct but disastrously
weak here: it lets the relaxation spread a fraction of a chip across every gameweek
at once, and solve times ran past fifteen minutes. Replacing the big-M lower bounds
with counting constraints — a Bench Boost boosts *exactly* four players, a Triple
Captain triples *exactly* one — is equally exact and brings the same solve back to
around twelve seconds.

**Robustness.** The MILP maximises expected points against a single point estimate.
To avoid over-fitting to model noise we also solve across bootstrap resamples of the
projections and report **how often each recommended transfer appears** — a transfer
chosen in 95% of resamples is a real edge; one chosen in 40% is a coin-flip that
should not cost you 4 points.

---

## 5. State tracking

`state/squad.json`, auto-updated on each sync, human-readable and editable:

- Current 15, with **purchase price** per player (the ledger — required for correct
  selling prices, and *not* available from any endpoint retroactively, so it must be
  maintained from the first sync onward).
- Bank, squad value, free transfers available, chips used/remaining per half.
- Gameweek history of transfers made, points scored, rank.

Free transfers are reconstructed from `entry/{id}/history` transfer counts and the
FT carry-over rule, with a manual override in the file for the rare edge case.

**Price-change model.** The scheduled Action snapshots `bootstrap-static` several
times daily. Net transfer velocity relative to ownership predicts overnight rises and
falls; the tool flags "this player rises tonight — buy now" and "your player falls
tonight — you lose 0.1 of value". Secondary to points, but free money over a season.

---

## 6. Validation — how we know it works

No model ships on vibes. Every component is measured:

| Component | Metric | Baseline it must beat |
|---|---|---|
| Minutes | Log-loss on P(60+), Brier | "started last match" |
| Team goals | RPS on match outcomes, log-loss on CS | de-vigged market (target: match it) |
| Player points | MAE / RMSE per player-GW, Spearman ρ | FPL's own `ep_next`, and season-form |
| Clean sheets | Reliability curve, calibration slope | naive FDR-based estimate |
| Bonus | MAE on E[bonus], top-3 hit rate | ICT-index ranking |

**Calibration over accuracy.** A model that says 20% and is right 20% of the time is
worth more to an optimiser than one with lower MAE but miscalibrated tails, because
the optimiser consumes probabilities. Reliability plots for every probabilistic output.

**Season replay backtest.** Re-run the full pipeline over 2024/25 and 2025/26 with a
strict information cutoff at each deadline (enforced by the observed-at timestamps),
letting the optimiser make transfers with a real starting squad and budget. Report
total points versus: (a) actual FPL top-10k average, (b) a "never transfer" hold
strategy, (c) a greedy one-gameweek-ahead optimiser. If the multi-gameweek MILP
cannot beat greedy in backtest, the horizon and `γ` are wrong and get retuned.

---

## 7. Weekly workflow

```bash
fpl sync                 # pull FPL + odds, snapshot, update state
fpl brief                # ← the one command that matters
fpl brief --gw 5         # look ahead
fpl transfers --horizon 8 --max-hits 1
fpl captain              # EV table + haul probabilities
fpl chips                # optimal chip calendar for the remaining season
fpl explain <player>     # full breakdown of a projection, term by term
fpl backtest --season 2025-26
```

`fpl brief` outputs a single page:

> **GW5 — deadline Sat 12 Sep 13:30 (2d 4h)**
> **Transfer:** Sell X (£7.4m, sells at £7.3m) → Buy Y (£7.8m). Cost: 1 FT + £0.5m.
> Gain over horizon: +4.7 pts (chosen in 93% of bootstrap resamples).
> **XI / bench** with projections. **Captain:** Z, 8.9 xPts, 34% haul chance,
> +1.2 clear of the alternative. **Chips:** hold; Bench Boost currently projects
> best at GW14 (double gameweek), +11.3 pts vs playing it now.
> **Watch:** A rises tonight (+0.1), B has a fitness flag (55% to start).

The GitHub Action runs `sync` several times daily, and posts the brief on a schedule
before each deadline.

---

## 8. Phases and sequencing

Sequenced so the **maths is usable first**, as requested. Each phase ends with
something you can actually run.

**Phase 0 — Foundations.** *(done)* Repo skeleton, FPL + odds clients with caching, SQLite
schema, snapshotting, state ledger from your team ID, test fixtures captured for
offline development. *Ends with:* `fpl sync` and `fpl squad` working.

**Phase 1 — Projection engine v1 (the priority).** *(done)* Minutes model, odds-anchored team
goals with Dixon–Coles, shrunk player rates, analytic points assembly for goals,
assists, clean sheets, conceded, saves, DefCon, and a first-pass regression bonus
model. *Ends with:* `fpl projections` and `fpl captain` — a defensible xPts table for
the next gameweek. **This is the point at which the tool starts earning points.**

**Phase 2 — Optimiser.** *(done)* Multi-gameweek MILP with transfers, hits, FT
carry-over, selling prices, bench weighting. Chips (wildcard) supported; the rest
land in Phase 4. *Ends with:* `fpl transfers` and `fpl brief`.

**Phase 3 — Rigour.** Monte Carlo match engine replacing the analytic assembly,
proper ranked-BPS bonus, correlation-aware captaincy, bootstrap-robust transfer
recommendations, calibration suite and the full season-replay backtest. Hyperparameters
(`γ`, horizon, shrinkage `k`, decay `ξ`) tuned against backtest rather than intuition.
*Ends with:* measured evidence the thing beats its baselines.

**Phase 4 — Chips, automation, polish.** *(mostly done)* Bench Boost and Triple
Captain are decided inside the MILP; Wildcard and Free Hit are evaluated by
re-solving the horizon with the chip active in each candidate gameweek. Chip
eligibility enforces the two-set, GW19-expiry structure. GitHub Actions handle
scheduled sync and pre-deadline briefs. `fpl explain` gives term-by-term
breakdowns. Still outstanding: price-change prediction, and modelling the Free
Hit's squad revert explicitly rather than bounding it with wildcard semantics.

Phases 0–1 are the urgent ones and are deliberately small enough to land quickly;
Phases 2–4 improve the edge without you being blocked from using the tool meanwhile.

---

## 9. Repository layout

```
fpl/
  cli.py                    # Typer CLI
  config.py
  data/
    fpl_client.py           # official API, cached
    odds_client.py          # the-odds-api, de-vigging
    store.py  schema.sql    # SQLite + snapshots
  models/
    minutes.py              # §3.1
    team_strength.py        # §3.2 Dixon–Coles + market solve
    devig.py                # Shin's method
    player_rates.py         # §3.3 empirical-Bayes shrinkage
    defensive.py            # §3.4
    bps.py                  # §3.5
    simulate.py             # §3.6 Monte Carlo engine
    scoring.py              # game_settings-driven scoring table
  optimize/
    milp.py                 # §4
    chips.py  bench.py
  state/
    ledger.py  squad.py     # §5
  report/
    brief.py  explain.py
  backtest/
    replay.py  metrics.py   # §6
tests/
  fixtures/                 # captured API JSON — offline dev
data/                       # snapshots, sqlite (gitignored)
.github/workflows/
  sync.yml  brief.yml
docs/PLAN.md                # this file
```

Stack: Python 3.11, `httpx`, `pandas`, `numpy`, `scipy`, `statsmodels`, `pulp`/`highspy`,
`typer`, `pytest`. No paid dependencies; the odds API free tier is sufficient.

---

## 10. What is needed from you

1. **Your FPL team ID** — recorded in `fpl.toml`. It is the number in the URL of
   your points page (`fantasy.premierleague.com/entry/XXXXXXX/event/2`), and FPL
   reissues it every season. Not strictly required: `fpl set-squad` records a
   squad by hand without one.
2. **Your current squad, bank, and free transfers** if you have already made GW2
   moves — the purchase-price ledger cannot be reconstructed retroactively from the
   API, so the sooner it is seeded, the more accurate selling prices stay all season.
3. **An odds API key** (free at the-odds-api.com) — optional for Phase 1, which
   degrades gracefully to the Dixon–Coles model, but it is the single largest
   accuracy gain available.
4. **Where you'll run it** — confirmation that you can run Python locally, since live
   FPL pulls cannot happen from the build sandbox.

## 11. Risks and how they are handled

| Risk | Mitigation |
|---|---|
| Blocked egress in the build sandbox | Cached fixtures for all development; live sync runs locally / in Actions |
| FPL changes scoring rules mid-season | `fpl verify` replays history against our table and fails loudly; JSON override fixes it without a release |
| Early-season small samples | Empirical-Bayes shrinkage to positional and prior-season priors; market-anchored team strength |
| Over-fitting to noise | Bootstrap-resampled transfer recommendations; strict no-lookahead backtest |
| Optimiser over-trusting point estimates | Report resample stability alongside every recommendation |
| Odds API rate limits | Daily pull, cached; graceful fallback to the fitted model |
| Model looks confident but is wrong | Calibration plots and backtest published before the optimiser is trusted |
