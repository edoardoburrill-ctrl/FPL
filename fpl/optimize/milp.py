"""Multi-gameweek squad optimisation.

Picking this week's best transfer greedily is a well-known way to lose points:
it will happily spend a free transfer on a 0.3-point upgrade that leaves you
unable to afford the fixture swing you actually needed in three gameweeks. So the
decision is posed over a rolling horizon and solved as one integer program —
squad, starting eleven, captain, transfers and hits, all decided together.

Future gameweeks are discounted (``discount`` per gameweek). This is not a
statement about interest rates; it reflects that a projection five gameweeks out
will be revised before it is ever acted on, so a point promised then is worth
less than a point available now. The value is a tuned hyperparameter.

The nonlinear parts of FPL's rules are linearised exactly rather than
approximated:

* ``hits = max(0, transfers - free_transfers)`` becomes two inequalities, which
  are tight at the optimum because the objective penalises hits.
* ``free_transfers = min(5, remaining + 1)`` becomes two upper bounds, which are
  tight at the optimum because the objective prefers more free transfers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import pulp

from fpl.state.ledger import MAX_FREE_TRANSFERS, TRANSFER_HIT_COST

SQUAD_SIZE = 15
SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD
XI_SIZE = 11
XI_MIN = {1: 1, 2: 3, 3: 0, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3

# Probability an unused bench player is pulled into the eleven by an auto-sub.
# Calibrated at REFERENCE_APPEARANCE, the appearance chance of a fairly nailed
# player. Bench points are real but heavily discounted, and a specific bench
# player's own chance of playing at all (from the minutes model) scales this
# base rate: a fringe player is worth far less bench insurance than a nailed
# one, regardless of position.
BENCH_WEIGHT = {1: 0.06, 2: 0.13, 3: 0.13, 4: 0.13}
REFERENCE_APPEARANCE = 0.75


def _bench_weight(player: "Player", gameweek: int) -> float:
    base = BENCH_WEIGHT.get(player.position, 0.12)
    p_appear = player.appearance.get(gameweek, REFERENCE_APPEARANCE)
    return max(0.0, min(1.0, base * p_appear / REFERENCE_APPEARANCE))

# Token objective weight on holding free transfers; see the note where it is used.
FREE_TRANSFER_EPSILON = 0.01


@dataclass(frozen=True)
class Player:
    """Everything the optimiser needs to know about one player."""

    element: int
    position: int
    team: int
    buy_price: int  # tenths of a million
    sell_price: int  # what *you* would get for him, from the ledger
    points: Mapping[int, float]  # gameweek -> expected points
    appearance: Mapping[int, float] = field(default_factory=dict)  # gw -> P(plays any minutes)
    std: Mapping[int, float] = field(default_factory=dict)  # gw -> point_std, for bootstrap resampling
    name: str = ""

    def ep(self, gameweek: int) -> float:
        return float(self.points.get(gameweek, 0.0))


@dataclass
class Decision:
    """The plan for one gameweek."""

    gameweek: int
    squad: list[int]
    starting: list[int]
    captain: int | None
    vice: int | None
    transfers_in: list[int]
    transfers_out: list[int]
    free_transfers: int
    hits: int
    bank: float
    expected_points: float
    chip: str | None = None

    @property
    def transfer_count(self) -> int:
        return len(self.transfers_in)


@dataclass
class Plan:
    decisions: list[Decision]
    objective: float
    status: str

    @property
    def solved(self) -> bool:
        return self.status in ("Optimal", "Feasible")

    @property
    def proven_optimal(self) -> bool:
        return self.status == "Optimal"

    def first(self) -> Decision | None:
        return self.decisions[0] if self.decisions else None


def prune_pool(
    players: Sequence[Player],
    gameweeks: Sequence[int],
    keep: Mapping[int, int] | None = None,
    always_include: Iterable[int] = (),
) -> list[Player]:
    """Cut the pool to plausible picks before solving.

    A 700-player, five-gameweek program is a lot of binaries for a free solver,
    and the bottom two thirds of the pool are never selected in any optimal
    solution. Everyone currently owned is kept regardless, so the incumbent squad
    is always feasible and selling is always an option.
    """
    keep = keep or {1: 20, 2: 45, 3: 45, 4: 30}
    forced = set(always_include)

    ranked: dict[int, list[Player]] = {}
    for player in players:
        ranked.setdefault(player.position, []).append(player)

    def total(player: Player) -> float:
        return sum(player.ep(gw) for gw in gameweeks)

    pool: list[Player] = []
    for position, group in ranked.items():
        group.sort(key=lambda p: -total(p))
        budget = keep.get(position, 50)

        # Keeping only the highest-projecting players would leave a pool of
        # premiums that no £100m squad can afford. Reserve part of the budget for
        # the best option in each price band, so cheap enablers — the players who
        # make a premium affordable — are always available to the solver.
        bands: dict[int, list[Player]] = {}
        for player in group:
            bands.setdefault(player.buy_price // 5, []).append(player)

        per_band = max(2, budget // max(len(bands), 1))
        banded: list[Player] = []
        for members in bands.values():
            banded.extend(members[:per_band])

        selected = {p.element for p in banded}
        for player in group:
            if len(selected) >= budget:
                break
            selected.add(player.element)
        pool.extend(p for p in group if p.element in selected)

    chosen = {p.element for p in pool}
    pool.extend(p for p in players if p.element in forced and p.element not in chosen)
    return pool


def _is_legal(
    players: Sequence[Player],
    squad: Mapping,
    bank: int,
    owned: set[int],
    index: Iterable[int],
) -> bool:
    """Does the extracted solution actually obey FPL's rules?

    Used to decide whether an unproven solve is safe to act on. Cheap, and the
    only thing standing between a solver quirk and an illegal recommendation.
    """
    by_id = {p.element: p for p in players}
    for t in index:
        chosen = [
            p.element for p in players if (squad[p.element][t].value() or 0) > 0.5
        ]
        if len(chosen) != SQUAD_SIZE:
            return False

        shape: dict[int, int] = {}
        clubs: dict[int, int] = {}
        for element in chosen:
            player = by_id[element]
            shape[player.position] = shape.get(player.position, 0) + 1
            clubs[player.team] = clubs.get(player.team, 0) + 1
        if shape != SQUAD_SHAPE or max(clubs.values()) > MAX_PER_CLUB:
            return False

        if t == 0:
            # The budget is only checkable from outside the model in the first
            # gameweek, where the incoming squad and bank are both known facts.
            # Later gameweeks depend on intermediate state the model holds.
            spend = sum(by_id[e].buy_price for e in chosen if e not in owned)
            proceeds = sum(
                by_id[e].sell_price
                for e in owned
                if e not in chosen and e in by_id
            )
            if spend > bank + proceeds:
                return False
    return True


def optimise(
    players: Sequence[Player],
    gameweeks: Sequence[int],
    current_squad: Sequence[int],
    bank: int,
    free_transfers: int,
    *,
    discount: float = 0.85,
    max_hits_per_gameweek: int = 2,
    max_transfers_per_gameweek: int = 3,
    transfer_caps: Mapping[int, int] | None = None,
    wildcard_gameweek: int | None = None,
    forced_out: Iterable[int] = (),
    captain_candidates: int = 25,
    bench_boost_gameweeks: Iterable[int] | None = None,
    triple_captain_gameweeks: Iterable[int] | None = None,
    time_limit: float | None = 90.0,
    solver: pulp.LpSolver | None = None,
) -> Plan:
    """Solve for the best trajectory over ``gameweeks``.

    ``bank`` and prices are in tenths of a million, matching FPL. Prices are held
    constant across the horizon; modelling price drift is a Phase 4 refinement
    and is worth far less than getting the points right.

    ``transfer_caps`` overrides the per-gameweek transfer limit for specific
    gameweeks — setting one to zero is how the "what if I just hold?" comparison
    that quantifies a transfer's value is computed.
    """
    if not gameweeks:
        raise ValueError("need at least one gameweek to optimise")

    gameweeks = list(gameweeks)
    players = list(players)
    allowed_bench_boost = set(bench_boost_gameweeks or ())
    allowed_triple_captain = set(triple_captain_gameweeks or ())
    by_id = {p.element: p for p in players}
    owned = [e for e in current_squad if e in by_id]
    if current_squad and len(owned) != len(current_squad):
        missing = set(current_squad) - set(by_id)
        raise ValueError(f"current squad contains unknown players: {sorted(missing)}")

    forced_out = set(forced_out)
    problem = pulp.LpProblem("fpl_multi_gameweek", pulp.LpMaximize)

    ids = [p.element for p in players]
    index = range(len(gameweeks))

    squad = pulp.LpVariable.dicts("squad", (ids, index), cat="Binary")
    start = pulp.LpVariable.dicts("start", (ids, index), cat="Binary")
    buy = pulp.LpVariable.dicts("buy", (ids, index), cat="Binary")
    sell = pulp.LpVariable.dicts("sell", (ids, index), cat="Binary")

    # The armband only ever goes on a high-projection player, so giving every
    # player a captaincy binary in every gameweek is thousands of variables the
    # solver must branch on merely to rule out. Restrict the candidates to the
    # players who could plausibly be captained.
    captain_pool = {
        t: {
            p.element
            for p in sorted(players, key=lambda q: -q.ep(gameweeks[t]))[
                :captain_candidates
            ]
        }
        for t in index
    }
    captain = {
        pid: {
            t: pulp.LpVariable(f"captain_{pid}_{t}", cat="Binary")
            for t in index
            if pid in captain_pool[t]
        }
        for pid in ids
    }

    # Chips. Bench Boost and Triple Captain are decided inside the program: both
    # are products of a chip binary with a selection binary, and both linearise
    # cleanly, so "should I bench boost this week?" becomes a direct comparison
    # of optimised trajectories rather than a rule of thumb. Wildcard and Free
    # Hit restructure the problem itself and are passed in as scenarios.
    bench_boost = {
        t: pulp.LpVariable(f"bench_boost_{t}", cat="Binary")
        for t in index
        if gameweeks[t] in allowed_bench_boost
    }
    triple_captain = {
        t: pulp.LpVariable(f"triple_captain_{t}", cat="Binary")
        for t in index
        if gameweeks[t] in allowed_triple_captain
    }
    # boosted[p][t]: this bench player's points count in full this gameweek.
    boosted = {
        p.element: {
            t: pulp.LpVariable(f"boosted_{p.element}_{t}", lowBound=0, upBound=1)
            for t in bench_boost
        }
        for p in players
    }

    hits = pulp.LpVariable.dicts("hits", index, lowBound=0, cat="Integer")
    free = pulp.LpVariable.dicts(
        "free", index, lowBound=0, upBound=MAX_FREE_TRANSFERS, cat="Integer"
    )
    remaining = pulp.LpVariable.dicts(
        "remaining", index, lowBound=0, upBound=MAX_FREE_TRANSFERS, cat="Integer"
    )

    # tripled[p][t]: this captain's points count a third time this gameweek.
    tripled = {
        pid: {
            t: pulp.LpVariable(f"tripled_{pid}_{t}", lowBound=0, upBound=1)
            for t in triple_captain
            if t in captain[pid]
        }
        for pid in ids
    }

    # -- objective -------------------------------------------------------

    contributions = []
    for t, gameweek in enumerate(gameweeks):
        weight = discount**t
        for player in players:
            points = player.ep(gameweek)
            if points == 0:
                continue
            bench = _bench_weight(player, gameweek)
            pid = player.element
            contributions.append(weight * points * start[pid][t])
            if t in captain[pid]:
                contributions.append(weight * points * captain[pid][t])
            # Bench value: in the squad but not starting.
            contributions.append(
                weight * points * bench * (squad[pid][t] - start[pid][t])
            )
            if t in bench_boost:
                # A Bench Boost pays the remaining fraction of the bench.
                contributions.append(
                    weight * points * (1.0 - bench) * boosted[pid][t]
                )
            if t in triple_captain and t in captain[pid]:
                contributions.append(weight * points * tripled[pid][t])
        contributions.append(-weight * TRANSFER_HIT_COST * hits[t])
        # `free_transfers = min(5, remaining + 1)` is expressed as upper bounds,
        # which the solver only pushes against when a hit is at stake. A token
        # reward makes the bound tight in every gameweek, so the reported balance
        # is always the true one. It is far too small to buy a decision: a free
        # transfer is worth at most four points, this is worth a hundredth of one.
        contributions.append(FREE_TRANSFER_EPSILON * free[t])

    problem += pulp.lpSum(contributions)

    # -- squad structure -------------------------------------------------

    teams = {p.team for p in players}
    for t in index:
        problem += pulp.lpSum(squad[p.element][t] for p in players) == SQUAD_SIZE
        for position, count in SQUAD_SHAPE.items():
            problem += (
                pulp.lpSum(squad[p.element][t] for p in players if p.position == position)
                == count
            )
        for team in teams:
            problem += (
                pulp.lpSum(squad[p.element][t] for p in players if p.team == team)
                <= MAX_PER_CLUB
            )

        problem += pulp.lpSum(start[p.element][t] for p in players) == XI_SIZE
        for position in SQUAD_SHAPE:
            in_xi = pulp.lpSum(
                start[p.element][t] for p in players if p.position == position
            )
            problem += in_xi >= XI_MIN[position]
            problem += in_xi <= XI_MAX[position]

        problem += (
            pulp.lpSum(captain[pid][t] for pid in captain_pool[t] if t in captain[pid])
            == 1
        )
        for player in players:
            pid = player.element
            problem += start[pid][t] <= squad[pid][t]
            if t in captain[pid]:
                problem += captain[pid][t] <= start[pid][t]

    # -- transfers -------------------------------------------------------

    owned_set = set(owned)
    for t in index:
        for player in players:
            pid = player.element
            previous = (
                squad[pid][t - 1] if t > 0 else (1 if pid in owned_set else 0)
            )
            problem += squad[pid][t] - previous == buy[pid][t] - sell[pid][t]
            # Buying and selling the same player in the same gameweek is never
            # useful and would otherwise let the solver launder budget.
            problem += buy[pid][t] + sell[pid][t] <= 1

        transfers = pulp.lpSum(buy[p.element][t] for p in players)

        # Valid inequalities: the squad is always 15 players in a fixed shape, so
        # every gameweek buys exactly as many players as it sells, position by
        # position. Both are implied by the constraints above, but stating them
        # explicitly tightens the linear relaxation and cuts solve time sharply.
        if t > 0 or owned_set:
            problem += transfers == pulp.lpSum(sell[p.element][t] for p in players)
            for position in SQUAD_SHAPE:
                problem += pulp.lpSum(
                    buy[p.element][t] for p in players if p.position == position
                ) == pulp.lpSum(
                    sell[p.element][t] for p in players if p.position == position
                )

        building_from_scratch = t == 0 and not owned_set
        if building_from_scratch or (
            wildcard_gameweek is not None and gameweeks[t] == wildcard_gameweek
        ):
            # A wildcard — or picking an initial squad — makes any number of
            # transfers free, and does not consume the free transfer you were
            # holding: it survives into the following gameweek.
            problem += hits[t] == 0
            problem += transfers <= (transfer_caps or {}).get(
                gameweeks[t], SQUAD_SIZE
            )
            problem += remaining[t] == free[t]
        else:
            cap = (transfer_caps or {}).get(
                gameweeks[t], max_transfers_per_gameweek
            )
            problem += transfers <= cap
            problem += hits[t] >= transfers - free[t]
            problem += hits[t] <= max_hits_per_gameweek
            # remaining = max(0, free - transfers), tight at the optimum because
            # hits carry a cost the objective is trying to avoid.
            problem += remaining[t] >= free[t] - transfers
            problem += remaining[t] <= free[t] - transfers + hits[t]

        if t == 0:
            problem += free[t] == min(free_transfers, MAX_FREE_TRANSFERS)
        else:
            problem += free[t] <= remaining[t - 1] + 1
            problem += free[t] <= MAX_FREE_TRANSFERS

    # -- chips -----------------------------------------------------------

    for t in index:
        # FPL allows at most one chip per gameweek.
        active = []
        if t in bench_boost:
            active.append(bench_boost[t])
        if t in triple_captain:
            active.append(triple_captain[t])
        if wildcard_gameweek is not None and gameweeks[t] == wildcard_gameweek:
            active.append(1)
        if len(active) > 1:
            problem += pulp.lpSum(active) <= 1

        for player in players:
            pid = player.element
            if t in bench_boost:
                # boosted = bench_boost AND (in squad, not starting)
                problem += boosted[pid][t] <= bench_boost[t]
                problem += boosted[pid][t] <= squad[pid][t] - start[pid][t]
            if t in triple_captain and t in captain[pid]:
                problem += tripled[pid][t] <= triple_captain[t]
                problem += tripled[pid][t] <= captain[pid][t]

        # Counting constraints instead of the usual big-M lower bounds. The
        # squad is always 15 and the eleven is always 11, so a Bench Boost
        # always boosts exactly four players and a Triple Captain always
        # triples exactly one. Stating it that way keeps the relaxation from
        # spreading a fraction of a chip across every gameweek at once, which
        # is what turns this from a tractable program into an intractable one.
        if t in bench_boost:
            problem += (
                pulp.lpSum(boosted[p.element][t] for p in players)
                == (SQUAD_SIZE - XI_SIZE) * bench_boost[t]
            )
        if t in triple_captain:
            problem += (
                pulp.lpSum(
                    tripled[pid][t] for pid in captain_pool[t] if t in tripled[pid]
                )
                == triple_captain[t]
            )

    # Each chip is playable at most once across the horizon given to us; the
    # caller decides which gameweeks are eligible, which is where the two-set,
    # half-season structure is enforced.
    if bench_boost:
        problem += pulp.lpSum(bench_boost.values()) <= 1
    if triple_captain:
        problem += pulp.lpSum(triple_captain.values()) <= 1

    for element in forced_out:
        if element in by_id:
            problem += squad[element][0] == 0

    # -- budget ----------------------------------------------------------

    for t in index:
        spend = pulp.lpSum(p.buy_price * buy[p.element][t] for p in players)
        # Selling an incumbent returns the ledger's selling price; selling a
        # player bought during the plan returns what was just paid for him.
        proceeds = pulp.lpSum(
            (p.sell_price if p.element in owned_set else p.buy_price)
            * sell[p.element][t]
            for p in players
        )
        prior = pulp.lpSum(
            pulp.lpSum(p.buy_price * buy[p.element][u] for p in players)
            - pulp.lpSum(
                (p.sell_price if p.element in owned_set else p.buy_price)
                * sell[p.element][u]
                for p in players
            )
            for u in range(t)
        )
        problem += bank - prior - spend + proceeds >= 0

    # -- solve -----------------------------------------------------------

    solver = solver or pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    problem.solve(solver)
    status = pulp.LpStatus[problem.status]

    # A time-limited solve can stop with a good incumbent rather than a proven
    # optimum, and that plan is still worth acting on. But an infeasible solve
    # also leaves values on the variables — relaxation values that violate the
    # budget — so accepting anything that merely *looks* like a squad would emit
    # illegal advice. Only an unproven-but-complete solve is promoted, and only
    # after the resulting squad is checked against the rules it must obey.
    if problem.status == pulp.LpStatusNotSolved and _is_legal(
        players, squad, bank, owned_set, index
    ):
        status = "Feasible"

    decisions: list[Decision] = []
    if status in ("Optimal", "Feasible"):
        running_bank = float(bank)
        for t, gameweek in enumerate(gameweeks):
            chosen = [p.element for p in players if squad[p.element][t].value() > 0.5]
            starting = [p.element for p in players if start[p.element][t].value() > 0.5]
            captains = [
                pid
                for pid in captain_pool[t]
                if t in captain[pid] and (captain[pid][t].value() or 0) > 0.5
            ]
            bought = [p.element for p in players if buy[p.element][t].value() > 0.5]
            sold = [p.element for p in players if sell[p.element][t].value() > 0.5]

            running_bank -= sum(by_id[e].buy_price for e in bought)
            running_bank += sum(
                (by_id[e].sell_price if e in owned_set else by_id[e].buy_price)
                for e in sold
            )

            starters = sorted(starting, key=lambda e: -by_id[e].ep(gameweek))
            skipper = captains[0] if captains else (starters[0] if starters else None)
            vice = next((e for e in starters if e != skipper), None)

            chip = None
            if t in bench_boost and (bench_boost[t].value() or 0) > 0.5:
                chip = "bboost"
            elif t in triple_captain and (triple_captain[t].value() or 0) > 0.5:
                chip = "3xc"
            elif wildcard_gameweek is not None and gameweek == wildcard_gameweek:
                chip = "wildcard"

            points = sum(by_id[e].ep(gameweek) for e in starting)
            points += by_id[skipper].ep(gameweek) if skipper else 0.0
            if chip == "3xc" and skipper:
                points += by_id[skipper].ep(gameweek)
            if chip == "bboost":
                points += sum(
                    by_id[e].ep(gameweek) for e in chosen if e not in starting
                )
            points -= TRANSFER_HIT_COST * int(round(hits[t].value() or 0))

            decisions.append(
                Decision(
                    gameweek=gameweek,
                    squad=chosen,
                    starting=starting,
                    captain=skipper,
                    vice=vice,
                    transfers_in=bought,
                    transfers_out=sold,
                    free_transfers=int(round(free[t].value() or 0)),
                    hits=int(round(hits[t].value() or 0)),
                    bank=running_bank / 10.0,
                    expected_points=points,
                    chip=chip,
                )
            )

    return Plan(
        decisions=decisions,
        objective=float(pulp.value(problem.objective) or 0.0),
        status=status,
    )
