"""Command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from fpl.app import App
from fpl.backtest.replay import replay_season
from fpl.data.historical import HistoricalDataUnavailable
from fpl.config import InvalidEntryId, parse_entry_id
from fpl.data.fpl_client import FPLUnavailable
from fpl.models.projection import project_gameweek
from fpl.models.scoring import POSITION_NAMES, verify_rules
from fpl.optimize.inputs import build_players, horizon_gameweeks, project_horizon
from fpl.optimize.milp import prune_pool
from fpl.optimize.chips import chip_calendar
from fpl.report.brief import build_brief, render_markdown
from fpl.state.ledger import CHIP_NAMES, Holding, TRANSFER_HIT_COST, SquadState
from fpl.state.resolve import (
    PlayerNotFound,
    candidates,
    parse_pick,
    resolve_one,
    split_picks,
)

app = typer.Typer(add_completion=False, help="Quantitative FPL advisor.")
console = Console()


def _app(offline: bool, entry: str | int | None = None) -> App:
    try:
        return App.create(offline=offline, entry_id=parse_entry_id(entry))
    except InvalidEntryId as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def _resolve_gameweek(ctx, gameweek: int | None) -> int:
    if gameweek:
        return gameweek
    current = ctx.current_event
    if current is None:
        raise typer.BadParameter("could not determine the next gameweek; pass --gw")
    return current


def _name(ctx, element_id: int) -> str:
    element = ctx.element(element_id)
    team = next((t for t in ctx.teams if int(t["id"]) == int(element["team"])), {})
    return f"{element['web_name']} ({team.get('short_name', '?')})"


@app.command()
def sync(
    entry: Optional[str] = typer.Option(
        None, help="Your numeric FPL entry (team) id."
    ),
    deep: bool = typer.Option(False, help="Also pull per-fixture minutes logs."),
    offline: bool = typer.Option(False, help="Use cached data only."),
) -> None:
    """Pull the latest data, snapshot it, and refresh squad state."""
    application = _app(offline, entry)
    try:
        report = application.sync(deep=deep)
    except FPLUnavailable as exc:
        console.print(f"[red]Sync failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]Snapshot[/green] {report['observed_at']}  "
                  f"{report['players']} players, next GW {report['next_event']}")
    console.print(f"Odds matched to {report['odds_matched']} fixtures")
    if report.get("odds_unmatched"):
        console.print(f"[yellow]Unmatched odds events:[/yellow] "
                      f"{', '.join(report['odds_unmatched'])}")
    if "squad_gameweek" in report:
        console.print(f"Squad synced at GW{report['squad_gameweek']}  "
                      f"bank £{report['bank']/10:.1f}m  "
                      f"free transfers {report['free_transfers']}")
    if report.get("deep_histories"):
        console.print(f"Deep history pulled for {report['deep_histories']} players")
    for error in report.get("errors", []):
        console.print(f"[yellow]{error}[/yellow]")

    changes = application.store.price_changes()
    if changes:
        console.print(f"\n[bold]Price changes since last snapshot[/bold]")
        for change in changes[:15]:
            direction = "[green]+[/green]" if change["delta"] > 0 else "[red]-[/red]"
            console.print(f"  {direction} element {change['element']}: "
                          f"£{change['was']/10:.1f}m -> £{change['now']/10:.1f}m")


@app.command()
def projections(
    gameweek: Optional[int] = typer.Option(None, "--gw", help="Gameweek to project."),
    top: int = typer.Option(25, help="How many players to show."),
    position: Optional[str] = typer.Option(None, help="GKP, DEF, MID or FWD."),
    max_cost: Optional[float] = typer.Option(None, help="Price ceiling in millions."),
    offline: bool = typer.Option(True, help="Use cached data only."),
    deep: bool = typer.Option(False, help="Use per-fixture minutes logs if pulled."),
) -> None:
    """Expected points per player for a gameweek."""
    application = _app(offline)
    ctx = application.context()
    gameweek = _resolve_gameweek(ctx, gameweek)
    histories = application.histories(application.bootstrap(), deep=deep) if deep else {}
    results = project_gameweek(ctx, gameweek, histories=histories)

    wanted = None
    if position:
        wanted = {v: k for k, v in POSITION_NAMES.items()}.get(position.upper())
        if wanted is None:
            raise typer.BadParameter("position must be GKP, DEF, MID or FWD")

    rows = []
    for projection in results.values():
        element = ctx.element(projection.element)
        if wanted and int(element["element_type"]) != wanted:
            continue
        if max_cost and element["now_cost"] / 10 > max_cost:
            continue
        rows.append((projection, element))
    rows.sort(key=lambda r: -r[0].expected_points)

    table = Table(title=f"Gameweek {gameweek} projections")
    for column in ("Player", "Pos", "£m", "xPts", "P(start)", "Apps", "Goals",
                   "Assists", "CS", "DefCon", "Bonus"):
        table.add_column(column, justify="right" if column != "Player" else "left")

    for projection, element in rows[:top]:
        b = projection.breakdown
        table.add_row(
            _name(ctx, projection.element),
            POSITION_NAMES.get(int(element["element_type"]), "?"),
            f"{element['now_cost']/10:.1f}",
            f"[bold]{projection.expected_points:.2f}[/bold]",
            f"{projection.minutes.p_start:.0%}",
            f"{b.get('appearance', 0):.2f}",
            f"{b.get('goals', 0):.2f}",
            f"{b.get('assists', 0):.2f}",
            f"{b.get('clean_sheet', 0):.2f}",
            f"{b.get('defensive_contribution', 0):.2f}",
            f"{b.get('bonus', 0):.2f}",
        )
    console.print(table)


@app.command()
def captain(
    gameweek: Optional[int] = typer.Option(None, "--gw"),
    top: int = typer.Option(8),
    squad_only: bool = typer.Option(True, help="Only consider players you own."),
    offline: bool = typer.Option(True),
) -> None:
    """Captaincy ranking for a gameweek."""
    application = _app(offline)
    ctx = application.context()
    gameweek = _resolve_gameweek(ctx, gameweek)
    results = project_gameweek(ctx, gameweek)

    state = application.load_state()
    pool = list(results.values())
    if squad_only and state.holdings:
        owned = set(state.element_ids)
        pool = [p for p in pool if p.element in owned]
    pool.sort(key=lambda p: -p.expected_points)

    if not pool:
        console.print("[yellow]No players to rank — sync your squad first.[/yellow]")
        raise typer.Exit(1)

    table = Table(title=f"Gameweek {gameweek} captaincy")
    table.add_column("Player")
    table.add_column("xPts", justify="right")
    table.add_column("Captained", justify="right")
    table.add_column("Gap to next", justify="right")

    best = pool[0].expected_points
    for i, projection in enumerate(pool[:top]):
        gap = projection.expected_points - (
            pool[i + 1].expected_points if i + 1 < len(pool) else projection.expected_points
        )
        table.add_row(
            _name(ctx, projection.element),
            f"{projection.expected_points:.2f}",
            f"{2 * projection.expected_points:.2f}",
            f"{gap:+.2f}" if i + 1 < len(pool) else "-",
        )
    console.print(table)

    margin = best - (pool[1].expected_points if len(pool) > 1 else best)
    if margin < 0.35:
        console.print(
            "[yellow]The top two are within a third of a point — this is close to a "
            "coin-flip, so ownership and risk appetite matter more than the model "
            "here.[/yellow]"
        )


@app.command()
def squad(offline: bool = typer.Option(True)) -> None:
    """Show your current squad, its value and what you can do this week."""
    application = _app(offline)
    state = application.load_state()
    if not state.holdings:
        console.print("[yellow]No squad recorded yet. Run `fpl sync --entry <id>`.[/yellow]")
        raise typer.Exit(1)

    ctx = application.context()
    prices = {int(e["id"]): int(e["now_cost"]) for e in ctx.elements}

    table = Table(title=f"Squad after GW{state.gameweek}")
    for column in ("Player", "Pos", "Bought", "Now", "Sells for", "Profit"):
        table.add_column(column, justify="right" if column != "Player" else "left")

    for holding in state.holdings:
        element = ctx.element(holding.element)
        now = prices.get(holding.element, holding.purchase_price)
        sells = holding.selling_price(now)
        table.add_row(
            _name(ctx, holding.element),
            POSITION_NAMES.get(int(element["element_type"]), "?"),
            f"{holding.purchase_price/10:.1f}",
            f"{now/10:.1f}",
            f"{sells/10:.1f}",
            f"{(sells - holding.purchase_price)/10:+.1f}",
        )
    console.print(table)
    console.print(
        f"Bank £{state.bank/10:.1f}m · squad sells for "
        f"£{state.squad_value(prices)/10:.1f}m · total budget "
        f"£{state.budget(prices)/10:.1f}m"
    )
    console.print(
        f"Free transfers: {state.free_transfers} · chips available: "
        f"{', '.join(state.chips_available()) or 'none'}"
    )


@app.command("set-squad")
def set_squad(
    players: list[str] = typer.Argument(
        ...,
        help="Your fifteen players, comma or space separated. Append the price "
             "you paid with @, e.g. 'Haaland@14.2'. Disambiguate a shared "
             "surname with the club, e.g. 'Silva (FUL)'.",
    ),
    bank: float = typer.Option(0.0, help="Money in the bank, in millions."),
    free_transfers: int = typer.Option(1, help="Free transfers available."),
    gameweek: Optional[int] = typer.Option(None, "--gw", help="Gameweek this reflects."),
    entry: Optional[str] = typer.Option(
        None, help="Your numeric FPL entry id, if known."
    ),
    chips_used: Optional[str] = typer.Option(
        None,
        help="Chips already played, as name:gameweek pairs, "
             "e.g. 'wildcard:8,bboost:12'.",
    ),
    offline: bool = typer.Option(True),
) -> None:
    """Record your squad by hand, without needing an FPL entry id.

    Useful when you cannot find your entry id, and as the way to seed purchase
    prices — which no endpoint reports retrospectively, so they can only be
    recovered from you.
    """
    application = _app(offline, entry)
    ctx = application.context()
    pool = candidates(ctx.elements, ctx.teams)

    names = split_picks(players)
    if len(names) != 15:
        console.print(
            f"[red]Expected 15 players, got {len(names)}.[/red] "
            "A squad is 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards."
        )
        raise typer.Exit(1)

    holdings: list[Holding] = []
    shape: dict[int, int] = {}
    clubs: dict[str, int] = {}
    for name in names:
        try:
            pick = parse_pick(name)
            found = resolve_one(pick.query, pool)
        except PlayerNotFound as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

        element = ctx.element(found.element)
        position = int(element["element_type"])
        shape[position] = shape.get(position, 0) + 1
        clubs[found.team] = clubs.get(found.team, 0) + 1
        holdings.append(
            Holding(
                element=found.element,
                purchase_price=(
                    pick.purchase_price
                    if pick.purchase_price is not None
                    else found.now_cost
                ),
                purchased_gameweek=gameweek or 0,
            )
        )

    expected_shape = {1: 2, 2: 5, 3: 5, 4: 3}
    if shape != expected_shape:
        readable = ", ".join(
            f"{POSITION_NAMES[p]} {shape.get(p, 0)}" for p in (1, 2, 3, 4)
        )
        console.print(
            f"[red]That is not a legal squad shape ({readable}).[/red] "
            "FPL requires 2 GKP, 5 DEF, 5 MID, 3 FWD."
        )
        raise typer.Exit(1)

    over = [club for club, count in clubs.items() if count > 3]
    if over:
        console.print(
            f"[red]More than three players from {', '.join(over)}.[/red] "
            "FPL allows at most three per club."
        )
        raise typer.Exit(1)

    if len({h.element for h in holdings}) != 15:
        console.print("[red]That squad contains duplicate players.[/red]")
        raise typer.Exit(1)

    used: dict[str, list[int]] = {}
    if chips_used:
        for pair in chips_used.split(","):
            chip, _, when = pair.strip().partition(":")
            chip = chip.strip()
            if chip not in CHIP_NAMES or not when.strip().isdigit():
                console.print(
                    f"[red]Could not read chip {pair!r}.[/red] Expected "
                    f"name:gameweek with a name from {', '.join(CHIP_NAMES)}."
                )
                raise typer.Exit(1)
            used.setdefault(chip, []).append(int(when))

    existing = application.load_state()
    state = SquadState(
        entry_id=parse_entry_id(entry) or existing.entry_id,
        gameweek=gameweek or ctx.current_event or 0,
        holdings=holdings,
        bank=int(round(bank * 10)),
        free_transfers=free_transfers,
        chips_used=used,
        notes="entered manually",
    )
    application.save_state(state)

    prices = {int(e["id"]): int(e["now_cost"]) for e in ctx.elements}
    console.print(
        f"[green]Squad recorded[/green] for GW{state.gameweek}: 15 players worth "
        f"£{state.squad_value(prices) / 10:.1f}m, bank £{bank:.1f}m, "
        f"{free_transfers} free transfer(s)."
    )
    assumed = sum(1 for name in names if parse_pick(name).purchase_price is None)
    if assumed:
        console.print(
            f"[yellow]Purchase price assumed to equal current price for "
            f"{assumed} of 15 players.[/yellow] FPL returns only half of a "
            "player's rise when you sell, so a wrong purchase price quietly "
            "overstates your budget. Re-run with '@' prices for anyone you "
            "bought at a different price, e.g. 'Haaland@14.2'."
        )
    console.print("Now run `fpl brief`.")


@app.command()
def explain(
    player: str = typer.Argument(..., help="Player name or element id."),
    gameweek: Optional[int] = typer.Option(None, "--gw"),
    offline: bool = typer.Option(True),
) -> None:
    """Break a projection down term by term."""
    application = _app(offline)
    ctx = application.context()
    gameweek = _resolve_gameweek(ctx, gameweek)

    matches = []
    if player.isdigit():
        matches = [int(player)]
    else:
        needle = player.lower()
        matches = [
            int(e["id"])
            for e in ctx.elements
            if needle in str(e.get("web_name", "")).lower()
            or needle in f"{e.get('first_name','')} {e.get('second_name','')}".lower()
        ]
    if not matches:
        console.print(f"[red]No player matching {player!r}[/red]")
        raise typer.Exit(1)

    results = project_gameweek(ctx, gameweek)
    for element_id in matches[:5]:
        projection = results.get(element_id)
        if projection is None:
            continue
        console.print(f"\n[bold]{_name(ctx, element_id)}[/bold] — GW{gameweek}")
        console.print(projection.explain())


@app.command()
def transfers(
    horizon: int = typer.Option(5, help="How many gameweeks to plan over."),
    max_hits: int = typer.Option(1, help="Most points hits to consider per week."),
    discount: float = typer.Option(0.85, help="Per-gameweek discount on future points."),
    wildcard: Optional[int] = typer.Option(None, help="Gameweek to play a wildcard."),
    time_limit: float = typer.Option(90.0, help="Solver time limit in seconds."),
    bootstrap: int = typer.Option(
        0,
        help="Resample the projections this many times and report how often "
             "each transfer survives (0 = off). Roughly multiplies solve time "
             "by this count, capped at a 20s solve per resample.",
    ),
    offline: bool = typer.Option(True),
) -> None:
    """Plan transfers over a rolling horizon."""
    application = _app(offline)
    ctx = application.context()
    state = application.load_state()
    if not state.holdings:
        console.print(
            "[yellow]No squad recorded. Run `fpl sync --entry <id>` first, or use "
            "`fpl brief` to see projections without one.[/yellow]"
        )
        raise typer.Exit(1)

    gameweeks = horizon_gameweeks(ctx, horizon)
    console.print(f"Projecting gameweeks {gameweeks[0]}-{gameweeks[-1]}...")
    projections = project_horizon(ctx, gameweeks)
    players = build_players(ctx, projections, state)

    brief = build_brief(
        ctx, state, players, projections, gameweeks,
        discount=discount, max_hits=max_hits,
        wildcard_gameweek=wildcard, time_limit=time_limit,
        bootstrap_resamples=bootstrap,
    )
    _render_plan(ctx, brief)


def _render_plan(ctx, brief) -> None:
    if not brief.plan.solved:
        for warning in brief.warnings:
            console.print(f"[red]{warning}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Plan from GW{brief.gameweek}")
    for column in ("GW", "Out", "In", "FT", "Hits", "Bank", "Captain", "xPts"):
        table.add_column(column, justify="right" if column != "Out" else "left")

    for decision in brief.plan.decisions:
        outs = ", ".join(_name(ctx, e) for e in decision.transfers_out) or "-"
        ins = ", ".join(_name(ctx, e) for e in decision.transfers_in) or "-"
        table.add_row(
            str(decision.gameweek), outs, ins,
            str(decision.free_transfers), str(decision.hits),
            f"{decision.bank:.1f}",
            _name(ctx, decision.captain) if decision.captain else "-",
            f"{decision.expected_points:.1f}",
        )
    console.print(table)
    console.print(
        f"Objective over the horizon: [bold]{brief.plan.objective:.2f}[/bold] "
        f"(holding instead: {brief.hold_plan.objective:.2f}) — "
        f"acting is worth [bold]{brief.gain:+.2f}[/bold] points"
    )
    if brief.bootstrap is not None:
        console.print(_stability_text(brief.moves, brief.bootstrap))


def _stability_text(moves, bootstrap) -> str:
    if bootstrap.solved == 0:
        return "[yellow]Bootstrap resamples never produced a plan.[/yellow]"
    if not moves:
        return (
            f"Held in {bootstrap.hold_frequency:.0%} of {bootstrap.solved} "
            f"bootstrap resamples."
        )
    parts = [
        f"{move.out_name} → {move.in_name}: chosen in {move.stability:.0%}"
        for move in moves
    ]
    return f"Bootstrap ({bootstrap.solved} resamples): " + "; ".join(parts)


@app.command()
def brief(
    horizon: int = typer.Option(5, help="How many gameweeks to plan over."),
    max_hits: int = typer.Option(1),
    wildcard: Optional[int] = typer.Option(None),
    time_limit: float = typer.Option(90.0),
    markdown: bool = typer.Option(False, help="Emit markdown instead of a table."),
    bootstrap: int = typer.Option(
        0,
        help="Resample the projections this many times and report how often "
             "each transfer survives (0 = off).",
    ),
    offline: bool = typer.Option(True),
) -> None:
    """The one command that matters: what to do before this deadline."""
    application = _app(offline)
    ctx = application.context()
    state = application.load_state()
    if not state.holdings:
        console.print("[yellow]No squad recorded. Run `fpl sync --entry <id>`.[/yellow]")
        raise typer.Exit(1)

    gameweeks = horizon_gameweeks(ctx, horizon)
    projections = project_horizon(ctx, gameweeks)
    players = build_players(ctx, projections, state)
    result = build_brief(
        ctx, state, players, projections, gameweeks,
        max_hits=max_hits, wildcard_gameweek=wildcard, time_limit=time_limit,
        bootstrap_resamples=bootstrap,
    )

    if markdown:
        names = {int(e["id"]): _name(ctx, int(e["id"])) for e in ctx.elements}
        print(render_markdown(result, names, projections, gameweeks))
        return

    if not result.plan.solved:
        for warning in result.warnings:
            console.print(f"[red]{warning}[/red]")
        raise typer.Exit(1)

    header = f"[bold]Gameweek {result.gameweek}[/bold]"
    if result.deadline:
        header += f" — deadline {result.deadline:%a %d %b %H:%M} UTC"
    console.print(header)

    console.print("\n[bold]Transfers[/bold]")
    if not result.moves:
        console.print(
            f"  Hold. No move clears the bar this week "
            f"(best available is worth {result.gain:+.2f} over the horizon)."
        )
    else:
        for move in result.moves:
            stability = (
                f" (chosen in {move.stability:.0%} of bootstrap resamples)"
                if move.stability is not None
                else ""
            )
            console.print(
                f"  Sell {_name(ctx, move.out_element)} "
                f"(£{move.out_sell_price/10:.1f}m) → "
                f"buy {_name(ctx, move.in_element)} (£{move.in_buy_price/10:.1f}m)"
                f"{stability}"
            )
        cost = f"{result.hits} hit(s), -{result.hits * TRANSFER_HIT_COST} pts" \
            if result.hits else "free"
        console.print(f"  Cost: {cost}. Bank afterwards £{result.bank:.1f}m.")
        console.print(
            f"  Worth [bold]{result.gain:+.2f}[/bold] points over "
            f"GW{gameweeks[0]}-{gameweeks[-1]} versus holding."
        )
        if not result.gain_is_meaningful:
            console.print(
                "  [yellow]That is inside the model's own noise — holding the "
                "transfer is defensible.[/yellow]"
            )

    console.print("\n[bold]Starting eleven[/bold]")
    ranked = sorted(
        result.starting, key=lambda e: -projections[result.gameweek][e].expected_points
    )
    for element in ranked:
        marker = ""
        if element == result.captain:
            marker = " (C)"
        elif element == result.vice:
            marker = " (V)"
        console.print(
            f"  {_name(ctx, element)}{marker}: "
            f"{projections[result.gameweek][element].expected_points:.2f}"
        )
    console.print("[bold]Bench[/bold] " + ", ".join(
        f"{_name(ctx, e)} ({projections[result.gameweek][e].expected_points:.2f})"
        for e in sorted(result.bench,
                        key=lambda e: -projections[result.gameweek][e].expected_points)
    ))

    console.print("\n[bold]Captain[/bold]")
    if result.captain:
        console.print(
            f"  {_name(ctx, result.captain)} — "
            f"{projections[result.gameweek][result.captain].expected_points:.2f} xPts, "
            f"{margin_text(result.captain_margin)}"
        )

    console.print(
        f"\n[bold]Chips available[/bold]: {', '.join(result.chips_available) or 'none'}"
        f" · free transfers after this week: {result.free_transfers}"
    )

    if result.warnings:
        console.print("\n[bold]Watch[/bold]")
        for warning in result.warnings:
            console.print(f"  [yellow]{warning}[/yellow]")


def margin_text(margin: float) -> str:
    if margin < 0.35:
        return f"only {margin:+.2f} clear of the next option — effectively a coin-flip"
    return f"{margin:+.2f} clear of the next option"


@app.command()
def chips(
    horizon: int = typer.Option(8, help="How many gameweeks to consider."),
    restructuring: bool = typer.Option(
        False,
        help="Also evaluate Wildcard and Free Hit. Much slower: these re-solve "
             "the horizon once per candidate gameweek, and a wildcard week makes "
             "every transfer free, which is by far the hardest case to solve.",
    ),
    time_limit: float = typer.Option(45.0, help="Solver time limit per solve."),
    offline: bool = typer.Option(True),
) -> None:
    """When to play each chip, measured in points rather than folklore."""
    application = _app(offline)
    ctx = application.context()
    state = application.load_state()
    if not state.holdings:
        console.print("[yellow]No squad recorded. Run `fpl sync --entry <id>`.[/yellow]")
        raise typer.Exit(1)

    gameweeks = horizon_gameweeks(ctx, horizon)
    console.print(f"Evaluating chips over GW{gameweeks[0]}-{gameweeks[-1]}...")
    projections = project_horizon(ctx, gameweeks)
    players = build_players(ctx, projections, state)
    pool = prune_pool(players, gameweeks, always_include=state.element_ids)

    baseline, advice = chip_calendar(
        pool, gameweeks, state,
        include_restructuring=restructuring, time_limit=time_limit,
    )
    if not baseline.solved:
        console.print(f"[red]Optimiser returned {baseline.status}[/red]")
        raise typer.Exit(1)

    table = Table(title="Chip calendar")
    table.add_column("Chip")
    table.add_column("Best gameweek", justify="right")
    table.add_column("Gain vs no chip", justify="right")
    for item in advice:
        table.add_row(
            item.label,
            f"GW{item.gameweek}" if item.gameweek else "-",
            f"{item.gain:+.2f}" if item.worth_playing else "-",
        )
    console.print(table)
    if not restructuring:
        console.print(
            "[dim]Wildcard and Free Hit not evaluated — pass --restructuring to "
            "include them (expect several minutes).[/dim]"
        )
    console.print(
        "Gains are measured against playing no chip at all over the same horizon. "
        "Holding a chip is worth nothing in itself — but a chip played outside "
        "this horizon may still be worth more, so widen --horizon before "
        "committing."
    )


@app.command()
def backtest(
    season: str = typer.Option(
        "2024-25", help="Historical season, e.g. 2023-24 or 2024-25."
    ),
    start_gw: int = typer.Option(
        6, help="First gameweek to score (early weeks have too little history)."
    ),
    end_gw: Optional[int] = typer.Option(None, help="Last gameweek; default the season end."),
) -> None:
    """Walk a real season forward with no lookahead, and score the projections.

    Uses github.com/vaastav/Fantasy-Premier-League, a public mirror of FPL's
    own data — not the live API, and not the FPL_ENTRY_ID/fpl.toml squad this
    tool otherwise uses. Needs network access to fetch a season the first time;
    cached indefinitely after that, since a finished season never changes.

    This validates the projection engine — minutes, team strength, rates, and
    points assembly — against real outcomes. It does not replay the optimiser's
    transfer decisions with a real squad and budget, which is a larger, separate
    piece of work.
    """
    console.print(f"Replaying {season} from GW{start_gw}"
                 f"{f' to GW{end_gw}' if end_gw else ''} (no lookahead)...")

    def progress(gameweek: int, n: int) -> None:
        console.print(f"  GW{gameweek}: {n} player-gameweeks scored")

    try:
        report = replay_season(season, start_gameweek=start_gw, end_gameweek=end_gw,
                               progress=progress)
    except HistoricalDataUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if not report.gameweeks:
        console.print("[yellow]No gameweeks were scored — check the season and range.[/yellow]")
        raise typer.Exit(1)

    console.print()
    console.print(report.summary())


@app.command()
def verify(offline: bool = typer.Option(True), players: int = typer.Option(60)) -> None:
    """Check the encoded scoring rules against real historical results."""
    application = _app(offline)
    bootstrap = application.bootstrap()
    positions = {int(e["id"]): int(e["element_type"]) for e in bootstrap["elements"]}

    candidates = sorted(
        bootstrap["elements"], key=lambda e: -int(e.get("minutes", 0) or 0)
    )[:players]

    lines = []
    for element in candidates:
        try:
            summary = application.client.element_summary(int(element["id"]))
        except FPLUnavailable:
            continue
        for row in summary.get("history", []):
            lines.append((row, positions[int(element["id"])]))

    if not lines:
        console.print(
            "[yellow]No per-fixture history available. Run `fpl sync --deep` "
            "somewhere with network access first.[/yellow]"
        )
        raise typer.Exit(1)

    report = verify_rules(lines)
    colour = "green" if report.ok else "red"
    console.print(f"[{colour}]{report.summary()}[/{colour}]")
    for example in report.examples[:5]:
        console.print(
            f"  element {example['element']} fixture {example['fixture']} "
            f"({example['position']}): predicted {example['predicted']}, "
            f"actual {example['actual']}"
        )


@app.command()
def prices(offline: bool = typer.Option(True), top: int = typer.Option(15)) -> None:
    """Recent price changes and current transfer momentum."""
    application = _app(offline)
    ctx = application.context()

    changes = application.store.price_changes()
    if changes:
        table = Table(title="Price changes since the previous snapshot")
        table.add_column("Player")
        table.add_column("Was", justify="right")
        table.add_column("Now", justify="right")
        for change in changes[:top]:
            table.add_row(
                _name(ctx, change["element"]),
                f"{change['was']/10:.1f}",
                f"{change['now']/10:.1f}",
            )
        console.print(table)
    else:
        console.print("No price changes recorded yet (needs two snapshots).")

    momentum = application.store.transfer_momentum(top=top)
    if momentum:
        table = Table(title="Transfer momentum this gameweek")
        table.add_column("Player")
        table.add_column("Net transfers", justify="right")
        table.add_column("£m", justify="right")
        for row in momentum:
            table.add_row(
                _name(ctx, row["element"]),
                f"{row['net']:+,}",
                f"{row['now_cost']/10:.1f}",
            )
        console.print(table)


if __name__ == "__main__":
    app()
