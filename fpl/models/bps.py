"""The Bonus Points System: per-action scores, and how bonus is awarded from them.

Two very different confidence levels live in this module, and the code says so
at every constant.

**Bonus allocation is exact.** FPL's tie rule is documented and deterministic:
players are ranked by BPS using standard competition ranking ("1224" — a tie for
1st means both take rank 1 and the next player is rank 3, not rank 2), and rank
1/2/3 pay 3/2/1. `award_bonus` implements this precisely and is tested against
the worked examples FPL themselves publish.

**The BPS table itself is not fully verifiable from anywhere this codebase can
reach.** FPL does not publish a machine-readable table. A handful of 2026/27
values are confirmed from the official rules-change article; the rest are
long-stable values carried from seasons of public observation. Historical
archives that do exist (`data/historical.py`) predate both the defensive-
contribution scoring category and this season's BPS changes, and lack the raw
action counts (clearances, blocks, interceptions, tackles) needed to check the
defensive terms even in principle.

`partial_bps` was originally built to be a strict lower bound on the theory
that every omitted term (CBI, goalline clearances) only adds to the true
figure. Checking it against two real seasons of ground truth disproved that:
fouls conceded is also unobserved here, and it subtracts, so the "bound" isn't
one — real residuals run from −20 to well above what any plausible number of
missed fouls explains. `verify_plausibility` reports the residual distribution
instead of asserting a direction, which is what the evidence actually supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fpl.models.scoring import DEF, FWD, GK, MID


def _get(stats: Mapping, *names: str) -> float:
    for name in names:
        value = stats.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


@dataclass(frozen=True)
class BPSTable:
    """Per-action BPS values for the current season.

    Each field below is commented with its source. "Confirmed 2026/27" means it
    was named explicitly in Premier League's own rules-change article for this
    season. "Long-stable" means it has not moved across many seasons of public
    observation, but FPL has never published it, so it is not falsifiable the
    way the scoring table is.
    """

    # Confirmed 2026/27 (premierleague.com, "Changes to Bonus Points System").
    goal_by_position: Mapping[int, int] = None  # set in __post_init__
    goal_penalty: int = 12  # any position, if it was a penalty (unflagged in most data)
    cbi_per_point: int = 3  # 1 BPS per 3 clearances/blocks/interceptions (was 2)
    save_flat: int = 2  # flat 2 BPS per save (was inside/outside-box split)
    save_big_chance_bonus: int = 1  # +1 for saving a shot from a big chance
    tackled_penalty: int = 0  # removed for 2026/27 (was -1)

    # Long-stable, unconfirmed for this specific season.
    assist: int = 9
    clean_sheet_def_gk: int = 12
    clean_sheet_mid: int = 6
    tackle_won: int = 2
    yellow_card: int = -3
    red_card: int = -9
    own_goal: int = -6
    penalty_miss: int = -6
    penalty_save: int = 8  # additional, on top of the save itself
    goalline_clearance: int = 9  # a 2025/26 change; not reconfirmed for 2026/27
    appearance_short: int = 3  # under 60 minutes, any appearance
    appearance_long: int = 6  # 60+ minutes

    def __post_init__(self) -> None:
        if self.goal_by_position is None:
            object.__setattr__(
                self, "goal_by_position", {GK: 12, DEF: 12, MID: 18, FWD: 24}
            )


DEFAULT_TABLE = BPSTable()

# Components of `partial_bps` that are, at minimum, checkable in principle once
# defensive-action counts are available from a live sync. Excluded entirely:
# goalline clearances (no reliable signal in any data source available to this
# codebase) and the CBI term (no historical counts, and the coefficient itself
# changed this season).
PARTIAL_COMPONENTS = (
    "appearance", "goals", "assists", "clean_sheet", "saves", "penalty_save",
    "penalty_miss", "cards", "own_goal",
)


def partial_bps(
    stats: Mapping, element_type: int, table: BPSTable = DEFAULT_TABLE
) -> int:
    """A deliberately incomplete BPS reconstruction.

    Omits clearances/blocks/interceptions, tackles, and goalline clearances —
    terms this codebase cannot currently verify. **This was originally designed
    to be a strict lower bound** ("every omitted term is additive, so the
    partial figure can only undershoot"), on the assumption that CBI was the
    only thing missing. Checking against two real seasons of ground truth
    disproved that: real matches show partial figures exceeding the true BPS by
    up to 20, because fouls conceded is also an unobserved term, and it
    subtracts. See `verify_plausibility` for what can actually be concluded
    from a comparison against real data, and `DECISIONS.md` for the full story.
    """
    minutes = _get(stats, "minutes")
    if minutes <= 0:
        return 0

    total = table.appearance_long if minutes >= 60 else table.appearance_short
    total += int(_get(stats, "goals_scored")) * table.goal_by_position.get(element_type, 0)
    total += int(_get(stats, "assists")) * table.assist

    if _get(stats, "clean_sheets") > 0:
        total += table.clean_sheet_mid if element_type == MID else (
            table.clean_sheet_def_gk if element_type in (GK, DEF) else 0
        )

    if element_type == GK:
        total += int(_get(stats, "saves")) * table.save_flat
        total += int(_get(stats, "penalties_saved")) * (
            table.save_flat + table.penalty_save
        )

    total += int(_get(stats, "penalties_missed")) * table.penalty_miss
    total += int(_get(stats, "yellow_cards")) * table.yellow_card
    total += int(_get(stats, "red_cards")) * table.red_card
    total += int(_get(stats, "own_goals")) * table.own_goal
    return total


@dataclass
class PlausibilityReport:
    """Distribution of (real BPS − partial BPS) across real stat lines.

    Not a pass/fail check. `partial_bps` omits terms that can push the true
    figure in *either* direction relative to our estimate — CBI and goalline
    clearances add, fouls conceded subtracts — so a nonzero residual is
    expected and uninformative on its own. What is informative is an *extreme*
    residual, since a handful of missed fouls or CBI actions plausibly explain
    a swing of a few points but not of twenty.
    """

    residuals: list[int]
    extreme: list[dict]  # residual magnitude beyond `extreme_threshold`

    @property
    def checked(self) -> int:
        return len(self.residuals)

    def summary(self) -> str:
        if not self.residuals:
            return "no stat lines with a real bps field to check"
        import statistics

        return (
            f"{self.checked} stat lines; residual (real − partial) "
            f"median={statistics.median(self.residuals):.0f}, "
            f"{len(self.extreme)} extreme ({len(self.extreme) / self.checked:.2%})"
        )


def verify_plausibility(
    stat_lines,
    table: BPSTable = DEFAULT_TABLE,
    extreme_threshold: int = 15,
    max_examples: int = 10,
) -> PlausibilityReport:
    """Compare `partial_bps` against real BPS, without assuming a direction.

    ``stat_lines`` is an iterable of ``(stats, element_type)`` where ``stats``
    includes a real ``bps`` field (from a live sync's ``element-summary``, or a
    historical archive via `data/historical.py`). Residuals beyond
    ``extreme_threshold`` in either direction are collected for inspection —
    they are the closest thing available to a signal that an included constant,
    not just an omitted one, is wrong.
    """
    residuals: list[int] = []
    extreme: list[dict] = []
    for stats, element_type in stat_lines:
        real = stats.get("bps")
        if real is None or _get(stats, "minutes") <= 0:
            continue
        estimate = partial_bps(stats, element_type, table)
        residual = int(real) - estimate
        residuals.append(residual)
        if abs(residual) > extreme_threshold and len(extreme) < max_examples:
            extreme.append(
                {"element_type": element_type, "estimate": estimate,
                 "real": int(real), "residual": residual, "stats": dict(stats)}
            )
    return PlausibilityReport(residuals, extreme)


def award_bonus(bps_by_player: Mapping[int, int]) -> dict[int, int]:
    """FPL's bonus-point tie rule: standard competition ranking on BPS.

    A tie for 1st means both players take rank 1, and the next-best player is
    rank 3 (not rank 2) because two positions were consumed by the tie. Ranks
    1/2/3 pay 3/2/1; rank 4 and below pay nothing. Verified against FPL's own
    worked examples in the accompanying tests.
    """
    payouts = {1: 3, 2: 2, 3: 1}  # rank -> bonus points
    bonus = {element: 0 for element in bps_by_player}
    if not bps_by_player:
        return bonus

    values_desc = sorted(set(bps_by_player.values()), reverse=True)
    for value in values_desc:
        players = [e for e, v in bps_by_player.items() if v == value]
        rank = sum(1 for v in bps_by_player.values() if v > value) + 1
        if rank > 3:
            break
        for element in players:
            bonus[element] = payouts.get(rank, 0)
    return bonus
