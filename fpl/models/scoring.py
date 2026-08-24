"""FPL scoring rules, and an empirical check that they are correct.

The public API does not publish a machine-readable table of point values, so the
rules below are encoded by hand from the official 2026/27 rules. Hand-encoded
rules rot silently when FPL tweaks the game, so :func:`verify_rules` re-derives
every historical stat line and compares it with the ``total_points`` the API
reported. A mismatch rate above a fraction of a percent means our table is wrong,
not that the API is — and the tool says so loudly rather than quietly projecting
nonsense.

Position codes follow ``element_type`` in ``bootstrap-static``:
1 = goalkeeper, 2 = defender, 3 = midfielder, 4 = forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

GK, DEF, MID, FWD = 1, 2, 3, 4
POSITION_NAMES = {GK: "GKP", DEF: "DEF", MID: "MID", FWD: "FWD"}


@dataclass(frozen=True)
class ScoringRules:
    """Point values for one season of FPL."""

    season: str = "2026-27"

    appearance_short: int = 1  # played 1-59 minutes
    appearance_long: int = 2  # played 60+ minutes
    long_appearance_minutes: int = 60

    goal: Mapping[int, int] = field(
        default_factory=lambda: {GK: 6, DEF: 6, MID: 5, FWD: 4}
    )
    assist: int = 3
    clean_sheet: Mapping[int, int] = field(
        default_factory=lambda: {GK: 4, DEF: 4, MID: 1, FWD: 0}
    )

    saves_per_point: int = 3
    penalty_save: int = 5
    penalty_miss: int = -2

    # Goalkeepers and defenders only: -1 for every 2 conceded.
    conceded_per_penalty: int = 2
    conceded_penalty_positions: tuple[int, ...] = (GK, DEF)

    yellow_card: int = -1
    red_card: int = -3
    own_goal: int = -2

    bonus_awards: tuple[int, int, int] = (3, 2, 1)

    # Defensive contribution, introduced 2025/26 and unchanged for 2026/27.
    # Defenders count clearances/blocks/interceptions/tackles (CBIT); midfielders
    # and forwards additionally count ball recoveries (CBIRT). Capped at one award.
    defcon_threshold: Mapping[int, int] = field(
        default_factory=lambda: {DEF: 10, MID: 12, FWD: 12}
    )
    defcon_points: int = 2

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ScoringRules":
        known = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        return replace(self, **known)


def load_rules(override_path: Path | None = None) -> ScoringRules:
    """Load the default rules, optionally patched by a JSON override file.

    The override exists so a mid-season rule change can be corrected in one
    place without a code release.
    """
    rules = ScoringRules()
    if override_path and override_path.exists():
        payload = json.loads(override_path.read_text())
        # JSON object keys are strings; position-keyed maps need integer keys back.
        for key in ("goal", "clean_sheet", "defcon_threshold"):
            if key in payload and isinstance(payload[key], dict):
                payload[key] = {int(k): v for k, v in payload[key].items()}
        rules = rules.with_overrides(payload)
    return rules


def _get(stats: Mapping[str, Any], *names: str) -> int:
    """Read the first present stat key, tolerating API renames."""
    for name in names:
        value = stats.get(name)
        if value is not None:
            return int(value)
    return 0


def defensive_contribution_count(stats: Mapping[str, Any], element_type: int) -> int:
    """CBIT for defenders, CBIRT for midfielders and forwards.

    The API exposes a pre-computed ``defensive_contribution`` figure on modern
    stat lines; where it is absent (older seasons) we sum the components.
    """
    precomputed = stats.get("defensive_contribution")
    if precomputed is not None:
        return int(precomputed)

    count = (
        _get(stats, "clearances_blocks_interceptions")
        if "clearances_blocks_interceptions" in stats
        else _get(stats, "clearances") + _get(stats, "blocks") + _get(stats, "interceptions")
    )
    count += _get(stats, "tackles")
    if element_type in (MID, FWD):
        count += _get(stats, "recoveries", "ball_recoveries")
    return count


def score_stat_line(
    stats: Mapping[str, Any],
    element_type: int,
    rules: ScoringRules | None = None,
    *,
    include_bonus: bool = True,
) -> int:
    """Points scored by one player in one match, from a raw FPL stat line."""
    rules = rules or ScoringRules()
    minutes = _get(stats, "minutes")
    if minutes <= 0:
        # No appearance: cards and own goals can still only be recorded for
        # players who took the field, so a blank line is worth nothing.
        return 0

    points = (
        rules.appearance_long
        if minutes >= rules.long_appearance_minutes
        else rules.appearance_short
    )

    points += _get(stats, "goals_scored") * rules.goal.get(element_type, 0)
    points += _get(stats, "assists") * rules.assist

    if _get(stats, "clean_sheets") > 0:
        points += rules.clean_sheet.get(element_type, 0)

    if element_type == GK:
        points += _get(stats, "saves") // rules.saves_per_point
        points += _get(stats, "penalties_saved") * rules.penalty_save

    if element_type in rules.conceded_penalty_positions:
        points -= _get(stats, "goals_conceded") // rules.conceded_per_penalty

    points += _get(stats, "penalties_missed") * rules.penalty_miss
    points += _get(stats, "yellow_cards") * rules.yellow_card
    points += _get(stats, "red_cards") * rules.red_card
    points += _get(stats, "own_goals") * rules.own_goal

    threshold = rules.defcon_threshold.get(element_type)
    if threshold is not None:
        if defensive_contribution_count(stats, element_type) >= threshold:
            points += rules.defcon_points

    if include_bonus:
        points += _get(stats, "bonus")

    return points


@dataclass
class VerificationReport:
    """Result of replaying historical stat lines through :func:`score_stat_line`."""

    checked: int
    mismatched: int
    examples: list[dict]

    @property
    def mismatch_rate(self) -> float:
        return self.mismatched / self.checked if self.checked else 0.0

    @property
    def ok(self) -> bool:
        # A handful of stat lines are genuinely irreconcilable (retrospective
        # dubious-goal changes applied to points but not to the stat line), so a
        # tiny nonzero rate is expected. Anything more means a real rule error.
        return self.mismatch_rate <= 0.005

    def summary(self) -> str:
        verdict = "OK" if self.ok else "RULES LOOK WRONG"
        return (
            f"{verdict}: {self.checked - self.mismatched}/{self.checked} stat lines "
            f"reproduced ({self.mismatch_rate:.3%} mismatch)"
        )


def verify_rules(
    stat_lines: Iterable[tuple[Mapping[str, Any], int]],
    rules: ScoringRules | None = None,
    max_examples: int = 10,
) -> VerificationReport:
    """Replay ``(stat_line, element_type)`` pairs and compare to ``total_points``.

    This is the guard that keeps a hand-encoded rules table honest.
    """
    rules = rules or ScoringRules()
    checked = mismatched = 0
    examples: list[dict] = []

    for stats, element_type in stat_lines:
        actual = stats.get("total_points")
        if actual is None or int(stats.get("minutes", 0)) <= 0:
            continue
        checked += 1
        predicted = score_stat_line(stats, element_type, rules)
        if predicted != int(actual):
            mismatched += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "element": stats.get("element"),
                        "fixture": stats.get("fixture"),
                        "position": POSITION_NAMES.get(element_type, element_type),
                        "predicted": predicted,
                        "actual": int(actual),
                        "stats": dict(stats),
                    }
                )

    return VerificationReport(checked, mismatched, examples)
