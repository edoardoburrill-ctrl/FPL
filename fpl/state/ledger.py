"""Squad state: what you own, what you paid, and what you can still do.

The purchase-price ledger is the part that cannot be recovered later. FPL's
selling-price rule gives you back only half of a player's rise (rounded down),
so the correct sale value depends on what you paid — and no public endpoint
reports historical purchase prices. Every gameweek this is not being recorded is
a gameweek of drift between what the optimiser thinks your squad is worth and
what it is actually worth.

Free transfers and chip availability are reconstructed from the entry history
where possible, with manual overrides for the cases the API cannot express.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Chips come in two sets in 2026/27: one usable up to the GW19 deadline, one
# after. Unused first-half chips are lost rather than carried over.
CHIP_NAMES = ("wildcard", "freehit", "3xc", "bboost")
FIRST_HALF_LAST_GAMEWEEK = 19

MAX_FREE_TRANSFERS = 5
TRANSFER_HIT_COST = 4


@dataclass
class Holding:
    """One player in the squad, with what he cost to buy."""

    element: int
    purchase_price: int  # tenths of a million, as FPL reports prices
    purchased_gameweek: int = 0

    def selling_price(self, now_cost: int) -> int:
        """FPL returns the purchase price plus half of any rise, rounded down."""
        if now_cost <= self.purchase_price:
            return int(now_cost)
        rise = now_cost - self.purchase_price
        return int(self.purchase_price + rise // 2)


@dataclass
class SquadState:
    entry_id: int | None = None
    gameweek: int = 0
    holdings: list[Holding] = field(default_factory=list)
    bank: int = 0  # tenths of a million
    free_transfers: int = 1
    chips_used: dict[str, list[int]] = field(default_factory=dict)
    notes: str = ""
    updated_at: str = ""

    # -- derived ---------------------------------------------------------

    @property
    def element_ids(self) -> list[int]:
        return [h.element for h in self.holdings]

    def holding(self, element: int) -> Holding | None:
        for h in self.holdings:
            if h.element == element:
                return h
        return None

    def squad_value(self, prices: Mapping[int, int]) -> int:
        """Selling value of the squad, which is what you can actually spend."""
        return sum(
            h.selling_price(int(prices.get(h.element, h.purchase_price)))
            for h in self.holdings
        )

    def budget(self, prices: Mapping[int, int]) -> int:
        return self.bank + self.squad_value(prices)

    def chips_available(self, gameweek: int | None = None) -> list[str]:
        """Chips still playable, respecting the two-set, half-season structure."""
        gameweek = gameweek or self.gameweek or 1
        half = 1 if gameweek <= FIRST_HALF_LAST_GAMEWEEK else 2
        available = []
        for chip in CHIP_NAMES:
            used_in_half = [
                gw
                for gw in self.chips_used.get(chip, [])
                if (1 if gw <= FIRST_HALF_LAST_GAMEWEEK else 2) == half
            ]
            if not used_in_half:
                available.append(chip)
        return available

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "SquadState":
        holdings = [Holding(**h) for h in payload.get("holdings", [])]
        return cls(
            entry_id=payload.get("entry_id"),
            gameweek=int(payload.get("gameweek", 0)),
            holdings=holdings,
            bank=int(payload.get("bank", 0)),
            free_transfers=int(payload.get("free_transfers", 1)),
            chips_used={k: list(v) for k, v in payload.get("chips_used", {}).items()},
            notes=payload.get("notes", ""),
            updated_at=payload.get("updated_at", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "SquadState":
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))


def free_transfers_after(previous: int, transfers_made: int) -> int:
    """FPL's carry-over rule: unused transfers accumulate, capped at five."""
    remaining = max(0, previous - transfers_made)
    return min(MAX_FREE_TRANSFERS, remaining + 1)


def transfer_cost(transfers_made: int, free_transfers: int) -> int:
    return max(0, transfers_made - free_transfers) * TRANSFER_HIT_COST


def reconstruct_free_transfers(history: Mapping) -> int:
    """Rebuild the current free-transfer count from the entry's history.

    The API reports transfers made and points hits per gameweek but never the
    free-transfer balance itself, so it has to be replayed forward. A wildcard or
    free hit gameweek does not consume free transfers, so those are skipped.
    """
    free = 1
    chip_gameweeks = {
        int(play["event"]): play.get("name")
        for play in history.get("chips", []) or []
    }
    for row in history.get("current", []) or []:
        gameweek = int(row.get("event", 0))
        chip = chip_gameweeks.get(gameweek)
        if chip in ("wildcard", "freehit"):
            free = min(MAX_FREE_TRANSFERS, free + 1)
            continue
        free = free_transfers_after(free, int(row.get("event_transfers", 0) or 0))
    return free


def chips_used_from_history(history: Mapping) -> dict[str, list[int]]:
    used: dict[str, list[int]] = {}
    for play in history.get("chips", []) or []:
        name = play.get("name")
        if name:
            used.setdefault(name, []).append(int(play.get("event", 0)))
    return used


def state_from_api(
    entry: Mapping,
    picks: Mapping,
    history: Mapping,
    existing: SquadState | None = None,
    prices: Mapping[int, int] | None = None,
) -> SquadState:
    """Build squad state from the public endpoints.

    Purchase prices are preserved from ``existing`` wherever the player was
    already held; new players are recorded at their current price, which is right
    if the sync happens promptly after a transfer and drifts if it does not.
    """
    prices = prices or {}
    gameweek = int(picks.get("entry_history", {}).get("event", 0) or 0)
    previous = {h.element: h for h in (existing.holdings if existing else [])}

    holdings: list[Holding] = []
    for pick in picks.get("picks", []) or []:
        element = int(pick["element"])
        prior = previous.get(element)
        if prior is not None:
            holdings.append(prior)
        else:
            holdings.append(
                Holding(
                    element=element,
                    purchase_price=int(prices.get(element, 0)),
                    purchased_gameweek=gameweek,
                )
            )

    entry_history = picks.get("entry_history", {}) or {}
    return SquadState(
        entry_id=int(entry.get("id")) if entry.get("id") else (existing.entry_id if existing else None),
        gameweek=gameweek,
        holdings=holdings,
        bank=int(entry_history.get("bank", entry.get("last_deadline_bank", 0)) or 0),
        free_transfers=reconstruct_free_transfers(history),
        chips_used=chips_used_from_history(history),
        notes=existing.notes if existing else "",
    )
