"""Pairing raw buy/sell decisions into readable "sell X, buy Y" moves.

The optimiser only knows it bought three players and sold three in a
gameweek; presenting that as paired moves is what a manager actually reads,
and the pairing is possible because the squad-shape constraint guarantees
outgoing and incoming players match position for position.

Lives in `optimize/` rather than `report/` because `optimize/robustness.py`
needs it too, and `optimize` must not depend on `report` (see `DESIGN.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fpl.optimize.milp import Decision, Player


@dataclass
class TransferMove:
    out_element: int
    in_element: int
    out_name: str
    in_name: str
    out_sell_price: int
    in_buy_price: int


def pair_transfers(
    decision: Decision, players: Mapping[int, Player]
) -> list[TransferMove]:
    """Pair sales with purchases by position, which is how a manager reads them."""
    outgoing = sorted(decision.transfers_out, key=lambda e: players[e].position)
    incoming = sorted(decision.transfers_in, key=lambda e: players[e].position)

    moves: list[TransferMove] = []
    for out_element, in_element in zip(outgoing, incoming):
        out_player, in_player = players[out_element], players[in_element]
        moves.append(
            TransferMove(
                out_element=out_element,
                in_element=in_element,
                out_name=out_player.name,
                in_name=in_player.name,
                out_sell_price=out_player.sell_price,
                in_buy_price=in_player.buy_price,
            )
        )
    return moves
