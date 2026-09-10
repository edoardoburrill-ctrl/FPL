"""Turning an optimiser Decision into manager-readable transfer moves.

Split out from `report/brief.py` so `optimize/robustness.py` can use it
without a report -> optimize -> report import cycle: `report` depends on
`optimize`, never the other way around.
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
    """Pair sales with purchases by position, which is how a manager reads them.

    The optimiser only knows it bought three players and sold three; presenting
    that as "sell X, buy Y" needs the positional pairing that the squad-shape
    constraint guarantees exists.
    """
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
