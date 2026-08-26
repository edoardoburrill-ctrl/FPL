"""Time-stamped snapshots of the player pool.

Two things need history that the API does not provide. Price changes happen
overnight and are never reported retrospectively, so catching them requires
having looked before and after. And an honest backtest needs to know what was
knowable at each deadline, which means every row carries the moment it was
observed and the backtester may only read rows older than the deadline it is
replaying.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshot (
    observed_at TEXT NOT NULL,
    event       INTEGER,
    element     INTEGER NOT NULL,
    now_cost    INTEGER NOT NULL,
    selected_by REAL,
    transfers_in_event  INTEGER,
    transfers_out_event INTEGER,
    minutes     INTEGER,
    total_points INTEGER,
    status      TEXT,
    chance_of_playing INTEGER,
    PRIMARY KEY (observed_at, element)
);
CREATE INDEX IF NOT EXISTS snapshot_element_time ON snapshot (element, observed_at);

CREATE TABLE IF NOT EXISTS sync_log (
    observed_at TEXT PRIMARY KEY,
    event       INTEGER,
    players     INTEGER,
    note        TEXT
);
"""


def _num(value, default=0):
    try:
        return type(default)(value)
    except (TypeError, ValueError):
        return default


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def record_snapshot(
        self,
        elements: Sequence[Mapping],
        event: int | None = None,
        note: str = "",
        observed_at: str | None = None,
    ) -> str:
        stamp = observed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [
            (
                stamp,
                event,
                int(el["id"]),
                int(el["now_cost"]),
                _num(el.get("selected_by_percent"), 0.0),
                _num(el.get("transfers_in_event"), 0),
                _num(el.get("transfers_out_event"), 0),
                _num(el.get("minutes"), 0),
                _num(el.get("total_points"), 0),
                el.get("status"),
                el.get("chance_of_playing_next_round"),
            )
            for el in elements
        ]
        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO snapshot VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            conn.execute(
                "INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?)",
                (stamp, event, len(rows), note),
            )
            conn.commit()
        return stamp

    def snapshot_times(self, limit: int = 20) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT observed_at FROM sync_log ORDER BY observed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["observed_at"] for r in rows]

    def price_changes(self, since: str | None = None) -> list[dict]:
        """Players whose price moved between the latest two snapshots."""
        times = self.snapshot_times(limit=2)
        if len(times) < 2:
            return []
        latest, previous = times[0], since or times[1]
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT a.element AS element,
                       b.now_cost AS was,
                       a.now_cost AS now
                FROM snapshot a
                JOIN snapshot b ON a.element = b.element
                WHERE a.observed_at = ? AND b.observed_at = ? AND a.now_cost != b.now_cost
                ORDER BY (a.now_cost - b.now_cost) DESC
                """,
                (latest, previous),
            ).fetchall()
        return [
            {"element": r["element"], "was": r["was"], "now": r["now"],
             "delta": r["now"] - r["was"]}
            for r in rows
        ]

    def element_history(self, element: int, limit: int = 100) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT observed_at, now_cost, selected_by, transfers_in_event,
                          transfers_out_event
                   FROM snapshot WHERE element = ?
                   ORDER BY observed_at DESC LIMIT ?""",
                (element, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def transfer_momentum(self, top: int = 20) -> list[dict]:
        """Net transfers in the latest snapshot — the driver of price changes."""
        times = self.snapshot_times(limit=1)
        if not times:
            return []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT element, transfers_in_event - transfers_out_event AS net,
                          now_cost, selected_by
                   FROM snapshot WHERE observed_at = ?
                   ORDER BY ABS(transfers_in_event - transfers_out_event) DESC
                   LIMIT ?""",
                (times[0], top),
            ).fetchall()
        return [dict(r) for r in rows]
