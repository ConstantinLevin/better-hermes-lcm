"""fork: betterlcm — sidecar table ``lcm_node_meta`` (escalation level + index block).

``summary_nodes`` rows decode positionally, the v5 shape classifier fails closed on any
unregistered core column and the rollup trigger SQL is byte-compared, so per-node data the
fork adds lives in its own feature table rather than in a new core column:

    lcm_node_meta(node_id PK, level, index_block, updated_at)

- ``level``: which summariser route produced the node (1 = L1 prose, 2 = L2 bullets,
  0 = deterministic marker such as a rotate marker). Rendered in the node header so a
  reader knows an L2 node is the thinner form.
- ``index_block``: the whole "Expand for details about: …" block (bounded), not just its
  first line. ``SummaryNode.expand_hint`` stays single-line because it is emitted inline
  and as a flat JSON field in several tool paths.

The table is created idempotently from the DAG's own init (recorded as the named
``betterlcm_node_meta_v1`` migration step) and its prefix ``lcm_node`` is registered with
the classifier so a base-build check treats it as a known feature family. Rows cascade
with node deletes through ``SummaryDAG.delete_node_batch``.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import Dict, Iterable, Optional

from .db_bootstrap import mark_migration_step_complete

NODE_META_TABLE = "lcm_node_meta"
MIGRATION_STEP = "betterlcm_node_meta_v1"
INDEX_BLOCK_MARKER = "Expand for details about:"
INDEX_BLOCK_MAX_CHARS = 1600  # historical: the cut this fork removed (see extract_index_block)

LEVEL_MARKER = 0
LEVEL_L1 = 1
LEVEL_L2 = 2

LEVEL_LABELS = {
    LEVEL_MARKER: "deterministic marker",
    LEVEL_L1: "",
    LEVEL_L2: "L2 bullet summary",
}


def ensure_node_meta_table(conn: sqlite3.Connection) -> None:
    """Idempotent, additive; safe to run concurrently from several processes."""
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {NODE_META_TABLE} (
            node_id INTEGER PRIMARY KEY,
            level INTEGER NOT NULL DEFAULT 1,
            index_block TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL DEFAULT 0
        );
        """
    )
    mark_migration_step_complete(conn, MIGRATION_STEP)


def extract_index_block(summary: str) -> str:
    """Everything after the LAST ``Expand for details about:`` marker, whitespace-normalised
    per line. Empty when the summary carries no marker.

    fork: betterlcm — this used to be cut at 1,600 characters, which sliced the index in the
    middle of a topic: a 200-topic block ended partway through topic 84, and the tools that
    surface the sidecar showed the cut copy with no continuation. Cutting the index is exactly
    the loss this fork exists to remove, and the block is bounded by the summary that contains
    it, so it is stored whole.
    """
    text = str(summary or "")
    idx = text.rfind(INDEX_BLOCK_MARKER)
    if idx < 0:
        return ""
    block = text[idx + len(INDEX_BLOCK_MARKER):]
    lines = [" ".join(line.split()) for line in block.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


class NodeMetaStore:
    """Thin accessor over the sidecar; shares the DAG connection and lock."""

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = conn
        self._lock = lock

    def write(self, node_id: int, *, level: int, summary: str = "", index_block: Optional[str] = None) -> None:
        block = extract_index_block(summary) if index_block is None else str(index_block)
        with self._lock:
            self._conn.execute(
                f"""INSERT INTO {NODE_META_TABLE}(node_id, level, index_block, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        level = excluded.level,
                        index_block = excluded.index_block,
                        updated_at = excluded.updated_at""",
                (int(node_id), int(level), block, time.time()),
            )
            self._conn.commit()

    def write_statement(self, node_id: int, *, level: int, summary: str = "",
                        index_block: Optional[str] = None) -> None:
        """fork: betterlcm — the same write, WITHOUT its own commit.

        For callers that publish the node and its sidecar in one transaction, so a summary can
        never become visible without the level and index block that describe it (audit p05
        CP03). The caller owns the lock and the commit.
        """
        block = extract_index_block(summary) if index_block is None else str(index_block)
        self._conn.execute(
            f"""INSERT INTO {NODE_META_TABLE}(node_id, level, index_block, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    level = excluded.level,
                    index_block = excluded.index_block,
                    updated_at = excluded.updated_at""",
            (int(node_id), int(level), block, time.time()),
        )

    def read_many(self, node_ids: Iterable[int]) -> Dict[int, Dict[str, object]]:
        ids = sorted({int(node_id) for node_id in node_ids})
        if not ids:
            return {}
        out: Dict[int, Dict[str, object]] = {}
        with self._lock:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = self._conn.execute(
                    f"SELECT node_id, level, index_block FROM {NODE_META_TABLE} "
                    f"WHERE node_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for node_id, level, index_block in rows:
                    out[int(node_id)] = {"level": int(level), "index_block": str(index_block or "")}
        return out

    def read(self, node_id: int) -> Optional[Dict[str, object]]:
        return self.read_many([node_id]).get(int(node_id))

    @staticmethod
    def delete_many(conn: sqlite3.Connection, node_ids: Iterable[int]) -> None:
        """Cascade helper for ``SummaryDAG.delete_node_batch`` (caller owns the txn)."""
        ids = [int(node_id) for node_id in node_ids]
        if not ids:
            return
        # fork: betterlcm — batched under SQLite's bound-variable ceiling (verify-3 p04 ST5)
        for start in range(0, len(ids), 900):
            chunk = ids[start:start + 900]
            placeholders = ",".join("?" for _ in chunk)
            try:
                conn.execute(
                    f"DELETE FROM {NODE_META_TABLE} WHERE node_id IN ({placeholders})", chunk
                )
            except sqlite3.OperationalError:
                # Table absent: a connection that bypassed the DAG bootstrap. Nothing to cascade.
                return


def level_header_tag(level: Optional[int]) -> str:
    """`` [L2 bullet summary]`` style tag placed AFTER the assembly node header (empty for L1).

    It must follow the closing bracket: the engine recognises replayed summary scaffolds
    with ``[<label> Summary (d<n>, node <id>)]`` and that shape has to stay intact.
    """
    if level is None:
        return ""
    label = LEVEL_LABELS.get(int(level), f"level {int(level)}")
    return f" [{label}]" if label else ""
