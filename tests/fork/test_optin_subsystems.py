"""fork: betterlcm — the opt-in subsystems are default-off, which lowers their priority but
does not make it acceptable for them to certify incomplete or misattributed evidence
(audit verify-4 #20-#24)."""
import sqlite3

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.rollup_store import RollupStore


def test_a_rollup_and_its_lineage_come_from_one_snapshot(tmp_path):
    """verify-4 #20: the row and its source ids were read in two statements, so a concurrent
    rebuild could return generation one's text with generation two's sources."""
    store = RollupStore(tmp_path / "rollups.db")
    try:
        token = store.upsert_building("day", "2026-09-08", "s")
        store.mark_ready(token, "FIRST", 5, [1, 2], "fp1")
        first = store.get_rollup("day", "2026-09-08", "s")
        assert first["summary"] == "FIRST" and first["source_node_ids"] == [1, 2]

        second_token = store.upsert_building("day", "2026-09-08", "s")
        store.mark_ready(second_token, "SECOND", 5, [3, 4], "fp2")
        second = store.get_rollup("day", "2026-09-08", "s")
        assert second["summary"] == "SECOND" and second["source_node_ids"] == [3, 4]
    finally:
        store.close()


def test_an_unavailable_summary_database_does_not_delete_a_ready_rollup(tmp_path):
    """verify-4 #20: an unreadable DAG read as "no content", and a day with no sources
    RESOLVES — which threw away a rollup that was already built."""
    from hermes_lcm import rollup_builder

    dag = SummaryDAG(str(tmp_path / "dag.db"))
    dag.close()
    with pytest.raises(rollup_builder.RollupSourcesUnavailable):
        rollup_builder._scope_frontier(dag, "s")
