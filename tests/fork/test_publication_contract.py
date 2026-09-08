"""fork: betterlcm — a summary and everything that describes it become visible together, and
the raw frontier never calls a row uncompacted that a published summary already covers
(audit p05 CP02 / CP03)."""
import sqlite3
import time

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine


def _node(session="s", **kw):
    base = dict(session_id=session, depth=0, summary="a decision\nExpand for details about: x",
                token_count=5, source_token_count=50, source_ids=[1, 2, 3],
                source_type="messages", created_at=time.time())
    base.update(kw)
    return SummaryNode(**base)


def test_a_node_and_its_sidecar_are_published_together(tmp_path):
    dag = SummaryDAG(str(tmp_path / "pub.db"))
    try:
        node_id = dag.add_node_with_meta(_node(), level=2)
        meta = dag.node_meta.read(node_id)
        assert meta["level"] == 2 and meta["index_block"]

        # a sidecar failure leaves no half-published node behind
        original = dag.node_meta.write_statement

        def explode(*a, **k):
            raise sqlite3.OperationalError("sidecar unavailable")

        dag.node_meta.write_statement = explode
        try:
            with pytest.raises(sqlite3.OperationalError):
                dag.add_node_with_meta(_node(summary="second"), level=1)
        finally:
            dag.node_meta.write_statement = original
        assert [n.node_id for n in dag.get_session_nodes("s")] == [node_id]
    finally:
        dag.close()


def test_the_frontier_catches_up_with_what_the_summaries_already_cover(tmp_path):
    """CP02/CP03: the node is written before the frontier marker, so a failure between the two
    left rows the DAG had summarised looking raw — and the next compaction summarised them a
    second time, publishing a duplicate index over the same sources."""
    cfg = LCMConfig(database_path=str(tmp_path / "frontier.db"))
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("fs", platform="cli", context_length=200_000)
        for index in range(5):
            engine._store.append("fs", {"role": "user", "content": f"m{index}"}, source="cli")
        engine._store.commit()
        rows = engine._store.get_session_messages("fs")
        covered = [row["store_id"] for row in rows[:3]]
        engine._dag.add_node_with_meta(
            _node(session="fs", source_ids=covered), level=1
        )
        # the marker never made it out (crash between the two writes)
        assert engine._lifecycle.bind_session("fs").current_frontier_store_id < max(covered)

        engine._bind_lifecycle_state("fs")
        assert engine._last_compacted_store_id == max(covered)
    finally:
        engine.shutdown()
