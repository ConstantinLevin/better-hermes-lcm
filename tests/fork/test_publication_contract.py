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

        # a refused COMMIT must not leave the transaction open for a later, unrelated commit
        # to publish the node that failed (verify-1 on CP03)
        class _RefusingCommit:
            def __init__(self, connection):
                self._connection = connection

            def __getattr__(self, name):
                return getattr(self._connection, name)

            def commit(self):
                raise sqlite3.OperationalError("disk full")

        real_connection = dag._conn
        dag._conn = _RefusingCommit(real_connection)
        try:
            with pytest.raises(sqlite3.OperationalError):
                dag.add_node_with_meta(_node(summary="third"), level=1)
        finally:
            dag._conn = real_connection
        dag._conn.commit()
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


def test_the_frontier_never_steps_over_a_row_no_node_covers(tmp_path):
    """verify-2 regression #5: taking the MAXIMUM source id treated uncovered rows as
    compacted. A leaf may summarise a sparse selection, and an imported graph need not cover a
    prefix at all — only a proven contiguous run may advance the frontier."""
    cfg = LCMConfig(database_path=str(tmp_path / "sparse.db"))
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("sp", platform="cli", context_length=200_000)
        for index in range(5):
            engine._store.append("sp", {"role": "user", "content": f"m{index}"}, source="cli")
        engine._store.commit()
        rows = [row["store_id"] for row in engine._store.get_session_messages("sp")]
        engine._dag.add_node_with_meta(
            _node(session="sp", source_ids=[rows[1], rows[2]]), level=1
        )
        engine._bind_lifecycle_state("sp")
        assert engine._last_compacted_store_id == 0, "row 1 is covered by nothing"
    finally:
        engine.shutdown()


def test_a_short_estimate_list_never_costs_a_message(tmp_path):
    """verify-3 #30 (p04 ST2): append_many zipped messages against token estimates, so a
    caller that supplied fewer estimates than messages silently stored only that many rows."""
    from hermes_lcm.store import MessageStore
    store = MessageStore(tmp_path / "short.db")
    try:
        ids = store.append_batch(
            "s",
            [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
            token_estimates=[7],
        )
        store.commit()
        assert len(ids) == 2
        contents = [row["content"] for row in store.get_session_messages("s")]
        assert contents == ["first", "second"]
    finally:
        store.close()


def test_a_node_with_thousands_of_sources_can_still_be_read(tmp_path):
    """verify-3 #30 (p04 ST5): the exact-id read bound one variable per id, so a node
    summarising more rows than SQLite's variable ceiling raised instead of returning them."""
    from hermes_lcm.store import MessageStore
    store = MessageStore(tmp_path / "many.db")
    try:
        ids = store.append_batch(
            "s", [{"role": "user", "content": f"m{index}"} for index in range(2500)]
        )
        store.commit()
        fetched = store.get_batch(ids)
        assert len(fetched) == 2500
        assert fetched[ids[-1]]["content"] == "m2499"
    finally:
        store.close()


def test_an_interrupted_publication_cannot_be_committed_by_a_later_one(tmp_path):
    """verify-4 #4: rollback protection caught Exception only, so a KeyboardInterrupt (or a
    host cancellation) left the node insert pending and the NEXT successful publication
    committed it — without its sidecar."""
    dag = SummaryDAG(str(tmp_path / "interrupt.db"))
    try:
        original = dag.node_meta.write_statement

        def interrupt(*a, **k):
            raise KeyboardInterrupt()

        dag.node_meta.write_statement = interrupt
        try:
            with pytest.raises(KeyboardInterrupt):
                dag.add_node_with_meta(_node(summary="interrupted"), level=1)
        finally:
            dag.node_meta.write_statement = original

        good = dag.add_node_with_meta(_node(summary="the next one"), level=1)
        summaries = [n.summary for n in dag.get_session_nodes("s")]
        assert summaries == ["the next one"], summaries
        assert dag.node_meta.read(good)["level"] == 1
    finally:
        dag.close()
