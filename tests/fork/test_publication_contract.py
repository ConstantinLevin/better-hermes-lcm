"""fork: better-hermes-lcm — a summary and everything that describes it become visible together, and
the raw frontier never calls a row uncompacted that a published summary already covers
(audit p05 CP02 / CP03)."""
import sqlite3
import time

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.errors import SummaryUnavailableError
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine


def _node(session="s", **kw):
    base = dict(session_id=session, depth=0, summary="a decision\nExpand for details about: x",
                token_count=5, source_token_count=50, source_ids=[1, 2, 3],
                source_type="messages", created_at=time.time())
    base.update(kw)
    return SummaryNode(**base)


def test_the_frontier_is_a_sql_predicate_not_a_page_of_nodes(tmp_path):
    """B7: the engine built the frontier by loading nodes with a limit and filtering in Python,
    so a session past that limit computed its frontier from a truncated set with nothing to say
    so. The SQL predicate is unbounded and agrees with the token sum over the same set."""
    dag = SummaryDAG(str(tmp_path / "frontier.db"))
    try:
        leaf_ids = [dag.add_node(_node(token_count=7)) for _ in range(1200)]
        # get_session_nodes' own default page is 1000 — the frontier must not inherit it
        assert len(dag.get_session_nodes("s")) == 1000
        frontier = dag.get_frontier_nodes("s")
        assert len(frontier) == 1200
        assert dag.get_frontier_token_total("s") == sum(n.token_count for n in frontier)

        # a condensed parent takes its children off the frontier, and only those
        dag.add_node(_node(depth=1, source_ids=leaf_ids[:500], source_type="nodes"))
        frontier = dag.get_frontier_nodes("s")
        assert len(frontier) == 701
        assert {n.node_id for n in frontier}.isdisjoint(set(leaf_ids[:500]))
        assert dag.get_frontier_token_total("s") == sum(n.token_count for n in frontier)
    finally:
        dag.close()


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


def test_a_refused_insert_leaves_no_open_transaction(tmp_path):
    """round-2 verify-3 #3: the INSERT itself sat outside the rollback protection, so a
    statement the database refused still left the transaction it had opened open — holding the
    write lock, and letting the next unrelated commit publish whatever was pending in it."""
    dag = SummaryDAG(str(tmp_path / "refused.db"))
    try:
        dag._conn.execute(
            "CREATE TRIGGER refuse BEFORE INSERT ON summary_nodes "
            "WHEN NEW.summary LIKE 'refused%' BEGIN SELECT RAISE(FAIL, 'no'); END"
        )
        with pytest.raises(sqlite3.IntegrityError):
            dag.add_node_with_meta(_node(summary="refused one"), level=1)
        assert dag._conn.in_transaction is False, "the refused insert held the transaction open"

        with pytest.raises(sqlite3.IntegrityError):
            dag.add_node(_node(summary="refused two"))
        assert dag._conn.in_transaction is False

        dag._conn.commit()  # an unrelated commit must publish nothing
        assert dag.get_session_nodes("s") == []
        good = dag.add_node_with_meta(_node(summary="the good one"), level=1)
        assert [n.summary for n in dag.get_session_nodes("s")] == ["the good one"]
        assert dag.node_meta.read(good)["level"] == 1
    finally:
        dag.close()


def test_a_session_change_during_summarisation_does_not_publish_under_the_new_session(tmp_path):
    """round-2 verify-4 #3 (RS02): publication read self._session_id AFTER the summariser
    returned, so changing sessions inside the model call published the OLD session's content as
    a node in the NEW one — wrong provenance, and a frontier advanced over rows nothing covers."""
    from hermes_lcm import escalation
    cfg = LCMConfig(database_path=str(tmp_path / "fence.db"), condensation_fanin=2,
                    incremental_max_depth=2)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("s", platform="cli", context_length=200_000)
        children = [
            e._dag.add_node_with_meta(SummaryNode(
                session_id="s", depth=0, summary=f"child {index} decided X",
                token_count=40, source_token_count=100, source_ids=[index + 1],
                source_type="messages", created_at=time.time() + index), level=1)
            for index in range(2)
        ]
        nodes = [e._dag.get_node(node_id) for node_id in children]

        def rebind(*args, **kwargs):
            e.on_session_start("new_session", platform="cli", context_length=200_000)
            return "merged\nExpand for details about: merged"

        original = escalation._call_llm_for_summary
        escalation._call_llm_for_summary = rebind
        try:
            with pytest.raises(SummaryUnavailableError):
                e._condense_summary_nodes(nodes)
        finally:
            escalation._call_llm_for_summary = original

        assert e._dag.get_session_nodes("new_session") == [], "stale work was published"
        assert all(node.depth == 0 for node in e._dag.get_session_nodes("s"))
    finally:
        e.shutdown()


def test_a_failed_gc_callback_leaves_the_row_and_its_chunks_alone(tmp_path):
    """round-2 verify-4 #6: the content rewrite, the caller's archive callback and the commit
    were not one transaction, so a callback that raised left the rewrite PENDING and the next
    unrelated commit published the GC placeholder without the chunk archive."""
    from hermes_lcm.store import MessageStore
    store = MessageStore(str(tmp_path / "gc.db"))
    try:
        store_id = store.append("s", {"role": "tool", "tool_call_id": "c1",
                                      "content": "the original result bytes"}, source="cli")
        store.commit()

        def failing(conn, sid):
            raise RuntimeError("archive failed")

        with pytest.raises(RuntimeError):
            store.gc_externalized_tool_result(store_id, "[placeholder]", before_commit=failing)
        assert store._conn.in_transaction is False
        store._conn.commit()  # an unrelated commit must not publish the rewrite
        assert store.get(store_id)["content"] == "the original result bytes"

        assert store.gc_externalized_tool_result(store_id, "[placeholder]") is True
        assert store.get(store_id)["content"] == "[placeholder]"
    finally:
        store.close()


def test_the_store_keeps_the_whole_host_envelope(tmp_path):
    """round-2 verify-4 #4 / verify-3 rank 1: the columns are a PROJECTION of the host's
    message, and everything else it sent — name, reasoning metadata, error flags, provider ids
    — was dropped at the door with no marker, so outcome and attribution were lost before
    summarisation started."""
    from hermes_lcm.store import MessageStore
    store = MessageStore(str(tmp_path / "envelope.db"))
    try:
        store_id = store.append("s", {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "the result",
            "name": "terminal",
            "reasoning_content": "I checked the disk first",
            "is_error": True,
            "provider_metadata": {"vendor": "acme", "request_id": "r-42"},
        }, source="cli")
        store.commit()

        row = store.get(store_id)
        assert row["envelope"]["reasoning_content"] == "I checked the disk first"
        assert row["envelope"]["is_error"] is True
        assert row["envelope"]["provider_metadata"]["request_id"] == "r-42"

        replayed = store.to_openai_msg(row)
        assert replayed["content"] == "the result"
        assert replayed["reasoning_content"] == "I checked the disk first"
        assert replayed["is_error"] is True

        # a batch row keeps it too, and a message with nothing extra stores nothing extra
        ids = store.append_batch("s", [
            {"role": "assistant", "content": "plain"},
            {"role": "user", "content": "with id", "message_id": "m-7"},
        ])
        store.commit()
        assert "envelope" not in store.get(ids[0])
        assert store.get(ids[1])["envelope"] == {"message_id": "m-7"}
    finally:
        store.close()


def test_an_edited_message_with_a_host_id_is_archived_as_a_revision(tmp_path):
    """round-2 verify-4 #5: ingesting [user: ORIGINAL] and then [user: CORRECTED] left only
    ORIGINAL in SQLite — the cursor treats the second snapshot as containing nothing new, so a
    correction the host made to an already-stored message reached the plugin and vanished."""
    cfg = LCMConfig(database_path=str(tmp_path / "revision.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("rev", platform="cli", context_length=200_000)
        e._ingest_messages([{"role": "user", "content": "ORIGINAL", "message_id": "m1"}])
        e._store.commit()
        e._ingest_messages([{"role": "user", "content": "CORRECTED", "message_id": "m1"}])
        e._store.commit()

        contents = [str(row.get("content") or "")
                    for row in e._store.get_session_messages("rev")]
        assert "ORIGINAL" in contents, "the superseded version stays in the archive"
        assert "CORRECTED" in contents, "the correction was lost"
        revision = next(row for row in e._store.get_session_messages("rev")
                        if str(row.get("content")) == "CORRECTED")
        assert revision["envelope"]["lcm_supersedes_store_id"] > 0
        assert "already stored" in revision["envelope"]["lcm_revision_reason"]

        # an unchanged snapshot stores nothing new
        before = len(e._store.get_session_messages("rev"))
        e._ingest_messages([{"role": "user", "content": "CORRECTED", "message_id": "m1"}])
        e._store.commit()
        assert len(e._store.get_session_messages("rev")) == before
    finally:
        e.shutdown()


def test_a_tool_argument_correction_is_archived_and_still_maps(tmp_path):
    """round-3 verify-4 #3 / verify-2 #8: the revision check compared CONTENT only, so an edit
    that changed a command's arguments was never archived; and the archived correction, being
    appended at the end, made the chronological replay matching skip the rows between it and
    the original."""
    from hermes_lcm import escalation
    cfg = LCMConfig(database_path=str(tmp_path / "revision2.db"), fresh_tail_count=1,
                    leaf_chunk_tokens=10)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("rv2", platform="cli", context_length=200_000)
        first = [
            {"role": "user", "content": "run it", "message_id": "m1"},
            {"role": "assistant", "content": "running", "message_id": "m2", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "terminal", "arguments": "alpha"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "done", "message_id": "m3"},
            {"role": "user", "content": "and now?", "message_id": "m4"},
        ]
        e._ingest_messages(first)
        e._store.commit()

        edited = [dict(message) for message in first]
        edited[1] = dict(edited[1], tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "terminal", "arguments": "beta"}}])
        e._ingest_messages(edited)
        e._store.commit()

        rows = e._store.get_session_messages("rv2")
        assert any("beta" in str(row.get("tool_calls") or "") for row in rows), \
            "the corrected command was not archived"

        # ... and the ORIGINAL rows still map, so compaction can still publish
        mapped = e._get_store_ids_for_messages(edited)
        assert len(mapped) == len(edited), mapped

        e.threshold_tokens = 1
        e._resolve_window_scaled_settings()
        original = escalation._call_llm_for_summary
        escalation._call_llm_for_summary = lambda *a, **k: "ran the command\nExpand for details about: command"
        try:
            e.compress(list(edited) + [{"role": "user", "content": "tail"}], current_tokens=400_000)
        finally:
            escalation._call_llm_for_summary = original
        nodes = e._dag.get_session_nodes("rv2")
        if nodes:
            covered = {int(value) for node in nodes for value in node.source_ids}
            revision_ids = {int(row["store_id"]) for row in rows
                            if "beta" in str(row.get("tool_calls") or "")}
            assert revision_ids <= covered, (revision_ids, covered)
    finally:
        e.shutdown()


def test_a_late_session_end_stores_under_the_session_that_ended(tmp_path):
    """round-3 verify-4 #4: _ingest_messages takes its ownership from mutable engine state, so
    a session-end callback arriving after the engine had rebound stored the OLD session's
    history — its whole prefix and final response — under the NEW session."""
    cfg = LCMConfig(database_path=str(tmp_path / "lateend.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("old", platform="cli", context_length=200_000)
        e._ingest_messages([{"role": "user", "content": "the old question"}])
        e._store.commit()
        e.on_session_start("new", platform="cli", context_length=200_000)

        e.on_session_end("old", [
            {"role": "user", "content": "the old question"},
            {"role": "assistant", "content": "THE OLD FINAL ANSWER"},
        ])
        e._store.commit()

        old_rows = [str(row.get("content") or "") for row in e._store.get_session_messages("old")]
        new_rows = [str(row.get("content") or "") for row in e._store.get_session_messages("new")]
        assert any("THE OLD FINAL ANSWER" in text for text in old_rows), old_rows
        assert not any("THE OLD FINAL ANSWER" in text for text in new_rows), new_rows
        assert e.current_session_id == "new", "the binding is restored"
    finally:
        e.shutdown()


def test_a_rebind_after_publication_does_not_hand_the_new_session_old_context(tmp_path):
    """round-3 verify-2 #3 / verify-4 #1: the fence guarded publication, but a rebind landing
    between publication and the caller receiving the result handed the NEW session the old
    session's summaries and cursor."""
    from hermes_lcm import escalation
    cfg = LCMConfig(database_path=str(tmp_path / "assemblyfence.db"), fresh_tail_count=1,
                    leaf_chunk_tokens=10, incremental_max_depth=0)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("old", platform="cli", context_length=200_000)
        e.threshold_tokens = 1
        e._resolve_window_scaled_settings()
        messages = [
            {"role": "user", "content": "u" * 400},
            {"role": "assistant", "content": "a" * 400},
            {"role": "user", "content": "the newest turn"},
        ]
        e._ingest_messages(messages)
        e._store.commit()

        original = escalation._call_llm_for_summary
        escalation._call_llm_for_summary = (
            lambda *a, **k: "a summary\nExpand for details about: it"
        )
        real_publish = e._dag.add_node_with_meta
        state = {"rebound": False}

        def publish_then_rebind(*args, **kwargs):
            node_id = real_publish(*args, **kwargs)
            if not state["rebound"]:
                state["rebound"] = True
                # the publication itself is fenced; this lands immediately AFTER it, while the
                # result is still being assembled
                e._publication_generation = int(
                    getattr(e, "_publication_generation", 0)
                ) + 1
            return node_id

        e._dag.add_node_with_meta = publish_then_rebind
        try:
            returned = e.compress(list(messages), current_tokens=400_000)
        finally:
            escalation._call_llm_for_summary = original
            e._dag.add_node_with_meta = real_publish

        assert returned == messages, "stale assembled context was handed to the caller"
        assert e._last_compression_status in {"noop", "sanitized"}
    finally:
        e.shutdown()


def test_a_failed_revision_write_is_retried_on_the_next_turn(tmp_path):
    """round-4 verify-2 #1: the fingerprint cache advanced before the archive write succeeded,
    so a transient failure was cached as "already checked" and an unchanged retry never wrote
    the correction again — the edit stayed lost."""
    import sqlite3
    cfg = LCMConfig(database_path=str(tmp_path / "retryrev.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("rr", platform="cli", context_length=200_000)
        e._ingest_messages([{"role": "user", "content": "ORIGINAL", "message_id": "m1"}])
        e._store.commit()

        real_append = e._store.append
        state = {"failed": False}

        def failing_append(*args, **kwargs):
            if not state["failed"] and "CORRECTED" in str(args[1].get("content") or ""):
                state["failed"] = True
                raise sqlite3.OperationalError("disk full")
            return real_append(*args, **kwargs)

        e._store.append = failing_append
        try:
            e._ingest_messages([{"role": "user", "content": "CORRECTED", "message_id": "m1"}])
        finally:
            e._store.append = real_append
        contents = [str(row.get("content") or "") for row in e._store.get_session_messages("rr")]
        assert "CORRECTED" not in contents, "the probe did not exercise the failure"

        # the same snapshot again: the correction must be attempted once more
        e._ingest_messages([{"role": "user", "content": "CORRECTED", "message_id": "m1"}])
        e._store.commit()
        contents = [str(row.get("content") or "") for row in e._store.get_session_messages("rr")]
        assert "CORRECTED" in contents, contents
        assert "ORIGINAL" in contents
    finally:
        e.shutdown()


def test_a_consumed_revision_row_does_not_move_the_frontier_forward(tmp_path):
    """round-4 verify-2 #3: a revision row is appended at the END of the archive, and taking
    its id as a chronological position made the frontier jump to it and then move backward,
    leaving an active row past the frontier that a raw-after-frontier query could not see."""
    from hermes_lcm import escalation
    cfg = LCMConfig(database_path=str(tmp_path / "revfrontier.db"), fresh_tail_count=1,
                    leaf_chunk_tokens=10, incremental_max_depth=0)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("rf", platform="cli", context_length=200_000)
        messages = [
            {"role": "user", "content": "first " + "w" * 200, "message_id": "m1"},
            {"role": "user", "content": "second " + "w" * 200, "message_id": "m2"},
            {"role": "user", "content": "third " + "w" * 200, "message_id": "m3"},
            {"role": "user", "content": "fourth " + "w" * 200, "message_id": "m4"},
        ]
        e._ingest_messages(messages)
        e._store.commit()
        edited = [dict(message) for message in messages]
        edited[0] = dict(edited[0], content="FIRST CORRECTED " + "w" * 200)
        e._ingest_messages(edited)
        e._store.commit()

        e.threshold_tokens = 1
        e._resolve_window_scaled_settings()
        original = escalation._call_llm_for_summary
        escalation._call_llm_for_summary = lambda *a, **k: "s\nExpand for details about: s"
        try:
            e.compress(list(edited) + [{"role": "user", "content": "tail"}], current_tokens=400_000)
        finally:
            escalation._call_llm_for_summary = original

        rows = e._store.get_session_messages("rf")
        revision_ids = {int(row["store_id"]) for row in rows
                        if "FIRST CORRECTED" in str(row.get("content") or "")}
        assert revision_ids, "the correction was not archived"
        assert e._last_compacted_store_id not in revision_ids, (
            "the frontier stopped on an appended revision row"
        )
        chronological = [int(row["store_id"]) for row in rows
                         if int(row["store_id"]) not in revision_ids]
        assert e._last_compacted_store_id <= max(chronological)
    finally:
        e.shutdown()


def test_a_correction_of_a_correction_is_still_covered(tmp_path):
    """round-4 verify-4 #3: only the direct supersession edge was followed, so a correction of
    a correction sat outside every published leaf while the receipt advertised one newer
    version."""
    from hermes_lcm.store import MessageStore
    store = MessageStore(str(tmp_path / "chain.db"))
    try:
        original = store.append("s", {"role": "user", "content": "v1"}, source="cli")
        first = store.append("s", {"role": "user", "content": "v2",
                                   "lcm_supersedes_store_id": original}, source="cli")
        second = store.append("s", {"role": "user", "content": "v3",
                                    "lcm_supersedes_store_id": first}, source="cli")
        store.commit()
        assert store.revision_rows_for("s", [original]) == [first, second]
        assert store.superseded_ids_for("s", [second]) == {second: first}
    finally:
        store.close()


def test_an_edit_that_changes_only_a_projected_field_is_archived(tmp_path):
    """round-4 verify-4 #2: the revision fingerprint omitted the projected columns, so changing
    only a tool_call_id, a tool name or the host's timestamp produced an identical fingerprint
    and the edit was never archived."""
    from hermes_lcm.store import message_envelope_fingerprint

    base = {"role": "tool", "content": "result", "tool_call_id": "c1", "message_id": "m1"}
    assert message_envelope_fingerprint(base) != message_envelope_fingerprint(
        {**base, "tool_call_id": "c2"})
    assert message_envelope_fingerprint(base) != message_envelope_fingerprint(
        {**base, "tool_name": "terminal"})
    assert message_envelope_fingerprint({**base, "timestamp": 1.0}) != (
        message_envelope_fingerprint({**base, "timestamp": 2.0}))

    # ... and two long host ids stay distinct
    from hermes_lcm.store import host_message_id_of
    long_a = "x" * 250 + "A"
    long_b = "x" * 250 + "B"
    assert host_message_id_of({"message_id": long_a}) != host_message_id_of({"message_id": long_b})


def test_a_corrupt_envelope_is_returned_whole_and_paged(tmp_path):
    """round-4: corrupt envelope JSON was cut at 20,000 chars in the store with no marker and
    no cursor, and `lcm_expand` on a raw row did not report the corruption at all — it said
    has_more=false while the host fields sat unreadable in the column."""
    import json
    from hermes_lcm.tools import lcm_expand

    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "corrupt_envelope.db")
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    engine.on_session_start("s", context_length=200_000)
    store = engine._store
    try:
        store_id = store.append("s", {"role": "user", "content": "hi"}, source="cli")
        broken = '{"reasoning_content": "' + "z" * 60_000  # never closed
        store._conn.execute(
            "UPDATE messages SET envelope_extra = ? WHERE store_id = ?", (broken, store_id)
        )
        store.commit()

        row = store.get(store_id)
        assert row["envelope_corrupt"] is True
        assert row["envelope_raw"] == broken          # whole, not 20,000 chars
        assert row["envelope_raw_chars"] == len(broken)

        first = json.loads(lcm_expand({"store_id": store_id, "max_tokens": 500}, engine=engine))
        assert first["envelope_corrupt"] is True
        assert first["envelope_raw_truncated"] is True
        assert first["has_more"] is True
        cursor = first["envelope_raw_continue_with"]["envelope_offset"]
        assert cursor == first["envelope_raw_next_offset"] > 0

        seen = first["envelope_raw"]
        while True:
            page = json.loads(lcm_expand(
                {"store_id": store_id, "max_tokens": 500, "envelope_offset": cursor},
                engine=engine,
            ))
            seen += page["envelope_raw"]
            if not page.get("envelope_raw_truncated"):
                break
            cursor = page["envelope_raw_next_offset"]
        assert seen == broken                          # every character is reachable
    finally:
        engine.shutdown()


def test_an_ingest_that_outlives_its_session_files_under_the_session_it_started_in(tmp_path):
    """round-5 verify-6 #1: ownership was read from mutable engine state again AFTER the
    protection work, so a rebind landing in between filed the old turn under the new session
    and then set the NEW session's cursor past a request that had never been stored."""
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "fence.db")
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("old", context_length=200_000)
        real_batch = engine._store._append_protected_batch
        rebound = {"done": False}

        def rebind_then_store(session_id, rows, estimates, **kwargs):
            if not rebound["done"]:
                rebound["done"] = True
                engine.on_session_start("new", context_length=200_000)
            return real_batch(session_id, rows, estimates, **kwargs)

        engine._store._append_protected_batch = rebind_then_store
        engine._ingest_messages([{"role": "user", "content": "OLD SESSION SECRET"}])
        engine._store._append_protected_batch = real_batch

        old_rows = [r["content"] for r in engine._store.get_session_messages("old")]
        new_rows = [r["content"] for r in engine._store.get_session_messages("new")]
        assert old_rows == ["OLD SESSION SECRET"], (old_rows, new_rows)
        assert new_rows == []
        # and the new session's cursor was not advanced past its own unstored request
        engine._ingest_messages([{"role": "user", "content": "NEW SESSION REQUEST"}])
        assert [r["content"] for r in engine._store.get_session_messages("new")] == [
            "NEW SESSION REQUEST"
        ]
    finally:
        engine.shutdown()


def test_a_failed_carry_over_commit_leaves_no_pending_ownership_change(tmp_path):
    """round-5 verify-6 #11: `reassign_session_nodes` lacked the rollback discipline used by
    node publication, so a failed commit left the transaction open with the ownership change
    pending, and the next unrelated publication committed it."""
    dag = SummaryDAG(str(tmp_path / "carry.db"))
    try:
        node_id = dag.add_node(_node(session="old"))

        class _RefusingCommit:
            def __init__(self, connection):
                self._connection = connection

            def __getattr__(self, name):
                return getattr(self._connection, name)

            def commit(self):
                raise sqlite3.OperationalError("disk I/O error")

        real_connection = dag._conn
        dag._conn = _RefusingCommit(real_connection)
        try:
            with pytest.raises(sqlite3.OperationalError):
                dag.reassign_session_nodes("old", "new")
        finally:
            dag._conn = real_connection

        assert dag._conn.in_transaction is False
        assert [n.node_id for n in dag.get_session_nodes("old")] == [node_id]
        assert dag.get_session_nodes("new") == []
    finally:
        dag.close()
