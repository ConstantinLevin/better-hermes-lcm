"""Step 6 — lcm_node_meta sidecar: level + index block per node, cascade delete, classifier."""
import json
import shutil
import sqlite3
import time
from pathlib import Path

import os

import pytest

from hermes_lcm import node_meta
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.db_bootstrap import (
    VERSION_MISMATCH_GENUINELY_NEWER,
    classify_version_mismatch,
)
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import RECOVERED_FOR_KEY


def _node(session, depth, summary, created=None):
    return SummaryNode(
        session_id=session, depth=depth, summary=summary, token_count=10,
        source_token_count=50, source_ids=[], source_type="messages",
        created_at=created or time.time(),
    )


def test_migration_is_idempotent_and_marked(tmp_path):
    path = tmp_path / "meta.db"
    dag = SummaryDAG(path)
    dag.close()
    dag = SummaryDAG(path)  # second bootstrap: no error, table still there
    try:
        tables = {r[0] for r in dag.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert node_meta.NODE_META_TABLE in tables
        steps = {r[0] for r in dag.connection.execute("SELECT step_name FROM lcm_migration_state")}
        assert node_meta.MIGRATION_STEP in steps
    finally:
        dag.close()


def test_classifier_accepts_the_sidecar(tmp_path):
    # the classifier needs the full core schema (store + DAG), so bootstrap an engine
    path = tmp_path / "classify.db"
    cfg = LCMConfig()
    cfg.database_path = str(path)
    LCMEngine(config=cfg, hermes_home=str(tmp_path)).shutdown()
    conn = sqlite3.connect(str(path))
    try:
        assert classify_version_mismatch(conn) != VERSION_MISMATCH_GENUINELY_NEWER
    finally:
        conn.close()


def test_roundtrip_and_index_block_extraction(tmp_path):
    dag = SummaryDAG(tmp_path / "rt.db")
    try:
        summary = (
            "Decisions: use sqlite.\nExpand for details about:\n- the sqlite decision and its rationale\n"
            "- the rejected postgres option\n- files: dag.py, node_meta.py"
        )
        node_id = dag.add_node(_node("s", 0, summary))
        dag.node_meta.write(node_id, level=2, summary=summary)
        meta = dag.node_meta.read(node_id)
        assert meta["level"] == 2
        assert meta["index_block"].splitlines() == [
            "- the sqlite decision and its rationale",
            "- the rejected postgres option",
            "- files: dag.py, node_meta.py",
        ]
        # expand_hint contract unchanged: first line only
        assert LCMEngine._extract_expand_hint(summary) == "- the sqlite decision and its rationale"
        # re-write updates in place
        dag.node_meta.write(node_id, level=1, summary="no marker here")
        assert dag.node_meta.read(node_id) == {"level": 1, "index_block": ""}
    finally:
        dag.close()


def test_index_block_is_stored_whole():
    """The index block used to be cut at 1,600 characters — mid-topic, with no continuation,
    and surfaced in that state by the tools. Cutting the index is the loss this fork removes.
    """
    topics = "\n".join(f"- topic {i}: what happened and where to look" for i in range(200))
    block = node_meta.extract_index_block("body\nExpand for details about:\n" + topics)
    assert block.count("\n") == 199
    assert "topic 199" in block and not block.endswith("…")
    assert len(block) > node_meta.INDEX_BLOCK_MAX_CHARS  # the historical cut would have hit here


def test_cascade_delete_removes_sidecar_rows(tmp_path):
    dag = SummaryDAG(tmp_path / "cascade.db")
    try:
        ids = [dag.add_node(_node("s", d, f"d{d}")) for d in range(3)]
        for node_id in ids:
            dag.node_meta.write(node_id, level=1, summary="")
        assert len(dag.node_meta.read_many(ids)) == 3
        dag.delete_below_depth("s", 2)
        assert set(dag.node_meta.read_many(ids)) == {ids[2]}
        dag.delete_session_nodes("s")
        assert dag.node_meta.read_many(ids) == {}
    finally:
        dag.close()


def test_leaf_and_condense_writes_record_level(tmp_path, monkeypatch):
    from hermes_lcm import escalation
    cfg = LCMConfig(fresh_tail_count=1, leaf_chunk_tokens=1, condensation_fanin=2, incremental_max_depth=3)
    cfg.database_path = str(tmp_path / "levels.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e._session_id = "lv"
    e.threshold_tokens = 1
    try:
        # first call answers on L1; the second refuses L1 so the engine takes L2 (bullets)
        state = {"n": 0}

        def fake(prompt, max_tokens, model="", timeout=None):
            state["n"] += 1
            system = prompt[0]["content"] if isinstance(prompt, list) else ""
            if state["n"] > 1 and "bullet" not in system:
                return None
            return "s\nExpand for details about: mock"

        monkeypatch.setattr(escalation, "_call_llm_for_summary", fake)
        # replay like the host does: the next turn carries the context compress() returned,
        # so the new raw messages are actually ingested and the leaf keeps its lineage
        ctx = e.compress([{"role": "user", "content": "one " * 30}, {"role": "user", "content": "tail"}])
        e.compress(list(ctx) + [{"role": "user", "content": "two " * 30},
                                {"role": "user", "content": "tail2"}])
        nodes = e._dag.get_session_nodes("lv")
        levels = {n.node_id: e._dag.node_meta.read(n.node_id)["level"] for n in nodes}
        assert sorted(levels.values())[0] == 1 and 2 in levels.values()
        assembled = e._assemble_context(None, [{"role": "user", "content": "t"}])
        assert "L2 bullet summary" in assembled[0]["content"]
    finally:
        e.shutdown()


def test_rotate_marker_records_marker_level(tmp_path):
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "rot.db")
    cfg.fresh_tail_count = 2
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path / "home"))
    try:
        e._session_id = e._conversation_id = "live"
        e._session_platform = "cli"
        e._lifecycle.bind_session("live", conversation_id="live")
        e.context_length = 200_000
        for i in range(6):
            e._store.append("live", {"role": "user", "content": f"m{i} " + "x" * 50}, source="test")
        e._store._conn.commit()
        result = e.rotate_active_session(apply=True)
        assert e._dag.node_meta.read(result["marker_node_id"])["level"] == node_meta.LEVEL_MARKER
        assembled = e._assemble_context(None, [{"role": "user", "content": "t"}])
        assert "deterministic marker" in assembled[0]["content"]
    finally:
        e.shutdown()


# the suite runs with an isolated HOME (audit E, E07), so this one test,
# which deliberately reads a COPY of the live database to prove bootstrap compatibility with a
# pre-existing file, asks conftest for the real home instead of expanding "~".
_LIVE_DB = Path(os.environ.get("LCM_TESTS_ORIGINAL_HOME", "~")).expanduser() / ".hermes" / "lcm.db"


def _consistent_snapshot(source: Path, destination: Path) -> None:
    """Copy a live SQLite database with FILE READS ONLY, and with its log.

    `shutil.copy2` of the main file alone is not a snapshot: the database runs in WAL mode, so
    every commit since the last checkpoint lives in `-wal` and a main-file-only copy is missing
    exactly the most recent history a compatibility check cares about.

    SQLite's backup API would give a cleaner point-in-time copy, and it is the wrong tool here.
    It needs a connection to the SOURCE, and opening a WAL database — even `mode=ro` — creates
    `-shm` and `-wal` next to it if they are not already there. This test's source is the
    operator's live store; the rule for it is read-only, and creating files beside it is not
    read-only. Measured: a `mode=ro` open of a checkpointed WAL database leaves both sidecars
    behind after close.

    So: copy the main file, then the log. A WAL is append-only between checkpoints, so a `-wal`
    copied after the main file holds at least the frames that copy is missing. The `-shm` is
    deliberately not copied — it is a scratch index SQLite rebuilds from the `-wal`, and a stale
    copy of it is worse than no copy at all.
    """
    shutil.copy2(source, destination)
    log = Path(f"{source}-wal")
    if log.exists():
        shutil.copy2(log, Path(f"{destination}-wal"))


def _snapshot_state(path: Path) -> dict:
    """Everything the roundtrip has to still be true about, read before the fork opens the file.

    Opened read-write because `path` is our own temp copy; that also replays its `-wal` so the
    engine below sees the same database this function measured.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            (r["store_id"], r["session_id"], r["role"], r["content"], r["tool_calls"],
             r["tool_call_id"], r["envelope_extra"], r["observed_at"], r["observed_at_source"],
             r["host_message_id"])
            for r in conn.execute(
                "SELECT store_id, session_id, role, content, tool_calls, tool_call_id, "
                "envelope_extra, observed_at, observed_at_source, host_message_id "
                "FROM messages ORDER BY store_id")
        ]
        lifecycle = [tuple(r) for r in conn.execute(
            "SELECT conversation_id, current_session_id, last_finalized_session_id, "
            "current_frontier_store_id, last_finalized_frontier_store_id "
            "FROM lcm_lifecycle_state ORDER BY conversation_id")]
        return {
            "rows": rows,
            "node_count": conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0],
            "lifecycle": lifecycle,
            "attachment_rows": conn.execute(
                "SELECT COUNT(*) FROM messages WHERE envelope_extra LIKE ?",
                (f"%{RECOVERED_FOR_KEY}%",)).fetchone()[0],
        }
    finally:
        conn.close()


# Kept as a skipif and not softened into a synthetic fixture: this is the one test whose whole
# subject is a database somebody else's build wrote, and a fresh synthetic file cannot honestly
# stand in for that. No supplied store, no claim. Equally, a damaged or partial one must not be
# made to look compatible by quietly leaving out what did not read back — every count below is
# compared against the same count taken from the snapshot, so an empty legacy store proves
# emptiness rather than success.
@pytest.mark.skipif(not _LIVE_DB.exists(), reason="no pre-existing lcm.db on this box")
def test_bootstrap_against_a_copy_of_the_preexisting_db(tmp_path):
    snapshot = tmp_path / "snapshot.db"
    _consistent_snapshot(_LIVE_DB, snapshot)
    before = _snapshot_state(snapshot)

    cfg = LCMConfig()
    cfg.database_path = str(snapshot)
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path / "home"))
    try:
        # the sidecar migration is additive and an older build's file still classifies as
        # openable rather than "written by something newer"
        tables = {r[0] for r in engine._dag.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert node_meta.NODE_META_TABLE in tables
        assert engine._dag.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == before["node_count"]

        # every original row reads back through the store's own reader, with the fields a
        # projected-column check would miss: the envelope, the tool-call payload, the host's
        # source time and its provenance, and the host message id
        sessions = sorted({row[1] for row in before["rows"]})
        roundtripped = {}
        for session_id in sessions:
            for row in engine._store.get_session_messages(session_id):
                envelope = row.get("envelope")
                roundtripped[int(row["store_id"])] = (
                    row.get("role"),
                    row.get("content"),
                    row.get("tool_call_id"),
                    envelope,
                    row.get("observed_at"),
                    row.get("observed_at_source"),
                )
        assert len(roundtripped) == len(before["rows"]), (
            "the fork's reader does not return every row the legacy file holds"
        )
        for (store_id, _session, role, content, tool_calls, tool_call_id,
             envelope_extra, observed_at, observed_at_source, _host_id) in before["rows"]:
            actual = roundtripped[int(store_id)]
            assert actual[0] == role and actual[1] == content, store_id
            assert actual[2] == tool_call_id, store_id
            assert actual[4] == observed_at and actual[5] == observed_at_source, store_id
            expected_envelope = json.loads(envelope_extra) if envelope_extra else {}
            for key, value in expected_envelope.items():
                assert (actual[3] or {}).get(key) == value, (store_id, key)
            if tool_calls:
                stored_calls = engine._store.connection.execute(
                    "SELECT tool_calls FROM messages WHERE store_id = ?", (store_id,)
                ).fetchone()[0]
                assert stored_calls == tool_calls, store_id

        # attachment rows: recovered bodies stay attached to the row they belong to. Zero found
        # is asserted as zero rather than skipped, so this says "the legacy store has none"
        # instead of quietly saying nothing.
        attachments_after = engine._store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE envelope_extra LIKE ?",
            (f"%{RECOVERED_FOR_KEY}%",)).fetchone()[0]
        assert attachments_after == before["attachment_rows"]
        for session_id in sessions:
            ids = [int(row["store_id"]) for row in engine._store.get_session_messages(session_id)]
            for attached in engine._store.attached_recovered_body_ids_for_rows(session_id, ids):
                assert attached in ids, (
                    "a recovered body row is attached to a row outside its own session"
                )

        # the frontier every conversation had is still the frontier it has
        lifecycle_after = [tuple(r) for r in engine._store.connection.execute(
            "SELECT conversation_id, current_session_id, last_finalized_session_id, "
            "current_frontier_store_id, last_finalized_frontier_store_id "
            "FROM lcm_lifecycle_state ORDER BY conversation_id")]
        assert lifecycle_after == before["lifecycle"], "bootstrap rewrote the frontier"

        # resume: binding an existing session must not lose, duplicate or renumber its rows
        resumed = sessions[-1]
        expected = [(int(row[0]), row[3]) for row in before["rows"] if row[1] == resumed]
        engine.on_session_start(resumed, platform="cli", context_length=262_144)
        after_resume = [(int(row["store_id"]), row.get("content"))
                        for row in engine._store.get_session_messages(resumed)]
        assert after_resume == expected, "resuming an existing session changed its rows"

        conn = sqlite3.connect(str(snapshot))
        try:
            assert classify_version_mismatch(conn) != VERSION_MISMATCH_GENUINELY_NEWER
        finally:
            conn.close()
    finally:
        engine.shutdown()


def test_node_results_carry_the_index_block_when_present(tmp_path):
    import json
    from hermes_lcm import tools as lcm_tools
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "ib.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("ib", platform="cli", context_length=200_000)
        summary = "Body.\nExpand for details about: first topic\n- second topic\n- third topic"
        rich = e._dag.add_node(SummaryNode(
            session_id="ib", depth=0, summary=summary, token_count=10, source_token_count=50,
            source_ids=[], source_type="messages", created_at=time.time(),
            expand_hint=LCMEngine._extract_expand_hint(summary),
        ))
        e._dag.node_meta.write(rich, level=1, summary=summary)
        plain = e._dag.add_node(_node("ib", 0, "no marker"))
        e._dag.node_meta.write(plain, level=1, summary="no marker")
        assert lcm_tools._node_index_block_payload(e, e._dag.get_node(rich)) == {
            "index_block": "first topic\n- second topic\n- third topic"
        }
        assert lcm_tools._node_index_block_payload(e, e._dag.get_node(plain)) == {}
        described = json.loads(lcm_tools.lcm_describe({}, engine=e))
        by_id = {n["node_id"]: n for depth in described.get("depths", {}).values() for n in depth.get("nodes", [])} \
            if isinstance(described.get("depths"), dict) else {}
        blob = json.dumps(described)
        assert "- second topic" in blob  # the multi-line block reaches the tool result
        assert blob.count('"index_block"') == 1  # only the node that has one
    finally:
        e.shutdown()


def test_index_block_is_bounded_in_responses_but_never_in_storage(tmp_path):
    """Audit D #5: storing the block whole is right; attaching it unbounded to every node in a
    retrieval result let one call carry far more than the tool's own budget."""
    from hermes_lcm import tools as lcm_tools
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "bounded.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("b", platform="cli", context_length=200_000)
        topics = "\n".join(f"- topic {i}: what happened and where to look for it" for i in range(400))
        summary = "body\nExpand for details about:\n" + topics
        node_id = e._dag.add_node(SummaryNode(
            session_id="b", depth=0, summary=summary, token_count=10, source_token_count=50,
            source_ids=[], source_type="messages", created_at=time.time(),
            expand_hint=LCMEngine._extract_expand_hint(summary)))
        e._dag.node_meta.write(node_id, level=1, summary=summary)

        stored = e._dag.node_meta.read(node_id)["index_block"]
        assert "topic 399" in stored, "storage is never shortened"

        payload = lcm_tools._node_index_block_payload(e, e._dag.get_node(node_id))
        assert payload["index_block_truncated"] is True
        assert len(payload["index_block"]) <= 4_000
        assert payload["index_block_total_chars"] == len(stored)
        assert payload["index_block_continue_with"]["node_id"] == node_id
        assert not payload["index_block"].endswith("- topic")  # cut on a line boundary
    finally:
        e.shutdown()
