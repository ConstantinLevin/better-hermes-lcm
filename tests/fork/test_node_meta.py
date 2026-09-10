"""Step 6 — lcm_node_meta sidecar: level + index block per node, cascade delete, classifier."""
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


# fork: better-hermes-lcm — the suite runs with an isolated HOME (audit E, E07), so this one test,
# which deliberately reads a COPY of the live database to prove bootstrap compatibility with a
# pre-existing file, asks conftest for the real home instead of expanding "~".
_LIVE_DB = Path(os.environ.get("LCM_TESTS_ORIGINAL_HOME", "~")).expanduser() / ".hermes" / "lcm.db"


@pytest.mark.skipif(not _LIVE_DB.exists(), reason="no pre-existing lcm.db on this box")
def test_bootstrap_against_a_copy_of_the_preexisting_db(tmp_path):
    src = _LIVE_DB
    dst = tmp_path / "copy.db"
    shutil.copy2(src, dst)
    before = sqlite3.connect(str(dst))
    node_count = before.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]
    before.close()
    dag = SummaryDAG(dst)
    try:
        assert dag.connection.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == node_count
        tables = {r[0] for r in dag.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert node_meta.NODE_META_TABLE in tables
        conn = sqlite3.connect(str(dst))
        try:
            assert classify_version_mismatch(conn) != VERSION_MISMATCH_GENUINELY_NEWER
        finally:
            conn.close()
    finally:
        dag.close()


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
