"""Step 5 — no unmarked loss: every remaining cut or drop leaves a marker with provenance."""
import re
import time

import pytest

from hermes_lcm import marked_loss
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


class _RegexPattern:
    """The optional ``regex`` package may be absent; message patterns need a timeout-capable
    engine, so mirror upstream's test double."""

    def __init__(self, pattern):
        self._compiled = re.compile(pattern)

    def search(self, text, *, timeout=None):
        assert timeout is not None
        return self._compiled.search(text)


class _RegexEngine:
    error = re.error

    @staticmethod
    def compile(pattern):
        return _RegexPattern(pattern)


@pytest.fixture
def ignore_patterns_engine(monkeypatch):
    from hermes_lcm import message_patterns as message_patterns_mod
    monkeypatch.setattr(message_patterns_mod, "_regex_engine", _RegexEngine)


def _engine(tmp_path, name="lcm.db", **kw):
    cfg = LCMConfig(**kw)
    cfg.database_path = str(tmp_path / name)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e._session_id = "marked-session"
    return e


# ── serialize ──────────────────────────────────────────────────────────────────────────────

def test_serialize_at_256k_matches_upstream_literals(tmp_path):
    e = _engine(tmp_path)
    try:
        e._set_context_length(262_144, source="test")
        content = "u" * 5000
        serialized = e._serialize_messages([{"role": "user", "content": content}])
        # upstream: head 2000 + "...[truncated]..." + tail 800
        assert serialized.startswith("[USER]: " + "u" * 2000 + "\n...[truncated]...")
        assert serialized.endswith("u" * 800)
        assert "u" * 2001 not in serialized
    finally:
        e.shutdown()


def test_serialize_marks_elision_with_sizes(tmp_path):
    e = _engine(tmp_path)
    try:
        e._set_context_length(262_144, source="test")
        serialized = e._serialize_messages([{"role": "user", "content": "x" * 5000}])
        assert "[LCM elided 2200 of 5000 chars before summarising" in serialized
        assert "lcm_expand" in serialized
    finally:
        e.shutdown()


def test_serialize_at_1m_keeps_whole_message(tmp_path):
    e = _engine(tmp_path)
    try:
        e._set_context_length(1_000_000, source="test")
        content = "w" * 50_000
        serialized = e._serialize_messages([{"role": "user", "content": content}])
        assert content in serialized
        assert "...[truncated]..." not in serialized
    finally:
        e.shutdown()


def test_serialize_cap_is_weighted_between_anchors(tmp_path):
    e = _engine(tmp_path)
    try:
        e._set_context_length(512_000, source="test")
        cap = int(e.effective_serialize_message_max_chars)
        assert 3000 < cap < 4_000_000
        serialized = e._serialize_messages([{"role": "user", "content": "v" * (cap + 1000)}])
        assert "[LCM elided" in serialized
    finally:
        e.shutdown()


def test_serialize_keeps_unmatched_tool_calls_marked(tmp_path):
    e = _engine(tmp_path)
    try:
        messages = [
            {
                "role": "assistant",
                "content": "I will inspect the repo.",
                "tool_calls": [
                    {"id": "call_ok", "type": "function",
                     "function": {"name": "read_file", "arguments": "{\"path\": \"README.md\"}"}},
                    {"id": "call_missing", "type": "function",
                     "function": {"name": "terminal", "arguments": "{\"command\": \"rm -rf noisy-orphan\"}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_ok", "content": "README says hello"},
        ]
        serialized = e._serialize_messages(messages)
        assert "read_file(" in serialized
        assert "terminal(" in serialized and "noisy-orphan" in serialized
        assert marked_loss.unmatched_tool_call_note() in serialized
        # the matched call carries no note
        line = next(l for l in serialized.splitlines() if "read_file(" in l)
        assert marked_loss.unmatched_tool_call_note() not in line
    finally:
        e.shutdown()


def test_serialize_marks_argument_elision(tmp_path):
    e = _engine(tmp_path)
    try:
        args = "{\"command\": \"" + "a" * 2000 + "\"}"
        serialized = e._serialize_messages([
            {"role": "assistant", "content": "run", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": args}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ])
        assert "[LCM elided" in serialized and "chars of arguments]" in serialized
        assert "a" * 380 in serialized and "a" * 400 not in serialized
    finally:
        e.shutdown()


def test_serialize_externalized_output_carries_a_head(tmp_path):
    e = _engine(
        tmp_path,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
    )
    try:
        content = "first line of a big output that identifies it\n" + "z" * 5000
        serialized = e._serialize_messages([{"role": "tool", "tool_call_id": "call_big", "content": content}])
        assert "[Externalized tool output" in serialized
        assert "[LCM head of externalized output: first line of a big output" in serialized
        assert content[:500] not in serialized
    finally:
        e.shutdown()


def test_content_head_is_single_line_and_bracket_free():
    head = marked_loss.content_head("a [b]; c\n\n d\te" + "x" * 1000, limit=40)
    assert "\n" not in head and "[" not in head and "]" not in head and ";" not in head
    assert len(head) <= 40 and head.endswith("…")


# ── assembly ───────────────────────────────────────────────────────────────────────────────

def _add(e, session, depth, text, created):
    return e._dag.add_node(SummaryNode(
        session_id=session, depth=depth, summary=text, token_count=10,
        source_token_count=50, source_ids=[], source_type="messages", created_at=created,
    ))


def test_assembly_renders_all_frontier_nodes_past_100(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)  # no condensation
    try:
        base = time.time()
        ids = [_add(e, "marked-session", 0, f"leaf number {i}", base + i) for i in range(130)]
        assembled = e._assemble_context(None, [{"role": "user", "content": "tail"}])
        prefix = assembled[0]["content"]
        for node_id in ids:
            assert f"node {node_id})" in prefix
        assert "[LCM assembly omissions" not in prefix
    finally:
        e.shutdown()


def test_assembly_depth_cap_hit_is_marked(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0, assembly_max_nodes_per_depth=5)
    try:
        base = time.time()
        for i in range(8):
            _add(e, "marked-session", 0, f"leaf number {i}", base + i)
        assembled = e._assemble_context(None, [{"role": "user", "content": "tail"}])
        prefix = assembled[0]["content"]
        assert "[LCM assembly omissions" in prefix
        assert "more d0 summaries exist than assembly_max_nodes_per_depth renders" in prefix
    finally:
        e.shutdown()


def test_an_assistant_turn_dropped_as_internal_only_is_named_in_the_prefix(tmp_path):
    """Audit p05 SA01: active-context cleanup drops assistant turns whose only content was
    internal/reasoning material. Upstream logged that for the operator; the agent's own view
    of its history was simply one turn shorter with nothing to say so."""
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        tail = [
            {"role": "user", "content": "what did you decide?"},
            {"role": "assistant", "content": "<thinking>internal only</thinking>"},
            {"role": "user", "content": "well?"},
        ]
        assembled = e._assemble_context(None, tail)
        rendered = "\n".join(str(m.get("content")) for m in assembled)
        assert "[LCM assembly omissions" in rendered
        assert "held only internal/reasoning content" in rendered
        assert all("internal only" not in str(m.get("content")) for m in assembled)
    finally:
        e.shutdown()


def test_assembly_budget_skips_are_marked_with_node_ids(tmp_path):
    e = _engine(tmp_path, incremental_max_depth=0)
    try:
        base = time.time()
        big = _add(e, "marked-session", 0, "big " * 400, base)
        small = _add(e, "marked-session", 0, "small summary", base + 1)
        assembled = e._assemble_context(
            None, [{"role": "user", "content": "tail"}], assembly_cap_override=200,
        )
        prefix = "\n".join(str(m.get("content")) for m in assembled)
        assert "[LCM assembly omissions" in prefix
        assert f"lcm_expand(node_id=…) for {big}" in prefix or f"for {big}, {small}" in prefix
    finally:
        e.shutdown()


# ── /new keeps index nodes ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("retain,carried_depths", [(-1, [0, 1, 2, 3]), (0, []), (2, [2, 3])])
def test_new_session_keeps_index_nodes(tmp_path, retain, carried_depths):
    e = _engine(tmp_path, f"retain-{retain}.db", new_session_retain_depth=retain)
    try:
        e._session_id = "old"
        for d in range(4):
            _add(e, "old", d, f"d{d} summary", time.time())
        e.on_session_reset()
        # nothing deleted
        assert len(e._dag.get_session_nodes("old")) == 4
        moved = e.carry_over_new_session_context("old", "new")
        assert moved == len(carried_depths)
        assert sorted(n.depth for n in e._dag.get_session_nodes("new")) == carried_depths
        left = sorted(n.depth for n in e._dag.get_session_nodes("old"))
        assert left == [d for d in range(4) if d not in carried_depths]
        assert len(e._dag.get_session_nodes("old")) + len(e._dag.get_session_nodes("new")) == 4
    finally:
        e.shutdown()


# ── rotate ─────────────────────────────────────────────────────────────────────────────────

def test_rotate_leaves_marker_node(tmp_path):
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "rotate.db")
    cfg.fresh_tail_count = 3
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path / "home"))
    try:
        e._session_id = "live"
        e._session_platform = "cli"
        e._conversation_id = "live"
        e._lifecycle.bind_session("live", conversation_id="live")
        e.context_length = 200_000
        for i in range(10):
            e._store.append("live", {"role": "user", "content": f"msg-{i} " + "x" * 100}, source="test")
        e._store._conn.commit()
        result = e.rotate_active_session(apply=True)
        assert result["ok"] is True and result["noop"] is False
        assert "marker_node_id" in result
        nodes = e._dag.get_session_nodes("live")
        assert len(nodes) == 1
        marker = nodes[0]
        assert marker.summary.startswith(marked_loss.ROTATE_MARKER_PREFIX)
        assert marker.source_ids and max(marker.source_ids) == result["new_frontier_store_id"]
        assert "msg-0" in marker.summary
        assert "Expand for details about:" in marker.summary
        # rendered into the prefix
        assembled = e._assemble_context(None, [{"role": "user", "content": "tail"}])
        assert marked_loss.ROTATE_MARKER_PREFIX in assembled[0]["content"]
        # rotating again is a no-op: no second marker
        again = e.rotate_active_session(apply=True)
        assert again["noop"] is True
        assert len(e._dag.get_session_nodes("live")) == 1
    finally:
        e.shutdown()


# ── bypass ─────────────────────────────────────────────────────────────────────────────────

def test_bypass_trim_is_marked(tmp_path):
    e = _engine(tmp_path)
    try:
        messages = [
            {"role": "user", "content": "x" * 10_000},
            {"role": "assistant", "content": "y" * 10_000},
        ]
        result = e._trim_bypass_compacted_to_cap(messages, target_tokens=400)
        assert count_messages_tokens(result) <= 400
        text = "\n".join(str(m.get("content")) for m in result)
        assert "[LCM bypass trim" in text or "[LCM cut]" in text
    finally:
        e.shutdown()


# ── cleanup-only compress below threshold ──────────────────────────────────────────────────

def test_cleanup_request_below_threshold_runs_no_leaf_pass_and_keeps_ignore_drops(tmp_path, monkeypatch, ignore_patterns_engine):
    from hermes_lcm import escalation
    e = _engine(
        tmp_path, "cleanup.db",
        fresh_tail_count=1, leaf_chunk_tokens=10,
        ignore_message_patterns=[r"secret-token-[0-9]+"],
    )
    try:
        e._set_context_length(200_000, source="test")
        e.threshold_tokens = 100_000
        calls = []
        monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: calls.append(1) or "s")
        messages = [
            {"role": "user", "content": "backlog " + "b" * 400},
            {"role": "user", "content": "please ignore secret-token-42"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "fresh tail"},
        ]
        # the host's preflight sees replay != messages (ignore placeholder) and requests
        # a compress() while under the threshold: cleanup only
        assert e.should_compress_preflight(messages) is True
        result = e.compress(messages, current_tokens=5_000)
        assert calls == []
        assert e._last_compression_noop_reason.startswith("below threshold: cleanup only") or e._last_compression_status == "sanitized"
        assert e._dag.get_session_nodes("marked-session") == []
        text = "\n".join(str(m.get("content")) for m in result)
        assert "secret-token-42" not in text
        assert "backlog " + "b" * 400 in text  # raw kept, nothing summarised
    finally:
        e.shutdown()


def test_below_threshold_compress_without_cleanup_returns_input_identity(tmp_path, monkeypatch):
    from hermes_lcm import escalation
    e = _engine(tmp_path, "identity.db", fresh_tail_count=1, leaf_chunk_tokens=10)
    try:
        e._set_context_length(200_000, source="test")
        e.threshold_tokens = 100_000
        monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: pytest.fail("no summariser call"))
        messages = [
            {"role": "user", "content": "backlog " + "b" * 400},
            {"role": "user", "content": "fresh tail"},
        ]
        # default config: replay == messages, preflight does not request anything under
        # the threshold, and a compress() that changes nothing hands back the same object
        assert e.should_compress_preflight(messages) is False
        e._preflight_cleanup_only_below_threshold = True  # as if the replay branch asked
        result = e.compress(messages, current_tokens=5_000)
        assert result is messages
    finally:
        e.shutdown()


def test_direct_compress_below_threshold_still_summarises(tmp_path):
    """Only the preflight's replay-diff request is cleanup-only; a direct compress() (the
    host decided, or a test drives it) keeps upstream semantics."""
    e = _engine(tmp_path, "direct.db", fresh_tail_count=1, leaf_chunk_tokens=10)
    try:
        e._set_context_length(200_000, source="test")
        e.threshold_tokens = 100_000
        messages = [
            {"role": "user", "content": "backlog " + "b" * 400},
            {"role": "user", "content": "fresh tail"},
        ]
        e.compress(messages, current_tokens=5_000)
        assert len(e._dag.get_session_nodes("marked-session")) == 1
    finally:
        e.shutdown()


def test_above_threshold_compress_still_summarises(tmp_path):
    e = _engine(tmp_path, "above.db", fresh_tail_count=1, leaf_chunk_tokens=10)
    try:
        e._set_context_length(200_000, source="test")
        e.threshold_tokens = 10
        messages = [
            {"role": "user", "content": "backlog " + "b" * 400},
            {"role": "user", "content": "fresh tail"},
        ]
        e.compress(messages, current_tokens=5_000)
        assert len(e._dag.get_session_nodes("marked-session")) == 1
    finally:
        e.shutdown()


def test_summariser_failure_with_zero_passes_still_publishes_cleanup(tmp_path, monkeypatch, ignore_patterns_engine):
    from hermes_lcm import escalation
    e = _engine(
        tmp_path, "fail-cleanup.db",
        fresh_tail_count=1, leaf_chunk_tokens=10,
        ignore_message_patterns=[r"secret-token-[0-9]+"],
    )
    try:
        e._set_context_length(200_000, source="test")
        e.threshold_tokens = 10
        monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: None)
        messages = [
            {"role": "user", "content": "backlog " + "b" * 400},
            {"role": "user", "content": "please ignore secret-token-42"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "fresh tail"},
        ]
        result = e.compress(messages, current_tokens=5_000)
        assert e._dag.get_session_nodes("marked-session") == []
        text = "\n".join(str(m.get("content")) for m in result)
        assert "secret-token-42" not in text
        assert e.get_active_compression_failure_cooldown() is not None
        assert e.should_compress(5_000) is False
    finally:
        e.shutdown()


# ── audit A1 / p02 / p11: a carried-over node must expand through its own lineage ───────────

def test_carried_over_node_expands_through_children_in_the_old_session(tmp_path):
    """`/new` carries depth >= retain into the new session and leaves the shallower nodes with
    the old one. Expansion must follow the recorded lineage across that boundary; requiring
    every descendant to belong to the current session made a retained parent report
    "no children, has_more=false" — an index that reads complete and is empty.
    """
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "carry.db", new_session_retain_depth=2)
    try:
        e._session_id = "old"
        store_id = e._store.append("old", {"role": "user", "content": "the original decision"}, source="test")
        e._store._conn.commit()
        leaf = e._dag.add_node(SummaryNode(session_id="old", depth=0, summary="d0 leaf",
                                           token_count=5, source_token_count=9, source_ids=[store_id],
                                           source_type="messages", created_at=time.time()))
        mid = e._dag.add_node(SummaryNode(session_id="old", depth=1, summary="d1 mid",
                                          token_count=5, source_token_count=9, source_ids=[leaf],
                                          source_type="nodes", created_at=time.time()))
        top = e._dag.add_node(SummaryNode(session_id="old", depth=2, summary="d2 top",
                                          token_count=5, source_token_count=9, source_ids=[mid],
                                          source_type="nodes", created_at=time.time()))
        e.on_session_reset()
        assert e.carry_over_new_session_context("old", "new") == 1
        e._session_id = "new"

        expanded = json.loads(lcm_tools.lcm_expand({"node_id": top}, engine=e))
        assert [c["node_id"] for c in expanded["expanded"]] == [mid], expanded
        assert expanded["pagination"]["returned_sources"] == 1

        deeper = json.loads(lcm_tools.lcm_expand({"node_id": mid}, engine=e))
        assert [c["node_id"] for c in deeper["expanded"]] == [leaf]

        raw = json.loads(lcm_tools.lcm_expand({"node_id": leaf}, engine=e))
        assert "the original decision" in json.dumps(raw["expanded"])
    finally:
        e.shutdown()


def test_expansion_reports_a_recorded_child_that_no_longer_exists(tmp_path):
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "missing.db")
    try:
        e._session_id = "s"
        parent = e._dag.add_node(SummaryNode(session_id="s", depth=1, summary="parent",
                                             token_count=5, source_token_count=9,
                                             source_ids=[999_999], source_type="nodes",
                                             created_at=time.time()))
        result = json.loads(lcm_tools.lcm_expand({"node_id": parent}, engine=e))
        assert result["expanded"] == []
        assert result["pagination"]["incomplete"] is True
        assert result["pagination"]["missing_source_node_ids"] == [999_999]
    finally:
        e.shutdown()


def test_unrelated_sessions_are_still_refused(tmp_path):
    """Reachability, not a free-for-all: a node with no ancestor in the current session stays
    unexpandable through the current-session tools."""
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "unrelated.db")
    try:
        e._session_id = "mine"
        stranger = e._dag.add_node(SummaryNode(session_id="someone-else", depth=0, summary="private",
                                               token_count=5, source_token_count=9, source_ids=[],
                                               source_type="messages", created_at=time.time()))
        result = json.loads(lcm_tools.lcm_expand({"node_id": stranger}, engine=e))
        assert "error" in result and "not found in current session" in result["error"]
        assert "private" not in json.dumps(result)
    finally:
        e.shutdown()


def test_rotate_refuses_to_advance_when_its_marker_cannot_be_written(tmp_path, monkeypatch):
    """Audit A2: rotate advanced the frontier first and only logged a failed marker write, so
    the next bootstrap skipped raw rows with nothing pointing at them — and a retry was a
    no-op because the frontier was already ahead."""
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "rotate-fail.db")
    cfg.fresh_tail_count = 2
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path / "home"))
    try:
        e._session_id = e._conversation_id = "live"
        e._session_platform = "cli"
        e._lifecycle.bind_session("live", conversation_id="live")
        e.context_length = 200_000
        for i in range(8):
            e._store.append("live", {"role": "user", "content": f"m{i} " + "x" * 80}, source="test")
        e._store._conn.commit()
        before = e._lifecycle.get_by_conversation("live").current_frontier_store_id

        monkeypatch.setattr(e, "_write_rotate_marker_node", lambda *a, **k: None)
        result = e.rotate_active_session(apply=True)

        assert result["ok"] is False and result["reason"] == "marker_write_failed"
        after = e._lifecycle.get_by_conversation("live").current_frontier_store_id
        assert after == before, "the frontier must not move without its marker"

        # and with the marker working, the retry succeeds and does advance
        monkeypatch.undo()
        retry = e.rotate_active_session(apply=True)
        assert retry["ok"] is True and retry["noop"] is False and "marker_node_id" in retry
        assert e._lifecycle.get_by_conversation("live").current_frontier_store_id > before
    finally:
        e.shutdown()


# ── audit B9 / p02 / p05: a failed search is not an empty one ───────────────────────────────

def test_search_failure_is_reported_not_returned_as_no_matches(tmp_path, monkeypatch):
    import json
    import sqlite3
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "searchfail.db")
    try:
        e.on_session_start("sf", platform="cli", context_length=200_000)
        e._store.append("sf", {"role": "user", "content": "the decision about redis"}, source="cli")
        e._store._conn.commit()

        ok = json.loads(lcm_tools.lcm_grep({"query": "redis"}, engine=e))
        assert ok["complete"] is True and "search_failures" not in ok

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(e._store, "search", boom)
        monkeypatch.setattr(e._dag, "search", boom)
        degraded = json.loads(lcm_tools.lcm_grep({"query": "redis"}, engine=e))
        assert degraded["results"] == [] and degraded["total_results"] == 0
        assert degraded["complete"] is False
        assert {f["source"] for f in degraded["search_failures"]} == {"messages", "summaries"}
        assert "not evidence of absence" in degraded["search_note"]
    finally:
        e.shutdown()


def test_transcript_gc_never_rewrites_a_row_against_a_sanitized_only_payload(tmp_path):
    """Audit p01 E12 / B21: GC looked up a SANITIZED copy of the row first and accepted that
    match, so a payload holding only the sanitized text authorised replacing the row with a
    reference to it. Sanitisation removes whole injected blocks — what it removed would then
    exist nowhere. Failing to GC is a missed optimisation; GC against a partial payload is
    unrecoverable loss.
    """
    from hermes_lcm.externalize import maybe_externalize_tool_output
    from hermes_lcm.extraction import sanitize_pre_compaction_content
    e = _engine(
        tmp_path, "gc.db",
        large_output_externalization_enabled=True,
        # high threshold: ingest must NOT externalize the row itself — this test is about the
        # GC step deciding whether an existing payload may replace a row that still holds raw
        large_output_externalization_threshold_chars=1_000_000,
        large_output_transcript_gc_enabled=True,
    )
    try:
        e.on_session_start("gc", platform="cli", context_length=200_000)
        original = ("<active_memory>injected block</active_memory>\n"
                    "THE DECISION WAS TO CANCEL THE LAUNCH\n" + "z" * 400)
        sanitized = sanitize_pre_compaction_content(original)
        assert sanitized != original, "fixture must exercise the sanitising path"

        store_id = e._store.append("gc", {"role": "tool", "tool_call_id": "c1",
                                          "content": original}, source="cli")
        e._store._conn.commit()
        # a payload that holds only the SANITIZED text
        assert maybe_externalize_tool_output(sanitized, tool_call_id="c1", session_id="gc",
                                             config=e._config, hermes_home=e._hermes_home,
                                             force=True) is not None

        chunk = [{"role": "tool", "tool_call_id": "c1", "content": original}]
        e._maybe_gc_compacted_tool_results(chunk, [store_id])
        row = e._store.get(store_id)
        assert row["content"] == original, "the row must survive: no payload holds its bytes"

        # once the payload really does hold the original, GC proceeds
        assert maybe_externalize_tool_output(original, tool_call_id="c1", session_id="gc",
                                             config=e._config, hermes_home=e._hermes_home,
                                             force=True) is not None
        e._maybe_gc_compacted_tool_results(chunk, [store_id])
        assert e._store.get(store_id)["content"] != original
    finally:
        e.shutdown()


def test_out_of_order_tool_results_are_reordered_not_discarded(tmp_path):
    """Audit p01 E10 / audit A #9: results arriving in a different order than the calls were
    made made the repair drop the real result and insert a stub reading "see context summary
    above" — with nothing proving any summary covered it."""
    e = _engine(tmp_path, "toolpairs.db")
    try:
        messages = [
            {"role": "assistant", "content": "two calls", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "t", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "t", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "b", "content": "REAL RESULT B"},
            {"role": "tool", "tool_call_id": "a", "content": "REAL RESULT A"},
        ]
        sanitized = e._sanitize_active_context_messages(messages)
        blob = "\n".join(str(m.get("content")) for m in sanitized)
        assert "REAL RESULT A" in blob and "REAL RESULT B" in blob
        assert "see context summary above" not in blob
        # provider contract still holds: results follow their call, in call order
        assert [m.get("tool_call_id") for m in sanitized if m.get("role") == "tool"] == ["a", "b"]
    finally:
        e.shutdown()


def test_user_text_quoting_a_summary_header_is_still_stored(tmp_path):
    """Audit A #2 / p01 E11: the scaffold classifier decides whether a message is EXCLUDED
    FROM STORAGE, and it matched the header anywhere in any message — so a user pasting a
    summary block to ask about it was silently never stored."""
    e = _engine(tmp_path, "scaffold.db")
    try:
        e._session_id = "s"
        real = e._dag.add_node(SummaryNode(session_id="s", depth=0, summary="real",
                                           token_count=5, source_token_count=9, source_ids=[],
                                           source_type="messages", created_at=time.time()))
        quoting_user = {"role": "user", "content":
                        "why did you write [Recent Summary (d0, node 999)] like this?\n"
                        "[Expand for details: something]"}
        assert e._is_replayed_context_scaffold_message(quoting_user) is False

        embedded = {"role": "user", "content":
                    f"debug this: [Recent Summary (d0, node {real})] ... [Expand for details: x]"}
        assert e._is_replayed_context_scaffold_message(embedded) is False, "header must START it"

        trailing = {"role": "user", "content":
                    f"[Recent Summary (d0, node {real})]\nbody\n[Expand for details: x]\n"
                    "^ this is what you produced. why is the decision missing?"}
        assert e._is_replayed_context_scaffold_message(trailing) is False, "trailing text is theirs"

        our_own = {"role": "user", "content":
                   f"[Recent Summary (d0, node {real})]\nbody\n[Expand for details: x]"}
        assert e._is_replayed_context_scaffold_message(our_own) is True

        # an UNBACKED marker replayed from an earlier process is still our scaffolding: it must
        # be dropped, never re-summarised as if it were raw conversation
        unbacked = {"role": "assistant", "content":
                    "[Recent Summary (d0, node 999999)]\nbody\n[Expand for details: x]"}
        assert e._is_replayed_context_scaffold_message(unbacked) is True

        # an expand hint that contains a bracket of its own is still our scaffolding: matching
        # the trailer with a "no closing bracket" pattern made the whole prefix look like the
        # user's own text, so it was re-ingested and stored as raw conversation
        bracketed = {"role": "user", "content":
                     f"[Recent Summary (d0, node {real})]\nbody\n"
                     "[Expand for details: Expand for details about: items[0]]"}
        assert e._is_replayed_context_scaffold_message(bracketed) is True

        # ... and so is a prefix whose last part is the assembly omission marker
        omitted = {"role": "user", "content":
                   f"[Recent Summary (d0, node {real})]\nbody\n[Expand for details: x]\n\n---\n\n"
                   + marked_loss.assembly_omission_marker(
                       omitted_node_ids=[7], depth_cap_hits=[], omitted_tail_messages=0)}
        assert e._is_replayed_context_scaffold_message(omitted) is True
    finally:
        e.shutdown()


def test_paged_tool_calls_are_charged_to_the_budget_and_can_be_continued(tmp_path):
    """verify-1 on p02 T04: the rendered tool calls were not charged to the expansion budget,
    so every call-only row got the whole remaining allowance, and the advertised continuation
    pointed at raw-store expansion, which does not render tool calls at all."""
    import json
    from hermes_lcm import tools as lcm_tools
    from hermes_lcm.tokens import count_tokens
    e = _engine(tmp_path, "t04page.db", incremental_max_depth=0)
    try:
        e.on_session_start("t4", platform="cli", context_length=200_000)
        store_ids = []
        for index in range(3):
            store_ids.append(e._store.append("t4", {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"c{index}", "type": "function", "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": f"deploy-{index} " + "x" * 3000})}}],
            }, source="cli"))
        e._store.commit()
        node_id = e._dag.add_node_with_meta(SummaryNode(
            session_id="t4", depth=0, summary="three deploys\n[Expand for details: deploys]",
            token_count=5, source_token_count=900, source_ids=store_ids,
            source_type="messages", created_at=time.time()), level=1)

        payload = json.loads(lcm_tools.lcm_expand({"node_id": node_id, "max_tokens": 40}, engine=e))
        returned = sum(count_tokens(str(m.get("content") or "")) + count_tokens(str(m.get("tool_calls") or ""))
                       for m in payload["expanded"])
        assert returned <= 40 * 3, f"the budget was multiplied: {returned} tokens for 40"
        assert payload["pagination"]["has_more"] is True

        # and the continuation really returns the rest of the calls
        first = payload["expanded"][0]
        assert first.get("tool_calls_truncated") is True
        continuation = first["tool_calls_continue_with"]
        assert continuation["tool"] == "lcm_expand" and continuation["node_id"] == node_id
        second = json.loads(lcm_tools.lcm_expand(
            {k: v for k, v in continuation.items() if k != "tool"} | {"max_tokens": 400},
            engine=e,
        ))
        assert second["expanded"][0]["tool_calls"], "the continuation returned no tool calls"
        assert second["expanded"][0]["tool_calls_offset"] == continuation["tool_calls_offset"]
    finally:
        e.shutdown()


def test_expansion_returns_the_tool_calls_an_assistant_made(tmp_path):
    """Audit p02 T04 / audit A #11: an assistant turn that is mostly tool-call arguments
    expanded as EMPTY content with has_more:false — a recovery path reporting success while
    returning nothing of what the agent actually did."""
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "toolcalls.db")
    try:
        e.on_session_start("tc", platform="cli", context_length=200_000)
        store_id = e._store.append("tc", {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "terminal", "arguments": '{"command": "rm -rf /srv//important"}'}}],
        }, source="cli")
        e._store._conn.commit()
        node_id = e._dag.add_node(SummaryNode(
            session_id="tc", depth=0, summary="ran a command", token_count=5,
            source_token_count=20, source_ids=[store_id], source_type="messages",
            created_at=time.time()))
        result = json.loads(lcm_tools.lcm_expand({"node_id": node_id}, engine=e))
        blob = json.dumps(result["expanded"])
        assert "terminal" in blob and "important" in blob, result
    finally:
        e.shutdown()


def test_recent_reports_an_unscannable_window_instead_of_an_empty_one(tmp_path, monkeypatch):
    """Audit p02 T16: more matches than the work cap, or a read failure, returned a bare [] —
    the serializer then reported zero sections with truncated=false, an exhaustive negative
    over history that exists."""
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "recent.db")
    try:
        e.on_session_start("r", platform="cli", context_length=200_000)
        monkeypatch.setattr(lcm_tools, "_recent_leaf_sections",
                            lambda *a, **k: (_ for _ in ()).throw(
                                lcm_tools._RecentIncomplete("more than 4096 summaries match")))
        payload = json.loads(lcm_tools.lcm_recent({"period": "today"}, engine=e))
        assert payload["complete"] is False
        assert "4096" in payload["incomplete_reason"]

        monkeypatch.setattr(lcm_tools, "_recent_leaf_sections",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("database is locked")))
        payload = json.loads(lcm_tools.lcm_recent({"period": "today"}, engine=e))
        assert payload["complete"] is False and "database is locked" in payload["incomplete_reason"]

        # fork: the helper now returns (sections, total_matching)
        monkeypatch.setattr(lcm_tools, "_recent_leaf_sections", lambda *a, **k: ([], 0))
        payload = json.loads(lcm_tools.lcm_recent({"period": "today"}, engine=e))
        assert payload["complete"] is True and "incomplete_reason" not in payload
    finally:
        e.shutdown()


def test_clean_refuses_to_delete_sources_a_retained_summary_still_references(tmp_path):
    """Audit p03 C01 / audit A #6: `/lcm clean` deleted a session's rows without checking
    incoming references, so cleaning the predecessor of a `/new` could strand a summary
    retained in the CURRENT session — the node survives, its lineage does not, and expanding
    it returns nothing. Backup-first does not help: the live database is still wrong."""
    from hermes_lcm import command as lcm_command
    e = _engine(tmp_path, "clean.db", new_session_retain_depth=2)
    try:
        e._session_id = "old"
        store_id = e._store.append("old", {"role": "user", "content": "the source"}, source="cli")
        e._store._conn.commit()
        leaf = e._dag.add_node(SummaryNode(session_id="old", depth=0, summary="leaf",
                                           token_count=5, source_token_count=9,
                                           source_ids=[store_id], source_type="messages",
                                           created_at=time.time()))
        top = e._dag.add_node(SummaryNode(session_id="old", depth=2, summary="retained",
                                          token_count=5, source_token_count=9,
                                          source_ids=[leaf], source_type="nodes",
                                          created_at=time.time()))
        e.on_session_reset()
        assert e.carry_over_new_session_context("old", "new") == 1
        e._session_id = "new"

        with pytest.raises(lcm_command.CleanupWouldBreakProvenance) as caught:
            lcm_command._delete_clean_candidates_atomically(e, {"old"})
        assert leaf in caught.value.node_ids

        # nothing was deleted, and the retained node is still expandable
        assert e._dag.get_node(leaf) is not None
        assert e._store.get(store_id)["content"] == "the source"
        assert e._dag.get_node(top) is not None
    finally:
        e.shutdown()


def test_a_restricted_toolset_alone_does_not_make_an_agent_auxiliary(tmp_path):
    """Audit p05 AX01: any agent whose toolsets were a subset of {"memory","skills"} was
    classified auxiliary, so a legitimate foreground agent restricted to those tools had its
    conversation never stored — storage bypassed without the operator excluding anything."""
    e = _engine(tmp_path, "aux.db")
    try:
        class _Foreground:
            enabled_toolsets = ["memory"]
            log_prefix = "[agent]"

        class _Subagent:
            enabled_toolsets = ["memory"]
            log_prefix = "[subagent-7]"

        class _HostMarkedChild:
            enabled_toolsets = ["memory", "skills"]
            log_prefix = "[agent]"
            parent_session_id = "parent-1"

        assert e._caller_is_auxiliary_agent_frame(_Foreground()) is False
        assert e._caller_is_auxiliary_agent_frame(_Subagent()) is True
        assert e._caller_is_auxiliary_agent_frame(_HostMarkedChild()) is True
    finally:
        e.shutdown()


def test_every_row_a_leaf_consumes_becomes_a_source_of_it(tmp_path, monkeypatch, ignore_patterns_engine):
    """Audit p05 CP01: replies to host-injected ignored messages are excluded from the
    summariser input on purpose, but upstream also left them out of the node's source_ids
    while sweeping them past the frontier — so the rows were consumed and then reachable
    from no node at all. They must be sources of the leaf, and marked as not summarised."""
    from hermes_lcm import engine as lcm_engine
    e = _engine(tmp_path, "cp01.db", fresh_tail_count=1, leaf_chunk_tokens=10,
                ignore_message_patterns=["SECRET"])
    try:
        e._session_id = "cp01"
        monkeypatch.setattr(lcm_engine, "summarize_with_escalation",
                            lambda **kw: ("visible summary\n[Expand for details: visible]", 1))
        messages = [
            {"role": "user", "content": "SECRET ignored backlog " + "x" * 200},
            {"role": "assistant", "content": "dependent reply to the ignored message"},
            {"role": "user", "content": "visible backlog " + "y" * 200},
            {"role": "assistant", "content": "fresh tail"},
        ]
        e.compress(messages, current_tokens=count_messages_tokens(messages))

        nodes = e._dag.get_session_nodes("cp01")
        assert nodes, "the leaf must have been published"
        node = nodes[0]
        rows = {row["store_id"]: str(row.get("content") or "")
                for row in e._store.get_session_messages("cp01")}
        dependent_ids = [sid for sid, text in rows.items() if "dependent reply" in text]
        assert dependent_ids, "the dependent reply must be stored"
        consumed = [sid for sid in dependent_ids if sid <= e._last_compacted_store_id]
        for store_id in consumed:
            assert store_id in node.source_ids, "a consumed row must be expandable from its leaf"
            assert str(store_id) in node.summary, "and named, not silently added"
        if consumed:
            assert "NOT summarised" in node.summary
            assert "dependent reply" not in node.summary, "excluded content stays out of the text"
    finally:
        e.shutdown()


def test_recent_reports_a_real_work_cap_hit_not_an_empty_window(tmp_path):
    """verify-1 on p02 T16: the work-cap signal was raised inside the helper and then caught
    by the helper's own broad handler, so a 4,097-node window still reported complete:true
    with zero sections. The signal has to reach the caller."""
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "recentcap.db", incremental_max_depth=0)
    try:
        e.on_session_start("rc", platform="cli", context_length=200_000)
        cap = lcm_tools._LCM_RECENT_FRONTIER_WORK_LIMIT
        now = time.time()
        nodes = [
            SummaryNode(session_id="rc", depth=0, summary=f"leaf {index}", token_count=3,
                        source_token_count=9, source_ids=[], source_type="messages",
                        created_at=now, earliest_at=now, latest_at=now)
            for index in range(cap + 1)
        ]
        for node in nodes:
            e._dag.add_node(node)

        payload = json.loads(lcm_tools.lcm_recent({"period": "today"}, engine=e))
        assert payload["complete"] is False
        assert str(cap) in payload["incomplete_reason"]
    finally:
        e.shutdown()


def test_an_orphan_tool_result_is_named_not_dropped(tmp_path):
    """verify-4 #10: an orphan result was dropped with only a log line, and a missing result
    got a stub claiming it was covered by "the context summary above" — with an empty DAG."""
    e = _engine(tmp_path, "orphan.db")
    try:
        messages = [
            {"role": "user", "content": "what happened?"},
            {"role": "tool", "tool_call_id": "gone", "content": "ACTION FAILED: disk full"},
        ]
        sanitized = e._sanitize_tool_pairs([dict(m) for m in messages])
        rendered = "\n".join(str(m.get("content") or "") for m in sanitized)
        assert "answered no call in this replay window" in rendered
        assert "ACTION FAILED" in rendered
        assert "lcm_grep" in rendered

        # the receipt must never become "the newest message": appended at the end it displaced
        # the live request under an assembly or bypass cap (round-2 verify-2 #1)
        with_request = e._sanitize_tool_pairs([
            {"role": "tool", "tool_call_id": "gone", "content": "ACTION FAILED"},
            {"role": "user", "content": "LATEST REQUEST: cancel the rollout"},
        ])
        assert "LATEST REQUEST" in str(with_request[-1].get("content") or "")

        unanswered = e._sanitize_tool_pairs([
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        ])
        stub = next(m for m in unanswered if m.get("role") == "tool")
        assert "not in the replayed window" in stub["content"]
        assert "context summary above" not in stub["content"]
    finally:
        e.shutdown()


def test_an_unreadable_source_row_is_reported_not_skipped(tmp_path):
    """verify-4 #15: expanding a leaf whose only source row was missing returned no messages,
    remaining_sources=0 and has_more=false — indistinguishable from an exhausted list."""
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "missing.db", incremental_max_depth=0)
    try:
        e.on_session_start("ms", platform="cli", context_length=200_000)
        node_id = e._dag.add_node_with_meta(SummaryNode(
            session_id="ms", depth=0, summary="a decision\n[Expand for details: it]",
            token_count=5, source_token_count=50, source_ids=[999999],
            source_type="messages", created_at=time.time()), level=1)
        payload = json.loads(lcm_tools.lcm_expand({"node_id": node_id}, engine=e))
        pagination = payload["pagination"]
        assert pagination["complete"] is False
        assert pagination["missing_source_store_ids"] == [999999]
        assert "could not be read" in pagination["incomplete_reason"]
    finally:
        e.shutdown()


def test_recent_says_the_database_was_unavailable_instead_of_empty(tmp_path):
    """verify-4 #13: with the DAG closed, lcm_recent returned zero sections, complete:true and
    truncated:false — a successful scan of a database it never opened."""
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "closeddag.db")
    try:
        e.on_session_start("cd", platform="cli", context_length=200_000)
        e._dag.close()
        payload = json.loads(lcm_tools.lcm_recent({"period": "today"}, engine=e))
        assert payload["complete"] is False
        assert "not available" in payload["incomplete_reason"]
    finally:
        try:
            e.shutdown()
        except Exception:
            pass


def test_a_tight_budget_gets_the_one_line_receipt_and_never_silence(tmp_path):
    """verify-4 #9: the receipt was the first thing dropped when the budget got tight — which
    is exactly when something HAS been omitted. It now degrades to a one-line form, and if
    even that cannot fit, the note is recorded in status rather than lost."""
    e = _engine(tmp_path, "receipt.db", incremental_max_depth=0)
    try:
        base = time.time()
        for index in range(6):
            _add(e, "marked-session", 0, f"summary number {index} " + "s" * 400, base + index)
        assembled = e._assemble_context(
            None, [{"role": "user", "content": "tail"}], assembly_cap_override=300,
        )
        rendered = "\n".join(str(m.get("content") or "") for m in assembled)
        assert "[LCM assembly omissions" in rendered, rendered[:400]
        assert "not rendered this turn" in rendered or "did not fit" in rendered

        # and under a budget too small even for the one-line form, the note survives in status
        e._assemble_context(
            None, [{"role": "user", "content": "tail"}], assembly_cap_override=40,
        )
        status = e.get_status()
        assert (
            "[LCM assembly omissions" in rendered
            or "[LCM assembly omissions" in status.get("last_assembly_omission_note", "")
        )
    finally:
        e.shutdown()


def test_a_retained_child_is_reachable_past_the_reverse_edge_cap(tmp_path):
    """verify-4 #16: the reverse-edge query stops at 256 parents, so a node whose legitimate
    current-session parent came after the first 256 was refused as absent from the session."""
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "reach.db", incremental_max_depth=0)
    try:
        e.on_session_start("new-session", platform="cli", context_length=200_000)
        child = e._dag.add_node(SummaryNode(
            session_id="old-session", depth=0, summary="a child\n[Expand for details: x]",
            token_count=5, source_token_count=50, source_ids=[1], source_type="messages",
            created_at=time.time()))
        for index in range(300):
            e._dag.add_node(SummaryNode(
                session_id="old-session", depth=1, summary=f"other parent {index}",
                token_count=5, source_token_count=50, source_ids=[child],
                source_type="nodes", created_at=time.time()))
        e._dag.add_node(SummaryNode(
            session_id="new-session", depth=1, summary="the retained parent",
            token_count=5, source_token_count=50, source_ids=[child],
            source_type="nodes", created_at=time.time()))

        assert lcm_tools._get_session_node(e, child) is not None
    finally:
        e.shutdown()


def test_every_assembled_prefix_shape_is_recognised_as_scaffolding(tmp_path):
    """round-2 verify-2 #5: the compact omission footer was not recognised, so an assembled
    prefix ending in it was ingested and stored as raw conversation."""
    from hermes_lcm import marked_loss
    e = _engine(tmp_path, "shapes.db", incremental_max_depth=0)
    try:
        e.on_session_start("sh", platform="cli", context_length=200_000)
        base = time.time()
        for index in range(4):
            _add(e, "sh", 0, f"summary {index} " + "s" * 300, base + index)
        # a budget that fits the summary and the one-line receipt but not the full sentence
        for cap in (120, 200, 320, 500):
            assembled = e._assemble_context(
                None, [{"role": "user", "content": "tail"}], assembly_cap_override=cap,
            )
            prefix = assembled[0]
            # whatever shape assembly chose, the engine must recognise it as its own
            assert e._is_replayed_context_scaffold_message(prefix) is True, (cap, prefix)
        compact = {
            "role": "user",
            "content": "[Recent Summary (d0, node 1)]\nbody\n\n---\n\n"
                       + marked_loss.compact_assembly_omission_marker(
                           omitted_node_ids=[2, 3], depth_cap_hits=[],
                           omitted_tail_messages=0),
        }
        assert e._is_replayed_context_scaffold_message(compact) is True
    finally:
        e.shutdown()


def test_a_candidate_that_exactly_fits_is_still_rendered(tmp_path):
    """round-2 verify-2 #4: the incremental packing estimate is not additive, so a summary
    that exactly fitted the budget was omitted by the estimate alone."""
    e = _engine(tmp_path, "exactfit.db", incremental_max_depth=0)
    try:
        e.on_session_start("ef", platform="cli", context_length=262_144)
        base = time.time()
        _add(e, "ef", 0, "first summary", base)
        _add(e, "ef", 0, "second summary", base + 1)
        exact = e._assemble_context(None, [{"role": "user", "content": "t"}])
        prefix = str(exact[0].get("content") or "")
        budget = count_messages_tokens([{"role": exact[0]["role"], "content": prefix}])
        again = e._assemble_context(
            None, [{"role": "user", "content": "t"}], assembly_cap_override=budget + 40,
        )
        rendered = "\n".join(str(m.get("content") or "") for m in again)
        assert "first summary" in rendered and "second summary" in rendered, rendered
    finally:
        e.shutdown()


def test_a_finished_body_is_not_re_sent_on_every_tool_call_page(tmp_path):
    """round-2 verify-2 #8: a finished body reported next_content_offset 0, and the paginator
    re-used that zero while staying on the same source — a 196-character body with a
    100-character call took 101 pages and returned 19,600 characters of body."""
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "bodypage.db", incremental_max_depth=0)
    try:
        e.on_session_start("bp", platform="cli", context_length=200_000)
        body = "b" * 196
        first = e._store.append("bp", {
            "role": "assistant", "content": body,
            "tool_calls": [{"id": "c0", "type": "function", "function": {
                "name": "t", "arguments": json.dumps({"cmd": "y" * 60})}}],
        }, source="cli")
        second = e._store.append("bp", {"role": "user", "content": "the next source"}, source="cli")
        e._store.commit()
        node_id = e._dag.add_node_with_meta(SummaryNode(
            session_id="bp", depth=0, summary="one call\n[Expand for details: call]",
            token_count=5, source_token_count=200, source_ids=[first, second],
            source_type="messages", created_at=time.time()), level=1)

        args = {"node_id": node_id, "max_tokens": 50}
        pages = 0
        body_chars = 0
        seen_second_source = False
        while pages < 40:
            payload = json.loads(lcm_tools.lcm_expand(dict(args), engine=e))
            pages += 1
            for message in payload["expanded"]:
                if message["store_id"] == first:
                    body_chars += len(message["content"])
                if message["store_id"] == second:
                    seen_second_source = True
            pagination = payload["pagination"]
            if not pagination.get("has_more"):
                break
            args = {"node_id": node_id, "max_tokens": 50,
                    "source_offset": pagination["next_source_offset"],
                    "content_offset": pagination.get("next_content_offset") or 0,
                    "tool_calls_offset": pagination.get("next_tool_calls_offset") or 0}
        assert seen_second_source, "the later source never came back"
        assert body_chars <= len(body), f"the body was re-sent: {body_chars} chars of {len(body)}"
        assert pages <= 8, f"{pages} pages to walk 196 characters and one call"
    finally:
        e.shutdown()


def test_recent_counts_the_window_before_the_display_limit(tmp_path, monkeypatch):
    """round-2 verify-3 #9: eleven matching sections with limit=10 reported total_sections=10
    and truncated=false — a window that had more in it read as fully shown. And a frontier
    computation that raised failed CLOSED, answering with an empty, complete window."""
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "recentcount.db", incremental_max_depth=0)
    try:
        e.on_session_start("rn", platform="cli", context_length=200_000)
        e._config.temporal_rollups_enabled = False
        now = time.time()
        for index in range(11):
            e._dag.add_node(SummaryNode(
                session_id="rn", depth=0, summary=f"leaf {index}", token_count=3,
                source_token_count=9, source_ids=[], source_type="messages",
                created_at=now - index, earliest_at=now - index, latest_at=now - index))

        payload = json.loads(lcm_tools.lcm_recent({"period": "today", "limit": 10}, engine=e))
        assert payload["complete"] is True
        assert payload["total_sections"] == 11, payload["total_sections"]
        assert payload["returned_sections"] == 10
        assert payload["truncated"] is True

        def explode(*a, **k):
            raise RuntimeError("frontier is broken")

        monkeypatch.setattr(lcm_tools, "canonical_frontier", explode)
        failed = json.loads(lcm_tools.lcm_recent({"period": "today", "limit": 10}, engine=e))
        assert failed["complete"] is False, failed
        assert "frontier" in failed["incomplete_reason"]
    finally:
        e.shutdown()


def test_a_user_message_that_quotes_a_receipt_and_adds_a_decision_is_stored(tmp_path):
    """round-2 verify-4 #2: an omission receipt ANYWHERE in a message beginning with a summary
    header was enough to classify it as our own scaffolding, so a user who pasted a summary,
    receipt and all, and then wrote their new decision under it had that message dropped —
    never stored, never summarised, referenced by no node."""
    from hermes_lcm import marked_loss
    e = _engine(tmp_path, "pasted.db", incremental_max_depth=0)
    try:
        e.on_session_start("pa", platform="cli", context_length=200_000)
        compact = marked_loss.compact_assembly_omission_marker(
            omitted_node_ids=[2, 3], depth_cap_hits=[], omitted_tail_messages=0)
        full = marked_loss.assembly_omission_marker(
            omitted_node_ids=[2], depth_cap_hits=[], omitted_tail_messages=1)
        for receipt in (compact, full):
            pasted = {
                "role": "user",
                "content": "[Recent Summary (d0, node 1)]\nthe summary body\n\n---\n\n"
                           + receipt + "\n\nMY NEW DECISION: cancel deployment.",
            }
            assert e._is_replayed_context_scaffold_message(pasted) is False, receipt
            # ... and our own prefix, which ends with the receipt, is still recognised
            ours = {"role": "user", "content":
                    "[Recent Summary (d0, node 1)]\nthe summary body\n\n---\n\n" + receipt}
            assert e._is_replayed_context_scaffold_message(ours) is True, receipt

        e._ingest_messages([{"role": "user", "content":
                             "[Recent Summary (d0, node 1)]\nbody\n\n---\n\n" + compact
                             + "\n\nMY NEW DECISION: cancel deployment."}])
        stored = e._store.search("cancel deployment", session_id="pa", limit=5)
        assert stored, "the user's decision was not stored"
    finally:
        e.shutdown()


def test_a_missing_tool_result_stub_only_promises_what_exists(tmp_path):
    """round-2 verify-4 #19: the stub told the reader the result was in the raw store and
    could be found by tool_call_id — for a call that had never received a result at all, over
    an empty store. Three different situations were being described with one sentence."""
    e = _engine(tmp_path, "stub.db", incremental_max_depth=0)
    try:
        e.on_session_start("st", platform="cli", context_length=200_000)
        never = e._sanitize_tool_pairs([
            {"role": "assistant", "tool_calls": [
                {"id": "never", "function": {"name": "t", "arguments": "{}"}}]},
        ])
        assert "none was ever received" in never[1]["content"], never[1]["content"]

        store_id = e._store.append("st", {"role": "tool", "tool_call_id": "archived",
                                          "content": "the real result"}, source="cli")
        e._store.commit()
        archived = e._sanitize_tool_pairs([
            {"role": "assistant", "tool_calls": [
                {"id": "archived", "function": {"name": "t", "arguments": "{}"}}]},
        ])
        assert f"lcm_expand(store_id={store_id})" in archived[1]["content"], archived[1]["content"]
    finally:
        e.shutdown()


def test_condensation_inherits_every_marker_spelling_not_only_the_colon_one(tmp_path):
    """round-2 verify-4 #18: inheritance recognised "[LCM:" alone, so a child carrying a rotate
    marker — the one that says its span was never summarised at all — condensed into a parent
    with no warning, and the parent read as an ordinary summary of summarised material."""
    from hermes_lcm import escalation, marked_loss
    e = _engine(tmp_path, "spelling.db", condensation_fanin=2, incremental_max_depth=2)
    try:
        e.on_session_start("sp", platform="cli", context_length=200_000)
        rotate = marked_loss.rotate_marker_summary(
            session_id="sp", store_ids=[1, 2], message_count=2, token_count=40,
            roles=["user", "assistant"], first_head="first", last_head="last")
        children = []
        for index, text in enumerate((rotate, "an ordinary leaf\nExpand for details about: x")):
            children.append(e._dag.add_node_with_meta(SummaryNode(
                session_id="sp", depth=0, summary=text, token_count=60, source_token_count=200,
                source_ids=[index + 1], source_type="messages",
                created_at=time.time() + index), level=1))
        original = escalation._call_llm_for_summary
        escalation._call_llm_for_summary = lambda *a, **k: "merged\nExpand for details about: merged"
        try:
            e._condense_summary_nodes([e._dag.get_node(node_id) for node_id in children])
        finally:
            escalation._call_llm_for_summary = original
        parent = next(n for n in e._dag.get_session_nodes("sp") if n.depth == 1)
        assert marked_loss.ROTATE_MARKER_PREFIX in parent.summary, parent.summary
    finally:
        e.shutdown()


def test_raw_row_expansion_renders_the_row_s_tool_calls(tmp_path):
    """round-2 verify-4 #20: lcm_expand(store_id=…) returned an assistant row's text with
    has_more=false and never mentioned the tool call stored on the same row — a recovery path
    answering "this is the whole row" while omitting what the agent actually did."""
    import json
    from hermes_lcm import tools as lcm_tools
    e = _engine(tmp_path, "rawcalls.db", incremental_max_depth=0)
    try:
        e.on_session_start("rw", platform="cli", context_length=200_000)
        store_id = e._store.append("rw", {
            "role": "assistant", "content": "reading the file",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "read", "arguments": json.dumps({"path": "/etc/hosts"})}}],
        }, source="cli")
        e._store.commit()
        payload = json.loads(lcm_tools.lcm_expand({"store_id": store_id}, engine=e))
        assert "read" in payload["tool_calls"], payload
        assert "/etc/hosts" in payload["tool_calls"]

        paged = json.loads(lcm_tools.lcm_expand(
            {"store_id": store_id, "max_tokens": 8}, engine=e))
        if paged.get("tool_calls_truncated"):
            continuation = paged["tool_calls_continue_with"]
            rest = json.loads(lcm_tools.lcm_expand(
                {k: v for k, v in continuation.items() if k != "tool"} | {"max_tokens": 400},
                engine=e))
            assert rest["tool_calls"], "the continuation returned no tool calls"
    finally:
        e.shutdown()


def test_a_budget_too_small_for_the_receipt_still_says_something_is_missing(tmp_path):
    """round-2 verify-4 #17: when neither the full nor the one-line receipt fitted, the receipt
    was dropped and kept only in _last_assembly_omission_note — the prefix then omitted content
    in silence, which is the one thing the receipt exists to prevent."""
    from hermes_lcm import marked_loss
    e = _engine(tmp_path, "tinybudget.db", incremental_max_depth=0)
    try:
        e.on_session_start("tb", platform="cli", context_length=200_000)
        base = time.time()
        for index in range(3):
            _add(e, "tb", 0, f"summary {index} " + "s" * 400, base + index)
        for cap in (40, 60, 120, 240):
            assembled = e._assemble_context(
                None, [{"role": "user", "content": "last"}], assembly_cap_override=cap)
            rendered = "\n".join(str(m.get("content") or "") for m in assembled)
            says_something = (
                marked_loss.MINIMAL_ASSEMBLY_OMISSION_MARKER in rendered
                or marked_loss.COMPACT_ASSEMBLY_OMISSION_PREFIX in rendered
                or marked_loss.ASSEMBLY_OMISSION_MARKER_HEADER in rendered
            )
            assert says_something, (cap, rendered)
    finally:
        e.shutdown()
