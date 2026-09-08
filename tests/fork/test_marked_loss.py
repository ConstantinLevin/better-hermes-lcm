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
