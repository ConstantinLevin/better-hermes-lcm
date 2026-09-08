"""Step 4 — no L3; summariser failure arms a host-visible cooldown and never kills the turn."""
import pytest

from hermes_lcm import engine as engine_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.errors import SummaryUnavailableError
from hermes_lcm.tokens import count_messages_tokens


def _engine(tmp_path, **kw):
    cfg = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1, **kw)
    cfg.database_path = str(tmp_path / "lcm.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e._session_id = "cooldown-session"
    e.threshold_tokens = 1
    return e


def _messages(n=4):
    return [{"role": "system", "content": "system"}] + [
        {"role": "user", "content": f"turn-{i} " + ("token " * 20)} for i in range(n)
    ] + [{"role": "user", "content": "fresh"}, {"role": "assistant", "content": "answer"}]


def test_compress_returns_input_identity_and_arms_cooldown_on_summariser_failure(tmp_path, monkeypatch):
    e = _engine(tmp_path)

    def always_fail(chunk, focus_topic=None, deadline=None):
        raise SummaryUnavailableError("provider down")

    monkeypatch.setattr(e, "_summarize_leaf_chunk_with_rescue", always_fail)
    msgs = _messages()
    try:
        out = e.compress(msgs, current_tokens=count_messages_tokens(msgs))
        assert out is msgs                                   # identity: host sees "not compressed"
        assert e._dag.get_session_nodes("cooldown-session") == []
        cd = e.get_active_compression_failure_cooldown()
        assert cd and cd["remaining_seconds"] > 0 and "provider down" in cd["error"]
        assert type(e)._automatic_compression_blocked(e) is True
        assert type(e)._automatic_compression_blocked(e, ignore_cooldown=True) is False
        assert e.should_compress(10_000) is False
        should, reason = e.should_compress_info(10_000)
        assert should is False and reason.startswith("cooldown:")
        assert e.compression_failure_status()["failure_count"] == 1
    finally:
        e.shutdown()


def test_partial_failure_publishes_persisted_passes_and_arms_cooldown(tmp_path, monkeypatch):
    # non-sweep, dynamic chunking: pass 1 persists, pass 2 fails -> progress published, cooldown armed
    e = _engine(tmp_path, dynamic_leaf_chunk_enabled=True)
    calls = 0

    def flaky(chunk, focus_topic=None, deadline=None):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise SummaryUnavailableError("later summary failed")
        return chunk, count_messages_tokens(chunk), "persisted first pass", 1, 0

    monkeypatch.setattr(e, "_summarize_leaf_chunk_with_rescue", flaky)
    msgs = _messages(6)
    try:
        out = e.compress(msgs, current_tokens=count_messages_tokens(msgs))
        assert out is not msgs                               # progress was published
        assert len(e._dag.get_session_nodes("cooldown-session")) == 1
        assert e.get_active_compression_failure_cooldown() is not None
    finally:
        e.shutdown()


def test_force_clears_cooldown_and_spend_guard(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    e._record_compression_failure("boom")
    e._summary_spend_guard.record_call()
    assert e.should_compress(10_000) is False
    monkeypatch.setattr(e, "_summarize_leaf_chunk_with_rescue",
                        lambda chunk, focus_topic=None, deadline=None: (chunk, count_messages_tokens(chunk), "ok", 1, 0))
    msgs = _messages()
    try:
        e.compress(msgs, current_tokens=count_messages_tokens(msgs), force=True)
        assert e.get_active_compression_failure_cooldown() is None
        assert e._summary_spend_guard.allows()
    finally:
        e.shutdown()


def test_preflight_still_ingests_during_cooldown(tmp_path):
    e = _engine(tmp_path)
    e._record_compression_failure("boom")
    msgs = _messages()
    try:
        assert e.should_compress_preflight(msgs) is False
        # ingestion happened regardless of the cooldown
        assert e._store.get_session_count("cooldown-session") > 0
    finally:
        e.shutdown()


def test_cooldown_expires(tmp_path, monkeypatch):
    e = _engine(tmp_path, summary_failure_cooldown_seconds=0.05)
    e._record_compression_failure("boom")
    assert e.get_active_compression_failure_cooldown() is not None
    import time
    time.sleep(0.08)
    assert e.get_active_compression_failure_cooldown() is None
    e.shutdown()


def test_unexpected_exception_still_propagates(tmp_path, monkeypatch):
    # Only summariser unavailability is absorbed; real bugs stay loud.
    e = _engine(tmp_path)

    def bug(chunk, focus_topic=None, deadline=None):
        raise ValueError("a real bug")

    monkeypatch.setattr(e, "_summarize_leaf_chunk_with_rescue", bug)
    msgs = _messages()
    try:
        with pytest.raises(ValueError):
            e.compress(msgs, current_tokens=count_messages_tokens(msgs))
    finally:
        e.shutdown()


def test_cancellation_propagates_through_call_llm(monkeypatch):
    # AuxiliaryExplicitCancellation is a BaseException: `except Exception` must not swallow it.
    import sys
    import types
    from hermes_lcm import escalation
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    def cancel(**kwargs):
        raise AuxiliaryExplicitCancellation()

    stub = types.ModuleType("agent.auxiliary_client")
    stub.call_llm = cancel
    stub.AuxiliaryExplicitCancellation = AuxiliaryExplicitCancellation
    # a test-installed aux client makes the conftest mock delegate to the real call path
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", stub)
    with pytest.raises(AuxiliaryExplicitCancellation):
        escalation._call_llm_for_summary("prompt", 10)


def test_no_truncation_text_ever_reaches_the_dag(tmp_path, monkeypatch):
    from hermes_lcm import escalation
    e = _engine(tmp_path)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: None)
    msgs = _messages()
    try:
        e.compress(msgs, current_tokens=count_messages_tokens(msgs))
        for node in e._dag.get_session_nodes("cooldown-session"):
            assert "deterministic truncation" not in node.summary
        assert e._dag.get_session_nodes("cooldown-session") == []
    finally:
        e.shutdown()


def test_host_call_shapes_for_the_cooldown_protocol(tmp_path, monkeypatch):
    """The exact ways hermes-agent reads a plugin engine's cooldown state
    (conversation_compression._refresh_persisted_compression_guards /
    _automatic_compression_gate_blocks, turn_context_compaction._blocked_compress_reason,
    turn_preflight, _codex_compaction_cooldown_remaining)."""
    import inspect
    from hermes_lcm import escalation
    e = _engine(tmp_path)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: None)
    try:
        # clear: every shape says "not blocked"
        assert getattr(type(e), "get_active_compression_failure_cooldown")(e, refresh=True) is None
        assert getattr(e, "get_active_compression_failure_cooldown", lambda: None)() is None
        blocked = getattr(type(e), "_automatic_compression_blocked")
        assert "ignore_cooldown" in inspect.signature(blocked).parameters
        assert blocked(e, ignore_cooldown=True) is False and blocked(e) is False
        assert e.should_compress_info(10)[0] is True
        # arm it the way the host would: a compress() whose summariser is dead
        msgs = _messages()
        assert e.compress(msgs, current_tokens=count_messages_tokens(msgs)) is msgs
        state = getattr(type(e), "get_active_compression_failure_cooldown")(e, refresh=True)
        assert state and float(state["remaining_seconds"]) > 0
        assert blocked(e) is True
        assert blocked(e, ignore_cooldown=True) is False  # manual paths bypass the cooldown
        should, reason = e.should_compress_info(10)
        assert should is False and reason.startswith("cooldown:")
        getter = getattr(e, "get_active_compression_failure_cooldown", None)
        assert float(getter(refresh=True).get("remaining_seconds")) > 0
    finally:
        e.shutdown()


def test_length_rejection_is_named_in_the_error(monkeypatch):
    from hermes_lcm import escalation
    # every route answers, but never shorter than the source
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: "word " * 400)
    with pytest.raises(SummaryUnavailableError, match="not shorter than the 20-token source"):
        escalation.summarize_with_escalation(text="short source", source_tokens=20, token_budget=2000, depth=0)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: None)
    with pytest.raises(SummaryUnavailableError, match="summariser unavailable after L1/L2"):
        escalation.summarize_with_escalation(text="short source", source_tokens=20, token_budget=2000, depth=0)


def test_cooldown_is_scoped_to_the_session_that_failed(tmp_path, monkeypatch):
    """Audit A #16: the deadline lives on the engine and the engine outlives a session, so an
    unrelated next session inherited a block it never earned."""
    from hermes_lcm import escalation
    e = _engine(tmp_path)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: None)
    try:
        msgs = _messages()
        e.compress(msgs, current_tokens=count_messages_tokens(msgs))
        assert e.get_active_compression_failure_cooldown() is not None
        assert e.should_compress(10_000) is False

        e._session_id = "a-different-session"          # /new, or a foreground rebind
        assert e.get_active_compression_failure_cooldown() is None
        assert e.should_compress(10_000) is not False

        e._session_id = "cooldown-session"             # back to the one that failed
        assert e.get_active_compression_failure_cooldown() is not None
    finally:
        e.shutdown()


def test_session_reset_clears_the_cooldown(tmp_path, monkeypatch):
    from hermes_lcm import escalation
    e = _engine(tmp_path)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: None)
    try:
        msgs = _messages()
        e.compress(msgs, current_tokens=count_messages_tokens(msgs))
        assert e.get_active_compression_failure_cooldown() is not None
        e.on_session_reset()
        assert e.get_active_compression_failure_cooldown() is None
    finally:
        e.shutdown()


def test_condensation_failure_publishes_leaf_progress_instead_of_discarding_it(tmp_path, monkeypatch):
    """Audit D #1: leaf passes are already committed and the raw cursor has advanced when
    condensation runs. Letting a condensation failure escape returned the ORIGINAL prompt while
    the DAG had moved on, and the next attempt published a leaf with empty provenance.
    Upstream never reached this state because its L3 fallback always converged.
    """
    from hermes_lcm import engine as engine_mod
    from hermes_lcm.errors import SummaryUnavailableError
    e = _engine(tmp_path, condensation_fanin=1, incremental_max_depth=3)
    try:
        monkeypatch.setattr(engine_mod, "summarize_with_escalation",
                            lambda **kw: ("leaf summary\nExpand for details about: x", 1))
        monkeypatch.setattr(e, "_condense_summary_nodes",
                            lambda *a, **k: (_ for _ in ()).throw(SummaryUnavailableError("dead")))
        msgs = _messages(6)
        result = e.compress(msgs, current_tokens=count_messages_tokens(msgs))

        leaves = [n for n in e._dag.get_session_nodes("cooldown-session") if n.depth == 0]
        assert leaves, "the leaf that succeeded must be kept"
        assert all(n.source_ids for n in leaves), "no node may be published without provenance"
        assert result is not msgs, "the caller must get the compacted context, not the stale one"
        assert e.get_active_compression_failure_cooldown() is not None
    finally:
        e.shutdown()
