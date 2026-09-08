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
