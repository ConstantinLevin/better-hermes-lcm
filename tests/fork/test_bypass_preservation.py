"""#62: a session LCM does not store is never shortened by LCM.

Ignored, stateless and in-process auxiliary contexts reach the plugin but are never written to
the store, so anything LCM removed from one would be unrecoverable by construction — no raw
row, no summary node, no marker that could lead anywhere. The fork therefore removes nothing
on this path: the host's own compressor is the only component allowed to shorten these
sessions, and when it cannot, the context comes back exactly as it arrived and the turn is
reported through the host's own "compression aborted, no messages were dropped" channel.
"""
import copy
from types import ModuleType
import sys

import pytest

from hermes_lcm import marked_loss
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _no_native_compressor(monkeypatch):
    """Make Hermes' native compressor genuinely unavailable, the way a non-Hermes host is."""
    agent_module = sys.modules.get("agent") or ModuleType("agent")
    if not hasattr(agent_module, "__path__"):
        agent_module.__path__ = []
    compressor_module = ModuleType("agent.context_compressor")  # no ContextCompressor attribute
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)


def _install_native_compressor(monkeypatch, factory):
    agent_module = sys.modules.get("agent") or ModuleType("agent")
    if not hasattr(agent_module, "__path__"):
        agent_module.__path__ = []
    compressor_module = ModuleType("agent.context_compressor")
    setattr(compressor_module, "ContextCompressor", factory)
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)


def _ignored_engine(tmp_path, name, **config_kwargs):
    config = LCMConfig(
        database_path=str(tmp_path / name),
        ignore_session_patterns=["ignored:*"],
        **config_kwargs,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("ignored:session", platform="cli", context_length=10_000)
    engine.threshold_tokens = 50
    return engine


def _conversation():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "the original objective " + "o" * 4_000},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "terminal", "arguments": "x" * 4_000}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result " + "r" * 4_000},
        {"role": "assistant", "content": "a decision that matters " + "d" * 4_000},
        {"role": "user", "content": "LATEST: cancel deployment"},
    ]


def test_a_bypassed_session_without_a_native_compressor_comes_back_whole(tmp_path, monkeypatch):
    """The plugin holds no copy of this session, so LCM may not drop or cut any part of it."""
    _no_native_compressor(monkeypatch)
    engine = _ignored_engine(tmp_path, "preserve-no-native.db")
    try:
        messages = _conversation()
        original = copy.deepcopy(messages)

        result = engine.compress(messages, current_tokens=100_000, force=True)

        assert result == original
        assert engine._store.get_session_count("ignored:session") == 0
    finally:
        engine.shutdown()


def test_a_short_bypassed_session_keeps_every_turn_and_the_newest_request_last(tmp_path, monkeypatch):
    """The regression this forbids: on a short session a receipt landed behind the newest
    message, and because the trim protects receipts it then deleted the live request
    (`LATEST: cancel deployment`, gone). No message LCM wrote may enter this list at all."""
    _no_native_compressor(monkeypatch)
    engine = _ignored_engine(tmp_path, "preserve-newest.db")
    try:
        messages = [
            {"role": "user", "content": "the original objective " + "o" * 4_000},
            {"role": "assistant", "content": "a decision that matters " + "d" * 4_000},
            {"role": "user", "content": "LATEST: cancel deployment"},
        ]
        original = copy.deepcopy(messages)

        result = engine.compress(messages, current_tokens=100_000, force=True)

        assert result == original
        assert result[-1] == {"role": "user", "content": "LATEST: cancel deployment"}
    finally:
        engine.shutdown()


def test_a_bypassed_session_gets_no_omission_receipt(tmp_path, monkeypatch):
    """A receipt is a claim that something was removed; nothing is removed here, so claiming it
    would be its own defect — and a receipt in a list LCM may not change is content it displaced."""
    _no_native_compressor(monkeypatch)
    engine = _ignored_engine(tmp_path, "preserve-no-receipt.db")
    try:
        result = engine.compress(_conversation(), current_tokens=100_000, force=True)

        rendered = "\n".join(str(message.get("content")) for message in result)
        assert "[Context omitted:" not in rendered
        assert "[LCM bypass trim" not in rendered
        assert "[LCM cut]" not in rendered
    finally:
        engine.shutdown()


def test_a_preserved_bypassed_turn_is_reported_as_an_abort(tmp_path, monkeypatch):
    """Fail-before-loss: the turn survives, and the host's own "nothing was dropped" channel
    carries the failure so the operator is told rather than quietly handed a shortened session."""
    _no_native_compressor(monkeypatch)
    engine = _ignored_engine(tmp_path, "preserve-abort.db")
    try:
        engine.compress(_conversation(), current_tokens=100_000, force=True)

        assert engine._last_compress_aborted is True
        assert "does not store" in str(engine._last_summary_error)
        assert engine._last_compression_status == "bypass_not_compacted"
        assert "unchanged" in engine._last_compression_noop_reason
        assert engine.compression_count == 0
    finally:
        engine.shutdown()


def test_a_declined_bypass_does_not_follow_the_engine_into_the_next_session(tmp_path, monkeypatch):
    """The abort flag is the declined-compaction signal, and Hermes shows it to the user. It
    belongs to the session that earned it: carried into a normal session it would report an
    abort that did not happen, right after a compaction that did."""
    _no_native_compressor(monkeypatch)
    engine = _ignored_engine(tmp_path, "abort-not-inherited.db")
    try:
        engine.compress(_conversation(), current_tokens=100_000, force=True)
        assert engine._last_compress_aborted is True

        engine.on_session_start("normal:session", platform="cli", context_length=10_000)

        assert engine._last_compress_aborted is False
    finally:
        engine.shutdown()


def test_a_failing_native_compressor_does_not_license_an_lcm_trim(tmp_path, monkeypatch):
    """The native compressor raising is a missing capability, not permission to cut."""
    class _FailingCompressor:
        compression_count = 0

        def __init__(self, **kwargs):
            pass

        def compress(self, messages, **kwargs):
            raise RuntimeError("native compressor failed")

    _install_native_compressor(monkeypatch, _FailingCompressor)
    engine = _ignored_engine(tmp_path, "preserve-native-error.db")
    try:
        messages = _conversation()
        original = copy.deepcopy(messages)

        result = engine.compress(messages, current_tokens=100_000, force=True)

        assert result == original
        assert engine._last_compress_aborted is True
    finally:
        engine.shutdown()


def test_an_explicit_native_abort_is_preserved_not_overridden(tmp_path, monkeypatch):
    """An abort is a decision to PRESERVE. LCM may not answer it with its own destructive trim,
    and may not clear the flag that tells the host (and the user) nothing was dropped."""
    class _AbortingCompressor:
        def __init__(self, **kwargs):
            self.compression_count = 0
            self._last_compress_aborted = False
            self._last_summary_error = None

        def compress(self, messages, **kwargs):
            self._last_compress_aborted = True
            self._last_summary_error = "summariser unavailable"
            return list(messages)

    _install_native_compressor(monkeypatch, _AbortingCompressor)
    engine = _ignored_engine(tmp_path, "preserve-native-abort.db")
    try:
        messages = _conversation()
        original = copy.deepcopy(messages)

        result = engine.compress(messages, current_tokens=100_000, force=True)

        assert result == original
        assert engine._last_compress_aborted is True
    finally:
        engine.shutdown()


def test_a_native_compaction_is_returned_exactly_as_the_host_built_it(tmp_path, monkeypatch):
    """Delegation is allowed; post-processing the host's own result is not. LCM neither repairs
    nor re-cuts what the host decided to keep."""
    native_result = [
        {"role": "assistant", "content": "native summary",
         "tool_calls": [{"id": "kept", "type": "function"}]},
        {"role": "user", "content": "interrupt before the tool result"},
        {"role": "tool", "tool_call_id": "orphan", "content": "late result"},
    ]

    class _CompactingCompressor:
        def __init__(self, **kwargs):
            self.compression_count = 0

        def compress(self, messages, **kwargs):
            self.compression_count += 1
            return copy.deepcopy(native_result)

    _install_native_compressor(monkeypatch, _CompactingCompressor)
    engine = _ignored_engine(tmp_path, "native-verbatim.db")
    try:
        result = engine.compress(_conversation(), current_tokens=100_000, force=True)

        assert result == native_result
        assert engine.compression_count == 1
        assert engine._last_compress_aborted is False
    finally:
        engine.shutdown()


def test_a_fresh_auxiliary_context_with_no_window_is_not_trimmed(tmp_path, monkeypatch):
    """A fresh child engine has window 0 and threshold 0, so the old fallback ran with no size
    target at all: 101 messages came back as 36, with a message missing from the middle."""
    _no_native_compressor(monkeypatch)
    config = LCMConfig(database_path=str(tmp_path / "aux-no-window.db"))
    engine = LCMEngine(config=config)
    try:
        engine._mark_thread_context_stateless("auxiliary:session")
        engine.threshold_tokens = 0
        messages = [{"role": "user", "content": f"turn {index} " + "t" * 200} for index in range(101)]
        original = copy.deepcopy(messages)

        result = engine._compress_lcm_bypassed_session(messages, current_tokens=0, force=True)

        assert result == original
    finally:
        engine.shutdown()


def test_user_text_that_looks_like_an_omission_receipt_passes_through(tmp_path, monkeypatch):
    """#46: `[Context omitted:` is plugin text a user or tool can write by hand. It was treated
    as infrastructure — protected from the trim, rewritten into a generic receipt, and a second
    one deleted as a duplicate — so the note's own words were gone."""
    _no_native_compressor(monkeypatch)
    engine = _ignored_engine(tmp_path, "receipt-lookalike.db")
    try:
        messages = [
            {"role": "user", "content": "[Context omitted: my own note about the design]"},
            {"role": "user", "content": "[Context omitted: a second note, with other wording]"},
            {"role": "assistant", "content": "older answer " + "y" * 8_000},
            {"role": "user", "content": "LATEST: cancel deployment"},
        ]
        original = copy.deepcopy(messages)

        result = engine.compress(messages, current_tokens=100_000, force=True)

        assert result == original
    finally:
        engine.shutdown()


def test_the_api_content_sidecar_survives_byte_for_byte(tmp_path, monkeypatch):
    """#64: the trim wrote only `content`, while the host substitutes the untouched
    `api_content` sidecar when it rebuilds the request — so the cut removed what a reader sees
    and nothing of what is sent. Neither field may be cut, dropped or rebuilt."""
    _no_native_compressor(monkeypatch)
    engine = _ignored_engine(tmp_path, "sidecar.db")
    try:
        messages = [
            {"role": "user", "content": "U" * 16_000, "api_content": "U" * 16_000 + " plus context"},
            {"role": "user", "content": "LATEST: cancel deployment"},
        ]
        original = copy.deepcopy(messages)

        result = engine.compress(messages, current_tokens=100_000, force=True)

        assert result == original
        assert result[0]["api_content"] == original[0]["api_content"]
    finally:
        engine.shutdown()


def test_no_lcm_owned_bypass_trim_survives_to_be_rewired(tmp_path):
    """LCM's own trim may not remain as a success fallback — not even unreachable. The machinery
    and its receipt vocabulary are gone, so no later change can route back into them."""
    config = LCMConfig(database_path=str(tmp_path / "no-trim.db"))
    engine = LCMEngine(config=config)
    try:
        for attribute in (
            "_fallback_tail_compaction",
            "_trim_bypass_compacted_to_cap",
            "_truncate_bypass_content_value",
            "_refresh_bypass_receipt",
            "_bypass_envelope_chars",
        ):
            assert not hasattr(engine, attribute), attribute
        for helper in (
            "bypass_omission_marker",
            "bypass_omission_counts",
            "compact_bypass_omission_marker",
            "is_bypass_omission_marker",
            "BYPASS_TRIM_SUFFIX",
            "BYPASS_FINAL_TRIM_SUFFIX",
        ):
            assert not hasattr(marked_loss, helper), helper
    finally:
        engine.shutdown()
