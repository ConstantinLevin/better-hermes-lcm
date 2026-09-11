"""the bypass path hands ignored/stateless/auxiliary context to the host's native
compressor and changes nothing itself. Three upstream behaviours turned host decisions into
loss: BY01 a deterministic trim deleted and cut a session the plugin never stored, BY02 an
abort became a destructive trim reported as success, and BY03 one unsupported constructor
keyword sent bypassed summarisation to a different model/route than configured.

The preservation contract itself lives in ``test_bypass_preservation.py``; what remains here is
the host-delegation behaviour plus the inputs the deleted trim used to be measured on.
"""
import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm import marked_loss
from hermes_lcm.engine import LCMEngine


def _bypassed_engine(tmp_path, name):
    cfg = LCMConfig(database_path=str(tmp_path / name))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e.on_session_start("bypassed", platform="cli", context_length=200_000)
    e._session_stateless = True  # the bypass path this test is about
    e.threshold_tokens = 10
    return e


def _no_native_compressor(monkeypatch):
    """Make Hermes' native compressor genuinely unavailable, the way a non-Hermes host is."""
    agent_module = sys.modules.get("agent") or ModuleType("agent")
    if not hasattr(agent_module, "__path__"):
        agent_module.__path__ = []
    compressor_module = ModuleType("agent.context_compressor")  # no ContextCompressor attribute
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)


def test_an_unchanged_native_return_is_not_counted_as_a_compression(tmp_path):
    e = _bypassed_engine(tmp_path, "by02.db")
    try:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first decision " + "a" * 400},
            {"role": "assistant", "content": "second decision " + "b" * 400},
            {"role": "user", "content": "newest"},
        ]

        class _AbortingCompressor:
            compression_count = 0
            _last_compress_aborted = True

            def compress(self, msgs, **_kwargs):
                return msgs

        e._host_fallback_compressor = _AbortingCompressor()
        e._host_fallback_session_id = e._bypass_lcm_session_id()
        before = e.compression_count

        result = e._compress_lcm_bypassed_session(messages, current_tokens=100_000, force=True)

        blob = "\n".join(str(m.get("content")) for m in result)
        # the host's preservation decision stands untouched, cap pressure or not
        assert "first decision" in blob and "second decision" in blob, "the abort was overridden"
        assert e.compression_count == before, "an unchanged return is not a compression"
        assert e._last_compress_aborted is True
    finally:
        e.shutdown()


def test_an_older_host_signature_drops_only_what_it_cannot_take(tmp_path, monkeypatch):
    e = _bypassed_engine(tmp_path, "by03.db")
    try:
        seen: dict = {}

        class _OlderCompressor:
            def __init__(self, model, *, threshold_percent=0.5, protect_first_n=0,
                         protect_last_n=0, quiet_mode=False, summary_model_override=None,
                         config_context_length=None):
                seen["model"] = model
                seen["summary_model_override"] = summary_model_override
                seen["config_context_length"] = config_context_length

        assert e._constructor_supported_kwargs(_OlderCompressor, {
            "model": "m", "threshold_percent": 0.5, "protect_first_n": 1, "protect_last_n": 2,
            "quiet_mode": True, "summary_model_override": "chosen-route",
            "base_url": "http://x", "api_key": "k", "config_context_length": 900_000,
            "provider": "p", "api_mode": "a",
        }) == {
            "model": "m", "threshold_percent": 0.5, "protect_first_n": 1, "protect_last_n": 2,
            "quiet_mode": True, "summary_model_override": "chosen-route",
            "config_context_length": 900_000,
        }
    finally:
        e.shutdown()


# fork: better-hermes-lcm — this used to assert that the deterministic fallback's omission
# receipt survived every trimming stage (audit p05 BY01). The fork no longer has a fallback
# that drops or cuts a session it does not store, so there is no receipt to survive: the same
# backlog now comes back whole.
def test_a_long_backlog_in_a_bypassed_session_is_not_trimmed(tmp_path, monkeypatch):
    _no_native_compressor(monkeypatch)
    e = _bypassed_engine(tmp_path, "by01.db")
    try:
        messages = [{"role": "system", "content": "system"}]
        messages += [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index} " + "x" * 300}
            for index in range(20)
        ]
        original = copy.deepcopy(messages)

        result = e._compress_lcm_bypassed_session(messages, current_tokens=100_000, force=True)

        assert result == original
        rendered = "\n".join(str(m.get("content")) for m in result)
        assert "[Context omitted:" not in rendered
    finally:
        e.shutdown()


# fork: better-hermes-lcm — this used to assert that a cut at a zero character budget still
# appended a cut marker (audit p05 BY01: upstream dropped the marker exactly where it removed
# everything). The fork cuts no characters on this path at all, so the helper that did it is
# gone and the text it would have cut is returned intact.
def test_a_bypassed_message_is_never_cut_to_a_marker(tmp_path, monkeypatch):
    _no_native_compressor(monkeypatch)
    e = _bypassed_engine(tmp_path, "by01b.db")
    try:
        messages = [
            {"role": "user", "content": "a decision that matters"},
            {"role": "user", "content": "LATEST: what is the status?"},
        ]
        original = copy.deepcopy(messages)

        result = e._compress_lcm_bypassed_session(messages, current_tokens=100_000, force=True)

        assert result == original
    finally:
        e.shutdown()


# fork: better-hermes-lcm — this used to assert that protecting the omission receipt never cost
# the newest message (verify-2 regression #2). Nothing is removed now, so nothing competes with
# the newest message; a hand-written text that merely LOOKS like a receipt is ordinary content
# and survives with it (#46).
def test_a_receipt_lookalike_never_costs_the_newest_request(tmp_path, monkeypatch):
    _no_native_compressor(monkeypatch)
    e = _bypassed_engine(tmp_path, "by01c.db")
    try:
        messages = [
            {"role": "user", "content": "the original objective " + "o" * 200},
            {"role": "user", "content": "[Context omitted: my own note about the design]"},
            {"role": "user", "content": "LATEST REQUEST: what is the status?"},
        ]
        original = copy.deepcopy(messages)

        result = e._compress_lcm_bypassed_session(messages, current_tokens=100_000, force=True)

        assert result == original
        assert result[-1]["content"] == "LATEST REQUEST: what is the status?"
    finally:
        e.shutdown()


# fork: better-hermes-lcm — this used to assert that the receipt's counts matched what the cap
# loop had actually removed (verify-4 #17). The cap loop is gone: the correct count of removed
# messages is now zero for every input, which is what this asserts instead.
def test_a_bypassed_session_loses_no_message_to_a_cap(tmp_path, monkeypatch):
    _no_native_compressor(monkeypatch)
    e = _bypassed_engine(tmp_path, "by17.db")
    try:
        messages = [{"role": "user", "content": f"turn {index} " + "t" * 400} for index in range(10)]
        original = copy.deepcopy(messages)

        result = e._compress_lcm_bypassed_session(messages, current_tokens=100_000, force=True)

        assert result == original
    finally:
        e.shutdown()


# fork: better-hermes-lcm — this used to assert that a dropped assistant turn's tool-call
# arguments were counted in the receipt's character total (round-2 verify-4 #29). No assistant
# turn is dropped now, so the arguments themselves — not a number describing them — survive.
def test_tool_call_arguments_survive_a_bypassed_compaction(tmp_path, monkeypatch):
    _no_native_compressor(monkeypatch)
    e = _bypassed_engine(tmp_path, "bypasscalls.db")
    try:
        e.protect_first_n = 1
        e.protect_last_n = 1
        big = "x" * 10_000
        messages = [
            {"role": "user", "content": "keep me"},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "terminal", "arguments": big}}]},
            {"role": "user", "content": "and me"},
        ]
        original = copy.deepcopy(messages)

        result = e._compress_lcm_bypassed_session(messages, current_tokens=100_000, force=True)

        assert result == original
        assert result[1]["tool_calls"][0]["function"]["arguments"] == big
    finally:
        e.shutdown()


# fork: better-hermes-lcm — this used to assert that a second bypass compaction carried the
# first receipt's counts forward instead of erasing them (round-5 verify-6 #10). Neither pass
# removes anything now, so there is no accounting to carry: repeated compaction of a growing
# bypassed session is idempotent on its content.
def test_repeated_bypass_compaction_keeps_every_earlier_turn(tmp_path, monkeypatch):
    _no_native_compressor(monkeypatch)
    e = _bypassed_engine(tmp_path, "bycum.db")
    try:
        first_input = [{"role": "user", "content": f"turn {i} " + "t" * 400} for i in range(10)]
        first = e._compress_lcm_bypassed_session(
            copy.deepcopy(first_input), current_tokens=100_000, force=True
        )
        assert first == first_input

        second_input = first + [
            {"role": "user", "content": f"later {i} " + "u" * 400} for i in range(5)
        ]
        second = e._compress_lcm_bypassed_session(
            copy.deepcopy(second_input), current_tokens=100_000, force=True
        )

        assert second == second_input
    finally:
        e.shutdown()
