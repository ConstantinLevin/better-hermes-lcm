"""#31 MA01: a real assistant reply is never deleted on the strength of its wording.

`_is_synthetic_assistant_noise` classified an assistant turn as synthetic by matching its text
against a word set. The normal Hermes producer always supplies `finish_reason`, so the marker
branch never fired and the text was replaced by `""` with nothing in its place: the summariser
was handed `[ASSISTANT]:  [finish_reason=stop]` for a turn that really said `Acknowledged`.
Synthetic origin needs a host signal the plugin does not have; guessing it from words is not
one.
"""
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture()
def engine(tmp_path):
    e = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "ack.db")),
                  hermes_home=str(tmp_path))
    e.on_session_start("ack", platform="cli", context_length=262_144)
    try:
        yield e
    finally:
        e.shutdown()


def test_a_real_acknowledgement_reaches_the_source_byte_identical(engine):
    """The shape the real host producer emits: visible text plus finish_reason."""
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": "Acknowledged", "finish_reason": "stop"},
    ])
    assert "[ASSISTANT]: Acknowledged" in serialized, serialized


def test_an_acknowledgement_with_no_envelope_is_not_replaced_by_a_receipt(engine):
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": "Acknowledged"},
    ])
    assert "[ASSISTANT]: Acknowledged" in serialized, serialized
    assert "acknowledgement-shaped" not in serialized, serialized


@pytest.mark.parametrize("text", ["ack", "Acknowledged", "heartbeat", "keepalive",
                                  "keep alive", "pong", "[heartbeat]", "**ACK**"])
def test_no_wording_removes_an_assistant_turn_from_the_source(engine, text):
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": text, "finish_reason": "stop"},
    ])
    assert f"[ASSISTANT]: {text}" in serialized, serialized


def test_an_empty_assistant_turn_does_not_claim_an_acknowledgement_was_removed(engine):
    """A receipt is a claim that something was removed; a turn that held nothing must not
    produce one."""
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": ""},
    ])
    assert "acknowledgement-shaped" not in serialized, serialized


def test_an_acknowledgement_that_also_made_a_tool_call_keeps_both(engine):
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": "Acknowledged", "finish_reason": "stop",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "deploy", "arguments": '{"env":"prod"}'}}]},
    ])
    assert "Acknowledged" in serialized, serialized
    assert "deploy(" in serialized, serialized
