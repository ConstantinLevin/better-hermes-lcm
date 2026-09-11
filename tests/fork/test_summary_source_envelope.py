"""#67: the summariser is offered the VALUE of every host envelope field, not its name and size.

`api_content` is the text Hermes actually sends to the provider in place of the display
content, and `reasoning`/`reasoning_content` carry the turn's own reasoning. All three were
archived verbatim and handed to the summariser as "name (N chars)", so a turn indexed as
"the user asked to continue" could just as well have said the opposite.
"""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture()
def engine(tmp_path):
    e = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "envelope.db")),
                  hermes_home=str(tmp_path))
    e.on_session_start("env", platform="cli", context_length=262_144)
    try:
        yield e
    finally:
        e.shutdown()


def test_a_semantic_envelope_value_reaches_the_summariser_whole(engine):
    """The host's api_content substitution carried the only copy of the instruction."""
    serialized = engine._serialize_messages([
        {
            "role": "user",
            "content": "please continue",
            "api_content": "please continue\nMUST_DELIVER_NOTE_ALPHA: deployment revoked.",
        },
    ])
    assert "MUST_DELIVER_NOTE_ALPHA: deployment revoked." in serialized, serialized
    assert "api_content (" not in serialized, serialized


def test_two_opposite_same_length_sidecars_do_not_serialise_identically(engine):
    """#67's own counter-probe: 'revoked.' and 'allowed.' are the same length, so a
    name-and-size rendering made the distinction disappear before the model."""
    def source(note: str) -> str:
        return engine._serialize_messages([
            {"role": "user", "content": "please continue",
             "api_content": f"please continue\nMUST_DELIVER_NOTE_ALPHA: deployment {note}"},
        ])

    assert source("revoked.") != source("allowed.")


def test_a_reasoning_field_is_offered_as_text_not_as_a_character_count(engine):
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": "Visible",
         "reasoning_content": "DECISION cancel the rollout", "is_error": True},
    ])
    assert "DECISION cancel the rollout" in serialized, serialized
    assert "is_error=True" in serialized, serialized


def test_a_structured_envelope_value_is_rendered_with_its_contents(engine):
    """A dict or list value was never eligible for inline rendering at all, so every
    provider-structured field reached the summariser as a name and a length."""
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": "done",
         "reasoning_details": [{"type": "summary", "text": "rejected the rollback plan"}]},
    ])
    assert "rejected the rollback plan" in serialized, serialized


def test_every_short_envelope_field_is_rendered_not_only_the_first_ten(engine):
    """The inline list was cut at ten with nothing in place of the rest."""
    msg = {"role": "assistant", "content": "done"}
    for index in range(14):
        msg[f"field_{index:02d}"] = f"value_{index:02d}"
    serialized = engine._serialize_messages([msg])
    for index in range(14):
        assert f"value_{index:02d}" in serialized, (index, serialized)


def test_an_envelope_value_obeys_the_operator_s_sensitive_pattern_policy(tmp_path):
    """Offering the value (#67) must not walk it past a policy the operator switched on.

    `engine.py` redacts content and tool arguments before serialising, and ingest redacts
    content only, so an envelope field was the one way a configured pattern reached the model
    in plaintext. The repair is redaction, not removal: the field is still offered.
    """
    e = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "redact.db"),
                         sensitive_patterns_enabled=True),
        hermes_home=str(tmp_path),
    )
    e.on_session_start("red", platform="cli", context_length=262_144)
    try:
        serialized = e._serialize_messages([
            {"role": "user", "content": "api_key=SUPERSECRETVALUE1234567890",
             "api_content": "api_key=SUPERSECRETVALUE1234567890",
             "finish_reason": "stop"},
        ])
        assert "SUPERSECRETVALUE1234567890" not in serialized, serialized
        # ... and the field is still offered, not removed
        assert "api_content" in serialized, serialized
        assert "finish_reason=stop" in serialized, serialized
    finally:
        e.shutdown()


def test_an_envelope_key_named_like_a_host_field_is_not_dropped_for_its_name(engine):
    """#67 removed a name-based rule about what the model may see; `lcm_`-prefixed HOST keys
    were still being dropped by one, with no receipt."""
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": "done", "lcm_run_label": "nightly-rollback"},
    ])
    assert "nightly-rollback" in serialized, serialized


def test_an_lcm_internal_envelope_key_is_named_rather_than_dropped_in_silence(engine):
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": "done",
         "envelope": {"_lcm_internal": "bookkeeping", "finish_reason": "stop"}},
    ])
    assert "_lcm_internal" in serialized, serialized
    assert "lcm_expand" in serialized, serialized
    assert "finish_reason=stop" in serialized, serialized


def test_an_unrenderable_value_is_named_while_its_renderable_neighbour_is_shown(engine):
    """Fail-visibly, never silently, and never take one bad field's neighbours down with it."""
    class _Unrenderable:
        def __repr__(self):
            raise RuntimeError("this value cannot be rendered")

        __str__ = __repr__

    serialized = engine._serialize_messages([
        {"role": "assistant", "content": "done",
         "provider_blob": _Unrenderable(),
         "api_content": "done\nNOTE_BETA: rollback approved.",
         "finish_reason": "stop"},
    ])
    assert "NOTE_BETA: rollback approved." in serialized, serialized
    assert "provider_blob" in serialized, serialized
    assert "lcm_expand" in serialized, serialized
    assert "finish_reason=stop" in serialized, serialized
