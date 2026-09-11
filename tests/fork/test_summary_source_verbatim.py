"""#56: the shared pre-compaction sanitizer no longer alters original text.

One helper sat on two paths — the text handed to the summariser and the text `lcm_expand`
returns for a raw row. It removed the first line of a line-wrapped `data:` URI while a marker
claimed the medium had been removed, it removed host-injected context blocks, and it
round-tripped any valid JSON argument string through `json.loads`/`json.dumps`, so
`0.123456789012345678901234567890` became `0.12345678901234568` in both places.

Widening the media regex is not the fix — the permissive form eats the words after the payload
(claw's does exactly that to `STOP\\nDo not deploy.`). The source is offered whole, and a
payload the configured summary route cannot process fails there instead.
"""
import json

import pytest

from hermes_lcm import tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.extraction import (
    sanitize_pre_compaction_content,
    sanitize_pre_compaction_tool_arguments,
)

EXACT_DECIMAL = '{"amount":0.123456789012345678901234567890,"note":"exact decimal"}'
WRAPPED_MEDIA = (
    "data:image/png;base64,"
    + "\n".join(["QUJD" * 16] * 32)
    + "\nSTOP\nDo not deploy."
)


@pytest.fixture()
def engine(tmp_path):
    e = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "verbatim.db")),
                  hermes_home=str(tmp_path))
    e.on_session_start("v", platform="cli", context_length=262_144)
    try:
        yield e
    finally:
        e.shutdown()


def test_a_line_wrapped_data_uri_reaches_the_summariser_whole(engine):
    serialized = engine._serialize_messages([{"role": "user", "content": WRAPPED_MEDIA}])
    assert WRAPPED_MEDIA in serialized, serialized[:400]


def test_no_marker_claims_a_medium_was_removed_when_it_was_not(engine):
    serialized = engine._serialize_messages([{"role": "user", "content": WRAPPED_MEDIA}])
    assert "media attachment" not in serialized.lower(), serialized[:400]


def test_a_single_line_data_uri_is_not_removed_either(engine):
    text = "before data:image/png;base64," + "A" * 40 + " after"
    serialized = engine._serialize_messages([{"role": "user", "content": text}])
    assert text in serialized, serialized


def test_host_injected_context_reaches_the_summary_source(engine):
    text = 'keep this <active_memory decision="CANCEL"/> and this'
    serialized = engine._serialize_messages([{"role": "user", "content": text}])
    assert text in serialized, serialized
    assert "chars of injected context removed" not in serialized, serialized


def test_an_exact_decimal_reaches_the_summary_source_unrounded(engine):
    serialized = engine._serialize_messages([
        {"role": "assistant", "content": "calling",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "pay", "arguments": EXACT_DECIMAL}}]},
    ])
    assert "0.123456789012345678901234567890" in serialized, serialized
    assert "0.12345678901234568" not in serialized, serialized


def test_the_shared_helper_returns_a_json_argument_string_byte_identical():
    assert sanitize_pre_compaction_tool_arguments(EXACT_DECIMAL) == EXACT_DECIMAL


def test_the_shared_helper_returns_ordinary_text_byte_identical():
    assert sanitize_pre_compaction_content(WRAPPED_MEDIA) == WRAPPED_MEDIA


def test_a_stored_tool_call_expands_to_its_original_bytes(engine):
    engine._store.append("v", {
        "role": "assistant", "content": "calling",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "pay", "arguments": EXACT_DECIMAL}}],
    }, source="cli")
    engine._store.commit()
    store_id = engine._store.get_session_tail("v", limit=1)[-1]["store_id"]

    payload = json.loads(lcm_tools.lcm_expand({"store_id": store_id}, engine=engine))
    assert "0.123456789012345678901234567890" in payload["tool_calls"], payload
    assert "tool_calls_note" not in payload, payload


def test_a_raw_expand_continuation_pages_over_the_original_bytes(engine):
    engine._store.append("v", {
        "role": "assistant", "content": "calling",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "pay", "arguments": EXACT_DECIMAL}}],
    }, source="cli")
    engine._store.commit()
    stored = engine._store.get_session_tail("v", limit=1)[-1]
    store_id = stored["store_id"]
    original = json.dumps(stored["tool_calls"], ensure_ascii=False, default=str)

    walked = ""
    offset = 0
    for _ in range(50):
        payload = json.loads(lcm_tools.lcm_expand(
            {"store_id": store_id, "max_tokens": 8, "tool_calls_offset": offset},
            engine=engine,
        ))
        walked += payload["tool_calls"]
        if not payload.get("tool_calls_truncated"):
            break
        offset = payload["tool_calls_next_offset"]
    assert walked == original, (walked, original)
