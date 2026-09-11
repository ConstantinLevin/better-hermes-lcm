"""#31 MC01: a native content list and a string holding that list's JSON must stay different.

Structured content is persisted as canonical JSON and the envelope fingerprint is computed
from the same normalised text, so a host-produced content list and a user message whose
literal text is exactly that list's JSON became one row shape and one fingerprint. The
original type was then unrecoverable from the archive, and a stored literal could start
matching a pattern it had not matched live. The stored text stays the search/token projection;
the type is recorded beside it, and a row written before that was recorded stays UNKNOWN.
"""
import json

from hermes_lcm.message_content import (
    CONTENT_KIND_KEY,
    CONTENT_KIND_LIST,
    CONTENT_KIND_STRING,
    CONTENT_KIND_UNKNOWN,
    content_kind,
    original_content_from_stored,
    stored_text_content_for_pattern_matching,
)
from hermes_lcm.store import MessageStore, message_envelope_fingerprint


_IMAGE_PARTS = [
    {"type": "text", "text": "HEARTBEAT: Do not deploy."},
    {"type": "image_url", "image_url": {"url": "https://example.invalid/shot.png"}},
]
_IMAGE_PARTS_JSON = json.dumps(_IMAGE_PARTS, ensure_ascii=False, sort_keys=True)


def _stored_rows(tmp_path, name="types.db"):
    store = MessageStore(str(tmp_path / name))
    store.append_batch("s", [
        {"role": "user", "content": _IMAGE_PARTS},
        {"role": "user", "content": _IMAGE_PARTS_JSON},
    ])
    store.commit()
    return store, store.get_session_messages("s")


def test_the_list_and_its_json_literal_round_trip_to_different_originals(tmp_path):
    store, rows = _stored_rows(tmp_path)
    try:
        structured, literal = rows[0], rows[1]
        # the stored text is still the one search/token projection, unchanged
        assert structured["content"] == literal["content"] == _IMAGE_PARTS_JSON
        # …and the type is recorded beside it
        assert structured["content_kind"] == CONTENT_KIND_LIST
        assert literal["content_kind"] == CONTENT_KIND_STRING
        assert original_content_from_stored(
            structured["content"], structured["content_kind"]
        ) == _IMAGE_PARTS
        assert original_content_from_stored(
            literal["content"], literal["content_kind"]
        ) == _IMAGE_PARTS_JSON
    finally:
        store.close()


def test_the_envelope_fingerprint_separates_them(tmp_path):
    store, rows = _stored_rows(tmp_path, "fingerprint.db")
    try:
        assert message_envelope_fingerprint(rows[0]) != message_envelope_fingerprint(rows[1])
    finally:
        store.close()


def test_a_stored_row_still_matches_the_incoming_message_it_came_from(tmp_path):
    """The fingerprint is the host-edit change detector. Recording the type must not make an
    unchanged message look edited on every turn."""
    store, rows = _stored_rows(tmp_path, "unchanged.db")
    try:
        assert message_envelope_fingerprint({"role": "user", "content": _IMAGE_PARTS}) == (
            message_envelope_fingerprint(rows[0]))
        assert message_envelope_fingerprint({"role": "user", "content": _IMAGE_PARTS_JSON}) == (
            message_envelope_fingerprint(rows[1]))
    finally:
        store.close()


def test_ordinary_text_keeps_the_fingerprint_it_already_had():
    """Digests computed by the released build are stored on rows. Content whose text cannot be
    read as anything but a string is not ambiguous, so its fingerprint must not move."""
    fingerprint = message_envelope_fingerprint({"role": "assistant", "content": "Acknowledged"})
    assert "content_kind" not in fingerprint
    assert "content_kind" not in message_envelope_fingerprint({"role": "user", "content": None})


def test_a_row_written_before_the_type_was_recorded_stays_unknown():
    """Parsing as JSON is not evidence of having been JSON. Unknown stays unknown: it is not
    backfilled as a list, and it is not asserted to have been a string either."""
    legacy = {
        "store_id": 7,
        "role": "user",
        "content": _IMAGE_PARTS_JSON,
        "envelope": {},
    }
    assert legacy.get("content_kind") is None
    fingerprint = message_envelope_fingerprint(legacy)
    assert f'"content_kind": "{CONTENT_KIND_UNKNOWN}"' in fingerprint
    assert message_envelope_fingerprint(legacy) != message_envelope_fingerprint(
        {"role": "user", "content": _IMAGE_PARTS})
    assert message_envelope_fingerprint(legacy) != message_envelope_fingerprint(
        {"role": "user", "content": _IMAGE_PARTS_JSON})


def test_stored_pattern_text_reads_the_recorded_type_instead_of_guessing():
    """A better regex is not the fix: the missing type is upstream of it. Given the type, the
    literal keeps its own text and only the row that really was structured is decoded — and a
    row whose type nobody recorded is decoded into nothing at all."""
    assert stored_text_content_for_pattern_matching(
        _IMAGE_PARTS_JSON, CONTENT_KIND_LIST
    ) == "HEARTBEAT: Do not deploy."
    assert stored_text_content_for_pattern_matching(
        _IMAGE_PARTS_JSON, CONTENT_KIND_STRING
    ) == _IMAGE_PARTS_JSON
    assert stored_text_content_for_pattern_matching(
        _IMAGE_PARTS_JSON, CONTENT_KIND_UNKNOWN
    ) == _IMAGE_PARTS_JSON


def test_omitting_the_type_still_guesses_it_and_that_is_the_remaining_consumer():
    """Pins the one path that still guesses, so it is impossible to think it was fixed here:
    `LCMEngine._matches_ignore_message_patterns` reads `msg.get("content")` and drops the row's
    `content_kind`, and the durable-row half of the ignore policy has nothing else to
    recognise a structured row by. It goes away when that caller passes the field."""
    assert stored_text_content_for_pattern_matching(
        _IMAGE_PARTS_JSON
    ) == "HEARTBEAT: Do not deploy."


def test_a_recorded_type_the_stored_text_no_longer_supports_fails_closed():
    """GC and ingest protection rewrite a row's content in place. The recorded type then no
    longer describes the text, and inventing a list from it would be worse than saying so."""
    rewritten = original_content_from_stored("[LCM externalized tool output]", CONTENT_KIND_LIST)
    assert rewritten == "[LCM externalized tool output]"
    assert content_kind(rewritten) != CONTENT_KIND_LIST


def test_the_type_marker_never_reaches_the_replayed_message(tmp_path):
    """The record is the store's, not the host's: it stays out of the envelope the host sent
    and out of everything replayed back to a provider."""
    store = MessageStore(str(tmp_path / "leak.db"))
    try:
        plain = store.append("s", {"role": "assistant", "content": "plain"})
        tagged = store.append("s", {"role": "user", "content": "with id", "message_id": "m-7"})
        store.commit()
        assert "envelope" not in store.get(plain)
        assert store.get(tagged)["envelope"] == {"message_id": "m-7"}
        assert CONTENT_KIND_KEY not in store.to_openai_msg(store.get(tagged))
    finally:
        store.close()
