"""#31 MC01: a native content list and a string holding that list's JSON must stay different.

Structured content is persisted as canonical JSON and the envelope fingerprint is computed
from the same normalised text, so a host-produced content list and a user message whose
literal text is exactly that list's JSON became one row shape and one fingerprint. The
original type was then unrecoverable from the archive, and a stored literal could start
matching a pattern it had not matched live. The stored text stays the search/token projection;
the type is recorded beside it, and a row written before that was recorded stays UNKNOWN.
"""
import json

from hermes_lcm import marked_loss
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_content import (
    CONTENT_KIND_KEY,
    CONTENT_KIND_LIST,
    CONTENT_KIND_NONE,
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
        assert structured[CONTENT_KIND_KEY] == CONTENT_KIND_LIST
        assert literal[CONTENT_KIND_KEY] == CONTENT_KIND_STRING
        assert original_content_from_stored(
            structured["content"], structured[CONTENT_KIND_KEY]
        ) == _IMAGE_PARTS
        assert original_content_from_stored(
            literal["content"], literal[CONTENT_KIND_KEY]
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
    assert legacy.get(CONTENT_KIND_KEY) is None
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


def test_a_recorded_type_the_stored_text_no_longer_supports_fails_closed():
    """GC and ingest protection rewrite a row's content in place. The recorded type then no
    longer describes the text, and inventing a list from it would be worse than saying so —
    for every recorded type, `none` included."""
    rewritten = original_content_from_stored("[LCM externalized tool output]", CONTENT_KIND_LIST)
    assert rewritten == "[LCM externalized tool output]"
    assert content_kind(rewritten) != CONTENT_KIND_LIST
    # a row recorded as holding nothing, whose content was later rewritten to a placeholder,
    # still reads back as the text it now holds rather than as the nothing it once held
    assert original_content_from_stored(
        "[LCM externalized tool output]", CONTENT_KIND_NONE
    ) == "[LCM externalized tool output]"
    assert original_content_from_stored(None, CONTENT_KIND_NONE) is None


def test_the_type_record_is_never_mistaken_for_a_field_the_host_sent(tmp_path):
    """The record is the store's bookkeeping. Named without the `_lcm` prefix every generic
    consumer of a message dict already skips, it read as a HOST field on a row that has no
    host envelope: the summariser's envelope receipt named it as content it had not
    summarised, and its mere presence is what decides whether an acknowledgement-shaped turn
    gets the marker that exists for it or is blanked to empty content instead."""
    store = MessageStore(str(tmp_path / "phantom.db"))
    try:
        store_id = store.append("s", {"role": "assistant", "content": "Acknowledged"})
        store.commit()
        row = store.get(store_id)
        assert LCMEngine._message_envelope_fields(row) == {}    # not a host field…
        assert marked_loss.envelope_summary_suffix(
            LCMEngine._message_envelope_fields(row)
        ) == ""
        assert row[CONTENT_KIND_KEY] == CONTENT_KIND_STRING     # …but the record is there
        # and re-storing a row dict cannot turn the record into one either
        again = store.append("s2", row)
        store.commit()
        assert CONTENT_KIND_KEY not in (store.get(again).get("envelope") or {})
        assert "content_kind" not in store.to_openai_msg(store.get(again))
    finally:
        store.close()


def _legacy_shaped_db(tmp_path, name, incoming):
    """A database as the released build wrote it: the row is there, the type record is not."""
    db = str(tmp_path / name)
    store = MessageStore(db)
    store.append("s", incoming)
    store.commit()
    row_id, extra = store.connection.execute(
        "SELECT store_id, envelope_extra FROM messages WHERE session_id='s'").fetchone()
    envelope = json.loads(extra or "{}")
    envelope.pop(CONTENT_KIND_KEY, None)
    store.connection.execute(
        "UPDATE messages SET envelope_extra = ? WHERE store_id = ?",
        (json.dumps(envelope, ensure_ascii=False, sort_keys=True) if envelope else None, row_id))
    store.connection.commit()
    store.close()
    return db


def test_an_unchanged_legacy_row_is_not_reported_as_a_host_correction(tmp_path):
    """Refusing to guess a legacy row's type must not fabricate an edit. Archiving a revision
    does not merely add a row: the summary text then tells the reader that messages summarised
    there "were later CORRECTED by the host", a correction that never happened."""
    incoming = {"role": "user", "content": _IMAGE_PARTS, "message_id": "m-1"}
    engine = LCMEngine(config=LCMConfig(
        database_path=_legacy_shaped_db(tmp_path, "revision.db", incoming)))
    try:
        engine.on_session_start("s", context_length=200_000)
        engine._session_id = "s"
        assert engine._record_host_message_revisions([incoming]) == 0
        rows = engine._store.get_session_messages("s")
        assert len(rows) == 1
        assert rows[0][CONTENT_KIND_KEY] == CONTENT_KIND_UNKNOWN
    finally:
        engine.shutdown()


def test_a_real_edit_of_a_legacy_row_is_still_archived(tmp_path):
    """Tolerating an unrecorded type must not make the detector blind. That trade is the
    reason the type is in the fingerprint at all."""
    incoming = {"role": "user", "content": _IMAGE_PARTS, "message_id": "m-1"}
    engine = LCMEngine(config=LCMConfig(
        database_path=_legacy_shaped_db(tmp_path, "edited.db", incoming)))
    try:
        engine.on_session_start("s", context_length=200_000)
        engine._session_id = "s"
        edited = {"role": "user", "content": "the host replaced it", "message_id": "m-1"}
        assert engine._record_host_message_revisions([edited]) == 1
        rows = engine._store.get_session_messages("s")
        assert len(rows) == 2
        assert rows[1]["envelope"]["lcm_supersedes_store_id"] == rows[0]["store_id"]
    finally:
        engine.shutdown()


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
