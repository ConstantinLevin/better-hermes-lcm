"""What the beta's source-fidelity fixes must achieve, written before they exist (#19).

Every test here FAILS on the tree it was written against. That is the point: the assertions say
what #56, #67, #37, #31 and #63 have to make true, and they were written by someone who is not
writing those fixes, so a fix is measured against a contract it did not author.

They are gated on ``LCM_BETA_TARGET=1`` (see ``tests/conftest.py``) so the default suite stays
green for the groups still working on them, and the skip names its issue rather than passing
quietly.

The contract asserted is always the ORIGINAL BYTES reaching the string the summariser is handed,
never a marker in their place. An assertion a convincing receipt could satisfy is the old
contract — the one this fork exists to remove — and is the wrong assertion.
"""
import importlib
import json
import time
from datetime import datetime, timezone

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.store import MessageStore, message_envelope_fingerprint


# Every value in this fork that is a preference is a smooth interpolation between these two
# anchors, and the fork's whole thesis is that a cut which fires at one of them and not the other
# is a defect. A preservation fix built against one window can therefore regress at the other
# with nothing red, so the byte-preservation assertions run at both.
WINDOWS = (262_144, 1_000_000)


def _engine(tmp_path, name, window=262_144, **cfg):
    config = LCMConfig(database_path=str(tmp_path / f"{name}.db"), **cfg)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.on_session_start(name, platform="cli", context_length=window)
    return engine


# ── #56 — the shared source sanitiser must not rewrite what the host sent ────────────────────

def _wrapped_data_uri_message() -> str:
    """A textual data: URI wrapped at 64 characters, with prose immediately after it.

    This is ordinary user input — a pasted diagnostic sample — not a claim that the payload
    decodes to a real PNG. `_MEDIA_DATA_URI_RE` admits no whitespace in its payload class, so it
    matches the header plus the FIRST line only: that line is deleted and replaced with a media
    marker while the other 31 lines stay, so the marker claims the medium went and the message
    silently lost its first 86 characters.
    """
    lines = ["QUJD" * 16 for _ in range(32)]
    return "data:image/png;base64," + "\n".join(lines) + "\nSTOP\nDo not deploy."


@pytest.mark.beta_target("#56")
@pytest.mark.parametrize("window", WINDOWS)
def test_a_line_wrapped_data_uri_reaches_the_summariser_source_byte_identical(tmp_path, window):
    content = _wrapped_data_uri_message()
    engine = _engine(tmp_path, "wrapped56", window)
    try:
        serialized = engine._serialize_messages([{"role": "user", "content": content}])
        assert content in serialized, (
            "the source handed to the summariser is not the message the host sent"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#56")
@pytest.mark.parametrize("window", WINDOWS)
def test_a_json_argument_string_is_not_reinterpreted_before_the_summariser(tmp_path, window):
    """A decimal a provider really sent must not be rounded on the way to the source.

    `sanitize_pre_compaction_tool_arguments` round-trips every parseable argument string through
    `json.loads`/`json.dumps`, which is lossy for a literal wider than a float: the exact decimal
    below comes out as 0.12345678901234568. No data URI, no redaction and no externalization is
    involved — an ordinary tool call is enough.
    """
    exact = "0.123456789012345678901234567890"
    arguments = '{"amount":' + exact + ',"note":"exact decimal"}'
    engine = _engine(tmp_path, "decimal56", window)
    try:
        serialized = engine._serialize_messages([
            {"role": "assistant", "content": "paying", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "pay", "arguments": arguments}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ])
        assert exact in serialized, "the summariser was given a different number"
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#56")
@pytest.mark.parametrize("window", WINDOWS)
def test_the_raw_row_reader_returns_the_argument_string_it_stored(tmp_path, window):
    """The same transformation helper sits on `lcm_expand(store_id=…)`.

    Summary input and direct raw expansion share `_sanitized_tool_calls_for_response`, so fixing
    only the compaction call site leaves the recovery route returning a number the store does not
    hold — and a caller paging that response pages the wrong text.
    """
    exact = "0.123456789012345678901234567890"
    arguments = '{"amount":' + exact + ',"note":"exact decimal"}'
    engine = _engine(tmp_path, "reader56", window)
    try:
        store_id = engine._store.append("reader56", {
            "role": "assistant", "content": "paying",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "pay", "arguments": arguments}}],
        }, source="cli")
        engine._store.commit()

        payload = json.loads(engine.handle_tool_call(
            "lcm_expand", {"store_id": int(store_id), "max_tokens": 100_000}))
        rendered = json.dumps(payload, ensure_ascii=False)
        assert exact in rendered, (
            "lcm_expand returned a rounded number for a row that stores the exact one"
        )
    finally:
        engine.shutdown()


# ── #67 — semantic envelope values must reach the summariser AS VALUES ───────────────────────

@pytest.mark.beta_target("#67")
@pytest.mark.parametrize("window", WINDOWS)
def test_a_semantic_envelope_value_reaches_the_summariser_as_a_value(tmp_path, window):
    """`api_content` is what the host actually substitutes as the API text for a turn.

    Today the serialiser inventories it as `api_content (N chars)`. A name and a length are not
    the value: the instruction below never reaches the model at all, while the receipt says a
    field of that size exists in the store.
    """
    note = "MUST_DELIVER_NOTE_ALPHA: deployment revoked."
    engine = _engine(tmp_path, "envelope67", window)
    try:
        serialized = engine._serialize_messages([
            {"role": "user", "content": "Proceed with the rollout.",
             "api_content": "Proceed with the rollout.\n\n" + note},
        ])
        assert note in serialized, "the host's own API text never reached the summariser"
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#67")
def test_two_opposite_envelope_values_do_not_serialise_identically(tmp_path):
    """The discriminating form of the same defect, and the one a marker cannot satisfy.

    A name-and-length inventory renders "deployment revoked." and "deployment allowed." — equal
    lengths, opposite meanings — as the same source text. Whatever shape the fix takes, these two
    turns must not be indistinguishable before the model.
    """
    engine = _engine(tmp_path, "opposite67")
    try:
        def source(decision: str) -> str:
            return engine._serialize_messages([
                {"role": "user", "content": "Proceed with the rollout.",
                 "api_content": "Proceed with the rollout.\n\ndeployment " + decision},
            ])

        assert source("revoked.") != source("allowed."), (
            "a value and its opposite produce the same summariser source"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#67")
@pytest.mark.parametrize("window", WINDOWS)
def test_reasoning_carriers_reach_the_summariser_as_values(tmp_path, window):
    """`reasoning` and `reasoning_content` are inventoried by the same branch as `api_content`.

    The issue does not claim every provider's continuation semantics are covered; it claims the
    stored value is withheld from the model. These are the two carriers the host's own assistant
    builder produces.
    """
    engine = _engine(tmp_path, "reasoning67", window)
    try:
        serialized = engine._serialize_messages([
            {"role": "assistant", "content": "Done.",
             "reasoning": "REASONING_ALPHA: the rollback was chosen over the retry.",
             "reasoning_content": "REASONING_BETA: credentials expire at 14:02."},
        ])
        assert "REASONING_ALPHA: the rollback was chosen over the retry." in serialized
        assert "REASONING_BETA: credentials expire at 14:02." in serialized
    finally:
        engine.shutdown()


# ── #37 — the source timestamp the host already stamped ──────────────────────────────────────

_MILLENNIUM = 946684800.0          # 2000-01-01T00:00:00Z
_MILLENNIUM_NEXT_DAY = 946771200.0  # 2000-01-02T00:00:00Z


@pytest.mark.beta_target("#37")
def test_two_turns_from_different_days_do_not_serialise_identically(tmp_path):
    """`_message_envelope_fields` excludes the time keys outright, so the day is deleted.

    "The meeting is tomorrow." from two different days becomes the same source string. This is
    the form of the assertion that no rendering choice can dodge: whichever way the fix carries
    the time, two different instants must not collapse into one.
    """
    engine = _engine(tmp_path, "days37")
    try:
        def source(stamp: float) -> str:
            return engine._serialize_messages([
                {"role": "user", "content": "The meeting is tomorrow.", "timestamp": stamp},
            ])

        assert source(_MILLENNIUM) != source(_MILLENNIUM_NEXT_DAY), (
            "two turns from different days reach the summariser as the same text"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#37")
def test_the_stamped_source_time_is_identifiable_in_the_summariser_source(tmp_path):
    """...and the instant itself has to be there, not merely a difference.

    Either rendering is accepted — the ISO day or the raw epoch seconds — because the format is
    the fix's to choose; withholding the instant is not.
    """
    engine = _engine(tmp_path, "instant37")
    try:
        serialized = engine._serialize_messages([
            {"role": "user", "content": "The meeting is tomorrow.", "timestamp": _MILLENNIUM},
        ])
        assert "2000-01-01" in serialized or str(int(_MILLENNIUM)) in serialized, (
            "the source time the store already holds is not offered to the summariser"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#37")
def test_an_unknown_source_time_is_not_substituted(tmp_path):
    """A turn the host never stamped must not be presented as if it had been.

    The store already separates `observed_at` (the host's stamp) from `ingested_at` (LCM's write
    time) and records which is which; the summariser source must keep that separation rather than
    letting the write time stand in for an event time nobody recorded.

    `unstamped != stamped` alone is not enough, and an earlier version of this test asserted only
    that: a fix rendering `observed_at or ingested_at` — substituting LCM's write time for an
    event time nobody recorded, which is precisely what #37 forbids — makes the two differ and
    passes. So the write time must not appear at all. Today's date and the current epoch second
    are what a substitution would put there; neither may show up beside a turn whose source time
    is unknown.

    Caveat, stated rather than hidden: a fix that renders the ingest time as its OWN labelled
    field alongside "source time unknown" would also trip this, and would arguably be compliant.
    Nothing in the code does that today; if a fix chooses to, this assertion is the one to revisit
    — not the contract.
    """
    now = time.time()
    today_iso = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
    engine = _engine(tmp_path, "unknown37")
    try:
        unstamped = engine._serialize_messages([
            {"role": "user", "content": "The meeting is tomorrow."},
        ])
        stamped = engine._serialize_messages([
            {"role": "user", "content": "The meeting is tomorrow.", "timestamp": _MILLENNIUM},
        ])
        assert unstamped != stamped, (
            "a turn with no source time is indistinguishable from one that has one"
        )
        assert today_iso not in unstamped, (
            "the write time was substituted for a source time the host never recorded"
        )
        assert str(int(now)) not in unstamped, (
            "the write time was substituted for a source time the host never recorded"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#37")
def test_the_source_time_is_labelled_as_the_source_time(tmp_path):
    """...and the instant that IS offered has to say which of the two times it is.

    Source time, LCM's ingest time and a node's bounds are three different things, and #37's
    whole point is that they must not collapse into one unqualified `timestamp`. A bare instant
    beside a turn is exactly that collapse: the reader cannot tell whether it is when the user
    said this or when the plugin happened to write it down.

    The label vocabulary below is a floor, not a specification — any of those words is accepted,
    because naming the field would be pinning an implementation the fix group has not chosen yet.
    """
    engine = _engine(tmp_path, "labelled37")
    try:
        serialized = engine._serialize_messages([
            {"role": "user", "content": "The meeting is tomorrow.", "timestamp": _MILLENNIUM},
        ])
        assert "2000-01-01" in serialized or str(int(_MILLENNIUM)) in serialized
        lowered = serialized.lower()
        assert any(label in lowered for label in ("source", "observed", "sent", "host time")), (
            "an instant reached the summariser with nothing saying which time it is"
        )
    finally:
        engine.shutdown()


# ── #31 MA01 — a real acknowledgement is not synthetic noise ─────────────────────────────────

@pytest.mark.beta_target("#31 MA01")
@pytest.mark.parametrize("window", WINDOWS)
def test_a_genuine_acknowledgement_survives_into_the_source_byte_identical(tmp_path, window):
    """The host's ordinary assistant builder produces exactly this shape.

    `content="Acknowledged"` with `finish_reason="stop"` is a real provider reply. The serialiser
    classifies it by WORDING, and because an envelope field is present it takes the branch that
    sets the content to "" — so `[ASSISTANT]:  [finish_reason=stop]` is what the summariser gets.
    Identifying synthetic origin needs a host signal the plugin does not have; guessing it from
    words deletes real text.
    """
    engine = _engine(tmp_path, "ack31", window)
    try:
        serialized = engine._serialize_messages([
            {"role": "assistant", "content": "Acknowledged",
             "finish_reason": "stop", "reasoning": None},
        ])
        assert "Acknowledged" in serialized, (
            "a genuine assistant reply was removed from the summariser source by its wording"
        )
    finally:
        engine.shutdown()


# ── #31 MC01 — a structured content list and its JSON text are different things ──────────────

@pytest.mark.beta_target("#31 MC01")
def test_a_structured_content_list_and_its_json_text_stay_distinguishable(tmp_path):
    """The host produces structured content lists for native image passing.

    Both the stored content and the envelope fingerprint normalise the list to its canonical
    JSON, so a native image part and a user who pasted that exact JSON become the same row and
    the same identity. Nothing later can tell them apart — not a reader, not a pattern filter,
    not change detection.

    # fork: better-hermes-lcm — this asserted `native["content"] == structured`, that the row
    # dict from `get_session_messages` returns a LIST. That is over-demanding and #31 says so in
    # as many words: "Originalcontent samt Typ dauerhaft eindeutig erhalten, getrennt von
    # Such-/Tokenprojektionen". The row dict IS the search/comparison projection — pattern
    # matching, search and token counting read it — and the issue names the conflation of the
    # original, the comparison form and the visible search text as the causal problem. Demanding
    # a list there merges the three representations it wants kept apart, and the claw model the
    # issue holds up as the useful partial principle stores the SAME `messages.content`
    # projection for both with DIFFERENT parts.
    #
    # So the assertion moves to the reconstruction, which is the reader whose job it is. That
    # keeps what the previous version was right about: a fingerprint inequality alone would pass
    # if the two were told apart only by a hash nobody can read back into a content list. It is
    # answered here by actually reading one back.
    #
    # The kind vocabulary is deliberately not pinned — the assertion is that the two differ and
    # that each reconstructs to its own original, so a fix may name the types what it likes.
    """
    structured = [
        {"type": "text", "text": "HEARTBEAT: Do not deploy."},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/chart.png"}},
    ]
    # the literal is the CANONICAL rendering, which is what a user quoting a payload back at the
    # agent would paste and what the normaliser produces for the list — anything else differs by
    # whitespace alone and would make this test pass for a reason that has nothing to do with the
    # defect.
    literal = normalize_content_value(structured)
    store = MessageStore(str(tmp_path / "types31.db"))
    try:
        native_id = store.append("t", {"role": "user", "content": structured}, source="cli")
        text_id = store.append("t", {"role": "user", "content": literal}, source="cli")
        # a row as a build that did not keep the type wrote it: the same canonical JSON as text,
        # with no envelope at all. "Altzeilen unbekannten Typs nicht als sicher typisiert
        # behandeln" is the half of MC01 that a fix is most likely to skip, because inferring the
        # type back from valid JSON syntax looks like a free upgrade and is the original defect
        # wearing a different hat.
        legacy_id = store.append("t", {"role": "user", "content": literal}, source="cli")
        store.commit()
        store.connection.execute(
            "UPDATE messages SET envelope_extra = NULL WHERE store_id = ?", (int(legacy_id),))
        store.commit()

        rows = {int(row["store_id"]): row for row in store.get_session_messages("t")}
        native, text, legacy = rows[int(native_id)], rows[int(text_id)], rows[int(legacy_id)]

        store_module = importlib.import_module("hermes_lcm.store")
        content_module = importlib.import_module("hermes_lcm.message_content")
        recorded_kind = getattr(store_module, "stored_content_kind", None)
        reconstruct = getattr(content_module, "original_content_from_stored", None)
        assert recorded_kind is not None and reconstruct is not None, (
            "nothing records a stored row's original type and nothing reads one back, so the "
            "store keeps only the canonical text both values flatten to and the roundtrip MC01 "
            "asks for has no reader at all"
        )

        assert message_envelope_fingerprint(native) != message_envelope_fingerprint(text), (
            "a native content list and a quoted copy of it have the same envelope identity"
        )
        assert recorded_kind(native) != recorded_kind(text), (
            "the two rows recorded the same original type"
        )
        assert reconstruct(native["content"], recorded_kind(native)) == structured, (
            "the native row does not read back as the list it was written as"
        )
        assert reconstruct(text["content"], recorded_kind(text)) == literal, (
            "the literal row does not read back as the string the user typed"
        )
        assert not isinstance(
            reconstruct(legacy["content"], recorded_kind(legacy)), (list, dict)
        ), "a row with no recorded type was typed from its JSON syntax, which is the guess MC01 removes"
    finally:
        store.close()


# ── #63 — an empty display turn with a real transport sidecar is still a turn ────────────────

@pytest.mark.beta_target("#63")
def test_an_empty_display_turn_with_an_api_sidecar_is_not_removed(tmp_path):
    """The host's own interrupt producer builds this turn deliberately.

    `_apply_active_turn_redirect` emits an assistant placeholder with empty display content and
    `api_content='[response interrupted]'`, and the request builder substitutes that sidecar as
    the turn's API text. The active-context cleaner inspects `content` and `tool_calls` only, so
    it deletes the whole message: the replayed conversation shows no turn where one happened, and
    leaves nothing in its place.
    """
    engine = _engine(tmp_path, "empty63")
    try:
        messages = [
            {"role": "user", "content": "Begin the rollout."},
            {"role": "assistant", "content": "", "display_kind": "hidden",
             "api_content": "[response interrupted]"},
            {"role": "user", "content": "Actually, cancel the rollout."},
        ]
        cleaned = engine._sanitize_active_context_messages(list(messages))

        assert len(cleaned) == 3, "the interrupted turn was deleted from the replayed context"
        assert cleaned[1].get("api_content") == "[response interrupted]", (
            "the sidecar the host substitutes as API text did not survive"
        )
    finally:
        engine.shutdown()
