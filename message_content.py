"""Message content normalization helpers.

Hermes/OpenAI-format messages may carry ``content`` as plain text or as
structured content parts (for example text + image blocks). LCM persists and
accounts for message content as text, so all write/matching/token paths should
use deliberate normalization.
"""

from __future__ import annotations

import json
from typing import Any

_TEXT_PART_TYPES = {"text", "input_text", "output_text"}

# The store's own record of what the host's ``content`` WAS: written into ``envelope_extra``
# beside the projected text, and published back at row level under this same name. The `_lcm`
# prefix is the one every generic consumer of a message dict already skips — the envelope
# writer, ``MessageStore.to_openai_msg``, ``LCMEngine._message_envelope_fields`` and the
# active cleaner — so the record can never be replayed to a provider, never be counted as a
# host field the summariser did not summarise, and never be spoofed by a host key of the same
# name. It is the store's bookkeeping, and it has to be unmistakable as such at both levels.
CONTENT_KIND_KEY = "_lcm_content_kind"

CONTENT_KIND_STRING = "str"
CONTENT_KIND_LIST = "list"
CONTENT_KIND_DICT = "dict"
CONTENT_KIND_NONE = "none"
CONTENT_KIND_OTHER = "other"
# A row stored before the type was recorded. It is NOT a guess to be resolved later: valid
# JSON syntax is not evidence of having been JSON, so unknown stays unknown.
CONTENT_KIND_UNKNOWN = "unknown"


def content_kind(content: Any) -> str:
    """What the content VALUE is, as handed to the writer.

    It describes the value the row will hold, not what the host sent before ingest protection
    may have rewritten it — that is what makes the stored text reconstructable from it.
    """
    if content is None:
        return CONTENT_KIND_NONE
    if isinstance(content, str):
        return CONTENT_KIND_STRING
    if isinstance(content, list):
        return CONTENT_KIND_LIST
    if isinstance(content, dict):
        return CONTENT_KIND_DICT
    return CONTENT_KIND_OTHER


def text_is_ambiguously_typed(text: Any) -> bool:
    """Could this stored text have been either a structured value or a string of its JSON?

    Only then does the recorded type change anything: for every other text the value itself
    fixes the type, and callers can leave their comparisons exactly as they were.
    """
    # the cheap test first: the canonical form of a list or a dict opens with its own bracket
    # and carries no leading space, so any other text cannot equal one and must not pay for a
    # parse. This runs inside `message_envelope_fingerprint`, which the per-turn
    # prefix-revision path calls for every message carrying a host id — the path that was
    # explicitly optimised because reading whole prefixes tripled the ingest hot path.
    if not isinstance(text, str) or not text or text[0] not in "[{":
        return False
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(decoded, (list, dict)) and normalize_content_value(decoded) == text


def original_content_from_stored(stored_text: Any, kind: Any) -> Any:
    """The original value a stored row held, given the type recorded with it.

    Fails closed for every recorded type, ``none`` included: when the type no longer describes
    the stored text — ingest protection and tool-result GC rewrite a row's content in place,
    so a row recorded as having held nothing can now hold a placeholder — the stored text is
    returned as it stands rather than a shape invented from it, and a row whose text really is
    absent still returns ``None`` because that is what its text is. A caller that must know
    compares ``content_kind(result)`` against the recorded kind.
    """
    if kind in (CONTENT_KIND_LIST, CONTENT_KIND_DICT) and isinstance(stored_text, str):
        try:
            decoded = json.loads(stored_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return stored_text
        expected = list if kind == CONTENT_KIND_LIST else dict
        if isinstance(decoded, expected) and normalize_content_value(decoded) == stored_text:
            return decoded
    return stored_text


def _extract_text_part_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, str):
            return nested
        nested = value.get("content")
        if isinstance(nested, str):
            return nested
    return None


def normalize_content_value(content: Any) -> str | None:
    """Return a stable text representation for message content.

    ``None`` remains ``None`` so callers that distinguish SQL NULL from an empty
    string can preserve that behavior. Strings are returned unchanged. Structured
    content is serialized deterministically so storage, source-id matching, and
    token accounting all see the same value.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(content)


def text_content_for_pattern_matching(content: Any) -> str | None:
    """Return the operator-visible text string used by message filters.

    Structured multimodal payloads often arrive as lists of content parts. For
    ignore-pattern matching, prefer concatenated text parts so anchored patterns
    bind to the text an operator sees. If no text parts are present, fall back to
    the stable normalized representation used for storage.
    """
    if content is None or isinstance(content, str):
        return normalize_content_value(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type in _TEXT_PART_TYPES:
                    text = _extract_text_part_value(part.get("text"))
                    if text is None:
                        text = _extract_text_part_value(part.get("content"))
                    if text:
                        parts.append(text)
        if parts:
            return "\n".join(parts)
    return normalize_content_value(content)


def stored_text_content_for_pattern_matching(content: Any, kind: Any = None) -> str | None:
    """Return message-filter text for content read back from storage.

    Structured content is persisted as canonical JSON, so a row that WAS a structured value
    has to be decoded before the text-first ignore policy reads the text it reads live. Which
    rows those are used to be GUESSED, by decoding any stored string that round-trips to the
    same canonical text: a user message whose literal text happened to be that JSON was
    decoded too, and started matching a pattern it had not matched live.

    ``kind`` is the type recorded with the row (``MessageStore`` writes it on every row it
    stores) and answers it exactly — including ``CONTENT_KIND_UNKNOWN`` for a row written
    before the type was kept, where the answer is "nobody knows", so nothing is decoded and
    valid JSON syntax is not read as evidence of having been JSON.

    Omitting ``kind`` keeps the old guess. It is not a default the store needs — every row it
    hands back carries ``CONTENT_KIND_KEY`` — but the one caller that reads stored rows against
    the ignore patterns still drops that field before it gets here
    (``LCMEngine._matches_ignore_message_patterns``), and without the type the guess is the
    only thing that still recognises a durable structured row as the one the live ignore
    policy filtered out. Removing the guess before that caller passes the field made restart
    reconciliation miss such a row and re-ingest the turn before it as a duplicate.
    """
    if kind is None:
        if isinstance(content, str):
            try:
                decoded = json.loads(content)
            except (TypeError, ValueError, json.JSONDecodeError):
                return text_content_for_pattern_matching(content)
            if isinstance(decoded, (list, dict)) and normalize_content_value(decoded) == content:
                return text_content_for_pattern_matching(decoded)
        return text_content_for_pattern_matching(content)
    if kind in (CONTENT_KIND_LIST, CONTENT_KIND_DICT) and isinstance(content, str):
        decoded = original_content_from_stored(content, kind)
        if decoded is not content:
            return text_content_for_pattern_matching(decoded)
    return text_content_for_pattern_matching(content)
