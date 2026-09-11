"""Active-context message sanitization helpers.

Pure functions that shape the active replay context emitted back to providers:
strip internal/reasoning content from assistant messages, decide whether an
assistant message still has visible content, and detect sensitive-redaction
markers. Raw store and DAG history stay lossless -- these only sanitize the
active context, never stored rows.

Extracted verbatim from ``LCMEngine`` (WS5 seam 2). These depend only on
``escalation._strip_reasoning_blocks`` and each other, so this module never
imports the engine and introduces no import cycle.
"""

from __future__ import annotations

from typing import Any, Dict

from .escalation import _strip_reasoning_blocks
from .marked_loss import INTERNAL_REPLAY_MARKER, internal_replay_marker_part


_VISIBLE_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
_INTERNAL_ASSISTANT_PART_TYPES = {
    "analysis",
    "chain_of_thought",
    "internal",
    "reasoning",
    "redacted_thinking",
    "scratchpad",
    "thought",
    "thinking",
}


def _contains_sensitive_redaction(value: Any) -> bool:
    if isinstance(value, str):
        return "[LCM sensitive redaction:" in value
    if isinstance(value, dict):
        return any(
            _contains_sensitive_redaction(item)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, list):
        return any(_contains_sensitive_redaction(item) for item in value)
    return False


def _structured_part_text(part: Dict[str, Any]) -> str:
    """EVERY text-bearing field of the block, not just the first one found.

    A block can carry `text` (a reasoning stream) beside `content` (the visible answer, or a
    failure message). Returning the first key meant the visible sibling was invisible to the
    "does this block still say anything?" test, and the whole turn was dropped as internal-only
    (round-5 verify-6 #9).
    """
    collected: list[str] = []
    for key in ("text", "content", "value"):
        value = part.get(key)
        if isinstance(value, str):
            if value:
                collected.append(value)
            continue
        if isinstance(value, dict):
            for nested_key in ("value", "content", "text"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested:
                    collected.append(nested)
    return "\n".join(collected)


_STRUCTURAL_PART_KEYS = frozenset({"type", "cache_control", "text", "content", "value"})


def _part_has_substantive_fields(part: Any) -> bool:
    """Anything on this block other than its type and its (already inspected) text fields."""
    if not isinstance(part, dict):
        return False
    for key, value in part.items():
        if not isinstance(key, str) or key in _STRUCTURAL_PART_KEYS:
            continue
        if value is None or value == "" or value == [] or value == {}:
            continue
        return True
    return False


def _structured_part_has_visible_assistant_content(part: Any) -> bool:
    if part is None:
        return False
    if isinstance(part, str):
        return bool(_strip_reasoning_blocks(part).strip())
    if not isinstance(part, dict):
        return bool(str(part).strip())

    part_type = str(part.get("type") or "").strip().lower()
    if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
        return False
    if part_type in _VISIBLE_TEXT_PART_TYPES:
        if _strip_reasoning_blocks(_structured_part_text(part)).strip():
            return True
        # a typed text block whose text is empty can still carry the thing
        # that matters (annotations, a citation list, an outcome flag). Judging it by its text
        # alone dropped the whole turn with no receipt (round-5 verify-6 #9).
        return _part_has_substantive_fields(part)

    # Unknown non-internal content blocks may be visible (for example
    # images/audio/annotations in provider-specific formats).  Preserve
    # them rather than risk dropping a legitimate assistant turn.
    return True


def _assistant_message_has_visible_content(msg: Dict[str, Any]) -> bool:
    content = msg.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return bool(_strip_reasoning_blocks(content).strip())
    if isinstance(content, list):
        return any(_structured_part_has_visible_assistant_content(part) for part in content)
    if isinstance(content, dict):
        return _structured_part_has_visible_assistant_content(content)
    return bool(str(content).strip())


def _strip_structured_text_part(part: Dict[str, Any]) -> Dict[str, Any] | None:
    """strip EVERY text field of the block, and judge the block afterwards.

    This returned as soon as it had handled one key, so a block carrying `text` (reasoning)
    beside `content` (the visible answer) either kept the reasoning and never cleaned the
    sibling, or — when the first field stripped to nothing — threw the whole block away with
    its siblings and its outcome flags still on it (round-5 verify-6 #9).
    """
    cleaned = dict(part)
    for key in ("text", "content", "value"):
        value = cleaned.get(key)
        if isinstance(value, str):
            cleaned[key] = _strip_reasoning_blocks(value)
            continue
        if isinstance(value, dict):
            nested = dict(value)
            for nested_key in ("value", "content", "text"):
                nested_value = nested.get(nested_key)
                if isinstance(nested_value, str):
                    nested[nested_key] = _strip_reasoning_blocks(nested_value)
            cleaned[key] = nested
    return cleaned if _structured_part_has_visible_assistant_content(cleaned) else None


def _sanitize_active_assistant_content(content: Any) -> Any | None:
    if content is None:
        return None
    if isinstance(content, str):
        stripped = _strip_reasoning_blocks(content)
        return stripped if stripped.strip() else None
    if isinstance(content, list):
        cleaned_parts: list[Any] = []
        for part in content:
            if isinstance(part, str):
                stripped = _strip_reasoning_blocks(part)
                if stripped.strip():
                    cleaned_parts.append(stripped)
                continue
            if isinstance(part, dict):
                part_type = str(part.get("type") or "").strip().lower()
                if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
                    continue
                if part_type in _VISIBLE_TEXT_PART_TYPES:
                    cleaned_part = _strip_structured_text_part(part)
                    if cleaned_part is not None:
                        cleaned_parts.append(cleaned_part)
                    continue
            if _structured_part_has_visible_assistant_content(part):
                cleaned_parts.append(part)
        return cleaned_parts or None
    if isinstance(content, dict):
        part_type = str(content.get("type") or "").strip().lower()
        if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
            return None
        if part_type in _VISIBLE_TEXT_PART_TYPES:
            return _strip_structured_text_part(content)
        return content if _structured_part_has_visible_assistant_content(content) else None
    return content if str(content).strip() else None


def _mark_internal_removal(cleaned_content: Any) -> Any:
    """say, in the replay itself, that this turn was cut.

    Upstream stripped ``<think>`` (and reasoning/analysis parts) out of the assistant turns it
    replays and left nothing behind: the model saw a turn that silently differed from the one
    the store holds. Only ``_assemble_context`` counted the removals, so every other path that
    returns an active context (below-threshold cleanup, bypass trimming, forced overflow
    recovery) reported none. The receipt travels with the turn instead, so it is positionally
    neutral — it can never displace the caller's newest message — and it is a fixed string, so
    a replayed turn still matches its stored row through ``reconcile``.
    """
    if isinstance(cleaned_content, str):
        if INTERNAL_REPLAY_MARKER in cleaned_content:
            return cleaned_content
        text = cleaned_content.rstrip()
        return f"{text}\n{INTERNAL_REPLAY_MARKER}" if text else INTERNAL_REPLAY_MARKER
    if isinstance(cleaned_content, list):
        if any(INTERNAL_REPLAY_MARKER in str(part) for part in cleaned_content):
            return cleaned_content
        structured = any(isinstance(part, dict) for part in cleaned_content) or not cleaned_content
        return list(cleaned_content) + [internal_replay_marker_part(structured)]
    if isinstance(cleaned_content, dict):
        if INTERNAL_REPLAY_MARKER in str(cleaned_content):
            return cleaned_content
        return [cleaned_content, internal_replay_marker_part(True)]
    return cleaned_content


def _content_carries_text(value: Any) -> bool:
    """Did this content hold anything at all? A blank turn loses nothing when it is dropped.

    this asked a fixed list of text-bearing keys, so a reasoning block whose
    payload sat in `encrypted_content`, or a text block carrying only `annotations`, counted as
    empty and vanished with no receipt (round-5 verify-6 #9). Every key except the structural
    ones counts; only `type`/`cache_control` are scaffolding rather than content.
    """
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_content_carries_text(item) for item in value)
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key in ("type", "cache_control"):
                continue
            if isinstance(item, (str, list, dict)):
                if _content_carries_text(item):
                    return True
                continue
            if item is not None:
                return True
        return False
    return value is not None


# Message-level keys that record WHEN, WHERE and HOW a turn was captured rather than WHAT it
# said. The host stamps them on every message it builds, including the ones that really did say
# nothing, so a turn carrying only these still held nothing. Everything else is treated as
# payload — the same direction `_structured_part_has_visible_assistant_content` already takes
# for unknown content blocks, because dropping a legitimate turn is the worse error.
_MESSAGE_SCAFFOLDING_KEYS = frozenset({
    "role", "content",                                    # decided by the caller, above
    "tool_calls", "tool_call_id", "tool_name", "name",     # tool sequencing, decided separately
    "timestamp", "observed_at", "observed_at_source", "ingested_at",  # when
    "display_kind", "finish_reason", "cache_control",      # how it was rendered/ended
    "message_id", "id", "uuid", "event_id", "host_message_id",  # which message
    "store_id", "session_id", "conversation_id", "source", "token_estimate", "pinned",
    "envelope", "envelope_extra", "envelope_corrupt", "envelope_raw", "envelope_raw_chars",
    "content_kind",                                        # the store's record of the type
})


def _message_carries_more_than_its_content(msg: Dict[str, Any]) -> bool:
    """Does this turn hold anything BESIDES the content already found to be empty?

    The existence of a whole host message used to be decided from ``content`` and
    ``tool_calls`` alone: ``_content_carries_text`` is handed the content VALUE and never sees
    the message, so a sibling field carrying the turn's actual payload could not keep it alive.
    Hermes' user-redirect placeholder is exactly that shape — empty display content,
    ``display_kind='hidden'`` and ``api_content='[response interrupted]'``, the string the host
    substitutes into its own API copy — and it was removed wholesale from the replayed context
    with no marker, so the context showed no turn where one happened. Deciding a host message
    does not exist is not LCM's to make.
    """
    for key, value in msg.items():
        if not isinstance(key, str) or key in _MESSAGE_SCAFFOLDING_KEYS:
            continue
        if key.startswith("lcm_") or key.startswith("_lcm"):
            continue  # this plugin's own bookkeeping, not something the host sent
        if _content_carries_text(value):
            return True
    return False


def _clean_active_assistant_message(msg: Dict[str, Any]) -> Dict[str, Any] | None:
    if msg.get("role") != "assistant":
        return msg
    if "content" not in msg:
        return msg
    original_content = msg.get("content")
    cleaned_content = _sanitize_active_assistant_content(original_content)
    if cleaned_content is None:
        # a turn holding ONLY internal content is not dropped without a
        # trace either: the receipt takes its place, so the model still sees that a turn
        # happened here and can read it whole from the store. A turn that held NOTHING is
        # still dropped outright — an empty turn loses nothing, and inventing a receipt for
        # it would be a false claim of removal.
        if not _content_carries_text(original_content):
            if msg.get("tool_calls") or _message_carries_more_than_its_content(msg):
                return msg
            return None
        cleaned_content = ""
    if cleaned_content == original_content:
        return msg
    cleaned = dict(msg)
    cleaned["content"] = _mark_internal_removal(cleaned_content)
    return cleaned


def _is_internal_replay_receipt_only(msg: Any) -> bool:
    """A turn whose whole replayed body is the internal-removal receipt.

    such a turn carries no content of its own, so under assembly budget
    pressure it is dropped and named in the prefix's omission marker instead of competing
    with the caller's live messages for room.
    """
    if not isinstance(msg, dict) or msg.get("role") != "assistant" or msg.get("tool_calls"):
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip() == INTERNAL_REPLAY_MARKER
    if isinstance(content, list) and len(content) == 1:
        part = content[0]
        if isinstance(part, str):
            return part.strip() == INTERNAL_REPLAY_MARKER
        if isinstance(part, dict):
            return str(part.get("text") or "").strip() == INTERNAL_REPLAY_MARKER
    return False


def _should_drop_active_assistant_message(msg: Dict[str, Any]) -> bool:
    if msg.get("role") != "assistant":
        return False
    if msg.get("tool_calls"):
        return False
    return _clean_active_assistant_message(msg) is None
