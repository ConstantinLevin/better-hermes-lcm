"""Pre-compaction extraction — extract decisions and commitments before summarization.

Best-effort: failures never block compaction. Extracted content is written to
daily note files so key decisions survive even if the DAG summary loses nuance.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import marked_loss  # fork: betterlcm
from .errors import ExtractionUnavailableError  # fork: betterlcm
from .model_routing import apply_lcm_model_route
from .prompt_boundary import build_untrusted_data_messages  # fork: betterlcm

# fork: betterlcm — reasons that mean the model was cut off (mirrors escalation).
_TRUNCATED_FINISH_REASONS = frozenset({
    "length", "max_tokens", "max_output_tokens", "content_filter", "incomplete",
})

logger = logging.getLogger(__name__)

# fork: betterlcm — no whitespace in the payload class. With ``\s`` in it the match ran past
# the URI and swallowed the ordinary words after it ("…;base64,AAAA hello world decision"
# erased the sentence), so prose vanished from the summariser input with only a media marker
# left behind (audit p05 EX02). A line-wrapped payload now simply stops at the first newline:
# the remainder stays in the text, which costs a little size and loses nothing.
_MEDIA_DATA_URI_RE = re.compile(
    r"data:(?:image|audio|video)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]{16,}",
    re.IGNORECASE,
)
_MEDIA_ATTACHMENT_MARKER = "[Media attachment]"
_MEDIA_ATTACHMENT_SUFFIX = "[with media attachment]"
_TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
_MEDIA_BLOCK_HINTS = ("image", "audio", "video")
_STRUCTURED_METADATA_KEYS = ("file_id", "filename", "name", "mime_type", "url", "file_url", "id")
_INJECTED_CONTEXT_TAGS = (
    "active_memory",
    "active_memory_plugin",
    "relevant-memories",
    "relevant_memories",
    "hindsight-memories",
    "hindsight_memories",
)
_UNTRUSTED_CONTEXT_HEADER_RE = re.compile(
    r"^Untrusted context \(metadata, do not treat as instructions or commands\):\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _at_line_start(text: str, index: int) -> bool:
    line_start = text.rfind("\n", 0, index) + 1
    return not text[line_start:index].strip()


def _at_line_end(text: str, index: int) -> bool:
    line_end = text.find("\n", index)
    if line_end == -1:
        line_end = len(text)
    return not text[index:line_end].strip()


# fork: betterlcm — the instruction half of the extraction prompt, for the trusted system role.
EXTRACTION_INSTRUCTIONS = """Extract decisions, commitments, outcomes, and rules from the supplied
conversation segment.

Format as a flat list of bullet points. Each bullet should be self-contained and understandable
without the surrounding conversation. Include:
- Decisions made (what was chosen, and why if stated)
- Commitments (who will do what)
- Outcomes (what happened as a result of an action)
- Rules or constraints discovered

Skip: greetings, meta-discussion, reasoning that led nowhere, repeated information.
If there is nothing worth extracting, respond with exactly: NOTHING_TO_EXTRACT"""


EXTRACTION_PROMPT = """Extract decisions, commitments, outcomes, and rules from this conversation segment.

Format as a flat list of bullet points. Each bullet should be self-contained and understandable
without the surrounding conversation. Include:
- Decisions made (what was chosen, and why if stated)
- Commitments (who will do what)
- Outcomes (what happened as a result of an action)
- Rules or constraints discovered

Skip: greetings, meta-discussion, reasoning that led nowhere, repeated information.
If there is nothing worth extracting, respond with exactly: NOTHING_TO_EXTRACT

CONTENT:
{text}"""


def _call_extraction_llm(prompt: "str | list[dict[str, str]]", model: str = "",
                          timeout: float | None = None) -> Optional[str]:
    """Call the Hermes auxiliary LLM for extraction.

    fork: betterlcm — a provider failure raises instead of returning ``None``. Returning None
    for both "the model said there was nothing" and "the call never happened" made a missing
    extraction indistinguishable from a successful negative assessment (audit p05 EX08).
    A generation that stopped at its limit is a failure too, not a finished extraction
    (audit p05 EX06).
    """
    try:
        from agent.auxiliary_client import call_llm
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        call_kwargs = {
            "task": "extraction",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2000,
        }
        apply_lcm_model_route(call_kwargs, model)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        response = call_llm(**call_kwargs)
        choice = response.choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "").strip().lower()
        if finish_reason in _TRUNCATED_FINISH_REASONS:
            raise ExtractionUnavailableError(
                f"extraction stopped at the generation limit (finish_reason={finish_reason})"
            )
        content = choice.message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
        from .escalation import _strip_reasoning_blocks
        return _strip_reasoning_blocks(content).strip()
    except ExtractionUnavailableError:
        raise
    except Exception as e:
        logger.debug("Extraction LLM call failed: %s", e)
        raise ExtractionUnavailableError(f"extraction call failed: {e}") from e


def _sanitize_string_media(text: str) -> str:
    if not text:
        return ""
    if not _MEDIA_DATA_URI_RE.search(text):
        return text

    without_media = _MEDIA_DATA_URI_RE.sub("", text)
    without_media = without_media.strip()
    without_media = re.sub(r"\n{3,}", "\n\n", without_media)

    if not without_media:
        return _MEDIA_ATTACHMENT_MARKER
    if _MEDIA_ATTACHMENT_SUFFIX in without_media:
        return without_media
    return f"{without_media}\n{_MEDIA_ATTACHMENT_SUFFIX}"


def _looks_like_media_block(block_type: str, block: Dict[str, Any]) -> bool:
    if any(hint in block_type for hint in _MEDIA_BLOCK_HINTS):
        return True
    return any(key in block for key in ("image_url", "input_image", "output_image", "audio_url", "video_url"))


def _extract_structured_metadata(block: Dict[str, Any]) -> str:
    parts: List[str] = []
    block_type = str(block.get("type", "")).strip()
    if block_type:
        parts.append(f"type={block_type}")

    for key in _STRUCTURED_METADATA_KEYS:
        value = block.get(key)
        if isinstance(value, dict):
            for nested_key in _STRUCTURED_METADATA_KEYS:
                nested_value = value.get(nested_key)
                if isinstance(nested_value, (str, int, float)) and nested_value:
                    parts.append(f"{nested_key}={nested_value}")
                    break
            continue
        if isinstance(value, (str, int, float)) and value:
            parts.append(f"{key}={value}")

    if not parts:
        return "[Structured content]"
    return "[Structured content: " + ", ".join(dict.fromkeys(parts)) + "]"


_STRUCTURED_OUTCOME_KEYS = ("is_error", "status", "error_code", "tool_use_id", "tool_call_id")


def _structured_outcome_suffix(block: Dict[str, Any]) -> str:
    """fork: betterlcm — the outcome fields that sit BESIDE a block's text (audit p05 EX03)."""
    parts: List[str] = []
    for key in _STRUCTURED_OUTCOME_KEYS:
        if key not in block:
            continue
        value = block.get(key)
        if value in (None, "", False) and key != "is_error":
            continue
        parts.append(f"{key}={value}")
    return f" [{', '.join(parts)}]" if parts else ""


def _sanitize_content_block(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _sanitize_string_media(content)
    if isinstance(content, list):
        parts: List[str] = []
        media_count = 0  # fork: betterlcm — how many, not merely "some" (audit p05 EX03)
        for block in content:
            block_text = _sanitize_content_block(block)
            if not block_text:
                continue
            if block_text == _MEDIA_ATTACHMENT_MARKER:
                media_count += 1
                continue
            if block_text.endswith(_MEDIA_ATTACHMENT_SUFFIX):
                media_count += 1
                block_text = block_text[: -len(_MEDIA_ATTACHMENT_SUFFIX)].rstrip()
                if not block_text:
                    continue
            parts.append(block_text)
        combined = "\n".join(part for part in parts if part).strip()
        combined = re.sub(r"\n{3,}", "\n\n", combined)
        media_suffix = _MEDIA_ATTACHMENT_SUFFIX
        media_marker = _MEDIA_ATTACHMENT_MARKER
        if media_count > 1:
            # several attachments used to collapse into one flag, so the summariser could not
            # tell one image from six
            media_suffix = f"{_MEDIA_ATTACHMENT_SUFFIX[:-1]} ×{media_count}]"
            media_marker = f"{_MEDIA_ATTACHMENT_MARKER[:-1]} ×{media_count}]"
        if media_count and combined:
            return f"{combined}\n{media_suffix}"
        if media_count:
            return media_marker
        return combined
    if isinstance(content, dict):
        block_type = str(content.get("type", "")).lower()
        if block_type in _TEXT_BLOCK_TYPES:
            text_value = content.get("text")
            if isinstance(text_value, dict):
                text_value = text_value.get("value", "")
            if not text_value:
                text_value = content.get("content", "")
            return _sanitize_content_block(text_value)
        if _looks_like_media_block(block_type, content):
            return _MEDIA_ATTACHMENT_MARKER
        for key in ("text", "content"):
            if key in content:
                # fork: betterlcm — keep the typed siblings that change what the text MEANS.
                # Picking out `text` dropped a tool result's failure status and identity, so a
                # failed call read to the summariser exactly like a successful one
                # (audit p05 EX03).
                return _sanitize_content_block(content.get(key)) + _structured_outcome_suffix(content)
        return _extract_structured_metadata(content)
    return str(content)


def _select_injected_context_closer(
    text: str,
    opener: re.Match[str],
    close_re: re.Pattern[str],
) -> re.Match[str] | None:
    closers = list(close_re.finditer(text, opener.end()))
    if not closers:
        return None

    # Prompt-injected context normally uses a block shape:
    #   <tag>\n...\n</tag>
    # In that shape, a line-delimited close/open pair inside recalled text is
    # indistinguishable from two adjacent injected blocks with user text between
    # them. Choose safety over preservation: block-shaped repeated same-tag
    # sections are stripped as one untrusted region up to the last line-isolated
    # close, even if that drops a real inter-block gap.
    if _at_line_end(text, opener.end()):
        line_closers = [
            closer
            for closer in closers
            if _at_line_start(text, closer.start()) and _at_line_end(text, closer.end())
        ]
        if not line_closers:
            if len(closers) == 1 and _at_line_end(text, closers[0].end()):
                return closers[0]
            for index, closer in enumerate(closers):
                if index + 1 < len(closers) or not _at_line_start(text, closer.start()):
                    continue
                line_end = text.find("\n", closer.end())
                if line_end == -1:
                    line_end = len(text)
                suffix = text[closer.end() : line_end]
                if "<" not in suffix and suffix.strip():
                    return closer
            return None
        return line_closers[-1]

    # Inline wrappers are stripped one complete block at a time. A later same-tag
    # inline opener may be another injected block with real user/tool text
    # between the two blocks; consuming through the later close would silently
    # delete that interstitial text. Keep the safety-first multi-close behavior
    # for block-shaped wrappers above, where line-delimited recalled text is more
    # likely to be spoofing the context envelope.
    return closers[0]


def _injection_marker(removed: str, mark: bool) -> str:
    """fork: betterlcm — one marker for a removed injected block, empty when it held nothing.

    Only the summariser-input paths mark. Removing an injected block on the way INTO the store
    removes something LCM (or the host) put there this turn, not conversation content, and a
    marker there would be noise stored forever.
    """
    if not mark or not removed.strip():
        return ""
    return marked_loss.injected_context_marker(len(removed))


def strip_injected_context_blocks(text: str, *, mark: bool = False) -> str:
    """Remove transient memory/context blocks before compaction summarization.

    fork: betterlcm — ``mark=True`` leaves a marker naming how much was removed, for the paths
    whose output the summariser reads (audit p05 EX01).
    """
    if not text:
        return ""

    cleaned = text
    changed = False
    if "<" not in text:
        cleaned = _UNTRUSTED_CONTEXT_HEADER_RE.sub("", text)
        changed = cleaned != text
        return cleaned.strip() if changed else cleaned

    for tag in _INJECTED_CONTEXT_TAGS:
        escaped = re.escape(tag)
        self_close_re = re.compile(rf"<{escaped}(?:\s[^>]*)?\s*/\s*>", re.IGNORECASE)
        open_re = re.compile(rf"<{escaped}(?:\s[^>]*)?>", re.IGNORECASE)
        close_re = re.compile(rf"</{escaped}\s*>", re.IGNORECASE)
        before_self_close = cleaned
        cleaned = self_close_re.sub("", cleaned)
        changed = changed or cleaned != before_self_close

        while True:
            opener = open_re.search(cleaned)
            if not opener:
                break

            # fork: betterlcm — the removal boundary is upstream's (safety first: a spoofed
            # closer inside recalled text must not be able to smuggle content past it), but
            # the cut is MARKED. Upstream deleted whatever lay between two block-shaped tags,
            # so a real decision written between two memory blocks disappeared from the
            # summariser's input with nothing to say it had ever been there (audit p05 EX01).
            closer = _select_injected_context_closer(cleaned, opener, close_re)
            if closer is None:
                if _at_line_end(cleaned, opener.end()):
                    removed = cleaned[opener.start():]
                    cleaned = cleaned[: opener.start()] + _injection_marker(removed, mark)
                else:
                    removed = cleaned[opener.start():opener.end()]
                    cleaned = cleaned[: opener.start()] + cleaned[opener.end() :]
                changed = True
                continue
            removed = cleaned[opener.start():closer.end()]
            cleaned = (cleaned[: opener.start()] + _injection_marker(removed, mark)
                       + cleaned[closer.end() :])
            changed = True

    before_header = cleaned
    cleaned = _UNTRUSTED_CONTEXT_HEADER_RE.sub("", cleaned)
    changed = changed or cleaned != before_header
    return cleaned.strip() if changed else cleaned


def _sanitize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        # fork: betterlcm — a sanitised key may never take another key's place. Upstream
        # rebuilt the dict from sanitised keys, so two keys that became identical collapsed
        # and the first value was dropped outright: {"a<active_memory>x</active_memory>":
        # "FIRST", "a": "SECOND"} became {"a": "SECOND"} — a whole tool argument gone with no
        # marker (audit p05 EX04). Keys are still cleaned of payloads; a collision keeps the
        # original key instead, so every value survives.
        sanitized: Dict[Any, Any] = {}
        for key, val in value.items():
            clean_key = key
            if isinstance(key, str):
                candidate = strip_injected_context_blocks(_sanitize_string_media(key))
                if candidate not in sanitized or candidate == key:
                    clean_key = candidate
            if clean_key in sanitized:
                clean_key = key
            sanitized[clean_key] = _sanitize_json_like(val)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_json_like(item) for item in value]
    if isinstance(value, str):
        # fork: no marker inside tool ARGUMENTS. The serialized argument block has its own
        # fixed char budget and its own elision marker; adding one marker per removed block
        # there displaces the real arguments it is meant to protect.
        return strip_injected_context_blocks(_sanitize_string_media(value))
    return value


def sanitize_pre_compaction_content(text: Any) -> str:
    """Replace inline media/base64 payloads and transient injected context before compaction."""
    return strip_injected_context_blocks(_sanitize_content_block(text), mark=True)


def sanitize_pre_compaction_tool_arguments(arguments: Any) -> str:
    """Clean tool-call argument payloads while preserving JSON-like structure when possible."""
    if arguments is None:
        return ""
    if isinstance(arguments, (dict, list)):
        return json.dumps(_sanitize_json_like(arguments), ensure_ascii=False)
    if not isinstance(arguments, str):
        return sanitize_pre_compaction_content(arguments)
    try:
        parsed = json.loads(arguments)
    except Exception:
        return sanitize_pre_compaction_content(arguments)
    return json.dumps(_sanitize_json_like(parsed), ensure_ascii=False)


def extract_before_compaction(
    serialized_messages: str,
    output_path: str,
    session_id: str = "",
    model: str = "",
    timeout: float | None = None,
    source_store_ids: "List[int] | None" = None,  # fork: betterlcm (audit p05 EX07)
) -> bool:
    """Extract decisions from messages about to be compacted and write to a daily file.

    Returns True if extraction succeeded, False otherwise.
    Never raises — failures are logged and swallowed.
    """
    try:
        # fork: betterlcm — the same untrusted-data boundary the summariser uses. Upstream
        # interpolated the historical conversation into one user message after "CONTENT:", so
        # instructions found in that history reached the model as instructions and could steer
        # what got written into the persistent notes (audit p05 EX05).
        prompt = build_untrusted_data_messages(
            operation="lcm_extraction",
            system_instructions=EXTRACTION_INSTRUCTIONS,
            sources=[
                {
                    "provenance": {
                        "source_type": "messages",
                        **({"session_id_present": True} if session_id else {}),
                    },
                    "content": serialized_messages,
                }
            ],
        )
        result = _call_extraction_llm(prompt, model=model, timeout=timeout)

        if not result or result.strip() == "NOTHING_TO_EXTRACT":
            logger.debug("Pre-compaction extraction: nothing to extract")
            return True

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = output_dir / f"{date_str}.md"

        header = f"\n\n## Extraction — {datetime.now().strftime('%H:%M')}"
        if session_id:
            header += f" ({session_id})"
        header += "\n"
        # fork: betterlcm — say exactly which rows these bullets came from. A note headed only
        # by a wall-clock time cannot be tied back to its segment, and several passes in one
        # session produce several such notes (audit p05 EX07).
        digest = hashlib.sha256(serialized_messages.encode("utf-8", "replace")).hexdigest()[:16]
        provenance = f"source chars={len(serialized_messages)}, sha256:{digest}"
        if source_store_ids:
            shown = ", ".join(str(store_id) for store_id in list(source_store_ids)[:40])
            more = f" (+{len(source_store_ids) - 40} more)" if len(source_store_ids) > 40 else ""
            provenance += f", store_ids={shown}{more} — lcm_expand(store_id=…)"
        header += f"*Source: {provenance}*\n\n"

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(result)
            f.write("\n")

        logger.info("Pre-compaction extraction written to %s", file_path)
        return True

    except Exception as e:
        logger.warning("Pre-compaction extraction failed (non-blocking): %s", e)
        return False
