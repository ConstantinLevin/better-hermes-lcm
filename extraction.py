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

from . import marked_loss
from .errors import ExtractionUnavailableError
from .model_routing import apply_lcm_model_route
from .prompt_boundary import build_untrusted_data_messages

# reasons that mean the model was cut off (mirrors escalation).
_TRUNCATED_FINISH_REASONS = frozenset({
    "length", "max_tokens", "max_output_tokens", "content_filter", "incomplete",
})

logger = logging.getLogger(__name__)

# An inline `data:` URI in ordinary TEXT is left alone. A regex used to delete it and put a
# media marker in its place, and no spelling of that regex is safe: the conservative form
# matched only as far as the first newline, so a line-wrapped payload lost one line while the
# marker said the medium had been removed — a receipt claiming more than it took, with the
# base64 still in the text (#56). The permissive form eats the words after the payload
# instead (claw's does exactly that to "…AAAA\nAAAA\nSTOP\nDo not deploy."), and the earlier
# attempts to bound it by wrap width did the same here. There is no third spelling to find:
# LCM does not cut the source at all, and a payload the configured summary route cannot
# process fails at that route, which is the only place that knows its real capacity.
# Structured media BLOCKS still project to the marker below — that is a representation
# question (#31 MC01), not this removal.
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


# the instruction half of the extraction prompt, for the trusted system role.
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

    a provider failure raises instead of returning ``None``. Returning None
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
    """the outcome fields that sit BESIDE a block's text (audit p05 EX03)."""
    parts: List[str] = []
    for key in _STRUCTURED_OUTCOME_KEYS:
        if key not in block:
            continue
        value = block.get(key)
        if value in (None, "", False) and key != "is_error":
            continue
        parts.append(f"{key}={value}")
    return f" [{', '.join(parts)}]" if parts else ""


# Keys a rendered block already accounts for: its own type, the streams that are rendered, the
# outcome fields appended as a suffix, and the identity fields the metadata form prints.
# keys that carry no content of their own. Everything else is inventoried by
# the branch that did NOT render it: exempting "text"/"content" globally hid them in the media
# branch, which renders neither, and citations/annotations are substantive (round-3 verify-4 #9).
_ACCOUNTED_BLOCK_KEYS = frozenset(
    ("type", "cache_control") + _STRUCTURED_OUTCOME_KEYS + _STRUCTURED_METADATA_KEYS
)


def _unrendered_field_receipt(block: Dict[str, Any], rendered_keys, prefix: str = "") -> str:
    """name substantive fields the rendering does not show.

    A typed text block carrying ``{"text": "stdout", "content": "stderr FAILED",
    "extra": "FATAL"}`` rendered "stdout" and dropped the rest with nothing in its place
    (round-2 verify-4 #14). The stored row is unchanged; this says what the summariser is not
    being shown, and where to read it.
    """
    omitted: List[str] = []
    for key, value in block.items():
        if not isinstance(key, str):
            continue
        if key in rendered_keys or key in _ACCOUNTED_BLOCK_KEYS:
            continue
        # 0 and False are VALUES, not absence (round-3 verify-4 #9)
        if value is None or value == "" or value == [] or value == {}:
            continue
        omitted.append(f"{prefix}{key}")
    if not omitted:
        return ""
    shown = ", ".join(sorted(omitted)[:10])
    more = f" (+{len(omitted) - 10} more)" if len(omitted) > 10 else ""
    return (
        f"\n[LCM: {len(omitted)} further field(s) of this block are not rendered here "
        f"({shown}{more}); the stored message is unchanged — lcm_expand]"
    )


def _sanitize_content_block(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content  # text is offered as the host wrote it (#56)
    if isinstance(content, list):
        parts: List[str] = []
        media_count = 0  # how many, not merely "some" (audit p05 EX03)
        for block in content:
            block_text = _sanitize_content_block(block)
            if not block_text:
                continue
            if block_text.startswith(_MEDIA_ATTACHMENT_MARKER):
                # a media block can now carry a receipt for the substantive
                # fields beside it (round-2 verify-4 #14); count the attachment and keep the
                # receipt as its own part rather than losing both to an equality test.
                media_count += 1
                remainder = block_text[len(_MEDIA_ATTACHMENT_MARKER):].strip()
                if remainder:
                    parts.append(remainder)
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
            nested_receipt = ""
            if isinstance(text_value, dict):
                # a NESTED text object's other fields (annotations, a
                # citation list, a status) were dropped without a receipt: the outer
                # inventory only sees the outer block's keys (round-4).
                nested_source = text_value
                nested_key = "value" if nested_source.get("value") not in (None, "") else "text"
                text_value = nested_source.get("value", "") or nested_source.get("text", "")
                nested_receipt = _unrendered_field_receipt(
                    nested_source, (nested_key,), prefix="text."
                )
            # a typed TEXT block can still carry an outcome beside its text
            # (is_error, a status, its call id). Returning only the text made a failed step
            # read exactly like a successful one (verify-4 #6). And it can carry a SECOND
            # stream: `text` plus `content` (stdout and stderr) — rendering one and dropping
            # the other lost a whole output stream even in the typed branch
            # (round-2 verify-4 #14).
            rendered_parts: List[str] = []
            rendered_keys: List[str] = []
            for key, value in (("text", text_value), ("content", content.get("content"))):
                if value in (None, ""):
                    continue
                part = _sanitize_content_block(value)
                if part and part not in rendered_parts:
                    rendered_parts.append(part)
                    rendered_keys.append(key)
            return (
                "\n".join(rendered_parts)
                + _structured_outcome_suffix(content)
                + _unrendered_field_receipt(content, rendered_keys)
                + nested_receipt
            )
        if _looks_like_media_block(block_type, content):
            # the marker stands for the media; anything substantive beside it is still named
            # the marker stands for the media; the branch renders no field of its own, so
            # every substantive key is inventoried (round-3 verify-4 #9)
            return (
                _MEDIA_ATTACHMENT_MARKER
                + _structured_outcome_suffix(content)
                + _unrendered_field_receipt(content, ())
            )
        # a block may carry BOTH `text` and `content` (stdout and stderr, for
        # example). Taking the first and ignoring the second dropped a whole output stream
        # (verify-4 #6). Keep every distinct one, then the typed siblings that change what the
        # text MEANS: picking out `text` alone dropped a tool result's failure status and
        # identity, so a failed call read exactly like a successful one (audit p05 EX03).
        rendered_parts: List[str] = []
        for key in ("text", "content"):
            if key not in content:
                continue
            part = _sanitize_content_block(content.get(key))
            if part and part not in rendered_parts:
                rendered_parts.append(part)
        if rendered_parts:
            return (
                "\n".join(rendered_parts)
                + _structured_outcome_suffix(content)
                + _unrendered_field_receipt(content, ("text", "content"))
            )
        return _extract_structured_metadata(content) + _unrendered_field_receipt(content, ())
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


def _injection_marker(removed: str, mark: bool, compact: bool = False) -> str:
    """one marker for a removed injected block, empty when it held nothing.

    Only the summariser-input paths mark. Removing an injected block on the way INTO the store
    removes something LCM (or the host) put there this turn, not conversation content, and a
    marker there would be noise stored forever. ``compact`` is the short form used inside tool
    arguments, where the block has its own character budget (verify-4 #7).
    """
    if not mark or not removed.strip():
        return ""
    if compact:
        return f"[LCM-{len(removed)}c]"
    return marked_loss.injected_context_marker(len(removed))


def _mark_header_removal(text: str, mark: bool, compact: bool) -> str:
    """the untrusted-context header is removed WITH a receipt when asked.

    The header-removal branches ignored ``mark=True``, so one removal shape in this function
    left no trace while every other one did (round-2 verify-4 #15).
    """
    def _replace(match: "re.Match[str]") -> str:
        return _injection_marker(match.group(0), mark, compact)

    return _UNTRUSTED_CONTEXT_HEADER_RE.sub(_replace, text)


def strip_injected_context_blocks(text: str, *, mark: bool = False, compact: bool = False) -> str:
    """Remove transient memory/context blocks before compaction summarization.

    ``mark=True`` leaves a marker naming how much was removed, for the paths
    whose output the summariser reads (audit p05 EX01).
    """
    if not text:
        return ""

    cleaned = text
    changed = False
    if "<" not in text:
        cleaned = _mark_header_removal(text, mark, compact)
        changed = cleaned != text
        return cleaned.strip() if changed else cleaned

    for tag in _INJECTED_CONTEXT_TAGS:
        escaped = re.escape(tag)
        self_close_re = re.compile(rf"<{escaped}(?:\s[^>]*)?\s*/\s*>", re.IGNORECASE)
        open_re = re.compile(rf"<{escaped}(?:\s[^>]*)?>", re.IGNORECASE)
        close_re = re.compile(rf"</{escaped}\s*>", re.IGNORECASE)
        before_self_close = cleaned
        # a self-closing tag carries its content in its ATTRIBUTES
        # (<active_memory decision="CANCEL"/>), so removing it silently dropped that text
        # (verify-4 #7). Marked like every other removal.
        def _mark_self_close(match: "re.Match[str]") -> str:
            return _injection_marker(match.group(0), mark, compact)

        cleaned = self_close_re.sub(_mark_self_close, cleaned)
        changed = changed or cleaned != before_self_close

        while True:
            opener = open_re.search(cleaned)
            if not opener:
                break

            # the removal boundary is upstream's (safety first: a spoofed
            # closer inside recalled text must not be able to smuggle content past it), but
            # the cut is MARKED. Upstream deleted whatever lay between two block-shaped tags,
            # so a real decision written between two memory blocks disappeared from the
            # summariser's input with nothing to say it had ever been there (audit p05 EX01).
            closer = _select_injected_context_closer(cleaned, opener, close_re)
            if closer is None:
                if _at_line_end(cleaned, opener.end()):
                    removed = cleaned[opener.start():]
                    cleaned = cleaned[: opener.start()] + _injection_marker(removed, mark, compact)
                else:
                    # an unmatched INLINE opening tag carries its content in
                    # its attributes just as a self-closing one does; dropping it unmarked lost
                    # that text silently (round-2 verify-4 #15).
                    removed = cleaned[opener.start():opener.end()]
                    cleaned = (cleaned[: opener.start()]
                               + _injection_marker(removed, mark, compact)
                               + cleaned[opener.end():])
                changed = True
                continue
            removed = cleaned[opener.start():closer.end()]
            cleaned = (cleaned[: opener.start()] + _injection_marker(removed, mark, compact)
                       + cleaned[closer.end() :])
            changed = True

    before_header = cleaned
    cleaned = _mark_header_removal(cleaned, mark, compact)
    changed = changed or cleaned != before_header
    return cleaned.strip() if changed else cleaned


def sanitize_pre_compaction_content(text: Any) -> str:
    """Render a message's content as text for the summariser, changing nothing in it.

    This used to remove inline media payloads and host-injected context blocks on the way to
    the summariser, leaving markers behind. Both were LCM's own cuts to the source (#56): the
    media marker claimed a whole medium for a payload it had only trimmed one line from, and
    the injected block was a span the host had actually put in the conversation. The source is
    the message now; only STRUCTURE is projected, because the summariser reads text and a
    content list is not text. That projection's own losses (a media block becoming a marker)
    are #31 MC01 and are not made worse here.

    Nothing here removes a part or spills a file. A payload the configured summary route
    cannot process fails at that route — `escalation.summarize_with_escalation` raises
    `SummaryUnavailableError`, the leaf rescue retries with smaller chunks, and the raw
    messages stay in context — which is the fail-before-loss outcome the contract asks for.
    Steering by recalled text is handled where it costs nothing: the summariser's source is
    carried inside `prompt_boundary.build_untrusted_data_messages`, whose system role says the
    sources are untrusted evidence and never an instruction channel.
    """
    if isinstance(text, str):
        return text
    return _sanitize_content_block(text)


def sanitize_pre_compaction_tool_arguments(arguments: Any) -> str:
    """Render tool-call arguments for the summariser and for raw expansion, unchanged.

    A valid JSON argument string was parsed and re-serialised, so
    `0.123456789012345678901234567890` reached both the summary source and `lcm_expand`'s
    raw-row branch as `0.12345678901234568` — a value the provider never sent, in the one
    place a reader goes to get the exact bytes (#56). The string a provider sent IS the
    argument; it is returned as it stands, which also makes the duplicate-key and key-collision
    hazards the round trip created impossible rather than detected.
    """
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, (dict, list)):
        # already a structure rather than the provider's own text: JSON is the
        # faithful rendering of it, and there is no original string to preserve instead.
        try:
            return json.dumps(arguments, ensure_ascii=False)
        except Exception:
            return json.dumps(arguments, ensure_ascii=False, default=str)
    return sanitize_pre_compaction_content(arguments)


def extract_before_compaction(
    serialized_messages: str,
    output_path: str,
    session_id: str = "",
    model: str = "",
    timeout: float | None = None,
    source_store_ids: "List[int] | None" = None,  # audit p05 EX07
) -> bool:
    """Extract decisions from messages about to be compacted and write to a daily file.

    Returns True if extraction succeeded, False otherwise.
    Never raises — failures are logged and swallowed.
    """
    try:
        # the same untrusted-data boundary the summariser uses. Upstream
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

        # "the model said there is nothing to extract" and "the route
        # returned nothing at all" are different outcomes; both reported success, so a dead
        # extraction route read as a segment that held no decisions (round-2 verify-4 #38).
        if result is None or not str(result).strip():
            logger.warning(
                "Pre-compaction extraction returned an empty response; treating it as a "
                "failed extraction, not as an absence of content"
            )
            return False
        if result.strip() == "NOTHING_TO_EXTRACT":
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
        # say exactly which rows these bullets came from. A note headed only
        # by a wall-clock time cannot be tied back to its segment, and several passes in one
        # session produce several such notes (audit p05 EX07).
        digest = hashlib.sha256(serialized_messages.encode("utf-8", "replace")).hexdigest()[:16]
        provenance = f"source chars={len(serialized_messages)}, sha256:{digest}"
        if source_store_ids:
            ids = sorted({int(store_id) for store_id in source_store_ids})
            # the COMPLETE manifest, compressed into ranges. "+N more" and a
            # bare min..max span are not resolvable when the ids are sparse (round-3 verify-3):
            # a reader cannot tell which rows the note came from. Contiguous runs collapse, so
            # even a long segment stays one line.
            runs: list[str] = []
            start = previous = ids[0]
            for store_id in ids[1:]:
                if store_id == previous + 1:
                    previous = store_id
                    continue
                runs.append(str(start) if start == previous else f"{start}-{previous}")
                start = previous = store_id
            runs.append(str(start) if start == previous else f"{start}-{previous}")
            provenance += (
                f", store_ids={','.join(runs)} ({len(ids)} row(s))"
                " — lcm_expand(store_id=…)"
            )
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
