"""LCM-bypass compaction and host-fallback-compressor handling.

Extracted verbatim from :mod:`hermes_lcm.engine` as ``BypassMixin`` (WS5 seam).
The methods manage sessions that opt out of LCM context management: detecting
the bypass, mirroring the host's native fallback compressor, and applying the
deterministic tail-compaction fallback. State stays on the engine (accessed via
``self``); mixing this in leaves every call site and ``self._*`` reference
unchanged.
"""

import importlib
import json
import inspect  # fork: betterlcm
import logging
from typing import Any, Dict, List, Optional

from .message_analysis import _assistant_tool_call_ids
from .message_content import normalize_content_value
from .session_patterns import build_session_match_keys, matches_session_pattern
from .tokens import count_messages_tokens
from . import marked_loss  # fork: betterlcm

logger = logging.getLogger(__name__)


class BypassMixin:
    def _bypasses_lcm_context_management(self) -> bool:
        """Return True when this binding must not write/manage LCM state.

        Ignored, stateless, and in-process auxiliary sessions are excluded from
        LCM storage. They still need context-size protection because Hermes has
        exactly one active context engine; returning a pure no-op here would
        disable every compaction layer for the session.
        """
        return bool(
            self._session_ignored
            or self._session_stateless
            or self._thread_context_stateless()
        )

    def _bypass_lcm_reason(self) -> str:
        if self._thread_context_stateless():
            return "auxiliary thread context"
        if self._session_ignored:
            return "ignored session"
        if self._session_stateless:
            return "stateless session"
        return "active session"

    def _bypass_lcm_session_id(self) -> str:
        return self._thread_context_session_id() or self._session_id or "(unknown)"

    def _session_id_matches_lcm_bypass_filters(
        self,
        session_id: str,
        *,
        platform: str = "",
    ) -> bool:
        if not session_id:
            return False
        match_keys = build_session_match_keys(session_id, platform=platform)
        if matches_session_pattern(match_keys, self._compiled_ignore_session_patterns):
            return True
        return matches_session_pattern(match_keys, self._compiled_stateless_session_patterns)

    def _ended_session_directly_bypasses_lcm(self, session_id: str) -> bool:
        """Classify a session-end callback by the ended id, not the active binding."""
        if not session_id:
            return False
        if session_id == self._thread_context_session_id():
            return True
        return self._session_id_matches_lcm_bypass_filters(session_id)

    def _end_host_fallback_compressor_for_session(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        current_session_bypasses: bool,
    ) -> None:
        if self._host_fallback_compressor is not None and (
            current_session_bypasses
            or not self._host_fallback_session_id
            or self._host_fallback_session_id == session_id
        ):
            compressor = self._host_fallback_compressor
            fallback_session_id = self._host_fallback_session_id or session_id
            on_session_end = getattr(compressor, "on_session_end", None)
            if callable(on_session_end) and fallback_session_id:
                try:
                    on_session_end(fallback_session_id, messages)
                except Exception:
                    logger.debug("LCM host fallback compressor session-end reset failed", exc_info=True)
            on_session_reset = getattr(compressor, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""

    def _get_host_fallback_compressor(self) -> Any:
        """Return Hermes' native compressor for LCM-bypassed sessions if available."""
        session_id = self._bypass_lcm_session_id()
        if self._host_fallback_compressor is not None:
            if session_id == self._host_fallback_session_id:
                return self._host_fallback_compressor
            previous = self._host_fallback_compressor
            on_session_end = getattr(previous, "on_session_end", None)
            if callable(on_session_end) and self._host_fallback_session_id:
                try:
                    on_session_end(self._host_fallback_session_id, [])
                except Exception:
                    logger.debug("LCM host fallback compressor session-end reset failed", exc_info=True)
            on_session_reset = getattr(previous, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
        try:
            ContextCompressor = getattr(
                importlib.import_module("agent.context_compressor"),
                "ContextCompressor",
            )
        except Exception as exc:  # pragma: no cover - only hit on non-Hermes hosts
            if not self._host_fallback_import_warning_logged:
                logger.warning(
                    "LCM could not load Hermes native ContextCompressor for bypassed session fallback: %s",
                    exc,
                )
                self._host_fallback_import_warning_logged = True
            return None

        kwargs = {
            "model": self.model or "unknown",
            "threshold_percent": self.context_threshold or self.threshold_percent or 0.50,
            "protect_first_n": self.protect_first_n,
            "protect_last_n": self.protect_last_n,
            "quiet_mode": True,
            "summary_model_override": self._config.summary_model or None,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "config_context_length": self.context_length or self.raw_context_length or None,
            "provider": self.provider,
            "api_mode": self.api_mode,
        }
        try:
            compressor = ContextCompressor(**kwargs)
        except TypeError:
            # Older Hermes hosts may not expose all constructor kwargs. Keep the
            # fallback deliberately conservative rather than failing open to an
            # unbounded ignored/stateless transcript.
            #
            # fork: betterlcm — drop only the kwargs this host cannot take. Upstream fell back
            # to a fixed five-argument call, which silently discarded the operator's summary
            # model, provider, credentials and context length: one unsupported keyword sent
            # bypassed summarisation to a different route than the one that was configured
            # (audit p05 BY03).
            supported = self._constructor_supported_kwargs(ContextCompressor, kwargs)
            dropped = sorted(set(kwargs) - set(supported))
            if dropped:
                logger.warning(
                    "LCM native ContextCompressor does not accept %s on this host; "
                    "initializing the bypassed-session fallback without them",
                    ", ".join(dropped),
                )
            try:
                compressor = ContextCompressor(**supported)
            except Exception as exc:
                logger.warning(
                    "LCM could not initialize Hermes native ContextCompressor for bypassed session fallback; using deterministic trim: %s",
                    exc,
                )
                return None
        except Exception as exc:
            logger.warning(
                "LCM could not initialize Hermes native ContextCompressor for bypassed session fallback; using deterministic trim: %s",
                exc,
            )
            return None
        self._host_fallback_compressor = compressor
        self._host_fallback_session_id = session_id
        self._sync_host_fallback_compressor(compressor)
        return compressor

    @staticmethod
    def _constructor_supported_kwargs(factory: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """fork: betterlcm — the subset of ``kwargs`` this constructor actually accepts.

        Falls back to the minimum every known host supports when the signature cannot be
        read, so an unreadable signature degrades the same way the old fixed call did.
        """
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins / C constructors
            return {
                key: value
                for key, value in kwargs.items()
                if key in {"model", "threshold_percent", "protect_first_n", "protect_last_n", "quiet_mode"}
            }
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return dict(kwargs)
        return {key: value for key, value in kwargs.items() if key in parameters}

    def _sync_host_fallback_compressor(self, compressor: Any) -> None:
        """Keep the delegated native compressor aligned with LCM runtime metadata."""
        update_model = getattr(compressor, "update_model", None)
        context_length = self.context_length or self.raw_context_length
        if callable(update_model) and context_length > 0:
            try:
                update_model(
                    model=self.model or "unknown",
                    context_length=context_length,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    provider=self.provider,
                    api_mode=self.api_mode,
                )
            except TypeError:
                try:
                    update_model(self.model or "unknown", context_length, self.base_url, self.api_key)
                except TypeError:
                    pass
                except Exception:
                    logger.debug("LCM host fallback compressor model sync failed", exc_info=True)
            except Exception:
                logger.debug("LCM host fallback compressor model sync failed", exc_info=True)
        for attr, value in (
            ("threshold_percent", self.context_threshold or self.threshold_percent),
            ("protect_first_n", self.protect_first_n),
            ("protect_last_n", self.protect_last_n),
        ):
            try:
                setattr(compressor, attr, value)
            except Exception:
                pass
        on_session_start = getattr(compressor, "on_session_start", None)
        session_id = self._bypass_lcm_session_id()
        if callable(on_session_start) and session_id:
            try:
                on_session_start(
                    session_id,
                    platform=self._session_platform,
                    model=self.model,
                    provider=self.provider,
                    context_length=context_length,
                )
            except Exception:
                logger.debug("LCM host fallback compressor session bind failed", exc_info=True)

    def _mirror_host_fallback_state(self, compressor: Any) -> None:
        for attr in (
            "_last_compress_aborted",
            "_last_summary_error",
            "_last_summary_auth_failure",
            "_last_summary_network_failure",
            "_last_summary_dropped_count",
            "_last_summary_fallback_used",
            "_last_aux_model_failure_error",
            "_last_aux_model_failure_model",
        ):
            if hasattr(compressor, attr):
                setattr(self, attr, getattr(compressor, attr))

    def _bypass_compaction_target_tokens(
        self,
        *,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        caps: list[int] = []
        assembly_cap = self._overflow_recovery_assembly_cap(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if assembly_cap is not None:
            caps.append(assembly_cap)
        if self.threshold_tokens > 0:
            caps.append(self.threshold_tokens)
        elif self.context_length > 0:
            caps.append(self.context_length)
        return min(caps) if caps else None

    @staticmethod
    def _truncate_bypass_content_value(content: Any, char_budget: int, *, suffix: str = "") -> Any:
        """fork: betterlcm — the cut marker is kept even at a zero budget (audit p05 BY01).

        Upstream appended the suffix only while ``char_budget > 0``, so the one case where the
        whole text disappeared was also the one case that left no trace of it.
        """
        if char_budget < 0:
            char_budget = 0
        if isinstance(content, str):
            if len(content) <= char_budget:
                return content
            return content[:char_budget] + suffix
        if isinstance(content, list):
            truncated_parts: list[Any] = []
            changed = False
            for part in content:
                if isinstance(part, str):
                    next_part = (
                        part if len(part) <= char_budget else part[:char_budget] + suffix
                    )
                    changed = changed or next_part != part
                    truncated_parts.append(next_part)
                    continue
                if isinstance(part, dict):
                    next_part = dict(part)
                    for key in ("text", "content"):
                        value = next_part.get(key)
                        if isinstance(value, str) and len(value) > char_budget:
                            next_part[key] = value[:char_budget] + suffix
                            changed = True
                        elif isinstance(value, dict):
                            nested = dict(value)
                            for nested_key in ("value", "content"):
                                nested_value = nested.get(nested_key)
                                if isinstance(nested_value, str) and len(nested_value) > char_budget:
                                    nested[nested_key] = nested_value[:char_budget] + suffix
                                    changed = True
                            next_part[key] = nested
                    truncated_parts.append(next_part)
                    continue
                truncated_parts.append(part)
            if changed:
                return truncated_parts
        normalized = normalize_content_value(content)
        if isinstance(normalized, str) and len(normalized) > char_budget:
            return normalized[:char_budget] + suffix
        return content

    def _trim_bypass_compacted_to_cap(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: Optional[int],
    ) -> List[Dict[str, Any]]:
        compacted = self._sanitize_active_context_messages(messages)
        if target_tokens is None or target_tokens <= 0:
            return compacted

        while len(compacted) > 2 and count_messages_tokens(compacted) > target_tokens:
            remove_indices: list[int] = []
            for idx, msg in enumerate(compacted):
                if idx == 0:
                    continue
                # fork: betterlcm — never remove the receipt that says messages were removed
                if marked_loss.is_bypass_omission_marker(msg):
                    continue
                if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                    continue
                call_ids = _assistant_tool_call_ids([msg])
                remove_indices = [idx]
                remove_indices.extend(
                    follow_idx
                    for follow_idx in range(idx + 1, len(compacted))
                    if compacted[follow_idx].get("role") == "tool"
                    and str(compacted[follow_idx].get("tool_call_id") or "") in call_ids
                )
                break
            if not remove_indices:
                remove_index = 1
                if (
                    compacted[1].get("role") == "tool"
                    and compacted[0].get("role") == "assistant"
                    and compacted[0].get("tool_calls")
                ):
                    remove_index = 0
                if marked_loss.is_bypass_omission_marker(compacted[remove_index]):
                    # fork: the receipt is not a removal candidate; take the next message
                    following = next(
                        (
                            index
                            for index in range(remove_index + 1, len(compacted))
                            if not marked_loss.is_bypass_omission_marker(compacted[index])
                        ),
                        None,
                    )
                    if following is None:
                        break
                    remove_index = following
                if remove_index >= len(compacted) - 1:
                    # fork: betterlcm — protecting the receipt must never cost the NEWEST
                    # message: skipping the receipt at index 1 made the request the agent has
                    # to answer the next removal candidate (verify-2 regression #2). Stop
                    # deleting whole messages here and let the character-trim stages below
                    # shrink text instead — they mark every cut they make.
                    break
                remove_indices = [remove_index]
            before_shape = [
                (msg.get("role"), msg.get("tool_call_id"), bool(msg.get("tool_calls")))
                for msg in compacted
            ]
            before_tokens = count_messages_tokens(compacted)
            for remove_index in sorted(set(remove_indices), reverse=True):
                if 0 <= remove_index < len(compacted):
                    del compacted[remove_index]
            compacted = self._sanitize_active_context_messages(compacted)
            after_shape = [
                (msg.get("role"), msg.get("tool_call_id"), bool(msg.get("tool_calls")))
                for msg in compacted
            ]
            if after_shape == before_shape and count_messages_tokens(compacted) >= before_tokens:
                break

        if count_messages_tokens(compacted) <= target_tokens:
            return compacted

        char_budget = max(0, min(500, target_tokens * 4 // max(1, len(compacted))))
        truncated = compacted
        previous_budget = -1
        for _ in range(12):
            next_messages: list[Dict[str, Any]] = []
            newest_index = len(compacted) - 1
            for index, msg in enumerate(compacted):
                if marked_loss.is_bypass_omission_marker(msg):
                    next_messages.append(msg)  # fork: the receipt is never shortened
                    continue
                if index == newest_index:
                    # fork: betterlcm — the newest message is the request the agent has to
                    # answer; the ordered last-resort stage below shrinks it only after
                    # everything else, including the receipt, has already given way.
                    next_messages.append(msg)
                    continue
                next_msg = dict(msg)
                content = next_msg.get("content")
                # fork: betterlcm — a cut carries a marker (marked_loss.BYPASS_TRIM_SUFFIX)
                next_msg["content"] = self._truncate_bypass_content_value(
                    content, char_budget, suffix=marked_loss.BYPASS_TRIM_SUFFIX
                )
                next_messages.append(next_msg)
            truncated = self._sanitize_active_context_messages(next_messages)
            token_count = count_messages_tokens(truncated)
            if token_count <= target_tokens:
                return truncated
            if char_budget == 0 or char_budget == previous_budget:
                break
            previous_budget = char_budget
            ratio = target_tokens / max(1, token_count)
            char_budget = max(0, min(char_budget - 1, int(char_budget * max(0.25, ratio * 0.8))))

        # fork: betterlcm — drop from the front, but never the receipt and never the newest
        # message. Upstream dropped whatever was first; keeping the receipt at the front then
        # made the live request the thing that went (verify-2 regression #2). When only the
        # receipt and the newest message are left, the character-trim stage below shrinks them
        # instead — and marks every cut.
        compacted = truncated
        while len(compacted) > 2 and count_messages_tokens(compacted) > target_tokens:
            droppable = next(
                (
                    index
                    for index in range(0, len(compacted) - 1)
                    if not marked_loss.is_bypass_omission_marker(compacted[index])
                ),
                None,
            )
            if droppable is None:
                break
            remainder = compacted[:droppable] + compacted[droppable + 1:]
            sanitized = self._sanitize_active_context_messages(remainder)
            if len(sanitized) >= len(compacted):
                break
            compacted = sanitized

        # fork: betterlcm — an ORDER of last resorts, because trimming every message together
        # cut the live request to a bare marker while older context was still present
        # (verify-2 regression #2):
        #   1. shrink the older messages,
        #   2. shrink the receipt to its shortest honest form (the counts survive),
        #   3. only then shrink the newest message.
        def _shrink(messages, budget, *, protect_newest, suffix):
            newest_index = len(messages) - 1
            shrunk: list[Dict[str, Any]] = []
            for index, msg in enumerate(messages):
                if marked_loss.is_bypass_omission_marker(msg):
                    shrunk.append(msg)
                    continue
                if protect_newest and index == newest_index:
                    shrunk.append(msg)
                    continue
                next_msg = dict(msg)
                # the suffix marker is kept even at a zero char budget. Upstream dropped it
                # exactly there, so the most destructive trim of all was the one that said
                # nothing about itself (audit p05 BY01).
                next_msg["content"] = self._truncate_bypass_content_value(
                    next_msg.get("content"), budget, suffix=suffix
                )
                shrunk.append(next_msg)
            return self._sanitize_active_context_messages(shrunk)

        for protect_newest in (True, False):
            char_budget = max(0, min(80, target_tokens * 4))
            previous_budget = -1
            while (
                count_messages_tokens(compacted) > target_tokens
                and char_budget != previous_budget
            ):
                previous_budget = char_budget
                compacted = _shrink(
                    compacted, char_budget,
                    protect_newest=protect_newest,
                    suffix=marked_loss.BYPASS_FINAL_TRIM_SUFFIX,
                )
                char_budget = max(0, char_budget // 2)
            if count_messages_tokens(compacted) <= target_tokens:
                break
            if protect_newest:
                # step 2: the receipt becomes its shortest honest form before the live request
                # loses anything at all.
                compacted = [
                    {**msg, "content": marked_loss.compact_bypass_omission_marker(msg.get("content"))}
                    if marked_loss.is_bypass_omission_marker(msg) else msg
                    for msg in compacted
                ]
                if count_messages_tokens(compacted) <= target_tokens:
                    break

        return compacted

    def _fallback_tail_compaction(
        self,
        messages: List[Dict[str, Any]],
        *,
        target_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Last-resort size guard when Hermes' native compressor is unavailable."""
        if len(messages) <= 2:
            return self._trim_bypass_compacted_to_cap(messages, target_tokens)
        head_count = max(1, min(self.protect_first_n, len(messages)))
        tail_count = max(1, min(self.protect_last_n, len(messages) - head_count))
        # fork: betterlcm — the receipt says HOW MUCH went. Upstream's marker named neither the
        # number of messages nor their size, so a bypassed session could lose most of its
        # history behind a sentence that read like boilerplate (audit p05 BY01).
        dropped = messages[head_count:len(messages) - tail_count]
        dropped_chars = sum(self._bypass_envelope_chars(message) for message in dropped)
        marker = {
            "role": "user",
            "content": marked_loss.bypass_omission_marker(len(dropped), dropped_chars),
        }
        compacted = list(messages[:head_count]) + [marker] + list(messages[-tail_count:])
        trimmed = self._trim_bypass_compacted_to_cap(compacted, target_tokens)
        # fork: betterlcm — the cap loop above may remove more messages after the receipt was
        # written, so the counts are recomputed against what actually SURVIVED. Upstream's
        # receipt (and the fork's first version of it) claimed "8 messages dropped" while nine
        # had gone (verify-4 #17).
        return self._refresh_bypass_receipt(messages, trimmed)

    @staticmethod
    def _bypass_envelope_chars(message: Dict[str, Any]) -> int:
        """fork: betterlcm — the size of everything a dropped message carried.

        A dropped assistant turn's tool CALLS are part of what was removed; counting content
        alone reported "~2 chars" for a message holding 10,000 characters of arguments
        (round-2 verify-4 #29).
        """
        total = len(str(normalize_content_value(message.get("content")) or ""))
        calls = message.get("tool_calls")
        if calls:
            try:
                total += len(json.dumps(calls, ensure_ascii=False, default=str))
            except Exception:  # pragma: no cover - defensive
                total += len(str(calls))
        return total

    def _refresh_bypass_receipt(
        self,
        original: List[Dict[str, Any]],
        trimmed: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """fork: betterlcm — restate the receipt's counts from the FINAL result.

        Counted in aggregate rather than by object identity: the surviving messages are
        trimmed COPIES, and the characters the trim removed from them are gone too.
        """
        def _chars(messages: List[Dict[str, Any]]) -> int:
            return sum(
                self._bypass_envelope_chars(message)
                for message in messages
                if not marked_loss.is_bypass_omission_marker(message)
            )

        surviving = [
            message for message in trimmed
            if not marked_loss.is_bypass_omission_marker(message)
        ]
        original_messages = [
            message for message in original
            if not marked_loss.is_bypass_omission_marker(message)
        ]
        dropped = original_messages[:max(0, len(original_messages) - len(surviving))]
        dropped_chars = max(0, _chars(original_messages) - _chars(surviving))
        # fork: betterlcm — the receipt is CUMULATIVE. Every surviving receipt was rewritten
        # with counts from the latest reduction alone, so a second bypass compaction erased
        # the record of the first: two receipts both claimed 6 messages / 3,046 characters
        # where 10 / 5,100 had already gone (round-5 verify-6 #10). Earlier receipts state
        # what they removed; those numbers are carried forward and this call's own reduction
        # is added exactly once.
        prior_messages = 0
        prior_chars = 0
        for message in original:
            if not marked_loss.is_bypass_omission_marker(message):
                continue
            counted_messages, counted_chars = marked_loss.bypass_omission_counts(
                str(message.get("content") or "")
            )
            prior_messages += counted_messages
            prior_chars += counted_chars
        total_messages = prior_messages + len(dropped)
        total_chars = prior_chars + dropped_chars
        refreshed: List[Dict[str, Any]] = []
        receipt_written = False
        for message in trimmed:
            if not marked_loss.is_bypass_omission_marker(message):
                refreshed.append(message)
                continue
            if receipt_written:
                # one cumulative receipt, not several copies each restating the same total
                continue
            was_compact = "msg /" in str(message.get("content") or "")
            content = marked_loss.bypass_omission_marker(total_messages, total_chars)
            if was_compact:
                content = marked_loss.compact_bypass_omission_marker(content)
            refreshed.append({**message, "content": content})
            receipt_written = True
        return refreshed

    def _compress_lcm_bypassed_session(
        self,
        messages: List[Dict[str, Any]],
        *,
        current_tokens: int | None = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Delegate ignored/stateless context bounding without writing to LCM."""
        reason = self._bypass_lcm_reason()
        session_id = self._bypass_lcm_session_id()
        self._remember_lcm_bypass_message_prefix(session_id, messages)
        observed_tokens = current_tokens if current_tokens and current_tokens > 0 else count_messages_tokens(messages)
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if not force and not force_overflow and self.threshold_tokens > 0 and observed_tokens < self.threshold_tokens:
            logger.debug(
                "LCM compaction bypass no-op for %s %s below threshold (%s < %s)",
                reason,
                session_id,
                observed_tokens,
                self.threshold_tokens,
            )
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = f"LCM bypassed below threshold: {reason}"
            return self._redact_active_replay_messages(messages)

        logger.debug("LCM delegating compaction for bypassed %s %s", reason, session_id)
        self._last_compression_status = "host_fallback"
        self._last_compression_noop_reason = f"LCM bypassed: {reason}"
        safe_messages = self._redact_active_replay_messages(messages)
        target_tokens = self._bypass_compaction_target_tokens(
            observed_tokens=observed_tokens,
            messages=safe_messages,
        )

        compressor = self._get_host_fallback_compressor()
        if compressor is None:
            compacted = self._fallback_tail_compaction(safe_messages, target_tokens=target_tokens)
            if compacted != messages:
                self.compression_count += 1
            return compacted

        self._sync_host_fallback_compressor(compressor)
        before_count = int(getattr(compressor, "compression_count", 0) or 0)
        try:
            try:
                compacted = compressor.compress(
                    safe_messages,
                    current_tokens=current_tokens,
                    focus_topic=focus_topic,
                    force=force,
                )
            except TypeError:
                compacted = compressor.compress(
                    safe_messages,
                    current_tokens=current_tokens,
                    focus_topic=focus_topic,
                )
        except Exception as exc:
            self._mirror_host_fallback_state(compressor)
            logger.warning(
                "LCM Hermes native ContextCompressor failed for bypassed %s %s; using deterministic trim: %s",
                reason,
                session_id,
                exc,
            )
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
            compacted = self._fallback_tail_compaction(safe_messages, target_tokens=target_tokens)
            if compacted != safe_messages:
                self.compression_count += 1
                self._last_compress_aborted = False
            return compacted
        self._mirror_host_fallback_state(compressor)
        # fork: betterlcm — an abort is a decision to PRESERVE, not a failed attempt. Upstream
        # counted every native return as at least one compression and, when the unchanged
        # result was still over target, ran the deterministic trim over it and cleared the
        # abort flag — so the host's explicit "do not compress this" became a destructive
        # delete reported as success (audit p05 BY02).
        native_aborted = bool(getattr(compressor, "_last_compress_aborted", False))
        native_changed = compacted is not safe_messages and compacted != safe_messages
        after_count = int(getattr(compressor, "compression_count", before_count) or before_count)
        if native_changed:
            self.compression_count += max(1, after_count - before_count)
        elif after_count > before_count:
            # the host counted an attempt that changed nothing; mirror its count, no more
            self.compression_count += after_count - before_count
        compacted = self._sanitize_active_context_messages(compacted)
        if target_tokens is not None and count_messages_tokens(compacted) > target_tokens:
            if native_aborted and not native_changed:
                # fork: betterlcm — say plainly that the host's preservation decision is being
                # overridden. The assembly cap is a hard provider bound for a session LCM does
                # not store, so the deterministic trim still has to run, but upstream counted
                # the untouched native return as a compression first and reported the whole
                # sequence as ordinary success (audit p05 BY02).
                logger.warning(
                    "LCM native compressor aborted for bypassed %s %s but its context is still "
                    "over the assembly cap; falling back to the deterministic trim",
                    reason,
                    session_id,
                )
            compacted = self._fallback_tail_compaction(safe_messages, target_tokens=target_tokens)
            if compacted != safe_messages:
                self._last_compress_aborted = False
        return compacted
