"""Leaf-compaction pipeline for the LCM engine (WS5 Seam 6).

The ``CompactionMixin`` holds the compaction gate + pipeline: ``should_compress``
/ ``should_compress_preflight`` (public), the leaf-candidate and chunk-selection
helpers, and the main ``compress`` entry point. These methods were lifted
verbatim out of ``LCMEngine`` and continue to run bound to the engine instance
(``self`` is the ``LCMEngine``), so they read and write the engine's runtime
state (``_ingest_cursor``, ``_store``, ``_dag``, ``_lifecycle``, status/telemetry
fields, per-turn caches) and call back into engine helpers (ingest,
reconciliation, placeholder-ledger, the summarize-with-rescue step, assembly,
lifecycle) through normal attribute lookup. ``LCMEngine`` mixes this in ahead of
``ContextEngine`` so the mixin's ``compress`` / ``should_compress`` /
``should_compress_preflight`` override the ContextEngine protocol defaults.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .dag import SummaryNode
from .message_content import text_content_for_pattern_matching
from .sanitize import _contains_sensitive_redaction
from .tokens import count_message_tokens, count_messages_tokens, count_tokens
from .errors import SummaryUnavailableError  # fork: betterlcm
from .message_analysis import _tool_call_id  # fork: betterlcm
from . import marked_loss  # fork: betterlcm

logger = logging.getLogger(__name__)

# fork: betterlcm — kept as the documented upstream defaults; the loop reads
# ``config.sweep_max_passes`` and the curved ``effective_leaf_loop_max_seconds``.
_THRESHOLD_FULL_SWEEP_MAX_PASSES = 12
_THRESHOLD_FULL_SWEEP_MAX_SECONDS = 120.0


class CompactionMixin:
    def _maybe_reclassify_late_auxiliary_before_compaction_write(self) -> None:
        maybe_reclassify = getattr(
            self,
            "_maybe_reclassify_current_session_as_auxiliary_before_message_ingest",
            None,
        )
        if callable(maybe_reclassify):
            maybe_reclassify()

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._bypasses_lcm_context_management():
            if self._compression_boundary_cooldown_active():
                return False
            if prompt_tokens is not None:
                tokens = prompt_tokens
            else:
                auxiliary_session_id = self._thread_context_session_id()
                if auxiliary_session_id:
                    tokens = self._current_auxiliary_prompt_tokens(auxiliary_session_id)
                else:
                    tokens = self.last_prompt_tokens
            if self._should_force_overflow_recovery(observed_tokens=tokens):
                return True
            if self.threshold_tokens <= 0:
                return False
            return tokens >= self.threshold_tokens
        if self._compression_boundary_cooldown_active():
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self._should_force_overflow_recovery(observed_tokens=tokens):
            return True
        if self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages):
        """Pre-flight check — also ingests messages into the store."""
        self._preflight_cleanup_only_due_to_boundary_cooldown = False
        self._preflight_cleanup_only_below_threshold = False  # fork: betterlcm
        self._maybe_reclassify_late_auxiliary_before_compaction_write()
        if self._bypasses_lcm_context_management():
            self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            rough = count_messages_tokens(messages)
            if self._compression_boundary_cooldown_active():
                return False
            if self._should_force_overflow_recovery(observed_tokens=rough, messages=messages):
                return True
            return self.threshold_tokens > 0 and rough >= self.threshold_tokens
        rough = count_messages_tokens(messages)
        pre_ingest_placeholder_ambiguous_noop = False
        pre_ingest_noop_reason = ""
        if (
            self.threshold_tokens > 0
            and rough >= self.threshold_tokens
            and not self._compiled_ignore_message_patterns
            and any(
                self._is_ignored_active_replay_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                for msg in messages
            )
        ):
            eligible, reason = self._leaf_compaction_candidate_status(
                messages,
                allow_partial_leaf=self._config.threshold_full_sweep_enabled,
            )
            pre_ingest_placeholder_ambiguous_noop = not eligible
            pre_ingest_noop_reason = reason
        replay_messages = None
        if self._session_id and messages:
            try:
                replay_messages = self._ingest_messages(messages)
                self._record_ingest_success()
            except Exception as e:
                # Fail closed for NORMAL threshold compaction: the store did not
                # accept this turn, so do not compact against a store missing the
                # latest messages - that could rebuild active context without
                # them. But still honor emergency overflow recovery, whose whole
                # job is to keep the prompt under the provider limit; it converges
                # via deterministic L3 truncation without needing the store write.
                self._record_ingest_failure("preflight", e)
                if self._should_force_overflow_recovery(observed_tokens=rough):
                    return True
                return False
        if replay_messages is not None and replay_messages != messages:
            replay_rough = count_messages_tokens(replay_messages)
            cleanup_requested = self._replay_diff_requests_ingest_cleanup(
                messages,
                replay_messages,
            )
            force_overflow_requested = self._should_force_overflow_recovery(
                observed_tokens=rough,
                messages=messages,
            ) or self._should_force_overflow_recovery(
                observed_tokens=replay_rough,
                messages=replay_messages,
            )
            if cleanup_requested:
                if (
                    not force_overflow_requested
                    and self._compression_boundary_cooldown_active()
                ):
                    self._preflight_cleanup_only_due_to_boundary_cooldown = True
                # fork: betterlcm — a replay-diff cleanup under the threshold asks
                # compress() for the cleanup preamble only (no leaf pass); see
                # _compress_is_cleanup_only.
                if (
                    not force_overflow_requested
                    and self.threshold_tokens > 0
                    and replay_rough < self.threshold_tokens
                ):
                    self._preflight_cleanup_only_below_threshold = True
                return self._mark_preflight_compression_requested()
            if force_overflow_requested:
                return self._mark_preflight_compression_requested()
            # A boundary skip cools down summary-producing leaf/condensation
            # work. It must not prevent the host from adopting a replay cleanup
            # that ingest has already made durable (for example a live tool
            # result stub); those returns above are deterministic and add no
            # summarizer spend.
            if self._compression_boundary_cooldown_active():
                return False
            if pre_ingest_placeholder_ambiguous_noop:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = pre_ingest_noop_reason
                logger.info("LCM preflight compression no-op: %s", pre_ingest_noop_reason)
                return False
            eligible, reason = self._leaf_compaction_candidate_status(
                replay_messages,
                allow_partial_leaf=bool(
                    self._config.threshold_full_sweep_enabled
                    and self.threshold_tokens > 0
                    and replay_rough >= self.threshold_tokens
                ),
            )
            if eligible:
                return self._mark_preflight_compression_requested()
            if self._has_ignored_backlog_outside_fresh_tail(replay_messages):
                return self._mark_preflight_compression_requested()
            if self.threshold_tokens > 0 and replay_rough >= self.threshold_tokens:
                if self._should_run_deferred_maintenance(replay_messages, observed_tokens=replay_rough):
                    return self._mark_preflight_compression_requested()
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = reason
                logger.info("LCM preflight compression no-op: %s", reason)
                return False
            self._refresh_raw_backlog_debt(replay_messages, observed_tokens=replay_rough)
            if self._should_run_deferred_maintenance(replay_messages, observed_tokens=replay_rough):
                return self._mark_preflight_compression_requested()
            return False
        if self._compression_boundary_cooldown_active():
            return False
        if self._should_force_overflow_recovery(observed_tokens=rough):
            return self._mark_preflight_compression_requested()
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            if pre_ingest_placeholder_ambiguous_noop:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = pre_ingest_noop_reason
                logger.info("LCM preflight compression no-op: %s", pre_ingest_noop_reason)
                return False
            eligible, reason = self._leaf_compaction_candidate_status(
                messages,
                allow_partial_leaf=self._config.threshold_full_sweep_enabled,
            )
            if eligible:
                return self._mark_preflight_compression_requested()
            if self._has_ignored_backlog_outside_fresh_tail(messages):
                return self._mark_preflight_compression_requested()
            if self._should_run_deferred_maintenance(messages, observed_tokens=rough):
                return self._mark_preflight_compression_requested()
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = reason
            logger.info("LCM preflight compression no-op: %s", reason)
            return False
        self._refresh_raw_backlog_debt(messages, observed_tokens=rough)
        if self._should_run_deferred_maintenance(messages, observed_tokens=rough):
            return self._mark_preflight_compression_requested()
        return False

    def _compress_is_cleanup_only(
        self,
        *,
        force: bool,
        force_overflow: bool,
        estimated_active_tokens: int,
        deferred_maintenance_active: bool,
        critical_budget_pressure: bool,
        working_messages: List[Dict[str, Any]],
    ) -> bool:
        """fork: betterlcm — True when this compress() must not spend a leaf pass."""
        requested = bool(getattr(self, "_preflight_cleanup_only_below_threshold", False))
        self._preflight_cleanup_only_below_threshold = False
        if not requested:
            return False
        if force or force_overflow or deferred_maintenance_active or critical_budget_pressure:
            return False
        threshold = int(self.threshold_tokens or 0)
        if threshold <= 0 or int(estimated_active_tokens or 0) >= threshold:
            return False
        if self._has_ignored_backlog_outside_fresh_tail(working_messages):
            return False
        return True

    def _replay_diff_requests_ingest_cleanup(
        self,
        original_messages: List[Dict[str, Any]],
        replay_messages: List[Dict[str, Any]],
    ) -> bool:
        if len(original_messages) != len(replay_messages):
            return True
        for original_msg, replay_msg in zip(original_messages, replay_messages):
            original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
            replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
            if original_text != replay_text:
                if replay_text.startswith("[Externalized LCM ingest payload:"):
                    return True
                if replay_text.startswith("[Externalized payload: kind=raw_payload;"):
                    return True
                if replay_text.startswith("[Externalized tool output:"):
                    return True
                if replay_text.startswith("[LCM active replay placeholder: assistant output quarantined;"):
                    return True
                if replay_text.startswith("[LCM active replay placeholder: message ignored;"):
                    return True
                if "[LCM sensitive redaction:" in replay_text:
                    return True
            if original_msg.get("content") != replay_msg.get("content") and _contains_sensitive_redaction(
                replay_msg.get("content")
            ):
                return True
            if original_msg.get("tool_calls") != replay_msg.get("tool_calls") and _contains_sensitive_redaction(
                replay_msg.get("tool_calls")
            ):
                return True
        return False

    def _has_ignored_backlog_outside_fresh_tail(self, messages: List[Dict[str, Any]]) -> bool:
        if not self._compiled_ignore_message_patterns or not messages:
            return False
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False
        previous_store_id_map = self._current_compress_store_ids_by_message_id
        self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(
            messages[leading_anchor_count:fresh_tail_start]
        )
        try:
            return any(
                self._matches_ignore_message_patterns(msg)
                or self._mapped_stored_row_matches_ignore_message_patterns(msg)
                for msg in messages[leading_anchor_count:fresh_tail_start]
            )
        finally:
            self._current_compress_store_ids_by_message_id = previous_store_id_map

    def _leaf_compaction_candidate_status(
        self,
        messages: List[Dict[str, Any]],
        *,
        force_overflow: bool = False,
        allow_partial_leaf: bool = False,
    ) -> tuple[bool, str]:
        """Return whether a normal leaf compaction pass can actually run.

        The host asks ``should_compress_preflight`` before it emits user-visible
        compression status. A session can be over the global context threshold
        while all pressure sits in the protected fresh tail, or while the raw
        backlog outside that tail is still smaller than the configured leaf
        chunk. In that case ``compress()`` would immediately no-op, so preflight
        should not advertise a compaction attempt yet.
        """
        if not messages:
            return False, "empty message list"
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False, "no eligible raw backlog outside fresh tail"

        candidate_raw = messages[leading_anchor_count:fresh_tail_start]
        if not candidate_raw:
            return False, "no eligible raw backlog outside fresh tail"
        generated_placeholder_hashes = self._load_generated_ignored_placeholder_hashes()
        if self._compiled_ignore_message_patterns or generated_placeholder_hashes:
            previous_store_id_map = self._current_compress_store_ids_by_message_id
            self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(candidate_raw)
            try:
                filtered_candidate_raw: list[Dict[str, Any]] = []
                for msg in candidate_raw:
                    content_text = text_content_for_pattern_matching(msg.get("content")) or ""
                    volatile_digest = self._active_replay_placeholder_digest(content_text)
                    generated_volatile_placeholder = (
                        self._is_volatile_ignored_quarantine_placeholder(msg, content_text)
                        and volatile_digest is not None
                        and volatile_digest in generated_placeholder_hashes
                    )
                    if (
                        self._matches_ignore_message_patterns(msg)
                        or self._mapped_stored_row_matches_ignore_message_patterns(msg)
                        or self._is_ignored_active_replay_placeholder(msg, content_text)
                        or generated_volatile_placeholder
                    ):
                        continue
                    filtered_candidate_raw.append(msg)
            finally:
                self._current_compress_store_ids_by_message_id = previous_store_id_map
            candidate_raw = filtered_candidate_raw
            if not candidate_raw:
                return False, "no eligible raw backlog outside fresh tail"

        if force_overflow:
            return True, "forced overflow recovery"

        raw_tokens_outside_tail = count_messages_tokens(candidate_raw)
        if allow_partial_leaf:
            return True, "eligible partial threshold-sweep leaf"
        if self._config.dynamic_leaf_chunk_enabled:
            working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
        else:
            working_leaf_chunk_tokens = self._config.leaf_chunk_tokens
        if raw_tokens_outside_tail < working_leaf_chunk_tokens:
            return False, "raw backlog outside fresh tail is below leaf chunk threshold"
        return True, "eligible raw backlog outside fresh tail"

    def _working_leaf_chunk_tokens(self, raw_tokens_outside_tail: int) -> int:
        base = max(1, self._config.leaf_chunk_tokens)
        if not self._config.dynamic_leaf_chunk_enabled:
            return base
        ceiling = max(base, self._config.dynamic_leaf_chunk_max)
        working = base
        while working < ceiling and raw_tokens_outside_tail > working * 2:
            working = min(ceiling, working * 2)
        return working

    def _start_leaf_lookahead(
        self,
        candidate_raw: List[Dict[str, Any]],
        chunk_tokens: int,
        *,
        dependent_reply_message_ids: set[int],
        focus_topic: Optional[str],
        deadline: Optional[float],
        estimated_active_tokens: int,
        remaining_passes: int,
    ):
        """fork: betterlcm — plan the chunks this compress() will take and start summarising
        them ahead; None when concurrency is 1 or nothing beyond the first chunk is planned."""
        concurrency = int(self.effective_summary_concurrency or 1)
        if concurrency <= 1 or remaining_passes <= 1:
            return None
        from .leaf_pipeline import DaemonThreadPoolExecutor, LeafLookahead, chunks_needed, plan_chunks
        needed = chunks_needed(
            estimated_active_tokens,
            self._non_sweep_drain_stop_tokens(),
            chunk_tokens,
            float(getattr(self._config, "leaf_summary_ratio", 0.20) or 0.20),
            extra=concurrency - 1,
        )
        chunks = plan_chunks(
            candidate_raw,
            chunk_tokens,
            self._select_oldest_leaf_chunk_aligned,
            max_chunks=max(1, min(remaining_passes, needed)),
        )
        if len(chunks) <= 1:
            return None
        lookahead = LeafLookahead(
            self._summarize_leaf_chunk_with_rescue,
            chunks,
            concurrency=concurrency,
            focus_topic=focus_topic,
            deadline=deadline,
            input_filter=lambda chunk: [m for m in chunk if id(m) not in dependent_reply_message_ids],
            executor=self._leaf_worker_pool(concurrency),  # fork: one pool per engine
        )
        logger.info(
            "LCM leaf lookahead: %d chunk(s) planned, concurrency %d", lookahead.planned, concurrency
        )
        return lookahead

    def _leaf_worker_pool(self, concurrency: int):
        """fork: betterlcm — ONE worker pool per engine, reused across compaction attempts.

        A per-attempt pool bounded each attempt on its own: a host that abandoned a compaction
        and retried left the first attempt's blocked summariser calls running and started a
        second set of workers (round-2 verify-2 #6). Sharing the pool bounds the live workers
        for the engine, whatever the host does; abandoned calls simply occupy it until they
        return, which is exactly the back-pressure the retry needs to see.
        """
        from .leaf_pipeline import DaemonThreadPoolExecutor

        pool = getattr(self, "_leaf_pool", None)
        if pool is None or getattr(pool, "_max_workers", 0) < concurrency:
            if pool is not None:
                pool.shutdown(wait=False)
            pool = DaemonThreadPoolExecutor(
                max_workers=max(1, int(concurrency)), thread_name_prefix="lcm-leaf"
            )
            self._leaf_pool = pool
        return pool

    def _take_leaf_lookahead(self, summary_input_chunk: List[Dict[str, Any]]):
        """fork: betterlcm — the planned result for this chunk, or None to summarise inline."""
        lookahead = getattr(self, "_leaf_lookahead", None)
        if lookahead is None:
            return None
        if not lookahead.matches_next(summary_input_chunk):
            self._close_leaf_lookahead()
            return None
        result = lookahead.take(summary_input_chunk)
        if lookahead.remaining == 0:
            self._close_leaf_lookahead()
        return result

    def _close_leaf_lookahead(self) -> None:
        lookahead = getattr(self, "_leaf_lookahead", None)
        if lookahead is not None:
            self._leaf_lookahead = None
            lookahead.close()

    def _fork_leaf_scheduling_active(self) -> bool:
        """fork: betterlcm — True once the curve has actually left the low anchor.

        At or below ``scale_low_window`` every resolved value is upstream's, so the fork must
        also behave like upstream: one whole-backlog pass, no wall clock of its own. The fork's
        chunking, pass cap and clock only apply above it.
        """
        window = int(getattr(self, "context_length", 0) or 0)
        low = int(getattr(self._config, "scale_low_window", 0) or 262_144)
        return window > low

    def _non_sweep_drain_stop_tokens(self) -> int:
        """fork: betterlcm — where the non-sweep loop stops draining (wire units).

        The curve's low anchor is the resolved context threshold (upstream stopped the
        moment it was under it); at 1M it is 0.30*W.
        """
        window = int(self.context_length or 0)
        if window <= 0:
            return int(self.threshold_tokens or 0)
        fraction = float(self.effective_drain_stop_fraction or 0.0)
        if fraction <= 0:
            return int(self.threshold_tokens or 0)
        return int(window * fraction)

    def _non_sweep_should_continue(
        self,
        estimated_active_tokens: int,
        *,
        force_overflow: bool,
        deferred_maintenance_active: bool,
    ) -> bool:
        """fork: betterlcm — after a non-dynamic pass, run another only while over the stop."""
        if force_overflow or deferred_maintenance_active:
            return False
        stop = self._non_sweep_drain_stop_tokens()
        if stop <= 0:
            return False
        return int(estimated_active_tokens or 0) > stop

    def _select_oldest_leaf_chunk_aligned(
        self,
        candidate_raw: List[Dict[str, Any]],
        working_leaf_chunk_tokens: int,
    ) -> List[Dict[str, Any]]:
        """fork: betterlcm — token-greedy oldest chunk, extended so it never ends between an
        assistant tool call and the tool results that answer it (mirrors fresh_tail.py)."""
        selected = self._select_oldest_leaf_chunk(candidate_raw, working_leaf_chunk_tokens)
        if not selected or len(selected) >= len(candidate_raw):
            return selected
        end = len(selected)
        last = selected[-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            call_ids = {_tool_call_id(tc) for tc in (last.get("tool_calls") or [])}
            while end < len(candidate_raw):
                following = candidate_raw[end]
                if following.get("role") == "tool" and str(following.get("tool_call_id") or "").strip() in call_ids:
                    end += 1
                    continue
                break
        elif last.get("role") == "tool":
            # ends inside a result run: take the rest of that run
            while end < len(candidate_raw) and candidate_raw[end].get("role") == "tool":
                end += 1
        return candidate_raw[:end]

    def _select_oldest_leaf_chunk(
        self,
        candidate_raw: List[Dict[str, Any]],
        working_leaf_chunk_tokens: int,
    ) -> List[Dict[str, Any]]:
        selected: list[Dict[str, Any]] = []
        used = 0
        for msg in candidate_raw:
            msg_tokens = count_message_tokens(msg)
            if used + msg_tokens > working_leaf_chunk_tokens and selected:
                break
            selected.append(msg)
            used += msg_tokens
        return selected

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None,
                 focus_topic: Optional[str] = None,
                 force: bool = False) -> List[Dict[str, Any]]:
        """Run compaction and leave a terminal public status on every failure."""
        try:
            return self._compress_impl(
                messages,
                current_tokens=current_tokens,
                focus_topic=focus_topic,
                force=force,
            )
        except BaseException:
            self._last_compression_status = "error"
            self._last_compression_noop_reason = ""
            raise

    def _compress_impl(self, messages: List[Dict[str, Any]],
                       current_tokens: int = None,
                       focus_topic: Optional[str] = None,
                       force: bool = False) -> List[Dict[str, Any]]:
        """Main compaction entry point.

        1. Ingest any new messages into the store
        2. Identify messages outside the fresh tail
        3. Summarize them into DAG leaf nodes
        4. Check if condensation is needed
        5. Assemble new active context: summaries + fresh tail
        """
        if not messages:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "empty message list"
            return messages

        self._last_compression_status = "running"
        self._last_compression_noop_reason = ""
        _compress_started = time.perf_counter()

        self._maybe_reclassify_late_auxiliary_before_compaction_write()
        if self._bypasses_lcm_context_management():
            bypass_current_tokens = current_tokens
            if bypass_current_tokens is None or bypass_current_tokens <= 0:
                auxiliary_session_id = self._thread_context_session_id()
                if auxiliary_session_id:
                    auxiliary_prompt_tokens = self._current_auxiliary_prompt_tokens(
                        auxiliary_session_id
                    )
                    if auxiliary_prompt_tokens > 0:
                        bypass_current_tokens = auxiliary_prompt_tokens
            return self._compress_lcm_bypassed_session(
                messages,
                current_tokens=bypass_current_tokens,
                focus_topic=focus_topic,
                force=force,
            )

        observed_prompt_tokens = current_tokens if current_tokens is not None else None
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_prompt_tokens,
            messages=messages,
        )
        # NOTE: deliberately do NOT clear the spend guard on force_overflow.
        # force_overflow is automatic (set every turn the prompt exceeds the
        # assembly cap), which is exactly the sustained-over-cap state a runaway
        # compaction loop produces - clearing it per turn would defeat the guard
        # in the case it exists for. A tripped guard still converges the
        # emergency via deterministic L3 truncation (no LLM spend).
        recovery_assembly_cap = (
            self._overflow_recovery_assembly_cap(
                observed_tokens=observed_prompt_tokens,
                messages=messages,
            )
            if force_overflow
            else None
        )

        # Step 1: Ingest new messages into the immutable store. Work from a
        # replay-safe view so quarantined assistant loops do not enter summaries
        # or provider context after the durable row has been written.
        working_messages = self._ingest_messages(messages)
        ingest_cleanup_changed_active_context = working_messages != messages
        cleanup_only_due_to_boundary_cooldown = bool(
            self._preflight_cleanup_only_due_to_boundary_cooldown
            and not force_overflow
        )
        self._preflight_cleanup_only_due_to_boundary_cooldown = False
        if cleanup_only_due_to_boundary_cooldown:
            sanitized_messages = self._sanitize_active_context_messages(
                working_messages,
                insert_missing_tool_stubs=False,
            )
            self._refresh_raw_backlog_debt(
                sanitized_messages,
                observed_tokens=observed_prompt_tokens,
            )
            self._ingest_cursor = len(sanitized_messages)
            self._last_compression_status = "sanitized"
            self._last_compression_noop_reason = ""
            self._write_generated_ignored_placeholder_hash_counts(
                self._generated_placeholder_digest_budget_for_active_replay(
                    sanitized_messages
                )
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                self._generated_placeholder_digest_ordinals_for_active_replay(
                    sanitized_messages
                )
            )
            return sanitized_messages
        anchor_source_messages = list(working_messages)
        pressure_messages = messages if len(messages) == len(working_messages) else working_messages
        leaf_compacted_this_turn = False
        dropped_replayed_scaffold_messages = False
        leaf_passes = 0
        estimated_active_tokens = (
            observed_prompt_tokens
            if observed_prompt_tokens is not None and observed_prompt_tokens > 0
            else count_messages_tokens(messages)
        )
        threshold_full_sweep_active = bool(
            self._config.threshold_full_sweep_enabled
            and not force_overflow
            and self.threshold_tokens > 0
            and estimated_active_tokens >= self.threshold_tokens
        )
        # fork: betterlcm — one wall budget for the whole leaf loop on BOTH paths
        # (120 s at 256k = upstream's sweep constant; 200 s at 1M). Upstream bounded only
        # the sweep; the non-sweep path ran a single pass so it needed no clock.
        leaf_loop_max_seconds = float(self.effective_leaf_loop_max_seconds or _THRESHOLD_FULL_SWEEP_MAX_SECONDS)
        sweep_max_passes = max(1, int(getattr(self._config, "sweep_max_passes", 0) or _THRESHOLD_FULL_SWEEP_MAX_PASSES))
        sweep_deadline = time.monotonic() + leaf_loop_max_seconds
        # fork: betterlcm — the wall clock belongs to the FORK's multi-pass scheduling. The
        # sweep keeps its own (upstream had one there); upstream's ordinary and dynamic-chunk
        # paths had none, so imposing one on them changed behaviour at the low anchor where the
        # contract says the fork must be upstream (audit D #4).
        leaf_deadline = sweep_deadline if (threshold_full_sweep_active
                                           or self._fork_leaf_scheduling_active()) else None
        # fork: curved. Explicit summary_prefix_target_tokens wins; otherwise the curve, whose low
        # anchor is leaf_chunk_tokens exactly like upstream's fallback.
        sweep_target_tokens = max(1, int(self.effective_sweep_target_tokens))
        sweep_summary_prefix_before = (
            self._summary_frontier_tokens() if threshold_full_sweep_active else 0
        )
        if threshold_full_sweep_active:
            self._last_threshold_full_sweep = {
                "status": "running",
                "leaf_passes": 0,
                "condensation_passes": 0,
                "total_passes": 0,
                "duration_ms": 0.0,
                "tokens_before": estimated_active_tokens,
                "tokens_after": estimated_active_tokens,
                "summary_prefix_tokens_before": sweep_summary_prefix_before,
                "summary_prefix_tokens_after": sweep_summary_prefix_before,
                "summary_prefix_target_tokens": sweep_target_tokens,
                "stop_reason": "",
                "budget_exhausted": False,
            }
        critical_budget_pressure = self._critical_budget_pressure_reached(
            observed_tokens=observed_prompt_tokens,
            messages=working_messages,
        )
        deferred_maintenance_active = (
            not force_overflow
            and not threshold_full_sweep_active
            and self._should_run_deferred_maintenance(
                working_messages,
                observed_tokens=observed_prompt_tokens,
            )
        )
        if deferred_maintenance_active:
            self._lifecycle.record_maintenance_attempt(self._conversation_id)
        # fork: betterlcm — when the preflight's replay-diff branch requested this
        # compress() under the threshold and nothing else forces summarising (manual
        # /compress, overflow, deferred maintenance, critical pressure, ignored backlog to
        # consume), run the cleanup preamble (scaffold / ignored / dependent-reply drops)
        # and publish, but spend no summariser calls and shrink nothing. Default configs
        # never take this path (replay == messages, so preflight never requests a cleanup).
        cleanup_only = self._compress_is_cleanup_only(
            force=force,
            force_overflow=force_overflow,
            estimated_active_tokens=estimated_active_tokens,
            deferred_maintenance_active=deferred_maintenance_active,
            critical_budget_pressure=critical_budget_pressure,
            working_messages=working_messages,
        )
        # fork: betterlcm — non-dynamic pass cap is curved (1 at 256k = upstream; 64 at 1M)
        base_max_leaf_passes = (
            4 if self._config.dynamic_leaf_chunk_enabled else max(1, int(self.effective_leaf_pass_cap or 1))
        )
        max_leaf_passes = base_max_leaf_passes
        if threshold_full_sweep_active:
            max_leaf_passes = sweep_max_passes
        if deferred_maintenance_active:
            max_leaf_passes = max(1, self._config.deferred_maintenance_max_passes)

        explicit_focus_topic = focus_topic is not None

        noop_reason = "no eligible raw backlog outside fresh tail"
        sweep_stop_reason = ""
        sweep_raw_drained = False
        dependent_reply_message_ids: set[int] = set()
        preexisting_dependent_reply_records = self._load_generated_ignored_dependent_reply_records()

        while leaf_passes < max_leaf_passes:
            if leaf_deadline is not None and time.monotonic() >= leaf_deadline:
                if threshold_full_sweep_active:
                    sweep_stop_reason = "time_budget_exhausted"
                else:
                    noop_reason = "leaf loop time budget exhausted"  # fork: non-sweep clock
                break
            fresh_tail_start = self._fresh_tail_start(pressure_messages)

            # Keep only a real system prompt anchored. Gateway sessions may
            # pass only conversation messages, so index 0 can be an old user
            # turn; that must remain eligible for compaction instead of being
            # replayed forever as fresh-looking intent.
            leading_anchor_count = self._leading_anchor_count(working_messages)
            if fresh_tail_start <= leading_anchor_count:
                noop_reason = "no eligible raw backlog outside fresh tail"
                if threshold_full_sweep_active:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                break

            candidate_start = leading_anchor_count
            while (
                candidate_start < fresh_tail_start
                and self._is_replayed_context_scaffold_message(working_messages[candidate_start])
            ):
                candidate_start += 1
            if candidate_start > leading_anchor_count:
                dropped_replayed_scaffold_messages = True
                working_messages = working_messages[:leading_anchor_count] + working_messages[candidate_start:]
                pressure_messages = pressure_messages[:leading_anchor_count] + pressure_messages[candidate_start:]
                candidate_start = leading_anchor_count
                fresh_tail_start = self._fresh_tail_start(pressure_messages)
                if fresh_tail_start <= leading_anchor_count:
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            if candidate_start < fresh_tail_start:
                self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(
                    working_messages[leading_anchor_count:]
                )
                compactable_pairs = list(
                    zip(
                        working_messages[candidate_start:fresh_tail_start],
                        pressure_messages[candidate_start:fresh_tail_start],
                    )
                )
                kept_working: list[Dict[str, Any]] = []
                kept_pressure: list[Dict[str, Any]] = []
                dropped_ignored_backlog = False
                drop_dependent_reply = False
                # fork: betterlcm — one metadata read per pass, not one per message
                generated_placeholder_hashes = self._load_generated_ignored_placeholder_hashes()
                for working_msg, pressure_msg in compactable_pairs:
                    role = str(working_msg.get("role") or "")
                    content_text = text_content_for_pattern_matching(working_msg.get("content")) or ""
                    generated_dependent_reply = self._is_generated_ignored_dependent_reply(
                        working_msg,
                        content_text,
                    )
                    volatile_digest = self._active_replay_placeholder_digest(content_text)
                    generated_volatile_placeholder = (
                        self._is_volatile_ignored_quarantine_placeholder(working_msg, content_text)
                        and volatile_digest is not None
                        and volatile_digest in generated_placeholder_hashes
                    )
                    if (
                        self._matches_ignore_message_patterns(working_msg)
                        or self._matches_ignore_message_patterns(pressure_msg)
                        or self._mapped_stored_row_matches_ignore_message_patterns(working_msg)
                        or self._is_ignored_active_replay_placeholder(working_msg, content_text)
                        or generated_volatile_placeholder
                    ):
                        dropped_ignored_backlog = True
                        if role in {"user", "system", "tool", "assistant"}:
                            drop_dependent_reply = True
                        continue
                    if generated_dependent_reply:
                        dependent_reply_message_ids.add(id(working_msg))
                        if role in {"assistant", "tool"}:
                            drop_dependent_reply = True
                    if drop_dependent_reply and role in {"assistant", "tool"}:
                        dependent_reply_message_ids.add(id(working_msg))
                        self._remember_generated_ignored_dependent_reply(working_msg, content_text)
                    if role in {"user", "system"}:
                        drop_dependent_reply = False
                    kept_working.append(working_msg)
                    kept_pressure.append(pressure_msg)
                drop_dependent_reply_into_tail = drop_dependent_reply
                if dropped_ignored_backlog:
                    dropped_replayed_scaffold_messages = True
                    working_messages = (
                        working_messages[:candidate_start]
                        + kept_working
                        + working_messages[fresh_tail_start:]
                    )
                    pressure_messages = (
                        pressure_messages[:candidate_start]
                        + kept_pressure
                        + pressure_messages[fresh_tail_start:]
                    )
                    fresh_tail_start = self._fresh_tail_start(pressure_messages)
                if drop_dependent_reply_into_tail:
                    tail_scan_start = max(fresh_tail_start, leading_anchor_count)
                    pending_tail_dependents: list[tuple[Dict[str, Any], str]] = []
                    saw_tail_boundary = False
                    for tail_msg in working_messages[tail_scan_start:]:
                        if not isinstance(tail_msg, dict):
                            continue
                        tail_role = str(tail_msg.get("role") or "")
                        if tail_role in {"user", "system"}:
                            saw_tail_boundary = True
                            break
                        if tail_role in {"assistant", "tool"}:
                            tail_text = text_content_for_pattern_matching(tail_msg.get("content")) or ""
                            self._remember_generated_ignored_dependent_reply(tail_msg, tail_text)
                            pending_tail_dependents.append((tail_msg, tail_text))
                    if saw_tail_boundary or leading_anchor_count > 0 or kept_working:
                        for tail_msg, _tail_text in pending_tail_dependents:
                            dependent_reply_message_ids.add(id(tail_msg))
                if dropped_ignored_backlog and fresh_tail_start <= leading_anchor_count:
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            # Auto-derive focus topic from the post-filter compaction view when
            # not explicitly provided.  The derived focus is summarizer-visible,
            # so it must follow the same ignored-message filtering as the leaf
            # chunk itself.
            if not explicit_focus_topic:
                focus_topic = self._derive_auto_focus_topic(working_messages)

            candidate_raw = working_messages[leading_anchor_count:fresh_tail_start]
            if not candidate_raw:
                noop_reason = "no eligible raw backlog outside fresh tail"
                if threshold_full_sweep_active:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                break

            pressure_candidate_raw = pressure_messages[leading_anchor_count:fresh_tail_start]
            raw_tokens_outside_tail = count_messages_tokens(pressure_candidate_raw)
            if threshold_full_sweep_active:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(
                    raw_tokens_outside_tail
                )
                # fork: betterlcm — tool-group aligned here too. A chunk boundary inside an
                # assistant/tool group shows the summariser a call with no result and leaves
                # the result to a different leaf; correctness cannot depend on which optional
                # mode is on (audit p05 CP06).
                to_compact = self._select_oldest_leaf_chunk_aligned(
                    candidate_raw,
                    working_leaf_chunk_tokens,
                )
            elif self._config.dynamic_leaf_chunk_enabled:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
                if raw_tokens_outside_tail < working_leaf_chunk_tokens and not force_overflow:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                if force_overflow:
                    to_compact = candidate_raw
                else:
                    # fork: betterlcm — aligned in the dynamic branch too (audit p05 CP06)
                    to_compact = self._select_oldest_leaf_chunk_aligned(
                        candidate_raw, working_leaf_chunk_tokens
                    )
            else:
                if raw_tokens_outside_tail < self._config.leaf_chunk_tokens and not force_overflow:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                # fork: betterlcm — curved chunk: the whole backlog at 256k (upstream), 0.04*W
                # at 1M; chunk boundaries never split an assistant/tool group.
                curved_chunk_tokens = int(self.effective_leaf_chunk_tokens or 0)
                if (force_overflow
                        or curved_chunk_tokens <= 0
                        or not self._fork_leaf_scheduling_active()   # fork: at the low anchor,
                        or raw_tokens_outside_tail <= curved_chunk_tokens):
                    # "the entire eligible backlog" is upstream's BEHAVIOUR, not a size; a
                    # resumed or imported history larger than the window must still go in one
                    # pass there, rather than being chunked by a value that merely equals W.
                    to_compact = candidate_raw
                else:
                    to_compact = self._select_oldest_leaf_chunk_aligned(candidate_raw, curved_chunk_tokens)
                    # fork: betterlcm — with concurrency > 1, summarise the NEXT chunks on
                    # workers while this one is persisted (leaf_pipeline.LeafLookahead).
                    if (
                        getattr(self, "_leaf_lookahead", None) is None
                        and not deferred_maintenance_active
                        and not cleanup_only  # fork: betterlcm — see the cleanup-only break below
                    ):
                        self._leaf_lookahead = self._start_leaf_lookahead(
                            candidate_raw,
                            curved_chunk_tokens,
                            dependent_reply_message_ids=dependent_reply_message_ids,
                            focus_topic=focus_topic,
                            deadline=leaf_deadline,
                            estimated_active_tokens=estimated_active_tokens,
                            remaining_passes=max_leaf_passes - leaf_passes,
                        )

            if not to_compact:
                noop_reason = "no eligible leaf chunk selected"
                break

            if cleanup_only:
                # fork: betterlcm — this pass publishes nothing, so it must not START any
                # model work either. Upstream checked the restriction only AFTER launching
                # lookahead summarisation, pre-compaction extraction and assertion scheduling,
                # spending time and the spend guard's budget on results it then discarded —
                # and at 1M, where lookahead runs several chunks concurrently, that can block
                # the compaction that actually needs to happen (audit p05 CP07).
                noop_reason = "below threshold: cleanup only, no leaf pass"
                break

            selected_raw_chunk = to_compact
            # fork: betterlcm — the identity this leaf is being built FOR, captured before any
            # summariser work (round-2 verify-4 #3 / RS02)
            leaf_fence = self._publication_fence()
            summary_input_chunk = [
                message for message in selected_raw_chunk if id(message) not in dependent_reply_message_ids
            ]
            if not summary_input_chunk:
                compacted_chunk = selected_raw_chunk
                source_tokens = count_messages_tokens(selected_raw_chunk)
                summary_text = (
                    "Filtered replies derived from ignored messages.\n"
                    "[Expand for details: ignored-dependent reply]"
                )
                _level = 0
                _rescue_attempts = 0
            else:
                # Pre-compaction extraction: best-effort, never blocks compaction.
                # Use the same dependency-filtered view as summarization so ignored
                # turns cannot leak through derived assistant/tool replies.
                if self._config.extraction_enabled:
                    extraction_timeout = None
                    if threshold_full_sweep_active:
                        extraction_timeout = max(0.001, sweep_deadline - time.monotonic())
                    self._run_pre_compaction_extraction(
                        summary_input_chunk,
                        timeout_seconds=extraction_timeout,
                    )
                if bool(
                    getattr(
                        self._config,
                        "assertion_extraction_enabled",
                        False,
                    )
                ):
                    self._schedule_pre_compaction_assertions(summary_input_chunk)

                try:
                    summary_kwargs: dict[str, Any] = {"focus_topic": focus_topic}
                    # fork: betterlcm — the leaf loop's clock reaches the summariser in every
                    # mode that has one, not only under the sweep flag. Upstream checked its
                    # deadline between passes while each pass could still spend one timeout
                    # per route per level inside escalation (audit p05 CP05).
                    if threshold_full_sweep_active:
                        summary_kwargs["deadline"] = sweep_deadline
                    elif leaf_deadline is not None:
                        summary_kwargs["deadline"] = leaf_deadline
                    lookahead_result = self._take_leaf_lookahead(summary_input_chunk)  # fork
                    if lookahead_result is not None:
                        (
                            compacted_chunk,
                            source_tokens,
                            summary_text,
                            _level,
                            _rescue_attempts,
                        ) = lookahead_result
                    else:
                        (
                            compacted_chunk,
                            source_tokens,
                            summary_text,
                            _level,
                            _rescue_attempts,
                        ) = self._summarize_leaf_chunk_with_rescue(
                            summary_input_chunk,
                            **summary_kwargs,
                        )
                    if len(compacted_chunk) != len(summary_input_chunk):
                        # a rescue shrank the chunk: the plan no longer matches the residual
                        self._close_leaf_lookahead()
                except Exception as exc:
                    # fork: betterlcm — an unavailable summariser never discards what this
                    # call already did: passes persisted so far stay, the cleanup preamble's
                    # drops are published, and the error is recorded so HostCooldownMixin
                    # arms the cooldown. (Upstream tolerated failures only under the sweep
                    # flag; its ``raise`` was unreachable while L3 guaranteed convergence.)
                    tolerated = isinstance(exc, SummaryUnavailableError) or threshold_full_sweep_active
                    if not tolerated:
                        raise
                    self._last_leaf_summary_error = str(exc)
                    if threshold_full_sweep_active:
                        sweep_stop_reason = "leaf_summary_error"
                    logger.warning(
                        "LCM leaf compaction stopped after %d persisted leaf pass(es): %s",
                        leaf_passes,
                        exc,
                    )
                    break
            compacted_summary_ids = {id(message) for message in compacted_chunk}
            compacted_positions = [
                idx for idx, message in enumerate(selected_raw_chunk) if id(message) in compacted_summary_ids
            ]
            last_compacted_raw_pos = max(compacted_positions) if compacted_positions else len(compacted_chunk) - 1
            last_consumed_raw_pos = last_compacted_raw_pos
            while (
                last_consumed_raw_pos + 1 < len(selected_raw_chunk)
                and id(selected_raw_chunk[last_consumed_raw_pos + 1]) in dependent_reply_message_ids
            ):
                last_consumed_raw_pos += 1
            source_lookup_chunk = selected_raw_chunk[: last_consumed_raw_pos + 1]
            selected_raw_len = len(source_lookup_chunk)
            remaining_messages = working_messages[leading_anchor_count + selected_raw_len:]
            source_tokens = count_messages_tokens(source_lookup_chunk)

            source_lineage_chunk = [
                message for message in source_lookup_chunk if id(message) not in dependent_reply_message_ids
            ]
            source_store_ids = self._get_store_ids_for_messages(source_lineage_chunk)
            source_store_ids = sorted(dict.fromkeys(source_store_ids))
            # fork: betterlcm — a summary with no provenance is exactly the thing this engine
            # exists to prevent: unexpandable, unverifiable, and indistinguishable from an
            # invented one. Refusing only a mapping of ZERO was not enough: a PARTIAL mapping
            # published a node over some of the consumed rows, advanced the frontier past all
            # of them, and left the unmapped rows belonging to no summary at all while the
            # text claimed to cover them (verify-4 #3). Every consumed row must map.
            mapped_by_message_id = self._get_store_id_map_for_messages(source_lookup_chunk)
            unmapped = [
                message for message in source_lookup_chunk
                if id(message) not in mapped_by_message_id
            ]
            # A host truncation marker with no durable copy legitimately has no row of its own:
            # the host owns that file and the plugin could not copy it. Refusing forever would
            # stop compaction for the whole session, so those are published WITH a marker
            # saying they cannot be expanded; anything else unmapped is a real provenance
            # failure and still refuses (round-2 verify-2 #2).
            unmappable_markers = [
                message for message in unmapped
                if str(message.get("role") or "") == "tool"
                and self._is_unmappable_host_truncation_marker(message)
            ]
            if len(unmappable_markers) == len(unmapped):
                unmapped = []
            if source_lookup_chunk and unmapped:
                noop_reason = (
                    "selected leaf chunk lost its raw store lineage"
                    if len(unmapped) == len(source_lookup_chunk)
                    else "selected leaf chunk maps only part of its consumed rows"
                )
                self._last_leaf_summary_error = noop_reason
                logger.warning(
                    "LCM refusing to publish a leaf: %d of %d consumed message(s) have no raw "
                    "store lineage",
                    len(unmapped), len(source_lookup_chunk),
                )
                break
            consumed_store_ids = self._get_store_ids_for_messages(source_lookup_chunk)
            consumed_store_ids = sorted(dict.fromkeys(consumed_store_ids))

            # fork: betterlcm — every row this leaf CONSUMES becomes a source of it. Upstream
            # published only the summarised lineage, so replies to host-injected placeholders
            # (excluded from the summariser input on purpose) were swept past the frontier and
            # then reachable from no node at all: expanding the summary that covers their span
            # returned the other messages and never mentioned them (audit p05 CP01). They stay
            # out of the summary TEXT and are named in a marker instead, so the extra sources
            # can never read as content the summariser claimed to cover.
            summarised_source_ids = set(source_store_ids)  # built once, not per source id
            # fork: betterlcm — an archive row holding the bytes of a recovered host output
            # sits next to the marker row it belongs to and is in no active-context message, so
            # it mapped to no node and the summary covering its marker read as unexpandable
            # (round-2 verify-4 #1). It is a source of this leaf, named in its own receipt.
            revision_ids = self._store.revision_rows_for(  # fork: round-3 verify-2 #8
                self._session_id, consumed_store_ids
            )
            # fork: betterlcm — the explicit attachment link first (round-3 verify-4 #24); the
            # call-id lookup remains for rows written before that link existed.
            recovered_body_ids = sorted(set(
                self._store.attached_recovered_body_ids_for_rows(
                    self._session_id, consumed_store_ids
                )
            ) | set(
                self._store.attached_recovered_body_ids(
                    self._session_id,
                    [
                        str(message.get("tool_call_id") or "")
                        for message in source_lookup_chunk
                        if str(message.get("role") or "") == "tool"
                    ],
                    exclude_ids=consumed_store_ids,
                )
            ))
            published_source_ids = sorted(
                summarised_source_ids
                | set(consumed_store_ids)
                | set(recovered_body_ids)
                | set(revision_ids)
            )
            excluded_source_ids = [
                store_id for store_id in published_source_ids
                if store_id not in summarised_source_ids
                and store_id not in set(recovered_body_ids)
                and store_id not in set(revision_ids)
            ]
            if revision_ids:
                summary_text = summary_text.rstrip() + "\n" + marked_loss.revision_rows_marker(
                    revision_ids
                )
            if recovered_body_ids:
                summary_text = summary_text.rstrip() + "\n" + marked_loss.recovered_body_rows_marker(
                    recovered_body_ids
                )
            if excluded_source_ids:
                summary_text = summary_text.rstrip() + "\n" + marked_loss.excluded_reply_marker(
                    excluded_source_ids
                )
            if unmappable_markers:
                summary_text = summary_text.rstrip() + "\n" + marked_loss.unmappable_rows_marker(
                    len(unmappable_markers)
                )
            earliest_at, latest_at = self._store.get_time_bounds(published_source_ids)
            summary_tokens = count_tokens(summary_text)

            # fork: betterlcm — validation, the node write and the frontier advance happen
            # under ONE lock. Checking the fence and then publishing left a window in which a
            # rebind published the old session's work and advanced the NEW session's frontier
            # over it (round-3 verify-2 #3 / verify-4 #1). on_session_start and reset take the
            # same lock, so a rebind either happens entirely before this block or waits.
            with self._publication_lock:
                try:
                    self._check_publication_fence(leaf_fence, what="leaf summary")
                except SummaryUnavailableError as exc:
                    # the session was rebound while this chunk was being summarised: publishing
                    # it now would attribute the old session's content to the new one and
                    # advance a frontier that belongs to neither. Keep the raw messages.
                    self._last_leaf_summary_error = str(exc)
                    logger.warning("LCM discarding a stale leaf summary: %s", exc)
                    break

                node = SummaryNode(
                    session_id=leaf_fence[0],
                    depth=0,
                    summary=summary_text,
                    token_count=summary_tokens,
                    source_token_count=source_tokens,
                    source_ids=published_source_ids,
                    source_type="messages",
                    created_at=time.time(),
                    earliest_at=earliest_at,
                    latest_at=latest_at,
                    expand_hint=self._extract_expand_hint(summary_text),
                )
                # fork: betterlcm — node + sidecar in one transaction (audit p05 CP03)
                self._dag.add_node_with_meta(node, level=int(_level), summary=summary_text)
                self._invalidate_rollups_for_published_node(node)
                self._maybe_gc_compacted_tool_results(compacted_chunk, source_store_ids)
                self._last_compacted_store_id = max(consumed_store_ids) if consumed_store_ids else 0
                self._persist_frontier_marker()

            pressure_consumed_chunk = pressure_messages[
                leading_anchor_count:leading_anchor_count + selected_raw_len
            ]
            pressure_remaining_messages = pressure_messages[leading_anchor_count + selected_raw_len:]
            working_messages = working_messages[:leading_anchor_count] + remaining_messages
            pressure_messages = pressure_messages[:leading_anchor_count] + pressure_remaining_messages
            leaf_compacted_this_turn = True
            leaf_passes += 1
            # fork: betterlcm — subtract what the PRESSURE view holds for the span just
            # consumed. ``estimated_active_tokens`` starts from the host's observed prompt
            # size, which counts the original messages; ``source_tokens`` counts the working
            # copy, which stubbing/redaction can have shortened by orders of magnitude (2,505
            # tokens of tool output against a 15-token stub). Subtracting the small number
            # from the large estimate left phantom pressure behind and drove compaction and
            # spend that the window-scaled target never asked for (audit p05 CP08).
            consumed_pressure_tokens = (
                count_messages_tokens(pressure_consumed_chunk)
                if pressure_consumed_chunk
                else source_tokens
            )
            estimated_active_tokens = max(
                0, estimated_active_tokens - consumed_pressure_tokens + summary_tokens
            )

            if threshold_full_sweep_active:
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_tail_start = self._fresh_tail_start(pressure_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                if not remaining_raw:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                    break
                continue

            if not self._config.dynamic_leaf_chunk_enabled:
                # fork: betterlcm — keep draining toward the curved stop (at 256k the pass
                # cap is 1, so this is upstream's single pass exactly).
                if not self._non_sweep_should_continue(
                    estimated_active_tokens,
                    force_overflow=force_overflow,
                    deferred_maintenance_active=deferred_maintenance_active,
                ):
                    break
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_tail_start = self._fresh_tail_start(pressure_messages)
                if not working_messages[leading_anchor_count:remaining_fresh_tail_start]:
                    break
                continue

            if not force_overflow:
                if (not deferred_maintenance_active) and self.threshold_tokens > 0 and estimated_active_tokens < self.threshold_tokens:
                    break
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_tail_start = self._fresh_tail_start(pressure_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                if not remaining_raw:
                    break
                pressure_remaining_raw = pressure_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                remaining_raw_tokens = count_messages_tokens(pressure_remaining_raw)
                remaining_threshold = self._working_leaf_chunk_tokens(remaining_raw_tokens)
                if remaining_raw_tokens < remaining_threshold:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        break

        if (
            threshold_full_sweep_active
            and not sweep_raw_drained
            and not sweep_stop_reason
            and leaf_passes >= max_leaf_passes
        ):
            sweep_stop_reason = "pass_budget_exhausted"

        if not leaf_compacted_this_turn:
            self._refresh_raw_backlog_debt(
                working_messages,
                observed_tokens=observed_prompt_tokens,
            )
            # fork: betterlcm — summary pressure can exist without raw backlog. Upstream
            # returned from here before the condensation step, so a session whose raw prefix
            # was drained (or wholly inside the fresh tail, or below the leaf floor) kept an
            # oversized pile of uncondensed summaries forever: nothing shrank the frontier
            # because nothing new could be compacted (audit p05 CP04). Condensation gates
            # itself on the frontier budget and the fan-in, so this is a no-op unless the pile
            # really is too large.
            condensation_published = 0  # fork: betterlcm — round-2 verify-2 #7
            if not cleanup_only:
                try:
                    if threshold_full_sweep_active:
                        if sweep_raw_drained:
                            condensation_published, sweep_stop_reason = self._run_threshold_sweep_condensation(
                                target_tokens=sweep_target_tokens,
                                pass_budget=max(0, sweep_max_passes - leaf_passes),
                                deadline=sweep_deadline,
                                focus_topic=focus_topic,
                            )
                    else:
                        condensation_published = int(self._maybe_condense(
                            focus_topic=focus_topic,
                            leaf_compacted_this_turn=False,
                            force_overflow=force_overflow,
                            critical_budget_pressure=critical_budget_pressure,
                            deadline=leaf_deadline,
                        ) or 0)
                except SummaryUnavailableError as exc:
                    self._last_leaf_summary_error = str(exc)
                    # fork: betterlcm — a group that FAILED does not erase the groups that were
                    # published before it: the caller kept the original context and reported
                    # compression_count 0 although a new parent existed (round-3 verify-3).
                    condensation_published = int(
                        getattr(self, "_last_condensation_published", 0) or 0
                    )
                    logger.warning(
                        "LCM condensation unavailable with no leaf pass this turn "
                        "(%d group(s) already published): %s",
                        condensation_published, exc,
                    )
            if force_overflow and len(messages) >= 1:
                leading_anchor_count = self._leading_anchor_count(working_messages)
                compressed = self._assemble_overflow_recovery_context(
                    working_messages[0] if leading_anchor_count else None,
                    working_messages[leading_anchor_count:],
                    assembly_cap_override=recovery_assembly_cap,
                )
                return self._finalize_forced_overflow_result(
                    working_messages,
                    compressed,
                    assembly_cap_override=recovery_assembly_cap,
                    ingest_cleanup_changed_active_context=ingest_cleanup_changed_active_context,
                )
            active_context_messages = self._drop_preexisting_generated_ignored_dependent_eof_replies(
                working_messages,
                preexisting_dependent_reply_records,
            )
            # fork: betterlcm — condensation publishes a new parent, so the summary prefix the
            # agent reads has CHANGED. Reassembling only when replayed scaffolding was dropped
            # meant a spent model call and a new depth-1 node returned status "noop" with the
            # original context and no compression accounting (round-2 verify-2 #7).
            reassemble_active_context = bool(
                dropped_replayed_scaffold_messages or condensation_published
            )
            if reassemble_active_context:
                leading_anchor_count = self._leading_anchor_count(active_context_messages)
                anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
                self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
                try:
                    sanitized_messages = self._assemble_context(
                        active_context_messages[0] if leading_anchor_count else None,
                        active_context_messages[leading_anchor_count:],
                        assembly_cap_override=recovery_assembly_cap,
                    )
                finally:
                    self._pending_context_anchor_messages = None
            else:
                sanitized_messages = self._sanitize_active_context_messages(
                    active_context_messages,
                    insert_missing_tool_stubs=False,
                )
            if condensation_published:
                # fork: betterlcm — real published work, even with no leaf pass this turn
                self._ingest_cursor = len(sanitized_messages)
                self.compression_count += 1
                self._last_compaction_duration_ms = (
                    time.perf_counter() - _compress_started
                ) * 1000.0
                self._last_compression_status = "compacted"
                self._last_compression_noop_reason = ""
            elif sanitized_messages != working_messages or ingest_cleanup_changed_active_context:
                # _ingest_messages() already advanced the cursor to the original
                # active-context length. If the host continues from a sanitized
                # or reassembled context, keeping the old cursor could make the
                # next appended messages look already ingested. This applies to
                # content-only cleanup as well as dropped-message cleanup.
                self._ingest_cursor = len(sanitized_messages)
                self._last_compression_status = "sanitized"
                self._last_compression_noop_reason = ""
            else:
                if reassemble_active_context:
                    # The active context changed even though no new leaf node was
                    # written. Keep the cursor aligned with the returned context
                    # so the next appended turn is ingested instead of skipped.
                    self._ingest_cursor = len(sanitized_messages)
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = noop_reason
                logger.info("LCM compression no-op: %s", noop_reason)
            if threshold_full_sweep_active:
                duration_ms = (time.perf_counter() - _compress_started) * 1000.0
                self._last_threshold_full_sweep = {
                    **self._last_threshold_full_sweep,
                    "status": "compacted" if condensation_published else "noop",
                    "duration_ms": round(duration_ms, 3),
                    "stop_reason": sweep_stop_reason or noop_reason,
                    "budget_exhausted": sweep_stop_reason
                    in {"pass_budget_exhausted", "time_budget_exhausted"},
                }
            self._write_generated_ignored_placeholder_hash_counts(
                self._generated_placeholder_digest_budget_for_active_replay(sanitized_messages)
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                self._generated_placeholder_digest_ordinals_for_active_replay(sanitized_messages)
            )
            return sanitized_messages

        # Step 6: Check if condensation is needed. A threshold full sweep only
        # condenses after the eligible raw prefix has been drained, and shares
        # the same total pass/deadline budget as its leaf work.
        condensation_passes = 0
        if threshold_full_sweep_active:
            if sweep_raw_drained:
                remaining_passes = max(
                    0,
                    sweep_max_passes - leaf_passes,
                )
                condensation_passes, sweep_stop_reason = (
                    self._run_threshold_sweep_condensation(
                        target_tokens=sweep_target_tokens,
                        pass_budget=remaining_passes,
                        deadline=sweep_deadline,
                        focus_topic=focus_topic,
                    )
                )
        else:
            try:
                self._maybe_condense(
                    focus_topic=focus_topic,
                    leaf_compacted_this_turn=True,
                    force_overflow=force_overflow,
                    critical_budget_pressure=critical_budget_pressure,
                    deadline=leaf_deadline,  # fork: budget-regime condensation shares the clock
                )
            except SummaryUnavailableError as exc:
                # fork: betterlcm — leaf passes above are already COMMITTED and the raw cursor
                # has advanced. Letting a later condensation failure escape made compress()
                # return the original, uncompacted prompt while the DAG had moved on; the next
                # attempt then mapped no sources and published a leaf with empty provenance.
                # Upstream never reached this state because its L3 fallback always converged.
                # Publish the leaf progress that succeeded and arm the cooldown instead.
                self._last_leaf_summary_error = str(exc)
                logger.warning(
                    "LCM condensation unavailable after %d persisted leaf pass(es); "
                    "publishing leaf progress and cooling down: %s",
                    leaf_passes,
                    exc,
                )

        # Step 7: Assemble new active context
        self._refresh_raw_backlog_debt(
            working_messages,
            observed_tokens=observed_prompt_tokens,
        )
        leading_anchor_count = self._leading_anchor_count(working_messages)
        anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
        self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
        try:
            compressed = self._assemble_context(
                working_messages[0] if leading_anchor_count else None,
                working_messages[leading_anchor_count:],
                assembly_cap_override=recovery_assembly_cap,
            )
        finally:
            self._pending_context_anchor_messages = None
        self.compression_count += 1
        self._last_compaction_duration_ms = (time.perf_counter() - _compress_started) * 1000.0
        logger.info(
            "LCM leaf compaction finished in %.1fms", self._last_compaction_duration_ms
        )
        # fork: betterlcm — the assembled context belongs to the session it was built from. A
        # rebind that lands after publication but before the caller receives the result would
        # hand the NEW session the old session's summaries and cursor (round-3 verify-2 #3 /
        # verify-4 #1). The publication itself is already fenced under the lock; here the fence
        # is re-checked, and stale work is returned as an unchanged context instead.
        with self._publication_lock:
            if self._publication_fence() != leaf_fence:
                logger.warning(
                    "LCM discarding an assembled context built for %s#%s: the session is now "
                    "%s#%s",
                    leaf_fence[0], leaf_fence[1], *self._publication_fence(),
                )
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = (
                    "the session was rebound while this compaction was assembling its result"
                )
                return messages
        self._last_compression_status = "compacted"
        self._last_compression_noop_reason = ""
        if recovery_assembly_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > recovery_assembly_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d after compaction; returning best-effort context (%d tokens)",
                    recovery_assembly_cap,
                    count_messages_tokens(compressed),
                )
        # Reset cursor to the length of the compressed context so that
        # only messages appended *after* this point get ingested next time.
        self._ingest_cursor = len(compressed)
        self._ingest_cursor_needs_reconcile = False

        logger.info(
            "LCM compaction #%d: %d messages → %d (%d leaf pass%s, %d→%d tokens, %d DAG nodes%s)",
            self.compression_count,
            len(messages),
            len(compressed),
            leaf_passes,
            "es" if leaf_passes != 1 else "",
            count_messages_tokens(messages),
            count_messages_tokens(compressed),
            len(self._dag.get_session_nodes(self._session_id)),
            ", forced overflow recovery" if force_overflow else "",
        )

        # ── Active-context cleanup / tool-pair guardrail (same as _assemble_context) ──
        # compress() output is consumed directly by the main loop in some
        # edge cases (e.g. forced overflow recovery bypassing _assemble_context).
        compressed = self._sanitize_active_context_messages(compressed)
        if threshold_full_sweep_active:
            total_passes = leaf_passes + condensation_passes
            duration_ms = (time.perf_counter() - _compress_started) * 1000.0
            final_stop_reason = sweep_stop_reason or "raw_prefix_drained"
            partial_stop_reasons = {
                "pass_budget_exhausted",
                "time_budget_exhausted",
                "leaf_summary_error",
                "condensation_error",
                "condensation_no_progress",
                "no_same_depth_condensation_group",
            }
            self._last_threshold_full_sweep = {
                "status": "partial" if final_stop_reason in partial_stop_reasons else "completed",
                "leaf_passes": leaf_passes,
                "condensation_passes": condensation_passes,
                "total_passes": total_passes,
                "duration_ms": round(duration_ms, 3),
                "tokens_before": self._last_threshold_full_sweep["tokens_before"],
                "tokens_after": count_messages_tokens(compressed),
                "summary_prefix_tokens_before": sweep_summary_prefix_before,
                "summary_prefix_tokens_after": self._summary_frontier_tokens(),
                "summary_prefix_target_tokens": sweep_target_tokens,
                "stop_reason": final_stop_reason,
                "budget_exhausted": final_stop_reason
                in {"pass_budget_exhausted", "time_budget_exhausted"},
            }
        self._write_generated_ignored_placeholder_hash_counts(
            self._generated_placeholder_digest_budget_for_active_replay(compressed)
        )
        self._write_generated_ignored_placeholder_hash_ordinals(
            self._generated_placeholder_digest_ordinals_for_active_replay(compressed)
        )
        record_successful_compaction = getattr(
            self,
            "_record_successful_compaction_telemetry",
            None,
        )
        if callable(record_successful_compaction):
            record_successful_compaction()

        return compressed
