"""LCM-bypass compaction and host-fallback-compressor handling.

Extracted verbatim from :mod:`hermes_lcm.engine` as ``BypassMixin`` (WS5 seam).
The methods manage sessions that opt out of LCM context management: detecting
the bypass, mirroring the host's native fallback compressor, and delegating the
bounding of the context to it. State stays on the engine (accessed via
``self``); mixing this in leaves every call site and ``self._*`` reference
unchanged.

Nothing here shortens a message. LCM writes no row and no summary node for an ignored,
stateless or auxiliary session, so a message it dropped or cut would be reachable from
nothing afterwards — not from an expand, not from a summary, not from a marker. Hermes' own
compressor is therefore the only component allowed to shorten these sessions; when it is
unavailable, fails, or decides to abort, the context is handed back whole and the turn is
reported as an abort instead.
"""

import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional

from .session_patterns import build_session_match_keys, matches_session_pattern
from .tokens import count_messages_tokens

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
            # Older Hermes hosts may not expose all constructor kwargs.
            #
            # drop only the kwargs this host cannot take. Upstream fell back
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
                    "LCM could not initialize Hermes native ContextCompressor for bypassed session fallback: %s",
                    exc,
                )
                return None
        except Exception as exc:
            logger.warning(
                "LCM could not initialize Hermes native ContextCompressor for bypassed session fallback: %s",
                exc,
            )
            return None
        self._host_fallback_compressor = compressor
        self._host_fallback_session_id = session_id
        self._sync_host_fallback_compressor(compressor)
        return compressor

    @staticmethod
    def _constructor_supported_kwargs(factory: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """the subset of ``kwargs`` this constructor actually accepts.

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
        """The hard size this context must fit in, or None when none is known.

        It is a bound to REPORT against, never one to cut to: see
        ``_compress_lcm_bypassed_session``. It is therefore the WINDOW (and an assembly cap if
        the operator set one), not ``threshold_tokens``: the threshold is the preference for
        when to start compacting, and a host compaction that lands above it is ordinary, so
        measuring against it would report every delegated compaction as an overflow.
        """
        caps: list[int] = []
        assembly_cap = self._overflow_recovery_assembly_cap(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if assembly_cap is not None:
            caps.append(assembly_cap)
        if self.context_length > 0:
            caps.append(self.context_length)
        return min(caps) if caps else None

    def _decline_bypass_compaction(
        self,
        messages: List[Dict[str, Any]],
        *,
        reason: str,
        session_id: str,
        detail: str,
    ) -> List[Dict[str, Any]]:
        """Hand the context back whole and say that nothing was compacted.

        Fail-before-loss. There is no copy of this session anywhere in LCM, so no size
        pressure can justify removing part of it: upstream answered exactly this situation
        with a head/tail delete plus a character trim and reported it as success (audit p05
        BY01/BY02). The outcome is announced through the flag Hermes already surfaces to the
        user as "Context compression aborted … No messages were dropped — conversation is
        unchanged", so a declined compaction is visible rather than silent.
        """
        error = f"LCM cannot compact a session it does not store ({reason}): {detail}"
        self._last_compress_aborted = True
        self._last_summary_error = error
        self._last_compression_status = "bypass_not_compacted"
        self._last_compression_noop_reason = f"{error}; every message is returned unchanged"
        logger.warning("LCM declined to compact bypassed session %s: %s", session_id, error)
        return messages

    def _compress_lcm_bypassed_session(
        self,
        messages: List[Dict[str, Any]],
        *,
        current_tokens: int | None = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Delegate ignored/stateless context bounding without writing to LCM.

        LCM neither shortens the context on the way in nor edits what the host's compressor
        returns: this session has no stored copy, so every such edit would be unrecoverable.
        Delegation succeeds, or the whole context comes back and the turn is declined.
        """
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

        compressor = self._get_host_fallback_compressor()
        if compressor is None:
            return self._decline_bypass_compaction(
                safe_messages,
                reason=reason,
                session_id=session_id,
                detail="Hermes' native ContextCompressor is unavailable on this host",
            )

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
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
            return self._decline_bypass_compaction(
                safe_messages,
                reason=reason,
                session_id=session_id,
                detail=f"Hermes' native ContextCompressor failed: {exc}",
            )
        self._mirror_host_fallback_state(compressor)
        native_changed = compacted is not safe_messages and compacted != safe_messages
        after_count = int(getattr(compressor, "compression_count", before_count) or before_count)
        if native_changed:
            self.compression_count += max(1, after_count - before_count)
        elif after_count > before_count:
            # the host counted an attempt that changed nothing; mirror its count, no more
            self.compression_count += after_count - before_count

        if not native_changed:
            if bool(getattr(compressor, "_last_compress_aborted", False)):
                # an abort is a decision to PRESERVE, not a failed attempt. Upstream ran its
                # own deterministic delete over the untouched result and cleared this flag, so
                # the host's explicit "do not compress this" became a destructive delete
                # reported as ordinary success (audit p05 BY02). The decision and the flag both
                # stand; the host already tells the user nothing was dropped.
                self._last_compression_status = "bypass_not_compacted"
                self._last_compression_noop_reason = (
                    f"LCM bypassed {reason}: Hermes' native compressor aborted and LCM stores "
                    "no copy of this session; every message is returned unchanged"
                )
                logger.warning(
                    "LCM native compressor aborted for bypassed %s %s; returning its context "
                    "unchanged (LCM stores no copy of this session)",
                    reason,
                    session_id,
                )
                return compacted
            return self._decline_bypass_compaction(
                compacted,
                reason=reason,
                session_id=session_id,
                detail="Hermes' native ContextCompressor compacted nothing",
            )

        # The host's own compaction, returned exactly as the host built it: LCM does not
        # sanitise, re-cut or annotate it. Its own tool-pair repair already ran
        # (``agent/context_compressor.py``), and there is no LCM copy to repair it from.
        if not hasattr(compressor, "_last_compress_aborted"):
            # the mirror above had no flag to copy on this host, so a compaction that really
            # happened would otherwise inherit a previous turn's abort and tell the user
            # nothing was dropped. A host that DOES expose the flag keeps its own answer.
            self._last_compress_aborted = False
        target_tokens = self._bypass_compaction_target_tokens(
            observed_tokens=observed_tokens,
            messages=safe_messages,
        )
        if target_tokens is not None and count_messages_tokens(compacted) > target_tokens:
            # a bounded result is never presented as a complete one
            self._last_compression_status = "bypass_over_bound"
            self._last_compression_noop_reason = (
                f"LCM bypassed {reason}: Hermes' native compressor compacted this session but "
                f"it is still over the {target_tokens}-token bound; LCM stores no copy and "
                "does not cut it further"
            )
            logger.warning(
                "LCM bypassed %s %s is still over the %d-token bound after Hermes' native "
                "compression; LCM stores no copy of this session and will not shorten it",
                reason,
                session_id,
                target_tokens,
            )
        return compacted
