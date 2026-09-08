"""Compression-failure cooldown for a plugin engine (fork: betterlcm).

Hermes only implements cooldown / "compression blocked" handling for its built-in
``ContextCompressor``: on an exception from a *plugin* engine's ``compress()`` the host
re-raises and the turn dies (``conversation_compression.py`` ``compress_context``; no
``except`` around ``_compress_context`` in ``turn_context_compaction.py`` /
``turn_preflight.py``). The host does, however, probe a few duck-typed attributes on any
engine, and honours them when present:

* ``get_active_compression_failure_cooldown(*, refresh=False) -> dict | None``
  (``turn_context_compaction.py``, ``turn_preflight.py``, ``_refresh_persisted_compression_guards``)
* ``type(engine)._automatic_compression_blocked(engine, *, ignore_cooldown=False) -> bool``
  (``conversation_compression._automatic_compression_gate_blocks``)
* ``should_compress_info(prompt_tokens) -> (bool, reason | None)``
  (``turn_context_compaction._blocked_compress_reason`` -> the FAILURE-class warning)

``HostCooldownMixin`` implements exactly that protocol. It must precede ``CompactionMixin``
in ``LCMEngine``'s bases so its ``compress`` / ``should_compress*`` wrappers run first.

Behaviour on summariser failure (``SummaryUnavailableError``):
1. the exception is caught here, never re-raised to the host;
2. the cooldown is armed for ``summary_failure_cooldown_seconds``;
3. ``compress()`` returns the **input list object** unchanged -- identity matters: the host
   treats a new list as "compressed" and compares progress with a different estimator;
4. while the cooldown is active ``should_compress`` is False and ``should_compress_info``
   reports ``cooldown:<seconds>``, so the host prints its blocked-compaction warning
   instead of silently skipping. Ingestion keeps running (``should_compress_preflight``
   still calls upstream so every turn is stored).
Manual ``/compress`` (``force=True``) clears the cooldown and the summariser spend guard.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

try:
    from .errors import SummaryUnavailableError
except ImportError:  # pragma: no cover
    from errors import SummaryUnavailableError  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600.0


class HostCooldownMixin:
    """Fail loud without killing the turn."""

    _lcm_failure_cooldown_until: float = 0.0
    _lcm_failure_error: str = ""
    _lcm_failure_count: int = 0
    _last_leaf_summary_error: str = ""

    # -- cooldown state ----------------------------------------------------------------

    def _cooldown_seconds(self) -> float:
        raw = getattr(self._config, "summary_failure_cooldown_seconds", None)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = DEFAULT_SUMMARY_FAILURE_COOLDOWN_SECONDS
        return max(0.0, value)

    def _cooldown_applies_to_current_session(self) -> bool:
        """fork: betterlcm — a failure cools down the session that suffered it.

        The deadline lives on the engine, and the engine outlives a session: after `/new` or a
        foreground rebind the next session inherited a block it never earned. Backoff for
        retries of the SAME session is the point; punishing an unrelated one is not.
        """
        armed_for = getattr(self, "_lcm_failure_session", None)
        if armed_for is None:
            return True
        return str(armed_for) == str(getattr(self, "_session_id", "") or "")

    def _record_compression_failure(self, error: str) -> None:
        seconds = self._cooldown_seconds()
        self._lcm_failure_cooldown_until = time.monotonic() + seconds
        self._lcm_failure_session = str(getattr(self, "_session_id", "") or "")  # fork: scope
        self._lcm_failure_error = str(error or "")[:500]
        self._lcm_failure_count = int(getattr(self, "_lcm_failure_count", 0)) + 1
        logger.warning(
            "LCM summariser unavailable; compaction cooling down for %.0fs (failure #%d): %s",
            seconds, self._lcm_failure_count, self._lcm_failure_error,
        )

    def on_session_reset(self) -> None:  # type: ignore[override]
        """fork: betterlcm — a reset ends the session the cooldown was armed for."""
        self.clear_compression_failure_cooldown()
        parent = getattr(super(), "on_session_reset", None)
        if callable(parent):
            parent()

    def clear_compression_failure_cooldown(self) -> None:
        self._lcm_failure_cooldown_until = 0.0
        self._lcm_failure_error = ""

    def _cooldown_remaining_for_current_session(self) -> float:
        return self._cooldown_remaining() if self._cooldown_applies_to_current_session() else 0.0

    def _cooldown_remaining(self) -> float:
        return max(0.0, float(getattr(self, "_lcm_failure_cooldown_until", 0.0)) - time.monotonic())

    # -- host protocol -----------------------------------------------------------------

    def get_active_compression_failure_cooldown(self, *, refresh: bool = False) -> Optional[Dict[str, Any]]:
        del refresh  # nothing durable to refresh: the cooldown is process-local
        remaining = self._cooldown_remaining_for_current_session()
        if remaining <= 0:
            return None
        return {
            "cooldown_until": time.time() + remaining,
            "remaining_seconds": remaining,
            "error": self._lcm_failure_error,
        }

    def _automatic_compression_blocked(self, *, ignore_cooldown: bool = False) -> bool:
        if ignore_cooldown:
            return False
        return self._cooldown_remaining_for_current_session() > 0

    def should_compress(self, prompt_tokens: int = None) -> bool:  # type: ignore[override]
        if self._automatic_compression_blocked():
            return False
        return super().should_compress(prompt_tokens)  # type: ignore[misc]

    def should_compress_preflight(self, messages):  # type: ignore[override]
        # Upstream's preflight is also where every turn is ingested into the store; it must
        # always run. Only the *decision* is suppressed during a cooldown.
        wants = super().should_compress_preflight(messages)  # type: ignore[misc]
        if self._automatic_compression_blocked():
            return False
        return wants

    def should_compress_info(self, prompt_tokens: int = None):  # type: ignore[override]
        remaining = self._cooldown_remaining_for_current_session()
        if remaining > 0:
            return False, f"cooldown:{remaining:.0f}"
        return self.should_compress(prompt_tokens), None

    # -- compress wrapper --------------------------------------------------------------

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None,
                 focus_topic: Optional[str] = None, force: bool = False) -> List[Dict[str, Any]]:
        if force:
            self.clear_compression_failure_cooldown()
            guard = getattr(self, "_summary_spend_guard", None)
            if guard is not None and hasattr(guard, "clear"):
                guard.clear()
        self._last_leaf_summary_error = ""
        lock = self._compaction_lock_object()
        if not lock.try_acquire():
            # A compaction the host abandoned is still running on its worker thread: never
            # let two writers into the same session. The host retries on its next turn.
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "compaction already in progress on another worker"
            logger.warning("LCM compress() skipped: %s", self._last_compression_noop_reason)
            return messages
        try:
            result = super().compress(  # type: ignore[misc]
                messages, current_tokens=current_tokens, focus_topic=focus_topic, force=force,
            )
        except SummaryUnavailableError as exc:
            self._record_compression_failure(str(exc))
            self._last_compression_status = "cooldown"
            self._last_compression_noop_reason = f"summariser unavailable: {exc}"
            return messages
        finally:
            self._close_leaf_lookahead()
            lock.release()
        # A run whose leaf loop hit an unavailable summariser publishes whatever it did
        # (persisted passes, cleanup drops) but still arms the cooldown so the host stops
        # re-trying every turn.
        if getattr(self, "_last_leaf_summary_error", ""):
            self._record_compression_failure(self._last_leaf_summary_error)
            if self._last_compression_status == "noop":
                self._last_compression_status = "cooldown"
                self._last_compression_noop_reason = (
                    f"summariser unavailable: {self._last_leaf_summary_error}"
                )
        # Identity contract: a context the engine did not change is returned as the very
        # object the host passed in (the host treats a *new* list as "compressed").
        if result is not messages and isinstance(result, list) and result == messages:
            return messages
        return result

    def _compaction_lock_object(self):
        # fork: betterlcm — normally constructed in LCMEngine.__init__ (see H01). The lazy
        # branch remains only for objects that mix this in without that constructor; it is not
        # the ordinary path, so it cannot re-introduce the two-locks race for the engine.
        lock = getattr(self, "_compaction_lock", None)
        if lock is None:
            from .leaf_pipeline import CompactionLock
            lock = CompactionLock()
            self._compaction_lock = lock
        return lock

    def _close_leaf_lookahead(self) -> None:
        lookahead = getattr(self, "_leaf_lookahead", None)
        if lookahead is not None:
            self._leaf_lookahead = None
            try:
                lookahead.close()
            except Exception:  # pragma: no cover
                logger.debug("LCM leaf lookahead close failed", exc_info=True)

    # -- status ------------------------------------------------------------------------

    def compression_failure_status(self) -> Dict[str, Any]:
        remaining = self._cooldown_remaining_for_current_session()
        return {
            "cooldown_active": remaining > 0,
            "cooldown_remaining_seconds": round(remaining, 1),
            "cooldown_seconds": self._cooldown_seconds(),
            "failure_count": int(getattr(self, "_lcm_failure_count", 0)),
            "last_error": self._lcm_failure_error,
        }
