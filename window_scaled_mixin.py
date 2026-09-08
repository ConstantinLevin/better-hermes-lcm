"""Engine-side integration of the weighting curve (fork: betterlcm).

``WindowScaledSettingsMixin`` is mixed into ``LCMEngine``. It owns the ``effective_*``
attributes that consumers read instead of raw ``self._config`` values, and re-resolves them
whenever the engine learns (or loses) its context window. Three hook points in upstream
code, each a single call:

* ``LCMEngine.__init__``          -> ``_init_window_scaled_settings()``  (upstream values, no window yet)
* ``_set_context_length`` (both return paths) -> ``_resolve_window_scaled_settings()``
* ``tools.py`` status dict         -> ``window_scaling_status()``

The threshold is special: upstream derives ``context_threshold`` in ``_runtime_context_threshold``
and reports the source ``manual_or_default`` when nothing configured it. Only in that case does
the curve replace it; every configured source (env, ``lcm.context_threshold``,
``compression.threshold``, the Codex autoraise) is left exactly as upstream computed it.
"""
from __future__ import annotations

from typing import Any, Dict

try:  # package import (installed plugin / tests register ``hermes_lcm``)
    from .window_scaling import (
        DEFAULT_SCALE_HIGH_WINDOW,
        DEFAULT_SCALE_LOW_WINDOW,
        WINDOW_SCALED_DEFAULTS,
        Resolved,
        resolve_window_scaled,
        status_payload,
    )
except ImportError:  # pragma: no cover - standalone import
    from window_scaling import (  # type: ignore
        DEFAULT_SCALE_HIGH_WINDOW,
        DEFAULT_SCALE_LOW_WINDOW,
        WINDOW_SCALED_DEFAULTS,
        Resolved,
        resolve_window_scaled,
        status_payload,
    )

_DEFAULT_THRESHOLD_SOURCE = "manual_or_default"


class WindowScaledSettingsMixin:
    """Resolve and expose window-weighted settings on the engine."""

    _window_scaled: Dict[str, Resolved]

    # -- lifecycle -------------------------------------------------------------------------

    def _init_window_scaled_settings(self) -> None:
        """Seed every ``effective_*`` with upstream's value: no window is known yet."""
        self._window_scaled = {}
        self._apply_window_scaled(resolve_window_scaled(self._config, 0))

    def _resolve_window_scaled_settings(self) -> None:
        """Re-resolve for the current (capped) ``self.context_length``.

        Called at the end of BOTH ``_set_context_length`` return paths, so clearing the window
        resets every value to upstream's rather than leaving a stale curve behind.
        """
        resolved = resolve_window_scaled(self._config, int(self.context_length or 0))
        self._apply_window_scaled(resolved)
        self._apply_curved_threshold(resolved)
        self._retune_summary_guards()

    def _retune_summary_guards(self) -> None:
        """Push the curved limits into the guard objects built in ``__init__``."""
        guard = getattr(self, "_summary_spend_guard", None)
        if guard is not None and hasattr(guard, "max_calls"):
            guard.max_calls = int(self.effective_summary_spend_max_calls)
        breaker = getattr(self, "_summary_circuit_breaker", None)
        if breaker is not None and hasattr(breaker, "failure_threshold"):
            breaker.failure_threshold = int(self.effective_summary_circuit_breaker_failure_threshold)

    # -- internals -------------------------------------------------------------------------

    def _apply_window_scaled(self, resolved: Dict[str, Resolved]) -> None:
        self._window_scaled = dict(resolved)
        for anchor in WINDOW_SCALED_DEFAULTS:
            entry = resolved.get(anchor.name)
            if entry is not None:
                setattr(self, f"effective_{anchor.name}", entry.value)

    def _apply_curved_threshold(self, resolved: Dict[str, Resolved]) -> None:
        """Let the curve supply the threshold *default* only.

        ``_runtime_context_threshold`` has already run and set ``context_threshold`` /
        ``_context_threshold_source``. When that source is ``manual_or_default`` (nothing
        configured it) the curve's value replaces it and ``threshold_tokens`` is recomputed
        exactly the way ``_set_context_length`` does. Any configured or autoraised threshold
        is left untouched.
        """
        entry = resolved.get("context_threshold")
        if entry is None:
            return
        source = getattr(self, "_context_threshold_source", _DEFAULT_THRESHOLD_SOURCE)
        if source != _DEFAULT_THRESHOLD_SOURCE:
            return
        if not entry.source.startswith("curve@"):
            return
        self.context_threshold = float(entry.value)
        self.threshold_percent = self.context_threshold
        self._context_threshold_source = entry.source
        window = int(self.context_length or 0)
        if window > 0:
            self.threshold_tokens = self._effective_threshold_tokens(
                int(window * self.context_threshold)
            )

    # -- status ----------------------------------------------------------------------------

    def window_scaling_status(self) -> Dict[str, Any]:
        low = int(getattr(self._config, "scale_low_window", 0) or DEFAULT_SCALE_LOW_WINDOW)
        high = int(getattr(self._config, "scale_high_window", 0) or DEFAULT_SCALE_HIGH_WINDOW)
        return status_payload(getattr(self, "_window_scaled", {}), int(self.context_length or 0), low, high)
