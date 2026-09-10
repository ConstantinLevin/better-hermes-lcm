"""Engine-side integration of the weighting curve (fork: better-hermes-lcm).

``WindowScaledSettingsMixin`` is mixed into ``LCMEngine``. It owns the ``effective_*``
attributes that consumers read instead of raw ``self._config`` values, and re-resolves them
whenever the engine learns (or loses) its context window. Three hook points in upstream
code, each a single call:

* ``LCMEngine.__init__``          -> ``_init_window_scaled_settings()``  (upstream values, no window yet)
* ``_set_context_length`` (both return paths) -> ``_resolve_window_scaled_settings()``
* ``tools.py`` status dict         -> ``window_scaling_status()``

On an upstream merge: a NEW upstream read of any field the curve owns (``ANCHORS_BY_NAME``) must
be switched to ``self.effective_<name>``. It arrives without a textual conflict and silently
un-curves that value.

The threshold is special: upstream derives ``context_threshold`` in ``_runtime_context_threshold``
and reports the source ``manual_or_default`` when nothing configured it. Only in that case does
the curve replace it; every configured source (env, ``lcm.context_threshold``,
``compression.threshold``, the Codex autoraise) is left exactly as upstream computed it.
"""
from __future__ import annotations

from typing import Any, Dict

try:  # package import (installed plugin / tests register ``hermes_lcm``)
    from .window_scaling import (
        ANCHORS_BY_NAME,
        DEFAULT_SCALE_HIGH_WINDOW,
        DEFAULT_SCALE_LOW_WINDOW,
        THRESHOLD,
        Resolved,
        explicit_override,
        resolve_window_scaled,
        status_payload,
    )
except ImportError:  # pragma: no cover - standalone import
    from window_scaling import (  # type: ignore
        ANCHORS_BY_NAME,
        DEFAULT_SCALE_HIGH_WINDOW,
        DEFAULT_SCALE_LOW_WINDOW,
        THRESHOLD,
        Resolved,
        explicit_override,
        resolve_window_scaled,
        status_payload,
    )

_DEFAULT_THRESHOLD_SOURCE = "manual_or_default"


class WindowScaledSettingsMixin:
    """Resolve and expose window-weighted settings on the engine.

    ``effective_<name>`` are *properties*: an explicit value on ``self._config`` (set at
    construction, by a preset, or mutated at runtime by a tool/test) is honoured immediately;
    otherwise the curve value cached at the last ``_set_context_length`` is returned.
    """

    _window_scaled: Dict[str, Resolved]

    # -- lifecycle -------------------------------------------------------------------------

    def _init_window_scaled_settings(self) -> None:
        """Seed the curve cache with upstream's values: no window is known yet."""
        self._window_scaled = dict(resolve_window_scaled(self._config, 0))

    def _resolve_window_scaled_settings(self) -> None:
        """Re-resolve for the current (capped) ``self.context_length``.

        Called at the end of BOTH ``_set_context_length`` return paths, so clearing the window
        resets every value to upstream's rather than leaving a stale curve behind.
        """
        resolved = resolve_window_scaled(self._config, int(self.context_length or 0))
        self._window_scaled = dict(resolved)
        self._apply_curved_threshold(resolved)
        self._retune_summary_guards()
        self._apply_runtime_caches()

    # -- lookup ----------------------------------------------------------------------------

    def _effective(self, name: str) -> Any:
        value = self._effective_raw(name)
        if name == "fresh_tail_max_tokens" and self._is_curved(name):
            # fork: better-hermes-lcm — the curve interpolates this cap from "the whole window" (which
            # cannot bind, i.e. upstream's `0 = disabled`) down to 0.15*W at 1M. While it still
            # cannot bind, report upstream's literal 0. A curved cap must also never *add* a
            # message to a tail the count limit excluded. An OPERATOR's explicit cap is never
            # normalised away — it applies exactly as configured.
            window = int(getattr(self, "context_length", 0) or 0)
            if window <= 0 or int(value or 0) >= window:
                return 0
            if int(self._effective_raw("fresh_tail_count") or 0) <= 0:
                return 0
        return value

    def _is_curved(self, name: str) -> bool:
        """True when this setting's value came from the curve rather than an operator."""
        entry = (getattr(self, "_window_scaled", None) or {}).get(name)
        return bool(entry is not None and str(entry.source).startswith("curve@"))

    def _effective_raw(self, name: str) -> Any:
        anchor = ANCHORS_BY_NAME[name]
        config = self._config
        explicit, _source = explicit_override(config, anchor)
        if explicit:
            raw = getattr(config, anchor.field)
            if anchor.unset is not None and isinstance(anchor.unset, float) and name in (
                "leaf_chunk_tokens", "condense_budget_tokens",
            ):
                return anchor.cast(round(float(raw) * max(int(self.context_length or 0), 1)))
            return raw
        cache = getattr(self, "_window_scaled", None)
        if not cache:
            cache = dict(resolve_window_scaled(config, int(getattr(self, "context_length", 0) or 0)))
            self._window_scaled = cache
        entry = cache.get(name)
        if entry is None:
            return getattr(config, anchor.field, anchor.low if anchor.low is not THRESHOLD else None)
        return entry.value

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
        # fork: better-hermes-lcm — the RESOLVER is the single authority on "did anything configure
        # this?". An earlier version also required the engine's own
        # ``_context_threshold_source`` to equal "manual_or_default", but ``LCMConfig.from_env``
        # records "default" for an unconfigured threshold — so on a clean install the guard
        # rejected the curve, the engine compacted at 0.35 (350k on a 1M model) while
        # ``lcm_status`` reported the curve's 0.80. Two predicates for one question is how that
        # happened; there is now one.
        if not entry.source.startswith("curve@"):
            return
        if getattr(self, "_context_threshold_autoraised", None):
            return  # a route-specific autoraise is a deliberate runtime decision, not a default
        self.context_threshold = float(entry.value)
        self.threshold_percent = self.context_threshold
        self._context_threshold_source = entry.source
        window = int(self.context_length or 0)
        if window > 0:
            self.threshold_tokens = self._effective_threshold_tokens(
                int(window * self.context_threshold)
            )

    def _apply_runtime_caches(self) -> None:
        """SQLite page cache (KiB) on the store/DAG connections and the token LRU size."""
        cache_kib = int(self._effective("sqlite_cache_kib") or 0)
        if cache_kib > 0:
            for owner in ("_store", "_dag"):
                conn = getattr(getattr(self, owner, None), "connection", None)
                if conn is None:
                    continue
                try:
                    conn.execute(f"PRAGMA cache_size=-{cache_kib}")
                except Exception:  # pragma: no cover - a closed connection is not an error here
                    pass
        try:
            from .tokens import set_token_cache_size
        except ImportError:  # pragma: no cover
            from tokens import set_token_cache_size  # type: ignore
        try:
            # fork: the memo is process-global while engines are not; identify the requester
            # so a second engine's smaller window cannot shrink (and empty) the shared cache
            set_token_cache_size(int(self._effective("token_cache_size") or 0), owner=self)
        except Exception:  # pragma: no cover
            pass

    def _retune_summary_guards(self) -> None:
        """Push the curved limits into the guard objects built in ``__init__``."""
        guard = getattr(self, "_summary_spend_guard", None)
        if guard is not None and hasattr(guard, "max_calls"):
            guard.max_calls = int(self.effective_summary_spend_max_calls)
        breaker = getattr(self, "_summary_circuit_breaker", None)
        if breaker is not None and hasattr(breaker, "failure_threshold"):
            breaker.failure_threshold = int(self.effective_summary_circuit_breaker_failure_threshold)

    # -- status ----------------------------------------------------------------------------

    def window_scaling_status(self) -> Dict[str, Any]:
        low = int(getattr(self._config, "scale_low_window", 0) or DEFAULT_SCALE_LOW_WINDOW)
        high = int(getattr(self._config, "scale_high_window", 0) or DEFAULT_SCALE_HIGH_WINDOW)
        # Report the values consumers actually see (explicit overrides included).
        live = {
            name: Resolved(name, self._effective(name),
                           explicit_override(self._config, ANCHORS_BY_NAME[name])[1]
                           if explicit_override(self._config, ANCHORS_BY_NAME[name])[0]
                           else (getattr(self, "_window_scaled", {}).get(name).source
                                 if getattr(self, "_window_scaled", {}).get(name) else "default"),
                           0.0)
            for name in ANCHORS_BY_NAME
        }
        return status_payload(live, int(self.context_length or 0), low, high)


def _install_effective_properties() -> None:
    for _name in ANCHORS_BY_NAME:
        def _getter(self, _n=_name):
            return self._effective(_n)
        setattr(WindowScaledSettingsMixin, f"effective_{_name}", property(_getter))


_install_effective_properties()
