"""Window-weighted tuning defaults (fork: betterlcm).

Every LCM size that is a *preference* about how to spend a context window is resolved as a
linear function of ``context_length`` (``W``) between two anchors:

* ``low``  -- upstream's default, pinned at ``W_low``  (default 262,144 tokens)
* ``high`` -- the large-window design,  pinned at ``W_high`` (default 1,000,000 tokens)

``t = clamp((W - W_low) / (W_high - W_low), 0, 1)`` and ``value = low + t * (high - low)``.
Below ``W_low`` (or when no window is known) the value is upstream's; above ``W_high`` it is
the large-window value; in between it slides. An explicit operator override of a setting
always wins over its curve -- the curve only replaces the *default*.

Anchors that are fractions of the window are marked ``*_is_fraction``; they are converted to
tokens (or chars) for the given ``W`` before interpolation, so e.g. a leaf chunk anchored at
``1.0 * W`` (upstream: the whole backlog in one node) slides toward ``0.04 * W`` (40k at 1M).

This module is pure: no engine state, no I/O beyond reading ``os.environ`` to detect explicit
overrides. The engine calls :func:`resolve_window_scaled` from ``_set_context_length`` and
stores the results as ``effective_*`` attributes; consumers read those.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

DEFAULT_SCALE_LOW_WINDOW = 262_144
DEFAULT_SCALE_HIGH_WINDOW = 1_000_000

# Sentinel for "same as the resolved context_threshold" (drain stop's low anchor).
THRESHOLD = object()


@dataclass(frozen=True)
class Anchor:
    """One weighted setting.

    ``field`` is the ``LCMConfig`` attribute that carries an explicit operator override.
    For upstream fields the attribute's default *is* ``low`` and explicitness is detected via
    the field's env var / tracked source. For fork-added fields ``unset`` is the sentinel
    meaning "no override" (e.g. ``0``), and any other value is explicit.
    """

    name: str                      # effective_<name> on the engine
    field: str                     # LCMConfig attribute holding an explicit override
    low: Any                       # value at W_low (upstream default) or THRESHOLD
    high: Any                      # value at W_high
    low_is_fraction: bool = False  # low is a fraction of W
    high_is_fraction: bool = False # high is a fraction of W
    cast: type = int
    unset: Optional[Any] = None    # sentinel for fork-added fields; None => upstream field


# The single source of truth for the curve. Keep in sync with docs/fork-design.md.
WINDOW_SCALED_DEFAULTS: tuple[Anchor, ...] = (
    Anchor("context_threshold", "context_threshold", 0.35, 0.80, cast=float),
    # Non-sweep drain stop: upstream stops the instant it is under the threshold; at 1M drain
    # to 0.30*W. Expressed as a fraction of W.
    Anchor("drain_stop_fraction", "drain_stop_fraction", THRESHOLD, 0.30, cast=float, unset=0.0),
    # Leaf chunk size: upstream (non-dynamic) summarises the WHOLE backlog outside the tail in
    # one node -> anchor 1.0*W; at 1M 40k chunks. Explicit dynamic_leaf_chunk_enabled keeps
    # upstream's doubling behaviour instead (handled by the consumer).
    Anchor("leaf_chunk_tokens", "leaf_chunk_fraction", 1.0, 0.04,
           low_is_fraction=True, high_is_fraction=True, cast=int, unset=0.0),
    Anchor("leaf_pass_cap", "leaf_pass_cap", 1, 64, cast=int, unset=0),
    Anchor("leaf_loop_max_seconds", "leaf_loop_max_seconds", 120.0, 200.0, cast=float, unset=0.0),
    Anchor("summary_timeout_ms", "summary_timeout_ms", 60_000, 200_000, cast=int),
    Anchor("expansion_timeout_ms", "expansion_timeout_ms", 120_000, 200_000, cast=int),
    Anchor("fresh_tail_count", "fresh_tail_count", 32, 400, cast=int),
    Anchor("fresh_tail_max_tokens", "fresh_tail_max_tokens", 0, 0.15, high_is_fraction=True, cast=int),
    # Condensation trigger budget: upstream has no token gate (0); at 1M condense only once
    # the summary pile exceeds 0.20*W.
    Anchor("condense_budget_tokens", "summary_budget_fraction", 0, 0.20,
           high_is_fraction=True, cast=int, unset=0.0),
    # Sweep-flag condensation target: upstream falls back to leaf_chunk_tokens (20k).
    Anchor("sweep_target_tokens", "summary_prefix_target_tokens", 20_000, 0.20,
           high_is_fraction=True, cast=int, unset=0),
    Anchor("incremental_max_depth", "incremental_max_depth", 3, 5, cast=int),
    Anchor("summary_concurrency", "summary_concurrency", 1, 6, cast=int, unset=0),
    Anchor("summary_spend_max_calls", "summary_spend_max_calls", 24, 120, cast=int),
    Anchor("summary_circuit_breaker_failure_threshold",
           "summary_circuit_breaker_failure_threshold", 2, 4, cast=int),
    Anchor("l2_budget_ratio", "l2_budget_ratio", 0.50, 0.80, cast=float),
    # Pre-summariser per-message cap in chars: upstream 3000 (head 2000 + tail 800); at 1M
    # effectively the whole message (4 chars/token * W).
    Anchor("serialize_message_max_chars", "serialize_message_max_chars", 3000, 4.0,
           high_is_fraction=True, cast=int, unset=0),
    Anchor("stub_threshold_tokens", "large_output_active_replay_stub_threshold_tokens",
           25_000, 100_000, cast=int),
    Anchor("expansion_context_tokens", "expansion_context_tokens", 32_000, 125_000, cast=int),
    Anchor("expand_page_tokens", "expand_page_tokens", 4_000, 32_000, cast=int, unset=0),
    Anchor("tool_response_char_scale", "tool_response_char_scale", 1.0, 4.0, cast=float, unset=0.0),
    Anchor("sqlite_cache_kib", "sqlite_cache_kib", 2_048, 65_536, cast=int, unset=0),
    Anchor("token_cache_size", "token_cache_size", 2_048, 8_192, cast=int, unset=0),
)

ANCHORS_BY_NAME: Dict[str, Anchor] = {a.name: a for a in WINDOW_SCALED_DEFAULTS}


@dataclass(frozen=True)
class Resolved:
    name: str
    value: Any
    source: str   # "curve@t=0.63" | "env" | "config_yaml:..." | "explicit" | "upstream(no window)"
    t: float


def curve_t(context_length: int, low_window: int = DEFAULT_SCALE_LOW_WINDOW,
            high_window: int = DEFAULT_SCALE_HIGH_WINDOW) -> float:
    """Position of ``context_length`` between the anchors, clamped to [0, 1]."""
    try:
        w = int(context_length)
    except (TypeError, ValueError):
        return 0.0
    lo, hi = int(low_window), int(high_window)
    if w <= lo or hi <= lo:
        return 0.0
    if w >= hi:
        return 1.0
    return (w - lo) / float(hi - lo)


def _anchor_value(raw: Any, is_fraction: bool, context_length: int) -> float:
    return float(raw) * context_length if is_fraction else float(raw)


def interpolate(anchor: Anchor, context_length: int, t: float, *,
                threshold_value: Optional[float] = None) -> Any:
    """Curve value for ``anchor`` at ``t``. ``threshold_value`` resolves the THRESHOLD sentinel."""
    low = anchor.low
    if low is THRESHOLD:
        low = threshold_value if threshold_value is not None else 0.0
        low_val = float(low)          # already a fraction of W
    else:
        low_val = _anchor_value(low, anchor.low_is_fraction, context_length)
    high_val = _anchor_value(anchor.high, anchor.high_is_fraction, context_length)
    value = low_val + t * (high_val - low_val)
    if anchor.cast is int:
        return int(round(value))
    return anchor.cast(value)


def _env_key_for(field: str) -> Optional[str]:
    try:
        from .config import ENV_FIELD_SPECS  # type: ignore
    except ImportError:  # pragma: no cover - package-less import (tests / standalone)
        from config import ENV_FIELD_SPECS  # type: ignore
    for spec in ENV_FIELD_SPECS:
        if spec.name == field:
            return spec.env_key
    return None


def explicit_override(config: Any, anchor: Anchor,
                      env: Optional[Mapping[str, str]] = None) -> tuple[bool, str]:
    """Return ``(is_explicit, source)`` for the anchor's config field."""
    environ = os.environ if env is None else env
    value = getattr(config, anchor.field, None)
    if anchor.unset is not None:
        return (value is not None and value != anchor.unset), "explicit"
    sources = getattr(config, "config_sources", None) or {}
    tracked = sources.get(anchor.field)
    if tracked and tracked != "default":
        return True, str(tracked)
    key = _env_key_for(anchor.field)
    if key and key in environ:
        return True, "env"
    return False, "default"


def resolve_window_scaled(config: Any, context_length: int,
                          env: Optional[Mapping[str, str]] = None) -> Dict[str, Resolved]:
    """Resolve every anchor for ``context_length``.

    Explicit overrides win. With no window known (``context_length <= 0``) every non-explicit
    setting is upstream's ``low`` value, so an engine that has not learned its window yet
    behaves exactly like upstream.
    """
    low_w = int(getattr(config, "scale_low_window", 0) or DEFAULT_SCALE_LOW_WINDOW)
    high_w = int(getattr(config, "scale_high_window", 0) or DEFAULT_SCALE_HIGH_WINDOW)
    W = int(context_length or 0)
    t = curve_t(W, low_w, high_w) if W > 0 else 0.0
    out: Dict[str, Resolved] = {}

    # context_threshold first: the drain stop's low anchor depends on it.
    thr_anchor = ANCHORS_BY_NAME["context_threshold"]
    explicit, source = explicit_override(config, thr_anchor, env)
    if explicit:
        threshold = float(getattr(config, thr_anchor.field))
        out[thr_anchor.name] = Resolved(thr_anchor.name, threshold, source, t)
    else:
        threshold = interpolate(thr_anchor, max(W, 1), t)
        out[thr_anchor.name] = Resolved(thr_anchor.name, threshold,
                                        f"curve@t={t:.2f}" if W > 0 else "upstream(no window)", t)

    for anchor in WINDOW_SCALED_DEFAULTS:
        if anchor.name == "context_threshold":
            continue
        explicit, source = explicit_override(config, anchor, env)
        if explicit:
            raw = getattr(config, anchor.field)
            # Fraction-style fork fields are stored as fractions of W.
            if anchor.unset is not None and isinstance(anchor.unset, float) and anchor.name in (
                "leaf_chunk_tokens", "condense_budget_tokens", "drain_stop_fraction",
            ):
                value = raw if anchor.name == "drain_stop_fraction" else anchor.cast(round(float(raw) * max(W, 1)))
            else:
                value = raw
            out[anchor.name] = Resolved(anchor.name, value, source, t)
            continue
        if W <= 0:
            # No window: upstream behaviour. Fraction lows without a window resolve to 0.
            if anchor.low is THRESHOLD:
                value = threshold
            elif anchor.low_is_fraction:
                value = 0
            else:
                value = anchor.cast(anchor.low)
            out[anchor.name] = Resolved(anchor.name, value, "upstream(no window)", 0.0)
            continue
        value = interpolate(anchor, W, t, threshold_value=threshold)
        out[anchor.name] = Resolved(anchor.name, value, f"curve@t={t:.2f}", t)
    return out


def status_payload(resolved: Mapping[str, Resolved], context_length: int,
                   low_window: int, high_window: int) -> Dict[str, Any]:
    """Compact, JSON-safe rendering for lcm_status / lcm_doctor."""
    return {
        "context_length": int(context_length or 0),
        "scale_low_window": int(low_window),
        "scale_high_window": int(high_window),
        "t": round(curve_t(context_length or 0, low_window, high_window), 4) if context_length else 0.0,
        "settings": {
            name: {"value": r.value, "source": r.source}
            for name, r in sorted(resolved.items())
        },
    }
