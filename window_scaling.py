"""Window-weighted tuning defaults (fork: better-hermeslcm).

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
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional

DEFAULT_SCALE_LOW_WINDOW = 262_144
DEFAULT_SCALE_HIGH_WINDOW = 1_000_000

# Sentinel for "same as the resolved context_threshold" (drain stop's low anchor).
THRESHOLD = object()
# Sentinel for "same as config.leaf_chunk_tokens" (sweep target's low anchor: upstream falls back to it).
LEAF_CHUNK = object()

# fork: better-hermeslcm — how much raw history one summariser call turns into one leaf node,
# as a FRACTION of the window. The same fraction at both anchors, so it slides with the
# window like every other weighted value instead of being a hardcoded token count.
#
# 0.04 gives 40,000 tokens at 1M, ~10,500 at 256k, ~5,000 at 128k: about 25 leaves per full
# window at any size. A fixed token count would have been wrong in both directions — 40k is a
# reasonable span of a 1M window and 31 % of a 128k one — and a value that is only correct at
# one window is exactly the thing this table exists to avoid.
LEAF_CHUNK_FRACTION = 0.04


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
    # Non-sweep drain stop. Fixed anchors: upstream stops as soon as it is under its own
    # threshold (0.35 default) -> 0.30 of the window at 1M. The resolver additionally clamps the
    # result to the resolved threshold, so an operator who lowers the threshold below the curve
    # still gets a stop point that is reachable. (Was: the *curved* threshold as a moving low
    # anchor, which made the drain stop rise to 0.44 mid-range before falling.)
    # fork: better-hermeslcm — the low anchor was THRESHOLD ("stop the moment we are back under it"),
    # which is upstream's rule for a loop that only ever takes one whole-backlog pass. With real
    # chunking that rule stops the drain after the first chunk, leaving the rest of the backlog
    # raw until the next turn crosses the threshold again — chunking on paper, one-shot in
    # practice. Both endpoints are now a genuine drain target: compact until the raw backlog is
    # under 0.30 of the window, then leave the rest verbatim.
    Anchor("drain_stop_fraction", "drain_stop_fraction", 0.30, 0.30, cast=float, unset=0.0),
    # ── Leaf chunking: NOT window-scaled, and that is the point ───────────────────────────────
    #
    # DO NOT "restore upstream's value" at the low anchor. This is the single most important
    # comment in this file.
    #
    # Chunk size is the GRANULARITY OF THE INDEX, not a tuning preference. One leaf is one
    # expandable unit: everything inside it is summarised together and comes back together.
    # Upstream (non-dynamic) hands the WHOLE backlog outside the fresh tail to one summariser
    # call, which produces exactly one summary of everything — that is a one-shot compaction,
    # the thing LCM exists to replace. This fork was originally written with the low anchor at
    # `1.0*W` "because that is upstream's behaviour", and a later commit added a gate to enforce
    # it even for oversized histories. Both were wrong: good LCM *is* chunking, and a fork whose
    # stated purpose is "no loss, no truncation" must not ship a degenerate index at its low
    # anchor.
    #
    # Two independent reasons the size must stay bounded, both of which hold at every window:
    #   1. Summarisation quality degrades with input length. A model given 500k tokens produces
    #      a topic list; given 40k it can still name decisions, identifiers, paths and errors —
    #      which is what makes a leaf an index into recoverable history rather than a gist.
    #   2. Expansion granularity. `lcm_expand(node_id=N)` returns that leaf's sources. A leaf
    #      over the whole backlog hands back hundreds of messages; a 40k leaf hands back a span
    #      the reader actually wanted.
    # Neither reason mentions the host's context window, so neither endpoint may either.
    #
    # It stays a FRACTION and slides with the window, like every other weighted value here —
    # the same fraction at both anchors, so the index has the same relative resolution
    # (~25 leaves per full window) whatever the model's window is. What changes with the window
    # is WHEN to compact (threshold), HOW MANY chunks one compaction may take (leaf_pass_cap),
    # how many are in flight (summary_concurrency) and HOW FAR to drain (drain_stop_fraction).
    Anchor("leaf_chunk_tokens", "leaf_chunk_fraction", LEAF_CHUNK_FRACTION, LEAF_CHUNK_FRACTION,
           low_is_fraction=True, high_is_fraction=True, cast=int, unset=0.0),
    # Pass cap: a wall-clock/spend SAFETY limit, never the thing that stops a drain. It was 1 at
    # the low anchor, which meant one chunk per compaction — so even with chunking switched on,
    # a backlog could never drain. Both endpoints are now generous enough that the real guards
    # (leaf_loop_max_seconds, the spend guard, drain_stop_fraction) are what actually stop the
    # loop.
    Anchor("leaf_pass_cap", "leaf_pass_cap", 16, 64, cast=int, unset=0),
    Anchor("leaf_loop_max_seconds", "leaf_loop_max_seconds", 120.0, 200.0, cast=float, unset=0.0),
    Anchor("summary_timeout_ms", "summary_timeout_ms", 60_000, 200_000, cast=int),
    Anchor("expansion_timeout_ms", "expansion_timeout_ms", 120_000, 200_000, cast=int),
    # ── The protected fresh tail: sized in TOKENS, at every window ────────────────────────────
    # How much of the recent conversation is never compacted and stays verbatim. Upstream sizes
    # it by message COUNT alone (32) with no token cap, so what it actually protects depends on
    # how long the messages happen to be — 32 one-line turns protect a few hundred tokens of a
    # 262,144-token window. The 1M design protects 0.15*W. Copying upstream's 32 at the low
    # anchor meant a 256k session kept ~4 % of its window verbatim where a 1M session keeps 15 %,
    # for no reason other than "that is upstream's number".
    #
    # The token fraction is therefore the same at both anchors and does the real work; the count
    # is a generous upper bound so a flood of tiny messages cannot make the tail unboundedly
    # long in MESSAGE terms. Selection takes the last `count` messages and then trims them to
    # the token cap, so the count can never ADD a message the cap excluded.
    Anchor("fresh_tail_count", "fresh_tail_count", 400, 400, cast=int),
    Anchor("fresh_tail_max_tokens", "fresh_tail_max_tokens", 0.15, 0.15,
           low_is_fraction=True, high_is_fraction=True, cast=int),
    # ── When to condense leaves into a parent ────────────────────────────────────────────────
    # Upstream has no token gate (0), so condensation runs purely on a COUNT rule: every Nth
    # leaf, whatever those leaves are worth. That was tolerable while a compaction produced one
    # whole-backlog leaf per call; with real chunking it merges a handful of small leaves almost
    # immediately, making the rendered frontier coarser long before there is any pressure to
    # make it coarser. The 1M design waits until the summary pile is worth 0.20*W. Same fraction
    # at both anchors, so the DAG gains depth at the same relative point whatever the window is.
    Anchor("condense_budget_tokens", "summary_budget_fraction", 0.20, 0.20,
           low_is_fraction=True, high_is_fraction=True, cast=int, unset=0.0),
    # Sweep-flag condensation target: upstream falls back to leaf_chunk_tokens (20k).
    Anchor("sweep_target_tokens", "summary_prefix_target_tokens", LEAF_CHUNK, 0.20,
           high_is_fraction=True, cast=int, unset=0),
    Anchor("incremental_max_depth", "incremental_max_depth", 3, 5, cast=int),
    # Summarise the next chunks on workers while the current one is persisted
    # (leaf_pipeline.LeafLookahead). Upstream is serial because upstream has one chunk; this
    # fork chunks at every window, so pinning the low anchor to 1 meant 256k did its chunks
    # strictly one after another for no reason. Concurrency is bounded by the number of pending
    # chunks anyway, so the same value at both anchors costs nothing when there is only one
    # chunk to do. Persistence stays sequential and chronological, so the published DAG is
    # identical whatever this is set to.
    Anchor("summary_concurrency", "summary_concurrency", 6, 6, cast=int, unset=0),
    # How many condensation groups one compress() may publish. Upstream does ONE pass of its
    # depth loop per call, which matched its one-leaf-per-call production rate. This fork can
    # publish up to `leaf_pass_cap` leaves per call, so keeping the upstream schedule let leaves
    # accumulate faster than they were merged (audit D #3). The rule is `leaf_pass_cap / fanin`
    # — enough groups to absorb one compaction's worth of leaves — which is 16 groups x fanin 4
    # at the 64-leaf cap, and 4 at the 16-leaf cap. The low anchor was 1 ("upstream exactly"),
    # which is only correct for a fork that produces one leaf per call, and this one no longer
    # does at any window.
    Anchor("condense_group_cap", "condense_group_cap", 4, 16, cast=int, unset=0),
    # Summariser spend guard: a sliding-window limiter counting CALLS. Upstream's 24 is
    # calibrated for one whole-backlog call per compaction, so once the fork chunks, the same
    # amount of work costs many small calls and the guard trips on its own design — a full drain
    # could not finish, compaction stopped, and the backlog stayed raw. The unit is wrong for a
    # chunked engine, so the anchors are set from what one drain actually needs:
    # `4 * (leaf_pass_cap + condense_group_cap)`, i.e. four full drains inside one window.
    #
    # Measured in TOKENS — which is what "spend" means — this is more conservative than upstream
    # at both anchors: 80 x 10.5k = 840k per window at 256k against upstream's 24 x ~85k = 2.0M,
    # and 320 x 40k = 12.8M at 1M against upstream's 24 x ~700k = 16.8M.
    Anchor("summary_spend_max_calls", "summary_spend_max_calls", 80, 320, cast=int),
    Anchor("summary_circuit_breaker_failure_threshold",
           "summary_circuit_breaker_failure_threshold", 2, 4, cast=int),
    Anchor("l2_budget_ratio", "l2_budget_ratio", 0.50, 0.80, cast=float),
    # Pre-summariser per-message cap in chars. Upstream cut every message to 3000 chars
    # (head 2000 + tail 800), unmarked and unrecoverable. That is TRUNCATION, which this fork
    # removes at EVERY window — the curve carries tuning values, not loss. Both endpoints are
    # therefore the whole window (4 chars/token * W): ~1,048,576 chars at 256k, 4,000,000 at
    # 1M. A message can never be larger than the window it arrived in, so the elision below
    # never fires in practice; marked_loss.elide_text stays as the guard for a pathological
    # input, and it marks what it cuts.
    Anchor("serialize_message_max_chars", "serialize_message_max_chars", 4.0, 4.0,
           low_is_fraction=True, high_is_fraction=True, cast=int, unset=0),
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


def _anchor_value(raw: Any, is_fraction: bool, anchor_window: int) -> float:
    """Resolve one endpoint.

    fork: better-hermeslcm — a fraction endpoint is resolved against **its own** anchor window
    (``scale_low_window`` for ``low``, ``scale_high_window`` for ``high``), never against the
    current window. Resolving both endpoints against the current window made the curve
    non-monotonic: ``leaf_chunk`` (1.0*W -> 0.04*W) produced 262k tokens/call at 256k, 345k at
    512k and 40k at 1M, i.e. intermediate windows sent larger summariser requests than either
    endpoint. Fixed endpoints keep the interpolation linear and monotone, which is what
    "slides smoothly from the 256k value to the 1M value" means.
    """
    return float(raw) * anchor_window if is_fraction else float(raw)


def interpolate(anchor: Anchor, context_length: int, t: float, *,
                threshold_value: Optional[float] = None,
                leaf_chunk_tokens: Optional[int] = None,
                low_window: int = DEFAULT_SCALE_LOW_WINDOW,
                high_window: int = DEFAULT_SCALE_HIGH_WINDOW) -> Any:
    """Curve value for ``anchor`` at ``t``, from fixed endpoints.

    ``context_length`` is no longer used to resolve the endpoints; it is kept in the signature
    because callers pass it and because a future anchor may legitimately need it.
    """
    del context_length
    low = anchor.low
    if low is THRESHOLD:
        low = threshold_value if threshold_value is not None else 0.0
        low_val = float(low)          # already a fraction of W
    elif low is LEAF_CHUNK:
        low_val = float(leaf_chunk_tokens if leaf_chunk_tokens is not None else 20_000)
    else:
        low_val = _anchor_value(low, anchor.low_is_fraction, low_window)
    high_val = _anchor_value(anchor.high, anchor.high_is_fraction, high_window)
    value = low_val + t * (high_val - low_val)
    if anchor.cast is int:
        return int(round(value))
    return anchor.cast(value)


@lru_cache(maxsize=1)
def _env_keys_by_field() -> Mapping[str, str]:
    """fork: better-hermeslcm — the env-spec list is immutable; index it once.

    Every resolved setting used to walk the whole specification list, and every explicitness
    check walked the dataclass fields, on every resolve (audit verify-3 O9).
    """
    try:
        from .config import ENV_FIELD_SPECS  # type: ignore
    except ImportError:  # pragma: no cover - package-less import (tests / standalone)
        from config import ENV_FIELD_SPECS  # type: ignore
    return {spec.name: spec.env_key for spec in ENV_FIELD_SPECS}


def _env_key_for(field: str) -> Optional[str]:
    return _env_keys_by_field().get(field)


_NO_DEFAULT = object()


@lru_cache(maxsize=8)
def _field_defaults_for(config_type: type) -> Mapping[str, Any]:
    """fork: better-hermeslcm — dataclass defaults for one config type, computed once (verify-3 O9)."""
    defaults: dict[str, Any] = {}
    try:
        import dataclasses
        for field in dataclasses.fields(config_type):
            if field.default is not dataclasses.MISSING:
                defaults[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                defaults[field.name] = field.default_factory()  # type: ignore[misc]
    except Exception:  # pragma: no cover - not a dataclass
        return {}
    return defaults


def _field_default(config: Any, field_name: str) -> Any:
    """The dataclass default for ``field_name``, or ``_NO_DEFAULT``."""
    return _field_defaults_for(type(config)).get(field_name, _NO_DEFAULT)


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
    # A config built directly (LCMConfig(context_threshold=0.9), presets, tests) carries no
    # tracked source. If its value differs from the FIELD'S OWN DEFAULT it was set on purpose.
    # (This must compare against the dataclass default, not against ``anchor.low``: the two
    # differ wherever upstream's default is a sentinel — ``fresh_tail_max_tokens`` defaults to
    # 0 meaning "no cap" while the curve's low endpoint is a cap that cannot bind.)
    default = _field_default(config, anchor.field)
    if default is not _NO_DEFAULT and value is not None:
        try:
            if anchor.cast(value) != anchor.cast(default):
                return True, "manual"
        except (TypeError, ValueError):
            pass
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
            elif anchor.low is LEAF_CHUNK:
                value = int(getattr(config, "leaf_chunk_tokens", 20_000) or 20_000)
            elif anchor.low_is_fraction:
                # No window known: upstream behaviour verbatim, which for a sentinel field is
                # its dataclass default (``fresh_tail_max_tokens = 0`` = no cap), not the
                # curve's low endpoint.
                default = _field_default(config, anchor.field)
                value = (anchor.cast(default) if default is not _NO_DEFAULT
                         else anchor.cast(round(float(anchor.low) * low_w)))
            else:
                value = anchor.cast(anchor.low)
            out[anchor.name] = Resolved(anchor.name, value, "upstream(no window)", 0.0)
            continue
        value = interpolate(anchor, W, t, threshold_value=threshold,
                            leaf_chunk_tokens=int(getattr(config, "leaf_chunk_tokens", 20_000) or 20_000),
                            low_window=low_w, high_window=high_w)
        if anchor.name == "drain_stop_fraction":
            # never ask the loop to drain past a point the threshold would not have reached
            value = min(float(value), float(threshold))
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
