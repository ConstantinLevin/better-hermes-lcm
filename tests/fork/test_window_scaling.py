"""Step 1 — the weighting curve. At 256k every value is upstream's; at 1M the design's."""
import dataclasses

import pytest

from hermes_lcm import window_scaling as ws
from hermes_lcm.config import LCMConfig

K = 1024
W256, W1M = 262_144, 1_000_000

# What the curve must resolve to at the LOW anchor.
#
# The old name for this was UPSTREAM, and it asserted that every value at 256k is upstream's.
# That was the mistake this fork spent a long time shipping: upstream is the FLOOR ("never worse
# than upstream"), never the target. A value is upstream's at 256k only when it is a genuine
# tuning preference — a cost/latency/headroom tradeoff where upstream's choice is as good as any
# other. Where a value decides how much is LOST or how coarse the index is, upstream's number is
# not adopted at any window.
LOW_ANCHOR_TUNING = {  # preferences: upstream's own values, because they are fine
    "context_threshold": 0.35,          # when to compact: a real cost/verbatim tradeoff
    "leaf_loop_max_seconds": 120.0, "summary_timeout_ms": 60_000,
    "expansion_timeout_ms": 120_000,
    "sweep_target_tokens": 20_000, "incremental_max_depth": 3,
    "summary_circuit_breaker_failure_threshold": 2,
    "l2_budget_ratio": 0.50, "stub_threshold_tokens": 25_000,
    "expansion_context_tokens": 32_000, "expand_page_tokens": 4_000,
    "tool_response_char_scale": 1.0, "sqlite_cache_kib": 2_048, "token_cache_size": 2_048,
}
LOW_ANCHOR_NOT_UPSTREAM = {  # quality/loss: deliberately NOT upstream's number
    # upstream: the whole backlog in one summariser call — one node standing for everything,
    # which is a one-shot compaction. 4 % of the window at every anchor instead.
    "leaf_chunk_tokens": round(W256 * 0.04),
    # upstream: 1 pass per compaction, which cannot drain a chunked backlog.
    "leaf_pass_cap": 16,
    # upstream: stop the moment we are under the threshold — one chunk and done.
    "drain_stop_fraction": 0.30,
    # upstream: 32 messages and no token cap, so what stays verbatim depends on how long the
    # messages happen to be. The same fraction of the window as at 1M instead.
    "fresh_tail_count": 400, "fresh_tail_max_tokens": round(W256 * 0.15),
    # upstream: no token gate, so condensation runs on a count rule and coarsens the frontier
    # long before there is pressure to.
    "condense_budget_tokens": round(W256 * 0.20),
    "condense_group_cap": 4,
    # upstream: serial, because upstream has one chunk. This fork chunks at every window.
    "summary_concurrency": 6,
    # upstream: 24 calls per window, calibrated for ONE whole-backlog call per compaction. A
    # chunked engine spends many small calls for the same work, so upstream's number stopped a
    # drain mid-way. Set from what one drain needs; in TOKENS it is below upstream's.
    "summary_spend_max_calls": 80,
    # upstream: 3000 chars per message (head 2000 + tail 800), unmarked. That is truncation.
    "serialize_message_max_chars": 4 * W256,
}
LOW_ANCHOR = {**LOW_ANCHOR_TUNING, **LOW_ANCHOR_NOT_UPSTREAM}
DESIGN_1M = {
    "context_threshold": 0.80, "drain_stop_fraction": 0.30, "leaf_chunk_tokens": 40_000,
    "leaf_pass_cap": 64, "leaf_loop_max_seconds": 200.0, "summary_timeout_ms": 200_000,
    "fresh_tail_count": 400, "fresh_tail_max_tokens": 150_000, "condense_budget_tokens": 200_000,
    "sweep_target_tokens": 200_000, "incremental_max_depth": 5, "summary_concurrency": 6,
    "condense_group_cap": 16,
    "summary_spend_max_calls": 320, "summary_circuit_breaker_failure_threshold": 4,
    "l2_budget_ratio": 0.80, "serialize_message_max_chars": 4_000_000,
    "stub_threshold_tokens": 100_000, "expansion_context_tokens": 125_000,
    "expand_page_tokens": 32_000, "tool_response_char_scale": 4.0,
}


def _cfg(**overrides):
    c = LCMConfig()
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def test_curve_t_is_clamped_and_linear():
    assert ws.curve_t(0) == 0.0
    assert ws.curve_t(W256) == 0.0
    assert ws.curve_t(W1M) == 1.0
    assert ws.curve_t(2 * W1M) == 1.0
    mid = (W256 + W1M) // 2
    assert abs(ws.curve_t(mid) - 0.5) < 1e-6


def test_the_low_anchor_takes_upstreams_tuning_and_rejects_its_losses():
    """Upstream is the floor, not the target.

    Every value at 256k is upstream's own where the value is a preference — a cost, latency or
    headroom tradeoff. Where it decides how much is lost, or how coarse the index is, upstream's
    number is not adopted at any window. Splitting the two is the point of this test: a single
    "equals upstream" assertion is what let a whole-backlog leaf chunk survive here for so long.
    """
    r = ws.resolve_window_scaled(_cfg(), W256, env={})
    for name, expected in LOW_ANCHOR.items():
        assert r[name].value == pytest.approx(expected), (name, r[name])
        assert r[name].source.startswith("curve@t=0.00")

    upstream_defaults = {f.name: f.default for f in dataclasses.fields(LCMConfig)}
    for name in LOW_ANCHOR_NOT_UPSTREAM:
        anchor = ws.ANCHORS_BY_NAME[name]
        upstream_value = upstream_defaults.get(anchor.field)
        if isinstance(upstream_value, (int, float)) and upstream_value:
            assert r[name].value != upstream_value, (
                f"{name} resolved to upstream's own value; it is a quality/loss setting and "
                "must be decided on merit at every window"
            )


def test_no_window_still_gets_the_low_anchor_quality_values():
    """With no window known yet, tuning falls back to upstream's literals — but a value that
    exists to prevent loss must not wait for a window to start protecting anything."""
    r = ws.resolve_window_scaled(_cfg(), 0, env={})
    defaults = {f.name: f.default for f in dataclasses.fields(LCMConfig)}
    for name, expected in LOW_ANCHOR_TUNING.items():
        assert r[name].value == expected, name
        assert r[name].source == "upstream(no window)"
    for name in ("leaf_pass_cap", "condense_group_cap", "summary_concurrency"):
        assert r[name].value == LOW_ANCHOR[name], name
        assert r[name].value != defaults[ws.ANCHORS_BY_NAME[name].field]


def test_at_1m_equals_design():
    r = ws.resolve_window_scaled(_cfg(), W1M, env={})
    for name, expected in DESIGN_1M.items():
        assert r[name].value == pytest.approx(expected), (name, r[name])
        assert r[name].source.startswith("curve@t=1.00")


@pytest.mark.parametrize("W", [384 * K, 512 * K, 768 * K])
def test_between_anchors_is_strictly_between(W):
    r = ws.resolve_window_scaled(_cfg(), W, env={})
    t = ws.curve_t(W)
    assert 0.0 < t < 1.0
    assert 0.35 < r["context_threshold"].value < 0.80
    assert 3 <= r["incremental_max_depth"].value <= 5
    assert 80 < r["summary_spend_max_calls"].value < 320
    # FLAT anchors are the quality/loss ones: the same fraction at both ends, so they slide with
    # the window but never with `t`. "Strictly between" does not apply to them, by design.
    assert r["leaf_chunk_tokens"].value == round(W * 0.04)
    assert r["fresh_tail_max_tokens"].value == round(W * 0.15)
    assert r["condense_budget_tokens"].value == round(W * 0.20)
    assert r["fresh_tail_count"].value == 400
    assert r["summary_concurrency"].value == 6
    assert r["drain_stop_fraction"].value == pytest.approx(0.30)


def test_worked_values_at_512k():
    W = 512 * K
    r = ws.resolve_window_scaled(_cfg(), W, env={})
    t = ws.curve_t(W)
    assert r["context_threshold"].value == pytest.approx(0.35 + t * 0.45)
    assert r["incremental_max_depth"].value in (3, 4)
    assert r["summary_concurrency"].value == 6  # flat: chunking runs at every window
    # The chunk is the same FRACTION at both anchors, so it slides with the window and is
    # never a value that only makes sense at one end. It used to interpolate from "the whole
    # window" down to 40,000, which made a 256k session summarise its entire backlog into one
    # node — a one-shot compaction with a DAG drawn around it.
    assert r["leaf_chunk_tokens"].value == round(W * 0.04)
    assert 40_000 > r["leaf_chunk_tokens"].value > round(W256 * 0.04)


def test_explicit_env_override_wins_over_curve():
    r = ws.resolve_window_scaled(_cfg(fresh_tail_count=77), W1M, env={"LCM_FRESH_TAIL_COUNT": "77"})
    assert r["fresh_tail_count"].value == 77 and r["fresh_tail_count"].source == "env"


def test_tracked_config_source_wins_over_curve():
    c = _cfg(context_threshold=0.5)
    c.config_sources = {"context_threshold": "config_yaml:lcm.context_threshold"}
    r = ws.resolve_window_scaled(c, W1M, env={})
    assert r["context_threshold"].value == 0.5
    assert r["context_threshold"].source == "config_yaml:lcm.context_threshold"
    # the drain stop has fixed anchors (0.35 -> 0.30) and is clamped to the resolved
    # threshold, so an explicit 0.5 threshold does not drag the stop above the curve
    r0 = ws.resolve_window_scaled(c, W256, env={})
    assert r0["drain_stop_fraction"].value == pytest.approx(0.30)


def test_fork_field_override_is_explicit_by_value():
    r = ws.resolve_window_scaled(_cfg(summary_concurrency=3), W256, env={})
    assert r["summary_concurrency"].value == 3 and r["summary_concurrency"].source == "explicit"
    r = ws.resolve_window_scaled(_cfg(leaf_chunk_fraction=0.05), W1M, env={})
    assert r["leaf_chunk_tokens"].value == 50_000


def test_config_defaults_do_not_change_upstream_fields():
    c = LCMConfig()
    assert (c.fresh_tail_count, c.leaf_chunk_tokens, c.context_threshold) == (32, 20_000, 0.35)
    assert (c.summary_timeout_ms, c.incremental_max_depth, c.l2_budget_ratio) == (60_000, 3, 0.50)
    assert c.summary_concurrency == 0 and c.leaf_pass_cap == 0 and c.drain_stop_fraction == 0.0


def test_status_payload_is_json_safe():
    import json
    r = ws.resolve_window_scaled(_cfg(), W1M, env={})
    json.dumps(ws.status_payload(r, W1M, W256, W1M))


# ── audit A (W1/W2/W3): the curve must be monotone and behaviourally continuous ─────────────

_WINDOWS = [W256, W256 + 10, 263_000, 272_000, 300_000, 400_000, 512 * K, 600_000, 800_000, W1M]


def test_curve_is_monotone_across_the_whole_range():
    """Every weighted value moves in ONE direction from its 256k anchor to its 1M anchor.

    The first version of this curve resolved fraction endpoints against the current window,
    which made `leaf_chunk_tokens` rise to 345k at 512k before falling to 40k, and interpolated
    the fresh-tail cap out of a `0` sentinel, which made it 1 token just above 256k.
    """
    rising = {"context_threshold", "fresh_tail_count", "condense_budget_tokens", "sweep_target_tokens",
              "incremental_max_depth", "summary_concurrency", "summary_spend_max_calls",
              "summary_circuit_breaker_failure_threshold", "l2_budget_ratio", "leaf_pass_cap", "condense_group_cap",
              "leaf_loop_max_seconds", "summary_timeout_ms", "expansion_timeout_ms",
              "serialize_message_max_chars", "stub_threshold_tokens", "expansion_context_tokens",
              "expand_page_tokens", "tool_response_char_scale", "sqlite_cache_kib", "token_cache_size"}
    # `fresh_tail_max_tokens`, `leaf_chunk_tokens` and `condense_budget_tokens` are FLAT
    # fractions now — the same share of the window at both anchors — so they rise with the
    # window, not with t. `drain_stop_fraction` is flat outright.
    falling: set[str] = set()
    rising |= {"leaf_chunk_tokens", "fresh_tail_max_tokens", "drain_stop_fraction"}
    previous = None
    for window in _WINDOWS:
        resolved = ws.resolve_window_scaled(_cfg(), window, env={})
        if previous is not None:
            for name in rising:
                assert resolved[name].value >= previous[name].value - 1e-9, (name, window)
            for name in falling:
                assert resolved[name].value <= previous[name].value + 1e-9, (name, window)
        previous = resolved
    assert set(rising) | set(falling) == set(ws.ANCHORS_BY_NAME), "an anchor has no monotonicity direction"


def test_no_setting_jumps_meaning_just_above_the_low_anchor():
    """Nothing may change MEANING as the window crosses the low anchor.

    The original failure this pins: a `0` sentinel ("no cap") interpolated into a 1-token cap
    just above 256k. The rule is the same now that several anchors are flat — a value ten
    tokens above the anchor must be indistinguishable from the value at it.
    """
    at = ws.resolve_window_scaled(_cfg(), W256, env={})
    just_above = ws.resolve_window_scaled(_cfg(), W256 + 10, env={})
    for name in ws.ANCHORS_BY_NAME:
        low, high = at[name].value, just_above[name].value
        if isinstance(low, (int, float)) and low:
            assert abs(high - low) <= max(1.0, abs(low) * 0.001), (name, low, high)


def test_every_anchor_is_exactly_linear_between_its_endpoints():
    """Audit p11: the endpoint tests did not pin the SHAPE of the curve — replacing the
    interpolation with a quadratic left all of them passing. Assert the actual value at
    several interior windows against the linear formula, per anchor.
    """
    low = ws.DEFAULT_SCALE_LOW_WINDOW
    high = ws.DEFAULT_SCALE_HIGH_WINDOW
    endpoints = {name: (ws.resolve_window_scaled(_cfg(), low, env={})[name].value,
                        ws.resolve_window_scaled(_cfg(), high, env={})[name].value)
                 for name in ws.ANCHORS_BY_NAME}
    for window in (300_000, 400_000, 512 * K, 700_000, 900_000):
        t = ws.curve_t(window)
        resolved = ws.resolve_window_scaled(_cfg(), window, env={})
        for name, (lo, hi) in endpoints.items():
            anchor = ws.ANCHORS_BY_NAME[name]
            if name == "drain_stop_fraction":
                continue                 # clamped to the resolved threshold; covered separately
            expected = lo + t * (hi - lo)
            actual = resolved[name].value
            if anchor.cast is int:
                assert actual == int(round(expected)), (name, window, actual, expected)
            else:
                assert actual == pytest.approx(expected), (name, window, actual, expected)


def test_drain_stop_is_linear_until_the_threshold_clamps_it():
    for window in (300_000, 512 * K, 900_000):
        t = ws.curve_t(window)
        r = ws.resolve_window_scaled(_cfg(), window, env={})
        # flat at 0.30 now: upstream's "stop the moment we are under the threshold" is only
        # correct for a loop that takes one whole-backlog pass, and this fork chunks everywhere
        expected = min(0.30, r["context_threshold"].value)
        assert r["drain_stop_fraction"].value == pytest.approx(expected)
