"""Step 1 — the weighting curve. At 256k every value is upstream's; at 1M the design's."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import window_scaling as ws  # noqa: E402
from config import LCMConfig  # noqa: E402

K = 1024
W256, W1M = 262_144, 1_000_000

UPSTREAM = {  # what every anchor must resolve to at 256k with default config
    "context_threshold": 0.35, "leaf_chunk_tokens": W256, "leaf_pass_cap": 1,
    "leaf_loop_max_seconds": 120.0, "summary_timeout_ms": 60_000, "expansion_timeout_ms": 120_000,
    "fresh_tail_count": 32, "fresh_tail_max_tokens": 0, "condense_budget_tokens": 0,
    "sweep_target_tokens": 20_000, "incremental_max_depth": 3, "summary_concurrency": 1,
    "summary_spend_max_calls": 24, "summary_circuit_breaker_failure_threshold": 2,
    "l2_budget_ratio": 0.50, "serialize_message_max_chars": 3000, "stub_threshold_tokens": 25_000,
    "expansion_context_tokens": 32_000, "expand_page_tokens": 4_000,
    "tool_response_char_scale": 1.0, "sqlite_cache_kib": 2_048, "token_cache_size": 2_048,
}
DESIGN_1M = {
    "context_threshold": 0.80, "drain_stop_fraction": 0.30, "leaf_chunk_tokens": 40_000,
    "leaf_pass_cap": 64, "leaf_loop_max_seconds": 200.0, "summary_timeout_ms": 200_000,
    "fresh_tail_count": 400, "fresh_tail_max_tokens": 150_000, "condense_budget_tokens": 200_000,
    "sweep_target_tokens": 200_000, "incremental_max_depth": 5, "summary_concurrency": 6,
    "summary_spend_max_calls": 120, "summary_circuit_breaker_failure_threshold": 4,
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


def test_at_256k_equals_upstream():
    r = ws.resolve_window_scaled(_cfg(), W256, env={})
    for name, expected in UPSTREAM.items():
        assert r[name].value == expected, (name, r[name])
        assert r[name].source.startswith("curve@t=0.00")
    # drain stop at t=0 equals the threshold (stop once under)
    assert r["drain_stop_fraction"].value == pytest.approx(0.35)


def test_no_window_is_upstream():
    r = ws.resolve_window_scaled(_cfg(), 0, env={})
    for name, expected in UPSTREAM.items():
        if name == "leaf_chunk_tokens":
            continue  # fraction lows have no meaning without a window
        assert r[name].value == expected, name
        assert r[name].source == "upstream(no window)"


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
    assert 32 < r["fresh_tail_count"].value < 400
    assert 1 <= r["summary_concurrency"].value <= 6
    # monotone: chunk shrinks toward 40k, budget grows toward 0.2W
    assert 40_000 < r["leaf_chunk_tokens"].value < W
    assert 0 < r["condense_budget_tokens"].value < 0.20 * W


def test_worked_values_at_512k():
    W = 512 * K
    r = ws.resolve_window_scaled(_cfg(), W, env={})
    t = ws.curve_t(W)
    assert r["context_threshold"].value == pytest.approx(0.35 + t * 0.45)
    assert r["incremental_max_depth"].value in (3, 4)
    assert r["summary_concurrency"].value in (2, 3)
    assert r["leaf_chunk_tokens"].value == int(round(W * (1.0 - 0.96 * t)))


def test_explicit_env_override_wins_over_curve():
    r = ws.resolve_window_scaled(_cfg(fresh_tail_count=77), W1M, env={"LCM_FRESH_TAIL_COUNT": "77"})
    assert r["fresh_tail_count"].value == 77 and r["fresh_tail_count"].source == "env"


def test_tracked_config_source_wins_over_curve():
    c = _cfg(context_threshold=0.5)
    c.config_sources = {"context_threshold": "config_yaml:lcm.context_threshold"}
    r = ws.resolve_window_scaled(c, W1M, env={})
    assert r["context_threshold"].value == 0.5
    assert r["context_threshold"].source == "config_yaml:lcm.context_threshold"
    # the drain stop's low anchor follows the *explicit* threshold at t=0
    r0 = ws.resolve_window_scaled(c, W256, env={})
    assert r0["drain_stop_fraction"].value == pytest.approx(0.5)


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
