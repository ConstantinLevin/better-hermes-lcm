"""Step 3 — consumers read effective_* (curved) values; upstream at 256k, design at 1M."""
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

W256, W1M = 262_144, 1_000_000


def _engine(tmp_path, **cfg_kwargs):
    cfg = LCMConfig(**cfg_kwargs)
    cfg.database_path = str(tmp_path / "lcm.db")
    return LCMEngine(config=cfg, hermes_home=str(tmp_path))


def _msgs(n, tokens_each=500):
    body = "x " * tokens_each  # ~tokens_each tokens with the char fallback
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + body} for i in range(n)]


def test_fresh_tail_uses_curved_caps(tmp_path):
    e = _engine(tmp_path)
    msgs = _msgs(600)
    e._set_context_length(W256, source="test")
    b256 = e._fresh_tail_boundary(msgs)
    assert b256.count == 32 and b256.token_limit == 0
    e._set_context_length(W1M, source="test")
    b1m = e._fresh_tail_boundary(msgs)
    assert b1m.count_limit == 400 and b1m.token_limit == 150_000
    assert b1m.count <= 400 and b1m.tokens <= 150_000 + 2_000  # newest always kept


def test_explicit_fresh_tail_count_is_not_curved(tmp_path):
    e = _engine(tmp_path, fresh_tail_count=4)
    e._set_context_length(W1M, source="test")
    assert e.effective_fresh_tail_count == 4
    assert e._fresh_tail_boundary(_msgs(50)).count == 4


def test_guard_and_breaker_are_retuned_on_resolve(tmp_path):
    e = _engine(tmp_path)
    assert e._summary_spend_guard.max_calls == 24
    assert e._summary_circuit_breaker.failure_threshold == 2
    e._set_context_length(W1M, source="test")
    assert e._summary_spend_guard.max_calls == 120
    assert e._summary_circuit_breaker.failure_threshold == 4
    e._set_context_length(W256, source="test")
    assert e._summary_spend_guard.max_calls == 24
    assert e._summary_circuit_breaker.failure_threshold == 2


def test_sweep_target_low_anchor_follows_leaf_chunk_tokens(tmp_path):
    # upstream fallback: summary_prefix_target_tokens == 0 -> leaf_chunk_tokens
    e = _engine(tmp_path, leaf_chunk_tokens=100)
    e._set_context_length(W256, source="test")
    assert e.effective_sweep_target_tokens == 100
    e._set_context_length(0, source="test")
    assert e.effective_sweep_target_tokens == 100
    e2 = _engine(tmp_path / "b", summary_prefix_target_tokens=5_000)
    e2._set_context_length(W1M, source="test")
    assert e2.effective_sweep_target_tokens == 5_000
    e3 = _engine(tmp_path / "c")
    e3._set_context_length(W1M, source="test")
    assert e3.effective_sweep_target_tokens == 200_000


def test_timeouts_and_l2_ratio_curve(tmp_path):
    e = _engine(tmp_path)
    e._set_context_length(W1M, source="test")
    assert e.effective_summary_timeout_ms == 200_000
    assert e.effective_expansion_timeout_ms == 200_000
    assert e.effective_l2_budget_ratio == pytest.approx(0.80)
    assert e.effective_stub_threshold_tokens == 100_000
    e._set_context_length(W256, source="test")
    assert (e.effective_summary_timeout_ms, e.effective_expansion_timeout_ms) == (60_000, 120_000)
    assert e.effective_l2_budget_ratio == pytest.approx(0.50)
    assert e.effective_stub_threshold_tokens == 25_000
