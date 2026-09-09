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
    # at the low anchor the cap cannot bind and is reported as upstream's 0
    # fork: the protected tail is sized in TOKENS at every window (0.15*W), not by upstream's
    # flat 32 messages — what 32 messages protect depends on how long they happen to be.
    assert b256.token_limit == round(W256 * 0.15) and b256.token_limited is True
    assert b256.count_limit == 400 and b256.count > 32
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
    # fork: better-hermeslcm — the guard counts CALLS, and a chunked engine spends many small calls
    # where upstream spent one big one, so upstream's 24 stopped a drain mid-way. Anchored to
    # what one drain needs; in tokens it is below upstream's spend at both ends.
    # before a window is known there is no chunk fraction to resolve, so compaction is still
    # one call per pass and upstream's own call budget is the right one
    assert e._summary_spend_guard.max_calls == 24
    assert e._summary_circuit_breaker.failure_threshold == 2
    e._set_context_length(W1M, source="test")
    assert e._summary_spend_guard.max_calls == 320
    assert e._summary_circuit_breaker.failure_threshold == 4
    e._set_context_length(W256, source="test")
    assert e._summary_spend_guard.max_calls == 80
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


# ── audit additions: the last two anchors get consumers ─────────────────────────────────────

def test_expansion_context_tokens_follows_the_curve(tmp_path):
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "expctx.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("x", platform="cli", context_length=262_144)
        assert int(e.effective_expansion_context_tokens) == 32_000
        e._set_context_length(1_000_000, source="test")
        assert int(e.effective_expansion_context_tokens) == 125_000
    finally:
        e.shutdown()


def test_tool_response_caps_scale_with_the_window(tmp_path):
    from hermes_lcm import tools as lcm_tools
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "caps.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("x", platform="cli", context_length=262_144)
        assert lcm_tools._scaled_cap(64_000, engine=e) == 64_000
        e._set_context_length(1_000_000, source="test")
        assert lcm_tools._scaled_cap(64_000, engine=e) == 256_000
        assert lcm_tools._scaled_cap(20_000, engine=e) == 80_000
        # the cap sites have no engine parameter: the engine the tool call was handed is used
        lcm_tools._require_engine({"engine": e})
        assert lcm_tools._scaled_cap(64_000) == 256_000
        lcm_tools._require_engine({})
        e.shutdown()
        assert lcm_tools._scaled_cap(64_000) == 64_000  # nothing bound -> upstream's cap
    finally:
        pass


def test_fresh_tail_cap_never_adds_messages_the_count_limit_excluded(tmp_path):
    """Upstream: `fresh_tail_count=0` means no protected tail. A curved token cap must not
    turn that into one (audit A W1 fallout: the cap is only ever a shrinking bound)."""
    e = _engine(tmp_path, fresh_tail_count=0)
    try:
        for window in (W256, W256 + 10, 400_000, W1M):
            e._set_context_length(window, source="test")
            assert e._fresh_tail_boundary(_msgs(20)).count == 0, window
    finally:
        e.shutdown()
