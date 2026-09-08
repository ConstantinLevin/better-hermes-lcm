"""Step 2 — the curve resolves on the engine in _set_context_length (both paths) and clone."""
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

W256, W1M = 262_144, 1_000_000


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)
    monkeypatch.delenv("LCM_FRESH_TAIL_COUNT", raising=False)
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "lcm.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    yield e
    close = getattr(e, "close", None)
    if callable(close):
        close()


def test_init_seeds_upstream_values_without_window(engine):
    assert engine.context_length == 0
    assert engine.effective_fresh_tail_count == 32
    assert engine.effective_summary_timeout_ms == 60_000
    assert engine.effective_incremental_max_depth == 3
    assert engine.effective_summary_concurrency == 1
    assert engine.context_threshold == pytest.approx(0.35)


def test_set_context_length_at_1m_resolves_design(engine):
    assert engine._set_context_length(W1M, source="test") is True
    assert engine.effective_fresh_tail_count == 400
    assert engine.effective_fresh_tail_max_tokens == 150_000
    assert engine.effective_condense_budget_tokens == 200_000
    assert engine.effective_leaf_chunk_tokens == 40_000
    assert engine.effective_summary_concurrency == 6
    assert engine.effective_incremental_max_depth == 5
    # threshold default is curved and threshold_tokens follows it
    assert engine.context_threshold == pytest.approx(0.80)
    assert engine._context_threshold_source.startswith("curve@t=1.00")
    assert engine.threshold_tokens == 800_000
    assert engine.effective_drain_stop_fraction == pytest.approx(0.30)


def test_set_context_length_at_256k_is_upstream(engine):
    engine._set_context_length(W256, source="test")
    assert engine.effective_fresh_tail_count == 32
    assert engine.effective_fresh_tail_max_tokens == 0
    assert engine.effective_condense_budget_tokens == 0
    assert engine.effective_leaf_chunk_tokens == W256
    assert engine.context_threshold == pytest.approx(0.35)
    assert engine.threshold_tokens == int(W256 * 0.35)


def test_cleared_context_length_resets_effective_values(engine):
    engine._set_context_length(W1M, source="test")
    assert engine.effective_fresh_tail_count == 400
    engine._set_context_length(0, source="test")
    assert engine.context_length == 0
    assert engine.effective_fresh_tail_count == 32
    assert engine.effective_condense_budget_tokens == 0
    assert engine.threshold_tokens == 0


def test_configured_threshold_is_never_curved(tmp_path):
    cfg = LCMConfig()
    cfg.database_path = str(tmp_path / "lcm.db")
    cfg.context_threshold = 0.5
    cfg.config_sources = {"context_threshold": "config_yaml:lcm.context_threshold"}
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e._set_context_length(W1M, source="test")
    assert e.context_threshold == 0.5
    assert e._context_threshold_source == "config_yaml:lcm.context_threshold"
    assert e.threshold_tokens == 500_000
    # drain stop at t=1 is still the design value; at t=0 it equals the explicit threshold
    assert e.effective_drain_stop_fraction == pytest.approx(0.30)
    e._set_context_length(W256, source="test")
    assert e.effective_drain_stop_fraction == pytest.approx(0.5)


def test_uses_capped_window(engine, monkeypatch):
    # A route cap must feed the curve, not the raw window.
    from hermes_lcm import engine as engine_mod
    monkeypatch.setattr(engine_mod, "_codex_oauth_context_cap", lambda m, p: W256)
    engine._set_context_length(W1M, source="test", model="x", provider="y")
    assert engine.context_length == W256
    assert engine.effective_context_length_cap == W256
    assert engine.effective_fresh_tail_count == 32


def test_protect_last_n_is_not_curved(engine):
    engine._set_context_length(W1M, source="test")
    assert engine.protect_last_n == 32


def test_clone_carries_resolved_values(engine):
    engine._set_context_length(W1M, source="test")
    clone = engine.clone_for_agent()
    try:
        assert clone.effective_fresh_tail_count == 400
        assert clone.effective_summary_concurrency == 6
        assert clone.context_threshold == pytest.approx(0.80)
    finally:
        close = getattr(clone, "close", None)
        if callable(close):
            close()


def test_status_payload_lists_every_anchor(engine):
    engine._set_context_length(W1M, source="test")
    payload = engine.window_scaling_status()
    assert payload["context_length"] == W1M and payload["t"] == 1.0
    assert payload["settings"]["fresh_tail_count"]["value"] == 400
    assert payload["settings"]["fresh_tail_count"]["source"].startswith("curve@")
