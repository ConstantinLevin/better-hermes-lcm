"""Step 8 — weighted chunking, drain stop, pass cap and time budget on the non-sweep path."""
import pytest

from hermes_lcm import escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

W256 = 262_144
W1M = 1_000_000


def _engine(tmp_path, window, **kw):
    cfg = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=50, **kw)
    cfg.database_path = str(tmp_path / f"chunk-{window}.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e.on_session_start(f"s{window}", platform="cli", context_length=window)
    return e


def _backlog(n, size=60):
    return [{"role": "user", "content": f"turn-{i} " + ("word " * size)} for i in range(n)]


def _tail():
    return [{"role": "user", "content": "fresh"}, {"role": "assistant", "content": "answer"}]


@pytest.fixture
def summariser_calls(monkeypatch):
    calls = []

    def fake(prompt, max_tokens, model="", timeout=None):
        calls.append(prompt)
        return "s\nExpand for details about: mock"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", fake)
    return calls


def test_non_sweep_at_256k_is_one_whole_backlog_pass(tmp_path, summariser_calls):
    e = _engine(tmp_path, W256)
    try:
        assert int(e.effective_leaf_pass_cap) == 1
        assert int(e.effective_leaf_chunk_tokens) == W256
        messages = _backlog(40) + _tail()
        e.compress(messages, current_tokens=e.threshold_tokens + 1)
        nodes = e._dag.get_session_nodes(e._session_id)
        assert len(nodes) == 1
        assert len(nodes[0].source_ids) == 40
        assert len(summariser_calls) == 1
    finally:
        e.shutdown()


def test_non_sweep_at_1m_drains_in_curved_chunks_to_stop_fraction(tmp_path, summariser_calls):
    e = _engine(tmp_path, W1M)
    try:
        assert int(e.effective_leaf_pass_cap) == 64
        assert int(e.effective_leaf_chunk_tokens) == 40_000
        assert e._non_sweep_drain_stop_tokens() == 300_000
        # pretend the prompt is at 900k of 1M; the backlog is ~40 messages of ~65 tokens
        messages = _backlog(40) + _tail()
        e._config.leaf_chunk_fraction = 0.0002  # 200-token chunks so several passes are needed
        e._resolve_window_scaled_settings()
        assert int(e.effective_leaf_chunk_tokens) == 200
        e.compress(messages, current_tokens=900_000)
        leaves = [n for n in e._dag.get_session_nodes(e._session_id) if n.depth == 0]
        # several chunk-sized leaves, oldest first, covering the whole backlog exactly once
        assert len(leaves) > 1
        covered = sorted(sid for n in leaves for sid in n.source_ids)
        assert len(covered) == 40 and covered == list(range(covered[0], covered[0] + 40))
        for n in leaves:
            assert n.source_token_count <= 200 + 80  # one chunk (+ the message that overflows it)
        assert len(summariser_calls) >= len(leaves)  # condensation calls may follow
    finally:
        e.shutdown()


def test_non_sweep_stops_once_under_the_drain_stop(tmp_path, summariser_calls):
    e = _engine(tmp_path, W1M)
    try:
        e._config.leaf_chunk_fraction = 0.0002
        e._resolve_window_scaled_settings()
        messages = _backlog(40) + _tail()
        # current usage just above the stop (300k): one pass gets under it -> stop
        e.compress(messages, current_tokens=300_100)
        leaves = [n for n in e._dag.get_session_nodes(e._session_id) if n.depth == 0]
        assert len(leaves) == 1
        assert len(leaves[0].source_ids) < 40
    finally:
        e.shutdown()


def test_pass_cap_bounds_the_non_sweep_loop(tmp_path, summariser_calls):
    e = _engine(tmp_path, W1M, leaf_pass_cap=2)
    try:
        e._config.leaf_chunk_fraction = 0.0002
        e._resolve_window_scaled_settings()
        e.compress(_backlog(40) + _tail(), current_tokens=900_000)
        assert len([n for n in e._dag.get_session_nodes(e._session_id) if n.depth == 0]) == 2
    finally:
        e.shutdown()


def test_time_budget_bounds_the_non_sweep_loop(tmp_path, summariser_calls, monkeypatch):
    e = _engine(tmp_path, W1M, leaf_loop_max_seconds=0.05)
    try:
        e._config.leaf_chunk_fraction = 0.0002
        e._resolve_window_scaled_settings()
        import time as _time
        real = _time.monotonic
        clock = {"now": real()}
        monkeypatch.setattr("hermes_lcm.compaction.time.monotonic", lambda: clock["now"])
        original = escalation._call_llm_for_summary

        def slow(prompt, max_tokens, model="", timeout=None):
            clock["now"] += 0.03
            return original(prompt, max_tokens, model=model, timeout=timeout)

        monkeypatch.setattr(escalation, "_call_llm_for_summary", slow)
        e.compress(_backlog(40) + _tail(), current_tokens=900_000)
        assert 1 <= len([n for n in e._dag.get_session_nodes(e._session_id) if n.depth == 0]) <= 2
        assert e._last_compression_status == "compacted"
    finally:
        e.shutdown()


def test_chunk_boundary_never_splits_a_tool_group(tmp_path):
    e = _engine(tmp_path, W1M)
    try:
        candidate = [
            {"role": "user", "content": "q " * 40},
            {"role": "assistant", "content": "calling", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "t", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1 " * 40},
            {"role": "tool", "tool_call_id": "c2", "content": "r2 " * 40},
            {"role": "user", "content": "next " * 40},
        ]
        greedy = e._select_oldest_leaf_chunk(candidate, 60)
        assert greedy[-1]["role"] == "assistant"  # upstream would cut between call and results
        aligned = e._select_oldest_leaf_chunk_aligned(candidate, 60)
        assert [m.get("role") for m in aligned] == ["user", "assistant", "tool", "tool"]
        # a budget that ends inside the result run also takes the rest of the run
        aligned2 = e._select_oldest_leaf_chunk_aligned(candidate, 120)
        assert [m.get("role") for m in aligned2][:4] == ["user", "assistant", "tool", "tool"]
    finally:
        e.shutdown()


def test_sweep_budgets_come_from_config_and_curve(tmp_path):
    e = _engine(tmp_path, W256, sweep_max_passes=5)
    try:
        import json
        from hermes_lcm import tools as lcm_tools
        payload = json.loads(lcm_tools.lcm_status({}, engine=e))
        cfg = payload.get("config") or payload
        blob = json.dumps(payload)
        assert '"threshold_full_sweep_max_passes": 5' in blob
        assert '"threshold_full_sweep_max_seconds": 120.0' in blob
    finally:
        e.shutdown()
