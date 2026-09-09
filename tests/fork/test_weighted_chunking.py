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


def test_the_leaf_chunk_is_bounded_at_every_window(tmp_path, summariser_calls):
    """Chunking is what LCM IS. One summariser call per bounded chunk, at 256k exactly as at
    1M — a leaf that stands for the whole backlog is a one-shot compaction wearing a DAG."""
    e = _engine(tmp_path, W256)
    try:
        # the chunk is the same FRACTION of the window at both anchors, never the window itself
        assert int(e.effective_leaf_chunk_tokens) == round(W256 * 0.04)
        assert int(e.effective_leaf_pass_cap) > 1, "one pass per compaction cannot drain"
        e._config.leaf_chunk_fraction = 0.0008  # ~200-token chunks so several passes are needed
        e._resolve_window_scaled_settings()
        messages = _backlog(40) + _tail()
        e.compress(messages, current_tokens=e.threshold_tokens + 1)
        leaves = [n for n in e._dag.get_session_nodes(e._session_id) if n.depth == 0]
        assert len(leaves) > 1, "256k must chunk, not swallow the backlog"
        covered = sorted(sid for n in leaves for sid in n.source_ids)
        assert covered == sorted(set(covered)), "each row belongs to exactly one leaf"
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
        # the drain stops once the context is under the stop fraction, so it covers a
        # CONTIGUOUS oldest prefix rather than necessarily all 40 — each row exactly once
        covered = sorted(sid for n in leaves for sid in n.source_ids)
        assert covered == list(range(covered[0], covered[0] + len(covered)))
        assert len(covered) == len(set(covered))
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


def test_rescue_fallback_never_splits_a_tool_group(tmp_path):
    """Audit A #9: after the aligned shrink attempts, the last resort was `chunk[:-1]`, which
    strips a result from its call — the summariser sees an unanswered call, the result stays
    outside the node, and assembly's orphan guard then removes it."""
    e = _engine(tmp_path, W1M)
    try:
        call = {"role": "assistant", "content": "calling", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
        result = {"role": "tool", "tool_call_id": "c1", "content": "r " * 50}
        chunk = [{"role": "user", "content": "q " * 50}, call, result]
        # the floor forces both aligned attempts to be rejected, so the fallback runs
        e._config.leaf_chunk_tokens = 10 ** 9
        shrunk = e._next_leaf_rescue_chunk(chunk, current_source_tokens=999)
        assert [m.get("role") for m in shrunk] == ["user"]
        # one indivisible group: give up rather than split it
        assert e._next_leaf_rescue_chunk([call, result], current_source_tokens=999) == []
    finally:
        e.shutdown()


# ── audit D4: the low anchor is upstream's BEHAVIOUR, not values that merely equal it ───────

def test_an_oversized_imported_history_is_chunked_not_swallowed(tmp_path, summariser_calls):
    """A resumed or imported history far larger than the window is exactly the case where one
    whole-backlog summary is most useless: one node standing for everything. It is chunked like
    any other backlog, so each leaf covers a span a reader can expand."""
    e = _engine(tmp_path, W256)
    try:
        messages = [{"role": "user", "content": f"turn-{i} " + ("word " * 9000)} for i in range(40)]
        messages += _tail()
        e.compress(messages, current_tokens=e.threshold_tokens + 1)
        leaves = [n for n in e._dag.get_session_nodes(e._session_id) if n.depth == 0]
        assert len(leaves) > 1, "an oversized history must not become one node"
        chunk = int(e.effective_leaf_chunk_tokens)
        for node in leaves:
            # one chunk, plus at most the single message that overflowed it
            assert node.source_token_count <= chunk + 12_000, node.source_token_count
    finally:
        e.shutdown()


def test_dynamic_chunking_keeps_upstream_timing_at_the_low_anchor(tmp_path, monkeypatch, summariser_calls):
    """With upstream's own dynamic-chunk policy enabled, a slow but SUCCESSFUL run must not be
    cut short by a clock upstream never had."""
    import time as _time
    e = _engine(tmp_path, W256, dynamic_leaf_chunk_enabled=True, dynamic_leaf_chunk_max=40_000)
    try:
        clock = {"now": _time.monotonic()}
        monkeypatch.setattr("hermes_lcm.compaction.time.monotonic", lambda: clock["now"])
        from hermes_lcm import escalation
        original = escalation._call_llm_for_summary

        def slow(prompt, max_tokens, model="", timeout=None):
            clock["now"] += 500  # far beyond any fork budget
            return original(prompt, max_tokens, model=model, timeout=timeout)

        monkeypatch.setattr(escalation, "_call_llm_for_summary", slow)
        messages = _backlog(30) + _tail()
        e.compress(messages, current_tokens=e.threshold_tokens + 1)
        assert e._last_compression_status == "compacted"
        assert e._last_compression_noop_reason != "leaf loop time budget exhausted"
    finally:
        e.shutdown()
