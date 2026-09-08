"""Steps 10-11 — lookahead leaf pipeline: DAG identical to serial, contiguous prefix on
failure, tool-group boundaries, compaction lock, per-worker progress hook."""
import sys
import threading
import time
import types

import pytest

from hermes_lcm import escalation, leaf_pipeline
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

W1M = 1_000_000


def _engine(tmp_path, name, **kw):
    cfg = LCMConfig(**{"fresh_tail_count": 2, "leaf_chunk_tokens": 50, "incremental_max_depth": 0, **kw})
    cfg.database_path = str(tmp_path / f"{name}.db")
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path / name))
    e.on_session_start(name, platform="cli", context_length=W1M)
    e._config.leaf_chunk_fraction = 0.0002  # 200-token chunks -> many passes
    e._resolve_window_scaled_settings()
    return e


def _messages(n=40):
    out = []
    for i in range(n):
        if i % 5 == 3:
            out.append({"role": "assistant", "content": f"call-{i}", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "t", "arguments": "{}"}}]})
            out.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result-{i} " + "r " * 30})
        else:
            out.append({"role": "user", "content": f"turn-{i} " + "word " * 40})
    return out + [{"role": "user", "content": "fresh"}, {"role": "assistant", "content": "answer"}]


def _dag_shape(e):
    return [
        (n.depth, tuple(n.source_ids), n.summary.split("\n")[0])
        for n in sorted(e._dag.get_session_nodes(e._session_id), key=lambda n: n.node_id)
    ]


@pytest.fixture
def deterministic_summariser(monkeypatch):
    calls = {"n": 0, "threads": set()}
    lock = threading.Lock()

    def fake(prompt, max_tokens, model="", timeout=None):
        with lock:
            calls["n"] += 1
            calls["threads"].add(threading.current_thread().name)
        # summary text derived from the source so identity is checkable across runs
        text = prompt[1]["content"] if isinstance(prompt, list) else str(prompt)
        import hashlib
        digest = hashlib.sha1(text.encode()).hexdigest()[:8]
        time.sleep(0.005)
        return f"sum-{digest}\nExpand for details about: {digest}"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", fake)
    return calls


@pytest.mark.parametrize("iteration", range(3))
def test_concurrency_6_dag_identical_to_concurrency_1(tmp_path, deterministic_summariser, iteration):
    serial = _engine(tmp_path, f"serial{iteration}", summary_concurrency=1)
    parallel = _engine(tmp_path, f"parallel{iteration}", summary_concurrency=6)
    try:
        assert int(serial.effective_summary_concurrency) == 1
        assert int(parallel.effective_summary_concurrency) == 6
        serial.compress(_messages(), current_tokens=900_000)
        serial_threads = set(deterministic_summariser["threads"])
        deterministic_summariser["threads"].clear()
        parallel.compress(_messages(), current_tokens=900_000)
        assert any(name.startswith("lcm-leaf") for name in deterministic_summariser["threads"])
        assert all(not name.startswith("lcm-leaf") for name in serial_threads)
        s_shape, p_shape = _dag_shape(serial), _dag_shape(parallel)
        assert len(s_shape) > 3
        # store ids are per-database, so compare relative offsets
        def rel(shape):
            base = min(sid for _d, ids, _s in shape for sid in ids)
            return [(d, tuple(i - base for i in ids), s) for d, ids, s in shape]
        assert rel(s_shape) == rel(p_shape)
        assert parallel._leaf_lookahead is None
    finally:
        serial.shutdown()
        parallel.shutdown()


def test_default_curve_gives_concurrency_6_at_1m_and_1_at_256k(tmp_path):
    e = _engine(tmp_path, "curve")
    try:
        assert int(e.effective_summary_concurrency) == 6
        e._set_context_length(262_144, source="test")
        assert int(e.effective_summary_concurrency) == 1
    finally:
        e.shutdown()


def test_partial_failure_persists_contiguous_prefix(tmp_path, monkeypatch):
    e = _engine(tmp_path, "partial", summary_concurrency=4)
    try:
        counter = {"n": 0}
        lock = threading.Lock()

        def fake(prompt, max_tokens, model="", timeout=None):
            with lock:
                counter["n"] += 1
                n = counter["n"]
            text = prompt[1]["content"] if isinstance(prompt, list) else str(prompt)
            if "turn-10 " in text:  # the chunk holding turn-10 fails on every route
                return None
            return "ok\nExpand for details about: ok"

        monkeypatch.setattr(escalation, "_call_llm_for_summary", fake)
        msgs = _messages()
        result = e.compress(msgs, current_tokens=900_000)
        leaves = sorted(e._dag.get_session_nodes(e._session_id), key=lambda n: n.node_id)
        assert leaves, "chunks before the failing one must be persisted"
        covered = [sid for n in leaves for sid in n.source_ids]
        assert covered == sorted(covered)
        assert covered == list(range(covered[0], covered[0] + len(covered)))  # contiguous prefix
        failing_store_id = next(
            row["store_id"] for row in e._store.get_session_messages(e._session_id, limit=1000)
            if "turn-10 " in str(row.get("content"))
        )
        assert max(covered) < failing_store_id
        assert e.get_active_compression_failure_cooldown() is not None
        assert e._leaf_lookahead is None
        # the published context keeps the raw of everything not persisted
        assert any("turn-10 " in str(m.get("content")) for m in result)
    finally:
        e.shutdown()


def test_tool_call_result_pair_not_split_at_boundary(tmp_path, deterministic_summariser):
    e = _engine(tmp_path, "pairs", summary_concurrency=4)
    try:
        e.compress(_messages(), current_tokens=900_000)
        rows = {row["store_id"]: row for row in e._store.get_session_messages(e._session_id, limit=1000)}
        for node in e._dag.get_session_nodes(e._session_id):
            ids = node.source_ids
            first, last = rows[ids[0]], rows[ids[-1]]
            assert first.get("role") != "tool", "a chunk never starts with an orphaned result"
            assert not (last.get("role") == "assistant" and last.get("tool_calls")), "never ends on an unanswered call"
    finally:
        e.shutdown()


def test_second_worker_blocked_by_lock(tmp_path, monkeypatch):
    e = _engine(tmp_path, "lock", summary_concurrency=1)
    try:
        started = threading.Event()
        release = threading.Event()

        def slow(prompt, max_tokens, model="", timeout=None):
            started.set()
            release.wait(5)
            return "s\nExpand for details about: s"

        monkeypatch.setattr(escalation, "_call_llm_for_summary", slow)
        msgs = _messages(10)
        outcome = {}

        def first():
            outcome["first"] = e.compress(msgs, current_tokens=900_000)

        t = threading.Thread(target=first)
        t.start()
        assert started.wait(5)
        # the host abandoned the first worker and retries: refused, input returned untouched
        second = e.compress(msgs, current_tokens=900_000)
        assert second is msgs
        assert "already in progress" in e._last_compression_noop_reason
        release.set()
        t.join(10)
        assert outcome["first"] is not msgs
        assert not e._compaction_lock.held
    finally:
        e.shutdown()


def test_workers_tick_the_host_progress_hook(tmp_path, monkeypatch):
    e = _engine(tmp_path, "hook", summary_concurrency=4)
    try:
        ticks = []
        hook_seen = {"threads": set()}
        aux = types.ModuleType("agent.auxiliary_client")

        class _Local(threading.local):
            pass

        aux._aux_progress = _Local()
        aux._aux_progress.hook = lambda *a, **k: ticks.append(1)

        import contextlib

        @contextlib.contextmanager
        def aux_progress_hook(hook):
            aux._aux_progress.hook = hook
            try:
                yield
            finally:
                aux._aux_progress.hook = None

        aux.aux_progress_hook = aux_progress_hook
        monkeypatch.setattr(leaf_pipeline, "_host_aux_progress", aux._aux_progress)
        monkeypatch.setattr(leaf_pipeline, "_host_aux_progress_hook", aux_progress_hook)

        def fake(prompt, max_tokens, model="", timeout=None):
            hook = getattr(aux._aux_progress, "hook", None)
            if hook is not None:
                hook_seen["threads"].add(threading.current_thread().name)
                hook()
            return "s\nExpand for details about: s"

        monkeypatch.setattr(escalation, "_call_llm_for_summary", fake)
        e.compress(_messages(), current_tokens=900_000)
        assert ticks
        assert any(name.startswith("lcm-leaf") for name in hook_seen["threads"])
    finally:
        e.shutdown()


def test_extraction_and_assertions_stay_on_the_compaction_thread(tmp_path, deterministic_summariser, monkeypatch):
    e = _engine(tmp_path, "extract", summary_concurrency=4)
    try:
        seen = {"threads": set()}

        def fake_extraction(*args, **kwargs):
            seen["threads"].add(threading.current_thread().name)
            return None

        e._config.pre_compaction_extraction_enabled = True
        monkeypatch.setattr(e, "_run_pre_compaction_extraction", fake_extraction, raising=False)
        e.compress(_messages(), current_tokens=900_000)
        if seen["threads"]:
            assert all(not name.startswith("lcm-leaf") for name in seen["threads"])
    finally:
        e.shutdown()


def test_plan_chunks_and_chunks_needed():
    msgs = [{"role": "user", "content": "x " * 40} for _ in range(10)]
    select = lambda residual, tokens: residual[:2]
    chunks = leaf_pipeline.plan_chunks(msgs, 1, select, max_chunks=3)
    assert [len(c) for c in chunks] == [2, 2, 2]
    assert chunks[1][0] is msgs[2]
    assert leaf_pipeline.chunks_needed(900_000, 300_000, 40_000, 0.20) == int((600_000 / 32_000) + 0.999) + 1
    assert leaf_pipeline.chunks_needed(100, 300_000, 40_000, 0.20, extra=5) == 6
