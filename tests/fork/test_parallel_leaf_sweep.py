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


# ── audit D2: workers must run inside the host's execution scopes ───────────────────────────

def test_workers_inherit_context_vars_deadline_and_cancellation(tmp_path, monkeypatch):
    """Upstream called the summariser on the thread the host had prepared, inside three
    thread-local scopes (progress hook, stream deadline, interrupt protection) and with the
    caller's context variables — which carry the profile/secret scope and therefore the model
    ROUTING the operator configured. A bare worker starts with none of that, so its call could
    take a different route, outlive the host's deadline and ignore /stop.
    """
    import contextvars
    from hermes_lcm import leaf_pipeline as lp

    route = contextvars.ContextVar("lcm_test_route", default="unset")
    route.set("operator-configured")

    class _Local:
        pass

    host_progress, host_deadline, host_interrupt = _Local(), _Local(), _Local()
    host_progress.hook = lambda *a, **k: None
    host_deadline.value = 12345.0
    host_interrupt.active = True
    host_interrupt.cancel_check = lambda: False
    host_interrupt.cancel_event = None

    import contextlib as _ctx
    monkeypatch.setattr(lp, "_host_aux_progress", host_progress)
    monkeypatch.setattr(lp, "_host_stream_deadline", lambda: host_deadline.value)
    monkeypatch.setattr(lp, "_host_aux_interrupt", host_interrupt)

    seen = {}

    @_ctx.contextmanager
    def fake_progress(hook):
        seen["progress"] = hook
        yield

    @_ctx.contextmanager
    def fake_deadline(value):
        seen["deadline"] = value
        yield

    @_ctx.contextmanager
    def fake_interrupt(active=True, cancel_check=None, cancel_event=None):
        seen["interrupt"] = (active, cancel_check)
        yield

    monkeypatch.setattr(lp, "_host_aux_progress_hook", fake_progress)
    monkeypatch.setattr(lp, "_host_aux_stream_deadline", fake_deadline)
    monkeypatch.setattr(lp, "_host_aux_interrupt_scope", fake_interrupt)

    def summarize(chunk, **kwargs):
        seen["route"] = route.get()
        seen["thread"] = threading.current_thread().name
        return ("ok", 1, "s", 1, 1)

    lookahead = lp.LeafLookahead(summarize, [[{"role": "user", "content": "a"}],
                                             [{"role": "user", "content": "b"}]],
                                 concurrency=2, focus_topic=None, deadline=None)
    try:
        assert lookahead.take([{"role": "user", "content": "a"}]) == ("ok", 1, "s", 1, 1)
        assert seen["thread"].startswith("lcm-leaf"), "must actually run on a worker"
        assert seen["route"] == "operator-configured", "context vars (and routing) must cross"
        assert seen["progress"] is host_progress.hook
        assert seen["deadline"] == 12345.0
        assert seen["interrupt"] == (True, host_interrupt.cancel_check)
    finally:
        lookahead.close()


def test_worker_wait_is_bounded_by_the_shared_deadline(tmp_path):
    """An unbounded future.result() kept the compaction thread — and the per-engine compaction
    lock — occupied after the host abandoned the attempt, so the host's retry found the engine
    busy and did nothing."""
    import time as _time
    from hermes_lcm import leaf_pipeline as lp
    from hermes_lcm.errors import SummaryUnavailableError

    release = threading.Event()

    def slow(chunk, **kwargs):
        release.wait(30)
        return ("late", 1, "s", 1, 1)

    lookahead = lp.LeafLookahead(slow, [[{"role": "user", "content": "a"}]],
                                 concurrency=1, focus_topic=None,
                                 deadline=_time.monotonic() + 0.2)
    try:
        started = _time.monotonic()
        with pytest.raises(SummaryUnavailableError, match="waiting for a worker"):
            lookahead.take([{"role": "user", "content": "a"}])
        assert _time.monotonic() - started < 5, "must not wait for the abandoned call"
    finally:
        release.set()
        lookahead.close()


def test_lookahead_workers_do_not_block_process_exit():
    """The pool is the fork's OWN, not a ThreadPoolExecutor subclass: the previous version
    overrode CPython's private `_adjust_thread_count`, and on Python 3.14 — the deployed
    interpreter — the first submission raised AttributeError('_initializer') and every parallel
    leaf pass died with it. An end-to-end 1M run found that; no unit test did."""
    from hermes_lcm import leaf_pipeline as lp
    pool = lp.DaemonThreadPoolExecutor(max_workers=2, thread_name_prefix="lcm-probe")
    try:
        assert pool.submit(lambda value: value * 2, 21).result(timeout=5) == 42
        with pytest.raises(ValueError):
            pool.submit(lambda: (_ for _ in ()).throw(ValueError("boom"))).result(timeout=5)
        # a second worker really is created, and every thread is a daemon
        futures = [pool.submit(lambda: time.sleep(0.05)) for _ in range(4)]
        for future in futures:
            future.result(timeout=5)
        assert pool._threads and all(t.daemon for t in pool._threads)
        assert len(pool._threads) <= 2
    finally:
        pool.shutdown(wait=False)


def test_the_pool_survives_this_interpreter_creating_real_worker_threads():
    """Regression guard for the crash above: submitting more work than workers must not touch
    any private CPython pool API."""
    from hermes_lcm import leaf_pipeline as lp
    pool = lp.DaemonThreadPoolExecutor(max_workers=3, thread_name_prefix="lcm-probe")
    try:
        results = [pool.submit(lambda index=index: index) for index in range(12)]
        assert sorted(future.result(timeout=5) for future in results) == list(range(12))
    finally:
        pool.shutdown(wait=True)
