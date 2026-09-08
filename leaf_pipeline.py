"""fork: betterlcm — concurrent leaf summarisation as a *lookahead* over the serial loop.

Upstream's leaf loop is kept verbatim in shape: preamble → select the oldest chunk →
summarise → persist → repeat. This module lets the summariser calls for the NEXT chunks run
while the loop persists the current one, without changing the order anything is written in:

- Chunk boundaries are planned up front from the residual span with the same aligned slicing
  the loop uses (`_select_oldest_leaf_chunk_aligned`), so the loop's k-th selection equals the
  k-th planned chunk. `take()` verifies that by message identity and steps aside otherwise.
- Only `_summarize_leaf_chunk_with_rescue` runs on workers (pure: serialise + LLM call; the
  breaker and spend guard are lock-protected). Every DAG/store write stays on the compaction
  thread, in chronological order, so a failure leaves a contiguous persisted prefix.
- Each worker installs the compaction thread's auxiliary progress hook so the host watchdog
  (`compression.context_timeout_seconds`, an *inactivity* budget) sees tokens moving.
- `CompactionLock` is the per-engine mutex: a compaction the host abandoned (its worker
  thread kept running) can never write concurrently with the host's retry.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import time
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:  # the host's thread-local execution scopes (absent in package-less tests)
    from agent.auxiliary_client import (
        _aux_interrupt_protection as _host_aux_interrupt,
        _aux_progress as _host_aux_progress,
        _current_aux_stream_deadline as _host_stream_deadline,
        aux_interrupt_protection as _host_aux_interrupt_scope,
        aux_progress_hook as _host_aux_progress_hook,
        aux_stream_deadline as _host_aux_stream_deadline,
    )
except Exception:  # pragma: no cover - CI without hermes-agent
    _host_aux_progress = None
    _host_aux_progress_hook = None
    _host_aux_interrupt = None
    _host_aux_interrupt_scope = None
    _host_stream_deadline = None
    _host_aux_stream_deadline = None


class HostExecutionScope:
    """fork: betterlcm — the host's per-call execution context, captured on the compaction
    thread and re-installed inside every worker.

    Upstream called the summariser directly on the thread the host had prepared, so it ran
    inside three thread-local scopes the host installs around compression: the forward-progress
    hook (its inactivity watchdog), the stream deadline (the wall clock the host is actually
    waiting on), and interrupt protection with the cancellation check (`/stop`). It also ran
    with the caller's context variables, which carry the profile/secret scope and the HOME
    override — and therefore the model ROUTING the operator configured.

    A bare worker thread starts with none of that: an empty Context and no scopes. Its
    summariser call could take a different route than the same call on the compaction thread,
    outlive the host's deadline, and ignore cancellation. Everything here is captured once,
    before submission, and applied per call.
    """

    def __init__(self) -> None:
        self.progress_hook = getattr(_host_aux_progress, "hook", None) if _host_aux_progress else None
        self.stream_deadline = _host_stream_deadline() if _host_stream_deadline else None
        self.interrupt_active = bool(getattr(_host_aux_interrupt, "active", False)) if _host_aux_interrupt else False
        self.cancel_check = getattr(_host_aux_interrupt, "cancel_check", None) if _host_aux_interrupt else None
        self.cancel_event = getattr(_host_aux_interrupt, "cancel_event", None) if _host_aux_interrupt else None
        self.context = contextvars.copy_context()

    @contextlib.contextmanager
    def applied(self):
        """Re-install the captured host scopes for the duration of one worker call."""
        with contextlib.ExitStack() as stack:
            if self.progress_hook is not None and _host_aux_progress_hook is not None:
                stack.enter_context(_host_aux_progress_hook(self.progress_hook))
            if self.stream_deadline is not None and _host_aux_stream_deadline is not None:
                stack.enter_context(_host_aux_stream_deadline(self.stream_deadline))
            if self.interrupt_active and _host_aux_interrupt_scope is not None:
                stack.enter_context(_host_aux_interrupt_scope(
                    True, cancel_check=self.cancel_check, cancel_event=self.cancel_event))
            yield

    def run(self, fn, *args, **kwargs):
        """Run ``fn`` in a COPY of the captured context, inside the captured scopes.

        A fresh copy per call: `Context.run` refuses re-entry, so sharing one copy across
        concurrent workers would raise.
        """
        def _inner():
            with self.applied():
                return fn(*args, **kwargs)
        return self.context.copy().run(_inner)


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """fork: betterlcm — workers that do not block process exit.

    The stdlib pool joins its threads at interpreter exit, so a summariser call the host has
    already abandoned would hold up shutdown. The host solves this the same way for its own
    auxiliary work (tools/daemon_pool.py).
    """

    def _adjust_thread_count(self) -> None:  # pragma: no cover - mirrors CPython with daemon=True
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, work_queue=self._work_queue):
            work_queue.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            import weakref
            from concurrent.futures.thread import _worker
            thread = threading.Thread(
                name=f"{self._thread_name_prefix or self.__class__.__name__}_{num_threads}",
                target=_worker,
                args=(weakref.ref(self, weakref_cb), self._work_queue,
                      self._initializer, self._initargs),
                daemon=True,
            )
            thread.start()
            self._threads.add(thread)


def plan_chunks(
    candidate_raw: Sequence[Dict[str, Any]],
    chunk_tokens: int,
    select: Callable[[List[Dict[str, Any]], int], List[Dict[str, Any]]],
    *,
    max_chunks: int,
) -> List[List[Dict[str, Any]]]:
    """Slice ``candidate_raw`` the way the loop will: repeatedly the oldest aligned chunk."""
    chunks: List[List[Dict[str, Any]]] = []
    residual = list(candidate_raw)
    while residual and len(chunks) < max_chunks:
        chunk = select(residual, chunk_tokens)
        if not chunk:
            break
        chunks.append(chunk)
        residual = residual[len(chunk):]
    return chunks


def chunks_needed(estimated_active_tokens: int, stop_tokens: int, chunk_tokens: int,
                  leaf_summary_ratio: float, *, extra: int = 0) -> int:
    """How many chunk passes it should take to drain from ``estimated`` to ``stop``."""
    over = max(0, int(estimated_active_tokens) - int(stop_tokens))
    if over <= 0:
        return 1 + extra
    per_pass = max(1.0, float(chunk_tokens) * (1.0 - float(leaf_summary_ratio or 0.0)))
    return int(math.ceil(over / per_pass)) + 1 + extra


class LeafLookahead:
    """Submits summarisation for planned chunks ahead of the loop; hands results back in order."""

    def __init__(
        self,
        summarize: Callable[..., Any],
        chunks: Sequence[Sequence[Dict[str, Any]]],
        *,
        concurrency: int,
        focus_topic: Optional[str],
        deadline: Optional[float],
        input_filter: Callable[[Sequence[Dict[str, Any]]], List[Dict[str, Any]]] = list,
    ) -> None:
        self._summarize = summarize
        self._inputs: List[List[Dict[str, Any]]] = [input_filter(chunk) for chunk in chunks]
        self._concurrency = max(1, int(concurrency))
        self._focus_topic = focus_topic
        self._deadline = deadline
        self._scope = HostExecutionScope()  # fork: captured on the compaction thread
        self._executor = DaemonThreadPoolExecutor(
            max_workers=self._concurrency, thread_name_prefix="lcm-leaf")
        self._futures: Dict[int, Future] = {}
        self._next_submit = 0
        self._next_take = 0
        self._closed = False
        self._lock = threading.Lock()
        self._submit_ahead()

    # -- planning -------------------------------------------------------------------------

    @property
    def planned(self) -> int:
        return len(self._inputs)

    @property
    def remaining(self) -> int:
        return max(0, len(self._inputs) - self._next_take)

    def _submit_ahead(self) -> None:
        with self._lock:
            if self._closed:
                return
            limit = min(len(self._inputs), self._next_take + self._concurrency)
            while self._next_submit < limit:
                index = self._next_submit
                chunk = self._inputs[index]
                if chunk:
                    self._futures[index] = self._executor.submit(self._run, chunk)
                self._next_submit += 1

    def _run(self, chunk: List[Dict[str, Any]]) -> Any:
        return self._scope.run(self._summarize_chunk, chunk)

    def _summarize_chunk(self, chunk: List[Dict[str, Any]]) -> Any:
        kwargs: Dict[str, Any] = {"focus_topic": self._focus_topic}
        if self._deadline is not None:
            kwargs["deadline"] = self._deadline
        try:
            return self._summarize(chunk, **kwargs)
        except TimeoutError as exc:
            # The loop's wall budget ran out under this call: that is "summariser unavailable
            # within budget", which the loop tolerates (publishes the persisted prefix, arms
            # the cooldown) — never a turn-killing error.
            from .errors import SummaryUnavailableError
            raise SummaryUnavailableError(f"leaf summarisation exceeded the compaction time budget: {exc}") from exc

    # -- consumption ----------------------------------------------------------------------

    def matches_next(self, summary_input_chunk: Sequence[Dict[str, Any]]) -> bool:
        if self._closed or self._next_take >= len(self._inputs):
            return False
        planned = self._inputs[self._next_take]
        return len(planned) == len(summary_input_chunk) and all(
            a is b for a, b in zip(planned, summary_input_chunk)
        )

    def take(self, summary_input_chunk: Sequence[Dict[str, Any]]) -> Any:
        """Result for the loop's current chunk (raises what the summariser raised).

        The caller must have checked ``matches_next``; a mismatch means the loop diverged
        from the plan (e.g. a rescue shrank a chunk) and the lookahead must be closed.
        """
        index = self._next_take
        future = self._futures.get(index)
        self._next_take += 1
        self._submit_ahead()
        if future is None:
            raise RuntimeError("lookahead chunk had no summariser input")
        # fork: betterlcm — a BOUNDED wait. An unbounded `future.result()` kept the compaction
        # thread (and the per-engine compaction lock) occupied after the host had already
        # abandoned the attempt, so the host's retry found the engine busy and did nothing.
        timeout = None
        if self._deadline is not None:
            timeout = max(0.0, self._deadline - time.monotonic())
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            from .errors import SummaryUnavailableError
            future.cancel()
            raise SummaryUnavailableError(
                "leaf summarisation exceeded the compaction time budget while waiting for a worker"
            ) from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = [f for i, f in self._futures.items() if i >= self._next_take]
        for future in pending:
            future.cancel()
        # never block the compaction thread on abandoned LLM calls
        self._executor.shutdown(wait=False, cancel_futures=True)


class CompactionLock:
    """Per-engine, process-local, non-blocking mutex around compress()."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.owner_thread: Optional[int] = None

    def try_acquire(self) -> bool:
        if self._lock.acquire(blocking=False):
            self.owner_thread = threading.get_ident()
            return True
        return False

    def release(self) -> None:
        self.owner_thread = None
        try:
            self._lock.release()
        except RuntimeError:  # pragma: no cover - release without acquire
            pass

    @property
    def held(self) -> bool:
        return self._lock.locked()
