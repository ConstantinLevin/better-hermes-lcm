"""fork: better-hermeslcm — concurrent leaf summarisation as a *lookahead* over the serial loop.

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
import queue
import threading
from concurrent.futures import Future
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
    """fork: better-hermeslcm — the host's per-call execution context, captured on the compaction
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


class DaemonThreadPoolExecutor:
    """fork: better-hermeslcm — a small pool of DAEMON workers that cannot block process exit.

    The stdlib pool joins its threads at interpreter exit, so a summariser call the host has
    already abandoned would hold up shutdown; the host solves the same problem the same way for
    its own auxiliary work (tools/daemon_pool.py).

    This is deliberately its OWN pool rather than a ``ThreadPoolExecutor`` subclass. The
    previous version overrode CPython's private ``_adjust_thread_count`` to pass ``daemon=True``
    — and CPython 3.14 changed both that method and the ``_worker`` signature, so on the
    deployed interpreter the first lookahead submission raised
    ``AttributeError: '_initializer'`` and every parallel leaf pass at 1M died with it. Only
    ``submit`` and ``shutdown`` are needed here, and neither needs a private API.
    """

    def __init__(self, max_workers: int = 1, thread_name_prefix: str = "") -> None:
        self._max_workers = max(1, int(max_workers))
        self._thread_name_prefix = thread_name_prefix or "lcm-worker"
        self._queue: "queue.SimpleQueue[Optional[tuple[Future, Callable[..., Any], tuple, dict]]]" = (
            queue.SimpleQueue()
        )
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def _ensure_worker(self) -> None:
        if len(self._threads) >= self._max_workers:
            return
        thread = threading.Thread(
            target=self._work,
            name=f"{self._thread_name_prefix}-{len(self._threads)}",
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 - mirrors Future semantics
                future.set_exception(exc)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._queue.put((future, fn, args, kwargs))
            self._ensure_worker()
        return future

    def purge_cancelled(self) -> int:
        """Drop queued work whose future was already cancelled; return how many went.

        fork: better-hermeslcm — cancelling a future does not remove its queued callable, and the
        callable holds the lookahead, its input messages and the captured host scope. Four
        abandoned attempts with blocked callbacks left six queued entries and four retained
        lookaheads alive (round-3 verify-2 #4). Live entries are re-queued in order.
        """
        with self._lock:
            live: list = []
            dropped = 0
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    live.append(item)
                    continue
                if item[0].cancelled():
                    dropped += 1
                    continue
                live.append(item)
            for item in live:
                self._queue.put(item)
        return dropped

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            if cancel_futures:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None:
                        item[0].cancel()
            for _ in self._threads:
                self._queue.put(None)
            threads = list(self._threads)
        if wait:
            for thread in threads:
                thread.join()


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
        executor: "DaemonThreadPoolExecutor | None" = None,  # fork: shared per engine
    ) -> None:
        self._summarize = summarize
        self._inputs: List[List[Dict[str, Any]]] = [input_filter(chunk) for chunk in chunks]
        self._concurrency = max(1, int(concurrency))
        self._focus_topic = focus_topic
        self._deadline = deadline
        self._scope = HostExecutionScope()  # fork: captured on the compaction thread
        # fork: better-hermeslcm — the pool may be SHARED across attempts. A per-instance pool bounded
        # each attempt on its own, so a host that abandoned one compaction and retried left the
        # previous attempt's blocked workers running and started a second set: 2 → 4 → 6 live
        # workers over three retries (round-2 verify-2 #6). One pool per engine bounds the
        # total, whatever the host does.
        self._owns_executor = executor is None
        self._executor = executor or DaemonThreadPoolExecutor(
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
        # fork: better-hermeslcm — a BOUNDED wait. An unbounded `future.result()` kept the compaction
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
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        else:
            # fork: better-hermeslcm — the shared pool outlives this attempt, so its queue must not
            # keep the cancelled work (and everything that work holds) alive across retries
            # (round-3 verify-2 #4).
            purge = getattr(self._executor, "purge_cancelled", None)
            if callable(purge):
                try:
                    purge()
                except Exception:  # pragma: no cover - purging is best effort
                    logger.debug("LCM could not purge cancelled leaf work", exc_info=True)


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
