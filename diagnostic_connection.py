"""A private SQLite handle for diagnostics that must not own a live transaction.

A ``SAVEPOINT`` (or a ``BEGIN``, or a ``commit()``) belongs to the CONNECTION,
not to the thread that issued it. The FTS5 integrity-check is a probe INSERT
wrapped in ``SAVEPOINT`` / ``ROLLBACK TO`` / ``RELEASE``, and the explicit FTS
repair opens ``BEGIN IMMEDIATE`` and commits -- so running either on the store's
or the DAG's live shared connection makes a diagnostic the owner of whatever
transaction is open on that connection at the time.

The host does not join a tool call it has timed out: it returns a tool error,
sets a cooperative interrupt bit the SQLite check never reads, and moves on to
the next turn, which ingests through the same engine. A Doctor still inside its
savepoint when that happens will ``ROLLBACK TO`` over rows the ingest inserted
in the meantime -- an ingest that already reported success and advanced its
cursor -- or ``RELEASE``/``commit()`` a batch its writer had not finished.
Narrowing that window is not a fix; the diagnostic has to stop owning a
transaction it did not open.

So diagnostics take their own handle on the same database file, which is what
``db_bootstrap._run_background_integrity_scan`` already does for the startup
scan. The savepoint then belongs to a connection nobody else uses: an abandoned
or killed diagnostic can commit and roll back only its own work, whenever it
finally gets there.

The rejected alternative was holding the store's and the DAG's owner locks for
the whole check. It would also have kept foreign rows safe, and CPython does not
kill the abandoned worker, so the lock would have been released when the check
finished rather than held forever -- but the block would have lasted the check's
full duration with no timeout and no failure, and it depends on every writer
taking the same lock, which the DAG's second handle on the same file does not.
The decisive argument is the foreign savepoint, not the lock's duration: a
private handle removes the hazard by construction instead of by cooperation, and
what serializes it against live writes is then SQLite's own single-writer rule,
which is bounded and reports a real error.

A check the diagnostic cannot finish says ``unchecked`` with the true reason.
``unchecked_fts_remedy`` turns that reason into the remedy that would actually
work, because the remedy for "a write was in flight" is not the remedy for "this
database is read-only".
"""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .db_bootstrap import SQLITE_BUSY_TIMEOUT_MS, _database_path_for_connection
from .sqlite_util import _temporary_sqlite_busy_timeout

try:  # the host's cooperative per-thread interrupt bit. ``tools`` here is the
    # HOST's package: this plugin's tools.py is only importable as
    # hermes_lcm.tools, and anything else under that name fails the import and
    # leaves the signal simply absent.
    from tools.interrupt import is_interrupted as _host_is_interrupted
except Exception:  # pragma: no cover - CI without hermes-agent
    _host_is_interrupted = None

# How long the diagnostic waits for SQLite's write lock before giving up.
# Short on purpose: the diagnostic yields to the live write path rather than
# queueing behind it, and a check that could not take the lock reports
# ``unchecked`` with SQLite's own message instead of holding the tool call open
# for the 30 s a store connection is willing to wait.
#
# It is flat, not one of the fork's window-scaled values, for the same reason
# ``host_cooldown``'s constants are: it is a lock-acquisition backoff, and there
# is no anchor pair to interpolate. That is not the same as claiming it behaves
# identically at both windows -- a larger window means a larger database and
# longer write transactions, so the SAME 2 s yields MORE ``unchecked`` results at
# 1M than at 256k. That is a degradation of the DIAGNOSTIC, never of content, and
# it is only tolerable because ``unchecked`` is honest everywhere it surfaces:
# it never reads as a healthy index, and it carries the remedy for its own cause.
DIAGNOSTIC_BUSY_TIMEOUT_MS = 2_000

# How long the check may HOLD the write lock once it has it. The FTS5
# integrity-check is O(index size) -- the cost that made it a startup problem in
# the first place -- and while it runs, every concurrent writer waits. Half the
# store connection's own busy timeout means a writer that started waiting when
# the check began still gets the lock with the same margin to spare, so a
# bounded check never turns into a failed ingest. Derived from the constant it
# has to stay under rather than chosen, and flat for the reason above.
DIAGNOSTIC_BUDGET_SECONDS = SQLITE_BUSY_TIMEOUT_MS / 1000.0 / 2

# VM instructions between progress callbacks, matching the read-path deadline
# helpers in tools.py.
_PROGRESS_INSTRUCTIONS = 1000


CALLER_WRITABLE = "writable"
CALLER_READ_ONLY = "read-only"
CALLER_UNDETERMINED = "undetermined"


def caller_connection_access(conn: sqlite3.Connection) -> str:
    """Whether the caller opened this connection read-only, and say so honestly.

    Python's sqlite3 exposes no ``sqlite3_db_readonly``, so ask SQLite with the
    one statement that is refused on a read-only connection and that can be
    shown to do nothing on a writable one: ``PRAGMA incremental_vacuum``. It is
    inert exactly when there is nothing for it to free, which is either
    ``auto_vacuum`` NONE (it does nothing in that mode at all) or an empty
    freelist. Measured across every mode, comparing the sha256 of the database,
    the ``-wal`` and the ``-shm`` before and after:

        auto_vacuum=NONE        freelist=0    identical   ro refuses
        auto_vacuum=NONE        freelist=88   identical   ro refuses
        auto_vacuum=FULL        freelist=0    identical   ro refuses
        auto_vacuum=INCREMENTAL freelist=0    identical   ro refuses
        auto_vacuum=INCREMENTAL freelist=88   CHANGED     ro refuses

    It also neither starts nor joins a transaction: inside a foreign write
    transaction the pending row stays pending and the owner's rollback still
    discards it.

    The last row is the one configuration where the probe would really free a
    page, so it is not run there and the answer is ``CALLER_UNDETERMINED``.
    Callers must treat that as "do not write", not as "probably writable": this
    plugin never sets ``auto_vacuum``, so reaching it takes an operator who set
    it themselves, and guessing ``writable`` there is a silent write to a
    database they may have opened read-only, reported as ``ok``.

    Only SQLite's own read-only refusal answers ``CALLER_READ_ONLY``. Anything
    else -- a busy database, a locked one -- is not evidence about the caller's
    intent, and the refusal wins over the busy lock, so a read-only caller under
    contention still answers read-only.
    """
    try:
        auto_vacuum = conn.execute("PRAGMA auto_vacuum").fetchone()
        freelist = conn.execute("PRAGMA freelist_count").fetchone()
        inert = (auto_vacuum and int(auto_vacuum[0]) == 0) or (
            freelist and int(freelist[0]) == 0
        )
        if not inert:
            return CALLER_UNDETERMINED
        with _temporary_sqlite_busy_timeout([conn], DIAGNOSTIC_BUSY_TIMEOUT_MS):
            conn.execute("PRAGMA incremental_vacuum(0)").fetchall()
    except sqlite3.DatabaseError as exc:
        lowered = str(exc).lower()
        if "readonly" in lowered or "read-only" in lowered:
            return CALLER_READ_ONLY
        return CALLER_WRITABLE
    except Exception:  # pragma: no cover - a probe must never break the doctor
        return CALLER_UNDETERMINED
    return CALLER_WRITABLE


def undeterminable_access_result(spec_name: str) -> dict:
    """The honest answer when the caller's access mode could not be established."""
    return {
        "status": "unchecked",
        "detail": (
            f"deep FTS integrity-check for '{spec_name}' was not run: this database uses "
            "incremental auto_vacuum with pages on its freelist, which is the one "
            "configuration where the read-only probe would itself modify the file, so "
            "whether the caller opened it read-only could not be established -- and a "
            "diagnostic must not write to a database that may have been opened read-only"
        ),
    }


@contextmanager
def private_diagnostic_connection(
    conn: Optional[sqlite3.Connection],
    *,
    access: Optional[str] = None,
) -> Iterator[Optional[sqlite3.Connection]]:
    """Yield a handle on ``conn``'s database that the caller owns outright.

    Yields ``None`` when there is no second handle to be had -- a closed store,
    or an in-memory/anonymous database, where ``sqlite3.connect`` would open a
    DIFFERENT, empty database and any check run against it would report a
    healthy index nobody looked at. Callers must turn that into ``unchecked``,
    never into a pass.

    The handle is closed on the way out, which rolls back anything still open on
    it. **Nothing the diagnostic runs on the handle touches the caller's
    transaction** -- that is the guarantee this module exists for, and it holds.
    What the caller's connection DOES see, once, before the handle is opened, is
    the access probe in ``caller_connection_access`` (pass ``access`` to skip
    it), and it is worth being exact about what that costs, because a future
    caller will decide from this paragraph whether this module is safe to use
    somewhere new:

    - it runs ``PRAGMA auto_vacuum`` and ``PRAGMA freelist_count`` (reads), and
      a ``PRAGMA incremental_vacuum`` that is inert but is still a WRITE, so it
      asks SQLite for the write lock and blocks (returning ``database is
      locked``) while another writer holds it;
    - it lowers the caller's ``busy_timeout`` to ``DIAGNOSTIC_BUSY_TIMEOUT_MS``
      for the duration and restores it, through the tree's own
      ``_temporary_sqlite_busy_timeout``. So for those ~2 s the caller's own
      lock waits are bounded at 2 s instead of its usual 30 s. Two overlapping
      users of that helper on one connection could in principle restore each
      other's value -- the other two users (``store.py``, ``engine.py``) both
      hold the store's owner lock and this probe does not, so nothing serialises
      them. Rigging the window did not reproduce a lost restore; it is recorded
      as a mechanism, not as an observed defect;
    - run inside a foreign DEFERRED read transaction, the probe upgrades that
      transaction to a write one, and its owner then holds the write lock until
      it commits. Not reachable today: every explicit ``BEGIN`` on the store and
      DAG connections is ``BEGIN IMMEDIATE``, and the only ``BEGIN DEFERRED`` in
      the tree is in a default-off subsystem on its own connection. Keep it that
      way on purpose rather than by luck.
    """
    db_path = _database_path_for_connection(conn) if conn is not None else ""
    if not db_path or db_path == ":memory:" or not Path(db_path).exists():
        yield None
        return
    # Match the caller's access. Reopening read-write what the operator opened
    # ``mode=ro`` would let a DIAGNOSTIC write to a database they deliberately
    # closed to writes, and then report that nothing needing permission
    # happened. The deep check genuinely needs a write, so on a read-only caller
    # it fails with SQLite's own readonly message and becomes ``unchecked`` --
    # which is what the read-only remedy has always told the operator.
    resolved = access if access is not None else caller_connection_access(conn)
    # Anything short of a positive "writable" opens read-only: never write to a
    # database on a guess about whether the operator closed it to writes.
    read_only = resolved != CALLER_WRITABLE
    target = f"{Path(db_path).as_uri()}?mode=ro" if read_only else db_path
    probe = sqlite3.connect(
        target,
        uri=read_only,
        timeout=DIAGNOSTIC_BUSY_TIMEOUT_MS / 1000.0,
        check_same_thread=False,
    )
    try:
        # Only the busy timeout: a diagnostic handle must not re-run the
        # journal-mode/synchronous/mmap setup a bound store owns, and the file is
        # already configured by whoever opened it.
        probe.execute(f"PRAGMA busy_timeout={DIAGNOSTIC_BUSY_TIMEOUT_MS}")
        yield probe
    finally:
        try:
            probe.rollback()
        except sqlite3.Error:  # pragma: no cover - a dead handle has nothing to undo
            pass
        probe.close()


def unavailable_check_result(spec_name: str) -> dict:
    """The honest answer when no private handle could be opened."""
    return {
        "status": "unchecked",
        "detail": (
            f"deep FTS integrity-check for '{spec_name}' was not run: this database "
            "has no second handle to open (in-memory or anonymous), and a diagnostic "
            "must not open a savepoint on the live shared connection"
        ),
    }


def run_isolated_fts_check(
    conn: Optional[sqlite3.Connection],
    spec: Any,
    check: Callable[[sqlite3.Connection, Any], dict],
) -> dict:
    """Run ``check`` on a private handle, bounded, and report honestly.

    ``check`` is passed in rather than imported here so each doctor surface keeps
    its own module-level binding of ``check_external_content_fts_integrity`` --
    that binding is the seam the existing tests patch.

    Two things end the check early, and both make it ``unchecked`` rather than a
    verdict: the wall-clock budget, and the host's interrupt bit. A check that
    was cut short has no opinion about the index -- reporting its partial state
    as ``fail`` would flag corruption that was never observed, and reporting it
    as ``pass`` is the clause this fork exists to remove.
    """
    access = caller_connection_access(conn) if conn is not None else CALLER_UNDETERMINED
    if conn is not None and access == CALLER_UNDETERMINED:
        # Refuse before opening anything: the honest answer beats a check run
        # against a database whose access mode nobody established.
        return undeterminable_access_result(spec.table_name)

    with private_diagnostic_connection(conn, access=access) as probe:
        if probe is None:
            return unavailable_check_result(spec.table_name)

        deadline = time.monotonic() + max(0.0, DIAGNOSTIC_BUDGET_SECONDS)
        stopped: list[str] = []

        def stop_reason() -> str:
            if _host_is_interrupted is not None:
                try:
                    if _host_is_interrupted():
                        return (
                            "the host interrupted this call before the deep FTS "
                            "integrity-check finished; it was stopped rather than left "
                            "holding SQLite's write lock for a caller that had given up"
                        )
                except Exception:
                    # `pass`, never `return`: returning here would skip the
                    # deadline test below, and a host signal that raised would
                    # silently restore the unbounded write-lock hold this
                    # budget exists to remove. The bound must not depend on
                    # another component behaving.
                    pass
            if time.monotonic() >= deadline:
                return (
                    f"the deep FTS integrity-check exceeded its "
                    f"{DIAGNOSTIC_BUDGET_SECONDS:.0f}s budget and was stopped; it holds "
                    "SQLite's write lock while it runs, and a longer hold would start "
                    "failing concurrent writes"
                )
            return ""

        def interrupt_if_stopped() -> int:
            if stopped:
                return 1
            reason = stop_reason()
            if reason:
                stopped.append(reason)
                return 1
            return 0

        # Before starting at all: an exhausted budget or an already-interrupted
        # call must not take the write lock even briefly.
        reason = stop_reason()
        if reason:
            return {"status": "unchecked", "detail": reason}

        probe.set_progress_handler(interrupt_if_stopped, _PROGRESS_INSTRUCTIONS)
        try:
            result = check(probe, spec)
        except Exception as exc:
            # Every way this can end is ``unchecked``, including the exotic
            # ones. An exception that escapes lands in the doctor's generic
            # handler and is reported as ``fail`` -- a corruption verdict for
            # something nobody observed.
            result = {"status": "unchecked", "detail": str(exc)}
        finally:
            # Uninstall before the handle's rollback, or the rollback is
            # interrupted too and the handle closes with the savepoint still open.
            probe.set_progress_handler(None, 0)

        if stopped:
            return {"status": "unchecked", "detail": stopped[0]}
        return result


def unchecked_fts_remedy(reason: str) -> str:
    """The remedy that would actually work, for the reason the check gave.

    A generic "rerun with read-write SQLite access" is the right advice for a
    read-only database and useless for every other cause: an operator who already
    has write access is told to do nothing, and the rerun that would succeed --
    when no write is in flight -- is never printed.
    """
    lowered = (reason or "").lower()
    if "readonly" in lowered or "read-only" in lowered:
        return (
            "rerun `/lcm doctor` with read-write SQLite access if a deep FTS "
            "integrity result is needed"
        )
    if "could not be established" in lowered:
        return (
            "empty this database's freelist (`VACUUM`, or `PRAGMA incremental_vacuum`) or "
            "set `PRAGMA auto_vacuum=NONE`, if a deep FTS integrity result is needed: "
            "until then LCM cannot tell whether this connection was opened read-only, and "
            "will not write to it to find out"
        )
    if "no second handle" in lowered or "in-memory" in lowered:
        return (
            "point LCM at a file-backed database if a deep FTS integrity result is "
            "needed: the check needs a second connection, and an in-memory or "
            "anonymous database cannot provide one"
        )
    if "interrupt" in lowered:
        return (
            "rerun `/lcm doctor` and let it finish: the deep FTS check was stopped "
            "when this call was interrupted, not because anything is wrong"
        )
    if "budget" in lowered:
        return (
            "rerun `/lcm doctor` when the database is idle: the deep FTS check ran "
            "out of its time budget, which it is given so it cannot stall concurrent "
            "writes"
        )
    if "locked" in lowered or "busy" in lowered:
        return (
            "rerun `/lcm doctor` when no write is in flight: the deep FTS check needs "
            "SQLite's write lock and another connection was holding it"
        )
    return (
        "rerun `/lcm doctor` once the reported condition has cleared if a deep FTS "
        f"integrity result is needed ({reason})"
        if reason
        else "rerun `/lcm doctor` if a deep FTS integrity result is needed"
    )
