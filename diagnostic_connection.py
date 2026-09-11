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
finally gets there, and SQLite's single-writer rule -- not a Python lock another
thread has to wait on forever -- is what serializes it against live writes.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .db_bootstrap import _database_path_for_connection

# How long the diagnostic waits for SQLite's write lock before giving up.
# It is not the fork's window-scaled family: nothing about a lock-acquisition
# backoff depends on the size of the context window. It is short on purpose --
# the diagnostic yields to the live write path rather than queueing behind it,
# and a check that could not take the lock reports ``unchecked`` with SQLite's
# own message, which is true, instead of holding the tool call open for the
# 30 s a store connection is willing to wait.
DIAGNOSTIC_BUSY_TIMEOUT_MS = 2_000


@contextmanager
def private_diagnostic_connection(
    conn: Optional[sqlite3.Connection],
) -> Iterator[Optional[sqlite3.Connection]]:
    """Yield a handle on ``conn``'s database that the caller owns outright.

    Yields ``None`` when there is no second handle to be had -- a closed store,
    or an in-memory/anonymous database, where ``sqlite3.connect`` would open a
    DIFFERENT, empty database and any check run against it would report a
    healthy index nobody looked at. Callers must turn that into ``unchecked``,
    never into a pass.

    The handle is closed on the way out, which rolls back anything still open on
    it. Nothing on the caller's connection is touched at any point.
    """
    db_path = _database_path_for_connection(conn) if conn is not None else ""
    if not db_path or db_path == ":memory:" or not Path(db_path).exists():
        yield None
        return
    probe = sqlite3.connect(
        db_path, timeout=DIAGNOSTIC_BUSY_TIMEOUT_MS / 1000.0, check_same_thread=False
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
