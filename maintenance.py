"""Backup and rotate maintenance operations for the LCM store.

These are the data-layer maintenance primitives behind ``/lcm backup`` and
``/lcm rotate``: they flush the engine's SQLite connections and snapshot the
store to a timestamped or rolling backup file. They are pure functions that
take the engine so the command layer (``command.py``) keeps only the text
formatting, and the store/dag/lifecycle connection handling lives in one place.
"""

from __future__ import annotations

from datetime import datetime
import errno
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from typing import Any

from .sqlite_util import (
    _prepare_private_sqlite_file,
    _restrict_existing_sqlite_artifacts,
)


_FCHMOD = getattr(os, "fchmod", None)


def _prepare_private_backup_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    expected = os.lstat(path)
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(expected.st_mode):
        raise OSError(f"backup directory is not a real directory: {path}")

    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        path.chmod(0o700)
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            raise OSError(f"backup directory changed during validation: {path}")
        if callable(_FCHMOD):
            _FCHMOD(fd, 0o700)
        elif os.chmod in getattr(os, "supports_fd", ()):
            os.chmod(fd, 0o700)
        else:
            raise OSError("descriptor-based directory chmod is unavailable")
    finally:
        os.close(fd)

# backup helpers. A backup is the fork's last line of defence against loss,
# so it may never overwrite another backup, never share a scratch file with a concurrent
# writer, and never report success for bytes that are still only in the page cache
# (audit p05 MT01 / MT02 / MT03).

def _create_unique_backup_file(backup_dir: Path, stem: str, timestamp: str) -> Path:
    """Create an empty 0600 backup file whose name is not already taken.

    ``datetime.now()`` has one-second resolution, so two backups in the same second used to
    resolve to the same path and the second silently replaced the first. Creation is now
    exclusive: the returned path is a file this call brought into existence.
    """
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for attempt in range(1, 1000):
        suffix = "" if attempt == 1 else f"-{attempt}"
        candidate = backup_dir / f"{stem}-{timestamp}{suffix}.sqlite3"
        try:
            fd = os.open(candidate, flags, 0o600)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    unique = os.urandom(6).hex()
    candidate = backup_dir / f"{stem}-{timestamp}-{unique}.sqlite3"
    fd = os.open(candidate, flags, 0o600)
    os.close(fd)
    return candidate


# Errors that mean "this platform/filesystem does not support the operation", as opposed to a
# real I/O failure that makes the backup non-durable.
_UNSUPPORTED_FSYNC_ERRNOS = frozenset(
    value for value in (
        getattr(errno, "EINVAL", None), getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None), getattr(errno, "EPERM", None),
        getattr(errno, "EACCES", None), getattr(errno, "EISDIR", None),
        getattr(errno, "ENOSYS", None), getattr(errno, "EBADF", None),
    ) if value is not None
)


def _fsync_backup(path: Path) -> None:
    """Flush the finished snapshot and its directory entry to disk.

    Without this, ``/lcm backup`` reported a byte count for a file that only existed in the
    page cache; the crash the operator took the backup against could still lose it.
    """
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd = os.open(path.parent, directory_flags)
    except OSError as exc:  # platforms without directory descriptors
        if exc.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
        return
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        # "this platform cannot fsync a directory" and "the disk refused the
        # write" were both swallowed, so a backup whose directory entry never reached the disk
        # was reported as a durable one (round-2 verify-4 #37). Only the former is tolerated.
        if exc.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
    finally:
        os.close(directory_fd)


def flush_engine_connections(engine) -> None:
    """Commit pending writes on every SQLite connection the engine owns.

    Shared by ``backup_database`` (timestamped backup) and
    ``rotate_backup_database`` (rolling backup) so the connection-flush
    contract stays in one place.
    """
    # a flush must never commit ANOTHER operation's unfinished transaction.
    # Calling this between a node INSERT and its metadata write committed the node without its
    # sidecar, and the publisher's rollback could no longer undo it (round-3 verify-4 #2). Both
    # stores expose their write locks; take them, so a publication in flight finishes first.
    store_lock = getattr(engine._store, "_write_lock", None)
    if store_lock is not None:
        with store_lock:
            engine._store.commit()
    else:  # pragma: no cover - a store without the lock attribute
        engine._store.commit()
    dag_lock = getattr(engine._dag, "_db_lock", None)
    if dag_lock is not None:
        with dag_lock:
            engine._dag._conn.commit()
    else:  # pragma: no cover
        engine._dag._conn.commit()
    lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "_conn", None)
    if lifecycle_conn is not None:
        lifecycle_conn.commit()
    assertion_store = getattr(engine, "_assertions", None)
    if assertion_store is not None:
        # AssertionStore owns a multi-statement publication transaction. Its
        # lock-taking API must serialize this flush with publish_source() so a
        # backup cannot commit a half-written receipt behind the publisher.
        assertion_store.commit()
    query_views = getattr(engine, "_query_views", None)
    if query_views is not None:
        query_views.commit()


def backup_database(engine) -> dict[str, Any]:
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_dir = engine.backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path: Path | None = None

    try:
        _prepare_private_backup_directory(backup_dir)
        flush_engine_connections(engine)
        # exclusive creation — a second backup in the same second gets its own file
        backup_path = _create_unique_backup_file(backup_dir, db_path.stem, timestamp)
        _prepare_private_sqlite_file(backup_path)

        dest = sqlite3.connect(str(backup_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
        _restrict_existing_sqlite_artifacts(backup_path)
        _fsync_backup(backup_path)  # durable before we report success
    except (OSError, sqlite3.Error) as exc:
        # an incomplete snapshot must not be left behind looking like one.
        try:
            if backup_path is not None and backup_path.exists():
                backup_path.unlink()
        except OSError:  # pragma: no cover - best effort
            pass
        return {
            "ok": False,
            "db_path": db_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }


def rotate_backup_database(engine) -> dict[str, Any]:
    """Write a rolling rotate-latest SQLite snapshot of the LCM store.

    Atomic via tmp-then-rename so the slot is never half-written. Unlike
    ``backup_database`` which produces timestamped files, this overwrites a
    single rolling slot so disk usage stays bounded across repeated rotates.
    """
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_path = engine.rotate_backup_path()
    backup_dir = backup_path.parent
    tmp_path: Path | None = None

    try:
        _prepare_private_backup_directory(backup_dir)
        _restrict_existing_sqlite_artifacts(backup_path)
        flush_engine_connections(engine)

        # a per-call scratch file. Every rotate used to write
        # "<slot>.tmp", so two rotates running together wrote one file: the loser's snapshot
        # was replaced mid-write and the winner renamed a database another writer was still
        # filling in (audit p05 MT02).
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(backup_dir), prefix=backup_path.name + ".", suffix=".tmp"
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        _prepare_private_sqlite_file(tmp_path)
        dest = sqlite3.connect(str(tmp_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
        _restrict_existing_sqlite_artifacts(tmp_path)
        _fsync_backup(tmp_path)  # the bytes are on disk before the rename publishes them
        # Atomic replace so the rolling slot is never half-written.
        tmp_path.replace(backup_path)
        tmp_path = None
        _restrict_existing_sqlite_artifacts(backup_path)
        _fsync_backup(backup_path)  # and the rename itself is durable
    except (OSError, sqlite3.Error) as exc:
        # Best-effort cleanup of the tmp file if something failed midway.
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return {
            "ok": False,
            "db_path": db_path,
            "backup_path": backup_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }
