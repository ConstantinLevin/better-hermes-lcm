"""fork: betterlcm — the backup is the last line of defence against loss, so it must never
overwrite another backup, never share a scratch file with a concurrent rotate, and never
report success for bytes that are still only in the page cache (audit p05 MT01/MT02/MT03)."""
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import maintenance
from hermes_lcm.store import MessageStore


def _engine(tmp_path):
    store = MessageStore(tmp_path / "db" / "lcm.db")
    store.append("s", {"role": "user", "content": "backup"}, source="cli")
    store.commit()
    backup_dir = tmp_path / "backups"
    return SimpleNamespace(
        _store=store,
        _dag=SimpleNamespace(_conn=store._conn),
        backup_dir=lambda: backup_dir,
        rotate_backup_path=lambda: backup_dir / "rotate-latest.sqlite3",
    ), store, backup_dir


def test_two_backups_in_the_same_second_do_not_overwrite_each_other(tmp_path, monkeypatch):
    engine, store, backup_dir = _engine(tmp_path)
    try:
        frozen = type("_Clock", (), {"now": staticmethod(lambda: _Frozen())})

        class _Frozen:
            def strftime(self, _fmt):
                return "20260908_120000"

        monkeypatch.setattr(maintenance, "datetime", frozen)
        first = maintenance.backup_database(engine)
        store.append("s", {"role": "user", "content": "later"}, source="cli")
        store.commit()
        second = maintenance.backup_database(engine)

        assert first["ok"] and second["ok"]
        assert first["backup_path"] != second["backup_path"], "the first backup was overwritten"
        assert first["backup_path"].exists() and second["backup_path"].exists()
        with sqlite3.connect(first["backup_path"]) as restored:
            assert restored.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        with sqlite3.connect(second["backup_path"]) as restored:
            assert restored.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    finally:
        store.close()


def test_a_failed_backup_leaves_no_file_that_looks_like_a_snapshot(tmp_path, monkeypatch):
    engine, store, backup_dir = _engine(tmp_path)
    try:
        def fail_backup(_dest):
            raise sqlite3.Error("synthetic backup failure")

        monkeypatch.setattr(store, "backup", fail_backup)
        result = maintenance.backup_database(engine)
        assert result["ok"] is False
        assert list(backup_dir.glob("*.sqlite3")) == [], "an empty file was left behind"
    finally:
        store.close()


def test_rotate_uses_a_private_scratch_file_per_call(tmp_path, monkeypatch):
    engine, store, backup_dir = _engine(tmp_path)
    seen: list[str] = []
    real_mkstemp = maintenance.tempfile.mkstemp

    def record(**kwargs):
        fd, name = real_mkstemp(**kwargs)
        seen.append(name)
        return fd, name

    try:
        monkeypatch.setattr(maintenance.tempfile, "mkstemp", record)
        assert maintenance.rotate_backup_database(engine)["ok"] is True
        assert maintenance.rotate_backup_database(engine)["ok"] is True
        assert len(set(seen)) == 2, "two rotates shared one scratch file"
        assert not list(backup_dir.glob("*.tmp")), "scratch files leaked"
    finally:
        store.close()


def test_backup_is_fsynced_before_success_is_reported(tmp_path, monkeypatch):
    engine, store, backup_dir = _engine(tmp_path)
    synced: list[int] = []
    real_fsync = os.fsync
    try:
        monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
        result = maintenance.backup_database(engine)
        assert result["ok"] is True
        # the snapshot itself and its directory entry
        assert len(synced) >= 2, "success was reported for cached-only bytes"
    finally:
        store.close()


def test_the_suite_cannot_reach_live_storage(tmp_path):
    """fork: betterlcm (audit E, E07) — an engine built without an explicit database_path must
    resolve inside the test home, never $HOME/.hermes of the live account."""
    import os
    from pathlib import Path
    from hermes_lcm.config import LCMConfig

    hermes_home = os.environ.get("HERMES_HOME", "")
    assert hermes_home and "lcm-tests-home-" in hermes_home
    assert "lcm-tests-home-" in str(Path.home())
    # plugin configuration is scrubbed; the harness's own LCM_TESTS_* controls survive
    assert not [
        name for name in os.environ
        if name.startswith("LCM_") and not name.startswith("LCM_TESTS_")
    ]
    default = LCMConfig()
    assert default.database_path == "" or "lcm-tests-home-" in default.database_path


def test_a_directory_fsync_that_really_fails_is_not_reported_as_a_durable_backup(tmp_path, monkeypatch):
    """round-2 verify-4 #37: every directory-fsync error was treated as "this platform does not
    support it", so an EIO — the disk refusing the write the operator took the backup FOR —
    was swallowed and the backup reported as durable."""
    import errno
    import os as _os
    from hermes_lcm import maintenance

    target = tmp_path / "snapshot.sqlite3"
    target.write_text("data", encoding="utf-8")
    real_fsync = _os.fsync
    calls = {"n": 0}

    def failing_fsync(fd):
        calls["n"] += 1
        if calls["n"] == 1:  # the file itself succeeds
            return real_fsync(fd)
        raise OSError(errno.EIO, "input/output error")

    monkeypatch.setattr(maintenance.os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        maintenance._fsync_backup(target)

    def unsupported_fsync(fd):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_fsync(fd)
        raise OSError(errno.EINVAL, "not supported here")

    calls["n"] = 0
    monkeypatch.setattr(maintenance.os, "fsync", unsupported_fsync)
    maintenance._fsync_backup(target)  # tolerated, as before
