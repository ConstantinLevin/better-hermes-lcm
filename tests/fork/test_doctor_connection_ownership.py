"""A diagnostic must never own a transaction it did not open.

``check_external_content_fts_integrity`` wraps its probe INSERT in
``SAVEPOINT`` / ``ROLLBACK TO`` / ``RELEASE``, and a savepoint belongs to the
CONNECTION, not to the thread that opened it. Every doctor surface handed the
store's or the DAG's live shared connection could therefore roll back rows
another thread wrote on it AFTER the savepoint opened -- an ingest that reported
success and advanced its cursor loses its originals -- and the explicit repair,
which calls ``conn.commit()`` on the same shared connection, could publish a
batch its writer had not finished (issue #10).

Note what does NOT reproduce it: a write that is already pending when the
diagnostic starts survives, because ``ROLLBACK TO`` only unwinds as far as the
savepoint. The hazard needs the foreign write to land INSIDE the savepoint
window, which is what the threaded test here arranges and what the host's
tool-deadline probe drives end to end.
"""
import json
import sqlite3
import threading

import pytest

from hermes_lcm import command as command_mod
from hermes_lcm import tools as lcm_tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


PENDING = "PENDING-ROW-FROM-A-CONCURRENT-WRITER"
_TRANSACTION_CONTROL = ("BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "END")


@pytest.fixture(autouse=True)
def _fast_diagnostic_backoff(monkeypatch):
    """Do not spend the real lock-acquisition budget waiting on these fixtures.

    Some tests here deliberately hold a live write transaction open on the
    shared connection, so the diagnostic's private handle cannot take the write
    lock and waits out its whole backoff. That backoff is a wall-clock
    preference; what is under test is what the diagnostic does when it loses the
    race, which is identical at 50 ms.
    """
    try:
        from hermes_lcm import diagnostic_connection
    except ImportError:  # the module under construction: let tests fail on behaviour
        return
    monkeypatch.setattr(diagnostic_connection, "DIAGNOSTIC_BUSY_TIMEOUT_MS", 50)


class _Watched:
    """Pass-through sqlite3 connection wrapper: records verbs, changes nothing."""

    def __init__(self, real, hooks=None, on_commit=None):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_hooks", hooks)
        object.__setattr__(self, "_on_commit", on_commit)
        object.__setattr__(self, "verbs", [])

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        # `MessageStore._append_protected_batch` is `with self._write_lock,
        # self._conn:` -- this __exit__ IS its commit, and the window between the
        # INSERTs and it is where a foreign ROLLBACK TO does the damage.
        if self._on_commit is not None:
            self._on_commit()
        return self._real.__exit__(exc_type, exc, tb)

    def _record(self, verb):
        self.verbs.append(verb)

    def execute(self, sql, *args, **kwargs):
        text = " ".join(str(sql).split()).upper()
        for keyword in _TRANSACTION_CONTROL:
            if text.startswith(keyword):
                self._record(text)
                break
        if self._hooks is not None:
            return self._hooks(self, text, sql, args, kwargs)
        return self._real.execute(sql, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._real.executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self._real.executescript(*args, **kwargs)

    def cursor(self, *args, **kwargs):
        return self._real.cursor(*args, **kwargs)

    def commit(self):
        self._record("commit()")
        return self._real.commit()

    def rollback(self):
        self._record("rollback()")
        return self._real.rollback()

    def close(self):
        return self._real.close()


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "doctor-session"
    engine._session_platform = "cli"
    engine._conversation_id = "doctor-session"
    engine.update_model("model-under-test", 262144)
    engine._store.append(
        "doctor-session", {"role": "user", "content": "already stored"}, token_estimate=2
    )
    return engine


def _watch_live_connections(engine):
    """Watch both live connections the way a concurrent writer's owner would."""
    engine._store._conn = _Watched(engine._store._conn)
    engine._dag._conn = _Watched(engine._dag._conn)
    return engine._store._conn, engine._dag._conn


def _insert_pending_message(engine) -> sqlite3.Connection:
    """Open the write transaction a concurrent ingest would have open."""
    conn = engine._store.connection
    conn.execute(
        "INSERT INTO messages(session_id, role, content, timestamp, token_estimate,"
        " pinned, ingested_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
        ("doctor-session", "user", PENDING, 1.0, 3, 0, 1.0),
    )
    assert conn.in_transaction, "the fixture did not open a write transaction"
    return conn


def _durable_message_count(engine, content=PENDING) -> int:
    """Count committed rows through a handle none of the parties owns."""
    probe = sqlite3.connect(str(engine._store.db_path), timeout=5.0)
    try:
        return int(
            probe.execute(
                "SELECT COUNT(*) FROM messages WHERE content = ?", (content,)
            ).fetchone()[0]
        )
    finally:
        probe.close()


# ---------------------------------------------------------------------------
# the interleaving the issue is about
# ---------------------------------------------------------------------------
def test_a_write_inside_the_doctors_savepoint_window_survives(tmp_path, monkeypatch):
    """The foreign INSERT lands between the diagnostic's SAVEPOINT and its
    ROLLBACK TO, and commits afterwards -- the ordering the host's abandoned
    tool worker produces. On a shared connection ROLLBACK TO discards it and the
    writer's commit publishes nothing."""
    engine = _engine(tmp_path)
    store_conn = engine._store.connection
    savepoint_open = threading.Event()
    writer_inserted = threading.Event()
    diagnostic_released = threading.Event()

    def hooks(watched, text, sql, args, kwargs):
        if text.startswith("SAVEPOINT") and "LCM_FTS_INTEGRITY" in text:
            result = watched._real.execute(sql, *args, **kwargs)
            savepoint_open.set()
            return result
        if text.startswith("ROLLBACK TO") and not diagnostic_released.is_set():
            writer_inserted.wait(1.0)
            return watched._real.execute(sql, *args, **kwargs)
        if text.startswith("RELEASE") and not diagnostic_released.is_set():
            result = watched._real.execute(sql, *args, **kwargs)
            diagnostic_released.set()
            return result
        return watched._real.execute(sql, *args, **kwargs)

    def hold_the_writers_commit():
        writer_inserted.set()
        diagnostic_released.wait(5.0)

    # Watch wherever the SAVEPOINT is opened: on the live connection today, on
    # the diagnostic's own handle once it has one.
    engine._store._conn = _Watched(store_conn, hooks, hold_the_writers_commit)
    try:
        from hermes_lcm import diagnostic_connection
    except ImportError:
        pass
    else:
        real_connect = diagnostic_connection.sqlite3.connect

        class _Shim:
            def __getattr__(self, name):
                return getattr(sqlite3, name)

            @staticmethod
            def connect(*a, **kw):
                return _Watched(real_connect(*a, **kw), hooks)

        monkeypatch.setattr(diagnostic_connection, "sqlite3", _Shim())

    def writer():
        savepoint_open.wait(3.0)
        engine._store.append_batch(
            "doctor-session",
            [{"role": "user", "content": PENDING}],
            [3],
        )

    thread = threading.Thread(target=writer, name="probe-writer")
    thread.start()
    try:
        lcm_tools.lcm_doctor({}, engine=engine)
    finally:
        thread.join(30)

    assert not thread.is_alive()
    assert _durable_message_count(engine) == 1, (
        "the diagnostic rolled back a row a concurrent writer had committed"
    )


# ---------------------------------------------------------------------------
# the contract, stated once per doctor surface
# ---------------------------------------------------------------------------
def test_doctor_tool_issues_no_transaction_control_on_the_live_connections(tmp_path):
    engine = _engine(tmp_path)
    store_conn, dag_conn = _watch_live_connections(engine)

    lcm_tools.lcm_doctor({}, engine=engine)

    assert store_conn.verbs == [], f"lcm_doctor drove the store's transaction: {store_conn.verbs}"
    assert dag_conn.verbs == [], f"lcm_doctor drove the DAG's transaction: {dag_conn.verbs}"


def test_doctor_text_issues_no_transaction_control_on_the_live_connections(tmp_path):
    engine = _engine(tmp_path)
    store_conn, dag_conn = _watch_live_connections(engine)

    handle_lcm_command("doctor", engine)

    assert store_conn.verbs == [], f"/lcm doctor drove the store's transaction: {store_conn.verbs}"
    assert dag_conn.verbs == [], f"/lcm doctor drove the DAG's transaction: {dag_conn.verbs}"


def test_doctor_repair_scan_issues_no_transaction_control_on_the_live_connection(tmp_path):
    engine = _engine(tmp_path)
    store_conn, _dag_conn = _watch_live_connections(engine)

    handle_lcm_command("doctor repair", engine)

    assert store_conn.verbs == [], (
        f"/lcm doctor repair drove the store's transaction: {store_conn.verbs}"
    )


def test_doctor_repair_apply_issues_no_transaction_control_after_its_backup(
    tmp_path, monkeypatch
):
    """The backup's flush is the store owner's own commit, taken under the
    store write lock, and stays. Everything the repair does afterwards must
    leave the live connection alone."""
    engine = _engine(tmp_path)
    store_conn, _dag_conn = _watch_live_connections(engine)
    real_join = command_mod.join_background_integrity_scans

    def join_then_start_watching(*args, **kwargs):
        real_join(*args, **kwargs)
        store_conn.verbs.clear()

    monkeypatch.setattr(
        command_mod, "join_background_integrity_scans", join_then_start_watching
    )

    handle_lcm_command("doctor repair apply", engine)

    assert store_conn.verbs == [], (
        f"/lcm doctor repair apply drove the store's transaction: {store_conn.verbs}"
    )


def test_doctor_repair_apply_does_not_commit_a_pending_store_write(tmp_path, monkeypatch):
    """`repair apply` flushes the store for its backup, so the concurrent write
    has to arrive AFTER that flush. `join_background_integrity_scans` is the last
    thing it does before touching the FTS tables, so it is the seam that puts a
    writer's open transaction exactly where the repair would adopt it."""
    engine = _engine(tmp_path)
    real_join = command_mod.join_background_integrity_scans
    opened: list[sqlite3.Connection] = []

    def join_then_a_writer_starts(*args, **kwargs):
        real_join(*args, **kwargs)
        opened.append(_insert_pending_message(engine))

    monkeypatch.setattr(
        command_mod, "join_background_integrity_scans", join_then_a_writer_starts
    )

    handle_lcm_command("doctor repair apply", engine)

    assert opened, "the seam never ran; the test proved nothing"
    opened[0].rollback()
    assert _durable_message_count(engine) == 0, (
        "repair apply committed a transaction its writer had not finished"
    )


# ---------------------------------------------------------------------------
# a check that could not run must say so
# ---------------------------------------------------------------------------
def test_doctor_tool_never_reports_pass_for_a_check_it_could_not_run(tmp_path):
    engine = _engine(tmp_path)
    conn = _insert_pending_message(engine)
    try:
        report = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    finally:
        conn.commit()

    messages_fts = next(
        check for check in report["checks"] if check["check"] == "messages_fts_integrity"
    )
    assert messages_fts["status"] == "warn", (
        "a deep FTS check that could not take the write lock was reported as a "
        f"completed one: {messages_fts}"
    )
    assert report["overall"] != "healthy"


def test_a_diagnostic_handle_is_refused_for_a_database_that_cannot_be_reopened():
    """An in-memory database has no second handle: a private connection to
    ``:memory:`` is a DIFFERENT, empty database, and a check that ran against it
    would report a healthy index nobody looked at."""
    from hermes_lcm.diagnostic_connection import private_diagnostic_connection

    conn = sqlite3.connect(":memory:")
    try:
        with private_diagnostic_connection(conn) as probe:
            assert probe is None
    finally:
        conn.close()


def test_a_diagnostic_handle_is_a_different_connection_on_the_same_database(tmp_path):
    from hermes_lcm.diagnostic_connection import private_diagnostic_connection

    engine = _engine(tmp_path)
    conn = engine._store.connection
    with private_diagnostic_connection(conn) as probe:
        assert probe is not None
        assert probe is not conn
        assert (
            probe.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        ), "the private handle did not open the same database"
        probe.execute("SAVEPOINT probe_owns_this")
        probe.execute("ROLLBACK TO probe_owns_this")
        probe.execute("RELEASE probe_owns_this")
    assert not conn.in_transaction, "the diagnostic left the live connection in a transaction"
