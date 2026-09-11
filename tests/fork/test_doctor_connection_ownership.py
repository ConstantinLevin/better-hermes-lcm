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

    text = handle_lcm_command("doctor repair apply", engine)

    assert opened, "the seam never ran; the test proved nothing"
    # the absence of a side effect is also what a repair apply that silently
    # became a no-op would produce, so pin what the command actually reported.
    assert "LCM doctor repair apply" in text
    assert "status: ok" in text or "error: FTS repair failed" in text, text
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


# ---------------------------------------------------------------------------
# an `unchecked` deep check must stay unchecked all the way to the operator
# ---------------------------------------------------------------------------
def test_doctor_repair_scan_never_reports_ok_for_a_check_that_did_not_run(tmp_path):
    """`/lcm doctor repair` exists to answer whether the index needs repair, and
    the deep check is the only thing that sees same-row-count drift. A scan whose
    deep check did not run knows nothing, and must not headline `ok`."""
    engine = _engine(tmp_path)
    conn = _insert_pending_message(engine)
    try:
        text = handle_lcm_command("doctor repair", engine)
    finally:
        conn.commit()

    assert "messages_fts_integrity_status: unchecked" in text, text
    assert "status: ok" not in text, text
    assert "messages_fts: ok" not in text, text
    assert "status: unchecked" in text, text


def test_the_unchecked_remedy_names_a_lock_rather_than_read_only_access():
    """The remedy has to match the cause. After the diagnostic moved onto its own
    handle the usual cause is a write in flight, not a read-only database, and an
    operator who already has read-write access is told to do nothing useful."""
    from hermes_lcm.diagnostics import doctor_guidance_for_check

    guidance = doctor_guidance_for_check({
        "check": "messages_fts_integrity",
        "status": "warn",
        "detail": {"status": "unchecked", "detail": "database is locked"},
    })

    assert guidance is not None
    action = guidance["operator_action"]
    assert "read-write SQLite access" not in action, action
    assert "no write" in action or "not in flight" in action or "idle" in action, action


def test_doctor_text_unchecked_action_names_the_actual_cause(tmp_path):
    engine = _engine(tmp_path)
    conn = _insert_pending_message(engine)
    try:
        text = handle_lcm_command("doctor", engine)
    finally:
        conn.commit()

    assert "messages_fts: unchecked" in text, text
    assert "read-write SQLite access" not in text, text


# ---------------------------------------------------------------------------
# the check is bounded in wall clock, not only against the lock
# ---------------------------------------------------------------------------
def test_an_over_budget_fts_check_reports_unchecked_with_the_reason(tmp_path, monkeypatch):
    from hermes_lcm import diagnostic_connection

    monkeypatch.setattr(diagnostic_connection, "DIAGNOSTIC_BUDGET_SECONDS", 0.0, raising=False)
    engine = _engine(tmp_path)

    report = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    messages_fts = next(
        check for check in report["checks"] if check["check"] == "messages_fts_integrity"
    )

    assert messages_fts["status"] == "warn", messages_fts
    detail = messages_fts["detail"]
    assert detail["status"] == "unchecked", detail
    assert "budget" in detail["detail"], detail


def test_a_host_interrupt_stops_the_fts_check_and_says_so(tmp_path, monkeypatch):
    """The host's tool deadline sets a cooperative interrupt bit on this thread
    and stops waiting. A diagnostic that keeps holding SQLite's write lock past
    that point is stalling a write path nobody is waiting on any more."""
    from hermes_lcm import diagnostic_connection

    monkeypatch.setattr(
        diagnostic_connection, "_host_is_interrupted", lambda: True, raising=False
    )
    engine = _engine(tmp_path)

    report = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    messages_fts = next(
        check for check in report["checks"] if check["check"] == "messages_fts_integrity"
    )

    assert messages_fts["status"] == "warn", messages_fts
    assert messages_fts["detail"]["status"] == "unchecked"
    assert "interrupt" in messages_fts["detail"]["detail"], messages_fts


# ---------------------------------------------------------------------------
# a partial repair must name the part that already landed
# ---------------------------------------------------------------------------
def test_repair_apply_error_names_the_repair_that_already_committed(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    def repair_messages_then_fail(conn, spec, **kwargs):
        if spec.table_name == "messages_fts":
            return {"rebuilt": True, "degraded": False, "triggers_recreated": False}
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(command_mod, "repair_external_content_fts", repair_messages_then_fail)

    text = handle_lcm_command("doctor repair apply", engine)

    assert "status: error" in text, text
    assert "messages_fts_rebuilt: yes" in text, text  # _fmt_bool renders yes/no
    assert "nodes_fts: not repaired" in text, text
    assert "PARTIAL" in text, text


# ---------------------------------------------------------------------------
# the same defect one connection over: the backup flush the doctor path reaches
# ---------------------------------------------------------------------------
def test_backup_flush_does_not_commit_a_pending_lifecycle_transaction(tmp_path):
    """`flush_engine_connections` takes the store and DAG owner locks and then
    commits the lifecycle connection with none. `prune_empty_sessions` holds
    BEGIN IMMEDIATE across a multi-row DELETE loop and relies on rollback; a
    flush from the `/lcm doctor repair apply` backup makes those deletes durable
    and the rollback a no-op."""
    from hermes_lcm.maintenance import flush_engine_connections

    engine = _engine(tmp_path)
    lifecycle = engine._lifecycle
    lifecycle.bind_session("doomed-session", conversation_id="doomed-conversation")
    assert lifecycle.get_by_conversation("doomed-conversation") is not None

    pending = threading.Event()
    release = threading.Event()

    def deleter_that_rolls_back():
        with lifecycle._lock:
            conn = lifecycle._conn
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM lcm_lifecycle_state WHERE conversation_id = ?",
                ("doomed-conversation",),
            )
            pending.set()
            release.wait(5.0)
            conn.rollback()

    holder = threading.Thread(target=deleter_that_rolls_back, name="lifecycle-pruner")
    holder.start()
    try:
        assert pending.wait(5.0), "the pruner never opened its transaction"
        flusher = threading.Thread(target=flush_engine_connections, args=(engine,))
        flusher.start()
        release.set()
        flusher.join(15)
        assert not flusher.is_alive()
    finally:
        release.set()
        holder.join(15)

    assert lifecycle.get_by_conversation("doomed-conversation") is not None, (
        "the backup flush committed a lifecycle deletion its owner rolled back"
    )


# ---------------------------------------------------------------------------
# the bound must not depend on the host signal being well-behaved
# ---------------------------------------------------------------------------
def test_the_budget_still_stops_the_check_when_the_host_signal_raises(tmp_path):
    """A raising host bit must not take the deadline down with it. If it does,
    the check reverts to the unbounded write-lock hold the budget exists to
    remove -- and it does so silently."""
    from hermes_lcm import db_bootstrap, diagnostic_connection
    from hermes_lcm.store import build_message_fts_spec

    engine = _engine(tmp_path)

    def raising_host_bit():
        raise RuntimeError("host interrupt registry unavailable")

    original_interrupt = diagnostic_connection._host_is_interrupted
    original_budget = diagnostic_connection.DIAGNOSTIC_BUDGET_SECONDS
    diagnostic_connection._host_is_interrupted = raising_host_bit
    diagnostic_connection.DIAGNOSTIC_BUDGET_SECONDS = 0.0
    try:
        result = diagnostic_connection.run_isolated_fts_check(
            engine._store.connection,
            build_message_fts_spec(),
            db_bootstrap.check_external_content_fts_integrity,
        )
    finally:
        diagnostic_connection._host_is_interrupted = original_interrupt
        diagnostic_connection.DIAGNOSTIC_BUDGET_SECONDS = original_budget

    assert result["status"] == "unchecked", result
    assert "budget" in result["detail"], result


def test_a_stopped_check_never_reaches_the_caller_as_an_exception(tmp_path, monkeypatch):
    """Whatever ends the check, the caller gets `unchecked`. An exception that
    escapes here lands in the doctor's generic handler and is reported as
    `fail` -- a corruption verdict for something nobody observed."""
    from hermes_lcm import diagnostic_connection
    from hermes_lcm.store import build_message_fts_spec

    engine = _engine(tmp_path)

    def exotic_failure(_conn, _spec):
        raise RuntimeError("the check died in an unexpected way")

    result = diagnostic_connection.run_isolated_fts_check(
        engine._store.connection, build_message_fts_spec(), exotic_failure
    )

    assert result["status"] == "unchecked", result
    assert "unexpected way" in result["detail"], result


def test_repair_apply_names_the_partial_for_a_non_sqlite_failure(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    def repair_messages_then_fail(conn, spec, **kwargs):
        if spec.table_name == "messages_fts":
            return {"rebuilt": True, "degraded": False, "triggers_recreated": False}
        raise RuntimeError("the rebuild died in an unexpected way")

    monkeypatch.setattr(command_mod, "repair_external_content_fts", repair_messages_then_fail)

    text = handle_lcm_command("doctor repair apply", engine)

    assert "status: error" in text, text
    assert "messages_fts_rebuilt: yes" in text, text
    assert "nodes_fts: not repaired" in text, text
    assert "PARTIAL" in text, text


# ---------------------------------------------------------------------------
# a caller that opened the database read-only means it
# ---------------------------------------------------------------------------
def _read_only_engine(tmp_path):
    """An engine whose store connection was deliberately opened `mode=ro`."""
    engine = _engine(tmp_path)
    db_path = str(engine._store.db_path)
    engine._store.close()
    engine._dag.close()
    engine._lifecycle.close()
    ro_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    store_db_path = db_path

    class FakeStore:
        _conn = ro_conn
        db_path = store_db_path

        @property
        def connection(self):
            return self._conn

    class FakeEngine:
        _store = FakeStore()

    return FakeEngine(), ro_conn, db_path


def test_a_read_only_caller_gets_a_read_only_diagnostic_handle(tmp_path):
    """The private handle must not reopen read-write what the operator opened
    read-only: that overrides an explicit choice, and then reports that nothing
    needing permission happened."""
    from hermes_lcm.diagnostic_connection import private_diagnostic_connection

    _fake_engine, ro_conn, _db_path = _read_only_engine(tmp_path)
    try:
        with private_diagnostic_connection(ro_conn) as probe:
            assert probe is not None
            assert probe.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
            with pytest.raises(sqlite3.OperationalError) as caught:
                probe.execute("CREATE TABLE written_anyway(a)")
            assert "readonly" in str(caught.value).lower()
    finally:
        ro_conn.close()


def test_doctor_repair_on_a_read_only_database_reports_unchecked(tmp_path):
    fake_engine, ro_conn, _db_path = _read_only_engine(tmp_path)
    try:
        text = handle_lcm_command("doctor repair", fake_engine)
    finally:
        ro_conn.close()

    assert "status: unchecked" in text, text
    assert "messages_fts: unchecked" in text, text
    assert "status: ok" not in text, text
    assert "read-write SQLite access" in text, text


# ---------------------------------------------------------------------------
# a receipt for a repair that changed nothing is a false claim
# ---------------------------------------------------------------------------
def test_repair_apply_does_not_claim_a_partial_when_nothing_was_repaired(tmp_path, monkeypatch):
    """A read-only caller fails on the FIRST index, so nothing is committed.
    Announcing a PARTIAL repair there claims a change that never happened --
    the same defect as hiding one, pointing the other way."""
    engine = _engine(tmp_path)

    def always_refused(conn, spec, **kwargs):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(command_mod, "repair_external_content_fts", always_refused)

    text = handle_lcm_command("doctor repair apply", engine)

    assert "status: error" in text, text
    assert "PARTIAL" not in text, text
    assert "no FTS tables were repaired" in text, text


# ---------------------------------------------------------------------------
# the access probe must not go silent on a database it cannot probe
# ---------------------------------------------------------------------------
def _auto_vacuum_database(tmp_path, mode: str, *, freelist: bool):
    db_path = tmp_path / f"av_{mode.lower()}_{int(freelist)}.db"
    builder = sqlite3.connect(str(db_path))
    builder.execute(f"PRAGMA auto_vacuum={mode}")
    builder.execute("VACUUM")
    builder.execute("PRAGMA journal_mode=WAL")
    builder.execute("CREATE TABLE t(a)")
    builder.executemany("INSERT INTO t VALUES(?)", [("x" * 400,)] * 800)
    builder.commit()
    if freelist:
        builder.execute("DELETE FROM t")
        builder.commit()
    builder.close()
    return db_path


def test_a_read_only_caller_is_detected_under_incremental_auto_vacuum(tmp_path):
    """auto_vacuum other than NONE used to skip the probe entirely, so a
    read-only caller read as writable and the diagnostic wrote to it."""
    from hermes_lcm.diagnostic_connection import CALLER_READ_ONLY, caller_connection_access

    db_path = _auto_vacuum_database(tmp_path, "INCREMENTAL", freelist=False)
    ro_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # `.state`: the probe carries its cause alongside the state, because
        # CALLER_UNDETERMINED has more than one producer.
        assert caller_connection_access(ro_conn).state == CALLER_READ_ONLY
    finally:
        ro_conn.close()


def test_an_undeterminable_access_mode_reports_unchecked_and_writes_nothing(tmp_path):
    """The one configuration the probe is unsafe in -- incremental auto_vacuum
    with pages on the freelist, where the probe would really free one. Refusing
    to guess is the honest answer; guessing `writable` writes to a database the
    caller may have opened read-only and then reports `ok`."""
    from types import SimpleNamespace

    from hermes_lcm.diagnostic_connection import (
        CALLER_UNDETERMINED,
        caller_connection_access,
        run_isolated_fts_check,
    )

    db_path = _auto_vacuum_database(tmp_path, "INCREMENTAL", freelist=True)
    before = db_path.read_bytes()
    conn = sqlite3.connect(str(db_path))
    ran: list[bool] = []

    def never_should_run(_conn, _spec):
        ran.append(True)
        return {"status": "pass", "detail": "ok"}

    try:
        assert caller_connection_access(conn).state == CALLER_UNDETERMINED
        result = run_isolated_fts_check(
            conn, SimpleNamespace(table_name="messages_fts"), never_should_run
        )
    finally:
        conn.close()

    assert result["status"] == "unchecked", result
    assert "read-only" in result["detail"], result
    assert not ran, "the check ran against a database whose access mode was unknown"
    assert db_path.read_bytes() == before, "the probe wrote to the database"


# ---------------------------------------------------------------------------
# the receipt must be right about WHY, not only about what
# ---------------------------------------------------------------------------
def test_the_undeterminable_remedy_is_not_the_read_only_one(tmp_path):
    """The remedy that would clear this condition is emptying the freelist. An
    operator who already has write access being told to obtain write access is
    the exact failure `unchecked_fts_remedy` was written to prevent."""
    from types import SimpleNamespace

    from hermes_lcm.diagnostic_connection import run_isolated_fts_check, unchecked_fts_remedy

    db_path = _auto_vacuum_database(tmp_path, "INCREMENTAL", freelist=True)
    conn = sqlite3.connect(str(db_path))
    try:
        result = run_isolated_fts_check(
            conn,
            SimpleNamespace(table_name="messages_fts"),
            lambda _c, _s: {"status": "pass", "detail": "ok"},
        )
    finally:
        conn.close()

    remedy = unchecked_fts_remedy(result["detail"])
    assert "read-write SQLite access" not in remedy, remedy
    assert "auto_vacuum" in remedy or "VACUUM" in remedy, remedy


def test_an_undetermined_access_names_its_own_cause(tmp_path):
    """`CALLER_UNDETERMINED` has two producers. A probe that failed for some
    other reason must not print the freelist diagnosis, which is not its cause."""
    from hermes_lcm.diagnostic_connection import CALLER_UNDETERMINED, caller_connection_access

    class BrokenConnection:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("the access probe blew up")

    access = caller_connection_access(BrokenConnection())

    assert access.state == CALLER_UNDETERMINED
    assert "freelist" not in access.reason, access.reason
    assert "blew up" in access.reason, access.reason


def test_a_database_with_no_second_handle_is_told_that_and_not_about_its_freelist(tmp_path):
    """The no-second-handle answer comes first: an in-memory database cannot be
    reopened at all, whatever its auto_vacuum setting says."""
    from types import SimpleNamespace

    from hermes_lcm.diagnostic_connection import run_isolated_fts_check

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("VACUUM")
        conn.execute("CREATE TABLE t(a)")
        conn.executemany("INSERT INTO t VALUES(?)", [("x" * 400,)] * 800)
        conn.execute("DELETE FROM t")
        conn.commit()
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] > 0
        result = run_isolated_fts_check(
            conn,
            SimpleNamespace(table_name="messages_fts"),
            lambda _c, _s: {"status": "pass", "detail": "ok"},
        )
    finally:
        conn.close()

    assert result["status"] == "unchecked", result
    assert "second handle" in result["detail"], result
    assert "freelist" not in result["detail"], result


def test_repair_apply_on_a_writable_database_never_blames_read_only(tmp_path):
    """Round 4 opened the handle read-only whenever the access mode could not be
    determined -- right -- and then reported SQLite's `readonly` refusal as if
    the operator's database were read-only. It is not; LCM declined to find out."""
    engine = _engine(tmp_path)
    conn = engine._store.connection
    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
    conn.execute("VACUUM")
    engine._store.append_batch(
        "doctor-session", [{"role": "user", "content": "x" * 4000}] * 200, [1] * 200
    )
    conn.execute("DELETE FROM messages WHERE store_id > 1")
    conn.commit()
    assert conn.execute("PRAGMA freelist_count").fetchone()[0] > 0, "no freelist to trip the probe"

    text = handle_lcm_command("doctor repair apply", engine)

    assert "status: error" in text, text
    assert "readonly" not in text.lower(), text
    assert "could not determine" in text.lower(), text
    assert "PARTIAL" not in text, text
