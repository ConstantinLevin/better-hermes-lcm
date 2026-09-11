"""What the beta's doctor-concurrency fix must achieve (#10), written before it exists (#19).

`check_external_content_fts_integrity` wraps its FTS5 integrity INSERT in a SAVEPOINT on the
connection the doctor was handed — and the doctor is handed the live store connection. A savepoint
belongs to the CONNECTION, not to the thread that opened it, so its `ROLLBACK TO` undoes whatever
any other thread wrote on that connection in the meantime.

The host makes that concurrency reachable: a tool call that passes its deadline returns a timeout
error to the model and sets a cooperative interrupt bit, but does not stop the worker. The FTS
check never reads that bit, so a doctor the host has already given up on is still holding an open
savepoint while the next turn's `post_llm_call` ingests new history through the same engine.

The test FAILS on the tree it was written against. Measured there: `append()` returns a store_id
and reports success, the doctor reports `messages_fts_integrity: pass`, and the row is gone.
"""
import json
import threading

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


class _SteppableConnection:
    """The live connection with a barrier between statements, not inside one.

    The barrier has to sit between `conn.execute` calls: Python's sqlite3 holds the connection
    for the duration of one, so pausing inside a trace callback would block the other thread
    on the connection rather than letting it interleave — which would test the driver's lock
    instead of the savepoint's ownership.
    """

    def __init__(self, real, after_statement):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_after", after_statement)

    def execute(self, sql, *args, **kwargs):
        real = object.__getattribute__(self, "_real")
        try:
            return real.execute(sql, *args, **kwargs)
        finally:
            object.__getattribute__(self, "_after")(" ".join(str(sql).split()))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


@pytest.mark.beta_target("#10")
def test_a_paused_doctor_cannot_roll_back_a_successful_concurrent_ingest(tmp_path):
    """A diagnostic that was abandoned must not be able to undo somebody else's committed work.

    The assertion is deliberately about the OUTCOME rather than the mechanism: whether the fix
    takes the store's owner lock, moves the check to its own connection, or makes it honour the
    interrupt, the row an ingest reported as written has to still be there afterwards.

    The interleaving below is exactly the one the host produces — doctor still inside its
    savepoint, ingest inserts, doctor rolls back, ingest commits and reports success. Every wait
    is bounded, and a fix that serialises the two simply makes the ingest arrive after the
    doctor: that path is taken too, and ends at the same assertion.
    """
    config = LCMConfig(database_path=str(tmp_path / "doctor10.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("doctor10", platform="cli", context_length=262_144)
        engine._store.append("doctor10", {"role": "user", "content": "seed"}, source="cli")
        engine._store.commit()

        doctor_in_savepoint = threading.Event()
        doctor_may_resume = threading.Event()
        doctor_finished = threading.Event()
        ingest_interleaved = threading.Event()
        armed = threading.Event()
        already_hooked = threading.Event()

        def after_statement(sql: str) -> None:
            is_doctor = threading.current_thread().name == "lcm-doctor"
            if "integrity-check" in sql and is_doctor:
                doctor_in_savepoint.set()
                doctor_may_resume.wait(20)
                return
            if (armed.is_set() and not is_doctor and not already_hooked.is_set()
                    and "INSERT INTO MESSAGES" in sql.upper()):
                # the row exists but is not committed: this is the instant at which the
                # doctor's ROLLBACK TO can take it away
                already_hooked.set()
                ingest_interleaved.set()
                doctor_may_resume.set()
                doctor_finished.wait(20)

        engine._store._conn = _SteppableConnection(engine._store._conn, after_statement)

        doctor_result: dict = {}
        ingest_result: dict = {}

        def run_doctor():
            try:
                doctor_result["payload"] = engine.handle_tool_call("lcm_doctor", {})
            finally:
                doctor_finished.set()

        def run_ingest():
            ingest_result["store_id"] = engine._store.append(
                "doctor10", {"role": "user", "content": "CONCURRENT_SURVIVOR"}, source="cli")

        doctor = threading.Thread(target=run_doctor, name="lcm-doctor")
        doctor.start()
        doctor_in_savepoint.wait(20)  # may not fire at all once the check owns its own connection
        armed.set()
        ingest = threading.Thread(target=run_ingest, name="lcm-ingest")
        ingest.start()
        if not ingest_interleaved.wait(5):
            # the two were serialised, or there is no shared savepoint left to interleave with
            doctor_may_resume.set()
        ingest.join(30)
        doctor_may_resume.set()
        doctor.join(30)
        assert not doctor.is_alive() and not ingest.is_alive(), "the probe did not finish"

        assert ingest_result.get("store_id"), "the ingest did not report a successful write"
        contents = [row["content"] for row in engine._store.get_session_messages("doctor10")]
        assert "CONCURRENT_SURVIVOR" in contents, (
            "a row an ingest reported as written was rolled back by the doctor's savepoint; "
            f"doctor said {json.loads(doctor_result.get('payload') or '{}').get('status')!r}"
        )
    finally:
        engine.shutdown()
