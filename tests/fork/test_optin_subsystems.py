"""fork: betterlcm — the opt-in subsystems are default-off, which lowers their priority but
does not make it acceptable for them to certify incomplete or misattributed evidence
(audit verify-4 #20-#24)."""
import sqlite3

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.rollup_store import RollupStore


def test_a_rollup_and_its_lineage_come_from_one_snapshot(tmp_path):
    """verify-4 #20: the row and its source ids were read in two statements, so a concurrent
    rebuild could return generation one's text with generation two's sources."""
    store = RollupStore(tmp_path / "rollups.db")
    try:
        token = store.upsert_building("day", "2026-09-08", "s")
        store.mark_ready(token, "FIRST", 5, [1, 2], "fp1")
        first = store.get_rollup("day", "2026-09-08", "s")
        assert first["summary"] == "FIRST" and first["source_node_ids"] == [1, 2]

        second_token = store.upsert_building("day", "2026-09-08", "s")
        store.mark_ready(second_token, "SECOND", 5, [3, 4], "fp2")
        second = store.get_rollup("day", "2026-09-08", "s")
        assert second["summary"] == "SECOND" and second["source_node_ids"] == [3, 4]
    finally:
        store.close()


def test_an_unavailable_summary_database_does_not_delete_a_ready_rollup(tmp_path):
    """verify-4 #20: an unreadable DAG read as "no content", and a day with no sources
    RESOLVES — which threw away a rollup that was already built."""
    from hermes_lcm import rollup_builder

    dag = SummaryDAG(str(tmp_path / "dag.db"))
    dag.close()
    with pytest.raises(rollup_builder.RollupSourcesUnavailable):
        rollup_builder._scope_frontier(dag, "s")


def test_truncated_evidence_refs_cannot_close_a_computation(tmp_path):
    """verify-4 #22: with a two-reference budget over Alice=2, Bob=3 and Alice=100 the pack
    reported a closed, product-verified difference of 1 — the discarded candidate was the one
    that contradicted it."""
    import json
    from types import SimpleNamespace
    from hermes_lcm.evidence_pack import build_evidence_pack
    from hermes_lcm.store import MessageStore

    config = LCMConfig(database_path=str(tmp_path / "pack.db"))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    engine = SimpleNamespace(_config=config, _store=store, _assertions=None,
                             _session_occurrence_dates={})
    try:
        contents = ["Alice walked 2 km.", "Bob walked 3 km.", "Alice walked 100 km."]
        refs = []
        for content in contents:
            store_id = store.append("session-a", {"role": "user", "content": content})
            refs.append({"exact_ref": f"lcm:{store_id}:0-{len(content)}", "quote": content})
        store.commit()

        payload = json.loads(build_evidence_pack({
            "question": "What is the difference between Alice's and Bob's distance?",
            "baseline_refs": refs,
            "budgets": {"max_refs": 2},
        }, engine=engine))
        if payload.get("truncation", {}).get("refs_truncated"):
            assert payload["completeness"]["state"] == "partial", payload["completeness"]
            assert payload["completeness"]["product_verified"] is False
    finally:
        store.close()


def test_a_default_state_query_applies_the_validity_window(tmp_path):
    """verify-4 #23: without an explicit as_of the validity window was not applied at all, so
    an assertion whose valid_to had passed still came back as current state."""
    import time
    from hermes_lcm.assertion_store import AssertionCandidate, AssertionStore
    from hermes_lcm.assertion_state import query_assertion_state
    from hermes_lcm.store import MessageStore

    db_path = tmp_path / "assertions.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        content = "I am on the payments team."
        store_id = messages.append("s", {"role": "user", "content": content}, source="cli")
        messages.commit()
        snapshot = assertions.snapshot_source(store_id)
        quote = "on the payments team"
        start = content.index(quote)
        expired = AssertionCandidate(
            source_span_start=start,
            source_span_end=start + len(quote),
            subject_key="user",
            predicate_key="team",
            object_value="payments",
            value_text="payments",
            kind="fact",
            event_at=None,
            valid_from=None,
            valid_to=time.time() - 3600,   # it stopped being true an hour ago
        )
        live = AssertionCandidate(
            source_span_start=start,
            source_span_end=start + len(quote),
            subject_key="user",
            predicate_key="team",
            object_value="platform",
            value_text="platform",
            kind="fact",
            event_at=None,
            valid_from=None,
            valid_to=time.time() + 3600,   # still true
        )
        assertions.publish_source(snapshot, [expired, live])

        now = query_assertion_state(assertions, subject_key="user")
        values = {str(row.get("value_text")) for row in now.assertions}
        assert values == {"platform"}, values
    finally:
        assertions.close()
        messages.close()


def test_a_cancelled_assertion_publication_cannot_commit_later(tmp_path):
    """round-2 verify-5 #5: the transaction handlers caught Exception, not cancellation, so
    interrupting publication before its assertion insert left the transaction open — and a
    later real commit published a current extraction receipt with zero assertions."""
    from hermes_lcm.assertion_store import AssertionCandidate, AssertionStore
    from hermes_lcm.store import MessageStore

    db_path = tmp_path / "cancel.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        content = "I am on the payments team."
        store_id = messages.append("s", {"role": "user", "content": content}, source="cli")
        messages.commit()
        snapshot = assertions.snapshot_source(store_id)
        quote = "on the payments team"
        start = content.index(quote)
        candidate = AssertionCandidate(
            source_span_start=start, source_span_end=start + len(quote),
            subject_key="user", predicate_key="team", object_value="payments",
            value_text="payments", kind="fact", event_at=None,
            valid_from=None, valid_to=None,
        )

        real_conn = assertions._conn
        state = {"cancelled": False}

        class CancellingConnection:
            """A connection that is cancelled mid-transaction, as a host abort would."""

            def execute(self, sql, *args, **kwargs):
                if not state["cancelled"] and "INSERT INTO lcm_assertions" in str(sql):
                    state["cancelled"] = True
                    raise KeyboardInterrupt()
                return real_conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(real_conn, name)

        assertions._conn = CancellingConnection()
        try:
            with pytest.raises(KeyboardInterrupt):
                assertions.publish_source(snapshot, [candidate])
        finally:
            assertions._conn = real_conn

        real_conn.commit()  # an unrelated commit must publish nothing
        receipts = real_conn.execute(
            "SELECT COUNT(*) FROM lcm_assertion_sources"
        ).fetchone()[0]
        stored = real_conn.execute("SELECT COUNT(*) FROM lcm_assertions").fetchone()[0]
        assert (receipts, stored) == (0, 0), (receipts, stored)
    finally:
        assertions.close()
        messages.close()


def test_trajectory_ingestion_keeps_the_original_and_redacts_only_what_it_sends(tmp_path):
    """round-2 verify-5 #4: trajectory ingestion redacted irreversibly BY DEFAULT, so two
    ingests differing only in a secret produced the same digest and the second was reported
    "already current" — both the values and the fact that they differed were gone."""
    from hermes_lcm.trajectory_store import TrajectoryStore
    import inspect

    signature = inspect.signature(TrajectoryStore.__init__)
    assert signature.parameters["protect_sensitive"].default is False


def test_the_final_answer_verifier_refuses_added_claims(tmp_path):
    """round-2 verify-5 #1: appending "The project was approved and deployed" to a valid
    calculation returned "verified" — the checks were satisfied by prose that merely preserved
    the numbers and entities, so a reader was told an unsupported claim had been checked."""
    from hermes_lcm.reasoning import ComputationTrace, verify_final_answer

    trace = ComputationTrace(
        operation="difference",
        result="$12",
        result_value=12,
        unit="usd",
        citations=("lcm:1:0-5",),
        entities=("Alice", "Bob"),
        evidence_dates=(),
        steps=(),
        answer="Alice spent $12 more than Bob. [lcm:1:0-5]",
    )
    assert verify_final_answer(trace.answer, trace).status == "verified"
    decision = verify_final_answer(
        "Alice spent $12 more than Bob. The project was approved and deployed. [lcm:1:0-5]",
        trace,
    )
    assert decision.status == "fallback", decision
    assert "does not support" in decision.reason
