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
    for candidate in (
        "Alice spent $12 more than Bob. The project was approved and deployed. [lcm:1:0-5]",
        # round-3: naming a grounded entity does not make a claim supported
        "Alice spent $12 more than Bob. Alice authorized fraud. [lcm:1:0-5]",
        # ... nor does joining it with a comma
        "Alice spent $12 more than Bob, and Alice authorized fraud. [lcm:1:0-5]",
    ):
        decision = verify_final_answer(candidate, trace)
        assert decision.status == "fallback", (candidate, decision)
        assert "own answer" in decision.reason


def test_grounding_refuses_a_value_the_quote_does_not_attribute(tmp_path):
    """round-3 verify-4 #18/#19: presence in the quote was treated as attribution, so a span
    saying "Alice paid 10 USD; Bob paid 30 USD" grounded Alice=30 and the computation attached
    genuine citations to a relationship they do not support. A quote can also state a number in
    order to DENY it."""
    from hermes_lcm.reasoning import ground_evidence, question_date_as_of_epoch
    from hermes_lcm.assertion_store import AssertionStore
    from hermes_lcm.store import MessageStore

    db_path = tmp_path / "grounding.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        both = "Alice paid 10 USD; Bob paid 30 USD."
        store_id = messages.append("s", {"role": "user", "content": both}, source="cli")
        denial = "Atlas did not cost 10 USD."
        denial_id = messages.append("s", {"role": "user", "content": denial}, source="cli")
        messages.commit()
        as_of = question_date_as_of_epoch("2099-12-31")

        misattributed = ground_evidence(
            [{"store_id": store_id, "span_start": 0, "span_end": len(both), "quote": both,
              "value": 30, "unit": "usd", "label": "Alice"}],
            messages=messages, assertions=assertions, as_of=as_of,
        )
        assert misattributed.status != "grounded", misattributed
        assert "different clauses" in misattributed.reason, misattributed.reason

        correct = ground_evidence(
            [{"store_id": store_id, "span_start": 0, "span_end": len(both), "quote": both,
              "value": 10, "unit": "usd", "label": "Alice"}],
            messages=messages, assertions=assertions, as_of=as_of,
        )
        assert correct.status == "grounded", correct.reason

        negated = ground_evidence(
            [{"store_id": denial_id, "span_start": 0, "span_end": len(denial), "quote": denial,
              "value": 10, "unit": "usd", "label": "Atlas"}],
            messages=messages, assertions=assertions, as_of=as_of,
        )
        assert negated.status != "grounded", negated
        assert "negates" in negated.reason, negated.reason
    finally:
        assertions.close()
        messages.close()


def test_unexamined_candidates_that_could_change_the_answer_block_sufficiency(tmp_path):
    """round-3 verify-4 #15: twelve references saying 15 points followed by a thirteenth saying
    20 returned answer_sufficient with 15 and trace.truncated=false — the thirteenth was never
    hydrated. Unexamined NOISE is fine; an unexamined row stating a value of the requested kind
    is not."""
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    from hermes_lcm.requirements_compiler import compile_preanswer_evidence

    cfg = LCMConfig(database_path=str(tmp_path / "deferred.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("dc", platform="cli", context_length=200_000)
        refs = []
        for index in range(13):
            points = 20 if index == 12 else 15
            store_id = e._store.append(
                "dc", {"role": "user", "content": f"Alice scored {points} points in game {index}."},
                source="cli")
            refs.append(f"lcm:{store_id}:0-{len(f'Alice scored {points} points in game {index}.')}")
        e._store.commit()

        result = compile_preanswer_evidence(
            "How many points did Alice score?",
            baseline_refs=refs,
            engine=e,
            enabled=True,
            budgets={"max_retrieval_calls": 0},
        )
        assert result["metrics"]["deferred_candidates"] >= 1, result["metrics"]
        assert result["metrics"]["deferred_material_candidates"] >= 1, result["metrics"]
        assert result["state"] != "answer_sufficient", result["state"]
        assert result["trace"]["truncated"] is True
    finally:
        e.shutdown()


def test_a_question_s_own_year_and_currency_survive_compilation():
    """round-3 verify-4 #17: "March 2024" anchored in 2026 compiled to March 2026, and a
    question asking for euros compiled to a usd contract — the pipeline answered a different
    question from the one asked, with genuine evidence."""
    from hermes_lcm.answer_contract import compile_answer_contract, _requested_unit

    decision = compile_answer_contract(
        "How many vacations did I take in March 2024?", "2026-09-09")
    assert decision.status == "fallback", decision
    assert decision.reason_code == "explicit_year_not_representable"

    assert _requested_unit("How much did Atlas cost in euros?") == "eur"
    assert _requested_unit("How much did Atlas cost in dollars?") == "usd"
    assert _requested_unit("How much did Atlas cost in £?") == "gbp"

    # a relative window is still resolved as before
    relative = compile_answer_contract(
        "How many vacations did I take last month?", "2026-09-09")
    assert relative.status == "planned", relative


def test_structured_adapters_refuse_an_unfinished_generation():
    """round-3 verify-4 #27: every structured adapter accepted a payload that arrived with
    finish_reason="length" — a truncated extraction or selection acquired a successful
    receipt."""
    from types import SimpleNamespace
    from hermes_lcm.escalation import unfinished_generation_reason

    truncated = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='{"a": 1}'), finish_reason="length")])
    assert unfinished_generation_reason(truncated) == "finish_reason=length"

    incomplete = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
        status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    assert unfinished_generation_reason(incomplete).startswith("status=")

    finished = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="{}"), finish_reason="stop")])
    assert unfinished_generation_reason(finished) == ""
