"""#32 — a generation counts as finished only on positive terminal evidence.

Both core consumers (the summariser and lcm_expand_query's synthesis) accepted a generation on
the strength of a `finish_reason` the Hermes adapters fabricate: the chat stream aggregator
turns a stream that ended without a terminal chunk into `finish_reason="stop"`, and the Codex
Responses adapter projects `incomplete`/`failed` into a chat choice with `stop` and no status
at all. The old gate was a negative list — known cut reasons plus `status="incomplete"` — so an
aborted generation became a durable node and a moved frontier, and `status="failed"` with text
present was never checked.
"""
import json
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm import escalation, tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.errors import GenerationNotTerminatedError, SummaryUnavailableError

SUMMARY_TEXT = "Topic A: decided X.\nExpand for details about: topic A"


def _install(monkeypatch, response):
    module = ModuleType("agent.auxiliary_client")
    module.call_llm = lambda **kwargs: response
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)


def _choice(finish_reason=..., content=SUMMARY_TEXT):
    fields = {"message": SimpleNamespace(content=content)}
    if finish_reason is not ...:
        fields["finish_reason"] = finish_reason
    return SimpleNamespace(**fields)


def test_a_response_with_no_terminal_evidence_is_not_a_finished_summary(monkeypatch):
    """An adapter that says nothing about how the generation ended has not said it ended."""
    _install(monkeypatch, SimpleNamespace(choices=[_choice()]))
    assert escalation._call_llm_for_summary("summarize this", 200) is None
    recorded = escalation._LAST_ROUTE_ERROR.error
    assert isinstance(recorded, GenerationNotTerminatedError)
    assert "terminal" in str(recorded)


def test_a_failed_status_is_refused_as_an_object_and_as_a_dict(monkeypatch):
    """The generic shape recovery rebuilds a choice with a fabricated `stop`; the object form
    keeps `status="failed"` and the dict form is read through mapping keys. Neither is a
    finished generation, and both were accepted."""
    _install(monkeypatch, SimpleNamespace(choices=[_choice("stop")], status="failed"))
    assert escalation._call_llm_for_summary("summarize this", 200) is None

    _install(monkeypatch, {
        "choices": [{"message": {"content": SUMMARY_TEXT}, "finish_reason": "stop"}],
        "status": "failed",
    })
    assert escalation._call_llm_for_summary("summarize this", 200) is None

    _install(monkeypatch, SimpleNamespace(
        choices=[_choice("stop")], error={"message": "upstream aborted the run"}))
    assert escalation._call_llm_for_summary("summarize this", 200) is None


def test_an_unrecognised_terminal_state_is_not_proof_of_termination(monkeypatch):
    for finish_reason in ("aborted", "cancelled", "error", "", None):
        _install(monkeypatch, SimpleNamespace(choices=[_choice(finish_reason)]))
        assert escalation._call_llm_for_summary("summarize this", 200) is None, finish_reason


def test_a_generation_that_used_its_whole_cap_is_not_terminal_even_saying_stop(monkeypatch):
    """The response's own accounting contradicts its terminal claim."""
    _install(monkeypatch, SimpleNamespace(
        choices=[_choice("stop")], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=200)))
    assert escalation._call_llm_for_summary("summarize this", 200) is None

    _install(monkeypatch, SimpleNamespace(
        choices=[_choice("stop")], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=12)))
    assert escalation._call_llm_for_summary("summarize this", 200) == SUMMARY_TEXT


def test_a_positively_terminal_response_is_still_accepted(monkeypatch):
    for terminal in ("stop", "end_turn", "stop_sequence", "tool_calls"):
        _install(monkeypatch, SimpleNamespace(choices=[_choice(terminal)]))
        assert escalation._call_llm_for_summary("summarize this", 200) == SUMMARY_TEXT, terminal

    # a provider status is terminal evidence of its own
    _install(monkeypatch, SimpleNamespace(choices=[_choice()], status="completed"))
    assert escalation._call_llm_for_summary("summarize this", 200) == SUMMARY_TEXT
    _install(monkeypatch, {
        "choices": [{"message": {"content": SUMMARY_TEXT}, "finish_reason": "stop"}],
        "status": "completed",
    })
    assert escalation._call_llm_for_summary("summarize this", 200) == SUMMARY_TEXT


def test_the_refusal_is_the_fail_closed_carrier_the_engine_already_handles():
    assert issubclass(GenerationNotTerminatedError, SummaryUnavailableError)


def test_an_unterminated_route_leaves_the_conversation_raw(tmp_path, monkeypatch):
    """Fail-before-loss: no node is published, the frontier does not move, and the messages
    stay in the caller's context."""
    _install(monkeypatch, SimpleNamespace(choices=[_choice()]))
    cfg = LCMConfig(database_path=str(tmp_path / "unterminated.db"),
                    fresh_tail_count=1, leaf_chunk_tokens=1, incremental_max_depth=0)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("gc", platform="cli", context_length=200_000)
        e.threshold_tokens = 1
        conversation = [
            {"role": "user", "content": "DECISION: cancel the launch, the regression is real."},
            {"role": "assistant", "content": "Understood, cancelling and recording the reason."},
            {"role": "user", "content": "tail"},
        ]
        result = e.compress(list(conversation))
        assert e._dag.get_session_nodes("gc") == []
        kept = "\n".join(str(m.get("content") or "") for m in result)
        assert "DECISION: cancel the launch" in kept
    finally:
        e.shutdown()


def test_the_query_route_refuses_a_fabricated_terminal_instead_of_answering(tmp_path, monkeypatch):
    """`lcm_expand_query` has its own completion check with the same hole. A refused answer
    keeps the sources — node ids and matches — and never reports complete."""
    cfg = LCMConfig(database_path=str(tmp_path / "query.db"), incremental_max_depth=0)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("qr", platform="cli", context_length=200_000)
        store_id = e._store.append(
            "qr", {"role": "user", "content": "DECISION: cancel the launch."}, source="cli")
        e._store.commit()
        node_id = e._dag.add_node_with_meta(
            SummaryNode(session_id="qr", depth=0, summary="a decision\nExpand for details about: it",
                        token_count=5, source_token_count=50, source_ids=[store_id],
                        source_type="messages", created_at=time.time()),
            level=1,
        )

        _install(monkeypatch, SimpleNamespace(
            choices=[_choice("stop", content="The launch was cancelled.")], status="failed"))
        payload = json.loads(e.handle_tool_call(
            "lcm_expand_query", {"prompt": "what was decided?", "node_ids": [node_id]}))

        assert payload.get("complete") is not True, payload
        assert "answer" not in payload, payload
        assert payload["degraded"] is True
        assert "failed" in payload["error"]
        # the sources are untouched and still nameable by the caller
        assert payload["node_ids"] == [node_id]
        assert payload["matches"][0]["node_id"] == node_id
    finally:
        e.shutdown()


def test_the_query_route_still_answers_a_terminated_generation(tmp_path, monkeypatch):
    cfg = LCMConfig(database_path=str(tmp_path / "query-ok.db"), incremental_max_depth=0)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("qo", platform="cli", context_length=200_000)
        store_id = e._store.append(
            "qo", {"role": "user", "content": "DECISION: cancel the launch."}, source="cli")
        e._store.commit()
        node_id = e._dag.add_node_with_meta(
            SummaryNode(session_id="qo", depth=0, summary="a decision\nExpand for details about: it",
                        token_count=5, source_token_count=50, source_ids=[store_id],
                        source_type="messages", created_at=time.time()),
            level=1,
        )
        _install(monkeypatch, SimpleNamespace(
            choices=[_choice("stop", content="The launch was cancelled.")], status="completed"))
        payload = json.loads(e.handle_tool_call(
            "lcm_expand_query", {"prompt": "what was decided?", "node_ids": [node_id]}))
        assert payload["answer"] == "The launch was cancelled."
        assert "degraded" not in payload
    finally:
        e.shutdown()
