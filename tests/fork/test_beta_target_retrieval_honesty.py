"""What the beta's retrieval fixes must achieve (#50/#51, #52), written before they exist (#19).

The shared property is one sentence: a bounded, capped or unread retrieval must never be
presented as complete. `complete` today is set False only for a missing source row and a corrupt
externalized payload, so a page that stopped at its token budget, a child summary cut at that
budget, and a descendant walk that never started all come back alongside `complete: true`.

#52 is the other half of the same honesty problem from the caller's side: the tool that is
supposed to let a reader plan recovery returns one character by default, so the "continue from
here" cursor it advertises is the only thing it really delivers.

Every test here FAILS on the tree it was written against.
"""
import json
import os
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import maybe_externalize_payload
from hermes_lcm.schemas import LCM_DESCRIBE, LCM_EXPAND
from hermes_lcm.tokens import count_tokens


def _engine(tmp_path, name):
    config = LCMConfig(database_path=str(tmp_path / f"{name}.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.on_session_start(name, platform="cli", context_length=262_144)
    return engine


def _node(engine, session, depth, summary, source_ids, source_type):
    return int(engine._dag.add_node(SummaryNode(
        session_id=session, depth=depth, summary=summary,
        token_count=count_tokens(summary), source_token_count=count_tokens(summary) * 4,
        source_ids=[int(value) for value in source_ids], source_type=source_type,
        created_at=time.time(),
    )))


@pytest.mark.beta_target("#50")
def test_a_page_that_stopped_at_its_budget_is_not_reported_complete(tmp_path):
    """`_expand_message_sources` sets `complete=False` for missing rows and corrupt payloads only.

    Everything else gets `complete=True`, including a page that ran out of token budget halfway
    through the body it was asked for. The response then says `has_more: true` and
    `complete: true` at once, and the consumers that look for `complete is False` — the block
    filter and the query-synthesis completeness sum — cannot see the shortfall at all.

    The bounded page itself is legitimate: it was explicitly ordered and it carries a working
    cursor. The claim that it is complete is not.
    """
    engine = _engine(tmp_path, "budget50")
    try:
        body = "PAGE_50A " + ("evidence about the cancelled rollout. " * 200)
        store_id = engine._store.append(
            "budget50", {"role": "user", "content": body}, source="cli")
        engine._store.commit()
        node_id = _node(engine, "budget50", 0, "The rollout was discussed.",
                        [store_id], "messages")

        payload = json.loads(engine.handle_tool_call(
            "lcm_expand", {"node_id": node_id, "max_tokens": 100}))
        pagination = payload["pagination"]
        expanded = payload["expanded"]

        assert expanded and expanded[0].get("content_truncated") is True, (
            "the fixture no longer hits the budget, so it proves nothing — raise the body size"
        )
        assert pagination.get("has_more") is True
        assert pagination.get("complete") is not True, (
            "a page that withheld content reported itself complete"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#50")
def test_a_last_child_summary_cut_at_the_budget_is_not_reported_complete(tmp_path):
    """`has_more` in `_expand_child_nodes` answers "is there another child", not "is this whole".

    With exactly one child whose summary is cut at the budget, no child follows, so `has_more`
    stays false and `complete` stays true — while the summary the caller asked for came back with
    its tail missing. The end of a child LIST and the end of a child's CONTENT are two different
    states and the response collapses them into one.
    """
    engine = _engine(tmp_path, "child50")
    try:
        child_summary = "CHILD_50B " + ("decisions, rejected options and file paths. " * 200)
        child_id = _node(engine, "child50", 0, child_summary, [], "messages")
        parent_id = _node(engine, "child50", 1, "A parent over one long child.",
                          [child_id], "nodes")

        payload = json.loads(engine.handle_tool_call(
            "lcm_expand", {"node_id": parent_id, "max_tokens": 60}))
        expanded = payload["expanded"]
        pagination = payload["pagination"]

        assert expanded and expanded[0].get("summary_truncated") is True, (
            "the fixture no longer cuts the child summary, so it proves nothing"
        )
        assert pagination.get("has_more") is False, (
            "another child follows, so this is not the last-child case under test"
        )
        assert pagination.get("complete") is not True, (
            "a cut child summary was returned inside a response reported as complete"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#50")
def test_a_requested_hydration_that_could_not_read_its_payload_is_not_complete(tmp_path):
    """50c, and the triage calls it one of the two HARD blockers.

    `_get_externalized_payload` returns None when the archive file is gone, and
    `_expand_message_sources` distinguishes only "payload present" from "payload corrupt" — so
    "ref present, file missing" falls through to ordinary message content. The caller asked for
    the full bytes with `hydrate=true`, got the reference marker instead, and the response says
    `complete: true` over it.

    The missing file is an injected integrity fault, not a claim that this handler deletes
    archives. What is being asserted is fail-closed: a hydration that could not read what it was
    asked to read must not be reported as one that did.
    """
    home = tmp_path / "home"
    home.mkdir()
    os.chmod(home, 0o700)
    config = LCMConfig(database_path=str(tmp_path / "hydrate50c.db"))
    engine = LCMEngine(config=config, hermes_home=str(home))
    try:
        engine.on_session_start("hydrate50c", platform="cli", context_length=262_144)
        body = "HYDRATE_NEEDLE_50C " + ("the full tool result. " * 1_000)
        # the archive the fork keeps of output HERMES spilled — `ignore_enabled_flag` is that
        # path, not the ingest-side externalization this fork does not enable
        archived = maybe_externalize_payload(
            body, kind="tool_result", tool_call_id="c1", session_id="hydrate50c", role="tool",
            config=config, hermes_home=str(home), force=True, ignore_enabled_flag=True)
        store_id = engine._store.append(
            "hydrate50c",
            {"role": "tool", "tool_call_id": "c1", "content": archived["placeholder"]},
            source="cli")
        engine._store.commit()
        node_id = _node(engine, "hydrate50c", 0, "A tool result was archived.",
                        [store_id], "messages")

        intact = json.loads(engine.handle_tool_call(
            "lcm_expand", {"node_id": node_id, "hydrate": True, "max_tokens": 200_000}))
        assert "HYDRATE_NEEDLE_50C" in json.dumps(intact), (
            "the fixture never hydrated anything, so its negative half proves nothing"
        )

        os.remove(archived["path"])
        payload = json.loads(engine.handle_tool_call(
            "lcm_expand", {"node_id": node_id, "hydrate": True, "max_tokens": 200_000}))

        assert "HYDRATE_NEEDLE_50C" not in json.dumps(payload), (
            "the payload came back after its file was removed; the fixture is wrong"
        )
        assert payload["pagination"].get("complete") is not True, (
            "a hydration that returned the reference marker instead of the bytes it was asked "
            "for reported itself complete"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#50")
def test_a_missing_row_under_a_descendant_is_not_silently_dropped(tmp_path):
    """50d, the other hard blocker, and the one that disappears rather than lying quietly.

    The top-level expansion keeps an error-only block, but the recursive walk appends a block
    only `if messages or has_more`. A descendant leaf whose source row does not exist produces
    neither, so its error block is filtered out and the answer is synthesised from a summary and
    a child manifest with no mention that a source could not be read — `complete: true`, no
    `missing_source_store_ids`.

    Every node and edge here exists; only the raw row is missing, which is the injected fault.
    The synthesis route is stubbed as explicitly finished so the claim under test is about
    evidence and nothing else.
    """
    config = LCMConfig(database_path=str(tmp_path / "descendant50d.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("desc50d", platform="cli", context_length=262_144)
        # a leaf that points at a store_id nobody ever wrote
        leaf_id = _node(engine, "desc50d", 0, "A leaf over a row that is gone.",
                        [987_654_321], "messages")
        parent_id = _node(engine, "desc50d", 1, "A parent over that leaf.",
                          [leaf_id], "nodes")

        direct = json.loads(engine.handle_tool_call(
            "lcm_expand", {"node_id": leaf_id, "max_tokens": 100_000}))
        assert direct["pagination"].get("complete") is False, (
            "the fixture's missing row is not even detected directly, so it proves nothing"
        )

        _install_finished_auxiliary_client("Nothing could be read.")
        try:
            payload = json.loads(engine.handle_tool_call("lcm_expand_query", {
                "prompt": "What is under this node?",
                "node_ids": [parent_id],
            }))
        finally:
            _restore_auxiliary_client()

        assert payload.get("complete") is not True, (
            "a source row that could not be read vanished from the walk and the answer over it "
            "reported itself complete"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#51")
def test_a_descendant_walk_that_never_started_is_not_reported_complete(tmp_path):
    """The walk runs only if budget remains after the parent summary and the child manifest.

    When those two exactly exhaust the chosen context budget, no walk happens, no block records
    that it did not happen, and the answer is returned as complete over raw evidence that was
    never read. Nothing is corrupt and no row is missing — every source exists and is reachable;
    the retrieval simply never went and never said so.

    The synthesis route is stubbed as explicitly finished, so the completeness claim under test
    is about the EVIDENCE and nothing else.
    """
    engine = _engine(tmp_path, "walk51")
    try:
        body = "NEEDLE_51C the rollback was signed off by the release manager."
        store_id = engine._store.append(
            "walk51", {"role": "user", "content": body}, source="cli")
        engine._store.commit()
        leaf_summary = "b"
        parent_summary = "a"
        leaf_id = _node(engine, "walk51", 0, leaf_summary, [store_id], "messages")
        parent_id = _node(engine, "walk51", 1, parent_summary, [leaf_id], "nodes")

        _install_finished_auxiliary_client("The rollback was signed off.")
        try:
            payload = json.loads(engine.handle_tool_call("lcm_expand_query", {
                "prompt": "Who signed off the rollback?",
                "node_ids": [parent_id],
                # exactly the parent summary plus the child manifest: the handler's own counter
                # charges those two and nothing is left for the walk
                "context_max_tokens": count_tokens(parent_summary) + count_tokens(leaf_summary),
            }))
        finally:
            _restore_auxiliary_client()

        assert "NEEDLE_51C" not in json.dumps(payload), (
            "the fixture delivered the raw evidence after all, so it proves nothing"
        )
        assert payload.get("complete") is not True, (
            "an answer that never read the raw evidence under its node reported itself complete"
        )
    finally:
        engine.shutdown()


_SAVED_AUXILIARY_CLIENT: list = []


def _install_finished_auxiliary_client(answer: str) -> None:
    """A synthesis route that is unambiguously finished, so only evidence coverage is on trial."""
    module = ModuleType("agent.auxiliary_client")
    module.call_llm = lambda **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=answer), finish_reason="stop")],
        status="completed", incomplete_details=None,
    )
    _SAVED_AUXILIARY_CLIENT.append(sys.modules.get("agent.auxiliary_client"))
    sys.modules["agent.auxiliary_client"] = module


def _restore_auxiliary_client() -> None:
    previous = _SAVED_AUXILIARY_CLIENT.pop()
    if previous is None:
        sys.modules.pop("agent.auxiliary_client", None)
    else:
        sys.modules["agent.auxiliary_client"] = previous


# ── #52 — the plain describe call has to describe something ──────────────────────────────────

@pytest.mark.beta_target("#52")
def test_a_plain_describe_returns_at_least_what_the_documented_default_would(tmp_path):
    """`_parse_positive_int(args.get("summary_max_chars"), 0)` clamps to 1 before the `or 4000`.

    So the intended 4,000-character fallback is unreachable and an ordinary
    `lcm_describe(node_id=…)` returns one character of the summary with a continuation cursor at
    offset 1. The bytes are all still stored and the response does not hide its partiality — but
    a reader planning recovery gets nothing to plan with, and the truncated child-summary path
    points at exactly this tool for the rest of a summary.

    Stated as "no worse than the explicit call" so a fix that returns the whole summary passes
    just as well as one that restores the 4,000-character default.
    """
    engine = _engine(tmp_path, "describe52")
    try:
        summary = "Decisions and their rationale. " * 300
        node_id = _node(engine, "describe52", 0, summary, [], "messages")

        default = json.loads(engine.handle_tool_call("lcm_describe", {"node_id": node_id}))
        explicit = json.loads(engine.handle_tool_call(
            "lcm_describe", {"node_id": node_id, "summary_max_chars": 4_000}))

        assert len(explicit["summary"]) > 1, "the fixture's summary is too short to show anything"
        assert len(default["summary"]) >= len(explicit["summary"]), (
            "the default describe call returns less than asking for the documented default"
        )
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#52")
def test_the_continuation_parameters_the_handlers_support_are_in_their_schemas():
    """A cursor a caller cannot see is a cursor a schema-driven model will not use.

    `lcm_describe` already supports `summary_offset`/`summary_max_chars` and `lcm_expand` already
    supports `envelope_offset` — the handlers accept them and the responses advertise them as the
    way to continue — but none of the three is declared. Native dispatch happens to pass
    undeclared keys through, so this is a discoverability defect rather than a hard block; a
    strictly validating integration would reject the continuation outright.
    """
    describe_properties = LCM_DESCRIBE["parameters"]["properties"]
    expand_properties = LCM_EXPAND["parameters"]["properties"]

    assert "summary_offset" in describe_properties, "lcm_describe hides its summary cursor"
    assert "summary_max_chars" in describe_properties, "lcm_describe hides its summary length"
    assert "envelope_offset" in expand_properties, "lcm_expand hides its envelope cursor"
