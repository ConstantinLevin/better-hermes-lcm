"""What the beta's bypass and generation-completion fixes must achieve (#62, #32), written
before they exist (#19).

Both of these are places where the plugin turns somebody else's decision into its own loss: the
bypass path shortens a session it never stored and overrides the host's explicit refusal to
compress, and the completion check reads only two of the ways a route can stop, so a generation
the host reported as `failed` becomes a durable node.

Every test here FAILS on the tree it was written against; see the module docstring of
``test_beta_target_source_fidelity.py`` for why that is the deliverable.

Fail-before-loss is explicitly allowed by both issues, so the bypass assertions accept a named
refusal that keeps the originals — what they do not accept is a shortened list returned as
success, with or without a receipt.
"""
import json
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm import escalation
from hermes_lcm.errors import (
    ExtractionUnavailableError,
    SummaryUnavailableError,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


# ── #62 — the bypass path must not transform what it does not store ──────────────────────────

# Both anchors, because a cut that fires at one window and not the other is a defect in this fork
# and the bypass path's target size is window-derived. 200_000 — what this fixture used at first —
# is neither anchor.
WINDOWS = (262_144, 1_000_000)


def _bypassed_engine(tmp_path, name, window):
    config = LCMConfig(database_path=str(tmp_path / f"{name}.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.on_session_start("bypassed", platform="cli", context_length=window)
    engine._session_stateless = True   # the operator policy that selects this path
    engine.threshold_tokens = 10
    return engine


def _bypass_history():
    """Five turns, ~20k characters, one of them carrying the host's API sidecar.

    The sidecar matters on its own (#64, folded into #62): the host substitutes `api_content` as
    the turn's request content, so deleting or shortening it is a second removal on top of the
    message one.
    """
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "FIRST: approve the migration. " + "a" * 5_000},
        {"role": "assistant", "content": "SECOND: the migration is approved. " + "b" * 5_000,
         "api_content": "SECOND: the migration is approved, with the staging caveat. " + "b" * 5_000},
        {"role": "user", "content": "THIRD: and the rollback plan. " + "c" * 5_000},
        {"role": "user", "content": "LATEST: cancel deployment"},
    ]


def _call_bypass(engine, messages):
    """Run the bypass and return its list, or None if it REFUSED.

    The bound method is resolved before the try, and only a named refusal counts as one. An
    earlier version caught bare `Exception` around the call itself, which meant that if the #62
    fix renamed or re-signatured `_compress_lcm_bypassed_session` — the most likely shape of a
    fix to that very function — the resulting AttributeError or TypeError became "it refused"
    and both assertions below passed over nothing at all. A gate that stops checking has to say
    so, not go quiet.
    """
    call = engine._compress_lcm_bypassed_session  # AttributeError here fails the test
    try:
        return call(messages, current_tokens=100_000, force=True)
    except (SummaryUnavailableError, ExtractionUnavailableError):
        # fail-before-loss, which both #62 and the fork's own rule prefer to a success that
        # shortened the session
        return None


def _assert_nothing_was_shortened(returned, snapshot, live, *, what):
    """Either every message came back with its fields intact, or the call refused.

    `returned` is None only when the call raised one of the fork's own named unavailability
    errors. What is never acceptable is a shorter list, a truncated body, or a receipt standing
    in for either.
    """
    assert [dict(m) for m in live] == snapshot, "the call mutated the caller's own list"
    if returned is None:
        return
    assert [m.get("role") for m in returned] == [m.get("role") for m in snapshot], what
    assert [m.get("content") for m in returned] == [m.get("content") for m in snapshot], what
    assert [m.get("api_content") for m in returned] == [
        m.get("api_content") for m in snapshot
    ], f"{what} (the host's api_content sidecar)"


@pytest.mark.beta_target("#62")
@pytest.mark.parametrize("window", WINDOWS)
def test_a_bypassed_session_comes_back_unchanged_when_no_native_compressor_exists(tmp_path, window):
    """With no native compressor the plugin builds its own head/tail context.

    `_fallback_tail_compaction` deletes whole messages out of the middle and truncates the
    survivors, leaving a sentence saying so — for a session the plugin never stored, so nothing
    it writes can bring those messages back. Measured on this fixture today: five messages of
    ~20,000 characters come back as two of ~370, the newest one among the shortened.
    """
    messages = _bypass_history()
    snapshot = [dict(m) for m in messages]
    engine = _bypassed_engine(tmp_path, "nonative62", window)
    try:
        engine._get_host_fallback_compressor = lambda: None
        engine._bypass_compaction_target_tokens = lambda **_kwargs: 300  # real cap pressure
        returned = _call_bypass(engine, messages)
        _assert_nothing_was_shortened(
            returned, snapshot, messages,
            what="the bypass deleted or truncated a session it does not store")
    finally:
        engine.shutdown()


@pytest.mark.beta_target("#62")
@pytest.mark.parametrize("window", WINDOWS)
def test_an_explicit_native_abort_is_not_overridden_by_the_local_trim(tmp_path, window):
    """An abort is a decision to PRESERVE, not a failed attempt.

    When the host's compressor returns the list unchanged with `_last_compress_aborted=True` and
    the result is still over the local target, the plugin runs its own destructive trim over the
    host's decision and clears the abort flag. The host said do not compress this; overruling
    that is the plugin deciding to lose content the host chose to keep.
    """
    messages = _bypass_history()
    snapshot = [dict(m) for m in messages]

    class _AbortingCompressor:
        compression_count = 0
        _last_compress_aborted = True

        def compress(self, msgs, **_kwargs):
            return msgs

    engine = _bypassed_engine(tmp_path, "abort62", window)
    try:
        compressor = _AbortingCompressor()
        engine._host_fallback_compressor = compressor
        engine._host_fallback_session_id = engine._bypass_lcm_session_id()
        engine._get_host_fallback_compressor = lambda: compressor
        engine._bypass_compaction_target_tokens = lambda **_kwargs: 300
        returned = _call_bypass(engine, messages)
        _assert_nothing_was_shortened(
            returned, snapshot, messages,
            what="the host's explicit abort was overridden by the local trim")
        assert engine._last_compress_aborted is True, (
            "the abort flag was cleared, so the override is reported as an ordinary success"
        )
    finally:
        engine.shutdown()


# ── #32 — a route that did not finish must not produce a durable node or a complete answer ───

def _install_auxiliary_client(monkeypatch, *, content, finish_reason="stop", **response_fields):
    """A host adapter whose Chat-shaped return carries the fields the real adapters carry.

    The host normalises a missing terminal signal to `finish_reason="stop"` before LCM ever sees
    it, so a fabricated `stop` is not proof of anything. What the adapters DO forward on some
    routes is the response-level `status` and its `error`/`incomplete_details`, and those are
    what the plugin has to read.
    """
    module = ModuleType("agent.auxiliary_client")
    module.call_llm = lambda **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        **response_fields,
    )
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)


@pytest.mark.beta_target("#32")
def test_a_failed_generation_is_refused_by_the_summariser(monkeypatch):
    """`escalation` checks truncation reasons and `status == "incomplete"` — and nothing else.

    A response the host hands over with `status="failed"` and an error object still carries
    `finish_reason="stop"` on its choice, so the text is accepted, published as a leaf and the
    frontier moves past sources that were never actually summarised. A partial sentence that
    happens to look plausible is not a completion guarantee.
    """
    _install_auxiliary_client(
        monkeypatch,
        content="Deployment approved; remaining constraints are",
        status="failed",
        error={"code": "server_error", "message": "upstream cancelled the generation"},
    )
    assert escalation._call_llm_for_summary("summarize this", 200) is None, (
        "a generation the host reported as failed was accepted as a summary"
    )


@pytest.mark.beta_target("#32")
def test_a_failed_generation_is_refused_by_the_expand_query_route(tmp_path, monkeypatch):
    """The query route has its own completion check, and the same hole.

    `lcm_expand_query` reads truncation reasons and `incomplete`; a `failed` status reaches its
    answer as an ordinary one and the tool reports `complete: true` over it. Fixing only the
    summariser leaves this consumer answering questions from a generation that did not finish.
    """
    config = LCMConfig(database_path=str(tmp_path / "failedquery32.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    try:
        engine.on_session_start("q32", platform="cli", context_length=262_144)
        store_id = engine._store.append(
            "q32", {"role": "user", "content": "The rollout was cancelled at 14:02."},
            source="cli")
        engine._store.commit()
        node_id = engine._dag.add_node(SummaryNode(
            session_id="q32", depth=0, summary="The rollout was cancelled.",
            token_count=6, source_token_count=12, source_ids=[int(store_id)],
            source_type="messages", created_at=time.time()))

        _install_auxiliary_client(
            monkeypatch,
            content="The rollout was cancelled at",
            status="failed",
            error={"code": "server_error", "message": "upstream cancelled the generation"},
        )
        payload = json.loads(engine.handle_tool_call("lcm_expand_query", {
            "prompt": "When was the rollout cancelled?",
            "node_ids": [int(node_id)],
        }))
        assert payload.get("complete") is not True, (
            "an answer from a failed generation was reported as complete"
        )
    finally:
        engine.shutdown()
