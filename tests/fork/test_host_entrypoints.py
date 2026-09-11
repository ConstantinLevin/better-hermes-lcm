"""Entry-to-effect checks for the contracts the fork's hooks depend on (#16).

The gap #16 names is narrow and real. Registration is already covered
(``tests/test_host_capability.py`` loads the real entrypoint and calls ``register(ctx)``), and
so is calling a registered callback (``tests/test_packaging_install.py`` fires ``post_llm_call``
against a ``SimpleNamespace(_hooks={})``). What neither covers is whether the CURRENT host
actually produces those calls with the payloads the hooks read. The callback is real there; the
producer is not. Upstream can rename a function, move a call, or stop passing a field, and every
one of those tests stays green while the fork behaviour never runs — that is failure mode M2, and
nothing in the suite catches it.

So each test here starts at a host function, uses the payload shape that host's own producer
builds, and asserts an effect in the store or the assembled context. What is deliberately
replaced is named in each test.

It is NOT a claim of completeness over host variants: it covers the connections that were read,
against one revision, and says which one in the report. Contract 3 of #16 (preflight preserves
originals on failure) is not duplicated here — ``tests/fork/test_host_integration.py`` already
drives it from the host's real ``build_turn_context``.
"""
import json
import time
import types

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine

from .test_host_integration import _host_modules  # the same strict/optional gate (#14)


def _plugin_entrypoint(name: str):
    """Execute the plugin's real ``__init__.py`` body, the way the host loader does."""
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        name, str(repo_root / "__init__.py"), submodule_search_locations=[str(repo_root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _HostCtx:
    """What Hermes' plugin loader offers a context-engine plugin."""

    def __init__(self):
        self.engine = None

    def register_context_engine(self, engine):
        self.engine = engine

    def register_tool(self, **_kwargs):  # the host's registry entry point
        pass


def _registered_plugin(tmp_path, monkeypatch, name):
    """Register the plugin against the host's REAL plugin manager.

    The manager is the host's own ``hermes_cli.plugins.PluginManager`` for a throwaway
    HERMES_HOME, not a stand-in with a ``_hooks`` dict: the plugin attaches its post-turn hook to
    whatever that class actually exposes, and a stand-in cannot tell us whether it still does.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    import hermes_cli.plugins as host_plugins

    manager = host_plugins.PluginManager()
    monkeypatch.setattr(host_plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(host_plugins, "_plugin_manager", manager, raising=False)

    module = _plugin_entrypoint(name)
    ctx = _HostCtx()
    module.register(ctx)
    assert ctx.engine is not None, "the host loader's register(ctx) produced no context engine"
    return module, manager, ctx.engine


def _fire_post_turn_hook(turn_finalizer, *, agent_session_id, messages, platform="cli",
                         assistant_response="done"):
    """Call the host's own post-turn hook dispatcher with its own producer's payload.

    `agent/turn_finalizer.py` fires this hook once per turn with exactly these keyword
    arguments, `conversation_history=list(messages)` among them. The boundary deliberately
    replaced is the argument assembly around it — building the values needs a live provider turn.
    Everything from `_invoke_hook_safely` down is the host's: the lifecycle dispatcher, the
    plugin manager, and the callback the plugin's own `register()` attached.
    """
    return turn_finalizer._invoke_hook_safely(
        "post_llm_call",
        turn_finalizer.logger if hasattr(turn_finalizer, "logger") else _NullLogger(),
        session_id=agent_session_id,
        task_id="task-1",
        turn_id="turn-1",
        user_message=str(messages[0].get("content") if messages else ""),
        assistant_response=assistant_response,
        conversation_history=list(messages),
        model="test-model",
        platform=platform,
    )


class _NullLogger:
    def warning(self, *_a, **_k):
        pass


# ── contract: turn completion ingests the originals, with no compaction ─────────────────────

def test_a_finished_turn_ingests_its_originals_through_the_host_hook(tmp_path, monkeypatch):
    """Entry: `agent.turn_finalizer._invoke_hook_safely("post_llm_call", …)`.

    Effect: every message of the turn is durable, byte-identical, and no summary node was
    published — ingest at turn completion must not be conditional on compaction happening.
    """
    with _host_modules() as (_run_agent, _turn_context, _conversation_loop):
        import importlib
        turn_finalizer = importlib.import_module("agent.turn_finalizer")
        _module, _manager, engine = _registered_plugin(
            tmp_path, monkeypatch, "hermes_lcm_entrypoint_turn")
        try:
            engine.on_session_start("turn-session", platform="cli", context_length=262_144)
            messages = [
                {"role": "user", "content": "ENTRYPOINT_ORIGINAL_ONE: cancel the rollout"},
                {"role": "assistant", "content": "ENTRYPOINT_ORIGINAL_TWO: rollout cancelled"},
            ]
            _fire_post_turn_hook(turn_finalizer, agent_session_id="turn-session",
                                 messages=messages)

            stored = [row.get("content")
                      for row in engine._store.get_session_messages("turn-session")]
            assert stored == [m["content"] for m in messages], (
                "the host's post-turn hook did not reach the plugin's ingest"
            )
            assert engine._dag.get_session_nodes("turn-session") == [], (
                "ingest at turn completion must not depend on a compaction happening"
            )
        finally:
            engine.shutdown()


# ── contract: session rebinding binds only matching owners ───────────────────────────────────

def test_the_post_turn_hook_ingests_into_the_clone_that_owns_the_session(tmp_path, monkeypatch):
    """Entry: the same host hook, with two runtimes alive.

    The host finalises with `conversation_history` and a session id and, on this revision, no
    `context_compressor` in the payload, so the plugin has to resolve the owning runtime itself.
    Effect: the turn lands in the clone bound to that session and nowhere else — a rebind that
    followed the wrong owner would move a live runtime off the session it is serving and split
    one conversation across two stores.
    """
    with _host_modules() as (_run_agent, _turn_context, _conversation_loop):
        import importlib
        turn_finalizer = importlib.import_module("agent.turn_finalizer")
        _module, _manager, prototype = _registered_plugin(
            tmp_path, monkeypatch, "hermes_lcm_entrypoint_rebind")
        owner = prototype.clone_for_agent()
        other = prototype.clone_for_agent()
        try:
            # three live bindings, so "the turn was stored" cannot pass for "the right runtime
            # stored it": the clones share one database, and an earlier version of this test
            # survived a mutation that routed every turn to the singleton because of exactly
            # that. What discriminates is which runtime is bound to what afterwards.
            prototype.on_session_start("session-prototype", platform="cli", context_length=262_144)
            owner.on_session_start("session-owner", platform="cli", context_length=262_144)
            other.on_session_start("session-other", platform="cli", context_length=262_144)

            _fire_post_turn_hook(
                turn_finalizer, agent_session_id="session-owner",
                messages=[{"role": "user", "content": "OWNER_TURN: only this runtime"}])

            assert [row.get("content") for row in owner._store.get_session_messages(
                "session-owner")] == ["OWNER_TURN: only this runtime"]
            assert prototype.current_session_id == "session-prototype", (
                "the singleton was rebound onto a session a live clone owns"
            )
            assert other.current_session_id == "session-other", (
                "an unrelated runtime was rebound away from the session it serves"
            )
            assert owner.current_session_id == "session-owner"
            assert other._store.get_session_messages("session-other") == []
            assert prototype._store.get_session_messages("session-prototype") == []
        finally:
            owner.shutdown()
            other.shutdown()
            prototype.shutdown()


# ── contract: tool dispatch receives the current messages ────────────────────────────────────

def test_the_hosts_tool_dispatch_hands_the_current_turn_to_the_engine(tmp_path, monkeypatch):
    """Entry: `agent.tool_executor._resolve_sequential_dispatch`, the host's own branch chooser.

    That function is what decides a context-engine tool call goes to
    `context_compressor.handle_tool_call(name, args, messages=messages)`. If the host stops
    passing `messages` there, the plugin silently loses current-turn ingest before any recovery
    tool runs, and nothing else in the suite notices.

    Effect: a message that exists only in the live turn is durable after the dispatched call.
    The boundary replaced is the AIAgent — a host-shaped object carrying the three attributes the
    branch reads.
    """
    with _host_modules() as (_run_agent, _turn_context, _conversation_loop):
        import importlib
        tool_executor = importlib.import_module("agent.tool_executor")
        config = LCMConfig(database_path=str(tmp_path / "dispatch.db"))
        engine = LCMEngine(config=config, hermes_home=str(tmp_path))
        try:
            engine.on_session_start("dispatch", platform="cli", context_length=262_144)
            live = [{"role": "user", "content": "LIVE_TURN_ONLY: the credentials expired at 14:02"}]

            agent = types.SimpleNamespace(
                _context_engine_tool_names={"lcm_recent"},
                context_compressor=engine,
                _memory_manager=None,
                quiet_mode=True,
                # the branch starts the host's quiet-mode spinner; saying no keeps this test
                # about dispatch rather than about terminal output
                _should_emit_quiet_tool_messages=lambda: False,
                _should_start_quiet_spinner=lambda: False,
            )
            ref = tool_executor._ToolCallRef(
                name="lcm_recent", args={}, task_id="t", call_id="c1", trace=None)
            dispatch = tool_executor._resolve_sequential_dispatch(agent, ref, live)
            result = dispatch.execute({})

            assert json.loads(result), "the dispatched context-engine tool returned nothing"
            stored = [row.get("content") for row in engine._store.get_session_messages("dispatch")]
            assert stored == [live[0]["content"]], (
                "the host's dispatch did not carry the current turn into the engine"
            )
        finally:
            engine.shutdown()


# ── contract: prefix assembly reaches the stored sources ─────────────────────────────────────

def test_the_assembled_prefix_leads_back_to_the_stored_originals(tmp_path, monkeypatch):
    """Entry: `agent.turn_context.build_turn_context`, the host's real turn prologue.

    Effect: after a compaction the context the host hands the provider carries a summary node
    identifier, and expanding that identifier returns the original rows verbatim. A summary the
    agent cannot walk back to its sources is an index that lies by omission, and it looks exactly
    like a working one from inside the prefix.

    The boundary replaced is the summariser call, which returns a deterministic index-shaped
    string; nothing else about assembly, publication or expansion is stubbed.
    """
    from hermes_lcm import escalation

    with _host_modules() as (run_agent, turn_context, conversation_loop):
        from unittest.mock import MagicMock, patch
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
        ):
            agent = run_agent.AIAgent(api_key="test-key-1234567890",
                                      base_url="https://openrouter.ai/api/v1",
                                      quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.save_trajectories = False
        agent.compression_enabled = True

        config = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1)
        config.database_path = str(tmp_path / "assembly.db")
        engine = LCMEngine(config=config, hermes_home=str(tmp_path))
        try:
            engine.on_session_start(agent.session_id or "assembly", platform="cli",
                                    context_length=200_000)
            engine.threshold_tokens = 1
            engine.protect_first_n = 0
            engine.protect_last_n = 0
            agent.context_compressor = engine

            needle = "ASSEMBLY_NEEDLE: the release manager signed the rollback off at 14:02"
            history = [{"role": "user", "content": needle}]
            for index in range(20):
                history.append({"role": "user", "content": f"request {index} " + "detail " * 60})
                history.append({"role": "assistant", "content": f"reply {index} " + "done " * 60})

            monkeypatch.setattr(
                escalation, "_call_llm_for_summary",
                lambda *a, **k: "The rollback was signed off.\n"
                                "Expand for details about: the rollback sign-off")
            with (
                patch.object(agent, "_persist_session", create=True),
                patch.object(agent, "_save_trajectory", create=True),
            ):
                ctx = turn_context.build_turn_context(
                    agent, "continue", None, list(history), None, None, None,
                    restore_or_build_system_prompt=conversation_loop._restore_or_build_system_prompt,
                    install_safe_stdio=conversation_loop._install_safe_stdio,
                    sanitize_surrogates=conversation_loop._sanitize_surrogates,
                    summarize_user_message_for_log=conversation_loop._summarize_user_message_for_log,
                    set_session_context=conversation_loop.set_session_context,
                    set_current_write_origin=conversation_loop.set_current_write_origin,
                    ra=getattr(conversation_loop, "_ra", None),
                )

            nodes = engine._dag.get_session_nodes(engine._session_id)
            assert nodes, "the host's prologue produced no compaction to assemble from"
            prefix = "\n".join(str(m.get("content") or "") for m in ctx.messages)
            # only the explicit forms count. A bare `str(node_id)` match was in this test first
            # and it survived a mutation that stripped the node identity out of the header
            # entirely — "1" appears in any prefix that contains a number.
            named = [node for node in nodes
                     if f"node_id={node.node_id}" in prefix or f"node {node.node_id}" in prefix]
            assert named, "the assembled prefix names no node to expand"

            recovered = ""
            for node in named:
                payload = json.loads(engine.handle_tool_call(
                    "lcm_expand", {"node_id": int(node.node_id), "max_tokens": 200_000}))
                recovered += json.dumps(payload, ensure_ascii=False)
            assert needle in recovered, (
                "the prefix the host will send cannot be walked back to the original rows"
            )
        finally:
            engine.shutdown()


# ── contract: condensation publication is atomic ─────────────────────────────────────────────

def test_a_node_and_its_sidecar_are_published_together_or_not_at_all(tmp_path, monkeypatch):
    """Entry: `SummaryDAG.add_node_with_meta`, the single publication path condensation uses.

    Effect: when the sidecar cannot be written, no node exists either. A node published without
    its level and index block renders without its level tag in assembly and cannot be answered
    for by the index reader — a half-published state that reads as a whole one.

    Not a host entrypoint, and it is in #16's list because the host is what makes it concurrent:
    the failure it guards against is a crash between two commits during a turn.
    """
    dag = SummaryDAG(tmp_path / "atomic.db")
    try:
        node = SummaryNode(session_id="atomic", depth=1, summary="a condensation",
                           token_count=4, source_token_count=40, source_ids=[1, 2],
                           source_type="nodes", created_at=time.time())

        def _refuse(*_args, **_kwargs):
            raise RuntimeError("sidecar write refused")

        monkeypatch.setattr(dag.node_meta, "write_statement", _refuse)
        # the error is pinned, not merely "something raised": a bare `raises(Exception)` is the
        # same vacuity as catching one — a renamed `add_node_with_meta` would raise AttributeError,
        # satisfy the context manager, and leave `get_session_nodes(...) == []` trivially true
        # because nothing was ever published
        with pytest.raises(RuntimeError, match="sidecar"):
            dag.add_node_with_meta(node, level=2)

        assert dag.get_session_nodes("atomic") == [], (
            "a node was published without the sidecar that describes it"
        )
        assert dag.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
    finally:
        dag.close()
