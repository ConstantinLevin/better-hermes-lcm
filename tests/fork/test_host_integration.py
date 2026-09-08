"""Plan verification 4 — a summariser failure injected through the host's REAL turn prologue
(`agent.turn_context.build_turn_context`) must not kill the turn: the context comes back
intact, the engine's cooldown is armed and the host reports the block as `cooldown:<s>`.

Needs hermes-agent importable (it is in the plugin's venv); skipped otherwise.
"""
from unittest.mock import MagicMock, patch

import pytest

from hermes_lcm import escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

import importlib
import os
import sys
from pathlib import Path


def _import_host():
    """hermes-agent's top-level `tools` package is shadowed by this plugin's tools.py while the
    plugin dir sits first on sys.path (pytest rootdir); import the host with its root first."""
    agent_pkg = pytest.importorskip("agent")
    host_root = str(Path(agent_pkg.__file__).resolve().parent.parent)
    plugin_dir = str(Path(__file__).resolve().parent.parent)
    saved = list(sys.path)
    saved_env = dict(os.environ)  # importing the host may load ~/.hermes/.env process-wide
    sys.path[:] = [host_root] + [p for p in saved if os.path.abspath(p or ".") != plugin_dir and p != ""]
    shadow = sys.modules.get("tools")
    if shadow is not None and not hasattr(shadow, "registry"):
        del sys.modules["tools"]
    try:
        run_agent = importlib.import_module("run_agent")
        turn_context = importlib.import_module("agent.turn_context")
        conversation_loop = importlib.import_module("agent.conversation_loop")
    except Exception as exc:  # pragma: no cover - host not importable here
        pytest.skip(f"hermes-agent host not importable: {exc}")
    finally:
        sys.path[:] = saved
        os.environ.clear()
        os.environ.update(saved_env)
    return run_agent, turn_context, conversation_loop


run_agent, turn_context, conversation_loop = _import_host()


@pytest.fixture
def clean_environ():
    """AIAgent() loads ~/.hermes/.env into os.environ; other tests must not inherit it."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = run_agent.AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = True
    return agent


def _history(n=30):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"request {i} " + ("detail " * 120)})
        out.append({"role": "assistant", "content": f"response {i} " + ("done " * 120)})
    return out


def test_summariser_failure_through_build_turn_context_keeps_the_turn(tmp_path, monkeypatch, clean_environ):
    agent = _agent()
    cfg = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1)
    cfg.database_path = str(tmp_path / "host.db")
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    engine.on_session_start(agent.session_id or "host-session", platform="cli", context_length=200_000)
    engine.threshold_tokens = 1  # every turn is over threshold
    engine.protect_first_n = 0
    engine.protect_last_n = 0
    agent.context_compressor = engine
    # the summariser is dead on every route
    monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: None)
    warned = []
    monkeypatch.setattr(agent, "_warn_context_overflow_blocked", lambda reason, *a, **k: warned.append(reason), raising=False)

    history = _history()
    try:
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
        # the turn goes on with its history intact (plus the new user turn)
        assert isinstance(ctx, turn_context.TurnContext)
        contents = [m.get("content") for m in ctx.messages if m.get("role") in ("user", "assistant")]
        assert all(m["content"] in contents for m in history)
        assert not any("deterministic truncation" in str(m.get("content")) for m in ctx.messages)
        # the engine armed its cooldown and the host reads it as the block reason
        state = engine.get_active_compression_failure_cooldown()
        assert state and state["remaining_seconds"] > 0
        should, reason = engine.should_compress_info(10_000)
        assert should is False and reason.startswith("cooldown:")
        assert engine._dag.get_session_nodes(engine._session_id) == []
        if warned:
            assert any(str(r).startswith("cooldown:") for r in warned)
    finally:
        engine.shutdown()
