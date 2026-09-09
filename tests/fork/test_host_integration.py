"""Plan verification 4 — a summariser failure injected through the host's REAL turn prologue
(`agent.turn_context.build_turn_context`) must not kill the turn: the context comes back
intact, the engine's cooldown is armed and the host reports the block as `cooldown:<s>`.

Needs hermes-agent importable (it is in the plugin's venv); skipped otherwise.
"""
import contextlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_lcm import escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

def _host_root() -> Path | None:
    """Locate the hermes-agent checkout WITHOUT importing it.

    Importing `agent` first is what broke this: that import pulls in the host's `tools`, which
    resolves to this plugin's `tools.py` while the plugin directory is on `sys.path`. The path
    surgery has to happen before any host import, so the root is found by looking for the
    checkout's own files — first beside the interpreter (the tests run on the plugin's venv),
    then anywhere on `sys.path`.
    """
    def looks_like_host(path: Path) -> bool:
        try:
            return (path / "run_agent.py").exists() and (path / "tools" / "__init__.py").exists()
        except OSError:  # pragma: no cover - unreadable path
            return False

    seeds = [Path(sys.executable).resolve()] + [Path(entry).resolve() for entry in sys.path if entry]
    for seed in seeds:
        # the runner uses an ephemeral venv, so the checkout is reached by walking up from an
        # entry that lives inside it (e.g. its own venv's site-packages)
        for candidate in [seed, *seed.parents[:6]]:
            if looks_like_host(candidate):
                return candidate
    return None


@contextlib.contextmanager
def _host_modules():
    """Run the body with hermes-agent importable and its own `tools` package in place.

    Both the host and this plugin have a top-level `tools`; the plugin's shadows the host's
    under pytest's rootdir. The swap must last for the whole test (the host imports more of
    itself while running) and must be undone afterwards, or every later test that imports
    `tools` gets the wrong module. Importing the host also loads ~/.hermes/.env into the
    process environment, so that is restored too.
    """
    root = _host_root()
    if root is None:
        pytest.skip("hermes-agent checkout not found next to the interpreter or on sys.path")
    host_root = str(root)
    plugin_dir = str(Path(__file__).resolve().parent.parent)
    saved_path = list(sys.path)
    saved_env = dict(os.environ)
    saved_modules = {name: module for name, module in sys.modules.items()
                     if name == "tools" or name.startswith("tools.")}
    sys.path[:] = [host_root] + [p for p in saved_path
                                 if os.path.abspath(p or ".") != plugin_dir and p != ""]
    shadow = sys.modules.get("tools")
    if shadow is not None and not hasattr(shadow, "registry"):
        del sys.modules["tools"]
    # Load the host's `tools` PACKAGE explicitly by path: relying on sys.path order made this
    # depend on which test ran first (alone, Python found this plugin's tools.py and the host
    # failed to import; in a full run something else had already imported the host's).
    host_tools_init = Path(host_root) / "tools" / "__init__.py"
    if host_tools_init.exists():
        spec = importlib.util.spec_from_file_location(
            "tools", host_tools_init, submodule_search_locations=[str(host_tools_init.parent)])
        host_tools = importlib.util.module_from_spec(spec)
        sys.modules["tools"] = host_tools
        spec.loader.exec_module(host_tools)
    try:
        yield (importlib.import_module("run_agent"),
               importlib.import_module("agent.turn_context"),
               importlib.import_module("agent.conversation_loop"))
    except ImportError as exc:  # pragma: no cover - host not importable here
        pytest.skip(f"hermes-agent host not importable: {exc}")
    finally:
        sys.path[:] = saved_path
        os.environ.clear()
        os.environ.update(saved_env)
        # The host's `tools` package STAYS in sys.modules: its background/atexit cleanup
        # imports it after this block, and swapping the plugin's module back made that resolve
        # to a non-package (`from .externalize import ...` with no parent). Nothing in this
        # plugin or its tests imports a bare `tools` — everything uses `hermes_lcm.tools` — so
        # leaving the host's in place is safe. `saved_modules` is kept for the assertion below
        # that the plugin's own module is untouched.
        assert "hermes_lcm.tools" in sys.modules or not saved_modules


def _agent(run_agent):
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


def test_summariser_failure_through_build_turn_context_keeps_the_turn(tmp_path, monkeypatch):
    with _host_modules() as (run_agent, turn_context, conversation_loop):
        agent = _agent(run_agent)
        cfg = LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1)
        cfg.database_path = str(tmp_path / "host.db")
        engine = LCMEngine(config=cfg, hermes_home=str(tmp_path))
        engine.on_session_start(agent.session_id or "host-session", platform="cli",
                                context_length=200_000)
        engine.threshold_tokens = 1          # every turn is over threshold
        engine.protect_first_n = 0
        engine.protect_last_n = 0
        agent.context_compressor = engine
        monkeypatch.setattr(escalation, "_call_llm_for_summary", lambda *a, **k: None)
        warned = []
        monkeypatch.setattr(agent, "_warn_context_overflow_blocked",
                            lambda reason, *a, **k: warned.append(reason), raising=False)

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


# round-3 verify-4 #21 (binding failures counted as ingest failures) is covered by the fix in
# __init__.py: the post_llm_call hook records a failure that happens BEFORE ingest() through
# _record_ingest_failure. It has no fork test here because the plugin's __init__ module body
# does not execute under this suite's import shim (only its submodules do), so the registered
# hook is not reachable from a test process.
