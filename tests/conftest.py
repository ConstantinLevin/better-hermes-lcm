"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import sys
import importlib
from pathlib import Path

# Make the repo root importable (for agent.context_engine etc.)
repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Register the plugin directory as a proper package
plugin_dir = Path(__file__).resolve().parent.parent
pkg_name = "hermes_lcm"

if pkg_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(plugin_dir / "__init__.py"),
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(plugin_dir)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod
    # Don't exec the module (it tries to register with ctx)
    # Just make submodules importable

    # Register each submodule
    for py_file in plugin_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"{pkg_name}.{py_file.stem}"
        if sub_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                sub_name, str(py_file),
                submodule_search_locations=[],
            )
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sub_mod.__package__ = pkg_name
            sys.modules[sub_name] = sub_mod
            setattr(mod, py_file.stem, sub_mod)
            try:
                sub_spec.loader.exec_module(sub_mod)
            except Exception:
                pass  # some modules may fail (e.g. engine needs agent)


# ── fork: betterlcm ──────────────────────────────────────────────────────────────────────
# Upstream's tests ran compactions without any LLM and silently relied on the deterministic
# L3 truncation fallback to produce "summaries". The fork removed L3 (every summariser
# failure now arms a cooldown and leaves the raw messages in place), so the suite needs an
# explicit stand-in: a deterministic mock summariser installed for every test unless
#   - the test patches `escalation._call_llm_for_summary` itself (its patch wins),
#   - the test installs its own `agent.auxiliary_client` in sys.modules (it is exercising the
#     real call path against a fake client — the mock steps aside and delegates), or
#   - LCM_TESTS_REAL_SUMMARISER=1 is set.
import json as _json
import os as _os
import pytest as _pytest

try:  # the real host module, so a test-installed stub can be told apart from it
    import agent.auxiliary_client as _real_auxiliary_client
except Exception:  # pragma: no cover - CI without hermes-agent
    _real_auxiliary_client = None

from hermes_lcm import escalation as _escalation
from hermes_lcm.tokens import count_tokens as _count_tokens

_REAL_CALL_LLM_FOR_SUMMARY = _escalation._call_llm_for_summary
_MOCK_MARKER = "[...mock summary — details available via lcm_expand...]"
_MOCK_HINT = "Expand for details about: mock summary"


def _mock_source_text(prompt):
    """The untrusted-data envelope's source content(s); the acceptance rule
    (`count_tokens(result) < source_tokens`) counts exactly this text."""
    if not isinstance(prompt, list):
        return str(prompt)
    for message in prompt:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or message.get("role") != "user":
            continue
        try:
            envelope = _json.loads(content)
            sources = envelope.get("sources") or []
            parts = [s.get("content") for s in sources if isinstance(s, dict)]
            parts = [p for p in parts if isinstance(p, str)]
            if parts:
                return "\n".join(parts)
        except Exception:
            pass
        return content
    return ""


def _fork_mock_summary(prompt, max_tokens, model="", timeout=None):
    """Deterministic, strictly shorter than its source, ends with the expand hint line."""
    if sys.modules.get("agent.auxiliary_client") is not _real_auxiliary_client:
        return _REAL_CALL_LLM_FOR_SUMMARY(prompt, max_tokens, model=model, timeout=timeout)
    text = _mock_source_text(prompt)
    source_tokens = _count_tokens(text)
    budget = max(1, min(int(max_tokens or 0) // 2, source_tokens // 2))
    chars = max(8, budget * 3)
    while True:
        if len(text) <= chars:
            body = text
        else:
            body = text[: chars * 2 // 3].rstrip() + "\n\n" + _MOCK_MARKER + "\n\n" + text[-(chars // 3):].lstrip()
        candidate = body + "\n" + _MOCK_HINT
        if _count_tokens(candidate) < source_tokens or chars <= 8:
            break
        chars //= 2
    if _count_tokens(candidate) >= source_tokens:
        # tiny source: no room for the hint line, keep the shortest non-empty head
        candidate = text[: max(1, len(text) // 3)]
    return candidate


@_pytest.fixture(autouse=True)
def _fork_mock_summariser(monkeypatch):
    if _os.environ.get("LCM_TESTS_REAL_SUMMARISER") == "1":
        yield
        return
    monkeypatch.setattr(_escalation, "_call_llm_for_summary", _fork_mock_summary)
    yield
