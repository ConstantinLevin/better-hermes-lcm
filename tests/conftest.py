"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import os as _bootstrap_os
import sys
import importlib
import tempfile as _bootstrap_tempfile
from pathlib import Path

# isolate the run from live storage BEFORE anything imports the plugin
# (audit E, E07). A test that builds an LCMEngine without an explicit database_path resolves
# to $HERMES_HOME/lcm.db, and inherited LCM_* variables would silently change what is being
# tested. Nothing here is a claim that the suite currently writes to live data; it removes
# the possibility.
# Stash the real home for the one test that reads a COPY of the live database. Kept in the
# environment (and set once) so a second import of this module cannot overwrite it with the
# isolated home it just installed.
_bootstrap_os.environ.setdefault("LCM_TESTS_ORIGINAL_HOME", _bootstrap_os.path.expanduser("~"))
ORIGINAL_HOME = Path(_bootstrap_os.environ["LCM_TESTS_ORIGINAL_HOME"])
_TEST_HOME = Path(_bootstrap_tempfile.mkdtemp(prefix="lcm-tests-home-"))
(_TEST_HOME / ".hermes").mkdir(parents=True, exist_ok=True)
_bootstrap_os.chmod(_TEST_HOME, 0o700)
_bootstrap_os.chmod(_TEST_HOME / ".hermes", 0o700)
_bootstrap_os.environ["HOME"] = str(_TEST_HOME)
_bootstrap_os.environ["HERMES_HOME"] = str(_TEST_HOME / ".hermes")
# LCM_TESTS_* are the harness's own controls (the real-summariser switch, the stashed home),
# not plugin configuration: scrubbing them disabled the documented real-summary mode.
# LCM_BETA_TARGET and LCM_REQUIRE_HOST are harness controls too — they select which tests run and
# how strictly, and no production module reads either — but they were named without the prefix in
# the issues and on the documented command lines, so they are exempted by name rather than
# renamed. Adding a plugin setting here would be a mistake; adding a runner switch is not.
_HARNESS_CONTROL_VARS = ("LCM_BETA_TARGET", "LCM_REQUIRE_HOST")
for _inherited in [
    name for name in _bootstrap_os.environ
    if name.startswith("LCM_")
    and not name.startswith("LCM_TESTS_")
    and name not in _HARNESS_CONTROL_VARS
]:
    _bootstrap_os.environ.pop(_inherited, None)
assert str(Path.home()) == str(_TEST_HOME), "tests must not resolve HOME to the live account"

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


# ──  ──────────────────────────────────────────────────────────────────────
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


def manifest_version() -> str:
    """The version this checkout ships, read straight from ``plugin.yaml``.

    Eight tests hardcoded ``1.0.0-rc.1`` while the manifest had moved to ``1.1.0-beta.2``, so
    the release-identity gate had been red for two releases and nobody could cut the next one.
    Deriving the number here is what stops that drifting again.

    Deliberately NOT ``runtime_identity._plugin_metadata()``: that is the production reader
    those tests exist to check, and comparing it against itself would assert nothing.
    """
    manifest = Path(__file__).resolve().parent.parent / "plugin.yaml"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    raise AssertionError(f"{manifest} declares no version")


# ── beta target-state assertions (issue #19) ─────────────────────────────────────────────────
# `beta_target` marks an assertion that says what a beta preservation fix MUST achieve. They are
# written against a tree where those fixes do not exist, so they FAIL here by design and would
# otherwise turn the default suite red for every group still working on them.
#
# The gate is an env var rather than a quiet deselect because a skipped one has to say what it is
# not testing. An empty result that reads as "there is nothing" is the exact dishonesty this fork
# forbids, and a silently deselected contract is that dishonesty inside a test runner.
#
#   run them:      LCM_BETA_TARGET=1 bash scripts/test.sh tests/ -m beta_target
#   release gate:  scripts/validate_release.sh --full  (runs the selection above)
BETA_TARGET_ENV = "LCM_BETA_TARGET"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "beta_target(issue): an assertion for a beta fix that is not in this tree yet. "
        f"Runs only with {BETA_TARGET_ENV}=1; select the set with `-m beta_target`.",
    )


def pytest_collection_modifyitems(config, items):
    if _os.environ.get(BETA_TARGET_ENV) == "1":
        return
    for item in items:
        marker = item.get_closest_marker("beta_target")
        if marker is None:
            continue
        issue = str(marker.args[0]) if marker.args else "#19"
        item.add_marker(_pytest.mark.skip(reason=(
            f"NOT RUN: beta target-state assertion for {issue} (tracked by #19). It encodes what "
            f"the fix must achieve and fails until that fix lands. Run it with "
            f"{BETA_TARGET_ENV}=1 and `-m beta_target`."
        )))


@_pytest.fixture(autouse=True)
def _fork_mock_summariser(monkeypatch):
    # autouse because upstream's compaction tests ship no summariser and
    # relied on L3 deterministic truncation, which this fork removed. Drop this fixture and they
    # fail with SummaryUnavailableError; the answer is this mock, never restoring L3.
    if _os.environ.get("LCM_TESTS_REAL_SUMMARISER") == "1":
        yield
        return
    monkeypatch.setattr(_escalation, "_call_llm_for_summary", _fork_mock_summary)
    yield
