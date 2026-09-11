"""Whether a host message exists at all — the decision upstream of every active return.

The active cleaner answered it from ``content`` and ``tool_calls`` alone, so a real assistant
turn whose display content is empty while a transport sidecar carries its payload was removed
wholesale from the replayed context, with no marker and nothing in its place. Each form the
host actually produces is proved separately here; one green example clears only itself.
"""
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.sanitize import (
    _clean_active_assistant_message,
    _should_drop_active_assistant_message,
)


def _engine(tmp_path, name="shape.db", **kw):
    cfg = LCMConfig(**kw)
    cfg.database_path = str(tmp_path / name)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e._session_id = "shape-session"
    return e


# ── #63: the existence decision reads the whole message ────────────────────────────────────


def test_the_host_redirect_placeholder_is_not_deleted():
    """Hermes' user-redirect placeholder: empty display content, hidden, and an api_content
    sidecar the host substitutes into its API copy. Judging existence from `content` alone
    removed the whole turn from the replay with no marker."""
    redirect = {
        "role": "assistant",
        "content": "",
        "display_kind": "hidden",
        "api_content": "[response interrupted]",
    }
    assert _should_drop_active_assistant_message(redirect) is False
    assert _clean_active_assistant_message(redirect) is redirect


def test_a_reasoning_sidecar_keeps_its_turn():
    """A sibling field, not a content part: `_content_carries_text` never saw the message
    dict, so a turn whose payload sat in `reasoning_content` was dropped."""
    reasoning = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "PRESERVED_REASONING_VALUE",
    }
    assert _should_drop_active_assistant_message(reasoning) is False
    assert _clean_active_assistant_message(reasoning) is reasoning


def test_an_unknown_native_carrier_keeps_its_turn():
    """Unknown provider carriers are preserved rather than risked, the same way unknown
    content blocks already are."""
    native = {
        "role": "assistant",
        "content": "",
        "audio": {"id": "aud-1", "transcript": "spoken answer"},
    }
    assert _should_drop_active_assistant_message(native) is False
    assert _clean_active_assistant_message(native) is native


def test_a_tool_call_turn_with_no_text_is_still_kept():
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "a", "type": "function",
                        "function": {"name": "t", "arguments": "{}"}}],
    }
    assert _should_drop_active_assistant_message(call) is False
    assert _clean_active_assistant_message(call) is call


def test_a_genuinely_empty_turn_is_still_dropped():
    """A turn that held nothing loses nothing when it goes, and inventing a receipt for it
    would be a false claim of removal."""
    assert _clean_active_assistant_message({"role": "assistant", "content": ""}) is None
    assert _should_drop_active_assistant_message({"role": "assistant", "content": ""}) is True


def test_recording_scaffolding_alone_does_not_resurrect_an_empty_turn():
    """When the turn happened, how it ended and how it was rendered describe the record, not
    what the turn said. A turn carrying only those still held nothing."""
    scaffolded = {
        "role": "assistant",
        "content": "",
        "timestamp": 1_700_000_000.0,
        "finish_reason": "stop",
        "display_kind": "hidden",
        "reasoning": None,
    }
    assert _clean_active_assistant_message(scaffolded) is None
    assert _should_drop_active_assistant_message(scaffolded) is True


def test_the_redirect_placeholder_survives_the_whole_active_return(tmp_path):
    """The drop is executed inside `_sanitize_active_context_messages`, which every active
    return boundary shares; nothing entered the returned context in its place."""
    e = _engine(tmp_path, context_threshold=0.95, fresh_tail_count=10)
    try:
        e.on_session_start("shape-session", context_length=200_000)
        returned = e.compress([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "start the job"},
            {"role": "assistant", "content": "", "display_kind": "hidden",
             "api_content": "[response interrupted]"},
            {"role": "user", "content": "actually, stop"},
        ])
        sidecars = [m.get("api_content") for m in returned if isinstance(m, dict)]
        assert "[response interrupted]" in sidecars
    finally:
        e.shutdown()


def test_an_empty_turn_is_not_named_as_an_internal_only_removal(tmp_path):
    """The assembly omission marker says a turn `held only internal/reasoning content`. A turn
    that held nothing did not, so counting it there is a false claim of removal."""
    e = _engine(tmp_path, "omission.db", incremental_max_depth=0)
    try:
        assembled = e._assemble_context(
            {"role": "system", "content": "sys"},
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": ""},
            ],
        )
        prefix = "\n".join(str(m.get("content")) for m in assembled)
        assert "held only internal/reasoning content" not in prefix
    finally:
        e.shutdown()
