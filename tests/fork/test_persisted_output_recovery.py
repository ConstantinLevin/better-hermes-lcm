"""fork: betterlcm — Hermes writes oversized tool results to $HERMES_HOME/cache/spillover,
names that path in the marker, and deletes the file after 24 hours. Recovery accepted only the
older <tmp>/hermes-results directory, so on the deployed configuration LCM stored the preview
and let the complete output expire: unrecoverable loss on the default setup (audit p06 I2)."""
import hashlib
import os

import pytest

from hermes_lcm import ingest_protection


def _marker(path, content, preview_chars=120):
    """The host's own <persisted-output> shape (tools/tool_result_storage.py)."""
    preview = content[:preview_chars]
    has_more = len(content) > preview_chars
    return (
        "<persisted-output>\n"
        f"This tool result was too large ({len(content):,} characters, "
        f"{len(content) / 1024:.1f} KB).\n"
        f"Full output saved to: {path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n"
        "Recovery: page through the saved file with read_file (offset/limit) or "
        "process it with execute_code — do NOT re-request the same data from the "
        "remote API; the full result is already on disk.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        + preview + ("\n..." if has_more else "")
        + "\n</persisted-output>"
    )


def _spillover(tmp_path):
    directory = tmp_path / "hermes-home" / "cache" / "spillover"
    directory.mkdir(parents=True)
    os.chmod(tmp_path / "hermes-home", 0o700)
    return directory


def test_a_marker_pointing_at_the_host_spillover_directory_is_recovered(tmp_path):
    directory = _spillover(tmp_path)
    content = "DECISION: cancel the launch\n" + ("payload " * 5000)
    target = directory / "tool_result_call7.txt"
    target.write_text(content, encoding="utf-8")

    marker = _marker(target, content)
    recovered = ingest_protection.recover_hermes_persisted_output(
        marker, str(tmp_path / "hermes-home")
    )
    assert recovered == content, "the complete host output was not recovered"


def test_a_marker_pointing_anywhere_else_is_still_refused(tmp_path):
    elsewhere = tmp_path / "somewhere"
    elsewhere.mkdir()
    content = "not a host directory " * 500
    target = elsewhere / "leak.txt"
    target.write_text(content, encoding="utf-8")

    marker = _marker(target, content)
    assert ingest_protection.recover_hermes_persisted_output(
        marker, str(tmp_path / "hermes-home")
    ) is None


def test_the_recovered_output_is_copied_durably_even_with_externalization_disabled(tmp_path):
    """The point of recovery: the host deletes its spillover file after 24 hours. Upstream's
    generic externalization flag is off by default, so the recovered bytes were never copied
    and the archive kept a preview of an output that no longer existed (audit p06 I2)."""
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    home = tmp_path / "hermes-home"
    directory = _spillover(tmp_path)
    content = "DECISION: cancel the launch\n" + ("payload " * 5000)
    target = directory / "tool_result_call9.txt"
    target.write_text(content, encoding="utf-8")

    cfg = LCMConfig(database_path=str(home / "lcm.db"),
                    large_output_externalization_enabled=False)
    engine = LCMEngine(config=cfg, hermes_home=str(home))
    try:
        engine.on_session_start("po", platform="cli", context_length=200_000)
        engine._ingest_messages([
            {"role": "user", "content": "read the log"},
            {"role": "assistant", "content": "reading", "tool_calls": [
                {"id": "call9", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call9", "content": _marker(target, content)},
        ])
        # the host's file expires
        target.unlink()

        payloads = list(home.rglob("*.json"))
        assert payloads, "no durable copy was made; the output expires with the host file"
        texts = [path.read_text(encoding="utf-8") for path in payloads]
        assert any("DECISION: cancel the launch" in text for text in texts), texts[0][:400]
    finally:
        engine.shutdown()


def test_recovered_bytes_are_kept_even_when_the_durable_copy_cannot_be_written(tmp_path, monkeypatch):
    """verify-4 #18: when the payload write failed, ingest fell back to the host's marker and
    the recovered bytes — already in hand — were discarded. The host then deletes its file."""
    from hermes_lcm import ingest_protection
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    home = tmp_path / "hermes-home"
    directory = _spillover(tmp_path)
    content = "FULL RECOVERED BODY: the rollout was reverted\n" + ("body " * 3000)
    target = directory / "tool_result_call11.txt"
    target.write_text(content, encoding="utf-8")

    monkeypatch.setattr(ingest_protection, "maybe_externalize_payload", lambda *a, **k: None)
    cfg = LCMConfig(database_path=str(home / "lcm.db"))
    engine = LCMEngine(config=cfg, hermes_home=str(home))
    try:
        engine.on_session_start("pf", platform="cli", context_length=200_000)
        engine._ingest_messages([
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": "reading", "tool_calls": [
                {"id": "call11", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call11", "content": _marker(target, content)},
        ])
        target.unlink()
        stored = "\n".join(str(row.get("content") or "")
                           for row in engine._store.get_session_messages("pf"))
        assert "FULL RECOVERED BODY" in stored, "the recovered bytes were thrown away"
    finally:
        engine.shutdown()
