"""fork: better-hermes-lcm — Hermes writes oversized tool results to $HERMES_HOME/cache/spillover,
names that path in the marker, and deletes the file after 24 hours. Recovery accepted only the
older <tmp>/hermes-results directory, so on the deployed configuration LCM stored the preview
and let the complete output expire: unrecoverable loss on the default setup (audit p06 I2)."""
import hashlib
import os

import sqlite3

import pytest

from hermes_lcm import escalation, ingest_protection


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
        active_messages = [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": "reading", "tool_calls": [
                {"id": "call11", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call11", "content": _marker(target, content)},
        ]
        engine._ingest_messages(active_messages)
        # the marker itself has no durable copy, so it maps to no row BY DESIGN — and the
        # compaction guard knows that shape and publishes with a marker instead of refusing
        marker_message = active_messages[-1]
        assert engine._is_unmappable_host_truncation_marker(marker_message) is True

        target.unlink()
        rows = engine._store.get_session_messages("pf")
        stored = "\n".join(str(row.get("content") or "") for row in rows)
        assert "FULL RECOVERED BODY" in stored, "the recovered bytes were thrown away"
        # the marker row keeps its own content, so replay identity still matches the host's
        # message and source mapping (hence leaf publication) still works
        assert any("<persisted-output>" in str(row.get("content") or "") for row in rows)
        archive_rows = [row for row in rows
                        if str(row.get("content") or "").startswith("[LCM recovered host output")]
        assert archive_rows

        # round-2 verify-4 #1: the archive row belonged to no summary node, so the bytes were
        # stored and yet stranded outside the graph, and the leaf covering the marker read as
        # if expansion were impossible. It is a source of the leaf now, with its own receipt.
        engine.threshold_tokens = 1
        engine._config.fresh_tail_count = 1
        engine._config.leaf_chunk_tokens = 10
        engine._config.leaf_chunk_fraction = 0.0
        engine._resolve_window_scaled_settings()
        monkeypatch.setattr(
            escalation, "_call_llm_for_summary",
            lambda *a, **k: "the rollout was reverted\nExpand for details about: rollout")
        engine.compress(active_messages + [{"role": "user", "content": "and now?"}],
                        current_tokens=500_000)
        nodes = engine._dag.get_session_nodes("pf")
        assert nodes, "no leaf was published"
        covered = {int(value) for node in nodes for value in node.source_ids}
        archive_ids = {int(row["store_id"]) for row in archive_rows}
        assert archive_ids <= covered, (archive_ids, covered)
        assert any("recovered host-output archive row" in node.summary for node in nodes)
    finally:
        engine.shutdown()


def test_a_truncated_payload_file_is_not_a_successful_empty_result(tmp_path):
    """round-2 verify-4 #33: a payload file holding only {"content_chars": 200} loaded
    successfully with content "", so a half-written artifact read as a recovered empty result
    — an empty answer where bytes are missing is exactly the unmarked loss this fork removes."""
    import json as _json
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.externalize import get_large_output_storage_dir, load_externalized_payload

    home = tmp_path / "home"
    cfg = LCMConfig(database_path=str(home / "lcm.db"))
    directory = get_large_output_storage_dir(cfg, hermes_home=str(home), create=True)
    (directory / "half.json").write_text(_json.dumps({"content_chars": 200}), encoding="utf-8")
    payload = load_externalized_payload("half.json", config=cfg, hermes_home=str(home))
    assert payload is not None
    assert payload["corrupt"] is True
    assert "truncated" in payload["corrupt_reason"]

    (directory / "short.json").write_text(
        _json.dumps({"content": "abc", "content_chars": 200}), encoding="utf-8")
    short = load_externalized_payload("short.json", config=cfg, hermes_home=str(home))
    assert short["corrupt"] is True and short["content"] == "abc"

    (directory / "good.json").write_text(
        _json.dumps({"content": "abc", "content_chars": 3}), encoding="utf-8")
    good = load_externalized_payload("good.json", config=cfg, hermes_home=str(home))
    assert good.get("corrupt") is not True and good["content"] == "abc"


def test_a_recovery_row_does_not_shift_the_replay_replacements(tmp_path, monkeypatch):
    """round-3 verify-2 #1: the extra recovered-body row was appended INLINE to a list the
    caller pairs positionally with its own messages, so every later replacement moved one
    position — an externalized document's placeholder landed on the live user request."""
    from hermes_lcm import ingest_protection
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    home = tmp_path / "hermes-home"
    directory = _spillover(tmp_path)
    content = "FULL RECOVERED BODY " + ("body " * 3000)
    target = directory / "tool_result_call77.txt"
    target.write_text(content, encoding="utf-8")
    monkeypatch.setattr(ingest_protection, "maybe_externalize_payload", lambda *a, **k: None)

    cfg = LCMConfig(database_path=str(home / "lcm.db"))
    engine = LCMEngine(config=cfg, hermes_home=str(home))
    try:
        engine.on_session_start("shift", platform="cli", context_length=200_000)
        messages = [
            {"role": "assistant", "content": "reading", "tool_calls": [
                {"id": "call77", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call77", "content": _marker(target, content)},
            {"role": "user", "content": "a document: " + ("d" * 200)},
            {"role": "user", "content": "LATEST REQUEST: cancel deployment"},
        ]
        replay = engine._ingest_messages(messages)
        assert len(replay) == len(messages)
        assert "LATEST REQUEST: cancel deployment" in str(replay[-1].get("content") or ""), replay[-1]
        rows = engine._store.get_session_messages("shift")
        assert any(str(row.get("content") or "").startswith("[LCM recovered host output")
                   for row in rows), "the recovered bytes were not archived"
        assert any("cancel deployment" in str(row.get("content") or "") for row in rows)
    finally:
        engine.shutdown()


def test_a_failed_attachment_insert_leaves_no_committable_marker(tmp_path, monkeypatch):
    """round-3 verify-2 #2: the marker row and its attached body were inserted separately with
    no rollback boundary, so a failed attachment left a committable marker-only transaction and
    the next unrelated append published the marker without the recovered body."""
    from hermes_lcm import store as lcm_store_module
    from hermes_lcm.store import MessageStore

    store = MessageStore(str(tmp_path / "attach.db"))
    try:
        def with_attachment(messages, *_args, **_kwargs):
            protected = [dict(message) for message in messages]
            attachments = {}
            for index, message in enumerate(protected):
                if str(message.get("content") or "").startswith("MARKER"):
                    attachments[index] = [{
                        "role": "tool", "tool_call_id": "c1",
                        "content": "[LCM recovered host output for tool_call_id=c1]\nBODY",
                    }]
            return protected, attachments

        monkeypatch.setattr(
            lcm_store_module, "protect_messages_for_ingest_with_attachments", with_attachment
        )
        store._conn.execute(
            "CREATE TRIGGER refuse_attachment BEFORE INSERT ON messages "
            "WHEN NEW.content LIKE '[LCM recovered host output%' "
            "BEGIN SELECT RAISE(FAIL, 'no'); END"
        )
        store.commit()
        with pytest.raises(sqlite3.IntegrityError):
            store.append("s", {"role": "tool", "tool_call_id": "c1",
                               "content": "MARKER for the recovered output"}, source="cli")
        assert store._conn.in_transaction is False
        store.append("s", {"role": "user", "content": "an unrelated later message"}, source="cli")
        store.commit()
        contents = [str(row.get("content") or "") for row in store.get_session_messages("s")]
        assert contents == ["an unrelated later message"], contents
    finally:
        store.close()


def test_one_modern_attachment_does_not_strand_a_legacy_body(tmp_path):
    """round-5 verify-6 #2: the legacy call-id fallback ran only when the WHOLE chunk had no
    explicit attachment, so a single explicitly linked output suppressed legacy recovery for
    every other row in that chunk — the legacy body stayed stored, behind the advanced
    frontier, and reachable from no node."""
    import json
    from hermes_lcm.store import MessageStore, RECOVERED_FOR_KEY

    store = MessageStore(str(tmp_path / "mixed.db"))
    try:
        legacy_owner = store.append(
            "s", {"role": "tool", "tool_call_id": "legacy", "content": "<persisted-output/>"},
            source="cli")
        legacy_body = store.append(
            "s", {"role": "tool", "tool_call_id": "legacy",
                  "content": "[LCM recovered host output for tool_call_id=legacy] LEGACY BYTES"},
            source="cli")
        modern_owner = store.append(
            "s", {"role": "tool", "tool_call_id": "modern", "content": "<persisted-output/>"},
            source="cli")
        modern_body = store.append(
            "s", {"role": "tool", "tool_call_id": "modern",
                  "content": "[LCM recovered host output for tool_call_id=modern] MODERN BYTES"},
            source="cli")
        store._conn.execute(
            "UPDATE messages SET envelope_extra = ? WHERE store_id = ?",
            (json.dumps({RECOVERED_FOR_KEY: modern_owner}), modern_body),
        )
        store.commit()

        found = store.recovered_body_ids_for_consumed_rows(
            "s", [legacy_owner, modern_owner], ["legacy", "modern"],
        )
        assert set(found) == {legacy_body, modern_body}, found

        # and a reused call id still cannot cross occurrences: the modern owner's call id is
        # claimed by an explicit link, so it is never used for a session-wide legacy sweep
        only_modern = store.recovered_body_ids_for_consumed_rows("s", [modern_owner], ["modern"])
        assert only_modern == [modern_body]
    finally:
        store.close()
