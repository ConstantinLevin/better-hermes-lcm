"""fork: betterlcm — the bypass path bounds ignored/stateless/auxiliary context through the
host's native compressor. Two upstream behaviours turned host decisions into loss:
BY02 an abort became a destructive trim reported as success, and BY03 one unsupported
constructor keyword sent bypassed summarisation to a different model/route than configured."""
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm import marked_loss
from hermes_lcm.engine import LCMEngine


def _bypassed_engine(tmp_path, name):
    cfg = LCMConfig(database_path=str(tmp_path / name))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e.on_session_start("bypassed", platform="cli", context_length=200_000)
    e._session_stateless = True  # the bypass path this test is about
    e.threshold_tokens = 10
    return e


def test_an_unchanged_native_return_is_not_counted_as_a_compression(tmp_path):
    e = _bypassed_engine(tmp_path, "by02.db")
    try:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first decision " + "a" * 400},
            {"role": "assistant", "content": "second decision " + "b" * 400},
            {"role": "user", "content": "newest"},
        ]

        class _AbortingCompressor:
            compression_count = 0
            _last_compress_aborted = True

            def compress(self, msgs, **_kwargs):
                return msgs

        e._host_fallback_compressor = _AbortingCompressor()
        e._host_fallback_session_id = e._bypass_lcm_session_id()
        # no cap pressure: this test is about the accounting, not the assembly bound
        e._bypass_compaction_target_tokens = lambda **_kwargs: None
        before = e.compression_count

        result = e._compress_lcm_bypassed_session(messages, current_tokens=100_000, force=True)

        blob = "\n".join(str(m.get("content")) for m in result)
        # no cap pressure here, so the host's preservation decision stands untouched
        assert "first decision" in blob and "second decision" in blob, "the abort was overridden"
        assert e.compression_count == before, "an unchanged return is not a compression"
        assert e._last_compress_aborted is True
    finally:
        e.shutdown()


def test_an_older_host_signature_drops_only_what_it_cannot_take(tmp_path, monkeypatch):
    e = _bypassed_engine(tmp_path, "by03.db")
    try:
        seen: dict = {}

        class _OlderCompressor:
            def __init__(self, model, *, threshold_percent=0.5, protect_first_n=0,
                         protect_last_n=0, quiet_mode=False, summary_model_override=None,
                         config_context_length=None):
                seen["model"] = model
                seen["summary_model_override"] = summary_model_override
                seen["config_context_length"] = config_context_length

        assert e._constructor_supported_kwargs(_OlderCompressor, {
            "model": "m", "threshold_percent": 0.5, "protect_first_n": 1, "protect_last_n": 2,
            "quiet_mode": True, "summary_model_override": "chosen-route",
            "base_url": "http://x", "api_key": "k", "config_context_length": 900_000,
            "provider": "p", "api_mode": "a",
        }) == {
            "model": "m", "threshold_percent": 0.5, "protect_first_n": 1, "protect_last_n": 2,
            "quiet_mode": True, "summary_model_override": "chosen-route",
            "config_context_length": 900_000,
        }
    finally:
        e.shutdown()


def test_the_bypass_receipt_survives_every_trimming_stage(tmp_path):
    """Audit p05 BY01: the deterministic fallback deletes messages from a session LCM does not
    store, and its omission marker was itself removable, said nothing about how much went, and
    the zero-budget stage dropped even the per-message cut marker."""
    from hermes_lcm import marked_loss
    e = _bypassed_engine(tmp_path, "by01.db")
    try:
        messages = [{"role": "system", "content": "system"}]
        messages += [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index} " + "x" * 300}
            for index in range(20)
        ]
        result = e._fallback_tail_compaction(messages, target_tokens=40)

        rendered = "\n".join(str(m.get("content")) for m in result)
        assert marked_loss.BYPASS_OMISSION_PREFIX in rendered, "the receipt was trimmed away"
        # either the full sentence or its compact last-resort form, but always the counts
        assert ("older message(s)" in rendered and "chars) were dropped" in rendered) or (
            "msg /" in rendered and "chars dropped" in rendered)
        marker = next(m for m in result if marked_loss.is_bypass_omission_marker(m))
        assert "[LCM cut]" not in str(marker["content"]), "the receipt itself was shortened"
    finally:
        e.shutdown()


def test_a_zero_budget_cut_still_says_it_cut(tmp_path):
    e = _bypassed_engine(tmp_path, "by01b.db")
    try:
        from hermes_lcm import marked_loss
        cut = e._truncate_bypass_content_value("a decision that matters", 0,
                                               suffix=marked_loss.BYPASS_FINAL_TRIM_SUFFIX)
        assert cut == marked_loss.BYPASS_FINAL_TRIM_SUFFIX
    finally:
        e.shutdown()


def test_the_receipt_never_costs_the_newest_request(tmp_path):
    """verify-2 regression #2: skipping the omission receipt at index 1 made the newest
    message — the request the agent has to answer — the next removal candidate."""
    from hermes_lcm import marked_loss
    e = _bypassed_engine(tmp_path, "by01c.db")
    try:
        messages = [
            {"role": "user", "content": "the original objective " + "o" * 200},
            {"role": "user", "content": marked_loss.bypass_omission_marker(9, 4000)},
            {"role": "user", "content": "LATEST REQUEST: what is the status?"},
        ]
        result = e._trim_bypass_compacted_to_cap(list(messages), 60)
        rendered = "\n".join(str(m.get("content")) for m in result)
        assert "LATEST REQUEST" in rendered, "the live request was deleted to keep the receipt"
    finally:
        e.shutdown()


def test_the_receipt_counts_what_actually_went(tmp_path):
    """verify-4 #17: the counts were computed before the cap loop removed more messages, so
    the receipt claimed eight dropped while nine had gone; compacting an already compact
    receipt also turned its counts into a generic sentence."""
    from hermes_lcm import marked_loss
    e = _bypassed_engine(tmp_path, "by17.db")
    try:
        messages = [{"role": "user", "content": f"turn {index} " + "t" * 400} for index in range(10)]
        result = e._fallback_tail_compaction(messages, target_tokens=60)
        receipt = next(m for m in result if marked_loss.is_bypass_omission_marker(m))
        counted, chars = marked_loss.bypass_omission_counts(receipt["content"])
        survivors = [m for m in result if not marked_loss.is_bypass_omission_marker(m)]
        assert counted == len(messages) - len(survivors), (counted, len(survivors))
        assert chars > 0

        # compaction is idempotent: the counts survive being shortened twice
        once = marked_loss.compact_bypass_omission_marker(receipt["content"])
        twice = marked_loss.compact_bypass_omission_marker(once)
        assert marked_loss.bypass_omission_counts(twice) == (counted, chars)
    finally:
        e.shutdown()


def test_the_bypass_receipt_counts_tool_call_arguments_too(tmp_path):
    """round-2 verify-4 #29: dropping an assistant turn carrying 10,000 characters of tool-call
    arguments produced a receipt saying about two characters had gone — the receipt existed and
    understated the loss by three orders of magnitude."""
    e = _bypassed_engine(tmp_path, "bypasscalls.db")
    try:
        e.protect_first_n = 1
        e.protect_last_n = 1
        big = "x" * 10_000
        messages = [
            {"role": "user", "content": "keep me"},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "terminal", "arguments": big}}]},
            {"role": "user", "content": "and me"},
        ]
        result = e._fallback_tail_compaction(messages, target_tokens=100_000)
        receipt = next(str(m.get("content") or "") for m in result
                       if marked_loss.is_bypass_omission_marker(m))
        counts = marked_loss.bypass_omission_counts(receipt)
        assert counts is not None
        assert counts[1] >= 10_000, receipt
    finally:
        e.shutdown()


def test_a_second_bypass_compaction_does_not_erase_the_first_receipt(tmp_path):
    """round-5 verify-6 #10: the refresh excluded existing receipts from its accounting and
    then rewrote every surviving receipt with counts from the latest reduction alone, so a
    second compaction erased the record of the first — two receipts both claimed the newest
    numbers and the earlier loss was gone."""
    e = _bypassed_engine(tmp_path, "bycum.db")
    try:
        first_input = [{"role": "user", "content": f"turn {i} " + "t" * 400} for i in range(10)]
        first = e._fallback_tail_compaction(first_input, target_tokens=60)
        first_receipt = next(m for m in first if marked_loss.is_bypass_omission_marker(m))
        first_counts = marked_loss.bypass_omission_counts(first_receipt["content"])
        assert first_counts[0] > 0

        second_input = first + [
            {"role": "user", "content": f"later {i} " + "u" * 400} for i in range(5)
        ]
        second = e._fallback_tail_compaction(second_input, target_tokens=60)
        receipts = [m for m in second if marked_loss.is_bypass_omission_marker(m)]
        assert len(receipts) == 1, "one cumulative receipt, not several restating one total"
        counts = marked_loss.bypass_omission_counts(receipts[0]["content"])
        assert counts[0] > first_counts[0], (first_counts, counts)
        assert counts[1] > first_counts[1], (first_counts, counts)
    finally:
        e.shutdown()
