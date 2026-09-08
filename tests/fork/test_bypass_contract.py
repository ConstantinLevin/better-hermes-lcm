"""fork: betterlcm — the bypass path bounds ignored/stateless/auxiliary context through the
host's native compressor. Two upstream behaviours turned host decisions into loss:
BY02 an abort became a destructive trim reported as success, and BY03 one unsupported
constructor keyword sent bypassed summarisation to a different model/route than configured."""
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
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
