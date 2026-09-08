"""fork: betterlcm — nothing may be lost before the summariser sees it, and when a route does
fail the reason has to survive so the leaf-rescue path can act on it (audit p05 PB01 / ES03)."""
import json

import pytest

from hermes_lcm import escalation
from hermes_lcm.errors import SummaryUnavailableError
from hermes_lcm.prompt_boundary import build_untrusted_data_messages


def test_the_envelope_carries_an_escape_heavy_source_whole():
    source = 'quote=" slash=\\ tab=\t newline=\n control=\x01 DECISION: cancel the launch\n' * 200
    messages = build_untrusted_data_messages(
        operation="lcm_summary_l1",
        system_instructions="Summarize.",
        sources=[{"provenance": {"source_type": "messages"}, "content": source}],
        source_content_token_budget=10,  # far below what escaping needs
    )
    envelope = json.loads(messages[1]["content"])
    assert envelope["sources"][0]["content"] == source
    assert "content_truncated" not in envelope["sources"][0]
    assert "source reduced" not in messages[1]["content"]
    assert source.count("DECISION") == envelope["sources"][0]["content"].count("DECISION")


def test_a_capacity_failure_reaches_the_caller_as_a_capacity_failure(monkeypatch):
    def refuse(prompt, max_tokens, model="", timeout=None):
        raise RuntimeError("This model's maximum context length is 8192 tokens")

    monkeypatch.setattr(escalation, "_call_llm_for_summary", refuse)
    with pytest.raises(SummaryUnavailableError) as raised:
        escalation.summarize_with_escalation("some source text", source_tokens=50, token_budget=20)
    assert "maximum context length" in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_the_engine_reads_the_cause_chain_when_deciding_to_retry_smaller(tmp_path):
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    cfg = LCMConfig(database_path=str(tmp_path / "rescue.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        wrapped = SummaryUnavailableError("summariser unavailable after L1/L2")
        wrapped.__cause__ = RuntimeError("maximum context length exceeded")
        assert e._is_retry_worthy_leaf_summary_error(wrapped) is True
        assert e._is_retry_worthy_leaf_summary_error(
            SummaryUnavailableError("no route configured")) is False
    finally:
        e.shutdown()


def test_a_summary_cut_off_at_the_generation_limit_is_not_accepted(monkeypatch):
    """Audit p05 ES01: the route's finish status was never read, so a summary that stopped at
    the generation limit became a durable node — an index missing everything after the cut."""
    import sys
    from types import ModuleType, SimpleNamespace

    def _install(finish_reason):
        module = ModuleType("agent.auxiliary_client")
        module.call_llm = lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Topic A: decided X."),
            finish_reason=finish_reason)])
        monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)

    _install("length")
    assert escalation._call_llm_for_summary("summarize this", 200) is None

    _install("stop")
    assert escalation._call_llm_for_summary("summarize this", 200) == "Topic A: decided X."


def test_an_acknowledgement_is_not_a_summary(monkeypatch):
    """Audit p05 ES06: acceptance tested only that the reply was SMALLER than the source, so
    "OK" was a successful summary of a chunk holding a decision, a rejection and a fix."""
    replies = ["OK", "Sure.", "I can't help with that.", "Decision: cancel launch; fix merged."]
    seen = []

    def route(prompt, max_tokens, model="", timeout=None):
        seen.append(len(seen))
        return replies[min(len(seen) - 1, len(replies) - 1)]

    monkeypatch.setattr(escalation, "_call_llm_for_summary", route)
    for rejected in replies[:3]:
        assert escalation._is_index_shaped_summary(rejected) is False, rejected
    assert escalation._is_index_shaped_summary(replies[3]) is True

    monkeypatch.setattr(escalation, "_call_llm_for_summary",
                        lambda *a, **k: "OK")
    with pytest.raises(SummaryUnavailableError) as raised:
        escalation.summarize_with_escalation("a decision, a rejection and a fix",
                                             source_tokens=200, token_budget=50)
    assert "index" in str(raised.value)


def test_extraction_failure_is_not_reported_as_nothing_to_extract(tmp_path, monkeypatch):
    """Audit p05 EX08/EX06: a provider failure and a truncated generation both returned None,
    which the caller read as a successful "nothing worth extracting"."""
    from hermes_lcm import extraction as ext
    from hermes_lcm.errors import ExtractionUnavailableError

    def down(prompt, model="", timeout=None):
        raise ExtractionUnavailableError("extraction call failed: service down")

    monkeypatch.setattr(ext, "_call_extraction_llm", down)
    assert ext.extract_before_compaction("[USER]: something", str(tmp_path / "notes")) is False
    assert not list((tmp_path / "notes").glob("*.md")) if (tmp_path / "notes").exists() else True

    monkeypatch.setattr(ext, "_call_extraction_llm", lambda *a, **k: "NOTHING_TO_EXTRACT")
    assert ext.extract_before_compaction("[USER]: something", str(tmp_path / "notes")) is True


def test_an_extraction_note_names_the_rows_it_came_from(tmp_path, monkeypatch):
    """Audit p05 EX07: notes carried a wall-clock header and nothing that ties the bullets to
    the segment they were derived from, even across several passes in one session."""
    from hermes_lcm import extraction as ext

    monkeypatch.setattr(ext, "_call_extraction_llm", lambda *a, **k: "- Decided: ship on Friday")
    assert ext.extract_before_compaction(
        "[USER]: ship it", str(tmp_path / "notes"), session_id="s1", source_store_ids=[7, 8, 9]
    ) is True
    note = next((tmp_path / "notes").glob("*.md")).read_text()
    assert "store_ids=7, 8, 9" in note
    assert "sha256:" in note
    assert "lcm_expand(store_id=…)" in note


def test_a_shortened_focus_says_how_much_it_lost(monkeypatch):
    """Audit p05 ES05: the focus was cut at 160 characters with a bare ellipsis, so a
    qualifier past that point silently changed what the summariser was asked to emphasise."""
    long_focus = "migrate the store " * 20 + "but only for the staging cluster"
    shortened = escalation._normalized_focus_topic(long_focus)
    assert shortened.endswith("chars shown]")
    assert str(len(" ".join(long_focus.split()))) in shortened
