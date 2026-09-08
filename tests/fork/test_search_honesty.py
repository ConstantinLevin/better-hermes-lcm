"""fork: betterlcm — a search that stopped early must say so, and it must be able to find the
text it holds. Upstream returned a bounded scan the way it returned an exhaustive one, routed
supplementary CJK to an index that cannot spell it, and built snippets at offsets taken from a
case-folded copy of the text (audit p05 SQ02 / SQ03 / SQ05)."""
import json
import time

from hermes_lcm import search_query
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine


def _dag(tmp_path):
    return SummaryDAG(str(tmp_path / "search.db"))


def test_supplementary_ideographs_route_to_the_scan_that_can_find_them(tmp_path):
    # unicode61 does not index CJK, and a supplementary ideograph is still CJK
    assert search_query.requires_like_fallback("𠀀") is True
    assert search_query.requires_like_fallback("漢字") is True
    dag = _dag(tmp_path)
    try:
        dag.add_node(SummaryNode(session_id="s", depth=0, summary="𠀀𠀁 the decision",
                                 token_count=3, source_token_count=9, source_ids=[1],
                                 source_type="messages", created_at=time.time()))
        assert [n.summary for n in dag.search("𠀀", session_id="s")] == ["𠀀𠀁 the decision"]
    finally:
        dag.close()


def test_a_snippet_survives_length_changing_case_folding(tmp_path):
    text = "İ" * 100 + " NEEDLE here"
    snippet = search_query.build_snippet(text, ["needle"])
    assert "NEEDLE" in snippet, "the snippet held neither the match nor its surroundings"


def test_a_capped_scan_reports_itself_as_incomplete(tmp_path):
    cfg = LCMConfig(database_path=str(tmp_path / "capped.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    dag = e._dag
    try:
        cap = search_query.compute_search_candidate_cap(1)
        for index in range(cap + 20):
            dag.add_node(SummaryNode(session_id="s", depth=0, summary=f"alpha match {index}",
                                     token_count=3, source_token_count=9, source_ids=[index + 1],
                                     source_type="messages", created_at=time.time()))
        progress: dict = {}
        dag.search("alpha", session_id="s", limit=1, source="messages", progress=progress)
        assert progress["complete"] is False
        assert progress["scanned_rows"] <= progress["candidate_cap"]

        exhaustive: dict = {}
        dag.search("nothingmatchesthis", session_id="s", limit=1, progress=exhaustive)
        assert exhaustive["complete"] is True, "a genuinely exhausted scan is complete"
    finally:
        e.shutdown()


def test_grep_tells_the_agent_when_a_scan_stopped_at_a_work_cap(tmp_path, monkeypatch):
    from hermes_lcm import tools as lcm_tools
    cfg = LCMConfig(database_path=str(tmp_path / "grep.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("g", platform="cli", context_length=200_000)
        real_search = e._dag.search

        def capped(*args, progress=None, **kwargs):
            result = real_search(*args, **kwargs)
            if progress is not None:
                progress.update({"complete": False, "scanned_rows": 500, "candidate_cap": 500,
                                 "path": "fts"})
            return result

        monkeypatch.setattr(e._dag, "search", capped)
        payload = json.loads(lcm_tools.lcm_grep({"query": "anything"}, engine=e))
        assert payload["complete"] is False
        assert payload["bounded_scans"][0]["source"] == "summaries"
        assert "not evidence of absence" in payload["search_note"]
    finally:
        e.shutdown()
