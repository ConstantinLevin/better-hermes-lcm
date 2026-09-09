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


def test_grep_says_which_query_tokens_it_did_not_search_for(tmp_path):
    """Audit p05 SQ01: bare Boolean words and punctuation are removed from the term list, so
    a search for "AND 🚀" or "*" quietly became a different search and its empty result read
    as an exhaustive negative."""
    from hermes_lcm import tools as lcm_tools
    cfg = LCMConfig(database_path=str(tmp_path / "terms.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("q", platform="cli", context_length=200_000)
        payload = json.loads(lcm_tools.lcm_grep({"query": "AND rocket"}, engine=e))
        assert payload["query_interpretation"]["dropped_tokens"] == ["AND"]
        assert "rocket" in payload["query_interpretation"]["terms"]

        plain = json.loads(lcm_tools.lcm_grep({"query": "rocket"}, engine=e))
        assert "query_interpretation" not in plain, "nothing was dropped, so nothing to report"
    finally:
        e.shutdown()


def test_a_search_after_a_failed_ingest_is_not_reported_as_complete(tmp_path):
    """verify-4 #11: the engine logged that the current turn could not be stored and the tool
    still answered "no matching history" — a false exhaustive negative over content that had
    just reached the plugin."""
    from hermes_lcm import tools as lcm_tools
    cfg = LCMConfig(database_path=str(tmp_path / "ingestfail.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("if", platform="cli", context_length=200_000)
        clean = json.loads(lcm_tools.lcm_grep({"query": "LATEST"}, engine=e))
        assert clean["complete"] is True

        e._record_ingest_failure("test", RuntimeError("database is locked"))
        payload = json.loads(lcm_tools.lcm_grep({"query": "LATEST"}, engine=e))
        assert payload["complete"] is False
        assert any(item["source"] == "current_turn_ingest" for item in payload["search_failures"])
        assert "database is locked" in json.dumps(payload["search_failures"])
    finally:
        e.shutdown()


def test_a_capped_raw_message_scan_is_reported_too(tmp_path):
    """verify-4 #12: the DAG scan disclosed its cap but the RAW message scan did not, so a
    capped search over 500+ matching rows still answered complete:true."""
    from hermes_lcm import search_query
    from hermes_lcm import tools as lcm_tools
    cfg = LCMConfig(database_path=str(tmp_path / "rawcap.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("rc", platform="cli", context_length=200_000)
        cap = search_query.compute_search_candidate_cap(1)
        e._store.append_batch(
            "rc", [{"role": "user", "content": f"alpha match {index}"} for index in range(cap + 40)]
        )
        e._store.commit()
        progress: dict = {}
        e._store.search("alpha", session_id="rc", limit=1, sort="relevance", progress=progress)
        assert progress["complete"] is False, progress
        assert progress["scanned_rows"] >= progress["candidate_cap"]
    finally:
        e.shutdown()


def test_an_exact_ordered_page_is_not_reported_as_a_work_cap_failure(tmp_path):
    """round-2 verify-2 #11: "give me the newest match" over 1000 matching summaries returned
    the correct newest row and reported complete=false with a work-cap warning, although the
    cap was never reached. A full page in the database's own order IS the exact answer."""
    from hermes_lcm import tools as lcm_tools
    cfg = LCMConfig(database_path=str(tmp_path / "topk.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("tk", platform="cli", context_length=200_000)
        base = time.time()
        for index in range(1000):
            e._dag.add_node(SummaryNode(session_id="tk", depth=0, summary=f"漢字 note {index}",
                                        token_count=3, source_token_count=9,
                                        source_ids=[index + 1], source_type="messages",
                                        created_at=base + index))
        progress: dict = {}
        hits = e._dag.search("漢字", session_id="tk", limit=1, sort="recency", progress=progress)
        assert [n.summary for n in hits] == ["漢字 note 999"], "the newest match"
        assert progress["complete"] is True, progress
        assert progress["work_capped"] is False, progress
        assert progress["more_available"] is True, "the corpus really does hold more"
        assert progress["scanned_rows"] < progress["candidate_cap"]

        payload = json.loads(lcm_tools.lcm_grep(
            {"query": "漢字", "sort": "recency", "limit": 1, "scope": "history"}, engine=e))
        assert payload["complete"] is True, payload
        assert "bounded_scans" not in payload
        assert payload["more_results_available"] is True
        assert "work cap" not in json.dumps(payload)
    finally:
        e.shutdown()


def test_ascii_snippets_never_pay_for_a_regex_scan_per_absent_term(monkeypatch):
    """round-2 verify-2 #12: build_snippet ran an ignore-case regex over the whole source for
    every term and re-folded the source after each miss — 104ms on a 2.2M-character source
    against 5.6ms for a plain scan, on a helper that runs per LIKE-search hit."""
    def forbidden(*args, **kwargs):  # pragma: no cover - the point is that it is not called
        raise AssertionError("ASCII content must not take the regex path")

    monkeypatch.setattr(search_query.re, "search", forbidden)
    source = ("lorem ipsum dolor sit amet " * 20_000) + " NEEDLE tail"
    snippet = search_query.build_snippet(source, ["absent1", "absent2", "needle"])
    assert "NEEDLE" in snippet
    # and a term that is absent everywhere still falls back to the head of the source
    assert search_query.build_snippet("plain ascii text", ["zzz"]).startswith("plain ascii")


def test_a_query_whose_symbol_the_index_drops_goes_to_the_scan_that_keeps_it(tmp_path):
    """round-2 verify-4 #27: the FTS term form deletes non-ASCII symbols, so "flag∀" and
    "flag∃" became the same query — a search for one returned both while reporting the original
    query and a complete result."""
    assert search_query.requires_like_fallback("flag∀") is True
    assert search_query.requires_like_fallback("ordinary query") is False
    dag = _dag(tmp_path)
    try:
        for text in ("flag∀ universal", "flag∃ existential"):
            dag.add_node(SummaryNode(session_id="s", depth=0, summary=text, token_count=3,
                                     source_token_count=9, source_ids=[1],
                                     source_type="messages", created_at=time.time()))
        hits = [node.summary for node in dag.search("flag∀", session_id="s")]
        assert hits == ["flag∀ universal"], hits
    finally:
        dag.close()


def test_recall_keeps_the_full_text_arm_s_incompleteness(tmp_path, monkeypatch):
    """round-2 verify-4 #23: the FTS arm converted a grep result carrying complete=false, its
    failures and its work caps into ([], None), so recall answered as if that arm had run
    cleanly over the whole corpus."""
    import json
    from hermes_lcm import tools as lcm_tools
    cfg = LCMConfig(database_path=str(tmp_path / "recall.db"))
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    try:
        e.on_session_start("rc", platform="cli", context_length=200_000)
        e._store.append("rc", {"role": "user", "content": "alpha the decision"}, source="cli")
        e._store.commit()

        def bounded(*args, **kwargs):
            return {
                "results": [{"store_id": 1, "session_id": "rc", "role": "user",
                             "snippet": "alpha the decision", "timestamp": 1.0}],
                "complete": False,
                "bounded_scans": [{"source": "messages", "reason": "stopped at the candidate work cap",
                                   "work_capped": True}],
            }

        monkeypatch.setattr(lcm_tools, "_lcm_grep_full_text_with_deadline", bounded)
        hits, note = lcm_tools._lcm_recall_fts_arm(
            e, "alpha", candidate_limit=10, deadline=time.monotonic() + 5)
        assert hits, "the hits that DID come back are kept"
        assert note is not None and note["complete"] is False, note
        assert note["bounded_scans"], note
    finally:
        e.shutdown()
