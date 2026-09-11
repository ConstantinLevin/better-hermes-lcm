"""The two scorers behind scripts/e2e_index_navigation.py, on inputs whose score is known.

A scorer that mis-scores produces confident wrong numbers, which is worse than no numbers, so
every case here is a hand-labelled input: half of them are known-GOOD and half are known-WRONG
(a false success claim, a smoothed hedge, a missed topic, an empty reader reply). The scorer has
to separate them, and — the part that matters for this fork — it has to refuse to score what it
cannot score instead of reporting a zero or a hundred per cent.
"""

from benchmarking.nav_fidelity import (
    NavigationCase,
    ReaderTrace,
    ScoredText,
    StateClaim,
    score_fidelity,
    score_navigation,
)


# ── navigation (#2) ─────────────────────────────────────────────────────────────────────────

def _case(**overrides) -> NavigationCase:
    base = dict(
        question_id="q1",
        question="Which retry strategy was dropped, and what replaced it?",
        expected_store_ids=(11,),
        covering_node_ids={11: (7,)},
        raw_in_context_store_ids=(),
        evidence_snippets={11: "fixed 900ms interval instead"},
        must_mention=("exponential backoff",),
        must_not_claim=(r"exponential backoff (was )?(adopted|chosen)",),
        category="discarded-alternative",
    )
    base.update(overrides)
    return NavigationCase(**base)


def _trace(**overrides) -> ReaderTrace:
    base = dict(
        question_id="q1",
        status="ok",
        detail="",
        chosen_node_ids=(7,),
        recovered_store_ids=(11,),
        recovered_text="we rejected exponential backoff and took a fixed 900ms interval instead",
        answer="Exponential backoff was rejected; a fixed 900ms interval replaced it.",
        evidence_defects=(),
    )
    base.update(overrides)
    return ReaderTrace(**base)


def test_navigating_to_the_covering_node_and_recovering_the_line_scores_both_recalls():
    result = score_navigation([_case()], [_trace()])

    assert result["node_recall"] == {"expected": 1, "hit": 1, "fraction": 1.0}
    assert result["source_recall"] == {"expected": 1, "hit": 1, "fraction": 1.0}
    assert result["scored"] == 1
    assert result["complete"] is True


def test_choosing_the_wrong_node_is_a_reader_miss_not_an_evidence_defect():
    trace = _trace(chosen_node_ids=(99,), recovered_store_ids=(), recovered_text="")

    result = score_navigation([_case()], [trace])

    assert result["node_recall"]["hit"] == 0
    assert result["source_recall"]["hit"] == 0
    assert result["evidence_defects"] == {}
    assert result["per_case"][0]["reasons"] == ["reader:wrong_node"]
    assert result["complete"] is True


def test_an_empty_reader_reply_is_unscored_rather_than_a_zero():
    """An empty result that reads as 'there is nothing' is the defect this fork removes."""
    trace = _trace(status="empty", detail="model returned no content",
                   chosen_node_ids=(), recovered_store_ids=(), recovered_text="", answer="")

    result = score_navigation([_case()], [trace])

    assert result["scored"] == 0
    assert result["unscored_reader_unavailable"] == 1
    assert result["node_recall"]["fraction"] is None
    assert result["source_recall"]["fraction"] is None
    assert result["complete"] is False
    assert any("q1" in reason for reason in result["incomplete_reasons"])


def test_a_source_reachable_from_no_delivered_node_is_an_evidence_defect():
    """#5/#31/#35/#37/#56/#63/#67 remove the reader's evidence; that is not a model failure."""
    case = _case(covering_node_ids={11: ()})
    trace = _trace(chosen_node_ids=(), recovered_store_ids=(), recovered_text="", answer="")

    result = score_navigation([case], [trace])

    assert result["unscored_evidence_defect"] == 1
    assert result["evidence_defects"] == {"evidence:no_frontier_coverage": 1}
    assert result["node_recall"]["expected"] == 0
    assert result["node_recall"]["fraction"] is None
    assert result["complete"] is False


def test_a_false_success_claim_in_the_answer_is_counted():
    trace = _trace(answer="Exponential backoff was adopted for the ledger-sync retries.")

    result = score_navigation([_case()], [trace])

    assert result["false_assertions"]["found"] == 1
    assert result["false_assertions"]["sample"][0]["question_id"] == "q1"


def test_a_topic_the_answer_never_mentions_is_counted_as_missed():
    trace = _trace(answer="A fixed interval was chosen.")

    result = score_navigation([_case()], [trace])

    assert result["missed_topics"]["missed"] == 1
    assert result["missed_topics"]["sample"][0]["topic"] == "exponential backoff"


def test_answering_without_expanding_anything_is_recorded_separately():
    trace = _trace(chosen_node_ids=(), recovered_store_ids=(), recovered_text="")

    result = score_navigation([_case()], [trace])

    assert result["answered_without_expansion"] == 1
    assert result["node_recall"]["hit"] == 0


def test_a_source_still_verbatim_in_the_delivered_context_needs_no_navigation():
    case = _case(raw_in_context_store_ids=(11,))
    trace = _trace(chosen_node_ids=(), recovered_store_ids=(), recovered_text="")

    result = score_navigation([case], [trace])

    assert result["not_applicable_raw_in_context"] == 1
    assert result["node_recall"]["expected"] == 0


def test_a_missing_line_that_was_paged_out_is_an_evidence_defect_not_a_reader_miss():
    """lcm_expand returns one page; the rest of the node is behind a cursor (#52 territory).

    A labelled line that did not come back from a PAGED recovery may simply be on the next
    page. Booking that as "the reader navigated badly" would attribute another issue's bound
    to the model.
    """
    trace = _trace(recovered_text="we rejected exponential backoff",
                   observations=("evidence:paged_result",))

    result = score_navigation([_case()], [trace])

    assert result["unscored_evidence_defect"] == 1
    assert result["evidence_defects"] == {"evidence:bounded_recovery": 1}
    assert result["source_recall"]["expected"] == 0
    assert result["complete"] is False


def test_a_missing_line_from_a_recovery_carrying_a_cut_marker_is_also_an_evidence_defect():
    """An [LCM ...] marker in the recovered text says content was replaced (#50/#56)."""
    trace = _trace(recovered_text="we rejected exponential backoff [LCM elided 900 of 2000 chars]",
                   observations=("evidence:truncated_or_marked",))

    result = score_navigation([_case()], [trace])

    assert result["evidence_defects"] == {"evidence:bounded_recovery": 1}
    assert result["per_case"][0]["bounded_by"] == ["evidence:truncated_or_marked"]


def test_a_paged_recovery_that_did_return_the_line_is_scored_normally():
    trace = _trace(observations=("evidence:paged_result",))

    result = score_navigation([_case()], [trace])

    assert result["scored"] == 1
    assert result["source_recall"] == {"expected": 1, "hit": 1, "fraction": 1.0}
    assert result["observations"] == {"evidence:paged_result": 1}


def test_expanding_the_whole_dag_shows_up_as_low_node_precision():
    """Recall alone rewards a reader that expands everything; precision exposes it."""
    trace = _trace(chosen_node_ids=(7, 12, 13, 14))

    result = score_navigation([_case()], [trace])

    assert result["node_recall"]["fraction"] == 1.0
    assert result["node_precision"] == {"expanded": 4, "needed": 1, "fraction": 0.25}


def test_descending_through_a_parent_to_reach_the_holding_leaf_costs_no_precision():
    """Once the frontier is condensed, reaching a leaf REQUIRES opening its ancestors."""
    case = _case(path_node_ids=(2, 5))
    trace = _trace(chosen_node_ids=(2, 5, 7))

    result = score_navigation([case], [trace])

    assert result["node_precision"] == {"expanded": 3, "needed": 3, "fraction": 1.0}


def test_the_navigation_result_carries_its_own_recalls_and_no_fidelity_rate():
    """#2 and #8 are separate results; one rate covering both would hide either."""
    result = score_navigation([_case()], [_trace()])

    assert set(result) >= {"node_recall", "source_recall", "missed_topics", "false_assertions"}
    assert "false_claims" not in result
    assert "fidelity" not in result


# ── fidelity (#8) ───────────────────────────────────────────────────────────────────────────

STARTED = StateClaim(
    claim_id="migrate-started",
    entity="migrate_ledger_v3",
    true_state="started",
    source_store_ids=(11,),
    truth_patterns=(r"migrate_ledger_v3[^.]{0,120}\bstart(ed)?\b",),
    forbidden_patterns=(r"migrate_ledger_v3[^.]{0,120}\b(succeed(ed)?|completed|finished)\b",),
    hedge_patterns=(r"\b(unknown|unconfirmed|not confirmed)\b",),
)


def _text(**overrides) -> ScoredText:
    base = dict(
        text_id="node:7",
        kind="leaf",
        depth=0,
        text="",
        claim_ids=("migrate-started",),
        source_carries_claim={"migrate-started": True},
        entity_coverage={"fraction": 1.0, "entities": 2, "present": 2},
    )
    base.update(overrides)
    return ScoredText(**base)


def test_a_started_migration_rendered_as_succeeded_is_a_false_claim():
    """The headline failure: 'was started' turning into 'succeeded'."""
    text = _text(text="migrate_ledger_v3 succeeded and the shard is clean.")

    result = score_fidelity([STARTED], [text])

    assert result["false_claims"]["count"] == 1
    assert result["false_claims"]["sample"][0]["claim_id"] == "migrate-started"
    assert result["omitted_statements"]["count"] == 0
    assert result["complete"] is True


def test_the_false_claim_sample_shows_the_matched_span_not_the_whole_block():
    """A summary is one unbroken block as often as not; a 200-char head shows nothing."""
    text = _text(text="Index over 40 source sentences: " + ("unrelated filler, " * 30)
                      + "migrate_ledger_v3 succeeded, and the shard is clean")

    result = score_fidelity([STARTED], [text])

    evidence = result["false_claims"]["sample"][0]["evidence"]
    assert "migrate_ledger_v3 succeeded" in evidence
    assert len(evidence) < 200


def test_a_hedge_far_from_the_claim_does_not_excuse_it():
    """Hedge suppression is local; over a whole block one stray 'unknown' would hide a lie."""
    text = _text(text="migrate_ledger_v3 succeeded, and " + ("x" * 400)
                      + ", the paging rota is unknown")

    result = score_fidelity([STARTED], [text])

    assert result["false_claims"]["count"] == 1


def test_a_preserved_hedge_counts_as_correctly_named_uncertainty():
    text = _text(text="migrate_ledger_v3 was started; whether it completed is unknown.")

    result = score_fidelity([STARTED], [text])

    assert result["correctly_named_uncertainty"]["count"] == 1
    assert result["false_claims"]["count"] == 0


def test_a_statement_dropped_from_the_summary_is_an_omission_not_a_false_claim():
    text = _text(text="The team discussed shard rebalancing and paging rotas.")

    result = score_fidelity([STARTED], [text])

    assert result["omitted_statements"]["count"] == 1
    assert result["false_claims"]["count"] == 0
    assert result["complete"] is False


def test_a_mention_whose_state_cannot_be_read_is_undetermined_not_faithful():
    """A smoothed sentence that neither hedges nor upgrades must not score as faithful."""
    text = _text(text="migrate_ledger_v3 came up during the incident review.")

    result = score_fidelity([STARTED], [text])

    assert result["undetermined"]["count"] == 1
    assert result["faithful"] == 0
    assert result["false_claims"]["count"] == 0
    assert result["complete"] is False


def test_a_claim_missing_from_the_texts_own_sources_is_an_evidence_defect():
    """If the distinguishing text never reached the summariser, the model did not lose it."""
    text = _text(text="Nothing about the migration here.",
                 source_carries_claim={"migrate-started": False})

    result = score_fidelity([STARTED], [text])

    assert result["unscored_evidence_defect"]["count"] == 1
    assert result["unscored_evidence_defect"]["labels"] == {
        "evidence:distinguishing_field_absent": 1
    }
    assert result["omitted_statements"]["count"] == 0
    assert result["complete"] is False


def test_leaf_and_condensation_results_are_reported_separately():
    """Condensation summarises summary text — where a hedge is likeliest to be smoothed."""
    leaf = _text(text_id="node:7", kind="leaf", depth=0,
                 text="migrate_ledger_v3 was started; the outcome is unknown.")
    parent = _text(text_id="node:9", kind="condensation", depth=1,
                   text="migrate_ledger_v3 completed during the window.")

    result = score_fidelity([STARTED], [leaf, parent])

    assert result["by_kind"]["leaf"]["correctly_named_uncertainty"] == 1
    assert result["by_kind"]["leaf"]["false_claims"] == 0
    assert result["by_kind"]["condensation"]["false_claims"] == 1
    assert result["by_depth"]["1"]["false_claims"] == 1


def test_the_entity_coverage_baseline_travels_beside_the_fidelity_score():
    """coverage_doctor is the contrast, not a substitute: same identifier, wrong state."""
    text = _text(text="migrate_ledger_v3 succeeded.",
                 entity_coverage={"fraction": 1.0, "entities": 2, "present": 2})

    result = score_fidelity([STARTED], [text])

    assert result["entity_coverage_baseline"] == [
        {"text_id": "node:7", "kind": "leaf", "depth": 0,
         "coverage": {"fraction": 1.0, "entities": 2, "present": 2}}
    ]
    assert result["false_claims"]["count"] == 1
