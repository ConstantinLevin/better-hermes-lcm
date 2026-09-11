"""The two scorers behind scripts/e2e_index_navigation.py, on inputs whose score is known.

A scorer that mis-scores produces confident wrong numbers, which is worse than no numbers, so
every case here is a hand-labelled input: half of them are known-GOOD and half are known-WRONG
(a false success claim, a smoothed hedge, a missed topic, an empty reader reply). The scorer has
to separate them, and — the part that matters for this fork — it has to refuse to score what it
cannot score instead of reporting a zero or a hundred per cent.
"""

import pytest

from benchmarking.nav_fidelity import (
    CorpusPatternError,
    RecoveryCause,
    attribute_recovery_defects,
    recovery_causes,
    NavigationCase,
    ReaderTrace,
    ScoredText,
    StateClaim,
    score_fidelity,
    score_navigation,
    validate_patterns,
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


def test_a_bounded_recovery_withdraws_only_the_missing_line_not_the_whole_case():
    """Withdrawing the case would drop node recall too, keeping only questions already right.

    On the model path every leaf expansion is paged, so a case-wide withdrawal would remove
    every question the reader got wrong from both denominators and walk the headline fractions
    toward 1.0 by selection.
    """
    case = _case(
        expected_store_ids=(11, 12),
        covering_node_ids={11: (7,), 12: (7,)},
        evidence_snippets={11: "fixed 900ms interval instead", 12: "the second labelled line"},
    )
    trace = _trace(bounded_store_ids={12: ("evidence:paged_result",)})

    result = score_navigation([case], [trace])

    assert result["scored"] == 1
    assert result["node_recall"] == {"expected": 2, "hit": 2, "fraction": 1.0}
    assert result["source_recall"] == {"expected": 1, "hit": 1, "fraction": 1.0}
    assert result["source_recall_withdrawn"]["count"] == 1
    assert result["evidence_defects"] == {"evidence:bounded_recovery": 1}
    assert result["complete"] is False


def test_a_bounded_recovery_from_a_node_the_reader_never_opened_is_still_a_miss():
    """A bound only excuses a line the reader actually went to the right place for."""
    trace = _trace(chosen_node_ids=(99,), recovered_text="",
                   bounded_store_ids={11: ("evidence:paged_result",)})

    result = score_navigation([_case()], [trace])

    assert result["node_recall"] == {"expected": 1, "hit": 0, "fraction": 0.0}
    assert result["source_recall"] == {"expected": 1, "hit": 0, "fraction": 0.0}
    assert result["evidence_defects"] == {}


def test_a_source_reachable_from_no_delivered_node_withdraws_only_itself():
    """An uncovered source must not take the rest of its question down with it.

    `no_frontier_coverage` fires exactly where another group's preservation fix is still
    incomplete, i.e. on the hardest questions of a candidate commit. Withdrawing the whole
    case there is the same upward selection bias as withdrawing it for a paged recovery.
    """
    case = _case(
        expected_store_ids=(11, 12),
        covering_node_ids={11: (7,), 12: ()},
        evidence_snippets={11: "fixed 900ms interval instead", 12: "the second labelled line"},
    )

    result = score_navigation([case], [_trace()])

    assert result["scored"] == 1
    assert result["node_recall"] == {"expected": 1, "hit": 1, "fraction": 1.0}
    assert result["source_recall"] == {"expected": 1, "hit": 1, "fraction": 1.0}
    assert result["evidence_defects"] == {"evidence:no_frontier_coverage": 1}
    assert result["complete"] is False


def test_a_broken_recovery_does_not_erase_what_the_reader_chose():
    """An evidence defect bears on what came BACK, never on what was CHOSEN.

    # fork: better-hermes-lcm — this used to pass the tool error only as a TRACE-level
    # evidence_defect. That let one error anywhere in a question excuse every line of it, so
    # the defect now has to be attributed to the line whose recovery it actually broke.
    """
    trace = _trace(recovered_text="",
                   evidence_defects=("evidence:tool_error",),
                   bounded_store_ids={11: ("evidence:tool_error",)})

    result = score_navigation([_case()], [trace])

    assert result["scored"] == 1
    assert result["node_recall"] == {"expected": 1, "hit": 1, "fraction": 1.0}
    assert result["source_recall"] == {"expected": 0, "hit": 0, "fraction": None}
    assert result["evidence_defects"] == {"evidence:tool_error": 1}
    assert result["complete"] is False


def test_an_error_on_one_recovery_does_not_excuse_an_unrelated_missing_line():
    """A trace-level defect cannot say WHICH line it broke, so it may not excuse any of them.

    Both sources sit under the node the reader opened. One recovery genuinely errored; the
    other line is simply absent, which is a real miss. Withdrawing both would hide the miss
    behind an unrelated error — the same per-line-versus-per-question confusion, a third time.
    """
    case = _case(
        expected_store_ids=(11, 12),
        covering_node_ids={11: (7,), 12: (7,)},
        evidence_snippets={11: "the line that is simply absent", 12: "the line that errored"},
    )
    trace = _trace(recovered_text="nothing useful came back",
                   evidence_defects=("evidence:tool_error",),
                   bounded_store_ids={12: ("evidence:tool_error",)})

    result = score_navigation([case], [trace])

    assert result["source_recall"] == {"expected": 1, "hit": 0, "fraction": 0.0}
    assert [w["store_id"] for w in result["source_recall_withdrawn"]["sample"]] == [12]
    assert result["per_case"][0]["reasons"].count("reader:incomplete_recovery") == 1


# fork: better-hermes-lcm — these three used to call attribute_recovery_defects(labels,
# store_ids): one flat label list plus one row list, i.e. one result has one failure with one
# scope. That model cannot represent a result carrying TWO failures with DIFFERENT scopes, and
# on a mixed failure it silently collapsed the unidentified one into the named rows. Causes
# carry their own scope now, and the split is mechanical.
#
# The change is not only to what they CALL. The third one's expected VALUE changed too: a
# corrupt payload with no named rows used to expect only
# ("evidence:incomplete_recovery_declared",) node-wide, and now also expects
# "evidence:source_missing" there, because the corruption is a cause in its own right rather
# than something the derived incompleteness stood in for.
#
# All three also gained has_more=False. Deriving an incompleteness from `complete: false`
# depends on whether paging is its only cause, and these three are about non-paging causes.

def test_a_tool_that_names_the_missing_rows_binds_only_those_rows():
    """`complete: false` is DOWNSTREAM of the named rows, so it must not bind the whole node.

    On the base tree the emitter sets `complete: false` *because* of missing_source_store_ids,
    so when those rows are the whole story the two labels are one event. The citation this
    docstring used to carry (tools.py:1746) is deliberately gone: a retrieval branch adds
    `has_more` to the same reason list, so the line number no longer establishes the claim and
    the suppression rule above is what makes the derivation safe.
    """
    causes = recovery_causes(tool_failed=False, identified_missing_rows=(42,),
                             unidentified_failure=False, declared_incomplete=True,
                             has_more=False)
    per_row, _per_node, node_wide = attribute_recovery_defects(causes)

    assert per_row == {42: ("evidence:incomplete_recovery_declared", "evidence:source_missing")}
    assert node_wide == ()


def test_a_whole_call_failure_still_binds_the_whole_node():
    """A tool error returned nothing at all, so it bounds every row that call was about."""
    causes = recovery_causes(tool_failed=True, identified_missing_rows=(42,),
                             unidentified_failure=False, declared_incomplete=False,
                             has_more=False)
    per_row, _per_node, node_wide = attribute_recovery_defects(causes)

    assert per_row == {42: ("evidence:source_missing",)}
    assert node_wide == ("evidence:tool_error",)


def test_an_incomplete_recovery_naming_no_rows_still_binds_the_node():
    """A corrupt payload declares incompleteness without naming a store_id; the node is all
    the attribution available, and losing it would charge the reader for the corruption."""
    causes = recovery_causes(tool_failed=False, identified_missing_rows=(),
                             unidentified_failure=True, declared_incomplete=True,
                             has_more=False)
    per_row, _per_node, node_wide = attribute_recovery_defects(causes)

    assert per_row == {}
    assert node_wide == ("evidence:incomplete_recovery_declared", "evidence:source_missing")


def test_an_ordinary_paged_page_raises_no_incompleteness_cause():
    """`beta/retrieval` sets `complete: false` for `has_more` alone (tools.py:1813-1826).

    Deriving an incompleteness cause from that binds a whole subtree on an ordinary page, and
    unlike the paging observation it is never cleared. The paging channel already self-clears
    when the reader follows the cursor, so paging must not raise a second, stickier signal.
    """
    causes = recovery_causes(tool_failed=False, identified_missing_rows=(),
                             unidentified_failure=False, declared_incomplete=True,
                             has_more=True)

    assert causes == ()


def test_an_incompleteness_with_a_real_cause_survives_a_paged_page():
    """Paging suppresses the derived signal only when paging is the whole story."""
    causes = recovery_causes(tool_failed=False, identified_missing_rows=(),
                             unidentified_failure=True, declared_incomplete=True,
                             has_more=True)

    assert RecoveryCause("evidence:incomplete_recovery_declared", None) in causes
    assert RecoveryCause("evidence:source_missing", None) in causes


def test_an_unread_root_binds_that_node_and_not_the_reader():
    """A synthesis block naming roots it never read identifies nodes, not rows."""
    per_row, per_node, node_wide = attribute_recovery_defects(
        (RecoveryCause("evidence:source_unread", node_ids=(12, 13)),))

    assert per_row == {}
    assert per_node == {12: ("evidence:source_unread",), 13: ("evidence:source_unread",)}
    assert node_wide == ()


def test_a_mixed_recovery_failure_keeps_both_attributions():
    """One result, two failures, two scopes — and neither may swallow the other.

    The tool named row 42 unreadable AND reported a corrupt payload whose owning row it did
    not name. Collapsing the corruption into row 42 loses it, and a line missing because ITS
    payload is corrupt then gets charged to the reader.
    """
    causes = recovery_causes(tool_failed=False, identified_missing_rows=(42,),
                             unidentified_failure=True, declared_incomplete=True,
                             has_more=False)
    per_row, _per_node, node_wide = attribute_recovery_defects(causes)

    assert per_row == {42: ("evidence:source_missing",)}
    assert node_wide == ("evidence:incomplete_recovery_declared", "evidence:source_missing")


def test_a_declared_incompleteness_is_row_scoped_only_when_the_named_rows_explain_it():
    """With an unidentified failure in the same result, "incomplete" is not about row 42 alone."""
    narrow = recovery_causes(tool_failed=False, identified_missing_rows=(42,),
                             unidentified_failure=False, declared_incomplete=True,
                             has_more=False)
    broad = recovery_causes(tool_failed=False, identified_missing_rows=(42,),
                            unidentified_failure=True, declared_incomplete=True,
                            has_more=False)

    assert RecoveryCause("evidence:incomplete_recovery_declared", (42,)) in narrow
    assert RecoveryCause("evidence:incomplete_recovery_declared", None) in broad


def test_a_node_id_the_reader_could_not_have_seen_is_not_credited_as_navigation():
    """Guessing small integers must not score: the holder leaves here are literally 1, 2, 4, 5."""
    trace = _trace(chosen_node_ids=(), unsourced_node_ids=(7,),
                   recovered_store_ids=(), recovered_text="")

    result = score_navigation([_case()], [trace])

    assert result["node_recall"] == {"expected": 1, "hit": 0, "fraction": 0.0}
    assert result["unsourced_node_ids"] == 1
    assert "reader:unsourced_node_id" in result["per_case"][0]["reasons"]


def test_a_missing_line_that_was_paged_out_is_an_evidence_defect_not_a_reader_miss():
    """lcm_expand returns one page; the rest of the node is behind a cursor (#52 territory).

    A labelled line that did not come back from a PAGED recovery may simply be on the next
    page. Booking that as "the reader navigated badly" would attribute another issue's bound
    to the model.

    # fork: better-hermes-lcm — this used to assert that the WHOLE CASE was withdrawn
    # (unscored_evidence_defect == 1, source_recall.expected == 0), keyed off a run-level
    # `observations` label. Adversarial review showed that is wrong and dangerous: on the model
    # path every leaf expansion is paged, so a case-wide withdrawal removes every question the
    # reader got wrong from BOTH denominators and walks the headline fractions toward 1.0 by
    # selection. Only the missing line is withdrawn now, and only from source recall; node
    # recall is about the choice and is counted either way.
    """
    trace = _trace(recovered_text="we rejected exponential backoff",
                   bounded_store_ids={11: ("evidence:paged_result",)})

    result = score_navigation([_case()], [trace])

    assert result["scored"] == 1
    assert result["node_recall"] == {"expected": 1, "hit": 1, "fraction": 1.0}
    assert result["source_recall"]["expected"] == 0
    assert result["source_recall_withdrawn"]["count"] == 1
    assert result["evidence_defects"] == {"evidence:bounded_recovery": 1}
    assert result["complete"] is False


def test_a_missing_line_from_a_recovery_carrying_a_cut_marker_is_also_an_evidence_defect():
    """An [LCM ...] marker in the recovered text says content was replaced (#50/#56).

    # fork: better-hermes-lcm — this used to read the marker off a run-level `observations`
    # label and assert a per-case `bounded_by`. A marker anywhere in the concatenated recovery
    # then voided every line of every node; the bound is attributed per node by the caller now.
    """
    trace = _trace(recovered_text="we rejected exponential backoff [LCM elided 900 of 2000 chars]",
                   bounded_store_ids={11: ("evidence:truncated_or_marked",)})

    result = score_navigation([_case()], [trace])

    assert result["evidence_defects"] == {"evidence:bounded_recovery": 1}
    assert result["per_case"][0]["source_recall_withdrawn"][0]["bounded_by"] == [
        "evidence:truncated_or_marked"
    ]


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


def test_a_hedge_in_another_sentence_does_not_credit_this_claim():
    """A published leaf is one multi-topic block; another topic's hedge is not this one's."""
    text = _text(text="The shard-7 replay outcome is unknown. migrate_ledger_v3 is fine.")

    result = score_fidelity([STARTED], [text])

    assert result["correctly_named_uncertainty"]["count"] == 0
    assert result["undetermined"]["count"] == 1
    assert result["complete"] is False


def test_a_pattern_that_does_not_compile_is_named_and_refused():
    """A swallowed re.error silently disables a detector and the run still reports clean."""
    with pytest.raises(CorpusPatternError) as excinfo:
        validate_patterns([
            ("state_claims[migration-started].forbidden_patterns[0]", "(unbalanced"),
        ])

    assert "migration-started" in str(excinfo.value)
    assert "(unbalanced" in str(excinfo.value)


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
    # fork: better-hermes-lcm — this pinned the unqualified
    # "evidence:distinguishing_field_absent". The label now carries the level, because at a leaf
    # it means the source ROWS lacked the text and at a condensation it means the child
    # SUMMARIES did — two different causes that were being reported as one.
    assert result["unscored_evidence_defect"]["labels"] == {
        "evidence:distinguishing_field_absent@leaf": 1
    }
    assert result["omitted_statements"]["count"] == 0
    assert result["complete"] is False


def test_an_absent_statement_is_labelled_by_the_level_whose_input_lacked_it():
    """A leaf whose SOURCE rows lacked it and a parent whose CHILD SUMMARIES lacked it are two
    different defects: the first points at ingest/projection, the second at the level below."""
    leaf = _text(text_id="node:7", kind="leaf", depth=0,
                 source_carries_claim={"migrate-started": False})
    parent = _text(text_id="node:9", kind="condensation", depth=1,
                   source_carries_claim={"migrate-started": False})

    result = score_fidelity([STARTED], [leaf, parent])

    assert result["unscored_evidence_defect"]["labels"] == {
        "evidence:distinguishing_field_absent@leaf": 1,
        "evidence:distinguishing_field_absent@condensation": 1,
    }


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
