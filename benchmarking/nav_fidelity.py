"""Two separate scorers for the index-navigation / statement-fidelity run.

``score_navigation`` (#2) answers one question: handed the frontier the engine really
delivered, did a reader choose a node that covers the labelled source, and did the right
original lines come back out of it? ``score_fidelity`` (#8) answers a different one over the
same run: do the published summaries and the reader's answer still say what the sources said —
"was started" must not become "succeeded".

They are two functions returning two results **on purpose**. A reader can navigate perfectly to
a summary that lies, and a faithful sentence can be unfindable; one combined rate would hide
either, so neither result carries the other's numbers.

What these scorers can establish, and what they cannot
------------------------------------------------------
They are deterministic detectors over hand-labelled inputs. On a labelled corpus they can say:
this exact false upgrade did or did not appear in this text; this labelled source did or did not
come back from the node the reader chose. They cannot establish that a text is faithful in
general — an unfamiliar paraphrase matches no pattern, which is why an unrecognised state is
reported as ``undetermined`` and never as ``faithful``, and why a run with any undetermined pair
reports ``complete: false``. They are not an LLM judge and are not a substitute for one; an
LLM judge alone would not be proof either.

Nothing here reports a zero for an absent measurement. A reader that returned nothing, a source
that was reachable from no delivered node, and a statement whose distinguishing text never
reached the summariser are each counted in their own bucket, named in ``incomplete_reasons``,
and excluded from the denominators — an empty result must never read as "there is nothing".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# States whose faithful rendering is an open outcome, so naming the uncertainty is the CORRECT
# behaviour rather than a hedge to be smoothed away.
UNCERTAIN_STATES = frozenset({"planned", "started", "claimed", "uncertain"})

# The false-claim detector runs per sentence, and a sentence that hedges is not a false claim:
# "migrate_x was started; whether it completed is unknown" names the uncertainty in the same
# breath as the word "completed". The cost is a deliberate false negative — a sentence that
# hedges AND falsely claims in one breath scores as uncertainty — which is the safer direction
# for a detector whose output is read as "this specific lie was found".
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# How far either side of a forbidden match a hedge still counts as hedging it.
_HEDGE_WINDOW = 120

_WHITESPACE = re.compile(r"\s+")

# Recovery shapes that BOUND what came back without being defects in themselves. lcm_expand
# returning one page and a cursor is its normal contract, and an "[LCM ...]" marker names what
# it replaced rather than hiding it. Either one explains a labelled line that did not come back,
# so a miss under one of them is withdrawn from the score instead of being charged to the
# reader — that bound belongs to another issue (#50/#51/#52), not to the model.
BOUNDED_OBSERVATIONS = ("evidence:paged_result", "evidence:truncated_or_marked")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(str(text or ""))]
    return [part for part in parts if part]


class CorpusPatternError(ValueError):
    """A corpus pattern does not compile. Named and refused, never silently inert."""


def validate_patterns(entries: Sequence[tuple[str, str]]) -> None:
    """Compile every corpus pattern up front, naming the first that does not.

    ``_search`` treats an uncompilable pattern as unmatched so one bad entry cannot crash a
    whole run mid-flight — which means a detector can be silently switched off by a typo and
    the run still reports ``false_claims: 0``. This is the signal that makes that impossible:
    the caller validates at load and refuses to start.
    """
    for where, pattern in entries:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise CorpusPatternError(
                f"{where} does not compile: {exc}. Pattern: {pattern!r}. Nothing is scored "
                f"with a detector that cannot run — fix the pattern or remove it."
            ) from exc


def _search(patterns: Sequence[str], text: str):
    """Return the first pattern's match object, or ``None``."""
    for pattern in patterns:
        try:
            found = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        except re.error:
            # A malformed pattern is a corpus defect, not a passing text: treat it as
            # unmatched here and let the caller's corpus validation report it.
            continue
        if found:
            return found
    return None


def _matches(patterns: Sequence[str], text: str) -> str:
    """Return the first pattern that matches, or ``""``."""
    found = _search(patterns, text)
    return found.re.pattern if found is not None else ""


def _fraction(hit: int, expected: int) -> float | None:
    # No evidence is NOT a perfect score and not a zero (coverage_doctor.py made the same
    # call for the same reason): an unmeasured recall is reported as unmeasured.
    if expected <= 0:
        return None
    return round(hit / expected, 3)


# ── navigation (#2) ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NavigationCase:
    """One labelled question and the sources that answer it.

    ``covering_node_ids`` maps a labelled ``store_id`` to the node ids that COUNT as having
    reached it — the nodes that hold the row directly, restricted to the ones reachable from
    the frontier the engine actually delivered. Naming every frontier ancestor instead would
    make the score vacuous the moment condensation collapses the frontier to one node: there
    would be exactly one node to "choose". An empty tuple means the row is reachable from
    nothing the reader was handed, which is an evidence defect (the assembly cap, a removed
    field, an unbound source) and never a reader failure.

    ``path_node_ids`` are the ancestors between the delivered frontier and those holders.
    Opening one is how a reader gets down to a leaf, so it costs no precision — only nodes on
    neither list count as a wasted expansion.
    """

    question_id: str
    question: str
    expected_store_ids: tuple[int, ...]
    covering_node_ids: Mapping[int, tuple[int, ...]]
    path_node_ids: tuple[int, ...] = ()
    raw_in_context_store_ids: tuple[int, ...] = ()
    evidence_snippets: Mapping[int, str] = field(default_factory=dict)
    must_mention: tuple[str, ...] = ()      # literal strings, matched case-insensitively
    must_not_claim: tuple[str, ...] = ()    # regexes, matched case-insensitively
    category: str = ""


@dataclass(frozen=True)
class ReaderTrace:
    """What the reader did, as executed: chosen nodes, recovered text, final answer.

    ``status`` is ``"ok"``, ``"empty"`` (the reader produced no content) or ``"error"`` (the
    route failed). Only ``"ok"`` is scored; the rest are counted as unscored and named.
    """

    question_id: str
    status: str
    detail: str = ""
    chosen_node_ids: tuple[int, ...] = ()
    recovered_store_ids: tuple[int, ...] = ()
    recovered_text: str = ""
    answer: str = ""
    evidence_defects: tuple[str, ...] = ()
    # Things that were true of the recovery but do not on their own void a score — a paged
    # result is the normal shape of lcm_expand, not a defect. Tallied for reporting only.
    observations: tuple[str, ...] = ()
    # store_id -> the bounds that applied to the node(s) it lives under, attributed per node by
    # the caller. A labelled line missing from a bounded recovery is withdrawn from source
    # recall; nothing else is.
    bounded_store_ids: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    # Node ids the reader asked for that it could not have seen in the frontier or in any
    # earlier tool result. Never credited as navigation.
    unsourced_node_ids: tuple[int, ...] = ()


def score_navigation(cases: Sequence[NavigationCase],
                     traces: Sequence[ReaderTrace]) -> dict[str, Any]:
    traces_by_id = {trace.question_id: trace for trace in traces}

    node_expected = node_hit = 0
    nodes_expanded = nodes_needed = 0
    source_expected = source_hit = 0
    topics_checked = topics_missed = 0
    claims_checked = claims_found = 0
    scored = reader_unavailable = evidence_unscored = not_applicable = 0
    answered_without_expansion = unsourced_cases = 0
    defect_counts: dict[str, int] = {}
    observation_counts: dict[str, int] = {}
    withdrawn_samples: list[dict[str, Any]] = []
    recovery_defects_seen: dict[str, int] = {}
    missed_samples: list[dict[str, Any]] = []
    claim_samples: list[dict[str, Any]] = []
    incomplete: list[str] = []
    per_case: list[dict[str, Any]] = []

    for case in cases:
        entry: dict[str, Any] = {
            "question_id": case.question_id,
            "category": case.category,
            "reasons": [],
        }
        trace = traces_by_id.get(case.question_id)

        if trace is None or trace.status != "ok":
            reader_unavailable += 1
            detail = (trace.detail if trace is not None else "no reader trace recorded")
            status = (trace.status if trace is not None else "missing")
            entry.update(status="unscored_reader_unavailable",
                         reader_status=status, detail=detail)
            entry["reasons"].append(f"reader:{status}")
            incomplete.append(
                f"{case.question_id}: reader produced no scorable reply ({status}: {detail})"
            )
            per_case.append(entry)
            continue

        raw = set(case.raw_in_context_store_ids)
        required = [sid for sid in case.expected_store_ids if sid not in raw]
        entry["raw_in_context_excluded"] = len(case.expected_store_ids) - len(required)

        if not required:
            not_applicable += 1
            entry.update(status="not_applicable_raw_in_context")
            incomplete.append(
                f"{case.question_id}: every labelled source was still verbatim in the "
                f"delivered context, so navigation was never exercised"
            )
            per_case.append(entry)
            continue

        # THE RULE, stated once: an evidence defect bears on what came BACK, never on what was
        # CHOSEN. Node recall is a fact about the reader's choice and survives a broken, paged
        # or truncated recovery. The only thing that withdraws a store_id from node recall is
        # its being reachable from nothing the reader was handed — then the choice was not
        # available to make.
        #
        # Withdrawing the whole CASE for either was a real defect, twice over: a case-wide
        # withdrawal removes the questions the reader got WRONG from both denominators and
        # walks the fractions toward 1.0 by selection. The bounded-recovery branch did it for
        # every paged expansion, which on the model path is all of them; this branch did it
        # wherever a preservation fix was still incomplete, which is the hardest questions of
        # the very candidate this harness exists to measure.
        recovery_defects = sorted(dict.fromkeys(trace.evidence_defects))
        chosen = set(trace.chosen_node_ids)
        recovered = _normalize(trace.recovered_text).lower()

        case_node_expected = case_node_hit = 0
        case_source_hit = case_source_expected = 0
        case_withdrawn: list[dict[str, Any]] = []
        uncovered: list[int] = []
        for store_id in required:
            covering = case.covering_node_ids.get(store_id) or ()
            if not covering:
                # Reachable from nothing delivered (#5, #31/#35/#37/#56/#63/#67): the reader
                # could not have chosen it and could not have recovered it. Out of BOTH
                # denominators — this line only, not its question.
                uncovered.append(store_id)
                continue
            case_node_expected += 1
            reached = bool(chosen.intersection(covering))
            if reached:
                case_node_hit += 1
            snippet = _normalize(case.evidence_snippets.get(store_id, "")).lower()
            if not snippet:
                continue
            if snippet in recovered:
                case_source_expected += 1
                case_source_hit += 1
                continue
            # A line that did not come back out of a BOUNDED or BROKEN recovery may be on the
            # next page, inside what an "[LCM …]" marker names, or behind a tool error. That
            # belongs to another issue (#50/#51/#52), so the LINE leaves source recall — only
            # when the reader actually opened a node covering it, and never node recall.
            #
            # ONLY per-line attribution counts. A trace-level defect cannot say which line it
            # broke, so it may not excuse any of them: folding one in here let a single tool
            # error anywhere in a question withdraw every line of it, hiding real misses
            # behind an unrelated failure. The caller attributes each defect to the rows of
            # the node whose recovery it actually broke.
            bounds = list(dict.fromkeys(trace.bounded_store_ids.get(store_id) or ()))
            if reached and bounds:
                case_withdrawn.append({"question_id": case.question_id,
                                       "store_id": store_id,
                                       "bounded_by": sorted(dict.fromkeys(bounds))})
                continue
            case_source_expected += 1

        if uncovered:
            label = "evidence:no_frontier_coverage"
            defect_counts[label] = defect_counts.get(label, 0) + len(uncovered)
            entry["uncovered_store_ids"] = uncovered
            entry["reasons"].append(label)
            incomplete.append(
                f"{case.question_id}: {len(uncovered)} labelled source(s) are reachable from "
                f"no delivered node ({label}) — withdrawn from both recalls, not scored as a "
                f"model failure"
            )
        if recovery_defects:
            # Reported and made incomplete, but NOT counted into evidence_defects: what it
            # actually withdrew is counted per line above, and counting it here too would
            # report one event under two names.
            entry["recovery_defects_seen"] = recovery_defects
            for label in recovery_defects:
                recovery_defects_seen[label] = recovery_defects_seen.get(label, 0) + 1
            incomplete.append(
                f"{case.question_id}: the recovery reported "
                f"{', '.join(recovery_defects)} on at least one call; only the lines it could "
                f"be attributed to were withdrawn"
            )

        if case_node_expected == 0:
            # Nothing this question asked for was reachable at all: there is no choice left to
            # score, so the question is withdrawn rather than recorded as a zero.
            evidence_unscored += 1
            entry.update(status="unscored_evidence_defect",
                         evidence_defects=sorted(dict.fromkeys(
                             recovery_defects + ["evidence:no_frontier_coverage"] * bool(uncovered)
                         )))
            per_case.append(entry)
            continue

        scored += 1
        if trace.answer.strip() and not trace.chosen_node_ids:
            answered_without_expansion += 1
        node_expected += case_node_expected
        node_hit += case_node_hit
        source_expected += case_source_expected
        source_hit += case_source_hit
        if case_withdrawn:
            # Each withdrawn line is counted once, under the cause that withdrew it: the
            # "bounded_recovery" umbrella for a page boundary or a cut marker, and the defect's
            # own name for anything else. One event, one name.
            for item in case_withdrawn:
                label = ("evidence:bounded_recovery"
                         if any(name in BOUNDED_OBSERVATIONS for name in item["bounded_by"])
                         else item["bounded_by"][0])
                defect_counts[label] = defect_counts.get(label, 0) + 1
            withdrawn_samples.extend(case_withdrawn)
            entry["source_recall_withdrawn"] = case_withdrawn
            entry["reasons"].append(label)
            incomplete.append(
                f"{case.question_id}: {len(case_withdrawn)} labelled line(s) did not come back "
                f"from a bounded recovery "
                f"({', '.join(sorted({b for w in case_withdrawn for b in w['bounded_by']}))}) "
                f"— withdrawn from source recall, not scored as a model failure. Node recall "
                f"for this question is unaffected."
            )
        if trace.unsourced_node_ids:
            # A node id the reader could not have seen is a guess, not navigation: at the low
            # anchor the holder leaves are literally 1, 2, 4, 5. The caller keeps these out of
            # chosen_node_ids so they cannot credit recall; this is where they are named.
            unsourced_cases += 1
            entry["unsourced_node_ids"] = list(trace.unsourced_node_ids)
            entry["reasons"].append("reader:unsourced_node_id")
        for label in dict.fromkeys(trace.observations):
            observation_counts[label] = observation_counts.get(label, 0) + 1
        # Recall alone rewards a reader that expands the whole DAG; precision is what makes
        # "it opened everything" visible next to a perfect recall.
        nodes_expanded += len(chosen)
        needed_nodes = {node for store_id in required
                        for node in case.covering_node_ids.get(store_id, ())}
        needed_nodes.update(case.path_node_ids)
        nodes_needed += len(chosen & needed_nodes)

        if not trace.chosen_node_ids:
            entry["reasons"].append("reader:no_expansion")
        elif case_node_hit < case_node_expected:
            entry["reasons"].append("reader:wrong_node")
        elif case_source_hit < case_source_expected:
            entry["reasons"].append("reader:incomplete_recovery")

        answer = trace.answer
        for topic in case.must_mention:
            topics_checked += 1
            if topic.lower() not in answer.lower():
                topics_missed += 1
                missed_samples.append({"question_id": case.question_id, "topic": topic})
        for pattern in case.must_not_claim:
            claims_checked += 1
            matched = _matches([pattern], answer)
            if matched:
                claims_found += 1
                claim_samples.append({
                    "question_id": case.question_id,
                    "pattern": pattern,
                    "excerpt": _normalize(answer)[:200],
                })

        entry.update(
            status="scored",
            node_recall={"expected": case_node_expected, "hit": case_node_hit},
            source_recall={"expected": case_source_expected, "hit": case_source_hit},
            chosen_node_ids=list(trace.chosen_node_ids),
            recovered_store_ids=list(trace.recovered_store_ids),
        )
        per_case.append(entry)

    if node_expected == 0:
        incomplete.append(
            "no question in this run required navigation, so the navigation recalls are "
            "unmeasured rather than perfect"
        )

    return {
        "cases_total": len(cases),
        "scored": scored,
        "unscored_reader_unavailable": reader_unavailable,
        "unscored_evidence_defect": evidence_unscored,
        "not_applicable_raw_in_context": not_applicable,
        "answered_without_expansion": answered_without_expansion,
        "node_recall": {"expected": node_expected, "hit": node_hit,
                        "fraction": _fraction(node_hit, node_expected)},
        "node_precision": {"expanded": nodes_expanded, "needed": nodes_needed,
                           "fraction": _fraction(nodes_needed, nodes_expanded)},
        "source_recall": {"expected": source_expected, "hit": source_hit,
                          "fraction": _fraction(source_hit, source_expected)},
        "source_recall_withdrawn": {"count": len(withdrawn_samples),
                                    "sample": withdrawn_samples[:12]},
        "unsourced_node_ids": unsourced_cases,
        "missed_topics": {"checked": topics_checked, "missed": topics_missed,
                          "sample": missed_samples[:12]},
        "false_assertions": {"checked": claims_checked, "found": claims_found,
                             "sample": claim_samples[:12]},
        "evidence_defects": defect_counts,
        # Reported separately from evidence_defects: a defect SEEN on some call is not
        # the same claim as a line withdrawn because of it, and only the second is a
        # statement about a specific labelled source.
        "recovery_defects_seen": recovery_defects_seen,
        "observations": observation_counts,
        "complete": not incomplete,
        "incomplete_reasons": incomplete,
        "per_case": per_case,
    }


# ── fidelity (#8) ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StateClaim:
    """One labelled statement about one entity, in one state, at one point in the corpus.

    The same entity appears under several claim ids in several states — planned, started,
    failed, claimed, externally confirmed, negated, corrected — because the failure this scores
    is precisely a summary collapsing those into one another.
    """

    claim_id: str
    entity: str
    true_state: str
    source_store_ids: tuple[int, ...] = ()
    truth_patterns: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    hedge_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredText:
    """A published summary (leaf or condensation) or a reader answer, with its claims.

    ``source_carries_claim[claim_id]`` says whether the statement was present in **this text's
    own input** — the source rows for a leaf, the child summaries for a condensation, the
    recovered evidence for an answer. When it is false the distinguishing text never reached
    the writer, so the writer did not lose it: that is an evidence defect belonging to another
    issue, and it is excluded from the fidelity counts instead of being booked as a model
    failure.
    """

    text_id: str
    kind: str                     # "leaf" | "condensation" | "answer"
    depth: int | None
    text: str
    claim_ids: tuple[str, ...] = ()
    source_carries_claim: Mapping[str, bool] = field(default_factory=dict)
    entity_coverage: Mapping[str, Any] | None = None


_OUTCOMES = (
    "false_claims",
    "omitted_statements",
    "correctly_named_uncertainty",
    "faithful",
    "undetermined",
    "unscored_evidence_defect",
)


def _empty_bucket() -> dict[str, int]:
    return {name: 0 for name in _OUTCOMES}


def _local_hedge(claim: StateClaim, text: str) -> str:
    """A hedge in the same sentence as this claim's anchor, and near it. Else ``""``.

    The anchor is a truth-pattern match if there is one, otherwise the entity name. Requiring
    both the sentence and the ``_HEDGE_WINDOW`` proximity mirrors the false-claim rule, so the
    two directions are symmetric rather than one being generous and the other strict.
    """
    for sentence in _sentences(text):
        anchor = _search(claim.truth_patterns, sentence)
        if anchor is not None:
            start, end = anchor.start(), anchor.end()
        elif claim.entity and claim.entity.lower() in sentence.lower():
            start = sentence.lower().index(claim.entity.lower())
            end = start + len(claim.entity)
        else:
            continue
        window = sentence[max(0, start - _HEDGE_WINDOW):end + _HEDGE_WINDOW]
        hedge = _matches(claim.hedge_patterns, window)
        if hedge:
            return hedge
    return ""


def _classify(claim: StateClaim, text: str) -> tuple[str, str]:
    """Return ``(outcome, evidence)`` for one claim against one text."""
    for sentence in _sentences(text):
        found = _search(claim.forbidden_patterns, sentence)
        if found is None:
            continue
        # Hedge suppression is LOCAL. A published summary is often one unbroken block, and
        # over a whole block a single stray "unknown" four hundred characters away would
        # excuse a lie at the other end.
        window = sentence[max(0, found.start() - _HEDGE_WINDOW):found.end() + _HEDGE_WINDOW]
        if not _matches(claim.hedge_patterns, window):
            # Report the span that matched, not the head of the block it sits in: a sample a
            # reader cannot check is not evidence.
            return "false_claims", found.group(0)[:180]
    if claim.true_state in UNCERTAIN_STATES:
        # LOCAL, exactly like the false-claim test above. A published leaf is one multi-topic
        # block, so a hedge belonging to another topic elsewhere in it would credit this claim
        # with naming an uncertainty it never named — moving pairs out of the honest
        # "undetermined" bucket into #8's positive one.
        hedged = _local_hedge(claim, text)
        if hedged:
            return "correctly_named_uncertainty", hedged
    truth = _matches(claim.truth_patterns, text)
    if truth:
        return "faithful", truth
    if claim.entity and claim.entity.lower() in text.lower():
        # The entity survived but its state did not resolve to anything the corpus labelled.
        # Reporting that as faithful is how an entity-counting score certifies a lie
        # (coverage_doctor scored 2/2 on "started" rewritten to "succeeded").
        return "undetermined", _normalize(text)[:200]
    return "omitted_statements", ""


def score_fidelity(claims: Sequence[StateClaim],
                   texts: Sequence[ScoredText]) -> dict[str, Any]:
    claims_by_id = {claim.claim_id: claim for claim in claims}

    totals = _empty_bucket()
    by_kind: dict[str, dict[str, int]] = {}
    by_depth: dict[str, dict[str, int]] = {}
    samples: dict[str, list[dict[str, Any]]] = {name: [] for name in _OUTCOMES}
    defect_labels: dict[str, int] = {}
    baseline: list[dict[str, Any]] = []
    incomplete: list[str] = []
    per_pair: list[dict[str, Any]] = []
    pairs = 0

    for text in texts:
        if text.entity_coverage is not None:
            baseline.append({
                "text_id": text.text_id,
                "kind": text.kind,
                "depth": text.depth,
                "coverage": dict(text.entity_coverage),
            })
        kind_bucket = by_kind.setdefault(text.kind, _empty_bucket())
        depth_bucket = by_depth.setdefault(str(text.depth), _empty_bucket())

        for claim_id in text.claim_ids:
            claim = claims_by_id.get(claim_id)
            if claim is None:
                continue
            pairs += 1
            if not text.source_carries_claim.get(claim_id, True):
                outcome, evidence = "unscored_evidence_defect", ""
                # The level matters and was being lost. At a LEAF this means the source rows
                # themselves lacked the distinguishing text — ingest or the summariser
                # projection (#31/#35/#37/#56/#63/#67). At a CONDENSATION it means the child
                # summaries lacked it, so the level below dropped it and is already counted
                # there as an omission; the parent is not to blame for either.
                label = f"evidence:distinguishing_field_absent@{text.kind}"
                defect_labels[label] = defect_labels.get(label, 0) + 1
                incomplete.append(
                    f"{text.text_id}/{claim_id}: the statement was absent from this text's own "
                    f"{'source rows' if text.kind == 'leaf' else 'inputs'} ({label}) — not "
                    f"scored as a model failure"
                )
            else:
                outcome, evidence = _classify(claim, text.text)
                if outcome == "omitted_statements":
                    incomplete.append(
                        f"{text.text_id}/{claim_id}: the statement does not appear, so its "
                        f"rendering could not be judged"
                    )
                elif outcome == "undetermined":
                    incomplete.append(
                        f"{text.text_id}/{claim_id}: '{claim.entity}' appears but its state "
                        f"matched no labelled rendering — unresolved, not faithful"
                    )

            totals[outcome] += 1
            kind_bucket[outcome] += 1
            depth_bucket[outcome] += 1
            record = {
                "text_id": text.text_id,
                "kind": text.kind,
                "depth": text.depth,
                "claim_id": claim_id,
                "entity": claim.entity,
                "true_state": claim.true_state,
                "outcome": outcome,
                "evidence": evidence,
            }
            per_pair.append(record)
            if len(samples[outcome]) < 12:
                samples[outcome].append(record)

    def block(name: str) -> dict[str, Any]:
        return {"count": totals[name], "sample": samples[name]}

    result: dict[str, Any] = {
        "claims_total": len(claims),
        "texts_total": len(texts),
        "pairs_total": pairs,
        "false_claims": block("false_claims"),
        "omitted_statements": block("omitted_statements"),
        "correctly_named_uncertainty": block("correctly_named_uncertainty"),
        "faithful": totals["faithful"],
        "undetermined": block("undetermined"),
        "unscored_evidence_defect": {
            "count": totals["unscored_evidence_defect"],
            "labels": defect_labels,
            "sample": samples["unscored_evidence_defect"],
        },
        "by_kind": by_kind,
        "by_depth": by_depth,
        # coverage_doctor's entity fraction travels BESIDE this, as the contrast it is: the
        # same identifier in the wrong state scores full entity coverage.
        "entity_coverage_baseline": baseline,
        "detection_rule": (
            "per-sentence regex detection; a sentence that hedges is not counted as a false "
            "claim; an unrecognised state is 'undetermined', never 'faithful'"
        ),
        "complete": not incomplete,
        "incomplete_reasons": incomplete,
        "per_pair": per_pair,
    }
    return result
