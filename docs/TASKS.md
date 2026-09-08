# betterlcm — open tasks (proposed; nothing here is started without a go)

Status legend: `proposed` · `agreed` · `in progress` · `done (commit)` · `rejected (why)`.
Each task carries an acceptance criterion; a task is not done until that holds and the full
suite plus the fork tests are green.

## Phase 0 — the summariser prompt, designed this time
| id | task | acceptance | status |
|---|---|---|---|
| P0.1 | Write `docs/prompt-design.md`: what a reader needs from a summary to choose what to expand; failure modes seen (dropped topics, focus skew, cut-off endings, process-detail loss at depth); one coherent instruction per depth (leaf / merge of children / deep merge); how length, chronology, focus and custom instructions interact; what stays in the untrusted-data wrapper | you have read and agreed the design | proposed |
| P0.2 | Replace the five-layer patchwork in `escalation.py` with the designed prompt (single fork module `summary_prompts.py`; upstream tests that pin old substrings re-pointed and listed in touchpoints) | prompt text == design; no contradiction remains (no "or drop", no 60–70 % skew) | proposed |
| P0.3 | Measurement: drill-down evaluation (real model, rendered prefix → does it pick the right node), plus coverage doctor with a "no evidence" state | numbers on a fixture set at 256k and 1M before/after P0.2 | proposed |

## Phase 1 — the index must never lie (correctness)
| id | task | acceptance | status |
|---|---|---|---|
| T1.1 | A1: carried-over nodes expand through their children across the session boundary (root authorised, descendants followed by edge; archive scope; cross-session payload hydration) | connected d2→d1→d0 DAG, retain 2/-1, repeated `/new`, restart: every level expands; unrelated sessions still refused | proposed |
| T1.2 | B1: provider generation limit — large window-weighted `max_tokens`; `finish_reason` `length`/`incomplete` ⇒ re-ask with more room, never stored; exhausted ⇒ `SummaryUnavailableError` | injected `finish_reason="length"` never yields a node; healthy responses unchanged | proposed |
| T1.3 | B2 + A2: atomic publication — node + sidecar + frontier in one transaction; rotate writes marker and frontier together or nothing | fault injection after every write step leaves a consistent DB | proposed |
| T1.4 | B9: search failure ≠ empty result — partial/error outcome reported to the agent | store error ⇒ structured failure, hits preserved | proposed |
| T1.5 | A4 + B18: structural provenance audit separate from semantic coverage; zero evidence never passes; orphan check covers node-sourced nodes, unbounded | node with missing child ⇒ fail | proposed |

## Phase 2 — small contract fixes
| id | task | acceptance | status |
|---|---|---|---|
| T2.1 | B10 LIKE fallback orders before limiting | newest match returned for `sort=recency` | proposed |
| T2.2 | B11 `describe(node_id)` returns own summary + index block, paged | known node readable without expansion | proposed |
| T2.3 | B3 sanitiser: keep inline literal tags, strip trailing standalone reasoning | claw's fixtures pass | proposed |
| T2.4 | A5 + B22 docs/skill/defaults synchronised from `ENV_FIELD_SPECS` + anchors; operator guide's 1M advice replaced | checker passes; no "0.35 at 1M" text | proposed |
| T2.5 | A6 CI on `betterlcm`; host-integration lane fails (not skips) on host import failure | workflow runs on the fork branch | proposed |
| T2.6 | B21 transcript GC: exact-original equality before any rewrite | sanitised-only match ⇒ no rewrite | proposed |

## Phase 3 — engine/summariser quality
| id | task | acceptance | status |
|---|---|---|---|
| T3.1 | B4 timestamps + child provenance in summariser input and headers (observed vs ingested time kept distinct) | superseded-decision fixture summarised chronologically | proposed |
| T3.2 | B5 condensation reachable without a leaf pass under summary-side pressure | interrupted-sweep fixture drains via `compress()` | proposed |
| T3.3 | B6 one deadline through serial/lookahead/fallbacks/L2/condensation; bounded future waits | never-resolving future ⇒ partial publish + cooldown within budget | proposed |
| T3.4 | B7 frontier query in SQL (no 100k truncation) | >100k-node fixture selects real frontier | proposed |
| T3.5 | B8 condensation input bounded by tokens (window-weighted), oldest-first kept | 4×12k children never one 48k request | proposed |
| T3.6 | B14 escaped historical-data boundary around rendered summaries + policy line | fake headers/closers inert; replay recognition intact | proposed |

## Phase 4 — I/O and operability at scale
| id | task | status |
|---|---|---|
| T4.1 | B12 page-bounded expansion I/O | proposed |
| T4.2 | B16 payload catalog instead of 240-char head | proposed |
| T4.3 | B19 FTS optional at bootstrap | proposed |
| T4.4 | B20 read-only operator entry point without engine startup | proposed |
| T4.5 | B23 running-total prefix selection (only with a cap) | proposed |

## Phase 5 — the large ones
| id | task | status |
|---|---|---|
| T5.1 | B13 durable prepare-then-publish pipeline (biggest 1M win) | proposed |
| T5.2 | B17 repair of degraded historical summaries from sources | proposed |
| T5.3 | B15 evaluation harness maintained as the standing quality gate (feeds P0.3) | proposed |

## Process rules (so this list stays honest)
- Every task gets an adversarial check ("how can the index still lie after this?"), not a plan-conformance check.
- No task is closed with a full-suite run that was started before its last edit.
- Prompt changes never land without P0.3 numbers.
