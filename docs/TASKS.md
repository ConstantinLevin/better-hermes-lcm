# betterlcm — open tasks (proposed; nothing here is started without a go)

Status legend: `proposed` · `agreed` · `in progress` · `done (commit)` · `rejected (why)`.
Each task carries an acceptance criterion; a task is not done until that holds and the full
suite plus the fork tests are green.

## Fixed in this pass (verified by probe, tests added; see git log)

These were all confirmed against the code before changing anything. The first five are defects
the fork itself introduced.

| # | defect | fix |
|---|---|---|
| F1 | **The curve was not monotone and jumped in meaning just above 256k.** `fresh_tail_max_tokens` interpolated out of a `0` *sentinel* ("no cap"), so at W=262,154 the cap was **1 token** and the protected tail collapsed from 32 messages to 1 (6 at 272k, 28 at 300k). `condense_budget_tokens` did the same. `leaf_chunk_tokens` resolved both fraction endpoints against the *current* window, peaking at 336k tokens/call around 600k — larger requests in the middle than at either end. | Fraction endpoints resolve against their own anchor window (fixed endpoints ⇒ linear, monotone); the tail cap's low endpoint is a cap that cannot bind and is reported as upstream's `0` while it cannot; `test_curve_is_monotone_across_the_whole_range` and a per-anchor linearity test now pin the shape, not just the endpoints (audit p11 showed a quadratic passed all 13 old tests). |
| F2 | **The condensation budget replaced upstream's loop instead of gating it**, so with the tiny budget the curve produced just above 256k it condensed the entire frontier on every compaction. | The budget is now the conjunction the design documented: upstream's loop runs, gated by `frontier > budget`; oldest-first selection only while the gate is active. |
| F3 | **`lcm_status` reported a threshold the engine was not using.** On a clean install `LCMConfig.from_env()` records source `"default"`, which the mixin's guard rejected while the resolver accepted it: the engine compacted at **350,000** tokens on a 1M model while status said 800,000. | One authority: the resolver's `curve@` source decides, plus an autoraise guard. |
| F4 | **`/new` produced a dead end.** Carried-over nodes kept their children in the old session, and every traversal required session equality, so a retained parent expanded to "no children, has_more=false". | Authorization is reachability from the current session (`_is_reachable_from_current_session`), not session equality; unrelated sessions stay refused; a recorded child that no longer exists is reported (`missing_source_node_ids`, `incomplete`) instead of skipped. |
| F5 | **Rotate advanced the frontier before writing its marker**, so a failed marker write left raw rows skipped with nothing pointing at them — and the retry was a no-op. | Marker first; if it cannot be written and the span needs one, the frontier does not move and the caller is told (`marker_write_failed`). |
| F6 | **The focus-topic prompt contradicted the coverage contract** — "spend 60-70% of the budget on it", "reduce resolved topics to one-liners **or drop**" — and a focus topic is auto-derived on nearly every compaction. | Focus sets emphasis and order only; a resolved or off-focus topic may be one line but must still say what it was and how it ended. |
| F7 | **Tool arguments arriving as a dict were counted by key count** (`len(dict)//4+1`): a 50,000-character call cost **11 tokens**, corrupting every pressure, tail, chunk and assembly decision. | Non-string values are serialised before counting. |
| F8 | **The coverage doctor certified on no evidence** (a node pointing at a missing child scored 1.0/pass). | Three outcomes: scored, unscored, structurally broken; unreadable sources fail; a truncated scan can never report a clean bill. |
| F9 | **The sidecar index block was cut at 1,600 chars**, mid-topic, and surfaced in that state. | Stored whole. |
| F10 | **A failed search was returned as "no matches"** — false negative evidence over retained data. | `complete: false`, `search_failures`, and a note that absence from the results is not absence from history. |
| F11 | **The LIKE fallback limited before ordering** (CJK/emoji queries are routed there by design): `sort="recency", limit=1` returned the 50th of 100. | `ORDER BY` in SQL before the limit. |
| F12 | **The failure cooldown outlived its session**, blocking an unrelated one after `/new`. | Scoped to the session that failed; cleared on reset. |
| F14 | **A failed condensation discarded committed leaf work and could publish an empty-provenance node.** Leaf passes commit and advance the raw cursor before condensation runs; letting a condensation failure escape returned the ORIGINAL uncompacted prompt while the DAG had moved on, and the retry mapped no sources and published a leaf with no provenance, resetting the cursor. Upstream never reached this state because L3 always converged (audit D #1). | Condensation failure publishes the leaf progress and arms the cooldown; a leaf whose source mapping is empty is refused, not published. |
| F13 | **The leaf rescue's last resort was `chunk[:-1]`**, stripping a result from its call. | Cuts on a tool-group boundary; gives up rather than splitting an indivisible group. |

## Still open — from the partitioned audits

Fourteen audits ran: four aspect comparisons against lossless-claw, three cross-cutting
(whole-plugin, plan critique, unplanned claw capabilities), a regression hunt against upstream,
and eleven exhaustive partitions covering all 70 modules, the non-code assets and the test
suite. Their reports are in `docs/claw-comparison/`. The findings below are NOT yet fixed; each
report carries the citations and the reproduction.

- **p01 engine-core** — publication and session ownership not coupled; late work can publish
  under a different session; indexes that describe only the ends of their sources.
- **p02 agent-tools** — tools that mistake incomplete work for complete results (eleven items
  reported as a complete ten; an orphan check passing after 1,000 nodes).
- **p03 operator-config** — `/lcm clean` can delete sources a retained node still references;
  its transaction can lose rollback protection under concurrent use.
- **p04 storage-schema** — the raw store is a projection, not an archive (host fields dropped);
  reconciliation can discard new occurrences and attach wrong provenance.
- **p05 compaction-flow** — a failed condensation can leave a published leaf behind and a retry
  can publish with empty provenance; the envelope fitter still removes middle content.
- **p06 payloads-embeddings** — Hermes' current spillover directory is not recognised; the
  always-on ingest guard can canonicalise surrounding JSON; embedding workers can publish
  offsets against replaced text.
- **p07 rollups-assertions** — adaptive retrieval can certify incomplete evidence; the assertion
  sidecar can publish unsupported state and present it as current.
- **p08/p09 trajectory, evidence** — "grounded"/"verified"/"sufficient" do not reliably mean the
  evidence supports the answer; limits can change the answer or erase its recovery handles.
- **p11 tests** — rows are tested far more thoroughly than usable provenance; the suite does not
  establish that a stored summary is complete or functions as an index.
- **audit B (plan critique)** — one publication contract (validated state transition) subsumes
  several planned tasks; the prompt rewrite must come *after* source completeness and the
  generation contract; the host fabricates `finish_reason`, so B1 needs a host contract.
- **audit C** — the message envelope and a durable ingest receipt keyed by host event identity.
- **audit E (unfiltered claw sweep)** — 37 items, none previously planned. It states that the
  "does it change a guarantee" filter applied to the first claw comparison was an invalid
  reason to reject candidates. Nine are cheap and valuable (test-home isolation, prompt-prefix
  divergence diagnostics, release-commit validation, payload-reference disambiguators, a
  script-aware token estimator, prompt inspection commands, copied-reference parsing, shadow-
  install/drift detection, release fragments); six are capability-level decisions
  (`context_items` projection, operator TUI, persistent focus briefs, delegated retrieval
  workers, richer maintenance debt, a paged expansion-cost manifest).
- **audit D (regressions vs upstream)** — landed. Three regressions the fork introduced: the
  condensation-failure defect above (now fixed), the lookahead losing the host's routing
  context / deadline / cancellation scope with an unbounded wait that can hold the compaction
  lock, and the 1M leaf production rate (up to 64 per call) outrunning the retained
  one-group-per-depth condensation schedule. Plus two medium: 256k structural equivalence still
  fails on oversized/imported histories and on the dynamic-chunk path, and whole sidecar index
  blocks now escape retrieval budgets.

Tracked as tasks #1-#12 in this session's task list; the ~375 ranked findings across the nine
partition reports are not yet individually triaged.

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
