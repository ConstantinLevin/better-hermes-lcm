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

## Second pass — audit findings closed (commits `cfcbb9f`, `1b21bc8`, `07c92be`, `6ee0eea`, `2ce8c7e`)

Each row names the audit id, so the report in `docs/claw-comparison/` carries the citation and
the reproduction. Every row has a fork test that fails without the change.

| id | defect | fix |
|---|---|---|
| p05 AX01 | A restricted TOOLSET was treated as an auxiliary identity: any agent whose toolsets were a subset of `{memory, skills}` was classified auxiliary and its conversation was **never stored**. | Toolset shape is corroboration only; an explicit child marker (subagent prefix, parent/subagent attribute, delegate depth) is required. |
| p05 CP01 | Replies to ignored host messages were consumed by a leaf but left out of its `source_ids`, so the rows were swept past the frontier and reachable from no node. | Every consumed row is a source of the leaf; the excluded ones are named in a marker and stay out of the summary text. |
| p05 CP04 | The no-leaf branch returned before condensation, so a session with an oversized summary frontier and no eligible raw backlog could never shrink. | Summary pressure reaches condensation (and the sweep's condensation) with no leaf pass. |
| p05 CP05 | The advertised loop deadline was not end-to-end: escalation reused one timeout per route per level. | `deadline` is an absolute instant threaded into escalation; every route recomputes its timeout from the time left. The leaf loop passes its clock in every mode. |
| p05 CP06 | Full-sweep and dynamic chunking used the unaligned selector, splitting a tool call from its results. | Every mode selects through the aligned selector. |
| p05 CP07 | "Cleanup only" started lookahead, extraction and assertion scheduling before checking the restriction. | The cleanup-only exit is taken before any model work is scheduled. |
| p05 CP08 | Pressure started from the host's observed prompt size but was decremented by the shortened working copy (2,505 tokens of tool output against a 15-token stub). | The subtraction uses the pressure view of the same span. |
| p05 CP02/CP03 | Node insert, sidecar write and frontier marker were separate commits; a failure between them left a summary without its level/index block, or rows the DAG had summarised looking raw and indexed again. | `add_node_with_meta` publishes node + sidecar in one transaction; binding takes the frontier as `max(marker, highest store_id the session's leaves summarise)`. |
| p05 ES01 | A summary that stopped at the generation limit was accepted and stored. | An explicit truncation `finish_reason` is a route failure. `"stop"` is still not taken as proof (some host paths fabricate it). |
| p05 ES03 | Route errors became `None`, so leaf rescue could not tell a too-large chunk from a dead route. | The cause travels with `SummaryUnavailableError`; the retry predicate walks the cause chain. |
| p05 ES05 | The focus was cut at 160 chars with a bare ellipsis. | The cut says how much of it is shown. |
| p05 ES06 | Acceptance tested only "smaller than the source", so `"OK"` was a valid summary. | Bare acknowledgements and refusals are rejected as route failures (deliberately narrow: terse is still a summary). |
| p05 PB01 | The prompt fitter compared the JSON-escaped envelope against the source's own token count and replaced the middle of the source with a marker. | Nothing is cut; a chunk too large for the route fails there and the leaf-rescue path retries smaller. |
| p05 EX01 | Removing injected memory blocks consumed real text between two blocks. | The removal boundary is unchanged (safety), but the cut is marked in the summariser's input. |
| p05 EX02 | The data-URI pattern included whitespace and letters, so it swallowed the prose after a URI. | The payload class stops at the first non-base64 character. |
| p05 EX03 | Selecting a block's `text` dropped `is_error`/identity, and several attachments collapsed into one flag. | Outcome siblings are appended; attachments are counted. |
| p05 EX04 | Sanitising dictionary KEYS let two keys collapse into one and dropped a value. | A sanitised key never displaces another; on collision the original key is kept. |
| p05 EX05 | Extraction interpolated historical source into one user message after `CONTENT:`. | Extraction uses the same untrusted-data envelope as summarisation. |
| p05 EX06/EX08 | A provider failure and a generation-limited answer both returned `None`, read by the caller as "nothing to extract". | `ExtractionUnavailableError`; the caller distinguishes and logs. |
| p05 EX07 | Extraction notes carried a wall-clock header and nothing tying them to their segment. | Each note names its source store ids and a content digest. |
| p05 SA01 | Assistant turns dropped for holding only internal content vanished from the agent's own view. | Named in the assembly omission marker. |
| p05 BY01 | The deterministic bypass trim could delete its own omission marker, never said how much went, and dropped the per-message cut marker at a zero budget. | The receipt names messages and characters, is never removed or shortened, and the cut marker survives a zero budget. |
| p05 BY02 | An unchanged native return was counted as a compression and its abort flag cleared. | Only a changed result counts; an overridden abort is logged. The assembly cap still holds. |
| p05 BY03 | One unsupported constructor keyword sent bypassed summarisation to a different model/route. | Only unsupported kwargs are dropped, and the drop is logged. |
| p05 SQ01 | Boolean words and punctuation were removed from the query silently. | `lcm_grep` reports `query_interpretation` when tokens were dropped. |
| p05 SQ02 | Supplementary CJK and most symbol ranges were routed to an index that cannot spell them. | The fallback ranges cover the extension planes and the whole symbol span. |
| p05 SQ03 | A scan that stopped at its candidate cap was returned like an exhaustive one. | `progress` reports completeness; `lcm_grep` reports `bounded_scans` and `complete: false`. |
| p05 SQ05 | Snippet offsets were taken in a case-folded copy and applied to the original. | Matching runs on the original text. |
| p05 RS01 | Preflight cleanup decisions survived a session reset. | Cleared with the rest of the session-scoped state. |
| p05 AX03 | A bounded or failed ancestry walk was indistinguishable from a resolved negative. | The walk reaches 256 hops and says when it was truncated or failed. The answer stays "foreground" **on purpose**: a wrong auxiliary classification means never storing the conversation. |
| p05 MT01/MT02/MT03 | Two backups in one second overwrote each other, concurrent rotates shared one scratch file, and success was reported for cached-only bytes. | Exclusive creation, per-call scratch file, fsync of file and directory, and no leftover on failure. |
| p05 OT01/OT02/OT03 | The session anchor overwrote the observation timestamp, an explicit date suppressed conflicting relative evidence, and a huge relative count raised `OverflowError`. | `session_date_at` is its own field; mixed evidence is ambiguous; out-of-range counts return unknown metadata. |
| p03 C01 | `/lcm clean` could delete nodes and messages that a node outside the deletion scope still referenced. | The delete transaction refuses and names what would be stranded. |
| p04 DG1 | Every `summary_nodes` read was `SELECT *`, decoded positionally. | One explicit projection, including the two FTS joins. |
| p02 T04/T16 | Expansion returned an assistant turn's tool calls as empty content; `lcm_recent` returned a bare `[]` for an unscannable window. | Tool calls are paged into the expansion; `complete`/`incomplete_reason` on recent. |

## Third pass — the four verification audits (commits `b832ea2`…`376b4e9`)

Four Codex astra auditors re-ran against the second pass: **verify-1** checked every claimed
fix, **verify-2** hunted regressions the pass introduced, **verify-3** asked whether the plugin
is finished on "every defect and every clear, definite optimization", **verify-4** audited the
whole plugin against the single no-unmarked-loss rule. Their reports are
`docs/claw-comparison/v1.0.0-verify-*.md`. All four said "not finished"; what they found and
what was done:

**Regressions the second pass introduced (verify-2) — all fixed**

| id | regression | fix |
|---|---|---|
| #1 | recovery used the engine's hermes_home on replay but not on ingest, so the two identities disagreed and a tool result dropped out of a leaf's sources | one home through every recovery call |
| #2 | protecting the bypass receipt made the NEWEST message the next removal candidate | whole-message removal stops before the live request; character trimming shrinks older messages first; the receipt gives way to its compact form before the request does |
| #5 | frontier recovery took the MAXIMUM source id as proof of coverage | only a proven contiguous covered prefix advances the frontier |
| #6 | the daemon pool called CPython private APIs; Python 3.14 changed them, so every parallel leaf pass at 1M crashed | the fork's own bounded daemon pool (found by the end-to-end run, not the suite) |
| #7 | the condensation cap counted groups, so the 256k anchor condensed only the first depth | the cap counts traversals of the depth loop, exactly upstream's single pass |
| #9 | an expand hint containing a bracket made the assembled prefix look like the sender's text, and it was stored as raw | the trailer is recognised as the whole last line |
| #10 | the index-shape gate rejected summaries that merely BEGIN with a refusal phrase | the phrase counts only when almost nothing follows it |
| #12 | a full ordered page was reported as an incomplete scan | completeness distinguishes "the rows ran out" from "the cap stopped us" |
| #13 | the leaf publication rebuilt its source set per source id | built once |
| #14 | snippets ran an ignore-case regex per term over the whole text | matched on the original text without the per-term rescan |
| #15 | an explicit date plus any relative expression was called ambiguous | only a relative expression that resolves to a DIFFERENT day conflicts |

**No-unmarked-loss violations (verify-4) — fixed**

| id | violation | fix |
|---|---|---|
| #3 | publication accepted a PARTIAL source mapping and advanced past the unmapped rows | every consumed row must map, or the leaf is refused |
| #4 | a KeyboardInterrupt left publication and cleanup transactions open for a later commit | rollback protection covers BaseException, commit included |
| #5 | a sanitised key could still displace another in one insertion order; duplicate JSON keys collapsed | the original keyspace is reserved first; duplicate-key arguments are cleaned as raw text |
| #6 | typed text blocks dropped their outcome siblings; a block with both `text` and `content` lost one stream | both are kept, with the outcome fields |
| #7 | removals inside tool arguments and self-closing injected tags were unmarked | a compact marker inside arguments; self-closing tags marked like any other |
| #8 | a condensed parent lost its children's loss receipts | the parent inherits them |
| #9 | the assembly omission receipt was the first thing dropped under budget pressure | it degrades to a one-line form, then to `last_assembly_omission_note` in status |
| #10 | orphan tool results were dropped with a log line; the stub for a missing result claimed a summary covered it | orphans are named with their call id and head; the stub says the result is in the raw store |
| #11 | a failed current-turn ingest was followed by a search reporting `complete: true` | `lcm_grep` reports the ingest failure and is incomplete |
| #12 | the raw-message scan's candidate cap was invisible | `MessageStore.search` reports progress; grep reports it as a bounded scan |
| #13 | a closed/unavailable summary database read as an empty window | `lcm_recent` reports it as incomplete |
| #14 | a truncated index block named a continuation that returned subtree metadata | `lcm_describe(index_offset=…)` pages the stored index block |
| #15 | an unreadable source row vanished from an expansion | `missing_source_store_ids` and `complete: false` |
| #16 | a node with more parents than the reverse-edge cap hid its retained parent | reachability asks for the current session's parents first |
| #17 | receipt counts were computed before later trimming, and compaction was not idempotent | counts are restated from the final result; both receipt forms parse |
| #18 | recovered spillover bytes were discarded when the durable copy failed | the bytes are stored inline instead |

**Also from verify-1/verify-3**: `add_node_with_meta`'s commit is inside its rollback
protection; the DAG LIKE fallback orders by the clock the ranking uses; `append_batch` never
loses a message to a short estimate list; every exact-id read is batched under SQLite's
variable ceiling; `lcm_recent`'s work-cap signal is no longer swallowed by the helper's own
handler; tool-call expansion is charged to the budget and continues from a `tool_calls_offset`;
window_scaling indexes its immutable metadata once; the test harness keeps its own controls.

### Still open after the third pass

- **verify-4 #2 — an edited already-ingested position loses the correction.** A cursor-side
  heuristic was tried and REVERTED: the active context legitimately reshapes after a
  compaction, so "this position differs from last turn" produced duplicate rows. The fix
  belongs in reconciliation (occurrence identity and revision hashes), i.e. the message
  envelope task below, and duplicating rows would be its own corruption.
- **verify-4 #1 — the raw store is still a projection.** `name`, `reasoning_content` and other
  host envelope fields are not persisted. This is the message-envelope task.
- **verify-4 #19 — filters, redaction and bypassed sessions are deliberate exceptions.** An
  ignored message becomes a hash placeholder with no stored row; a stateless session stores
  nothing; sensitive redaction is irreversible. These are operator policies, not defects, but
  they ARE exceptions to the no-loss rule and are now written down as such in FORK.md.
- **verify-4 #20–#24 and verify-3 #14–#28** — the opt-in subsystems (rollups, assertions,
  adaptive retrieval, evidence packs, trajectory, embeddings, query views) certify incomplete
  or stale evidence in several places. All are default-off; they are the next block of work.
- **verify-3 #4 / p05 RS02** — reset and rebind are still unfenced against in-flight
  compaction; it needs a session generation carried through publication.

### Deliberately not changed (and why)

- **p05 MA01** (a genuine "Acknowledged." is skipped as synthetic noise): the message is stored
  and retrievable; only the summariser's input skips it. Fixing it properly needs a trusted
  synthetic-origin signal from the host, which belongs with the message envelope (§ below).
- **p05 MC01** (a literal JSON string is indistinguishable from structured content after
  storage): the bytes are never lost — this only affects filter policy. The fix is a persisted
  content-type identity, i.e. the message envelope task, not a local patch.
- **p05 AX02** (recorded subagent start events are not authoritative in the child classifier):
  broadening auxiliary classification is the direction that ends in **not storing** a
  conversation. It waits for a host contract that distinguishes a recorded subagent identity
  from generic parent metadata.
- **p05 RS02** (reset/rebind is not fenced against in-flight compaction): needs a session
  generation carried through publication — the one publication contract in Phase 5, not a
  local guard.

## Fourth pass — round-2 verify-2 regressions (commits `3f318a0`, this one)

The second auditor round re-ran verify-2 against the third pass and reported 12 confirmed
regressions (3 P1, 9 P2). All twelve are closed; each has a fork test that fails on the
previous revision.

| # | what the audit found | what the fork does now |
|---|---|---|
| #1 | the orphan-result receipt became the newest message and displaced the live request | receipts are placed BEFORE the newest user turn and never take its protection |
| #2 | inline-recovered bytes had no replay identity, so compaction could not map the row | the marker row keeps its identity; the recovered bytes live in an extra archive row |
| #3 | deadline expiry between condensation depths escaped as a bare `TimeoutError` | it is raised as `SummaryUnavailableError`, so committed progress is published and the cooldown arms |
| #4 | the incremental packing estimate is not additive, so an exactly-fitting summary was dropped | a candidate is re-counted exactly before it is rejected |
| #5 | the compact omission footer was not recognised as scaffolding and was re-ingested as raw text | renderer and recogniser share the prefix constant; the test walks every emitted shape |
| #6 | each attempt built its own lookahead pool: 2 → 4 → 6 live workers over three retries | one pool per engine, reused across attempts and released in `shutdown()` |
| #7 | condensation without leaf work published a parent and returned `noop` with the old context | `_maybe_condense()` returns what it published; compress() reassembles and accounts for it |
| #8 | a finished body reported offset 0, so every tool-call page re-sent the whole body | body and calls keep independent EOF cursors, both carried in the continuation |
| #9 | inherited receipts made a "condensation" larger than its sources | verbatim while it fits, else one aggregate receipt naming the children that hold them |
| #10 | the index-shape gate rejected short historical facts that open like a refusal | with the source in hand the gate asks what the reply SHARES with it (two specific words) |
| #11 | a correct ordered page was reported as a work-cap failure | `complete` / `more_available` / `work_capped` are separate facts in every search path |
| #12 | `build_snippet` regex-scanned per term and re-folded the source after each miss | ASCII sources take a plain scan over one folded copy: 104ms → 8ms on 2.2M characters |

## Fifth pass — round-2 verify-3 (the cheap, definite half)

The round-2 verify-3 report ("remaining defects and definite optimizations") lists 27 defect
groups and 6 optimizations, most of them architectural or in the default-off subsystems. These
are the ones that were concrete, reachable in the default configuration and repairable locally;
each has its own probe.

| what the audit found | what the fork does now |
|---|---|
| `_temporary_sqlite_busy_timeout` logged through a name it never imported: one refusing connection raised `NameError` and left the others at the 5ms timeout | `sqlite_util` has its own logger; every changed connection is restored |
| an explicit token-cache size below 2048 was floored back to 2048 | the size is the maximum of the LIVE owners' requests; the default applies only when there are none |
| the node INSERT sat outside the rollback protection, so a refused statement left its transaction open for an unrelated commit to publish | insert, sidecar and commit are inside one `BaseException` guard in both `add_node` and `add_node_with_meta` |
| `lcm_recent` counted its sections AFTER the display limit: 11 matches with `limit=10` read as "10 sections, not truncated" | the window's own count is taken first; `total_sections` is what the window holds |
| a frontier computation that raised failed closed: an empty window reported `complete: true` | it raises `_RecentIncomplete`, so the answer says it could not be computed |
| a response with `status="incomplete"` was accepted when the choice said `"stop"` | either signal refuses the text and escalates |
| a padded data URI followed immediately by prose swallowed the words after the padding | the payload class stops at its `=` padding |
| tool-call continuation restarted the finished body (also verify-2 #8 / O4) | body and calls keep independent EOF cursors |

## Sixth pass — round-2 verify-4 (the doctrine audit, second round)

The second verify-4 round reported 39 violations. Everything default-reachable and repairable
without a new subsystem is closed; each has a fork test that fails on the previous revision.

| # | what the audit found | what the fork does now |
|---|---|---|
| #1 | recovered bytes were dropped by `append()` and the batch archive row belonged to no node | `append()` protects through the list path; the leaf claims the archive rows for its own tool_call_ids with a receipt |
| #2 | (a fork regression) an omission receipt anywhere in a message beginning with a summary header made it "our scaffolding" | the receipt has to END the message, exactly as assembly emits it |
| #3 | publication read `self._session_id` AFTER the summariser returned | a publication fence (session id + generation) is captured first and validated before the node is written |
| #4 | the raw store dropped every host field the columns do not project | `envelope_extra` keeps them verbatim; replay and expansion give them back |
| #6 | a GC callback that raised left the content rewrite pending for a later commit | rewrite, callback and commit are one `BaseException`-protected transaction |
| #13 | serialisation elision cut away an earlier injected-context receipt and under-reported the loss | marker fragments are carried out of the elided span; the pre-sanitisation size is named |
| #14 | typed-text and media blocks dropped substantive sibling fields | both streams render; further fields are named in a receipt |
| #15 | inline unmatched tags, header removal and multiple inline data URIs were unmarked | all three marked; attachments are counted |
| #16 | a turn that lost only its `<think>` block left no trace | the assembly receipt counts redacted turns beside dropped ones |
| #17 | the assembly receipt was dropped when neither form fitted | a ~12-token minimal receipt is emitted whenever it fits at all |
| #18 | receipt inheritance recognised only the `[LCM:` spelling | every marker spelling this module writes is inherited |
| #19 | the missing-result stub promised an archived result that never existed | archived (with ids), present-but-unreplayable, or never received |
| #20 | `lcm_expand(store_id=…)` omitted the row's tool calls | rendered and paged, arguments through the compaction sanitiser |
| #21 | a reachability bound or a failed parent read read as "not found" | tri-state: found, absent, or unresolved with the reason |
| #22 | `lcm_recent` counted sections after the display limit; a failed frontier read as empty | closed in the fifth pass |
| #23 | the recall full-text arm erased the grep result's incompleteness | hits kept, coverage reported as bounded with the reason |
| #24 | expansion synthesis hid missing selections, unprocessed nodes and truncated answers | all four are reported; `complete:false` |
| #26 | an unreadable sidecar was indistinguishable from a node without one | description says unavailable and why; status carries the read error |
| #27 | non-ASCII symbols were deleted from the query, so `flag∀` matched `flag∃` | such queries go to the substring scan |
| #29 | the bypass receipt counted content only | the whole removed envelope, tool-call arguments included |
| #33 | a truncated payload file loaded as a successful empty result | missing or short content is an explicit corruption outcome |
| #34 | coverage scored what survived two silent caps; the orphan check ignored node sources | bounded scores warn; both source types are walked and the population is named |
| #35 | a same-named trigger with a wrong body passed repair | definitions are compared, owned triggers are recreated, the index is rebuilt |
| #37 | a real EIO on the backup directory fsync was swallowed | only genuinely unsupported operations are tolerated |
| #38 | an empty extraction response meant "nothing to extract" | only the explicit answer does; the note names its complete span |
| #39 | rotation advanced past what its capped marker read | the span is paged; a failed page keeps the frontier where it is |

**Still open from this round:** #5 is now CLOSED for hosts that give their messages stable ids
(the correction is archived as a new row naming the row it supersedes, and the superseded
version stays); without host ids an edit to an already-ingested position is still invisible,
because the only signal left is list position and the cursor-side heuristic that used it was
reverted for producing duplicate rows. Also open: #7–#12 and
#30–#32, #36 (the default-off subsystems: adaptive retrieval, requirements/assertions, rollups,
embeddings, trajectory, backfill counters), #25 (the host's own response normalisation, outside
the plugin), and #28 (the configured exclusions, documented in FORK.md as deliberate).

## Seventh pass — round 3 of the verification audits

Round 3 ran against a pinned tree (`5475e94`) and reported, per auditor: verify-2 ten
regressions (3 P1), verify-3 a 25-group closure list, verify-4 39 doctrine violations,
verify-5 the subsystem list. It also credited as fixed: the three cancellation boundaries,
trajectory losslessness, `lcm_recent` totals and frontier failures, the closed-DAG rollup
refusal, the GC rewrite primitive, recall's adapter, exact-fit assembly, the condensation
deadline, body-first pagination, ordered-search completion, ASCII snippets, FTS trigger
repair, index-block continuation, batching, and the DAG rollback guard.

Closed in this pass:

| finding | what the fork does now |
|---|---|
| verify-2 #1 (P1) | protection returns one row per input; recovered-body archive rows are returned separately and stored after the row they belong to, so a placeholder can no longer land on the live request |
| verify-2 #2 (P1) | the marker row, its attachments and the commit are one `BaseException`-protected transaction |
| verify-2 #3 / verify-4 #1 (P1) | one publication lock across validation, the node write and the frontier; `on_session_start`/`on_session_reset` take it; the assembled result is fenced again before it is handed over |
| verify-2 #4 | the shared pool purges cancelled work when an attempt closes |
| verify-2 #5 | the refusal gate no longer demands lexical overlap: only a reply that is both almost empty and shares nothing with the source is a non-answer |
| verify-2 #7 | a term whose meaning is a symbol the index deletes must actually match; a standalone symbol stays a routing trigger |
| verify-2 #8 / verify-4 #3 | revisions compare the whole envelope; revision rows stay out of the chronological walk, resolve by host id, and are covered by the leaf that covers their original |
| verify-2 #9 | a standalone minimal receipt is recognised as scaffolding |
| verify-2 #10 | an unchanged prefix costs no database work; missing-result lookups are one batched query |
| verify-3 | partial condensation keeps its published count; the GC chunk-archive failure reaches its transaction; the extraction manifest names every source id as ranges |
| verify-4 #4 | a late session-end flush runs under the ended session's identity |
| verify-4 #5 | the omission header's bullets must be the ones the marker writes |
| verify-4 #6 | every abandoned tail message is counted; the fallback that rescues the latest message tests for content, not for message count |
| verify-4 #7 | leading turns a request cannot start with are named in a receipt |
| verify-4 #8 | the envelope reaches the summariser (inline outcomes, receipt for the rest) and node expansion; unrepresentable timestamps and corrupt envelope JSON are preserved |
| verify-4 #9 | every rendering branch inventories what it did not render; 0 and False are values; a rewritten JSON key carries a receipt |
| verify-4 #10 | marker fragments are carried whole; inheritance catches receipts after visible text |
| verify-4 #11/#12 | payload corruption survives every expansion path; synthesis completeness is the conjunction of selection, searches, hydration, truncation and synthesis |
| verify-4 #13 | `lcm_describe` pages a node's own summary and a truncated child carries that continuation |
| verify-4 #14 | a timed-out or bounded retrieval is reported as incomplete, not as "no progress" |
| verify-4 #15/#16 | unexamined candidates that state values of the requested kind refuse sufficiency; a unit clause the filter cannot read refuses finite coverage |
| verify-4 #17 | a question keeps its own year and currency, or the contract is refused |
| verify-4 #18/#19 | a value must be stated by one clause of the quote, the label must be in that clause, a negated numeric value is refused, and numbers are complete lexemes compared exactly |
| verify-4 #20 | only the computation's own answer is verifiable |
| verify-4 #21 | a binding failure before ingest counts as an ingest failure |
| verify-4 #24/#25 | archive rows name the row they belong to; payload filenames stay inside the reader's grammar |
| verify-4 #27 | a truncated generation is not a payload, in every structured adapter |
| verify-5 #1/#4/#5 | cancellation-safe sidecar transactions, lossless trajectory ingestion, canonical-only verification |

### Eighth pass — no truncation at ANY window (user directive)

The 256k anchor carries upstream's TUNING values only. It never carried upstream's LOSS: the
no-truncation fixes and the optimizations apply at every window. Two places still cut content
at 256k and no longer do:

| what cut | now |
|---|---|
| `serialize_message_max_chars` | both curve endpoints are 4 chars/token of their own anchor window (1,048,576 at 256k, 4,000,000 at 1M), so the summariser sees whole messages at every window. `0` (no window known yet) means NO CAP — `engine.py` used to clamp that to 64 chars. An explicit operator cap is still honoured and still cuts only through a sized `[LCM elided …]` marker. |
| externalization fallback | when externalization is disabled or its path is unwritable, upstream truncated the inline tool body to 3000 chars. The body now stays inline and whole. Two upstream tests (`test_lcm_core.py`) were rewritten to assert that. |

### Ninth pass — core only (the opt-in subsystems stay in, stay off, stay unaudited)

| what was lost | now |
|---|---|
| active-context cleanup stripped `<think>` / reasoning parts out of every replayed assistant turn and said so only in the log. Only `_assemble_context` counted it; below-threshold cleanup, bypass trimming and forced-overflow recovery reported nothing. | the turn carries `marked_loss.INTERNAL_REPLAY_MARKER` itself (`sanitize._mark_internal_removal`). It is positionally neutral, so it can never displace the caller's newest message, and it is a fixed string, so `reconcile` still maps the replayed turn back to its stored row. A turn that held NOTHING is still dropped outright — inventing a receipt for an empty turn would be a false claim of removal. Under an assembly cap, a turn that is now *only* the receipt is held out of the budget pass and named in the prefix instead (`_is_internal_replay_receipt_only`). |

| corrupt envelope JSON was cut at 20,000 chars in `store._row_to_dict` with no marker and no cursor, and `lcm_expand` on a RAW row did not report the corruption at all — it answered `has_more=false` while the host fields sat unreadable in the column. | the store returns `envelope_raw` whole with `envelope_raw_chars`; both expansion paths page it through the existing `envelope_offset` cursor and emit `envelope_raw_truncated` / `_next_offset` / `_continue_with`. |
| summariser input rendered a tool call as `name(arguments)` and dropped everything else the provider attached to it; a tool call that was not a dict was filtered out entirely; a nested `{"type":"text","text":{"value":…,"annotations":[…]}}` object's other fields vanished, because the block inventory only sees the OUTER block's keys. | `marked_loss.tool_call_fields_note` / `unrepresentable_tool_call_note` name the first two; `_unrendered_field_receipt` takes a `prefix` and inventories the nested object as `text.<key>`. |
| `_summary_frontier_nodes` loaded every node with `limit=100_000` and filtered in Python, so a session past that limit computed its frontier from a TRUNCATED set: condensation could re-publish over sources a node above the cut already covered, and the prefix would omit real summaries while reporting nothing (B7). | `dag.get_frontier_nodes` — the same SQL predicate as `get_frontier_token_total`, unbounded, so the count and the token sum cannot diverge. |

### Tenth pass — verify-6 (core-only audit, prompt `verify-6-core.txt`, report `docs/claw-comparison/`)

Ten P1 and one P2, all confirmed with probes against `015eef4`. Four were regressions this
fork introduced in the eighth/ninth passes. All eleven are fixed:

| # | what was wrong | fix |
|---|---|---|
| 1 | ingest read its ownership from mutable engine state again AFTER the (slow) protection work, so a rebind landing in between filed the old turn under the NEW session and then advanced the new session's cursor past a request that had never been stored | the session, conversation and publication fence are captured once at entry; rows are written under that identity, and the cursor is only advanced while the engine is still bound to it |
| 2 | the legacy recovered-body fallback ran only when the WHOLE chunk had no explicit attachment, so one modern attachment stranded every legacy body in that chunk behind the advanced frontier (regression from the ninth pass) | `store.recovered_body_ids_for_consumed_rows` resolves per consumed row: explicit link first, legacy call-id fallback for rows without one, and never a call id an explicit link already claims |
| 3 | a stable-ID edit was DISCARDED and cached as settled whenever the stored version was an externalized reference — an identifiable correction disappeared | rows rewritten by ingest protection carry `lcm_pre_protection_fingerprint` (a digest of the message as it arrived); the comparison uses it, and without one the id stays unsettled so a later turn retries |
| 4 | a failed revision lookup or write was logged and swallowed; the enclosing ingest reported success and `lcm_grep` answered `{"complete": true, "results": []}` for the corrected text with both failure counters at zero | the failure goes through `_record_ingest_failure`, the host id is held in `_unarchived_revision_host_ids`, `_record_ingest_success` refuses to clear the streak while one is outstanding, and `lcm_grep` names the ids in its search failures |
| 5 | the message cap was divided by six for tool arguments, so a 200,000-char argument lost its tail at 256k and survived whole at 1M — with an injected-context receipt inside that tail | arguments share the message cap (which is the whole window), and `elide_args` carries every receipt whose span crosses an operator-configured cut |
| 6 | envelope slices were neither charged to the node-expansion budget nor consulted by its continuation decision, completed tool-call cursors reset to 0 in envelope continuations, and the raw path's call continuation reset the CORRUPT envelope cursor | one accounting and one cursor for body, calls, normal envelope and corrupt envelope: every field is charged, an unfinished field keeps the source open, a finished field parks its cursor at its own end, and `envelope_offset`/`next_envelope_offset` travel with every page |
| 7 | expansion synthesis upgraded incomplete traversal to `complete=true`: child blocks carrying only a failure were dropped, and the final check saw missing raw rows but not corrupt payloads, missing children or field truncations | `_expand_child_nodes` reports `complete=False` with a reason, failure-only blocks are kept, field-level truncation makes the context truncated, and `complete` is the conjunction of every block outcome (reported as `incomplete_context_blocks`) |
| 8 | `lcm_load_session` returned content and the column fields only, so a tool row's `is_error`/`exit_code` never reached the reader — a failed operation read exactly like a successful one | outcome fields are rendered (`marked_loss.envelope_inventory`), the rest are named in `envelope_fields_omitted`, corruption is flagged, and `envelope_recover_with` says how to read them in full |
| 9 | the structured cleaner returned after the FIRST text field and judged the block by that one field, so a block with reasoning in `text` and a visible failure in `content` lost the failure, and a text block whose only payload was `annotations` (or a reasoning block holding `encrypted_content`) disappeared with no receipt (regression from the ninth pass) | every text field is stripped and the block is judged afterwards; `_structured_part_text` joins all of them; `_part_has_substantive_fields` keeps a block whose payload is not its text; `_content_carries_text` counts every non-structural key |
| 10 | each bypass compaction rewrote every surviving receipt with counts from the LATEST reduction alone, erasing the earlier record — two receipts both claimed 6 messages / 3,046 chars where 10 / 5,100 had gone | the receipt is cumulative: earlier receipts' recorded counts are carried forward, this call's reduction is added once, and one receipt is emitted instead of several copies of the same total |
| 11 (P2) | `dag.reassign_session_nodes` lacked the rollback discipline of node publication, so a failed carry-over commit left the ownership change pending and an unrelated later publication committed it | same `except BaseException: rollback; raise` protection as `add_node_with_meta` |

**Default-reachability, fixed on the owner's instruction:** `lcm_compile_evidence` and
`lcm_evidence_pack` were advertised AND dispatched in the default configuration, passing
`enabled=True` to the compiler themselves, so an opt-in subsystem ran with every flag off. They
now answer `status: disabled` until `preanswer_evidence_enabled` is set — the contract
`lcm_query_state` and `lcm_retrieve` already followed. Four upstream test modules that exercise
the enabled behaviour set the flag in their engine fixture (`test_evidence_pack.py`,
`test_evidence_compiler.py`, `test_evidence_contract.py`,
`test_evidence_pack_host_activation.py`); they are listed in `docs/fork-touchpoints.md`.
`lcm_compute` stays callable — it is a pure function over refs the caller supplies.

Rejected in this pass: a completion-cue/negation gate on `_source_supports_assertion_value`
(verify-4 #23). Every whitelist of completion verbs rejects real ones ("I submitted the
report"), and a wrong rejection drops an assertion silently — that is loss too. Left open.

Also rejected: a generated-scaffold digest ledger, so that a user pasting our assembled prefix
verbatim is stored instead of classified as our own scaffolding. The ledger would have to be
authoritative to help, and then any host-side normalisation of our prefix would make it fail to
match — re-ingesting our own generated summaries as raw conversation, which is a worse and far
more common failure than the paste. The paste case also loses no CONTENT: what is dropped is a
copy of summaries the DAG already holds. Left open.

**Still open after this pass:** verify-4 #23 (positive "completed" accepted from a negated
source), verify-4 #6's smallest case (a budget too small for even a
12-token receipt records the omission in `lcm_status` instead of the prefix — a deliberate
trade-off, since making room would drop the caller's own latest message), #10's structured
omission records (architectural), #26 adaptive finalisation staleness, and the verify-5 tail:
embeddings association/coverage, trajectory recovery and indexing, rollup snapshots, query-view
freshness, assertion polarity and scope, and the window-policy integration for optional
subsystem capacities.

## WHAT IS LEFT TO DO

This is the authoritative remaining-work list as of the tenth pass (`52bf344` + the evidence
gate). Everything above this line is history; the partitioned-audit list below it is the raw
material these items were distilled from and contains entries that have since been fixed.

**Ship state:** no known loss on the core path. Suite 3326 passed / 1 skipped / 12 xfailed;
both e2e anchors clean (0 unreachable rows, 0 missing facts, 0 facts never offered to the
summariser) at 262144×400 and 1000000×3000 against the deployed plugin.

Sections: **A** core path (the only category that can still lose something) · **B** correctness
and operability · **C** ported-capability decisions · **E** routine standing tasks (upstream
sync, lossless-claw watch) · **F** outstanding verification (strict superiority, claw gap) ·
**G** explicitly not being done.

### A. Core path — the only category that can still lose something

| id | what | why it is still open |
|---|---|---|
| A1 | **Summariser prompt design** (Phase 0 below). The prompt is five layers of patchwork with contradictions ("or drop", a 60–70 % focus skew). A summary that omits a topic is loss the markers cannot describe, because the summariser was never told the topic mattered. | Needs a written design (`docs/prompt-design.md`) agreed first, then one fork module `summary_prompts.py`, then measurement — not a quick edit. |
| A2 | **Index-navigation evaluation gate** (Phase 0.3 / Phase 5). There is no number for "can a reader pick the right node to expand from the rendered prefix". `lcm_doctor coverage` measures term survival, not navigability. | Requires a fixture set and a real model run at both anchors, before/after A1. |
| A3 | **Backup must cover externalized payloads**, not just SQLite. A restored database can reference payload files the backup never copied — the reference resolves to nothing and the expansion says the bytes are unrecoverable. | Task #14. Physical backup publication and `fsync` durability were also *not* exercised by the verify-6 audit (read-only workspace), so power-loss recovery is unverified. |
| A4 | **A pasted copy of our own assembled prefix is classified as scaffolding and dropped.** No content is lost (it is a copy of summaries the DAG holds) but the fact that the user said it is. Rejected fix recorded in the ninth pass. | Needs a trusted generated-message identity from the host, or a digest ledger whose failure mode is worse. Blocked on a host contract. |
| A5 | **verify-4 #6's smallest case**: an assembly budget too small for even the 12-token receipt records the omission in `lcm_status` instead of the prefix. | Deliberate trade-off — making room would drop the caller's own latest message. Revisit only if a host reports it. |
| A6 | **verify-4 #10 structured omission records**: receipts are prose lines, not machine-readable records. | Architectural; would change every marker's shape. |

### B. Correctness and operability (no known loss, but unproven or rough)

| id | what |
|---|---|
| B1 | `p03`: a transaction can still lose rollback protection under concurrent use (the `reassign_session_nodes` case is fixed; the general pattern is not audited). |
| B2 | `p06`: Hermes' current spillover directory is not recognised; the always-on ingest guard can canonicalise surrounding JSON. |
| B3 | `p11`: the suite tests rows far more thoroughly than usable provenance — it does not establish that a stored summary is complete or functions as an index. (A1/A2 subsume most of this.) |
| B4 | `audit D` medium: 256k structural equivalence still fails on oversized/imported histories and on the dynamic-chunk path; whole sidecar index blocks escape retrieval budgets. |
| B5 | CI on the `betterlcm` branch; the host-integration lane must fail, not skip, when the host import fails (T2.5). |
| B6 | Docs/skill/defaults generated from `ENV_FIELD_SPECS` + anchors so they cannot drift from the curve (T2.4). |
| B7 | **Make `docs/fork-touchpoints.md` executable (catches M2).** Today it is prose: a human has to notice that upstream renamed the function our hook lived in, because the hook's own unit test still passes while the hook is never reached. Turn the table into a manifest (`docs/fork-touchpoints.yaml` or a dict beside it) naming, per entry, the module, the symbol, and a reachability assertion; then one `tests/fork/test_touchpoints.py` that imports each symbol, fails if it is gone, and — for hooks whose whole point is that they RUN — exercises the real call path and fails if the fork behaviour is absent. Every existing entry needs one; this is the single highest-value piece of merge insurance the fork does not have. |
| B8 | **Anchor-vs-upstream-default check (catches M6).** Every `window_scaling.py` low endpoint claims to *be* upstream's value at 256k, and the README and FORK.md repeat that claim. Nothing enforces it. Add a test that resolves the curve at 262,144 and asserts each non-fraction anchor equals the corresponding upstream `LCMConfig` dataclass default at the pinned `.upstream-base` — so the day upstream moves a default, the fork's central claim fails loudly instead of quietly becoming false. (`tests/fork/test_window_scaling.py::test_at_256k_equals_upstream` hardcodes the numbers today; it should read them from upstream.) |
| B9 | **Fork-invariant probe set for merges (supports M7/M8).** The e2e anchors prove no-loss end to end but say nothing about the cross-module contracts the fork *reads* rather than owns: the serialised-message shape, the externalized-placeholder string, replay identity in `reconcile`, tool dispatch, the store row projection, and the migration/classifier interaction between the fork's schema additions and upstream's. Collect one probe per contract so a merge can re-verify them in minutes, and run a merge once against a COPY of a real `lcm.db` rather than only fresh test databases. |

### C. Ported-capability decisions (audit E, still unmade)

| id | what |
|---|---|
| C1 | The nine cheap claw items: test-home isolation, prompt-prefix divergence diagnostics, release-commit validation, payload-reference disambiguators, a script-aware token estimator, prompt inspection commands, copied-reference parsing, shadow-install/drift detection, release fragments. |
| C2 | The six capability-level ones: `context_items` projection, operator TUI, persistent focus briefs, delegated retrieval workers, richer maintenance debt, a paged expansion-cost manifest. |
| C3 | (moved — see R2 under Routine tasks.) |

### E. Routine (standing) tasks — triggered, never "done"

These two have no completion state. They fire on an external event and each run ends with a
dated record, so the next maintainer can see when the fork was last brought level.

**R1 — Upstream changed: bring the fork to the newest upstream state.**
Trigger: any new commit on `stephenschoettler/hermes-lcm` past `.upstream-base`
(currently `8d1b1e6`).

**Textual conflicts are the easy part and the least of it.** `git` reports a conflict only when
two edits touch the same lines. This fork's real exposure is everything git merges *cleanly*:
upstream changing something in a place we already changed, in a different hunk, in a caller, or
in an assumption. That is the standing hazard of maintaining a substantial fork, and it is
invisible to both `git` and a green test suite — our suite tests OUR behaviour, so it stays
green while upstream's new behaviour goes unexercised.

**The failure modes, each with the only thing that actually catches it:**

| # | how a clean merge breaks the fork | what catches it |
|---|---|---|
| M1 | **Semantic conflict, no textual conflict.** Upstream edits a different hunk of a function the fork also edited. Both edits apply; the *combination* is wrong (an upstream early-return placed before our marker; a reordering that makes our guard run after the thing it guards). | Review per FUNCTION, not per hunk: for every symbol named in `docs/fork-touchpoints.md`, read the merged body end to end and re-derive what it now does. A three-way diff of the hunks is not enough. |
| M2 | **The hook silently detaches.** Upstream renames, moves, splits or stops calling the function our hook lives in. Our fork code still exists, still passes its unit test, and never runs. | Every fork hook needs a test that fails when the hook stops being **reached** — not one that only checks the helper's output. This is work item **B7** and does not exist yet. |
| M3 | **Upstream fixes the same bug differently.** Now there are two mechanisms for one problem: two markers on one cut, an elision applied twice, a receipt counted twice, or two competing guards that disagree at the edges. | For every upstream fix, ask "did we already fix this?" before taking it. Keep exactly one mechanism, delete the other, and say in the ledger which survived and why. |
| M4 | **Upstream reintroduces loss somewhere new.** A brand-new code path that truncates, drops, or claims completeness over bounded work. No test of ours covers code that did not exist yesterday. | The doctrine question asked of every added path, plus a scan of the merged tree for the vocabulary of loss (`[:N]`, `[-N:]`, `...`, `truncat`, `limit=`, `break` in a collection loop, `except: pass`, `complete.*True`) in anything upstream added. |
| M5 | **Our fix becomes obsolete or actively wrong.** Upstream restructures the thing we worked around; our patch is now dead weight, or worse, fights the new structure. | The touchpoints table has a "keep ours / re-apply / reconcile" column for exactly this. Removing a fork patch is a legitimate merge outcome; record it. |
| M6 | **Upstream changes a default the curve mirrors.** Every `window_scaling.py` low endpoint claims to BE upstream's value at 256k. If upstream moves a default and the anchor does not, the fork's central claim quietly becomes false. | A test that reads upstream's dataclass defaults at the pinned base and asserts each anchor's low endpoint still equals them. Work item **B8**; does not exist yet. |
| M7 | **A contract we depend on inverts.** Upstream changes the shape of something our code consumes but does not own — the serialised message format, the externalized-placeholder string, replay identity, the tool-schema dispatch path, a store row's projection. Our code keeps parsing the old shape and silently matches nothing. | Enumerate the cross-module contracts the fork reads rather than owns and re-verify each one by probe after the merge. The reconcile/replay-identity path is the sharpest of these. |
| M8 | **Schema and migration divergence.** Upstream adds a column, index or migration next to the fork's own (`envelope_extra`, `host_message_id`, `lcm_node_meta`, `betterlcm_node_meta_v1`). Migration order, classifier logic and downgrade behaviour all interact. | Run a merge against a COPY of a real `lcm.db`, not only fresh test databases, and check both directions (fork build reading an upstream DB and back). |
| M9 | **Test drift in both directions.** Upstream adds tests asserting behaviour we deliberately removed (should fail — good, that is the signal), or edits a test we had re-pointed, and the merge silently restores upstream's assertion. | Every re-pointed or removed upstream test is listed in `docs/fork-touchpoints.md`; after a merge, re-check that list line by line rather than trusting a green run. |
| M10 | **Divergence debt.** Each merge that says "keep ours" without reconciling widens the gap until the next merge is unreviewable. | Merge early and often. A merge deferred until upstream has moved 500 commits is not a merge, it is a rewrite. |

**Procedure — analysis first, merge second:**

1. `git fetch upstream`; read `.upstream-base..upstream/main` in full — runtime diff, the
   surrounding upstream code, and every changed test. Write down what changed and why.
2. Compute the **intersection**: which upstream-changed symbols also appear in
   `docs/fork-touchpoints.md`. That set is the M1/M3/M5 risk surface and gets read function by
   function, in the merged tree, after merging — not as hunks.
3. Classify every upstream hunk: *(a)* file the fork only hooks → keep ours, re-apply the hook
   on top of theirs; *(b)* upstream fixed something we also fixed → reconcile, keep one
   mechanism (M3); *(c)* behaviour the fork depends on → decide explicitly, never by merge
   default (M7); *(d)* new feature → is it coherent with the no-loss doctrine, and does it need
   a curve anchor rather than a fixed 256k-shaped constant?
4. **Ask of every upstream change: does it (re)introduce loss?** New truncation, a new silent
   drop, a new completeness claim over bounded work. If yes, take the feature and remove the
   loss, exactly as the fork already did for the L3 fallback and the 3000-char cut (M4).
5. Walk the whole touchpoints table line by line and confirm each entry is still true of the
   merged tree: the hook exists, it is still reached, and its reason still applies (M2, M5, M9).
6. `bash scripts/test.sh` green. Upstream tests that pin removed behaviour get re-pointed and
   listed in the touchpoints file — never deleted silently.
7. Both e2e anchors clean against the DEPLOYED plugin
   (`scripts/e2e_no_loss.py 262144 400` and `... 1000000 3000`). These are the executable form
   of the fork's invariants: 0 unreachable rows, 0 missing facts, 0 facts never offered to the
   summariser. A merge that cannot produce those numbers is not finished.
8. Redeploy the clone, re-pin the revision, update `.upstream-base`, record the merge in this
   ledger as a new pass — including every "keep ours", every reconciliation and every fork
   patch deleted as obsolete.
9. Re-run **V1**: an upstream merge is exactly when strict superiority can silently break.

**R2 — lossless-claw released or moved: evaluate and port what is better.**
Trigger: any commit or release of lossless-claw newer than the one already analysed —
**v1.0.0** (analysis in `docs/claw-comparison/v1.0.0.md`, sources were at
`lossless-claw v1.0.0`).

1. Diff the new claw version against the analysed one. Every changed file, not just release
   notes.
2. Sort the changes into: **new capability**, **behaviour change**, and **bugfix**.
3. **A claw bugfix is a lead, not just a port candidate.** Both projects solve the same problem,
   so a bug claw fixed very often exists here in an analogous shape — different code, same
   mistake. For each fix, find the corresponding place in this fork and prove by probe whether
   the same defect is present, then fix it here on its own merits even if the claw patch itself
   is not portable.
4. For each capability/behaviour change, decide **useful AND coherent**: does it serve the
   fork's purpose (no loss, 1M windows, an index into recoverable history), and does it fit the
   architecture without importing an assumption the fork rejects? Rejecting is a valid outcome
   — record the reason.
5. Run the four-aspect comparison (`docs/claw-comparison/prompts/prompt-{1..4}-*.txt`) against
   the new version and write `docs/claw-comparison/vX.Y.Z.md` with ported / rejected / open,
   each with a reason. That file is the record that this task ran.
6. Anything ported lands with a fork test and goes through the usual loop (suite → commit →
   redeploy → re-pin → both e2e anchors).

### F. Outstanding verification (never yet run to completion)

| id | what | why it matters |
|---|---|---|
| V1 | **Is this fork strictly superior to upstream mainline?** A dedicated audit that walks every behavioural difference and asks, for each: is the fork at least as good in every respect, or did it trade something away? Audit D (`v1.0.0-audit-d-regressions.md`) did this once against `8d1b1e6` and found three P1 fork regressions; it has **not** been re-run since, and this fork has changed heavily. Known open leads from that round: 256k structural equivalence still fails on oversized/imported histories and on the dynamic-chunk path, and whole sidecar index blocks escape retrieval budgets (B4). | "No loss" is not the only guarantee upstream offers. A fork that fixes loss and quietly loses throughput, a routing behaviour, or a host contract is not strictly better. Re-run after every R1. |
| V2 | **Did we miss anything the CURRENT lossless-claw does better?** The four-aspect comparison and audit E were run against claw **v1.0.0** and 37 audit-E items were left undecided (C1/C2 above). Repeat against whatever version is current, and treat the undecided v1.0.0 items as part of the same question rather than a separate backlog. | The v1.0.0 sweep explicitly found that the earlier "does it change a guarantee" filter had been an invalid reason to reject candidates — so the fork is known to have rejected good ideas for a bad reason at least once. |

Both are audits, not fixes: each ends with a written report under `docs/claw-comparison/` and a
pass entry in this ledger, whether or not it produces work.

### G. Explicitly not being done

- **The opt-in subsystems** (`reasoning`/`lcm_compute`, assertions, the evidence compilers,
  adaptive retrieval + query views, embeddings, rollups, trajectory — ~20,700 lines). Owner's
  decision: they stay in the fork, stay default-off, and are neither audited nor modified.
  Deleting them would only make the upstream merge painful for no behavioural gain. The whole
  verify-5 subsystem tail (embeddings association/coverage, trajectory recovery and indexing,
  rollup snapshots, query-view freshness, assertion polarity and scope, verify-4 #23, #26) is
  closed as "not our problem" under that decision.
- **Reproducing upstream's cuts at 256k.** The anchor carries tuning values only.

## Still open — from the partitioned audits (historical raw material)

Fourteen audits ran: four aspect comparisons against lossless-claw, three cross-cutting
(whole-plugin, plan critique, unplanned claw capabilities), a regression hunt against upstream,
and eleven exhaustive partitions covering all 70 modules, the non-code assets and the test
suite. Their reports are in `docs/claw-comparison/`. The findings below are NOT yet fixed; each
report carries the citations and the reproduction.

- **p01 engine-core** — publication and session ownership not coupled; late work can publish
  under a different session; indexes that describe only the ends of their sources.
- **p02 agent-tools** — tools that mistake incomplete work for complete results (eleven items
  reported as a complete ten; an orphan check passing after 1,000 nodes).
- **p03 operator-config** — the stranding case is closed (C01); the transaction can still lose
  rollback protection under concurrent use.
- **p04 storage-schema** — the raw store is a projection, not an archive (host fields dropped);
  reconciliation can discard new occurrences and attach wrong provenance.
- **p05 compaction-flow** — closed in the second pass above, except RS02 (reset/rebind is not
  fenced against in-flight compaction) and the two deferred items named there (MA01, MC01).
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
