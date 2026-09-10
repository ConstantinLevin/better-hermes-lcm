# Retrieval tools reference

The exact tool contracts. [`README.md`](../README.md) has the one-line summary of each tool and
the install/configuration ground; this page has the arguments, bounds and degradation behaviour.

Every number here was read out of `tools.py`, `retrieval_core.py`, `schemas.py` and `config.py`.

Five of the fifteen tools belong to upstream subsystems that this fork keeps default-off and does
not audit or modify (`lcm_query_state`, `lcm_compute`, `lcm_compile_evidence`, `lcm_evidence_pack`,
`lcm_retrieve`), as do `lcm_grep`'s `semantic`/`hybrid` modes, `lcm_recall`'s two vector arms, and
`lcm_recent`'s rollup path. They are described below because the schemas are exposed to the model
whether or not you enable them — not as a recommendation to switch them on. See
[Operator guide → Opt-in subsystems](operator-guide.md#opt-in-subsystems-exposed-off-and-out-of-scope).

## The tools

Recommended escalation:

- current compacted conversation: `lcm_grep` → `lcm_describe` → `lcm_expand_query`;
- cross-conversation memory: `lcm_recall` → the returned `lcm_load_session` or `lcm_expand` hint;
- recent / time-bounded recall: `lcm_recent`, or a time-bounded `lcm_grep`.

`lcm_expand` is known-handle drill-down, not broad first-step discovery. Use the host's
`session_search` for Hermes-tracked history that is not in `lcm.db`. The canonical runtime policy
is `skills/hermes-lcm/references/recall-policy.md`, injected only while LCM is the active context
engine.

| Tool | Contract |
|------|----------|
| `lcm_grep` | Search current-session raw messages and summaries. `mode='full_text'` is the default; `sort` is `recency` (default), `relevance` or `hybrid` and orders inside the full-text arm. `session_scope` is `current` (default), `all` or `session` (with `session_id`); broader scopes return raw-message hits only, in full-text mode only, and cannot search externalized payloads. `content_scope` is `history` (default), `externalized` or `both`. Raw-message filters `role`, `time_from`, `time_to`, `source` and `conversation_id` are pushed into the full-text query; when any is supplied, externalized results are omitted and `summary_results_omitted` is reported for the role/time filters. `limit` is clamped to 200. |
| `lcm_recall` | Search the whole local database across all conversations and all time. Three arms — full-text raw messages, summary-vector KNN, chunk-vector KNN — fused with weighted RRF, then a soft prior: `final_score = rank_score × (1 + scope_bias × is_current_conversation) × recency_boost`. `scope_bias` defaults to `0.5`; recency is `2**(-age/30d)` floored at `0.5`. Both are ranking boosts, never filters. `include` is `all`/`summaries`/`verbatim`; `detail` is `snippets`/`answer_ready`; `limit` defaults to 8, capped at 25. With embeddings disabled or the vector corpora empty it degrades to the full-text arm — including for `include='summaries'`, so a summaries request returns full-text hits rather than nothing. Each hit carries an `expand_hint`: an `lcm_expand(...)` handle for verbatim/current-session hits, an `lcm_load_session(...)` handle for cross-session summary hits (`lcm_expand`'s `node_id` mode is current-session only). |
| `lcm_query_state` | Opt-in (`LCM_ASSERTIONS_ENABLED`). Query the same-DB assertion sidecar by canonical subject, optional predicate/kind/scope/speaker, and optional as-of boundary. Returns typed lifecycle state with exact store ids, character spans, hashes and quotes; unresolved conflicts are preserved and recency alone is never treated as supersession. `limit` defaults to 25, capped at 50. Returns `status: disabled` when the flag is off. |
| `lcm_compute` | Reachable without a flag. Executes a question-derived, provider-neutral date/count/sum/difference/order/latest-state operation over exact raw spans or assertion ids. Values, units, labels, keys, dates, operand order, completeness and final wording are validated; unsupported or ambiguous inputs return an evidence-only fallback. |
| `lcm_compile_evidence` | Reachable without a flag. Turns one bounded semantic proposal into an exact-source-grounded evidence brief. The proposal may name facets and operands, but the code validates refs, quotes, spans, entities, dates, values, units, distinct keys, roles and sources; finite coverage is never accepted from the proposal alone. |
| `lcm_evidence_pack` | Reachable without a flag. Builds a bounded packet from baseline exact refs in the same `lcm.db`. Resolves only unique quote spans inside declared windows, validates facets and occurrence time, deduplicates refs, and may emit an immutable canonical computation trace. Returns no prose and never accepts caller-asserted open-cardinality completeness as proof. |
| `lcm_retrieve` | Opt-in (`LCM_ADAPTIVE_RETRIEVAL_ENABLED`). Coordinates one bounded retrieval episode inside the answerer turn: named evidence requirements close only against exact observed refs, and at most three calls to `lcm_recall`, `lcm_recent`, `lcm_query_state`, `lcm_load_session` or `lcm_expand` are allowed. Warm reuse validates exact positive dependencies and the corpus coverage watermark. Final prose is never cached. Returns `status: disabled` when the flag is off. |
| `lcm_recent` | Opt-in (`LCM_TEMPORAL_ROLLUPS_ENABLED`), and returns `status: disabled` when the flag is off — which is the default. It is the front end for the temporal-rollup subsystem; with rollups off it degraded to "fetch leaf summaries overlapping a time window", which bypasses the index instead of using it. The summaries exist to show where to expand: expand until you reach the leaf. `limit` defaults to 10, capped at 200. |
| `lcm_load_session` | Load one ordered raw-message transcript page for an explicit `session_id`. Not search: raw rows in `store_id` order, `limit` defaulting to 100 and capped at 200, per-message content bounded by `max_content_chars` (default 4,000, honoured as given — the fork removed the 20,000 clamp, since the argument is the caller's contract), continuing with `after_store_id` from `next_cursor`. Set `include_exact_ref=true` when rows feed exact citation or computation. |
| `lcm_describe` | Inspect the current-session DAG, or preview an `externalized_ref` without loading full content. |
| `lcm_expand` | Recover source messages, child summaries or externalized payloads with pagination. `node_id` mode is current-session only; `store_id` mode fetches a single raw message regardless of session, which is what a cross-session `lcm_grep` or `lcm_recall` hit hands you. `include_exact_ref=true` adds the exact returned slice without changing default bytes. |
| `lcm_expand_query` | Answer a question from expanded current-session context, returning a bounded answer. `context_max_tokens` bounds the material handed to the auxiliary model; it defaults to the curved `expansion_context_tokens` (32,000 at 256k → 125,000 at 1M). |
| `lcm_status` | Runtime health, context pressure, effective config with per-field sources, config-source warnings, source lineage and lifecycle stats. In this fork it also carries a `window_scaling` block: every curved setting, its resolved value, its source, and the `t` position between the anchors. |
| `lcm_inspect` | Read-only operator inventory for current-session lineage, message/frontier metadata, fresh tail, externalized refs and their readability, compaction skip/no-op reasons, and matched ignore/stateless patterns. It returns metadata only; use `lcm_load_session`/`lcm_expand` when you need content. `limit` defaults to 20, capped at 200. |
| `lcm_doctor` | Database, FTS, lifecycle, config and context-pressure diagnostics, each with a `guidance` entry classifying it as `safe/ignore`, `inspect` or `backup-first cleanup`. |

## Scope, filters and lineage

LCM retrieval defaults to current-session scope. `lcm_grep(session_scope='all'|'session')` is the
explicit opt-in for bounded archive search over rows already in `lcm.db` (raw-message hits only,
including externally backfilled rows carrying source strings such as `openclaw-lcm:*`). Once a
session id is known, `lcm_load_session` enumerates that session's raw transcript in chronological
pages without a query.

Within the current session, `source` filters raw rows directly and filters summary nodes by
descendant raw-message source lineage. `unknown` is a real source value, not a wildcard; legacy
blank-source rows are treated as `unknown`. `role`, `time_from` and `time_to` are raw-message
filters applied in the message search query before result limiting. `time_from`/`time_to` accept
Unix seconds or timezone-aware ISO 8601; naive ISO strings are rejected so the same query means
the same thing on every machine. When a raw-message filter is active, `lcm_grep` returns raw rows
only and reports `summary_results_omitted`.

Carried-over summary nodes can become current-session content after `/new` — note that this fork
defaults `LCM_NEW_SESSION_RETAIN_DEPTH` to `0`, so by default nothing is carried over. When it is,
source eligibility still comes from the descendant raw messages, and expanding a carried-over node
recovers those original sources even though they belong to the previous session.

## Bounded delivery, and what is not bounded

A retrieval call must not flood the main context, but a bound that silently shortens a result is
loss. In this fork the core retrieval tools have **no response character cap**: `limit` is the
caller's bound, and an oversized response is the host's spillover problem. The caps that remain
are `lcm_query_state` and `lcm_compute`, both default-off subsystem tools, and both scale with the
window (×1 at 256k → ×4 at 1M via `LCM_TOOL_RESPONSE_CHAR_SCALE`).

What is bounded, and how the bound is disclosed:

- `lcm_expand(node_id=…)` pages immediate sources with `source_offset`/`source_limit` and reports
  `next_source_offset`, `next_content_offset` and `has_more`. A page boundary is a cursor, not a
  cut.
- `lcm_expand(store_id=…)` and `lcm_expand(externalized_ref=…)` page content with
  `content_offset`, reporting `content_truncated`, `next_content_offset` and `has_more`.
- `lcm_load_session` pages ordered rows with `after_store_id`/`next_cursor`; each row carries
  bounded content plus truncation metadata, and a large individual row is recovered with
  `lcm_expand(store_id=…, content_offset=…)`.
- `lcm_expand_query` reports truncation/pagination hints when its `context_max_tokens` budget
  binds.
- Vector-arm coverage is reported as `full` (exact scan of the whole profile), `bounded` (the
  dependency-free bounded-scan fallback), `full_approx` (the two-stage binary-prescreen path
  reached the whole corpus but stage 1 kept only the closest `LCM_KNN_PRESCREEN_MULTIPLIER × k`
  candidates before the exact rescore, so the top-k is approximate) or `none`. `lcm_recall`
  repeats a `bounded` or `full_approx` arm in its `degraded_reason`, naming the arm, so a caller
  never mistakes a truncated or approximate arm's contribution for an exhaustive one.

## Full-text, semantic and hybrid modes

`mode='full_text'` is the default and needs no provider. `mode='semantic'` and `mode='hybrid'`
require the default-off embeddings subsystem and a backfilled corpus; with neither, `lcm_grep`
answers exactly as it does today.

`mode='semantic'` embeds the query with the provider's query path (preserving query/document input
asymmetry) and searches the current embedding profile with cosine KNN. Hits are summary nodes with
the normal `node_id`, depth, session, time-window, `expand_hint` and a 300-character snippet, plus
`score`/`cosine_score` and a `confidence`/`confidence_band`:

| Cosine score | Confidence |
|---:|---|
| `>= 0.65` | `high` |
| `>= 0.50` | `medium` |
| `>= 0.35` | `low` |
| `< 0.35` | `noise` |

`mode='hybrid'` runs both arms, deduplicates shared summary nodes by `node_id`, and fuses ranks
with unweighted reciprocal-rank fusion, `rrf_score = sum(1 / (60 + rank))`. Both arms inspect
`min(500, max(50, limit * 3))` candidates before the public limit is applied. Hits surface
`fts_rank` and/or `semantic_rank` plus `rrf_score`; semantic hits keep `semantic_score` and
`confidence`. No external reranker is called in either mode.

Semantic and hybrid requests run under one absolute wall-clock deadline (`embedding_query_timeout_s`,
3 s by default) started at `lcm_grep` entry, covering provider resolution, query embedding, the
optional NumPy import, KNN, hydration, FTS fallback, both hybrid arms and fusion. A
disabled/missing provider or a transient semantic failure degrades to the existing full-text
result when one is available, adding `degraded_to_fts: true`, `degraded_reason` and
`coverage: 'none'`. If the deadline is exhausted before any usable result exists, the tool returns
an explicit `timeout` error and starts no further fallback or arm. Hybrid may return FTS results
computed before its semantic arm timed out, starting no new I/O. Provider authentication failures
stay operator-visible rather than degrading silently.

Filters constrain which arm can serve. `source` uses the SQL-bounded candidate window first, then
verifies descendant lineage within it, so a source-filtered semantic result reports `bounded`
coverage rather than claiming pre-bound source coverage; the lineage walk has a hard work cap, and
exceeding it — or a legacy DB with no `messages.source` column — yields `unverifiable_provenance`
and **fails closed** rather than treating "cannot check" as "all allowed". `role`, `time_from`,
`time_to`, `conversation_id` and the broader `session_scope` values return raw-message hits only:
a summary node has no single role or lane and is cross-session, so the semantic arm degrades to
the raw full-text path (reporting `degraded_to_fts`) whenever one of them is supplied. The
semantic arm therefore produces summary hits only for a current-session query with no
role/time/conversation filter.

## Weighted RRF fusion (`lcm_recall`)

`lcm_recall` fuses its three arms with `rrf_score = sum(weight_arm / (60 + rank))`. Naive
equal-weight fusion measured **−21 R@5** against pure summary vectors on LongMemEval — the weak
FTS arm got equal say and pulled strong vector matches down — so the weights exist to stop the
fusion falling below its best arm.

Defaults are `fts=0.5, summary=1.0, chunk=1.0`, overridable with a lenient `arm=weight` list:

```bash
LCM_RECALL_ARM_WEIGHTS="fts=0.5,summary=1.0,chunk=1.0"
```

Unknown arm names, malformed pairs and non-finite/non-numeric weights are skipped; a negative
weight is rejected with a warning (it would invert RRF rank-monotonicity) and `0.0` cleanly drops
an arm. Any arm left unspecified keeps its default, so a wholly unparsable value degrades to the
defaults rather than erroring the tool. The weights actually applied are echoed under
`provenance.arm_weights`. `lcm_grep`'s hybrid RRF is unaffected and keeps implicit `1.0` weights.

Chunk-arm hits are merged onto the same `store_id` when FTS also surfaced it, keeping the
best-ranked chunk per store and taking the better-ranked arm's snippet **and** its offsets
together, so the preview and the expand handle describe the same span.

With `detail='answer_ready'`, reference-strict delivery (`LCM_RECALL_REFERENCE_STRICT`, on by
default) never delivers a hit that cannot carry a truthful `(store_id, char_start, char_end)`
span: the next-ranked citable hit takes its slot and the omission count is surfaced in
`provenance.answer_ready`.

## Searching externalized payloads

Opt-in, so the default `lcm_grep` cost and result shape are unchanged:

```text
lcm_grep(query="distinctive error", content_scope="externalized")
lcm_grep(query="distinctive error", content_scope="both",
         externalized_refs=["20260714_....json"])
```

Deliberately narrow:

- case-insensitive literal matching inside payload content; history keeps its FTS5 query behaviour;
- only payloads whose stored session id exactly matches the active session, so every returned ref
  stays recoverable with `lcm_expand`;
- at most 256 files scanned (from at most 4,096 discovered) and the first 512,000 encoded
  content-field bytes per file;
- each result carries the ref, tool-call id, bounded snippet, line and byte position, original
  content byte/character sizes, and `scan_truncated`; the scan bounds themselves are echoed under
  `externalized_scan`;
- `externalized_refs` is optional and capped at 256; invalid, missing, symlinked or
  foreign-session refs are rejected rather than silently widened.

There is no separate character cap on the externalized result material in this fork — `limit` is
the caller's contract, and a cap that shortened the list without saying so reported fewer matches
than it had found.

Cross-session externalized search remains unsupported: search the historical row first, load the
intended session, or recover a known current-session ref directly with `lcm_expand`.

## Importing a lossless-claw / OpenClaw archive

`scripts/import_lossless_claw.py` backfills raw message rows (and, with `--include-summaries`,
compatible source summaries) from a lossless-claw/OpenClaw SQLite `lcm.db` or from JSONL session
exports. Dry-run is the default, apply backs up an existing target first, and imported rows keep
explicit provenance in `session_id` and `source` (for example
`openclaw-lcm:agent:<agent>:<source-session>`). Reruns are idempotent for the same `--import-id`,
which is path-derived unless you pass one, so pass a stable id if the same database may be
imported from different paths. Changing `--agent`, `--namespace` or `--session-identity` under an
existing `--import-id` is treated as the same import and skips already-tracked messages; use a new
id for a different mapping. No OpenClaw config or secret tables are imported, but raw transcripts
and tool payloads are, and may contain sensitive data.

This is a local archive migration path. It does not make LCM a general memory provider and does
not change the current-session retrieval contract.

## Related references

- [README](../README.md)
- [Operator guide](operator-guide.md)
- [Benchmarking and stress checks](../benchmarks/README.md)
