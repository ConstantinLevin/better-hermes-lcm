# Recall tools

Use these when the answer depends on historical evidence that was compacted or lives in another
LCM session.

Two rules cut across every tool here:

- **No core retrieval tool trims its response to a character budget.** `limit` and
  `max_content_chars` are the caller's contract. Each `limit` still has its own documented cap; when
  it clamps, the response says so in `limit_clamped_from`.
- **Read the completeness fields.** `complete`, `incomplete_reason`, `truncated`, `has_more`,
  `degraded`, `coverage`. A bounded, capped, or failed scan is never reported as exhaustive.

## Current compacted conversation

Start from the summaries already in context: each is labelled with its node id and an expand hint.
Expanding a node you can see beats searching for what it contains.

### `lcm_describe`

Inexpensive inspection of a known current-session summary node or externalized payload reference:
token counts, child manifest, expand hints. With no handle it returns the current-session DAG
overview. Use `index_offset` to continue a node index block that did not fit. Planning step, not
discovery.

### `lcm_expand`

Drill-down from a known handle. Exactly one mode:

- `node_id` — the sources compacted into a current-session summary node, paged by
  `source_offset`/`source_limit`, `content_offset`, and `tool_calls_offset`;
- `store_id` — one raw message, and it works across LCM sessions;
- `externalized_ref` — a current-session payload, paged by `content_offset`.

`max_tokens` defaults to a window-weighted page (4000 at a 256k context, 32000 at 1M). Set
`include_exact_ref=true` with `store_id` when the slice will be cited or passed to
`lcm_evidence_pack`/`lcm_compile_evidence`/`lcm_compute`; the default is off for byte compatibility.

Not a first-step discovery tool.

### `lcm_grep`

Discovery across current-session raw messages and summary nodes at all depths.

- `query` is FTS5 text by default, not a regex. FTS5 combines terms with AND, so prefer 1-3
  distinctive terms or one quoted phrase.
- `sort='recency'` (default) for recent events, `'relevance'` for the strongest older match,
  `'hybrid'` when both matter.
- `mode='semantic'` or `'hybrid'` needs embeddings; degraded coverage is reported rather than hidden.
- `session_scope='all'|'session'` is explicit, bounded, **raw-message-only** archive recovery inside
  `lcm.db`; cross-session summary expansion is deferred, so drill in with `lcm_expand(store_id=…)`.
- A `role`, `time_from`, or `time_to` filter makes the search return raw messages only — no summary
  hits. `source` and `conversation_id` keep summaries.
- `content_scope='externalized'|'both'` opts into bounded search over active-session payload
  sidecars.
- `limit` defaults to 10, capped at 200.

A search snippet is not evidence for a detail-heavy answer. Expand the hit.

### `lcm_expand_query`

Use when current-session compacted material must be expanded *and synthesized* into a bounded
answer.

- `prompt` is required.
- Supply either a small `query` (same narrow FTS rules as `lcm_grep`) or explicit `node_ids`.
- Model-backed and bounded by answer and expansion-context token budgets; the context budget is
  window-weighted. A run that stopped at its generation limit says so.

## Cross-conversation memory

### `lcm_recall`

Semantic discovery across all conversations stored in the local LCM database. It fuses raw
full-text, summary-vector, and verbatim-chunk arms and degrades honestly when embeddings are
unavailable.

- `scope_bias` and recency are ranking boosts, never hard filters. For hard time bounds use
  `lcm_grep`.
- `include` selects all, summary, or verbatim hits. `limit` defaults to 8, capped at 25.
- `detail='snippets'` is the byte-compatible default; `detail='answer_ready'` adds bounded
  per-session diversity and exact-ref hydration.
- Follow each hit's `expand_hint`: verbatim/current-session hits use `lcm_expand`; cross-session
  summaries use `lcm_load_session`, because `lcm_expand(node_id=…)` is current-session only.

### `lcm_load_session`

Enumeration, not search, once a session ID is known: raw rows in chronological `store_id` pages.

- Continue with `after_store_id` from `next_cursor`.
- `limit` defaults to 100, capped at 200. `max_content_chars` defaults to 4000 and has **no** upper
  clamp — ask for what you need.
- A row cut at `max_content_chars` reports `content_truncated` and is recovered with
  `lcm_expand(store_id=…, content_offset=…)`.
- Set `include_exact_ref=true` when rows will feed exact evidence or computation.

Use the host's `session_search` for Hermes-tracked sessions that are not in `lcm.db`.

## Time-bounded recall

### `lcm_recent`

The front end for the temporal-rollup subsystem, which is **off by default**: it answers
`status: disabled` unless `temporal_rollups_enabled` is set. That is deliberate — its fallback
bypasses the summary index instead of using it.

For a time window, use `lcm_grep` with explicit `time_from`/`time_to`, or expand the summary nodes
covering that span.

## Exact evidence and computation

### `lcm_compile_evidence`

For a question needing multiple named facets, exact operands, conflict handling, or latest-state
reasoning. Supply one bounded semantic proposal over already-retrieved exact refs; it is only a
hint, and product code revalidates every ref, quote span, entity, date, value, unit, distinct key,
role, and source. When deterministic parsing yields only the catch-all `answer` facet, the proposal
may add up to 12 unique generic `requested_facets`; otherwise facets can only extend, never erase,
deterministic requirements. `answer_sufficient` closes a specifically answered open-ended question;
exhaustive counts, lists, or arithmetic need `finite_coverage` or `computation_sufficient`. On
`partial`, `conflicted`, or `unknown`, preserve the uncertainty.

### `lcm_evidence_pack`

Lower-level hydration once bounded baseline refs exist: validates exact refs, keeps occurrence and
observation time distinct, deduplicates, and may return a canonical computation trace. Evidence, not
prose; open cardinality stays partial unless coverage is product-verifiable.

### `lcm_compute`

Only over compiler-validated exact evidence: date intervals and filters, distinct counts,
compatible-unit sums, directed or absolute differences, ordering, latest-state selection. Invalid
spans, mixed units, ambiguity, or unsupported closure fail closed.

## Default-off tools

`lcm_query_state` (assertion sidecar) and `lcm_retrieve` (adaptive retrieval controller) both answer
`status: disabled` until their subsystem is enabled. Neither is needed for ordinary recall, and
neither should replace the workflow above without measured benefit.

## Operator tools

`lcm_status`, `lcm_inspect`, and `lcm_doctor` report health and metadata. They are not content
retrieval.
