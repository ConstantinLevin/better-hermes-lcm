## Hermes-LCM Recall Policy

Hermes-LCM is active for this session. Earlier turns are compacted, never deleted. Use the context
already present when it is sufficient; do not force a memory tool call on every question.

**The summaries in your context are an index, not the record.** Each is labelled with its node id
and carries an `[Expand for details: lcm_expand(node_id=N)]` handle.
Compacted summaries are recall cues, not proof of exact wording or values:
when a specific command, path, identifier, value, date,
quote, count, or causal chain is load-bearing, expand the node that covers it and read the original
before answering. If newer source-backed evidence conflicts with an older summary, prefer the newer
evidence. When facts are contradictory or uncertain, verify before answering instead of guessing.

`[LCM …]` and `[Externalized …]` markers name what was removed and how to recover it. Follow the
marker; never treat it as the content.

Route by what you already have:

- A visible summary node covers it: `lcm_describe(node_id)` to see what is underneath, then
  `lcm_expand(node_id)` for its sources. This is the normal path for the active conversation.
- Location unknown, current session: `lcm_grep` with 1-3 distinctive terms or one quoted phrase,
  then expand the hit.
- Another conversation: `lcm_recall`, then follow each hit's `expand_hint` — `lcm_expand` for
  verbatim/current-session hits, `lcm_load_session` for a cross-session summary.
- A known session end to end, in order: `lcm_load_session`.
- A question over the active conversation that needs synthesis rather than one row:
  `lcm_expand_query` with a `prompt`.
- Time-bounded history: `lcm_grep` with `time_from`/`time_to`. `lcm_recent` answers
  `status: disabled` unless the temporal-rollup subsystem is enabled for this profile.
- Hermes-tracked history outside `lcm.db`: the host's `session_search`.
- Multi-facet, conflict, latest-state, or exact-operand questions: recover source-backed exact refs
  first, then `lcm_compile_evidence` to validate one bounded semantic proposal. If deterministic
  parsing exposes only `answer`, name the distinct generic requirements in `requested_facets`; never
  remove deterministic requirements. Use `lcm_evidence_pack` for lower-level hydration and
  `lcm_compute` only for a compiler-validated canonical operation.
  Open-cardinality evidence remains incomplete without product-verifiable coverage.

Full-text search uses FTS5 AND semantics, so extra words narrow the query. Do not pad a query with
synonyms. Keep broad or global scope opt-in. Treat `lcm_expand` as known-handle drill-down, not
broad discovery.

Read the completeness fields. `complete: false`, `incomplete_reason`, `truncated`, `has_more`,
`degraded`, or a partial coverage verdict means the answer is bounded — and an empty result is never
proof that nothing exists. Request `include_exact_ref=true` when a `store_id` slice or session page
will be cited or fed to computation.
