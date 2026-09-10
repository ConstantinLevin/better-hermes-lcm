# Architecture

Hermes-LCM keeps raw messages in profile-local SQLite and builds a summary DAG so active context
stays bounded while history stays recoverable.

## Core flow

1. The active context engine ingests every message into `lcm.db`, verbatim.
2. Raw messages older than the protected fresh tail are compacted, one bounded chunk at a time,
   into leaf summary nodes (depth 0).
3. When the summary pile is worth enough, leaves are condensed into higher depths.
4. Active context = system prompt + rendered summaries + fresh raw tail.
5. Recall tools recover exact source rows, or bounded expanded context, when a summary is not
   enough.

Raw messages are source truth. Summary nodes, embeddings, temporal rollups, query views, and
assertions are derived, rebuildable layers carrying explicit provenance.

## Summaries are an index

A leaf covers a bounded span, and each rendered summary is labelled with its node id plus an
`[Expand for details: lcm_expand(node_id=N)]` handle. The summary exists so a reader can tell *what
is underneath it* and choose what to expand — not to stand in for it. The summarizer is instructed
to state only what its source states: hedges stay hedges, and attempted, claimed, and confirmed stay
distinct. It is still a summary, so an exact value comes from the expanded original.

Anything removed from what the summarizer read, or from what the agent reads back, leaves an
`[LCM …]` or `[Externalized …]` marker naming what was removed and how to recover it. A marker is
never itself trimmed away.

## Scope model

- Current-session DAG operations use the active engine/session binding. `lcm_expand(node_id=…)` and
  `lcm_describe` are current-session only.
- `lcm_recall` searches every conversation already stored in the local LCM database.
- `lcm_grep` with `session_scope='all'|'session'` is bounded, raw-message-only cross-session search.
- `lcm_load_session` enumerates one known LCM session.
- The host's `session_search` covers Hermes-tracked history outside `lcm.db`.

Do not silently treat those scopes as interchangeable.

## Derived state (default-off)

Same-database assertion and query-view state, plus provider-neutral reasoning and evidence
components, are all disabled by default and remain subordinate to raw messages:

- assertions require exact message IDs, spans, quotes, and lifecycle provenance;
- query views cache evidence dependencies and coverage, never final prose;
- computation validates exact operands and emits an immutable trace;
- evidence packs return bounded evidence, not an authoritative answer.

Unknown source/event time and unresolved conflict are valid states. Derived data fails closed rather
than manufacturing certainty.
