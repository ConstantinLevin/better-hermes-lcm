# Session lifecycle and rotate

Hermes `/new` starts a new host session; Hermes-LCM binds it to its own lifecycle row. By default
**nothing is carried over**: `new_session_retain_depth` is `0`, so a new session starts with no
earlier summaries in front of it. Setting it to `2` (or `-1`) restores carry-over of depth-2 and
higher nodes; carried summaries never rewrite source ownership, and expansion still comes from the
descendant raw messages.

`/new` deletes no LCM data. Earlier rows and nodes stay in `lcm.db` and stay reachable through
`lcm_recall`, `lcm_grep(session_scope='all')`, `lcm_load_session`, and `lcm_expand(store_id=…)`
unless an explicitly authorized cleanup removes them. Do not promise otherwise.

## `/lcm rotate`

Available only when `/lcm` slash commands are enabled. It differs from `/new`:

- it keeps the current `session_id` and `conversation_id`;
- `/lcm rotate` is a read-only preview; `/lcm rotate apply` mutates;
- apply writes the rolling rotate backup first;
- it preserves the resolved fresh tail (count- and token-bounded);
- it advances the lifecycle frontier past older raw messages so bootstrap does not replay them into
  active context;
- it deletes no raw source rows and calls no summarization model.

Run normal compaction before rotate when older material should be represented in summary nodes.
Even without a summary, pre-tail raw rows stay recoverable through `lcm_load_session` and
`lcm_expand`, and a rotate marker records the span.

Rotate refuses unbound, ignored, and stateless sessions, and reports the reason
(`no_active_session`, `session_ignored`, `session_stateless`, `no_pre_tail_content`,
`frontier_already_ahead`, …). Repeating an already-satisfied rotate is a no-op and deliberately does
not overwrite the previous known-good rolling backup.

Use a new session when the user wants a new conversational boundary. Use rotate when the problem is
active transcript size without a change of identity.
