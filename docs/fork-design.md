# hermes-lcm fork: 1M-context done right, weighted smoothly from ~256k

## Context

Hermes (v0.21.0, `~/.hermes/hermes-agent`) runs the `hermes-lcm` context engine
(`~/.hermes/plugins/hermes-lcm`, v1.0.0-rc.1, clean git checkout at `8d1b1e6`) on a
1,000,000-token model. Operator rule: **no loss**. A summary is acceptable only if its
provenance is intact AND it still hints at what is beneath it well enough that a future reader
knows which node to expand. Loss = provenance gone, or a summary/stub too thin to act as an
index.

LCM's sizes are absolute tokens tuned for ~128k-272k windows. At 1M they fire on counters
instead of pressure, drain far too slowly, or silently truncate. This fork makes every
*tuning* value a **smooth function of `context_length`** — equal to upstream at 256k, equal to
the 1M design at 1M, linearly weighted between — and applies the *pure* optimizations
(no-loss, throughput, index quality) identically at every window. Hard constraint: nothing
degrades behaviour for smaller-window models; at 256k with default config the resolved values
and the resulting DAG are upstream's. Three adversarial reviews were run against this plan;
every surviving finding is folded in below and the earlier wrong claims are corrected.

### Verified facts (read from source; anchors in Mechanics)
- **Upstream default path** (`dynamic_leaf_chunk_enabled=False`, `threshold_full_sweep_enabled=False`):
  ONE leaf pass per `compress()` (`base_max_leaf_passes = 4 if dynamic else 1`,
  `compaction.py:764`), and that pass summarises the **entire backlog outside the fresh tail
  into one node** (`to_compact = candidate_raw`, `:697`) under the 12k output cap. At a 256k
  window that is a ~150k -> 12k leaf. Condensation then fires on `len(uncondensed) >= fanin`
  (`engine.py:5588-5590`) — **no token gate**; `summary_prefix_target_tokens` is read only by
  the sweep (`compaction.py:735-741`).
- Compaction fires once per threshold turn (host breaks its re-invoke loop the moment
  `should_compress` is false, `turn_context_compaction.py:411`). Trigger counts are
  usage-anchored and **include system prompt + tool schemas**; the host passes that number as
  `current_tokens`, so the engine's `estimated_active_tokens` is in the same wire units.
- Sweep (flag on): drains all eligible raw, 12 passes / 120 s TOTAL (`compaction.py:29-30`),
  serial; then condenses while `frontier > summary_prefix_target_tokens` (0 -> `leaf_chunk_tokens`).
  Six behaviours hang off the flag: preflight `allow_partial_leaf`, the pass budget shared
  with condensation, deadline injection, partial-failure tolerance, the telemetry dict, and
  chunk selection. It cannot be replaced by a single parameter.
- Leaf budget `min(max(2000, 0.20*src), 12000)` (soft; `max_tokens`=2x; accept if shorter);
  condensation `max(1000, 0.40*src)`, uncapped. Both hardcoded.
- **Pre-summariser truncation inside LCM**: `_serialize_messages` cuts every message over
  3000 chars to head 2000 + tail 800 and tool-call args over 500 to 400
  (`engine.py:5183-5223`); unmatched assistant `tool_calls` are dropped (`:5192-5195`).
- Escalation L1 -> L2 (bullets, 50%) -> L3 truncation to 512 tokens; spend guard 24/600 s ->
  1800 s of L3, `clear()` unwired; escalation level not persisted; `_call_llm_for_summary`
  swallows `AuxiliaryExplicitCancellation` (`escalation.py:270-272`).
- Both condensation selectors take the lowest depth first (= newest summaries).
- `_extract_expand_hint` keeps the first line only; `expand_hint` is rendered inline and
  emitted as a flat JSON field in five tool paths — it must stay single-line.
- Stubbing: `_maybe_stub_active_tool_result` at ingest on new messages (no tail check) and
  `_stub_large_tool_results_for_active_replay` at assembly (tail-protected). Stub text is
  metadata only.
- Leaf chunk boundaries are token-greedy; a split call/result pair loses the call and
  orphans the result.
- Assembly drops silently: `get_uncondensed_at_depth(limit=100)` per depth
  (`engine.py:6110`, `dag.py:529`); `get_session_nodes(limit=1000)` at `engine.py:6103`
  cuts the oldest depths. `/new` **deletes** d0/d1 nodes (`new_session_retain_depth=2`,
  `engine.py:3575-3587`). `rotate_active_session` advances the frontier past
  un-summarised raw (`engine.py:6483-6513`).
- **Host on a plugin-engine exception: the exception is re-raised and kills the turn**
  (`conversation_compression.py:3439-3444`; no `except` at `turn_context_compaction.py:392`,
  `turn_preflight.py:144,304`). Cooldown/warning exist only for `ContextCompressor`
  (`_capture_authoritative_cooldown_under_lease` -> `(None, None)` for others;
  `_automatic_compression_gate_blocks` reads `type(compressor)._automatic_compression_blocked`;
  `get_active_compression_failure_cooldown` via `getattr(..., lambda: None)`). LCM defines none.
- **Host watchdog**: `compression.context_timeout_seconds` (120) is an *inactivity* budget
  whose only activity signal is `commit_fence.touch_progress` via a **thread-local** aux hook
  installed on the compaction worker thread (`auxiliary_client.py:290,399-402`;
  `conversation_compression.py:2660,2692`). `context_total_ceiling_seconds` is a hard 600 s
  wall (`:900-902`, clamped `max(600, idle)`). On abort the host discards the returned
  context, does not roll back the engine's DB, and retries on a new worker while the
  orphaned one keeps writing (`:1110-1113`).
- Host pre-engine cuts: terminal output at `tool_output.max_bytes` (50k) **is spilled to a
  readable file** (recoverable); `execute_code` has its own `MAX_STDOUT_BYTES` with a spill;
  `file_read_max_chars` 100k (file on disk); `web.extract_char_limit` 15k (stored);
  `_INLINE_SHELL_MAX_OUTPUT` 4000 (`agent/skill_preprocessing.py:19`) and the MCP 2 MB cap
  are hardcoded and **unrecoverable**.
- `hermes plugins update` autostashes a dirty tree with `reset --hard` and re-resets on
  conflict (`hermes_cli/plugins_cmd.py:1861-1927`); refuses `pinned: true` checkouts (`:776`).
- Sidecar hazard: `summary_nodes` rows decode positionally, the v5 shape classifier fails
  closed on unregistered columns, rollup trigger SQL is byte-compared; node deletes are
  `DELETE FROM summary_nodes` only (`dag.py:326-330`).

## Design

### Two classes of change
**Weighted tuning** — preferences about how to spend a window. Anchors: `low` = upstream's
value at `W_low = 262,144`; `high` = the 1M design at `W_high = 1,000,000`.
`t = clamp((W - W_low)/(W_high - W_low), 0, 1)`; `value(W) = low + t*(high - low)` (ints
rounded; fractions applied to W after interpolation). Anchors in one table
(`WINDOW_SCALED_DEFAULTS`, `config.py`); `W_low`/`W_high` configurable. An explicit env/config
override of a setting wins over its curve; the curve replaces the *default* only. Resolved in
`engine._set_context_length` from the capped `self.context_length`, stored as `effective_*`
with `_source`; `lcm_status`/doctor print fraction, resolved value, source.

**Pure optimizations** — identical at every window: remove loss, add throughput, improve
index quality. At 256k with default config the DAG must be identical to upstream's.

### Weighted tuning table (low @256k = upstream; high @1M)

| setting | low (upstream) | high (1M) | note |
|---|---|---|---|
| `context_threshold` default | 0.35 | 0.80 | `lcm.context_threshold` / `compression.threshold` / env still win |
| non-sweep drain stop (fraction of W, wire units) | = threshold (stop once under) | 0.30 | only the `estimated < threshold` comparison in the non-sweep loop changes; sweep flag untouched |
| leaf chunk size (fraction of W) | 1.0 (= whole backlog in one node) | 0.04 (=40k) | explicit `dynamic_leaf_chunk_enabled=true` keeps upstream 20k->40k doubling as an override |
| leaf pass cap per `compress()` | 1 | 64 | |
| leaf-loop time budget (s) | 120 (sweep only today; non-sweep unbounded) | 200 | applies to both paths |
| `summary_timeout_ms` | 60000 (`config.py:600`) | 200000 | this box overrides 120000 via env |
| `expansion_timeout_ms` | 120000 | 200000 | |
| fresh tail count cap | 32 | 400 | remains a CAP (upstream semantics; a floor cannot materialise: `_get_session_fresh_tail` loads `min(total, count)` rows) |
| fresh tail token cap (fraction of W) | off (0) | 0.15 | enabled only for t>0; whichever cap binds first |
| condensation trigger | `count >= fanin` | `count >= fanin AND frontier > 0.20*W` | rule: `count >= fanin AND frontier > t*0.20*W` — at t=0 the budget term is 0, i.e. upstream exactly |
| sweep condensation target (flag on) | `summary_prefix_target_tokens` or `leaf_chunk_tokens` | 0.20*W | flag-on path only |
| depth cap `incremental_max_depth` | 3 | 5 | |
| `summary_concurrency` | 1 | 6 | infrastructure pure, degree weighted |
| spend guard `max_calls` | 24 | 120 | tripping is a cooldown, never truncation |
| breaker `failure_threshold` | 2 | 4 | |
| `l2_budget_ratio` | 0.50 | 0.80 | |
| serialized message cap (chars) | 3000 (2000+800); args 500->400 | whole message | marked elision whenever a cut still happens |
| stub threshold / expansion context (fraction of W) | 0.10 / 0.125 | same | constant fractions (=25k / 32k at 256k) |
| `lcm_expand` page `max_tokens` | 4000 | 32000 | |
| tool response char caps | 20k inspect/recent, 64k grep/recall | x4 | one weighted cap |
| SQLite `cache_size`, `journal_size_limit`; `tokens.py` lru size | defaults / 2048 | weighted | `synchronous=FULL` stays |

Not curved: `protect_last_n` stays = configured `fresh_tail_count` (32). It is a host
preflight gate (`turn_context.py:288-298` skips the anchored estimate when
`len(messages) <= protect_first_n + protect_last_n + 1`) and feeds host-native compression of
bypassed sessions (`bypass.py:159,205,402`); curving it would widen the host's blind spot.

Worked values at 512k (t=0.34): threshold 0.50, drain stop 0.43, chunk 0.67*W (still whole
backlog), passes 22, tail 157 msgs / 61k, condense when `count>=4 and frontier>21k`,
depth 4, concurrency 3, guard 57, breaker 3, L2 0.60, serialize cap ~ (3000 + 0.34*(rest)).

## Pure optimizations (every window)

### A. No silent loss — and no turn-killing
1. **Delete L3** (`escalation.py:671-673`). L1 -> L2 -> raise `SummaryUnavailableError`.
   `_deterministic_truncate` and its marker are removed. Re-raise
   `AuxiliaryExplicitCancellation` (and `BaseException`) from `_call_llm_for_summary`
   (`:270-272`) so the host's fence rollback path (`conversation_compression.py:3415-3438`)
   still runs on `/stop`.
2. **Fail loud without killing the turn.** `compress()` catches `SummaryUnavailableError`,
   persists the contiguous prefix of finished passes, records a failure cooldown, and
   **returns the input list object unchanged** (identity matters: the host treats a new list
   as "compressed", `turn_context_compaction.py:462`, and compares progress with a different
   estimator, `turn_context.py:250`). Implement the host's cooldown protocol on `LCMEngine`:
   `get_active_compression_failure_cooldown()` (dict with `remaining_seconds`, read via
   `getattr` at `turn_context_compaction.py:172,275`, `turn_preflight.py:86`) and
   `_automatic_compression_blocked()` (read via `type(compressor)`,
   `conversation_compression.py:1313-1315`) so `_blocked_compress_reason` yields
   `cooldown:<s>` and the host prints its FAILURE-class warning. Manual `/compress`
   (`force=True`) clears the cooldown and `SummarySpendGuard.clear()`.
3. **Guard and breaker end in that cooldown**, never `None`/L3 (`escalation.py:329-377`).
4. **Escalation level + index block persisted** in the sidecar and rendered
   (`[L2 bullet summary]`, `engine.py:6113-6121`).
5. **Every remaining drop is marked**: serialize elisions (`[LCM elided N chars; store_id=…]`)
   and orphan `tool_calls` serialized, not dropped; assembly-cap tail skips and prefix-selector
   skips (`engine.py:6058-6070`, `:6126-6135`); bypass byte-truncation (`bypass.py:259-297`);
   externalized stubs carry `head=<~300 chars>` (`externalize.py:896-909`).
6. **Assembly renders the whole frontier**: per-depth limit derived from the summary budget
   instead of `limit=100`; depth discovery via `SELECT DISTINCT depth`.
7. **`/new` never deletes index nodes**: `new_session_retain_depth` becomes a render filter;
   the `DELETE` at `engine.py:3575-3587` goes. **Rotate** writes a marker node (or
   compact-then-rotate) so rotated raw stays hinted.
8. **Below-threshold cleanup calls do not summarise.** A `compress()` requested by the replay-
   diff branch of `should_compress_preflight` while under the threshold (and not force-overflow)
   runs the cleanup preamble — scaffold drop, ignored-message and dependent-reply drops
   (`compaction.py:548-647`, `_drop_preexisting_generated_ignored_dependent_eof_replies`) —
   publishes, and returns; no leaf pass, no condensation. **Deliberate deviation from
   upstream**, confined to non-default configs (ignore patterns / externalization on); with
   default config replay equals messages and the branch is never taken. Returns the input
   object when nothing changed.

### B. Throughput
9. **Three-phase leaf loop** (`compaction.py` ~530-900): sequential preamble; concurrent
   `_summarize_leaf_chunk_with_rescue` only; sequential chronological persist. Structure and
   shared-state hazards: Mechanics. Applies to both the non-sweep and sweep paths; the sweep
   flag's six behaviours are preserved verbatim.
10. **Watchdog-safe workers**: capture the compaction thread's `_aux_progress.hook`
    (`auxiliary_client.py:290`) and install it in every worker (pattern `:449-465`) so each
    summariser call ticks `commit_fence.touch_progress`; otherwise the host aborts the batch
    at the idle budget.
11. **Per-DB compaction lease**: a durable mutex (reuse the lifecycle lease machinery,
    cf. `try_acquire_rollup_operator_lease`) so an orphaned worker after a host abort cannot
    write concurrently with the host's retry worker. `compress()` returns before the idle
    budget with certainty (deadline-aware batch dispatch).
12. **Tool-group-aligned chunk boundaries** in the pre-slicer and `_next_leaf_rescue_chunk`
    (`engine.py:1660-1673`), mirroring `fresh_tail.py:29-49`.
13. **Hot paths** (results identical): projected `_summary_frontier_tokens()` + incremental
    delta (`engine.py:5687-5699`); hoist `_load_generated_ignored_placeholder_hashes()` and
    batch `store.get` via `store.get_batch` (`compaction.py:589`, `engine.py:4234-4253`);
    running total in the prefix selector. Sweep constants -> config; `tools.py:6505-6506`
    reads config; operator-guide "120 seconds between calls" corrected to "total".

### C. Condensation
14. **Interpolated trigger** (table): `_maybe_condense` condenses while
    `len(uncondensed) >= fanin AND frontier > t*0.20*W`. Sweep-flag condensation keeps its
    target rule with the target curved. Fallback pair-condensation past the cap
    (`engine.py:5716-5722`) is **left as is** — gating it on the (default-disabled) pressure
    ratio would stop the sweep reaching its target and break `test_lcm_engine.py:11725`.
15. **Age-ordered selection** shared by both paths (`engine.py:5588-5609`, `:5701-5722`):
    oldest frontier node by `earliest_at`, group up to `fanin` same-depth nodes from it, skip
    depths at the cap, else next-oldest eligible.
16. Ratios/caps -> config with unchanged defaults (`engine.py:1694-1695`, `:5638`).

### D. Index contract
17. **Prompts** (`escalation.py:412-513`): the summary is an index into recoverable
    provenance; the failure is an item the reader could not discover, not length. Coverage
    list in the source's exact terms: decisions + rationale; approaches rejected + why;
    constraints/preferences; files/paths/commands/identifiers/URLs/versions/values; errors and
    resolution; what informative tool outputs contained; end state and open items; every topic
    touched, one clause each. Depth >=1: merge child indexes keeping coverage of every child,
    preserve child order, never "various/etc.". "Exceed the target rather than omit an
    item." Depth prompts' drop-process-detail stays with the explicit rejection clause. L2:
    same list in bullets.
18. **Index block capture**: the whole "Expand for details about:" block (bounded ~400 tokens)
    stored as `index_block` in the sidecar and rendered under the node header.
    `SummaryNode.expand_hint` stays first-line (it is rendered inline and emitted as a flat
    JSON field in `tools.py:1375/1519/3292/5301/5706`, `adaptive_retrieval.py:553/605`).
19. **System note** (`_append_lcm_note_to_content`): summaries are indexes; absence from
    visible context is not absence from history; stubs and `[LCM elided …]` markers expand by
    store id; `lcm_grep` before assuming.
20. **Recovery path**: `lcm_expand` node mode gets a hydration flag (`tools.py:1190/1240/1244`)
    and the weighted page default; depth labels past 2 (`engine.py:6113-6117`); empty hint
    falls back to `lcm_expand(node_id=N)` text.
21. **`lcm_doctor coverage`**: per node, extract index-bearing entities from its sources
    (paths, identifiers, quoted strings, numbers, decision/rejection keywords) and report the
    fraction present in summary + index block. The executable definition of "no loss".

### E. Suggester
22. `suggest_preset_for_engine` (`presets.py:410-424`, `>=200k` branch at `:414`) selects on
    window alone. Add a first branch `W >= 512k -> None, "window-scaled defaults active (t=…)"`;
    branches below 512k and their tests untouched. Presets document the two anchors.

## Mechanics (verified)

### Test runner
```
cd ~/.hermes/plugins/hermes-lcm && /home/agent/.hermes/bin/uv run --no-project \
  --python /home/agent/.hermes/hermes-agent/venv/bin/python --with pytest --with numpy \
  python -m pytest tests/ -q
```

### Tests that pin behaviour this plan changes
- L3: `tests/test_lcm_core.py:1386-1416`, `:4436`, `:4441` -> raise; spend->L3 `:569` ->
  raise + zero calls; `:480`, `:516`, `:528`, `:542` stay.
- `summary_timeout_ms == 60_000` pinned at `test_lcm_engine.py:~20737` — low anchor is 60000.
- Condensation `:11298` (4 nodes of 10 tokens) and `:20695` (3 of 100) call `_maybe_condense`
  directly with no window set -> t=0 -> count rule -> pass unchanged; `:11725` (sweep target,
  beyond preferred depth) stays green because the fallback group is untouched.
- Sweep: `:11669` (`allow_partial_leaf`), `:11718` (`total_passes == 12` shared budget),
  `:11780`, `:11864`, `:11823` (partial failure), `:11609` (`leaf_passes == len(calls)` ->
  persisted count), `tests/test_threshold_full_sweep_benchmark.py:18` — flag semantics kept.
- Fresh tail `tests/test_fresh_tail.py:28-150` unchanged (both bounds stay caps).
- `summary_prefix_target_tokens` `test_lcm_core.py:720`, `:798` stay.
- Presets `tests/test_lcm_preset_command.py`, `test_lcm_engine.py:25762` untouched (<512k).

### Sidecar, not a core column
`summary_nodes` rows decode positionally (`dag.py:860-876`), the v5 classifier fails closed on
unregistered columns (`db_bootstrap.py:281-285`, `:461-471`), trigger SQL is byte-compared
(`:975-977`, `:1265-1300`). Feature table `lcm_node_meta(node_id PK, level INT DEFAULT 1,
index_block TEXT DEFAULT '')`, prefix `lcm_node` in `_KNOWN_FEATURE_TABLE_PREFIXES`
(`:313-326`), created under a named migration marker. Written at the node write
(`compaction.py:790-800`) and in `_condense_summary_nodes`; read in one `IN (...)` query by
`_assemble_context`. Add cascade deletes in `delete_nodes`/`_delete_nodes_batched`
(`dag.py:326-330`) so rows do not orphan. Downgrade is one-way: upstream classifies a DB with
`lcm_node_meta` as `VERSION_MISMATCH_GENUINELY_NEWER` — document. Verify migration against a
copy of a pre-existing DB.

### Config plumbing
Scalar field = dataclass field (`config.py:446+`) + one `_EnvFieldSpec` (`:307-411`) + one
status line (`tools.py:6491-6515`). New fields: `scale_low_window`, `scale_high_window`,
`drain_stop_fraction`, `leaf_chunk_fraction`, `leaf_pass_cap`, `leaf_loop_max_seconds`,
`fresh_tail_fraction`, `summary_budget_fraction`, `stub_threshold_fraction`,
`expansion_context_fraction`, `serialize_message_max_chars` (+head/tail/args),
`summary_concurrency`, `leaf_summary_ratio/min/max`, `condensation_ratio/min`,
`expand_page_tokens`, `tool_response_char_cap`, `sqlite_cache_kib`, `token_cache_size`.
Presets carry only `_PRESET_FIELDS` (`presets.py:39-48`) mirrored in `config.py:434-441`.

### Where the curve resolves
`engine._set_context_length` (`engine.py:922-970`) — add `_resolve_window_scaled_settings()`
on BOTH return paths (the `<= 0` branch must reset `effective_*`). Resolve from
`self.context_length` (post route cap); derive the drain stop from `context_threshold * W`,
not from `threshold_tokens` (post-min with the assembly cap, `:906-919`). Re-runs on every
`update_model` (`agent_init.py:1834`, `:2035`) / `on_session_start`. `clone_for_agent()`
(`engine.py:671-683`, via `__deepcopy__`) must allocate the new executor, locks, lease and
sidecar handle — a copy failure silently downgrades the agent to the built-in compressor
(`agent_init.py:1778-1787`). Delegation children build their own agent -> own copy -> own
`update_model(child window)` -> own curve; auxiliary children early-return in `update_model`
(`engine.py:4052-4058`) and keep the parent's curve. Consumers read config lazily per call
(`engine.py:2034-2054`, `:2103`, `:1660`, `compaction.py:735-741`, `:325-333`) — point them
at `effective_*` with config fallback when `context_length == 0`.

### Parallel leaf loop — required structure
Pre-slicing is order-equivalent: the compacted chunk is *deleted* from the front of
`working_messages` (`compaction.py:802`), never spliced; the fresh-tail identity set is
invariant across passes.
1. **Preamble (compaction thread):** scaffold drop (`:548-562`); ignore/dependent-reply
   filter (`:564-647`) once; `dependent_reply_message_ids` for the span; store-id map
   (`_get_store_id_map_for_messages`, `:565-567`) computed **per pre-sliced chunk in
   chronological order** to preserve upstream's placeholder identity/ordinal counts (it
   clobbers `_current_compress_placeholder_identity_counts`; a single whole-span call differs
   under ignore patterns); `focus_topic` per chunk when the tail holds fewer than
   `_AUTO_FOCUS_MAX_TURNS` user turns, else once (`engine.py:6390-6405`); pre-slice by
   iterating the sizing rule on the residual span (budget from `pressure_messages`, fill from
   `working_messages`, `:664-670`) with tool-group alignment; deadline check before dispatch.
2. **Concurrent:** only `_summarize_leaf_chunk_with_rescue` (`engine.py:1675-1741`), bounded
   by `summary_concurrency`, shared deadline (per-call cap `:1699-1703`), aux progress hook
   installed per worker (item 10). Breaker/guard are lock-protected. Duplicate externalized
   payload files from concurrent `_serialize_messages` are tolerable.
   `_run_pre_compaction_extraction` / `_schedule_pre_compaction_assertions` stay out.
3. **Persist (compaction thread, chronological):** per node in today's order (`:773-800`):
   `source_store_ids`, `store.get_time_bounds`, strictly increasing `created_at`
   (`dag.py:545` has no tiebreak), `dag.add_node` + sidecar row, rollup invalidation,
   `_last_compacted_store_id` (bare assignment `:799`), `_persist_frontier_marker`. All DAG/
   store writes on this thread under the compaction lease. Partial failure/deadline: persist
   the contiguous chronological prefix only; `leaf_passes += persisted`; recompute
   `raw_prefix_drained` / frontier tokens once after the batch; `estimated_active_tokens`
   applied as the batch sum but incremental where it is the stop condition. GC/debt hooks
   (`_maybe_gc_compacted_tool_results`, `_refresh_raw_backlog_debt`) are inert by default.
`tests/test_async_background_compaction_design.py` (xfail RED) supplies batch-validation
rejection reasons to reuse.

## Implementation sequence (each step leaves the suite green)
1. **Config scalars + anchor table + curve function**, no consumers.
   `tests/test_window_scaled_settings.py::test_curve_matrix[256k,384k,512k,768k,1M]`,
   `::test_at_256k_equals_upstream` (every value incl. `summary_timeout_ms == 60000`).
2. **`_resolve_window_scaled_settings()` in `_set_context_length`** (both paths; capped W;
   `clone_for_agent` allocations; status/doctor rendering). `protect_last_n` NOT curved.
   `::test_cleared_context_length_resets_effective_values`, `::test_uses_capped_window`,
   `::test_clone_allocates_runtime_helpers`.
3. **Consumers switch to `effective_*`** (tail caps, chunk size, pass cap, time budget,
   timeouts, guard/breaker construction, depth cap, L2 ratio, serialize caps, expand page,
   response caps, SQLite/lru sizes). Existing suites pass unmodified at 256k.
4. **Delete L3; cooldown protocol on `LCMEngine`; guard/breaker -> cooldown; cancellation
   re-raised; `clear()` wired.** Update `test_lcm_core.py:1386-1416`, `:4436`, `:4441`, `:569`;
   new `test_compress_returns_input_identity_and_arms_cooldown_on_summariser_failure`,
   `test_host_blocked_reason_reads_lcm_cooldown`, `test_cancellation_propagates`; failure
   injection asserts no L3-shaped text anywhere in the DAG and the turn is not killed.
5. **No unmarked loss anywhere** (items 5-8): serialize elision + orphan tool_calls; assembly
   limits + DISTINCT depth + selector marker; `/new` render filter; rotate marker; bypass and
   cap markers; stub `head=`; below-threshold cleanup without leaf pass (input identity when
   unchanged). Tests: `test_serialize_marks_elision_with_store_id`,
   `test_serialize_at_256k_matches_upstream_literals`, `test_assembly_renders_all_frontier_nodes`,
   `test_new_session_keeps_index_nodes`, `test_rotate_leaves_marker_node`,
   `test_cleanup_request_below_threshold_runs_no_leaf_pass_and_keeps_ignore_drops`.
6. **`lcm_node_meta` sidecar** with cascade delete. Migration idempotent; v5 classifier
   accepts; roundtrip; L2 header; bootstrap against a copied pre-existing DB.
7. **Prompts, index block, system note, recovery path** (items 17-20).
   `tests/test_index_contract.py`, `test_expand_node_hydrates_externalized_when_asked`,
   `test_expand_hint_stays_single_line`.
8. **Weighted chunking + drain stop + pass cap + time budget on the non-sweep path; sweep
   flag untouched.** `test_non_sweep_at_256k_is_one_whole_backlog_pass`,
   `test_non_sweep_at_1m_drains_to_stop_fraction`; sweep tests green.
9. **Condensation: interpolated trigger, age-ordered shared selector, ratios to config.**
   `test_condense_at_256k_uses_count_rule`, `test_condense_at_1m_requires_budget`,
   `test_condense_selects_oldest_frontier_node`; `:11298`, `:11725`, `:20695` unchanged.
9a. **Hot paths + SQLite/lru weighting** (item 13). `test_frontier_tokens_projection_equals_full_decode`.
10. **Three-phase loop, concurrency pinned at 1, tool-group-aligned slicing, compaction
    lease.** Update `:11609`, `:11823`; `tests/test_parallel_leaf_sweep.py::
    {test_serial_dag_identical_to_upstream_baseline, test_store_id_map_per_chunk_matches_per_pass,
    test_partial_failure_persists_contiguous_prefix, test_tool_call_result_pair_not_split_at_boundary,
    test_second_worker_blocked_by_lease}`.
11. **Enable `summary_concurrency > 1` with the aux progress hook per worker.**
    `test_concurrency_6_dag_identical_to_concurrency_1` (>=20 iterations),
    `test_workers_tick_commit_fence_progress`, `test_extraction_and_assertions_stay_sequential`.
12. **Suggester branch, `lcm_doctor coverage`, presets as anchor docs.**

Highest-risk: 4 (host protocol — verify by driving a real `build_turn_context` with an
injected failure: turn survives, warning printed), 10/11 (DAG identity, watchdog survival at
`context_timeout_seconds=120` with a 6-call batch), 6 (classifier fail-closed on a real DB),
8 (256k must remain one whole-backlog pass).

## Deployment on this box after the fork
- **Commit the fork and pin it**: `hermes plugins update` runs `_autostash_dirty_tree` with
  `reset --hard`; a dirty fork survives only as `stash@{0}`. Set `pinned: true` for the
  plugin in `~/.hermes/plugins/.install-metadata` (update refuses pinned checkouts) and point
  the remote at the fork.
- With the curve, W=1M resolves threshold 0.80, drain stop 0.30, chunk 40k, tail 400/150k,
  budget 200k, concurrency 6, depth 5, timeouts 200 s — remove the now-redundant explicit
  values from `~/.hermes/.env` and `lcm.context_threshold` from `config.yaml`, or keep them
  as deliberate overrides (`lcm_status` shows the source).
- Host: `compression.context_timeout_seconds: 200` AND `compression.context_total_ceiling_seconds:
  900` (raising idle does not raise the hard 600 s ceiling); `compression.threshold` aligned
  (dead config); `tool_output.max_bytes: 400000`, `tool_output.max_line_length: 8000`,
  `file_read_max_chars: 400000`; note `execute_code`'s own `MAX_STDOUT_BYTES` and the
  hardcoded `_INLINE_SHELL_MAX_OUTPUT` (4000) / MCP 2 MB caps are unrecoverable host cuts
  outside this fork (Hermes PR: scale like `_dynamic_context_file_max_chars`).
  `auxiliary.compression` = the operator's summariser (not specified here). Leave deferred
  maintenance off and both assembly caps 0.
- Restart the gateway; confirm via `lcm_status` (resolved `effective_*` with sources).

## Verification
1. Suite green after every step; steps 4/8/9/10 update the named tests.
2. Curve matrix at {256k, 384k, 512k, 768k, 1M}; at 256k every value equals upstream's literal
   and a fixture session produces a DAG identical to upstream's (one whole-backlog leaf per
   threshold turn, count-triggered condensation).
3. Synthetic session past threshold at 1M: one compaction per threshold turn; drains to
   ~0.30*W in one `compress()` under 200 s with concurrency 6; no condensation until the pile
   exceeds 200k; oldest-first when it does; no node above depth 5; concurrency 6 DAG identical
   to concurrency 1.
4. Failure injection through a real `build_turn_context`: summariser raising -> no node
   written, raw intact, input list identity returned, cooldown armed, host warning printed,
   turn completes; no truncation text anywhere in the DAG.
5. Host watchdog: a 6-call batch at `context_timeout_seconds=120` survives (progress ticks).
6. `lcm_doctor coverage` >= agreed floor on fixture transcripts.
7. Live on this box: one real session to the trigger; read leaves and index blocks; one
   prefix rewrite per threshold turn; `lcm_doctor coverage`.

## Do not
Enable deferred maintenance; set either assembly cap; set `incremental_max_depth=0`; enable
sensitive-pattern redaction (explicitly not lossless); curve `protect_last_n`; gate the
sweep's fallback pair-condensation on the pressure ratio; leave the fork checkout dirty.

## Implementation notes (deviations from the plan text, by step)
### Step 5 — no unmarked loss
- Serialize elision markers do not carry a `store_id` (the map is not available inside
  `_serialize_messages`); they carry sizes and point at the node's own sources (`lcm_expand`
  on the node), which is the provenance path anyway.
- The externalized stub keeps upstream's placeholder string byte-for-byte (the
  `_EXTERNALIZED_REF_RE.fullmatch` guard and ~45 tests depend on it); the head is a separate
  `[LCM head of externalized output: …]` line appended after the placeholder in summariser
  input and in active-replay stubs.
- Assembly per-depth limit is a config cap (`assembly_max_nodes_per_depth`, 100000, marked
  when hit) rather than a budget-derived number: with the interpolated condensation trigger
  the frontier is bounded by condensation, so any budget-derived limit would only ever be a
  second, weaker bound.
- `/new`: the retain depth is a **carry-over filter** (`carry_over_new_session_context`
  moves depth >= retain; -1 all; 0 nothing) — the shallow nodes stay with the old session
  (visible via `session_scope='all'`). Upstream's boundary-mismatch logic used "session has
  nodes" as its carry-over signal, which only worked because the reset had pruned; the fork
  reproduces those decisions with `_carry_over_candidate_nodes` (same filter applied to the
  pending-reset session only).
- Rotate writes a marker node only over rotated rows that no node covers yet
  (`store_id > _last_compacted_store_id`). Residual: a `compress()` later in the same process
  can summarise the same rows again (upstream's in-process marker rule), giving two nodes
  over one span — double coverage, not loss.
- Bypass (sessions the operator excluded from LCM): content trims are marked; whole-message
  drops in `_trim_bypass_compacted_to_cap` are not — those sessions are outside LCM's store and
  their transcript lives with the host.
- Cleanup-only is gated on the **preflight's replay-diff request** under the threshold
  (`_preflight_cleanup_only_below_threshold`, consumed by the next `compress()`), not on
  "any compress() below the threshold": direct `compress()` calls (the host decided, or tests)
  keep upstream semantics.
- New: a summariser failure with zero persisted passes no longer returns the input untouched
  — the cleanup preamble's drops (ignored-message placeholders etc.) are still published; the
  cooldown is armed either way and an unchanged context is returned as the same object.
### Step 6 — sidecar
- `lcm_node_meta(node_id PK, level, index_block, updated_at)`; created from `SummaryDAG._init_db`
  under the named migration step `betterlcm_node_meta_v1`; prefix `lcm_node` registered with the
  classifier. Rows cascade in `delete_node_batch`. Downgrade to an upstream build: its classifier
  reports `genuinely_newer` for a DB carrying the table — drop `lcm_node_meta` and the
  `betterlcm_node_meta_v1` row in `lcm_migration_state` first.
- The index block is stored, not re-rendered: `node.summary` already contains the whole
  "Expand for details about:" block verbatim, so the prefix would only duplicate it. The
  sidecar copy exists for tools/doctor and for the coverage check.
### Step 7 — index contract
- The level tag is rendered AFTER the header bracket (`[Recent Summary (d0, node 12)] [L2 bullet
  summary]`) because `_is_replayed_context_scaffold_message` recognises the bracket shape.
- Depth labels past 2 keep upstream's `Depth-<n>` (same regex).
- `lcm_expand` default page = `effective_expand_page_tokens` (4000 at 256k → 32000 at 1M);
  `hydrate=true` returns externalized tool outputs inline in node mode.
### Step 8 — non-sweep path
- Chunk boundaries are tool-group aligned already here (plan listed it under step 10) because
  40k chunks at 1M appear with this step; the pre-slicer of step 10 reuses
  `_select_oldest_leaf_chunk_aligned`.
- The wall clock applies to both paths from the same `leaf_deadline`; at 256k the non-sweep
  path runs exactly one pass (cap 1) so the clock never changes upstream behaviour there.
- Sweep pass budget moved to `config.sweep_max_passes` (12); its time budget is the curved
  `leaf_loop_max_seconds` (120 s at 256k = upstream's constant). README's "120 seconds
  between calls" is wrong upstream: it is the total budget for one sweep.
### Step 9 / 9a — condensation and hot paths
- Selection policy cannot be interpolated, so it follows the trigger: while the budget term is
  0 (t = 0) the upstream depth loop runs verbatim (256k DAG identical); once the budget is
  positive the regime is "oldest frontier material first, one fanin group at a time, until
  the frontier is back under budget", bounded by the compress() clock. The switch is driven
  by a continuously weighted value, and at W just above 256k the budget is so small that the
  gate is practically always open.
- `journal_size_limit` stays upstream's 64 MiB (not a bottleneck); `cache_size` and the
  token LRU are weighted.
