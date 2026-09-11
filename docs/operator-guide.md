# Operator guide

[`README.md`](../README.md) is the operator reference: install, activate, verify, update, the
configuration tables, the slash commands, the retrieval contract and troubleshooting all live
there. This page holds only what the README does not — the upgrade path, what silently switches
the fork's weighting curve off, the levers this fork refuses to honour, the opt-in subsystems and
their privacy boundary, the full diagnostics surface, and the operator scripts.

Every value stated here was read out of the code, not out of another document. Where a value
comes from the weighting curve it is given at **both** anchors (256k and 1M), because a single
number would only be true at one window.

## Verify

`hermes plugins` should report the plugin, the engine and all 15 tool schemas:

```text
Plugins (1):
  ✓ hermes-lcm v1.1.0-beta.2 (15 tools)

Provider Plugins:
  Context Engine: lcm
```

`plugin.yaml` deliberately keeps upstream's version string, so that line does **not** tell you
whether you are running this fork. `lcm_status`, `lcm_inspect` and `lcm_doctor` report
`plugin_git_commit`, `plugin_git_branch` and `plugin_git_dirty` for a source checkout — the branch
is what identifies the fork. `lcm_status` also carries a `window_scaling` block that only this
fork emits.

## Upgrade from v0.20.0 or v0.21.0-rc2 to v1.0.0-rc.1

1. While the old runtime is running, run `/lcm backup`. If Hermes or any other SQLite writer may
   still be running, this is the only supported online backup path.

   **`/lcm backup` copies the SQLite database only.** Externalized payloads live in separate files
   (`lcm-large-outputs/` under the Hermes home by default, or
   `LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH`), and the storage-boundary guard that spills inline
   media/base64 blobs is always on, so a database restored without that directory keeps references
   whose content cannot be recovered. Copy the payload directory alongside the database, and
   restore them together.
2. Alternatively, stop Hermes and every other process that can write the database. After all
   writers are fully stopped, copy the profile's `lcm.db` plus any existing `lcm.db-wal` and
   `lcm.db-shm` companions together as one quiescent snapshot. Do not copy these files separately
   while a writer is live.
3. Update the plugin checkout and restart Hermes.
4. Send one normal message, then confirm `lcm_status` reports the expected database path.
5. For a migration-shape audit, query that database with
   `SELECT value FROM metadata WHERE key = 'schema_version';`; the expected result is `5`.

No manual core migration, data import, or embedding backfill is required. A database created by
either v0.20.0 or v0.21.0-rc2 opens in place and remains on core schema version 5. The assertion,
query-view and trajectory families use additive named feature markers and create their tables only
when the corresponding store or workflow is invoked, so a stock/default-off upgrade creates none
of those optional tables. For rollback to either v0.20.0 or v0.21.0-rc2, restore the pre-upgrade
backup rather than opening a database modified by v1.0.0-rc.1 with the older plugin.

**Fork-specific:** this fork adds one sidecar table, `lcm_node_meta`, on first use, plus a
`better_hermeslcm_node_meta_v1` row in `lcm_migration_state`. Neither moves `schema_version`, but
an *upstream* build classifies such a database as newer than itself and refuses it; drop the table
and that row to go back to an upstream build.

## What switches the weighting curve off

Every setting that is a *preference* about how to spend the window is resolved as a linear
function of the model's context length between a 256k anchor (upstream's value) and a 1M anchor.
The anchors and their per-value justification are in
[`window_scaling.py`](../window_scaling.py); the classified table is in
[`FORK.md` → Fork configuration reference](../FORK.md#fork-configuration-reference).

**An explicit value always wins over the curve.** That is the intended contract, and it is also
the trap: several things count as "explicit" without an operator thinking of them as configuration.

| what you set | what stops being curved |
|---|---|
| `LCM_CONTEXT_THRESHOLD` | `context_threshold` (curve: `0.35` at 256k → `0.80` at 1M) |
| `lcm.context_threshold` in `~/.hermes/config.yaml` | the same — it is read as a configured value, not a default |
| `compression.threshold` in `~/.hermes/config.yaml` | the same, unless `compression.enabled` is false. A host-wide compression threshold is inherited by LCM and suppresses the curve |
| the Codex gpt-5.5 route autoraise (`compression.codex_gpt55_autoraise`, on by default) | the same, whenever it actually raises the threshold on that route |
| `auxiliary.compression.timeout` in `~/.hermes/config.yaml` | `summary_timeout_ms` (curve: 60 s at 256k → 200 s at 1M) |
| `/lcm preset apply` values, or setting them by hand | `context_threshold` and `fresh_tail_count` for that session |
| `LCM_DYNAMIC_LEAF_CHUNK_ENABLED=true` | the curved leaf chunk — the pass reverts to upstream's doubling policy from `LCM_LEAF_CHUNK_TOKENS` up to `LCM_DYNAMIC_LEAF_CHUNK_MAX` |
| `LCM_THRESHOLD_FULL_SWEEP_ENABLED=true` | the non-sweep drain: the sweep path is bounded by `LCM_SWEEP_MAX_PASSES` (12) and the curved leaf-loop budget (120 s at 256k → 200 s at 1M) instead |

`lcm_status` → `window_scaling` prints every curved setting with its resolved value and its
source (`curve@t=…`, `env`, `config_yaml:…`, `explicit`, `manual`, `upstream(no window)`), plus
the `t` position between the anchors. That block is the answer to "is the curve actually running
here?" — read it before tuning anything.

Two defaults the curve gives that differ sharply from upstream, and that operators inherit
silently:

- `fresh_tail_max_tokens` — upstream's default is `0`, meaning **no token cap**. In this fork the
  curve gives it `0.15 × window` at every anchor (≈39,300 tokens at 256k, 150,000 at 1M), and
  `fresh_tail_count` is `400` at every anchor rather than upstream's 32. The token cap is what
  sizes the tail; the count is only an upper bound on messages. Setting `LCM_FRESH_TAIL_MAX_TOKENS`
  to `0` restores upstream's uncapped behaviour rather than "disabling" a cap you wanted.
- `new_session_retain_depth` — `0` in this fork (upstream: `2`). After a manual `/new`, no DAG
  depth is carried into the new session. Nothing is deleted: the previous session keeps its nodes
  and they stay reachable through session-scoped retrieval. Set `2` (or `-1`) for upstream's
  carry-over.

`LCM_LEAF_CHUNK_TOKENS` (default `20000`) is **not** the chunk size in this fork. It is the floor
— "do not bother compacting a backlog smaller than this" — and the effective floor is
`min(LCM_LEAF_CHUNK_TOKENS, one chunk)`, so lowering it still makes compaction start earlier while
a full chunk is never refused for being under it. The chunk size itself is
`LCM_LEAF_CHUNK_FRACTION`, 4 % of the window at both anchors (≈10,500 tokens at 256k, 40,000 at
1M).

## Levers this fork refuses to honour

Two upstream settings are retired. They are not defaults an operator may flip; a configured value
is overwritten at load, a warning is logged, and the warning is surfaced in `lcm_status` under the
config-source warnings:

| setting | env var |
|---|---|
| `large_output_externalization_enabled` | `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` |
| `large_output_active_replay_stubbing_enabled` | `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED` |

Why: the host already spills oversized tool output to its own directory, leaves a
`<persisted-output>` marker in the transcript, and lets the agent read the file — doing it again
in the plugin duplicates a mechanism that works. And externalizing an inline body puts a reference
into the replay that the agent never saw; the active context is supposed to be what happened.

Consequences for the surrounding settings.
`LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS` and
`LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS` gate only the two retired paths, so with
those forced off neither has anything to act on.
`LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED` (still `false` by default, and still yours to set) does
have work available: it rewrites a compacted tool-role row to a compact placeholder whenever the
row's whole content *is* an externalized payload — which the always-on paths below still produce.
It keeps the `store_id` and the payload file, so recovery through `externalized_ref` survives, but
after GC `lcm_grep` no longer matches the original blob text directly.
`LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH` still matters: it names the directory the always-on
storage-boundary guard writes inline `data:*;base64` and long base64-looking runs into, and the
directory `persisted_output_recovery_copy_enabled` (on by default) copies host-spilled tool output
into before the host expires it after 24 hours.

## Opt-in subsystems: exposed, off, and out of scope

Hermes exposes all 15 LCM tool schemas whenever LCM is the active context engine. **Exposure is
not activation.** About a third of the plugin is an upstream question-answering apparatus over the
same `lcm.db` — reasoning/`lcm_compute`, the assertion family, four generations of evidence
compiler, adaptive retrieval and its query views, embeddings and the vector store, the temporal
rollups, and trajectories. None of it sits on the path that carries a conversation into the
context window. **In this fork it stays in, stays default-off, and is neither audited nor
modified.** Enabling one of these means running upstream code this fork makes no claims about.

On a stock install:

- `lcm_compute`, `lcm_compile_evidence` and `lcm_evidence_pack` are bounded, provider-neutral
  operations over caller-supplied exact refs. Calling them does not enable a store, run an
  extractor, or activate an answering model.
- `lcm_query_state` returns `status: disabled` until `LCM_ASSERTIONS_ENABLED=true` creates and
  binds the rebuildable assertion sidecar.
- `lcm_retrieve` returns `status: disabled` until `LCM_ADAPTIVE_RETRIEVAL_ENABLED=true`. The
  controller itself has no model or provider client, but retrieval calls it dispatches retain
  their existing embedding-provider behaviour.
- `lcm_recent` works without rollups, falling back to time-bounded leaf summaries.
- `lcm_recall` works without embeddings, degrading to its full-text arm.

| Variable | Default | Use |
|----------|---------|-----|
| `LCM_ASSERTIONS_ENABLED` | `false` | Create and bind the same-DB assertion sidecar so `lcm_query_state` can return typed, source-cited state. This alone performs no extraction or backfill. |
| `LCM_ASSERTION_EXTRACTION_ENABLED` | `false` | With assertions enabled, run bounded structured extraction over exact persisted rows before compaction. This may send source text to the configured extraction/summary model. |
| `LCM_ASSERTION_EXTRACTION_MODEL` | empty | Extraction-model override; otherwise uses `LCM_EXTRACTION_MODEL`, then the summary model. |
| `LCM_ASSERTION_EXTRACTION_MAX_SOURCES_PER_PASS` | `4` | Maximum exact source rows per extraction pass; runtime clamps to 1–8. |
| `LCM_ASSERTION_EXTRACTION_TIMEOUT_SECONDS` | `30` | Timeout for each exact-source extraction call; runtime clamps to 0.1–120 seconds. |
| `LCM_QUERY_VIEWS_ENABLED` | `false` | Create and bind demand-shaped evidence views without invoking a model or retrieval provider. |
| `LCM_ADAPTIVE_RETRIEVAL_ENABLED` | `false` | Enable `lcm_retrieve` and bind query views for evidence reuse. Episodes are bounded to existing retrieval tools and store evidence/traces, never final prose. |
| `LCM_PREANSWER_EVIDENCE_ENABLED` | `false` | Enable the automatic pre-answer evidence hook. Disabled preserves the ordinary hook context and performs no retrieval or computation. |
| `LCM_PREANSWER_EVIDENCE_MODE` | empty | When the master flag is enabled, empty selects legacy selective behavior; explicit values are `off`, `legacy_selective`, or `requirements_v1`. |
| `LCM_SELECTIVE_COMPILER_ENABLED` | `false` | Separately opt into the semantic selector for code-derived closed operations. Disabling the selective compiler does not prevent the pre-answer hook from retrieving a baseline. |
| `LCM_SELECTIVE_COMPILER_MODEL` | empty | Optional model override for the selective compiler. |
| `LCM_EMBEDDINGS_ENABLED` | `false` | Bind embedding storage and the semantic/hybrid retrieval arms. Requires a provider and model, and an explicit `/lcm embed warmup` plus `/lcm embed backfill --apply` before anything is searchable. |
| `LCM_TEMPORAL_ROLLUPS_ENABLED` | `false` | Build derived UTC day/week/month rollups and their maintenance hooks; `lcm_recent` prefers ready rollups and otherwise falls back to leaf summaries. |

When no caller-supplied baseline is available, routed pre-answer turns may call `lcm_recall` to
build one. If embeddings are enabled, this retrieval inherits the configured embedding provider
and may send the current question to that provider. Disabling the selective compiler prevents its
selector and answering model calls; it does not disable this retrieval path.

**Privacy boundary.** Assertion and query-view records live in the selected profile's existing
`lcm.db` and may include exact quotes, spans, and dependency refs. Provider-neutral evidence tools
do not upload them by themselves. Embedding-backed retrieval — including automatic pre-answer
baseline retrieval, assertion extraction, and the optional selective compiler — can send
configured content to their selected providers. The chunk corpus in particular embeds raw verbatim
message text rather than generated summaries. Review those provider and redaction settings before
opting in; sensitive-pattern redaction is also default-off and is forward-only, so enabling it
never retro-redacts history already stored.

## Diagnostics

Slash commands are disabled by default; enable them only in trusted operator contexts with
`LCM_ENABLE_SLASH_COMMAND=1`. Without them, the read-only checks are still reachable as agent
tools (`lcm_status`, `lcm_inspect`, `lcm_doctor`). The complete surface:

```text
/lcm | /lcm status
/lcm doctor
/lcm doctor clean | clean apply | clean lifecycle | clean lifecycle apply
/lcm doctor repair | repair apply | repair schema-stamp | repair schema-stamp apply
/lcm doctor source | source apply
/lcm doctor retention
/lcm doctor coverage                     # fork-only: index-adequacy check
/lcm backup
/lcm rotate | rotate apply
/lcm rollups …                           # temporal rollups (opt-in subsystem)
/lcm assertions …                        # assertion sidecar (opt-in subsystem)
/lcm preset show | suggest | apply --dry-run
/lcm embed warmup | backfill …           # embeddings (opt-in subsystem)
/lcm help
```

Every `apply` path requires the corresponding read-only preview first and takes a backup;
`clean apply` and `clean lifecycle apply` additionally require
`LCM_DOCTOR_CLEAN_APPLY_ENABLED=true`.

`/lcm preset` is inspection only — `apply` is dry-run and previews environment variables; it never
writes files or mutates process state. `lcm_status` exposes the same data as read-only JSON under
`preset_suggestion`. Note that the shipped presets set `context_threshold`, `fresh_tail_count` and
`leaf_chunk_tokens` explicitly, which opts the first two out of the weighting curve for that
session; `target_after_compaction` in their provenance is a benchmark field, not a runtime knob.

**Doctor triage.** `lcm_doctor` JSON carries a top-level `guidance` array for every warning or
failure, and the slash command prints `triage_guidance` using the same vocabulary. Each item maps
a warning class to one of three operator actions:

- `safe/ignore` — informational operating state; leave it alone unless it is crowding useful
  recall or repeatedly surprising operators.
- `inspect` — read the named rows, session ids or config before making changes.
- `backup-first cleanup` — run the read-only preview, create `/lcm backup`, then run the explicit
  apply command only if the preview still matches intent.

Warning-only classes must not auto-clean state: `summary_quality`, broad
`lifecycle_fragmentation`, payload-storage suspicion and `context_pressure` are evidence for
review, not proof that mutation is safe.

## Troubleshooting

The README covers the two common ones (`lcm (not found)` in `hermes plugins`, and an unbound
`/lcm status` after a restart). One more:

### Startup log mentions `context-engine schemas` or `Path B fallback`

Expected on hosts that do not advertise `context_engine_tool_handlers_receive_messages`, including
Hermes Agent v0.16. LCM tools are still available through the context-engine schema/dispatch path
(Path B). The plugin intentionally avoids standalone plugin-registry tool registration (Path A) on
those hosts, because Path A would shadow Path B and lose current-turn ingest. Healthy signals are
unchanged: selected context engine `lcm`, all 15 `lcm_*` tools in the live tool list, and
`lcm_status` / `lcm_inspect` / `lcm_doctor` responding after one normal message has initialized the
session.

## Operator scripts

### Historical tool-output sidecars

`scripts/backfill_externalized_tool_outputs.py` pre-creates externalized-payload sidecars for
large textual tool rows already present in an LCM database. This fork forces ingest-side
externalization off, so on a stock install nothing new arrives in that shape — the script is for a
database written by an upstream build, or by an install that had the lever set before it was
retired.

It opens SQLite read-only, never rewrites messages or summaries, and is dry-run by default:

```bash
python scripts/backfill_externalized_tool_outputs.py \
  --database <path to lcm.db> \
  --hermes-home <hermes home> \
  --manifest ./externalization-backfill.json
# add --apply only after reviewing the manifest; --rollback <manifest> to undo
```

The manifest is a durable ownership journal — refs, digests, provenance proofs, target-identity
hashes, the active redaction policy, counts, sizes and token estimates — never raw payload
content, session ids or tool-call ids. It is bound to one database file and one payload storage
root and refuses reuse against another target, or resumption under a different redaction policy.
Apply and rollback require the storage directory to be owned by the current user and not
group/other-writable, and hold an advisory lock on it, so stop the profile that owns the target
database (and anything else running as that account that can write the payload directory) before
running either. Rollback deletes a sidecar only when its provenance binds it to that journal, its
content still matches the recorded digest, and no message, nested `messages.tool_calls` value, or
summary references it.

### OpenClaw / lossless-claw history

`scripts/import_lossless_claw.py` imports OpenClaw history from a source `lcm.db` or from JSONL
session exports. Dry-run is the default, apply backs up the target first, and reruns are
idempotent for the same `--import-id`. See
[README → OpenClaw/lossless-claw import](../README.md#openclawlossless-claw-import).

## Related references

- [README](../README.md) — install, activation, configuration tables, retrieval contract
- [`CLAUDE.md`](../CLAUDE.md) — the contributor contract and the fork's verification doctrine
- [`FORK.md`](../FORK.md) — every curve anchor, classified, with both endpoints
- [Retrieval tools reference](retrieval-tools.md) — exact tool contracts
- [Release validation](release-validation.md)
- [Dependency assurance](dependency-assurance.md)
