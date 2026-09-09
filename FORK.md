# betterlcm — fork of stephenschoettler/hermes-lcm

Base: upstream commit recorded in `.upstream-base` (`git log -1 $(cat .upstream-base)`).
Branch `betterlcm` carries every fork change; remote `upstream` tracks the original.

## Why this fork exists
Make a 1,000,000-token context window work well under a strict no-loss rule, without
degrading ~256k-window behaviour. Full design: the plan this fork implements is summarised
in `docs/fork-design.md`. Two rules govern every change:

1. **Weighted tuning** — every size that is a *preference* is a linear function of
   `context_length` between two anchors: upstream's value at 256k and the 1M design at 1M.
   Anchors live in one table: `window_scaling.py::WINDOW_SCALED_DEFAULTS`.
2. **Pure optimizations** — loss removal, throughput, index quality — apply at EVERY window.
   The 256k anchor carries upstream's TUNING VALUES, never upstream's loss: a cut that fires at
   256k but not at 1M is a defect, not fidelity to upstream. The 256k DAG therefore differs
   from upstream's wherever upstream truncated (the 3,000-char summariser cut and its argument
   cut are gone at both anchors); everything that is a preference still matches upstream there.

### The four configured exceptions to the no-loss rule

Every other path either keeps the content or marks what it removed. These four are operator
policies that deliberately do not, so they are named here rather than discovered later
(audit verify-4 #19). None is on by default except the auxiliary one.

| exception | what is not recoverable | how to turn it off |
|---|---|---|
| `ignore_message_patterns` | a matching message is never stored; the active context keeps a hash placeholder | leave the setting empty (default) |
| sensitive redaction (`sensitive_patterns_enabled`) | the redacted span is replaced before storage, and two different secrets can produce the same placeholder | leave it disabled (default) |
| ignored / stateless / auxiliary sessions | the conversation is bounded but never stored at all | `ignore_session_patterns` / `stateless_session_patterns` empty; auxiliary (subagent) sessions are excluded by design |
| trajectory protection | the protected payload is redacted in place before hashing | do not enable the trajectory subsystem |

If an operator needs those sessions or spans archived, the fork's answer today is: do not use
the exception. Preserving originals behind protection is a design decision, not a bug fix.

### The upstream opt-in subsystems: kept, off, and out of scope

About 20,700 of the plugin's ~73,500 lines are upstream subsystems that have nothing to do with
storing, compacting or assembling the conversation. They are a question-answering apparatus over
the same `lcm.db`: `reasoning.py` (`lcm_compute` — a calculator that refuses to compute over
anything it cannot find verbatim in a cited span), the assertion family (`lcm_query_state` —
typed facts pinned to a source row and span hash, with supersede/fulfil lifecycle), four
generations of pre-answer evidence compiler (`lcm_compile_evidence`, `lcm_evidence_pack`),
adaptive retrieval and its query views (`lcm_retrieve`), `vector_store.py` (embeddings + KNN),
the temporal rollups, and `trajectory_store.py` (agent trajectories as a second corpus).

**They stay in the fork, they stay default-off, and they are not audited or modified.** Deleting
them would only make the eventual upstream merge painful for no behavioural gain. None of them
sits on the path that carries a conversation into the context window, so none can drop or
shorten a message; they read from the store and answer questions. `lcm_compute`,
`lcm_compile_evidence` and `lcm_evidence_pack` are reachable without a flag (the model must call
them); the rest answer `status: disabled` until their flag is set. Fork effort belongs on the
core path: ingest → store → reconcile → compaction → summariser → assembly → bypass → the
retrieval tools → operator, backup and maintenance.

## How to upgrade from upstream (for the next maintainer)

> **Analyse first, merge second — and expect the damage to be where git reports nothing.**
> A textual conflict is the easy case: git stops and asks. The hazard of a substantial fork is
> everything that merges *cleanly* while upstream changed something in a place we had already
> changed — a different hunk of a function we also edited, a caller, a renamed host function our
> hook no longer attaches to, an upstream fix that duplicates ours, a default the curve claims
> to mirror. None of that conflicts, and none of it turns the suite red: our tests exercise OUR
> behaviour, so they pass while upstream's new behaviour goes unexercised. Read the whole diff,
> compute which upstream-changed symbols appear in `docs/fork-touchpoints.md` and re-read those
> functions **in the merged tree, whole**, walk the touchpoints table line by line, and ask of
> every change *does this (re)introduce loss?* The ten failure modes and the full procedure are
> standing task **R1** in [`docs/TASKS.md`](docs/TASKS.md); merge early and often, because a
> merge deferred until upstream has moved hundreds of commits is a rewrite, not a merge.

```
git fetch upstream
git diff $(cat .upstream-base)..upstream/main   # READ THIS, all of it, before merging
git merge upstream/main          # or rebase betterlcm onto upstream/main
scripts/test.sh                  # must be green
```
Then redeploy: `~/.hermes/plugins/hermes-lcm` is a clone of this repo (remote `fork`), pinned in
`~/.hermes/plugins/.install-metadata.json` so `hermes plugins update` refuses to touch it.
```
git -C ~/.hermes/plugins/hermes-lcm pull fork betterlcm
python3 - <<'PY'   # re-pin the deployed revision
import json, subprocess, pathlib
p = pathlib.Path.home()/".hermes/plugins/.install-metadata.json"; d = json.loads(p.read_text())
d["hermes-lcm"]["revision"] = subprocess.check_output(["git","-C",str(pathlib.Path.home()/".hermes/plugins/hermes-lcm"),"rev-parse","HEAD"],text=True).strip()
p.write_text(json.dumps(d, indent=2)+"\n")
PY
```
Then, before calling the upgrade done: both e2e anchors clean against the DEPLOYED plugin
(`python3 scripts/e2e_no_loss.py 262144 400` and `... 1000000 3000`), update `.upstream-base`,
record the merge as a pass in `docs/TASKS.md`, and re-run audit **V1** — an upstream merge is
exactly when "strictly better than upstream" can silently stop being true.

`plugin.yaml` keeps upstream's version string (four upstream tests pin it; the fork is
identified by this file and `git log`). The live `lcm.db` gains the
`lcm_node_meta` table on first use; an upstream build classifies such a DB as newer (drop the
table and the `betterlcm_node_meta_v1` row in `lcm_migration_state` to go back).
Fork code is kept in NEW modules wherever possible so upstream files receive only small,
localized hook calls. The complete list of upstream files touched, with the reason for each
touch, is in `docs/fork-touchpoints.md` — read it before resolving a conflict: it tells you
whether a conflicting hunk is a hook (keep ours, re-apply on top of theirs) or a behavioural
change upstream also made (reconcile). Every fork-only test lives under `tests/fork/`.

## Tracking lossless-claw (the second maintenance duty)
The fork was inspired by [lossless-claw](https://github.com/Martian-Engineering/lossless-claw)
(OpenClaw, TypeScript). Watch **commits, not only tags** — the last version analysed in depth is
**v1.0.0** (`docs/claw-comparison/v1.0.0.md`). On every release *or* meaningful commit past it:
1. Clone the release tag to a temp dir (`git clone --depth 1 --branch vX.Y.Z … /tmp/lossless-claw-vX.Y.Z`).
2. Compare it deeply against this fork, one agent per aspect, code-level with citations on both
   sides: (1) compaction/DAG algorithm, (2) loss avoidance/provenance/recovery, (3) summariser
   prompts/index quality/evaluation, (4) retrieval tools/host integration/operability. The
   prompts used for v1.0.0 are in `docs/claw-comparison/prompts/`; reuse them.
3. Keep only findings where claw is genuinely better *for this fork's purpose* (no loss; 1M
   without degrading 256k) or fixes something the hermes-lcm base does badly. Port them into
   the fork modules, with tests under `tests/fork/`, and record each ported item (and each
   rejected one, with the reason) in `docs/claw-comparison/vX.Y.Z.md`.
4. **Treat a claw BUGFIX as a lead, not only as a port candidate.** Both projects solve the same
   problem, so a bug claw fixed very often exists here in an analogous shape — different code,
   same mistake. For each fix, find the corresponding place in this fork and prove by probe
   whether the defect is present, then fix it here on its own merits even when the claw patch
   itself is not portable.
5. The same two hard rules apply as for upstream merges: 256k keeps upstream's TUNING VALUES
   (not its cuts — see rule 2 at the top), and nothing drops content without a marker.

The full procedure, its trigger and the two outstanding audits (is the fork strictly superior to
upstream mainline; did we miss anything the current claw does better) are written up as standing
tasks R1/R2 and V1/V2 in [`docs/TASKS.md`](docs/TASKS.md) under "WHAT IS LEFT TO DO".

## Fork configuration reference
Every fork setting has a dataclass field, an env var and (for the weighted ones) an anchor in
`window_scaling.py`. `0` / `0.0` means "no override — use the curve"; `lcm_status` →
`window_scaling` shows the resolved value and its source for the current window.

| env var | default | field |
|---|---|---|
| `LCM_SCALE_LOW_WINDOW` | `262144` | `scale_low_window` — window where every weighted value equals upstream's |
| `LCM_SCALE_HIGH_WINDOW` | `1000000` | `scale_high_window` — window where it equals the large-window design |
| `LCM_DRAIN_STOP_FRACTION` | `0.0` | `drain_stop_fraction` (curve: threshold → 0.30 of W) |
| `LCM_LEAF_CHUNK_FRACTION` | `0.0` | `leaf_chunk_fraction` (curve: 1.0 → 0.04 of W) |
| `LCM_LEAF_PASS_CAP` | `0` | `leaf_pass_cap` (curve: 1 → 64) |
| `LCM_LEAF_LOOP_MAX_SECONDS` | `0.0` | `leaf_loop_max_seconds` (curve: 120 → 200) |
| `LCM_SUMMARY_BUDGET_FRACTION` | `0.0` | `summary_budget_fraction` (curve: 0 → 0.20 of W; condensation trigger) |
| `LCM_SUMMARY_CONCURRENCY` | `0` | `summary_concurrency` (curve: 1 → 6) |
| `LCM_SERIALIZE_MESSAGE_MAX_CHARS` | `0` | `serialize_message_max_chars` (curve: 4·W → 4·W — the whole window at both anchors, so it never binds; `0` means no cap at all. An explicit value is an operator cap and still cuts only through a sized `[LCM elided …]` marker. Tool-call arguments share this cap.) |
| `LCM_EXPAND_PAGE_TOKENS` | `0` | `expand_page_tokens` (curve: 4000 → 32000) |
| `LCM_TOOL_RESPONSE_CHAR_SCALE` | `0.0` | `tool_response_char_scale` (curve: 1 → 4) |
| `LCM_SQLITE_CACHE_KIB` | `0` | `sqlite_cache_kib` (curve: 2048 → 65536) |
| `LCM_TOKEN_CACHE_SIZE` | `0` | `token_cache_size` (curve: 2048 → 8192) |
| `LCM_SUMMARY_FAILURE_COOLDOWN_SECONDS` | `600.0` | `summary_failure_cooldown_seconds` |
| `LCM_ASSEMBLY_MAX_NODES_PER_DEPTH` | `100000` | `assembly_max_nodes_per_depth` (cap hit is marked in the prefix) |
| `LCM_SWEEP_MAX_PASSES` | `12` | `sweep_max_passes` (upstream's constant) |
| `LCM_LEAF_SUMMARY_RATIO` / `_MIN_TOKENS` / `_MAX_TOKENS` | `0.2` / `2000` / `12000` | leaf summary size rule (upstream's literals) |
| `LCM_CONDENSATION_RATIO` / `_MIN_TOKENS` | `0.4` / `1000` | condensation size rule (upstream's literals) |

Upstream settings the curve also drives when not set explicitly: `context_threshold`
(0.35 → 0.80), `summary_timeout_ms` (60 s → 200 s), `expansion_timeout_ms` (120 s → 200 s),
`fresh_tail_count` (32 → 400), `fresh_tail_max_tokens` (off → 0.15·W), `incremental_max_depth`
(3 → 5), `summary_spend_max_calls` (24 → 120), `summary_circuit_breaker_failure_threshold`
(2 → 4), `l2_budget_ratio` (0.5 → 0.8), `large_output_active_replay_stub_threshold_tokens`
(25k → 100k), `expansion_context_tokens` (32k → 125k), `summary_prefix_target_tokens`
(leaf_chunk_tokens → 0.20·W, sweep flag only).

## Layout of fork-only code
- `window_scaling.py`     — the anchor table and the curve; `resolve_window_scaled(config, W)`
- `host_cooldown.py`      — the host's compression-failure cooldown protocol for a plugin engine
- `errors.py`             — `SummaryUnavailableError` (replaces upstream's silent L3 truncation)
- `marked_loss.py`        — every marker text the fork leaves where upstream cut or dropped silently
- `node_meta.py`          — sidecar table `lcm_node_meta` (escalation level + index block)
- `leaf_pipeline.py`      — concurrent leaf summarisation as a lookahead over the serial loop; compaction lock
- `coverage_doctor.py`    — `lcm_doctor coverage` / `/lcm doctor coverage`: index-adequacy check
- `tests/fork/`           — fork tests, organised by the step that introduced them
