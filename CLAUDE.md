# better-hermes-lcm — what this is, and how to work on it

**This repository is `better-hermes-lcm`, a fork of [`stephenschoettler/hermes-lcm`](https://github.com/stephenschoettler/hermes-lcm)** (the LCM context-engine plugin for Hermes Agent — Python, SQLite-backed message store plus a summary DAG). Upstream base commit is in `.upstream-base`. The fork branch is `better-hermes-lcm`.

Read this before touching anything. Then [`FORK.md`](FORK.md) (the maintenance contract). The backlog is the **GitHub issue tracker** — open work only; what was done is in the code and `git log`.

**Do not write anything down that is not code, the README, this file, or an issue.** No design docs, no pass logs, no notes-to-self, no ledger files. A finding is implemented, an open issue, or nonsense.

Three rules that keep the backlog honest, carried over from the file it used to live in:

- Every task gets an adversarial check ("how can the index still lie after this?"), not a plan-conformance check.
- No task is closed with a run that was started before its last edit.
- Nothing is written in an issue that is already in the code or in `git log`.

**Answer from the code, never from a document — this one and the backlog included.** Not read it yet? Then read it, then answer.

**A finding stands until it is refuted, not until someone sounds annoyed.**

## Why the fork exists — three goals, in priority order

**a) Opinionated hatred of loss and truncation.** This is the top-priority rule and it overrides convenience, elegance and upstream fidelity. Concretely:

- Content that reaches the plugin never becomes unreachable — not unstored, deleted, overwritten, or reachable from no summary node.
- Anything removed from what the summariser reads, or from what the agent reads back, leaves a marker in its place saying **what** was removed and **how to recover it**. Summaries are acceptable *only* because they are clearly marked, carry provenance, and act as an index into retained history.
- A bounded, capped, timed-out, failed or partial operation is never presented as complete. An empty result must not read as "there is nothing"; a summary must not read as covering everything; a status must not read as healthy; a tool must not report `complete: true` after hitting a work cap.
- A marker can never itself be trimmed away, and never says less than what was removed.

**b) Optimise for large context windows** (up to 1M tokens) without degrading small ones. Every value that is a *preference* is a smooth weighted interpolation between two anchors — 256k and 1M — resolved through `window_scaling.py`. **Never a band switch, never an `if window > X` branch.**

**c) Take what lossless-claw does better.** [lossless-claw](https://github.com/Martian-Engineering/lossless-claw) (TypeScript, OpenClaw) solves the same problem; where it is better *for this fork's purpose*, port it. **The goal is to be strictly superior to upstream hermes-lcm** — better on the no-loss axis without being worse on any other. Audit prompts are in `docs/claw-comparison/prompts/`; what a round finds is either fixed in the code or an open issue — audit reports are not kept.

## The rule that is most often gotten wrong

> **Upstream is the FLOOR — never worse than it — not the target. The 256k anchor carries upstream's TUNING VALUES, never upstream's LOSS.**

A value takes upstream's number at 256k only when it is a genuine *preference*: a cost, latency or headroom tradeoff where upstream's choice is as good as any other. Thresholds, timeouts, wall clocks, depth caps and cache sizes are like that.

Where a value decides **how much is lost, or how coarse the index is**, upstream's number is not adopted at any window — it is decided on merit. That covers truncation and silent drops, and also: the leaf chunk (the granularity of the index), the pass cap and drain stop (whether a backlog can drain at all), the fresh tail (how much stays verbatim), the condensation gate (when the frontier is coarsened), and summariser concurrency.

A cut that fires at 256k but not at 1M is a **defect in this fork**. "That is what upstream does" is a description, never a justification — it is the exact sentence that let a whole-backlog leaf chunk survive four audit rounds here. If you find yourself writing it in a comment, you are about to ship the bug this fork exists to remove.

## Architecture: where the fork's code lives

Fork logic goes in **new modules** so upstream merges stay reviewable; upstream files get small hooks, each with the reason for it in the comment beside it.

> **Naming:** everything is **better-hermes-lcm** — project, repository and branch. Two identifiers deliberately keep the older spelling because they exist in databases already written: `node_meta.MIGRATION_STEP` (`better_hermeslcm_node_meta_v1`) and `node_meta.LEGACY_MIGRATION_STEPS` beside it, which names the pre-rename row so an older build's database has it retired rather than recording one migration twice.

Fork modules: `window_scaling.py` (the anchor table and curve), `window_scaled_mixin.py` (`effective_*` resolution), `marked_loss.py` (**every marker the fork emits**), `host_cooldown.py`, `leaf_pipeline.py`, `node_meta.py`, `coverage_doctor.py`, `errors.py`.

The core path — the only code that can lose a conversation, and where effort belongs:

```
ingest → store.py → reconcile.py → compaction.py (leaf + condensation)
       → escalation.py (summariser) → engine.py::_assemble_context → bypass.py
       → tools.py (retrieval) → maintenance / lifecycle / backup
```

**`git diff $(cat .upstream-base)..HEAD` is the merge risk surface.** It is exact and it is always current, which is why there is no marker convention and no table: a hand-maintained list of the same thing had a quarter of its rows describing code that had moved by the time it was deleted, and a label repeated in five hundred comments told a reader nothing they did not already know from being in this repository. Put the *reason* for a hook in a comment beside it, where it cannot drift from what it describes, and let the diff say which lines are ours.

## Things that will bite you

- **The opt-in subsystems are out of scope.** ~20,700 of ~73,500 lines are an upstream question-answering apparatus over the same database: `reasoning.py`/`lcm_compute`, the assertion family, four generations of evidence compiler, adaptive retrieval + query views, `vector_store.py`, the rollups, `trajectory_store.py`. **They stay in, stay default-off, and are neither audited nor modified** — deleting them would only make the upstream merge painful for zero behavioural gain. None of them sits on the path that carries a conversation into the context window. Do not spend effort there.
- **Never assume or mention which summariser model or gateway is configured.** That is the operator's choice; the plugin is provider-neutral and so is every comment and document in it.
- **A receipt must never displace live content.** Markers are positionally neutral or they are budgeted; a receipt that pushes out the caller's newest message has caused loss to prevent loss. This has been a real regression more than once.
- **A receipt is a claim that something was removed.** Do not emit one for a turn that held nothing — a false claim of removal is its own defect.
- **Test edits:** fork tests go in `tests/fork/`. An upstream test that pins behaviour the fork deliberately changed is re-pointed with a `# fork: better-hermes-lcm` comment saying what it used to assert and why that changed — never deleted silently, and never recorded anywhere but in place.
- **Edit files with the editor.** Do not patch by piping heredoc Python/sed scripts through the shell: the diff becomes unreadable and those scripts have corrupted source files here before.

## Do not

Enable deferred maintenance; set either assembly cap (`max_assembly_tokens`,
`reserve_tokens_floor`); set `incremental_max_depth=0`; enable sensitive-pattern redaction (it is
explicitly not lossless); curve `protect_last_n` (it belongs to the bypass path, not the LCM
tail); enable ingest-side externalization (the host already spills oversized tool output — doing
it ourselves puts a ref in the replay the agent never saw); leave the checkout dirty.

**The four places the fork knowingly does not keep everything.** Every other path either keeps
the content or marks what it removed. These are operator policies, all off by default, named
here so they are not discovered later:

| exception | what is not recoverable | how to avoid it |
|---|---|---|
| `ignore_message_patterns` | a matching message is never stored; the active context keeps a hash placeholder | leave empty (default) |
| `sensitive_patterns_enabled` | the redacted span is replaced before storage, and two different secrets can collide on one placeholder | leave disabled (default) |
| ignored / stateless / auxiliary sessions | the conversation is bounded but never stored at all | leave `ignore_session_patterns` / `stateless_session_patterns` empty; auxiliary (subagent) sessions are excluded by design |
| trajectory protection | the protected payload is redacted in place before hashing | do not enable the trajectory subsystem |

## Verification — what counts as evidence

A green suite is a **regression guard**, not evidence. Evidence is end-to-end behaviour and real probes.

```bash
bash scripts/test.sh                          # full suite; umask 077 matters (SQLite refuses group-writable dirs)
bash scripts/test.sh tests/fork/test_x.py     # one file

# the executable form of the no-loss doctrine — runs THIS checkout, no installation needed:
python3 scripts/e2e_no_loss.py 262144 400
python3 scripts/e2e_no_loss.py 1000000 3000
```

Both anchors must report **0 unreachable rows, 0 missing facts, 0 facts never offered to the summariser**. A change that cannot produce those numbers is not finished.

But they stub the summariser (`fake_summary`, `scripts/e2e_no_loss.py:118`), so they say nothing about summary quality or route limits. The stub emits ~600 tokens where a real leaf emits 8,000, so the frontier never crosses the condensation gate and **both anchors end at `depths [0]`** — condensation never runs. Depth ≥ 1 is verifiable by reading, not by these runs. Padding the stub would only test the stub.

## The working loop

1. Fix a batch → full suite green → commit.
2. Re-run **both** e2e anchors. They run this checkout; nothing has to be installed.

> **The installed version is the latest RELEASE, never a working commit.** `~/.hermes` is the
> owner's running system, not a test bed. Do not push development commits into it: no
> `git reset --hard` in `~/.hermes/plugins/hermes-lcm`, no edit of `.install-metadata.json`,
> no change to `~/.hermes/config.yaml`. An installation happens when there is a release to
> install, and it installs that release. If something genuinely has to be verified against an
> installation, point `LCM_E2E_PLUGIN_DIR` at it — that checks what is running, it does not
> change it.

Audits (Codex astra, prompts in `docs/claw-comparison/prompts/`) want a **stable tree** — do not commit while a round is running.

## Standing tasks — these never complete

Each is an open issue labelled `standing` or `verification`, written out in full there:

- **R1 — upstream changed.** Analysis first, merge second. Textual conflicts are the *easy* case; the hazard is everything git merges cleanly while upstream changed something where the fork had already changed something. Ten named failure modes (M1–M10) with what actually catches each.
- **R2 — lossless-claw moved past the analysed v1.0.0.** Diff it; sort changes into capability / behaviour / **bugfix**; treat every claw bugfix as a *lead*, because the same bug very likely exists here in an analogous shape.
- **R3 — real sessions exist.** The day one does, its `lcm.db` is evidence no scripted run can produce; hunt bugs in it structurally.
- **V1 — is the fork strictly superior to upstream mainline?** Not re-run since audit D. Re-run after every upstream merge.
- **V2 — did we miss anything the current claw does better?** 37 items from the v1.0.0 sweep are still undecided.

## Map of the documents

| file | what it is |
|---|---|
| [`README.md`](README.md) | user-facing; "What the fork changes" is the upstream-vs-fork comparison |
| [`FORK.md`](FORK.md) | maintenance contract: the two rules, the configured exceptions to no-loss, upgrade and claw-tracking procedures, the fork config reference |
| [the issue tracker](https://github.com/ConstantinLevin/better-hermes-lcm/issues) | the backlog: open work only |
| `docs/claw-comparison/prompts/` | the four standing lossless-claw comparison prompts |
