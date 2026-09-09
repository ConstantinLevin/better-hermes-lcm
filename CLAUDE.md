# better-hermeslcm — what this is, and how to work on it

**This repository is `better-hermeslcm`, a fork of [`stephenschoettler/hermes-lcm`](https://github.com/stephenschoettler/hermes-lcm)** (the LCM context-engine plugin for Hermes Agent — Python, SQLite-backed message store plus a summary DAG). Upstream base commit is in `.upstream-base`. The fork branch is `better-hermeslcm`.

Read this before touching anything. Then read [`FORK.md`](FORK.md) (the maintenance contract) and [`docs/TASKS.md`](docs/TASKS.md) → **"WHAT IS LEFT TO DO"** (the authoritative work list and the standing tasks).

## Why the fork exists — three goals, in priority order

**a) Opinionated hatred of loss and truncation.** This is the top-priority rule and it overrides convenience, elegance and upstream fidelity. Concretely:

- Content that reaches the plugin never becomes unreachable — not unstored, deleted, overwritten, or reachable from no summary node.
- Anything removed from what the summariser reads, or from what the agent reads back, leaves a marker in its place saying **what** was removed and **how to recover it**. Summaries are acceptable *only* because they are clearly marked, carry provenance, and act as an index into retained history.
- A bounded, capped, timed-out, failed or partial operation is never presented as complete. An empty result must not read as "there is nothing"; a summary must not read as covering everything; a status must not read as healthy; a tool must not report `complete: true` after hitting a work cap.
- A marker can never itself be trimmed away, and never says less than what was removed.

**b) Optimise for large context windows** (up to 1M tokens) without degrading small ones. Every value that is a *preference* is a smooth weighted interpolation between two anchors — 256k and 1M — resolved through `window_scaling.py`. **Never a band switch, never an `if window > X` branch.**

**c) Take what lossless-claw does better.** [lossless-claw](https://github.com/Martian-Engineering/lossless-claw) (TypeScript, OpenClaw) solves the same problem; where it is better *for this fork's purpose*, port it. **The goal is to be strictly superior to upstream hermes-lcm** — better on the no-loss axis without being worse on any other. See `docs/claw-comparison/`.

## The rule that is most often gotten wrong

> **The 256k anchor carries upstream's TUNING VALUES. It never carries upstream's LOSS.**

Thresholds, chunk sizes, timeouts, concurrency, budgets: at 256k these resolve to upstream's own numbers. Truncation, silent drops, unmarked removals and false completeness claims are removed at **every** window. A cut that fires at 256k but not at 1M is a **defect in this fork**, not fidelity to upstream. The 256k DAG therefore deliberately differs from upstream's wherever upstream truncated.

## Architecture: where the fork's code lives

Fork logic goes in **new modules** so upstream merges stay reviewable; upstream files get small, marked `# fork: betterlcm` hooks.

> **Naming:** the project, the repository and the branch are **better-hermeslcm**. Two identifiers deliberately keep the older `betterlcm` spelling and must NOT be renamed: the in-code marker `# fork: betterlcm` (it is the grep token every audit report and every line of `docs/fork-touchpoints.md` cites) and the migration row `betterlcm_node_meta_v1` (it exists in live databases). Grep for `# fork: betterlcm` to find every fork hook.

Fork modules: `window_scaling.py` (the anchor table and curve), `window_scaled_mixin.py` (`effective_*` resolution), `marked_loss.py` (**every marker the fork emits**), `host_cooldown.py`, `leaf_pipeline.py`, `node_meta.py`, `coverage_doctor.py`, `errors.py`.

The core path — the only code that can lose a conversation, and where effort belongs:

```
ingest → store.py → reconcile.py → compaction.py (leaf + condensation)
       → escalation.py (summariser) → engine.py::_assemble_context → bypass.py
       → tools.py (retrieval) → maintenance / lifecycle / backup
```

**Every upstream file you touch must be recorded in [`docs/fork-touchpoints.md`](docs/fork-touchpoints.md)** with the hook, the reason, and how to re-apply it on a conflict. That file is the merge risk surface, not documentation garnish.

## Things that will bite you

- **The opt-in subsystems are out of scope.** ~20,700 of ~73,500 lines are an upstream question-answering apparatus over the same database: `reasoning.py`/`lcm_compute`, the assertion family, four generations of evidence compiler, adaptive retrieval + query views, `vector_store.py`, the rollups, `trajectory_store.py`. **They stay in, stay default-off, and are neither audited nor modified** — deleting them would only make the upstream merge painful for zero behavioural gain. None of them sits on the path that carries a conversation into the context window. Do not spend effort there.
- **Never assume or mention which summariser model or gateway is configured.** That is the operator's choice; the plugin is provider-neutral and so is every comment and document in it.
- **A receipt must never displace live content.** Markers are positionally neutral or they are budgeted; a receipt that pushes out the caller's newest message has caused loss to prevent loss. This has been a real regression more than once.
- **A receipt is a claim that something was removed.** Do not emit one for a turn that held nothing — a false claim of removal is its own defect.
- **Test edits:** fork tests go in `tests/fork/`. Upstream tests that pin behaviour the fork deliberately changed get **re-pointed and recorded** in `docs/fork-touchpoints.md` — never deleted silently.
- **Edit files with the editor.** Do not patch by piping heredoc Python/sed scripts through the shell: the diff becomes unreadable and those scripts have corrupted source files here before.

## Verification — what counts as evidence

A green suite is a **regression guard**, not evidence. Evidence is end-to-end behaviour and real probes.

```bash
bash scripts/test.sh                          # full suite; umask 077 matters (SQLite refuses group-writable dirs)
bash scripts/test.sh tests/fork/test_x.py     # one file

# the executable form of the no-loss doctrine, run against the DEPLOYED plugin:
python3 scripts/e2e_no_loss.py 262144 400
python3 scripts/e2e_no_loss.py 1000000 3000
```

Both anchors must report **0 unreachable rows, 0 missing facts, 0 facts never offered to the summariser**. A change that cannot produce those numbers is not finished.

## The working loop

1. Fix a batch → full suite green → commit.
2. Redeploy: `~/.hermes/plugins/hermes-lcm` is a clone of this repo (remote `fork`); `git fetch fork better-hermeslcm && git reset --hard FETCH_HEAD`.
3. Re-pin the deployed revision in `~/.hermes/plugins/.install-metadata.json` (it is pinned so `hermes plugins update` refuses to replace it).
4. Re-run **both** e2e anchors against the deployed plugin.
5. Record the pass in `docs/TASKS.md`.

Audits (Codex astra, prompts in `docs/claw-comparison/prompts/`) want a **stable tree** — do not commit while a round is running.

## Standing tasks — these never complete

Both are written up in `docs/TASKS.md` §E, with the outstanding audits in §F:

- **R1 — upstream changed.** Analysis first, merge second. Textual conflicts are the *easy* case; the hazard is everything git merges cleanly while upstream changed something where the fork had already changed something. Ten named failure modes (M1–M10) with what actually catches each.
- **R2 — lossless-claw moved past the analysed v1.0.0.** Diff it; sort changes into capability / behaviour / **bugfix**; treat every claw bugfix as a *lead*, because the same bug very likely exists here in an analogous shape.
- **V1 — is the fork strictly superior to upstream mainline?** Not re-run since audit D. Re-run after every upstream merge.
- **V2 — did we miss anything the current claw does better?** 37 items from the v1.0.0 sweep are still undecided.

## Map of the documents

| file | what it is |
|---|---|
| [`README.md`](README.md) | user-facing; "What the fork changes" is the upstream-vs-fork comparison |
| [`FORK.md`](FORK.md) | maintenance contract: the two rules, the configured exceptions to no-loss, upgrade and claw-tracking procedures, the fork config reference |
| [`docs/TASKS.md`](docs/TASKS.md) | the ledger. Pass history above, **"WHAT IS LEFT TO DO"** below — start there |
| [`docs/fork-design.md`](docs/fork-design.md) | the design and every deviation from it |
| [`docs/fork-touchpoints.md`](docs/fork-touchpoints.md) | every upstream file touched, why, and how to re-apply on conflict — the merge risk surface |
| `docs/claw-comparison/` | the lossless-claw analyses, the audit reports, and the audit prompts |
