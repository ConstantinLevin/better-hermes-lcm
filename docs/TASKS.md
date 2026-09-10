# better-hermeslcm — backlog

The open work, and nothing else. What was done is in the code and in `git log`; a finding is
either implemented, in this list, or nonsense.

## WHAT IS LEFT TO DO

This is the authoritative remaining-work list as of the tenth pass (`52bf344` + the evidence
gate). Everything above this line is history; the partitioned-audit list below it is the raw
material these items were distilled from and contains entries that have since been fixed.

**Ship state:** no known loss on the core path. Suite 3326 passed / 1 skipped / 12 xfailed;
both e2e anchors clean (0 unreachable rows, 0 missing facts, 0 facts never offered to the
summariser) at 262144×400 and 1000000×3000 against the deployed plugin.

Sections: **A** core path (the only category that can still lose something) · **A-bis** trust and truthfulness of a summary · **B** correctness
and operability · **C** ported-capability decisions · **E** routine standing tasks (upstream
sync, lossless-claw watch) · **F** outstanding verification (strict superiority, claw gap) ·
**G** explicitly not being done.

### A. Core path — the only category that can still lose something

| id | what | why it is still open |
|---|---|---|
| A1 | **Summariser prompt design** (Phase 0 below). The prompt is five layers of patchwork with contradictions ("or drop", a 60–70 % focus skew). A summary that omits a topic is loss the markers cannot describe, because the summariser was never told the topic mattered. | Needs a written design (`docs/prompt-design.md`) agreed first, then one fork module `summary_prompts.py`, then measurement — not a quick edit. |
| A2 | **Index-navigation evaluation gate** (Phase 0.3 / Phase 5). There is no number for "can a reader pick the right node to expand from the rendered prefix". `lcm_doctor coverage` measures term survival, not navigability. | Requires a fixture set and a real model run at both anchors, before/after A1. |
| A3 | **A backup is only ever needed before a schema migration, and the migration does not take one.** The rest of this entry used to claim that a restored database references payload files the backup never copied; that is wrong and is written out here so nobody re-derives it. Nothing in the running plugin deletes the SQLite -- the only deleting path is `/lcm doctor clean apply`, which needs `doctor_clean_apply_enabled` (off) and has no candidates unless `ignore_session_patterns`/`stateless_session_patterns` are set (empty). Nothing deletes an externalized payload either: the payload directory has two production consumers and both only read it, there is no retention field in `config.py`, and the only real deletion lives in a hand-run backfill script. So a local restore already works, because the payloads sit untouched beside the DB, and `backup_database` copying SQLite alone is correct. What is actually left: `run_versioned_migrations` runs when the DB is opened, there is no downgrade, and `backup_database` is only ever called from operator commands -- decide whether the pinned deployment (an upgrade is deliberate) is enough, or whether a migration should snapshot first. Physical backup publication and `fsync` durability were never exercised by an audit (read-only workspace), so power-loss recovery stays unverified. |
| A4 | **A pasted copy of our own assembled prefix is classified as scaffolding and dropped.** No content is lost (it is a copy of summaries the DAG holds) but the fact that the user said it is. Rejected fix recorded in the ninth pass. | Needs a trusted generated-message identity from the host, or a digest ledger whose failure mode is worse. Blocked on a host contract. |
| A5 | **verify-4 #6's smallest case**: an assembly budget too small for even the 12-token receipt records the omission in `lcm_status` instead of the prefix. | Deliberate trade-off — making room would drop the caller's own latest message. Revisit only if a host reports it. |
| A6 | **verify-4 #10 structured omission records**: receipts are prose lines, not machine-readable records. | Architectural; would change every marker's shape. |

### A-bis. Trust and truthfulness of what a summary says

| id | what | why it matters |
|---|---|---|
| A8 | **Every rendered summary carries a "this is a summary" disclaimer**, on by default: it says the text is a summary, that it is not to be trusted for a load-bearing or path-dependent decision, and that the reader should expand (`lcm_expand`) until the specific thing is verified against the original. **Before implementing this, decide whether it belongs in the plugin at all**: a repeated per-summary disclaimer costs prompt tokens on every turn and models learn to skim boilerplate. The alternatives are a SKILL (recall/verification guidance the agent loads once) or an addition to the host's `SOUL.md` (a standing instruction about how to treat summarised history). Pick the placement first, with the reasoning written down; the config option is the fallback, not the assumption. If it does land in the plugin it is one line of policy text and a default-on flag, and it must be counted in the assembly budget like any other prefix content. | A summary is the one thing in the context that is *not* what was said. The fork guarantees the original is recoverable; it does not currently tell the reader to go and recover it before betting on the summary. |
| A9 | **The fidelity contract is in the prompt now; what is missing is any way to tell whether it works.** A prompt instruction is a claim, not a guarantee: nothing measures how often a summary still turns "was started" into "succeeded". That measurement is A2's job, and until it exists this is unverified rather than done. | This is the one failure mode no marker can catch: provenance is intact, the node is expandable, and the text is still wrong. |
| A10 | **Closed, and the reasoning is kept here so it is not re-derived.** This claimed the condensation budget (`0.40 × source`, growing 1.6× per level) was the wrong shape and cost the fork most of its published capacity. It is not. Condensation SHRINKS the rendered prefix — four children totalling 32k become one 12.8k parent — and 0.40 is deliberately more generous than the leaf ratio because summary text is already dense. The budget reaches the route as a prompt target and as `max_tokens` at twice that, which is a ceiling, not a demand: a model writes what it writes and a high ceiling costs nothing. The only residual risk is a route that rejects a `max_tokens` above its own output limit, which is the operator's choice of summariser and is named in the README. Nothing to do. | — |

### B. Correctness and operability (no known loss, but unproven or rough)

| id | what |
|---|---|
| B1 | `p03`: a transaction can still lose rollback protection under concurrent use (the `reassign_session_nodes` case is fixed; the general pattern is not audited). |
| B2 | `p06`: Hermes' current spillover directory is not recognised; the always-on ingest guard can canonicalise surrounding JSON. |
| B3 | `p11`: the suite tests rows far more thoroughly than usable provenance — it does not establish that a stored summary is complete or functions as an index. (A1/A2 subsume most of this.) |
| B4 | `audit D` medium: 256k structural equivalence still fails on oversized/imported histories and on the dynamic-chunk path; whole sidecar index blocks escape retrieval budgets. |
| B5 | CI on the `better-hermeslcm` branch; the host-integration lane must fail, not skip, when the host import fails (T2.5). |
| B6 | Docs/skill/defaults generated from `ENV_FIELD_SPECS` + anchors so they cannot drift from the curve (T2.4). |
| B7 | **Make `docs/fork-touchpoints.md` executable (catches M2).** Today it is prose: a human has to notice that upstream renamed the function our hook lived in, because the hook's own unit test still passes while the hook is never reached. Turn the table into a manifest (`docs/fork-touchpoints.yaml` or a dict beside it) naming, per entry, the module, the symbol, and a reachability assertion; then one `tests/fork/test_touchpoints.py` that imports each symbol, fails if it is gone, and — for hooks whose whole point is that they RUN — exercises the real call path and fails if the fork behaviour is absent. Every existing entry needs one; this is the single highest-value piece of merge insurance the fork does not have. |
| B8 | **Anchor-vs-upstream-default check (catches M6).** Every `window_scaling.py` low endpoint claims to *be* upstream's value at 256k, and the README and FORK.md repeat that claim. Nothing enforces it. Add a test that resolves the curve at 262,144 and asserts each non-fraction anchor equals the corresponding upstream `LCMConfig` dataclass default at the pinned `.upstream-base` — so the day upstream moves a default, the fork's central claim fails loudly instead of quietly becoming false. (`tests/fork/test_window_scaling.py::test_at_256k_equals_upstream` hardcodes the numbers today; it should read them from upstream.) |
| B12 | **Delete `docs/fork-touchpoints.md`.** Its `file | hook` columns are the diff, which `git diff $(cat .upstream-base)..HEAD` gives accurately and always current, unlike a hand-maintained table; its `why | on conflict` columns are prose, and most of them say "keep ours". The few genuinely non-obvious re-apply notes ("if upstream adds another consumer of this field, switch it too") belong as comments at the code site, where they cannot drift from the code. The entry point is already in CLAUDE.md: grep `# fork: better-hermeslcm`. Move those notes, delete the file, and change R1 to say "diff against `.upstream-base`, grep the markers". This also sharpens B7: its valuable half was never the table but a test that fails when a hook stops being **reached**, and that test does not need the file. |
| B11 | **The nine upstream docs under `docs/` still describe upstream, not this fork.** They were inherited and never updated, so they now contradict the code an operator is running. Confirmed examples in `docs/operator-guide.md` alone: the threshold given as a flat `0.35` with no mention of the curve (:196), `LCM_FRESH_TAIL_MAX_TOKENS` documented as defaulting to `0` = disabled when it is 0.15 of the window (:198), replay stubbing offered as an option that A12 hard-codes off (:213), and window-sizing advice that predates the curve entirely (:491). `features-overview.md`, `retrieval-tools.md`, `architecture.md` and `agent-config-profiles.md` have the same problem in smaller doses. Decide per file: update it to the fork's behaviour, or delete it and let the README carry that ground. Whatever survives must be checkable — B6 (generate the config tables from `ENV_FIELD_SPECS` and the anchors) is the mechanism, otherwise this recurs on every change. |
| B10 | **`/lcm compact`: leave everything as it is and compact the tail only.** Today the two manual controls are `/new` (start a different session) and `/lcm rotate` (compact everything *except* the protected tail, advancing the frontier past pre-tail raw). Neither is "I am done with this stretch of work — fold it into the index now and give me the room back". That is what a compact command should mean: the session, the DAG, the ownership and the existing summaries stay exactly as they are, and the protected fresh tail (0.15 of the window, ~150k at 1M) is compacted in place. It is the honest alternative to `/new` for the common case, which is not "different work" but "same work, too much recent detail". Confirm the semantics before building: whether it compacts the whole tail or drains it to a target, and what it does when the tail is smaller than one chunk. |
| B9 | **Fork-invariant probe set for merges (supports M7/M8).** The e2e anchors prove no-loss end to end but say nothing about the cross-module contracts the fork *reads* rather than owns: the serialised-message shape, the externalized-placeholder string, replay identity in `reconcile`, tool dispatch, the store row projection, and the migration/classifier interaction between the fork's schema additions and upstream's. Collect one probe per contract so a merge can re-verify them in minutes, and run a merge once against a COPY of a real `lcm.db` rather than only fresh test databases. |

### C. Ported-capability decisions (audit E, still unmade)

| id | what |
|---|---|
| C1 | The nine cheap claw items: test-home isolation, prompt-prefix divergence diagnostics, release-commit validation, payload-reference disambiguators, a script-aware token estimator, prompt inspection commands, copied-reference parsing, shadow-install/drift detection, release fragments. |
| C2 | The six capability-level ones: `context_items` projection, operator TUI, persistent focus briefs, delegated retrieval workers, richer maintenance debt, a paged expansion-cost manifest. |
| C3 | (moved — see R2 under Routine tasks.) |

### E. Routine (standing) tasks — triggered, never "done"

These two have no completion state. They fire on an external event and each run ends with a
dated record, so the next maintainer can see when the fork was last brought level.

**R1 — Upstream changed: bring the fork to the newest upstream state.**
Trigger: any new commit on `stephenschoettler/hermes-lcm` past `.upstream-base`
(currently `8d1b1e6`).

**Textual conflicts are the easy part and the least of it.** `git` reports a conflict only when
two edits touch the same lines. This fork's real exposure is everything git merges *cleanly*:
upstream changing something in a place we already changed, in a different hunk, in a caller, or
in an assumption. That is the standing hazard of maintaining a substantial fork, and it is
invisible to both `git` and a green test suite — our suite tests OUR behaviour, so it stays
green while upstream's new behaviour goes unexercised.

**The failure modes, each with the only thing that actually catches it:**

| # | how a clean merge breaks the fork | what catches it |
|---|---|---|
| M1 | **Semantic conflict, no textual conflict.** Upstream edits a different hunk of a function the fork also edited. Both edits apply; the *combination* is wrong (an upstream early-return placed before our marker; a reordering that makes our guard run after the thing it guards). | Review per FUNCTION, not per hunk: for every symbol named in `docs/fork-touchpoints.md`, read the merged body end to end and re-derive what it now does. A three-way diff of the hunks is not enough. |
| M2 | **The hook silently detaches.** Upstream renames, moves, splits or stops calling the function our hook lives in. Our fork code still exists, still passes its unit test, and never runs. | Every fork hook needs a test that fails when the hook stops being **reached** — not one that only checks the helper's output. This is work item **B7** and does not exist yet. |
| M3 | **Upstream fixes the same bug differently.** Now there are two mechanisms for one problem: two markers on one cut, an elision applied twice, a receipt counted twice, or two competing guards that disagree at the edges. | For every upstream fix, ask "did we already fix this?" before taking it. Keep exactly one mechanism, delete the other, and say in the ledger which survived and why. |
| M4 | **Upstream reintroduces loss somewhere new.** A brand-new code path that truncates, drops, or claims completeness over bounded work. No test of ours covers code that did not exist yesterday. | The doctrine question asked of every added path, plus a scan of the merged tree for the vocabulary of loss (`[:N]`, `[-N:]`, `...`, `truncat`, `limit=`, `break` in a collection loop, `except: pass`, `complete.*True`) in anything upstream added. |
| M5 | **Our fix becomes obsolete or actively wrong.** Upstream restructures the thing we worked around; our patch is now dead weight, or worse, fights the new structure. | The touchpoints table has a "keep ours / re-apply / reconcile" column for exactly this. Removing a fork patch is a legitimate merge outcome; record it. |
| M6 | **Upstream changes a default the curve mirrors.** Every `window_scaling.py` low endpoint claims to BE upstream's value at 256k. If upstream moves a default and the anchor does not, the fork's central claim quietly becomes false. | A test that reads upstream's dataclass defaults at the pinned base and asserts each anchor's low endpoint still equals them. Work item **B8**; does not exist yet. |
| M7 | **A contract we depend on inverts.** Upstream changes the shape of something our code consumes but does not own — the serialised message format, the externalized-placeholder string, replay identity, the tool-schema dispatch path, a store row's projection. Our code keeps parsing the old shape and silently matches nothing. | Enumerate the cross-module contracts the fork reads rather than owns and re-verify each one by probe after the merge. The reconcile/replay-identity path is the sharpest of these. |
| M8 | **Schema and migration divergence.** Upstream adds a column, index or migration next to the fork's own (`envelope_extra`, `host_message_id`, `lcm_node_meta`, `better_hermeslcm_node_meta_v1`). Migration order, classifier logic and downgrade behaviour all interact. | Run a merge against a COPY of a real `lcm.db`, not only fresh test databases, and check both directions (fork build reading an upstream DB and back). |
| M9 | **Test drift in both directions.** Upstream adds tests asserting behaviour we deliberately removed (should fail — good, that is the signal), or edits a test we had re-pointed, and the merge silently restores upstream's assertion. | Every re-pointed or removed upstream test is listed in `docs/fork-touchpoints.md`; after a merge, re-check that list line by line rather than trusting a green run. |
| M10 | **Divergence debt.** Each merge that says "keep ours" without reconciling widens the gap until the next merge is unreviewable. | Merge early and often. A merge deferred until upstream has moved 500 commits is not a merge, it is a rewrite. |

**Procedure — analysis first, merge second:**

1. `git fetch upstream`; read `.upstream-base..upstream/main` in full — runtime diff, the
   surrounding upstream code, and every changed test. Write down what changed and why.
2. Compute the **intersection**: which upstream-changed symbols also appear in
   `docs/fork-touchpoints.md`. That set is the M1/M3/M5 risk surface and gets read function by
   function, in the merged tree, after merging — not as hunks.
3. Classify every upstream hunk: *(a)* file the fork only hooks → keep ours, re-apply the hook
   on top of theirs; *(b)* upstream fixed something we also fixed → reconcile, keep one
   mechanism (M3); *(c)* behaviour the fork depends on → decide explicitly, never by merge
   default (M7); *(d)* new feature → is it coherent with the no-loss doctrine, and does it need
   a curve anchor rather than a fixed 256k-shaped constant?
4. **Ask of every upstream change: does it (re)introduce loss?** New truncation, a new silent
   drop, a new completeness claim over bounded work. If yes, take the feature and remove the
   loss, exactly as the fork already did for the L3 fallback and the 3000-char cut (M4).
5. Walk the whole touchpoints table line by line and confirm each entry is still true of the
   merged tree: the hook exists, it is still reached, and its reason still applies (M2, M5, M9).
6. `bash scripts/test.sh` green. Upstream tests that pin removed behaviour get re-pointed and
   listed in the touchpoints file — never deleted silently.
7. Both e2e anchors clean against the DEPLOYED plugin
   (`scripts/e2e_no_loss.py 262144 400` and `... 1000000 3000`). These are the executable form
   of the fork's invariants: 0 unreachable rows, 0 missing facts, 0 facts never offered to the
   summariser. A merge that cannot produce those numbers is not finished.
8. Redeploy the clone, re-pin the revision, update `.upstream-base`, record the merge in this
   ledger as a new pass — including every "keep ours", every reconciliation and every fork
   patch deleted as obsolete.
9. Re-run **V1**: an upstream merge is exactly when strict superiority can silently break.

**R2 — lossless-claw released or moved: evaluate and port what is better.**
Trigger: any commit or release of lossless-claw newer than the one already analysed —
**v1.0.0** (what that round decided is in sections C and G below; sources were at
`lossless-claw v1.0.0`).

1. Diff the new claw version against the analysed one. Every changed file, not just release
   notes.
2. Sort the changes into: **new capability**, **behaviour change**, and **bugfix**.
3. **A claw bugfix is a lead, not just a port candidate.** Both projects solve the same problem,
   so a bug claw fixed very often exists here in an analogous shape — different code, same
   mistake. For each fix, find the corresponding place in this fork and prove by probe whether
   the same defect is present, then fix it here on its own merits even if the claw patch itself
   is not portable.
4. For each capability/behaviour change, decide **useful AND coherent**: does it serve the
   fork's purpose (no loss, 1M windows, an index into recoverable history), and does it fit the
   architecture without importing an assumption the fork rejects? Rejecting is a valid outcome
   — record the reason.
5. Run the four-aspect comparison (`docs/claw-comparison/prompts/prompt-{1..4}-*.txt`) against
   the new version and record what is ported / rejected / still open as backlog entries here,
   each with a reason. That file is the record that this task ran.
6. Anything ported lands with a fork test and goes through the usual loop (suite → commit →
   redeploy → re-pin → both e2e anchors).

### F. Outstanding verification (never yet run to completion)

| id | what | why it matters |
|---|---|---|
| V1 | **Is this fork strictly superior to upstream mainline?** A dedicated audit that walks every behavioural difference and asks, for each: is the fork at least as good in every respect, or did it trade something away? Audit D did this once against `8d1b1e6` and found three P1 fork regressions; it has **not** been re-run since, and this fork has changed heavily. Known open leads from that round: 256k structural equivalence still fails on oversized/imported histories and on the dynamic-chunk path, and whole sidecar index blocks escape retrieval budgets (B4). | "No loss" is not the only guarantee upstream offers. A fork that fixes loss and quietly loses throughput, a routing behaviour, or a host contract is not strictly better. Re-run after every R1. |
| V2 | **Did we miss anything the CURRENT lossless-claw does better?** The four-aspect comparison and audit E were run against claw **v1.0.0** and 37 audit-E items were left undecided (C1/C2 above). Repeat against whatever version is current, and treat the undecided v1.0.0 items as part of the same question rather than a separate backlog. | The v1.0.0 sweep explicitly found that the earlier "does it change a guarantee" filter had been an invalid reason to reject candidates — so the fork is known to have rejected good ideas for a bad reason at least once. |

Both are audits, not fixes: each ends with a backlog entry here for anything it leaves open,
and nothing written down for what it closes.

### G. Explicitly not being done

- **The opt-in subsystems** (`reasoning`/`lcm_compute`, assertions, the evidence compilers,
  adaptive retrieval + query views, embeddings, rollups, trajectory — ~20,700 lines). Owner's
  decision: they stay in the fork, stay default-off, and are neither audited nor modified.
  Deleting them would only make the upstream merge painful for no behavioural gain. The whole
  verify-5 subsystem tail (embeddings association/coverage, trajectory recovery and indexing,
  rollup snapshots, query-view freshness, assertion polarity and scope, verify-4 #23, #26) is
  closed as "not our problem" under that decision.
- **Reproducing upstream's cuts at 256k.** The anchor carries tuning values only.

## Still open — distilled from the audit rounds

Fourteen audit rounds ran against this fork: four aspect comparisons with lossless-claw v1.0.0,
three cross-cutting (whole-plugin, plan critique, unplanned claw capabilities), a regression hunt
against upstream, eleven exhaustive partitions covering every module, the non-code assets and
the test suite, and six verification rounds.

**The reports themselves are not part of this repository.** A finding is either worked in — and
then the CODE is where it lives, not a document describing it — or it is still open, and then it
belongs in this backlog. A directory of 6,500 lines of old findings is neither. Everything below
is what those rounds left unfixed; it is the backlog, not a pointer to one.

- **p01 engine-core** — publication and session ownership not coupled; late work can publish
  under a different session; indexes that describe only the ends of their sources.
- **p02 agent-tools** — tools that mistake incomplete work for complete results (eleven items
  reported as a complete ten; an orphan check passing after 1,000 nodes).
- **p03 operator-config** — the stranding case is closed (C01); the transaction can still lose
  rollback protection under concurrent use.
- **p04 storage-schema** — the raw store is a projection, not an archive (host fields dropped);
  reconciliation can discard new occurrences and attach wrong provenance.
- **p05 compaction-flow** — closed in the second pass above, except RS02 (reset/rebind is not
  fenced against in-flight compaction) and the two deferred items named there (MA01, MC01).
- **p06 payloads-embeddings** — Hermes' current spillover directory is not recognised; the
  always-on ingest guard can canonicalise surrounding JSON; embedding workers can publish
  offsets against replaced text.
- **p07 rollups-assertions** — adaptive retrieval can certify incomplete evidence; the assertion
  sidecar can publish unsupported state and present it as current.
- **p08/p09 trajectory, evidence** — "grounded"/"verified"/"sufficient" do not reliably mean the
  evidence supports the answer; limits can change the answer or erase its recovery handles.
- **p11 tests** — rows are tested far more thoroughly than usable provenance; the suite does not
  establish that a stored summary is complete or functions as an index.
- **audit B (plan critique)** — one publication contract (validated state transition) subsumes
  several planned tasks; the prompt rewrite must come *after* source completeness and the
  generation contract; the host fabricates `finish_reason`, so B1 needs a host contract.
- **audit C** — the message envelope and a durable ingest receipt keyed by host event identity.
- **audit E (unfiltered claw sweep)** — 37 items, none previously planned. It states that the
  "does it change a guarantee" filter applied to the first claw comparison was an invalid
  reason to reject candidates. Nine are cheap and valuable (test-home isolation, prompt-prefix
  divergence diagnostics, release-commit validation, payload-reference disambiguators, a
  script-aware token estimator, prompt inspection commands, copied-reference parsing, shadow-
  install/drift detection, release fragments); six are capability-level decisions
  (`context_items` projection, operator TUI, persistent focus briefs, delegated retrieval
  workers, richer maintenance debt, a paged expansion-cost manifest).
- **audit D (regressions vs upstream)** — landed. Three regressions the fork introduced: the
  condensation-failure defect above (now fixed), the lookahead losing the host's routing
  context / deadline / cancellation scope with an unbounded wait that can hold the compaction
  lock, and the 1M leaf production rate (up to 64 per call) outrunning the retained
  one-group-per-depth condensation schedule. Plus two medium: 256k structural equivalence still
  fails on oversized/imported histories and on the dynamic-chunk path, and whole sidecar index
  blocks now escape retrieval budgets.

Tracked as tasks #1-#12 in this session's task list; the ~375 ranked findings across the nine
partition reports are not yet individually triaged.

## H. Older open items, not yet triaged into the sections above

Each still needs a decision before it is worth a full entry; none is a loss defect.

| id | task |
|---|---|
| T2.3 | Sanitiser: keep inline literal `<think>` tags in quoted text, strip only trailing standalone reasoning. The current rule strips both; a quoted example in a code block can lose its tags. |
| T2.6 | Transcript GC must compare exact original text before any rewrite (a sanitised-only match must not trigger one). The GC is opt-in (`large_output_transcript_gc_enabled`). |
| T3.1 | Timestamps and child provenance in summariser input and node headers, with observed time kept distinct from ingested time. |
| T3.5 | Condensation input bounded by tokens as well as by group count, oldest-first — four 12k children should not become one 48k request. |
| T4.1-T4.5 | Page-bounded expansion I/O; a payload catalog instead of the 240-char head; FTS optional at bootstrap; a read-only operator entry point that does not start an engine; running-total prefix selection. |
| T5.1-T5.3 | A durable prepare-then-publish pipeline (the biggest 1M win); repair of degraded historical summaries from their sources; the evaluation harness as a standing quality gate (feeds A2). |

## How this list stays honest
- Every task gets an adversarial check ("how can the index still lie after this?"), not a
  plan-conformance check.
- No task is closed with a full-suite run that was started before its last edit.
- Nothing is recorded here that is already in the code or in `git log`.

## W1 — every setting must be a WEIGHT that is right at every window, not two ends and a line

This is the biggest open design item and it is written out in full because the session that
found it ran out of context. Read all of it before touching `window_scaling.py`.

### What exists today

`window_scaling.py` holds a table of `Anchor(name, field, low, high)` rows. Each row is **two
numbers** — the setting's value at a 262,144-token window and its value at a 1,000,000-token
window — and the resolver draws a straight line between them:

    t = clamp((W - 262_144) / (1_000_000 - 262_144), 0, 1)
    value = low + t * (high - low)

"Anchor" is that pair of endpoints. The word and the shape are both an invention of the
implementation, not a requirement of the design.

### What it is supposed to be

**One weight per setting: a function of the window that produces the right value at EVERY
window size.** Not a value fitted at 256k, another fitted at 1M, and whatever a straight line
gives in between. Three consequences the current shape does not honour:

1. **A weight need not be linear.** Nothing says the correct value moves in a straight line
   with the window. Linearity is an assumption baked into the resolver, never a decision about
   any individual setting.
2. **A weight is often DERIVED from another setting**, and then it is that derivation — not a
   pair of endpoints — that must hold everywhere.
3. **256k and 1M are two points you can check, not the two points the design is fitted to.**
   A 400k or 700k model is a first-class case, not an interpolation artefact.

### The evidence that the current shape is wrong

Resolved values across the range, measured:

| setting | 256k | 400k | 512k | 700k | 1M |
|---|---|---|---|---|---|
| `context_threshold` | 0.35 | 0.43 | 0.50 | 0.62 | 0.80 |
| `incremental_max_depth` | 3 | 3 | 4 | 4 | 5 |
| `summary_timeout_ms` | 60,000 | 86,157 | 107,407 | 143,078 | 200,000 |
| `leaf_pass_cap` | 16 | 25 | 32 | 44 | 64 |
| `summary_spend_max_calls` | 80 | 125 | 161 | 222 | 320 |

Three distinct failures in that table:

- **Nobody chose the middle.** `context_threshold` is 0.62 at 700k purely because that is where
  the line passes. If 0.62 is wrong for a 700k model, nothing in the design would ever say so.
- **Linear is the wrong KIND of function.** `incremental_max_depth` interpolates a discrete tree
  depth: 3, 3, 4, 4, 5. Depth should be *derived* — a window holds about 25 leaves (the chunk is
  0.04·W), and the depth needed to index N leaves is about log(N)/log(fanin). Likewise
  `summary_timeout_ms` runs on its own line from 60 s when the work in one call is the chunk,
  which is proportional to W: the timeout should follow the chunk, not a separate line.
- **A derivation that only holds by luck.** `summary_spend_max_calls` is *meant* to be
  `4 * (leaf_pass_cap + condense_group_cap)`. Measured, it is 80/80, 125/124, 161/160, 222/220,
  320/320 — consistent, but only because both settings happen to be straight lines and therefore
  stay parallel. Change one endpoint and the relation breaks at every window except the two ends,
  which is exactly where the tests look.

### Why this stayed invisible, which matters more than the defect

The tests could not have caught it. `test_the_low_anchor_takes_upstreams_tuning_and_rejects_its_losses`
and `test_at_1m_equals_design` check the two ends. `test_every_anchor_is_exactly_linear_between_its_endpoints`
checks that intermediate values sit on the line — that is, it verifies the interpolation is
linear; it can never say whether any intermediate value is a value a human would choose. It is a
test written from the assumption, so it can only ever confirm the assumption.

This is the same failure as the leaf chunk: a choice was encoded in a test, the test then made
the choice look like a requirement, and four audit rounds went past it. **When touching this
area, delete the tests that pin the current shape rather than making them pass.**

### What the work is

1. Decide the shape: each setting expressed as a function of W (constant, fraction of W, derived
   from another setting, or an explicit non-linear curve), with the reason recorded next to it.
2. Re-derive the fifteen settings that currently differ at the two ends and are therefore fitted
   rather than chosen: `context_threshold`, `leaf_pass_cap`, `incremental_max_depth`,
   `summary_timeout_ms`, `expansion_timeout_ms`, `leaf_loop_max_seconds`,
   `summary_spend_max_calls`, `condense_group_cap`, the circuit-breaker threshold,
   `l2_budget_ratio`, `stub_threshold_tokens`, `expansion_context_tokens`, `expand_page_tokens`,
   `tool_response_char_scale`, and the two cache sizes.
   (The ones already expressed as a single weight, and therefore fine: leaf chunk 0.04·W, fresh
   tail 0.15·W, condensation gate 0.20·W, drain stop 0.30, plus three flat values — concurrency
   6, tail count 400, and the per-message char cap at 4 chars per token of window.)
3. Rename accordingly. `Anchor(name, low, high)` is the wrong shape and "anchor" is the wrong
   word once a setting is a weight; `window_scaling.py` should express the function per setting.
4. Verify at several windows including ones nobody designed for — 128k, 400k, 700k, 2M — not at
   the two ends.

**A10 (the condensation budget) is one of these**, so settle this first: `0.40 × children`
compounds per level, and what it should be depends on what shape settings take.
