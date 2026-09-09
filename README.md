<p align="center">
  <img src="docs/banner.png" alt="HERMES-LCM" width="800">
</p>

[![Python 3.11-3.14](https://img.shields.io/badge/Python-3.11--3.14-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# better-hermeslcm

**A fork of [stephenschoettler/hermes-lcm](https://github.com/stephenschoettler/hermes-lcm)** —
the Lossless Context Management plugin for
[Hermes Agent](https://github.com/NousResearch/hermes-agent) — that actually keeps the promise
in its name, and works at a 1M-token context window.

Upstream says *"Bounded context, unbounded memory. Nothing is ever lost."* It is a good design
and mostly delivers. But it truncates in five places, drops host fields at the door, reports
bounded work as complete, and is sized in fixed token constants tuned for ~128k–272k windows —
so at 1M it compacts at a third of the window, into a single summariser call, behind a 32-message
tail. This fork fixes all of that. **[What upstream does badly](#what-upstream-does-badly)** is
the honest list; **[What the fork changes](#what-the-fork-changes)** is what was done about it.

Three goals, in priority order:

1. **Opinionated hatred of loss and truncation.** Nothing that reaches the plugin becomes
   unreachable; anything removed leaves a marker saying what went and how to get it back;
   bounded, capped, failed or timed-out work is never reported as complete. This overrides
   convenience, elegance and upstream fidelity.
2. **Work at large context windows** (up to 1M) without degrading small ones — every tuning
   value is a smooth weighted interpolation between a 256k and a 1M anchor, never a band switch.
3. **Take what [lossless-claw](https://github.com/martian-engineering/lossless-claw) does
   better**, with *strictly superior to upstream* as the bar.

Working on this? Read [`CLAUDE.md`](CLAUDE.md) first, then [`FORK.md`](FORK.md) (maintenance
contract) and [`docs/TASKS.md`](docs/TASKS.md) → "WHAT IS LEFT TO DO".

`hermes-lcm` replaces one-shot active-context compression with a SQLite-backed,
DAG-based context engine. It keeps the live prompt bounded, preserves raw
messages, and gives the agent tools to recover exact detail after compaction.

Based on the [LCM paper](https://papers.voltropy.com/LCM) by Ehrlich & Blackman
(Voltropy PBC, Feb 2026). Inspired by
[lossless-claw](https://github.com/martian-engineering/lossless-claw) for
OpenClaw. For an interactive visualization of the LCM idea, see
[losslesscontext.ai](https://losslesscontext.ai/).

## Table of contents

- [What it does](#what-it-does)
- [What upstream does badly](#what-upstream-does-badly)
- [What the fork changes](#what-the-fork-changes)
- [Current limitations](#current-limitations)
- [LCM vs built-in compression](#lcm-vs-built-in-compression)
- [Quick start](#quick-start)
- [Commands and tools](#commands-and-tools)
- [Recall skill and policy](#recall-skill-and-policy)
- [Configuration](#configuration)
- [Retrieval contract](#retrieval-contract)
- [OpenClaw/lossless-claw import](#openclawlossless-claw-import)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Documentation](#documentation)
- [Development](#development)
- [Fork maintenance](#fork-maintenance)
- [Contributing](#contributing)
- [License](#license)

## What it does

Hermes Agent's built-in compressor is a practical continuity layer: when the
prompt crosses its configured threshold, it prunes older tool results, asks an
auxiliary model to summarize the middle/older conversation, and rebuilds the
active prompt from that summary plus a protected recent tail. The original
session rows can still live in Hermes `state.db` and remain searchable through
host tools such as `session_search`, but the model's active context no longer
contains the compacted turns verbatim or a structured drill-down path back to
them.

`hermes-lcm` instead:

1. **Persists messages** in a plugin-local SQLite store with FTS metadata.
2. **Compacts older context** into depth-aware summary nodes.
3. **Condenses summaries** into a hierarchical DAG as they accumulate.
4. **Assembles active context** from system prompt, highest-value summaries, and
   the protected fresh tail.
5. **Provides recall tools** so agents can search, inspect, and expand compacted
   material without flooding the main prompt.

Nothing is lost in normal operation. Raw messages stay recoverable in bounded
pages, summaries retain source lineage, and oversized externalized payloads keep
stable refs for later expansion.

<p align="center">
  <img src="docs/standard_compression.png" alt="Standard compression" width="700">
</p>

<p align="center">
  <img src="docs/lcm_compression.png" alt="LCM compression" width="700">
</p>

Core capabilities:

- **SQLite message store** - preserves raw messages before compaction
- **Summary DAG** - builds depth-aware summary nodes over compacted history
- **Bounded recovery** - pages raw messages, child summaries, and externalized
  payloads instead of dumping everything into the prompt
- **Agent tools** - `lcm_grep`, `lcm_recall`, `lcm_query_state`, `lcm_compute`, `lcm_compile_evidence`, `lcm_evidence_pack`, `lcm_retrieve`, `lcm_recent`, `lcm_load_session`,
  `lcm_describe`, `lcm_expand`, `lcm_expand_query`, `lcm_status`, `lcm_inspect`,
  and `lcm_doctor`
- **Source-aware retrieval** - filters raw rows and summaries by descendant
  source lineage
- **Session controls** - ignore noisy sessions or keep sessions read-only with
  glob patterns
- **Large payload controls** - externalize oversized tool/media/raw payloads and
  protect SQLite from inline media-ish base64 blobs
- **Sensitive-pattern controls** - optional named redaction of API keys, bearer
  tokens, passwords, and private keys before LCM stores or summarizes them
- **Diagnostics** - runtime health, database checks, optional `/lcm` slash
  commands, backup-first repair/rotate paths

Beyond the core loop, three opt-in (default-off) feature families extend LCM
from a compression layer into a memory system: **large-output externalization
and context-budget controls** (giant tool results move to recoverable refs
instead of crowding the prompt), **temporal memory** (day/week/month rollups
plus natural-time recall through `lcm_recent`), and **semantic retrieval**
(embedding-backed `lcm_grep` semantic/hybrid modes with free-tier cloud or
fully-local providers). See the
[Feature overview](docs/features-overview.md) for what each family does and
why, and [Agent configuration profiles](docs/agent-config-profiles.md) for
copy-paste setups per agent type.

## What upstream does badly

None of this is a swipe at upstream — it is a good design, and this fork keeps its architecture
almost entirely. But a fork needs a reason, so here is the honest list.

**1. It is not built for a 1M-token window.** Every size is a fixed token constant chosen for
~128k–272k. Point it at a 1M model and the constants do not scale with it:

- compaction fires at `0.35 × window` — **350,000 tokens**, leaving two thirds of a 1M window
  unused while the agent pays summariser latency it did not need to pay;
- the leaf chunk is the **whole backlog in one summariser call**, so that first compaction is a
  single request over hundreds of thousands of tokens — slow, expensive, and prone to the
  output-limit failure below;
- the protected fresh tail stays at **32 messages** — on a 1M window, a rounding error;
- condensation triggers on a **count rule** (every 4th leaf) rather than on how much summary
  actually accumulated, and the DAG depth cap stays at 3;
- one summariser call in flight, a 60 s timeout and a 24-call spend guard, sized for a backlog
  an order of magnitude smaller.

**2. "Nothing is ever lost" is not upheld when things go wrong.** Upstream's last-resort path
when every summariser route fails is **deterministic truncation** — it writes a chopped
"summary" and moves on. That is precisely the moment the guarantee is supposed to matter.

**3. It truncates before the summariser ever sees the text.** Every message over 3,000 chars is
cut to head 2,000 + tail 800, and tool-call arguments over 500 chars to 400 — **unmarked**, so
neither the summariser nor the reader can tell that a command, a stack trace or a decision was
sliced in half. With externalization disabled or its directory unwritable, oversized tool output
is cut inline too.

**4. The raw store is a projection, not an archive.** Only the columns the schema knows about
survive ingest. Everything else the host sent — `reasoning_content`, `is_error`, `exit_code`,
provider ids, tool metadata — is dropped at the door, so a failed step reads exactly like a
successful one. An edited message overwrites rather than supersedes.

**5. Bounded work is reported as complete.** A search that hit a work cap, an expansion that
could not read a source row or a recorded child node, a corrupt externalized payload, a field
left unread by a page budget — all of these could still come back as `complete: true` or as an
empty result that reads like "there is nothing here". An index that lies about its own coverage
is worse than no index.

**6. Removals inside the active context leave no trace.** Internal reasoning (`<think>` blocks,
reasoning parts) is stripped from every replayed assistant turn with nothing in its place;
assistant tool calls whose result is not in the same chunk are dropped from the summariser's
input; whole turns judged "acknowledgement-shaped" disappear by wording alone.

## What the fork changes

The fork keeps upstream's architecture and changes two things:

> **The 256k anchor carries upstream's TUNING VALUES, never upstream's loss.** Everything that
> is a *preference* (thresholds, chunk sizes, timeouts, concurrency) resolves to upstream's own
> number at 256k. Everything that is *loss* — truncation, silent drops, unmarked removals,
> completeness claims over work that was cut short — is removed at **every** window. A cut that
> fires at 256k but not at 1M is a defect in this fork, not fidelity to upstream.

> The bold rows are where the fork's 256k column differs from upstream. Upstream is the
> **floor** — never worse than it — not the target. A value takes upstream's number at 256k only
> when it is a genuine preference (a cost, latency or headroom tradeoff). Where it decides how
> much is lost, or how coarse the index is, it is decided on merit at every window.

**1. Every tuning value is a smooth function of the model's context window.**
`t = clamp((W − 256k) / (1M − 256k), 0, 1)`; each setting is `upstream_value + t × (large_window_value − upstream_value)`.
At 256k the resolved values *are* upstream's, so a fixture session produces the same DAG
*structure* under upstream and the fork — the anchor is about sizing, not about reproducing
upstream's cuts (see the note above). At 1M they are the large-window design; between, they
slide.
Anchors live in one table, [`window_scaling.py`](window_scaling.py); explicit env/config values
always win over the curve; `lcm_status → window_scaling` shows every resolved value and its source.

Read the table with the middle column in mind: **upstream's value is the same number at 256k and
at 1M** — that is the whole problem. The fork matches it at 256k for everything that is a
*preference*, and differs there for everything that is *loss*.

| setting | upstream (any window) | fork @ 256k | fork @ 1M |
|---|---|---|---|
| compaction threshold (`LCM_CONTEXT_THRESHOLD` default) | 0.35 | 0.35 | 0.80 |
| **leaf chunk per summariser call** | **the whole backlog in one pass — one node standing for everything** | **4 % of the window (~10,500 tokens), up to 16 passes per compaction** | **4 % of the window (40,000 tokens), up to 64 passes, draining to 30 % of the window** |
| **summariser calls in flight** | **1 (upstream has one chunk)** | **6** | **6** — persistence stays sequential, so the published DAG is identical |
| **protected fresh tail** | **32 messages, no token cap — what it protects depends on how long the messages are** | **15 % of the window (~39,000 tokens), max 400 messages** | **15 % of the window (150,000 tokens), max 400 messages** |
| **condensation trigger** | **every 4th leaf, whatever those leaves are worth** | **once the summary pile exceeds 20 % of the window (~52,000 tokens), oldest first** | **once the pile exceeds 20 % of the window (200,000 tokens), oldest first** |
| DAG depth cap | 3 | 3 | 5 |
| summariser / expansion timeouts, leaf-loop wall clock | 60 s / 120 s / 120 s | same as upstream | 200 s / 200 s / 200 s |
| **summariser spend guard** / breaker | **24 calls per 10 min, 2 failures** | **80 calls, 2 failures** | **320 calls, 4 failures** — the guard counts CALLS, and chunking makes the same work cost many small ones; measured in TOKENS these are below upstream's spend at both ends |
| `lcm_expand` page, tool response caps, SQLite/token caches | 4k tokens, ×1, 2 MiB / 2048 | same as upstream | 32k tokens, ×4, 64 MiB / 8192 |
| **pre-summariser per-message cap** | **3000 chars (head 2000 + tail 800), unmarked** | **none — the cap is the whole window** | **none** |
| **tool-call argument cap** | **500 chars → 400, unmarked** | **none — shares the message cap** | **none** |
| **inline fallback when externalization is off/unwritable** | **cut to 3000 chars** | **body stays whole** | **body stays whole** |
| **every summariser route failed** | **deterministic truncation (L3): a chopped summary is published** | **raw context kept, host-visible cooldown armed, turn continues** | **same** |
| **internal reasoning stripped from a replayed turn** | **removed silently** | **the turn carries its own receipt** | **same** |
| **tool call with no result in this chunk** | **dropped from summariser input** | **serialised and marked** | **same** |
| **host envelope fields (`is_error`, `exit_code`, reasoning, provider ids)** | **dropped at ingest** | **stored, rendered or named with a recovery route** | **same** |
| **bounded / failed / capped work** | **can report `complete: true`** | **`complete: false` with the reason** | **same** |

Everything bold is loss or index quality, so it is fixed at every window. Everything unbolded is
a preference, and there upstream's number is as good as any other.

**2. No unmarked loss, at any window** (pure changes, identical everywhere):

- **Truncation is gone, at every window.** The deterministic-truncation fallback (upstream's
  "L3") is removed: when every summariser route fails the raw context stays in place, the engine
  arms a host-visible cooldown (the host prints its usual `cooldown:<s>` warning) and the turn
  continues — a compaction can never write a chopped "summary" or kill a turn. Upstream's other
  two cuts are gone as well: the pre-summariser per-message cut (3000 chars = head 2000 + tail
  800, and tool-call arguments 500 → 400) and the inline fallback cut when externalization is
  disabled or its directory is unwritable. An operator who *sets* a cap explicitly still gets
  one, and it still cuts only through a sized `[LCM elided …]` marker that carries any earlier
  receipt its span crossed.
- **The raw store is an archive, not a projection.** Everything the host sent is kept: fields
  the columns do not hold live in an `envelope` (name, reasoning metadata, `is_error`,
  `exit_code`, provider ids), a stable host message id makes an edited message archivable as a
  revision row that supersedes — never overwrites — the original, and envelope JSON that is
  corrupt is preserved verbatim and paged rather than cut. A host edit that cannot be archived
  is an ingest failure, so retrieval never answers an exhaustive negative over it.
- **Internal reasoning removed from a replay says so.** Upstream stripped `<think>` and
  reasoning blocks out of every assistant turn it replays and left nothing behind. The turn now
  carries its own receipt, in its own position, so it can never displace the newest message —
  and a turn that held nothing at all is still dropped without a receipt, because inventing one
  would be a false claim of removal.
- Every remaining cut or drop is marked and points at its provenance: sized `[LCM elided …]`
  markers in summariser input, unmatched tool calls serialised (not dropped), externalized
  stubs carry a head note, assembly renders the whole frontier and names anything omitted,
  `/new` keeps index nodes (retain depth is a carry-over filter, not a delete), `/lcm rotate`
  writes a marker node over rotated raw, bypass trims are marked.
- Summaries are written as **indexes into recoverable history**: the prompts require coverage
  of decisions and rationale, rejected approaches, constraints, identifiers/paths/values,
  errors, tool-output contents, end state and open items — "exceed the target rather than omit
  an item". The whole `Expand for details about:` block is stored per node (`lcm_node_meta`
  sidecar, with the escalation level) and returned on `lcm_grep`/`lcm_describe`/expand
  results; the summary header shows `[L2 bullet summary]` when the thinner form was used.
- Recovery: `lcm_expand(node_id=…, hydrate=true)` returns externalized tool outputs inline;
  empty hints still point at `lcm_expand(node_id=N)`; the system note tells the model that
  absence from the visible context is never absence from the record.
- `lcm_doctor {"coverage": true}` / `/lcm doctor coverage` measure how much of each node's
  sources (paths, identifiers, quoted strings, numbers, decision keywords) is still
  discoverable from its summary — the executable definition of "no loss".
- **Bounded work is never reported as complete.** A search that hit a work cap, an expansion
  that could not read a source row or a recorded child node, a corrupt externalized payload, a
  field left unread by a page budget, a timed-out retrieval — each makes the answer
  `complete: false` and says why. Expansion pages body, tool calls and envelope against one
  budget with one cursor per field, so a continuation never re-sends what the caller already
  has and never stops while something is unread.
- **One publication contract.** A node and its sidecar are written in a single transaction; the
  raw frontier only advances over a proven contiguous run of covered rows; every result branch
  is fenced against a session rebind landing mid-compaction, and ingest files its rows under the
  session it started in. The summary frontier is a SQL predicate, so it is not capped at a page
  of nodes.
- Chunk boundaries never split an assistant tool call from its results; a compaction lock keeps
  a host-abandoned worker from writing concurrently with the retry; hot paths (frontier token
  projection, per-pass metadata reads) are cheaper.

**Upstream's opt-in subsystems are kept, and stay off.** About 20,700 of the plugin's ~73,500
lines are an upstream question-answering apparatus over the same database — `lcm_compute`
(a calculator that refuses anything not verbatim in a cited span), typed assertions
(`lcm_query_state`), four generations of pre-answer evidence compiler (`lcm_compile_evidence`,
`lcm_evidence_pack`), adaptive retrieval and query views (`lcm_retrieve`), summary embeddings,
temporal rollups and a trajectory corpus. None of them sits on the path that carries a
conversation into the context window, so none can drop or shorten a message. The fork keeps them
untouched so upstream merges stay clean; every one of them answers `status: disabled` until its
flag is set, except `lcm_compute`, which the model may call at any time.

Costs at 256k: the system note is ~54 tokens longer and index-style summaries tend to be longer
than upstream's terse ones (still under the same 12k cap). Every *tuning* value at 256k is
upstream's; the loss removal above applies there too, so the 256k DAG deliberately differs from
upstream's wherever upstream truncated.

## Current limitations

**How long a session can actually get.** The DAG has a depth cap (3 at 256k, 5 at 1M), and once
a node reaches it nothing can condense it further — top-depth nodes accumulate in the frontier,
and the frontier is rendered into every prompt. So the ceiling is *how much conversation the
top-depth nodes can stand for before they no longer fit alongside the protected tail*.

With the defaults (chunk 4 % of the window, leaf ratio 0.20, condensation fanin 4, condensation
ratio 0.40, tail 15 % of the window):

| window | one top-depth node | it stands for | how many fit | total conversation |
|---|---|---|---|---|
| 256k, depth 3 | 8,589 tokens | 671,104 tokens | 6 | **~4.0M tokens** |
| 1M, depth 5 | 83,886 tokens | 40,960,000 tokens | 7 | **~287M tokens** |

Without LCM the session ends when the conversation reaches the window: **262,144** and
**1,000,000** tokens. So the arithmetic says roughly **15×** at 256k and **287×** at 1M.

**The number you should actually plan around is lower**, because of a limitation this fork has
not fixed: the condensation budget is `0.40 × source` with no ceiling, so it grows with depth,
and it is a request for *output* tokens. At 1M that is 32,768 tokens at depth 3, 52,429 at depth
4 and 83,886 at depth 5. Most summarisers cannot emit that much in one response. This fork
refuses a truncated generation rather than storing a chopped node, so the effect is not a
corrupt index — condensation simply stops succeeding and the DAG stalls at whatever depth the
model can still write. **Stalling at depth 3 gives ~48.6M tokens at 1M** (19 nodes × 2.56M),
still ~49× the window. At 256k the deepest request is 8,589 tokens, comfortably within any
model, so 256k reaches its cap.

**What happens at the ceiling is degradation, not an ending.** When the frontier outgrows the
prefix, assembly renders what fits and names what it left out; the omitted nodes stay in the DAG
and stay reachable through `lcm_grep`, `lcm_describe` and `lcm_expand`. Nothing is lost — the
*rendered* index becomes partial and the agent has to search for the rest instead of seeing it.
Without LCM the session simply ends.

Two caveats on the table: it is arithmetic from the default settings, not a measurement, and it
assumes every leaf is a full chunk and every condensation group is full. Real sessions produce
partial chunks and partial groups, so treat these as an upper bound on the same order of
magnitude.

## LCM vs built-in compression

Hermes core may persist original conversation history in `state.db` before
built-in compression rewrites the active prompt. Built-in compression can still
be lossy in the active context, but previous content may be recoverable later
through host-level history tools such as `session_search`.

`hermes-lcm` is different because recall is part of the active context engine:

- plugin-local store and DAG built specifically for drill-down
- current-session retrieval through LCM tools, not an auxiliary cross-session
  search step
- explicit source-lineage and session-boundary rules

Position LCM around retrieval quality, autonomy, and drill-down behavior. Do not
claim that Hermes core has no persisted record of pre-compression history.

## Quick start

### Prerequisites

- Hermes Agent
- Python 3.11+
- No required third-party runtime dependencies

`tiktoken` is used if available; otherwise LCM falls back to character-based
token estimates. `regex` is used if available to apply timeouts to message ignore
patterns; without it, message-level regex filtering is disabled with a warning
rather than running unbounded stdlib `re` matches.

The versioned [host-owned dependency contract](docs/dependency-assurance.md)
lists every shipped runtime import, supported host/Python versions, ownership,
and the mechanical validation command. It is an assurance boundary, not a claim
that the host's resolved environment is free of known vulnerabilities.

### Install the plugin

Clone the plugin as a general user plugin, then **pin it** so `hermes plugins update` cannot
replace it with upstream:

```bash
git clone -b better-hermeslcm https://github.com/ConstantinLevin/better-hermeslcm \
  ~/.hermes/plugins/hermes-lcm
# pin: ~/.hermes/plugins/.install-metadata.json ->
#   {"hermes-lcm": {"pinned": true, "revision": "<git rev-parse HEAD>",
#                   "source": "https://github.com/ConstantinLevin/better-hermeslcm"}}
```

The directory must stay named `hermes-lcm` — that is the plugin id Hermes loads. Pinning is not
optional: without it an update silently swaps this fork for upstream and every guarantee above
goes with it. Upstream's own install is the same command against
`https://github.com/stephenschoettler/hermes-lcm`.

For a profile-specific install:

```bash
git clone -b better-hermeslcm https://github.com/ConstantinLevin/better-hermeslcm \
  ~/.hermes/profiles/myprofile/plugins/hermes-lcm
```

From an existing checkout, install a symlink:

```bash
./scripts/install.sh

# Optional profile-aware install:
HERMES_PROFILE=myprofile ./scripts/install.sh
```

Run `scripts/install.sh` even when the checkout already lives at the canonical
plugin path. It leaves that checkout in place and exposes the bundled
`hermes-lcm` skill in the matching global/profile `skills/` directory. The
installer preflights both paths and refuses conflicts before creating links.

### Activate it

The plugin has two names:

- plugin manifest name: `hermes-lcm`
- runtime context engine name: `lcm`

Both must be configured:

```yaml
plugins:
  enabled:
    - hermes-lcm

context:
  engine: lcm
```

Restart Hermes after changing plugin or context-engine config.

### Verify it loaded

Run:

```bash
hermes plugins
```

Expected signals:

- plugin list includes `hermes-lcm`
- selected context engine is `lcm`
- tool list includes `lcm_grep`, `lcm_recall`, `lcm_recent`,
  `lcm_load_session`, `lcm_describe`, `lcm_expand`, `lcm_expand_query`,
  `lcm_status`, `lcm_inspect`, and `lcm_doctor`
- the normal available-skills index includes `hermes-lcm`; current hosts can
  also resolve the explicit plugin-qualified skill `hermes-lcm:hermes-lcm`

Typical output:

```text
Plugins (1):
  ✓ hermes-lcm v1.0.0-rc.1 (15 tools)

Provider Plugins:
  Context Engine: lcm
```

For source checkouts, `lcm_status`, `/lcm status`, `lcm_inspect`,
`lcm_doctor`, and `/lcm doctor` also report the loaded plugin path and
best-effort git identity:
`plugin_git_commit`, `plugin_git_branch`, and `plugin_git_dirty`.

If startup logs say LCM tools are available through `context-engine schemas` or
mention the `Path B fallback`, that is expected on older Hermes hosts such as
Hermes Agent v0.16. All 15 `lcm_*` tools remain available through the
context-engine path; standalone plugin-registry registration is not required
there.

### Update it

Fork checkouts: see [Fork maintenance](#fork-maintenance) — upstream changes are merged in the
fork repository first, tested, then pulled into the plugin directory and re-pinned.

Upstream checkouts, if you cloned directly into the plugin directory:

```bash
cd ~/.hermes/plugins/hermes-lcm && git pull --ff-only
```

For a profile-specific install:

```bash
cd ~/.hermes/profiles/myprofile/plugins/hermes-lcm && git pull --ff-only
```

If you installed a symlink from a separate checkout:

```bash
./scripts/update.sh
```

Restart Hermes after updating.

For the `v1.0.0-rc.1` line, take a normal backup of `lcm.db` before updating,
then update the checkout and restart Hermes. No manual core migration or
backfill is required: the core schema remains version 5. New assertion,
query-view, and adaptive-retrieval state is additive, created only after the
corresponding opt-in is enabled, and stored in the same profile database under
named feature markers. The five new query/evidence tool schemas are visible in
the tool list on stock installs, but automatic extraction, pre-answer evidence,
assertion storage, query-view storage, and adaptive retrieval remain off. See
[the operator upgrade and opt-in notes](docs/operator-guide.md#upgrade-from-v0200-or-v0210-rc2-to-v100-rc1)
before enabling them.

## Commands and tools

### Agent tools

Use these tools for current-session recall after compaction. Use Hermes
`session_search` for earlier separate sessions or broad cross-session history
outside the LCM database.

| Tool | Use |
|------|-----|
| `lcm_grep` | Search current-session raw messages and summaries. Opt into `content_scope='externalized'|'both'` for bounded active-session payload search, or `session_scope='all'|'session'` for bounded raw-message archive recovery; broader scopes return raw-message hits only. |
| `lcm_recall` | Search the entire memory across ALL conversations and all time by meaning. Fuses full-text, summary-vector, and chunk-vector arms with RRF, then applies a soft current-conversation (`scope_bias`) and recency prior. Returns bounded summary and verbatim-excerpt hits with `lcm_expand` handles; works FTS-only when embeddings are disabled. |
| `lcm_query_state` | Query the opt-in V4 assertion sidecar for bounded current or historical facts, preferences, recommendations, commitments, actions, and status. Every result carries an exact message ref/span/quote; conflicts stay visible. |
| `lcm_compute` | Execute supported dates, distinct counts, compatible-unit sums, directed/absolute differences, ordering, and latest-state operations over exact cited evidence. Planning, arithmetic, and trace verification are dependency-free and provider-neutral; ambiguous or incomplete inputs fail closed. |
| `lcm_compile_evidence` | Validate one bounded provider-neutral semantic proposal against exact stored refs and return a compact evidence brief with explicit sufficiency state and an optional canonical computation. It never returns final prose or treats model claims as finite-coverage proof. |
| `lcm_evidence_pack` | Hydrate and validate bounded baseline exact refs in the same `lcm.db`, repair only unique in-window quote spans, preserve occurrence/observation time separation, and optionally emit an immutable canonical computation trace without prose. |
| `lcm_retrieve` | Opt-in bounded controller for one continuous answerer tool turn. It tracks typed evidence slots, permits at most three targeted calls to existing retrieval tools, accepts only exact observed refs, caps the evidence context, and can finish through `lcm_compute`. It persists evidence views and traces, never final prose. |
| `lcm_recent` | Retrieve recent summaries by natural UTC period, preferring ready rollups and transparently falling back to time-bounded leaf summaries. |
| `lcm_load_session` | Load one ordered raw-message transcript page for an explicit `session_id`. Continues with `after_store_id` from `next_cursor`; opt into exact slice refs with `include_exact_ref=true`. |
| `lcm_describe` | Inspect the current-session DAG or preview an `externalized_ref` without loading full content. |
| `lcm_expand` | Recover source messages, child summaries, or externalized payloads with pagination. Use `store_id` to fetch a single raw message from a cross-session `lcm_grep` result and `include_exact_ref=true` when its returned slice must be cited or computed. |
| `lcm_expand_query` | Answer a question using expanded current-session LCM context while returning a bounded answer. |
| `lcm_status` | Show runtime health, context pressure, config, source lineage, and lifecycle stats. |
| `lcm_inspect` | Read-only operator inventory for current-session lineage, frontier/fresh-tail metadata, externalized refs/readability, compaction skip/no-op reasons, and matched ignore/stateless patterns. Returns metadata only; use retrieval tools for content. |
| `lcm_doctor` | Run database, FTS, lifecycle, config, and context-pressure diagnostics. |

## Recall skill and policy

Hermes-LCM ships `skills/hermes-lcm/SKILL.md` plus progressive-disclosure
references for configuration, architecture, diagnostics, recall routing, and
session lifecycle. The installer links that directory into the active Hermes
profile so it appears in ordinary skill discovery. On hosts with plugin skill
registration, it is also available explicitly as `hermes-lcm:hermes-lcm`.

When LCM is the active context engine for a bound session, the plugin registers
one deterministic `pre_llm_call` hook. Hermes injects the canonical policy into
the current user-message context, not the system prompt, preserving the stable
system-prompt cache prefix. The policy:

- treats summaries as recall cues rather than exact proof;
- prefers newer source-backed evidence and verifies contradictions;
- teaches narrow FTS query construction and bounded scope selection;
- routes current compacted, cross-conversation, and recent/time-bounded recall
  through the appropriate existing tools;
- does not force a tool call when the current context is already sufficient.

The canonical bytes live in
`skills/hermes-lcm/references/recall-policy.md`. Merely loading the plugin does
not inject them when another context engine is serving the session. Older hosts
without skill or hook registration keep their existing schema-driven behavior.

### Slash commands

Slash commands are disabled by default. Enable them only in trusted operator
contexts:

```bash
export LCM_ENABLE_SLASH_COMMAND=1
```

Available commands:

- `/lcm` or `/lcm status` - current runtime/session status
- `/lcm doctor` - read-only health checks
- `/lcm doctor clean` - read-only scan for obvious junk/noise session candidates
- `/lcm doctor clean apply` - backup-first cleanup for safe pattern-matched
  candidates; requires `LCM_DOCTOR_CLEAN_APPLY_ENABLED=true`
- `/lcm doctor repair` - read-only SQLite/FTS repair diagnostics
- `/lcm doctor repair apply` - backup-first SQLite/FTS repair
- `/lcm doctor source` - read-only scan for legacy blank-source rows
- `/lcm doctor source apply` - backup-first normalization of legacy blank-source
  rows to `unknown`
- `/lcm doctor retention` - read-only retention analysis
- `/lcm backup` - timestamped SQLite backup
- `/lcm rotate` - read-only preview of an in-place tail-preserving compact of
  the active session
- `/lcm rotate apply` - backup-first rotate that advances the lifecycle frontier
  past pre-tail raw messages
- `/lcm help` - command help

Apply paths are intentionally narrow and backup-first. Start with diagnostics
before cleanup or repair.

### Rotate: compact in place without changing session identity

`/lcm rotate` compacts a long-running session in place without changing
`session_id` or `conversation_id`. It is the in-session counterpart to
Hermes-level `/new` and to `/lcm doctor clean`.

What rotate does:

- preserves the live tail (`LCM_FRESH_TAIL_COUNT` most-recent messages)
- advances the lifecycle frontier marker past every raw message before the tail,
  so subsequent bootstrap stops replaying them into the active prompt
- writes a rolling `*-rotate-latest.sqlite3` backup under the same backup
  directory as `/lcm backup`, overwriting the previous rotate slot atomically so
  disk usage stays bounded across repeated rotates

What rotate does not do:

- it does not delete raw messages; pre-tail rows remain recoverable through
  `lcm_load_session` and `lcm_expand`
- it does not invoke the summarization model; trigger normal compaction first if
  you want pre-tail content covered by summary nodes before rotating
- it does not change session or conversation identity, and it refuses ignored or
  stateless sessions with a clear reason

`lcm_status` and `/lcm status` surface `last_rotate_at` and the rolling backup
path. Re-running `/lcm rotate apply` after the frontier is already at or ahead of
the target boundary reports `status: noop` and does not overwrite the previous
known-good rolling backup.

## Configuration

Most installs only need `plugins.enabled` and `context.engine: lcm`.

### Common settings

| Variable | Default | Use |
|----------|---------|-----|
| `LCM_CONTEXT_THRESHOLD` | `0.35` → curve → `0.80` at 1M | Fraction of the context window that triggers LCM compaction. Unset = window-weighted (fork); set = wins over the curve |
| `LCM_FRESH_TAIL_COUNT` | `400` at every window | Upper bound on recent messages protected from compaction; the token cap below is what actually sizes the tail |
| `LCM_FRESH_TAIL_MAX_TOKENS` | `0.15·W` at every window | Token cap for the protected fresh tail — what stays verbatim is the same share of the window at any size; always retains the newest message and complete assistant/tool-result groups |
| `LCM_INCREMENTAL_MAX_DEPTH` | `3` → `5` at 1M | Max DAG condensation depth (`-1` = unlimited, `0` = leaf only); enables hierarchical summarization |
| `LCM_LEAF_CHUNK_TOKENS` | `20000` | Raw-backlog floor before leaf compaction (lowered to one chunk when the chunk is smaller); with dynamic chunking enabled, the base chunk target. The chunk size itself is `LCM_LEAF_CHUNK_FRACTION` — 4 % of the window at every anchor |
| `LCM_DYNAMIC_LEAF_CHUNK_ENABLED` | `false` | Upstream's doubling chunk policy; enabling it keeps upstream's serial behaviour instead of the fork's curved chunking |
| `LCM_DYNAMIC_LEAF_CHUNK_MAX` | `40000` | Upper bound for dynamic leaf chunk targets |
| `LCM_THRESHOLD_FULL_SWEEP_ENABLED` | `false` | At threshold, opt into one synchronous bounded sweep that drains chunked raw history before publishing one new active context (upstream's serial path; the fork's curved drain/pass cap/concurrency apply to the default path) |
| `LCM_SUMMARY_PREFIX_TARGET_TOKENS` | `0` → `0.20·W` at 1M | Sweep-only summary-frontier target; `0` derives it from the curve (upstream: one `LCM_LEAF_CHUNK_TOKENS` budget) |
| `LCM_NEW_SESSION_RETAIN_DEPTH` | `2` | DAG depth retained after manual `/new` (`-1` all, `0` none) |
| `LCM_DATABASE_PATH` | auto | SQLite database path. Empty config resolves to `HERMES_HOME/lcm.db`; plugin installs or operators may set this env var to another profile-scoped path such as `~/.hermes/hermes-lcm.db`. |
| `LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS` | `24` | Minimum hours between startup FTS5 deep integrity-checks (O(index size)). `0` checks every startup; a negative value never checks on startup. Structural checks always run regardless. |
| `LCM_ENABLE_SLASH_COMMAND` | `false` | Enable the optional `/lcm` operator command surface |

When `LCM_FRESH_TAIL_MAX_TOKENS` is enabled, the protected suffix must satisfy
both the message-count and token bounds. The newest message is never dropped,
and a boundary that would begin inside an assistant tool-call/result group is
moved back to that assistant even when doing so exceeds a configured bound.

### Filtering and storage settings

| Variable | Default | Use |
|----------|---------|-----|
| `LCM_IGNORE_SESSION_PATTERNS` | empty | Comma-separated session globs excluded from LCM storage |
| `LCM_STATELESS_SESSION_PATTERNS` | empty | Comma-separated session globs kept read-only |
| `LCM_IGNORE_MESSAGE_PATTERNS` | empty | Comma-separated regex patterns; matching message content is excluded from LCM storage |
| `LCM_SENSITIVE_PATTERNS_ENABLED` | `false` | Opt in to deterministic redaction before LCM storage, FTS indexing, summarization, active replay, and externalized ingest payloads |
| `LCM_SENSITIVE_PATTERNS` | `api_key,bearer_token,password_assignment,private_key` | Comma-separated named sensitive pattern catalog entries to apply when redaction is enabled |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` | `false` | Store oversized ingest payloads, including tool results, media blocks, and generic raw content, in plugin-managed JSON files |
| `LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS` | `12000` | Externalization threshold for normalized payload text |
| `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED` | `false` | Replace token-heavy textual tool results with recoverable externalized refs in active replay; current-turn ingest is immediate and historical assembly respects the protected fresh tail; requires large-output externalization |
| `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS` | `25000` | Token-aware threshold for active-replay tool-result stubbing |
| `LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED` | `false` | Rewrite already-externalized summarized tool rows to compact placeholders |
| `LCM_DOCTOR_CLEAN_APPLY_ENABLED` | `false` | Permit destructive `/lcm doctor clean apply` in trusted operator contexts |
| `LCM_EMPTY_LIFECYCLE_GC_ENABLED` | `true` | Master toggle for automatic pruning of lifecycle rows for sessions that never ingested any messages or summary nodes |
| `LCM_EMPTY_LIFECYCLE_GC_THRESHOLD` | `200` | Number of lifecycle rows at which the GC pass fires |
| `LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS` | `24` | Automatic GC only deletes empty lifecycle rows at least this old; set `0` only in trusted/test environments that intentionally want immediate empty-row pruning |

### Model and timeout settings

| Variable | Default | Use |
|----------|---------|-----|
| `LCM_SUMMARY_MODEL` | auxiliary | Override summarization model |
| `LCM_SUMMARY_FALLBACK_MODELS` | empty | Comma-separated summarization models tried after `LCM_SUMMARY_MODEL` or the auxiliary task default fails |
| `LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `2` → `4` at 1M | Consecutive failed summarization calls before a route is skipped temporarily |
| `LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `300` | Seconds to skip an open summary route before retrying it |
| `LCM_EXPANSION_MODEL` | summary model / auxiliary | Override `lcm_expand_query` synthesis model |
| `LCM_EXPANSION_CONTEXT_TOKENS` | `32000` → `125000` at 1M | Context budget used by the auxiliary LLM for `lcm_expand_query` |
| `LCM_SUMMARY_TIMEOUT_MS` | `60000` → `200000` at 1M | Timeout for one summarization call |
| `LCM_EXPANSION_TIMEOUT_MS` | `120000` → `200000` at 1M | Timeout for one `lcm_expand_query` synthesis call |
| `LCM_SUMMARY_FAILURE_COOLDOWN_SECONDS` | `600` | Fork: cooldown armed when every summariser route fails (replaces upstream's silent truncation) |
| `LCM_CRITICAL_BUDGET_PRESSURE_RATIO` | `0.0` | Disabled at `0.0`; when set, permits critical-pressure bypasses for bounded deferred catch-up and cache-friendly follow-on condensation only |

Advanced compaction, assembly, and extraction knobs are defined in `config.py`. The fork's
own settings (`LCM_SCALE_LOW_WINDOW`, `LCM_SCALE_HIGH_WINDOW`, `LCM_LEAF_CHUNK_FRACTION`,
`LCM_LEAF_PASS_CAP`, `LCM_DRAIN_STOP_FRACTION`, `LCM_SUMMARY_BUDGET_FRACTION`,
`LCM_SUMMARY_CONCURRENCY`, `LCM_LEAF_LOOP_MAX_SECONDS`, … ) are listed with their curve anchors
in [`FORK.md` → Fork configuration reference](FORK.md#fork-configuration-reference).

### Sensitive-pattern redaction

Sensitive-pattern handling is disabled by default so ordinary LCM storage and
`lcm_expand` remain lossless. When `LCM_SENSITIVE_PATTERNS_ENABLED=true`, matched
secret values are replaced with metadata-only placeholders before SQLite, FTS,
summaries, active replay, and externalized payload JSON receive the content. This
is intentionally not lossless for matching values: the raw matched secret is
unrecoverable after redaction.

Supported named catalog entries are:

- `api_key`: `api_key`, `api_token`, `access_token`, `secret_key`, and
  `client_secret` assignments or JSON keys.
- `bearer_token`: `Bearer ...` strings and token-like JSON keys.
- `password_assignment`: `password`, `passwd`, `pwd`, and `passphrase`
  assignments or JSON keys, including quoted values with spaces.
- `private_key`: PEM private-key blocks.

Redaction is forward-only. Enabling it does not rewrite existing SQLite rows,
FTS shadow tables, DAG summaries, or externalized payload JSON that were written
before the setting was enabled. Non-password placeholders include a short
truncated SHA-256 digest for correlation. `password_assignment` placeholders omit
the digest to avoid making password-like values easier to dictionary-check.
`lcm_status`, `lcm_inspect`, and `lcm_doctor` expose the enabled state, configured pattern names,
unknown names, source, and placeholder format without exposing raw secret values.

### Threshold ownership

When `context.engine: lcm` is active, `LCM_CONTEXT_THRESHOLD` is the compaction
threshold LCM uses. Hermes core `compression.threshold` belongs to the built-in
compressor. Hermes core `compression.enabled` is still the global gate that
allows compaction, so leave it enabled when using LCM.

If startup/status output shows a host-side compression percentage that disagrees
with LCM, trust live LCM status after a normal message has initialized the
session.

### Tuning for large context windows

In the fork you normally do not tune for a large window at all: every size that is a
preference is derived from the model's effective `context_length` (see
[What the fork changes](#what-the-fork-changes)). Check the result with `lcm_status` —
the `window_scaling` section prints `t`, and every setting with its resolved value and source
(`curve@t=…`, `env`, `config_yaml:…`, `manual`).

Set explicit values only when you want something other than the curve. The two you are most
likely to set:

```text
compaction trigger = effective context window * LCM_CONTEXT_THRESHOLD
```

- `LCM_CONTEXT_THRESHOLD` (or `lcm.context_threshold` in `config.yaml`): the curve gives 0.80
  at 1M; set it if you want compaction earlier (cheaper prompts) or later (more raw context).
- `LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` / `LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED`:
  pure features, worth enabling at any window (stubs carry a head note in the fork).

Upstream's opt-in policies still exist and keep their upstream semantics when enabled:
`LCM_DYNAMIC_LEAF_CHUNK_ENABLED` (doubling chunks, serial) and
`LCM_THRESHOLD_FULL_SWEEP_ENABLED` (one bounded synchronous sweep: `LCM_SWEEP_MAX_PASSES`,
default 12, total leaf plus condensation calls, and the curved leaf-loop wall clock — 120 s at
256k, 200 s at 1M — per `compress()` call; it persists each completed DAG pass and publishes one
newly assembled active context at the end). On the default path the fork already drains in
curved chunks with a pass cap and a wall clock, so the sweep flag is rarely needed.

Tune against your effective `context_length` if Hermes caps the provider's advertised window.

### Cache policy boundary

LCM is **cache-friendly**, not fully cache-aware. It may avoid some follow-on
condensation churn, but current cache usage counters are retrospective status
data only; they do not tell the plugin whether the next prompt mutation will
break a hot provider cache.

`LCM_CRITICAL_BUDGET_PRESSURE_RATIO` is a narrow escape hatch. It is disabled by
default (`0.0`). When set, LCM compares prompt pressure against the context
window and only at or above that ratio may bypass existing polite gates for
bounded deferred maintenance catch-up and cache-friendly follow-on condensation.
Revisit full cache-aware deferred compaction only after Hermes core exposes
reliable cache state / cache-break signals.

### Session pattern syntax

Pattern matching checks multiple keys: raw `session_id`, `platform`, and
`platform:session_id`.

- `*` matches within one colon-delimited segment
- `**` can span across colons

Example: `cron:*` can match Hermes cron sessions, while exact raw session IDs
still work.

### Noise suppression

LCM offers two layers of noise filtering:

- **Session-level filters** (`LCM_IGNORE_SESSION_PATTERNS`,
  `LCM_STATELESS_SESSION_PATTERNS`) catch noisy traffic that arrives as its own
  session or platform.
- **Message-level patterns** (`LCM_IGNORE_MESSAGE_PATTERNS`) catch cron alerts or
  other noise injected into a normal Telegram or WhatsApp conversation as
  ordinary visible messages.

Message-level patterns are comma-separated Python regex strings compiled once at
engine start. They run against plain text first; structured multimodal payloads
use concatenated text parts first, then normalized JSON fallback when there are
no text parts. Matching messages are skipped before storage.

Example operator config:

```bash
LCM_IGNORE_MESSAGE_PATTERNS=^Cronjob Response:,^>>>Cronjob Response<<<:
```

Invalid regex entries are logged at warning level and dropped. Pattern matching
uses a 50 ms per-pattern timeout when the optional `regex` package is installed.
If `regex` is not installed, LCM logs a warning and disables message-level regex
filtering rather than running unbounded stdlib `re` matches in the ingest path.

Known limitation: the filter runs at ingest time. When a matching message is part
of the chunk summarized in the same turn it arrived, the text may appear inside
the resulting summary node. The filtered message is still not written to the
message store, so DAG lineage stays clean; only serialized summary text can carry
it.

`lcm_status` surfaces the full filter contract under `session_filters`, including
pattern sources, whether the current session is ignored/stateless, and a
process-lifetime `ignored_message_count`.

Ignored/stateless sessions are a storage ownership boundary, not a context-window
opt-out. LCM does not ingest raw messages or create DAG nodes for sessions that
match `LCM_IGNORE_SESSION_PATTERNS`, `LCM_STATELESS_SESSION_PATTERNS`, or the
in-process auxiliary/thread stateless marker. If those sessions cross the normal
context threshold, LCM delegates the compaction call to Hermes' native
`ContextCompressor` so the active request is still bounded before model overflow.
If the native compressor is unavailable, LCM falls back to a deterministic
head/tail trim as a last-resort safety net, still without writing the bypassed
session to `lcm.db`.

### Large tool-output handling

Storage-boundary payload guard contract: LCM prevents media-ish inline payloads from being written into plugin-local SQLite rows at the storage boundary.

Externalization for ordinary large tool output is opt-in. When enabled,
oversized tool results are written to plugin-managed JSON files and referenced
from summaries. They remain inspectable through
`lcm_describe(externalized_ref=...)` and `lcm_expand(externalized_ref=...)`.

Active-replay stubbing is a second, independently opt-in replay policy. When
both externalization and active-replay stubbing are enabled, newly ingested
textual tool results above the token threshold are durably externalized and
replaced immediately in provider-visible replay, including results in the
protected fresh tail. This lets a stub-only replay change converge even when no
leaf is eligible for compaction. A historical assembly pass applies the same
policy to older tool results before budgeting, while respecting the protected
fresh tail. Tool-call ids and compatible structured text block types/keys are
retained; raw SQLite rows and DAG lineage are not rewritten by the historical
pass. Structured image/media results remain inline, preserving the provider
replay contract established by Hermes-LCM PR #226. If durable externalization
cannot be confirmed, replay keeps the original payload inline. Results from
`lcm_describe` and `lcm_expand` also remain inline so recovery does not
recursively produce another ref.

The storage-boundary payload guard is separate from that opt-in. LCM always
scans messages at the store boundary before writing `messages.content` or
`messages.tool_calls` to SQLite. Inline `data:*;base64,...` payloads and
conservative long base64-looking runs are replaced with compact placeholders and
written to the same plugin-managed externalized-payload directory.

This avoids duplicating media-ish payload bytes into `lcm.db`, FTS shadow tables,
WAL files, or ordinary SQLite backups while preserving lossless recovery via the
placeholder `ref` and `lcm_expand(externalized_ref=...)`. If externalization
fails, LCM logs a warning and leaves the original text inline rather than
dropping data.

`lcm_doctor` reports SQLite `journal_mode`, `quick_check`, database/WAL sizes,
largest content/tool-call rows, suspicious inline payload rows, and aggregate
externalized-payload stats. Doctor output is metadata-only for these scans.

This guard is scoped to LCM's own `lcm.db` write boundary. It does not prevent
Hermes core, or any other host layer, from writing inline payloads to Hermes
`state.db`, and it does not rewrite historical rows already present in `lcm.db`.
If bytes already landed in Hermes `state.db`, that is upstream/outside LCM scope;
use backup-first cleanup or migration procedures before mutating historical host
rows.

Transcript GC is separate and opt-in. It only rewrites already-externalized,
already-summarized tool-role rows to compact placeholders. It keeps the same
`store_id`, keeps payload files, skips pinned messages, and preserves lossless
recovery through `externalized_ref`.

## Retrieval contract

LCM retrieval tools default to current-session scope. `lcm_grep` accepts
`session_scope='all'` or `session_scope='session'` as an explicit opt-in for
bounded archive search over rows already present in `lcm.db` (raw-message hits
only). Once a session id is known, `lcm_load_session` can enumerate that
session's raw transcript in chronological `store_id` pages without a search
query. Use Hermes `session_search` for broad cross-session history outside the
LCM database.

Within the current session, `source` filters raw rows directly and filters
summary nodes by descendant raw-message source lineage. `unknown` is a real
source value, not a wildcard. Legacy blank-source rows are treated as `unknown`.
`role`, `time_from`, and `time_to` are raw-message filters applied in the message
search query before result limiting. When a raw-message filter is active,
`lcm_grep` returns raw rows only and reports `summary_results_omitted`.

Tool responses are bounded so one retrieval call cannot flood the main context.

### Lossless raw recovery contract

Lossless recovery means raw content is stored with stable source lineage and can
be recovered in deterministic pages:

- `lcm_expand(node_id=...)` pages immediate sources with `source_offset` and
  `source_limit`
- `lcm_load_session(session_id=...)` pages ordered raw session rows with
  `after_store_id` and `next_cursor`
- oversized raw messages continue with `content_offset`
- `lcm_expand(externalized_ref=...)` pages payload content with `content_offset`
- `lcm_expand_query` uses `context_max_tokens` for auxiliary context and reports
  truncation/pagination hints when needed

Carried-over summary nodes can become current-session content after `/new`, but
their source eligibility still comes from descendant raw messages. Expanding a
carried-over current-session node recovers the original raw message sources even
when those sources still belong to a previous session.

## OpenClaw/lossless-claw import

`hermes-lcm` includes an opt-in operator script for backfilling OpenClaw history
into the local hermes-lcm SQLite store. It supports two source shapes:

1. **SQLite LCM database** (`--source-db`) for migrations from an existing
   lossless-claw/OpenClaw `lcm.db`.
2. **JSONL session exports** (`--source-jsonl` / `--source-jsonl-dir`) for fresh
   installs, plugin-off catch-up, or migrations where no source SQLite database
   is available.

SQLite source example:

```bash
python scripts/import_lossless_claw.py \
  --source-db ~/.openclaw/path/to/lcm.db \
  --target-db ~/.hermes/lcm.db \
  --agent sammy
```

JSONL source example:

```bash
python scripts/import_lossless_claw.py \
  --source-jsonl ~/.openclaw/agents/sammy/sessions/session-a.jsonl \
  --target-db ~/.hermes/lcm.db \
  --agent sammy \
  --json
```

For a directory of session exports:

```bash
python scripts/import_lossless_claw.py \
  --source-jsonl-dir ~/.openclaw/agents/sammy/sessions \
  --target-db ~/.hermes/lcm.db \
  --agent sammy
```

The script is intentionally conservative:

- dry-run is the default; pass `--apply` to write
- run it against an explicit target DB path, preferably while Hermes is stopped
  for that profile
- writes create a timestamped target DB backup first when the target already
  exists
- SQLite imports can include OpenClaw summaries with `--include-summaries`; this
  migrates compatible summary rows into Hermes `summary_nodes`
- JSONL imports migrate raw message rows only; JSONL does not carry a summary DAG
- imported rows keep explicit provenance in `session_id` and `source`, for
  example `openclaw-lcm:agent:sammy:<source-session>` or
  `openclaw-jsonl:agent:sammy:<source-session>`
- the SQLite default provenance identity is the source `conversations.session_id`,
  preserving source session boundaries even when many conversations share one
  `session_key`
- pass `--session-identity session_key` only for SQLite imports when you
  intentionally want conversations with the same source session key grouped into
  one imported LCM session
- reruns are idempotent for the same `--import-id`; the default `import_id` is
  source-path-derived, so pass a stable `--import-id` if you may import the same
  copied DB or JSONL export set from different paths
- `--json` prints a reconciliation report with `scanned`, `eligible`,
  `would_import`, `imported`, `skipped_existing`, `skipped_empty`,
  `invalid_rows`, `warnings`, and summary counters
- changing `--agent`, `--namespace`, or `--session-identity` under the same
  `--import-id` is treated as the same import and will skip already-tracked
  source messages; use a new `--import-id` for a different mapping
- no OpenClaw config or separate secret tables are imported, but raw transcripts,
  summaries, and tool payloads may contain sensitive user data

This is a local archive migration path. It does not make LCM a general memory
provider, and it does not change the current-session retrieval contract for
agent tools.

For databases that already contain large historical tool rows,
`scripts/backfill_externalized_tool_outputs.py` can pre-create native recovery
sidecars without rewriting SQLite. It is dry-run by default, writes a scrubbed
digest/count manifest, is idempotent on repeat apply, and supports guarded
manifest-based rollback. See the [operator guide](docs/operator-guide.md#historical-tool-output-sidecars).

## Troubleshooting

### `hermes plugins` shows `lcm (not found)` but LCM tools exist

If `plugins.enabled` contains `hermes-lcm`, `context.engine: lcm` is set, and
the runtime exposes LCM tools, LCM is loaded. The `lcm (not found)` line is a
Hermes host discovery/status mismatch, not an LCM storage or compaction failure.

### `/lcm status` looks unbound after restart

Compatible Hermes gateway hosts expose task-local lane metadata while
dispatching `/lcm`. Once that lane has constructed an agent, `/lcm status`
resolves the same active LCM runtime as `lcm_status`, including its foreground
view while an ignored or stateless side channel is bound. Before the first
normal message constructs an agent, or in a genuinely sessionless process,
`session_id: (unbound)` and `threshold_tokens: (uninitialized)` remain expected.

## Architecture

The engine sits between Hermes context assembly and the backing conversation
store. It records raw messages, compacts old material into a summary DAG, and
exposes retrieval tools that can drill back into exact stored sources.

<p align="center">
  <img src="docs/architecture.png" alt="hermes-lcm architecture" width="700">
</p>

## How it works

1. **Ingest** - persist each message in SQLite with FTS metadata
2. **Compact** - summarize older messages outside the fresh tail into D0 leaf
   nodes
3. **Condense** - merge same-depth nodes into higher-depth summaries
4. **Escalate** - shrink oversize summaries from detailed to bullets; if every route fails,
   keep the raw context, arm a cooldown and tell the host (the fork removed upstream's
   deterministic truncation)
5. **Assemble** - combine system prompt, highest-depth summaries, and fresh tail
6. **Retrieve** - use LCM tools to drill into compacted history or synthesize
   from expanded context

## Documentation

- [Feature overview](docs/features-overview.md) — every feature family, what
  it does, why it exists, and the switch that enables it
- [Agent configuration profiles](docs/agent-config-profiles.md) — copy-paste
  env profiles: coding agent, long-horizon assistant, fully-local, cost-guarded
- [Operator guide](docs/operator-guide.md) — install, activation, full
  configuration reference, diagnostics
- [Retrieval tools reference](docs/retrieval-tools.md) — exact tool contracts
- [Embeddings setup](docs/embeddings-setup.md) — free-tier and local embedding
  providers, warmup, backfill
- [LCM paper](https://papers.voltropy.com/LCM)
- [Architecture diagram](docs/architecture.png)
- [Standard compression diagram](docs/standard_compression.png)
- [LCM compression diagram](docs/lcm_compression.png)
- [Contributing guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Upstream releases](https://github.com/stephenschoettler/hermes-lcm/releases) (this fork tracks them; see [`FORK.md`](FORK.md))

## Development

Important files:

```text
plugin.yaml      manifest
__init__.py      plugin registration and optional slash-command registration
engine.py        LCMEngine main orchestrator
store.py         SQLite message store and FTS
dag.py           summary DAG and FTS
config.py        env var defaults and overrides
command.py       /lcm command handlers
tools.py         lcm_grep, lcm_load_session, lcm_describe, lcm_expand, lcm_expand_query
schemas.py       tool schemas shown to the model
tests/           standalone pytest coverage (tests/fork/ = fork tests, by step)

window_scaling.py      fork: anchor table + curve + resolver
window_scaled_mixin.py fork: resolves the curve on the engine, exposes effective_* values
host_cooldown.py       fork: the host's compression-failure cooldown protocol
marked_loss.py         fork: every marker left where upstream cut or dropped silently
node_meta.py           fork: lcm_node_meta sidecar (escalation level + index block)
leaf_pipeline.py       fork: concurrent leaf summarisation as a lookahead over the serial loop
coverage_doctor.py     fork: lcm_doctor coverage
errors.py              fork: SummaryUnavailableError
```

Run tests:

```bash
pip install pytest
python -m pytest tests/ -v
# fork: the same suite through the plugin's host venv, with the umask the SQLite guard needs
scripts/test.sh
```

No Hermes Agent checkout is required for the test suite; tests include a
lightweight ABC stub.

## Fork maintenance

The fork is meant to track upstream **and** the other lossless-context implementation it was
inspired by. The contract, in full, is in [`FORK.md`](FORK.md); the short form:

1. **Upstream hermes-lcm.** `git fetch upstream && git merge upstream/main` in the fork
   repository; resolve conflicts with [`docs/fork-touchpoints.md`](docs/fork-touchpoints.md)
   (every touched upstream line, why, and what to do on conflict); `scripts/test.sh` must be
   green; then pull into `~/.hermes/plugins/hermes-lcm` and re-pin.
2. **lossless-claw.** On every
   [lossless-claw](https://github.com/Martian-Engineering/lossless-claw) release, compare it
   deeply against this fork — compaction/DAG algorithm, loss-avoidance and provenance,
   summariser prompts and index quality, retrieval tools and operability — and port everything
   it does better for this fork's purpose (no loss; 1M windows without degrading 256k). The
   audit prompts are under `docs/claw-comparison/prompts/`; what each round decided is recorded
   as a pass entry in `docs/TASKS.md`, and ported work lives in the code.
3. Never merge anything that drops content without a marker, coarsens the index, or makes a
   bounded result read as a complete one. Upstream is the floor — never worse than it — not the
   target: matching its number at 256k is only correct where the value is a preference.

## Contributing

Issues and PRs welcome. Bug fixes and correctness improvements are highest
priority. New features should be scoped, backwards-compatible, and tested.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch, validation, and PR guidance.
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for project conduct expectations
and [SECURITY.md](SECURITY.md) for vulnerability reporting.
Upstream's [releases page](https://github.com/stephenschoettler/hermes-lcm/releases) carries
the base project's changelog; this fork's history is its commit log and the pass ledger in
[`docs/TASKS.md`](docs/TASKS.md).

## License

[MIT](LICENSE)
