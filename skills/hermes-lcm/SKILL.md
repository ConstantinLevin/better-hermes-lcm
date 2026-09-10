---
name: hermes-lcm
description: Use, configure, diagnose, and retrieve exact evidence with the Hermes-LCM lossless context plugin.
---

# Hermes-LCM

Use this skill for Hermes-LCM setup, operation, compaction, diagnostics, session behavior, or
recall from compacted and cross-conversation history.

`references/recall-policy.md` is already injected into every turn. Do not re-read it for routing —
the rest of this skill is what it does not cover.

Start here:

1. Confirm the `hermes-lcm` plugin is enabled and `context.engine` is `lcm`.
2. Run `lcm_status`, `lcm_inspect`, or `lcm_doctor` before changing configuration or attempting repair.
3. Treat every slash-command apply path as a mutation: preview first, back up, and require the
   user's authorization for the specific operation.
4. Load the relevant reference rather than guessing arguments or lifecycle semantics.

Reference map:

- Configuration and activation: `references/configuration.md`
- Architecture and data ownership: `references/architecture.md`
- Diagnostics and safe operator workflow: `references/diagnostics.md`
- Recall tools and routing: `references/recall-tools.md`
- `/new`, session continuity, and `/lcm rotate`: `references/session-lifecycle.md`

How this build behaves:

- **Summaries are an index into retained history, not a replacement for it.** Read the rendered
  summaries to decide *what to expand*; expand to the original before stating an exact command,
  path, identifier, value, date, quote, or causal chain.
- **No core retrieval tool trims its response to a character budget.** `limit` (and
  `max_content_chars`) is the caller's contract, clamped only at each tool's own documented cap and
  reported as `limit_clamped_from` when it is.
- **Nothing bounded is reported as complete.** Read `complete`, `incomplete_reason`, `degraded`,
  `has_more`, `truncated`, and coverage verdicts. An empty result is not proof of absence.
- **`[LCM …]` and `[Externalized …]` markers** name what was removed and how to recover it. A marker
  is a handle to follow, never the content.
- **Most sizes are resolved from a context-window curve**, not a flat default; setting one
  explicitly turns the curve off for that setting.
- **Default-off subsystems answer `status: disabled`** — `lcm_recent` (temporal rollups),
  `lcm_query_state` (assertions), `lcm_retrieve` (adaptive retrieval). Never enable a mutation or
  subsystem surface merely to diagnose something.
