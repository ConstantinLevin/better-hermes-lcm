# Upstream files touched by the fork

Format: `file` — what the hook does — why — how to re-apply on a conflict.
(Maintained by hand; every fork commit that touches an upstream file updates this list.)

| file | hook | why | on conflict |
|---|---|---|---|
| `config.py` | 14 new dataclass fields (block before `from_env`) + 14 `_EnvFieldSpec` entries (tail of `ENV_FIELD_SPECS`) | window-weighted tuning overrides; defaults are "unset" sentinels so upstream behaviour is unchanged | pure additions at the tails — keep ours; if upstream added fields in the same spot, keep both |
| `engine.py` | `WindowScaledSettingsMixin` import + base class; `_init_window_scaled_settings()` in `__init__`; `_resolve_window_scaled_settings()` before both `return True` in `_set_context_length` | curve resolution point | keep our 4 one-liners on top of theirs; if upstream restructured `_set_context_length`, re-add the call on every return path |
| `tools.py` | `"window_scaling": engine.window_scaling_status()` in the status dict | expose resolved values + sources | keep ours |
| `tests/test_lcm_engine.py`, `tests/test_lcm_core.py` | 14 assertions of the form `X.threshold_tokens == int(W * <config>.context_threshold)` -> `int(W * X.context_threshold)`, and `X.context_threshold == X._config.context_threshold` -> `X.effective_context_threshold` (marked `# fork: resolved (curved) threshold`) | they pinned a window-independent threshold default at windows > 256k; the design curves it | re-apply the same regex (see step 2/3 commits) if upstream adds more such assertions |
| `engine.py` | 14 consumer lines read `self.effective_*` instead of `self._config.*` (marked `# fork: curved`): fresh tail (x6), depth cap (x2), summary timeout (x3), l2 ratio (x2), stub threshold (x1) | curve consumers | keep ours; if upstream added a new consumer of one of these fields, switch it too |
| `compaction.py` | sweep target derived from `self.effective_sweep_target_tokens` (low anchor = `leaf_chunk_tokens`, matching upstream's fallback) | curve consumer | keep ours |
| `tools.py` | `expansion_timeout_ms` and `fresh_tail_count` consumers read `effective_*` | curve consumers | keep ours |
| `escalation.py` | L3 helpers (`_L3_TRUNCATION_MARKER`, `_truncate_text_to_tokens`, `_deterministic_truncate`) deleted; `summarize_with_escalation` raises `errors.SummaryUnavailableError` after L2; `l3_truncate_tokens` param kept, ignored | no silent truncation | keep ours; if upstream changed the escalation tail, keep the raise |
| `engine.py` | `HostCooldownMixin` first in bases; rescue exhaustion raises `SummaryUnavailableError` (`_summarize_leaf_chunk_with_rescue`) | fail loud without killing the turn | keep ours |
| `compaction.py` | leaf-loop `except`: a later pass failing after >=1 persisted pass publishes progress in BOTH modes (upstream: sweep only) and records `_last_leaf_summary_error` | upstream's `raise` branch was unreachable while L3 existed | keep ours |
| `rollup_builder.py` | `_summarize_capped` raises `SummaryUnavailableError` instead of truncating a non-converging rollup | no silent truncation; callers already `_mark_failed` | keep ours |
| `config.py` | `summary_failure_cooldown_seconds` (600) + env spec | cooldown length | additive |
| `tests/conftest.py` | autouse `_fork_mock_summariser` fixture: deterministic summary strictly shorter than the envelope's source; delegates to the real `_call_llm_for_summary` when a test installed its own `agent.auxiliary_client`; opt-out `LCM_TESTS_REAL_SUMMARISER=1` | upstream tests silently relied on L3 as their no-LLM compaction path | keep ours, append after upstream's content |
| `tests/test_lcm_core.py` | `TestDeterministicTruncate` rewritten (helpers gone + raise); `TestEscalation.test_truncate_*` and `TestTokens.test_l3_truncate_text_to_tokens_respects_budget` removed; spend-guard test expects raise | they asserted L3 output | drop any new L3 assertions upstream adds |
