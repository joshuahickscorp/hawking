# Hawking — Commit Plan (2026-06-22)

The working tree spans 5 independent lanes. This plan splits them into narrow, reviewable commits. **It is a PLAN —
not executed here** (committing is an attended decision; the tree mixes user/agent lanes). `reports/` is evidence — never stage it.
Commit messages must look human-authored (no AI-attribution trailers).

## Lane 1 — RWKV serve admission + HTTP robustness  (MINE · ready · gated)
- `crates/hawking-core/src/model/rwkv7.rs` — adds the 3 batch trait methods (`encode_prompt_for_batch`,
  `decode_token_for_batch`, `eos_id_for_batch`); fmt-clean (pre-existing drift elsewhere in the file is NOT in this diff).
- `crates/hawking-serve/src/http.rs` — `driver.admit` `Err` → clear SSE error (`code: admit_unsupported`) instead of a
  silent forever-queue.
- `crates/hawking-core/tests/rwkv7_prefill_slot_multiseq_parity.rs` — the serve-correctness parity gate (the proof for
  this lane: admission works; remaining decode bug reproduced).
- **Gate:** `cargo check -p hawking-serve` ✅; `ssm_serve_smoke.sh` (admitted=1, no hang). Parity test FAILS by design
  (it gates the *next* fix) — commit it as a known-red gate, or `#[ignore]` keeps the suite green (it is ignored).
```bash
git add crates/hawking-core/src/model/rwkv7.rs crates/hawking-serve/src/http.rs \
        crates/hawking-core/tests/rwkv7_prefill_slot_multiseq_parity.rs
git commit -m "feat(serve): RWKV continuous-batch admission + robust admit-error handling + parity gate"
```

## Lane 2 — SSM productization harness + CI  (MINE · ready · bash -n clean)
- `tools/ci/ssm_serve_smoke.sh`, `tools/ci/ssm_product_gate.sh`, `tools/ci/ssm_quality_suite.sh`,
  `tools/ci/overnight_hardening.sh`, `tools/ci/preflight.sh`, `tools/ci/README.md`, `tools/bench/ratios.sh`.
- **Gate:** all `bash -n` ✅. (Some authored by Codex — confirm joint authorship before squashing.)
```bash
git add tools/ci/ tools/bench/ratios.sh
git commit -m "test(ci): SSM serve smoke, product gate, quality suite, overnight hardening, ratios bench"
```

## Lane 3 — Campaign documentation  (MINE + Codex · docs only)
- `docs/campaign/` (all standing artifacts), `docs/architecture.md`, `docs/env_flags.md`,
  `docs/plans/{q6k_predec_design,ratios_roadmap_2026_06_21,rwkv7_dynamic_batch_hardening_2026_06_20,rwkv7_micro_frontier_prompt_2026_06_20}.md`.
- **Gate:** docs only.
```bash
git add docs/campaign/ docs/architecture.md docs/env_flags.md docs/plans/q6k_predec_design.md \
        docs/plans/ratios_roadmap_2026_06_21.md
git commit -m "docs(campaign): project-standing packet, roadmap, kill-ledger, env-flag + architecture reference"
```

## Lane 4 — Diagnostic test (PRE-EXISTING · leave / confirm owner)
- `crates/hawking-core/src/model/qwen_dense.rs` — `mod bsize_verify_diag` (`#[cfg(test)]`, +114). Was modified BEFORE this
  campaign (the spec-decode investigation). **Do NOT fold into Lane 1.** Confirm the owner still wants it; commit separately
  if so: `git commit -m "test(spec): bsize verify-cost diagnostic"`.

## Lane 5 — Training / KD experiments (SEPARATE WIP · do NOT commit unattended)
- `tools/training/*.py`, `tools/training/*.sh`, `docs/plans/{g1a_v2_expansion_results,rwkv7_competitive_scorecard,rwkv7_spec_hardening}_2026_06_20.md`.
- Belongs to the KD/training campaign (R8). Owner = training lane; out of inference-runtime scope. Leave dirty.

## Order + safety
1. `git diff --check` (clean ✅). 2. Commit Lane 1 → Lane 2 → Lane 3 (the reviewed, gated, mine lanes). 3. Leave Lanes 4
and 5 for their owners. 4. **Never** `git add reports/` or `git add -A`. 5. **Do not run global `cargo fmt`** before
committing (repo-wide drift; would balloon every diff — see `open_risks_and_gates.md` R6).
