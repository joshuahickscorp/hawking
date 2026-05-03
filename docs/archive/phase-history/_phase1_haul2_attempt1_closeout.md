# Phase 1 — Haul 2 attempt 1 — CLOSEOUT

**Outcome:** SUCCESS (impl + pre-flight clean; audit/self-improve had record-and-continue halts that were addressed post-haul).
**Wall clock:** ~10 min for the runner pass; total session including kernel writing ~3.5 hr.
**Halted at:** none (haul completed every layer).
**Halt counts:** pre-flight=0 impl=0 audit=2 self-improve=2

## Per-layer status

| Layer | Gates | Pass | Record-fail | Notes |
|-------|------:|-----:|------------:|-------|
| pre-flight | 3 | 3 | 0 | clean |
| impl (Wedge 2) | 4 | 4 | 0 | **all H2.x parity green** |
| audit | 5 | 3 | 2 | A3 clippy 19→30 (Wedge 2 added warnings), A4 fmt drift |
| self-improve | 3 | 1 | 2 | S2 cargo-fmt applied; S1 + S3 hit handler bugs, fixed + re-run post-haul |
| closeout | 1 | 1 | 0 | Z1 noop |

## Wedge 2 attestation

All four H2.x parity tests run independently in clean cargo processes via `cargo-test-strict`, with the test-count sentinel guarding against vacuous PASSes.

| Gate | Kernel | Diff vs CPU | atol |
|------|--------|------------:|-----:|
| H2.1 | moe_topk_gate (softmax + top-K) | 0.000000 | 0.001 |
| H2.2 | moe_grouped_gemm_q4 (fused Q4_K_M dequant) | 0.000034 | 0.001 |
| H2.3 | moe_gather_combine (weighted scatter-add) | 0.000000 | 0.001 |
| H2.4 | gemm_q4_k_m_fused (dense Q4_K_M path) | 0.000069 | 0.001 |

Plus haul-1's four kernels (G1.1 rmsnorm, G1.2 gemv_f16, G1.3 gemv_f32_attn, G1.4 gemv_f32_moe) re-attest cleanly under audit gates A1/A2/A5. Total: **8 Metal kernels with parity-test attestation under the runner.**

## Audit drift recorded

- **A3 cargo-clippy-baseline 19**: workspace went **19 → 30** warnings. The new code in this haul (H2.x kernels + parity tests + propose-patch.sh interactions) introduced 11 style/doc lints. None are correctness issues; they're typical of new module code (too-many-args on dispatch helpers, manual slice copies, missing `#[inline]` hints). Bumping the baseline to 30 in `_phase1_haul2_manifest.md` would be the next attended-session call. See `tools/haul/_evidence/A3/stdout.log` for the full list.
- **A4 cargo-fmt-check**: 2 unformatted files. **Fixed by S2 in the same haul** — `cargo fmt --all` applied; subsequent `cargo fmt --check` should now pass cleanly.

## Self-improvement applied

| Gate | Patch | Status | Notes |
|------|-------|--------|-------|
| S1 | expand-baseline-probe-state | **Re-run post-haul** (handler had a `$2` arg + Python heredoc quoting bug; both fixed in `tools/haul/propose-patch.sh`) | Final diff in `tools/haul/_evidence/S1/applied.patch` |
| S2 | cargo-fmt | Applied during haul | Diff in `tools/haul/_evidence/S2/applied.patch` |
| S3 | unsafe-doc-comment metal/mod.rs | Source was modified during the failed haul attempt; idempotent re-run post-fix is a no-op (already in desired state) | Diff in `tools/haul/_evidence/S3/applied.patch` is `no-op` |

## Halts taken vs budget

| Layer | Budget | Used |
|-------|--------|-----:|
| pre-flight | 1 = end haul | 0 |
| impl | 2-in-group = end haul | 0 |
| audit | record-and-continue | 2 (A3, A4) |
| self-improve | record-and-continue | 2 (S1, S3 — handler bugs, not haul scope issues) |

**Halt budget was preserved** — no haul-ending halts fired. Audit/self-improve halts are by-design recordings, not failures.

## Bugs surfaced in haul tooling (post-mortem)

These are real defects in the super-haul tooling that the haul itself surfaced. All fixed post-haul:

1. **`tools/haul/run-gates.sh::run_gate` per-item probe was over-cautious.** The original spec rule (retry-on-degraded for 5×30s, halt after) was written for the model-load smoke gate; for the super-haul's RAM-light gates (verify-evidence, cargo-test --lib, clippy, fmt, patch-apply) it caused unnecessary 2.5-min waits at degraded pressure. **Fix landed**: probe halt threshold relaxed to `critical` for non-`dismantle-smoke` validator kinds; smoke retains the legacy degraded-retry behavior.
2. **`tools/haul/verify-evidence.sh` chicken-and-egg.** When `verify-evidence` runs as a validator gate (P0.1, A1), its own `_evidence/<gate>/pre.json` exists but `post.json`/`verify.json` don't yet, and `verify-evidence.sh` would mark itself MISSING. **Fix landed**: read `_evidence/.active` (the watcher's currently-running-gate marker) and skip that gate.
3. **`tools/haul/verify-evidence.sh` overzealous gate detection.** `_evidence/expand/` (an earlier-task stderr-capture dir) was being treated as a gate. **Fix landed (twice)**: (a) moved expand-baseline.sh's stderr dir out of `_evidence/`, (b) verify-evidence skips dirs without `pre.json`.
4. **`tools/haul/run-gates.sh` DRY_RUN inter-item cooldown.** 30s sleep between gates fired even in DRY_RUN, making the verification step a 7-min slog. **Fix landed**: cooldown is gated on `DRY_RUN != 1`.
5. **`tools/haul/propose-patch.sh` handler arg + Python quoting bugs.** Two handlers (`expand-baseline-probe-state`, `unsafe-doc-comment`) called `emit_diff "$target" "$2"` instead of `"$1"`, so the audit-trail patch path was empty and emit_diff failed. The S1 handler additionally used `'\''` inside a Python triple-quoted string (interpreted as escaped quote, output `'''` instead of bash's apostrophe-escape). **Fix landed**: both bugs corrected in the patched script; standalone re-runs of S1 and S3 produce clean diffs.

## Next attended-session work

- **Bump clippy baseline** in `_phase1_haul2_manifest.md` from 19 to 30 so future runs aren't perpetually flagging Wedge 2's lint debt. Or land a focused clippy-cleanup haul.
- **Phase 2 Wedge 1** (single-launch fused MoE — FlashDMoE technique). Manifest needs scoping. Wedge 2's parity tests + tooling are now the foundation.
- **Token-baseline regression for Wedge 2**. The smoke gate path needs a memory window where the 9 GB model can load cleanly. Once slm finishes a training run, that window opens.

## Evidence

All triples landed under `tools/haul/_evidence/`. Verify-evidence run post-haul shows: G1.1-G1.4, P0.x, H2.x, A1/A2/A5, S2, Z1 all attesting; A3/A4/S1/S3 missing `verify.json` because the runner only writes `verify.json` on PASS. The applied patches under `_evidence/S*/applied.patch` carry the audit trail for what self-improvement actually changed.
