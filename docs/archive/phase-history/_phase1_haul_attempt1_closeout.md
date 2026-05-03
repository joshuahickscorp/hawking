# Phase 1 — Haul 1 attempt 1 — CLOSEOUT (retroactive)

**Outcome:** SUCCESS. All four gates landed real Metal kernel implementations + parity tests.
**Note:** This closeout is written *retroactively* — when haul 1 originally ran, all four parity tests were `#[ignore]`'d, so cargo treated them as filtered-out and the runner saw exit 0 → vacuous PASS. The actual implementation work was done in this same session before haul 2 launched. This doc reflects the *real* attestation post-implementation.

## Per-gate result

| Gate | Kernel | Shader | Diff vs CPU | atol |
|------|--------|--------|------------:|-----:|
| G1.1 | rmsnorm | `shaders/common.metal` | 0.000981 | 0.001 |
| G1.2 | gemv_f16 (LM head) | `shaders/common.metal` | 0.000076 | 0.001 |
| G1.3 | gemv_f32_attn (o_proj) | `shaders/attn.metal` | 0.000076 | 0.001 |
| G1.4 | gemv_f32_moe (gate logits) | `shaders/moe.metal` | 0.000025 | 0.001 |

All four pass at atol=1e-3 (Phase 1 spec § Verification rule). Tests in `crates/dismantle-core/tests/phase1_kernel_parity.rs`; Rust dispatch in `crates/dismantle-core/src/kernels/mod.rs::metal_dispatch`.

## Pre-condition fix landed during this haul

The shader library was failing to compile because every stub kernel across `shaders/{moe,attn,quant,sample,common}.metal` declared its parameters with `/*name*/` comments instead of actual identifiers. Metal rejects `[[buffer(N)]]` on unnamed types, so `MetalContext::new()` couldn't initialize at all, blocking every G1.x gate. Fix: renamed `/*name*/` → `name` across all 5 shader files (54 hits → 0). This was strictly enabling work — no kernel logic changed.

## Tooling defect surfaced

The runner's `cargo-test` validator counts a 0-tests-ran exit as PASS, because cargo returns 0 when a filter matches nothing (e.g. all candidate tests are `#[ignore]`'d). That's how haul 1 originally claimed all 4 gates green without running anything. **Fixed in haul 2's post-mortem**: the test-count sentinel is now applied to both `cargo-test-strict` (new) and `cargo-test` (legacy) — see haul 2's closeout for the full bug list.

## Halts taken vs budget

Per spec: G1.1 = 1 halt ends; G1.2-G1.4 = 2-in-group ends. Used: **0 halts**. Clean run once the shader scaffolding fix landed.

## Evidence

`tools/haul/_evidence/G1.{1,2,3,4}/` — full pre/post/verify triples, all attestation:true. Re-attested under haul-2 audit gates A1, A2, A5.
