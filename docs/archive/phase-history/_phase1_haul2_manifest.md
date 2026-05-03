# Phase 1 — Haul 2 manifest (Wedge 2 + audits + self-improvement)

The "super haul." Bundles four work layers into a single launch:

1. **Pre-flight** — re-attest haul 1 evidence and confirm a green build/test
   baseline before introducing new code surface.
2. **Implementation (Wedge 2)** — fused Q4_K_M dequant, the actual ROADMAP
   Phase-1 deliverable. Four kernels, all RAM-light synthetic parity tests
   at atol=1e-3.
3. **Audit** — clippy / fmt / parity / unit-test attestation. Drift is logged
   but never ends the haul.
4. **Self-improvement** — auto-applies three small fixes: the
   `expand-baseline.sh` `degraded\nunknown` log artifact, `cargo fmt`, and a
   missing `# Safety` doc on `metal::new_buffer_no_copy`. Patches land as
   source edits and unified diffs in `tools/haul/_evidence/S*/applied.patch`.

Halt budget per layer (enforced by `tools/haul/run-gates.sh`):

| Layer | Halt rule |
|-------|-----------|
| pre-flight | 1 halt = end haul |
| impl | 2 halts in {H2.1..H2.4} = end haul; 1st halt continues to next item |
| audit | record-and-continue (never ends haul) |
| self-improve | record-and-continue (never ends haul) |
| closeout | always runs |

Launch:

```bash
SLM_PID=$(pgrep -f mamba_byte_train | head -1) \
HAUL=2 \
PER_VALIDATOR_TIMEOUT_S=2400 \
./tools/haul/coexist.sh launch phase1
```

## Wedge 2 kernels (impl layer)

Each H2.x lands a Metal kernel + Rust dispatch + parity test against a CPU
reference at atol=1e-3. All four were implemented before this manifest's
launch (kernel bodies live in `shaders/{moe,quant}.metal`; Rust dispatch
in `crates/dismantle-core/src/kernels/mod.rs`).

| Gate | Kernel | Shader | Pattern |
|------|--------|--------|---------|
| H2.1 | `moe_topk_gate` | `shaders/moe.metal` | softmax + top-K, one workgroup per token |
| H2.2 | `moe_grouped_gemm_q4` | `shaders/moe.metal` | fused Q4_K_M dequant inside FMA, one workgroup per row |
| H2.3 | `moe_gather_combine` | `shaders/moe.metal` | weighted scatter-add over (token, expert) pairs |
| H2.4 | `gemm_q4_k_m_fused` | `shaders/quant.metal` | dense-path twin of H2.2; same body |

## Self-improvement (auto-apply)

Per user directive, the agent applies these directly. Each handler is a
named subroutine in `tools/haul/propose-patch.sh`; each writes a unified
diff to `_evidence/<gate>/applied.patch` for audit.

- **S1 expand-baseline-probe-state** — fixes the `degraded\nunknown` log
  artifact in `tools/haul/expand-baseline.sh::probe_state()`.
- **S2 cargo-fmt** — runs `cargo fmt --all` on the workspace.
- **S3 unsafe-doc-comment** — adds a `# Safety` doc paragraph to
  `crates/dismantle-core/src/metal/mod.rs::new_buffer_no_copy`.

## Closeout

On haul end (success or halt), `_phase1_haul2_attempt${N}_blocked.md` is
written by the runner if a halt fired; an attended-session agent fleshes
in the per-layer status table from the evidence triples. On success, an
additional `_phase1_haul2_attempt${N}_closeout.md` records the
attestation summary.

## Gate runner manifest

The fenced block below is parsed by `tools/haul/run-gates.sh`. The
`# layer:` markers set the current halt-budget context for everything
that follows until the next marker.

```
# layer: pre-flight
P0.1 verify-evidence phase1
P0.2 cargo-build
P0.3 cargo-test-strict --workspace --lib

# layer: impl
H2.1 cargo-test-strict --release --test phase1_kernel_parity test_moe_topk_gate_matches_cpu
H2.2 cargo-test-strict --release --test phase1_kernel_parity test_moe_grouped_gemm_q4_matches_cpu
H2.3 cargo-test-strict --release --test phase1_kernel_parity test_moe_gather_combine_matches_cpu
H2.4 cargo-test-strict --release --test phase1_kernel_parity test_gemm_q4_k_m_fused_matches_cpu

# layer: audit
A1 verify-evidence phase1
A2 cargo-test-strict --workspace --lib
A3 cargo-clippy-baseline 30
A4 cargo-fmt-check
A5 cargo-test-strict --release --test phase1_kernel_parity

# layer: self-improve
S1 patch-apply expand-baseline-probe-state
S2 patch-apply cargo-fmt
S3 patch-apply unsafe-doc-comment crates/dismantle-core/src/metal/mod.rs

# layer: closeout
Z1 noop super-closeout
```
