# Phase 2 — Wedge 1 manifest (weight-pinning + MLA/Q-LoRA Metal migration)

The first Phase-2 haul. Lands two of the three speed items captured in
`_phase2_speed_followups.md`:

- **W1A (item ②) — persistent device-side weight buffers.** Stop the
  per-dispatch `new_buffer_with_bytes` memcpy. Pin every kernel-bound
  weight tensor as a `metal::Buffer` once at load and add `*_pinned`
  variants of the kernel entry points that take `&Buffer`.
- **W1B (item ③) — MLA / Q-LoRA gemvs onto Metal.** Wire `q_a_proj`,
  `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj` to
  `gemv_f32_attn_metal` via four new dispatchers on `DeepSeekV2`,
  identical to the A1.1–A1.5 pattern from haul 3.

FlashDMoE batched expert dispatch (item ① — the big unlock) is **not**
in this wedge. It needs a new shader, an additional parity-test layer,
and ~3–5 days of focused work. It gets its own manifest as **Phase 2
Wedge 2**. The Stage-1 perf gate (the deferred B4 layer from haul 3)
becomes **Phase 2 Wedge 3**, run after Wedge 2 lands.

Estimated cumulative speedup from this haul alone: **~1.4× × ~1.3× ≈
1.8× decode** vs the post-haul-3 ~0.13 dec_tps baseline. That is
nowhere near closing the Stage-1 ratio (`r_llama ≥ 1.5`); it validates
the "production model uses Metal end-to-end with persistent buffers"
claim and removes the two cheapest sources of dispatch-amortizable
overhead before Wedge 2 spends days on the actual unlock.

Layers in execution order:

1. **Pre-flight** — re-attest haul-3 evidence, clean release build,
   lib tests green.
2. **Impl W1A — weight-pinning.** Buffer fields on `Layer` and
   `DeepSeekV2`, `*_pinned` kernel entry points, dispatcher rewires.
3. **Impl W1B — MLA/Q-LoRA Metal migration.** Four new dispatchers
   in `model::deepseek_v2::attention`.
4. **Audit** — clippy / fmt / parity / lib-test re-attest. Drift
   logged, never ends haul.
5. **Closeout** — always runs.

## Pre-launch attended-session prep (NOT haul gates)

The runner needs a small edit before this manifest can launch:
extend the layer-name `case` block in `tools/haul/run-gates.sh`
(currently around lines 787–810, where `impl-A`, `impl-B5`, and
`impl-B4` cases live) with two new entries:

```bash
impl-W1A)
    log "impl-W1A halt at $gate_id — pinning regression, ending haul"
    HAUL_HALTED=1
    ;;
impl-W1B)
    log "impl-W1B halt at $gate_id — MLA migration regression, ending haul"
    HAUL_HALTED=1
    ;;
```

Both follow the same 1-halt-ends pattern as `impl-B5`. Land this
edit as a single attended commit with subject `phase 2: runner —
add impl-W1A/W1B layer cases`. Without it the runner halts
defensively the moment it sees an unknown layer name (per the
diagnostic from haul 3 attempt 2).

No new validator kinds are needed — every gate in this manifest uses
existing kinds (`verify-evidence`, `cargo-build`, `cargo-test-strict`,
`dismantle-token-regression`, `cargo-clippy-baseline`, `cargo-fmt-check`,
`noop`).

## Time discipline

- **Per-item soft ceiling: 60 min** (standard). Hits trigger a
  per-item halt.
- **Haul hard ceiling: 4 hr** (the spec default — no waiver this
  haul). Wedge 1 is small; it should finish well under 1 hr if
  green, well under 2 hr with one halt. If it's threatening 4 hr
  something is genuinely wrong.

## Halt budgets

Per the user's brief: each item is a small focused diff, so any
failure inside it is the finding. 1-halt-ends throughout the impl
layers.

| Layer | Halt rule |
|-------|-----------|
| pre-flight | 1 halt = end haul |
| impl-W1A (weight-pinning) | 1 halt in {W1A.1..W1A.3} = end haul |
| impl-W1B (MLA/Q-LoRA Metal) | 1 halt in {W1B.1..W1B.3} = end haul |
| audit | record-and-continue |
| closeout | always runs |

Asymmetry vs haul 3: there is no deliberate "one continue-on-halt"
slack on either impl layer. W1A.1 is `cargo test --lib` after the
pinning impl — if it red-lines, the impl is broken and there is
nothing useful for W1A.2 to attest. Same for W1B.

## Launch

No model pre-pull is needed (no MLX bench in this wedge). Launch
directly via the haul runner:

```bash
SLM_PID=$(pgrep -f mamba_byte_train | head -1) \
  HAUL=p2w1 \
  PER_VALIDATOR_TIMEOUT_S=2400 \
  HAUL_COOLDOWN_S=0 \
  ./tools/haul/coexist.sh launch phase2
```

Notes:

- `HAUL=p2w1` so blocked / closeout docs land at
  `_phase2_w1_attempt${N}_{blocked,closeout}.md`. (Confirm the runner
  honors arbitrary `HAUL` strings in its closeout-path templating; if
  not, the attended fix is one line near the end of `run-gates.sh`.)
- `SLM_PID` empty is fine — `coexist.sh probe` falls back to
  absolute pressure when slm isn't running. Wedge 1 doesn't co-exist
  with anything heavy by default.
- `HAUL_COOLDOWN_S=0` mirrors haul 3 — no inter-gate sleep when
  there's no GPU contender.
- `verify-evidence phase2` in pre-flight: implementation-wise this
  re-attests every gate listed under `tools/haul/_evidence/` whose
  evidence file declares `phase: phase1` or `phase: phase2`. The
  haul-3 evidence is the only thing currently there; this wedge's
  pre-flight is effectively "haul 3 still passes."

## Layer 2 — Impl W1A: weight-pinning

**Problem.** Every kernel dispatch in the post-haul-3 path calls
`MetalContext::new_buffer_with_bytes(weight_bytes)`
([metal/mod.rs:119](crates/dismantle-core/src/metal/mod.rs#L119)),
which memcpys the host-side weight slice into a fresh Metal buffer.
Per the followups breakdown that's ~220 ms/token of redundant
allocation + memcpy, dominated by the LM head (400 MB × 1/token),
o_proj (16 MB × 27 layers), and routed-expert (1.4 MB × 189
dispatches). The weights never change, so all of this is waste.

**Fix.** Allocate one `metal::Buffer` per weight tensor at
`Engine::load` time, hold it on the model, and pass `&Buffer` to a
new family of `*_pinned` kernel entry points. Existing byte-slice
entry points stay (they remain the parity-test surface — see
"Risks acknowledged"). Only the production forward path migrates.

**Files to edit.**

- [crates/dismantle-core/src/model/deepseek_v2.rs](crates/dismantle-core/src/model/deepseek_v2.rs)
  - `pub struct Layer` (around line 159) — add `metal_buffers:
    LayerMetalBuffers` (a new struct holding `Option<Buffer>` for
    each kernel-bound weight: `attn_norm`, `ffn_norm`, `kv_a_norm`,
    `q_a_norm`, `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`,
    `kv_b_proj`, `q_proj`, `o_proj`, `gate`, plus per-expert
    `Vec<ExpertMetalBuffers>`).
  - `pub struct DeepSeekV2` (around line 139) — add `embed_buf`,
    `final_norm_buf`, `lm_head_buf` Buffer fields.
  - `Engine::load` (around line 424 where `metal_ctx` is
    initialized) — when `metal_ctx.is_some()`, walk every weight
    tensor and call `ctx.new_buffer_with_bytes` once, store on the
    model. Use `unsafe new_buffer_no_copy`
    ([metal/mod.rs:141](crates/dismantle-core/src/metal/mod.rs#L141))
    for the GGUF-mmap'd Q4_K bytes (page-aligned, no-copy keeps
    them in the mmap region).
  - `*_dispatch` helpers (lines 545–628) — call the new `*_pinned`
    kernel entry points when the buffer is available; fall through
    to byte-slice kernels otherwise (CPU path stays valid for
    non-macos and parity tests).
- [crates/dismantle-core/src/kernels/mod.rs](crates/dismantle-core/src/kernels/mod.rs)
  - Add `rmsnorm_metal_pinned`, `gemv_f16_metal_pinned`,
    `gemv_f32_attn_metal_pinned`, `gemv_f32_moe_metal_pinned`,
    `gemv_q4_k_m_pinned` next to their existing twins (lines 263,
    331, 401, 416, 237). Each takes `&Buffer` instead of the
    weight-bytes slice; everything else identical.

**New parity test.**
[crates/dismantle-core/tests/phase2_weight_pinning_parity.rs](crates/dismantle-core/tests/phase2_weight_pinning_parity.rs)
— new file. For each pinned kernel, generate a fixed-seed input,
run both the byte-slice variant and the pinned variant, assert
`atol=1e-3` element-wise. Same precision regime as the haul-1/2
kernel parity tests (`phase1_kernel_parity.rs`).

**Gate runner items.**

| Gate | Validator | Notes |
|------|-----------|-------|
| W1A.1 | `cargo-test-strict --workspace --lib` | Pre-existing 15+ lib tests still pass. Catches `cargo build` failures + lib-side regressions. |
| W1A.2 | `cargo-test-strict --release --test phase2_weight_pinning_parity` | New parity test. Pinned vs byte-slice atol=1e-3 fp16. |
| W1A.3 | `dismantle-token-regression _phase1_token_baseline_50.hashes` | End-to-end determinism on the locked Phase-1 baseline. |

W1A.3 is the load-bearing item: weight-pinning cannot change the
forward path's bit-pattern output (same kernels, same data, just
fetched once). If 50/50 prompts still match, the pinning lands.

## Layer 3 — Impl W1B: MLA / Q-LoRA gemvs onto Metal

**Problem.** Haul 3 wired only the LM-head, o_proj, and MoE
gate-logits gemvs onto Metal. The MLA gemvs in `attention()`
([model/deepseek_v2.rs:680+](crates/dismantle-core/src/model/deepseek_v2.rs#L680))
still go through the CPU `gemv_f32` path:

| Call | Site | Op count | macOS Metal target |
|------|------|---------:|--------------------|
| `q_a_proj` | line 694 | 2048×1536 × 27 layers ≈ 81 M ops/token | `gemv_f32_attn_metal_pinned` |
| `q_b_proj` | line 697 | 1536×3072 × 27 ≈ 127 M ops/token | `gemv_f32_attn_metal_pinned` |
| `kv_a_proj_with_mqa` | line 708 | 2048×576 × 27 ≈ 32 M ops/token | `gemv_f32_attn_metal_pinned` |
| `kv_b_proj` | line 729 | 576×2048 × 27 ≈ 32 M ops/token | `gemv_f32_attn_metal_pinned` |

Total ~270 M scalar fp32 ops/token at ~70 ms wall-clock CPU; ~5 ms
on Metal once W1A's pinning has amortized the dispatch overhead.

The `q_proj` fallback path (no Q-LoRA branch, line 699) is **not**
migrated this haul: DeepSeek-V2-Lite always uses Q-LoRA so the
branch never executes. Migrating it is dead-code work; skip until a
non-LoRA model lands.

The `q_a_norm` rmsnorm at line 696 already routes through
`rmsnorm_dispatch` from A1.1; nothing new there.

**Fix.** Reuse `gemv_f32_attn_metal_pinned` (added in W1A). Add four
new dispatchers on `impl DeepSeekV2`, each modelled exactly on the
existing `gemv_f32_attn_dispatch` (line 574):

- `q_a_proj_dispatch`
- `q_b_proj_dispatch`
- `kv_a_proj_dispatch`
- `kv_b_proj_dispatch`

Each picks Metal when `metal_ctx.is_some()` AND the corresponding
pinned buffer exists, else falls through to `gemv_f32`. Replace the
four bare `gemv_f32(...)` calls in `attention()` with dispatcher
calls.

**Files to edit.**

- [crates/dismantle-core/src/model/deepseek_v2.rs](crates/dismantle-core/src/model/deepseek_v2.rs)
  - Add four `_dispatch` helpers in the dispatcher cluster (around
    line 574, alongside `gemv_f32_attn_dispatch`).
  - Replace 4 call sites in `attention()` (lines 694, 697, 708, 729).

That's it — everything else (buffer field plumbing, kernel
function) is already in place from W1A.

**New parity test.**
[crates/dismantle-core/tests/phase2_mla_metal_parity.rs](crates/dismantle-core/tests/phase2_mla_metal_parity.rs)
— new file. Drive a single MLA attention block on a fixed-seed
input through both the CPU `gemv_f32` path and the new dispatcher
path, assert `atol=1e-3`. Single test function per dispatcher (4
total) plus one composite end-to-end attention block.

**Gate runner items.**

| Gate | Validator | Notes |
|------|-----------|-------|
| W1B.1 | `cargo-test-strict --workspace --lib` | Lib tests still pass after dispatcher rewires. |
| W1B.2 | `cargo-test-strict --release --test phase2_mla_metal_parity` | New parity test. |
| W1B.3 | `dismantle-token-regression _phase1_token_baseline_50.hashes` | End-to-end determinism. |

Same load-bearing structure as W1A: if 50/50 still match, the
migration lands.

## Layer 4 — Audit

Identical to haul 3's audit layer. Clippy baseline stays at 30 (it
was bumped post-haul-2 to absorb Wedge 2 lint debt). New code in
W1A/W1B is small enough that a fresh bump is unlikely; if clippy
trips it's record-and-continue and an attended-session followup
either fixes the lint or bumps the baseline.

| Gate | Validator |
|------|-----------|
| AU1 | `verify-evidence phase2` |
| AU2 | `cargo-test-strict --workspace --lib` |
| AU3 | `cargo-clippy-baseline 30` |
| AU4 | `cargo-fmt-check` |
| AU5 | `cargo-test-strict --release --test phase1_kernel_parity` |

AU5 stays — Phase-1 kernel parity is the regression floor for
every kernel-touching haul. The new Phase-2 parity tests are
already attested at W1A.2 / W1B.2 inside the impl layers, so they
don't need an audit re-run.

## Closeout

On haul end:

- If a halt-budget threshold tripped, `tools/haul/run-gates.sh`
  writes `_phase2_w1_attempt${N}_blocked.md` with root cause + what
  attended work unblocks + followups (per CLAUDE.md tone-of-artifacts
  rule).
- An attended-session agent then writes
  `_phase2_w1_attempt${N}_closeout.md` with per-layer pass/fail
  table, before/after dec_tps comparison from a manual probe (the
  perf gate proper is Wedge 3), and audit drift summary.

## Gate runner manifest

```
# layer: pre-flight
P0.1 verify-evidence phase2
P0.2 cargo-build
P0.3 cargo-test-strict --workspace --lib

# layer: impl-W1A
W1A.1 cargo-test-strict --workspace --lib
W1A.2 cargo-test-strict --release --test phase2_weight_pinning_parity
W1A.3 dismantle-token-regression _phase1_token_baseline_50.hashes

# layer: impl-W1B
W1B.1 cargo-test-strict --workspace --lib
W1B.2 cargo-test-strict --release --test phase2_mla_metal_parity
W1B.3 dismantle-token-regression _phase1_token_baseline_50.hashes

# layer: audit
AU1 verify-evidence phase2
AU2 cargo-test-strict --workspace --lib
AU3 cargo-clippy-baseline 30
AU4 cargo-fmt-check
AU5 cargo-test-strict --release --test phase1_kernel_parity

# layer: closeout
Z1 noop super-closeout
```

## Risks acknowledged

- **Dual-path policy for parity tests is implicit and untested.**
  W1A keeps the byte-slice kernel entry points alongside new
  `*_pinned` variants. Existing `phase1_kernel_parity.rs` stays on
  the byte-slice path; new `phase2_weight_pinning_parity.rs` covers
  the pinned variants. This means we now have two surfaces to keep
  numerically identical forever. If someone later "unifies" them by
  deleting the byte-slice path, every Phase-1 parity attestation
  silently changes meaning. **Decision needed in this haul's
  closeout:** does the dual-path policy stick, or do the Phase-1
  parity tests migrate onto the pinned path too (and the byte-slice
  entry points become deprecated)? Recommend deciding *after*
  Wedge 1 ships so we have data on whether the byte-slice path is
  actually exercised anywhere besides parity tests.
- **W1B is mostly mechanical, but `attention()` is denser than
  `forward_token()` was.** A1.1–A1.5 swapped one call per kernel; W1B
  swaps four calls in a single function. The risk is a typo (e.g.,
  passing `kv_b_proj` weight to `q_b_proj_dispatch`). The parity
  test (W1B.2) catches it; the token regression (W1B.3) catches it
  for a second time. Cost of the redundancy is ~17 min of B5.3
  runtime and it's worth it.
- **All shell tooling under `tools/haul/` must be bash 3.2
  compatible.** macOS default bash is 3.2. Haul 3's
  `token-regression.sh` originally used `declare -A` and silently
  produced a false PASS on B5.3 (the EXIT trap masked the
  `set -u` failure). The script is fixed; the policy isn't written
  down anywhere. Restating it here so future shell additions don't
  repeat the bug: **no associative arrays; if you need a map, use
  grep-based lookups against a temp file.** Also: never let an EXIT
  trap reset the script's exit code unintentionally.
- **No FlashDMoE in this wedge.** This will surprise nobody who
  read `_phase2_speed_followups.md`, but it's worth restating in
  the closeout: r_llama after Wedge 1 will still be ~0.16, well
  below the Stage-1 1.5 threshold. Wedge 2 is the perf claim.
- **`HAUL=p2w1` template path is unverified.** The runner's
  closeout-path code may assume `HAUL` is a numeric phase haul
  (`HAUL=3` → `_phase1_haul3_attempt${N}`). Worst case the blocked
  doc lands at `_phase1_haulp2w1_attempt${N}_blocked.md` and an
  attended `mv` fixes it. Pre-flight a dry-run with
  `DRY_RUN=1 HAUL=p2w1 ./tools/haul/run-gates.sh ...` to confirm
  the template before launching for real.

## Cross-references

- Speed-item triage: `_phase2_speed_followups.md`
- Operating contract: `CLAUDE.md`
- Phase-1 closure record: `_phase1_haul3_attempt4_closeout.md`
- Existing kernel parity tests:
  `crates/dismantle-core/tests/phase1_kernel_parity.rs`
- Locked correctness baseline:
  `_phase1_token_baseline_50.hashes` (50 prompts × 3 tokens, post-Metal)
- Forward path under change:
  `crates/dismantle-core/src/model/deepseek_v2.rs::forward_token`
  and `::attention`
- MetalContext API:
  `crates/dismantle-core/src/metal/mod.rs`
- Runner: `tools/haul/run-gates.sh` (layer cases ~787, validator
  dispatch ~355)

## What's next after this wedge

- **Wedge 2 manifest:** FlashDMoE batched expert dispatch
  (item ① in the followups doc). New shader
  `moe_block_fused` in `shaders/moe.metal`, `moe_block_fused_metal`
  host dispatcher, new parity test
  `phase2_moe_block_parity.rs`, replace the
  `for (eid, weight) in routes` loop in `model::deepseek_v2::ffn`.
  Estimated 3–5 days; deserves its own halt budget. Expected
  decode uplift ~15–30×.
- **Wedge 3 manifest:** Stage-1 perf gate. Re-attempt the deferred
  B4 layer from haul 3. The MLX competitor + bench validators are
  already wired; this is "run them, assert ratios, write closeout."
  Should fit in ≤1 hr if W2 lands the unlock. The Phase-2 ROADMAP
  acceptance bar (`≥2× prefill on 1024-token prompts vs Phase 1;
  decode ≥0.9× MLX`) is what Wedge 3 actually attests.
