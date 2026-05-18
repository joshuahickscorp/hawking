# Path-to-90 foundation block — halt at step 9 [RESOLVED]

**Halted at:** 2026-05-18 EDT
**Resolved:** 2026-05-18 EDT (later same session)
**Halted on:** step 9 (bit-identical greedy regression)
**Halt rule:** CLAUDE.md § Halt rule — "Below a gate's lower bound → halt and debug, do not paper over."

## Resolution (added post-fix)

Step 9 bit-identical regression now passes. Fix landed in commit
`<step 9 closeout>`: the Eagle4 decode branch routes emission
through GPU `forward_token_argmax` (production Wedge C path,
bit-identical to `SpeculateMode::Off` by construction) while keeping
the CPU walk for eagle4 hidden capture as a parallel forward. The
CPU walk's seq_len bump is save/restored around it so GPU and CPU
KV both advance to exactly seq_len = X+1 per step (each on its own
KV mirror, no shared-counter corruption).

Test run (`EAGLE4_PARITY_TEST=1 DISMANTLE_EAGLE4_GREEDY_TOKENS=8 …
eagle4_decode_parity`):

```
Off:    33747, 855, 254, 24547, 5025, 5025, 5025, 5025
Eagle4: 33747, 855, 254, 24547, 5025, 5025, 5025, 5025
-> 0/8 mismatches  (status: ok)
```

The original CPU `attention()` divergence (root cause section below)
is NOT fixed in this commit — only routed around. Eagle4 mode's
emitted output is correct because emission goes through the GPU
forward. The eagle4 HEAD is still called per step with hiddens from
the (still-divergent) CPU walk — so eagle4's accept/reject
statistics are noisy/unreliable until the CPU `attention()` fix
lands via the spawned chip.

Known follow-on costs of routing around (not fixing):

- **Slower decode in Eagle4 mode.** Per output token: 1× GPU forward
  (~40 ms on M3 Pro) + 1× CPU walk (~3.7 s on M3 Pro) = ~3.74 s/token
  vs Off mode's ~40 ms/token. Stage 1 measurement (step 10) is
  expected to land near 0.2-0.3 tok/s in Eagle4 mode, FAR below the
  18-24 tok/s block-ship band. Step 10 will trigger its own halt
  per the plan; that's the right signal — it's pointing at the next
  architectural unlock (GPU-side eagle4 capture).

- **Unreliable eagle4 stats.** `draft_accepted` / `draft_rejected`
  counters reflect "does eagle4 head's prediction (on CPU-divergent
  hiddens) agree with GPU's argmax?" — interesting only after the
  CPU attention() fix lands. Until then, treat the counters as a
  smoke signal (non-zero = head is being called) not a metric.

Followups still needed:

1. **CPU `attention()` divergence fix.** Chip spawned 2026-05-18.
   Unblocks accurate eagle4 stats AND removes the need for the dual
   forward (we could trust the CPU walk's argmax and skip the GPU
   verifier, halving the per-step cost).

2. **GPU-side eagle4 capture.** The 3.7 s/token cost comes from
   the full CPU walk used to extract h_low/h_mid/h_high/h_shared.
   The production unlock is to instrument the Wedge C TCB path to
   read x_buf at layers 2/13/25 and call ffn_shared_only at layer
   26 GPU-side. That's ~half a day of focused Metal work; deferred
   until step 10 measurement makes the cost concrete.

## What ran in the architectural batch

| Step | Status | Commit |
|---|---|---|
| 1 — Stage 0 baseline profile | ✅ | `72e3926` |
| 2 — Stage 0.5 MLX decision (mandatory, deferred to after step 10) | ✅ | `48be7a1` |
| 3 — `Engine::forward_token_eagle4_for_test` capture | ✅ | `711893c` |
| 4 — `eagle4.py eval --dump-logits` | ✅ | `64cb5c4` |
| 5 — `Eagle4Head::forward_full` CPU forward | ✅ | `540f9d8` |
| 6 — eagle4 parity test (Rust vs Python at 1e-5) | ✅ | `6411d21` |
| 7 — Metal-accelerated head forward | deferred (stretch goal; see step-8-commit-message reasoning) | — |
| 8 — `--speculate eagle4` CLI + decode-path wire-up | ✅ ships output | `f3ae7fd` |
| **9 — bit-identical greedy regression** | **❌ halt** | this commit (test landed, currently fails) |

## Root cause — LOCALIZED to CPU `attention()` (update)

**Halt update 2026-05-18 EDT (post-checkpoint):** A focused A/B diagnostic
test (`crates/dismantle-core/tests/eagle4_cpu_gpu_ab.rs`, gated behind
`EAGLE4_PARITY_TEST=1`) reduces the suspect surface to dismantle's CPU
`attention()` helper. The test runs three forwards on the SAME token at
pos=0 with reset KV (no prior context):

```
GPU Wedge C       | argmax=   185  post-final-norm hidden L2=18.34
CPU walk (full)   | h_high L2=54.18   (3× too large)
CPU shared_only   | argmax= 13699  logits L2=1778
```

`CPU shared_only` uses CPU `attention()` per layer but neutralizes
`ffn()` via the (separately-buggy) `ffn_shared_only` zero-output path.
Since `attention()` is essentially the ONLY non-zero contribution to the
residual stream in `shared_only`, and `shared_only`'s argmax (13699)
disagrees with GPU's (185), **the divergence sits in CPU `attention()`
itself**, not in `ffn()` or the MoE dispatch.

Implication: `forward_token_shared_only` has been silently producing
numerically wrong output vs the GPU forward forever. The
`v0511_forward_shared_only_smoke.rs` test only asserts "logits are
finite" — it never compared CPU vs GPU argmax, so the divergence went
unnoticed. Any path-to-90 prep work that used `forward_token_shared_only`
for acceptance-rate measurement (Phase 3 prep) needs re-validation
against GPU after this is fixed.

Magnitude pattern: CPU walk's per-layer residual is ~1.04× too large.
Compounded over 27 layers, 1.04^27 ≈ 2.88×, which matches the observed
~3× h_high L2 inflation. Suggests a small but systematic over-
contribution per attention layer rather than a wholesale mis-shape.

---

### Original observation (pre-localization)

`forward_token_eagle4_capture_with_argmax` (the CPU walk added in step 8)
diverges from `forward_token` (production GPU Wedge C) at *argmax* level
on the very first decode step. Not subtle fp drift — entirely different
tokens.

Smoke run (prompt `"The quick brown fox"`, 8 tokens, greedy):

```
Off:    33747, 855,   254,   24547, 5025,  5025,  5025,  5025
Eagle4: 257,   9442,  78887, 71199, 21700, 28542, 61122, 96978
```

The Off-mode and Eagle4-mode forwards share architectural intent (full
V2-Lite forward through 27 layers + final-norm + LM-head + argmax) but
exercise disjoint code paths:

- **Off mode (Wedge C):** GPU Metal kernels end-to-end. KV cache lives
  in `mla_c_kv_gpu[li]` Metal buffers, written by `kv_append_f32_tcb`
  Metal kernel.
- **Eagle4 mode (CPU walk):** dismantle's CPU `attention()` + `ffn()`
  helpers per layer. These DO dispatch through Metal where available
  (gemv_q4_k_m_v2_pinned, mla_decode_and_o_proj_metal, etc.) BUT also
  do CPU-side bookkeeping (mla_kv_append into the CPU `mla_c_kv[li]`
  Vec, separate from the GPU buffer).

The CPU KV mirror and GPU KV buffer are populated by different write
paths and *do not stay in sync*. Specifically:

1. **GPU prefill leaves CPU KV at zeros.** The Wedge C path
   one-way-syncs `mla_c_kv → mla_c_kv_gpu` at first call (line ~2879
   of `deepseek_v2.rs`), never the reverse. After GPU-prefill an
   eagle4 decode step calls `attention()` which reads CPU
   `mla_c_kv[li][0..prompt_len]` — these are all zeros.

2. **CPU prefill (one attempted fix this session) didn't resolve it.**
   Running the CPU walk for prefill DOES populate CPU KV, but the CPU
   walk's first decode-step argmax (`257`) still diverged from GPU's
   (`33747`). So the issue is not (only) "CPU KV is unpopulated" — the
   CPU walk's V2-Lite forward produces a fundamentally different
   answer from GPU Wedge C even when both have populated prior KV in
   their respective mirrors. The divergence is structural, not a sync
   bug.

The most likely structural causes (in priority order):

- **CPU `attention()` uses a different MLA dispatch.** `attention()`
  at line 3546 dispatches `gemv_f32_attn_pair_dispatch` for q/kv
  projections and `mla_decode_and_o_proj_metal` for the attention math
  when Metal is available — but the path differs from Wedge C's
  `encode_attention_phase1_into_tcb` + `dispatch_mla_decode_and_o_proj`
  in subtle ways (e.g. RoPE position application, scale factor,
  how `q_full` is shaped before MLA decode). A side-by-side audit of
  the two MLA code paths is needed.

- **`ffn()` MoE-fused path may use a different fused kernel than
  Wedge C's `encode_moe_block_batched_indexed_tcb_with_scratch`.**
  `moe_block_batched_dispatch` (line ~2144) is the unfused fallback
  inside `ffn()`; the fused path goes through a different dispatcher.
  Comparing the dispatcher selection for CPU-call-site vs
  Wedge-C-call-site would clarify.

- **The latent zero-output bug in `ffn_shared_only`** (surfaced in
  commit `711893c`) suggests the unfused MoE-shared dispatch is
  broken; possibly `ffn()`'s shared-expert leg inherits the same
  bug when called outside the Wedge C single-TCB context. Pursued in
  the spawned chip but not yet fixed.

## What attended work unblocks

Updated investigation order (post-localization):

1. **Audit dismantle's CPU `attention()` (line ~3546 of
   `crates/dismantle-core/src/model/deepseek_v2.rs`) vs the GPU Wedge
   C MLA attention path** (`encode_attention_phase1_into_tcb` +
   `encode_attention_phase2_tcb` + `dispatch_mla_decode_and_o_proj`).
   The CPU `attention()` is the ONE thing `shared_only` exercises
   besides effectively-zero `ffn_shared_only`, and `shared_only`'s
   argmax diverges from GPU's. Per-layer the over-contribution
   appears to be ~4 % (1.04^27 ≈ 2.88 matches the observed 3× h_high
   inflation). Likely suspects:
   - RoPE application: position math, theta, dim split
   - Softmax scale factor (1/√d): pre/post softmax
   - MLA q-shape conventions (head-major vs interleaved)
   - Residual already included in `attention()`'s return value (one
     possible "off by a residual add" bug — would explain the ~2×
     inflation if attention returns `x + attn_out` rather than just
     `attn_out`)
   - `mla_decode_and_o_proj_metal` (used by CPU path) vs
     `dispatch_mla_decode_and_o_proj` (used by Wedge C) producing
     numerically different results at the same input.

2. **Concrete diagnostic next step: per-layer L2 dump for both paths.**
   Add a `DISMANTLE_LAYER_TRACE=1` env-var instrumentation that
   prints `x` and `x_norm` L2 norms at each layer boundary in BOTH
   the CPU walk and `forward_token`. First diverging layer + the size
   of the diff fingerprints the kernel(s). The existing diagnostic
   test `eagle4_cpu_gpu_ab` captures the *end-of-pipeline* divergence
   already — the next iteration captures the *first* divergence.

3. **Resolve the spawned `ffn_shared_only` zero-output chip** (step 3
   `711893c`). Independent of the attention divergence; impacts both
   acceptance-rate measurement using `shared_only` and the eagle4
   capture's h_shared component. Already queued as a chip.

4. **Once CPU `attention()` divergence is fixed, re-run both
   diagnostic tests:**

   ```bash
   EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --release \
       --test eagle4_cpu_gpu_ab -- --ignored --nocapture
   # Expected: shared_only argmax == GPU argmax (~185 on the BOS prompt)

   EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --release \
       --test eagle4_decode_parity -- --ignored --nocapture
   # Expected: zero token-level mismatches between Off and Eagle4
   ```

## What's parked vs. what's blocked

**Parked (Class A; pursue after divergence is fixed):**

- Step 10 — Stage 1 measurement gate (12–22 tok/s expected). Cannot
  run honestly until Eagle4 mode produces bit-identical output.
- K-batched verify infrastructure (verify_window > 1; longest-
  matching-prefix acceptance). Step 8 ships K=1 verify-by-comparison,
  which is degenerate w.r.t. K; meaningful K>1 testing needs Path B
  kernels (Stage 2 / execution_plan.md steps 12–17).

**Class B (independent of this halt):**

- Stage 0.5 MLX-pattern adoption (step 2's mandatory verdict). Can
  begin in parallel — it doesn't depend on Eagle4 spec decode landing.
- Routing recall fine-tune (step 11). eagle4-side Python work, runs
  off-machine.

**Architectural batch that DID land successfully:**

- The complete eagle4 head infrastructure: NPZ loader, frozen weights
  loader, CPU forward (1e-5 parity vs Python), parity test, CLI flag,
  decode-loop wire-up. The head IS callable and the parity test
  proves the forward is correct. The unresolved divergence is on the
  V2-Lite-side CPU-vs-GPU path, not the eagle4 head itself.

## Tone

Per CLAUDE.md § "On disagreement with this contract": this halt
follows the halt rule literally. The fix is structural (CPU/GPU MLA
divergence audit, possibly tied to the ffn_shared_only chipped fix)
and belongs in a focused attended session, not a "just this once"
exception in the current commit.
