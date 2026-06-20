# STRAND speed moonshot research

_2026-06-15 pass. Goal: find levers big enough to beat llama.cpp-class serving, not just
close the gap. This note covers the quant/runtime path, not media compression._

## Decision

The current G4 bitslice shape is still the kernel to integrate. The new scratch gates
mostly killed tempting micro-optimizations:

| lever | gate result | verdict |
|---|---:|---|
| Staged/coalesced Q12 decode writes | k3: 0.48x, k2: 0.21x vs deployed decode | kill |
| Compact GPU table, 84 B/block -> 40 B/block | k3: 0.78x, k2: 0.99x fused B=1 | kill |
| Computed codebook instead of staged LUT | +1-2% fused B=1 | too small |
| Cooperative windowed per-block decode | 0.37x k3, 0.26x k2 | kill |
| Vector trellis d=2 fused B=1 | 1.13x best, identity clean | marginal, quality/training-gated |

The lesson is sharp: the wall is not "one obvious bad load" anymore. The simple
one-thread-per-block bitslice kernel is already the least-bad local shape. To get
llama.cpp-destroying speed, the next levers must change the problem size or serving
semantics, not shave the current inner loop.

## Local Gates Run

### 1. Staged/coalesced decode

Fixed a real ABI drift first: `strand_bitslice_staged.metal` had an 80 B Metal
`BitsliceEntry`, while the host struct is 84 B after the `d` field. After adding `uint d`,
the sizeof probe passed.

Command:

```bash
cargo run -p strand-decode-kernel --release --bin gate-bitslice-staged -- --force
```

Result:

| config | deployed | staged | ratio |
|---|---:|---:|---:|
| k3 L7 | 10.02 Gw/s | 4.78 Gw/s | 0.48x |
| k2 L12 | 8.50 Gw/s | 1.75 Gw/s | 0.21x |

Identity: 12 staged cells byte-identical to `decode_tensor_fixed`.

Interpretation: coalescing the Q12 stores via a threadgroup tile costs more than the
scatter stores cost. Do not wire this into `BitsliceGpu`.

### 2. Compact table

Command:

```bash
cargo run -p strand-decode-kernel --release --bin gate-tablecompact
```

Result:

| config | expanded table | compact table | ratio |
|---|---:|---:|---:|
| k3 L7 | 38.20 Gw/s | 29.99 Gw/s | 0.78x |
| k2 L12 | 32.24 Gw/s | 32.03 Gw/s | 0.99x |

Identity: 384 cells byte-identical.

Interpretation: table bytes are not the fused B=1 limiter. At k2, cutting modeled
traffic by about 30% gives no speedup; at k3 it regresses. The extra per-sub-block
expansion cancels or exceeds the memory win.

### 3. Computed codebook and cooperative windowing

Command:

```bash
cargo run -p strand-decode-kernel --release --bin gate-coopwindow
```

Result:

| config | reference fused | computed-codebook | cooperative windowed |
|---|---:|---:|---:|
| k3 L7 | 38.66 Gw/s | 39.27 Gw/s, 1.02x | 14.19 Gw/s, 0.37x |
| k2 L12 | 34.53 Gw/s | 34.88 Gw/s, 1.01x | 8.97 Gw/s, 0.26x |

Identity: serial decode and windowed decode both passed; fused outputs within tolerance.

Interpretation: the data-dependent LUT is not the wall, and parallelizing within the
256-step block by recomputing window states is far too expensive on M3.

### 4. Vector d=2

Command:

```bash
cargo run -p strand-decode-kernel --release --bin gate-bitslice -- --vec-only --no-wait
```

Result:

| config | fused B=1 | ratio vs scalar L12 |
|---|---:|---:|
| scalar L12,d1 | 33.02 Gw/s | 1.00x |
| vector L7,d2 | 37.25 Gw/s | 1.13x |
| vector L10,d2 | 36.16 Gw/s | 1.10x |

Identity: 864 GPU vector cells byte-identical to `decode_tensor_fixed_with_lut_vec`.

Interpretation: vector-d2 is not a speed-only pass. Keep it alive only if PV/training
needs fractional payload rungs anyway.

### 5. Scratch lab hygiene

`cargo check -p strand-decode-kernel --bins` fails because untracked
`gate-chainsplit.rs` references missing `BitsliceGpu::{matvec_split, bench_matvec_split}`.
Either make that gate self-contained like `gate-tablecompact`/`gate-coopwindow`, or remove
it from the scratch set before using all-bins checks.

## Moonshot Levers Still Alive

### A. Column-sign RHT: make activation rotation cheap

This is the cleanest "have cake and eat it" candidate in the codebase.

Current row-sign RHT is exact for quantization, but serving a multi-row tensor requires a
different rotated activation per output row. The test
`outlier_mac::single_rotation_recipe_diverges_for_multirow` correctly prevents the false
shortcut.

The repo already has `rht_forward_cols_*`: same per-row FWHT, but signs are shared by
column (`sign_at(seed, col)`). That makes one transformed activation vector valid for every
output row:

```text
y = decode(W_col_rht) * RHT_cols(x)
```

This is a format/quality gate, not a runtime patch. Run a 0.5B A/B:

1. Add an off-by-default `--rht-cols` encode flag.
2. Quantize q3 and q2/L12/outlier on 0.5B with row-RHT vs col-RHT.
3. Compare PPL and output-KL, not just rel-RMS.
4. Adopt if the quality tax is <=0.5% or if PV erases it.

If it passes, the fused Metal path becomes much more honest: one activation transform per
tensor, not per row, and no row-space decode fallback.

### B. Token-multiplying decode: Medusa/LayerSkip/Lookahead/SpecInfer

Micro-kernels are giving 1.0x to 1.13x. Multi-token verification can give 1.5x to 3x when
acceptance is good. This is how to beat llama.cpp without needing a 5x raw GEMV miracle.

Likely STRAND-specific route:

1. Use the G4 fused/GEMM kernels as the verifier path.
2. Start with exact or near-exact methods that do not require a separate large draft:
   Lookahead Decoding, LayerSkip-style self-speculation, or Medusa heads trained on top of
   the same backbone.
3. For prompt/code workloads, verify candidate trees in B=4/B=16 lanes where G4 is strongest.
4. Measure accepted tokens per full model pass, not just kernel throughput.

This lever is especially aligned with STRAND because our problem is moving/decoding the
same dark weights every token. If one full-weight pass accepts two tokens, the effective
weight-decode cost halves.

### C. Native trained sub-2-bit / ternary runtime

BitNet-style systems get speed because the model is trained for the representation, not
because a PTQ kernel found a clever dequant trick. STRAND's vector/fractional rungs and PV
are the local equivalent.

Do not spend more time on vector PTQ as a product claim. Spend it on progressive PV:

```text
q4 -> q3 -> q2 -> vector d=2
```

If trained vector d=2 quality becomes acceptable, the 1.13x speed result becomes a bonus
on top of density. If quality does not land, the speed win alone is too small.

### D. Whole-token command buffer and no sync gaps

The old prepared gate showed per-tensor Metal commits are expensive. Dismantle integration
must batch all projection dispatches for a token into one command buffer. A kernel that is
3x faster can lose the win through 100+ tiny commits.

Pre-integration gate:

1. Build a synthetic Qwen-shaped token pass with all projection tensors prepared.
2. Dispatch all tensors in one command buffer.
3. Add OUTL as CPU residual only after profiling confirms sync cost is not dominant.
4. Report token-wall, not per-tensor wall.

### E. Same-harness llama.cpp baseline

Before claiming "beats llama.cpp," measure it locally:

1. Same model family and shape where possible.
2. Metal on, flash attention configured, identical prompt/decode split.
3. Record `pp` and `tg` separately.
4. Compare STRAND G4 token-wall, not just decode-primitive Gw/s.

## External Research Signals

- QTIP confirms the trellis/incoherence family and specifically calls out hardware-efficient
  bitshift trellis codes with quality and speed results: https://arxiv.org/abs/2406.11235
- BitNet b1.58 and bitnet.cpp show that trained ternary/1-bit models need custom kernels and
  can produce large CPU speedups, but they are native-trained, not PTQ rescue:
  https://www.microsoft.com/en-us/research/publication/1-bit-ai-infra-part-1-1-fast-and-lossless-bitnet-b1-58-inference-on-cpus/
- Medusa reports multi-token heads with roughly 2.2x to 2.8x speedup ranges:
  https://proceedings.mlr.press/v235/cai24b.html
- LayerSkip shows self-speculative early exit can reach about 2x-class task speedups without
  a separate draft model: https://aclanthology.org/2024.acl-long.681/
- Lookahead Decoding is exact, draft-free, and targets the same bandwidth-bound AR decoding
  waste that STRAND has: https://arxiv.org/html/2402.02057v1
- SpecInfer is the tree-verification serving ancestor: https://arxiv.org/abs/2305.09781
- Arm CPU kernel work shows llama.cpp can be beaten by layout and unpack amortization on Arm,
  but our local gates say STRAND's current M3 limiter is not solved by table compaction:
  https://arxiv.org/abs/2501.00032

## Appendix: Claude Portfolio Reconciliation

User prompt, 2026-06-15: reconcile the Claude moonshot sweep with this repo and search
for custom STRAND-native ways to build on the ideas instead of giving up.

Verdict: the Claude portfolio is directionally useful, but it mixed three readiness levels:

1. **Real and mostly on disk:** C2/SDSQ side-info, de-bias/DBIA math, actmean calibration,
   selective-PV recipes.
2. **Promising but not wired:** full C2F all-stream side-info, deploy-time DBIA, P4
   activation-aware preconditioning.
3. **Names without a STRAND gate yet:** YAQA/output-Fisher rounding, learned rotations,
   LDLQ retests, ANE/AMX offload.

The workable response is not "port AWQ/SmoothQuant/SpinQuant/AQLM." The response is to
translate their mechanisms into STRAND's constraints: deterministic archive, integer decode,
RHT/trellis codec, self-description, and mmap/runtime loaders.

### Reconciled bets

| bet | local evidence | outside signal | workable STRAND version | decision |
|---|---|---|---|---|
| C2/SDSQ side-info and position compression | `sideinfo_wire.rs` is live for scale_q; `c2_final.rs` codes scale_q, sub_scale, outl_pos but is not integrated | integer rANS/entropy coding is standard; no quality risk if byte-identical | C2F whole-model side-info section with shared models, covering scale_q + sub_scale + outlier positions | build now |
| DBIA de-bias | `debias.rs`, `quantize-model --actmean`, PPL docs show q2 -28.7% and q3 -10.9% on 0.5B | output-space correction is orthogonal to weight-MSE; no direct paper port needed | seal DBIA in provenance, parse it in loader, apply one bf16->f32 row add after sparse residual | build now, but seal first |
| P4 activation-aware preconditioning | docs report scalar proxy -15.6% output-error; no `--act-precond` flag yet | AWQ/SmoothQuant show equivalent diagonal activation/weight transforms can preserve model function while improving quantization | P4C: column-scale weights before RHT, store quantized scale vector, apply inverse scale to activations before RHT | prototype now |
| Learned/found rotations | repo has deterministic RHT and per-column RHT primitives; no seed search wired | SpinQuant shows rotation choice matters a lot; QuIP#/QTIP confirm randomized Hadamard/incoherence is central | fast-family seed/permutation search, not arbitrary learned dense rotations | prototype after P4C |
| LDLQ / Hessian | product spec says vanilla block-Hessian/LDLQ lost 9.66 vs 9.42 at 3-bit | QuIP/QuIP# use adaptive rounding under incoherence, but also use different codebooks/fine-tuning | retest only as output-weighted Viterbi after P4C, with activation diagonal/low-rank signal, not old post-RHT block-Hessian | narrow retest |
| YAQA / output Fisher | no repo symbol found | plausible family: minimize output error, not weight RMS | YAQA-lite: use activation covariance / output Fisher to weight trellis distortion; only if P4C clears PPL | speculative |
| PV / PVT | 0.5B docs show PTQ q2 ~80 PPL to PV ~26.77; scripts exist | PV-Tuning says 1-2 bit quality wants train-through-compressed-params; AQLM/QuIP# also lean on tuning | progressive q4->q3->q2 selective PV with DBIA and P4C baked into init | later gate |
| Native ternary / BitNet | no current model architecture match | BitNet speed comes from native ternary training plus custom kernels | separate STRAND-T research branch, not current dismantle integration | do not block |
| Token-multiplying decode | not model-integrated | Medusa/LayerSkip/Lookahead/SpecInfer all attack token/pass, not kernel Gw/s | verify multiple candidate tokens per full STRAND weight pass once G4 path is integrated | speed roadmap |

### Custom solution queue

#### 1. C2F: lossless position + side-info compression

This is the cleanest density moonshot because it does not change reconstruction.

Current state:

- `--sdsq-sideinfo` appends an SDSQ section for `scale_q` only.
- `c2_final.rs` has wrappers for `scale_q`, `sub_scale`, and `outl_pos`, but is not
  declared in `lib.rs` and is not wired into `format.rs` / `encode.rs` / loader.
- The measured target in the repo docs is about **0.237 bpw recovered** at q2 when streams
  are concatenated whole-model so table costs amortize.

Build:

1. Export `c2_final` behind a stable module name or move its winning API into
   `sideinfo_rans`.
2. Add a new section version (`C2F` or `SDSQ v2`) that carries:
   - one whole-model `scale_q` stream,
   - one whole-model `sub_scale` stream,
   - one whole-model sorted/gap-coded outlier-position stream.
3. Keep v1 fixed-width seek tables readable; add an opt-in stream-mode archive that can
   omit or zero redundant fixed-width fields only after the C2F identity gate passes.
4. Teach all EOF-chain walkers the new magic before adding the section to any real file.
5. Add the moat gate: C2F decode -> Q12 decode must be byte-identical to fixed-width decode
   over the same archive, including OUTL and SPRV.

This is the "position compression" bet: outlier positions are the largest recoverable
side channel, and gap coding them is lossless.

#### 2. DBIA: seal and deploy the de-bias quality win

DBIA is not just a research note; the encode-side correction exists. The missing part is
turning it into an attested runtime feature.

Required order:

1. Export `debias_wire` from `strand-quant`.
2. Extend `descriptor_digest` / `verify_archive` to bind a DBIA digest, mirroring OUTL.
3. Make every chained-section reader step over `{SDSC, OUTL, SDSQ/C2F, DBIA, SPRV, RSLT}`
   as appropriate. Silent `Ok(None)` is the danger.
4. Add `StrandModel::debias` or per-tensor correction storage in the decode loader.
5. Apply `y[o] += bf16_to_f32(c[o])` exactly once after the dense MAC and sparse OUTL
   residual.
6. Gate on:
   - DBIA absent == byte-identical old output,
   - DBIA present == reference epilogue,
   - tampered/dropped DBIA fails SPRV verification,
   - C4/PTB + one downstream task before scaling the claim beyond WikiText.

Custom stretch: DBIA and P4 attack different components of output error. DBIA is the DC
post-quant mean correction; P4C is the AC/energy pre-quant weighting. Do not sum their
individual wins until the combined A/B runs.

#### 3. P4C: activation-aware preconditioning inside STRAND

External support:

- AWQ uses activation statistics to find/protect important channels and uses equivalent
  scaling instead of mixed precision.
- SmoothQuant moves difficulty between activations and weights through a mathematically
  equivalent diagonal transform.

STRAND-native design:

```text
D_j = clipped activation energy statistic for input feature j
W_pre[:,j] = W[:,j] / D_j^alpha
x_pre[j]  = x[j] * D_j^alpha
y         = W_pre x_pre
```

Then quantize `W_pre` through the existing RHT + trellis path. For column-sign RHT, the
runtime path becomes:

```text
x_rht = rht_forward_cols(D^alpha * x, seed)
y     = decode(W_pre_col_rht) * x_rht
```

Why this is custom instead of a vanilla AWQ port:

- The transform must happen before STRAND's RHT/trellis encode.
- The activation scale vector must be stored or deterministically regenerated and billed.
- The decode stays integer; only the activation preprocessing changes.
- The P4 scalar win must survive real trellis PPL, not just output-RMS.

Prototype:

1. Extend `calib-actmean.py` output to treat existing `sum_sq` / `sum_abs` as feature
   energy and emit clipped `D`.
2. Add `quantize-model --act-precond <json> --act-precond-alpha <a>`.
3. Sweep `alpha in {0.25, 0.5, 0.75, 1.0}` with clipping percentiles.
4. Run arms: baseline, DBIA, P4C, P4C+DBIA.
5. Promote only if PPL improves on WikiText + C4/PTB + one downstream sanity task.

#### 4. RHT seed/basis search: cheap SpinQuant-shaped bet

SpinQuant's useful warning is that rotations are not all equal. Dense learned rotations
would break STRAND's fast deterministic basis, but a seed/basis search inside the existing
fast family is cheap and moat-safe.

Build:

1. For each tensor, search a small deterministic bank:
   - current seed,
   - alternate splitmix/FNV seeds,
   - optional fixed column permutations,
   - row-sign vs column-sign RHT variants.
2. Score with output-error proxy first, then PPL on promoted candidates.
3. Store only a seed/basis id in metadata.
4. Runtime cost is unchanged for row-RHT; for column-RHT it remains one activation transform
   per tensor.

This is the non-give-up version of "learned rotations": do not learn arbitrary dense
matrices; search the fast family STRAND can actually serve.

#### 5. Output-weighted Viterbi: the only LDLQ/YAQA retest worth running

Vanilla block-Hessian/LDLQ is banked dead in the product spec. The custom retest is:

```text
distortion_j = (w_j - q_j)^2 * activation_energy_j
```

or, one level stronger:

```text
distortion_block = delta^T C_x delta
```

where `C_x` is diagonal or very-low-rank activation covariance. Feed that into Viterbi's
per-candidate cost, not into a post-RHT block-Hessian replay. This reconciles:

- why old Hessian/LDLQ died,
- why P4 seems alive,
- why AWQ-style activation importance is relevant.

Gate:

1. Run diagonal-energy Viterbi on 0.5B for q2 and q3.
2. Compare against P4C. If it cannot beat or stack with P4C, kill.
3. Only try low-rank covariance if diagonal clears PPL.

#### 6. Speed bets that still compose with this queue

The local M3 micro-gates killed staged writes, compact tables, and cooperative windows.
The remaining speed bets are system-level:

- **Column-sign RHT:** if quality holds, one activation transform per tensor instead of
  row-specific transform trouble.
- **Whole-token command buffer:** integrate G4 as one token graph, not per-tensor commits.
- **Token-multiplying decode:** Medusa/LayerSkip/Lookahead-style verification can multiply
  accepted tokens per full STRAND weight pass.
- **C2F stream-mode:** density win first; speed win only if the runtime stops streaming
  redundant side-info in the deployed path.
- **cloud-GPU bitslice:** still a separate pod/GPU bet; do not claim from arithmetic.

### Updated build order

1. C2F bit-identity gate and whole-model position/side-info section.
2. DBIA seal + loader + epilogue apply.
3. P4C `--act-precond` scalar-to-trellis PPL gate.
4. P4C + DBIA overlap A/B on WikiText, C4/PTB, and one downstream task.
5. Column-sign RHT encode A/B, ideally combined with P4C.
6. Seed/basis search inside the fast RHT family.
7. Output-weighted Viterbi retest only if P4C leaves measurable headroom.
8. Whole-token G4 integration, then token-multiplying decode.

This gives STRAND a real bet portfolio:

- **density:** C2F / outlier-position compression, zero quality cost;
- **quality:** DBIA + P4C + maybe output-weighted Viterbi;
- **speed:** column-sign RHT + one-command-buffer runtime + token-multiplying decode;
- **frontier:** progressive PV only after the local stack is sealed and measured.

Sources used for this reconciliation:

- AWQ activation-aware weight scaling: https://arxiv.org/abs/2306.00978
- SmoothQuant diagonal activation/weight migration: https://arxiv.org/abs/2211.10438
- SpinQuant learned/selected rotations: https://arxiv.org/abs/2405.16406
- PV-Tuning for extreme 1-2 bit compression: https://arxiv.org/abs/2405.14852
- AQLM additive codebooks and learned compression: https://arxiv.org/abs/2401.06118
- QuIP# randomized Hadamard + lattice codebooks + tuning: https://arxiv.org/abs/2402.04396
- QTIP trellis coded quantization and computed-code design: https://openreview.net/forum?id=7sdkLVuYCU

### Sanity checks after appending

Commands run:

```bash
cargo test -p strand-quant --test c2_final_harness -- --nocapture
cargo test -p strand-quant --test debias_determinism -- --nocapture
cargo check -p strand-quant --bin quantize-model
```

Results:

- `c2_final_harness`: 26/26 passed. The all-three synthetic q2 measure reported
  `0.24831 bpw` recovered (`scale_q 0.08681`, `sub_scale 0.01366`,
  `outl_pos 0.14784`). This validates the codec bet, not the final archive wiring.
- `debias_determinism`: 15 passed, 2 ignored. The ignored tests are the expected
  production-binding gap: `debias_wire` is not exported/wired into `lib.rs` and the
  deploy path yet.
- `quantize-model` check passed. The only warning was the pre-existing unexpected `cfg(kani)`.

## Appendix: CPU And Throughput Addendum

User prompt, 2026-06-15: check whether STRAND's different structure creates more throughput
opportunities, including CPU, and add only credible progress points.

Verdict: yes, CPU deserves its own track, but the winning shape is **not** "more NEON" or
"histogram everything." The CPU path's edge is lean on-disk density + exact integer decode +
portable fast scalar/block scheduling. A separate **turbo** mode could trade memory for more
tokens/sec by caching decoded/panelized weights.

### CPU gates run

Commands:

```bash
cargo run -p strand-decode-kernel --release --bin gate-cpu-fastpath
cargo run -p strand-decode-kernel --release --bin gate-interleave
cargo run -p strand-decode-kernel --release --bin gate-neonlut
cargo run -p strand-decode-kernel --release --bin gate-neonlut -- --bench
cargo run -p strand-decode-kernel --release --bin gate-histogram
```

Results:

| bet | result | decision |
|---|---:|---|
| CPU fast decode, 3-bit deploy | `decode_q12_fast == decode_lean`; `724-764 Mw/s` decode; `470-481 Mw/s` full decode->GEMV; `0.4175 B/w` vs Q4_K `0.5625 B/w` | ship/keep |
| Scalar ILP interleave | single-core `+11%` at k3 L7, `+13%` at k2 L12; all-core `0.90x` at k3, `1.02x` at k2 | use only for single-thread/WASM/mobile experiments |
| NEON LUT/table gather | identity `1536` cells OK; perf k3 L7 `0.85x` of fast, k4 L8 `0.55x` | kill current shape |
| Histogram GEMV | exact integer dot and B-column identity OK; 3-bit unit-scale has theoretical `0.036` mul ratio but still `0.72x` fused at B=1, `0.65x` at B=16, `0.55x` at B=64; varied-scale worse | kill hot path |

Interpretation:

- The CPU decode core is already decent and moat-safe. It wins on footprint and determinism,
  not raw llama.cpp-class token throughput.
- The current full CPU decode->GEMV path is MAC-bound after fast decode. More decode-only tricks
  cannot create a 5x token win unless they also reduce or amortize the dot product.
- STRAND's structure gives us one CPU throughput stretch that llama.cpp does not naturally
  prioritize: **dual-mode compressed/turbo execution**.

### New progress point: CPU Turbo Mode

Add this to the build portfolio:

```text
normal mode: .strand mmap -> lean integer decode each token
turbo mode:  .strand mmap -> decode once at load -> packed Q12/i16 or f16 panels -> fast CPU GEMV
```

Why this is credible:

- The compressed `.strand` file remains the source of truth.
- The decoded cache is deterministic and can be hashed against Q12 block roots.
- It trades RAM for throughput only when the device has RAM to spare.
- It can be tensor-selective: cache only MLP or hot tensors first.
- It creates a fair CPU head-to-head: STRAND compressed artifact plus optional hot panels versus
  llama.cpp quantized hot weights.

Prototype:

1. Add `StrandCpuPlan::{Compressed, TurboPanels}`.
2. At load, optionally decode selected tensors to row-major and transposed panel layouts:
   - row-major for simple parity,
   - column-major/panel-major for cache-friendly `x` reuse,
   - optional f16 panel if Accelerate/AMX wins and determinism profile allows it.
3. Gate memory:
   - q3 compressed remains `~0.4175 B/w`,
   - Q12/i16 turbo cache costs `2.0 B/w`,
   - tensor-selective cache must report bytes and tokens/sec separately.
4. Gate correctness:
   - cache Q12 equals `decode_q12_fast`,
   - turbo output equals compressed output within the documented float MAC profile,
   - compressed mode remains bit-identical.
5. Bench:
   - B=1 token,
   - B=16/B=64 prompt,
   - hot MLP-only cache,
   - full cache.

This is the CPU version of "have cake and eat it": ship the tiny deterministic artifact, then
optionally expand only on machines where throughput matters more than RAM.

### CPU candidates not promoted

- **Histogram/dendritic summation:** exact and elegant, but killed by bookkeeping on real varied
  scales. Keep only as a correctness/research tool.
- **NEON LUT gather:** bit-exact but slower than scalar fast/split. Do not spend more time unless
  a new table layout removes the gather/scatter overhead.
- **Event/delta MAC:** code exists and identity gates exist, but no real activation/PPL gate has
  promoted it. It remains a future activation-sparsity study, not a throughput progress point.

### Throughput progress list, updated

1. G4 Metal bitslice remains the current GPU integration target.
2. C2F / outlier-position compression is the cleanest density win.
3. DBIA de-bias is the cleanest quality win, but must be sealed in provenance first.
4. P4C activation-aware preconditioning is the best new PTQ quality stretch.
5. Column-sign RHT is the biggest format-level speed stretch if quality holds.
6. Whole-token command buffers are mandatory before dismantle integration.
7. Token-multiplying decode is the biggest serving-level speed bet.
8. **CPU fastpath is real and should stay:** lean `.strand` CPU decode ships at `0.4175 B/w`
   with bit-identical integer decode.
9. **CPU Turbo Mode is the new throughput bet:** optional decoded/panelized Q12 cache for
   RAM-rich devices.
10. Histogram GEMV and current NEON LUT are killed for hot-path throughput.

## Next Order

Two tracks should run in order, without mixing claims:

1. **Local moonshot stack:** C2F bit-identity + DBIA seal/deploy + P4C PPL gate. These are
   the highest-EV density/quality bets and should be settled before any new cloud spend.
2. **Speed integration stack:** keep the staged ABI fix or delete the dead gate, add the
   column-sign RHT quality gate, build the one-command-buffer G4 token pass, add a CPU
   Turbo Mode prototype, and run same-machine llama.cpp Metal/CPU `pp`/`tg` baselines.

Dismantle should receive the proven G4 shape only after the speed integration stack clears;
it should not receive staged decode, compact table, cooperative windowing, or any DBIA/C2F/P4C
feature before those features have their own identity and PPL gates.
