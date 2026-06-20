# STRAND loss-aware Viterbi requant — proposal (Tier 2b)

**Status: PROPOSAL — design only, nothing built, nothing measured.** This is a deliberate,
gated REOPEN of the dead Hessian-objective family (will.md §4: diag-H +44/+45% at 2-bit,
block-H twice dead; "do not re-run any Hessian-weighted variant without first defeating the
RHT-whitening argument"). §5.6 doctrine applies: closed-under-framing-X, not closed-forever.
This document is the new framing and the gate ladder that decides it. Verdicts land in
will.md §4 and the memory ledger; if Phase 0 kills it, total spend is < 30 minutes.

**One paragraph:** weight the Viterbi per-step cost by accumulated grad² from the LIVE PV
training loop (WikiText-train through the actual delta-forward/STE loss, at the current
quantized iterate) instead of an external calibration corpus at the bf16 point. The
gradient is rotated into post-RHT coordinates BEFORE squaring (`g_y = R g_w`, exact chain
rule), so the resulting curvature is the one object the whitening argument does not exactly
flatten. The whole lever is
encoder-side: `EncodedTensor`, the wire format, and `decode_lean`/`decode_tensor_fixed` are
untouched — bit-identity is structurally unaffected. Explicit `--weight-map <file>` only;
no directory sniffing, ever (the `hessian.hsdi` auto-pickup trap is named and closed).

Gate ladder (cheap-first; each phase can kill without running the next):

| phase | question | cost | kill criterion |
|---|---|---|---|
| 0a | does live-Fisher u have within-block structure at all? | <30 min | median within-block CV < 0.10 → DEAD |
| 0b | is that structure real or sampling noise? | same run | median within-block split-half r < 0.2 (after one N=200 retry) → DEAD |
| 1 | does a weighted requant move held-out PPL? | ~2 h | best arm < 2% PPL gain vs unweighted, same checkpoint → DEAD |
| 2 | does it survive the full PV loop? | one night lane | end-of-arm gain < 2%, or gain < Phase-1 gain (PV absorbed it) → DEAD/ABSORBED |

---

## 1. The framing change vs the dead levers

### 1.1 Autopsy: what `--diag-h` actually was (recovered from git, pre-`e059844`)

The dead lever was a **length-`in_features` per-input-column vector**, applied per block as
`dh[(g0 + i) % in_features]` (`block_hessian_weights`, old encode.rs). Three structural facts:

1. **Zero output-row resolution.** One value per input column, broadcast across all 896/4864
   output rows of the tensor.
2. **Diagonal-of-activation-second-moment.** The HSDI contract required H accumulated from
   RHT-transformed activations, i.e. `u_i = diag(R E[x xᵀ] Rᵀ)_i` — the GPTQ proxy curvature
   (layer-local output-MSE), measured near-flat post-RHT.
3. **C4 corpus, bf16 point.** Calibrated once on C4 against the pretrained weights.

Measured verdict (deterministic A/B, only `--diag-h` differs): q2_l12 210→302 (+44%),
q2_l12_out1 80.7→117 (+45%); rel-RMS *fell* while PPL *rose* — the error got reshaped toward
what C4 said and away from what WikiText needed.

### 1.2 The whitening argument, made exact — and what it does NOT cover

Sharpen the §4 verdict from "near-flat" to a theorem, because the reopen has to live in the
gap the theorem leaves.

Per output row, the RHT is `y = R w` with `R = H_h · diag(s)` per length-`h` segment
(`h = 256` for every 256-aligned tensor; `h = 128` on the 0.5B's 896-dim rows —
`pow2_block_for`, rht.rs). Every entry of a normalized Hadamard matrix has `R_ij² = 1/h`
exactly; the Rademacher signs do not change squares. Therefore, for ANY curvature that is
**diagonal in weight space**, `D = diag(d)`:

```
diag(R D Rᵀ)_i = Σ_j R_ij² d_j = (1/h) Σ_{j∈segment} d_j      — EXACTLY constant per segment.
```

Not "near-flat": exactly flat. A diagonal weight-space curvature, transported correctly to the
encode basis, collapses to one number per Hadamard segment. And segment-constants are
**provably inert** in this encoder (see 1.4): the trellis runs an independent fixed-rate
Viterbi per 256-block, and scaling a whole block's cost by a constant changes no argmin.
This is why the Hessian family is structurally closed for STRAND, not just empirically dead.

What the theorem does NOT flatten: the diagonal **in y-space** of a curvature that is
**non-diagonal in weight space**. `diag(R C Rᵀ)_i = R_iᵀ C R_i` varies across `i` exactly by
how the off-diagonal mass of `C` projects onto the different (fixed-seed) Hadamard rows. The
empirical Fisher of the network loss is such an object: for a linear layer, the per-sample
gradient is `g_W = δ ⊗ x` (rank-1), so `F = E[g g ᵀ]` has dense within-row off-diagonal
structure `E[δ_r² x_i x_j]`. The reopen's entire mathematical license is this term.

### 1.3 The new signal, decomposed — including exactly where it dies

Accumulate, in the live PV loop, the EMA of squared **y-space** gradients:

```
u_(r,i) = EMA_t [ ( R · g_w(t) )_(r,i)² ]        with  g_y = R g_w  (chain rule, R orthogonal)
```

For one sample, `(R g_w)_(r,·) = δ_r · (R x)`, so the per-coordinate signal is

```
u_(r,i) = E[ δ_r² · x̃_i² ]          x̃ = R x  (the RHT'd activation)
        = E[δ_r²] · E[x̃_i²]   +   Cov( δ_r² , x̃_i² )
          └────── inert ──────┘     └────── the ONLY live term ──────┘
```

Be adversarial about each piece:

- `E[x̃_i²] = diag(R E[xxᵀ] Rᵀ)_i` is **the old dead signal** (rotated activation second
  moment, measured near-flat).
- `E[δ_r²]` is a per-output-row factor. The row-aware RHT never mixes rows, so it survives
  the transform exactly — but at fixed uniform rate it is **inert** (1.4): for 256-aligned
  tensors each block lies inside one row, so a per-row factor is a per-block constant.
- `Cov(δ_r², x̃_i²)` across samples/positions — whether the loss magnitude co-varies with
  *which rotated coordinate fires* — is the only term that creates reproducible
  **within-block** structure. **If this covariance is ≈ 0, the idea is dead, full stop**, and
  Phase 0 measures it directly for ~30 minutes of compute before any encoder work.

Honest caveat on the 0.5B specifically: with `h = 128` and `block_len = 256`, Hadamard
segments and trellis blocks are misaligned (a block spans 2 segments; every other 256-block
also straddles a row boundary at 896-wide rows). So on the 0.5B even segment-constant signals
leak a 2-level step *inside* blocks. That is a real mechanism but a **scale-specific
artifact**: every 7B tensor is 256-aligned and the channel vanishes there. Phase 0 therefore
reports within-block stats **restricted to non-straddling, single-segment blocks** as the
primary number; the straddle-inclusive number is recorded separately (risk R6).

### 1.4 Why "between-block" structure cannot save it (and didn't save diag-H)

The encoder (`encode_tensor_with_lut`, encode.rs) runs per 256-weight chunk: scale search,
sub-scale search, then one Viterbi — each block fully independent, at a **fixed** rate
(`k` bits/weight everywhere; there is no bit-allocation knob between blocks). Multiplying all
of a block's per-step costs by a constant changes neither the Viterbi argmin path nor any
scale/sub-scale argmin. Consequences:

- Per-tensor, per-layer, per-row (aligned), per-block components of `u` are **no-ops**.
- The lever's entire action surface is *relative* weights **within** one 256-block: where the
  Viterbi parks error among the block's 256 weights, and which scale/sub-scale/affine codes
  fit best under the reweighted error.
- Between-block sensitivity is mixed-precision's territory (will.md queue #7, HAWQ-style),
  explicitly **not** this proposal. If Phase 0 shows strong between-block but weak
  within-block structure, the correct disposition is "feed the map to the mp-allocator,
  close this proposal" — record that verdict, do not mission-creep it here.

### 1.5 The three deltas vs the dead family, and the residual overfit question

| axis | dead diag-H | this proposal |
|---|---|---|
| curvature object | diagonal activation 2nd moment (layer-local proxy), per-column, no row axis | empirical Fisher diag of the END loss, per-weight `[out,in]`, rotated pre-squaring |
| corpus | C4 (external) | WikiText-2 **train** — the PV loop's own loss stream |
| point | bf16 pretrained weights | the live quantized iterate (delta forward), refreshed by EMA every step |
| transport to encode basis | per-column values reused post-RHT | exact chain rule `g_y = R g_w` before squaring |

"Isn't WikiText-train-derived weighting just overfitting the benchmark like diag-H overfit
C4?" Two honest answers. (a) Train/test discipline holds: the accumulator only ever sees
train chunks (gradients exist only in the training path; `eval_ppl` is `no_grad`), and the
canon eval is the test split. (b) The PV arm *already* optimizes WikiText-train CE in every
one of its 300 steps — the requant is currently the only loss-blind operation inside an
otherwise loss-aware loop. Making the encoder's objective consistent with the surrounding
optimization is removing an objective mismatch, not adding contamination. What this does NOT
license: using this map for plain PTQ on a pretrained model (grad² from a few
forward/backwards at bf16). That configuration is the dead lever with a different corpus and
the §4 verdict predicts it fails — **scope = the PV loop only**.

If Phase 0 shows `u` is flat (or noise) within blocks, the correct write-up is: "the
whitening argument extends to the live empirical Fisher — `Cov(δ², x̃²) ≈ 0` at the PV
operating point — the Hessian family stays closed under the strongest framing we could give
it." That is a publishable negative and sharpens §4 from 'corpus overfit' to 'structural'.

---

## 2. Plumbing spec

Three pieces: (A) accumulation in `scripts/strand-qat.py`, (B) the map artifact, (C)
consumption in `quantize-model` → `encode.rs`. Decode is untouched at every layer:
`EncodedTensor`/`BlockMeta` fields, the `.strand` formats, `decode_tensor_fixed`,
`decode_lean`, and both GEMV kernels do not change — only WHICH integers the encoder emits
changes, and the decode remains a pure function of those integers. The map ships zero wire
bits (encoder-side only); per §5.11 there is nothing to bill — `total_bpw` is unchanged.

### 2.A strand-qat.py — y-space grad² EMA ("Adam's v, computed in the encode basis")

New flags:

```
--grad-sq                 enable the accumulator (default off — observer-only otherwise)
--grad-sq-beta 0.98       EMA decay (horizon ≈ 50 steps ≈ one requant period at 75)
--grad-sq-out PATH        export path; strand mode default: <strand-dir>/gradsq-y.safetensors
```

Accumulation point: in the train loop at `gi % grad_accum == 0`, **after** the last
micro-batch `backward()` and **before** `clip_grad_norm_` (the raw loss gradient is the
Fisher object; clip rescales globally per step and would re-weight steps inside the EMA).
Per `QuantLinear` module `m` with tensor key `name + ".weight"`:

1. `g = m.weight.grad` (fp32, on device, shape `[out, in]`). In strand/PV mode the delta
   forward is `wq = base + w`, so `∂L/∂w ≡ ∂L/∂wq` — the gradient at the **deployed
   quantized point**, which is exactly the curvature the requant should respect.
2. `gy = row_rht(g, seed(name))` — the same per-tensor rotation the encoder applies (2.A.1).
   On-device, transient (≤ 17.4 MB for the largest 0.5B proj tensor, 4864×896 fp32).
3. EMA update, CPU-side: each `QuantLinear` gains a **non-trainable registered buffer**
   `gradsq_y` (`persistent=True`; re-pinned to CPU after `model.to(dev)` — reassigning the
   attribute keeps the registration, and only the transient `gy` ever touches MPS), with
   **bf16 storage / fp32 update staging** (`g2 = gy.square_().to('cpu', torch.float32);
   buf32 = buf.float().mul_(beta).add_(g2, alpha=1-beta); buf.copy_(buf32)`): grad² spans
   ~1e-12..1e2; fp16 underflows at ~6e-8 —
   bf16 keeps fp32's exponent range, and mantissa coarseness is irrelevant for a relative
   weighting. A registered buffer — NOT a python dict — so the **existing**
   `--save`/`--init-state` state_dict path persists and restores the EMA together with the
   shadows it was accumulated at: checkpoint and map cannot desynchronize by construction
   (R3), and a `--steps 0` finisher can re-export the map from the restored buffers. One
   module-level int buffer `gradsq_steps` (optimizer steps accumulated) rides along — it
   gates consumption (2.B) and lands in the export metadata. No bias correction: it is a
   global per-tensor scalar and the weighting is argmin-invariant to global scale (1.4).

Estimator honesty: this is the squared accumulated (over `grad_accum=4` micro-batches,
batch-mean) gradient — Adam-v, not the per-sample Fisher diag (cross-sample terms included).
It is the standard cheap sensitivity proxy; Phase 0b's split-half test is what verifies the
signal content empirically, so the estimator-vs-Fisher gap is measured, not assumed.

Memory + time budget on the 18 GB box (the freeze trap, will.md §7, is the binding
constraint — budget BEFORE running):

| item | where | size | note |
|---|---|---|---|
| EMA store (357.9M trainable proj weights × bf16) | CPU RSS | **0.72 GB** | persistent |
| fp32 update staging, per tensor | CPU RSS | ≤ 17.4 MB | transient, sequential per tensor |
| `gy` rotation buffer, per tensor | MPS | ≤ 17.4 MB | transient, inside the step envelope |
| MPS driver envelope | MPS | **13.3 GB unchanged** | nothing persistent added on-device |
| export write (bf16 safetensors) | disk | 0.72 GB/requant | overwrite in place, scratch/ |
| `--save` checkpoint growth | disk | +0.72 GB/ckpt | the EMA buffers ride in state_dict (that is the R3 design); disk killed the ternary-300 save once — check free space before arming |

Added wall-clock per optimizer step (168 tensors, ~358M elements: 7–8 FWHT stages + one
device→CPU copy): estimate +0.2–0.5 s on the ~4–5 s/step PV pace, i.e. +5–10%. **Measure at
Phase 0 and print it**; if it exceeds +15%, accumulate every 2nd step (EMA horizon doubles —
still inside one requant period).

Watch `drv=` per step as always; abort the arm if the driver pool exceeds the proven
13.3 GB envelope by > 0.2 GB. `--grad-sq` runs ALONE like every QAT run (no concurrent
quantize-model), launched with the standard watermark env caps.

#### 2.A.1 Replicating the encoder's RHT in torch (exact spec — this is the coupling)

Shadows live in weight space; encoding happens post-RHT inside quantize-model. grad² cannot
be rotated after squaring (the rotation needs signs), so **python owns the rotation** and the
map ships already in y-space. The python implementation must reproduce, for tensor key
`name` (the full safetensors key, including `.weight`):

- **Seed:** FNV-1a-64 over the key bytes, then `| 1` (mirrors `rht_seed_for`,
  quantize-model.rs): offset `0xcbf29ce484222325`, prime `0x100000001b3`, wrapping u64.
- **Signs by GLOBAL FLAT index `i`** (row-major over `[out, in]`, NOT per-row index —
  mirrors `rht_forward_rows_inplace`, rht.rs): `s = seed ^ (i * 0x9E3779B97F4A7C15)`; one
  SplitMix64 step (`s += 0x9E3779B97F4A7C15; z = s; z = (z ^ z>>30) * 0xBF58476D1CE4E5B9;
  z = (z ^ z>>27) * 0x94D049BB133111EB; z ^= z>>31`); sign = `+1` if `z >> 63 == 0` else
  `−1`. All wrapping u64 arithmetic (numpy uint64 with overflow semantics, or torch int64
  with explicit masking).
- **Geometry:** `h = min(256, largest power of two dividing in_features)` (`pow2_block_for`
  with cap `HADAMARD_BLOCK = 256`); FWHT length-`h` segments restarting at every row
  boundary (h divides in_features by construction → no per-row tail); normalized `1/√h`.
  4864 → h=256; 896 → h=128; 7B dims → 256.
- **Order:** sign flip first (by flat index), then per-segment FWHT — matching
  `rht_forward_rows_inplace` exactly. fp32 throughout; small float divergence from the Rust
  f32 path only blurs the weighting (it is never on a decode path), but the SIGN/SEED logic
  must be bit-exact or the map is in the wrong basis (the old `diag_h_pre_rht` failure mode).
- **Required test before first use:** a golden-vector check — quantize a 1-block synthetic
  tensor with a known name through the Rust `rht_forward_rows` (a tiny `#[test]` in
  tests.rs dumps the expected post-RHT values once) and assert the torch port matches to
  ≤ 1e-5 rel. Wrong-basis maps fail SILENTLY otherwise (they just hurt PPL — exactly the
  class of bug §5.4 exists for).

The FWHT in torch is `reshape(..., h)` + log2(h) butterfly passes — vectorized, on-device,
no python loop over elements.

### 2.B The map artifact

`gradsq-y.safetensors`: one BF16 tensor per wrapped projection, key identical to the shadow
dump key (`<module>.weight`), shape `[out, in]`, values = the raw y-space EMA (untempered —
tempering is a consumer knob, §2.C). Safetensors `__metadata__` (all strings):

```
space     = "post-rht"            REQUIRED — the consumer refuses anything else under --rht
kind      = "gradsq-ema"
beta      = "0.98"
steps     = "<optimizer steps accumulated>"
clip      = "pre"
run       = "<qat invocation tag + seed>"
```

In strand/PV mode, `strand_requant` writes the export fresh from the live EMA buffers at
every requant instant (the init requant included — it is just another requant instant),
right next to `shadow.safetensors`, and appends `--weight-map <abs path>` to the
`quantize-model` command it builds — **only when `--grad-sq` is on AND `gradsq_steps > 0`**.
A cold-start init requant (step 0, empty EMA) is therefore always unweighted in every arm:
an all-zero map is an un-accumulated EMA, not a flat prior, and would floor-divide by
`mean(u) = 0` (the consumer also panics on it, 2.C). A finisher that restored
`gradsq_steps = 50 > 0` via `--init-state` passes the gate and its init requant IS weighted
— which is exactly Phase 1. The map is created and consumed inside one process invocation,
by absolute path — it is never discovered, never inherited from a previous run.

### 2.C quantize-model + encode.rs — the consumer

**quantize-model.rs** (new `Args` fields + plumbing):

- `--weight-map <path>` — **explicit only. No default. No directory probing. Never inferred
  from the model dir.** (The `hessian.hsdi` trap: a calibration file dropped in the model
  dir silently turned every subsequent "plain" run calibrated. This flag is the entire
  anti-trap design: absence of the flag MUST mean unweighted, regardless of what files
  exist anywhere.)
- `--weight-map-temper <τ>` default `0.5`; `--weight-map-floor <eps>` default `0.01`.
- Loader (in `main`, next to the input open): parse with the existing `SafeTensors::open` +
  `to_f32`; verify `__metadata__["space"] == "post-rht"` when `args.rht` (panic with the
  exact mismatch otherwise); **every** quantizable tensor selected for this run must be
  present in the map — panic listing the missing names (a partial map silently mixes
  weighted and unweighted tensors inside one "weighted" run = uninterpretable A/B).
- Per tensor, at load: validate every value finite and `mean(u) > 0` — panic otherwise
  (an all-zero tensor is an un-accumulated EMA reaching the consumer, a launch bug, see
  2.B). Then `m = mean(u)`; `u_i ← (max(u_i, eps·m) / m)^τ`. The floor keeps
  zero-gradient coordinates (dead units under THIS corpus) from getting zero cost — without
  it the Viterbi parks unbounded error there and any distribution shift detonates (the
  diag-H lesson in miniature). The normalization makes τ corpus-scale-free. τ is the
  aggression knob: τ=0 is the literal identity (see the inertness gate below), τ=1 raw EMA.
- `TensorJob` gains `loss_map: Option<Vec<f32>>` (filled in `main` at job build, from the
  loaded+tempered map by tensor name); `quantize_one` threads `job.loss_map.as_deref()`
  into `EncodeOpts.loss_map` (below). **Indexing contract:** the map is y-space, so
  inside `quantize_one` it aligns with `work` (the post-RHT buffer), NOT with `job.gt` —
  flat index for flat index, no transformation in Rust. The outlier channel zeroes bulk
  positions pre-RHT in *value* space; the sensitivity basis is unchanged by that, so the
  map applies to the bulk encode as-is (outliers' own sensitivity is spent on their exact
  8-bit side channel; the smear of their grad² into block coordinates is second-order —
  noted, accepted).
- Sidecar JSON `"config"` gains `"weight_map": null` or
  `{"path": ..., "sha256_8": ..., "temper": τ, "floor": eps, "steps": ...}`. The existing
  pinned `"calibrated": false, "block_hessian": false` keys stay pinned false — they name
  the dead levers, not this one. Every downstream ppl json must remain attributable to its
  map (or to no-map) from the sidecar alone.
- `--weight-map` with `--vec-dim > 1`: **panic**, not ignore. (v1 is scalar-only — the PV
  deployment config `--bits 2 --l 12 --outlier-channel 1` is scalar. Silent ignore is how
  15-digit-identical PPLs happen.)

**encode.rs** (the metric change — restores the shape the dead lever used, minus its sins):

- `EncodeOpts` regains a lifetime: `pub struct EncodeOpts<'a> { adaptive, tail_biting,
  affine_min, /** per-coordinate ≥0 cost weights in POST-RHT flat order, len == weights.len();
  None ⇒ unweighted (byte-identical to today) */ pub loss_map: Option<&'a [f32]> }`.
  Still `Clone + Copy` (a shared slice ref is Copy). Default `None`.
- `encode_tensor_with` (the dispatcher, line ~346): `gpu_eligible &=
  opts.loss_map.is_none()` — Metal/cloud-GPU kernels bind the unweighted SSE and stay that way;
  weighted encode is CPU-only in v1. No perf cliff in the target path: PV requants already
  run `STRAND_NO_GPU=1` (the 7B-wide-tensor SIGKILL trap) and 12-way CPU beats the
  serialized GPU encode ~8× anyway.
- `encode_tensor_with_lut` (the chunk loop, line ~403): track the running block offset `g0`
  (the chunk loop already walks `weights.chunks(cfg.block_len)`); derive
  `let u_blk = opts.loss_map.map(|u| &u[g0 .. g0 + chunk.len()]);` and pass it down to every
  scoring site of the block:
  - `choose_scale_q` / `choose_sub_scales` / `choose_affine_min` → all bottom out in
    `greedy_replay_mse_off` (line ~971): the accumulation `acc += best_err` becomes
    weighted — per weight `i`, candidate error is `u[i] * (target − lvl)²` (sub-block calls
    pass the matching `u` sub-slice). Weighting the Viterbi but not the scale searches would
    leave scales fit to a different objective than the path — half-weighted runs are
    uninterpretable, so ALL squared-error sites take `u` or none do.
  - `viterbi_path_buf` (line ~1013) → `viterbi_forward` (line ~1089) and `backtrack_buf`
    (line ~1174). Both have exactly two cost sites each: the scalar branch
    `let nc = c + d * d;` becomes `let nc = c + u_i * d * d;`, and the SIMD branch
    `nc_v = c_v + d_v * d_v` becomes `nc_v = c_v + u_v * d_v * d_v` where
    `u_v = f64x4::splat(u[i] as f64)` — `u` is per-WEIGHT (per step), constant across the
    states of a step, so it is one splat per step, zero change to the lane structure or the
    back-pointer logic. `pick_terminal`, tail-biting, `build_sub_levels`: untouched.
  - Vector path (`*_vec` functions): `debug_assert!(opts.loss_map.is_none())` in v1 (the
    bin already panics upstream).
- **The inertness gate (mandatory test, the τ=0/constant-map identity):** a constant map —
  and equivalently τ=0 — normalizes to all-ones and scales every block's costs uniformly ⇒
  every argmin (scale, sub-scale, affine, path, terminal) is unchanged ⇒ the emitted
  `EncodedTensor` must be **byte-identical** to the no-map encode. Ship this as (a) a unit
  test in tests.rs and (b) a one-off CLI A/B (`--weight-map <const-map>` vs no flag on one
  real tensor, diff the artifacts). If byte-identity fails, the plumbing leaks the weighting
  somewhere it claims not to — fix before any measurement. This is also the live tripwire
  for accidental no-ops in the other direction: in arms where the map is NON-constant, a
  15-digit-identical PPL vs baseline means the map was silently dropped (§5.4 tell).

Determinism statement: the weighted encode is float (allowed: offline search, module
header doctrine), CPU, fixed iteration order, thread-per-tensor with no cross-tensor state —
same binary + same inputs (model, map, flags) ⇒ same artifact bytes. Decode determinism is
not merely preserved, it is untouchable from here: nothing downstream of `EncodedTensor`
changes.

---

## 3. A/B protocol on the 0.5B PV arm

Common fixings for every phase: model `scratch/qwen-05b`; `--seed 0`; canon eval =
`--eval-chunks 64 --eval-ctx 2048` (the canon 64-ch protocol, comparable to bf16 12.55 and
the 80.7 PTQ floor); deployment flags `--bits 2 --l 12 --outlier-channel 1 --threads 8` +
`STRAND_NO_GPU=1`; the pinned binary convention (`scratch/bin/quantize-model`, rebuilt once
per campaign — the 62 GB-cleanup lesson); fixed train-chunk schedule (`--train-chunks` /
`--chunk-offset` as in the live pv re-pass). Everything `nice -n 19`; **all runs serialize
behind the live marathon** (conductor owns the lanes; this whole campaign is queued work —
idle-filler or post-marathon, per the two-track separation directive). QAT runs alone.

**Phase 0 — flatness + noise gate (decides the fork for <30 min; NO encoder changes needed).**
Run the PV arm from the best existing pv checkpoint:
`--quant strand --init-state <pv ckpt> --grad-sq --steps 50 --eval-chunks 8 --skip-after
--save <p0.pt>` — observer only: no `--weight-map` anywhere (the `gradsq_steps` gate keeps
even the init requant unweighted on the cold counter), training math identical to baseline.
8-chunk eval because Phase 0's PPL is not a measurement, and `--skip-after` drops the final
requant+eval this phase does not need; the run-end `--save` banks shadows + `base` + the
EMA buffers in one state_dict (Phase 1's input). Export TWO EMAs accumulated on disjoint
chunk halves (steps 1–25 → `u_A`, 26–50 → `u_B`; a second buffer switched at the halfway
step, or two 25-step runs walked forward with `--chunk-offset`).
Analysis (numpy, scratch/):

- **0a flatness:** per tensor, per 256-block, `CV = std(u)/mean(u)` on the COMBINED EMA;
  primary statistic = median CV over **non-straddling, single-Hadamard-segment blocks**
  (1.3 caveat); straddle-inclusive median recorded separately. KILL if median CV < 0.10
  (within-block structure too weak to redirect a Viterbi whose state cost differences are
  O(1) relative) → ledger: "whitening extends to the live Fisher".
- **0b reproducibility:** per block, Pearson `r(u_A, u_B)`; median within-block r. KILL if
  median r < 0.2 — the structure is chi-square sampling noise and weighting by it injects
  noise into the objective. One bounded retry at N=200 steps (EMA β=0.995) if r lands in
  [0.2, 0.5); still <0.5 after the retry → DEAD (bounded re-litigation, §5.6 discipline).
- Bonus diagnostic (free): correlation of block-mean(u) against the per-row `E[δ²]` factor
  and against block-mean of the old rotated-activation proxy — tells us whether any PASS is
  genuinely the covariance term or a leak of the known-dead components.

Cost: init requant ~15 min (CPU; the dominant term) + BEFORE eval ~2 min (8 ch) + 50 steps
≈ 5 min + save/export + script ≈ **~25 min wall**. Also measure and record the per-step
accumulator overhead here (budget says +5–10%).

**Phase 1 — pure requant A/B at one checkpoint (isolates the encoder objective; zero
training noise).** Take ONE checkpoint file: the Phase-0 run's end-of-run `--save` artifact
(`p0.pt` — shadows, `base`, AND the `gradsq_y`/`gradsq_steps` buffers in the same
state_dict, so checkpoint and map are one file's contents). Three finisher invocations of
`strand-qat.py --quant strand --steps 0 --init-state p0.pt --eval-chunks 64` — the
`--steps 0` path does init-requant → canon BEFORE eval in a pristine process (the
segmented-arm machinery, proven). The W arms add `--grad-sq` (arming the
`gradsq_steps = 50 > 0` gate, 2.B), so their finisher exports `gradsq-y.safetensors` from
the restored buffers and auto-appends `--weight-map` to its own init requant:

| arm | invocation delta | expected |
|---|---|---|
| U (control) | none (no `--grad-sq` ⇒ unweighted requant) | the checkpoint's unweighted requant PPL — the control anchor (expect the pv floor class, ~79; Phase 0's `--skip-after` means this is its first measurement) |
| W½ (headline) | `--grad-sq` + `--weight-map-temper 0.5` in `--strand-flags` | the decision number |
| W1 | `--grad-sq` + `--weight-map-temper 1.0` in `--strand-flags` | dose-response read |

Identical shadow bytes into the encoder; the ONLY difference is the encoder objective.
Record per arm: PPL, weighted-SSE, rel-RMS, bpw (must be bit-equal across arms — the map
adds zero wire bits; a bpw difference is a bug). Cost per arm ≈ requant ~15 min (0.5B, CPU)
+ canon eval ~10–15 min (estimate; MPS) → **~2 h for three arms, nice'd**.

KILL: best of {W½, W1} < 2% PPL improvement vs U → DEAD, ledger entry, stop.
INSTANT KILL (the diag-H signature, §5.5): any weighted arm with rel-RMS DOWN and PPL UP vs
U → dead on the spot, no Phase 2, record the signature explicitly.

**Phase 2 — full PV arm (only on a Phase-1 pass).** Two 300-step arms, back-to-back same
night, identical seeds/schedule/binary, differing ONLY in `--grad-sq` + the auto-appended
`--weight-map` at the 4 requant boundaries (75/150/225/300; the init requant is unweighted
in BOTH arms — cold counter, 2.B). Free determinism tripwire: the accumulator is
observer-only and touches no RNG, so the two arms' loss prints must match digit-for-digit
through step 75, up to the first weighted requant — any earlier divergence means the
accumulator leaked into training math (a bug, not a result; stop and fix). Compare end PPL
and the per-boundary requant deltas against the control arm. Cost ≈ 2 × (300 steps × ~5 s +
4 requants × 15 min + segment evals) ≈ **2 × ~3 h = one night lane**.

KILL: end-of-arm gain < 2% → DEAD. Distinct honest outcome to watch for: Phase-1 gain real
but Phase-2 gain ≈ 0 — PV re-learning absorbs the encoder improvement within a segment (the
moonshot showed re-learning dwarfs encoder effects: 115× vs our ≥2% bar). Verdict label
"ABSORBED-BY-PV": the lever works but doesn't matter where it ships; record as dead-for-PV,
note possible value for requant-sparse settings only if some future framing needs it.

Every verdict (pass or kill, each phase) gets: a will.md §4 row (DEAD table or
ALIVE/CONFIRMED), a §10 update-log line with the decisive numbers first, the memory ledger
(`strand-7b-ppl-track.md`), and a status banner at the top of THIS file.

---

## 4. Risk register — ways this silently contaminates comparisons

| # | risk | mechanism | mitigation (designed in, not aspirational) |
|---|---|---|---|
| R1 | **Auto-pickup contamination** (the `hessian.hsdi` trap reborn) | a map file lying in a model/scratch dir gets inherited by later "plain" runs | `--weight-map` explicit-path-only; NO default, NO directory sniff, absence of flag ⇒ unweighted unconditionally; map written+consumed inside one PV invocation by absolute path; sidecar JSON records `weight_map` (path+sha256_8+τ) or `null` — every ppl json is attributable |
| R2 | **Wrong-basis map** (the `diag_h_pre_rht` failure mode) | weight-space or stale-seed map consumed as y-space → misallocated error, looks like "the lever is bad" | `space="post-rht"` metadata REQUIRED under `--rht` (panic on mismatch); python RHT port gated by the golden-vector test vs the Rust transform before first use |
| R3 | **Stale map / checkpoint mismatch** | requanting shadow X with a map accumulated at shadow Y (or another run) = silent garbage | the EMA lives in registered buffers in the SAME state_dict as the shadows (`--save`/`--init-state` round-trip them together — they cannot desynchronize); the export is regenerated fresh from those buffers at every requant instant, in the same process that dumps the shadow; metadata carries run tag + `gradsq_steps`; quantize-model logs map sha256_8 + steps at startup |
| R4 | **Silent no-op** (the inverse failure) | flag typo / dropped plumbing / all-flat map ⇒ "weighted" arm is secretly the control | the 15-digit-identical-PPL tell (§5.4) checked between arms expected to differ; per-tensor startup log `weight-map <name>: min/mean/max/CV after temper`; the constant-map byte-identity test proves the OFF state, the CV log proves the ON state |
| R5 | **Train→test leakage** | accumulating grad² on eval data would tune the encoder to the test split | structurally impossible as specced: accumulation hooks only the optimizer-step path; the test split flows only through `eval_parked` → `eval_ppl`, both `@no_grad`, so no `.grad` is ever produced on eval data |
| R6 | **0.5B straddle artifact** | h=128 vs block_len=256 misalignment manufactures within-block structure that does not exist on 256-aligned 7B tensors → a 0.5B win that cannot scale | Phase 0 primary statistic restricted to single-segment non-straddling blocks; any Phase-1/2 pass must note the straddle-restricted CV before a 7B confirm is even proposed |
| R7 | **Proxy-vs-truth inversion** (the family's signature) | weighted-SSE and rel-RMS improve while PPL worsens — error reshaped toward the map, away from the truth | rel-RMS + PPL recorded per arm; rel-RMS↓ + PPL↑ = instant kill at Phase 1 (no Phase-2 spend); rel-RMS is ranking-within-family only, never closes the channel (§5.5) |
| R8 | **Memory creep → machine freeze** (the box's worst failure) | +persistent MPS state on the 13.3 GB strand-mode envelope | nothing persistent on MPS (EMA is CPU bf16, 0.72 GB RSS; transients ≤ 17.4 MB); `drv=` watched per step, abort at envelope +0.2 GB; QAT alone; watermark env caps; budget table in §2.A measured at Phase 0 |
| R9 | **Half-weighted objective** | weighting the Viterbi but not scale/sub-scale/affine search (or silently skipping the vector path) → uninterpretable arms | ALL squared-error sites take `u` (one shared helper); `--vec-dim>1` + map = panic, never ignore |
| R10 | **Estimator noise dressed as signal** | 50-step EMA of squared batch-grads is high-variance; chi-square noise has nonzero CV by construction | Phase 0b split-half reproducibility is the gate — CV alone (0a) can NOT pass the idea; both gates must pass |
| R11 | **Zero-cost sinks** | u→0 coordinates (dead under this corpus) absorb unbounded error; any distribution shift detonates | floor `eps·mean(u)` before tempering (default 0.01), applied in the consumer so every artifact's floor is recorded in its sidecar |
| R12 | **Scope creep into rate allocation** | between-block u is inert at fixed rate (1.4); the temptation is to "fix" that by per-block rates inside this lever | explicit non-goal; if Phase 0 finds between-block-only structure, the map is handed to the mixed-precision queue item (#7) and this proposal closes |

---

## 5. Non-goals + standing constraints

- **Not a PTQ lever.** No `--weight-map` run against pretrained weights outside the PV loop
  (§1.5). The flag existing makes the misuse possible; the sidecar attribution (R1) makes it
  visible.
- **Not a decode/format change.** Zero new wire bits, zero decode-path edits, the
  determinism moat untouched by construction.
- **Not a GPU-encode feature** (v1 CPU-only; the target path already forces CPU).
- **Not a 7B campaign.** 7B confirm is only even proposable after a Phase-2 pass PLUS the R6
  straddle check, and goes through the normal cheap-first ladder.
- The 2% kill bar is deliberately above run-to-run noise on the canon 64-ch eval
  (deterministic same-harness arms differ only through the lever) — a sub-2% "win" at 2-bit
  PPL ~79 is ~1.6 PPL, not worth the plumbing's permanent complexity tax. Dead at <2%,
  recorded, sealed-under-this-framing.
