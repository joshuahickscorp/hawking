# Condense / TQ — Output-Space Quality Push: Session Handoff (2026-06-22)

Chain target. A fresh session picks this up. Read this top-to-bottom, then
**start at L0 (run the harness)**. Do not re-derive; act.

## The mission (black-hole / condensation pillar)

Download a frontier-scale parent (e.g. 32B), quantize it locally to **3-bit or
2-bit**, **near-lossless** ("same quality as llama's Q4_K"), **denser** than any
format that would push RAM, and run it on Hawking's custom inference. Build →
test → iterate relentlessly. When something doesn't work, find the **custom
workaround** before conceding. This is the product wedge: compete at *artifact
creation time*, not just inference.

## The single most important reframe (internalize this)

1. **Density is the speed play — via the RAM cliff, NOT decode bandwidth.**
   Going from "doesn't fit / swaps to SSD" to "fits in unified memory" is a
   10–100× tok/s cliff. The trellis being slower-per-weight than Q4_K is
   irrelevant when the alternative is Q4_K thrashing swap or not loading. For a
   32B on 18 GB: **3-bit (~13 GB) is the enabling tier**; 4-bit (~18 GB) is over
   the cliff; 2-bit (~9 GB) needs recovery.

2. **The "3-bit is worse than Q4_K" verdict is a WEIGHT-SPACE PROXY that never
   ran the real codec.** `dead_levers.md:422-427`: the bracket
   `bits_needed=[+1.37,+0.44]` is weight-RMSE, the decisive codec was skipped
   (`ALLOW_FRESH_QTIP_CODEC=False`), and the project's own doctrine
   (`condense_frontier_2026_06_22.md` rule 4) says **grade in OUTPUT space; weight
   MSE is a scout, not a gate.** The quality arm is in *proxy-limbo*, not killed.
   Resolving it honestly is the whole game.

3. **Do NOT chase decode speed by rebuilding the trellis kernel.** The serial
   trellis decode is **Type-1 dead** on Apple Silicon (`dead_levers.md:224-232`):
   compute-bound + serial state (`state[i]←state[i-1]`), 24% of peak vs ~55% GO
   bar. The ONLY live speed reframe is the **lane-independent sub-block layout**
   (`dead_levers.md:229`) — hard, optional, do last. Get tok/s from the RAM cliff
   + the RWKV/SSM moat + spec/prefix instead.

## Ground truth — what's already BUILT (do not rebuild)

- **Full bit-exact trellis codec** (`vendor/strand-quant`): computed Acklam
  Gaussian codebook (gather-free), RHT (per-column = cheap serving), outlier
  section. GPU `strand_bitslice.metal` decode is **parity-verified** vs CPU
  oracle (`tq_trellis_parity.rs`).
- **TQ wired end-to-end into the model forward**: `qwen_dense.rs:3770`
  (`matvec_rht`) and `rwkv7.rs:1697` (`TqPreparedGpu`). Serving trigger:
  env `HAWKING_QWEN_TQ=1` + a `<model>.tq` (or `models/<stem>.tq`) sidecar.
- **Encoder quality levers that the kill-ledger never credited**
  (`vendor/strand-quant/src/bin/quantize-model.rs`):
  - `--actmean <calib.json>` → output de-bias `c = -(recon-orig)@mu`; the driver
    claims **−28.7% PPL for ~0.014 bpw**. (calib via `scripts/calib-actmean.py`.)
  - `--outlier-channel/--outlier-pct/--outlier-bits` → 1% outlier protection.
  - `--rung-config <json>` → per-tensor mixed-precision allocation.
  - `for_bpw_quality` (L=k+6 vs k+4) → more trellis states, same payload bpw.
  - `--rht-cols` → incoherence preconditioning (cheap-serving variant).
- **Real Q4_K ENCODER exists**: `hawking_core::quant::quantize_q4_k` (+
  `dequant_into(GgmlType::Q4_K, …)`). This kills the quant-of-quant problem — you
  can go f16→Q4_K exactly like llama.cpp for an apples-to-apples baseline.
- **A true bf16 source is local**: `models/rwkv7-g1-04-hf/model.safetensors`
  (read via `strand_quant::safetensor_io::SafeTensors`). Real linear tensors:
  `model.layers.N.ffn.key.weight` (4096×1024), `...ffn.value.weight` (1024×4096),
  `lm_head.weight`. f16 GGUF also present: `models/mamba2-370m-f16.gguf`.
- **Activation capture path exists** in `hawking generate` (`generate_main` args
  `batched_capture` / `capture_out`) — the route to REAL activations for L1.
- **QAT/KD scaffold exists**: `tools/strand/scripts/strand-qat.py`,
  `rung-kl.py` (output-space damage ranking), `strand-debias-ppl.py`.

## L0 — RUN THIS FIRST (the decisive output-space measurement)

A harness was written this session but is **UNCOMPILED / UNRUN** (the session was
interrupted before first build):

  `crates/hawking-core/tests/tq_output_space_quality.rs`

It loads the real bf16 RWKV-7 tensors, reconstructs `Ŵ` for **Q4_K** and
**TQ{2,3,4} × {RHT, quality-L, AWQ-importance-scale}**, and reports **weight-RMSE
AND output error** `||(Ŵ-W)·X||/||W·X||` under Gaussian *and* heavy-tailed
(1%-super-outlier-channel) activations, with a calib/eval split so the AWQ lever
is non-circular. GO = a TQ row at **fewer bpw than Q4_K with output-err ≤ Q4_K's**.

Run:
```bash
cargo test -p hawking-core --release --features tq \
  --test tq_output_space_quality -- --nocapture report
```
**First job: make it compile + run, fix any nits.** Candidate gotchas to check:
- `hawking_core::quant::{Q_K, Q4_K_BLOCK_BYTES}` re-export path (they're
  `pub const` in `quant.rs`; adjust if not visible at that path).
- `SafeTensors::open / .tensors / .to_f32 / StTensor.shape` API shape.
- `strand_quant::QUANTILE_SHIFT`, `TrellisConfig::{for_bpw,for_bpw_quality}`,
  `rht::{rht_forward_cols,rht_inverse_cols,RhtConfig::from_seed}`,
  `gate_utils::rht_seed_for`, `encode::{encode_tensor_with,EncodeOpts}`,
  `decode::decode_tensor_fixed`.

**Interpret honestly.** With the CURRENT (weight-MSE) encoder, output-err under
Gaussian acts ≈ weight-RMSE — so the kill-ledger's weight verdict is ~right *for
this encoder*. The expected findings:
- RHT + quality-L cut the **absolute** error (helps both spaces) — measure how far
  TQ3+L+rht closes to Q4_K.
- The AWQ-scale row is the **headroom probe**: if it beats TQ3+L+rht under
  heavy-tail acts, activation-aware (output-space) encoding is the real lever →
  go build L1 with REAL activations.

## L0 + L1-probe RESULTS (2026-06-22 — output-space, REAL bf16, REAL acts)

Harness RAN (after the cwd fix → `CARGO_MANIFEST_DIR`). Output rel-err
`o = ||(Ŵ-W)X||/||WX||`, means over the real RWKV-7 bf16 tensors:

- **Q4_K** (4.50 bpw): o ≈ **0.079** (the bar).
- **TQ3+L+rht** naive (3.35 bpw): o ≈ **0.155** ≈ 1.96× Q4_K → weight-space verdict holds *for the naive encoder*.
- **TQ3+L+rht+AWQ** (3.35 bpw, regularized geomean-norm + clip, α=0.5 sweet spot across a 0.25–1.0 sweep):
  - synthetic heavy-tail (1%×20 / 0.5%×30): o ≈ **0.08 ≈ Q4_K** (optimistic).
  - **REAL measured acts** (`reports/w4a8_activation_dist.csv`, `real-w4a8` sweep row): o ≈ **0.102–0.114 ≈ 1.37× Q4_K** at 0.74× bits.
  - gaussian/benign: o unchanged (**harmless** — the reg).

**HONEST VERDICT:** the kill-ledger weight-space verdict is **REOPENED** — AWQ closes ~HALF
the output-err gap (1.96×→1.37× Q4_K) at 26% fewer bits, and the outlier structure powering
it is REAL (measured Qwen-3B: max 28.9× median, ~0.2% of channels ≥20×, top-1% ≈ 10% of
rms-mass — not synthetic). But **AWQ-scale alone is NOT near-lossless on real data.** A fake
GO would claim "TQ3 = Q4_K" from the synthetic row; the real row says ~1.37×. Evidence:
`reports/condense/{L0_output_space,L1_real_acts_awq_sweep}_20260622.txt`.

**+ OUTLIER PROTECTION tested (top-1% σ columns at f16):** key 0.114→**0.108**, value 0.102→**0.095**
at eff_bpw≈3.48 — only marginal. The residual output-err is **spread across many channels**
(the measured tail: 1.3% ≥5×, not just the top 1%), so 1%-outlier-protection can't close it.
Net PTQ result: AWQ+outlier removes **~70% of naive TQ3's excess** vs Q4_K → **~1.28× Q4_K
output-err @ ~3.48 bpw (26% denser)**. Real but NOT near-lossless.

**REVISED NEXT-LEVER GUIDANCE (honest, after testing the cheap PTQ levers):**
- `encode_tensor_with_lut_metric` exposes only an f32/int *precision* flag — NOT a per-column
  weight. A true Hessian-diagonal metric = real trellis-DP surgery; AND **AWQ-scale already
  approximates the diagonal objective** (it's the standard AWQ trick), so the marginal gain of
  the weighted DP over AWQ is expected to be small. Deprioritize vs:
- **L2 RECOVERY is the path to near-lossless 3-bit.** PTQ alone (AWQ+outlier, no retrain) caps
  at ~1.28× Q4_K. To reach ≤1.0× at 3.35 bpw, do QAT/KD (`strand-qat.py`): re-fit the
  low-bit weights to the f16 teacher logits + damage-ranked mixed-precision (`rung-kl.py`,
  attn/lm_head high-bit). This is the heavy lever the goal demands.
- **OR validate "good enough" first:** ~1.28× per-tensor output-err may be an acceptable PPL
  bump for the RAM-cliff win (3-bit fits a 32B on 18 GB where Q4_K swaps). Add a full-model
  logits/PPL gate (handoff §L1 note) and measure real PPL of TQ3+AWQ+outlier vs Q4_K — if the
  PPL delta is small, the density wedge already wins WITHOUT recovery. **Cheapest decisive next step.**

## L1 — the custom lever: activation-aware (output-space) encode

If L0 shows a residual gap (likely), this is the "extra effort / custom way
around" the goal demands. Order:
1. **Capture REAL activations** per linear module (Hawking `generate` capture
   path, or a small teacher-forced forward) → per-channel σ_j and, ideally, the
   diagonal of the activation Hessian H=E[xxᵀ].
2. **Wire `--actmean`** (the −28.7%-PPL debias) into a real bake + measure.
3. **AWQ-importance scaling** around the existing encoder (diag-Hessian 80/20):
   scale columns by σ_jᵅ pre-encode, unscale post-decode. Non-circular: estimate
   σ on calib, evaluate on held-out.
4. If still short, **Hessian-aware sequential rounding (LDLQ/GPTQ-style)**: fold a
   per-column importance weight into the trellis path metric
   (`encode_tensor_with_lut_metric` already exposes a pluggable metric — extend it
   to minimize `Σ h_jj (w_j-ŵ_j)²` instead of `Σ (w_j-ŵ_j)²`). This is the piece
   that makes output-err < weight-err, i.e. beats the weight-space verdict.

Grade everything in OUTPUT space (logit-KL / PPL or the per-tensor output-err
above). Note: `hawking generate` cannot dump logits today — for full-model PPL
either add a logits writer or use the per-tensor output-err harness as the gate.

## L2 — push the bpw floor (3→2-bit) with recovery

2-bit PTQ collapses without recovery. Combine: QAT/KD in the loop
(`strand-qat.py`) + outlier protection + **damage-ranked allocation**
(`rung-kl.py` to rank, keep attn/lm_head high-bit, push tolerant FFN low). Find
the lowest bpw that holds output quality.

## L3 — speed + cross-platform compare

Measure TQ decode tok/s on the wired serving path. Compare **bytes + tps +
quality** vs `llama.cpp` Q4_K and MLX-4bit at iso-quality (engines under
`tools/bench/engines`, `compare_sota.sh`). Frame speed via the RAM cliff. Only
then consider the lane-independent decode reframe.

## Guardrails (hard)

- **Output space or it doesn't count.** Never claim quality from weight-RMSE alone.
- **f16/f32 sources only** for quality artifacts (the baker warns: Q4_K source =
  quant-of-quant, pipeline-validation only).
- **Respect Type-1 kills:** no serial-trellis-decode speed rebuild; no uniform
  sub-4-bit hoping to beat Q4_K weight-RMSE (oracle says it loses 7/7 families).
- **Never attribute Claude in git** (no Co-Authored-By / Generated-with trailers).
- **Owner-gate:** ask before downloading frontier weights, paid cloud, or
  publishing derivative claims.
- `strand-quant` / `strand-decode-kernel` are **excluded from the workspace**;
  build standalone (`--manifest-path vendor/strand-quant/Cargo.toml`). The `tq`
  feature is what pulls the codec into `hawking-core`.

## Task ladder (mirror into the tracker)

- **L0** Run the output-space harness; get real Q4_K-vs-TQ numbers. ✅ DONE — TQ3+AWQ ≈ 1.37× Q4_K @0.74× bits on REAL acts.
- **L1** Activation-aware encode (actmean → AWQ-scale → LDLQ metric) on REAL acts. 🔄 AWQ-scale done (real-data validated,
  ~half the gap); NEXT = Hessian-diagonal metric in `encode_tensor_with_lut_metric` + `--actmean`.
- **L2** 2-bit with QAT/KD + outliers + damage-ranked allocation.
- **L3** Speed + bytes + quality vs llama.cpp / MLX.

## Key references

- `docs/plans/condense_frontier_2026_06_22.md` — master plan C1–C6 (planner MVP
  `hawking press --dry-run` already landed in `crates/hawking/src/main.rs`).
- `docs/dead_levers.md` — `:224-232` decode Type-1 + lane-independent reframe;
  `:419-427` quality proxy-limbo & uniform-mix kill.
- `crates/hawking-core/src/tq.rs`, `tq_gpu.rs` — CPU/GPU serving + parity oracle.
- `crates/hawking-core/shaders/strand_bitslice.metal` — the decode kernel.
- `tools/tq_bake/src/main.rs`, `vendor/strand-quant/src/bin/quantize-model.rs` —
  the bakers (the latter has all the levers).
