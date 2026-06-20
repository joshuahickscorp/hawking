# QTIP Colab readiness — the decisive f16 byte-cut quality gate

**Notebook:** `colab/03_qtip_3bit.ipynb` (hardened 2026-05-31, Task H)
**Companion oracle:** `tools/bench/oracle_qtip_quality.py` (committed f51933c)
**Companion report:** `reports/oracle_qtip_quality.md` (local proxy, committed)
**Design:** `plans/qtip_bytecut_design_2026_05_31.md` §5.1 (quality oracle)

> Scope: this documents what the hardened notebook now does and the **exact**
> pass/fail criterion it bakes in. It is a readiness note, not a result — the
> result is `qtip_3bit_results.json`, produced by running the notebook on Colab.
> All numbers below carry a (measured)/(proxy)/(estimate) tag.

---

## 1. Why this notebook exists (the one number that sends us to Colab)

The committed local proxy (`oracle_qtip_quality.py --local-proxy`) returned the
decisive direction-only read:

- median Q4_K_M weight-RMSE **0.0808** (proxy, NumPy Q4_K_M on a clean bootstrap
  of the real Qwen marginal) vs QTIP bracket **[0.1854 lower, 0.0973 upper]**
  (proxy) at ~98 B/256-blk (−32% bytes);
- to MATCH Q4_K_M weight-RMSE at the ~3-bit budget, QTIP must extract
  **~1.20 bits** (proxy/estimate) of combined RHT-whitening + trellis coding
  gain over a 3-bit scalar — the **upper edge** of the TCQ envelope, and the
  modeled +1-bit upper bound still fell **~20% short** (proxy).

Per CLAUDE.md, that proxy **cannot** record a NO-GO: it runs on resampled
Q4_K-dequant values (not f16), it brackets the trellis rather than running it,
and weight-RMSE is not the functional metric. Killing QTIP from it would be a
**Type-2 error**. The decisive gate needs three things absent on the M3: the
**real f16 Qwen2.5-3B**, the **real QTIP RHT+trellis codec fit from f16**, and a
**forward pass**. This notebook is that gate.

---

## 2. What the hardened notebook now does (cell by cell)

| cell | role |
|---|---|
| 0 (md) | states the exact pass/fail criterion up front |
| 1 (code) | config + GPU preflight + **baked-in decision constants** (`GATE_BITS_NEEDED=0.0`, `PROXY_BITS_NEEDED=1.20`, `GATE_PPL_RATIO=1.10`); sets `RUN_REAL_QTIP = vram>=24` |
| 2 (code) | pip-installs deps **and builds llama.cpp** (cloud-GPU) — the gold Q4_K_M quantizer (gguf-python has no K-quant quantizer) |
| 3 (code) | fetches the **f16 Qwen** (HF snapshot + a near-lossless q8_0/f16 GGUF), quantizes the gold **Q4_K_M from that near-lossless source** (from-f16-class, NOT requant-from-an-already-Q4 file), fetches the **code** corpora |
| 4 (code, 3a) | the **faithful bracket** codec — RHT (exact) + Lloyd-Max {K, K+1}-bit scalar at the same stored ~K bits; byte-identical to the committed oracle's `qtip_quantize` (verified 98 B/256-blk, bracket-ordered). Self-contained (scipy `erfinv` with a NumPy Winitzki fallback). |
| 5 (code, 3b) | the **REAL upstream QTIP codec** (Cornell-RelaxML/QTIP): builds Hessians on the **code calib** corpus, quantizes Qwen with the **bitshift trellis K=3** (lane-parallel `td_x=td_y=16` sub-blocks), HF-izes the result. Guarded: if the QTIP cloud-GPU kernels fail to build, it falls back to the bracket and the verdict downgrades (it does **not** fabricate a trellis number). |
| 6 (code) | **LEG 1** — per-tensor weight-RMSE vs f16 for gold Q4_K_M, the bracket [lower, upper], and (if it ran) the real QTIP decode; computes the **measured** `bits_needed = log2(rmse_qtip / rmse_q4km)` |
| 7 (code) | **LEG 2 (decisive)** — forward-pass f16 / gold Q4_K_M / real QTIP on held-out **code** tokens; logit-cosine, KL(f16‖·), greedy argmax-agreement. Q4_K_M logits via `llama-cpp-python` (faithful GGUF forward), with a transformers-GGUF fallback. |
| 8 (code) | **LEG 3 (corroborating)** — code PPL: gold Q4_K_M via `llama-perplexity` (same tool as the local oracle, directly comparable), f16 + QTIP via transformers |
| 9 (code) | **VERDICT** — the three legs against the baked-in lines; writes `qtip_3bit_results.json` |
| 10 (md) | what GO / NO-GO / NEEDS-MEASUREMENT each mean + the M3 integration boundary |

**Cells are ordered, self-contained, and pip-install their deps** (cell 2 deps +
llama.cpp build; cell 5 the QTIP repo + kernels). A "Run all" on an L4/A100
executes the full decisive gate end-to-end.

---

## 3. The EXACT pass/fail criterion (baked into cell 9)

QTIP-3.0 (fit **from f16**) is judged against the **real Q4_K_M GGUF** (the
shipped incumbent it would replace), on a **code** corpus. **GO requires the
REAL codec to clear all three legs:**

1. **LEG 1 — weight-RMSE / the ~1.20-bit line.**
   `bits_needed = log2(median_rmse_qtip / median_rmse_q4km) <= 0`
   (i.e. QTIP-3.0 RMSE ≤ Q4_K_M RMSE) at fewer bytes (~96–104 vs 144 weight
   B/256-blk). This is the proxy's ~1.20-bit gap measured directly on f16: the
   proxy estimated **+1.20 bits**; GO needs the **measured** value to reach **0
   or below**. (The bracket also reports its own `bits_needed_bracket =
   [log2(lower/q4k), log2(upper/q4k)]` for continuity with the local proxy, but
   the bracket value is never used to pass or fail — only the real codec's is.)
2. **LEG 2 (decisive) — logit divergence vs Q4_K_M on code.**
   `cos(QTIP,f16) >= cos(Q4K,f16)` **AND** `argmax(QTIP,f16) >= argmax(Q4K,f16)`
   **AND** `KL(f16‖QTIP) <= KL(f16‖Q4K)`. QTIP must be **at least as good as the
   incumbent** on the W4A8 metric class, on the product workload (code).
3. **LEG 3 (corroborating) — code PPL.**
   `PPL(QTIP) <= 1.10 * PPL(f16)` **AND** `PPL(QTIP) <= PPL(Q4K)`.

**Verdict mapping (cell 9, dry-run verified):**

- **GO** — real codec ran AND leg1 ∧ leg2 ∧ leg3 all true.
- **NO-GO** — real codec ran AND any leg false. Records a **quality Type-1** kill
  (`results["kill_type"]`): a 3B at 3 bits via this codec is not good enough on
  code — a measured property, not effort. Routes to `reports/dead_levers.md`;
  closes the byte-cut axis with the §5.0/§6 speed read; never re-tested on vibes.
- **NEEDS-MEASUREMENT** — only the bracket ran (QTIP kernels failed to build, or
  VRAM<24GB). The bracket [lower, upper] is reported but **cannot** kill or pass
  QTIP (the Type-2 error CLAUDE.md forbids). `results["decisive_gate"]` names the
  fix: re-run with `RUN_REAL_QTIP=True` on an L4/A100.

This mapping was exercised in three dry-runs (proxy-only → NEEDS-MEASUREMENT;
real-codec-all-fail → NO-GO; real-codec-all-pass → GO) and behaves as specified.

---

## 4. Kill-respects honored (design §8, dead_levers.md)

- **From-f16, never requant-from-Q4_K** (`dead_levers.md:118`; the imatrix-Q3-
  from-Q4 +32% PPL kill). The QTIP codec input is the **f16 HF Qwen**; the Q4_K_M
  baseline is quantized from a **near-lossless q8_0/f16** GGUF source, not from an
  already-Q4 file. Both sides are from-f16-class — the fair comparison.
- **gather-free / no K-quant in gguf-python** (verified `NotImplementedError`):
  the Q4_K_M reference is the **shipped llama.cpp quantizer**, the gold standard,
  not a NumPy reimplementation (which the local proxy used and flagged as "a
  touch below llama.cpp's importance-weighted optimum").
- **"A wrong trellis sim is worse than none"** (bible §8.3.1): the bracket models
  RHT exactly and refuses a single trellis number; the **real** trellis is run by
  the upstream codec. A bracket-only run never produces a kill.
- **Quality gate is necessary, not sufficient** (design §0): a GO here only
  clears the **quality** gate. The **decode-cost** gate (M3 trellis kernel must be
  bandwidth-bound — design §5.2/§6.3, forecast by the §5.0 `q3k_bytecut_bench`
  by-proxy read) is still required before any kernel work. Cell 10 states this.

---

## 5. Runtime + what it produces

- **GPU REQUIRED.** ~30–60 min on L4/A100 (the QTIP Hessian + quant is the heavy
  step). On a <24GB GPU the notebook still runs LEGs 1–3 for Q4_K_M + the bracket
  and returns NEEDS-MEASUREMENT (the real codec is skipped).
- **Output:** `qtip_3bit_results.json` — all three legs, every number tagged, the
  baked-in gate lines, the verdict, and (on NEEDS-MEASUREMENT) the named gate.

**Decisive gate, one line:** run `colab/03_qtip_3bit.ipynb` on an L4/A100 with
`RUN_REAL_QTIP=True`; QTIP is **GO** iff the real RHT+trellis-from-f16 codec
hits `bits_needed <= 0` on weight-RMSE **and** matches-or-beats Q4_K_M on
logit-cosine/KL/argmax on code **and** PPL ≤ 1.10× f16 — otherwise a measured
**quality Type-1 NO-GO**.
