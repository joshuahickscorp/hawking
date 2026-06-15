# Colab handoff — section-close 2026-06-01

Two decisive quality gates that need a big GPU (the f16 Qwen2.5-3B forward pass
+ real codecs don't fit the M3 Pro 18 GB). **You run them; I resume on the
returned verdict JSONs.** Both are self-contained "Run all" notebooks: they build
llama.cpp, fetch the f16/q8 Qwen2.5-3B themselves, fetch the code corpora from
the repo, and write a machine-readable verdict JSON (with a `files.download()`).
Every leg is guarded — a leg that can't run faithfully reports `null` and the
verdict downgrades to `NEEDS-MEASUREMENT`; neither notebook fabricates a number.

**Where to drop the results for Phase 3:** put the downloaded JSONs at
```
reports/colab_verdicts/qtip_3bit_results.json
reports/colab_verdicts/imatrix_mixprec_results.json
reports/colab_verdicts/awq_results.json          # optional 3rd run (AWQ flavour)
```
(create `reports/colab_verdicts/`; it's fine if the optional AWQ one is absent).

---

## Gate 1 (P2-G) — QTIP 3-bit trellis vs Q4_K_M  →  `colab/03_qtip_3bit.ipynb`

**What it decides.** Does a QTIP-3.0 codec (RHT + bitshift trellis K=3, fit FROM
f16) beat the shipped Q4_K_M at ~3 bits, on code? This is the only live sub-Q4
byte-cut bet (`reports/moat_status_forward_path_2026_05_31.md`); the local proxy
estimated QTIP needs ~1.20 bits of RHT+trellis gain and fell ~20% short, but a
weight-RMSE proxy cannot kill an activation method (Type-2). This notebook is the
decisive f16 gate.

- **Run:** open in Colab → Runtime → **Run all**. Confirm cell 1 prints
  `RUN_REAL_QTIP=True` (it auto-sets from `vram>=24`; on a smaller GPU it stays
  False and you get NEEDS-MEASUREMENT). No manual input needed.
- **Runtime / GPU:** **A100-40GB preferred (~30–45 min)**; **L4-24GB OK (~45–60
  min)**. The QTIP Hessian + trellis quant is the heavy step. <24 GB → runs
  LEGs 1–3 for Q4_K_M + the RHT bracket only → NEEDS-MEASUREMENT.
- **Input it needs:** none you provide — it fetches f16 Qwen2.5-3B (HF) +
  quantizes the gold Q4_K_M from a near-lossless q8_0/f16 source (from-f16, never
  requant-from-Q4) + fetches the code corpora.
- **Produces:** `qtip_3bit_results.json`.
- **PASS (GO) — all three legs, real codec only:**
  1. **LEG 1** `bits_needed = log2(rmse_qtip / rmse_q4km) <= 0` (QTIP RMSE ≤ Q4_K_M at fewer bytes).
  2. **LEG 2 (decisive)** `cos(QTIP,f16) >= cos(Q4K,f16)` **AND** `argmax(QTIP,f16) >= argmax(Q4K,f16)` **AND** `KL(f16‖QTIP) <= KL(f16‖Q4K)`.
  3. **LEG 3** `PPL(QTIP) <= 1.10*PPL(f16)` **AND** `PPL(QTIP) <= PPL(Q4K)`.
  - **NO-GO** = real codec ran AND any leg false (records a quality Type-1 kill).
  - **NEEDS-MEASUREMENT** = only the bracket ran (kernels failed / VRAM<24).

## Gate 2 (P2-H) — imatrix mixed-precision vs Q4_K_M  →  `colab/04_imatrix_mixprec_gate.ipynb`

**What it decides.** Does an imatrix-guided mixed-precision GGUF (~3.82 eff bits;
keep attn+ffn_gate at Q4_K, demote ffn_down+ffn_up to Q3_K) beat uniform Q4_K_M
on next-token LOGITS on held-out code? `reports/oracle_imatrix_mixprec.md` left
this NEEDS-MEASUREMENT (weight-RMSE can't see the logit metric the real
activation imatrix optimizes). Stays in GGUF → a GO needs only loader byte
accounting, **no new kernel** (the clean win vs AWQ/QTIP).

- **Run:** open in Colab → Runtime → **Run all**. No manual input.
- **Runtime / GPU:** **L4-24GB fine (~20–35 min)**; A100 faster. CPU works but
  the forward/PPL legs are slow.
- **Input it needs:** none you provide — fetches near-lossless q8_0/f16 Qwen
  (the from-f16 source AND the logit reference) + the code corpora.
- **Produces:** `imatrix_mixprec_results.json`.
- **PASS (GO):** the decisive **logit leg ran** AND
  `cos(mix,ref) >= cos(q4k,ref)` **AND** `argmax(mix,ref) >= argmax(q4k,ref)`
  **AND** `KL(ref‖mix) <= KL(ref‖q4k)` AND mixed bytes ≤ uniform Q4_K_M.
  - **NO-GO** = logit leg ran AND failed (records a quality Type-1 kill; the
    imatrix mixed-prec axis closes).
  - **NEEDS-MEASUREMENT** = the logit leg didn't run (llama-cpp-python build /
    VRAM) — recon-RMSE + PPL alone cannot pass or kill (Type-2 forbidden).
  - Note: if the installed `llama-quantize` lacks `--tensor-type`, the notebook
    falls back to uniform Q3_K_M+imatrix as the "mixed" stand-in and logs it in
    `results["notes"]` — read that field before trusting a NO-GO.

### Optional 3rd run — AWQ-smoothing W4A8 (the wired downstream)  →  `colab/awq_w4a8_validate.py`
P2-H also names AWQ-3b. The AWQ flavour is the already-wired downstream
(`DISMANTLE_QWEN_AWQ=1` + `DISMANTLE_QWEN_W4A8=1`, `PREDEC=0`). This script asks:
does AWQ per-channel smoothing improve W4A8 fake-quant quality vs no smoothing?
```
python colab/awq_w4a8_validate.py --out /content/awq_results.json          # A100 fp16
python colab/awq_w4a8_validate.py --out /content/awq_results.json --load-4bit   # L4/T4
```
Input: `profiles/qwen3b_awq_smoothing.json` (in the repo). ~15–30 min.
**PASS (AWQ helps):** `with_awq.greedy_match_first_32 > without_awq.greedy_match_first_32`
**AND** `with_awq.mean_kl_per_token < without_awq.mean_kl_per_token`. A pass means
re-run the W4A8 ship gate (held at paired 1.115× < the 1.20× bar); only a GO that
*also* clears 1.20× bit-identity flips `DISMANTLE_QWEN_AWQ` on by default.

---

## Return-data schema (what Phase 3 parses)

**`qtip_3bit_results.json`** (keys present on a full run):
```
verdict: "GO" | "NO-GO" | "NEEDS-MEASUREMENT"
real_codec_ran: bool
bits_needed: float            # LEG1, real codec; GO needs <= 0
leg2: { cos_qtip, cos_q4k, argmax_qtip, argmax_q4k, kl_f16_qtip, kl_f16_q4k }
leg3: { ppl_f16, ppl_q4k, ppl_qtip }
kill_type?: str               # present iff NO-GO
decisive_gate?: str           # present iff NEEDS-MEASUREMENT (names the fix)
```

**`imatrix_mixprec_results.json`:**
```
verdict: "GO" | "NO-GO" | "NEEDS-MEASUREMENT"
gib: { q4km, mixed }
mixed_under_budget: bool
ppl: { ref, q4km, mixed }
logit: { cos_mix_ref, cos_q4k_ref, kl_ref_mix, kl_ref_q4k, argmax_mix_ref, argmax_q4k_ref, tokens }
legs: { recon_floor, ppl, logit_ran, logit }
kill_type?: str               # present iff NO-GO
notes: [str]                  # read this (e.g. mixed fell back to uniform Q3_K_M)
```

**`awq_results.json`** (optional): per-condition `{ mean_kl_per_token, p95_kl_per_token,
greedy_match_first_8/16/32 }` for `with_awq` and `without_awq`.

---

## What Phase 3 does with each (so you know what "resume" means)

- **QTIP GO** → STAGE the QTIP Metal trellis-kernel build as the next section's
  opened lever (NOT built in this run — new construction beyond close-out).
  **NO-GO** → record Type-1 in `dead_levers.md`; the sub-Q4 byte-cut axis has no
  live bet. **NEEDS-MEASUREMENT** → re-run on ≥24 GB.
- **imatrix mixed-prec GO** → STAGE the loader byte-accounting wire-in + paired
  decode bench (no kernel). **NO-GO** → record the kill; the axis closes.
- **AWQ GO** → run the W4A8 1.20× bit-identity re-gate; flip `DISMANTLE_QWEN_AWQ`
  default-on only if it ALSO clears 1.20×; else STAGE. **NO-GO** → record kill.

Either way: verdicts are recorded as facts; only a GO that also clears its
parity/bit-id gate may flip a default.
