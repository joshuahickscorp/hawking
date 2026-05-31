# Bible Colab + errata-resolution — AUDIT (2026-05-30)

**Read this cold in a fresh chat.** Standalone audit of an autonomous session that
(a) resolved the locally-runnable errata from `plans/bible_execution_2026_05_30.md`
by actually running them, and (b) authored Colab notebooks for the Colab-dependent
Bible stages. The canonical strategy is `plans/throughput_bible_2026_05_30.md` (6
stages, 0–5). Stages 0–1 shipped earlier this session; this pass attacks the errata
for stages 2–5.

**Trust level legend:** ✅ ran + verified locally · 📓 authored + syntax-validated, **NOT
GPU-executed** (needs Colab to confirm) · 🔭 documented-only (needs M3 multi-session).

---

## TL;DR

| item | stage | status | result / gate |
|---|---|---|---|
| Oracle B — SVD lm_head | 0 | ✅ | **NO-GO** — head full-rank (99% energy @ rank 1987/2048) |
| Existing EAGLE head measure | 4 | ✅ | **0.000 accept, 4.5× slower** — head broken on Q4_K_M |
| `01_awq_bytecut.ipynb` | 3 | 📓 | gate: W4 PPL/f16 ≤1.05, **W3 ≤1.12** (the byte-cut prize) |
| `02_eagle3_train.ipynb` | 4 | 📓 | gate: **τ ≥ 2.5** on code + head↔runtime parity |
| `03_qtip_3bit.ipynb` | 3-deep | 📓 | gate: 3-bit PPL/f16 ≤1.10; **blocked on M3 trellis kernel** |
| Stage 2 MLX-class GEMV | 2 | 🔭 | M3 multi-session; start from `silicon-builds/dismantle-q4k-mma` |
| Stage 5 fused int4-KV | 5 | 🔭 | M3; from `silicon-builds/dismantle-int4kv` |
| Instruments calibration | 0 | 🔭 | GUI; command in execution doc E1 |

---

## Part 1 — Local errata RESOLVED (ran + verified) ✅

**Oracle B — SVD lm_head energy spectrum.** `tools/bench/oracle_svd_lmhead.py`
(run with a python3.12 gguf venv — system python3.14 can't import gguf). The LM head
is **tied** (`token_embd.weight`, Q6_K, 151936×2048). Energy: rank@90/95/99/99.9% =
1662/1829/1987/2037 of 2048 → **99% energy needs 97% of the rank**. **FULL-RANK ⇒ SVD
screening NO-GO** (no compressible structure). `reports/oracle/svd_lmhead.json`.
Confirms lm_head→predec is the only LM-head lever (and it's ~4% of decode — low value).

**Existing EAGLE head — actually measured.** Ran `tools/bench/eagle5_paired_bench.sh`
on Qwen-3B with `checkpoints/eagle5_final/q3b/head_final.safetensors` (1.83 GB). The
runtime **loads** it (`[eagle5] loading trained head…`) but:
```
no-spec greedy      dec_tps=34.35
eagle5 K=2          dec_tps=14.90  accept=0.000  rej=47
eagle5 K=4          dec_tps=11.15  accept=0.000  rej=92
eagle5 K=8          dec_tps= 7.58  accept=0.000  rej=176
```
**0.000 acceptance, 4.5× SLOWER.** Confirms the f16-trained / Q4_K_M-served distribution
shift. (The "(mock head)" string in the bench output is a stale hardcoded label, not the
truth — a real head was loaded.) **This is the reason notebook 02 exists.** `bench_results/oracle/eagle_head_measure.log`.

*(Recap, shipped earlier this session: Stage-1 path-to-50 fusions verified bit-identical,
2-row ILP flipped default-on → 24.08→31.89 dec_tps (+32.5%). §1 methodology gate enforced
in `analyze_tcb_trace.py`. Oracles A (spec τ=1.43 NO-GO) and C (imatrix≫naive byte-cut)
done. See `plans/bible_execution_2026_05_30.md`.)*

---

## Part 2 — Colab notebooks BUILT (authored + syntax-validated, NOT GPU-run) 📓

All three: percent-format `.py` (the editable source, `py_compile`-clean) +
generated `.ipynb` (via `colab/py_to_ipynb.py`, a dependency-free converter). Each has
a fail-fast GPU check, pinned deps, a deterministic seed, an explicit GO/NO-GO gate, and
an honest M3-integration-boundary cell. **I could not execute them** (no GPU / no f16
model / no internet for HF downloads here) — they self-verify when you run them.

**`colab/01_awq_bytecut.ipynb` — Stage 3 byte-cut (the practical one).**
AWQ-W4 (stable, ~Q4_K_M bits, higher quality) + GPTQ-W3 (the ~3-bit byte-cut prize).
Code-calibrated; measures code-PPL vs an f16 reference computed in-notebook. **GATE:**
W4 PPL/f16 ≤1.05, **W3 ≤1.12**. A W3 GO means usable quality at ~25% fewer bytes than
Q4_K_M → justifies the M3 low-bit kernel. Upload the repo's `artifacts/quant/calib_trim.txt`
+ `ppl_trim.txt` for apples-to-apples with local oracle C (4.485 / 5.915 / 11.432); else
it falls back to a public code dataset. Sections are independent (AWQ failing still yields
the GPTQ verdict). ~20–40 min on T4/L4.

**`colab/02_eagle3_train.ipynb` — Stage 4 spec head (orchestrator + fix).**
Wraps the repo's **existing** scripts (`colab/eagle5_train_pytorch.py`,
`eagle5_tau_eval_pytorch.py`, `mega_calibrate.py`, `tools/training/build_qwen3b_frozen.py`)
— it does NOT reinvent them. Two root-cause gates for the 0%-accept finding:
1. **Capture provenance** — asserts the training residuals are **Q4_K_M** captures (not
   f16); includes the exact M3 capture commands (`DISMANTLE_QWEN_EAGLE5_CAPTURE=1`).
2. **Head↔runtime parity** — directs you to `cargo test eagle5_forward_parity` on the M3
   (the fixture `crates/dismantle-core/tests/fixtures/eagle5_parity_q3b.json` exists)
   before trusting τ; a logit mismatch there *is* the 0%-accept bug.
**GATE:** τ ≥ 2.5 at depth K on code AND parity passes. Requires uploaded captures + frozen
npz (produced on M3).

**`colab/03_qtip_3bit.ipynb` — Stage 3 deep byte-cut (SCAFFOLD, lowest priority).**
Clones Cornell-RelaxML/QTIP, computes code Hessians, quantizes to 3-bit, PPL-gates.
**GATE:** 3-bit PPL/f16 ≤1.10. **Hard caveats baked in:** (a) needs a NEW M3 trellis-decode
Metal kernel (no prior art — multi-session); (b) only pays off after Stage 2 is
bandwidth-bound; (c) QTIP is Llama-oriented, Qwen2 may need a model-load patch. Do
notebook 01 first — it's the practical byte-cut.

---

## Part 3 — Checks and balances (how each artifact self-verifies)

- **Every Colab notebook** has an explicit numeric GO/NO-GO gate (table above) that
  writes a `*_results.json` — audit that file, not the prose.
- **§1 methodology gate** (`tools/bench/analyze_tcb_trace.py`) — any decode bench must
  pass it (exits 2 on a physics violation: busy-time BW ≤150 GB/s, token-count from
  `sample_*`, etc.). Use it on every M3 number these notebooks lead to.
- **Bit-identical parity** — exact levers (predec, fusions) gate on greedy bit-identical
  (`tools/bench/path_to_50_verify.sh parity`). EAGLE head gates on `eagle5_forward_parity`.
- **Provenance assertion** — notebook 02 refuses an f16 corpus (the documented failure mode).
- **Disjoint calib/eval** — every PPL/τ split keeps calibration and evaluation separate.

---

## Part 4 — Verified vs UNVERIFIED (be explicit)

**Verified locally:** oracle B (ran, full-rank), EAGLE head 0%-accept (ran), all notebook
`.py` sources `py_compile`-clean, `.ipynb` JSON well-formed (nbformat 4), the referenced
prototypes/scripts/tests all exist on disk.

**NOT verified (needs you):** the notebooks have **not** been executed on a GPU — AWQ/GPTQ
quantization quality, QTIP Hessian/quant success, and EAGLE τ are all **unconfirmed
numbers** until you run them. Qwen2-arch compatibility of AutoAWQ/gptqmodel/QTIP is
assumed-close, not tested. The QTIP loader cell may need manual completion (its eval API
varies by version). Treat every Colab PPL/τ as **indicative until its gate prints GO** and,
for byte-cut, until it survives the GGUF round-trip on the M3 (oracle-C scale).

---

## Part 5 — Assumptions + risks
- AWQ/GPTQ/QTIP all assume Qwen2.5 ≈ Llama arch for their tooling; a load patch may be needed.
- EAGLE fix hypothesis (Q4_K_M captures + parity) is **inferred** from the 0%-accept + a
  memory note; if τ is still low after retraining, the bug is in the runtime forward, not data.
- Colab PPL (HF) and local PPL (llama.cpp) are different scales — compare **ratios**, and
  re-measure winners on the GGUF scale before believing absolute gains.
- QTIP's value is the most speculative lever in the whole program (per the Bible itself).

## Part 6 — Next actions (ordered by value/risk)
1. **Run `01_awq_bytecut.ipynb`** → is sub-4-bit (W3) viable on code? Cheapest high-value answer.
2. **M3:** produce Q4_K_M captures (commands in nb 02), then **run `02_eagle3_train.ipynb`**;
   gate τ≥2.5 + run `eagle5_forward_parity`. This is the only live spec path (n-gram was NO-GO).
3. **M3 multi-session:** Stage 2 MLX-class Q4_K GEMV from `silicon-builds/dismantle-q4k-mma`
   (the high-confidence ~50-tps dense lever; bandwidth-bound is the prereq for QTIP + EAGLE payoff).
4. Only after 3: `03_qtip_3bit.ipynb` + the M3 trellis kernel.
5. One-time: Instruments calibration (execution doc E1).

## Part 7 — How to audit this (fresh chat)
- Open the three `*_results.json` gates after running — they are the verdicts.
- Re-run `tools/bench/oracle_svd_lmhead.py` and `eagle5_paired_bench.sh` to reproduce Part 1.
- Diff `colab/*.py` (readable) rather than the `.ipynb`. Regenerate ipynb with
  `python3 colab/py_to_ipynb.py colab/*.py`.
- Cross-check every claimed file exists; nothing here was committed (see below).

## Files added/changed this session (uncommitted)
- `tools/bench/oracle_svd_lmhead.py` (new, ran), `reports/oracle/svd_lmhead.json`
- `colab/py_to_ipynb.py` (new) + `colab/0{1,2,3}_*.py` + `.ipynb` (new)
- `bench_results/oracle/eagle_head_measure.log`
- this doc + updates to `plans/bible_execution_2026_05_30.md`
- (earlier this session: `analyze_tcb_trace.py` gate, `path_to_50_verify.sh`,
  `oracle_spec_accept.py`, the 2r default-flip in `kernels/mod.rs`, profile rehash)
- **gguf venv** at `/tmp/ggufenv` (python3.12) — throwaway; recreate with
  `python3.12 -m venv … && pip install 'numpy<2.2' gguf`.
