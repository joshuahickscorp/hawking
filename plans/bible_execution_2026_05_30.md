# Bible execution + errata — autonomous session 2026-05-30

Executed the locally-completable, low-risk portion of the Throughput Bible
(`throughput_bible_2026_05_30.md`): **Stage 0 in full** (methodology gate + the
offline oracles) and **Stage 1 verification** (the uncommitted path-to-50 work).
Everything requiring Colab, Instruments, or multi-session kernel engineering is
captured as **errata** below with exact commands. Numbers under the §1 gate;
paired deltas valid with Claude open (`memory/feedback_bench_with_claude_open`).

---

## What ran this session (local, M3 Pro, machine idle — no slm coexist)

| Bible stage | item | status | result |
|---|---|---|---|
| 0 methodology | §1 invariant gate in `analyze_tcb_trace.py` | ✅ DONE | 4 asserts wired + self-tested; default footprint → Qwen-3B |
| 0 oracle A | spec acceptance (n-gram/PLD on code) | ✅ DONE | **τ=1.43 warm → NO-GO** |
| 0 oracle B | SVD lm_head recall | ⚠️ ERRATA | needs Q6_K dequant; lowest-value oracle |
| 0 oracle C | mixed-precision PPL byte-cut | ✅ DONE | **imatrix ≫ naive; Q3 +32% PPL; Q2 garbage** |
| 1 exact wins | path-to-50 fusions (predec pair / ffn_down / 2r ILP) | ✅ VERIFIED | **parity bit-identical; bench below** |
| 1 exact wins | LM-head → predec | ⚠️ ERRATA | not present; ~+1–2% (LM head ~4% of decode) |
| 2–5 | MLX kernels / byte-cut / spec / long-ctx | ⚠️ ERRATA | Colab + multi-session; see below |

---

## Oracle verdicts (these rank the whole expensive program)

### A — speculation acceptance (Bible's single most informative number)
`tools/bench/oracle_spec_accept.py` simulates lossless PLD/n-gram on a real
40k-token **code** stream (disjoint repo source). τ = tokens emitted per verify
forward = the speedup ceiling of a ~free CPU-automaton draft.

```
n=2 K=16: τ_warm=1.43  hit_rate=0.16   ← best
n=3 K=16: τ_warm=1.29  hit_rate=0.09
```
**Verdict: NO-GO** (threshold τ≥2.5 GO, ≥1.6 MARGINAL). On this code corpus,
n-gram/SAM speculation is *not* the hoped-for 1.5–2.5×. **Consequence:** the
"safe free spec lever" is weak here — real axis-3 gains require the **trained
EAGLE head**, not n-gram. Re-run on the product's *real* transcripts before
discarding (a more boilerplate-heavy/long-context workload may clear 2.5):
`llama-tokenize -m <model> -f transcripts.txt > t.txt && python3 tools/bench/oracle_spec_accept.py t.txt`

### C — mixed-precision byte-cut quality cost (silicon #16/#17, realized locally)
`artifacts/quant/run_oracle_c.sh` → `reports/quality/mixedprec_ppl.tsv`. PPL on
a disjoint code holdout; requant **from Q4_K_M** (pessimistic vs AWQ-from-f16).

| quant | bytes | PPL | Δ vs Q4_K_M |
|---|---|---|---|
| Q4_K_M (baseline) | 1.797 GiB | 4.485 | — |
| Q3_K_M naive | 1.481 GiB (−18%) | 11.432 | **+155%** (catastrophic) |
| Q3_K_M + code imatrix | 1.481 GiB (−18%) | 5.915 | **+32%** |
| Q2_K naive | 1.187 GiB (−34%) | 1.65e7 | model destroyed |

**Verdict:** the byte-cut prize is **real but requires *smart* quant** — imatrix
recovers most of naive's damage (11.4→5.9), exactly silicon #16's "naive is DEAD,
AWQ is the lever." Even imatrix-Q3 costs +32% PPL for −18% bytes *requant-from-Q4*;
the honest test is **AWQ/GPTQ-from-f16 on Colab** (errata). Q2 is a non-starter on
a 3B regardless. **GO-condition for the Colab run:** AWQ ≤4-bit must land PPL within
~5% of 4.485 on this holdout to be worth a custom low-bit kernel.

---

## Stage 1 — path-to-50 (the uncommitted axis-1 work, now verified)

The dirty tree was **not** deprioritized megakernel work — it's the Bible's
Stage-1 axis-1 fusions, bit-identical and previously "user-reviewing":
- `gemm_q4_k_v4_predec_pair` — gate+up fused predec GEMV (**default ON**)
- ffn_down → predec routing (**default ON**)
- `gemm_q4_k_v4_predec_2r` — 2-row ILP (opt-in `DISMANTLE_QWEN_PREDEC_2R=1`)
- k+v fuse (default OFF)

**Parity:** greedy 32-tok Qwen-3B, conditions A=base / B=fusions / C=fusions+2r →
**B and C bit-identical to A. PASS.** (`tools/bench/path_to_50_verify.sh parity`)

**Paired bench** (`… verify.sh bench`, 6×32-tok interleaved, Claude-contaminated
but paired-valid; low variance):

| condition | dec_tps | vs base |
|---|---|---|
| A — predec base, fusions OFF | 24.08 | — |
| B — fusions (gate+up + ffn_down), dirty-tree default | 30.03 | **+24.7%** |
| C — fusions + 2-row ILP | 31.89 | **+32.5%** (+6.2% over B) |

All bit-identical. **Action taken this session:** the 2-row ILP delivered a clean
+6.2% bit-identical (above memory's "+4% opt-in"), and Bible Stage 1 calls for it
to be default — so `DISMANTLE_QWEN_PREDEC_2R` was **flipped default-on** in
`kernels/mod.rs` (opt out `=0`). Net Stage-1 axis-1 gain vs pre-path-to-50:
**24.08 → 31.89 dec_tps = +32.5%, bit-identical.** Gap to llama ~50: now ~1.57×.

**Companion change required to bench at all:** the path-to-50 shaders changed the
shader hash, so `profiles/qwen3b-instruct-q4k.m3pro18.json` `shader_hash` was
re-stamped `d29fff7f…` → `f67726340e…` (additions-only diff; autotuned schedules
unaffected — the new kernels are env-dispatched, not profile-selected).

---

## ERRATA — manual / Colab / multi-session (turnkey)

> **Update (later same session):** several items below are now resolved or have
> Colab notebooks. See `plans/bible_colab_audit_2026_05_30.md` for the audit.
> E2→`colab/01_awq_bytecut.ipynb`, E3→`colab/02_eagle3_train.ipynb` (existing head
> measured: **0.000 accept, 4.5× slower** — retrain needed), QTIP→`colab/03_qtip_3bit.ipynb`,
> E6 SVD→**ran: NO-GO, full-rank** (`reports/oracle/svd_lmhead.json`).

**E1 · Instruments calibration (Bible §1, one-time, GUI).** Anchor the homemade
analyzer to ground truth:
```
xcrun xctrace record --template "Metal System Trace" --output qwen.trace \
  --launch -- ./target/release/dismantle generate --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --kernel-profile profiles/qwen3b-instruct-q4k.m3pro18.json --prompt "fn main() {" --max-new-tokens 64 --temperature 0
```
Open `qwen.trace`, read GPU-busy vs wall, compare to `analyze_tcb_trace.py`'s
busy-fraction note. If >5% off, the analyzer is miscalibrated — fix before
trusting any kernel A/B.

**E2 · AWQ/GPTQ byte-cut from f16 (Colab — the #16/#17 prize).** Oracle C gates
this. On Colab: load Qwen2.5-3B **f16**, run AutoAWQ (or GPTQ) to W4/W3 with a
**code** calibration set, export, convert to GGUF/dismantle layout. **PPL-gate**
on `artifacts/quant/ppl_trim.txt` — must beat imatrix-Q3's 5.915 and ideally
approach 4.485 at <4 bits. Then needs an M3 low-bit GEMV kernel (E4).

**E3 · EAGLE-3 spec head (Colab train → M3 integrate).** Oracle A says n-gram is
NO-GO, so the trained head is the only spec path. **A head already exists**:
`checkpoints/eagle5_final/q3b/head_final.safetensors` (1.83 GB, 2026-05-29). FIRST
measure it, don't retrain blindly:
```
WEIGHTS=models/qwen2.5-3b-instruct-q4_k_m.gguf \
PROFILE=profiles/qwen3b-instruct-q4k.m3pro18.json \
EAGLE5_HEAD=checkpoints/eagle5_final/q3b/head_final.safetensors \
PROMPT='fn quicksort' bash tools/bench/eagle5_paired_bench.sh
```
If accepted-length <2.5 (memory says the old v2 head trained on f16 but serves
Q4_K_M → ~1% accept), retrain EAGLE-3 on Colab on **Q4_K_M captures** (FR-Spec
32K vocab, fusion layers {2,18,33} for 36L). Gate: τ≥2.5 on code.

**E4 · Stage 2 — MLX-class Q4_K GEMV (M3, multi-session, THE primary dense lever).**
Start from the working prototype `silicon-builds/dismantle-q4k-mma`
(`gemm_q4k_mma`, +10–20% batched, bit-identical). Order (each parity-gated +
benched under §1 busy-time BW): vectorized nibble unpack → multi-row register
blocking (4–8 rows) → simdgroup-matrix decode (the M=1 hard part) → split-K →
threadgroup tuning. Port to the **verify/prefill** dispatch first (batched, where
MMA wins); decode M=1 last. Target 47%→60% peak (~50 tps, llama-class, high
confidence); stretch 70–80% (~55–64, MLX-class).

**E5 · Stage 1 leftover — LM-head→predec (M3, ~+1–2%).** Route the Q4_K LM-head
GEMV through the predec cache (`DISMANTLE_QWEN_Q4K_LMHEAD=1` path). Low value (LM
head ~4% of decode) but exact. Parity + bench gate.

**E6 · Oracle B — SVD lm_head recall (M3, low priority).** Needs a Q6_K
dequantizer (no `gguf` pip locally). `pip install gguf`, then SVD `output.weight`,
report energy@rank-r + top-32 logit recall. Only matters if it beats
lm_head→predec — unlikely (LM head ~4%).

**E7 · Stage 5 — fused quantized-KV attention (M3, long-context only).** From
`silicon-builds/dismantle-int4kv` (#15 per-channel int4, cosine 0.998). Build the
read-KV-inline attention kernel; neutral at short ctx, real win >16K. Gate on
real-model PPL at long context.

**Deprioritized (evidence):** megakernel/persistent-loop (committed history
e03ce26/dc7fdf2/a9c6280; attacks the ~12–15% gap, ceiling ~+15%; 8-layer POC 4.4×
slower); n-gram/SAM runtime (oracle A τ=1.43); all host-dispatch / multi-engine /
memory-placement levers (11 dead silicon prototypes).

---

## Using the hardened §1 gate
```
./target/release/dismantle bench … --trace-dispatch --json trace.json
python3 tools/bench/analyze_tcb_trace.py trace.json   # exits 2 on any §1 violation
```
Invariants: (1) busy-time BW ≤ 150 GiB/s, (2) 'other' bucket ≤5% + Σkernel≈busy,
(3) token count from `sample_*` dispatches, (4) parity recorded. `--no-gate` to
print-without-exit; `--model v2lite` for DeepSeek.

## Files touched (uncommitted)
- `tools/bench/analyze_tcb_trace.py` — §1 gate (hardened)
- `tools/bench/oracle_spec_accept.py` — oracle A (new)
- `artifacts/quant/run_oracle_c.sh` — oracle C (new)
- `tools/bench/path_to_50_verify.sh` — Stage-1 parity+bench (new)
- `profiles/qwen3b-instruct-q4k.m3pro18.json` — shader_hash re-stamp
- `reports/oracle/spec_accept.json`, `reports/quality/mixedprec_ppl.tsv` — results
- `crates/dismantle-core/src/kernels/mod.rs` — 2-row ILP **flipped default-on** (this session)
- plus the pre-existing path-to-50 source diff (qwen_dense.rs / quant.metal / kernels/mod.rs / metal/mod.rs / tcb_dispatch_cost.rs)

Left uncommitted to respect the path-to-50 in-review status + the prior session's
deliberately-uncommitted `plans/` and `silicon-builds/`. Build + `cargo test --lib`
green, parity bit-identical. To land (inline identity, no push):
```bash
# Stage-1 axis-1 win (path-to-50 fusions + 2r default + profile rehash)
git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' add \
  crates/dismantle-core/shaders/quant.metal crates/dismantle-core/src/kernels/mod.rs \
  crates/dismantle-core/src/metal/mod.rs crates/dismantle-core/src/model/qwen_dense.rs \
  crates/dismantle-core/tests/tcb_dispatch_cost.rs profiles/qwen3b-instruct-q4k.m3pro18.json
git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' commit -m \
  "path-to-50: predec gate+up/ffn_down fusion + 2r ILP default-on (+32.5% Qwen-3B, bit-identical)"
# Stage-0 methodology gate + oracles + execution log
git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' add \
  tools/bench/analyze_tcb_trace.py tools/bench/oracle_spec_accept.py \
  tools/bench/path_to_50_verify.sh plans/bible_execution_2026_05_30.md
git -c user.name='Joshua Hicks' -c user.email='joshuahicksboba@gmail.com' commit -m \
  "bench: enforce Bible §1 invariant gate + Stage-0 oracles (spec-accept, mixed-prec PPL)"
```
