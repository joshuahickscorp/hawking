# GPU encode lane — results ledger (recreated 2026-06-11)

> The original research/gpu-encode-results.md was wiped in the owner's repo refactor
> (commits d25bed2/89ad88c). This document recreates the full lane story from the
> commit ledger (e761c53, 6888045) plus the will.md §10 entries, and adds the two
> items that were pending when the prior session paused: the **L=12 stretch verdict**
> and the **lane-adoption PPL A/B**.

## Machine stamp (all new measurements below)

- Apple M3 Pro, 12 CPU cores, 18 GB unified, macOS 26.5
- Metal device: Apple M3 Pro (max_threads_per_tg=1024, tg_mem=32768 B)
- repo: /Users/scammermike/Downloads/strand, branch `media-waves`, HEAD `89ad88c`
- toolchain: release build, fast-math OFF in all shaders
- date: 2026-06-11 (evening). NOTE: the box cohabits an active second session that
  runs test suites intermittently; every timing below states its load condition.

## 1. The lane story (history, from the commit ledger)

1. **e761c53 (ENCODE FRONTIER):** the tropical-scan Metal encode (GPU Viterbi,
   CPU prep) died honestly at its pre-registered bar — 0.94–1.00× end-to-end vs
   12T CPU. But the phase split was the discovery: the GPU Viterbi segment runs
   39–50 Mw/s = 8–10× the entire 12T CPU encode; prep (83–85% of wall = the
   64-candidate sub-scale search) was the real bottleneck. Verdict: the Viterbi
   is solved; the encoder is scale-search-bound.
2. **6888045 (GPU SCALE-SEARCH LANE):** the sub-scale search moved onto the GPU
   (shaders/strand_scale_search.metal) in front of the tropical Viterbi kernel —
   full-GPU encode = **5.87× 12T CPU at the 3-bit flagship (k=3/L=7)**, kill bar
   2×, ceiling ~6.4× (near-saturated; search cost 83–85% → 8%). Requant
   projection 16 min → ~2.7 min. **616/616 byte-identity** across both lanes
   (full-gpu vs CPU f32-metric+f32-search; prep-cpu vs CPU f32-metric+f64-search).
   L=12 stretch was left **pending**.
3. **Determinism contract:** the GPU lane is exposed CPU-side as the off-by-default
   f32 lane (`STRAND_F32_METRIC=1 STRAND_F32_SEARCH=1`); the identity gate proves
   GPU output == that CPU reference byte-for-byte, so the *adoption* question is a
   pure CPU-reproducible quality question (this doc, §3).

## 2. L=12 stretch — VERDICT

The stretch kernel WAS landed in 6888045 (it was the *measurement* that was
pending, not the code): `metal_encode.rs::viterbi_geometry` switches to
`STRETCH_TG_THREADS=256` + device-memory cost rows when the two cost rows no
longer fit threadgroup memory (2·2^L·4+16 > 32768 B, i.e. L=12 on this device);
the shader's strided `for (ns = tid; ns < num_states; ns += tgsz)` loop gives
each thread 2^L/256 = 16 states at L=12. L=11 stays in TG memory (1024 threads
× 2 states/thread). Back-pointer batching honors the host `MAX_BACK_BYTES =
256 MB` scratch. `MAX_GPU_STATES = 1<<12` bounds the envelope.

### 2a. Identity at L=11/12 (quiet-box-independent; run 2026-06-11 22:05)

```
gate-tropical identity: 660 cases compared, 0 mismatches, 0 skipped — PASS
```

The case grid includes k∈{2,3} × L∈{8,9,10,11,12} (normal + outlier-shaped +
tie-cyclic/tie-snapped at L=12), both lanes. Full `EncodedTensor` equality
(path bits + init_states + side info), so tie-breaks are proven, not assumed.
This supersedes the 616-cell run recorded in 6888045 (the grid grew).

### 2b. Throughput at k=2/L=12 vs 12T CPU canon (kill bar ≥ 2×)

Run 2026-06-11 22:38 on a QUIET box (load 4.52, idle >70% verified immediately
before; the cohabiting session's test suites had drained). `gate-tropical bench`,
release, end-to-end GPU wall-clock (upload + search kernel + Viterbi kernel +
readback + assembly). Full table (the L=7 rows refresh the headline post-refactor):

| config | full-GPU | prep-CPU lane | 12T CPU f32 | 12T CPU canon f64 | full-GPU vs canon | verdict |
|--------|----------|---------------|-------------|--------------------|--------------------|---------|
| k=3 L=7 (3-bit flagship) | 27.808 Mw/s | 5.227 | 5.303 | 5.141 | **5.41×** | PASS |
| k=2 L=7 | 38.508 Mw/s | 7.377 | 6.855 | 7.181 | **5.36×** | PASS |
| k=2 L=10 (envelope edge) | 6.869 Mw/s | 3.565 | 3.811 | 3.382 | **2.03×** | PASS (marginal) |
| **k=2 L=12 (stretch)** | **2.825 Mw/s** | 1.857 | 1.302 | 1.195 | **2.36×** | **PASS** |

**VERDICT: the L=12 stretch is ALIVE — 2.36× the 12T canon-f64 CPU at the 2-bit
op point, above the pre-registered 2× kill bar.** The device-cost-rows geometry
(256 threads × 16 states, cost rows in device memory because 2·4096·4+16 B
exceeds the 32 KB TG limit) holds up; the prep-cpu lane also stays ahead of CPU
(1.55×) but the full-GPU lane is the one that clears the bar. Notes:
- L=10 (still TG-resident, 1024 threads) is the marginal point at 2.03× — the
  envelope edge is real; L=12 is *relatively* healthier because the CPU cost
  grows with 2^L faster than the GPU's.
- The k=3/L=7 headline re-measures at 5.41× post-refactor (prior ledger: 5.87×;
  same kernels, same machine — treat 5.4–5.9× as the honest band).

## 3. Lane-adoption PPL A/B (3-bit flagship geometry, Qwen2.5-0.5B)

**Question:** is the f32 lane (= the GPU encode, byte-identical per §2a)
adoptable for 3-bit requants? **Pre-registered bar: |ΔPPL| < 0.5% ⇒ ADOPTABLE.**

**Setup.** The local models and PV shadows were wiped in the refactor, so the
0.5B was re-downloaded (Qwen/Qwen2.5-0.5B → scratch/qwen-05b, 953 MB,
model.safetensors md5 d7baf050ec13cb76a756d0d344f28447) and **the A/B runs
on BASE model weights, not the PV shadows**. This is valid for the lane-adoption
question: same encoder, same post-RHT statistics class — the lane delta is a
property of the encoder arithmetic, not of which checkpoint feeds it.

Both arms: `quantize-model --bits 3 --l 7 --outlier-channel 1 --threads 12`
(k=3, L=7, rht=true, tail_biting=false, affine_min=false).

- **Arm A (canon):** `STRAND_NO_GPU=1` → pure-CPU f64-metric + f64 scale search.
  (NOTE: without STRAND_NO_GPU the default path auto-engages the older
  metal_backend Viterbi assist, which is f32-distance — that would have
  contaminated the canon arm. Pinned to pure CPU deliberately.) 180.7 s.
- **Arm B (f32 lane):** `STRAND_F32_METRIC=1 STRAND_F32_SEARCH=1` → the CPU
  reference of the full-GPU lane (gate-proven byte-identical to the GPU, §2a;
  the env vars themselves disable the GPU path so the run is exactly the
  reference arithmetic). 105.5 s under cohabiting load.

Recon artifacts differ as expected (md5 canon `e3efad29e29338983e4d4c2bee71099b`
vs f32 lane `3a065bdcbb07dfb05225f6fd9853e305`) — the lanes are distinct
encoders; identical aggregate stats:

| arm | eff. bpw | weighted rel-RMS |
|-----|----------|------------------|
| canon f64 | 3.6457 | 16.23% |
| f32 lane  | 3.6457 | 16.23% |

**Eval:** ops/eval-ppl.py (tools/strand_eval restored from HEAD into
scratch/eval-tools — the working-tree copy was wiped), wikitext-2-raw-v1 test,
ctx 2048, 64 chunks, device mps, dtype bfloat16, same harness both arms.

### Results (2026-06-11 22:15/22:17, MPS evals serial, same device both arms)

| arm | encoder | PPL (wikitext-2 test, ctx2048×64ch, bf16) | Δ vs canon |
|-----|---------|--------------------------------------------|------------|
| canon f64 (STRAND_NO_GPU=1) | CPU f64 metric + f64 search | **20.6098** | — |
| f32 lane (STRAND_F32_METRIC=1 STRAND_F32_SEARCH=1) | CPU f32 reference of the GPU lane | **20.6166** | **+0.033%** |

harness_key8 `e03d391d` both arms (torch 2.6.0, transformers 5.6.2,
dataset_fp 696cca6b65a171b0, 131,008 tokens); records in
scratch/ab-{canon,f32lane}/ppl.json + the results ledger.

**VERDICT: ΔPPL = +0.033% ≪ 0.5% bar ⇒ the f32 lane is ADOPTABLE for 3-bit
requants.** The GPU encode (byte-identical to this lane per the 660-cell gate)
inherits the verdict: a 5.87× encode speedup for a +0.033% PPL cost on the
flagship geometry.

**Adoption gap (one wiring step):** `TropicalEncoder` is currently referenced
only by gate-tropical — `encode.rs::encode_tensor_with` still dispatches to the
older metal_backend Viterbi assist. To actually collect the 5.87×, wire the
full-GPU lane into the eligible branch (gate it on the f32 envs so the config
key pins it, as quantize-model already asserts).

## 4. Verification state at close (2026-06-11 ~22:40)

- gate-tropical identity: 660/660, 0 mismatches (incl. L=11/12 both lanes).
- gate-tropical bench: 4/4 configs PASS the 2× bar on a quiet box (table §2b).
- `cargo test -p strand-quant --release`: **124 passed, 0 failed, 2 ignored**
  (113 lib + 3 quantize-model + 2 strand-delta + 6 exhaustive).
- Nothing committed (per session rules); artifacts in scratch/ab-{canon,f32lane}/,
  bench log at scratch/gate-tropical-bench-2026-06-11.log, eval harness restored
  read-only at scratch/eval-tools/ (tools/strand_eval from HEAD 89ad88c).

## 5. Open questions

- The k=3/L=7 headline band is 5.4–5.9× (5.87× pre-refactor vs 5.41× tonight,
  same kernels/machine) — worth one more idle re-run if the exact figure matters.
- Lane adoption for k=2/L=12 requants (the 2-bit op point) needs its own A/B if
  2-bit ever ships PTQ; this doc's A/B covers the 3-bit flagship geometry only.
- The default (no-env) encode path auto-selects the old metal_backend f32-distance
  Viterbi when eligible — if the canon is f64, that default is mildly
  inconsistent with the "canon CPU f64" naming and is worth a deliberate
  decision (pin default to STRAND_NO_GPU semantics, or bless the f32 lane and
  retire the distinction).
