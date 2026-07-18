# Hawking Seed — A / B / C comparison and final selection

Three real, measured candidates, each executing SmolLM-135M to the exact golden `2d1559cf`.

| Dimension | A (microkernel) | B (self-contained scalar) | C (Event Horizon) |
|---|--:|--:|--:|
| authority LOC | 1,068 | 417 | 420 |
| **complete ship LOC** | **~53,481** | **1,945** | **2,204** |
| runtime LOC | 52,413 (delegated) | 1,090 | 1,011 + adapter/metal/subbit/f2 |
| binary size | 1.68 MB | 0.55 MB | 0.60 MB |
| model support | many (predecessor) | SmolLM/Llama | SmolLM/Llama + MoE-ready IR |
| direct quantized execution | N/A (delegated) | no (f32 at load) | **YES** (mmap, per-row tile) |
| Metal | via predecessor | no | **YES** (15× LM-head, measured) |
| peak RSS | 381 MB | 726 MB | **212 MB** |
| throughput | predecessor (Metal) | 9.68 tok/s | ~3–6 tok/s direct-quant CPU |
| sub-bit direct execution | no | no | **YES** (0.401 BPW ternary) |
| 120B F2 bridge | no | no | bounded (fail-closed + synthetic MoE) |
| parity | bit-identical (delegated) | bit-identical | bit-identical |
| auditability | low (52k engine) | high | high |

## Selection — winner: **C (Event Horizon engine)**

Selection is **not** by LOC alone. C dominates on the axes that matter for Hawking's actual mission —
sub-bit-first compression of giant parents, executed hardware-native:

- **Direct-quant execution** with **3.5× less memory than B** (212 vs 726 MB): weights are mmap'd and
  executed from their compressed representation; there is no dense f32 shadow. This is the property that
  scales to 120B/685B/1T parents — B's dequant-at-load does not.
- **Metal-native**: a single descriptor-light kernel accelerates the measured bottleneck (the tied vocab
  projection) **15×**, executed directly on Q8_0 blocks, with argmax agreeing with the CPU reference.
- **Genuine sub-bit direct execution**: a ternary latent factorization runs at **0.401 BPW** as
  `scale·A·(B·x)` without ever forming a dense matrix — the physical proof that Hawking Seed can execute
  a real sub-bit component, the core thesis of the whole campaign.
- **Giant-model extensible**: an MoE-ready IR (Route / Expert / WeightedCombine) and a bounded parent F2
  bridge — the same contract that will host the prepared 685B / 1T / 1.6T adapters.
- All of this in **2,204 LOC** (under the 3,000 gravitational target) with the **same bit-identical
  parity and in-crate auditability as B**.

**A is rejected**: it delegates model math to the 52k predecessor engine — the very thing the Seed exists
to escape — and ships ~53k LOC. **B is a superb pure-dense-CPU result** and the smallest fully self-
contained dense runtime, but its f32 dequant-at-load (726 MB, no Metal, no direct-quant, no sub-bit, no
MoE path) is a dense shadow that does not scale to giants — exactly what C removes.

## Features imported into the final Seed
- **Wire the validated Metal LM-head into the decode loop** (with a near-tie guard) so the final Seed has
  both C's bounded memory and Metal speed while preserving bit-identical tokens — the single highest-value
  next optimization.
- The **shared authority** (record/state/gravity/pack/evidence, ~420 LOC) is identical across B and C and
  is kept as-is.
- B's dequant-at-load could be offered as an *optional* "dense-fast" profile for tiny models, but it
  reintroduces the dense shadow, so the preferred path is to SIMD/Metal the direct-quant GEMV instead.

## Final Seed
The selected base is **Candidate C** — branch `codex/hawking-seed-c`, tagged `hawking-seed-final`, merge
PR **#30**. Full validation rerun green (23 tests + SmolLM vertical path + F2 bridge). A (#28) and B (#29)
are retained as **sealed experiments**; their PRs are superseded by C and are not merged.

## Honest caveats
- The **real GPT-OSS-120B is absent** on this machine; the F2 bridge fails closed on the real path and
  proves the MoE contract on a bounded **synthetic** fixture. No 120B capability is claimed.
- C's CPU decode (~3–6 tok/s) is **slower than B's** (9.68) — the honest cost of on-the-fly dequant
  (incl. per-element f16 rounding across the 49k-row LM head). Metal (15× on the bottleneck) is the
  intended remedy, measured but not yet wired into the decode loop.
- Metal logits are within 6.7e-6 of CPU (argmax agrees) but not bit-identical; parity is taken on CPU.
