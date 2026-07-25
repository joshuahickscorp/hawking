# Sub-bit capability-density reset — precheck

Campaign branch `campaign/subbit-capability-density-reset`, TCC-safe worktree
`~/HawkingWorktrees/subbit-reset`, based on the one-bit-ceiling foundry (`5a85a593`) merged with
the Qwen tooling branch (`c056a105`).

## 1. Live truth

| Fact | Value |
|---|---|
| Heavy lease | **FREE** — `QWEN_GRAVITY_STATE.json` `final=true`, no lease file, controller pid 86590 dead |
| Sealed Qwen result | 18 rows, `SEALED`, ladder sha `563ad51128717126` |
| Parent | `Qwen/Qwen3-235B-A22B-Instruct-2507` rev `ac9c66cc…`, 438 GiB / 118 shards, **resident** |
| Parent logits | cached for all 6 holdout prompts, verified healthy (`gen_paris` argmax = `" Paris"`) |
| Free disk | ~118 GiB |
| Nothing interrupted | no running campaign was touched |

## 2. What the Qwen result actually falsified

`FIXED_WEIGHT_PQ_VQ_CAPABILITY_FAILURE`. Both arms collapsed 6/6 on the real forward while the
bf16 parent stayed healthy: A1 at 1.0075 complete BPW (argmax agreement **0.0**), R2 at 0.4930
complete BPW (argmax agreement **0.0**). Halving the rate barely moved symKL, so the rate was never
the binding constraint at that point — the representation was.

The sealed report's own highest-value untried lever was *row-norm stratification*, on the premise
that **"94 percent of gate/up rows collapse onto ONE codeword."**

## 3. That premise is false — measured, not argued

`tools/condense/qwen_function_aware_probe.py` on real weights, 5 layers × 3 organs × 8 experts, at
the exact R2 geometry:

| | measured |
|---|---|
| single-codeword share, baseline raw Lloyd | **0.0267** (never above 0.047 in any cell) |
| single-codeword share, scale-invariant | 0.0161 |

The 94 % figure does not reproduce at `d16 k1024`. The declared `R5_rownorm_strat` lever is
premised on a pathology that is not there. Registry status must move `alive_untested` →
`premise_falsified`.

Row-norm span **is** real, and larger than claimed: up to **15.5 decades** at layer 0
(2.4e-8 … 1.05), not 5.

## 4. The real wall

Scale-invariant VQ (M01′: quantize direction only, ship one bf16 scale per row at +0.0039 BPW) is a
genuine win, largest exactly where the norm span is largest:

| cell | span (decades) | baseline | scale-invariant | RD floor |
|---|---|---|---|---|
| L0 gate | 15.54 | 0.7214 | **0.6435** | 0.6484 |
| L23 gate | 5.44 | 0.7285 | 0.7028 | 0.6484 |
| L70 gate | 0.40 | 0.7041 | 0.7029 | 0.6484 |
| L93 gate | 10.44 | 0.7156 | 0.7027 | 0.6484 |

The last column is the decisive one. `RD floor` is the memoryless-Gaussian rate–distortion bound
`sqrt(2^-2R)` at that exact index rate. **Layer 0 gate measures 0.6435 against a floor of 0.6484 —
below it.** Every other gate/up cell sits within 8 % of the floor. `down` at 0.156 BPW measures
0.90–0.92 against a floor of 0.8974.

Post-hoc coding of the frozen weights has **≤ 8 % headroom left, and the floor itself is
catastrophic**. No cleverness inside Lane A converts `rel_error 0.65` into capability. Lane A is
closed quantitatively — and that is emphatically *not* an argument for raising the ceiling. It is
the proof that **the source must change**.

## 5. The move: the inventory is a free variable

The ceiling constrains total bits over the *original* weight count. Halving the expert inventory
and doubling the survivor rate is therefore **budget-neutral**:

| arm | complete BPW | gate/up RD floor | routing kept (88-token) |
|---|---|---|---|
| S128 g1.25 d0.3125 | 0.951385 | 0.4204 | 100 % |
| **S64 g2.5 d0.625** | **0.948410** | **0.1768** | 92.2 % (worst layer 76.9 %) |
| S32 g5.0 d1.25 | 0.947080 | 0.0312 | 68.8 % |

All three pass `one_bit_ceiling.assert_complete_bpw_le_one` with exact rational arithmetic. S64 is
the first candidate in this campaign whose gate/up organ lands in a survivable reconstruction
regime at all.

This converts an unsolvable coding problem into a measurable capability question: how much function
survives when the router is restricted to the hottest 64 experts and the survivors are coded well?

## 6. Calibration honesty

The sealed 88-token routing calibration was collected on **the same six prompts the campaign scores
on** — calibration/validation contamination — and its own analysis says it is *"NOT trustworthy for
per-expert bit allocation"*, needing ≈ 979 tokens. This campaign runs a **1200-token frozen corpus
that is disjoint from the scored holdout** (`tools/condense/qwen_calibration_corpus.py`, asserted
disjoint, content-hashed, spanning code / math / reasoning / instruction / tool-format / prose /
rare-token).

## 7. Claims

No capability is claimed by any byte plan or weight-space number above. Only a real
parent-vs-packed forward on the frozen holdout may select a frontier. No arm above 1.0 complete BPW
is scheduled, and none will be.
