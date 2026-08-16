# DSV4F resident gravity — feasibility verdict

Lane `dsv-resident-gravity`. Status **ARITHMETIC_FEASIBLE_QUALITY_UNPROVEN**.

A <=1.5 complete-physical DSV4F *would* sit in the 96 GiB envelope with working room, because 97.43% of logical mass is routed experts and MLA KV is a 512-wide latent. Source precision (148.7 GiB) does not. Whether any codec actually reaches 1.5 complete *and* coherent generation is unproven: experts are already FP4, existing X is underdetermined, and organ cosine is not the gate.

## Claim boundary

- No packed artifact.
- No activation-weighted DSV4F fit.
- No coherent-generation test.
- Runtime not modified.
- Q80 codec rates are an unfitted envelope.
- Model-level gate is coherent generation. Organ estimates are not that gate.

## Geometry (live manifest re-walk)

- Artifact seal prefix match: `True`
- Tensors: 69187 / 69187
- Byte residual: 0
- Logical unpacked params: 290942289362
- Scope logical: base=283273109074 mtp=6610048891 global=1059131397
- Source bytes: 159609485896 (148.648 GiB) at 4.388760 complete BPW
- Official card 284B/13B vs measured logical 290942289362 (routed 283467841536, MTP 6610048891, active geometry lower bound 13233029120). The 284B card is the routed+shared neighborhood; this lane bills every unpacked logical param.

| organ | tensors | bytes | logical params | % params | source BPW |
|---|---:|---:|---:|---:|---:|
| routed_expert | 67584 | 150592290816 | 283467841536 | 97.4310 | 4.2500 |
| shared_expert | 264 | 1107363840 | 1107296256 | 0.3806 | 8.0005 |
| mla | 484 | 4706307584 | 4706011904 | 1.6175 | 8.0005 |
| mhc | 270 | 138945864 | 34736466 | 0.0119 | 32.0000 |
| indexer_compressor | 311 | 801075968 | 487194752 | 0.1675 | 13.1541 |
| hash_layers | 3 | 18616320 | 0 | 0.0000 | — |
| embeddings | 1 | 1059061760 | 529530880 | 0.1820 | 16.0000 |
| lm_head | 1 | 1059061760 | 529530880 | 0.1820 | 16.0000 |
| norms | 180 | 888832 | 444416 | 0.0002 | 16.0000 |
| router_gate | 85 | 92316672 | 46147840 | 0.0159 | 16.0036 |
| other | 4 | 33556480 | 33554432 | 0.0115 | 8.0005 |

## What has to be non-expert

Q80 crushed **routed** experts and left shared + attention + embeddings + lm_head at 8-bit. The same structure holds, more extremely:

- f_routed = **0.974309517** (283467841536 params, 97.431%)
- f_protect = **0.025690483** (7474447826 params)
- source routed BPW (already FP4+UE8M0) = **4.250000**
- source protect BPW = **9.651223**

Shared expert is **protected** (always-on, 0.381% of params). The earlier tensor-schedule envelope that lumped shared with routed is the wrong split for a Q80-analogous policy.

## Complete-physical footprints

Complete physical = codes + scales + codebooks + rank factors + indices + padding + compact catalog (30856008 bytes). Uniform rate against all logical params:

| target BPW | payload GiB | +packaging GiB |
|---:|---:|---:|
| 1.5 | 50.805 | 50.834 |
| 1.4 | 47.418 | 47.447 |
| 1.3 | 44.031 | 44.060 |
| 1.0 | 33.870 | 33.899 |

### Mixed policy (Q80 rates as envelope only)

- Routed at 1.22957 (w1=1.1269, w3=1.2918, w2=1.27), protect at source: **1.446775 BPW, 49.002 GiB**, clears 1.5 = True
- Same routed rate, protect at 8-bit: **1.404354 BPW, 47.566 GiB**, clears 1.5 = True
- Max routed BPW with protect-at-source to hold 1.5: **1.285069**
- Max routed BPW with protect-at-8-bit to hold 1.5: **1.328609**

These are not DSV4F scores. Q80's own 1.43051 was a screen (organ cosine), not a packed coherent artifact.

## Residency arithmetic (96 GiB M3 Ultra)

- Host RAM: 96 GiB
- CLEAN exclusive reserve: 11.0 GiB (8 OS + 2 Metal + 1 scratch)
- Available for model + KV: 85.0 GiB

MLA KV is one 512-wide BF16 latent per slot (`num_key_value_heads=1`). `hc_pre` reduces 4 streams to 1 before attention, so KV is not ×4. Ratio-4 layers add a 128-wide indexer cache.

### uniform 1.5 complete + packaging

| ctx | model+working GiB | margin GiB | fits CLEAN |
|---:|---:|---:|---|
| 4096 | 50.877 | 34.123 | True |
| 32768 | 51.061 | 33.939 | True |
| 131072 | 51.691 | 33.309 | True |
| 1048576 | 57.570 | 27.430 | True |

### Q80-rate routed + protect-at-source (unfitted)

| ctx | model+working GiB | margin GiB | fits CLEAN |
|---:|---:|---:|---|
| 4096 | 49.046 | 35.954 | True |
| 32768 | 49.229 | 35.771 | True |
| 131072 | 49.859 | 35.141 | True |
| 1048576 | 55.738 | 29.262 | True |

### source precision (what is on disk today)

| ctx | model+working GiB | margin GiB | fits CLEAN |
|---:|---:|---:|---|
| 4096 | 148.720 | -63.720 | False |
| 32768 | 148.904 | -63.904 | False |
| 131072 | 149.533 | -64.533 | False |
| 1048576 | 155.412 | -70.412 | False |

1M-context working set (BF16 KV): attn 5.411 GiB + indexer 1.312 GiB + compressor state 0.011 GiB.

**Source precision does not fit.** **1.5 complete, if achieved, does — at 1M context with tens of GiB of CLEAN margin.**

## Why this is not a yes

1. Experts are already FP4. The transferable Q80 rates were measured on BF16.
2. Existing capture is underdetermined: 1275 fullseq tokens, late_hidden 32×4096, AX batch sample 1 row/expert. w2 post-SwiGLU X is not captured.
3. Writer defaults (first-N 64, threshold 16) are far below dim 4096/2048.
4. Q30 static ≤1.5 coherence failed. This lane does not copy that approach.
5. Q80 mixed 1.43051 itself was SCREEN_PASSED_NOT_YET_PACKED_OR_GENERATED.

## Capture plan (if a later lane fits)

- Teacher = official mixed source forward. Not Gaussian. Not a degraded pack.
- Per-(layer, expert) first-N in the existing compact f32le mmap shape. No giant JSON.
- w1/w3 X = post-ffn_norm hidden. w2 X = **post-SwiGLU**, dim 2048.
- Determined floor: ≥512 rows for a rank-256 score; never `rank = min(budget, n_rows)`.
- Uniform-routing fill: 21846 tokens, reused across 43 layers.
- Layer-tile: capture layer L, fit, discard X. Peak X ≈ one layer's w1+w2 buffers.
- Hash layers 0–2: report expert holes; do not invent rows.
- GPU lock for the source forward. This tool does not run that forward.

## Verdict

- Arithmetic: **FEASIBLE** (source resident False; 1.5-if-achieved resident True)
- Quality: **UNPROVEN**
- Generation: **NOT_RUN**

A well-evidenced negative would be a *determined* activation-weighted fit that cannot clear a stated numeric gate. That experiment has not been run. Declaring ≤1.5 resident DSV4F a success from this census would be manufactured optimism.

## Next bottleneck

Bytes are not the wall *if* 1.5 complete is achieved. The wall is a determined teacher-X fit (existing capture is underdetermined). A 21,846-token fill at the streamed warm body (3,275 / 3,323 / 3,425 ms, DIRTY_ENGINEERING) is about 20 hours of 43-layer source forward. After a hypothetical resident pack, the already-measured runtime wall is `metal.gpu` 2.083 s/token (act_quant), not I/O.
