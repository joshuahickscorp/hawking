# SUPERWAVE STATE — femtosecond + generalized kernel (2026-08-16)

HEAD (this worktree) grok/fs-occupancy-20260816-143029
M3 Ultra, 60 GPU cores, 96 GB unified, 819 GB/s published peak

## Law
fs_per_weight = 152.6252 * ACTIVE_decode_BPW / efficiency
efficiency = achieved_gbps / 819
STORAGE complete_bpw is not ACTIVE decode BPW. 97% of stored mass is unused experts.

## Honest bandwidth control (G014, measured this lane)
Unique-bytes-once decode-shaped traffic, no model logic, GPUStart/GPUEnd only:

| probe | median GB/s | spread |
| token-shape mixed-sub655 (1.14 GB, 98 CBs) | 319.7 | 314.6 / 321.0 / 319.7 |
| token-shape mixed-1p5 (2.23 GB, 98 CBs) | 363.2 | 365.5 / 360.7 / 363.2 |
| unique-once 512 MiB full occupancy | 411.5 | 411.8 / 411.5 / 391.6 |
| unique-once 1024 MiB full occupancy | 301.6 | 301.6 / 303.5 / 300.1 |
| reuse 64 MiB x 4096 (NOT decode) | 536-637 | seq 515-546, conflict 633-649 |

Use 320-411 GB/s as the decode ceiling. Do not use 819. Do not use 560-647 for a
unique-once token. Receipt: receipts/ascent-2026-08-16/Q80_DECODE_SHAPE_BANDWIDTH.json
and receipts/ascent-2026-08-16/G014_FS_OCCUPANCY_CONTROL.json

## Density vs efficiency
mixed-1p5-v1   storage 1.4444 BPW  ACTIVE 4.9795 BPW  2.22 GB/token  1714 fs at 363 GB/s
mixed-sub655-v1 storage 0.6462 BPW  ACTIVE 2.5180 BPW  1.14 GB/token   985 fs at 320 GB/s
Attention + lm_head are 86-88% of bytes moved. Crushing unused experts does not move fs/weight.

## Sub-100 fs
Required ACTIVE BPW: 0.256 (at 320 GB/s) to 0.329 (at 411 GB/s). Unity's 0.6552 is not a
target this box can use. Even if 0.6462 applied to every active weight, best measured
case is 196 fs (411 GB/s) or 253 fs (320 GB/s). UNREACHABLE with existing packs.

## Occupancy caps, ranked by measured ns (sub655-sized token)
1. 1155 serial CBs HOST 222 ms — counterfactual; production is 98 CBs
2. token-shape host wall 22.0 ms — 98 CBs of the honest control
3. 98 CB host-minus-GPU 20.1 ms — serialization, not DRAM
4. unbatched 512-thread organs 6.35 ms — 16.8 GB/s vs 455 GB/s 30-wide
5. token-shape GPU 3.56 ms — the actual byte move
6. unique-once DRAM floor 2.77 ms
7. host memcpy analog 1.71 ms — does not explain 246 ms moe_table_build
8. 1155 nops in one CB GPU 1.48 ms
9. fma extra one organ 103 us
10. residency cold-warm 52 us
11. 10-of-512 gather vs sequential 0 ns (scatter not slower)

Live Q80 at 3.38 GB/s wall / 8.16 GB/s GPU is 1.1% / 2.6% of the honest control,
not 0.135% of peak. The leftover vs the 22 ms control is host identity
(moe_table_build ~246 ms) plus packed-kernel occupancy, not the access pattern.

## Artifacts on disk
mixed-1p5-v1      1.4444457 storage BPW
mixed-1p5-ne4-v1  1.3257127 storage BPW
mixed-sub655-v1   0.6462039 storage BPW
uniform-q4-group64-v1  DE-AUTHORISED. Do not benchmark it.

## Rules
- GPU timing = completed MTLCommandBuffer GPUStartTime/GPUEndTime ONLY
- Paired reps, full spread. Single runs are page-cache confounded
- Q4 is de-authorised
- Do not push, merge, or touch any remote
