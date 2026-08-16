I'll mine the preserved-branch backlog for mechanisms only — git/CPU analysis, no GPU runs, no merges. First I'll locate S004 section 9 and the remote `grok/` branch set so the ranking has a real obligation map.Repo is checked out on the backlog-mining branch. Next I’ll pull S004 section 9 and the remote `grok/` branch list.Only one remote `grok/` branch showed up, so the preserved set is likely local or named differently. I'll inventory every branch and find S004.S004 isn’t in the repo docs I have so far. I’ll hunt STEERS, the 90-branch set, and any existing backlog-mining notes.S004 §9 is the integration-backlog mine: extract mechanisms, do not bulk-merge. Next I’ll inventory `grok/*` vs `main` and separate source from artifact noise so the 90-branch / 404-file census is reproducible.The 90/404 numbers came from a prior census. I'll recover that set from the promotion queue and a source-only diff against `main`.Census is running in the background. I’ll start mining the high-priority branch names — Qwen3.8, kernels, Q80 mixed, DSV4F — in parallel.Branch-name parsing tripped on worktree markers. I’ll fix that and compare each unmerged tip to `main` with both three-dot and two-dot diffs.Census is 160 unmerged / 159 with source — I’ll recover the 90/404 cut and start reading the high-priority diffs.Two-dot diffs are polluted by trunk-only files. I’ll use three-dot patches and check whether those symbols still exist on `main`.New-file list is in. Next I’ll read the actual shaders and runtime modules and check whether main already has the same mechanisms under other names.Unique commits name the real mechanisms. I’ll read those patches and check whether main already absorbed them under other names.Many DSV4F files are already on main. I’ll check whether the unique mechanism commits themselves are ancestors of `main` and only the preserve commits are leftover.The 90-branch figure is the 2026-08-16 unmerged slice. Today that slice is **108 tips** (107 with source). Their noise-file touches sum to **10,641**, which is the 10,442 number. I mined **all 160 unmerged `grok/*` tips** against `main` (`a67c37ba0`, the `gk_*` merge). No GPU, no merge, no push.

**The headline:** almost no unmerged Qwen3.8 *runtime* remains — that 33.5 ms path is already on `main`. What is sitting in the backlog, and still missing from `main`, is the Q80 mixed generate path still dispatching the **serial** `q80_binary_group_matvec` (one thread per row) even though the occupancy kernels (`tg256`, `csr_tg256`, `simd3`) are compiled and registered. That is the 863 ms `gpu_matvec` wall.

---

## Census (so the next lane does not re-count)

| cut | branches | unique source | source touches | noise touches |
|---|---:|---:|---:|---:|
| all local `grok/*` | 505 | — | — | — |
| already ancestor of `main` | 345 | 0 (3-dot empty) | 0 | 0 |
| unmerged | 160 | 345 | 1,028 | 166,703 |
| unmerged 2026-08-16 (the “90”) | 108 | 97 | 503 | 10,641 |
| new files that do **not** exist on `main` | 50 tips | 67 files | — | — |

`origin/grok/` has one remote (`wave0-integrate`). The corpus is **local** `refs/heads/grok/*`.

`git diff main..branch` (two-dot) is a trap: old tips “differ” in 2,000 files because they are *behind* `gk_*`, not because they still own those files. Applicability below is 3-dot + “is the unique commit’s *mechanism* on `main`?”

---

## Ranked mechanisms (priority order)

Applies-to-main = “the *idea* is absent from current `main`,” not “the branch fast-forwards.” Every **integrate** row needs a rebase onto `a67c37ba0` first.

| rank | branch | mechanism | obligation | source files (3-dot, noise stripped) | applies to `main`? | verdict |
|---:|---|---|---|---|---|---|
| 1 | `grok/auto-q80-native-mixed-decode-kernel-20260816-063253` **and later** `grok/auto-q80-leftover-deltanet-host-instant-20260816-124239` | Bind already-compiled occupancy mixed-decode kernels (`tg256` / `csr-tg256` / `hgravs simd3`) on the **generate** path; skip `catalog.load_packed` + per-GEMV parse once the GPU weight is resident (`MixedScratch` / `HAWKING_Q80_MIXED_HOST_CACHE`). Dirty: 1.31 s → 445 ms (occupancy bind); leftover Instant 3.23 s → 2.54 s, token 298 ms → 246–283 ms. Same mixed ids, 0 fallbacks. | G003 / G008 / S004 §7.5–7.6 (reconstruction fused into consumption); the 863 ms `gpu_matvec` | `qwen80_mixed_hybrid_decode.rs`, `q80_mixed_decode.metal`, `q80_mixed_decode.rs`, `qwen80_mixed_catalog.rs`, `metal/mod.rs`, `ascension_qwen80_mixed_hybrid_greedy.rs`, `ascension_qwen80_mixed_deltanet_mixer.rs`, `q80_compose_mixed_catalog.py` + tests + pack scripts | **Yes.** `main` has `MetalMixedAccel` + MixedScratch and *registers* `q80_binary_group_matvec_tg256`, but `encode_weight` still launches `"q80_binary_group_matvec"` at `(rows,1,1)`. `mixed_host_cache_enabled` is **absent**. | **integrate** leftover as the tip of this family (it also has 1p5-ne4 pack + DeltaNet mixer simd4 + moe_routed + device SiLU). Rebase — do not take the earlier `d888f3ebb` alone. |
| 2 | `grok/auto-q80-weight-decode-alu-after-20260816-055422` | Ablation: decode ALU is **not** the wall (8× body repeat = 1.02×). Launch+reduce is 73%. `q80_binary_group_matvec_batch10_sharex` runs 10 same-x experts in one launch: 1,036,416 ns → 28,958 ns (**35.8×**). Wired into routed wave. Isolated organ win; token still host-dominated on that tree. | G003 / S004 §7.5–7.6 / reconstruction-in-register | `q80_weight_decode_alu.metal` (**new**), `q80_mixed_decode.metal`, `metal/mod.rs`, `qwen80_mixed_hybrid_decode.rs`, `ascension_qwen80_weight_decode_alu.rs` | **Yes.** No `sharex` / `batch10` symbol on `main`. Not an ancestor of leftover. | **integrate** after leftover (orthogonal: one launch for 10 experts). Needs-rerun of the 35.8× claim on post-`gk_*` `main` before calling it a token win. |
| 3 | `grok/auto-q80-fused-layer-routed-gpu-20260816-123224` | Occupy the 10-expert mixed wave **inside one fused CB** (3 concurrent groups: gate+up, R, L) instead of 10 serial encoders. Dirty GPU 134 ms → 113–123 ms. Plus fold routed wave into the next layer CB / host mixers into the named layer CB / sequential mixer GEMVs into the suffix CB. | G003 / S004 §7.9–7.10 (CB collapse / dispatch fuse) | `qwen80_mixed_hybrid_decode.rs`, `ascension_qwen80_mixed_hybrid_greedy.rs` (plus the 4 unique commits) | **Yes.** Not in leftover. `main` still opens per-expert encoders. | **integrate** after #1. Rebase; the four commits are one stack. |
| 4 | `grok/auto-q80-down-proj-hgravs01-two-20260816-033402` | `down_proj` hgravs01 two-stage: serial 1-thread/row R (160×512) was 199 µs of a 261 µs organ. Winner `tg256_b3` on R + rank-160 on L → **11 µs** (2e-5 numeric). Wired into mixed hybrid. | G003 / S004 §7.5 | `q80_mixed_decode.metal` (+168), `qwen80_mixed_hybrid_decode.rs`, `metal/mod.rs`, `ascension_q80_down_proj_hgravs01_two.rs` | **Yes.** `main` never dispatches `q80_hgravs01_two_stage_*`. | **integrate** into the same mixed-decode rebase as #1. |
| 5 | `grok/auto-q80-up-proj-fused-binary-20260816-033715` | `up_proj` fused binary+CSR: lid-0 serial CSR was the extra 7.6 µs. Cooperative CSR + bind-time columns → 8.04 µs organ; 10-expert pack 2.76 µs/expert. No dense W. | G003 / S004 §7.6 | `q80_up_fused_attack.metal` (**new**), `metal/mod.rs`, `ascension_qwen80_up_proj_fused_binary.rs` | **Yes.** File does not exist on `main`. | **needs-rerun** then integrate: organ-level DIRTY only; confirm it still beats `csr_tg256` on current `main` before grafting. |
| 6 | `grok/q80-device-topk-20260816-010759` | Move top-k + address-table construction onto the device (`qwen80_postnorm_router_top10_select` + `dispatch_qwen80_device_topk_tcb`). | G003 / S004 Q80-VELOCITY (device gather / kill host routing) | `qwen80_device_expert_table.rs/.metal`, `qwen80_uniform_q4_hybrid_decode.rs`, `metal/mod.rs` | **Partially.** `write_selected_expert_table` / `write_top10_address_table` exist on `main` and the token-ns ledger still says “removable if device top-k lands.” The Q4 hybrid decode on `main` does **not** call `qwen80_device_topk_enabled()`. Mixed path never got it. | **needs-rerun** on mixed generate (not Q4). Mechanism still applies; the Q4 tree is stale vs `vecgroup_x64`. |
| 7 | `grok/auto-dsv4f-attention-gpu-still-default-20260816-134406` | Bit-identical **ordered** FP8 fold (one TG/row, same association as serial) + WO-A E8M0 hoist on the **default** attention path. Dirty attn GPU 124 → 72 ms. `hc_sha c94da765` held. Token body did not move (host-exclusive still majority). | G004 / G014 (theory, but this is a real GPU cut) | `matmul.metal` (+170), `gravity_deepseek_v4_native_token_graph.rs`, `metal/mod.rs` | **Yes, and distinct from A005.** A005 *held* the aggressive WO-A/FP8 simd path because it moved `hc_sha`. This commit claims identity-preserving occupancy. | **needs-rerun** on current `main` (A005 / `gk_*` landed after). Integrate only if `hc_sha` still holds. Do **not** enable `HAWKING_DSV4F_MLA_WO_A_SIMD`. |
| 8 | `grok/dsv-attn-weight-io-20260816-010759` | Pin all 43 layers of attention weights; kill `attn_weight_io_prefetch`. Plus a **device-resident LRU** of routed expert payloads (`DeviceExpertLru`, ~12.75 MiB/slot, `weight_slot` bind). | G004 (slab I/O / host-exclusive) | `gravity_deepseek_v4_attn_weight_cache.rs` (**new**), `gravity_deepseek_v4_device_expert_cache.rs` (**new**), `gravity_deepseek_v4_native_token_graph.rs`, `dsv4f_native_token_graph.metal` | **Yes.** `main` has a *host* `gravity_deepseek_v4_expert_cache.rs` (source-chunk cache, no Metal slot). No `DeviceExpertLru` / `AttnWeightCache` on `main`. Prefetch *exists* on `main` (`prefetch_next_attn_only`) — the pin-and-kill is the opposite bet. | **needs-rerun** vs current prefetch. Device LRU is the keep; pin-vs-prefetch is a discriminator, not a bulk merge. |
| 9 | `grok/auto-dsv4f-metal-gpu-true-gpu-20260816-125255` | Pin shared-expert + moe-control so `metal.gpu` 390 ms reproduces (removes that I/O from the GPU-idle wall). | G004 | + `gravity_deepseek_v4_shared_expert_cache.rs` (**new**) on top of #8 | **Yes.** | **integrate** with #8 (same family, later tip). |
| 10 | `grok/q80-component-simdgroup-20260816-010759` | Wire simdgroup component matvec on Q4 hybrid; numeric equivalence. | G002 / G003 (Q4 vehicle — de-authorised as a *target*, still a kernel transfer) | `qwen_uniform_q4.metal`, `qwen80_uniform_q4_hybrid_decode.rs`, `kernels/mod.rs`, `metal/mod.rs`, probe example | **Superseded.** `main` already defaulted the component path to `qwen_uniform_q4_group64_matvec_vecgroup_x64`. | **skip** unless a dirty A/B vs `vecgroup_x64` wins. Do not displace the occupancy default. |
| 11 | `grok/auto-qwen38-unmeasured-next-cost-q80-20260816-064827` | Isolated Q38 TOKEN_NS: dispatch “locks” are host Err / silent no-ops, **not** GPU time. 64-layer Q4 SwiGLU is 26.7 ms of a 63.4 ms dirty sum. Concurrent gate+up saved 0.57 ms. Production path already 33.5 ms. | G001 (Q38 TOKEN_NS) / G006 | `ascension_qwen38_dispatch_lock_probe.rs` (**new**), +55-line stale delta on `qwen38_device_activations.metal` | Probe is unique; shader delta is **behind** `main`. | **skip integrate.** Keep as **negative science** for G001. Do not rebase the shader. |
| 12 | `grok/auto-q80-isolate-q4-non-expert-20260816-091221` | Collapse of mixed-1p5-ne4 is **Q4 DeltaNet × expert crush**, not ne4 alone. | G006 / G008 / S004 §3 (density × runtime co-design) | `q80_isolate_q4_*.py/sh`, pack scripts | Finding is unique; packer already on leftover. | **skip integrate.** Keep the receipt/scripts as negative science. Do not pack another ne4 until attention is touched. |
| 13 | `grok/auto-q80-down-proj-capability-holdout-20260816-064315` | `down_proj` holdout cosine 0.768 **is** 3-bit factor quant, not a packer bug. | G008 / G009-class density / Doctor | `lab/operators/q80_down_proj_capability_holdout.py` + test (**new**) | **Yes.** | **integrate** the operator/test (cheap, CPU). No runtime graft. |
| 14 | `grok/auto-q80-route-starved-tail-p10-20260816-031702` | Measure + extend the route-starved tail (capture coverage). | G008 / Doctor | `lab/operators/q80_route_starved_tail.py` + test (**new**) | **Yes.** | **integrate** operator/test. Capture extension is a different lane. |
| 15 | `grok/q80-residual-encoding-20260815-200718` | Residual compact codec + doctor6 rung + sweep. | G008 density | `residual_compact_codec.py`, `q80_residual_encoding_sweep.py`, `doctor6/rungs.py`, tests | Preserve-commit tip; files may have landed in evolved form. | **needs-rerun** file-level vs `main` `residual_compact_codec.py` (exists). Only graft if the sweep/rung is actually missing. |
| 16 | `grok/fs-occupancy-20260816-143029` | Decode-shape bandwidth probe: sub-100 fs unreachable at active BPW. | G010 | `q80_decode_shape_bandwidth.rs` (**new**) | Finding already in `G013_FS_EFFICIENCY_CLOSURE_V2.json`. | **skip.** Receipt is the artifact; don’t merge the probe unless G001 wants it as a harness. |
| 17 | `grok/qwen38-native-bringup-20260816-024947` | Original native Q38 (the 33.5 ms discovery). | G005 / G001 | `qwen38_*.rs`, `qwen38_device_activations.metal` | **Already on `main`, newer.** 2-dot is *deletions* (branch behind). Unique commit `575eb6b5b` is not an ancestor — cherry-picked/reapplied. | **skip.** Do not rebase. |
| 18 | `grok/q80-runtime-residency-20260816-003204` | Payload-generic persistent address table. | G003 | `device_residency.rs` | **`device_residency.rs` is byte-identical to `main`.** Default-on. | **skip.** |
| 19 | DSV early I/O stack: `dsv-zerocopy-reader`, `dsv-admission-trust`, `dsv-init-index` | Zero-copy verified reader, admission-time chunk trust, mmap artifact index 13.4 s → 139 ms. | G004 | matching `gravity_deepseek_v4*.rs` + tests | **On `main` in evolved form.** `artifact_index.rs` identical; zerocopy/trust files are *larger* on `main`. | **skip.** Next lane: do not re-read. |
| 20 | Live DSV host-exclusive tips (`auto-dsv4f-host-exclusive-not-this-*`, `auto-dsv4f-level-host-exclusive-not-*`) | Overlap/hoist of post-route host-exclusive / next-layer attn. | G004 | `gravity_deepseek_v4_native_token_graph.rs` | In-flight; `main` already has `prefetch_next_attn_only`. | **skip this lane.** Other worktrees own them. Do not merge from here. |

---

## What this means for S004 §9 priorities

1. **Qwen3.8 improvements** — the corpus has been mined. Unmerged remainder is a *negative* (dispatch locks are not token-ns) plus the already-merged native runtime. Density (G006, ≤2.0 BPW) is **not** in this backlog; it is the live `qwen38-mixed-pack` / `qwen38-attention-density` / `qwen38-bpw-descent` worktrees, which are at or behind `main` and are other lanes.
2. **Generalized Qwen kernels** — `gk_*` is HEAD. The missing transfer is not a new shader family; it is **binding the occupancy kernels the family already compiled** on the Q80 mixed generate path (#1–#5).
3. **Q80 mixed matvec/reconstruction** — this is the only frontier-sized unmerged pile. Four independent stacks (occupancy bind + host-cache, sharex, fused 10-expert wave, two-stage down_proj) all still apply and all touch `qwen80_mixed_hybrid_decode.rs`. They must be **composed on one rebase**, not cherry-picked in isolation.
4. **DSV4F I/O** — zerocopy / admission / mmap index / attn prefetch are already on `main`. Remaining unique work is device LRU + pin shared-expert + the identity-preserving ordered-FP8 default (#7–#9). Host-exclusive is live elsewhere.
5. **Gravity/Doctor** — two small CPU operators still apply (#13–#14). July/Q30 doctor lanes do not.
6. **Unattributed TOKEN_NS** — Q38 probe + fs-occupancy already produced receipts that G001/G010 cite. No new ledger code to merge.

---

## WORTHLESS (negative census — do not re-read)

Grouped so the next lane can skip by prefix.

### A. Already on `main` (evolved or identical)

- `grok/dsv-zerocopy-reader-20260815-165045`
- `grok/dsv-admission-trust-20260815-190536`
- `grok/dsv-init-index-20260815-194613`
- `grok/q80-runtime-residency-20260816-003204` (`device_residency.rs` identical)
- `grok/qwen38-native-bringup-20260816-024947` (mechanism on `main`, tip stale)
- `grok/qwen38-reuse-matrix-20260816-021330` (ancestor of `main`)
- `grok/generalized-kernel-20260816-143026`, `grok/qwen38-mixed-pack-*`, `grok/qwen38-token-ns-*`, `grok/q80-host-facets-*` (ancestors)
- `grok/q80-matvec-reconstruction-*`, `grok/qwen38-attention-density-*`, `grok/qwen38-special-unit-*`, `grok/kernel-transfer-screen-*` (tips == `main`, empty)

### B. Blunt `git add -A` / preserve-only (artifact noise is the commit)

Every `lane-*-20260810-*` tip whose subject is `wip: preserve lane source before reap` / `checkpoint: preserve uncommitted` — 18 branches (`lane-admit`, `lane-bw`, `lane-capture-gpu`, `lane-capture-grouped`, `lane-cold-under-10`, `lane-fit`, `lane-g032-beats-baseline`, `lane-integrate-repack` + `-r2`, `lane-iter-startup-q4`, `lane-ladder-v2-fresh-capture`, `lane-loop2-seconds`, `lane-ms-evaluate-seal`, `lane-qn-doctor`, `lane-qn-frontier`, `lane-repack-latency`, `lane-roof`, `lane-speed-recover`, `lane-tracebug`). Source that isn’t `target-*` build residue is pre-ascent Q30 doctor scratch.

Also preserve-only: `dsv-capture-run`, `dsv4fb-capture`, `dsv4fc-forward`, `q80-activation-device`, `q80-downproj-capture`, `q80-velocity-20260815-143906`, `q30-capture-gemm`, `q30-icb-multitoken-fix`.

### C. Q30 / de-authorised vehicle (not a contender; S003 killed Q4-as-target)

`q30-tokenizer-bridge`, `q30-unpin-organ-count`, `q30-all-layer-capture`, `q30-teacher-chain`, `q30-mixed-reader`, `q80-allayer-chain`, `q80-lifecycle-param`, `q80-gravity-coherence`, `q80-gqa-encode` — each carries **~19,200 workspace files** from a `git add -A`. Any real Q80 encode work has been re-implemented on `main`.

### D. July AgentOS / foundry / HIDE (S004: no AgentOS this campaign)

`s4-surface`, `s2b-lab-cutover`, `s6-enabling-branch-audit`, `corec-hist-r1-enabling-builder`, `hawking-e0-adapters-table`, `hcli-sol-terra-luna` (commit itself says superseded), `v0-bridge-train-real`.

### E. Superseded *inside* the Q80 auto-* swarm (keep only the family tip)

Dozens of `auto-q80-*` tips are earlier attempts at the same four mechanisms as ranks 1–5. Do not rebase these; leftover / fused-layer-routed / weight-decode-alu / down-proj-hgravs01 are the survivors.

Worthless-as-duplicates (mechanism absorbed by a later tip in the table):

- DeltaNet mixer / prefix / leftover: `auto-q80-deltanet-mixer-*` except the leftover tip; `deltanet-mixed-prefix-*`, `deltanet-prefix-cb-mixed`, `deltanet-still-opens-cbs`, `deltanet-mixer-d3`, `deltanet-mixer-gpu-true`, `deltanet-mixer-still-serial`, `deltanet-mixer-wall-forward`, `deltanet-mixer-gpu-forward`, `mixed-deltanet-prefix-wait`
- Occupancy bind earlier copies: `auto-q80-native-mixed-decode-kernel` (use leftover), `host-activated-graph-after`, `gpu-time-inside-fused` (occupancy half only)
- Packer: `auto-q80-pack-real-mixed-1p5`, `pack-real-sub-catalog`, `q80-pack-sub655`, `moe-routed-packed-1p5` (all in leftover)
- Suffix/expert-table/geo_tpr64: `cbs-dispatches-gpu-idle`, `suffix-moe-combine-b1`, `suffix-expert-table-gpu`, `suffix-gpu-forward-clean`, `simdgroup-expert-table-suffix`, `q4-organ-still-s`, `isolated-mib-qmv-launch` (`geo_tpr64` already Q38/Q80 default)
- Host first-touch / moe_table: `expert-first-touch-upload`, `first-touch-expert-payload`, `host-moe-table-build`, `host-top-expert-address`, `q80-complete-wall-expert`, `host-top-k-forces` — `main` already has persistent address table + first-touch mmap; remaining host bind is a different wall
- Named GEMV / mixer fold earlier copies: `col-mixer-out-proj`, `remaining-*`, `named-gemv-*`, `named-q8-mixer-shared`, `serial-down-proj-two`, `moe-combine-*`, `mixed-suffix-cb-*`, `routed-mixed-wave-*`, `fused-layer-cb-gpu` (use fused-layer-**routed**-gpu)
- DSV prefetch/hoist copies (live lanes + `main` already prefetch): `auto-dsv4f-host-attn-weight-io-*`, `auto-dsv4f-host-expert-slab-io-*`, `auto-dsv4f-post-route-host-remap`, `auto-dsv4f-host-exclusive-inter-cb`, `auto-q80-dsv4f-host-expert-slab-*`, `auto-q80-host-expert-slab-read`, `auto-q80-host-exclusive-expert-mmap`, `auto-q80-metal-gpu-*`, `auto-q80-probe-attn-prefetch-thread`
- Receipt-hash-only: any tip whose unique commit is `Record landed commit hash on the … receipt`

### F. In-flight (not worthless — just not this lane)

`auto-dsv4f-host-exclusive-not-this-20260816-140433`, `auto-dsv4f-level-host-exclusive-not-20260816-135420`, plus the empty-at-`main` siblings listed in A.

---

## Recommended next consume (not this lane — report only)

One rebase worktree from `main`, in this order, all into `qwen80_mixed_hybrid_decode.rs` / `q80_mixed_decode.metal`:

1. leftover occupancy bind + host-cache + 1p5-ne4  
2. two-stage down_proj  
3. sharex batch-10  
4. fused 10-expert wave  

Then one dirty 3–6 token generate. Only then a paired promotion. DSV LRU + ordered-FP8 are a separate rebase and must not share a GPU window with that generate.

---

## Completion report

- **Task:** S004 §9 backlog mine. CPU/git only. Nothing merged, pushed, or remotely touched. No GPU.
- **Corpus:** 505 local `grok/*`; 160 unmerged; 108 of them are the 2026-08-16 “90” slice (10,641 noise touches ≈ 10,442).
- **Frontier still unmerged:** Q80 mixed generate still launches serial `q80_binary_group_matvec`. Four stacks fix that and still apply. Qwen3.8 native is already on `main`.
- **Skip list:** sections A–E above. Next lane should not open those tips.
