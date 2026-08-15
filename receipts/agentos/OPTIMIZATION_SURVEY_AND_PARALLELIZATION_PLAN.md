I'll survey the receipts and kernels first, then build the observed-optimization table and a non-contaminating parallelization plan from what's actually measured.Primary receipts are in. Next I’ll pull the tournament report, latency/dispatch ledgers, and hunt for the Q4 kernel-TPS artifacts.The tournament report is stale versus SUBSTRATE (3.35 TPS). Next I’ll pull kernel/runtime sources and later campaign receipts so the table matches what’s actually landed.Kernel sources and campaign docs next — those should surface levers that never made it into SUBSTRATE.The Q4-kernel receipts aren’t in the tree root. I’ll search worktrees, runtime Q4 flags, and campaign docs next.Found the Q4-kernel worktree. I'll read its receipts and the uniform-q4 / capture lanes next.Q4-kernel receipts are in the worktree. Reading those measurements and the corresponding kernel/runtime code next.Measured Q4 numbers are in. Next I’ll map remaining dispatch-fold / lm_head / Q80 / capture levers from runtime and campaign docs.Q80 status in the tree may contradict the tournament report. I'll verify that, then pull capture-timing, AgentOS primitives, and remaining kernel levers.# Observed-optimization survey + parallelization plan

Read-only. No files changed. Evidence is from `receipts/agentos/SUBSTRATE.jsonl` (23 lines), the Q4-kernel worktree receipts (not present under this repo’s `receipts/q4-kernel-tps/`), campaign artifacts, and the kernels/runtime.

**Stale docs (do not schedule from these):**
- `receipts/q30-startup-latency/TOURNAMENT_READINESS_REPORT.md` still says Q30 @ **3.35 TPS** and device-table “IN FLIGHT”. That is pre-`KG001`/`KG004`.
- The same report’s “Q80 = only a sub-bit artifact” is a mislabel. The complete on-disk Q80 pack is **binary sign+scale**, not activation-weighted-SVD. See §2.

**Landing gap (controller-critical):** `HAWKING_QWEN30_Q4_KERNEL` / simdgroup8 exist only in worktree `~/.claude-grok/worktrees/q4-kernel-tps-20260814-163106/`. This repo’s `crates/hawking-core/src/metal/mod.rs` does **not** register `qwen30_expert_table_uniform_q4_matvec_simdgroup*`. Trunk can run serial device-table Q4 (~10.8 TPS). The **61 TPS path is not on this tree.**

**61 TPS is coherence-verified, not clean-box BASE_TRUE_TPS.** Lane record: `verified_by_me: "coherence only (TPS contention-blocked)"`. `NS007` still holds.

---

## 1. Observed-optimization table

Status = one of `EXPLOITED` | `AVAILABLE-UNEXPLOITED` | `DEAD`.

| lever | where | measured effect | status |
|---|---|---|---|
| Drop `st_dev` from admission file-identity hard-fail | `LG001`; `LATENCY_GENOME.json` `admit_source_chain_device_identity` + `admit_warm_receipt_identity`; R1/R2 in `mod.rs` / `admission_warm_receipt` | Warm receipt no longer false-invalidated on remount. Startup **13.5→2.3 s** (complete-binary), **~28→2.2 s** (AW). Cold rehash **9585 ms** paid once. Warm admit **815 ms**. | **EXPLOITED** |
| `BASE_TRUE_TPS` = `sum(step_us)/n` including drain | `RUG001`; `NS007` | Naive steady-median ~180–200 tok/s is a lie. True: sub-bit ~24; Q4 serial **3.35→10.76**; drain scales (605 ms @16 tok → 1150 ms @32). Per-step 5 ms timer misses the matmul (in the drain). | **EXPLOITED** (measurement law) |
| Device-expert-table (on-device routing, 1 CB/token) | `KG001`; `qwen30_device_expert_table.metal`; `qwen30_complete_runtime.rs:252–266`, `:4506–4737`; `receipts/q30-dispatch-gap/` M1; verification `uniformq4-tps-20260814-161029` | Binary + Q4: **98→1 CB/token**, host route readbacks **48→0**. Q4: **3.35→10.76 TPS**, bit-identical France ids `[785,6722,315,9625,374,12095,13,151645]`, disp **2165→821**. Host path still 98 CB / 3.29 TPS in `BASELINE_GATES.json`. | **EXPLOITED** |
| Device-resident AR (no per-token sample-id wait) | `qwen30_complete_runtime.rs:231–250` (`HAWKING_QWEN30_DEVICE_RESIDENT_AR`, default on) | Enables the 32-token `tps_device_resident` numbers. Off path is `tps_wait_per_token` **9.98 TPS** vs resident **10.79** on serial (`BASELINE_GATES.json`). | **EXPLOITED** |
| Serial encoder groups (M3) | `receipts/q30-dispatch-gap/` M3; `qwen30_complete_runtime.rs:2455–2461`, `:4527` | Implemented, default ON. Component test exists. **No clean serialized A/B of topology-gap** after device-table (the table’s “first serialized experiment” is still open). | **EXPLOITED** (code) / gap-Δ **unmeasured on clean box** |
| Cold RMSNorm vector-decode fold + `prewarm_static_decoded_vectors` | M2 in `Q30_S_BUCKET_MECHANISM_TABLE.md`; `qwen30_complete_runtime.rs:2375`, `:428` | Cold **291→~98** CB (193 vector CBs folded). Warm structural floor then became 98, now **1** after M1. | **EXPLOITED** |
| RowBlock4 fused-QKV (binary path) | `KG002`; `qwen30_complete_runtime.rs:200–205`, `:2814–2898`; verification `integrate-wins` | Binary: **725→629** disp/token (−96), bit-identical ids `[9835,9835,92603,92603,93298,93298]`. Occupancy ~+3.6% whole-token. Gated to `ScalarControl`. | **EXPLOITED** (binary) |
| Fused QKV for **uniform-Q4** (serial rows) | `KG004`; worktree `qwen30_complete_runtime.rs:206–210`; `MEASURED.json` `lever_a.fused_qkv_serial`; `LEVERS2_GATES.json` kernel=`fused` | **821→725** disp, **10.79→13.831 TPS**, **bit-identical** (logit Δ=0, 32-token ids match serial). | **EXPLOITED in worktree only** (not on this tree) |
| Q4 simdgroup R=1 | `LEVERS_GATES.json` kernel=`simdgroup`; `MEASURED.json` `simdgroup_r1` | **41.806 TPS**, coherent Paris/4/Jupiter, ids match 8+32 vs serial, logit max-abs **4.96e-5**. | **EXPLOITED in worktree** (dominated by simdgroup8) |
| Q4 simdgroup rowblock4 | `MEASURED.json` `simdgroup_r4`; `LEVERS_GATES.json` | **58.198 TPS**, same 4.96e-5 drift, coherent. | **EXPLOITED in worktree** (dominated by simdgroup8) |
| Q4 simdgroup8 (fast coherent default candidate) | `KG004`; `NS006`; worktree env `HAWKING_QWEN30_Q4_KERNEL=simdgroup8`; `MEASURED.json` `simdgroup_r8`; `LEVERS2`/`LEVERS3` | **60.986 / replicate 61.152 TPS**, 725 disp, 1 CB, coherent 3 probes, ids match sealed France, logit max-abs **4.959e-5**, hidden max-abs **2.38e-5**. `simd64` **60.637** — no further width win. | **EXPLOITED in worktree; coherence-only; TPS not clean-box** |
| Coherence composition law | `RG001`; `TOURNAMENT_READINESS_REPORT.md` §1 | Need per-organ output_cosine ~0.98+ because survival ~ `mean_organ_cos ** 48`. Q4 **0.987** lives; sub-bit **0.83** dies. | **EXPLOITED** (selection rule) |
| Uniform-Q4 as the coherent rep | verification `coherence-refit` + `uniformq4-tps`; artifact `uniform-q4-group64-v1` (`mean_component_cosine` **0.994**, seal `2558da7b…`) | Paris / `2+2=4` / Jupiter. Group64 is the working driver codec. | **EXPLOITED** |
| Shared-factor SVD reuse + fit-cache (repack) | `CURRENT_REPACK_WALL_LEDGER.json`; `OLD_REPACK_WALL_LEDGER.json`; NS003 contrast | Cold **4606→3567.8 s** (1.29×). 72-organ micro **42.2→11.0 s** (3.83×). rSVD withheld. Byte-identity tests 13/13. | **EXPLOITED** |
| Grouped/batched MPS expert GEMM for capture | `KG003`; worktree `qwen30_source_bf16_layer_major.rs:1835–1847`, `:2006–2009` | Per-expert MPS **slower** than host (8.05 vs 5.41 min). Grouped: dispatches/layer **256→43.7**, capture **324.8→285.8 s**, ULP 3.4e-5, routes identical by construction. | **EXPLOITED** (capture path) |
| Fused top-k weight renormalize | `qwen30_complete_runtime.rs:207–213` (`HAWKING_QWEN30_FUSED_TOPK_NORM`, default on) | One fewer dispatch vs `qwen_complete_normalize_route_weights`. | **EXPLOITED** (binary/Q4 shared control) |
| Fused add + post-attn RMSNorm | `qwen30_complete_runtime.rs:215–222` | Default on; auto-off when postnorm+router fusion is on. | **EXPLOITED** |
| Fused post-attn RMSNorm + router matvec | `qwen30_complete_runtime.rs:224–228`, `:4659` | Default on. | **EXPLOITED** |
| Paired gate/up SwiGLU (binary) | `Q30_S_BUCKET_MECHANISM_TABLE.md` L37; `dispatch_device_expert_wave` `:3971–3991`; shaders `qwen_direct_packed_gate_up_swiglu_paired_scalar_order.metal` | Production, 2 fused expert dispatches (gate/up + down) on **binary**. Not the Q4 path. | **EXPLOITED** (binary only) |
| Metal pipeline/shader cache | `LATENCY_GENOME.json` `runtime_metal_cold_init` | Cold 1284+1177 ms → warm **24 ms**. Closed as a startup lever. | **EXPLOITED** / **DEAD** as remaining target |
| Capture attention 108 s host | `KG003.transfer`; timing bucket `CaptureTiming.attention_secs` (`qwen30_source_bf16_layer_major.rs:1838`) | After grouped GEMM, capture is **attention-bound ~108 s host**. Metal never touches routing/topk/retention. | **AVAILABLE-UNEXPLOITED** |
| Warm payload load floor (18867 files) | `LATENCY_GENOME.json` `admit_payload_warm_load_no_rehash`; `REPACK_STARTUP_LATENCY_LEDGER.json` | **490 ms** floor: stat-all **135 ms** (7.1 µs/file) + **~355 ms** host copy of 4.294 GB. ≤100 ms **impossible** without packed-blob + mmap/`new_buffer_no_copy`. | **AVAILABLE-UNEXPLOITED** (repack-format; not admission-patchable) |
| Compact-binary `SELECTION_RECEIPT` | `NS004` | Admission parses **117.3 MB** verbose JSON. Seal-cache workaround failed (exit 1). ≤100 ms needs compact-binary selection, not an admission tweak. | **AVAILABLE-UNEXPLOITED** (repack-scope) |
| Dispatch-fold / lm_head fusion (the 1.6×) | `KG004.transfer`; `NS006`; token graph `qwen30_complete_runtime.rs:4539–4716`; worktree still **725 disp/token** after fused-QKV | Active-token Q4 MACs: lm_head **151936×2048 = 311.2e6** vs experts **8×3×768×2048 = 37.7e6** + attn **~18.9e6** + router **0.26e6** → lm_head **≈84.5%** of decode MACs. simd64 **did not** beat simdgroup8. **100 TPS needs ~1.6× via fewer launches / fused lm_head, not wider reduction.** Still 725 dispatches. | **AVAILABLE-UNEXPLOITED** (critical path) |
| Q4 paired gate/up/SiLU (port of binary paired) | binary already at `:3971`; Q4 falls through to 4-dispatch `:3992–4035` | Q4 expert wave is still **gate + up + silu + down** × 48 = **192** of the 725. Fusion would cut ~96–144 launches. Coherence risk: same class as simdgroup (accumulation order) unless scalar-order paired kernel is used. | **AVAILABLE-UNEXPLOITED** |
| Q4 fused final-norm + lm_head + argmax | `:4686–4716` (blit split + 4 launches) | Serial group **breaks** for a blit fill, then reopens for final RMSNorm + **151936-row** Q4 matvec + finite-check + argmax. Biggest remaining kernel. | **AVAILABLE-UNEXPLOITED** |
| Q80 coherent Q4-class complete pack | `QWEN80_COMPLETE_GRAVITY_STATUS.json` is **binary** 1.13 BPW, weight cosine **0.796**, “LOW_FIDELITY…NOT_ELIGIBLE”; no complete `uniform_q4` Q80 | Only per-organ `uniform_q4_group64` evolution candidates. No generate result. | **AVAILABLE-UNEXPLOITED** (blocker) |
| Q80 hybrid native token graph | `qwen80_complete_runtime.rs:1–12`, `:1101–1151` — **9** `QWEN80_HYBRID_NATIVE_OPERATOR_GAPS`; `has_complete_native_operator_backend` false | Module **admits + plans**; “does not make a full token”. TPS gate `WAITING_FOR_CANONICAL_EXACT_RUNTIME`. | **AVAILABLE-UNEXPLOITED** (blocker) |
| Q80 KV Q8 / Q4 state codecs | `QWEN80_ATTENTION_KV_STATE_CODEC_RECEIPT.json` | Component-only (not a token): Q8 cosine **0.99998**; Q4 **0.994**; protected-residual Q4 **0.995**. Q30 decode KV is still **f32**. Not on the 100-TPS critical path. | **AVAILABLE-UNEXPLOITED** (Q80 long-ctx later) |
| Wire Q4 group128 into device-table driver | tournament report “group128 frozen”; artifact `uniform-q4-group128-v1` (`mean_component_cosine` **0.993**, seal `84f390fd…`); `S_GENERATE_GREEDY_RESULT.json` **Paris, same ids**, but **98 CB / 2165 disp** (old host-route path) | Group128 is **coherent** on the old path. Device-table/Q4-kernel ladder is group64-only. Optional, not required for 100 TPS. | **AVAILABLE-UNEXPLOITED** |
| AgentOS typed API (`verify_*`, `profile_token`, `repack`, `promote_candidate`) | `receipts/agentos/README.md:30–32` | Data substrate + `machine_state.py` exist. Typed API **not implemented**. | **AVAILABLE-UNEXPLOITED** |
| Sub-bit AW-SVD Q30 coherence via more capture | `NS001`; perexpert64 all-layer mean **0.8255**, gmean 0.808, late 0.771, product **~3.6e-5** vs ≥0.5 bar | Ceiling is the **rep**, not capture. Q80 sub-bit inherits. | **DEAD** |
| Bit-identical 100 TPS from Q4 matmul | `NS002`, `NS006` | Bit-id ceiling **~13.8** (serial+fused). Parallel reduction breaks bit-id. Tournament gate must be **coherence**, not serial-oracle identity. | **DEAD** |
| Binary RowBlock R=4 transferred onto Q4 serial oracle | `NS005`; `MEASURED.json` `rowblock_r4` | **10.76→4.581 TPS** (occupancy starvation). Bit-identical, slower. | **DEAD** |
| Wider simdgroup (simd64) past simdgroup8 | `MEASURED.json` `simdgroup_x64` **60.637** vs simdgroup8 **61.15** | Flat / slight regression. | **DEAD** as a 1.6× lever |
| Approximate rSVD repack | `NS003`; audit 5406 organs, **223** admit/reject flips, budget_differs only 0.24% | Not byte-identical. Withheld. | **DEAD** (for identity-bound pack) |
| `selection_snapshot` admission-stage shrink | `NS004` | Format-floored. | **DEAD** as an admission fix |
| Clean BASE_TRUE_TPS overlapping another GPU/MEMORY lane | `NS007`; `tools/agentos/machine_state.py:88–99` | Load 7.4 / 3.5 s drain swamps kernel Δ. `clean_box_ok`: **any live worktree**, disk <15 GiB, or `load_1m > 0.5×ncpu` → refuse. | **DEAD** (as a practice) |
| ICB pre-encode of the per-token graph | `dead_levers.md` ICB Type-1; M5 rejected; encode 0.22–0.51% wall | Recoverable ≪5–10 ms unless encode share reopens after fold. | **DEAD** as primary (conditional resurrection) |
| Residency / `use_resource` batching as primary S lever | M4 | ≪20 ms. Rejected as primary. | **DEAD** as primary |
| CPU+GPU pipelining, megakernel/8-layer fusion, multi-CQ, host per-dispatch µs polish | `dead_levers.md`; M-table “Explicitly not reopened”; megakernel **4.4× slower** | Type-1 kills. | **DEAD** |
| Re-ship paired gate/up as a *new* idea | M-table L37 | Already production on binary. | **DEAD** to re-propose |
| Metal compile as startup lever | `LATENCY_GENOME` `runtime_metal_cold_init` | Warm 24 ms. Closed. | **DEAD** |

---

## 2. Remaining tournament work items

Gate: **coherent + BASE_TRUE_TPS ≥ 100 on both Q30 and Q80**.

| # | item | resource_class | clean box? | notes |
|---|---|---|---|---|
| T1 | **Land q4-kernel-tps** (`HAWKING_QWEN30_Q4_KERNEL`, simdgroup8 kernels) onto the promotion branch | `GPU_HEAVY` (compile + coherence generate) | **No** for compile/Paris-class. **Yes** for any TPS number. | Worktree-only today. Without this, trunk is ~10.8 TPS. |
| T2 | **Q30 dispatch-fold + lm_head fusion** to close 61→100 | `GPU_HEAVY` | **No** while implementing + coherence-checking. **Yes** for the 100 claim. | Concrete folds: Q4 paired gate/up/SiLU (`:3992` vs binary `:3971`); fuse q/k row-RMSNorm into QKV/rope (`:4604–4637`); fuse final-norm+lm_head+argmax and remove the blit split (`:4668–4716`). Target ≤~450 disp or an lm_head kernel that actually eats the 84% MAC. |
| T3 | **Clean-box re-confirm of 61** (and later of any fold candidate) | `GPU_HEAVY` | **YES — exclusive** | `clean_box_ok()` must be true. No other GPU_HEAVY/MEMORY_HEAVY. No live worktrees (current encoder treats **any** `~/.claude-grok/worktrees/*` as a blocker). Use `sum(step_us)/n` including drain, device-resident AR, untraced. |
| T4 | **Q80 coherent complete pack** (uniform-Q4 analog of Q30 group64) | `CPU_HEAVY` + `MEMORY_HEAVY` + `IO_HEAVY` | **Yes if it is a large RAM pack** (contends with generate). Authoring/unit tests: no. | Do **not** build sub-bit (`NS001`). Existing Q80 complete is **binary** 11.3 GB, weight cosine **0.796**, never generated (`QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json:44`). Q30 Q4 pack was **38.8 s** — Q80 is 74391 tensors / 512 experts / hybrid, much heavier. |
| T5 | **Compose Q80 hybrid native generate** (close 9 operator gaps) | `GPU_HEAVY` (Metal graph) + `CPU_HEAVY` (plan/bind) | **No** for implementation. **Yes** for first generate timing. | Gaps hardcoded at `qwen80_complete_runtime.rs:1141–1151`. This is the real Q80 blocker, not “port the Q30 kernel.” Q80 is Qwen3-Next: 36 DeltaNet + 12 GQA, 512 experts top-10, shared expert. |
| T6 | **Q80 coherence gate** (Paris-class) on the Q4 pack + composed runtime | `GPU_HEAVY` | **No** (functional). Do not publish TPS. | Same three needles. If binary Q80 is ever generated, treat 0.796 weight cosine as a **red flag**, not a plan. |
| T7 | **Port device-table + simdgroup8 + folds onto Q80** | `GPU_HEAVY` | **No** implement / **Yes** measure | Transfer `KG001`+`KG004`. Q80 lm_head is the **same** 151936×2048, so the 84% MAC lesson transfers. |
| T8 | **Q80 clean BASE_TRUE_TPS ≥ 100** | `GPU_HEAVY` | **YES — exclusive** | After T5–T7. Currently `WAITING_FOR_CANONICAL_EXACT_RUNTIME` (`QWEN80_BASE_TRUE_TPS_GATE_STATUS.json:25`). |
| T9 | **Paired Q30+Q80 100-TPS promotion + TG3/capability receipts** | `GPU_HEAVY` + `DOC_SCHEMA` | **YES** for the TPS half | Tournament-open gate. Do not write TG3 off a contended wall. |
| T10 | **AgentOS primitives** (`verify_*`, `profile_token`, `repack`, `promote_candidate`) + scheduler on `machine_state` | `DOC_SCHEMA` + `TEST_AUTHORING` + light `CPU_HEAVY` | **No** | Explicitly “idle window” work. Must not spawn a generate. |
| T11 | **Capture attention 108 s host** (Metal/attention for capture) | `GPU_HEAVY` (if it touches GPU) or `CPU_HEAVY` | **Yes if GPU** | **Not on the 100-TPS-both path.** Only if a new Q80 Q4 pack needs a new capture. |
| T12 | **Packed-blob + mmap no-copy admission** (490 ms floor) | `IO_HEAVY` + `MEMORY_HEAVY` | Avoid overlapping T3/T8 | Startup, not TPS. Q80 will hurt more (74k files). |
| T13 | **Compact-binary selection format** | `DOC_SCHEMA` + `CPU_HEAVY` | No | `NS004`. Only if another identity-bound repack is required. |
| T14 | **Q4 group128 device-table wire** | `GPU_HEAVY` | No for coherence | Optional. Group64 already coherent. |
| T15 | **Serial-encoder clean A/B** (M3 falsifier) | `GPU_HEAVY` | **YES** | Nice-to-have; M1 already landed the big S win. Do not spend an exclusive window on this before T3. |

---

## 3. Parallelization plan

**Hard isolation rules**

1. At most **one** `GPU_HEAVY` *execution* (generate / profile / pack-on-GPU) at a time.
2. **Zero** `GPU_HEAVY` or `MEMORY_HEAVY` during a clean-box TPS (`T3`, `T8`, `T9`).
3. `STATIC_ANALYSIS` / `DOC_SCHEMA` / `TEST_AUTHORING` may always ride along **unless** they start a generate or a large mmap.
4. `CPU_HEAVY` compile in a **separate worktree** is OK beside a GPU generate **only if** it does not run the binary and does not allocate tens of GB.
5. `clean_box_ok()` today is conservative: **any** live dir under `~/.claude-grok/worktrees` fails the gate. Before T3 the controller must **park/reap idle worktrees** or the gate will never open even if the GPU is idle.
6. Current box (operator): load **7.4**, contended. Treat **now** as WAVE 0 — progress without measurement.

### WAVE 0 — now, contended (no TPS claims)

`T10 AgentOS primitives (DOC_SCHEMA+TEST_AUTHORING)`
`+ T2 design/impl of dispatch-fold in the existing q4-kernel-tps worktree (GPU_HEAVY compile + Paris/4/Jupiter only)`
`+ T5 static close-out of the 9 Q80 operator gaps (STATIC_ANALYSIS → then code, no generate)`
`+ T4 pack-operator authoring/unit tests (TEST_AUTHORING / CPU_HEAVY, no 74k-tensor run)`

**Do not:** T3, T8, T9, a second GPU generate, Q80 full pack (MEMORY_HEAVY).

Open-item placement: **AgentOS** here; **Q30 dispatch-fold implementation** starts here; **Q80 pack/TPS** only as paper/code, not as a run.

### WAVE 1 — still contended, one GPU implementation lane

**Concurrent:**
- **A = T1+T2** (`GPU_HEAVY`): finish fold kernels; coherence-only. One worktree. No `BASE_TRUE_TPS` in the report.
- **B = T10** (`DOC_SCHEMA`/`TEST_AUTHORING`): AgentOS typed API + tests reading `SUBSTRATE.jsonl`.
- **C = T5** (`STATIC_ANALYSIS` → `CPU_HEAVY` file edits): Q80 hybrid graph composition **without** loading the 11 GB catalog onto GPU.

**Not concurrent with A:** T4 pack execution, T6/T7 generate, T11 GPU capture.

### WAVE 2 — exclusive clean box (measurement only)

**Alone:**
1. Reap/park other worktrees so `clean_box_ok()` is true.
2. **T3**: simdgroup8 (and fused-QKV serial as bit-id control) `BASE_TRUE_TPS` re-confirm. This is the “is 61 real?” item.
3. If WAVE 1 produced a fold binary: **immediately after**, same exclusive window, measure the fold candidate vs the just-confirmed 61 (paired, same process if possible).

Nothing else runs. Not AgentOS if it shells out to generate. Not Q80 pack.

Open-item placement: **clean-box re-confirm of 61** lives **only** here.

### WAVE 3 — after 61 is a real number

**If fold < 100:** continue **A = T2** (`GPU_HEAVY` impl) and **do not** start T4 pack yet if A will generate.

**If fold ≥ 100 on Q30 (coherence held):**
- **A = T9 Q30 half** (`GPU_HEAVY`, exclusive) — write the Q30 100-TPS receipt.
- **Then** **B = T4** Q80 Q4 pack (`MEMORY_HEAVY`/`IO_HEAVY`) **alone or with T10 only**. Do not generate Q30 during the pack.
- **C = T5 continued** (CPU/static) may ride with the pack **if** it does not allocate the Q80 catalog a second time.

Open-item placement: **Q80 coherent pack** executes here, not earlier.

### WAVE 4 — Q80 runtime + first coherent tokens

**One GPU lane:** **T5 remaining Metal gaps + T6 coherence generate**.
**Concurrent:** T10; T13 if a pack format bug appeared.
**Not concurrent:** T3-style Q30 remeasure, T11 capture, a second model load.

### WAVE 5 — Q80 TPS engineering

**T7** port (`GPU_HEAVY` impl, coherence-only) — same isolation as WAVE 1.A.
Then **WAVE 5b exclusive: T8** Q80 clean TPS.
Then **WAVE 5c exclusive: T9** both-model 100 + TG3.

Open-item placement: **Q80 TPS** is WAVE 5, never WAVE 0–2.

### What not to parallelize even when it looks independent

| Tempting pair | Why not |
|---|---|
| Two GPU generates (Q30 fold A/B + Q80 L0) | `NS007`. Drain noise kills both. |
| Q80 74k-tensor pack + any generate | MEMORY_HEAVY vs GPU_HEAVY on 96 GB UMA. |
| Clean TPS + “just a compile” in a live worktree | `clean_box_ok` counts the worktree. |
| Capture attention Metal (T11) + Q30 fold generate | Two GPU_HEAVY. T11 is not tournament-critical. |
| ICB / megakernel / sub-bit recapture “just in case” | `DEAD`. Burns the only GPU lane. |

---

## 4. Additional levers (not already a genome/NS row)

Estimates are engineering, not new measurements. Bit-id = vs serial Q4 oracle.

| lever | where | estimate | bit-id / coherence risk |
|---|---|---|---|
| **Q4 paired gate/up/SiLU** (port `qwen30_expert_table_paired_gate_up_swiglu`) | binary `:3817–3991`; Q4 misses it | −96 to −144 disp (48×2–3). Helps launch tax, not the 84% MAC. Maybe **+5–15%** if 725 islands still cost ~µs each; **~0** if already compute-bound in lm_head. | Scalar-order paired can stay bit-id (binary did). A simdgroup fused SwiGLU will take the 5e-5 class. |
| **Fuse q_norm/k_norm + rope + KV append** | `:4604–4637` (3 launches) | −96 disp. Tiny MAC. Same “launch tax vs 84%” caveat. | Low if they stay elementwise fused; rope+cache is already one kernel. |
| **lm_head-specialized tall GEMV** (151936×2048, not the expert-shaped simdgroup8) | `:4699`; `dispatch_binary_matvec` uses the **same** Q4 kernel for lm_head and experts | This **is** the 1.6× if simdgroup8 is occupancy-wrong on a 151936-row grid. Expert tiles assume `rows_per_route` ~768/2048, not 152k. A persistent/split-K or different TG map on **lm_head only** is the highest-EV untried kernel. | Same as simdgroup8 if reduction order changes (coherence OK at 5e-5). A serial-epilogue hybrid could keep bit-id on the head only. |
| **Remove blit split at final head** | `:4668–4685` | One encoder break + fill + reopen per token. Small vs 10 ms, but it is a forced bubble on the 1-CB path. Device-zero the flag without ending the serial group. | Bit-id safe (control, not math). |
| **Fuse finite-check + argmax into lm_head epilogue** | `:4707–4716` | −2 launches. Sampler is cheap; only worth it in a fused head. | Argmax is discrete — if logits stay in the 5e-5 band, token ids should hold (they did for simdgroup8). |
| **Do not count “geometry decode” as a TPS lever** | `ensure_decoded_vector_on_tcb` `:4569–4588`; integrate-wins `geometry=0 warm` | Already warm-zero. Prewarm exists. | n/a |
| **Q80 DeltaNet kernels already in-tree** | `shaders/qwen_next.metal`; `qwen80_*` shaders; gaps still listed as required | Composition, not invention. Closing the 9 gaps is wiring + residuallines, then generate. | High if Q80 binary (0.796 weight cosine) is the body; **use Q4 pack** for coherence. |
| **Q80 shared-expert + top-10 wave** | `qwen80_shared_expert_wave.metal`, `qwen80_all_ten_routed_expert_wave.metal` | Exists as shaders; gap `RoutedTopTen…` still open. Port Q30 device-table (512 experts, k=10) rather than 128/k=8 host bind. | Same as Q30 device-table (bit-id if serial; coherent if simdgroup8). |
| **ICB resurrection after fold** | M5; `dead_levers.md` | Only if post-fold CPU encode **>1 ms/tok**. Unlikely on a 10 ms token. | n/a until encode share is re-profiled **after** T2, on a clean box. |
| **Q8-KV in the Q30 decode loop** | Q30 KV is f32 (`mha_decode_f32_tcb`); Q80 Q8 component cosine 0.99998 | Decode traffic is not the 61→100 wall (lm_head is). Skip for Q30 100. Hold for Q80 long ctx. | Q8: low (0.99998 component). Q4 KV: 0.994 component, **relative L2 0.107** — do not put on a coherence-critical path without a generate gate. |
| **int4 KV** | `dead_levers.md` #15 held | Per-row collapse historically. Not a 100-TPS lever. | High. |
| **mmap packed-blob for Q30 Q4 (18867 `*.hq30uq4`)** | `LATENCY_GENOME` 490 ms floor | Startup only. Transfers to Q80 74k files. | Bit-id if contents unchanged (content sha still binds). |
| **Host capture attention** | `KG003` 108 s | Only if T4 needs a **new** Q80 activation capture. Q30 Q4 pack already exists (38.8 s, no new capture). | Routes stay host by construction if only GEMM moves. |
| **Group128 as a quality hedge** | already coherent Paris on old path | Not a TPS lever (same MAC, slightly less scale traffic). Only if group64 ever fails a harder capability probe. | Low — already matched France ids on 98-CB path. |

### Independent MAC check (why “84%” is real)

Q30 decode, one token, active tensors only:

| op | MACs |
|---|---:|
| lm_head 151936×2048 | 311.2e6 |
| 8 experts × (gate+up+down) 768×2048 | 37.7e6 |
| Q 4096×2048 + K/V 512×2048 + O 2048×4096 | 18.9e6 |
| router 128×2048 | 0.26e6 |
| **lm_head share** | **84.5%** |

That is why `NS006` says “not wider reduction”: simd64 already lost to simdgroup8 on the **same** launches. The remaining 1.6× is **lm_head-shaped work + 725-launch tax**.

### Controller one-liner

**This week, on a dirty box:** one GPU lane implements Q30 dispatch-fold/lm_head (in the q4-kernel-tps worktree) + AgentOS API + Q80 graph paper/code. **First exclusive window:** re-confirm 61, then measure the fold. **Do not** start a Q80 pack or a second GPU generate until that window is done. **Do not** reopen sub-bit, ICB, rowblock-on-Q4, or rSVD.

---

**Sources:** `receipts/agentos/SUBSTRATE.jsonl` LG001–KG004, NS001–NS007, lanes/verifications; `receipts/q30-startup-latency/*`; `receipts/q30-dispatch-gap/*`; `~/.claude-grok/worktrees/q4-kernel-tps-20260814-163106/receipts/q4-kernel-tps/{MEASURED,BASELINE_GATES,LEVERS_GATES,LEVERS2_GATES,LEVERS3_SERIAL_VS_SIMD8}.json`; `crates/hawking-core/src/model/qwen30_complete_runtime.rs`; `crates/hawking-core/shaders/qwen30_device_expert_table.metal`; `tools/agentos/machine_state.py`; Q30 `uniform-q4-group64-v1` / `group128-v1` statuses; Q80 complete-gravity + complete-runtime + state-kv receipts.
