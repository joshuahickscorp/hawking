I'll start by reading the Canon, the live goal ledger, the steers, and the proven-state receipts so the plan is grounded in measured reality rather than the ontology.The ledger and survey give the frontier; next I’ll pull Bible gates 1–7, SUBSTRATE, and the Q80/Q30 kernel facts the roof math has to rest on.Bible mid-sections and the remaining receipts next — I need gate criteria plus the exact Q30/Q80 active-byte numbers for the roof math.The 84% lm_head figure looks like a missing 48-layer multiplier — I'll verify Q30/Q80 active bytes, the Q4 codec, and what's actually in flight before locking the roof.Q30 TG3 has no qualification receipt on disk, and `load_uniform_q4` still rejects the paired SwiGLU path — both go into the plan as hard constraints, not footnotes. The 2000-TPS figure does not survive the Q4 byte math; the rest of the plan is written against the roof that does.

# PROGRAM_PLAN — Hawking Canon Phase A → G

**Status:** executable, read-only, no questions.  
**Authority:** Bible GATES 1–7 (`HAWKING_WORLD_MODEL_V2_EXECUTION_BIBLE_RESHUFFLED.md` sha `7ecff5f9…8f43`) · GOAL G001–G019 · STEERS S001–S004 · SUBSTRATE.jsonl 23 records · opt-survey T1–T15 · this machine’s measured Q4 path.  
**Machine:** M3 Ultra 96 GB UMA, **819 GB/s advertised** (`workspace/docs/reference/BASELINES.md:11`). Operator said “~800”; use **819** in every roof number below (800 changes nothing that matters).  
**Frontier now:** G001–G005 VERIFIED. Live work is G007 (Q30 ≥100) + G008 (Q80 runtime + coherent Q4). 61 TPS is **coherence-verified, not clean-box** (`NS007`, SUBSTRATE lane `q4-kernel-tps` `verified_by_me: "coherence only"`).  
**Hard isolation (do not renegotiate):** ≤1 GPU *execution* at a time; 0 GPU_HEAVY / MEMORY_HEAVY during any clean-box TPS; `clean_box_ok()` fails on **any** live `~/.claude-grok/worktrees/*` (`tools/agentos/machine_state.py:88–99`). Park/reap before measuring.

---

## 1. ROOF MATH

### 1.1 The 2000-TPS claim is the **binary / ~1-bit** roof. It is **not** the coherent-Q4 roof.

Decode is weight-traffic bound. Roof TPS = `BW / active_bytes_per_token`.  
`BASE_TRUE_TPS` forbids speculative-accept, prefill, fixtures, partial-forward (`GOAL.md:35`, Bible §3). So the only legal bytes are the weights (and tiny activations/KV) actually touched for **one generated token**.

**Q4 physical bytes/param is measured, not assumed.**  
`HQ30UQ4`: 4-bit codes + fp16 scale / group-64 (`crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs:3–18`).

- bits/param = `4 + 16/64 = 4.25`
- Q30 complete pack confirms it: `complete_physical_bpw = 4.255977838692962`, `all_required_weight_artifact_bytes = 16_243_004_657` (`…/uniform-q4-group64-v1/…TERMINAL_RECEIPT.json:13–14`)
- bytes/param = `4.25598 / 8 = 0.5320`

**Do not count the full embedding as decode traffic.** Marketing “3.3B active” includes the 151936×2048 embed table. Decode gathers **one row** (~1 KB). Roof uses the tensors the token graph actually reads.

### 1.2 Q30 uniform-q4 — active bytes / token

Geometry from `qwen30_complete_runtime.rs:103–117`: 48 layers, h=2048, 32×128 Q / 4×128 KV, 128 experts top-8, intermediate 768, vocab 151936. Source elements `30_532_122_624` (`ASCENSION_PHYSICAL_TOURNAMENT_GATE_STATUS.json:265`).

| tensor class | params / token | note |
|---|---:|---|
| attn Q/K/V/O × 48 | 905,969,664 | Q 4096×2048 + K 512×2048 + V 512×2048 + O 2048×4096 |
| router 128×2048 × 48 | 12,582,912 | |
| 8 × (gate+up+down) 768×2048 × 48 | 1,811,939,328 | device-table already fuses the 8 routes |
| lm_head 151936×2048 | 311,164,928 | once |
| norms + embed row | ~0.21e6 | noise |
| **active params** | **3.042e9** | “A3B”; marketing 3.3B = this + full embed |

Active bytes = `3.042e9 × 0.5320 = 1.618 GB/token`.  
KV at tournament n=32 is `2 × 4 × 128 × 32 × 4 × 48 = 6.3 MB` (f32). Ignore vs 1.62 GB.

| BW | memory-roof TPS | 70% achievable | 50% achievable |
|---|---:|---:|---:|
| 819 GB/s | **506** | 354 | 253 |
| 800 GB/s | 494 | 346 | 247 |

**Current:** 61 TPS × 1.618 GB = **99 GB/s = 12% of 819**.  
**100 TPS** needs 162 GB/s = **20% of peak** — an occupancy/launch problem, not a new representation.  
**333 TPS** needs 539 GB/s = **66% of peak** — a real kernel campaign.  
**2000 TPS** needs 3236 GB/s = **4.0× the hardware**. Impossible on this rep.

### 1.3 Q80 Q4 — active bytes / token

Geometry from `qwen80_complete_runtime.rs:46–67, 479–489, 509–609`: 48 layers, `(layer+1)%4==0` → 12 GQA + 36 DeltaNet, 512 experts top-10 + shared expert (intermediate 512), vocab 151936. Source elements `79_674_391_296` (`QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json:39`). No complete Q4 pack exists; bytes use the same 0.5320 B/param as the proven Q30 codec.

| tensor class | params / token |
|---|---:|
| 36 × DeltaNet (qkvz 12288×2048 + ba 64×2048 + out 2048×4096 + conv) | 1.214e9 |
| 12 × gated-GQA (q 8192×2048 + k/v 512×2048 + o 2048×4096) | 0.327e9 |
| router 512×2048 × 48 | 0.050e9 |
| 10 routed + 1 shared, 3×512×2048, × 48 | 1.661e9 |
| lm_head 151936×2048 | 0.311e9 |
| **active params** | **3.56e9** |

Active bytes = `3.56e9 × 0.5320 = 1.89 GB/token`.  
Q80 Q4 **resident body** (all 79.67B at 4.256 BPW) ≈ **42.4 GB**. Fits 96 GB; does **not** fit beside a 16.2 GB Q30 body + two runtimes. Comparative tournament **must serialize model residency**.

| BW | memory-roof TPS | 70% | 50% |
|---|---:|---:|---:|
| 819 GB/s | **433** | 303 | 217 |

**Q80 @ 333 TPS = 77% of peak.** That is the scariest number in GATE 1. 100 TPS is 23% of peak — same class as Q30@100. 2000 TPS is **4.6× over roof**.

### 1.4 Where 2000 *is* real — and why it is illegal

| rep | Q30 bytes/tok | roof @ 819 | legal? |
|---|---:|---:|---|
| uniform-q4 4.256 BPW (coherent, doctor6 0.987) | 1.62 GB | **506** | **YES — tournament body** |
| uniform-q3 3.125 BPW (codec exists, no complete pack, coherence unknown) | 1.19 GB | 689 | maybe later |
| uniform-q2 2.125 BPW (same) | 0.81 GB | 1014 | maybe later |
| complete-binary 1.13 BPW (Q80 measured 1.133, cosine **0.796**) | 0.43 GB | **~1900** | **NO — NS001 / LOW_FIDELITY** |
| ~1.00 BPW “one bit” on 3.3B marketing-active | 0.41 GB | **~1985** | this is the operator’s 2000 |
| sub-bit AW-SVD ~0.17 BPW | ≪0.1 GB | >8000 paper | **DEAD (NS001)** |

Q80 binary is on disk at 11.285 GB, `mean_component_cosine 0.796`, verdict `LOW_FIDELITY_BINARY_BASELINE_NOT_ELIGIBLE_FOR_RUNTIME_OR_CAPABILITY_PROMOTION` (`QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json:42–45`). Q30 binary is 4.29 GB / 1.13 BPW (`GATE_STATUS.json:262–266`) and is the **legacy ≤1.5 BPW admission**, not the coherent artifact.

**G013 correction the controller must write into the next GOAL evidence:**  
the winner’s fully-optimized-kernel target is **~500 TPS (Q30 Q4) / ~430 TPS (Q80 Q4)** at 819 GB/s, 100% of peak, with a realistic fully-optimized band of **~250–350 (Q30) / ~220–300 (Q80)** if the kernel holds 50–70% of peak. **2000 is the dead binary roof.** Do not chase it on Q4. Do not weaken a seal to name 2000.

### 1.5 The opt-survey “lm_head = 84.5% of decode MACs” is wrong

`OPTIMIZATION_SURVEY…md:197–203` computed **one layer** of experts (37.7e6) against one lm_head (311e6). The token graph loops all 48 layers (`qwen30_complete_runtime.rs:4539–4666`). Correct Q30 MAC/param share:

| op | params | share |
|---|---:|---:|
| experts × 48 | 1812e6 | **59.6%** |
| attn × 48 | 906e6 | **29.8%** |
| lm_head | 311e6 | **10.2%** |
| router × 48 | 13e6 | 0.4% |

So “100 TPS needs ~1.6× via lm_head fusion, not wider reduction” (`NS006`, survey T2) is **half-right on the second clause, wrong on the first**. simd64 already lost to simdgroup8 on the **same 725 launches** (`MEASURED.json` 60.64 vs 61.15) — width is dead. But the remaining 1.6× cannot be “just the head”: the head is 10% of MACs. 61→100 is whichever of {launch tax, small-tile occupancy on 48×expert/attn, lm_head TG-map} the **profile** says. That profile does not exist yet.

The 13.8 → 61 jump at **identical 725 dispatches** (`MEASURED.json` lever_a.fused vs lever_b.simdgroup_r8) proves pre-simdgroup8 was **compute-bound on one-thread-per-row**, not launch-bound. Post-simdgroup8 the bound is unknown. Treat “fold 725→450 ⇒ 100 TPS” as a **hypothesis**, not a plan.

### 1.6 Kernel/graph changes that close 61 → roof

**Already on the Q4 path (worktree only, not trunk):**

| lever | genome | measured | bit-id vs coherent |
|---|---|---|---|
| device-expert-table 98→1 CB | KG001 | 3.35→10.76, ids identical France `[785,6722,315,9625,374,12095,13,151645]` | bit-id |
| fused-QKV for Q4 | KG004 | 10.79→13.83, 821→725 disp, logit Δ=0 | **bit-id ceiling** |
| simdgroup8 | KG004 | 13.83→**61**, 725 disp, logit max-abs 4.96e-5, 3/3 coherent, 0/8+0/32 id-match vs serial | **coherence-gated** (NS002/NS006) |

**Landing gap:** `HAWKING_QWEN30_Q4_KERNEL` / `qwen30_expert_table_uniform_q4_matvec_simdgroup*` live only in `~/.claude-grok/worktrees/q4-kernel-tps-20260814-163106/`. Trunk is serial device-table ~10.8 TPS (opt-survey L9). **T1 is still required.**

**AVAILABLE, do these, in this order:**

| # | change | where | expected | class |
|---|---|---|---|---|
| D0 | **Discriminate the 16.4 ms.** `HAWKING_TCB_TRACE=gpu_prod` on simdgroup8: GPU-us by kernel vs encode/submit. Isolated microbench: (a) lm_head 151936×2048 Q4 GEMV GB/s, (b) one expert-wave 8×3×768×2048 Q4 GB/s | existing trace path (`measure_device_route_ab.sh:10–11,48`) | tells us whether 61→100 is launch, head occupancy, or 48×small tiles | measurement; no math change |
| D1 | **Lift `load_uniform_q4` paired-SwiGLU ban** then port binary paired gate/up | `qwen30_complete_runtime.rs:1812–1814` **refuses** anything except `ThreeDispatchControl`; Q4 falls through `:3992–4035` (4 disp/layer = 192/725); binary paired is `:3971–3991` | −96 to −144 disp. **+5–15% only if D0 says launch-bound; ~0 if compute-bound** | scalar-order paired = **bit-id**; simdgroup fused SwiGLU = coherence-gated |
| D2 | Fuse q_norm/k_norm + rope + KV append | `:4604–4637` (3 launches × 48) | −96 disp, tiny MAC | low risk if elementwise; coherence if reduction changes |
| D3 | Remove blit-split at final head; device-zero the flag inside the serial group | `:4668–4685` | one encoder break/reopen per token, bit-id, small vs 10 ms | **bit-id** |
| D4 | **lm_head-specialized tall GEMV** (split-K / different TG map). Today `dispatch_binary_matvec` uses the **expert-shaped** Q4 kernel on 151936 rows (`:4699`, survey L178) | one launch, 10% of MACs, possibly much more of wall if occupancy-wrong | this is the only remaining *single-kernel* 1.2–1.6× if D0 blames the head | coherence-gated (same 5e-5 class). Serial-epilogue hybrid could keep bit-id on the head only — not required |
| D5 | Fuse final-norm + lm_head + finite-check + argmax | `:4686–4716` | −3 launches + no blit | coherence-gated on the matvec; argmax discrete (ids held for simdgroup8) |
| D6 | After 100: fatter **per-layer** fused waves (attn chain, expert wave) aimed at 50–70% of 819 GB/s | 48-layer occupancy | the 333 / roof path | coherence-gated |

**DEAD — do not reopen (SUBSTRATE NS + survey):**

- NS001 sub-bit capture variants
- NS002/NS006 bit-identical 100 TPS
- NS005 rowblock R=4 on Q4 serial (10.76→4.58)
- simd64 (60.64 < 61.15)
- NS003 rSVD
- ICB / megakernel (4.4× slower) / multi-CQ / host µs polish
- Q8-KV on the Q30 100 path (decode traffic is not the wall)
- group128 device-table wire (group64 already coherent; optional hedge only)

**Q80-specific (after G008 runtime exists):**

- Shaders already on disk: `qwen_next.metal`, `qwen80_all_ten_routed_expert_wave.metal`, `qwen80_shared_expert_wave.metal`, `qwen80_postnorm_router_top10.metal`, plus component passes (router top10, route0/expert65). T5 is **composition**, not invention (`qwen80_complete_runtime.rs:1–12, 1141–1151`).
- Port KG001 device-table to 512-expert / k=10 + shared.
- Port KG004 simdgroup8 + D1–D6. lm_head is the **same** 151936×2048 — D4 transfers. Experts are **skinnier** (512 vs 768) so occupancy will be worse, not better.
- Do **not** first-generate on the 0.796 binary body.

### 1.7 Smallest discriminating experiments (research unknowns)

**Unknown U1 — is 61→100 launch or compute?**  
Cheapest: one `HAWKING_TCB_TRACE=gpu_prod` generate-greedy, n=8, simdgroup8, dirty-box OK (relative kernel shares, not TPS). If encode+submit > ~4 ms/tok → D1/D2/D3 first. If `qwen_uniform_q4*lm_head*` (or the 151936-row dispatch) is >30% of GPU-us → D4 first. If time is smeared across 192 expert + ~200 attn launches → D1 then D6; 100 is still likely, 333 is the real fight.

**Unknown U2 — Q80 DeltaNet ⊗ Q4 composition.**  
Organs can be faithful: champion `uniform_q4_group64` on `layers.0.linear_attn.in_proj_ba` cosine **0.995** (`CHAMPIONS.json:50–68`). RG001 says composition needs ~0.98+ per organ over 48 layers. Smallest test: **do not generate on binary**. After T5 wires generate and T4 emits a Q4 body, run the same 3 needles (Paris / 4 / Jupiter). If fail: one-prompt per-layer hidden cosine vs BF16 teacher, stop at the first layer <0.95 — that names the failing operator. Do not recapture (NS001).

**Unknown U3 — can Q80 Q4 reach TG3 333?**  
Roof is 433; 333 is 77%. Cheapest pre-runtime probe: Q4-pack **one** 512×2048 expert tile + one DeltaNet `in_proj_qkvz` (12288×2048) and microbench GB/s. If a single fat tile cannot beat ~400 GB/s, whole-token 333 is in doubt **before** you spend the pack. If G011 later has a measured Q80 ceiling of (say) 240 with every lever in, **record the lower bound**. Do not weaken TG3. Escalate only if GATE 1 is physically impossible; that is a real fork, not a planning fudge.

**Unknown U4 — 2000 on a new coherent ~1.1 BPW rep.**  
Only legal path to 2000. Sub-bit is closed. Do **not** schedule this in Phase A. After Q4@100 both, a cheap doctor6 on a **Q3 group128** organ (codec already in `load_uniform_qn`, `:1824–1841`) tells you if a denser roof is even thinkable. Idle only.

---

## 2. PHASE-A CRITICAL PATH → MANAGER SELECTED (G007–G013)

Gates from Bible §3 / §30 GATE 1 and `ASCENSION_MANAGER_TOURNAMENT_WORKFLOW.json:412–420`: operational floor **100** BASE_TRUE_TPS both → `MANAGER_ASCENT_TOURNAMENT_ACTIVE` → recursive 125→150→200→250→333 → TG3 both (`tg3_base_true_tps: 333.0`) → capability / HCLI / fit / restart → protected comparative final (`FINAL_MANAGER_TOURNAMENT_PROTOCOL.json`, status `PREPARED_…_NOT_EXECUTED`) → `MANAGER SELECTED`.

**Amendment the controller must apply immediately:** the legacy admission `complete_admitted_artifact_at_most_1_5_bpw` (`GATE_STATUS.json:249–273`) selects the **incoherent binary**. Tournament participants are the **coherent Q4-class** bodies (Q30 4.256 BPW / 16.24 GB; Q80 Q4 ~42 GB once packed). Do not reject Q4 for missing 1.5 BPW. Do not promote binary because it has an admission receipt.

**In-flight to integrate, not duplicate:**

| existing | state | action |
|---|---|---|
| `~/.claude-grok/worktrees/q4-kernel-tps-20260814-163106/` | VERIFIED coherence-only; 61 / 13.8; kernels not on trunk | **source of T1 land**. Do not re-implement simdgroup8. |
| `~/.claude-grok/worktrees/dispatch-fold-20260814-172438/` | worktree exists, `target-t2/` + `target-parallel/` built, `receipts/q4-dispatch-fold/` **empty** | **this is T2**. Resume it. One GPU impl lane. |
| q80-scout | **no worktree of that name** | spawn T5 as `q80-runtime-compose-*`. Static/code, no generate. |
| ~20 other `~/.claude-grok/worktrees/*` (census-build, hook-prefilter, Aug-10 dead lanes, …) | `clean_box_ok` will never go true | park/reap **before** any T3/T8/T9. Dead Aug-10 hawking lanes: absorb-or-delete per scratch rules; non-hawking foreign worktrees: park (rename out of that directory) — do not delete foreign work. |

### Wave list (ordered). One GPU execution. Clean-box exclusive where marked.

| id | what | resource_class | clean_box? | depends_on | expected artifact / gate |
|---|---|---|---|---|---|
| **W0.T1** | Land `HAWKING_QWEN30_Q4_KERNEL` + simdgroup8 shaders from q4-kernel-tps worktree onto the promotion branch (base `grok/integrate-wins@c79c009e2`). Compile + Paris/4/Jupiter only. No TPS number. | GPU_HEAVY compile; generate = functional only | n | — (q4-kernel-tps already has the code) | promotion worktree runs simdgroup8; France ids `[785,6722,315,9625,374,12095,13,151645]`; trunk still 10.8 until this lands. **G007 prep.** |
| **W0.T2a** | Resume `dispatch-fold-20260814-172438`. Implement D0 profile hooks + D1 (must first lift `:1812–1814`) + D3 blit-split (bit-id, cheap) + D4 lm_head kernel **or** D2, **in the order D0 dictates**. Same worktree as T1 after land, **or** rebase fold onto T1. One worktree. Coherence-only generate. | GPU_HEAVY impl | n | W0.T1 (or fold already contains simdgroup8 — check before duplicating) | fold candidate binary; Paris-class 3/3; dispatch count receipt; **no** BASE_TRUE_TPS. **G007.** |
| **W0.T5s** | Close the 9 `QWEN80_HYBRID_NATIVE_OPERATOR_GAPS` (`qwen80_complete_runtime.rs:1141–1151`) as **graph composition**: EmbeddingGather, RmsNorm, DeltaNet conv/rearrange/gated-norm, DeltaNet out+residual, GQA KV/RoPE/gate+residual, RoutedTop10 gate/up/down+combine, SharedExpert, FinalNorm+lm_head+tail-mask+sampler, DeviceResident AR. Shaders exist. `has_complete_native_operator_backend()` becomes true when `required_native_operator_gaps` is empty (`:1251–1253`). **No catalog load, no generate.** | STATIC_ANALYSIS → CPU_HEAVY file edits | n | — | `has_complete_native_operator_backend()==true` in unit/plan tests; still `WAITING_FOR_…` on the TPS gate until first generate. **G006/G008.** Parallel with W0.T2a. |
| **W0.T10** | AgentOS wrappers: `verify_coherence`, `verify_bit_identity`, substrate append, `park_worktrees` / scheduler on `machine_state.py`. No generate. | DOC_SCHEMA + TEST_AUTHORING | n | — | callable CLIs + tests over SUBSTRATE + LEVERS3 needles. **G012.** Parallel with W0.T2a / T5s. |
| **W0.T4s** | Author `admit_qwen80_uniform_q4` + streaming packer (Q30 analog; **no such function exists today** — grep is empty). Unit tests on 1–2 tensors. Do **not** run the 74 391-tensor pack. | TEST_AUTHORING / CPU_HEAVY | n | — | packer + admission compile; no 42 GB artifact yet. **G008 prep.** Parallel. |
| **W1.D0** | Dirty-box `gpu_prod` profile of simdgroup8 vs fold-so-far. Relative kernel shares only. | GPU_HEAVY (one generate) | n | W0.T1; W0.T2a at least D3 or a fold stub | `receipts/q4-dispatch-fold/D0_PROFILE.json`. Steers T2 remaining work. **G007/G013.** Serial with other GPU. |
| **W2.T3** | **Exclusive clean box.** (1) park/reap until `python3 tools/agentos/machine_state.py` prints `clean_box_ok: true`. (2) Re-confirm simdgroup8 BASE_TRUE_TPS = `sum(step_us)/n` incl drain, device-resident AR, untraced, n=32. (3) Same window, paired fold vs that 61. | GPU_HEAVY | **YES exclusive.** 0 GPU/MEMORY other. 0 live worktrees. | W0.T1 + enough of T2 to have a candidate; W0.T10 park helper | sealed `receipts/q4-dispatch-fold/CLEAN_61.json` + `CLEAN_FOLD.json`. **G007 verify.** If fold ≥100 and 3/3 coherent → Q30-100 receipt. |
| **W3.T2b** | **If fold < 100:** next D-lever D0 named (almost certainly D4 lm_head or D1 if launch). Impl in same worktree. Coherence-only. Then **another W2 exclusive** to claim 100. **If fold ≥ 100:** skip to W3.T9a. | GPU_HEAVY | n impl / **Y** remeasure | W2.T3 | Q30 ≥100 clean-box, Paris-class. **G007.** |
| **W3.T9a** | Write Q30 100-TPS protected receipt (half of G010). Do not flip tournament yet. | GPU_HEAVY (already measured) + DOC | used the W2/W3 window | W3.T2b or W2≥100 | `QWEN30_MANAGER_KERNEL_OPERATIONAL`-class receipt, performance-truth. **G007 done.** |
| **W3.T4** | Stream-pack Q80 uniform-q4-group64 complete body from `…/runs/qwen-80b/Qwen3-Coder-Next` (79.67B, 74 391 tensors). Source BF16 is ~160 GB — **must stream**, never hold source+pack. Expect ~42 GB out, minutes not 38.8 s (Q30 was 38.8 s / 18 867 tensors). | MEMORY_HEAVY + IO_HEAVY + CPU_HEAVY | n (but **alone** — no generate, no Q30 load) | W0.T4s; **after** W3.T9a so Q30 100 is in the bag first | complete Q4 Q80 artifact + admission. Cosine vs source on a sample of organs (expect ~0.99 like Q30 0.994 / Q80 organ 0.995). **G008.** |
| **W4.T5e+T6** | Finish any remaining Metal binds; first Q80 generate on the **Q4** pack; 3-needle coherence. If fail → U2 per-layer cosine, NS receipt, do not silently switch to binary. | GPU_HEAVY | n (functional). No TPS claim. | W0.T5s + W3.T4 | coherent Q80 completions **or** sealed NS naming the failing operator. **G008 / G006.** |
| **W5.T7** | Port device-table + simdgroup8 + whatever D-levers won on Q30, onto Q80 expert/lm_head/DeltaNet. Coherence-only. | GPU_HEAVY impl | n | W4.T6 pass | Q80 fast-kernel candidate, 3/3 coherent. **G009.** |
| **W5.T8** | Exclusive clean-box Q80 BASE_TRUE_TPS. Same formula. Park worktrees first. | GPU_HEAVY | **YES exclusive** | W5.T7 | Q80 ≥100 or evidence-backed lower bound + next lever. **G009.** |
| **W5.T9** | Exclusive: both-100 protected authority → flip `MANAGER_ASCENT_TOURNAMENT_ACTIVE`. | GPU_HEAVY + DOC | **YES exclusive** | W3.T9a + W5.T8 | G010 receipt. Tournament opens. |
| **W6.R** | Recursive per-contender 125→150→200→250→333. Each step: impl (GPU, not exclusive) → exclusive remeasure → keep if coherent. Apply D6 / remaining AVAILABLE levers. **One GPU.** Alternate Q30/Q80 only if a lever is model-specific; otherwise finish Q30 ladder then Q80 (Q80 333 is harder). | GPU_HEAVY | exclusive **per claimed number** | W5.T9 | rung receipts. **G011 + G013.** |
| **W6.TG3** | TG3 both: `base_true_tokens_per_second >= 333.0`, complete token loop, no fallback (`GATE_STATUS.json:216–220`). Receipts `QWEN30_TG3_QUALIFICATION_RECEIPT.json` / Q80 twin **do not exist today** (Q30 path 404). Do not reuse any historical 180-TPS-median lie (`RUG001`). | GPU_HEAVY | **YES exclusive** | W6.R ≥333 both | TG3 both PASS. **G011.** |
| **W6.CAP** | Capability eval + measured HCLI + fit + restart/restore. Today Q30 HCLI receipt is **absent** (`QWEN30_MANAGER_OPERATIONS_PREFLIGHT_STATUS.json:10–25`); runtime exact-full-token is PASS. Q80 manager preflight is WAITING. | GPU_HEAVY (eval generate) + DOC | n for authoring; exclusive if a TPS number is in the receipt | W6.TG3 | `QWEN30/80_CAPABILITY_EVALUATION_RECEIPT`, `…MEASURED_HCLI_RECEIPT`, restart receipt. **G011.** |
| **W7.FT** | Protected comparative final. Protocol already sealed (`FINAL_MANAGER_TOURNAMENT_PROTOCOL.json`). Blind corpus, opposing candidate is read-only red-team, protected verifier adjudicates, candidates cannot self-promote (`:23–27`). **Serialize residency** (42 GB + 16 GB). Winner freeze + loser cold-store (`WORKFLOW.json:400–405`). Human + protected controller certify (`:430–432`). | GPU_HEAVY + DOC | exclusive for any timed task | W6.CAP | `ASCENSION_MANAGER_TOURNAMENT` + `ASCENSION_MANAGER_WINNER` + `ASCENSION_ALTERNATE_OFFLOAD`. **G011 = GATE 1.** |
| **W7.ROOF** | Drive the **winner only** toward the §1 roof (~500 / ~430), not 2000. Same levers, exclusive measures. Stop on evidence-backed ceiling. Write G013 roof-math receipt (this section) + measured approach. | GPU_HEAVY | exclusive for claims | W7.FT | G013. Then Phase B may start. |
| **W*.G012** | Every wave above **appends** genome/NS/lane/verification to `receipts/agentos/SUBSTRATE.jsonl` via the W0.T10 API. No second ontology. | DOC / LIGHT | n | rides along | G012 embryo actually exercised. |

**Do not launch:** T11 capture-attention Metal, T12 packed-blob admission, T13 compact selection, T14 group128, T15 serial-encoder A/B, any sub-bit, any Q80 full pack before W3.T9a, any second GPU generate, HIDE, V4 Flash port, Odyssey training (`ODYSSEY_LAUNCH_AUTHORIZED` is false and is a **different program** — Ramanujan/GLM, not Bible Phase B).

---

## 3. SUPPORTS TO SCAFFOLD (Bible §4–18), prioritized by proven rent

S001: do not start the future roadmap. Wrap what this campaign already does. Derive from SUBSTRATE + the generate example `ascension_qwen30_complete_native_runtime` + `machine_state.py` + `tools/verify/perfgate.py` (rebuild TPS gate, not our authority).

### Build NOW on idle (4 primitives). Ride with W0.T2a / T5s.

| primitive | why-needed-now | resource_class | build-on-idle? | derived-from |
|---|---|---|---|---|
| **1. `verify_coherence`** | Every GPU lane already runs Paris / 4 / Jupiter (`LEVERS3_SERIAL_VS_SIMD8.json`, `BASELINE_GATES.json`). G007/G008/G009 cannot promote without it. | TEST_AUTHORING | **yes** (the wrapper). The generate it shells to is GPU_HEAVY and must go through the scheduler. | G002 + G005 3-needle; sealed France ids `[785,6722,315,9625,374,12095,13,151645]`; needles `Paris` / `4` / `Jupiter` |
| **2. `verify_bit_identity`** | Serial-oracle id compare + optional logit max-abs. Tells D1/D3 (bit-id) from D4 (coherence). G001 already used this. | TEST_AUTHORING | **yes** | G001 integrate-wins `[9835,9835,92603,…]`; G005 `ids_identical_8/32`; NS002 |
| **3. `benchmark` / BASE_TRUE_TPS authority** | One function: refuse unless `clean_box_ok()`; run untraced device-resident generate; `tps = n / sum(step_elapsed)` including drain (`RUG001`). Print CONTAMINATED and refuse to seal if the gate fails. G007/G009/G010/G011/G013 are this function. | TEST_AUTHORING + LIGHT_CONTROL | **yes to write**. **Never** invoke on idle during a GPU lane. | RUG001, NS007, `MEASURED.json` method string, `machine_state.clean_box_ok` |
| **4. substrate append + `park_worktrees` scheduler** | `SUBSTRATE.jsonl` is hand-edited. Lanes will lose genomes if this stays tribal. `clean_box_ok` is unusable until something parks the ~20 live worktrees. Scheduler: given a resource_class, allow/deny vs snapshot (S001 §5C). | DOC_SCHEMA + LIGHT_CONTROL | **yes** | S001 idle-fill README:30–32; `machine_state.py:65–99`; SUBSTRATE types lane/verification/negative_science/genome |

These four **are** the G012 embryo Odyssey and V4 Flash actually need first (Bible §6: profile, capability, clean promotion, store genome, retrieve NS, query machine, open/park worktree).

### Wait — wrap later, when a tournament lane first needs the call twice.

| primitive | why not now | resource_class | idle? | derived-from |
|---|---|---|---|---|
| `verify_capability` | Capability receipt schema exists but the eval is a W6.CAP generate suite. Wrapper without a suite is theater. | GPU_HEAVY when run | author stub only | `physical_capability_evaluation.v1`; HCLI receipt still absent |
| `profile_token` | D0 can call `HAWKING_TCB_TRACE=gpu_prod` directly once. Promote to API **after** D0 proves the trace shape. | GPU_HEAVY when run | yes to wrap after D0 | `measure_device_route_ab.sh` |
| `repack` / `compile_kernel` / Gravity / Doctor | Exist as `dual_gravity_worker.py`, cargo, doctor6. Q30 Q4 pack already exists (38.8 s). Q80 packer is T4s **code**, not a product API. | CPU/MEMORY | no extra wrapper | G001 repack 13/13 byte-id; doctor6 0.987 vs 0.685 |
| `promote` / `revert` | Needed at W5.T9 / W7.FT. A two-function script over git worktree + receipt seal is enough; do it when the first promotion happens, not now. | LIGHT | when first used | G001 compose-wins; Bible §4 candidate/revert |
| typed Campaign/Experiment/Lane/Candidate/Receipt objects | SUBSTRATE **records** already are the objects. A pydantic layer with no caller is the ontology S001 forbids. Add types the first time the scheduler has two producers. | DOC_SCHEMA | later | S001 §8 “Grok lane → ExperimentLane” |
| World State / Self Model / Model Auto / Cognitive Scheduler / event bus | Phase D. A `machine_state` snapshot **is** the Machine Genome ancestor. Stop there. | — | no | Bible §12, S001 “DO NOT build yet” |
| HIDE / U / Chat / VisionMCP / V4 Flash port / fleet / CUDA | Forbidden until their gates. | — | **no** | S001 + Bible §4, §19, §27 |

**Rent test (Bible §4):** “Can a resident optimizer invoke this without a human translating intention into a one-off shell ritual?” After W0.T10 the answer is yes for coherence, bit-id, clean TPS, genome write, and “is the box clean / park it”. That is enough for Phase A + the first Odyssey rung.

---

## 4. PHASES B–G (GATES 2–7)

Do not start these. Sequence only. Phase A must leave the G012 callable set or B/C stall (Bible §6).

### GATE 2 — Odyssey foundation (Bible §5, §30). Obligation G014.

- **Criteria:** transferable genomes; callable research machinery; Context/KV begun; Machine Genome; negative science. Test: *model N+1 begins closer to a good physical solution because N taught something.*
- **Inherits from GATE 1:** winner manager + fully-optimized **Q4** kernel (the real roof, not 2000); KG001–KG004, RG001, RUG001, NS001–NS007; per-architecture lesson “Qwen3-MoE vs Qwen3-Next (DeltaNet+GQA+shared)”; G012 APIs; worktree discipline; performance-truth law.
- **Biggest work items:** (1) first Odyssey **rung** on a **dense** Qwen already staged (`Qwen/Qwen2.5-7B-Instruct` / 14B, `BASELINES.md:47–52`) — cheap transfer of attn/lm_head/Q4/simdgroup8 without a new MoE runtime; (2) emit `TransferReport` with a measured head-start (e.g. days-to-coherent-100 vs Q30’s campaign); (3) second rung on an **MLA** body (`deepseek-ai/DeepSeek-V2-Lite`, `BASELINES.md:56`) as the information-gain step into V4 Flash. **Do not** flip `ODYSSEY_LAUNCH_AUTHORIZED` — that file is the Ramanujan/GLM-5.2 **training** package (`ODYSSEY_LAUNCH.md`, `ODYSSEY_PACKAGE.json` status `PREPARED_NOT_STARTED`). Bible Phase B is transfer-optimization of subsequent **models**, not that trainer.
- **Depends on:** G011 + G012 + G013 receipt. Winner kernel is the starting genome, not a fresh search.

### GATE 3 — V4 Flash resident optimizer (Bible §7–9, §30). Obligation G015.

- **Criteria:** verified source → ArchitectureGenome → Gravity frontier → lowest **coherent** executable rep → native Apple runtime → machine-specific tune → HCLI + AgentOS first-class → Hawking-specific research qualification (Bible §8: root-cause, Gravity prescription, false-win rejection, `VERIFIED_DELTA_NS_PER_RESEARCH_HOUR`, …) → `RESIDENT OPTIMIZER LOCK`.
- **Inherits:** Odyssey genomes + G012 machinery **must already be machine-callable** (Bible §6). V4 Flash is 284B / 13B (`BASELINES.md:68`). Existing `receipts/dsv4f_fullseq_capture_L0` / `L1` + Metal (`deepseek_v4_*.metal`) are **assets**, not a restart. Sec 9: do not blindly rerun dead science; reopen only if a precondition changed.
- **Biggest work items:** (1) Gravity the Flash body to a coherent executable (Q4-class prior, not sub-bit); (2) native runtime (MLA/MHC — new architecture, expected per S002); (3) qualification suite that proves it improves **Hawking** (not MMLU).
- **Depends on:** GATE 2 (machinery + at least one transfer rung so Flash is not also teaching “what is a genome”).

### GATE 4 — HCLI/AgentOS stands alone (Bible §10–19, §30). Obligation G016. **`HIDE_REVAMP_ALLOWED = TRUE` only after this.**

- **Criteria:** the §19 checklist — runtime/Gravity stable, Q30/Q80 lineage + Odyssey transfer + V4 Flash resident, Model Registry/Auto, durable sessions, task graph, tools, worktrees, Memory/Context/KV, Skill Foundry, World State + Self Model, resident optimizer invokes profiler/compiler/Gravity, HCLI installs/restarts, models restore, machine-readable APIs.
- **Inherits:** G012 embryo + whatever Odyssey/Flash productized. HCLI already exists as crates (`hawking`, `hide-backend` is **not** the gate). Grow the **headless** surface a serious user can live in (`hawking status/machine/models/tasks/…`, Bible §12).
- **Biggest work items:** (1) World State + Model Auto + Cognitive Scheduler as **real state**, not command grammar; (2) model acquisition + Gravity-install + autotune (Bible §15–16) using Phase A’s pack/verify/benchmark; (3) persistence/restart that Q30 preflight is still missing (`MEASURED_HCLI_RECEIPT` absent).
- **Depends on:** GATE 3 (resident optimizer is on the checklist). G012 is the seed, not the gate.

### GATE 5 — HIDE archaeology (Bible §20–24, §30). Obligation G017. **No visual design.**

- **Criteria:** historical HIDE recovered (`HIDE_ARCHAEOLOGY/`); Claude Essence Pack; Codex Essence Pack; vendor mass distilled; `HIDE_BACKEND_SYNTHESIS.md` + frozen `HIDE_BACKEND_CONTRACT_V1`.
- **Inherits:** a complete headless organism to compare against (GATE 4). Without that, archaeology has nothing to accept/reject.
- **Biggest work items:** (1) own-repo HIDE recovery (branches/tags/deleted U-Chat-Code); (2) lawful VisionMCP essence extraction — no DRM/auth bypass, no proprietary source import (Bible §21); (3) freeze the backend contract (Session/Task/Agent/Tool/Worktree/…) **before** any UI.
- **Depends on:** `HIDE_REVAMP_ALLOWED` from GATE 4. **Do not plan colors, panels, fonts** (Bible §27).

### GATE 6 — HIDE built with HCLI (Bible §25–26, §30). Obligation G018.

- **Criteria:** U / Chat / Code / ExecutionCapsule / Model Center / Machine Center on the **same** control plane. HIDE renders HCLI; invents no graphical-only substitutes.
- **Inherits:** GATE 5 contract + GATE 4 organism + resident optimizer as the HIDE build **manager** (Bible §25 dogfood).
- **Biggest work items:** (1) surfaces bound 1:1 to contract objects; (2) Model Center = already-built acquire→Gravity→autotune flow; (3) Machine Center = Machine Genome / residency / health.
- **Depends on:** GATE 5 contract freeze. Visual system last.

### GATE 7 — Public machine-adaptive Hawking (Bible §28, §30). Obligation G019.

- **Criteria:** repo-first HCLI package + HIDE package; local acquire + Gravity + autotune; offline-first; no account/subscription/telemetry required; HCLI never requires HIDE.
- **Inherits:** GATE 6 + Machine Genome methods (this M3 Ultra is an **ancestor**, not a constant — Bible §15).
- **Biggest work items:** (1) install/restore receipts; (2) machine fingerprint → priors → local profile; (3) optional external MCP/OpenAI-compat (Bible §17–18) — already useful beside Claude Code/Codex.
- **Depends on:** GATE 6. Terminal.

### What Phase A must leave behind (G012 + G013) so B–G are not a rewrite

1. Callable `verify_coherence` / `verify_bit_identity` / `benchmark`(clean-box) / substrate genome+NS / `park_worktrees`+`clean_box_ok`.
2. Kernel genomes that transfer: device-table, fused-QKV, simdgroup8, and whichever of D1–D6 actually moved TPS — each with invalidation + bit-id-vs-coherence class.
3. Representation law RG001 (Q4 lives, sub-bit dies) and the **corrected roof math** (this §1).
4. Two architecture runtimes: Qwen3-MoE (Q30) and Qwen3-Next hybrid (Q80) — S002: building the per-arch runtime **is** the work.
5. Performance-truth law as code, not a paragraph.
6. A winner kernel pushed toward **~500/430**, with a sealed measured ceiling if short of peak.
7. **No** HIDE UI, no V4 Flash port, no Odyssey trainer launch.

---

## 5. EXECUTION SCHEDULE — next 12 lanes from NOW

Controller launches these autonomously. One GPU execution. Tag = `parallel` (with the current GPU owner) | `serial` (is the GPU owner) | `clean-box-exclusive`. Map → G-obligation.

**Before lane 1:** do not start a third hawking GPU worktree. Resume `dispatch-fold-20260814-172438` **or** rebase it onto q4-kernel-tps. Confirm whether that tree already contains simdgroup8; if yes, T1 land is a cherry-pick onto the promotion branch, not a rewrite.

| # | lane | tag | G | exact action |
|---|---|---|---|---|
| **1** | `T10-agentos-wrappers` | **parallel** | G012 | In-repo (or a DOC worktree): implement the four NOW primitives in `tools/agentos/` — `verify_coherence.py`, `verify_bit_identity.py`, `benchmark.py` (hard-refuses unless `clean_box_ok`), `substrate.py append`, `park_worktrees.py` (moves **idle** hawking worktrees to a parked dir the enumerator ignores; never delete dirty trees). Tests: parse `LEVERS3` + SUBSTRATE; `machine_state` self-check. **No generate. No cargo --release of the runtime.** |
| **2** | `T5-q80-compose` | **parallel** | G006/G008 | New worktree `q80-runtime-compose-*`. Close gaps at `qwen80_complete_runtime.rs:1141–1151` by wiring existing `qwen_next_*` / `qwen80_*` shaders into one token graph so `has_complete_native_operator_backend()` is true. Unit/plan tests only. **Do not admit the 11 GB binary, do not generate, do not allocate the catalog on GPU.** This **is** the missing q80-scout. |
| **3** | `T4s-q80-q4-packer` | **parallel** | G008 | Same tree as #2 or a CPU worktree. Write `admit_qwen80_uniform_q4` + streaming packer (clone `uniform_q4.rs` / Q30 38.8 s path). Unit-test 1 tensor. **Do not pack 74 391 tensors.** |
| **4** | `T1-land-simdgroup8` | **serial** (GPU compile; functional generate only if needed to prove the land) | G007 | Cherry-pick/land q4-kernel-tps `HAWKING_QWEN30_Q4_KERNEL` + shaders onto the fold/promotion tree. Coherence 3/3. **No TPS seal.** If fold tree already has this, skip generate and just record the land commit. |
| **5** | `T2-D0-profile` | **serial** | G007/G013 | On the landed simdgroup8 binary: `HAWKING_TCB_TRACE=gpu_prod` generate-greedy n=8 on `uniform-q4-group64-v1`. Write `receipts/q4-dispatch-fold/D0_PROFILE.json` (GPU-us by kernel, encode/submit, lm_head share, expert-wave share). Dirty-box OK. **This decides the next impl, not a human.** |
| **6** | `T2-fold-impl` | **serial** | G007 | Resume `dispatch-fold-20260814-172438`. Order: always D3 (blit-split, bit-id, cheap); then **D0 winner** among D1 (first lift `:1812–1814`) / D4 (lm_head tall GEMV) / D2. Coherence 3/3 after each lever. Stop when a candidate exists for clean measure **or** two levers are in. Do not claim TPS. |
| **7** | `T3-clean-61-and-fold` | **clean-box-exclusive** | G007 | `park_worktrees` until `clean_box_ok`. Zero other GPU/MEMORY. Measure simdgroup8 (`sum(step_us)/n` incl drain, device-resident AR, untraced, n=32) then the fold candidate paired. Seal `CLEAN_61.json` + `CLEAN_FOLD.json`. If fold ≥100 and coherent → also seal Q30-100 (lane 8 becomes a no-op). |
| **8** | `T2b-until-Q30-100` | **serial** then **clean-box-exclusive** to claim | G007 | **Only if lane 7 fold < 100.** Next D-lever D0 still blames. Repeat 6→7 until ≥100 or three consecutive NS (then write the lower bound and continue Q80 — 100 is 20% of roof; three genuine misses would be a surprise, not a reason to invent a new rep). |
| **9** | `T9a-Q30-100-receipt` | **parallel** (DOC; measurement already exclusive) | G007/G010-half | Write the protected Q30 100 receipt. Do **not** flip `MANAGER_ASCENT_TOURNAMENT_ACTIVE`. Append KG/NS to SUBSTRATE. |
| **10** | `T4-q80-q4-pack` | **serial** (MEMORY/IO exclusive — no generate) | G008 | Run the packer. Stream source. ~42 GB out. Sample-organ cosine vs BF16 (expect ~0.99). **Do not overlap any Q30 generate or a second mmap.** |
| **11** | `T5e+T6-q80-first-tokens` | **serial** | G008 | First generate on the **Q4** pack. 3 needles. Success → G008 pass. Failure → U2 per-layer cosine + NS; do not fall back to 0.796 binary. |
| **12** | `T7-q80-fast-kernel` | **serial** | G009 | Port whatever won on Q30 (device-table + simdgroup8 + the D-lever that made 100). Coherence 3/3. **Stop here** if the box is still contended; the next clean-box window is T8 Q80-100 (G009) then T9 both-100 (G010) — same recipe as lanes 7–9, not a new invention. |

**After these 12 the controller does not need a new plan for:** T8 Q80 clean 100 → T9 both-100 + tournament flip (G010) → W6 recursive 125…333 + TG3 (G011) → capability/HCLI/restart → W7 final tournament → W7.ROOF (G013). Those are the same isolation rules and the same `benchmark`/`verify_coherence` calls.

**Idle fill while #4–6 own the GPU:** only #1–3. If #1–3 finish, stop scaffolding (S001). Optional idle after that: U4 doctor6 on one Q3 organ — not a pack, not a generate.

**Park list before lane 7 (mandatory):** every directory under `~/.claude-grok/worktrees/` including `q4-kernel-tps-*` and `dispatch-fold-*` after their binaries are copied to a **stable promotion path the enumerator does not treat as a live lane** (or extend `clean_box_ok` to ignore a single `parked_ok` marker — do this in lane 1, because today **any** dir fails the gate). Dead Aug-10 hawking trees (`lane-fit-20260810-*`, `fit-961mb-*`, …): `git worktree remove` only if clean; otherwise park.

**Performance-truth checklist on every sealed TPS number:** untraced; device-resident AR on; `sum(step_us)/n` including drain; not prefill/fixture/partial/zero-token/fallback/component/speculative-accept; `clean_box_ok==true`; coherence 3/3 still green; Q4 body (not binary, not sub-bit).

---

That is the plan. The 2000-TPS winner-kernel story is the binary roof of a representation GATE 1 is no longer allowed to ship. Phase A’s real first number is **clean-box 61**, then **100 on Q30 Q4**, then **build the Q80 hybrid and its Q4 body**, then **100 both**, then a 333 that Q80 may or may not physically own. Everything else is either idle wrappers or a later gate.
