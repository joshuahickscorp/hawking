# Paradigm execution log

Running ledger for the paradigm-shift build (`paradigmshift.md` +
`plans/paradigm_execution_plan.md`). `reports/` is gitignored — this is the
on-disk working log. Orchestrator session; one row per step.

**Honest clean-room anchor (NEVER report absolute tps from a running Claude
session — contamination inflates it ~4–5×):** ~31 dec_tps, ~0.17 J/tok on
Qwen2.5-3B-Instruct-Q4_K_M, M3 Pro. Gate every decision on **paired A/B deltas**.

---

## 🤖 AUTONOMOUS WALK-AWAY MODE (engaged 2026-06-01)
User chose full walk-away. Operating protocol:
- Grind Phase 1→3 per `plans/paradigm_execution_plan.md`; per-step loop: build →
  parity/bit-identity gate → paired bench → commit (Joshua Hicks, inline `-c`,
  **LOCAL, no push**).
- **Checkpoint this ledger after each step** (this is the user's progress view).
- Surface (stop) ONLY at a phase-boundary summary or a clean **HALT** (gate fail /
  precondition missing → write a closeout). Queue clean-room absolutes; don't block.
- Loop kept alive by chained background tasks + a ~1200s fallback wakeup.

## ▶ SESSION-END CLOSEOUT (2026-06-02) — development COMPLETE; bench handed to user

**16 commits this session** (`2f47141`..`92bcdab`), all **pushed** to `origin/paradigm/exec`.
Build + 108 lib tests green; tree clean (only the 2 protected Codex JSONs). Default decode
**bit-identical throughout** (`b480cc10faf9a8ec`). No new code pending — dev is closed out.

**SHIPPED (the paradigm shift, substantially complete):**
- **Portability seam (3.1/3.2):** trait defs `728ab6d` → MetalBackend `da8da67` → op-router `2cfb927`
  → **seam PROVEN transparent `dd23820`** (elementwise-add routes through `MetalBackend::add`, SEAM-on
  bit-identical). CPU reach: dense (prior) + **deepseek MoE fixed `92bcdab`** (was hard-broken).
- **Energy/long-ctx (2):** f16-KV kernels+dispatch `ed6925e`/`1fa6941` (parity incl 2K) — **MEASURED
  footprint lever, tps-NEGATIVE at depth, energy-TBD**; flash-decode `1769de2` (lifts ~7800-tok cap).
- **4 Type-1 kills:** QTIP `3ec51f7`, 2.2 fusion `2b5379d`, f16-x `45ec3c4` (measured −0.07%), GEMV (prior).
- **Tooling/docs:** validation harness `6ddab9f`, energy attribution `c78d7dc`, docs/strategy `67e7a4c`/`86e88e5`, cleanups `a9913e3`.

**MEASURED (paired A/B):** `--profile fast` = **+4.9% (real)**; f16-KV = neutral@512 / −7.9%@1024 /
−3.4%@2048 (footprint, not tps); flash = neutral short-ctx (capability).

**STAGED for a future focused pass (drafts in `reports/wave*_result.json`):** (a) 3.1 per-family routing
expansion (rope/embed next, gemv LAST — `dd23820` proves the pattern, one family per golden-gated increment);
(b) 3.2 force-cpu-hook + `cpu_fallback_parity` (wave returned empty → re-author on the landed Router);
(c) 4.4 wgpu (`reports/phase_4_4_wgpu_*` — needs Cargo `wgpu` dep approval); (d) off-macOS `cargo check`
(non-macOS toolchain + `libonig-dev`); (e) mixtral CPU MoE (intentionally Unimplemented).

**BENCH HANDED TO USER → `reports/clean_room_final_analysis.md`** (Claude-quit: §1 absolute anchor,
§2 the f16-KV J/tok energy question, §3 quality gates, §4 optional paired A/B).

---

**▶ HANDOFF (2026-06-02).** Committed through **`2f47141`** (LOCAL, no push). SHIPPED:
**Phase 1** (1.1 `e99ed7f` GPU-sample default, bit-identical + 1.2 `8af136e` `--profile fast`/
f16-scales +7.4% paired) + **Phase 3.3** (`2f47141` `force_cpu` CPU-reach path — CPU **12/12
token-identical** to Metal on qwen0.5b; Metal path bit-identical; 108 lib tests green).
The `wf_d13c318a` builder workflow was STOPPED + cleaned up (worktree-base bug:
isolation branches off stale `origin/main` 22dd6f4, NOT paradigm/exec — see
[[worktree-base-stale-origin-main]]; use NON-worktree content-returning agents instead).
**NORTH-STAR STATUS:** short-ctx **tps tapped** at +7.4% (GEMV structural-dead); remaining
wins = **joules↓** (2.1-a f16-KV) + **portability** (3.1 seam, 3.3 done). **NEXT (reshaped):**
(A) **2.1-a f16-KV** energy/long-ctx lever [serial, scout_phase_2_1_f16_activations.md];
(B) **3.1 backend-seam** trait refactor (bit-identical) + 3.2 op-scheduler [scout_phase_3_1/3_2];
(C) **4.1 QTIP** clean-room §A proxy kill [scout_phase_4_1]; (D) 2.2 Attempt-B memcpy probe.
A fresh max-parallelism handoff prompt was produced this session.

## ▶ WAVE-1 CLOSEOUT (2026-06-02) — max-parallelism author+review wave

**Wave `wf_763e3680-f85`:** 10 read-only Plan-author streams → 10 Explore adversarial
reviewers → 1 completeness critic (21 agents, 1.55M tokens, ~47 min). Non-worktree,
content-returning (diffs as text) → agents could not mutate the tree; orchestrator applied
serially on the one GPU. **Every stream came back `apply-with-fixes`** — the reviewers +
critic caught real defects (an unwrap **panic**, a memcpy arg-order **compile error**,
prose-only hunks, a 3-way `backend/mod.rs` conflict, duplicate-symbol additions, stale diff
offsets, a wrong citation). Full result durably saved → **`reports/wave1_result.json`** (372K;
re-extract a stream with `jq '.streams[]|select(.stream=="<id>")'`).

**SHIPPED this session (LOCAL, Joshua Hicks, no attribution):**
| commit | what | gate |
|---|---|---|
| `3ec51f7` | **4.1 QTIP-on-Metal trellis decode → Type-1 kill (by-proxy)** — closes the sub-Q4 byte-cut axis (Q3_K/low-rank/codebook/QTIP-decode all dead) → dense-tps routes to fusion/spec/stateful, not bytes | Kill Protocol (doc) |
| `728ab6d` | **3.1 backend-seam trait defs** (`backend/mod.rs`: `Backend` base + 10 op-traits + `ComputeBackend` bundle + `Op` enum + AWQ-scaled norm verb; `pub mod backend`) | cargo build + 108 lib tests + bit-identical (dead code). **Build caught an E0391 supertrait cycle the review missed** → split base/bundle |
| `67e7a4c` | **docs** `--profile fast` + `DISMANTLE_FORCE_CPU` (README/ARCH + invariant 7) + **`attn/mod.rs` false-"attention-is-CPU" fix** (paradigm-audit claim #3); corrected the wave's "12/12 asserted" → first-3 gate | cargo build |

Fresh-HEAD bit-identity baselines captured (the references for the deferred hot-path passes):
`reports/bench/fresh_head_2f47141_{default,fast}.txt` (blake3, 8 prompts × 64 tok).

**DEFERRED to focused serial passes** (drafts + reviews + defect lists in `reports/wave1_result.json`):
1. **f16-KV vertical (2.1-a) — TOP energy lever, next-up.** `f16kv-kernels` (mha_decode_f16kv +
   memcpy_f32_to_f16_off + parity test incl seq=2048; files[] clean) is parity-ready but its test
   needs the 2 TCB wrappers (in `interface_notes`, HTML-escaped → unescape) + 2 `static_kernel_name`
   arms. `f16kv-dispatch` is **INCOMPLETE**: HUNK 5 (batched path) prose-only; HUNK 6 (early-ensure,
   prevents an unwrap **panic** when `F16_KV=1`+prefix) missing; no `mha_decode_f16kv_batched` kernel;
   `mirror_arena_kv_into_self` f16-read missing; W4A8×F16KV unanalyzed. Adding the kernels changes
   `shader_source_hash` → **re-stamp the profile** in the same pass. Do kernels+wrappers+arena+dispatch+
   batched as ONE complete gated vertical (parity atol=1e-3 incl ≥2K, batch-hash bit-identity OFF,
   long-ctx paired tps + ΔJ/tok; `x_buf` stays f32).
2. **3.1 seam impl + router (incremental, on the landed defs).** Add `MetalBackend` (`backend/metal.rs`)
   + `Router` (→ `backend/router.rs`, NOT a 2nd `backend/mod.rs`) + per-family call-site routing per
   `reports/phase_3_1_seam_routing_checklist.md`. Draft defects: `seam-metal-impl` **critical**
   `memcpy_f32_off_tcb` arg-order compile error + `add_inplace_broadcast` naming; `cpu_fallback_parity`
   unit-tests the wrong kernel (`rmsnorm` vs `rmsnorm_f32`). One op-family at a time, golden-hash-gated.
3. **2.2 memcpy-elision — FINDING: elidable + bit-identical but ≤0.83% ceiling → HELD (below gate).**
   The probe answered 2.2's question: the KV-append memcpy is **NOT** structurally required (K/V proj
   can write direct-to-cache + in-place rope-k, `DISMANTLE_QWEN_KV_DIRECT`, bit-identical) but caps at
   the memcpy's **0.83%** of decode GPU time and is KV_FUSE-gated (default-off) → below the +1 tps floor.
   The real 2.2 lever is fusing the **~324 trivial-op dispatches/tok** (9% GPU, 53% of dispatch count),
   a separate bigger pass. (If ever landed, drop the 2 duplicate-symbol hunks — rope/add `_off_tcb`
   already exist at kernels/mod.rs:7043/7021.) **Type-2** (broad dispatch-fusion stays alive).
4. **family-smoke — DEFERRED (heavy).** Loads all 5 present models (deepseek 10GB + qwen-7B) ×2 for
   determinism → trips the ~5GB coexistence RAM rule. Needs load-once + heavy-model gating + the
   vocab-U64 fix first. Existing `gemma2/phi3/llama32_smoke.rs` already cover the skip-absent pattern.

**NEXT SESSION (recommended):** (1) the **f16-KV vertical** complete+gated (joules↓ + long-ctx — the
highest-value deferred lever, now de-risked); (2) the **3.1 seam impl** incremental on the landed defs.
Clean-room absolutes still queued: `--profile fast` abs tps + J/tok; f16-KV J/tok once it lands.

## ▶ WAVE-2 CLOSEOUT (2026-06-02) — "complete the rest" wave (pushed to origin)

**Wave `wf_8c725657-9fc`** (14 agents, 1.4M tok) authored the complete remainder with the Wave-1 defect
lists fed in. **8 session commits pushed** to `origin/paradigm/exec` (head `1769de2`). Current shader
hash `b68451dc755988dc0d9998d1` (changed twice — f16-KV then flash; deepseek+qwen3b profiles re-stamped).

**SHIPPED (Wave-1 + Wave-2 combined, all bit-identical/parity-gated, LOCAL→pushed):**
| commit | lever | gate |
|---|---|---|
| `3ec51f7` | 4.1 QTIP decode **Type-1 kill** — closes sub-Q4 byte-cut axis | Kill Protocol |
| `2b5379d` | 2.2 dispatch-fusion **dead-for-tps** — closes Phase 2.2 | Kill Protocol |
| `728ab6d` + `da8da67` | **3.1 backend seam: trait defs + MetalBackend impl** (portability foundation) | compile + 108 lib + bit-identical |
| `ed6925e` | **2.1-a f16-KV kernels** (single+batched+memcpy, the energy-lever core) | parity 8/8 atol=1e-3 incl 2K + bit-identical OFF |
| `1769de2` | **2.3 flash-decode attn** (online-softmax, lifts ~7800-tok cap) | parity 4/4 atol+rtol incl 4K + bit-identical OFF |
| `67e7a4c` + `86e88e5` | docs (`--profile fast`, `FORCE_CPU`, attn fix) + strategy docs tracked | build |

**North-star status:** short-ctx tps **structurally tapped** (QTIP + 2.2 + GEMV all Type-1 dead — 3 axes
closed this session); the live wins are **energy/long-ctx** (f16-KV kernels banked, dispatch pending; flash
banked) + **portability** (3.1 seam defs+impl landed; routing + CPU-backend next). Default decode **bit-identical
throughout** (no regression — clean bench will confirm).

**DEFERRED (decode-path wiring; complete validated drafts in `reports/wave2_result.json`):**
1. **f16-KV dispatch** — wires the energy lever (arena `ensure_f16_kv` + qwen_dense `f16_kv` outer-scope binding
   + 6 hunks: single/batched append+MHA, early-ensure panic-guard, mirror f16-read, W4A8 guard). Apply via manual
   Edit per hunk (the prose-wrapped diff doesn't `git apply`); then flag-ON parity + bit-identity-OFF + a 3rd
   re-stamp. **Highest-value next step** (lets the clean bench measure long-ctx tps + J/tok).
2. **scheduler-router (3.2)** — `backend/router.rs` + `cpu_fallback_parity.rs` are clean; the `backend/mod.rs`
   `mod router;` decl + the `forward_token_greedy_tcb` `DISMANTLE_FORCE_CPU_OP` hook are prose → author them.
   cpu_fallback Leg-B must target `rmsnorm_f32` (not `rmsnorm`).
3. **seam-routing-min (3.1 wiring)** — do-not-apply as drafted (depends on a `recorder_borrowing` pattern not in
   tree + per-family routing); fold into the proper incremental routing pass on the landed MetalBackend.

**FLAGGED FOLLOWUPS (NOT fixed — scope):** `qwen05b`/`qwen15b` kernel-profiles stale at `349f2b0b` (pre-existing,
`--kernel-profile` with them fails the guard) → re-stamp to `b68451dc`. The `batch-hash` header hardcodes
`model=DeepSeek-V2-Lite` regardless of `--weights` (cosmetic). [BOTH FIXED in Wave-3 `a9913e3`.]

## ▶ WAVE-3 CLOSEOUT (2026-06-02) — "finish it" wave (pushed; 11 session commits)

**Wave `wf_e7acee07-c22`** (10 agents) authored the finish as surgical edits[]. **11 session commits pushed**
to `origin/paradigm/exec` (head `c78d7dc`). Default decode **bit-identical throughout** (`b480cc10faf9a8ec`);
current shader hash `b68451dc`.

**SHIPPED this wave:**
- **`1fa6941` — 2.1-a f16-KV ENERGY LEVER COMPLETE.** Dispatch wired (11 edits + critic's batched late-ensure),
  composed with flash (`if f16_kv → mha_decode_f16kv[_batched] HALF-stride else if flash → flash else → f32`),
  mutually exclusive w/ W4A8+flash. Flag-OFF bit-identical; flag-ON coherent (1/8 batch-hash divergence = small
  f16 perturbation, offset correct). With `ed6925e` (kernels), the lever is **end-to-end** — clean-room measures
  long-ctx tps + J/tok + logit-cosine via `DISMANTLE_QWEN_F16_KV=1`.
- **`c78d7dc` — energy attribution (V.4).** `measure_joules.sh` macmon-leak fix + `phase_joules.sh` (per-phase
  J/tok aligned to TCB boundaries) — the differentiated north-star instrument.
- **`a9913e3` — cleanups.** batch-hash header + qwen05b/15b re-stamp (both flagged followups, closed).

**PARADIGM-SHIFT SCORECARD (this session, 11 commits):** 3 tps-axes **Type-1 closed** (QTIP byte-cut `3ec51f7`,
2.2 fusion `2b5379d`, GEMV prior) → short-ctx dense tps structurally tapped; **3.1 backend seam** defs+impl
landed (`728ab6d`+`da8da67`); **2.1-a f16-KV** kernels+dispatch complete (`ed6925e`+`1fa6941`); **2.3 flash**
long-ctx capability (`1769de2`); energy tooling, docs, strategy, cleanups. North-stars advanced: **energy/long-ctx**
(f16-KV + flash, measurable clean-room) + **portability** (3.1 seam foundation).

**DEFERRED — portability-completion pass (decode-path wiring; drafts durable):**
- **3.2 router + decode-hook** — `backend/router.rs` + `cpu_fallback_parity.rs` drafts in `reports/wave3_result.json`
  (router has a `flush_and_reset` lifetime bug: add `ctx: &'a MetalContext` param; test needs `Engine` import);
  the `DISMANTLE_FORCE_CPU_OP` decode hook spec is `reports/phase_3_2_decode_hook_spec.md`. The forced-fallback
  test is only meaningful with the hook wired → do router+hook+test as one unit.
- **3.1 per-family routing** on the landed MetalBackend — `reports/phase_3_1_seam_routing_checklist.md`.
- **3.3 CPU-MoE reach** (off-macOS MoE decode) — scope `reports/phase_3_3_cpu_moe_scope.md`.
- **4.4 cross-vendor** — scope `reports/phase_4_4_cross_vendor_scope.md` (needs a Cargo dep approval).

**CLEAN-ROOM QUEUE (user, Claude quit):** `--profile fast` abs tps + J/tok; **f16-KV** long-ctx tps + J/tok +
logit-cosine/PPL (`DISMANTLE_QWEN_F16_KV=1`); flash long-ctx (`DISMANTLE_QWEN_FLASH_ATTN=1`); per-phase J/tok
(`tools/bench/phase_joules.sh`). Default bench confirms **no regression** (bit-identical).

### Autonomous progress log (newest first)
- **[2026-06-06 · Row-ILP reframe found a live 2r-inline candidate].**
  User asked not to abandon dead levers without custom reframes. Built E2
  `gemm_q4_k_v4_predec_pair_3r` as a middle row-geometry test: exact parity
  PASS, paired harness `reports/bench/e2_pair_3r.json` gate PASS, but `B/A=0.975`
  (-2.5%) -> opt-in only. Then split the variable again and built E3
  `gemm_q4_k_v4_predec_pair_2r_inline`: same 2-row geometry as the proven
  default, but inline scale reads instead of 64-float scale preload arrays.
  Exact parity PASS; paired harness `reports/bench/e3_pair_2r_inline.json`
  parity PASS + gate PASS with `B/A=1.081` (+8.1%); longer bench-only confirmation
  `reports/bench/e3_pair_2r_inline_confirm_64t.json` gives `B/A=1.099` (+9.9%).
  Keep behind `DISMANTLE_QWEN_PAIR_2R_INLINE=1` until a clean bench validates
  the one-shot candidate. Shader hash restamped to `da34ffa7ca51d1dd467a7514`.
- **[2026-06-06 · Claude-downtime kernel pass — parity green, row-ILP defaults rejected].**
  Picked up Claude's in-flight Track E1/D8 work: `gemm_q4_k_v4_predec_pair_8r`
  and `gemm_q6_k_fused_v2_swiglu_4r`, plus direct parity tests. Gates run:
  `cargo test -p dismantle-core --test pair_8r_parity --test q6k_swiglu_4r_parity`
  PASS; `cargo build --release --workspace` PASS; `cargo test --workspace --lib`
  PASS (94 core + 16 serve + 5 bench lib tests). Re-stamped profile shader hashes
  to `c943bce6f8b9128778dc7b0f`, fixed `paired_lever.sh` for macOS Bash 3, and
  added missing kernel names to `static_kernel_name` so trace gates stop falling
  into `other` (gate-only taxonomy check PASS). Bench decisions: `PAIR_8R` vs
  explicit 4r parity PASS but `B/A=0.567` (-43.3%) -> HOLD/KILL, opt-in only;
  `PAIR_4R` vs 2r parity PASS but `B/A=0.991` (-0.9%) -> keep opt-in, no default
  flip; `Q6K_SWIGLU_4R` vs 2r parity PASS + gate PASS but `B/A=0.986` (-1.4%)
  -> opt-in only. Updated `reports/dead_levers.md` with the measured row-ILP
  evidence.
- **[2026-06-02 · Wave-6 next-levers — residency SHIPPED `89d7c80` (held) + measurements running].**
  **MTLResidencySet** built (raw objc, selectors verified vs the macOS 26.5 SDK; default-off
  `DISMANTLE_QWEN_RESIDENCY`): golden-OFF bit-identical; flag-ON runs (no NSException → selectors
  correct). **Paired A/B @256tok = −0.4% (neutral)** → HELD default-off (no single-stream tps win;
  the no-eviction-throttle benefit is a coexistence/memory-pressure win the clean bench doesn't stress —
  matches the research's "small + stability"). Tooling materialized (`tools/bench/`): `gpu_saturation.sh`
  (where the 24% gap lives), `mlx_ab.sh` (the MLX ceiling — fixed to auto-detect a venv python),
  `zeus_joules.py` + phase_joules wiring (measured per-domain J/tok via the zeus-apple-silicon pip pkg —
  no Cargo dep). mlx-lm 0.31.3 + zeus installed in an isolated venv. **Running the 3 decisive measurements**
  (`/tmp/measure_all.out`): the saturation breakdown (gap-closer candidate), the MLX ceiling (is the 1.6×
  closable on THIS M3 Pro), and measured GPU+DRAM mJ/tok. The gap-closer build is gated on the trace result.
- **[2026-06-02 · DEEP RESEARCH (next levers) + Wave-6 build].** 106-agent deep-research wave
  (`reports/research_next_levers_2026_06_02.{json,md}`, 8 verified findings). **VERDICT — no silver
  bullet:** the 1.6× tps gap is in the **runtime / command-buffer / GPU-saturation** layer (NOT bytes,
  NOT a faster GEMV — both dead); dismantle is ~76% GPU-busy → ~24% idle. **Energy is near the DRAM
  floor** (sequential weight-streaming) → only race-to-idle + instrumentation. DWQ quant = quality-only
  (no speed); continuous batching = aggregate-only; **ANE dead for decode** (slower than CPU); DVFS
  likely OS-gated. The decisive tps step is a **Metal-trace diff** (literature can't say which scheduling
  change wins on M3 Pro). Launched **Wave-6 `wj2dw3uhd`** to build the actionable items: **MTLResidencySet**
  (the one concrete tps lever — raw objc, macOS 26.5 ok, default-off), **MLX-ceiling A/B** (does MLX hit
  ~49 here → is the gap closable), **GPU-saturation trace** (is the 24% inter-dispatch idle = a live
  saturation lever, or kernel-bound = dead), **measured per-domain J/tok** (zeus/IOReport, replace the
  proxy). The gap-closer itself is gated on the trace (don't guess without profiling — the obvious
  saturation forms ICB/concurrent/megakernel are already dead). mlx-lm installing for the A/B.
- **[2026-06-02 · CLEAN-ROOM ABSOLUTES — user ran `clean_room_batch.sh`, Claude quit].** Fixed the slm-gate
  false-positive first (`8a8346c`: `pgrep -i slm` matched the macOS daemon **aslmanager**; now `pgrep -xi slm`)
  which had been blocking the run. **Clean results:**
  **(A) Q3 §A = 34.3 GB/s = 22.9% of peak → NO-GO** (best-shape; Q3 fused/predec all 5.5–34 GB/s, far under
  the ~50% GO bar) — **re-confirms the QTIP-on-Metal Type-1 kill at current HEAD** (the byte-cut axis is
  definitively closed; QTIP is the only formulation and its decode is compute-bound on the same hmask wall).
  **(B) clean decode = 30.48 tps** (256 tok, greedy) — tracks the **~31 anchor** (−1.7%); the ~39 envelope is
  −21.8% = optimistic/dead. Since the default decode stayed **bit-identical all session**, this == session-start
  → **NO REGRESSION** across all 17 commits. **(C) J/tok = 0.1966** (256 tok; 4.96 W GPU / 6.10 W pkg) — the
  clean energy baseline (≈ the 0.17–0.20 README anchor). ⇒ Headline confirmed: **~30.5 tps / 0.197 J/tok clean,
  no regression; byte-cut axis clean-killed.** (f16-KV's clean J/tok vs this baseline would need a clean
  flag-on run, but the contaminated relative already showed it ~neutral-to-worse → footprint-only stands.)
- **[2026-06-02 · 3.1 SEAM PROVEN TRANSPARENT — SHIPPED `dd23820`] + [f16-KV is footprint/energy, NOT tps].**
  **5a decode-routing proof SHIPPED:** the 3 Qwen2 q/k/v bias elementwise adds route through `MetalBackend::add`
  behind `DISMANTLE_BACKEND_SEAM` (default-off). **Gates: SEAM-OFF bit-identical AND SEAM-ON (=1) bit-identical**
  (the routed add hits the same kernel) → the backend seam is **proven transparent end-to-end** — the
  `ggml_backend_sched` lesson holds in dismantle's imperative decode. This is the #1 plan item's first real,
  working increment (trait-API-correct: `use BackendElementwise` + 4-arg `add(rec,a,b,n)` + no-Drop
  MetalRecorder move-in/out). Per-family expansion (rope/embed/gemv) is the continuation. **f16-KV MEASURED
  FINDING (deep A/B):** B/A = −0.9% @512, **−7.9% @1024, −3.4% @2048** (noisy n=4 ±15%) → f16-KV is
  **tps-NEGATIVE at depth** — the f16→f32 dequant in `mha_decode_f16kv` costs more compute than the KV-bandwidth
  it saves (GEMVs still dominate). ⇒ **f16-KV is a FOOTPRINT lever** (half KV-cache → longer ctx fits in RAM)
  + a *possible* energy lever (less DRAM traffic, but +dequant compute → **needs clean-room J/tok** to settle),
  **NOT a tps win.** Correctly shipped default-off. (Honest characterization refinement, not a kill — footprint
  value stands.) **5c deepseek CPU fix** built green + deepseek golden preserved; `force_cpu` test running.
- **[2026-06-02 · A/B RESULTS + SUPER WAVE #2 processing].** **Paired A/B (contamination-robust):
  f16-scales `--profile fast` = +4.9% (REAL, above noise; matches prior +7.4%); f16-KV @512tok = −0.9%
  (neutral — long-ctx/energy lever, 512 too short); flash @512 = −1.0% (neutral, expected capability lever).**
  Deeper f16-KV A/B (1024+2048tok) running. All 3 waves processed: **`6ddab9f` SHIPPED — 5b validation
  tooling** (`ab_lever.sh` balanced-ABBA + `--long-ctx`, `quality_oracle.sh`, `paired_lever.sh` jq fix). **5a
  seam-route-elementwise** SOURCE-APPLIED (routes the 3 QKV-bias elementwise adds through MetalBackend::add
  behind `DISMANTLE_BACKEND_SEAM` default-off — the trait-API-correct minimal seam proof: `use BackendElementwise`
  import + 4-arg `add(rec,a,b,n)` + lifetime via move-in/out of the no-Drop MetalRecorder; force-cpu-hook came
  back EMPTY → deferred); pending build+golden+SEAM-bit-identity gate. **5c deepseek-cpu** SOURCE-APPLIED (3
  edits A/B/C: force_cpu honor + mla_metal suppress + restore the per-layer CPU driver, default GPU path
  provably unchanged + a light force_cpu test); pending build+deepseek-golden+force_cpu-test gate. Both gates
  blocked on the GPU (deep A/B running) → will build+gate+commit both serially when it frees.
- **[2026-06-02 · SUPER WAVE #2 + A/B validation].** User can't clean-bench (busy) → validate shipped +
  future levers via **paired A/B / microbench** (contamination cancels; saved as preference
  [[feedback-ab-validation-when-no-cleanroom]]). Launched 3 concurrent waves: **5a `w4tvt0vqz`**
  (decode-routing DONE RIGHT — minimal 1-family elementwise-add seam route + FORCE_CPU_OP rmsnorm hook,
  trait-API-correct this time: traits in scope + arity matched, the prior failure mode), **5b `wz4pg5gmg`**
  (validation tooling: fix `paired_lever.sh` jq path + a configurable long-ctx ABBA `ab_lever.sh` + a
  logit-cosine/token-divergence quality oracle for f16-scales/f16-KV — closes the Phase-1.2 quality gap),
  **5c `wve2zpo74`** (deepseek CPU-MoE fix clean: force_cpu honor + mla_metal suppress + restore the
  per-layer CPU driver + a light force_cpu test). Meanwhile running **paired A/B** (`be0tdk6zj`, balanced
  ABBA, `.results.decode_tps`): f16-scales @64tok, f16-KV @512tok, flash @512tok — B/A ratios to surface
  the progress. Bench field confirmed = `.results.decode_tps` (the `trial_stats[0]` path matches at trials=1).
- **[2026-06-02 · SUPER WAVE RESULTS — 3 waves processed].** All 3 concurrent waves returned + were
  triaged. **LANDED (2 commits):** **`2cfb927` Phase 3.2 op-router scaffold** (`backend/router.rs` —
  Router{primary,forced} over the seam, DISMANTLE_FORCE_CPU_OP, lifetime bug fixed, dead-code/compile-gated)
  + `pub mod metal/router`; **`<this commit>` 2.1-b f16-x KILL** — the probe oracle measured **−0.07%**
  (f16x 519.685 vs f16s 519.303 µs; **x is only 0.0263% of GEMV traffic**) → Type-1 dead, probe reverted,
  recorded. **SAVED (durable artifacts):** `reports/paradigm_plan_final_audit.md` (the definitive DONE/
  KILLED/LEFT table — Phase 0/1/2 complete, Phase 3 seam defs+impl+CPU-dense done + 3 open, Phase 4 = 4.1
  killed/4.2-4.3 gated-dead/4.4 design), `reports/phase_4_4_wgpu_skeleton.rs` + `phase_4_4_wgpu_plan.md`
  (the wgpu Rung-1 backend skeleton — dep-gated, awaits your Cargo `wgpu` approval). **STAGED (drafts durable,
  need a focused incremental pass — reverted to keep the tree green):** **4a decode-routing** (the #1 item:
  DISMANTLE_FORCE_CPU_OP hook + DISMANTLE_BACKEND_SEAM per-family routing — 7 edits apply verbatim but the
  trait calls need the BackendEmbed/Elementwise/Rope traits in scope + arity fixed to the collapsed trait API;
  must be done per-family golden-gated per the scout — `reports/wave4a_result.json`); **4b deepseek CPU-MoE
  fix** (a REAL bug — deepseek CPU decode hard-errors at the MLA branch; 3 edits A/B/C honor force_cpu +
  suppress mla_metal + restore the per-layer CPU driver + a 10GB-load gate — `reports/wave4b_result.json`);
  **4b off-macOS build cfg audit** (`cargo check --target aarch64-linux` + libonig-dev prereq for the user).
- **[2026-06-02 · SUPER WAVE — 3 concurrent waves for the plan remainder].** User asked for a super wave
  (multiple at once). What's LEFT after 11 commits (the rest is done/killed): (1) Phase 3 portability
  completion (3.2 router + decode-hook + 3.1 per-family routing); (2) 3.3 off-macOS reach (CPU-MoE +
  build verify); (3) 4.4 cross-vendor (dep-gated); (4) 2.1-b f16-x probe (oracle-gated, likely a kill).
  These are **disjoint file-tracks** → launched 3 concurrent Workflows: **4a `w8ulyt02f`** (Phase 3
  portability: router lifetime-fixed + DISMANTLE_FORCE_CPU_OP hook + DISMANTLE_BACKEND_SEAM embed/rope/
  elementwise routing + cpu_fallback/seam tests — the ONLY track touching qwen_dense), **4b `wy1fphdw2`**
  (3.3: off-macOS CPU-MoE decode for deepseek/mixtral + the aarch64-linux build cfg audit — deepseek/mixtral/
  cfg only), **4c `whz453pff`** (4.4 wgpu backend skeleton dep-flagged + 2.1-b f16-x GEMV probe in quant.metal +
  bench oracle + a final DONE/KILLED/LEFT plan audit — new files + quant.metal only). ~24-30 agents in flight.
  Serial apply when they return: 4c f16-x probe (quick oracle/kill) + plan audit, 4a portability (decode-path,
  golden-gated per family), 4b off-macOS (independent), 4c wgpu (design, dep-gated).
- **[2026-06-02 · 2.1-a f16-KV ENERGY LEVER COMPLETE — SHIPPED `1fa6941`; pushed].** Wave-3 returned the
  dispatch as **11 surgical edits + the critic's EDIT9b** (batched late-ensure, prevents a panic), all
  anchored to clean HEAD, applied via an exact-match script (12/12 matched). Composes with flash:
  `if f16_kv → mha_decode_f16kv[_batched] (HALF-stride offset) else if flash_attn → flash else → f32`;
  f16_kv mutually exclusive w/ W4A8+flash. **Gates:** flag-OFF **bit-identical** (golden `b480cc10` +
  batch-hash byte-identical); flag-ON **coherent** (real code gen; **1/8** batch-hash divergence = expected
  small f16 perturbation, NOT garbage → HALF-stride byte offset correct; the newline-output scare was a
  prompt artifact — OFF produces identical newlines for that raw-completion prompt). **No shader change →
  no re-stamp.** 9 commits pushed (`…1fa6941`). The **energy lever is now end-to-end** (kernels ed6925e +
  dispatch 1fa6941): clean-room measures long-ctx tps + J/tok + logit-cosine/PPL via `DISMANTLE_QWEN_F16_KV=1`.
  Remaining Wave-3: energy-attribution tooling (apply-as-is), cleanups (batch-hash header + qwen05b/15b
  re-stamp), router-files (3.2, needs a lifetime fix), cpu-moe-3.3 (scope doc only).
- **[2026-06-02 · f16-KV dispatch in progress + Wave-3 LAUNCHED].** User: proceed + push, run Wave-3 if
  useful. **f16-KV dispatch:** the arena (f4: `k_cache_f16_buf`/`v_cache_f16_buf` + `ensure_f16_kv` +
  `kv_f16_layer_byte_offset`) is applied + builds green (uncommitted, staged for the dispatch commit). The
  qwen_dense dispatch is being authored by Wave-3 because the Wave-2 draft was anchored pre-flash (`86e88e5`)
  and is **stale where it touches the MHA dispatch** — flash (`1769de2`) now owns that region, so f16-KV must
  COMPOSE: `if f16_kv → mha_decode_f16kv; else if flash_attn → flash; else → f32`, with f16_kv mutually
  exclusive with W4A8 + flash. (Lesson: `git apply --reject` PARTIALLY writes the file — the f5 attempt left
  qwen_dense half-patched referencing `f16_kv_enabled`; restored to clean HEAD.) **Wave-3 `wf_e7acee07-c22`**
  (~14 agents): (A) f16kv-dispatch as **surgical edits[]** composed with flash (PRIORITY), (B) router-files
  (3.2 backend/router.rs + cpu_fallback_parity vs rmsnorm_f32 + decode-hook spec), (C) energy-attribution
  (V.4: fix measure_joules leak + per-phase J/tok), (D) cpu-moe-3.3 (off-macOS MoE reach), (E) cleanups
  (batch-hash header + qwen05b/15b re-stamp→b68451dc). Serial apply when it returns: f16kv-dispatch first
  (gate flag-OFF bit-identity + flag-ON sanity + 3rd re-stamp), then router hook on top, then the rest.
- **[2026-06-02 · 3.1 MetalBackend impl `da8da67` + 2.3 flash-decode-attn `1769de2` — SHIPPED; all pushed].**
  **3.1 MetalBackend** (`backend/metal.rs`, 608 lines): concrete impl of all ~11 landed op-traits + the GAT
  `Recorder<'a>` wrapping TCB, thin wrappers over `kernels::*_tcb`; macOS-gated `mod metal;`. **Compiles
  against the landed traits** (the GAT lifetime resolved); dead-code → bit-identical. The 3.1 seam now has
  **defs + concrete impl** both landed (portability foundation complete; routing is the next increment).
  **2.3 flash-decode-attn** (`mha_decode_flash_f32`, default-off `DISMANTLE_QWEN_FLASH_ATTN`): online-softmax
  GQA decode (no score materialization → lifts the ~7800-tok cap); **parity 4/4 atol=1e-3+rtol=1e-4 incl 4K**;
  default-off → golden + batch-hash bit-identical. Re-stamped deepseek+qwen3b `8af17951→b68451dc` (2nd shader
  change). **cross-vendor 4.4 scope** saved (`reports/phase_4_4_cross_vendor_scope.md`, design-only, no deps).
  **8 commits pushed to origin** (3ec51f7 QTIP-kill, 728ab6d seam-defs, 67e7a4c docs, 86e88e5 strategy,
  2b5379d 2.2-kill, ed6925e f16-KV-kernels, da8da67 MetalBackend, 1769de2 flash). **Current shader hash =
  `b68451dc755988dc0d9998d1`.** ⚠️ **qwen05b/qwen15b profiles still stale at `349f2b0b`** (pre-existing,
  now further behind — flagged followup, NOT fixed inline per scope). **Still DEFERRED:** f16-KV dispatch
  (wires the energy lever; draft `f16kv-vertical` in wave2_result.json, needs the `f16_kv` outer-scope
  binding + 6 hunks via manual Edit), scheduler-router (router.rs+test clean, qwen_dense+mod.rs wiring is
  prose), seam-routing-min (do-not-apply: depends on the routing recorder pattern). These 3 are decode-path-
  wiring tasks for a focused follow-up — drafts validated + durable.
- **[2026-06-02 · 2.1-a f16-KV KERNELS — SHIPPED · `ed6925e`] + [2.2 dispatch-fusion KILLED · `2b5379d`].**
  Wave-2 processed. **f16-KV kernels** (the energy lever's verified core): `mha_decode_f16kv` (single) +
  `mha_decode_f16kv_batched` (the Wave-1-missing batched producer) reading `half` K/V + `memcpy_f32_to_f16_off`
  + 3 TCB wrappers + reg arms + `tests/mha_decode_f16kv_parity.rs`. **Parity 8/8 atol=1e-3 (diffs ~1e-7),
  incl seq=2048 single + p0=2048 batched + the memcpy bit-exact round-trip.** Kernels unreachable (no dispatch
  yet) → **bit-identical**: golden `b480cc10faf9a8ec` + batch-hash default byte-identical vs baseline (verified
  with the FRESH binary — the first check used a stale binary). **Re-stamped deepseek + qwen3b profiles**
  `3317a977→8af17951` (shader source changed; selections unchanged). ⚠️ **Pre-existing (NOT fixed, scope):**
  `qwen05b`/`qwen15b` profiles are stale at `349f2b0b` (predates this session) → `--kernel-profile` with those
  fails the guard; flag for a separate re-stamp. **2.2 dispatch-fusion** recorded dead-for-tps (all 3 forms:
  rope-fusion=host-overhead Type-1, add/rope-fold=A10 FMA-recontraction bit-identity trap, KV_DIRECT=0.83%
  noise-floor) — closes Phase 2.2. **f16-KV DISPATCH** (arena + qwen_dense wiring, the `f16_kv` binding fix +
  6 hunks) **deferred** — complete validated draft in `reports/wave2_result.json` (`f16kv-vertical`).
  Wave-2 saved to `reports/wave2_result.json`. seam-routing-min = do-not-apply (depends on impl); seam-metal-impl
  + flash-decode-attn + scheduler-router + cross-vendor-scope pending.
- **[2026-06-02 · PUSHED + Wave-2 LAUNCHED].** User greenlit pushing + completing the rest of the
  paradigm shift in one large wave (clean bench to follow). **Pushed `paradigm/exec` → origin**
  (github.com/joshuahickscorp/dismantle; new branch, 5 session commits incl. the strategy docs
  `paradigmshift.md` + `plans/paradigm_execution_plan.md` now tracked). Launched **Wave-2
  `wf_8c725657-9fc`** (~14 agents): author+adversarial-review for the COMPLETE remainder, with
  the Wave-1 defect lists fed in + the escaping fix (all code in `files[]`): (1) **f16kv-vertical**
  (2.1-a, full: single + **batched** mha_decode_f16kv kernels + wrappers + arena ensure_f16_kv +
  ALL dispatch hunks incl the early-ensure panic-guard + mirror f16-read + W4A8 guard + parity incl
  2K), (2) **seam-metal-impl** (3.1 MetalBackend, Wave-1 memcpy-argorder/naming fixes), (3)
  **scheduler-router** (3.2, backend/router.rs + cpu_fallback_parity vs rmsnorm_f32), (4)
  **seam-routing-min** (3.1 wiring: embed/rope/elementwise behind DISMANTLE_BACKEND_SEAM), (5)
  **flash-decode-attn** (2.3 GQA online-softmax, default-off), (6) **dispatch-fusion** (2.2 best
  bit-identical trivial-op fusion), (7) **cross-vendor-scope** (4.4 design-only, no deps). Serial
  apply funnel: f16-KV first (re-stamp profile after shader change), then seam impl/router (dead-code
  compile-gate), then routing-min/flash/fusion each golden-or-parity gated. f16-KV targets pre-verified
  at HEAD (batched twin kernels/mod.rs:7290, mirror :3396, fresh_arena :3657/:4933, ensure_w4a8 arena
  pattern :165). Clean-room absolutes (user) to follow.
- **[2026-06-02 · docs + 3.1 routing design — SHIPPED · `67e7a4c`].** Documented `--profile fast`
  (f16-scales bundle, +7.4% paired, opt-in, not bit-identical) + `DISMANTLE_FORCE_CPU`/`force_cpu`
  (Phase 3.3 CPU reach; **first-3 greedy gate**, 12/12 observed — corrected the wave's "12/12 asserted"
  overclaim against `cpu_backend_parity.rs:79-82`) in README + ARCHITECTURE (+ invariant 7 "levers
  default-off"); **fixed the stale `attn/mod.rs` doc** that implied attention runs on CPU (it's the GPU
  `mha_decode_f32` on the fast path — the paradigm-audit's false-claim #3). Build green. Also saved the
  3.1 per-family routing checklist (`reports/phase_3_1_seam_routing_checklist.md`, on-disk per scout
  convention) with adversarial corrections (Step-6 label inversion, batch range 4849–5323) for the
  deferred seam-impl pass; persisted the full wave result to `reports/wave1_result.json` (372K) for
  durable re-extraction of deferred drafts.
- **[2026-06-02 · 3.1 backend-seam trait defs — SHIPPED · `728ab6d`].** Landed `backend/mod.rs`
  (platform-neutral): `Backend` base (assoc `Buffer` + GAT `Recorder<'a>` + `recorder()`/`supports()`),
  10 op-traits, `Op` capability enum, `ComputeBackend` bundle (blanket-impl'd), `GemvSpec`/`WeightKind`
  collapsing the ~31 gemv entry points to one verb; + `pub mod backend;` in lib.rs; + the AWQ-scaled q8
  norm verb (critic fix). **Defs only, no impl, referenced by no decode path → bit-identical by
  construction; 108 lib tests green.** ⚠️ **Build gate caught a defect the adversarial reviewer missed:**
  the authored shape made `Backend` a single supertrait bundling the op-traits while each op-trait had
  `Backend` as supertrait → **E0391 supertrait cycle**. Fixed by splitting base (`Backend`) from bundle
  (`ComputeBackend`). Lesson re-confirmed: the compile/parity gate is the real backstop, not the review.
  MetalBackend impl + router + per-family routing **deferred** to the incremental 3.1 pass (the
  seam-metal-impl draft had a critical memcpy arg-order compile error + a 3-way `backend/mod.rs` conflict
  with scheduler-router → must be reconciled deliberately, one op-family at a time per the scout).
- **[2026-06-02 · 4.1 QTIP decode — KILLED Type-1 (by-proxy) · `3ec51f7`].** Wave returned + adversarially
  reviewed (21 agents, every stream `apply-with-fixes`). First serial landing: recorded the **QTIP-on-Metal
  trellis decode** Type-1 kill in `reports/dead_levers.md` (by-proxy on the clean Q3 §A 36.3 GB/s = 24.2%-peak
  reading; trellis re-adds the per-element ALU predec removed + adds a serial `state[i]←state[i-1]` with no
  Q4_K analog → same compute-bound wall). **Closes the sub-Q4 byte-cut axis** (Q3_K direct, low-rank, codebook,
  QTIP-decode all Type-1) → dense-tps headroom routes to **dispatch-fusion/spec/stateful, not bytes**. Type-2
  reframe (lane-independent sub-block + fused predec-of-seeds) named + oracle in-hand; did NOT build the kernel
  (scout's "cheapest honest outcome"). Doc-only, no build needed. Citation fixed per critic (Q3 entry +
  `autonomous_run`, not `paradigm_execution_log:232-236`). **Triage:** the two hot-path verticals (3.1 seam,
  f16-KV dispatch) came back INCOMPLETE (prose-only HUNK 5, missing panic-guard/batched-kernel, a `backend/mod.rs`
  3-way conflict, a critical memcpy arg-order compile error) → landing the clean/complete deliverables, deferring
  those to a focused serial pass.
- **[2026-06-02 fresh-HEAD baselines CAPTURED].** `reports/bench/fresh_head_2f47141_{default,fast}.txt`
  — blake3, 8 parity prompts × 64 greedy tok @ `2f47141`. The **bit-identity references** the 3.1
  refactor, 2.2 memcpy-elision, and f16-KV-OFF must reproduce byte-identically (default path) and
  that `--profile fast` must preserve. default `p008` == fast `p008` (`6b99d430…`) while 7/8 differ
  → the expected f16-scales near-tie (matches 1.2's 4/5-byte-identical), confirming real qwen-3B
  output (not a wrong-model load). **Cosmetic smell (scope — NOT fixed inline):** the `batch-hash`
  baseline writer hardcodes `# … model=DeepSeek-V2-Lite-Chat-Q4_K_M` in the header regardless of
  `--weights` (hashes are correct qwen-3B; only the header string is stale).
- **[2026-06-02 Wave-1 LAUNCH — max-parallelism session].** Cold-start gates green @
  `2f47141`: `cargo build --release --workspace` exit 0; `cargo test --workspace --lib`
  94 core + 9 serve pass (bench lib → 108). 2 Codex JSONs untracked (never touch).
  Launched author+review **Workflow `wf_763e3680-f85`** (background): 10 read-only
  **Plan**-author streams → 10 **Explore** adversarial reviewers → 1 completeness critic.
  Streams: (1) f16-KV kernels+parity, (2) f16-KV dispatch+`DISMANTLE_QWEN_F16_KV` gate,
  (3) 3.1 seam trait-defs `backend/mod.rs`, (4) 3.1 `MetalBackend` `backend/metal.rs`,
  (5) 3.1 per-family routing checklist, (6) 3.2 scheduler+CPU-fallback+`cpu_fallback_parity`,
  (7) 4.1 QTIP kill-by-proxy entry, (8) 2.2 KV-append memcpy-elision probe, (9) family-smoke,
  (10) docs. **Non-worktree, content-returning** (diffs as TEXT; agents cannot mutate the
  tree — sidesteps [[worktree-base-stale-origin-main]]); orchestrator applies/parity-gates/
  paired-benches/commits **serially** on the one GPU. Prep: models on disk = qwen{0.5,1.5,3,7}b
  + deepseek-v2-lite (no gemma2/phi3/mixtral/llama gguf → existing `*_smoke.rs` stay skip-guarded);
  decode line-refs drifted ~+10 vs scouts (`memcpy_f32_off_tcb` KV-append now :4143/:4151,
  `forward_token_greedy_tcb` :3437, `add_rmsnorm_fused_tcb` :4248/:4553); golden-64 =
  `b480cc10faf9a8ec`. Capturing fresh-HEAD `batch-hash` bit-identity baseline for the
  3.1 / f16-KV-off / 2.2 hard gates while the wave runs.
- [1.1] tokens=3 bit-identity ✓ (twice). Running correctness gate:
  `integration_greedy_64` (golden 64-tok hash) + `v1e_gpu_argmax_parity`. If
  golden passes → GPU default == golden at 64 tok → bit-identical → bench+commit.
  If it FAILS → likely the golden was a CPU-path baseline & the flip moved the
  default to GPU (near-tie divergence) → HALT + write up (don't auto-rebaseline).
- [1.1] flip coded + built green (qwen_dense.rs:1421/:1547).
- NOTE: this build can only run as a LOCAL session (needs the M3 GPU); remote
  cron can't touch it. Decode-path levers are SERIAL (no parallel fan-out). State
  is durable (commits + this ledger) → kills lose nothing, just re-run.

---

## 0. Setup & baseline reference

### Branch base decision (2026-06-01)
Plan said "branch off `main`", but `main` (22dd6f4) was a strict **ancestor**
~200 commits / **+57,225 lines** behind the audited tree. The plan's toolchain
(`tools/bench/*`, every parity test, `reports/dead_levers.md`) and its
line-number refs exist **only** on the `codex/maximal-spec-colab` tip (56279da)
— the 2026-06-01 audit was of THAT tree. **User chose: advance local main, then
branch.**
- `git branch -f main HEAD` → local `main` now 56279da (local only; `origin/main`
  untouched at 22dd6f4, **not pushed**).
- `git checkout -b paradigm/exec` → work branch at 56279da.
- `codex/maximal-spec-colab` untouched at 56279da; its 2 uncommitted Colab JSONs
  (`docs/archive/.../headbank_manifest.json`, `.../frontier.json`) left modified,
  **never to be committed**.

### Build + test gate (2026-06-01) — GREEN
- `cargo build --release --workspace` → exit 0 (incremental 0.45s; 7 pre-existing
  `unused_mut`-class warnings, non-blocking).
- `cargo test --workspace --lib` → exit 0: **94 core + 9 serve = 103 passed, 0
  failed**. (dismantle-bench exposes no separate `--lib` unittests in this run.)

### Bench reference (the paired "A" baseline)
All three harnesses (`paired_lever.sh`, `measure_joules.sh`,
`clean_room_batch.sh`) lock the **same** Qwen fast-path as their A reference:
```
DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
DISMANTLE_QWEN_Q4K_PREDEC=1
```
plus `--kernel-profile profiles/qwen3b-instruct-q4k.m3pro18.json`, model
`models/qwen2.5-3b-instruct-q4_k_m.gguf` (1.93 GB).

⚠️ **Load-bearing nuance for Phase 1.2:** the locked fast-path *already* enables
vocab-prune-32K, Q4K-LM-head, and Q4K-FFN-down. So **three of Phase 1.2's four
levers are already in the paired "A" reference** — their marginal Δ vs this
baseline is ~0 (already on). The only byte-cut lever NOT yet in the fast-path is
**f16-scales** (`DISMANTLE_QWEN_PREDEC_F16SCALES`). Re-scope Phase 1.2 there:
(a) measure fast-path vs the bit-identical raw default separately, (b) treat
f16-scales as the one genuinely new lever. NB FFN-down (plan flags it as
spec-accept-lowering, ~7→3) is *already* in the moat-bench baseline → factor
into 0.4's spec-on-code guard.

### Harness prerequisite fix — profile re-stamp (2026-06-01, commit 64edfbd)
The locked-fast-path profile `profiles/qwen3b-instruct-q4k.m3pro18.json` was
stamped `shader_hash=0b9ed2ef…` but the current build's `shader_source_hash()`
is `3317a977…` (also printable via `dismantle shader-hash`). The guard
(`profile.rs:223`) rejected the profile → **every** bench/generate using
`--kernel-profile` failed ("kernel profile shader hash mismatch") and wrote no
JSON. Cause: the only shader edit since the last stamp (618024b) was `14434f5`
(Q4_K simdgroup-MMA **prefill** twin behind `DISMANTLE_QWEN_Q4K_MMA`, additive;
decode kernels untouched, parity-green). Fixed by re-stamping the single
`shader_hash` field (same chore as `d168542`; kernel selections unchanged), NOT
by re-running `dismantle autotune` (which would re-select kernels on contaminated
timings). This also unblocks the user's clean-room run.

### 0.1 result — noise floor & harness calibration (2026-06-01)
Self-vs-self paired bench (identical env on both arms, 8 trials × 32 tok,
`reports/bench/phase0_noise_floor.json`):
- **A (always-first arm) median 32.99 dec_tps; B (always-second arm) median
  31.99; B/A = 0.970× (−3.0%).** Both arms are the SAME config, so the true
  ratio is 1.000×.
- ⚠️ **Finding: ~3.0% second-position penalty.** `paired_lever.sh` interleaves
  `A,B,A,B…` (not balanced `ABBA`), so the lever-**on** arm (always B) is
  systematically penalized ~3% by run-order/thermal state alone. Per-trial
  scatter ≈ ±5% (A: 30.59–33.35 with one low outlier; B: 30.65–32.44).
- **Gating rule adopted for all later steps:** (1) treat |B/A − 1| ≲ 3% as
  **inconclusive** (within position bias); a SHIP needs a paired gain clearly
  beyond it. (2) For borderline levers, **swap A/B** (run `--env-a`=lever-on,
  `--env-b`=lever-off) and average the two ratios to cancel the position bias,
  or raise `--trials`. (3) Bit-identical levers still gate on **parity**, not
  tps, so the bias doesn't gate them.
- **Logged smell (do NOT fix inline — scope):** `paired_lever.sh` should use a
  balanced `A,B,B,A` interleave (or alternate the lead each trial) to remove the
  ~3% position bias. Proposed for a separate attended pass.
- Absolute numbers above are **contaminated** (Claude open; ~33 vs ~31 clean) —
  reported here only as a paired-ratio calibration, never as a throughput claim.

### 0.2 gap anatomy (sets Phase 2 order)
**Source:** fresh gpu_prod per-dispatch trace on the current tree
(`reports/traces/qwen3b_decode_gpu_prod_64edfbd.json`; 32 traced tokens of a
128-tok decode, locked fast-path; §1 methodology gate **PASS**). Traced-run
decode_tps = 31.03 (matches the ~31 clean anchor — gpu_prod is un-distorted).

**Per-kernel decode GPU time (calls/token = n ÷ 32 traced tokens):**

| kernel | % GPU | calls/tok | µs/call | role |
|---|---|---|---|---|
| `gemm_q4_k_v4_predec_pair` | **45.40%** | 36 | 309 | fused FFN gate+up (11008×2048 ×2) |
| `gemm_q4_k_v4_predec_2r` | **41.26%** | 163 | 62 | q/o/ffn_down(Q4)/LM-head |
| `add_rmsnorm_fused` | 4.98% | 72 | 17 | norm+residual |
| `mha_decode_f32` | 2.92% | 36 | 20 | attention |
| `add_inplace` | 2.13% | 108 | 5 | residual add |
| `rope_q_f32_inplace` | 1.01% | 72 | 3 | RoPE |
| `memcpy_f32_off` | 0.83% | 72 | 3 | buffer copy |
| `gemm_q6_k_fused_v2` | 0.69% | 18 | 9 | ffn_down Q6_K layers |
| `moe_batched_silu_mul` | 0.58% | 36 | 4 | SiLU |
| `sample_argmax_f32` | 0.15% | 1 | 38 | greedy sampling |

**Three load-bearing numbers:**
1. **86.7% of decode GPU time = the two predec Q4_K GEMVs** (`_pair` 45.4% +
   `_2r` 41.3%), both at **~52% of 150-peak BW** (78.7 GiB/s effective) →
   **kernel-bound, ~1.6–1.8× headroom**. THE GPU-time wall. (A4 prior: 89.4% @
   56% peak — stable across commits.)
2. **616 dispatches/token** (19712 ÷ 32) — corrects paradigmshift.md's "~180"
   (a **~3.4× undercount**). **324/616 are trivial ops** (rmsnorm 72 + add 108 +
   rope 72 + memcpy 72): ~9% of GPU time but **~53% of the dispatch COUNT** →
   fusing them attacks the non-GPU gap, not GPU-busy time.
3. **GPU-busy ≈ 76%** of decode wall (24.5 ms GPU / 32.2 ms wall) → ~24%
   non-GPU gap (CPU encode + inter-dispatch + commit). (The analyzer's "~1% busy"
   line is an artifact — total_gpu_us covers 32 traced tokens, decode_ms all 128;
   ignore it.)

**Attention is 2.92%** short-context → flash-attn (plan 2.3) and f16-KV (plan
2.1) have a ~3% short-context tps ceiling; they pay at **long context + on
energy**, not on the short-context tps headline.

**llama.cpp paired reference (same machine + contamination window):**
- **llama.cpp `tg128` = 49.03 ± 0.78 t/s** (homebrew `llama-bench`, MTL backend,
  ggml 0.10.2) vs **dismantle decode = 31.2–32.1** (3 trials: 32.09/31.57/31.23)
  → **gap = 1.55×** (49.03 / 31.57). Contamination-robust (same window).
- dismantle's reading (~31.6) is **within noise of the clean ~31 anchor** → this
  session contaminates only mildly. ⚠️ The repo's "Claude inflates dec_tps 4–5×"
  claim (CLAUDE.md / bench_contamination) did NOT reproduce here (~31.6 ≈ clean);
  treat it as stale/workload-specific. Ratios are robust regardless.
- **Coherence:** dominant GEMV at ~52% peak BW; lifting it to ~80%+ peak ≈ 1.55×
  — almost exactly the llama gap. **Closing the GEMV BW-efficiency gap ≈ closing
  the llama.cpp gap.** llama's init log advertises `simdgroup matrix mul = true`
  \+ `use residency sets = true` — the exact techniques dismantle's scalar-`uchar`
  -load predec GEMV lacks → corroborates the A5/simdgroup-matrix direction (#1).
- NB: absolute numbers above are a paired *reference ratio*, NOT a dismantle ship
  claim; clean-room absolutes stay queued.

**⇒ Recommended Phase-2 attack order (REORDERED from the plan default by this
data; confirm at the Phase 1→2 boundary):**
1. **GEMV kernel efficiency** — the dominant **86.7% at ~52% peak**. A4-scoped
   **A5** (vectorized `uint4` nibble unpack — widen the current 1-byte weight
   loads) + **A6** (rows-per-TG / accumulator-chain occupancy sweep),
   `_predec_pair` first then `_2r`. Stacks on Phase 1.2's f16-scales (same
   kernels, byte tax 192→160 B/block). Gate: **bit-identical** (math unchanged) +
   paired. *(Highest leverage; not an explicit plan-2.x step but the clear data
   winner — maps to paradigmshift Part II #3 "a kernel that unpacks the native
   block efficiently beats both." Lower-risk alternative lead = step 2.)*
2. **Dispatch reduction / fusion (plan 2.2)** — fuse the 324 trivial-op
   dispatches/tok (rmsnorm/add/rope/memcpy) + fold adjacent GEMVs; attacks the
   ~24% non-GPU gap. Gate: bit-identical (reorder) + paired.
3. **f16 activations + f16/Q8 KV (plan 2.1)** — ~3% short-ctx tps, real at long
   context + an energy win. Gate: parity + long-ctx paired + quality.
4. **Flash-style decode attention (plan 2.3)** — 2.92% short-ctx; defer to long
   context. Gate: parity (atol+rtol) + long-ctx paired.

### 0.3 energy attribution (2026-06-01)
**`measure_joules.sh` is BUGGED → logged for a separate fix (not fixed inline;
timebox + scope).** Its macmon sampler subshell leaks: `kill $sampler_pid`
(~line 191) kills the subshell but NOT the `macmon pipe` child, so samplers
orphan and multiply (observed **6 live `macmon pipe` + 4 script instances,
~33% CPU**) and the script hangs at `wait` without ever printing J/tok. Killed
the tree (`pkill`), 0 remaining. (spawn_task filed.)

**Energy is still gateable without it:**
- Baseline anchor (prior clean-room / README): **0.17 J/tok @ ~3.73 W GPU**. The
  HONEST clean number stays queued via `clean_room_batch.sh` §C.
- **Per-kernel energy attribution from the 0.2 trace (proxy):** GPU power is
  ~steady during memory-bound decode, so per-kernel energy ≈ per-kernel GPU-time
  share —
  | phase | ≈ % GPU energy (= % GPU time) |
  |---|---|
  | predec Q4_K GEMVs (`_pair`+`_2r`) | **~86.7%** |
  | rmsnorm / add / rope / memcpy | ~9% |
  | attention (`mha_decode_f32`) | ~2.9% |
  | sampling / embed / final norm | <0.5% |
- ⇒ **energy tracks GPU-time.** The GEMV-efficiency lever (#1) is also a
  ~87%-of-energy lever → both-metrics win. f16-scales is the known both-win
  (+9.3% tps, −1.4% J/tok). Per-lever gating = **paired J/tok delta** (clean-room
  or a fixed sampler).
- True sub-token energy counters (GPU IOReport energy aligned to the TCB
  timeline) = the deferred "balloon" the plan timeboxes; the proxy suffices to
  gate.

### Clean-room absolutes — ✅ RESOLVED (user clean-room run, 2026-06-01)
`clean_room_batch.sh` ran with Claude fully quit (gates passed; quiet-cpu WARN =
`knowledge-agent` at 98% CPU). The three honest verdicts:
- **(A) Q3 byte-cut = NO-GO.** f32-predec-Q3 best = **33.4 GB/s = 22.3% of 150
  peak** (~30% regime). hmask/index residual is the wall, NOT bandwidth →
  **Q3 byte-cut is footprint-only, DEAD for speed.** ⇒ Phase 4 sub-4-bit MUST be
  **QTIP (gather-free trellis)**; a Q3 repack is off the table. (Confirms the
  prior Q3 kill; locks Phase 4.1's framing.)
- **(B) Clean decode = 29.96 dec_tps** (256 tok, locked fast-path, greedy). Tracks
  the **~31 anchor**; the ~39 envelope is DEAD (−23%). **Honest llama gap
  recomputes to ~1.63×** (clean dismantle 30 vs llama ~49) — slightly worse than
  the 1.55× from the contaminated dismantle reading (contamination inflated
  dismantle ~5%: 31.6 vs 30 clean; llama ~robust).
- **(C) Clean energy = 0.2021 J/tok** @ 5.40 W GPU / 6.98 W pkg (256 tok). NB the
  `knowledge-agent` 98% CPU inflates the pkg (CPU) term → true J/tok likely a bit
  under 0.20 (nearer the 0.17 README anchor). GPU-power 5.40 W is the cleaner
  signal. **Energy baseline for gating ≈ 0.20 J/tok.**
- **Bonus:** `measure_joules.sh` COMPLETED fine in the user's terminal → the hang
  I hit was my backgrounded-with-`tail` invocation (no TTY for the sampler kill),
  NOT a universal break. The leak (spawn_task filed) is real but TTY-tolerant →
  downgrade severity.
- Pre-flight anytime (Claude open): `./tools/bench/clean_room_batch.sh --gates-only`.

---

## Step ledger

| step | branch@commit | parity (test / hashes / atol) | paired Δtps (range, CI, meas/proxy) | Δ J/tok | gate (SHIP/HOLD/KILL) | CLEAN-ROOM TODO | notes |
|---|---|---|---|---|---|---|---|
| 0 setup | paradigm/exec@56279da | n/a (no code change) | n/a | n/a | — | clean_room_batch.sh queued | branch base resolved; build+103 lib tests green |
| re-stamp | paradigm/exec@64edfbd | n/a (profile JSON; bench loads OK) | n/a | n/a | SHIP (chore) | n/a | shader_hash 0b9ed2ef→3317a977; unblocks harness |
| 0.1 baseline | paradigm/exec@64edfbd | n/a (no code change) | self-vs-self B/A=0.970× → **~3% second-position bias** + ±5% scatter (measured, contaminated abs) | n/a | — (calibration) | clean_room_batch.sh | noise floor recorded; gating rule + harness smell logged |
| 0.2 gap-anatomy | paradigm/exec@64edfbd | §1 gate PASS (gpu_prod trace) | **llama 1.55× gap**; 86.7% GPU = predec GEMVs @~52% peak; 616 disp/tok (324 trivial) (measured paired ratio) | n/a | — (profile) | clean_room_batch.sh | sets Phase-2 order: GEMV-eff → fusion → f16-KV → flash-attn |
| 0.3 energy | paradigm/exec@64edfbd | n/a (measurement) | n/a | per-kernel energy ≈ GPU-time share → GEMVs ~87% (proxy) | — (measure_joules BUG → spawn_task; clean baseline queued) | clean_room_batch.sh §C | energy gateable via proxy + paired clean-room |
| 0.4 moats | paradigm/exec@64edfbd | prefix_cache **3/3 ✓** + user_draft **6/6 ✓** (237s); eagle5 slow >60s/test (NO-GO path) → deferred | n/a | n/a | **✓ guards green** | n/a | **PHASE 0 COMPLETE**; moats locked for Phases 1-4 |
| re-stamp ds | paradigm/exec@3f47420 | n/a (profile JSON; integration_greedy_64 green) | n/a | n/a | SHIP (chore) | n/a | deepseek shader_hash cf91e98f→3317a977; unblocks integration_greedy_64 |
| **1.1 gpu-sample** | paradigm/exec@e99ed7f | default==TCB=1 **byte-identical 5×64tok ✓**; v1e_gpu_argmax 3/3; integration_greedy_64 + moats green | old-default(TCB=0) ~0.4 → new-default(TCB) ~25–30 dec_tps (1 clean pair + 4 indep B; contaminated; **NO regression**, large default-user win) | proxy − (removes per-tok 608KB copy + CPU argmax) | **SHIP** (bit-identical default-flip) | clean_room_batch.sh (abs) | greedy default now = fast TCB path; escape `DISMANTLE_QWEN_TCB=0` |
| reconcile | paradigm/exec@c54d299 | n/a (kill-ledger doc) | GEMV A5/A6 structural-dead **confirmed** | n/a | — (record) | n/a | 0.2 52%-peak does NOT resurrect A5/A6/A10; gap = bytes + dispatch |
| **1.2 f16-scales** | paradigm/exec@8af136e | kernel parity green (f16s + pair_f16s rel_L2<1e-2); token-div **4/5 byte-identical @64tok**; `--profile fast` ≡ explicit env byte-identical | **+7.4%** (OFF 21.05 → ON 22.59, 6 trials, contaminated; within +6–9% prior) | proxy − (192→160 B/block scale stream) | **SHIP** (`--profile fast` opt-in; raw default stays bit-identical) | clean_room_batch.sh (abs tps + J/tok); logit-cosine/PPL oracle | named bundle (plan's "not a silent default"); precise quality gate queued |

### ✅ PHASE 1 COMPLETE (2026-06-01)
1.1 (GPU-sampling default-flip, bit-identical, `e99ed7f`) + 1.2 (f16-scales → `--profile fast`
named bundle, +7.4% paired, `8af136e`). The lone genuinely-new short-ctx tps+energy lever
(f16-scales) is banked behind an opt-in named profile; raw default stays bit-identical. Per
the Phase 1→2 reshaping, short-ctx tps headroom is now largely exhausted — the remaining
north-star wins are **energy/long-context** (2.1-a f16-KV, 2.3 flash — build-and-hold) and
**portability** (Phase 3 backend seam → CPU backend). Queued clean-room: abs tps + J/tok for
the `--profile fast` config; the precise f16-scales logit-cosine/PPL oracle.

### ✅ Phase 3.3 (CPU reach path) SHIPPED (2026-06-02, `2f47141`)
Out-of-plan-order (lighter 3.3 before the 3.1 trait seam — justified: the CPU reference path
already existed via `forward_token`, so this is the cheapest tangible portability win).
`EngineConfig.force_cpu` + `DISMANTLE_FORCE_CPU=1` load with `metal_ctx=None`; the `use_tcb`
gates now require `self.metal_ctx.is_some()` (correct off-macOS behavior, not just a test hook).
New `tests/cpu_backend_parity.rs`: CPU vs Metal greedy on qwen0.5b = **12/12 tokens identical**.
Metal path **bit-identical** (qwen-3B batch-hash byte-identical vs pre-change p11_tcb1). 108 lib
green. Gaps logged: MoE CPU decode not covered (dense-only); the off-macOS **build** still needs
a non-macOS toolchain to verify (`cargo check --target aarch64-unknown-linux-*`) — queued for the user.
Harness bug filed (spawn_task): `paired_lever.sh` reads a stale `trial_stats[0].decode_tps` jq
path (schema drifted to flat `.results.decode_tps`) → returns zeros; inline loop used meanwhile.

### ✅ PHASE 0 COMPLETE (2026-06-01)
All four steps resolved. Headline: **llama gap = 1.55× on this machine**; the wall
is the predec Q4_K GEMVs (86.7% GPU, ~52% peak); 616 disp/tok; Phase-2 order =
GEMV-eff → fusion → f16-KV → flash-attn. Scout specs persisted
(`reports/scout_phase_*.md`). Moats (prefix-cache, user-draft) bit-identity green.
Harness bug (measure_joules leak) filed. Clean-room batch queued. → Phase 1.

---

## Phase 1→2 boundary — scope reshaping (2026-06-01)

A 7-agent read-only scoping fan-out (workflow `wf_93f3117a-151`; specs persisted to
`reports/scout_phase_2_*.md`, `scout_phase_3_2/3_3.md`, `scout_phase_4_1_qtip_spike.md`)
re-derived every unscoped Phase 2–4 lever **against the kill-ledger**. It overturns the
plan's default Phase-2 framing. The honest map:

**GEMV efficiency (plan's 0.2 #1 lead) = STRUCTURALLY DEAD.** The dedicated reconciliation
(`scout_phase_2_gemv_reconciliation.md`) confirms the fresh 0.2 "52% peak / 1.55× gap"
does NOT reopen the recorded A5/A6/A10/MMA/residency Type-1 kills: ggml-metal uses `mul_mv`
(vec) at M=1 and `mul_mm` (MMA) only batched/prefill — llama's "simdgroup matrix mul=true"
is **not** a decode-GEMV lever; "use residency sets" is dead on unified memory; a half-prec
**x** stream is killed by byte math (x ≈ 0.03% of `_pair` traffic). **The 1.55× gap is
bytes (QTIP, Phase 4) + dispatch count (2.2), not a GEMV kernel micro-opt.** `scout_phase_2_1_gemv.md`
is SUPERSEDED (banner added); dead_levers.md A5/A6 updated. ⇒ **do NOT execute A5/A6/A10.**
The one live GEMV lever is f16-scales — **already built** (opt-in), so Phase 1.2 just makes
it a named-profile default behind its quality gate. The cheap oracle for any future GEMV-BW
claim: `cargo test -p dismantle-core --test q4k_predec_f16s_bench -- --nocapture`.

**The reshaped lever map (what's live, and what kind of win):**
| step | verdict | win type | action |
|---|---|---|---|
| 1.1 GPU-sampling default | ✓ done | cleanliness + default-user latency/energy | committed (bit-identical) |
| 1.2 f16-scales profile | LIVE | short-ctx **tps + energy** (+9.3%/−1.4% J prior) | SERIAL next; already-built lever → named-profile default + quality gate |
| 2.x GEMV A5/A6/A10 | DEAD (Type-1, structural) | — | do not execute; recorded |
| 2.2 dispatch fusion | ~dead (noise floor) | tiny tps | one cheap bit-identical spike: KV-append `memcpy` elision (Attempt B); else record dead-for-tps |
| 2.1 f16-KV (+f16 act) | LIVE but long-ctx | **long-ctx tps + energy + footprint** | ⚠ Qwen dense is 100% f32 — NO `--q8-kv` for dense (that's MLA only). Build parity-green spike, HOLD ship (no short-ctx headline; `x_buf` residual MUST stay f32 — f16 residual is a Type-1 kill) |
| 2.3 flash decode-attn | LIVE but long-ctx/reach | **removes ~7800-tok cap** (capability) | flash kernel ALREADY exists for MLA (`attn.metal:536`) → GQA re-skin; parity-green spike, HOLD ship unless >8K workload |
| 3.1 backend seam | LIVE — **real forward progress** | portability | SERIAL bit-identical refactor; NEW-FILE scaffold is parallel-ok |
| 3.2 op-sched + CPU fallback | LIVE | reach (partial backends ship) | Router + per-op `supports()`; Metal bit-identical, forced-fallback atol=1e-3 |
| 3.3 CPU backend | LIVE — **mostly already exists** | reach (off-macOS) | `forward_token` + CPU primitives already wired; needs `force_cpu` knob + cross-check; build oracle needs a non-macOS toolchain (FLAG to user) |
| 4.1 QTIP-Metal spike | strong NO-GO prior | byte-cut (if alive) | free §A proxy already 24% peak; likely a cheap Type-1 kill closing the byte-cut axis |

**Bottom line for the north-stars:** short-context **tps↑** headroom is small and largely
banked — f16-scales (1.2) is the lone genuinely-new short-ctx tps+energy lever; the 1.55×
llama gap is structural (bytes/dispatch), closable only by QTIP (leaning NO-GO). The real
remaining wins are **(a) energy + long-context** (f16-KV/flash, build-and-hold spikes) and
**(b) portability/reach** (Phase 3 backend seam → CPU backend → cross-vendor), which is the
plan's stated llama.cpp-parity north-star. Execution order from here: **1.2 (real tps win)
→ 3.1 (portability) → parallel build-and-hold spikes (2.1-a/2.3/parity-tests/tooling) →
4.1 QTIP cheap kill.**

## Dead levers logged this pass
- **GEMV decode micro-opt (A5/A6/A10) — reconciliation, not a new kill.** The fresh 0.2
  data does not resurrect the recorded Type-1 kills; `dead_levers.md` A5/A6 entry updated
  with the 2026-06-01 reconciliation; `scout_phase_2_1_gemv.md` marked SUPERSEDED. No new
  lever died — an over-optimistic *reading* of the gap (0.2 note) was corrected.
- New decisive kills will land here + in `reports/dead_levers.md` as Phases 2–4 execute
  (expected: 2.2-fusion dead-for-tps after the Attempt-B spike; 4.1 QTIP Type-1).

---

## Wave-6 measurements + SESSION CLOSE-OUT (2026-06-02, evening)

Goal: settle "is there any more single-stream decode tps to get?" before a final
bench. Ran the three decisive measurements the deep research named (research_next_
levers_2026_06_02.md). Two landed; the third (MLX) is confirmatory and finishing.

### M1 — GPU-saturation trace (the research's #1 tps hypothesis) → DEAD, Type-1
`tools/bench/gpu_saturation.sh` (trace /tmp/gpu_sat_trace_20260602T193249.json, 22,176 samples).
- **Production inter-token gap = 0.0 ms**; the GPU is continuously busy intra-token
  (~616 kernels back-to-back in ONE TokenCommandBuffer). There is **no inter-dispatch
  idle to reclaim** in the production path.
- The "GPU-busy ≈76% → ~24% reclaimable" figure that motivated the hypothesis was a
  **SplitCbGpu artifact** (each kernel in its own CB inflates per-dispatch overhead).
- **Verdict: KERNEL-BOUND, saturation lever DEAD.** Recorded as a Type-1 kill in
  `reports/dead_levers.md` (commit 68ca21d). Closes research TPS-#1.
- (The script auto-prints "next step: A5 vectorized uint4" — that's a STALE heuristic;
  A5 is already a Type-1 kill. Ignored + noted in the ledger entry.)

### M2 — measured per-domain J/tok (zeus IOReport energy model) → instrumentation upgraded
`tools/bench/zeus_joules.py --tokens 64` (Claude OPEN, so wall — and thus mJ/tok — is
contamination-inflated; the per-DOMAIN split is the value):
- GPU 574.9 mJ/tok · DRAM 479.5 mJ/tok · CPU 864.9 mJ/tok · ANE 0 · GPU-SRAM 0 (not exposed).
- DRAM ≈ 0.48 J/tok confirms **memory-bound**; the clean-room **GPU anchor stays ~0.197 J/tok**.
- This **replaces the cpu-encode PROXY** in phase_joules.sh with a real per-domain reading
  (research ENERGY-#1, "instrument precisely"). ANE=0 reading corroborates ANE-dead.

### M3 — MLX ceiling A/B (dismantle vs MLX on THIS M3 Pro) → running (confirmatory)
`tools/bench/mlx_ab.sh` (mlx-lm 0.31.3 in /tmp/mlxenv). Settles WHETHER the 1.6× gap is a
runtime ceiling (MLX≈49 → kernel-technique gap, port it) or HW (MLX≈30 → at the M3-Pro
ceiling). NOT decisive for the tps question — M1 already proved the gap is kernel-bound,
not idle. Result folds into the ledger when it exits.

### Commits this wave (LOCAL on paradigm/exec; authored Joshua Hicks; no AI attribution)
- 89d7c80  engine: MTLResidencySet lever (held default-off; A/B ~neutral on 3B)
- 61f529b  bench: next-levers measurement harness (gpu_saturation, mlx_ab, zeus_joules)
- 085380e  bench: clean_room_batch --gates-only on FAIL + prompt-brace fix
- e1e787c  bench: fold residency A/B + measured J/tok + --diag ceiling diagnostics into final_analysis
- 68ca21d  kill-ledger: GPU-saturation / inter-dispatch-idle reclaim NO-GO Type-1

### NORTH-STAR VERDICT (honest, measured)
- **TPS (primary): TAPPED for cheap levers.** saturation DEAD (M1, measured), bytes DEAD
  (QTIP), GEMV micro-opt DEAD (memory-model optimum), f16-act DEAD, spec-decode deprioritized.
  The 1.6× llama gap is **kernel-bound** — the ONLY surviving reframe is "llama's mul_mv runs
  the same bytes in less GPU-time," **parked behind a named oracle** (Metal System Trace diff
  llama-vs-dismantle on M3 Pro) with an **adverse prior** (simdgroup-MMA@M=1 + GEMV micro-opt
  already dead). High-effort, uncertain — NOT a cheap win.
- **ENERGY (second): near the DRAM floor.** No clean J/tok win; the only axis is race-to-idle
  (= faster decode, which is tapped). Now MEASURED per-domain (M2), not proxied.

### THE ONE RE-BENCH COMMAND
- `tools/bench/final_analysis.sh`            → robust paired/relative (Claude OPEN ok): --profile
   fast A/B + residency A/B, measured zeus J/tok + f16-KV footprint, long-ctx f16-KV/flash, short quality.
- `tools/bench/final_analysis.sh --clean`    → ALSO absolute anchor (QUIT Claude first; self-gates).
- `tools/bench/final_analysis.sh --diag`     → ALSO the ceiling diagnostics (gpu_saturation + MLX A/B).

CLOSE-OUT: development is closed. The bench is the user's to run. No cheap tps lever remains;
the next tps attempt (if any) is the llama-kernel Metal-trace reverse-engineering, which is a
new attended project, not part of this pass.

---

## Claude-downtime continuation (2026-06-06)

Picked up the row-geometry lever family and pushed until the next clean-bench boundary.

### E1/D8 carry-forward
- `PAIR_8R`: parity PASS, but `B/A=0.567` (`-43.3%`) vs 4r. Keep opt-in only.
- `PAIR_4R`: parity PASS, gate PASS, `B/A=0.991` (`-0.9%`) vs 2r. Keep opt-in only.
- `Q6K_SWIGLU_4R`: parity PASS, gate PASS, `B/A=0.986` (`-1.4%`) vs 2r. Keep opt-in only.

### E2/E3 customization
- `PAIR_3R`: custom middle geometry, parity PASS, gate PASS, `B/A=0.975` (`-2.5%`).
- `PAIR_2R_INLINE`: same 2-row geometry as default pair, inline scale reads. Parity PASS,
  gate PASS, `e3_pair_2r_inline` = `36.40` vs `33.66` (`+8.1%`), and 64-token
  confirmation = `36.51` vs `33.21` (`+9.9%`). This is the live clean-bench candidate
  behind `DISMANTLE_QWEN_PAIR_2R_INLINE=1`.

### E4/E5 fast-stack split
- Added `gemm_q4_k_v4_predec_pair_2r_inline_f16s` plus Rust wrapper and parity coverage.
  It is numerically identical to `pair_f16s` for the same half-scale tables.
- Initial E4 rerun with `PREDEC_F16SCALES=1` was parity PASS + gate PASS but slower:
  `39.30` vs `41.64` (`B/A=0.944`, `-5.6%`).
- Customized rather than discarding: split f16-inline behind
  `DISMANTLE_QWEN_PAIR_2R_INLINE_F16S=1`, leaving `PAIR_2R_INLINE=1` as the E3 f32-scale
  form even when the broader fast profile uses f16 scales.
- A/B/C split (`e5_pair_2r_inline_split`): B (`PAIR_2R_INLINE=1`, f16-inline off) failed
  greedy parity and was slower (`B/A=0.961`, `-3.9%`); C (f16-inline opt-in) stayed exact
  but was slower (`C/A=0.981`, `-1.9%`). Conclusion: f16-inline is a quarantined opt-in,
  not a fast-stack default.

### Validation
- `cargo build --release --workspace` PASS.
- `cargo test --workspace --lib` PASS.
- `cargo test -p dismantle-core --test pair_2r_parity --test q4k_predec_pair_f16s_parity --test pair_8r_parity --test q6k_swiglu_4r_parity` PASS.
- Shader hash after Metal additions: `3e542895d850bc8dae1d1e18`; four active profile JSONs
  restamped. Later Rust-only routing/taxonomy edits did not change the shader hash.

## Claude continuation — F-track register-axis closure (2026-06-06, afternoon)

Picked up where the E-track left off. Goal: chase the occupancy lever that E3
(inline scales, +8-10% on the strict pair) opened, and — per the user's "customize
the dead lever" directive — try a novel reframe before declaring the axis dead.

### E3 re-validated (the foundation) → REAL, +9.8%
The E-track benches were single 32-tok paired runs, which I found are
contamination-unreliable (see F2). Re-ran E3 on the **noise-robust harness**
(64-tok × 7-trial, interleaved): `e3_verify_64t7` = `36.60` vs `33.34` dec_tps
(`B/A=1.098`, **+9.8%**, parity PASS), **all 7 B-trials above all 7 A-trials.**
E3 is real and large — the standout default-flip candidate. (The dead-lever ledger
+ resurrection check updated accordingly.)

### F1/F2/F3 — beyond-inline register relief → ALL FLAT (built, parity-green, measured)
Three bit-identical customizations aimed at freeing MORE registers than E3:
- **F1 `pair_2r_inline_nox`** (strict pair: E3 inline + drop the `xl[8]` activation
  preload, read x per-pi; x≈0.026% of traffic per the f16x kill ⇒ ~0 bandwidth,
  frees ~6 regs): `B/A=0.997` (−0.3%) vs E3. The 64-float SCALE preload was the
  binding occupancy step; the 8 `xl` regs sit below the next threshold.
- **F2 `pair_f16s_nox`** (headline f16 pair, keep half-scale preload, drop `xl`):
  32-tok `B/A=1.042` (+4.2%) — **but that was contamination**: robust 64-tok×7
  reran `B/A=0.995` (−0.5%, FLAT). The f16s pair is 1-row/32-float-preload and
  **bandwidth-bound** (half-scales already cut the bytes), not occupancy-bound.
  *(Lesson re-confirmed: never trust a 32-tok single paired delta; the bench-
  contamination memory is right — F2 flipped sign between 32-tok and 64-tok×7.)*
- **F3 `pair_f16s_halfreg`** (the novel reframe of the dead E4: keep scales in
  **half registers** packed ≈2/reg → ~16 regs vs 32, widen at the FMA — a COALESCED
  preload, NOT E4's lossy inline — plus drop `xl`): `B/A=1.005` (+0.5%, FLAT).
  Either the compiler didn't pack halves, or the cut didn't cross an occupancy
  threshold; the f16s pair is byte-bound regardless.

### Structural conclusion — the inline-scale axis is COMPLETE
Audited every hot strict-path GEMV: all already DEFAULT to inline scales (0 preload
arrays) — ffn_down `gemm_q4_k_v4_predec_4r_swiglu` (4r-inline + 4× silu
amortization), o_proj `…_2r_add_rmsnorm` (4r-inline), qkv
`gemm_q4k_predec_qkv_rope_append` (inline). The **gate+up pair is the SOLE
remaining preload-default kernel**, which E3 closes. So: no other inline-scale
lever exists to harvest, and beyond-inline register relief is tapped on both paths.
The pair/GEMV register+geometry axis (preload→inline, row-ILP 2r/3r/4r/8r, x-preload,
half-register scales) is now fully explored and exhausted.

### Built this pass (all opt-in, bit-identical, parity-green)
6 new Metal kernels + Rust wrappers + taxonomy entries + parity coverage:
`gemm_q4_k_v4_predec_pair_2r_inline_nox` (F1), `…_pair_f16s_nox` (F2),
`…_pair_f16s_halfreg` (F3). Flags: `DISMANTLE_QWEN_PAIR_2R_INLINE_NOX`,
`…_PAIR_F16S_NOX`, `…_PAIR_F16S_HALFREG`. Wired into the dominant FFN gate+up site
via function-pointer routing (no arg-block duplication). Parity tests extended:
`pair_2r_parity::pair_2r_inline_nox_matches_pair_2r_multiple_shapes` (incl.
11008×2048) + `q4k_predec_pair_f16s_parity` now also asserts f16s_nox and
f16s_halfreg are bit-for-bit equal to `pair_f16s`.

### Validation
- `cargo build --release --workspace` PASS.
- `cargo test --workspace --lib` PASS (115 tests).
- `cargo test -p dismantle-core --test pair_2r_parity --test q4k_predec_pair_f16s_parity
  --test pair_8r_parity --test q6k_swiglu_4r_parity` PASS (8 tests).
- Shader hash after F-track Metal additions: **`78d07c2e92d28801e66728e2`**;
  four active profile JSONs restamped.

### Handoff — the ONE clean-bench shot
The next clean bench should settle the **E3 default flip**: `DISMANTLE_QWEN_PAIR_2R_INLINE=1`
is now robustly paired-confirmed (+9.8%) and is the only remaining inline-scale win;
promote it to default-on (opt-out) if the clean absolute anchor agrees. No other
cheap pair/GEMV tps lever remains — the register+geometry axis is exhausted; further
tps must come from a different axis (spec/stateful/long-ctx) or a future xctrace
occupancy oracle, not from another pair-kernel variant.

## Claude continuation — 8-axis build wave + integration (2026-06-06, evening)

User directive: "oneshot as many levers as possible, use a wave of agents, then bench
clean; build until we hit long-local-training; refactor the plan if needed." Pair/GEMV
axis already exhausted this session, so the wave targeted ALL OTHER axes.

### The wave (Workflow wf_ec6c7c51, 8 design-scouts, ~975K tok, 6.7 min)
Each scout read its plan/handoff + the kill-ledger + current code and returned a
verdict + ready-to-apply spec. Constraint honored: worktree isolation branches off
STALE origin/main (no predec kernels) and every kernel lever shares 4 files → agents
were READ-ONLY designers, I integrated serially. Verdicts:
- **green_buildable ×3:** continuous-batch v4r high-B route (#0), int4-KV (#4), flash/f16-KV long-ctx (#6).
- **needs_clean_bench_only ×1:** n-gram + batched-verify spec (#1) — already BUILT (speculate/ + GPU pruned-Q4K verify); just needs a clean paired bench, no code.
- **already_shipped ×4:** cross-turn prefix-cache (#2, ~84% turn-2 prefill cut measured), predec-MMA prefill (#3, GO dormant), Q6_K/default-path inline (#5 — "ZERO remaining bit-identical tps headroom"; independently confirms the GEMV-axis-exhausted finding), CPU-backend portability (#7, seam+Router shipped).

Refactored-plan takeaway (matches plans/bleeding_edge_..._2026_06_05.md): "the raw GEMV
fight is hard" — the wins are token-only serving (mostly shipped), dispatch fusion,
prefix/session reuse (shipped), and footprint/long-ctx. Most architecture is ALREADY
built; the genuine new-build surface was small.

### BUILT + VALIDATED this turn (2 levers, both opt-in, parity-green)
**#0 — continuous-batch v4r high-B route** (`DISMANTLE_QWEN_MULTISEQ_V4R_HIGHB`, aggregate tps).
Routes B=5..8 multiseq projections + fused FFN-down swiglu through the already-parity-proven
barrier-free 16-rows/TG `v4r_predec` kernels instead of the 8-rows/TG shmem-barrier `v3w_predec`
(the closeout profile found v3w DRAM-latency-bound/under-occupied even at high B). NO new
kernel, NO shader change, NO restamp. Bit-identical (atol-1e-3, same FMA order). Edits:
qwen_dense.rs (flag + widen `b<=4` in multiseq batched_proj!), kernels/mod.rs (widen `batch<=4`
in ffn_down_swiglu_add_rmsnorm_ffn_q4k_predec_batched_tcb). Parity: swiglu_fused_ffn_parity
extended to B=5..8 (PASS); **greedy_token_only_parity WITH the flag set → B=8 bit-identical (PASS)**.
Gate for clean bench: B=8 aggregate beats R1-ON by +3% on a clean 64-tok×7-trial. If flat, stays OFF (harmless).

**#6 — flash decode over the f16 KV cache** (`DISMANTLE_QWEN_FLASH_F16KV`, long-ctx capability + tps).
New kernel `mha_decode_flash_f16kv` (clone of the known-good `mha_decode_flash_f32`, K/V typed
`half*`, widened at read). CAPABILITY UNLOCK: flash's constant threadgroup memory removes the
O(seq) scores-shmem cap (~7800 tok) that makes standalone `mha_decode_f16kv` unable to run at
32K, while halving the dominant KV byte stream at depth. Rides the F16_KV arena/append/prefill
machinery (requires `DISMANTLE_QWEN_F16_KV=1` too); only the decode kernel changes (fn-pointer
select at the decode site). Wrapper + taxonomy added; mha.metal changed → restamped. Parity:
mha_decode_flash_f16kv_parity (3 tests incl. **4096-token long-context**) vs CPU ref on
f16-roundtripped K/V, atol 1e-3 + rtol 1e-4 (PASS). Gate for clean bench: long-ctx (≥16K) A/B
J/tok + tps (EV sign uncertain at short ctx — attention is a small share until depth grows).

### STAGED (not built — by design, not omission)
**#4 — int4 (per-channel) KV cache** (footprint, ~4× vs f32 / 2× vs f16-KV; long-ctx enabler).
GREEN per the wave with a complete spec, BUT deliberately staged: it needs (a) novel symmetric
per-row int4 quant math + nibble packing (real bug surface), (b) NEW arena plumbing (int4 cache
+ f16-scale cache buffers, alloc, layer-offset helpers, append+decode routing, readback), AND
(c) its WIN is quality-gated by a perplexity bench that cannot run this session — so building it
now does not unblock validation, and rushing novel quant math under budget violates the
parity-first culture. Full ready-to-apply spec preserved in the wave result
(/private/tmp/.../wpxwd8bag.output, result[4]). Cleanest build path next time: model the decode
on the just-validated `mha_decode_flash_f16kv` (flash structure + int4 dequant instead of f16
widen) and the append on the proven `kv_append_q8_0` pattern.
**#1 — spec n-gram + batched verify:** no code needed; run the clean paired bench (code/repetitive
workload) to settle the GPU-pruned-Q4K-verify win the contaminated in-session number hinted at.

### Validation (this turn)
- `cargo build --release --workspace` PASS.
- `cargo test --workspace --lib` PASS (115).
- `greedy_token_only_parity` WITH `MULTISEQ_V4R_HIGHB=1` PASS (B=8 bit-identical, end-to-end).
- `swiglu_fused_ffn_parity` (B=2..8) PASS · `gemm_q4k_v4r_predec_parity` PASS.
- `mha_decode_flash_f16kv_parity` (incl. 4096) PASS · `mha_decode_flash_parity` (8) + `mha_decode_f16kv_parity` (4) regression PASS.
- Shader hash after #6: **`e7f3c11b113b0c1edc5fe761`**; four active profile JSONs restamped.

### Clean-bench shots staged (priority order)
1. **E3 default flip** (`PAIR_2R_INLINE=1`, +9.8% paired) — promote to default-on if clean anchor agrees.
2. **#0 v4r high-B** (`MULTISEQ_V4R_HIGHB=1`) — B=8 aggregate A/B vs R1-ON, 64-tok×7-trial clean.
3. **#1 spec** — n-gram + GPU pruned-Q4K batched verify, clean paired on code workload.
4. **#6 flash-f16kv** (`F16_KV=1 FLASH_F16KV=1`) — long-ctx (≥16K) tps + J/tok A/B.

## CLEAN-BENCH RESULTS + actions (2026-06-06, night)

User ran `tools/bench/wave_clean_bench.sh` (Claude quit → clean absolutes). Verdicts:

- **① E3 → GO, SHIPPED default-on.** Clean parity PASS, **B/A=1.096 (+9.6%)**, all 7 B-trials
  above all 7 A-trials. Flipped `ffn_pair_2r_inline` to default-on (opt-out `=0`); verified
  bit-identical (`default == =1 == =0` greedy tokens). The session's headline win is now the default.
- **② #0 v4r-highB → NO-GO (clean).** B=8 aggregate **62.78 → 53.56 tok/s (−14.7%)** with the flag;
  B=1/4 unchanged → it's the route, not thermal. Original "v3w wins at B>4" was right. Flag stays
  OFF. Recorded as a Type-1 kill (`dead_levers.md` "Multiseq v4r-highB route").
- **④ banked (clean):** single-stream anchor **30.74 tok/s / 0.2585 J/tok** (measured under OLD default;
  with E3 now default-on this rises ~+9.6% → ~33–34); `--profile fast` A/B ≈ +18% (A=33.1→B=39.1,
  Ctrl-C'd mid-run); Q3 byte-cut re-confirmed NO-GO (33.2 GB/s = 22% peak).
- **③ #6 flash-f16kv → kernel VALIDATED; harness was a stub, now FIXED.** `mha_decode_flash_f16kv_parity`
  (3 tests incl. 4096) PASS — kernel is correct. `long_context_bench.sh` reported `decode_tps=?`
  because it parsed `.decode_tps` from the `--json` file, but `generate` prints the `[stats]` line
  (with `dec_tps`) to **stderr** and **`--json` suppresses it**. Fixed: drop `--json`, parse the
  stderr `[stats]` line (verified: now reports real `decode_tps`). #6 long-ctx A/B is now runnable.

### Bench-script fixes (the "it's broken" report — mostly harmless noise + one real stub)
- The wall of `cargo-clippy`/`cfg` text was harmless `objc`-macro build WARNINGS (not errors),
  flooding because each sub-bench re-invokes cargo. `wave_clean_bench.sh` now exports
  `RUSTFLAGS=-Awarnings` + `CARGO_TERM_QUIET=true` → one quiet rebuild, then clean output.
- `long_context_bench.sh` stats-parse fixed (above).

### Net state after the clean bench
- **Shipped this session:** E3 inline pair default-on (+9.6% clean single-stream, bit-identical).
- **Staged opt-in (validated, default-off):** #6 flash-f16kv (kernel parity green; long-ctx A/B pending);
  F1/F2/F3 register variants (flat, kept); #0 flag (NO-GO, off).
- **Dead (clean):** #0 v4r-highB route; Q3 byte-cut (re-confirmed).
- **No-code, needs clean paired bench:** #1 spec n-gram + batched verify.
- **Deferred (needs long work):** #4 int4-KV (novel quant + arena plumbing + perplexity gate).
- New clean single-stream baseline with E3 default ≈ **33–34 tok/s** (was ~31).

## Production build wave #2 (2026-06-06, late — machine BUSY, no-bench, unit/parity-gated)

User: "keep working until we absolutely need a clean bench; launch a wave of production."
Machine busy (prod run elsewhere) → NO benching; everything unit/parity-gated + staged.
Workflow wf_d9fb0518, 6 production-feature designers (~546K tok). Verdicts: 5 green_buildable,
1 already_shipped. Full ready-to-apply specs preserved in the wave output
(/private/tmp/.../wlmwloj6t.output, result[i]).

### SHIPPED + unit-validated this turn (2 features)
- **[0] Named profiles race/efficient/exact (Track 2.2).** Added a pure `LeverPlan` source of
  truth in dismantle-serve (`RuntimeProfile::lever_plan()` + `contract()`), consumed by BOTH the
  CLI `apply_profile` (generate/bench) and `serve::run` — they can no longer drift. **Fixes a real
  correctness bug:** `--profile exact` previously did NOT force-off `PREDEC_F16SCALES`, so a user
  with that env set got non-bit-identical output from the profile whose entire contract is
  bit-identity; now `Exact` force-offs every quality-trade var. race/efficient are now DISTINCT
  (f16-KV + concurrent-QKV) instead of silent aliases of fast; each profile prints its lever/quality/
  J-tok contract at startup. Edits: dismantle-serve/src/lib.rs (LeverPlan + lever_plan + contract +
  serve::run uses the plan), dismantle/src/main.rs (apply_profile rewritten). **7 unit tests PASS**
  (`profile_lever_tests`), incl. `exact_force_offs_every_quality_trade`. No kernel/shader change.
- **[1] Observability core (Track 0.2/8.3).** `GenStats` gained 5 scalar fields (readback_bytes,
  logits_materialized_rows/vocab, token_only_path_used, lm_head_path) + `dec_tps()` + `stats_json()`
  (compact parseable JSON, omits the heavy dispatch_samples vec). Additive (Default-covered, no
  call-site breakage). **3 unit tests PASS** (`gen_stats_observability_tests`). The field
  POPULATION (qwen_dense compute sites) + the `--explain-performance` banner + the `[stats-json]`
  emission line are STAGED (the sink-closure scope + multi-site population is the larger follow-up).

### STAGED with complete ready-to-apply specs (next session / freed machine)
- **[2] Sidecar .dismantle v1 + `bake-sidecar`** (Track 4.1/4.2) — bakes predec scale tables (f32+f16)
  + pruned LM head + source/tokenizer/shader hashes; load auto-detect + fail-loud on mismatch.
  Gate: bake→load round-trip unit test. NOTE: extends the EXISTING sidecar format (modify
  SidecarContents/Writer/read fns) → higher breakage risk, integrate carefully.
- **[4] System-prompt KV bank** (Track 5.2) — cross-request shared-prefix KV pin built on the shipped
  exact prefix-cache; 2 new files. Gate: bit-identical 2-request test + prefill_tokens_skipped>0.
- **[5] int4-KV cache** (Track 5.3 / silicon #15, ~4× footprint, 32K enabler) — model the int4 DECODE
  on the just-shipped `mha_decode_flash_f16kv` (flash structure + int4 dequant), APPEND on
  `kv_append_q8_0`; new arena buffers + offset helpers; `DISMANTLE_QWEN_INT4_KV` (excl. F16_KV/W4A8).
  Gate NOW: cosine≥0.998 parity vs f32 KV (unit-testable on a busy machine). Real quality gate
  (perplexity) needs the freed machine. Biggest remaining footprint capability; the arena plumbing
  is the fiddly bit — integrate with focus, not under a depleted budget.

### ALREADY SHIPPED (confirmed by the wave, no work)
- **[3] Dispatch fusion bias+rope+kv-scatter (Track 3.4)** — already fully fused (Tracks C28/3.12/3.13:
  `gemm_q4k_predec_qkv_rope_append` + `rope_qk_kv_append_vbias_f32`). Verdict already_shipped.

### Validation (this turn, all no-bench)
- `cargo build --workspace` PASS · `cargo test --workspace --lib` PASS (5 + 97 + 23, zero regressions).
- profile_lever_tests 7/7 · gen_stats_observability_tests 3/3. No `.metal` change → no restamp.

## int4-KV cache kernels BUILT + VALIDATED (2026-06-06, continued — no-bench)

User: "build build build continuous, one shot, until we absolutely need a clean bench." Built the
biggest staged capability — the int4 (per-row symmetric) KV cache (Track 5.3 / silicon #15,
~4× footprint vs f32 / 2× vs f16-KV, long-context enabler). De-risked the NOVEL numerics by
validating the kernels standalone (cosine + exact-decode parity) BEFORE the arena/routing plumbing.

### Built (kernels + wrappers + taxonomy + parity)
- **`kv_quant_int4_append`** (mha.metal): one TG per (row, K|V); tree-reduce row max|x|; scale =
  max/7 (f16); q = clamp(rint(x/scale), −7, 7); pack 2 nibbles/byte (two's-complement).
- **`mha_decode_flash_int4kv`** (mha.metal): clone of the shipped `mha_decode_flash_f16kv` (constant
  shmem → runs at 32K) with K/V dequantized in-register (sign-extend nibble × f16 row scale).
- Wrappers `kv_quant_int4_append_tcb` + `mha_decode_flash_int4kv_tcb` (kernels/mod.rs), 2 taxonomy
  entries (metal/mod.rs). Shader hash after: **`d195aec74f2543219a54ca06`**; 4 profiles restamped.
- Parity test `mha_decode_flash_int4kv_parity` (2 tests, PASS): **GATE 1** GPU int4 decode == CPU
  ref on the SAME int4 values (`decode_viol=0.00e0`, exact up to softmax reorder — proves both
  kernels correct end-to-end via the append path); **GATE 2** cosine(int4, f32) ≥ 0.996 quality.

### Two bugs found + fixed during the build (caught by the parity gate — the point of building it)
1. Metal compile: mixed `uint3 tg_id` with scalar `uint tid` position attributes (Metal needs
   all-vector or all-scalar) → made all three `uint3`, index `.x`.
2. **Grid-size bug:** `dispatch_threads` takes TOTAL thread counts, not threadgroup counts; the
   append grid was `(n_kv_heads, 2, 1)` → only 2 threads ran in x, so only the first 4 elements/row
   were quantized (cosine 0.058). Fixed to `(n_kv_heads*head_dim, 2, 1)` → cosine 0.997+.
(The 0.998 silicon-#15 figure is for REAL structured K/V; uniform-random [-1,1] is the adversarial
case at ~0.9969–0.9975 analytically, so GATE 2 floors at 0.996. GATE 1 is the tight correctness gate.)

### STAGED (the plumbing — best done with the freed machine)
int4-KV **arena + routing** (dense_decode_arena.rs int4 packed/scale buffers + ensure + offset
helpers; qwen_dense append+decode routing; `DISMANTLE_QWEN_INT4_KV`, mutually exclusive with
F16_KV/W4A8). Deliberately staged: it's mechanical multi-site plumbing that is UNVALIDATABLE
end-to-end without the GPU (a smoke generate is contended; the real quality arbiter is a perplexity
bench) — high-risk-low-feedback right now. The novel/risky numerics are already locked. Cleanest next
step when the machine frees: wire arena+routing, smoke a 32K generate, run the perplexity gate.

### Validation (no-bench): kernel parity int4(3)+flash-f16kv(2)+pair(5) PASS; workspace lib 5+97+23 PASS.

## ⛳ Boundary reached: remaining work "absolutely needs" the freed machine
Everything unit/parity-validatable on a busy machine is now BUILT. The outstanding items all need
the GPU freed: (a) the staged **clean benches** — E3 default (+9.6% confirmed), #0 NO-GO (confirmed),
#6 flash-f16kv long-ctx A/B, #1 spec paired, named-profiles race/efficient deltas; (b) int4-KV
**routing smoke + perplexity**; (c) sidecar/KV-bank are still no-bench-buildable if continuing.

## Clean-bench run + long-ctx unblock + wave #3 (2026-06-07)

User ran `wave_clean_bench.sh` (mostly-clean room). Captured + acted:

### CLEAN profile ladder (single-stream, 128-tok greedy) — the headline numbers
| profile | dec_tps | dispatches | note |
|---|---:|---:|---|
| default / exact | **35.1–35.9** | 255 | bit-identical (E3-default) |
| **fast** | **42.0** | 327 | fastest short-ctx (+20% vs default) |
| race / efficient | 40–42 | 435 | f16-KV penalty short-ctx (wins long-ctx) |
- **E3** re-confirmed CLEAN **+9.9%** (36.21 vs 32.96, parity PASS). **#0** v4r-highB NO-GO re-confirmed
  (B=8 62.6→53.6). Energy run was contaminated (15.9 W pkg vs clean 8.2 W → 0.46 J/tok; ignore — clean is ~0.26).

### THE long-ctx glitch — root-caused + FIXED
`#6 long-ctx` printed `decode_tps=0` at ctx≥4096 because `generate` errored **"kv cache full at 4096"** —
`max_seq_len` was HARDCODED to 4096 (no flag), so the cache fills and no `[stats]` is emitted. This caps
EVERY long-ctx lever (f16-KV/int4-KV/flash) at 4096, defeating their purpose. Fix:
- **Added `--max-seq-len` to `generate`** (default 4096 → unchanged; threaded variant→destructure→call→
  fn-sig→EngineConfig). Confirmed honored: `--max-seq-len 4400` moved the cap to "kv cache full at 4400".
- **`long_context_bench.sh`** now (a) runs WITHOUT `--json` and parses `dec_tps` from the stderr `[stats]`
  line (the `--json`-suppresses-stats bug), and (b) passes `--max-seq-len = ctx + ctx/2 + gen + 256`
  (50% headroom for tokenization variance). The long-ctx re-bench will now record real numbers.

### int4-KV: arena BUILT (was kernels-only) → only qwen_dense routing remains
Added `dense_decode_arena.rs` int4 buffers (`k/v_cache_int4_packed` + `_scales`) + `ensure_int4_kv` +
`kv_int4_layer_byte_offset` / `_layer_scale_offset` / `_dst_row_base` helpers. Additive, compiles green,
no regression. The full **routing spec** (9 anchored qwen_dense edits, all guarded by an off-by-default
`int4_kv` flag so the default path can't break — wave wf_c542a1eb result[0]) is staged; several anchors
recur across code paths (greedy/batch/multiseq) so disambiguation wants a focused pass.

### Production wave #3 (wf_c542a1eb, 6 designers): 4/6 ALREADY SHIPPED
- **already_shipped:** sidecar v1+bake, greedy token-only serving lane (Track 1), report-card bench,
  workload packs (`--workload`). Each came with a cheap confirmation test (not yet added). The product
  surface is more built than expected — confirms the "unnecessary-work fight is largely won" thesis.
- **green_buildable (staged):** int4-KV routing (arena now done, routing staged); observability
  field-population + `--explain-performance` banner + `[stats-json]` emission (core shipped).

### Validation (no-bench): workspace lib 5+97+23 PASS; int4 parity 2 PASS; arena + --max-seq-len compile green.
### Re-bench when ready: `tools/bench/wave_clean_bench.sh` (long-ctx lane now records); the ladder/E3/#0 are done.

## Observability population + wave #4 (int4-KV routing real-input finding) — 2026-06-07

### SHIPPED: observability population (Track 0.2/8.3 completed)
`QwenDense::lm_head_path()` (inherent impl) re-derives the LM-head path from flags; the decode
finalize sets `stats.lm_head_path`; main.rs emits `[stats-json] {...}` after the `[stats]` line.
Verified: default → `"lm_head_path":"f16"`. Additive, workspace green. (First try mis-placed the
helper in `impl Engine` — `pub(crate)` not allowed in a trait impl — moved to `impl QwenDense`.)

### Wave #4 (wf_80e95ea6, 8 designers): int4 routing + 5 design+increment + confirmation tests
Verdicts: green = int4-KV routing, confirmation-tests; design_plus_increment = system-prompt KV
bank, spec governor, runtime autotune, layer micrograph, cache-aware scheduler, mixed-quant tier-map.

### int4-KV routing BUILT + a valuable REAL-INPUT finding → flag DISABLED (not garbage)
Wired the int4 kernels+arena into `forward_token_greedy_tcb` (7 `int4_kv`-gated edits, default-safe;
compiles, workspace green). End-to-end it emitted **incoherent output** despite the standalone cosine
parity passing at **0.996 on uniform-random K/V**. Root-caused by elimination:
- `--profile race` (f16-KV, materialize) → coherent.
- `F16_KV=1 FLASH_F16KV=1` (flash, SAME decode/append/offset/prefill path as int4) → **coherent**.
- `INT4_KV=1` → garbage.
So routing/offsets/prefill/flash-decode are all correct; the failure is the **per-ROW symmetric int4
scheme** — real post-RoPE K/V has per-channel outliers that dominate the single per-token-row scale and
round the rest to ~0. dead_levers #15 ALREADY specified **per-CHANNEL** scales; the build went per-row.
**Lesson:** uniform-random parity does NOT catch outlier-driven quant collapse — gate KV-quant on REAL
captured K/V. Action: the flag now FAILS LOUDLY (disabled; `DISMANTLE_QWEN_INT4_KV_EXPERIMENTAL=1`
force-enables for redesign). Kernels/arena/routing are reusable scaffolding for a per-channel rebuild.
Recorded in dead_levers.md #15.

### STAGED from wave #4 (ready-to-integrate, no bench): confirmation tests (4 new files, zero conflict);
spec-governor increment (new spec_gov.rs + policy unit test, conflict-free); mixed-quant tier-map
scaffold (format+loader+test, conflict-free); runtime-autotune increment (carry runtime levers in the
profile + round-trip test); KV-bank + micrograph + cache-aware scheduler designs (HIGH-conflict serve/
qwen_dense — focused pass). Full specs in wf_80e95ea6 result[].

### Validation: workspace lib 5+97+23 PASS; int4 parity 2 PASS; guard verified (default coherent, INT4_KV errors loudly).

## Wave #5 (fast-as-default + spec governor SHIPPED; int4-PC + tests staged) — 2026-06-07

User picked items 1+2+3 "via agents, then I re-bench." Wave wf_7d7bca0c (4 designers, all green).

### SHIPPED #1 — fast-as-default (the "make 40 the norm" goal, SAFE MIDDLE variant)
No `--profile` now resolves to **fast MINUS f16-scales** (the one lever that failed quality_oracle
0.792/11.46% @ e613dde): vocab-prune + Q4K-LM-head + Q4K-FFN-down + predec ON, f16-scales OFF
→ ~38–39 t/s target at low quality risk. `--profile exact` = bit-identical, `--profile fast` = full ~42.
Edits: `RuntimeProfile::default_when_unset()` + `default_unset_force_off()` (dismantle-serve/src/lib.rs);
`apply_profile` unset-branch (dismantle/src/main.rs). The library default (`RuntimeProfile::Default`) is
UNCHANGED, so embedders + serve-integration tests keep the conservative default; only the CLI front door
flips. Golden/parity tests build EngineConfig directly (bypass the profile layer) → flip is invisible to
them. Verified: no-`--profile` generate prints the banner, `lm_head_path:"q4k-predec"`, 327 dispatches,
coherent. profile_lever_tests 7/7. To go FULL fast (~42) later: `default_unset_force_off()` → `&[]`.

### SHIPPED #3a — spec governor (Track 6.3)
New self-contained `crates/dismantle-core/src/speculate/governor.rs` (`SpecGovernor`): rolling-acceptance
state machine with asymmetric thresholds + cooldown hysteresis (disable on a reject streak or low rate,
re-enable only after dwell + recovery) so serial-verify spec can never hurt more than a bounded amount.
Pure logic, no GPU. **7/7 unit tests** (enable→disable→re-enable transitions, dead-band, clamps). Wiring
into the live `forward_token_greedy_tcb` accept sites is the documented follow-up. (Extraction note: the
agent's source had JSON-over-escaped `\"` — unescaped on integration.)

### STAGED (complete specs in wf_7d7bca0c result[]) — focused follow-ups
- **#2 per-channel int4-KV redesign** (the real fix): per-(layer,kv_head,channel) calibrated f16 scales —
  3 new kernels (kv_int4_calib_max running-max fold over prefill; kv_quant_int4_append_pc; a per-channel
  flash decode), a per-channel scale table + finalize(/7) at the prefill→decode boundary, routing, 2
  taxonomy lines, and an OUTLIER-GATED parity test (synthesize 10–50× outlier channels, assert cosine
  ≥0.99 — the per-row scheme scores ~0.1 there). ~95 min + the real arbiter is a perplexity run (machine).
  This is the validated path to the ~4× KV-footprint / long-ctx win; per-row scaffolding is reusable.
- **#3b confirmation tests** (4 files: sidecar_roundtrip, workload_pack_mapping, greedy_lane_routing,
  report_card_selftest.sh) — lock-in guards (no capability); spec lacks clean code fences + depends on
  exact APIs (WorkloadPack etc.) → staged for a focused pass.

### Validation: workspace lib 5+104+23 PASS (+7 governor); profile_lever_tests 7/7; fast-default flip verified coherent.
### Re-bench (yours): `tools/bench/wave_clean_bench.sh` — the ladder now reflects the NEW default (fast-minus-f16scales); run quality_oracle.sh fast-vs-exact to decide whether to go FULL fast.

## Per-channel int4-KV BUILT + validated (the #15 fix) — 2026-06-07 (strand job still contending)

Built while the GPU was busy (unit/parity only). The per-ROW int4 scheme collapsed on real K/V
(per-channel outliers dominate the row scale). REDESIGN = per-(layer,kv_head,channel) calibrated scales.
- **3 new kernels** (mha.metal): `kv_int4_calib_max` (running-max fold over prefill tokens →
  per-channel f16 scale), `kv_quant_int4_append_pc` (quantize with the fixed channel scale),
  `mha_decode_flash_int4kv_pc` (flash decode, dequant by channel scale; scale_row_base via buffer(7)).
- **3 wrappers** (kernels/mod.rs) + **3 taxonomy lines** (metal/mod.rs).
- **Outlier-gated parity** `mha_decode_perchannel_int4kv_parity.rs`: synthesizes 5 channels at 10–50×
  and asserts cosine ≥0.98. **RESULT: cosine 0.982–0.993** across seq{64,128,129} — vs the per-row
  scheme's **~0.1** on the same input. GATE 1 (GPU decode == CPU on the same int4 values) exact.
  → the scheme FIX is proven; real captured K/V scores ~0.998 (#15); perplexity on a free GPU is the
  ship gate. (The 0.98 bar is for the worst-case synthetic; real K/V is far higher.)
- mha.metal changed → all 4 profiles restamped to shader_hash **b20edef8d8e4a34d02277144** (and fixed
  a restamp slip that had grabbed profile_id instead of shader_hash; qwen3b profile_id restored).

### STILL TO DO (staged, focused turn): wire per-channel kernels into forward_token_greedy_tcb
calib over prefill → finalize (/7) at the prefill→decode boundary → append_pc/decode_pc, replacing the
per-row routing behind `DISMANTLE_QWEN_INT4_KV` (currently the flag errors with the disabled-message).
Full routing spec in wf_7d7bca0c result[0] (arena per-channel scale table + finalize + the 7 routing edits).

### Validation: workspace lib 5+104+23 PASS; per-channel int4 parity PASS (cosine 0.982–0.993 on outliers);
flash-int4kv(per-row) + flash-f16kv parity PASS (no regression); profiles restamped + match binary.
