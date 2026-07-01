# Hawking — Maximal Handoff & Codebase Understanding (2026-06-28)

> The deep, current map of this repo, written so a fresh session (or the operator) can resume
> any surface cold and know exactly what is worth doing on the M3 Pro 18 GB before the Mac Studio
> M2 Max (96 GB / 2 TB) arrives. This supersedes the short dismantle handoff. Honest by rule
> (no fake GO). Branch `serve/rwkv-multiseq-fix`, HEAD `272c864`, 857 commits since 2026-05-02.
> Working tree is dirty (campaign/plan pruning) and this session's receipt harness + plan docs
> are untracked. Canonical plan: `docs/plans/studio_maximization_2026_06_27.md` (archived, see
> `docs/ARCHIVE_INDEX.md`).

---

## 0. Orient: there are TWO systems in this repo

1. **The inference engine + condensation pipeline** (the "Hawking" product). 4 Rust crates +
   `tools/condense/` Python + `vendor/strand-quant`. This is what the Studio plan is about.
2. **HIDE** — a separate 11-crate agent/coding product (`hide-*`, `hawking-context/index/orch/
   research`). Planner -> Executor -> Verifier loop, a 13-chapter "bible," ~410 tests, a
   decoupled Tauri -> HTTP/WS transport. **Per the git log this is the operator's ACTIVE surface
   right now** (15 commits 06-27 -> 06-28, the whole last day). It is unrelated to the Studio:
   it runs anywhere and is gated on product/agent work, not on 96 GB of RAM.

Implication for "what to work on before the M2": there are two honest tracks — (A) the
condensation pre-Studio work that de-risks the moment the 96 GB lands, and (B) HIDE, which is
where active coding energy is and does not need the Studio at all. Keep them separate; do not let
condensation block HIDE or vice versa.

Ignore when reading: agent-local worktree copies, `crates/dismantle-*`, `vendor/strand-decode-kernel/`
are duplicate/fork trees of the same code.

---

## 1. The Rust inference engine

**Crates (the engine is exactly 4):** `hawking` (clap CLI binary) -> `{hawking-serve` (axum
OpenAI server), `hawking-bench`, `hawking-core` (the engine library)}`. Clean one-direction DAG.
`hawking-core` depends only on `metal`/`objc2*` (Metal), `memmap2`, `tokenizers`, `half`. The
`tq` cargo feature (default-OFF) pulls `vendor/strand-quant`; default builds are byte-identical
without it.

**Inference data path (read these to resume engine work cold):**
- Load: `model/mod.rs:70 load_engine` reads `general.architecture` and dispatches. `gguf.rs:218
  GgufFile::open` mmaps the file (`Mmap::map`), parses header/metadata/tensor-index, computes
  absolute per-tensor offsets, and **never copies weight bytes**. `QwenDense::load` eagerly
  dequantizes small tensors (norms/biases/embed->f16/lm_head); large proj/FFN weights stay as
  `TensorRef{offset,dtype}` into the mmap. The whole mmap is pinned as ONE no-copy `MTLBuffer`
  (`metal/mod.rs:918 newBufferWithBytesNoCopy`, StorageModeShared) so GPU GEMVs read an
  `(offset,len)` window directly (~1.9 GB RSS + ~324 ms TTFT saved).
- Forward, CPU reference (the parity oracle): `qwen_dense.rs:3852 forward_token` — embed ->
  per layer {rmsnorm -> QKV Q4_K-dequant GEMV + bias -> RoPE per head -> KV append ->
  `attn.rs:23 mha_decode_step` (GQA, scale 1/sqrt(d), softmax) -> O-proj -> residual ->
  ffn_norm -> gate/up GEMV -> silu_mul -> down GEMV -> residual} -> final_norm -> LM-head GEMV.
- Forward, production GPU: `qwen_dense.rs:4819 forward_token_greedy_tcb` — same math, all ops
  encoded into one `TokenCommandBuffer` committed once/token, KV + scratch in a persistent
  `DenseDecodeArena`. Greedy reads back only the 4-byte argmax (never the vocab logit row).
- Sample `sample.rs:39`; detok per token via `tokenizer.decode_one`; streamed through the
  `sink(StreamEvent::Token)` callback in `qwen_dense.rs:1381 generate`.

**Architecture support:** genuinely wired forward passes — Qwen2.5-dense (**the one verified
path**), Llama/Mistral, Gemma2, Phi3, DeepSeek-V2-Lite (real MLA+MoE), Mixtral, OLMoE, RWKV-7,
Mamba2. `qwen2moe/qwen3moe` is detected but a `Unimplemented` stub. DeepSeek routing is plain
softmax-topk (correct for V2-Lite; wrong for full V2/V3 — no sigmoid group-topk, no yarn/mscale).

**CLI/server:** `hawking/src/main.rs` clap enum, 18 subcommands (`generate`, `serve`, `bench`,
`doctor` = hardware fit, `autotune`, `fit`, `press` = condense planner dry-run, `bake-sidecar`,
`verify`, `tokenize`...). Server (`hawking-serve/src/{lib.rs:405,http.rs:291}`): axum, one
`spawn_blocking` loop owns all GPU dispatch, **continuous batching fully wired** (`batch/
scheduler.rs`+`driver.rs`: per-request Slot KV region, prefill cohorts, batched decode, prefix
reuse). `/v1/chat/completions` uses arch-specific string templating (Qwen ChatML, not Jinja).
Caveats: server-side `abort` is always None (cancel only via SSE disconnect); `stop` strings
parsed but not honored; SpecGovernor never instantiated.

**Quality read:** clean 3-layer separation, tidy zero-copy loader, strong parity-test discipline.
Debt: `kernels/mod.rs` is 13,144 lines with ~38 `gemv_q4_k_*` permutations (accumulated micro-opt
experiments, most below bench noise); `qwen_dense.rs` 10,401 lines; ~119 `HAWKING_*` env levers in
these crates (284 repo-wide); `megakernel.rs` is a pass-through POC; EAGLE/spec-decode ships a
mock random-weights head and is honestly net-negative (gated off, logged in dead_levers.md).

---

## 2. Metal kernels + native `.tq` serving (CORRECTED — it is built)

**Kernels:** 10 hand-written `.metal` files in `crates/hawking-core/shaders/` (~14.6k lines;
`quant.metal` is 5,732), `include_str!`-embedded and compiled at runtime via
`new_library_with_source` (`metal/mod.rs:728`), pipelines lazily built and cached in a
`Mutex<HashMap>`. Families: Q4_K GEMV ladder (`gemm_q4_k_*`, predec-fused v4), Q6_K, attention
(`mha_decode_flash_*`, `mla_decode_kernel`), RoPE/RMSNorm/softmax, sampling, MoE
(`moe_topk_gate`, indexed batched GEMM), RWKV7, Qwen3B megakernels. Autotune (`profile.rs`)
serializes per-device/shape kernel schedules to `profiles/*.json`, validated against
device/shader/model hash before applying.

**Native `.tq` serving — the crux, and the docs are STALE.** `docs/plans/native_tq_serving_impl.md`
and the capability-frontier doc still say "the `.tq` decoder is test-only; generate cannot serve
`.tq`." **That is no longer true in the source.** Built and live (env-gated `HAWKING_QWEN_TQ=1`,
default-off so gates stay green):
- Bitslice GEMV kernels in `shaders/strand_bitslice.metal`: `strand_bitslice_gemv_partials`
  (:327, threadgroup Q12 LUT, Viterbi inner loop), `_reduce_rows` (:443), and the
  **residual-accumulate** `_reduce_rows_accum` (:467, `y += acc`), plus `strand_rht_forward_cols`
  and `strand_outlier_correct`.
- Host drivers `strand_bitslice_gemv_tcb[_accum]` (`kernels/mod.rs:12109,12147`) fully
  implemented. `TqPreparedGpu::from_strand_tensor`/`upload_to_gpu` (`tq_gpu.rs:365,507`) complete
  (the `:363` "stub" doc-string is stale; the body is real).
- Production wiring: `QwenDense::ensure_tq_cache` (`qwen_dense.rs:4306`) reads the `.tq` via
  `crate::tq::read_strand`, builds per-linear `TqPreparedGpu`, and the GEMV override fires in
  `matmul_q4_dispatch` (:3787, base then residual accum) and the fused arena path (:5491).
  Invoked from both forward entry points; `hawking generate` reaches it. RWKV7 has the same path.
- Parity-green: CPU decode round-trip is **bit-exact** vs `strand_quant::decode::decode_tensor_fixed`
  across k in {2,3,4}, L in {5,7,8,12}; GPU residual two-part `y_gpu == (decode(base)+decode(res))·x`
  within **REL_TOL 2e-3** on 7B-shaped cols (3584, 18944).
- **The remaining gap is THROUGHPUT, not correctness:** the per-call path round-trips x/out through
  fresh GPU buffers and `commit_and_wait` per GEMV (correct but not the fused arena hot path;
  `qwen_dense.rs:3793` flags it). No `.tq` decode/GEMV tps has been measured yet — the RAM-cliff
  bench is unrun (needs a >=32B parent + Studio).

**Out-of-core / RAM cliff:** `press --dry-run --memory-budget` (`main.rs:1065 press_main`) reads
metadata only, computes out-of-core peak (`largest_tensor*4 + max_output_in_target_bpw`) vs
full-resident (`total*4`), prints the 4/3/2/1-bit ladder + FITS/EXCEEDS for each path + the wedge
diagnosis. `parse_size_arg` accepts "18gb"/"64gb".

**Deployability constraints (real, in code):** GPU bitslice needs `in_features % 256 == 0`
(0.5B's 896 falls back to CPU); `RhtMode::Rows` is NOT GPU-served (the ~1 tok/s CPU-only wall —
no baked artifact uses it); the served GEMV decodes raw Q12, so `residual_tq.py` bakes
`--no-rht`/no-outlier by default (the only combo the two-part path reproduces bit-faithfully
today); ragged tensors (n>256/block) reject at bake with CPU fallback.

---

## 3. The condensation pipeline (`tools/condense/` + `vendor/strand-quant`)

"Condensation" = compress f16 -> dynamic 1/2/3/4-bit per tensor (STRAND), then a recovery
"doctor" heals toward 1:1. All runs CPU-bf16 on 18 GB (`DOCTOR_DEVICE=cpu DOCTOR_DTYPE=bfloat16`;
f16 forbidden — MPS GQA bug + 7B fp16->nan).

**The live research loop is the 7B frontier** (`run_7b_frontier.sh`), five detached self-adopting
processes:
- `audit_ladder.py` = the engine: walks `CONFIGS["frontier"]`, for each config builds an override
  (RHT/AWQ/residual/recover), measures real ppl over `MULTIWINDOW` 2048-tok windows,
  `degr = ppl/ppl_f16 - 1`, writes one JSONL line, resumable + single-instance-locked. **Chunked
  baking** (`BAKE_CHUNKS=8`) feeds the Rust baker one byte-balanced chunk at a time so the 7B's
  ~28 GB F32 recon never lands resident.
- `frontier_conductor.py` = deterministic outer loop (never kills work): every 180 s reads JSONL,
  computes the **Pareto frontier** over (eff_bpw, degr), promotes good `+dr` and Pareto points to
  `_promotions.{jsonl,md}`, re-adopts dead supervisors, writes `_conductor_state.json` with a
  `branch` label.
- `frontier_autopilot.py` = config picker: emits the hardened seed set and capacity-ladders
  rank/alpha/outlier **only if the smaller adapter helped** (gated on measured ppl gain), appending
  via a live `_inject.py` code hook (no restart).
- `frontier_verifier.py` = independent enforcement: waits until the ladder is idle (never contends
  for RAM), re-derives each ship-candidate's recipe from its name and **reproduces from scratch**
  at `MULTIWINDOW=5`, writes VERIFIED/FLAGGED.
- `ladder.py` = the size/tier manifest (no execution): `BPW={1:1.34,2:2.34,3:3.34,4:4.50}`, the
  27-model floor-search ladder, serve/condense RAM math. Encodes the central hypothesis: the
  bit-floor descends as params rise.

**STRAND codec:** a trellis-coded scalar quantizer (QTIP/trellis-class) — each column is a path
through a `2^L`-state Viterbi trellis over a deterministic Gaussian LUT, `k` bits/weight,
`k=round(bpw)`, `L=k+6` (deeper = lower ppl, ~4x slower). `MAX_K=4` (tops out at 4-bit payload;
`vec_dim>1` enables AQLM-class vector codebooks). Folds RHT (Hadamard col transform, spreads
outliers), an 8-bit sparse outlier channel (train-free sub-3-bit rescue), and actmean
output-debias. Baker `vendor/strand-quant/target/release/quantize-model`: `--out` = decoded f16
(for ppl), `--packed-v2-out` = the STR2 deploy archive (the `.tq`, on-disk magic `STR2`). Honest
`AGGREGATE effective bpw` line includes RHT+outlier+side-info — that is what every Python builder
parses. Integer Q12 decode is the cross-platform determinism moat.

**Encode levers (all measure OUTPUT-space error, not weight-space RMS):** `awq_plus.py`
(per-tensor AWQ alpha search on folded col-weighted error), `mixed_precision.py` (output
sensitivity -> water-fill bit allocation -> `--mp-config`; produces `mp-4a3f` = 4-bit attn/3-bit
FFN), `residual_plus.py`/`residual_tq.py` (full-rank codec-native residual W ~= STRAND_b1(W) +
STRAND_b2(residual) — captures the high-rank error LoRA can't; `residual_tq.py` writes the
two-part serve archives), `calib_build.py` (multi-domain calib corpus).

**The doctor (recovery):** `doctor_lora.py` is the live one — freezes the STRAND base, trains
tiny rank-r LoRA adapters with KD: phase 1 caches the teacher's top-`KD_TOPK`(=64) logits then
frees the teacher (never co-resident in 18 GB), phase 2 streams the base in-place, phase 3 trains
forward-KL. Deployed = STRAND base + f16 LoRA (QLoRA-style). Variants are characterized
dead-ends: `doctor_qat.py` (uniform STE-QAT — mis-optimizes for the trellis), `doctor_strand.py`
(STRAND-aware QAT, SGD not AdamW to fit RAM), `doctor_blockwise.py` (BRECQ-lite full-rank,
swap-dies on 18 GB). **Best recipe today: AWQ/mixed-precision base + KD-LoRA heal
(`mp-4a3f+dr-r16-v3`). Known ceiling: LoRA is low-rank, the 2-bit error is high-rank — the
high-rank fix is codec-native residual or full-rank blockwise QAT, both Studio-only.**

**Eval:** `ppl_bench.py` (real ppl delta), `mlx_ppl.py` (3-way Hawking/llama.cpp/MLX, each vs own
f16), `multi_eval.py` (capability tripwire — QA/cloze/math/code, catches ppl healing while
arithmetic dies), `recovery_ledger.py` (per-lever recovered-points, flags the tier with most
headroom), `verdict.py` (vs llama Q4_K +2.1% @ 4.5 bpw).

---

## 4. HIDE — the active agent product (the other half of the repo)

11 crates (`hide-*`, `hawking-context/index/orch/research`). A Planner -> Executor -> Verifier
agent loop with a 13-chapter "bible," ~410 tests, and a decoupled Tauri -> HTTP/WS transport.
This is the **current active development surface** (the entire last day of commits). It does not
need the Studio. It is out of scope for the condensation Studio plan but in scope for "what the
operator is actually building." If resuming HIDE, ask for a dedicated deep-dive — these four
audits focused on the engine/condense side and only established HIDE's existence and shape, not
its internals.

---

## 5. Verification & measured reality (honest)

**Measured numbers:** Qwen2.5-3B-Q4_K_M decode ~26.6 tok/s on M3 Pro 18 GB vs llama.cpp ~50 (gap
1.88x, down from 2.46x). 7B iso-quant ~0.71x llama.cpp — treated as a permanent loss; never
compete on decode tps in public copy. Condensation 0.5B (f16 ppl 28.31): TQ3 PTQ 38.92 (+37.5%,
usable), TQ2 PTQ 449.87 (+1489%, collapsed). Output-space err: Q4_K 0.079, TQ3+AWQ+outlier
0.095-0.108 (~1.28x). 3-way quality: llama Q4_K +2.1% @4.5bpw vs Hawking TQ3 PTQ +43.7% @3.35bpw;
after recovery TQ3 LoRA-KD +30.9% and **plateaus**. **Honest: at PTQ Hawking loses on quality
badly; the entire win is gated on the doctor, which 18 GB cannot run.**

**The CI gap (cheapest credibility fix):** `.github/workflows/ci.yml` runs `cargo test
--workspace --lib --bins` — `--lib --bins` **excludes every `tests/` target**, so all
parity/quality gates only compile, never run. The only test that checks Hawking against an
independent implementation is `llama_cpp_oracle.rs` (needs `llama-cli`, installed) — and CI never
runs it. The W4A8 quality gate is `#[ignore]`d. MoE/DeepSeek is unverified (weights absent). Most
gates are GPU-vs-own-CPU or kernel-vs-scalar self-consistency (a bug shared by both paths is
invisible).

**Proof/receipt harness (built this session, works):** `receipts/schema/
condensation_receipt.schema.json`, `tools/condense/receipt_verify.py` (`--self-test` PASSES:
validates the clean R1 fixture, rejects an invalid density-win with 7 distinct rule reasons),
`emit_receipt.py`, `receipts/official/qwen-05b-tq3.json` (real R1, eff_bpw 3.6535, real sha256s),
`receipts/failures/FAIL-001.json`, frozen `prompt_suite_v1.txt`+sha. R0-R5 levels + 8 invalidation
rules + BASELINES/WATCHLIST/FAILURES all exist.

---

## 6. Hard-won dead-ends — do NOT re-litigate (dead_levers.md + condense plans)

1. Uniform-proxy STE-QAT through the STRAND trellis is **catastrophic** (healed TQ2 ppl 676 > 481
   PTQ — recovery made it worse).
2. Low-rank LoRA recovery **plateaus ~step 25** — quant error is high-rank (rank-64 SVD heals TQ3
   only 0.114->0.104 and costs 4.6 bpw > Q4_K 4.5).
3. AWQ x residual is a **non-win** (+3.72% vs plain res3+2 +1.4% at the same 6.30 bpw).
4. Diverse calib **loses** to domain-matched (+17.7% vs +14.6% on prose).
5. 2-bit PTQ collapses — needs gradient recovery, not allocation (allocator ties uniform).
6. The "99.86% recovery" was overfit theater (one-passage calib; held-out +667%). Recovery counts
   only on held-out.
7. Data-free low-rank residual codec — NO-GO (top-64 SVD captures 3-9% Frobenius energy).
8. Q3_K/QTIP sub-Q4 is dead for **tps** (compute-bound GEMV; footprint-only win).
- Methodology: judge low-bit on **big** models, never 0.5B (it floors ~3-bit and lies
  pessimistically); recovery must use the **actual STRAND codec** in the loop, full-rank; report
  **effective** bpw always. Wins: AWQ pre-scale @3-4-bit, plain full-rank residual (~1:1 at
  +1.4%), mixed-precision where redundancy exists.

---

## 7. Live run state (alive, parked at the 18 GB floor)

PIDs `62417` audit_ladder (10h+), `97736` conductor, `18265` verifier, plus keepalive/health.
Conductor stuck at `branch=waiting-for-v3, records=32, candidate_count=0`. Pareto (bpw ->
degradation): 1-RHT 1.68->+43358%, 2-AWQ 2.68->+401%, mp-3a2f 2.81->+127%, mp-4a2f 2.95->+110%,
3-AWQ 3.69->+8.6%, mp-4a3f 3.83->**+4.7%**, 4-AWQ 4.85->**-2.6%** (beats f16). **All 7 promotions
are PTQ baselines — zero doctor/recovery config has ever promoted.** Every `+dr` config swap-died
(`swap 6926/9003/11780 MB > 6000 ceiling`, `120m timeout`, leaked semaphore; run.log peak swap
25.3 GB). This is **FAIL-001: a measured hardware floor, not a recipe failure** — exactly what the
96 GB Studio removes. The baker is currently re-baking a harmless 4-AWQ.50 PTQ point.

---

## 8. What is right to work on BEFORE the M2

### Track A — condensation pre-Studio (all 18 GB-safe, all receipt-producing, all high-leverage)
1. **Wire the gates into CI.** Make `ci.yml` actually run `llama_cpp_oracle.rs` (the only external
   oracle) + un-`#[ignore]` the W4A8 quality gate on a runner that has the 3B gguf + `llama-cli`.
   Closing "compiles but never runs" is the single cheapest credibility win and needs zero new RAM.
2. **Populate `BASELINES.md`** with the exact runnable MLX-4bit and llama.cpp-Q4 commands +
   frozen prompt suite, so every receipt's baseline rows are real (not "best effort").
3. **Fold the `.tq` serve GEMV into the fused arena** (the remaining *throughput* step — the
   per-call round-trip at `qwen_dense.rs:3793`). Pure Rust/Metal, no 18 GB wall. This is what
   turns the already-correct density win into a fast shippable artifact and arms the RAM-cliff
   bench the instant a >=32B parent lands.
4. **0.5B/1.5B recovery-recipe lab.** Lock the full-rank / codec-native recipe (residual depth,
   blockwise local-MSE, KD top-k, calib) that will transfer to 7B-32B on the Studio. Emit a
   receipt per run; grow `FAILURES.md`.
5. **Fix the docs that lie** (native_tq_serving_impl.md + capability-frontier H9 still say `.tq`
   is test-only) — they understate the real build-state.

### Track B — HIDE
Whatever the agent-product roadmap is. Independent of the Studio; this is where the active coding
energy already is. (Request a HIDE deep-dive before resuming if cold.)

### Studio-only (do NOT attempt on 18 GB)
7B/14B/32B doctoring, full-rank/codec-native QAT at scale, the bit-floor-vs-scale curve, the
RAM-cliff tps bench (needs a model that does not fit at Q4_K, i.e. >=32B). Note: the 32B currently
on disk is a **Q4_K GGUF, not a bf16 parent** — a bf16 parent must be downloaded before 32B can be
doctored.

---

## 9. Invariants, git rules, locked context

- Build the instrument before the run: every condense run emits a receipt (R0-R5 tagged); no public
  win below R3, no "first/only" below R4. Effective bpw always. Output-space ppl + capability
  tripwire, never weight-space RMS alone. Judge low-bit on big models.
- Locked: Mac Studio M2 Max 96 GB / 2 TB, Apple Silicon only (Metal/MPS, NO CUDA), one project owns
  the machine, wall-clock cheap, human focus scarce, 18-year-old-intern budget. Do not reopen
  M3 Ultra / M4 Max / cloud / hiring / fundraising. bf16 at scale. 96 GB caps doctoring ~32B,
  inference ~70B; the 405B tail is cloud/rented.
- Git: NO commit/push without explicit approval. `tools/strand/` + `vendor/strand-quant` are
  production / audit-only (branch + PR, never direct main). Never add AI attribution to commits
  or PRs. The working tree is already dirty with this session's untracked work — do not blow it away.
- Do NOT disturb the live frontier processes (PIDs above) unless deliberately stopping the run.

---

## 10. Paste-into-a-fresh-session resume prompt

You are resuming **hawking** (`/Users/scammermike/Downloads/hawking`), a from-scratch Rust+Metal
Apple-Silicon LLM inference engine + a Python condensation pipeline, plus a separate active agent
product **HIDE** (`hide-*` crates). Read `docs/plans/hawking_handoff_2026_06_28.md` (this file)
in full first, then `docs/plans/studio_maximization_2026_06_27.md` (archived, see
`docs/ARCHIVE_INDEX.md`). Honor the invariants in section
9: receipts on every run, effective bpw, judge low-bit on big models, do not re-litigate the
section-6 dead-ends, no commit/push without approval, do not disturb the live frontier PIDs, Apple
Silicon only, locked hardware context. Current truth: the engine's verified path is Qwen2.5-dense;
native `.tq` serving is BUILT and parity-green (the remaining gap is throughput, not correctness);
the 7B doctor is swap-bound on 18 GB (FAIL-001) and that is the hardware floor the incoming M2 Max
96 GB clears; HIDE is the active coding surface. Tell me which surface to drive — condensation
pre-Studio Track A (CI gates / BASELINES / arena-fold the `.tq` GEMV / 0.5B recovery lab), HIDE, or
something else — and propose the next single receipt-producing or test-producing step.
