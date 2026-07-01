# Hawking — Shippability Master Plan (2026-06-22)

> **Status of this document:** the maximal, deliberately over-engineered enumeration of *everything*
> between "powerful research engine" and "shipped product line." It is the union of (a) the owner's
> stated goals — finish spec-decode, validate higher-param models, make our speed/compression wins
> *persist*, do another speed pass and another compression pass, a headless app with file-format
> auto-detection, and a Hugging Face "Hawking Lab" releasing pre-quantized + distilled models — and
> (b) the gaps surfaced by a four-lane codebase sweep on 2026-06-22 (training/distillation, CLI/serve/
> format, quant/compression, gates/CI).
>
> **Honesty tags** used throughout: `MEASURED` (warm, on this binary), `BUILT` (code exists, validated),
> `STUB` (code path exists but returns Unimplemented / placeholder), `GAP` (does not exist), `ASPIRATION`
> (product goal, not yet specced). Effort tags: `S` (<1 day), `M` (days), `L` (1–2 weeks), `XL` (multi-week).
>
> Companion docs: `docs/campaign/project_standing_snapshot.md` (current numbers, archived — see
> `docs/ARCHIVE_INDEX.md`), `docs/plans/ratios_roadmap_2026_06_21.md` (speed/compression levers),
> `docs/campaign/kill_ledger.md` (archived, see `docs/ARCHIVE_INDEX.md`) +
> `docs/dead_levers.md` (rejected ideas — do not re-litigate), `docs/architecture.md`, `docs/env_flags.md`.

---

## 0. What "shipped" means (define the product before building it)

Hawking is three products stacked on one engine. We are not "done" until each has a release gate.

| Layer | What it is | Ship artifact |
|---|---|---|
| **P1 — The Runtime** | Apple-Silicon-first Rust/Metal inference engine + OpenAI-compatible server | Signed binary (Homebrew/GitHub release), stable `/v1/*` API, stable `.hawking` file format |
| **P2 — The Model Press / Condense** | A repeatable, memory-budgeted **bake → validate → publish** pipeline that turns any supported base model, including parents too large to hold fully resident, into an extreme-compressed Hawking artifact with a recorded recipe + quality card | `hawking press` (one command), dry-run memory planner, resumable out-of-core bakes, reproducible recipe files, per-quant quality gates |
| **P3 — Hawking Lab (Hugging Face)** | Public org releasing (a) pre-quantized models and (b) **compress-then-distill** models that recover quality toward their full-precision parents | HF org + model cards + a demo Space + a quality leaderboard |

**The dependency chain is strict:** P3 sells what P2 produces; P2 produces what P1 can load and serve correctly and *fast*; P1's wins only matter if they *persist* (regression-gated). So the build order is P1-hardening → regression gates → P2 → P3, with the distillation pipeline (the longest pole) started in parallel early.

### 0.1 Scope decisions that gate everything (OPEN — owner call)
These change what we build; flag them now rather than discover them late.

1. **Target hardware:** Apple-Silicon-only (current moat, simplest) vs add CUDA/CPU for the Lab's reach? The SSM moat is *measured* on Metal; HF users are mostly CUDA. **Recommendation:** ship P1 Apple-only; make P2/P3 artifacts also runnable by a reference path (llama.cpp/MLX/transformers) so Lab models aren't Hawking-locked.
2. **Flagship model(s):** what does Hawking Lab launch *with*? Options: RWKV-7-0.4B (our trained SSM, the moat) + one extreme-compressed Qwen. Decide the launch SKU list (§5, §10).
3. **License & provenance:** every base model we compress carries a license (Qwen=Apache-2.0/Tongyi, RWKV=Apache-2.0, Llama=community). The Lab must comply per-model (§12).
4. **"Extreme compression" target bpw:** what number do we advertise — 3.0 bpw (TQ-3, near-ready) or 2.x (research)? This sets the P2/P3 quality-recovery burden (§3, §6).
5. **Frontier-condensation targets:** do we pursue GLM-class open weights after ship finalization? GLM-5.2-class releases make "can a user quantize this at all?" a live product problem, but downloads, storage, cloud runs, and derivative publication all need owner approval.

---

## 1. Engine correctness & throughput (P1 core)

**Goal:** the server is *correct* and *fast* for every model we ship, single- and multi-stream.

**Current state:** RWKV-7 serve is now **correct end-to-end** (`MEASURED` 2026-06-22: `ssm_serve_smoke.sh` fail=0; parity gate `rwkv7_prefill_slot_multiseq_parity` green). The remaining wall is **throughput**: single-stream serve ~10 tps vs ~119 single-stream `generate` (~12×). **The "arena does 8-stream work" theory was tested and DISPROVEN 2026-06-22** (see §1.1) — bounding dispatch to `active=1` did not move serve tps, so the cost is per-token *fixed* overhead, not stream width.

| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 1.1 | **Cut per-token serve decode overhead** (was: "size the arena to active slots" — DISPROVEN: `active=1` dispatch left serve at ~9.9 tps, `MEASURED`, change reverted). Real lever: the multiseq decode path is ~12× slower/token than single-stream `forward_token_gpu` even at b=1 — profile dispatch count / `commit_and_wait` / serve-loop CPU, and diff multiseq GEMVs vs single-stream predec GEMVs. | `GAP` (re-aimed) | L | serve dec_tps → toward single-stream for 1 active stream; parity gate stays green |
| 1.2 | Per-slot `fresh` flag (replace bundle-wide bool) to enable prefill-while-streaming at B>1 | `GAP` (latent bug noted in `rwkv7.rs:~1222`) | M | concurrent admit during active decode stays coherent (new isolation test) |
| 1.3 | Request-isolation test (2 concurrent prompts, no cross-talk) — currently a TODO in `ssm_product_gate.sh:107` | `STUB` | S | automated 2-stream isolation assertion in CI-runnable gate |
| 1.4 | Dynamic batch admission tuning (already started: `docs/plans/rwkv7_dynamic_batch_hardening_2026_06_20.md`) | `BUILT`-partial | M | sustained multi-stream throughput curve recorded in `reports/` |
| 1.5 | Backpressure / queue-depth limits + graceful 429 when saturated | `GAP` | S | serve survives 100 concurrent admits without OOM/hang |

**Ship gate (P1-correctness):** `ssm_product_gate.sh` fully green (serve smoke + speed matrix + quality probes + **isolation**), for every launch SKU.

---

## 2. Speed pass #2 (the second pass you asked for)

**Goal:** close as much of the structural ~1.6× llama.cpp decode gap as is real, and bank any kernel wins.

**Current state (`MEASURED`):** the **env/config ceiling is tapped** (`--profile fast` = +7.5%, the only shippable knob). Decode-kernel micro-opt is **tapped** at M=1 (bible §3.0; simdgroup-matrix dead at M=1). The red-team (`ratios_roadmap_2026_06_21.md`) confirmed Q6_K row-blocking is *already* at the 2r optimum (free A/B: 2r 40.48 > 4r 40.04 > 1r 39.91). So pass #2 is **not** more micro-opt — it is two specific, higher-ceiling bets:

| # | Lever | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 2.1 | **MLX-diff the 1.6× gap.** MLX is open-source and faster than llama.cpp (hawking ~56% of peak vs MLX ~70–80%). Diff its qmv/qmm layout + register-blocking against ours instead of opaque MST-diffing. | `GAP` (method identified) | L | a written diff identifying ≥1 concrete layout/occupancy delta; prototype kernel A/B'd warm |
| 2.2 | **Q6_K ffn_down repack** (the stride-8 `ql` load is confirmed-uncoalesced, shared by all variants; 20.4% of per-token bandwidth). High-effort (sidecar repack + new kernel; the bible's A10 layout attempt hit −16.8%). | `GAP`, deprioritized | L | achieved GB/s A/B beats 2r default warm; else formally kill in `kill_ledger.md` |
| 2.3 | **f16-activations in GEMV** (x never byte-cut; argmax-gated) | `GAP` (red-team survivor) | M | warm tps gain ≥ noise floor (1.03×) at ≥90% argmax-identity |
| 2.4 | **GQA group-coalesced MHA** (long-ctx, 8× KV-read cut) — pairs with the compression pass | `GAP` | M | long-ctx tps gain at parity |
| 2.5 | **Prefill TTFT pass** (separate from decode; MMA prefill exists) — measure & tune time-to-first-token under serve load | `BUILT`-partial | M | TTFT curve recorded; no regression vs current |
| 2.6 | **Higher-param kernel occupancy** (7B shapes differ from 3B; re-autotune) — depends on §5 | `GAP` | M | `hawking autotune` produces a 7B profile; warm tps recorded |

**Method rails (carry forward, non-negotiable):** warm-median only (≥5 trials — cold single runs measure PSO shader-compile, the FFN_DOWN_Q4K "+29%" cold-artifact lesson); validate quality over a *distribution*, not one prompt.

**Ship gate (speed):** any lever that ships must be (a) warm-median measured, (b) quality-gated (argmax-identity or token-identity threshold), (c) added to the perf-regression baseline (§7) so it cannot silently regress.

---

## 3. Compression pass #2 (the second pass you asked for)

**Goal:** ship a genuinely *extreme* compression tier (sub-4-bit that works on Metal), wire the compression levers that are already built but dead-called, and make the Model Press capable of quantizing under an explicit memory budget. This is a general Condense pass: Hawking should pursue lower peak creation memory, lower bpw at comparable quality, higher retained quality at the same bpw, and successful artifacts from parent models that ordinary tools cannot quantize on the target machine.

**Current state:** Q4_K_M (~4.83 bpw) is the shipped floor. The two big opportunities are both **partially built**:

| # | Lever | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 3.1 | **Finish the TQ / trellis (sub-4-bit) GPU path.** CPU reference is bit-identical complete; GPU bitslice is ~70% (`tq_gpu.rs` Slice-0 bake done, Slice-2 dispatch staged, Slice-3 integration pending). This is *the* extreme-compression product (TQ-3 ≈ 3.0 bpw, TQ-2 ≈ 2.0 bpw). | `BUILT`-partial | XL | `tq_trellis_parity.rs` GPU==CPU green at decode; warm decode tps recorded; perplexity gate per bpw |
| 3.2 | **Wire per-channel int4-KV** (`kv_*_int4_*_pc_tcb` in `kernels/mod.rs:9143`; cosine 0.998, parity passes, **no flag + no dispatch today**) behind `HAWKING_QWEN_INT4_KV_PC`. −75% KV vs F16's −50%. | `BUILT`, dead-called | M | parity (decode==CPU) + long-ctx coherence @8k + real-model perplexity vs f32 KV |
| 3.3 | **Un-stub the bake backend.** `bake_sidecar_predec()` returns **`Unimplemented`** (`engine.rs:~170`) — the CLI wiring exists but the QwenDense backend has no real bake path. **This blocks P2 entirely.** | `STUB` ⚠️ | L | `hawking bake-sidecar` produces a loadable, parity-verified sidecar end-to-end |
| 3.4 | **Record the bake recipe in the format.** Sidecar stores `bake_profile` + tier-map but NOT TrellisConfig (k,L,tail_biting,affine_min), AWQ α, or W4A8 scales → can't reproduce a bake from the artifact alone. | `GAP` | M | sidecar/`.tq` header carries full recipe; `hawking verify` re-derives it |
| 3.5 | **AWQ grid-search** (today heuristic α=0.5 only; `awq_calibrate.py` TODO) → real activation-aware calibration | `BUILT`-heuristic | M | grid-searched α beats heuristic on perplexity; baked + gated |
| 3.6 | **Decide W4A8's fate.** Kernel works (−34% time) but quality fails (20% top-1 vs 95% gate) and perf 1.115× < 1.20× gate. Either fix with per-channel activation scaling (`w4a8_activation_distribution` memo) or formally retire. | `HELD` | M | meets both gates, or moved to `kill_ledger.md` |
| 3.7 | **Compression footprint reporting** as a first-class metric (bpw + RSS + KV bytes) in `hawking doctor` and the quality card | `BUILT`-basic | S | every artifact reports measured bpw + footprint |
| 3.8 | **Memory-budgeted press planner.** Before any bake, report RAM, scratch, shard plan, peak resident tensor/window, expected bpw, and unsupported features. | `GAP` | M | `hawking press --dry-run --memory-budget 18gb <model>` tells the truth and starts nothing |
| 3.9 | **Out-of-core / resumable bake.** Stream safetensors/GGUF shards tensor-by-tensor, emit verified sections early, persist a manifest, and resume after crash without repeating completed tensors. | `GAP` | L | press a model larger than target RAM with bounded RSS and a successful resume test |
| 3.10 | **4/3/2/1-bit ladder as first-class targets.** Promote existing Condense/legacy-STRAND/TQ rungs into named Hawking Press targets: q4 compatibility, tq3 shipping extreme, q2/PV recovery, ternary/native-low-bit research. | `BUILT`-partial | M | each target has recipe, quality gate, and honest speed/density label |
| 3.11 | **Output-damage-ranked bit allocation.** Join scout metrics (`rung-screen`) with KL/NLL/task damage (`rung-kl`) and allocate bits by quality budget, not naive tensor family rules. | `BUILT`-tools, `GAP`-integration | M | water-filling allocator emits a per-tensor rung map and predicted/verified quality delta |
| 3.12 | **Frontier MoE dry-run support.** Add planner support for GLM-class expert shards, router/gate protection, active-vs-total parameter accounting, and license/download warnings. | `GAP` | M | planner can inspect metadata/shard manifests and name support/blockers before terabyte downloads |
| 3.13 | **Condense-then-recover proof.** Compose QAT/KD/PV with the press so 2-bit and ternary lanes are recovery lanes, not raw PTQ claims. | `BUILT`-research, `GAP`-product | XL | a compressed student beats raw PTQ floor and ships with a quality card |
| 3.14 | **Iso-bpw competitor comparisons.** Compare Hawking artifacts against common local quantization baselines where supported, including creation peak RSS/scratch, final bpw, retained quality, and warm inference speed. | `GAP` | M | quality card shows what Hawking improved, matched, or lost versus baseline at the same bpw |

**Ship gate (compression):** an extreme tier ships only with a **per-quant-level quality card** (perplexity delta + task-class pass/fail vs the fp16 parent) and a recorded, reproducible recipe. A Condense frontier claim additionally requires measured peak RSS/scratch, a resumable shard manifest, and proof that the parent did not need to be fully resident.

### 3.A Condense frontier

Companion plan: `docs/plans/condense_frontier_2026_06_22.md`.
Naming migration: `docs/plans/condense_naming_migration_2026_06_22.md`.

The frontier is not just a lower-bit format. It is artifact creation under
constraints: streaming a massive open-weight parent through a bounded local
memory aperture and leaving behind a verified Hawking artifact. Current
GLM-5.2-class signals (MIT open weights, 1M context, roughly 744B total /
40B active MoE as reported by the GLM-5 paper and model pages) make this a real
market wedge: many users may be able to run a derivative but cannot quantize the
parent locally.

The Condense claim must stay empirical. Hawking should be able to say exactly
where it beats, matches, or loses to common quantization methods: peak creation
memory, bpw, retained quality, warm speed, recoverability after QAT/KD, and
whether the parent ever had to fit fully resident. The flagship win is making a
previously impractical condensation job possible, then recovering enough quality
that the resulting artifact is worth publishing.

This lane begins after the core P1/P2 gates are stable. It starts with dry-run
planning and small-model proofs, then scales to 7B/14B, and only then attempts a
GLM-class MoE dry-run or bake with owner approval.

---

## 4. Spec-decode: finish or formally kill (resolve the open lane)

**Goal:** stop carrying an undecided lane. Spec-decode ("Event Horizon") is built through Phase 8 + made **bit-identical to greedy** (`7ba0590`) + given a cost-aware `ProposalRouter` (`73fc5b4`).

**The honest finding:** spec-decode is **net-negative for single-stream decode *speed*** on this engine (per-cycle verify overhead wall; `kill_ledger.md` + the `eh_verify_kernel_not_lossless` memo showed batched-verify ≠ greedy at near-ties — the property gate caught it). So "finish spec-decode" means **decide its purpose**, not blindly push it:

| # | Decision path | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 4.1 | **Path A — finish for a regime where it wins.** Re-bench spec-decode for (a) *throughput* under serve batching and (b) memory-bound long-context, not single-stream latency. If it wins there, ship gated. | `BUILT`, unbenched in-regime | L | clean warm bench showing net win in a named regime, lossless-verified |
| 4.2 | **Path B — formally kill for speed + reclaim LoC.** If 4.1 shows no win, retire the speed claim; `eagle5*` (~1.6k LoC) becomes a density-consolidation target. **CAUTION:** the cost-aware router may reference spec infra; needs a reference-audit + parity gate before removal. | decision pending | M | either kept-gated with a documented regime, or removed with green parity |
| 4.3 | **Lossless guarantee.** Whatever ships must keep the Event-Horizon property gate (`event_horizon_parity_prop.rs`) green — no near-tie divergence. | `BUILT` | — | property gate green in CI |

**Recommendation:** time-box 4.1 to one clean bench; if it doesn't win in throughput/long-ctx, execute 4.2. Do **not** ship spec-decode as a single-stream "speed" feature — that claim is dead with evidence.

---

## 5. Higher-parameter model validation (you asked to test bigger models)

**Goal:** prove Hawking's wins hold beyond the current 0.4B (RWKV) / 3B (Qwen) and pick larger launch SKUs.

**Current state:** the loader dispatches 11 arch families (`model/mod.rs` `load_engine()`), so 7B/14B *load*; what's unvalidated is **memory fit + warm tps + quality at scale on M-series**, and whether kernel autotune profiles exist for the larger shapes.

| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 5.1 | **Memory-fit matrix** (`hawking doctor`) for 7B/14B Q4_K/TQ on 18–36–64–128 GB Apple-Silicon | `BUILT` tool, `GAP` data | S | a table: model × bpw × RAM → fits/swaps |
| 5.2 | **Warm tps + TTFT at 7B / 14B**, short + long context, vs llama.cpp | `GAP` | M | speed matrix in `reports/`; gap-to-llama recorded at scale |
| 5.3 | **Re-autotune kernels for larger shapes** (§2.6) | `GAP` | M | per-model `--kernel-profile` JSON committed |
| 5.4 | **Quality at scale** via the valid eval harness (§9) — does the SSM moat hold for a 7B SSM if one exists? | `GAP` | M | per-task-class pass/fail at 7B |
| 5.5 | **MoE at scale** (Mixtral/Qwen-MoE engines exist) — validate or de-scope for launch | `BUILT`-engine, `GAP`-validation | M | smoke + speed, or explicitly out-of-launch |
| 5.6 | **Fix or drop mamba2 long-ctx** (returns 0.00 tps @8k — kernel bug; secondary SSM) | `GAP` | M | fixed + re-benched, or de-scoped in `kill_ledger.md` |

**Ship gate (scale):** each larger launch SKU has a memory-fit verdict, a warm speed matrix, and a quality card.

---

## 6. The distillation & tuning pipeline (the longest pole — start now)

**Goal:** a production pipeline that takes a base model → **extreme-compresses it** → **distills/tunes the compressed student back toward the parent's quality.** This is the core IP of Hawking Lab (P3). You flagged it as needing acceleration "to the bleeding edge like the rest of the project" — correct: today it's a strong *single-device research* pipeline, not a production one.

**Current state (`BUILT`, single-device):** RWKV-7 has SFT (`rwkv7_sft_torch.py` + streaming `rwkv7_sft_stream.py`, the ~16× batch speedup), KD (`rwkv7_train_draft.py`: top-k KL + CE mix, α=0.5; teacher capture `rwkv7_capture_teacher_logits.py`), DPO/SimPO (`rwkv7_dpo_torch.py`), and QAT (`rwkv7_qat.py`, STE fake-quant). Teacher logits are cached (708 MB / ~2.4k records). **But:** MPS single-device only (no DDP/FSDP), no eval-in-loop, no Mamba2/other-arch students, no joint compress+distill, and the drafts are **undertrained** (`MEASURED`: best 75M ~19.4% top-1 vs ~60% target).

| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 6.1 | **Multi-GPU / cloud trainer** (DDP/FSDP; the single biggest gap — caps us at one 18 GB device today) | `GAP` | XL | a converged run at >1 device; throughput scales near-linearly |
| 6.2 | **Eval-in-loop + early stopping + best-ckpt selection** (today `rwkv7_eval_ppl.py` is post-hoc only) | `GAP` | M | training auto-selects on heldout metric; no manual babysitting |
| 6.3 | **Distillation loss library** (today: top-k KL + CE only). Add intermediate-layer / feature-matching / attention-relation KD + learnable temperature — what actually recovers quality after extreme compression | `GAP` | L | ≥3 KD variants behind one `DistillationLoss`; ablation recorded |
| 6.4 | **Joint compress+distill (QAT-aware KD).** Today QAT and KD are orthogonal scripts. The product *is* their composition: distill a student that is *already* in the target low-bit format. | `GAP` ⚠️ (the core P3 IP) | XL | a TQ-3/int4 student trained with quant-in-the-loop beats post-hoc quantize on quality |
| 6.5 | **Teacher capture at scale** (today blocking, full top-k per position; 100k examples ≈ 23 GB) — streaming/on-policy capture + logit compression | `BUILT`-small | L | capture 100k+ seqs without disk/time blowup |
| 6.6 | **Multi-architecture students** (today RWKV-7 only) — generalize the KD wiring to Mamba2 and to compressed transformers | `GAP` | L | a non-RWKV student trained through the same pipeline |
| 6.7 | **Data pipeline** (today static JSONL): corpus generation, curriculum, hard-example mining, dedup | `BUILT`-static | M | a documented, regenerable corpus with provenance |
| 6.8 | **Experiment tracking + recipe registry** (today: hardcoded bash chains, seed 1337, no W&B/MLflow) | `GAP` | M | every run has a versioned config + tracked metrics |
| 6.9 | **Convergence campaign** — actually push a flagship student from ~19% to the ~60% target (separate compute effort) | `GAP` | XL | a launch-quality distilled SKU |

**The product loop P2+P6 must deliver:** `base.gguf` → `hawking press --target tq3 --distill` → a compressed+recovered artifact + a quality card showing "X% of parent quality at Y bpw." For the Condense frontier, the same loop must also accept memory budgets and streamed parents: `hawking press <frontier-parent> --target tq3 --memory-budget 64gb --resume`. That one command is the Lab's value proposition.

**Ship gate (distillation):** at least one flagship "compress-then-recover" SKU where the distilled compressed student measurably closes the quality gap to its fp16 parent (per-task-class), reproducibly, from a committed recipe.

---

## 7. Make the wins persist — regression protection (you flagged this explicitly)

**Goal:** guarantee that a future code change **cannot silently** regress decode tps, compression footprint, or output quality. This is the "ensure the compression and speed set by our custom code persists" item — and today it is the **biggest shippability hole.**

**Current state:** correctness is locked (`MEASURED`: 128 test files, **193 golden token hashes**, bit-identity greedy gates, llama.cpp oracle). **Landed 2026-06-22 (§7.1/7.2/7.3):** an *enforcing* regression gate now exists — `tools/ci/regression_gate.sh` + committed `tools/ci/baselines/regression_baseline.json` gate footprint + decode_tps + lever argmax-identity and exit non-zero on a category regression (live GREEN, 6 enforced, `reports/regression/20260622T140213Z/`; wired into `preflight.sh` + `overnight_hardening.sh`). **Still open:** GitHub-CI enforces only fmt/clippy/build/test-compile/lib+bin (`.github/workflows/ci.yml`, macos-14) — the perf gate runs *locally/overnight*, not yet in the GitHub workflow (§7.5); serve-tps (§7.4), trend dashboard (§7.6), and golden-recipe re-bake (§7.8) remain.

| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 7.1 | **Perf baseline + gate.** Committed warm-median tps floors per SKU (`regression_gate.sh` + `baselines/`). Floors are CATEGORY thresholds (~10–15% below warm median) — the ±several-% noise floor makes <10% deltas unresolvable, so a "PR that slows decode >5% goes red" DoD is not physically achievable; the gate instead catches a lever silently disabled (predec OFF = −46.7%). | ✅ DONE 2026-06-22 (local + overnight; GitHub-CI wire = 7.5) | — | a category decode regression goes red ✔ (proven) |
| 7.2 | **Compression footprint gate.** `regression_gate.sh` asserts on-disk bytes ≤ committed ceiling (CPU-safe, in `preflight.sh` always). | ✅ DONE 2026-06-22 | — | a change that disables a compression path / swaps a bigger model goes red ✔ (proven) |
| 7.3 | **Quality floor enforced.** Lever argmax-identity floors (`profile_fast` ≥0.80, `f16_kv` ≥0.85) enforced via `regression_gate.sh`. `quality_oracle.sh`'s richer token-identity/drift thresholds are a superset still worth wiring. | 🟡 PARTIAL 2026-06-22 (argmax-identity floors enforced; drift-oracle not yet wired) | S | lever fidelity collapse goes red ✔ |
| 7.4 | **Serve quality/throughput gate** (today `ssm_serve_smoke.sh` checks non-empty only) — add TPS floor + the isolation test (§1.3) | `STUB` | M | serve TPS regression goes red |
| 7.5 | **GPU CI runner** for the perf/quality gates (GitHub macos-14 runs tests but no perf tracking) | `GAP` | M | nightly GPU job posts a trend |
| 7.6 | **Perf trend dashboard** (today: timestamped `reports/`, never aggregated) | `GAP` | M | HEAD-vs-baseline visible over time |
| 7.7 | **Flakiness control** (no retry/percentile logic today; thermal/GPU contention can diverge parity) | `GAP` | S | P95 tracked; flakes quarantined not ignored |
| 7.8 | **Golden-recipe gate for P2** — published Lab artifacts re-bake bit-identically from their committed recipe | `GAP` | M | CI re-bakes a sample SKU and diffs |

**Ship gate (persistence):** no launch SKU ships until its tps, footprint, and quality have committed baselines enforced by an automated gate.

---

## 8. File format & headless app (you asked: custom format auto-detection + headless app)

**Goal:** a stable, self-describing Hawking artifact and a headless service that auto-discovers and serves it — the "point it at a folder and it just works" experience.

**Current state:** loading is GGUF + `.hawking` sidecar (SHA-256-verified, v1 locked) + `.tq` (feature-gated). Architecture auto-detects from `general.architecture` (`gguf.rs` → `model/mod.rs load_engine()`). **But there is no daemon, no model discovery, no registry, no packaging** — it's a CLI you point at an explicit `--weights` path.

| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 8.1 | **Stable Hawking artifact identity.** Today the format is implicit (GGUF + adjacent `.hawking`). Decide: keep sidecar-beside-GGUF, or define a single self-describing container that records arch + quant recipe + quality card + tokenizer. | `GAP` (design) ⚠️ | L | a versioned format spec doc + `hawking verify` validates it |
| 8.2 | **Auto-detection by content** (today: GGUF magic only; a `.pt`/safetensors drop fails with "bad magic"). Detect format + arch from the file, surface a clear remediation on mismatch. | `BUILT`-partial | M | dropping any supported file "just works"; unsupported gives an actionable error |
| 8.3 | **Headless daemon / service** — long-running `hawkingd` that stays resident, exposes the `/v1/*` API, hot-loads/unloads models | `GAP` | L | a background service with health + lifecycle |
| 8.4 | **Model auto-discovery + registry** — scan a models dir, register by id, `hawking models list/install/remove`, alias instead of raw paths (the Ollama/LM-Studio experience) | `GAP` | L | `hawking serve --model qwen2.5-3b` resolves without a path |
| 8.5 | **Config file** (`~/.hawking/config.toml`) — saved profiles, per-model defaults, default models dir (today every flag is re-typed) | `GAP` | S | defaults load from config; flags override |
| 8.6 | **Auto-pull from Hugging Face** by model id (closes the P3 loop: Lab models install in one command) | `GAP` | M | `hawking pull hawking-lab/<sku>` works |
| 8.7 | **macOS service integration** (LaunchAgent) + auto-bind to a free port | `GAP` | S | `hawkingd` runs as a managed service |

**Ship gate (format/app):** format spec versioned + `hawkingd` serves an auto-discovered model with zero explicit paths.

---

## 9. Quality evaluation (the gate everything else trusts)

**Goal:** a *valid* quality measurement so every speed/compression/distillation claim is trustworthy. Today this is partially blocked and partially missing.

**Current state:** raw `hawking generate` has **no chat template** → instruct/Q&A eval is invalid; only argmax-identity lever gates and "Write X…" prompts are trustworthy. The serve fix (2026-06-22) **unblocks** valid eval via `/v1/chat/completions`. `ssm_quality_suite.sh` does 5 task classes (retrieval/JSON/math/instruction/multilingual) but there are **no standard benchmarks** (MMLU/GSM8K/etc.) and no perplexity/LLM-judge gates in CI.

| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 9.1 | **Route the quality suite through `/v1/chat/completions`** (applies the template) — now unblocked | `GAP` | S | per-task-class pass/fail is valid instruct eval |
| 9.2 | **Standard benchmark harness** (MMLU, GSM8K/MATH, IFEval, HumanEval, long-ctx retrieval) | `GAP` | L | per-SKU scores reproducible |
| 9.3 | **Per-quant-level quality cards** (perplexity Δ + task scores vs fp16 parent) — feeds P2/P3 | `GAP` | M | every artifact ships a card |
| 9.4 | **LLM-judge harness** for open-ended quality (beyond exact-match) | `GAP` | M | judged win-rate vs parent recorded |
| 9.5 | **Per-task-class routing data** (when to use the 0.4B SSM vs the 3B transformer) — fills `ssm_model_selection.md` | `GAP` | M | a routing table backed by measured per-class quality |
| 9.6 | **RWKV-7-0.4B quality quantification** (its raw quality vs Qwen-3B is unknown per class; routing must not assume parity) | `GAP` | M | per-class verdict recorded |

**Ship gate (quality):** every launch SKU has a quality card from a valid (chat-templated) eval; routing claims are backed by measured per-class data.

---

## 10. Hawking Lab on Hugging Face (P3 — the launch)

**Goal:** a public org releasing pre-quantized + compress-then-distill models, with the demo and leaderboard that make the moat legible.

**Current state:** `ASPIRATION` — nothing exists yet. Everything here depends on P2 (a working bake/press pipeline) and §9 (valid quality cards).

| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 10.1 | **HF org + naming + versioning convention** (`hawking-lab/<base>-<quant>-<recipe>@v`) | `GAP` | S | org live; naming doc'd |
| 10.2 | **Model card template** (recipe, bpw, quality card, license, how-to-run on Hawking *and* a reference path) | `GAP` | S | one card renders end-to-end |
| 10.3 | **Launch SKU list** (decision §0.1.2): e.g. RWKV-7-0.4B-SFT (the moat) + an extreme-compressed Qwen + one compress-then-distill showcase | `GAP` | M | SKUs picked + produced via `hawking press` |
| 10.4 | **Demo Space** (live serve, long-ctx moat visible: flat tps vs transformer KV wall) | `GAP` | M | a working Space |
| 10.5 | **Quality/compression leaderboard** ("X% of parent quality at Y bpw, Z tps @8k") — the differentiator | `GAP` | M | published, reproducible |
| 10.6 | **Reproducibility** — each SKU links its committed recipe + re-bake instructions (ties to §7.8) | `GAP` | M | a third party can re-bake |
| 10.7 | **Release automation** (CI builds artifact → quality card → pushes to HF on tag) | `GAP` | L | tag → published SKU |

**Ship gate (Lab):** ≥1 pre-quantized SKU **and** ≥1 compress-then-distill SKU live, each with a quality card and a reproducible recipe, plus a working demo.

---

## 11. Packaging & distribution (P1 reachability)
| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 11.1 | Signed binary releases (GitHub) for Apple Silicon | `GAP` | M | `download → run`, no Rust toolchain |
| 11.2 | Homebrew formula (`brew install hawking`) | `GAP` | S | installs + runs |
| 11.3 | Versioning + changelog + semver for the `/v1` API and `.hawking` format | `GAP` | S | documented compat policy |
| 11.4 | Install verification (`hawking doctor` self-check on fresh machine) | `BUILT`-partial | S | green on a clean box |

## 12. Security, licensing, provenance
| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 12.1 | **Serve auth + TLS** (today: none; README says "reverse-proxy if exposed") | `GAP` | M | optional API-key + TLS in-binary |
| 12.2 | Input limits / DoS guards (max tokens, max concurrent, body size) | `GAP` | S | serve survives hostile inputs |
| 12.3 | **Per-model license compliance** for every Lab SKU (Qwen/RWKV/Llama terms) | `GAP` ⚠️ | M | each card carries correct license + attribution |
| 12.4 | Provenance: record base-model hash + recipe in the artifact (ties §3.4) | `GAP` | S | artifact states its lineage |
| 12.5 | Supply-chain: pin deps, audit `vendor/strand-quant` license for redistribution | `BUILT`-partial | S | clean `cargo audit` + license review |

## 13. Docs & onboarding
| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 13.1 | User quickstart (install → pull → serve → call) | `GAP` | S | a new user serves in <10 min |
| 13.2 | `hawking press` / bake guide (P2) | `GAP` | M | reproduce a SKU from docs |
| 13.3 | API reference (OpenAPI spec for `/v1/*`) | `GAP` | S | published spec |
| 13.4 | Architecture + moat explainer (why SSM long-ctx wins) | `BUILT` (`architecture.md`) | S | current |

## 14. Observability
| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 14.1 | Real `/metrics` (today partial: "real metrics in v0.2") + token-usage in chat responses | `STUB` | M | Prometheus scrape shows tps/latency/slots |
| 14.2 | Structured request logging (opt-in, privacy-aware) | `GAP` | S | debuggable serve |
| 14.3 | Energy reporting surfaced (J/tok tooling exists: `phase_joules.sh`) | `BUILT`-tool | S | per-SKU energy on the card |

### 14.A Apple Fit frontier

Companion plan: `docs/plans/apple_fit_frontier_2026_06_22.md`.

The Apple Fit frontier makes Hawking feel native to the exact Apple Silicon
machine in front of the user: fit planning, memory-pressure awareness, context
and KV policy selection, energy/thermal reporting, and `serve --auto` intents.
This must not become a governor. The fit layer is a capability amplifier:
Hawking should expose the usable envelope, pick the strongest stable
configuration for the declared intent, and keep expert override paths open.

| # | Task | State | Effort | Gate / DoD |
|---|---|---|---|---|
| 14.A.1 | **Hardware/runtime profiler.** Detect chip family, OS, unified memory, pressure state, Metal limits, scratch, thermal/power state where available, and active Hawking jobs. | `GAP` | M | `hawking doctor --json` gives repeatable fit inputs |
| 14.A.2 | **Fit planner.** Predict model/quant/context/KV/batch envelopes before serving starts. | `GAP` | M | `hawking fit <model>` reports fastest, highest-quality, longest-context, lowest-energy, and safest-fit options |
| 14.A.3 | **Capability-first auto serve.** Add explicit intents: max-capability, max-speed, max-quality, max-context, max-battery, safe-fit. | `GAP` | M | `serve --auto` explains the selected policy and alternatives |
| 14.A.4 | **Memory-pressure engine without hidden throttles.** Warn, queue, or visibly adapt only to avoid hard failure. Restore when pressure clears. | `GAP` | L | pressure test avoids OOM/swap collapse and logs every intervention |
| 14.A.5 | **Anti-throttle gates.** Compare auto-selected profiles against best known manual profiles. | `GAP` | M | auto cannot materially lose speed, quality, or context without a stated user intent or hard resource constraint |

---

## 15. Phased roadmap (sequencing the above)

The strict dependency is P1→gates→P2→P3, with distillation (the longest pole) running in parallel from the start.

### Phase A — Runtime GA (make P1 shippable & un-regressable)
Serve **throughput** (§1.1–1.5) · speed pass #2 MLX-diff (§2.1) · **regression gates wired into CI** (§7.1–7.4, the persistence hole) · valid quality eval unblocked (§9.1) · packaging (§11) · serve auth (§12.1).
**Exit:** a signed binary serving every launch SKU correctly, fast, with perf/quality/footprint regression-gated.

### Phase B — The Model Press (make P2 real)
**Un-stub the bake backend** (§3.3 ⚠️) · memory-budgeted press planner (§3.8) · resumable out-of-core bake (§3.9) · finish TQ sub-4-bit GPU path (§3.1) · wire int4-KV (§3.2) · record recipe in format (§3.4) · per-quant quality cards (§9.3) · higher-param validation (§5) · resolve spec-decode (§4).
**Exit:** `hawking press base.gguf --target tq3` → a loadable, parity-verified, quality-carded artifact; and `hawking press --dry-run --memory-budget <N>` can honestly plan a parent larger than resident memory.

### Phase C — Distillation product (the IP)
Multi-GPU trainer (§6.1) · eval-in-loop (§6.2) · **joint compress+distill** (§6.4 ⚠️) · distillation loss library (§6.3) · convergence campaign on a flagship (§6.9).
**Exit:** ≥1 compress-then-recover SKU that measurably closes the quality gap to its parent, from a committed recipe.

### Phase D — Hawking Lab launch (P3)
File-format identity + `hawkingd` headless app + discovery (§8) · HF org + cards + SKUs + Space + leaderboard (§10) · release automation (§10.7) · license compliance (§12.3).
**Exit:** the Lab is live with pre-quantized **and** distilled SKUs, a demo, and a leaderboard.

### Phase E — Condense expansion
Frontier plan (§3.A) · GLM-class MoE dry-run support (§3.12) · out-of-core frontier bake with owner-approved storage/cloud budget · public case study only after license and quality review.
**Exit:** Hawking demonstrates a verified artifact from a parent model that could not be quantized by fully loading it on the target machine.

### Phase F — Apple Fit expansion
Apple Fit frontier (§14.A) · `hawking fit` · capability-first `serve --auto` · memory-pressure engine · energy/thermal cards · anti-throttle gates.
**Exit:** Hawking can inspect a supported Apple Silicon machine, expose the usable performance envelope, and choose the strongest stable configuration without silently weakening speed, quality, context, or model capability.

> Phases A and the *start* of C (multi-GPU trainer, capture-at-scale) should run concurrently — the distillation convergence campaign is compute-bound and slow, so it must not wait for Phase B.

---

## 16. Critical-path risks (the things most likely to bite)

1. **The bake backend is a stub (§3.3).** Everything in P2/P3 assumes `hawking bake-sidecar` works; it returns `Unimplemented`. This is the highest-leverage hidden blocker — surface it first.
2. **No perf/compression/quality regression gates (§7).** Our wins are real but *unprotected*; without gates, Phase A–D changes will erode them silently. This is the "make it persist" item and it is currently the biggest hole.
3. **Distillation is single-device (§6.1) and undertrained (§6.9).** The Lab's differentiator (recover quality after extreme compression) needs cloud/multi-GPU and a real convergence run — the longest pole; start now.
4. **Sub-4-bit is 70% done (§3.1).** "Extreme compression" as a *product* needs the TQ GPU path finished and quality-carded; the CPU path alone is too slow to ship as an inference tier.
5. **Out-of-core pressing does not exist (§3.8-§3.9).** Without it, Hawking cannot yet claim to quantize models that cannot be fully resident on the user's machine.
6. **Quality eval validity (§9).** Until the chat-templated eval + standard benchmarks exist, every quality claim on a Lab card is unanchored.
7. **Spec-decode is an undecided lane (§4)** carrying ~1.6k LoC. Decide its regime or retire it; don't ship a dead speed claim.
8. **Scope creep (§0.1).** Apple-only vs cross-platform, frontier-download budgets, and the launch SKU list change the size of P2/P3 dramatically. Decide before Phase B/E.
9. **Auto policy can accidentally become a throttle (§14.A).** Any fit/auto layer must be regression-gated against best known manual profiles, or the "helpful" path can silently make Hawking weaker.

---

## 17. Open decisions for the owner (these change what we build)

1. **Hardware reach:** Apple-only runtime, with Lab artifacts also runnable on a reference path (CUDA/MLX)? *(rec: yes)*
2. **Launch SKUs:** which models does Hawking Lab launch with? *(rec: RWKV-7-0.4B-SFT moat + 1 extreme-compressed Qwen + 1 compress-then-distill showcase)*
3. **Advertised extreme-compression bpw:** 3.0 (TQ-3, near-ready) or push 2.x (research)? *(rec: launch 3.0, R&D 2.x)*
4. **Spec-decode:** time-box a throughput/long-ctx re-bench, else retire for speed? *(rec: yes, time-box then decide)*
5. **Format identity:** keep GGUF + `.hawking` sidecar, or define one self-describing Hawking container? *(rec: self-describing container for Lab artifacts, recording recipe + quality card)*
6. **Distillation compute:** where does the convergence campaign run (the single biggest cost/longest pole)?
7. **Condense frontier target:** which frontier parent, storage budget, and publication policy should Hawking attempt after finalization? *(rec: planner + small proofs first; GLM-class dry-run before any full download)*

---

*Maintained alongside `project_standing_snapshot.md`. When an item lands, move its number to a "DONE" line with the commit + the gate that proved it — and add the regression gate that keeps it landed (§7).*
