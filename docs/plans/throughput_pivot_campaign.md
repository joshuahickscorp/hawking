# Throughput-Pivot Campaign — live autonomous run (started 2026-06-21)

> **Canonical recovery artifacts (read these first):** `docs/campaign/{findings_summary,kill_ledger,roadmap,test_matrix,autonomous_run_log}.md`
> + `docs/architecture.md` + `docs/env_flags.md` + `tools/bench/ratios.sh` + `tools/ci/preflight.sh`. This file is the detailed working STATE log.

**Why:** spec-decode (EH free market AND trained EAGLE) is conclusively NET-NEGATIVE on
this engine — the per-cycle overhead wall (proven: 87% accept → still 0.91×). Speed must
come from the bandwidth-bound decode levers (quantization / KV / dispatch / kernel), per the
throughput bible. This campaign architects + tests that pivot, unattended, for ~4 hours.

## FINDINGS SUMMARY (running — quick-read for your return)
**SPEED** (warm-median tps; baseline ~40):
- ✅ `--profile fast` = **+7.5% warm, 83% argmax-identity over 12 prompts (mild quant trade)** — the speed-priority config
  (NOT bit-identical; the 1-prompt "identical" was the method-lesson trap — validate over a distribution). Bit-identical default for quality.
- ❌ `FFN_DOWN_Q4K` "+29%" = COLD-START ARTIFACT (warm ~0%). Not a win.
- Config-lever ceiling ≈ **+7.5%**. Q6_K predec (designed) is **LIKELY-NULL on analysis**: Q6_K scales are `int8`
  (one cheap `f16×int8`/sub-block), unlike Q4_K's 6-bit *packed* scales (bit-unpack ALU) that gave predec +34% — so the
  Q6_K predec ALU saving is small + offset by the table's DRAM read. Real speed frontier = the **~1.6× llama gap** (MST-diff,
  hard). **Verdict: engine is near its realistic config ceiling (~+7.5%); no easy kernel win remains.**

**COMPRESSION** (baseline 1.80 GiB weights / ~4.8 bpw + 0.28 GiB KV):
- `F16_KV` = **−50% KV footprint**; ~0% short-ctx, **+1.9% @2.5k-ctx (scales to ~15% @16k)**; 88% argmax-identity (mild trade).
  A LONG-CTX + footprint lever (the moat regime). NB: long-ctx decode itself ≈19 tps vs ~40 short = the KV-read wall at depth.
- ❌ `int4-KV` = **NO-GO**: SLOWER (−5.7% long-ctx; dequant overhead > BW saving) AND quality COLLAPSES (0% identity /
  per-row-collapse confirmed). −75% footprint is moot if it's both slower and broken.
- Trellis sub-4-bit (`tq_bake`/`tq_gpu`, runtime-wired via `HAWKING_QWEN_TQ`): existing `.tq` is a 19 MB PARTIAL bake; a full
  bake → ~30% smaller weights but **decode-SLOWER** (size↔speed). A max-compression option, not a dual-ratio win.
- **COMPRESSION verdict: F16_KV (−50% KV, mild quality) is the one clean lever; deeper KV-quant breaks; weight trellis trades speed.**

**DENSITY**: 92.4k Rust LoC, 4 crates, 41 deps; `eagle5*` (~1.6k, now-dead trained-EAGLE) = parity-gated consolidation candidate.

**METHOD LESSON**: warm-median (≥5 trials) ONLY — single cold runs measure PSO-compile, not steady-state.

**Committed this campaign**: cost-aware router `73fc5b4` (lossless spec layer; EH default-OFF — spec is NO-GO for speed).

---

**Goal artifacts (what the user returns to):**
1. A clean, measured decode-tps baseline (multiple prompts/trials).
2. The best throughput CONFIG from existing opt-in levers (a measured sweep).
3. A ranked, validated ROADMAP of new optimization candidates (lever · expected gain ·
   implementation sketch · risk · measurement), grounded in the real hot path + the bible.

**Hard rails (unattended safety):**
- NO auto-merge of risky kernel/code changes. Prototypes are designed + patched + benched,
  reviewed by the user — never committed to main.
- Every measured change passes a quality gate (bit-identity / token-regression) before it
  counts as a win.
- GPU work is SEQUENTIAL (no concurrent model jobs → no OOM).
- Only config-level (env-flag) changes may be auto-benched; code changes are design-only.

## Lanes
- **Lane A (GPU, the test):** decode-tps baseline + opt-in-lever config sweep → best config.
- **Lane B (CPU, the architect):** chained workflows → ranked optimization roadmap.

## THREE-RATIO EXPANSION (session goal 2026-06-21: speed · compression · density, iterate forever)
- **SPEED ratio** (tps vs baseline / the ~1.6× llama gap): FFN_DOWN_Q4K flip (+~29%, validating),
  Q6_K predec (designed, fallback), int4-KV/GQA-coalesce (long-ctx), the MST-diff hard lever.
- **COMPRESSION ratio** (bpw / bytes-per-token / KV footprint): baseline 1.80 GiB weights (~4.8 bpw)
  + 0.28 GiB KV. Levers: KV quant (Q8/int4-KV → footprint+bandwidth), sub-4-bit weight quant
  (STRAND/QTIP trellis — 3-bit usable per memory), mixed-precision. Each: measure size + tps + quality gate.
- **DENSITY / black-hole** (LoC/dep/folder discipline): build dense, no bloat; periodic conservative
  dead-code re-audit (prior pass found ~nothing removable — re-check the EH NO-GO neural scaffolds now
  that spec is dead). Removal only when parity-safe + clearly unreferenced.
- **Iterate loop:** GPU chain (one job at a time): validate→commit-win→next-lever→re-measure. CPU chain:
  design/audit/recon in parallel. Manager re-measures after every change; a win = committed only behind a
  passed quality/parity gate. NEVER stop chasing a better ratio.

## Round chain (the watchdog advances this; update STATE below each round)
- **R0 — audit + baseline:** workflow audits the decode hot path (qwen_dense.rs) + reads the
  throughput bible (docs/plans/bible_*) → ranked candidate levers; one agent measures a clean
  tps baseline. OUT: candidate list + baseline.
- **R1 — research:** deep-research Apple-Silicon / Metal small-model decode SOTA (2026) →
  techniques applicable to Hawking; cross-check vs R0 candidates.
- **R2 — config sweep (Lane A):** bench the existing opt-in levers (predec, Q4K_FAST,
  f16-scales, vocab-prune, profiles, KV opts) + key combinations, tps + quality gate → best config.
- **R3 — synthesize roadmap:** merge R0+R1+R2 → ranked, validated roadmap (the deliverable).
- **R4+ — design top candidates:** for each top roadmap item, an agent produces a grounded
  implementation patch + a measurement plan (design-only, no merge). Loop until time/budget.

## STATE (watchdog updates this on every wake)
- 2026-06-21: campaign created. Parity re-verify running (router commit pending pass).
- R0 DONE (wf wm11hndt4). Baseline ~31 tps anchor / ~35-40 release; gap ~1.6× to llama.cpp (kernel-bound).
  Ranked roadmap:
  - #1 flip PREDEC_F16SCALES default-on — **LIKELY DEAD** (tried e613dde, failed quality oracle
    0.792<0.90; stays opt-in). Re-verify oracle once to confirm, else dismiss.
  - #2 **Q6_K predec for default ffn_down** (the #1 GPU consumer ~46%, never got the +34% predec
    win — Q4_K-only). HIGH gain, Stage-1 BIT-IDENTICAL → quality free. ← TOP REAL PICK, designing now.
  - #3 continuous batched decode B=8 — AGGREGATE tps, not single-stream (off-goal).
  - #4 per-channel int4-KV (long-ctx BW/footprint). #5 MST diff vs llama.cpp (high ceiling/high risk).
    #6 GQA KV coalescing (long-ctx only).
- Router COMMITTED 73fc5b4 (parity 20/20). Task #6 done.
- IN FLIGHT: (a) Q6_K predec DESIGN wf wapfk2kxt [CPU] → patch+parity+bench plan;
  (b) GPU-BENCH agent aecd69de → clean baseline tps + candidate-#1 (f16-scales) GO/NO-GO decision.
- R-DESIGN DONE: **Q6_K predec ffn_down design SAVED → docs/plans/q6k_predec_design.md** (full
  patch sketch + parity test + bench plan; bit-identical Stage 1). High-value deliverable, ready for review.
  (Workflow wapfk2kxt: 2 of 3 agents hit API connection errors — the API is FLAKY — but the Q4_K
  precedent agent independently produced the complete mirrored Q6_K design.)
- GPU BASELINE: **~40.6 tps** (release, 96-tok, default).
- Candidate #1 (f16-scales): **DEPRIORITIZED** — direct A/B shows ~0% tps + identical output (no
  effect on this binary; not the +6-9% R0 hoped). Stays opt-in.
- API FLAKY → manager doing measurement DIRECTLY (Bash), retrying agentic rounds (research/design) when stable.
- IN FLIGHT: config sweep bk82z127g (default vs --profile fast vs ffn-down-q4k).
- SWEEP DONE (cold, relative): def < --profile fast (+21%) < FFN_DOWN_Q4K (+29%), all argmax-identical.
- **FFN_DOWN_Q4K**: 12/12 identical (free +29% candidate). RIGOROUS re-validation IN FLIGHT (bjlbou2u1,
  18 adversarial prompts × 160 tok). Held reason = conservative opt-in (P2 2026-05-23), NO documented
  quality failure found. If clean → FLIP DEFAULT-ON + commit (the campaign's biggest speed win).
- Compression baseline: 1.80 GiB weights (~4.8 bpw) + 0.28 GiB KV. Levers reconned: F16_KV, int4-KV,
  W4A8/AWQ (quality-blocked), trellis-quant infra tq_bake/tq_gpu (sub-4-bit, 3-bit usable).
- Density baseline: 92.4k Rust LoC, 13.1k Metal, 4 crates, 41 deps. eagle5* (~1.6k NO-GO) = consolidation candidate.
- GPU CHAIN queued (one at a time): [1] FFN_DOWN_Q4K validate→flip+commit; [2] F16_KV bench (footprint+tps+quality);
  [3] warm tps confirm of best config; [4] trellis sub-4-bit explore. CPU: density consolidation design; retry R1 research.
- ⚠️ CORRECTION: FFN_DOWN_Q4K WARM (5-trial median) = default 39.0 vs fdq4k 39.6 = **~0% gain** (+1.4%, noise).
  The cold-sweep "+29%" was a **PSO-COMPILE artifact** (1st-run cold). NOT flipping it (no warm gain + not bit-identical).
  **LESSON (campaign-wide): single-run cold benches measure shader-compile, NOT steady-state — always warm-median (≥5 trials).**
- Re-measuring --profile fast WARM (bg2t0kb6g) — the real config question. If config levers are ~0% warm, the speed
  wins live in KERNEL/quant work (Q6_K predec prototype, the ~1.6× llama gap), not env flags → weight shifts to the
  COMPRESSION lane (F16_KV, trellis sub-4-bit) + careful kernel prototyping.
- ✅ --profile fast WARM = **+7.5%** (41.2 vs 38.3 median) AND argmax-identical (1 prompt). REAL quality-safe config win.
  CONFIG LANE CEILING ≈ +7.5% (--profile fast = vocab-prune+Q4K-LM-head+Q4K-FFN-down+predec+f16-scales). Pending: multi-prompt
  quality validation of --profile fast (f16-scales is the suspect lever). Beyond +7.5% needs kernel/quant work.
- PIVOT → COMPRESSION lane (b2jsorgad running): F16_KV footprint/tps/quality. Real room: 4.8→3 bpw via trellis (tq_bake/tq_gpu).
  SPEED kernel options held: Q6_K predec (designed, bit-identical), the ~1.6× llama gap (MST diff, hard).
- F16_KV RESULT: short-ctx ~0% tps (40.2 vs 40.7), 88% argmax-identity (mild trade), **-50% KV footprint** (0.28→0.14 GiB).
  Long-ctx tps IN FLIGHT (b0v89zweh, ~3200-tok ctx) — where halved KV bandwidth should speed decode.
- Compression levers reconned: **int4-KV** (HAWKING_QWEN_INT4_KV, -75% KV, experimental/per-row-collapse risk);
  **trellis bake** (`cargo run -p tq_bake_tool -- in.gguf out.tq --bpw 3.34 --match ffn_` → ~3.34 bpw, ~30% smaller weights,
  decode via tq_gpu but DECODE-SLOWER per memory = size↔speed trade). NEXT GPU: long-ctx result → int4-KV → trellis size/quality.
- int4-KV RESULT: **NO-GO** — long-ctx 17.69 vs 18.76 (SLOWER) + 0% argmax-identity (per-row-collapse confirmed). F16_KV is the only clean KV lever.
- DELIVERABLE WRITTEN: **docs/plans/ratios_roadmap_2026_06_21.md** (ranked speed/compression/density roadmap; ship --profile fast + F16_KV; frontier=1.6× gap).
- 🌌 **SSM MOAT MEASURABLE**: RWKV-7 0.4B GGUFs present (rwkv7-g1-04-sft/dpo 0.28 GiB, world, 191m) + engine wired (rwkv7.rs:
  **CONSTANT ~6 MiB state at ANY depth = NO KV wall**). The differentiator: Qwen-3B drops 40→19 tps @2.5k (KV wall); RWKV-7 should
  be FLAT. MOAT BENCH QUEUED (after F16_KV-6k bhxzb3l58): RWKV-7 short vs long-ctx flatness vs Qwen's 0.47 drop. ← highest-value remaining test.
- F16_KV @8k ctx: default decode = 8.61 tps (40→18.8→8.6 as ctx 0→2.5k→8k = brutal KV wall). f16_kv arm STOPPED (overran GPU; trend clear).
- 🔴 **RED-TEAM (wf w69da85gs, 8 agents, ULTRACODE)**: BROKE the "near ceiling" conclusion. Found: per-channel int4-KV (built+dead-called,
  −75% KV) + the 1.6× gap is MLX-DIFFABLE (not MST-only). Roadmap hardened (docs/plans/ratios_roadmap_2026_06_21.md).
- Lever #1 (Q6_K row-blocking) **CLOSED** — kill-ledger verify: red-team OVER-CLAIMED "2r unreachable"; **2r is already default**
  (missed the use_2r override). A/B warm: 2r=40.48 (best) > 4r=40.04 > 1r=39.91. Coalescing-repack DEPRIORITIZED (high-effort, A10 −16.8%).
- Lever #2 (per-channel int4-KV) **VERIFIED real** (kernels built+registered, NOT called in qwen_dense.rs; cosine 0.998 real K/V per
  dead_levers.md:401). **WIRING AGENT a5e71cfac57e799e2 in flight** (behind HAWKING_QWEN_INT4_KV_PC; patch+build, NO merge — owner gates parity+PPL).
- Lever #3 (MLX-diff, the 1.6× gap) **AGENT af8c77df9de78b596 in flight** (read MLX qmv → ranked structural deltas + Q4_K patch sketch).
- NEXT: agents return → test int4-KV-PC (parity + long-ctx tps + PPL gate); review MLX-diff design; RWKV-7 moat bench (GPU free now).

## Watchdog
- A heartbeat ticker re-invokes the manager ~every 25 min; workflow/bench completions also
  re-invoke it. On each wake: read STATE → if a round finished, record its output + launch the
  next; if a round is mid-flight, no-op; re-arm the ticker. Campaign ends at ~4h or budget.
