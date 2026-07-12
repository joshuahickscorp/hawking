# STUDIO GO — the one-command entry point for the Hawking frontier program

> Paste target: Hawking is now on the Mac Studio. Establish the machine and efficiency receipts,
> review the safe ladder plan, then tell the coding agent "go". Heavy work remains gated and resumable.

## STEP 1 — preflight (always run first)

```
cargo run -p hawking --bin hawking -- studio preflight
```

Checks Python deps, Rust toolchain, RAM/disk, the HIDE app Node/pnpm engine, that every
`tools/condense/*.py` compiles, that `cargo check --workspace` is clean, which model parents are staged,
that the frontier refresh ledger and
refreshed HF metadata ledger can be written, that the model-aware frontier launch gate is green against
that refresh artifact, that `reports/condense/studio_preflight_summary.json` is written with a canonical
SHA-256 signature over check results plus machine/network/power/thermal/developer-environment evidence,
and that the receipt harness verifies.
Exits 0 (green, safe to `go`) or 1 (red, prints exactly what to fix). Do not run `go` on a red preflight.
The product-facing command delegates to `tools/condense/preflight.py`.
Verify a saved summary with:

```
cargo run -p hawking --bin hawking -- studio verify-summary --path reports/condense/studio_preflight_summary.json
```

## STEP 1B — signed environment receipt

```
cargo run -p hawking --bin hawking -- studio environment-capture --out reports/condense/studio_environment.json
cargo run -p hawking --bin hawking -- studio environment-verify --path reports/condense/studio_environment.json
```

This is a no-download proof-environment capture. It records the target machine class, RAM/disk envelope,
CPU/model identifiers, Hugging Face DNS/route/API reachability, expected link budget, power source,
thermal status, and powermetrics availability. The receipt is independent of preflight so benchmark,
RAM-cliff, and claim bundles can point to a stable signed "this exact Studio was ready" artifact.

The delivered machine is an **M3 Ultra Mac Studio with 96 GB unified memory, 819 GB/s advertised
memory bandwidth, and a 1 TB SSD**. The operator budget is never the nominal SSD size: derive it from
current free space, leave **150 GB untouched**, and budget **64 GB scratch + 32 GB HF/Xet cache** inside
the remaining space. Regenerate the signed environment, preflight, ledger, launch-packet, and claim-wall
artifacts on this machine; receipts captured earlier on the M3 Pro or against an M1 Ultra profile are not
same-box evidence.

## STEP 1C — efficiency charter and E0 baseline (before the training ladder)

Read [`computational_efficiency_paradigms_2026_07_11.md`](computational_efficiency_paradigms_2026_07_11.md).
It governs this run alongside the proof discipline below. FLOPS and raw tok/s remain diagnostic counters,
not the objective. Every promoted result must preserve the declared capability gate and report, where the
platform exposes them:

- useful/correct completions per joule and joules per accepted token;
- useful capability per byte moved and per resident byte;
- SLO goodput, TTFT, inter-token latency, p50/p95 wall time, and output length;
- model, KV, cache, SSD, and network bytes, plus rejected speculative work and recomputation;
- peak unified memory, memory-pressure state, swap delta, thermal state, and current free disk.

**E0 is first:** capture a no-training baseline for cold load, prefill, warm decode, long-context decode,
and the existing same-box baseline set before interpreting ladder wins. The first implementation may use
analytical tensor-byte accounting plus whole-system energy sampling, but must label estimates separately
from measured counters. Target `<2%` measurement overhead and phase-level accounting closure within 10%.
An apparent win that merely shifts work into SSD, CPU, rejected drafts, longer output, or quality loss is
not a win.

## STEP 2 — THE COMMAND

```
python3.12 tools/condense/studio_run.py go
```

Runs the guarded program in the foreground against the **96 GB** envelope. For the normal phone/remote
workflow, use the detached supervisor controls instead:

```
python3.12 tools/condense/studio_run.py start       # detached + caffeinated; resumes checkpoints
python3.12 tools/condense/studio_run.py --status    # resource + phase + move-safety JSON
python3.12 tools/condense/studio_run.py drain       # stop launches, checkpoint, fsync
python3.12 tools/condense/studio_run.py resume      # clear a completed drain and continue
```

`start`, direct `go`, and `resume` fail closed while memory pressure is elevated, swap is material,
scratch headroom is below reserve, or another CPU/RAM-heavy process is active. Wait for a green status;
only deliberate overlap may set `HAWKING_STUDIO_ALLOW_CONCURRENT=1`.

The run is resumable: only phases and models with durable pass manifests skip; a phase left `running`
after power loss reruns from its underlying completed-config checkpoints. Dry-preview first with:

```
python3.12 tools/condense/studio_run.py --go-plan
```

The safe training order is **0.5B -> 1.5B -> 7B**, then **14B alone** after the live memory oracle is
green. The **32B rung is blocked** until a measured peak proves it fits with the interactive reserve or a
streamed/blockwise path checkpoints without memory-pressure escalation. Do not let a nominal `solo` flag
override the live gate.

## WHAT `go` DOES (E0 plus ten guarded phases — see `docs/plans/quintessential_engine_2026_06_29.md` for the full design)

- **E0 CAPABILITY-EFFICIENCY BASELINE** — establish same-box quality, energy, byte, memory-pressure,
  and wall-clock receipts before the ladder. This is the accounting control for every later phase.

- **P0 CODEC TRIAGE + STAGE/ADVISE** — one-time: `codec_parallelism.py --catalog` scores every
  candidate codec/kernel design for decode PARALLELISM (not just density) before any Rust build
  time is spent — the direct lesson from the QTIP-on-Metal dead end (serial decode ate the
  bandwidth win). Per-model (inline in P1/P4): `auto_bits.py` + `size_frontier.py` +
  `doctor.py registry --select` + `arch_coverage.py` recommend the bit format, the serve regime
  (RESIDENT/MOE-PAGED/DENSE-OOC), the auto-composed recovery chain, and which Doctor levers are
  architecture-compatible (dense/SSM/MoE — Mamba2 and RWKV-7 both get their real flat-state math,
  not an approximation) — all before any bake. For MoE frontier models, `expert.py cache`
  simulates hot-expert cache hit-rate/blended-tok-s across cache sizes so the eventual OOC pager's
  cache size is chosen from a measured sweep, not a guess.
- **P1 CONDENSE** — the bit-floor-vs-scale curve across {0.5B,1.5B,7B,14B,32B}, subject to the safe
  order above, via the Doctor registry, multiwindow ppl + capability tripwire, one floor receipt per
  model, then the curve fit (H1 descent vs H0 flat). L0-L3 run first. **L4 block-QAT is gated** until
  the train-free stack leaves a measured quality gap on 7B+ and its checkpoint/memory oracle is green.
  **L5 GPTQ-Hessian is gated** until L4 or the best train-free baseline is reproducible and L5's predicted
  recoverable gap clears its own cost ceiling. Do not automatically execute L4/L5 just because registered.
  -> `reports/cron/bit_floor_curve.jsonl`, `receipts/official/*-floor.json`.
- **P2 SUBBIT** — the sub-1-bit frontier lane (PTQ1.61, residual two-part, codec-native/recover),
  gated per model by `subbit.py measure` (SUBBIT-0 entropy floor) and, for MoE, `expert.py sensitivity`.
  -> `reports/cron/bit_floor_subbit.jsonl`.
- **P3 SPEC** — remains **DEAD/BLOCKED by default**. Existing EAGLE (`tau≈0.877`) and n-gram
  (`tau≈1.43`) results miss Hawking's `tau>=2.5` resurrection threshold. No capture-retrain or governor
  run is allowed until an offline proposal oracle clears that threshold and a genuinely one-pass batched
  verifier is measured. If reopened, distribution-preserving exact-match, energy, and p95 gates all apply.
- **P4 FRONTIER** — the 100B+ research prize (235B-A22B / 405B / 671B / DeepSeek-V4-Flash /
  DeepSeek-V4-Pro / GLM-5.2 / Kimi-K2.6 / Kimi-K2.7-Code / Kimi-K2-Instruct; exact HF ids in `BASELINES.md` and
  `tools/condense/studio_manifest.py`),
  serve-oriented since they don't fit the doctor
  budget. Runs on streamed shards (entropy floor + per-expert sensitivity + serve-fit record + the
  auto-composed recovery chain). The native-serve quality + RAM-cliff are the serve build.
- **P5 EVAL + LONG-CONTEXT** — `eval_suite.py` (capability + NIAH) + `ctx_extend.py` (YaRN) +
  `kv.py frontier` (int2/trellis KV, SSD-paging, SSM) + `kv.py hybrid` (STKV: exact recall + unbounded reach).
- **P6 BASELINE** — `bench_baselines.py`: the wedge gate vs llama.cpp IQ1_S/IQ2 + MLX-4bit at matched
  effective bpw. WIN iff it beats IQ2 on 7B+; else reframe to portfolio.
- **P7 CLIFF** — `ramcliff_bench.py --all`: RAM-cliff tok/s + energy J/tok — the headline + the
  energy moat. A CLIFF-WIN requires native serve + >10x tok/s + lower J/tok.
- **P8 CODEC** — `codec_bakeoff.py`: STRAND vs QTIP/QuIP#/AQLM at matched bpw (CUDA-locked rivals
  are offline-encode-only; STRAND is the lone Metal-native trellis serve).
- **P9 SYNTH + SCORECARD** — fit both lane curves + the 70B/405B extrapolation, then `scorecard.py`:
  the populated competitive matrix. **Refuses any WIN cell without an R3+ receipt.**
  -> `reports/condense/SCORECARD.md`. **The deliverable.**

## PHASE E PILOTS — the research program beside the ladder

These pilots come from the computational-efficiency agenda and remain default-off. E0 accounting precedes
the ladder; the other pilots use ladder evidence and cannot displace a live proof gate without a measured
upper bound.

1. **E1 useful-token/locality scheduler:** price prefill, decode, cache hits, and protected p95; GO only
   at `>=1.5x` SLO goodput on a mixed trace without a protected-decode regression over 5%.
2. **E2 content-addressed state and copy-on-write:** exact hits must preserve byte-identical output; GO at
   `>=20%` TTFT reduction on a trace with 30% reuse and metadata below 10% of state bytes.
3. **E3 typed state/KV:** all-position addressability and lossless backing are mandatory; GO at `>=50%`
   state-byte reduction and `>=10%` long-context latency/energy reduction at the full quality gate.
4. **E4 predictive-innovation oracle:** count entropy-model, metadata, index, and decoder costs; GO only if
   realized held-out bytes beat the best static quantized representation by at least 20%.
5. **E5 retrieval/speculation oracle:** this is research evidence, not permission to revive P3; GO only
   after the existing `tau>=2.5` and one-pass-verifier prerequisites, then require lower completion energy,
   latency, and non-regressing p95 across the intended workload.

## LOCKED CONTEXT — do NOT reopen

- Hardware: this M3 Ultra Studio, 96 GB unified, 819 GB/s advertised memory bandwidth, 1 TB SSD.
  Metal/MPS only, NO CUDA, no cloud, no 512 GB box. The GPT/Codex app remains an interactive tenant,
  so the training process does not own all unified memory. Run one heavy job at a time; the active memory
  monitor must stop launches under pressure and drain/checkpoint Hawking before swap threatens the app.
  Wall-clock is cheap — optimize for maximum proof, not nominal utilization. Use the highest-fidelity public source:
  bf16 where available, explicit compressed-source receipts where not.
- Respect the measured dead-ends: low-rank LoRA plateaus (use full-rank), NO uniform-STE through
  the trellis (codec-aware only), AWQ x residual is a non-win, calib = domain-matched not diverse,
  judge low-bit on 7B+ never on 0.5B. `subbit.py admm` already re-confirmed NanoQuant is a low-rank
  resurrection (KILLs on real qwen-05b) — do not iterate on it.

## PROOF DISCIPLINE (the program enforces this; do not relax it)

- EFFECTIVE bpw only (baker AGGREGATE incl. RHT + outlier + side-info), never nominal.
- Quality = output-space ppl vs the f16 parent with MULTIWINDOW>=4 + the multi_eval capability
  tripwire. A floor claim is void if ppl passes but a capability collapses.
- Production headline numbers are CPU-bf16. No public WIN below repro level R3.
- FAKE-WIN BAN: a rung counts ONLY if the compressed payload stays in RAM and decode is folded into
  the GEMV. Any recipe whose served tensor is rehydrated to f16 counts ZERO. Spec-decode counts ONLY
  under the exact-match (bit-lossless) gate.

## THE TWO GATES THAT DECIDE THE MOONSHOT (both currently UNMEASURED, not refuted)

1. Does doctor recovery work inside the pressure-gated 96 GB envelope? (every +dr died on the 18 GB box
   by swap/timeout, not recipe; 14B runs alone and 32B stays gated until measured or streamed.)
2. Is MoE expert sensitivity non-uniform? (dense was uniform ~3% spread = dead; MoE is a different regime.)

If both pass: build toward the resident prize that this machine can actually support. **235B-A22B @
1.34 bpw = 39 GB, 405B @ 1.34 bpw = 68 GB, and DeepSeek-V4-Flash @ 1.34 bpw = 48 GB** are resident
targets with interactive headroom subject to measured KV and runtime peaks. **DeepSeek-V3 671B @ 1.0 bpw
= 84 GB is a pressure-sensitive edge/paging target, not a default resident claim.** GLM-5.2 @ 1.0 bpw
= 94 GB, Kimi @ 0.75 bpw = 94-103 GB, and DeepSeek-V4-Pro @ 0.50 bpw = 100 GB require paging,
streaming, a smaller verified artifact, or a non-interactive capacity mode. If recovery fails: density-only, usable floor
~3.3-3.8 bpw. If expert sensitivity is uniform: fall back to 405B @ 1.34 = 68 GB dense.
0.33/0.5 DENSE is below the information floor — fantasy; only MoE-amortized sub-1 is real.

## THE SERVE-BUILD CRITICAL PATH (the one gate on real wins, in order)

See `docs/plans/quintessential_engine_2026_06_29.md` §"Serve-build critical path" for the full spec.
RE-DERIVED FOR 96 GB: 235B-A22B, 405B, and V4-Flash have plausible resident rungs; 671B is an edge/paging
case, and GLM/Kimi/V4-Pro overflow the interactive resident budget. The OOC expert pager therefore remains
on the critical path for those larger-capability lanes, but cannot be marketed as a resident-speed win. The path:
(1) residual two-part GPU decode parity, (2) all-tensor `.tq` loader, (3) per-expert `.tq` writer +
resident heterogeneous MoE serve, (4) frontier native quality + RAM-cliff RESIDENT (flips P4/P7
GATED->MEASURED) for the three resident candidates, and (5) measured expert paging for 671B+.
Speculation is not on this critical path; its dead-lane resurrection gates remain separate. Until (1)-(4)
land, the size/quality/tps numbers stay honestly GATED.

Proof-mode native serve must fail closed. For Qwen-family `.tq` receipts, run with:

```
HAWKING_QWEN_TQ=1 \
HAWKING_QWEN_TQ_STRICT=1 \
HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR=1 \
HAWKING_QWEN_TQ_REQUIRE_GPU=1
```

Those levers make a missing sidecar, partial all-linear coverage, or silent CPU fallback an error instead
of a backward-compatible no-op. A RAM-cliff or tok/s receipt is not admissible without this proof-mode
coverage line plus the served-forward parity command.

The laptop-safe runtime contract is now a signed artifact:
`reports/condense/studio_runtime_contract.local.json`. Build and verify it before launch-packet. It
hashes the product runtime source of truth for profiles, workload defaults, energy modes, and the strict
native `.tq` proof-mode receipt requirements; it does not claim that a model has served natively.

The scorecard reads native serve receipts from `reports/condense/<LABEL>_serve.json`. A passing receipt
must state `status=pass`, `native_tq=true`, `rehydrate_f16=false`, `tq_strict=true`, `all_linear=true`,
`gpu_bitslice=true`, `served_forward_pass=true`, and a positive `tok_s`.

## STAGING (download on the Studio; `go` skips what is not present)

14B/32B/72B/MoE/100B+ parents/checkpoints are owner-gated downloads on a 1 TB SSD. Exact HF ids + sizes are in
`BASELINES.md`. `go` runs whatever is staged and skips the rest. Begin with the already staged
0.5B/1.5B/7B parents; stage 14B next and run it alone. Downloading 32B is allowed, but processing it
remains blocked until its pressure-gated or streamed plan is green. Add 72B/frontier sources only through
the current-free lifecycle. The 7B substrate + its calib/recovery data are the primary training control.
`procure.py --all-frontier
--link-mbs 300 --efficiency 0.7` estimates ~9.8 h download-only for the full nine-model frontier
manifest; perfect 300 MB/s sustain is ~6.9 h. `procure.py --cycle-frontier --link-mbs 300
--efficiency 0.7` is the operational view: download one source, bake/receipt the `.tq`, then release
that source before the next checkpoint. Every plan is derived from **current free space minus a 150 GB
safety reserve**, with **64 GB scratch and 32 GB HF/Xet cache** charged explicitly. Never plan against the
nominal 1 TB capacity. On the current disk, V4-Flash is the first feasible frontier source; 235B-A22B is
too tight as a whole-source download once reserves are charged, and all larger sources need verified
shard-streaming, external storage, or a later capacity change.

## CHECKPOINT, DRAIN, MOVE, AND REPLUG PROTOCOL

The machine may be moved between network locations. A terminal disconnect is harmless; removing power is
not. Use these boundaries:

1. **Before a long step:** record the model, phase, exact command, input hashes, current free disk, memory
   pressure, and expected checkpoint path in the Studio lifecycle ledger. Downloads use the same local
   directory and project-local HF/Xet cache so completed shards survive a restart. Processing writes a
   durable per-model/per-config checkpoint before advancing.
2. **Before unplugging:** request a drain. Stop launching new work, allow the active download shard or
   training save interval to finish, SIGTERM only the Hawking child if needed so `doctor.py` writes its
   atomic latest adapter, flush the phase/download ledger, and verify the newest artifact or HF cache.
   A PID file or an empty queue is not sufficient; the operator must see an explicit `SAFE TO UNPLUG`
   state with no active writer. The implemented command is
   `python3.12 tools/condense/studio_run.py drain`; confirm
   `python3.12 tools/condense/studio_run.py --status` reports `safe_to_unplug=true`. It also refuses
   whole-machine safety while an unrelated CPU/RAM-heavy process is still active.
3. **Move:** shut the Studio down normally. Do not unplug while a `.partial`, safetensors write, receipt
   signature, cache verification, or source-release step is active.
4. **After replugging:** confirm AC power, network route, thermal state, current free disk, memory pressure,
   and swap; rerun preflight/environment verification; then run `studio_run.py resume`. Completed
   shards/configurations/phases skip, and an incomplete current unit restarts from its last durable point.
5. **Never release a source checkpoint** until the `.tq` artifact inventory and required receipt verify.

The active monitor samples macOS memory pressure, swap delta, process-group RSS, free disk, power, and
thermal state. Yellow pressure pauses new launches; red pressure or sustained swap growth requests a
graceful Hawking checkpoint/drain. It must never kill GPT/Codex to make a benchmark pass.

Operator loop for the giant frontier:

```
FREE_GB=$(df -g "$PWD" | awk 'NR==2 {print $4}')
STORAGE_BUDGET_GB=$((FREE_GB - 150))
test "$STORAGE_BUDGET_GB" -gt 0
cargo run -p hawking --bin hawking -- studio snapshot
cargo run -p hawking --bin hawking -- studio worktree-plan --out reports/condense/worktree_split_plan.local.json
cargo run -p hawking --bin hawking -- studio worktree-plan --verify reports/condense/worktree_split_plan.local.json
cargo run -p hawking --bin hawking -- studio density-receipt-build --out reports/condense/studio_density_receipt.local.json
cargo run -p hawking --bin hawking -- studio density-receipt-verify --path reports/condense/studio_density_receipt.local.json
cargo run -p hawking --bin hawking -- studio runtime-contract-build --out reports/condense/studio_runtime_contract.local.json
cargo run -p hawking --bin hawking -- studio runtime-contract-verify --path reports/condense/studio_runtime_contract.local.json
cargo run -p hawking --bin hawking -- studio status --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
cargo run -p hawking --bin hawking -- studio storage-plan --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32 --link-mbs 300 --efficiency 0.7
cargo run -p hawking --bin hawking -- studio lifecycle --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
cargo run -p hawking --bin hawking -- studio gate --phase procure --require-refresh reports/condense/frontier_refresh.preflight.json --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
cargo run -p hawking --bin hawking -- studio license-plan
cargo run -p hawking --bin hawking -- studio license-decisions draft --out reports/condense/frontier_license_decisions.draft.json
cargo run -p hawking --bin hawking -- studio license-decisions verify --path reports/condense/frontier_license_decisions.draft.json
cargo run -p hawking --bin hawking -- studio review-plan --refresh reports/condense/frontier_refresh.preflight.json --out reports/condense/frontier_review_plan.local.json
cargo run -p hawking --bin hawking -- studio review-decisions draft --refresh reports/condense/frontier_refresh.preflight.json --out reports/condense/frontier_refresh_review_decisions.draft.json
cargo run -p hawking --bin hawking -- studio review-decisions verify --path reports/condense/frontier_refresh_review_decisions.draft.json
cargo run -p hawking --bin hawking -- studio proof-pack --force
cargo run -p hawking --bin hawking -- studio launch-packet-build --out reports/condense/studio_wave0_launch_packet.json --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
cargo run -p hawking --bin hawking -- studio launch-packet-verify --path reports/condense/studio_wave0_launch_packet.json
cargo run -p hawking --bin hawking -- studio audit-grade-build --out reports/condense/studio_audit_grade.local.json
cargo run -p hawking --bin hawking -- studio audit-grade-verify --path reports/condense/studio_audit_grade.local.json
cargo run -p hawking --bin hawking -- studio run-next --require-refresh reports/condense/frontier_refresh.preflight.json --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
python3.12 tools/condense/frontier_ops.py ledger --refresh-hf --out reports/condense/frontier_ledger.launch.json
python3.12 tools/condense/frontier_ops.py refresh
cargo run -p hawking --bin hawking -- studio review-plan --refresh reports/condense/frontier_refresh.json --out reports/condense/frontier_review_plan.launch.json
cargo run -p hawking --bin hawking -- studio review-decisions draft --refresh reports/condense/frontier_refresh.json --out reports/condense/frontier_refresh_review_decisions.launch.json
cargo run -p hawking --bin hawking -- studio review-decisions verify --path reports/condense/frontier_refresh_review_decisions.launch.json
# for every REVIEW candidate printed by refresh:
cargo run -p hawking --bin hawking -- studio review-candidate <hf_id> --decision watch --by <name> --note <why>
# or, after editing/re-drafting a final signed workbook with --by/--note/--final:
cargo run -p hawking --bin hawking -- studio review-decisions apply --path reports/condense/frontier_refresh_review_decisions.launch.json --confirm
cargo run -p hawking --bin hawking -- studio license-plan
cargo run -p hawking --bin hawking -- studio record-license <label> --status accepted --by <name> --license <id> --terms-url <url> --allowed-use research --redistribution none --source-policy local-only-delete-after-bake --note <decision>
# or, after filling a final signed license workbook per model:
cargo run -p hawking --bin hawking -- studio license-decisions sign --path reports/condense/frontier_license_decisions.draft.json --out reports/condense/frontier_license_decisions.final.json
cargo run -p hawking --bin hawking -- studio license-decisions verify --path reports/condense/frontier_license_decisions.final.json
cargo run -p hawking --bin hawking -- studio license-decisions apply --path reports/condense/frontier_license_decisions.final.json --confirm
cargo run -p hawking --bin hawking -- studio storage-plan --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32 --link-mbs 300 --efficiency 0.7
cargo run -p hawking --bin hawking -- studio lifecycle --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32 --link-mbs 300 --efficiency 0.7
cargo run -p hawking --bin hawking -- studio proof-pack --force --out reports/condense/frontier_proof_pack.local.json
cargo run -p hawking --bin hawking -- studio source-provenance-plan
cargo run -p hawking --bin hawking -- studio coverage-plan
cargo run -p hawking --bin hawking -- studio source-provenance-receipt draft <label> --sign-draft
# after procurement fills final HF revision, source format, file manifest, and download/verify receipt:
cargo run -p hawking --bin hawking -- studio source-provenance-receipt sign <label>
cargo run -p hawking --bin hawking -- studio source-provenance-receipt verify <label>
cargo run -p hawking --bin hawking -- studio parity-receipt draft <label> --sign-draft
# after architecture parity runs fill final rows, exact commands, hashes, trace hashes,
# adapter/tensor-map receipts, tokenizer/context contracts, unsupported exits, and verified features:
cargo run -p hawking --bin hawking -- studio parity-receipt sign <label>
cargo run -p hawking --bin hawking -- studio parity-receipt verify <label>
cargo run -p hawking --bin hawking -- studio coverage-receipt draft <label> --kind both --sign-draft
# after baseline/eval runs fill final rows, exact commands, trace artifacts,
# machine/environment proof, same-box group, and frozen suite/score-set hashes:
cargo run -p hawking --bin hawking -- studio coverage-receipt sign <label> --kind both
cargo run -p hawking --bin hawking -- studio coverage-receipt verify <label> --kind both
cargo run -p hawking --bin hawking -- studio receipt-plan
# before native serve/RAM-cliff runs:
cargo run -p hawking --bin hawking -- studio receipt-record draft <label> --kind both --sign-draft
# after native serve emits a strict JSON report:
cargo run -p hawking --bin hawking -- studio serve-capture <label> --artifact <artifact.tq> --bench-json <serve_report.json> --command '<exact hawking serve bench command>' --load-receipt <load_trace.json> --served-forward-receipt <served_forward_trace.json> --parity-receipt <serve_parity_trace.json> --force
# after native serve/RAM-cliff rows are final, measured, and traced:
cargo run -p hawking --bin hawking -- studio receipt-record sign <label> --kind both
cargo run -p hawking --bin hawking -- studio receipt-record verify <label> --kind both
cargo run -p hawking --bin hawking -- studio experiment-plan
# before expensive-mode experiments:
cargo run -p hawking --bin hawking -- studio experiment-receipt draft <label> --sign-draft
# after seeds/ablations/rungs/repeats/nulls/rebake rows are final, same-run, and trace-hashed:
cargo run -p hawking --bin hawking -- studio experiment-receipt sign <label>
cargo run -p hawking --bin hawking -- studio experiment-receipt verify <label>
cargo run -p hawking --bin hawking -- studio claim-bundle-build <label>
cargo run -p hawking --bin hawking -- studio claim-bundle-verify reports/condense/<LABEL>_claim_bundle.json
cargo run -p hawking --bin hawking -- studio run-next --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32 --link-mbs 300 --efficiency 0.7
python3.12 tools/condense/frontier_ops.py launch-gate --phase procure --require-refresh reports/condense/frontier_refresh.json --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32
python3.12 tools/condense/procure.py --cycle-frontier --link-mbs 300 --efficiency 0.7
python3.12 tools/condense/procure.py --cache-status
python3.12 tools/condense/procure.py <label> --retries 2 --min-observed-mbs 80 --verify --progress-interval-s 60 --stall-timeout-s 900
cargo run -p hawking --bin hawking -- studio status --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32   # refresh FREE_GB after each wave
# after bake + receipt + verify:
python3.12 tools/condense/frontier_ops.py record-event <label> --stage bake --status pass --duration-s <seconds> --artifact <path>
python3.12 tools/condense/frontier_ops.py artifact-inventory <label>
python3.12 tools/condense/frontier_ops.py release-source <label> --dry-run
```

`release-source` is dry-run by default and refuses to delete source checkpoints unless a `.tq` artifact
exists and either a frontier record or official receipt exists. Use `--yes` only after the dry-run evidence
is correct. `procure.py` pins `HF_HOME`, `HF_HUB_CACHE`, and `HF_XET_CACHE` under `scratch/` by default,
records cache deltas in every download receipt, and exposes `procure.py --cache-prune` as a dry-run cache
maintenance hook. The maximal download command retries resumably with fewer workers if the command fails
or under-runs the observed-throughput floor, records live progress samples, terminates a long no-progress
stall, attaches route/HF/DNS/network diagnostics to bad attempts, then runs `hf cache verify` against the
local dir before the attempt is considered green.
`license-plan` prints the accepted-terms command for every frontier model;
`reviewed` is not enough for procurement. The product-facing `hawking studio record-license` command
must record license id, terms URL, allowed use, redistribution policy, source-retention policy, signer,
and note for accepted terms. `license-decisions draft|sign|verify|apply` provides the same accepted-terms
gate as a signed batch workbook; apply still requires complete final rows and `--confirm`.
`frontier_ops.py lifecycle` is the
operator DAG: if it says `needs-license-review`, do not download; if it says `ready-bake`, bake; if it
says `ready-release-source`, run the release dry-run.
`run-next` is dry-run by default and refuses human-proof gates or placeholder commands; it requires explicit
flags before downloads or heavy compute, and downloads also require a `--require-refresh` artifact. `record-event`
makes bake/serve/eval/archive durations part of the ledger. `artifact-inventory` hashes the durable `.tq`
output and source release refuses to proceed without a matching inventory. Refresh candidates tagged `REVIEW`
must receive a `hawking studio review-candidate` decision before the launch gate can go green;
`review-plan` writes the durable candidate-decision queue from the refresh ledger, and
`review-decisions draft|verify|apply` provides a signed batch workbook for the same human decisions.
The Rust CLI now exposes the same proof state through `hawking studio`: `snapshot` reads signed local
artifacts and prints the current red/green wall; `worktree-plan` groups the dirty tree by subsystem for
review splitting and verifies the signed split receipt; `density-receipt-build` and
`density-receipt-verify` sign the largest-file, LOC, disk, and local artifact-mass snapshot without
deleting evidence or weakening gates; `runtime-contract-build` and
`runtime-contract-verify` seal the native `.tq` proof-mode/runtime policy before launch-packet; `status`, `storage-plan`, `lifecycle`, `gate`,
`license-plan`, `record-license`, `license-decisions`, `review-plan`, `review-candidate`, `review-decisions`, `source-provenance-plan`,
`source-provenance-receipt`, `parity-receipt`, `coverage-plan`, `coverage-receipt`, `receipt-plan`, `experiment-plan`,
`claim-bundle-build`, `claim-bundle-verify`, `proof-pack`, `launch-packet-build`, `launch-packet-verify`, and
`audit-grade-build`, `audit-grade-verify`, `receipt-record`, `experiment-receipt`, `serve-capture` delegate to the guarded
operator tool; and `run-next` prints the next command without executing it. This is the product-facing
shell for Studio wave 0; heavy work still requires explicit proof gates and allow flags.
`proof-pack` is the one-command non-compute wall for the frontier manifest. It writes a signed manifest,
signed but blocked draft envelopes for source provenance, parity, baseline/eval coverage, native
serve/RAM-cliff, and experiment depth for each frontier label, then builds
`<LABEL>_claim_bundle.local.json` files that hash those drafts and remain claim-inadmissible. It preserves
final receipts unless `--force-final` is explicitly passed, so it can be rerun before Studio measurements
without erasing completed proof. Use
`hawking studio proof-pack --force` as the product-facing entry point; `frontier_ops.py proof-pack` is the
lower-level equivalent. Verifying those `.local` bundles should stay red for admissibility while reporting
`signature_ok=true`; only final `<LABEL>_claim_bundle.json` bundles may satisfy public claims.
`launch-packet-build` signs the Studio wave-0 packet by hashing/summarizing the preflight summary,
environment receipt, signed worktree split plan, signed runtime contract, refresh ledger,
license/review workbooks, storage plan, lifecycle dry-run, procurement gate, proof pack, and run-next
dry-run. A valid packet can still be red; `launch-packet-verify` proves the packet did not drift while
preserving `procurement_permitted=false` until every launch gate is green.
`audit-grade-build` signs the audit-grade receipt by parsing the Studio deep-audit facet table, hashing
the external harsher audit, and summarizing the launch packet, proof pack, worktree split, runtime
contract, density receipt, claim gate, procurement gate, and scorecard artifact. `audit-grade-verify` proves that receipt
did not drift. A valid audit-grade receipt can still report `target_reached=false`; on the laptop it
should also report `runtime_contract_ok=true` and `frontier_claims_walled=true` while the public-claim
gates remain red.
`source-provenance plan` prints the required source checkpoint provenance path for every frontier model.
Before a public claim, each `<LABEL>_source_provenance.json` must be final and signed with exact HF
revision, source kind, source format, procurement command, download/cache verification receipt, and
file-manifest evidence. Compressed/FP4/FP8 sources must record `source_is_prequantized=true` and a
source-format receipt; bf16 parents must record `source_is_prequantized=false` and `source_format=bf16`.
`hawking studio source-provenance-receipt draft` writes signed but blocked source-provenance envelopes for a label.
After verified procurement fills final HF revision, source kind/format, exact procurement command,
download/cache verification receipt, and file-manifest evidence, `hawking studio source-provenance-receipt sign`
refuses to sign anything draft, placeholder-filled, source-format-inconsistent, or missing verification
evidence. `hawking studio source-provenance-receipt verify` rejects unsigned, tampered, draft, or
placeholder source provenance.
`coverage-plan` prints the required baseline and eval receipt paths for every frontier model. Before
any public claim, each `<LABEL>_baselines.json` must contain same-box measured rows or explicit N/A rows
with reasons for llama.cpp Q4_K_M, llama.cpp IQ2_S, llama.cpp mmap OOC, MLX 4-bit, Unsloth Dyn 2.0,
and EXL3/PonyExl3. Each `<LABEL>_eval.json` must cover ppl multiwindow, capability QA, math, coding,
tool-use, long-context recall, RAM-cliff, and native-serve domains with pass or reasoned N/A rows.
`hawking studio parity-receipt draft` writes signed but blocked architecture parity envelopes for a label.
After the real reference-backend and Hawking/native runs are complete, `hawking studio parity-receipt sign`
refuses to sign unless the record is final, measured, threshold-clean, trace-hashed, exact-commanded,
adapter/tensor-map backed, tokenizer/context contracted, and confirms every family-specific required
native feature from `frontier_parity.py`.
`hawking studio parity-receipt verify` rejects unsigned, tampered, draft, placeholder, loose-logit,
trace-free, contract-free, or feature-incomplete parity records.
`hawking studio coverage-receipt draft` writes signed but blocked baseline/eval envelopes for a label.
After the real same-box runs are complete, `hawking studio coverage-receipt sign` refuses to sign unless
the record is final, covers every required row, carries machine fingerprint/environment receipt,
same-box group, frozen suite hash, frozen score-set hash, exact non-placeholder commands, and a
receipt/artifact/log trace for every measured/pass row. Best-effort baseline rows cannot unlock claim
bundles. `hawking studio coverage-receipt verify` rejects unsigned or tampered records.
`receipt-plan` prints the stricter serve/RAM-cliff receipt contract. `<LABEL>_serve.json` must use
schema `hawking.frontier_serve.v1`, identify the artifact hash, record exact commands and commit, pass
native `.tq` proof mode, prove no f16 rehydrate, prove all-linear/GPU ownership, set `parity_pass=true`,
report positive tok/s, include a load trace, and prove positive peak/resident/unified-memory fields with
`resident_memory_ok=true`. `<LABEL>_ramcliff.json` must use schema `hawking.frontier_ramcliff.v1`,
be `source=measured`, not modeled, identify the artifact hash, show native `.tq` serving, Q4_K overflow,
>10x cliff, lower resident J/tok, exact commands, commit, and Studio machine class.
`hawking studio receipt-record draft` writes signed but blocked native serve/RAM-cliff envelopes for a label. After the
real native serve and RAM-cliff runs are complete, `hawking studio receipt-record sign` refuses to sign unless the record
is final, measured, strict-native, trace-backed, and non-placeholder. Serve records need a served-forward
or parity trace; RAM-cliff records need powermetrics/energy and baseline traces. `hawking studio receipt-record verify`
rejects unsigned, tampered, draft, placeholder, synthetic/modelled, or trace-free native receipts.
`serve-capture` is the Studio bridge for the native serve half. Feed it an existing `.tq` artifact, the
JSON report emitted by Hawking's native serve bench, the exact command, a load trace, and served-forward
plus parity trace receipts. It hashes the artifact and bench JSON, refuses f16 rehydrate/fallback reports,
requires strict/all-linear/GPU ownership, positive tok/s, resident memory proof, served-forward pass, and
parity pass, then writes the signed `<LABEL>_serve.json`. Use `hawking studio serve-capture` as the product-facing entry point;
`frontier_ops.py serve-capture` is the lower-level equivalent.
`experiment-plan` prints the expensive-mode matrix contract. `<LABEL>_experiment_matrix.json` must cover
at least 3 floor seeds, required calibration ablations, at least 4 bpw rungs, MoE expert allocation
or reasoned dense/N/A, 3 cold and 3 warm RAM-cliff runs, baseline variants, at least 2 archived null
certifications, and a rebake/hash verification row.
`hawking studio experiment-receipt draft` writes signed but blocked expensive-mode matrix envelopes for a label. After
the real experiment rows are complete, `hawking studio experiment-receipt sign` refuses to sign unless the matrix is
final, real/measured, covers every depth requirement, has exact non-placeholder commands, binds to one
Studio run with machine fingerprint, environment receipt, artifact inventory receipt/hash,
source-provenance receipt, and experiment-plan hash, and carries a row receipt/artifact/log/report trace
plus trace SHA-256 for every pass/measured/certified row. `hawking studio experiment-receipt verify`
rejects unsigned, tampered, draft, placeholder, trace-free, hash-free, or depth-incomplete matrices.
`hawking studio claim-bundle-build` signs the final public-claim evidence by SHA-256 after signed source
provenance, signed parity, signed baseline/eval, signed native serve/RAM-cliff, and signed experiment
matrix files exist. `hawking studio claim-bundle-verify` rejects stale, missing, or
claim-inadmissible bundles, reports whether the bundle signature itself is valid, and
`launch-gate --phase claim` treats missing signed bundles as a hard
failure.

Before any quality, tok/s, or RAM-cliff claim, run:

```
cargo run -p hawking --bin hawking -- studio coverage-plan
cargo run -p hawking --bin hawking -- studio source-provenance-receipt verify <label>
cargo run -p hawking --bin hawking -- studio coverage-receipt verify <label> --kind both
cargo run -p hawking --bin hawking -- studio parity-receipt verify <label>
cargo run -p hawking --bin hawking -- studio receipt-plan
cargo run -p hawking --bin hawking -- studio receipt-record verify <label> --kind both
cargo run -p hawking --bin hawking -- studio experiment-plan
cargo run -p hawking --bin hawking -- studio experiment-receipt verify <label>
python3.12 tools/condense/frontier_parity.py status
HAWKING_QWEN_TQ=1 HAWKING_QWEN_TQ_STRICT=1 HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR=1 HAWKING_QWEN_TQ_REQUIRE_GPU=1 \
  cargo test -p hawking-core --features tq --test qwen_tq_serve_parity -- --ignored --nocapture
cargo run -p hawking --bin hawking -- studio claim-bundle-build <label>
cargo run -p hawking --bin hawking -- studio claim-bundle-verify reports/condense/<LABEL>_claim_bundle.json
python3.12 tools/condense/frontier_ops.py launch-gate --phase claim --require-refresh reports/condense/frontier_refresh.json
```

The claim gate stays red until each frontier model has a passing signed `<LABEL>_parity.json` receipt plus
passing signed baseline/eval coverage receipts, strict signed native-serve/RAM-cliff receipts, and a complete
signed expensive-mode experiment matrix verified by the signed receipt runner, then a signed
`<LABEL>_claim_bundle.json` that verifies every evidence file by hash. A modeled or synthetic RAM-cliff
record is useful as a probe, but it is not claim-admissible.
