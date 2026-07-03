# STUDIO GO — the one-command entry point for the Hawking frontier program

> Paste target: when Hawking is on the Mac Studio, run preflight, then tell the coding agent "go".
> Everything downstream is already built, gated, resumable, and continuous. No pauses.

## STEP 1 — preflight (always run first)

```
python3.12 tools/condense/preflight.py
```

Checks Python deps, Rust toolchain, RAM/disk, that every `tools/condense/*.py` compiles, that
`cargo check --workspace` is clean, which model parents are staged, and that the receipt harness
verifies. Exits 0 (green, safe to `go`) or 1 (red, prints exactly what to fix). Do not run `go`
on a red preflight.

## STEP 2 — THE COMMAND

```
python3.12 tools/condense/studio_run.py go
```

Runs the entire frontier program end-to-end, RAM-packed across the 128 GB, continuous, and
resumable (re-run `go` after any interruption — completed models/lanes skip via per-lane floor
files + receipts). Dry-preview first with:

```
python3.12 tools/condense/studio_run.py --go-plan
```

## WHAT `go` DOES (ten phases, automatic — see `docs/plans/quintessential_engine_2026_06_29.md` for the full design)

- **P0 CODEC TRIAGE + STAGE/ADVISE** — one-time: `codec_parallelism.py --catalog` scores every
  candidate codec/kernel design for decode PARALLELISM (not just density) before any Rust build
  time is spent — the direct lesson from the QTIP-on-Metal dead end (serial decode ate the
  bandwidth win). Per-model (inline in P1/P4): `auto_bits.py` + `size_frontier.py` +
  `doctor_registry.py --select` + `arch_coverage.py` recommend the bit format, the serve regime
  (RESIDENT/MOE-PAGED/DENSE-OOC), the auto-composed recovery chain, and which Doctor levers are
  architecture-compatible (dense/SSM/MoE — Mamba2 and RWKV-7 both get their real flat-state math,
  not an approximation) — all before any bake. For MoE frontier models, `expert_cache_policy.py`
  simulates hot-expert cache hit-rate/blended-tok-s across cache sizes so the eventual OOC pager's
  cache size is chosen from a measured sweep, not a guess.
- **P1 CONDENSE** — the bit-floor-vs-scale curve across {0.5B,1.5B,7B,14B,32B} via the Doctor
  registry's auto-composed L0-L6 stack, multiwindow ppl + capability tripwire, one floor receipt
  per model, then the curve fit (H1 descent vs H0 flat).
  -> `reports/cron/bit_floor_curve.jsonl`, `receipts/official/*-floor.json`.
- **P2 SUBBIT** — the sub-1-bit frontier lane (PTQ1.61, residual two-part, codec-native/recover),
  gated per model by `subbit_measure.py` (SUBBIT-0 entropy floor) and, for MoE, `expert_sensitivity.py`.
  -> `reports/cron/bit_floor_subbit.jsonl`.
- **P3 SPEC** — `spec_revive.py` on the condensed substrate (7B) + capstone (32B): lossless-verify
  gate -> capture-retrain the eagle5 head -> acceptance measure -> governor bench (exact-match).
  Density (RAM) x spec (latency) stack multiplicatively.
- **P4 FRONTIER** — the 100B+ research prize (235B-A22B / 405B / 671B / 744B; exact HF ids in
  `BASELINES.md` and `FRONTIER` in `studio_run.py`), serve-oriented since they don't fit the doctor
  budget. Runs on streamed shards (entropy floor + per-expert sensitivity + serve-fit record + the
  auto-composed recovery chain). The native-serve quality + RAM-cliff are the serve build.
- **P5 EVAL + LONG-CONTEXT** — `eval_suite.py` (capability + NIAH) + `ctx_extend.py` (YaRN) +
  `kv_frontier.py` (int2/trellis KV, SSD-paging, SSM) + `kv_hybrid.py` (STKV: exact recall + unbounded reach).
- **P6 BASELINE** — `bench_baselines.py`: the wedge gate vs llama.cpp IQ1_S/IQ2 + MLX-4bit at matched
  effective bpw. WIN iff it beats IQ2 on 7B+; else reframe to portfolio.
- **P7 CLIFF** — `ramcliff_bench.py --all`: RAM-cliff tok/s + energy J/tok — the headline + the
  energy moat. A CLIFF-WIN requires native serve + >10x tok/s + lower J/tok.
- **P8 CODEC** — `codec_bakeoff.py`: STRAND vs QTIP/QuIP#/AQLM at matched bpw (CUDA-locked rivals
  are offline-encode-only; STRAND is the lone Metal-native trellis serve).
- **P9 SYNTH + SCORECARD** — fit both lane curves + the 70B/405B extrapolation, then `scorecard.py`:
  the populated competitive matrix. **Refuses any WIN cell without an R3+ receipt.**
  -> `reports/condense/SCORECARD.md`. **The deliverable.**

## LOCKED CONTEXT — do NOT reopen

- Hardware: this M1 Ultra Studio, 128 GB unified, ~800 GB/s, 8 TB SSD. Metal/MPS only, NO CUDA, no cloud, no 512 GB box. One project owns
  the whole machine, one heavy job at a time (the RAM scheduler enforces it). Wall-clock is FREE,
  plugged in 24/7 — optimize for maximum proof, not speed. bf16 throughout.
- Respect the measured dead-ends: low-rank LoRA plateaus (use full-rank), NO uniform-STE through
  the trellis (codec-aware only), AWQ x residual is a non-win, calib = domain-matched not diverse,
  judge low-bit on 7B+ never on 0.5B. `subbit_admm.py` already re-confirmed NanoQuant is a low-rank
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

1. Does doctor recovery work resident on 128 GB? (every +dr died on the 18 GB box by swap/timeout, not recipe.)
2. Is MoE expert sensitivity non-uniform? (dense was uniform ~3% spread = dead; MoE is a different regime.)

If both pass: build toward the dream — **DeepSeek-V3 671B @ 1.0 bpw = 84 GB served entirely from RAM (RESIDENT, no pager)**
on a single Studio where llama.cpp Q4_K (377 GB) cannot even load. On 128 GB, 235B/405B/671B all fit RESIDENT, no expert pager needed. If recovery fails: density-only,
usable floor ~3.3-3.8 bpw. If expert sensitivity is uniform: fall back to 405B @ 1.34 = 68 GB dense.
0.33/0.5 DENSE is below the information floor — fantasy; only MoE-amortized sub-1 is real.

## THE SERVE-BUILD CRITICAL PATH (the one gate on real wins, in order)

See `docs/plans/quintessential_engine_2026_06_29.md` §"Serve-build critical path" for the full spec.
RE-DERIVED FOR 128 GB: because 235B/405B/671B all fit RESIDENT, the OOC expert pager is NO LONGER on
the critical path for the prize (it is Type-1 dead in the free-RAM regime anyway); it is deferred to
the deep frontier only (744B/1T/3T, SSD-bound). The shortened path:
(1) residual two-part GPU decode parity, (2) all-tensor `.tq` loader, (3) per-expert `.tq` writer +
resident heterogeneous MoE serve, (4) frontier native quality + RAM-cliff RESIDENT (flips P4/P7
GATED->MEASURED), (5) spec-decode governor. [deferred] the OOC pager, only for models > ~112 GB.
Until (1)-(4) land, the size/quality/tps numbers stay honestly GATED.

## STAGING (download on the Studio; `go` skips what is not present)

14B/32B/72B/MoE/100B+ parents are owner-gated downloads (8 TB SSD). Exact HF ids + sizes are in
`BASELINES.md`. `go` runs whatever is staged and skips the rest, so you can start with 7B+14B
present and add 32B/72B/235B-A22B/671B as they land. The 7B substrate + its calib/recovery data
are the baseline that makes P3 (spec) work.
