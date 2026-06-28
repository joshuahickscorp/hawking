# STUDIO GO — the one-command entry point for the Hawking frontier program

> Paste target: when Hawking is on the Mac Studio, the operator tells Claude Code "go" and Claude
> reads THIS file and runs ONE command. Everything downstream is already built, gated, resumable,
> and continuous. No pauses, no babysitting — the pipeline decides what to do.

## THE COMMAND

```
python3.12 tools/condense/studio_run.py go
```

That's it. It runs the entire frontier program end-to-end, RAM-packed across the 96 GB, continuous,
and resumable (re-run `go` after any interruption — completed models/lanes skip via per-lane floor
files + receipts). Dry-preview the whole program first with:

```
python3.12 tools/condense/studio_run.py --go-plan
```

## WHAT `go` DOES (four phases, automatic)

- **P1 CONDENSE** — `run_all('studio')`: the bit-floor-vs-scale curve across {0.5B,1.5B,7B,14B,32B}
  via the full L0-L6 stack (AWQ/mixed/residual + L6 LoRA-KD + L4 block-QAT + L5 codec-native
  GPTQ-Hessian), multiwindow ppl + capability tripwire, one floor receipt per model, then the curve
  fit (H1 descent vs H0 flat). -> `reports/cron/bit_floor_curve.jsonl`, `receipts/official/*-floor.json`.
- **P2 SUBBIT** — `run_all('subbit')`: the sub-1-bit frontier lane (PTQ1.61 1-bit+outlier, residual
  two-part, 1-bit codec-native / block-QAT / recover), gated per model by `subbit_measure.py`
  (SUBBIT-0 entropy floor) and, for MoE models, `expert_sensitivity.py` (SUBBIT-4 per-expert spread).
  -> `reports/cron/bit_floor_subbit.jsonl`.
- **P3 SPEC** — `spec_revive.py` on the condensed substrate (7B) + capstone (32B): lossless-verify
  gate -> capture-retrain the eagle5 head against the condensed distribution -> acceptance measure
  -> governor bench with the exact-match gate. Density (RAM) x spec (latency) stack multiplicatively.
- **P4 SYNTH** — fit both lane curves + the pre-registered 70B/405B extrapolation.
- **P5 FRONTIER** — the 100B+ research prize (235B-A22B / 405B / 671B / 744B), serve-oriented since
  they do NOT fit the doctor budget. Runs what works on streamed shards (SUBBIT-0 entropy floor +
  per-expert MoE sensitivity + the serve-fit record); the block-wise condense-to-`.tq`, the
  native-serve quality number, and the RAM-cliff tps demo are the serve build (read_strand into the
  serve binary + per-expert `.tq` writer). Skips unstaged models. **These are the primary research target.**

## LOCKED CONTEXT — do NOT reopen

- Hardware: this 96 GB Studio. Metal/MPS only, NO CUDA, no cloud, no 512 GB box. One project owns
  the whole machine, one heavy job at a time (the RAM scheduler enforces it). Wall-clock is FREE,
  plugged in 24/7 — optimize for maximum proof, not speed. bf16 throughout.
- Respect the measured dead-ends: low-rank LoRA plateaus (use full-rank), NO uniform-STE through the
  trellis (codec-aware only), AWQ x residual is a non-win, calib = domain-matched not diverse, judge
  low-bit on 7B+ never on 0.5B. `subbit_admm.py` already re-confirmed NanoQuant is a low-rank
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

1. Does doctor recovery work on 96 GB? (every +dr died on the 18 GB box by swap/timeout, not recipe.)
2. Is MoE expert sensitivity non-uniform? (dense was uniform ~3% spread = dead; MoE is a different regime.)

If both pass: build toward the dream — **DeepSeek-V3 671B @ 1.0 bpw = 84 GB served entirely from RAM**
on a single Studio where llama.cpp Q4_K (377 GB) cannot even load. If recovery fails: density-only,
usable floor ~3.3-3.8 bpw. If expert sensitivity is uniform: fall back to 405B @ 1.34 = 68 GB dense.
0.33/0.5 DENSE is below the information floor — fantasy; only MoE-amortized sub-1 is real.

## STAGING (download on the Studio; `go` skips what is not present)

14B/32B/72B/MoE parents are owner-gated downloads (2 TB SSD). `go` runs whatever is staged and
skips the rest, so you can start with 7B+14B present and add 32B/72B/235B-A22B/671B as they land.
The 7B substrate + its calib/recovery data are the baseline that makes P3 (spec) work.
