# Hawking operations

This is the compact operator guide for builds, validation, benchmarks, receipts,
profiles, and offline training. Scientific campaign policy remains in
[`plans/DOCTOR_V5.md`](plans/DOCTOR_V5.md); runtime flags remain in
[`env_flags.md`](env_flags.md).

## Safety boundary

- Treat a detached Doctor supervisor as an active owner. Read status freely, but
  do not replace its binary, source-bound scripts, queue state, runtime specs,
  checkpoints, or evidence.
- Build experimental binaries in an isolated `CARGO_TARGET_DIR`.
- Never turn a component microbenchmark into an ETA or production claim.
- Never stage generated `reports/`, `scratch/`, build outputs, models, or secrets.
- A skipped gate is `SKIPPED`, never `PASS`.
- Promote only at a quiescent checkpoint with exact source, artifact, receipt,
  rollback, disk, memory, and process-ownership bindings.

## Build and local validation

Use the narrowest relevant checks while a heavy owner is present:

```sh
cargo check -p hawking-core -j 2
python3.12 -m compileall -q tools/condense
python3.12 -m unittest discover -s tools/condense/tests
FAST=1 tools/ci/preflight.sh
```

Before a source change is published, run the full applicable tests and:

```sh
tools/ci/preflight.sh
FOOTPRINT_ONLY=1 tools/ci/regression_gate.sh
git diff --check
```

`tools/ci/preflight.sh` is the local CI mirror. The regression gate compares
footprint, throughput floors, and lever identity against
`tools/ci/regression_baseline.json`. Update that baseline only from
fresh attended evidence.

For a CPU-safe proof pass while another process owns the model/GPU:

```sh
FAST=1 tools/ci/preflight.sh
FOOTPRINT_ONLY=1 tools/ci/regression_gate.sh
```

Capture command output below an ignored `reports/` directory when an unattended
record is needed. Machine-readable receipts remain authoritative.

## Benchmarks

Use fixed parameters so results remain comparable:

| Use | Command | Parameters |
|---|---|---|
| development or commit gate | `tools/bench/coexist_bench.sh` | `TRIALS=4 TOKENS=24` |
| phase close | `tools/bench/coexist_bench.sh` | `TRIALS=6 TOKENS=64` |
| authoritative release | `tools/bench/clean_bench.sh` | `TRIALS=10 TOKENS=64` |
| kernel comparison | `hawking bench-kernel` | `--iterations 500` |
| lever A/B decision | `python3 tools/ops.py bench run paired -- …` | matched A/B environments |

The median is the primary metric. Overlapping confidence intervals are not a
measured win. Treat `IQR / median > 15%` as contaminated coexistence evidence
and rerun. Authoritative absolute numbers require an otherwise quiet machine;
paired A/B runs may use coexistence only when both arms share the same envelope.

Example:

```sh
python3 tools/ops.py bench run paired -- \
  --label candidate --env-a "FEATURE=0" --env-b "FEATURE=1"
./target/release/hawking bench-kernel \
  --kernel gemv_q4_k_m_v2_pinned_tcb \
  --shape 1408x2048 \
  --iterations 500
```

## Performance bisect

Confirm a regression with the same `coexist` profile at both revisions, then
choose a threshold roughly one decode token per second below the lower
known-good median:

```sh
git bisect start
git bisect bad HEAD
git bisect good <last-good>
git bisect run tools/bisect_v2_lite.sh 18.5
git bisect reset
```

Every step rebuilds, regenerates an autotune profile, and runs the fixed
four-trial coexistence benchmark. Common culprits are shader drift, profile
drift, dispatch geometry, and Metal buffer alignment.

## Autotune and kernel profiles

Autotune measures dispatch candidates for one model, shader revision, and
device, then writes a profile:

```sh
hawking autotune \
  --weights models/deepseek-v2-lite-q4.gguf \
  --profile m3-pro-18gb \
  --max-hours 8 \
  --out profiles/deepseek-v2-lite-q4.m3pro18.json
```

Use the profile explicitly or through `HAWKING_KERNEL_PROFILE`. A profile binds
the model/tensor layout, device, shader hash, selected schedule, and evidence.
If the shader or tensor-layout hash changes, the runtime warns and falls back to
the deterministic default.

```sh
hawking shader-hash
hawking generate \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  --prompt "Hello" --max-new-tokens 16
```

Regenerate after changing Metal sources, moving machines, or a material
OS/driver update. A profile is advisory until the corresponding output and
performance gates pass.

## Receipts

Receipts make runs independently rerunnable:

```sh
python3.12 -m tools.condense legacy receipt_verify --self-test
python3.12 -m tools.condense legacy receipt_verify receipts/official/*.json
python3.12 -m tools.condense legacy emit_receipt
```

The verifier rejects missing effective bpw, single-window quality, hidden
worst-window results, excessive parent-to-condensed KL, missing source/artifact
hashes, missing commands or commit identity, mislabeled claim types, an MPS-only
headline without CPU confirmation, and best-effort baselines presented as
public wins.

Reproduction levels are:

- `R0`: private;
- `R1`: author rerunnable;
- `R2`: artifact identified and measured;
- `R3`: one-command reproduction on the same machine class;
- `R4`: independent Mac reproduction;
- `R5`: externally adopted format.

`R3` is the minimum public-win bar.

## Serve and quality smoke

The API is summarized in the [handbook](README.md#http-api). For state-space
models:

```sh
tools/ci/ssm_serve_smoke.sh
tools/ci/ssm_quality_chat.sh
```

Quality evaluation must use the chat endpoint so the architecture-specific chat
template is applied. Raw generation is not an instruct-quality claim.

## Training and headbank

`tools/training/` contains offline corpus generation, Eagle speculative-head
training/quantization/evaluation, and AWQ calibration. Outputs belong under
ignored `artifacts/`.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/training/requirements.txt
python3 tools/training/build_corpus.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-sequences 10000 \
  --out artifacts/calibration/v2_lite_corpus
```

`tools/headbank.py` stages a manifest-selected head into
`$HAWKING_HOME/headbank/<slug>/` and emits an environment file:

```sh
python3 tools/headbank.py --manifest <manifest.json> --list
python3 tools/headbank.py \
  --manifest <manifest.json> --slug q7b --env-file ~/.hawking/q7b.env
```

The head, AWQ scales, runtime profile, source commit, and metrics must remain
hash-bound as one unit.

## Generated evidence and recovery

Generated summaries are views over JSON receipts and checkpoints. Recover from
the machine-readable artifacts, not from a copied Markdown report. Closed
incident procedures and old run logs remain in Git history; live recovery tools
must fail closed when their pinned target has completed or drifted.

## Doctor runtime operations

The detached Ultra supervisor and its source-bound runtime generation are the
only execution authority. Read-only observers, Telegram delivery, staging
tools, forecasts, and default-off acceleration modules cannot launch cells,
rewrite results, accept checkpoints, change queue control, or promote a
runtime. Completed cells and evidence are immutable.

### Status and Telegram

A healthy queue status shows a valid supervisor identity, valid state
generation, known active cells, and no structural errors. Telegram is
operational telemetry, never evidence:

```sh
python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py configure-token
python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py discover-chat
python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py prime
python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py send-test
python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py install
python3.12 tools/condense/doctor_v5_telegram_rung_notifier.py status
```

Credentials stay in macOS Keychain. `prime` records existing rungs without
historical spam. Messages should report GOOD/BAD, optimization possible,
density leader, actual/target physical bpw, quality and capability deltas,
Pareto pruning, next-block and overall ETA, wall time, attempts, progress,
pressure, swap, disk, and thermal state.

### Admission and swap

Pending work uses exact evidence rather than nominal model-size guesses:

- process-tree RSS profiles bind plan, request, process membership, and samples;
- exact `(tier, rate)` profiles require real 8/12/16/20-thread candidates;
- uncalibrated heavy profiles remain exclusive;
- packing uses measured CPU/RAM reservations;
- unknown pressure or swap state stops launches.

The swap controller starts from a sealed owner-free baseline. Relative growth
may throttle, stop launches, or request one checkpoint-preserving shed. It
cannot re-baseline upward after a crash, and it never invalidates completed
evidence.

### Elastic execution

A future reviewed generation may schedule one heavy prepare, one primary
encoder, one serial finalizer, and at most one measured-idle companion. A
20-thread encoder is exclusive. Prepare and encode cannot overlap.

Any other overlap requires three fresh ordered local observations binding
topology, exact process identities, leases, state generation, invocation
manifest, RAM, idle CPU, swap, pressure, thermals, and power. Encoder return
closes lending; a companion checkpoints and releases before finalization.
Caller-declared process or resource state is not authority.

### Single-device stack

The default-off acceleration stack contains:

1. exact thread and process-tree RAM selection;
2. bounded read/preprocess/encode/write overlap;
3. one-unit source/preprocessing reuse, never a whole-model cache;
4. elastic phases and measured-idle companionship;
5. native CPU, PGO, mmap/preallocated I/O, and durable partials;
6. bit-exact CPU/Metal RHT probing;
7. reversible priority, QoS, and caffeination controls;
8. randomized full-stack paired A/B authority;
9. phase-aware remaining-scratch accounting.

Synthetic gates prove mechanics only. They cannot change defaults or ETA.

### Disk accounting

Resident evaluation credits reconstruction bytes only when the same shard has
a durable checkpoint-bound encode, attestation, and decode chain with stable
no-follow identity:

```text
remaining_scratch = max(0, declared_scratch - durable_materialized)
remaining_packed = max(0, projected_packed - durable_attested_packed)
required_free = 150,000,000,000 + remaining_scratch + remaining_packed
```

Packed output stays separate from scratch. Existing but uncheckpointed files
count as zero. Ledger output remains read-only until a new queue generation
binds its source and validator.

### Incident recovery

The former 14B/4bpw blocked-cell and 14B/3bpw resource-stop incidents are
closed. Their pinned tools must reject the terminal state. Reusable rules are:

1. pin one incident, cell, attempt, state generation, request, runtime,
   checkpoint prefix, binaries, and evidence;
2. stage outside live state;
3. prove zero supervisor children and external heavy owners;
4. acquire queue, campaign-heavy, and recovery locks;
5. verify completed artifacts after the owner-free gate;
6. compare-and-swap only permitted target fields;
7. preserve attempts, completed units, other rows, and top-level state;
8. durably record intent, rollback, and result;
9. resume through the active hash-verified entrypoint;
10. refuse terminal, stale, ambiguous, symlinked, or drifted state.

### 120B and higher tiers

GPT-OSS 120B is a source-unit graph, not an oversized small-model cell. Its
adapter binds tokenizer, architecture, tensor inventory, shard identity,
source-reading plan, bounded preprocessing, operations, lifecycle, ten rates,
and four Doctor branches. Structural readiness and current disk admissibility
are separate. Completion ETA remains null until required operations,
registries, runtime specs, codec, quality, and lifecycle gates pass.

Models beyond 120B are separate admission campaigns. Each requires immutable
architecture, exact logical parameters, source manifest, streamed storage
plan, tokenizer/adapter review, lifecycle admission, physical receipts, and
rollback. A display label or available download is not authority.

### Final interpretation and promotion

Final interpretation requires every cell terminal, required group reports
complete, reporter checkpoints accepted by queue state, all path/file/receipt/
report/covered-cell hashes matching, and no active children. Freeze exact final
inputs before sealing an interpretation packet.

At the signed quiescent boundary:

1. prove zero heavy owners and stable disk/RAM/swap/power/thermal state;
2. qualify real 8/12/16/20 matrices for every exact tier/rate;
3. re-run native, PGO, I/O, pipeline, reuse, and Metal exactness gates;
4. qualify CPU and GPU separately before stacking;
5. run at least three randomized full-stack baseline/candidate pairs against a
   separately frozen authority;
6. admit only the conservative paired speedup for its exact segment;
7. create a pending-only source-bound runtime;
8. pass admission, crash/resume, rollback, and evidence-integrity tests;
9. promote atomically without changing terminal results.

Sub-120B evidence never automatically authorizes 120B or Appendix ETA changes.
