# Hawking Local CI / Hardening Tools

This directory holds non-destructive local validation scripts. They are designed for
agent runs: record evidence, avoid git mutations, and leave a clear recovery trail.

## `preflight.sh`

Local mirror of the CI gate plus optional parity and bench smoke.

```bash
tools/ci/preflight.sh
FAST=1 tools/ci/preflight.sh
SKIP_BENCH=1 tools/ci/preflight.sh
TESTS="greedy_token_only_parity q6k_swiglu_2r_parity" tools/ci/preflight.sh
```

Use `FAST=1` during active development. Use the full version before pushing or when
validating a code path that touches model output, kernels, or quantization.

## `regression_gate.sh`

Enforces that the speed / compression / quality wins do not regress silently. Correctness
is already locked by 193 golden token hashes; this is the missing half. It measures
(reusing `tools/bench/ratios.sh` + `stat` + `jq`), compares against the committed baseline
`tools/ci/baselines/regression_baseline.json`, and **exits non-zero on a breach**.

```bash
# CPU-safe, deterministic — enforces footprint ceilings only (no GPU needed).
FOOTPRINT_ONLY=1 tools/ci/regression_gate.sh

# Full gate — footprint + decode_tps floors + lever argmax-identity floors.
# Needs the release binary + models + a free GPU (skips GPU checks if busy).
tools/ci/regression_gate.sh

# Wait for a busy GPU instead of skipping the GPU checks.
GPU_WAIT=1 tools/ci/regression_gate.sh
```

It is a **category-regression** gate, not a micro-benchmark: floors sit ~10–15% below the
measured warm median because the fresh-process warm-median has a ±several-% noise floor
(see `docs/campaign/test_matrix.md`). It catches a lever being silently disabled
(predec OFF = −46.7%), a quant path regressing, or a quality collapse — without flapping.
A check whose inputs are unavailable is reported SKIPPED, never a false PASS. The baseline's
`pending_not_enforced` block (serve-tps, int4-KV perplexity, instruct-quality) is printed so
gaps are not mistaken for coverage. `preflight.sh` runs the footprint gate always and the full
gate in its bench block; `overnight_hardening.sh` runs it via `RUN_REGRESSION=1` (default on).

Update the baseline only with attended review + fresh warm-median evidence, and record the
change in `docs/campaign/test_matrix.md`.

## `overnight_hardening.sh`

Timestamped unattended runner. It writes all logs and a recovery summary under
`reports/overnight/<timestamp>/`.

```bash
# CPU-only proof pass; safe while another model job owns the GPU.
RUN_GPU=0 tools/ci/overnight_hardening.sh

# Full run; waits for active `hawking generate` jobs by default.
tools/ci/overnight_hardening.sh

# Fast smoke of the runner plumbing only.
RUN_CARGO_CHECK=0 RUN_LIB_TESTS=0 RUN_PARITY=0 RUN_PREFLIGHT=0 RUN_GPU=0 \
  OUT=reports/overnight/smoke-local tools/ci/overnight_hardening.sh
```

Useful knobs:

- `RUN_CARGO_CHECK=0`
- `RUN_LIB_TESTS=0`
- `RUN_PARITY=0`
- `RUN_PREFLIGHT=0`
- `RUN_REGRESSION=0`
- `RUN_GPU=0`
- `RUN_SERVE_SMOKE=0`
- `RUN_MAMBA_SERVE_SMOKE=1`
- `FULL_PREFLIGHT=1`
- `GPU_WAIT=0`
- `MAX_GPU_WAIT_SECS=7200`
- `TRIALS=3`
- `TOK=96`

Recovery:

```bash
ls -td reports/overnight/* | head
sed -n '1,240p' "$(ls -td reports/overnight/* | head -1)/summary.md"
```

Reports are generated evidence and ignored by git. Do not stage them.

## `ssm_serve_smoke.sh`

Serve-path production smoke for SSM models. It starts `hawking serve`, waits for
`/healthz`, reads `/v1/models`, streams one `/v1/hawking/generate` request,
captures `/metrics`, writes a report, and shuts the server down.

```bash
# Default RWKV-7 SSM smoke.
tools/ci/ssm_serve_smoke.sh

# Explicit model and output directory.
OUT=reports/serve-smoke/rwkv7-manual \
  tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf

# Optional mamba2 smoke when the GPU/model slot is free.
OUT=reports/serve-smoke/mamba2-manual \
  tools/ci/ssm_serve_smoke.sh models/mamba2-370m-Q4_K_M.gguf
```

The full `overnight_hardening.sh` run invokes RWKV serve smoke by default after
the SSM benchmark matrix. Set `RUN_SERVE_SMOKE=0` to skip it or
`RUN_MAMBA_SERVE_SMOKE=1` to include mamba2 too.

## `ssm_quality_chat.sh`

VALID instruct quality eval. The raw-`generate` suite (`ssm_quality_suite.sh`) is
explicitly NOT a valid instruct eval (no chat template). This one drives the serve
`/v1/chat/completions` endpoint, which applies the per-arch chat template, so the
model sees a real instruct prompt. It evaluates each model on the same task classes
(retrieval, json, math, instruction, multilingual), grades automatically, and writes
one comparison report. Use it to quantify RWKV-7 instruct quality per class and to
make routing decisions (R3/R5).

```bash
# Default: RWKV-7-SFT vs Qwen-3B, both via chat.
tools/ci/ssm_quality_chat.sh

# One model, more tokens.
MODELS=models/rwkv7-g1-04-sft-Q4_K_M.gguf TOK=128 tools/ci/ssm_quality_chat.sh
```

Per model it starts `hawking serve`, runs the classes, and stops the server (GPU jobs
stay sequential). It is informational (characterizes quality); a hard pass/fail
routing threshold is an owner decision. bash 3.2-safe.
