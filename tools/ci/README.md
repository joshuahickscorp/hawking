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
