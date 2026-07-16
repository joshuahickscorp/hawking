# Contributing

Hawking is owner-maintained, but focused fixes and evidence-backed reports are
welcome.

## Required checks

Use an isolated target directory when a detached model run owns the machine:

```sh
cargo fmt --all -- --check
CARGO_TARGET_DIR=/tmp/hawking-target cargo test --workspace --no-run
FAST=1 CARGO_TARGET_DIR=/tmp/hawking-target tools/ci/preflight.sh
python3 tools/ops.py selftest
git diff --check
```

Run focused tests for every changed behavior. Kernel changes require a CPU or
reference-path parity gate. Approximate paths require a declared quality gate;
a skipped gate is never a pass.

## Performance changes

Measure A and B under the same model, prompt, output length, profile, machine
state, and cold/warm policy:

```sh
python3 tools/ops.py bench run paired -- \
  --label my-lever \
  --env-a "HAWKING_EXAMPLE=0" \
  --env-b "HAWKING_EXAMPLE=1"
```

Defaults change only after repeatable correctness, quality, and whole-path
performance evidence.

## Pull requests

- Keep one concern per change.
- Keep new levers default-off until their gates pass.
- Include the commands and evidence supporting the result.
- Do not hide warnings with broad `allow` attributes.
- Preserve artifact, receipt, failure, and rollback semantics.
- Keep generated reports, models, caches, and credentials untracked.

Contributions are MIT-licensed under [LICENSE](LICENSE).
