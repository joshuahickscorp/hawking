#!/usr/bin/env bash
# tools/ci/preflight.sh — run the CI gate locally before pushing.
# Mirrors .github/workflows/ci.yml (fmt · clippy · build · compile-tests) and adds a
# parity-test subset + a warm bench smoke (tools/bench/ratios.sh).
#
#   tools/ci/preflight.sh            # full
#   FAST=1 tools/ci/preflight.sh     # fmt + clippy + build + compile-tests only (no test-run, no bench)
#   SKIP_BENCH=1 tools/ci/preflight.sh
#   TESTS="greedy_token_only_parity q6k_swiglu_2r_parity" tools/ci/preflight.sh   # override the subset
set -uo pipefail
REPO="${REPO:-$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)}"; cd "$REPO" || exit 2
fail=0
step() { printf "\n\033[1m== %s ==\033[0m\n" "$1"; }
run()  { echo "+ $*"; if "$@"; then :; else echo "FAILED: $*"; fail=1; fi; }

step "pinned source packs"
PYTHON="${PYTHON:-python3}"
if [ "${HAWKING_PACK_OFFLINE:-0}" = 1 ]; then
  "$PYTHON" tools/hawking_packs.py fetch --offline || exit 1
else
  "$PYTHON" tools/hawking_packs.py fetch || exit 1
fi
"$PYTHON" tools/hawking_packs.py hydrate || exit 1
"$PYTHON" tools/hawking_packs.py verify || exit 1
"$PYTHON" tools/hawking_packs.py validation || exit 1

step "fmt --check"                      # CI: `cargo fmt -- --check` (workspace members only, not vendor/)
[ "${SKIP_FMT:-0}" = 1 ] || run cargo fmt -- --check

step "clippy (exact CI allowlist)"
[ "${SKIP_CLIPPY:-0}" = 1 ] || run cargo clippy --workspace -- -D warnings \
  -A unexpected_cfgs -A unused_assignments -A clippy::should_implement_trait \
  -A clippy::type_complexity -A clippy::too_many_arguments

step "build --workspace"
[ "${SKIP_BUILD:-0}" = 1 ] || run cargo build --workspace

step "compile all tests (--no-run; a broken test can't hide behind a skip)"
[ "${SKIP_BUILD:-0}" = 1 ] || run cargo test --workspace --no-run

if [ "${FAST:-0}" != 1 ]; then
  step "parity subset (release; model/GPU gates skip cleanly without weights)"
  T="${TESTS:-greedy_token_only_parity integration_greedy_64 gemm_q4k_v4r_predec_parity q6k_swiglu_2r_parity q6k_swiglu_4r_parity mha_decode_perchannel_int4kv_parity event_horizon_parity_prop}"
  for t in $T; do run cargo test -p hawking-core --release --test "$t"; done

  step "regression gate — footprint (CPU-safe; enforces compression floors)"
  [ -x tools/ci/regression_gate.sh ] && run env FOOTPRINT_ONLY=1 tools/ci/regression_gate.sh

  if [ "${SKIP_BENCH:-0}" != 1 ]; then
    if [ -x ./target/release/hawking ] && [ -f models/qwen2.5-3b-instruct-q4_k_m.gguf ]; then
      step "regression gate — full (warm tps + quality floors, enforced)"
      run tools/ci/regression_gate.sh
    else
      echo "(bench/regression smoke skipped — need ./target/release/hawking + the Qwen-3B gguf)"
    fi
  fi
fi

step "RESULT"
if [ "$fail" = 0 ]; then echo "preflight PASS"; else echo "preflight FAIL (see above)"; exit 1; fi
