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
cd "${REPO:-$HOME/Downloads/hawking}" || exit 2
fail=0
step() { printf "\n\033[1m== %s ==\033[0m\n" "$1"; }
run()  { echo "+ $*"; if "$@"; then :; else echo "FAILED: $*"; fail=1; fi; }

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

  if [ "${SKIP_BENCH:-0}" != 1 ]; then
    if [ -x ./target/release/hawking ] && [ -f models/qwen2.5-3b-instruct-q4_k_m.gguf ]; then
      step "bench smoke (warm tps + adversarial quality, tools/bench/ratios.sh)"
      run ./tools/bench/ratios.sh ab "" short 3
      PROFILE=fast run ./tools/bench/ratios.sh ab "" short 3
      run ./tools/bench/ratios.sh qual "HAWKING_QWEN_F16_KV=1" "" 60
    else
      echo "(bench smoke skipped — need ./target/release/hawking + the Qwen-3B gguf)"
    fi
  fi
fi

step "RESULT"
if [ "$fail" = 0 ]; then echo "preflight PASS"; else echo "preflight FAIL (see above)"; exit 1; fi
