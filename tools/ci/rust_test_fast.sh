#!/usr/bin/env bash
# Fast, complete Rust test-source lanes for local iteration and CI.
#
# The routine lanes compile unit/binary tests plus the explicitly selected
# aggregate integration targets. They never execute tests. Every historical
# source file remains an explicit `test = false` target, so focused debugging
# keeps its original process isolation instead of running merged stateful tests.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: tools/ci/rust_test_fast.sh <check|compile|unit|policy|core|verify> [scope or arguments...]

  check [all|core|hide|bake|tq]
           Typecheck the selected test sources without linking or running them.
           Default scope: all.
  compile [all|core|hide|bake|tq]
           Link-check the selected test sources without running them.
           Default scope: all.
  unit [all|core|hide|bake]
           Run only unit and binary tests in the selected scope. Default: core.
  policy
           Run the pure Hawking serving-policy regression suite. It has no
           model, hardware, or daemon dependency.
  core <legacy-test-target> [libtest arguments...]
           Run one Hawking Core integration target in its original isolated
           process. Aggregate targets are deliberately refused here.
  verify   Validate the checked-in test topology without invoking Cargo.
USAGE
}

verify_topology() {
  python3 tools/ci/check_rust_test_topology.py
}

test_targets() {
  local cargo_verb=$1
  shift
  case "$cargo_verb" in
    check) cargo check "$@" ;;
    compile) cargo test "$@" --no-run ;;
    *)
      printf 'internal error: unknown Cargo verb %s\n' "$cargo_verb" >&2
      exit 2
      ;;
  esac
}

core_targets() {
  local cargo_verb=$1
  test_targets "$cargo_verb" \
    -p hawking -p hawking-core -p hawking-serve \
    --features hawking-core/tq \
    --lib --bins --tests --test compile_default --test compile_tq --test compile_all
}

hide_targets() {
  local cargo_verb=$1
  test_targets "$cargo_verb" \
    -p hide-core -p hawking-process-authority -p hide-kernel -p hide-fleet -p hide-protocol \
    -p hawking-context -p hawking-index -p hawking-orch -p hawking-research \
    -p hawking-events -p hawking-speculate -p hawking-adapters \
    --lib --bins --tests --test compile_all
}

bake_targets() {
  local cargo_verb=$1
  test_targets "$cargo_verb" \
    -p q4k_fast_tool -p awq_bake_tool -p tq_bake_tool \
    --bins --tests --test compile_all
}

tq_targets() {
  local cargo_verb=$1
  test_targets "$cargo_verb" \
    -p hawking-core --features tq --lib --tests --test compile_tq
}

all_targets() {
  local cargo_verb=$1
  test_targets "$cargo_verb" \
    --workspace --features hawking-core/tq \
    --lib --bins --tests --test compile_all --test compile_default --test compile_tq
}

run_lane() {
  local cargo_verb=$1
  local scope=$2
  verify_topology
  case "$scope" in
    all) all_targets "$cargo_verb" ;;
    core) core_targets "$cargo_verb" ;;
    hide) hide_targets "$cargo_verb" ;;
    bake) bake_targets "$cargo_verb" ;;
    tq) tq_targets "$cargo_verb" ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

command=${1:-}
case "$command" in
  check|compile)
    shift
    scope=${1:-all}
    test "$#" -le 1 || { usage >&2; exit 2; }
    run_lane "$command" "$scope"
    ;;
  unit)
    shift
    scope=${1:-core}
    test "$#" -le 1 || { usage >&2; exit 2; }
    case "$scope" in
      all) cargo test --workspace --lib --bins ;;
      core) cargo test -p hawking -p hawking-core -p hawking-serve --lib --bins ;;
      hide)
        cargo test \
          -p hide-core -p hawking-process-authority -p hide-kernel -p hide-fleet -p hide-protocol \
          -p hawking-context -p hawking-index -p hawking-orch -p hawking-research \
          -p hawking-events -p hawking-speculate -p hawking-adapters --lib --bins
        ;;
      # The bake packages are binary-only; asking Cargo for a library target
      # makes an otherwise valid lane fail before any test executes.
      bake) cargo test -p q4k_fast_tool -p awq_bake_tool -p tq_bake_tool --bins ;;
      *) usage >&2; exit 2 ;;
    esac
    ;;
  policy)
    shift
    test "$#" -eq 0 || { usage >&2; exit 2; }
    verify_topology
    cargo test -p hawking-serve --test effective_policy
    ;;
  core)
    shift
    target=${1:-}
    test -n "$target" || { usage >&2; exit 2; }
    shift
    case "$target" in
      compile_default|compile_tq)
        printf 'aggregate target %s is compile-only; select a legacy test target instead\n' "$target" >&2
        exit 2
        ;;
    esac
    cargo test -p hawking-core --test "$target" -- "$@"
    ;;
  verify)
    shift
    test "$#" -eq 0 || { usage >&2; exit 2; }
    verify_topology
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
