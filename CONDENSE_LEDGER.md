# CONDENSE_LEDGER.md · /goal condense run · branch condense/run-20260703

Every change this run, with before-to-after measurements. Baseline tag `condense-baseline-20260703`
@ `05df9315`. House style: no em or en dashes · middot separators.

## Bound commands (Phase 0)

- BUILD_CMD · `cargo check --workspace` (Rust) + `python3.12 -m py_compile tools/condense/*.py` (Python)
- TEST_CMD · `cargo test --workspace` (1127 Rust test fns) for any Rust change · `studio_run.py --go-plan`
  byte-diff for any tooling change (the conveyor-wiring oracle)
- COVERAGE_CMD · none available on this box (no cargo-llvm-cov / tarpaulin installed). Substitute, honestly
  stated: coverage held BY CONSTRUCTION. No covered line is deleted and no test is touched, so covered-line
  count cannot fall. Any Rust change is additionally gated on the full suite staying green.
- PERF_CMD · the decode bench is model-gated and Claude-session-contaminated on this M3 Pro (see the
  bench_contamination memory: a live session inflates dec_tps 4-5x). Perf held BY CONSTRUCTION instead: no
  hot-path (crates serve/kernel) edit lands in an autonomous commit. Any candidate that needs one is
  stop-and-report for the Studio, where perf is measurable clean-room. Final tree == baseline, so perf
  delta is exactly zero.
- BUDGET · UNBOUNDED WALL-CLOCK (operator standing directive). Terminate on fixpoint or a stop-and-report
  trigger, never on a clock.
- PERF_RED_LINE · 2x (moot this run: the final tree is byte-identical to baseline).

## Baseline manifest (the denominator)

Owned = excludes `target/`, `vendor/` (audit-only), `scratch/`, `.venv-rwkv/`, `node_modules/`, `.git/`.

| metric | baseline |
|---|--:|
| folders (owned) | 523 |
| files (owned) | 4322 |
| Rust LOC (owned, incl tests) | 400,885 |
| Python LOC (touchable) | 35,291 · condense 13,044 · training 12,544 · bench 9,703 |
| Python LOC (off-limits, audit-only) | tools/strand 15,140 |
| docs LOC | docs 14,363 · prompts 9,715 |
| Rust test fns | 1,127 |
| Rust assertions | 2,811 |
| build | `cargo check --workspace` green (exit 0) |
| oracle | `studio_run.py --go-plan` deterministic, 3x identical, 134 lines |
| tools/condense files | 52 (zero orphans · all wired to studio_run / a shell script / a doc) |
| public-surface hash | `82867d520601be432ff143c57385a6b2ee13a270ed38b3f89439145da7c2fd85` |
| surface components | 2695 Rust pub items · 9 serve routes · documented Python CLIs · STR2 format |

## Determinism (Phase 0 step 3)

`studio_run.py --go-plan` run 3x, byte-identical each time. No flaky tests quarantined (the Rust suite was
not executed per-iteration because no Rust change was committed; it compiles green under
`cargo check --workspace --all-targets`).

## Iterations

### Iteration 1 · class DEAD CODE · crates/hide-backend/src/host.rs::len · REVERTED

- Candidate · the single compiler-flagged dead item that was neither in a frozen `tests/` file nor in the
  perf-moat `qwen_dense.rs`: the private `fn len(&self)` on `GateBook` (host.rs:952), flagged
  `#[warn(dead_code)]` "method `len` is never used", non-`pub` (surface-neutral), perf-neutral.
- Applied · removed the 4-line method.
- GATE · FAILED. `cargo check --workspace --all-targets` went RED (E0599): frozen tests at host.rs:1445 and
  :1450 assert `book.len()`. The `dead_code` lint fired only because the library never calls `len`; the test
  target does. Removing it would require editing a frozen test (invariant 5 breach).
- Action · REVERTED (`git checkout crates/hide-backend/src/host.rs`). Build green again. Tree byte-identical
  to baseline. Net footprint delta 0 (a no-op is a revert, per the loop).
- Class DEAD CODE marked EXHAUSTED. All 5 compiler-flagged items are unremovable: 3 live in frozen `tests/`
  files (swiglu_gemv_b1_parity.rs, cpu_backend_parity_deepseek.rs, and the `use super::*` in llama.rs's
  `#[cfg(test)] mod tests`), 1 is in the perf-moat qwen_dense.rs (deferred to grader), 1 (host.rs::len) is
  used by frozen tests.

## Resumed under grader approval (the keep-going directive)

The operator (the grader) authorized proceeding with 4A, the `tools/condense` behavior-preserving
consolidation, as a strict automated loop. Resumed here. Each family is one iteration and one commit,
verified by a functional byte-equivalence gate (dry-mode stdout + emitted JSON identical), not just a static
check. Any non-path diff or any broken gate is an automatic revert.

### Iteration 2 · class SIBLING FILES · kv_frontier.py + kv_hybrid.py -> kv.py · PASS

- Applied · merged the two long-context KV tools into one `kv.py` with subcommands `frontier` / `hybrid`.
  Identical symbols shared (`kv_bpt`, `OUT`); colliding symbols namespaced verbatim (`run` ->
  `run_frontier`/`run_hybrid`, `_geom` -> `_geom_frontier`/`_geom_hybrid`); each tool's `__main__` preserved
  behind an argv[1] dispatcher that restores the original argv. studio_run.py P5 call-sites updated
  (`kv_frontier.py <a>` -> `kv.py frontier <a>`, likewise hybrid). Not imported anywhere, so no import fixups.
- GATE (all green) · py_compile kv.py + studio_run.py OK · `kv.py frontier --synthetic` stderr BYTE-IDENTICAL
  to the pre-merge `kv_frontier.py --synthetic` · `kv.py hybrid --synthetic` stderr BYTE-IDENTICAL · both
  emitted JSON artifacts (`*_kvfrontier.json`, `*_kvhybrid.json`) BYTE-IDENTICAL · `studio_run.py --go-plan`
  BYTE-IDENTICAL to baseline (134 lines, exit 0) · surface hash HELD (`82867d52..`) · net files 52 -> 51.
- before -> after · files 52 -> 51 (-1) · LOC ~193 -> ~185 (shared kv_bpt/OUT/imports deduped) · capability
  identical · behavior proven identical on every runnable path.

### Iteration 3 · class SIBLING FILES · expert_sensitivity.py + expert_cache_policy.py -> expert.py · PASS

- Applied · merged the two MoE-expert tools into `expert.py` with subcommands `sensitivity` / `cache`.
  Built PROGRAMMATICALLY by a script that copies both bodies byte-verbatim (473 + 106 LOC), safe because a
  collision scan proved 0 top-level name collisions. Each original `__main__` preserved behind an argv[1]
  dispatcher. Call-sites updated: studio_run.py x3 (two `sensitivity` P0/P4 sites at different indents, one
  `cache` P4 site) + doctor_registry.py `expert_alloc.tool` string metadata.
- GATE (all green) · py_compile all tools/condense OK · sensitivity `--help` (full argparse parser)
  BYTE-IDENTICAL to pre-merge · sensitivity body proven VERBATIM by region diff (expert.py 13..482 ==
  expert_sensitivity.py 2..471, ignoring rstrip'd blanks) · sensitivity `--synthetic` runs exit 0
  (unseeded-random, so not byte-diffable; verbatim + parser proof cover it) · cache `--sim` BYTE-IDENTICAL
  (deterministic) · no dangling old-name refs · `--go-plan` BYTE-IDENTICAL to baseline · surface HELD · net
  files 51 -> 50.
- before -> after · files 51 -> 50 (-1, from -2 originals +1 merged) · capability identical · behavior proven
  identical on every runnable + parser-defined path.

## Progress

Baseline 52 -> 50 tools/condense files. Two families folded, both fully verified (functional byte-equivalence
+ verbatim proof). Boundary reached: the collision scan shows every REMAINING family has top-level name
collisions (awq 3, subbit 3, sweep 4, residual 12, doctor 12) requiring non-verbatim symbol renaming, and/or
shell-orchestrator + audit-only (tools/strand) entanglement. Next: probe whether any collision is benign
(identical-constant), else stop-and-report the rest for grader/Studio (where models allow functional
verification). See CONDENSE_AUDIT.md.
