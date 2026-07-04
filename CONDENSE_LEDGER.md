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

## Resumed at the SILVER bar (grader chose "keep going now")

Collision families are merged with the WRAP technique: each tool's body is copied byte-verbatim, indented +4,
wrapped in `def _run_<sub>():` so top-level collisions become isolated function locals. Safe because the
families are proven `global`-free + reflection-free. Verification is STRUCTURAL (verbatim-body substring proof
+ scope-safety + compile + `--go-plan` byte-identical + no-dangling-refs) plus any model-free output-diff a
tool happens to expose. Builder: scratchpad/wrap_merge.py (reads originals; no transcription). sweep (huge
blast) + frontier (live daemons) deferred by decision.

### Iteration 4 · class SIBLING FILES · subbit_ladder + subbit_measure + subbit_admm -> subbit.py · PASS

- Applied · wrap-merged into `subbit.py` (measure / ladder / admm). subbit_measure had 2 direct call-sites
  (studio_run.py:109 and :179, one 8-space one 4-space indent); subbit_ladder + subbit_admm are standalone
  (no programmatic callers). Collisions log/main/SCALE_BITS isolated by wrapping.
- GATE (all green) · all 3 wrapper bodies proven BYTE-VERBATIM (exact substring of subbit.py) · `admm
  --self-test` (deterministic, model-free) BYTE-IDENTICAL pre/post = GOLD-level for that subcommand · all 3
  subcommands dispatch (exit 0) · compile all OK · `--go-plan` BYTE-IDENTICAL · surface HELD · net files
  50 -> 48. No functional dangling refs (call-sites updated; remaining grep hits are prose comments,
  in-file docstrings, and the intentional argv[0] prog-name preservation).

### Iteration 5 · class SIBLING FILES · residual_bake + residual_tq + residual_plus -> residual.py · PASS

- Applied · wrap-merged into `residual.py` (bake / tq / plus). bake + tq are bare top-to-bottom scripts (0
  main-guards, handled by the builder's 0-guard path); plus is main()-based. 12 collisions (B1/B2/BAKER/W/
  bake/log/SRC/...) isolated by wrapping. Real call-site: sweep.py:98 (residual_bake). Cosmetic: STACK
  display row + doctor_registry string. residual_tq + residual_plus are standalone.
- GATE (all green) · all 3 wrapper bodies BYTE-VERBATIM · compile all OK · dispatch routes (exit 0) · no real
  dangling refs (invocation sites updated) · `--go-plan` BYTE-IDENTICAL · surface HELD · net files 48 -> 46.
  Structural bar (no model-free output path on these tools).

### Iteration 6 · class SIBLING FILES · awq_bake + awq_plus -> awq.py · PASS

- Applied · wrap-merged into `awq.py` (bake / plus). awq_bake is a bare top-to-bottom script (loads the
  model at module level); the CALIB semantic collision (bake reads corpus TEXT at import, plus stores a
  PATH) is isolated by the per-wrapper scope. Real call-sites: win7b_watchdog.sh:30, sweep.py:95. Cosmetic:
  STACK row, registry string. awq_plus standalone.
- GATE (all green) · both bodies BYTE-VERBATIM · compile all OK · dispatch routes · no real dangling refs ·
  `--go-plan` BYTE-IDENTICAL · surface HELD · net 46 -> 45.

### Iteration 7 · class SIBLING FILES · doctor_{blockwise,strand,qat,lora,registry} -> doctor.py · PASS

- Applied · wrap-merged 5 tools into `doctor.py` (blockwise / strand / qat / lora / registry). The widest
  blast: 15 exact-string edits across 7 files (audit_ladder.py x3, studio_run.py x4, win7b_watchdog.sh,
  win_test.sh, recovery_sweep.sh, condense_rebench_cron.sh x3, doctor.py registry metadata x4) applied by a
  COUNT-ASSERTED rewire script (aborts if any old string is not found exactly N times). doctor_registry's
  `class RecoveryMethod` + @register decorators localize cleanly (cli-only).
- GATE (all green) · all 5 bodies BYTE-VERBATIM · compile all OK · `registry --list` (model-free) output
  BYTE-IDENTICAL pre/post = gold-level for that subcommand · no real dangling refs across .py + .sh ·
  `--go-plan` BYTE-IDENTICAL · surface HELD · net 45 -> 41.

## Resumed under "not aggressive enough, continue without causing harm"

The prior handoff deferred sweep and frontier. Re-audited both. Frontier stays deferred because the launcher
and keepalive scripts key process identity on `frontier_verifier.py`, and `frontier_conductor.py` imports
`frontier_autopilot.py` by path with long-running supervisor semantics. Folding it would disturb live-run
adoption behavior, not just a filename. Sweep has a model-free renderer and a small rewire surface, so it is
safe to fold.

### Iteration 8 · class SIBLING FILES · sweep_render.py -> sweep.py render · PASS

- Applied · folded the matrix renderer into `sweep.py` as the `render` subcommand, preserving the normal
  sweep entrypoint (`sweep.py --profile here`) byte-for-byte. Internal call-sites updated:
  `sweep.py --go` now invokes `sweep.py render`; `sweep_watchdog.sh` invokes `sweep.py render`; the active
  parameter-sweep plan doc now names the subcommand. Deleted the standalone `sweep_render.py`.
- GATE (all green) · `python3.12 -m py_compile tools/condense/*.py` OK · `sweep.py --profile here`
  stdout/stderr BYTE-IDENTICAL pre/post (289 stdout lines, 0 stderr) · `sweep.py render` stdout/stderr
  BYTE-IDENTICAL to pre-delete `sweep_render.py` (1 stdout line, 0 stderr) · no remaining
  executable or active-doc command refs to `sweep_render.py` outside this audit trail · `studio_run.py --go-plan` green at 134 lines and
  deterministic across reruns · `cargo check --workspace` green (warnings pre-existing) · Rust/tests/assets
  untouched · surface HELD by construction · net files 41 -> 40, tools/condense LOC 13,164 -> 13,157 (-7).

## Progress / stop for the code track

Baseline 52 -> 39 tools/condense Python files (-13, -25%). Seven mergeable families folded: kv, expert (gold) ·
subbit (silver + admm gold) · residual, awq (silver) · doctor (silver + registry gold) · sweep-render
(gold), plus one single-use helper inlined into its only caller. Every commit holds the surface hash / construction-equivalent plus a green 134-line `--go-plan`;
model-free runnable paths were output-diffed byte-identical.

Post-iteration scan:

- `frontier_{verifier,autopilot,conductor}` remains DEFERRED by safety, not convenience. The live launcher
  and keepalive logic key on script identity (`pgrep -f frontier_verifier.py`) and import paths; folding it
  would change supervisor/adoption behavior.
- `codec_parallelism.py` + `codec_bakeoff.py` was probed and REJECTED before commit. A lazy merged
  dispatcher can preserve runtime behavior, but `studio_run.py --go-plan` explicitly prints both old
  command names. Removing either file would force a stale dry plan or a non-byte-identical oracle output.
- Duplicate scan found no auto-deletable tracked code/docs: duplicates are app assets/icons, generated
  Tauri schemas, frozen test fixtures, audit-only STRAND mirrors, or a byte-identical requirements pair now
  listed in CONDENSE_DOCS_REVIEW.md for human review only.

## Resumed for placeholder sweep

### Iteration 9 · class EMPTY PLACEHOLDERS · redundant receipt .gitkeep files · PASS

- Applied · deleted `receipts/failures/.gitkeep` and `receipts/official/.gitkeep`. Both directories already
  contain real tracked receipt JSON (`FAIL-001.json`, `qwen-05b-tq3.json`), so the placeholders are
  redundant. Left `receipts/third_party/.gitkeep` intact because it is the only tracked file preserving the
  documented third-party receipt drop directory.
- GATE (all green) · `python3.12 -m py_compile tools/condense/*.py` OK ·
  `python3.12 tools/condense/receipt_verify.py receipts/official/qwen-05b-tq3.json` OK ·
  `studio_run.py --go-plan` green at 134 lines and byte-identical to the prior oracle ·
  `cargo check --workspace` green (warnings pre-existing) · no tests/assets/docs content touched ·
  net tracked files -2, no behavior surface change.

### Iteration 10 · class SINGLE-USE HELPER · verdict.py -> recovery_sweep.sh inline verdict · PASS

- Applied · inlined the 11-line `verdict.py` formatter into its only executable caller,
  `recovery_sweep.sh`, then deleted the standalone helper. The one active doc reference in
  `docs/plans/hawking_handoff_2026_06_28.md` now points at the inline verdict in the sweep script.
- GATE (all green) · extracted `recovery_sweep.sh`'s `verdict()` function and compared a representative
  invocation (`10 10.2 3.4 TEST`) against the pre-delete `verdict.py` output byte-for-byte · no remaining
  live refs to `verdict.py` · `python3.12 -m py_compile tools/condense/*.py` OK · net
  `tools/condense` Python files 40 -> 39, Python LOC 13,157 -> 13,146 (-11); total `.py + .sh` lines flat
  because the formatter moved into shell.
