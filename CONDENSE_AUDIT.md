# CONDENSE_AUDIT.md · /goal condense · branch condense/run-20260703 · 2026-07-03

Baseline `condense-baseline-20260703` @ `05df9315`. House style: no em or en dashes · middot separators.

## 1. Baseline vs final (per metric)

Two families folded, both proven behavior-preserving by OUTPUT byte-equivalence (the gold bar). All other
invariants byte-identical.

| metric | baseline | final | delta |
|---|--:|--:|--:|
| tools/condense files | 52 | 50 | -2 |
| files (owned) | 4322 | 4320 | -2 |
| Rust LOC | 400,885 | 400,885 | 0 |
| Rust test fns | 1,127 | 1,127 | 0 (frozen) |
| public-surface hash | 82867d52.. | 82867d52.. | HELD |
| `--go-plan` oracle | 134 lines | 134 lines | BYTE-IDENTICAL |
| build | green | green | held |

## 2. Invariants · all HELD

1 surface frozen · 2 behavior unchanged (proven per merge, below) · 3 build green · 4 assets untouched ·
5 tests frozen (counts unchanged) · 6 coverage held by construction · 7 docs no loss. Perf: no hot-path
edit; Python tooling only. Flaky: none (oracle deterministic).

## 3. What landed (gold bar · output byte-equivalence)

- Iteration 2 · `kv_frontier.py` + `kv_hybrid.py` -> `kv.py` (frontier/hybrid subcommands). 0 collisions,
  verbatim concat. Proven: `--synthetic` stdout AND emitted JSON BYTE-IDENTICAL both subcommands ·
  `--go-plan` identical · surface held.
- Iteration 3 · `expert_sensitivity.py` + `expert_cache_policy.py` -> `expert.py` (sensitivity/cache). 0
  collisions, programmatic verbatim build (473+106 LOC copied, not retyped). Proven: sensitivity `--help`
  (full argparse parser) BYTE-IDENTICAL · sensitivity body VERBATIM by region diff · cache `--sim`
  BYTE-IDENTICAL · `--go-plan` identical · surface held. Call-sites updated: studio_run x3 + doctor_registry
  string.

Iteration 1 (dead code, host.rs::len) was attempted and REVERTED (a frozen test depends on it). See
CONDENSE_LEDGER.md.

## 4. The boundary: why the gold bar is at fixpoint on this box

Every REMAINING family (awq, subbit, residual, doctor, sweep, frontier) was analyzed. The good news: all are
`global`-free, reflection-free, and NOT entangled with the audit-only `tools/strand` (the earlier
"sweep<->strand" was a substring false positive; strand calls its own sibling `sweep.py`). So the
function-wrap merge technique is STRUCTURALLY safe for all of them.

The blocker is VERIFICATION, not safety:

- 0 collisions is required for verbatim concat (kv, expert). Every remaining family HAS collisions
  (awq 3, subbit 3, sweep 4, residual 12, doctor 12), several SEMANTIC: e.g. `awq_bake.CALIB` reads the
  corpus FILE CONTENTS at import while `awq_plus.CALIB` is a PATH string; `BAKER` is relative vs absolute.
  Concat would silently corrupt. The wrap technique isolates each in its own function scope and fixes this,
  but the merge is then indent-shifted, not byte-verbatim.
- NONE of these tools use argparse, and all need a real model to run their functional path. So the two
  proofs that cleared kv/expert (deterministic dry-mode output byte-diff, and argparse `--help`
  byte-identical) are BOTH UNAVAILABLE here. The strongest proof I can produce on this box is STRUCTURAL:
  verbatim-indented body diff + scope-safety (no global/reflection) + compile + `--go-plan` byte-identical +
  no-dangling-refs. That is rigorous, but it is not the OUTPUT byte-equivalence bar.

So under a strict reading (do not ship a merge whose behavior I cannot output-verify), the gold bar is at
FIXPOINT at 52 -> 50. The remaining ~8-10 files of reduction are real and the technique is known; they need
either (a) an explicit decision to accept the SILVER bar (structural verification) on this box, or (b) the
Studio, where the models make output byte-equivalence checkable exactly as kv/expert were.

## 5. Remaining reductions (ready to execute on approval)

Behavior-preserving via function-wrap, all `global`/reflection/strand clean. Blast = call-sites to update
(all in modifiable files: tools/condense, root/tools/bench/tools/condense shell scripts):

| family | files -> 1 | collisions | approx call-sites | note |
|---|---|--:|--:|---|
| awq | awq_bake + awq_plus (2->1) | 3 (CALIB semantic) | 3 (studio_run, win7b_watchdog.sh, sweep.py) | low blast |
| residual | residual_bake + residual_tq + residual_plus (3->1) | 12 | ~2-3 real | residual_tq has module-exec (lazy under wrap) |
| subbit | subbit_ladder + subbit_measure + subbit_admm (3->1) | 3 (log/main) | ~10 | wide caller set |
| doctor | 5 tools -> 1 | 12 (main/ppl/dev) | ~12 + 3 shell | largest win (-4), widest blast |
| sweep | sweep + sweep_render (2->1) | 4 | very wide (bench/training/docs) | sweep is near-public; defer |
| frontier | 3 live-daemon tools -> 1 | (few) | run_7b_frontier.sh + keepalive | live research daemons; defer |

If all safe families fold: 50 -> ~40 (awq/residual/subbit/doctor). sweep + frontier stay standalone (too
wide / live daemons).

## 6. One line for the grader

Branch `condense/run-20260703`: 2 commits, tools/condense 52 -> 50, both merges OUTPUT-verified, every
invariant held, `--go-plan` + surface byte-identical. The gold verification bar (output byte-equivalence) is
at fixpoint on this box; the remaining ~8-10 files need either an explicit SILVER-bar (structural only) OK or
the Studio for gold. Merge these 2 to main now? And which bar for the rest?
