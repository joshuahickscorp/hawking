# CONDENSE_AUDIT.md · /goal condense · branch condense/run-20260703 · 2026-07-03

Baseline `condense-baseline-20260703` @ `05df9315`. House style: no em or en dashes · middot separators.

## 1. Baseline vs final

Six sibling-file families in `tools/condense/` folded into one subcommand-tool each. Every merge holds the
surface hash and a byte-identical `--go-plan`; every wrapper body proven verbatim.

| metric | baseline | final | delta |
|---|--:|--:|--:|
| tools/condense files | 52 | 41 | -11 (-21%) |
| files (owned) | 4322 | 4311 | -11 |
| tools/condense LOC | 13,044 | ~13,000 | ~0 (reorganized, near-zero deleted) |
| Rust LOC | 400,885 | 400,885 | 0 (untouched) |
| Rust test fns | 1,127 | 1,127 | 0 (frozen) |
| public-surface hash | 82867d52.. | 82867d52.. | HELD every commit |
| `--go-plan` oracle | 134 lines | 134 lines | BYTE-IDENTICAL every commit |
| build | green | green | held |

## 2. Invariants · all HELD

1 surface frozen (hash byte-identical every commit) · 2 behavior unchanged (verbatim bodies + output-diff
where a model-free path existed) · 3 build green · 4 assets untouched · 5 tests frozen (counts unchanged;
the one merge that touched a test dep was reverted) · 6 coverage held by construction · 7 docs no content
lost (only path-reference updates). Perf: Python tooling only, no hot-path edit. Flaky: none.

## 3. Iterations

| # | class | change | verification bar | result |
|---|---|---|---|--:|
| 1 | dead code | hide-backend host.rs::len | gate caught frozen-test dep | REVERTED |
| 2 | sibling | kv_frontier+kv_hybrid -> kv.py | GOLD (--synthetic + JSON byte-identical) | 52->51 |
| 3 | sibling | expert_sensitivity+expert_cache_policy -> expert.py | GOLD (--help + --sim byte-identical, verbatim) | 51->50 |
| 4 | sibling | subbit_{ladder,measure,admm} -> subbit.py | SILVER + GOLD (admm --self-test byte-identical) | 50->48 |
| 5 | sibling | residual_{bake,tq,plus} -> residual.py | SILVER (verbatim + structural) | 48->46 |
| 6 | sibling | awq_bake+awq_plus -> awq.py | SILVER (verbatim; CALIB collision isolated) | 46->45 |
| 7 | sibling | doctor_{blockwise,strand,qat,lora,registry} -> doctor.py | SILVER + GOLD (registry --list byte-identical) | 45->41 |

Technique: 0-collision families concatenated verbatim (kv, expert); collision families wrapped
(`def _run_<sub>():` isolates top-level name collisions as function locals) after proving each is
`global`-free + reflection-free. Every wrapper body verified a byte-verbatim substring of the merged file.
The builder (scratchpad/wrap_merge.py) reads the originals, so no code was retyped. doctor's 15 call-site
edits across 7 files were applied by a count-asserted rewire (aborts unless each old string is found exactly
N times).

Verification bars: GOLD = proved identical program OUTPUT on a runnable model-free path. SILVER = verbatim
body + scope-safety (no global/reflection/module-class hazard) + compile + `--go-plan` byte-identical +
no-dangling-refs across every modifiable .py/.sh. The grader chose SILVER-on-this-box over
GOLD-on-the-Studio for the model-gated families.

## 4. Deferred by decision (not merged)

- `sweep` (sweep + sweep_render): near-public, referenced across tools/bench, tools/training, and many docs;
  the blast radius is too wide to change safely without the Studio.
- `frontier` (frontier_verifier + frontier_autopilot + frontier_conductor): live research daemons launched
  by `run_7b_frontier.sh` / `frontier_keepalive.sh`; do not disturb a running frontier.

Folding these would reach ~41 -> ~37. Recommend doing them on the Studio (frontier idle) with output-diff.

## 5. Docs track

No footprint deletions (docs already at fixpoint; zero byte-identical dupes). Content-accuracy: 8 references
to the old tool names in the two ACTIVE canonical docs (STUDIO_GO.md, quintessential_engine) updated to the
new subcommand form (count-asserted). 8 archival/dated plan docs still carry prose mentions of old names;
these are historical and out of footprint scope, listed in CONDENSE_DOCS_REVIEW.md.

## 6. One line for the grader

Branch `condense/run-20260703`: 8 commits, `tools/condense` 52 -> 41 (-21%), every invariant held every
commit (surface hash + `--go-plan` byte-identical), every wrapper body proven verbatim, four subcommands
additionally output-verified byte-identical. Rust and tests untouched. sweep + frontier deferred to the
Studio. Nothing pushed, nothing on main. Merge `condense/run-20260703` to main?
