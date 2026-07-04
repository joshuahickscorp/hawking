# CONDENSE_AUDIT.md · /goal condense · branch condense/run-20260703 · 2026-07-03

Baseline `condense-baseline-20260703` @ `05df9315`. House style: no em or en dashes · middot separators.

## 1. Baseline vs final

Seven sibling-file families in `tools/condense/` folded into one subcommand-tool each, one single-use helper
inlined into its only caller, and one duplicate orchestrator helper removed by reusing the neighboring
implementation. Every merge holds the surface hash / construction-equivalent and a green byte-stable
`--go-plan`; model-free runnable paths are output-diffed byte-identical.

| metric | baseline | final | delta |
|---|--:|--:|--:|
| tools/condense Python files | 52 | 39 | -13 (-25%) |
| files (owned) | 4322 | 4307 | -15 |
| tools/condense Python LOC | 13,044 | 13,146 | +102 cumulative wrapper overhead; -11 this iteration |
| tools/orchestrator pack_corpus.py LOC | 203 | 193 | -10 |
| Rust LOC | 400,885 | 400,885 | 0 (untouched) |
| Rust test fns | 1,127 | 1,127 | 0 (frozen) |
| public-surface hash | 82867d52.. | 82867d52.. | HELD every commit |
| `--go-plan` oracle | 134 lines | 134 lines | BYTE-IDENTICAL every commit |
| build | green | green | held |

## 2. Invariants · all HELD

1 surface frozen (hash byte-identical or held by construction every commit) · 2 behavior unchanged (verbatim bodies + output-diff
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
| 8 | sibling | sweep_render.py -> sweep.py render | GOLD (sweep plan + render byte-identical) | 41->40 |
| 9 | placeholders | redundant receipt .gitkeep files | receipt verify + build/go-plan green | tracked files -2 |
| 10 | single-use helper | verdict.py -> recovery_sweep.sh inline verdict | GOLD (representative verdict byte-identical) | 40->39 |
| 11 | duplicate helper | pack_corpus.py reuses pack_ffn.py quantize_int8 | GOLD (synthetic parquet rows byte-equivalent) | LOC -10 |

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

- `frontier` (frontier_verifier + frontier_autopilot + frontier_conductor): live research daemons launched
  by `run_7b_frontier.sh` / `frontier_keepalive.sh`; the launcher keys live process detection on
  `frontier_verifier.py`, and the conductor imports the autopilot by path. Folding it would change
  supervisor/adoption behavior, not just code organization.
- `codec` (codec_parallelism + codec_bakeoff): rejected before commit. The runtime can be lazily merged, but
  `studio_run.py --go-plan` prints the two old command names. Removing those entrypoints would either make
  the dry plan stale or change the byte-stable go-plan oracle.
- `strand_eval` mirror (`tools/strand/scripts/strand_eval` vs `tools/strand/tools/strand_eval`): exact code
  duplicate, but the package-copy/self-location behavior is documented by frozen tests. The live launcher
  path uses `tools/strand/tools/strand_eval`; replacing the scripts-side package with a shim would change the
  copied-package contract rather than just remove duplication.
- Generated Tauri schemas: `desktop-schema.json` and `macOS-schema.json` are byte-identical today, but their
  distinct platform names are generated schema support. A build cannot prove editor/schema consumers, so the
  duplicate stays.
- STRAND rung config mirror: the two `rung-attn4-ffn3.json` files are semantically identical after JSON
  normalization, but both are plausible user-facing CLI input paths.
- `w4a8_activation_dist.csv`: report-looking, but actively read by `tq_output_space_quality.rs`; treated as a
  frozen fixture.
- `tools/training/build_corpus.py` duplicate hook helpers: real internal duplication, deferred because the
  behavior is model-forward capture and async tensor transfer with no cheap oracle.

Folding these would reach ~40 -> ~37. Recommend doing frontier only when the frontier is known idle, with a
compatibility shim or launcher rewrite accepted by the grader. Recommend doing codec only if the grader
accepts a planned `--go-plan` text change or keeps compatibility shims despite no file-count win.

## 5. Docs track

No markdown footprint deletions (docs already at fixpoint; zero byte-identical markdown dupes).
Content-accuracy: 8 references to old tool names in the two ACTIVE canonical docs (STUDIO_GO.md,
quintessential_engine) updated to the new subcommand form (count-asserted), plus 2 active parameter-sweep
references updated from `sweep_render.py` to `sweep.py render`. One byte-identical non-md requirements pair
is staged for human review only in CONDENSE_DOCS_REVIEW.md.

## 6. Stop reason

Safe local fixpoint for this autonomous pass. A full tracked duplicate scan found only no-touch assets,
generated schemas, frozen fixtures, audit-only STRAND mirrors, and the requirements pair staged for human
review. The remaining code-fold candidates either disturb live supervisor identity (frontier), break the
byte-stable go-plan oracle (codec), change a copied-package/self-location contract (`strand_eval`), or lack a
model-free behavioral oracle (`build_corpus.py` hook refactor). One duplicate orchestrator quantizer helper
was removed with a synthetic parquet oracle. Redundant `.gitkeep` placeholders in non-empty receipt
directories were removed; the sole `receipts/third_party/.gitkeep` remains because it preserves a documented
empty drop directory. `verdict.py`, a single-use helper, was inlined with byte-identical output. No tests,
assets, generated schemas, fixtures, or docs were deleted.

## 7. One line for the grader

Branch `condense/run-20260703`: 14 commits since baseline, `tools/condense` Python files 52 -> 39 (-25%),
tracked files 4322 -> 4307 (-15), every
invariant held every commit (`--go-plan` green at 134 lines, model-free paths byte-identical, Rust build
green), Rust and tests untouched. frontier and codec remain deferred for the safety reasons above. Nothing
pushed, nothing auto-merged. Merge `condense/run-20260703` to main?
