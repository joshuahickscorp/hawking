# aud19 — harness consolidation (discovery only)

HEAD `04193ccbc`. This pass does not implement, move, merge, or delete
anything. Every capability state below is the strongest state the source
and call sites actually support. Nothing here is `MEASURED`: no GPU lease,
no `cargo test`, no `--measure` run.

The question per family: is the scientific variable data/config, or does it
need new executable structure? If config suffices, the scripts should become
one harness + a parameterized spec + a receipt.

Machine-readable twin: `receipts/audit/aud19-harness-consolidation.json`.

## What already exists (do not mint a fifth spec)

Four experiment-shaped authorities are already live, plus a scaffold:

| Authority | Schema | State | Who actually calls it |
|---|---|---|---|
| `lab.bench_harness` | `hawking.lab.experiment.v1` | **TESTED** | `tools/bench/*.sh` exec `python3.12 -m lab.bench_harness run <spec>`; `tools/foundry/tests/test_foundry_tables_lifecycle.py` calls `Runner(...).run()` |
| `lab.spec` + `lab.runtime` | `hawking.lab.experiment_spec.v1` | **TESTED** | campaign engine; `tools/condense/tests/test_campaign_engine.py` calls `load_spec_path` |
| `tools/future/experiment_receipt.py` | `hawking.experiment.receipt.v1` | **TESTED** | `attach()` is called from `ba_delta_ab.py:177`, `capability_eval.py:631`, `lpc_baselines.py:536`, and `test_experiment_receipt.py`. Live complete-token A/B sidecars do **not** call it |
| `tools/accelerator/accelerator_runner.py` | `AcceleratorExperimentSpec` | **TESTED** | `test_accelerator_runner.py` |
| `crates/hawking-research/src/experiments.rs` | `ExperimentRun` | **SCAFFOLDED** | **no callers** of `ExperimentRun::new` outside that file |

`lab.bench_harness.ExperimentSpec` and `lab.spec.ExperimentSpec` share a class
name and are not the same type. Unifying the names without unifying the
engines would hide a second authority.

H-ROADMAP.md is **not in this repository** (README points at
`/Users/scammermike/Downloads/H-ROADMAP.md`). As DATA, P13 asks to “add
parameterized experiment specs to the hot executor so a new NR hypothesis
does not require a bespoke executable.” The runners exist. The science
one-offs did not move onto them. That is `STALE_ROADMAP_TEXT` relative to
the science families, not evidence they already did.

**Target if an implementation lane is opened:** reuse `lab.bench_harness` as
the runner and `hawking.experiment.receipt.v1` as the envelope. Per-family
adjudication stays a plugin. Do not build on `hawking-research::ExperimentRun`.

## Loud surprises

These are the findings that would be expensive to smooth over.

1. **`dispatch_size_sweep.rs` is not in HEAD.** Sidecar
   `tools/future/dispatch_size_sweep.py` and sealed receipt
   `receipts/future/DISPATCH_SIZE_SWEEP.json` name
   `crates/hawking-core/examples/dispatch_size_sweep.rs` via
   `Qwen38HybridDecodeSession::measure_dispatch_size_sweep`. `git cat-file`
   fails for the example. `git grep measure_dispatch_size_sweep` over
   `crates/` is empty. The receipt still records verdict
   `PER_DISPATCH_SIZE_REFUTED` / `DIAGNOSTIC_RELATIVE`. **What would settle
   it:** restore the producer, retarget `source`, or mark the receipt stale.
   Do not migrate the sidecar while it names a missing binary.

2. **`tools/future/kernel_geometry.py` is not in HEAD.**
   `receipts/future/KERNEL_GEOMETRY.json` still has
   `bench.recorded_by = tools/future/kernel_geometry.py`.

3. **`mlp_region_falsifier.rs` is not in HEAD.**
   `tools/future/mlp_region_falsifier.py` still looks up that example name.

4. **Prefill GEMM described in `docs/PREFILL.md` is not in this HEAD.**
   The doc (dated 2026-09-02 in-file) lists `qwen38_hybrid_prefill.rs`,
   `shaders/qwen38_prefill.metal`, `HAWKING_QWEN38_PREFILL_CHUNK`, and names
   branch `grok/prefill-gemm-20260901-232724` / commit `8391a0ef8`. This
   checkout is `04193ccbc`. The env var greps to that doc only.
   **What would settle it:** `git show 8391a0ef8 --stat` / confirm the
   branch never landed. A chunk-sweep harness against this tree would be
   fiction.

5. **`capability_curve.bind_workunit` looks for the wrong symbols.**
   It getattr-s `run_one` / `measure_one` / `measure` / `point`.
   `perturbation_workunit` exposes `perturb` and `run`. The only
   `bind_workunit` test monkeypatches `ImportError`. The curve receipt
   path sets `live_sweep.ran = False`.
   **What would settle it:** `bind_workunit()` against the real module
   (expected today: `CurveRefused`).

6. **Fourteen copies of `def example_binaries()`** in `tools/future/`.
   The scientific variable is the example name and argv. The
   locator / `gpu_lane_lock` / raw JSON / record loop is copy-paste. Two
   of those names point at rust examples that are not in HEAD (items 1 and 3).

7. **“Odyssey transfer” is four sciences** that share a substring. Merging
   them would be the import-inflation failure mode applied to filenames.

## Families

### 1. Kernel geometry sweeps — `CALLABLE`, migrate **PARTIAL**

**Scripts.** `geometry_table_sweep.rs` + `geometry_table.py` (already a
measure/adjudicate pair). `dispatch_size_sweep.py` (rust example **ABSENT**).
Q80 `ascension_qwen80_matvec_geometry_sweep.rs`. Four DSV4
`gravity_deepseek_v4_*_sweep.rs`. MLP sidecars `mlp_issue_rate_ladder.py`,
`mlp_stream_count.py`. Archived `hawking-experiments/superwave/g1/..._sweep.py`
is **SUPERSEDED**.

**What varies.** Threadgroup, rows-per-tg, warmup, reps, organ shape: **config**.
Shaders, CPU oracles, “never materialize dense weights”, P0 gate C1–C7
candidates: **structure**.

**Clone ratios (set-line, SOURCE_INSPECTION).** geometry_table vs Q80 matvec
0.060; vs any DSV4 sweep 0.020–0.033. DSV4 act_quant vs split-K 0.191. These
are not the same program with different constants.

**Call sites (not imports).** `geometry_table.py:run_example` subprocesses the
rust example under `gpu_lane_lock.sh`. `lab/operators/deepseek_v4_gravity.py`
calls `_child_baseline_v3_simdgroup_sweep`. 
`test_geometry_table.py` / `test_dispatch_size_sweep.py` invoke the python
adjudicators. The rust examples naming themselves are not external callers.

**Proposed spec.** `hawking.lab.experiment.v1` matrix over
`(example, organ, rows, cols, dtype, tg, rows_per_tg)` + required adjudicator
plugin. Extract the sidecar locator. **Do not merge DSV4 byte-exact sweeps
into the geometry table.**

**Would delete.** Copied `example_binaries`/`run_example` in the python
sidecars, after a shared runner exists. Not the rust examples.

**Risk.** Dropping a CPU-oracle byte compare while folding DSV4 into a
generic sweep. Migrating `dispatch_size_sweep.py` while it cites a missing
binary freezes a broken producer.

**Protecting tests.** `tools/future/test_geometry_table.py`,
`test_dispatch_size_sweep.py`,
`tools/accelerator/test_physical_qualification.py::test_flash_p6_act_quant_simdgroup_has_a_byte_exact_control`.

### 2. Representation A/B — `CALLABLE`, migrate **YES** for live A/B

Three clusters, not one.

**Live complete-token A/B (config + small plugin).**
`ascension_qwen38_deltanet_widen_ab.rs` vs `fold_addqx_token_ab.rs` share
0.620 of unique lines. Python sidecars share 0.481 and twenty function
names; nine of those functions are line-identical. What varies: kernel
names, fusion flags, parity predicate (token-id vs named-buffer bytes).
Argmax is refused in both. Tests call the adjudicators
(`test_deltanet_widen_ab.py`, `test_fold_addqx_ab.py`).

**Live isolated-kernel A/B (config + shader plugin).**
`mlp_aux_merge_ab.rs` / `aux_u8_ab.rs` / `aux_u8_lut_ab.rs` share 0.419–0.683
of unique lines. Same process, two or three arms, warmup/spread/byte identity.

**Receipt-from-raw (do not merge).** `ba_delta_ab.py`, `bitcast_dequant_ab.py`,
`dequant_hoist_ab.py`, `q4_bitcast_ab.py`, `intra_token_concurrency_ab.py`
share 0.03–0.05 of lines with the live A/B sidecars. They reseal sealed
numbers; they do not launch the rust examples. Only `ba_delta_ab.py` calls
`experiment_receipt.attach`.

**CPU codec screens (config, untested).** `lab/operators/q80_rank_bits_sweep.py`,
`q80_representation_frontier_sweep.py`, `q80_residual_encoding_sweep.py`,
`qwen38_bpw_descent_sweep.py` share `BAR=0.8604`, `LE_PAIRS`, allowance, X
cache. **No `test_q80_*sweep` file exists in HEAD.** Do not migrate until
tests exist that fail when BAR / pairs / codec chain is gutted.

**Already a plugin lab.** `representation_lab.py` is **TESTED**, measures
nothing (`STATIC_ONLY` / `FUNCTIONAL_SIM` round-trip). Keep it.

**Proposed spec.** One complete-token A/B harness with required
`parity ∈ {token_ids, named_buffers}` and `argmax_is_not_parity: true`.
One isolated-kernel A/B harness. One CPU codec-screen spec later.

**Would delete.** Duplicate python helpers / CLI in the live A/B pairs.
Not the Metal shaders. Not the receipt-from-raw modules.

**Risk.** A shared harness that defaults parity to argmax reopens the hole
`DELTANET_MULTISTEP` / fold tests refuse. Guard every path that can weaken
that check.

### 3. Context-length sweeps — `CALLABLE`, migrate **ALREADY ONE MATRIX**

`tools/headless/prefill_kv.py` already has `LENGTHS = ("short", "4k", "16k",
"long")`. `long` (32K) is **derived** from measured 16K, not walked.
`latency_ledger.py` actually invokes `python3 tools/headless/prefill_kv.py
--measure` (line 672) and the reuse path (line 924). Tests:
`test_prefill_kv.py`.

`context_budget_probe.py` has no external python caller (CAPABILITY_GRAPH
mentions are catalogue). `hcli/test_long_context_runtime.py` and
`tools/acceptance/context/` are product/acceptance, not science sweeps.
`crates/hawking-bench/src/suites/prefill.rs` hardcodes a ~1024-token
pangram.

**Proposed spec.** Lift `LENGTHS` into `hawking.lab.experiment.v1` and feed
both `prefill_kv` and hawking-bench prefill. Do not invent a second runner.
Do not label derived 32K as measured.

### 4. Prefill chunk sweeps — `STALE_ROADMAP_TEXT`, migrate **NO**

There is no family of chunk-sweep experiment scripts in HEAD.

What **does** exist: `hawking-serve` `prefill_chunks_token_budgeted` /
`commit_prefill_chunk`, called from `crates/hawking-serve/src/lib.rs:811`
and from inline tests in `scheduler.rs`. That is product batching.

What **does not** exist: `qwen38_hybrid_prefill.rs`, `qwen38_prefill.metal`,
`HAWKING_QWEN38_PREFILL_CHUNK` as a code symbol. Historic G117–G119 receipts
do not restore those files.

If GEMM prefill lands, chunk size 1..=64 is **config** and belongs on the
same spec as context-length, not a new executable.

### 5. Resident throughput — `CALLABLE`, migrate **PARTIAL**

Measurement producers: `resident_reprofile.py` (tested),
`measure_genesis_resident.py`. Arithmetic: `resident_token_budget.py`,
`harness_reconciliation.py` (decomposes two walls into GPU vs host; **not**
a script consolidator). Product: `resident_install/health/supervisor/api`.
`hawking-bench` throughput suite is **SCAFFOLDED** (`phase4_pending: true`).

**Proposed spec.** One complete-token measure with required `timing_label`
and wall-TPS-as-range when host moves (the rule
`harness_reconciliation.py` already installed). Do not merge install/health
into it. Do not treat the bench stub as a harness.

**Risk.** Collapsing `DIRTY_ENGINEERING` and `PROTECTED_WINDOW` into one TPS.

### 6. Perturbation experiments — `CALLABLE`, migrate **ALREADY ONE**, wiring broken

`perturbation_workunit.py` is already parameterized over
`tensor ∈ {gate,up,down}`, layer, side, fraction,
`kind ∈ {zero,quantize,noise,low_rank}`. `test_perturbation_workunit.py`
**calls** `pw.perturb(...)` for every kind (matched-fraction, copy-not-mutated).
`main()` calls `run()`. `hcli_sovereign.py:458` subprocesses the CLI.

`COMPONENT_PERTURBATION.json` is a **contract** (`--build`), not a measured
curve (no `points` / `curve` keys). Replay is not exercised in tests.

Do not write a second perturbation executable. Fix `bind_workunit` to call
`run()` (surprise 5). `capability_byte_elimination.py:1081` already calls
`capability_curve.sweep` with its own measure — that is a real `sweep` call
site, not a `perturbation.run` call site.

### 7. FPGA simulation — `TESTED` + `BLOCKED_HARDWARE`, migrate **NO**

`fpga_engines.py` (bit-exact numpy golden models, no HDL) and
`fpga_fidelity.py` (ladder ANALYTICAL → FUNCTIONAL_SIM → CYCLE_MODEL;
HW_EMULATION / REAL_HARDWARE `UNIMPLEMENTED`, tests assert they raise:
“no emulation seat” / “no U50 board”). This machine has no FPGA/U50DD.

They are already the parameterized FPGA school. Do not merge engines with
the estimator. FPGA stays inside Accelerator/Fusion. No Era VI. Coefficient
table is `v1-assumed-not-measured`.

### 8. HBM mappings — `TESTED` + `BLOCKED_HARDWARE`, migrate **ALREADY ONE**

`hbm_doctor.py` is the only `hbm*.py`/`hbm*.rs` in HEAD. Budget is already
`--budget-bytes` (default 8 GiB). `test_hbm_doctor.py` **calls** `hd.build()`,
locks `selected_ids == []` because undecidable is the decision, and refuses
numeric hardware fields. FPGA latency is a class, never a number.

Do not write a second mapper. Do not default missing scores to zero
(the anti-pattern the tests lock is “fill HBM with the largest items”).

### 9. ModelLake policy tests — `TESTED`, migrate **table-driven tests**, not a science harness

Nine `tools/odyssey/test_modellake*.py` files plus product modules. They
**call** `ml.admit`, `guard_protected`, promote-on-`tmp_path`, index
build/query (the index test’s own docstring requires call sites, not
imports).

This is product policy, not a sweep of a scientific variable. Optional
pytest parametrize over `(policy, fixture lake, expected)`. Do **not**
point anything at `/Volumes/corpdrive`. Keep the AST admission-cap test
and the G010 red gate as structure. `_tiny_safetensors` is duplicated
across index and lineage fixtures — that is the only config-shaped delete.

### 10. Odyssey transfer tests — `TESTED`, migrate **NO, do not merge homonyms**

| Script | Actual variable |
|---|---|
| `tools/future/odyssey2_transfer.py` | Odyssey II law-store trial onto another sealed specimen (`STATIC_ONLY`). **Already the harness.** Tests call `build`, `run_transfers`, `transfer_values_l1_to_falcon` |
| `tools/odyssey/transfer_rehearsal.py` | Rehearse inheritance **fields** of a report |
| `tools/odyssey/cold_vs_transfer.py` | Cold start vs transfer |
| `tools/headless/doctor_transfer.py` | Doctor technique library on a new specimen |
| `tools/headless/dense_subbit_transfer.py` | GLM 0.167 BPW method onto dense Qwen organs |

`test_odyssey2_transfer.py` asserts `similarity_score is None` and that
citations are copied measurements. A unified “transfer harness” would let a
similarity score stand in for a failed value transfer.

## What to do first (still not this lane)

1. Shared sidecar runner for the 14 `example_binaries` copies, with a test
   that fails if a spec names a rust example `git cat-file` cannot see
   (would have caught surprises 1 and 3).
2. Complete-token A/B: deltanet_widen + fold_addqx (highest clone ratios,
   tests already call the adjudicators).
3. Isolated-kernel A/B: mlp_aux_merge + aux_u8 + aux_u8_lut.
4. Alias `perturbation_workunit.run` into `capability_curve.bind_workunit`
   instead of writing a new perturbation harness.

Do not: merge DSV4 sweeps into geometry_table; merge receipt-from-raw A/B
into live A/B; merge FPGA engines with fidelity; merge Odyssey “transfer”
homonyms; build on `ExperimentRun`; certify `docs/PREFILL.md` as a chunk
sweep; write `MEASURED` for any of the above.

## Authority limits

GPU times, token-identity of sealed A/B receipts, and HBM/FPGA board
numbers in existing receipts were **not re-run**. They are cited as
`SOURCE_INSPECTION` / `STATIC_VERIFICATION` of files and call sites.
Hardware present: Apple M3 Ultra, ANE. Absent: FPGA/U50DD, DGX, eGPU.
