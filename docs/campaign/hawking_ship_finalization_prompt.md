# Hawking Ship Finalization Prompt

This is the maximal operating prompt for taking Hawking from a powerful research
runtime to a shippable product line. It supersedes the older Claude handoff
prompts in this directory for future autonomous work.

Use this file as the source prompt. The short goal prompts in
`hawking_ship_goal_prompts.md` should tell the agent to read this file first,
then execute the requested lane. Keep the short prompts thin; keep the real
policy, state, and sequencing here.

## Mission

Finalize Hawking as three stacked products:

1. **P1 - Runtime:** an Apple-Silicon-first Rust/Metal inference runtime with a
   correct, fast OpenAI-compatible server, robust model loading, regression
   gates, packaging, and user-grade docs.
2. **P2 - Model Press / Condense:** a memory-budgeted,
   reproducible bake -> validate -> publish pipeline that creates Hawking
   artifacts with recorded recipes, quality cards, and enforced
   footprint/speed/quality gates, including models too large to quantize by
   loading the parent fully on the user's machine.
3. **P3 - Hawking Lab:** public Hugging Face releases of pre-quantized and
   compress-then-distill models, with model cards, a demo, a leaderboard, and
   reproducible recipes.

Do not treat "ship" as a single feature landing. Shipping means that the runtime
can be installed, can serve launch models correctly and fast, can publish
compressed artifacts reproducibly, and can prove quality and performance claims
with automated gates.

## Read First

Read these before choosing or editing anything:

1. `docs/plans/hawking_shippability_masterplan_2026_06_22.md`
2. `docs/campaign/project_standing_snapshot.md`
3. `docs/campaign/open_risks_and_gates.md`
4. `docs/campaign/test_matrix.md`
5. `docs/campaign/autonomous_run_log.md`
6. `docs/campaign/commit_plan.md`
7. `docs/campaign/pruning_inventory.md`
8. `docs/campaign/kill_ledger.md`
9. `docs/env_flags.md`
10. `docs/architecture.md`

If any status conflicts, prefer the newest dated evidence and command output.
Never carry forward an old prompt's state without checking these files.

## Current Ground Truth

Runtime:

- Qwen2.5-3B Q4_K_M `generate` is the mature short-context/default transformer
  path.
- `predec` is the real Qwen decode win and is default-on; disabling it costs
  about 46.7 percent warm throughput.
- `--profile fast` is only a small speed-priority option, roughly +3 to +7
  percent at the bench noise floor, with a mild quality trade.
- `F16_KV` is a clean long-context footprint option: about -50 percent KV, high
  quality, little short-context speed change.
- RWKV-7 0.4B SSM single-stream `generate` is the long-context moat: roughly
  110 to 119 tps flat out to about 8k, versus Qwen collapsing to about 8.6 tps
  at about 8k.
- RWKV serve correctness is fixed end-to-end as of 2026-06-22:
  admission fixed, recurrent state handoff fixed, stream terminator fixed, and
  `ssm_serve_smoke.sh` is fail=0 with coherent output.
- The new top runtime gap is RWKV serve throughput: the B=8 multiseq arena does
  8-stream work for 1 active stream, giving about 7.8 tps in serve versus about
  119 tps single-stream. Size the arena to active slots and keep parity green.
- Raw `hawking generate` is not a valid instruct/Q&A quality gate because it
  does not apply chat templates. Valid instruct evaluation must go through
  `/v1/chat/completions` or explicit per-model templates.

Compression and format:

- Q4_K_M is the current practical floor.
- TQ/trellis sub-4-bit is the extreme-compression product lane: CPU reference is
  bit-identical; GPU path is partial and must be completed before it is a
  shipped inference tier.
- Condense/Hawking already has active 4/3/2/1-bit ingredients: 4-bit practical
  floor, 3-bit shipping candidate, 2-bit PV/QAT recovery lane, and
  1-bit/ternary/native-low-bit research lane. Future work should strengthen the
  whole ladder instead of treating "sub-4-bit" as one undifferentiated bucket.
- Emerging open-weight frontier models, such as GLM-5.2-class MoE releases,
  make "can I even quantize this locally?" a product problem. The Model Press
  opportunity is to stream, shard, rank, and verify quantization under a
  declared memory budget so users can produce artifacts that were previously
  impossible on their machine.
- Per-channel int4-KV numerics pass and survive outlier tests, but the path is
  not wired end-to-end. Wire only behind a default-off flag with parity,
  long-context coherence, and real-model perplexity gates.
- Per-row int4-KV is dead: slower and quality collapse.
- The bake backend is a strategic blocker if still stubbed. `hawking press` and
  any Lab release depend on a real bake -> verify path.
- The artifact must record enough recipe data to reproduce the bake: base hash,
  quant profile, TQ config, AWQ/QAT parameters, KV policy, tokenizer, quality
  card, peak memory/scratch budget, streamed shard manifest, and provenance.

Condensation doctrine:

- Budget first. Every bake/press plan must name RAM, scratch disk, optional GPU,
  and optional cloud budget before work starts.
- Stream the parent. Prefer shardwise/tensorwise processing and early verified
  artifact emission over requiring the full parent model in memory.
- Rank by output damage. Use rel-RMS and MSE as scouts only; bit allocation and
  promotion need KL/NLL, perplexity, task deltas, or heldout quality.
- Treat 4/3/2/1-bit as a ladder. 4-bit is compatibility, 3-bit is the first
  extreme tier, 2-bit requires recovery, and 1-bit/ternary requires retraining
  or native-low-bit assumptions.
- Recover quality, do not merely shrink. Compose quantization with QAT, KD,
  activation de-bias, outlier channels, and selective protection when raw PTQ is
  not enough.
- Separate density from speed. A smaller artifact may decode slower and still
  be valuable if it turns an impossible model into a possible local model.

Spec decode:

- Event Horizon/spec-decode is lossless but speed-negative for single-stream
  decode on this engine. Do not sell it as a speed feature.
- The only acceptable finish paths are:
  1. prove a named winning regime, such as batched throughput or long-context
     memory pressure, with warm measurements and lossless parity; or
  2. formally retire the speed lane and prune/park the code only with attended
     review and parity.

Distillation:

- KD/SFT/QAT scripts exist, but the pipeline is single-device and undertrained.
- The Lab value proposition requires joint compress+distill, not just post-hoc
  quantization and not just a small draft experiment.
- A launch-quality SKU needs eval-in-loop, a committed recipe, tracked metrics,
  and a quality card showing recovered quality versus the fp16/base parent.

Quality:

- Argmax-identity gates are useful for narrow Qwen lever safety.
- Product quality claims require chat-templated evaluation and standard
  benchmark or task-class coverage.
- Every launch SKU needs a quality card with perplexity delta, task-class
  scores, failure classes, footprint, speed, energy if available, and exact
  commands.

Distribution:

- The runtime needs installable artifacts: signed binary and/or Homebrew.
- The app/service lane needs a headless `hawkingd`, config file, model registry,
  model auto-discovery, and `hawking pull` or equivalent HF install path.
- The file format decision is an owner-level product decision. Prefer a
  self-describing Hawking artifact for Lab releases, even if the implementation
  starts as GGUF plus sidecar.

## Non-Negotiable Rails

- No destructive git: no reset, checkout, revert, branch deletion, or history
  rewrite unless the user explicitly asks.
- Do not revert user/agent changes. If the dirty tree mixes lanes, work around
  it and document the lane boundary.
- Do not run global `cargo fmt`; repo-wide fmt drift is known. Format only files
  you intentionally edit if the local style supports it.
- Keep GPU/model jobs sequential. Check for active `hawking generate`,
  `hawking serve`, overnight runners, cargo builds, and training jobs before
  starting another model job.
- Do not stage generated reports under `reports/`.
- Do not change output semantics, quant defaults, model-selection defaults, or
  hot-path kernels without a parity/quality gate.
- Do not spend cloud credits, launch long training runs, publish to Hugging
  Face, delete large code paths, or choose the final file-format identity
  without owner approval.
- Every material claim must be marked as measured, built, stub, gap, or
  aspiration. If evidence is missing, say so.
- Any speed claim must be warm-median measured and above noise, not a cold PSO
  artifact.
- Any quality claim must name the evaluation path and whether chat templates
  were applied.

## Execution Algorithm

1. Inspect the current tree:
   - `git status --short`
   - `git diff --stat`
   - relevant diffs for files you will touch
2. Read the required docs above.
3. Identify the highest-leverage open gate in the chosen lane.
4. Prefer a narrow, reversible implementation.
5. Add or use the smallest meaningful gate before changing risky behavior.
6. Run the smallest relevant checks.
7. Update the standing docs with what changed, exact commands, evidence, and
   the next command.
8. Leave the tree in a reviewable state. Do not mix unrelated lanes.

If blocked, write a concrete blocker note with:

- what you tried,
- exact command/output evidence,
- why it is blocked,
- the next smallest diagnostic,
- whether owner approval is required.

## Priority Lanes

### Lane A - Runtime GA

Goal: P1 is installable, correct, fast enough, and regression-gated.

Top tasks:

1. Recover RWKV serve throughput by sizing RWKV/Qwen multiseq decode work to
   active slots instead of fixed maximum batch.
2. Keep `rwkv7_prefill_slot_multiseq_parity` green.
3. Add request-isolation tests for concurrent prompts and no cross-talk.
4. Route valid quality eval through `/v1/chat/completions`.
5. Add TPS floors, quality floors, and footprint floors to automated gates.
6. Harden HTTP errors: unsupported batch engines must return a clear structured
   error or deliberate fallback, never silently queue forever.
7. Add backpressure, request/body/token limits, and graceful 429 behavior.
8. Package the runtime: signed release, Homebrew, install self-check.

Done means:

- serve correctness gate is green,
- serve throughput is no longer pathologically below single-stream for one
  active slot,
- valid quality eval is running through chat templates,
- regressions fail a gate,
- a user can install and run the runtime without a Rust toolchain.

### Lane B - Model Press

Goal: P2 can produce reproducible compressed artifacts, including artifacts
whose parent model cannot fit fully resident on the target machine.

Top tasks:

1. Un-stub the bake backend if it is still a stub.
2. Implement `hawking press` or equivalent one-command bake -> verify flow.
3. Add a dry-run press planner that reports RAM, scratch, shard plan, expected
   bpw, and unsupported features before starting a bake.
4. Make the bake pipeline resumable and out-of-core: stream shards/tensors,
   emit verified sections early, and avoid full-parent residency.
5. Complete TQ/trellis GPU decode path and parity against the CPU reference.
6. Wire per-channel int4-KV behind `HAWKING_QWEN_INT4_KV_PC`, default-off.
7. Add bit-ladder recipes for 4/3/2/1-bit operating points with measured gates.
8. Record full bake recipe and provenance in the artifact/sidecar.
9. Add `hawking verify` to validate magic, architecture, recipe, base hash,
   tokenizer, quant data, quality card, and loadability.
10. Generate per-quant quality cards automatically.
11. Add a golden-recipe re-bake gate for at least one sample SKU.

Done means:

- a base model can be pressed into a loadable Hawking artifact,
- the press can operate under a declared memory budget and resume after failure,
- the artifact can be re-verified and recipe-reproduced,
- speed/footprint/quality are measured and gated,
- failed quality prevents publication.

### Lane C - Distillation Product

Goal: a launch-quality compress-then-recover model exists.

Top tasks:

1. Add multi-GPU/cloud-ready training support, but do not run paid jobs without
   owner approval.
2. Add eval-in-loop, best-checkpoint selection, early stopping, and tracked
   configs.
3. Build a loss library: CE, top-k KL, temperature schedules, feature matching,
   intermediate-layer KD, and quant-aware KD.
4. Compose QAT and KD into joint compress+distill.
5. Scale teacher capture without disk blowup.
6. Track experiments in a recipe registry.
7. Run a convergence campaign only after approval and a compute plan.

Done means:

- at least one compressed student measurably closes the quality gap to its
  parent,
- the run is reproducible from committed config,
- a quality card explains what recovered, what failed, and what the artifact is
  for.

### Lane D - Hawking Lab

Goal: P3 is legible and reproducible to external users.

Top tasks:

1. Prepare HF org naming/versioning conventions.
2. Create model-card templates with license, provenance, recipe, bpw, quality,
   speed, footprint, and how-to-run on Hawking plus a reference path.
3. Pick launch SKUs with owner approval.
4. Build a demo Space only after at least one artifact is real and quality
   carded.
5. Publish a leaderboard for parent-quality retained versus bpw and tps.
6. Automate release on tag only after verify/quality gates pass.

Done means:

- at least one pre-quantized SKU and one compress-then-distill SKU are live,
- each has a model card, recipe, quality card, and license review,
- users can reproduce or at least verify the artifact.

### Lane E - Format And Headless App

Goal: the user can drop in or pull a model and serve it without knowing paths or
format internals.

Top tasks:

1. Decide format identity with owner input: GGUF plus sidecar versus a single
   self-describing container.
2. Write a versioned format spec.
3. Make format/architecture detection content-based with helpful errors.
4. Add `hawkingd` as a resident headless service.
5. Add config at `~/.hawking/config.toml`.
6. Add model registry commands: list, install/pull, remove, verify, doctor.
7. Integrate HF pull for Hawking Lab SKUs.
8. Add macOS LaunchAgent support after daemon behavior is stable.

Done means:

- a supported artifact is auto-detected and served by model id,
- unsupported artifacts produce actionable errors,
- the format is versioned and verification is first-class.

### Lane F - Spec Decode Resolution

Goal: stop carrying an undecided lane.

Top tasks:

1. Re-bench only named plausible regimes: batched throughput and long-context
   memory-bound workloads.
2. Keep lossless/property gates green.
3. If no regime wins, remove speed claims and mark the lane parked or retired.
4. Do not delete entangled code without attended review.

Done means:

- spec decode is either a gated feature with a measured winning regime or a
  formally retired speed lane with docs updated.

### Lane G - Condense Frontier

Goal: after the core ship-finalization gates, expand Model Press into Condense:
quantize and recover open-weight parents that ordinary local machines cannot
fully load, let alone quantize. This is a general compression/recovery pass
across Hawking, not just a naming lane and not just a GLM experiment.

The sector wedge is:

- lower peak memory for artifact creation than common quantization workflows,
- lower bpw at comparable quality or higher retained quality at the same bpw,
- successful pressing of parent models that cannot fit fully resident on the
  target machine,
- compress-then-recover quality through QAT, KD/distillation, activation
  correction, output-damage-ranked bit allocation, and selective protection.

Read first:

1. `docs/plans/condense_frontier_2026_06_22.md`
2. `docs/plans/condense_naming_migration_2026_06_22.md`
3. `tools/strand/scripts/rung-screen.py`
4. `tools/strand/scripts/rung-kl.py`
5. `tools/strand/scripts/pv-recipe.sh`
6. `tools/strand/scripts/bake-attested.sh`

Top tasks:

1. Build the memory-budgeted press planner.
2. Prove out-of-core tensor/shard quantization on small models first.
3. Convert Condense/legacy-STRAND 4/3/2/1-bit recipes into first-class Hawking
   Press targets.
4. Strengthen 4/3/2/1-bit recipes with QAT/KD recovery, activation de-bias,
   outlier/protected-channel handling, and output-damage-ranked bit allocation.
5. Add quality-card generation and iso-bpw comparisons against ordinary
   quantization baselines where supported.
6. Add GLM-class MoE dry-run support: expert shards, router/gate protection,
   active-vs-total parameter accounting, and license/download warnings.
7. Run frontier downloads, paid cloud jobs, or public derivative publication only
   with owner approval.

Done means:

- Hawking can dry-run a too-large open-weight parent and tell the truth about
  RAM, scratch, time, unsupported features, and expected artifact size,
- at least one model larger than target RAM is pressed through a resumable
  out-of-core path,
- the resulting artifact has a quality card, provenance, and verify gate.

### Lane H - Apple Fit Frontier

Goal: after the core runtime and Condense gates are stable, make Hawking the
obvious native choice on Apple Silicon by finding and running the strongest
usable configuration for the current Mac.

This lane is a capability amplifier, not a throttle. `fit`, `doctor`, and
`serve --auto` must not silently cap tps, context length, batch size, precision,
or model capability. Auto modes choose the strongest stable configuration for
the declared user intent, show stronger and safer alternatives, and allow expert
override. Memory-pressure responses exist to prevent hard failure, swap
collapse, or thermal collapse; they must be visible, reversible when pressure
clears, and regression-gated against the best manual profile.

Read first:

1. `docs/plans/apple_fit_frontier_2026_06_22.md`
2. `docs/plans/hawking_shippability_masterplan_2026_06_22.md`
3. `docs/env_flags.md`
4. `docs/architecture.md`

Top tasks:

1. Add a hardware/runtime profiler for chip family, unified memory, pressure,
   Metal limits, thermal/power state where available, scratch, and active jobs.
2. Build `hawking fit` to report model/quant/context/KV/batch envelopes before
   serving.
3. Add capability-first `hawking serve --auto` intents: max-capability,
   max-speed, max-quality, max-context, max-battery, and safe-fit.
4. Add memory-pressure handling that warns, queues, or adapts visibly instead of
   hiding quality/performance downgrades.
5. Route long-context workloads between transformer and SSM paths only with
   measured quality and speed evidence.
6. Surface energy/thermal metrics in quality cards where measurable.
7. Add anti-throttle regression gates comparing auto-selected runs against the
   best known manual profiles.

Done means:

- Hawking can inspect an Apple Silicon machine and show the usable performance
  envelope before serving starts,
- `serve --auto --intent max-capability` chooses and explains the strongest
  stable configuration, not the safest weak one,
- all downgrades are explicit, overrideable, and justified by user intent or
  hard resource pressure,
- auto policy cannot silently regress speed, quality, or context versus the
  best known manual configuration.

## Validation Ladder

Use the narrowest check that proves the change.

General:

```bash
git status --short
git diff --stat
git diff --check
```

Process/GPU safety:

```bash
ps ax -o pid= -o ppid= -o etime= -o %cpu= -o %mem= -o command= | rg 'hawking (generate|serve)|overnight_hardening|cargo (build|check|test)|mamba2|rwkv7' | rg -v 'rg '
```

Shell syntax:

```bash
bash -n tools/ci/overnight_hardening.sh
bash -n tools/ci/ssm_serve_smoke.sh
bash -n tools/ci/ssm_product_gate.sh
bash -n tools/ci/ssm_quality_suite.sh
```

Runtime/RWKV correctness:

```bash
cargo test --release -p hawking-core --test rwkv7_prefill_slot_multiseq_parity -- --ignored --test-threads=1
tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf
```

CPU-safe hardening:

```bash
RUN_GPU=0 RUN_PREFLIGHT=0 tools/ci/overnight_hardening.sh
```

Full unattended validation only when GPU/model slot is free:

```bash
tools/ci/overnight_hardening.sh
```

Quality:

```bash
tools/ci/ssm_quality_suite.sh
```

If this uses raw `hawking generate`, treat it as a raw-completion smoke only.
For instruct quality, use `/v1/chat/completions` or explicit templates.

## Documentation Updates Required

For every material change, update the smallest relevant set:

- `docs/campaign/autonomous_run_log.md` for chronological work and evidence.
- `docs/campaign/test_matrix.md` for measured speed, quality, footprint, or
  gate results.
- `docs/campaign/open_risks_and_gates.md` when a risk opens, closes, or changes
  rank.
- `docs/campaign/project_standing_snapshot.md` when the current standing changes.
- `docs/plans/hawking_shippability_masterplan_2026_06_22.md` only for strategy
  or roadmap changes, not every small run.
- `docs/campaign/kill_ledger.md` for formally dead levers.
- `docs/campaign/commit_plan.md` when the dirty-tree lane split changes.

Never overwrite evidence with vibes. Prefer exact commands and pass/fail
summaries.

## Owner Decisions

Stop and ask before deciding:

- Apple-only runtime versus broader CUDA/CPU runtime scope.
- Final launch SKU list.
- Advertised extreme-compression target, especially 3.0 bpw versus 2.x bpw.
- Frontier-condensation targets and budgets: GLM-class parent selection,
  download/storage limits, local versus cloud execution, and publication rights.
- Final file-format identity.
- Cloud training budget and provider.
- HF org publishing or public release timing.
- Default-on quality-affecting flags.
- Deleting or untangling spec-decode/Eagle code.

## Definition Of Ready To Ship

Hawking is ready only when:

1. The runtime installs cleanly on a fresh supported Mac.
2. `/v1/chat/completions` and native generation work for launch SKUs.
3. RWKV/SSM long-context serve path is correct and not throughput-pathological.
4. Qwen transformer path remains correct, fast, and regression-gated.
5. Speed, quality, and footprint baselines exist and regressions fail a gate.
6. At least one compressed artifact is produced by a reproducible bake pipeline.
7. Each artifact has a quality card and provenance.
8. The headless app can discover or pull a model and serve it by id.
9. Docs let a new user install, pull, serve, and call the API in under 10
   minutes.
10. Known dead lanes are retired or explicitly parked so they do not distract
    the launch.

## Final Response Shape For Agents

When stopping, report:

- what changed,
- what evidence proves it,
- exact files touched,
- checks run and results,
- what remains,
- the next single best command.

If unable to run a gate, say exactly why.
