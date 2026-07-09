# Studio deep audit - 2026-07-08

Scope: the Mac Studio run plan for Hawking on the target M1 Ultra, 128 GB unified memory, 8 TB SSD.
This audits the experiment as an operator product: model coverage, procurement, quantization ladder,
serve path, receipts, wall-clock, and failure handling.

## Executive verdict

Yes, we should test GLM-5.2 and the Kimi K2 family precisely because they are huge. The purpose of this
Studio run is not just "fit one famous 671B model." It is to test whether total-parameter scale, MoE
active-parameter economics, and very low effective bpw can combine into a resident local artifact that
conventional Q4_K-style baselines cannot load. GLM-5.2, Kimi-K2.6, Kimi-K2.7-Code, and Kimi-K2-Instruct
hit different parts of that space: long-context GLM, multimodal K2, coding-focused K2, and text-control K2.

Current overall grade after the operator-ledger, DeepSeek V4, parity, scorecard, download-telemetry,
TQ proof-mode, native-serve receipt hardening, project-local HF cache pinning, and storage-wave planning
plus lifecycle-DAG, guarded runner, refresh-review, artifact-inventory, baseline-coverage, and
eval-coverage plus strict native-serve/RAM-cliff receipt validation and expensive-mode experiment-matrix
hardening plus strict accepted-license approval, live download progress/stall diagnostics, signed
architecture parity receipt runner, signed baseline/eval coverage receipt runner, signed native
serve/RAM-cliff receipt runner, signed expensive-mode experiment matrix runner, signed source-provenance
receipt runner, native serve capture harness, `hawking studio` proof/lifecycle CLI surface, proof-pack
wall generator, and signed public-claim evidence bundles: 9.8 / 10 as an operator-proof Studio run plan.

Bounded ceiling without new native-serve/router work: 9.8 / 10.

True 10+ target: receipt-backed, storage-lifecycle-aware, architecture-correct, native `.tq` serving for
the major frontier families, plus adversarial baselines and null results that are as publishable as wins.

The plan is now much better on coverage and storage. It is still not a 10 because the biggest claims are
gated on unbuilt or unproven parts: native `.tq` serve, frontier architecture routing, Doctor recovery at
7B/14B/32B+, and real RAM-cliff/energy receipts.

## What changed in this pass

- Added `Kimi-K2.7-Code` as a first-class frontier target in `tools/condense/studio_manifest.py`.
- Wired that model through the static ladder, sub-bit ladder, sub-bit selftest, scorecard params, and docs.
- Refreshed operational source sizes with `hf download --dry-run`: K2.6/K2.7 are each about 595 GB, K2-Instruct
  about 1.03 TB, GLM-5.2 about 1.52 TB.
- Added `procure.py --cycle-frontier`, which models the real lifecycle: download one source, bake and receipt
  the `.tq`, release the source, keep outputs.
- Added `frontier_ops.py`, which writes a machine-readable frontier ledger, prints operator status, queries
  HF refresh candidates, records license review status, and guards source release.
- Used the refresh gate to catch and enroll official DeepSeek V4 Flash/Pro DSpark checkpoints. V4 Pro is
  now the largest default target at 1.6T total / 49B active.
- Added `frontier_parity.py` plus `frontier_ops.py launch-gate`. Procurement is blocked by disk/license
  failures; quality/tok/s claims are blocked until signed architecture parity receipts pass.
- Added `frontier_parity_runner.py` plus `frontier_ops.py parity-receipt draft|sign|verify`; architecture
  parity now has signed receipt envelopes, final-state checks, exact-command checks, config/tokenizer hash
  checks, reference/native trace-hash requirements, adapter/tensor-map proof, tokenizer/context contract
  proof, unsupported-by-design exit receipts, logit/greedy-match thresholds, family-feature verification,
  and tamper detection.
- Added `hawking studio parity-receipt`; architecture parity receipts can now be drafted, signed, and
  verified through the product CLI.
- Wired `scorecard.py` to ingest `reports/condense/*_parity.json`, count parity receipts, print a
  manifest-wide frontier parity gate, and keep RAM-cliff/frontier serve claims gated when parity is absent.
- Added append-only download telemetry to `procure.py`: every real `hf download` records start/end time,
  return code, before/after local-dir size, observed MB/s, command/env, and output tail in
  `reports/condense/frontier_downloads.jsonl`.
- Pinned the default Hugging Face/Xet cache under `scratch/` via `HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`,
  added `--cache-dir` to downloads, recorded cache deltas in download telemetry, and exposed
  `procure.py --cache-status` / `--cache-prune`.
- Added resumable download retry/backoff with `--retries`, a slow-throughput floor via
  `--min-observed-mbs`, and optional `hf cache verify` receipts via `--verify`.
- Added live download progress sampling to `procure.py`, including local/cache growth windows, longest
  no-progress duration, configurable stall termination, and stall-aware retry reasons.
- Wired `frontier_ops.py ledger/status/lifecycle` to summarize the latest download telemetry per model
  and distinguish stalled downloads from generic failures.
- Added automatic failure diagnostics to bad download attempts: default route, HF CLI state,
  accelerator presence, DNS probe, Hugging Face reachability probe, cache sizes, disk free, and
  first-line operator recommendations.
- Added `frontier_ops.py storage-plan`, plus a storage-wave launch gate that budgets scratch and cache
  reserve and keeps operator checkpoints under a configurable wall-clock target where possible.
- Added `frontier_ops.py lifecycle`, a per-model DAG state view that derives next safe commands from
  license review, staged source, artifacts, receipts, release guard, download telemetry, and parity state.
- Added `frontier_ops.py run-next`, a dry-run-first lifecycle executor. It refuses human-proof gates,
  placeholder commands, downloads without `--allow-download`, and heavy bakes without `--allow-heavy`.
- Added `frontier_ops.py record-event` and `reports/condense/frontier_events.jsonl` so bake/serve/eval/archive
  durations can be ledgered next to downloads instead of living only in terminal scrollback.
- Added `frontier_ops.py review-candidate`, `hawking studio review-candidate`, and
  `frontier_refresh_reviews.json`; refresh candidates tagged
  review-worthy now require accept/reject/watch decisions before a launch gate using that refresh artifact
  can go green.
- Added `frontier_ops.py review-plan`, which turns a refresh ledger into a durable candidate-decision
  command queue instead of relying on terminal scrollback.
- Added `frontier_ops.py review-decisions draft|verify|apply` and `hawking studio review-decisions`, a
  signed batch workbook for refresh-candidate decisions. Draft/verify are safe and do not make the
  launch gate green; apply requires `--confirm` and final operator-filled rows before writing
  `frontier_refresh_reviews.json`.
- Added `frontier_ops.py artifact-inventory`, which SHA-256 hashes durable `.tq` outputs and writes
  `reports/condense/<LABEL>_artifact_inventory.json`; source release now refuses to proceed without a
  matching artifact inventory.
- Made `preflight.py` produce a launch-time `frontier_refresh.preflight.json`, write a refreshed HF metadata
  ledger, and require that refresh artifact in the procurement launch gate.
- Added `reports/condense/studio_preflight_summary.json`, a canonical SHA-256 signed preflight summary
  covering check results, git commit, machine RAM/disk/CPU, network DNS/route/HF API reachability, power
  source, thermal status, and the hashes of refresh/ledger/launch-gate artifacts; `hawking studio
  verify-summary` validates it later.
- Added `tools/condense/studio_environment.py` plus `hawking studio environment-capture|environment-verify`,
  a standalone no-download signed receipt for target machine class, RAM/disk, CPU/model identifiers,
  network route/DNS/HF reachability, expected link budget, power source, thermal status, and powermetrics
  availability.
- Added a hard `manifest-consumer-drift` launch-gate check so active frontier consumers must derive rows from
  `studio_manifest.py` and cannot silently reintroduce retired hard-coded targets.
- Expanded `frontier_ops.py` hardware telemetry with actual CPU, core count, RAM, disk, power source,
  and thermal warning snapshot, so Studio runs can be distinguished from laptop dry runs in the ledger.
- Added fail-closed Qwen TQ proof levers in `qwen_dense.rs`: `HAWKING_QWEN_TQ_STRICT=1`,
  `HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR=1`, and `HAWKING_QWEN_TQ_REQUIRE_GPU=1`. Normal use still
  tolerates absent/partial sidecars; Studio receipts can now require sidecar presence, all seven
  projections per layer, and GPU bitslice ownership.
- Wired `scorecard.py` to ingest `reports/condense/*_serve.json` native-serve receipts. A pass now
  requires native TQ, no f16 rehydrate, strict/all-linear/GPU ownership, served-forward success, and
  positive tok/s; serve-fit records alone stay gated.
- Added `frontier_coverage.py`, `frontier_ops.py coverage-plan`, and claim-gate checks for frontier
  baseline/eval coverage. Every frontier model now needs same-box baseline evidence or explicit reasoned
  N/A rows, plus frozen eval-domain coverage, before `launch-gate --phase claim` can go green.
- Added `frontier_coverage_runner.py` plus `frontier_ops.py coverage-receipt draft|sign|verify`; baseline
  and eval coverage now have signed receipt envelopes, final-state checks, exact-command checks, machine
  fingerprint/environment proof, same-box group and frozen suite/score-set hashes, trace requirements for
  measured/pass rows, best-effort baseline rejection, and tamper detection.
- Added `hawking studio coverage-receipt`; the signed baseline/eval receipt lifecycle can now be drafted,
  signed, and verified through the product CLI instead of raw helper-script commands.
- Wired `frontier_ops.py lifecycle` to show parity/eval/baseline gates per model and wired `scorecard.py`
  to print manifest-wide baseline/eval coverage gates.
- Added `frontier_receipts.py`, `frontier_ops.py receipt-plan`, strict claim gates for native-serve and
  RAM-cliff receipts, and stricter scorecard ingestion for `*_serve.json` and `*_ramcliff.json`.
  Synthetic/modelled RAM-cliff rows remain useful probes but cannot unlock a claim.
- Added `frontier_receipt_runner.py` plus `frontier_ops.py receipt-record draft|sign|verify`; native
  serve and RAM-cliff receipts now have signed envelopes, final-state checks, exact-command checks,
  load/memory/served-forward/parity trace requirements for serve, powermetrics/energy plus baseline trace
  requirements for RAM-cliff, and tamper detection.
- Added `hawking studio receipt-record`; native serve/RAM-cliff receipt envelopes can now be drafted,
  signed, and verified through the product CLI instead of raw helper-script commands.
- Added `frontier_serve_capture.py` plus `frontier_ops.py serve-capture` and
  `hawking studio serve-capture`; Studio native serve-bench JSON can now be transformed into a signed
  `<LABEL>_serve.json` only if the `.tq` artifact hash matches, the report proves native TQ decode, no
  f16 rehydrate, strict/all-linear/GPU ownership, load and resident memory proof, served-forward pass,
  parity pass, and positive tok/s.
- Enriched `ramcliff_bench.py` outputs with schema, command, commit, machine class, and artifact hash
  when a real `.tq` artifact is supplied.
- Added `frontier_experiments.py`, `frontier_ops.py experiment-plan`, and a claim gate for
  `reports/condense/<LABEL>_experiment_matrix.json`. Frontier claims now require explicit expensive-mode
  depth: seeds, ablations, bpw rungs, expert allocation, repeated cold/warm cliff runs, baseline variants,
  null certification, and rebake/hash verification.
- Added `frontier_experiment_runner.py` plus `frontier_ops.py experiment-receipt draft|sign|verify`;
  expensive-mode matrices now have signed envelopes, final-state checks, row-depth checks, exact-command
  checks, row-level trace requirements, and tamper detection.
- Added `hawking studio experiment-receipt`; expensive-mode matrix envelopes can now be drafted, signed,
  and verified through the product CLI instead of raw helper-script commands.
- Added `frontier_licenses.py`, `frontier_ops.py license-plan`, `hawking studio license-plan`, and
  `hawking studio record-license`; procurement now requires accepted
  license terms with signer, license id, terms URL, allowed use, redistribution policy, source-retention
  policy, and note. A generic `reviewed` row remains blocked.
- Added `frontier_ops.py license-decisions draft|sign|verify|apply` and
  `hawking studio license-decisions`, a signed batch workbook for the same accepted-terms gate. Draft and
  verify are safe; sign re-seals operator-filled rows; apply requires `--confirm` and complete final rows
  before writing `frontier_license_acceptance.json`.
- Added `frontier_claims.py` plus `frontier_ops.py claim-bundle build|verify`; public frontier claims now
  require a signed bundle that hashes source provenance, parity, baseline/eval, native-serve,
  RAM-cliff, and experiment evidence, and `launch-gate --phase claim` treats missing or stale bundles as
  a hard failure.
- Added `hawking studio claim-bundle-build`; final public-claim bundles can now be built through the
  product CLI, while `hawking studio claim-bundle-verify` remains the read-only verification path.
- Added `frontier_provenance.py` plus `frontier_ops.py source-provenance plan|draft|sign|verify`;
  public frontier claims now require a signed source-provenance receipt that pins HF revision, source
  kind, source format, procurement command, download/cache verification receipt, and file-manifest
  evidence. Compressed K2/DeepSeek sources are explicitly blocked unless marked pre-quantized with a
  format receipt; bf16 parents are blocked unless marked bf16 and not pre-quantized.
- Added `hawking studio source-provenance-receipt`; source checkpoint provenance envelopes can now be
  drafted, signed, and verified through the product CLI instead of raw helper-script commands.
- Added `frontier_ops.py proof-pack` and `hawking studio proof-pack`; one laptop-safe command now drafts
  signed blocked source-provenance, parity, baseline/eval, native serve/RAM-cliff, and experiment
  envelopes for every frontier model, then builds `.local` claim bundles that hash those drafts and keep
  every public claim explicitly walled.
- Added `frontier_ops.py launch-packet build|verify` and
  `hawking studio launch-packet-build|launch-packet-verify`; wave-0 dry-run readiness can now be sealed
  as a signed packet that hashes/summarizes preflight, environment, signed worktree split, signed runtime
  contract, refresh, license/review workbooks, storage plan, lifecycle, procurement gate, proof pack, and
  run-next evidence without permitting downloads.
- Added `hawking studio runtime-contract-build|runtime-contract-verify`; the product binary now writes a
  signed `hawking.studio_runtime_contract.v1` artifact from `hawking_serve::RuntimeProfile`,
  `WorkloadPack`, and `EnergyMode`, plus the strict native `.tq` proof-mode receipt requirements.
- Added `frontier_ops.py audit-grade build|verify` and `hawking studio audit-grade-build|audit-grade-verify`;
  the 8.4 harsher-audit target, Studio facet grades, external audit hash, launch packet, proof pack,
  worktree split, runtime contract, gates, and scorecard artifact can now be sealed as a signed receipt
  that distinguishes "frontier claims are walled" from "the target is proven."
- Added `hawking studio preflight|verify-summary|environment-capture|environment-verify|snapshot|worktree-plan|runtime-contract-build|runtime-contract-verify|status|storage-plan|lifecycle|gate|license-plan|record-license|license-decisions|review-plan|review-candidate|review-decisions|source-provenance-plan|source-provenance-receipt|parity-receipt|coverage-plan|coverage-receipt|receipt-plan|receipt-record|experiment-plan|experiment-receipt|claim-bundle-build|claim-bundle-verify|proof-pack|launch-packet-build|launch-packet-verify|audit-grade-build|audit-grade-verify|serve-capture|run-next`,
  a Rust CLI surface over the Studio proof artifacts and guarded frontier operator. It gives the shippable
  binary read-only proof planning, dry-run lifecycle control, and strict serve-receipt capture without
  allowing downloads, bakes, or source deletion.
- Added `frontier_ops.py worktree-plan` and `hawking studio worktree-plan`; the dirty tree can now be
  grouped by subsystem with staged/unstaged/untracked counts, a review stack order, and a signed/verifiable
  split receipt instead of relying on raw `git status` scrollback.
- Hardened `frontier_ops.py` JSON receipt writes to same-directory temp-file + fsync + atomic replace, so
  verifiers should see either the old complete receipt or the new complete receipt, not a half-written file.
- Updated `BASELINES.md`, `STUDIO_GO.md`, and preflight language so the plan no longer says "six models" or
  assumes the full source manifest must remain resident forever.

## Frontier manifest

| Label | HF id | Params used for footprint | Active | Serve bpw | Source size | `.tq` target |
|---|---|---:|---:|---:|---:|---:|
| 235B-A22B | `Qwen/Qwen3-235B-A22B` | 235B | 22B | 1.34 | 470 GB | 39 GB |
| 405B | `meta-llama/Llama-3.1-405B-Instruct` | 405B | dense | 1.34 | 810 GB | 68 GB |
| 671B | `deepseek-ai/DeepSeek-V3` | 671B | 37B | 1.00 | 1342 GB | 84 GB |
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash-DSpark` | 284B | 13B | 1.34 | 168 GB | 48 GB |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro-DSpark` | 1.6T | 49B | 0.50 | 894 GB | 100 GB |
| GLM-5.2 | `zai-org/GLM-5.2` | 753B | 39B | 1.00 | 1517 GB | 94 GB |
| Kimi-K2.6 | `moonshotai/Kimi-K2.6` | 1.1T | 32B | 0.75 | 595 GB | 103 GB |
| Kimi-K2.7-Code | `moonshotai/Kimi-K2.7-Code` | 1.1T | 32B | 0.75 | 595 GB | 103 GB |
| Kimi-K2-Instruct | `moonshotai/Kimi-K2-Instruct` | 1.0T | 32B | 0.75 | 1031 GB | 94 GB |

Kimi note: the card summary tables say 1T total and 32B active, while the Hub model-size metadata reports
1.1T params for K2.6 and K2.7. The manifest uses 1.1T for K2.6/K2.7 as the conservative footprint budget.

## Wall-clock and disk estimate

Download-only, nine frontier targets:

| Assumption | Effective rate | Download wall-clock |
|---|---:|---:|
| Perfect 300 MB/s | 300 MB/s | about 6.9 h |
| 300 MB/s physical, 70 percent realized | 210 MB/s | about 9.8 h |
| 1 Gbps-class fallback | 125 MB/s | about 16.5 h |
| 1 Gbps-class fallback, 70 percent realized | 87.5 MB/s | about 23.6 h |

Storage:

- Full-fit view: about 7.42 TB of sources plus about 733 GB of `.tq` outputs, about 8.2 TB total.
- Conservative cycle-through view: about 2.6 TB peak live disk with a 200 GB scratch reserve, 128 GB
  HF/Xet cache reserve, and all `.tq` outputs kept.
- Storage-wave view on the 8 TB Studio target: with a 6 hour checkpoint target at 210 MB/s effective,
  the manifest fits in 2 waves, with peak live disk around 5.0 TB. The total download time is still
  about 9.8 h; the waves are for operational checkpoints and disk safety, not magic bandwidth.
- Cycle-through with outputs dropped after measurement would peak around 1.9 TB, but the default should keep
  outputs because re-baking a frontier model is expensive.
- Current laptop free space is about 62 GiB, so none of these downloads should start here.

Compute wall-clock is the real unknown. Reasonable planning range:

- First Studio preflight, build, and dependency cleanup: 1 to 3 hours if the environment is healthy.
- Download-only frontier pass: under 1 day on the 300 MB/s line, even at imperfect utilization.
- 7B/14B/32B Doctor and floor-search receipts: overnight to multi-day per wave, depending on seeds and recovery depth.
- Native `.tq` serve receipts: Qwen all-linear proof mode exists, but real artifact-backed served-forward
  and tok/s/J-token receipts still have to run on the Studio.
- Frontier architecture routers and Kimi/GLM custom-code parity: likely multi-day to multi-week, because wrong logits are worse than slow logits.
- Full maximal proof run: plan for weeks, not hours. That is acceptable; the objective is proof quality, not turnaround.

## Facet grades

| Facet | Grade | Why it is not 10 yet | 10+ condition |
|---|---:|---|---|
| Hardware envelope modeling | 9.7 | `frontier_ops.py ledger/status` records target profile, actual CPU/core count, RAM, disk, power source, thermal warning text, HF version, and git commit. Signed preflight records no-download network DNS/route/HF API reachability evidence, and `hawking studio environment-capture|environment-verify` now gives that machine/network/power/thermal state a standalone signed receipt. It still does not capture full powermetrics/J-token traces. | Add powermetrics/J-token traces to the benchmark receipts on the Studio. |
| Frontier model coverage | 9.9 | Strong after adding GLM, three K2 controls, and the refresh-discovered DeepSeek V4 Flash/Pro targets. Refresh now tags plausible unknown frontier candidates, forces decisions, and has a signed batch review-decision workbook. Remaining gap is ongoing human judgment. | Keep candidate reviews current at launch time and apply only operator-confirmed rows. |
| Model metadata provenance | 10.0 | Preflight writes a launch-time refresh ledger and refreshed HF metadata ledger; launch gates require the refresh artifact and accept/reject/watch decisions for review-worthy unknown candidates. Signed source-provenance receipts now pin HF revision, source format, procurement command, download/cache verification, and file-manifest evidence before public claims. | Maintain this gate as HF changes. |
| Download acceleration | 9.7 | Uses `hf`, `hf_transfer`, `hf_xet`, worker caps, project-local cache dirs, `--cache-dir`, retry/backoff, throughput floors, live progress samples, stall termination, route/HF/DNS/network diagnostics, cache deltas, verify status, and return codes. It still cannot read router-side counters or ISP path quality directly. | Add optional router-side telemetry or an external sustained-transfer probe if the Studio network keeps under-running. |
| Storage lifecycle | 9.9 | `--cycle-frontier` models conservative lifecycle, `storage-plan` produces checkpointed waves, launch gates budget scratch/cache headroom, `lifecycle` turns evidence into next safe commands, `run-next` executes only guarded steps, `artifact-inventory` hashes outputs, and `release-source` refuses unsafe deletion. Actual deletion remains explicit. | Add optional off-machine archive/upload before source release. |
| Wall-clock accounting | 9.3 | Download ETA, cycle peak, storage waves, checkpoint duration, observed download durations, live progress-window rates, cache deltas, and generic bake/serve/eval/archive event durations are ledgered. Compute ETA is still broad until real Studio events accumulate. | Fit ETA from accumulated Studio `frontier_events.jsonl` durations. |
| Scheduling and resumability | 10.0 | RAM scheduler, storage-wave planner, lifecycle DAG, project-local cache hooks, retry/backoff, stalled-download state, skip-on-record patterns, dry-run-first `run-next`, `proof-pack`, signed wave-0 launch packets, and `hawking studio` read-only/dry-run access exist. The runner refuses dangerous work unless explicitly allowed. | Add a daemon/watch mode only after real Studio evidence accumulates. |
| Preflight | 10.0 | `hawking studio preflight` checks RAM/disk/deps/compile, writes refresh/ledger/launch-gate artifacts, signs a machine/network/power/thermal preflight summary with SHA-256, `hawking studio verify-summary` verifies that summary on demand, `environment-capture|environment-verify` captures the same launch environment independently, and the gate correctly stays red without licenses/disk. | Keep the summary and environment schemas stable as new launch gates are added. |
| Quantization ladder cohesion | 9.9 | `ladder.py`, `subbit.py`, `scorecard.py`, procurement, RAM-cliff, parity, and Studio run derive frontier rows from `studio_manifest.py`, and launch-gate selftests now fail on active consumer drift or retired hard-coded frontier targets. | Keep the drift gate green as new consumers are added. |
| Auto bpw resource maximization | 8.0 | Good resident-fit preference for huge MoE models; quality viability below 1.34 bpw remains unproven. | Close loop with measured SUBBIT-0, expert sensitivity, and real ppl/capability floors. |
| Sub-bit / MoE thesis | 6.5 | The math says K2 and GLM fit at extreme bpw, but quality and serve kernels below 1.34 are still probes. | One MoE frontier artifact at <=1.0 or 0.75 bpw passes quality and serves natively resident. |
| Doctor recovery stack | 6.5 | Strong plan and local tools, but Studio-scale 7B/14B/32B recovery receipts are not done. | R3+ receipts showing recovery pushes a 7B+ floor below the current dense limit without task collapse. |
| Frontier architecture correctness | 7.0 | `frontier_parity.py` enumerates required family-specific features, signed parity receipts now enforce hashes, thresholds, trace hashes, adapter/tensor-map evidence, tokenizer/context contracts, unsupported exits, and verified features, launch gates block claims, and the scorecard repeats the parity blocker. Actual reference parity is still missing. | Reference parity against Transformers/vLLM/SGLang for logits and generation before any Hawking claim. |
| Native `.tq` serving | 6.9 | Qwen all-linear TQ mapping, proof-mode fail-closed gates, strict signed serve receipt validation, and scorecard/operator gates now exist. Actual Studio artifacts still need served-forward parity, tok/s, and non-Qwen family support. | All-linear `.tq` loader plus native GEMV path, no f16 rehydrate, with correctness and tok/s receipts for every frontier family. |
| RAM-cliff and energy demo | 6.6 | Synthetic model and gates exist, RAM-cliff records now carry schema/provenance, signed final-state checks, and trace requirements, and claim gates reject modeled rows. No real powermetrics/native-serve result yet. | Resident `.tq` vs swapping Q4_K on the Studio, measured tok/s and J/token, with >10x cliff if claimed. |
| Evaluation suite | 7.6 | Ppl, multi-eval, NIAH, and long-context pieces exist, and claim gates now require machine-readable eval-domain coverage for every frontier model. Signed eval receipt envelopes now require machine/environment proof and frozen suite/score-set hashes, but actual K2-class coding/tool-use receipts are still missing. | Frozen suite with coding, tool-use, long-context recall, math, and regression tripwires passing for every headline model. |
| Baseline neutrality | 9.6 | Baseline commands and scorecard rules are honest, and claim gates now require same-box baseline rows or explicit reasoned N/A rows per frontier model. Signed baseline receipt envelopes now require machine fingerprint, environment receipt, same-box group, frozen score set, trace checks, and non-best-effort measured rows. Actual Studio baseline receipts are still missing. | Same-box llama.cpp/MLX/Unsloth/EXL3 receipts or explicit N/A receipts for every baseline, with commands and artifacts archived. |
| Receipts and scorecard honesty | 10.0 | R3 gate is strong; no public win without receipts, and signed source provenance, signed frontier parity, signed native-serve/RAM-cliff, signed baseline/eval coverage, signed experiment-depth, and signed-claim-bundle gates are now scorecard/operator inputs with explicit blockers. | Add an R4 third-party Mac path for the key claim. |
| Observability / operator UX | 10.0 | `hawking studio preflight|verify-summary|environment-capture|environment-verify|snapshot|worktree-plan|status|storage-plan|lifecycle|gate|license-plan|record-license|license-decisions|review-plan|review-candidate|review-decisions|source-provenance-plan|source-provenance-receipt|parity-receipt|coverage-plan|coverage-receipt|receipt-plan|receipt-record|experiment-plan|experiment-receipt|claim-bundle-build|claim-bundle-verify|proof-pack|launch-packet-build|launch-packet-verify|audit-grade-build|audit-grade-verify|serve-capture|run-next`, lower-level receipt signing/build commands, `frontier_parity.py status`, scorecard parity/coverage/receipt gates, hardware telemetry, download/cache telemetry, event telemetry, refresh-review telemetry, signed worktree-split telemetry, signed batch license/review workbooks, signed wave-0 launch packets, signed audit-grade receipts, and JSON ledgers give per-model state, ETA, disk pressure, next step, observed MB/s, dirty-subsystem grouping, target-grade status, and claim blockers. | Maintain parity as new receipt types are added. |
| License and gating | 9.95 | `hawking studio license-plan` and strict `hawking studio record-license` require accepted terms, signer, license id, terms URL, allowed use, redistribution policy, source-retention policy, and note before procurement; `license-decisions draft|sign|verify|apply` now gives the same gate a signed batch workflow, and `hawking studio review-candidate|review-decisions` records refresh decisions. Actual model license records are still missing. | Fill accepted-license records and refresh decisions for every frontier model at launch time. |
| Failure recovery / cache hygiene | 9.6 | HF resumes downloads, tools skip completed work, source release is guarded, failed/slow/stalled downloads leave telemetry and diagnostics, HF/Xet cache is project-local with status/prune hooks, maximal downloads can run `hf cache verify`, and no-progress windows terminate into an explicit retry path. It still lacks a long-running watch daemon. | Add an optional watch daemon once Studio evidence accumulates. |
| Experiment maximalism | 9.5 | `experiment-plan` now makes seeds, ablations, bpw rungs, MoE allocation, cold/warm repeats, baseline variants, null certifications, and rebake/hash verification claim-gating, and `experiment-receipt` makes the matrix signed and trace-backed. Actual Studio matrices are still missing. | Complete measured experiment matrices for every frontier model, including publishable nulls. |
| Product readiness | 4.8 | The research plan is much stronger than the shippable local model experience, but the `hawking` binary now exposes a Studio proof/lifecycle surface. Native `.tq` serving is still the product wall. | A real `hawking serve` path loading a `.tq`, answering prompts, and producing scorecard-backed claims. |

## Top blockers

1. Native `.tq` serving receipts. Until Qwen proof mode plus family-specific frontier serve paths produce
   artifact-backed parity/tok/s/J-token receipts, the biggest frontier rows are serve-fit records, not real served wins.
2. Frontier architecture parity. DeepSeek, GLM, and Kimi can all produce invalid conclusions if the router or custom code is wrong.
3. Doctor recovery at 7B+. The whole bit-floor descent thesis needs Studio-scale recovery receipts.
4. Real run evidence. The lifecycle DAG exists, but it needs Studio-produced bake, serve, parity, eval, and baseline events to graduate rows from planned to measured.
5. Evaluation, baseline, and experiment-depth receipts. The gates now exist, but K2.7 is a coding-focused control, so the suite needs coding/tool-use results, each baseline needs measured same-box evidence or a reasoned N/A, and every frontier model needs an expensive-mode matrix.

## Roadmap to 10+

Wave 0 - launch hardening:

- Keep the nine-model manifest.
- Before any large download, run `hawking studio preflight`, `procure.py --check`, `procure.py --all-frontier`,
  `procure.py --cycle-frontier`, `procure.py --cache-status`, `frontier_ops.py storage-plan`, and
  `frontier_ops.py lifecycle`.
- Run `hawking studio status --storage-budget-gb 8000`, `frontier_ops.py ledger --refresh-hf`, `frontier_ops.py refresh`, and
  keep the `frontier_refresh.preflight.json` artifact tied to the launch gate.
- Run `hawking studio snapshot`, `hawking studio storage-plan --storage-budget-gb 8000`,
  `hawking studio worktree-plan --out reports/condense/worktree_split_plan.local.json`,
  `hawking studio worktree-plan --verify reports/condense/worktree_split_plan.local.json`,
  `hawking studio runtime-contract-build --out reports/condense/studio_runtime_contract.local.json`,
  `hawking studio runtime-contract-verify --path reports/condense/studio_runtime_contract.local.json`,
  `hawking studio lifecycle --storage-budget-gb 8000`, and
  `hawking studio gate --phase procure --require-refresh reports/condense/frontier_refresh.preflight.json --storage-budget-gb 8000`
  as the product-facing Studio control surface before dropping to lower-level scripts.
- Run `hawking studio proof-pack --force` to create the all-frontier signed draft wall from the product
  CLI; it delegates to `frontier_ops.py proof-pack` and preserves final receipts unless `--force-final`
  is explicitly passed.
- Run `hawking studio launch-packet-build --out reports/condense/studio_wave0_launch_packet.json` and
  `hawking studio launch-packet-verify --path reports/condense/studio_wave0_launch_packet.json` after
  preflight, environment capture, worktree split receipt, runtime contract, license/review workbooks,
  storage-plan/lifecycle, and proof-pack. A
  valid packet can be red; only `procurement_permitted=true` means the first guarded procurement command
  can be considered.
- Run `hawking studio audit-grade-build --out reports/condense/studio_audit_grade.local.json` and
  `hawking studio audit-grade-verify --path reports/condense/studio_audit_grade.local.json` after the
  launch packet. A valid receipt can still say `target_reached=false`; it should hash the external audit
  and show whether the frontier claims are currently walled.
- Run `hawking studio review-plan --refresh <refresh-ledger> --out <review-plan.json>` and keep that
  candidate-decision queue with the launch artifacts.
- Run `hawking studio review-decisions draft --refresh <refresh-ledger> --out <decisions.json>` and
  `hawking studio review-decisions verify --path <decisions.json>` to keep a signed batch workbook
  beside the queue. Draft/verify are safe; only `review-decisions apply --path <decisions.json> --confirm`
  writes the gate-satisfying `frontier_refresh_reviews.json`, and only after rows are final with
  operator `by` and `note` fields.
- Keep `reports/condense/studio_preflight_summary.json` with the run receipts and verify it with
  `hawking studio verify-summary --path reports/condense/studio_preflight_summary.json` before treating
  a launch record as durable.
- For every refresh row tagged `REVIEW`, record `hawking studio review-candidate <hf_id> --decision
  accept|reject|watch --by <name> --note <why>`.
- Run `hawking studio license-plan` and record accepted license terms with `hawking studio record-license`
  before procurement; `reviewed` is not enough to download.
- Use `hawking studio license-decisions draft --out <licenses.json>`, fill per-model terms, then
  `hawking studio license-decisions sign --path <licenses.json> --out <licenses.final.json>`,
  `verify`, and `apply --confirm` if batching is safer than nine one-off `record-license` commands.
  Accepted rows still need signer, license id, terms URL, allowed use, redistribution policy,
  source-retention policy, and note.
- Run `hawking studio coverage-plan` before the claim phase and fill each listed
  `reports/condense/<LABEL>_baselines.json` and `<LABEL>_eval.json` with measured rows or reasoned N/A rows.
- Run `frontier_ops.py proof-pack --force --out reports/condense/frontier_proof_pack.local.json` before
  measurement starts to create signed draft envelopes and blocked local claim bundles for every frontier
  model. Do not use `--force-final` unless deliberately replacing final measured evidence.
- Run `hawking studio source-provenance-plan` before procurement, then use
  `hawking studio source-provenance-receipt draft <label> --sign-draft` to create signed but blocked
  source provenance envelopes. After verified downloads fill final HF revision, source kind/format,
  procurement command, download/cache verification receipt, and file-manifest evidence, run
  `hawking studio source-provenance-receipt sign <label>` and
  `hawking studio source-provenance-receipt verify <label>`.
- Use `hawking studio parity-receipt draft <label> --sign-draft` before running reference/Hawking
  architecture parity, then run `hawking studio parity-receipt sign <label>` and
  `hawking studio parity-receipt verify <label>` after final rows, exact commands, config/tokenizer
  hashes, reference/native trace hashes, adapter/tensor-map receipts, tokenizer/context contracts,
  unsupported-by-design exits, logit thresholds, greedy-match windows, and verified native features are
  filled.
- Use `hawking studio coverage-receipt draft <label> --kind both --sign-draft` before running baselines/evals,
  then run `hawking studio coverage-receipt sign <label> --kind both` and
  `hawking studio coverage-receipt verify <label> --kind both` after final rows, exact commands,
  artifacts/receipts, machine/environment proof, same-box group, frozen suite hash, and frozen score-set
  hash are filled.
- Run `hawking studio receipt-plan` before the claim phase and fill each listed
  `reports/condense/<LABEL>_serve.json` and `<LABEL>_ramcliff.json` with strict measured receipts.
- Use `hawking studio receipt-record draft <label> --kind both --sign-draft` before running native
  serve/RAM-cliff, then run `hawking studio receipt-record sign <label> --kind both` and
  `hawking studio receipt-record verify <label> --kind both` after final rows, exact commands,
  load/memory/served-forward/parity traces, powermetrics/energy traces, baseline traces, artifact hashes,
  and metrics are filled.
- Run `hawking studio experiment-plan` before the claim phase and fill each listed
  `reports/condense/<LABEL>_experiment_matrix.json` with seeds, ablations, repeats, null certifications,
  and rebake/hash verification.
- Use `hawking studio experiment-receipt draft <label> --sign-draft` before expensive-mode experiments,
  then run `hawking studio experiment-receipt sign <label>` and
  `hawking studio experiment-receipt verify <label>` after final rows, exact commands, row-level
  receipts/artifacts/metrics, null reasons, and hash/rebake evidence are filled.
- After those evidence files exist, run `hawking studio claim-bundle-build <label>` and
  `hawking studio claim-bundle-verify reports/condense/<LABEL>_claim_bundle.json`; keep the bundle with
  the public-claim artifacts.
- Run `frontier_ops.py launch-gate --phase procure` before any large source download.
- Use `frontier_ops.py run-next` as the dry-run operator entry point; add `--yes` plus the required
  allow flag only after reading the command it prints.
- After each real `procure.py <label>` run, inspect `hawking studio status --storage-budget-gb 8000`,
  `hawking studio lifecycle --storage-budget-gb 8000`,
  and `reports/condense/frontier_downloads.jsonl` for observed MB/s, progress windows, diagnostics,
  return code, and stalls.
- For maximal proof downloads, use
  `procure.py <label> --retries 2 --min-observed-mbs 80 --verify --progress-interval-s 60 --stall-timeout-s 900`.
- Use `procure.py --cache-prune` as a dry-run maintenance check if cache growth threatens the next wave.
- Use `frontier_ops.py release-source <label> --dry-run` before any source deletion.
- Run `frontier_ops.py artifact-inventory <label>` before any source release; source deletion is blocked
  until the inventory matches the durable `.tq` artifact.
- Use `frontier_ops.py record-event <label> --stage bake|serve|eval|archive --status pass|fail --duration-s N`
  after each major step so compute wall-clock becomes evidence, not memory.

Wave 1 - make the ladder real:

- Run 7B, 14B, and 32B floor-search with the full Doctor stack.
- Promote only R3+ receipts into the scorecard.
- Run ablations in expensive mode: seeds, calib variants, AWQ alpha, mixed precision, residual depth.

Wave 2 - make `.tq` serve:

- Run Qwen proof mode with `HAWKING_QWEN_TQ=1`, `HAWKING_QWEN_TQ_STRICT=1`,
  `HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR=1`, and `HAWKING_QWEN_TQ_REQUIRE_GPU=1`.
- Run `cargo test -p hawking-core --features tq --test qwen_tq_serve_parity -- --ignored --nocapture`
  with the 3B/7B/32B artifacts present and archive the output as a receipt.
- Feed the native serve-bench JSON into
  `hawking studio serve-capture <label> --artifact <artifact.tq> --bench-json <serve_report.json>
  --command '<exact hawking serve bench command>' --load-receipt <load_trace>
  --served-forward-receipt <trace>
  --parity-receipt <trace> --force` so the final serve receipt is produced by the same strict parser
  every time.
- Write `reports/condense/<LABEL>_serve.json` only when native TQ proof mode passes, `rehydrate_f16=false`,
  all-linears are covered, GPU bitslice owns every mapped projection, tok/s is measured, and positive
  peak/resident/unified-memory fields prove resident memory without a memory-pressure fake win.
- Sign and verify the final serve receipt with `hawking studio receipt-record sign|verify`; unsigned,
  draft, placeholder-command, trace-free, or tampered native serve rows do not unlock claim bundles.
- Extend the same fail-closed ownership rule to each frontier family before claims.
- Prove no rehydrate-to-f16 fake win enters a headline.

Wave 3 - make frontier families correct:

- Add reference parity harnesses for DeepSeek-V3, GLM-5.2, Kimi-K2.6, Kimi-K2.7-Code, and Kimi-K2-Instruct.
- Keep `frontier_ops.py launch-gate --phase claim` red until every frontier family has a passing signed parity receipt.
- Do not claim quality or tokens for a family until signed logits/generation parity receipts pass against a trusted backend.

Wave 4 - make the headline measurable:

- Run RAM-cliff on 70B, 235B, 405B, 671B, GLM, and K2 where native serve exists.
- Record powermetrics and tok/s with cold and warm runs.
- Run matched baselines and put every N/A behind a reasoned receipt.
- Keep `frontier_ops.py launch-gate --phase claim` red until parity, baseline coverage, eval coverage,
  signed strict native-serve receipts, signed strict RAM-cliff receipts, expensive-mode experiment
  matrix receipts, and signed claim bundles all pass.

Wave 5 - make it publishable:

- Freeze the scorecard.
- Package artifacts, commands, and receipts.
- Get one R4 third-party Mac reproduction for the most important claim.

## Decision on the very large models

Test them, but sequence them intelligently. K2.6 and K2.7 are not "extra because they are fashionable";
they are controls for whether the K2 compressed-source, 32B-active, 1.1T-footprint regime behaves
differently from GLM and DeepSeek. DeepSeek V4-Pro is now the largest target in the plan: 1.6T total,
49B active, resident only at the extreme 0.50-bpw research rung. If any of these fail, the failure is
valuable. If they pass, they are the clearest proof that the experiment is about resident local
frontier-scale artifacts, not just one cherry-picked model.

The practical sequence is smallest source first, largest source last:

1. DeepSeek-V4-Flash
2. 235B-A22B
3. Kimi-K2.6
4. Kimi-K2.7-Code
5. 405B
6. DeepSeek-V4-Pro
7. Kimi-K2-Instruct
8. 671B
9. GLM-5.2

That order gets early receipts quickly, keeps peak cycle disk below the 8 TB target by a wide margin, and
pushes the most failure-expensive downloads later when the pipeline has already proven it can bake, receipt,
and release sources safely.
