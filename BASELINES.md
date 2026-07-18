# BASELINES.md — baseline neutrality spec

> Active spec sources: `docs/plans/STUDIO_GO.md`,
> `docs/plans/quintessential_engine_2026_06_29.md`, and
> `docs/plans/computational_efficiency_paradigms_2026_07_11.md`.
> The honest baseline set, each with its **exact command** and a **best-effort note**.
> Rule: a baseline marked **best-effort** can support a *contingent* or *negative* claim
> but **never a public win** (§20.3 invalidation rule 8). No rhetorical sandbagging — if a
> baseline beats Hawking, its receipt ships unchanged.

## The baseline set (run all on the SAME named machine, same frozen suite)

The named machine is the M3 Ultra Studio: 96 GB unified memory, 819 GB/s advertised memory bandwidth,
and 1 TB SSD. Regenerate prior M3 Pro/M1 Ultra environment and baseline receipts. Alongside quality and
effective bpw, every measured row records cold/warm mode, output length, TTFT, inter-token latency,
p50/p95 wall time, useful/SLO goodput, joules per accepted token, bytes moved/resident bytes where
available, useful capability per joule/byte/resident byte/active parameter/wall-clock second, peak unified
memory, memory-pressure state, swap delta, free disk, and thermal state. "Capability" means a completion
that clears the frozen correctness/quality gate; rejected speculative tokens and quality-lost outputs are
not useful work.

The GPT/Codex app is a protected interactive tenant, so measurements use the 78 GiB process admission
budget rather than treating all 96 GB as available. Current free space, not nominal 1 TB capacity, is the
input to every storage decision; preserve the 150 GB hard floor plus the declared 64 GB processing
scratch and 32 GB HF/Xet transient-cache reserves. Low-overhead detached supervisors monitor pressure,
swap, thermal/power state, process-group RSS, and disk. An attached conversation is not the monitor and
must not be kept sampling just to prove that a detached job exists.

| baseline | exact command (fill in `<...>`) | tuning status |
|---|---|---|
| llama.cpp Q4_K_M | `llama-quantize <parent.gguf> <out.gguf> Q4_K_M ; llama-bench -m <out.gguf> -p <prompt>` | **tuned** — also try Q4_K_S, IQ4_XS |
| llama.cpp mmap OOC | `llama-cli -m <parent.gguf> --no-mmap=false -p <prompt>` | **best-effort** — out-of-core serving baseline for §5 |
| MLX 4-bit | `mlx_lm.convert -q --q-bits 4 --hf-path <parent> --mlx-path <out> ; mlx_lm.generate --model <out> --prompt <prompt>` | **tuned** — group sizes 32 / 64 |
| Unsloth Dyn 2.0 | `<HF dynamic GGUF id>` in llama.cpp | **tuned** where a dynamic GGUF exists |
| EXL3 / PonyExl3 | `<only where runnable on the target Mac>` | **best-effort** — in-core only; note N/A if it won't run on Apple Silicon |

## Quantization, Doctor, and sub-bit controls

Run these against the same parent revision, calibration/evaluation text, tokenizer, dtype, task suite,
and Studio environment receipt. The receipt must preserve the full expanded command and hashes; the
patterns below identify the required comparison, not permission to substitute a different parent.

| evidence class | exact command pattern | what it controls | admissibility |
|---|---|---|---|
| f16 parent | `STUDIO_TRIPWIRE=1 python3.12 tools/condense/audit_ladder.py <hf-dir> <label> studio <out-prefix>` | Parent PPL and 22-item capability baseline for scalar 4/3/2/1, mixed, residual, and Doctor rows. | Quality reference only; not a compression baseline. |
| scalar quantization | same `SETNAME=studio` run; preserve `4-AWQ`, `3-AWQ`, `2-AWQ`, `1-AWQ`, mixed, and residual rows | Best conventional Hawking floor at exact aggregate bpw, including scale/outlier/side-info overhead. | Candidate only after PPL + relative tripwire; native wins still need packed/resident receipts. |
| VTQ frozen controls | `STUDIO_TRIPWIRE=1 python3.12 tools/condense/audit_ladder.py <hf-dir> <label> subbit <out-prefix>` | `k1/d2` vs `k2/d4` and `k1/d4` vs `k2/d8` isolate vector dimension from nominal symbol payload. | `reconstruction_oracle`, `deployable=false`. |
| VTQ learned curve | same `SETNAME=subbit` run; preserve learned `k1/d2`, `k1/d3`, `k1/d4`, `k1/d8` and block-sweep rows | Measures whether learned per-tensor LUTs and side-information amortization improve reconstruction at real oracle bpw. | `reconstruction_oracle`, `deployable=false`; learned LUT is not packed by `.tq` v2. |
| VTQ + Doctor | same `SETNAME=subbit` run; preserve the mandatory `+dr-r8` rows | Tests restoration while charging exact serialized rank-8 adapter bytes; rank 16 is a conditional follow-up, not a hidden replacement. | No density claim without complete Doctor evidence, PPL, and tripwire; still non-deployable while the VTQ base is an oracle. |
| SUBBIT-0-THEORY | `python3.12 tools/condense/subbit.py measure <hf-dir> <label>` | Order-0 symbol-entropy lower bound (`k=1` is true binary sign) and logical side-information warning; no bulk-symbol codec exists. | `product_gate=false`, `deployable=false`; never compare its bpw to an artifact file. |
| sub-bit footprint math | `python3.12 tools/condense/subbit.py ladder --fit <params-b>` | Capacity arithmetic for 0.75/0.50/0.33 targets. | Probe only; no quality, packing, residency, or speed evidence. |
| speculative target readiness | `python3.12 tools/condense/spec_revive.py --plan <label>` then, only for a durable target, `python3.12 tools/condense/spec_revive.py --status <model.tq> <label>` | Proves that a condensed target has TQ single-versus-batched parity and a cost-aware proposal oracle before any spec runner is considered. | Currently blocked: readiness checker only, no experiment launch. |

For VTQ, symbol payload is `k/d`; effective **oracle** bpw additionally charges the logical trellis side
streams, the complete vector-LUT record (`52 + 4*(2^L)*d` bytes for every vector tensor: SDSC-v2
descriptor/hash envelope plus Q12 i32 entries), outliers, and exact Doctor bytes over the baker's exact quantized
weight count. It is not artifact bpw: container framing/page alignment and pass-through tensors must be
charged from the eventual packed file itself. A row may be operationally
**complete-negative** when it ran correctly but missed quality or efficiency. That lets detached work
advance without converting a negative result into a pass or deleting it from the scaling curve.
The canonical VTQ comparison uses the raw `awq_alpha=0.0` transform with column RHT; a sigma-scaled AWQ
variant is a separate recipe and is inadmissible unless its state is explicitly bound and billed.

## Download manifest (B2 — the parent + teacher + MoE ids the plan pins)

The current Studio supervisor is detached; it waits without holding the heavy lease while unrelated
whole-machine work is above the admission gate. The 0.5B/1.5B/7B parents are staged, 14B and 32B are
path-bound to successful verification markers in isolated staging, and 72B is downloading through the
detached queue. The current GO consumes the verified 14B staging parent directly for its pressure-gated
solo quantization/Doctor run; the later processing supervisor publishes the canonical link and validates
those exact receipts. The 32B full-model rung remains blocked until a measured peak proves the
interactive reserve or a streamed/blockwise checkpoint path is green. This paragraph is a snapshot, not
the source of truth: `python3.12 tools/condense/download_queue.py status` and
`python3.12 tools/condense/processing_queue.py status` report the durable current state without attaching
an expensive observer.

The detached chain reconciles/verifies 32B, downloads/verifies 72B, waits for the 14B processing/coverage
barrier, then admits the 120B MXFP4 source only through the live disk gate. The 284B/13B-active
DeepSeek-V4-Flash bring-up and terminal 1.1T/32B-active Kimi-K2.6 install remain `planned-blocked` until
their own architecture and disk receipts are green. Download
completion never implies processing admission. The processing supervisor and Studio compute share the
exclusive heavy-work lease. The download supervisor has a separate singleton transfer lock and may
overlap compute only while its path/command/ancestry/heartbeat telemetry is current and the continuous
`65 GB Studio peak + max(10 GiB, measured download tree) + 2 GB margin <= 78 GB` gate stays green.
All supervisors share the Studio drain
request, safety budget, and atomic checkpoints; none may delete a source merely to advance the queue.

| rung / role | exact HF id | source size | local now? | where |
|---|---|--:|---|---|
| 0.5B parent | `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB | yes (`scratch/qwen-05b`) | here |
| 1.5B parent | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | yes (`scratch/qwen-15b`) | here |
| 7B parent | `Qwen/Qwen2.5-7B-Instruct` | ~15 GB | yes (`scratch/qwen-7b`) | here |
| 14B parent | `Qwen/Qwen2.5-14B-Instruct` | 29.55 GB bf16 | verified staging | STUDIO |
| 32B parent | `Qwen/Qwen2.5-32B-Instruct` | 65.54 GB bf16 | verified staging; processing blocked at 85 GB estimate | STUDIO |
| 72B parent | `Qwen/Qwen2.5-72B-Instruct` | 145.42 GB bf16 | downloading; processing needs streaming | STUDIO / download-only |
| 120B MoE parent | `openai/gpt-oss-120b` `original/*` | 65.25 GB native MXFP4 | queued | STUDIO / download-only |
| MoE (T1.4) | `deepseek-ai/DeepSeek-V2-Lite` | ~31 GB | no | STUDIO |
| MoE (T1.4) | `Qwen/Qwen3-30B-A3B` | MoE | no | STUDIO |
| KD teacher (T2.5) | `mistralai/Mixtral-8x7B-v0.1` | ~94 GB | no | STUDIO / serve-only |

### FRONTIER — the 100B+ research prize (`studio_run.py --frontier <label>`, exact ids pinned in `tools/condense/studio_manifest.py`)

Serve-oriented (do NOT fit the doctor's f16-resident budget). `.tq` sizes at the recipe's serve bpw.

| label | exact HF id | total params | active (MoE) | serve bpw | `.tq` size | source download |
|---|---|--:|--:|--:|--:|--:|
| 235B-A22B | `Qwen/Qwen3-235B-A22B` | 235B | 22B | 1.34 | ~39 GB (resident target) | ~470 GB bf16; whole-source disk-tight |
| 405B | `meta-llama/Llama-3.1-405B-Instruct` | 405B | dense | 1.34 | ~68 GB (resident target) | ~810 GB bf16; streamed/external only |
| 671B | `deepseek-ai/DeepSeek-V3` | 671B | 37B | 1.00 | ~84 GB (pressure edge/paged) | ~1.34 TB bf16; streamed/external only |
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash-DSpark` | 284B | 13B | 1.34 | ~48 GB (resident target) | ~168 GB FP4+FP8 mixed; first feasible frontier source |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro-DSpark` | 1.6T | 49B | 0.50 | ~100 GB (paged/capacity) | ~894 GB FP4+FP8 mixed; streamed/external only |
| GLM-5.2 | `zai-org/GLM-5.2` | 753B | ~39B | 1.00 | ~94 GB (paged/capacity) | ~1.52 TB bf16; streamed/external only |
| Kimi-K2.6 | `moonshotai/Kimi-K2.6` | 1.1T | 32B | 0.75 | ~103 GB (paged/capacity) | 595.2 GB compressed-tensors; terminal internal-SSD install after guarded source release |
| Kimi-K2.7-Code | `moonshotai/Kimi-K2.7-Code` | 1.1T | 32B | 0.75 | ~103 GB (paged/capacity) | ~595 GB compressed-tensors; streamed/external only |
| Kimi-K2-Instruct | `moonshotai/Kimi-K2-Instruct` | 1.0T | 32B | 0.75 | ~94 GB (paged/capacity) | ~1.03 TB public checkpoint; streamed/external only |

Download with the FASTEST-SOTA path: `python3.12 tools/condense/procure.py <label>` (forces
`HF_HUB_ENABLE_HF_TRANSFER=1` + the `hf_xet` backend + `--max-workers`, so it is link-bound not
software-bound). `procure.py --all-frontier --link-mbps <your link>` prints the full-fit manifest +
per-model ETA; `procure.py --cycle-frontier --link-mbs 300 --efficiency 0.7` prints the practical
cycle plan where source checkpoints are deleted/reclaimed after the `.tq` bake and receipts pass.
Storage planning uses **current free space minus 150 GB**, never nominal SSD capacity, and charges
**64 GB scratch + 32 GB HF/Xet cache**. Recompute `STORAGE_BUDGET_GB=current_free_gb-150` before each
wave; `frontier_ops.py storage-plan --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32 --link-mbs 300 --efficiency 0.7`
prints the checkpointed plan. `procure.py --check` confirms the accelerators
and project-local HF cache are active. For maximal runs use
`procure.py <label> --retries 2 --min-observed-mbs 80 --verify --progress-interval-s 60 --stall-timeout-s 900`;
retries are resumable with fewer workers and `--verify` runs `hf cache verify` against the local dir.
Each real `procure.py <label>` run appends observed MB/s, live progress samples, longest no-progress
window, stall termination status, cache delta, route/HF/DNS/network diagnostics for bad attempts,
retry/verify status, return code, and output tail to
`reports/condense/frontier_downloads.jsonl`, and
`hawking studio status --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32`
surfaces the latest event per model.
`hawking studio lifecycle` distinguishes stalled downloads from generic failures so the next run can
retry with explicit evidence. `frontier_ops.py ledger --refresh-hf`
writes model/revision/license/storage provenance; `frontier_ops.py launch-gate --phase procure`
enforces the model-aware procurement gate, including storage-wave/cache headroom; and
`hawking studio lifecycle --storage-budget-gb "$STORAGE_BUDGET_GB" --scratch-gb 64 --cache-reserve-gb 32`
prints the per-model DAG state and next safe
command. `frontier_ops.py record-event <label> --stage bake|serve|eval --status pass --duration-s N`
feeds compute wall-clock evidence back into the ledger. `frontier_ops.py artifact-inventory <label>` hashes
durable `.tq` outputs; `frontier_ops.py release-source <label> --dry-run` refuses source deletion until
that inventory plus receipt/record evidence exists. `frontier_ops.py refresh` marks plausible unknown
frontier candidates as `REVIEW`; record `hawking studio review-candidate <hf_id> --decision accept|reject|watch`
before a real launch so surprise new releases are explicit, not accidental. `hawking studio review-plan
--refresh <refresh-ledger> --out <review-plan.json>` writes the candidate-decision queue as a durable
artifact.
`hawking studio proof-pack --force --out reports/condense/frontier_proof_pack.local.json` writes the
non-compute claim wall for the full frontier manifest: a signed proof-pack manifest, signed draft
source-provenance, parity, baseline/eval, native serve/RAM-cliff, and experiment envelopes, plus `.local`
claim bundles that hash those drafts and remain inadmissible. The manifest reports `local_signed_count`
separately from final claim-bundle admissibility. It preserves final receipts unless `--force-final` is
explicitly passed.
`hawking studio density-receipt-build --out reports/condense/studio_density_receipt.local.json` and
`hawking studio density-receipt-verify --path reports/condense/studio_density_receipt.local.json` keep a
signed local stabilization snapshot of repo size, largest files, tracked LOC, disk headroom, and generated
artifact/model mass. The receipt records cleanup recommendations only; it does not delete evidence and it
does not unlock baseline, native-serve, RAM-cliff, or public-claim gates.
`hawking studio coverage-plan` prints the claim-phase coverage contract. For each frontier label,
`reports/condense/<LABEL>_baselines.json` must cover llama.cpp Q4_K_M, llama.cpp IQ2_S, llama.cpp
mmap OOC, MLX 4-bit, Unsloth Dyn 2.0, and EXL3/PonyExl3 with either same-box measured rows or explicit
N/A rows with reasons. `reports/condense/<LABEL>_eval.json` must cover ppl multiwindow, capability QA,
math, coding, tool-use, long-context recall, RAM-cliff, and native-serve domains with pass or reasoned
N/A rows. Use `hawking studio coverage-receipt draft <label> --kind both --sign-draft` to create signed
but blocked envelopes, then after real same-box runs fill final rows and run
`hawking studio coverage-receipt sign <label> --kind both` and
`hawking studio coverage-receipt verify <label> --kind both`. Signing refuses missing coverage,
placeholder commands, missing machine fingerprint/environment receipt, missing same-box group, missing
frozen suite or score-set hashes, best-effort baseline rows, and measured/pass rows without a
receipt/artifact/log trace.
`frontier_ops.py launch-gate --phase claim` refuses public claims until those coverage rows,
frontier parity, and the relevant serve/parity receipts are present. Use
`hawking studio parity-receipt draft <label> --sign-draft` to create signed but blocked architecture
parity envelopes, then after real reference-backend and Hawking/native runs fill final rows and run
`hawking studio parity-receipt sign <label>` and `hawking studio parity-receipt verify <label>`.
Signing refuses draft state, placeholder commands, missing config/tokenizer hashes, missing adapter or
tensor-map proof, missing tokenizer/context contracts, missing reference or native trace hashes, loose
logit parity, short greedy-match windows, unsupported custom-code paths without exit receipts, and
unverified family-specific native features. `hawking studio receipt-plan`
prints the strict receipt contract for `reports/condense/<LABEL>_serve.json` and
`reports/condense/<LABEL>_ramcliff.json`; synthetic or modeled RAM-cliff rows are probes only and cannot
unlock a claim. Use `hawking studio receipt-record draft <label> --kind both --sign-draft` to create
signed but blocked native serve/RAM-cliff envelopes, then after real native serve and RAM-cliff runs fill
final measured rows and run `hawking studio receipt-record sign <label> --kind both` and
`hawking studio receipt-record verify <label> --kind both`. Signing refuses draft state, placeholder
commands, missing artifact hashes, non-native or f16-rehydrate serve rows, modeled RAM-cliff rows, and
trace-free evidence; serve needs load, memory, served-forward, and parity traces, and RAM-cliff needs
powermetrics/energy plus baseline traces. `hawking studio serve-capture <label> --artifact <artifact.tq> --bench-json
<serve_report.json> --command '<exact hawking serve bench command>' --load-receipt <trace>
--served-forward-receipt <trace>
--parity-receipt <trace> --force` is the product-facing strict bridge from Hawking's native serve-bench
JSON to a signed `<LABEL>_serve.json`; it refuses f16 rehydrate/fallback reports, missing all-linear/GPU ownership,
missing load/memory proof, missing served-forward/parity pass, nonpositive tok/s, or artifact-hash drift. `frontier_ops.py
serve-capture` is the lower-level equivalent. `hawking studio experiment-plan` prints the expensive-mode
matrix contract for
`reports/condense/<LABEL>_experiment_matrix.json`: seeds, calibration ablations, bpw rungs, expert
allocation, cold/warm cliff repeats, baseline variants, null certifications, and rebake/hash verification.
Use `hawking studio experiment-receipt draft <label> --sign-draft` to create signed but blocked
experiment envelopes, then after real rows are complete run `hawking studio experiment-receipt sign <label>`
and `hawking studio experiment-receipt verify <label>`. Signing refuses draft state, missing depth,
placeholder commands/traces, synthetic rows, missing same-run contract, missing machine fingerprint,
missing environment/artifact/source-provenance receipts, missing artifact/plan hashes, and
pass/measured/certified rows without a concrete row receipt/artifact/log/report reference plus trace
SHA-256.
`hawking studio source-provenance-plan` prints the required source checkpoint provenance path for every
frontier model. Before a public claim, `reports/condense/<LABEL>_source_provenance.json` must be final
and signed with the exact HF revision, source kind, source format, procurement command, download/cache
verification receipt, and file-manifest hash/counts. Compressed or FP4/FP8 sources must explicitly record
`source_is_prequantized=true` plus a source-format receipt; bf16 parents must record
`source_is_prequantized=false` and `source_format=bf16`. Use
`hawking studio source-provenance-receipt draft <label> --sign-draft` before procurement, then after
verified downloads fill the final row and run `hawking studio source-provenance-receipt sign <label>` and
`hawking studio source-provenance-receipt verify <label>`.
`hawking studio claim-bundle-build <label>` signs the final public-claim evidence by SHA-256, and
`hawking studio claim-bundle-verify reports/condense/<LABEL>_claim_bundle.json` rejects missing, stale,
or claim-inadmissible bundles while still reporting `signature_ok` for signed local walls.
`frontier_ops.py launch-gate --phase claim` refuses public claims until
the signed bundle verifies.
`hawking studio license-plan` prints the accepted-terms command for each model. `reviewed` is not
procurement-safe; `hawking studio record-license` must store `status=accepted`, signer, license id,
terms URL, allowed use, redistribution policy, source-retention policy, and note before downloads.
The nine-target default frontier manifest is ~7.42 TB of source downloads plus ~0.73 TB of `.tq`
outputs. At a sustained 300 MB/s it is ~6.9 h download-only serial; at 70% realized throughput from the
same line it is ~9.8 h. Neither full fit nor the old retain-all cycle fits the 1 TB Studio. Whole-source
procurement is currently feasible for V4-Flash; 235B-A22B is disk-tight once reserves are charged, and
larger sources require verified shard-streaming or external storage. Download/bake/verify/release one
eligible source at a time, and never release the source before artifact inventory and receipt verification.

## Honesty rules

- A throughput win is not an efficiency win unless the frozen capability suite, output length, cold/warm
  regime, memory pressure, discarded work, energy, and byte accounting are comparable. FLOPS and nominal
  zero counts do not substitute for measured useful work.
- SUBBIT-0-THEORY and `subbit.py ladder` are probes, never baselines for artifact density or serving.
  VTQ rows are reconstruction oracles until learned LUT/framing state survives a packed round trip,
  actual file bpw is measured, native CPU/Metal parity passes, and resident execution proves no decoded
  parent copy. `k/d` payload is not effective bpw.
- Doctor recovery is compared at exact total bpw: base aggregate bits plus the serialized adapter bytes
  divided by the baker's exact quantized-weight count. Rank 8 is the first VTQ recovery control; any rank
  escalation is a separately billed row. Old underbilled rank-64 receipts are not density evidence.
  Restart is currently at the config boundary: interruption reruns that Doctor config. Atomic best/latest
  adapters are retained, but exact optimizer, RNG/sampler, and microstep state are not resumed, so a
  restarted run is not claimed as bit-exact continuation.
- A bounded negative result is operationally complete and must remain in the ledger. It may advance the
  detached queue but cannot satisfy a promotion, winner, native-serve, or public-claim gate.
- Speculative decoding is baseline-comparable only when the single-token target and batched verifier use
  the same hash-bound `.tq` artifact with exact token parity and zero skipped cases. Charge draft, verify,
  synchronization, rejected-token, dual-residency, and p95 costs; extra unified memory alone is not a win.
- Downloads and processing are interruption-safe only at a durable checkpoint. Before moving the Studio,
  drain new launches, finish or gracefully stop the active writer, verify the latest HF cache/artifact and
  lifecycle record, and require `SAFE TO UNPLUG`. Resume in the same local directory after environment
  verification; do not treat a merely populated directory as a verified download.
- Every baseline receipt sets `baseline_best_effort` truthfully. `true` => the baseline can
  only support a *contingent* or *negative* result (R8).
- Missing baseline rows are not neutral. Use same-box measured evidence or an explicit N/A with a
  reason; silence blocks the claim gate. Unsigned, draft, tampered, placeholder-command, or trace-free
  coverage receipts do not unlock claim bundles. Final coverage receipts must bind the run to a named
  Studio machine, machine fingerprint, environment receipt, same-box group, frozen suite hash, and frozen
  score-set hash.
- Missing or loose parity rows are not neutral. A claim-admissible parity receipt needs schema,
  config/tokenizer hashes, exact commands, commit, reference backend, architecture adapter and
  tensor-map receipts, tokenizer/context contracts, reference/native trace hashes, logit and
  greedy-token thresholds, unsupported-by-design exits where applicable, and family-specific native
  feature verification. Unsigned, draft, tampered, placeholder-command, trace-free, loose-threshold, or
  feature-incomplete parity receipts do not unlock claim bundles.
- Missing or loose native-serve/RAM-cliff rows are not neutral. A claim-admissible serve receipt needs
  schema, artifact hash, exact commands, commit, Studio machine class, native `.tq`, no f16 rehydrate,
  all-linear/GPU ownership, load and resident-memory proof, parity pass, and positive tok/s. A RAM-cliff receipt additionally needs
  measured source, >10x resident-vs-swap tok/s, Q4_K overflow, and lower resident J/tok. Unsigned,
  draft, tampered, placeholder-command, synthetic/modelled, or trace-free native receipts do not unlock
  claim bundles.
- Missing experiment-depth rows are not neutral. A public frontier claim needs the expensive-mode
  matrix: multiple seeds, ablations, cold/warm repeats, baseline variants, null results, and rebake/hash
  verification. Final matrices must bind all rows to the same Studio run with machine fingerprint,
  environment receipt, artifact inventory receipt/hash, source-provenance receipt, experiment-plan hash,
  exact commands, row trace references, row trace hashes, and null/N/A reasons. Unsigned, draft, tampered,
  placeholder-command, trace-free, synthetic, or depth-incomplete experiment matrices do not unlock claim
  bundles.
- Missing signed claim bundles are not neutral. A public frontier claim needs a verifying
  `<LABEL>_claim_bundle.json` that hashes source provenance, parity, coverage, serve, RAM-cliff, and
  experiment evidence.
- Missing source-provenance rows are not neutral. A public frontier claim needs a signed
  `<LABEL>_source_provenance.json` that pins the HF revision, source format, procurement command,
  download/verify receipt, and file-manifest evidence, especially for compressed K2/DeepSeek sources.
- Missing accepted license terms are not neutral. A model can be reviewed conceptually and still blocked
  operationally until the accepted terms and source handling policy are recorded.
- Compare under the **same memory pressure** (same KV length, same resident load) — never a
  generous Hawking run vs a starved baseline (§20.10).
- Headline numbers come from **CPU-bf16**; MPS is lab-only and needs a CPU-bf16 confirmation
  (`mps_headline` + `cpu_bf16_confirmed`, §20.3 rule 7).
