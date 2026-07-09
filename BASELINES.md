# BASELINES.md — baseline neutrality spec

> Spec source: `docs/plans/studio_maximization_2026_06_27.md` §20.11 / §20-BASELINES.
> The honest baseline set, each with its **exact command** and a **best-effort note**.
> Rule: a baseline marked **best-effort** can support a *contingent* or *negative* claim
> but **never a public win** (§20.3 invalidation rule 8). No rhetorical sandbagging — if a
> baseline beats Hawking, its receipt ships unchanged.

## The baseline set (run all on the SAME named machine, same frozen suite)

| baseline | exact command (fill in `<...>`) | tuning status |
|---|---|---|
| llama.cpp Q4_K_M | `llama-quantize <parent.gguf> <out.gguf> Q4_K_M ; llama-bench -m <out.gguf> -p <prompt>` | **tuned** — also try Q4_K_S, IQ4_XS |
| llama.cpp mmap OOC | `llama-cli -m <parent.gguf> --no-mmap=false -p <prompt>` | **best-effort** — out-of-core serving baseline for §5 |
| MLX 4-bit | `mlx_lm.convert -q --q-bits 4 --hf-path <parent> --mlx-path <out> ; mlx_lm.generate --model <out> --prompt <prompt>` | **tuned** — group sizes 32 / 64 |
| Unsloth Dyn 2.0 | `<HF dynamic GGUF id>` in llama.cpp | **tuned** where a dynamic GGUF exists |
| EXL3 / PonyExl3 | `<only where runnable on the target Mac>` | **best-effort** — in-core only; note N/A if it won't run on Apple Silicon |

## Download manifest (B2 — the parent + teacher + MoE ids the plan pins)

Record-only on the M3 Pro 18 GB; download on the Studio (disk-bound now).

| rung / role | exact HF id | bf16 size | local now? | where |
|---|---|--:|---|---|
| 0.5B parent | `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB | yes (`scratch/qwen-05b`) | here |
| 1.5B parent | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | yes (`scratch/qwen-15b`) | here |
| 7B parent | `Qwen/Qwen2.5-7B-Instruct` | ~15 GB | yes (`scratch/qwen-7b`) | here |
| 14B parent | `Qwen/Qwen2.5-14B-Instruct` | ~28 GB | no | DO-NOW download (fits 55 GB) |
| 32B parent | `Qwen/Qwen2.5-32B-Instruct` | ~64 GB | no (only Q4_K GGUF) | STUDIO |
| 70B serve | `Qwen/Qwen2.5-72B-Instruct` | ~140 GB | no | STUDIO |
| MoE (T1.4) | `deepseek-ai/DeepSeek-V2-Lite` | ~31 GB | no | STUDIO |
| MoE (T1.4) | `Qwen/Qwen3-30B-A3B` | MoE | no | STUDIO |
| KD teacher (T2.5) | `mistralai/Mixtral-8x7B-v0.1` | ~94 GB | no | STUDIO / serve-only |

### FRONTIER — the 100B+ research prize (`studio_run.py --frontier <label>`, exact ids pinned in `tools/condense/studio_manifest.py`)

Serve-oriented (do NOT fit the doctor's f16-resident budget). `.tq` sizes at the recipe's serve bpw.

| label | exact HF id | total params | active (MoE) | serve bpw | `.tq` size | source download |
|---|---|--:|--:|--:|--:|--:|
| 235B-A22B | `Qwen/Qwen3-235B-A22B` | 235B | 22B | 1.34 | ~39 GB (COMFY) | ~470 GB bf16 |
| 405B | `meta-llama/Llama-3.1-405B-Instruct` | 405B | dense | 1.34 | ~68 GB (COMFY) | ~810 GB bf16, gated |
| 671B | `deepseek-ai/DeepSeek-V3` | 671B | 37B | 1.00 | ~84 GB (resident edge) | ~1.34 TB bf16 |
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash-DSpark` | 284B | 13B | 1.34 | ~48 GB (1M-context flash) | ~168 GB FP4+FP8 mixed |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro-DSpark` | 1.6T | 49B | 0.50 | ~100 GB (research stretch) | ~894 GB FP4+FP8 mixed |
| GLM-5.2 | `zai-org/GLM-5.2` | 753B | ~39B | 1.00 | ~94 GB (resident capstone) | ~1.52 TB bf16 |
| Kimi-K2.6 | `moonshotai/Kimi-K2.6` | 1.1T | 32B | 0.75 | ~103 GB (resident stretch) | ~595 GB compressed-tensors |
| Kimi-K2.7-Code | `moonshotai/Kimi-K2.7-Code` | 1.1T | 32B | 0.75 | ~103 GB (coding control) | ~595 GB compressed-tensors |
| Kimi-K2-Instruct | `moonshotai/Kimi-K2-Instruct` | 1.0T | 32B | 0.75 | ~94 GB (text control) | ~1.03 TB public checkpoint |

Download with the FASTEST-SOTA path: `python3.12 tools/condense/procure.py <label>` (forces
`HF_HUB_ENABLE_HF_TRANSFER=1` + the `hf_xet` backend + `--max-workers`, so it is link-bound not
software-bound). `procure.py --all-frontier --link-mbps <your link>` prints the full-fit manifest +
per-model ETA; `procure.py --cycle-frontier --link-mbs 300 --efficiency 0.7` prints the practical
cycle plan where source checkpoints are deleted/reclaimed after the `.tq` bake and receipts pass.
`frontier_ops.py storage-plan --storage-budget-gb 8000 --link-mbs 300 --efficiency 0.7` prints the
aggressive checkpointed wave plan for the 8 TB Studio. `procure.py --check` confirms the accelerators
and project-local HF cache are active. For maximal runs use
`procure.py <label> --retries 2 --min-observed-mbs 80 --verify --progress-interval-s 60 --stall-timeout-s 900`;
retries are resumable with fewer workers and `--verify` runs `hf cache verify` against the local dir.
Each real `procure.py <label>` run appends observed MB/s, live progress samples, longest no-progress
window, stall termination status, cache delta, route/HF/DNS/network diagnostics for bad attempts,
retry/verify status, return code, and output tail to
`reports/condense/frontier_downloads.jsonl`, and
`hawking studio status --storage-budget-gb 8000` surfaces the latest event per model.
`hawking studio lifecycle` distinguishes stalled downloads from generic failures so the next run can
retry with explicit evidence. `frontier_ops.py ledger --refresh-hf`
writes model/revision/license/storage provenance; `frontier_ops.py launch-gate --phase procure`
enforces the model-aware procurement gate, including storage-wave/cache headroom; and
`hawking studio lifecycle --storage-budget-gb 8000` prints the per-model DAG state and next safe
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
claim bundles that hash those drafts and remain inadmissible. It preserves final receipts unless
`--force-final` is explicitly passed.
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
or claim-inadmissible bundles. `frontier_ops.py launch-gate --phase claim` refuses public claims until
the signed bundle verifies.
`hawking studio license-plan` prints the accepted-terms command for each model. `reviewed` is not
procurement-safe; `hawking studio record-license` must store `status=accepted`, signer, license id,
terms URL, allowed use, redistribution policy, source-retention policy, and note before downloads.
The nine-target default frontier manifest is ~7.42 TB of source downloads plus ~0.73 TB of `.tq`
outputs. At a sustained 300 MB/s it is ~6.9 h download-only serial; at 70% realized throughput from the
same line it is ~9.8 h. The full-fit view is ~8.2 TB, while the conservative cycle-through view peaks
around ~2.6 TB with a 200 GB scratch reserve, 128 GB HF/Xet cache reserve, and all `.tq` outputs
retained. Download smallest-first / biggest-last so the ladder + condense start immediately on the
smaller sources.

## Honesty rules

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
