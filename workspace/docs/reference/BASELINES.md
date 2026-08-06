# BASELINES.md — baseline neutrality spec

> Active living docs: `GO.md`, `docs/dead_levers.md`, `docs/serve.md`, `docs/BENCHMARKS.md`.
> Historical Studio plan docs archived at tag `pre-floor-prune-20260728`. Exact command + best-effort note required.
> **Rule:** best-effort baselines support contingent/negative claims only — **never a public win**. If a baseline beats Hawking, its receipt ships unchanged.
>
> **Entrypoint note (2026-07-28):** `[SEALED] tools/condense/studio_run.py` / `hawking studio` not in active tree. Commands naming them are historical. Frontier labels live in `tools/condense/studio_manifest.py` (library only).

## Machine + measurement contract

Named machine: M3 Ultra Studio, 96 GB UMA, 819 GB/s advertised BW, 1 TB SSD. Same machine + frozen suite for all baselines. Record cold/warm, output length, TTFT, inter-token, p50/p95 wall, useful/SLO goodput, J/accepted-token, bytes moved/resident, capability per joule/byte/resident/active-param/wall-s, peak UMA, pressure, swap delta, free disk, thermal. Capability = completion clearing frozen quality gate; rejected speculative tokens are not useful work.

The coding-agent app is a protected interactive tenant → use **78 GiB** process admission budget, not full 96 GB. Storage decisions use **current free space** with **150 GB hard floor + 64 GB scratch + 32 GB HF/Xet cache**. Detached supervisors own pressure/swap/thermal/RSS/disk; attached chat is not the monitor.

## External baselines (same box)

| baseline | exact command (fill `<...>`) | tuning |
|---|---|---|
| llama.cpp Q4_K_M | `llama-quantize <parent.gguf> <out.gguf> Q4_K_M ; llama-bench -m <out.gguf> -p <prompt>` | **tuned** — also Q4_K_S, IQ4_XS |
| llama.cpp mmap OOC | `llama-cli -m <parent.gguf> --no-mmap=false -p <prompt>` | **best-effort** — OOC for §5 |
| MLX 4-bit | `mlx_lm.convert -q --q-bits 4 --hf-path <parent> --mlx-path <out> ; mlx_lm.generate --model <out> --prompt <prompt>` | **tuned** — groups 32/64 |
| Unsloth Dyn 2.0 | `<HF dynamic GGUF id>` in llama.cpp | **tuned** where dynamic GGUF exists |
| EXL3 / PonyExl3 | `<only where runnable on target Mac>` | **best-effort** — N/A if not Apple Silicon |

## Quantization / Doctor / sub-bit controls

Same parent revision, calib/eval text, tokenizer, dtype, suite, Studio environment receipt. Patterns identify required comparison — not permission to swap parents.

| class | command pattern | controls | admissibility |
|---|---|---|---|
| f16 parent | `STUDIO_TRIPWIRE=1 python3.12 tools/condense/audit_ladder.py <hf-dir> <label> studio <out-prefix>` | Parent PPL + 22-item baseline for scalar/mixed/Doctor | Quality ref only |
| scalar quant | same `SETNAME=studio`; keep 4/3/2/1-AWQ, mixed, residual | Best conventional floor at exact aggregate bpw | After PPL+tripwire; native needs packed/resident |
| VTQ frozen | `… audit_ladder.py … subbit …` | k1/d2 vs k2/d4; k1/d4 vs k2/d8 | `reconstruction_oracle`, `deployable=false` |
| VTQ learned | same subbit; learned k1/d{2,3,4,8} + block-sweep | Learned LUT / side-info amortization | Oracle only; not packed by `.tq` v2 |
| VTQ + Doctor | mandatory `+dr-r8` rows | Restoration charging rank-8 adapter bytes | No density without complete Doctor evidence |
| SUBBIT-0-THEORY | `python3.12 tools/condense/subbit.py measure <hf-dir> <label>` | Order-0 entropy lower bound | `product_gate=false`; never vs artifact file |
| sub-bit footprint | `python3.12 tools/condense/subbit.py ladder --fit <params-b>` | Capacity math 0.75/0.50/0.33 | Probe only |
| speculative readiness | `spec_revive.py --plan` then `--status <model.tq> <label>` | TQ single-vs-batched parity + cost oracle | Blocked: readiness only |

VTQ oracle bpw charges trellis side streams, full vector-LUT (`52+4*(2^L)*d`), outliers, Doctor bytes over baker weight count — not file bpw until packed. Canonical VTQ uses raw `awq_alpha=0.0` + column RHT; sigma-scaled AWQ is a separate billed recipe. Complete-negative rows stay on the curve.

## Download manifest (pinned parents / teachers)

Durable truth: `python3.12 tools/condense/download_queue.py status` and `processing_queue.py status`. Snapshot: 0.5B/1.5B/7B staged; 14B/32B verified staging; 72B download path; 120B MXFP4 through disk gate only. Download completion ≠ processing admission. Exclusive heavy-work lease shared by processing+Studio; download has separate transfer lock with continuous `65 GB peak + max(10 GiB, download tree) + 2 GB ≤ 78 GB` gate.

| rung / role | exact HF id | notes |
|---|---|---|
| 0.5B parent | `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB; `scratch/qwen-05b` |
| 1.5B parent | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB |
| 7B parent | `Qwen/Qwen2.5-7B-Instruct` | ~15 GB |
| 14B parent | `Qwen/Qwen2.5-14B-Instruct` | 29.55 GB bf16 verified staging |
| 32B parent | `Qwen/Qwen2.5-32B-Instruct` | 65.54 GB; processing blocked at ~85 GB estimate |
| 72B parent | `Qwen/Qwen2.5-72B-Instruct` | 145.42 GB; needs streaming |
| 120B MoE | `openai/gpt-oss-120b` `original/*` | 65.25 GB MXFP4 queued |
| MoE T1.4 | `deepseek-ai/DeepSeek-V2-Lite` | ~31 GB |
| MoE T1.4 | `Qwen/Qwen3-30B-A3B` | MoE |
| KD teacher | `mistralai/Mixtral-8x7B-v0.1` | ~94 GB serve-only |

### FRONTIER 100B+ (ids in `studio_manifest.py`; sealed `studio_run.py --frontier` removed)

Serve-oriented; do not assume f16-resident doctor budget.

| label | exact HF id | params / active | serve bpw | `.tq` target |
|---|---|--:|--:|--:|
| 235B-A22B | `Qwen/Qwen3-235B-A22B` | 235B / 22B | 1.34 | ~39 GB |
| 405B | `meta-llama/Llama-3.1-405B-Instruct` | 405B dense | 1.34 | ~68 GB |
| 671B | `deepseek-ai/DeepSeek-V3` | 671B / 37B | 1.00 | ~84 GB |
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash-DSpark` | 284B / 13B | 1.34 | ~48 GB |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro-DSpark` | 1.6T / 49B | 0.50 | ~100 GB |
| GLM-5.2 | `zai-org/GLM-5.2` | 753B / ~39B | 1.00 | ~94 GB |
| Kimi-K2.6 | `moonshotai/Kimi-K2.6` | 1.1T / 32B | 0.75 | ~103 GB |
| Kimi-K2.7-Code | `moonshotai/Kimi-K2.7-Code` | 1.1T / 32B | 0.75 | ~103 GB |
| Kimi-K2-Instruct | `moonshotai/Kimi-K2-Instruct` | 1.0T / 32B | 0.75 | ~94 GB |

Procurement: `python3.12 tools/condense/procure.py <label>` (HF transfer + xet). Storage: `STORAGE_BUDGET_GB=current_free_gb-150`; charge 64+32 GB reserves. `frontier_ops.py` owns storage-plan, ledger, launch-gate (procure/claim), artifact-inventory, release-source, record-event, serve-capture. Sealed `hawking studio *` receipt/claim/coverage/parity/provenance commands remain historical; active gates refuse public claims without signed same-box coverage, parity, serve, RAM-cliff, experiment depth, source provenance, license acceptance, and verifying claim bundles. Nine-target default frontier ~7.42 TB source + ~0.73 TB `.tq`; cycle one eligible source at a time; never release source before inventory+receipt verify.

## Honesty rules (binding)

- Throughput ≠ efficiency unless frozen capability, length, cold/warm, pressure, discarded work, energy, and bytes are comparable.
- SUBBIT-0-THEORY / ladder probes are never artifact density baselines; VTQ is oracle until packed round-trip + native parity + no decoded parent copy.
- Doctor recovery at exact total bpw (base + serialized adapter / baker weight count); rank-8 first; restarts not bit-exact continuations.
- Bounded negatives stay in the ledger; they advance queues but not promotion/claim gates.
- Spec comparable only with same hash-bound `.tq`, exact token parity, full cost charge (draft/verify/sync/reject/dual-residency/p95).
- Downloads/processing interruption-safe only at durable checkpoints; `SAFE TO UNPLUG` before move.
- `baseline_best_effort=true` ⇒ contingent/negative only (R8). Missing rows block claims: need same-box measured or explicit N/A with reason; bind machine fingerprint, environment receipt, suite/score hashes.
- Parity/serve/RAM-cliff/experiment/source-provenance/license/claim-bundle incompleteness is not neutral — each needs signed final evidence with traces; synthetic/modelled RAM-cliff cannot unlock claims.
- Compare under same memory pressure; headline numbers from **CPU-bf16** (`mps_headline` needs `cpu_bf16_confirmed`).
