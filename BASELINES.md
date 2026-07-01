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

### FRONTIER — the 100B+ research prize (`studio_run.py --frontier <label>`, exact ids pinned in `FRONTIER` in `tools/condense/studio_run.py`)

Serve-oriented (do NOT fit the doctor's f16-resident budget). `.tq` sizes at the recipe's serve bpw.

| label | exact HF id | total params | active (MoE) | serve bpw | `.tq` size |
|---|---|--:|--:|--:|--:|
| 235B-A22B | `Qwen/Qwen3-235B-A22B` | 235B | 22B | 1.34 | ~39 GB (COMFY) |
| 405B | `meta-llama/Llama-3.1-405B-Instruct` | 405B | dense | 1.34 | ~68 GB (TIGHT) |
| 671B | `deepseek-ai/DeepSeek-V3` | 671B | 37B | 1.00 | ~84 GB (the EDGE) |
| 744B | `zai-org/GLM-4.5` | 744B | 32B | 0.75 | ~70 GB (research) |

Download each with `hf download <id> --local-dir scratch/<dir>` per the `FRONTIER` table — large
(hundreds of GB), do them serially, confirm free disk before each (`size_frontier.py --ceiling`).

## Honesty rules

- Every baseline receipt sets `baseline_best_effort` truthfully. `true` => the baseline can
  only support a *contingent* or *negative* result (R8).
- Compare under the **same memory pressure** (same KV length, same resident load) — never a
  generous Hawking run vs a starved baseline (§20.10).
- Headline numbers come from **CPU-bf16**; MPS is lab-only and needs a CPU-bf16 confirmation
  (`mps_headline` + `cpu_bf16_confirmed`, §20.3 rule 7).
