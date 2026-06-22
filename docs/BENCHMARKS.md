# Hawking Benchmarks — SOTA comparison

Hawking ships a single comprehensive head-to-head harness that compares it against
the closest local-inference SOTA — **llama.cpp** and **MLX** — across every axis,
and also runs a full Hawking self-diagnostic (including the capabilities the others
do not have, with the nearest SOTA equivalent noted).

Harness: [`tools/bench/compare_sota.sh`](../tools/bench/compare_sota.sh)

## Run it (clean room)

For trustworthy *absolute* numbers, **quit Claude/Cursor and every other heavy GPU
app first** — a background AI session inflates tps/J by ~4–5×. Then, in a terminal:

```bash
cd ~/Downloads/hawking
cargo build --release -p hawking            # ensure the binary is current
bash tools/bench/compare_sota.sh            # full run (wall-clock is not optimized)
```

Useful variants:

```bash
QUICK=1 bash tools/bench/compare_sota.sh            # fewer trials/contexts/prompts
STRICT_CLEAN=1 bash tools/bench/compare_sota.sh     # abort if Claude is still running
TRIALS=5 TOK=256 bash tools/bench/compare_sota.sh   # heavier, lower-variance
```

The report lands in `reports/sota-compare/<timestamp>/report.md` (plus per-task
answers and the full diagnostic transcript). `reports/` is git-ignored.

## What it measures

| # | Dimension | How |
|---|---|---|
| 1 | **Footprint / compression** | on-disk bpw; Hawking + llama.cpp share the GGUF (identical), MLX uses its own 4-bit. Plus the Hawking-only out-of-core `press` planner. |
| 2 | **Speed** | warm-median decode tps + prefill on the **same** Qwen2.5-3B-Q4_K_M (Hawking `generate` vs `llama-bench` vs `mlx_lm.generate`). |
| 3 | **Long context (the moat)** | decode tps vs context: the RWKV-7 **SSM** path stays flat (no KV cache) while transformers fall off the KV wall. |
| 4 | **Quality** | deterministic task prompts (math/JSON/retrieval), greedy, side-by-side, pass/fail on the expected answer. |
| 5 | **Hawking diagnostic** | every subcommand (`version`/`doctor --json`/`fit`/`press`/`stats`/…) with the nearest SOTA equivalent for the unique ones. |
| 6 | **Energy** (optional) | J/tok via `macmon` (`tools/bench/energy_paired.sh` / `phase_joules.sh`). |

## Robustness notes (why this harness does not hang)

Earlier comparison runs hung in an endless `>>>>>` loop. Root causes, both handled:

- **Modern `llama-cli` defaults to interactive chat** and rejects `-no-cnv` (it
  loops `>` forever). The harness uses **`llama-completion`** for generation and
  `llama-bench` for speed — both non-interactive — with stdin from `/dev/null`.
- **macOS ships no `timeout`.** The harness defines a portable `TO()` wrapper
  (`timeout`/`gtimeout`/`perl alarm` fallback) and wraps *every* external call, so
  any runaway is killed rather than hanging the run.

A missing engine is **skipped with a clear note + install hint**, never a failure.

## Models (portable across all three frameworks)

The harness auto-prefers the **Qwen2.5-7B-Instruct** portable model when present
(same base model in each framework's native quant), else falls back to the 3B:

| framework | artifact | path / id |
|---|---|---|
| Hawking + llama.cpp | GGUF Q4_K_M | `models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` |
| MLX | MLX 4-bit | `models/mlx-Qwen2.5-7B-Instruct-4bit` (or `mlx-community/Qwen2.5-7B-Instruct-4bit`) |
| Hawking (SSM moat) | GGUF | `models/rwkv7-g1-04-sft-Q4_K_M.gguf` |

Fetch the 7B (the CLI is `hf`, not the deprecated `huggingface-cli`):

```bash
hf download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models
hf download mlx-community/Qwen2.5-7B-Instruct-4bit --local-dir models/mlx-Qwen2.5-7B-Instruct-4bit
```

Override either with `QWEN_GGUF=… MLX_MODEL=… bash tools/bench/compare_sota.sh`.

## MLX

MLX runs from a Python env that has `mlx_lm` (on this machine: **`python3.12`,
mlx_lm 0.31.3**). The harness auto-probes `$MLX_PYTHON`, `python3.12`,
`~/.mlxenv/bin/python`, then `python3`. To (re)install or pin:

```bash
python3.12 -m pip install -U mlx-lm
MLX_PYTHON=python3.12 bash tools/bench/compare_sota.sh   # pin a specific interpreter
```

## Where Hawking is differentiated

- **Long-context throughput** — the RWKV-7 SSM path has no KV cache, so decode tps
  stays flat as context grows (transformers fall off the KV wall). No shipping
  competitor has an optimized small instruct SSM.
- **Per-Mac fit planning** — `hawking fit` / `doctor --json` report the usable
  envelope for the exact machine; `serve --auto` picks the strongest stable config
  capability-first. llama.cpp/MLX/ollama have no equivalent.
- **Out-of-core condense** — `hawking press` plans quantizing a parent that does not
  fit fully resident; `llama-quantize`/AWQ/GPTQ require the full parent in memory.

The other axes (short-context decode tps, compression at a given bpw, instruct
quality) are head-to-head on the same artifact — that is the honest comparison the
harness produces.
