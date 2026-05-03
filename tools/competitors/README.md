# tools/competitors

Head-to-head benchmark plumbing. Spawns `llama-cli` and the
`dismantle` release binary against the same prompt set, captures
their stdout/stderr, parses the per-backend tok/s line, and emits a
unified JSON document.

(MLX is treated as an analysis-only competitor in
`docs/competitive_audit.md`; the harness here measures only
GGUF-format engines so the comparison is apples-to-apples on both
weights file and quantization scheme.)

## Files

| File | Purpose |
|---|---|
| `run_competitors.sh` | The harness. Reads `prompts.txt`, runs every backend × every prompt × `$TRIALS` trials, writes `results.json` and `versions.json`. |
| `prompts.txt` | 20-prompt corpus, one per line, format `TIER\|<text>` where TIER ∈ {SHORT, MED, LONG}. |
| `versions.json` | (gitignored) Pinned versions of all four backends, captured at run time. |
| `results.json`  | (gitignored) Output of the harness. The audit doc cites this. |

## Pre-flight

```sh
# Once:
./tools/fetch-model.sh            # downloads ~9 GB DeepSeek-V2-Lite Q4_K_M
brew install llama.cpp            # provides llama-cli  (already done on the audit Mac)
cargo build --release             # produces target/release/dismantle
```

## Run

```sh
./tools/competitors/run_competitors.sh           # 3 trials per cell (default)
./tools/competitors/run_competitors.sh 5         # 5 trials, more honest
```

Cell count: 20 prompts × 2 backends × 3 trials = 120 invocations.
Wall-clock dominated by dismantle's Phase-0 CPU speed (~10 min per
256-token prompt); llama.cpp finishes each prompt in seconds. Plan
on overnight for the full matrix at v0.0.1.

For a single-prompt smoke instead of the full matrix, use
`./tools/competitors/smoke.sh` — same backend wiring, one prompt,
roughly 12 minutes total.

## Honesty rules

(Per `docs/m3_audit.md`.)

- 5 minutes idle between full backends to let thermals settle (the
  harness does only a 2 s gap between backends within a prompt; for
  publishable numbers re-run with manual breaks).
- Power adapter connected, lid open, hard surface, macOS Low Power
  Mode off.
- Median of TRIALS trials per cell.
- The `versions.json` file is the single source of truth for what
  was measured. Any number cited in the audit doc must reference it.
