#!/usr/bin/env bash
# tools/bench/eh_market_bench.sh — Event Horizon proposal-market bench driver.
#
# WHAT THIS RUNS
# ==============
# Drives eh_market_bench.py (smoke mode by default) over three decode arms:
#   A  plain greedy         HAWKING_QWEN_USER_DRAFT=0  HAWKING_QWEN_EVENT_HORIZON=0
#   B  n-gram only          HAWKING_QWEN_USER_DRAFT=1  HAWKING_QWEN_EVENT_HORIZON=0
#   C  full free market     HAWKING_QWEN_USER_DRAFT=1  HAWKING_QWEN_EVENT_HORIZON=1
#
# PRIMARY METRIC: accepted_tps = completion_tokens / decode_s
#   Arm A  accepted_tps == dec_tps (no drafts, every token is greedy)
#   Arm B  accepted_tps > dec_tps when n-gram hits; ratio B/A = ngram payoff
#   Arm C  accepted_tps >= B (router adds suffix-array as second free slot)
#
# SMOKE (default)
#   GPU footprint: 3 arms x 16 tokens = 48 forward steps.
#   Run during a KD job: the GPU is MPS-shared; hawking runs at nice -n 19
#   via the embedded subprocess call (see eh_market_bench.py run_bench()).
#   NOTE: GPU is owned by a running KD job — keep it TINY.
#
# FULL BENCH (--full)
#   Runs 3 prompts x 3 trials per arm. Launch after the KD job finishes.
#
# OUTPUTS
#   reports/eh_market_bench_smoke.md   — Markdown table (smoke)
#   reports/eh_market_bench_smoke.json — machine-readable (smoke)
#   reports/eh_market_bench_full.md    — Markdown table (full)
#   reports/eh_market_bench_full.json  — machine-readable (full)
#
# USAGE
#   bash tools/bench/eh_market_bench.sh                      # smoke (default)
#   bash tools/bench/eh_market_bench.sh --full               # full bench
#   MODEL=/path/to/model.gguf bash tools/bench/eh_market_bench.sh
#   BIN=/path/to/hawking bash tools/bench/eh_market_bench.sh
#
# REQUIREMENTS
#   * cargo build --release -p hawking  (or set BIN= to a pre-built binary)
#   * python3  (no new pip deps beyond tools/training/requirements.txt)
#
# CO-EXISTENCE
#   hawking runs as a child process of python; if you want extra nice/taskpolicy
#   wrapping, set HAWKING_NICE=1 and call this script via:
#     nice -n 19 taskpolicy -b bash tools/bench/eh_market_bench.sh
#   The bench does not auto-nice its subprocesses (they are already short GPU
#   bursts; the KD job holds the MPS context and will preempt as needed).

set -uo pipefail
cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
MODEL="${MODEL:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
BIN="${BIN:-./target/release/hawking}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"
FULL=0

die() { printf 'error: %s\n' "$*" >&2; exit 64; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL=1; shift;;
    --model) MODEL="$2"; shift 2;;
    --bin)   BIN="$2";   shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    -h|--help)
      sed -n '2,60p' "$0"
      exit 0
      ;;
    *) die "unknown arg: $1";;
  esac
done

# Validate
[[ -f "$MODEL" ]] || die "model not found: $MODEL (set MODEL= or run from repo root)"
[[ -x "$BIN"   ]] || die "binary not found/executable: $BIN (cargo build --release -p hawking)"
"$PY" -c 'import json' 2>/dev/null || die "python3 missing"

printf '=== eh_market_bench.sh ===\n'
printf '  model  : %s\n' "$MODEL"
printf '  binary : %s\n' "$BIN"
printf '  profile: %s\n' "$PROFILE"
printf '\n'

if [[ "$FULL" == 1 ]]; then
  printf '  MODE: full bench (3 prompts x 3 trials per arm)\n'
  printf '  WARNING: run this after the KD job finishes (significant GPU use).\n\n'
  "$PY" tools/bench/eh_market_bench.py \
    --model "$MODEL" \
    --bin   "$BIN"   \
    --kernel-profile "$PROFILE" \
    --max-tokens 32  \
    --trials 3       \
    --verbose        \
    --out reports/eh_market_bench_full.md
else
  printf '  MODE: smoke (1 prompt, 16 tokens, 1 trial per arm)\n'
  printf '  GPU footprint: 3 x 16 = 48 forward steps. Safe during KD job.\n\n'
  "$PY" tools/bench/eh_market_bench.py \
    --smoke          \
    --model "$MODEL" \
    --bin   "$BIN"   \
    --kernel-profile "$PROFILE" \
    --out reports/eh_market_bench_smoke.md
fi

exit $?
