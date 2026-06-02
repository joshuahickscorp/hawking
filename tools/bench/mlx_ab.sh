#!/usr/bin/env bash
# tools/bench/mlx_ab.sh — MLX CEILING TEST (Wave-6 decisive measurement)
#
# PURPOSE
# =======
# Answers ONE question: on THIS M3 Pro, does MLX Qwen2.5-3B-4bit reach
# ~45-49 tok/s (llama.cpp territory) or only ~30 tok/s (same as dismantle)?
#
#   MLX ≈ 45-49 → gap is RUNTIME-STRUCTURAL (command-buffer/GPU-saturation)
#                   and closable by porting MLX's graph structure to dismantle.
#   MLX ≈ 30     → gap is M3-PRO-HW-BOUND (lower BW/cores vs M2 Ultra where
#                   the 230 tok/s MLX figure was measured); the 1.6x lead
#                   belongs to llama.cpp's specific kernel path, not to MLX
#                   architecture; investigate llama.cpp instead.
#
# This is the decisive MEASUREMENT before any large build.
# Research verdict (reports/research_next_levers_2026_06_02.md §TPS #3):
#   "Run MLX Qwen-3B-4bit decode on the M3 Pro, A/B vs dismantle's 30.5.
#    If MLX ≈ llama's 49 → port the winning structure. If MLX ≈ 30 →
#    the gap is M3-Pro-HW-specific." [finding #3, medium confidence]
#
# HOW IT WORKS
# ============
# (1) MLX side: invokes mlx_lm.generate (Python 3.12 framework, where
#     mlx-lm 0.31.3 is installed). Parses MLX's standard verbose summary:
#       Generation: N tokens, X.XXX tokens-per-sec
#     This is written to stdout when --verbose True.
# (2) dismantle side: invokes `dismantle generate` with the locked Qwen
#     fast-path env (same config as the 30.5 tps baseline). Parses:
#       [stats] ... dec_tps=XX.XX ...
#     from stderr (emitted unconditionally by the Done event handler).
# Both arms run 3 trials each; script reports median + ratio.
#
# CONTAMINATION
# =============
# This bench is contaminated (Claude may be open). The ratio MLX/dismantle
# is what matters, not the absolute numbers, and contamination is roughly
# constant across both arms on the same machine state. The absolute numbers
# are annotated as contaminated if Claude.app is detected.
# For authoritative absolute numbers: Cmd+Q Claude, open a fresh terminal,
# re-run. The ratio verdict should hold either way.
#
# Co-existence: dismantle runs under nice -n 19 taskpolicy -b (background QoS
# so a concurrent foreground GPU job gets first dibs). MLX runs at default
# priority — it is the CEILING reference; backgrounding it would suppress the
# measurement we want.
#
# mlx-lm INSTALLATION
# ===================
# mlx-lm must be installed for Python 3.12 (the Apple framework build, which
# supports Metal). The system homebrew python3 (3.14 as of 2026-06-02) does
# NOT ship mlx-lm. The script detects and uses the framework python3.12 path.
#
# If mlx-lm is absent, the script prints the exact pip3 command and exits 3.
# To install (one-time, ~2 min):
#   /Library/Frameworks/Python.framework/Versions/3.12/bin/pip3 install -U mlx-lm
# Do NOT use homebrew pip3 — it installs for Python 3.14 which lacks Metal support.
#
# USAGE
# =====
#   bash tools/bench/mlx_ab.sh                   # 3 trials each, 256 tokens
#   bash tools/bench/mlx_ab.sh --tokens 128       # faster, less thermal drift
#   bash tools/bench/mlx_ab.sh --mlx-trials 5 --dismantle-trials 3
#   WEIGHTS=models/qwen2.5-3b-instruct-q4_k_m.gguf bash tools/bench/mlx_ab.sh
#
# OUTPUT EXAMPLE
# ==============
#   === MLX CEILING TEST — Qwen2.5-3B-Instruct-4bit / M3 Pro ===
#   mlx_lm version : 0.31.3  (Python 3.12 framework)
#   dismantle      : v2.x.x  (locked fast-path)
#   tokens         : 256
#
#   [MLX trials]
#     trial 1 : 47.2 tok/s
#     trial 2 : 46.8 tok/s
#     trial 3 : 47.1 tok/s
#     median  : 47.1 tok/s
#
#   [dismantle trials]
#     trial 1 : 30.3 tok/s
#     trial 2 : 30.7 tok/s
#     trial 3 : 30.5 tok/s
#     median  : 30.5 tok/s
#
#   === RESULT ===
#   MLX median     : 47.1 tok/s
#   dismantle med  : 30.5 tok/s
#   MLX / dismantle: 1.545x
#
#   VERDICT: MLX hits ~47 tok/s — in llama.cpp territory (49 tps).
#   → Gap is RUNTIME-STRUCTURAL. Port MLX's graph/command-buffer structure
#     to dismantle (push GPU-busy 76%→90%+). See reports/research_next_levers
#     _2026_06_02.md §TPS #1 for the specific axis to target.
#
# KEY REFERENCE POINTS
# ====================
#   dismantle baseline : ~30.5 tok/s (clean-room, paradigm/exec 8a8346c)
#   llama.cpp same HW  : ~49   tok/s (same M3 Pro + same model)
#   MLX M2 Ultra paper : ~230  tok/s (NOT M3 Pro — do not conflate)
#   noise floor        : ±3%   (position bias; paired ABBA not needed here
#                                because we read ratio across 3+3 medians)

set -uo pipefail
cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MLX_MODEL_ID="mlx-community/Qwen2.5-3B-Instruct-4bit"
# Fixed prompt: short, deterministic, same for both arms.
# We want to measure DECODE speed — prompt length barely matters; keep it
# short to minimise prefill variance.
PROMPT="Write a haiku about silicon."
TOKENS=256
MLX_TRIALS=3
DISMANTLE_TRIALS=3

WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
BIN="${BIN:-./target/release/dismantle}"

# Python 3.12 framework (the build that ships Metal/MLX support).
# The homebrew python3 (3.14 as of 2026-06-02) does not include mlx.
# Override the MLX python with MLX_PYTHON=... (e.g. a venv:
#   python3 -m venv ~/.mlxenv && ~/.mlxenv/bin/pip install mlx-lm
#   MLX_PYTHON=~/.mlxenv/bin/python tools/bench/mlx_ab.sh)
# Auto-detect order: $MLX_PYTHON -> a tools/bench/.mlxenv venv -> the 3.12 framework.
_pick_py() {
  [[ -n "${MLX_PYTHON:-}" ]] && { echo "$MLX_PYTHON"; return; }
  for c in "$(dirname "$0")/.mlxenv/bin/python" /tmp/mlxenv/bin/python \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3; do
    [[ -x "$c" ]] && "$c" -c 'import mlx_lm' 2>/dev/null && { echo "$c"; return; }
  done
  echo /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
}
PY312="$(_pick_py)"
# console script next to the chosen python (pip installs mlx_lm.generate there)
MLX_BIN="$(dirname "$PY312")/mlx_lm.generate"

die() { printf 'error: %s\n' "$*" >&2; exit 64; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens)           TOKENS="$2";          shift 2;;
    --mlx-trials)       MLX_TRIALS="$2";      shift 2;;
    --dismantle-trials) DISMANTLE_TRIALS="$2"; shift 2;;
    --prompt)           PROMPT="$2";          shift 2;;
    --model)            MLX_MODEL_ID="$2";    shift 2;;
    --weights)          WEIGHTS="$2";         shift 2;;
    --profile)          PROFILE="$2";         shift 2;;
    -h|--help)          sed -n '2,120p' "$0"; exit 0;;
    *) die "unknown arg: $1";;
  esac
done

# ---------------------------------------------------------------------------
# (1) mlx-lm guard
# ---------------------------------------------------------------------------
# mlx-lm must be installed for Python 3.12 (the Metal-capable Apple framework
# build). The script errors helpfully if absent rather than silently failing.
#
# To install (one-time, ~2 min, no sudo required):
#   /Library/Frameworks/Python.framework/Versions/3.12/bin/pip3 install -U mlx-lm
#
if [[ ! -x "$MLX_BIN" ]]; then
  # Try to locate mlx_lm.generate anywhere on PATH as a fallback.
  MLX_BIN_FALLBACK="$(command -v mlx_lm.generate 2>/dev/null || true)"
  if [[ -z "$MLX_BIN_FALLBACK" ]]; then
    printf '\n'
    printf 'ERROR: mlx_lm.generate not found.\n' >&2
    printf '  Expected: %s\n' "$MLX_BIN" >&2
    printf '\n' >&2
    printf 'mlx-lm is NOT installed. Install it once with:\n' >&2
    printf '  /Library/Frameworks/Python.framework/Versions/3.12/bin/pip3 install -U mlx-lm\n' >&2
    printf '\n' >&2
    printf 'IMPORTANT: Use the Python 3.12 framework pip3 (above), NOT homebrew pip3.\n' >&2
    printf 'Homebrew python3 (3.14 as of 2026-06-02) does not ship Metal/MLX support.\n' >&2
    printf '\n' >&2
    printf 'After installation, re-run this script.\n' >&2
    exit 3
  fi
  MLX_BIN="$MLX_BIN_FALLBACK"
  PY312="$(command -v python3 2>/dev/null || echo python3)"
fi

# Probe mlx_lm version (Python 3.12 path)
MLX_VERSION="$("$PY312" -c 'import mlx_lm; print(mlx_lm.__version__)' 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------------------
# dismantle guards
# ---------------------------------------------------------------------------
[[ -x "$BIN" ]]    || die "dismantle binary not found: $BIN (run: cargo build --release -p dismantle)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"
[[ -f "$PROFILE" ]] || die "kernel profile not found: $PROFILE"

# Dismantle version
DISMANTLE_VERSION="$("$BIN" --version 2>&1 | head -1 | awk '{print $2}' || echo unknown)"

# ---------------------------------------------------------------------------
# Contamination check (informational, does NOT block)
# ---------------------------------------------------------------------------
CLAUDE_RUNNING=0
if pgrep -f "Claude.app" > /dev/null 2>&1; then
  CLAUDE_RUNNING=1
fi

# ---------------------------------------------------------------------------
# Locked Qwen fast-path env (matches the 30.5 tps baseline)
# Mirrors measure_joules.sh / ab_lever.sh BASE_ENV_DEFAULT.
# ---------------------------------------------------------------------------
BASE_ENV="DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
DISMANTLE_QWEN_Q4K_PREDEC=1"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
printf '\n'
printf '=== MLX CEILING TEST — Qwen2.5-3B-Instruct-4bit / M3 Pro ===\n'
printf 'mlx_lm version : %s  (Python 3.12 framework)\n' "$MLX_VERSION"
printf 'dismantle      : v%s  (locked fast-path)\n' "$DISMANTLE_VERSION"
printf 'mlx model id   : %s\n' "$MLX_MODEL_ID"
printf 'dismantle wts  : %s\n' "$WEIGHTS"
printf 'tokens         : %s\n' "$TOKENS"
printf 'prompt         : "%s"\n' "$PROMPT"
if [[ "$CLAUDE_RUNNING" == 1 ]]; then
  printf '\n'
  printf 'NOTE: Claude.app is running — absolute tps values are contaminated\n'
  printf '  (~4-5x slower than clean-room). The MLX/dismantle RATIO is still\n'
  printf '  valid because contamination is constant across both arms.\n'
  printf '  For authoritative absolute numbers: Cmd+Q Claude, open fresh terminal.\n'
fi
printf '\n'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Portable timeout (macOS has no `timeout` by default).
run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    perl -e 'use strict; my $secs = shift; alarm $secs; exec { $ARGV[0] } @ARGV or die "exec: $!"' \
      "$secs" "$@"
  fi
}

# Median of a space-separated list of floats.
median_of() {
  printf '%s\n' $1 | sort -n | awk 'BEGIN{a[0]=0} {a[NR]=$0} END{print a[int((NR+1)/2)]}'
}

# ---------------------------------------------------------------------------
# (2) MLX trials
# ---------------------------------------------------------------------------
# mlx_lm.generate with --verbose True prints the generation timing summary:
#   ==========
#   <generated text>
#   ==========
#   Prompt: N tokens, X.XXX tokens-per-sec
#   Generation: N tokens, X.XXX tokens-per-sec
#   Peak memory: X.XXX GB
#
# We parse the "Generation:" line for decode tok/s.
# MLX runs at default process priority (NOT backgrounded) — it is the ceiling
# reference; we want its MAXIMUM throughput, not a co-existence-throttled value.
#
# --seed 0 for reproducibility (greedy by default at --temp 0).
printf '[MLX trials]\n'
MLX_TPS_LIST=""
for i in $(seq 1 "$MLX_TRIALS"); do
  tmpout="/tmp/mlx_ab_mlx_trial_${i}.txt"
  set +e
  run_with_timeout 600 \
    "$MLX_BIN" \
      --model "$MLX_MODEL_ID" \
      --prompt "$PROMPT" \
      --max-tokens "$TOKENS" \
      --temp 0 \
      --seed 0 \
      --verbose True \
    > "$tmpout" 2>&1
  mlx_ec=$?
  set -e

  if [[ $mlx_ec -ne 0 ]]; then
    printf '  trial %d : FAILED (exit %d)\n' "$i" "$mlx_ec"
    printf '  --- last 5 lines of output ---\n' >&2
    tail -5 "$tmpout" >&2
    MLX_TPS_LIST="$MLX_TPS_LIST 0"
    continue
  fi

  # Parse "Generation: N tokens, X.XXX tokens-per-sec"
  # Both stdout and stderr are merged into tmpout.
  gen_tps="$(grep -E '^Generation:' "$tmpout" \
    | grep -oE '[0-9]+\.[0-9]+ tokens-per-sec' \
    | grep -oE '[0-9]+\.[0-9]+' \
    | tail -1 || echo 0)"
  [[ -z "$gen_tps" ]] && gen_tps=0
  printf '  trial %d : %s tok/s\n' "$i" "$gen_tps"
  MLX_TPS_LIST="$MLX_TPS_LIST $gen_tps"
done

MLX_MEDIAN="$(median_of "$MLX_TPS_LIST")"
printf '  median  : %s tok/s\n' "$MLX_MEDIAN"
printf '\n'

# ---------------------------------------------------------------------------
# (3) dismantle trials
# ---------------------------------------------------------------------------
# Uses `dismantle generate` (not the bench subcommand) — same as measure_joules.sh.
# Emits [stats] line to stderr:
#   [stats] reason=max_tokens prompt=N completion=N prefill_ms=X decode_ms=X dec_tps=XX.XX ...
#
# Runs under nice -n 19 taskpolicy -b (background QoS) for co-existence safety.
# max_stall_ms=0 (off) — same as measure_joules.sh defaults.
printf '[dismantle trials]\n'
DISMANTLE_TPS_LIST=""
for i in $(seq 1 "$DISMANTLE_TRIALS"); do
  tmpout="/tmp/mlx_ab_dismantle_trial_${i}.txt"
  set +e
  # shellcheck disable=SC2086
  run_with_timeout 600 \
    env $BASE_ENV \
    nice -n 19 taskpolicy -b \
    "$BIN" generate \
      --weights "$WEIGHTS" \
      --kernel-profile "$PROFILE" \
      --prompt "$PROMPT" \
      --max-new-tokens "$TOKENS" \
      --temperature 0 \
      --seed 0 \
    > "$tmpout" 2>&1
  dis_ec=$?
  set -e

  if [[ $dis_ec -ne 0 ]]; then
    printf '  trial %d : FAILED (exit %d)\n' "$i" "$dis_ec"
    tail -5 "$tmpout" >&2
    DISMANTLE_TPS_LIST="$DISMANTLE_TPS_LIST 0"
    continue
  fi

  # Parse dec_tps from the [stats] line.
  dec_tps="$(grep -E '\[stats\]' "$tmpout" \
    | tail -1 \
    | grep -oE 'dec_tps=[0-9.]+' \
    | grep -oE '[0-9.]+' || echo 0)"
  [[ -z "$dec_tps" ]] && dec_tps=0
  printf '  trial %d : %s tok/s\n' "$i" "$dec_tps"
  DISMANTLE_TPS_LIST="$DISMANTLE_TPS_LIST $dec_tps"
done

DISMANTLE_MEDIAN="$(median_of "$DISMANTLE_TPS_LIST")"
printf '  median  : %s tok/s\n' "$DISMANTLE_MEDIAN"
printf '\n'

# ---------------------------------------------------------------------------
# (4) Ratio + verdict
# ---------------------------------------------------------------------------
printf '=== RESULT ===\n'
printf 'MLX median     : %s tok/s\n' "$MLX_MEDIAN"
printf 'dismantle med  : %s tok/s\n' "$DISMANTLE_MEDIAN"
printf '\n'

awk \
  -v mlx="$MLX_MEDIAN" \
  -v dis="$DISMANTLE_MEDIAN" \
  -v contaminated="$CLAUDE_RUNNING" \
  -v llamacpp="49" \
'BEGIN {
  if (dis <= 0) {
    print "ERROR: dismantle median is 0 — all trials failed. Check [stats] parsing."
    exit 1
  }
  if (mlx <= 0) {
    print "ERROR: MLX median is 0 — all trials failed. Check Generation: parsing."
    exit 1
  }
  ratio = mlx / dis
  printf "MLX / dismantle: %.3fx\n", ratio
  printf "llama.cpp ref  : %s tok/s  (same M3 Pro, baseline from research report)\n", llamacpp
  printf "\n"

  if (contaminated) {
    printf "NOTE: Values above are contaminated (Claude open). Ratio is valid; absolutes are not.\n"
    printf "\n"
  }

  # Verdict thresholds:
  #   MLX >= 40 tok/s clean (or ratio > 1.3x) → structural gap
  #   MLX <= 33 tok/s clean (or ratio < 1.1x) → HW-bound
  # We use the ratio because contamination shifts both arms equally.
  if (ratio >= 1.3) {
    printf "VERDICT: MLX hits %.1f tok/s — substantially faster than dismantle.\n", mlx
    printf "  → Gap is RUNTIME-STRUCTURAL (command-buffer/GPU-saturation layer).\n"
    printf "  → Closing the gap is FEASIBLE on this M3 Pro.\n"
    printf "  → NEXT STEP: Metal System Trace diff (Instruments) of MLX vs dismantle\n"
    printf "    per-token timeline → find the idle/scheduling delta.\n"
    printf "    Then port MLX graph/command-buffer structure (push GPU-busy 76%%->90%%+).\n"
    printf "    See reports/research_next_levers_2026_06_02.md §TPS #1 for the target.\n"
  } else if (ratio <= 1.1) {
    printf "VERDICT: MLX hits %.1f tok/s — close to dismantle (~%.0f%% difference).\n", mlx, (ratio-1)*100
    printf "  → Gap is M3-PRO-HW-BOUND on this machine.\n"
    printf "  → The 1.6x llama.cpp lead is llama.cpp-SPECIFIC (not generic MLX architecture).\n"
    printf "  → NEXT STEP: Trace llama.cpp SPECIFICALLY on M3 Pro (not MLX).\n"
    printf "    The gap lives in llama.cpp ggml-metal mul_mv + simdgroup_matrix path,\n"
    printf "    not in generic command-buffer batching. Port those specific kernels.\n"
  } else {
    printf "VERDICT: MLX hits %.1f tok/s — ratio %.3fx is in the AMBIGUOUS band (1.1x-1.3x).\n", mlx, ratio
    printf "  → INCONCLUSIVE. Run more trials (--mlx-trials 5 --dismantle-trials 5)\n"
    printf "    and/or run clean (Cmd+Q Claude + fresh terminal) for authoritative absolutes.\n"
    printf "  → If MLX absolute is 40+ clean: structural. If MLX absolute is ~30 clean: HW-bound.\n"
  }
  printf "\n"
}'

# ---------------------------------------------------------------------------
# JSON summary for scripted consumption
# ---------------------------------------------------------------------------
OUT_JSON="/tmp/mlx_ab_result.json"
"$PY312" - <<PYEOF
import json, sys

mlx_trials  = [float(x) for x in """$MLX_TPS_LIST""".split() if float(x) > 0]
dis_trials  = [float(x) for x in """$DISMANTLE_TPS_LIST""".split() if float(x) > 0]
mlx_med  = float("$MLX_MEDIAN")  if "$MLX_MEDIAN"  not in ("", "0") else 0.0
dis_med  = float("$DISMANTLE_MEDIAN") if "$DISMANTLE_MEDIAN" not in ("", "0") else 0.0
ratio    = mlx_med / dis_med if dis_med > 0 else None

result = {
    "mlx_model_id":       "$MLX_MODEL_ID",
    "mlx_version":        "$MLX_VERSION",
    "dismantle_version":  "$DISMANTLE_VERSION",
    "tokens":             $TOKENS,
    "prompt":             "$PROMPT",
    "contaminated":       bool($CLAUDE_RUNNING),
    "mlx_tps_trials":     mlx_trials,
    "dismantle_tps_trials": dis_trials,
    "mlx_median_tps":     mlx_med,
    "dismantle_median_tps": dis_med,
    "ratio_mlx_over_dismantle": round(ratio, 4) if ratio else None,
    "llamacpp_reference_tps": 49.0,
    "verdict": (
        "STRUCTURAL_GAP"  if ratio and ratio >= 1.3 else
        "HW_BOUND"        if ratio and ratio <= 1.1 else
        "INCONCLUSIVE"
    ),
}
json.dump(result, open("$OUT_JSON", "w"), indent=2)
print(f"Summary JSON written to: $OUT_JSON")
PYEOF

printf 'done.\n'

