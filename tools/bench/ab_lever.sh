#!/usr/bin/env bash
# tools/bench/ab_lever.sh — balanced ABBA paired A/B for any DISMANTLE_QWEN_*
# flag (or --profile fast) at configurable context length.
#
# DESIGN RATIONALE
# ================
# Context: the user cannot run a clean-room absolute bench with Claude open.
# Contamination factor (~4-5x) is CONSTANT across consecutive trials on the
# same machine state, so it CANCELS in the B/A ratio — paired A/B is valid.
#
# Second-position bias: a ~3% bias toward the second arm of any A/B was found
# during the phase-0 noise-floor analysis (reports/bench/phase0_noise_floor.json
# shows B/A = 0.970 for an identical A=B pair — meaning the second arm ran ~3%
# slower, i.e. position-2 is SLOWER, not faster). A balanced ABBA interleave
# cancels this: each 4-trial block runs A B B A, so position bias cancels across
# the block in both the A and B pools.
#
# Long-ctx: f16-KV (DISMANTLE_QWEN_F16_KV) and flash-attn
# (DISMANTLE_QWEN_FLASH_ATTN) pay at long context where KV-cache read traffic
# rivals weight read traffic. Use --long-ctx to synthesize a ~2K-token prompt
# (≈8192 chars of a Rust fn snippet) and pass it via DISMANTLE_BENCH_PROMPT_FILE,
# which the decode suite already honors (crates/dismantle-bench/src/suites/decode.rs:17).
# At short context (~16 prompt tokens), both levers contribute <3% — within the
# noise floor — so --long-ctx is mandatory to see a signal.
#
# MUTUAL EXCLUSION: DISMANTLE_QWEN_F16_KV and DISMANTLE_QWEN_FLASH_ATTN are
# mutually exclusive (crates/dismantle-core/src/model/qwen_dense.rs:3545 —
# the binary returns an error if both are set). Test them separately.
#
# --profile fast: this is a CLI flag (--profile fast), NOT an env var.
# It sets DISMANTLE_QWEN_VOCAB_PRUNE, Q4K_LMHEAD, FFN_DOWN_Q4K, Q4K_PREDEC,
# and PREDEC_F16SCALES internally (crates/dismantle/src/main.rs:35-51).
# Explicitly-set env vars take precedence over --profile. Use --cli-b for the B arm.
#
# USAGE
# =====
#   tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_F16_KV --long-ctx
#   tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_FLASH_ATTN --long-ctx
#   tools/bench/ab_lever.sh --cli-b "--profile fast"
#   tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_F16_KV --long-ctx \
#       --blocks 3 --tokens 32
#
# OPTIONS
#   --lever VAR       DISMANTLE_QWEN_* env var name to toggle (A=0, B=1).
#                     Cannot be combined with --cli-b.
#   --cli-b FLAGS     CLI flags to append to the B arm invocation (e.g.
#                     "--profile fast"). A arm receives no extra CLI flags.
#                     Cannot be combined with --lever.
#   --long-ctx        Synthesize a ~2048-token prompt (~8192 chars). Mandatory
#                     for f16-KV and flash-attn levers; optional otherwise.
#   --ctx-tokens N    Approx prompt token count for --long-ctx (default 2048).
#   --tokens N        Decode token count per trial (default 32).
#   --blocks N        Number of ABBA blocks (4 trials each); total trials = 4N
#                     (default 2 = 8 trials total, matching paradigm guidance).
#   --base-env ENV    Space-separated base env for both arms (default: locked
#                     Qwen fast-path minus the lever under test).
#   --weights PATH    (default models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   --profile PATH    Kernel profile JSON (default profiles/qwen3b-instruct-q4k.m3pro18.json)
#   --out FILE        JSON summary output path (default reports/bench/ab_<label>.json)
#   -h|--help         Print this header and exit.
#
# EXAMPLES (the exact run_commands requested in Wave-5b)
# ======================================================
# f16-KV at long ctx (energy/long-ctx lever, paradigm plan 2.1-a):
#   bash tools/bench/ab_lever.sh \
#       --lever DISMANTLE_QWEN_F16_KV \
#       --long-ctx
#
# flash-attn at long ctx (paradigm plan 2.3):
#   bash tools/bench/ab_lever.sh \
#       --lever DISMANTLE_QWEN_FLASH_ATTN \
#       --long-ctx
#
# --profile fast at short ctx (+7.4% prior, f16-scales bundle, Phase 1.2):
#   bash tools/bench/ab_lever.sh \
#       --cli-b "--profile fast"
#
# INTERPRETATION
# ==============
#   B/A > 1.03  -> clear gain (outside ~3% noise floor)
#   B/A < 0.97  -> clear regression
#   0.97..1.03  -> INCONCLUSIVE (within the measured ~3% noise floor)
#   Range = (max_B / max_A) .. (min_B / min_A) — spread of the ratio;
#            a tight range (< 0.03 width) confirms the contamination cancels.
#
# Co-existence: every dismantle subprocess runs under nice -n 19 taskpolicy -b
# so a concurrent foreground GPU job keeps first dibs (same as paired_lever.sh).

set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/dismantle}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
KERNEL_PROFILE="${KERNEL_PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"

# Locked Qwen fast-path (constant across A/B). Match paired_lever.sh default.
BASE_ENV_DEFAULT="DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
DISMANTLE_QWEN_Q4K_PREDEC=1"

LEVER=""          # DISMANTLE_QWEN_* env var name
CLI_B=""          # Extra CLI flags for B arm (e.g. "--profile fast")
LONG_CTX=0
CTX_TOKENS=2048   # Approximate prompt token count for --long-ctx
TOKENS=32         # Decode tokens per trial
BLOCKS=2          # Number of ABBA blocks (4 trials each)
BASE_ENV="$BASE_ENV_DEFAULT"
OUT=""

die() { printf 'error: %s\n' "$*" >&2; exit 64; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lever)       LEVER="$2"; shift 2;;
    --cli-b)       CLI_B="$2"; shift 2;;
    --long-ctx)    LONG_CTX=1; shift;;
    --ctx-tokens)  CTX_TOKENS="$2"; shift 2;;
    --tokens)      TOKENS="$2"; shift 2;;
    --blocks)      BLOCKS="$2"; shift 2;;
    --base-env)    BASE_ENV="$2"; shift 2;;
    --weights)     WEIGHTS="$2"; shift 2;;
    --profile)     KERNEL_PROFILE="$2"; shift 2;;
    --out)         OUT="$2"; shift 2;;
    -h|--help)     sed -n '2,100p' "$0"; exit 0;;
    *) die "unknown arg: $1";;
  esac
done

# Validate: exactly one of --lever / --cli-b
[[ -n "$LEVER" && -n "$CLI_B" ]] && die "--lever and --cli-b are mutually exclusive"
[[ -z "$LEVER" && -z "$CLI_B" ]] && die "one of --lever or --cli-b is required"

# Warn about F16_KV + FLASH_ATTN incompatibility if both appear in env or lever
if [[ "$LEVER" == "DISMANTLE_QWEN_F16_KV" ]] && [[ -n "${DISMANTLE_QWEN_FLASH_ATTN:-}" ]]; then
  die "DISMANTLE_QWEN_F16_KV and DISMANTLE_QWEN_FLASH_ATTN are mutually exclusive (binary will error)"
fi
if [[ "$LEVER" == "DISMANTLE_QWEN_FLASH_ATTN" ]] && [[ -n "${DISMANTLE_QWEN_F16_KV:-}" ]]; then
  die "DISMANTLE_QWEN_F16_KV and DISMANTLE_QWEN_FLASH_ATTN are mutually exclusive (binary will error)"
fi

[[ -x "$BIN" ]] || die "binary not found/executable: $BIN (cargo build --release?)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"

# Build label
if [[ -n "$LEVER" ]]; then
  LABEL="${LEVER//DISMANTLE_QWEN_/}"
  LABEL="${LABEL,,}"  # lowercase
else
  # Sanitize --cli-b into a label (e.g. "--profile fast" -> "profile_fast")
  LABEL=$(printf '%s' "$CLI_B" | tr -cs 'a-zA-Z0-9' '_' | sed 's/^_//;s/_$//')
fi
[[ "$LONG_CTX" == 1 ]] && LABEL="${LABEL}_longctx"

[[ -z "$OUT" ]] && OUT="reports/bench/ab_${LABEL}.json"
mkdir -p "$(dirname "$OUT")"

TOTAL_TRIALS=$(( BLOCKS * 4 ))

printf '=== ab_lever.sh: ABBA paired A/B ===\n'
printf '  lever/flag : %s\n' "${LEVER:-$CLI_B}"
printf '  long-ctx   : %s  (ctx_tokens=%s)\n' "$LONG_CTX" "$CTX_TOKENS"
printf '  decode tok : %s  blocks=%s  total=%s trials\n' "$TOKENS" "$BLOCKS" "$TOTAL_TRIALS"
printf '  weights    : %s\n' "$WEIGHTS"
printf '  kernel-prof: %s\n' "$KERNEL_PROFILE"
printf '  base-env   : %s\n' "$BASE_ENV"
printf '  output     : %s\n' "$OUT"
printf '\n'
printf '  ABBA interleave cancels ~3%% second-position bias found in phase-0 noise floor.\n'
printf '  Within 3%% of B/A = 1.00 is INCONCLUSIVE (within noise floor).\n'
printf '\n'

# ── Long-ctx prompt synthesis ────────────────────────────────────────────────
# ~4 chars/token (same heuristic as long_context_bench.sh).
# Uses the same Rust fn snippet unit to be consistent with existing harnesses.
PROMPT_FILE=""
if [[ "$LONG_CTX" == 1 ]]; then
  PROMPT_FILE="/tmp/ab_lever_longctx_${CTX_TOKENS}.txt"
  target_chars=$(( CTX_TOKENS * 4 ))
  unit='fn step(s: &mut State, i: usize) -> u64 { s.acc += i as u64; s.acc } '
  acc=""
  while [[ ${#acc} -lt $target_chars ]]; do acc+="$unit"; done
  printf '%s' "${acc:0:$target_chars}" > "$PROMPT_FILE"
  printf '  long-ctx prompt: %s chars -> %s (approx %s tokens)\n' \
    "$target_chars" "$PROMPT_FILE" "$CTX_TOKENS"
  printf '\n'
fi

# ── Trial runner ─────────────────────────────────────────────────────────────
# run_arm ARM TRIAL_INDEX -> emits decode_tps to stdout
# ARM is "A" or "B"
run_arm() {
  local arm="$1" t="$2"
  local j="/tmp/ab_lever_${LABEL}_${arm}_${t}.json"
  local extra_env=""
  local extra_cli=""

  if [[ -n "$LEVER" ]]; then
    # Env-var lever: A=0 (disabled), B=1 (enabled)
    if [[ "$arm" == "B" ]]; then
      extra_env="${LEVER}=1"
    else
      extra_env="${LEVER}=0"
    fi
  else
    # CLI-flag lever: B gets the extra flags
    [[ "$arm" == "B" ]] && extra_cli="$CLI_B"
  fi

  local bench_env="$BASE_ENV${extra_env:+ }$extra_env"

  if [[ -n "$PROMPT_FILE" ]]; then
    # Pass the long prompt via the env var that the decode suite reads.
    bench_env="$bench_env DISMANTLE_BENCH_PROMPT_FILE=$PROMPT_FILE"
  fi

  # shellcheck disable=SC2086
  env $bench_env nice -n 19 taskpolicy -b \
    "$BIN" $extra_cli bench \
      --backend dismantle --suite decode \
      --weights "$WEIGHTS" \
      --trials 1 \
      --max-new-tokens "$TOKENS" \
      --kernel-profile "$KERNEL_PROFILE" \
      --json "$j" \
    >/dev/null 2>&1

  # Parse decode_tps: prefer flat .results.decode_tps (always present), fall
  # back to nested .results.trial_stats[0].decode_tps for compat.
  jq -r '(.results.decode_tps // .results.trial_stats[0].decode_tps // 0)' \
    "$j" 2>/dev/null || printf '0'
}

# ── ABBA interleave ───────────────────────────────────────────────────────────
# Each block = A B B A. This places each arm in position-1 and position-2
# equally, so the ~3% second-position bias cancels across the block.
# Reference: phase0_noise_floor.json shows B/A = 0.970 for identical A=B runs,
# confirming position-2 is ~3% slower. ABBA distributes both arms across
# both positions equally.
TPS_A=""
TPS_B=""
trial_global=0

for block in $(seq 1 "$BLOCKS"); do
  printf '  block %d/%d  (ABBA)\n' "$block" "$BLOCKS"
  for arm in A B B A; do
    trial_global=$(( trial_global + 1 ))
    tps=$(run_arm "$arm" "$trial_global")
    printf '    trial %2d  arm=%s  decode_tps=%s\n' "$trial_global" "$arm" "$tps"
    if [[ "$arm" == "A" ]]; then
      TPS_A="$TPS_A $tps"
    else
      TPS_B="$TPS_B $tps"
    fi
  done
done

# ── Statistics ───────────────────────────────────────────────────────────────
med() {
  # Median of a space-separated list; matches paired_lever.sh med() function.
  printf '%s\n' $1 | sort -n | awk '{a[NR]=$0} END{print a[int((NR+1)/2)]}'
}

MED_A=$(med "$TPS_A")
MED_B=$(med "$TPS_B")

MIN_A=$(printf '%s\n' $TPS_A | sort -n | head -1)
MAX_A=$(printf '%s\n' $TPS_A | sort -n | tail -1)
MIN_B=$(printf '%s\n' $TPS_B | sort -n | head -1)
MAX_B=$(printf '%s\n' $TPS_B | sort -n | tail -1)

printf '\n'
printf '=== RESULTS ===\n'
printf '  A (baseline) trials: %s\n' "$TPS_A"
printf '  B (lever ON) trials: %s\n' "$TPS_B"
printf '  A median=%.3f  (min=%.3f max=%.3f)\n' "$MED_A" "$MIN_A" "$MAX_A"
printf '  B median=%.3f  (min=%.3f max=%.3f)\n' "$MED_B" "$MIN_B" "$MAX_B"
printf '\n'

awk -v ma="$MED_A" -v mb="$MED_B" \
    -v mina="$MIN_A" -v maxa="$MAX_A" \
    -v minb="$MIN_B" -v maxb="$MAX_B" 'BEGIN {
  if (ma <= 0) { print "  B/A: UNDEFINED (A median = 0)"; exit }
  ratio = mb / ma
  pct   = (ratio - 1.0) * 100.0
  # Ratio range: best-case B / worst-case A to worst-case B / best-case A
  lo    = (mina > 0) ? minb / maxa : 0
  hi    = (mina > 0) ? maxb / mina : 0
  printf "  B/A median ratio : %.4f  (%+.1f%%)\n", ratio, pct
  printf "  B/A range        : [%.4f .. %.4f]\n", lo, hi
  printf "\n"
  if (ratio > 1.03)
    printf "  VERDICT: GAIN (B/A = %.3fx > 1.03 noise-floor threshold)\n", ratio
  else if (ratio < 0.97)
    printf "  VERDICT: REGRESSION (B/A = %.3fx < 0.97)\n", ratio
  else
    printf "  VERDICT: INCONCLUSIVE — B/A = %.3fx is within the ±3%% noise floor.\n"\
           "           Run more blocks (--blocks 3+) or use a longer --tokens budget.\n", ratio
}'

# ── Machine-readable JSON summary ────────────────────────────────────────────
PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"
"$PY" - "$OUT" <<PYEOF
import json, sys
out = sys.argv[1]

def med(s):
    xs = [float(x) for x in s.split() if float(x) > 0]
    if not xs: return 0.0
    xs.sort()
    return xs[len(xs) // 2]

def ratio_range(tps_a_str, tps_b_str):
    aa = sorted([float(x) for x in tps_a_str.split() if float(x) > 0])
    bb = sorted([float(x) for x in tps_b_str.split() if float(x) > 0])
    if not aa or not bb: return None, None
    lo = bb[0] / aa[-1]  # worst B / best A
    hi = bb[-1] / aa[0]  # best B / worst A
    return lo, hi

tps_a = """$TPS_A"""
tps_b = """$TPS_B"""
ma = med(tps_a)
mb = med(tps_b)
ratio = (mb / ma) if ma > 0 else None
pct   = ((ratio - 1.0) * 100.0) if ratio else None
lo, hi = ratio_range(tps_a, tps_b)

doc = {
    "label": "$LABEL",
    "lever": "$LEVER",
    "cli_b": "$CLI_B",
    "long_ctx": bool($LONG_CTX),
    "ctx_tokens": $CTX_TOKENS,
    "tokens": $TOKENS,
    "blocks": $BLOCKS,
    "total_trials": $TOTAL_TRIALS,
    "interleave": "ABBA",
    "base_env": """$BASE_ENV""".split(),
    "tps_a_trials": [float(x) for x in tps_a.split() if x],
    "tps_b_trials": [float(x) for x in tps_b.split() if x],
    "median_tps_a": ma,
    "median_tps_b": mb,
    "ratio_b_over_a": ratio,
    "pct_delta": round(pct, 2) if pct else None,
    "ratio_range_lo": round(lo, 4) if lo else None,
    "ratio_range_hi": round(hi, 4) if hi else None,
    "verdict": (
        "GAIN" if ratio and ratio > 1.03
        else "REGRESSION" if ratio and ratio < 0.97
        else "INCONCLUSIVE"
    ),
}
json.dump(doc, open(out, "w"), indent=2)
print(f"wrote {out}")
print(f"  {doc['verdict']}  B/A={doc['ratio_b_over_a']!r}  ({doc['pct_delta']}%)")
PYEOF
# EXACT RUN_COMMAND EXAMPLES REQUESTED IN WAVE-5b
# ================================================

# (1) f16-KV at long context (DISMANTLE_QWEN_F16_KV, energy/long-ctx lever, plan 2.1-a):

#     bash tools/bench/ab_lever.sh \
#         --lever DISMANTLE_QWEN_F16_KV \
#         --long-ctx

#   This synthesizes a ~2048-token (~8192-char) Rust fn prompt, passes it via
#   DISMANTLE_BENCH_PROMPT_FILE, and runs 2 ABBA blocks (8 trials: A B B A A B B A).
#   A arm: DISMANTLE_QWEN_F16_KV=0 (disabled)
#   B arm: DISMANTLE_QWEN_F16_KV=1 (enabled)
#   NOTE: cannot combine with DISMANTLE_QWEN_FLASH_ATTN in the environment
#   (binary will error at startup: mutual exclusion check in qwen_dense.rs:3545).

# (2) flash-attn at long context (DISMANTLE_QWEN_FLASH_ATTN, plan 2.3):

#     bash tools/bench/ab_lever.sh \
#         --lever DISMANTLE_QWEN_FLASH_ATTN \
#         --long-ctx

#   Same prompt synthesis as above. Flash-attn also only wins at long context:
#   the gap-anatomy profile (paradigm_execution_log.md line 446) shows attention
#   is 2.92% at short context — within the noise floor. The ~2K prompt puts
#   KV-cache traffic into a regime where the lever can show signal.

# (3) --profile fast at short context (+7.4% prior, f16-scales bundle, Phase 1.2):

#     bash tools/bench/ab_lever.sh \
#         --cli-b "--profile fast"

#   A arm: base-env only (bit-identical default decode)
#   B arm: base-env + --profile fast CLI flag appended before 'bench' subcommand
#   --profile fast sets: VOCAB_PRUNE=32000, Q4K_LMHEAD=1, FFN_DOWN_Q4K=1,
#     Q4K_PREDEC=1, PREDEC_F16SCALES=1 — but only if those vars are not already
#     set in the environment (explicit env always wins).
#   NOTE: the base-env already sets VOCAB_PRUNE, Q4K_LMHEAD, FFN_DOWN_Q4K, and
#     Q4K_PREDEC — so the ONLY marginal delta from --profile fast vs the base-env
#     is PREDEC_F16SCALES=1. This is the right comparison: it isolates f16-scales
#     on top of the already-locked fast-path, matching the +7.4% prior pairing.
#   Short context is correct here because f16-scales is a predec path optimization
#     that pays at all context lengths (it cuts scale-read traffic in the GEMV
#     predec path, not the attention path).

# ADDITIONAL INVOCATIONS
# ======================

#   More blocks for tighter statistics:
#     bash tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_F16_KV --long-ctx --blocks 3
    # 12 trials total, 6 per arm

#   Longer decode for larger per-trial signal:
#     bash tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_F16_KV --long-ctx --tokens 64

#   Override ctx (e.g. 4K tokens where KV share is ~25%):
#     bash tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_F16_KV \
#         --long-ctx --ctx-tokens 4096 --tokens 32

# NOISE FLOOR / INTERPRETATION GUIDE
# ===================================
#   The phase-0 noise-floor run (reports/bench/phase0_noise_floor.json) used
#   A=B (identical env) and found B/A = 0.970 (-3.0%). This means:
#   - Second position is ~3% slower on this machine under Claude-open conditions.
#   - ABBA interleave distributes each arm across both positions equally,
#     so the position bias cancels in the pooled medians.
#   - Any B/A outside [0.97, 1.03] after ABBA interleave is a real signal.
#   - B/A inside [0.97, 1.03] is INCONCLUSIVE regardless of block count.
#   The f16-scales prior (a6_5_pair_f16s_confirm.json) shows B/A = 1.061 (+6.1%),
#   which is well above the noise floor — that lever is detectable.
#   f16-KV and flash-attn have <3% short-context ceiling (attention is 2.92%
#   of total decode time at short ctx). At long ctx (2K+) the ceiling rises
#   significantly as KV-read traffic grows linearly with sequence length.

# SCHEMA NOTE — WHY FLAT .results.decode_tps
# ==========================================
#   The bench suite finalize() function (crates/dismantle-bench/src/suites/decode.rs:62)
#   emits .results.decode_tps as the first field, always equal to the median tps.
#   .results.trial_stats[0].decode_tps is also present in the current schema
#   (each trial object has decode_tps) but is fragile if trial_stats is empty.
#   ab_lever.sh uses the two-path fallback:
#     jq -r '(.results.decode_tps // .results.trial_stats[0].decode_tps // 0)'
#   matching clean_bench.sh:229's established pattern.
