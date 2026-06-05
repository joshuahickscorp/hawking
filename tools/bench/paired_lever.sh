#!/usr/bin/env bash
# §1-gated paired lever bench — the reusable harness every Tier-1 lever runs.
#
# Generalizes path_to_50_verify.sh: arbitrary A/B (+ optional C) env deltas on
# top of the locked Qwen fast-path, run as
#   1. PARITY   — greedy temp=0 bit-identical token check (B,C vs A).
#   2. BENCH    — interleaved paired decode_tps trials (A,B,A,B,... cancels
#                 thermal drift; Claude-open is fine for paired deltas, see
#                 memory/feedback_bench_with_claude_open.md). Prints medians +
#                 B/A ratio.
#   3. GATE     — a SEPARATE instrumented run (DISMANTLE_TCB_TRACE=gpu) of the
#                 B variant fed to analyze_tcb_trace.py. The §1 methodology gate
#                 refuses a physics-violating measurement (exit 2). NB: the gpu
#                 trace is split-CB-distorted, so its decode_tps is NOT the ship
#                 number — only its gate verdict + busy-fraction note are used.
#                 The clean BENCH median above is the ship/hold number.
#
# Bit-identical levers gate on PARITY here. Quality-trade (atol 1e-3) levers run
# with --no-parity and gate via their Rust parity test (cargo test ... parity).
#
# Usage:
#   tools/bench/paired_lever.sh --label NAME --env-a "K=0" --env-b "K=1" \
#       [--env-c "K=1 J=1"] [--mode all|parity|bench|gate] \
#       [--tokens 32] [--trials 5] [--prompt '...'] [--no-parity] \
#       [--base-env "..."] [--out reports/bench/NAME.json]
#
# Co-existence: every dismantle subprocess runs under `nice -n 19 taskpolicy -b`
# so a concurrent foreground GPU job (EAGLE capture, slm) keeps first dibs.
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/dismantle}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"

# Locked Qwen fast-path (constant across A/B/C). Override with --base-env.
BASE_ENV_DEFAULT="DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
DISMANTLE_QWEN_Q4K_PREDEC=1"

LABEL="lever"
ENV_A=""; ENV_B=""; ENV_C=""
MODE="all"; TOKENS=32; TRIALS=5; DO_PARITY=1
PROMPT='fn fibonacci(n: u64) -> u64 {'
BASE_ENV="$BASE_ENV_DEFAULT"
OUT=""

die() { echo "error: $*" >&2; exit 64; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)     LABEL="$2"; shift 2;;
    --env-a)     ENV_A="$2"; shift 2;;
    --env-b)     ENV_B="$2"; shift 2;;
    --env-c)     ENV_C="$2"; shift 2;;
    --mode)      MODE="$2"; shift 2;;
    --tokens)    TOKENS="$2"; shift 2;;
    --trials)    TRIALS="$2"; shift 2;;
    --prompt)    PROMPT="$2"; shift 2;;
    --base-env)  BASE_ENV="$2"; shift 2;;
    --no-parity) DO_PARITY=0; shift;;
    --out)       OUT="$2"; shift 2;;
    -h|--help)   sed -n '2,40p' "$0"; exit 0;;
    *) die "unknown arg: $1";;
  esac
done
[[ -n "$ENV_A" ]] || die "--env-a required (baseline env delta)"
[[ -n "$ENV_B" ]] || die "--env-b required (lever env delta)"
[[ -x "$BIN" ]] || die "binary not found/executable: $BIN (cargo build --release?)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"
[[ -z "$OUT" ]] && OUT="reports/bench/${LABEL}.json"
mkdir -p "$(dirname "$OUT")"

# variants present (A,B always; C optional)
VARIANTS=(A B); [[ -n "$ENV_C" ]] && VARIANTS+=(C)
env_for() { case "$1" in A) echo "$ENV_A";; B) echo "$ENV_B";; C) echo "$ENV_C";; esac; }
run() { env $BASE_ENV $(env_for "$1") nice -n 19 taskpolicy -b "$BIN" "${@:2}"; }
med() { printf '%s\n' $1 | sort -n | awk '{a[NR]=$0} END{print a[int((NR+1)/2)]}'; }

PARITY_RESULT="skipped"
if [[ "$DO_PARITY" == 1 && ( "$MODE" == parity || "$MODE" == all ) ]]; then
  echo "=== PARITY (greedy ${TOKENS}tok, bit-identical) ==="
  for c in "${VARIANTS[@]}"; do
    run "$c" generate --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
      --prompt "$PROMPT" --max-new-tokens "$TOKENS" --temperature 0 --seed 0 \
      > "/tmp/pl_${LABEL}_$c.txt" 2>/dev/null
  done
  PARITY_RESULT="pass"
  for c in "${VARIANTS[@]:1}"; do
    if diff -q "/tmp/pl_${LABEL}_A.txt" "/tmp/pl_${LABEL}_$c.txt" >/dev/null; then
      echo "  $c vs A: IDENTICAL ✓"
    else echo "  $c vs A: DIFF ✗"; PARITY_RESULT="fail"; fi
  done
  echo "  PARITY: ${PARITY_RESULT^^}"
fi

MED_A=0; MED_B=0; MED_C=0; TPS_A=""; TPS_B=""; TPS_C=""
if [[ "$MODE" == bench || "$MODE" == all ]]; then
  echo "=== PAIRED BENCH (${TRIALS} trials × ${TOKENS}tok, interleaved, clean) ==="
  for t in $(seq 1 "$TRIALS"); do
    for c in "${VARIANTS[@]}"; do
      j="/tmp/pl_${LABEL}_bench_${c}_${t}.json"
      run "$c" bench --backend dismantle --suite decode --weights "$WEIGHTS" \
        --trials 1 --max-new-tokens "$TOKENS" --kernel-profile "$PROFILE" \
        --json "$j" >/dev/null 2>&1
      v=$(jq -r '(.results.decode_tps // .results.trial_stats[0].decode_tps // 0)' "$j" 2>/dev/null)
      eval "TPS_$c=\"\$TPS_$c $v\""
    done
  done
  MED_A=$(med "$TPS_A"); MED_B=$(med "$TPS_B"); [[ -n "$ENV_C" ]] && MED_C=$(med "$TPS_C")
  printf "  A (base)  trials:%s  median=%.2f\n" "$TPS_A" "$MED_A"
  printf "  B (lever) trials:%s  median=%.2f\n" "$TPS_B" "$MED_B"
  [[ -n "$ENV_C" ]] && printf "  C         trials:%s  median=%.2f\n" "$TPS_C" "$MED_C"
  awk -v a="$MED_A" -v b="$MED_B" 'BEGIN{ if(a>0) printf "  B/A = %.3fx  (%+.1f%%)\n", b/a, (b/a-1)*100; }'
  [[ -n "$ENV_C" ]] && awk -v a="$MED_A" -v c="$MED_C" 'BEGIN{ if(a>0) printf "  C/A = %.3fx  (%+.1f%%)\n", c/a, (c/a-1)*100; }'
fi

GATE_PASSED="skipped"; GATE_TRACE=""
if [[ "$MODE" == gate || "$MODE" == all ]]; then
  echo "=== §1 METHODOLOGY GATE (instrumented B run; NOT the ship tps) ==="
  GATE_TRACE="/tmp/pl_${LABEL}_gate.json"
  env $BASE_ENV $(env_for B) DISMANTLE_TCB_TRACE=gpu DISMANTLE_TRACE_DISPATCH=1 \
    nice -n 19 taskpolicy -b "$BIN" bench --trace-dispatch --backend dismantle \
    --suite decode --weights "$WEIGHTS" --trials 1 --max-new-tokens "$TOKENS" \
    --kernel-profile "$PROFILE" --json "$GATE_TRACE" >/dev/null 2>&1
  PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"
  if "$PY" tools/bench/analyze_tcb_trace.py "$GATE_TRACE" --model qwen3b --json > "/tmp/pl_${LABEL}_gate_analysis.json" 2>/dev/null; then
    GATE_PASSED="pass"
  else
    GATE_PASSED="fail"
  fi
  "$PY" tools/bench/analyze_tcb_trace.py "$GATE_TRACE" --model qwen3b --no-gate 2>/dev/null \
    | sed -n '/METHODOLOGY GATE/,$p'
fi

# machine-readable summary
PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"
"$PY" - "$OUT" <<PYEOF
import json, sys
out = sys.argv[1]
def med(s):
    xs=[float(x) for x in s.split()] or [0.0]
    xs.sort(); return xs[len(xs)//2]
mA, mB = $MED_A, $MED_B
doc = {
  "label": "$LABEL", "tokens": $TOKENS, "trials": $TRIALS,
  "base_env": """$BASE_ENV""".split(),
  "env_a": "$ENV_A", "env_b": "$ENV_B", "env_c": "$ENV_C" or None,
  "parity": "$PARITY_RESULT",
  "median_tps_a": mA, "median_tps_b": mB,
  "ratio_b_over_a": (mB/mA) if mA else None,
  "pct_delta": ((mB/mA-1)*100) if mA else None,
  "gate": "$GATE_PASSED", "gate_trace": "$GATE_TRACE" or None,
}
json.dump(doc, open(out, "w"), indent=2)
print(f"\nwrote {out}")
print(f"  parity={doc['parity']} gate={doc['gate']} "
      f"B/A={doc['ratio_b_over_a']!r} ({doc['pct_delta'] and round(doc['pct_delta'],1)}%)")
PYEOF
