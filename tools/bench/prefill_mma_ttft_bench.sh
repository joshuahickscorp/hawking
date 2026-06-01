#!/usr/bin/env bash
# =============================================================================
# prefill_mma_ttft_bench.sh — paired prefill-TTFT bench for the Q4_K
#   simdgroup-MMA prefill kernel (silicon #8), in the SHIPPED config.
# =============================================================================
#
# WHAT THIS DECIDES  (plans/prefill_mma_build_plan.md §7 — the lever's perf gate)
#   The unit parity test (q4k_batched_mma_parity, atol 1e-3) proves the MMA
#   kernel is numerically correct. This bench is the *lever decision*: does the
#   rows>cols MMA twin actually cut prefill time / TTFT on the real ffn_gate +
#   ffn_up GEMMs at a long prompt? The MMA touches ONLY the batched prefill GEMM
#   (decode is the closed Type-1 bandwidth front — build plan §0/§9), so the
#   metric is prefill_ms / TTFT, NOT decode_tps.
#
# GO / NO-GO  (build plan §0 + §7 — paired delta, predec-ON)
#   env-b (MMA on) prefill_ms < env-a (MMA off) prefill_ms on the LONG prompt,
#   by more than the 3-run spread  => the rows>cols predec-MMA twin pays
#   (expected on ffn_gate/up, 11008x2048, the +22-24% in-tree microbench shape)
#   => GO. The kill for rows<=cols (attn/ffn_down occupancy) is ALREADY recorded
#   (Type-1, dead_levers.md §"Q4_K batched MMA"); this bench does NOT re-test it.
#
#   ⚠ THE PREDEC-DEFAULT-ON TRAP (build plan §3 + §7): the shipped batched path
#   is predec-ON, and the predec cache covers ffn_gate+ffn_up
#   (qwen_dense.rs:2897-2898). The MMA only moves shipped prefill if the
#   *predec-MMA twin* (gemm_q4_k_m_batched_v3w_mma_predec) is wired into the
#   PREDEC branch (Option B). A "+0% flat" result almost certainly means the
#   swap did NOT engage (MMA on but predec branch still ran scalar), NOT that
#   MMA is worthless — debug the dispatch before concluding the lever is dead.
#   This script prints a SWAP-FIRED preflight reminder.
#
# TWO REQUIRED FLAGS  (build plan §2/§7 — name BOTH)
#   DISMANTLE_QWEN_BATCH_PREFILL — default-OFF; the MMA only runs under batched
#     prefill, so this bench turns it ON for both arms (held constant).
#   DISMANTLE_QWEN_Q4K_MMA       — the lever: 0 (arm A) vs 1 (arm B).
#   predec is held ON (the shipped config) so Option B is the path under test.
#
# CONTAMINATION  (feedback_bench_with_claude_open.md + build plan §7)
#   This is a PAIRED delta (A and B share the same Claude-app GPU load), so the
#   RELATIVE prefill delta is the signal even with Claude open — no clean room
#   required. Still take >=3 runs and report the FULL spread, not a single mean
#   (feedback_report_spread_and_label_estimates). Absolute prefill_ms is
#   contaminated; the A/B ratio is not.
#
# PROMPT  (build plan §7: LONG so B>1 prefill dominates; >=256 tokens)
#   Synthesized by repeating a code unit to ~PROMPT_TOKENS tokens (~4 ch/token).
#   A 10-token prompt won't separate from noise. Override with PROMPT_FILE.
#
# USAGE
#   tools/bench/prefill_mma_ttft_bench.sh                       # 256-tok prompt, 3 runs
#   PROMPT_TOKENS=512 RUNS=5 tools/bench/prefill_mma_ttft_bench.sh
#   PROMPT_FILE=long.txt tools/bench/prefill_mma_ttft_bench.sh
#   DECODE_TOKENS=4 ...                                         # tiny decode (TTFT focus)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

# ---- config (override via env) ---------------------------------------------
BIN="${BIN:-./target/release/dismantle}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
PROMPT_TOKENS="${PROMPT_TOKENS:-256}"     # >= 256 so the batched prefill GEMM dominates
DECODE_TOKENS="${DECODE_TOKENS:-8}"       # small: TTFT = prefill + first decode steps
RUNS="${RUNS:-3}"                         # >=3, report full spread (build plan §7)
OUT="${OUT:-reports/bench/prefill_mma_ttft.json}"
PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"

die() { echo "error: $*" >&2; exit 64; }
hr()  { printf '%s\n' "================================================================================"; }

# ---- pre-flight -------------------------------------------------------------
[[ -x "$BIN" ]]     || die "binary not found/executable: $BIN (cargo build --release --workspace)."
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"
[[ -f "$PROFILE" ]] || die "kernel profile not found: $PROFILE"
"$PY" -c 'import json' 2>/dev/null || die "python3 missing for the JSON summary."
mkdir -p "$(dirname "$OUT")"

# ---- long prompt ------------------------------------------------------------
synth_prompt() {  # ~PROMPT_TOKENS tokens (~4 chars/token), repeated code unit
  local target_chars=$(( PROMPT_TOKENS * 4 )) acc=""
  local unit='fn step(s: &mut State, i: usize) -> u64 { s.acc += i as u64; s.acc } '
  while [[ ${#acc} -lt $target_chars ]]; do acc+="$unit"; done
  printf '%s' "${acc:0:$target_chars}"
}
if [[ -n "${PROMPT_FILE:-}" ]]; then
  [[ -f "$PROMPT_FILE" ]] || die "PROMPT_FILE not found: $PROMPT_FILE"
  PROMPT="$(cat "$PROMPT_FILE")"
else
  PROMPT="$(synth_prompt)"
fi
PROMPT_CHARS=${#PROMPT}

# ---- arm env (predec held ON = shipped; BATCH_PREFILL held ON; MMA = lever) -
# The locked Qwen fast-path (shipped decode config) + batched prefill on.
BASE_ENV="DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
DISMANTLE_QWEN_Q4K_PREDEC=1 DISMANTLE_QWEN_BATCH_PREFILL=1"
env_for() { case "$1" in A) echo "DISMANTLE_QWEN_Q4K_MMA=0";; B) echo "DISMANTLE_QWEN_Q4K_MMA=1";; esac; }
arm_label() { case "$1" in A) echo "A MMA=0 (v3w/predec scalar)";; B) echo "B MMA=1 (predec-MMA twin)";; esac; }

# One run -> emits "<prefill_ms> <dec_tps> <completion>". TTFT proxy = prefill_ms
# (the time to the first decodable token; the [stats] line carries it directly).
run_one() {  # $1=arm
  local arm="$1" log
  log="$(mktemp -t prefmma.XXXXXX)"
  # shellcheck disable=SC2086
  env $BASE_ENV $(env_for "$arm") \
    nice -n 19 taskpolicy -b "$BIN" generate \
      --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
      --prompt "$PROMPT" --max-new-tokens "$DECODE_TOKENS" --temperature 0 --seed 0 \
      > /dev/null 2> "$log"
  local stats pre tps comp
  stats="$(grep -E '^\[stats\]' "$log" | tail -1)"
  pre="$(printf '%s' "$stats"  | grep -oE 'prefill_ms=[0-9.]+' | grep -oE '[0-9.]+')"
  tps="$(printf '%s' "$stats"  | grep -oE 'dec_tps=[0-9.]+'    | grep -oE '[0-9.]+')"
  comp="$(printf '%s' "$stats" | grep -oE 'completion=[0-9]+'  | grep -oE '[0-9]+')"
  rm -f "$log"
  echo "${pre:-0} ${tps:-0} ${comp:-0}"
}
med() { printf '%s\n' $1 | tr ' ' '\n' | grep -E '.' | sort -n | awk '{a[NR]=$0} END{if(NR)print a[int((NR+1)/2)]; else print 0}'; }
minv() { printf '%s\n' $1 | tr ' ' '\n' | grep -E '.' | sort -n | head -1; }
maxv() { printf '%s\n' $1 | tr ' ' '\n' | grep -E '.' | sort -n | tail -1; }

# ---- contamination note -----------------------------------------------------
DIRTY=0
pgrep -f "Claude.app" >/dev/null 2>&1 && DIRTY=1

hr
echo "  prefill-MMA TTFT paired bench (silicon #8, build plan §7)"
echo "  weights=$WEIGHTS  profile=$PROFILE"
echo "  prompt ~${PROMPT_TOKENS} tok (${PROMPT_CHARS} chars), decode=${DECODE_TOKENS} tok, runs=$RUNS"
echo "  env (both arms): predec ON + BATCH_PREFILL ON (shipped). Lever: Q4K_MMA 0 vs 1."
echo "  paired delta => Claude-open OK ($([[ $DIRTY == 1 ]] && echo "Claude RUNNING — relative delta is still valid" || echo "clean room")); 3-run spread reported."
hr
echo "  PREFLIGHT REMINDER (build plan §3/§7 — the predec-default-ON trap):"
echo "    The MMA only moves shipped prefill if the predec-MMA twin is wired into"
echo "    the PREDEC branch (Option B). If B/A is ~+0% flat, the swap likely did"
echo "    NOT engage on ffn_gate/up — confirm gemm_q4_k_m_batched_v3w_mma_predec"
echo "    actually dispatched before recording NO-GO."
echo ""

# ---- run, interleaved A/B ---------------------------------------------------
PRE_A="" PRE_B="" TPS_A="" TPS_B="" COMP_A="" COMP_B=""
for r in $(seq 1 "$RUNS"); do
  for arm in A B; do
    read -r pre tps comp <<<"$(run_one "$arm")"
    eval "PRE_$arm=\"\$PRE_$arm $pre\"; TPS_$arm=\"\$TPS_$arm $tps\"; COMP_$arm=\"\$COMP_$arm $comp\""
  done
done

mPA="$(med "$PRE_A")"; mPB="$(med "$PRE_B")"
loPA="$(minv "$PRE_A")"; hiPA="$(maxv "$PRE_A")"
loPB="$(minv "$PRE_B")"; hiPB="$(maxv "$PRE_B")"
mTA="$(med "$TPS_A")"; mTB="$(med "$TPS_B")"

echo ""
printf "  %-30s prefill_ms median=%-8s  spread=[%s, %s]   (decode_tps med=%s)\n" \
  "$(arm_label A)" "$mPA" "$loPA" "$hiPA" "$mTA"
printf "  %-30s prefill_ms median=%-8s  spread=[%s, %s]   (decode_tps med=%s)\n" \
  "$(arm_label B)" "$mPB" "$loPB" "$hiPB" "$mTB"
echo "    A prefill_ms runs:$PRE_A"
echo "    B prefill_ms runs:$PRE_B"
awk -v a="$mPA" -v b="$mPB" 'BEGIN{
  if (a>0) {
    printf "  >> prefill B/A = %.3fx  (%+.1f%%)   [<1.0 = MMA faster = GO direction]\n", b/a, (b/a-1)*100;
  }
}'
echo "  >> metric = prefill_ms (TTFT proxy). decode_tps shown only to confirm"
echo "     decode is UNTOUCHED (MMA is prefill-only; build plan §0)."

# ---- machine-readable summary ----------------------------------------------
TPS_TAG="$([[ $DIRTY == 1 ]] && echo "paired-delta-valid (Claude open)" || echo clean)"
"$PY" - "$OUT" "$TPS_TAG" "$PROMPT_TOKENS" "$DECODE_TOKENS" "$RUNS" \
  "$mPA" "$mPB" "$loPA" "$hiPA" "$loPB" "$hiPB" "$mTA" "$mTB" \
  "$PRE_A" "$PRE_B" <<'PYEOF'
import json, sys
(out, tag, ptok, dtok, runs, mPA, mPB, loPA, hiPA, loPB, hiPB, mTA, mTB,
 preA, preB) = sys.argv[1:16]
f = float
mPA, mPB = f(mPA), f(mPB)
doc = {
    "bench": "prefill_mma_ttft",
    "note": tag,
    "prompt_tokens": int(ptok), "decode_tokens": int(dtok), "runs": int(runs),
    "env_held": ["DISMANTLE_QWEN_Q4K_PREDEC=1 (shipped)",
                 "DISMANTLE_QWEN_BATCH_PREFILL=1"],
    "lever": "DISMANTLE_QWEN_Q4K_MMA  (A=0, B=1)",
    "prefill_ms": {
        "A_mma_off": {"median": mPA, "spread": [f(loPA), f(hiPA)],
                      "runs": [f(x) for x in preA.split()]},
        "B_mma_on":  {"median": mPB, "spread": [f(loPB), f(hiPB)],
                      "runs": [f(x) for x in preB.split()]},
        "ratio_B_over_A": (mPB/mPA if mPA else None),
        "pct_delta": ((mPB/mPA - 1) * 100 if mPA else None),
    },
    "decode_tps_sanity": {"A": f(mTA), "B": f(mTB),
                          "expect": "~equal; MMA is prefill-only"},
    "decision": "B/A < 1.0 by > 3-run spread => MMA cuts prefill => GO. "
                "~+0% flat => check the swap fired (predec-default-ON trap).",
}
json.dump(doc, open(out, "w"), indent=2)
print(f"\nwrote {out}")
PYEOF

hr
echo "  DONE. GO if prefill B/A < 1.0 by more than the 3-run spread (build plan §7)."
echo "  If flat, the predec-MMA swap likely didn't engage — debug dispatch first."
hr
