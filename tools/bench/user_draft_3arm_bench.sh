#!/usr/bin/env bash
# =============================================================================
# user_draft_3arm_bench.sh — 3-arm paired bench for the user-ngram draft loop.
#   A = plain fast decode          (HAWKING_QWEN_USER_DRAFT off)
#   B = bonus-first  user-draft    (HAWKING_QWEN_USER_DRAFT=1)
#   C = propose-first user-draft   (+ HAWKING_QWEN_USER_DRAFT_PROPOSE_FIRST=1)
# =============================================================================
#
# WHAT THIS DECIDES  (reports/move2_user_draft_diagnosis.md §6 — the single
#   decisive gate for whether propose-first is a real tps win)
#   The diagnosis pinned that the prior "draft_accepted=0, zero [verify-timing]"
#   non-result was a HARNESS gap: no script in the repo exported
#   HAWKING_QWEN_USER_DRAFT (clean_room_batch.sh / paired_lever.sh both omit
#   it) or HAWKING_QWEN_VERIFY_TIMING. This script closes that gap: it turns
#   the draft ON for arms B/C, sets VERIFY_TIMING, and reports the mechanism
#   check (forward count via the [verify-timing] line count) alongside dec_tps.
#
#   Mechanism arithmetic (diagnosis §3): bonus-first pays 2 GPU forwards/cycle
#   (Stage-1 bonus + Stage-3 verify); propose-first pays 1 (verify only, +1
#   one-time bootstrap). So C should show ~2x FEWER [verify-timing]-era forwards
#   per emitted token than B, with the gap largest at LOW acceptance.
#
# GO / NO-GO  (diagnosis §6 decision rule — paired deltas, NOT absolute tps)
#   C > B on the LOW-acceptance prompt by more than the inter-trial spread
#       => propose-first removes the 2-forward penalty as predicted => KEEP
#          (consider default-on behind the flag).
#   C ~= B on the HIGH-acceptance prompt (bonus amortized either way) => no
#       regression, fine.
#   C < A on EITHER prompt => net-negative even propose-first => keep default-
#       off; the lever is repetition-only.
#   The draft is LOSSLESS by construction (bit-identical to plain greedy, gated
#   by commit 354d718 + the new parity tests); this bench is PERF-only. Quality
#   is not at stake here.
#
# CLEAN-ROOM  ⚠  This is an ABSOLUTE-tps comparison across THREE different
#   decode loops, so it is contamination-sensitive in a way a pure A/B is not:
#   the in-session table in draft_tuning_verify_findings is contaminated (1.3-30x
#   swings, memory/bench_contamination.md). The diagnosis §6 mandates the agent
#   QUIT for the tps verdict. The harness still INTERLEAVES the arms (A,B,C,A,..)
#   so thermal drift cancels; the per-arm draft_accepted + forward-count
#   mechanism check is load-independent and trustworthy even if run dirty, but
#   the dec_tps verdict requires a clean room. Pre-flight gates enforce it
#   unless ALLOW_DIRTY=1 (then tps is tagged CONTAMINATED in the output).
#
# PROMPTS  (diagnosis §6 step 4: two prompt classes, >=128 tok each)
#   LOW  acceptance: non-repetitive / natural — where bonus-first's 2nd forward
#                    should hurt and propose-first should recover.
#   HIGH acceptance: repetitive code — where the n-gram draft hits often.
#   Both are synthesized to >=128 tokens (override with LOW_PROMPT/HIGH_PROMPT
#   or LOW_PROMPT_FILE/HIGH_PROMPT_FILE).
#
# ENV  (the locked Qwen fast-path = the SHIPPED decode config, + the draft flags)
#   base : TCB=1 VOCAB_PRUNE=32000 Q4K_LMHEAD=1 FFN_DOWN_Q4K=1 Q4K_PREDEC=1
#   A    : (base only)
#   B    : base + HAWKING_QWEN_USER_DRAFT=1            HAWKING_QWEN_VERIFY_TIMING=1
#   C    : base + HAWKING_QWEN_USER_DRAFT=1 + _PROPOSE_FIRST=1  + VERIFY_TIMING=1
#   USER_DRAFT_K override via DRAFT_K (default: engine default).
#
# USAGE
#   tools/bench/user_draft_3arm_bench.sh                     # both prompts, TRIALS=20
#   TRIALS=10 TOKENS=128 tools/bench/user_draft_3arm_bench.sh
#   DRAFT_K=4 tools/bench/user_draft_3arm_bench.sh
#   ALLOW_DIRTY=1 tools/bench/user_draft_3arm_bench.sh       # mechanism-only, tps CONTAMINATED
# =============================================================================
set -uo pipefail
_agent_env="$(git rev-parse --show-toplevel 2>/dev/null)/.agent_env"
[ -f "$_agent_env" ] && source "$_agent_env"
unset _agent_env
cd "$(dirname "$0")/../.."

# ---- config (override via env) ---------------------------------------------
BIN="${BIN:-./target/release/hawking}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
TOKENS="${TOKENS:-160}"                # >= 128 decode tokens (diagnosis §6)
TRIALS="${TRIALS:-20}"                 # M5 stack-matrix discipline (diagnosis §6 step 5)
DRAFT_K="${DRAFT_K:-}"                 # optional HAWKING_QWEN_USER_DRAFT_K
OUT="${OUT:-reports/bench/user_draft_3arm.json}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"

die() { echo "error: $*" >&2; exit 64; }
hr()  { printf '%s\n' "================================================================================"; }

# ---- prompts: >=128 tokens, one low- and one high-acceptance ---------------
# HIGH acceptance = highly repetitive code (the n-gram re-emits prior chunks).
synth_high() {  # repeat a short code unit to ~TOKENS tokens (~4 chars/token)
  local target_chars=$(( TOKENS * 6 )) acc=""
  local unit='fn step(s: &mut State, i: usize) -> u64 { s.acc += i as u64; s.acc }
'
  while [[ ${#acc} -lt $target_chars ]]; do acc+="$unit"; done
  printf '%s' "${acc:0:$target_chars}"
}
# LOW acceptance = non-repetitive natural prose (few n-gram hits).
synth_low() {
  printf '%s' \
"Summarize the tradeoffs between optimistic and pessimistic concurrency control \
in distributed databases, then explain how a hybrid scheme might adapt its \
strategy under varying contention, latency, and failure conditions across \
geographically separated replicas, with attention to the implications for \
linearizability, throughput, tail latency, and operator observability in a \
production system that serves a mixed read-write workload at scale."
}

if [[ -n "${HIGH_PROMPT_FILE:-}" ]]; then HIGH_PROMPT="$(cat "$HIGH_PROMPT_FILE")"; fi
if [[ -n "${LOW_PROMPT_FILE:-}"  ]]; then LOW_PROMPT="$(cat "$LOW_PROMPT_FILE")";  fi
HIGH_PROMPT="${HIGH_PROMPT:-$(synth_high)}"
LOW_PROMPT="${LOW_PROMPT:-$(synth_low)}"

# ---- pre-flight -------------------------------------------------------------
[[ -x "$BIN" ]]     || die "binary not found/executable: $BIN (cargo build --release -p hawking)."
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"
[[ -f "$PROFILE" ]] || die "kernel profile not found: $PROFILE"
"$PY" -c 'import json' 2>/dev/null || die "python3 missing for the JSON summary."
mkdir -p "$(dirname "$OUT")"

# Clean-room gate (the dec_tps verdict needs it; mechanism check does not).
DIRTY=0
if pgrep -f "${AGENT_APP_PGREP:?see .agent_env.example}" >/dev/null 2>&1; then DIRTY=1; fi
pgrep -x "${AGENT_CLI_PGREP:?see .agent_env.example}" >/dev/null 2>&1 && DIRTY=1
pgrep -f "MASTER_LOOP" >/dev/null 2>&1 && DIRTY=1
pgrep -i slm 2>/dev/null | grep -vq aslmanager && DIRTY=1
if [[ "$DIRTY" == 1 ]]; then
  if [[ "$ALLOW_DIRTY" == 1 ]]; then
    echo "  ⚠ DIRTY room (agent/slm running) + ALLOW_DIRTY=1 — dec_tps will be" >&2
    echo "    CONTAMINATED (1.3-30x). Only draft_accepted + forward-count are"   >&2
    echo "    trustworthy. The diagnosis §6 tps verdict needs the agent QUIT."   >&2
  else
    echo "  ❌ CLEAN-ROOM GATE FAILED: agent/slm running. The 3-arm dec_tps" >&2
    echo "     verdict is absolute-tps and contaminates (diagnosis §6). Cmd+Q" >&2
    echo "     the agent + quit slm, then re-run. (ALLOW_DIRTY=1 for mechanism-"  >&2
    echo "     only: draft_accepted + forward count, tps tagged CONTAMINATED.)" >&2
    exit 1
  fi
fi

# ---- arm env builders -------------------------------------------------------
BASE_ENV="HAWKING_QWEN_TCB=1 HAWKING_QWEN_VOCAB_PRUNE=32000 \
HAWKING_QWEN_Q4K_LMHEAD=1 HAWKING_QWEN_FFN_DOWN_Q4K=1 \
HAWKING_QWEN_Q4K_PREDEC=1"
[[ -n "$DRAFT_K" ]] && DRAFTK_ENV="HAWKING_QWEN_USER_DRAFT_K=$DRAFT_K" || DRAFTK_ENV=""

env_for() {  # $1 = arm A|B|C
  case "$1" in
    A) echo "" ;;
    B) echo "HAWKING_QWEN_USER_DRAFT=1 HAWKING_QWEN_VERIFY_TIMING=1 $DRAFTK_ENV" ;;
    C) echo "HAWKING_QWEN_USER_DRAFT=1 HAWKING_QWEN_USER_DRAFT_PROPOSE_FIRST=1 \
HAWKING_QWEN_VERIFY_TIMING=1 $DRAFTK_ENV" ;;
  esac
}
arm_label() { case "$1" in A) echo "A plain";; B) echo "B bonus-first";; C) echo "C propose-first";; esac; }

# One run -> emits "<dec_tps> <draft_accepted> <draft_rejected> <verify_timing_lines>".
# Captures stderr (both [stats] and [verify-timing] go there).
run_one() {  # $1=arm  $2=prompt
  local arm="$1" prompt="$2" log
  log="$(mktemp -t ud3arm.XXXXXX)"
  # shellcheck disable=SC2086
  env $BASE_ENV $(env_for "$arm") \
    nice -n 19 taskpolicy -b "$BIN" generate \
      --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
      --prompt "$prompt" --max-new-tokens "$TOKENS" --temperature 0 --seed 0 \
      > /dev/null 2> "$log"
  local stats vt tps acc rej
  stats="$(grep -E '^\[stats\]' "$log" | tail -1)"
  vt="$(grep -c '^\[verify-timing\]' "$log")"
  tps="$(printf '%s' "$stats" | grep -oE 'dec_tps=[0-9.]+'        | grep -oE '[0-9.]+')"
  acc="$(printf '%s' "$stats" | grep -oE 'draft_accepted=[0-9]+' | grep -oE '[0-9]+')"
  rej="$(printf '%s' "$stats" | grep -oE 'draft_rejected=[0-9]+' | grep -oE '[0-9]+')"
  rm -f "$log"
  echo "${tps:-0} ${acc:-0} ${rej:-0} ${vt:-0}"
}

med() { printf '%s\n' $1 | tr ' ' '\n' | grep -E '.' | sort -n | awk '{a[NR]=$0} END{if(NR)print a[int((NR+1)/2)]; else print 0}'; }

# ---- run one prompt class across A/B/C, interleaved ------------------------
run_prompt_class() {  # $1=class name  $2=prompt   -> sets RESULT_* arrays via globals
  local cls="$1" prompt="$2"
  local TPS_A="" TPS_B="" TPS_C="" ACC_A="" ACC_B="" ACC_C=""
  local REJ_A="" REJ_B="" REJ_C="" VT_A="" VT_B="" VT_C=""
  hr
  echo "  PROMPT CLASS: $cls  (TOKENS=$TOKENS, TRIALS=$TRIALS, interleaved A/B/C)"
  echo "  prompt[0:80]: ${prompt:0:80}..."
  hr
  local t arm fields tps acc rej vt
  for t in $(seq 1 "$TRIALS"); do
    for arm in A B C; do
      fields="$(run_one "$arm" "$prompt")"
      read -r tps acc rej vt <<<"$fields"
      eval "TPS_$arm=\"\$TPS_$arm $tps\"; ACC_$arm=\"\$ACC_$arm $acc\""
      eval "REJ_$arm=\"\$REJ_$arm $rej\"; VT_$arm=\"\$VT_$arm $vt\""
    done
  done
  local mA mB mC
  mA="$(med "$TPS_A")"; mB="$(med "$TPS_B")"; mC="$(med "$TPS_C")"
  local aA aB aC vA vB vC
  aA="$(med "$ACC_A")"; aB="$(med "$ACC_B")"; aC="$(med "$ACC_C")"
  vA="$(med "$VT_A")";  vB="$(med "$VT_B")";  vC="$(med "$VT_C")"
  echo ""
  printf "  %-16s median_dec_tps=%-8s draft_accepted(med)=%-5s verify_timing_lines(med)=%-5s\n" \
    "$(arm_label A)" "$mA" "$aA" "$vA"
  printf "  %-16s median_dec_tps=%-8s draft_accepted(med)=%-5s verify_timing_lines(med)=%-5s\n" \
    "$(arm_label B)" "$mB" "$aB" "$vB"
  printf "  %-16s median_dec_tps=%-8s draft_accepted(med)=%-5s verify_timing_lines(med)=%-5s\n" \
    "$(arm_label C)" "$mC" "$aC" "$vC"
  echo "    A trials tps:$TPS_A"
  echo "    B trials tps:$TPS_B"
  echo "    C trials tps:$TPS_C"
  awk -v a="$mA" -v b="$mB" -v c="$mC" 'BEGIN{
    if (a>0) {
      printf "  >> B/A = %.3fx (%+.1f%%)   C/A = %.3fx (%+.1f%%)   C/B = %.3fx (%+.1f%%)\n",
        b/a,(b/a-1)*100, c/a,(c/a-1)*100, (b>0? c/b:0),(b>0?(c/b-1)*100:0);
    }
  }'
  echo "  >> mechanism: B verify-forwards/token ~ $(awk -v v="$vB" -v t="$TOKENS" 'BEGIN{if(t>0)printf "%.3f", v/t; else print "?"}'),"
  echo "                C verify-forwards/token ~ $(awk -v v="$vC" -v t="$TOKENS" 'BEGIN{if(t>0)printf "%.3f", v/t; else print "?"}'),"
  echo "                (A has 0 by construction — plain decode never calls forward_tokens_verify)."
  # stash for JSON
  eval "JSON_${cls}_mA=$mA JSON_${cls}_mB=$mB JSON_${cls}_mC=$mC"
  eval "JSON_${cls}_aA=$aA JSON_${cls}_aB=$aB JSON_${cls}_aC=$aC"
  eval "JSON_${cls}_vA=$vA JSON_${cls}_vB=$vB JSON_${cls}_vC=$vC"
}

hr
echo "  user-draft 3-arm paired bench  (A plain / B bonus-first / C propose-first)"
echo "  clean-room=$([[ $DIRTY == 1 ]] && echo "DIRTY (tps CONTAMINATED)" || echo OK)"
hr
run_prompt_class low  "$LOW_PROMPT"
run_prompt_class high "$HIGH_PROMPT"

# ---- machine-readable summary ----------------------------------------------
TPS_TAG="$([[ $DIRTY == 1 ]] && echo CONTAMINATED || echo clean)"
"$PY" - "$OUT" "$TPS_TAG" "$TOKENS" "$TRIALS" \
  "${JSON_low_mA:-0}"  "${JSON_low_mB:-0}"  "${JSON_low_mC:-0}" \
  "${JSON_low_aA:-0}"  "${JSON_low_aB:-0}"  "${JSON_low_aC:-0}" \
  "${JSON_low_vA:-0}"  "${JSON_low_vB:-0}"  "${JSON_low_vC:-0}" \
  "${JSON_high_mA:-0}" "${JSON_high_mB:-0}" "${JSON_high_mC:-0}" \
  "${JSON_high_aA:-0}" "${JSON_high_aB:-0}" "${JSON_high_aC:-0}" \
  "${JSON_high_vA:-0}" "${JSON_high_vB:-0}" "${JSON_high_vC:-0}" <<'PYEOF'
import json, sys
(out, tag, tokens, trials,
 lmA, lmB, lmC, laA, laB, laC, lvA, lvB, lvC,
 hmA, hmB, hmC, haA, haB, haC, hvA, hvB, hvC) = sys.argv[1:23]
f = float
def cls(mA, mB, mC, aA, aB, aC, vA, vB, vC):
    mA, mB, mC = f(mA), f(mB), f(mC)
    return {
        "median_dec_tps": {"A_plain": mA, "B_bonus_first": mB, "C_propose_first": mC},
        "ratios": {
            "B_over_A": (mB/mA if mA else None),
            "C_over_A": (mC/mA if mA else None),
            "C_over_B": (mC/mB if mB else None),
        },
        "draft_accepted_median": {"A": f(aA), "B": f(aB), "C": f(aC)},
        "verify_timing_lines_median": {"A": f(vA), "B": f(vB), "C": f(vC)},
    }
doc = {
    "bench": "user_draft_3arm",
    "tps_quality": tag,  # clean | CONTAMINATED
    "tokens": int(tokens), "trials": int(trials),
    "arms": {"A": "plain", "B": "bonus-first (USER_DRAFT=1)",
             "C": "propose-first (USER_DRAFT=1 + PROPOSE_FIRST=1)"},
    "low_acceptance": cls(lmA, lmB, lmC, laA, laB, laC, lvA, lvB, lvC),
    "high_acceptance": cls(hmA, hmB, hmC, haA, haB, haC, hvA, hvB, hvC),
    "decision_rule": "C>B on LOW by > inter-trial spread => keep propose-first; "
                     "C<A on either => keep default-off (repetition-only).",
}
json.dump(doc, open(out, "w"), indent=2)
print(f"\nwrote {out}  (tps_quality={tag})")
PYEOF

hr
echo "  DONE. Verdict (diagnosis §6): C>B on the LOW prompt by > spread => propose-"
echo "  first removes the 2-forward penalty => keep it. C<A on either => default-off."
echo "  If tps_quality=CONTAMINATED, re-run with the agent QUIT before trusting tps;"
echo "  the draft_accepted + verify-timing-line counts above are already trustworthy."
hr
