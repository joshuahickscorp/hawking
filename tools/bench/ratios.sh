#!/usr/bin/env bash
# tools/bench/ratios.sh — reproducible warm-median decode-tps + argmax-identity quality harness.
#
# The campaign's #1 method lesson: a single COLD run measures PSO shader-compile, NOT steady-state.
# This harness always takes a warm median over >=TRIALS fresh runs and prints the exact config.
#
# Subcommands:
#   tps   <label> <extra-env> [ctx] [trials]   warm-median dec_tps for one config
#   ab    <extra-env> [ctx] [trials]           A/B: default vs the config (tps + Δ%), sequential
#   abi   <extra-env> [ctx] [trials]           INTERLEAVED A/B (alternating trials) — for fine (<10%) deltas
#   qual  <extra-env> [profile] [tok]          argmax-identity over the adversarial suite vs the bit-identical default
#
#   ctx = short | long | <path-to-prompt-file>     (long expects /tmp/ctx6k.txt, ~8k tokens)
#   extra-env = space-separated KEY=VAL passed to `env` (use "" for none)
#   PROFILE=fast as env, or the qual <profile> arg, adds `--profile <p>`.
#
# Examples:
#   tools/bench/ratios.sh ab "HAWKING_QWEN_F16_KV=1" long
#   tools/bench/ratios.sh tps q6k4r "HAWKING_QWEN_Q6K_SWIGLU_4R=1" short 5
#   tools/bench/ratios.sh qual "" fast 100
set -u
REPO="${REPO:-$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)}"; cd "$REPO" || exit 2
BIN="${BIN:-./target/release/hawking}"
M="${M:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
TOK="${TOK:-128}"; SEED="${SEED:-5}"
SHORTP="Write a detailed Python implementation of a binary search tree with insert, search, delete, and in-order traversal, with docstrings and type hints."
# adversarial quality suite: code, math, JSON, multilingual, formatting, edge cases
SUITE=(
  "Write a Python function to reverse a singly linked list, with type hints."
  "Compute 47 times 89 step by step, then verify."
  "Return ONLY valid JSON: an array of 3 users with id, name, and email fields."
  "Translate 'good morning, how are you' into French, German, Japanese, and Arabic."
  "Write a haiku about a thunderstorm, exactly three lines."
  "List the first 10 prime numbers separated by commas."
  "Write a SQL query for the top 5 customers by total revenue this year."
  "Explain TCP vs UDP in exactly three sentences."
  "Write a regex that matches an RFC-5322-ish email address."
  "Sort these numbers ascending: 8, 3, 91, 0, 17, 5, 42."
)
med() { sort -n | awk '{a[NR]=$1} END{print (NR>0)?a[int((NR+1)/2)]:"ERR"}'; }
prompt_for() { case "$1" in short) printf '%s' "$SHORTP";; long) cat /tmp/ctx6k.txt 2>/dev/null || printf '%s' "$SHORTP";; *) cat "$1";; esac; }
# warm-median tps for: $1=extra-env $2=prompt $3=trials $4=profile
tps_of() {
  local ev="$1" pr="$2" tr="$3" prof="$4" t out
  for t in $(seq 1 "$tr"); do
    out=$(env HAWKING_QWEN_USER_DRAFT=0 $ev "$BIN" generate --weights "$M" --prompt "$pr" \
            --max-new-tokens "$TOK" --temperature 0 --seed "$SEED" ${prof:+--profile $prof} 2>&1)
    printf '%s\n' "$out" | grep -oE 'dec_tps=[0-9.]+' | cut -d= -f2
  done | med
}
gen_text() { env HAWKING_QWEN_USER_DRAFT=0 $1 "$BIN" generate --weights "$M" --prompt "$2" \
              --max-new-tokens "${3:-80}" --temperature 0 --seed 7 ${4:+--profile $4} 2>/dev/null; }

cmd="${1:-}"; shift || true
case "$cmd" in
  tps)
    label="$1"; ev="${2:-}"; ctx="${3:-short}"; tr="${4:-5}"
    printf "%-16s ctx=%-6s trials=%s tps=%s\n" "$label" "$ctx" "$tr" "$(tps_of "$ev" "$(prompt_for "$ctx")" "$tr" "${PROFILE:-}")"
    ;;
  ab)
    ev="${1:-}"; ctx="${2:-short}"; tr="${3:-5}"; p="$(prompt_for "$ctx")"
    d=$(tps_of "" "$p" "$tr" ""); c=$(tps_of "$ev" "$p" "$tr" "${PROFILE:-}")
    delta=$(awk -v d="$d" -v c="$c" 'BEGIN{ if(d+0>0) printf "%+.1f%%",(c-d)/d*100; else print "n/a"}')
    printf "ctx=%-6s default=%s  cfg[%s]=%s  Δ=%s\n" "$ctx" "$d" "${ev:-PROFILE=${PROFILE:-}}" "$c" "$delta"
    ;;
  abi)
    # INTERLEAVED A/B — alternate default/cfg trials so per-process cold-PSO noise hits both
    # arms equally (the `ab` mode runs all-default-then-all-cfg, which biased the 2026-06-21
    # attribution sweep; default drifted 30-40). Use this for fine (<10%) deltas.
    ev="${1:-}"; ctx="${2:-short}"; tr="${3:-8}"; p="$(prompt_for "$ctx")"; dd=""; cc=""
    for i in $(seq 1 "$tr"); do
      dd="$dd
$(env HAWKING_QWEN_USER_DRAFT=0 "$BIN" generate --weights "$M" --prompt "$p" --max-new-tokens "$TOK" --temperature 0 --seed "$SEED" 2>&1 | grep -oE 'dec_tps=[0-9.]+' | cut -d= -f2)"
      cc="$cc
$(env HAWKING_QWEN_USER_DRAFT=0 $ev "$BIN" generate --weights "$M" --prompt "$p" --max-new-tokens "$TOK" --temperature 0 --seed "$SEED" ${PROFILE:+--profile $PROFILE} 2>&1 | grep -oE 'dec_tps=[0-9.]+' | cut -d= -f2)"
    done
    d=$(printf '%s\n' "$dd" | grep . | med); c=$(printf '%s\n' "$cc" | grep . | med)
    delta=$(awk -v d="$d" -v c="$c" 'BEGIN{ if(d+0>0) printf "%+.1f%%",(c-d)/d*100; else print "n/a"}')
    printf "ctx=%-6s [interleaved n=%s] default=%s  cfg[%s]=%s  Δ=%s\n" "$ctx" "$tr" "$d" "${ev:-PROFILE=${PROFILE:-}}" "$c" "$delta"
    ;;
  qual)
    ev="${1:-}"; prof="${2:-}"; tok="${3:-80}"; div=0; n=0
    for p in "${SUITE[@]}"; do
      a=$(gen_text "" "$p" "$tok" ""); b=$(gen_text "$ev" "$p" "$tok" "$prof")
      [ "$a" = "$b" ] || div=$((div+1)); n=$((n+1))
    done
    printf "qual cfg[%s%s]: %d/%d argmax-identical = %s\n" "${ev}" "${prof:+ --profile $prof}" "$((n-div))" "$n" \
      "$(awk -v d="$div" -v n="$n" 'BEGIN{printf "%.0f%%",(n-d)/n*100}')"
    ;;
  *)
    grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
