#!/usr/bin/env bash
# tools/ci/ssm_quality_suite.sh -- SSM (RWKV) vs Qwen quality gate (Lane 4).
#
# Speed is not quality. This runs real task classes through single-stream
# `generate` on BOTH the SSM model and a Qwen reference, applies an automatic
# pass/fail check per class, and writes one timestamped report. Use it to decide
# whether RWKV may be the default for a class (see ssm_model_selection.md).
#
# Classes: long-context retrieval (fact near the START), JSON extraction,
# math sanity, instruction-following (format constraint), multilingual.
# Non-destructive. GPU jobs run SEQUENTIALLY (SSM then Qwen per prompt).
set -u

REPO="${REPO:-$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)}"; cd "$REPO" || exit 2
SSM="${SSM:-models/rwkv7-g1-04-sft-Q4_K_M.gguf}"
QWEN="${QWEN:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
BIN="${BIN:-./target/release/hawking}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-reports/ssm-quality/$STAMP}"
TOK="${TOK:-96}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.md"; LOG="$OUT/commands.log"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
gen() { # $1=model $2=prompt $3=tok -> stdout text (stats stripped)
  env HAWKING_QWEN_USER_DRAFT=0 "$BIN" generate --weights "$1" --prompt "$2" \
    --max-new-tokens "${3:-$TOK}" --temperature 0 --seed 7 2>/dev/null | grep -vE '^\[(stats|hawking)'
}

{
  echo "# SSM vs Qwen Quality Suite — $STAMP"
  echo; echo "- SSM: \`$SSM\`  | Qwen: \`$QWEN\`  | bin: \`$BIN\`"
  echo "- ⚠️ NOTE: \`hawking generate\` is RAW COMPLETION (no chat template) → Q&A results below are NOT a valid instruct"
  echo "  eval. Use the serve /v1/chat/completions endpoint (gated on the RWKV serve fix) or per-model templated prompts."
  echo "  RWKV-7-0.4B is also small; this gate is for per-class routing, not raw parity."
  echo; echo "| class | SSM | Qwen | check |"; echo "|---|---|---|---|"
} >"$SUMMARY"

ssm_pass=0; ssm_total=0
record() { # $1=class $2=ssm_ok(0/1) $3=qwen_ok(0/1)
  ssm_total=$((ssm_total+1)); [ "$2" = "1" ] && ssm_pass=$((ssm_pass+1))
  local s="❌"; [ "$2" = "1" ] && s="✅"
  local q="❌"; [ "$3" = "1" ] && q="✅"
  echo "| $1 | $s | $q | see $OUT/$1.md |" >>"$SUMMARY"
  log "$1: SSM=$2 Qwen=$3"
}
chk_contains() { printf '%s' "$1" | grep -qi "$2" && echo 1 || echo 0; }
chk_json() { printf '%s' "$1" | python3 -c 'import sys,json,re
t=sys.stdin.read()
m=re.search(r"[\[{].*[\]}]", t, re.S)
sys.exit(0 if (m and (lambda x: (json.loads(x) or True))(m.group(0))) else 1)' 2>/dev/null && echo 1 || echo 0; }

# 1. long-context RETRIEVAL — unique fact near the START of a long filler context
SECRET="the launch code is HAWKING-7741"
RET_PROMPT="$(python3 -c "print('NOTE: $SECRET.\n\n' + ('Filler sentence about unrelated systems engineering. ' * 320) + '\n\nQuestion: what is the launch code mentioned at the very beginning?')")"
a=$(gen "$SSM" "$RET_PROMPT" 32); b=$(gen "$QWEN" "$RET_PROMPT" 32)
printf '## SSM\n%s\n\n## Qwen\n%s\n' "$a" "$b" >"$OUT/retrieval.md"
record retrieval "$(chk_contains "$a" '7741')" "$(chk_contains "$b" '7741')"

# 2. JSON extraction (format-constrained)
JP="Return ONLY a JSON object with keys name and age for: Alice is 30 years old."
a=$(gen "$SSM" "$JP" 48); b=$(gen "$QWEN" "$JP" 48)
printf '## SSM\n%s\n\n## Qwen\n%s\n' "$a" "$b" >"$OUT/json.md"
record json "$(chk_json "$a")" "$(chk_json "$b")"

# 3. math sanity (deterministic answer)
MP="What is 17 multiplied by 23? Answer with the number only."
a=$(gen "$SSM" "$MP" 24); b=$(gen "$QWEN" "$MP" 24)
printf '## SSM\n%s\n\n## Qwen\n%s\n' "$a" "$b" >"$OUT/math.md"
record math "$(chk_contains "$a" '391')" "$(chk_contains "$b" '391')"

# 4. instruction-following — exact format constraint
IP="List exactly three colors, one per line, nothing else."
a=$(gen "$SSM" "$IP" 32); b=$(gen "$QWEN" "$IP" 32)
printf '## SSM\n%s\n\n## Qwen\n%s\n' "$a" "$b" >"$OUT/instruction.md"
# check: exactly 3 non-empty lines
il_a=$(printf '%s' "$a" | grep -cE '[^[:space:]]'); il_b=$(printf '%s' "$b" | grep -cE '[^[:space:]]')
record instruction "$([ "${il_a:-0}" = "3" ] && echo 1 || echo 0)" "$([ "${il_b:-0}" = "3" ] && echo 1 || echo 0)"

# 5. multilingual (weak: non-empty + contains a plausible target)
LP="Translate 'thank you very much' into Spanish. Answer with the translation only."
a=$(gen "$SSM" "$LP" 24); b=$(gen "$QWEN" "$LP" 24)
printf '## SSM\n%s\n\n## Qwen\n%s\n' "$a" "$b" >"$OUT/multilingual.md"
record multilingual "$(chk_contains "$a" 'gracias')" "$(chk_contains "$b" 'gracias')"

{
  echo; echo "## Verdict"
  echo "- SSM passed $ssm_pass / $ssm_total classes."
  echo "- Routing: a class where SSM ❌ but Qwen ✅ should route to Qwen (or hybrid). See ssm_model_selection.md."
  echo "- This gate is REQUIRED before recommending RWKV as a default for any quality-sensitive class."
} >>"$SUMMARY"
log "quality suite complete: SSM $ssm_pass/$ssm_total -> $SUMMARY"
