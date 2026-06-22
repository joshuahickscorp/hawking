#!/usr/bin/env bash
# tools/ci/ssm_quality_chat.sh — VALID instruct quality eval via /v1/chat/completions.
#
# The raw-`generate` suite (ssm_quality_suite.sh) is explicitly NOT a valid instruct
# eval (no chat template). This one drives the SERVE chat endpoint, which applies the
# per-arch chat template, so the model sees a real instruct prompt. It runs the same
# task classes through chat on each model and writes one comparison report. Use it to
# quantify RWKV-7 instruct quality per class and to make routing decisions (R3/R5).
#
# Per model: start `hawking serve`, POST each class via /v1/chat/completions, grade,
# stop the server. GPU jobs are SEQUENTIAL (one serve process at a time).
# bash 3.2-safe (no associative arrays); results accumulate in a TSV.
#
# Env:
#   MODELS="<path> <path> ..."   models to eval (default: RWKV-7-SFT then Qwen-3B)
#   TOK=96                        max_tokens per answer
#   PORT=18450                    base serve port (incremented per model)
#   OUT=<dir>                     report dir (default reports/ssm-quality-chat/<stamp>)
set -u

REPO="${REPO:-$HOME/Downloads/hawking}"; cd "$REPO" || exit 2
BIN="${BIN:-./target/release/hawking}"
MODELS="${MODELS:-models/rwkv7-g1-04-sft-Q4_K_M.gguf models/qwen2.5-3b-instruct-q4_k_m.gguf}"
TOK="${TOK:-96}"
PORT="${PORT:-18450}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-reports/ssm-quality-chat/$STAMP}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.md"; LOG="$OUT/commands.log"; TSV="$OUT/results.tsv"

if [ ! -x "$BIN" ]; then echo "need $BIN (cargo build --release -p hawking)"; exit 2; fi
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# --- grading (same checks as ssm_quality_suite.sh) ---
chk_contains() { printf '%s' "$1" | grep -qi "$2" && echo 1 || echo 0; }
chk_json() { printf '%s' "$1" | python3 -c 'import sys,json,re
t=sys.stdin.read()
m=re.search(r"[\[{].*[\]}]", t, re.S)
sys.exit(0 if (m and (lambda x: (json.loads(x) or True))(m.group(0))) else 1)' 2>/dev/null && echo 1 || echo 0; }
chk_lines3() { [ "$(printf '%s' "$1" | grep -cE '[^[:space:]]')" = "3" ] && echo 1 || echo 0; }

# --- serve lifecycle ---
SPID=""
stop_serve() { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=""; }
trap stop_serve EXIT
start_serve() { # $1=model $2=port
  ./target/release/hawking serve --weights "$1" --addr "127.0.0.1:$2" >"$OUT/serve_$2.log" 2>&1 &
  SPID=$!
  local i; for i in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$2/healthz" >/dev/null 2>&1 && return 0
    if ! kill -0 "$SPID" 2>/dev/null; then log "serve died on startup (see $OUT/serve_$2.log)"; return 1; fi
    sleep 1
  done
  log "serve healthz timeout"; return 1
}
chat() { # $1=port $2=prompt $3=max_tokens -> assistant content
  local body; body="$(python3 -c 'import json,sys
print(json.dumps({"model":"m","messages":[{"role":"user","content":sys.argv[1]}],
  "max_tokens":int(sys.argv[2]),"temperature":0,"seed":7,"stream":False}))' "$2" "$3")"
  curl -sS --max-time 120 -H 'content-type: application/json' -d "$body" \
    "http://127.0.0.1:$1/v1/chat/completions" 2>/dev/null | python3 -c 'import sys,json
try:
  d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])
except Exception: print("")'
}

SECRET="HAWKING-7741"
RET_PROMPT="$(python3 -c "print('NOTE: the launch code is $SECRET.\n\n' + ('Filler sentence about unrelated systems engineering. ' * 320) + '\n\nQuestion: what is the launch code mentioned at the very beginning?')")"

{
  echo "# Valid Instruct Quality (chat-templated, /v1/chat/completions) — $STAMP"
  echo; echo "- Endpoint: \`/v1/chat/completions\` (applies the per-arch chat template — a VALID instruct eval, unlike raw \`generate\`)."
  echo "- TOK=$TOK, temperature=0, seed=7. Classes: retrieval, json, math, instruction, multilingual."
  echo
} >"$SUMMARY"
: >"$TSV"

for M in $MODELS; do
  tag="$(basename "$M" | sed 's/\.gguf$//')"
  log "=== model: $tag ==="
  if [ ! -f "$M" ]; then log "SKIP $tag (file absent)"; continue; fi
  if ! start_serve "$M" "$PORT"; then log "SKIP $tag (serve failed)"; PORT=$((PORT+1)); continue; fi

  a=$(chat "$PORT" "$RET_PROMPT" 32);  r_ret=$(chk_contains "$a" '7741')
  printf '## %s — retrieval\n%s\n' "$tag" "$a" >"$OUT/${tag}_retrieval.md"
  a=$(chat "$PORT" "Return ONLY a JSON object with keys name and age for: Alice is 30 years old." 48); r_json=$(chk_json "$a")
  printf '## %s — json\n%s\n' "$tag" "$a" >"$OUT/${tag}_json.md"
  a=$(chat "$PORT" "What is 17 multiplied by 23? Answer with the number only." 24); r_math=$(chk_contains "$a" '391')
  printf '## %s — math\n%s\n' "$tag" "$a" >"$OUT/${tag}_math.md"
  a=$(chat "$PORT" "List exactly three colors, one per line, nothing else." 32); r_instr=$(chk_lines3 "$a")
  printf '## %s — instruction\n%s\n' "$tag" "$a" >"$OUT/${tag}_instruction.md"
  a=$(chat "$PORT" "Translate 'thank you very much' into Spanish. Answer with the translation only." 24); r_multi=$(chk_contains "$a" 'gracias')
  printf '## %s — multilingual\n%s\n' "$tag" "$a" >"$OUT/${tag}_multilingual.md"

  total=$(( r_ret + r_json + r_math + r_instr + r_multi ))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$r_ret" "$r_json" "$r_math" "$r_instr" "$r_multi" "$total" >>"$TSV"
  log "$tag passed $total/5 (ret=$r_ret json=$r_json math=$r_math instr=$r_instr multi=$r_multi)"
  stop_serve
  PORT=$((PORT+1))
done

# comparison table (models as rows; 1->✅ 0->❌)
{
  echo "| model | retrieval | json | math | instruction | multilingual | total |"
  echo "|---|---|---|---|---|---|---|"
  awk -F'\t' '{
    for(i=2;i<=6;i++){ $i=($i=="1")?"✅":"❌" }
    printf "| %s | %s | %s | %s | %s | %s | %s/5 |\n",$1,$2,$3,$4,$5,$6,$7
  }' "$TSV"
  echo
  echo "## Verdict"
  echo "- VALID instruct eval (chat-templated via \`/v1/chat/completions\`), unlike the raw-\`generate\` suite."
  echo "- Per-class answers: \`$OUT/<model>_<class>.md\`. A class where RWKV ❌ but Qwen ✅ → route to Qwen (R5 / ssm_model_selection.md)."
  echo "- RWKV-7-0.4B is a small model — read the answers, not just the marks. This characterizes quality; thresholds for a"
  echo "  hard routing gate are an owner decision."
} >>"$SUMMARY"
log "chat quality suite complete -> $SUMMARY"
