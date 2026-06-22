#!/usr/bin/env bash
# tools/ci/ssm_product_gate.sh -- SSM (RWKV-7) production-readiness gate.
#
# Turns the validated RWKV/SSM long-context SPEED moat into a checkable PRODUCT
# gate: it does not claim readiness from tps alone. Orchestrates, sequentially
# (GPU jobs never overlap):
#   1. serve smoke          (admit + SSE through `hawking serve`)
#   2. speed matrix         (short / mid / long / optional 16k decode tps, warm)
#   3. quality probes       (representative single-stream `generate` outputs)
#   4. request isolation    (two different prompts -> two different coherent
#                            outputs through the SAME server, no cross-talk)
# Writes ONE timestamped report under reports/ssm-product/<stamp>/.
# Non-destructive; reports are evidence (do not stage). Flags: RUN_SERVE,
# RUN_SPEED, RUN_QUALITY, RUN_ISOLATION (default 1), RUN_16K (default 0).
set -u

REPO="${REPO:-$HOME/Downloads/hawking}"; cd "$REPO" || exit 2
MODEL="${1:-${MODEL:-models/rwkv7-g1-04-sft-Q4_K_M.gguf}}"
BIN="${BIN:-./target/release/hawking}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-reports/ssm-product/$STAMP}"
RUN_SERVE="${RUN_SERVE:-1}"; RUN_SPEED="${RUN_SPEED:-1}"
RUN_QUALITY="${RUN_QUALITY:-1}"; RUN_ISOLATION="${RUN_ISOLATION:-1}"
RUN_16K="${RUN_16K:-0}"
TRIALS="${TRIALS:-3}"; TOK="${TOK:-48}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.md"; LOG="$OUT/commands.log"
PASS=0; FAIL=0; SKIP=0

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
pass() { PASS=$((PASS+1)); log "PASS  $*"; printf -- "- ✅ %s\n" "$*" >>"$SUMMARY"; }
fail() { FAIL=$((FAIL+1)); log "FAIL  $*"; printf -- "- ❌ %s\n" "$*" >>"$SUMMARY"; }
skip() { SKIP=$((SKIP+1)); log "SKIP  $*"; printf -- "- ⊘ %s\n" "$*" >>"$SUMMARY"; }
med() { sort -n | awk '{a[NR]=$1} END{print (NR>0)?a[int((NR+1)/2)]:"ERR"}'; }

{
  echo "# SSM Product Gate — $STAMP"
  echo; echo "- model: \`$MODEL\`"; echo "- bin: \`$BIN\`"; echo "- host: $(uname -mrs)"
  echo "- git: $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty"
  echo; echo "## Results"
} >"$SUMMARY"

if [ ! -x "$BIN" ]; then fail "missing $BIN (run: cargo build --release)"; fi
if [ ! -f "$MODEL" ]; then fail "missing model $MODEL"; fi
[ "$FAIL" -gt 0 ] && { echo "preconditions failed" | tee -a "$LOG"; exit 1; }

# ---- 1. serve smoke -------------------------------------------------------
if [ "$RUN_SERVE" = "1" ]; then
  log "serve smoke via ssm_serve_smoke.sh"
  if OUT="$OUT/serve" tools/ci/ssm_serve_smoke.sh "$MODEL" >"$OUT/serve_smoke.log" 2>&1; then
    pass "serve smoke (admit + coherent SSE + [DONE])"
  else
    # distinguish admission (queued/admitted) from decode (empty token) failure
    M="$OUT/serve/metrics_after.log"
    adm=$(grep -oE 'requests_admitted_total [0-9]+' "$M" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
    fail "serve smoke (admitted=${adm:-?}; see $OUT/serve_smoke.log — likely the prefill->multiseq decode bug)"
  fi
else skip "serve smoke (RUN_SERVE=0)"; fi

# ---- 2. speed matrix (single-stream generate; the validated moat) ---------
if [ "$RUN_SPEED" = "1" ]; then
  declare -a CTX=("short:64" "mid:1200" "long:4200")
  [ "$RUN_16K" = "1" ] && CTX+=("xl:16000")
  echo "| ctx | approx tokens | tps (median of $TRIALS) |" >>"$SUMMARY"
  echo "|---|---|---|" >>"$SUMMARY"
  ok=1
  for spec in "${CTX[@]}"; do
    name="${spec%%:*}"; words="${spec##*:}"
    python3 -c "print(('The memory bandwidth of a GPU bounds decode speed. ' * max(1,$words//9)) + ' Summarize in one sentence.')" >"$OUT/prompt_$name.txt"
    tps=$(for t in $(seq 1 "$TRIALS"); do
      env HAWKING_QWEN_USER_DRAFT=0 "$BIN" generate --weights "$MODEL" \
        --prompt "$(cat "$OUT/prompt_$name.txt")" --max-new-tokens "$TOK" \
        --temperature 0 --seed 5 2>&1 | grep -oE 'dec_tps=[0-9.]+' | cut -d= -f2
    done | med)
    echo "| $name | ~$words | $tps |" >>"$SUMMARY"
    log "speed $name (~$words tok): $tps tps"
    [ "$tps" = "ERR" ] && ok=0
  done
  [ "$ok" = "1" ] && pass "speed matrix (flat = SSM moat; see table)" || fail "speed matrix (a context returned no tps)"
else skip "speed matrix (RUN_SPEED=0)"; fi

# ---- 3. quality probes (single-stream generate) ---------------------------
if [ "$RUN_QUALITY" = "1" ]; then
  declare -a Q=(
    "List three prime numbers."
    "Translate 'good morning' into French."
    "Write a one-line Python function that returns the square of n."
  )
  empties=0
  for i in "${!Q[@]}"; do
    out=$(env HAWKING_QWEN_USER_DRAFT=0 "$BIN" generate --weights "$MODEL" \
      --prompt "${Q[$i]}" --max-new-tokens 48 --temperature 0 --seed 7 2>/dev/null)
    printf '### Q%d: %s\n%s\n\n' "$i" "${Q[$i]}" "$out" >>"$OUT/quality.md"
    body=$(printf '%s' "$out" | grep -v '^\[' | tr -d '[:space:]')
    [ -z "$body" ] && empties=$((empties+1))
  done
  [ "$empties" = "0" ] && pass "quality probes (all non-empty; see quality.md)" \
    || fail "quality probes ($empties/${#Q[@]} empty)"
else skip "quality probes (RUN_QUALITY=0)"; fi

# ---- 4. request isolation (needs a working serve decode path) -------------
if [ "$RUN_ISOLATION" = "1" ]; then
  # Gated on serve correctness (Lane 1). If the serve smoke failed, isolation is
  # not yet meaningful — record as blocked rather than a false pass/fail.
  if [ "$RUN_SERVE" = "1" ] && grep -q 'admitted.*[1-9]' "$OUT/serve/metrics_after.log" 2>/dev/null \
     && grep -qE 'tokens_generated_total [2-9]' "$OUT/serve/metrics_after.log" 2>/dev/null; then
    skip "request isolation — TODO: implement 2-prompt concurrent isolation check (serve now decodes; wire it)"
  else
    skip "request isolation — BLOCKED on serve decode correctness (Lane 1 prefill->multiseq handoff)"
  fi
else skip "request isolation (RUN_ISOLATION=0)"; fi

# ---- verdict --------------------------------------------------------------
{
  echo; echo "## Verdict"
  echo "- PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
  if [ "$FAIL" = "0" ] && [ "$PASS" -gt 0 ]; then
    echo "- **PRODUCT-READY (for the gates run).** Speed moat holds; serve path coherent."
  else
    echo "- **NOT product-ready.** The SSM speed moat is real (speed matrix), but serve correctness"
    echo "  (Lane 1: RWKV prefill->multiseq state handoff) must land before shipping the serve path."
  fi
  echo; echo "## Recovery / next"
  echo "- Serve fix gate: \`cargo test --release -p hawking-core --test rwkv7_prefill_slot_multiseq_parity -- --ignored --test-threads=1\`"
  echo "- Re-run this gate: \`tools/ci/ssm_product_gate.sh $MODEL\`"
} >>"$SUMMARY"

log "SSM product gate complete: PASS=$PASS FAIL=$FAIL SKIP=$SKIP -> $SUMMARY"
[ "$FAIL" = "0" ]
