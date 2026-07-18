#!/usr/bin/env bash
# =============================================================================
# tools/bench/mst_diff.sh — the single-stream 1.6x-gap DECIDER.
#
# Captures a Metal System Trace of BOTH engines on the SAME
# model/prompt/seed/temp0/256-tok run — dismantle (labeled gpu_prod encoders)
# and llama-cli — then diffs them per-kernel: dispatch-count/token, GPU-busy
# fraction, per-kernel GPU-us/call, the inter-dispatch gap distribution, and
# commit->GPU-start latency. The output is the named, cheap oracle for the only
# surviving single-stream reframe (adverse prior): does llama's dominant GEMV
# run FASTER per call than dismantle's at equal bytes?  See the decision tree
# at the bottom of this file and in mst_gap.py.
#
# WHY a trace and not gpu_saturation.sh: the "24% idle" framing is DEAD
# (SplitCbGpu artifact; production decode = ONE command buffer per token =
# ~0.0 ms intra-token inter-dispatch idle). gpu_saturation.sh already showed
# that. What remains is a per-kernel GPU-us/call DIFF vs llama.cpp for the
# dominant GEMV — that requires a real MST on BOTH engines, which is THIS tool.
#
# CONTAMINATION: absolute GPU-us are meaningless with another GPU workload
# open. This script HARD-ABORTS if the agent app is running. RUN IT AGENT-QUIT.
#
# Usage:
#   tools/bench/mst_diff.sh                 # Qwen-3B-Q4_K_M defaults, 256 tok
#   TOKENS=128 tools/bench/mst_diff.sh
#   WEIGHTS=models/... PROFILE=profiles/... PROMPT="..." tools/bench/mst_diff.sh
#   GEMV_MATCH="q4_K,mul_mv,mul_mm,gemm_q4" tools/bench/mst_diff.sh
#
# Env:
#   BIN        dismantle binary (default ./target/release/hawking)
#   LLAMA      llama-cli binary (default: $(command -v llama-cli))
#   WEIGHTS    gguf (default models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   PROFILE    dismantle kernel profile (default qwen3b-instruct-q4k.m3pro18.json)
#   PROMPT     decode prompt (default a fixed sentence; same string to both)
#   TOKENS     tokens to decode (default 256)
#   GEMV_MATCH comma-list of GEMV name substrings (passed to mst_gap.py)
#   OUT_DIR    report dir (default reports/)
#
# Output: reports/mst_diff_<stamp>.md (side-by-side), plus the two .trace
# bundles and exported XML under traces/mst_diff_<stamp>/.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

_agent_env="$(git rev-parse --show-toplevel 2>/dev/null)/.agent_env"
[ -f "$_agent_env" ] && source "$_agent_env"
unset _agent_env

# --- knobs -----------------------------------------------------------------
BIN="${BIN:-./target/release/hawking}"
LLAMA="${LLAMA:-$(command -v llama-cli || true)}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
PROMPT="${PROMPT:-Explain how a CPU pipeline hazard is resolved by forwarding.}"
TOKENS="${TOKENS:-256}"
SEED="${SEED:-42}"
GEMV_MATCH="${GEMV_MATCH:-q4_k,q4_K,gemv,mul_mv,mul_mm,gemm_q4}"
OUT_DIR="${OUT_DIR:-reports}"
EXPORT_SH="tools/bench/mst_export.sh"
GAP_PY="tools/bench/mst_gap.py"

# Locked Qwen fast-path (matches gpu_saturation.sh / clean_room_batch.sh /
# measure_joules.sh) + gpu_prod tracer so dismantle's encoders are LABELED in
# the trace (matches its per-kernel names to the right XML rows).
BASE_ENV="HAWKING_QWEN_TCB=1 HAWKING_QWEN_VOCAB_PRUNE=32000 \
HAWKING_QWEN_Q4K_LMHEAD=1 HAWKING_QWEN_FFN_DOWN_Q4K=1 \
HAWKING_QWEN_Q4K_PREDEC=1 HAWKING_TCB_TRACE=gpu_prod"

PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"
STAMP="$(date +%Y%m%dT%H%M%S)"
TDIR="traces/mst_diff_${STAMP}"
DM_TRACE="$TDIR/dismantle.trace"
LL_TRACE="$TDIR/llama.trace"
REPORT="$OUT_DIR/mst_diff_${STAMP}.md"

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '%s\n' "$*"; }
hr()   { printf '\n=================== %s ===================\n' "$1"; }

# ===========================================================================
# STAGE 0 — PREFLIGHT (hard aborts)
# ===========================================================================
hr "STAGE 0: preflight"

# 0a. agent app => HARD ABORT (absolute GPU-us would be contaminated).
if pgrep -f "${AGENT_APP_PGREP:?see .agent_env.example}" >/dev/null 2>&1; then
    die "The agent app is running. Absolute GPU-us in an MST are contaminated 4-5x.
       Cmd+Q the agent and re-run this script. (This is the single check that makes
       the per-call GPU-us DIFF trustworthy — it is not optional.)"
fi
note "[ok] agent app not running"

# 0b. xcrun present.
command -v xcrun >/dev/null 2>&1 \
    || die "xcrun not found — MST needs macOS + Xcode (not just CLT)."
note "[ok] xcrun present"

# 0c. 'Metal System Trace' template present. CAPTURE then grep (never pipe
#     'xctrace list templates' through grep -q: with pipefail, grep's early
#     exit SIGPIPEs xctrace and falsely fails). Same idiom as mst_capture.sh.
_tmpl="$(xcrun xctrace list templates 2>/dev/null || true)"
printf '%s\n' "$_tmpl" | grep -q "Metal System Trace" \
    || die "xctrace 'Metal System Trace' template missing — install full Xcode."
note "[ok] 'Metal System Trace' template available"

# 0d. dismantle binary.
[[ -x "$BIN" ]] || die "$BIN not built (cargo build --release --workspace)."
note "[ok] dismantle binary: $BIN"

# 0e. llama-cli on PATH.
[[ -n "$LLAMA" && -x "$LLAMA" ]] \
    || die "llama-cli not found. Install: brew install llama.cpp (sets /opt/homebrew/bin/llama-cli)."
note "[ok] llama-cli: $LLAMA"

# 0f. model + profile present.
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"
[[ -f "$PROFILE" ]] || die "profile not found: $PROFILE"
note "[ok] weights:  $WEIGHTS"
note "[ok] profile:  $PROFILE"

# 0g. export + parser tooling.
[[ -x "$EXPORT_SH" || -f "$EXPORT_SH" ]] || die "$EXPORT_SH missing"
[[ -f "$GAP_PY" ]] || die "$GAP_PY missing"
note "[ok] export: $EXPORT_SH   parser: $GAP_PY"

mkdir -p "$TDIR" "$OUT_DIR"
note ""
note "run config:  tokens=$TOKENS seed=$SEED"
note "prompt    :  $PROMPT"
note "gemv match:  $GEMV_MATCH"
note "trace dir :  $TDIR"
note "report    :  $REPORT"

# ===========================================================================
# STAGE 1 — dismantle MST capture (gpu_prod labeled encoders, fast path)
# ===========================================================================
hr "STAGE 1: dismantle MST capture (HAWKING_TCB_TRACE=gpu_prod)"
# We trace `dismantle bench --suite decode`; the locked env + gpu_prod tracer
# label every compute encoder so the XML kernel-name column carries the real
# names (gemm_q4_k_v4_predec_pair, etc.). nice+taskpolicy keep it cooperative.
# xctrace --launch needs an ABSOLUTE first executable (no PATH lookup), so lead
# with /usr/bin/env — it sets BASE_ENV and PATH-resolves nice/taskpolicy/$BIN.
DM_CMD=(/usr/bin/env $BASE_ENV nice -n 19 taskpolicy -b "$BIN" bench
        --backend hawking --suite decode
        --weights "$WEIGHTS" --kernel-profile "$PROFILE"
        --trials 1 --max-new-tokens "$TOKENS")
note "command: ${DM_CMD[*]}"
# macOS 26.x xctrace can return a non-zero exit through the env+nice+taskpolicy
# launch chain EVEN WHEN the trace bundle is captured fine (confirmed: the bench
# it launches exits 0 with valid stats, and the .trace bundle is fully populated).
# The real success signal is "did the bundle get written?", not xctrace's code.
xcrun xctrace record --template "Metal System Trace" --output "$DM_TRACE" \
    --launch -- "${DM_CMD[@]}" \
    || note "[warn] xctrace returned non-zero (macOS 26.x launch-chain quirk) — checking for the bundle anyway."
if [[ -d "$DM_TRACE" ]]; then
    note "[ok] dismantle trace: $DM_TRACE"
else
    die "dismantle MST record failed — no trace bundle at $DM_TRACE (see above)."
fi

# ===========================================================================
# STAGE 2 — llama-cli MST capture (same model/prompt/seed/temp0/256 tok)
# ===========================================================================
hr "STAGE 2: llama-cli MST capture"
# -ngl 99: all layers on GPU (Metal). -n TOKENS, --temp 0 greedy, --seed.
# -no-cnv: raw completion (no chat template) to mirror the bench decode loop.
# -p: same prompt string. Metal is the default backend in Homebrew llama.cpp.
LL_CMD=(/usr/bin/env nice -n 19 taskpolicy -b "$LLAMA"
        -m "$WEIGHTS" -p "$PROMPT" -n "$TOKENS"
        --temp 0 --seed "$SEED" -ngl 99 -no-cnv)
note "command: ${LL_CMD[*]}"
LL_OK=1
xcrun xctrace record --template "Metal System Trace" --output "$LL_TRACE" \
    --launch -- "${LL_CMD[@]}" \
    || { note "[warn] llama-cli MST record failed (llama-cli flag drift? see above). Proceeding DISMANTLE-ONLY so the expensive dismantle trace is not wasted."; LL_OK=0; }
[[ "$LL_OK" == 1 ]] && note "[ok] llama trace: $LL_TRACE"

# ===========================================================================
# STAGE 3 — export both via mst_export.sh, locate the GPU-interval table
# ===========================================================================
hr "STAGE 3: export both traces to XML"

# mst_export.sh dumps a TOC + exports each GPU-interval-candidate schema into
# <trace>_export/<schema>.xml. We then pick the exported XML with the most
# <row>s as the per-kernel GPU-interval table to diff.
export_and_pick() {
    local trace="$1" label="$2"
    local edir="${trace%.trace}_export"
    OUT_DIR="$edir" bash "$EXPORT_SH" "$trace" >/dev/null 2>&1 \
        || { note "[warn] export failed for $label ($trace)"; echo ""; return; }
    # biggest exported xml (most rows) = the GPU interval table, skip toc.xml
    local best="" bestlines=0 f lines
    for f in "$edir"/*.xml; do
        [[ -e "$f" ]] || continue
        [[ "$(basename "$f")" == "toc.xml" ]] && continue
        lines=$(wc -l < "$f" 2>/dev/null || echo 0)
        if [[ "$lines" -gt "$bestlines" ]]; then bestlines="$lines"; best="$f"; fi
    done
    echo "$best"
}

DM_XML="$(export_and_pick "$DM_TRACE" dismantle)"
[[ -n "$DM_XML" && -f "$DM_XML" ]] || die "no dismantle GPU-interval XML exported (inspect $TDIR/dismantle_export/toc.xml)."
note "[ok] dismantle XML: $DM_XML"
LL_XML=""
if [[ "$LL_OK" == 1 ]]; then
  LL_XML="$(export_and_pick "$LL_TRACE" llama)"
  if [[ -n "$LL_XML" && -f "$LL_XML" ]]; then
    note "[ok] llama XML    : $LL_XML"
  else
    note "[warn] no llama GPU-interval XML exported (inspect $TDIR/llama_export/toc.xml); DISMANTLE-ONLY."
    LL_XML=""
  fi
fi
note ""
note "[hint] if the diff's columns look wrong, inspect them:"
note "       $PY $GAP_PY --inspect $DM_XML"
note "       $PY $GAP_PY --inspect $LL_XML"

# ===========================================================================
# STAGE 4 — diff + emit side-by-side report
# ===========================================================================
hr "STAGE 4: per-kernel diff + report"

if [[ -n "$LL_XML" && -f "$LL_XML" ]]; then
  DIFF_TXT="$($PY "$GAP_PY" --dismantle "$DM_XML" --llama "$LL_XML" \
              --tokens "$TOKENS" --gemv-match "$GEMV_MATCH" 2>&1)"
  DIFF_JSON="$($PY "$GAP_PY" --dismantle "$DM_XML" --llama "$LL_XML" \
              --tokens "$TOKENS" --gemv-match "$GEMV_MATCH" --json 2>/dev/null || echo '{}')"
else
  DIFF_TXT="DISMANTLE-ONLY (llama capture/export unavailable). The dismantle trace is saved at $DM_TRACE — open it in Instruments, or re-run mst_diff.sh for the llama diff. Per-kernel dismantle GPU-us: $PY $GAP_PY --inspect $DM_XML"
  DIFF_JSON='{}'
fi

# The report is plain markdown; the human reads it agent-quit and pastes the
# verdict line into the kill-ledger / closeout.
{
  echo "# MST diff — dismantle vs llama.cpp (single-stream 1.6x-gap decider)"
  echo
  echo "- stamp: \`$STAMP\`"
  echo "- model: \`$WEIGHTS\`"
  echo "- profile: \`$PROFILE\`"
  echo "- prompt: \`$PROMPT\`"
  echo "- tokens: $TOKENS   seed: $SEED   temp: 0 (greedy)"
  echo "- dismantle env: \`$BASE_ENV\`"
  echo "- llama cmd: \`${LL_CMD[*]}\`"
  echo "- traces: \`$DM_TRACE\` , \`$LL_TRACE\`"
  echo "- XML: \`$DM_XML\` , \`$LL_XML\`"
  echo
  echo "## Side-by-side diff + verdict"
  echo
  echo '```'
  printf '%s\n' "$DIFF_TXT"
  echo '```'
  echo
  echo "## Raw JSON (per-kernel, gap distribution, GEMV per-call)"
  echo
  echo '```json'
  printf '%s\n' "$DIFF_JSON"
  echo '```'
  echo
  echo "## Decision tree"
  echo
  echo "- per-kernel GPU-us/call, dominant GEMV: **llama < dismantle** (>8%) =>"
  echo "  Type-2 reframe ALIVE — a faster dismantle GEMV closes part of the 1.6x."
  echo "- **~equal** (within 8%) at ~equal GiB/s/call => single-stream decode-tps"
  echo "  is a CLOSED Type-1 kill (gap is dispatch-rate/per-token-overhead, not the"
  echo "  GEMV; production already shows 0.0 ms intra-token inter-dispatch idle)."
  echo "- **dismantle < llama** => GEMV already faster; the gap is whole-pipeline"
  echo "  (dispatch count/token, non-GEMV work) — look at the dispatch/token line."
} > "$REPORT"

note ""
note "$DIFF_TXT"
note ""
note "[done] report: $REPORT"
note "       traces: $DM_TRACE , $LL_TRACE"
