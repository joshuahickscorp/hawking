#!/usr/bin/env bash
# =============================================================================
# tools/bench/gpu_saturation.sh — pin WHERE the ~24% non-GPU decode gap lives.
#
# CONTEXT
# =======
# Baseline: ~30.5 tps / 0.197 J/tok (M3 Pro, Qwen-3B-Q4_K_M).
# llama.cpp: ~49 tps same machine.  Gap = 1.6x.
# The 0.2 trace analysis showed GPU-busy ≈ 76% of decode wall → ~24% idle.
# RESEARCH VERDICT (reports/research_next_levers_2026_06_02.md): the gap is in
# the runtime/GPU-saturation layer, not bytes or kernel compute.  The question
# this script answers: is the 24% gap INTER-DISPATCH IDLE (a saturation /
# command-buffer lever, alive) or is it TOKEN-TO-TOKEN scheduling overhead
# (CPU forward-pass time between tokens)?
#
# METHOD (two-pass)
# =================
# Pass A — PRODUCTION baseline (no trace overhead):
#   Run bench --suite decode --max-new-tokens 32 WITHOUT HAWKING_TCB_TRACE.
#   This gives decode_tps_production → the true per-token wall time.
#   decode_wall_per_token_us = 1e6 / decode_tps_production
#
# Pass B — SplitCbGpu trace:
#   Run bench --suite decode --max-new-tokens 32 WITH HAWKING_TCB_TRACE=gpu
#   (SplitCbGpu mode: each dispatch in its own CB with gpuStartTime/gpuEndTime).
#   Wall semantics in SplitCbGpu:
#     wall_us = CPU pipeline-lookup + encode + end_encoding time ONLY
#               (measured before commit(); NOT including commit/wait round-trip).
#     gpu_us  = gpuStartTime → gpuEndTime (real GPU kernel execution time).
#   The commit()+wait_until_completed() overhead per dispatch is NOT captured
#   in either field — it is the per-kernel CB scheduling cost.
#   In PRODUCTION mode (single CB per token), this per-dispatch overhead is
#   AMORTIZED: one commit/wait per token for all ~616 kernels.  So the
#   SplitCbGpu inter-dispatch overhead is NOT a production bottleneck.
#
# GAP DECOMPOSITION
# =================
# GPU-busy fraction (correct formula):
#   gpu_busy_frac = per_token_gpu_us / decode_wall_per_token_us
#   where per_token_gpu_us = Σ(gpu_us for all traced dispatches) / traced_tokens.
#   NOTE: analyze_tcb_trace.py's built-in busy_frac formula is inaccurate when
#   completion_tokens != traced_tokens (it divides per_token_gpu_us by total
#   decode_ms instead of per-token decode_us, giving ~0.5% rather than ~66%).
#   This script computes the correct formula independently.
#
# Non-GPU gap = 1 - gpu_busy_frac, decomposed as:
#   1. Per-token CPU encode time = Σ(wall_us) / traced_tokens
#      (pipeline lookup + kernel encoding overhead per token; typically ~2 ms).
#   2. Inter-token scheduling gap = decode_wall_per_token - per_token_gpu_us
#                                    - per_token_encode_us
#      (CPU forward-pass overhead between tokens: sample decode, state update,
#       next-token input prep, CB encode, commit; the 'race-to-idle' gap).
#
# VERDICT FRAMING
# ===============
# If inter-token gap >> 2 ms: gap is TOKEN-TO-TOKEN scheduling overhead.
#   → The race-to-idle / command-buffer scheduling lever is ALIVE.
#   → Reducing CPU overhead between token commits closes the gap.
#   → NOT intra-token inter-dispatch idle (that's near-zero in single-CB mode).
# If inter-token gap ≈ 0 and gpu_busy_frac ≈ 100%: kernel-bound.
#   → Only faster kernels help; scheduling not the lever.
#
# CONTAMINATION NOTE
# ==================
# SplitCbGpu gpu_us absolute values are contaminated by any open GPU workload
# (e.g. the coding agent app).  However, PER-KERNEL SHARES (% of total gpu_us) are
# ~robust: contamination affects all kernels proportionally.
# The gpu_busy_frac uses production TPS (also contaminated in the same ratio),
# so the FRACTION is approximately contamination-robust.
# Running with the agent open inflates absolute numbers ~4-5x but the RATIOS hold.
# For clean absolute tps numbers, use tools/bench/clean_bench.sh.
#
# USAGE
#   tools/bench/gpu_saturation.sh
#   TOKENS=64 tools/bench/gpu_saturation.sh    # more tokens (slower, same verdict)
#   WEIGHTS=models/... PROFILE=profiles/... tools/bench/gpu_saturation.sh
#
# OUTPUT
#   Stdout: gap decomposition report + per-kernel GPU share table + VERDICT.
#   Trace JSON: /tmp/gpu_sat_trace_<timestamp>.json (kept for follow-on analysis).
#   Analyze JSON: /tmp/gpu_sat_analyze_<timestamp>.json
# =============================================================================
set -uo pipefail

_agent_env="$(git rev-parse --show-toplevel 2>/dev/null)/.agent_env"
[ -f "$_agent_env" ] && source "$_agent_env"
unset _agent_env

cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/hawking}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
TOKENS="${TOKENS:-32}"

# Locked Qwen fast-path (matches clean_room_batch.sh / ab_lever.sh / measure_joules.sh).
BASE_ENV="HAWKING_QWEN_TCB=1 HAWKING_QWEN_VOCAB_PRUNE=32000 \
HAWKING_QWEN_Q4K_LMHEAD=1 HAWKING_QWEN_FFN_DOWN_Q4K=1 \
HAWKING_QWEN_Q4K_PREDEC=1"

TS=$(date +%Y%m%dT%H%M%S)
TRACE_JSON="/tmp/gpu_sat_trace_${TS}.json"
ANALYZE_JSON="/tmp/gpu_sat_analyze_${TS}.json"
PROD_JSON="/tmp/gpu_sat_prod_${TS}.json"

PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"

hr()  { printf '\n=================== %s ===================\n' "$1"; }
die() { printf 'error: %s\n' "$*" >&2; exit 64; }

[[ -x "$BIN" ]]     || die "$BIN not built (cargo build --release --workspace?)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"

if pgrep -f "${AGENT_APP_PGREP:?see .agent_env.example}" >/dev/null 2>&1; then
    printf '\n[note] the agent app is running — absolute gpu_us values are contaminated\n'
    printf '       PER-KERNEL SHARES and the GAP FRACTION are ~robust (contamination cancels).\n'
    printf '       For clean absolute numbers: quit the agent and re-run.\n'
fi

# ---------------------------------------------------------------------------
# PASS A — PRODUCTION TPS (no trace overhead)
# ---------------------------------------------------------------------------
hr "PASS A: production decode_tps (no trace)"
printf 'weights  : %s\n' "$WEIGHTS"
printf 'profile  : %s\n' "$PROFILE"
printf 'tokens   : %s\n' "$TOKENS"
printf 'running production pass (no HAWKING_TCB_TRACE)...\n'

set +e
env $BASE_ENV nice -n 19 taskpolicy -b "$BIN" bench \
    --backend hawking --suite decode \
    --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
    --trials 1 --max-new-tokens "$TOKENS" \
    --json "$PROD_JSON" \
  >/dev/null 2>&1
PROD_RC=$?
set -e

if [[ $PROD_RC -ne 0 || ! -f "$PROD_JSON" ]]; then
    die "production bench failed (exit=$PROD_RC); check $BIN and $WEIGHTS"
fi

DECODE_TPS_PROD=$(jq -r '(.results.decode_tps // .results.trial_stats[0].decode_tps // 0)' \
    "$PROD_JSON" 2>/dev/null || echo 0)
if [[ $(awk -v t="$DECODE_TPS_PROD" 'BEGIN{print (t+0 < 1.0) ? 1 : 0}') -eq 1 ]]; then
    die "production bench returned implausible decode_tps=$DECODE_TPS_PROD"
fi
printf 'decode_tps_production = %s tps\n' "$DECODE_TPS_PROD"
printf 'per_token_wall_us     = %.1f us (= 1e6 / tps)\n' \
    "$(awk -v t="$DECODE_TPS_PROD" 'BEGIN{printf "%.1f", 1e6/t}')"

# ---------------------------------------------------------------------------
# PASS B — SplitCbGpu trace
# ---------------------------------------------------------------------------
hr "PASS B: SplitCbGpu trace (HAWKING_TCB_TRACE=gpu)"
printf 'trace output : %s\n' "$TRACE_JSON"
printf 'SplitCbGpu mode: each dispatch in its own CB; gpuStartTime/gpuEndTime read\n'
printf 'per kernel.  wall_us = encode-only (before commit); gpu_us = real GPU kernel time.\n'
printf 'This run is ~%dx slower than production (one commit/wait per dispatch).\n' \
    "$(awk -v t="$TOKENS" 'BEGIN{printf "%d", 616}')"  # ~616 dispatches/token
printf 'running trace pass...\n'

set +e
env $BASE_ENV HAWKING_TCB_TRACE=gpu HAWKING_TRACE_DISPATCH=1 \
  nice -n 19 taskpolicy -b "$BIN" bench \
    --backend hawking --suite decode \
    --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
    --trials 1 --max-new-tokens "$TOKENS" \
    --trace-json "$TRACE_JSON" \
    --trace-dispatch \
  >/dev/null 2>&1
TRACE_RC=$?
set -e

if [[ $TRACE_RC -ne 0 || ! -f "$TRACE_JSON" ]]; then
    die "trace bench failed (exit=$TRACE_RC). Check that HAWKING_TCB_TRACE=gpu is\n"\
        "supported (requires macos + Metal; the trace JSON must be non-empty)."
fi
printf 'trace JSON written: %s\n' "$TRACE_JSON"

# ---------------------------------------------------------------------------
# PASS C — Feed to analyze_tcb_trace.py for per-kernel breakdown
# ---------------------------------------------------------------------------
hr "PASS C: analyze_tcb_trace.py --json"
ANALYZE_PY="tools/bench/analyze_tcb_trace.py"
[[ -f "$ANALYZE_PY" ]] || die "$ANALYZE_PY not found (expected at repo root)"

set +e
"$PY" "$ANALYZE_PY" --json --model qwen3b --no-gate \
    "$TRACE_JSON" > "$ANALYZE_JSON" 2>/dev/null
ANALYZE_RC=$?
set -e

if [[ $ANALYZE_RC -ne 0 || ! -s "$ANALYZE_JSON" ]]; then
    printf '[warn] analyze_tcb_trace.py returned non-zero (rc=%d); continuing with raw JSON.\n' \
        "$ANALYZE_RC"
fi

# ---------------------------------------------------------------------------
# PASS D — Gap decomposition + verdict (inline Python3)
# ---------------------------------------------------------------------------
hr "GAP DECOMPOSITION + VERDICT"

"$PY" - "$TRACE_JSON" "$ANALYZE_JSON" "$DECODE_TPS_PROD" "$TOKENS" <<'PYEOF'
import json, sys, collections, pathlib

trace_path   = pathlib.Path(sys.argv[1])
analyze_path = pathlib.Path(sys.argv[2])
decode_tps_prod = float(sys.argv[3])
tokens_arg   = int(sys.argv[4])

# ── Load trace JSON (the --trace-json output from lib.rs) ─────────────────
try:
    doc = json.loads(trace_path.read_text())
except Exception as e:
    sys.exit(f"error reading trace JSON: {e}")

# ── Find dispatch_samples ─────────────────────────────────────────────────
# analyze_tcb_trace walks the whole doc; we look at the top-level key first
# (lib.rs writes dispatch_samples at the trace root for --trace-json output).
def find_samples(obj, depth=0):
    if depth > 6:
        return None
    if isinstance(obj, dict):
        if "dispatch_samples" in obj and isinstance(obj["dispatch_samples"], list):
            return obj["dispatch_samples"]
        for v in obj.values():
            r = find_samples(v, depth+1)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_samples(v, depth+1)
            if r:
                return r
    return None

samples = find_samples(doc)
if not samples:
    sys.exit("error: no dispatch_samples found in trace JSON")

# ── Token count: use sample_* dispatches (INV3) ───────────────────────────
traced_tokens = sum(1 for s in samples if s.get("kernel_name", "").startswith("sample_"))
if traced_tokens == 0:
    # Fallback: use tokens_arg (less accurate; note it)
    traced_tokens = tokens_arg
    print(f"[warn] no sample_* dispatches found; using --max-new-tokens={tokens_arg} as token count.")
    print(f"       Per-token math may be off. Re-run with HAWKING_QWEN_TCB=1 set.")
else:
    if traced_tokens != tokens_arg:
        print(f"[note] trace covers {traced_tokens} tokens (sample_* count), not {tokens_arg}."
              f" Using {traced_tokens} for per-token math.")

# ── Aggregate by kernel ───────────────────────────────────────────────────
by_k = collections.defaultdict(lambda: {"n": 0, "gpu_us": 0, "wall_us": 0})
for s in samples:
    k = s.get("kernel_name", "other")
    by_k[k]["n"]       += 1
    by_k[k]["gpu_us"]  += s.get("gpu_us")  or 0
    by_k[k]["wall_us"] += s.get("wall_us") or 0

total_gpu_us  = sum(v["gpu_us"]  for v in by_k.values())
total_wall_us = sum(v["wall_us"] for v in by_k.values())
n_with_gpu    = sum(1 for s in samples if s.get("gpu_us") is not None)

if total_gpu_us == 0:
    sys.exit("error: total_gpu_us == 0. Is HAWKING_TCB_TRACE=gpu set? "
             "Run with HAWKING_TCB_TRACE=gpu to get per-kernel GPU times.")

per_token_gpu_us    = total_gpu_us  / traced_tokens
per_token_wall_us   = total_wall_us / traced_tokens   # encode-only (SplitCbGpu)
per_token_wall_prod = 1e6 / decode_tps_prod           # production wall time

# ── GPU busy fraction (correct formula — uses production TPS wall) ─────────
# analyze_tcb_trace.py's built-in busy_frac = (gpu_us/tokens) / (dec_ms*1000)
# which divides per_token_gpu_us by the TOTAL trial decode_ms, giving ~0.5%
# rather than ~66%.  We compute correctly here.
gpu_busy_frac   = per_token_gpu_us / per_token_wall_prod
non_gpu_frac    = 1.0 - gpu_busy_frac

# ── Gap decomposition ─────────────────────────────────────────────────────
# In SplitCbGpu mode, wall_us = encode-only (before commit).
# In production (single CB per token), encode time ≈ per_token_encode_us.
# The inter-token gap = production_wall - gpu_busy - encode_overhead.
#
# SplitCbGpu commit/wait overhead per dispatch:
#   In production, one commit+wait per TOKEN amortizes the ~100-500 us/CB overhead.
#   In SplitCbGpu, EVERY dispatch has its own CB → overhead inflates total decode
#   ~8-20x vs production (visible in the SplitCbGpu decode_tps being much lower).
#   This overhead is NOT in wall_us or gpu_us fields → we cannot isolate it directly.
#   But in PRODUCTION mode it is ONE commit/wait per token (≈100-500 us).
per_token_encode_us  = per_token_wall_us    # CPU encode of all kernels per token
inter_token_gap_us   = per_token_wall_prod - per_token_gpu_us - per_token_encode_us
# Clamp to 0 if slightly negative due to measurement noise
inter_token_gap_us   = max(inter_token_gap_us, 0.0)

f_gpu     = gpu_busy_frac
f_encode  = per_token_encode_us / per_token_wall_prod
f_intertok = inter_token_gap_us / per_token_wall_prod

# ── Per-kernel table ──────────────────────────────────────────────────────
print()
print("=== Per-kernel GPU time breakdown (SplitCbGpu gpu_us — shares are ~robust) ===")
print(f"{'kernel':48s} {'n/tok':>7s}  {'us/call':>8s}  {'us/tok':>8s}  {'% GPU':>7s}  {'% wall':>7s}")
print("-" * 92)
rows = sorted(by_k.items(), key=lambda kv: -kv[1]["gpu_us"])
for k, v in rows:
    if v["gpu_us"] == 0 and v["n"] == 0:
        continue
    pct_gpu  = v["gpu_us"] / total_gpu_us * 100 if total_gpu_us else 0
    pct_wall = v["gpu_us"] / traced_tokens / per_token_wall_prod * 100
    n_per_tok = v["n"] / traced_tokens
    us_call   = v["gpu_us"] / max(v["n"], 1)
    us_tok    = v["gpu_us"] / traced_tokens
    flag = "  <- UNMAPPED" if k == "other" else ""
    print(f"{k:48s} {n_per_tok:7.1f}  {us_call:8.1f}  {us_tok:8.0f}  {pct_gpu:7.2f}%  {pct_wall:7.2f}%{flag}")

# ── Bandwidth estimate ─────────────────────────────────────────────────────
QWEN3B_BYTES_PER_TOKEN = int(1.93 * 1024**3)
M3_PRO_PEAK_GBPS = 150.0
per_token_s = per_token_gpu_us / 1e6
eff_gbps = QWEN3B_BYTES_PER_TOKEN / per_token_s / 1024**3 if per_token_s > 0 else 0

# ── Summary table ─────────────────────────────────────────────────────────
print()
print("=== Gap decomposition (production wall = 1/decode_tps_prod) ===")
print(f"  decode_tps_production       : {decode_tps_prod:.2f} tps")
print(f"  per_token_wall_prod         : {per_token_wall_prod/1000:.3f} ms  (= 1 / {decode_tps_prod:.2f} tps)")
print(f"  per_token_gpu_us (SplitCb)  : {per_token_gpu_us/1000:.3f} ms  (Σkernel gpu_us / {traced_tokens} tokens)")
print(f"  per_token_encode_us (wall)  : {per_token_encode_us/1000:.3f} ms  (CPU encode overhead/token)")
print(f"  inter_token_gap_us          : {inter_token_gap_us/1000:.3f} ms  (wall - gpu - encode)")
print()
print(f"  GPU-busy fraction           : {f_gpu*100:.1f}%  (gpu_us / production_wall)")
print(f"  CPU encode fraction         : {f_encode*100:.1f}%  (encode_us / production_wall)")
print(f"  inter-token gap fraction    : {f_intertok*100:.1f}%  (idle between token commits)")
print(f"  ─────────────────────────────────────────────────────")
print(f"  check (should sum to ~100%) : {(f_gpu+f_encode+f_intertok)*100:.1f}%")
print()
print(f"  effective GPU bandwidth     : {eff_gbps:.1f} GiB/s ({eff_gbps/M3_PRO_PEAK_GBPS*100:.0f}% of {M3_PRO_PEAK_GBPS:.0f} GiB/s peak)")
print(f"  samples traced              : {len(samples)}  ({n_with_gpu} with gpu_us)")

# ── Methodology note ──────────────────────────────────────────────────────
print()
print("[methodology note]")
print("  SplitCbGpu inflates per-dispatch overhead: each kernel runs in its own CB.")
print("  In PRODUCTION (single CB per token): all ~616 kernels run sequentially in")
print("  one CB with NO intra-token inter-dispatch idle.  The GPU is continuously")
print("  busy from the first to the last kernel within each token.")
print("  → The intra-token inter-dispatch idle contribution is near ZERO in production.")
print("  → The SplitCbGpu wall_us/gpu_us gap (per dispatch) is NOT a production lever.")
print("  The per-kernel GPU SHARES reported above are robust across modes.")
print("  The inter-token gap (above) is the production scheduling overhead — the lever.")

# ── VERDICT ───────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  VERDICT: is the ~24% non-GPU gap INTER-DISPATCH IDLE or KERNEL-BOUND?")
print("=" * 70)

if inter_token_gap_us / 1000 > 2.0:          # gap > 2 ms = real signal
    gap_pct_of_tgt = (inter_token_gap_us / per_token_wall_prod) * 100
    llama_wall = 1e6 / 49.0                  # llama.cpp reference
    closing_tps = 1e6 / (per_token_wall_prod - inter_token_gap_us)
    print()
    print(f"  INTER-TOKEN SCHEDULING GAP detected: {inter_token_gap_us/1000:.1f} ms / token ({gap_pct_of_tgt:.1f}% of wall).")
    print()
    print("  This gap is TOKEN-TO-TOKEN CPU overhead (CPU forward pass between tokens):")
    print("    sample decode → state update → next-token input prep → CB encode → commit.")
    print("  It is NOT intra-token inter-dispatch idle (GPU is fully busy within each token")
    print("  because all kernels share one CB in production mode — no intra-CB idle).")
    print()
    print("  SATURATION LEVER STATUS: ALIVE.")
    print(f"  Closing this gap could push tps from {decode_tps_prod:.1f} → {closing_tps:.1f} tps")
    print(f"  (upper bound: fully closing the {inter_token_gap_us/1000:.1f} ms inter-token gap).")
    print(f"  llama.cpp reference: ~49 tps ({llama_wall/1000:.1f} ms/token wall).")
    print()
    print("  NEXT BUILD CANDIDATE: reduce inter-token CPU overhead.")
    print("  Levers (examine in order):")
    print("    1. MTLResidencySet — wire weight buffers so OS cannot evict/throttle them")
    print("       between tokens (llama.cpp PR #11427, macOS>=15). Low effort, A/B first.")
    print("    2. Pre-encode the next token's CB while GPU is running the current one")
    print("       (double-buffered CB pipeline). Requires CB dependency tracking.")
    print("    3. Profile the CPU forward pass time with Instruments Time Profiler to")
    print("       locate the dominant non-GPU hotspot (sample decode / rope / state update).")
elif f_gpu > 0.95:
    print()
    print(f"  KERNEL-BOUND: GPU-busy = {f_gpu*100:.1f}%, inter-token gap = {inter_token_gap_us/1000:.1f} ms.")
    print("  The saturation lever is DEAD for this configuration.")
    print("  Only faster kernels (higher BW utilization) will close the gap to llama.cpp.")
    print(f"  Current effective BW: {eff_gbps:.1f} GiB/s ({eff_gbps/M3_PRO_PEAK_GBPS*100:.0f}% of peak).")
    print("  Next step: vectorized uint4 nibble load in gemm_q4_k_v4_predec_pair (A5).")
else:
    print()
    print(f"  AMBIGUOUS: GPU-busy = {f_gpu*100:.1f}%, inter-token gap = {inter_token_gap_us/1000:.1f} ms.")
    print("  Gap exists but is smaller than expected. Check for measurement noise:")
    print("   - Was the agent app running? Contamination degrades absolute values.")
    print("   - Use TOKENS=64 or TOKENS=128 for more stable per-token averages.")
    print("   - Compare with A4 baseline: per_token_gpu_us should be ~21.7 ms for Qwen-3B.")

print("=" * 70)
print()
print(f"trace JSON : {sys.argv[1]}")
print(f"analyze JSON : {sys.argv[2]}  (pass to analyze_tcb_trace.py for full kernel table)")
print("re-analyze:")
print(f"  python3 tools/bench/analyze_tcb_trace.py {sys.argv[1]}")
PYEOF

printf '\n[done] trace: %s\n' "$TRACE_JSON"
printf '       analyze: %s\n' "$ANALYZE_JSON"

