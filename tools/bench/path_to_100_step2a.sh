#!/usr/bin/env bash
# path-to-100 Step 2A — eagle4 K=1 tax allocation harness.
#
# Background (reports/path_to_90/plans/path_to_100_repath.md §Step 2A):
#   Clean-window bench 2026-05-20 shows eagle4 / sequential / K=1 is
#   18.01 dec_tps vs off=26.87 → 8.9-tps tax with ZERO acceptance
#   benefit (K=1 is bit-identical to off by construction). The tax is
#   pure capture + head propose + head argmax overhead. This harness
#   isolates which of the four sub-steps owns the cost:
#
#     (a) capture forward    — forward_token_argmax with eagle4_capture_active
#     (b) h_shared compute   — either GPU read or cpu_shared_expert_forward(26)
#     (c) head propose       — forward_full_{amx,metal,cpu}_no_lm_head
#     (d) head argmax        — gemv_f16_argmax_dispatch
#
# Methodology:
#   Phase 1 — median dec_tps across 4 configs × 3 prompts × 3 trials:
#     1. off            (baseline control)
#     2. eagle4 K=1 / AMX backend     (production macOS default)
#     3. eagle4 K=1 / Metal backend   (EAGLE4_BACKEND=metal)
#     4. eagle4 K=1 / CPU backend     (EAGLE4_BACKEND=cpu)
#
#   Phase 2 — single-prompt DISMANTLE_SPEC_LOG=1 run per eagle4 config,
#   captures per-step [spec/eagle4-step] log lines with per-phase µs
#   breakdown emitted from deepseek_v2.rs.
#
# Allocation matrix (which sub-step is implicated if dominant):
#   - capture_us large in ALL backends → (a) capture forward is the tax
#   - hshared_us large + hshared_fallback=true → (b) GPU shared expert
#       capture is broken; CPU fallback owns the cost
#   - head_us much larger in metal vs amx → (c) head propose Metal
#       dispatch is the cost; AMX dormant kernel would already cover it
#   - head_us large in all backends → (c) head's compute is the cost
#       (lever B chain-pipelining is the implementation answer)
#   - argmax_us large → (d) gemv_f16_argmax_dispatch is the cost (L5
#       Lever A becomes the wiring target)
#
# REQUIRES: Claude.app must be quit (Cmd-Q). Contended GPU produces
# 4-5× inflated dec_tps (per memory bench_contamination); the script
# refuses to run if pgrep finds Claude alive.
#
# Outputs:
#   reports/path_to_90/_bench_step2a_<TS>/
#     raw.jsonl       — per-trial dec_tps records
#     summary.txt     — human-readable medians + allocation hints
#     spec_log_<cfg>.txt — DISMANTLE_SPEC_LOG=1 step traces (one per backend)
#     step_breakdown.csv — parsed step_us per-phase (post-Phase-2)

set -euo pipefail

usage() {
  cat <<'EOF'
path_to_100_step2a.sh — eagle4 K=1 tax allocation harness

Usage:
  ./tools/bench/path_to_100_step2a.sh           # full run (Phase 1 + Phase 2)
  ./tools/bench/path_to_100_step2a.sh --help

Configs measured:
  1. off / sequential / K=1                       (baseline control)
  2. eagle4 / sequential / K=1 / EAGLE4_BACKEND=amx     (production default)
  3. eagle4 / sequential / K=1 / EAGLE4_BACKEND=metal
  4. eagle4 / sequential / K=1 / EAGLE4_BACKEND=cpu

Prerequisites:
  - Claude.app quit (Cmd-Q)
  - cargo build --release --workspace done
  - models/deepseek-v2-lite-q4.gguf present
  - eagle4/v2lite_frozen.npz + eagle4/checkpoints/eagle4_v3/best.npz present

Outputs land under reports/path_to_90/_bench_step2a_<timestamp>/.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if pgrep -i "Claude" >/dev/null 2>&1; then
  echo "ERROR: Claude is still running. Quit Claude.app (Cmd-Q) before benching." >&2
  echo "       Contended GPU produces 4-5x inflated dec_tps — useless data." >&2
  exit 2
fi

if pgrep -f "slm" >/dev/null 2>&1; then
  echo "WARN: slm process detected — pause it (kill -STOP <pid>) or wait until idle." >&2
  echo "      Sleeping 30s for any background activity to drain..." >&2
  sleep 30
fi

CUR=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")
if [[ "$CUR" -lt 14336 ]]; then
  echo "WARN: iogpu.wired_limit_mb=$CUR; recommend 14336 for sustained bench:" >&2
  echo "  sudo sysctl iogpu.wired_limit_mb=14336" >&2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

WEIGHTS="$REPO_ROOT/models/deepseek-v2-lite-q4.gguf"
PROFILE_SEQ="$REPO_ROOT/profiles/deepseek-v2-lite-q4.m3pro18.json"
FROZEN_NPZ="$REPO_ROOT/eagle4/v2lite_frozen.npz"
DRAFT_NPZ="${EAGLE4_CKPT:-$REPO_ROOT/eagle4/checkpoints/eagle4_v3/best.npz}"
DISMANTLE="$REPO_ROOT/target/release/dismantle"

if [[ ! -x "$DISMANTLE" ]]; then
  echo "ERROR: $DISMANTLE missing. Build first: cargo build --release --workspace" >&2
  exit 3
fi
for f in "$WEIGHTS" "$PROFILE_SEQ" "$FROZEN_NPZ" "$DRAFT_NPZ"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing artifact: $f" >&2
    exit 4
  fi
done

TS="$(date +%Y%m%dT%H%M%S)"
OUTDIR="$REPO_ROOT/reports/path_to_90/_bench_step2a_${TS}"
mkdir -p "$OUTDIR"
RAW="$OUTDIR/raw.jsonl"
SUMMARY="$OUTDIR/summary.txt"
STEP_CSV="$OUTDIR/step_breakdown.csv"

PROMPTS=(
  "The quick brown fox"
  "Write a Python function to compute Fibonacci numbers"
  "Summarize the plot of Hamlet in three sentences"
)
TOKENS=64
TRIALS=3

# Phase 1 — dec_tps median collection.
# Args: mode backend prompt trial
run_trial() {
  local mode="$1"
  local backend="$2"      # "" | metal | cpu | amx
  local prompt="$3"
  local trial="$4"

  local out
  if [[ "$mode" == "eagle4" ]]; then
    local env_prefix=()
    case "$backend" in
      metal) env_prefix=(env EAGLE4_BACKEND=metal) ;;
      cpu)   env_prefix=(env EAGLE4_BACKEND=cpu)   ;;
      amx)   env_prefix=(env -u EAGLE4_BACKEND)    ;;
    esac
    out=$("${env_prefix[@]}" nice -n 19 "$DISMANTLE" generate \
            --weights "$WEIGHTS" --kernel-profile "$PROFILE_SEQ" \
            --prompt "$prompt" --max-new-tokens "$TOKENS" --temperature 0 \
            --speculate eagle4 --draft-head "$DRAFT_NPZ" --eagle4-frozen "$FROZEN_NPZ" \
            2>&1)
  else
    out=$(nice -n 19 "$DISMANTLE" generate \
            --weights "$WEIGHTS" --kernel-profile "$PROFILE_SEQ" \
            --prompt "$prompt" --max-new-tokens "$TOKENS" --temperature 0 \
            2>&1)
  fi

  local dec_tps
  dec_tps=$(echo "$out" | grep -oE 'dec_tps=[0-9.]+' | head -1 | cut -d= -f2)
  dec_tps="${dec_tps:-0}"

  printf '{"mode":"%s","backend":"%s","prompt":%s,"trial":%d,"dec_tps":%s}\n' \
    "$mode" "$backend" \
    "$(printf '%s' "$prompt" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')" \
    "$trial" "$dec_tps"
}

# Phase 2 — single spec_log capture per eagle4 backend.
# Args: backend outfile
spec_log_capture() {
  local backend="$1"
  local outfile="$2"
  # macOS env requires options (-u) BEFORE name=value pairs — anything
  # after the first name=value is treated as the utility to exec. The
  # AMX run silently produced `env: -u: No such file or directory`
  # under the older ordering. Keep -u flags first; name=values second.
  local env_prefix=(env DISMANTLE_SPEC_LOG=1)
  case "$backend" in
    metal) env_prefix=(env DISMANTLE_SPEC_LOG=1 EAGLE4_BACKEND=metal) ;;
    cpu)   env_prefix=(env DISMANTLE_SPEC_LOG=1 EAGLE4_BACKEND=cpu)   ;;
    amx)   env_prefix=(env -u EAGLE4_BACKEND DISMANTLE_SPEC_LOG=1)    ;;
  esac
  "${env_prefix[@]}" nice -n 19 "$DISMANTLE" generate \
    --weights "$WEIGHTS" --kernel-profile "$PROFILE_SEQ" \
    --prompt "${PROMPTS[0]}" --max-new-tokens 32 --temperature 0 \
    --speculate eagle4 --draft-head "$DRAFT_NPZ" --eagle4-frozen "$FROZEN_NPZ" \
    >"$outfile" 2>&1 || true
}

echo "=== path-to-100 Step 2A — eagle4 K=1 tax allocation @ $TS ===" | tee "$SUMMARY"
echo "draft head: $DRAFT_NPZ" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# Configs: (mode, backend_label, output_label)
CONFIGS=(
  "off       none   off"
  "eagle4    amx    eagle4-amx"
  "eagle4    metal  eagle4-metal"
  "eagle4    cpu    eagle4-cpu"
)

: > "$RAW"
echo "--- Phase 1 — median dec_tps (12 trials per config) ---" | tee -a "$SUMMARY"
for cfg in "${CONFIGS[@]}"; do
  read -r mode backend label <<<"$cfg"
  echo "" | tee -a "$SUMMARY"
  echo "[$label]" | tee -a "$SUMMARY"
  cnt=0
  for prompt in "${PROMPTS[@]}"; do
    for t in $(seq 1 $TRIALS); do
      backend_arg="$backend"
      [[ "$backend" == "none" ]] && backend_arg=""
      line=$(run_trial "$mode" "$backend_arg" "$prompt" "$t")
      echo "$line" >> "$RAW"
      dec_tps=$(echo "$line" | python3 -c 'import sys,json; print(json.load(sys.stdin)["dec_tps"])')
      cnt=$((cnt+1))
      printf '  trial=%d prompt=%-30s dec_tps=%s\n' \
        "$t" "$(echo "$prompt" | cut -c1-30)" "$dec_tps" | tee -a "$SUMMARY"
    done
  done
done

echo "" | tee -a "$SUMMARY"
echo "=== Phase 1 RESULT ===" | tee -a "$SUMMARY"
python3 - "$RAW" <<'PY' | tee -a "$SUMMARY"
import sys, json, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
groups = collections.defaultdict(list)
for r in rows:
    key = (r["mode"], r["backend"])
    groups[key].append(r["dec_tps"])
print("")
print("config                            median   mean    min     max     n")
for (mode, backend), xs in groups.items():
    xs = sorted(xs)
    med = xs[len(xs)//2]
    mean = sum(xs) / len(xs)
    label = f"{mode}/{backend or 'off'}/K1"
    print(f"{label:34s} {med:6.2f}  {mean:6.2f}  {min(xs):6.2f}  {max(xs):6.2f}  {len(xs)}")
PY

# Phase 2 — spec_log per backend.
echo "" | tee -a "$SUMMARY"
echo "--- Phase 2 — DISMANTLE_SPEC_LOG=1 per-step breakdown (1 prompt × 32 tokens) ---" | tee -a "$SUMMARY"
for backend in amx metal cpu; do
  LOG="$OUTDIR/spec_log_eagle4-${backend}.txt"
  echo "  capturing eagle4-${backend} -> ${LOG##*/}" | tee -a "$SUMMARY"
  spec_log_capture "$backend" "$LOG"
done

# Parse step lines into CSV.
echo "step,backend,capture_us,hshared_us,hshared_fallback,head_us,argmax_us,total_us" > "$STEP_CSV"
for backend in amx metal cpu; do
  LOG="$OUTDIR/spec_log_eagle4-${backend}.txt"
  grep -hE '\[spec/eagle4-step\]' "$LOG" 2>/dev/null | \
    sed -E 's/.*step=([0-9]+) backend=([^ ]+) capture_us=([0-9]+) hshared_us=([0-9]+) hshared_fallback=([a-z]+) head_us=([0-9]+) argmax_us=([0-9]+) total_us=([0-9]+).*/\1,\2,\3,\4,\5,\6,\7,\8/' \
    >> "$STEP_CSV" || true
done

# Per-backend medians of each phase.
echo "" | tee -a "$SUMMARY"
echo "=== Phase 2 RESULT — per-phase µs medians (Step 1+, warm) ===" | tee -a "$SUMMARY"
python3 - "$STEP_CSV" <<'PY' | tee -a "$SUMMARY"
import sys, csv, collections
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("(no step lines captured — instrumentation may not be active)")
    sys.exit(0)
by = collections.defaultdict(list)
for r in rows:
    if int(r["step"]) == 0:
        continue
    by[r["backend"]].append({
        "capture": int(r["capture_us"]),
        "hshared": int(r["hshared_us"]),
        "fallback": r["hshared_fallback"],
        "head": int(r["head_us"]),
        "argmax": int(r["argmax_us"]),
        "total": int(r["total_us"]),
    })
def med(xs):
    s = sorted(xs)
    return s[len(s)//2]
print("")
print(f"{'backend':10s} {'capture_us':>11s} {'hshared_us':>11s} {'head_us':>9s} {'argmax_us':>10s} {'total_us':>10s} {'fallback':>10s}")
for bk in ("amx", "metal", "cpu"):
    xs = by.get(bk, [])
    if not xs:
        print(f"{bk:10s} (no data)")
        continue
    fb = sum(1 for r in xs if r["fallback"] == "true")
    print(f"{bk:10s} "
          f"{med(r['capture'] for r in xs):11d} "
          f"{med(r['hshared'] for r in xs):11d} "
          f"{med(r['head'] for r in xs):9d} "
          f"{med(r['argmax'] for r in xs):10d} "
          f"{med(r['total'] for r in xs):10d} "
          f"{fb}/{len(xs):>4d}".rjust(10))
PY

echo "" | tee -a "$SUMMARY"
echo "=== outputs ===" | tee -a "$SUMMARY"
echo "raw:           $RAW"            | tee -a "$SUMMARY"
echo "step CSV:      $STEP_CSV"       | tee -a "$SUMMARY"
echo "spec logs:     $OUTDIR/spec_log_eagle4-*.txt" | tee -a "$SUMMARY"

if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "Step 2A bench complete" with title "dismantle"' || true
fi
echo ""
echo "Bench complete. See $SUMMARY"
