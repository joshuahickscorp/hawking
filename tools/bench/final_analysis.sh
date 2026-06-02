#!/usr/bin/env bash
# =============================================================================
# tools/bench/final_analysis.sh — ONE command for the lever analysis.
#
# DEFAULT MODE is CONTAMINATION-ROBUST — it works with Claude OPEN, because it
# only uses paired/relative metrics (A/B ratios + back-to-back J/tok), where the
# ~4-5x session inflation CANCELS. No Claude-quit required, no long-context
# hang. This is the mode to use day-to-day.
#
#   tools/bench/final_analysis.sh            # robust (Claude open OK) — DEFAULT
#   tools/bench/final_analysis.sh --clean    # ALSO run absolute anchor (QUIT Claude)
#   tools/bench/final_analysis.sh --diag     # ALSO run the "is there more tps?" diagnostics
#
# DEFAULT runs (~6-12 min):
#   A. --profile fast + residency  paired A/B   (the banked tps win + held lever; short ctx)
#   B. measured J/tok (zeus per-domain) + f16-KV relative @1024 (energy: measured + footprint check)
#   C. f16-KV + flash  paired A/B --long-ctx    (long-context tps behaviour)
#   D. quality (SHORT) f16-scales + f16-KV      (token drift; no slow long tier)
#
# --clean additionally runs tools/bench/clean_room_batch.sh (absolute tps/J/tok
#   + Q3 §A) — meaningless unless Claude is fully QUIT; it self-gates and will
#   refuse if Claude.app is running.
#
# --diag additionally runs the settle-the-ceiling diagnostics (one-time, slow):
#   gpu_saturation.sh (kernel-bound vs idle — verdict: kernel-bound, saturation DEAD)
#   mlx_ab.sh         (dismantle vs MLX on this machine — is the 1.6x gap a runtime ceiling)
#   These answer "is more tps reachable" — already run 2026-06-02; re-run to re-confirm.
#
# Everything tee's to reports/bench/final_analysis_<ts>.log.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

CLEAN=0
DIAG=0
for a in "$@"; do
  case "$a" in
    --clean) CLEAN=1 ;;
    --diag)  DIAG=1 ;;
  esac
done

TS=$(date +%Y%m%dT%H%M%S)
LOG="reports/bench/final_analysis_${TS}.log"
mkdir -p reports/bench reports/quality
exec > >(tee "$LOG") 2>&1
banner() { printf '\n\n=================== %s ===================\n' "$1"; }

echo "dismantle final analysis — $(date)"
echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null)"
echo "mode:   $([[ $CLEAN == 1 ]] && echo 'CLEAN (absolute — QUIT Claude!)' || echo 'ROBUST (paired/relative — Claude open OK)')$([[ $DIAG == 1 ]] && echo ' +DIAG')"
echo "log:    $LOG"

banner "build (idempotent)"
cargo build --release --workspace 2>&1 | tail -2

if [[ "$CLEAN" == 1 ]]; then
  banner "0  ABSOLUTE ANCHOR  (clean_room_batch — needs Claude QUIT)"
  tools/bench/clean_room_batch.sh || echo "[note] clean_room_batch self-gated or non-zero (read above)"
fi

banner "A  --profile fast  paired A/B  (the banked tps win)"
tools/bench/ab_lever.sh --cli-b "--profile fast" || echo "[note] ab_lever profile-fast non-zero"
echo "------ residency (held lever; expect ~neutral on 3B) ------"
tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_RESIDENCY || echo "[note] ab_lever residency non-zero"

banner "B  ENERGY  (measured per-domain J/tok + f16-KV footprint check)"
echo "------ MEASURED per-domain J/tok (zeus IOReport energy model) ------"
ZEUS_PY="$(command -v python3)"; [[ -x /tmp/mlxenv/bin/python ]] && ZEUS_PY=/tmp/mlxenv/bin/python
"$ZEUS_PY" tools/bench/zeus_joules.py --tokens 64 2>&1 | tail -18 \
  || echo "[note] zeus_joules non-zero (needs zeus-apple-silicon; see header)"
echo "------ f16-KV relative J/tok @1024 (off vs on — is f16-KV an energy lever?) ------"
echo "  baseline (f16-KV OFF):"
tools/bench/phase_joules.sh --tokens 1024 || echo "[note] phase_joules baseline non-zero"
echo "  f16-KV ON:"
DISMANTLE_QWEN_F16_KV=1 tools/bench/phase_joules.sh --tokens 1024 || echo "[note] phase_joules f16-KV non-zero"

banner "C  f16-KV + flash  paired A/B --long-ctx  (long-context tps)"
tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_F16_KV   --long-ctx || echo "[note] ab_lever f16-KV non-zero"
tools/bench/ab_lever.sh --lever DISMANTLE_QWEN_FLASH_ATTN --long-ctx || echo "[note] ab_lever flash non-zero"

banner "D  QUALITY (SHORT tier — fast, no long hang)"
tools/bench/quality_oracle.sh --lever DISMANTLE_QWEN_PREDEC_F16SCALES --label f16scales --short-only || echo "[note] quality f16scales returned non-zero (FAIL/WARN is data, not an error)"
tools/bench/quality_oracle.sh --lever DISMANTLE_QWEN_F16_KV --label f16kv --short-only || echo "[note] quality f16kv returned non-zero"

if [[ "$DIAG" == 1 ]]; then
  banner "DIAG-1  GPU-SATURATION TRACE  (kernel-bound vs inter-dispatch idle)"
  tools/bench/gpu_saturation.sh || echo "[note] gpu_saturation non-zero"
  banner "DIAG-2  MLX CEILING A/B  (dismantle vs MLX on this machine — is the 1.6x gap a runtime ceiling)"
  MLX_PYTHON="${MLX_PYTHON:-/tmp/mlxenv/bin/python}" tools/bench/mlx_ab.sh \
    || echo "[note] mlx_ab non-zero (needs mlx-lm; MLX_PYTHON=path/to/venv/bin/python)"
fi

sleep 1
banner "SUMMARY (best-effort — full numbers in the sections above)"
echo "--- tps ratios / J/tok ---"
grep -hiE "B/A=|J/token|J/tok|dec_tps|GPU power|inconclusive|gain|regression" "$LOG" \
  | grep -viE "warning|note:|help:|^#|echo" | head -40 || true
echo "--- quality ---"
grep -hiE "token_identical|corpus_drift|TIER VERDICT|OVERALL|identical\)" "$LOG" \
  | grep -viE "warning|note:" | head -20 || true
echo
echo "DONE. log: $LOG"
echo "READS:"
echo "  A  --profile fast B/A   -> the banked tps win (expect ~+5%); residency B/A ~1.0 (held)"
echo "  B  zeus mJ/tok          -> MEASURED GPU/DRAM/CPU per-domain energy (clean GPU anchor ~0.197 J/tok)"
echo "     f16-KV J/tok on<off? -> energy lever; on>=off -> footprint-only (current: footprint-only)"
echo "  C  f16-KV/flash B/A     -> long-ctx tps (expect f16-KV neutral/neg; flash neutral)"
echo "  D  drift%               -> --profile fast / f16-KV quality cost (logit-cosine needs a logit-export feature)"
if [[ "$DIAG" == 1 ]]; then
echo "  DIAG-1 saturation       -> kernel-bound vs idle (verdict 2026-06-02: KERNEL-BOUND, saturation DEAD)"
echo "  DIAG-2 MLX B/A          -> dismantle/MLX on this box; settles if the 1.6x gap is a runtime ceiling"
fi
[[ "$CLEAN" == 0 ]] && echo "  (run with --clean + Claude QUIT for absolute tps/J/tok; --diag for the ceiling diagnostics)"
