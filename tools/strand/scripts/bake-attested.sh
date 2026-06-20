#!/bin/bash
# bake-attested.sh — bake the first TRAINED-2-BIT .strand artifact and attest it.
#
# The dismantle-integration rehearsal: trained PV shadows -> per-tensor STRAND
# trellis quant (canon 2-bit config: --bits 2 --l 12 --outlier-channel 1) ->
# packed .strand v2 (STR2) archive **+ OUTL outlier section + SPRV v2
# self-verifying provenance trailer** (the honest-artifact cluster: the archive
# now decodes to the SAME weights as the recon path — the pre-O1 bake silently
# dropped the outlier channel) -> recon safetensors for the byte-equality gate
# -> verification stanza (mmap load via the strand-decode-kernel loader,
# 3-tensor decode spot check with the fixed==lean determinism gate, SPRV
# self-verify, archive==recon byte gate, SPV3 provenance model_root, and the
# audit-O7 billing headline: file_len / n_weights).
#
# ROOT-STABILITY NOTE (measured 2026-06-11 re-bake): the model_root is
# UNCHANGED vs the pre-O1 bake (6e1b0e4e…be54) — correct by construction. The
# root hashes the Q12 bulk plane only, and the encode is deterministic; the R2
# binding lives in SPRV v2's per-record descriptor digests (checked by
# verify_archive), NOT in the root preimage. What changed is the FILE: it now
# carries the OUTL outlier channel (+~14 MB) and the SPRV v2 trailer, so the
# archive finally decodes to the 26.77-PPL weights the root's lineage claims.
#
# Output:  scratch/artifacts/qwen05b-pv2-2bit.strand  (+ .quant.log / .attest.log
#          / .attestation.txt next to it; recon safetensors kept only with KEEP_RECON=1)
#
# Rules respected:
#  - waits for any running strand-delta measurement job before the 12-thread quant
#  - builds are nice -n 10 -j 4 (light, allowed anytime)
#  - STRAND_NO_GPU=1 (the Metal encode path SIGKILLs on wide tensors; CPU SIMD
#    encode beats the serialized GPU ~8x anyway)
#  - --ragged-v2: the 0.5B has 896-dim tensors (not 256-aligned) -> STRICT v2
#    rejects them; RAGGED v2 is the correct archive flavor for this model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/scratch/artifacts"
ARCHIVE="$OUT_DIR/qwen05b-pv2-2bit.strand"
RECON="$OUT_DIR/qwen05b-pv2-2bit.recon.safetensors"
THREADS="${THREADS:-12}"
OLD_MODEL_ROOT="6e1b0e4e36a5aa53ade2b95d372a56eeb96d6572457f50d96e93950db401be54"

# Shadow source: the best trained-2-bit shadows (pv2, PPL 26.77 post-requant).
# Fallback: the pv arm (27.02) if pv2 is missing.
SHADOWS="$ROOT/scratch/qwen-05b/qat-pv2-hf/model.safetensors"
LINEAGE="qat-pv2-hf (PV 300+600 steps, PPL 26.77 post-requant)"
if [[ ! -f "$SHADOWS" ]]; then
  SHADOWS="$ROOT/scratch/qwen-05b/qat-pv-hf/model.safetensors"
  LINEAGE="qat-pv-hf (PV 300 steps, PPL 27.02 post-requant) [pv2 missing - fallback]"
fi
[[ -f "$SHADOWS" ]] || { echo "FATAL: no trained shadow safetensors found" >&2; exit 1; }

mkdir -p "$OUT_DIR"
echo "[bake] shadows : $SHADOWS"
echo "[bake] lineage : $LINEAGE"
echo "[bake] archive : $ARCHIVE"
echo "[bake] machine : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fGB", $1/1e9}') RAM, load=$(uptime | awk -F'load averages?: ' '{print $2}')"

# ── 1. light builds (allowed while measurement jobs run) ─────────────────────
echo "[bake] building quantize-model + attest-strand (nice, -j4)..."
( cd "$ROOT" && nice -n 10 cargo build --release -j 4 \
    -p strand-quant --bin quantize-model )
( cd "$ROOT" && nice -n 10 cargo build --release -j 4 \
    -p strand-decode-kernel --bin attest-strand )
QM="$ROOT/target/release/quantize-model"
ATTEST="$ROOT/target/release/attest-strand"

# ── 2. wait for any strand-delta measurement job (12-thread quant is heavy) ──
# pgrep -x (exact process name): the -f form false-matches OTHER watcher shells
# whose command line merely mentions "strand-delta".
while pgrep -x strand-delta >/dev/null 2>&1; do
  echo "[bake] strand-delta measurement job running — waiting 60s ($(date +%H:%M:%S))"
  sleep 60
done
# If a rung-screen sweep grabbed the box after the delta drained, drop to the
# light tier (4 threads, nice) instead of stomping its 12 threads.
NICE_PREFIX=""
if pgrep -f rung-screen.py >/dev/null 2>&1; then
  THREADS=4
  NICE_PREFIX="nice -n 10"
  echo "[bake] rung-screen.py holds the box — light tier: $THREADS threads, nice 10"
fi
echo "[bake] starting the quant ($THREADS threads)"

# ── 3. per-tensor quant -> packed .strand v2 + OUTL + SPRV (automatic) ───────
QUANT_LOG="$ARCHIVE.quant.log"
STRAND_NO_GPU=1 $NICE_PREFIX "$QM" \
  --in "$SHADOWS" \
  --bits 2 --l 12 --outlier-channel 1 \
  --packed-v2-out "$ARCHIVE" --ragged-v2 \
  --threads "$THREADS" 2>&1 | tee "$QUANT_LOG"
[[ -s "$ARCHIVE" ]] || { echo "FATAL: archive not written" >&2; exit 1; }
grep -q 'appended OUTL section' "$QUANT_LOG" || { echo "FATAL: OUTL section missing from bake" >&2; exit 1; }
grep -q 'appended SPRV trailer' "$QUANT_LOG" || { echo "FATAL: SPRV trailer missing from bake" >&2; exit 1; }

# ── 3b. recon safetensors (same shadows, same config) for the O1 byte gate ───
RECON_LOG="$ARCHIVE.recon.log"
STRAND_NO_GPU=1 $NICE_PREFIX "$QM" \
  --in "$SHADOWS" \
  --bits 2 --l 12 --outlier-channel 1 \
  --out "$RECON" \
  --threads "$THREADS" 2>&1 | tee "$RECON_LOG"
[[ -s "$RECON" ]] || { echo "FATAL: recon safetensors not written" >&2; exit 1; }

# ── 4. verification stanza: mmap load + determinism spot check + SPRV
#       self-verify + archive==recon byte gate + provenance root + billing ────
ATTEST_LOG="$ARCHIVE.attest.log"
"$ATTEST" "$ARCHIVE" --roots --recon-check "$RECON" 2>&1 | tee "$ATTEST_LOG"
grep -q 'RECON-CHECK      PASS' "$ATTEST_LOG" || { echo "FATAL: archive-vs-recon byte gate FAILED" >&2; exit 1; }

# recon is a ~2GB intermediate; keep only on request.
if [[ "${KEEP_RECON:-0}" != "1" ]]; then
  rm -f "$RECON" "$RECON.json"
  echo "[bake] recon safetensors removed (KEEP_RECON=1 to keep)"
fi

# ── 5. attestation summary ───────────────────────────────────────────────────
SIZE_BYTES=$(stat -f %z "$ARCHIVE" 2>/dev/null || stat -c %s "$ARCHIVE")
MODEL_ROOT=$(grep '^model_root ' "$ATTEST_LOG" | awk '{print $2}')
BILLING_LINE=$(grep '^BILLING' "$ATTEST_LOG" | head -1)
OUTL_LINE=$(grep '^outliers' "$ATTEST_LOG" | head -1)
SPRV_LINE=$(grep '^sprv' "$ATTEST_LOG" | head -1)
WRITER_LINE=$(grep 'bytes/weight' "$QUANT_LOG" | tail -1 || true)

SUMMARY="$OUT_DIR/qwen05b-pv2-2bit.attestation.txt"
{
  echo "STRAND first trained-2-bit artifact — attestation (honest-artifact re-bake)"
  echo "date              $(date '+%Y-%m-%d %H:%M:%S')"
  echo "shadows           $SHADOWS"
  echo "lineage           $LINEAGE"
  echo "quant config      --bits 2 --l 12 --outlier-channel 1 (STRAND_NO_GPU=1, ragged-v2)"
  echo "archive           $ARCHIVE (STR2 + OUTL + SPRV v2)"
  echo "archive bytes     $SIZE_BYTES"
  echo "$BILLING_LINE"
  echo "$OUTL_LINE"
  echo "$SPRV_LINE"
  echo "recon byte gate   PASS (archive-only patched decode == recon path, all tensors)"
  echo "writer line       $WRITER_LINE"
  echo "model_root        $MODEL_ROOT"
  if [[ "$MODEL_ROOT" == "$OLD_MODEL_ROOT" ]]; then
    echo "model_root check  UNCHANGED vs pre-O1 bake (expected: root = Q12 plane; R2 binding is per-record)"
  else
    echo "model_root (OLD)  $OLD_MODEL_ROOT  [ROOT CHANGED — encode drift?! investigate before shipping]"
  fi
} | tee "$SUMMARY"
echo "[bake] DONE — attestation at $SUMMARY"
