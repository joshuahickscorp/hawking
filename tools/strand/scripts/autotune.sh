#!/usr/bin/env bash
# autotune.sh — the idle autotuner ("the cerebellum"). v1, 2026-06-11.
#
# Runs the tunable sweep (tools/autotune/sweep.py) while the box is quiet, so the
# performance constants we hand-picked once (rayon decode threads, interleave S,
# requant --threads, ...) get re-learned per-machine by a machine. The tuner
# NEVER changes a default in code: its only output is research/tuned-profile.toml;
# consumers opt in via tools/autotune/apply.py.
#
# Contract (mirrors ops/replay.sh):
#   - idle-gated on the same pgrep discipline (science/sibling processes running
#     -> skip quietly, not an error); bracket-constructed patterns (self-match trap)
#   - at most one sweep per quiet window: scratch/.autotune-last stamped at launch,
#     6 h window (AUTOTUNE_FORCE=1 bypasses the window, never the idle gate)
#   - single-instance pid lock scratch/.autotune-lock
#   - degrades gracefully: sweep.py SKIPs any tunable whose gate bin is missing
#     (the sibling speed wave owns those bins and may have them mid-rebuild);
#     "nothing runnable" -> rc=2 here = quiet skip, no profile, no false PASS
#   - verdict -> scratch/.autotune-verdict; one JSONL record appended to
#     research/results-ledger.jsonl per COMPLETED sweep (harness_key=autotune,
#     no ppl field -- `ledger check` WARNs "record without ppl"; known + benign,
#     same as replay records)
#   - ALL timings in the profile are ADVISORY and machine-stamped (meta.advisory)
#   - exit 0 on PASS or quiet-skip; exit 1 on FAIL (a guard or invariance broke --
#     that is bit-rot-shaped news, not a tuning result)
set -u
cd "$(dirname "$0")/.."
LOCK=scratch/.autotune-lock
STAMP=scratch/.autotune-last
VF=scratch/.autotune-verdict
LEDGER=research/results-ledger.jsonl
PROFILE=research/tuned-profile.toml
WINDOW=$((6 * 3600))
log(){ echo "[autotune $(date '+%d %H:%M:%S')] $*"; }

# ── idle gate (same pgrep discipline as replay.sh / the conductor idle jobs) ───
# bracket-constructed patterns: the literal must not appear in our own cmdline.
if pgrep -f 'strand-qat[.]py|quantize-mode[l]|strand-7b-pp[l]|strand-act[2]|ops/repla[y].sh' >/dev/null 2>&1; then
    log "box busy (qat/quant/eval/replay running) — skipping, not an error"
    exit 0
fi
# single-instance pid lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    log "another autotune holds the lock ($(cat "$LOCK")) — skipping"
    exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

# ── one sweep per quiet window ─────────────────────────────────────────────────
if [ "${AUTOTUNE_FORCE:-0}" != 1 ] && [ -f "$STAMP" ]; then
    last=$(cat "$STAMP" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ $((now - last)) -lt "$WINDOW" ]; then
        log "swept $(( (now - last) / 60 )) min ago (< $((WINDOW / 3600))h window) — skipping"
        exit 0
    fi
fi
date +%s > "$STAMP"

# ── degrade check: sweep.py SKIPs per-tunable, but say it up front in the log ──
for bin in target/release/gate-decode-speed target/release/gate-interleave target/release/quantize-model; do
    [ -x "$bin" ] || log "note: $bin missing (sibling-wave rebuild?) — its tunables will SKIP"
done

# ── the sweep ──────────────────────────────────────────────────────────────────
PY=python3; command -v /usr/local/bin/python3 >/dev/null 2>&1 && PY=/usr/local/bin/python3
$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null || PY=python3
t0=$(date +%s)
out=$("$PY" tools/autotune/sweep.py --reps "${AUTOTUNE_REPS:-2}" --out "$PROFILE" 2>&1)
rc=$?
secs=$(( $(date +%s) - t0 ))
echo "$out" | tail -20

summary=$(echo "$out" | grep -o 'summary: .*' | tail -1 | cut -c10-)
case $rc in
    0) V=PASS;;
    2) log "nothing runnable (all tunables SKIP/DISABLED) — quiet skip, no ledger record"
       echo "SKIP :: nothing runnable (${secs}s)" > "$VF"
       exit 0;;
    *) V=FAIL;;
esac
GIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

# ── one ledger line per completed sweep (harness_key=autotune) ─────────────────
AT_VERDICT="$V" AT_SUMMARY="${summary:-?}" AT_GIT="$GIT" AT_SECS="$secs" "$PY" - <<'PY' || log "WARNING: ledger append failed (verdict still written)"
import json, os, time
rec = {
    "schema": 1, "kind": "autotune",
    "harness_key": "autotune", "harness_key8": "autotune", "harness_version": "autotune-1.0",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "epoch": int(time.time()),
    "verdict": os.environ["AT_VERDICT"],
    "tunables": os.environ["AT_SUMMARY"].strip(),
    "git": os.environ["AT_GIT"], "secs": int(os.environ["AT_SECS"]),
    "provenance": "autotune",
    "note": "idle tunable sweep: per-machine perf constants -> research/tuned-profile.toml "
            "(ADVISORY timings; guards = gate-bin bit-identity + encode result-invariance; "
            "the tuner never changes code defaults — consumers opt in via tools/autotune/apply.py)",
}
with open("research/results-ledger.jsonl", "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
PY

echo "$V :: ${summary:-?} (git $GIT, ${secs}s)" > "$VF"
log "verdict $V :: ${summary:-?} (git $GIT, ${secs}s)"
[ "$V" = PASS ] && exit 0 || exit 1
