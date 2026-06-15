#!/usr/bin/env bash
# replay.sh — the idle invariant sweep ("dreaming" = the immune system). v1, 2026-06-11.
#
# Re-runs the CHEAP exactness suite while the box is quiet, so bit-rot (toolchain
# drift, accidental edits, disk corruption, dependency bumps) is caught by a machine
# between milestones instead of by the next milestone. EXACT pass/fail checks only —
# no perf numbers, nothing advisory about the verdict itself.
#
# Checks:
#   1. gate_kernels     ./target/release/gate-kernels — the 5,380-cell decode-path
#                       identity registry (every kernel byte-identical to
#                       decode_tensor_fixed; the determinism law, will.md §5.2)
#   2. provenance_kats  cargo test -p strand-quant --lib provenance — the SPV3 hash
#                       KATs that pin tensor_root/model_root serialization
#   3. attest_artifact  ./target/release/attest-strand on the shipped artifact —
#                       loads via the real mmap consumer path, asserts fixed==lean
#                       decode, recomputes the model root and requires it to MATCH
#                       the SPRV-stored root (the "stored == recomputed" gate)
#   4. ledger_check     scripts/strand-eval ledger check — 0 ERROR lines (the
#                       15-digit tell / contamination checks)
#
# Contract with the conductor (ops/conductor.sh v7 §5b/5c):
#   - the conductor launches this at most once per quiet 6h window
#     (it stamps scratch/.replay-last at launch)
#   - we self-gate AGAIN on idleness + a pid lock, so running it by hand any time
#     is safe (it skips quietly if the box is busy)
#   - verdict -> scratch/.replay-verdict (one line: "PASS|FAIL :: per-check summary");
#     the conductor turns PASS into an S3 digest line and FAIL into an S1 wake
#     (bit-rot detected = maximum salience)
#   - one JSONL record appended to research/results-ledger.jsonl per COMPLETED
#     replay (harness_key=replay, no ppl field — `ledger check` will WARN
#     "record without ppl" on these; known and benign, replay records are not
#     canon comparisons and never enter the 15-digit grouping)
#   - a SKIP (missing binary/artifact) is not an invariant failure, but a replay
#     where EVERYTHING skipped verified nothing -> verdict FAIL (honesty rule:
#     "nothing checked" must not read as "all good")
#   - exit 0 on PASS or quiet-skip; exit 1 on FAIL
set -u
cd "$(dirname "$0")/.."
LOCK=scratch/.replay-lock
VF=scratch/.replay-verdict
LEDGER=research/results-ledger.jsonl
ART=scratch/artifacts/qwen05b-pv2-2bit.strand
log(){ echo "[replay $(date '+%d %H:%M:%S')] $*"; }

# ── idle gate (same pgrep discipline as the conductor's idle jobs) ─────────────
# bracket-constructed patterns: the self-match trap (will.md 2026-06-11) — the
# literal must not appear in our own cmdline or this script's text.
if pgrep -f 'strand-qat[.]py|quantize-mode[l]|strand-7b-pp[l]|strand-act[2]' >/dev/null 2>&1; then
    log "box busy (qat/quant/eval running) — skipping, not an error"
    exit 0
fi
# single-instance pid lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    log "another replay holds the lock ($(cat "$LOCK")) — skipping"
    exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

summary=""; fails=0; ran=0
note(){ # name STATUS
    summary="${summary}${1}=${2} "
    case $2 in FAIL) fails=$((fails+1)); ran=$((ran+1));;
               PASS) ran=$((ran+1));;
               *) :;; esac
    log "check $1: $2"
}
t0=$(date +%s)

# 1) gate-kernels — the decode-path identity registry
if [ -x target/release/gate-kernels ]; then
    if out=$(target/release/gate-kernels 2>&1) && echo "$out" | grep -q 'byte-identical'; then
        note gate_kernels PASS
    else
        log "gate-kernels output tail: $(echo "$out" | tail -3 | tr '\n' '|')"
        note gate_kernels FAIL
    fi
else
    note gate_kernels SKIP
fi

# 2) provenance KATs (require >0 tests to have actually run — a filter matching
# nothing exits 0 and would be a silent no-op)
if out=$(cargo test -p strand-quant --lib provenance 2>&1); then
    passed=$(echo "$out" | grep -Eo '[0-9]+ passed' | tail -1 | awk '{print $1}')
    if [ "${passed:-0}" -gt 0 ] 2>/dev/null; then
        note provenance_kats PASS
    else
        log "provenance KATs: filter matched 0 tests"
        note provenance_kats FAIL
    fi
else
    log "provenance KATs output tail: $(echo "$out" | tail -5 | tr '\n' '|')"
    note provenance_kats FAIL
fi

# 3) attest-strand on the shipped artifact: stored SPRV root == recomputed root
if [ -x target/release/attest-strand ] && [ -f "$ART" ]; then
    if out=$(target/release/attest-strand "$ART" 2>&1) \
       && echo "$out" | grep -q 'matches SPRV stored root'; then
        note attest_artifact PASS
    else
        log "attest-strand output tail: $(echo "$out" | tail -4 | tr '\n' '|')"
        note attest_artifact FAIL
    fi
else
    note attest_artifact SKIP
fi

# 4) ledger check: 0 ERROR lines (WARNs are expected — incl. our own replay records)
if out=$(./scripts/strand-eval ledger check 2>&1); then
    ec=$(echo "$out" | grep -c '^ERROR ')
    if [ "${ec:-0}" -eq 0 ]; then
        note ledger_check PASS
    else
        log "ledger errors: $(echo "$out" | grep '^ERROR ' | head -2 | tr '\n' '|')"
        note ledger_check FAIL
    fi
else
    log "ledger check crashed: $(echo "$out" | tail -3 | tr '\n' '|')"
    note ledger_check FAIL
fi

# ── verdict ────────────────────────────────────────────────────────────────────
if [ "$fails" -gt 0 ]; then V=FAIL
elif [ "$ran" -eq 0 ]; then V=FAIL; summary="${summary}(all-skipped=verified-nothing) "
else V=PASS; fi
secs=$(( $(date +%s) - t0 ))
GIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

# one ledger line per completed replay (harness_key=replay; torch-free python)
REPLAY_VERDICT="$V" REPLAY_SUMMARY="$summary" REPLAY_GIT="$GIT" REPLAY_SECS="$secs" \
python3 - <<'PY' || log "WARNING: ledger append failed (verdict still written)"
import json, os, time
rec = {
    "schema": 1, "kind": "replay",
    "harness_key": "replay", "harness_key8": "replay", "harness_version": "replay-1.0",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "epoch": int(time.time()),
    "verdict": os.environ["REPLAY_VERDICT"],
    "checks": os.environ["REPLAY_SUMMARY"].strip(),
    "git": os.environ["REPLAY_GIT"], "secs": int(os.environ["REPLAY_SECS"]),
    "provenance": "replay",
    "note": "idle invariant sweep: gate-kernels identity registry, provenance KATs, "
            "attest-strand stored==recomputed root, ledger check (exact gates only)",
}
with open("research/results-ledger.jsonl", "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
PY

echo "$V :: ${summary}(git $GIT, ${secs}s)" > "$VF"
log "verdict $V :: ${summary}(git $GIT, ${secs}s)"
[ "$V" = PASS ] && exit 0 || exit 1
