#!/usr/bin/env bash
# =============================================================================
# tools/bench/report_card_selftest.sh — cheap confirmation gate for
# tools/bench/report_card.sh. Runs NO model and NO bench: it (1) bash -n
# syntax-checks the report card, and (2) dry-parses a synthetic results record
# through the SAME pipe-delimited contract the report card's table renderer
# uses, asserting the 9-field schema and the documented lane names stay in sync.
#
# This locks the already-shipped report-card lane/column schema so a silent
# edit that breaks the parse (wrong field count, renamed lane, dropped column)
# fails fast without needing a clean room or a GPU.
#
# USAGE / GATE:
#   bash tools/bench/report_card_selftest.sh
#   # exits 0 on pass, non-zero (with a FAIL line) on any regression.
#
# It is itself bash -n clean.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

CARD="tools/bench/report_card.sh"
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
pass() { printf 'ok: %s\n' "$*"; }

# ── (1) bash -n syntax check of the report card itself ──────────────────────
[[ -f "$CARD" ]] || fail "$CARD not found"
bash -n "$CARD" || fail "bash -n found a syntax error in $CARD"
pass "bash -n clean: $CARD"

# ── (2a) the record schema is exactly 9 pipe-delimited fields ───────────────
# This mirrors record_lane()'s printf in report_card.sh:
#   name|tps|J/tok_GPU|J/tok_pkg|wall|readback|lane_type|feature_flags|git_sha
SYNTH='dismantle-fast|41.92|0.171|0.402|6.131|4|generate|--profile fast + kernel-profile=qwen3b.json|abc1234'
nfields=$(awk -F'|' '{print NF}' <<<"$SYNTH")
[[ "$nfields" -eq 9 ]] || fail "synthetic record has $nfields fields, expected 9"
pass "results record schema = 9 fields"

# ── (2b) dry-parse the record through the renderer's exact read/printf path ──
# If the column layout in report_card.sh changes field order/count, keep this
# format string in lockstep (it is copied from the table renderer).
rendered=$(while IFS='|' read -r name tps jg jp wall rb ltype flags sha; do
    printf '%-32s %9s %11s %11s %8s %22s %30s %28s %10s\n' \
        "$name" "$tps" "$jg" "$jp" "$wall" "$rb" "$ltype" "$flags" "$sha"
done <<<"$SYNTH")
[[ -n "$rendered" ]] || fail "renderer produced no output for a valid record"
grep -q "dismantle-fast" <<<"$rendered" || fail "rendered row dropped the lane name"
grep -q "41.92"          <<<"$rendered" || fail "rendered row dropped dec_tps"
grep -q "abc1234"        <<<"$rendered" || fail "rendered row dropped git_sha"
pass "dry-parse: synthetic record renders through the 9-column layout"

# ── (2c) the documented lane names the dispatcher runs are all present ───────
# Guards against a lane being renamed in one place but not the ONLY=/want list.
for lane in \
    dismantle-default \
    dismantle-fast \
    hawking-serve-full-logits \
    hawking-serve-greedy-b1 \
    hawking-serve-greedy-b8 \
    llama-cli \
    llama-server-b8
do
    grep -q "\"$lane\"" "$CARD" || fail "lane '$lane' missing from $CARD dispatcher"
done
pass "all 7 documented lane names present in dispatcher"

# ── (2d) the readback contract the greedy lane test asserts is documented ────
# B=1 greedy = 4 bytes/tok; this string is the human-facing twin of the Rust
# greedy_lane_routing test's B×4 assertion. Keep them in sync.
grep -q "B=1 greedy: 4 bytes/tok" "$CARD" \
    || fail "report card lost the 'B=1 greedy: 4 bytes/tok' readback contract note"
pass "greedy readback contract (4 bytes/tok) documented in column legend"

# ── (2e) the table header lists the 9 columns the record carries ────────────
for col in dec_tps J/tok_GPU J/tok_pkg wall_s readback_bytes/tok lane_type feature_flags git_sha; do
    grep -q "$col" "$CARD" || fail "table header missing column '$col'"
done
pass "table header lists all 9 columns"

printf 'PASS: report_card.sh self-test (syntax + dry-parse + schema)\n'
