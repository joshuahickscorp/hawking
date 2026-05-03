#!/usr/bin/env bash
# Verify the evidence triples produced by the most recent haul.
# Reads tools/haul/_evidence/*/{pre,post,verify}.json and checks:
#   - all three files present per gate
#   - post.json's exit_code is 0
#   - verify.json's attestation is true
#
# Usage:
#   ./tools/haul/verify-evidence.sh phase1
#
# Exit:
#   0 — all gates have green evidence triples
#   1 — at least one gate has missing or failed evidence

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE="${1:-phase1}"
EVIDENCE_DIR="$REPO_ROOT/tools/haul/_evidence"

if [[ ! -d "$EVIDENCE_DIR" ]] || [[ -z "$(ls -A "$EVIDENCE_DIR" 2>/dev/null)" ]]; then
    echo "verify: no evidence at $EVIDENCE_DIR (haul never ran or was wiped)"
    exit 1
fi

# JSON parse helper. macOS default has python3 but no jq. The exit
# code distinguishes:
#   0   key found and value printed
#   2   file unreadable / not valid JSON (path + byte-length echoed to stderr)
#   3   key absent in an otherwise-valid JSON object
# A truncated mid-write JSON (memory died during haul) hits 2, not a
# silent empty-string fallback.
json_get() {
    python3 - "$1" "$2" <<'PY'
import json, os, sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path, "rb") as f:
        raw = f.read()
    d = json.loads(raw)
except FileNotFoundError:
    sys.stderr.write(f"json_get: missing {path}\n")
    sys.exit(2)
except json.JSONDecodeError as e:
    sys.stderr.write(
        f"json_get: invalid JSON in {path} ({len(raw)} bytes, "
        f"line {e.lineno} col {e.colno}): {e.msg}\n"
    )
    sys.exit(2)
if key not in d:
    sys.stderr.write(f"json_get: key '{key}' absent in {path}\n")
    sys.exit(3)
v = d[key]
print(v if not isinstance(v, bool) else ("true" if v else "false"))
PY
}

failed=0

# When verify-evidence is invoked as a validator gate (e.g. P0.1 in the
# super haul), run-gates.sh publishes "<gate> <pid>" to .active *before*
# launching the validator. We must skip that gate while it's mid-write
# — pre.json exists but post.json doesn't yet, and treating it as a
# failure would create a chicken-and-egg halt at the very gate doing
# the verifying.
ACTIVE_GATE=""
if [[ -f "$EVIDENCE_DIR/.active" ]]; then
    read -r ACTIVE_GATE _ <"$EVIDENCE_DIR/.active" 2>/dev/null || true
fi

for gate_dir in "$EVIDENCE_DIR"/*/; do
    [[ -d "$gate_dir" ]] || continue
    gate_id="$(basename "$gate_dir")"

    # Skip the currently-running gate (chicken-and-egg, see comment above).
    [[ -n "$ACTIVE_GATE" && "$gate_id" == "$ACTIVE_GATE" ]] && continue

    pre="$gate_dir/pre.json"
    post="$gate_dir/post.json"
    verify="$gate_dir/verify.json"

    # Defensive: directories without a pre.json aren't gates — they're
    # auxiliary scratch (stderr captures, applied-patch artifacts, etc.).
    # Skip them silently rather than treating them as failed gates.
    if [[ ! -f "$pre" ]]; then
        continue
    fi

    missing=()
    [[ -f "$post" ]]   || missing+=("post.json")
    [[ -f "$verify" ]] || missing+=("verify.json")

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "verify: $gate_id MISSING — ${missing[*]}"
        failed=1
        continue
    fi

    # Capture stderr from json_get so parse failures surface in the
    # audit output, not just /dev/null.
    post_rc="$(json_get "$post" exit_code 2> >(sed "s|^|verify: $gate_id post.json: |" >&2))"
    pg_rc=$?
    attestation="$(json_get "$verify" attestation 2> >(sed "s|^|verify: $gate_id verify.json: |" >&2))"
    at_rc=$?

    if [[ $pg_rc -ne 0 ]]; then
        echo "verify: $gate_id FAIL — post.json unreadable (rc=$pg_rc)"
        failed=1
        continue
    fi

    if [[ $at_rc -ne 0 ]]; then
        echo "verify: $gate_id FAIL — verify.json unreadable (rc=$at_rc)"
        failed=1
        continue
    fi

    if [[ "$post_rc" != "0" ]]; then
        echo "verify: $gate_id FAIL — post.exit_code=$post_rc"
        failed=1
        continue
    fi

    # post.json's timed_out field is best-effort; surface it for the
    # audit trail when present.
    timed_out="$(json_get "$post" timed_out 2>/dev/null || echo unknown)"
    if [[ "$timed_out" == "true" ]]; then
        echo "verify: $gate_id FAIL — validator timed out"
        failed=1
        continue
    fi

    if [[ "$attestation" != "True" && "$attestation" != "true" ]]; then
        echo "verify: $gate_id FAIL — verify.attestation=$attestation"
        failed=1
        continue
    fi

    echo "verify: $gate_id OK"
done

if [[ $failed -ne 0 ]]; then
    exit 1
fi
echo "verify: all evidence triples green"
exit 0
