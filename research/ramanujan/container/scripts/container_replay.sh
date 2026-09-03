#!/usr/bin/env bash
# Inside-container entrypoint: re-check a proof capsule against the baked Mathlib.
# Exit 0 on successful machine-check; non-zero otherwise.
set -euo pipefail

PINS_FILE="${RAMANUJAN_PINS:-/opt/ramanujan/pins.json}"
MATHLIB_DIR="${MATHLIB_DIR:-/opt/mathlib4}"
WORK="${RAMANUJAN_WORK:-/work/capsule_replay}"

usage() {
  cat <<'EOF'
container_replay CAPSULE.json|.lean

Re-prove a Tier-3 proof capsule inside the clean container.
Expects Mathlib and Lean to already be baked into the image (no network, no host state).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CAPSULE_PATH="${1:-/opt/ramanujan/capsules/two_plus_two.capsule.json}"
if [[ ! -e "$CAPSULE_PATH" ]]; then
  echo "error: capsule not found: $CAPSULE_PATH" >&2
  exit 2
fi

mkdir -p "$WORK"
rm -rf "${WORK:?}/"*
LEAN_FILE="$WORK/Capsule.lean"
RECEIPT="$WORK/replay_receipt.json"

CAPSULE_ID="capsule"
EXPECTED_MATHLIB=""
EXPECTED_LEAN=""

if [[ "$CAPSULE_PATH" == *.lean ]]; then
  cp "$CAPSULE_PATH" "$LEAN_FILE"
  CAPSULE_ID="$(basename "$CAPSULE_PATH" .lean)"
else
  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 required to parse JSON capsules" >&2
    exit 2
  fi
  META="$(python3 - "$CAPSULE_PATH" "$LEAN_FILE" <<'PY'
import json, sys
capsule_path, lean_out = sys.argv[1], sys.argv[2]
doc = json.load(open(capsule_path))
proof = doc.get("proof_lean") or doc.get("lean_source") or ""
if not str(proof).strip():
    sys.stderr.write("error: capsule missing proof_lean\n")
    sys.exit(2)
open(lean_out, "w").write(proof if proof.endswith("\n") else proof + "\n")
pins = doc.get("pins") or {}
print(doc.get("id") or "capsule")
print(pins.get("mathlib_commit") or "")
print(pins.get("lean_commit") or pins.get("lean_toolchain") or "")
PY
)"
  CAPSULE_ID="$(printf '%s\n' "$META" | sed -n '1p')"
  EXPECTED_MATHLIB="$(printf '%s\n' "$META" | sed -n '2p')"
  EXPECTED_LEAN="$(printf '%s\n' "$META" | sed -n '3p')"
fi

IMAGE_MATHLIB="$(python3 -c 'import json; print(json.load(open("'"$PINS_FILE"'"))["mathlib"]["commit"])')"
IMAGE_LEAN="$(python3 -c 'import json; print(json.load(open("'"$PINS_FILE"'"))["lean_for_mathlib_checks"]["commit"])')"

if [[ -n "${EXPECTED_MATHLIB}" && "$EXPECTED_MATHLIB" != "$IMAGE_MATHLIB" ]]; then
  echo "error: capsule mathlib_commit $EXPECTED_MATHLIB != image $IMAGE_MATHLIB" >&2
  exit 3
fi
if [[ -n "${EXPECTED_LEAN}" ]]; then
  case "$EXPECTED_LEAN" in
    "$IMAGE_LEAN"|leanprover/lean4:v4.33.0-rc1|4.33.0-rc1) ;;
    *)
      echo "error: capsule lean pin $EXPECTED_LEAN does not match image lean $IMAGE_LEAN" >&2
      exit 3
      ;;
  esac
fi

export GIT_TERMINAL_PROMPT=0
cd "$MATHLIB_DIR"

echo "replay: capsule=${CAPSULE_ID}"
echo "replay: mathlib=$IMAGE_MATHLIB"
echo "replay: lean=$(lean --version | head -1)"
echo "replay: checking $LEAN_FILE"

set +e
OUT="$(lake env lean "$LEAN_FILE" 2>&1)"
EC=$?
set -e

if [[ $EC -eq 0 ]]; then
  STATUS="REPLAY_OK"
else
  STATUS="REPLAY_FAILED"
fi

python3 - "$RECEIPT" "$STATUS" "$EC" "$IMAGE_MATHLIB" "$IMAGE_LEAN" "$CAPSULE_ID" <<'PY'
import json, sys, time
path, status, ec, mathlib, lean, cid = sys.argv[1:7]
doc = {
    "schema": "hawking.ramanujan.capsule_replay_receipt.v1",
    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "capsule_id": cid,
    "status": status,
    "exit_code": int(ec),
    "mathlib_commit": mathlib,
    "lean_commit": lean,
    "network": "none (expected; container should be started with --network=none)",
    "host_state": "none (image-local Mathlib and Lean only)",
}
open(path, "w").write(json.dumps(doc, indent=2) + "\n")
print(json.dumps(doc, indent=2))
PY

if [[ -n "$OUT" ]]; then
  echo "$OUT"
fi

if [[ $EC -ne 0 ]]; then
  echo "error: lake env lean failed with exit $EC" >&2
  exit "$EC"
fi

echo "replay: OK"
exit 0
