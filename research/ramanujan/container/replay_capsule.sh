#!/usr/bin/env bash
# Q0 clean-proof replay contract (host side).
#
# Usage:
#   ./replay_capsule.sh [CAPSULE.json|.lean]
#
# Runs the capsule inside the pinned clean container with:
#   --network=none
#   no host Mathlib / elan mounts
# Exit 0 if the machine-check succeeds; non-zero otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CAPSULE="${1:-$ROOT/capsules/two_plus_two.capsule.json}"
if [[ ! -e "$CAPSULE" ]]; then
  echo "error: capsule not found: $CAPSULE" >&2
  exit 2
fi
CAPSULE="$(cd "$(dirname "$CAPSULE")" && pwd)/$(basename "$CAPSULE")"

MATHLIB_COMMIT="$(python3 -c 'import json; print(json.load(open("'"$ROOT"'/pins.json"))["mathlib"]["commit"])')"
SHORT="${MATHLIB_COMMIT:0:8}"
IMAGE="${RAMANUJAN_REPLAY_IMAGE:-ramanujan-clean-replay:${SHORT}}"

if [[ -z "${DOCKER_HOST:-}" ]]; then
  if [[ -S "${HOME}/.colima/default/docker.sock" ]]; then
    export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker CLI not found" >&2
  exit 127
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "error: image $IMAGE not present. Build first: $ROOT/build.sh" >&2
  exit 127
fi

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
CAPSULE_DIR="$(dirname "$CAPSULE")"
CAPSULE_BASE="$(basename "$CAPSULE")"
RECEIPT_DIR="${RAMANUJAN_REPLAY_RECEIPT_DIR:-$ROOT}"
mkdir -p "$RECEIPT_DIR"
HOST_RECEIPT="$RECEIPT_DIR/REPLAY_RECEIPT.json"

echo "replay_contract: image=$IMAGE"
echo "replay_contract: image_id=$IMAGE_ID"
echo "replay_contract: capsule=$CAPSULE"
echo "replay_contract: network=none"

set +e
# --network=none is the contract. No host Lean/Mathlib mounts.
# HOME on tmpfs so lake/elan write no host state and need no image RW home.
docker run --rm \
  --network=none \
  -e HOME=/tmp/ramanujan-home \
  --tmpfs /tmp:rw,exec,nosuid,size=1g \
  -v "$CAPSULE_DIR:/capsule:ro" \
  "$IMAGE" \
  "/capsule/$CAPSULE_BASE"
EC=$?
set -e

STATUS="REPLAY_OK"
if [[ $EC -ne 0 ]]; then
  STATUS="REPLAY_FAILED"
fi

python3 - "$HOST_RECEIPT" "$STATUS" "$EC" "$IMAGE" "$IMAGE_ID" "$CAPSULE" "$MATHLIB_COMMIT" <<'PY'
import json, sys, time
path, status, ec, image, image_id, capsule, mathlib = sys.argv[1:8]
doc = {
    "schema": "hawking.ramanujan.clean_proof_replay_receipt.v1",
    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": status,
    "exit_code": int(ec),
    "image": image,
    "image_id": image_id,
    "capsule": capsule,
    "mathlib_commit": mathlib,
    "network": "none",
    "host_state": "none (capsule bind-mounted read-only; no host toolchain mounts)",
    "contract": "a Tier 3 claim must be re-provable from its capsule in a container built only from the lock, with no network and no host state",
}
open(path, "w").write(json.dumps(doc, indent=2) + "\n")
print(json.dumps(doc, indent=2))
PY

exit "$EC"
