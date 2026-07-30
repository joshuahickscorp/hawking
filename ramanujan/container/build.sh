#!/usr/bin/env bash
# Build the Q0 clean-container image from pins in this directory.
# Network is allowed at build time only. Replay must run with --network=none.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MATHLIB_COMMIT="$(python3 -c 'import json; print(json.load(open("pins.json"))["mathlib"]["commit"])')"
SHORT="${MATHLIB_COMMIT:0:8}"
IMAGE="ramanujan-clean-replay:${SHORT}"

# Prefer Colima when the default context (OrbStack on this host) is hung.
if [[ -z "${DOCKER_HOST:-}" ]]; then
  if [[ -S "${HOME}/.colima/default/docker.sock" ]]; then
    export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker CLI not found" >&2
  exit 127
fi

if ! docker info >/dev/null 2>&1; then
  echo "error: docker daemon not reachable (DOCKER_HOST=${DOCKER_HOST:-default})" >&2
  echo "hint: on this host Colima works with DOCKER_HOST=unix://\$HOME/.colima/default/docker.sock" >&2
  exit 127
fi

echo "building $IMAGE (mathlib $MATHLIB_COMMIT)"
echo "docker host: ${DOCKER_HOST:-default context}"

# This host's docker CLI lacks a working buildx; legacy builder is the reliable path.
# Override with DOCKER_BUILDKIT=1 only when buildx is known good.
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"

docker build \
  --platform linux/arm64 \
  -t "$IMAGE" \
  -f Dockerfile \
  .

# Record image id for the lock / receipt.
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
DIGEST_LOCAL="$IMAGE_ID"
cat > "$ROOT/BUILD_RECEIPT.json" <<EOF
{
  "schema": "hawking.ramanujan.container_build_receipt.v1",
  "at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "image": "$IMAGE",
  "image_id": "$IMAGE_ID",
  "mathlib_commit": "$MATHLIB_COMMIT",
  "docker_host": "${DOCKER_HOST:-default}",
  "platform": "linux/arm64",
  "status": "BUILT"
}
EOF

echo "built $IMAGE ($DIGEST_LOCAL)"
echo "receipt: $ROOT/BUILD_RECEIPT.json"
