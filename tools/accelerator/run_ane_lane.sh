#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h}"
SDK="${HAWKING_COREML_SDK:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}"
OUT="${ROOT}/receipts/headless/APPLE_ANE_DEVICE_PROFILE.json"
MODEL="${HAWKING_ANE_COMPILED_MODEL:-}"
ANE_MODULE_CACHE="${HAWKING_ANE_MODULE_CACHE:-/tmp/hawking-ane-module-cache}"

if [[ "$#" -gt 0 ]]; then
  MODEL="$1"
fi

if [[ ! -d "${SDK}" ]]; then
  print -u2 "ANE lane blocked: Core ML SDK not found at ${SDK}"
  exit 2
fi

mkdir -p "${ANE_MODULE_CACHE}"
swiftc -module-cache-path "${ANE_MODULE_CACHE}" -parse-as-library -sdk "${SDK}" \
  -framework CoreML "${ROOT}/tools/accelerator/ane_probe.swift" -o /tmp/hawking_ane_probe
if [[ -n "${MODEL}" ]]; then
  /tmp/hawking_ane_probe "${OUT}" "${MODEL}"
else
  /tmp/hawking_ane_probe "${OUT}"
fi
PYTHONPATH="${ROOT}/workspace/ops/ane/python:${ROOT}" \
  python3 "${ROOT}/tools/accelerator/ane_micrograph_author.py"
