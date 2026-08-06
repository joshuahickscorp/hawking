#!/usr/bin/env bash
# Fail-closed parent restream launcher.  It deliberately rejects whole-shard
# downloads, unsealed D5/D8/D9 source decisions, and an untested physical
# operator before a parent body byte can be admitted.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

schedule="${GLM52_STREAMING_SCHEDULE_PATH:-workspace/campaign/evidence/models/glm52/GLM52_STREAMING_SCHEDULE_90GB.json}"
policy="${GLM52_RESOURCE_RESERVE_POLICY_PATH:-workspace/campaign/evidence/models/glm52/GLM52_RESOURCE_RESERVE_POLICY_90GB.json}"
python_bin="${HAWKING_GLM52_PYTHON:-$repo_root/.venv/glm52/bin/python}"
range_executor="${HAWKING_GLM52_RANGE_STREAM_EXECUTOR:-$repo_root/tools/condense/glm52_range_stream_executor.py}"
export HAWKING_GLM52_RANGE_STREAM_EXECUTOR="$range_executor"
external_receipt="${HAWKING_EXTERNAL_SOURCE_FREEZE_RECEIPT:-}"
window_operator="${HAWKING_GLM52_WINDOW_OPERATOR:-}"
production_lease_receipt="${HAWKING_PRODUCTION_GPU_LEASE_RECEIPT:-}"
operator_approval_receipt="${HAWKING_GLM52_OPERATOR_APPROVAL_RECEIPT:-}"
owner_authorization="${HAWKING_OWNER_GREEN_LIGHT_AUTHORIZATION:-}"
owner_public_key="$({ "$python_bin" - <<'PY'
from ramanujan.restream_guard import pinned_owner_public_key_path
print(pinned_owner_public_key_path())
PY
} 2>/dev/null || true)"
declare -a refusals=()

refuse() {
  refusals+=("$1")
}

# Collect every missing prelaunch input before returning.  In particular, a
# missing public D5/D8/D9 receipt names all three owner decisions individually
# rather than hiding two of them behind a first-error exit.
if [[ "${HAWKING_PARENT_RESTREAM_AUTHORIZED:-}" != "YES" ]]; then
  refuse "OWNER_PARENT_RESTREAM_AUTHORIZATION: set HAWKING_PARENT_RESTREAM_AUTHORIZED=YES after explicit run-owner approval."
fi
if [[ -z "${HAWKING_CLEAN_GPU_LEASE_ID:-}" ]]; then
  refuse "CLEAN_LIVE_GPU_LEASE: set HAWKING_CLEAN_GPU_LEASE_ID to the clean heavy-GPU lease identity."
fi
if [[ -z "$production_lease_receipt" || ! -f "$production_lease_receipt" ]]; then
  refuse "PRODUCTION_GPU_LEASE_RECEIPT: provide HAWKING_PRODUCTION_GPU_LEASE_RECEIPT; a lease-id string or fixture lease is insufficient."
fi
if [[ -z "$operator_approval_receipt" || ! -f "$operator_approval_receipt" ]]; then
  refuse "OWNER_OPERATOR_APPROVAL_RECEIPT: provide HAWKING_GLM52_OPERATOR_APPROVAL_RECEIPT bound to the physical executable SHA-256 and framed protocol."
fi
if [[ -z "$owner_authorization" || ! -f "$owner_authorization" ]]; then
  refuse "OWNER_SIGNED_GREEN_LIGHT_AUTHORIZATION: provide the single-use Ed25519 authorization receipt bound to schedule, policy, sources, lease, hardware and operator."
fi
if [[ -z "$owner_public_key" || ! -f "$owner_public_key" ]]; then
  refuse "PINNED_OWNER_ED25519_TRUST_ANCHOR: install the protected owner public key at the fixed OS trust-anchor path: $owner_public_key"
fi
if [[ -z "$window_operator" ]]; then
  refuse "OWNER_APPROVED_TESTED_PHYSICAL_GLM52_WINDOW_OPERATOR: set HAWKING_GLM52_WINDOW_OPERATOR to the owner-approved, tested, executable framed-range operator."
elif [[ ! -f "$window_operator" || ! -x "$window_operator" ]]; then
  refuse "OWNER_APPROVED_TESTED_PHYSICAL_GLM52_WINDOW_OPERATOR: HAWKING_GLM52_WINDOW_OPERATOR is not an executable regular file: $window_operator"
fi
if [[ -z "$external_receipt" ]]; then
  refuse "OWNER_D5_SOURCE_LICENSE_SELECTION: provide HAWKING_EXTERNAL_SOURCE_FREEZE_RECEIPT for an authority-validated public D5/D8/D9 receipt."
  refuse "OWNER_D8_SOURCE_LICENSE_AND_HIDDEN_MEMBERSHIP_SELECTION: provide the same receipt only after private D8 mode, hidden membership, and D1-D7 no-leak checks pass."
  refuse "OWNER_D9_SOURCE_LICENSE_AND_VARIANT_GENERATOR_SELECTION: provide the same receipt only after D9 source/license bytes, independent generator executable hash, and seed commitment are bound."
elif [[ ! -f "$external_receipt" ]]; then
  refuse "OWNER_D5_SOURCE_LICENSE_SELECTION: HAWKING_EXTERNAL_SOURCE_FREEZE_RECEIPT does not name a readable public receipt: $external_receipt"
  refuse "OWNER_D8_SOURCE_LICENSE_AND_HIDDEN_MEMBERSHIP_SELECTION: a validated public D8 commitment-only receipt is absent."
  refuse "OWNER_D9_SOURCE_LICENSE_AND_VARIANT_GENERATOR_SELECTION: a validated public D9 generator-binding receipt is absent."
fi
if [[ ! -f "$schedule" ]]; then
  refuse "SEALED_90GB_SCHEDULE: missing schedule: $schedule"
fi
if [[ ! -f "$policy" ]]; then
  refuse "SEALED_90GB_RESOURCE_POLICY: missing policy: $policy"
fi
if [[ ! -x "$python_bin" ]]; then
  refuse "PINNED_GLM52_RUNTIME: missing or non-executable Python: $python_bin"
fi
if [[ -x "$python_bin" ]] && ! "$python_bin" -m ramanujan.status --require-hawking-complete >/dev/null; then
  refuse "HAWKING_EVOLUTION_COMPLETE: Ramanujan remains a local scaffold until Hawking's prepared handoff gate is validated and opened."
fi
if [[ ! -f "$range_executor" || ! -x "$range_executor" ]]; then
  refuse "TESTED_RANGE_TENSOR_EXECUTOR: HAWKING_GLM52_RANGE_STREAM_EXECUTOR is not an executable regular file: $range_executor"
fi

if [[ -n "$external_receipt" && -f "$external_receipt" && -x "$python_bin" ]]; then
  if ! receipt_result="$("$python_bin" - "$external_receipt" <<'PY'
import sys
from pathlib import Path

from lab.operators.glm52_common import read_sealed_json

path = Path(sys.argv[1])
receipt = read_sealed_json(path)
if receipt.get("schema") != "hawking.ramanujan.external_source_freeze_receipt.v1":
    raise SystemExit("wrong external source receipt schema")
if receipt.get("status") != "PASS_INPUTS_FROZEN_RESEARCH_AND_CANDIDATE_AUTHORITY_FALSE":
    raise SystemExit("external source receipt is not a non-authorizing PASS freeze")
if receipt.get("RAMANUJAN_RESEARCH_AUTHORIZED") is not False or receipt.get("candidate_launch_authorized") is not False:
    raise SystemExit("external source receipt illegally grants research or candidate authority")
if receipt.get("independent_adjudication_complete") is not False or receipt.get("counterexample_search_complete") is not False:
    raise SystemExit("external source receipt claims unfinished independent work")
rows = receipt.get("sources")
if not isinstance(rows, list) or [row.get("id") for row in rows if isinstance(row, dict)] != ["D5", "D8", "D9"]:
    raise SystemExit("external source receipt must contain exactly D5, D8, D9 in order")
by_id = {row["id"]: row for row in rows}
if any(row.get("status") != "FROZEN_PENDING_INDEPENDENT_EVALUATION" for row in rows):
    raise SystemExit("external source receipt contains a non-frozen source")
if any(row.get("source_path_or_item_ids_serialized") is not False for row in rows):
    raise SystemExit("public source receipt serializes a forbidden source path or item id")
d5 = by_id["D5"].get("no_leak_audit", {})
if d5.get("direction") != "D5_training_candidate_against_sealed_odyssey_evaluation" or d5.get("exact_or_near_matches") != 0:
    raise SystemExit("D5 Odyssey evaluation no-leak audit is absent or nonzero")
d8 = by_id["D8"].get("no_leak_audit", {})
if d8.get("direction") != "all_current_D1_D7_training_against_D8_hidden_membership" or d8.get("exact_or_near_matches") != 0:
    raise SystemExit("D8 hidden-membership no-leak audit is absent or nonzero")
visible = receipt.get("training_visible")
if not isinstance(visible, dict) or visible.get("D8_hidden_item_ids") is not None or visible.get("D8_commitment_only") is not True:
    raise SystemExit("D8 privacy boundary is absent")
d9_generator = by_id["D9"].get("variant_generator")
if not isinstance(d9_generator, dict) or d9_generator.get("executed_by_freeze") is not False:
    raise SystemExit("D9 generator binding is absent or was executed by freeze")
if not isinstance(d9_generator.get("executable_sha256"), str) or len(d9_generator["executable_sha256"]) != 64:
    raise SystemExit("D9 generator executable SHA-256 is absent")
if not isinstance(d9_generator.get("seed_commitment_sha256"), str) or len(d9_generator["seed_commitment_sha256"]) != 64:
    raise SystemExit("D9 generator seed commitment is absent")
print(f"external owner receipt accepted: {path}; seal={receipt['seal_sha256']}")
PY
)"; then
    refuse "OWNER_D5_SOURCE_LICENSE_SELECTION: supplied public receipt is not a valid sealed D5/D8/D9 authority result: ${receipt_result//$'\n'/ }"
    refuse "OWNER_D8_SOURCE_LICENSE_AND_HIDDEN_MEMBERSHIP_SELECTION: supplied receipt does not prove the required D8 privacy/no-leak boundary: ${receipt_result//$'\n'/ }"
    refuse "OWNER_D9_SOURCE_LICENSE_AND_VARIANT_GENERATOR_SELECTION: supplied receipt does not prove the required D9 independent generator binding: ${receipt_result//$'\n'/ }"
  fi
fi

if [[ -f "$schedule" && -f "$policy" && -x "$python_bin" ]]; then
  if ! bounded_result="$("$python_bin" - "$schedule" "$policy" <<'PY'
import importlib.metadata
import sys
from pathlib import Path

from lab.operators.glm52_common import read_sealed_json
from ramanujan.restream_guard import RestreamGuardError, validate_bounded_restream

expected = {"hf_xet": "1.5.2", "huggingface-hub": "1.24.0"}
observed = {name: importlib.metadata.version(name) for name in expected}
if observed != expected:
    raise SystemExit(f"pinned GLM-5.2 runtime mismatch: {observed} != {expected}")
schedule = read_sealed_json(Path(sys.argv[1]))
policy = read_sealed_json(Path(sys.argv[2]))
if schedule.get("repo") != "zai-org/GLM-5.2" or schedule.get("revision") != "b4734de4facf877f85769a911abafc5283eab3d9":
    raise SystemExit("schedule is not bound to the final parent revision")
try:
    result = validate_bounded_restream(schedule, policy)
except RestreamGuardError as exc:
    raise SystemExit(f"incomplete or over-budget 90-GB accounting: {exc}") from exc
print(f"90-GB schedule/policy accepted: peak={result['peak_incremental_bytes']} bytes, windows={result['window_count']}")
PY
)"; then
    refuse "SEALED_90GB_SCHEDULE_OR_PINNED_RUNTIME: ${bounded_result//$'\n'/ }"
  fi
fi

# One state/gate authority evaluates the entire ordered transition, persists a
# restart-safe snapshot, and prints free bytes, floor, model-lane contents and
# the exact next owner action on every invocation.  The legacy shell checks
# above remain only as friendly diagnostics; they cannot promote this result.
green_light_result=""
if [[ -f "$schedule" && -f "$policy" && -x "$python_bin" ]]; then
  if ! green_light_result="$("$python_bin" -m ramanujan.restream_guard status --schedule "$schedule" --policy "$policy")"; then
    refuse "ATOMIC_GREEN_LIGHT_TRANSITION: ${green_light_result//$'\n'/ }"
  fi
fi

if ((${#refusals[@]})); then
  if [[ -n "$green_light_result" ]]; then
    printf '%s\n' "$green_light_result" >&2
  fi
  printf 'REFUSED: no parent body byte was admitted. Resolve every listed gate:\n' >&2
  for refusal in "${refusals[@]}"; do
    printf '  - %s\n' "$refusal" >&2
  done
  printf '%s\n' 'LIVE_ALLOCATION_CAPABILITY_AND_TERMINAL_RESTREAM_RECEIPTS remains a post-run gate: do not pre-claim it; the window operator must emit fresh allocation, remote-hash, eviction, recovery, and terminal receipts.' >&2
  exit 78
fi

printf '%s\n' "$receipt_result"
printf '%s\n' "$bounded_result"
printf '%s\n' "$green_light_result"
exec "$python_bin" "$range_executor" --schedule "$schedule" --policy "$policy" --operator "$window_operator"
