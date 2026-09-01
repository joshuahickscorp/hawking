"""Machine-readable physical candidate frontier for the Accelerator campaign.

The architecture-repatriation queue answers ``which architectural behaviors
are worth studying``.  This queue answers the next operational question:
``which concrete model/kernel mutation is ready for which evidence rung``.

This module is deliberately plan-first.  Building or emitting the queue does
not inspect a live GPU, start a resident, acquire a lock, or claim a timing
result.  HCLI may turn only READY rows into WorkUnit proposals; the benchmark
and receipt gates remain authoritative for every later status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hcli.persist import atomic_write_json
from hcli.workunit import WorkUnit


SCHEMA = "hawking.accelerator.physical_qualification_queue.v1"
DEFAULT_QUEUE_OUT = Path("receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json")
STATUSES = frozenset(
    {
        "STATIC_ONLY",
        "READY_DIAGNOSTIC",
        "DIAGNOSTIC_PASS",
        "DIAGNOSTIC_REJECT",
        "READY_PROTECTED",
        "PROTECTED_PASS",
        "PROTECTED_REJECT",
        "INTEGRATED",
        "BLOCKED",
    }
)
READY_STATUSES = frozenset({"READY_DIAGNOSTIC", "READY_PROTECTED"})
PASS_STATUSES = frozenset({"DIAGNOSTIC_PASS", "PROTECTED_PASS", "INTEGRATED"})
EVIDENCE_STATUSES = PASS_STATUSES | frozenset({"DIAGNOSTIC_REJECT", "PROTECTED_REJECT"})
PROTECTED_MEASURED_STATUSES = frozenset({"PROTECTED_PASS", "INTEGRATED"})
CANDIDATE_SCOPE_TAGS = frozenset(
    {
        "MODEL_LOCAL",
        "ARCHITECTURE_FAMILY",
        "BACKEND_FAMILY",
        "GENERIC_CANDIDATE",
        "GENERIC_VERIFIED",
    }
)
# These candidates share the same Hawking Metal runtime implementation across
# the Qwen and Flash frontiers. They are tagged as generic candidates, never
# generic laws: protected cross-model transfer evidence is still required.
GENERIC_RUNTIME_CANDIDATES = frozenset(
    {
        "qwen27-pipeline-state-elision",
        "qwen27-pipeline-cache-reuse",
        "qwen27-pipeline-id-resolution",
        "qwen27-encoder-label-elision",
        "flash-pipeline-cache-reuse",
        "flash-pipeline-id-resolution",
        "flash-encoder-label-elision",
    }
)
MEASUREMENT_FIELDS = (
    "total_nx_bytes",
    "resident_bytes",
    "active_representation_bytes_per_token",
    "actual_read_bytes_per_token",
    "transient_bytes_per_token",
    "gpu_ns_per_token",
    "complete_wall_ns_per_accepted_token",
    "dispatches_per_token",
    "sync_ns_per_token",
    "accepted_tps",
    "fallback_count",
)
# Progression is intentionally one-way across evidence rungs.  A rejection or
# block can only return to STATIC_ONLY, forcing a fresh static review before a
# candidate is made runnable again.
STATUS_TRANSITIONS = {
    "STATIC_ONLY": frozenset({"READY_DIAGNOSTIC", "READY_PROTECTED", "BLOCKED"}),
    "READY_DIAGNOSTIC": frozenset({"DIAGNOSTIC_PASS", "DIAGNOSTIC_REJECT", "BLOCKED"}),
    "DIAGNOSTIC_PASS": frozenset({"READY_PROTECTED", "BLOCKED"}),
    "DIAGNOSTIC_REJECT": frozenset({"STATIC_ONLY", "BLOCKED"}),
    "READY_PROTECTED": frozenset({"PROTECTED_PASS", "PROTECTED_REJECT", "BLOCKED"}),
    "PROTECTED_PASS": frozenset({"INTEGRATED", "BLOCKED"}),
    "PROTECTED_REJECT": frozenset({"STATIC_ONLY", "BLOCKED"}),
    "INTEGRATED": frozenset(),
    "BLOCKED": frozenset({"STATIC_ONLY"}),
}
REQUIRED_FIELDS = (
    "candidate_id",
    "model",
    "affected_physical_region",
    "baseline_path",
    "exact_mutation",
    "expected_eliminated_work",
    "expected_dispatch_reduction",
    "expected_intermediate_byte_reduction",
    "expected_active_byte_change",
    "expected_gpu_ns_mechanism",
    "parity_contract",
    "capability_contract",
    "diagnostic_command",
    "protected_command",
    "dependencies",
    "status",
    "scope_tags",
    "transfer_evidence",
)


class PhysicalQueueError(ValueError):
    """A candidate queue violates the physical qualification contract."""


def _default_scope_tags(candidate_id: str) -> tuple[str, ...]:
    if candidate_id in GENERIC_RUNTIME_CANDIDATES:
        return ("BACKEND_FAMILY", "GENERIC_CANDIDATE")
    return ("MODEL_LOCAL",)


def _default_transfer_evidence(candidate_id: str) -> tuple[str, ...]:
    if candidate_id in GENERIC_RUNTIME_CANDIDATES:
        return (
            "shared Metal runtime implementation is exercised by matched Qwen27 and Flash queue rows; protected cross-model transfer remains unverified",
        )
    return ()


def _empty_measurements() -> dict[str, Any]:
    return {
        "status": "NOT_MEASURED",
        **{name: None for name in MEASUREMENT_FIELDS},
        "absence_reasons": {
            name: "awaiting native protected complete-token receipt" for name in MEASUREMENT_FIELDS
        },
    }


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    return int(value) if value.is_integer() else value


def _receipt_metric(
    receipt: Mapping[str, Any],
    *names: str,
) -> int | float | None:
    """Read a metric without turning an absent provider field into zero."""
    aggregate = receipt.get("aggregate")
    rows = receipt.get("measurements")
    for name in names:
        if isinstance(aggregate, Mapping):
            value = aggregate.get(name)
            if isinstance(value, Mapping):
                value = value.get("median")
            value = _number(value)
            if value is not None:
                return value
        value = _number(receipt.get(name))
        if value is not None:
            return value
        if isinstance(rows, list):
            values: list[int | float] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                value = row.get(name)
                if value is None and isinstance(row.get("native_metrics"), Mapping):
                    value = row["native_metrics"].get(name)
                value = _number(value)
                if value is not None:
                    values.append(value)
            if values:
                values.sort()
                return values[len(values) // 2]
    return None


def _receipt_fallback_count(receipt: Mapping[str, Any]) -> int | float | None:
    direct = _receipt_metric(receipt, "fallback_count")
    if direct is not None:
        return direct
    rows = receipt.get("measurements")
    if not isinstance(rows, list):
        return None
    values = [
        value
        for row in rows
        if isinstance(row, Mapping)
        for value in [_number(row.get("fallbacks"))]
        if value is not None
    ]
    return max(values) if values else None


def _receipt_checks_are_protected(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema") != "hcli.agentos.protected_accelerator_benchmark.v1":
        raise PhysicalQueueError("receipt is not an HCLI protected accelerator benchmark")
    if receipt.get("status") != "PASSED":
        raise PhysicalQueueError("receipt status is not PASSED")
    if receipt.get("benchmark_class") != "QUALIFIED_PROTECTED":
        raise PhysicalQueueError("receipt is not classified QUALIFIED_PROTECTED")
    if receipt.get("measurement_verdict") != "ACCEPT" or receipt.get("qualification") is not True:
        raise PhysicalQueueError("receipt did not pass its protected measurement gate")
    if receipt.get("protected_window") is not True:
        raise PhysicalQueueError("receipt did not record a protected window")
    bench = receipt.get("bench")
    if not isinstance(bench, Mapping) or bench.get("state") != "QUIESCED":
        raise PhysicalQueueError("receipt bench state is not QUIESCED")
    if receipt.get("contamination") not in (None, []):
        raise PhysicalQueueError("receipt contains contamination records")
    checks = receipt.get("checks")
    required_checks = (
        "all_measurements_ok",
        "all_requested_measurements_completed",
        "capability_sanity",
        "zero_fallbacks",
        "one_persistent_pid_when_declared",
        "no_connector_restart",
        "no_unqualified_promotion",
    )
    if not isinstance(checks, Mapping) or any(checks.get(name) is not True for name in required_checks):
        raise PhysicalQueueError("receipt protected checks are incomplete or failed")


def measurements_from_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_mutation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project only declared physical metrics from a protected HCLI receipt.

    This adapter intentionally leaves footprint/read/synchronization fields
    null when the provider did not emit them. A receipt that is valid for the
    protected benchmark is still not sufficient for candidate promotion until
    every queue metric is present.
    """
    if not isinstance(receipt, Mapping):
        raise PhysicalQueueError("receipt must contain an object")
    _receipt_checks_are_protected(receipt)
    if expected_mutation is not None:
        actual = receipt.get("fusion_env_overrides")
        if actual is None and isinstance(receipt.get("identity"), Mapping):
            actual = receipt["identity"].get("child_fusion_env_overrides")
        actual = dict(actual or {}) if isinstance(actual, Mapping) else {}
        expected = {str(key): str(value) for key, value in expected_mutation.items()}
        if actual != expected:
            raise PhysicalQueueError(
                "receipt mutation does not match the candidate: "
                f"expected={expected!r} actual={actual!r}"
            )

    values: dict[str, int | float | None] = {
        "total_nx_bytes": _receipt_metric(receipt, "total_nx_bytes"),
        "resident_bytes": _receipt_metric(receipt, "resident_bytes"),
        "active_representation_bytes_per_token": _receipt_metric(
            receipt,
            "active_representation_bytes_per_token",
            "active_bytes_per_token",
            "active_weight_bytes_per_generated_token",
        ),
        "actual_read_bytes_per_token": _receipt_metric(
            receipt, "actual_read_bytes_per_token"
        ),
        "transient_bytes_per_token": _receipt_metric(
            receipt, "transient_bytes_per_token"
        ),
        "gpu_ns_per_token": _receipt_metric(
            receipt, "gpu_ns_per_token", "gpu_ns_per_generated_token"
        ),
        "complete_wall_ns_per_accepted_token": _receipt_metric(
            receipt,
            "complete_wall_ns_per_accepted_token",
            "complete_wall_ns_per_token",
        ),
        "dispatches_per_token": _receipt_metric(
            receipt, "dispatches_per_token", "dispatches_per_generated_token"
        ),
        "sync_ns_per_token": _receipt_metric(
            receipt,
            "sync_ns_per_token",
            "synchronization_ns_per_token",
            "host_sync_ns_per_token",
        ),
        "accepted_tps": _receipt_metric(receipt, "accepted_tps"),
        "fallback_count": _receipt_fallback_count(receipt),
    }
    if values["accepted_tps"] is None:
        wall = values["complete_wall_ns_per_accepted_token"]
        if isinstance(wall, (int, float)) and wall > 0:
            values["accepted_tps"] = 1_000_000_000.0 / float(wall)

    result: dict[str, Any] = {"status": "RECORDED", **values}
    result["absence_reasons"] = {
        name: (
            "protected receipt did not declare this metric; retain null until the native "
            "complete-token ledger emits it"
        )
        for name, value in values.items()
        if value is None
    }
    return result


def load_receipt_measurements(
    path: str | Path,
    *,
    expected_mutation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt_path = Path(path).expanduser()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    measurements = measurements_from_receipt(payload, expected_mutation=expected_mutation)
    return measurements


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _strings(value: Any, *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PhysicalQueueError(f"{name} must be a list")
    result = tuple(str(item) for item in value)
    if not allow_empty and (not result or any(not item for item in result)):
        raise PhysicalQueueError(f"{name} must contain non-empty strings")
    return result


def _command_has_shell(command: Iterable[str]) -> bool:
    return any(item in {"--shell", "-c", "-lc"} for item in command)


def _hcli_qwen_command(
    output: str,
    env: Mapping[str, str],
    *,
    diagnostic: bool,
) -> tuple[str, ...]:
    command = [
        "python3",
        "-m",
        "hcli",
        "agentos",
        "protected-accelerator-bench",
        "--profile",
        "hcli/hawking-native.sealed-3.14.json",
        "--repo-root",
        ".",
        "--warmup-requests",
        "1" if diagnostic else "2",
        "--measure-requests",
        "3" if diagnostic else "10",
        "--max-new-tokens",
        "8" if diagnostic else "32",
    ]
    for key, value in sorted(env.items()):
        command.extend(("--fusion-env", f"{key}={value}"))
    command.extend(("--emit", output))
    return tuple(command)


def _hcli_flash_plan_command(output: str) -> tuple[str, ...]:
    # Flash remains source/oracle or scaffold-only until a complete NX
    # executable exists.  Keeping this command in the queue makes the blocked
    # boundary executable and inspectable without pretending it is a protected
    # Flash timing command.
    return (
        "python3",
        "-m",
        "hcli",
        "agentos",
        "flash-executable",
        "--repo-root",
        ".",
        "--emit",
        output,
    )


@dataclass(frozen=True)
class PhysicalCandidate:
    candidate_id: str
    model: str
    affected_physical_region: str
    baseline_path: str
    exact_mutation: Mapping[str, Any]
    expected_eliminated_work: str
    expected_dispatch_reduction: str
    expected_intermediate_byte_reduction: str
    expected_active_byte_change: str
    expected_gpu_ns_mechanism: str
    parity_contract: str
    capability_contract: str
    diagnostic_command: tuple[str, ...]
    protected_command: tuple[str, ...]
    dependencies: tuple[str, ...]
    status: str
    scope_tags: tuple[str, ...]
    transfer_evidence: tuple[str, ...]
    control_configuration: Mapping[str, Any] = field(default_factory=dict)
    source_evidence: tuple[str, ...] = ()
    blocked_reason: str | None = None
    evidence: tuple[str, ...] = ()
    measurements: Mapping[str, Any] = field(default_factory=_empty_measurements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model": self.model,
            "affected_physical_region": self.affected_physical_region,
            "baseline_path": self.baseline_path,
            "exact_mutation": _copy(self.exact_mutation),
            "control_configuration": _copy(self.control_configuration),
            "expected_eliminated_work": self.expected_eliminated_work,
            "expected_dispatch_reduction": self.expected_dispatch_reduction,
            "expected_intermediate_byte_reduction": self.expected_intermediate_byte_reduction,
            "expected_active_byte_change": self.expected_active_byte_change,
            "expected_gpu_ns_mechanism": self.expected_gpu_ns_mechanism,
            "parity_contract": self.parity_contract,
            "capability_contract": self.capability_contract,
            "diagnostic_command": list(self.diagnostic_command),
            "protected_command": list(self.protected_command),
            "dependencies": list(self.dependencies),
            "status": self.status,
            "scope_tags": list(self.scope_tags),
            "transfer_evidence": list(self.transfer_evidence),
            "source_evidence": list(self.source_evidence),
            "blocked_reason": self.blocked_reason,
            "evidence": list(self.evidence),
            "measurements": _copy(self.measurements),
        }


def _qwen_candidate(
    candidate_id: str,
    *,
    mutation: Mapping[str, str],
    control: Mapping[str, str],
    status: str,
    region: str,
    eliminated: str,
    dispatch: str,
    intermediate: str,
    active: str,
    gpu: str,
    dependencies: tuple[str, ...] = (),
    source_evidence: tuple[str, ...] = (),
    scope_tags: tuple[str, ...] | None = None,
    transfer_evidence: tuple[str, ...] | None = None,
) -> PhysicalCandidate:
    base = f"receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION/{candidate_id}"
    candidate_env = {str(key): str(value) for key, value in mutation.items()}
    return PhysicalCandidate(
        candidate_id=candidate_id,
        model="Qwen27",
        affected_physical_region=region,
        baseline_path="hcli/hawking-native.sealed-3.14.json",
        exact_mutation={"child_fusion_env": candidate_env},
        control_configuration={"child_fusion_env": dict(control)},
        expected_eliminated_work=eliminated,
        expected_dispatch_reduction=dispatch,
        expected_intermediate_byte_reduction=intermediate,
        expected_active_byte_change=active,
        expected_gpu_ns_mechanism=gpu,
        parity_contract=(
            "identical tokenizer-bound output ids and source-approved graph semantics; "
            "any numerical/output divergence rejects the candidate"
        ),
        capability_contract=(
            "complete accepted-token capability, zero fallback, independent resident "
            "identity, and explicit native metric absence reasons"
        ),
        diagnostic_command=_hcli_qwen_command(f"{base}-diagnostic.json", candidate_env, diagnostic=True),
        protected_command=_hcli_qwen_command(f"{base}-protected.json", candidate_env, diagnostic=False),
        dependencies=dependencies,
        status=status,
        scope_tags=scope_tags or _default_scope_tags(candidate_id),
        transfer_evidence=(
            _default_transfer_evidence(candidate_id)
            if transfer_evidence is None
            else transfer_evidence
        ),
        source_evidence=source_evidence,
    )


def _flash_candidate(
    candidate_id: str,
    *,
    mutation: Mapping[str, Any],
    control: Mapping[str, Any] | None = None,
    status: str,
    region: str,
    eliminated: str,
    dispatch: str,
    intermediate: str,
    active: str,
    gpu: str,
    blocked_reason: str,
    dependencies: tuple[str, ...] = (),
    source_evidence: tuple[str, ...] = (),
    scope_tags: tuple[str, ...] | None = None,
    transfer_evidence: tuple[str, ...] | None = None,
) -> PhysicalCandidate:
    base = f"receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION/{candidate_id}"
    command = _hcli_flash_plan_command(f"{base}-plan.json")
    return PhysicalCandidate(
        candidate_id=candidate_id,
        model="Flash",
        affected_physical_region=region,
        baseline_path="receipts/headless/FLASH_NEXT_NOETIC_EXECUTABLE.json",
        exact_mutation={"source_oracle_controls": _copy(mutation)},
        control_configuration={
            "mode": "source_oracle_or_scaffold",
            **({"source_oracle_controls": _copy(control)} if control is not None else {}),
        },
        expected_eliminated_work=eliminated,
        expected_dispatch_reduction=dispatch,
        expected_intermediate_byte_reduction=intermediate,
        expected_active_byte_change=active,
        expected_gpu_ns_mechanism=gpu,
        parity_contract=(
            "source organ/reference parity is required; no whole-model Flash parity "
            "or NX claim is implied"
        ),
        capability_contract=(
            "blocked until a complete source-independent NX executable and independent "
            "capability path exist"
        ),
        diagnostic_command=command,
        protected_command=command,
        dependencies=dependencies,
        status=status,
        scope_tags=scope_tags or _default_scope_tags(candidate_id),
        transfer_evidence=(
            _default_transfer_evidence(candidate_id)
            if transfer_evidence is None
            else transfer_evidence
        ),
        source_evidence=source_evidence,
        blocked_reason=blocked_reason,
    )


def frontier_candidates() -> list[PhysicalCandidate]:
    """Return the deliberately compact, current candidate frontier."""
    return [
        _qwen_candidate(
            "qwen27-fast-profile",
            mutation={"HAWKING_QWEN38_FAST": "1"},
            control={},
            status="READY_PROTECTED",
            region="complete Qwen27 resident token",
            eliminated="profile-selected fusion, geometry, and state-organ boundaries",
            dispatch="measure from current sealed control; no assumed count",
            intermediate="no representation change; measure transient movement",
            active="unchanged packed resident; measure actual reads/token",
            gpu="remove repeated dispatch/synchronization and select measured packed geometry",
            source_evidence=(
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "receipts/headless/QWEN27_MLP_PROTECTED_AB_AFTER_FLASH.json",
            ),
        ),
        _qwen_candidate(
            "qwen27-pipeline-state-elision",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_METAL_PIPELINE_STATE_ELISION": "1",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_METAL_PIPELINE_STATE_ELISION": "0",
            },
            status="READY_PROTECTED",
            region="Qwen27 shared Metal ordered/concurrent encoder state binding",
            eliminated="redundant host-side setComputePipelineState calls plus sticky-name heap churn for repeated kernels in one shared encoder",
            dispatch="0; command-buffer and dispatch topology are unchanged",
            intermediate="0; no buffers or representation are changed",
            active="unchanged resident bytes and reads",
            gpu="remove encoder-state submission ceremony only; arithmetic and kernel selection remain identical",
            dependencies=("qwen27-fast-profile",),
            source_evidence=("crates/hawking-core/src/metal/mod.rs",),
        ),
        _qwen_candidate(
            "qwen27-pipeline-cache-reuse",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_METAL_PIPELINE_CACHE_REUSE": "1",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_METAL_PIPELINE_CACHE_REUSE": "0",
            },
            status="READY_PROTECTED",
            region="Qwen27 resident per-token Metal pipeline lookup",
            eliminated="steady-state context-wide pipeline-cache lock acquisition for each distinct kernel in every token",
            dispatch="0; command-buffer and dispatch topology are unchanged",
            intermediate="0; no buffers or representation are changed",
            active="unchanged resident bytes and reads",
            gpu="move warmed pipeline handles across token command buffers so host lookup ceremony is removed without changing kernel arithmetic or binding identity",
            dependencies=("qwen27-fast-profile",),
            source_evidence=(
                "crates/hawking-core/src/metal/mod.rs",
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
            ),
        ),
        _qwen_candidate(
            "qwen27-encoder-label-elision",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_METAL_ENCODER_LABEL_ELISION": "1",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_METAL_ENCODER_LABEL_ELISION": "0",
            },
            status="READY_PROTECTED",
            region="Qwen27 shared Metal ordinary encoder labeling",
            eliminated="per-dispatch diagnostic setLabel calls when neither trace nor physical capture is active",
            dispatch="0; command-buffer and dispatch topology are unchanged",
            intermediate="0; no buffers or representation are changed",
            active="unchanged resident bytes and reads",
            gpu="remove ordinary encoder-label Objective-C ceremony from the fast path while retaining labels for trace and physical capture",
            dependencies=("qwen27-fast-profile",),
            source_evidence=("crates/hawking-core/src/metal/mod.rs",),
        ),
        _qwen_candidate(
            "qwen27-pipeline-id-resolution",
            mutation={
                "HAWKING_METAL_PIPELINE_CACHE_REUSE": "1",
                "HAWKING_METAL_PIPELINE_ID_RESOLUTION": "1",
                "HAWKING_QWEN38_FAST": "1",
            },
            control={
                "HAWKING_METAL_PIPELINE_CACHE_REUSE": "1",
                "HAWKING_METAL_PIPELINE_ID_RESOLUTION": "0",
                "HAWKING_QWEN38_FAST": "1",
            },
            status="READY_PROTECTED",
            region="Qwen27 resident per-dispatch pipeline handle resolution",
            eliminated="the second kernel-name hash/map probe after resident pipeline admission",
            dispatch="0; command-buffer, encoder, and dispatch topology are unchanged",
            intermediate="0; no buffers or representation are changed",
            active="unchanged resident bytes and reads",
            gpu="resolve warmed pipeline handles through stable per-TCB integer IDs and vector indexing while preserving the name-map control path",
            dependencies=("qwen27-pipeline-cache-reuse",),
            source_evidence=(
                "crates/hawking-core/src/metal/mod.rs",
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
            ),
        ),
        _qwen_candidate(
            "qwen27-commit-timing-elision",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_METAL_COMMIT_TIMING_ELISION": "1",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_METAL_COMMIT_TIMING_ELISION": "0",
            },
            status="READY_PROTECTED",
            region="Qwen27 uninstrumented TokenCommandBuffer commit/fence",
            eliminated="CPU clock reads, GPU timeline queries, and the post-fence status query on the no-timing commit API",
            dispatch="0; command-buffer, encoder, and dispatch topology are unchanged",
            intermediate="0; no buffers or representation are changed",
            active="unchanged resident bytes and reads",
            gpu="remove commit-side measurement ceremony only; the command-buffer fence and kernel arithmetic remain identical",
            dependencies=("qwen27-fast-profile",),
            source_evidence=("crates/hawking-core/src/metal/mod.rs",),
        ),
        _qwen_candidate(
            "qwen27-resident-untimed-decode",
            mutation={
                "HAWKING_METAL_COMMIT_TIMING_ELISION": "1",
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_QWEN38_SERVE_UNTIMED": "1",
            },
            control={
                "HAWKING_METAL_COMMIT_TIMING_ELISION": "1",
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_QWEN38_SERVE_UNTIMED": "0",
            },
            status="STATIC_ONLY",
            region="Genesis resident Qwen27 serving token loop",
            eliminated="per-token host timing clocks, timing-vector growth, and timestamp-query ceremony on the serving path",
            dispatch="0; token graph, command-buffer count, and sampling order are unchanged",
            intermediate="0; workspace, recurrent state, KV, and packed activations are unchanged",
            active="unchanged packed resident bytes; serving result intentionally omits per-token counters",
            gpu="reach the plain commit/fence path from the resident proposer while preserving the measured qualification API as a separate route",
            dependencies=("qwen27-commit-timing-elision",),
            source_evidence=(
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "tools/agentos/genesis_body/src/main.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _qwen_candidate(
            "qwen27-q4-vecgroup-x64",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_QWEN38_Q4_GEO": "vecgroup_x64",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_QWEN38_Q4_GEO": "tpr64",
            },
            status="READY_PROTECTED",
            region="Qwen27 standalone uniform-Q4 GEMV",
            eliminated="none; geometry-only candidate",
            dispatch="0; standalone dispatch count is unchanged",
            intermediate="0; packed Q4 buffers and output contract are unchanged",
            active="unchanged packed Q4 bytes; measure actual reads/token",
            gpu="retarget standalone group-64 Q4 GEMVs to the existing 64-thread-per-row vector-group tile; fused Q4 and affine-Q2 kernels remain on their incumbent bindings",
            dependencies=("qwen27-fast-profile",),
            source_evidence=(
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "crates/hawking-core/shaders/qwen_uniform_q4.metal",
                "receipts/ascent-2026-08-16/matvec-occupancy-230x.json",
            ),
        ),
        _qwen_candidate(
            "qwen27-affine2-splitk4",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_AFFINE2_GEO": "splitk4",
                "HAWKING_Q2F_GEO": "tpr64",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_AFFINE2_GEO": "tpr64",
                "HAWKING_Q2F_GEO": "tpr64",
            },
            status="READY_PROTECTED",
            region="Qwen27 HGRAVF01 affine Q2 GEMV",
            eliminated="none; geometry-only candidate",
            dispatch="0",
            intermediate="0",
            active="unchanged packed Q2; split-K changes concurrent reads only",
            gpu="four-way split-K increases row-local parallelism for wide projections",
            dependencies=("qwen27-fast-profile",),
            source_evidence=(
                "crates/hawking-core/shaders/q80_mixed_decode.metal",
                "receipts/headless/ACCELERATOR_KERNEL_BOTTLENECK.json",
            ),
        ),
        _qwen_candidate(
            "qwen27-affine2-splitk4-vec",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_AFFINE2_GEO": "splitk4_vec",
                "HAWKING_Q2F_GEO": "splitk4",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_AFFINE2_GEO": "splitk4",
                "HAWKING_Q2F_GEO": "splitk4",
            },
            status="STATIC_ONLY",
            region="Qwen27 HGRAVF01 affine Q2 GEMV",
            eliminated="scalar x loads in the eight-element dequant tile",
            dispatch="0",
            intermediate="0",
            active="unchanged packed Q2",
            gpu="float4 input loads plus one-tile packed/scale/bias lookahead; register pressure is the falsifier",
            dependencies=("qwen27-affine2-splitk4",),
            source_evidence=(
                "crates/hawking-core/shaders/q80_mixed_decode.metal::N035",
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs::Affine2Geo",
            ),
        ),
        _qwen_candidate(
            "qwen27-q2f-splitk4",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_AFFINE2_GEO": "tpr64",
                "HAWKING_Q2F_GEO": "splitk4",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_AFFINE2_GEO": "tpr64",
                "HAWKING_Q2F_GEO": "tpr64",
            },
            status="READY_PROTECTED",
            region="Qwen27 biasless Q2F projection family",
            eliminated="none; geometry-only candidate",
            dispatch="0",
            intermediate="0",
            active="unchanged packed Q2F; split-K changes concurrent reads only",
            gpu="four-way split-K increases row-local parallelism for wide Q2F projections",
            dependencies=("qwen27-fast-profile",),
            source_evidence=(
                "crates/hawking-core/shaders/q80_mixed_decode.metal",
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs::qwen38_q2f_geo_from_env",
            ),
        ),
        _qwen_candidate(
            "qwen27-q2f-splitk4-vec",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_AFFINE2_GEO": "tpr64",
                "HAWKING_Q2F_GEO": "splitk4_vec",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_AFFINE2_GEO": "tpr64",
                "HAWKING_Q2F_GEO": "splitk4",
            },
            status="STATIC_ONLY",
            region="Qwen27 biasless Q2F projection family",
            eliminated="scalar x loads in single and gate/up Q2F projections",
            dispatch="0",
            intermediate="0",
            active="unchanged packed Q2F",
            gpu="vectorized dequant tiles with prefetch across Q2F split-K rows",
            dependencies=("qwen27-q2f-splitk4",),
            source_evidence=(
                "crates/hawking-core/shaders/q80_mixed_decode.metal::N035",
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs::qwen38_q2f_matvec_launch",
            ),
        ),
        _qwen_candidate(
            "qwen27-attention-gate-fusion",
            mutation={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_QWEN38_FUSE_ATTENTION_GATE": "1",
            },
            control={
                "HAWKING_QWEN38_FAST": "1",
                "HAWKING_QWEN38_FUSE_ATTENTION_GATE": "0",
            },
            status="READY_PROTECTED",
            region="Qwen27 GQA attention output sigmoid gate",
            eliminated="one standalone attention-gate dispatch per GQA layer",
            dispatch="reduce by the number of GQA layers; verify trace",
            intermediate="remove gated attention output intermediate read/write",
            active="unchanged resident weights; lower transient activation traffic expected",
            gpu="fuse sigmoid gate into the final attention write without changing output order",
            dependencies=("qwen27-fast-profile",),
            source_evidence=("crates/hawking-core/src/model/qwen38_hybrid_decode.rs",),
        ),
        _qwen_candidate(
            "qwen27-gqa-qkv-fusion",
            mutation={"HAWKING_QWEN38_FAST": "1", "HAWKING_QWEN38_FUSE_GQA_QKV": "1"},
            control={"HAWKING_QWEN38_FAST": "1", "HAWKING_QWEN38_FUSE_GQA_QKV": "0"},
            status="READY_PROTECTED",
            region="Qwen27 GQA Q/K/V packed projection",
            eliminated="two projection dispatch boundaries per GQA layer",
            dispatch="reduce by two per GQA layer; verify trace",
            intermediate="remove separate Q/K/V projection staging where fused contract permits",
            active="unchanged packed weights",
            gpu="shared input traversal and one command boundary for Q/K/V outputs",
            dependencies=("qwen27-fast-profile",),
            source_evidence=("crates/hawking-core/src/model/qwen38_hybrid_decode.rs",),
        ),
        _qwen_candidate(
            "qwen27-deltanet-inproj-fusion",
            mutation={"HAWKING_QWEN38_FAST": "1", "HAWKING_QWEN38_FUSE_DN_INPROJ": "1"},
            control={"HAWKING_QWEN38_FAST": "1", "HAWKING_QWEN38_FUSE_DN_INPROJ": "0"},
            status="READY_PROTECTED",
            region="Qwen27 DeltaNet input projection",
            eliminated="one DeltaNet input-projection dispatch per recurrent layer",
            dispatch="reduce by the number of DeltaNet layers; verify trace",
            intermediate="remove fused in-projection staging if parity holds",
            active="unchanged packed weights",
            gpu="fuse the shared projection producer with recurrent input preparation",
            dependencies=("qwen27-fast-profile",),
            source_evidence=(
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "receipts/headless/DELTANET_ORGAN.json",
            ),
        ),
        _qwen_candidate(
            "qwen27-ba-delta-fusion",
            mutation={"HAWKING_QWEN38_FAST": "1", "HAWKING_QWEN38_FUSE_BA_DELTA": "1"},
            control={"HAWKING_QWEN38_FAST": "1", "HAWKING_QWEN38_FUSE_BA_DELTA": "0"},
            status="READY_PROTECTED",
            region="Qwen27 DeltaNet BA/state transition",
            eliminated="one standalone BA/delta state dispatch per recurrent layer",
            dispatch="reduce by the number of fused recurrent layers; verify trace",
            intermediate="remove BA/delta staging only if state parity is exact",
            active="unchanged resident recurrent state and weights",
            gpu="fuse recurrent state update with BA transition and keep state resident",
            dependencies=("qwen27-fast-profile",),
            source_evidence=(
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "receipts/headless/QWEN27_MLP_PROTECTED_AB_AFTER_FLASH.json",
            ),
        ),
        _flash_candidate(
            "flash-hc-staged-threadgroup",
            mutation={"HAWKING_FLASH_HC_STAGED": "1"},
            status="BLOCKED",
            region="Flash HyperConnection shared reduction",
            eliminated="repeated 2,560-wide norm recomputation per output element",
            dispatch="0; organ-level staging candidate",
            intermediate="remove redundant HC norm reads/writes",
            active="unchanged source BF16 footprint; NX active bytes unqualified",
            gpu="one threadgroup reduction consumed by all HC output streams",
            blocked_reason="Flash complete source-independent NX executable is not qualified",
            source_evidence=(
                "crates/hawking-core/examples/flash_full_attention_layer3.rs",
                "receipts/headless/FLASH_FUSED_TIMING_L3_SERIAL_DIRECT.json",
            ),
        ),
        _flash_candidate(
            "flash-p7-mhc-pre-simdgroup",
            mutation={"HAWKING_DSV4F_MHC_PRE_SIMD": "1"},
            control={"HAWKING_DSV4F_MHC_PRE_SIMD": "0"},
            status="BLOCKED",
            region="Flash P4B/P7 mHC-pre wide reduction",
            eliminated="one-thread serialization across the 24 source mHC rows and 16,384-wide dot products in the bounded P4B attention and P7 FFN graphs",
            dispatch="0; the shared mHC-pre remains one dispatch with a 24-SIMDgroup threadgroup",
            intermediate="0; the existing reduced BF16 and mHC diagnostic buffers are retained",
            active="unchanged source BF16 residual and F32 control bytes; NX active bytes unqualified",
            gpu="assign one 24-SIMDgroup threadgroup to the shared P4B/P7 mHC-pre rows, stage the RMS partials, and leave the nonlinear Sinkhorn/control grammar source-ordered",
            blocked_reason="Flash complete source-independent NX is not qualified and the SIMD mHC reduction has no bounded parity/coherence receipt",
            source_evidence=(
                "crates/hawking-core/shaders/deepseek_v4_p7.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p4b_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p7_device.rs",
                "workspace/docs/plans/KERNEL_BROKERS_TUNING_PLAN.md",
            ),
        ),
        _flash_candidate(
            "flash-device-mhc-state",
            mutation={
                "HAWKING_DSV4F_DEVICE_MHC": "1",
                "HAWKING_DSV4F_MHC_NORM_SIMD": "1",
            },
            control={
                "HAWKING_DSV4F_DEVICE_MHC": "0",
                "HAWKING_DSV4F_MHC_NORM_SIMD": "0",
            },
            status="BLOCKED",
            region="Flash native token graph device-resident mHC state boundary",
            eliminated="two HIDDEN_SIZE BF16 activation readbacks, host mHC pre/post transforms, and two serialized HIDDEN_SIZE RMSNorm passes at every native layer",
            dispatch="adds six exact P7 state dispatches per layer but removes the host-visible activation handoff boundaries; no GPU dispatch reduction is claimed",
            intermediate="remove per-layer host activation vectors and the BF16 readback bridge; retain persistent four-lane HC state and one final report/head readback",
            active="adds bounded persistent device mHC control/state scratch only when enabled; the default path keeps one-byte dormant placeholders",
            gpu="run the exact P7 mHC pre/norm/post stages with separate attention/FFN control slots, parallelize both 4096-wide RMSNorm boundaries, ping-pong HC state across layers, and keep the state resident through the P6 MoE tail",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified; device mHC still requires matched parity/coherence and protected latency",
            dependencies=("flash-p7-mhc-pre-simdgroup",),
            source_evidence=(
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/shaders/deepseek_v4_p7.metal",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-hash-single-command-buffer",
            mutation={"HAWKING_DSV4F_P6_SINGLE_CB": "1"},
            control={"HAWKING_DSV4F_P6_SINGLE_CB": "0"},
            status="BLOCKED",
            region="Flash P6 command-buffer topology after route resolution",
            eliminated="one CPU-visible commit/wait between the device-resident up/SwiGLU wave and down/combine wave; on learned layers the same switch removes the post-residency phase fence",
            dispatch="hash path: 1 command buffer and wait instead of historical 2; learned path: 2 instead of 3; 60 dispatches remain unchanged",
            intermediate="0; all routed/shared BF16 and route-control buffers remain resident",
            active="unchanged source BF16/FP4/FP8 bytes; NX active bytes unqualified",
            gpu="append dependency-ordered down/combine waves to the first P6 command buffer while preserving the route/readback residency boundary, explicit wave boundaries, and Metal hazards",
            blocked_reason="Flash complete source-independent NX is not qualified and the single-command-buffer path has no matched parity/coherence and protected-latency receipt",
            source_evidence=(
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/metal/mod.rs",
                "workspace/docs/plans/KERNEL_BROKERS_TUNING_PLAN.md",
            ),
        ),
        _flash_candidate(
            "flash-p6-prefix-concurrent-wave",
            mutation={"HAWKING_DSV4F_P6_PREFIX_CONCURRENT": "1"},
            control={"HAWKING_DSV4F_P6_PREFIX_CONCURRENT": "0"},
            status="BLOCKED",
            region="Flash P6 Gate and activation-quantization prefix",
            eliminated="one compute-encoder boundary between the independent Gate reduction and activation quantization",
            dispatch="0; Gate, QAT, and route dispatches remain unchanged",
            intermediate="0; Gate logits and E4M3FN/UE8M0 quantized buffers remain separate and source-visible",
            active="unchanged source BF16 input, Gate BF16 weights, and quantizer output bytes",
            gpu="place Gate and activation quantization in one concurrent encoder because both only read the shared input and write disjoint outputs, then retain a separate route dependency boundary",
            blocked_reason="Flash complete source-independent NX is not qualified and the shared prefix has no matched parity/coherence and protected-latency receipt",
            source_evidence=(
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-act-quant-simdgroup",
            mutation={"HAWKING_DSV4F_P6_ACT_QUANT_SIMD": "1"},
            control={"HAWKING_DSV4F_P6_ACT_QUANT_SIMD": "0"},
            status="BLOCKED",
            region="Flash P6 BF16-to-E4M3FN activation quantization",
            eliminated="one serial thread per 128-wide activation block across the routed/shared MoE projections",
            dispatch="0; the quantizer remains one dispatch per source activation, with wider block ownership",
            intermediate="0; quantized bytes and UE8M0 scale bytes retain the existing buffers",
            active="unchanged source BF16 reads and E4M3FN/UE8M0 output bytes; NX active bytes unqualified",
            gpu="one SIMD-group owns each 128-wide block with packed BF16/uchar4 loads and the same finite-table RNE encoding",
            blocked_reason="Flash complete source-independent NX is not qualified and the P6 quantizer candidate has no byte-exact parity/coherence and protected-latency receipt",
            source_evidence=(
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "workspace/docs/plans/KERNEL_BROKERS_TUNING_PLAN.md",
            ),
        ),
        _flash_candidate(
            "flash-p6-routed-fp4-simdgroup",
            mutation={"HAWKING_DSV4F_P6_FP4_SIMD": "1"},
            control={"HAWKING_DSV4F_P6_FP4_SIMD": "0"},
            status="BLOCKED",
            region="Flash P6 routed-expert FP4 matvec",
            eliminated="one serial thread per routed-expert output row across the 18 W1/W3/W2 FP4 projections per full MoE layer",
            dispatch="0; six expert waves remain concurrent and the 18 matvec dispatches remain unchanged",
            intermediate="0; packed FP4 weights, E8M0 scales, activation bytes, and output buffers retain their existing ABI",
            active="unchanged source FP4/FP8 bytes; NX active bytes unqualified",
            gpu="one 64-lane-x-4-row threadgroup uses packed uchar4 loads and SIMDgroup split-K partials before a deterministic row reduction",
            blocked_reason="Flash complete source-independent NX is not qualified and the P6 FP4 candidate has no NumericParity/coherence and protected-latency receipt",
            source_evidence=(
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "workspace/docs/plans/KERNEL_BROKERS_TUNING_PLAN.md",
            ),
        ),
        _flash_candidate(
            "flash-p6-shared-fp8-simdgroup",
            mutation={"HAWKING_DSV4F_P6_FP8_SIMD": "1"},
            control={"HAWKING_DSV4F_P6_FP8_SIMD": "0"},
            status="BLOCKED",
            region="Flash P6 shared-expert FP8 matvec",
            eliminated="one serial thread per shared-expert output row across the three W1/W3/W2 FP8 projections per full MoE layer",
            dispatch="0; the shared W1/W3/W2 matvec dispatches remain unchanged",
            intermediate="0; FP8 weights, E8M0 scales, activation bytes, and output buffers retain their existing ABI",
            active="unchanged source FP8 bytes; NX active bytes unqualified",
            gpu="one 256-threadgroup assigns eight SIMDgroups to the source 128-wide blocks and combines block partials in source block order",
            blocked_reason="Flash complete source-independent NX is not qualified and the P6 FP8 candidate has no NumericParity/coherence and protected-latency receipt",
            source_evidence=(
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "workspace/docs/plans/KERNEL_BROKERS_TUNING_PLAN.md",
            ),
        ),
        _flash_candidate(
            "flash-fullseq-ordered-encoder",
            mutation={"HAWKING_DSV4F_FULLSEQ_ORDERED_ENCODER": "1"},
            status="BLOCKED",
            region="Flash fullseq attention command-encoder topology",
            eliminated="per-dispatch compute-encoder creation across the 22/25-dispatch dependent chain",
            dispatch="0; dispatch count is intentionally unchanged",
            intermediate="0; producer/consumer buffers remain resident within the same command buffer",
            active="unchanged source BF16 footprint; compact NX bytes unqualified",
            gpu="one ordered encoder removes host/driver encoder ceremony while preserving dispatch order",
            blocked_reason="fullseq source path is not a complete source-independent Flash NX executable",
            source_evidence=(
                "crates/hawking-core/src/gravity_deepseek_v4_fullseq_attention_device.rs",
                "crates/hawking-core/examples/gravity_deepseek_v4_fullseq_capture.rs",
            ),
        ),
        _flash_candidate(
            "flash-pipeline-cache-reuse",
            mutation={"HAWKING_FLASH_PIPELINE_CACHE_REUSE": "1"},
            control={"HAWKING_FLASH_PIPELINE_CACHE_REUSE": "0"},
            status="BLOCKED",
            region="Flash native/fullseq/P6 per-batch Metal pipeline lookup",
            eliminated="steady-state context-wide pipeline-cache lock acquisition for each distinct kernel in every native organ batch, repeated fullseq attention position, and adjacent P6 MoE batch",
            dispatch="0; command-buffer and dispatch topology are unchanged",
            intermediate="0; no buffers or representation are changed",
            active="unchanged source and compact resident bytes",
            gpu="move warmed pipeline handles across native graph batches, repeated fullseq positions, and the four P6 MoE batches so host lookup ceremony is removed without changing kernel arithmetic or binding identity",
            blocked_reason="source-independent Flash NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/src/metal/mod.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_fullseq_attention_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
            ),
        ),
        _flash_candidate(
            "flash-encoder-label-elision",
            mutation={"HAWKING_METAL_ENCODER_LABEL_ELISION": "1"},
            status="BLOCKED",
            region="Flash native/fullseq ordinary Metal encoder labeling",
            eliminated="per-dispatch diagnostic setLabel calls when neither trace nor physical capture is active",
            dispatch="0; command-buffer and dispatch topology are unchanged",
            intermediate="0; no device buffers or representation are changed",
            active="unchanged source and compact resident bytes",
            gpu="remove ordinary encoder-label Objective-C ceremony while retaining labels for trace and physical capture",
            blocked_reason="source-independent Flash NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/src/metal/mod.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_fullseq_attention_device.rs",
            ),
        ),
        _flash_candidate(
            "flash-pipeline-id-resolution",
            mutation={
                "HAWKING_FLASH_PIPELINE_CACHE_REUSE": "1",
                "HAWKING_METAL_PIPELINE_ID_RESOLUTION": "1",
            },
            control={
                "HAWKING_FLASH_PIPELINE_CACHE_REUSE": "1",
                "HAWKING_METAL_PIPELINE_ID_RESOLUTION": "0",
            },
            status="BLOCKED",
            region="Flash native/fullseq resident pipeline handle resolution",
            eliminated="the second kernel-name hash/map probe after resident pipeline admission",
            dispatch="0; command-buffer, encoder, and dispatch topology are unchanged",
            intermediate="0; no buffers or representation are changed",
            active="unchanged source and compact resident bytes",
            gpu="resolve warmed pipeline handles through stable batch integer IDs and vector indexing without changing kernel selection or arithmetic",
            blocked_reason="source-independent Flash NX and protected complete-token capability are not qualified",
            dependencies=("flash-pipeline-cache-reuse",),
            source_evidence=(
                "crates/hawking-core/src/metal/mod.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_fullseq_attention_device.rs",
            ),
        ),
        _flash_candidate(
            "flash-fullseq-catalog-cache",
            mutation={"HAWKING_DSV4F_FULLSEQ_CATALOG_CACHE": "1"},
            status="BLOCKED",
            region="Flash fullseq source-anchor admission between positions",
            eliminated="repeated immutable 43-layer source-anchor plus tokenizer/config verification, bounded RoPE-table trigonometry, and per-position static RoPE uploads",
            dispatch="0; command-buffer and dispatch topology are unchanged",
            intermediate="0; device buffers and position-dependent state are unchanged",
            active="unchanged source and compact resident bytes",
            gpu="reuse the prepared source-bound layer catalog and static RoPE controls while retaining the exact manifest hash guard and position admission checks",
            blocked_reason="source-independent Flash NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/src/gravity_deepseek_v4_fullseq_attention_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_layer_plan.rs",
                "crates/hawking-core/src/gravity_deepseek_v4.rs",
            ),
        ),
        _flash_candidate(
            "flash-qkv-gqa-rope-fusion",
            mutation={"HAWKING_FLASH_QKV_GQA_FUSED": "1"},
            status="BLOCKED",
            region="Flash source-BF16 Q/K/V projection and Q/K norm-RoPE-KV boundary",
            eliminated="one separate Q/K/V projection plus Q/K norm-RoPE/cache command boundary per full-attention organ",
            dispatch="reduce the attention projection path from two launches to one; verify trace",
            intermediate="retain raw Q/K/V diagnostic buffers while removing the producer-consumer launch boundary",
            active="unchanged source BF16 weights, raw projection buffers, and KV-cache row bytes",
            gpu="one head-local launch shares input traversal, writes projection diagnostics and cache, and keeps exact sequential norm reductions",
            blocked_reason="source-independent Flash NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/shaders/qwen_next.metal::qwen_next_bf16_qkv_gqa_rope_cache",
                "crates/hawking-core/src/kernels/mod.rs",
                "crates/hawking-core/examples/flash_full_attention_layer3.rs",
            ),
        ),
        _flash_candidate(
            "flash-source-bf16-simd",
            mutation={"HAWKING_FLASH_BF16_GEO": "1", "HAWKING_FLASH_BF16_VEC4": "1"},
            status="BLOCKED",
            region="Flash source-BF16 projection oracle",
            eliminated="scalar source-BF16 loads and repeated input traversal",
            dispatch="0",
            intermediate="measure source oracle staging; no NX claim",
            active="source BF16 active bytes unchanged; compact NX bytes unqualified",
            gpu="SIMD-group/vector load geometry for oracle throughput",
            blocked_reason="source oracle is a control; Flash NX and protected complete-token path are open",
            source_evidence=(
                "crates/hawking-core/examples/flash_noetic_complete_layer0.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_streamed_native.rs",
                "crates/hawking-core/src/kernels/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-compact-moe-bf16-vec4",
            mutation={"HAWKING_FLASH_MOE_VEC4": "1"},
            status="BLOCKED",
            region="Flash compact source-BF16 routed/shared MoE epilogue",
            eliminated="scalar BF16/FP32 load and pointer overhead in compact gate/up and direct down-to-HC reductions",
            dispatch="0; gate/up and down-to-HC topology is unchanged",
            intermediate="0; diagnostic outputs and the resident activation contract are unchanged",
            active="unchanged compact source-BF16 weights and activation bytes",
            gpu="exact-order vec4 loads preserve scalar accumulation while improving contiguous source traffic",
            blocked_reason="source-independent Flash NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/shaders/qwen_next.metal",
                "crates/hawking-core/src/kernels/mod.rs",
                "crates/hawking-core/src/metal/mod.rs",
                "crates/hawking-core/examples/flash_noetic_complete_layer0.rs",
                "crates/hawking-core/examples/flash_full_attention_layer3.rs",
            ),
        ),
        _flash_candidate(
            "flash-routed-fp4-gate-up-swiglu-fused",
            mutation={"HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1"},
            status="BLOCKED",
            region="Flash routed FP4 gate/up/SwiGLU epilogue",
            eliminated="two FP32-to-BF16 staging dispatches and one standalone routed SwiGLU dispatch per layer",
            dispatch="reduce the routed gate/up/SwiGLU sequence from five dispatches to one",
            intermediate="remove routed gate/up FP32 and BF16 staging; retain the BF16 SwiGLU output contract",
            active="unchanged FP4 weight and activation reads; lower transient intermediate footprint",
            gpu="pair the two source-order FP4 reductions and consume their explicit BF16 round-trips in one kernel",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/shaders/gk_family.metal",
                "crates/hawking-core/shaders/dsv4f_native_token_graph.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/src/decode_family.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-routed-fp4-gate-up-swiglu-fused",
            mutation={
                "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1",
                "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "0",
            },
            control={
                "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "0",
                "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "0",
            },
            status="BLOCKED",
            region="Flash P6A fixed-six routed FP4 gate/up/SwiGLU epilogue",
            eliminated="six routed W1 launches, six routed W3 launches, twelve routed FP32-to-BF16 casts, and six routed SwiGLU launches",
            dispatch="reduce P6 batch 1 from 38 to 9 and the fixed hash graph from 60 to 31 dispatches",
            intermediate="routed gate/up FP32 and BF16 staging is replaced by one-byte guarded placeholders; the live per-expert BF16 SwiGLU output contract remains explicit",
            active="unchanged six routed FP4 weight/scale buffers and activation bytes; inactive gate/up scratch allocation is reduced by 147432 structural bytes",
            gpu="one fixed-six indirect-address launch performs paired source-order FP4 reductions, exact BF16 round-trips, clamp/SwiGLU, and device route weighting; explicit read/write residency covers every indirect buffer",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/shaders/moe.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-routed-fp4-gate-up-swiglu-simd",
            mutation={
                "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1",
                "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "1",
            },
            control={
                "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1",
                "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "0",
            },
            status="BLOCKED",
            region="Flash P6A fused routed FP4 gate/up/SwiGLU occupancy",
            eliminated="the same routed P6 epilogue dispatch and staging work as the scalar-fused candidate while changing only row reduction geometry",
            dispatch="0 relative to scalar fusion; one fixed-six launch remains",
            intermediate="0 relative to scalar fusion; per-expert BF16 output pointers and source route slots remain unchanged",
            active="unchanged FP4 source bytes and routed output bytes",
            gpu="eight SIMDgroups per threadgroup split each 32-value FP4 block across lanes, with an explicit independent A/B against scalar fusion",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            dependencies=("flash-p6-routed-fp4-gate-up-swiglu-fused",),
            source_evidence=(
                "crates/hawking-core/shaders/moe.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-routed-fp4-down-bf16-fused",
            mutation={"HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "1"},
            control={"HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "0"},
            status="BLOCKED",
            region="Flash P6A routed FP4 W2/down projection and BF16 boundary",
            eliminated="six routed W2 launches and six routed FP32-to-BF16 cast launches",
            dispatch="reduce the fixed hash graph from 60 to 49 dispatches; P6 batch 2 falls from 22 to 11",
            intermediate="routed down FP32 staging is bypassed; the existing six per-expert BF16 outputs remain the combine contract",
            active="unchanged six routed FP4 W2 weight/scale buffers, per-expert QAT outputs, and final BF16 combine bytes; inactive routed down_f32 scratch is reduced by 98298 structural bytes",
            gpu="one fixed-six indirect launch consumes the already-authoritative per-expert E4M3FN/UE8M0 QAT buffers, preserves FP4 block accumulation, and writes exact BF16 W2 outputs",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/shaders/moe.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-routed-fp4-down-bf16-simd",
            mutation={
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "1",
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_SIMD": "1",
            },
            control={
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "1",
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_SIMD": "0",
            },
            status="BLOCKED",
            region="Flash P6A fused routed FP4 W2/down occupancy",
            eliminated="the same six routed W2 launches and six routed FP32-to-BF16 cast launches as scalar fusion while changing only row-reduction geometry",
            dispatch="0 relative to scalar fusion; one fixed-six indirect launch remains",
            intermediate="0 relative to scalar fusion; existing six per-expert BF16 combine buffers remain the contract",
            active="unchanged routed FP4 W2 weights, E8M0 scales, per-expert QAT bytes, and BF16 output bytes",
            gpu="eight SIMDgroups per 256-threadgroup split each 32-value FP4 block across lanes, preserving source block order while isolating intra-block reduction association for an A/B",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            dependencies=("flash-p6-routed-fp4-down-bf16-fused",),
            source_evidence=(
                "crates/hawking-core/shaders/moe.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-learned-reader-reuse",
            mutation={"HAWKING_DSV4F_P6_LEARNED_READER_REUSE": "1"},
            control={"HAWKING_DSV4F_P6_LEARNED_READER_REUSE": "0"},
            status="BLOCKED",
            region="Flash P6 learned-route source admission boundary",
            eliminated="repeated sealed-reader manifest/index admission and metadata-map construction on each changed learned route",
            dispatch="0; device dispatch topology and the selected-ID residency boundary are unchanged",
            intermediate="0; the candidate retains a metadata-only reader and still creates a fresh bounded six-expert cache",
            active="unchanged source FP4/FP8 and activation bytes; no decoded weights are retained by the reader",
            gpu="remove repeated host admission ceremony before the second GPU wave while preserving the manifest-seal guard, source reads, uploads, and route-ID readback",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_expert_cache.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-learned-expert-cache-reuse",
            mutation={
                "HAWKING_DSV4F_P6_LEARNED_READER_REUSE": "1",
                "HAWKING_DSV4F_P6_LEARNED_EXPERT_CACHE_REUSE": "1",
            },
            control={
                "HAWKING_DSV4F_P6_LEARNED_READER_REUSE": "1",
                "HAWKING_DSV4F_P6_LEARNED_EXPERT_CACHE_REUSE": "0",
            },
            status="BLOCKED",
            region="Flash P6 learned-route bounded expert source cache",
            eliminated="repeated source chunk materialization and GPU re-upload of expert bundles that overlap successive learned routes",
            dispatch="0; device dispatch topology, route-ID readback, and GPU upload boundaries are unchanged",
            intermediate="retain at most the exact six-bundle hot capacity plus the same six-entry GPU weight handle cache; pre-route FP4 placeholders are one byte per pointer component, and no dense or decoded representation is added",
            active="unchanged source FP4/FP8 and activation bytes; overlapping routes reuse existing GPU weight buffers rather than uploading a second copy",
            gpu="avoid repeated host source reads and GPU buffer uploads before the expert wave when route overlap exists; route overlap, allocation bytes, and complete-token wall time are the falsifiers",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            dependencies=("flash-p6-learned-reader-reuse",),
            source_evidence=(
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_expert_cache.rs",
                "crates/hawking-core/src/gravity_deepseek_v4.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-batched-down-qat",
            mutation={"HAWKING_DSV4F_P6_BATCHED_DOWN_QAT": "1"},
            control={"HAWKING_DSV4F_P6_BATCHED_DOWN_QAT": "0"},
            status="BLOCKED",
            region="Flash P6 fixed-seven down activation quantization wave",
            eliminated="six routed and one shared independent BF16-to-E4M3FN/E8M0 activation-quantization launch boundaries",
            dispatch="replace seven down-QAT dispatches with one fixed-seven indirect dispatch; P6 structural topology falls from 60 to 54 while compute-encoder count is unchanged",
            intermediate="same seven quantized activation buffers and seven E8M0FNU scale buffers remain the downstream ABI; only pointer binding is packed",
            active="unchanged BF16 SwiGLU inputs, E4M3FN quantized outputs, and E8M0FNU scale bytes",
            gpu="one logical source 128-value block per thread reads one of six routed or one shared scratch tensor through a fixed indirect record; block arithmetic and byte order match the authority quantizer",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-shared-fp8-gate-up-swiglu-fused",
            mutation={"HAWKING_DSV4F_FP8_SHARED_GATE_UP_SWIGLU_FUSED": "1"},
            control={"HAWKING_DSV4F_FP8_SHARED_GATE_UP_SWIGLU_FUSED": "0"},
            status="BLOCKED",
            region="Flash shared FP8 gate/up/SwiGLU epilogue",
            eliminated="two shared FP8 matvec dispatches, two FP32-to-BF16 staging dispatches, and one standalone shared SwiGLU dispatch per layer",
            dispatch="reduce the shared gate/up/SwiGLU sequence from five dispatches to one",
            intermediate="remove shared gate/up FP32 and BF16 staging traffic; inactive shared scratch is replaced by one-byte guarded placeholders",
            active="unchanged FP8 weight and activation reads; inactive shared gate/up scratch allocation is reduced by 24572 structural bytes",
            gpu="pair the two source-order FP8 reductions and consume their explicit BF16 round-trips in one kernel",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-shared-fp8-down-combine-fused",
            mutation={"HAWKING_DSV4F_FP8_SHARED_DOWN_COMBINE_FUSED": "1"},
            control={"HAWKING_DSV4F_FP8_SHARED_DOWN_COMBINE_FUSED": "0"},
            status="BLOCKED",
            region="Flash shared FP8 W2/down and fixed-six combine boundary",
            eliminated="one shared FP8 W2 dispatch, one shared FP32-to-BF16 cast dispatch, and one standalone routed/shared combine dispatch per layer",
            dispatch="reduce the shared W2/cast/combine sequence from three dispatches to one; P6 hash topology falls from 60 to 58",
            intermediate="remove shared down FP32 and BF16 staging from the fused path; routed BF16 outputs and final BF16 output remain the source contract",
            active="unchanged shared FP8 W2 weights, E8M0 scales, shared activation-quantized bytes, and six routed BF16 output rows; inactive shared down scratch is reduced by 24574 structural bytes",
            gpu="compute the source-order shared FP8 W2 row, explicitly round-trip it through BF16, then add the six routed BF16 rows in numeric-expert order before the final BF16 store",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified",
            source_evidence=(
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-fused-down-shared-combine",
            mutation={
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "1",
                "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED": "1",
            },
            control={
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "0",
                "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED": "0",
            },
            status="BLOCKED",
            region="Flash P6 routed FP4 W2 plus shared FP8 W2 and BF16 combine",
            eliminated=(
                "six routed FP4 W2 dispatches, one shared FP8 W2 dispatch, seven "
                "FP32-to-BF16 cast dispatches, one final combine dispatch, and the "
                "routed/shared down intermediates"
            ),
            dispatch=(
                "replace the fifteen downstream W2/cast/combine dispatches with one "
                "source-order fused launch; fixed-hash P6 structural topology falls "
                "from 60 to 46 while the seven authoritative QAT launches remain"
            ),
            intermediate=(
                "bypass six routed down_f32/down_bf16 buffers and the shared down "
                "FP32/BF16 staging at the consumer; inactive authority scratch is "
                "one-byte guarded storage, while QAT bytes and final caller-owned "
                "BF16 output remain explicit"
            ),
            active=(
                "unchanged routed FP4 weights, shared FP8 weights, E8M0 scales, and "
                "seven activation-quantized input buffers; inactive scratch allocation "
                "is reduced by 344022 structural bytes and transient down traffic is reduced"
            ),
            gpu=(
                "one 256-thread row launch reads six resident indirect FP4 records, "
                "computes source-order FP4 rows with BF16 round-trips, computes the "
                "shared FP8 row with 128-value blocks, then performs the fixed-six "
                "numeric-order BF16 combine directly into output"
            ),
            blocked_reason=(
                "Flash source-independent NX and protected complete-token capability "
                "are not qualified; this full composition still requires matched "
                "parity/coherence and protected latency"
            ),
            dependencies=(
                "flash-p6-routed-fp4-down-bf16-fused",
                "flash-shared-fp8-down-combine-fused",
            ),
            source_evidence=(
                "crates/hawking-core/shaders/moe.metal",
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-p6-fused-epilogue-stack",
            mutation={
                "HAWKING_DSV4F_P6_SINGLE_CB": "1",
                "HAWKING_DSV4F_P6_PREFIX_CONCURRENT": "1",
                "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1",
                "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "0",
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "1",
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_SIMD": "0",
                "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED": "1",
                "HAWKING_DSV4F_FP8_SHARED_GATE_UP_SWIGLU_FUSED": "1",
                "HAWKING_DSV4F_P6_BATCHED_DOWN_QAT": "1",
            },
            control={
                "HAWKING_DSV4F_P6_SINGLE_CB": "0",
                "HAWKING_DSV4F_P6_PREFIX_CONCURRENT": "0",
                "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "0",
                "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "0",
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "0",
                "HAWKING_DSV4F_P6_FP4_DOWN_BF16_SIMD": "0",
                "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED": "0",
                "HAWKING_DSV4F_FP8_SHARED_GATE_UP_SWIGLU_FUSED": "0",
                "HAWKING_DSV4F_P6_BATCHED_DOWN_QAT": "0",
            },
            status="BLOCKED",
            region="Flash reusable P6 composed latency stack",
            eliminated="the independently bounded routed/shared gate-up epilogues, seven-to-one down activation-quantization wave, full routed FP4/shared FP8 down-and-combine tail, one prefix encoder, and one hash-route command-buffer wait",
            dispatch="P6 structural target is 60 to 7 dispatches, 10 to 4 compute encoders, and 2 to 1 command buffer on the hash path; the fixed-seven QAT producer and full down fusion share one serial dependency encoder, so measure interaction rather than summing isolated wins",
            intermediate="retain only the output buffers required by the fixed-six combine contract; the full down fusion bypasses routed/shared down staging while dormant authority scratch remains allocated for matched A/B compatibility",
            active="unchanged source FP4/FP8 weights, E8M0 scales, route IDs/weights, and BF16 final output contract",
            gpu="compose the existing guarded primitives with explicit dependency waves; no new arithmetic or representation is introduced by the stack row",
            blocked_reason="Flash source-independent NX and protected complete-token capability are not qualified; composition interactions require a fresh parity/coherence and latency receipt",
            dependencies=(
                "flash-p6-hash-single-command-buffer",
                "flash-p6-prefix-concurrent-wave",
                "flash-p6-routed-fp4-gate-up-swiglu-fused",
                "flash-p6-fused-down-shared-combine",
                "flash-shared-fp8-gate-up-swiglu-fused",
                "flash-p6-batched-down-qat",
            ),
            source_evidence=(
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/shaders/moe.metal",
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/metal/mod.rs",
            ),
        ),
        _flash_candidate(
            "flash-meta-sub1-coherent",
            mutation={
                "HAWKING_FLASH_META_REPRESENTATION": "teacher_distilled_sub1_v1",
                "HAWKING_FLASH_META_BPW_TARGET": "0.8871807728336929",
                "HAWKING_FLASH_META_ROUTER_GUARD": "exact",
                "HAWKING_FLASH_META_DENSE_REMATERIALIZE": "0",
            },
            status="BLOCKED",
            region="Flash whole-model functional representation (routed experts + n-gram bank)",
            eliminated="dense weight storage and dense rematerialization in the dominant routed-expert and n-gram families",
            dispatch="not countable until a source-independent meta-NX consumer exists; no reduction claimed",
            intermediate="target zero dense rematerialization; measure generated-tile and n-gram staging explicitly",
            active="prospective meta_bpw 0.8871807728336929; physical active bytes remain unqualified",
            gpu="resident shared decoder, route-before-payload, fused generated-tile MoE accumulation, and generated n-gram lookup must beat a matched control",
            blocked_reason="meta budget is only a prospective function description; no serialized functional artifact, source-independent Flash NX consumer, or protected complete-token path exists",
            source_evidence=(
                "tools/flash_meta_representation.py",
                "tools/flash_meta_coherence_screen.py",
                "crates/hawking-core/examples/flash_meta_teacher_trace.rs",
                "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json",
                "receipts/headless/FLASH_META_TEACHER_L4.json",
                "receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
                "receipts/headless/FLASH_META_COHERENCE_SCREEN_L4.json",
                "receipts/headless/FLASH_ORGAN_CENSUS.json",
                "receipts/headless/FLASH_ROUTE_CONDITIONED_SHARED_BASIS_L4.json",
                "receipts/headless/FLASH_DOCTOR_NGRAM_SCREEN.json",
                "receipts/headless/FLASH_COMPLETE_V0.BYTE_LEDGER.json",
            ),
        ),
        _flash_candidate(
            "flash-router-topk-fusion",
            mutation={"HAWKING_FLASH_ROUTER_TOPK_FUSED": "1"},
            control={"HAWKING_FLASH_ROUTER_TOPK_FUSED": "0"},
            status="BLOCKED",
            region="Flash router logits, top-K, and route weights",
            eliminated="one router/top-K dispatch boundary and router intermediate",
            dispatch="reduce by one in the layer organ; verify graph ledger",
            intermediate="remove router logits/selection staging where outputs remain observable",
            active="route metadata remains dynamic; expert active bytes unchanged until NX closes",
            gpu="fuse route selection while preserving exact tie and top-K contract",
            blocked_reason="source-independent Flash route consumer and protected capability are not qualified",
            source_evidence=("crates/hawking-core/examples/flash_full_attention_layer3.rs",),
        ),
        _flash_candidate(
            "flash-hc-router-topk-fusion",
            mutation={"HAWKING_FLASH_HC_ROUTER_FUSED": "1"},
            status="STATIC_ONLY",
            region="Flash HC output, block projection, router, shared scalar, top-K",
            eliminated="MLP-HC intermediate plus two routing dispatch boundaries",
            dispatch="reduce by two in the layer organ; verify graph ledger",
            intermediate="remove staged MLP input and router preparation intermediates",
            active="unchanged source BF16; compact NX active bytes remain unqualified",
            gpu="reuse one staged HC output across block logits, router logits, scalar, and top-K",
            blocked_reason="source-independent Flash NX and protected capability are not qualified",
            dependencies=("flash-hc-staged-threadgroup", "flash-router-topk-fusion"),
            source_evidence=(
                "crates/hawking-core/shaders/qwen_next.metal::qwen_next_hyperconnection_input_fused_with_block_router_topk",
                "crates/hawking-core/examples/flash_full_attention_layer3.rs",
            ),
        ),
        _flash_candidate(
            "flash-compact-moe-epilogue",
            mutation={"HAWKING_FLASH_MOE_GEO": "compact"},
            status="BLOCKED",
            region="Flash routed/shared expert output epilogue",
            eliminated="dense expert materialization and routed/shared output staging",
            dispatch="measure compact organ ledger; no assumed reduction",
            intermediate="remove compact routed/shared accumulator materialization",
            active="reduce only if route-conditioned compact body is independently verified",
            gpu="direct compact expert accumulation into residual/final-state output",
            blocked_reason="Flash source-independent compact expert consumer is not yet qualified",
            source_evidence=(
                "crates/hawking-core/examples/flash_noetic_complete_layer0.rs",
                "receipts/headless/ACCELERATOR_MLP_OPERAND_REUSE_REFUTED.json",
            ),
        ),
        _flash_candidate(
            "flash-attention-gate-fusion",
            mutation={"HAWKING_FLASH_FUSE_ATTENTION_GATE": "1"},
            status="BLOCKED",
            region="Flash attention output sigmoid gate",
            eliminated="one standalone attention-gate write/read boundary",
            dispatch="reduce by one in the attention organ; verify graph ledger",
            intermediate="remove gated attention output intermediate",
            active="unchanged source footprint; NX active bytes unqualified",
            gpu="apply the attention gate at the final attention write",
            blocked_reason="Flash complete-token and source-independent capability gates remain open",
            source_evidence=("crates/hawking-core/examples/flash_full_attention_layer3.rs",),
        ),
    ]


def _candidate_from_dict(raw: Mapping[str, Any]) -> PhysicalCandidate:
    if not isinstance(raw, Mapping):
        raise PhysicalQueueError("candidate rows must be objects")
    missing = [name for name in REQUIRED_FIELDS if name not in raw]
    if missing:
        raise PhysicalQueueError(f"candidate is missing fields: {missing}")
    raw_measurements = raw.get("measurements")
    if raw_measurements is None:
        raw_measurements = _empty_measurements()
    if not isinstance(raw_measurements, Mapping):
        raise PhysicalQueueError("candidate measurements must be an object")
    return PhysicalCandidate(
        candidate_id=str(raw.get("candidate_id") or ""),
        model=str(raw.get("model") or ""),
        affected_physical_region=str(raw.get("affected_physical_region") or ""),
        baseline_path=str(raw.get("baseline_path") or ""),
        exact_mutation=dict(raw.get("exact_mutation") or {}),
        expected_eliminated_work=str(raw.get("expected_eliminated_work") or ""),
        expected_dispatch_reduction=str(raw.get("expected_dispatch_reduction") or ""),
        expected_intermediate_byte_reduction=str(raw.get("expected_intermediate_byte_reduction") or ""),
        expected_active_byte_change=str(raw.get("expected_active_byte_change") or ""),
        expected_gpu_ns_mechanism=str(raw.get("expected_gpu_ns_mechanism") or ""),
        parity_contract=str(raw.get("parity_contract") or ""),
        capability_contract=str(raw.get("capability_contract") or ""),
        diagnostic_command=_strings(raw.get("diagnostic_command"), name="diagnostic_command"),
        protected_command=_strings(raw.get("protected_command"), name="protected_command"),
        dependencies=_strings(raw.get("dependencies") or [], name="dependencies", allow_empty=True),
        status=str(raw.get("status") or ""),
        scope_tags=_strings(raw.get("scope_tags"), name="scope_tags"),
        transfer_evidence=_strings(
            raw.get("transfer_evidence") or [],
            name="transfer_evidence",
            allow_empty=True,
        ),
        control_configuration=dict(raw.get("control_configuration") or {}),
        source_evidence=_strings(raw.get("source_evidence") or [], name="source_evidence", allow_empty=True),
        blocked_reason=str(raw.get("blocked_reason")) if raw.get("blocked_reason") else None,
        evidence=_strings(raw.get("evidence") or [], name="evidence", allow_empty=True),
        measurements=dict(raw_measurements),
    )


def validate_candidate(candidate: PhysicalCandidate | Mapping[str, Any]) -> dict[str, Any]:
    value = candidate if isinstance(candidate, PhysicalCandidate) else _candidate_from_dict(candidate)
    for name in (
        "candidate_id",
        "model",
        "affected_physical_region",
        "baseline_path",
        "expected_eliminated_work",
        "expected_dispatch_reduction",
        "expected_intermediate_byte_reduction",
        "expected_active_byte_change",
        "expected_gpu_ns_mechanism",
        "parity_contract",
        "capability_contract",
    ):
        if not str(getattr(value, name)).strip():
            raise PhysicalQueueError(f"{value.candidate_id or '<candidate>'}: {name} is empty")
    if value.status not in STATUSES:
        raise PhysicalQueueError(f"{value.candidate_id}: invalid status {value.status!r}")
    if not value.scope_tags:
        raise PhysicalQueueError(f"{value.candidate_id}: scope_tags must be non-empty")
    unknown_scope_tags = sorted(set(value.scope_tags) - CANDIDATE_SCOPE_TAGS)
    if unknown_scope_tags:
        raise PhysicalQueueError(
            f"{value.candidate_id}: invalid scope_tags {unknown_scope_tags}"
        )
    if "GENERIC_VERIFIED" in value.scope_tags and (
        value.status != "INTEGRATED" or not value.transfer_evidence
    ):
        raise PhysicalQueueError(
            f"{value.candidate_id}: GENERIC_VERIFIED requires an integrated cross-model receipt"
        )
    if "GENERIC_CANDIDATE" in value.scope_tags and not value.transfer_evidence:
        raise PhysicalQueueError(
            f"{value.candidate_id}: generic candidate requires transfer_evidence"
        )
    if not isinstance(value.exact_mutation, Mapping):
        raise PhysicalQueueError(f"{value.candidate_id}: exact_mutation must be an object")
    if not isinstance(value.measurements, Mapping):
        raise PhysicalQueueError(f"{value.candidate_id}: measurements must be an object")
    missing_measurements = [name for name in MEASUREMENT_FIELDS if name not in value.measurements]
    if missing_measurements:
        raise PhysicalQueueError(
            f"{value.candidate_id}: measurements missing fields {missing_measurements}"
        )
    measurement_status = str(value.measurements.get("status") or "")
    if measurement_status not in {"NOT_MEASURED", "RECORDED"}:
        raise PhysicalQueueError(
            f"{value.candidate_id}: invalid measurements.status {measurement_status!r}"
        )
    absence_reasons = value.measurements.get("absence_reasons")
    if not isinstance(absence_reasons, Mapping):
        raise PhysicalQueueError(f"{value.candidate_id}: measurements.absence_reasons must be an object")
    missing_reasons = [
        name
        for name in MEASUREMENT_FIELDS
        if value.measurements.get(name) is None and not str(absence_reasons.get(name) or "").strip()
    ]
    if missing_reasons:
        raise PhysicalQueueError(
            f"{value.candidate_id}: missing absence reasons for metrics {missing_reasons}"
        )
    invalid_measurements = [
        name
        for name in MEASUREMENT_FIELDS
        if value.measurements.get(name) is not None
        and (
            isinstance(value.measurements.get(name), bool)
            or not isinstance(value.measurements.get(name), (int, float))
            or not math.isfinite(float(value.measurements.get(name)))
            or float(value.measurements.get(name)) < 0
        )
    ]
    if invalid_measurements:
        raise PhysicalQueueError(
            f"{value.candidate_id}: measurements must be non-negative numbers {invalid_measurements}"
        )
    if _command_has_shell(value.diagnostic_command) or _command_has_shell(value.protected_command):
        raise PhysicalQueueError(f"{value.candidate_id}: shell command markers are forbidden")
    if any("\x00" in item or "\n" in item or "\r" in item for item in (*value.diagnostic_command, *value.protected_command)):
        raise PhysicalQueueError(f"{value.candidate_id}: command contains a control character")
    if value.status == "BLOCKED" and not value.blocked_reason:
        raise PhysicalQueueError(f"{value.candidate_id}: BLOCKED requires blocked_reason")
    if value.status in EVIDENCE_STATUSES and not value.evidence:
        raise PhysicalQueueError(f"{value.candidate_id}: terminal status requires evidence")
    if value.status in PROTECTED_MEASURED_STATUSES:
        missing_values = [name for name in MEASUREMENT_FIELDS if value.measurements.get(name) is None]
        if missing_values or measurement_status != "RECORDED":
            raise PhysicalQueueError(
                f"{value.candidate_id}: protected pass requires recorded metrics {missing_values}"
            )
    return {
        "schema": "hawking.accelerator.physical_candidate_validation.v1",
        "passed": True,
        "candidate_id": value.candidate_id,
        "status": value.status,
        "claim_boundary": "candidate validation is not a physical measurement",
    }


def workunits_for_candidates(candidates: Iterable[PhysicalCandidate]) -> list[dict[str, Any]]:
    ready = {candidate.candidate_id for candidate in candidates if candidate.status in READY_STATUSES}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.status not in READY_STATUSES:
            continue
        dependencies = [
            f"accelerator.physical.{dependency}"
            for dependency in candidate.dependencies
            if dependency in ready
        ]
        unit = WorkUnit(
            id=f"accelerator.physical.{candidate.candidate_id}",
            role="accelerator_physical_qualification",
            description=(
                f"Qualify {candidate.candidate_id} on {candidate.model}; "
                f"region={candidate.affected_physical_region}; "
                f"falsifier={candidate.parity_contract}"
            ),
            dependencies=dependencies,
            resource_class="GPU_EXCLUSIVE",
            preferred_backend="metal",
            provider="accelerator_physical_queue",
            verifier=f"accelerator.physical.{candidate.candidate_id}",
            effect_class="REVERSIBLE",
            workspace="repo-root",
        )
        row = unit.to_dict()
        row.update(
            {
                "candidate_id": candidate.candidate_id,
                "model": candidate.model,
                "status": "pending",
                "diagnostic_command": list(candidate.diagnostic_command),
                "protected_command": list(candidate.protected_command),
                "claim_boundary": "WorkUnit is a proposal; receipt and protected capability gates remain authoritative",
            }
        )
        rows.append(row)
    return rows


def _derived_queue_fields(candidates: Sequence[PhysicalCandidate]) -> dict[str, Any]:
    """Recompute queue-owned projections after a candidate status change."""
    ordered = list(candidates)
    workunits = workunits_for_candidates(ordered)
    by_status = {status: sum(candidate.status == status for candidate in ordered) for status in sorted(STATUSES)}
    return {
        "candidates": [candidate.to_dict() for candidate in ordered],
        "work_units": workunits,
        "funnel": {
            "static_validation": [candidate.candidate_id for candidate in ordered if candidate.status != "BLOCKED"],
            "native_parity": [],
            "diagnostic_relative_ab": [
                candidate.candidate_id for candidate in ordered if candidate.status == "READY_DIAGNOSTIC"
            ],
            "protected_absolute_complete_wall": [
                candidate.candidate_id for candidate in ordered if candidate.status == "READY_PROTECTED"
            ],
            "promotion": [candidate.candidate_id for candidate in ordered if candidate.status in PASS_STATUSES],
            "promotion_rule": "only a protected complete-token receipt with capability and zero-fallback gates may promote",
        },
        "counts": {
            "candidates": len(ordered),
            "work_units": len(workunits),
            "ready": sum(candidate.status in READY_STATUSES for candidate in ordered),
            "by_status": by_status,
        },
    }


def _with_fingerprint(body: Mapping[str, Any]) -> dict[str, Any]:
    result = _copy(dict(body))
    result.pop("fingerprint", None)
    result["fingerprint"] = _hash(result)
    return result


def _ordered_candidates(candidates: Iterable[PhysicalCandidate]) -> list[PhysicalCandidate]:
    return sorted(
        candidates,
        key=lambda item: (0 if item.status in READY_STATUSES else 1, item.model, item.candidate_id),
    )


def build_queue(*, model: str | None = None) -> dict[str, Any]:
    candidates = [
        candidate for candidate in frontier_candidates() if model is None or candidate.model == model
    ]
    for candidate in candidates:
        validate_candidate(candidate)
    ids = {candidate.candidate_id for candidate in candidates}
    for candidate in candidates:
        unknown = sorted(set(candidate.dependencies) - ids)
        if unknown:
            raise PhysicalQueueError(f"{candidate.candidate_id}: unknown dependencies {unknown}")
    candidates = _ordered_candidates(candidates)
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "selection": {"model": model, "ordering": "ready first, then model, then candidate id"},
        "bench": {
            "state": "UNKNOWN",
            "machine": "planning artifact; no physical benchmark was executed",
            "recorded_by": "tools/accelerator/physical_qualification.py",
            "rule": "unknown physical state is not quiet and no candidate status implies a timing result",
        },
        "candidate_statuses": sorted(STATUSES),
        "status_transitions": {
            status: sorted(STATUS_TRANSITIONS[status]) for status in sorted(STATUSES)
        },
        "measurement_contract": {
            "required_fields": list(MEASUREMENT_FIELDS),
            "protected_pass_requires_all_fields": True,
            "null_policy": "missing physical metrics remain null until a native protected complete-token receipt records them",
            "metric_scope": "accepted complete generated token; not isolated kernel time",
        },
        "queue_policy": {
            "planning_is_side_effect_free": True,
            "commands_are_argv_only": True,
            "protected_start_requires_existing_hcli_lease": True,
            "protected_start_requires_machine_quiescence": True,
            "diagnostic_results_do_not_promote": True,
            "missing_gpu_or_ane_metrics_remain_missing": True,
            "flash_source_oracle_is_not_flash_nx": True,
        },
        "claim_boundary": (
            "This is a physical qualification frontier and HCLI WorkUnit proposal. "
            "It contains hypotheses and commands, not timing, placement, energy, TPS, "
            "capability, or promotion claims."
        ),
    }
    body.update(_derived_queue_fields(candidates))
    return _with_fingerprint(body)


def advance_queue(
    queue: Mapping[str, Any],
    *,
    candidate_id: str,
    status: str,
    evidence: Iterable[str] = (),
    blocked_reason: str | None = None,
    measurements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance one candidate and rebuild HCLI's derived WorkUnit projections.

    This function only records evidence supplied by an external diagnostic or
    protected runner.  It never infers a pass, starts a benchmark, or changes
    a candidate's mutation.
    """
    validate_queue(queue)
    if status not in STATUSES:
        raise PhysicalQueueError(f"invalid target status {status!r}")
    supplied_evidence = tuple(str(item) for item in evidence)
    if any(not item or "\x00" in item or "\n" in item or "\r" in item for item in supplied_evidence):
        raise PhysicalQueueError("evidence entries must be non-empty and single-line")
    raw_candidates = queue.get("candidates")
    assert isinstance(raw_candidates, list)  # validate_queue established this
    candidates = [_candidate_from_dict(raw) for raw in raw_candidates]
    try:
        index = next(index for index, item in enumerate(candidates) if item.candidate_id == candidate_id)
    except StopIteration as exc:
        raise PhysicalQueueError(f"unknown candidate {candidate_id!r}") from exc
    current = candidates[index]
    if status != current.status and status not in STATUS_TRANSITIONS[current.status]:
        allowed = sorted(STATUS_TRANSITIONS[current.status])
        raise PhysicalQueueError(
            f"{candidate_id}: cannot advance {current.status} -> {status}; allowed={allowed}"
        )
    merged_evidence = tuple(dict.fromkeys((*current.evidence, *supplied_evidence)))
    next_blocked_reason = current.blocked_reason if blocked_reason is None else blocked_reason
    next_measurements = dict(current.measurements)
    if measurements is not None:
        if not isinstance(measurements, Mapping):
            raise PhysicalQueueError("measurements must be an object")
        unknown_measurements = sorted(
            set(measurements) - set(MEASUREMENT_FIELDS) - {"status", "absence_reasons"}
        )
        if unknown_measurements:
            raise PhysicalQueueError(f"unknown measurement fields {unknown_measurements}")
        next_measurements.update(_copy(dict(measurements)))
        reasons = dict(next_measurements.get("absence_reasons") or {})
        for name in MEASUREMENT_FIELDS:
            if name in measurements and measurements[name] is not None:
                reasons.pop(name, None)
        if "absence_reasons" in measurements:
            supplied_reasons = measurements["absence_reasons"]
            if not isinstance(supplied_reasons, Mapping):
                raise PhysicalQueueError("measurements.absence_reasons must be an object")
            reasons.update(_copy(dict(supplied_reasons)))
        next_measurements["absence_reasons"] = reasons
        next_measurements["status"] = "RECORDED"
    if status == "BLOCKED" and not next_blocked_reason:
        raise PhysicalQueueError(f"{candidate_id}: BLOCKED requires blocked_reason")
    if status in EVIDENCE_STATUSES and not merged_evidence:
        raise PhysicalQueueError(f"{candidate_id}: terminal status requires evidence")
    if status in PROTECTED_MEASURED_STATUSES:
        missing_values = [name for name in MEASUREMENT_FIELDS if next_measurements.get(name) is None]
        if missing_values or next_measurements.get("status") != "RECORDED":
            raise PhysicalQueueError(
                f"{candidate_id}: protected pass requires recorded metrics {missing_values}"
            )
    candidates[index] = replace(
        current,
        status=status,
        evidence=merged_evidence,
        blocked_reason=next_blocked_reason,
        measurements=next_measurements,
    )
    candidates = _ordered_candidates(candidates)
    updated = _copy(dict(queue))
    updated.update(_derived_queue_fields(candidates))
    updated["claim_boundary"] = (
        "This is a physical qualification frontier and HCLI WorkUnit proposal. "
        "It contains hypotheses and commands, not timing, placement, energy, TPS, "
        "capability, or promotion claims. Status changes are evidence references, "
        "not measurements by themselves."
    )
    result = _with_fingerprint(updated)
    validate_queue(result)
    return result


def validate_queue(queue: Mapping[str, Any]) -> dict[str, Any]:
    if queue.get("schema") != SCHEMA:
        raise PhysicalQueueError(f"schema must be {SCHEMA}")
    raw_candidates = queue.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise PhysicalQueueError("candidates must be a non-empty list")
    candidates = [_candidate_from_dict(raw) for raw in raw_candidates]
    ids: set[str] = set()
    for candidate in candidates:
        validate_candidate(candidate)
        if candidate.candidate_id in ids:
            raise PhysicalQueueError(f"duplicate candidate id {candidate.candidate_id!r}")
        ids.add(candidate.candidate_id)
    for candidate in candidates:
        unknown = sorted(set(candidate.dependencies) - ids)
        if unknown:
            raise PhysicalQueueError(f"{candidate.candidate_id}: unknown dependencies {unknown}")
    raw_units = queue.get("work_units")
    if not isinstance(raw_units, list):
        raise PhysicalQueueError("work_units must be a list")
    ready_ids = {candidate.candidate_id for candidate in candidates if candidate.status in READY_STATUSES}
    unit_ids: set[str] = set()
    for unit in raw_units:
        if not isinstance(unit, Mapping):
            raise PhysicalQueueError("work_units must contain objects")
        candidate_id = str(unit.get("candidate_id") or "")
        if candidate_id not in ready_ids:
            raise PhysicalQueueError(f"work unit references non-ready candidate {candidate_id!r}")
        if candidate_id in unit_ids:
            raise PhysicalQueueError(f"duplicate work unit for {candidate_id!r}")
        unit_ids.add(candidate_id)
    if unit_ids != ready_ids:
        raise PhysicalQueueError("every READY candidate must have exactly one HCLI WorkUnit")
    counts = queue.get("counts")
    expected_counts = {
        "candidates": len(candidates),
        "work_units": len(raw_units),
        "ready": len(ready_ids),
        "by_status": {status: sum(candidate.status == status for candidate in candidates) for status in sorted(STATUSES)},
    }
    if counts != expected_counts:
        raise PhysicalQueueError("queue counts are stale")
    expected_funnel = _derived_queue_fields(candidates)["funnel"]
    if queue.get("funnel") != expected_funnel:
        raise PhysicalQueueError("queue funnel is stale")
    if queue.get("candidate_statuses") != sorted(STATUSES):
        raise PhysicalQueueError("candidate_statuses must enumerate the controlled status set")
    expected_transitions = {
        status: sorted(STATUS_TRANSITIONS[status]) for status in sorted(STATUSES)
    }
    if queue.get("status_transitions") != expected_transitions:
        raise PhysicalQueueError("status_transitions do not match the HCLI state machine")
    expected_measurement_contract = {
        "required_fields": list(MEASUREMENT_FIELDS),
        "protected_pass_requires_all_fields": True,
        "null_policy": "missing physical metrics remain null until a native protected complete-token receipt records them",
        "metric_scope": "accepted complete generated token; not isolated kernel time",
    }
    if queue.get("measurement_contract") != expected_measurement_contract:
        raise PhysicalQueueError("measurement_contract is stale")
    fingerprint = queue.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != _hash(
        {key: value for key, value in queue.items() if key != "fingerprint"}
    ):
        raise PhysicalQueueError("queue fingerprint is stale")
    return {
        "schema": "hawking.accelerator.physical_qualification_queue_validation.v1",
        "passed": True,
        "candidates": len(candidates),
        "ready": len(ready_ids),
        "claim_boundary": "queue validation is not a physical measurement",
    }


def emit_queue(
    *,
    output: str | Path | None = None,
    model: str | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    destination = Path(output).expanduser() if output else root / DEFAULT_QUEUE_OUT
    if not destination.is_absolute():
        destination = root / destination
    body = build_queue(model=model)
    validate_queue(body)
    atomic_write_json(destination, body)
    return destination


def emit_advanced_queue(
    *,
    queue_path: str | Path,
    candidate_id: str,
    status: str,
    evidence: Iterable[str] = (),
    blocked_reason: str | None = None,
    measurements: Mapping[str, Any] | None = None,
    receipt: str | Path | None = None,
    output: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    source = Path(queue_path).expanduser()
    if not source.is_absolute():
        source = root / source
    destination = Path(output).expanduser() if output else source
    if not destination.is_absolute():
        destination = root / destination
    queue = json.loads(source.read_text(encoding="utf-8"))
    if receipt is not None:
        if measurements is not None:
            raise PhysicalQueueError("--receipt cannot be combined with --measurements")
        raw_candidates = queue.get("candidates")
        if not isinstance(raw_candidates, list):
            raise PhysicalQueueError("queue candidates must be a list before receipt import")
        raw_candidate = next(
            (
                raw
                for raw in raw_candidates
                if isinstance(raw, Mapping) and raw.get("candidate_id") == candidate_id
            ),
            None,
        )
        if raw_candidate is None:
            raise PhysicalQueueError(f"unknown candidate {candidate_id!r}")
        candidate = _candidate_from_dict(raw_candidate)
        mutation = candidate.exact_mutation.get("child_fusion_env")
        expected_mutation = mutation if isinstance(mutation, Mapping) else None
        receipt_path = Path(receipt).expanduser()
        measurements = load_receipt_measurements(
            receipt_path,
            expected_mutation=expected_mutation,
        )
        evidence = (*tuple(evidence), str(receipt_path.resolve()))
    updated = advance_queue(
        queue,
        candidate_id=candidate_id,
        status=status,
        evidence=evidence,
        blocked_reason=blocked_reason,
        measurements=measurements,
    )
    validate_queue(updated)
    atomic_write_json(destination, updated)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--emit", default=None)
    parser.add_argument("--validate", default=None)
    parser.add_argument("--model", choices=("Qwen27", "Flash"), default=None)
    parser.add_argument("--queue", default=None, help="existing queue JSON to advance")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--advance-status", choices=sorted(STATUSES), default=None)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--blocked-reason", default=None)
    parser.add_argument("--measurements", default=None, help="JSON object containing recorded physical metrics")
    parser.add_argument("--receipt", default=None, help="protected HCLI receipt to import as measurements")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if any(value is not None for value in (args.candidate_id, args.advance_status, args.queue, args.measurements, args.receipt)):
        if not args.queue or not args.candidate_id or not args.advance_status:
            parser.error("--queue, --candidate-id, and --advance-status are required together")
        if args.validate or args.model:
            parser.error("queue advancement cannot be combined with --validate or --model")
        measurements = None
        if args.measurements:
            measurements = json.loads(Path(args.measurements).expanduser().read_text(encoding="utf-8"))
        destination = emit_advanced_queue(
            queue_path=args.queue,
            candidate_id=args.candidate_id,
            status=args.advance_status,
            evidence=args.evidence,
            blocked_reason=args.blocked_reason,
            measurements=measurements,
            receipt=args.receipt,
            output=args.emit,
            repo_root=args.repo_root,
        )
        body = json.loads(destination.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": "PASSED",
                    "path": str(destination),
                    "candidate_id": args.candidate_id,
                    "candidate_status": args.advance_status,
                    "fingerprint": body["fingerprint"],
                    "ready": body["counts"]["ready"],
                    "work_units": body["counts"]["work_units"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.queue:
        parser.error("--queue requires --candidate-id and --advance-status")
    if args.validate:
        value = json.loads(Path(args.validate).expanduser().read_text(encoding="utf-8"))
        print(json.dumps(validate_queue(value), indent=2, sort_keys=True))
        return 0
    destination = emit_queue(output=args.emit, model=args.model, repo_root=args.repo_root)
    body = json.loads(destination.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "PASSED",
                "path": str(destination),
                "fingerprint": body["fingerprint"],
                "candidates": body["counts"]["candidates"],
                "ready": body["counts"]["ready"],
                "work_units": body["counts"]["work_units"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DEFAULT_QUEUE_OUT",
    "EVIDENCE_STATUSES",
    "MEASUREMENT_FIELDS",
    "PASS_STATUSES",
    "PROTECTED_MEASURED_STATUSES",
    "PhysicalCandidate",
    "PhysicalQueueError",
    "READY_STATUSES",
    "SCHEMA",
    "STATUSES",
    "STATUS_TRANSITIONS",
    "advance_queue",
    "build_queue",
    "emit_advanced_queue",
    "emit_queue",
    "frontier_candidates",
    "load_receipt_measurements",
    "measurements_from_receipt",
    "main",
    "validate_candidate",
    "validate_queue",
    "workunits_for_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
