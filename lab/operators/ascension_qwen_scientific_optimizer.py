"""Deterministic scientific optimization lane for the live Qwen campaign.

This is deliberately an *additive research lane*, not another Ascension
controller.  It has no authority to select a tournament winner, alter a pack,
start a benchmark, or make a runtime/capability/TPS claim.  Its job is to turn
the evidence already emitted by the two physical workers into durable,
cross-family, non-superficial experiments.

Every proposed experiment is bound to the current source/pack/runtime state,
the peer worker, and the shared knowledge ledgers.  It states the current
blocker, three genuinely different possible mechanisms, the cheapest test
that distinguishes them, prior initialization, and explicit PASS/FAIL/REOPEN
conditions.  In the absence of a sealed exact native full-token receipt it
fails closed: it emits ``BLOCKED_NATIVE_RUNTIME`` work rather than inventing a
TPS value.  The only experiment this operator executes before that gate is a
small, source-bound packed-artifact I/O/header profile.  That profile is
strictly component evidence, never a model throughput result.

The process is intended for ``launchd``.  State updates and evidence records
are atomic, and worker liveness explicitly rejects a changing heartbeat that
is not accompanied by an externally visible material frontier change.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import stat
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "records" / "ascension-sandbox" / "physical"
)

SCHEMA = "hawking.ascension.qwen_scientific_optimizer.v1"
EXPERIMENT_SCHEMA = "hawking.ascension.qwen_scientific_optimizer_experiment.v1"
COMPONENT_PROFILE_SCHEMA = "hawking.ascension.qwen_packed_component_io_profile.v1"
FRONTIER_SCHEMA = "hawking.ascension.qwen_scientific_optimizer_frontier.v1"
NEGATIVE_SCHEMA = "hawking.ascension.negative_science.v1"
RUNTIME_SCHEMA = "hawking.ascension.physical_exact_full_token_runtime.v1"
RUNTIME_STATUS = "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME"
GATE_UP_QUALITY_SCHEMA = "hawking.ascension.qwen30_direct_packed_gate_up_quality_diagnostic.v1"
CANONICAL_TEMPLATE_EVIDENCE_SCHEMA = "hawking.ascension.qwen30_canonical_template_kernel_evidence.v1"

MAX_COMPONENT_PROFILE_BYTES = 4 * 1024 * 1024
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
GENOME_TAIL = 12
SENSITIVE_ROUTER_QUALITY_REVISION = "v2_source_loader_signature_repaired"
GATE_UP_QUALITY_REVISION = "v1_source_bound_paired_swiglu_control"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    prefix: str
    legacy_worker_status: str
    model_family: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen30",
        prefix="QWEN30",
        legacy_worker_status="QWEN30_REAL_CAMPAIGN_STATUS.json",
        model_family="qwen3_moe",
    ),
    ModelSpec(
        key="qwen80",
        prefix="QWEN80",
        legacy_worker_status="QWEN80_PHYSICAL_CAMPAIGN_STATUS.json",
        model_family="qwen3_next_hybrid",
    ),
)


class ScientificOptimizerError(RuntimeError):
    """A deterministic research cycle cannot safely advance."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    raw = value if isinstance(value, (bytes, bytearray)) else _canonical(value)
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_DOCUMENT_BYTES:
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(loaded) if isinstance(loaded, Mapping) else None


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_jsonl_once(path: Path, document: Mapping[str, Any], *, record_id: str) -> bool:
    """Append an immutable shared-science row once, even across a retry race."""

    lock_path = path.with_name(f".{path.name}.lock")
    with _locked(lock_path):
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, Mapping) and row.get("record_id") == record_id:
                        return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(document), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o640)
    return True


def _is_regular_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        observed = os.lstat(path)
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode)


def _bounded_file_summary(path: Path, *, seal_expected: bool = False) -> dict[str, Any]:
    """Read small JSON evidence without treating an unsealed status as proof."""

    summary: dict[str, Any] = {"path": str(path), "present": path.is_file()}
    if not path.is_file():
        summary.update({"state": "MISSING", "sealed": False})
        return summary
    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        summary.update({"state": "UNREADABLE", "sealed": False, "error": str(exc)})
        return summary
    summary["bytes"] = size
    if size > MAX_DOCUMENT_BYTES:
        summary.update({"state": "TOO_LARGE", "sealed": False})
        return summary
    raw = _read_json(path)
    if raw is None:
        summary.update({"state": "UNREADABLE_OR_INVALID_JSON", "sealed": False})
        return summary
    summary["document_sha256"] = _digest(raw)
    summary.update(
        {
            "schema": raw.get("schema"),
            "status": raw.get("status"),
            "phase": raw.get("phase"),
            "recorded_at": raw.get("recorded_at"),
        }
    )
    try:
        checked = verify(raw, label=str(path))
    except Exception as exc:
        summary.update(
            {
                "state": "OBSERVED_UNSEALED" if not seal_expected else "MISSING_OR_INVALID_SEAL",
                "sealed": False,
                "error": str(exc),
                "document": raw,
            }
        )
        return summary
    summary.update(
        {
            "state": "SEALED",
            "sealed": True,
            "seal_sha256": checked.get("seal_sha256"),
            "document": checked,
        }
    )
    return summary


def _document_fields(summary: Mapping[str, Any], *names: str) -> dict[str, Any]:
    document = summary.get("document")
    if not isinstance(document, Mapping):
        return {}
    return {name: document.get(name) for name in names if name in document}


def _receipt_reference(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Keep an evidence binding compact; never embed a complete tensor catalog."""

    keys = (
        "path",
        "present",
        "bytes",
        "state",
        "sealed",
        "seal_sha256",
        "document_sha256",
        "schema",
        "status",
        "phase",
        "recorded_at",
        "error",
    )
    return {key: summary.get(key) for key in keys if key in summary}


def _genome_summary(path: Path) -> dict[str, Any]:
    """Return a sealed tail of a shared ledger, never an unbounded copy."""

    output: dict[str, Any] = {"path": str(path), "present": path.is_file(), "record_count": 0, "tail": []}
    if not path.is_file():
        output["state"] = "MISSING"
        return output
    try:
        output["bytes"] = int(path.stat().st_size)
        output["sha256"] = _sha256_file(path)
        tail: list[dict[str, Any]] = []
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, Mapping):
                    tail.append(dict(row))
                    if len(tail) > GENOME_TAIL:
                        tail.pop(0)
    except OSError as exc:
        output.update({"state": "UNREADABLE", "error": str(exc)})
        return output
    condensed: list[dict[str, Any]] = []
    for row in tail:
        try:
            verified = verify(row, label=f"{path}:tail")
            sealed = True
            seal_value = verified.get("seal_sha256")
        except Exception:
            sealed = False
            seal_value = None
        condensed.append(
            {
                "record_id": row.get("record_id"),
                "schema": row.get("schema"),
                "status": row.get("status"),
                "recorded_at": row.get("recorded_at"),
                "model": row.get("model"),
                "model_family": row.get("model_family"),
                "representation": row.get("representation"),
                "mechanism": row.get("mechanism"),
                "mechanism_key": row.get("mechanism_key"),
                "seal_sha256": seal_value,
                "sealed": sealed,
            }
        )
    output.update({"state": "OBSERVED", "record_count": count, "tail": condensed})
    return output


def _external_ledger_fingerprint(summary: Mapping[str, Any]) -> str | None:
    """Digest peer research while excluding this lane's own appended receipts.

    Otherwise a successful profile would alter the shared kernel/scheduler
    ledger, cause a different task fingerprint on the next watch tick, and
    accidentally manufacture a superficial self-rerun.  Peer-worker evidence
    remains fully visible and any new peer record changes this digest.
    """

    tail = summary.get("tail")
    if not isinstance(tail, list):
        return None
    external: list[dict[str, Any]] = []
    for row in tail:
        if not isinstance(row, Mapping):
            continue
        record_id = row.get("record_id")
        if isinstance(record_id, str) and (
            record_id.startswith("kernel-profile:")
            or record_id.startswith("scheduler:")
            or record_id.startswith("negative-optimizer:")
        ):
            continue
        external.append(
            {
                "record_id": record_id,
                "seal_sha256": row.get("seal_sha256"),
                "status": row.get("status"),
                "mechanism_key": row.get("mechanism_key"),
            }
        )
    return _digest(external)


def _pid_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return {"state": "NO_PID_DECLARED", "pid": value, "alive": False}
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return {"state": "PID_NOT_ALIVE", "pid": value, "alive": False}
    except PermissionError:
        return {"state": "PID_EXISTS_PERMISSION_DENIED", "pid": value, "alive": True}
    return {"state": "PID_ALIVE", "pid": value, "alive": True}


class QwenScientificOptimizer:
    """Read-only observer plus bounded component-profile experiment runner."""

    def __init__(self, *, physical_root: Path = DEFAULT_PHYSICAL_ROOT) -> None:
        self.physical_root = Path(physical_root)
        self.root = self.physical_root / "qwen-family" / "scientific-optimizer"
        self.experiments_dir = self.root / "experiments"
        self.profiles_dir = self.root / "component-profiles"
        self.state_path = self.root / "WORKER_STATE.json"
        self.status_path = self.root / "QWEN_SCIENTIFIC_OPTIMIZER_STATUS.json"
        self.frontier_path = self.root / "QWEN_SCIENTIFIC_OPTIMIZER_FRONTIER.json"
        self.lock_path = self.root / ".qwen-scientific-optimizer.lock"
        self.shared_root = self.physical_root / "qwen-family" / "dual-gravity"
        self.shared_kernel_path = self.shared_root / "ASCENSION_KERNEL_GENOME.jsonl"
        self.shared_scheduler_path = self.shared_root / "ASCENSION_SCHEDULER_GENOME.jsonl"
        self.shared_representation_path = self.shared_root / "ASCENSION_REPRESENTATION_GENOME.jsonl"
        self.shared_negative_path = self.shared_root / "ASCENSION_NEGATIVE_SCIENCE.jsonl"
        self._stopping = False

    @staticmethod
    def _model_root(spec: ModelSpec, root: Path) -> Path:
        return root / spec.key

    def _paths(self, spec: ModelSpec) -> dict[str, Path]:
        base = self._model_root(spec, self.physical_root)
        return {
            "identity": base / "evolution" / "SOURCE_CONTENT_IDENTITY.json",
            "revalidation": base / "complete-gravity" / f"{spec.prefix}_CURRENT_SOURCE_SHARD_REVALIDATION.json",
            "worker": base / spec.legacy_worker_status,
            "worker_evolution": base / "evolution" / f"{spec.prefix}_DUAL_GRAVITY_STATUS.json",
            "state": base / "evolution" / "WORKER_STATE.json",
            "champions": base / "evolution" / "CHAMPIONS.json",
            "frontier": base / "evolution" / "PARETO_FRONTIER.json",
            "pack_status": base / "complete-gravity" / f"{spec.prefix}_COMPLETE_GRAVITY_STATUS.json",
            "manifest": base / "complete-gravity" / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json",
            "admission": base / "complete-gravity" / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json",
            "runtime_status": base / "complete-runtime" / f"{spec.prefix}_COMPLETE_RUNTIME_STATUS.json",
            "runtime": base / "complete-runtime" / f"{spec.prefix}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json",
            "direct_full_token": base / "complete-runtime" / f"{spec.prefix}_DIRECT_PACKED_NATIVE_FULL_TOKEN_RESULT.json",
            "direct_prompt_a": base / "complete-runtime" / f"{spec.prefix}_DIRECT_PACKED_NATIVE_PROMPT_A_RESULT.json",
            "hcli": base / "complete-runtime" / f"{spec.prefix}_MEASURED_HCLI_RECEIPT.json",
            # Qwen30's canonical-template receipts are deliberately observed
            # separately from the qualified HCLI path.  A transport smoke is
            # not an HCLI pass, and a rejected SIMD candidate is useful
            # negative science rather than a reason to replace the scalar
            # control or the admitted artifact.
            "hcli_unqualified_transport": base / "complete-runtime" / f"{spec.prefix}_HCLI_UNQUALIFIED_CHAT_SSE_TRANSPORT_RECEIPT.json",
            "simd_template_parity": base / "complete-runtime" / f"{spec.prefix}_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY_RECEIPT.json",
            "partial_gpu_profile_negative": base / "complete-runtime" / f"{spec.prefix}_DIRECT_PACKED_NATIVE_KERNEL_PROFILE_REJECTED_PARTIAL_GPU_TRACE.json",
            "gate_up_pair_component": base / "complete-token-profiler" / f"{spec.prefix}_DIRECT_PACKED_GATE_UP_PAIR_COMPONENT_RECEIPT.json",
            "gate_up_fused_raw": base / "complete-token-profiler" / f"{spec.prefix}_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_COMPONENT_RAW_RESULT.json",
            "capability": base / "evaluation" / f"{spec.prefix}_CAPABILITY_EVALUATION_RECEIPT.json",
            "kernel": self.physical_root / "kernel" / f"{spec.prefix}_CUSTOM_KERNEL_OPERATIONAL_RECEIPT.json",
            "tg3_status": base / "tg3" / f"{spec.prefix}_TG3_ASCENT_STATUS.json",
            "tg3": base / "tg3" / f"{spec.prefix}_TG3_QUALIFICATION_RECEIPT.json",
            "state_kv": base / "state-kv" / f"{spec.prefix}_STATE_KV_STATUS.json",
            "evolution_root": base / "evolution",
        }

    def _shared_knowledge(self) -> dict[str, Any]:
        return {
            "representation_genome": _genome_summary(self.shared_representation_path),
            "kernel_genome": _genome_summary(self.shared_kernel_path),
            "scheduler_genome": _genome_summary(self.shared_scheduler_path),
            "negative_science": _genome_summary(self.shared_negative_path),
            "component_kernel_receipts": [
                _bounded_file_summary(path, seal_expected=True)
                for path in sorted((self.physical_root / "kernel").glob("QWEN*_*.json"))
            ],
        }

    @staticmethod
    def _complete_bpw(manifest: Mapping[str, Any]) -> float | None:
        document = manifest.get("document")
        if not isinstance(document, Mapping):
            return None
        ledger = document.get("complete_physical_bpw_ledger")
        if not isinstance(ledger, Mapping):
            return None
        value = ledger.get("complete_physical_bpw")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    @staticmethod
    def _latest_worker_results(worker: Mapping[str, Any]) -> dict[str, Any]:
        document = worker.get("document")
        if not isinstance(document, Mapping):
            return {"state": worker.get("state"), "worker_status_path": worker.get("path")}
        population = document.get("population") if isinstance(document.get("population"), Mapping) else {}
        current = document.get("current_experiment") if isinstance(document.get("current_experiment"), Mapping) else {}
        return {
            "worker_status_path": worker.get("path"),
            "worker_status_sealed": worker.get("sealed"),
            "phase": document.get("phase"),
            "heartbeat": document.get("heartbeat"),
            "pid": document.get("pid"),
            "ppid": document.get("ppid"),
            "last_material_progress_at": document.get("last_material_progress_at"),
            "completed_candidate_count": population.get("completed_candidate_count"),
            "candidate_count": population.get("candidate_count"),
            "current_experiment_id": current.get("candidate_id"),
            "current_representation": current.get("representation"),
        }

    @staticmethod
    def _component_champion(champions: Mapping[str, Any]) -> dict[str, Any]:
        document = champions.get("document")
        if not isinstance(document, Mapping):
            return {"state": champions.get("state"), "path": champions.get("path")}
        rows = document.get("champions") if isinstance(document.get("champions"), Mapping) else document
        if not isinstance(rows, Mapping):
            return {"state": "MALFORMED_CHAMPIONS", "path": champions.get("path")}
        return {
            "path": champions.get("path"),
            "seal_sha256": champions.get("seal_sha256"),
            "current_fastest_component": rows.get("current_fastest_component")
            or rows.get("current_fastest_component_champion"),
            "current_lowest_bpw_component": rows.get("current_lowest_bpw_component")
            or rows.get("current_lowest_bpw_champion"),
            "current_capable": rows.get("current_capable") or rows.get("current_capable_champion"),
        }

    @staticmethod
    def _runtime_earned(runtime: Mapping[str, Any]) -> bool:
        document = runtime.get("document")
        return bool(
            runtime.get("sealed")
            and isinstance(document, Mapping)
            and document.get("schema") == RUNTIME_SCHEMA
            and document.get("status") == RUNTIME_STATUS
        )

    def _observe_model(self, spec: ModelSpec, shared: Mapping[str, Any]) -> dict[str, Any]:
        paths = self._paths(spec)
        identity = _bounded_file_summary(paths["identity"], seal_expected=True)
        revalidation = _bounded_file_summary(paths["revalidation"], seal_expected=True)
        worker = _bounded_file_summary(paths["worker"])
        if worker.get("state") == "MISSING":
            worker = _bounded_file_summary(paths["worker_evolution"])
        worker_state = _bounded_file_summary(paths["state"])
        champions = _bounded_file_summary(paths["champions"], seal_expected=True)
        frontier = _bounded_file_summary(paths["frontier"], seal_expected=True)
        pack_status = _bounded_file_summary(paths["pack_status"])
        manifest = _bounded_file_summary(paths["manifest"], seal_expected=True)
        admission = _bounded_file_summary(paths["admission"], seal_expected=True)
        runtime_status = _bounded_file_summary(paths["runtime_status"])
        runtime = _bounded_file_summary(paths["runtime"], seal_expected=True)
        direct_full_token = _bounded_file_summary(paths["direct_full_token"])
        direct_prompt_a = _bounded_file_summary(paths["direct_prompt_a"])
        hcli = _bounded_file_summary(paths["hcli"], seal_expected=True)
        hcli_unqualified_transport = _bounded_file_summary(paths["hcli_unqualified_transport"], seal_expected=True)
        simd_template_parity = _bounded_file_summary(paths["simd_template_parity"], seal_expected=True)
        partial_gpu_profile_negative = _bounded_file_summary(paths["partial_gpu_profile_negative"])
        gate_up_pair_component = _bounded_file_summary(paths["gate_up_pair_component"], seal_expected=True)
        gate_up_fused_raw = _bounded_file_summary(paths["gate_up_fused_raw"])
        capability = _bounded_file_summary(paths["capability"], seal_expected=True)
        kernel = _bounded_file_summary(paths["kernel"], seal_expected=True)
        tg3_status = _bounded_file_summary(paths["tg3_status"])
        tg3 = _bounded_file_summary(paths["tg3"], seal_expected=True)
        state_kv = _bounded_file_summary(paths["state_kv"], seal_expected=True)

        component_champion = self._component_champion(champions)
        current_complete_champion = {
            "manifest": {
                "path": manifest.get("path"),
                "sealed": manifest.get("sealed"),
                "seal_sha256": manifest.get("seal_sha256"),
                "status": manifest.get("status"),
                "complete_physical_bpw": self._complete_bpw(manifest),
            },
            "admission": {
                "path": admission.get("path"),
                "sealed": admission.get("sealed"),
                "seal_sha256": admission.get("seal_sha256"),
                "status": admission.get("status"),
            },
            "claim_boundary": "complete artifact evidence is separate from native runtime and capability",
        }
        runtime_document = runtime.get("document") if isinstance(runtime.get("document"), Mapping) else {}
        runtime_status_document = runtime_status.get("document") if isinstance(runtime_status.get("document"), Mapping) else {}
        complete_token_profile = {
            "exact_runtime_receipt": {
                "path": runtime.get("path"),
                "sealed": runtime.get("sealed"),
                "schema": runtime.get("schema"),
                "status": runtime.get("status"),
                "seal_sha256": runtime.get("seal_sha256"),
                "runtime": runtime_document.get("runtime"),
            },
            "watchdog": {
                "path": runtime_status.get("path"),
                "phase": runtime_status_document.get("phase"),
                "heartbeat": runtime_status_document.get("heartbeat"),
                "pid": runtime_status_document.get("pid"),
            },
            "profile_state": "EARNED_EXACT_RUNTIME" if self._runtime_earned(runtime) else "BLOCKED_NO_SEALED_EXACT_FULL_TOKEN_RUNTIME",
        }
        prompt_document = direct_prompt_a.get("document") if isinstance(direct_prompt_a.get("document"), Mapping) else {}
        execution = prompt_document.get("execution") if isinstance(prompt_document.get("execution"), Mapping) else {}
        binding = prompt_document.get("runtime_binding") if isinstance(prompt_document.get("runtime_binding"), Mapping) else {}
        observed_direct_generation = bool(
            prompt_document.get("status")
            == "EARNED_QWEN30_DIRECT_PACKED_NATIVE_GREEDY_AUTOREGRESSIVE_EXECUTED_UNQUALIFIED"
            and execution.get("all_48_layers_executed_for_each_forward") is True
            and execution.get("autoregressive_feedback_executed") is True
            and binding.get("metal_only") is True
            and binding.get("raw_bf16_loader_not_opened") is True
        )
        direct_generation = {
            "state": "OBSERVED_UNSEALED_DIRECT_PACKED_AUTOREGRESSIVE_TRACE_REQUIRES_CAPABILITY_QUALITY_GATE"
            if observed_direct_generation
            else "NO_DIRECT_PACKED_AUTOREGRESSIVE_TRACE",
            "receipt": _receipt_reference(direct_prompt_a),
            "full_token_receipt": _receipt_reference(direct_full_token),
            "completion_text_unscored": execution.get("completion_text_unscored"),
            "completion_token_ids": execution.get("completion_token_ids"),
            "full_model_forward_count": execution.get("full_model_forward_count"),
            "all_layers_executed": execution.get("all_48_layers_executed_for_each_forward"),
            "metal_only": binding.get("metal_only"),
            "raw_bf16_loader_not_opened": binding.get("raw_bf16_loader_not_opened"),
            "claim_boundary": "observed direct packed execution is not a sealed capability, HCLI, clean TPS, TG, or tournament result",
        }
        transport_document = (
            hcli_unqualified_transport.get("document")
            if isinstance(hcli_unqualified_transport.get("document"), Mapping)
            else {}
        )
        transport_measurement = (
            transport_document.get("measurement") if isinstance(transport_document.get("measurement"), Mapping) else {}
        )
        simd_document = (
            simd_template_parity.get("document")
            if isinstance(simd_template_parity.get("document"), Mapping)
            else {}
        )
        simd_failures = simd_document.get("failures") if isinstance(simd_document.get("failures"), list) else []
        pair_document = (
            gate_up_pair_component.get("document")
            if isinstance(gate_up_pair_component.get("document"), Mapping)
            else {}
        )
        raw_fused_document = (
            gate_up_fused_raw.get("document") if isinstance(gate_up_fused_raw.get("document"), Mapping) else {}
        )
        raw_fused_timing = raw_fused_document.get("timing") if isinstance(raw_fused_document.get("timing"), Mapping) else {}
        qwen30_canonical_template_evidence = {
            "exact_native_runtime": _receipt_reference(runtime),
            "unqualified_chat_sse_transport": {
                **_receipt_reference(hcli_unqualified_transport),
                "coherence": transport_measurement.get("coherence"),
                "clean_tps": transport_measurement.get("clean_tps"),
                "openai_chat_sse_framing_verified": transport_measurement.get("openai_chat_sse_framing_verified"),
                "uses_exact_native_runtime": transport_measurement.get("uses_exact_native_runtime"),
            },
            "rejected_template_simdgroup_candidate": {
                **_receipt_reference(simd_template_parity),
                "failures": [str(value) for value in simd_failures[:4]],
            },
            "direct_packed_gate_up_pair": {
                **_receipt_reference(gate_up_pair_component),
                "p50_component_host_wall_speedup_ratio": (
                    pair_document.get("timing", {}).get("p50_component_host_wall_speedup_ratio")
                    if isinstance(pair_document.get("timing"), Mapping)
                    else None
                ),
                "integration_state": "UNINTEGRATED_COMPONENT_ONLY",
            },
            "direct_packed_gate_up_swiglu_raw_observation": {
                **_receipt_reference(gate_up_fused_raw),
                "sealed": bool(gate_up_fused_raw.get("sealed")),
                "p50_component_host_wall_speedup_ratio": raw_fused_timing.get("p50_component_host_wall_speedup_ratio"),
                "observation_state": "UNSEALED_SUPPORTING_OBSERVATION_NOT_FRONTIER_AUTHORITY",
            },
            "partial_gpu_profile_negative": _receipt_reference(partial_gpu_profile_negative),
            "claim_boundary": (
                "evidence inventory only: SSE transport remains unqualified, rejected SIMD remains excluded, "
                "and gate/up timing is component-only rather than model TPS"
            ),
        }
        worker_results = self._latest_worker_results(worker)
        material_marker = {
            "completed_candidate_count": worker_results.get("completed_candidate_count"),
            "last_material_progress_at": worker_results.get("last_material_progress_at"),
            "pack_completed_tensors": _document_fields(pack_status, "progress").get("progress", {}).get("completed_tensors")
            if isinstance(_document_fields(pack_status, "progress").get("progress"), Mapping)
            else None,
            "pack_phase": pack_status.get("phase"),
            "runtime_phase": runtime_status_document.get("phase"),
            "runtime_receipt_seal": runtime.get("seal_sha256"),
            "complete_manifest_seal": manifest.get("seal_sha256"),
            "admission_seal": admission.get("seal_sha256"),
        }
        activity_marker = {
            "candidate_count": worker_results.get("candidate_count"),
            "current_experiment_id": worker_results.get("current_experiment_id"),
            "current_representation": worker_results.get("current_representation"),
        }
        return {
            "model": spec.key,
            "model_family": spec.model_family,
            "source_authority": {"identity": identity, "current_revalidation": revalidation},
            "current_complete_champion": current_complete_champion,
            "current_component_champion": component_champion,
            "current_capability_frontier": {
                "capability_receipt": _receipt_reference(capability),
                "current_capable_component_field": component_champion.get("current_capable"),
                "claim_boundary": "no component candidate becomes capable without the sealed capability receipt",
            },
            "current_bpw_frontier": {
                "complete_physical_bpw": self._complete_bpw(manifest),
                "component_lowest_bpw": (
                    component_champion.get("current_lowest_bpw_component", {}).get("physical_bpw")
                    if isinstance(component_champion.get("current_lowest_bpw_component"), Mapping)
                    else None
                ),
                "manifest": _receipt_reference(manifest),
                "component_frontier": _receipt_reference(frontier),
            },
            "current_complete_token_profile": complete_token_profile,
            "direct_packed_generation": direct_generation,
            "qwen30_canonical_template_evidence": qwen30_canonical_template_evidence,
            "kernel_operational_receipt": kernel,
            "hcli": hcli,
            "tg3": {"watchdog": tg3_status, "receipt": tg3},
            "state_kv": _receipt_reference(state_kv),
            "worker": worker,
            "worker_state": worker_state,
            "pack_status": pack_status,
            "latest_results": worker_results,
            "process": {
                "worker": _pid_snapshot(worker_results.get("pid")),
                "runtime_watchdog": _pid_snapshot(runtime_status_document.get("pid")),
                "pack_worker": _pid_snapshot(_document_fields(pack_status, "pid").get("pid")),
            },
            "material_marker": material_marker,
            "activity_marker": activity_marker,
            "shared_knowledge": shared,
            "paths": {name: str(path) for name, path in paths.items()},
        }

    @staticmethod
    def _material_liveness(
        observation: Mapping[str, Any], previous: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        worker = observation.get("worker") if isinstance(observation.get("worker"), Mapping) else {}
        worker_document = worker.get("document") if isinstance(worker.get("document"), Mapping) else {}
        heartbeat = worker_document.get("heartbeat")
        marker = observation.get("material_marker") if isinstance(observation.get("material_marker"), Mapping) else {}
        activity = observation.get("activity_marker") if isinstance(observation.get("activity_marker"), Mapping) else {}
        pid = observation.get("process", {}).get("worker") if isinstance(observation.get("process"), Mapping) else {}
        if not previous:
            return {
                "state": "OBSERVED_BASELINE_REQUIRES_NEXT_SAMPLE",
                "worker_pid": pid,
                "heartbeat": heartbeat,
                "material_marker": marker,
                "activity_marker": activity,
                "claim_boundary": "a first observation cannot establish liveness or material progress",
            }
        previous_marker = previous.get("material_marker") if isinstance(previous.get("material_marker"), Mapping) else {}
        previous_activity = previous.get("activity_marker") if isinstance(previous.get("activity_marker"), Mapping) else {}
        prior_heartbeat = previous.get("heartbeat")
        required_marker_keys = set(marker)
        if not required_marker_keys.issubset(set(previous_marker)) or "activity_marker" not in previous:
            return {
                "state": "OBSERVATION_SCHEMA_MIGRATION_REQUIRES_NEXT_SAMPLE",
                "worker_pid": pid,
                "heartbeat": heartbeat,
                "prior_heartbeat": prior_heartbeat,
                "material_marker": marker,
                "prior_material_marker": previous_marker,
                "activity_marker": activity,
                "prior_activity_marker": previous_activity,
                "claim_boundary": "a changed observer marker schema is not physical frontier movement; wait for a comparable next sample",
            }
        material_changed = marker != previous_marker
        selection_changed = activity != previous_activity
        heartbeat_changed = heartbeat != prior_heartbeat
        if material_changed:
            state = "LIVE_MATERIAL_PROGRESS"
        elif selection_changed:
            state = "HEARTBEAT_OR_SELECTION_ONLY_REJECTED"
        elif heartbeat_changed:
            state = "HEARTBEAT_ONLY_REJECTED"
        else:
            state = "NO_NEW_MATERIAL_PROGRESS"
        return {
            "state": state,
            "worker_pid": pid,
            "heartbeat": heartbeat,
            "prior_heartbeat": prior_heartbeat,
            "material_marker": marker,
            "prior_material_marker": previous_marker,
            "activity_marker": activity,
            "prior_activity_marker": previous_activity,
            "claim_boundary": "heartbeat refreshes and candidate-selection churn are explicitly not treated as physical frontier movement",
        }

    @staticmethod
    def _select_profile_candidate(observation: Mapping[str, Any]) -> dict[str, Any] | None:
        champions = observation.get("current_component_champion")
        if not isinstance(champions, Mapping):
            return None
        choices = (
            champions.get("current_fastest_component"),
            champions.get("current_lowest_bpw_component"),
        )
        for row in choices:
            if not isinstance(row, Mapping):
                continue
            path = row.get("record_path")
            if not isinstance(path, str):
                continue
            candidate = _bounded_file_summary(Path(path), seal_expected=True)
            document = candidate.get("document")
            if not candidate.get("sealed") or not isinstance(document, Mapping):
                continue
            artifact = document.get("artifact") if isinstance(document.get("artifact"), Mapping) else {}
            artifact_path = artifact.get("path")
            expected_sha = artifact.get("sha256")
            if not isinstance(artifact_path, str) or not isinstance(expected_sha, str):
                continue
            return {
                "candidate_record_path": str(path),
                "candidate_record_seal_sha256": candidate.get("seal_sha256"),
                "candidate_id": document.get("candidate_id"),
                "artifact_path": artifact_path,
                "artifact_sha256": expected_sha,
                "artifact_codec": artifact.get("codec"),
                "representation": document.get("representation", {}).get("family")
                if isinstance(document.get("representation"), Mapping)
                else None,
            }
        return None

    @staticmethod
    def _native_runtime_task(spec: ModelSpec, observation: Mapping[str, Any]) -> dict[str, Any]:
        direct_generation = observation.get("direct_packed_generation")
        if (
            spec.key == "qwen30"
            and isinstance(direct_generation, Mapping)
            and direct_generation.get("state")
            == "OBSERVED_UNSEALED_DIRECT_PACKED_AUTOREGRESSIVE_TRACE_REQUIRES_CAPABILITY_QUALITY_GATE"
        ):
            return {
                "stage": "BLOCKED_CAPABILITY_COHERENCE_AFTER_DIRECT_PACKED_GENERATION",
                "status": "BLOCKED_CAPABILITY_COHERENCE_AFTER_DIRECT_PACKED_GENERATION",
                "blocker_category": "capability",
                "what_currently_prevents_next_gate": (
                    "a direct packed Metal-only autoregressive trace reached all 48 layers without opening the BF16 loader, "
                    "but its completion is unscored/incoherent and therefore cannot establish prompt dependence, manager capability, HCLI, or TPS"
                ),
                "mechanisms": [
                    {
                        "mechanism": "binary_scale_projection_error_under_complete_bpw_budget",
                        "physical_cause": "sign-plus-group-scale packing can accumulate projection error across the 48-layer residual path even while the complete artifact remains <=1.5 BPW",
                    },
                    {
                        "mechanism": "router_top_k_instability",
                        "physical_cause": "small packed router-logit errors can alter top-8 expert selection, changing the entire routed expert wave for a token",
                    },
                    {
                        "mechanism": "misallocated_sparse_residual_budget",
                        "physical_cause": "a <=1.5 COMPLETE BPW residual allowance may be spent on low-leverage weights rather than sensitive router/gate/up/down organs",
                    },
                    {
                        "mechanism": "final_logit_path_sensitivity",
                        "physical_cause": "final norm/lm_head representation error can collapse otherwise usable hidden states into poor greedy tokens",
                    },
                ],
                "cheapest_discriminating_test": {
                    "test": "run a source-bound Qwen30 sensitive-router control experiment comparing pack-compatible binary grouping and sparse-residual variants, with deterministic top-8 route stability controls",
                    "distinguishes": "direct sign-scale error, router route instability, and residual-allocation benefit before changing the complete artifact",
                    "does_not_claim": "prompt coherence, capability, HCLI, clean TPS, TG10, TG3, or tournament qualification",
                },
                "prior_initialization": {
                    "direct_packed_generation": direct_generation,
                    "complete_bpw": observation.get("current_complete_champion", {}).get("manifest", {}).get("complete_physical_bpw")
                    if isinstance(observation.get("current_complete_champion"), Mapping)
                    else None,
                    "source_authority": observation.get("source_authority"),
                    "state_kv_evidence": observation.get("state_kv"),
                },
                "conditions": {
                    "PASS": "the bounded control identifies a <=1.5-BPW-compatible sensitive-organ representation with materially better deterministic route fidelity; update a transfer prior only",
                    "FAIL": "no compatible variant improves route controls, or source binding fails; seal a narrow negative result and keep the final-logit mechanism open",
                    "REOPEN_LATER": "the full artifact representation, source revalidation, router kernel, or exact direct-runtime trace materially changes",
                },
            }
        complete = observation.get("current_complete_champion") if isinstance(observation.get("current_complete_champion"), Mapping) else {}
        manifest = complete.get("manifest") if isinstance(complete.get("manifest"), Mapping) else {}
        admitted = complete.get("admission") if isinstance(complete.get("admission"), Mapping) else {}
        artifact_ready = bool(manifest.get("sealed") and admitted.get("sealed"))
        if artifact_ready:
            blocker = "native_runtime"
            stage = "BLOCKED_NATIVE_RUNTIME"
            blocker_text = (
                "a source-bound complete artifact is present, but the sealed exact native full-token runtime receipt does not exist"
            )
            mechanisms = [
                {
                    "mechanism": "complete_artifact_loader_binding",
                    "physical_cause": "catalog/packed tensor reader may not bind every exact native tensor and control input",
                },
                {
                    "mechanism": "full_decoder_serial_graph",
                    "physical_cause": "the architecture-specific 48-layer decoder path, final norm, lm_head, sampler, and feedback loop may be incomplete",
                },
                {
                    "mechanism": "native_command_graph_assembly",
                    "physical_cause": "available component kernels may not yet compose into one no-fallback Metal token graph",
                },
            ]
            cheapest = {
                "test": "admit one sealed packed component artifact through the exact candidate reader and validate its physical envelope before attempting a full token",
                "distinguishes": "reader/catalog corruption from the still-unimplemented decoder/command-graph mechanisms",
                "does_not_claim": "generation, HCLI, full-token TPS, capability, TG10, or TG3",
            }
        else:
            blocker = "bytes"
            stage = "BLOCKED_COMPLETE_ARTIFACT_AND_NATIVE_RUNTIME"
            blocker_text = (
                "the complete artifact/admission prerequisite is not yet sealed, and a native full-token runtime must remain blocked"
            )
            mechanisms = [
                {
                    "mechanism": "sealed_pack_completion",
                    "physical_cause": "not every source-bound tensor/control record is present in the complete artifact ledger",
                },
                {
                    "mechanism": "admission_catalog_legality",
                    "physical_cause": "the complete catalog may not yet pass strict native artifact admission",
                },
                {
                    "mechanism": "decoder_dependency_order",
                    "physical_cause": "even a completed pack still needs architecture-specific decoder/state graph wiring",
                },
            ]
            cheapest = {
                "test": "continue the detached packer to its next sealed material cursor and verify the existing bounded component artifact envelope",
                "distinguishes": "ongoing physical pack/admission work from an unsupported claim that a runtime can start today",
                "does_not_claim": "a complete model, generation, HCLI, or TPS",
            }
        return {
            "stage": stage,
            "status": stage,
            "blocker_category": blocker,
            "what_currently_prevents_next_gate": blocker_text,
            "mechanisms": mechanisms,
            "cheapest_discriminating_test": cheapest,
            "prior_initialization": {
                "complete_artifact": complete,
                "state_kv_evidence": observation.get("state_kv"),
                "component_kernel_evidence": [
                    _receipt_reference(row)
                    for row in observation.get("shared_knowledge", {}).get("component_kernel_receipts", [])
                    if isinstance(row, Mapping)
                ]
                if isinstance(observation.get("shared_knowledge"), Mapping)
                else [],
                "architecture_specific_rule": (
                    "do not transfer Qwen30 attention assumptions directly to Qwen80 DeltaNet"
                    if spec.key == "qwen80"
                    else "reuse shared MoE/Metal evidence only after direct Qwen30 validation"
                ),
            },
            "conditions": {
                "PASS": "a sealed exact full-token receipt proves all native layers/tensors, no fallback, and a complete model token loop",
                "FAIL": "reader/component evidence is invalid, or the native decoder cannot execute the bound artifact; seal the narrow failure and do not report TPS",
                "REOPEN_LATER": "the artifact admission seal, runtime implementation, or source-bound kernel implementation materially changes",
            },
        }

    @staticmethod
    def _qwen30_canonical_template_task(observation: Mapping[str, Any]) -> dict[str, Any] | None:
        """Turn the first real template/runtime evidence into one narrow next test.

        The exact decoder receipt is a runtime gate, not a quality pass.  The
        SSE smoke proves transport framing only, while the SIMD receipt is a
        rejected candidate.  Keeping those facts adjacent avoids the common
        error of treating either a component speedup or a working HTTP stream
        as evidence that the manager is coherent.
        """

        evidence = observation.get("qwen30_canonical_template_evidence")
        if not isinstance(evidence, Mapping):
            return None
        exact = evidence.get("exact_native_runtime")
        transport = evidence.get("unqualified_chat_sse_transport")
        simd = evidence.get("rejected_template_simdgroup_candidate")
        gate_up = evidence.get("direct_packed_gate_up_pair")
        if not all(isinstance(row, Mapping) and row.get("sealed") for row in (exact, transport, simd, gate_up)):
            return None
        if exact.get("status") != RUNTIME_STATUS:
            return None
        if transport.get("status") != "EARNED_DIRECT_PACKED_NATIVE_CHAT_SSE_TRANSPORT_HCLI_UNQUALIFIED":
            return None
        if simd.get("status") != "REJECTED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY":
            return None
        if gate_up.get("status") != "EARNED_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_PAIR_COMPONENT_NOT_MODEL_TPS":
            return None
        return {
            "stage": "BLOCKED_CANONICAL_TEMPLATE_COHERENCE_KERNEL_INTEGRATION_DIAGNOSIS",
            "status": "BLOCKED_CANONICAL_TEMPLATE_COHERENCE_KERNEL_INTEGRATION_DIAGNOSIS",
            "blocker_category": "capability",
            "what_currently_prevents_next_gate": (
                "the canonical direct-packed decoder is runtime-valid, but chat SSE remains explicitly unqualified "
                "with unscored/incoherent output. The scalar control is the only usable template path; the all-layer "
                "SIMDgroup candidate is rejected despite BOS parity, and the exact-parity gate/up fused component remains unintegrated."
            ),
            "mechanisms": [
                {
                    "mechanism": "template_simdgroup_reduction_or_indexing_divergence",
                    "physical_cause": (
                        "a SIMDgroup reduction, lane/index mapping, or accumulation-order error can preserve a BOS control "
                        "yet change prompt-template hidden states and greedy tokens across the all-layer path"
                    ),
                },
                {
                    "mechanism": "gate_up_fusion_command_topology_not_yet_integrated",
                    "physical_cause": (
                        "the direct-packed gate/up pair has exact component parity and an observed one-dispatch speedup, "
                        "but full-runtime command ordering, activation boundaries, and expert-wave integration remain unproven"
                    ),
                },
                {
                    "mechanism": "sensitive_organ_representation_error_under_complete_bpw_budget",
                    "physical_cause": (
                        "binary sign-plus-scale error in routed gate/up organs can compound through SwiGLU and residual paths; "
                        "any mitigation must still fit a newly measured complete artifact at <=1.5 BPW"
                    ),
                },
            ],
            "cheapest_discriminating_test": {
                "test": (
                    "run one source-bound layer-0 expert-0 gate/up paired-SwiGLU control: compare raw-source teacher output, "
                    "two-projection direct-binary output, and algebraically paired direct-binary output across deterministic activations"
                ),
                "distinguishes": (
                    "representation error in a sensitive routed organ from fusion algebra on the same direct-packed payload; "
                    "it leaves the template-only SIMD discrepancy isolated for a separate rejected-candidate regression repair"
                ),
                "does_not_claim": (
                    "prompt coherence, HCLI pass, clean TPS, TG10/TG3, a repack, runtime integration, or manager capability"
                ),
            },
            "prior_initialization": {
                "exact_native_runtime": exact,
                "unqualified_sse_transport": transport,
                "rejected_simd_candidate": simd,
                "unintegrated_gate_up_component": gate_up,
                "earlier_sensitive_router_repack_proposal_must_remain_unapplied": True,
                "source_authority": observation.get("source_authority"),
                "state_kv_evidence": observation.get("state_kv"),
                "qwen80_transfer_rule": (
                    "transfer only the direct-packed gated-MLP component methodology to Qwen80 routed/shared experts; "
                    "do not transfer Qwen30 template/SIMD or attention assumptions into Qwen80 DeltaNet without direct evidence"
                ),
            },
            "conditions": {
                "PASS": (
                    "the bounded source-bound control is sealed and identifies whether binary gated-MLP representation error is material "
                    "while paired algebra matches the same packed payload; update a component-only cross-family prior, not a model-quality pass"
                ),
                "FAIL": (
                    "source binding, pair geometry, budget accounting, or deterministic control fails; seal only that narrow negative result and "
                    "keep all three mechanisms open where the failed control was not discriminating"
                ),
                "REOPEN_LATER": (
                    "the source revalidation, admitted representation, exact runtime, SIMD candidate revision, or gate/up integration materially changes"
                ),
            },
        }

    @staticmethod
    def _runtime_profile_task(spec: ModelSpec, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Future-ready task emitted only after a strict runtime receipt exists."""

        return {
            "stage": "READY_FOR_COMPLETE_TOKEN_PROFILE",
            "status": "READY_FOR_COMPLETE_TOKEN_PROFILE",
            "blocker_category": "utilization",
            "what_currently_prevents_next_gate": "exact native runtime exists, but a clean full-token latency attribution is required before TPS work",
            "mechanisms": [
                {"mechanism": "packed_decode_dot_fusion", "physical_cause": "weight decode/materialization can dominate projection latency"},
                {"mechanism": "expert_wave_residency", "physical_cause": "route-dependent expert cache misses and gathers can dominate MoE time"},
                {"mechanism": "command_graph_and_readback", "physical_cause": "serial command buffers, synchronization, and logits readback can dominate short-token latency"},
            ],
            "cheapest_discriminating_test": {
                "test": "run the native complete-token profiler over the exact sealed artifact with bucket timings totaling at least 98% of elapsed token time",
                "distinguishes": "decode, expert/residency, and graph/readback bottlenecks before changing representation",
                "does_not_claim": "100 TPS until the official clean full-token operational receipt is sealed",
            },
            "prior_initialization": {
                "exact_runtime_receipt": observation.get("current_complete_token_profile"),
                "shared_kernel_genome": observation.get("shared_knowledge", {}).get("kernel_genome")
                if isinstance(observation.get("shared_knowledge"), Mapping)
                else {},
                "model_family": spec.model_family,
            },
            "conditions": {
                "PASS": "a sealed profile accounts for >=98% of real complete-token latency with fallback=0 and identifies one dominant physical bucket",
                "FAIL": "incomplete buckets, fallback, or an invalid receipt; seal the narrow cause and reopen only after runtime/profiler repair",
                "REOPEN_LATER": "a representation, state layout, kernel, or command graph materially changes",
            },
        }

    def _build_experiment(
        self,
        spec: ModelSpec,
        observation: Mapping[str, Any],
        peer: Mapping[str, Any],
        liveness: Mapping[str, Any],
    ) -> dict[str, Any]:
        task = self._qwen30_canonical_template_task(observation) if spec.key == "qwen30" else None
        if task is None:
            task = self._runtime_profile_task(spec, observation) if self._runtime_earned(
                observation.get("current_complete_token_profile", {}).get("exact_runtime_receipt", {})
                if isinstance(observation.get("current_complete_token_profile"), Mapping)
                else {}
            ) else self._native_runtime_task(spec, observation)
        profile_candidate = self._select_profile_candidate(observation)
        evidence = {
            "current_complete_champion": observation.get("current_complete_champion"),
            "current_capability_frontier": observation.get("current_capability_frontier"),
            "current_bpw_frontier": observation.get("current_bpw_frontier"),
            "current_complete_token_profile": observation.get("current_complete_token_profile"),
            "kernel_genome": observation.get("shared_knowledge", {}).get("kernel_genome")
            if isinstance(observation.get("shared_knowledge"), Mapping)
            else {},
            "representation_genome": observation.get("shared_knowledge", {}).get("representation_genome")
            if isinstance(observation.get("shared_knowledge"), Mapping)
            else {},
            "scheduler_genome": observation.get("shared_knowledge", {}).get("scheduler_genome")
            if isinstance(observation.get("shared_knowledge"), Mapping)
            else {},
            "negative_science": observation.get("shared_knowledge", {}).get("negative_science")
            if isinstance(observation.get("shared_knowledge"), Mapping)
            else {},
            "state_kv": observation.get("state_kv"),
            "direct_packed_generation": observation.get("direct_packed_generation"),
            "canonical_template_runtime_kernel_evidence": observation.get("qwen30_canonical_template_evidence"),
            "source_authority": {
                "identity": _receipt_reference(observation.get("source_authority", {}).get("identity", {}))
                if isinstance(observation.get("source_authority"), Mapping)
                else {},
                "current_revalidation": _receipt_reference(
                    observation.get("source_authority", {}).get("current_revalidation", {})
                )
                if isinstance(observation.get("source_authority"), Mapping)
                else {},
            },
            "peer_latest_results": peer.get("latest_results"),
            "peer_complete_champion": peer.get("current_complete_champion"),
            "worker_liveness": liveness,
        }
        fingerprint = _digest(
            {
                "model": spec.key,
                "stage": task["stage"],
                "source_identity": observation.get("source_authority", {}).get("identity", {}).get("seal_sha256")
                if isinstance(observation.get("source_authority"), Mapping)
                else None,
                "revalidation": observation.get("source_authority", {}).get("current_revalidation", {}).get("seal_sha256")
                if isinstance(observation.get("source_authority"), Mapping)
                else None,
                "manifest": observation.get("current_complete_champion", {}).get("manifest", {}).get("seal_sha256")
                if isinstance(observation.get("current_complete_champion"), Mapping)
                else None,
                "admission": observation.get("current_complete_champion", {}).get("admission", {}).get("seal_sha256")
                if isinstance(observation.get("current_complete_champion"), Mapping)
                else None,
                "runtime": observation.get("current_complete_token_profile", {}).get("exact_runtime_receipt", {}).get("seal_sha256")
                if isinstance(observation.get("current_complete_token_profile"), Mapping)
                else None,
                "direct_packed_generation": observation.get("direct_packed_generation", {}).get("receipt", {}).get("document_sha256")
                if isinstance(observation.get("direct_packed_generation"), Mapping)
                else None,
                "canonical_template_runtime_kernel_evidence": observation.get("qwen30_canonical_template_evidence"),
                "profile_candidate": profile_candidate,
                # The ledgers are read before every cycle and fully embedded
                # above, but their high-frequency component rows do not reopen
                # the same runtime task.  Only a direct source/artifact/runtime
                # change below can make a task eligible for reissue.  This is
                # what prevents a busy peer candidate worker from causing an
                # endless stream of superficial replicas of one experiment.
                "knowledge_read_contract": "all four shared ledgers read before selection; high-frequency rows alone do not reopen an unchanged task",
            }
        )
        task_id = f"{spec.key}-{task['stage'].lower()}-{fingerprint[:20]}"
        return seal(
            {
                "schema": EXPERIMENT_SCHEMA,
                "record_id": f"scientific-optimizer:{task_id}",
                "task_id": task_id,
                "recorded_at": _utc_now(),
                "model": spec.key,
                "model_family": spec.model_family,
                "status": task["status"],
                "experiment_kind": "bounded_component_profile_then_native_runtime_handoff",
                "evidence_fingerprint_sha256": fingerprint,
                "required_pre_experiment_reads": evidence,
                "reasoning": task,
                "safe_component_profile_candidate": profile_candidate,
                "execution_policy": {
                    "run_only_if": "sealed candidate record + regular bounded artifact + source-bound artifact SHA-256",
                    "before_native_runtime": "only packed-artifact I/O/header profile may execute; it is never model TPS",
                    "after_native_runtime": "publish a profile handoff; the runtime owner must perform exact full-token measurement",
                    "no_superficial_rerun": "same evidence fingerprint and component artifact never execute twice; a material seal/artifact change is required",
                },
                "claim_boundary": {
                    "not_a_controller": True,
                    "not_a_runtime_or_hcli_receipt": True,
                    "not_a_tps_or_tg_receipt": True,
                    "not_a_capability_or_tournament_receipt": True,
                    "raw_bf16_is_teacher_only": True,
                },
            }
        )

    def _run_component_profile(
        self, spec: ModelSpec, experiment: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, bool]:
        """Run a bounded physical artifact profile, or seal a narrow failure.

        This is intentionally not a decoder benchmark.  It verifies a selected
        candidate's sealed record, artifact byte binding, and actual envelope
        parse/read time.  It answers whether the component storage/reader
        premise is viable before a future exact native decoder consumes it.
        """

        candidate = experiment.get("safe_component_profile_candidate")
        if not isinstance(candidate, Mapping):
            return None, False
        artifact_path_value = candidate.get("artifact_path")
        expected_sha = candidate.get("artifact_sha256")
        record_path_value = candidate.get("candidate_record_path")
        if not isinstance(artifact_path_value, str) or not isinstance(expected_sha, str) or not isinstance(record_path_value, str):
            return None, False
        task_id = str(experiment["task_id"])
        # A profile binds a physical artifact, not a changing scheduler row.
        # Reusing it across later peer-ledger snapshots makes the no-rerun
        # contract concrete rather than advisory.
        profile_key = f"{spec.key}-{expected_sha}"
        profile_path = self.profiles_dir / f"{profile_key}.json"
        existing = _bounded_file_summary(profile_path, seal_expected=True)
        if existing.get("sealed"):
            return (
                existing.get("document") if isinstance(existing.get("document"), Mapping) else None,
                False,
            )
        artifact_path = Path(artifact_path_value)
        evolution_root = self._paths(spec)["evolution_root"]
        failure: str | None = None
        metrics: dict[str, Any] = {}
        try:
            record = _bounded_file_summary(Path(record_path_value), seal_expected=True)
            if not record.get("sealed"):
                raise ScientificOptimizerError("candidate record is not a sealed receipt")
            if not _is_regular_under(artifact_path, evolution_root):
                raise ScientificOptimizerError("candidate artifact is not a regular file under the model evolution root")
            artifact_bytes = int(artifact_path.stat().st_size)
            if artifact_bytes <= 12:
                raise ScientificOptimizerError("candidate artifact is too short for a Gravity envelope")
            if artifact_bytes > MAX_COMPONENT_PROFILE_BYTES:
                return None, False
            actual_sha = _sha256_file(artifact_path)
            if actual_sha != expected_sha:
                raise ScientificOptimizerError("candidate artifact SHA-256 differs from its sealed record")
            started = time.perf_counter()
            with artifact_path.open("rb") as handle:
                payload = handle.read()
            elapsed = time.perf_counter() - started
            if len(payload) != artifact_bytes:
                raise ScientificOptimizerError("artifact read length differs from the file identity")
            magic = payload[:8]
            header_bytes = int.from_bytes(payload[8:12], "little", signed=False)
            if header_bytes <= 1 or 12 + header_bytes > len(payload):
                raise ScientificOptimizerError("artifact envelope has an invalid header length")
            try:
                header = json.loads(payload[12 : 12 + header_bytes].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ScientificOptimizerError("artifact envelope header is not JSON") from exc
            if not isinstance(header, Mapping) or not isinstance(header.get("schema"), str):
                raise ScientificOptimizerError("artifact envelope header has no codec schema")
            declared_codec = candidate.get("artifact_codec")
            if isinstance(declared_codec, str) and header.get("schema") != declared_codec:
                raise ScientificOptimizerError("artifact envelope codec does not match its sealed candidate record")
            metrics = {
                "artifact_bytes": artifact_bytes,
                "artifact_sha256": actual_sha,
                "magic_hex": magic.hex(),
                "codec_schema": header.get("schema"),
                "header_bytes": header_bytes,
                "payload_body_bytes": artifact_bytes - 12 - header_bytes,
                "read_elapsed_seconds": elapsed,
                "artifact_read_mib_per_second": (artifact_bytes / 1024**2) / max(elapsed, 1e-12),
            }
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        status = (
            "PASS_SOURCE_BOUND_PACKED_COMPONENT_IO_PROFILE_NOT_MODEL_TPS"
            if failure is None
            else "FAIL_SOURCE_BOUND_PACKED_COMPONENT_IO_PROFILE"
        )
        profile = seal(
            {
                "schema": COMPONENT_PROFILE_SCHEMA,
                "record_id": f"component-profile:{task_id}",
                "recorded_at": _utc_now(),
                "model": spec.key,
                "task_id": task_id,
                "profile_key": profile_key,
                "profile_path": str(profile_path),
                "status": status,
                "candidate": dict(candidate),
                "metrics": metrics,
                "failure": failure,
                "pass_condition": "sealed candidate artifact hash and physical Gravity envelope read/header validation succeed",
                "fail_condition": "missing/invalid candidate record, artifact identity, hash, or envelope",
                "reopen_condition": "a new artifact SHA-256, a repaired candidate record, or a material reader implementation change",
                "claim_boundary": "artifact I/O/header profile only; it is not a custom Metal kernel result, decoder, generation, HCLI, capability, or tokens-per-second measurement",
            }
        )
        _atomic_json(profile_path, profile)
        return profile, True

    def _append_profile_knowledge(self, profile: Mapping[str, Any]) -> None:
        task_id = profile.get("task_id")
        model = profile.get("model")
        if not isinstance(task_id, str) or not isinstance(model, str):
            return
        if profile.get("status") == "PASS_SOURCE_BOUND_PACKED_COMPONENT_IO_PROFILE_NOT_MODEL_TPS":
            row = seal(
                {
                    "schema": COMPONENT_PROFILE_SCHEMA,
                    "record_id": f"kernel-profile:{task_id}",
                    "recorded_at": _utc_now(),
                    "model": model,
                    "status": "PASS_COMPONENT_ARTIFACT_IO_PROFILE_NOT_FULL_MODEL",
                    "profile_path": profile.get("profile_path"),
                    "profile_seal_sha256": profile.get("seal_sha256"),
                    "mechanism": "packed_artifact_reader_envelope",
                    "measurement": profile.get("metrics"),
                    "reopen_conditions": "new packed artifact or native reader implementation",
                    "claim_boundary": "shared kernel prior only; never model TPS or operational kernel qualification",
                }
            )
            _append_jsonl_once(self.shared_kernel_path, row, record_id=str(row["record_id"]))
            scheduler = seal(
                {
                    "schema": "hawking.ascension.qwen_scientific_optimizer_scheduler.v1",
                    "record_id": f"scheduler:{task_id}",
                    "recorded_at": _utc_now(),
                    "model": model,
                    "status": "PASS_BOUNDED_COMPONENT_PROFILE_ADVANCED",
                    "selected_experiment": task_id,
                    "next_gate": "sealed exact native full-token runtime",
                    "claim_boundary": "scheduling evidence only; no runtime/TPS inference",
                }
            )
            _append_jsonl_once(self.shared_scheduler_path, scheduler, record_id=str(scheduler["record_id"]))
            return
        if profile.get("status") == "FAIL_SOURCE_BOUND_PACKED_COMPONENT_IO_PROFILE":
            candidate = profile.get("candidate") if isinstance(profile.get("candidate"), Mapping) else {}
            artifact_sha = candidate.get("artifact_sha256")
            if not isinstance(artifact_sha, str):
                artifact_sha = "unknown"
            negative = seal(
                {
                    "schema": NEGATIVE_SCHEMA,
                    "record_id": f"negative-optimizer:{task_id}",
                    "recorded_at": _utc_now(),
                    "status": "BURIED",
                    "mechanism": "packed_artifact_reader_envelope",
                    "mechanism_key": f"optimizer:packed_artifact_reader_envelope:{model}:{artifact_sha}",
                    "model_geometry": model,
                    "measured_outcome": {"failure": profile.get("failure")},
                    "failure_reason": "source_bound_component_artifact_profile_failed",
                    "reopen_condition": profile.get("reopen_condition"),
                    "evidence_binding": {
                        "profile_path": profile.get("profile_path"),
                        "profile_seal_sha256": profile.get("seal_sha256"),
                    },
                    "claim_boundary": "narrow artifact/reader premise only; it does not bury a representation family or model architecture",
                }
            )
            _append_jsonl_once(self.shared_negative_path, negative, record_id=str(negative["record_id"]))

    def _ingest_qwen30_canonical_template_evidence(
        self, observation: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, bool]:
        """Seal current Qwen30 runtime/kernel facts into the shared research plane.

        This deliberately records a rejected SIMD implementation as a narrow
        negative and a gate/up result as a component-only kernel prior.  It
        never promotes either fact into a runtime, HCLI, quality, or TPS gate.
        """

        evidence = observation.get("qwen30_canonical_template_evidence")
        if not isinstance(evidence, Mapping):
            return None, False
        exact = evidence.get("exact_native_runtime")
        transport = evidence.get("unqualified_chat_sse_transport")
        simd = evidence.get("rejected_template_simdgroup_candidate")
        gate_up = evidence.get("direct_packed_gate_up_pair")
        if not all(isinstance(row, Mapping) and row.get("sealed") for row in (exact, transport, simd, gate_up)):
            return None, False
        required_statuses = {
            "exact": (exact, RUNTIME_STATUS),
            "transport": (transport, "EARNED_DIRECT_PACKED_NATIVE_CHAT_SSE_TRANSPORT_HCLI_UNQUALIFIED"),
            "simd": (simd, "REJECTED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY"),
            "gate_up": (gate_up, "EARNED_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_PAIR_COMPONENT_NOT_MODEL_TPS"),
        }
        if any(row.get("status") != expected for row, expected in required_statuses.values()):
            return None, False
        fingerprint = _digest(
            {
                "exact_runtime": exact.get("seal_sha256"),
                "unqualified_transport": transport.get("seal_sha256"),
                "simd_rejection": simd.get("seal_sha256"),
                "gate_up_component": gate_up.get("seal_sha256"),
                "raw_fused_observation": evidence.get("direct_packed_gate_up_swiglu_raw_observation", {}).get("document_sha256")
                if isinstance(evidence.get("direct_packed_gate_up_swiglu_raw_observation"), Mapping)
                else None,
            }
        )
        path = self.root / "evidence-frontier" / f"QWEN30_CANONICAL_TEMPLATE_KERNEL_EVIDENCE_{fingerprint[:24]}.json"
        existing = _bounded_file_summary(path, seal_expected=True)
        created = False
        if existing.get("sealed") and isinstance(existing.get("document"), Mapping):
            document = dict(existing["document"])
        else:
            document = seal(
                {
                    "schema": CANONICAL_TEMPLATE_EVIDENCE_SCHEMA,
                    "record_id": f"qwen30-canonical-template-kernel-evidence:{fingerprint}",
                    "recorded_at": _utc_now(),
                    "status": "INGESTED_RUNTIME_TRANSPORT_REJECTION_AND_COMPONENT_KERNEL_EVIDENCE",
                    "evidence_path": str(path),
                    "evidence_fingerprint_sha256": fingerprint,
                    "exact_native_runtime": dict(exact),
                    "unqualified_chat_sse_transport": dict(transport),
                    "rejected_template_simdgroup_candidate": dict(simd),
                    "direct_packed_gate_up_pair_component": dict(gate_up),
                    "direct_packed_gate_up_swiglu_raw_observation": evidence.get("direct_packed_gate_up_swiglu_raw_observation"),
                    "interpretation": {
                        "runtime": "canonical template/full-token native execution is earned only as a runtime gate",
                        "transport": "SSE framing is observed, but coherence is unscored and the transport is not an HCLI pass",
                        "simd": "all-layer SIMDgroup candidate is rejected for template-token divergence; BOS parity alone is insufficient",
                        "gate_up": "one-dispatch gate/up component parity and timing are a kernel frontier prior only until runtime integration and fresh complete-token profiling",
                    },
                    "qwen80_transfer_rule": (
                        "Qwen80 may reuse the direct-packed gated-MLP fusion test method for routed/shared experts after direct parity. "
                        "Do not transfer the Qwen30 template-SIMD result to DeltaNet/recurrent state or call it a Qwen80 pass."
                    ),
                    "claim_boundary": (
                        "evidence ingestion only; no SIMD candidate is applied, no pack is changed, and this is not a coherence, HCLI, "
                        "capability, clean TPS, TG, or tournament receipt"
                    ),
                }
            )
            _atomic_json(path, document)
            created = True

        simd_seal = str(simd.get("seal_sha256"))
        negative = seal(
            {
                "schema": NEGATIVE_SCHEMA,
                "record_id": f"negative-optimizer:qwen30-template-simdgroup:{simd_seal}",
                "recorded_at": _utc_now(),
                "status": "BURIED",
                "model_geometry": "qwen3_moe:all_layer_direct_packed_template_path",
                "mechanism": "packed_binary_simdgroup_template_parity",
                "mechanism_key": f"optimizer:qwen30-packed-binary-simdgroup-template:{simd_seal}",
                "failure_reason": "BOS control parity did not extend to either exact source-template prompt-token path",
                "measured_outcome": {"failures": simd.get("failures")},
                "evidence_binding": {
                    "simd_template_parity_receipt": _receipt_reference(simd),
                    "canonical_evidence_path": str(path),
                    "canonical_evidence_seal_sha256": document.get("seal_sha256"),
                },
                "reopen_condition": (
                    "a new SIMD candidate revision proves exact parity on the same two source-template controls, or the direct-packed "
                    "template/runtime implementation materially changes"
                ),
                "claim_boundary": "narrow rejected Qwen30 all-layer SIMD implementation/configuration only; not a universal SIMD or representation ban",
            }
        )
        negative_added = _append_jsonl_once(self.shared_negative_path, negative, record_id=str(negative["record_id"]))
        gate_up_seal = str(gate_up.get("seal_sha256"))
        kernel = seal(
            {
                "schema": "hawking.ascension.qwen_direct_packed_component_kernel_frontier.v1",
                "record_id": f"kernel-frontier:qwen30-direct-packed-gate-up:{gate_up_seal}",
                "recorded_at": _utc_now(),
                "status": "PASS_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_COMPONENT_FRONTIER_NOT_MODEL_TPS",
                "model": "qwen30",
                "model_family": "qwen3_moe",
                "mechanism": "direct_packed_gate_up_pair_one_dispatch",
                "mechanism_key": f"optimizer:qwen30-direct-packed-gate-up:{gate_up_seal}",
                "component_receipt": _receipt_reference(gate_up),
                "component_speedup_ratio": gate_up.get("p50_component_host_wall_speedup_ratio"),
                "integration_state": "UNINTEGRATED_COMPONENT_ONLY",
                "qwen80_transfer_rule": (
                    "validate this only on Qwen80 routed/shared gated MLPs with its own direct-packed parity; exclude DeltaNet/recurrent state "
                    "until it has separate architecture-specific evidence"
                ),
                "reopen_conditions": "direct-packed payload/layout, kernel implementation, or runtime integration changes",
                "claim_boundary": "component kernel frontier only; not a full layer/model, generation, HCLI, clean TPS, TG, or tournament result",
            }
        )
        kernel_added = _append_jsonl_once(self.shared_kernel_path, kernel, record_id=str(kernel["record_id"]))
        return document, bool(created or negative_added or kernel_added)

    @staticmethod
    def _router_control_quality(reference: Any, reconstruction: Any) -> dict[str, Any]:
        """Deterministic source-router controls; deliberately not a prompt trace."""

        import numpy as np

        raw = np.ascontiguousarray(reference, dtype=np.float32).reshape(reference.shape[0], -1)
        packed = np.ascontiguousarray(reconstruction, dtype=np.float32).reshape(raw.shape)
        top_k = min(8, raw.shape[0])
        generator = np.random.default_rng(0xA5C3_30)
        overlaps: list[float] = []
        top1: list[float] = []
        relative: list[float] = []
        for _ in range(32):
            activation = generator.standard_normal(raw.shape[1], dtype=np.float32)
            expected = raw @ activation
            observed = packed @ activation
            expected_ids = np.argpartition(expected, -top_k)[-top_k:]
            observed_ids = np.argpartition(observed, -top_k)[-top_k:]
            overlaps.append(len(set(expected_ids.tolist()) & set(observed_ids.tolist())) / top_k)
            top1.append(float(int(np.argmax(expected)) == int(np.argmax(observed))))
            relative.append(float(np.linalg.norm(expected - observed) / max(float(np.linalg.norm(expected)), 1e-12)))
        return {
            "control_count": 32,
            "top_k": top_k,
            "mean_top_k_overlap": float(np.mean(overlaps)),
            "minimum_top_k_overlap": float(np.min(overlaps)),
            "top1_agreement": float(np.mean(top1)),
            "mean_router_logit_relative_l2": float(np.mean(relative)),
            "claim_boundary": "deterministic source-router controls only; not prompt-dependent routing, generation, capability, HCLI, or TPS",
        }

    @staticmethod
    def _paired_swiglu_control_quality(
        gate_source: Any,
        up_source: Any,
        gate_packed: Any,
        up_packed: Any,
    ) -> dict[str, Any]:
        """Compare a source SwiGLU teacher with two equivalent packed forms.

        This is intentionally CPU/source-component work.  It verifies neither
        the Metal implementation nor a prompt path; those require their own
        runtime receipts.  Its value is that it tells the runtime owner whether
        a sensitive direct-binary MLP organ is already materially noisy before
        attempting to integrate the known component kernel win.
        """

        import numpy as np

        gate = np.ascontiguousarray(gate_source, dtype=np.float32).reshape(gate_source.shape[0], -1)
        up = np.ascontiguousarray(up_source, dtype=np.float32).reshape(up_source.shape[0], -1)
        gate_rebuilt = np.ascontiguousarray(gate_packed, dtype=np.float32).reshape(gate.shape)
        up_rebuilt = np.ascontiguousarray(up_packed, dtype=np.float32).reshape(up.shape)
        if gate.shape != up.shape or gate_rebuilt.shape != gate.shape or up_rebuilt.shape != up.shape:
            raise ScientificOptimizerError("gate/up source and direct-binary geometries must match")
        controls = ((0x51A7_3001, 0.25), (0x51A7_3002, 1.0), (0x51A7_3003, 4.0), (0x51A7_3004, 8.0))
        source_outputs: list[Any] = []
        separate_outputs: list[Any] = []
        paired_outputs: list[Any] = []
        for seed, scale in controls:
            generator = np.random.default_rng(seed)
            activation = generator.standard_normal((gate.shape[1], 4), dtype=np.float32) * np.float32(scale)
            raw_gate = gate @ activation
            raw_up = up @ activation
            packed_gate = gate_rebuilt @ activation
            packed_up = up_rebuilt @ activation
            # The two packed paths intentionally use the same direct-packed
            # values.  The paired form models algebraic gate/up fusion without
            # opening a raw weight body, materializing a decoded body, or
            # claiming GPU-kernel integration.
            separate = (packed_gate / (1.0 + np.exp(-np.clip(packed_gate, -60.0, 60.0)))) * packed_up
            paired = np.multiply(
                packed_gate / (1.0 + np.exp(-np.clip(packed_gate, -60.0, 60.0))),
                packed_up,
                dtype=np.float32,
            )
            source_outputs.append(
                (raw_gate / (1.0 + np.exp(-np.clip(raw_gate, -60.0, 60.0)))) * raw_up
            )
            separate_outputs.append(separate)
            paired_outputs.append(paired)
        raw = np.concatenate([row.reshape(-1) for row in source_outputs]).astype(np.float32, copy=False)
        separate = np.concatenate([row.reshape(-1) for row in separate_outputs]).astype(np.float32, copy=False)
        paired = np.concatenate([row.reshape(-1) for row in paired_outputs]).astype(np.float32, copy=False)
        raw_norm = max(float(np.linalg.norm(raw)), 1e-12)
        packed_norm = max(float(np.linalg.norm(separate)), 1e-12)
        return {
            "activation_control_count": len(controls) * 4,
            "activation_scales": [float(scale) for _, scale in controls],
            "source_to_direct_binary_swiglu": {
                "relative_l2": float(np.linalg.norm(separate - raw) / raw_norm),
                "cosine": float(np.dot(raw, separate) / (raw_norm * packed_norm)),
                "rmse": float(np.sqrt(np.mean(np.square(separate - raw)))),
                "max_abs": float(np.max(np.abs(separate - raw))),
            },
            "paired_vs_two_projection_direct_binary": {
                "max_abs": float(np.max(np.abs(paired - separate))),
                "relative_l2": float(np.linalg.norm(paired - separate) / max(float(np.linalg.norm(separate)), 1e-12)),
                "exact_float32_equivalent": bool(np.array_equal(paired, separate)),
            },
            "claim_boundary": "deterministic source-component control only; not prompt/template parity, Metal parity, generation, HCLI, capability, or TPS",
        }

    def _run_qwen30_gate_up_quality_experiment(
        self, observation: Mapping[str, Any], experiment: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, bool]:
        """Execute the cheapest current Qwen30 representation/fusion discriminator.

        It reads exactly two source tensors under the existing full-shard
        revalidation contract.  No pack, admitted baseline, runtime, HCLI
        adapter, or GPU command graph is changed by this lane.
        """

        if experiment.get("status") != "BLOCKED_CANONICAL_TEMPLATE_COHERENCE_KERNEL_INTEGRATION_DIAGNOSIS":
            return None, False
        source = observation.get("source_authority")
        evidence = observation.get("qwen30_canonical_template_evidence")
        if not isinstance(source, Mapping) or not isinstance(evidence, Mapping):
            return None, False
        identity = source.get("identity")
        revalidation = source.get("current_revalidation")
        exact = evidence.get("exact_native_runtime")
        gate_up_receipt = evidence.get("direct_packed_gate_up_pair")
        if not all(isinstance(row, Mapping) for row in (identity, revalidation, exact, gate_up_receipt)):
            return None, False
        source_identity = identity.get("seal_sha256")
        source_revalidation = revalidation.get("seal_sha256")
        exact_seal = exact.get("seal_sha256")
        component_seal = gate_up_receipt.get("seal_sha256")
        if not all(isinstance(value, str) for value in (source_identity, source_revalidation, exact_seal, component_seal)):
            return None, False
        gate_target = "model.layers.0.mlp.experts.0.gate_proj.weight"
        up_target = "model.layers.0.mlp.experts.0.up_proj.weight"
        key = _digest(
            {
                "source_identity": source_identity,
                "source_revalidation": source_revalidation,
                "exact_runtime": exact_seal,
                "gate_up_component": component_seal,
                "targets": [gate_target, up_target],
                "quality_revision": GATE_UP_QUALITY_REVISION,
                "representation": "binary_sign_scale_group128",
            }
        )
        quality_path = self.root / "capability-quality" / f"QWEN30_DIRECT_PACKED_GATE_UP_QUALITY_{key[:24]}.json"
        existing = _bounded_file_summary(quality_path, seal_expected=True)
        if existing.get("sealed"):
            return (
                existing.get("document") if isinstance(existing.get("document"), Mapping) else None,
                False,
            )
        failure: str | None = None
        source_binding: dict[str, Any] = {}
        measurement: dict[str, Any] = {}
        budget: dict[str, Any] = {}
        try:
            import numpy as np

            from lab.operators.ascension_dual_gravity_worker import DualGravityWorker, SPECS, _binary_codec
            from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

            worker = DualGravityWorker(SPECS["qwen30"])
            source_identity_document = _read_json(Path(str(identity.get("path", ""))))
            if source_identity_document is None:
                raise ScientificOptimizerError("sealed Qwen30 source identity is unavailable for the gate/up quality control")
            weight_map = load_weight_map(worker.spec.source_dir)
            gate_shard = weight_map.get(gate_target)
            up_shard = weight_map.get(up_target)
            if not isinstance(gate_shard, str) or not isinstance(up_shard, str):
                raise ScientificOptimizerError("source index does not bind the selected Qwen30 gate/up pair")
            gate_proof = worker._current_source_revalidation(source_identity_document, weight_map, target_shard=gate_shard)
            up_proof = gate_proof if up_shard == gate_shard else worker._current_source_revalidation(
                source_identity_document, weight_map, target_shard=up_shard
            )
            gate_values = np.ascontiguousarray(load_tensor(worker.spec.source_dir, weight_map, gate_target), dtype=np.float32)
            up_values = np.ascontiguousarray(load_tensor(worker.spec.source_dir, weight_map, up_target), dtype=np.float32)
            worker._assert_revalidated_target_unchanged(gate_proof)
            worker._assert_revalidated_target_unchanged(up_proof)
            if gate_values.ndim != 2 or up_values.ndim != 2 or gate_values.shape != up_values.shape:
                raise ScientificOptimizerError("selected source Qwen30 routed-expert gate/up tensors do not have matching 2D geometry")
            gate_codec = _binary_codec(gate_values, group_size=128)
            up_codec = _binary_codec(up_values, group_size=128)
            measurement = self._paired_swiglu_control_quality(
                gate_values, up_values, gate_codec.reconstruction, up_codec.reconstruction
            )
            pair_bytes = len(gate_codec.payload) + len(up_codec.payload)
            pair_elements = gate_values.size + up_values.size
            budget = {
                "pair_physical_bytes": pair_bytes,
                "pair_component_physical_bpw": pair_bytes * 8.0 / pair_elements,
                "within_1_5_component_bpw": pair_bytes * 8.0 / pair_elements <= 1.5,
                "rule": "component BPW only; it does not estimate or change a complete artifact BPW",
            }
            source_binding = {
                "source_content_identity_sha256": gate_proof.get("source_content_identity_sha256"),
                "revalidation_receipt_path": gate_proof.get("receipt_path"),
                "revalidation_receipt_seal_sha256": gate_proof.get("receipt_seal_sha256"),
                "tensors": [
                    {
                        "tensor_name": gate_target,
                        "source_shard": gate_proof.get("target_shard"),
                        "source_shard_sha256": gate_proof.get("target_shard_sha256"),
                        "tensor_shape": [int(value) for value in gate_values.shape],
                        "source_value_sha256": _digest(gate_values.astype("<f4", copy=False).tobytes()),
                    },
                    {
                        "tensor_name": up_target,
                        "source_shard": up_proof.get("target_shard"),
                        "source_shard_sha256": up_proof.get("target_shard_sha256"),
                        "tensor_shape": [int(value) for value in up_values.shape],
                        "source_value_sha256": _digest(up_values.astype("<f4", copy=False).tobytes()),
                    },
                ],
            }
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        paired = measurement.get("paired_vs_two_projection_direct_binary") if isinstance(measurement.get("paired_vs_two_projection_direct_binary"), Mapping) else {}
        status = (
            "PASS_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_QUALITY_DIAGNOSTIC_NOT_MODEL_QUALITY"
            if failure is None and bool(budget.get("within_1_5_component_bpw")) and paired.get("exact_float32_equivalent") is True
            else "FAIL_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_QUALITY_DIAGNOSTIC"
        )
        document = seal(
            {
                "schema": GATE_UP_QUALITY_SCHEMA,
                "record_id": f"qwen30-direct-packed-gate-up-quality:{key}",
                "recorded_at": _utc_now(),
                "status": status,
                "receipt_path": str(quality_path),
                "quality_revision": GATE_UP_QUALITY_REVISION,
                "runtime_binding": {
                    "exact_runtime_receipt": _receipt_reference(exact),
                    "direct_packed_component_receipt": _receipt_reference(gate_up_receipt),
                    "integration_state": "UNINTEGRATED_COMPONENT_ONLY",
                },
                "source_binding": source_binding,
                "budget": budget,
                "measurement": measurement,
                "diagnosis": {
                    "tested": [
                        "gate_up_fusion_command_topology_not_yet_integrated",
                        "sensitive_organ_representation_error_under_complete_bpw_budget",
                    ],
                    "not_tested": "template_simdgroup_reduction_or_indexing_divergence_requires_a_repaired_exact_template_candidate_regression",
                    "interpretation_rule": (
                        "paired/direct-binary algebraic equality excludes only a source-component algebra difference; it does not validate Metal integration. "
                        "The source-to-direct-binary error remains a representation diagnostic, not a prompt-quality score."
                    ),
                },
                "failure": failure,
                "pass_condition": "two source-bound direct-binary gate/up payloads fit the component budget and paired algebra matches the two-projection direct-binary control",
                "fail_condition": "source/revalidation, pair geometry, budget, or deterministic paired control fails",
                "reopen_condition": "source revalidation, direct-binary representation/layout, component kernel implementation, or runtime integration materially changes",
                "claim_boundary": "source-bound component diagnostic only; no repack or baseline swap, and no full-model coherence, HCLI, capability, TPS, TG, or tournament claim",
            }
        )
        _atomic_json(quality_path, document)
        return document, True

    def _append_qwen30_gate_up_cross_teach(self, quality: Mapping[str, Any]) -> None:
        if quality.get("status") != "PASS_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_QUALITY_DIAGNOSTIC_NOT_MODEL_QUALITY":
            return
        seal_sha = quality.get("seal_sha256")
        if not isinstance(seal_sha, str):
            return
        measurement = quality.get("measurement") if isinstance(quality.get("measurement"), Mapping) else {}
        row = seal(
            {
                "schema": "hawking.ascension.qwen_gate_up_component_cross_teach.v1",
                "record_id": f"cross-teach:qwen30-direct-packed-gate-up-quality:{seal_sha}",
                "recorded_at": _utc_now(),
                "status": "PASS_SOURCE_BOUND_QWEN30_GATE_UP_QUALITY_TRANSFER_PRIOR_NOT_QWEN80_QUALIFICATION",
                "model": "qwen30",
                "model_family": "qwen3_moe",
                "target_model": "qwen80",
                "target_model_family": "qwen3_next_hybrid",
                "mechanism": "direct_packed_gated_mlp_pair_quality",
                "mechanism_key": f"optimizer:qwen30-gate-up-quality:{seal_sha}",
                "quality_receipt_path": quality.get("receipt_path"),
                "quality_receipt_seal_sha256": seal_sha,
                "source_to_direct_binary_swiglu": measurement.get("source_to_direct_binary_swiglu"),
                "transfer_rule": (
                    "Qwen80 may use this as an initialization prior only for direct-packed routed/shared gated MLP pairs. "
                    "It must validate its own geometry/parity and must not apply this result to DeltaNet/recurrent state or claim capability/TPS."
                ),
                "reopen_conditions": quality.get("reopen_condition"),
                "claim_boundary": "cross-family component research prior only; not a Qwen80 kernel integration, native runtime, model-quality, HCLI, or TPS receipt",
            }
        )
        _append_jsonl_once(self.shared_kernel_path, row, record_id=str(row["record_id"]))

    def _write_qwen30_gate_up_repack_proposal(
        self, quality: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Emit a guarded candidate handoff without changing Qwen30's body.

        The measured direct-binary error makes gate/up a justified *research
        target*, not a reason to mutate the admitted artifact.  This proposal
        is intentionally less specific than an implementation: no residual
        budget or layout is treated as selected until it can satisfy a fresh
        complete-artifact ledger.
        """

        if quality.get("status") != "PASS_SOURCE_BOUND_DIRECT_PACKED_GATE_UP_QUALITY_DIAGNOSTIC_NOT_MODEL_QUALITY":
            return None
        quality_seal = quality.get("seal_sha256")
        if not isinstance(quality_seal, str):
            return None
        path = self.root / "repack-proposals" / f"QWEN30_GATE_UP_REPRESENTATION_REPACK_PROPOSAL_{quality_seal[:24]}.json"
        existing = _bounded_file_summary(path, seal_expected=True)
        if existing.get("sealed") and isinstance(existing.get("document"), Mapping):
            return dict(existing["document"])
        complete = observation.get("current_complete_champion") if isinstance(observation.get("current_complete_champion"), Mapping) else {}
        manifest = complete.get("manifest") if isinstance(complete.get("manifest"), Mapping) else {}
        admission = complete.get("admission") if isinstance(complete.get("admission"), Mapping) else {}
        measurement = quality.get("measurement") if isinstance(quality.get("measurement"), Mapping) else {}
        proposal = seal(
            {
                "schema": "hawking.ascension.qwen30_gate_up_representation_repack_proposal.v1",
                "record_id": f"qwen30-gate-up-repack-proposal:{quality_seal}",
                "recorded_at": _utc_now(),
                "status": "PROPOSED_NOT_APPLIED_COMPLETE_ACCOUNTING_AND_CAPABILITY_RETEST_REQUIRED",
                "proposal_path": str(path),
                "quality_receipt_path": quality.get("receipt_path"),
                "quality_receipt_seal_sha256": quality_seal,
                "baseline_control": {
                    "manifest_path": manifest.get("path"),
                    "manifest_seal_sha256": manifest.get("seal_sha256"),
                    "complete_physical_bpw": manifest.get("complete_physical_bpw"),
                    "admission_path": admission.get("path"),
                    "admission_seal_sha256": admission.get("seal_sha256"),
                    "preserve_as_rollback_control": True,
                    "replacement_forbidden_until_all_acceptance_gates_pass": True,
                },
                "evidence_trigger": {
                    "source_to_direct_binary_swiglu": measurement.get("source_to_direct_binary_swiglu"),
                    "paired_direct_binary_algebra": measurement.get("paired_vs_two_projection_direct_binary"),
                    "interpretation": "the source-bound diagnostic prioritizes this organ for a controlled representation branch; it does not select a residual ratio or prove prompt quality",
                },
                "proposed_candidate_branch": {
                    "family": "sensitive_gate_up_representation_allocation_under_global_bpw_budget",
                    "initial_organs": [
                        "model.layers.0.mlp.experts.0.gate_proj.weight",
                        "model.layers.0.mlp.experts.0.up_proj.weight",
                    ],
                    "allowed_next_work": [
                        "source-bound residual/codebook/grouping controls on gate/up pairs",
                        "whole-artifact byte accounting before any admission request",
                        "fresh native parity only after an admitted replacement artifact exists",
                    ],
                    "forbidden_now": ["repack", "baseline replacement", "runtime integration", "model-quality claim", "TPS claim"],
                },
                "hard_full_artifact_accounting_gate": {
                    "required_complete_physical_bpw_max": 1.5,
                    "state": "UNMEASURED_FULL_ARTIFACT_REPACK_REQUIRED",
                    "requirements": [
                        "repack every affected tensor into a new complete artifact; do not extrapolate a component BPW",
                        "seal an all-tensor physical-byte/BPW ledger at <=1.5 COMPLETE BPW, including controls, residuals, codebooks, scales, and layouts",
                        "bind the immutable source identity and a current full-shard revalidation receipt",
                        "pass full native artifact admission for every tensor/control payload before any runtime substitution",
                    ],
                },
                "post_runtime_capability_retest_gate": {
                    "state": "REQUIRED_AFTER_ANY_MATERIAL_REPRESENTATION_OR_RUNTIME_CHANGE",
                    "requirements": [
                        "all-layer native exact-token parity with no fallback and raw BF16 excluded from runtime",
                        "multiple prompt-dependent autoregressive continuations with coherent/structured/code scoring",
                        "fresh HCLI chat, structured-output, session, and restart evidence",
                        "fresh Context/KV, Agent OS, storage/rollback, and capability receipts on the replacement artifact",
                        "fresh >=98% complete-token profile and official clean BASE_TRUE_TPS/TG qualification; no prior timing transfers",
                    ],
                },
                "automatic_action": {
                    "may_enqueue_for_repack_owner": True,
                    "may_repack_or_apply_automatically": False,
                    "may_replace_admitted_baseline": False,
                    "may_claim_model_quality_hcli_tps_or_tg": False,
                },
                "claim_boundary": "guarded Qwen30 candidate/repack proposal only; it does not change the admitted baseline or prove model quality",
            }
        )
        _atomic_json(path, proposal)
        scheduler = seal(
            {
                "schema": "hawking.ascension.qwen_gate_up_repack_scheduler_handoff.v1",
                "record_id": f"scheduler-gate-up-repack:{quality_seal}",
                "recorded_at": _utc_now(),
                "model": "qwen30",
                "status": "NEXT_EXPLICIT_GATE_UP_REPACK_PROPOSAL_WAITING_FOR_FULL_ACCOUNTING",
                "proposal_path": str(path),
                "proposal_seal_sha256": proposal.get("seal_sha256"),
                "required_before_execution": proposal["hard_full_artifact_accounting_gate"]["requirements"],
                "claim_boundary": "handoff only; does not schedule or authorize an unaccounted representation mutation",
            }
        )
        _append_jsonl_once(self.shared_scheduler_path, scheduler, record_id=str(scheduler["record_id"]))
        return proposal

    def _run_qwen30_sensitive_quality_experiment(
        self, observation: Mapping[str, Any], experiment: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, bool]:
        """Differentiate compact router representation mechanisms with a real source tensor.

        The raw BF16 tensor is opened only as the source/teacher authority for a
        bounded component control.  The runtime is never invoked and no MPS
        full-model path is used.
        """

        if experiment.get("status") != "BLOCKED_CAPABILITY_COHERENCE_AFTER_DIRECT_PACKED_GENERATION":
            return None, False
        source = observation.get("source_authority")
        if not isinstance(source, Mapping):
            return None, False
        identity = source.get("identity")
        revalidation = source.get("current_revalidation")
        if not isinstance(identity, Mapping) or not isinstance(revalidation, Mapping):
            return None, False
        prompt = observation.get("direct_packed_generation")
        if not isinstance(prompt, Mapping):
            return None, False
        prompt_digest = prompt.get("receipt", {}).get("document_sha256") if isinstance(prompt.get("receipt"), Mapping) else None
        source_identity = identity.get("seal_sha256")
        source_revalidation = revalidation.get("seal_sha256")
        if not all(isinstance(value, str) for value in (prompt_digest, source_identity, source_revalidation)):
            return None, False
        key = _digest(
            {
                "prompt_receipt_sha256": prompt_digest,
                "source_identity": source_identity,
                "source_revalidation": source_revalidation,
                "target": "model.layers.0.mlp.gate.weight",
                "quality_revision": SENSITIVE_ROUTER_QUALITY_REVISION,
                "variants": ["binary_sign_scale_group128", "binary_sign_scale_group256", "binary_sign_scale_sparse_fp16_residual_0.0025"],
            }
        )
        quality_dir = self.root / "capability-quality"
        quality_path = quality_dir / f"QWEN30_SENSITIVE_ROUTER_REPRESENTATION_QUALITY_{key[:24]}.json"
        existing = _bounded_file_summary(quality_path, seal_expected=True)
        if existing.get("sealed"):
            return (
                existing.get("document") if isinstance(existing.get("document"), Mapping) else None,
                False,
            )
        target = "model.layers.0.mlp.gate.weight"
        failure: str | None = None
        source_proof: dict[str, Any] = {}
        variants: list[dict[str, Any]] = []
        try:
            import numpy as np

            from lab.operators.ascension_dual_gravity_worker import (
                DualGravityWorker,
                SPECS,
                _binary_codec,
                _quality,
                _residual_codec,
            )
            from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

            worker = DualGravityWorker(SPECS["qwen30"])
            source_identity_document = _read_json(Path(str(identity.get("path", ""))))
            if source_identity_document is None:
                raise ScientificOptimizerError("sealed Qwen30 source identity is unavailable for the sensitive quality control")
            weight_map = load_weight_map(worker.spec.source_dir)
            shard = weight_map.get(target)
            if not isinstance(shard, str):
                raise ScientificOptimizerError("sensitive Qwen30 router tensor is not in the verified source index")
            proof = worker._current_source_revalidation(source_identity_document, weight_map, target_shard=shard)
            values = np.ascontiguousarray(load_tensor(worker.spec.source_dir, weight_map, target), dtype=np.float32)
            worker._assert_revalidated_target_unchanged(proof)
            if values.ndim != 2 or values.shape[0] < 8:
                raise ScientificOptimizerError("sensitive router tensor has unexpected geometry")
            source_proof = {
                "source_content_identity_sha256": proof.get("source_content_identity_sha256"),
                "revalidation_receipt_path": proof.get("receipt_path"),
                "revalidation_receipt_seal_sha256": proof.get("receipt_seal_sha256"),
                "source_shard": proof.get("target_shard"),
                "source_shard_sha256": proof.get("target_shard_sha256"),
                "tensor_name": target,
                "tensor_shape": [int(value) for value in values.shape],
                "source_value_sha256": _digest(values.astype("<f4", copy=False).tobytes()),
            }
            codec_variants = (
                ("binary_sign_scale_group128", _binary_codec(values, group_size=128)),
                ("binary_sign_scale_group256", _binary_codec(values, group_size=256)),
                ("binary_sign_scale_sparse_fp16_residual_0.0025", _residual_codec(values, group_size=128, outlier_ratio=0.0025)),
            )
            for name, codec in codec_variants:
                bpw = len(codec.payload) * 8.0 / values.size
                compatible = bpw <= 1.5
                quality = _quality(values, codec.reconstruction)
                routing = self._router_control_quality(values, codec.reconstruction) if compatible else None
                variants.append(
                    {
                        "name": name,
                        "physical_bytes": len(codec.payload),
                        "component_physical_bpw": bpw,
                        "within_1_5_bpw_component_budget": compatible,
                        "weight_quality": quality,
                        "router_control": routing,
                    }
                )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        eligible = [row for row in variants if row.get("within_1_5_bpw_component_budget") and isinstance(row.get("router_control"), Mapping)]
        winner: dict[str, Any] | None = None
        if eligible:
            winner = max(
                eligible,
                key=lambda row: (
                    float(row["router_control"].get("top1_agreement", -1.0)),
                    float(row["router_control"].get("mean_top_k_overlap", -1.0)),
                    -float(row["router_control"].get("mean_router_logit_relative_l2", float("inf"))),
                ),
            )
        status = (
            "PASS_SOURCE_BOUND_SENSITIVE_ROUTER_REPRESENTATION_QUALITY_DIAGNOSTIC_NOT_CAPABILITY"
            if failure is None and winner is not None
            else "FAIL_SOURCE_BOUND_SENSITIVE_ROUTER_REPRESENTATION_QUALITY_DIAGNOSTIC"
        )
        document = seal(
            {
                "schema": "hawking.ascension.qwen30_sensitive_router_representation_quality.v1",
                "record_id": f"qwen30-sensitive-router-quality:{key}",
                "recorded_at": _utc_now(),
                "status": status,
                "receipt_path": str(quality_path),
                "quality_revision": SENSITIVE_ROUTER_QUALITY_REVISION,
                "prompt_trace_binding": {
                    "path": prompt.get("receipt", {}).get("path") if isinstance(prompt.get("receipt"), Mapping) else None,
                    "document_sha256": prompt_digest,
                    "observed_completion_text_unscored": prompt.get("completion_text_unscored"),
                    "all_48_layers_executed": prompt.get("all_layers_executed"),
                    "metal_only": prompt.get("metal_only"),
                    "raw_bf16_loader_not_opened": prompt.get("raw_bf16_loader_not_opened"),
                },
                "source_binding": source_proof,
                "budget": {
                    "complete_artifact_bpw": observation.get("current_complete_champion", {}).get("manifest", {}).get("complete_physical_bpw")
                    if isinstance(observation.get("current_complete_champion"), Mapping)
                    else None,
                    "candidate_component_bpw_limit": 1.5,
                    "rule": "only variants at or below 1.5 component physical BPW are eligible; this does not estimate a new full-artifact BPW",
                },
                "variants": variants,
                "winner": winner,
                "diagnosis": {
                    "distinguished_mechanisms": [
                        "binary_scale_projection_error_under_complete_bpw_budget",
                        "router_top_k_instability",
                        "misallocated_sparse_residual_budget",
                    ],
                    "mechanism_not_tested_here": "final_logit_path_sensitivity",
                    "next_discriminating_gate": "sealed multi-prompt coherence/capability evidence on a materially rebuilt exact artifact",
                },
                "failure": failure,
                "pass_condition": "a source-bound <=1.5 component-BPW variant completes deterministic sensitive-router controls",
                "fail_condition": "source binding, bounded codec, budget, or deterministic route control is unavailable",
                "reopen_condition": "new exact source revalidation, a materially changed complete representation, or new router kernel/error evidence",
                "claim_boundary": "component source-router quality diagnostic only; not a full-model quality score, prompt coherence pass, HCLI, TPS, TG, or manager capability receipt",
            }
        )
        _atomic_json(quality_path, document)
        return document, True

    def _append_qwen30_quality_cross_teach(self, quality: Mapping[str, Any]) -> None:
        if quality.get("status") != "PASS_SOURCE_BOUND_SENSITIVE_ROUTER_REPRESENTATION_QUALITY_DIAGNOSTIC_NOT_CAPABILITY":
            return
        winner = quality.get("winner") if isinstance(quality.get("winner"), Mapping) else {}
        name = winner.get("name")
        representation = (
            "binary_outlier_residual" if name == "binary_sign_scale_sparse_fp16_residual_0.0025" else "binary_sign_scale128"
        )
        row = seal(
            {
                "schema": "hawking.ascension.qwen_sensitive_router_cross_teach.v1",
                "record_id": f"cross-teach:qwen30-sensitive-router:{quality.get('seal_sha256')}",
                "recorded_at": _utc_now(),
                "status": "PASS_SOURCE_BOUND_QWEN30_ROUTER_QUALITY_TRANSFER_PRIOR_NOT_QWEN80_QUALIFICATION",
                "model": "qwen30",
                "model_family": "qwen3_moe",
                "target_family": "qwen3_next_hybrid",
                "representation": representation,
                "mechanism": "sensitive_router_representation_quality",
                "mechanism_key": f"qwen30-router-quality:{quality.get('seal_sha256')}",
                "quality_receipt_path": quality.get("receipt_path"),
                "quality_receipt_seal_sha256": quality.get("seal_sha256"),
                "winner": winner,
                "transfer_rule": "Qwen80 must directly validate its top-10 router and keep DeltaNet assumptions separate; this is only an initial representation prior",
                "reopen_conditions": quality.get("reopen_condition"),
                "claim_boundary": "cross-family research prior; not a Qwen80 component pass, full-model capability, or TPS result",
            }
        )
        _append_jsonl_once(self.shared_representation_path, row, record_id=str(row["record_id"]))

    def _write_qwen30_repack_proposal(
        self, quality: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Publish a candidate handoff without mutating the admitted baseline.

        A sensitive-organ component win is useful only as a *repack proposal*.
        It must never be silently substituted for a complete artifact ledger,
        native admission, or capability result.
        """

        if quality.get("status") != "PASS_SOURCE_BOUND_SENSITIVE_ROUTER_REPRESENTATION_QUALITY_DIAGNOSTIC_NOT_CAPABILITY":
            return None
        winner = quality.get("winner") if isinstance(quality.get("winner"), Mapping) else {}
        if winner.get("name") != "binary_sign_scale_sparse_fp16_residual_0.0025":
            return None
        quality_seal = quality.get("seal_sha256")
        if not isinstance(quality_seal, str):
            return None
        proposal_path = self.root / "repack-proposals" / f"QWEN30_ROUTER_RESIDUAL_REPACK_PROPOSAL_{quality_seal[:24]}.json"
        existing = _bounded_file_summary(proposal_path, seal_expected=True)
        if existing.get("sealed") and isinstance(existing.get("document"), Mapping):
            return dict(existing["document"])
        complete = observation.get("current_complete_champion") if isinstance(observation.get("current_complete_champion"), Mapping) else {}
        manifest = complete.get("manifest") if isinstance(complete.get("manifest"), Mapping) else {}
        admission = complete.get("admission") if isinstance(complete.get("admission"), Mapping) else {}
        proposal = seal(
            {
                "schema": "hawking.ascension.qwen30_sensitive_router_repack_proposal.v1",
                "record_id": f"qwen30-repack-proposal:{quality_seal}",
                "recorded_at": _utc_now(),
                "status": "PROPOSED_NOT_APPLIED_COMPLETE_ACCOUNTING_AND_CAPABILITY_RETEST_REQUIRED",
                "proposal_path": str(proposal_path),
                "quality_receipt_path": quality.get("receipt_path"),
                "quality_receipt_seal_sha256": quality_seal,
                "baseline_control": {
                    "manifest_path": manifest.get("path"),
                    "manifest_seal_sha256": manifest.get("seal_sha256"),
                    "complete_physical_bpw": manifest.get("complete_physical_bpw"),
                    "admission_path": admission.get("path"),
                    "admission_seal_sha256": admission.get("seal_sha256"),
                    "preserve_as_rollback_control": True,
                    "replacement_forbidden_until_all_acceptance_gates_pass": True,
                },
                "proposed_mutation": {
                    "representation": "binary_sign_scale_sparse_fp16_residual_0.0025",
                    "initial_sensitive_organ": "model.layers.0.mlp.gate.weight",
                    "scope_rule": "start with source-bound router-sensitive organs only; expand only after direct per-organ control evidence",
                    "component_evidence": winner,
                    "not_applied_to_complete_artifact": True,
                },
                "hard_full_artifact_accounting_gate": {
                    "required_complete_physical_bpw_max": 1.5,
                    "state": "UNMEASURED_FULL_ARTIFACT_REPACK_REQUIRED",
                    "requirements": [
                        "repack every affected tensor into a new complete artifact; do not extrapolate component BPW",
                        "seal a complete all-tensor physical-byte and BPW ledger at <=1.5 COMPLETE BPW",
                        "bind the same immutable source identity and a current full-shard revalidation receipt",
                        "verify every payload, tensor catalog, control tensor, hash, and layout through native artifact admission",
                    ],
                },
                "post_runtime_capability_retest_gate": {
                    "state": "REQUIRED_AFTER_ANY_MATERIAL_REPRESENTATION_OR_RUNTIME_CHANGE",
                    "requirements": [
                        "native all-layer exact token execution with no fallback and raw BF16 excluded from runtime",
                        "multiple prompt-dependent autoregressive continuations, including structured/code samples, with coherence scoring",
                        "actual logits/sampler/next-token feedback verification",
                        "fresh HCLI chat, structured-output, session, and restart receipts",
                        "capability, Context/KV, Agent OS, and storage/rollback evaluations on the new exact artifact",
                        "new clean complete-token profiler and official 100 BASE_TRUE_TPS/TG receipts; prior timing cannot transfer",
                    ],
                },
                "automatic_action": {
                    "may_enqueue_for_repack_owner": True,
                    "may_replace_admitted_baseline": False,
                    "may_claim_model_quality_pass": False,
                    "may_claim_capability_hcli_tps_or_tg": False,
                },
                "claim_boundary": "a physical repack candidate proposal only; it neither changes the admitted Qwen30 baseline nor proves model quality",
            }
        )
        _atomic_json(proposal_path, proposal)
        scheduler = seal(
            {
                "schema": "hawking.ascension.qwen_sensitive_repack_scheduler_handoff.v1",
                "record_id": f"scheduler-repack:{quality_seal}",
                "recorded_at": _utc_now(),
                "model": "qwen30",
                "status": "NEXT_EXPLICIT_REPACK_PROPOSAL_WAITING_FOR_FULL_ACCOUNTING",
                "proposal_path": str(proposal_path),
                "proposal_seal_sha256": proposal.get("seal_sha256"),
                "required_before_execution": proposal["hard_full_artifact_accounting_gate"]["requirements"],
                "claim_boundary": "handoff only; does not schedule an unaccounted representation change",
            }
        )
        _append_jsonl_once(self.shared_scheduler_path, scheduler, record_id=str(scheduler["record_id"]))
        return proposal

    def _update_frontier(self, profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        previous = _bounded_file_summary(self.frontier_path, seal_expected=True)
        if not profiles and previous.get("sealed") and isinstance(previous.get("document"), Mapping):
            return dict(previous["document"])
        existing: dict[str, Any] = {}
        if previous.get("sealed") and isinstance(previous.get("document"), Mapping):
            prior_rows = previous["document"].get("component_io_frontier")
            if isinstance(prior_rows, Mapping):
                existing = {str(key): dict(value) for key, value in prior_rows.items() if isinstance(value, Mapping)}
        for profile in profiles:
            if profile.get("status") != "PASS_SOURCE_BOUND_PACKED_COMPONENT_IO_PROFILE_NOT_MODEL_TPS":
                continue
            model = profile.get("model")
            metrics = profile.get("metrics") if isinstance(profile.get("metrics"), Mapping) else {}
            value = metrics.get("artifact_read_mib_per_second")
            if not isinstance(model, str) or not isinstance(value, (int, float)):
                continue
            current = existing.get(model)
            if not isinstance(current, Mapping) or float(value) > float(current.get("artifact_read_mib_per_second", -1.0)):
                existing[model] = {
                    "task_id": profile.get("task_id"),
                    "profile_seal_sha256": profile.get("seal_sha256"),
                    "artifact_sha256": metrics.get("artifact_sha256"),
                    "artifact_read_mib_per_second": float(value),
                    "updated_at": _utc_now(),
                    "claim_boundary": "bounded component artifact reader I/O only; not native decoder or model TPS",
                }
        document = seal(
            {
                "schema": FRONTIER_SCHEMA,
                "status": "COMPONENT_ONLY_FRONTIER_NOT_MODEL_SELECTION",
                "recorded_at": _utc_now(),
                "component_io_frontier": existing,
                "claim_boundary": "this frontier cannot select a manager or qualify any tournament gate",
            }
        )
        _atomic_json(self.frontier_path, document)
        return document

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "heartbeat": 0,
            "material_progress_count": 0,
            "last_material_progress_at": None,
            "previous_observations": {},
        }

    def _state(self) -> dict[str, Any]:
        existing = _read_json(self.state_path)
        if existing is None:
            return self._default_state()
        if existing.get("schema") != SCHEMA:
            raise ScientificOptimizerError("optimizer state belongs to an incompatible schema")
        return dict(existing)

    def run_cycle(self) -> dict[str, Any]:
        """Observe both workers, execute safe profiles once, and publish a handoff."""

        with _locked(self.lock_path):
            state = self._state()
            shared = self._shared_knowledge()
            observations = {spec.key: self._observe_model(spec, shared) for spec in MODEL_SPECS}
            previous = state.get("previous_observations") if isinstance(state.get("previous_observations"), Mapping) else {}
            liveness = {
                spec.key: self._material_liveness(observations[spec.key], previous.get(spec.key) if isinstance(previous.get(spec.key), Mapping) else None)
                for spec in MODEL_SPECS
            }
            emitted: list[dict[str, Any]] = []
            canonical_template_evidence: list[dict[str, Any]] = []
            newly_ingested_canonical_template_evidence: list[dict[str, Any]] = []
            profiles: list[dict[str, Any]] = []
            newly_run_profiles: list[dict[str, Any]] = []
            quality_results: list[dict[str, Any]] = []
            newly_run_quality: list[dict[str, Any]] = []
            gate_up_quality_results: list[dict[str, Any]] = []
            newly_run_gate_up_quality: list[dict[str, Any]] = []
            repack_proposals: list[dict[str, Any]] = []
            active_experiments: dict[str, Mapping[str, Any]] = {}
            canonical_evidence, canonical_evidence_added = self._ingest_qwen30_canonical_template_evidence(
                observations["qwen30"]
            )
            if canonical_evidence is not None:
                canonical_template_evidence.append(canonical_evidence)
                if canonical_evidence_added:
                    newly_ingested_canonical_template_evidence.append(canonical_evidence)
            for spec in MODEL_SPECS:
                peer_key = "qwen80" if spec.key == "qwen30" else "qwen30"
                experiment = self._build_experiment(spec, observations[spec.key], observations[peer_key], liveness[spec.key])
                path = self.experiments_dir / f"{experiment['task_id']}.json"
                existing = _bounded_file_summary(path, seal_expected=True)
                if existing.get("sealed"):
                    experiment = existing["document"] if isinstance(existing.get("document"), Mapping) else experiment
                else:
                    _atomic_json(path, experiment)
                    emitted.append(experiment)
                active_experiments[spec.key] = experiment
                profile, ran_now = self._run_component_profile(spec, experiment)
                if profile is not None:
                    profiles.append(profile)
                    if ran_now:
                        newly_run_profiles.append(profile)
                        self._append_profile_knowledge(profile)
                if spec.key == "qwen30":
                    gate_up_quality, gate_up_quality_ran_now = self._run_qwen30_gate_up_quality_experiment(
                        observations[spec.key], experiment
                    )
                    if gate_up_quality is not None:
                        gate_up_quality_results.append(gate_up_quality)
                        if gate_up_quality_ran_now:
                            newly_run_gate_up_quality.append(gate_up_quality)
                            self._append_qwen30_gate_up_cross_teach(gate_up_quality)
                        gate_up_proposal = self._write_qwen30_gate_up_repack_proposal(
                            gate_up_quality, observations[spec.key]
                        )
                        if gate_up_proposal is not None:
                            repack_proposals.append(gate_up_proposal)
                    quality, quality_ran_now = self._run_qwen30_sensitive_quality_experiment(
                        observations[spec.key], experiment
                    )
                    if quality is not None:
                        quality_results.append(quality)
                        if quality_ran_now:
                            newly_run_quality.append(quality)
                            self._append_qwen30_quality_cross_teach(quality)
                        proposal = self._write_qwen30_repack_proposal(quality, observations[spec.key])
                        if proposal is not None:
                            repack_proposals.append(proposal)
            frontier = self._update_frontier(newly_run_profiles)
            prior_material = {
                spec.key: {
                    "heartbeat": (
                        observations[spec.key].get("worker", {}).get("document", {}).get("heartbeat")
                        if isinstance(observations[spec.key].get("worker"), Mapping)
                        and isinstance(observations[spec.key].get("worker", {}).get("document"), Mapping)
                        else None
                    ),
                    "material_marker": observations[spec.key].get("material_marker"),
                    "activity_marker": observations[spec.key].get("activity_marker"),
                }
                for spec in MODEL_SPECS
            }
            material_events = len(emitted) + sum(
                1
                for profile in newly_run_profiles
                if profile.get("status") in {
                    "PASS_SOURCE_BOUND_PACKED_COMPONENT_IO_PROFILE_NOT_MODEL_TPS",
                    "FAIL_SOURCE_BOUND_PACKED_COMPONENT_IO_PROFILE",
                }
            ) + len(newly_ingested_canonical_template_evidence) + len(newly_run_quality) + len(newly_run_gate_up_quality) + sum(
                1 for row in liveness.values() if row.get("state") == "LIVE_MATERIAL_PROGRESS"
            )
            state["heartbeat"] = int(state.get("heartbeat", 0)) + 1
            state["updated_at"] = _utc_now()
            state["previous_observations"] = prior_material
            if material_events:
                state["material_progress_count"] = int(state.get("material_progress_count", 0)) + material_events
                state["last_material_progress_at"] = _utc_now()
            _atomic_json(self.state_path, state)
            status = {
                "schema": SCHEMA,
                "status": "REAL_EVIDENCE_DRIVEN_OPTIMIZATION_ADVANCING" if material_events else "AWAITING_MATERIAL_FRONTIER_CHANGE",
                "recorded_at": _utc_now(),
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "heartbeat": state["heartbeat"],
                "last_material_progress_at": state.get("last_material_progress_at"),
                "material_progress_count": state.get("material_progress_count"),
                "worker_liveness": liveness,
                "active_experiments": {
                    spec.key: {
                        "path": str(self.experiments_dir / f"{active_experiments[spec.key]['task_id']}.json"),
                        "status": active_experiments[spec.key].get("status"),
                    }
                    for spec in MODEL_SPECS
                },
                "new_experiment_count": len(emitted),
                "canonical_template_evidence": [
                    {
                        "path": evidence.get("evidence_path"),
                        "status": evidence.get("status"),
                        "seal_sha256": evidence.get("seal_sha256"),
                        "claim_boundary": evidence.get("claim_boundary"),
                    }
                    for evidence in canonical_template_evidence
                ],
                "profile_results": [
                    {
                        "task_id": profile.get("task_id"),
                        "model": profile.get("model"),
                        "status": profile.get("status"),
                        "seal_sha256": profile.get("seal_sha256"),
                    }
                    for profile in profiles
                ],
                "capability_quality_diagnostics": [
                    {
                        "record_id": quality.get("record_id"),
                        "status": quality.get("status"),
                        "seal_sha256": quality.get("seal_sha256"),
                        "winner": quality.get("winner", {}).get("name") if isinstance(quality.get("winner"), Mapping) else None,
                        "claim_boundary": quality.get("claim_boundary"),
                    }
                    for quality in quality_results
                ],
                "gate_up_quality_diagnostics": [
                    {
                        "record_id": quality.get("record_id"),
                        "status": quality.get("status"),
                        "seal_sha256": quality.get("seal_sha256"),
                        "source_to_direct_binary_swiglu": quality.get("measurement", {}).get("source_to_direct_binary_swiglu")
                        if isinstance(quality.get("measurement"), Mapping)
                        else None,
                        "claim_boundary": quality.get("claim_boundary"),
                    }
                    for quality in gate_up_quality_results
                ],
                "next_explicit_repack_proposals": [
                    {
                        "path": proposal.get("proposal_path"),
                        "status": proposal.get("status"),
                        "seal_sha256": proposal.get("seal_sha256"),
                        "baseline_replacement_forbidden": proposal.get("baseline_control", {}).get("replacement_forbidden_until_all_acceptance_gates_pass")
                        if isinstance(proposal.get("baseline_control"), Mapping)
                        else None,
                    }
                    for proposal in repack_proposals
                ],
                "frontier": {"path": str(self.frontier_path), "seal_sha256": frontier.get("seal_sha256")},
                "shared_knowledge_inputs": {
                    "representation_genome": str(self.shared_representation_path),
                    "kernel_genome": str(self.shared_kernel_path),
                    "scheduler_genome": str(self.shared_scheduler_path),
                    "negative_science": str(self.shared_negative_path),
                },
                "claim_boundary": {
                    "not_a_replacement_controller": True,
                    "heartbeat_only_is_not_liveness": True,
                    "no_native_runtime_means_no_tps_hcli_or_tg_claim": True,
                    "no_component_profile_is_model_tps": True,
                },
            }
            _atomic_json(self.status_path, status)
            return status

    def watch(self, *, idle_seconds: float) -> int:
        if idle_seconds <= 0:
            raise ScientificOptimizerError("idle seconds must be positive")

        def _stop(_: int, __: Any) -> None:
            self._stopping = True

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        while not self._stopping:
            try:
                self.run_cycle()
            except Exception as exc:
                _atomic_json(
                    self.status_path,
                    {
                        "schema": SCHEMA,
                        "status": "RECOVERABLE_OPTIMIZER_CYCLE_FAILURE",
                        "recorded_at": _utc_now(),
                        "pid": os.getpid(),
                        "ppid": os.getppid(),
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "claim_boundary": "a failed optimizer cycle makes no claim about campaign worker liveness",
                    },
                )
            deadline = time.monotonic() + idle_seconds
            while not self._stopping and time.monotonic() < deadline:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-root", type=Path, default=DEFAULT_PHYSICAL_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("once", help="run one deterministic evidence/experiment cycle")
    watch = commands.add_parser("watch", help="run the detached durable optimizer lane")
    watch.add_argument("--idle-seconds", type=float, default=45.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    optimizer = QwenScientificOptimizer(physical_root=args.physical_root)
    if args.command == "once":
        optimizer.run_cycle()
        return 0
    return optimizer.watch(idle_seconds=float(args.idle_seconds))


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper.
    raise SystemExit(main())
