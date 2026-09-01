"""Parameterized Accelerator experiment specs and safe HCLI dispatch.

The repository already has a general ``lab`` experiment runtime and AgentOS
has durable background jobs.  This module is the thin accelerator-specific
boundary between them: it turns architecture-repatriation hypotheses into
explicit inputs, a shell-free command, a verification ladder, and an HCLI
WorkUnit-compatible admission record.

Planning and queue compilation never launch work.  ``AcceleratorRunner.start``
only creates a detached job when its caller explicitly asks for execution and
the physical-window admission check is true.  A blocked or contended physical
experiment therefore remains visible without disturbing an unrelated live
model download or benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hcli.persist import atomic_write_json
from hcli.workunit import WorkUnit
from tools.accelerator.architecture_atlas import (
    PLANNING_BENCH,
    QUEUE_SCHEMA as ATLAS_QUEUE_SCHEMA,
    SCHEMA as ATLAS_SCHEMA,
    build_atlas,
    validate_atlas,
)


SPEC_SCHEMA = "hawking.accelerator.experiment_spec.v1"
COMPILED_QUEUE_SCHEMA = "hawking.accelerator.repatriation_queue_compiled.v1"
DEFAULT_QUEUE_OUT = Path("receipts/headless/ACCELERATOR_REPATRIATION_QUEUE.json")
_ALLOWED_STATUS = frozenset({"READY", "BLOCKED"})
_REQUIRED_METRICS = frozenset(
    {
        "complete_useful_wall_ns",
        "active_bytes_per_token",
        "fallback_count",
        "capability_verified",
    }
)
_SECRET_ARGUMENT_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|authorization|password|secret|private[_-]?key|bearer|(?:hf|gh|github|openai|anthropic)[_-]?token)\s*[:=]"
)
_MODEL_IDENTITIES = {
    "Qwen27": "Qwen3.8-27B",
    "Flash": "Qwen3.8-Flash-Next",
}
_NX_IDENTITIES = {
    "Qwen27": "QWEN27_NX_CURRENT_SEALED_3.14",
    "Flash": "FLASH_NEXT_NX_PROTECTED_NOT_YET_QUALIFIED",
}
_RESOURCE_CLASSES = {
    "metal": "GPU_EXCLUSIVE",
    "cuda": "GPU_EXCLUSIVE",
    "ane": "GPU_EXCLUSIVE",
    "fpga": "COMPILE",
    "cpu": "CPU_HEAVY",
    "remote": "IO_HEAVY",
}


class RunnerSpecError(ValueError):
    """A queue/spec is incomplete, unsafe, or overclaims execution."""


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _strings(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RunnerSpecError(f"{field_name} must be a list")
    values = tuple(str(item) for item in value)
    if not values or any(not item for item in values):
        raise RunnerSpecError(f"{field_name} must contain non-empty strings")
    return values


@dataclass(frozen=True)
class AcceleratorExperimentSpec:
    """All inputs needed to reproduce one accelerator experiment."""

    experiment_id: str
    behavior_id: str
    model_identity: str
    nx_identity: str
    nr_identity: str
    organ: str
    organ_range: Mapping[str, Any]
    backend: str
    kernel_lowering: str
    verification_mode: str
    benchmark_mode: str
    state_session_inputs: Mapping[str, Any]
    output_receipt_path: str
    command: tuple[str, ...]
    candidate: str
    control: str
    metrics: tuple[str, ...]
    falsifier: str
    status: str
    detached: bool
    requires_quiescence: bool
    protected_window: bool
    atlas_fingerprint: str
    expected_value_score: float
    expected_effect: Mapping[str, Any] = field(default_factory=dict)
    promotion: Mapping[str, Any] = field(default_factory=dict)
    source_evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SPEC_SCHEMA,
            "experiment_id": self.experiment_id,
            "behavior_id": self.behavior_id,
            "target": {
                "model_identity": self.model_identity,
                "nx_identity": self.nx_identity,
                "nr_identity": self.nr_identity,
                "organ": self.organ,
                "organ_range": _copy(self.organ_range),
                "backend": self.backend,
            },
            "model_identity": self.model_identity,
            "nx_identity": self.nx_identity,
            "nr_identity": self.nr_identity,
            "organ": self.organ,
            "organ_range": _copy(self.organ_range),
            "backend": self.backend,
            "kernel_lowering": self.kernel_lowering,
            "verification_mode": self.verification_mode,
            "benchmark_mode": self.benchmark_mode,
            "state_session_inputs": _copy(self.state_session_inputs),
            "output_receipt_path": self.output_receipt_path,
            "command": list(self.command),
            "candidate": self.candidate,
            "control": self.control,
            "metrics": list(self.metrics),
            "falsifier": self.falsifier,
            "status": self.status,
            "runner": {
                "detached": self.detached,
                "requires_quiescence": self.requires_quiescence,
                "protected_window": self.protected_window,
                "shell": False,
                "resumable": True,
                "lease": "HCLI/Accelerator physical-window lease",
            },
            "atlas_fingerprint": self.atlas_fingerprint,
            "expected_value_score": self.expected_value_score,
            "expected_effect": _copy(self.expected_effect),
            "promotion": _copy(self.promotion),
            "source_evidence": list(self.source_evidence),
            "claim_boundary": (
                "This is an executable hypothesis specification. It is not a "
                "performance result, capability result, or source-product claim."
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AcceleratorExperimentSpec":
        if not isinstance(raw, Mapping):
            raise RunnerSpecError("experiment spec must be an object")
        target = raw.get("target") if isinstance(raw.get("target"), Mapping) else {}
        runner = raw.get("runner") if isinstance(raw.get("runner"), Mapping) else {}
        spec = cls(
            experiment_id=str(raw.get("experiment_id") or ""),
            behavior_id=str(raw.get("behavior_id") or ""),
            model_identity=str(raw.get("model_identity") or target.get("model_identity") or ""),
            nx_identity=str(raw.get("nx_identity") or target.get("nx_identity") or ""),
            nr_identity=str(raw.get("nr_identity") or target.get("nr_identity") or ""),
            organ=str(raw.get("organ") or target.get("organ") or ""),
            organ_range=dict(raw.get("organ_range") or target.get("organ_range") or {}),
            backend=str(raw.get("backend") or target.get("backend") or ""),
            kernel_lowering=str(raw.get("kernel_lowering") or ""),
            verification_mode=str(raw.get("verification_mode") or ""),
            benchmark_mode=str(raw.get("benchmark_mode") or ""),
            state_session_inputs=dict(raw.get("state_session_inputs") or {}),
            output_receipt_path=str(raw.get("output_receipt_path") or ""),
            command=_strings(raw.get("command"), field_name="command"),
            candidate=str(raw.get("candidate") or ""),
            control=str(raw.get("control") or ""),
            metrics=_strings(raw.get("metrics"), field_name="metrics"),
            falsifier=str(raw.get("falsifier") or ""),
            status=str(raw.get("status") or ""),
            detached=bool(runner.get("detached", raw.get("detached", False))),
            requires_quiescence=bool(
                runner.get("requires_quiescence", raw.get("requires_quiescence", False))
            ),
            protected_window=bool(
                runner.get("protected_window", raw.get("protected_window", False))
            ),
            atlas_fingerprint=str(raw.get("atlas_fingerprint") or ""),
            expected_value_score=float(raw.get("expected_value_score") or 0.0),
            expected_effect=dict(raw.get("expected_effect") or {}),
            promotion=dict(raw.get("promotion") or {}),
            source_evidence=_strings(raw.get("source_evidence") or ["unknown"], field_name="source_evidence"),
        )
        validate_experiment_spec(spec)
        return spec


def validate_experiment_spec(spec: AcceleratorExperimentSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Validate the runner contract without touching hardware."""

    candidate = spec if isinstance(spec, AcceleratorExperimentSpec) else AcceleratorExperimentSpec.from_dict(spec)
    if not candidate.experiment_id or not candidate.behavior_id:
        raise RunnerSpecError("experiment_id and behavior_id are required")
    if candidate.status not in _ALLOWED_STATUS:
        raise RunnerSpecError(f"invalid experiment status {candidate.status!r}")
    for name in (
        "model_identity",
        "nx_identity",
        "nr_identity",
        "organ",
        "backend",
        "kernel_lowering",
        "verification_mode",
        "benchmark_mode",
        "output_receipt_path",
        "candidate",
        "control",
        "falsifier",
        "atlas_fingerprint",
    ):
        if not str(getattr(candidate, name)).strip():
            raise RunnerSpecError(f"{name} must be non-empty")
    if not isinstance(candidate.organ_range, Mapping) or not candidate.organ_range:
        raise RunnerSpecError("organ_range must be an object")
    if Path(candidate.output_receipt_path).is_absolute():
        raise RunnerSpecError("output_receipt_path must be repository-relative")
    if not candidate.detached:
        raise RunnerSpecError("accelerator work must be detached")
    if candidate.protected_window and not candidate.requires_quiescence:
        raise RunnerSpecError("protected_window requires quiescence admission")
    if not _REQUIRED_METRICS.issubset(set(candidate.metrics)):
        missing = sorted(_REQUIRED_METRICS - set(candidate.metrics))
        raise RunnerSpecError(f"required physical metrics missing: {missing}")
    if any("\x00" in item or "\n" in item or "\r" in item for item in candidate.command):
        raise RunnerSpecError("command contains a control character")
    if any(_SECRET_ARGUMENT_RE.search(item) for item in candidate.command):
        raise RunnerSpecError("credential-shaped command arguments must be supplied by environment")
    if any(item in {"--shell", "-c", "-lc"} for item in candidate.command):
        raise RunnerSpecError("accelerator runner does not accept shell commands")
    if candidate.status == "BLOCKED" and candidate.protected_window:
        # A blocked item may remain in the queue as a future hypothesis, but its
        # record must not accidentally look executable.
        if not candidate.state_session_inputs.get("blocked_reason"):
            raise RunnerSpecError("blocked protected experiment needs blocked_reason")
    return {
        "schema": "hawking.accelerator.experiment_spec_validation.v1",
        "passed": True,
        "experiment_id": candidate.experiment_id,
        "status": candidate.status,
        "claim_boundary": "contract validation is not a physical measurement",
    }


def _command(row: Mapping[str, Any], output_path: str) -> tuple[str, ...]:
    runner = row.get("runner") if isinstance(row.get("runner"), Mapping) else {}
    raw = runner.get("command")
    values = _strings(raw, field_name="runner.command")
    command = list(values)
    if "--repo-root" not in command:
        command.extend(["--repo-root", "."])
    if "--emit" not in command:
        command.extend(["--emit", output_path])
    return tuple(command)


def _state_session_inputs(model: str, primitive: str, status: str, blocked_reason: Any) -> dict[str, Any]:
    return {
        "session": "persistent_token_session",
        "state_residency": "resident_when_valid; explicit owner and checkpoint boundary",
        "dynamic_slots": ["token", "position", "route", "sampling", "variable_state"],
        "route_before_payload": model == "Flash" and primitive == "DirectRoutedAccumulate",
        "memory_tier_is_executable_identity": True,
        "blocked_reason": str(blocked_reason) if status == "BLOCKED" and blocked_reason else None,
    }


def compile_experiment_specs(
    atlas: Mapping[str, Any],
    *,
    model: str | None = None,
    backend: str | None = None,
) -> list[AcceleratorExperimentSpec]:
    """Compile atlas queue rows into fully parameterized accelerator specs."""

    validate_atlas(atlas)
    entries = {str(row["behavior_id"]): row for row in atlas["entries"]}
    queue = atlas["experiment_queue"]
    if queue.get("schema") != ATLAS_QUEUE_SCHEMA:
        raise RunnerSpecError("atlas queue schema is not supported")
    result: list[AcceleratorExperimentSpec] = []
    for row in queue.get("experiments") or []:
        target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
        model_alias = str(target.get("model") or "")
        backend_name = str(target.get("backend") or "")
        if model and model_alias != model:
            continue
        if backend and backend_name != backend:
            continue
        entry = entries.get(str(row.get("behavior_id")))
        if entry is None:
            raise RunnerSpecError(f"queue references unknown behavior {row.get('behavior_id')!r}")
        experiment_id = str(row.get("experiment_id") or "")
        if not experiment_id:
            raise RunnerSpecError("queue experiment has no experiment_id")
        output_path = f"receipts/headless/ACCELERATOR_REPATRIATION/{experiment_id}.json"
        status = str(row.get("status") or "READY")
        primitive = str(entry["hawking_primitive"])
        spec = AcceleratorExperimentSpec(
            experiment_id=experiment_id,
            behavior_id=str(row["behavior_id"]),
            model_identity=_MODEL_IDENTITIES.get(model_alias, model_alias),
            nx_identity=_NX_IDENTITIES.get(model_alias, f"{model_alias}_NX_UNDECLARED"),
            nr_identity=f"NR_REPATRIATION:{row['behavior_id']}",
            organ=str(target.get("organ") or "all"),
            organ_range={
                "organ": str(target.get("organ") or "all"),
                "range": "whole-body" if str(target.get("organ")) == "all" else "whole-organ",
                "selection": "candidate-defined; no implicit full-model claim",
            },
            backend=backend_name,
            kernel_lowering=primitive,
            verification_mode="structural_then_diagnostic_then_protected",
            benchmark_mode="complete_useful_wall_ns_authority",
            state_session_inputs=_state_session_inputs(
                model_alias, primitive, status, row.get("blocked_reason")
            ),
            output_receipt_path=output_path,
            command=_command(row, output_path),
            candidate=str(row.get("candidate") or ""),
            control=str(row.get("control") or ""),
            metrics=tuple(str(item) for item in row.get("metrics") or ()),
            falsifier=str(row.get("falsifier") or ""),
            status=status,
            detached=bool((row.get("runner") or {}).get("detached")),
            requires_quiescence=bool((row.get("runner") or {}).get("requires_quiescence")),
            protected_window=bool((row.get("runner") or {}).get("protected_window")),
            atlas_fingerprint=str(atlas["fingerprint"]),
            expected_value_score=float(entry.get("expected_value_score") or 0.0),
            expected_effect=entry.get("expected_effect") or {},
            promotion=row.get("promotion") or {},
            source_evidence=tuple(str(item) for item in entry.get("source_evidence") or ()),
        )
        validate_experiment_spec(spec)
        result.append(spec)
    result.sort(key=lambda item: (0 if item.status == "READY" else 1, -item.expected_value_score, item.experiment_id))
    return result


def workunits_for_specs(
    specs: Iterable[AcceleratorExperimentSpec],
    *,
    include_blocked: bool = False,
) -> list[dict[str, Any]]:
    """Return deterministic HCLI WorkUnit records without enqueueing them."""

    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.status == "BLOCKED" and not include_blocked:
            continue
        resource_class = _RESOURCE_CLASSES.get(spec.backend, "LIGHT_CONTROL")
        unit = WorkUnit(
            id=f"accelerator.{spec.experiment_id}",
            role="accelerator_repatriation",
            description=(
                f"Run {spec.candidate} on {spec.model_identity}/{spec.backend} "
                f"for {spec.organ}. Control: {spec.control}. "
                f"Falsifier: {spec.falsifier}"
            ),
            dependencies=[],
            resource_class=resource_class,
            preferred_backend=spec.backend,
            provider="accelerator_runner",
            verifier=f"accelerator.repatriation.{spec.experiment_id}",
            # The command may emit a receipt (and the FPGA pre-board command
            # refreshes derived maps), so this is reversible work rather than
            # read-only inspection.
            effect_class="REVERSIBLE",
            workspace="repo-root",
        )
        row = unit.to_dict()
        row.update(
            {
                "experiment_id": spec.experiment_id,
                "behavior_id": spec.behavior_id,
                "status": "blocked" if spec.status == "BLOCKED" else "pending",
                "command": list(spec.command),
                "output_receipt_path": spec.output_receipt_path,
                "requires_quiescence": spec.requires_quiescence,
                "claim_boundary": "WorkUnit is a proposal; verifier and protected receipt remain authoritative",
            }
        )
        rows.append(row)
    return rows


def build_compiled_queue(
    *,
    repo_root: str | Path | None = None,
    model: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    atlas = build_atlas(repo_root=root)
    specs = compile_experiment_specs(atlas, model=model, backend=backend)
    ready = [spec for spec in specs if spec.status == "READY"]
    blocked = [spec for spec in specs if spec.status == "BLOCKED"]
    spec_rows = [spec.to_dict() for spec in specs]
    workunits = workunits_for_specs(ready)
    ids = {spec.experiment_id for spec in specs}
    ready_ids = {spec.experiment_id for spec in ready}
    protected_ids = {
        spec.experiment_id
        for spec in ready
        if spec.protected_window and spec.command
    }
    body: dict[str, Any] = {
        "schema": COMPILED_QUEUE_SCHEMA,
        "version": 1,
        "bench": dict(PLANNING_BENCH),
        "atlas_schema": ATLAS_SCHEMA,
        "atlas_fingerprint": atlas["fingerprint"],
        "experiment_spec_schema": SPEC_SCHEMA,
        "selection": {
            "model": model,
            "backend": backend,
            "ordering": "ready first, then expected value score descending, then experiment id",
        },
        "specs": spec_rows,
        "work_units": workunits,
        "funnel": {
            "structural_compile_and_negative_controls": sorted(ready_ids),
            "diagnostic_relative_interleaved_ab": sorted(ready_ids),
            "protected_absolute_complete_wall_with_capability_gate": sorted(protected_ids),
            "promotion": [],
            "promotion_rule": "only a protected complete-token receipt may move an id beyond the queue",
        },
        "counts": {
            "specs": len(specs),
            "ready": len(ready),
            "blocked": len(blocked),
            "work_units": len(workunits),
        },
        "runner_policy": {
            "planning_is_side_effect_free": True,
            "commands_are_argv_only": True,
            "physical_start_requires_explicit_execute": True,
            "physical_start_requires_quiescence": True,
            "protected_benchmark_uses_existing_hcli_lease": True,
            "interrupted_jobs_rerun_from_start": True,
        },
        "claim_boundary": (
            "This compiled queue is a durable set of hypotheses and HCLI work "
            "proposals. It contains no physical timing, capability, or hardware claim."
        ),
    }
    body["fingerprint"] = _hash({key: value for key, value in body.items() if key != "fingerprint"})
    return body


def validate_compiled_queue(queue: Mapping[str, Any]) -> dict[str, Any]:
    if queue.get("schema") != COMPILED_QUEUE_SCHEMA:
        raise RunnerSpecError(f"schema must be {COMPILED_QUEUE_SCHEMA}")
    if queue.get("atlas_schema") != ATLAS_SCHEMA:
        raise RunnerSpecError("compiled queue references the wrong atlas schema")
    if queue.get("experiment_spec_schema") != SPEC_SCHEMA:
        raise RunnerSpecError("compiled queue references the wrong experiment spec schema")
    specs_raw = queue.get("specs")
    if not isinstance(specs_raw, list) or not specs_raw:
        raise RunnerSpecError("compiled queue specs must be a non-empty list")
    ids: set[str] = set()
    ready_ids: set[str] = set()
    for raw in specs_raw:
        spec = AcceleratorExperimentSpec.from_dict(raw)
        if spec.experiment_id in ids:
            raise RunnerSpecError(f"duplicate experiment id {spec.experiment_id!r}")
        ids.add(spec.experiment_id)
        if spec.atlas_fingerprint != queue.get("atlas_fingerprint"):
            raise RunnerSpecError(f"{spec.experiment_id} has a stale atlas fingerprint")
        if spec.status == "READY":
            ready_ids.add(spec.experiment_id)
    workunits = queue.get("work_units")
    if not isinstance(workunits, list):
        raise RunnerSpecError("compiled queue work_units must be a list")
    workunit_ids = set()
    for row in workunits:
        if not isinstance(row, Mapping):
            raise RunnerSpecError("work_units must contain objects")
        experiment_id = str(row.get("experiment_id") or "")
        if experiment_id not in ready_ids:
            raise RunnerSpecError(f"work unit references non-ready experiment {experiment_id!r}")
        workunit_ids.add(experiment_id)
    if workunit_ids != ready_ids:
        raise RunnerSpecError("every ready experiment must have exactly one HCLI WorkUnit")
    funnel = queue.get("funnel")
    if not isinstance(funnel, Mapping):
        raise RunnerSpecError("compiled queue funnel is missing")
    for key in (
        "structural_compile_and_negative_controls",
        "diagnostic_relative_interleaved_ab",
        "protected_absolute_complete_wall_with_capability_gate",
        "promotion",
    ):
        values = funnel.get(key)
        if not isinstance(values, list) or not set(values).issubset(ids):
            raise RunnerSpecError(f"funnel stage {key!r} references unknown experiments")
    expected_hash = _hash({key: value for key, value in queue.items() if key != "fingerprint"})
    if queue.get("fingerprint") != expected_hash:
        raise RunnerSpecError("compiled queue fingerprint does not match canonical body")
    counts = queue.get("counts") if isinstance(queue.get("counts"), Mapping) else {}
    if counts.get("specs") != len(specs_raw) or counts.get("ready") != len(ready_ids):
        raise RunnerSpecError("compiled queue counts are stale")
    return {
        "schema": "hawking.accelerator.repatriation_queue_validation.v1",
        "passed": True,
        "spec_count": len(specs_raw),
        "ready_count": len(ready_ids),
        "work_unit_count": len(workunits),
        "claim_boundary": "validation proves queue structure, not physical performance",
    }


def emit_compiled_queue(
    *,
    repo_root: str | Path | None = None,
    output: str | Path | None = None,
    model: str | None = None,
    backend: str | None = None,
) -> Path:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    destination = Path(output).expanduser() if output else root / DEFAULT_QUEUE_OUT
    if not destination.is_absolute():
        destination = root / destination
    body = build_compiled_queue(repo_root=root, model=model, backend=backend)
    validate_compiled_queue(body)
    atomic_write_json(destination, body)
    return destination


class AcceleratorRunner:
    """Safe adapter over AgentOS detached jobs for one compiled spec."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        repo_root: str | Path | None = None,
        quiescence: Callable[[], Mapping[str, Any]] | None = None,
        background_store_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root else self.workspace
        self._quiescence = quiescence
        self._background_store_factory = background_store_factory

    def plan(self, spec: AcceleratorExperimentSpec) -> dict[str, Any]:
        validate_experiment_spec(spec)
        return {
            "schema": "hawking.accelerator.runner_plan.v1",
            "status": "BLOCKED" if spec.status == "BLOCKED" else "PLANNED",
            "experiment_id": spec.experiment_id,
            "spec": spec.to_dict(),
            "started": False,
            "claim_boundary": "plan-only; no process was started",
        }

    def start(
        self,
        spec: AcceleratorExperimentSpec,
        *,
        execute: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Plan by default; explicitly execute only after admission succeeds."""

        validate_experiment_spec(spec)
        if spec.status == "BLOCKED":
            return {
                "schema": "hawking.accelerator.runner_result.v1",
                "status": "BLOCKED",
                "experiment_id": spec.experiment_id,
                "started": False,
                "reason": spec.state_session_inputs.get("blocked_reason") or "queue marked blocked",
            }
        if not execute:
            planned = self.plan(spec)
            planned["status"] = "PLANNED"
            planned["reason"] = "explicit execute=True is required to start detached work"
            return planned
        if spec.requires_quiescence:
            sample = self._quiescence() if self._quiescence else _machine_quiescence()
            if not isinstance(sample, Mapping) or sample.get("quiet") is not True:
                return {
                    "schema": "hawking.accelerator.runner_result.v1",
                    "status": "WAITING_FOR_QUIESCENCE",
                    "experiment_id": spec.experiment_id,
                    "started": False,
                    "quiescence": dict(sample) if isinstance(sample, Mapping) else None,
                    "claim_boundary": "no detached job was created while the physical window was unavailable",
                }
        if self._background_store_factory is None:
            from hcli.agentos.background import BackgroundJobStore

            factory = BackgroundJobStore
        else:
            factory = self._background_store_factory
        store = factory(self.workspace, allowed_roots=(self.repo_root,))
        job = store.start(
            spec.command,
            cwd=self.repo_root,
            label=f"accelerator:{spec.experiment_id}",
            resumable=True,
            timeout_s=timeout_s,
        )
        return {
            "schema": "hawking.accelerator.runner_result.v1",
            "status": "DETACHED",
            "experiment_id": spec.experiment_id,
            "started": True,
            "job": job,
            "claim_boundary": "detached launch only; the produced receipt remains subject to verification and promotion gates",
        }


def _machine_quiescence() -> Mapping[str, Any]:
    try:
        from tools.accelerator.bench import machine_quiescence

        return machine_quiescence()
    except Exception as exc:  # noqa: BLE001 - unavailable admission is not quiet
        return {"quiet": None, "method": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--emit", default=None)
    parser.add_argument("--validate", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.validate:
        value = json.loads(Path(args.validate).expanduser().read_text(encoding="utf-8"))
        print(json.dumps(validate_compiled_queue(value), indent=2, sort_keys=True))
        return 0
    destination = emit_compiled_queue(
        repo_root=args.repo_root,
        output=args.emit,
        model=args.model,
        backend=args.backend,
    )
    body = json.loads(destination.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "PASSED",
                "path": str(destination),
                "fingerprint": body["fingerprint"],
                "specs": body["counts"]["specs"],
                "work_units": body["counts"]["work_units"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "AcceleratorExperimentSpec",
    "AcceleratorRunner",
    "COMPILED_QUEUE_SCHEMA",
    "DEFAULT_QUEUE_OUT",
    "RunnerSpecError",
    "SPEC_SCHEMA",
    "build_compiled_queue",
    "compile_experiment_specs",
    "emit_compiled_queue",
    "main",
    "validate_compiled_queue",
    "validate_experiment_spec",
    "workunits_for_specs",
]


if __name__ == "__main__":
    raise SystemExit(main())
