"""Shared accelerator backend contract.

Every future device -- Metal, CPU, ANE, FPGA, later CUDA -- answers the same
typed surface: program in, capability, cost, lowering, validation, execute.
Adding a backend is incremental because the vocabulary is already owned:

  semantic_transport.DomainKind / Cost / ExecutionDomain
  placement.NodeKind / Placement
  architecture_atlas.PRIMITIVES
  fusion_bridge host GPU_UMA + declared FPGA_HBM helpers
  tools.future.hwir (FPGA lowering)
  repatriation_effects (scoped effect levels; no physical-law promotion)
  akb AKB-MACHINE-BANDWIDTH (the law this module repatriates)

CUDA is named, not registered. FPGA never emits HARDWARE_MEASURED.

Evidence tiers are honest and never merged:
  STATIC / FUNCTIONAL_SIM / COST_MODEL / CYCLE_APPROX / HARDWARE_MEASURED
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

_ACCEL = Path(__file__).resolve().parent
_REPO = _ACCEL.parents[1]
for _p in (str(_ACCEL), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from architecture_atlas import PRIMITIVES
from fusion_bridge import (
    DECLARED_BRIDGE_BW_GB_S,
    DECLARED_BRIDGE_LATENCY_S,
    DECLARED_FPGA_HBM,
    HOST_GPU_UMA,
    declared_fpga_hbm_domain,
    host_gpu_uma_domain,
)
from placement import (
    NodeKind,
    Placement,
    ResourceRequirements,
    ValidityCondition,
    kind_admits,
)
from repatriation_effects import GENERICITY, LEVELS, bind_cost_feature
from semantic_transport import (
    COST_MODEL,
    HARDWARE_MEASURED,
    CoherencyAssumption,
    ComputeVisibilityEvidence,
    Cost,
    DomainKind,
    DomainVisibility,
    ExecutionDomain,
    PayloadSemantics,
    cost_model,
    store_and_forward_cost,
)


REPO = _REPO
SCHEMA = "hawking.accelerator.backend_contract.v1"

# Honest evidence tiers. Never merge; never promote a model into a measurement.
STATIC = "STATIC"
FUNCTIONAL_SIM = "FUNCTIONAL_SIM"
CYCLE_APPROX = "CYCLE_APPROX"
EVIDENCE_TIERS = (
    STATIC,
    FUNCTIONAL_SIM,
    COST_MODEL,
    CYCLE_APPROX,
    HARDWARE_MEASURED,
)

# Product names the contract instantiates. Atlas uses lowercase; these ids are
# the registry keys. CUDA is the named-not-built abstraction.
BACKEND_IDS = ("CPU", "METAL", "ANE", "FPGA-HWIR")
CUDA_BACKEND_ID = "CUDA"
FPGA_BACKEND_ID = "FPGA-HWIR"
ATLAS_BACKEND_NAME = {
    "CPU": "cpu",
    "METAL": "metal",
    "ANE": "ane",
    "FPGA-HWIR": "fpga",
    CUDA_BACKEND_ID: "cuda",
}
PRODUCT_TO_KIND = {
    "CPU": DomainKind.CPU,
    "METAL": DomainKind.GPU_UMA,
    "ANE": DomainKind.NPU,
    "FPGA-HWIR": DomainKind.FPGA_HBM,
}

LEGAL_FPGA_TIERS = frozenset({STATIC, FUNCTIONAL_SIM, COST_MODEL, CYCLE_APPROX})

MACHINE_GENOME_RECEIPT = "receipts/headless/ACCELERATOR_MACHINE_GENOME.json"
ANE_DEVICE_PROFILE = "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json"
AKB_MACHINE_BANDWIDTH = "AKB-MACHINE-BANDWIDTH"

# Declared knobs. Not measurements. Used only when no repatriated feature is bound.
CPU_DECLARED_BW_GB_S = 50.0
METAL_DECLARED_BW_GB_S = 200.0
ANE_DECLARED_BW_GB_S = 80.0
CPU_DECLARED_LATENCY_S = 1.0e-6
METAL_DECLARED_LATENCY_S = 5.0e-6
ANE_DECLARED_LATENCY_S = 2.0e-5

ELEMENTWISE_KINDS = ("add", "mul")


class BackendContractError(RuntimeError):
    """Base for every error this module raises."""


class FpgaHardwareClaimError(BackendContractError):
    """FPGA-HWIR tried to emit HARDWARE_MEASURED. No board is present."""


class BackendNotRegistered(BackendContractError):
    """Asked for a backend the registry does not instantiate."""


# --------------------------------------------------------------------------- JSON / receipt load (disk, else git show for sparse checkouts)


def load_repo_json(rel: str, *, root: Path | None = None) -> dict[str, Any]:
    """Load a repo-relative JSON object. On-disk first; `git show HEAD:rel` if sparse."""
    base = Path(root) if root is not None else REPO
    path = Path(rel)
    if not path.is_absolute():
        path = base / path
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        proc = subprocess.run(
            ["git", "-C", str(base), "show", f"HEAD:{rel}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise FileNotFoundError(
                f"{rel} is not on disk and git show HEAD:{rel} failed: "
                f"{(proc.stderr or proc.stdout).strip()[:400]}"
            )
        value = json.loads(proc.stdout)
    if not isinstance(value, dict):
        raise BackendContractError(f"{rel} is not a JSON object")
    return value


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
    except Exception:
        return False
    return True


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# --------------------------------------------------------------------------- FPGA evidence guard (MUTATION_POINT)


def fpga_evidence_tier(claimed: str | None = None) -> str:
    """Honest FPGA evidence. MUTATION_POINT for the negative control.

    FPGA-HWIR is declared-only: there is no board on this machine. The only
    legal tiers are STATIC, FUNCTIONAL_SIM, COST_MODEL, CYCLE_APPROX.
    HARDWARE_MEASURED is refused on every path that would emit a tier.
    """
    tier = claimed or COST_MODEL
    if tier == HARDWARE_MEASURED:
        raise FpgaHardwareClaimError(
            "FPGA-HWIR cannot emit HARDWARE_MEASURED; no FPGA board is present"
        )
    if tier not in LEGAL_FPGA_TIERS:
        raise BackendContractError(
            f"FPGA-HWIR evidence tier {tier!r} is not one of {sorted(LEGAL_FPGA_TIERS)}"
        )
    return tier


def collect_evidence_tiers(payload: Any) -> set[str]:
    """Walk a nested payload and collect any evidence-tier strings."""
    found: set[str] = set()
    if isinstance(payload, str) and payload in EVIDENCE_TIERS:
        found.add(payload)
    elif isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in {"evidence_tier", "label", "tier"} and isinstance(value, str):
                if value in EVIDENCE_TIERS:
                    found.add(value)
            found |= collect_evidence_tiers(value)
    elif isinstance(payload, (list, tuple, set)):
        for item in payload:
            found |= collect_evidence_tiers(item)
    return found


# --------------------------------------------------------------------------- typed program / answers


@dataclass(frozen=True)
class ProgramOp:
    """One node of a device-neutral program. `primitive` is an atlas name."""
    op_id: str
    primitive: str
    node_kind: NodeKind
    payload: PayloadSemantics = PayloadSemantics.ACTIVATION
    nbytes: int = 0
    flops: int = 0
    opcode: str = "mul"

    def __post_init__(self) -> None:
        if not self.op_id:
            raise BackendContractError("ProgramOp.op_id must be non-empty")
        if self.primitive not in PRIMITIVES:
            raise BackendContractError(
                f"primitive {self.primitive!r} is not an atlas primitive; "
                f"known: {PRIMITIVES}"
            )
        if self.nbytes < 0 or self.flops < 0:
            raise BackendContractError("nbytes/flops must be >= 0")
        if self.opcode not in ELEMENTWISE_KINDS:
            raise BackendContractError(
                f"opcode {self.opcode!r} is not one of {ELEMENTWISE_KINDS}"
            )


@dataclass(frozen=True)
class TypedProgram:
    """Device-neutral program the contract accepts. Lowering is a backend's job."""
    program_id: str
    ops: tuple[ProgramOp, ...]
    n_elements: int = 4096
    dtype: str = "f32"

    def __post_init__(self) -> None:
        if not self.program_id:
            raise BackendContractError("TypedProgram.program_id must be non-empty")
        if not self.ops:
            raise BackendContractError("TypedProgram.ops must be non-empty")
        if self.n_elements <= 0:
            raise BackendContractError("n_elements must be > 0")
        if self.dtype != "f32":
            raise BackendContractError("only f32 is admitted in this contract slice")

    @property
    def nbytes(self) -> int:
        return int(self.ops[0].nbytes) if self.ops else self.n_elements * 4

    @property
    def opcode(self) -> str:
        return self.ops[0].opcode

    @property
    def primitive(self) -> str:
        return self.ops[0].primitive

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "n_elements": self.n_elements,
            "nbytes": self.nbytes,
            "opcode": self.opcode,
            "ops": [
                {
                    "flops": op.flops,
                    "nbytes": op.nbytes,
                    "node_kind": op.node_kind.value,
                    "op_id": op.op_id,
                    "opcode": op.opcode,
                    "payload": op.payload.value,
                    "primitive": op.primitive,
                }
                for op in self.ops
            ],
            "primitive": self.primitive,
            "program_id": self.program_id,
        }


def sample_elementwise_program(
    *, n: int = 4096, opcode: str = "mul", program_id: str = "contract_elementwise_mul",
) -> TypedProgram:
    """Tiny elementwise program every instantiated backend can lower and run."""
    nbytes = n * 4
    return TypedProgram(
        program_id=program_id,
        n_elements=n,
        ops=(
            ProgramOp(
                op_id="op0",
                primitive="TiledProjection",
                node_kind=NodeKind.COMPUTATION,
                payload=PayloadSemantics.ACTIVATION,
                nbytes=nbytes,
                flops=n,
                opcode=opcode,
            ),
        ),
    )


@dataclass(frozen=True)
class Capability:
    backend_id: str
    domain_kind: DomainKind
    product: str
    physical: bool
    present: bool
    evidence_tier: str
    primitives: tuple[str, ...]
    devices: tuple[str, ...]
    supports_program: bool
    notes: tuple[str, ...] = ()
    atlas_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_name": self.atlas_name or ATLAS_BACKEND_NAME.get(self.backend_id, ""),
            "backend_id": self.backend_id,
            "devices": list(self.devices),
            "domain_kind": self.domain_kind.value,
            "evidence_tier": self.evidence_tier,
            "notes": list(self.notes),
            "physical": self.physical,
            "present": self.present,
            "primitives": list(self.primitives),
            "product": self.product,
            "supports_program": self.supports_program,
        }


@dataclass(frozen=True)
class BackendCost:
    """Planner estimate. `cost.label` is always COST_MODEL (semantic_transport law).
    `evidence_tier` is the honest tier of THIS answer -- FPGA may not set
    HARDWARE_MEASURED even when a consumed law was measured elsewhere."""
    backend_id: str
    cost: Cost
    evidence_tier: str
    features: dict[str, float] = field(default_factory=dict)
    source_law_id: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.cost.label != COST_MODEL:
            raise BackendContractError(
                f"{self.backend_id} BackendCost.cost.label is {self.cost.label!r}; "
                f"estimates must be {COST_MODEL!r} (never {HARDWARE_MEASURED!r})"
            )
        if self.evidence_tier not in EVIDENCE_TIERS:
            raise BackendContractError(
                f"{self.backend_id} unknown evidence tier {self.evidence_tier!r}"
            )
        if self.backend_id == FPGA_BACKEND_ID:
            fpga_evidence_tier(self.evidence_tier)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "cost": self.cost.to_dict(),
            "evidence_tier": self.evidence_tier,
            "features": dict(self.features),
            "note": self.note,
            "source_law_id": self.source_law_id,
        }


@dataclass(frozen=True)
class LoweredProgram:
    backend_id: str
    target: str
    artifact: dict[str, Any]
    evidence_tier: str
    placements: tuple[Placement, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": dict(self.artifact),
            "backend_id": self.backend_id,
            "evidence_tier": self.evidence_tier,
            "note": self.note,
            "placements": [p.to_dict() for p in self.placements],
            "target": self.target,
        }


@dataclass(frozen=True)
class ExecutionResult:
    backend_id: str
    ok: bool
    evidence_tier: str
    simulated: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    wall_s: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "evidence_tier": self.evidence_tier,
            "note": self.note,
            "ok": self.ok,
            "outputs": dict(self.outputs),
            "simulated": self.simulated,
            "wall_s": self.wall_s,
        }


@dataclass(frozen=True)
class NeutralLaw:
    """A backend-neutral law extracted from a measured receipt.

    This is a cost-feature carrier, not a promotion of repatriation_effects
    to PHYSICAL_LAW. `level` stays ACCELERATOR_PRIMITIVE; `genericity` stays
    CANDIDATE_UNVERIFIED.
    """
    law_id: str
    statement: str
    source_backend: str
    source_receipt: str
    hawking_primitive: str
    cost_features: dict[str, float]
    evidence_tier: str
    claim_boundary: str
    knowledge_level: str = "INSTANCE"
    level: str = "ACCELERATOR_PRIMITIVE"
    genericity: str = "CANDIDATE_UNVERIFIED"

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise BackendContractError(f"unknown repatriation level {self.level!r}")
        if self.genericity not in GENERICITY:
            raise BackendContractError(f"unknown genericity {self.genericity!r}")
        if self.hawking_primitive not in PRIMITIVES:
            raise BackendContractError(
                f"law primitive {self.hawking_primitive!r} is not an atlas primitive"
            )
        if self.level == "PHYSICAL_LAW":
            raise BackendContractError(
                "this contract does not promote a receipt into PHYSICAL_LAW"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_boundary": self.claim_boundary,
            "cost_features": dict(self.cost_features),
            "evidence_tier": self.evidence_tier,
            "genericity": self.genericity,
            "hawking_primitive": self.hawking_primitive,
            "knowledge_level": self.knowledge_level,
            "law_id": self.law_id,
            "level": self.level,
            "source_backend": self.source_backend,
            "source_receipt": self.source_receipt,
            "statement": self.statement,
        }


@dataclass(frozen=True)
class RepatriationTrace:
    source_receipt: str
    source_backend: str
    consumer_backend: str
    law: NeutralLaw
    consumer_cost: BackendCost
    binding: dict[str, Any]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding": dict(self.binding),
            "consumer_backend": self.consumer_backend,
            "consumer_cost": self.consumer_cost.to_dict(),
            "law": self.law.to_dict(),
            "note": self.note,
            "source_backend": self.source_backend,
            "source_receipt": self.source_receipt,
        }


@dataclass
class ValidationReport:
    ok: bool
    errors: list[dict[str, str]]

    def codes(self) -> list[str]:
        return [e["code"] for e in self.errors]

    def to_dict(self) -> dict[str, Any]:
        return {"errors": list(self.errors), "ok": self.ok}


def _err(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message, "path": path}


# --------------------------------------------------------------------------- contract


class Backend(ABC):
    """The one interface. A module import is not a call; callers invoke these."""

    backend_id: str
    product: str
    domain_kind: DomainKind

    @abstractmethod
    def execution_domain(self) -> ExecutionDomain:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self, program: TypedProgram | None = None) -> Capability:
        raise NotImplementedError

    @abstractmethod
    def cost(self, program: TypedProgram, *, law: NeutralLaw | None = None) -> BackendCost:
        raise NotImplementedError

    @abstractmethod
    def lower(self, program: TypedProgram) -> LoweredProgram:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self, program: TypedProgram, inputs: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        raise NotImplementedError

    def validate_program(self, program: TypedProgram) -> ValidationReport:
        errors: list[dict[str, str]] = []
        domain = self.execution_domain()
        for i, op in enumerate(program.ops):
            path = f"ops[{i}]"
            if op.primitive not in PRIMITIVES:
                errors.append(_err("UNKNOWN_PRIMITIVE", path, f"{op.primitive!r}"))
            if not kind_admits(domain.kind, op.node_kind):
                errors.append(_err(
                    "KIND_REFUSED",
                    path,
                    f"{domain.kind.value} does not admit {op.node_kind.value}",
                ))
        if program.opcode not in ELEMENTWISE_KINDS:
            errors.append(_err("UNKNOWN_OPCODE", "opcode", program.opcode))
        return ValidationReport(ok=not errors, errors=errors)

    def consume_law(self, law: NeutralLaw, program: TypedProgram) -> BackendCost:
        """Second-backend consumption: a law becomes a cost feature."""
        return self.cost(program, law=law)


def _placement_for(backend: Backend, program: TypedProgram) -> Placement:
    domain = backend.execution_domain()
    return Placement(
        node_id=program.ops[0].op_id,
        node_kind=program.ops[0].node_kind,
        domain=domain.name,
        resources=ResourceRequirements(bytes=program.nbytes, compute_slots=1),
        validity=ValidityCondition(
            domain_registered=True,
            resources_fit=True,
            kind_admitted=kind_admits(domain.kind, program.ops[0].node_kind),
            reason=f"{backend.backend_id} placement of {program.program_id}",
        ),
    )


def _numpy_elementwise(opcode: str, a, b):
    import numpy as np
    if opcode == "add":
        return np.add(a, b)
    return np.multiply(a, b)


def _inputs_or_ones(program: TypedProgram, inputs: Mapping[str, Any] | None):
    import numpy as np
    n = program.n_elements
    if inputs and "a" in inputs and "b" in inputs:
        a = np.asarray(inputs["a"], dtype=np.float32).reshape(-1)[:n]
        b = np.asarray(inputs["b"], dtype=np.float32).reshape(-1)[:n]
        return a, b
    a = np.arange(n, dtype=np.float32)
    b = np.full(n, 2.0, dtype=np.float32)
    return a, b


def _estimate(
    *,
    backend_id: str,
    nbytes: int,
    bandwidth_gb_s: float,
    latency_s: float,
    evidence_tier: str,
    features: Mapping[str, float],
    note: str,
    source_law_id: str | None = None,
) -> BackendCost:
    cost = store_and_forward_cost(
        nbytes=nbytes,
        bandwidth_gb_s=bandwidth_gb_s,
        latency_s=latency_s,
        note=note,
    )
    return BackendCost(
        backend_id=backend_id,
        cost=cost,
        evidence_tier=evidence_tier,
        features=dict(features),
        source_law_id=source_law_id,
        note=note,
    )


def _law_bandwidth(law: NeutralLaw | None, default: float) -> tuple[float, str | None, dict[str, float]]:
    features: dict[str, float] = {}
    if law is None:
        return default, None, features
    bw = float(law.cost_features.get("uma_dram_bandwidth_gb_s") or default)
    features["uma_dram_bandwidth_gb_s"] = bw
    for key, value in law.cost_features.items():
        features[str(key)] = float(value)
    return bw, law.law_id, features


# --------------------------------------------------------------------------- CPU (real on this machine)


class CpuBackend(Backend):
    backend_id = "CPU"
    product = "host-cpu"
    domain_kind = DomainKind.CPU

    def execution_domain(self) -> ExecutionDomain:
        return ExecutionDomain(
            name="cpu_0",
            kind=DomainKind.CPU,
            physical=True,
            capacity_bytes=None,
            visibility=DomainVisibility(
                readback=True,
                device_compute=True,
                device_compute_evidence=ComputeVisibilityEvidence.EXPLICIT_CONTRACT,
            ),
            internal_coherency=CoherencyAssumption.SOFTWARE_MANAGED,
        )

    def capabilities(self, program: TypedProgram | None = None) -> Capability:
        supports = True if program is None else self.validate_program(program).ok
        return Capability(
            backend_id=self.backend_id,
            domain_kind=self.domain_kind,
            product=self.product,
            physical=True,
            present=True,
            evidence_tier=HARDWARE_MEASURED,
            primitives=PRIMITIVES,
            devices=(f"host-cpu:{platform.processor() or platform.machine()}",),
            supports_program=supports,
            notes=("numpy execution on this host CPU is a real run",),
            atlas_name="cpu",
        )

    def cost(self, program: TypedProgram, *, law: NeutralLaw | None = None) -> BackendCost:
        bw, law_id, features = _law_bandwidth(law, CPU_DECLARED_BW_GB_S)
        note = (
            f"CPU store-and-forward; bandwidth {bw} GB/s is a consumed law feature"
            if law_id else
            "CPU store-and-forward; declared COST_MODEL knob, not a measurement"
        )
        return _estimate(
            backend_id=self.backend_id,
            nbytes=program.nbytes,
            bandwidth_gb_s=bw,
            latency_s=CPU_DECLARED_LATENCY_S,
            evidence_tier=COST_MODEL,
            features=features,
            note=note,
            source_law_id=law_id,
        )

    def lower(self, program: TypedProgram) -> LoweredProgram:
        report = self.validate_program(program)
        if not report.ok:
            raise BackendContractError(f"CPU refuse lower: {report.codes()}")
        return LoweredProgram(
            backend_id=self.backend_id,
            target="numpy",
            artifact={
                "kernel": f"numpy.{program.opcode}",
                "dtype": program.dtype,
                "n_elements": program.n_elements,
                "primitive": program.primitive,
            },
            evidence_tier=STATIC,
            placements=(_placement_for(self, program),),
            note="CPU lowering is a numpy kernel name; STATIC",
        )

    def execute(
        self, program: TypedProgram, inputs: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        report = self.validate_program(program)
        if not report.ok:
            return ExecutionResult(
                backend_id=self.backend_id, ok=False, evidence_tier=STATIC,
                simulated=False, note=f"refused: {report.codes()}",
            )
        a, b = _inputs_or_ones(program, inputs)
        t0 = time.perf_counter()
        out = _numpy_elementwise(program.opcode, a, b)
        wall = time.perf_counter() - t0
        return ExecutionResult(
            backend_id=self.backend_id,
            ok=True,
            evidence_tier=HARDWARE_MEASURED,
            simulated=False,
            outputs={"checksum": float(out.sum()), "n": int(out.size)},
            wall_s=wall,
            note="numpy ran on this host CPU; INSTANCE, not a roof",
        )


# --------------------------------------------------------------------------- Metal (real device; mlx dispatch if this interpreter has it)


class MetalBackend(Backend):
    backend_id = "METAL"
    product = "Metal"
    domain_kind = DomainKind.GPU_UMA

    def execution_domain(self) -> ExecutionDomain:
        return host_gpu_uma_domain()

    def capabilities(self, program: TypedProgram | None = None) -> Capability:
        present = _is_apple_silicon()
        mlx = _mlx_available()
        execute_tier = HARDWARE_MEASURED if mlx else FUNCTIONAL_SIM
        supports = True if program is None else self.validate_program(program).ok
        notes = [
            f"host is Apple Silicon: {present}",
            f"mlx importable in this interpreter: {mlx}",
            f"{HOST_GPU_UMA} is fusion_bridge's host GPU_UMA domain",
        ]
        if present and not mlx:
            notes.append(
                "Metal device is present; this interpreter cannot dispatch, so "
                "execute() is FUNCTIONAL_SIM"
            )
        return Capability(
            backend_id=self.backend_id,
            domain_kind=self.domain_kind,
            product=self.product,
            physical=present,
            present=present,
            evidence_tier=execute_tier,
            primitives=PRIMITIVES,
            devices=(HOST_GPU_UMA,) if present else (),
            supports_program=supports,
            notes=tuple(notes),
            atlas_name="metal",
        )

    def cost(self, program: TypedProgram, *, law: NeutralLaw | None = None) -> BackendCost:
        bw, law_id, features = _law_bandwidth(law, METAL_DECLARED_BW_GB_S)
        note = (
            f"Metal store-and-forward; bandwidth {bw} GB/s from consumed law"
            if law_id else
            "Metal store-and-forward; declared COST_MODEL knob, not MACHINE_GENOME"
        )
        return _estimate(
            backend_id=self.backend_id,
            nbytes=program.nbytes,
            bandwidth_gb_s=bw,
            latency_s=METAL_DECLARED_LATENCY_S,
            evidence_tier=COST_MODEL,
            features=features,
            note=note,
            source_law_id=law_id,
        )

    def lower(self, program: TypedProgram) -> LoweredProgram:
        report = self.validate_program(program)
        if not report.ok:
            raise BackendContractError(f"Metal refuse lower: {report.codes()}")
        from air import AirOp, AirProgram, AirTensor, lower_to_msl

        n = program.n_elements
        air_prog = AirProgram(
            name=program.program_id,
            inputs=[
                AirTensor("a", (n,), "f32", memory_domain="APPLE_UM"),
                AirTensor("b", (n,), "f32", memory_domain="APPLE_UM"),
            ],
            ops=[AirOp(kind=program.opcode, inputs=("a", "b"), output="c")],
            output="c",
            device="APPLE_GPU_0",
            output_domain="APPLE_UM",
        )
        msl = lower_to_msl(air_prog)
        return LoweredProgram(
            backend_id=self.backend_id,
            target="air-msl",
            artifact={
                "air_program": air_prog.name,
                "msl": msl,
                "primitive": program.primitive,
                "n_elements": n,
            },
            evidence_tier=STATIC,
            placements=(_placement_for(self, program),),
            note="AIR elementwise lowering to MSL; STATIC. Execution is a separate path.",
        )

    def execute(
        self, program: TypedProgram, inputs: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        report = self.validate_program(program)
        if not report.ok:
            return ExecutionResult(
                backend_id=self.backend_id, ok=False, evidence_tier=STATIC,
                simulated=True, note=f"refused: {report.codes()}",
            )
        a, b = _inputs_or_ones(program, inputs)
        if _mlx_available():
            import mlx.core as mx
            t0 = time.perf_counter()
            xa = mx.array(a)
            xb = mx.array(b)
            xc = xa + xb if program.opcode == "add" else xa * xb
            mx.eval(xc)
            wall = time.perf_counter() - t0
            return ExecutionResult(
                backend_id=self.backend_id,
                ok=True,
                evidence_tier=HARDWARE_MEASURED,
                simulated=False,
                outputs={"checksum": float(xc.sum()), "n": int(program.n_elements)},
                wall_s=wall,
                note="mlx ran on Metal; INSTANCE, not a roof",
            )
        t0 = time.perf_counter()
        out = _numpy_elementwise(program.opcode, a, b)
        wall = time.perf_counter() - t0
        return ExecutionResult(
            backend_id=self.backend_id,
            ok=True,
            evidence_tier=FUNCTIONAL_SIM,
            simulated=True,
            outputs={"checksum": float(out.sum()), "n": int(out.size)},
            wall_s=wall,
            note=(
                "Metal device is present on this Apple Silicon host; mlx is not "
                "importable in this interpreter, so execute is FUNCTIONAL_SIM"
            ),
        )


# --------------------------------------------------------------------------- ANE (device present; execute is a functional sim in this interpreter)


def _ane_profile() -> dict[str, Any]:
    return load_repo_json(ANE_DEVICE_PROFILE)


class AneBackend(Backend):
    backend_id = "ANE"
    product = "Apple Neural Engine"
    domain_kind = DomainKind.NPU

    def execution_domain(self) -> ExecutionDomain:
        return ExecutionDomain(
            name="npu_ane_0",
            kind=DomainKind.NPU,
            physical=_is_apple_silicon(),
            capacity_bytes=None,
            visibility=DomainVisibility(
                readback=True,
                device_compute=True,
                device_compute_evidence=ComputeVisibilityEvidence.EXPLICIT_CONTRACT,
            ),
            internal_coherency=CoherencyAssumption.SOFTWARE_MANAGED,
        )

    def capabilities(self, program: TypedProgram | None = None) -> Capability:
        profile = _ane_profile()
        present = bool(profile.get("neural_engine_present"))
        devices = tuple(
            str(row.get("kind") or "")
            for row in (profile.get("compute_devices") or [])
            if isinstance(row, Mapping)
        )
        supports = True if program is None else self.validate_program(program).ok
        return Capability(
            backend_id=self.backend_id,
            domain_kind=self.domain_kind,
            product=self.product,
            physical=present,
            present=present,
            evidence_tier=FUNCTIONAL_SIM,
            primitives=PRIMITIVES,
            devices=devices,
            supports_program=supports,
            notes=(
                f"MLComputePlan profile {ANE_DEVICE_PROFILE}",
                f"status={profile.get('status')}",
                "public API only; preferred placement is not a runtime measurement",
                "execute() is FUNCTIONAL_SIM in this interpreter (no Core ML runtime here)",
            ),
            atlas_name="ane",
        )

    def cost(self, program: TypedProgram, *, law: NeutralLaw | None = None) -> BackendCost:
        bw, law_id, features = _law_bandwidth(law, ANE_DECLARED_BW_GB_S)
        note = (
            f"ANE store-and-forward; bandwidth {bw} GB/s from consumed law"
            if law_id else
            "ANE store-and-forward; declared COST_MODEL knob. MLComputePlan is not a timing claim."
        )
        return _estimate(
            backend_id=self.backend_id,
            nbytes=program.nbytes,
            bandwidth_gb_s=bw,
            latency_s=ANE_DECLARED_LATENCY_S,
            evidence_tier=COST_MODEL,
            features=features,
            note=note,
            source_law_id=law_id,
        )

    def lower(self, program: TypedProgram) -> LoweredProgram:
        report = self.validate_program(program)
        if not report.ok:
            raise BackendContractError(f"ANE refuse lower: {report.codes()}")
        profile = _ane_profile()
        plan = profile.get("mlcomputeplan") or {}
        ops = list(plan.get("operations") or [])
        preferred = None
        supported: list[str] = []
        if ops and isinstance(ops[0], Mapping):
            preferred = ops[0].get("preferred")
            supported = list(ops[0].get("supported") or [])
        operator = "ios16.mul" if program.opcode == "mul" else "ios16.add"
        return LoweredProgram(
            backend_id=self.backend_id,
            target="mlcomputeplan",
            artifact={
                "api": plan.get("api") or "MLComputePlan.load(contentsOf:configuration:)",
                "operator": operator,
                "preferred": preferred,
                "supported": supported,
                "model_structure": plan.get("model_structure"),
                "status": plan.get("status"),
                "primitive": program.primitive,
                "public_api_only": True,
            },
            evidence_tier=STATIC,
            placements=(_placement_for(self, program),),
            note=(
                "MLComputePlan operation support and preferred placement; "
                "not a runtime latency or energy claim"
            ),
        )

    def execute(
        self, program: TypedProgram, inputs: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        report = self.validate_program(program)
        if not report.ok:
            return ExecutionResult(
                backend_id=self.backend_id, ok=False, evidence_tier=STATIC,
                simulated=True, note=f"refused: {report.codes()}",
            )
        a, b = _inputs_or_ones(program, inputs)
        t0 = time.perf_counter()
        out = _numpy_elementwise(program.opcode, a, b)
        wall = time.perf_counter() - t0
        return ExecutionResult(
            backend_id=self.backend_id,
            ok=True,
            evidence_tier=FUNCTIONAL_SIM,
            simulated=True,
            outputs={"checksum": float(out.sum()), "n": int(out.size)},
            wall_s=wall,
            note=(
                "ANE is present (MLComputePlan NEURAL_ENGINE). This interpreter "
                "does not run MLModel.prediction; execute is FUNCTIONAL_SIM of ios16.mul/add"
            ),
        )


# --------------------------------------------------------------------------- FPGA-HWIR (declared only; never HARDWARE_MEASURED)


class FpgaHwirBackend(Backend):
    backend_id = FPGA_BACKEND_ID
    product = "FPGA-HWIR"
    domain_kind = DomainKind.FPGA_HBM

    def execution_domain(self) -> ExecutionDomain:
        return declared_fpga_hbm_domain()

    def capabilities(self, program: TypedProgram | None = None) -> Capability:
        tier = fpga_evidence_tier(COST_MODEL)
        supports = True if program is None else self.validate_program(program).ok
        return Capability(
            backend_id=self.backend_id,
            domain_kind=self.domain_kind,
            product=self.product,
            physical=False,
            present=False,
            evidence_tier=tier,
            primitives=PRIMITIVES,
            devices=(DECLARED_FPGA_HBM,),
            supports_program=supports,
            notes=(
                "declared FPGA-HWIR domain; no board on this machine",
                "costs are COST_MODEL knobs; execute is FUNCTIONAL_SIM of HWIR",
                "HARDWARE_MEASURED is refused on every path",
            ),
            atlas_name="fpga",
        )

    def cost(self, program: TypedProgram, *, law: NeutralLaw | None = None) -> BackendCost:
        tier = fpga_evidence_tier(COST_MODEL)
        host_bw, law_id, features = _law_bandwidth(law, METAL_DECLARED_BW_GB_S)
        features["declared_bridge_bw_gb_s"] = float(DECLARED_BRIDGE_BW_GB_S)
        features["declared_bridge_latency_s"] = float(DECLARED_BRIDGE_LATENCY_S)
        # Host-side hop can consume a Metal-measured UMA bandwidth as a FEATURE.
        # The FPGA hop stays the declared interconnect knob. Neither hop is an
        # FPGA hardware measurement.
        host = store_and_forward_cost(
            nbytes=program.nbytes,
            bandwidth_gb_s=host_bw,
            latency_s=METAL_DECLARED_LATENCY_S,
            note="host UMA hop; COST_MODEL (feature may be a consumed Metal measurement)",
        )
        bridge = store_and_forward_cost(
            nbytes=program.nbytes,
            bandwidth_gb_s=DECLARED_BRIDGE_BW_GB_S,
            latency_s=DECLARED_BRIDGE_LATENCY_S,
            note="declared host-to-fpga interconnect; COST_MODEL knob",
        )
        combined = cost_model(
            time_s=host.time_s + bridge.time_s,
            nbytes=program.nbytes,
            bandwidth_gb_s=min(host.bandwidth_gb_s, bridge.bandwidth_gb_s),
            latency_s=host.latency_s + bridge.latency_s,
            note=(
                "FPGA-HWIR two-hop COST_MODEL: host UMA feature + declared bridge. "
                "Not a board timing claim."
            ),
            all_hops_physical=False,
        )
        return BackendCost(
            backend_id=self.backend_id,
            cost=combined,
            evidence_tier=tier,
            features=features,
            source_law_id=law_id,
            note=combined.note,
        )

    def lower(self, program: TypedProgram) -> LoweredProgram:
        report = self.validate_program(program)
        if not report.ok:
            raise BackendContractError(f"FPGA refuse lower: {report.codes()}")
        from tools.future import hwir

        tier = fpga_evidence_tier(STATIC)
        node = hwir.HwirNode(
            id=program.ops[0].op_id,
            kind="compute",
            primitive=program.primitive,
            organ=program.program_id,
            inputs={"in": "activation"},
            outputs={"out": "activation"},
        )
        graph = hwir.HwirGraph(
            model="backend-contract",
            organ=program.program_id,
            qualification="STATIC_ONLY",
            semantics_consumed="backend_contract_typed_program",
            nodes=[node],
            edges=[],
            device_budget=hwir.DeviceBudget(
                device_id="declared-fpga-hwir",
                declared_not_measured=True,
            ),
            notes=[
                "lowered from backend_contract.TypedProgram",
                "declared FPGA-HWIR; not a board, bitstream, or timing claim",
            ],
        )
        hwir_report = hwir.validate(graph)
        if not hwir_report.ok:
            raise BackendContractError(f"HWIR refused: {hwir_report.codes()}")
        return LoweredProgram(
            backend_id=self.backend_id,
            target="hwir",
            artifact=graph.to_dict(),
            evidence_tier=tier,
            placements=(_placement_for(self, program),),
            note="HWIR graph, STATIC_ONLY, declared_not_measured budget",
        )

    def execute(
        self, program: TypedProgram, inputs: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        report = self.validate_program(program)
        if not report.ok:
            return ExecutionResult(
                backend_id=self.backend_id, ok=False,
                evidence_tier=fpga_evidence_tier(STATIC),
                simulated=True, note=f"refused: {report.codes()}",
            )
        # Lowering is the FPGA execution path's typed artifact; the numeric
        # kernel is a functional simulation. Never HARDWARE_MEASURED.
        lowered = self.lower(program)
        a, b = _inputs_or_ones(program, inputs)
        t0 = time.perf_counter()
        out = _numpy_elementwise(program.opcode, a, b)
        wall = time.perf_counter() - t0
        return ExecutionResult(
            backend_id=self.backend_id,
            ok=True,
            evidence_tier=fpga_evidence_tier(FUNCTIONAL_SIM),
            simulated=True,
            outputs={
                "checksum": float(out.sum()),
                "n": int(out.size),
                "hwir_schema": lowered.artifact.get("schema"),
                "hwir_qualification": lowered.artifact.get("qualification"),
            },
            wall_s=wall,
            note="FUNCTIONAL_SIM of a declared HWIR graph; no FPGA board",
        )


# --------------------------------------------------------------------------- registry


_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> None:
    _REGISTRY[backend.backend_id] = backend


def _register_defaults() -> None:
    if _REGISTRY:
        return
    register(CpuBackend())
    register(MetalBackend())
    register(AneBackend())
    register(FpgaHwirBackend())


def get_backend(backend_id: str) -> Backend:
    _register_defaults()
    if backend_id == CUDA_BACKEND_ID:
        raise BackendNotRegistered(
            "CUDA is a named future backend; this host is Apple Silicon and "
            "this contract does not instantiate a CUDA implementation"
        )
    backend = _REGISTRY.get(backend_id)
    if backend is None:
        raise BackendNotRegistered(
            f"{backend_id!r} is not registered; known: {list_backend_ids()}"
        )
    return backend


def list_backend_ids() -> tuple[str, ...]:
    _register_defaults()
    return tuple(k for k in BACKEND_IDS if k in _REGISTRY)


def enumerate_backends() -> tuple[Backend, ...]:
    """Stable order: CPU, METAL, ANE, FPGA-HWIR. Call site, not an import."""
    return tuple(get_backend(k) for k in list_backend_ids())


# --------------------------------------------------------------------------- repatriation: Metal measurement -> neutral law -> other backend cost


def law_from_machine_genome(receipt: Mapping[str, Any] | None = None) -> NeutralLaw:
    """AKB-MACHINE-BANDWIDTH from ACCELERATOR_MACHINE_GENOME.json.

    Observed on METAL (APPLE_GPU_0). The number is INSTANCE, one access
    pattern, not a SoC roof. Another backend may consume it as a cost
    feature; that consumption does not become an FPGA measurement.
    """
    doc = dict(receipt) if receipt is not None else load_repo_json(MACHINE_GENOME_RECEIPT)
    result = doc.get("result") if isinstance(doc.get("result"), Mapping) else {}
    measured = result.get("measured_bandwidth") if isinstance(result, Mapping) else {}
    if not isinstance(measured, Mapping):
        raise BackendContractError(f"{MACHINE_GENOME_RECEIPT} has no measured_bandwidth")
    median = measured.get("median_gb_s")
    if median is None:
        raise BackendContractError(
            f"{MACHINE_GENOME_RECEIPT} measured_bandwidth.median_gb_s is missing"
        )
    device = (doc.get("identities") or {}).get("device") or {}
    if str(device.get("api") or "").lower() != "metal":
        raise BackendContractError(
            f"{MACHINE_GENOME_RECEIPT} was not a Metal observation: {device!r}"
        )
    # Cite the AKB entry rather than invent a second law id.
    statement = None
    try:
        import akb
        for entry in akb.LAWS:
            if entry.get("law_id") == AKB_MACHINE_BANDWIDTH:
                statement = str(entry.get("statement") or "")
                break
    except Exception:
        statement = None
    if not statement:
        statement = (
            f"This box's memory bandwidth WAS a {median} GB/s median FOR an f32 "
            "triad (c = a + b). INSTANCE; not a SoC roof."
        )
    return NeutralLaw(
        law_id=AKB_MACHINE_BANDWIDTH,
        statement=statement,
        source_backend="METAL",
        source_receipt=MACHINE_GENOME_RECEIPT,
        hawking_primitive="MemoryTierIdentity",
        cost_features={
            "uma_dram_bandwidth_gb_s": float(median),
            "iqr_spread_pct": float(measured.get("iqr_spread_pct") or 0.0),
            "bytes_moved_per_rep": float(measured.get("bytes_moved_per_rep") or 0.0),
        },
        evidence_tier=HARDWARE_MEASURED,
        claim_boundary=str(doc.get("claim_boundary") or ""),
        knowledge_level=str(doc.get("knowledge_level") or "INSTANCE"),
        level="ACCELERATOR_PRIMITIVE",
        genericity="CANDIDATE_UNVERIFIED",
    )


def repatriate_machine_bandwidth(
    *,
    consumer_id: str = FPGA_BACKEND_ID,
    program: TypedProgram | None = None,
    repo_root: Path | None = None,
) -> RepatriationTrace:
    """End-to-end: Metal receipt -> neutral law -> consumer cost feature.

    Named receipt: receipts/headless/ACCELERATOR_MACHINE_GENOME.json
    """
    receipt = load_repo_json(MACHINE_GENOME_RECEIPT, root=repo_root)
    law = law_from_machine_genome(receipt)
    consumer = get_backend(consumer_id)
    prog = program or sample_elementwise_program()
    cost = consumer.consume_law(law, prog)
    binding = bind_cost_feature(
        law_id=law.law_id,
        features=cost.features,
        source_receipt=law.source_receipt,
        source_backend=law.source_backend,
        consumer_backend=consumer.backend_id,
    )
    return RepatriationTrace(
        source_receipt=MACHINE_GENOME_RECEIPT,
        source_backend="METAL",
        consumer_backend=consumer.backend_id,
        law=law,
        consumer_cost=cost,
        binding=binding,
        note=(
            f"{MACHINE_GENOME_RECEIPT} measured on METAL; {consumer.backend_id} "
            "consumed uma_dram_bandwidth_gb_s as a cost feature. FPGA consumption "
            "stays COST_MODEL."
        ),
    )


def capability_and_cost_snapshot(
    program: TypedProgram | None = None,
) -> dict[str, Any]:
    """Call every registered backend's capability and cost. Used by the audit."""
    prog = program or sample_elementwise_program()
    rows = []
    for backend in enumerate_backends():
        cap = backend.capabilities(prog)
        cst = backend.cost(prog)
        rows.append({
            "backend_id": backend.backend_id,
            "capability": cap.to_dict(),
            "cost": cst.to_dict(),
        })
    return {
        "schema": SCHEMA,
        "backends": rows,
        "count": len(rows),
        "ids": [row["backend_id"] for row in rows],
    }


_register_defaults()


__all__ = [
    "AKB_MACHINE_BANDWIDTH",
    "ANE_DEVICE_PROFILE",
    "BACKEND_IDS",
    "COST_MODEL",
    "CPU_DECLARED_BW_GB_S",
    "CUDA_BACKEND_ID",
    "CYCLE_APPROX",
    "EVIDENCE_TIERS",
    "FPGA_BACKEND_ID",
    "FUNCTIONAL_SIM",
    "HARDWARE_MEASURED",
    "LEGAL_FPGA_TIERS",
    "MACHINE_GENOME_RECEIPT",
    "SCHEMA",
    "STATIC",
    "AneBackend",
    "Backend",
    "BackendContractError",
    "BackendCost",
    "BackendNotRegistered",
    "Capability",
    "CpuBackend",
    "ExecutionResult",
    "FpgaHardwareClaimError",
    "FpgaHwirBackend",
    "LoweredProgram",
    "MetalBackend",
    "NeutralLaw",
    "ProgramOp",
    "RepatriationTrace",
    "TypedProgram",
    "ValidationReport",
    "capability_and_cost_snapshot",
    "collect_evidence_tiers",
    "enumerate_backends",
    "fpga_evidence_tier",
    "get_backend",
    "law_from_machine_genome",
    "list_backend_ids",
    "load_repo_json",
    "repatriate_machine_bandwidth",
    "register",
    "sample_elementwise_program",
]
