"""HCLI future WorkUnit species — types and a non-empty starting queue.

Defines the ten future-work species and emits units INTO the existing HCLI
WorkUnit field set. This module does not schedule, promote, or reimplement
HCLI. Disk-backed live queues remain authoritative for candidate identity.

    python3 tools/future/workunit_species.py --build
    python3 tools/future/workunit_species.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hcli.resources import ResourceClass, normalize_resource_class
from hcli.workunit import (
    DEFAULT_RETRY_BUDGET,
    MAX_REPAIR_DEPTH,
    MAX_REPAIRS_PER_ROOT,
    WorkUnit,
)
from tools.future._common import REPO, git, write_receipt

RECEIPT = "HCLI_FUTURE_WORKUNITS.json"
SCHEMA = "hawking.future.workunit_species.v1"

QUAL_REL = "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
REPAT_REL = "receipts/headless/ACCELERATOR_REPATRIATION_QUEUE.json"

PROPOSAL_CLAIM_BOUNDARY = (
    "WorkUnit is a proposal; receipt and protected capability gates remain authoritative"
)
SIDECAR_CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Cannot promote, "
    "weaken a verifier, choose the Singularity, or perform a destructive mutation."
)

ERAS = ("I", "II", "III", "IV", "V")
ODYSSEYS = ("I", "II", "III")

# Tokens a species may name as bounded_authority. Anything else is refused,
# so a new privilege cannot be smuggled in as a typo.
ALLOWED_AUTHORITY = frozenset(
    {
        "read_receipts",
        "propose_workunit",
        "emit_static_plan",
        "run_cpu_simulation",
        "run_static_analysis",
        "query_negative_index",
        "compile_experiment_spec",
        "record_unknown_metrics",
        "audit_dependency_chain",
        "seed_law_store",
        "write_sidecar_receipt",
        "copy_live_workunit_fields",
        "rank_falsifiable_experiments",
        "inject_faults_in_replica",
        "simulate_fpga_without_board",
        "simulate_fusion_graph",
        "transfer_law_within_declared_scope",
        "adversarially_attack_a_claimed_law",
    }
)

# Explicit refusals. The constructor rejects a species that lists any of these
# in bounded_authority, or that sets the matching may_* flag.
FORBIDDEN_AUTHORITY = frozenset(
    {
        "self_promotion",
        "promote_self",
        "promote_candidate",
        "promote_to_protected_absolute",
        "promote_diagnostic_relative",
        "weaken_verifier",
        "modify_verifier",
        "replace_verifier",
        "disable_verifier",
        "choose_singularity",
        "select_singularity",
        "install_singularity",
        "destructive_mutation",
        "destructive_write",
        "mutate_codex_surface",
        "acquire_gpu_lease",
        "override_bench_state",
        "claim_protected_absolute",
        "claim_hardware_measurement",
    }
)

ALLOWED_EFFECT = frozenset({"REVERSIBLE", "READ_ONLY", "INSPECT"})
KNOWN_RESOURCE = frozenset(item.value for item in ResourceClass)

HCLI_CORE_FIELDS = (
    "id",
    "role",
    "description",
    "dependencies",
    "status",
    "assigned_runtime",
    "attempts",
    "resource_class",
    "repairs",
    "failure_context",
    "preferred_backend",
    "assigned_backend",
    "backend_task_id",
    "verifier",
    "effect_class",
    "workspace",
    "verification",
    "repair_root",
    "repair_depth",
    "repair_reason",
    "repair_exhausted",
    "ready_at",
    "running_at",
    "finished_at",
    "classification",
    "provider",
    "content_hash",
)

# Union of extras observed on live physical + repatriation work_units.
OBSERVED_EXTRAS = (
    "candidate_id",
    "claim_boundary",
    "model",
    "diagnostic_command",
    "protected_command",
    "behavior_id",
    "experiment_id",
    "command",
    "output_receipt_path",
    "requires_quiescence",
    "blocked_reason",
)

# Contract-named core identities (still present on disk as of recovery).
CORE_READY_QWEN27 = (
    "qwen27-affine2-splitk4",
    "qwen27-attention-gate-fusion",
    "qwen27-ba-delta-fusion",
    "qwen27-commit-timing-elision",
    "qwen27-deltanet-inproj-fusion",
    "qwen27-encoder-label-elision",
    "qwen27-fast-profile",
    "qwen27-gqa-qkv-fusion",
    "qwen27-pipeline-cache-reuse",
    "qwen27-pipeline-state-elision",
    "qwen27-q2f-splitk4",
    "qwen27-q4-vecgroup-x64",
)
CORE_BLOCKED_FLASH = (
    "flash-attention-gate-fusion",
    "flash-compact-moe-bf16-vec4",
    "flash-compact-moe-epilogue",
    "flash-encoder-label-elision",
    "flash-fullseq-catalog-cache",
    "flash-fullseq-ordered-encoder",
    "flash-hc-staged-threadgroup",
    "flash-meta-sub1-coherent",
    "flash-pipeline-cache-reuse",
    "flash-qkv-gqa-rope-fusion",
    "flash-routed-fp4-gate-up-swiglu-fused",
    "flash-router-topk-fusion",
    "flash-shared-fp8-gate-up-swiglu-fused",
    "flash-source-bf16-simd",
)

REPAT_RESOURCE = {
    "metal": "GPU_EXCLUSIVE",
    "cuda": "GPU_EXCLUSIVE",
    "ane": "GPU_EXCLUSIVE",
    "fpga": "COMPILE",
    "cpu": "CPU_HEAVY",
    "remote": "IO_HEAVY",
}

SPECIES_IDS = (
    "accelerator_candidate_qualification",
    "architecture_transfer",
    "odyssey_ii_transfer_experiment",
    "odyssey_iii_adversarial_experiment",
    "fpga_simulation",
    "hardware_doctor_experiment",
    "learned_compiler_experiment",
    "fusion_simulation",
    "independent_reproduction",
    "green_machine_measurement",
)


class SpeciesAuthorityError(ValueError):
    """A species declared an authority the future work economy refuses."""


class WorkUnitShapeError(ValueError):
    """An emitted unit does not match the recovered HCLI work_units field set."""


def _checkout_roots() -> list[Path]:
    """This worktree, then the primary checkout (git common-dir parent)."""
    roots: list[Path] = [REPO]
    common = git("rev-parse", "--git-common-dir")
    if common:
        path = Path(common)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        else:
            path = path.resolve()
        parent = path.parent if path.name == ".git" else path.parent
        if parent not in roots:
            roots.append(parent)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def load_headless(rel: str) -> tuple[dict[str, Any] | None, str]:
    """Load a Codex receipt read-only. Missing in this sparse tree is not absence."""
    for root in _checkout_roots():
        path = root / rel
        if path.is_file():
            return json.loads(path.read_text()), str(path)
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            return json.loads(blob), f"git:HEAD:{rel}"
        except json.JSONDecodeError:
            pass
    return None, "recovered_snapshot"


def _stop_grants_authority(stop: str) -> bool:
    """True if the stop condition *instructs* promotion / Singularity choice."""
    text = f" {stop.lower()} "
    if " never " in text or " not " in text or "cannot " in text or "must not " in text:
        return False
    needles = (
        "then promote",
        "promote the",
        "promote to protected",
        "choose the singularity",
        "install the singularity",
        "select the singularity",
    )
    return any(n in text for n in needles)


def _budget(*, gpu_windows_requested: int = 0, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "attempts": DEFAULT_RETRY_BUDGET,
        "max_repair_depth": MAX_REPAIR_DEPTH,
        "max_repairs_per_root": MAX_REPAIRS_PER_ROOT,
        "gpu_windows_requested": int(gpu_windows_requested),
        "gpu_windows_held": 0,
        "wall_clock_s": None,
    }
    if extra:
        body.update(dict(extra))
    return body


def define_species(
    *,
    id: str,
    title: str,
    evidence_parents: Sequence[str],
    bounded_authority: Sequence[str],
    resource_class: str,
    verifier: str,
    budget: Mapping[str, Any],
    stop_condition: str,
    role: str,
    description: str,
    effect_class: str = "REVERSIBLE",
    era: str = "III",
    odyssey: str | None = None,
    preferred_backend: str | None = None,
    may_promote: bool = False,
    may_modify_verifier: bool = False,
    may_choose_singularity: bool = False,
    may_destructive_mutate: bool = False,
) -> dict[str, Any]:
    """Construct one species. Refuses self-promotion and verifier modification."""
    if may_promote:
        raise SpeciesAuthorityError(f"{id}: species may not declare self-promotion authority")
    if may_modify_verifier:
        raise SpeciesAuthorityError(f"{id}: species may not declare verifier-modification authority")
    if may_choose_singularity:
        raise SpeciesAuthorityError(f"{id}: species may not choose the Singularity")
    if may_destructive_mutate:
        raise SpeciesAuthorityError(f"{id}: species may not perform a destructive mutation")

    authority = tuple(str(item) for item in bounded_authority)
    forbidden = [item for item in authority if item in FORBIDDEN_AUTHORITY]
    if forbidden:
        raise SpeciesAuthorityError(
            f"{id}: forbidden authority {forbidden}; a species cannot promote itself, "
            "weaken a verifier, choose the Singularity, or mutate destructively"
        )
    unknown = [item for item in authority if item not in ALLOWED_AUTHORITY]
    if unknown:
        raise SpeciesAuthorityError(f"{id}: unknown authority {unknown} is refused")

    if not str(verifier or "").strip():
        raise SpeciesAuthorityError(f"{id}: verifier is required and is not choosable by the unit")
    lowered = str(verifier).strip().lower()
    if lowered in {"self", "none", "disable", "weaken"}:
        raise SpeciesAuthorityError(f"{id}: verifier {verifier!r} would weaken verification")

    effect = str(effect_class or "").strip().upper()
    if effect not in ALLOWED_EFFECT:
        raise SpeciesAuthorityError(f"{id}: effect_class {effect_class!r} is not permitted")
    if effect in {"DESTRUCTIVE", "MUTATING"}:
        raise SpeciesAuthorityError(f"{id}: destructive effect_class is refused")

    rc = normalize_resource_class(resource_class)
    if rc not in KNOWN_RESOURCE:
        raise SpeciesAuthorityError(f"{id}: resource_class {resource_class!r} is not an HCLI class")
    if rc == "MUTATION":
        raise SpeciesAuthorityError(f"{id}: MUTATION resource_class is not grantable to a species")

    era_n = str(era or "").strip().upper()
    if era_n.startswith("ERA "):
        era_n = era_n[4:].strip()
    if era_n not in ERAS:
        raise SpeciesAuthorityError(f"{id}: era {era!r} is not one of I-V (there is no Era VI)")
    ody = None
    if odyssey is not None and str(odyssey).strip():
        ody = str(odyssey).strip().upper()
        if ody.startswith("ODYSSEY "):
            ody = ody[8:].strip()
        if ody not in ODYSSEYS:
            raise SpeciesAuthorityError(
                f"{id}: odyssey {odyssey!r} is not one of I-III (there is no Odyssey IV)"
            )

    stop = str(stop_condition or "").strip()
    if not stop:
        raise SpeciesAuthorityError(f"{id}: stop_condition is required")
    if _stop_grants_authority(stop):
        raise SpeciesAuthorityError(f"{id}: stop_condition must not grant promotion or Singularity choice")

    parents = tuple(str(p) for p in evidence_parents)
    return {
        "id": str(id),
        "title": str(title),
        "description": str(description),
        "evidence_parents": list(parents),
        "bounded_authority": list(authority),
        "forbidden_authority": sorted(FORBIDDEN_AUTHORITY),
        "resource_class": rc,
        "verifier": str(verifier),
        "budget": dict(budget),
        "stop_condition": stop,
        "role": str(role),
        "effect_class": effect,
        "era": era_n,
        "odyssey": ody,
        "preferred_backend": preferred_backend,
        "may_promote": False,
        "may_modify_verifier": False,
        "may_choose_singularity": False,
        "may_destructive_mutate": False,
        "claim_boundary": SIDECAR_CLAIM_BOUNDARY,
    }


def _species_specs() -> tuple[dict[str, Any], ...]:
    return (
        dict(
            id="accelerator_candidate_qualification",
            title="Accelerator candidate qualification",
            description=(
                "Qualify a concrete model/kernel candidate through the existing "
                "physical qualification funnel. Proposes WorkUnits; does not "
                "acquire a GPU lease or write a protected measurement."
            ),
            evidence_parents=(
                QUAL_REL,
                "an existing HCLI protected lease",
                "machine quiescence",
            ),
            bounded_authority=(
                "read_receipts",
                "copy_live_workunit_fields",
                "propose_workunit",
                "audit_dependency_chain",
                "write_sidecar_receipt",
            ),
            resource_class="GPU_EXCLUSIVE",
            verifier="accelerator.physical.<candidate_id>",
            budget=_budget(gpu_windows_requested=1),
            stop_condition=(
                "stop when the unit's own verifier passes, the repair budget is "
                "exhausted, or the candidate remains BLOCKED because an evidence "
                "parent is not VERIFIED. DIAGNOSTIC_RELATIVE remains non-authoritative; "
                "this species cannot raise evidence class."
            ),
            role="accelerator_physical_qualification",
            preferred_backend="metal",
            era="III",
        ),
        dict(
            id="architecture_transfer",
            title="Architecture transfer",
            description=(
                "Run a compiled architecture-repatriation experiment spec as an "
                "HCLI WorkUnit proposal. Atlas and repatriation queue stay the "
                "source of identity."
            ),
            evidence_parents=(
                REPAT_REL,
                "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
            ),
            bounded_authority=(
                "read_receipts",
                "copy_live_workunit_fields",
                "compile_experiment_spec",
                "propose_workunit",
                "write_sidecar_receipt",
            ),
            resource_class="GPU_EXCLUSIVE",
            verifier="accelerator.repatriation.<experiment_id>",
            budget=_budget(gpu_windows_requested=1),
            stop_condition=(
                "stop when the spec verifier passes or the spec stays BLOCKED. "
                "A passing diagnostic does not move the id beyond the queue."
            ),
            role="accelerator_repatriation",
            era="III",
        ),
        dict(
            id="odyssey_ii_transfer_experiment",
            title="Odyssey II transfer experiment",
            description=(
                "Seed and query a scoped law store from existing transfer receipts. "
                "Refuses unevidenced promotion between MODEL_LOCAL and GENERIC_VERIFIED."
            ),
            evidence_parents=(
                "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",
                "receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json",
                "receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json",
            ),
            bounded_authority=(
                "read_receipts",
                "seed_law_store",
                "transfer_law_within_declared_scope",
                "emit_static_plan",
                "write_sidecar_receipt",
            ),
            resource_class="STATIC_ANALYSIS",
            verifier="future.odyssey_ii.law_scope",
            budget=_budget(),
            stop_condition=(
                "stop when the law is stored with an explicit scope level, or when "
                "the parent receipts are absent. Scope is never widened by this unit."
            ),
            role="science",
            effect_class="READ_ONLY",
            odyssey="II",
            era="III",
        ),
        dict(
            id="odyssey_iii_adversarial_experiment",
            title="Odyssey III adversarial experiment",
            description=(
                "Attack a claimed law or receipt with the existing adversarial "
                "tools. A hit is a scar, not a promotion of the attacker."
            ),
            evidence_parents=(
                "tools/headless/adversarial_sweep.py",
                "tools/headless/frontier_adversary.py",
                "tools/headless/negative_science.py",
            ),
            bounded_authority=(
                "read_receipts",
                "query_negative_index",
                "adversarially_attack_a_claimed_law",
                "run_static_analysis",
                "write_sidecar_receipt",
            ),
            resource_class="STATIC_ANALYSIS",
            verifier="future.odyssey_iii.adversary",
            budget=_budget(),
            stop_condition=(
                "stop when the attack receipt is sealed (hit or miss). The attacker "
                "does not rewrite the claimed law or its verifier."
            ),
            role="science",
            effect_class="READ_ONLY",
            odyssey="III",
            era="III",
        ),
        dict(
            id="fpga_simulation",
            title="FPGA simulation (Accelerator / Physical Compiler / Fusion)",
            description=(
                "CPU/COMPILE simulation of FPGA-relevant atlas behaviors. FPGA is "
                "part of Accelerator / Physical Compiler / Fusion — not its own "
                "civilization, and this species is not an FPGA backend."
            ),
            evidence_parents=(
                "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
                "hcli/agentos/fpga_preboard.py",
            ),
            bounded_authority=(
                "read_receipts",
                "simulate_fpga_without_board",
                "run_cpu_simulation",
                "emit_static_plan",
                "write_sidecar_receipt",
            ),
            resource_class="COMPILE",
            verifier="accelerator.repatriation.flash-semantic-transport-hwir",
            budget=_budget(),
            stop_condition=(
                "stop when the simulation receipt is sealed. Never claims a physical "
                "FPGA measurement and never builds an FPGA backend."
            ),
            role="accelerator_repatriation",
            preferred_backend="fpga",
            effect_class="REVERSIBLE",
            era="III",
        ),
        dict(
            id="hardware_doctor_experiment",
            title="Hardware Doctor experiment",
            description=(
                "Rank falsifiable hardware-axis experiments from atlas hypotheses. "
                "Doctor for representation already exists; this species does not "
                "fork it, and it does not run a GPU."
            ),
            evidence_parents=(
                "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
                "F003 HWIR (currently MISSING)",
            ),
            bounded_authority=(
                "read_receipts",
                "rank_falsifiable_experiments",
                "emit_static_plan",
                "write_sidecar_receipt",
            ),
            resource_class="STATIC_ANALYSIS",
            verifier="future.hardware_doctor.rank",
            budget=_budget(),
            stop_condition=(
                "stop when a ranked experiment list is sealed against atlas ids, or "
                "when HWIR is still absent. Ranking is not execution."
            ),
            role="science",
            effect_class="READ_ONLY",
            era="III",
        ),
        dict(
            id="learned_compiler_experiment",
            title="Learned compiler experiment",
            description=(
                "Data-contract work for a learned physical compiler. Data contracts "
                "precede any ML; this species does not train a model."
            ),
            evidence_parents=(
                "F011 contamination metadata as a required field",
                "tools/accelerator/perf_model.py (hand-written cost model, not a dataset)",
            ),
            bounded_authority=(
                "read_receipts",
                "emit_static_plan",
                "write_sidecar_receipt",
            ),
            resource_class="STATIC_ANALYSIS",
            verifier="future.lpc.dataset_contract",
            budget=_budget(),
            stop_condition=(
                "stop when the dataset contract receipt is sealed with contamination "
                "fields required. Does not fit weights and does not claim speedup."
            ),
            role="science",
            effect_class="READ_ONLY",
            era="III",
        ),
        dict(
            id="fusion_simulation",
            title="Fusion simulation",
            description=(
                "Simulate fusion-planner / fusion-ISA graphs on CPU. Not a protected "
                "complete-token measurement and not an FPGA backend."
            ),
            evidence_parents=(
                "tools/accelerator/fusion_planner.py",
                "tools/accelerator/fusion_isa.py",
                "tools/accelerator/fusion_wire.py",
            ),
            bounded_authority=(
                "read_receipts",
                "simulate_fusion_graph",
                "run_cpu_simulation",
                "emit_static_plan",
                "write_sidecar_receipt",
            ),
            resource_class="COMPILE",
            verifier="future.fusion.simulate",
            budget=_budget(),
            stop_condition=(
                "stop when the simulation graph receipt is sealed. Graph shape is "
                "not a timing result."
            ),
            role="science",
            effect_class="REVERSIBLE",
            era="III",
        ),
        dict(
            id="independent_reproduction",
            title="Independent reproduction",
            description=(
                "Reproduce a sealed receipt with a replica and optional fault "
                "injection. Autonomy cannot launder weak evidence through this species."
            ),
            evidence_parents=(
                "a sealed source receipt",
                "F008 provenance graph / replication bundle (currently MISSING)",
            ),
            bounded_authority=(
                "read_receipts",
                "inject_faults_in_replica",
                "run_static_analysis",
                "write_sidecar_receipt",
            ),
            resource_class="TEST",
            verifier="future.repro.bundle",
            budget=_budget(),
            stop_condition=(
                "stop when the replica either matches the sealed source bytes/claims "
                "or records a mismatch. The replica does not become the source."
            ),
            role="science",
            effect_class="READ_ONLY",
            era="III",
        ),
        dict(
            id="green_machine_measurement",
            title="Green Machine measurement",
            description=(
                "Energy-accounting contract. Meter-requiring fields stay UNKNOWN "
                "until a protected measurement exists. An honest UNKNOWN is the "
                "correct answer; an invented joule figure is a campaign failure."
            ),
            evidence_parents=("none for the contract; a real power meter for the numbers",),
            bounded_authority=(
                "record_unknown_metrics",
                "emit_static_plan",
                "write_sidecar_receipt",
            ),
            resource_class="STATIC_ANALYSIS",
            verifier="future.green_machine.contract",
            budget=_budget(),
            stop_condition=(
                "stop when the energy-accounting contract is sealed. Every "
                "meter-requiring field remains UNKNOWN."
            ),
            role="science",
            effect_class="READ_ONLY",
            era="III",
        ),
    )


def catalog() -> list[dict[str, Any]]:
    """The ten species, each passed through the authority constructor."""
    return [define_species(**spec) for spec in _species_specs()]


def emit_hcli_workunit(
    *,
    id: str,
    role: str,
    description: str,
    dependencies: Sequence[str],
    resource_class: str,
    verifier: str,
    provider: str,
    effect_class: str = "REVERSIBLE",
    preferred_backend: str | None = None,
    status: str = "pending",
    classification: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit one unit through the real HCLI WorkUnit constructor, then overlay extras."""
    unit = WorkUnit(
        id=id,
        role=role,
        description=description,
        dependencies=list(dependencies),
        resource_class=resource_class,
        preferred_backend=preferred_backend,
        provider=provider,
        verifier=verifier,
        effect_class=effect_class,
        workspace="repo-root",
        classification=classification,
        status="pending",
        repair_depth=0,
    )
    row = unit.to_dict()
    overlay: dict[str, Any] = {
        "status": status,
        "claim_boundary": PROPOSAL_CLAIM_BOUNDARY,
        "requires_quiescence": normalize_resource_class(resource_class) == "GPU_EXCLUSIVE",
    }
    if extras:
        overlay.update({k: v for k, v in extras.items() if v is not None or k in {"blocked_reason", "classification"}})
    row.update(overlay)
    if classification is not None:
        row["classification"] = classification
    row["status"] = status
    return row


def validate_emitted_unit(row: Mapping[str, Any]) -> None:
    """Refuse a unit that is missing the recovered HCLI field set."""
    missing = [name for name in HCLI_CORE_FIELDS if name not in row]
    if missing:
        raise WorkUnitShapeError(f"{row.get('id')}: missing HCLI fields {missing}")
    if not row.get("claim_boundary"):
        raise WorkUnitShapeError(f"{row.get('id')}: claim_boundary is required")
    if not row.get("verifier"):
        raise WorkUnitShapeError(f"{row.get('id')}: verifier is required")
    if not row.get("id"):
        raise WorkUnitShapeError("work unit id is required")
    if row.get("effect_class") not in ALLOWED_EFFECT:
        raise WorkUnitShapeError(f"{row.get('id')}: effect_class {row.get('effect_class')!r} is not HCLI-safe")
    if row.get("may_promote") or row.get("may_modify_verifier"):
        raise WorkUnitShapeError(f"{row.get('id')}: unit expressed a forbidden authority flag")
    # Round-trip the HCLI core so the scheduler can consume the unit.
    WorkUnit.from_dict(dict(row))


def _adopt_live_unit(row: Mapping[str, Any], species_id: str) -> dict[str, Any]:
    wu = WorkUnit.from_dict(dict(row))
    out = wu.to_dict()
    for key, value in row.items():
        if key == "content_hash":
            continue
        if key not in out or key in OBSERVED_EXTRAS or out.get(key) in (None, "", []):
            out[key] = value
        elif key in {"status", "classification", "provider", "role"}:
            out[key] = value
    out["species"] = species_id
    out.setdefault("claim_boundary", PROPOSAL_CLAIM_BOUNDARY)
    if "requires_quiescence" not in out:
        out["requires_quiescence"] = out.get("resource_class") == "GPU_EXCLUSIVE"
    validate_emitted_unit(out)
    return out


def _physical_unit(candidate: Mapping[str, Any], *, ready_ids: set[str]) -> dict[str, Any]:
    cid = str(candidate["candidate_id"])
    status = str(candidate.get("status") or "")
    raw_deps = [str(dep) for dep in (candidate.get("dependencies") or [])]
    if status in {"READY_PROTECTED", "READY_DIAGNOSTIC"}:
        # Mirror the live compiler: only READY deps are edges that can fire.
        deps = [f"accelerator.physical.{dep}" for dep in raw_deps if dep in ready_ids]
    else:
        deps = [f"accelerator.physical.{dep}" for dep in raw_deps]
    if status in {"READY_PROTECTED", "READY_DIAGNOSTIC"}:
        unit_status = "pending"
        classification = None
    elif status == "BLOCKED":
        unit_status = "blocked"
        classification = "BLOCKED"
    else:
        # STATIC_ONLY and any other non-ready rung: visible, not dispatchable.
        unit_status = "blocked"
        classification = status or "STATIC_ONLY"
    extras: dict[str, Any] = {
        "candidate_id": cid,
        "model": candidate.get("model"),
        "blocked_reason": candidate.get("blocked_reason")
        or (
            None
            if unit_status == "pending"
            else f"{status}: not admitted to a runnable qualification rung"
        ),
        "species": "accelerator_candidate_qualification",
        "candidate_status": status,
        "source_receipt": QUAL_REL,
    }
    if candidate.get("diagnostic_command"):
        extras["diagnostic_command"] = list(candidate["diagnostic_command"])
    if candidate.get("protected_command"):
        extras["protected_command"] = list(candidate["protected_command"])
    row = emit_hcli_workunit(
        id=f"accelerator.physical.{cid}",
        role="accelerator_physical_qualification",
        description=(
            f"Qualify {cid} on {candidate.get('model')}; "
            f"region={candidate.get('affected_physical_region')}; "
            f"falsifier={candidate.get('parity_contract')}"
        ),
        dependencies=deps,
        resource_class="GPU_EXCLUSIVE",
        verifier=f"accelerator.physical.{cid}",
        provider="accelerator_physical_queue",
        preferred_backend=str(candidate.get("preferred_backend") or "metal"),
        status=unit_status,
        classification=classification,
        extras=extras,
    )
    validate_emitted_unit(row)
    return row


def _repatriation_species(spec: Mapping[str, Any]) -> str:
    backend = str(spec.get("backend") or spec.get("preferred_backend") or "").lower()
    if backend == "fpga" or str(spec.get("experiment_id") or "").endswith("-hwir"):
        return "fpga_simulation"
    if backend == "ane":
        return "architecture_transfer"
    return "architecture_transfer"


def _repatriation_unit(spec: Mapping[str, Any]) -> dict[str, Any]:
    eid = str(spec["experiment_id"])
    backend = str(spec.get("backend") or "metal")
    blocked = str(spec.get("status") or "").upper() == "BLOCKED"
    extras = {
        "experiment_id": eid,
        "behavior_id": spec.get("behavior_id"),
        "command": list(spec.get("command") or []),
        "output_receipt_path": spec.get("output_receipt_path"),
        "requires_quiescence": bool(spec.get("requires_quiescence")),
        "blocked_reason": spec.get("blocked_reason"),
        "species": _repatriation_species(spec),
        "source_receipt": REPAT_REL,
    }
    row = emit_hcli_workunit(
        id=f"accelerator.{eid}",
        role="accelerator_repatriation",
        description=(
            f"Run {spec.get('candidate')} on {spec.get('model_identity')}/{backend} "
            f"for {spec.get('organ')}. Control: {spec.get('control')}. "
            f"Falsifier: {spec.get('falsifier')}"
        ),
        dependencies=[],
        resource_class=REPAT_RESOURCE.get(backend, "LIGHT_CONTROL"),
        verifier=f"accelerator.repatriation.{eid}",
        provider="accelerator_runner",
        preferred_backend=backend,
        status="blocked" if blocked else "pending",
        classification="BLOCKED" if blocked else None,
        extras=extras,
    )
    validate_emitted_unit(row)
    return row


def _planning_units() -> list[dict[str, Any]]:
    """Real disk-referenced work for species that are not the live GPU queues."""
    plans = (
        {
            "id": "future.odyssey-ii.seed-law-store",
            "species": "odyssey_ii_transfer_experiment",
            "role": "science",
            "description": (
                "Seed a scoped Odyssey II law store from ODYSSEY_TRANSFER_PROVEN, "
                "ACCELERATOR_TRANSFER_VERIFIED and QWEN38_ACCELERATOR_TRANSFER_MAP. "
                "Refuse unevidenced MODEL_LOCAL -> GENERIC_VERIFIED promotion."
            ),
            "resource_class": "STATIC_ANALYSIS",
            "verifier": "future.odyssey_ii.law_scope",
            "effect_class": "READ_ONLY",
            "evidence_parents": [
                "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",
                "receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json",
                "receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json",
            ],
        },
        {
            "id": "future.odyssey-iii.attack-claimed-law",
            "species": "odyssey_iii_adversarial_experiment",
            "role": "science",
            "description": (
                "Adversarially attack a claimed transfer law using the existing "
                "headless adversary and negative-science corpus. A hit is a scar."
            ),
            "resource_class": "STATIC_ANALYSIS",
            "verifier": "future.odyssey_iii.adversary",
            "effect_class": "READ_ONLY",
            "evidence_parents": [
                "tools/headless/adversarial_sweep.py",
                "tools/headless/negative_science.py",
            ],
        },
        {
            "id": "future.hardware-doctor.rank-atlas-hypotheses",
            "species": "hardware_doctor_experiment",
            "role": "science",
            "description": (
                "Rank atlas hwir_hypotheses into falsifiable Hardware Doctor "
                "experiments. Does not invent an FPGA backend and does not run a GPU."
            ),
            "resource_class": "STATIC_ANALYSIS",
            "verifier": "future.hardware_doctor.rank",
            "effect_class": "READ_ONLY",
            "evidence_parents": ["receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"],
        },
        {
            "id": "future.lpc.dataset-contract",
            "species": "learned_compiler_experiment",
            "role": "science",
            "description": (
                "Write the learned-physical-compiler dataset contract with "
                "contamination metadata as a required field. No training."
            ),
            "resource_class": "STATIC_ANALYSIS",
            "verifier": "future.lpc.dataset_contract",
            "effect_class": "READ_ONLY",
            "evidence_parents": ["tools/accelerator/perf_model.py"],
        },
        {
            "id": "future.fusion.simulate-isa-graph",
            "species": "fusion_simulation",
            "role": "science",
            "description": (
                "CPU simulation of fusion_planner / fusion_isa graphs. Graph shape "
                "only; no complete-token measurement."
            ),
            "resource_class": "COMPILE",
            "verifier": "future.fusion.simulate",
            "effect_class": "REVERSIBLE",
            "evidence_parents": [
                "tools/accelerator/fusion_planner.py",
                "tools/accelerator/fusion_isa.py",
            ],
        },
        {
            "id": "future.repro.replication-bundle",
            "species": "independent_reproduction",
            "role": "science",
            "description": (
                "Independent reproduction bundle for a sealed Codex receipt, with "
                "fault injection. The replica does not become the source."
            ),
            "resource_class": "TEST",
            "verifier": "future.repro.bundle",
            "effect_class": "READ_ONLY",
            "evidence_parents": ["receipts/headless/ACCELERATOR_SCOREBOARD.json"],
        },
        {
            "id": "future.green-machine.energy-contract",
            "species": "green_machine_measurement",
            "role": "science",
            "description": (
                "Seal the Green Machine energy-accounting contract. joules_per_token "
                "and related meter fields remain UNKNOWN."
            ),
            "resource_class": "STATIC_ANALYSIS",
            "verifier": "future.green_machine.contract",
            "effect_class": "READ_ONLY",
            "evidence_parents": [],
        },
    )
    units = []
    for plan in plans:
        extras = {
            "species": plan["species"],
            "evidence_parents": plan["evidence_parents"],
            "claim_boundary": SIDECAR_CLAIM_BOUNDARY,
            "requires_quiescence": False,
            "candidate_status": "STATIC_ONLY",
        }
        row = emit_hcli_workunit(
            id=plan["id"],
            role=plan["role"],
            description=plan["description"],
            dependencies=[],
            resource_class=plan["resource_class"],
            verifier=plan["verifier"],
            provider="future.workunit_species",
            effect_class=plan["effect_class"],
            status="pending",
            classification="STATIC_ONLY",
            extras=extras,
        )
        validate_emitted_unit(row)
        units.append(row)
    return units


def _candidates_from_qual(doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    if doc and isinstance(doc.get("candidates"), list) and doc["candidates"]:
        return list(doc["candidates"])
    return [dict(item) for item in RECOVERED_PHYSICAL_CANDIDATES]


def _specs_from_repat(doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    if doc and isinstance(doc.get("specs"), list) and doc["specs"]:
        rows = []
        for spec in doc["specs"]:
            item = dict(spec)
            blocked = (item.get("state_session_inputs") or {}).get("blocked_reason")
            item.setdefault("blocked_reason", blocked)
            runner = item.get("runner") or {}
            item.setdefault("requires_quiescence", bool(runner.get("requires_quiescence")))
            rows.append(item)
        return rows
    return [dict(item) for item in RECOVERED_REPATRIATION_SPECS]


def build_starting_queue(
    *,
    qual: dict[str, Any] | None = None,
    repat: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Non-empty queue of real disk-backed work, in HCLI work_units shape."""
    if qual is None:
        qual, _ = load_headless(QUAL_REL)
    if repat is None:
        repat, _ = load_headless(REPAT_REL)
    units: list[dict[str, Any]] = []
    seen: set[str] = set()

    live_wu = list((qual or {}).get("work_units") or [])
    for raw in live_wu:
        row = _adopt_live_unit(raw, "accelerator_candidate_qualification")
        units.append(row)
        seen.add(row["id"])

    candidates = _candidates_from_qual(qual)
    ready_ids = {
        str(c["candidate_id"])
        for c in candidates
        if str(c.get("status") or "") in {"READY_PROTECTED", "READY_DIAGNOSTIC"}
    }
    for cand in sorted(candidates, key=lambda c: str(c.get("candidate_id") or "")):
        uid = f"accelerator.physical.{cand['candidate_id']}"
        if uid in seen:
            # Overlay candidate_status / blocked_reason onto the adopted live unit.
            for row in units:
                if row["id"] == uid:
                    row.setdefault("candidate_status", cand.get("status"))
                    row.setdefault("blocked_reason", cand.get("blocked_reason"))
                    row.setdefault("species", "accelerator_candidate_qualification")
            continue
        units.append(_physical_unit(cand, ready_ids=ready_ids))
        seen.add(uid)

    live_repat = list((repat or {}).get("work_units") or [])
    for raw in live_repat:
        species = _repatriation_species(raw)
        row = _adopt_live_unit(raw, species)
        units.append(row)
        seen.add(row["id"])

    for spec in sorted(_specs_from_repat(repat), key=lambda s: str(s.get("experiment_id") or "")):
        uid = f"accelerator.{spec['experiment_id']}"
        if uid in seen:
            for row in units:
                if row["id"] == uid:
                    row.setdefault("species", _repatriation_species(spec))
                    row.setdefault("blocked_reason", spec.get("blocked_reason"))
            continue
        units.append(_repatriation_unit(spec))
        seen.add(uid)

    for plan in _planning_units():
        if plan["id"] not in seen:
            units.append(plan)
            seen.add(plan["id"])

    units.sort(key=lambda row: str(row["id"]))
    if not units:
        raise WorkUnitShapeError("starting queue is empty; the resident would wake into a question")
    for row in units:
        validate_emitted_unit(row)
    return units


def queue_identity_sets(units: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    candidate_ids = sorted(
        {
            str(row["candidate_id"])
            for row in units
            if row.get("candidate_id")
        }
    )
    experiment_ids = sorted(
        {
            str(row["experiment_id"])
            for row in units
            if row.get("experiment_id")
        }
    )
    return {
        "candidate_ids": candidate_ids,
        "experiment_ids": experiment_ids,
        "ids": [str(row["id"]) for row in units],
        "species": sorted({str(row.get("species")) for row in units if row.get("species")}),
    }


def _prove_authority_refusal() -> list[dict[str, str]]:
    """Watch the constructor actually refuse. A guard nobody has seen fail is not a guard."""
    trials = (
        ("self_promotion", {"bounded_authority": ("self_promotion", "read_receipts")}),
        ("weaken_verifier", {"bounded_authority": ("weaken_verifier", "read_receipts")}),
        ("modify_verifier_flag", {"may_modify_verifier": True}),
        ("may_promote_flag", {"may_promote": True}),
        ("choose_singularity", {"may_choose_singularity": True}),
        ("destructive_mutation", {"may_destructive_mutate": True}),
    )
    base = dict(_species_specs()[0])
    results = []
    for name, patch in trials:
        kwargs = dict(base)
        kwargs.update(patch)
        try:
            define_species(**kwargs)
        except SpeciesAuthorityError as exc:
            results.append({"trial": name, "refused": True, "error": str(exc)})
            continue
        raise SpeciesAuthorityError(f"authority guard did not fire for {name}")
    return results


def build() -> Path:
    species = catalog()
    qual, qual_src = load_headless(QUAL_REL)
    repat, repat_src = load_headless(REPAT_REL)
    units = build_starting_queue(qual=qual, repat=repat)
    identities = queue_identity_sets(units)
    refusals = _prove_authority_refusal()

    ready_qwen = [
        row
        for row in units
        if row.get("candidate_id") in set(CORE_READY_QWEN27)
        or str(row.get("candidate_status") or "") == "READY_PROTECTED"
        and str(row.get("model") or "") == "Qwen27"
    ]
    blocked_flash = [
        row
        for row in units
        if str(row.get("model") or "") == "Flash"
        and (
            str(row.get("candidate_status") or row.get("classification") or "") == "BLOCKED"
            or row.get("status") == "blocked"
        )
        and row.get("candidate_id")
    ]

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "WorkUnit species for the future HCLI resident: types with bounded "
            "authority, plus a non-empty starting queue recovered from live disk."
        ),
        "species": species,
        "species_ids": [item["id"] for item in species],
        "work_units": units,
        "starting_queue": {
            "count": len(units),
            "non_empty": True,
            "by_species": {
                sid: sum(1 for row in units if row.get("species") == sid)
                for sid in SPECIES_IDS
            },
            "by_status": _count_by(units, "status"),
            "ready_protected_qwen27": sorted(
                {
                    str(row["candidate_id"])
                    for row in units
                    if row.get("candidate_id")
                    and str(row.get("candidate_status") or "") == "READY_PROTECTED"
                    and str(row.get("model") or "") == "Qwen27"
                }
            ),
            "blocked_flash": sorted(
                {
                    str(row["candidate_id"])
                    for row in blocked_flash
                    if row.get("candidate_id")
                }
            ),
            "repatriation_experiment_ids": identities["experiment_ids"],
            "identities": identities,
        },
        "live_sources": {
            "qualification_queue": {"path": QUAL_REL, "loaded_from": qual_src, "present": qual is not None},
            "repatriation_queue": {"path": REPAT_REL, "loaded_from": repat_src, "present": repat is not None},
        },
        "hcli_field_set": {
            "core": list(HCLI_CORE_FIELDS),
            "observed_extras": list(OBSERVED_EXTRAS),
            "union": sorted(set(HCLI_CORE_FIELDS) | set(OBSERVED_EXTRAS)),
        },
        "authority": {
            "allowed": sorted(ALLOWED_AUTHORITY),
            "forbidden": sorted(FORBIDDEN_AUTHORITY),
            "refusals_proven": refusals,
        },
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is": "part of Accelerator / Physical Compiler / Fusion; not a civilization",
        },
        "recovered_implementation": {
            "hcli.workunit.WorkUnit": "hcli/workunit.py — canonical unit, content_hash, repair budget",
            "hcli.scheduler.Scheduler": "hcli/scheduler.py — dispatch only; does not invent work",
            "hcli.dag_store.DagStore": "hcli/dag_store.py — disk is authority",
            "hcli.ledger.Ledger": "hcli/ledger.py — obligation VERIFIED only via run_verify",
            "hcli.agentos.states.AgentState": "hcli/agentos/states.py (git show; not materialized here)",
            "hcli.agentos.autonomy_gate": "bounded census WorkUnits with a fixed verifier the model cannot nominate",
            "hcli.agentos.runtime.AgentOS": "composition facade; Mission/Scheduler remain authorities",
            "tools.accelerator.physical_qualification.workunits_for_candidates": (
                "emits READY physical candidates through WorkUnit.to_dict plus extras "
                "(candidate_id, model, diagnostic_command, protected_command, claim_boundary)"
            ),
            "tools.accelerator.accelerator_runner.workunits_for_specs": (
                "emits repatriation specs through WorkUnit.to_dict plus extras "
                "(experiment_id, behavior_id, command, output_receipt_path, requires_quiescence)"
            ),
            "live_qualification_queue": QUAL_REL,
            "live_repatriation_queue": REPAT_REL,
            "note": (
                "No WorkUnit species catalog existed. The live compilers already emit "
                "HCLI-shaped work_units for READY rows only. This module reuses that "
                "field set and the HCLI constructor; it does not fork a second scheduler."
            ),
        },
        "gaps_closed": [
            "ten species with evidence parents, bounded authority, resource class, verifier, budget, stop condition",
            "constructor refusal of self-promotion / verifier-modification / Singularity / destructive mutation",
            "starting queue includes BLOCKED Flash candidates and their blockers (live compiler skips non-READY)",
            "starting queue includes blocked repatriation specs (live compiler include_blocked=False by default)",
            "every species has at least one disk-referenced unit so the resident does not wake into an empty queue",
        ],
        "negative_findings": [
            f"{QUAL_REL} is not in git HEAD of this worktree; loaded from {qual_src}",
            f"{REPAT_REL} is not in git HEAD of this worktree; loaded from {repat_src}",
            "hcli/agentos/* is not materialized in this sparse checkout; recovered via git show HEAD:hcli/agentos/...",
            "sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE; bench.state stays UNKNOWN",
            "no GPU / FPGA / power meter in this lane; Green Machine numbers are UNKNOWN",
        ],
        "counts": {
            "species": len(species),
            "work_units": len(units),
            "ready_qwen27_units": len(ready_qwen),
            "blocked_flash_units": len(blocked_flash),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/workunit_species.py")


def selftest() -> Path:
    _prove_authority_refusal()
    species = catalog()
    if len(species) != 10:
        raise AssertionError(f"expected 10 species, got {len(species)}")
    units = build_starting_queue()
    if not units:
        raise AssertionError("starting queue is empty")
    ids = queue_identity_sets(units)
    missing_ready = [cid for cid in CORE_READY_QWEN27 if cid not in ids["candidate_ids"]]
    missing_blocked = [cid for cid in CORE_BLOCKED_FLASH if cid not in ids["candidate_ids"]]
    if missing_ready or missing_blocked:
        raise AssertionError(
            f"queue missing live ids ready={missing_ready} blocked={missing_blocked}"
        )
    return build()


def _count_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        name = str(row.get(key) or "null")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(selftest())
        return 0
    print(build())
    return 0


# Recovered 2026-08-29 from the live headless queues on the primary checkout.
# Used only when those receipts are not materialized in this sparse worktree.
# Structural identity only — no hardware numbers.
RECOVERED_PHYSICAL_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"candidate_id": "qwen27-affine2-splitk4", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 HGRAVF01 affine Q2 GEMV", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-attention-gate-fusion", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 GQA attention output sigmoid gate", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-ba-delta-fusion", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 DeltaNet BA/state transition", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-commit-timing-elision", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 uninstrumented TokenCommandBuffer commit/fence", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-deltanet-inproj-fusion", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 DeltaNet input projection", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-encoder-label-elision", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 shared Metal ordinary encoder labeling", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-fast-profile", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": [], "blocked_reason": None, "affected_physical_region": "complete Qwen27 resident token", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-gqa-qkv-fusion", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 GQA Q/K/V packed projection", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-pipeline-cache-reuse", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 resident per-token Metal pipeline lookup", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-pipeline-id-resolution", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-pipeline-cache-reuse"], "blocked_reason": None, "affected_physical_region": "Qwen27 resident per-dispatch pipeline handle resolution", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-pipeline-state-elision", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 shared Metal ordered/concurrent encoder state binding", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-q2f-splitk4", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 biasless Q2F projection family", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-q4-vecgroup-x64", "model": "Qwen27", "status": "READY_PROTECTED", "dependencies": ["qwen27-fast-profile"], "blocked_reason": None, "affected_physical_region": "Qwen27 standalone uniform-Q4 GEMV", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "flash-attention-gate-fusion", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "Flash complete-token and source-independent capability gates remain open", "affected_physical_region": "Flash attention output sigmoid gate", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-compact-moe-bf16-vec4", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "source-independent Flash NX and protected complete-token capability are not qualified", "affected_physical_region": "Flash compact source-BF16 routed/shared MoE epilogue", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-compact-moe-epilogue", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "Flash source-independent compact expert consumer is not yet qualified", "affected_physical_region": "Flash routed/shared expert output epilogue", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-encoder-label-elision", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "source-independent Flash NX and protected complete-token capability are not qualified", "affected_physical_region": "Flash native/fullseq ordinary Metal encoder labeling", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-fullseq-catalog-cache", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "source-independent Flash NX and protected complete-token capability are not qualified", "affected_physical_region": "Flash fullseq source-anchor admission between positions", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-fullseq-ordered-encoder", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "fullseq source path is not a complete source-independent Flash NX executable", "affected_physical_region": "Flash fullseq attention command-encoder topology", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-hc-router-topk-fusion", "model": "Flash", "status": "STATIC_ONLY", "dependencies": ["flash-hc-staged-threadgroup", "flash-router-topk-fusion"], "blocked_reason": "source-independent Flash NX and protected capability are not qualified", "affected_physical_region": "Flash HC output, block projection, router, shared scalar, top-K", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-hc-staged-threadgroup", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "Flash complete source-independent NX executable is not qualified", "affected_physical_region": "Flash HyperConnection shared reduction", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-meta-sub1-coherent", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "meta budget is only a prospective function description; no serialized functional artifact, source-independent Flash NX consumer, or protected complete-token path exists", "affected_physical_region": "Flash whole-model functional representation (routed experts + n-gram bank)", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-pipeline-cache-reuse", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "source-independent Flash NX and protected complete-token capability are not qualified", "affected_physical_region": "Flash native/fullseq per-batch Metal pipeline lookup", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-pipeline-id-resolution", "model": "Flash", "status": "BLOCKED", "dependencies": ["flash-pipeline-cache-reuse"], "blocked_reason": "source-independent Flash NX and protected complete-token capability are not qualified", "affected_physical_region": "Flash native/fullseq per-dispatch pipeline handle resolution", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-qkv-gqa-rope-fusion", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "source-independent Flash NX and protected complete-token capability are not qualified", "affected_physical_region": "Flash GQA Q/K/V plus RoPE", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-routed-fp4-gate-up-swiglu-fused", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "Flash source-independent NX and protected complete-token capability are not qualified", "affected_physical_region": "Flash routed FP4 gate/up/SwiGLU epilogue", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-router-topk-fusion", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "source-independent Flash route consumer and protected capability are not qualified", "affected_physical_region": "Flash router + top-K", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-shared-fp8-gate-up-swiglu-fused", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "Flash source-independent NX and protected complete-token capability are not qualified", "affected_physical_region": "Flash shared FP8 gate/up/SwiGLU epilogue", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "flash-source-bf16-simd", "model": "Flash", "status": "BLOCKED", "dependencies": [], "blocked_reason": "source oracle is a control; Flash NX and protected complete-token path are open", "affected_physical_region": "Flash source-BF16 projection oracle", "parity_contract": "source organ/reference parity is required; no whole-model Flash parity or NX claim is implied", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-affine2-splitk4-vec", "model": "Qwen27", "status": "STATIC_ONLY", "dependencies": ["qwen27-affine2-splitk4"], "blocked_reason": None, "affected_physical_region": "Qwen27 HGRAVF01 affine Q2 GEMV", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-q2f-splitk4-vec", "model": "Qwen27", "status": "STATIC_ONLY", "dependencies": ["qwen27-q2f-splitk4"], "blocked_reason": None, "affected_physical_region": "Qwen27 biasless Q2F projection family", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
    {"candidate_id": "qwen27-resident-untimed-decode", "model": "Qwen27", "status": "STATIC_ONLY", "dependencies": ["qwen27-commit-timing-elision"], "blocked_reason": None, "affected_physical_region": "Genesis resident Qwen27 serving token loop", "parity_contract": "identical tokenizer-bound output ids and source-approved graph semantics; any numerical/output divergence rejects the candidate", "preferred_backend": "metal"},
)

RECOVERED_REPATRIATION_SPECS: tuple[dict[str, Any], ...] = (
    {"experiment_id": "qwen27-move-or-recompute-boundary", "behavior_id": "move_or_recompute", "status": "READY", "backend": "metal", "model_identity": "Qwen3.8-27B", "organ": "all", "candidate": "costed dependency planner chooses resident/recompute/prefetch by complete boundary cost", "control": "framework-prescribed movement", "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting", "output_receipt_path": "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-move-or-recompute-boundary.json", "blocked_reason": None, "requires_quiescence": True, "command": ["python3", "-m", "hcli", "agentos", "protected-accelerator-bench", "--profile", "hcli/hawking-native.sealed-3.14.json", "--max-new-tokens", "32", "--repo-root", ".", "--emit", "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-move-or-recompute-boundary.json"]},
    {"experiment_id": "qwen27-graph-replay-token-skeleton", "behavior_id": "graph_replay", "status": "READY", "backend": "metal", "model_identity": "Qwen3.8-27B", "organ": "decode", "candidate": "replay static token graph with dynamic token/position slots", "control": "current persistent executor with per-step command encoding", "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting", "output_receipt_path": "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-graph-replay-token-skeleton.json", "blocked_reason": None, "requires_quiescence": True, "command": ["python3", "-m", "hcli", "agentos", "protected-accelerator-bench", "--profile", "hcli/hawking-native.sealed-3.14.json", "--max-new-tokens", "32", "--repo-root", ".", "--emit", "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-graph-replay-token-skeleton.json"]},
    {"experiment_id": "qwen27-layout-algebra-mlp", "behavior_id": "layout_algebra", "status": "READY", "backend": "metal", "model_identity": "Qwen3.8-27B", "organ": "mlp", "candidate": "parameterized packed GEMV layout/tile/lane mapping", "control": "sealed resident Qwen27 GeoTpr64Tg128 path", "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting", "output_receipt_path": "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-layout-algebra-mlp.json", "blocked_reason": None, "requires_quiescence": True, "command": ["python3", "-m", "hcli", "agentos", "protected-accelerator-bench", "--profile", "hcli/hawking-native.sealed-3.14.json", "--max-new-tokens", "32", "--repo-root", ".", "--emit", "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-layout-algebra-mlp.json"]},
    {"experiment_id": "qwen27-stationary-packed-weight", "behavior_id": "stationary_representation", "status": "READY", "backend": "metal", "model_identity": "Qwen3.8-27B", "organ": "mlp", "candidate": "keep packed representation resident and expose runtime active bytes separately", "control": "same-source packed path with no active-byte instrumentation", "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting", "output_receipt_path": "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-stationary-packed-weight.json", "blocked_reason": None, "requires_quiescence": True, "command": ["python3", "-m", "hcli", "agentos", "protected-accelerator-bench", "--profile", "hcli/hawking-native.sealed-3.14.json", "--max-new-tokens", "32", "--repo-root", ".", "--emit", "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-stationary-packed-weight.json"]},
    {"experiment_id": "flash-semantic-transport-hwir", "behavior_id": "semantic_transport", "status": "READY", "backend": "fpga", "model_identity": "Qwen3.8-Flash-Next", "organ": "route_and_state", "candidate": "typed route metadata/activation/partial-reduction edges in HWIR", "control": "untyped partition boundary", "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting", "output_receipt_path": "receipts/headless/ACCELERATOR_REPATRIATION/flash-semantic-transport-hwir.json", "blocked_reason": None, "requires_quiescence": False, "command": ["python3", "-m", "hcli", "agentos", "fpga-preboard", "--repo-root", ".", "--emit", "receipts/headless/ACCELERATOR_REPATRIATION/flash-semantic-transport-hwir.json"]},
    {"experiment_id": "qwen27-async-double-buffer", "behavior_id": "async_double_buffer", "status": "READY", "backend": "metal", "model_identity": "Qwen3.8-27B", "organ": "mlp", "candidate": "overlap next packed tile staging with current projection when ownership permits", "control": "current serial projection and command-buffer boundary", "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting", "output_receipt_path": "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-async-double-buffer.json", "blocked_reason": None, "requires_quiescence": True, "command": ["python3", "-m", "hcli", "agentos", "protected-accelerator-bench", "--profile", "hcli/hawking-native.sealed-3.14.json", "--max-new-tokens", "32", "--repo-root", ".", "--emit", "receipts/headless/ACCELERATOR_REPATRIATION/qwen27-async-double-buffer.json"]},
    {"experiment_id": "flash-local-state-machine", "behavior_id": "local_state_machine", "status": "BLOCKED", "backend": "metal", "model_identity": "Qwen3.8-Flash-Next", "organ": "deltanet", "candidate": "persistent DeltaNet state machine with checkpoint-bisection verifier", "control": "current Flash stateful seam", "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting", "output_receipt_path": "receipts/headless/ACCELERATOR_REPATRIATION/flash-local-state-machine.json", "blocked_reason": "requires the Flash protected complete-token runtime lane", "requires_quiescence": True, "command": ["python3", "-m", "hcli", "agentos", "fpga-preboard", "--repo-root", ".", "--emit", "receipts/headless/ACCELERATOR_REPATRIATION/flash-local-state-machine.json"]},
    {"experiment_id": "flash-direct-routed-accumulate", "behavior_id": "direct_routed_accumulate", "status": "BLOCKED", "backend": "metal", "model_identity": "Qwen3.8-Flash-Next", "organ": "moe", "candidate": "route-before-payload with selected-expert direct weighted accumulation", "control": "current Flash routed-expert component graph", "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting", "output_receipt_path": "receipts/headless/ACCELERATOR_REPATRIATION/flash-direct-routed-accumulate.json", "blocked_reason": "Flash native full-model executable/weights are not available in the current protected lane; retain as detached queue work", "requires_quiescence": True, "command": ["python3", "-m", "hcli", "agentos", "fpga-preboard", "--repo-root", ".", "--emit", "receipts/headless/ACCELERATOR_REPATRIATION/flash-direct-routed-accumulate.json"]},
    {"experiment_id": "ane-regular-island-probe", "behavior_id": "npu_regular_island", "status": "BLOCKED", "backend": "ane", "model_identity": "Qwen3.8-27B", "organ": "normalization", "candidate": "public Core ML/ML Program regular island with explicit transfer accounting", "control": "Metal normalization path", "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting", "output_receipt_path": "receipts/headless/ACCELERATOR_REPATRIATION/ane-regular-island-probe.json", "blocked_reason": "public ANE compile/runtime measurement is not available in this process; plan-only until the public path is executable", "requires_quiescence": True, "command": ["python3", "-m", "hcli", "agentos", "fpga-preboard", "--repo-root", ".", "--emit", "receipts/headless/ACCELERATOR_REPATRIATION/ane-regular-island-probe.json"]},
)


if __name__ == "__main__":
    raise SystemExit(main())
