"""Persist the HCLI resident identity to disk so restart needs no conversation.

A resident whose identity lives only in a prompt cannot restart, cannot be
compared to a child, and cannot be held to a capability claim. This module
recovers the incumbent Qwen27 sealed-3.14 identity from disk/git, fills every
named field, writes UNKNOWN where this sidecar cannot actually know, and
refuses an identity that invents a number or claims zero weaknesses while
real blockers exist.

    python3 tools/future/resident_identity.py --build
    python3 tools/future/resident_identity.py --load
    python3 tools/future/resident_identity.py --selftest

Not a fork of hcli/agentos/qwen27_runtime_identity.py (historical vs current
diff), hcli/agentos/resident_gate.py (lifecycle), hcli/machine.py (genome
prior), or tools/future/resident_install.py (generic winner slots). Those
remain. This sidecar is the restart-survivable identity document.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    git,
    sha256_file,
)

RECEIPT = "RESIDENT_IDENTITY.json"
SCHEMA = "hawking.future.resident_identity.v1"
VERSION = 1
RECORDED_BY = "tools/future/resident_identity.py"
UNKNOWN = "UNKNOWN"
CLAIM_CLASS = "STATIC_ONLY"
RESIDENCY_STATUS = "CURRENT_NONFINAL_HCLI_WORKER"
WORK_UNIT_ID = "future.resident-identity.persist"

# Named identity body. Derived from the contract's field list, not a frozen
# integer — the heading said "nineteen" and then named both children and
# sandbox authority. The tuple is the source of truth.
IDENTITY_FIELDS: tuple[str, ...] = (
    "model_family",
    "nx_id",
    "parent_lineage",
    "tokenizer_identity",
    "executable_hash",
    "ebpw",
    "active_bytes_evidence",
    "capability_receipt",
    "tps_token_ns_evidence",
    "machine_genome",
    "current_backend",
    "current_tools",
    "current_resource_pressure",
    "current_frontiers",
    "active_workunits",
    "blocked_workunits",
    "known_scars",
    "known_weaknesses",
    "children",
    "sandbox_authority",
)

# Slots this host cannot evidence. A numeric fill is invented and REJECTED.
REQUIRED_UNKNOWN_SLOTS: tuple[tuple[str, str], ...] = (
    ("ebpw", "qualified_physical"),
    ("tps_token_ns_evidence", "protected_complete_token"),
)

SEALED_REL = "hcli/hawking-native.sealed-3.14.json"
RUNTIME_IDENTITY_PY_REL = "hcli/agentos/qwen27_runtime_identity.py"
RESIDENT_GATE_REL = "hcli/agentos/resident_gate.py"
MACHINE_PY_REL = "hcli/machine.py"
MACHINE_GENOME_PY_REL = "tools/accelerator/machine_genome.py"
MACHINE_GENOME_REL = "receipts/headless/MACHINE_GENOME.json"
ACCEL_GENOME_REL = "receipts/headless/ACCELERATOR_MACHINE_GENOME.json"
CAPABILITY_REL = "receipts/headless/CAPABILITY_noetic-sealed-3.14.json"
SEAL_REL = "receipts/headless/HCLI_RESIDENT_SEAL.json"
ATLAS_REL = "receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json"
HISTORICAL_REL = "receipts/headless/QWEN27_HISTORICAL_RUNTIME_IDENTITY.json"
HANDOFF_REL = "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json"
FRONTIER_REL = "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"
WORKUNITS_REL = "receipts/future/HCLI_FUTURE_WORKUNITS.json"
CONTAMINATION_REL = "receipts/future/CONTAMINATION_SCIENCE.json"
QUAL_REL = "receipts/future/QUALIFICATION_PIPELINE.json"
FLASH_NX_REL = "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json"
NEGATIVE_REL = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"
PROFILE_SCHEMA_REL = "receipts/future/QWEN27_ACCELERATOR_PROFILE_SCHEMA.json"
SUCCESSION_TRIAL_REL = "receipts/future/SUCCESSION_TRIAL.json"

# Catalog name of the incumbent this sidecar documents. Disk (SUCCESSION_TRIAL
# then the sealed profile) is authority; this constant is the expected id
# those documents currently carry, not a substitute for reading them.
EXPECTED_INCUMBENT_ID = "qwen3.8-27b-sealed-3.14"

# Fields a launch binding must name. Presence of the receipt is not binding.
# bound is true only when every pin is present and sealed_model_id agrees
# with the incumbent that would actually be launched.
BIND_PIN_FIELDS: tuple[str, ...] = (
    "nx_id",
    "sealed_model_id",
    "executable_hash",
    "artifact_root",
    "tokenizer",
    "qualification",
)
TEACHER_RELS = (
    "receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
    "receipts/future/evidence/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
)
CODEX_HANDOFF_REL = "CODEX_ACCELERATOR_HANDOFF.json"
QUEUE_RELS = (
    "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
    "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
)

SINGULARITY_STATUSES = frozenset(
    {
        "SINGULARITY",
        "FINAL",
        "CURRENT_FINAL",
        "CURRENT_SINGULARITY",
        "HAWKING_SINGULARITY",
        "QWEN27_SINGULARITY",
        "FLASH_SINGULARITY",
    }
)

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)


class IdentityRejectedError(ValueError):
    """An identity that invents, omits, or lies about weaknesses is refused."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        named = "; ".join(reasons) if reasons else "<unspecified>"
        super().__init__(f"REJECTED: {named}")


# ---------------------------------------------------------------------------
# IO — disk first, git HEAD second. Missing on disk is not repository absence.
# ---------------------------------------------------------------------------


def load_authority(rel: str) -> tuple[str, dict[str, Any] | None]:
    """Load JSON from disk, else git HEAD. Disk state is authority."""
    path = REPO / rel
    if path.is_file():
        try:
            doc = load_json(path)
        except (OSError, json.JSONDecodeError, UnicodeError):
            return "ON_DISK_UNPARSEABLE", None
        return "ON_DISK", doc if isinstance(doc, dict) else None
    raw = git("show", f"HEAD:{rel}")
    if not raw:
        return "ABSENT", None
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return "GIT_HEAD_UNPARSEABLE", None
    return "GIT_HEAD", doc if isinstance(doc, dict) else None


def presence(rel: str) -> dict[str, Any]:
    on_disk = (REPO / rel).is_file()
    in_git = bool(git("show", f"HEAD:{rel}"))
    return {
        "path": rel,
        "on_disk": on_disk,
        "in_git_head": in_git,
        "source": "ON_DISK" if on_disk else ("GIT_HEAD" if in_git else "ABSENT"),
    }


def _first_authority(rels: tuple[str, ...]) -> tuple[str, str, dict[str, Any] | None]:
    """First rel that loads as a dict. Records which path it took."""
    last_src = "ABSENT"
    last_rel = rels[0] if rels else ""
    for rel in rels:
        src, doc = load_authority(rel)
        last_src, last_rel = src, rel
        if isinstance(doc, dict):
            return rel, src, doc
    return last_rel, last_src, None


def _sha256_path(path: Path) -> str | None:
    try:
        if path.is_file():
            return sha256_file(path)
    except OSError:
        return None
    return None


def _sha256_blob(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _host_file_record(path_s: str | None, *, declared_prefix: str | None = None) -> dict[str, Any]:
    """Hash a host path if it exists. Missing is UNKNOWN, never a fake digest."""
    if not path_s:
        return {
            "path": None,
            "exists": False,
            "sha256": UNKNOWN,
            "missing_evidence": ["no path declared"],
        }
    path = Path(path_s).expanduser()
    exists = path.is_file()
    digest = _sha256_path(path) if exists else None
    rec: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "sha256": digest if digest else UNKNOWN,
        "source_state": "COMPUTED" if digest else ("HOST_ABSENT" if not exists else "UNREADABLE"),
    }
    if digest is None:
        rec["missing_evidence"] = [
            f"file not hashed at {path}: {'absent' if not exists else 'unreadable'}"
        ]
    else:
        rec["missing_evidence"] = []
        if declared_prefix:
            rec["declared_sha256_16"] = declared_prefix
            rec["declared_prefix_match"] = digest[: len(declared_prefix)] == declared_prefix
            rec["declared_prefix_source"] = SEAL_REL
    return rec


# ---------------------------------------------------------------------------
# Slot constructors
# ---------------------------------------------------------------------------


def _unknown(*, missing: list[str], evidence: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    slot: dict[str, Any] = {
        "value": UNKNOWN,
        "missing_evidence": list(missing),
        "evidence": list(evidence or []),
        "claim_class": CLAIM_CLASS,
    }
    slot.update(extra)
    return slot


def _known(value: Any, *, evidence: list[str], **extra: Any) -> dict[str, Any]:
    slot: dict[str, Any] = {
        "value": value,
        "missing_evidence": [],
        "evidence": list(evidence),
        "claim_class": CLAIM_CLASS,
    }
    slot.update(extra)
    return slot


def _slot_value(field: Any) -> Any:
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def _as_list(field: Any) -> list[Any] | None:
    if isinstance(field, list):
        return field
    if isinstance(field, dict):
        value = field.get("value")
        if isinstance(value, list):
            return value
    return None


def _hardware_numeric_keys(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                hits.append(here)
            hits.extend(_hardware_numeric_keys(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hits.extend(_hardware_numeric_keys(value, f"{path}[{i}]"))
    return hits


# ---------------------------------------------------------------------------
# Field collectors — recover from disk, never invent a measurement
# ---------------------------------------------------------------------------


def _sealed() -> tuple[dict[str, Any], dict[str, Any] | None]:
    src, doc = load_authority(SEALED_REL)
    rec = presence(SEALED_REL)
    rec["load"] = src
    if src == "ON_DISK":
        rec["sha256"] = _sha256_path(REPO / SEALED_REL)
    else:
        rec["sha256"] = _sha256_blob(git("show", f"HEAD:{SEALED_REL}"))
    return rec, doc


def _collect_model_family(sealed: Mapping[str, Any] | None, sealed_rec: Mapping[str, Any]) -> dict[str, Any]:
    family = sealed.get("family") if isinstance(sealed, Mapping) else None
    if isinstance(family, str) and family.strip():
        return _known(family, evidence=[f"{SEALED_REL}#family"], source_state=sealed_rec.get("source"))
    return _unknown(missing=[f"{SEALED_REL}#family"], evidence=[SEALED_REL], source_state=sealed_rec.get("source"))


def _collect_nx_id(sealed: Mapping[str, Any] | None, sealed_rec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(sealed, Mapping):
        return _unknown(missing=[f"{SEALED_REL} (unreadable)"], evidence=[SEALED_REL])
    model_id = sealed.get("model_id")
    resident_id = sealed.get("resident_identity")
    protocol = sealed.get("protocol")
    if not isinstance(model_id, str) or not model_id.strip():
        return _unknown(missing=[f"{SEALED_REL}#model_id"], evidence=[SEALED_REL])
    return _known(
        {
            "model_id": model_id,
            "resident_identity": resident_id,
            "protocol": protocol,
            "profile_schema": sealed.get("profile_schema"),
        },
        evidence=[f"{SEALED_REL}#model_id", f"{SEALED_REL}#resident_identity"],
        source_state=sealed_rec.get("source"),
        profile_sha256=sealed_rec.get("sha256") or UNKNOWN,
    )


def _collect_parent_lineage(
    sealed: Mapping[str, Any] | None,
    seal: Mapping[str, Any] | None,
    atlas: Mapping[str, Any] | None,
) -> dict[str, Any]:
    artifact_root = sealed.get("artifact_root") if isinstance(sealed, Mapping) else None
    mix_path = Path(str(artifact_root)) / "MIX_REPORT.json" if artifact_root else None
    mix: dict[str, Any] | None = None
    mix_state = "HOST_ABSENT"
    if mix_path is not None and mix_path.is_file():
        try:
            loaded = load_json(mix_path)
            if isinstance(loaded, dict):
                mix = loaded
                mix_state = "ON_DISK"
        except (OSError, json.JSONDecodeError, UnicodeError):
            mix_state = "ON_DISK_UNPARSEABLE"
    closure = None
    if isinstance(seal, Mapping):
        fields = seal.get("fields") if isinstance(seal.get("fields"), dict) else {}
        physical = fields.get("physical_closure") if isinstance(fields, dict) else None
        if isinstance(physical, dict):
            closure = physical.get("value") if "value" in physical else physical
    art = atlas.get("artifact") if isinstance(atlas, Mapping) and isinstance(atlas.get("artifact"), dict) else {}
    parent_params = None
    mix_id = None
    if isinstance(mix, dict):
        parent_params = mix.get("parent_params")
        mix_id = mix.get("mix_id")
    if parent_params is None and isinstance(closure, dict):
        parent_params = closure.get("parent_params")
    if parent_params is None:
        parent_params = art.get("parent_params")
    hist_src, _hist = load_authority(HISTORICAL_REL)
    evidence = [SEALED_REL, SEAL_REL, ATLAS_REL]
    if mix_state == "ON_DISK" and mix_path is not None:
        evidence.append(str(mix_path))
    if hist_src != "ABSENT":
        evidence.append(HISTORICAL_REL)
    if not artifact_root and parent_params is None and mix_id is None:
        return _unknown(
            missing=["parent artifact_root / parent_params / mix_id"],
            evidence=evidence,
        )
    return _known(
        {
            "family": (sealed or {}).get("family") if isinstance(sealed, Mapping) else None,
            "artifact_root": artifact_root,
            "mix_id": mix_id if mix_id is not None else UNKNOWN,
            "parent_params": parent_params if parent_params is not None else UNKNOWN,
            "mix_report_state": mix_state,
            "historical_runtime_identity": presence(HISTORICAL_REL),
            "did_not_load_second_27b": (mix or {}).get("did_not_load_second_27b") if mix else UNKNOWN,
        },
        evidence=evidence,
        note=(
            "parent_params / mix_id are catalog identity of NOETIC_PARENT_A, "
            "not a qualified physical EBPW"
        ),
    )


def _collect_tokenizer(sealed: Mapping[str, Any] | None, seal: Mapping[str, Any] | None) -> dict[str, Any]:
    path = sealed.get("tokenizer") if isinstance(sealed, Mapping) else None
    declared = None
    if isinstance(seal, Mapping):
        fields = seal.get("fields") if isinstance(seal.get("fields"), dict) else {}
        tok = fields.get("tokenizer_sha256_16") if isinstance(fields, dict) else None
        if isinstance(tok, dict):
            declared = tok.get("value")
        elif isinstance(tok, str):
            declared = tok
    rec = _host_file_record(path if isinstance(path, str) else None, declared_prefix=declared)
    prompt = sealed.get("prompt_contract") if isinstance(sealed, Mapping) else None
    rec["prompt_contract"] = prompt if isinstance(prompt, dict) else UNKNOWN
    rec["evidence"] = [SEALED_REL, SEAL_REL]
    rec["claim_class"] = CLAIM_CLASS
    if rec.get("sha256") == UNKNOWN:
        rec["value"] = UNKNOWN
    else:
        rec["value"] = {
            "path": rec.get("path"),
            "sha256": rec.get("sha256"),
            "declared_sha256_16": rec.get("declared_sha256_16"),
            "declared_prefix_match": rec.get("declared_prefix_match"),
            "prompt_contract": rec.get("prompt_contract"),
        }
    return rec


def _collect_executable(sealed: Mapping[str, Any] | None, seal: Mapping[str, Any] | None) -> dict[str, Any]:
    resident_bin = sealed.get("resident_binary") if isinstance(sealed, Mapping) else None
    greedy_bin = sealed.get("binary") if isinstance(sealed, Mapping) else None
    declared = None
    seal_runtime = None
    if isinstance(seal, Mapping):
        fields = seal.get("fields") if isinstance(seal.get("fields"), dict) else {}
        pref = fields.get("runtime_binary_sha256_16") if isinstance(fields, dict) else None
        if isinstance(pref, dict):
            declared = pref.get("value")
        elif isinstance(pref, str):
            declared = pref
        rt = fields.get("runtime_binary") if isinstance(fields, dict) else None
        if isinstance(rt, dict):
            seal_runtime = rt.get("value")
        elif isinstance(rt, str):
            seal_runtime = rt
    resident_rec = _host_file_record(
        resident_bin if isinstance(resident_bin, str) else None, declared_prefix=declared
    )
    greedy_rec = _host_file_record(
        greedy_bin if isinstance(greedy_bin, str) else None, declared_prefix=declared
    )
    missing = []
    for rec, label in ((resident_rec, "resident_binary"), (greedy_rec, "binary")):
        if rec.get("sha256") == UNKNOWN:
            missing.extend(rec.get("missing_evidence") or [label])
    body = {
        "resident_binary": resident_rec,
        "binary": greedy_rec,
        "seal_runtime_binary": seal_runtime,
        "compiler": (sealed or {}).get("compiler") if isinstance(sealed, Mapping) else None,
        "executable_profile": (sealed or {}).get("executable_profile") if isinstance(sealed, Mapping) else None,
    }
    hashed = {}
    if isinstance(resident_rec.get("sha256"), str) and resident_rec["sha256"] != UNKNOWN:
        hashed["resident_binary"] = resident_rec["sha256"]
    if isinstance(greedy_rec.get("sha256"), str) and greedy_rec["sha256"] != UNKNOWN:
        hashed["binary"] = greedy_rec["sha256"]
    if not hashed:
        return _unknown(
            missing=missing or ["resident binary and greedy binary were not hashed"],
            evidence=[SEALED_REL, SEAL_REL],
            **body,
        )
    seal_match_role = None
    if greedy_rec.get("declared_prefix_match") is True:
        seal_match_role = "binary"
    elif resident_rec.get("declared_prefix_match") is True:
        seal_match_role = "resident_binary"
    return _known(
        {
            "by_role": hashed,
            "seal_declared_sha256_16": declared,
            "seal_matching_role": seal_match_role,
            "note": (
                "Two executables are declared. The resident seal's "
                "runtime_binary_sha256_16 matches 'binary' (hybrid greedy) when "
                "that file is present; resident_binary is a different artifact."
            ),
        },
        evidence=[SEALED_REL, SEAL_REL, *(r.get("path") or "" for r in (resident_rec, greedy_rec) if r.get("sha256") != UNKNOWN)],
        **body,
    )


def _collect_ebpw(
    flash: Mapping[str, Any] | None,
    sealed_rec: Mapping[str, Any],
) -> dict[str, Any]:
    flash_physical = None
    if isinstance(flash, Mapping):
        meta = flash.get("meta_measurement_state")
        if isinstance(meta, dict):
            flash_physical = meta.get("physical_ebpw")
    flash_missing = [
        "source-independent Flash NX is not qualified (FLASH_NX_COMPLETENESS_AUDIT seven_all_met is not true)",
        "physical_ebpw is NULL_BY_RULE / not a protected measurement for Flash",
    ]
    return _unknown(
        missing=[
            "qualified physical EBPW measurement on this host",
            "sidecar has no GPU authority and cannot produce PROTECTED_ABSOLUTE",
        ],
        evidence=[
            SEALED_REL,
            FLASH_NX_REL,
            PROFILE_SCHEMA_REL,
            "tools/future/ebpw_categories.py#CompletePhysicalEbpw",
        ],
        kind="qualified_physical_ebpw",
        category="complete_physical_ebpw",
        copied_declared_catalog_number=False,
        declared_catalog_pointer={
            "path": SEALED_REL,
            "fields": ["physical_ebpw", "representation.physical_ebpw"],
            "class": "STATIC_CATALOG_IDENTITY_NOT_QUALIFIED_ON_THIS_HOST",
            "sealed_source": sealed_rec.get("source"),
            "copied_into_identity_value": False,
        },
        flash=_unknown(
            missing=flash_missing,
            evidence=[FLASH_NX_REL],
            recorded_meta_physical_ebpw=flash_physical if flash_physical is not None else "NULL_BY_RULE",
        ),
        note=(
            "Sealed-3.14 catalog complete_ebpw lives in the sealed profile and "
            "MIX_REPORT; this identity does not copy it. Qualified physical "
            "EBPW is UNKNOWN everywhere on this host."
        ),
    )


def _collect_active_bytes(atlas: Mapping[str, Any] | None) -> dict[str, Any]:
    headline = atlas.get("headline") if isinstance(atlas, Mapping) else None
    catalog = None
    if isinstance(headline, dict) and isinstance(headline.get("active_weight_bytes_per_token"), (int, float)):
        catalog = headline.get("active_weight_bytes_per_token")
    missing_actual = [
        "protected complete-token receipt with actual_read_bytes_per_token",
        "sidecar cannot emit DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE",
    ]
    catalog_slot: dict[str, Any]
    if catalog is None:
        catalog_slot = _unknown(
            missing=[f"{ATLAS_REL}#headline.active_weight_bytes_per_token"],
            evidence=[ATLAS_REL],
        )
    else:
        catalog_slot = _known(
            catalog,
            evidence=[f"{ATLAS_REL}#headline.active_weight_bytes_per_token"],
            label="STATIC_DERIVATION_FROM_CATALOG",
            note="catalog WEIGHT traffic only; not actual_read_bytes_per_token",
        )
    value = {
        "catalog_weight_bytes_per_token": catalog_slot,
        "active_representation_bytes_per_token": _unknown(missing=missing_actual, evidence=[ATLAS_REL]),
        "actual_read_bytes_per_token": _unknown(missing=missing_actual, evidence=[ATLAS_REL]),
        "transient_bytes_per_token": _unknown(missing=missing_actual, evidence=[ATLAS_REL]),
        "activations_included": False,
        "kv_cache_included": False,
        "recurrent_state_included": False,
        "claim_boundary": (
            "Catalog-derived WEIGHT traffic only. Activations, KV cache, and "
            "DeltaNet recurrent state are not counted. actual_read and transient "
            "remain UNKNOWN until a protected complete-token receipt."
        ),
    }
    return _known(
        value,
        evidence=[ATLAS_REL, PROFILE_SCHEMA_REL],
        note="nested actual_read / transient / active_representation are UNKNOWN",
    )


def _collect_capability(cap: Mapping[str, Any] | None, cap_src: str) -> dict[str, Any]:
    if not isinstance(cap, Mapping):
        return _unknown(
            missing=[f"{CAPABILITY_REL} unreadable"],
            evidence=[CAPABILITY_REL],
            source_state=cap_src,
        )
    overall = cap.get("overall") if isinstance(cap.get("overall"), dict) else {}
    sufficient = cap.get("identity_sufficient")
    verdict = cap.get("identity_verdict")
    missing = []
    if sufficient is not True:
        missing.append(
            "capability receipt identity_sufficient is not true; the score does not identify which body earned it"
        )
    return _known(
        {
            "path": CAPABILITY_REL,
            "schema": cap.get("schema"),
            "label": cap.get("label"),
            "overall_passed": overall.get("passed"),
            "overall_total": overall.get("total"),
            "identity_sufficient": sufficient,
            "identity_verdict": verdict,
            "role": "CONTROL_HISTORICAL_NOT_CURRENT_PROOF",
            "not_a_target": True,
            "not_a_ceiling": True,
            "bench_state_on_receipt": (cap.get("bench") or {}).get("state")
            if isinstance(cap.get("bench"), dict)
            else UNKNOWN,
        },
        evidence=[CAPABILITY_REL, SEAL_REL],
        missing_evidence=missing,
        source_state=cap_src,
        note=(
            "30/43 is a historical CONTROL record. identity_sufficient is false "
            "(UNIDENTIFIED). This sidecar does not re-run the suite."
        ),
    )


def _collect_tps() -> dict[str, Any]:
    return _unknown(
        missing=[
            "PROTECTED_ABSOLUTE complete-token measurement under a proven HCLI lease",
            "sidecar has no GPU authority; copying sealed current_runtime numbers would launder a hardware claim",
        ],
        evidence=[SEALED_REL, QUAL_REL, "receipts/future/evidence/QWEN27_TOKEN_NS_BUDGET.json"],
        copied_sealed_runtime_numbers=False,
        declared_runtime_pointer={
            "path": SEALED_REL,
            "fields": [
                "current_runtime.complete_tps_current_measured",
                "current_runtime.complete_tps_historical_qualified",
            ],
            "class": "HISTORICAL_RECORD_NOT_A_SIDECAR_MEASUREMENT",
            "copied_into_identity_value": False,
            "do_not_promote": True,
        },
        token_ns_budget_pointer={
            "path": "receipts/future/evidence/QWEN27_TOKEN_NS_BUDGET.json",
            "lifecycle_buckets": "null slots until a native protected complete-token receipt",
        },
    )


def _collect_machine_genome() -> dict[str, Any]:
    src, doc = load_authority(MACHINE_GENOME_REL)
    rec = presence(MACHINE_GENOME_REL)
    rec["load"] = src
    py_rec = presence(MACHINE_GENOME_PY_REL)
    if not isinstance(doc, dict):
        return _unknown(
            missing=[f"{MACHINE_GENOME_REL} unreadable"],
            evidence=[MACHINE_GENOME_REL, MACHINE_GENOME_PY_REL, MACHINE_PY_REL],
            genome_py=py_rec,
            machine_py_rule=(
                "hcli/machine.py: a genome is a PRIOR; STALE genomes are not used; "
                "this sidecar does not remeasure and does not call measure_bandwidth"
            ),
        )
    toolchain = doc.get("toolchain") if isinstance(doc.get("toolchain"), dict) else {}
    metal = toolchain.get("metal_compiler")
    # Identity of the machine from the prior receipt. Bandwidth is NOT reattested.
    identity = {
        "schema": doc.get("schema"),
        "soc": doc.get("soc"),
        "arch": doc.get("arch"),
        "cpu_cores": doc.get("cpu_cores"),
        "perf_cores": doc.get("perf_cores"),
        "efficiency_cores": doc.get("efficiency_cores"),
        "gpu_cores": doc.get("gpu_cores"),
        "memory_bytes": doc.get("memory_bytes"),
        "os": doc.get("os"),
        "os_product": doc.get("os_product"),
        "knowledge_level": doc.get("knowledge_level"),
        "toolchain_metal_compiler": metal,
        "bench_state_on_receipt": (doc.get("bench") or {}).get("state")
        if isinstance(doc.get("bench"), dict)
        else UNKNOWN,
        "measured_bandwidth": _unknown(
            missing=[
                "this sidecar did not remeasure bandwidth and does not reattest the prior receipt",
                "prior MACHINE_GENOME.json bench.state is UNKNOWN (quiescence was not a benchmark input)",
            ],
            evidence=[MACHINE_GENOME_REL, ACCEL_GENOME_REL],
            this_sidecar_does_not_reattest=True,
            prior_receipt=MACHINE_GENOME_REL,
            prior_claim_boundary=(
                "589.73 GB/s in ACCELERATOR_MACHINE_GENOME.json is Codex's prior "
                "INSTANCE measurement; not copied into this identity value"
            ),
        ),
        "thermal_envelope": doc.get("thermal_envelope"),
        "sustained_behaviour": doc.get("sustained_behaviour"),
    }
    return _known(
        identity,
        evidence=[MACHINE_GENOME_REL, ACCEL_GENOME_REL, MACHINE_GENOME_PY_REL, MACHINE_PY_REL],
        source_state=src,
        genome_py=py_rec,
        read_only=True,
        measure_bandwidth_invoked=False,
        note=(
            "Cited prior genome identity (soc/cores/memory/toolchain). "
            "tools/accelerator/machine_genome.py was not executed."
        ),
    )


def _collect_backend(sealed: Mapping[str, Any] | None, genome: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(sealed, Mapping):
        return _unknown(missing=[f"{SEALED_REL} unreadable"], evidence=[SEALED_REL])
    metal = None
    gval = _slot_value(genome)
    if isinstance(gval, dict):
        metal = gval.get("toolchain_metal_compiler")
    return _known(
        {
            "provider": sealed.get("provider"),
            "runtime": sealed.get("runtime"),
            "protocol": sealed.get("protocol"),
            "mode": sealed.get("mode"),
            "fusion_env": sealed.get("fusion_env"),
            "executable_profile": sealed.get("executable_profile"),
            "require_fusion_env": sealed.get("require_fusion_env"),
            "metal_compiler": metal,
            "gpu_authority": False,
        },
        evidence=[SEALED_REL, MACHINE_GENOME_REL],
    )


def _collect_tools(handoff: Mapping[str, Any] | None) -> dict[str, Any]:
    systems = handoff.get("systems") if isinstance(handoff, Mapping) else None
    if not isinstance(systems, dict):
        return _unknown(
            missing=[f"{HANDOFF_REL}#systems"],
            evidence=[HANDOFF_REL],
        )
    rows = []
    for name in sorted(systems):
        row = systems[name] if isinstance(systems[name], dict) else {}
        rows.append(
            {
                "id": name,
                "module": row.get("module"),
                "state": row.get("state"),
                "receipt": row.get("receipt"),
            }
        )
    if not any(r["id"] == "resident_identity" for r in rows):
        rows.append(
            {
                "id": "resident_identity",
                "module": "tools/future/resident_identity.py",
                "state": "EXECUTABLE",
                "receipt": f"receipts/future/{RECEIPT}",
                "note": "this module; not yet in FUTURE_SUBSTRATE_HANDOFF inventory",
            }
        )
    rows = sorted(rows, key=lambda r: str(r["id"]))
    return _known(
        rows,
        evidence=[HANDOFF_REL],
        n=len(rows),
        handoff_system_counts=handoff.get("system_counts") if isinstance(handoff, Mapping) else None,
    )


def _collect_pressure(
    contamination: Mapping[str, Any] | None,
    qual: Mapping[str, Any] | None,
) -> dict[str, Any]:
    klass = None
    evidence = []
    if isinstance(contamination, dict):
        klass = contamination.get("contamination_class")
        evidence.append(CONTAMINATION_REL)
    pipe = qual.get("pipeline") if isinstance(qual, Mapping) else None
    if klass is None and isinstance(pipe, dict):
        klass = pipe.get("contamination_class")
    if isinstance(qual, dict):
        evidence.append(QUAL_REL)
    lock_rel = ".hcli/locks/protected-accelerator-bench.lock"
    lock_path = REPO / lock_rel
    lock_on_disk = lock_path.is_file()
    lease = None
    stop = None
    if isinstance(qual, dict):
        lease = (qual.get("pipeline") or {}).get("lease_present") if isinstance(qual.get("pipeline"), dict) else None
        stop = qual.get("dry_run_stop")
    if klass is None:
        return _unknown(
            missing=[f"{CONTAMINATION_REL}#contamination_class"],
            evidence=evidence or [CONTAMINATION_REL],
            lock_file={"path": lock_rel, "on_disk": lock_on_disk},
        )
    return _known(
        {
            "contamination_class": klass,
            "contamination_reason": (contamination or {}).get("contamination_reason")
            if isinstance(contamination, dict)
            else None,
            "lease_present_on_qualification_receipt": lease,
            "qualification_stop": stop,
            "lock_file": {
                "path": lock_rel,
                "on_disk": lock_on_disk,
                "probe": "pathlib is_file only; never flock",
            },
            "gpu_authority": False,
        },
        evidence=evidence,
        note=(
            "HEAVY is recorded on CONTAMINATION_SCIENCE / QUALIFICATION_PIPELINE. "
            "This sidecar does not quiesce standing workers and does not seize the lock."
        ),
    )


def _collect_frontiers(frontier: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(frontier, Mapping):
        return _unknown(missing=[f"{FRONTIER_REL} unreadable"], evidence=[FRONTIER_REL])
    entries = frontier.get("entries")
    if not isinstance(entries, list):
        return _unknown(missing=[f"{FRONTIER_REL}#entries"], evidence=[FRONTIER_REL])
    rows = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "id": item.get("id"),
                "classification": item.get("classification"),
                "title": item.get("title"),
            }
        )
    rows = sorted(rows, key=lambda r: str(r.get("id") or ""))
    return _known(
        rows,
        evidence=[FRONTIER_REL],
        n=len(rows),
        counts=frontier.get("counts"),
        note=(
            "CLAUDE_GLOBAL_FRONTIER.json is read, not mutated "
            "(tools/future/global_frontier.py is frozen for this lane)"
        ),
    )


def _collect_workunits(workunits: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(workunits, Mapping):
        missing = [f"{WORKUNITS_REL} unreadable"]
        u = _unknown(missing=missing, evidence=[WORKUNITS_REL])
        return u, u
    units = workunits.get("work_units")
    if not isinstance(units, list):
        missing = [f"{WORKUNITS_REL}#work_units"]
        u = _unknown(missing=missing, evidence=[WORKUNITS_REL])
        return u, u

    def _compact(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": item.get("id"),
            "status": item.get("status"),
            "species": item.get("species"),
            "role": item.get("role"),
            "blocked_reason": item.get("blocked_reason"),
        }

    active: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    by_status: dict[str, int] = {}
    for item in units:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "UNKNOWN")
        by_status[status] = by_status.get(status, 0) + 1
        row = _compact(item)
        if status.lower() == "blocked":
            blocked.append(row)
        else:
            active.append(row)
    active = sorted(active, key=lambda r: str(r.get("id") or ""))
    blocked = sorted(blocked, key=lambda r: str(r.get("id") or ""))
    by_status = {k: by_status[k] for k in sorted(by_status)}
    evidence = [WORKUNITS_REL]
    return (
        _known(
            active,
            evidence=evidence,
            n=len(active),
            by_status=by_status,
            rule="active = work_units whose status is not blocked (pending/running/other)",
        ),
        _known(
            blocked,
            evidence=evidence,
            n=len(blocked),
            by_status=by_status,
            rule="blocked = work_units with status blocked; physical units stay SLEEPING, never synthetic-complete",
        ),
    )


def _collect_scars() -> dict[str, Any]:
    src, doc = load_authority(NEGATIVE_REL)
    evidence = [NEGATIVE_REL]
    rows: list[dict[str, Any]] = []
    try:
        from tools.future.negative_index import query as scar_query

        hits = scar_query(model="qwen3.8-27b")
        if isinstance(hits, list):
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                rows.append(
                    {
                        "scar_id": hit.get("scar_id"),
                        "hypothesis_family": hit.get("hypothesis_family"),
                        "verdict": hit.get("verdict"),
                        "source_path": hit.get("source_path"),
                        "refuse_eligible": hit.get("refuse_eligible"),
                        "level": hit.get("level"),
                    }
                )
    except Exception as exc:  # noqa: BLE001 — record the failure as data
        return _unknown(
            missing=[f"negative_index.query failed: {type(exc).__name__}: {exc}"],
            evidence=evidence,
        )
    rows = sorted(rows, key=lambda r: (str(r.get("scar_id") or ""), str(r.get("hypothesis_family") or "")))
    coverage = doc.get("coverage") if isinstance(doc, dict) else None
    if not rows and not isinstance(coverage, dict):
        return _unknown(
            missing=["no scars recovered for model qwen3.8-27b"],
            evidence=evidence,
            source_state=src,
        )
    n_refuse = sum(1 for r in rows if r.get("refuse_eligible"))
    return _known(
        rows,
        evidence=evidence,
        n=len(rows),
        n_refuse_eligible=n_refuse,
        query={"model": "qwen3.8-27b"},
        index_coverage=coverage,
        source_state=src,
        note="compact projection of NEGATIVE_SCIENCE_INDEX; not the full corpus dump",
    )


def _collect_weaknesses(
    *,
    flash: Mapping[str, Any] | None,
    teacher: Mapping[str, Any] | None,
    teacher_rel: str,
    contamination: Mapping[str, Any] | None,
    qual: Mapping[str, Any] | None,
    cap: Mapping[str, Any] | None,
    genome: Mapping[str, Any],
    blocked: Mapping[str, Any],
    handoff: Mapping[str, Any] | None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    def add(wid: str, statement: str, evidence: list[str], blocks: list[str]) -> None:
        items.append(
            {
                "id": wid,
                "statement": statement,
                "evidence": evidence,
                "blocks": blocks,
            }
        )

    add(
        "W-NO-GPU-AUTHORITY",
        "This sidecar has no GPU authority; every hardware quantity is UNKNOWN by rule.",
        [HANDOFF_REL, QUAL_REL],
        ["protected measurement", "DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE"],
    )
    gval = _slot_value(genome)
    metal = gval.get("toolchain_metal_compiler") if isinstance(gval, dict) else None
    add(
        "W-METAL-COMPILER-ABSENT",
        "Machine genome records Metal compiler ABSENT (xcrun metal not installed).",
        [MACHINE_GENOME_REL],
        ["AOT metallib", "protected Metal qualification"],
    )
    if isinstance(teacher, dict):
        rows_written = teacher.get("teacher_rows_written")
        requested = teacher.get("requested_rows") or teacher.get("minimum_rows")
        add(
            "W-TEACHER-CAPTURE-BLOCKED-NO-METAL-GPU",
            (
                f"Flash teacher capture status={teacher.get('status')!r} "
                f"rows={rows_written}/{requested}; error="
                f"{(teacher.get('failure') or {}).get('error') if isinstance(teacher.get('failure'), dict) else teacher.get('failure')}"
            ),
            [teacher_rel],
            ["Flash meta teacher capture", "Flash source-independent NX"],
        )
    seven = flash.get("seven_all_met") if isinstance(flash, Mapping) else None
    add(
        "W-FLASH-NX-SCAFFOLD-ONLY",
        (
            "Flash source-independent NX is not qualified "
            f"(seven_all_met={seven!r}; physical_ebpw NULL_BY_RULE)."
        ),
        [FLASH_NX_REL],
        ["Flash candidate qualification", "Flash EBPW"],
    )
    klass = None
    if isinstance(contamination, dict):
        klass = contamination.get("contamination_class")
    add(
        "W-CONTAMINATION-HEAVY",
        f"Contamination class is {klass!r}; qualification will not quiesce standing workers.",
        [CONTAMINATION_REL, QUAL_REL],
        ["protected window", "machine quiescence"],
    )
    stop = (qual or {}).get("dry_run_stop") if isinstance(qual, Mapping) else None
    add(
        "W-LEASE-UNPROVEN",
        (
            "Protected bench lock: "
            + str((stop or {}).get("reason") or "no proven HCLI lease; flock would be a seizure")
        ),
        [QUAL_REL],
        ["protected accelerator benchmark"],
    )
    if isinstance(cap, dict) and cap.get("identity_sufficient") is not True:
        add(
            "W-CAPABILITY-BODY-UNIDENTIFIED",
            "CAPABILITY_noetic-sealed-3.14.json identity_sufficient is false (UNIDENTIFIED body).",
            [CAPABILITY_REL],
            ["comparing capability rows", "holding the resident to a score"],
        )
    add(
        "W-NONFINAL-27B",
        "The current 27B is CURRENT_NONFINAL_HCLI_WORKER, not a Singularity.",
        [SEALED_REL, "tools/future/resident_install.py"],
        ["treating sealed-3.14 as the released Hawking"],
    )
    add(
        "W-QUALIFIED-EBPW-UNKNOWN",
        "Qualified physical EBPW is UNKNOWN on this host; Flash EBPW is UNKNOWN.",
        [FLASH_NX_REL, SEALED_REL],
        ["EBPW promotion", "complete_physical_ebpw claims"],
    )
    add(
        "W-TPS-TOKEN-NS-UNMEASURED",
        "No protected complete-token TPS/token_ns evidence exists for this sidecar to attest.",
        [SEALED_REL, QUAL_REL],
        ["accepted_tps claims", "token_ns claims"],
    )
    n_blocked = (_as_list(blocked) or [])
    if n_blocked:
        add(
            "W-BLOCKED-WORKUNITS",
            f"{len(n_blocked)} blocked WorkUnits on HCLI_FUTURE_WORKUNITS (derived, not pinned).",
            [WORKUNITS_REL],
            ["advancing blocked physical candidates without hardware"],
        )
    if isinstance(handoff, dict):
        for i, blocker in enumerate(handoff.get("blockers") or [], start=1):
            add(
                f"W-HANDOFF-BLOCKER-{i:02d}",
                str(blocker),
                [HANDOFF_REL],
                ["campaign close conditions"],
            )
    # metal presence is recorded even if the genome slot is a dict ABSENT
    if metal is None:
        pass
    items = sorted(items, key=lambda r: (str(r["id"]), str(r["statement"])))
    # Deduplicate HANDOFF blockers that share an id by keeping all statements.
    return _known(
        items,
        evidence=[
            HANDOFF_REL,
            QUAL_REL,
            FLASH_NX_REL,
            CONTAMINATION_REL,
            CAPABILITY_REL,
            MACHINE_GENOME_REL,
            WORKUNITS_REL,
            teacher_rel,
        ],
        n=len(items),
        metal_compiler=metal,
    )


def _collect_children() -> dict[str, Any]:
    found: list[str] = []
    future = REPO / "receipts" / "future"
    if future.is_dir():
        for path in sorted(future.iterdir()):
            name = path.name.upper()
            if "SUCCESSION" in name or "CHILD_IDENTITY" in name:
                found.append(str(path.relative_to(REPO)))
    return _known(
        found,
        evidence=[SEALED_REL, "receipts/future"],
        status="NONE_RECORDED_ON_DISK" if not found else "RECORDED",
        integration_point=(
            "tools/future/succession.py (this-wave sibling, not imported). "
            "Swap this collector for succession's child list when that module lands."
        ),
        looked_for=["receipts/future/*SUCCESSION*", "receipts/future/*CHILD_IDENTITY*"],
        note="empty list is NONE_RECORDED_ON_DISK (we looked), not UNKNOWN",
    )


def _collect_sandbox(workunits: Mapping[str, Any] | None) -> dict[str, Any]:
    allowed: list[str] = []
    forbidden: list[str] = []
    if isinstance(workunits, Mapping):
        auth = workunits.get("authority")
        if isinstance(auth, dict):
            if isinstance(auth.get("allowed"), list):
                allowed = sorted(str(x) for x in auth["allowed"])
            if isinstance(auth.get("forbidden"), list):
                forbidden = sorted(str(x) for x in auth["forbidden"])
    if not allowed and not forbidden:
        return _unknown(
            missing=[f"{WORKUNITS_REL}#authority"],
            evidence=[WORKUNITS_REL],
            gpu_authority=False,
            integration_point=(
                "tools/future/sandbox.py (this-wave sibling, not imported). "
                "Bounded authority is read from HCLI_FUTURE_WORKUNITS until sandbox lands."
            ),
        )
    return _known(
        {
            "gpu_authority": False,
            "claim_class": CLAIM_CLASS,
            "may_acquire_gpu_lease": False,
            "may_promote": False,
            "may_choose_singularity": False,
            "may_modify_verifier": False,
            "bounded_authority": allowed,
            "forbidden_authority": forbidden,
        },
        evidence=[WORKUNITS_REL, QUAL_REL],
        integration_point=(
            "tools/future/sandbox.py (this-wave sibling, not imported). "
            "Authority copied from HCLI_FUTURE_WORKUNITS.json#authority."
        ),
    )


# ---------------------------------------------------------------------------
# Assemble / validate / persist
# ---------------------------------------------------------------------------


def collect() -> dict[str, Any]:
    """Build the honest identity from disk/git. Does not write."""
    sealed_rec, sealed = _sealed()
    cap_rec, cap = load_authority(CAPABILITY_REL)
    _seal_src, seal = load_authority(SEAL_REL)
    _atlas_src, atlas = load_authority(ATLAS_REL)
    _hand_src, handoff = load_authority(HANDOFF_REL)
    _front_src, frontier = load_authority(FRONTIER_REL)
    _wu_src, workunits = load_authority(WORKUNITS_REL)
    _c_src, contamination = load_authority(CONTAMINATION_REL)
    _q_src, qual = load_authority(QUAL_REL)
    _f_src, flash = load_authority(FLASH_NX_REL)
    teacher_rel, _t_src, teacher = _first_authority(TEACHER_RELS)
    genome = _collect_machine_genome()
    active_wu, blocked_wu = _collect_workunits(workunits)
    identity: dict[str, Any] = {
        "residency_status": RESIDENCY_STATUS,
        "claim_class": CLAIM_CLASS,
        "gpu_authority": False,
        "model_family": _collect_model_family(sealed, sealed_rec),
        "nx_id": _collect_nx_id(sealed, sealed_rec),
        "parent_lineage": _collect_parent_lineage(sealed, seal, atlas),
        "tokenizer_identity": _collect_tokenizer(sealed, seal),
        "executable_hash": _collect_executable(sealed, seal),
        "ebpw": _collect_ebpw(flash, sealed_rec),
        "active_bytes_evidence": _collect_active_bytes(atlas),
        "capability_receipt": _collect_capability(cap, cap_rec),
        "tps_token_ns_evidence": _collect_tps(),
        "machine_genome": genome,
        "current_backend": _collect_backend(sealed, genome),
        "current_tools": _collect_tools(handoff),
        "current_resource_pressure": _collect_pressure(contamination, qual),
        "current_frontiers": _collect_frontiers(frontier),
        "active_workunits": active_wu,
        "blocked_workunits": blocked_wu,
        "known_scars": _collect_scars(),
        "known_weaknesses": _collect_weaknesses(
            flash=flash,
            teacher=teacher,
            teacher_rel=teacher_rel,
            contamination=contamination,
            qual=qual,
            cap=cap,
            genome=genome,
            blocked=blocked_wu,
            handoff=handoff,
        ),
        "children": _collect_children(),
        "sandbox_authority": _collect_sandbox(workunits),
        "authority_sources": {
            "sealed_profile": sealed_rec,
            "capability": {**presence(CAPABILITY_REL), "load": cap_rec},
            "machine_genome": presence(MACHINE_GENOME_REL),
            "codex_accelerator_handoff": presence(CODEX_HANDOFF_REL),
            "qualification_queue": [presence(r) for r in QUEUE_RELS],
            "teacher_capture": presence(teacher_rel),
        },
    }
    return identity


def _blockers_evident(identity: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    blocked = _as_list(identity.get("blocked_workunits"))
    if blocked:
        found.append("blocked_workunits")
    sandbox = identity.get("sandbox_authority")
    svalue = _slot_value(sandbox) if isinstance(sandbox, dict) else sandbox
    if isinstance(svalue, dict) and svalue.get("gpu_authority") is False:
        found.append("sandbox_authority.gpu_authority=false")
    if identity.get("gpu_authority") is False:
        found.append("gpu_authority=false")
    pressure = identity.get("current_resource_pressure")
    pvalue = _slot_value(pressure) if isinstance(pressure, dict) else pressure
    if isinstance(pvalue, dict) and str(pvalue.get("contamination_class") or "").upper() == "HEAVY":
        found.append("contamination_class=HEAVY")
    if isinstance(pvalue, str) and pvalue.upper() == "HEAVY":
        found.append("contamination_class=HEAVY")
    ebpw = identity.get("ebpw")
    if _slot_value(ebpw) == UNKNOWN:
        found.append("qualified_physical_ebpw=UNKNOWN")
    tps = identity.get("tps_token_ns_evidence")
    if _slot_value(tps) == UNKNOWN:
        found.append("tps_token_ns_evidence=UNKNOWN")
    return found


def _unknown_required_broken(identity: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    ebpw = identity.get("ebpw")
    ebpw_val = _slot_value(ebpw)
    if ebpw_val != UNKNOWN:
        reasons.append(
            f"ebpw: qualified physical EBPW must be UNKNOWN on this host; got {ebpw_val!r}"
        )
    elif isinstance(ebpw, dict):
        missing = ebpw.get("missing_evidence")
        if not isinstance(missing, list) or not missing:
            reasons.append("ebpw: value is UNKNOWN but missing_evidence is not named")
        flash = ebpw.get("flash")
        if isinstance(flash, dict) and _slot_value(flash) != UNKNOWN:
            reasons.append(
                f"ebpw.flash: Flash EBPW must be UNKNOWN; got {_slot_value(flash)!r}"
            )
    tps = identity.get("tps_token_ns_evidence")
    tps_val = _slot_value(tps)
    if tps_val != UNKNOWN:
        reasons.append(
            f"tps_token_ns_evidence: protected TPS/token_ns must be UNKNOWN; got {tps_val!r}"
        )
    elif isinstance(tps, dict):
        missing = tps.get("missing_evidence")
        if not isinstance(missing, list) or not missing:
            reasons.append(
                "tps_token_ns_evidence: value is UNKNOWN but missing_evidence is not named"
            )
    active = identity.get("active_bytes_evidence")
    aval = _slot_value(active) if isinstance(active, dict) else active
    if isinstance(aval, dict):
        for key in (
            "actual_read_bytes_per_token",
            "transient_bytes_per_token",
            "active_representation_bytes_per_token",
        ):
            slot = aval.get(key)
            if _slot_value(slot) != UNKNOWN:
                reasons.append(
                    f"active_bytes_evidence.{key}: must be UNKNOWN without a protected receipt; "
                    f"got {_slot_value(slot)!r}"
                )
            elif isinstance(slot, dict):
                missing = slot.get("missing_evidence")
                if not isinstance(missing, list) or not missing:
                    reasons.append(
                        f"active_bytes_evidence.{key}: UNKNOWN without missing_evidence named"
                    )
    return reasons


def validate(identity: Any) -> dict[str, Any]:
    """ACCEPTED or REJECTED. Never warns-and-continues."""
    reasons: list[str] = []
    if not isinstance(identity, Mapping):
        return {
            "status": "REJECTED",
            "reasons": ["<root> is not a mapping"],
            "named_refusal": "REJECTED: <root> is not a mapping",
        }
    for field in IDENTITY_FIELDS:
        if field not in identity:
            reasons.append(f"missing required field {field}")
    status = identity.get("residency_status")
    if status != RESIDENCY_STATUS:
        reasons.append(
            f"residency_status must be {RESIDENCY_STATUS}; got {status!r}"
        )
    if isinstance(status, str) and status.upper() in SINGULARITY_STATUSES:
        reasons.append(f"residency_status {status!r} would mistake this 27B for a Singularity")
    reasons.extend(_unknown_required_broken(identity))
    hw = _hardware_numeric_keys(identity)
    for hit in hw:
        reasons.append(f"hardware field {hit} is numeric; sidecar has no GPU authority")
    blockers = _blockers_evident(identity)
    weaknesses = _as_list(identity.get("known_weaknesses"))
    if weaknesses is None:
        reasons.append("known_weaknesses is not a list")
    elif blockers and len(weaknesses) == 0:
        reasons.append(
            "known_weaknesses is empty while real blockers exist: " + ", ".join(blockers)
        )
    seen: set[str] = set()
    uniq: list[str] = []
    for item in reasons:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    if uniq:
        return {
            "status": "REJECTED",
            "reasons": uniq,
            "named_refusal": "REJECTED: " + "; ".join(uniq),
            "blockers_evident": blockers,
        }
    return {
        "status": "ACCEPTED",
        "reasons": [],
        "named_refusal": None,
        "blockers_evident": blockers,
    }


def accept(identity: Any) -> dict[str, Any]:
    result = validate(identity)
    if result["status"] == "REJECTED":
        raise IdentityRejectedError(tuple(result["reasons"]))
    return result


# ---------------------------------------------------------------------------
# Launch binding. Finding the receipt is not binding it.
# ---------------------------------------------------------------------------


def _sha256_ok(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _known_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text and text != UNKNOWN:
            return text
    return None


def resolve_incumbent() -> dict[str, Any]:
    """Who would actually be launched. Disk is authority; identity is not invented.

    SUCCESSION_TRIAL.json#incumbent.id first, sealed profile model_id second.
    A missing incumbent keeps the identity unbound.
    """
    src, doc = load_authority(SUCCESSION_TRIAL_REL)
    if isinstance(doc, dict):
        inc = doc.get("incumbent")
        if isinstance(inc, dict):
            iid = _known_str(inc.get("id"))
            if iid:
                return {
                    "id": iid,
                    "source": SUCCESSION_TRIAL_REL,
                    "load": src,
                    "residency_status": inc.get("residency_status"),
                    "artifact_path": inc.get("artifact_path"),
                    "role": inc.get("role"),
                }
    src, sealed = load_authority(SEALED_REL)
    if isinstance(sealed, dict):
        mid = _known_str(sealed.get("model_id"))
        if mid:
            return {
                "id": mid,
                "source": SEALED_REL,
                "load": src,
                "resident_identity": sealed.get("resident_identity"),
            }
    return {
        "id": None,
        "source": None,
        "load": "ABSENT",
        "missing": f"{SUCCESSION_TRIAL_REL}#incumbent.id and {SEALED_REL}#model_id",
    }


def _nx_pins(slot: Any) -> tuple[dict[str, Any] | None, str | None]:
    val = _slot_value(slot) if isinstance(slot, dict) else slot
    if isinstance(val, dict):
        model_id = _known_str(val.get("model_id"))
        if not model_id:
            return None, None
        return (
            {
                "model_id": model_id,
                "resident_identity": val.get("resident_identity"),
                "protocol": val.get("protocol"),
            },
            model_id,
        )
    model_id = _known_str(val)
    if model_id:
        return {"model_id": model_id}, model_id
    return None, None


def _executable_pins(slot: Any) -> dict[str, Any] | None:
    hashed: dict[str, str] = {}
    meta: dict[str, Any] = {}
    val = _slot_value(slot) if isinstance(slot, dict) else slot
    if isinstance(val, dict):
        by_role = val.get("by_role")
        if isinstance(by_role, dict):
            for role, digest in by_role.items():
                if _sha256_ok(digest):
                    hashed[str(role)] = digest
        if val.get("seal_matching_role") is not None:
            meta["seal_matching_role"] = val.get("seal_matching_role")
        if val.get("seal_declared_sha256_16") is not None:
            meta["seal_declared_sha256_16"] = val.get("seal_declared_sha256_16")
    if isinstance(slot, dict):
        for role in ("resident_binary", "binary"):
            rec = slot.get(role)
            if isinstance(rec, dict) and _sha256_ok(rec.get("sha256")):
                hashed.setdefault(role, rec["sha256"])
    if not hashed:
        return None
    return {"by_role": hashed, **meta}


def _artifact_root_pin(slot: Any) -> str | None:
    val = _slot_value(slot) if isinstance(slot, dict) else slot
    if isinstance(val, dict):
        return _known_str(val.get("artifact_root"))
    return _known_str(val)


def _tokenizer_pin(slot: Any) -> dict[str, Any] | None:
    val = _slot_value(slot) if isinstance(slot, dict) else slot
    path = None
    digest = None
    prefix_match = None
    if isinstance(val, dict):
        digest = val.get("sha256")
        path = val.get("path")
        prefix_match = val.get("declared_prefix_match")
    if not _sha256_ok(digest) and isinstance(slot, dict):
        digest = slot.get("sha256")
        path = path or slot.get("path")
        if prefix_match is None:
            prefix_match = slot.get("declared_prefix_match")
    if not _sha256_ok(digest):
        return None
    pin: dict[str, Any] = {"sha256": digest, "path": path}
    if prefix_match is not None:
        pin["declared_prefix_match"] = prefix_match
    return pin


def _qualification_pin(identity: Mapping[str, Any], *, validation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Qualification the body carries, not a physical pass.

    The capability receipt is CONTROL_HISTORICAL and identity_sufficient is
    false; that is a named qualification, not a missing one. A missing
    capability receipt is a missing pin.
    """
    cap_slot = identity.get("capability_receipt")
    cap = _slot_value(cap_slot) if isinstance(cap_slot, dict) else cap_slot
    if not isinstance(cap, dict):
        return None
    path = _known_str(cap.get("path"))
    label = _known_str(cap.get("label"))
    role = cap.get("role")
    if not path and not label and not role:
        return None
    pin: dict[str, Any] = {
        "path": path,
        "label": label,
        "role": role,
        "identity_sufficient": cap.get("identity_sufficient"),
        "residency_status": identity.get("residency_status"),
    }
    if validation is not None:
        pin["identity_validation"] = validation.get("status")
    return pin


def extract_pins(
    identity: Mapping[str, Any],
    *,
    validation: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return (pins, missing_field_names) for the launch-binding contract."""
    pins: dict[str, Any] = {}
    missing: list[str] = []

    nx, model_id = _nx_pins(identity.get("nx_id"))
    if nx is None or model_id is None:
        missing.append("nx_id")
    else:
        pins["nx_id"] = nx

    if model_id is None:
        missing.append("sealed_model_id")
    else:
        pins["sealed_model_id"] = model_id

    exe = _executable_pins(identity.get("executable_hash"))
    if exe is None:
        missing.append("executable_hash")
    else:
        pins["executable_hash"] = exe

    root = _artifact_root_pin(identity.get("parent_lineage"))
    if root is None:
        missing.append("artifact_root")
    else:
        pins["artifact_root"] = root

    tok = _tokenizer_pin(identity.get("tokenizer_identity"))
    if tok is None:
        missing.append("tokenizer")
    else:
        pins["tokenizer"] = tok

    qual = _qualification_pin(identity, validation=validation)
    if qual is None:
        missing.append("qualification")
    else:
        pins["qualification"] = qual

    return pins, missing


def _unbound_reason(
    *,
    missing: Sequence[str],
    agrees: bool,
    incumbent: Mapping[str, Any],
    sealed_model_id: Any,
    status: Any,
) -> str:
    parts: list[str] = []
    pin_missing = [m for m in missing if m not in {"incumbent", "identity_validation"}]
    if pin_missing:
        parts.append("found but does not pin " + ", ".join(pin_missing))
    if "incumbent" in missing or incumbent.get("id") is None:
        parts.append(
            "incumbent identity is not on disk "
            f"({SUCCESSION_TRIAL_REL} / {SEALED_REL})"
        )
    elif not agrees and _known_str(sealed_model_id):
        parts.append(
            f"sealed_model_id={sealed_model_id!r} does not agree with "
            f"incumbent={incumbent.get('id')!r}"
        )
    if status != "ACCEPTED":
        parts.append(f"status={status!r} (identity_validation is not ACCEPTED)")
    return "; ".join(parts) if parts else "found but unbound"


def describe_binding(
    identity: Any,
    *,
    receipt: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bound iff the document pins the resident that would be launched.

    bound is never true because the file exists. A missing pin is named.
    Status is identity_validation.status (ACCEPTED/REJECTED), never a
    silent null while the document carries residency_status.
    """
    if not isinstance(identity, Mapping):
        return {
            "bound": False,
            "status": "REJECTED",
            "residency_status": None,
            "pins": {},
            "pins_named": [],
            "missing": ["identity"],
            "incumbent": resolve_incumbent(),
            "agrees_with_incumbent": False,
            "unbound_reason": "identity is missing or not a mapping; identity is not invented",
        }
    if validation is None:
        validation = validate(identity)
    pins, missing = extract_pins(identity, validation=validation)
    incumbent = resolve_incumbent()
    sealed_model_id = pins.get("sealed_model_id")
    incumbent_id = incumbent.get("id")
    agrees = (
        isinstance(sealed_model_id, str)
        and isinstance(incumbent_id, str)
        and sealed_model_id == incumbent_id
    )
    if incumbent_id is None:
        missing.append("incumbent")
        agrees = False

    status = validation.get("status") if isinstance(validation, Mapping) else None
    if isinstance(receipt, Mapping):
        rec_val = receipt.get("identity_validation")
        if isinstance(rec_val, Mapping) and rec_val.get("status"):
            status = rec_val.get("status")
        elif _known_str(receipt.get("status")):
            status = receipt.get("status")
    if not _known_str(status):
        status = identity.get("residency_status")
    if status != "ACCEPTED":
        missing.append("identity_validation")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    missing_uniq: list[str] = []
    for item in missing:
        if item not in seen:
            seen.add(item)
            missing_uniq.append(item)
    missing = missing_uniq

    bound = (not missing) and agrees and status == "ACCEPTED"
    pins_named = [name for name in BIND_PIN_FIELDS if name in pins]
    return {
        "bound": bound,
        "status": status,
        "residency_status": identity.get("residency_status"),
        "pins": pins,
        "pins_named": pins_named,
        "missing": missing,
        "incumbent": incumbent,
        "agrees_with_incumbent": agrees,
        "unbound_reason": None
        if bound
        else _unbound_reason(
            missing=missing,
            agrees=agrees,
            incumbent=incumbent,
            sealed_model_id=sealed_model_id,
            status=status,
        ),
    }


def launch_binding(
    *,
    probe: Mapping[str, Any] | None = None,
    integration: str = "",
) -> dict[str, Any]:
    """Launch-receipt identity block. Missing receipt stays unbound; found is not bound."""
    note = (
        "Identity is not invented. A missing this-wave receipt stays unbound. "
        "Bound only when the document pins nx_id, sealed_model_id, executable_hash, "
        "artifact_root, tokenizer, and qualification, and sealed_model_id agrees "
        "with the incumbent. HCLI prior gates, if found, are cited and are not "
        "this identity."
    )
    if probe is None:
        target = RECEIPTS / RECEIPT
        if target.is_file():
            try:
                probe = {
                    "found": True,
                    "path_taken": "worktree",
                    "resolved": str(target),
                    "doc": load_json(target),
                }
            except (OSError, json.JSONDecodeError, UnicodeError) as exc:
                probe = {
                    "found": True,
                    "path_taken": "worktree",
                    "resolved": str(target),
                    "doc": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            src, doc = load_authority(f"receipts/future/{RECEIPT}")
            if isinstance(doc, dict):
                probe = {
                    "found": True,
                    "path_taken": src,
                    "resolved": f"{src}:receipts/future/{RECEIPT}",
                    "doc": doc,
                }
            else:
                probe = {
                    "found": False,
                    "path_taken": src,
                    "resolved": None,
                    "doc": None,
                }
    doc = probe.get("doc") if isinstance(probe.get("doc"), Mapping) else None
    base: dict[str, Any] = {
        "kind": "resident",
        "found": bool(probe.get("found") and doc is not None),
        "path_taken": probe.get("path_taken"),
        "resolved": probe.get("resolved"),
        "schema": None if not doc else doc.get("schema"),
        "integration_point": integration,
        "note": note,
    }
    if not base["found"]:
        return {
            **base,
            "bound": False,
            "status": None,
            "residency_status": None,
            "pins": {},
            "pins_named": [],
            "missing": ["receipt"],
            "incumbent": resolve_incumbent(),
            "agrees_with_incumbent": False,
            "unbound_reason": "identity receipt is missing; identity is not invented",
        }
    if doc.get("schema") != SCHEMA:
        status = doc.get("status") or (doc.get("identity_validation") or {}).get("status")
        return {
            **base,
            "bound": False,
            "status": status,
            "residency_status": doc.get("residency_status"),
            "pins": {},
            "pins_named": [],
            "missing": ["schema"],
            "incumbent": resolve_incumbent(),
            "agrees_with_incumbent": False,
            "unbound_reason": (
                f"found but schema is {doc.get('schema')!r}; want {SCHEMA}"
            ),
        }
    try:
        ident = identity_from_receipt(doc)
    except IdentityRejectedError as exc:
        return {
            **base,
            "bound": False,
            "status": (doc.get("identity_validation") or {}).get("status") or doc.get("status"),
            "residency_status": doc.get("residency_status"),
            "pins": {},
            "pins_named": [],
            "missing": ["identity"],
            "incumbent": resolve_incumbent(),
            "agrees_with_incumbent": False,
            "unbound_reason": f"found but identity is unreadable: {exc}",
        }
    described = describe_binding(ident, receipt=doc)
    return {**base, **described}


def work_unit() -> dict[str, Any]:
    return {
        "id": WORK_UNIT_ID,
        "role": "resident_identity",
        "description": (
            "Persist the incumbent HCLI resident identity to disk and prove "
            "restart recovery with no conversational reconstruction."
        ),
        "status": "complete",
        "species": "resident_identity_persistence",
        "resource_class": "cpu",
        "effect_class": "REVERSIBLE",
        "assigned_runtime": None,
        "preferred_backend": "cpu",
        "verifier": "tools/future/test_resident_identity.py",
        "output_receipt_path": f"receipts/future/{RECEIPT}",
        "claim_boundary": "STATIC_ONLY. No hardware measurement.",
        "requires_quiescence": False,
        "blocked_reason": None,
        "integration_point": (
            "tools/future/workgraph.py (this-wave sibling, not imported). "
            "This dict is the local WorkUnit shape until workgraph lands."
        ),
    }


def next_workunits() -> list[dict[str, Any]]:
    return [
        {
            "id": "future.resident-identity.refill-on-hardware",
            "status": "blocked",
            "species": "resident_identity_persistence",
            "description": (
                "Reload and refill identity when a protected GPU window qualifies; "
                "wake SLEEPING physical WorkUnits. Never synthesize a result."
            ),
            "blocked_reason": "no GPU authority; Metal compiler ABSENT; contamination HEAVY",
            "integration_point": "tools/future/wakeup.py (this-wave sibling, not imported)",
        },
        {
            "id": "future.resident-identity.bind-sandbox",
            "status": "pending",
            "species": "resident_identity_persistence",
            "description": "Replace local sandbox_authority projection with sandbox.py once that sibling lands.",
            "integration_point": "tools/future/sandbox.py (this-wave sibling, not imported)",
        },
        {
            "id": "future.resident-identity.bind-succession",
            "status": "pending",
            "species": "resident_identity_persistence",
            "description": "Bind children[] from succession.py once that sibling lands.",
            "integration_point": "tools/future/succession.py (this-wave sibling, not imported)",
        },
    ]


def recovered_implementation() -> dict[str, Any]:
    return {
        "incumbent_sealed_profile": {
            **presence(SEALED_REL),
            "role": "real shape of the Qwen27 resident as it exists today",
        },
        "runtime_identity_module": {
            **presence(RUNTIME_IDENTITY_PY_REL),
            "role": "historical vs current dispatch-fusion archaeology; not restart identity",
        },
        "resident_lifecycle": {
            **presence(RESIDENT_GATE_REL),
            "role": "live resident_gate; hcli/agentos/resident.py is not a git blob (gate is)",
        },
        "machine_prior": {
            **presence(MACHINE_PY_REL),
            "role": "genome is a PRIOR; STALE genomes are not used as live numbers",
        },
        "machine_genome_receipt": presence(MACHINE_GENOME_REL),
        "machine_genome_module": {
            **presence(MACHINE_GENOME_PY_REL),
            "role": "READ-ONLY; measure_bandwidth was not called",
        },
        "resident_install_contract": {
            "path": "tools/future/resident_install.py",
            "role": "generic tournament-winner install slots; not identity persistence",
        },
        "qwen27_profile_schema": {
            "path": PROFILE_SCHEMA_REL,
            "role": "CONTROL profile for optimizations; not a restart-survivable 19-field identity",
        },
        "hcli_resident_seal": presence(SEAL_REL),
        "capability_receipt": presence(CAPABILITY_REL),
        "codex_accelerator_handoff": {
            **presence(CODEX_HANDOFF_REL),
            "role": (
                "current_qwen27_incumbent_control_identity if present; this worktree "
                "copes with either ON_DISK or ABSENT and does not invent the block"
            ),
        },
        "fork_decision": (
            "No restart-survivable identity document existed. Sealed-3.14 + "
            "qwen27_runtime_identity + resident_install + profile schema are consumed, not forked."
        ),
        "gap_closed": (
            "Persist the full named identity to receipts/future/RESIDENT_IDENTITY.json; "
            "load() reconstructs it in a fresh process from disk alone."
        ),
    }


def gaps_closed() -> list[str]:
    return [
        "Named identity fields persisted to disk (tuple derived from the contract list, not a frozen count).",
        "UNKNOWN is mandatory for qualified physical EBPW, Flash EBPW, protected TPS/token_ns, and actual_read/transient bytes; missing evidence is named.",
        "load() recovers the full identity from disk in a fresh process with no in-memory state and no conversational input.",
        "known_weaknesses is populated from real scars and blockers; an empty list while blockers exist is REJECTED.",
        "residency_status is CURRENT_NONFINAL_HCLI_WORKER so nothing downstream mistakes 27B for a Singularity.",
        "Negative controls fire: invented measurement slot REJECTED; zero-weakness identity REJECTED; hardware numeric keys refused.",
        "Host tokenizer/binary hashed when present; UNKNOWN with the missing path named when not. Sparse checkout is not encoded as absence.",
        "Machine genome cited as a prior; measure_bandwidth was not invoked.",
        "Launch binding pins nx_id, sealed_model_id, executable_hash, artifact_root, tokenizer, and qualification against the succession incumbent; a missing pin stays unbound with the field named.",
    ]


def negative_findings() -> list[str]:
    return [
        "CODEX_ACCELERATOR_HANDOFF.json is not visible in this worktree disk or git HEAD; current_qwen27_incumbent_control_identity is therefore not copied. The collector copes if the file later appears.",
        "hcli/ is not materialized in this sparse checkout; sealed profile and machine.py were recovered via git show / load_authority.",
        "hcli/agentos/resident.py is not a git blob; resident_gate.py is the live lifecycle.",
        "CAPABILITY_noetic-sealed-3.14.json identity_sufficient is false (UNIDENTIFIED); 30/43 is CONTROL, not a body-identified current proof.",
        "Sealed profile current_runtime TPS numbers were not copied into this identity; copying them would launder a hardware claim.",
        "Qualified physical EBPW is UNKNOWN on this host; Flash physical_ebpw is NULL_BY_RULE.",
        "This sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE.",
        "tools/future/sandbox.py, succession.py, workgraph.py, wakeup.py, super_resident.py are this-wave siblings and were not imported.",
        "Protected bench flock is never taken; lock presence is pathlib only.",
        "Teacher capture 0/256 is cited from FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY, not invented.",
    ]


def resident_callable() -> dict[str, Any]:
    return {
        "entry_point": "python3 tools/future/resident_identity.py --build|--load|--selftest",
        "python_api": {
            "collect": "collect() -> dict",
            "accept": "accept(identity) -> dict  # raises IdentityRejectedError",
            "build": "build() -> Path",
            "load": "load(path=None) -> dict  # disk only, no collect()",
        },
        "work_unit_emitted": work_unit(),
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": (
            "Reads CLAUDE_GLOBAL_FRONTIER.json as disk authority. Does not mutate it "
            "(global_frontier.py is frozen). After restart, load() is how the resident "
            "knows which frontier entries remain BLOCKED without conversational reconstruction."
        ),
        "fail_closed": [
            "invented qualified EBPW / TPS / actual_read -> IdentityRejectedError naming the field",
            "empty known_weaknesses while blockers exist -> IdentityRejectedError",
            "residency_status other than CURRENT_NONFINAL_HCLI_WORKER -> IdentityRejectedError",
            "numeric HARDWARE_FIELDS -> HardwareClaimError from write_receipt",
            "missing/corrupt receipt on load -> IdentityRejectedError",
            "Singularity status -> IdentityRejectedError",
        ],
        "how_hcli_invokes": (
            "HCLI imports tools.future.resident_identity.load after restart, or "
            "runs the --load entry point. It must not reconstruct identity from chat."
        ),
        "integration_point_resident_api": (
            "tools/future/resident_api.py (this-wave sibling, not imported). "
            "load() is the swap-in implementation of 'who am I'."
        ),
    }


def build_document() -> dict[str, Any]:
    identity = collect()
    accepted = accept(identity)
    invented = dict(identity)
    invented["tps_token_ns_evidence"] = {
        "value": 24.4086,
        "missing_evidence": [],
        "evidence": [],
        "claim_class": CLAIM_CLASS,
    }
    invented_result = validate(invented)
    empty = dict(identity)
    empty["known_weaknesses"] = {
        "value": [],
        "missing_evidence": [],
        "evidence": [],
        "claim_class": CLAIM_CLASS,
    }
    empty_result = validate(empty)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Restart-survivable resident identity. Disk is authority. "
            "UNKNOWN where this sidecar cannot know. Models think; tools know; context is a cache."
        ),
        "residency_status": RESIDENCY_STATUS,
        "status": accepted["status"],
        "binding": describe_binding(identity, validation=accepted),
        "claim_class": CLAIM_CLASS,
        "gpu_authority": False,
        "nomenclature": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga": "Accelerator / Physical Compiler / Fusion; not its own civilization",
        },
        "persisted_fields": list(IDENTITY_FIELDS),
        "n_persisted_fields": len(IDENTITY_FIELDS),
        "identity": identity,
        "identity_validation": accepted,
        "work_unit": work_unit(),
        "next_workunits": next_workunits(),
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "negative_control": {
            "invented_tps": {
                "result": invented_result,
                "fires": invented_result["status"] == "REJECTED"
                and any("tps_token_ns_evidence" in r for r in invented_result["reasons"]),
            },
            "zero_weaknesses": {
                "result": empty_result,
                "fires": empty_result["status"] == "REJECTED"
                and any("known_weaknesses" in r for r in empty_result["reasons"]),
            },
        },
        "resident_callable": resident_callable(),
    }


def build() -> Path:
    doc = build_document()
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def identity_from_receipt(doc: Mapping[str, Any]) -> dict[str, Any]:
    ident = doc.get("identity")
    if not isinstance(ident, dict):
        raise IdentityRejectedError(("receipt.identity is missing or not a mapping",))
    return ident


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Recover the full identity from disk. No collect(), no conversational input."""
    target = Path(path) if path is not None else (RECEIPTS / RECEIPT)
    if not target.is_file():
        raise IdentityRejectedError((f"identity receipt not on disk: {target}",))
    try:
        doc = load_json(target)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise IdentityRejectedError((f"identity receipt unreadable: {type(exc).__name__}: {exc}",)) from exc
    if not isinstance(doc, dict):
        raise IdentityRejectedError(("identity receipt is not a JSON object",))
    if doc.get("schema") != SCHEMA:
        raise IdentityRejectedError((f"unexpected schema {doc.get('schema')!r}; want {SCHEMA}",))
    ident = identity_from_receipt(doc)
    accept(ident)
    return ident


def selftest() -> Path:
    ident = collect()
    accept(ident)
    invented = dict(ident)
    invented["ebpw"] = {"value": 3.1393, "missing_evidence": [], "evidence": []}
    invented_result = validate(invented)
    if invented_result["status"] != "REJECTED":
        raise IdentityRejectedError(("selftest: invented ebpw was not rejected",))
    empty = dict(ident)
    empty["known_weaknesses"] = []
    empty_result = validate(empty)
    if empty_result["status"] != "REJECTED":
        raise IdentityRejectedError(("selftest: empty weaknesses was not rejected",))
    return build()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.load:
        ident = load()
        print(json.dumps(ident, indent=2, sort_keys=True, default=str))
        return 0
    if args.selftest:
        out = selftest()
        print(out)
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
