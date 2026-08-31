"""Continuous complete Flash NR + seven strictly separated EBPW quantities.

Extends ``tools/future/ebpw_categories.py`` (five landed types, can_promote
false for prospective-only). This module does not fork that type system:
new quantities subclass ``Quantity`` so mixed arithmetic/assignment is a
``CategoryError``, and ``can_promote`` on prospective_meta_bpw alone stays
false.

A complete heterogeneous Flash NR candidate exists continuously: every
organ in the live census has either the best-known runtime-ready candidate
or an exact-control fallback marked COMPILE_TIME_SCIENCE_ONLY, with
promotion status derived from the organs (never asserted at the NR root).

0.887 prospective meta-BPW is a RESEARCH_TARGET. It is not a representation
and not a claim. qualified_complete_physical_ebpw is UNKNOWN on this host.

    python3 tools/future/flash_nr_complete.py --build
    python3 -m pytest tools/future/test_flash_nr_complete.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping

from tools.future import ebpw_categories as _ebpw
from tools.future._common import git
from tools.future.ebpw_categories import (
    CategoryError,
    CompletePhysicalEbpw,
    PRODUCTION,
    PROTECTED,
    ProspectiveMetaBpw,
    Quantity,
    VERIFICATION,
    judge_dense_rematerialization,
)

RECEIPT = "FLASH_NR_COMPLETE.json"
SCHEMA = "hawking.future.flash_nr_complete.v1"
VERSION = 1
RECORDED_BY = "tools/future/flash_nr_complete.py"

RESEARCH_TARGET = "RESEARCH_TARGET"
SCIENCE_ONLY = "COMPILE_TIME_SCIENCE_ONLY"
EXACT_CONTROL = "exact_control"
EXACT_REPR = "source_bf16_exact"
BEST_KNOWN = "BEST_KNOWN_CANDIDATE"
EXACT_FALLBACK = "EXACT_CONTROL_FALLBACK"
NOT_PROMOTABLE = "NOT_PROMOTABLE"
PROMOTABLE = "PROMOTABLE"
UNKNOWN = "UNKNOWN"

PINNED = REPO / "receipts" / "future" / "evidence"
HEADLESS = REPO / "receipts" / "headless"

# Pinned snapshot first (sparse-safe), then live Codex receipts, then the
# primary worktree. Missing is recorded; it is not proof the file is gone.
EVIDENCE = {
    "census": "FLASH_ORGAN_CENSUS.json",
    "nr_v2": "FLASH_COMPLETE_V2.nr.json",
    "nx_v0": "FLASH_COMPLETE_V0.nx.json",
    "nx_next": "FLASH_NEXT_MACHINE.nx.json",
    "meta": "FLASH_META_REPRESENTATION_SUB1.json",
    "ebpw": "FLASH_EBPW_BUDGET.json",
    "ledger": "FLASH_COMPLETE_V0.BYTE_LEDGER.json",
}

HANDOFF_REL = "CODEX_ACCELERATOR_HANDOFF.json"
FRONTIER_REL = "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"

# Map EBPW-budget organ names onto census families when they are the same
# population under a different slicing. Unmapped names stay extra organs.
EBPW_TO_FAMILY: dict[str, str] = {
    "embeddings": "embedding_lm_head",
    "lm_head": "embedding_lm_head",
    "deltanet": "linear_attention_hyperconnection",
    "recurrent_state": "linear_attention_hyperconnection",
    "sparse_attention": "full_attention",
    "router": "routed_experts",
    "routed_experts": "routed_experts",
    "shared_expert": "shared_expert",
    "ngram_engine": "ngram_embedding",
    "residual_hyperconnections": "mlp_hyperconnection",
    "support_misc": "other",
    "mtp": "other",
    "vision_backbone": "other",
}


class IncompleteNrError(ValueError):
    """A census organ has no candidate and no exact-control fallback."""


class NrBillingError(ValueError):
    """NX references NR at runtime but NR bytes are not in the complete ledger."""


class DenseRematError(ValueError):
    """Production path reconstructs a dense checkpoint."""


# ---------------------------------------------------------------------------
# Seven quantities. Assignment across them is a type error.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SourceControlEbpw(Quantity):
    """BF16 identity of the source. COMPILE_TIME_SCIENCE_ONLY."""

    category: ClassVar[str] = "source_control_ebpw"
    unit: ClassVar[str] = (
        "bits per source weight of the BF16 exact-control identity "
        "(COMPILE_TIME_SCIENCE_ONLY; not a packed artifact; not physical)"
    )


@dataclass(frozen=True, eq=False)
class StaticActiveEbpwEstimate(Quantity):
    """Header-derived active-path EBPW. STATIC_ONLY. Never promotes."""

    category: ClassVar[str] = "static_active_ebpw_estimate"
    unit: ClassVar[str] = (
        "STATIC estimate of active-path bits per source weight from census/"
        "budget headers; not a protected measurement"
    )


@dataclass(frozen=True, eq=False)
class StaticCompleteEbpwEstimate(Quantity):
    """Header-derived complete-system EBPW. STATIC_ONLY. Never promotes."""

    category: ClassVar[str] = "static_complete_ebpw_estimate"
    unit: ClassVar[str] = (
        "STATIC estimate of complete-system bits per source weight from "
        "indexed header payload; not a protected measurement"
    )


@dataclass(frozen=True, eq=False)
class SerializedNrInformation(Quantity):
    """Serialized NR information (bytes). Not source-control EBPW."""

    category: ClassVar[str] = "serialized_nr_information"
    unit: ClassVar[str] = (
        "bytes of a packed NR information payload (not a composition "
        "document; not source-control EBPW)"
    )


@dataclass(frozen=True, eq=False)
class SerializedNxEbpw(Quantity):
    """EBPW of a serialized NX artifact. Not qualified physical EBPW."""

    category: ClassVar[str] = "serialized_nx_ebpw"
    unit: ClassVar[str] = (
        "bits per weight of a serialized NX artifact; not "
        "qualified_complete_physical_ebpw"
    )


@dataclass(frozen=True, eq=False)
class QualifiedCompletePhysicalEbpw(Quantity):
    """The only quantity that can support a promotion claim. UNKNOWN here."""

    category: ClassVar[str] = "qualified_complete_physical_ebpw"
    unit: ClassVar[str] = (
        "bits per weight of a complete executable from PROTECTED_ABSOLUTE "
        "measurement — never from a description budget or a STATIC estimate"
    )


# ProspectiveMetaBpw is the landed type. Reused, not forked.
SEVEN_TYPES: dict[str, type[Quantity]] = {
    SourceControlEbpw.category: SourceControlEbpw,
    StaticActiveEbpwEstimate.category: StaticActiveEbpwEstimate,
    StaticCompleteEbpwEstimate.category: StaticCompleteEbpwEstimate,
    ProspectiveMetaBpw.category: ProspectiveMetaBpw,
    SerializedNrInformation.category: SerializedNrInformation,
    SerializedNxEbpw.category: SerializedNxEbpw,
    QualifiedCompletePhysicalEbpw.category: QualifiedCompletePhysicalEbpw,
}


def bind(cls: type[Quantity], value: Any, *, evidence: str = "bind") -> Quantity | None:
    """Place a value into a quantity type. Cross-type input is a type error."""
    if value is None:
        return None
    if isinstance(value, cls):
        return value
    if isinstance(value, Quantity):
        raise CategoryError(
            f"type error: cannot assign {value.category} to {cls.category}: "
            "EBPW categories are not interchangeable"
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CategoryError(
            f"type error: cannot assign {type(value).__name__} to {cls.category}"
        )
    return cls(float(value), evidence=evidence)


class SevenLedger:
    """Typed holder for the seven quantities. Field assignment is type-checked."""

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "_values", {name: None for name in SEVEN_TYPES})
        for name, value in kwargs.items():
            self.__setattr__(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_values":
            object.__setattr__(self, name, value)
            return
        if name not in SEVEN_TYPES:
            raise CategoryError(f"type error: {name!r} is not one of the seven EBPW quantities")
        cls = SEVEN_TYPES[name]
        if value is None:
            self._values[name] = None
            return
        if type(value) is not cls:
            other = getattr(value, "category", type(value).__name__)
            raise CategoryError(
                f"type error: cannot assign {other} to {cls.category}: "
                "EBPW categories are not interchangeable"
            )
        self._values[name] = value

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in SEVEN_TYPES:
            return values.get(name)
        raise AttributeError(name)

    def get(self, name: str) -> Quantity | None:
        return self._values.get(name)

    def present_names(self) -> list[str]:
        out = []
        for name, qty in self._values.items():
            if qty is not None and qty.value is not None:
                out.append(name)
        return out

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, cls in SEVEN_TYPES.items():
            qty = self._values.get(name)
            if qty is not None and qty.value is not None:
                row = qty.as_dict()
                row["present"] = True
            else:
                row = {
                    "category": name,
                    "unit": cls.unit,
                    "present": False,
                    "value": None,
                    "evidence": qty.evidence if qty is not None else "UNKNOWN",
                    "source": qty.source if qty is not None else None,
                }
            if name == ProspectiveMetaBpw.category:
                row["role"] = RESEARCH_TARGET
            if name == QualifiedCompletePhysicalEbpw.category:
                row["state"] = UNKNOWN
                row["gpu_authority"] = False
            out[name] = row
        return out


# ---------------------------------------------------------------------------
# Evidence. Sparse checkouts are not absences.
# ---------------------------------------------------------------------------


def _checkout_roots() -> list[Path]:
    roots: list[Path] = [REPO]
    common = git("rev-parse", "--git-common-dir")
    if common:
        path = Path(common)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        else:
            path = path.resolve()
        parent = path.parent if path.name == ".git" else path.parent
        if parent not in roots and parent.is_dir():
            roots.append(parent)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def resolve_evidence(filename: str) -> tuple[dict[str, Any] | None, str, str | None]:
    """Load a receipt. Records which path was taken. Missing is not absence."""
    candidates: list[tuple[str, Path]] = [
        ("pinned", PINNED / filename),
        ("headless", HEADLESS / filename),
    ]
    for root in _checkout_roots():
        if root == REPO:
            continue
        candidates.append(("primary_worktree", root / "receipts" / "headless" / filename))
        candidates.append(("primary_worktree", root / "receipts" / "future" / "evidence" / filename))
    seen: set[str] = set()
    for via, path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return load_json(path), via, str(path)
    rel_headless = f"receipts/headless/{filename}"
    blob = git("show", f"HEAD:{rel_headless}")
    if blob:
        try:
            return json.loads(blob), "git:HEAD", rel_headless
        except json.JSONDecodeError:
            pass
    return None, "missing", None


def resolve_handoff() -> tuple[dict[str, Any] | None, str]:
    candidates = [REPO / HANDOFF_REL]
    for root in _checkout_roots():
        candidates.append(root / HANDOFF_REL)
    for path in candidates:
        if path.is_file():
            via = "local" if path.parent == REPO else "primary_worktree"
            return load_json(path), via
    return None, "missing"


def load_docs() -> dict[str, Any]:
    docs: dict[str, Any] = {"resolution": {}}
    for key, filename in EVIDENCE.items():
        doc, via, resolved = resolve_evidence(filename)
        docs[key] = doc
        docs["resolution"][key] = {
            "filename": filename,
            "present": doc is not None,
            "resolved_via": via,
            "resolved": resolved,
        }
    handoff, h_via = resolve_handoff()
    docs["handoff"] = handoff
    docs["resolution"]["handoff"] = {
        "filename": HANDOFF_REL,
        "present": handoff is not None,
        "resolved_via": h_via,
    }
    return docs


def _dot(node: Any, dotted: str, default: Any = None) -> Any:
    cur = node
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _nullish(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str) and value.strip().upper() in {
        "NULL",
        "NULL_BY_RULE",
        "NOT_BUILT",
        "NOT_MEASURED",
        "NOT_TESTED",
        "UNKNOWN",
        "ABSENT",
        "NONE",
        "",
    }:
        return True
    return False


def _status_is_metadata_only(nx: Mapping[str, Any]) -> bool:
    status = str(nx.get("status") or "")
    return "METADATA_ONLY" in status or "NOT_FOR_PROMOTION" in status


# ---------------------------------------------------------------------------
# Continuous complete NR.
# ---------------------------------------------------------------------------


def census_families(census: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(census, Mapping):
        return []
    rows = [r for r in (census.get("family_summary") or []) if isinstance(r, dict) and r.get("family")]
    return rows


def ebpw_organs(ebpw: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ebpw, Mapping):
        return []
    return [r for r in (ebpw.get("organs") or []) if isinstance(r, dict) and r.get("organ")]


def required_organs(docs: Mapping[str, Any]) -> list[str]:
    """Union of census families and (if present) EBPW-budget organ names.

    Count is derived from disk. Never hard-coded.
    """
    names: list[str] = []
    seen: set[str] = set()
    for fam in census_families(docs.get("census")):
        name = str(fam["family"])
        if name not in seen:
            seen.add(name)
            names.append(name)
    for org in ebpw_organs(docs.get("ebpw")):
        name = str(org["organ"])
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _nr_parts(nr: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(nr, Mapping):
        return {}
    parts = _dot(nr, "representation.parts") or []
    out: dict[str, dict[str, Any]] = {}
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and part.get("family"):
                out[str(part["family"])] = part
    return out


def _nr_variants(nr: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(nr, Mapping):
        return []
    rows = _dot(nr, "representation.candidate_variants") or []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _family_for_required(name: str, census_names: set[str]) -> str:
    if name in census_names:
        return name
    mapped = EBPW_TO_FAMILY.get(name)
    if mapped:
        return mapped
    return name


def _variants_for_family(family: str, variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Family-matching variants, plus the global exact_control row."""
    key = family.lower().replace("-", "_")
    matched: list[dict[str, Any]] = []
    exact: dict[str, Any] | None = None
    for row in variants:
        name = str(row.get("name") or "")
        if name == EXACT_CONTROL:
            exact = row
            continue
        lowered = name.lower()
        if key in lowered or any(
            token in lowered
            for token in {
                "ngram" if "ngram" in key else "",
                "expert" if "expert" in key or "routed" in key else "",
            }
            if token
        ):
            matched.append(row)
    if exact is not None:
        matched.append(exact)
    return matched


def _best_runtime_ready(family: str, variants: list[dict[str, Any]]) -> dict[str, Any] | None:
    ready = [
        v
        for v in _variants_for_family(family, variants)
        if v.get("runtime_ready") is True and str(v.get("name") or "") != EXACT_CONTROL
    ]
    ready.sort(key=lambda v: str(v.get("name") or ""))
    return ready[0] if ready else None


def _exact_control_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    for row in variants:
        if row.get("name") == EXACT_CONTROL:
            return row
    return {
        "name": EXACT_CONTROL,
        "complete_bits_per_weight": 16.0,
        "runtime_ready": True,
        "capability_status": "source-control-only",
    }


def _meta_target_for(family: str, meta: Mapping[str, Any] | None) -> float | None:
    if not isinstance(meta, Mapping):
        return None
    for row in meta.get("family_budget") or []:
        if isinstance(row, dict) and row.get("family") == family:
            return _as_number(row.get("meta_bpw_target"))
    return None


def derive_organ_slot(
    organ: str,
    *,
    family: str,
    source_bytes: int | None,
    nr: Mapping[str, Any] | None,
    meta: Mapping[str, Any] | None,
    ebpw_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parts = _nr_parts(nr)
    variants = _nr_variants(nr)
    part = parts.get(family) or {}
    exact = _exact_control_variant(variants)
    compact = _best_runtime_ready(family, variants)
    part_repr = str(part.get("representation") or EXACT_REPR)
    best_known_name = part_repr if part_repr != EXACT_REPR else EXACT_CONTROL
    if compact is not None:
        occupying_kind = BEST_KNOWN
        occupying = compact
        science_mark = None
        occupying_repr = str(compact.get("name"))
    else:
        occupying_kind = EXACT_FALLBACK
        occupying = exact
        science_mark = SCIENCE_ONLY
        occupying_repr = EXACT_REPR

    reasons: list[str] = []
    if science_mark == SCIENCE_ONLY:
        reasons.append("occupying candidate is exact-control COMPILE_TIME_SCIENCE_ONLY")
    if compact is None and part_repr != EXACT_REPR:
        reasons.append(
            f"best-known part representation is {part_repr} but no runtime_ready compact variant occupies the slot"
        )
    cap = occupying.get("capability_status")
    if cap not in {"PASSED", "passed"}:
        reasons.append(f"capability_status={cap!r} is not PASSED")
    if occupying.get("runtime_ready") is not True and occupying_kind == BEST_KNOWN:
        reasons.append("occupying candidate is not runtime_ready")
    reasons.append("qualified_complete_physical_ebpw is UNKNOWN on this host")
    promotion_status = NOT_PROMOTABLE
    return {
        "organ": organ,
        "census_family": family,
        "source_bytes": source_bytes,
        "occupying": {
            "kind": occupying_kind,
            "name": occupying.get("name"),
            "representation": occupying_repr,
            "runtime_ready": occupying.get("runtime_ready"),
            "capability_status": occupying.get("capability_status"),
            "complete_bits_per_weight": occupying.get("complete_bits_per_weight"),
            "science_mark": science_mark,
        },
        "best_known": {
            "name": best_known_name,
            "part_representation": part_repr,
            "qualification": part.get("qualification"),
            "runtime_required": part.get("runtime_required"),
            "runtime_ready_compact": compact.get("name") if compact else None,
        },
        "prospective_meta_bpw_target": {
            "value": _meta_target_for(family, meta),
            "role": RESEARCH_TARGET,
            "promotes": False,
        },
        "ebpw_budget_row": (
            {
                "organ": ebpw_row.get("organ"),
                "representation_status": ebpw_row.get("representation_status"),
                "native_kernel_status": ebpw_row.get("native_kernel_status"),
                "native_loader_status": ebpw_row.get("native_loader_status"),
                "capability_status": ebpw_row.get("capability_status"),
                "source_active_bytes_per_token": ebpw_row.get("source_active_bytes_per_token"),
            }
            if isinstance(ebpw_row, Mapping)
            else None
        ),
        "promotion_status": promotion_status,
        "promotion_reasons": reasons,
    }


def build_continuous_nr(docs: Mapping[str, Any]) -> dict[str, Any]:
    census = docs.get("census")
    nr = docs.get("nr_v2")
    meta = docs.get("meta")
    ebpw = docs.get("ebpw")
    families = census_families(census if isinstance(census, Mapping) else None)
    if not families:
        raise IncompleteNrError(
            "FLASH_ORGAN_CENSUS family_summary is empty or unreadable; "
            "a complete NR cannot be claimed"
        )
    family_bytes = {
        str(row["family"]): _as_number(row.get("bytes")) for row in families
    }
    census_names = {str(row["family"]) for row in families}
    ebpw_by_name = {str(row["organ"]): row for row in ebpw_organs(ebpw if isinstance(ebpw, Mapping) else None)}
    ebpw_by_family: dict[str, dict[str, Any]] = {}
    for name, row in ebpw_by_name.items():
        fam = EBPW_TO_FAMILY.get(name, name)
        ebpw_by_family.setdefault(fam, row)

    required = required_organs(docs)
    slots: list[dict[str, Any]] = []
    for name in required:
        family = _family_for_required(name, census_names)
        source_bytes = family_bytes.get(family)
        if source_bytes is None and name in ebpw_by_name:
            source_bytes = _as_number(ebpw_by_name[name].get("source_bytes"))
        ebpw_row = ebpw_by_name.get(name) or ebpw_by_family.get(family)
        slots.append(
            derive_organ_slot(
                name,
                family=family,
                source_bytes=int(source_bytes) if source_bytes is not None else None,
                nr=nr if isinstance(nr, Mapping) else None,
                meta=meta if isinstance(meta, Mapping) else None,
                ebpw_row=ebpw_row,
            )
        )

    present = {s["organ"] for s in slots}
    missing = [name for name in required if name not in present]
    if missing:
        raise IncompleteNrError(f"NR missing organs {missing}")

    # Completeness is derived: every required organ has an occupying candidate.
    occupying_ok = all(
        s.get("occupying") and s["occupying"].get("name") for s in slots
    )
    complete = occupying_ok and not missing
    organ_statuses = [s["promotion_status"] for s in slots]
    if not complete:
        nr_status = "INCOMPLETE"
        nr_reason = "one or more required organs have no occupying candidate"
    elif organ_statuses and all(st == PROMOTABLE for st in organ_statuses):
        nr_status = PROMOTABLE
        nr_reason = "all organs PROMOTABLE"
    else:
        nr_status = NOT_PROMOTABLE
        nr_reason = (
            "NR promotion_status is derived from per-organ status; "
            f"{sum(1 for st in organ_statuses if st != PROMOTABLE)} of "
            f"{len(organ_statuses)} organs are not PROMOTABLE"
        )

    science_marks = sorted(
        {
            s["occupying"]["science_mark"]
            for s in slots
            if s.get("occupying") and s["occupying"].get("science_mark")
        }
    )
    return {
        "complete": complete,
        "organ_count": len(slots),
        "required_organs": list(required),
        "census_family_count": len(families),
        "ebpw_organ_count": len(ebpw_by_name),
        "organs": slots,
        "promotion_status": nr_status,
        "promotion_reason": nr_reason,
        "science_marks_present": science_marks,
        "nr_v2_status": (nr or {}).get("status") if isinstance(nr, Mapping) else None,
        "nr_v2_schema": (nr or {}).get("schema") if isinstance(nr, Mapping) else None,
        "claim_boundary": (
            "Complete heterogeneous portable NR candidate with exact-control "
            "fallback per organ. COMPILE_TIME_SCIENCE_ONLY. Does not claim a "
            "compact complete representation, accepted-token TPS, physical EBPW, "
            "capability preservation, or machine execution."
        ),
    }


def assert_complete(nr: Mapping[str, Any], required: Iterable[str]) -> None:
    present = {s["organ"] for s in (nr.get("organs") or []) if isinstance(s, dict)}
    missing = [name for name in required if name not in present]
    if missing:
        raise IncompleteNrError(f"NR missing organs {missing}")
    if nr.get("complete") is not True:
        raise IncompleteNrError("NR is not marked complete")


# ---------------------------------------------------------------------------
# Seven quantities from disk.
# ---------------------------------------------------------------------------


def _prospective_from_docs(docs: Mapping[str, Any]) -> ProspectiveMetaBpw:
    meta = docs.get("meta") if isinstance(docs.get("meta"), Mapping) else None
    handoff = docs.get("handoff") if isinstance(docs.get("handoff"), Mapping) else None
    value = None
    source = None
    if meta is not None:
        value = _as_number(_dot(meta, "metric.prospective_target"))
        source = "FLASH_META_REPRESENTATION_SUB1.metric.prospective_target"
    if value is None and handoff is not None:
        value = _as_number(handoff.get("current_prospective_meta_bpw"))
        source = "CODEX_ACCELERATOR_HANDOFF.current_prospective_meta_bpw"
    if value is None:
        value = _as_number(_dot(_ebpw.HONEST_FLASH_META_MINIMAL, "metric.prospective_target"))
        source = "ebpw_categories.HONEST_FLASH_META_MINIMAL"
    return ProspectiveMetaBpw(
        value,
        evidence=(
            f"{RESEARCH_TARGET}; description budget, not a representation, "
            "not a claim, never promotes alone"
        ),
        source=source,
    )


def _source_control_from_docs(docs: Mapping[str, Any]) -> SourceControlEbpw:
    nr = docs.get("nr_v2") if isinstance(docs.get("nr_v2"), Mapping) else None
    value = None
    source = None
    if nr is not None:
        value = _as_number(_dot(nr, "representation.complete_bits_per_weight"))
        source = "FLASH_COMPLETE_V2.nr.json representation.complete_bits_per_weight"
    if value is None:
        value = 16.0
        source = "BF16 identity default for exact-control (no NR on disk)"
    return SourceControlEbpw(
        value,
        evidence=(
            f"{SCIENCE_ONLY}; BF16 exact-control identity of the source; "
            "not qualified_complete_physical_ebpw"
        ),
        source=source,
    )


def _static_complete_from_docs(docs: Mapping[str, Any]) -> StaticCompleteEbpwEstimate:
    census = docs.get("census") if isinstance(docs.get("census"), Mapping) else None
    nr = docs.get("nr_v2") if isinstance(docs.get("nr_v2"), Mapping) else None
    src_bytes = _as_number((census or {}).get("source_parameter_bytes_indexed"))
    n_params = _as_number(_dot(nr or {}, "semantic_provenance.parameter_count"))
    if src_bytes is None or n_params is None or n_params <= 0:
        return StaticCompleteEbpwEstimate(
            None,
            evidence="census/NR parameter denominator unavailable; STATIC estimate withheld",
        )
    value = 8.0 * src_bytes / n_params
    return StaticCompleteEbpwEstimate(
        value,
        evidence=(
            f"STATIC estimate: 8 * source_parameter_bytes_indexed={int(src_bytes)} "
            f"/ n_params={int(n_params)}; header-derived; not a measurement"
        ),
        source="FLASH_ORGAN_CENSUS.source_parameter_bytes_indexed",
    )


def _static_active_from_docs(docs: Mapping[str, Any]) -> StaticActiveEbpwEstimate:
    ebpw = docs.get("ebpw") if isinstance(docs.get("ebpw"), Mapping) else None
    nr = docs.get("nr_v2") if isinstance(docs.get("nr_v2"), Mapping) else None
    n_params = _as_number(_dot(nr or {}, "semantic_provenance.parameter_count"))
    organs = ebpw_organs(ebpw)
    if not organs or n_params is None or n_params <= 0:
        return StaticActiveEbpwEstimate(
            None,
            evidence=(
                "FLASH_EBPW_BUDGET organs or parameter count unavailable in this "
                "checkout; STATIC estimate withheld rather than invented"
            ),
            source=docs.get("resolution", {}).get("ebpw", {}).get("resolved_via"),
        )
    total = 0.0
    n = 0
    for row in organs:
        v = _as_number(row.get("source_active_bytes_per_token"))
        if v is not None:
            total += v
            n += 1
    if n == 0:
        return StaticActiveEbpwEstimate(
            None,
            evidence="no source_active_bytes_per_token on EBPW organs",
        )
    value = 8.0 * total / n_params
    return StaticActiveEbpwEstimate(
        value,
        evidence=(
            f"STATIC estimate: 8 * sum(source_active_bytes_per_token)={total} "
            f"/ n_params={int(n_params)} over {n} EBPW organs; not a measurement"
        ),
        source="FLASH_EBPW_BUDGET.json organs[].source_active_bytes_per_token",
    )


def _serialized_nr_from_docs(docs: Mapping[str, Any]) -> SerializedNrInformation:
    res = (docs.get("resolution") or {}).get("nr_v2") or {}
    path = res.get("resolved")
    # The on-disk NR is a composition document, not a packed organ payload.
    return SerializedNrInformation(
        None,
        evidence=(
            "packed NR information payload is NOT_BUILT; "
            f"composition document resolved_via={res.get('resolved_via')} "
            f"path={path}; document bytes are not serialized_nr_information"
        ),
        source=path,
    )


def _serialized_nx_from_docs(docs: Mapping[str, Any]) -> SerializedNxEbpw:
    nx = docs.get("nx_v0") if isinstance(docs.get("nx_v0"), Mapping) else None
    if nx is None:
        return SerializedNxEbpw(
            None,
            evidence="FLASH_COMPLETE_V0.nx.json unavailable in this checkout",
        )
    claimed = _dot(nx, "qualification.complete_system_ebpw")
    if _nullish(claimed):
        return SerializedNxEbpw(
            None,
            evidence=(
                f"NX status={nx.get('status')}; qualification.complete_system_ebpw "
                "is null; metadata seal is not serialized_nx_ebpw"
            ),
            source="FLASH_COMPLETE_V0.nx.json",
        )
    # A number here would still not be qualified physical EBPW.
    return SerializedNxEbpw(
        _as_number(claimed),
        evidence="qualification.complete_system_ebpw on a metadata NX (not qualified physical)",
        source="FLASH_COMPLETE_V0.nx.json",
    )


def _qualified_physical_from_docs(docs: Mapping[str, Any]) -> QualifiedCompletePhysicalEbpw:
    handoff = docs.get("handoff") if isinstance(docs.get("handoff"), Mapping) else None
    evidence = (
        "UNKNOWN; this sidecar has no GPU authority; MetalContext reports no "
        "Metal-capable GPU on the host of record; no PROTECTED_ABSOLUTE "
        "complete-token measurement exists for a source-independent Flash NX"
    )
    source = None
    if handoff is not None:
        raw = handoff.get("current_qualified_physical_ebpw")
        source = "CODEX_ACCELERATOR_HANDOFF.current_qualified_physical_ebpw"
        if not _nullish(raw) and _as_number(raw) is not None:
            # A numeric claim in the handoff would still not be ours to echo
            # as a measurement this sidecar took. Record UNKNOWN.
            evidence = (
                f"handoff current_qualified_physical_ebpw={raw!r}; sidecar "
                "refuses to copy a physical EBPW it did not measure"
            )
        else:
            evidence = (
                f"handoff current_qualified_physical_ebpw={raw!r}; "
                + evidence
            )
    return QualifiedCompletePhysicalEbpw(None, evidence=evidence, source=source)


def seven_from_docs(docs: Mapping[str, Any]) -> SevenLedger:
    return SevenLedger(
        source_control_ebpw=_source_control_from_docs(docs),
        static_active_ebpw_estimate=_static_active_from_docs(docs),
        static_complete_ebpw_estimate=_static_complete_from_docs(docs),
        prospective_meta_bpw=_prospective_from_docs(docs),
        serialized_nr_information=_serialized_nr_from_docs(docs),
        serialized_nx_ebpw=_serialized_nx_from_docs(docs),
        qualified_complete_physical_ebpw=_qualified_physical_from_docs(docs),
    )


# ---------------------------------------------------------------------------
# Promotion. prospective_meta_bpw alone never promotes.
# ---------------------------------------------------------------------------


def _present(qty: Quantity | None) -> bool:
    return qty is not None and qty.value is not None


def can_promote(obj: Any) -> tuple[bool, str]:
    """False, with the reason, unless every promotion predicate holds.

    ``prospective_meta_bpw`` (including 0.887) is RESEARCH_TARGET and never
    promotes alone. Not with a caveat, not with a flag.
    """
    if isinstance(obj, ProspectiveMetaBpw):
        return False, (
            "prospective_meta_bpw is RESEARCH_TARGET and never promotes alone; "
            "0.887 is a description budget, not a representation or a claim"
        )
    if isinstance(obj, SevenLedger):
        return _can_promote_seven(obj, extra={})
    if isinstance(obj, _ebpw.PromotionLedger):
        ok, reason = _ebpw.can_promote(obj)
        if (
            _present(obj.prospective_meta_bpw)
            and obj.prospective_meta_bpw is not None
            and obj.prospective_meta_bpw.value is not None
            and obj.prospective_meta_bpw.value < 1.0
            and not ok
        ):
            if RESEARCH_TARGET.lower() not in reason.lower() and "never promotes alone" in reason:
                reason = f"prospective_meta_bpw is {RESEARCH_TARGET}; " + reason
        return ok, reason
    if isinstance(obj, Mapping):
        return _can_promote_mapping(obj)
    raise CategoryError(f"can_promote does not accept {type(obj).__name__}")


def _can_promote_mapping(doc: Mapping[str, Any]) -> tuple[bool, str]:
    keys = [k for k, v in doc.items() if v is not None and k in SEVEN_TYPES]
    if keys == [ProspectiveMetaBpw.category] or set(keys) <= {ProspectiveMetaBpw.category}:
        if ProspectiveMetaBpw.category in doc:
            return False, (
                "prospective_meta_bpw is RESEARCH_TARGET and never promotes alone; "
                "0.887 is a description budget, not a representation or a claim"
            )
    ledger = SevenLedger()
    for name, cls in SEVEN_TYPES.items():
        if name not in doc:
            continue
        raw = doc[name]
        if isinstance(raw, cls):
            setattr(ledger, name, raw)
        elif isinstance(raw, Quantity):
            raise CategoryError(
                f"type error: cannot assign {raw.category} to {name}"
            )
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            setattr(ledger, name, cls(float(raw), evidence="mapping"))
        elif raw is None:
            setattr(ledger, name, None)
    extra = {
        "executable_byte_ledger": doc.get("executable_byte_ledger"),
        "capability_preserving_runtime": doc.get("capability_preserving_runtime"),
        "physical_measurement_authority": doc.get("physical_measurement_authority"),
        "bench_state": doc.get("bench_state") or _dot(doc, "bench.state"),
        "measurement_state": doc.get("measurement_state"),
        "path_kind": doc.get("path_kind") or PRODUCTION,
        "dense_rematerialization": doc.get("dense_rematerialization"),
        "consumes_representation_directly": doc.get("consumes_representation_directly"),
        "nr_complete": doc.get("nr_complete"),
        "nr_bytes_billed": doc.get("nr_bytes_billed"),
        "runtime_references_nr": doc.get("runtime_references_nr"),
    }
    return _can_promote_seven(ledger, extra=extra, raw=doc)


def _can_promote_seven(
    ledger: SevenLedger,
    extra: Mapping[str, Any],
    raw: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    present = ledger.present_names()
    reasons: list[str] = []
    if present == [ProspectiveMetaBpw.category] or set(present) <= {ProspectiveMetaBpw.category}:
        return False, (
            "prospective_meta_bpw is RESEARCH_TARGET and never promotes alone; "
            "0.887 is a description budget, not a representation or a claim"
        )

    q = ledger.qualified_complete_physical_ebpw
    p = ledger.prospective_meta_bpw
    if not _present(q):
        reasons.append("qualified_complete_physical_ebpw is UNKNOWN on this host")
    elif p is not None and q is not None and p.value is not None and q.value == p.value:
        reasons.append(
            "qualified_complete_physical_ebpw numerically equals prospective_meta_bpw "
            "RESEARCH_TARGET; refusing laundering"
        )

    auth = str(extra.get("physical_measurement_authority") or "").upper()
    mstate = extra.get("measurement_state")
    if isinstance(mstate, Mapping):
        mlabel = str(mstate.get("authority") or mstate.get("measurement_state") or "")
    else:
        mlabel = str(mstate or "")
    bstate = str(extra.get("bench_state") or "").upper()
    if _present(q):
        if "DIAGNOSTIC_RELATIVE" in auth or "DIAGNOSTIC_RELATIVE" in mlabel.upper():
            reasons.append("DIAGNOSTIC_RELATIVE cannot back a promotion claim")
        elif auth != PROTECTED:
            reasons.append(
                "qualified_complete_physical_ebpw is not backed by PROTECTED_ABSOLUTE "
                f"(authority={extra.get('physical_measurement_authority')!r})"
            )
        if bstate in {"UNKNOWN", "DIAGNOSTIC_RELATIVE", ""}:
            reasons.append(f"bench.state {extra.get('bench_state')!r} is not a protected measurement")

    byte_ledger = extra.get("executable_byte_ledger")
    if not isinstance(byte_ledger, Mapping) or byte_ledger.get("self_contained") is not True:
        reasons.append("no self-contained executable byte ledger")
    if extra.get("capability_preserving_runtime") is not True:
        reasons.append("no capability-preserving runtime")

    path_kind = str(extra.get("path_kind") or PRODUCTION).upper()
    if path_kind == VERIFICATION:
        reasons.append("verification path may reconstruct; it cannot promote a production executable")

    remat_doc = raw if raw is not None else {
        "path_kind": path_kind,
        "dense_rematerialization": extra.get("dense_rematerialization"),
        "consumes_representation_directly": extra.get("consumes_representation_directly"),
        "execution_path": (raw or {}).get("execution_path") if raw else None,
        "production_path": (raw or {}).get("production_path") if raw else None,
    }
    remat = judge_production_dense_checkpoint(remat_doc)
    if path_kind != VERIFICATION and not remat["ok"]:
        reasons.append(remat["reason"])

    if extra.get("runtime_references_nr") is True and extra.get("nr_bytes_billed") is not True:
        reasons.append("NX references NR at runtime but NR bytes are not billed")

    if extra.get("nr_complete") is False:
        reasons.append("continuous NR is incomplete")

    if not reasons:
        return True, "all promotion predicates held"
    if _present(p) and p is not None and p.value is not None and p.value < 1.0:
        reasons.insert(
            0,
            f"prospective_meta_bpw is {RESEARCH_TARGET} and is never a promotion predicate",
        )
    return False, "; ".join(reasons)


# ---------------------------------------------------------------------------
# NR billing. Quiet runtime pointers are rejected.
# ---------------------------------------------------------------------------


def nx_references_nr_at_runtime(nx: Mapping[str, Any]) -> bool:
    """True when a production NX will need NR bytes at runtime."""
    if not isinstance(nx, Mapping):
        return False
    if nx.get("runtime_references_nr") is True:
        return True
    if nx.get("consumes_nr_at_runtime") is True or nx.get("loads_nr_at_runtime") is True:
        return True
    exe = nx.get("execution_path")
    if isinstance(exe, Mapping):
        if exe.get("consumes_nr") is True or exe.get("loads_nr") is True:
            return True
        if exe.get("runtime_references_nr") is True:
            return True
    prod = nx.get("production_path")
    if isinstance(prod, Mapping) and (
        prod.get("consumes_nr") is True or prod.get("runtime_references_nr") is True
    ):
        return True
    if nx.get("nr_inlined") is True:
        return False
    # Metadata-only seals that *name* an NR they lower at compile time are
    # not runtime consumers. A production executable that names lowers_nr
    # without inlining is a runtime reference.
    lowers = nx.get("lowers_nr")
    path_kind = str(nx.get("path_kind") or PRODUCTION).upper()
    if lowers and path_kind == PRODUCTION and not _status_is_metadata_only(nx):
        return True
    return False


def nr_bytes_billed(ledger: Mapping[str, Any] | None) -> bool:
    if not isinstance(ledger, Mapping):
        return False
    if ledger.get("nr_bytes_billed") is True:
        billed = _as_number(ledger.get("billed_nr_bytes"))
        if billed is not None and billed > 0:
            return True
        # Flag without bytes is not billing.
        return False
    billed = _as_number(ledger.get("billed_nr_bytes"))
    if billed is not None and billed > 0:
        included = ledger.get("included_byte_categories") or []
        if isinstance(included, list) and (
            "nr" in included
            or "compile_time_nr_bytes" in included
            or "serialized_nr_information" in included
        ):
            return True
        complete = _as_number(
            ledger.get("complete_storage_bytes") or ledger.get("runtime_required_bytes")
        )
        if complete is not None and complete >= billed:
            return True
    included = ledger.get("included_byte_categories") or []
    nr = _as_number(ledger.get("nr_bytes") or ledger.get("compile_time_nr_bytes"))
    complete = _as_number(
        ledger.get("complete_storage_bytes") or ledger.get("runtime_required_bytes")
    )
    if (
        isinstance(included, list)
        and (
            "nr" in included
            or "compile_time_nr_bytes" in included
            or "serialized_nr_information" in included
        )
        and nr is not None
        and nr > 0
        and complete is not None
        and complete >= nr
    ):
        return True
    return False


def check_nr_billing(
    nx: Mapping[str, Any],
    ledger: Mapping[str, Any] | None,
) -> dict[str, Any]:
    refs = nx_references_nr_at_runtime(nx)
    billed = nr_bytes_billed(ledger)
    if refs and not billed:
        raise NrBillingError(
            "NX references NR at runtime but NR bytes are not billed into the complete ledger"
        )
    return {
        "ok": True,
        "references_nr_at_runtime": refs,
        "nr_bytes_billed": billed,
        "metadata_only": _status_is_metadata_only(nx) if isinstance(nx, Mapping) else None,
        "reason": (
            "runtime NR pointer is billed"
            if refs and billed
            else "no runtime NR reference"
            if not refs
            else "billed"
        ),
    }


# ---------------------------------------------------------------------------
# Dense rematerialization: production may not reconstruct a dense checkpoint.
# ---------------------------------------------------------------------------


def _path_kind_of(nx: Mapping[str, Any]) -> str:
    raw = nx.get("path_kind") or nx.get("consumer")
    if isinstance(raw, str) and raw.strip().upper() in {PRODUCTION, VERIFICATION}:
        return raw.strip().upper()
    exe = nx.get("execution_path")
    if isinstance(exe, Mapping):
        raw = exe.get("path_kind") or exe.get("consumer")
        if isinstance(raw, str) and raw.strip().upper() in {PRODUCTION, VERIFICATION}:
            return raw.strip().upper()
    prod = nx.get("production_path")
    if isinstance(prod, Mapping):
        raw = prod.get("path_kind")
        if isinstance(raw, str) and raw.strip().upper() in {PRODUCTION, VERIFICATION}:
            return raw.strip().upper()
    return PRODUCTION


def _tri_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"true", "yes", "required", "present", "hidden", "enabled"}:
            return True
        if key in {"false", "no", "forbidden", "banned", "absent", "disabled", "never"}:
            return False
    return None


def judge_production_dense_checkpoint(nx: Mapping[str, Any]) -> dict[str, Any]:
    """Production must consume the representation directly.

    Verification MAY reconstruct a dense tensor / checkpoint. That path
    cannot promote.
    """
    if not isinstance(nx, Mapping):
        return {
            "ok": False,
            "path_kind": PRODUCTION,
            "reason": "NX document is not an object",
            "reconstructs_dense_checkpoint": None,
        }
    path_kind = _path_kind_of(nx)
    exe = nx.get("execution_path") if isinstance(nx.get("execution_path"), Mapping) else {}
    prod = nx.get("production_path") if isinstance(nx.get("production_path"), Mapping) else {}
    recon = _tri_bool(
        prod.get("reconstructs_dense_checkpoint")
        if prod
        else None
    )
    if recon is None:
        recon = _tri_bool(nx.get("reconstructs_dense_checkpoint"))
    if recon is None:
        recon = _tri_bool(exe.get("reconstructs_dense_checkpoint") if exe else None)
    if recon is None:
        recon = _tri_bool(exe.get("decompresses_to_dense_weight_tensor") if exe else None)
    if recon is None:
        recon = _tri_bool(nx.get("decompresses_to_dense_weight_tensor"))
    if recon is None:
        recon = _tri_bool(prod.get("rematerialize_dense") if prod else None)

    landed = judge_dense_rematerialization(nx)

    if path_kind == VERIFICATION:
        return {
            "ok": True,
            "path_kind": VERIFICATION,
            "reason": (
                "verification MAY reconstruct a dense checkpoint; "
                "this is not a production consumer"
            ),
            "reconstructs_dense_checkpoint": recon,
            "landed_remat": landed.as_dict(),
        }
    proven = recon is True or landed.decompresses_to_dense_weight_tensor is True
    if proven:
        reason = (
            "production path reconstructs a dense checkpoint; "
            "verification may reconstruct, production may not"
        )
        if landed.decompresses_to_dense_weight_tensor is True and not landed.ok:
            reason = landed.reason
        return {
            "ok": False,
            "path_kind": PRODUCTION,
            "reason": reason,
            "reconstructs_dense_checkpoint": True,
            "landed_remat": landed.as_dict(),
        }
    if _status_is_metadata_only(nx):
        return {
            "ok": True,
            "path_kind": PRODUCTION,
            "reason": (
                "metadata seal is not a production consumer; dense rematerialization "
                "is unknown because no production NX exists"
            ),
            "reconstructs_dense_checkpoint": recon,
            "landed_remat": landed.as_dict(),
        }
    return {
        "ok": landed.ok,
        "path_kind": PRODUCTION,
        "reason": landed.reason,
        "reconstructs_dense_checkpoint": recon,
        "landed_remat": landed.as_dict(),
    }


def reject_dense_production(nx: Mapping[str, Any]) -> dict[str, Any]:
    verdict = judge_production_dense_checkpoint(nx)
    if _path_kind_of(nx) != VERIFICATION and not verdict["ok"]:
        raise DenseRematError(verdict["reason"])
    return verdict


# ---------------------------------------------------------------------------
# WorkUnits. Physical measurement sleeps until hardware qualifies.
# ---------------------------------------------------------------------------


def _emit_workunit(**kwargs: Any) -> dict[str, Any]:
    try:
        from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit
    except Exception:
        emit_hcli_workunit = None
        validate_emitted_unit = None
    if emit_hcli_workunit is None:
        row = {
            "id": kwargs["id"],
            "role": kwargs.get("role"),
            "description": kwargs.get("description"),
            "dependencies": list(kwargs.get("dependencies") or []),
            "status": kwargs.get("status") or "pending",
            "assigned_runtime": None,
            "attempts": 0,
            "resource_class": kwargs.get("resource_class"),
            "repairs": None,
            "failure_context": None,
            "preferred_backend": kwargs.get("preferred_backend"),
            "assigned_backend": None,
            "backend_task_id": None,
            "verifier": kwargs.get("verifier"),
            "effect_class": kwargs.get("effect_class") or "READ_ONLY",
            "workspace": "repo-root",
            "verification": None,
            "repair_root": None,
            "repair_depth": 0,
            "repair_reason": None,
            "repair_exhausted": False,
            "ready_at": None,
            "running_at": None,
            "finished_at": None,
            "classification": kwargs.get("classification"),
            "provider": kwargs.get("provider"),
            "content_hash": None,
            "claim_boundary": (
                "WorkUnit is a proposal; receipt and protected capability gates remain authoritative"
            ),
        }
        extras = kwargs.get("extras") or {}
        row.update(extras)
        return row
    row = emit_hcli_workunit(
        id=kwargs["id"],
        role=kwargs["role"],
        description=kwargs["description"],
        dependencies=kwargs.get("dependencies") or [],
        resource_class=kwargs["resource_class"],
        verifier=kwargs["verifier"],
        provider=kwargs["provider"],
        effect_class=kwargs.get("effect_class") or "READ_ONLY",
        preferred_backend=kwargs.get("preferred_backend"),
        status=kwargs.get("status") or "pending",
        classification=kwargs.get("classification"),
        extras=kwargs.get("extras"),
    )
    if validate_emitted_unit is not None:
        validate_emitted_unit(row)
    return row


def emit_workunits(docs: Mapping[str, Any]) -> list[dict[str, Any]]:
    flash = _safe_flash_state(docs.get("handoff") if isinstance(docs.get("handoff"), Mapping) else None)
    sleeping = _emit_workunit(
        id="future.flash_nr_complete.qualified_physical_ebpw",
        role="science",
        description=(
            "SLEEPING: measure qualified_complete_physical_ebpw for a "
            "source-independent Flash NX under a real protected GPU lease. "
            "Must not synthesize a number. Wake when hardware qualifies."
        ),
        dependencies=["future.flash_nr_complete.build"],
        resource_class="GPU_EXCLUSIVE",
        verifier="future.flash_nr_complete.can_promote",
        provider="sidecar.flash_nr_complete",
        effect_class="READ_ONLY",
        preferred_backend="metal",
        status="sleeping",
        classification="SLEEPING",
        extras={
            "blocked_reason": (
                "no Metal-capable GPU / Metal compiler missing under CommandLineTools / "
                "protected bench lock unproven / machine HEAVY / Flash NX SCAFFOLD_ONLY / "
                "teacher capture 0/256"
            ),
            "wake_when": [
                "MetalContext reports a Metal-capable GPU",
                "xcrun locates the Metal compiler",
                "protected bench lock is held by a proven pid or is free",
                "qualification pipeline classifies the machine QUIESCENT",
                "Flash source-independent NX is no longer SCAFFOLD_ONLY",
            ],
            "must_not_synthesize_result": True,
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "species": "accelerator_candidate_qualification",
            "source_independent_nx_status": flash.get("source_independent_nx_status"),
            "current_qualified_physical_ebpw": UNKNOWN,
        },
    )
    build_unit = _emit_workunit(
        id="future.flash_nr_complete.build",
        role="science",
        description=(
            "Rebuild the continuous complete Flash NR and the seven typed "
            "EBPW quantities from disk. STATIC_ONLY. Never measures hardware."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.flash_nr_complete.selftest",
        provider="sidecar.flash_nr_complete",
        effect_class="READ_ONLY",
        preferred_backend=None,
        status="pending",
        classification=None,
        extras={
            "command": ["python3", "tools/future/flash_nr_complete.py", "--build"],
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "species": "independent_reproduction",
        },
    )
    return [build_unit, sleeping]


def _safe_flash_state(handoff: Mapping[str, Any] | None) -> dict[str, Any]:
    """Statuses only. Never copy GPU_ns / wall_ns / accepted_tps numbers."""
    if not isinstance(handoff, Mapping):
        return {
            "handoff_present": False,
            "current_qualified_physical_ebpw": UNKNOWN,
            "current_prospective_meta_bpw": None,
        }
    flash = handoff.get("current_flash_state")
    flash = flash if isinstance(flash, Mapping) else {}
    return {
        "handoff_present": True,
        "current_qualified_physical_ebpw": (
            handoff.get("current_qualified_physical_ebpw")
            if isinstance(handoff.get("current_qualified_physical_ebpw"), str)
            else UNKNOWN
        ),
        "current_prospective_meta_bpw": _as_number(handoff.get("current_prospective_meta_bpw")),
        "source_independent_nx_status": _dot(flash, "source_independent_nx.status"),
        "source_independent_nx_qualification": _dot(flash, "source_independent_nx.qualification"),
        "ebpw_budget_status": _dot(flash, "ebpw_budget.status"),
        "ebpw_budget_promotion_allowed": _dot(flash, "ebpw_budget.promotion_allowed"),
        "critical_path_status": _dot(flash, "critical_path.status"),
        "stateful_gate_status": _dot(flash, "stateful_gate.status"),
        "latest_bounded_chain_status": _dot(flash, "latest_bounded_chain.status"),
        "latest_bounded_chain_promotion": _dot(flash, "latest_bounded_chain.promotion"),
    }


# ---------------------------------------------------------------------------
# Negative controls. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def unbilled_runtime_nr_nx() -> dict[str, Any]:
    return {
        "schema": "hawking.flash.nx_genome.v1",
        "status": "SOURCE_INDEPENDENT_COMPLETE",
        "path_kind": PRODUCTION,
        "runtime_references_nr": True,
        "lowers_nr": {"path": "receipts/headless/FLASH_COMPLETE_V2.nr.json"},
        "execution_path": {
            "consumes_nr": True,
            "loads_nr": True,
            "consumes_representation_directly": True,
        },
    }


def billed_runtime_nr_ledger() -> dict[str, Any]:
    return {
        "self_contained": True,
        "for_this_executable": True,
        "complete_storage_bytes": 4096,
        "billed_nr_bytes": 2048,
        "nr_bytes_billed": True,
        "included_byte_categories": ["serialized_nr_information", "nx_runtime"],
    }


def unbilled_runtime_nr_ledger() -> dict[str, Any]:
    return {
        "self_contained": True,
        "for_this_executable": True,
        "complete_storage_bytes": 1024,
        "nr_bytes_billed": False,
        "billed_nr_bytes": 0,
        "included_byte_categories": ["nx_runtime"],
    }


def dense_production_nx() -> dict[str, Any]:
    return {
        "schema": "hawking.flash.nx_genome.v1",
        "status": "SOURCE_INDEPENDENT_COMPLETE",
        "path_kind": PRODUCTION,
        "production_path": {
            "reconstructs_dense_checkpoint": True,
            "rematerialize_dense": True,
        },
        "execution_path": {
            "decompresses_to_dense_weight_tensor": True,
            "runs_ordinary_kernels": True,
        },
    }


def verifying_dense_nx() -> dict[str, Any]:
    return {
        "schema": "hawking.flash.nx_genome.v1",
        "path_kind": VERIFICATION,
        "production_path": {"reconstructs_dense_checkpoint": True},
        "execution_path": {
            "decompresses_to_dense_weight_tensor": True,
            "runs_ordinary_kernels": True,
        },
    }


def selftest() -> dict[str, Any]:
    results: dict[str, Any] = {}

    a = SourceControlEbpw(16.0, evidence="selftest")
    b = ProspectiveMetaBpw(0.887, evidence=RESEARCH_TARGET)
    mixed_raised = False
    mixed_message = ""
    try:
        _ = a + b
    except CategoryError as exc:
        mixed_raised = True
        mixed_message = str(exc)
    results["cross_category_arithmetic_raises"] = mixed_raised
    results["cross_category_arithmetic_message"] = mixed_message
    if not mixed_raised:
        raise AssertionError("cross-category arithmetic did not raise")

    assign_raised = False
    ledger = SevenLedger(source_control_ebpw=a, prospective_meta_bpw=b)
    try:
        ledger.qualified_complete_physical_ebpw = b  # type: ignore[assignment]
    except CategoryError:
        assign_raised = True
    results["cross_category_assignment_raises"] = assign_raised
    if not assign_raised:
        raise AssertionError("cross-category assignment did not raise")

    coerce_raised = False
    try:
        bind(QualifiedCompletePhysicalEbpw, CompletePhysicalEbpw(0.887, evidence="launder"))
    except CategoryError:
        coerce_raised = True
    results["landed_complete_physical_cannot_bind_qualified"] = coerce_raised
    if not coerce_raised:
        raise AssertionError("CompletePhysicalEbpw bound into QualifiedCompletePhysicalEbpw")

    ok, reason = can_promote(b)
    results["meta_only_research_target_refused"] = ok is False
    results["meta_only_research_target_reason"] = reason
    if ok or RESEARCH_TARGET not in reason:
        raise AssertionError(f"can_promote accepted prospective_meta_bpw alone: {reason}")

    ok2, reason2 = can_promote({"prospective_meta_bpw": 0.887, "promotion_allowed": True})
    results["meta_with_flag_refused"] = ok2 is False
    if ok2:
        raise AssertionError("can_promote accepted a flag around RESEARCH_TARGET")

    billing_raised = False
    billing_message = ""
    try:
        check_nr_billing(unbilled_runtime_nr_nx(), unbilled_runtime_nr_ledger())
    except NrBillingError as exc:
        billing_raised = True
        billing_message = str(exc)
    results["unbilled_runtime_nr_rejected"] = billing_raised
    results["unbilled_runtime_nr_message"] = billing_message
    if not billing_raised:
        raise AssertionError("unbilled runtime NR pointer was not rejected")
    billed_ok = check_nr_billing(unbilled_runtime_nr_nx(), billed_runtime_nr_ledger())
    results["billed_runtime_nr_accepted"] = billed_ok["ok"] is True
    if not billed_ok["ok"]:
        raise AssertionError("honest billed runtime NR pointer was rejected")

    remat_raised = False
    remat_message = ""
    try:
        reject_dense_production(dense_production_nx())
    except DenseRematError as exc:
        remat_raised = True
        remat_message = str(exc)
    results["dense_production_rejected"] = remat_raised
    results["dense_production_message"] = remat_message
    if not remat_raised:
        raise AssertionError("dense-rematerializing production path was not rejected")
    verify = reject_dense_production(verifying_dense_nx())
    results["verification_may_reconstruct"] = verify["ok"] is True
    if not verify["ok"]:
        raise AssertionError("verification reconstruction was refused")

    missing_raised = False
    try:
        assert_complete({"complete": True, "organs": [{"organ": "norm"}]}, ["norm", "routed_experts"])
    except IncompleteNrError:
        missing_raised = True
    results["missing_organ_rejected"] = missing_raised
    if not missing_raised:
        raise AssertionError("missing organ was not rejected")

    # Positive combinator: the gate can open. Not a measurement.
    full = {
        "qualified_complete_physical_ebpw": QualifiedCompletePhysicalEbpw(
            2.4, evidence="synthetic combinator control (not a measurement)"
        ),
        "source_control_ebpw": SourceControlEbpw(16.0, evidence=SCIENCE_ONLY),
        "executable_byte_ledger": {
            "self_contained": True,
            "for_this_executable": True,
            "complete_storage_bytes": 4096,
        },
        "capability_preserving_runtime": True,
        "physical_measurement_authority": PROTECTED,
        "bench_state": "PROTECTED",
        "measurement_state": PROTECTED,
        "path_kind": PRODUCTION,
        "dense_rematerialization": False,
        "consumes_representation_directly": True,
        "nr_complete": True,
    }
    full_ok, full_reason = can_promote(full)
    results["positive_combinator_opens"] = full_ok is True
    results["positive_combinator_reason"] = full_reason
    results["positive_combinator_is_not_a_measurement"] = True
    if not full_ok:
        raise AssertionError(f"positive combinator failed to open: {full_reason}")

    equal_num_raised = False
    try:
        _ = SourceControlEbpw(16.0) == StaticCompleteEbpwEstimate(16.0)
    except CategoryError:
        equal_num_raised = True
    results["equal_numbers_are_not_interchangeable"] = equal_num_raised
    if not equal_num_raised:
        raise AssertionError("16.0 source_control compared equal to 16.0 static_complete")

    return results


# ---------------------------------------------------------------------------
# Recovered implementation / gaps.
# ---------------------------------------------------------------------------


def recovered_implementation(docs: Mapping[str, Any]) -> list[dict[str, Any]]:
    res = docs.get("resolution") if isinstance(docs.get("resolution"), Mapping) else {}
    return [
        {
            "path": "tools/future/ebpw_categories.py",
            "what": (
                "Five Quantity subclasses; mixed arithmetic is CategoryError; "
                "can_promote false for prospective-only; dense remat "
                "production/verification split. EXTENDED here to seven named "
                "EBPW quantities. Not forked: new types subclass Quantity."
            ),
            "adequate": False,
            "gap": "five quantities, not the seven EBPW names this lane owns; no continuous NR",
        },
        {
            "path": "tools/future/flash_nx_audit.py",
            "what": "Seven NX completeness requirements and the BLOCKED-Flash dependency chain.",
            "adequate": False,
            "gap": "audits NX promotion, does not maintain a complete-at-all-times NR",
        },
        {
            "path": "receipts/future/evidence/FLASH_COMPLETE_V2.nr.json",
            "what": (
                f"status={_dot(docs.get('nr_v2') or {}, 'status')}; "
                "COMPLETE_HETEROGENEOUS_CANDIDATE_NOT_FOR_PROMOTION; "
                "parts cover census families; compact routed/ngram not runtime_ready."
            ),
            "resolved_via": (res.get("nr_v2") or {}).get("resolved_via"),
            "adequate": False,
            "gap": (
                "composition snapshot, not a continuously rebuilt complete NR "
                "with derived per-organ promotion status and typed EBPW"
            ),
        },
        {
            "path": "receipts/future/evidence/FLASH_COMPLETE_V0.nx.json",
            "what": (
                f"status={_dot(docs.get('nx_v0') or {}, 'status')}; "
                "SEALED_METADATA_ONLY; complete_system_ebpw null; lowers_nr at compile time."
            ),
            "resolved_via": (res.get("nx_v0") or {}).get("resolved_via"),
            "adequate": False,
            "gap": "metadata seal, not a billed runtime NX and not a type system",
        },
        {
            "path": "receipts/future/evidence/FLASH_META_REPRESENTATION_SUB1.json",
            "what": (
                f"status={_dot(docs.get('meta') or {}, 'status')}; "
                "prospective_target is a description budget; physical_ebpw NULL_BY_RULE."
            ),
            "resolved_via": (res.get("meta") or {}).get("resolved_via"),
            "adequate": False,
            "gap": "honest RESEARCH_TARGET; not typed against the other six quantities",
        },
        {
            "path": "receipts/future/evidence/FLASH_ORGAN_CENSUS.json",
            "what": "STRUCTURAL_METADATA_SCREEN; family_summary is the organ authority.",
            "resolved_via": (res.get("census") or {}).get("resolved_via"),
            "adequate": False,
            "gap": "census, not a candidate-per-organ NR",
        },
        {
            "path": "CODEX_ACCELERATOR_HANDOFF.json",
            "what": (
                "current_flash_state (source_independent_nx SCAFFOLD_ONLY, "
                "ebpw_budget, critical_path, stateful_gate); "
                "current_prospective_meta_bpw; current_qualified_physical_ebpw UNKNOWN. "
                "Training trace, not an archive this module copies numbers from."
            ),
            "resolved_via": (res.get("handoff") or {}).get("resolved_via"),
            "adequate": False,
            "gap": "handoff; timed kernel/host fields must not enter this receipt",
        },
        {
            "path": "tools/flash_complete_nr.py",
            "what": "Codex composer of FLASH_COMPLETE_V2.nr.json (cited by flash_nx_audit; not in this sparse tree).",
            "adequate": False,
            "gap": "Codex-owned; sidecar must not mutate it; does not type-separate EBPW",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "Continuous complete NR: every census family (and, when present, every EBPW-budget organ) has an occupying candidate or exact-control COMPILE_TIME_SCIENCE_ONLY fallback.",
        "Seven Quantity subclasses (six new + landed ProspectiveMetaBpw); mixed arithmetic, comparison, bind(), and SevenLedger field assignment are CategoryError.",
        "0.887 prospective_meta_bpw recorded as RESEARCH_TARGET; can_promote on it alone returns False with that reason, flag or not.",
        "qualified_complete_physical_ebpw is UNKNOWN on this host and stays UNKNOWN; landed CompletePhysicalEbpw cannot bind into it.",
        "NX that references NR at runtime without billed NR bytes raises NrBillingError; honest billed ledger is accepted; metadata-only lowers_nr is not a runtime reference.",
        "Production path that reconstructs a dense checkpoint raises DenseRematError; verification may reconstruct and cannot promote.",
        "NR promotion_status is derived from per-organ status, never asserted at the root.",
        "SLEEPING GPU WorkUnit for the physical measurement; STATIC_ANALYSIS WorkUnit for --build. Fail closed: no synthetic physical EBPW.",
    ]


def negative_findings(docs: Mapping[str, Any]) -> list[dict[str, Any]]:
    res = docs.get("resolution") if isinstance(docs.get("resolution"), Mapping) else {}
    findings = [
        {
            "looked_for": "a packed NR information payload",
            "found": "FLASH_COMPLETE_V2.nr.json is a composition document; serialized_nr_information is NOT_BUILT",
        },
        {
            "looked_for": "a serialized NX with complete_system_ebpw",
            "found": "FLASH_COMPLETE_V0.nx.json is SEALED_METADATA_ONLY; qualification.complete_system_ebpw is null",
        },
        {
            "looked_for": "qualified_complete_physical_ebpw",
            "found": "UNKNOWN on this host; Codex handoff current_qualified_physical_ebpw is UNKNOWN; sidecar has no GPU",
        },
        {
            "looked_for": "runtime-ready compact occupying candidates",
            "found": "NR V2 compact routed/ngram variants are not runtime_ready; occupying slots are exact-control COMPILE_TIME_SCIENCE_ONLY",
        },
        {
            "looked_for": "hardware measurement",
            "found": "sidecar produces STATIC_ONLY / bench UNKNOWN; physical blockers remain sleeping WorkUnits",
        },
        {
            "looked_for": "identity of static_active_ebpw_estimate with the 0.887 research target",
            "found": (
                "header-derived static_active_ebpw_estimate is a different type and a "
                "different number from prospective_meta_bpw; assigning one to the other "
                "is a CategoryError. The two must not be added, compared, or promoted."
            ),
        },
    ]
    missing = [
        name
        for name, row in (res.items() if isinstance(res, dict) else [])
        if isinstance(row, dict) and row.get("present") is not True
    ]
    if missing:
        findings.append(
            {
                "looked_for": "optional evidence in this checkout",
                "found_missing": missing,
                "meaning": (
                    "missing in this sparse worktree is not proof the file is gone; "
                    "the module coped and recorded resolved_via"
                ),
            }
        )
    return findings


def resident_callable(work_units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "can_hcli_invoke": True,
        "entry_point": "python3 tools/future/flash_nr_complete.py --build",
        "module": "tools.future.flash_nr_complete",
        "callables": ["build", "selftest", "can_promote", "build_continuous_nr", "check_nr_billing"],
        "work_units_emitted": [u.get("id") for u in work_units],
        "work_unit_status": {u.get("id"): u.get("status") for u in work_units},
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": (
            "F001 Flash source-independent NX / continuous NR completeness: "
            "representation is complete-at-all-times so execution can refill "
            "against a derived promotion_status instead of waiting on a packed NX"
        ),
        "fail_closed": [
            "HardwareClaimError if a numeric hardware field is written into the receipt",
            "IncompleteNrError if a census organ has no occupying candidate",
            "CategoryError if one of the seven quantities is assigned/added/compared to another",
            "can_promote(prospective_meta_bpw) is False (RESEARCH_TARGET)",
            "NrBillingError if a production NX references NR at runtime without billed NR bytes",
            "DenseRematError if a production path reconstructs a dense checkpoint",
            "qualified_complete_physical_ebpw stays UNKNOWN; a synthetic 0.887 physical is refused",
            "SLEEPING GPU WorkUnit must not synthesize a result when hardware is unqualified",
        ],
        "integration_swap": (
            "resident_api.py / super_resident.py / workgraph.py / wakeup.py / "
            "frontiers.py are concurrent-wave siblings and are not imported. "
            "HCLI discovers this module via the entry_point and the emitted "
            "WorkUnit ids. Swap those in at integration."
        ),
    }


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------


def assemble(docs: Mapping[str, Any]) -> dict[str, Any]:
    controls = selftest()
    nr = build_continuous_nr(docs)
    assert_complete(nr, nr["required_organs"])
    seven = seven_from_docs(docs)
    units = emit_workunits(docs)
    nx = docs.get("nx_v0") if isinstance(docs.get("nx_v0"), Mapping) else {}
    live_billing = {
        "nx_status": (nx or {}).get("status"),
        "lowers_nr": _dot(nx or {}, "lowers_nr.path"),
        "runtime_references_nr": nx_references_nr_at_runtime(nx or {}),
        "note": (
            "V0 NX is SEALED_METADATA_ONLY; lowers_nr is a compile-time name, "
            "not a runtime consume. A production NX that quietly points at NR "
            "without billed NR bytes is rejected by check_nr_billing."
        ),
    }
    live_remat = judge_production_dense_checkpoint(nx or {"path_kind": PRODUCTION})
    promote_ok, promote_reason = can_promote(seven)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Keep a complete heterogeneous Flash NR candidate on disk at all "
            "times, with seven strictly separated EBPW quantities, so 0.887 "
            "prospective meta-BPW cannot be laundered into a physical claim."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "recovered_implementation": recovered_implementation(docs),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(docs),
        "evidence_resolution": docs.get("resolution"),
        "continuous_nr": nr,
        "seven_quantities": seven.as_dict(),
        "prospective_meta_bpw_role": RESEARCH_TARGET,
        "qualified_complete_physical_ebpw_state": UNKNOWN,
        "can_promote": promote_ok,
        "can_promote_reason": promote_reason,
        "nr_billing": {
            "rule": (
                "If a production NX references NR at runtime, NR bytes are "
                "BILLED into the complete ledger. A quiet pointer is rejected."
            ),
            "live_nx_v0": live_billing,
        },
        "dense_rematerialization": {
            "production": "must consume the representation directly; reconstructing a dense checkpoint is refused",
            "verification": "MAY reconstruct a dense checkpoint; that path cannot promote",
            "live_nx_v0": live_remat,
        },
        "current_flash_state": _safe_flash_state(
            docs.get("handoff") if isinstance(docs.get("handoff"), Mapping) else None
        ),
        "resident_callable": resident_callable(units),
        "work_units": units,
        "selftest": controls,
        "integration_points": [
            "tools/future/ebpw_categories.py (Quantity, can_promote, judge_dense_rematerialization) — imported, not forked",
            "tools/future/flash_nx_audit.py — NX completeness; not imported at runtime to avoid missing-ledger raises",
            "tools/future/workunit_species.py — emit_hcli_workunit when hcli is importable; local dict fallback otherwise",
            "hcli WorkUnit field set — proposal only; this module does not schedule",
            "resident_api.py / super_resident.py / workgraph.py / wakeup.py / frontiers.py — concurrent wave; swap at integration",
            "F001 CLAUDE_GLOBAL_FRONTIER — this receipt feeds the continuous-NR half of that blocker",
        ],
        "next_workunits": [
            {
                "id": "future.flash_nr_complete.build",
                "status": "pending",
                "resource_class": "STATIC_ANALYSIS",
            },
            {
                "id": "future.flash_nr_complete.qualified_physical_ebpw",
                "status": "sleeping",
                "resource_class": "GPU_EXCLUSIVE",
                "must_not_synthesize_result": True,
            },
        ],
    }


def build() -> Path:
    docs = load_docs()
    doc = assemble(docs)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
