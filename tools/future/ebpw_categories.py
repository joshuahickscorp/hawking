"""EBPW category validator — meta-BPW must never launder physical EBPW.

Five quantities are distinct types. Arithmetic between them is a type error.
`prospective_meta_bpw < 1` alone never promotes.

    python3 tools/future/ebpw_categories.py --build
    python3 tools/future/ebpw_categories.py --validate receipts/headless/FLASH_META_REPRESENTATION_SUB1.json
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Mapping, TypeVar

from tools.future._common import REPO, git, load_json, write_receipt

RECEIPT = "EBPW_CATEGORY_VALIDATOR.json"
SCHEMA = "hawking.future.ebpw_categories.v1"

PROTECTED = "PROTECTED_ABSOLUTE"
DIAGNOSTIC = "DIAGNOSTIC_RELATIVE"
PRODUCTION = "PRODUCTION"
VERIFICATION = "VERIFICATION"

# Receipts this validator inventories. Local checkout first; untracked
# primary-worktree copies are a read-only fallback (see resolve_repo_path).
INVENTORY_PATHS: tuple[str, ...] = (
    "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json",
    "receipts/headless/FLASH_COMPLETE_V2.nr.json",
    "receipts/headless/FLASH_COMPLETE_V0.nx.json",
    "receipts/headless/FLASH_COMPLETE_V0.BYTE_LEDGER.json",
    "receipts/headless/FLASH_EBPW_BUDGET.json",
    "receipts/headless/EBPW_NAMESPACE_SEPARATION.json",
    "receipts/QWEN80_BIT_BUDGET_LEDGER.json",
    "receipts/headless/QWEN80_BIT_BUDGET_LEDGER.json",
)


class CategoryError(TypeError):
    """Cross-category arithmetic, comparison, or coercion."""


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


@dataclass(frozen=True, eq=False)
class Quantity:
    """Base for the five non-interchangeable EBPW quantities.

    Subclasses are the public types. Operators refuse any other type,
    including bare numbers — extract `.value` to do numeric work.
    """

    value: float | None
    evidence: str = "unspecified"
    source: str | None = None

    category: ClassVar[str] = ""
    unit: ClassVar[str] = ""

    def _pair(self, other: object, op: str) -> tuple[float, float]:
        if type(other) is not type(self):
            other_cat = getattr(other, "category", type(other).__name__)
            raise CategoryError(
                f"refused {self.category} {op} {other_cat}: "
                "EBPW categories are not interchangeable"
            )
        assert isinstance(other, Quantity)
        if self.value is None or other.value is None:
            raise CategoryError(
                f"refused {self.category} {op} on a NULL quantity"
            )
        return self.value, other.value

    def __add__(self, other: object) -> Quantity:
        a, b = self._pair(other, "+")
        return type(self)(a + b, evidence="sum")

    def __sub__(self, other: object) -> Quantity:
        a, b = self._pair(other, "-")
        return type(self)(a - b, evidence="difference")

    def __mul__(self, other: object) -> Quantity:
        a, b = self._pair(other, "*")
        return type(self)(a * b, evidence="product")

    def __truediv__(self, other: object) -> Quantity:
        a, b = self._pair(other, "/")
        if b == 0.0:
            raise CategoryError(f"refused {self.category} / 0")
        return type(self)(a / b, evidence="quotient")

    def __radd__(self, other: object) -> Quantity:
        raise CategoryError(
            f"refused {type(other).__name__} + {self.category}: "
            "EBPW categories are not interchangeable"
        )

    __rsub__ = __rmul__ = __rtruediv__ = __radd__

    def __float__(self) -> float:
        raise CategoryError(
            f"refused float({self.category}); extract .value explicitly"
        )

    def __int__(self) -> int:
        raise CategoryError(
            f"refused int({self.category}); extract .value explicitly"
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            if isinstance(other, Quantity):
                raise CategoryError(
                    f"cannot compare {self.category} with {other.category}"
                )
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other: object) -> bool:
        a, b = self._pair(other, "<")
        return a < b

    def __le__(self, other: object) -> bool:
        a, b = self._pair(other, "<=")
        return a <= b

    def __gt__(self, other: object) -> bool:
        a, b = self._pair(other, ">")
        return a > b

    def __ge__(self, other: object) -> bool:
        a, b = self._pair(other, ">=")
        return a >= b

    def __hash__(self) -> int:
        return hash((type(self), self.value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "unit": self.unit,
            "value": self.value,
            "evidence": self.evidence,
            "source": self.source,
        }


@dataclass(frozen=True, eq=False)
class ProspectiveMetaBpw(Quantity):
    """A description budget. NOT bytes. NOT physical."""

    category: ClassVar[str] = "prospective_meta_bpw"
    unit: ClassVar[str] = (
        "bits per source weight (description budget, not serialized bytes)"
    )


@dataclass(frozen=True, eq=False)
class SerializedMetaBytes(Quantity):
    """Bytes of an artifact that actually exists on disk."""

    category: ClassVar[str] = "serialized_meta_bytes"
    unit: ClassVar[str] = "bytes of an on-disk serialized artifact"


@dataclass(frozen=True, eq=False)
class CompileTimeNrBytes(Quantity):
    """NR representation size at compile time."""

    category: ClassVar[str] = "compile_time_nr_bytes"
    unit: ClassVar[str] = "bytes of the NR representation"


@dataclass(frozen=True, eq=False)
class NxRuntimeBytes(Quantity):
    """What the executable needs at runtime."""

    category: ClassVar[str] = "nx_runtime_bytes"
    unit: ClassVar[str] = "bytes the executable needs at runtime"


@dataclass(frozen=True, eq=False)
class CompletePhysicalEbpw(Quantity):
    """The only quantity that can support a promotion claim."""

    category: ClassVar[str] = "complete_physical_ebpw"
    unit: ClassVar[str] = (
        "bits per weight of a complete executable, from artifact bytes "
        "or protected measurement — never from a description budget"
    )


CATEGORY_TYPES: dict[str, type[Quantity]] = {
    ProspectiveMetaBpw.category: ProspectiveMetaBpw,
    SerializedMetaBytes.category: SerializedMetaBytes,
    CompileTimeNrBytes.category: CompileTimeNrBytes,
    NxRuntimeBytes.category: NxRuntimeBytes,
    CompletePhysicalEbpw.category: CompletePhysicalEbpw,
}


@dataclass(frozen=True)
class ActiveBytesAccounting:
    """Five byte-touch fields. They stay separate; a tiny rematerializing
    artifact can lose to a larger one that touches fewer bytes per token.
    """

    total_artifact_bytes: int | None = None
    resident_bytes: int | None = None
    active_bytes_per_token: int | None = None
    actual_read_bytes_per_token: int | None = None
    transient_bytes: int | None = None
    evidence: str = "unspecified"

    def collapsed_total(self) -> int:
        raise CategoryError(
            "active-bytes fields must not be collapsed into a single number"
        )

    def __add__(self, other: object) -> ActiveBytesAccounting:
        raise CategoryError(
            "active-bytes fields cannot be added to each other or to EBPW categories"
        )

    __radd__ = __add__

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_artifact_bytes": self.total_artifact_bytes,
            "resident_bytes": self.resident_bytes,
            "active_bytes_per_token": self.active_bytes_per_token,
            "actual_read_bytes_per_token": self.actual_read_bytes_per_token,
            "transient_bytes": self.transient_bytes,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class RematVerdict:
    ok: bool
    path_kind: str
    reason: str
    decompresses_to_dense_weight_tensor: bool | None
    runs_ordinary_kernels: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "path_kind": self.path_kind,
            "reason": self.reason,
            "decompresses_to_dense_weight_tensor": self.decompresses_to_dense_weight_tensor,
            "runs_ordinary_kernels": self.runs_ordinary_kernels,
        }


@dataclass(frozen=True)
class PromotionLedger:
    prospective_meta_bpw: ProspectiveMetaBpw | None = None
    serialized_meta_bytes: SerializedMetaBytes | None = None
    compile_time_nr_bytes: CompileTimeNrBytes | None = None
    nx_runtime_bytes: NxRuntimeBytes | None = None
    complete_physical_ebpw: CompletePhysicalEbpw | None = None
    executable_byte_ledger: Mapping[str, Any] | None = None
    capability_preserving_runtime: bool = False
    physical_measurement_authority: str | None = None
    bench_state: str | None = None
    measurement_state: str | None = None
    path_kind: str = PRODUCTION
    dense_rematerialization: Any = None
    decompresses_to_dense_weight_tensor: bool | None = None
    runs_ordinary_kernels: bool | None = None
    consumes_representation_directly: bool | None = None
    active_bytes: ActiveBytesAccounting | None = None
    promotion_claimed: bool = False
    source_path: str | None = None
    schema: str | None = None
    status: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _present(qty: Quantity | None) -> bool:
    return qty is not None and qty.value is not None


def _self_contained_byte_ledger(block: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not isinstance(block, Mapping) or not block:
        return False, "no self-contained executable byte ledger"
    if block.get("self_contained") is not True:
        return False, "byte ledger is not marked self_contained"
    if block.get("for_this_executable") is not True:
        return False, "byte ledger is not for this executable"
    storage = _as_number(
        block.get("complete_storage_bytes", block.get("runtime_required_bytes"))
    )
    if storage is None or storage <= 0:
        return False, "byte ledger has no positive complete storage/runtime bytes"
    return True, "self-contained executable byte ledger present"


def _protected_physical(ledger: PromotionLedger) -> tuple[bool, str]:
    if not _present(ledger.complete_physical_ebpw):
        return False, "complete_physical_ebpw is absent"
    auth = (ledger.physical_measurement_authority or "").upper()
    mstate = (ledger.measurement_state or "").upper()
    bstate = (ledger.bench_state or "").upper()
    if DIAGNOSTIC in auth or DIAGNOSTIC in mstate:
        return False, "DIAGNOSTIC_RELATIVE cannot back a promotion claim"
    if auth != PROTECTED:
        return False, (
            "complete_physical_ebpw is not backed by protected measurement "
            f"(authority={ledger.physical_measurement_authority!r})"
        )
    if mstate and mstate != PROTECTED:
        return False, (
            f"measurement_state {ledger.measurement_state!r} is not {PROTECTED}"
        )
    if bstate in {"UNKNOWN", DIAGNOSTIC, ""}:
        return False, (
            f"bench.state {ledger.bench_state!r} is not a protected measurement"
        )
    return True, "complete_physical_ebpw backed by protected measurement"


def can_promote(ledger: PromotionLedger | Mapping[str, Any]) -> tuple[bool, str]:
    """False, with the reason, unless every promotion predicate holds.

    `prospective_meta_bpw < 1` alone never promotes. Not with a caveat,
    not with a flag.
    """
    if not isinstance(ledger, PromotionLedger):
        ledger = extract_ledger(ledger)

    reasons: list[str] = []

    if (
        _present(ledger.prospective_meta_bpw)
        and ledger.prospective_meta_bpw is not None
        and ledger.prospective_meta_bpw.value is not None
        and ledger.prospective_meta_bpw.value < 1.0
    ):
        # Record the trap; it is never itself a reason to promote.
        pass

    ok_ledger, ledger_reason = _self_contained_byte_ledger(ledger.executable_byte_ledger)
    if not ok_ledger:
        reasons.append(ledger_reason)

    if ledger.capability_preserving_runtime is not True:
        reasons.append("no capability-preserving runtime")

    ok_phys, phys_reason = _protected_physical(ledger)
    if not ok_phys:
        reasons.append(phys_reason)

    remat = judge_dense_rematerialization(ledger)
    if ledger.path_kind == VERIFICATION:
        reasons.append(
            "verification path may reconstruct; it cannot promote a production executable"
        )
    elif not remat.ok:
        reasons.append(remat.reason)

    if not reasons:
        return True, "all promotion predicates held"

    if (
        _present(ledger.prospective_meta_bpw)
        and ledger.prospective_meta_bpw is not None
        and ledger.prospective_meta_bpw.value is not None
        and ledger.prospective_meta_bpw.value < 1.0
        and not ok_phys
    ):
        reasons.insert(
            0,
            "prospective_meta_bpw < 1 is a description budget and never promotes alone",
        )
    return False, "; ".join(reasons)


def _tri_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 0:
            return False
        if value == 1:
            return True
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"true", "yes", "required", "present", "hidden", "enabled"}:
            return True
        if key in {
            "false",
            "no",
            "forbidden",
            "banned",
            "absent",
            "disabled",
            "never",
        }:
            return False
        if key in {"unknown", "null", "n/a", "not_measured", ""}:
            return None
    return None


def judge_dense_rematerialization(
    ledger: PromotionLedger | Mapping[str, Any],
) -> RematVerdict:
    """Production must consume the representation directly.

    Verification MAY reconstruct a dense tensor. That asymmetry is the
    whole rule: a reconstruct-for-check path is not a production executable.
    """
    if not isinstance(ledger, PromotionLedger):
        ledger = extract_ledger(ledger)
    path_kind = (ledger.path_kind or PRODUCTION).upper()
    if path_kind not in {PRODUCTION, VERIFICATION}:
        path_kind = PRODUCTION

    remat_flag = _tri_bool(ledger.dense_rematerialization)
    decompress = ledger.decompresses_to_dense_weight_tensor
    if decompress is None:
        decompress = remat_flag
    ordinary = ledger.runs_ordinary_kernels

    if path_kind == VERIFICATION:
        return RematVerdict(
            ok=True,
            path_kind=VERIFICATION,
            reason=(
                "verification MAY reconstruct a dense weight tensor; "
                "this is not a production consumer"
            ),
            decompresses_to_dense_weight_tensor=decompress,
            runs_ordinary_kernels=ordinary,
        )

    if decompress is True and ordinary is True:
        return RematVerdict(
            ok=False,
            path_kind=PRODUCTION,
            reason=(
                "production NX decompresses to a dense weight tensor and then "
                "runs ordinary kernels"
            ),
            decompresses_to_dense_weight_tensor=True,
            runs_ordinary_kernels=True,
        )
    if decompress is True or remat_flag is True:
        return RematVerdict(
            ok=False,
            path_kind=PRODUCTION,
            reason=(
                "production NX rematerializes dense weights; it must consume "
                "the representation directly"
            ),
            decompresses_to_dense_weight_tensor=True if decompress is True else remat_flag,
            runs_ordinary_kernels=ordinary,
        )
    if ledger.consumes_representation_directly is True or remat_flag is False:
        return RematVerdict(
            ok=True,
            path_kind=PRODUCTION,
            reason="production consumes the representation directly",
            decompresses_to_dense_weight_tensor=False,
            runs_ordinary_kernels=ordinary,
        )
    return RematVerdict(
        ok=False,
        path_kind=PRODUCTION,
        reason=(
            "production path does not prove direct representation consumption "
            "(dense rematerialization unknown)"
        ),
        decompresses_to_dense_weight_tensor=decompress,
        runs_ordinary_kernels=ordinary,
    )


Q = TypeVar("Q", bound=Quantity)


def _coerce(cls: type[Q], value: Any, evidence: str, source: str | None = None) -> Q | None:
    if value is None:
        return None
    if isinstance(value, cls):
        return value
    if isinstance(value, Quantity):
        raise CategoryError(
            f"cannot coerce {value.category} into {cls.category}"
        )
    if _nullish(value):
        return None
    number = _as_number(value)
    if number is None:
        return None
    return cls(number, evidence=evidence, source=source)


def _deep_get(node: Any, dotted: str) -> Any:
    cur = node
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _promotion_claimed(doc: Mapping[str, Any]) -> bool:
    if doc.get("promotion_allowed") is True:
        return True
    if doc.get("promotion_claimed") is True:
        return True
    if doc.get("resident_promotion") is True:
        return True
    promo = doc.get("promotion")
    if isinstance(promo, Mapping) and promo.get("allowed") is True:
        return True
    ms = doc.get("measurement_state")
    if isinstance(ms, Mapping) and ms.get("promotion_allowed") is True:
        return True
    qual = doc.get("qualification")
    if isinstance(qual, Mapping) and qual.get("resident_promotion") is True:
        return True
    return False


def _path_kind_of(doc: Mapping[str, Any]) -> str:
    for key in ("path_kind", "consumer"):
        raw = doc.get(key)
        if isinstance(raw, str) and raw.strip().upper() in {PRODUCTION, VERIFICATION}:
            return raw.strip().upper()
    exe = doc.get("execution_path")
    if isinstance(exe, Mapping):
        raw = exe.get("path_kind") or exe.get("consumer")
        if isinstance(raw, str) and raw.strip().upper() in {PRODUCTION, VERIFICATION}:
            return raw.strip().upper()
    return PRODUCTION


def _remat_fields(doc: Mapping[str, Any]) -> dict[str, Any]:
    exe = doc.get("execution_path") if isinstance(doc.get("execution_path"), Mapping) else {}
    meta = doc.get("meta_program") if isinstance(doc.get("meta_program"), Mapping) else {}
    acc = doc.get("accelerator_contract") if isinstance(doc.get("accelerator_contract"), Mapping) else {}
    loader = doc.get("native_loader") if isinstance(doc.get("native_loader"), Mapping) else {}
    measured = doc.get("measured") if isinstance(doc.get("measured"), Mapping) else {}

    dense = (
        exe.get("dense_rematerialization")
        if exe
        else None
    )
    if dense is None:
        dense = doc.get("dense_rematerialization")
    if dense is None:
        dense = acc.get("dense_rematerialization")
    if dense is None:
        dense = meta.get("dense_weight_materialization")
    if dense is None:
        dense = loader.get("dense_rematerialization")
    if dense is None:
        dense = measured.get("hidden_dense_rematerialization")
    if dense is None:
        dense = measured.get("dense_parent_execution_fallback")

    decompress = exe.get("decompresses_to_dense_weight_tensor") if exe else None
    if decompress is None:
        decompress = doc.get("decompresses_to_dense_weight_tensor")
    ordinary = exe.get("runs_ordinary_kernels") if exe else None
    if ordinary is None:
        ordinary = doc.get("runs_ordinary_kernels")
    direct = exe.get("consumes_representation_directly") if exe else None
    if direct is None:
        direct = doc.get("consumes_representation_directly")
    return {
        "dense_rematerialization": dense,
        "decompresses_to_dense_weight_tensor": _tri_bool(decompress),
        "runs_ordinary_kernels": _tri_bool(ordinary),
        "consumes_representation_directly": _tri_bool(direct),
    }


def _active_from(doc: Mapping[str, Any]) -> ActiveBytesAccounting | None:
    block = doc.get("active_bytes")
    if isinstance(block, ActiveBytesAccounting):
        return block
    if not isinstance(block, Mapping):
        block = {}
        # BYTE_LEDGER shape
        fast = doc.get("measured_fastpath_profile")
        exact = doc.get("complete_exact_control")
        if isinstance(fast, Mapping) or isinstance(exact, Mapping):
            fast = fast if isinstance(fast, Mapping) else {}
            exact = exact if isinstance(exact, Mapping) else {}
            return ActiveBytesAccounting(
                total_artifact_bytes=_as_int(exact.get("runtime_required_bytes")),
                resident_bytes=None,
                active_bytes_per_token=_as_int(fast.get("active_bytes_per_token_profile")),
                actual_read_bytes_per_token=_as_int(fast.get("source_bytes_read")),
                transient_bytes=None,
                evidence="FLASH_COMPLETE byte ledger exact-control / fastpath profile",
            )
        keys = (
            "total_artifact_bytes",
            "resident_bytes",
            "active_bytes_per_token",
            "actual_read_bytes_per_token",
            "transient_bytes",
        )
        if not any(k in doc for k in keys):
            return None
        block = doc
    if not any(
        block.get(k) is not None
        for k in (
            "total_artifact_bytes",
            "resident_bytes",
            "active_bytes_per_token",
            "actual_read_bytes_per_token",
            "transient_bytes",
        )
    ):
        return None
    return ActiveBytesAccounting(
        total_artifact_bytes=_as_int(block.get("total_artifact_bytes")),
        resident_bytes=_as_int(block.get("resident_bytes")),
        active_bytes_per_token=_as_int(block.get("active_bytes_per_token")),
        actual_read_bytes_per_token=_as_int(block.get("actual_read_bytes_per_token")),
        transient_bytes=_as_int(block.get("transient_bytes")),
        evidence=str(block.get("evidence") or "ledger fields"),
    )


def _as_int(value: Any) -> int | None:
    number = _as_number(value)
    if number is None:
        return None
    return int(number)


def _bench_state(doc: Mapping[str, Any]) -> str | None:
    bench = doc.get("bench")
    if isinstance(bench, Mapping):
        state = bench.get("state")
        if isinstance(state, str):
            return state
    raw = doc.get("bench_state")
    return raw if isinstance(raw, str) else None


def _measurement_state_label(doc: Mapping[str, Any]) -> str | None:
    raw = doc.get("measurement_state")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        for key in ("physical_ebpw", "complete_token", "capability", "measurement_state"):
            val = raw.get(key)
            if isinstance(val, str) and val:
                # Prefer an explicit authority if present.
                break
        auth = raw.get("authority") or raw.get("measurement_state")
        if isinstance(auth, str):
            return auth
    raw = doc.get("physical_measurement_authority")
    return raw if isinstance(raw, str) else None


def extract_ledger(doc: Mapping[str, Any], source_path: str | None = None) -> PromotionLedger:
    """Lift a receipt or compact test ledger into typed quantities."""
    schema = doc.get("schema") if isinstance(doc.get("schema"), str) else None
    status = doc.get("status") if isinstance(doc.get("status"), str) else None
    remat = _remat_fields(doc)
    notes: list[str] = []

    prospective = _coerce(
        ProspectiveMetaBpw,
        doc.get("prospective_meta_bpw"),
        "ledger.prospective_meta_bpw",
        source_path,
    )
    serialized = _coerce(
        SerializedMetaBytes,
        doc.get("serialized_meta_bytes"),
        "ledger.serialized_meta_bytes",
        source_path,
    )
    nr_bytes = _coerce(
        CompileTimeNrBytes,
        doc.get("compile_time_nr_bytes"),
        "ledger.compile_time_nr_bytes",
        source_path,
    )
    nx_bytes = _coerce(
        NxRuntimeBytes,
        doc.get("nx_runtime_bytes"),
        "ledger.nx_runtime_bytes",
        source_path,
    )
    physical = _coerce(
        CompletePhysicalEbpw,
        doc.get("complete_physical_ebpw"),
        "ledger.complete_physical_ebpw",
        source_path,
    )

    byte_ledger = doc.get("executable_byte_ledger")
    if not isinstance(byte_ledger, Mapping):
        byte_ledger = None

    if schema == "hawking.flash.meta_representation.v1":
        metric = doc.get("metric") if isinstance(doc.get("metric"), Mapping) else {}
        ms = doc.get("measurement_state") if isinstance(doc.get("measurement_state"), Mapping) else {}
        prospective = _coerce(
            ProspectiveMetaBpw,
            metric.get("prospective_target", metric.get("value")),
            "metric.prospective_target (description budget)",
            source_path,
        )
        if not _nullish(ms.get("serialized_artifact")):
            serialized = _coerce(
                SerializedMetaBytes,
                doc.get("serialized_meta_bytes"),
                "measurement_state.serialized_artifact",
                source_path,
            )
        else:
            notes.append("serialized_artifact=NOT_BUILT")
        if not _nullish(metric.get("physical_ebpw")) or not _nullish(ms.get("physical_ebpw")):
            # A number here is a laundering attempt, not a real physical quantity.
            notes.append("physical_ebpw claimed on a prospective meta receipt")
            physical = None
        else:
            physical = None
            notes.append("physical_ebpw=NULL_BY_RULE")
        if remat["consumes_representation_directly"] is None and _tri_bool(
            remat["dense_rematerialization"]
        ) is False:
            remat = {**remat, "consumes_representation_directly": True}

    elif isinstance(schema, str) and schema.startswith("hawking.flash.complete_nr"):
        family = _deep_get(doc, "representation.family_source_bytes")
        if isinstance(family, Mapping) and family:
            total = sum(_as_number(v) or 0.0 for v in family.values())
            nr_bytes = CompileTimeNrBytes(
                total,
                evidence="sum(representation.family_source_bytes) [NR exact-control, source-sized]",
                source=source_path,
            )
        bpw = _deep_get(doc, "representation.complete_bits_per_weight")
        if _as_number(bpw) is not None:
            notes.append(
                f"representation.complete_bits_per_weight={bpw} is an NR description, "
                "not complete_physical_ebpw"
            )

    elif schema == "hawking.flash.nx_genome.v1":
        runtime = _deep_get(doc, "qualification.runtime_required_bytes")
        nx_bytes = _coerce(
            NxRuntimeBytes,
            runtime if runtime is not None else doc.get("nx_runtime_bytes"),
            "qualification.runtime_required_bytes",
            source_path,
        )
        claimed = _deep_get(doc, "qualification.complete_system_ebpw")
        if not _nullish(claimed):
            physical = _coerce(
                CompletePhysicalEbpw,
                claimed,
                "qualification.complete_system_ebpw",
                source_path,
            )
        else:
            notes.append("qualification.complete_system_ebpw is null; NX is metadata only")
        # Shader/executor source bytes are not nx_runtime_bytes.
        notes.append("physical_program executor/shader bytes are source, not runtime working set")

    elif schema == "hawking.flash.complete_byte_ledger.v1":
        exact = doc.get("complete_exact_control") if isinstance(doc.get("complete_exact_control"), Mapping) else {}
        nr_bytes = _coerce(
            CompileTimeNrBytes,
            exact.get("runtime_required_bytes"),
            "complete_exact_control.runtime_required_bytes (exact-control, source-sized)",
            source_path,
        )
        ebpw = exact.get("complete_ebpw")
        if _as_number(ebpw) is not None:
            notes.append(
                f"complete_exact_control.complete_ebpw={ebpw} is BF16 identity, "
                "not a protected packed-NX measurement"
            )
        byte_ledger = {
            "self_contained": True,
            "for_this_executable": False,
            "complete_storage_bytes": exact.get("runtime_required_bytes"),
            "kind": "exact_control_source_identity",
        }

    elif schema == "hcli.agentos.flash_ebpw_budget.v1":
        ceiling = _deep_get(doc, "target_contract.complete_system_ebpw_max")
        prospective = _coerce(
            ProspectiveMetaBpw,
            ceiling,
            "target_contract.complete_system_ebpw_max (budget, not measurement)",
            source_path,
        )
        measured = doc.get("measured") if isinstance(doc.get("measured"), Mapping) else {}
        if not _nullish(measured.get("complete_system_ebpw")):
            physical = _coerce(
                CompletePhysicalEbpw,
                measured.get("complete_system_ebpw"),
                "measured.complete_system_ebpw",
                source_path,
            )
        else:
            notes.append("measured.complete_system_ebpw is null")
        actual = _deep_get(doc, "measured.complete_system_bytes")
        serialized = _coerce(
            SerializedMetaBytes,
            actual,
            "measured.complete_system_bytes",
            source_path,
        )

    elif schema == "hawking.ascension.qwen80_bit_budget_ledger.v1":
        prospective = _coerce(
            ProspectiveMetaBpw,
            doc.get("ceiling_complete_physical_bpw"),
            "ceiling_complete_physical_bpw (BUDGET_ENVELOPE_ONLY, not an artifact)",
            source_path,
        )
        notes.append("status=BUDGET_ENVELOPE_ONLY_NO_ARTIFACT_PACKED_YET")

    elif schema == "hawking.odyssey.ebpw_namespace_separation.v1":
        notes.append(
            "DESIGN_EXPECTED vs ARTIFACT_PHYSICAL vs RUNTIME_MEASURED naming; "
            "not the five-category type system"
        )

    capability = doc.get("capability_preserving_runtime")
    if capability is not True:
        # Receipts may encode this under measurement_state.capability.
        cap = _deep_get(doc, "measurement_state.capability")
        if isinstance(cap, str) and cap.upper() in {"PRESERVED", "PASSED", "MATCHED"}:
            capability = True
        else:
            capability = False

    authority = doc.get("physical_measurement_authority")
    if not isinstance(authority, str):
        ms = doc.get("measurement_state")
        if isinstance(ms, Mapping) and isinstance(ms.get("authority"), str):
            authority = ms.get("authority")
        else:
            authority = None

    if byte_ledger is None and isinstance(doc.get("executable_byte_ledger"), Mapping):
        byte_ledger = doc.get("executable_byte_ledger")

    return PromotionLedger(
        prospective_meta_bpw=prospective,
        serialized_meta_bytes=serialized,
        compile_time_nr_bytes=nr_bytes,
        nx_runtime_bytes=nx_bytes,
        complete_physical_ebpw=physical,
        executable_byte_ledger=byte_ledger,
        capability_preserving_runtime=bool(capability),
        physical_measurement_authority=authority if isinstance(authority, str) else None,
        bench_state=_bench_state(doc),
        measurement_state=_measurement_state_label(doc),
        path_kind=_path_kind_of(doc),
        dense_rematerialization=remat["dense_rematerialization"],
        decompresses_to_dense_weight_tensor=remat["decompresses_to_dense_weight_tensor"],
        runs_ordinary_kernels=remat["runs_ordinary_kernels"],
        consumes_representation_directly=remat["consumes_representation_directly"],
        active_bytes=_active_from(doc),
        promotion_claimed=_promotion_claimed(doc),
        source_path=source_path,
        schema=schema,
        status=status,
        notes=tuple(notes),
    )


def _launders_physical_from_meta(doc: Mapping[str, Any]) -> bool:
    metric = doc.get("metric") if isinstance(doc.get("metric"), Mapping) else {}
    ms = doc.get("measurement_state") if isinstance(doc.get("measurement_state"), Mapping) else {}
    status = doc.get("status")
    prospective = _as_number(metric.get("prospective_target", doc.get("prospective_meta_bpw")))
    physical = metric.get("physical_ebpw", ms.get("physical_ebpw", doc.get("complete_physical_ebpw")))
    if status == "PROSPECTIVE_META_ONLY" and not _nullish(physical):
        return True
    if (
        prospective is not None
        and prospective < 1.0
        and not _nullish(physical)
        and _nullish(ms.get("serialized_artifact", "NOT_BUILT") if ms else "NOT_BUILT")
    ):
        return True
    return False


def _quantities_report(ledger: PromotionLedger) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, qty in (
        ("prospective_meta_bpw", ledger.prospective_meta_bpw),
        ("serialized_meta_bytes", ledger.serialized_meta_bytes),
        ("compile_time_nr_bytes", ledger.compile_time_nr_bytes),
        ("nx_runtime_bytes", ledger.nx_runtime_bytes),
        ("complete_physical_ebpw", ledger.complete_physical_ebpw),
    ):
        if _present(qty) and qty is not None:
            out[name] = qty.as_dict()
        else:
            out[name] = {
                "category": name,
                "present": False,
                "value": None,
                "evidence": "absent on this ledger",
            }
            if _present(qty) is False and qty is not None:
                out[name]["evidence"] = qty.evidence
    return out


def doctor_zeros_for_doc(
    doc: Mapping[str, Any], source_path: str | None = None
) -> dict[str, Any]:
    """Call Doctor's three-zeros checks on organs this ledger names.

    A low-BPW figure that fails all three zeros is ordinary quantization
    (H-ROADMAP §9.2). This invokes ``tools.doctor.zeros.check_three_zeros``;
    importing the module is not a call site.
    """
    from tools.doctor.zeros import (
        PASS,
        check_three_zeros,
        ordinary_quantization,
        organs_from_doc,
        whole_artifact_organ,
        zeros_as_dict,
    )

    organs = organs_from_doc(doc)
    if not organs:
        organs = [whole_artifact_organ(doc)]
    rows: list[dict[str, Any]] = []
    any_ordinary = False
    any_zero = False
    for organ in organs:
        results = check_three_zeros(organ)
        ordinary = ordinary_quantization(results)
        if ordinary:
            any_ordinary = True
        if any(r.verdict == PASS for r in results.values()):
            any_zero = True
        rows.append(
            {
                "name": organ.get("name") or organ.get("organ_class"),
                "three_zeros": zeros_as_dict(results),
                "ordinary_quantization": ordinary,
            }
        )
    return {
        "applied": True,
        "called": "tools.doctor.zeros.check_three_zeros",
        "n_organs": len(rows),
        "organs": rows,
        "any_ordinary_quantization": any_ordinary,
        "any_zero_available": any_zero,
        "note": (
            "a low-BPW result that fails all three zeros may merely be ordinary "
            "quantization (roadmap 9.2). This annotation does not promote."
        ),
        "source_path": source_path,
        "evidence_tier": "STATIC",
    }


def validate(doc: Mapping[str, Any], source_path: str | None = None) -> dict[str, Any]:
    """GREEN if honest (including honest non-promotion). REFUSED if it launders."""
    ledger = extract_ledger(doc, source_path=source_path)
    promote_ok, promote_reason = can_promote(ledger)
    remat = judge_dense_rematerialization(ledger)
    refused: list[str] = []

    proven_remat = ledger.path_kind != VERIFICATION and (
        ledger.decompresses_to_dense_weight_tensor is True
        or _tri_bool(ledger.dense_rematerialization) is True
    )
    if proven_remat:
        refused.append(remat.reason)
    if ledger.promotion_claimed and not promote_ok:
        refused.append("promotion claimed without predicates: " + promote_reason)
    if _launders_physical_from_meta(doc):
        refused.append("physical EBPW claimed from a prospective meta budget")

    zeros_block = doctor_zeros_for_doc(doc, source_path=source_path)

    verdict = "REFUSED" if refused else "GREEN"
    return {
        "verdict": verdict,
        "can_promote": promote_ok,
        "can_promote_reason": promote_reason,
        "promotion_claimed": ledger.promotion_claimed,
        "dense_rematerialization": remat.as_dict(),
        "quantities": _quantities_report(ledger),
        "active_bytes": ledger.active_bytes.as_dict() if ledger.active_bytes else None,
        "schema": ledger.schema,
        "status": ledger.status,
        "notes": list(ledger.notes),
        "reasons": refused if refused else [promote_reason],
        "source_path": source_path,
        "doctor_three_zeros": zeros_block,
    }


@lru_cache(maxsize=1)
def _primary_worktree() -> Path | None:
    raw = git("worktree", "list", "--porcelain")
    for line in raw.splitlines():
        if line.startswith("worktree "):
            p = Path(line.split(" ", 1)[1])
            if p.is_dir():
                return p
    return None


def resolve_repo_path(rel: str | Path) -> Path | None:
    """Repo-relative path, then the primary worktree (untracked receipts)."""
    rel_s = str(rel)
    path = Path(rel_s)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(REPO / rel_s)
        main = _primary_worktree()
        if main is not None:
            alt = main / rel_s
            if alt not in candidates:
                candidates.append(alt)
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_named_receipt(rel: str) -> tuple[dict[str, Any] | None, str]:
    found = resolve_repo_path(rel)
    if found is None:
        return None, "missing"
    via = "local" if found == (REPO / rel) or found == Path(rel) else "primary_worktree"
    return load_json(found), via


def inventory_path(rel: str) -> dict[str, Any]:
    doc, via = load_named_receipt(rel)
    row: dict[str, Any] = {
        "path": rel,
        "present": doc is not None,
        "resolved_via": via if doc is not None else "missing",
    }
    if doc is None:
        row["quantities"] = {
            name: {"present": False, "value": None}
            for name in CATEGORY_TYPES
        }
        row["can_promote"] = False
        row["can_promote_reason"] = "receipt not on disk in this checkout"
        row["verdict"] = "ABSENT"
        return row
    result = validate(doc, source_path=rel)
    row.update(
        {
            "schema": result["schema"],
            "status": result["status"],
            "verdict": result["verdict"],
            "can_promote": result["can_promote"],
            "can_promote_reason": result["can_promote_reason"],
            "promotion_claimed": result["promotion_claimed"],
            "quantities": {
                name: {
                    "present": isinstance(info, dict) and info.get("value") is not None,
                    "value": info.get("value") if isinstance(info, dict) else None,
                    "evidence": info.get("evidence") if isinstance(info, dict) else None,
                }
                for name, info in result["quantities"].items()
            },
            "notes": result["notes"],
            "dense_rematerialization": result["dense_rematerialization"],
            "doctor_three_zeros": {
                "called": (result.get("doctor_three_zeros") or {}).get("called"),
                "any_ordinary_quantization": (result.get("doctor_three_zeros") or {}).get(
                    "any_ordinary_quantization"
                ),
                "any_zero_available": (result.get("doctor_three_zeros") or {}).get(
                    "any_zero_available"
                ),
            },
        }
    )
    return row


def _run_negative_controls() -> dict[str, Any]:
    """Watch the guards fail. A guard nobody has watched fail is not a guard."""
    results: dict[str, Any] = {}

    meta_only = PromotionLedger(
        prospective_meta_bpw=ProspectiveMetaBpw(
            0.88, evidence="synthetic negative control"
        )
    )
    ok, reason = can_promote(meta_only)
    results["meta_only_0_88_refused"] = ok is False
    results["meta_only_0_88_reason"] = reason
    if ok:
        raise AssertionError("can_promote accepted a meta-only 0.88 ledger")

    flagged = {
        "prospective_meta_bpw": 0.88,
        "promotion_allowed": True,
        "promotion_caveat": "budget only, but ship it",
        "force_promote": True,
    }
    flagged_ok, flagged_reason = can_promote(extract_ledger(flagged))
    results["meta_with_caveat_flag_refused"] = flagged_ok is False
    results["meta_with_caveat_flag_reason"] = flagged_reason
    if flagged_ok:
        raise AssertionError("can_promote accepted a caveat/flag around meta-only 0.88")

    mixed_raised = False
    mixed_message = ""
    try:
        _ = ProspectiveMetaBpw(0.88) + CompletePhysicalEbpw(0.88)
    except CategoryError as exc:
        mixed_raised = True
        mixed_message = str(exc)
    results["cross_category_arithmetic_raises"] = mixed_raised
    results["cross_category_arithmetic_message"] = mixed_message
    if not mixed_raised:
        raise AssertionError("cross-category arithmetic did not raise")

    remat_nx = {
        "schema": "hawking.flash.nx_genome.v1",
        "path_kind": PRODUCTION,
        "execution_path": {
            "decompresses_to_dense_weight_tensor": True,
            "runs_ordinary_kernels": True,
        },
    }
    remat = judge_dense_rematerialization(extract_ledger(remat_nx))
    remat_validate = validate(remat_nx)
    results["dense_rematerializing_nx_rejected"] = remat.ok is False and remat_validate["verdict"] == "REFUSED"
    results["dense_rematerializing_nx_reason"] = remat.reason
    if remat.ok or remat_validate["verdict"] != "REFUSED":
        raise AssertionError("dense-rematerializing production NX was not refused")

    verify_nx = {
        "path_kind": VERIFICATION,
        "execution_path": {
            "decompresses_to_dense_weight_tensor": True,
            "runs_ordinary_kernels": True,
        },
    }
    v_remat = judge_dense_rematerialization(extract_ledger(verify_nx))
    results["verification_may_reconstruct"] = v_remat.ok is True
    v_ok, v_reason = can_promote(extract_ledger(verify_nx))
    results["verification_cannot_promote"] = v_ok is False
    results["verification_cannot_promote_reason"] = v_reason
    if not v_remat.ok or v_ok:
        raise AssertionError("verification/production asymmetry broken")

    # Combinator can open: proves can_promote is not a constant False.
    full = PromotionLedger(
        complete_physical_ebpw=CompletePhysicalEbpw(
            2.4, evidence="synthetic combinator control (not a measurement)"
        ),
        executable_byte_ledger={
            "self_contained": True,
            "for_this_executable": True,
            "complete_storage_bytes": 4096,
        },
        capability_preserving_runtime=True,
        physical_measurement_authority=PROTECTED,
        bench_state="PROTECTED",
        measurement_state=PROTECTED,
        path_kind=PRODUCTION,
        dense_rematerialization=False,
        consumes_representation_directly=True,
    )
    full_ok, full_reason = can_promote(full)
    results["positive_combinator_opens"] = full_ok is True
    results["positive_combinator_reason"] = full_reason
    results["positive_combinator_is_not_a_measurement"] = True
    if not full_ok:
        raise AssertionError(f"positive combinator failed to open: {full_reason}")

    # Doctor 9.2: watch the three zeros FAIL on a broken organ and PASS on
    # good fixtures. This is a call of tools.doctor.zeros.check_three_zeros,
    # not an import-only mention.
    from tools.doctor.zeros import (
        FAIL as ZFAIL,
        PASS as ZPASS,
        BROKEN_ORGAN,
        ROUTED_EXPERT_ORGAN,
        TIED_EMBED_ORGAN,
        check_three_zeros,
        ordinary_quantization,
    )

    broken = check_three_zeros(BROKEN_ORGAN)
    tied = check_three_zeros(TIED_EMBED_ORGAN)
    routed = check_three_zeros(ROUTED_EXPERT_ORGAN)
    results["doctor_three_zeros_fail_on_broken"] = ordinary_quantization(broken) is True
    results["doctor_zero_storage_fail_on_broken"] = broken["ZERO_STORAGE"].verdict == ZFAIL
    results["doctor_zero_info_fail_on_broken"] = (
        broken["ZERO_INDEPENDENT_INFORMATION"].verdict == ZFAIL
    )
    results["doctor_zero_execution_fail_on_broken"] = (
        broken["ZERO_EXECUTION"].verdict == ZFAIL
    )
    results["doctor_zero_storage_pass_on_tied"] = tied["ZERO_STORAGE"].verdict == ZPASS
    results["doctor_zero_info_pass_on_tied"] = (
        tied["ZERO_INDEPENDENT_INFORMATION"].verdict == ZPASS
    )
    results["doctor_zero_execution_pass_on_routed"] = (
        routed["ZERO_EXECUTION"].verdict == ZPASS
    )
    if not results["doctor_three_zeros_fail_on_broken"]:
        raise AssertionError("Doctor three zeros did not FAIL on the broken organ")
    if not results["doctor_zero_storage_pass_on_tied"]:
        raise AssertionError("ZERO_STORAGE did not PASS on tied embeddings")
    if not results["doctor_zero_execution_pass_on_routed"]:
        raise AssertionError("ZERO_EXECUTION did not PASS on a routed expert")

    return results


HONEST_FLASH_META_MINIMAL: dict[str, Any] = {
    "schema": "hawking.flash.meta_representation.v1",
    "status": "PROSPECTIVE_META_ONLY",
    "metric": {
        "name": "meta_bpw",
        "unit": "bits per source weight (description budget, not serialized bytes)",
        "traditional_physical_metric": "physical_ebpw",
        "physical_ebpw": None,
        "prospective_target": 0.8871807728336929,
        "below_one_target": True,
    },
    "measurement_state": {
        "meta_budget": "COMPUTED_FROM_CENSUS_AND_PREREGISTERED_TARGETS",
        "serialized_artifact": "NOT_BUILT",
        "physical_loader": "NOT_BUILT",
        "native_kernel": "NOT_BUILT",
        "complete_token": "NOT_MEASURED",
        "capability": "NOT_MEASURED",
        "physical_ebpw": "NULL_BY_RULE",
        "promotion_allowed": False,
    },
    "meta_program": {"dense_weight_materialization": "forbidden"},
    "accelerator_contract": {"dense_rematerialization": False},
    "bench": {"state": "UNKNOWN"},
}


def validate_honest_flash_meta() -> dict[str, Any]:
    doc, via = load_named_receipt("receipts/headless/FLASH_META_REPRESENTATION_SUB1.json")
    if doc is None:
        result = validate(HONEST_FLASH_META_MINIMAL, source_path="HONEST_FLASH_META_MINIMAL")
        result["resolved_via"] = "embedded_minimal_fixture"
        return result
    result = validate(doc, source_path="receipts/headless/FLASH_META_REPRESENTATION_SUB1.json")
    result["resolved_via"] = via
    return result


def selftest() -> dict[str, Any]:
    controls = _run_negative_controls()
    honest = validate_honest_flash_meta()
    if honest["verdict"] != "GREEN":
        raise AssertionError(f"honest FLASH_META must be GREEN, got {honest}")
    if honest["can_promote"] is not False:
        raise AssertionError("honest FLASH_META must not promote")
    synthetic = dict(HONEST_FLASH_META_MINIMAL)
    synthetic = json.loads(json.dumps(synthetic))
    synthetic["measurement_state"] = dict(synthetic["measurement_state"])
    synthetic["measurement_state"]["promotion_allowed"] = True
    synthetic["measurement_state"]["physical_ebpw"] = 0.88
    synthetic["metric"] = dict(synthetic["metric"])
    synthetic["metric"]["physical_ebpw"] = 0.88
    synthetic["promotion_allowed"] = True
    refused = validate(synthetic, source_path="synthetic_promotion_variant")
    if refused["verdict"] != "REFUSED":
        raise AssertionError(f"synthetic promotion variant must be REFUSED, got {refused}")
    controls["honest_flash_meta_green"] = True
    controls["synthetic_promotion_variant_refused"] = True
    controls["honest_flash_meta"] = {
        "verdict": honest["verdict"],
        "can_promote": honest["can_promote"],
        "resolved_via": honest.get("resolved_via"),
        "quantities_present": {
            k: v.get("value") is not None if isinstance(v, dict) else False
            for k, v in honest["quantities"].items()
        },
    }
    return controls


def _recovered_implementation() -> list[dict[str, str]]:
    return [
        {
            "path": "receipts/headless/PHYSICAL_METRIC_AUDIT.json",
            "what": (
                "Historical DESIGN_EXPECTED vs ARTIFACT_PHYSICAL vs RUNTIME_MEASURED naming "
                "for the same kind of byte/EBPW figure. Watched-failing canaries. "
                "Does not type-separate the five quantities and does not make "
                "cross-category arithmetic a type error."
            ),
        },
        {
            "path": "receipts/headless/EBPW_NAMESPACE_SEPARATION.json",
            "what": "Sealed G059 evidence for the naming split above.",
        },
        {
            "path": "tools/flash_meta_representation.py",
            "what": (
                "Emits FLASH_META_REPRESENTATION_SUB1.json as PROSPECTIVE_META_ONLY "
                "with physical_ebpw NULL_BY_RULE. Honest, but not a type system."
            ),
        },
        {
            "path": "receipts/headless/FLASH_META_REPRESENTATION_SUB1.json",
            "what": (
                "Untracked in the primary worktree; not in git HEAD. Honest "
                "sub-1 description budget (prospective_target ~0.887)."
            ),
        },
        {
            "path": "tools/flash_complete_byte_ledger.py",
            "what": (
                "Exact-control Flash byte ledger closed at 16.0 EBPW (BF16 identity). "
                "Compact EBPW left open. Untracked in git HEAD."
            ),
        },
        {
            "path": "tools/accelerator/bytes_atlas.py",
            "what": (
                "STATIC token-byte atlas; reconciles 8*bytes/params against a sealed "
                "complete EBPW for the Noetic parent. Not the five-category validator."
            ),
        },
        {
            "path": "receipts/headless/FLASH_EBPW_BUDGET.json",
            "what": (
                "Flash organ budget; measured.complete_system_ebpw is null; "
                "promotion_allowed is false; hidden_dense_rematerialization is null."
            ),
        },
        {
            "path": "receipts/QWEN80_BIT_BUDGET_LEDGER.json",
            "what": (
                "Qwen80 search envelope (ceiling_complete_physical_bpw=1.5). "
                "status=BUDGET_ENVELOPE_ONLY_NO_ARTIFACT_PACKED_YET."
            ),
        },
        {
            "path": "receipts/headless/FLASH_COMPLETE_V2.nr.json",
            "what": "NR candidate, complete_bits_per_weight=16.0, promotion.allowed=false.",
        },
        {
            "path": "receipts/headless/FLASH_COMPLETE_V0.nx.json",
            "what": "NX metadata seal, complete_system_ebpw null, resident_promotion false.",
        },
    ]


def build() -> Path:
    controls = selftest()
    rows = [inventory_path(p) for p in INVENTORY_PATHS]
    # Drop the headless QWEN80 path if the receipts/ copy is the one that exists.
    seen: set[str] = set()
    inventory = []
    for row in rows:
        key = Path(row["path"]).name
        if key in seen and not row["present"]:
            continue
        if row["present"]:
            seen.add(key)
        inventory.append(row)

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Make category confusion between a prospective meta-BPW budget and "
            "a complete physical EBPW structurally impossible, so no future run "
            "can promote a sub-1 claim on a budget alone."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "categories": {
            name: {"type": cls.__name__, "unit": cls.unit, "interchangeable": False}
            for name, cls in CATEGORY_TYPES.items()
        },
        "promotion_rule": {
            "can_promote_requires": [
                "self-contained executable byte ledger for this executable",
                "capability-preserving runtime",
                "complete_physical_ebpw backed by PROTECTED_ABSOLUTE measurement",
                "production consumes the representation directly (no dense rematerialization)",
            ],
            "never_sufficient": [
                "prospective_meta_bpw < 1",
                "a caveat, flag, or promotion_allowed bit on a meta-only ledger",
                "DIAGNOSTIC_RELATIVE measurement",
                "verification-path reconstruction",
            ],
            "authority_split": {
                "DIAGNOSTIC_RELATIVE": "contaminated A/B on a busy machine. Guides. Never promotes.",
                "PROTECTED_ABSOLUTE": "measurement under a real protected GPU lease. Decides.",
                "STATIC_ONLY": "this sidecar. Everything it emits has bench.state UNKNOWN.",
            },
        },
        "dense_rematerialization": {
            "production": "must consume the representation directly; decompress-then-ordinary-kernels is refused",
            "verification": "MAY reconstruct a dense tensor for checking; that path cannot promote",
        },
        "active_bytes_fields": [
            "total_artifact_bytes",
            "resident_bytes",
            "active_bytes_per_token",
            "actual_read_bytes_per_token",
            "transient_bytes",
        ],
        "inventory": inventory,
        "selftest": controls,
        "recovered_implementation": _recovered_implementation(),
        "gaps_closed": [
            "Five Quantity subclasses; mixed arithmetic/comparison/coercion raises CategoryError.",
            "can_promote is false unless byte ledger + capability-preserving runtime + protected complete_physical_ebpw + direct production consumption all hold.",
            "prospective_meta_bpw < 1 is never a promotion predicate, with or without a flag.",
            "Dense rematerialization: production refused, verification allowed, verification cannot promote.",
            "Active-bytes accounting is five separate fields; collapsing them raises.",
            "Validator is GREEN on honest FLASH_META_REPRESENTATION_SUB1 and REFUSED on a synthetic promotion variant.",
            "validate() calls tools.doctor.zeros.check_three_zeros; a low-BPW ledger that fails all three zeros is ordinary quantization (roadmap 9.2) and still does not promote.",
        ],
        "negative_findings": [
            "FLASH_META_REPRESENTATION_SUB1.json, FLASH_COMPLETE_V2.nr.json, FLASH_COMPLETE_V0.nx.json, and tools/flash_complete_byte_ledger.py are not in git HEAD; they are untracked in the primary worktree.",
            "This sparse checkout does not materialize those untracked files; validate/inventory resolve them read-only from the primary worktree when present.",
            "No inspected receipt carries complete_physical_ebpw backed by PROTECTED_ABSOLUTE.",
            "This sidecar has no GPU and cannot produce the missing physical quantity.",
            "tools/accelerator/bytes_atlas.py is not materialized in this sparse checkout (present in git; recovered via git show).",
            "QWEN80_BIT_BUDGET_LEDGER.json lives at receipts/, not receipts/headless/.",
            "A PROTECTED_ABSOLUTE string inside a JSON ledger is structural; this validator does not confirm a real GPU lease occurred.",
        ],
        "integration": {
            "can_promote": "can_promote(ledger: PromotionLedger | Mapping) -> tuple[bool, str]",
            "validate": "validate(doc: Mapping, source_path: str | None = None) -> dict",
            "extract_ledger": "extract_ledger(doc: Mapping, source_path: str | None = None) -> PromotionLedger",
            "judge_dense_rematerialization": "judge_dense_rematerialization(ledger) -> RematVerdict",
            "doctor_zeros_for_doc": (
                "doctor_zeros_for_doc(doc) -> dict; CALLS "
                "tools.doctor.zeros.check_three_zeros (import is not a call site)"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/ebpw_categories.py")


def _print_validation(rel: str) -> int:
    found = resolve_repo_path(rel)
    if found is None:
        print(f"ABSENT {rel}", file=sys.stderr)
        print("receipt not on disk in this checkout or the primary worktree", file=sys.stderr)
        return 2
    doc = load_json(found)
    result = validate(doc, source_path=str(rel))
    via = "local" if found == (REPO / rel) or found == Path(rel) else "primary_worktree"
    print(f"{result['verdict']} {rel}")
    print(f"resolved_via={via}")
    print(f"status={result['status']}")
    print(f"can_promote={result['can_promote']}")
    print(f"reason={result['can_promote_reason']}")
    for name, info in result["quantities"].items():
        present = info.get("value") is not None
        print(f"  {name}: {'PRESENT '+repr(info.get('value')) if present else 'ABSENT'}")
        if info.get("evidence"):
            print(f"    evidence={info['evidence']}")
    print(f"dense_rematerialization.ok={result['dense_rematerialization']['ok']}")
    print(f"dense_rematerialization.reason={result['dense_rematerialization']['reason']}")
    if result["verdict"] == "REFUSED":
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--validate", metavar="PATH")
    a = ap.parse_args()
    if a.validate:
        return _print_validation(a.validate)
    if a.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
