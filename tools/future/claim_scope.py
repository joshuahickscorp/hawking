"""CLAIM_SCOPE — time-index the scientific universe so scope cannot drift.

ModelLake is streaming while Odyssey I runs. Specimens appear during the
mission. A law discovered when three specimens were sealed must not later
read as though it had been tested against six. That is hindsight
contamination, and this module is the refusal.

Rules this sidecar makes executable:

* Every law and experiment records WHICH SPECIMENS WERE AVAILABLE AT THE
  TIME. Later expansion distinguishes ORIGINAL SCOPE from REPLICATION
  SCOPE from FAILED TRANSFER.
* A law from the initial constellation BEGINS narrow — MODEL_LOCAL,
  ORGAN_LOCAL, MACHINE_LOCAL — and widens ONLY as a newly sealed specimen
  replicates it. Widening without a named replicating specimen RAISES.
  That refusal is the deliverable.
* Every experiment binds six identity fields: specimen seal, model
  revision, resident identity, code and build identity, machine genome,
  and the laws/scars version it was judged against.
* Every conclusion names its evidence universe: "within currently sealed
  specimens A/B/C". "All Hawking models behave this way" is refused while
  D and E are still downloading.

No GPU lease. Numbers that appear in retro-scoped statements are CITED
from named prior receipts. This sidecar does not remeasure them.

    python3 tools/future/claim_scope.py --build
    python3 -m pytest tools/future/test_claim_scope.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.future._common import RECEIPTS, REPO, git, load_json, write_receipt
from tools.future import specimen_verify as sv


RECEIPT = "CLAIM_SCOPE.json"
SCHEMA = "hawking.future.claim_scope.v1"
VERSION = 1
RECORDED_BY = "tools/future/claim_scope.py"
EVIDENCE_CLASS = "STATIC_ONLY"

EXTERNAL_SEAL_REL = "receipts/future/EXTERNAL_SPECIMEN_SEAL.json"
VERIFICATION_REL = "receipts/future/SPECIMEN_VERIFICATION.json"
CURRICULUM_REL = "receipts/future/SPECIMEN_CURRICULUM.json"
CAMPAIGN_SCARS_REL = "receipts/future/CAMPAIGN_SCARS.json"
LAW_STORE_REL = "receipts/future/ODYSSEY2_LAW_STORE.json"
NEGATIVE_INDEX_REL = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"
RESIDENT_IDENTITY_REL = "receipts/future/RESIDENT_IDENTITY.json"
MACHINE_GENOME_REL = "receipts/headless/MACHINE_GENOME.json"
DIRTY_SOURCE_REL = "receipts/future/DIRTY_SOURCE_SEAL.json"

ALU_REL = "receipts/future/MLP_ALU_ROOFLINE.json"
ECON_CAL_REL = "receipts/future/ECONOMICS_CALIBRATION.json"
STRUCT_REL = "receipts/future/MLP_STRUCTURED_OPERATOR.json"
FOLD_REL = "receipts/future/FOLD_ADDQX_AB.json"
ROOF_REL = "receipts/future/ROOF_ANCHOR.json"
TEACHER_REL = "receipts/future/MLP_TEACHER_CORPUS.json"

PARENT_SPECIMEN = "qwen3.8-27b-abliterated-bf16@local"
PARENT_MODEL_ID = "qwen3.8-27b-sealed-3.14"
PARENT_RESIDENT = "sealed-3.14"
PARENT_ORGAN_MLP = "mlp"
PARENT_MACHINE = "Apple M3 Ultra (MACHINE_GENOME.json; not remeasured)"

TEACHER_LAYERS = (3, 31, 38, 63)
TEACHER_ROWS = 45_076

AXES = ("model", "organ", "machine")
NARROW_TIERS = {
    "model": "MODEL_LOCAL",
    "organ": "ORGAN_LOCAL",
    "machine": "MACHINE_LOCAL",
}
REPLICATED_TIERS = {
    "model": "MODEL_REPLICATED",
    "organ": "ORGAN_REPLICATED",
    "machine": "MACHINE_REPLICATED",
}
SCOPE_KINDS = ("ORIGINAL", "REPLICATION", "FAILED_TRANSFER")
EXPERIMENT_IDENTITY_FIELDS = (
    "specimen_seal",
    "model_revision",
    "resident_identity",
    "code_and_build_identity",
    "machine_genome",
    "laws_scars_version",
)

# Placeholders that are not a named specimen. Widening with these RAISES.
NOT_A_SPECIMEN = frozenset(
    {
        "",
        "all",
        "all models",
        "all hawking models",
        "every model",
        "any model",
        "generic",
        "family",
        "universe",
        "currently sealed",
        "*",
        "none",
        "unknown",
    }
)

UNIVERSAL_CLAIM_MARKERS = (
    "all hawking models",
    "all models behave",
    "every hawking model",
    "every model behaves",
    "models in general",
    "universally true",
    "in general for all models",
)

LAKE_SEAL_NAME = "MODEL_LAKE_SPECIMEN_SEAL.json"
HF_DOWNLOAD = Path(".cache") / "huggingface" / "download"
# A partial last touched longer ago than this is a stalled download, not in flight.
IN_FLIGHT_MAX_AGE_S = 36 * 3600

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Available-set and "
    "seal times are read from ModelLake metadata, EXTERNAL_SPECIMEN_SEAL, "
    "and SPECIMEN_VERIFICATION. Scope begins MODEL_LOCAL / ORGAN_LOCAL / "
    "MACHINE_LOCAL and cannot widen without a named replicating specimen. "
    "Cited GB/s, ns, and ms figures are copied from named prior receipts "
    "and are not remeasured here."
)


class ClaimScopeError(ValueError):
    """Contract violation around time-indexed scientific scope."""


class ScopeViolation(ClaimScopeError):
    """Raised when widen() is asked to broaden a law its evidence does not support.

    Never silently clamped. The caller must see the exception.
    """

    def __init__(
        self,
        message: str,
        *,
        law_id: str | None = None,
        axis: str | None = None,
        from_tier: str | None = None,
        to_tier: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.law_id = law_id
        self.axis = axis
        self.from_tier = from_tier
        self.to_tier = to_tier
        self.reason = reason or message


class HindsightContamination(ClaimScopeError):
    """A law's available-set includes a specimen that was not sealed at as_of."""

    def __init__(
        self,
        message: str,
        *,
        law_id: str | None = None,
        specimen: str | None = None,
        as_of: str | None = None,
        sealed_at: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.law_id = law_id
        self.specimen = specimen
        self.as_of = as_of
        self.sealed_at = sealed_at
        self.reason = reason or message


class ExperimentIdentityError(ClaimScopeError):
    """An experiment is missing one of the six identity fields."""

    def __init__(self, message: str, *, missing: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.missing = tuple(missing)


class OverbroadConclusion(ClaimScopeError):
    """A conclusion claimed a universe wider than the law's available-set."""

    def __init__(self, message: str, *, law_id: str | None = None, reason: str | None = None) -> None:
        super().__init__(message)
        self.law_id = law_id
        self.reason = reason or message


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def parse_iso(value: str | None) -> datetime | None:
    if not value or value == "UNKNOWN":
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_z(value: str) -> str:
    """Normalise any parseable timestamp to UTC Z. Do not invent a time."""
    dt = parse_iso(value)
    if dt is None:
        return (value or "").strip()
    return iso_z(dt)


def unix_to_iso(ts: float) -> str:
    return iso_z(datetime.fromtimestamp(ts, timezone.utc))


def _le(a: str | None, b: str | None) -> bool:
    """True when a <= b. UNKNOWN/unparseable never beats a real timestamp."""
    da, db = parse_iso(a), parse_iso(b)
    if da is None or db is None:
        return False
    return da <= db


# ---------------------------------------------------------------------------
# Repo JSON (sparse checkout: git show when the file is not on disk)
# ---------------------------------------------------------------------------


def load_repo_json(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if path.is_file():
        return load_json(path)
    blob = git("show", f"HEAD:{rel}")
    if not blob:
        raise FileNotFoundError(rel)
    return json.loads(blob)


_JSON_CACHE: dict[str, tuple[dict[str, Any] | None, str | None]] = {}


def try_load(rel: str) -> tuple[dict[str, Any] | None, str | None]:
    if rel in _JSON_CACHE:
        return _JSON_CACHE[rel]
    try:
        hit: tuple[dict[str, Any] | None, str | None] = (load_repo_json(rel), None)
    except FileNotFoundError:
        hit = (None, f"not in working tree or git HEAD: {rel}")
    except json.JSONDecodeError as e:
        hit = (None, f"unreadable JSON at {rel}: {e}")
    _JSON_CACHE[rel] = hit
    return hit


def receipt_as_of(rel: str, doc: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """When this receipt can be placed in time. Never invent a measurement time."""
    if doc is None:
        doc, _ = try_load(rel)
    if isinstance(doc, Mapping):
        bench = doc.get("bench") if isinstance(doc.get("bench"), Mapping) else {}
        recorded = bench.get("recorded_at") if isinstance(bench, Mapping) else None
        if isinstance(recorded, str) and recorded.strip():
            return _to_z(recorded), f"{rel}#bench.recorded_at"
        prov = doc.get("measurement_provenance")
        if isinstance(prov, Mapping) and prov.get("measured_at"):
            return _to_z(str(prov["measured_at"])), f"{rel}#measurement_provenance.measured_at"
    line = git("log", "-1", "--format=%cI", "--", rel)
    if line:
        return _to_z(line), f"{rel} git commit time (landing, not a measurement clock)"
    path = REPO / rel
    if path.is_file():
        return unix_to_iso(path.stat().st_mtime), f"{rel} file mtime (no recorded_at, no git time)"
    return "UNKNOWN", f"{rel} as_of unknown; available-set will not be widened by guess"


# ---------------------------------------------------------------------------
# Live specimen timeline. Seal times come from disk, not a fixture.
# ---------------------------------------------------------------------------


def _hf_metadata_times(spec_dir: Path) -> list[float]:
    meta_dir = spec_dir / HF_DOWNLOAD
    times: list[float] = []
    try:
        if not meta_dir.is_dir():
            return times
        for path in meta_dir.iterdir():
            if not path.name.endswith(".metadata") or not path.is_file():
                continue
            try:
                lines = [ln.strip() for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
            except OSError:
                continue
            if len(lines) < 3:
                continue
            try:
                times.append(float(lines[2]))
            except ValueError:
                continue
    except OSError:
        return times
    return times


def _incomplete_markers(root: Path) -> int:
    """Top-level and HF download dir only. A deep rglob of a 360 GB tree is not a seal read."""
    n = 0
    try:
        for path in root.iterdir():
            if path.is_file() and "incomplete" in path.name.lower():
                n += 1
        download = root / HF_DOWNLOAD
        if download.is_dir():
            for path in download.iterdir():
                if "incomplete" in path.name.lower():
                    n += 1
    except OSError:
        return n
    return n


def _overlay_live_dir(row: dict[str, Any], spec_dir: Path) -> None:
    if not spec_dir.is_dir():
        row["on_disk"] = False
        return
    row["on_disk"] = True
    row["specimen_path"] = str(spec_dir)
    times = _hf_metadata_times(spec_dir)
    if times:
        row["landed_at"] = unix_to_iso(max(times))
        row["landed_at_source"] = str(spec_dir / HF_DOWNLOAD / "<file>.metadata#line3")
        row["n_hf_metadata"] = len(times)
    seal = spec_dir / LAKE_SEAL_NAME
    if seal.is_file():
        try:
            doc = json.loads(seal.read_text())
        except (OSError, json.JSONDecodeError, UnicodeError):
            doc = {}
        row["lake_seal_status"] = doc.get("final_status")
        row["lake_seal_mtime"] = unix_to_iso(seal.stat().st_mtime)
        row["lake_seal_mtime_source"] = str(seal)
    row["incomplete_markers"] = _incomplete_markers(spec_dir)


def _empty_row(specimen: str) -> dict[str, Any]:
    return {
        "specimen": specimen,
        "role": None,
        "owner": None,
        "status": "UNKNOWN",
        "sealed": False,
        "sealed_at": None,
        "sealed_at_source": None,
        "landed_at": None,
        "landed_at_source": None,
        "in_flight": False,
        "on_disk": False,
        "tree_digest": None,
        "resolved_sha": None,
        "repo": None,
        "n_files": None,
        "bytes_hashed": None,
        "curriculum_role": None,
    }


def _apply_verification_row(row: dict[str, Any], vrow: Mapping[str, Any], verified_at: str, verified_src: str) -> None:
    status = str(vrow.get("status") or "")
    whole = bool(
        status == "WHOLE_TREE_VERIFIED"
        and isinstance(vrow.get("bytes_hashed"), int)
        and vrow["bytes_hashed"] > 0
        and not vrow.get("mismatched")
        and vrow.get("verified") == vrow.get("n_files")
    )
    row["owner"] = vrow.get("owner") or row.get("owner")
    row["status"] = status
    row["n_files"] = vrow.get("n_files")
    row["bytes_hashed"] = vrow.get("bytes_hashed")
    row["verified"] = vrow.get("verified")
    if vrow.get("specimen_path"):
        row["specimen_path"] = vrow.get("specimen_path")
    if whole:
        row["sealed"] = True
        # Independent verification is when the specimen became a scientific object.
        # A later identity seal (authorized-external) may overwrite sealed_at.
        if not row.get("sealed_at"):
            row["sealed_at"] = verified_at
            row["sealed_at_source"] = verified_src
        row["verified_at"] = verified_at
        row["verified_at_source"] = verified_src


_TL_CACHE: dict[str, Any] | None = None


def load_timeline(*, lake: Path | None = None, force: bool = False) -> dict[str, Any]:
    """Read real seal times. A hardcoded ISO string is not a timeline."""
    global _TL_CACHE
    if _TL_CACHE is not None and not force and lake is None:
        return _TL_CACHE
    lake = lake if lake is not None else sv.LAKE
    rows: dict[str, dict[str, Any]] = {}

    vdoc, verr = try_load(VERIFICATION_REL)
    verified_at, verified_src = receipt_as_of(VERIFICATION_REL, vdoc)
    if isinstance(vdoc, Mapping):
        for vrow in vdoc.get("results") or []:
            if not isinstance(vrow, Mapping):
                continue
            name = str(vrow.get("specimen") or "")
            if not name:
                continue
            row = rows.setdefault(name, _empty_row(name))
            _apply_verification_row(row, vrow, verified_at, verified_src)

    edoc, _ = try_load(EXTERNAL_SEAL_REL)
    if isinstance(edoc, Mapping) and edoc.get("status") == "SEALED":
        name = str(edoc.get("specimen") or PARENT_SPECIMEN)
        row = rows.setdefault(name, _empty_row(name))
        identity_at, identity_src = receipt_as_of(EXTERNAL_SEAL_REL, edoc)
        row["sealed"] = True
        row["status"] = "SEALED"
        row["owner"] = edoc.get("specimen_owner") or "local_directory"
        row["tree_digest"] = edoc.get("tree_digest")
        row["n_files"] = edoc.get("n_files")
        row["bytes_hashed"] = (edoc.get("verification") or {}).get("bytes_hashed") or edoc.get("total_bytes")
        row["specimen_path"] = edoc.get("specimen_path") or row.get("specimen_path")
        row["authorized_external"] = True
        # The identity seal is the scientific seal for the authorized-external parent.
        row["sealed_at"] = identity_at
        row["sealed_at_source"] = identity_src
        v = edoc.get("verification") if isinstance(edoc.get("verification"), Mapping) else {}
        if v.get("recorded_at"):
            row["verified_at"] = _to_z(str(v["recorded_at"]))
            row["verified_at_source"] = f"{EXTERNAL_SEAL_REL}#verification.recorded_at"

    cdoc, _ = try_load(CURRICULUM_REL)
    if isinstance(cdoc, Mapping):
        for role in cdoc.get("roles") or []:
            if not isinstance(role, Mapping) or not role.get("ready"):
                continue
            vs = role.get("verified_specimen") if isinstance(role.get("verified_specimen"), Mapping) else {}
            slug = ""
            path = str(vs.get("specimen_path") or "")
            if path:
                slug = path.rstrip("/").split("/")[-1]
            if vs.get("specimen"):
                slug = str(vs.get("specimen"))
            if not slug:
                repo = str(vs.get("repo") or role.get("repo") or "")
                rev = str(vs.get("resolved_sha") or vs.get("revision") or "")[:12]
                if repo and rev:
                    slug = repo.replace("/", "--") + "@" + rev
            if not slug:
                continue
            row = rows.setdefault(slug, _empty_row(slug))
            row["curriculum_role"] = role.get("role")
            row["repo"] = vs.get("repo") or role.get("repo") or row.get("repo")
            row["resolved_sha"] = vs.get("resolved_sha") or vs.get("revision") or row.get("resolved_sha")
            if vs.get("whole_tree_verified") and not row.get("sealed"):
                row["sealed"] = True
                row["status"] = row.get("status") or "WHOLE_TREE_VERIFIED"
            if vs.get("identity_kind") == "authorized_external_tree_digest" and vs.get("tree_digest"):
                row["tree_digest"] = vs.get("tree_digest")

    mounted = bool(lake.is_dir())
    if mounted:
        specimens_dir = lake / "specimens"
        if specimens_dir.is_dir():
            for name, row in list(rows.items()):
                live = specimens_dir / name
                extra = sv.EXTRA_SPECIMENS.get(name)
                if live.is_dir():
                    _overlay_live_dir(row, live)
                elif extra is not None and extra.is_dir():
                    _overlay_live_dir(row, extra)
        for name, extra in sv.EXTRA_SPECIMENS.items():
            if extra.is_dir() and name in rows:
                _overlay_live_dir(rows[name], extra)

        in_flight = []
        stale_partials: list[dict[str, Any]] = []
        partial = lake / "partial"
        if partial.is_dir():
            try:
                children = sorted(p for p in partial.iterdir() if p.is_dir())
            except OSError:
                children = []
            for path in children:
                n_inc = _incomplete_markers(path)
                times = _hf_metadata_times(path)
                landed = unix_to_iso(max(times)) if times else unix_to_iso(path.stat().st_mtime)
                landed_src = (
                    str(path / HF_DOWNLOAD / "<file>.metadata#line3")
                    if times
                    else f"{path} dir mtime"
                )
                if rows.get(path.name, {}).get("sealed"):
                    continue
                dt = parse_iso(landed)
                age_s = (
                    (datetime.now(timezone.utc) - dt).total_seconds() if dt is not None else None
                )
                recent = age_s is not None and 0 <= age_s <= IN_FLIGHT_MAX_AGE_S
                prow = {
                    "specimen": path.name,
                    "path": str(path),
                    "incomplete_markers": n_inc,
                    "n_hf_metadata": len(times),
                    "landed_at": landed,
                    "landed_at_source": landed_src,
                    "age_s": age_s,
                    "sealed": False,
                    "why": (
                        "ModelLake partial/; not whole-tree sealed. "
                        "Not in the scientific universe."
                    ),
                }
                if recent:
                    prow["in_flight"] = True
                    in_flight.append(prow)
                else:
                    prow["in_flight"] = False
                    prow["stale_partial"] = True
                    stale_partials.append(prow)
    else:
        in_flight = []
        stale_partials = []

    sealed = [r for r in rows.values() if r.get("sealed") and r.get("sealed_at")]
    sealed.sort(key=lambda r: (r.get("sealed_at") or "", r["specimen"]))
    flying_sorted = sorted(in_flight, key=lambda r: r.get("landed_at") or "", reverse=True)

    result = {
        "lake": str(lake),
        "lake_mounted": mounted,
        "n_known": len(rows),
        "n_sealed": len(sealed),
        "n_in_flight": len(flying_sorted),
        "specimens": {k: rows[k] for k in sorted(rows)},
        "sealed": sealed,
        "in_flight": flying_sorted,
        "stale_partials": stale_partials,
        "verification_load_error": verr,
        "rule": (
            "sealed_at is read from EXTERNAL_SPECIMEN_SEAL / SPECIMEN_VERIFICATION "
            "/ live HuggingFace .metadata timestamps. A fixture list of times is "
            "not a timeline. in_flight is ModelLake partial/ and is not available."
        ),
    }
    if lake == sv.LAKE:
        _TL_CACHE = result
    return result


def available_at(at: str, timeline: Mapping[str, Any] | None = None) -> list[str]:
    """Specimens whose scientific seal is at or before `at`. Downloads do not count."""
    tl = timeline if timeline is not None else load_timeline()
    names: list[str] = []
    for row in (tl.get("specimens") or {}).values():
        if not isinstance(row, Mapping):
            continue
        if not row.get("sealed"):
            continue
        sealed_at = row.get("sealed_at")
        if not isinstance(sealed_at, str) or not sealed_at or sealed_at == "UNKNOWN":
            continue
        if _le(sealed_at, at):
            names.append(str(row["specimen"]))
    names.sort()
    return names


def evidence_universe_clause(specimens: Sequence[str]) -> str:
    if not specimens:
        return "within currently sealed specimens (none)"
    return "within currently sealed specimens " + "/".join(specimens)


def _looks_universal(statement: str) -> bool:
    blob = " ".join((statement or "").lower().split())
    return any(m in blob for m in UNIVERSAL_CLAIM_MARKERS)


# ---------------------------------------------------------------------------
# Claim record
# ---------------------------------------------------------------------------


def narrow_scope() -> dict[str, str]:
    return dict(NARROW_TIERS)


@dataclass(frozen=True)
class Claim:
    law_id: str
    statement: str
    tested_specimens: tuple[str, ...]
    available_specimens: tuple[str, ...]
    as_of: str
    as_of_source: str
    scope: dict[str, str]
    scope_kind: str
    organ: str
    machine: str
    parent: str
    evidence_refs: tuple[str, ...]
    experiment_identity: dict[str, Any]
    statement_before_narrowing: str | None = None
    narrowed: bool = False
    narrowing: str | None = None
    replicating_specimens: tuple[str, ...] = ()
    failed_transfers: tuple[dict[str, Any], ...] = ()
    layers: tuple[int, ...] | None = None
    teacher_corpus_rows: int | None = None
    original_scope: dict[str, str] | None = None
    notes: tuple[str, ...] = ()

    def evidence_universe(self) -> str:
        return evidence_universe_clause(self.available_specimens)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "law_id": self.law_id,
            "statement": self.statement,
            "statement_before_narrowing": self.statement_before_narrowing,
            "narrowed": self.narrowed,
            "narrowing": self.narrowing,
            "tested_specimens": list(self.tested_specimens),
            "available_specimens": list(self.available_specimens),
            "as_of": self.as_of,
            "as_of_source": self.as_of_source,
            "scope": dict(self.scope),
            "original_scope": dict(self.original_scope or self.scope),
            "scope_kind": self.scope_kind,
            "organ": self.organ,
            "machine": self.machine,
            "parent": self.parent,
            "evidence_refs": list(self.evidence_refs),
            "evidence_universe": self.evidence_universe(),
            "experiment_identity": dict(self.experiment_identity),
            "replicating_specimens": list(self.replicating_specimens),
            "failed_transfers": [dict(x) for x in self.failed_transfers],
            "layers": list(self.layers) if self.layers is not None else None,
            "teacher_corpus_rows": self.teacher_corpus_rows,
            "notes": list(self.notes),
        }
        return d


def validate_claim(claim: Claim, *, timeline: Mapping[str, Any] | None = None) -> Claim:
    if claim.scope_kind not in SCOPE_KINDS:
        raise ScopeViolation(
            f"{claim.law_id}: scope_kind {claim.scope_kind!r} is not one of {SCOPE_KINDS}",
            law_id=claim.law_id,
            reason="unknown_scope_kind",
        )
    for axis in AXES:
        tier = claim.scope.get(axis)
        legal = {NARROW_TIERS[axis], REPLICATED_TIERS[axis]}
        if tier not in legal:
            raise ScopeViolation(
                f"{claim.law_id}: scope[{axis}]={tier!r} is not in {sorted(legal)}",
                law_id=claim.law_id,
                axis=axis,
                from_tier=tier,
                reason="unknown_tier",
            )
    if not claim.tested_specimens:
        raise ScopeViolation(
            f"{claim.law_id}: a law with no tested specimen is not a law",
            law_id=claim.law_id,
            reason="no_tested_specimen",
        )
    missing = [f for f in EXPERIMENT_IDENTITY_FIELDS if not claim.experiment_identity.get(f)]
    if missing:
        raise ExperimentIdentityError(
            f"{claim.law_id}: experiment identity missing {missing}",
            missing=missing,
        )
    refuse_hindsight(claim, timeline=timeline)
    return claim


def refuse_hindsight(claim: Claim, *, timeline: Mapping[str, Any] | None = None) -> None:
    """A specimen sealed after as_of cannot sit in the law's available-set."""
    if claim.as_of in {"", "UNKNOWN", None}:
        # Cannot prove contamination or innocence. Keep available-set == tested-set.
        extra = [s for s in claim.available_specimens if s not in claim.tested_specimens]
        if extra:
            raise HindsightContamination(
                f"{claim.law_id}: as_of is UNKNOWN so available-set cannot include "
                f"untested specimens {extra}; that would be a guess, not a timeline",
                law_id=claim.law_id,
                reason="unknown_as_of_cannot_claim_later_specimens",
            )
        return
    actual = set(available_at(claim.as_of, timeline))
    # An empty timeline (unit tests that pass {}) cannot prove contamination.
    specimens = (timeline or {}).get("specimens") if isinstance(timeline, Mapping) else None
    if not actual and not specimens:
        return
    tested = set(claim.tested_specimens)
    for name in claim.available_specimens:
        # The parent an experiment actually ran on WAS available as a working
        # artifact. Its identity seal (tree digest) may have been minted later.
        # Hindsight is claiming an UNTESTED specimen that was not yet sealed.
        if name in tested:
            continue
        if actual and name not in actual:
            row = (specimens or {}).get(name) or {}
            raise HindsightContamination(
                f"{claim.law_id}: specimen {name!r} is in available-set at as_of "
                f"{claim.as_of} but was not sealed until {row.get('sealed_at')!r}. "
                f"That is hindsight contamination.",
                law_id=claim.law_id,
                specimen=name,
                as_of=claim.as_of,
                sealed_at=row.get("sealed_at"),
                reason="specimen_not_sealed_at_as_of",
            )


def conclude(claim: Claim, statement: str) -> dict[str, Any]:
    """Stamp the evidence universe. A universal claim on MODEL_LOCAL RAISES."""
    if _looks_universal(statement) and claim.scope.get("model") == "MODEL_LOCAL":
        raise OverbroadConclusion(
            f"{claim.law_id}: refusing {statement!r} while scope.model is MODEL_LOCAL. "
            f"Say {claim.evidence_universe()!r}, never 'all Hawking models'.",
            law_id=claim.law_id,
            reason="universal_claim_on_model_local",
        )
    return {
        "law_id": claim.law_id,
        "statement": statement,
        "evidence_universe": claim.evidence_universe(),
        "scope": dict(claim.scope),
        "scope_kind": claim.scope_kind,
        "tested_specimens": list(claim.tested_specimens),
        "available_specimens": list(claim.available_specimens),
        "as_of": claim.as_of,
    }


# ---------------------------------------------------------------------------
# Widen / fail-transfer. The refusal is the deliverable.
# ---------------------------------------------------------------------------


def _normalize_specimen_name(name: str | None) -> str:
    return " ".join((name or "").strip().lower().split())


def _is_named_specimen(name: str | None) -> bool:
    if not isinstance(name, str):
        return False
    trimmed = name.strip()
    if not trimmed:
        return False
    return _normalize_specimen_name(trimmed) not in NOT_A_SPECIMEN


def _specimen_is_sealed(name: str, timeline: Mapping[str, Any], at: str | None) -> bool:
    row = (timeline.get("specimens") or {}).get(name)
    if isinstance(row, Mapping) and row.get("sealed"):
        if at in {None, "", "UNKNOWN"}:
            return True
        sealed_at = row.get("sealed_at")
        return isinstance(sealed_at, str) and _le(sealed_at, at)
    # Allow a test timeline to mark sealed by membership in sealed[]
    for row in timeline.get("sealed") or []:
        if isinstance(row, Mapping) and row.get("specimen") == name and row.get("sealed"):
            return True
    return False


def widen(
    claim: Claim,
    axis: str,
    *,
    replicating_specimen: str | None,
    replication_receipt: str | None,
    replicating_organ: str | None = None,
    replicating_machine: str | None = None,
    at: str | None = None,
    timeline: Mapping[str, Any] | None = None,
) -> Claim:
    """Widen one axis by replication. Anything else RAISES.

    A named replicating specimen is mandatory. A receipt is mandatory.
    The specimen must be sealed at `at`. FAILED_TRANSFER does not go through
    this function — use record_failed_transfer, which does not widen.
    """
    validate_claim(claim, timeline=timeline)
    if axis not in AXES:
        raise ScopeViolation(
            f"{claim.law_id}: axis {axis!r} is not one of {AXES}",
            law_id=claim.law_id,
            axis=axis,
            reason="unknown_axis",
        )
    if not _is_named_specimen(replicating_specimen):
        raise ScopeViolation(
            f"{claim.law_id}: widening {axis} without a named replicating specimen "
            f"(got {replicating_specimen!r}). Widening without a replicating "
            f"specimen is impossible, not discouraged.",
            law_id=claim.law_id,
            axis=axis,
            from_tier=claim.scope[axis],
            to_tier=REPLICATED_TIERS[axis],
            reason="no_named_replicating_specimen",
        )
    assert replicating_specimen is not None
    specimen = replicating_specimen.strip()
    if not isinstance(replication_receipt, str) or not replication_receipt.strip():
        raise ScopeViolation(
            f"{claim.law_id}: replication of {specimen!r} has no receipt. "
            f"A name without evidence is not a replication.",
            law_id=claim.law_id,
            axis=axis,
            from_tier=claim.scope[axis],
            to_tier=REPLICATED_TIERS[axis],
            reason="replication_has_no_receipt",
        )
    if claim.scope[axis] == REPLICATED_TIERS[axis]:
        # Already wide on this axis — still require a new specimen to extend
        # the replicating set, but the tier does not change. A second
        # replication is recorded; a missing specimen still refuses.
        pass
    elif claim.scope[axis] != NARROW_TIERS[axis]:
        raise ScopeViolation(
            f"{claim.law_id}: scope[{axis}]={claim.scope[axis]!r} cannot widen",
            law_id=claim.law_id,
            axis=axis,
            from_tier=claim.scope[axis],
            reason="unknown_tier",
        )

    if axis == "model" and specimen in claim.tested_specimens:
        raise ScopeViolation(
            f"{claim.law_id}: {specimen!r} is already in the original tested set "
            f"{list(claim.tested_specimens)}. That is not a replicating specimen.",
            law_id=claim.law_id,
            axis=axis,
            from_tier=claim.scope[axis],
            to_tier=REPLICATED_TIERS[axis],
            reason="replicating_specimen_already_in_original_scope",
        )
    if axis == "organ":
        if not replicating_organ or replicating_organ == claim.organ:
            raise ScopeViolation(
                f"{claim.law_id}: ORGAN_LOCAL widens only when a named specimen "
                f"replicates the law on a DIFFERENT organ (got organ={replicating_organ!r}, "
                f"claim.organ={claim.organ!r})",
                law_id=claim.law_id,
                axis=axis,
                from_tier=claim.scope[axis],
                to_tier=REPLICATED_TIERS[axis],
                reason="replicating_organ_not_distinct",
            )
    if axis == "machine":
        if not replicating_machine or replicating_machine == claim.machine:
            raise ScopeViolation(
                f"{claim.law_id}: MACHINE_LOCAL widens only when a named specimen "
                f"replicates the law on a DIFFERENT machine",
                law_id=claim.law_id,
                axis=axis,
                from_tier=claim.scope[axis],
                to_tier=REPLICATED_TIERS[axis],
                reason="replicating_machine_not_distinct",
            )

    tl = timeline if timeline is not None else load_timeline()
    when = at or claim.as_of
    if not _specimen_is_sealed(specimen, tl, when):
        raise ScopeViolation(
            f"{claim.law_id}: replicating specimen {specimen!r} is not sealed at {when}. "
            f"A download in flight is not a replicating specimen.",
            law_id=claim.law_id,
            axis=axis,
            from_tier=claim.scope[axis],
            to_tier=REPLICATED_TIERS[axis],
            reason="replicating_specimen_not_sealed",
        )

    new_scope = dict(claim.scope)
    new_scope[axis] = REPLICATED_TIERS[axis]
    new_tested = tuple(dict.fromkeys([*claim.tested_specimens, specimen]))
    new_available = tuple(available_at(when, tl)) or tuple(
        dict.fromkeys([*claim.available_specimens, specimen])
    )
    new_replicating = tuple(dict.fromkeys([*claim.replicating_specimens, specimen]))
    note = (
        f"REPLICATION on axis={axis} specimen={specimen} receipt={replication_receipt.strip()} "
        f"at={when}"
    )
    widened = replace(
        claim,
        scope=new_scope,
        scope_kind="REPLICATION",
        tested_specimens=new_tested,
        available_specimens=new_available,
        replicating_specimens=new_replicating,
        original_scope=dict(claim.original_scope or claim.scope),
        as_of=when if when else claim.as_of,
        notes=tuple([*claim.notes, note]),
        experiment_identity=dict(claim.experiment_identity),
    )
    return validate_claim(widened, timeline=tl)


def record_failed_transfer(
    claim: Claim,
    *,
    specimen: str,
    receipt: str,
    why: str,
    at: str | None = None,
    timeline: Mapping[str, Any] | None = None,
) -> Claim:
    """A failed transfer is recorded. Scope tiers do not move."""
    validate_claim(claim, timeline=timeline)
    if not _is_named_specimen(specimen):
        raise ScopeViolation(
            f"{claim.law_id}: failed transfer requires a named specimen, got {specimen!r}",
            law_id=claim.law_id,
            reason="no_named_replicating_specimen",
        )
    if not isinstance(receipt, str) or not receipt.strip():
        raise ScopeViolation(
            f"{claim.law_id}: failed transfer of {specimen!r} has no receipt",
            law_id=claim.law_id,
            reason="replication_has_no_receipt",
        )
    row = {
        "specimen": specimen.strip(),
        "receipt": receipt.strip(),
        "why": why,
        "at": at or claim.as_of,
        "scope_unchanged": dict(claim.scope),
    }
    out = replace(
        claim,
        failed_transfers=tuple([*claim.failed_transfers, row]),
        notes=tuple([*claim.notes, f"FAILED_TRANSFER specimen={specimen.strip()} receipt={receipt.strip()}"]),
        experiment_identity=dict(claim.experiment_identity),
    )
    # Explicit: tiers did not move.
    if out.scope != claim.scope:
        raise ScopeViolation(
            f"{claim.law_id}: failed transfer mutated scope {claim.scope} -> {out.scope}",
            law_id=claim.law_id,
            reason="failed_transfer_must_not_widen",
        )
    return validate_claim(out, timeline=timeline)


# ---------------------------------------------------------------------------
# Experiment identity — six fields, or it is not an experiment
# ---------------------------------------------------------------------------


def _content_hash(parts: Iterable[str]) -> str:
    blob = "\n".join(parts).encode()
    return hashlib.sha256(blob).hexdigest()


def laws_scars_version() -> dict[str, Any]:
    """Pin the laws/scars corpus this experiment was judged against."""
    seals: dict[str, str | None] = {}
    for rel in (CAMPAIGN_SCARS_REL, LAW_STORE_REL, NEGATIVE_INDEX_REL, STRUCT_REL):
        doc, err = try_load(rel)
        if doc is None:
            seals[rel] = None
            seals[rel + "#error"] = err  # type: ignore[assignment]
        else:
            seals[rel] = doc.get("seal_sha256") or _content_hash(
                [json.dumps(doc, sort_keys=True, separators=(",", ":"))]
            )
    digest = _content_hash(f"{k}={v}" for k, v in sorted(seals.items()) if not str(k).endswith("#error"))
    return {
        "digest": digest,
        "pins": {k: v for k, v in seals.items() if not str(k).endswith("#error")},
        "missing": [k[:-6] for k, v in seals.items() if str(k).endswith("#error")],
        "rule": (
            "an experiment is judged against a named laws/scars version; "
            "a later scar does not silently rewrite an earlier verdict"
        ),
    }


def default_specimen_seal() -> dict[str, Any]:
    doc, err = try_load(EXTERNAL_SEAL_REL)
    if not isinstance(doc, Mapping) or doc.get("status") != "SEALED":
        return {
            "specimen": PARENT_SPECIMEN,
            "tree_digest": None,
            "status": "UNAVAILABLE",
            "error": err or "EXTERNAL_SPECIMEN_SEAL is not SEALED",
            "source": EXTERNAL_SEAL_REL,
        }
    return {
        "specimen": doc.get("specimen") or PARENT_SPECIMEN,
        "tree_digest": doc.get("tree_digest"),
        "n_files": doc.get("n_files"),
        "status": doc.get("status"),
        "source": EXTERNAL_SEAL_REL,
        "kind": doc.get("kind"),
    }


def default_model_revision() -> dict[str, Any]:
    return {
        "model_id": PARENT_MODEL_ID,
        "resident_identity": PARENT_RESIDENT,
        "weight_specimen": PARENT_SPECIMEN,
        "source": "hcli/hawking-native.sealed-3.14.json (cited; packed parent of the campaign)",
    }


def default_resident_identity() -> dict[str, Any]:
    doc, _ = try_load(RESIDENT_IDENTITY_REL)
    pins = {}
    if isinstance(doc, Mapping):
        binding = doc.get("binding") if isinstance(doc.get("binding"), Mapping) else {}
        pins = binding.get("pins") if isinstance(binding.get("pins"), Mapping) else {}
    return {
        "resident_identity": PARENT_RESIDENT,
        "sealed_model_id": (pins.get("sealed_model_id") if isinstance(pins, Mapping) else None)
        or PARENT_MODEL_ID,
        "nx_id": pins.get("nx_id") if isinstance(pins, Mapping) else None,
        "source": RESIDENT_IDENTITY_REL,
        "bound": bool(isinstance(doc, Mapping) and (doc.get("binding") or {}).get("bound")),
    }


def default_code_and_build() -> dict[str, Any]:
    head = git("rev-parse", "HEAD") or "UNKNOWN"
    short = git("rev-parse", "--short", "HEAD") or "UNKNOWN"
    binary = None
    ident, _ = try_load(RESIDENT_IDENTITY_REL)
    if isinstance(ident, Mapping):
        pins = ((ident.get("binding") or {}).get("pins") or {})
        eh = pins.get("executable_hash") if isinstance(pins, Mapping) else None
        if isinstance(eh, Mapping):
            binary = eh.get("seal_declared_sha256_16") or (eh.get("by_role") or {}).get("binary")
    dirty, _ = try_load(DIRTY_SOURCE_REL)
    dirty_label = None
    if isinstance(dirty, Mapping):
        dirty_label = dirty.get("measurement_label") or (
            "DIRTY_SOURCE_DIAGNOSTIC" if dirty.get("promoted") is False else None
        )
    return {
        "git_head": head,
        "git_head_short": short,
        "binary_sha256_pin": binary,
        "dirty_source_label": dirty_label,
        "source": "git rev-parse HEAD + RESIDENT_IDENTITY executable_hash pin",
    }


def default_machine_genome() -> dict[str, Any]:
    """Pin the genome receipt. Do not remeasure bandwidth. Do not invent a roof."""
    doc, err = try_load(MACHINE_GENOME_REL)
    if not isinstance(doc, Mapping):
        return {
            "receipt": MACHINE_GENOME_REL,
            "readable": False,
            "error": err,
            "remeasured": False,
            "note": "genome cited as a prior; this sidecar does not run measure_bandwidth",
        }
    return {
        "receipt": MACHINE_GENOME_REL,
        "schema": doc.get("schema"),
        "arch": doc.get("arch"),
        "soc": doc.get("soc"),
        "cpu_cores": doc.get("cpu_cores"),
        "gpu_cores": doc.get("gpu_cores"),
        "readable": True,
        "remeasured": False,
        "note": (
            "Identity pin of the machine the campaign ran on. Bandwidth figures "
            "in that receipt are NOT copied here and are NOT a roof."
        ),
    }


def bind_experiment(
    *,
    specimen_seal: Any = None,
    model_revision: Any = None,
    resident_identity: Any = None,
    code_and_build_identity: Any = None,
    machine_genome: Any = None,
    laws_scars_version: Any = None,
    fill_defaults: bool = False,
) -> dict[str, Any]:
    """Bind the six identity fields. Missing fields RAISE.

    fill_defaults=True is for build()/retro-scope, which recover pins from disk.
    Callers recording a new experiment should pass every field. A silent
    default is how an experiment inherits the wrong parent.
    """
    if fill_defaults:
        specimen_seal = specimen_seal if specimen_seal is not None else default_specimen_seal()
        model_revision = model_revision if model_revision is not None else default_model_revision()
        resident_identity = resident_identity if resident_identity is not None else default_resident_identity()
        code_and_build_identity = (
            code_and_build_identity if code_and_build_identity is not None else default_code_and_build()
        )
        machine_genome = machine_genome if machine_genome is not None else default_machine_genome()
        laws_scars_version = laws_scars_version if laws_scars_version is not None else laws_scars_version_pin()
    identity = {
        "specimen_seal": specimen_seal,
        "model_revision": model_revision,
        "resident_identity": resident_identity,
        "code_and_build_identity": code_and_build_identity,
        "machine_genome": machine_genome,
        "laws_scars_version": laws_scars_version,
    }
    missing = [k for k in EXPERIMENT_IDENTITY_FIELDS if identity.get(k) in (None, "", [], {})]
    if missing:
        raise ExperimentIdentityError(
            f"experiment identity missing {missing}; a dynamic curriculum still "
            f"has to produce a reproducible experiment",
            missing=missing,
        )
    # Nested pins must themselves be identities, not empty shells.
    seal = identity["specimen_seal"]
    if isinstance(seal, Mapping) and not (seal.get("tree_digest") or seal.get("specimen")):
        raise ExperimentIdentityError(
            "specimen_seal has no tree_digest and no specimen name",
            missing=("specimen_seal.tree_digest",),
        )
    return identity


def laws_scars_version_pin() -> dict[str, Any]:
    return laws_scars_version()


# ---------------------------------------------------------------------------
# Retro-scope the campaign's existing laws. Narrowing is a correction.
# ---------------------------------------------------------------------------


def retro_campaign_laws(
    *,
    timeline: Mapping[str, Any] | None = None,
    identity: Mapping[str, Any] | None = None,
) -> tuple[list[Claim], list[dict[str, Any]]]:
    """The five campaign laws, scoped to the parent they were actually run on.

    None of them silently inherit Falcon, Mistral, Flash, or Qwen3-0.6B just
    because those roles are sealed now. If a prior statement was broader than
    the evidence, it is narrowed here and the narrowing is recorded.
    """
    tl = timeline if timeline is not None else load_timeline()
    ident = dict(identity) if identity is not None else bind_experiment(fill_defaults=True)

    teacher, _ = try_load(TEACHER_REL)
    n_rows = TEACHER_ROWS
    layers = TEACHER_LAYERS
    if isinstance(teacher, Mapping):
        cap = teacher.get("capture") if isinstance(teacher.get("capture"), Mapping) else {}
        if isinstance(cap.get("n_rows"), int):
            n_rows = cap["n_rows"]
        layer_rows = cap.get("layers") if isinstance(cap.get("layers"), list) else []
        got = tuple(int(x["layer"]) for x in layer_rows if isinstance(x, Mapping) and isinstance(x.get("layer"), int))
        if got:
            layers = got

    parent = PARENT_SPECIMEN
    machine = PARENT_MACHINE
    claims: list[Claim] = []
    narrowings: list[dict[str, Any]] = []

    def _make(
        *,
        law_id: str,
        overbroad: str,
        honest: str,
        narrowing: str,
        rel: str,
        doc: Mapping[str, Any] | None,
        organ: str,
        layers_for: tuple[int, ...] | None,
        rows: int | None,
        extra_refs: tuple[str, ...] = (),
        extra_as_of: tuple[str, str] | None = None,
    ) -> Claim:
        as_of, as_of_source = extra_as_of if extra_as_of is not None else receipt_as_of(rel, doc)
        available = tuple(available_at(as_of, tl))
        # The packed parent was the working artifact. Include it even when the
        # authorized-external identity seal postdates the measurement clock.
        if parent and parent not in available:
            available = (parent, *available)
        if overbroad != honest:
            narrowings.append(
                {
                    "law_id": law_id,
                    "was": overbroad,
                    "now": honest,
                    "why": narrowing,
                    "correction_not_downgrade": True,
                }
            )
        claim = Claim(
            law_id=law_id,
            statement=honest,
            statement_before_narrowing=overbroad if overbroad != honest else None,
            narrowed=overbroad != honest,
            narrowing=narrowing if overbroad != honest else None,
            tested_specimens=(parent,),
            available_specimens=available,
            as_of=as_of,
            as_of_source=as_of_source,
            scope=narrow_scope(),
            original_scope=narrow_scope(),
            scope_kind="ORIGINAL",
            organ=organ,
            machine=machine,
            parent=parent,
            evidence_refs=(rel, *extra_refs),
            experiment_identity=dict(ident),
            layers=layers_for,
            teacher_corpus_rows=rows,
            notes=(
                "retro-scoped: produced against one parent (qwen3.8-27b sealed-3.14); "
                "later curriculum roles are NOT silently in tested_specimens",
            ),
        )
        return validate_claim(claim, timeline=tl)

    alu, _ = try_load(ALU_REL)
    claims.append(
        _make(
            law_id="LAW-MLP-ARITHMETIC-SENSITIVITY",
            overbroad=(
                "MLP is ALU-bound: stripping decode arithmetic is a machine-general "
                "way to reach the roof."
            ),
            honest=(
                "On qwen3.8-27b sealed-3.14, this machine, one representative MLP "
                "layer of affine2 geo_tpr64: ARM A (same bytes, arithmetic stripped) "
                "is cited at 497.4 effective GB/s against production 329.6 "
                f"(receipt {ALU_REL}#mlp.arm_a_stripped.effective_gb_s). Combined "
                "verdict MIXED — ARM B tracked bytes, so the pre-registered rule "
                "did not promote ALU_BOUND. Arithmetic-sensitivity of THIS organ "
                "on THIS parent on THIS machine. Not 'MLP is ALU-bound'."
            ),
            narrowing=(
                "The ALU-roofline receipt's own verdict is MIXED, not ALU_BOUND. "
                "Quoting ARM A's 1.51x jump as a generic MLP-is-ALU-bound law "
                "silently widens ORGAN, MODEL, and MACHINE. Narrowed to the "
                "measured organ/parent/machine. Citation, not a new measurement."
            ),
            rel=ALU_REL,
            doc=alu,
            organ="mlp",
            layers_for=(0,),
            rows=None,
        )
    )

    econ, _ = try_load(ECON_CAL_REL)
    claims.append(
        _make(
            law_id="LAW-BROADCAST-AUX-NON-CRITICALITY",
            overbroad=(
                "Broadcast aux is never on the critical path; aux bytes are free "
                "and must not be billed at the organ average."
            ),
            honest=(
                "On qwen3.8-27b sealed-3.14 MLP layer 3, this machine: a 50% drop "
                "of broadcast aux was within_noise (cited paired_dt_ns_at_50pct=-126, "
                f"noise floor 2666 ns, {ECON_CAL_REL}#stream_classes.broadcast_aux). "
                "catalog_billing measured_or_zero for THAT stream on THAT layer. "
                "Not a law that aux is free on other organs, parents, or machines."
            ),
            narrowing=(
                "EXECUTABLE_ECONOMICS applies this calibration as a stream-class "
                "rate. The measurement is one layer of one parent on one machine. "
                "Billing-by-stream-class is the method; the zero-rate is MODEL_LOCAL "
                "/ ORGAN_LOCAL / MACHINE_LOCAL until a named specimen replicates it."
            ),
            rel=ECON_CAL_REL,
            doc=econ,
            organ="mlp",
            layers_for=(3,),
            rows=None,
            extra_refs=("receipts/future/EXECUTABLE_ECONOMICS.json",),
        )
    )

    struct, _ = try_load(STRUCT_REL)
    claims.append(
        _make(
            law_id="LAW-MLP-FUNCTION-REPLACEMENT-CLOSED",
            overbroad="MLP function replacement is closed.",
            honest=(
                "Function replacement of F(x)=down(silu(gate(x))*up(x)) is CLOSED "
                f"for qwen3.8-27b sealed-3.14 MLP on layers {list(layers)} with a "
                f"teacher corpus of {n_rows} prompt-split rows "
                f"(scar MLP_FUNCTION_REPLACEMENT_CLOSED in {STRUCT_REL}). The "
                "receipt already labels the scar MODEL_SPECIFIC. A different parent "
                "reopens. Not 'MLP function replacement is closed' for Hawking."
            ),
            narrowing=(
                "The short form 'is_function_replacement_closed: YES' drops the "
                "parent, the organ, the four layers, and the 45,076-row corpus. "
                "That is the over-broad statement. Narrowed back to the evidence. "
                "The scar remains; the universe is named."
            ),
            rel=STRUCT_REL,
            doc=struct,
            organ="mlp",
            layers_for=layers,
            rows=n_rows,
            extra_refs=(TEACHER_REL,),
        )
    )

    fold, _ = try_load(FOLD_REL)
    claims.append(
        _make(
            law_id="LAW-PROBE-UNDERSELLS-TOKEN",
            overbroad=(
                "Isolated-organ probes undersell complete-token savings; a probe "
                "number is never the token number."
            ),
            honest=(
                "On qwen3.8-27b sealed-3.14, this machine, fold_addqx: the one-layer "
                "probe projected 1.745 ms and the complete-token A/B saved 3.9833 ms "
                f"and was NOT bit-identical ({FOLD_REL}#finding). Probe ≠ token for "
                "THIS lever on THIS parent on THIS machine. The class is a campaign "
                "scar in the making; the quantity is not a generic probe law."
            ),
            narrowing=(
                "Generalising one lever's probe/token disagreement into 'probes "
                "undersell tokens' would let a later specimen inherit a magnitude "
                "it never earned. Narrowed to fold_addqx / sealed-3.14 / this machine."
            ),
            rel=FOLD_REL,
            doc=fold,
            organ="mlp",
            layers_for=None,
            rows=None,
        )
    )

    roof, _ = try_load(ROOF_REL)
    # The 497.4 figure was measured in ALU_ROOFLINE (ARM A) and independently
    # observed on the LM head. roof_anchor is the registry that named it. Prefer
    # ALU's as_of if roof_anchor is later — the LAW is the measurement, not the
    # registry. Using ALU as_of keeps later-sealed specimens out of a law that
    # did not have them.
    alu_as_of = receipt_as_of(ALU_REL, alu)
    claims.append(
        _make(
            law_id="LAW-497P4-ANCHOR",
            overbroad=(
                "The roof is 497.4 GB/s. Every organ at the LM head's demonstrated "
                "497.4 is the honest machine ceiling."
            ),
            honest=(
                "497.4 GB/s is the ARM-A-stripped MLP affine2 rate on qwen3.8-27b "
                f"sealed-3.14, one representative layer, this machine ({ALU_REL}"
                "#mlp.arm_a_stripped.effective_gb_s). The production LM head on the "
                "same parent/machine is cited at the same figure "
                f"({ROOF_REL}#lm_head_production_497p4). It is not a DRAM roof, not "
                "a published-peak substitute, and not a law about other models or "
                "other organs. Causal-budget rungs that put 'every organ at 497.4' "
                "are hypothetical ladders, not replications."
            ),
            narrowing=(
                "roof_anchor correctly named 497.4 and refused 595.9/703.5 as the "
                "production roof. 'Every organ at 497.4' still over-reaches ORGAN. "
                "Treating 497.4 as the machine's DRAM property over-reaches the "
                "f32-triad genome and this parent's MLP/LM-head measurements. "
                "Narrowed to MODEL_LOCAL / ORGAN_LOCAL (mlp, with lm_head as a "
                "same-parent coincidence not a replication) / MACHINE_LOCAL."
            ),
            rel=ALU_REL,
            doc=alu,
            organ="mlp",
            layers_for=(0,),
            rows=None,
            extra_refs=(ROOF_REL,),
            extra_as_of=alu_as_of,
        )
    )
    # Silence unused if try_load failed; roof is cited either way.
    _ = roof

    return claims, narrowings


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build() -> dict[str, Any]:
    timeline = load_timeline()
    identity = bind_experiment(fill_defaults=True)
    now = _now_iso()
    available_now = available_at(now, timeline)
    claims, narrowings = retro_campaign_laws(timeline=timeline, identity=identity)

    # Live negative control, recorded not just tested: widening any retro-scoped
    # law without a replicating specimen must raise.
    refusal_witness: dict[str, Any]
    probe = claims[0] if claims else None
    try:
        if probe is None:
            raise ScopeViolation("no claim to probe", reason="no_named_replicating_specimen")
        widen(probe, "model", replicating_specimen=None, replication_receipt=None, timeline=timeline)
        refusal_witness = {"raised": False, "error": "widen() returned; the gate is broken"}
    except ScopeViolation as e:
        refusal_witness = {
            "raised": True,
            "type": type(e).__name__,
            "reason": e.reason,
            "law_id": e.law_id,
            "axis": e.axis,
            "message": str(e),
        }

    try:
        if probe is not None:
            conclude(probe, "All Hawking models behave this way")
        universal_refused = {"raised": False}
    except OverbroadConclusion as e:
        universal_refused = {"raised": True, "reason": e.reason, "message": str(e)}

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Time-index the scientific universe of Odyssey I so a law cannot "
            "silently inherit specimens that were not sealed when it was made. "
            "Scope begins MODEL_LOCAL / ORGAN_LOCAL / MACHINE_LOCAL. Widening "
            "without a named replicating specimen is impossible."
        ),
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "parent": {
            "specimen": PARENT_SPECIMEN,
            "model_id": PARENT_MODEL_ID,
            "resident_identity": PARENT_RESIDENT,
            "organ": PARENT_ORGAN_MLP,
            "machine": PARENT_MACHINE,
            "teacher_layers": list(TEACHER_LAYERS),
            "teacher_corpus_rows": TEACHER_ROWS,
        },
        "axes": list(AXES),
        "narrow_tiers": dict(NARROW_TIERS),
        "replicated_tiers": dict(REPLICATED_TIERS),
        "scope_kinds": list(SCOPE_KINDS),
        "experiment_identity_fields": list(EXPERIMENT_IDENTITY_FIELDS),
        "experiment_identity": identity,
        "timeline": {
            "lake": timeline.get("lake"),
            "lake_mounted": timeline.get("lake_mounted"),
            "n_known": timeline.get("n_known"),
            "n_sealed": timeline.get("n_sealed"),
            "n_in_flight": timeline.get("n_in_flight"),
            "rule": timeline.get("rule"),
            "sealed": [
                {
                    "specimen": r.get("specimen"),
                    "sealed": r.get("sealed"),
                    "sealed_at": r.get("sealed_at"),
                    "sealed_at_source": r.get("sealed_at_source"),
                    "landed_at": r.get("landed_at"),
                    "landed_at_source": r.get("landed_at_source"),
                    "owner": r.get("owner"),
                    "curriculum_role": r.get("curriculum_role"),
                    "tree_digest": r.get("tree_digest"),
                    "n_files": r.get("n_files"),
                    "bytes_hashed": r.get("bytes_hashed"),
                }
                for r in timeline.get("sealed") or []
            ],
            "in_flight": timeline.get("in_flight") or [],
            "n_stale_partials": len(timeline.get("stale_partials") or []),
            "stale_partials": [
                {
                    "specimen": r.get("specimen"),
                    "landed_at": r.get("landed_at"),
                    "incomplete_markers": r.get("incomplete_markers"),
                }
                for r in (timeline.get("stale_partials") or [])
            ],
            "available_now": available_now,
            "available_now_clause": evidence_universe_clause(available_now),
            "as_of_now": now,
        },
        "laws": [c.to_dict() for c in claims],
        "n_laws": len(claims),
        "n_narrowed": sum(1 for c in claims if c.narrowed),
        "narrowings": narrowings,
        "refusal_witness": {
            "widen_without_specimen": refusal_witness,
            "universal_conclusion": universal_refused,
        },
        "gaps_closed": [
            "laws carry the available-specimen set at as_of and a three-axis scope tier",
            "widen() raises ScopeViolation without a named replicating specimen",
            "failed transfer is recorded and does not move a tier",
            "hindsight contamination (a later-sealed specimen in an earlier available-set) raises",
            "every experiment binds the six identity fields or is refused",
            "campaign laws retro-scoped to qwen3.8-27b sealed-3.14; over-broad forms narrowed and recorded",
            "seal times read from EXTERNAL_SPECIMEN_SEAL / SPECIMEN_VERIFICATION / live HF metadata, not a fixture",
        ],
        "negative_findings": [
            "ModelLake seals on disk are often MANIFEST_ONLY; scientific seal is the independent whole-tree verification",
            "three (or more) downloads can sit in partial/ while five curriculum roles are sealed — the available-set is the sealed set, not the lake listing",
            "LM head landing on 497.4 is a same-parent coincidence, not an organ replication of the MLP ARM A law",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    path = write_receipt(RECEIPT, doc, RECORDED_BY)
    return {"path": str(path), "doc": doc, "n_laws": len(claims), "n_sealed": timeline.get("n_sealed")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true", help="write receipts/future/CLAIM_SCOPE.json")
    args = parser.parse_args(argv)
    if args.build:
        out = build()
        print(f"wrote {out['path']} n_laws={out['n_laws']} n_sealed={out['n_sealed']}")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
