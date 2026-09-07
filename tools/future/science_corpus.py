"""Science-dataset corpus: make historical Hawking records legible.

Hawking already produces hypotheses, experiments, scars, measurements,
WorkUnits, outcomes and laws. This module is the adapter layer that
projects REAL historical receipts onto one corpus schema so a future
learned policy or learned physical compiler can plug in.

It does not train a model. It does not rewrite history. Adapters read
old schema versions in place (schema evolution, not migration).

    python3 tools/future/science_corpus.py --build
    python3 -m pytest tools/future/test_science_corpus.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from tools.future._common import REPO, git, write_receipt
from tools.future.experiment_policy import apply_deterministic_belief_update
from tools.future.odyssey2_law_store import LAW_FIELDS

RECEIPT = "SCIENCE_CORPUS.json"
SCHEMA = "hawking.future.science_corpus.v1"
VERSION = 1
RECORDED_BY = "tools/future/science_corpus.py"

KINDS = (
    "hypothesis",
    "experiment",
    "outcome",
    "scar",
    "measurement",
    "law",
)

# Contract evidence tiers. Never merge. A historical DIAGNOSTIC_RELATIVE
# receipt is still STATIC here: we are reading a file, not measuring.
EVIDENCE_TIERS = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "COST_MODEL",
    "CYCLE_APPROX",
    "HARDWARE_MEASURED",
)

RECORD_KEYS = (
    "kind",
    "record_id",
    "source_receipt",
    "source_schema",
    "schema_family",
    "evidence_tier",
    "source_evidence_class",
    "key_fields",
)

# Stable field vocabulary for the retained CUDA hypothesis receipt. The
# one-shot literature generator that originally declared this tuple is
# retired; the corpus adapter must remain able to read its sealed output
# without resurrecting that execution machinery.
REQUIRED_HYPOTHESIS_FIELDS = (
    "id",
    "physical_invariant",
    "hawking_primitive",
    "target_organ",
    "cheapest_falsifier",
    "expected_removed_cost",
    "backend_candidate",
    "transfer_scope",
    "knowledge_source",
    "hypothesis_family",
    "behavior_id",
    "candidate",
)

# Named historical receipts this corpus actually loads. Absence of one
# source skips that adapter; an empty corpus after all adapters is a
# refusal, not a silent zero.
NAMED_SOURCES: tuple[str, ...] = (
    "receipts/future/CUDA_LOWBIT_HYPOTHESES.json",
    "receipts/future/HCLI_MISSION_KERNEL.json",
    "receipts/future/CAMPAIGN_SCARS.json",
    "receipts/future/AUTONOMY_SCARS.json",
    "receipts/future/NEGATIVE_SCIENCE_INDEX.json",
    "receipts/future/ODYSSEY2_LAW_STORE.json",
    "receipts/future/HCLI_FUTURE_WORKUNITS.json",
    "receipts/future/BA_DELTA_AB.json",
    "receipts/future/ORGAN_BANDWIDTH.json",
    "receipts/future/COMPLETE_EBPW.json",
    "receipts/future/ECONOMICS_CALIBRATION.json",
    "receipts/future/RESIDENT_TOKEN_BUDGET.json",
    "receipts/future/EXPERIMENT_TURNAROUND.json",
    "receipts/future/META_EXPERIMENT_FUNNEL.json",
)

_SCHEMA_VERSION = re.compile(r"\.v\d+$")


class CorpusRefused(RuntimeError):
    """The corpus cannot be built without guessing."""


def schema_family(schema: str | None) -> str:
    """Strip a trailing .vN so v0/v1/v2 of one family share an adapter."""
    if not schema:
        return ""
    return _SCHEMA_VERSION.sub("", str(schema))


def _first(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Schema-evolution lookup: first present non-empty key wins."""
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _as_mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _read_json(rel: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load a historical receipt from disk, else from HEAD (sparse checkout)."""
    path = REPO / rel
    meta = {"path": rel, "on_disk": path.is_file(), "loaded": False, "loaded_from": None}
    text: str | None = None
    if path.is_file():
        try:
            text = path.read_text()
            meta["loaded_from"] = "disk"
        except OSError:
            text = None
    if text is None:
        blob = git("show", f"HEAD:{rel}")
        if blob:
            text = blob
            meta["loaded_from"] = "git"
    if text is None:
        return None, meta
    try:
        doc = json.loads(text)
    except ValueError:
        return None, meta
    if not isinstance(doc, dict):
        return None, meta
    meta["loaded"] = True
    meta["source_schema"] = doc.get("schema")
    return doc, meta


def _tier_for_source(doc: Mapping[str, Any], *, default: str = "STATIC") -> str:
    """Our claim while reading a file. Never promote to HARDWARE_MEASURED."""
    klass = str(
        doc.get("evidence_class")
        or (doc.get("bench") or {}).get("measurement_state")
        or default
    ).upper()
    if "COST" in klass or klass in {"SELF_MEASURED_DIRTY"}:
        # Catalog rates / dirty self-measure: still not hardware-authority.
        # Economics is used as a cost model; the sidecar has no GPU lease.
        if doc.get("gpu_authority") is True:
            return "STATIC"
        return "COST_MODEL" if "stream_classes" in doc or "COST" in klass else "STATIC"
    return "STATIC"


def _record(
    *,
    kind: str,
    record_id: str,
    source_receipt: str,
    source_schema: str | None,
    evidence_tier: str,
    key_fields: Mapping[str, Any],
    source_evidence_class: Any = None,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise CorpusRefused(f"unknown corpus kind {kind!r}")
    if evidence_tier not in EVIDENCE_TIERS:
        raise CorpusRefused(f"unknown evidence_tier {evidence_tier!r}")
    if not record_id:
        raise CorpusRefused("record_id is empty")
    rec = {
        "kind": kind,
        "record_id": record_id,
        "source_receipt": source_receipt,
        "source_schema": source_schema,
        "schema_family": schema_family(source_schema),
        "evidence_tier": evidence_tier,
        "source_evidence_class": source_evidence_class,
        "key_fields": json.loads(json.dumps(dict(key_fields), default=str)),
    }
    return rec


def round_trip(record: Mapping[str, Any]) -> dict[str, Any]:
    """JSON round-trip. Key fields must survive exactly."""
    missing = [k for k in RECORD_KEYS if k not in record]
    if missing:
        raise CorpusRefused(f"corpus record missing {missing}")
    blob = json.dumps(dict(record), sort_keys=True, separators=(",", ":"), default=str)
    out = json.loads(blob)
    if out["key_fields"] != record["key_fields"]:
        raise CorpusRefused("round-trip did not preserve key_fields")
    if out["kind"] != record["kind"] or out["record_id"] != record["record_id"]:
        raise CorpusRefused("round-trip did not preserve kind/record_id")
    return out


def key_fields_preserved(original: Mapping[str, Any], restored: Mapping[str, Any]) -> bool:
    return restored.get("key_fields") == original.get("key_fields")


# ---------------------------------------------------------------------------
# Adapters. Each one reads one historical schema family, including older
# versions (id vs scar_id, claim vs physical_invariant, etc.).
# ---------------------------------------------------------------------------


def adapt_cuda_hypotheses(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    tier = _tier_for_source(doc)
    src_class = doc.get("evidence_class")
    out: list[dict[str, Any]] = []
    rows = doc.get("hypotheses")
    if not isinstance(rows, list):
        return out
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        hid = _first(raw, "id", "hypothesis_id")
        statement = _first(raw, "physical_invariant", "claim", "statement")
        if not hid or not statement:
            continue
        keys = {name: raw.get(name) for name in REQUIRED_HYPOTHESIS_FIELDS if name in raw}
        keys.update(
            {
                "id": hid,
                "statement": statement,
                "hypothesis_family": _first(raw, "hypothesis_family", "family"),
                "target_organ": raw.get("target_organ"),
                "cheapest_falsifier": raw.get("cheapest_falsifier"),
                "backend_candidate": raw.get("backend_candidate"),
                "status": _first(raw, "status", "verdict"),
                "expected_removed_cost": raw.get("expected_removed_cost"),
            }
        )
        out.append(
            _record(
                kind="hypothesis",
                record_id=f"hypothesis:{source_receipt}:{hid}",
                source_receipt=source_receipt,
                source_schema=schema,
                evidence_tier=tier,
                source_evidence_class=src_class,
                key_fields=keys,
            )
        )
    return out


def adapt_mission_kernel(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    """Mission-kernel hypotheses + verdicts as outcomes + a belief update."""
    schema = doc.get("schema")
    tier = _tier_for_source(doc)
    src_class = doc.get("evidence_class") or "STATIC_ONLY"
    out: list[dict[str, Any]] = []
    hyps = doc.get("hypotheses")
    belief: dict[str, Any] = {"hypotheses": {}}
    if isinstance(hyps, list):
        for raw in hyps:
            if not isinstance(raw, Mapping):
                continue
            hid = _first(raw, "id", "hypothesis_id")
            claim = _first(raw, "claim", "statement", "physical_invariant")
            if not hid or not claim:
                continue
            keys = {
                "id": hid,
                "statement": claim,
                "verdict": raw.get("verdict"),
                "evidence": raw.get("evidence"),
                "proposer": raw.get("proposer"),
            }
            out.append(
                _record(
                    kind="hypothesis",
                    record_id=f"hypothesis:{source_receipt}:{hid}",
                    source_receipt=source_receipt,
                    source_schema=schema,
                    evidence_tier=tier,
                    source_evidence_class=src_class,
                    key_fields=keys,
                )
            )
            verdict = raw.get("verdict")
            if verdict:
                outcome_keys = {
                    "id": f"outcome:{hid}",
                    "hypothesis_id": hid,
                    "status": verdict,
                    "observed": raw.get("evidence"),
                    "claim": claim,
                }
                out.append(
                    _record(
                        kind="outcome",
                        record_id=f"outcome:{source_receipt}:{hid}",
                        source_receipt=source_receipt,
                        source_schema=schema,
                        evidence_tier=tier,
                        source_evidence_class=src_class,
                        key_fields=outcome_keys,
                    )
                )
                # Production call: historical verdict writes a deterministic posterior.
                belief = apply_deterministic_belief_update(
                    belief,
                    {
                        "option_id": f"hyp:{hid}",
                        "hypothesis_id": hid,
                        "status": verdict,
                        "observed": raw.get("evidence"),
                        "evidence_tier": tier,
                    },
                )
    if belief.get("hypotheses"):
        out.append(
            _record(
                kind="outcome",
                record_id=f"outcome:{source_receipt}:belief_update",
                source_receipt=source_receipt,
                source_schema=schema,
                evidence_tier=tier,
                source_evidence_class=src_class,
                key_fields={
                    "id": "mission_kernel_belief_update",
                    "update_rule": belief.get("update_rule"),
                    "learned": False,
                    "n_hypotheses": len(belief["hypotheses"]),
                    "posterior_statuses": {
                        hid: row.get("status")
                        for hid, row in belief["hypotheses"].items()
                    },
                },
            )
        )
    scars = doc.get("scars")
    if isinstance(scars, list):
        for i, raw in enumerate(scars):
            if isinstance(raw, str) and raw.strip():
                sid = raw.strip()
                out.append(
                    _record(
                        kind="scar",
                        record_id=f"scar:{source_receipt}:{sid}",
                        source_receipt=source_receipt,
                        source_schema=schema,
                        evidence_tier=tier,
                        source_evidence_class=src_class,
                        key_fields={"id": sid, "statement": sid},
                    )
                )
            elif isinstance(raw, Mapping):
                sid = _first(raw, "id", "scar_id") or f"scar-{i}"
                out.append(
                    _record(
                        kind="scar",
                        record_id=f"scar:{source_receipt}:{sid}",
                        source_receipt=source_receipt,
                        source_schema=schema,
                        evidence_tier=tier,
                        source_evidence_class=src_class,
                        key_fields={
                            "id": sid,
                            "claim_refuted": _first(
                                raw, "claim_refuted", "what_was_wrong", "statement"
                            ),
                            "verdict": raw.get("verdict"),
                        },
                    )
                )
    return out


def adapt_scars(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    """Campaign / autonomy / funnel / economics scar lists, old and new keys."""
    schema = doc.get("schema")
    tier = _tier_for_source(doc)
    src_class = doc.get("evidence_class")
    rows = doc.get("scars")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            continue
        sid = _first(raw, "id", "scar_id", "original_id") or f"scar-{i}"
        claim = _first(
            raw,
            "claim_refuted",
            "what_was_wrong",
            "claim",
            "mechanism",
            "what_happened",
        )
        keys = {
            "id": sid,
            "verdict": _first(raw, "verdict", "status"),
            "claim_refuted": claim,
            "hypothesis_family": _first(raw, "hypothesis_family", "family"),
            "reopen_condition": _first(raw, "reopen_condition", "reopen_if"),
            "observed": _first(raw, "observed", "what_happened", "mechanism"),
            "level": raw.get("level"),
            "source_receipts": raw.get("source_receipts"),
            "generalized_class": _first(raw, "generalized_class", "law"),
            "model": raw.get("model"),
            "organ": raw.get("organ"),
        }
        out.append(
            _record(
                kind="scar",
                record_id=f"scar:{source_receipt}:{sid}",
                source_receipt=source_receipt,
                source_schema=schema,
                evidence_tier=tier,
                source_evidence_class=src_class,
                key_fields=keys,
            )
        )
    return out


def adapt_negative_index(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    return adapt_scars(doc, source_receipt=source_receipt)


def adapt_laws(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    tier = _tier_for_source(doc)
    src_class = doc.get("evidence_class") or doc.get("evidence_strengths")
    rows = doc.get("laws")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        lid = _first(raw, "law_id", "id")
        statement = _first(raw, "statement", "claim")
        if not lid or not statement:
            continue
        keys = {name: raw.get(name) for name in LAW_FIELDS if name in raw}
        keys.update(
            {
                "law_id": lid,
                "statement": statement,
                "scope": raw.get("scope"),
                "evidence_refs": raw.get("evidence_refs"),
                "evidence_strength": raw.get("evidence_strength"),
                "source_model": raw.get("source_model"),
                "architecture_family": raw.get("architecture_family"),
                "organ_class": raw.get("organ_class"),
                "backend": raw.get("backend"),
            }
        )
        out.append(
            _record(
                kind="law",
                record_id=f"law:{source_receipt}:{lid}",
                source_receipt=source_receipt,
                source_schema=schema,
                evidence_tier=tier,
                source_evidence_class=src_class if isinstance(src_class, str) else None,
                key_fields=keys,
            )
        )
    return out


def adapt_work_units(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    tier = _tier_for_source(doc)
    src_class = doc.get("evidence_class") or "STATIC_ONLY"
    rows = doc.get("work_units")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        uid = _first(raw, "id", "experiment_id")
        if not uid:
            continue
        keys = {
            "id": uid,
            "experiment_id": raw.get("experiment_id"),
            "description": raw.get("description"),
            "status": raw.get("status"),
            "classification": raw.get("classification"),
            "command": raw.get("command"),
            "preferred_backend": raw.get("preferred_backend"),
            "source_receipt": raw.get("source_receipt"),
            "blocked_reason": raw.get("blocked_reason"),
            "role": raw.get("role"),
            "species": raw.get("species"),
            "verifier": raw.get("verifier"),
        }
        out.append(
            _record(
                kind="experiment",
                record_id=f"experiment:{source_receipt}:{uid}",
                source_receipt=source_receipt,
                source_schema=schema,
                evidence_tier=tier,
                source_evidence_class=src_class,
                key_fields=keys,
            )
        )
    return out


def adapt_ba_delta(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    tier = _tier_for_source(doc)
    src_class = doc.get("evidence_class")
    out: list[dict[str, Any]] = []
    lever = doc.get("lever")
    exact = doc.get("exact") if isinstance(doc.get("exact"), Mapping) else {}
    derived = doc.get("derived") if isinstance(doc.get("derived"), Mapping) else {}
    out.append(
        _record(
            kind="experiment",
            record_id=f"experiment:{source_receipt}:ba_delta_ab",
            source_receipt=source_receipt,
            source_schema=schema,
            evidence_tier=tier,
            source_evidence_class=src_class,
            key_fields={
                "id": "ba_delta_ab",
                "lever": lever,
                "runs_compared": exact.get("runs_compared"),
                "dispatches_removed": exact.get("dispatches_removed"),
                "token_ids_identical": exact.get("token_ids_identical"),
            },
        )
    )
    # Historical numbers copied under names that are not hardware-claim keys.
    out.append(
        _record(
            kind="measurement",
            record_id=f"measurement:{source_receipt}:ba_delta_exact",
            source_receipt=source_receipt,
            source_schema=schema,
            evidence_tier=tier,
            source_evidence_class=src_class,
            key_fields={
                "id": "ba_delta_exact",
                "dispatches_removed": exact.get("dispatches_removed"),
                "sealed_dispatches_per_decode_step": exact.get(
                    "sealed_dispatches_per_decode_step"
                ),
                "badelta_dispatches_per_decode_step": exact.get(
                    "badelta_dispatches_per_decode_step"
                ),
                "ms_saved_per_token": derived.get("ms_saved_per_token"),
                "us_per_removed_dispatch": derived.get("us_per_removed_dispatch"),
                "gpu_authority": bool(doc.get("gpu_authority")),
            },
        )
    )
    findings = doc.get("findings")
    if isinstance(findings, list):
        for raw in findings:
            if not isinstance(raw, Mapping):
                continue
            fid = _first(raw, "id") or "finding"
            out.append(
                _record(
                    kind="outcome",
                    record_id=f"outcome:{source_receipt}:{fid}",
                    source_receipt=source_receipt,
                    source_schema=schema,
                    evidence_tier=tier,
                    source_evidence_class=src_class,
                    key_fields={
                        "id": fid,
                        "experiment_id": "ba_delta_ab",
                        "status": "RECORDED",
                        "observed": _first(raw, "what", "conclusion"),
                        "why_it_matters": raw.get("why_it_matters"),
                    },
                )
            )
    return out


def adapt_organ_bandwidth(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    tier = _tier_for_source(doc)
    src_class = doc.get("evidence_class")
    organs = doc.get("organs")
    if not isinstance(organs, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in organs:
        if not isinstance(raw, Mapping):
            continue
        organ = raw.get("organ")
        if not organ:
            continue
        out.append(
            _record(
                kind="measurement",
                record_id=f"measurement:{source_receipt}:{organ}",
                source_receipt=source_receipt,
                source_schema=schema,
                evidence_tier=tier,
                source_evidence_class=src_class,
                key_fields={
                    "id": f"organ_bandwidth:{organ}",
                    "organ": organ,
                    "active_bytes": raw.get("active_bytes"),
                    "dispatches": raw.get("dispatches"),
                    "organ_gpu_ms": raw.get("gpu_ms"),
                    "effective_gb_s": raw.get("effective_gb_s"),
                    "byte_share": raw.get("byte_share"),
                    "time_share": raw.get("time_share"),
                    "gpu_authority": bool(doc.get("gpu_authority")),
                },
            )
        )
    return out


def adapt_complete_ebpw(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    tier = "STATIC"
    src_class = doc.get("evidence_class") or "STATIC_ONLY"
    inc = doc.get("incumbent") if isinstance(doc.get("incumbent"), Mapping) else {}
    if not inc:
        return []
    return [
        _record(
            kind="measurement",
            record_id=f"measurement:{source_receipt}:incumbent",
            source_receipt=source_receipt,
            source_schema=schema,
            evidence_tier=tier,
            source_evidence_class=src_class,
            key_fields={
                "id": inc.get("id") or "incumbent",
                "complete_ebpw": inc.get("complete_ebpw"),
                "stored_bytes": inc.get("stored_bytes") or inc.get("payload_bytes"),
                "billed_ms": inc.get("billed_ms"),
                "parent_params": inc.get("parent_params"),
                "is_sub2_executable": inc.get("is_sub2_executable"),
            },
        )
    ]


def adapt_economics(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    # Catalog stream-class rates: COST_MODEL, not a new hardware measure.
    tier = "COST_MODEL"
    src_class = doc.get("evidence_class")
    classes = doc.get("stream_classes")
    out: list[dict[str, Any]] = []
    if isinstance(classes, Mapping):
        for name, row in classes.items():
            if not isinstance(row, Mapping):
                continue
            out.append(
                _record(
                    kind="measurement",
                    record_id=f"measurement:{source_receipt}:stream:{name}",
                    source_receipt=source_receipt,
                    source_schema=schema,
                    evidence_tier=tier,
                    source_evidence_class=src_class,
                    key_fields={
                        "id": f"stream_class:{name}",
                        "stream_class": name,
                        "ms_per_gb_saved": row.get("ms_per_gb_saved"),
                        "on_critical_path": row.get("on_critical_path"),
                        "catalog_billing": row.get("catalog_billing"),
                    },
                )
            )
    out.extend(adapt_scars(doc, source_receipt=source_receipt))
    for rec in out:
        if rec["kind"] == "scar":
            rec["evidence_tier"] = "STATIC"
    return out


def adapt_token_budget(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    tier = _tier_for_source(doc)
    src_class = doc.get("evidence_class")
    derived = doc.get("derived") if isinstance(doc.get("derived"), Mapping) else {}
    budget = doc.get("budget") if isinstance(doc.get("budget"), Mapping) else {}
    return [
        _record(
            kind="measurement",
            record_id=f"measurement:{source_receipt}:resident_token_budget",
            source_receipt=source_receipt,
            source_schema=schema,
            evidence_tier=tier,
            source_evidence_class=src_class,
            key_fields={
                "id": "resident_token_budget",
                "production_ms_per_token": derived.get("production_ms_per_token"),
                "production_gpu_ms_per_token": derived.get("production_gpu_ms_per_token"),
                "production_dispatches_per_token": derived.get(
                    "production_dispatches_per_token"
                ),
                "host_gap_ms_per_token": derived.get("host_gap_ms_per_token"),
                "total_decode_ms_per_token": budget.get("total_decode_ms_per_token"),
                "gpu_authority": bool(doc.get("gpu_authority")),
            },
        )
    ]


def adapt_turnaround(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    tier = "STATIC"
    src_class = doc.get("evidence_class")
    phases = doc.get("phases")
    out: list[dict[str, Any]] = [
        _record(
            kind="experiment",
            record_id=f"experiment:{source_receipt}:turnaround_census",
            source_receipt=source_receipt,
            source_schema=schema,
            evidence_tier=tier,
            source_evidence_class=src_class,
            key_fields={
                "id": "experiment_turnaround",
                "measurement_kind": doc.get("measurement_kind"),
                "dominant_cost": doc.get("dominant_cost"),
            },
        )
    ]
    if isinstance(phases, list):
        for raw in phases:
            if not isinstance(raw, Mapping):
                continue
            name = raw.get("name")
            if not name:
                continue
            out.append(
                _record(
                    kind="measurement",
                    record_id=f"measurement:{source_receipt}:phase:{name}",
                    source_receipt=source_receipt,
                    source_schema=schema,
                    evidence_tier=tier,
                    source_evidence_class=src_class,
                    key_fields={
                        "id": f"turnaround_phase:{name}",
                        "name": name,
                        "state": raw.get("state"),
                        "median_ms": raw.get("median_ms"),
                        "measurement_kind": raw.get("measurement_kind"),
                        "reason": raw.get("reason"),
                    },
                )
            )
    return out


def adapt_experiment_receipt(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    """Project a shared ExperimentReceipt. Nested on a producer doc, or top-level.

    Reuses schema_family / _first. Does not rewrite the historical document.
    """
    schema = doc.get("schema")
    family = schema_family(schema if isinstance(schema, str) else None)
    env: Mapping[str, Any]
    if family == "hawking.experiment.receipt":
        env = doc
    else:
        nested = doc.get("experiment_receipt")
        if not isinstance(nested, Mapping):
            return []
        env = nested
        schema = env.get("schema") or schema
    claim = _first(env, "claim", "statement")
    verdict = _first(env, "verdict", "status")
    if not claim:
        return []
    ident = env.get("identity") if isinstance(env.get("identity"), Mapping) else {}
    if not ident and isinstance(doc.get("artifact_identity"), Mapping):
        ident = doc["artifact_identity"]
    rid = ident.get("identity_key") or _first(env, "id") or "envelope"
    tier = str(env.get("evidence_tier") or _tier_for_source(doc))
    if tier not in EVIDENCE_TIERS:
        tier = "STATIC"
    keys = {
        "id": f"experiment_receipt:{rid}",
        "verdict": verdict,
        "claim": claim,
        "scope": env.get("scope"),
        "facts": _first(env, "facts", "verified_facts"),
        "hypotheses": env.get("hypotheses"),
        "negative_controls": env.get("negative_controls"),
        "failures": env.get("failures"),
        "resource_usage": _first(env, "resource_usage", "resource_use"),
        "uncertainty": env.get("uncertainty"),
        "falsifier": env.get("falsifier"),
        "evidence_tier": env.get("evidence_tier"),
        "identity_key": ident.get("identity_key"),
        "producer": ident.get("producer"),
        "content_sha256": ident.get("content_sha256"),
    }
    return [
        _record(
            kind="experiment",
            record_id=f"experiment:{source_receipt}:{rid}",
            source_receipt=source_receipt,
            source_schema=schema,
            evidence_tier=tier,
            source_evidence_class=doc.get("evidence_class") or env.get("qualification"),
            key_fields=keys,
        )
    ]


def adapt_meta_funnel(doc: Mapping[str, Any], *, source_receipt: str) -> list[dict[str, Any]]:
    schema = doc.get("schema")
    tier = _tier_for_source(doc)
    src_class = doc.get("evidence_class")
    out = [
        _record(
            kind="experiment",
            record_id=f"experiment:{source_receipt}:meta_funnel",
            source_receipt=source_receipt,
            source_schema=schema,
            evidence_tier=tier,
            source_evidence_class=src_class,
            key_fields={
                "id": "meta_experiment_funnel",
                "advance_rule": doc.get("advance_rule"),
                "counts": doc.get("counts"),
            },
        )
    ]
    out.extend(adapt_scars(doc, source_receipt=source_receipt))
    return out


FAMILY_ADAPTERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "hawking.future.cuda_lowbit": adapt_cuda_hypotheses,
    "hawking.future.hcli_mission_kernel": adapt_mission_kernel,
    "hawking.future.campaign_scars": adapt_scars,
    "hawking.future.autonomy_scars": adapt_scars,
    "hawking.future.negative_index": adapt_negative_index,
    "hawking.future.odyssey2_law_store": adapt_laws,
    "hawking.future.workunit_species": adapt_work_units,
    "hawking.future.ba_delta_ab": adapt_ba_delta,
    "hawking.future.organ_bandwidth": adapt_organ_bandwidth,
    "hawking.future.complete_ebpw": adapt_complete_ebpw,
    "hawking.future.economics_calibration": adapt_economics,
    "hawking.future.resident_token_budget": adapt_token_budget,
    "hawking.future.turnaround": adapt_turnaround,
    "hawking.future.meta_funnel": adapt_meta_funnel,
    "hawking.experiment.receipt": adapt_experiment_receipt,
}

# Filename fallback when a document has no schema (older dumps).
FILENAME_ADAPTERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "CUDA_LOWBIT_HYPOTHESES.json": adapt_cuda_hypotheses,
    "HCLI_MISSION_KERNEL.json": adapt_mission_kernel,
    "CAMPAIGN_SCARS.json": adapt_scars,
    "AUTONOMY_SCARS.json": adapt_scars,
    "NEGATIVE_SCIENCE_INDEX.json": adapt_negative_index,
    "ODYSSEY2_LAW_STORE.json": adapt_laws,
    "HCLI_FUTURE_WORKUNITS.json": adapt_work_units,
    "BA_DELTA_AB.json": adapt_ba_delta,
    "ORGAN_BANDWIDTH.json": adapt_organ_bandwidth,
    "COMPLETE_EBPW.json": adapt_complete_ebpw,
    "ECONOMICS_CALIBRATION.json": adapt_economics,
    "RESIDENT_TOKEN_BUDGET.json": adapt_token_budget,
    "EXPERIMENT_TURNAROUND.json": adapt_turnaround,
    "META_EXPERIMENT_FUNNEL.json": adapt_meta_funnel,
}


def adapt_document(
    doc: Mapping[str, Any],
    *,
    source_receipt: str,
) -> list[dict[str, Any]]:
    """Project one historical document. Unknown families are skipped, not rewritten."""
    family = schema_family(doc.get("schema") if isinstance(doc.get("schema"), str) else None)
    adapter = FAMILY_ADAPTERS.get(family)
    if adapter is None:
        adapter = FILENAME_ADAPTERS.get(Path(source_receipt).name)
    out: list[dict[str, Any]] = []
    if adapter is not None:
        out = list(adapter(doc, source_receipt=source_receipt))
    # A producer that adopted ExperimentReceipt still has its old family
    # adapter. Also project the nested envelope without duplicating history.
    if adapter is not adapt_experiment_receipt and isinstance(
        doc.get("experiment_receipt"), Mapping
    ):
        seen = {r["record_id"] for r in out}
        for rec in adapt_experiment_receipt(doc, source_receipt=source_receipt):
            if rec["record_id"] not in seen:
                out.append(rec)
    return out


def measurement_from_ebpw_bill(billed: Mapping[str, Any]) -> dict[str, Any]:
    """Project a complete_ebpw.cost() row into a corpus measurement.

    Called by complete_ebpw.build. STATIC arithmetic, not a hardware measure.
    """
    cid = billed.get("id") or "ebpw"
    return _record(
        kind="measurement",
        record_id=f"measurement:complete_ebpw:{cid}",
        source_receipt="tools/future/complete_ebpw.py",
        source_schema="hawking.future.complete_ebpw.v1",
        evidence_tier="STATIC",
        source_evidence_class="STATIC_ONLY",
        key_fields={
            "id": cid,
            "complete_ebpw": billed.get("complete_ebpw"),
            "stored_bytes": billed.get("stored_bytes"),
            "billed_ms": billed.get("billed_ms"),
            "parent_params": billed.get("parent_params"),
            "is_sub2_executable": billed.get("is_sub2_executable"),
            "reconciled": billed.get("reconciled"),
        },
    )


def load_historical_corpus(
    sources: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Load named historical receipts into a non-empty corpus."""
    wanted = tuple(sources) if sources is not None else NAMED_SOURCES
    records: list[dict[str, Any]] = []
    loaded: list[str] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rel in wanted:
        doc, meta = _read_json(rel)
        if doc is None:
            skipped.append({**meta, "why": "missing_or_unreadable"})
            continue
        adapted = adapt_document(doc, source_receipt=rel)
        if not adapted:
            skipped.append({**meta, "why": "adapter_emitted_nothing"})
            continue
        loaded.append(rel)
        for rec in adapted:
            rid = rec["record_id"]
            if rid in seen:
                continue
            seen.add(rid)
            records.append(rec)
    by_kind = {kind: 0 for kind in KINDS}
    for rec in records:
        by_kind[rec["kind"]] += 1
    if not records:
        raise CorpusRefused(
            "historical corpus is empty; adapters loaded nothing from "
            f"{list(wanted)}"
        )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "named_receipts": list(wanted),
        "named_receipts_loaded": loaded,
        "skipped": skipped,
        "n_records": len(records),
        "by_kind": by_kind,
        "records": records,
        "claim_boundary": (
            "Projection of historical receipts onto a corpus schema. "
            "evidence_tier is STATIC or COST_MODEL because this process is "
            "reading files, not measuring. source_evidence_class preserves "
            "what the receipt itself claimed. Not a learned policy or compiler."
        ),
    }


def build() -> dict[str, Any]:
    corpus = load_historical_corpus()
    # Sealed receipt is a catalog, not a dump of every key_field (those may
    # carry historical numbers under names write_receipt would refuse).
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "purpose": (
            "Legible science dataset over real historical receipts so a "
            "future learned policy / physical compiler can plug in."
        ),
        "named_receipts": corpus["named_receipts"],
        "named_receipts_loaded": corpus["named_receipts_loaded"],
        "n_records": corpus["n_records"],
        "by_kind": corpus["by_kind"],
        "kinds": list(KINDS),
        "evidence_tiers": list(EVIDENCE_TIERS),
        "schema_evolution": (
            "adapters key on schema family with the .vN suffix stripped; "
            "id/scar_id, claim/physical_invariant, reopen_if/reopen_condition "
            "are accepted. History is not rewritten."
        ),
        "claim_boundary": corpus["claim_boundary"],
        "not_a_learned_compiler": True,
        "not_a_learned_policy": True,
    }
    write_receipt(RECEIPT, doc, RECORDED_BY)
    return corpus


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    corpus = build() if args.build else load_historical_corpus()
    print(
        json.dumps(
            {
                "n_records": corpus["n_records"],
                "by_kind": corpus["by_kind"],
                "named_receipts_loaded": corpus["named_receipts_loaded"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
