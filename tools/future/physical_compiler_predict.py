"""Foundation for a learned physical compiler. This is NOT a learned compiler.

A schema is not a learned compiler. This module loads REAL historical
physical observations, graph features, backend costs and actual outcomes
from receipts/, exposes a train/eval split and an uncertainty field, and
offers a prediction API that REFUSES to fabricate a number it did not
learn.

Do not train a model here. The deterministic complete_ebpw calculator
remains the cost authority. lpc_dataset.ingest_from_disk and
lpc_baselines.held_out_splits are the existing LPC contract this plugs
into, not a parallel one.

    python3 tools/future/physical_compiler_predict.py --build
    python3 -m pytest tools/future/test_physical_compiler_predict.py -q
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
from typing import Any, Mapping, Sequence

from tools.future import lpc_baselines as lb
from tools.future import lpc_dataset as ds
from tools.future._common import REPO, write_receipt

RECEIPT = "PHYSICAL_COMPILER_FOUNDATION.json"
SCHEMA = "hawking.future.physical_compiler_foundation.v1"
VERSION = 1
RECORDED_BY = "tools/future/physical_compiler_predict.py"

UNLEARNED = "UNLEARNED"
UNLEARNED_STATEMENT = (
    "A schema is not a learned compiler. No model is trained in this "
    "process. predict() refuses to fabricate a cost, latency or TPS."
)

EVIDENCE_TIERS = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "COST_MODEL",
    "CYCLE_APPROX",
    "HARDWARE_MEASURED",
)

PHYSICAL_SOURCES: tuple[str, ...] = (
    "receipts/headless/ACCELERATOR_SCOREBOARD.json",
    "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
    "receipts/headless/QWEN27_TOKEN_NS_BUDGET.json",
    "receipts/future/ORGAN_BANDWIDTH.json",
    "receipts/future/BA_DELTA_AB.json",
    "receipts/future/ECONOMICS_CALIBRATION.json",
    "receipts/future/DEVICE_COMPILER.json",
    "receipts/future/PHYSICAL_PRIMITIVES.json",
    "receipts/future/COMPLETE_EBPW.json",
    "receipts/future/RESIDENT_TOKEN_BUDGET.json",
)


class UnlearnedCompilerError(RuntimeError):
    """Asked to predict without a trained model, or to honour a fake one."""


def _read(rel: str) -> dict[str, Any] | None:
    path = REPO / rel
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _obs(
    *,
    observation_id: str,
    source_receipt: str,
    evidence_tier: str,
    source_evidence_class: Any,
    graph_features: Mapping[str, Any],
    backend_cost: Mapping[str, Any] | None,
    actual_outcome: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if evidence_tier not in EVIDENCE_TIERS:
        raise UnlearnedCompilerError(f"unknown evidence_tier {evidence_tier!r}")
    return {
        "observation_id": observation_id,
        "source_receipt": source_receipt,
        "evidence_tier": evidence_tier,
        "source_evidence_class": source_evidence_class,
        "graph_features": dict(graph_features),
        "backend_cost": dict(backend_cost) if backend_cost else None,
        "actual_outcome": dict(actual_outcome) if actual_outcome else None,
        "trained_on": False,
        "schema_is_not_a_learned_compiler": True,
    }


def observations_from_lpc_ingest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """CALL SITE: tools.future.lpc_dataset.ingest_from_disk."""
    rows, report = ds.ingest_from_disk()
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        rid = str(row.get("row_id") or f"lpc:{i:04d}")
        klass = str(row.get("contamination_class") or "UNKNOWN")
        # Reading a scoreboard/queue/budget file is STATIC for us.
        # PROTECTED_ABSOLUTE on the source row is preserved, not promoted.
        out.append(
            _obs(
                observation_id=f"lpc:{rid}",
                source_receipt=str(
                    row.get("source_receipt")
                    or row.get("source")
                    or "lpc_dataset.ingest_from_disk"
                ),
                evidence_tier="STATIC",
                source_evidence_class=klass,
                graph_features={
                    "model": row.get("model"),
                    "organ": row.get("organ_fingerprint"),
                    "backend": row.get("backend"),
                    "representation": row.get("representation"),
                    "layout": row.get("layout"),
                    "tile": row.get("tile"),
                    "fusion": row.get("fusion"),
                    "physical_graph_identity": row.get("physical_graph_identity"),
                    "machine_genome": row.get("machine_genome"),
                },
                backend_cost=None,
                actual_outcome={
                    "latency": row.get("latency"),
                    "dispatches": row.get("dispatches"),
                    "active_bytes": row.get("active_bytes"),
                    "resident_bytes": row.get("resident_bytes"),
                    "synchronization": row.get("synchronization"),
                },
            )
        )
    return out, report


def observations_from_organ_bandwidth() -> list[dict[str, Any]]:
    rel = "receipts/future/ORGAN_BANDWIDTH.json"
    doc = _read(rel)
    if doc is None:
        return []
    organs = doc.get("organs")
    if not isinstance(organs, list):
        return []
    src_class = doc.get("evidence_class")
    out: list[dict[str, Any]] = []
    for raw in organs:
        if not isinstance(raw, Mapping) or not raw.get("organ"):
            continue
        organ = str(raw["organ"])
        out.append(
            _obs(
                observation_id=f"organ_bandwidth:{organ}",
                source_receipt=rel,
                evidence_tier="STATIC",
                source_evidence_class=src_class,
                graph_features={
                    "organ": organ,
                    "backend": "metal",
                },
                backend_cost=None,
                actual_outcome={
                    "active_bytes": raw.get("active_bytes"),
                    "dispatches": raw.get("dispatches"),
                    "organ_gpu_ms": raw.get("gpu_ms"),
                    "effective_gb_s": raw.get("effective_gb_s"),
                    "byte_share": raw.get("byte_share"),
                    "time_share": raw.get("time_share"),
                },
            )
        )
    return out


def observations_from_ba_delta() -> list[dict[str, Any]]:
    rel = "receipts/future/BA_DELTA_AB.json"
    doc = _read(rel)
    if doc is None:
        return []
    exact = doc.get("exact") if isinstance(doc.get("exact"), Mapping) else {}
    derived = doc.get("derived") if isinstance(doc.get("derived"), Mapping) else {}
    return [
        _obs(
            observation_id="ba_delta_ab",
            source_receipt=rel,
            evidence_tier="STATIC",
            source_evidence_class=doc.get("evidence_class"),
            graph_features={
                "organ": "deltanet",
                "backend": "metal",
                "lever": doc.get("lever"),
            },
            backend_cost=None,
            actual_outcome={
                "dispatches_removed": exact.get("dispatches_removed"),
                "sealed_dispatches_per_decode_step": exact.get(
                    "sealed_dispatches_per_decode_step"
                ),
                "badelta_dispatches_per_decode_step": exact.get(
                    "badelta_dispatches_per_decode_step"
                ),
                "ms_saved_per_token": derived.get("ms_saved_per_token"),
                "us_per_removed_dispatch": derived.get("us_per_removed_dispatch"),
                "token_ids_identical": exact.get("token_ids_identical"),
            },
        )
    ]


def backend_costs_from_economics() -> list[dict[str, Any]]:
    """Stream-class catalog rates. COST_MODEL, not HARDWARE_MEASURED."""
    rel = "receipts/future/ECONOMICS_CALIBRATION.json"
    doc = _read(rel)
    if doc is None:
        return []
    classes = doc.get("stream_classes")
    if not isinstance(classes, Mapping):
        return []
    src_class = doc.get("evidence_class")
    out: list[dict[str, Any]] = []
    for name, row in classes.items():
        if not isinstance(row, Mapping):
            continue
        out.append(
            _obs(
                observation_id=f"backend_cost:{name}",
                source_receipt=rel,
                evidence_tier="COST_MODEL",
                source_evidence_class=src_class,
                graph_features={"stream_class": name, "backend": "metal"},
                backend_cost={
                    "stream_class": name,
                    "ms_per_gb_saved": row.get("ms_per_gb_saved"),
                    "on_critical_path": row.get("on_critical_path"),
                    "catalog_billing": row.get("catalog_billing"),
                    "evidence_tier": "COST_MODEL",
                },
                actual_outcome=None,
            )
        )
    return out


def graph_features_from_device_compiler() -> list[dict[str, Any]]:
    rel = "receipts/future/DEVICE_COMPILER.json"
    doc = _read(rel)
    if doc is None:
        return []
    plan = ((doc.get("lowering") or {}) if isinstance(doc.get("lowering"), Mapping) else {}).get(
        "plan"
    )
    if not isinstance(plan, list):
        return []
    src_class = doc.get("evidence_class") or "STATIC_ONLY"
    out: list[dict[str, Any]] = []
    for raw in plan:
        if not isinstance(raw, Mapping):
            continue
        organ = raw.get("organ")
        if not organ:
            continue
        ident = raw.get("compiled_identity") if isinstance(raw.get("compiled_identity"), Mapping) else {}
        pipeline = ident.get("pipeline") if isinstance(ident.get("pipeline"), Mapping) else {}
        out.append(
            _obs(
                observation_id=f"device_compiler:{organ}",
                source_receipt=rel,
                evidence_tier="STATIC",
                source_evidence_class=src_class,
                graph_features={
                    "organ": organ,
                    "backend": "metal",
                    "status": raw.get("status"),
                    "entry_point": raw.get("entry_point"),
                    "occupying_kind": (raw.get("occupying") or {}).get("kind")
                    if isinstance(raw.get("occupying"), Mapping)
                    else None,
                    "shader_hash_present": bool(raw.get("shader_hash") or ident.get("shader_hash")),
                    "pipeline_created": bool(pipeline.get("created")),
                    "thread_execution_width": pipeline.get("thread_execution_width"),
                    "max_total_threads_per_threadgroup": pipeline.get(
                        "max_total_threads_per_threadgroup"
                    ),
                    "archive_bytes": ident.get("archive_bytes"),
                    "compile_time_science_only": True,
                },
                backend_cost=None,
                actual_outcome={
                    "compiled": raw.get("status") == "COMPILED",
                    "dispatched": False,
                },
            )
        )
    return out


def graph_features_from_primitives() -> list[dict[str, Any]]:
    rel = "receipts/future/PHYSICAL_PRIMITIVES.json"
    doc = _read(rel)
    if doc is None:
        return []
    prims = doc.get("primitives")
    if not isinstance(prims, list):
        return []
    src_class = doc.get("evidence_class") or "STATIC_ONLY"
    out: list[dict[str, Any]] = []
    for raw in prims:
        if not isinstance(raw, Mapping) or not raw.get("name"):
            continue
        name = str(raw["name"])
        out.append(
            _obs(
                observation_id=f"primitive:{name}",
                source_receipt=rel,
                evidence_tier="STATIC",
                source_evidence_class=src_class,
                graph_features={
                    "primitive": name,
                    "atlas_index": raw.get("atlas_index"),
                    "legal_memory_tiers": raw.get("legal_memory_tiers"),
                    "organ_classes": raw.get("organ_classes"),
                    "behavior_taxonomy": raw.get("behavior_taxonomy"),
                    "in_atlas": raw.get("in_atlas"),
                },
                backend_cost=None,
                actual_outcome=None,
            )
        )
    return out


def load_physical_observations() -> dict[str, Any]:
    """Load real historical physical observations. Non-empty or refused."""
    lpc_rows, lpc_report = observations_from_lpc_ingest()
    batches = (
        ("lpc_dataset.ingest_from_disk", lpc_rows),
        ("receipts/future/ORGAN_BANDWIDTH.json", observations_from_organ_bandwidth()),
        ("receipts/future/BA_DELTA_AB.json", observations_from_ba_delta()),
        ("receipts/future/ECONOMICS_CALIBRATION.json", backend_costs_from_economics()),
        ("receipts/future/DEVICE_COMPILER.json", graph_features_from_device_compiler()),
        ("receipts/future/PHYSICAL_PRIMITIVES.json", graph_features_from_primitives()),
    )
    observations: list[dict[str, Any]] = []
    loaded: list[str] = []
    by_source: dict[str, int] = {}
    seen: set[str] = set()
    for label, rows in batches:
        if not rows:
            continue
        loaded.append(label)
        by_source[label] = len(rows)
        for row in rows:
            oid = row["observation_id"]
            if oid in seen:
                continue
            seen.add(oid)
            observations.append(row)
    if not observations:
        raise UnlearnedCompilerError(
            "no historical physical observations loaded; the foundation is empty"
        )
    n_with_outcome = sum(1 for o in observations if o.get("actual_outcome"))
    n_with_graph = sum(1 for o in observations if o.get("graph_features"))
    n_with_cost = sum(1 for o in observations if o.get("backend_cost"))
    return {
        "schema": SCHEMA,
        "observations": observations,
        "n_observations": len(observations),
        "named_receipts_loaded": loaded,
        "by_source": by_source,
        "n_with_actual_outcome": n_with_outcome,
        "n_with_graph_features": n_with_graph,
        "n_with_backend_cost": n_with_cost,
        "lpc_ingest": {
            "n": lpc_report["inventory"]["n"],
            "complete": lpc_report["inventory"]["complete"],
            "by_source": lpc_report["inventory"]["by_source"],
            "call_site": "tools.future.lpc_dataset.ingest_from_disk",
        },
        "claim_boundary": (
            "Historical physical observations projected for a future learned "
            "compiler. evidence_tier is STATIC or COST_MODEL: this process "
            "reads receipts, it does not measure. Not a trained model."
        ),
    }


def train_eval_split(
    observations: Sequence[Mapping[str, Any]],
    *,
    eval_frac: float = 0.25,
) -> dict[str, Any]:
    """Deterministic hash split plus LPC held-out splits. No shuffle, no leak."""
    if not 0.0 < eval_frac < 1.0:
        raise UnlearnedCompilerError("eval_frac must be in (0, 1)")
    train_ids: list[str] = []
    eval_ids: list[str] = []
    for row in observations:
        oid = str(row.get("observation_id") or "")
        if not oid:
            continue
        digest = hashlib.sha256(oid.encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < eval_frac:
            eval_ids.append(oid)
        else:
            train_ids.append(oid)
    train_set = set(train_ids)
    eval_set = set(eval_ids)
    leaked = sorted(train_set & eval_set)
    if leaked:
        raise UnlearnedCompilerError(f"train/eval leak: {leaked[:5]}")

    # CALL SITE: tools.future.lpc_baselines.held_out_splits on LPC-shaped rows.
    lpc_like = [
        {
            "row_id": o.get("observation_id"),
            "organ_fingerprint": (o.get("graph_features") or {}).get("organ"),
            "model": (o.get("graph_features") or {}).get("model"),
            "machine_genome": (o.get("graph_features") or {}).get("machine_genome"),
            "backend": (o.get("graph_features") or {}).get("backend"),
            **{
                k: None
                for k in ds.REQUIRED_FIELDS
                if k
                not in {
                    "organ_fingerprint",
                    "model",
                    "machine_genome",
                    "backend",
                }
            },
            "contamination_class": "STATIC_ONLY",
            "absence_reasons": {
                k: "NOT_IN_SOURCE"
                for k in ds.REQUIRED_FIELDS
                if k
                not in {
                    "organ_fingerprint",
                    "model",
                    "machine_genome",
                    "backend",
                    "contamination_class",
                }
            },
        }
        for o in observations
        if (o.get("graph_features") or {}).get("organ")
    ]
    held_out: list[dict[str, Any]] = []
    if lpc_like:
        for split in lb.held_out_splits(lpc_like, axis="organ"):
            held_out.append(
                {
                    "axis": split.axis,
                    "field": split.field,
                    "holdout_key": split.holdout_key,
                    "n_train": len(split.train_ids),
                    "n_holdout": len(split.holdout_ids),
                    "no_leak": lb.split_has_no_leak(split),
                }
            )
    return {
        "eval_frac": eval_frac,
        "method": "sha256(observation_id) prefix vs eval_frac; plus lpc_baselines.held_out_splits(organ)",
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "leak": leaked,
        "held_out_by_organ": held_out,
        "call_site": "tools.future.lpc_baselines.held_out_splits",
    }


def _unlearned_prediction(
    *,
    features: Mapping[str, Any] | None,
    model_supplied: bool,
) -> dict[str, Any]:
    return {
        "status": UNLEARNED,
        "trained": False,
        "value": None,
        "confidence": None,
        "uncertainty": {
            "kind": "UNDEFINED",
            "reason": "no trained model; uncertainty is not estimable",
        },
        "evidence_tier": "STATIC",
        "statement": UNLEARNED_STATEMENT,
        "schema_is_not_a_learned_compiler": True,
        "model_supplied": model_supplied,
        "model_ignored": model_supplied,
        "features_seen": bool(features),
        "authority": (
            "deterministic complete_ebpw arithmetic remains the cost authority; "
            "this predictor is unlearned and is not consulted for a billed figure"
        ),
    }


def predict(
    features: Mapping[str, Any] | None = None,
    *,
    model: Any = None,
) -> dict[str, Any]:
    """Learned physical-compiler prediction API.

    Always UNLEARNED. A caller-supplied object claiming trained=True is
    ignored: this process never trained it. Never returns a confident
    numeric value.
    """
    return _unlearned_prediction(features=features, model_supplied=model is not None)


def require_numeric_prediction(
    features: Mapping[str, Any] | None = None,
    *,
    model: Any = None,
) -> float:
    """If asked to predict a number without a trained model, say so."""
    raise UnlearnedCompilerError(
        "no trained model; refusing to fabricate a prediction "
        f"(features_present={bool(features)}, model_supplied={model is not None})"
    )


def baseline_predict(
    query: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    method: str = "nearest",
) -> dict[str, Any]:
    """Hand-written LPC baseline (nearest / rule). Not a learned compiler.

    CALL SITE: tools.future.lpc_baselines.predict
    """
    pred = lb.predict(query, rows, method=method)
    return {
        "status": pred.status,
        "value": pred.value,
        "uncertainty": pred.uncertainty,
        "reason": pred.reason,
        "method": pred.method or method,
        "learned": False,
        "schema_is_not_a_learned_compiler": True,
        "note": (
            "lpc_baselines nearest/rule is a declared baseline, not a trained "
            "physical compiler. Use predict() for the learned-compiler API, "
            "which is unlearned."
        ),
        "call_site": "tools.future.lpc_baselines.predict",
    }


def build() -> dict[str, Any]:
    bundle = load_physical_observations()
    split = train_eval_split(bundle["observations"])
    refusal = predict({"organ": "mlp", "backend": "metal"})
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "purpose": (
            "Data and interface foundation for a future learned physical "
            "compiler. The prediction API is unlearned."
        ),
        "named_receipts_loaded": bundle["named_receipts_loaded"],
        "n_observations": bundle["n_observations"],
        "n_with_actual_outcome": bundle["n_with_actual_outcome"],
        "n_with_graph_features": bundle["n_with_graph_features"],
        "n_with_backend_cost": bundle["n_with_backend_cost"],
        "by_source": bundle["by_source"],
        "lpc_ingest": bundle["lpc_ingest"],
        "train_eval_split": {
            "n_train": split["n_train"],
            "n_eval": split["n_eval"],
            "eval_frac": split["eval_frac"],
            "method": split["method"],
            "leak": split["leak"],
            "n_held_out_by_organ": len(split["held_out_by_organ"]),
            "call_site": split["call_site"],
        },
        "prediction_api": refusal,
        "claim_boundary": bundle["claim_boundary"],
        "not_a_learned_compiler": True,
    }
    write_receipt(RECEIPT, doc, RECORDED_BY)
    bundle["train_eval_split"] = split
    bundle["prediction_api"] = refusal
    return bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    bundle = build() if args.build else load_physical_observations()
    print(
        json.dumps(
            {
                "n_observations": bundle["n_observations"],
                "named_receipts_loaded": bundle["named_receipts_loaded"],
                "by_source": bundle["by_source"],
                "prediction": predict(),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
