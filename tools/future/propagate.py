"""PROPAGATE — apply Codex ingest deltas into the seven sidecar consumers.

`codex_ingest.py` classifies Codex artifacts and emits LAW/SCAR deltas that
name downstream consumers. Nothing consumed them: the compounding loop was
open. This module closes it. Each delta is routed through the consumer's own
public API at the weakest evidence strength that consumer offers. A heuristic
cannot mint a verified law. Applying the same set twice is a no-op.

    python3 tools/future/propagate.py --dry-run
    python3 tools/future/propagate.py --apply
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git, RECEIPTS


import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from hcli.physical_graph import PhysicalGraph
from tools.future import hwir
from tools.future import lpc_dataset as lpc
from tools.future import negative_index as ni
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey3_adversary as o3
from tools.future import physical_primitives as phys
from tools.future import tournament as tn
from tools.future import workunit_species as wus


RECEIPT = "PROPAGATION_STATE.json"
SCHEMA = "hawking.future.propagate.v1"
INGEST_RECEIPT = "CODEX_INGEST_STATE.json"
RECORDED_BY = "tools/future/propagate.py"
SNAPSHOT = RECEIPTS / "evidence"

# The seven consumers named by the lane contract, plus the SCAR landing site.
SEVEN_CONSUMERS = (
    "odyssey2_law_store",
    "odyssey3_adversary",
    "lpc_dataset",
    "hwir",
    "workunit_species",
    "tournament",
    "physical_graph",
)
SCAR_LANDING = "negative_index"
ALL_CONSUMERS = SEVEN_CONSUMERS + (SCAR_LANDING,)

WEAKEST_O2_STRENGTH = "ANECDOTE"
ADMISSION_SCOPE = "MODEL_LOCAL"
WEAKEST_TRANSFER_CONFIDENCE = 0.10

LAW_CONSUMERS = (
    "odyssey2_law_store",
    "odyssey3_adversary",
    "lpc_dataset",
    "hwir",
    "workunit_species",
    "tournament",
    "physical_graph",
)
SCAR_CONSUMERS = (
    "odyssey2_law_store",
    "odyssey3_adversary",
    "lpc_dataset",
    "negative_index",
    "workunit_species",
    "tournament",
)

REFUTING_SCAR_TOKENS = frozenset(
    {
        "REFUTED",
        "PROTECTED_REJECT",
        "DIAGNOSTIC_REJECT",
        "NOT_FOR_PROMOTION",
        "REJECTED",
        "FAILED",
        "FAIL",
        "NO-GO",
        "NOGO",
    }
)

RECOVERY_PROBES: tuple[tuple[str, str], ...] = (
    ("tools/future/codex_ingest.py", "delta producer; this module consumes active_deltas[]"),
    ("receipts/future/CODEX_INGEST_STATE.json", "sealed ingest cursor and the 808 live deltas"),
    ("tools/future/odyssey2_law_store.py", "Law + validate_law + promote() that refuses"),
    ("tools/future/odyssey3_adversary.py", "emit_for_law / apply_result scope loop"),
    ("tools/future/lpc_dataset.py", "row_template / validate_row / forbid_zero_imputation"),
    ("tools/future/hwir.py", "HwirGraph + validate; from_organ_map when an organ exists"),
    ("tools/future/workunit_species.py", "define_species / emit_hcli_workunit / validate_emitted_unit"),
    ("tools/future/tournament.py", "can_run / run; sidecar cannot execute the tournament"),
    ("tools/future/physical_primitives.py", "contract / instantiate / PLAN_ONLY lowering"),
    ("tools/future/negative_index.py", "Scar / query / refuse_if_dead — SCAR landing site"),
    ("hcli/physical_graph.py", "PhysicalGraph dataclass; imported read-only, never written"),
    ("receipts/future/EVIDENCE_SNAPSHOT.json", "pinned Codex snapshot; preferred when resolving sources"),
    ("receipts/future/CLAUDE_GLOBAL_FRONTIER.json", "F015: Codex receipts were ingested but not consumed"),
)


def _clip(text: str, n: int = 240) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _probe_path(rel: str) -> dict[str, Any]:
    p = REPO / rel
    in_git = bool(git("ls-tree", "--name-only", "HEAD", rel).strip())
    return {"path": rel, "on_disk": p.exists(), "in_git": in_git}


def resolve_evidence(rel: str) -> dict[str, Any]:
    """Prefer the pinned snapshot; fall back to live headless. Never invent absence."""
    name = Path(rel).name
    pinned = SNAPSHOT / name
    live = REPO / rel
    if pinned.is_file():
        return {
            "path": rel,
            "resolved": str(pinned.relative_to(REPO)),
            "source": "pinned_snapshot",
            "present": True,
        }
    if live.is_file():
        return {
            "path": rel,
            "resolved": rel,
            "source": "live_headless",
            "present": True,
        }
    return {
        "path": rel,
        "resolved": None,
        "source": "unresolved",
        "present": None,
        "note": (
            "not materialized in this worktree; sparse checkout is not evidence "
            "of absence. Propagator did not need the original bytes (delta is self-contained)."
        ),
    }


def load_ingest_state(path: Path | None = None) -> dict[str, Any]:
    p = path or (RECEIPTS / INGEST_RECEIPT)
    if not p.is_file():
        return {}
    try:
        return load_json(p)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def load_ingest_deltas(path: Path | None = None) -> list[dict[str, Any]]:
    doc = load_ingest_state(path)
    deltas = doc.get("active_deltas") or []
    if not isinstance(deltas, list):
        return []
    return [d for d in deltas if isinstance(d, dict)]


def load_previous_apply(path: Path | None = None) -> dict[str, Any] | None:
    p = path or (RECEIPTS / RECEIPT)
    if not p.is_file():
        return None
    try:
        doc = load_json(p)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    if doc.get("dry_run"):
        return None
    return doc


def _ledger_keys(previous: dict[str, Any] | None) -> set[str]:
    if not previous or previous.get("dry_run"):
        return set()
    keys = (previous.get("ledger") or {}).get("applied_keys") or []
    return {str(k) for k in keys}


def _empty_bucket() -> dict[str, Any]:
    return {
        "applied": 0,
        "refused": 0,
        "skipped_as_duplicate": 0,
        "applied_records": [],
        "refusals": [],
    }


def _src_tag(source: str) -> str:
    """Stable short identity of a source path. builtin hash() is not stable."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:10]


def _key(sha: str, consumer: str, record_id: str) -> str:
    return f"{sha}|{consumer}|{record_id}"


class _Sink:
    """Per-run accounting. Keys already in `seen` are skipped as duplicates."""

    def __init__(self, seen: set[str]) -> None:
        self.seen = set(seen)
        self.buckets = {name: _empty_bucket() for name in ALL_CONSUMERS}
        self.new_keys: list[str] = []

    def _bucket(self, consumer: str) -> dict[str, Any]:
        if consumer not in self.buckets:
            self.buckets[consumer] = _empty_bucket()
        return self.buckets[consumer]

    def skip_or_begin(self, consumer: str, key: str) -> bool:
        """Return True if this key is a duplicate (already counted as skipped)."""
        b = self._bucket(consumer)
        if key in self.seen:
            b["skipped_as_duplicate"] += 1
            return True
        return False

    def applied(self, consumer: str, key: str, record: dict[str, Any]) -> None:
        b = self._bucket(consumer)
        b["applied"] += 1
        b["applied_records"].append(record)
        self.seen.add(key)
        self.new_keys.append(key)

    def refused(
        self,
        consumer: str,
        key: str,
        reason: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        b = self._bucket(consumer)
        b["refused"] += 1
        row = {"key": key, "reason": str(reason)}
        if extra:
            row.update(extra)
        b["refusals"].append(row)
        self.seen.add(key)
        self.new_keys.append(key)


def _source_fields(delta: dict[str, Any]) -> tuple[str, str]:
    source = str(delta.get("source") or "")
    sha = str(delta.get("source_sha256") or "")
    return source, sha


def _driver(delta: dict[str, Any]) -> dict[str, Any]:
    d = delta.get("driver")
    return d if isinstance(d, dict) else {}


# ---------------------------------------------------------------------------
# Odyssey II — always MODEL_LOCAL / ANECDOTE. promote() is how higher scopes die.
# ---------------------------------------------------------------------------


def _o2_law_id(sha: str, source: str = "") -> str:
    tag = _src_tag(source) if source else "nosrc"
    stem = sha[:12] if sha else "UNKNOWN"
    return f"LAW-CAND-{stem}-{tag}"


def _candidate_law(delta: dict[str, Any]) -> ols.Law:
    cand = delta.get("odyssey_ii_law_candidate") or {}
    source, sha = _source_fields(delta)
    model = str(cand.get("model") or "UNKNOWN")
    organ = str(cand.get("organ") or "UNKNOWN")
    statement = _clip(
        cand.get("statement_sketch")
        or (delta.get("invalidation") or {}).get("kills", [None])[0]
        or _driver(delta).get("reason")
        or source
        or "ingest candidate"
    )
    conf = {
        "value": WEAKEST_TRANSFER_CONFIDENCE,
        "basis": (
            "weakest evidence ANECDOTE at MODEL_LOCAL; heuristic ingest delta; "
            "sidecar has no promotion authority; driver.confidence is not evidence"
        ),
    }
    law = ols.Law(
        law_id=_o2_law_id(sha, source),
        statement=statement,
        source_model=model,
        source_device="UNKNOWN",
        architecture_family=ols.architecture_family_of(model),
        organ_class=organ,
        backend="UNKNOWN",
        evidence_strength=WEAKEST_O2_STRENGTH,
        evidence_refs=(source, f"sha256:{sha}") if source else (f"sha256:{sha}",),
        scope=ADMISSION_SCOPE,
        transfer_candidates=(),
        transfer_confidence=conf,
        counterexample_requirement=(
            "a measurement that fails the statement; a heuristic ingest delta "
            "cannot discharge a counterexample"
        ),
        expected_saved_experiments=None,
        actual_saved_experiments=None,
        time_to_first_useful_executable_ns=None,
    )
    return ols.validate_law(law)


def apply_odyssey2(delta: dict[str, Any], sink: _Sink) -> ols.Law | None:
    source, sha = _source_fields(delta)
    classification = delta.get("classification")
    law_id = _o2_law_id(sha, source)
    admit_key = _key(sha, "odyssey2_law_store", law_id)

    if classification == "SCAR":
        inv_key = _key(sha, "odyssey2_law_store", f"scar:{_src_tag(source)}")
        if sink.skip_or_begin("odyssey2_law_store", inv_key):
            return None
        inv = delta.get("invalidation") if isinstance(delta.get("invalidation"), dict) else {}
        sink.applied(
            "odyssey2_law_store",
            inv_key,
            {
                "kind": "scar_invalidation",
                "law_id": None,
                "scope": None,
                "source": source,
                "source_sha256": sha,
                "kills": list(inv.get("kills") or []),
                "reopen_condition": inv.get("reopen_condition"),
                "level": inv.get("level") or "MODEL_SPECIFIC",
                "action": "do not admit as a law; feed the negative index",
                "evidence_strength": WEAKEST_O2_STRENGTH,
            },
        )
        return None

    if sink.skip_or_begin("odyssey2_law_store", admit_key):
        return None
    try:
        law = _candidate_law(delta)
    except (ols.LawStoreError, ols.ScopeViolation, TypeError, ValueError) as e:
        sink.refused(
            "odyssey2_law_store",
            admit_key,
            str(e),
            extra={"source": source, "source_sha256": sha, "stage": "validate_law"},
        )
        return None

    rec = law.to_dict()
    rec["source"] = source
    rec["source_sha256"] = sha
    rec["admitted_as"] = "candidate"
    rec["claimed_scope"] = (delta.get("odyssey_ii_law_candidate") or {}).get("proposed_scope")
    sink.applied("odyssey2_law_store", admit_key, rec)

    claimed = (delta.get("odyssey_ii_law_candidate") or {}).get("proposed_scope")
    if claimed and claimed != ADMISSION_SCOPE:
        promo_key = _key(sha, "odyssey2_law_store", f"promote:{claimed}:{_src_tag(source)}")
        if not sink.skip_or_begin("odyssey2_law_store", promo_key):
            try:
                ols.promote(
                    law,
                    str(claimed),
                    {
                        "evidence_strength": WEAKEST_O2_STRENGTH,
                        "models": [law.source_model],
                        "architecture_families": [law.architecture_family],
                        "backends": [law.backend],
                        "machines": [law.source_device],
                        "evidence_refs": list(law.evidence_refs),
                        "counterexample_discharged": False,
                    },
                )
                # A heuristic must not be able to mint a promotion. If promote()
                # ever returns, that is a propagator bug — refuse it anyway.
                sink.refused(
                    "odyssey2_law_store",
                    promo_key,
                    (
                        f"promote() returned {claimed}; ingest heuristics cannot "
                        f"widen past {ADMISSION_SCOPE}"
                    ),
                    extra={
                        "source": source,
                        "source_sha256": sha,
                        "claimed_scope": claimed,
                        "from_scope": law.scope,
                        "stage": "promote_unexpected_success",
                    },
                )
            except ols.ScopeViolation as e:
                sink.refused(
                    "odyssey2_law_store",
                    promo_key,
                    e.reason or str(e),
                    extra={
                        "source": source,
                        "source_sha256": sha,
                        "claimed_scope": claimed,
                        "from_scope": e.from_scope or law.scope,
                        "to_scope": e.to_scope or claimed,
                        "stage": "promote",
                        "exception": type(e).__name__,
                    },
                )
    return law


# ---------------------------------------------------------------------------
# Odyssey III — MODEL_LOCAL law dict, emit_for_law. SCARs may apply_result.
# ---------------------------------------------------------------------------


def _o3_law_from_o2(law: ols.Law, source: str, sha: str) -> dict[str, Any]:
    return {
        "law_id": law.law_id,
        "statement": law.statement,
        "source_model": law.source_model,
        "source_device": law.source_device,
        "architecture_family": law.architecture_family,
        "organ_class": law.organ_class,
        "backend": law.backend,
        "evidence_strength": WEAKEST_O2_STRENGTH,
        "evidence_refs": list(law.evidence_refs),
        "scope": ADMISSION_SCOPE,
        "transfer_candidates": [],
        "transfer_confidence": WEAKEST_TRANSFER_CONFIDENCE,
        "counterexample_requirement": law.counterexample_requirement,
        "source": source,
        "source_sha256": sha,
    }


def _o3_law_from_scar(delta: dict[str, Any]) -> dict[str, Any]:
    source, sha = _source_fields(delta)
    inv = delta.get("invalidation") if isinstance(delta.get("invalidation"), dict) else {}
    kills = [str(k) for k in (inv.get("kills") or []) if k]
    statement = _clip(kills[0] if kills else _driver(delta).get("reason") or source)
    organ = str(inv.get("organ") or "UNKNOWN")
    return {
        "law_id": f"SCAR-TARGET-{sha[:12]}-{_src_tag(source)}",
        "statement": statement,
        "source_model": "UNKNOWN",
        "source_device": "UNKNOWN",
        "architecture_family": "UNKNOWN",
        "organ_class": organ,
        "backend": "UNKNOWN",
        "evidence_strength": WEAKEST_O2_STRENGTH,
        "evidence_refs": [source, f"sha256:{sha}"],
        "scope": ADMISSION_SCOPE,
        "transfer_candidates": [],
        "transfer_confidence": WEAKEST_TRANSFER_CONFIDENCE,
        "counterexample_requirement": str(inv.get("reopen_condition") or "UNKNOWN"),
        "source": source,
        "source_sha256": sha,
    }


def apply_odyssey3(
    delta: dict[str, Any],
    sink: _Sink,
    *,
    o2_law: ols.Law | None,
    runtime: dict[str, dict[str, Any]],
) -> None:
    source, sha = _source_fields(delta)
    if delta.get("classification") == "SCAR":
        body = _o3_law_from_scar(delta)
    elif o2_law is not None:
        body = _o3_law_from_o2(o2_law, source, sha)
    else:
        try:
            body = _o3_law_from_o2(_candidate_law(delta), source, sha)
        except (ols.LawStoreError, ols.ScopeViolation, TypeError, ValueError) as e:
            key = _key(sha, "odyssey3_adversary", f"no-law:{_src_tag(source)}")
            if not sink.skip_or_begin("odyssey3_adversary", key):
                sink.refused(
                    "odyssey3_adversary",
                    key,
                    str(e),
                    extra={"source": source, "source_sha256": sha, "stage": "law_construction"},
                )
            return

    rec_id = str(body["law_id"])
    key = _key(sha, "odyssey3_adversary", rec_id)
    if sink.skip_or_begin("odyssey3_adversary", key):
        return
    try:
        plan = o3.emit_for_law(body)
    except (o3.LawSchemaError, o3.NoAttackError, o3.ScopeUnmovedError, ValueError) as e:
        sink.refused(
            "odyssey3_adversary",
            key,
            str(e),
            extra={"source": source, "source_sha256": sha, "law_id": rec_id, "stage": "emit_for_law"},
        )
        return
    selected = (plan.get("ranked_attacks") or [None])[0] or {}
    runtime[rec_id] = {"law": body, "selected": selected, "source": source, "source_sha256": sha}
    sink.applied(
        "odyssey3_adversary",
        key,
        {
            "law_id": rec_id,
            "scope": plan.get("scope") or body["scope"],
            "n_attacks": plan.get("n_attacks"),
            "selected_attack_id": plan.get("selected_attack_id"),
            "selected_family": plan.get("selected_family"),
            "ranked_attack_ids": list(plan.get("ranked_attack_ids") or []),
            "source": source,
            "source_sha256": sha,
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        },
    )


# ---------------------------------------------------------------------------
# LPC — skeleton row, all hardware null, never imputed.
# ---------------------------------------------------------------------------


def _lpc_row(delta: dict[str, Any]) -> dict[str, Any]:
    source, sha = _source_fields(delta)
    raw = delta.get("learned_physical_compiler_row")
    raw = raw if isinstance(raw, dict) else {}
    inv = delta.get("invalidation") if isinstance(delta.get("invalidation"), dict) else {}
    model = raw.get("model")
    if model in (None, "", "UNKNOWN"):
        model = None
    organ = raw.get("organ") or inv.get("organ")
    if organ in (None, "", "UNKNOWN"):
        organ = None
    representation = raw.get("representation")
    if representation in (None, "", "UNKNOWN"):
        representation = None
    machine = raw.get("machine")
    if machine in (None, "", "UNKNOWN"):
        machine = None
    mapped = lpc.contamination_from_benchmark_class(
        raw.get("contamination") or raw.get("evidence_class") or "STATIC_ONLY"
    )
    # Heuristic ingest cannot raise contamination class. Sidecar emits STATIC_ONLY.
    contamination = "STATIC_ONLY" if mapped != "STATIC_ONLY" else mapped
    reasons: dict[str, str] = {}
    fields: dict[str, Any] = {
        "model": model,
        "organ_fingerprint": organ,
        "representation": representation,
        "machine_genome": machine,
        "physical_graph_identity": None,
        "backend": None,
        "layout": None,
        "tile": None,
        "grouping": None,
        "fusion": None,
        "persistent_resources": None,
        "active_bytes": None,
        "resident_bytes": None,
        "dispatches": None,
        "synchronization": None,
        "latency": None,
        "complete_token_effect": None,
        "contamination_class": contamination,
        "capability": None,
    }
    identity_reason = {
        "model": "NOT_IN_SOURCE",
        "organ_fingerprint": "NOT_IN_SOURCE",
        "representation": "NOT_IN_SOURCE",
        "machine_genome": "PARTIAL_IDENTITY",
        "physical_graph_identity": "STATIC_PLAN_ONLY",
        "backend": "NOT_IN_SOURCE",
        "layout": "NOT_IN_SOURCE",
        "tile": "NOT_IN_SOURCE",
        "grouping": "NOT_IN_SOURCE",
        "fusion": "NOT_IN_SOURCE",
        "persistent_resources": "NOT_IN_SOURCE",
        "complete_token_effect": "AWAITING_PROTECTED_RECEIPT",
        "capability": "AWAITING_PROTECTED_RECEIPT",
    }
    for name in lpc.REQUIRED_FIELDS:
        if fields[name] is None:
            if name in lpc.NUMERIC_FIELDS:
                reasons[name] = "HARDWARE_AUTHORITY_REQUIRED"
            elif name == "contamination_class":
                continue
            else:
                reasons[name] = identity_reason.get(name, "UNMEASURED")
    row = lpc.row_template(
        reasons_for_missing=None,
        absence_reasons=reasons,
        **fields,
        row_id=f"lpc:propagate:{sha[:12]}:{_src_tag(source)}",
        source="codex_ingest_delta",
        source_receipt=source,
        source_sha256=sha,
        evidence_class="STATIC_ONLY",
        bench_state="UNKNOWN",
        technique=raw.get("technique") or "UNKNOWN",
        label=raw.get("label") or delta.get("classification"),
    )
    return row


def apply_lpc(delta: dict[str, Any], sink: _Sink) -> dict[str, Any] | None:
    source, sha = _source_fields(delta)
    row_id = f"lpc:propagate:{sha[:12]}:{_src_tag(source)}"
    key = _key(sha, "lpc_dataset", row_id)
    if sink.skip_or_begin("lpc_dataset", key):
        return None
    row = _lpc_row(delta)
    verdict = lpc.validate_row(row)
    if verdict.get("status") != "VALID":
        sink.refused(
            "lpc_dataset",
            key,
            f"{verdict.get('status')}: {verdict.get('why')}",
            extra={
                "source": source,
                "source_sha256": sha,
                "row_id": row_id,
                "verdict": verdict,
                "stage": "validate_row",
            },
        )
        return None
    # Hardware fields on the ingest skeleton stay null; never copy a number.
    measured = (delta.get("learned_physical_compiler_row") or {}).get("measured")
    if isinstance(measured, dict):
        leaked = [
            k
            for k, v in measured.items()
            if k in {"tps", "token_ns", "gpu_ns", "joules_per_token", "bandwidth_gbps"}
            and isinstance(v, (int, float))
        ]
        if leaked:
            sink.refused(
                "lpc_dataset",
                key,
                f"ingest skeleton leaked hardware numbers on {leaked}; refusing to copy them",
                extra={"source": source, "source_sha256": sha, "stage": "hardware_guard"},
            )
            return None
    rec = {
        "row_id": row_id,
        "status": verdict["status"],
        "complete": verdict.get("complete"),
        "contamination_class": row.get("contamination_class"),
        "model": row.get("model"),
        "organ_fingerprint": row.get("organ_fingerprint"),
        "latency": row.get("latency"),
        "absence_reasons": row.get("absence_reasons"),
        "source": source,
        "source_sha256": sha,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    sink.applied("lpc_dataset", key, rec)
    return row


# ---------------------------------------------------------------------------
# HWIR — validate a candidate graph when spatially meaningful; else record stub.
# ---------------------------------------------------------------------------


def _stub_hwir(organ: str, source: str) -> hwir.HwirGraph:
    node = hwir.HwirNode(
        id=f"cmp.{organ or 'unknown'}",
        kind="compute",
        primitive="TiledProjection",
        organ=organ or "UNKNOWN",
        mapping="ingest candidate projection; no source-dense GEMM; no_dense_rematerialization",
        inputs={"in": "activation"},
        outputs={"out": "activation"},
    )
    return hwir.HwirGraph(
        organ=organ or "UNKNOWN",
        source_receipt=source,
        qualification="STATIC_ONLY",
        semantics_consumed="physical_graph_noetic_native",
        nodes=[node],
        notes=[
            "Propagated HWIR candidate from an ingest delta. Not a bitstream.",
            "Not lowered from an organ-map receipt (organ id did not match a map, or map unresolved).",
        ],
    )


def apply_hwir(delta: dict[str, Any], sink: _Sink) -> None:
    if delta.get("classification") != "LAW":
        return
    source, sha = _source_fields(delta)
    proj = delta.get("hwir_projection") if isinstance(delta.get("hwir_projection"), dict) else {}
    organ = str(proj.get("organ") or "UNKNOWN")
    spatial = bool(proj.get("spatially_meaningful"))
    rec_id = f"{'spatial' if spatial else 'stub'}:{organ}:{_src_tag(source)}"
    key = _key(sha, "hwir", rec_id)
    if sink.skip_or_begin("hwir", key):
        return
    if not spatial:
        sink.applied(
            "hwir",
            key,
            {
                "kind": "non_spatial_stub",
                "organ": organ,
                "spatially_meaningful": False,
                "action": proj.get("action") or "no spatial mapping suggested",
                "reason": proj.get("reason"),
                "source": source,
                "source_sha256": sha,
                "qualification": "STATIC_ONLY",
            },
        )
        return

    graph = None
    origin = "stub_candidate"
    for rel in (hwir.FLASH_ORGAN_MAP, hwir.QWEN_ORGAN_MAP):
        ev = resolve_evidence(rel)
        if not ev.get("resolved"):
            continue
        path = REPO / ev["resolved"]
        try:
            graph = hwir.from_organ_map(path, organ)
            origin = f"from_organ_map:{ev['source']}"
            break
        except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError, KeyError):
            graph = None
            continue
    if graph is None:
        graph = _stub_hwir(organ, source)
        origin = "stub_candidate"
    report = hwir.validate(graph)
    if not report.ok:
        sink.refused(
            "hwir",
            key,
            "; ".join(f"{e.get('code')}:{e.get('path')}" for e in report.errors[:8]) or "validate failed",
            extra={
                "source": source,
                "source_sha256": sha,
                "organ": organ,
                "origin": origin,
                "errors": report.to_dict()["errors"][:12],
                "stage": "validate",
            },
        )
        return
    sink.applied(
        "hwir",
        key,
        {
            "kind": "candidate_graph",
            "organ": graph.organ,
            "spatially_meaningful": True,
            "origin": origin,
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "qualification": graph.qualification,
            "validate_ok": True,
            "fingerprint": graph.fingerprint(),
            "source": source,
            "source_sha256": sha,
        },
    )


# ---------------------------------------------------------------------------
# PhysicalGraph + atlas primitives (read-only hcli; executable primitive contracts)
# ---------------------------------------------------------------------------


def _atlas_primitive_for(name: str) -> str | None:
    raw = str(name or "").strip()
    if not raw or raw == "UNKNOWN":
        return None
    if raw in phys.CONTRACTS:
        return raw
    slug = raw.replace("-", "_").replace(" ", "_")
    for prim, spec in phys.CONTRACTS.items():
        if slug.lower() == prim.lower():
            return prim
        ids = spec.get("behavior_ids") or ()
        if raw in ids or slug in ids or slug.lower() in {str(x).lower() for x in ids}:
            return prim
    return None


def apply_physical_graph(delta: dict[str, Any], sink: _Sink) -> None:
    if delta.get("classification") != "LAW":
        return
    source, sha = _source_fields(delta)
    atlas = delta.get("architecture_atlas_behaviour_reference")
    atlas = atlas if isinstance(atlas, dict) else {}
    sem = delta.get("physical_graph_candidate_semantic")
    sem = sem if isinstance(sem, dict) else {}
    organ = str(sem.get("organ") or atlas.get("behaviour") or "UNKNOWN")
    behaviour = str(atlas.get("behaviour") or organ)

    cite_key = _key(sha, "physical_graph", f"atlas:{behaviour}:{_src_tag(source)}")
    if not sink.skip_or_begin("physical_graph", cite_key):
        prim = _atlas_primitive_for(behaviour)
        if prim is None:
            sink.applied(
                "physical_graph",
                cite_key,
                {
                    "kind": "atlas_citation",
                    "behaviour": behaviour,
                    "primitive": None,
                    "in_atlas": False,
                    "action": atlas.get("action") or "cite as behaviour evidence; do not rewrite the atlas",
                    "atlas_path": atlas.get("atlas_path"),
                    "source": source,
                    "source_sha256": sha,
                },
            )
        else:
            try:
                spec = phys.contract(prim)
                sink.applied(
                    "physical_graph",
                    cite_key,
                    {
                        "kind": "atlas_citation",
                        "behaviour": behaviour,
                        "primitive": prim,
                        "in_atlas": spec.get("in_atlas"),
                        "atlas_index": spec.get("atlas_index"),
                        "invariant": spec.get("invariant"),
                        "action": "cite via physical_primitives.contract; atlas not rewritten",
                        "source": source,
                        "source_sha256": sha,
                    },
                )
            except phys.PrimitiveError as e:
                sink.refused(
                    "physical_graph",
                    cite_key,
                    str(e),
                    extra={"source": source, "source_sha256": sha, "behaviour": behaviour, "stage": "contract"},
                )

    plan_key = _key(sha, "physical_graph", f"plan:{organ}:{_src_tag(source)}")
    if sink.skip_or_begin("physical_graph", plan_key):
        return
    qualification = str(sem.get("qualification") or "PLAN_ONLY")
    if qualification != "PLAN_ONLY":
        # Sidecar does not compile. Force PLAN_ONLY; if the delta claimed more, refuse that claim.
        sink.refused(
            "physical_graph",
            _key(sha, "physical_graph", f"qualification:{qualification}:{_src_tag(source)}"),
            f"delta claimed qualification {qualification!r}; sidecar emits PLAN_ONLY only",
            extra={"source": source, "source_sha256": sha, "claimed": qualification, "stage": "qualification"},
        )
        qualification = "PLAN_ONLY"
    graph = PhysicalGraph(
        model_id="unknown",
        computation=[
            {
                "id": organ,
                "kind": "computation",
                "present": False,
                "source": "codex_ingest_delta",
                "semantic_type": sem.get("semantic_type") or "PhysicalGraphPlan",
            }
        ],
        data=[
            {
                "id": organ,
                "kind": "tensor_group",
                "bytes": None,
                "active_bytes_per_token": None,
                "source": "ingest delta; size unresolved (STATIC_ONLY)",
            }
        ],
        qualification=qualification,
        generated_at=0.0,
        evidence=[
            {
                "kind": "ingest_delta",
                "source": source,
                "source_sha256": sha,
                "claim": "PLAN_ONLY candidate semantic; sidecar does not compile a graph",
            }
        ],
    )
    body = graph.to_dict()
    # Drop wall-clock; generated_at was forced to 0.
    sink.applied(
        "physical_graph",
        plan_key,
        {
            "kind": "plan_only",
            "organ": organ,
            "semantic_type": body.get("semantic_type"),
            "qualification": body.get("qualification"),
            "fingerprint": body.get("fingerprint"),
            "action": sem.get("action") or "consider as a candidate semantic; sidecar does not compile a graph",
            "source": source,
            "source_sha256": sha,
        },
    )


# ---------------------------------------------------------------------------
# WorkUnit species — propose through emit_hcli_workunit, never promote.
# ---------------------------------------------------------------------------


_SPECIES_CACHE: dict[str, dict[str, Any]] | None = None


def _species() -> dict[str, dict[str, Any]]:
    global _SPECIES_CACHE
    if _SPECIES_CACHE is None:
        _SPECIES_CACHE = {row["id"]: row for row in wus.catalog()}
    return _SPECIES_CACHE


def apply_workunit(delta: dict[str, Any], sink: _Sink) -> None:
    source, sha = _source_fields(delta)
    classification = delta.get("classification")
    species_id = (
        "odyssey_iii_adversarial_experiment"
        if classification == "SCAR"
        else "odyssey_ii_transfer_experiment"
    )
    spec = _species()[species_id]
    unit_id = f"future.propagate.{species_id}.{sha[:12]}.{_src_tag(source)}"
    key = _key(sha, "workunit_species", unit_id)
    if sink.skip_or_begin("workunit_species", key):
        return
    try:
        row = wus.emit_hcli_workunit(
            id=unit_id,
            role=spec["role"],
            description=_clip(
                f"Propagate {classification} {source} through {species_id}. "
                f"{(delta.get('odyssey_ii_law_candidate') or {}).get('statement_sketch') or ''}"
            ),
            dependencies=[],
            resource_class=spec["resource_class"],
            verifier=spec["verifier"],
            provider="future.propagate",
            effect_class=spec["effect_class"],
            status="pending",
            classification="STATIC_ONLY",
            extras={
                "species": species_id,
                "source": source,
                "source_sha256": sha,
                "evidence_parents": list(spec.get("evidence_parents") or []),
                "candidate_status": "STATIC_ONLY",
            },
        )
        wus.validate_emitted_unit(row)
    except (wus.SpeciesAuthorityError, wus.WorkUnitShapeError, TypeError, ValueError) as e:
        sink.refused(
            "workunit_species",
            key,
            str(e),
            extra={"source": source, "source_sha256": sha, "species": species_id, "stage": "emit_hcli_workunit"},
        )
        return
    sink.applied(
        "workunit_species",
        key,
        {
            "id": row.get("id"),
            "species": species_id,
            "role": row.get("role"),
            "status": row.get("status"),
            "verifier": row.get("verifier"),
            "resource_class": row.get("resource_class"),
            "effect_class": row.get("effect_class"),
            "may_promote": False,
            "source": source,
            "source_sha256": sha,
        },
    )


# ---------------------------------------------------------------------------
# Tournament — consult can_run. A delta cannot make the harness runnable.
# ---------------------------------------------------------------------------


_TOURNAMENT_CACHE: tuple[bool, list[str]] | None = None


def _tournament_readiness() -> tuple[bool, list[str]]:
    global _TOURNAMENT_CACHE
    if _TOURNAMENT_CACHE is None:
        _TOURNAMENT_CACHE = tn.can_run()
    return _TOURNAMENT_CACHE


def apply_tournament(delta: dict[str, Any], sink: _Sink) -> None:
    source, sha = _source_fields(delta)
    key = _key(sha, "tournament", f"readiness:{_src_tag(source)}")
    if sink.skip_or_begin("tournament", key):
        return
    ok, reasons = _tournament_readiness()
    sink.applied(
        "tournament",
        key,
        {
            "kind": "readiness_consultation",
            "can_run": bool(ok),
            "reasons": list(reasons),
            "source": source,
            "source_sha256": sha,
            "note": (
                "Ingest deltas do not complete either NX and do not grant a GPU lease. "
                "can_run is consulted, not flipped."
            ),
        },
    )


def _tournament_run_guard() -> dict[str, Any]:
    """Call run() once so the TournamentNotReady guard is watched firing."""
    try:
        tn.run()
        return {"raised": False, "reason": "run() returned; sidecar must not execute a tournament"}
    except tn.TournamentNotReady as e:
        return {"raised": True, "exception": type(e).__name__, "reasons": list(e.reasons)}


# ---------------------------------------------------------------------------
# Negative index — SCARs land here. Public Scar + query + refuse_if_dead.
# ---------------------------------------------------------------------------


def _scar_from_delta(delta: dict[str, Any]) -> ni.Scar:
    source, sha = _source_fields(delta)
    inv = delta.get("invalidation") if isinstance(delta.get("invalidation"), dict) else {}
    driver = _driver(delta)
    kills = [str(k) for k in (inv.get("kills") or []) if k]
    organ_text = str(inv.get("organ") or "")
    family_text = kills[0] if kills else str(driver.get("token") or source)
    verdict = str(driver.get("token") or "NEGATIVE")
    level = str(inv.get("level") or "MODEL_SPECIFIC")
    scar = ni.Scar(
        scar_id=f"{source}#{sha[:16]}",
        source_path=source,
        source_origin="codex_ingest_delta",
        parse_status=ni.PARSED,
        model=ni.UNRECORDED,
        models=[ni.UNRECORDED],
        organ=ni.canon_organ(organ_text) if organ_text else ni.UNRECORDED,
        organs=ni.extract_organs(organ_text) if organ_text else [ni.UNRECORDED],
        representation=ni.UNRECORDED,
        machine=ni.UNRECORDED,
        hypothesis_family=ni.canon_family(family_text),
        failure_mechanism=_clip(kills[0] if kills else driver.get("reason") or verdict),
        verdict=verdict,
        refuse_eligible=str(verdict).upper().replace(" ", "_") in REFUTING_SCAR_TOKENS,
        reopen_condition=str(inv.get("reopen_condition") or ni.UNRECORDED),
        claim_refuted=_clip("; ".join(kills) if kills else driver.get("reason") or verdict),
        level=level if level in {"MODEL_SPECIFIC", "FAMILY", "GENERAL_PHYSICAL"} else "MODEL_SPECIFIC",
        original_id=sha[:16],
    )
    return scar.finalize()


def apply_negative_index(delta: dict[str, Any], sink: _Sink, landed: list[ni.Scar]) -> ni.Scar | None:
    if delta.get("classification") != "SCAR":
        return None
    source, sha = _source_fields(delta)
    scar_id = f"{source}#{sha[:16]}"
    key = _key(sha, "negative_index", scar_id)
    if sink.skip_or_begin("negative_index", key):
        return None
    scar = _scar_from_delta(delta)
    landed.append(scar)
    sink.applied(
        "negative_index",
        key,
        {
            "scar_id": scar.scar_id,
            "source_path": scar.source_path,
            "parse_status": scar.parse_status,
            "hypothesis_family": scar.hypothesis_family,
            "organ": scar.organ,
            "verdict": scar.verdict,
            "refuse_eligible": scar.refuse_eligible,
            "reopen_condition": scar.reopen_condition,
            "level": scar.level,
            "source_sha256": sha,
        },
    )
    return scar


# ---------------------------------------------------------------------------
# SCAR invalidation pass — report which already-applied records a scar hits.
# ---------------------------------------------------------------------------


def _token_refutes(delta: dict[str, Any]) -> bool:
    tok = str(_driver(delta).get("token") or "").upper().replace(" ", "_")
    field = str(_driver(delta).get("field") or "")
    if tok in REFUTING_SCAR_TOKENS:
        return True
    if field == "NOT_FOR_PROMOTION":
        return True
    return False


def _record_overlap(record: dict[str, Any], scar: ni.Scar, inv: dict[str, Any]) -> bool:
    """True when a downstream record is in this SCAR's MODEL_SPECIFIC blast radius.

    UNKNOWN/unrecorded organs do not globally prune. Same source always matches.
    """
    rec_source = str(record.get("source") or record.get("source_path") or record.get("source_receipt") or "")
    if rec_source and rec_source == scar.source_path:
        return True
    scar_organ = scar.organ
    if scar_organ and scar_organ not in {ni.UNRECORDED, "UNKNOWN", ""}:
        for field in ("organ_class", "organ", "organ_fingerprint"):
            val = record.get(field)
            if not val:
                continue
            if ni.canon_organ(str(val)) == scar_organ or str(val) == inv.get("organ"):
                return True
    kills = [str(k).lower() for k in (inv.get("kills") or []) if k]
    blob = " ".join(
        str(record.get(k) or "")
        for k in ("law_id", "statement", "row_id", "id", "selected_attack_id")
    ).lower()
    for kill in kills:
        if len(kill) >= 24 and kill[:80].lower() in blob:
            return True
    return False


def invalidate_dependents(
    *,
    scar_deltas: list[dict[str, Any]],
    sink: _Sink,
    landed: list[ni.Scar],
    runtime: dict[str, dict[str, Any]],
    apply: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    o2_recs = list(sink.buckets["odyssey2_law_store"]["applied_records"])
    o3_recs = list(sink.buckets["odyssey3_adversary"]["applied_records"])
    lpc_recs = list(sink.buckets["lpc_dataset"]["applied_records"])
    pool = ni.ingest() + landed

    for delta in scar_deltas:
        source, sha = _source_fields(delta)
        inv = delta.get("invalidation") if isinstance(delta.get("invalidation"), dict) else {}
        scar = next((s for s in landed if s.source_path == source and s.original_id == sha[:16]), None)
        if scar is None:
            try:
                scar = _scar_from_delta(delta)
            except (TypeError, ValueError):
                continue
        hits: list[dict[str, Any]] = []
        for rec in o2_recs:
            if rec.get("kind") == "scar_invalidation":
                continue
            if rec.get("source_sha256") == sha:
                continue
            if _record_overlap(rec, scar, inv):
                hits.append(
                    {
                        "consumer": "odyssey2_law_store",
                        "record_id": rec.get("law_id"),
                        "scope_before": rec.get("scope"),
                        "scope_after": rec.get("scope"),
                        "downgrade": "candidate_marked_scar_invalidated; not a verified-law demotion",
                    }
                )
                rec["scar_invalidated_by"] = sorted(
                    set(list(rec.get("scar_invalidated_by") or []) + [scar.scar_id])
                )
        for rec in o3_recs:
            if rec.get("source_sha256") == sha:
                continue
            if _record_overlap(rec, scar, inv):
                law_id = rec.get("law_id")
                rt = runtime.get(str(law_id) or "")
                scope_after = rec.get("scope")
                moved = False
                if apply and rt and _token_refutes(delta) and rt.get("selected"):
                    try:
                        update = o3.apply_result(
                            rt["law"],
                            rt["selected"],
                            {
                                "verdict": "REFUTED",
                                "synthetic": True,
                                "reason": (
                                    f"heuristic SCAR {source} token={_driver(delta).get('token')} "
                                    f"sha256={sha}; STATIC_ONLY, not a physical experiment"
                                ),
                                "evidence_class": "STATIC_ONLY",
                                "bench_state": "UNKNOWN",
                            },
                        )
                        scope_after = update.get("scope_after")
                        moved = bool(update.get("moved"))
                        rec["scope_after_scar"] = scope_after
                        rec["scar_scope_update"] = {
                            "moved": moved,
                            "direction": update.get("direction"),
                            "scope_before": update.get("scope_before"),
                            "scope_after": scope_after,
                            "synthetic": True,
                        }
                    except (o3.ScopeUnmovedError, o3.LawSchemaError, ValueError) as e:
                        rec["scar_apply_result_error"] = str(e)
                hits.append(
                    {
                        "consumer": "odyssey3_adversary",
                        "record_id": law_id,
                        "scope_before": rec.get("scope"),
                        "scope_after": scope_after,
                        "moved": moved,
                        "downgrade": "apply_result on heuristic SCAR"
                        if moved
                        else "reported overlap; scope unmoved (INCONCLUSIVE or no matching attack)",
                    }
                )
        for rec in lpc_recs:
            if rec.get("source_sha256") == sha:
                continue
            if _record_overlap(rec, scar, inv):
                rec["scar_invalidated_by"] = sorted(
                    set(list(rec.get("scar_invalidated_by") or []) + [scar.scar_id])
                )
                hits.append(
                    {
                        "consumer": "lpc_dataset",
                        "record_id": rec.get("row_id"),
                        "downgrade": "row remains VALID/incomplete; not completable from a SCAR",
                    }
                )

        proposal = {
            "hypothesis_family": scar.hypothesis_family,
            "organ": scar.organ if scar.organ != ni.UNRECORDED else None,
            "model": None,
        }
        refusal = ni.refuse_if_dead(proposal, scars=pool)
        reports.append(
            {
                "scar_id": scar.scar_id,
                "source": source,
                "source_sha256": sha,
                "n_downstream_hits": len(hits),
                "hits": hits,
                "refuse_if_dead": refusal,
                "reopen_condition": inv.get("reopen_condition"),
                "kills": list(inv.get("kills") or []),
                "makes_redundant": list(inv.get("makes_redundant") or []),
            }
        )
    reports.sort(key=lambda r: (r.get("source") or "", r.get("scar_id") or ""))
    return reports


# ---------------------------------------------------------------------------
# Recover / gaps / findings
# ---------------------------------------------------------------------------


def recovered_implementation() -> dict[str, Any]:
    rows = []
    for path, note in RECOVERY_PROBES:
        row = _probe_path(path)
        row["note"] = note
        rows.append(row)
    return {
        "already_existed": rows,
        "present": [r["path"] for r in rows if r["on_disk"] or r["in_git"]],
        "absent_from_head_and_disk": [r["path"] for r in rows if not r["on_disk"] and not r["in_git"]],
        "adequate_existing_propagator": None,
        "why_not_redundant": (
            "codex_ingest.py emits deltas and names consumers; each consumer "
            "validates its own records; nothing routed a delta through those APIs. "
            "F015's compounding loop was open. This module is the missing apply step."
        ),
        "consumer_apis_used": {
            "odyssey2_law_store": "Law / validate_law / promote (ScopeViolation is the correct outcome)",
            "odyssey3_adversary": "emit_for_law / apply_result",
            "lpc_dataset": "row_template / validate_row / contamination_from_benchmark_class / forbid_zero_imputation (tests)",
            "hwir": "from_organ_map / HwirGraph / validate",
            "workunit_species": "catalog / emit_hcli_workunit / validate_emitted_unit",
            "tournament": "can_run / run (TournamentNotReady)",
            "physical_graph": "hcli.physical_graph.PhysicalGraph (read-only) + physical_primitives.contract",
            "negative_index": "Scar / query / refuse_if_dead",
        },
    }


def gaps_closed() -> list[str]:
    return [
        "propagate(deltas, *, apply=True) routes each LAW/SCAR delta through the named consumers' public APIs",
        "LAW candidates admitted only at MODEL_LOCAL / ANECDOTE; proposed_scope above that is promote()'d and refused",
        "driver.confidence cannot raise evidence_strength or transfer_confidence",
        "idempotence keyed on source_sha256|consumer|record_id; a second apply adds zero records",
        "per-consumer applied / refused (consumer's own reason) / skipped-as-duplicate accounting",
        "SCAR lands in the negative-science index and reports downstream invalidations",
        "--dry-run walks the same routing without persisting a durable ledger",
        "writes only receipts/future/PROPAGATION_STATE.json; never another module's receipt",
    ]


def negative_findings(recovered: dict[str, Any], result: dict[str, Any]) -> list[str]:
    findings = [f"absent: {p}" for p in recovered.get("absent_from_head_and_disk") or []]
    findings.extend(
        [
            "ingest proposed_scope 'FAMILY' is not on the Odyssey II lattice (MODEL_LOCAL < ARCHITECTURE_FAMILY < ...); those claims refuse at promote()",
            "LAW deltas do not name workunit_species or tournament; they are still routed as the contract's seven consumers",
            "architecture_atlas_behaviour_reference says cite, do not rewrite the atlas — this module never writes ACCELERATOR_ARCHITECTURE_ATLAS.json",
            "hcli/physical_graph.py is imported read-only; compile_physical_graph is not invoked (no architecture recognizer report on a delta)",
            "tournament can_run remains false; ingest deltas do not complete FLASH_SINGULARITY.NX or QWEN27_SINGULARITY.NX",
            "this sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE; every propagated record is STATIC_ONLY / bench UNKNOWN",
            "hardware numbers in source receipts are not copied; LPC latency/active_bytes stay null with HARDWARE_AUTHORITY_REQUIRED",
            "consumer receipts (ODYSSEY2_LAW_STORE.json, LPC_DATASET.json, …) are not rewritten — validation is in-process, compounding state is PROPAGATION_STATE.json",
            "a SCAR invalidation of an Odyssey III law uses apply_result with synthetic=True; that is a reported heuristic downgrade, not a protected experiment",
        ]
    )
    o2 = (result.get("consumers") or {}).get("odyssey2_law_store") or {}
    if o2.get("refused"):
        findings.append(
            f"odyssey2_law_store refused {o2['refused']} promotion/validation attempt(s); "
            "high refusal is a finding about delta quality, not a reason to loosen the lattice"
        )
    return findings


# ---------------------------------------------------------------------------
# propagate()
# ---------------------------------------------------------------------------


def propagate(
    deltas: list[dict[str, Any]] | None = None,
    *,
    apply: bool = True,
    previous: dict[str, Any] | None = None,
    load_previous: bool = True,
) -> dict[str, Any]:
    """Route each delta through the named consumers' public APIs.

    `apply=False` is a dry-run: the same routing and validation run, but the
    durable ledger is not extended. Consumer module receipts are never written
    in either mode.
    """
    if deltas is None:
        deltas = load_ingest_deltas()
    if previous is None and load_previous and apply:
        previous = load_previous_apply()

    ordered = sorted(
        [d for d in deltas if isinstance(d, dict)],
        key=lambda d: (str(d.get("source") or ""), str(d.get("source_sha256") or "")),
    )
    sink = _Sink(_ledger_keys(previous))
    runtime: dict[str, dict[str, Any]] = {}
    landed: list[ni.Scar] = []
    evidence_resolution = {"pinned_snapshot": 0, "live_headless": 0, "unresolved": 0}
    by_class = {"LAW": 0, "SCAR": 0, "NEUTRAL": 0, "OTHER": 0}

    for delta in ordered:
        label = str(delta.get("classification") or "OTHER")
        if label in by_class:
            by_class[label] += 1
        else:
            by_class["OTHER"] += 1
        source, _sha = _source_fields(delta)
        if source:
            ev = resolve_evidence(source)
            src = ev.get("source") or "unresolved"
            evidence_resolution[src] = evidence_resolution.get(src, 0) + 1

        if label == "NEUTRAL":
            continue
        if label == "LAW":
            o2_law = apply_odyssey2(delta, sink)
            apply_odyssey3(delta, sink, o2_law=o2_law, runtime=runtime)
            apply_lpc(delta, sink)
            apply_hwir(delta, sink)
            apply_physical_graph(delta, sink)
            apply_workunit(delta, sink)
            apply_tournament(delta, sink)
            continue
        if label == "SCAR":
            apply_odyssey2(delta, sink)
            apply_odyssey3(delta, sink, o2_law=None, runtime=runtime)
            apply_lpc(delta, sink)
            apply_negative_index(delta, sink, landed)
            apply_workunit(delta, sink)
            apply_tournament(delta, sink)
            continue

    scar_deltas = [d for d in ordered if d.get("classification") == "SCAR"]
    invalidations = invalidate_dependents(
        scar_deltas=scar_deltas,
        sink=sink,
        landed=landed,
        runtime=runtime,
        apply=apply,
    )

    tournament_guard = _tournament_run_guard()
    recovered = recovered_implementation()

    totals = {
        "applied": 0,
        "refused": 0,
        "skipped_as_duplicate": 0,
    }
    consumers_out: dict[str, Any] = {}
    for name in ALL_CONSUMERS:
        b = sink.buckets[name]
        totals["applied"] += b["applied"]
        totals["refused"] += b["refused"]
        totals["skipped_as_duplicate"] += b["skipped_as_duplicate"]
        consumers_out[name] = {
            "applied": b["applied"],
            "refused": b["refused"],
            "skipped_as_duplicate": b["skipped_as_duplicate"],
            "applied_records": b["applied_records"],
            "refusals": b["refusals"],
        }

    prev_keys = sorted(_ledger_keys(previous))
    if apply:
        applied_keys = sorted(sink.seen)
    else:
        applied_keys = prev_keys

    primary = "pinned_snapshot"
    if evidence_resolution.get("pinned_snapshot", 0) == 0 and evidence_resolution.get("live_headless", 0) > 0:
        primary = "live_headless"
    ingest_doc = load_ingest_state()
    ingest_root = ingest_doc.get("root") or "receipts/headless"

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Apply CODEX_INGEST_STATE active_deltas through the seven consumers' "
            "public APIs so Codex movement compounds in the sidecar stores. "
            "Produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dry_run": not apply,
        "n_deltas": len(ordered),
        "by_classification": by_class,
        "consumers": consumers_out,
        "totals": totals,
        "scar_invalidations": invalidations,
        "n_scar_invalidations": len(invalidations),
        "n_scar_downstream_hits": sum(int(r.get("n_downstream_hits") or 0) for r in invalidations),
        "tournament_run_guard": tournament_guard,
        "ledger": {
            "key_format": "source_sha256|consumer|record_id (record_id includes a source-path tag so identical bytes at different paths are distinct records)",
            "applied_keys": applied_keys,
            "n_keys": len(applied_keys),
            "previous_n_keys": len(prev_keys),
            "new_keys_this_run": 0 if not apply else len(sink.new_keys),
        },
        "evidence_source": primary,
        "evidence_resolution": {
            "ingest_state": f"receipts/future/{INGEST_RECEIPT}",
            "ingest_root": ingest_root,
            "policy": "pinned_snapshot preferred, live_headless fallback; delta body is self-contained",
            "counts": evidence_resolution,
        },
        "admission_policy": {
            "odyssey2_scope": ADMISSION_SCOPE,
            "odyssey2_evidence_strength": WEAKEST_O2_STRENGTH,
            "odyssey3_scope": ADMISSION_SCOPE,
            "transfer_confidence": WEAKEST_TRANSFER_CONFIDENCE,
            "lpc_contamination_class": "STATIC_ONLY",
            "physical_graph_qualification": "PLAN_ONLY",
            "hwir_qualification": "STATIC_ONLY",
            "rule": (
                "every propagated record is admitted at the weakest evidence the "
                "consumer offers, with source path and sha256 attached; a heuristic "
                "cannot mint GENERIC_VERIFIED, PROTECTED_ABSOLUTE, or a complete LPC row"
            ),
        },
        "seven_consumers": list(SEVEN_CONSUMERS),
        "vocabulary": {
            "eras": ["I", "II", "III", "IV", "V"],
            "odysseys": [
                "I WHAT IS TRUE?",
                "II WHAT DID HAWKING ALREADY LEARN?",
                "III WHERE IS HAWKING WRONG?",
            ],
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is": "part of Accelerator / Physical Compiler / Fusion, not its own civilization",
            "evidence_class_we_emit": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        },
        "recovered_implementation": recovered,
        "gaps_closed": gaps_closed(),
        "negative_findings": [],  # filled below so findings can cite totals
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. Does not produce "
            "DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE. Does not rewrite consumer "
            "module receipts; compounding state lives here. Does not compile a "
            "PhysicalGraph or run the tournament."
        ),
    }
    doc["negative_findings"] = negative_findings(recovered, doc)
    return doc


def _print_summary(doc: dict[str, Any]) -> None:
    mode = "dry-run" if doc.get("dry_run") else "apply"
    byc = doc.get("by_classification") or {}
    print(
        f"{mode}: n_deltas={doc.get('n_deltas')} "
        f"LAW={byc.get('LAW', 0)} SCAR={byc.get('SCAR', 0)} NEUTRAL={byc.get('NEUTRAL', 0)}"
    )
    consumers = doc.get("consumers") or {}
    for name in ALL_CONSUMERS:
        b = consumers.get(name) or {}
        print(
            f"  {name}: applied={b.get('applied', 0)} "
            f"refused={b.get('refused', 0)} skipped_as_duplicate={b.get('skipped_as_duplicate', 0)}"
        )
    tot = doc.get("totals") or {}
    print(
        f"  totals: applied={tot.get('applied', 0)} "
        f"refused={tot.get('refused', 0)} skipped_as_duplicate={tot.get('skipped_as_duplicate', 0)}"
    )
    print(
        f"  scar_invalidations={doc.get('n_scar_invalidations', 0)} "
        f"downstream_hits={doc.get('n_scar_downstream_hits', 0)}"
    )
    guard = doc.get("tournament_run_guard") or {}
    print(f"  tournament_run_guard.raised={guard.get('raised')}")
    print(f"  evidence_source={doc.get('evidence_source')}")


def build(*, apply: bool = True, deltas: list[dict[str, Any]] | None = None) -> Path:
    doc = propagate(deltas, apply=apply)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build(apply=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="route and validate; do not persist the ledger")
    ap.add_argument("--apply", action="store_true", help="route, validate, and persist the ledger")
    ap.add_argument("--selftest", action="store_true", help="apply against current ingest state")
    a = ap.parse_args(argv)
    apply = bool(a.apply or a.selftest) and not a.dry_run
    out = build(apply=apply)
    print(out)
    _print_summary(load_json(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
