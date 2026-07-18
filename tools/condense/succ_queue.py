#!/usr/bin/env python3.12
"""Durable successor queue: status vocabulary, sealed rows, load/validate/persist/reload.

Implements the master-goal queue (section 6.5 status vocabulary + section 11.2 row schema).
A queue row is a sealed descriptor of a parent the successor will condense. "In queue" means
the controller loads, validates, persists, displays, and advances the row (section 3), not a
name in Markdown. Rows carry HONEST, capability-derived blockers (never a decorative status).

The store is a single sealed JSON file under the successor namespace. Every mutation is an
atomic write; reload re-validates every row seal. Row status transitions are constrained to
the queue vocabulary and each requires a reason + blocker set + next transition.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, now_iso, atomic_write_json, read_json_safe, eco_state_root,
)

QUEUE_SCHEMA = "hawking.successor.queue.v1"
ROW_SCHEMA = "hawking.successor.queue_row.v1"

STATUS_VOCAB: tuple[str, ...] = (
    "waiting_old_release", "waiting_source_authority", "waiting_source", "waiting_adapter",
    "waiting_tokenizer", "waiting_runtime", "waiting_quality_path", "waiting_disk",
    "waiting_resource", "waiting_predecessor_evidence", "planned", "compiled", "validated",
    "ready", "admitted", "running", "checkpointing", "provisional", "sealed",
    "replication_required", "evidence_closed", "retired", "failed", "invalid", "drained",
    "superseded",
)

# Terminal-ish for scheduling purposes (row will not advance without external change).
BLOCKED_STATUSES = frozenset(s for s in STATUS_VOCAB if s.startswith("waiting_"))


class QueueError(EcoError):
    """Fail-closed queue error."""


def make_row(**fields: Any) -> dict[str, Any]:
    """Build a sealed queue row. Enforces the status vocabulary and required fields."""
    status = fields.get("current_status")
    if status not in STATUS_VOCAB:
        raise QueueError(f"invalid current_status {status!r}")
    row = {
        "schema": ROW_SCHEMA,
        "queue_generation": fields.get("queue_generation", "gen-1"),
        "parent_label": fields["parent_label"],
        "hf_or_source_id": fields.get("hf_or_source_id"),
        "exact_revision": fields.get("exact_revision"),
        "architecture_family": fields.get("architecture_family"),
        "config_sha256": fields.get("config_sha256"),
        "tokenizer_sha256": fields.get("tokenizer_sha256"),
        "chat_template_sha256": fields.get("chat_template_sha256"),
        "source_manifest_sha256": fields.get("source_manifest_sha256"),
        "exact_stored_parameter_count": fields.get("exact_stored_parameter_count"),
        "active_parameter_count": fields.get("active_parameter_count"),
        "source_bytes": fields.get("source_bytes"),
        "current_local_bytes": fields.get("current_local_bytes", 0),
        "expected_output_rate_prior": fields.get("expected_output_rate_prior"),
        "prior_is_evidence": False,
        "candidate_representation_families": fields.get("candidate_representation_families", []),
        "candidate_doctor_families": fields.get("candidate_doctor_families", []),
        "required_predecessor_evidence": fields.get("required_predecessor_evidence", []),
        "adapter_id": fields.get("adapter_id"),
        "adapter_capabilities_sha256": fields.get("adapter_capabilities_sha256"),
        "runtime_spec_status": fields.get("runtime_spec_status", "unknown"),
        "quality_path_status": fields.get("quality_path_status", "unknown"),
        "streamed_lifecycle": fields.get("streamed_lifecycle", "unknown"),
        "disk_envelope": fields.get("disk_envelope"),
        "ram_envelope": fields.get("ram_envelope"),
        "scratch_envelope": fields.get("scratch_envelope"),
        "network_envelope": fields.get("network_envelope"),
        "resume_strategy": fields.get("resume_strategy", "checkpoint-exact"),
        "current_status": status,
        "blockers": fields.get("blockers", []),
        "exit_criteria": fields.get("exit_criteria", []),
        "next_transition": fields.get("next_transition"),
        "created_at": now_iso(),
    }
    # Additive Gravity augmentation (master goal section 16). Only present when supplied, so a
    # row built without Gravity is byte-identical to before and its seal is unchanged.
    if fields.get("gravity") is not None:
        row["gravity"] = fields["gravity"]
    return seal_field(row, "row_sha256")


class Queue:
    def __init__(self, root: str | os.PathLike[str] | None = None):
        self.root = Path(root) if root else eco_state_root().parent / "event_horizon_successor" / "queue"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "queue.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": QUEUE_SCHEMA, "rows": {}}
        doc = read_json_safe(self.path)
        if doc.get("schema") != QUEUE_SCHEMA:
            raise QueueError(f"queue schema mismatch: {doc.get('schema')!r}")
        # verify the container seal so whole-row insertion/removal is caught, not just
        # per-row edits (the container seal binds the exact row set).
        if not sealed(doc, "queue_sha256"):
            raise QueueError("queue container self-seal invalid on load (row set tampered)")
        for label, row in doc.get("rows", {}).items():
            if not sealed(row, "row_sha256"):
                raise QueueError(f"queue row {label} self-seal invalid on load")
        return doc

    def persist(self, doc: dict[str, Any]) -> None:
        doc = {"schema": QUEUE_SCHEMA, "rows": doc.get("rows", {}), "persisted_at": now_iso()}
        atomic_write_json(self.path, seal_field(doc, "queue_sha256"))

    def upsert(self, row: dict[str, Any]) -> dict[str, Any]:
        if not sealed(row, "row_sha256"):
            raise QueueError("cannot upsert an unsealed row")
        doc = self.load()
        doc.setdefault("rows", {})[row["parent_label"]] = row
        self.persist(doc)
        return row

    def update_status(self, label: str, status: str, *, reason: str, blockers: list[str],
                      next_transition: str | None) -> dict[str, Any]:
        if status not in STATUS_VOCAB:
            raise QueueError(f"invalid status {status!r}")
        doc = self.load()
        row = doc.get("rows", {}).get(label)
        if row is None:
            raise QueueError(f"no such row {label}")
        updated = {k: v for k, v in row.items() if k != "row_sha256"}
        updated.update({"current_status": status, "blockers": blockers,
                        "next_transition": next_transition,
                        "status_reason": reason, "updated_at": now_iso()})
        updated = seal_field(updated, "row_sha256")
        doc["rows"][label] = updated
        self.persist(doc)
        return updated

    def rows(self) -> list[dict[str, Any]]:
        return list(self.load().get("rows", {}).values())

    def summary(self) -> dict[str, Any]:
        rows = self.rows()
        return {
            "count": len(rows),
            "by_status": {r["parent_label"]: r["current_status"] for r in rows},
            "blocked": [r["parent_label"] for r in rows if r["current_status"] in BLOCKED_STATUSES],
        }


def build_default_rows(admissions: dict[str, dict[str, Any]] | None = None,
                       generation: str = "gen-1") -> list[dict[str, Any]]:
    """Build the mandated 72B, 120B, and 600B-1.1T rows with honest blockers.

    `admissions` maps a family key -> a succ_admission.admit() record; when absent, the rows
    carry conservative capability-derived blockers from the E0 audit.
    """
    admissions = admissions or {}
    from studio_manifest import frontier_by_label

    def adm_blockers(family: str, fallback: list[str]) -> tuple[bool, list[str], str | None, str | None]:
        rec = admissions.get(family)
        if rec is None:
            return False, fallback, None, None
        return (rec.get("ready_for_execution", False), rec.get("blockers", fallback),
                rec.get("adapter_id"), rec.get("adapter_capabilities_sha256"))

    rows: list[dict[str, Any]] = []

    # 72B: qwen2.5-dense, execution-ready adapter but legacy cell still running; post-release
    # calibration is precompiled. Status waits on the legacy release boundary.
    q_ready, q_block, q_adapter, q_caps = adm_blockers(
        "qwen2.5-dense", ["adapter_ready_but_quality_eval_disk_gated_for_72B"])
    rows.append(make_row(
        queue_generation=generation, parent_label="72B",
        hf_or_source_id="Qwen/Qwen2.5-72B-Instruct", architecture_family="qwen2.5-dense",
        exact_stored_parameter_count=None, active_parameter_count=None,
        candidate_representation_families=["strand_ladder"], candidate_doctor_families=["none"],
        adapter_id=q_adapter or "doctor-v5-strand-ladder-qwen25-dense",
        adapter_capabilities_sha256=q_caps,
        runtime_spec_status="executable", quality_path_status="resident_eval_disk_gated",
        streamed_lifecycle="valid", resume_strategy="checkpoint-exact",
        current_status="waiting_old_release",
        blockers=["legacy_72B_cell_running", "successor_activation_awaits_signed_release"],
        exit_criteria=["legacy release boundary sealed", "72B terminal evidence imported"],
        next_transition="import_legacy_72B_then_calibrate"))

    # 120B: gpt-oss-moe adapter is a fail-closed 0.1-contract; honest waiting_* blockers.
    g_ready, g_block, g_adapter, g_caps = adm_blockers(
        "gpt-oss-moe", ["source_to_str2_conversion", "apple_silicon_moe_str2_loader",
                        "tokenizer_template", "evaluator", "native_load_parity",
                        "disk_infeasible_183GB", "human_review_signoff"])
    rows.append(make_row(
        queue_generation=generation, parent_label="120B",
        hf_or_source_id="openai/gpt-oss-120b", architecture_family="gpt-oss-moe",
        exact_stored_parameter_count=None, active_parameter_count=None,
        candidate_representation_families=["strand_ladder_moe"],
        candidate_doctor_families=["expert_genome_codec", "none"],
        required_predecessor_evidence=["72B_scheduling_priors"],
        adapter_id=g_adapter or "doctor-v5-strand-ladder-gpt-oss-moe",
        adapter_capabilities_sha256=g_caps,
        runtime_spec_status="missing", quality_path_status="false", streamed_lifecycle="infeasible",
        current_status="waiting_adapter",
        blockers=g_block,
        exit_criteria=["adapter capability report reviewed=true and run no longer refuses",
                       "tokenizer+template bound", "evaluator executable", "disk feasible"],
        next_transition="hold_until_120B_adapter_ready"))

    # 600B-1.1T: DeepSeek-V3 671B, canonical capstone. No adapter, source not staged, disk-walled.
    m671 = frontier_by_label("671B")
    rows.append(make_row(
        queue_generation=generation, parent_label="671B",
        hf_or_source_id="deepseek-ai/DeepSeek-V3", exact_revision=None,
        architecture_family="deepseek-moe",
        exact_stored_parameter_count=int((m671.total_b if m671 else 671.0) * 1e9),
        active_parameter_count=int((m671.active_b if m671 else 37.0) * 1e9),
        source_bytes=int((m671.download_gb if m671 else 1342.0) * 1e9),
        current_local_bytes=0, expected_output_rate_prior=(m671.serve_bpw if m671 else 1.0),
        candidate_representation_families=["strand_ladder_moe"],
        candidate_doctor_families=["expert_genome_codec", "capability_immune_bank"],
        required_predecessor_evidence=["120B_scheduling_priors"],
        adapter_id=None, runtime_spec_status="missing", quality_path_status="missing",
        streamed_lifecycle="bounded_stream_designed_not_wired",
        disk_envelope="1342GB source >> 176GB free (disk-walled)", ram_envelope="MOE-PAGED",
        current_status="waiting_source_authority",
        blockers=["waiting_source_authority(exact_revision_unbound)",
                  "waiting_adapter(deepseek-moe_not_built)",
                  "waiting_disk(1342GB_source_vs_176GB_free)"],
        exit_criteria=["exact DeepSeek-V3 revision bound", "deepseek-moe adapter built + reviewed",
                       "bounded-stream lifecycle wired", "disk plan for source windows"],
        next_transition="resolve_source_authority_and_build_adapter"))

    return rows


def selftest() -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        q = Queue(Path(d) / "queue")
        for row in build_default_rows():
            q.upsert(row)
        loaded = Queue(Path(d) / "queue").load()  # reload from disk, re-validate seals
        labels = set(loaded["rows"])
        if labels != {"72B", "120B", "671B"}:
            raise QueueError(f"expected 72B/120B/671B rows, got {labels}")
        summ = Queue(Path(d) / "queue").summary()
        if summ["by_status"]["120B"] != "waiting_adapter":
            raise QueueError("120B should be waiting_adapter")
        if summ["by_status"]["671B"] != "waiting_source_authority":
            raise QueueError("671B should be waiting_source_authority")
        # invalid status refused
        bad = False
        try:
            make_row(parent_label="x", current_status="totally_done")
        except QueueError:
            bad = True
        if not bad:
            raise QueueError("invalid status not refused")
        # status update persists + re-seals
        q.update_status("72B", "waiting_predecessor_evidence", reason="release pending",
                        blockers=["legacy_running"], next_transition="import_legacy")
        if Queue(Path(d) / "queue").summary()["by_status"]["72B"] != "waiting_predecessor_evidence":
            raise QueueError("status update not persisted")
        # tamper detection on reload
        doc = json.loads((Path(d) / "queue" / "queue.json").read_text())
        doc["rows"]["671B"]["source_bytes"] = 1
        (Path(d) / "queue" / "queue.json").write_text(json.dumps(doc))
        tampered = False
        try:
            Queue(Path(d) / "queue").load()
        except QueueError:
            tampered = True
        if not tampered:
            raise QueueError("row tamper not detected on reload")
    return {"ok": True, "rows": ["72B", "120B", "671B"], "vocab_enforced": True,
            "reload_reseal_validated": True, "tamper_detected": True}


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
