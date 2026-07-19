#!/usr/bin/env python3.12
"""Durable, singleton controller for the Second Light GPT-OSS-120B PQ Gravity campaign.

This is the persistent program driver. It works the sealed 183-row program
(`GPT_OSS_120B_PQ_GRAVITY_PROGRAM.json`) one row at a time under exactly ONE controller
(fcntl singleton lease), with resumable per-row checkpoints, self-sealed row evidence,
heartbeats, and exact-budget enforcement (Section 15). It exists independently of any chat
session: a real detached run survives its parent, and a stale PID or a committed JSON can
never make status report RUNNING (the flock is the only liveness truth).

Durable primitives are REUSED, not rebuilt:
  * succ_watchdog.SingletonLease  - fcntl.flock exclusive singleton controller lease.
  * succ_watchdog.heartbeat       - atomic self-sealed heartbeat marker.
  * succ_watchdog.sample_resources- read-only swap/thermal/disk/RSS snapshot.
  * succ_events.EventLog          - append-only hash-chained event log (events.jsonl).
  * eco_common.atomic_write_json / seal_field / sealed / hash_value / now_iso.
  * gravity_forge.pack_transform_pq / pack_shared_grammar (STABLE pack functions only).
  * gptoss_moe_runtime.ProvenanceReader / load_router / load_expert (real 120B source).

Row work by representation_family:
  * kept_original       - accounting passthrough (bytes == exact_budget, sealed metrics).
  * shared_expert_grammar - load the layer's experts (BOUNDED by max_experts), pack a shared
                          additive grammar, verify physical bytes fit the (scaled) budget.
  * transform_pq        - load attn/embedding tensor, randomized-Hadamard + product quantize;
                          huge embeddings pack a bounded row-slice and extrapolate the budget
                          (clearly labelled bounded_slice=true).

Every processed row writes a SEALED per-row checkpoint under checkpoints/<row_id>.json
(atomic, self-hashed, program-bound) BEFORE the controller cursor advances. The per-row
checkpoint is the single source of truth for "a row is done"; the cursor and the event log
are reconciled to it on resume, so no sealed row is ever redone and no partial output is
accepted. A crash-injection hook (env HAWKING_SL_KILL_AT) raises SystemExit at five exact
points so all five kill/resume scenarios (Section 21) are testable.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, atomic_write_json, hash_value, now_iso, read_json_safe, repo_root,
    seal_field, sealed,
)
from succ_events import EventLog  # noqa: E402
from succ_watchdog import SingletonLease, WatchdogError, heartbeat, sample_resources  # noqa: E402

CONTROLLER_SCHEMA = "hawking.second_light.controller.v1"
ROW_CHECKPOINT_SCHEMA = "hawking.second_light.row_checkpoint.v1"
LEASE_LABEL = "com.hawking.second_light"

# Terminal per-row statuses. A row carrying either of these is DONE and never recomputed.
STATUS_SEALED = "SEALED"
STATUS_FAILED_OVER_BUDGET = "FAILED_OVER_BUDGET"
TERMINAL_STATUSES = (STATUS_SEALED, STATUS_FAILED_OVER_BUDGET)

# The five exact crash-injection points, in per-row execution order.
KILL_POINTS = (
    "fitting", "packing", "eval", "after_write_before_receipt", "after_receipt_before_transition",
)

DEFAULT_CAMPAIGN_ROOT = repo_root() / "reports" / "condense" / "second_light"
DEFAULT_PROGRAM_NAME = "GPT_OSS_120B_PQ_GRAVITY_PROGRAM.json"
DEFAULT_MANIFEST = "reports/condense/subbit_frontier/GRAVITY_120B_PROVENANCE.json"

HEARTBEAT_INTERVAL_SECONDS = 30
# When a 2D tensor has more rows than this, transform_pq packs a bounded representative
# slice and extrapolates the whole-tensor budget from its measured bits-per-weight.
DEFAULT_PQ_MAX_MATRIX_ROWS = 65536


class ControllerError(EcoError):
    """Fail-closed error in the Second Light controller."""


@dataclasses.dataclass
class ControllerConfig:
    campaign_root: Path
    program_path: Path
    manifest_path: str = DEFAULT_MANIFEST
    max_experts: int = 8
    pq_max_matrix_rows: int = DEFAULT_PQ_MAX_MATRIX_ROWS
    only_rows: tuple[str, ...] | None = None
    # A bounded row (fewer experts than full, or an embedding row-slice) seals as a real
    # checkpoint so the scaffold and crash/resume tests work. Opt in to redo_bounded for a
    # real full-fidelity run so bounded seals are recomputed rather than silently adopted.
    redo_bounded: bool = False
    kill_at: str | None = None
    kill_row: str | None = None
    # test injection: {row_id: factor}. The effective budget for that row is scaled by
    # factor, so factor < 1 forces a real over-budget comparison to fail (Section 15 proof).
    shrink_budget: dict[str, float] = dataclasses.field(default_factory=dict)
    heartbeat_interval: int = HEARTBEAT_INTERVAL_SECONDS


class SecondLightController:
    """One durable controller working the sealed PQ Gravity program row by row."""

    def __init__(self, config: ControllerConfig):
        self.cfg = config
        self.campaign_root = Path(config.campaign_root)
        self.program_path = Path(config.program_path)
        self.controller_dir = self.campaign_root / "controller"
        self.checkpoint_path = self.controller_dir / "checkpoint.json"
        self.events_path = self.controller_dir / "events.jsonl"
        self.lease_path = self.campaign_root / "leases" / "second_light.lease"
        self.heartbeat_path = self.campaign_root / "heartbeat" / "second_light.heartbeat.json"
        self.checkpoints_dir = self.campaign_root / "checkpoints"
        self.evidence_dir = self.campaign_root / "evidence"
        self.process_start_time = now_iso()
        self._lease: SingletonLease | None = None
        self.log: EventLog | None = None
        self.program: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []
        self.program_sha256: str | None = None

    # -- lease (singleton, acquired FIRST) ----------------------------------------------
    def acquire_lease(self) -> SingletonLease:
        """Acquire the exclusive singleton controller lease, or refuse if one is live.

        A second acquire while a live controller holds the flock raises ControllerError
        (heavy_controller_count must never exceed one).
        """
        if self._lease is not None:
            raise ControllerError("lease already held by this controller instance")
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lease = SingletonLease(self.lease_path, owner=LEASE_LABEL).acquire()
        except WatchdogError as exc:
            raise ControllerError(
                f"refusing to start: a live Second Light controller already holds the lease "
                f"({self.lease_path}): {exc}"
            ) from exc
        return self._lease

    def release_lease(self) -> None:
        if self._lease is not None:
            self._lease.close()
            self._lease = None

    # -- program ------------------------------------------------------------------------
    def load_program(self) -> dict[str, Any]:
        """Load the sealed program and verify program_sha256 exactly as it was sealed."""
        doc = read_json_safe(self.program_path)
        declared = doc.get("program_sha256")
        if not isinstance(declared, str) or len(declared) != 64:
            raise ControllerError("program missing a valid program_sha256")
        body = {k: v for k, v in doc.items() if k != "program_sha256"}
        # The generator sealed with hashlib.sha256(json.dumps(body, sort_keys=True)).
        import hashlib
        import json as _json
        recomputed = hashlib.sha256(_json.dumps(body, sort_keys=True).encode()).hexdigest()
        if recomputed != declared:
            raise ControllerError(
                f"program_sha256 mismatch: declared {declared[:16]} recomputed {recomputed[:16]}")
        self.program = doc
        self.rows = list(doc["rows"])
        self.program_sha256 = declared
        return doc

    def working_rows(self) -> list[dict[str, Any]]:
        if self.cfg.only_rows:
            wanted = set(self.cfg.only_rows)
            return [r for r in self.rows if r["row_id"] in wanted]
        return list(self.rows)

    # -- per-row checkpoints (the source of truth for "done") ---------------------------
    def _row_checkpoint_path(self, row_id: str) -> Path:
        return self.checkpoints_dir / f"{row_id}.json"

    def read_row_checkpoint(self, row_id: str) -> dict[str, Any] | None:
        """Return a VALID sealed per-row checkpoint, or None (absent / partial / unsealed).

        A checkpoint is valid only if it self-seals, binds to this program_sha256, names
        this row, and carries a terminal status. Anything else is rejected so the row is
        recomputed (never silently adopted).
        """
        path = self._row_checkpoint_path(row_id)
        if not path.exists():
            return None
        try:
            cp = read_json_safe(path)
        except EcoError:
            return None
        if cp.get("schema") != ROW_CHECKPOINT_SCHEMA:
            return None
        if not sealed(cp, "row_sha256"):
            return None
        if cp.get("row_id") != row_id:
            return None
        if cp.get("program_sha256") != self.program_sha256:
            return None
        if cp.get("status") not in TERMINAL_STATUSES:
            return None
        return cp

    def is_row_sealed(self, row_id: str) -> bool:
        return self.read_row_checkpoint(row_id) is not None

    def needs_pack(self, row_id: str) -> bool:
        """True if the run loop must (re)pack this row. A row with no valid checkpoint is
        pending; a bounded-fidelity checkpoint is repacked only when redo_bounded is set."""
        cp = self.read_row_checkpoint(row_id)
        if cp is None:
            return True
        if self.cfg.redo_bounded:
            metrics = cp.get("metrics", {})
            if metrics.get("bounded_experts") or metrics.get("bounded_slice"):
                return True
        return False

    # -- crash injection ----------------------------------------------------------------
    def _maybe_kill(self, point: str, row_id: str) -> None:
        if self.cfg.kill_at != point:
            return
        if self.cfg.kill_row is not None and self.cfg.kill_row != row_id:
            return
        # A real, abrupt exit at this exact point. In a subprocess the OS releases the
        # flock; the finally-block in the run loop also releases it for the in-process case.
        raise SystemExit(f"HAWKING_SL_KILL_AT={point} row={row_id}")

    # -- row processors -----------------------------------------------------------------
    def process_row(self, reader: Any, row: dict[str, Any]) -> dict[str, Any]:
        """Do one row's work, then WRITE its sealed per-row checkpoint before returning.

        The three compute kill points (fitting/packing/eval) live inside the family
        processors; the two post-write points (after_write_before_receipt,
        after_receipt_before_transition) are here, around the durable write and receipt.
        """
        row_id = row["row_id"]
        family = row["representation_family"]
        started = time.time()
        self._emit_heartbeat({"phase": "row_start", "row_id": row_id, "family": family})

        if row.get("kept_original") or family == "kept_original":
            result = self._process_kept_original(row)
        elif family == "shared_expert_grammar":
            result = self._process_shared_grammar(reader, row)
        elif family == "transform_pq":
            result = self._process_transform_pq(reader, row)
        else:
            raise ControllerError(f"unknown representation_family: {family!r} ({row_id})")

        elapsed = round(time.time() - started, 3)
        checkpoint = self._build_row_checkpoint(row, result, elapsed)
        # Durable, atomic, self-sealed row evidence. After this returns the row is DONE.
        atomic_write_json(self._row_checkpoint_path(row_id), checkpoint)
        self._maybe_kill("after_write_before_receipt", row_id)
        # Receipt: append the row_sealed event to the hash-chained log.
        assert self.log is not None
        self.log.append("row_sealed", {"row_id": row_id, "status": checkpoint["status"],
                                        "row_sha256": checkpoint["row_sha256"]})
        self._maybe_kill("after_receipt_before_transition", row_id)
        return checkpoint

    def _process_kept_original(self, row: dict[str, Any]) -> dict[str, Any]:
        eb = row["exact_budget"]
        n_weights = int(eb["n_weights"])
        bits_each = int(eb.get("kept_original_bits_each", 16))
        physical_bits = n_weights * bits_each
        return self._budget_verdict(
            row, physical_bits=physical_bits, n_weights_packed=n_weights, bounded=False,
            whole_bpw=float(eb.get("whole_artifact_bpw", bits_each)), rel_error=0.0,
            extra={"kept_original": True, "bits_each": bits_each,
                   "kept_reason": row.get("kept_reason")})

    def _process_shared_grammar(self, reader: Any, row: dict[str, Any]) -> dict[str, Any]:
        import gravity_forge as gf
        from gptoss_moe_runtime import load_expert
        block = int(row["layer"])
        which = "mlp1" if row["tensor_class"] == "expert_mlp1" else "mlp2"
        n_full = int(row.get("n_experts") or row["exact_budget"].get("n_experts", 128))
        n_pack = int(min(self.cfg.max_experts, n_full))

        # FULL-SCOPE path (the actual run): when all experts are requested, use the streaming
        # packer that reads the block tensor ONCE and assigns every expert exactly (memory-bounded,
        # ~10-20s/row) instead of the naive per-expert re-read. This is what makes the complete
        # 128-expert scope feasible (Section 3): not a bounded pilot.
        if n_pack >= n_full:
            import second_light_pack as slp
            kind = "mlp1_weight" if which == "mlp1" else "mlp2_weight"
            self._maybe_kill("fitting", row["row_id"])
            r = slp.pack_layer_grammar_full(
                reader, block, kind, dim=int(row["subvector_dim"]), k=int(row["codebook_size"]),
                stages=int(row["stages"]), seed=block)
            self._maybe_kill("packing", row["row_id"])
            self._maybe_kill("eval", row["row_id"])
            return self._budget_verdict(
                row, physical_bits=int(r["physical_bits"]), n_weights_packed=int(r["n_weights"]),
                bounded=False, whole_bpw=float(r["whole_artifact_bpw"]),
                rel_error=float(r["mean_rel_error"]),
                extra={"n_experts_packed": int(r["n_experts"]), "n_experts_full": n_full,
                       "bounded_experts": False, "full_scope": True,
                       "max_rel_error": round(float(r["max_rel_error"]), 6),
                       "family": "shared_expert_grammar_streaming"})

        experts: list[np.ndarray] = []
        for e in range(n_pack):
            ex = load_expert(reader, block, e)
            experts.append(np.ascontiguousarray(ex[which], dtype=np.float32))
        self._maybe_kill("fitting", row["row_id"])
        art = gf.pack_shared_grammar(
            experts, dim=int(row["subvector_dim"]), k=int(row["codebook_size"]),
            stages=int(row["stages"]), corr_rank=0, seed=block)
        self._maybe_kill("packing", row["row_id"])
        stack = np.stack(experts)
        rel = float(np.linalg.norm(stack - art.recon) / (np.linalg.norm(stack) + 1e-9))
        self._maybe_kill("eval", row["row_id"])
        n_weights_packed = int(sum(w.size for w in experts))
        return self._budget_verdict(
            row, physical_bits=art.physical_bytes * 8, n_weights_packed=n_weights_packed,
            bounded=(n_pack < n_full), whole_bpw=float(art.whole_artifact_bpw),
            rel_error=rel, extra={"n_experts_packed": n_pack, "n_experts_full": n_full,
                                  "bounded_experts": n_pack < n_full})

    def _process_transform_pq(self, reader: Any, row: dict[str, Any]) -> dict[str, Any]:
        import gravity_forge as gf
        group = row["tensor_group"]
        w = np.ascontiguousarray(reader.bf16(group), dtype=np.float32)
        eb = row["exact_budget"]
        n_full = int(eb["n_weights"])
        rows_full = int(w.shape[0])
        bounded = rows_full > self.cfg.pq_max_matrix_rows
        w_pack = np.ascontiguousarray(w[: self.cfg.pq_max_matrix_rows]) if bounded else w
        self._maybe_kill("fitting", row["row_id"])
        # Stable per-row seed: layer index for per-block rows, or the row_id ordinal for global
        # rows (embeddings/output projection have layer == None). Deterministic across runs.
        seed = int(row["layer"]) if row.get("layer") is not None else int(row["row_id"][1:])
        art = gf.pack_transform_pq(
            w_pack, dim=int(row["subvector_dim"]), subspaces=int(row["subspaces"]),
            k=int(row["codebook_size"]), seed=seed)
        self._maybe_kill("packing", row["row_id"])
        rel = float(np.linalg.norm(w_pack - art.recon) / (np.linalg.norm(w_pack) + 1e-9))
        self._maybe_kill("eval", row["row_id"])
        bpw = float(art.whole_artifact_bpw)
        if bounded:
            # Extrapolate the whole-tensor physical bits from the representative slice's bpw
            # and check it against the FULL target. Honest and clearly labelled.
            physical_bits = int(round(bpw * n_full))
            n_weights_packed = n_full
        else:
            physical_bits = art.physical_bytes * 8
            n_weights_packed = int(w_pack.size)
        extra = {"bounded_slice": bounded, "slice_rows": int(w_pack.shape[0]),
                 "slice_weights": int(w_pack.size), "full_rows": rows_full}
        return self._budget_verdict(
            row, physical_bits=physical_bits, n_weights_packed=n_weights_packed,
            bounded=False, whole_bpw=bpw, rel_error=rel, extra=extra)

    def _budget_verdict(self, row: dict[str, Any], *, physical_bits: int, n_weights_packed: int,
                        bounded: bool, whole_bpw: float, rel_error: float,
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Enforce the exact budget (Section 15): physical bits over budget is a HARD failure."""
        eb = row["exact_budget"]
        target_bits_full = int(eb["target_total_bits"])
        n_weights_full = int(eb["n_weights"])
        if bounded and n_weights_packed < n_weights_full and n_weights_full > 0:
            budget_bits = int(math.floor(target_bits_full * n_weights_packed / n_weights_full))
        else:
            budget_bits = target_bits_full
        factor = self.cfg.shrink_budget.get(row["row_id"])
        shrunk = factor is not None
        if shrunk:
            budget_bits = int(budget_bits * float(factor))
        over = int(physical_bits) > int(budget_bits)
        status = STATUS_FAILED_OVER_BUDGET if over else STATUS_SEALED
        metrics: dict[str, Any] = {
            "whole_artifact_bpw": round(float(whole_bpw), 6),
            "rel_error": round(float(rel_error), 6),
            "physical_bits": int(physical_bits),
            "budget_bits": int(budget_bits),
            "target_total_bits": target_bits_full,
            "n_weights_packed": int(n_weights_packed),
            "n_weights_full": n_weights_full,
            "within_budget": (not over),
            "budget_shrunk_for_test": shrunk,
        }
        if extra:
            metrics.update(extra)
        return {"status": status, "metrics": metrics}

    def _build_row_checkpoint(self, row: dict[str, Any], result: dict[str, Any],
                              elapsed: float) -> dict[str, Any]:
        cp = {
            "schema": ROW_CHECKPOINT_SCHEMA,
            "row_id": row["row_id"],
            "layer": row["layer"],
            "tensor_class": row["tensor_class"],
            "representation_family": row["representation_family"],
            "tensor_group": row["tensor_group"],
            "is_subbit": bool(row.get("is_subbit", False)),
            "program_sha256": self.program_sha256,
            "status": result["status"],
            "metrics": result["metrics"],
            "elapsed_seconds": float(elapsed),
            "sealed_at": now_iso(),
            "controller_pid": os.getpid(),
        }
        return seal_field(cp, "row_sha256")

    # -- reconciliation + cursor --------------------------------------------------------
    def _reconcile(self, reason: str) -> dict[str, Any]:
        """Rebuild ground truth from the durable per-row checkpoints, then reconcile the
        event log (append any missing receipts) and the cursor to match. Never redoes a
        sealed row; never trusts a stale cursor over the sealed files."""
        assert self.log is not None
        working = self.working_rows()
        existing_receipts = {ev["payload"]["row_id"] for ev in self.log
                             if ev.get("kind") == "row_sealed"}
        for row in working:
            cp = self.read_row_checkpoint(row["row_id"])
            if cp is None:
                continue
            if row["row_id"] not in existing_receipts:
                # A crash after the durable write but before the receipt: heal the log.
                self.log.append("row_sealed", {"row_id": row["row_id"], "status": cp["status"],
                                                "row_sha256": cp["row_sha256"],
                                                "reconciled": True})
        self.log.append("reconcile", {"reason": reason, "program_sha256": self.program_sha256})
        return self._collect_state()

    def _collect_state(self) -> dict[str, Any]:
        working = self.working_rows()
        completed: list[str] = []
        failed: list[str] = []
        elapseds: list[float] = []
        best: dict[str, Any] | None = None
        for row in working:
            cp = self.read_row_checkpoint(row["row_id"])
            if cp is None:
                continue
            if cp["status"] == STATUS_FAILED_OVER_BUDGET:
                failed.append(row["row_id"])
            else:
                completed.append(row["row_id"])
            elapsed = cp.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                elapseds.append(float(elapsed))
            rel = cp.get("metrics", {}).get("rel_error")
            if cp["status"] == STATUS_SEALED and isinstance(rel, (int, float)):
                if best is None or rel < best["rel_error"]:
                    best = {"row_id": row["row_id"], "rel_error": float(rel),
                            "whole_artifact_bpw": cp["metrics"].get("whole_artifact_bpw")}
        total = len(working)
        done = len(completed) + len(failed)
        pending = total - done
        avg = (sum(elapseds) / len(elapseds)) if elapseds else None
        eta = (avg * pending) if avg is not None else None
        return {
            "total_working_rows": total,
            "completed_rows": len(completed),
            "failed_rows": len(failed),
            "pending_rows": pending,
            "completed_row_ids": completed,
            "failed_row_ids": failed,
            "avg_row_seconds": (round(avg, 3) if avg is not None else None),
            "eta_seconds": (round(eta, 1) if eta is not None else None),
            "best_candidate": best,
        }

    def _write_cursor(self, current_row: str | None, state_hint: str) -> dict[str, Any]:
        state = self._collect_state()
        hb_at = None
        if self.heartbeat_path.exists():
            try:
                hb_at = read_json_safe(self.heartbeat_path).get("beat_at")
            except EcoError:
                hb_at = None
        doc = {
            "schema": CONTROLLER_SCHEMA,
            "controller_pid": os.getpid(),
            "process_start_time": self.process_start_time,
            "lease_identity": {"label": LEASE_LABEL, "pid": os.getpid(),
                               "lease_path": str(self.lease_path)},
            "program_sha256": self.program_sha256,
            "program_path": str(self.program_path),
            "queue_root": str(self.campaign_root),
            "checkpoint_root": str(self.checkpoints_dir),
            "evidence_root": str(self.evidence_dir),
            "controller_root": str(self.controller_dir),
            "last_heartbeat": {"path": str(self.heartbeat_path), "beat_at": hb_at},
            "current_row": current_row,
            "state_hint": state_hint,
            "resource_snapshot": sample_resources(path_for_disk=str(self.campaign_root)),
            "written_at": now_iso(),
            **state,
        }
        doc = seal_field(doc, "checkpoint_sha256")
        atomic_write_json(self.checkpoint_path, doc)
        return doc

    def _emit_heartbeat(self, payload: dict[str, Any]) -> None:
        heartbeat(self.heartbeat_path, {"label": LEASE_LABEL, "campaign": "second_light",
                                        "program_sha256": self.program_sha256, **payload})

    # -- run / resume -------------------------------------------------------------------
    def run(self, max_rows: int | None = None) -> dict[str, Any]:
        """Acquire the singleton lease FIRST, load + verify the program, reconcile to the
        durable per-row checkpoints, then work up to max_rows PENDING rows in program order."""
        self.acquire_lease()
        try:
            self.load_program()
            self.controller_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            self.log = EventLog(self.events_path)
            self._emit_heartbeat({"phase": "boot"})
            self.log.append("controller_start", {"pid": os.getpid(),
                                                  "process_start_time": self.process_start_time,
                                                  "max_rows": max_rows,
                                                  "max_experts": self.cfg.max_experts})
            self._reconcile(reason="run")
            # Write the cursor immediately so status has live state from the first instant.
            self._write_cursor(current_row=None, state_hint="starting")
            working = self.working_rows()
            pending = [r for r in working if self.needs_pack(r["row_id"])]
            to_do = pending if max_rows is None else pending[: max(0, int(max_rows))]
            if not to_do:
                doc = self._write_cursor(current_row=None, state_hint="complete")
                self._emit_heartbeat({"phase": "idle_complete"})
                return self._summary(doc, processed=0)
            from gptoss_moe_runtime import ProvenanceReader
            reader = ProvenanceReader(self.cfg.manifest_path)
            processed = 0
            for row in to_do:
                self._write_cursor(current_row=row["row_id"], state_hint="running")
                cp = self.process_row(reader, row)
                # Transition: the cursor advances only after the row is durably sealed
                # and its receipt is logged.
                self._write_cursor(current_row=None, state_hint="running")
                self._emit_heartbeat({"phase": "row_done", "row_id": row["row_id"],
                                      "status": cp["status"]})
                processed += 1
            doc = self._write_cursor(current_row=None, state_hint="batch_done")
            self.log.append("batch_done", {"processed": processed})
            return self._summary(doc, processed=processed)
        finally:
            self.release_lease()

    def resume(self, max_rows: int | None = None) -> dict[str, Any]:
        """Resume is run: it reconciles to the durable checkpoints and continues the queue."""
        return self.run(max_rows=max_rows)

    def _summary(self, cursor_doc: dict[str, Any], *, processed: int) -> dict[str, Any]:
        return {
            "processed_this_invocation": processed,
            "completed_rows": cursor_doc["completed_rows"],
            "failed_rows": cursor_doc["failed_rows"],
            "pending_rows": cursor_doc["pending_rows"],
            "total_working_rows": cursor_doc["total_working_rows"],
            "program_sha256": self.program_sha256,
            "best_candidate": cursor_doc["best_candidate"],
        }

    # -- reset --------------------------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        """Clear all mutable controller state for a clean test. Never touches the sealed
        program file (the input), only state this controller owns under campaign_root."""
        import shutil
        removed: list[str] = []
        if self.controller_dir.exists():
            shutil.rmtree(self.controller_dir)
            removed.append(str(self.controller_dir))
        # Clear ONLY controller-owned row checkpoints. The evidence_dir holds cross-cutting
        # readiness evidence (gates, parity, seed, crash-proof) that this controller does not
        # own and must never wipe.
        if self.checkpoints_dir.exists():
            for child in self.checkpoints_dir.glob("*"):
                if child.is_file():
                    child.unlink()
                    removed.append(str(child))
        for solo in (self.heartbeat_path, self.lease_path):
            if solo.exists() and not solo.is_symlink():
                solo.unlink()
                removed.append(str(solo))
        for directory in (self.controller_dir, self.checkpoints_dir, self.evidence_dir,
                          self.lease_path.parent, self.heartbeat_path.parent):
            directory.mkdir(parents=True, exist_ok=True)
        return {"reset": True, "removed_count": len(removed)}

    # -- detached spawn -----------------------------------------------------------------
    def spawn_detached(self, max_rows: int | None) -> int:
        """Re-spawn this controller as a real detached, supervised process (setsid) that
        survives the parent. The child acquires the lease and works the queue; the parent
        does NOT hold the lease (it only launches the child) and then exits."""
        self.controller_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.controller_dir / "detached.log"
        argv = [sys.executable, os.path.abspath(__file__), "run",
                "--root", str(self.campaign_root), "--program", str(self.program_path),
                "--manifest", str(self.cfg.manifest_path),
                "--max-experts", str(self.cfg.max_experts),
                "--pq-max-rows", str(self.cfg.pq_max_matrix_rows)]
        if max_rows is not None:
            argv += ["--max-rows", str(max_rows)]
        if self.cfg.only_rows:
            argv += ["--only", ",".join(self.cfg.only_rows)]
        if self.cfg.redo_bounded:
            argv += ["--redo-bounded"]
        log_handle = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            argv, stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=str(repo_root()))
        return proc.pid


# -- config from environment + CLI ------------------------------------------------------
def _config_from_args(args: argparse.Namespace) -> ControllerConfig:
    root = Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT
    program = Path(args.program) if args.program else (root / DEFAULT_PROGRAM_NAME)
    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) if args.only else None
    kill_at = os.environ.get("HAWKING_SL_KILL_AT") or None
    if kill_at is not None and kill_at not in KILL_POINTS:
        raise ControllerError(f"HAWKING_SL_KILL_AT must be one of {KILL_POINTS}; got {kill_at!r}")
    kill_row = os.environ.get("HAWKING_SL_KILL_ROW") or None
    return ControllerConfig(
        campaign_root=root, program_path=program,
        manifest_path=args.manifest or DEFAULT_MANIFEST,
        max_experts=int(args.max_experts), pq_max_matrix_rows=int(args.pq_max_rows),
        only_rows=only, redo_bounded=bool(getattr(args, "redo_bounded", False)),
        kill_at=kill_at, kill_row=kill_row)


def main(argv: list[str] | None = None) -> int:
    import json as _json
    ap = argparse.ArgumentParser(description="Second Light PQ Gravity durable controller.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--max-rows", type=int, default=None)
        p.add_argument("--max-experts", type=int, default=8)
        p.add_argument("--pq-max-rows", type=int, default=DEFAULT_PQ_MAX_MATRIX_ROWS)
        p.add_argument("--only", default=None, help="comma-separated row_ids to restrict to")
        p.add_argument("--root", default=None)
        p.add_argument("--program", default=None)
        p.add_argument("--manifest", default=None)
        p.add_argument("--redo-bounded", action="store_true",
                       help="recompute bounded-fidelity seals (real full-fidelity run)")
        p.add_argument("--detached", action="store_true")
    pr = sub.add_parser("reset")
    pr.add_argument("--max-experts", type=int, default=8)
    pr.add_argument("--pq-max-rows", type=int, default=DEFAULT_PQ_MAX_MATRIX_ROWS)
    pr.add_argument("--only", default=None)
    pr.add_argument("--root", default=None)
    pr.add_argument("--program", default=None)
    pr.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    cfg = _config_from_args(args)
    controller = SecondLightController(cfg)

    if args.command == "reset":
        print(_json.dumps(controller.reset(), indent=2, sort_keys=True))
        return 0

    if getattr(args, "detached", False):
        pid = controller.spawn_detached(max_rows=args.max_rows)
        print(_json.dumps({"detached": True, "child_pid": pid,
                           "log": str(controller.controller_dir / "detached.log")},
                          indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        summary = controller.run(max_rows=args.max_rows)
    else:
        summary = controller.resume(max_rows=args.max_rows)
    print(_json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
