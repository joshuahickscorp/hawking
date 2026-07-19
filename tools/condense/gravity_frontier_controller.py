#!/usr/bin/env python3.12
"""Durable, singleton controller for the Hawking Gravity FRONTIER campaign.

Second Light built ONE PQ geometry (rotated-PQ transform_pq + shared grammar) into a
0.770-BPW artifact, but the science came back NEGATIVE: functional output divergence ~0.69,
no preserved capability. The Frontier does not repeat that one artifact. It SEARCHES
representation GEOMETRY: at a fixed sub-bit rate, which full-rank geometry best preserves the
parent's REQUIRED computation (measured by the reference MoE forward divergence on the routed
experts), BEFORE any rise in rate. Priority order (goal Section 9, mirrored by the program's
search_doctrine): change representation -> change codebook sharing -> change subvector geometry
-> change protected-island allocation -> apply Doctor -> add residual/additive stages -> only
then raise rate. Trials at the SAME rate compete on functional divergence; the lowest-divergence
geometry WITHIN budget wins.

This is a persistent program driver, built from the SAME durable pattern proven by
`second_light_controller.py`, but wholly independent of it:
  * Singleton lease label  com.hawking.gravity_frontier  (DISTINCT from com.hawking.second_light
    so the two campaigns can never collide on one lock).
  * Campaign root          reports/condense/gravity_frontier.
  * Per-TRIAL sealed checkpoints checkpoints/<row_id>.json, atomic + self-hashed + program-bound,
    written BEFORE the cursor advances; a resume never redoes a sealed trial and never accepts a
    partial one (the flock is the sole liveness truth; a stale PID or a committed JSON can never
    report RUNNING).
  * Heartbeat each trial, detached setsid spawn, and a crash-injection hook (env HAWKING_GF_KILL_AT)
    that raises SystemExit at five exact points so every kill/resume path is testable.

For each TRIAL row the controller:
  1. loads the relevant weight (a bounded representative expert matrix of the tensor class, or a
     test-injected weight),
  2. packs it with the row's representation_family + family_params via the FROZEN gravity_forge
     packers (transform_pq / product_quant / naive_rvq / pq_protected_islands / repairability_shaped
     / pq_doctor_lowrank), measuring the WEIGHT relative error,
  3. measures the FUNCTIONAL output divergence with gravity_forge.output_divergence on the routed
     experts of the reference MoE forward (a tensor-class-isolated pack_fn: only the matrix of this
     trial's class is replaced, the sibling projection stays original), which is THE ranking signal,
  4. verifies the physical bits fit the exact per-matrix budget (else FAILED_OVER_BUDGET), and
  5. seals a per-trial checkpoint carrying {family, params, rate, weight_rel_error,
     functional_divergence, physical_bits, within_budget, status}.

`select_winner()` then picks, per tensor_class, the SEALED within-budget trial with the LOWEST
functional divergence and writes FRONTIER_SELECTION.json: the geometry that best preserves function
at that rate. Every byte stays inside the gravity_forge ByteLedger discipline; nothing is invented.
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
from typing import Any, Callable

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

CONTROLLER_SCHEMA = "hawking.gravity_frontier.controller.v1"
TRIAL_CHECKPOINT_SCHEMA = "hawking.gravity_frontier.trial_checkpoint.v1"
SELECTION_SCHEMA = "hawking.gravity_frontier.frontier_selection.v1"
# DISTINCT from com.hawking.second_light so the two heavy campaigns can never share a lock.
LEASE_LABEL = "com.hawking.gravity_frontier"

STATUS_SEALED = "SEALED"
STATUS_FAILED_OVER_BUDGET = "FAILED_OVER_BUDGET"
TERMINAL_STATUSES = (STATUS_SEALED, STATUS_FAILED_OVER_BUDGET)

# The five exact crash-injection points, in per-trial execution order.
KILL_POINTS = ("fit", "pack", "eval", "after_write", "after_receipt")

DEFAULT_CAMPAIGN_ROOT = repo_root() / "reports" / "condense" / "gravity_frontier"
DEFAULT_PROGRAM_NAME = "GPT_OSS_120B_GRAVITY_FRONTIER_PROGRAM.json"
DEFAULT_MANIFEST = "reports/condense/subbit_frontier/GRAVITY_120B_PROVENANCE.json"

HEARTBEAT_INTERVAL_SECONDS = 30

# Output-row count of each expert projection (used to isolate the packed matrix to this trial's
# tensor class inside the reference MoE forward). mlp1 = up/gate [5760,2880]; mlp2 = down [2880,2880].
_TENSOR_CLASS_ROWS = {"expert_mlp1": 5760, "expert_mlp2": 2880}
_TENSOR_CLASS_WHICH = {"expert_mlp1": "mlp1", "expert_mlp2": "mlp2"}


class FrontierError(EcoError):
    """Fail-closed error in the Gravity Frontier controller."""


@dataclasses.dataclass
class FrontierConfig:
    campaign_root: Path
    program_path: Path
    manifest_path: str = DEFAULT_MANIFEST
    # Which single expert is loaded as the tensor-class representative for weight rel-error and the
    # exact per-matrix budget check. The budget target is per matrix, so ONE full expert matrix is
    # the exact per-matrix cost (no extrapolation).
    rep_expert: int = 0
    # Functional divergence is measured on one sampled layer of the reference MoE forward. Bounded on
    # purpose: few synthetic inputs and a small top-k so only a handful of routed experts are packed.
    sample_layer_index: int = 0
    func_n_inputs: int = 4
    func_top_k: int = 2
    only_rows: tuple[str, ...] | None = None
    kill_at: str | None = None
    kill_row: str | None = None
    # test injection: {row_id: factor}; the effective budget for that row is scaled by factor, so
    # factor < 1 forces a real over-budget comparison to fail (exact-budget discipline proof).
    shrink_budget: dict[str, float] = dataclasses.field(default_factory=dict)
    heartbeat_interval: int = HEARTBEAT_INTERVAL_SECONDS
    # test injection: a callable (row) -> np.ndarray returning a tiny representative weight, and a
    # callable (row, pack_fn, weight) -> dict returning the functional-divergence evidence. Both are
    # None in production (real reader path). They let the durability suite run in seconds without
    # touching the real 120B source or packing 128 experts.
    weight_provider: Callable[[dict[str, Any]], np.ndarray] | None = None
    divergence_provider: Callable[[dict[str, Any], Callable[[np.ndarray], np.ndarray],
                                   np.ndarray], dict[str, Any]] | None = None


class GravityFrontierController:
    """One durable controller working the sealed geometry-search program trial by trial."""

    def __init__(self, config: FrontierConfig):
        self.cfg = config
        self.campaign_root = Path(config.campaign_root)
        self.program_path = Path(config.program_path)
        self.controller_dir = self.campaign_root / "controller"
        self.checkpoint_path = self.controller_dir / "checkpoint.json"
        self.events_path = self.controller_dir / "events.jsonl"
        self.lease_path = self.campaign_root / "leases" / "gravity_frontier.lease"
        self.heartbeat_path = self.campaign_root / "heartbeat" / "gravity_frontier.heartbeat.json"
        self.checkpoints_dir = self.campaign_root / "checkpoints"
        self.selection_path = self.campaign_root / "FRONTIER_SELECTION.json"
        self.process_start_time = now_iso()
        self._lease: SingletonLease | None = None
        self.log: EventLog | None = None
        self.program: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []
        self.program_sha256: str | None = None

    # -- lease (singleton, acquired FIRST) ----------------------------------------------
    def acquire_lease(self) -> SingletonLease:
        """Acquire the exclusive singleton controller lease, or refuse if one is live."""
        if self._lease is not None:
            raise FrontierError("lease already held by this controller instance")
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lease = SingletonLease(self.lease_path, owner=LEASE_LABEL).acquire()
        except WatchdogError as exc:
            raise FrontierError(
                f"refusing to start: a live Gravity Frontier controller already holds the lease "
                f"({self.lease_path}): {exc}"
            ) from exc
        return self._lease

    def release_lease(self) -> None:
        if self._lease is not None:
            self._lease.close()
            self._lease = None

    # -- program ------------------------------------------------------------------------
    def load_program(self) -> dict[str, Any]:
        """Load the program and verify program_sha256 exactly as it was sealed.

        The sealed real program declares program_sha256 (default-separator sha of the body); it is
        verified fail-closed. A synthetic fixture without a declared hash is accepted and the body
        hash is adopted as the program identity so the durability suite can bind tiny programs.
        """
        import hashlib
        import json as _json
        doc = read_json_safe(self.program_path)
        body = {k: v for k, v in doc.items() if k != "program_sha256"}
        recomputed = hashlib.sha256(_json.dumps(body, sort_keys=True).encode()).hexdigest()
        declared = doc.get("program_sha256")
        if declared is not None:
            if not isinstance(declared, str) or len(declared) != 64:
                raise FrontierError("program declares an invalid program_sha256")
            if recomputed != declared:
                raise FrontierError(
                    f"program_sha256 mismatch: declared {declared[:16]} recomputed {recomputed[:16]}")
        self.program = doc
        self.rows = list(doc["rows"])
        self.program_sha256 = declared or recomputed
        return doc

    def working_rows(self) -> list[dict[str, Any]]:
        if self.cfg.only_rows:
            wanted = set(self.cfg.only_rows)
            return [r for r in self.rows if r["row_id"] in wanted]
        return list(self.rows)

    # -- per-trial checkpoints (the source of truth for "done") -------------------------
    def _row_checkpoint_path(self, row_id: str) -> Path:
        return self.checkpoints_dir / f"{row_id}.json"

    def read_row_checkpoint(self, row_id: str) -> dict[str, Any] | None:
        """Return a VALID sealed per-trial checkpoint, or None (absent / partial / unsealed)."""
        path = self._row_checkpoint_path(row_id)
        if not path.exists():
            return None
        try:
            cp = read_json_safe(path)
        except EcoError:
            return None
        if cp.get("schema") != TRIAL_CHECKPOINT_SCHEMA:
            return None
        if not sealed(cp, "trial_sha256"):
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
        """A trial with no valid terminal checkpoint is pending; a sealed trial is never redone."""
        return self.read_row_checkpoint(row_id) is None

    # -- crash injection ----------------------------------------------------------------
    def _maybe_kill(self, point: str, row_id: str) -> None:
        if self.cfg.kill_at != point:
            return
        if self.cfg.kill_row is not None and self.cfg.kill_row != row_id:
            return
        raise SystemExit(f"HAWKING_GF_KILL_AT={point} row={row_id}")

    # -- family dispatch (frozen gravity_forge packers only) ----------------------------
    @staticmethod
    def _normalize_family(family: str) -> str:
        f = str(family).strip().lower()
        if f.startswith("pack_"):
            f = f[len("pack_"):]
        aliases = {
            "rotated_pq": "transform_pq", "transform_product_quant": "transform_pq",
            "plain_pq": "product_quant", "pq": "product_quant",
            "rvq": "naive_rvq", "residual_vq": "naive_rvq",
            "protected_islands": "pq_protected_islands",
            "shared_grammar": "shared_expert_grammar",
            "pq_doctor": "pq_doctor_lowrank",
        }
        return aliases.get(f, f)

    @staticmethod
    def _params(row: dict[str, Any]) -> dict[str, Any]:
        """Merge family_params with top-level fallbacks (subvector_dim/codebook_size/stages)."""
        fp = dict(row.get("family_params") or {})
        fallbacks = {
            "dim": row.get("subvector_dim"), "k": row.get("codebook_size"),
            "subspaces": row.get("subspaces"), "stages": row.get("stages"),
        }
        for key, val in fallbacks.items():
            if key not in fp and val is not None:
                fp[key] = val
        if "strategy" not in fp and row.get("protected_island_strategy"):
            fp["strategy"] = row["protected_island_strategy"]
        return fp

    def _forge_pack(self, family: str, params: dict[str, Any], w: np.ndarray,
                    seed: int) -> dict[str, Any]:
        """Pack one weight matrix with the named FROZEN forge family and report exact accounting.

        Returns {recon[rows,cols], physical_bits, whole_bpw, n_weights, doctor, family}. Every family
        stays inside the gravity_forge ByteLedger; the doctored family rebuilds its reconstruction
        from the Doctor's own reported second-stage geometry so the functional forward is exact.
        """
        import gravity_forge as gf
        fam = self._normalize_family(family)
        w = np.ascontiguousarray(w, dtype=np.float32)
        dim = int(params.get("dim") or 16)
        k = int(params.get("k") or 16)
        subspaces = int(params.get("subspaces") or 1)
        stages = int(params.get("stages") or 2)

        if fam in ("kept_original",):
            physical_bits = int(w.size) * 16
            return {"recon": w.copy(), "physical_bits": physical_bits,
                    "whole_bpw": 16.0, "n_weights": int(w.size), "doctor": None, "family": fam}

        if fam == "transform_pq":
            art = gf.pack_transform_pq(w, dim=dim, subspaces=subspaces, k=k, seed=seed)
        elif fam == "product_quant":
            art = gf.pack_product_quant(w, dim=dim, subspaces=subspaces, k=k, seed=seed)
        elif fam == "naive_rvq":
            art = gf.pack_naive_rvq(w, dim=dim, k=k, stages=stages, seed=seed)
        elif fam == "pq_protected_islands":
            strategy = str(params.get("strategy") or "residual_energy")
            budget_frac = float(params.get("budget_frac") or 0.03)
            rotate = bool(params.get("rotate") or False)
            art = gf.pack_pq_protected_islands(w, dim=dim, subspaces=subspaces, k=k,
                                               strategy=strategy, budget_frac=budget_frac,
                                               seed=seed, rotate=rotate)
        elif fam == "repairability_shaped":
            base_dim = int(params.get("base_dim") or dim)
            base_k = int(params.get("base_k") or k)
            corr_rank = int(params.get("corr_rank") or 4)
            sparse_rows = int(params.get("sparse_rows") or 4)
            art = gf.pack_repairability_shaped(w, base_dim=base_dim, base_k=base_k,
                                               corr_rank=corr_rank, sparse_rows=sparse_rows, seed=seed)
        elif fam == "shared_expert_grammar":
            corr_rank = int(params.get("corr_rank") or 0)
            art = gf.pack_shared_grammar([w], dim=dim, k=k, stages=stages,
                                         corr_rank=corr_rank, seed=seed)
            recon = np.asarray(art.recon, dtype=np.float32).reshape(w.shape)
            return {"recon": recon, "physical_bits": art.physical_bytes * 8,
                    "whole_bpw": float(art.whole_artifact_bpw), "n_weights": int(art.n_weights),
                    "doctor": None, "family": fam}
        elif fam == "pq_doctor_lowrank":
            return self._forge_pack_doctor(gf, params, w, seed, dim, k, subspaces)
        else:
            raise FrontierError(f"unknown representation_family: {family!r}")

        recon = np.asarray(art.recon, dtype=np.float32).reshape(w.shape)
        return {"recon": recon, "physical_bits": art.physical_bytes * 8,
                "whole_bpw": float(art.whole_artifact_bpw), "n_weights": int(art.n_weights),
                "doctor": None, "family": fam}

    def _forge_pack_doctor(self, gf: Any, params: dict[str, Any], w: np.ndarray, seed: int,
                           dim: int, k: int, subspaces: int) -> dict[str, Any]:
        """pq_doctor_lowrank: a plain-PQ base repaired by a budgeted PQ-aware Doctor. The Doctor's
        exact byte accounting is authoritative (gf.doctor_pq); the doctored reconstruction used for
        the functional forward is rebuilt from the Doctor's reported second-stage geometry so it is
        bit-consistent with what the Doctor billed."""
        strategy = str(params.get("doctor") or "residual_codebook")
        doctor_bpw = float(params.get("doctor_bpw") or 0.15)
        base = gf.pack_product_quant(w, dim=dim, subspaces=subspaces, k=k, seed=seed)
        byte_budget = max(1, int(round(doctor_bpw * w.size / 8.0)))
        doc = gf.doctor_pq(w, base, byte_budget=byte_budget, strategy=strategy, seed=seed)
        base_recon = np.asarray(base.recon, dtype=np.float32)
        recon = base_recon
        rebuilt = False
        if strategy == "residual_codebook":
            ev = doc.get("evidence", {})
            s2 = int(ev.get("stage2_subspaces") or subspaces)
            k2 = int(ev.get("stage2_k") or k)
            D = int(base.config.get("dim", dim))
            resid = (w - base_recon).astype(np.float32)
            # doctor_pq fits the residual codebook with iters=8 (its default); match it exactly.
            stage2 = gf.pack_product_quant(resid, dim=D, subspaces=s2, k=k2, seed=seed, iters=8)
            recon = (base_recon + np.asarray(stage2.recon, dtype=np.float32)).reshape(w.shape)
            rebuilt = True
        physical_bits = int((base.physical_bytes + int(doc["added_bytes"])) * 8)
        return {"recon": np.ascontiguousarray(recon, dtype=np.float32),
                "physical_bits": physical_bits,
                "whole_bpw": physical_bits / max(1, w.size), "n_weights": int(w.size),
                "doctor": {"treatment": doc["treatment"], "added_bytes": int(doc["added_bytes"]),
                           "err_before": doc["err_before"], "err_after": doc["err_after"],
                           "within_budget": bool(doc["within_budget"]),
                           "functional_recon_rebuilt": rebuilt, "byte_budget": byte_budget},
                "family": "pq_doctor_lowrank"}

    def _make_pack_fn(self, family: str, params: dict[str, Any], seed: int,
                      target_rows: int | None) -> Callable[[np.ndarray], np.ndarray]:
        """A pack_fn for gravity_forge.output_divergence that isolates THIS trial's tensor class:
        it packs only the matrix whose row count matches target_rows (the trial's projection) and
        returns the sibling projection unchanged, so the measured divergence is attributable to this
        one geometry on this one tensor class."""
        def pf(ww: np.ndarray) -> np.ndarray:
            ww = np.ascontiguousarray(ww, dtype=np.float32)
            if target_rows is not None and ww.shape[0] != target_rows:
                return ww
            return self._forge_pack(family, params, ww, seed)["recon"]
        return pf

    # -- weight + functional measurement ------------------------------------------------
    def _sample_block(self, row: dict[str, Any]) -> int:
        layers = row.get("sample_layers")
        if isinstance(layers, list) and layers:
            idx = max(0, min(self.cfg.sample_layer_index, len(layers) - 1))
            return int(layers[idx])
        if row.get("layer") is not None:
            return int(row["layer"])
        return 0

    def _tensor_group(self, row: dict[str, Any], block: int) -> str:
        if row.get("tensor_group"):
            return str(row["tensor_group"])
        fmt = row.get("tensor_group_fmt")
        if fmt:
            return str(fmt).replace("{b}", str(block))
        raise FrontierError(f"row {row['row_id']} has no tensor_group / tensor_group_fmt")

    def _load_representative_weight(self, reader: Any, row: dict[str, Any],
                                    block: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Load the tensor-class representative weight. For an expert class this is ONE full expert
        matrix (its size == exact_budget.n_weights_per_matrix, so the per-matrix budget check is
        exact, not extrapolated)."""
        from gptoss_moe_runtime import load_expert
        tc = row["tensor_class"]
        if tc in _TENSOR_CLASS_WHICH:
            which = _TENSOR_CLASS_WHICH[tc]
            ex = load_expert(reader, block, self.cfg.rep_expert)
            w = np.ascontiguousarray(ex[which], dtype=np.float32)
            return w, {"representative_expert": int(self.cfg.rep_expert),
                       "representative_of_class": tc, "sample_layer": block}
        # non-expert tensor class: read the tensor group directly.
        group = self._tensor_group(row, block)
        w = np.ascontiguousarray(reader.bf16(group), dtype=np.float32)
        return w, {"tensor_group": group, "sample_layer": block}

    def _measure_functional(self, reader: Any, row: dict[str, Any], block: int,
                            pack_fn: Callable[[np.ndarray], np.ndarray],
                            weight: np.ndarray) -> tuple[float | None, dict[str, Any]]:
        """Functional output divergence via the reference MoE forward on the routed experts."""
        if self.cfg.divergence_provider is not None:
            div = self.cfg.divergence_provider(row, pack_fn, weight)
            return div.get("mean_output_rel_div"), dict(div)
        tc = row["tensor_class"]
        if tc not in _TENSOR_CLASS_WHICH:
            return None, {"applicable": False,
                          "reason": "non-expert tensor class has no routed MoE forward"}
        import gravity_forge as gf
        div = gf.output_divergence(reader, block, pack_fn, n_inputs=self.cfg.func_n_inputs,
                                   top_k=self.cfg.func_top_k, seed=block)
        return div.get("mean_output_rel_div"), dict(div)

    # -- one trial ----------------------------------------------------------------------
    def process_row(self, reader: Any, row: dict[str, Any]) -> dict[str, Any]:
        """Do one trial's work, then WRITE its sealed per-trial checkpoint before returning."""
        row_id = row["row_id"]
        family = row["representation_family"]
        params = self._params(row)
        block = self._sample_block(row)
        seed = int(block)
        started = time.time()
        self._emit_heartbeat({"phase": "trial_start", "row_id": row_id, "family": family})

        # 1) load the representative weight (or a test-injected tiny weight).
        if self.cfg.weight_provider is not None:
            weight = np.ascontiguousarray(self.cfg.weight_provider(row), dtype=np.float32)
            weight_meta: dict[str, Any] = {"injected_weight": True,
                                           "shape": [int(weight.shape[0]), int(weight.shape[1])]}
        else:
            weight, weight_meta = self._load_representative_weight(reader, row, block)

        # 2) pack it (WEIGHT-space error + exact accounting). Kill point: fit.
        self._maybe_kill("fit", row_id)
        packed = self._forge_pack(family, params, weight, seed)
        recon = packed["recon"]
        weight_rel_error = float(np.linalg.norm(weight - recon) / (np.linalg.norm(weight) + 1e-9))
        self._maybe_kill("pack", row_id)

        # 3) functional output divergence (THE ranking signal). Kill point: eval.
        pack_fn = self._make_pack_fn(family, params, seed, target_rows=int(weight.shape[0]))
        functional_divergence, func_evidence = self._measure_functional(
            reader, row, block, pack_fn, weight)
        self._maybe_kill("eval", row_id)

        # 4) exact-budget verdict.
        result = self._budget_verdict(row, packed, weight_rel_error, functional_divergence,
                                      func_evidence, weight_meta, block)
        elapsed = round(time.time() - started, 3)
        checkpoint = self._build_trial_checkpoint(row, params, result, elapsed)

        # 5) durable, atomic, self-sealed trial evidence, then the receipt. Kill points around them.
        atomic_write_json(self._row_checkpoint_path(row_id), checkpoint)
        self._maybe_kill("after_write", row_id)
        assert self.log is not None
        self.log.append("trial_sealed", {"row_id": row_id, "status": checkpoint["status"],
                                          "trial_sha256": checkpoint["trial_sha256"]})
        self._maybe_kill("after_receipt", row_id)
        return checkpoint

    def _budget_verdict(self, row: dict[str, Any], packed: dict[str, Any], weight_rel_error: float,
                        functional_divergence: float | None, func_evidence: dict[str, Any],
                        weight_meta: dict[str, Any], block: int) -> dict[str, Any]:
        """Enforce the exact per-matrix budget: physical bits over budget is a HARD failure."""
        eb = row.get("exact_budget") or {}
        target_bits = int(eb.get("target_total_bits") or 0)
        budget_bits = target_bits
        factor = self.cfg.shrink_budget.get(row["row_id"])
        shrunk = factor is not None
        if shrunk:
            budget_bits = int(budget_bits * float(factor))
        physical_bits = int(packed["physical_bits"])
        over = target_bits > 0 and physical_bits > budget_bits
        status = STATUS_FAILED_OVER_BUDGET if over else STATUS_SEALED
        metrics: dict[str, Any] = {
            "weight_rel_error": round(float(weight_rel_error), 6),
            "functional_divergence": (round(float(functional_divergence), 6)
                                      if functional_divergence is not None else None),
            "functional_metric": row.get("functional_metric"),
            "whole_artifact_bpw": round(float(packed["whole_bpw"]), 6),
            "physical_bits": physical_bits,
            "budget_bits": int(budget_bits),
            "target_total_bits": target_bits,
            "n_weights": int(packed["n_weights"]),
            "within_budget": (not over),
            "budget_shrunk_for_test": shrunk,
            "sample_layer": int(block),
            "weight_meta": weight_meta,
            "functional_evidence": func_evidence,
        }
        if packed.get("doctor"):
            metrics["doctor"] = packed["doctor"]
        return {"status": status, "metrics": metrics}

    def _build_trial_checkpoint(self, row: dict[str, Any], params: dict[str, Any],
                                result: dict[str, Any], elapsed: float) -> dict[str, Any]:
        eb = row.get("exact_budget") or {}
        cp = {
            "schema": TRIAL_CHECKPOINT_SCHEMA,
            "row_id": row["row_id"],
            "tensor_class": row.get("tensor_class"),
            "representation_family": row.get("representation_family"),
            "family": self._normalize_family(row.get("representation_family", "")),
            "params": params,
            "sharing_group": row.get("sharing_group"),
            "protected_island_strategy": row.get("protected_island_strategy"),
            "rate": eb.get("rate") or row.get("exact_rate") or row.get("starting_rate"),
            "priority_rank": row.get("priority_rank"),
            "program_sha256": self.program_sha256,
            "status": result["status"],
            "metrics": result["metrics"],
            "elapsed_seconds": float(elapsed),
            "sealed_at": now_iso(),
            "controller_pid": os.getpid(),
        }
        return seal_field(cp, "trial_sha256")

    # -- winner selection (geometry frontier) -------------------------------------------
    @staticmethod
    def _winner_key(cp: dict[str, Any]) -> tuple[float, float, int, str]:
        m = cp.get("metrics", {})
        fd = m.get("functional_divergence")
        fd = float(fd) if isinstance(fd, (int, float)) else math.inf
        we = m.get("weight_rel_error")
        we = float(we) if isinstance(we, (int, float)) else math.inf
        pr = cp.get("priority_rank")
        pr = int(pr) if isinstance(pr, (int, float)) else 0
        return (fd, we, pr, str(cp.get("row_id")))

    def select_winner(self) -> dict[str, Any]:
        """Per tensor_class, pick the SEALED within-budget trial with the LOWEST functional
        divergence (tie-break: weight rel-error, then priority_rank, then row_id) and write
        FRONTIER_SELECTION.json: the geometry that best preserves function at that rate."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in self.working_rows():
            cp = self.read_row_checkpoint(row["row_id"])
            if cp is None or cp["status"] != STATUS_SEALED:
                continue
            if not cp.get("metrics", {}).get("within_budget", False):
                continue
            groups.setdefault(str(cp.get("tensor_class")), []).append(cp)
        winners: dict[str, Any] = {}
        for tensor_class, cps in sorted(groups.items()):
            winner = min(cps, key=self._winner_key)
            m = winner["metrics"]
            winners[tensor_class] = {
                "winner_row_id": winner["row_id"],
                "family": winner.get("family"),
                "representation_family": winner.get("representation_family"),
                "params": winner.get("params"),
                "rate": winner.get("rate"),
                "functional_divergence": m.get("functional_divergence"),
                "weight_rel_error": m.get("weight_rel_error"),
                "whole_artifact_bpw": m.get("whole_artifact_bpw"),
                "physical_bits": m.get("physical_bits"),
                "n_candidates": len(cps),
                "candidates": sorted(c["row_id"] for c in cps),
            }
        doc = {
            "schema": SELECTION_SCHEMA,
            "program_sha256": self.program_sha256,
            "program_path": str(self.program_path),
            "ranking_signal": "functional_output_divergence (lowest within budget)",
            "winners_by_tensor_class": winners,
            "selected_at": now_iso(),
        }
        doc = seal_field(doc, "selection_sha256")
        atomic_write_json(self.selection_path, doc)
        return doc

    # -- reconciliation + cursor --------------------------------------------------------
    def _reconcile(self, reason: str) -> dict[str, Any]:
        assert self.log is not None
        working = self.working_rows()
        existing_receipts = {ev["payload"]["row_id"] for ev in self.log
                             if ev.get("kind") == "trial_sealed"}
        for row in working:
            cp = self.read_row_checkpoint(row["row_id"])
            if cp is None:
                continue
            if row["row_id"] not in existing_receipts:
                # A crash after the durable write but before the receipt: heal the log.
                self.log.append("trial_sealed", {"row_id": row["row_id"], "status": cp["status"],
                                                  "trial_sha256": cp["trial_sha256"],
                                                  "reconciled": True})
        self.log.append("reconcile", {"reason": reason, "program_sha256": self.program_sha256})
        return self._collect_state()

    def _collect_state(self) -> dict[str, Any]:
        working = self.working_rows()
        completed: list[str] = []
        failed: list[str] = []
        elapseds: list[float] = []
        best: dict[str, Any] | None = None
        frontier_by_class: dict[str, dict[str, Any]] = {}
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
            m = cp.get("metrics", {})
            fd = m.get("functional_divergence")
            within = m.get("within_budget", False)
            if cp["status"] == STATUS_SEALED and within and isinstance(fd, (int, float)):
                cand = {"row_id": row["row_id"], "family": cp.get("family"),
                        "functional_divergence": float(fd),
                        "weight_rel_error": m.get("weight_rel_error"),
                        "whole_artifact_bpw": m.get("whole_artifact_bpw"),
                        "tensor_class": cp.get("tensor_class"), "rate": cp.get("rate")}
                if best is None or float(fd) < best["functional_divergence"]:
                    best = dict(cand)
                tc = str(cp.get("tensor_class"))
                prev = frontier_by_class.get(tc)
                if prev is None or float(fd) < prev["functional_divergence"]:
                    frontier_by_class[tc] = dict(cand)
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
            "best_by_functional_divergence": best,
            "frontier_by_tensor_class": frontier_by_class,
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
            "campaign_root": str(self.campaign_root),
            "checkpoint_root": str(self.checkpoints_dir),
            "controller_root": str(self.controller_dir),
            "selection_path": str(self.selection_path),
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
        heartbeat(self.heartbeat_path, {"label": LEASE_LABEL, "campaign": "gravity_frontier",
                                        "program_sha256": self.program_sha256, **payload})

    # -- run / resume -------------------------------------------------------------------
    def run(self, max_rows: int | None = None) -> dict[str, Any]:
        """Acquire the singleton lease FIRST, load + verify the program, reconcile to the durable
        per-trial checkpoints, then work up to max_rows PENDING trials in program order."""
        self.acquire_lease()
        try:
            self.load_program()
            self.controller_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.log = EventLog(self.events_path)
            self._emit_heartbeat({"phase": "boot"})
            self.log.append("controller_start", {"pid": os.getpid(),
                                                  "process_start_time": self.process_start_time,
                                                  "max_rows": max_rows})
            self._reconcile(reason="run")
            self._write_cursor(current_row=None, state_hint="starting")
            working = self.working_rows()
            pending = [r for r in working if self.needs_pack(r["row_id"])]
            to_do = pending if max_rows is None else pending[: max(0, int(max_rows))]
            if not to_do:
                self.select_winner()
                doc = self._write_cursor(current_row=None, state_hint="complete")
                self._emit_heartbeat({"phase": "idle_complete"})
                return self._summary(doc, processed=0)
            # A real reader is needed only when a measurement is NOT test-injected. With both the
            # weight and divergence providers supplied (the durability suite) no 120B source is touched.
            reader = None
            if self.cfg.weight_provider is None or self.cfg.divergence_provider is None:
                from gptoss_moe_runtime import ProvenanceReader
                reader = ProvenanceReader(self.cfg.manifest_path)
            processed = 0
            for row in to_do:
                self._write_cursor(current_row=row["row_id"], state_hint="running")
                cp = self.process_row(reader, row)
                self._write_cursor(current_row=None, state_hint="running")
                self._emit_heartbeat({"phase": "trial_done", "row_id": row["row_id"],
                                      "status": cp["status"]})
                processed += 1
            # refresh the geometry frontier selection after every batch.
            self.select_winner()
            doc = self._write_cursor(current_row=None, state_hint="batch_done")
            self.log.append("batch_done", {"processed": processed})
            return self._summary(doc, processed=processed)
        finally:
            self.release_lease()

    def resume(self, max_rows: int | None = None) -> dict[str, Any]:
        return self.run(max_rows=max_rows)

    def _summary(self, cursor_doc: dict[str, Any], *, processed: int) -> dict[str, Any]:
        return {
            "processed_this_invocation": processed,
            "completed_rows": cursor_doc["completed_rows"],
            "failed_rows": cursor_doc["failed_rows"],
            "pending_rows": cursor_doc["pending_rows"],
            "total_working_rows": cursor_doc["total_working_rows"],
            "program_sha256": self.program_sha256,
            "best_by_functional_divergence": cursor_doc["best_by_functional_divergence"],
            "frontier_by_tensor_class": cursor_doc["frontier_by_tensor_class"],
        }

    # -- reset --------------------------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        """Clear all mutable controller state for a clean run. Never touches the sealed program."""
        import shutil
        removed: list[str] = []
        if self.controller_dir.exists():
            shutil.rmtree(self.controller_dir)
            removed.append(str(self.controller_dir))
        if self.checkpoints_dir.exists():
            for child in self.checkpoints_dir.glob("*"):
                if child.is_file():
                    child.unlink()
                    removed.append(str(child))
        for solo in (self.heartbeat_path, self.lease_path, self.selection_path):
            if solo.exists() and not solo.is_symlink():
                solo.unlink()
                removed.append(str(solo))
        for directory in (self.controller_dir, self.checkpoints_dir,
                          self.lease_path.parent, self.heartbeat_path.parent):
            directory.mkdir(parents=True, exist_ok=True)
        return {"reset": True, "removed_count": len(removed)}

    # -- detached spawn -----------------------------------------------------------------
    def spawn_detached(self, max_rows: int | None) -> int:
        """Re-spawn this controller as a real detached, supervised process (setsid) that survives
        the parent. The child acquires the lease and works the queue; the parent only launches it."""
        self.controller_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.controller_dir / "detached.log"
        argv = [sys.executable, os.path.abspath(__file__), "run",
                "--root", str(self.campaign_root), "--program", str(self.program_path),
                "--manifest", str(self.cfg.manifest_path),
                "--func-n-inputs", str(self.cfg.func_n_inputs),
                "--func-top-k", str(self.cfg.func_top_k),
                "--rep-expert", str(self.cfg.rep_expert)]
        if max_rows is not None:
            argv += ["--max-rows", str(max_rows)]
        if self.cfg.only_rows:
            argv += ["--only", ",".join(self.cfg.only_rows)]
        log_handle = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            argv, stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=str(repo_root()))
        return proc.pid


# -- config from CLI --------------------------------------------------------------------
def _config_from_args(args: argparse.Namespace) -> FrontierConfig:
    root = Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT
    program = Path(args.program) if args.program else (root / DEFAULT_PROGRAM_NAME)
    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) if args.only else None
    kill_at = os.environ.get("HAWKING_GF_KILL_AT") or None
    if kill_at is not None and kill_at not in KILL_POINTS:
        raise FrontierError(f"HAWKING_GF_KILL_AT must be one of {KILL_POINTS}; got {kill_at!r}")
    kill_row = os.environ.get("HAWKING_GF_KILL_ROW") or None
    return FrontierConfig(
        campaign_root=root, program_path=program,
        manifest_path=args.manifest or DEFAULT_MANIFEST,
        rep_expert=int(getattr(args, "rep_expert", 0)),
        func_n_inputs=int(getattr(args, "func_n_inputs", 4)),
        func_top_k=int(getattr(args, "func_top_k", 2)),
        only_rows=only, kill_at=kill_at, kill_row=kill_row)


def main(argv: list[str] | None = None) -> int:
    import json as _json
    ap = argparse.ArgumentParser(description="Gravity Frontier durable geometry-search controller.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--max-rows", type=int, default=None)
        p.add_argument("--func-n-inputs", type=int, default=4)
        p.add_argument("--func-top-k", type=int, default=2)
        p.add_argument("--rep-expert", type=int, default=0)
        p.add_argument("--only", default=None, help="comma-separated row_ids to restrict to")
        p.add_argument("--root", default=None)
        p.add_argument("--program", default=None)
        p.add_argument("--manifest", default=None)
        p.add_argument("--detached", action="store_true")
    for name in ("reset", "select"):
        pr = sub.add_parser(name)
        pr.add_argument("--only", default=None)
        pr.add_argument("--root", default=None)
        pr.add_argument("--program", default=None)
        pr.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    cfg = _config_from_args(args) if args.command in ("run", "resume") else FrontierConfig(
        campaign_root=Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT,
        program_path=Path(args.program) if args.program
        else ((Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT) / DEFAULT_PROGRAM_NAME),
        manifest_path=args.manifest or DEFAULT_MANIFEST,
        only_rows=(tuple(x.strip() for x in args.only.split(",") if x.strip())
                   if args.only else None))
    controller = GravityFrontierController(cfg)

    if args.command == "reset":
        print(_json.dumps(controller.reset(), indent=2, sort_keys=True))
        return 0
    if args.command == "select":
        controller.load_program()
        print(_json.dumps(controller.select_winner(), indent=2, sort_keys=True, default=str))
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
