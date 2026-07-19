#!/usr/bin/env python3.12
"""Durable, singleton controller for the Hawking Full Frontier campaign - Gate G2 (COMPLETE LAYER).

G1 (gravity_frontier_g1.py) reproduced the bounded geometry winners on a LARGER real expert sample
with a calibration/validation split, but its functional signal was still the SYNTHETIC-activation
output divergence over the routed experts of ONE tensor class. G2 is the next fidelity: it runs a
COMPLETE GPT-OSS layer (layer 0) through each candidate representation and measures functional
quality on REAL residual-stream activations produced by the verified reference path
(gptoss_block.block0_moe_inputs -> the true post-attention, post-mlp-norm MoE input), under a durable
controller that survives chat termination and advances sealed checkpoints.

For each candidate ROW the controller:
  1. loads a representative expert matrix of the row's tensor class (the exact per-matrix budget
     cost, no extrapolation) and packs it with the row's FROZEN gravity_forge family + params,
     measuring the WEIGHT relative error and the exact physical bits;
  2. generates REAL layer-0 activations on a few Harmony-ish token sequences (a calibration set and a
     disjoint validation set) via the verified block0 forward, then runs the reference MoE forward
     ORIGINAL vs CANDIDATE-packed over the routed experts, with the candidate applied ONLY to the
     row's tensor class (class isolation, exactly like g1);
  3. measures the complete-layer functional signals:
        router_topk_agreement       - fraction of inputs whose router top-k set is unchanged
                                       (~1.0 whenever the router stays source-native, a valid control
                                        result confirming class-isolation does not perturb routing),
        expert_output_cosine         - cosine of the routed experts' outputs (original vs packed),
        expert_output_rel_error      - relative error of the routed experts' outputs,
        weighted_combine_divergence  - relative divergence of the router-weighted MoE combine,
        layer_hidden_state_cosine    - cosine of the complete layer-0 output hidden state
                                       (resid + MoE-out) original vs packed  ==> THE ranking signal,
        complete_layer_bpw           - whole-layer effective bits-per-weight, billing the packed
                                       tensor class at its packed rate and every other layer tensor at
                                       its source-native rate (MXFP4 experts, BF16 organs);
  4. verifies the packed physical bits fit the exact per-matrix budget (else FAILED_OVER_BUDGET);
  5. seals a per-row checkpoint (atomic, self-hashed, program-bound) BEFORE the cursor advances.

`select_winner()` picks, per tensor_class, the SEALED within-budget row with the HIGHEST
layer_hidden_state_cosine, EXCLUDING the source_native controls (they are reference boundaries, not
compression candidates), and writes G2_SELECTION.json.

Durability pattern is copied from gravity_frontier_controller.py (itself proven by
second_light_controller.py) but wholly independent:
  * Singleton lease label  com.hawking.frontier_g2  (DISTINCT from com.hawking.gravity_frontier and
    com.hawking.second_light so the three campaigns can never collide on one lock).
  * Campaign root          reports/condense/general_frontier/G2.
  * Per-row sealed checkpoints, written before the cursor advances; a resume never redoes a sealed
    row and never accepts a partial one (the flock is the sole liveness truth).
  * Heartbeat each row, detached setsid spawn, crash-injection env HAWKING_G2_KILL_AT at five points.

HONESTY BOUNDARY (do not weaken): the token sequences are synthetic Harmony-ish id streams pushed
through the REAL embedding + attention + mlp-norm path, so the residual-stream geometry and in-context
mixing are genuine but the token CONTENT is not real Harmony text. This is a real-activation FUNCTIONAL
PROXY, capability_parity False. It authorizes no Escape Receipt and no Event Horizon seal. Real
Harmony-tokenized text + a holdout corpus are the next fidelity.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
    EcoError, atomic_write_json, canonical_bytes, hash_value, now_iso, read_json_safe, repo_root,
    seal_field, sealed, sha_file,
)
from succ_events import EventLog  # noqa: E402
from succ_watchdog import SingletonLease, WatchdogError, heartbeat, sample_resources  # noqa: E402

CONTROLLER_SCHEMA = "hawking.frontier_g2.controller.v1"
ROW_CHECKPOINT_SCHEMA = "hawking.frontier_g2.row_checkpoint.v1"
SELECTION_SCHEMA = "hawking.frontier_g2.selection.v1"
PROGRAM_SCHEMA = "hawking.frontier_g2.complete_layer_program.v1"
# DISTINCT from com.hawking.gravity_frontier and com.hawking.second_light: the three heavy campaigns
# can never share a lock. Asserted distinct in the durability suite.
LEASE_LABEL = "com.hawking.frontier_g2"

STATUS_SEALED = "SEALED"
STATUS_FAILED_OVER_BUDGET = "FAILED_OVER_BUDGET"
TERMINAL_STATUSES = (STATUS_SEALED, STATUS_FAILED_OVER_BUDGET)

# The five exact crash-injection points, in per-row execution order.
KILL_POINTS = ("fit", "pack", "eval", "after_write", "after_receipt")

# Program-hash exclusion set (the timestamp-hash bug fix): the sealed program identity is computed
# over the program body EXCLUDING these volatile fields, so a regenerated program with a fresh
# generated_at hashes IDENTICALLY. Both the program builder and this controller use this exact set.
PROGRAM_HASH_EXCLUDE = ("program_sha256", "generated_at")

CONTROL_FAMILY = "source_native_control"

DEFAULT_CAMPAIGN_ROOT = repo_root() / "reports" / "condense" / "general_frontier" / "G2"
DEFAULT_PROGRAM_PATH = (repo_root() / "reports" / "condense" / "general_frontier"
                        / "GENERAL_FRONTIER_PROGRAMS" / "G2_COMPLETE_LAYER_PROGRAM.json")
DEFAULT_MANIFEST = "reports/condense/subbit_frontier/GRAVITY_120B_PROVENANCE.json"
DEFAULT_GENERATION_PATH = (repo_root() / "reports" / "condense" / "general_frontier"
                           / "HAWKING_FRONTIER_GENERATION_F.json")

HEARTBEAT_INTERVAL_SECONDS = 30

# GPT-OSS-120B layer-0 geometry (verified against the provenance manifest).
HIDDEN = 2880
N_EXPERTS = 128
MLP1_ROWS = 5760          # up/gate projection out rows
MLP2_ROWS = 2880          # down projection out rows
TOP_K = 4
# Native storage rates: MXFP4 experts = 4.25 bpw (4-bit values + one 8-bit scale per 32-value group),
# BF16 organs (router, biases, attention, norms) = 16 bpw.
MXFP4_BPW = 4.25
BF16_BPW = 16.0

EXPERT_CLASSES = ("expert_mlp1", "expert_mlp2")
_CLASS_ROWS = {"expert_mlp1": MLP1_ROWS, "expert_mlp2": MLP2_ROWS}
_CLASS_WHICH = {"expert_mlp1": "mlp1", "expert_mlp2": "mlp2"}
_EPS = 1e-9


# ── program hashing (shared with the program builder) ──────────────────────────────────
def program_body_hash(doc: dict[str, Any]) -> str:
    """Stable program identity over the body EXCLUDING the volatile fields (program_sha256 +
    generated_at). Canonical (eco_common) separators. A regenerated program hashes identically."""
    body = {k: v for k, v in doc.items() if k not in PROGRAM_HASH_EXCLUDE}
    return hash_value(body)


# ── whole-layer native inventory (billed pass-through for complete_layer_bpw) ────────────
def build_layer_inventory() -> list[dict[str, Any]]:
    """The complete layer-0 tensor inventory: logical weight count + native bits-per-weight for every
    tensor, tagging the two expert-matrix classes so a packed class can be re-billed at its packed
    rate. Everything else is billed at its source-native rate (nothing is free)."""
    inv: list[dict[str, Any]] = [
        {"name": "experts.mlp1", "n_weights": N_EXPERTS * MLP1_ROWS * HIDDEN,
         "native_bpw": MXFP4_BPW, "tensor_class": "expert_mlp1"},
        {"name": "experts.mlp2", "n_weights": N_EXPERTS * MLP2_ROWS * HIDDEN,
         "native_bpw": MXFP4_BPW, "tensor_class": "expert_mlp2"},
        {"name": "experts.mlp1_bias", "n_weights": N_EXPERTS * MLP1_ROWS, "native_bpw": BF16_BPW},
        {"name": "experts.mlp2_bias", "n_weights": N_EXPERTS * MLP2_ROWS, "native_bpw": BF16_BPW},
        {"name": "router.gate.weight", "n_weights": N_EXPERTS * HIDDEN, "native_bpw": BF16_BPW,
         "tensor_class": "router"},
        {"name": "router.gate.bias", "n_weights": N_EXPERTS, "native_bpw": BF16_BPW},
        {"name": "attn.qkv.weight", "n_weights": 5120 * HIDDEN, "native_bpw": BF16_BPW,
         "tensor_class": "attn"},
        {"name": "attn.qkv.bias", "n_weights": 5120, "native_bpw": BF16_BPW},
        {"name": "attn.out.weight", "n_weights": HIDDEN * 4096, "native_bpw": BF16_BPW,
         "tensor_class": "attn"},
        {"name": "attn.out.bias", "n_weights": HIDDEN, "native_bpw": BF16_BPW},
        {"name": "attn.sinks", "n_weights": 64, "native_bpw": BF16_BPW},
        {"name": "attn.norm.scale", "n_weights": HIDDEN, "native_bpw": BF16_BPW},
        {"name": "mlp.norm.scale", "n_weights": HIDDEN, "native_bpw": BF16_BPW},
    ]
    return inv


def complete_layer_bpw(inventory: list[dict[str, Any]], packed_class: str | None,
                       packed_bpw_per_weight: float | None) -> float:
    """Whole-layer effective bits-per-weight: the packed tensor class billed at packed_bpw_per_weight
    across ALL its weights, every other tensor billed at its native rate. PQ physical bits are
    data-independent (indices + codebook are fixed by shape/params), so one representative matrix's
    packed rate is the exact rate for all 128 experts of that class."""
    total_bits = 0.0
    total_weights = 0
    for item in inventory:
        n = int(item["n_weights"])
        if packed_class is not None and item.get("tensor_class") == packed_class \
                and packed_bpw_per_weight is not None:
            bpw = float(packed_bpw_per_weight)
        else:
            bpw = float(item["native_bpw"])
        total_bits += bpw * n
        total_weights += n
    return total_bits / max(1, total_weights)


def native_layer_bpw(inventory: list[dict[str, Any]]) -> float:
    return complete_layer_bpw(inventory, None, None)


# ── functional helpers (numpy, CPU reference; match gptoss_moe_runtime exactly) ──────────
def _swiglu(h: np.ndarray) -> np.ndarray:
    gate, up = np.split(h, 2, axis=-1)
    return (gate * (1.0 / (1.0 + np.exp(-gate)))) * up


def _expert_forward(ex: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = ex["mlp1"] @ x + ex["mlp1_bias"]
    a = _swiglu(h)
    return ex["mlp2"] @ a + ex["mlp2_bias"]


def _router_topk(router: dict[str, np.ndarray], x: np.ndarray, top_k: int) -> np.ndarray:
    logits = router["weight"] @ x + router["bias"]
    return np.argsort(-logits)[:top_k]


def _moe_forward(x: np.ndarray, router: dict[str, np.ndarray],
                 experts: dict[int, dict[str, np.ndarray]], top_k: int) -> np.ndarray:
    logits = router["weight"] @ x + router["bias"]
    idx = np.argsort(-logits)[:top_k]
    w = logits[idx]
    w = np.exp(w - w.max())
    w = w / w.sum()
    out = np.zeros_like(x)
    for e, gate_w in zip(idx, w):
        out += gate_w * _expert_forward(experts[int(e)], x)
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb + _EPS))


def _rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + _EPS))


# ── the real layer engine (production: reads the 120B source) ────────────────────────────
class RealLayerEngine:
    """Produces REAL layer-0 activations + router + routed experts from the verified reference path.
    Loads the embedding matrix ONCE for the whole activation build; caches loaded experts. The token
    streams are synthetic Harmony-ish id sequences (bounded, deterministic, seeded) pushed through the
    genuine embedding + attention + mlp-norm forward."""

    def __init__(self, manifest_path: str, *, n_calibration: int, n_validation: int,
                 tokens_per_seq: int, top_k: int, rep_expert: int, seed: int = 0):
        import gptoss_moe_runtime as rt
        self.rt = rt
        self.reader = rt.ProvenanceReader(manifest_path)
        self.n_calibration = int(n_calibration)
        self.n_validation = int(n_validation)
        self.tokens_per_seq = int(tokens_per_seq)
        self.top_k = int(top_k)
        self.rep_expert = int(rep_expert)
        self.seed = int(seed)
        self._router: dict[str, np.ndarray] | None = None
        self._experts: dict[int, dict[str, np.ndarray]] = {}
        self._activations: dict[str, Any] | None = None
        self.hidden = HIDDEN

    def source_present(self) -> bool:
        sample = self.reader.by_name.get("block.0.mlp.gate.weight")
        return sample is not None and Path(sample["shard_path"]).exists()

    def _vocab(self) -> int:
        emb = self.reader.by_name.get("embedding.weight")
        return int(emb["shape"][0]) if emb else 201088

    def _token_sequences(self, n_seqs: int, base_seed: int) -> list[list[int]]:
        vocab = self._vocab()
        seqs: list[list[int]] = []
        for i in range(n_seqs):
            rng = np.random.default_rng(base_seed + i)
            ids = rng.integers(0, vocab, size=self.tokens_per_seq, dtype=np.int64).tolist()
            seqs.append([int(t) for t in ids])
        return seqs

    def activations(self) -> dict[str, Any]:
        """(calibration, validation) real MoE-input activations, disjoint seeds. Each item is
        (moe_input[HIDDEN], resid[HIDDEN]) for one token position."""
        if self._activations is not None:
            return self._activations
        import gptoss_block as blk
        emb = self.reader.bf16("embedding.weight")                     # loaded ONCE (~1.1 GB)
        mlp_norm = self.reader.bf16("block.0.mlp.norm.scale")
        cal_seqs = self._token_sequences(self.n_calibration, self.seed + 1)
        val_seqs = self._token_sequences(self.n_validation, self.seed + 100_003)

        def run(seqs: list[list[int]]) -> list[tuple[np.ndarray, np.ndarray]]:
            items: list[tuple[np.ndarray, np.ndarray]] = []
            for ids in seqs:
                x = np.ascontiguousarray(emb[ids], dtype=np.float32)   # [seq, HIDDEN]
                attn = blk.block0_attention(self.reader, x)
                resid = x + attn                                        # post-attention residual
                m = blk.rmsnorm(resid, mlp_norm)                        # the true MoE input
                for p in range(m.shape[0]):
                    items.append((np.ascontiguousarray(m[p], dtype=np.float32),
                                  np.ascontiguousarray(resid[p], dtype=np.float32)))
            return items

        cal = run(cal_seqs)
        val = run(val_seqs)
        del emb
        token_digest = hashlib.sha256(
            canonical_bytes({"calibration": cal_seqs, "validation": val_seqs})).hexdigest()
        self._activations = {"calibration": cal, "validation": val,
                             "calibration_seqs": cal_seqs, "validation_seqs": val_seqs,
                             "token_digest": token_digest,
                             "source": "real_block0_moe_inputs_synthetic_harmony_tokens"}
        return self._activations

    def router(self) -> dict[str, np.ndarray]:
        if self._router is None:
            self._router = self.rt.load_router(self.reader, 0)
        return self._router

    def load_expert(self, e: int) -> dict[str, np.ndarray]:
        if e not in self._experts:
            self._experts[e] = self.rt.load_expert(self.reader, 0, e)
        return self._experts[e]

    def representative_matrix(self, tensor_class: str) -> np.ndarray:
        """One representative expert matrix of the class (the exact per-matrix budget cost). For a
        non-expert organ class the router gate weight stands in as the representative BF16 organ."""
        if tensor_class in _CLASS_WHICH:
            ex = self.load_expert(self.rep_expert)
            return np.ascontiguousarray(ex[_CLASS_WHICH[tensor_class]], dtype=np.float32)
        return np.ascontiguousarray(self.router()["weight"], dtype=np.float32)

    def inventory(self) -> list[dict[str, Any]]:
        return build_layer_inventory()


# ── config ───────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class G2Config:
    campaign_root: Path
    program_path: Path
    manifest_path: str = DEFAULT_MANIFEST
    generation_path: Path = DEFAULT_GENERATION_PATH
    rep_expert: int = 0
    n_calibration: int = 3
    n_validation: int = 2
    tokens_per_seq: int = 6
    top_k: int = TOP_K
    activation_seed: int = 0
    only_rows: tuple[str, ...] | None = None
    kill_at: str | None = None
    kill_row: str | None = None
    shrink_budget: dict[str, float] = dataclasses.field(default_factory=dict)
    heartbeat_interval: int = HEARTBEAT_INTERVAL_SECONDS
    # test injection: a factory (cfg) -> engine object exposing the RealLayerEngine surface. When set,
    # the real 120B source is never touched, so the durability suite runs in seconds on a tiny layer.
    engine_factory: Callable[["G2Config"], Any] | None = None


class G2Error(EcoError):
    """Fail-closed error in the G2 complete-layer controller."""


class G2Controller:
    """One durable controller working the sealed complete-layer program row by row."""

    def __init__(self, config: G2Config):
        self.cfg = config
        self.campaign_root = Path(config.campaign_root)
        self.program_path = Path(config.program_path)
        self.controller_dir = self.campaign_root / "controller"
        self.checkpoint_path = self.controller_dir / "checkpoint.json"
        self.events_path = self.controller_dir / "events.jsonl"
        self.lease_path = self.campaign_root / "leases" / "frontier_g2.lease"
        self.heartbeat_path = self.campaign_root / "heartbeat" / "frontier_g2.heartbeat.json"
        self.checkpoints_dir = self.campaign_root / "checkpoints"
        self.selection_path = self.campaign_root / "G2_SELECTION.json"
        self.process_start_time = now_iso()
        self._lease: SingletonLease | None = None
        self.log: EventLog | None = None
        self.program: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []
        self.program_sha256: str | None = None
        self._engine: Any = None

    # -- lease (singleton, acquired FIRST) ----------------------------------------------
    def acquire_lease(self) -> SingletonLease:
        if self._lease is not None:
            raise G2Error("lease already held by this controller instance")
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lease = SingletonLease(self.lease_path, owner=LEASE_LABEL).acquire()
        except WatchdogError as exc:
            raise G2Error(
                f"refusing to start: a live G2 controller already holds the lease "
                f"({self.lease_path}): {exc}") from exc
        return self._lease

    def release_lease(self) -> None:
        if self._lease is not None:
            self._lease.close()
            self._lease = None

    # -- program ------------------------------------------------------------------------
    def load_program(self) -> dict[str, Any]:
        """Load + verify the program identity over the body EXCLUDING volatile timestamp fields, so a
        regenerated program (fresh generated_at) validates identically. A synthetic fixture without a
        declared hash adopts its own body hash so the durability suite can bind tiny programs."""
        doc = read_json_safe(self.program_path)
        recomputed = program_body_hash(doc)
        declared = doc.get("program_sha256")
        if declared is not None:
            if not isinstance(declared, str) or len(declared) != 64:
                raise G2Error("program declares an invalid program_sha256")
            if recomputed != declared:
                raise G2Error(
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

    # -- per-row checkpoints (source of truth for "done") -------------------------------
    def _row_checkpoint_path(self, row_id: str) -> Path:
        return self.checkpoints_dir / f"{row_id}.json"

    def read_row_checkpoint(self, row_id: str) -> dict[str, Any] | None:
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

    def needs_work(self, row_id: str) -> bool:
        return self.read_row_checkpoint(row_id) is None

    # -- crash injection ----------------------------------------------------------------
    def _maybe_kill(self, point: str, row_id: str) -> None:
        if self.cfg.kill_at != point:
            return
        if self.cfg.kill_row is not None and self.cfg.kill_row != row_id:
            return
        raise SystemExit(f"HAWKING_G2_KILL_AT={point} row={row_id}")

    # -- forge dispatch (frozen gravity_forge packers only) -----------------------------
    @staticmethod
    def _normalize_family(family: str) -> str:
        f = str(family).strip().lower()
        if f.startswith("pack_"):
            f = f[len("pack_"):]
        aliases = {
            "rotated_pq": "transform_pq", "plain_pq": "product_quant", "pq": "product_quant",
            "rvq": "naive_rvq", "residual_vq": "naive_rvq",
            "protected_islands": "pq_protected_islands", "islands": "pq_protected_islands",
            "pq_islands": "pq_protected_islands", "pq_doctor": "pq_doctor_lowrank",
            "native": CONTROL_FAMILY, "source_native": CONTROL_FAMILY, "control": CONTROL_FAMILY,
        }
        return aliases.get(f, f)

    @staticmethod
    def _params(row: dict[str, Any]) -> dict[str, Any]:
        return dict(row.get("family_params") or {})

    @staticmethod
    def _native_bpw_for_class(tensor_class: str) -> float:
        return MXFP4_BPW if tensor_class in _CLASS_WHICH else BF16_BPW

    def _forge_pack(self, family: str, params: dict[str, Any], w: np.ndarray, seed: int,
                    tensor_class: str) -> dict[str, Any]:
        """Pack ONE matrix with the named FROZEN forge family and report exact accounting. The
        source_native control keeps the matrix verbatim and is billed at its true native rate (MXFP4
        for experts, BF16 for organs), so it is a genuine reference boundary, not a free identity."""
        fam = self._normalize_family(family)
        w = np.ascontiguousarray(w, dtype=np.float32)
        if fam == CONTROL_FAMILY:
            native_bpw = self._native_bpw_for_class(tensor_class)
            physical_bits = int(math.ceil(native_bpw * w.size))
            return {"recon": w.copy(), "physical_bits": physical_bits, "n_weights": int(w.size),
                    "packed_bpw_per_weight": native_bpw, "family": fam, "doctor": None}

        import gravity_forge as gf
        dim = int(params.get("dim") or 16)
        k = int(params.get("k") or 64)
        subspaces = int(params.get("subspaces") or 1)
        stages = int(params.get("stages") or 2)
        if fam == "transform_pq":
            art = gf.pack_transform_pq(w, dim=dim, subspaces=subspaces, k=k, seed=seed)
        elif fam == "product_quant":
            art = gf.pack_product_quant(w, dim=dim, subspaces=subspaces, k=k, seed=seed)
        elif fam == "naive_rvq":
            art = gf.pack_naive_rvq(w, dim=dim, k=k, stages=stages, seed=seed)
        elif fam == "pq_protected_islands":
            strategy = str(params.get("strategy") or "residual_energy")
            budget_frac = float(params.get("budget_frac") or 0.01)
            rotate = bool(params.get("rotate") or False)
            art = gf.pack_pq_protected_islands(w, dim=dim, subspaces=subspaces, k=k,
                                               strategy=strategy, budget_frac=budget_frac,
                                               seed=seed, rotate=rotate)
        elif fam == "pq_doctor_lowrank":
            return self._forge_pack_doctor(gf, params, w, seed, dim, k, subspaces)
        else:
            raise G2Error(f"unknown representation_family: {family!r}")
        recon = np.asarray(art.recon, dtype=np.float32).reshape(w.shape)
        physical_bits = int(art.physical_bytes * 8)
        return {"recon": recon, "physical_bits": physical_bits, "n_weights": int(w.size),
                "packed_bpw_per_weight": physical_bits / max(1, w.size), "family": fam,
                "doctor": None}

    def _forge_pack_doctor(self, gf: Any, params: dict[str, Any], w: np.ndarray, seed: int,
                           dim: int, k: int, subspaces: int) -> dict[str, Any]:
        """pq_doctor_lowrank: a plain-PQ base repaired by a budgeted PQ-aware Doctor. The Doctor's
        exact byte accounting is authoritative; the doctored reconstruction is rebuilt from the
        Doctor's reported second-stage geometry so the functional forward is bit-consistent."""
        strategy = str(params.get("doctor") or "residual_codebook")
        doctor_bpw = float(params.get("doctor_bpw") or 0.15)
        base = gf.pack_product_quant(w, dim=dim, subspaces=subspaces, k=k, seed=seed)
        byte_budget = max(1, int(round(doctor_bpw * w.size / 8.0)))
        doc = gf.doctor_pq(w, base, byte_budget=byte_budget, strategy=strategy, seed=seed)
        base_recon = np.asarray(base.recon, dtype=np.float32)
        recon = base_recon
        if strategy == "residual_codebook":
            ev = doc.get("evidence", {})
            s2 = int(ev.get("stage2_subspaces") or subspaces)
            k2 = int(ev.get("stage2_k") or k)
            D = int(base.config.get("dim", dim))
            resid = (w - base_recon).astype(np.float32)
            stage2 = gf.pack_product_quant(resid, dim=D, subspaces=s2, k=k2, seed=seed, iters=8)
            recon = (base_recon + np.asarray(stage2.recon, dtype=np.float32)).reshape(w.shape)
        physical_bits = int((base.physical_bytes + int(doc["added_bytes"])) * 8)
        return {"recon": np.ascontiguousarray(recon, dtype=np.float32),
                "physical_bits": physical_bits, "n_weights": int(w.size),
                "packed_bpw_per_weight": physical_bits / max(1, w.size), "family": "pq_doctor_lowrank",
                "doctor": {"treatment": doc["treatment"], "added_bytes": int(doc["added_bytes"]),
                           "err_before": doc["err_before"], "err_after": doc["err_after"],
                           "within_budget": bool(doc["within_budget"]), "byte_budget": byte_budget}}

    def _class_pack_recon(self, family: str, params: dict[str, Any], w: np.ndarray,
                          seed: int, tensor_class: str) -> np.ndarray:
        return self._forge_pack(family, params, w, seed, tensor_class)["recon"]

    # -- complete-layer functional measurement ------------------------------------------
    def _measure_set(self, engine: Any, router: dict[str, np.ndarray], inputs: list,
                     row: dict[str, Any], params: dict[str, Any], seed: int) -> dict[str, Any]:
        """Run the reference MoE forward ORIGINAL vs CANDIDATE-packed over the routed experts of one
        input set, with the candidate applied ONLY to the row's tensor class. Returns the complete-
        layer functional metrics for this set."""
        family = self._normalize_family(row["representation_family"])
        tc = row["tensor_class"]
        top_k = self.cfg.top_k
        is_control = family == CONTROL_FAMILY or tc not in _CLASS_WHICH

        # union of routed experts across the sampled inputs (only these are packed / exercised).
        routed: set[int] = set()
        per_input_idx: list[np.ndarray] = []
        for (m, _r) in inputs:
            idx = _router_topk(router, m, top_k)
            per_input_idx.append(idx)
            routed.update(int(e) for e in idx)

        orig: dict[int, dict[str, np.ndarray]] = {}
        packed: dict[int, dict[str, np.ndarray]] = {}
        for e in routed:
            ex = engine.load_expert(e)
            orig[e] = ex
            pe = dict(ex)
            if not is_control:
                which = _CLASS_WHICH.get(tc)
                if which == "mlp1":
                    pe["mlp1"] = self._class_pack_recon(family, params,
                                                        ex["mlp1"].astype(np.float32), seed, tc)
                elif which == "mlp2":
                    pe["mlp2"] = self._class_pack_recon(family, params,
                                                        ex["mlp2"].astype(np.float32), seed, tc)
            packed[e] = pe

        combine_div: list[float] = []
        layer_cos: list[float] = []
        expert_cos: list[float] = []
        expert_relerr: list[float] = []
        agree = 0
        for (m, r), idx in zip(inputs, per_input_idx):
            y0 = _moe_forward(m, router, orig, top_k)
            y1 = _moe_forward(m, router, packed, top_k)
            combine_div.append(_rel_error(y0, y1))
            layer_cos.append(_cosine(r + y0, r + y1))
            # router top-k agreement: routing is unchanged because no candidate repacks the router
            # gate, so this is ~1.0 by construction - a valid control confirming class isolation.
            idx_packed = _router_topk(router, m, top_k)
            agree += int(set(int(e) for e in idx) == set(int(e) for e in idx_packed))
            for e in idx:
                e = int(e)
                eo0 = _expert_forward(orig[e], m)
                eo1 = _expert_forward(packed[e], m)
                expert_cos.append(_cosine(eo0, eo1))
                expert_relerr.append(_rel_error(eo0, eo1))

        n = max(1, len(inputs))
        return {
            "n_inputs": len(inputs),
            "n_experts_exercised": len(routed),
            "router_topk_agreement": round(agree / n, 6),
            "expert_output_cosine": round(float(np.mean(expert_cos)) if expert_cos else 1.0, 6),
            "expert_output_rel_error": round(float(np.mean(expert_relerr)) if expert_relerr else 0.0,
                                             6),
            "weighted_combine_divergence": round(float(np.mean(combine_div)), 6),
            "layer_hidden_state_cosine": round(float(np.mean(layer_cos)), 6),
        }

    def _measure_complete_layer(self, engine: Any, row: dict[str, Any], params: dict[str, Any],
                                seed: int) -> dict[str, Any]:
        acts = engine.activations()
        router = engine.router()
        cal = self._measure_set(engine, router, acts["calibration"], row, params, seed)
        val = self._measure_set(engine, router, acts["validation"], row, params, seed)
        return {"calibration": cal, "validation": val,
                "activation_source": acts.get("source"),
                "token_digest": acts.get("token_digest"),
                "capability_parity": False}

    # -- one row ------------------------------------------------------------------------
    def process_row(self, engine: Any, row: dict[str, Any]) -> dict[str, Any]:
        row_id = row["row_id"]
        family = row["representation_family"]
        fam = self._normalize_family(family)
        params = self._params(row)
        tc = row["tensor_class"]
        seed = 0
        started = time.time()
        self._emit_heartbeat({"phase": "row_start", "row_id": row_id, "family": family})

        # 1) representative matrix + pack (WEIGHT-space error + exact per-matrix accounting).
        weight = np.ascontiguousarray(engine.representative_matrix(tc), dtype=np.float32)
        self._maybe_kill("fit", row_id)
        packed = self._forge_pack(family, params, weight, seed, tc)
        recon = packed["recon"]
        weight_rel_error = _rel_error(weight, recon)
        self._maybe_kill("pack", row_id)

        # 2) complete-layer functional measurement on REAL activations (THE ranking signal).
        functional = self._measure_complete_layer(engine, row, params, seed)
        self._maybe_kill("eval", row_id)

        # 3) exact per-matrix budget verdict + whole-layer bpw.
        result = self._verdict(row, packed, weight_rel_error, functional, engine)
        elapsed = round(time.time() - started, 3)
        checkpoint = self._build_row_checkpoint(row, params, fam, result, elapsed)

        # 4) durable, atomic, self-sealed row evidence, then the receipt.
        atomic_write_json(self._row_checkpoint_path(row_id), checkpoint)
        self._maybe_kill("after_write", row_id)
        assert self.log is not None
        self.log.append("row_sealed", {"row_id": row_id, "status": checkpoint["status"],
                                        "row_sha256": checkpoint["row_sha256"]})
        self._maybe_kill("after_receipt", row_id)
        return checkpoint

    def _verdict(self, row: dict[str, Any], packed: dict[str, Any], weight_rel_error: float,
                 functional: dict[str, Any], engine: Any) -> dict[str, Any]:
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

        tc = row["tensor_class"]
        fam = self._normalize_family(row["representation_family"])
        inventory = engine.inventory()
        packed_class = tc if (tc in _CLASS_WHICH and fam != CONTROL_FAMILY) else None
        packed_bpw = packed["packed_bpw_per_weight"] if packed_class is not None else None
        cl_bpw = complete_layer_bpw(inventory, packed_class, packed_bpw)

        metrics: dict[str, Any] = {
            "weight_rel_error": round(float(weight_rel_error), 6),
            "physical_bits": physical_bits,
            "budget_bits": int(budget_bits),
            "target_total_bits": target_bits,
            "packed_bpw_per_weight": (round(float(packed["packed_bpw_per_weight"]), 6)),
            "n_weights_per_matrix": int(packed["n_weights"]),
            "complete_layer_bpw": round(float(cl_bpw), 6),
            "native_layer_bpw": round(float(native_layer_bpw(inventory)), 6),
            "within_budget": (not over),
            "budget_shrunk_for_test": shrunk,
            "is_control": fam == CONTROL_FAMILY,
            "functional": functional,
        }
        if packed.get("doctor"):
            metrics["doctor"] = packed["doctor"]
        # promote the primary ranking signals to the top for easy reading.
        metrics["layer_hidden_state_cosine"] = functional["validation"]["layer_hidden_state_cosine"]
        metrics["router_topk_agreement"] = functional["validation"]["router_topk_agreement"]
        return {"status": status, "metrics": metrics}

    def _build_row_checkpoint(self, row: dict[str, Any], params: dict[str, Any], fam: str,
                              result: dict[str, Any], elapsed: float) -> dict[str, Any]:
        eb = row.get("exact_budget") or {}
        cp = {
            "schema": ROW_CHECKPOINT_SCHEMA,
            "row_id": row["row_id"],
            "tensor_class": row.get("tensor_class"),
            "representation_family": row.get("representation_family"),
            "family": fam,
            "params": params,
            "rate": eb.get("rate") or row.get("exact_rate"),
            "gate": "G2_complete_layer",
            "layer": row.get("layer", 0),
            "program_sha256": self.program_sha256,
            "status": result["status"],
            "metrics": result["metrics"],
            "elapsed_seconds": float(elapsed),
            "sealed_at": now_iso(),
            "controller_pid": os.getpid(),
        }
        return seal_field(cp, "row_sha256")

    # -- winner selection (highest hidden-state cosine, controls excluded) --------------
    @staticmethod
    def _winner_key(cp: dict[str, Any]) -> tuple[float, float, float, str]:
        m = cp.get("metrics", {})
        val = m.get("functional", {}).get("validation", {})
        cos = val.get("layer_hidden_state_cosine")
        cos = float(cos) if isinstance(cos, (int, float)) else -math.inf
        comb = val.get("weighted_combine_divergence")
        comb = float(comb) if isinstance(comb, (int, float)) else math.inf
        clb = m.get("complete_layer_bpw")
        clb = float(clb) if isinstance(clb, (int, float)) else math.inf
        # higher cosine wins -> negate; then lower combine divergence, then lower layer bpw.
        return (-cos, comb, clb, str(cp.get("row_id")))

    def select_winner(self) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in self.working_rows():
            cp = self.read_row_checkpoint(row["row_id"])
            if cp is None or cp["status"] != STATUS_SEALED:
                continue
            if not cp.get("metrics", {}).get("within_budget", False):
                continue
            if cp.get("family") == CONTROL_FAMILY:
                continue                                     # controls are reference boundaries
            groups.setdefault(str(cp.get("tensor_class")), []).append(cp)
        winners: dict[str, Any] = {}
        for tensor_class, cps in sorted(groups.items()):
            winner = min(cps, key=self._winner_key)
            m = winner["metrics"]
            val = m.get("functional", {}).get("validation", {})
            winners[tensor_class] = {
                "winner_row_id": winner["row_id"],
                "family": winner.get("family"),
                "representation_family": winner.get("representation_family"),
                "params": winner.get("params"),
                "rate": winner.get("rate"),
                "layer_hidden_state_cosine": val.get("layer_hidden_state_cosine"),
                "weighted_combine_divergence": val.get("weighted_combine_divergence"),
                "expert_output_cosine": val.get("expert_output_cosine"),
                "router_topk_agreement": val.get("router_topk_agreement"),
                "complete_layer_bpw": m.get("complete_layer_bpw"),
                "physical_bits": m.get("physical_bits"),
                "n_candidates": len(cps),
                "candidates": sorted(c["row_id"] for c in cps),
            }
        doc = {
            "schema": SELECTION_SCHEMA,
            "gate": "G2_complete_layer",
            "program_sha256": self.program_sha256,
            "program_path": str(self.program_path),
            "ranking_signal": "layer_hidden_state_cosine (highest within budget; controls excluded)",
            "winners_by_tensor_class": winners,
            "selected_at": now_iso(),
        }
        doc = seal_field(doc, "selection_sha256")
        atomic_write_json(self.selection_path, doc)
        return doc

    # -- reconciliation + cursor --------------------------------------------------------
    def _reconcile(self, reason: str) -> None:
        assert self.log is not None
        existing = {ev["payload"]["row_id"] for ev in self.log if ev.get("kind") == "row_sealed"}
        for row in self.working_rows():
            cp = self.read_row_checkpoint(row["row_id"])
            if cp is None:
                continue
            if row["row_id"] not in existing:
                self.log.append("row_sealed", {"row_id": row["row_id"], "status": cp["status"],
                                                "row_sha256": cp["row_sha256"], "reconciled": True})
        self.log.append("reconcile", {"reason": reason, "program_sha256": self.program_sha256})

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
            val = m.get("functional", {}).get("validation", {})
            cos = val.get("layer_hidden_state_cosine")
            within = m.get("within_budget", False)
            is_control = cp.get("family") == CONTROL_FAMILY
            if cp["status"] == STATUS_SEALED and within and not is_control \
                    and isinstance(cos, (int, float)):
                cand = {"row_id": row["row_id"], "family": cp.get("family"),
                        "tensor_class": cp.get("tensor_class"),
                        "layer_hidden_state_cosine": float(cos),
                        "weighted_combine_divergence": val.get("weighted_combine_divergence"),
                        "complete_layer_bpw": m.get("complete_layer_bpw"), "rate": cp.get("rate")}
                if best is None or float(cos) > best["layer_hidden_state_cosine"]:
                    best = dict(cand)
                tc = str(cp.get("tensor_class"))
                prev = frontier_by_class.get(tc)
                if prev is None or float(cos) > prev["layer_hidden_state_cosine"]:
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
            "best_by_hidden_cosine": best,
            "frontier_by_tensor_class": frontier_by_class,
        }

    def _generation_binding(self) -> dict[str, Any]:
        path = Path(self.cfg.generation_path)
        if not path.exists():
            return {"present": False, "path": str(path)}
        try:
            digest, size = sha_file(path)
            doc = read_json_safe(path)
        except EcoError:
            return {"present": True, "path": str(path), "readable": False}
        return {"present": True, "path": str(path), "file_sha256": digest, "file_sha16": digest[:16],
                "generation": doc.get("generation"), "declared_sha256": doc.get("sha256"),
                "size_bytes": size}

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
            "gate": "G2_complete_layer",
            "controller_pid": os.getpid(),
            "process_start_time": self.process_start_time,
            "lease_identity": {"label": LEASE_LABEL, "pid": os.getpid(),
                               "lease_path": str(self.lease_path)},
            "generation_binding": self._generation_binding(),
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
        heartbeat(self.heartbeat_path, {"label": LEASE_LABEL, "campaign": "frontier_g2",
                                        "program_sha256": self.program_sha256, **payload})

    # -- engine -------------------------------------------------------------------------
    def _build_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self.cfg.engine_factory is not None:
            self._engine = self.cfg.engine_factory(self.cfg)
        else:
            engine = RealLayerEngine(self.cfg.manifest_path,
                                     n_calibration=self.cfg.n_calibration,
                                     n_validation=self.cfg.n_validation,
                                     tokens_per_seq=self.cfg.tokens_per_seq,
                                     top_k=self.cfg.top_k, rep_expert=self.cfg.rep_expert,
                                     seed=self.cfg.activation_seed)
            if not engine.source_present():
                raise G2Error("120B source shards absent; cannot run G2 complete-layer measurement")
            self._engine = engine
        return self._engine

    # -- run / resume -------------------------------------------------------------------
    def run(self, max_rows: int | None = None) -> dict[str, Any]:
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
            pending = [r for r in working if self.needs_work(r["row_id"])]
            to_do = pending if max_rows is None else pending[: max(0, int(max_rows))]
            if not to_do:
                self.select_winner()
                doc = self._write_cursor(current_row=None, state_hint="complete")
                self._emit_heartbeat({"phase": "idle_complete"})
                return self._summary(doc, processed=0)
            engine = self._build_engine()
            processed = 0
            for row in to_do:
                self._write_cursor(current_row=row["row_id"], state_hint="running")
                cp = self.process_row(engine, row)
                self._write_cursor(current_row=None, state_hint="running")
                self._emit_heartbeat({"phase": "row_done", "row_id": row["row_id"],
                                      "status": cp["status"]})
                processed += 1
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
            "best_by_hidden_cosine": cursor_doc["best_by_hidden_cosine"],
            "frontier_by_tensor_class": cursor_doc["frontier_by_tensor_class"],
        }

    # -- reset --------------------------------------------------------------------------
    def reset(self) -> dict[str, Any]:
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
        self.controller_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.controller_dir / "detached.log"
        argv = [sys.executable, os.path.abspath(__file__), "run",
                "--root", str(self.campaign_root), "--program", str(self.program_path),
                "--manifest", str(self.cfg.manifest_path),
                "--n-cal", str(self.cfg.n_calibration), "--n-val", str(self.cfg.n_validation),
                "--tokens", str(self.cfg.tokens_per_seq), "--top-k", str(self.cfg.top_k),
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
def _config_from_args(args: argparse.Namespace) -> G2Config:
    root = Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT
    program = Path(args.program) if args.program else DEFAULT_PROGRAM_PATH
    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) if args.only else None
    kill_at = os.environ.get("HAWKING_G2_KILL_AT") or None
    if kill_at is not None and kill_at not in KILL_POINTS:
        raise G2Error(f"HAWKING_G2_KILL_AT must be one of {KILL_POINTS}; got {kill_at!r}")
    kill_row = os.environ.get("HAWKING_G2_KILL_ROW") or None
    return G2Config(
        campaign_root=root, program_path=program,
        manifest_path=args.manifest or DEFAULT_MANIFEST,
        rep_expert=int(getattr(args, "rep_expert", 0)),
        n_calibration=int(getattr(args, "n_cal", 3)),
        n_validation=int(getattr(args, "n_val", 2)),
        tokens_per_seq=int(getattr(args, "tokens", 6)),
        top_k=int(getattr(args, "top_k", TOP_K)),
        only_rows=only, kill_at=kill_at, kill_row=kill_row)


def main(argv: list[str] | None = None) -> int:
    import json as _json
    ap = argparse.ArgumentParser(description="G2 durable complete-layer controller (Full Frontier).")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--max-rows", type=int, default=None)
        p.add_argument("--n-cal", type=int, default=3)
        p.add_argument("--n-val", type=int, default=2)
        p.add_argument("--tokens", type=int, default=6)
        p.add_argument("--top-k", type=int, default=TOP_K)
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

    if args.command in ("run", "resume"):
        cfg = _config_from_args(args)
    else:
        cfg = G2Config(
            campaign_root=Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT,
            program_path=Path(args.program) if args.program else DEFAULT_PROGRAM_PATH,
            manifest_path=args.manifest or DEFAULT_MANIFEST,
            only_rows=(tuple(x.strip() for x in args.only.split(",") if x.strip())
                       if args.only else None))
    controller = G2Controller(cfg)

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
