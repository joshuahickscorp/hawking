#!/usr/bin/env python3.12
"""Durable, singleton controller for the Hawking Full Frontier campaign - Gate G3 (CROSS-LAYER).

G2 (gravity_frontier_g2_controller.py) ran the COMPLETE-LAYER functional measurement on ONE layer
(layer 0) and picked, per tensor class, the geometry with the highest complete-layer hidden-state
cosine. G3 is the next fidelity: CROSS-LAYER TRANSFER. It runs the SAME complete-layer measurement
over an EARLY (0), a MIDDLE (18) and a LATE (35) GPT-OSS-120B layer and records whether the geometry
that won at layer 0 still wins deeper in the network. The science stays honestly Gravity-NEGATIVE
(sub-bit representations lose capability); G3 measures which geometry TRANSFERS with depth, not
whether any of them crosses the Event Horizon.

REUSE (the whole point): the FROZEN G2 controller is imported and used as a pure MEASUREMENT + PACKING
KERNEL. G3 never runs it, never acquires its lease, never writes its campaign - it borrows only the
frozen packer (`_forge_pack`) and the frozen complete-layer measurement (`_measure_complete_layer` /
`_verdict`), which are engine-driven and therefore layer-agnostic. G3 supplies a per-LAYER engine
whose activations come from a GENERALIZED block-N forward, so the identical, verified measurement runs
at any depth. Frozen sources (gptoss_block.rmsnorm/_rope, gptoss_moe_runtime, gravity_forge) are read
and reused, never modified.

For each candidate ROW (geometry x probe layer) the controller:
  1. builds (or reuses) the layer's engine: the REAL block-N residual-stream activations via the
     generalized `block_n_moe_inputs`, the block-N router, and the block-N routed experts;
  2. packs a representative matrix of the row's tensor class with the FROZEN gravity_forge family +
     params (exact per-matrix bits, no extrapolation);
  3. runs the frozen complete-layer measurement ORIGINAL vs CANDIDATE-packed over the routed experts,
     class-isolated, on REAL block-N activations, yielding router_topk_agreement, expert_output_cosine,
     weighted_combine_divergence, layer_hidden_state_cosine (THE ranking signal), complete_layer_bpw;
  4. verifies the packed physical bits fit the exact per-matrix budget (else FAILED_OVER_BUDGET);
  5. seals a per-row checkpoint (atomic, self-hashed, program-bound, GENERATION-M-bound) BEFORE the
     cursor advances.

`select_transfer()` computes, per tensor_class, the winner geometry at each layer (highest
layer_hidden_state_cosine within budget, controls excluded) and the TRANSFER verdict: does the layer-0
winner still win at layers 18 and 35? It writes G3_TRANSFER.json (the real cross-layer finding).

Durability pattern is copied from gravity_frontier_g2_controller.py (proven durable) but WHOLLY
INDEPENDENT:
  * Singleton lease label  com.hawking.frontier_g3  (DISTINCT from com.hawking.frontier_g2,
    com.hawking.gravity_frontier, com.hawking.second_light, and the mechanics namespace: the heavy
    campaigns can never collide on one lock). Asserted distinct in the durability suite.
  * Campaign root          reports/condense/general_frontier/G3.
  * Per-row sealed checkpoints, written before the cursor advances; a resume never redoes a sealed row
    and never accepts a partial one (the flock is the sole liveness truth). Every checkpoint binds the
    Generation-M closure sha.
  * Heartbeat each row, detached setsid spawn, crash-injection env HAWKING_G3_KILL_AT at five points.

HONESTY BOUNDARY (do not weaken): (a) the token streams are synthetic Harmony-ish id sequences pushed
through the REAL embedding + attention + mlp-norm path, so the residual-stream geometry is genuine but
the token CONTENT is not real Harmony text; (b) the block-N attention is a from-config forward (an
APPROXIMATION valid only for the RELATIVE orig-vs-packed divergence, exactly as the block-0 forward
is - the shared residual and shared reference SwiGLU largely cancel the approximation; for seq < 128
the sliding-window / full-attention distinction is inactive so the same math is faithful at any
depth); (c) execution_generation="M" is quality-NEUTRAL: the m2_shared_lookup_linear grammar is EXACT
parity to the Gen-F direct-compact recon, so binding it changes mechanical cost, never the measured
divergence. capability_parity is False; G3 authorizes no Escape Receipt and no Event Horizon seal.
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

# FROZEN reuse: the G2 controller supplies the geometry constants, the stable program hash, the
# whole-layer inventory + bpw accounting, and - via a borrowed instance - the frozen packer and the
# frozen complete-layer measurement. gptoss_block supplies the frozen RoPE + RMSNorm math.
import gptoss_block as blk  # noqa: E402
from gravity_frontier_g2_controller import (  # noqa: E402
    BF16_BPW, CONTROL_FAMILY, HIDDEN, MLP1_ROWS, MLP2_ROWS, MXFP4_BPW, N_EXPERTS, TOP_K,
    _CLASS_WHICH, _rel_error, build_layer_inventory, complete_layer_bpw, native_layer_bpw,
    program_body_hash, G2Config, G2Controller,
)

CONTROLLER_SCHEMA = "hawking.frontier_g3.controller.v1"
ROW_CHECKPOINT_SCHEMA = "hawking.frontier_g3.row_checkpoint.v1"
TRANSFER_SCHEMA = "hawking.frontier_g3.transfer.v1"
PROGRAM_SCHEMA = "hawking.frontier_g3.cross_layer_program.v1"
# DISTINCT from every other heavy campaign lock (asserted in the durability suite).
LEASE_LABEL = "com.hawking.frontier_g3"

STATUS_SEALED = "SEALED"
STATUS_FAILED_OVER_BUDGET = "FAILED_OVER_BUDGET"
TERMINAL_STATUSES = (STATUS_SEALED, STATUS_FAILED_OVER_BUDGET)

# The five exact crash-injection points, in per-row execution order (same as G2).
KILL_POINTS = ("fit", "pack", "eval", "after_write", "after_receipt")

# Stable program identity excludes the volatile fields (same discipline as G2).
PROGRAM_HASH_EXCLUDE = ("program_sha256", "generated_at")

# The three probe layers of GPT-OSS-120B (36 layers total): early / middle / late.
LAYERS = (0, 18, 35)
LAYER_ROLES = ("early", "mid", "late")

FUNCTIONAL_METRICS = [
    "router_topk_agreement", "expert_output_cosine", "expert_output_rel_error",
    "weighted_combine_divergence", "layer_hidden_state_cosine", "complete_layer_bpw",
    "cross_layer_transfer",
]

# The Generation-M execution provider (verified parity-preserving; Gen-F direct-compact default).
EXECUTION_GENERATION = "M"
BASE_EXECUTION_PROVIDER = {
    "name": "m2_shared_lookup_linear",
    "execution_generation": "M",
    "cpu_reference": "mech_measure.m1_lookup_linear_np",
    "metal_provider": "mech_measure.m1_lookup_linear_torch",
    "parity_vs_genf": "EXACT end-to-end (logit cos 1.0, next-token 1.0; operator/expert/layer ~1e-7)",
    "status": "AVAILABLE (verified parity-preserving)",
    "default_fallback_provider": "genf_direct_compact",
    "quality_neutral": True,
    "note": ("Gen-M lookup-linear is EXACT parity to Gen-F direct-compact; it lowers logical FLOPs / "
             "movement / dispatches but does NOT change the measured divergence. Gen-F direct-compact "
             "is the wall-safe default provider."),
}

DEFAULT_CAMPAIGN_ROOT = repo_root() / "reports" / "condense" / "general_frontier" / "G3"
DEFAULT_PROGRAM_PATH = (repo_root() / "reports" / "condense" / "general_frontier"
                        / "GENERAL_FRONTIER_PROGRAMS" / "G3_CROSS_LAYER_PROGRAM.json")
DEFAULT_MANIFEST = "reports/condense/subbit_frontier/GRAVITY_120B_PROVENANCE.json"
DEFAULT_GENERATION_PATH = (repo_root() / "reports" / "mechanics_thermodynamics"
                           / "HAWKING_GENERATION_M.json")
DEFAULT_PROVIDER_MAP_PATH = (repo_root() / "reports" / "mechanics_thermodynamics"
                             / "HAWKING_GENERATION_M_PROVIDER_MAP.json")

HEARTBEAT_INTERVAL_SECONDS = 30
_EPS = 1e-9


# ── program hashing (shared with the program builder, via the frozen G2 hash) ────────────
# program_body_hash is imported from the frozen G2 controller (single source of truth).


# ── generalized block-N forward (the layer-parameterized reference path) ─────────────────
# gptoss_block.block0_attention / block0_moe_inputs are hardcoded to block 0. Every tensor exists per
# block with identical shape, so we generalize the SAME math to an arbitrary block N, reusing the
# frozen RoPE (_rope) and RMSNorm (rmsnorm) primitives verbatim. HONESTY: this is the same from-config
# approximation as block 0, valid only for the RELATIVE orig-vs-packed divergence.
def block_n_attention(reader: Any, n: int, x: np.ndarray) -> np.ndarray:
    """Block-N attention over a sequence x:[seq, HIDDEN] -> attn_out:[seq, HIDDEN]. GQA + RoPE +
    causal + per-head attention sinks - identical to gptoss_block.block0_attention with the block
    index parameterized."""
    seq = x.shape[0]
    N_Q, N_KV, HEAD_DIM = blk.N_Q, blk.N_KV, blk.HEAD_DIM
    nrm = reader.bf16(f"block.{n}.attn.norm.scale")
    qkvw = reader.bf16(f"block.{n}.attn.qkv.weight")   # [5120, 2880]
    qkvb = reader.bf16(f"block.{n}.attn.qkv.bias")     # [5120]
    outw = reader.bf16(f"block.{n}.attn.out.weight")   # [2880, 4096]
    outb = reader.bf16(f"block.{n}.attn.out.bias")     # [2880]
    sinks = reader.bf16(f"block.{n}.attn.sinks")       # [64]

    h = blk.rmsnorm(x, nrm)
    qkv = h @ qkvw.T + qkvb
    q = qkv[:, :N_Q * HEAD_DIM].reshape(seq, N_Q, HEAD_DIM)
    k = qkv[:, N_Q * HEAD_DIM:(N_Q + N_KV) * HEAD_DIM].reshape(seq, N_KV, HEAD_DIM)
    v = qkv[:, (N_Q + N_KV) * HEAD_DIM:].reshape(seq, N_KV, HEAD_DIM)
    pos = np.arange(seq)
    q = blk._rope(q, pos); k = blk._rope(k, pos)
    grp = N_Q // N_KV
    scale = 1.0 / np.sqrt(HEAD_DIM)
    causal = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), 1)
    out = np.zeros((seq, N_Q, HEAD_DIM), dtype=np.float32)
    for hh in range(N_Q):
        kv = hh // grp
        scores = (q[:, hh] @ k[:, kv].T) * scale + causal
        aug = np.concatenate([scores, np.full((seq, 1), sinks[hh], np.float32)], axis=1)
        aug -= aug.max(axis=1, keepdims=True)
        w = np.exp(aug); w /= w.sum(axis=1, keepdims=True)
        out[:, hh] = w[:, :seq] @ v[:, kv]
    return out.reshape(seq, N_Q * HEAD_DIM) @ outw.T + outb


def block_n_moe_inputs(reader: Any, n: int, token_ids: list[int],
                       embeddings: np.ndarray | None = None) -> np.ndarray:
    """The REAL block-N MoE-input activations for a token sequence: embed -> block-N attention ->
    residual -> block-N mlp.norm. embeddings may be passed in to avoid reloading the 2.3GB matrix."""
    if embeddings is None:
        emb = reader.bf16("embedding.weight")
        x = np.ascontiguousarray(emb[token_ids], dtype=np.float32)
        del emb
    else:
        x = np.ascontiguousarray(embeddings[token_ids], dtype=np.float32)
    attn = block_n_attention(reader, n, x)
    resid = x + attn
    mlp_norm = reader.bf16(f"block.{n}.mlp.norm.scale")
    return blk.rmsnorm(resid, mlp_norm)


# ── the real per-layer engine (production: reads the 120B source at block N) ──────────────
class G3LayerEngine:
    """Produces REAL block-N activations + router + routed experts from the verified reference path,
    parameterized by layer. Same engine surface the frozen G2 measurement consumes (activations /
    router / load_expert / representative_matrix / inventory), so the frozen kernel runs at any depth.
    Loads the embedding ONCE per engine; caches loaded experts. The token streams are IDENTICAL across
    layers (seeded independent of layer), so depth is the only variable in the transfer comparison."""

    def __init__(self, manifest_path: str, *, layer: int, n_calibration: int, n_validation: int,
                 tokens_per_seq: int, top_k: int, rep_expert: int, seed: int = 0):
        import gptoss_moe_runtime as rt
        self.rt = rt
        self.reader = rt.ProvenanceReader(manifest_path)
        self.layer = int(layer)
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
        sample = self.reader.by_name.get(f"block.{self.layer}.mlp.gate.weight")
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
        if self._activations is not None:
            return self._activations
        emb = self.reader.bf16("embedding.weight")                     # loaded ONCE (~1.1 GB)
        cal_seqs = self._token_sequences(self.n_calibration, self.seed + 1)
        val_seqs = self._token_sequences(self.n_validation, self.seed + 100_003)

        def run(seqs: list[list[int]]) -> list[tuple[np.ndarray, np.ndarray]]:
            items: list[tuple[np.ndarray, np.ndarray]] = []
            for ids in seqs:
                x = np.ascontiguousarray(emb[ids], dtype=np.float32)
                attn = block_n_attention(self.reader, self.layer, x)
                resid = x + attn
                m = blk.rmsnorm(resid, self.reader.bf16(f"block.{self.layer}.mlp.norm.scale"))
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
                             "token_digest": token_digest, "layer": self.layer,
                             "source": f"real_block{self.layer}_moe_inputs_synthetic_harmony_tokens"}
        return self._activations

    def router(self) -> dict[str, np.ndarray]:
        if self._router is None:
            self._router = self.rt.load_router(self.reader, self.layer)
        return self._router

    def load_expert(self, e: int) -> dict[str, np.ndarray]:
        if e not in self._experts:
            self._experts[e] = self.rt.load_expert(self.reader, self.layer, e)
        return self._experts[e]

    def representative_matrix(self, tensor_class: str) -> np.ndarray:
        if tensor_class in _CLASS_WHICH:
            ex = self.load_expert(self.rep_expert)
            return np.ascontiguousarray(ex[_CLASS_WHICH[tensor_class]], dtype=np.float32)
        return np.ascontiguousarray(self.router()["weight"], dtype=np.float32)

    def inventory(self) -> list[dict[str, Any]]:
        # identical shapes across blocks, so the frozen whole-layer inventory is layer-agnostic.
        return build_layer_inventory()


# ── config ───────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class G3Config:
    campaign_root: Path
    program_path: Path
    manifest_path: str = DEFAULT_MANIFEST
    generation_path: Path = DEFAULT_GENERATION_PATH
    provider_map_path: Path = DEFAULT_PROVIDER_MAP_PATH
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
    # test injection: (cfg, layer) -> engine exposing the G3LayerEngine surface. When set, the real
    # 120B source is never touched, so the durability suite runs in seconds on tiny synthetic layers.
    engine_factory: Callable[["G3Config", int], Any] | None = None


class G3Error(EcoError):
    """Fail-closed error in the G3 cross-layer controller."""


class G3Controller:
    """One durable controller working the sealed cross-layer program row by row, binding Generation M
    and reusing the FROZEN G2 measurement + packing kernel at each probe layer."""

    def __init__(self, config: G3Config):
        self.cfg = config
        self.campaign_root = Path(config.campaign_root)
        self.program_path = Path(config.program_path)
        self.controller_dir = self.campaign_root / "controller"
        self.checkpoint_path = self.controller_dir / "checkpoint.json"
        self.events_path = self.controller_dir / "events.jsonl"
        self.lease_path = self.campaign_root / "leases" / "frontier_g3.lease"
        self.heartbeat_path = self.campaign_root / "heartbeat" / "frontier_g3.heartbeat.json"
        self.checkpoints_dir = self.campaign_root / "checkpoints"
        self.transfer_path = self.campaign_root / "G3_TRANSFER.json"
        self.process_start_time = now_iso()
        self._lease: SingletonLease | None = None
        self.log: EventLog | None = None
        self.program: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []
        self.program_sha256: str | None = None
        self._engine: Any = None
        self._engine_layer: int | None = None
        # the FROZEN G2 controller, borrowed as a PURE measurement + packing kernel. It is never run,
        # never acquires a lease, never writes: only its engine-driven pack/measure methods are used.
        self._kernel: G2Controller | None = None
        self._gen_binding_cache: dict[str, Any] | None = None

    # -- borrowed frozen kernel ---------------------------------------------------------
    def _kernel_ctl(self) -> G2Controller:
        if self._kernel is None:
            kcfg = G2Config(campaign_root=self.campaign_root, program_path=self.program_path,
                            manifest_path=self.cfg.manifest_path, top_k=self.cfg.top_k,
                            n_calibration=self.cfg.n_calibration, n_validation=self.cfg.n_validation,
                            tokens_per_seq=self.cfg.tokens_per_seq, rep_expert=self.cfg.rep_expert,
                            shrink_budget=dict(self.cfg.shrink_budget))
            self._kernel = G2Controller(kcfg)
        return self._kernel

    # -- lease (singleton, acquired FIRST) ----------------------------------------------
    def acquire_lease(self) -> SingletonLease:
        if self._lease is not None:
            raise G3Error("lease already held by this controller instance")
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lease = SingletonLease(self.lease_path, owner=LEASE_LABEL).acquire()
        except WatchdogError as exc:
            raise G3Error(
                f"refusing to start: a live G3 controller already holds the lease "
                f"({self.lease_path}): {exc}") from exc
        return self._lease

    def release_lease(self) -> None:
        if self._lease is not None:
            self._lease.close()
            self._lease = None

    # -- program ------------------------------------------------------------------------
    def load_program(self) -> dict[str, Any]:
        doc = read_json_safe(self.program_path)
        recomputed = program_body_hash(doc)
        declared = doc.get("program_sha256")
        if declared is not None:
            if not isinstance(declared, str) or len(declared) != 64:
                raise G3Error("program declares an invalid program_sha256")
            if recomputed != declared:
                raise G3Error(
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

    # -- generation-M binding (every checkpoint binds it) -------------------------------
    def _generation_binding(self) -> dict[str, Any]:
        if self._gen_binding_cache is not None:
            return self._gen_binding_cache
        out: dict[str, Any] = {"generation": "M"}
        gen_path = Path(self.cfg.generation_path)
        prov_path = Path(self.cfg.provider_map_path)
        gen_sha = prov_sha = None
        out["generation_path"] = str(gen_path)
        out["provider_map_path"] = str(prov_path)
        if gen_path.exists():
            try:
                gen_sha, gsz = sha_file(gen_path)
                gdoc = read_json_safe(gen_path)
                out.update({"generation_present": True, "generation_file_sha256": gen_sha,
                            "generation_declared_sha256": gdoc.get("sha256"),
                            "generation_name": gdoc.get("generation"),
                            "generation_size_bytes": gsz})
            except EcoError:
                out["generation_present"] = True
                out["generation_readable"] = False
        else:
            out["generation_present"] = False
        if prov_path.exists():
            try:
                prov_sha, _ = sha_file(prov_path)
                pdoc = read_json_safe(prov_path)
                out.update({"provider_map_present": True, "provider_map_file_sha256": prov_sha,
                            "provider_map_declared_sha256": pdoc.get("sha256"),
                            "base_execution_provider_name": (pdoc.get("base_execution_provider") or {})
                            .get("name")})
            except EcoError:
                out["provider_map_present"] = True
                out["provider_map_readable"] = False
        else:
            out["provider_map_present"] = False
        # the closure sha binds BOTH artifacts (the Generation-M closure).
        out["closure_sha256"] = hash_value({"generation_file_sha256": gen_sha,
                                             "provider_map_file_sha256": prov_sha})
        out["execution_generation"] = EXECUTION_GENERATION
        out["base_execution_provider"] = BASE_EXECUTION_PROVIDER
        self._gen_binding_cache = out
        return out

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
        raise SystemExit(f"HAWKING_G3_KILL_AT={point} row={row_id}")

    # -- per-layer engine (bounded: one engine at a time) -------------------------------
    def _engine_for(self, layer: int) -> Any:
        if self._engine is not None and self._engine_layer == layer:
            return self._engine
        # drop the previous layer's engine (frees its embedding + cached experts) before building the
        # next; rows are grouped by layer, so this rebuilds only at the 3 layer boundaries.
        self._engine = None
        self._engine_layer = None
        if self.cfg.engine_factory is not None:
            engine = self.cfg.engine_factory(self.cfg, layer)
        else:
            engine = G3LayerEngine(self.cfg.manifest_path, layer=layer,
                                   n_calibration=self.cfg.n_calibration,
                                   n_validation=self.cfg.n_validation,
                                   tokens_per_seq=self.cfg.tokens_per_seq, top_k=self.cfg.top_k,
                                   rep_expert=self.cfg.rep_expert, seed=self.cfg.activation_seed)
            if not engine.source_present():
                raise G3Error(f"120B source shards absent for layer {layer}; cannot run G3")
        self._engine = engine
        self._engine_layer = layer
        return engine

    # -- one row ------------------------------------------------------------------------
    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        row_id = row["row_id"]
        family = row["representation_family"]
        kernel = self._kernel_ctl()
        fam = kernel._normalize_family(family)
        params = kernel._params(row)
        tc = row["tensor_class"]
        layer = int(row.get("layer", 0))
        seed = 0
        started = time.time()
        self._emit_heartbeat({"phase": "row_start", "row_id": row_id, "family": family,
                              "layer": layer})

        engine = self._engine_for(layer)

        # 1) representative matrix + pack (WEIGHT-space error + exact per-matrix accounting).
        weight = np.ascontiguousarray(engine.representative_matrix(tc), dtype=np.float32)
        self._maybe_kill("fit", row_id)
        packed = kernel._forge_pack(family, params, weight, seed, tc)
        weight_rel_error = _rel_error(weight, packed["recon"])
        self._maybe_kill("pack", row_id)

        # 2) complete-layer functional measurement on REAL block-N activations (frozen kernel).
        functional = kernel._measure_complete_layer(engine, row, params, seed)
        self._maybe_kill("eval", row_id)

        # 3) exact per-matrix budget verdict + whole-layer bpw (frozen kernel).
        result = kernel._verdict(row, packed, weight_rel_error, functional, engine)
        elapsed = round(time.time() - started, 3)
        checkpoint = self._build_row_checkpoint(row, params, fam, result, elapsed, layer)

        # 4) durable, atomic, self-sealed row evidence (generation-M-bound), then the receipt.
        atomic_write_json(self._row_checkpoint_path(row_id), checkpoint)
        self._maybe_kill("after_write", row_id)
        assert self.log is not None
        self.log.append("row_sealed", {"row_id": row_id, "status": checkpoint["status"],
                                        "layer": layer, "row_sha256": checkpoint["row_sha256"]})
        self._maybe_kill("after_receipt", row_id)
        return checkpoint

    def _build_row_checkpoint(self, row: dict[str, Any], params: dict[str, Any], fam: str,
                              result: dict[str, Any], elapsed: float, layer: int) -> dict[str, Any]:
        eb = row.get("exact_budget") or {}
        cp = {
            "schema": ROW_CHECKPOINT_SCHEMA,
            "row_id": row["row_id"],
            "tensor_class": row.get("tensor_class"),
            "representation_family": row.get("representation_family"),
            "family": fam,
            "params": params,
            "rate": eb.get("rate") or row.get("exact_rate"),
            "gate": "G3_cross_layer_transfer",
            "layer": layer,
            "layer_role": row.get("layer_role"),
            "program_sha256": self.program_sha256,
            "execution_generation": EXECUTION_GENERATION,
            "base_execution_provider": row.get("base_execution_provider") or BASE_EXECUTION_PROVIDER,
            "generation_binding": self._generation_binding(),
            "status": result["status"],
            "metrics": result["metrics"],
            "elapsed_seconds": float(elapsed),
            "sealed_at": now_iso(),
            "controller_pid": os.getpid(),
        }
        return seal_field(cp, "row_sha256")

    # -- transfer analysis (the real cross-layer finding) -------------------------------
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
        return (-cos, comb, clb, str(cp.get("row_id")))

    def _sealed_candidates(self) -> list[dict[str, Any]]:
        """All SEALED, within-budget, NON-control checkpoints (the eligible geometries)."""
        out: list[dict[str, Any]] = []
        for row in self.working_rows():
            cp = self.read_row_checkpoint(row["row_id"])
            if cp is None or cp["status"] != STATUS_SEALED:
                continue
            if not cp.get("metrics", {}).get("within_budget", False):
                continue
            if cp.get("family") == CONTROL_FAMILY:
                continue
            out.append(cp)
        return out

    def select_transfer(self) -> dict[str, Any]:
        """Per tensor_class, the winner geometry at each layer (highest hidden cosine within budget,
        controls excluded) + the TRANSFER verdict: does the layer-0 winner still win at 18 / 35?"""
        cands = self._sealed_candidates()
        # group by (tensor_class, layer)
        by_cl: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for cp in cands:
            key = (str(cp.get("tensor_class")), int(cp.get("layer")))
            by_cl.setdefault(key, []).append(cp)

        def entry(cp: dict[str, Any]) -> dict[str, Any]:
            m = cp["metrics"]
            val = m.get("functional", {}).get("validation", {})
            return {"winner_row_id": cp["row_id"], "family": cp.get("family"),
                    "representation_family": cp.get("representation_family"),
                    "params": cp.get("params"),
                    "layer_hidden_state_cosine": val.get("layer_hidden_state_cosine"),
                    "weighted_combine_divergence": val.get("weighted_combine_divergence"),
                    "expert_output_cosine": val.get("expert_output_cosine"),
                    "complete_layer_bpw": m.get("complete_layer_bpw"),
                    "n_candidates": None}

        winners_by_class: dict[str, dict[str, Any]] = {}
        classes = sorted({tc for (tc, _l) in by_cl})
        for tc in classes:
            per_layer: dict[str, Any] = {}
            for layer in LAYERS:
                cps = by_cl.get((tc, layer), [])
                if not cps:
                    continue
                win = min(cps, key=self._winner_key)
                e = entry(win)
                e["n_candidates"] = len(cps)
                e["candidates"] = sorted(c["row_id"] for c in cps)
                # per-family hidden cosine at this layer (the full ranking, for transparency).
                e["family_ranking"] = sorted(
                    ({"family": c.get("family"),
                      "layer_hidden_state_cosine": c.get("metrics", {}).get("functional", {})
                      .get("validation", {}).get("layer_hidden_state_cosine")}
                     for c in cps),
                    key=lambda d: (-(d["layer_hidden_state_cosine"]
                                     if isinstance(d["layer_hidden_state_cosine"], (int, float))
                                     else -math.inf), str(d["family"])))
                per_layer[str(layer)] = e
            if not per_layer:
                continue
            layer0 = per_layer.get(str(LAYERS[0]))
            base_family = layer0["family"] if layer0 else None
            per_layer_family = {lk: v["family"] for lk, v in per_layer.items()}
            transfers = {}
            for layer, role in zip(LAYERS[1:], LAYER_ROLES[1:]):
                lk = str(layer)
                if lk in per_layer and base_family is not None:
                    transfers[role] = bool(per_layer[lk]["family"] == base_family)
            winners_by_class[tc] = {
                "layer0_winner_family": base_family,
                "winner_per_layer": per_layer,
                "winner_family_per_layer": per_layer_family,
                "transfers_to_mid": transfers.get("mid"),
                "transfers_to_late": transfers.get("late"),
                "fully_transfers": (bool(transfers) and all(v for v in transfers.values())),
            }
        doc = {
            "schema": TRANSFER_SCHEMA,
            "gate": "G3_cross_layer_transfer",
            "program_sha256": self.program_sha256,
            "program_path": str(self.program_path),
            "layers": list(LAYERS),
            "layer_roles": dict(zip((str(x) for x in LAYERS), LAYER_ROLES)),
            "ranking_signal": "layer_hidden_state_cosine (highest within budget per (class, layer); "
                              "controls excluded)",
            "generation_binding": self._generation_binding(),
            "transfer_by_tensor_class": winners_by_class,
            "capability_parity": False,
            "honesty": ("generalized-block APPROXIMATION forward; relative orig-vs-packed divergence "
                        "only; sub-bit science Gravity-NEGATIVE; no Escape Receipt / Event Horizon"),
            "selected_at": now_iso(),
        }
        doc = seal_field(doc, "transfer_sha256")
        atomic_write_json(self.transfer_path, doc)
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
                                                "layer": cp.get("layer"),
                                                "row_sha256": cp["row_sha256"], "reconciled": True})
        self.log.append("reconcile", {"reason": reason, "program_sha256": self.program_sha256})

    def _collect_state(self) -> dict[str, Any]:
        working = self.working_rows()
        completed: list[str] = []
        failed: list[str] = []
        elapseds: list[float] = []
        best: dict[str, Any] | None = None
        # frontier keyed by (tensor_class, layer) and per-layer best.
        frontier_by_class_layer: dict[str, dict[str, Any]] = {}
        layers_seen: set[int] = set()
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
            layers_seen.add(int(cp.get("layer", -1)))
            m = cp.get("metrics", {})
            val = m.get("functional", {}).get("validation", {})
            cos = val.get("layer_hidden_state_cosine")
            within = m.get("within_budget", False)
            is_control = cp.get("family") == CONTROL_FAMILY
            if cp["status"] == STATUS_SEALED and within and not is_control \
                    and isinstance(cos, (int, float)):
                cand = {"row_id": row["row_id"], "family": cp.get("family"),
                        "tensor_class": cp.get("tensor_class"), "layer": cp.get("layer"),
                        "layer_role": cp.get("layer_role"),
                        "layer_hidden_state_cosine": float(cos),
                        "weighted_combine_divergence": val.get("weighted_combine_divergence"),
                        "complete_layer_bpw": m.get("complete_layer_bpw"), "rate": cp.get("rate")}
                if best is None or float(cos) > best["layer_hidden_state_cosine"]:
                    best = dict(cand)
                key = f"{cp.get('tensor_class')}@L{cp.get('layer')}"
                prev = frontier_by_class_layer.get(key)
                if prev is None or float(cos) > prev["layer_hidden_state_cosine"]:
                    frontier_by_class_layer[key] = dict(cand)
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
            "layers_touched": sorted(x for x in layers_seen if x >= 0),
            "avg_row_seconds": (round(avg, 3) if avg is not None else None),
            "eta_seconds": (round(eta, 1) if eta is not None else None),
            "best_by_hidden_cosine": best,
            "frontier_by_class_layer": frontier_by_class_layer,
        }

    def _write_cursor(self, current_row: str | None, state_hint: str) -> dict[str, Any]:
        state = self._collect_state()
        hb_at = None
        if self.heartbeat_path.exists():
            try:
                hb_at = read_json_safe(self.heartbeat_path).get("beat_at")
            except EcoError:
                hb_at = None
        current_layer = None
        if current_row is not None:
            for r in self.rows:
                if r["row_id"] == current_row:
                    current_layer = r.get("layer")
                    break
        doc = {
            "schema": CONTROLLER_SCHEMA,
            "gate": "G3_cross_layer_transfer",
            "controller_pid": os.getpid(),
            "process_start_time": self.process_start_time,
            "lease_identity": {"label": LEASE_LABEL, "pid": os.getpid(),
                               "lease_path": str(self.lease_path)},
            "generation_binding": self._generation_binding(),
            "active_generation": EXECUTION_GENERATION,
            "program_sha256": self.program_sha256,
            "program_path": str(self.program_path),
            "campaign_root": str(self.campaign_root),
            "checkpoint_root": str(self.checkpoints_dir),
            "controller_root": str(self.controller_dir),
            "transfer_path": str(self.transfer_path),
            "last_heartbeat": {"path": str(self.heartbeat_path), "beat_at": hb_at},
            "current_row": current_row,
            "current_layer": current_layer,
            "state_hint": state_hint,
            "resource_snapshot": sample_resources(path_for_disk=str(self.campaign_root)),
            "written_at": now_iso(),
            **state,
        }
        doc = seal_field(doc, "checkpoint_sha256")
        atomic_write_json(self.checkpoint_path, doc)
        return doc

    def _emit_heartbeat(self, payload: dict[str, Any]) -> None:
        heartbeat(self.heartbeat_path, {"label": LEASE_LABEL, "campaign": "frontier_g3",
                                        "active_generation": EXECUTION_GENERATION,
                                        "program_sha256": self.program_sha256, **payload})

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
                                                  "active_generation": EXECUTION_GENERATION,
                                                  "max_rows": max_rows})
            self._reconcile(reason="run")
            self._write_cursor(current_row=None, state_hint="starting")
            working = self.working_rows()
            pending = [r for r in working if self.needs_work(r["row_id"])]
            to_do = pending if max_rows is None else pending[: max(0, int(max_rows))]
            if not to_do:
                self.select_transfer()
                doc = self._write_cursor(current_row=None, state_hint="complete")
                self._emit_heartbeat({"phase": "idle_complete"})
                return self._summary(doc, processed=0)
            processed = 0
            for row in to_do:
                self._write_cursor(current_row=row["row_id"], state_hint="running")
                cp = self.process_row(row)
                self._write_cursor(current_row=None, state_hint="running")
                self._emit_heartbeat({"phase": "row_done", "row_id": row["row_id"],
                                      "status": cp["status"], "layer": cp.get("layer")})
                processed += 1
            self.select_transfer()
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
            "layers_touched": cursor_doc["layers_touched"],
            "program_sha256": self.program_sha256,
            "active_generation": EXECUTION_GENERATION,
            "best_by_hidden_cosine": cursor_doc["best_by_hidden_cosine"],
            "frontier_by_class_layer": cursor_doc["frontier_by_class_layer"],
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
        for solo in (self.heartbeat_path, self.lease_path, self.transfer_path):
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
def _config_from_args(args: argparse.Namespace) -> G3Config:
    root = Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT
    program = Path(args.program) if args.program else DEFAULT_PROGRAM_PATH
    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) if args.only else None
    kill_at = os.environ.get("HAWKING_G3_KILL_AT") or None
    if kill_at is not None and kill_at not in KILL_POINTS:
        raise G3Error(f"HAWKING_G3_KILL_AT must be one of {KILL_POINTS}; got {kill_at!r}")
    kill_row = os.environ.get("HAWKING_G3_KILL_ROW") or None
    return G3Config(
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
    ap = argparse.ArgumentParser(description="G3 durable cross-layer-transfer controller (Full Frontier).")
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
        cfg = G3Config(
            campaign_root=Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT,
            program_path=Path(args.program) if args.program else DEFAULT_PROGRAM_PATH,
            manifest_path=args.manifest or DEFAULT_MANIFEST,
            only_rows=(tuple(x.strip() for x in args.only.split(",") if x.strip())
                       if args.only else None))
    controller = G3Controller(cfg)

    if args.command == "reset":
        print(_json.dumps(controller.reset(), indent=2, sort_keys=True))
        return 0
    if args.command == "select":
        controller.load_program()
        print(_json.dumps(controller.select_transfer(), indent=2, sort_keys=True, default=str))
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
