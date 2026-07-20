#!/usr/bin/env python3.12
"""Qwen3-235B-A22B transfer wave - the Qwen analog of the gpt-oss Doctor correction-wave.

This is the controller the overnight supervisor's LAUNCH_QWEN spawns. It transfers the SCIENCE the
gpt-oss 120B campaign sealed (tensor-class-aware allocation, the Vulture champion priors) onto the
real Qwen3-MoE forward and asks the same decisive question at REAL fidelity: does class-aware
allocation beat the conventional control on THIS architecture, and where does the untreated
capability loss live?

Transfer ladder (each non-parent candidate an expert_hook that substitutes reconstructed
gate_proj/up_proj/down_proj into the real Qwen forward). Qwen experts have THREE projections
(SwiGLU: down(silu(gate(x)) * up(x))), so the gpt-oss two-class map is lifted as:
  gpt-oss mlp1 (up/gate)  ->  Qwen gate_proj + up_proj
  gpt-oss mlp2 (down)     ->  Qwen down_proj

  T0_parent            : source-native experts (the divergence + self-PPL reference).
  T1_vulture_champion  : the GPT_OSS_120B_TRANSFER_PRIORS champion, per-class. From
                         best_base_family_per_class if the priors file is present (mlp1 ->
                         gate/up, mlp2 -> down); else gate/up = product_quant, down =
                         pq_protected_islands.
  T2_product_quant     : second-best full-rank family - plain product_quant on all three.
  T3_qwen_organ_alloc  : Qwen-specific 120B-rate challenger. Asymmetric organ allocation exploiting the
                         3-projection structure: the SwiGLU pair (gate/up, robust) gets a tighter
                         product_quant (FEWER bytes); down (sensitive, heavy-tailed) gets looser
                         pq_protected_islands with a larger island budget (MORE bytes). Its expert
                         projection rate is designed below 0.77 BPW; the separate whole-model plan
                         bills attention/embed/head/router/norm too. Distinct mapping from T1/T2.
                         (A cluster-shared codebook is the other Qwen-specific
                         lever but needs whole-layer expert buffering the per-expert hook cannot
                         see; organ allocation is the per-expert form.)
  T4_naive_rvq_control : equal-byte conventional control - naive_rvq on all three.
  T5_qwen_input_realloc: Q2-driven challenger at the same expert rate as T3. Q2 showed gate/up,
                         not down, dominate the first Qwen activation error, so T5 shifts PQ index
                         capacity into gate/up and reduces down codebook cardinality while retaining
                         the 3% protected-row mechanism.

Metrics are identical in shape to the gpt-oss wave: real logit divergence vs the parent (mean
symmetric softmax KL, logit cosine, top-5 overlap, next-token argmax agreement) + candidate PPL +
a three-projection expert BPW audit read back from the byte-accounted gravity_forge packers. This
rate is never mislabeled as whole-model BPW; qwen_bpw_budget.py bills the remaining organs.
Sealed capability thresholds: mean_sym_kl <= 0.10 AND next_token_argmax_agreement >= 0.95 (do NOT
lower after seeing results). Rows are decisive-first: all T0 parent refs, then each candidate
across the 6-prompt Qwen-tokenized holdout, T1 (champion) first.

Durable exactly like gravity_frontier_correction_wave.py: campaign dir QWEN_TRANSFER/, lease
com.hawking.qwen_transfer (refuses while any other heavy lease is live - one-heavy-lease law),
heartbeat, per-row sealed sha256 checkpoints, resume (sealed rows skip), state
QWEN_TRANSFER_STATE.json, run/detach/status CLI, caffeinate detach, PressureAwareCache per candidate.

SOURCE GATE + HONESTY. The 438 GiB Qwen weight set is NOT staged locally (only
models/qwen3-235b-a22b/_meta metadata exists). run() checks source_present() FIRST: if the source is
absent it seals a WAITING_SOURCE state and exits 0 cleanly, so a premature supervisor launch is a
safe no-op, never a crash. Only when the shards are fully present does it take the heavy lease and run
real forwards. The real Qwen forward is therefore UNTESTED-PENDING-SOURCE (see the WAITING_SOURCE
receipt); the tests exercise only synthetic tiny tensors and never run a real forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

import qwen_real_forward as Q
import gravity_forge as gf
import bounded_cache as bc

ROOT = Path(_HERE).resolve().parents[1]
CAMPAIGN = ROOT / "reports/condense/general_frontier/QWEN_TRANSFER"
LEASES = CAMPAIGN / "leases"
HEARTBEAT = CAMPAIGN / "heartbeat"
CHECKPOINTS = CAMPAIGN / "checkpoints"
CONTROLLER = CAMPAIGN / "controller"
STATE_PATH = CAMPAIGN / "QWEN_TRANSFER_STATE.json"
WAITING_RECEIPT = CAMPAIGN / "QWEN_TRANSFER_WAITING_SOURCE.json"
BPW_PLAN_PATH = ROOT / "reports/condense/general_frontier/QWEN3_235B_GRAVITY_BPW_PLAN.json"
LEASE_PATH = LEASES / "qwen_transfer.lease"
HB_PATH = HEARTBEAT / "qwen_transfer.heartbeat.json"
LABEL = "com.hawking.qwen_transfer"
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
PRIORS_PATH = ROOT / "reports/condense/general_frontier/GPT_OSS_120B_TRANSFER_PRIORS.json"
TOKENIZER_PATH = ROOT / "models/qwen3-235b-a22b/_meta/tokenizer.json"
# One-heavy-lease law: refuse if ANY of these (incl the live gpt-oss doctor_campaign) is held live.
OTHER_LEASE_GLOBS = ["reports/condense/general_frontier/DOCTOR_CAMPAIGN/leases/doctor_campaign.lease",
                     "reports/condense/general_frontier/CORRECTION_WAVE/leases/correction_wave.lease",
                     "reports/condense/general_frontier/G4/leases/frontier_g4.lease",
                     "reports/condense/second_light/leases/second_light.lease"]

# Qwen expert projection classes (the substitution seam). gate/up = the SwiGLU pair, down = output.
CLASSES = ("gate", "up", "down")

# 6-prompt holdout (model-agnostic English), tokenized with the QWEN tokenizer at run time.
HOLDOUT: list[dict[str, str]] = [
    {"id": "gen_paris", "domain": "factual", "text": "The capital of France is"},
    {"id": "gen_science", "domain": "general", "text": "Water is made of two hydrogen atoms and one"},
    {"id": "code_py", "domain": "code", "text": "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n -"},
    {"id": "math_add", "domain": "math", "text": "If a train travels 60 miles in 2 hours, its average speed is 30"},
    {"id": "reason_syllogism", "domain": "reasoning", "text": "All humans are mortal. Socrates is a human. Therefore, Socrates is"},
    {"id": "instr_list", "domain": "instruction", "text": "Here are three primary colors: red, green, and"},
]

# G3-winner geometry regime (sub-bit): dim16/k64/subspaces2; islands residual_energy 1%; doctor 0.15bpw.
PARAMS = {"dim": 16, "k": 64, "subspaces": 2, "strategy": "residual_energy",
          "budget_frac": 0.01, "doctor": "residual_codebook", "doctor_bpw": 0.15,
          "stages": 2}

PROMOTE_KL = 0.10
PROMOTE_ARGMAX_AGREE = 0.95
TARGET_WHOLE_BPW = 0.77


def _spec(family: str, **overrides: Any) -> dict[str, Any]:
    """A per-class packing spec: a family plus a params dict (module PARAMS with per-class overrides)."""
    return {"family": family, "params": {**PARAMS, **overrides}}


def _vulture_champion() -> dict[str, dict[str, Any]]:
    """T1 map from the gpt-oss transfer priors: mlp1 -> Qwen gate/up, mlp2 -> Qwen down. Falls back
    to gate/up = product_quant, down = pq_protected_islands when the priors file is absent."""
    f_gateup, f_down = "product_quant", "pq_protected_islands"
    try:
        priors = json.loads(PRIORS_PATH.read_text())
        m = priors.get("best_base_family_per_class") or {}
        f_gateup = m.get("mlp1", f_gateup)
        f_down = m.get("mlp2", f_down)
    except Exception:
        pass
    return {"gate": _spec(f_gateup), "up": _spec(f_gateup), "down": _spec(f_down)}


# The transfer ladder (non-parent candidates). Each maps class -> per-class {family, params}.
CANDIDATES: dict[str, dict[str, Any]] = {
    "T1_vulture_champion": _vulture_champion(),
    "T2_product_quant": {c: _spec("product_quant") for c in CLASSES},
    # T3: gate/up are tight; down keeps most of the joint budget through four subspaces + 3%
    # protected rows. k=32 puts the three projections below the 120B 0.77-BPW construction rate.
    # Non-expert organs are still billed separately by qwen_bpw_budget.py.
    "T3_qwen_organ_alloc": {"gate": _spec("product_quant", dim=32, k=16, subspaces=2),
                            "up": _spec("product_quant", dim=32, k=16, subspaces=2),
                            "down": _spec("pq_protected_islands", dim=16, k=32, subspaces=4,
                                          budget_frac=0.03)},
    # T5: admitted by the first real Qwen organ diagnosis. Same ~0.744 expert BPW as T3, but
    # gate/up rise ~0.251->0.376 each while down falls ~1.732->1.481.
    "T5_qwen_input_realloc": {"gate": _spec("product_quant", dim=32, k=8, subspaces=4),
                              "up": _spec("product_quant", dim=32, k=8, subspaces=4),
                              "down": _spec("pq_protected_islands", dim=16, k=16, subspaces=4,
                                            budget_frac=0.03)},
    "T4_naive_rvq_control": {c: _spec("naive_rvq") for c in CLASSES},
}
# Decisive-first: the champion, then the Qwen challenger, then the full-rank baseline, then control.
CANDIDATE_ORDER = ["T1_vulture_champion", "T3_qwen_organ_alloc", "T5_qwen_input_realloc",
                   "T2_product_quant", "T4_naive_rvq_control"]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    try:
        if pid is None or int(pid) <= 0:   # -1 default / 0 would make os.kill BROADCAST to the group
            return False
        os.kill(int(pid), 0); return True
    except PermissionError:
        return True
    except Exception:
        return False


def _tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_PATH))


# -- faithful forge pack (per-class params) -> recon matrix + realized whole-artifact BPW ----------
def _mps_gc() -> None:
    """Release the MPS/GPU allocator cache accumulated during packing (gravity_forge packs on MPS,
    unified memory). Cheap, fail-closed; keeps a CPU byte budget honest about GPU-side residency."""
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def forge_pack(family: str, w: np.ndarray, seed: int = 0,
               params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pack `w` with `family` using `params` (defaults to module PARAMS). Returns the recon matrix
    (same shape as w) and the packer's byte-accounted whole_artifact_bpw so the class budget is
    auditable. Per-call params may override dim / k / subspaces / strategy / budget_frac / doctor /
    doctor_bpw / stages, letting each projection class spend a different byte budget."""
    p = PARAMS if params is None else params
    w = np.ascontiguousarray(w, dtype=np.float32)
    dim, k, sub = int(p["dim"]), int(p["k"]), int(p["subspaces"])
    if family == "kept_original":
        return {"recon": w.copy(), "whole_bpw": 16.0}
    if family == "product_quant":
        art = gf.pack_product_quant(w, dim=dim, subspaces=sub, k=k, seed=seed)
        return {"recon": np.asarray(art.recon, np.float32).reshape(w.shape),
                "whole_bpw": float(art.whole_artifact_bpw)}
    if family == "naive_rvq":
        art = gf.pack_naive_rvq(w, dim=dim, k=k, stages=int(p.get("stages", 2)), seed=seed)
        return {"recon": np.asarray(art.recon, np.float32).reshape(w.shape),
                "whole_bpw": float(art.whole_artifact_bpw)}
    if family == "pq_protected_islands":
        art = gf.pack_pq_protected_islands(w, dim=dim, subspaces=sub, k=k,
                                           strategy=p.get("strategy", "residual_energy"),
                                           budget_frac=float(p.get("budget_frac", 0.01)), seed=seed)
        return {"recon": np.asarray(art.recon, np.float32).reshape(w.shape),
                "whole_bpw": float(art.whole_artifact_bpw)}
    if family == "pq_doctor_lowrank":
        base = gf.pack_product_quant(w, dim=dim, subspaces=sub, k=k, seed=seed)
        base_recon = np.asarray(base.recon, np.float32)
        byte_budget = max(1, int(round(float(p.get("doctor_bpw", 0.15)) * w.size / 8.0)))
        try:
            doc = gf.doctor_pq(w, base, byte_budget=byte_budget,
                               strategy=p.get("doctor", "residual_codebook"), seed=seed)
            recon = base_recon
            if p.get("doctor", "residual_codebook") == "residual_codebook":
                ev = doc.get("evidence", {})
                s2 = int(ev.get("stage2_subspaces") or sub); k2 = int(ev.get("stage2_k") or k)
                D = int(base.config.get("dim", dim))
                resid = (w - base_recon).astype(np.float32)
                stage2 = gf.pack_product_quant(resid, dim=D, subspaces=s2, k=k2, seed=seed, iters=8)
                recon = (base_recon + np.asarray(stage2.recon, np.float32)).reshape(w.shape)
            physical_bits = int((base.physical_bytes + int(doc["added_bytes"])) * 8)
            return {"recon": np.ascontiguousarray(recon, np.float32),
                    "whole_bpw": physical_bits / max(1, w.size)}
        except AssertionError:
            # doctor budget too tight for this tensor (small-tensor edge); fall back to base PQ.
            return {"recon": np.ascontiguousarray(base_recon.reshape(w.shape), np.float32),
                    "whole_bpw": float(base.whole_artifact_bpw), "doctor_skipped": True}
    raise ValueError(f"unknown family {family}")


def _make_hook(mapping: dict[str, Any], audit: dict[str, Any]):
    """Return an expert_hook(L, e, ex) that substitutes reconstructed gate/up/down per `mapping`.
    The QwenRealForward's own PressureAwareCache dedups post-hook experts, so the hook packs at most
    once per (layer, expert). `audit` accumulates the realized whole-artifact BPW per class (recorded
    once, deterministic per family+params+shape) so the per-expert byte budget is auditable."""
    def hook(L: int, e: int, ex: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(ex)
        for cls in CLASSES:
            spec = mapping[cls]
            packed = forge_pack(spec["family"], ex[cls], seed=L * 1000 + e, params=spec["params"])
            out[cls] = packed["recon"]
            if cls not in audit:
                knobs = {kk: spec["params"][kk] for kk in ("dim", "k", "subspaces", "budget_frac")
                         if kk in spec["params"]}
                audit[cls] = {"family": spec["family"], "whole_bpw": round(float(packed["whole_bpw"]), 5),
                              "n_weights": int(ex[cls].size), "params": knobs}
                if packed.get("doctor_skipped"):
                    audit[cls]["doctor_skipped"] = True
        _mps_gc()
        return out
    return hook


def _budget_from_audit(audit: dict[str, Any]) -> dict[str, Any] | None:
    """Combine the three projection rates. This is explicitly not a whole-model BPW."""
    if not all(c in audit for c in CLASSES):
        return None
    ns = {c: audit[c]["n_weights"] for c in CLASSES}
    bits = sum(audit[c]["whole_bpw"] * ns[c] for c in CLASSES)
    tot = sum(ns.values())
    out = {f"{c}_class": audit[c] for c in CLASSES}
    out["per_expert_projection_bpw"] = round(bits / max(1, tot), 5)
    out["per_expert_whole_bpw"] = out["per_expert_projection_bpw"]  # receipt compatibility
    out["scope"] = "three expert projection tensors only; not whole-model BPW"
    out["meets_120b_numeric_ceiling"] = out["per_expert_projection_bpw"] <= TARGET_WHOLE_BPW
    return out


# -- metrics (identical definitions to the gpt-oss wave) ------------------------------------------
def _logsoftmax(z):
    z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _self_nll(logits, ids):
    ls = _logsoftmax(logits.astype(np.float64))
    nlls = [-ls[i, ids[i + 1]] for i in range(len(ids) - 1)]
    m = float(np.mean(nlls)) if nlls else float("nan")
    return {"nll": round(m, 5), "perplexity": round(float(np.exp(m)), 4), "n_positions": len(nlls)}


def _divergence(orig, packed):
    o, p = orig.astype(np.float64), packed.astype(np.float64)
    lo, lp = _logsoftmax(o), _logsoftmax(p)
    po, pp = np.exp(lo), np.exp(lp)
    sym_kl = 0.5 * ((po * (lo - lp)).sum(-1) + (pp * (lp - lo)).sum(-1))
    cos = (o * p).sum(-1) / (np.linalg.norm(o, axis=-1) * np.linalg.norm(p, axis=-1) + 1e-12)
    k = 5
    to, tp = np.argsort(-o, -1)[:, :k], np.argsort(-p, -1)[:, :k]
    overlap = [len(set(to[i]) & set(tp[i])) / k for i in range(o.shape[0])]
    return {"mean_sym_kl": round(float(sym_kl.mean()), 5),
            "mean_logit_cosine": round(float(cos.mean()), 5),
            "mean_top5_overlap": round(float(np.mean(overlap)), 4),
            "next_token_argmax_agreement": round(float(np.mean(o.argmax(-1) == p.argmax(-1))), 4)}


# -- durability ----------------------------------------------------------------------------------
def _other_heavy_lease_live() -> str | None:
    for rel in OTHER_LEASE_GLOBS:
        pth = ROOT / rel
        if pth.exists():
            try:
                h = json.loads(pth.read_text())
                if _pid_alive(int(h.get("pid", -1))):
                    return f"{rel} pid {h['pid']}"
            except Exception:
                pass
    return None


def _acquire_lease() -> None:
    LEASES.mkdir(parents=True, exist_ok=True)
    other = _other_heavy_lease_live()
    if other:
        raise SystemExit(f"another heavy lease is live ({other}); one-heavy-lease law - refusing to start")
    if LEASE_PATH.exists():
        try:
            held = json.loads(LEASE_PATH.read_text())
            if _pid_alive(int(held.get("pid", -1))):
                raise SystemExit(f"qwen_transfer lease held by live pid {held['pid']}")
        except (json.JSONDecodeError, ValueError):
            pass
    _atomic(LEASE_PATH, {"acquired_at": _now(), "owner": LABEL, "pid": os.getpid(),
                         "schema": "hawking.successor.watchdog_lease.v1"})


def _beat(row_id, phase, done, total):
    import shutil
    _atomic(HB_PATH, {"beat_at": _now(), "label": LABEL, "campaign": "qwen_transfer", "pid": os.getpid(),
                      "row_id": row_id, "phase": phase, "rows_done": done, "rows_total": total,
                      "free_disk_gb": round(shutil.disk_usage(str(ROOT)).free / 1e9, 1),
                      "schema": "hawking.successor.watchdog.v1", "status": "RUNNING"})


def _rows():
    """Decisive-first ordering: all T0 parent-refs, then each candidate across the holdout,
    T1 (champion) first."""
    rows = []
    for h in HOLDOUT:
        rows.append({"row_id": f"{h['id']}__T0_parent", "prompt": h, "candidate": "parent",
                     "kind": "parent"})
    for c in CANDIDATE_ORDER:
        for h in HOLDOUT:
            rows.append({"row_id": f"{h['id']}__{c}", "prompt": h, "candidate": c, "kind": "candidate"})
    return rows


def _parent_forward():
    """Build the source-native forward (metadata only; shards streamed lazily). Isolated so tests can
    monkeypatch a source-absent stub without touching real weights."""
    return Q.from_source()


def _seal_waiting_source() -> int:
    """Source-absent no-op: seal a WAITING_SOURCE state + receipt and exit 0 (never a crash). A
    premature supervisor LAUNCH_QWEN before the 438 GiB shards are staged lands here."""
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "hawking.qwen_transfer.state.v1", "status": "WAITING_SOURCE",
               "generated_at": _now(), "final": False, "rows_done": 0, "rows_total": len(_rows()),
               "gate": "qwen3_235b_transfer_real_forward",
               "reason": "Qwen3-235B source shards not fully present (only _meta metadata staged); "
                         "real forward is untested-pending-source. No lease taken, no forward run.",
               "source_dir": str(ROOT / Q.DEFAULT_SOURCE),
               "candidates": {c: {k: CANDIDATES[c][k]["family"] for k in CLASSES} for c in CANDIDATES},
               "candidate_order": CANDIDATE_ORDER, "params": PARAMS,
               "target_whole_artifact_bpw_ceiling": TARGET_WHOLE_BPW,
               "whole_model_bpw_plan": str(BPW_PLAN_PATH),
               "honesty": "UNTESTED-PENDING-SOURCE; run() is a safe no-op until the weights are staged"}
    _atomic(STATE_PATH, payload)
    _atomic(WAITING_RECEIPT, payload)
    print(json.dumps({"status": "WAITING_SOURCE", "state": str(STATE_PATH)}, indent=2))
    return 0


def run(max_rows: int | None = None) -> int:
    # SOURCE GATE FIRST: absent source is a clean no-op regardless of any other lease.
    fwd_parent = _parent_forward()
    if not fwd_parent.source_present():
        return _seal_waiting_source()
    for d in (LEASES, HEARTBEAT, CHECKPOINTS, CONTROLLER):
        d.mkdir(parents=True, exist_ok=True)
    _acquire_lease()
    tk = _tokenizer()
    rows = _rows()
    total = len(rows)
    orig_cache: dict[str, np.ndarray] = {}
    # At most parent + one candidate forward resident (rows are grouped by candidate).
    cur: dict[str, Any] = {"cand": None, "fwd": None, "hook": None, "audit": None}
    done = processed = 0
    t0 = time.time()
    for row in rows:
        rid = row["row_id"]
        cp = CHECKPOINTS / f"{rid}.json"
        if cp.exists():
            done += 1; continue
        if max_rows is not None and processed >= max_rows:
            break
        h = row["prompt"]; ids = tk.encode(h["text"]).ids
        _beat(rid, "forward", done, total)
        ts = time.time()
        if row["candidate"] == "parent":
            logits = fwd_parent.logits_for(ids, positions="all")
            orig_cache[h["id"]] = logits.astype(np.float32)
            rec = {"variant": "parent", "kind": "parent", "quality": _self_nll(logits, ids)}
        else:
            cand = row["candidate"]
            if cur["cand"] != cand:
                # ponytail: drop the previous candidate's reader/cache so residency stays flat
                # (one candidate at a time); PressureAwareCache also floor-evicts under pressure.
                if cur["fwd"] is not None:
                    try:
                        cur["fwd"].reader.close()
                    except Exception:
                        pass
                audit: dict[str, Any] = {}
                fwd_c = Q.from_source()
                fwd_c.cache = bc.PressureAwareCache(f"qwen_{cand}", disk_path=str(fwd_c.reader.source_dir),
                                                    verbose=False)
                cur = {"cand": cand, "fwd": fwd_c, "hook": _make_hook(CANDIDATES[cand], audit),
                       "audit": audit}
            fwd_c, hook, audit = cur["fwd"], cur["hook"], cur["audit"]
            logits = fwd_c.logits_for(ids, positions="all", expert_hook=hook)
            orig = orig_cache.get(h["id"])
            if orig is None:
                orig = fwd_parent.logits_for(ids, positions="all").astype(np.float32)
                orig_cache[h["id"]] = orig
            div = _divergence(orig, logits)
            rec = {"variant": cand, "kind": "candidate",
                   "mapping": {cls: {"family": CANDIDATES[cand][cls]["family"],
                                     "params": CANDIDATES[cand][cls]["params"]} for cls in CLASSES},
                   "quality": _self_nll(logits, ids), "divergence_vs_parent": div,
                   "budget": _budget_from_audit(audit),
                   "capability_candidate": bool(div["mean_sym_kl"] <= PROMOTE_KL
                                                and div["next_token_argmax_agreement"] >= PROMOTE_ARGMAX_AGREE)}
            rec["verdict"] = ("capability_candidate" if rec["capability_candidate"]
                              else ("degraded" if div["mean_sym_kl"] < 2.0 else "collapse"))
        rec.update({"row_id": rid, "prompt_id": h["id"], "domain": h["domain"],
                    "n_tokens": len(ids), "forward_seconds": round(time.time() - ts, 1),
                    "logits_finite": bool(np.isfinite(logits).all()), "sealed_at": _now(),
                    "params": PARAMS,
                    "honesty": "real full-model from-config Qwen3-MoE forward, UNTESTED-PENDING-SOURCE "
                               "until shards staged; real byte-accounted gravity_forge packers; "
                               "sub-bit science expected negative"})
        rec["sha256"] = _sha({k: v for k, v in rec.items() if k != "sha256"})
        _atomic(cp, rec)
        done += 1; processed += 1
        _beat(rid, "row_done", done, total)
        _write_state(done, total, t0)
        extra = ""
        if row["candidate"] != "parent":
            d_ = rec["divergence_vs_parent"]; b_ = rec.get("budget") or {}
            extra = (f"  symKL={d_['mean_sym_kl']} agree={d_['next_token_argmax_agreement']}"
                     f" bpw={b_.get('per_expert_whole_bpw')} -> {rec['verdict']}")
        print(f"[{done}/{total}] {rid}  {rec['forward_seconds']}s  ppl={rec['quality']['perplexity']}{extra}", flush=True)
    _write_state(done, total, t0, final=True)
    if LEASE_PATH.exists():
        LEASE_PATH.unlink()
    return 0


def _write_state(done, total, t0, final=False):
    seals = sorted(CHECKPOINTS.glob("*.json"))
    res = [json.loads(p.read_text()) for p in seals]
    cand = [r for r in res if r.get("variant") not in ("parent", None)]
    graded = [r for r in cand if r.get("divergence_vs_parent")]
    best = sorted(graded, key=lambda r: r["divergence_vs_parent"]["mean_sym_kl"])[:3]
    elapsed = time.time() - t0
    _atomic(STATE_PATH, {
        "schema": "hawking.qwen_transfer.state.v1", "status": "RUNNING" if not final else "SEALED",
        "generated_at": _now(), "final": final,
        "gate": "qwen3_235b_transfer_real_forward", "rows_done": done, "rows_total": total,
        "candidates": {c: {k: CANDIDATES[c][k]["family"] for k in CLASSES} for c in CANDIDATES},
        "candidate_order": CANDIDATE_ORDER, "params": PARAMS,
        "target_whole_artifact_bpw_ceiling": TARGET_WHOLE_BPW,
        "whole_model_bpw_plan": str(BPW_PLAN_PATH),
        "promote_thresholds": {"mean_sym_kl_max": PROMOTE_KL, "argmax_agreement_min": PROMOTE_ARGMAX_AGREE},
        "least_divergent_candidates": [{"row_id": r["row_id"], "variant": r["variant"],
                                        "mean_sym_kl": r["divergence_vs_parent"]["mean_sym_kl"],
                                        "argmax_agreement": r["divergence_vs_parent"]["next_token_argmax_agreement"],
                                        "per_expert_whole_bpw": (r.get("budget") or {}).get("per_expert_whole_bpw"),
                                        "verdict": r.get("verdict")} for r in best],
        "capability_candidates": [r["row_id"] for r in cand if r.get("capability_candidate")],
        "eta_seconds_remaining": round(elapsed / max(1, done) * (total - done), 0) if not final else 0,
        "honest_note": "transfer of the gpt-oss tensor-class science onto the real Qwen3-MoE forward; "
                       "sub-bit expected NEGATIVE; seals which class-aware allocation loses least AND "
                       "the realized per-expert byte budget. UNTESTED-PENDING-SOURCE until shards staged.",
    })


def detach() -> int:
    CONTROLLER.mkdir(parents=True, exist_ok=True)
    log = CONTROLLER / "detached.log"
    argv = ["caffeinate", "-i", "-s", PY, os.path.abspath(__file__), "run"]
    with open(log, "ab") as lh:
        proc = subprocess.Popen(argv, stdout=lh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(ROOT), start_new_session=True)
    print(json.dumps({"detached": True, "pid": proc.pid, "log": str(log)}, indent=2))
    return 0


def status() -> int:
    st = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"status": "no state"}
    hb = json.loads(HB_PATH.read_text()) if HB_PATH.exists() else {}
    lease = json.loads(LEASE_PATH.read_text()) if LEASE_PATH.exists() else {}
    print(json.dumps({"lease": lease,
                      "lease_pid_alive": _pid_alive(int(lease["pid"])) if lease.get("pid") else False,
                      "heartbeat": hb, "state": st}, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "detach", "status"])
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args(argv)
    return {"run": lambda: run(a.max_rows), "detach": detach, "status": status}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
