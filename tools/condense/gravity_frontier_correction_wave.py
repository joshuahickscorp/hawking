#!/usr/bin/env python3.12
"""Correction wave - tensor-class-aware candidates on the REAL forward (Part II/6 of the ladder).

G4 measured the source-native parent (real PPL) and a uniform sub-bit RVQ control. This wave answers
the campaign's core question at REAL fidelity: does TENSOR-CLASS-aware allocation (mlp1 and mlp2
treated differently, per the G0-G3 evidence) beat uniform RVQ?

Candidates (each an expert_hook that substitutes reconstructed experts into gptoss_real_forward):
  C2_tensor_class : mlp1 = product_quant        , mlp2 = pq_protected_islands
  C3_g3_winners   : mlp1 = pq_doctor_lowrank     , mlp2 = pq_protected_islands
(C0 source-native and C1 uniform RVQ are the G4 rows; the parent logits are recomputed here as the
divergence reference.)

Real packers are gravity_forge (gf.pack_product_quant / pack_pq_protected_islands / doctor_pq),
byte-accounted; the reconstruction fed to the forward is bit-consistent with what is billed. Metrics
are the real end-to-end divergence vs the parent (logit KL, cosine, top-5 overlap, next-token argmax
agreement) plus candidate PPL and delta.

Durable (lease com.hawking.correction_wave, heartbeat, per-row sealed checkpoints, resume, caffeinate
detach). Refuses to start while any other heavy lease is live (one-heavy-lease law). Bounded-memory:
shared (block,expert,candidate)->recon cache. Expected result: sub-bit still NEGATIVE, but the wave
seals WHICH tensor-class treatment loses least - the evidence the adaptive G5 allocator consumes. No
capability claim unless sealed thresholds are met.
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
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gptoss_real_forward as rf
import gravity_forge as gf
import bounded_cache as bc

ROOT = Path(_HERE).resolve().parents[1]
CAMPAIGN = ROOT / "reports/condense/general_frontier/CORRECTION_WAVE"
LEASES = CAMPAIGN / "leases"
HEARTBEAT = CAMPAIGN / "heartbeat"
CHECKPOINTS = CAMPAIGN / "checkpoints"
CONTROLLER = CAMPAIGN / "controller"
STATE_PATH = CAMPAIGN / "CORRECTION_WAVE_STATE.json"
LEASE_PATH = LEASES / "correction_wave.lease"
HB_PATH = HEARTBEAT / "correction_wave.heartbeat.json"
LABEL = "com.hawking.correction_wave"
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
OTHER_LEASE_GLOBS = ["reports/condense/general_frontier/G4/leases/frontier_g4.lease",
                     "reports/condense/second_light/leases/second_light.lease"]

# Reuse the exact G4 holdout (same prompts + tokenization = valid equal-prompt comparison).
HOLDOUT = rf and __import__("gravity_frontier_g4_controller").HOLDOUT  # single source of truth

# G3-winner geometry regime (sub-bit): dim16/k64/subspaces2; islands residual_energy 1%; doctor 0.15bpw
PARAMS = {"dim": 16, "k": 64, "subspaces": 2, "strategy": "residual_energy",
          "budget_frac": 0.01, "doctor": "residual_codebook", "doctor_bpw": 0.15}
CANDIDATES = {
    "C2_tensor_class": {"mlp1": "product_quant", "mlp2": "pq_protected_islands"},
    "C3_g3_winners": {"mlp1": "pq_doctor_lowrank", "mlp2": "pq_protected_islands"},
}
PROMOTE_KL = 0.10
PROMOTE_ARGMAX_AGREE = 0.95


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
        os.kill(pid, 0); return True
    except PermissionError:
        return True
    except Exception:
        return False


def _tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(ROOT / "models/gpt-oss-120b/tokenizer.json"))


# ── faithful forge pack (mirrors gravity_frontier_controller._forge_pack) -> recon matrix ──────
def forge_pack(family: str, w: np.ndarray, seed: int = 0) -> dict[str, Any]:
    w = np.ascontiguousarray(w, dtype=np.float32)
    dim, k, sub = PARAMS["dim"], PARAMS["k"], PARAMS["subspaces"]
    if family == "kept_original":
        return {"recon": w.copy(), "whole_bpw": 16.0}
    if family == "product_quant":
        art = gf.pack_product_quant(w, dim=dim, subspaces=sub, k=k, seed=seed)
        return {"recon": np.asarray(art.recon, np.float32).reshape(w.shape), "whole_bpw": float(art.whole_artifact_bpw)}
    if family == "naive_rvq":
        art = gf.pack_naive_rvq(w, dim=dim, k=k, stages=2, seed=seed)
        return {"recon": np.asarray(art.recon, np.float32).reshape(w.shape), "whole_bpw": float(art.whole_artifact_bpw)}
    if family == "pq_protected_islands":
        art = gf.pack_pq_protected_islands(w, dim=dim, subspaces=sub, k=k,
                                           strategy=PARAMS["strategy"], budget_frac=PARAMS["budget_frac"], seed=seed)
        return {"recon": np.asarray(art.recon, np.float32).reshape(w.shape), "whole_bpw": float(art.whole_artifact_bpw)}
    if family == "pq_doctor_lowrank":
        base = gf.pack_product_quant(w, dim=dim, subspaces=sub, k=k, seed=seed)
        base_recon = np.asarray(base.recon, np.float32)
        byte_budget = max(1, int(round(PARAMS["doctor_bpw"] * w.size / 8.0)))
        try:
            doc = gf.doctor_pq(w, base, byte_budget=byte_budget, strategy=PARAMS["doctor"], seed=seed)
            recon = base_recon
            if PARAMS["doctor"] == "residual_codebook":
                ev = doc.get("evidence", {})
                s2 = int(ev.get("stage2_subspaces") or sub); k2 = int(ev.get("stage2_k") or k)
                D = int(base.config.get("dim", dim))
                resid = (w - base_recon).astype(np.float32)
                stage2 = gf.pack_product_quant(resid, dim=D, subspaces=s2, k=k2, seed=seed, iters=8)
                recon = (base_recon + np.asarray(stage2.recon, np.float32)).reshape(w.shape)
            physical_bits = int((base.physical_bytes + int(doc["added_bytes"])) * 8)
            return {"recon": np.ascontiguousarray(recon, np.float32), "whole_bpw": physical_bits / max(1, w.size)}
        except AssertionError:
            # doctor budget too tight for this tensor (small-tensor edge); fall back to base PQ.
            return {"recon": np.ascontiguousarray(base_recon.reshape(w.shape), np.float32),
                    "whole_bpw": float(base.whole_artifact_bpw), "doctor_skipped": True}
    raise ValueError(f"unknown family {family}")


def _make_hook(mapping: dict[str, str], cache):
    """Decoded experts live in a PressureAwareCache (grow to fill RAM + swap, evict LRU only under
    genuine pressure). Two candidate caches are live at once; each self-limits on the shared floor."""
    def hook(block: int, expert: int, ex: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ck = (block, expert)
        hit = cache.get(ck)
        if hit is not None:
            out = dict(ex); out["mlp1"], out["mlp2"] = hit; return out
        out = dict(ex)
        out["mlp1"] = forge_pack(mapping["mlp1"], ex["mlp1"], seed=block * 1000 + expert)["recon"]
        out["mlp2"] = forge_pack(mapping["mlp2"], ex["mlp2"], seed=block * 1000 + expert)["recon"]
        cache.put(ck, (out["mlp1"], out["mlp2"]))
        return out
    return hook


# ── metrics (identical definitions to G4) ──────────────────────────────────────────────────
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


# ── durability ──────────────────────────────────────────────────────────────────────────────
def _other_heavy_lease_live() -> str | None:
    for rel in OTHER_LEASE_GLOBS:
        p = ROOT / rel
        if p.exists():
            try:
                h = json.loads(p.read_text())
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
                raise SystemExit(f"correction_wave lease held by live pid {held['pid']}")
        except (json.JSONDecodeError, ValueError):
            pass
    _atomic(LEASE_PATH, {"acquired_at": _now(), "owner": LABEL, "pid": os.getpid(),
                         "schema": "hawking.successor.watchdog_lease.v1"})


def _beat(row_id, phase, done, total):
    import shutil
    _atomic(HB_PATH, {"beat_at": _now(), "label": LABEL, "campaign": "correction_wave", "pid": os.getpid(),
                      "row_id": row_id, "phase": phase, "rows_done": done, "rows_total": total,
                      "free_disk_gb": round(shutil.disk_usage(str(ROOT)).free / 1e9, 1),
                      "schema": "hawking.successor.watchdog.v1", "status": "RUNNING"})


def _rows():
    rows = []
    for h in HOLDOUT:
        rows.append({"row_id": f"{h['id']}__parent_ref", "prompt": h, "candidate": "parent"})
        for c in CANDIDATES:
            rows.append({"row_id": f"{h['id']}__{c}", "prompt": h, "candidate": c})
    return rows


def run(max_rows: int | None = None) -> int:
    for d in (LEASES, HEARTBEAT, CHECKPOINTS, CONTROLLER):
        d.mkdir(parents=True, exist_ok=True)
    _acquire_lease()
    tk = _tokenizer()
    fwd = rf.RealForward()
    if not fwd.source_present():
        raise SystemExit("120B source absent")
    rows = _rows()
    total = len(rows)
    orig_cache: dict[str, np.ndarray] = {}
    expert_caches = {c: bc.PressureAwareCache(f"cw_{c}", disk_path=str(ROOT)) for c in CANDIDATES}
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
            logits = fwd.logits_for(ids, positions="all")
            orig_cache[h["id"]] = logits.astype(np.float32)
            rec = {"variant": "parent", "quality": _self_nll(logits, ids)}
        else:
            hook = _make_hook(CANDIDATES[row["candidate"]], expert_caches[row["candidate"]])
            logits = fwd.logits_for(ids, positions="all", expert_hook=hook)
            orig = orig_cache.get(h["id"])
            if orig is None:
                orig = fwd.logits_for(ids, positions="all").astype(np.float32)
                orig_cache[h["id"]] = orig
            div = _divergence(orig, logits)
            rec = {"variant": row["candidate"], "mapping": CANDIDATES[row["candidate"]],
                   "quality": _self_nll(logits, ids), "divergence_vs_parent": div,
                   "capability_candidate": bool(div["mean_sym_kl"] <= PROMOTE_KL and div["next_token_argmax_agreement"] >= PROMOTE_ARGMAX_AGREE)}
            rec["verdict"] = ("capability_candidate" if rec["capability_candidate"]
                              else ("degraded" if div["mean_sym_kl"] < 2.0 else "collapse"))
        rec.update({"row_id": rid, "prompt_id": h["id"], "domain": h["domain"],
                    "n_tokens": len(ids), "forward_seconds": round(time.time() - ts, 1),
                    "logits_finite": bool(np.isfinite(logits).all()), "sealed_at": _now(),
                    "params": PARAMS,
                    "honesty": "real full-model from-config forward (coherence-validated, not bit-exact HF); real byte-accounted gravity_forge packers; sub-bit science expected negative"})
        rec["sha256"] = _sha({k: v for k, v in rec.items() if k != "sha256"})
        _atomic(cp, rec)
        done += 1; processed += 1
        _beat(rid, "row_done", done, total)
        _write_state(done, total, t0)
        extra = ""
        if row["candidate"] != "parent":
            d_ = rec["divergence_vs_parent"]
            extra = f"  symKL={d_['mean_sym_kl']} agree={d_['next_token_argmax_agreement']} -> {rec['verdict']}"
        print(f"[{done}/{total}] {rid}  {rec['forward_seconds']}s  ppl={rec['quality']['perplexity']}{extra}", flush=True)
    _write_state(done, total, t0, final=True)
    if LEASE_PATH.exists():
        LEASE_PATH.unlink()
    return 0


def _write_state(done, total, t0, final=False):
    seals = sorted(CHECKPOINTS.glob("*.json"))
    res = [json.loads(p.read_text()) for p in seals]
    cand = [r for r in res if r["variant"] not in ("parent",)]
    best = sorted([r for r in cand if r.get("divergence_vs_parent")],
                  key=lambda r: r["divergence_vs_parent"]["mean_sym_kl"])[:3]
    elapsed = time.time() - t0
    _atomic(STATE_PATH, {
        "schema": "hawking.correction_wave.state.v1", "generated_at": _now(), "final": final,
        "gate": "correction_wave_tensor_class_real_forward", "rows_done": done, "rows_total": total,
        "candidates": CANDIDATES, "params": PARAMS,
        "least_divergent_candidates": [{"row_id": r["row_id"], "variant": r["variant"],
                                        "mean_sym_kl": r["divergence_vs_parent"]["mean_sym_kl"],
                                        "argmax_agreement": r["divergence_vs_parent"]["next_token_argmax_agreement"],
                                        "verdict": r.get("verdict")} for r in best],
        "capability_candidates": [r["row_id"] for r in cand if r.get("capability_candidate")],
        "eta_seconds_remaining": round(elapsed / max(1, done) * (total - done), 0) if not final else 0,
        "honest_note": "tensor-class real-forward evidence for the adaptive G5 allocator; sub-bit expected NEGATIVE, this seals WHICH treatment loses least per tensor class.",
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
    print(json.dumps({"lease": lease, "lease_pid_alive": _pid_alive(int(lease["pid"])) if lease.get("pid") else False,
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
