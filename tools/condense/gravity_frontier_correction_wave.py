#!/usr/bin/env python3.12
"""Doctor campaign - the final correction-wave ladder on the REAL forward (Part II/6 of the ladder).

G4 measured the source-native parent (real PPL) and the uniform sub-bit RVQ control (D1). This wave
answers the campaign's core question at REAL fidelity: does TENSOR-CLASS-aware and DOCTOR-treated
allocation (mlp1 and mlp2 treated differently, per the G0-G3 evidence) beat the uniform control, and
WHERE does the untreated capability loss actually live (mlp1 vs mlp2)?

Doctor ladder (each candidate an expert_hook substituting reconstructed experts into the real forward):
  D0_parent      : source-native experts (the divergence + PPL reference; 'parent' variant).
  D1_uniform_rvq : uniform naive_rvq control - ALREADY SEALED in G4 (rows *__rvq1.0). Referenced here
                   as the untreated baseline; NOT re-run (see D1_REFERENCE).
  D2_tensor_pq   : mlp1 = product_quant         , mlp2 = pq_protected_islands.
  D4_pq_doctor   : mlp1 = pq_doctor_lowrank      , mlp2 = pq_protected_islands (the G3 winners).
  D6_global_alloc: per-class ASYMMETRIC byte allocation - spend bytes where recovery is highest.
                   mlp1 (robust) gets FEWER bytes (tighter product_quant: larger dim / smaller k),
                   mlp2 (sensitive) gets MORE (looser pq_protected_islands + larger island budget).

Diagnosis (organ isolation, Part II/4 - localise the untreated failure per tensor class), on 2 probe
prompts (gen_paris, reason_syllogism):
  diag_mlp1_only : mlp1 = product_quant , mlp2 = kept_original (attributes loss to mlp1).
  diag_mlp2_only : mlp1 = kept_original , mlp2 = product_quant (attributes loss to mlp2).

Rows are ordered so the DECISIVE evidence seals first: all D0 parent-refs, then diagnosis, then
D4, D6, D2 across the 6-prompt holdout. Each candidate records its realized whole-artifact BPW PER
CLASS (read back from the byte-accounted packers) so the total per-expert budget is auditable.

Real packers are gravity_forge (pack_product_quant / pack_pq_protected_islands / doctor_pq /
pack_naive_rvq), byte-accounted; the reconstruction fed to the forward is bit-consistent with what is
billed. forge_pack accepts an optional per-call params dict (default the module PARAMS) so each tensor
class can use a different dim / k / subspaces / budget_frac. Metrics are the real end-to-end
divergence vs the parent (symmetric softmax KL, logit cosine, top-5 overlap, next-token argmax
agreement) plus candidate PPL.

Durable (lease com.hawking.doctor_campaign, heartbeat, per-row sealed checkpoints, resume, caffeinate
detach). Refuses to start while any other heavy lease is live (one-heavy-lease law). Bounded-memory:
a PressureAwareCache per candidate (block,expert)->recon. Expected result: sub-bit still NEGATIVE, but
the wave seals WHICH tensor-class treatment loses least AND which organ carries the loss - the
evidence the adaptive G5 allocator consumes. No capability claim unless sealed thresholds are met.
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

import gptoss_real_forward as rf
import gravity_forge as gf
import bounded_cache as bc

ROOT = Path(_HERE).resolve().parents[1]
CAMPAIGN = ROOT / "reports/condense/general_frontier/DOCTOR_CAMPAIGN"
LEASES = CAMPAIGN / "leases"
HEARTBEAT = CAMPAIGN / "heartbeat"
CHECKPOINTS = CAMPAIGN / "checkpoints"
CONTROLLER = CAMPAIGN / "controller"
STATE_PATH = CAMPAIGN / "DOCTOR_CAMPAIGN_STATE.json"
LEASE_PATH = LEASES / "doctor_campaign.lease"
HB_PATH = HEARTBEAT / "doctor_campaign.heartbeat.json"
LABEL = "com.hawking.doctor_campaign"
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
OTHER_LEASE_GLOBS = ["reports/condense/general_frontier/G4/leases/frontier_g4.lease",
                     "reports/condense/general_frontier/CORRECTION_WAVE/leases/correction_wave.lease",
                     "reports/condense/second_light/leases/second_light.lease"]

# Reuse the exact G4 holdout (same prompts + tokenization = valid equal-prompt comparison).
HOLDOUT = rf and __import__("gravity_frontier_g4_controller").HOLDOUT  # single source of truth
# Diagnosis probes: two prompts that carry a crisp next-token capability (factual + reasoning).
PROBE_IDS = ["gen_paris", "reason_syllogism"]

# G3-winner geometry regime (sub-bit): dim16/k64/subspaces2; islands residual_energy 1%; doctor 0.15bpw
PARAMS = {"dim": 16, "k": 64, "subspaces": 2, "strategy": "residual_energy",
          "budget_frac": 0.01, "doctor": "residual_codebook", "doctor_bpw": 0.15,
          "stages": 2}

# D1 = uniform naive_rvq control, ALREADY SEALED in G4 (do not re-run here).
D1_REFERENCE = {"candidate": "D1_uniform_rvq", "status": "sealed_in_G4",
                "family": "naive_rvq (uniform, both classes)", "rate_bpw": 1.0,
                "g4_checkpoints_glob": "reports/condense/general_frontier/G4/checkpoints/*__rvq1.0.json",
                "note": "the untreated control; referenced, not recomputed"}


def _spec(family: str, **overrides: Any) -> dict[str, Any]:
    """A per-class packing spec: a family plus a params dict (module PARAMS with per-class overrides)."""
    return {"family": family, "params": {**PARAMS, **overrides}}


# The Doctor ladder (decisive candidates). Each maps tensor class -> per-class {family, params}.
CANDIDATES: dict[str, dict[str, Any]] = {
    "D2_tensor_pq": {"mlp1": _spec("product_quant"),
                     "mlp2": _spec("pq_protected_islands")},
    "D4_pq_doctor": {"mlp1": _spec("pq_doctor_lowrank"),
                     "mlp2": _spec("pq_protected_islands")},
    # D6: asymmetric global allocation. mlp1 (robust) tighter -> FEWER bytes (dim 16->32, k 64->16);
    # mlp2 (sensitive) looser -> MORE bytes (subspaces 2->4, island budget_frac 0.01->0.03).
    "D6_global_alloc": {"mlp1": _spec("product_quant", dim=32, k=16, subspaces=2),
                        "mlp2": _spec("pq_protected_islands", dim=16, k=64, subspaces=4,
                                      budget_frac=0.03)},
}
# Organ-isolation diagnosis: pack one class, keep the other exact, to attribute capability loss.
DIAGNOSIS: dict[str, dict[str, Any]] = {
    "diag_mlp1_only": {"mlp1": _spec("product_quant"), "mlp2": _spec("kept_original")},
    "diag_mlp2_only": {"mlp1": _spec("kept_original"), "mlp2": _spec("product_quant")},
}
ALL_MAPPINGS: dict[str, dict[str, Any]] = {**DIAGNOSIS, **CANDIDATES}
# Order candidate rows so the most decisive evidence seals first.
CANDIDATE_ORDER = ["D4_pq_doctor", "D6_global_alloc", "D2_tensor_pq"]

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


# ── faithful forge pack (per-class params) -> recon matrix + realized whole-artifact BPW ────────
def forge_pack(family: str, w: np.ndarray, seed: int = 0,
               params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pack `w` with `family` using `params` (defaults to module PARAMS). Returns the recon matrix
    (same shape as w) and the packer's byte-accounted whole_artifact_bpw so the class budget is
    auditable. Any per-call params dict may override dim / k / subspaces / strategy / budget_frac /
    doctor / doctor_bpw / stages, letting each tensor class spend a different byte budget."""
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


def _make_hook(mapping: dict[str, Any], cache, audit: dict[str, Any]):
    """Decoded experts live in a PressureAwareCache (grow to fill RAM + swap, evict LRU only under
    genuine pressure). `mapping` gives a per-class {family, params}; `audit` accumulates the realized
    whole-artifact BPW per class (recorded once, deterministic per family+params+shape) so the
    candidate's total per-expert byte budget is auditable from the sealed checkpoints."""
    def hook(block: int, expert: int, ex: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ck = (block, expert)
        hit = cache.get(ck)
        if hit is not None:
            out = dict(ex); out["mlp1"], out["mlp2"] = hit; return out
        out = dict(ex)
        for cls in ("mlp1", "mlp2"):
            spec = mapping[cls]
            packed = forge_pack(spec["family"], ex[cls], seed=block * 1000 + expert,
                                params=spec["params"])
            out[cls] = packed["recon"]
            if cls not in audit:
                knobs = {kk: spec["params"][kk] for kk in ("dim", "k", "subspaces", "budget_frac")
                         if kk in spec["params"]}
                audit[cls] = {"family": spec["family"], "whole_bpw": round(float(packed["whole_bpw"]), 5),
                              "n_weights": int(ex[cls].size), "params": knobs}
                if packed.get("doctor_skipped"):
                    audit[cls]["doctor_skipped"] = True
        cache.put(ck, (out["mlp1"], out["mlp2"]))
        return out
    return hook


def _budget_from_audit(audit: dict[str, Any]) -> dict[str, Any] | None:
    """Combine the two per-class realized BPWs into a per-expert whole-artifact BPW (size-weighted)."""
    if "mlp1" not in audit or "mlp2" not in audit:
        return None
    n1, n2 = audit["mlp1"]["n_weights"], audit["mlp2"]["n_weights"]
    bits = audit["mlp1"]["whole_bpw"] * n1 + audit["mlp2"]["whole_bpw"] * n2
    return {"mlp1_class": audit["mlp1"], "mlp2_class": audit["mlp2"],
            "per_expert_whole_bpw": round(bits / max(1, n1 + n2), 5)}


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
                raise SystemExit(f"doctor_campaign lease held by live pid {held['pid']}")
        except (json.JSONDecodeError, ValueError):
            pass
    _atomic(LEASE_PATH, {"acquired_at": _now(), "owner": LABEL, "pid": os.getpid(),
                         "schema": "hawking.successor.watchdog_lease.v1"})


def _beat(row_id, phase, done, total):
    import shutil
    _atomic(HB_PATH, {"beat_at": _now(), "label": LABEL, "campaign": "doctor_campaign", "pid": os.getpid(),
                      "row_id": row_id, "phase": phase, "rows_done": done, "rows_total": total,
                      "free_disk_gb": round(shutil.disk_usage(str(ROOT)).free / 1e9, 1),
                      "schema": "hawking.successor.watchdog.v1", "status": "RUNNING"})


def _rows():
    """Decisive-first ordering: all D0 parent-refs, then diagnosis (2 probes), then D4, D6, D2."""
    by_id = {h["id"]: h for h in HOLDOUT}
    rows = []
    # 1) D0 parent references for every holdout prompt (the divergence + PPL reference).
    for h in HOLDOUT:
        rows.append({"row_id": f"{h['id']}__D0_parent", "prompt": h, "candidate": "parent",
                     "kind": "parent"})
    # 2) Organ-isolation diagnosis on the two probe prompts.
    for dc in ("diag_mlp1_only", "diag_mlp2_only"):
        for pid in PROBE_IDS:
            rows.append({"row_id": f"{by_id[pid]['id']}__{dc}", "prompt": by_id[pid],
                         "candidate": dc, "kind": "diagnosis"})
    # 3) Decisive candidates across the 6-prompt holdout, D4 -> D6 -> D2.
    for c in CANDIDATE_ORDER:
        for h in HOLDOUT:
            rows.append({"row_id": f"{h['id']}__{c}", "prompt": h, "candidate": c, "kind": "candidate"})
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
    # One PressureAwareCache + one persistent per-class byte audit per hooked candidate.
    expert_caches = {c: bc.PressureAwareCache(f"doc_{c}", disk_path=str(ROOT)) for c in ALL_MAPPINGS}
    expert_audits: dict[str, dict[str, Any]] = {c: {} for c in ALL_MAPPINGS}
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
            rec = {"variant": "parent", "kind": "parent", "quality": _self_nll(logits, ids)}
        else:
            cand = row["candidate"]
            mapping = ALL_MAPPINGS[cand]
            audit = expert_audits[cand]
            hook = _make_hook(mapping, expert_caches[cand], audit)
            logits = fwd.logits_for(ids, positions="all", expert_hook=hook)
            orig = orig_cache.get(h["id"])
            if orig is None:
                orig = fwd.logits_for(ids, positions="all").astype(np.float32)
                orig_cache[h["id"]] = orig
            div = _divergence(orig, logits)
            rec = {"variant": cand, "kind": row["kind"],
                   "mapping": {cls: {"family": mapping[cls]["family"],
                                     "params": mapping[cls]["params"]} for cls in ("mlp1", "mlp2")},
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
                    "honesty": "real full-model from-config forward (coherence-validated, not bit-exact HF); real byte-accounted gravity_forge packers; sub-bit science expected negative"})
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
    cand = [r for r in res if r["variant"] not in ("parent",)]
    graded = [r for r in cand if r.get("divergence_vs_parent")]
    best = sorted(graded, key=lambda r: r["divergence_vs_parent"]["mean_sym_kl"])[:3]
    diag = [r for r in cand if r.get("kind") == "diagnosis"]
    elapsed = time.time() - t0
    _atomic(STATE_PATH, {
        "schema": "hawking.doctor_campaign.state.v1", "generated_at": _now(), "final": final,
        "gate": "doctor_campaign_tensor_class_real_forward", "rows_done": done, "rows_total": total,
        "candidates": {c: {cls: ALL_MAPPINGS[c][cls]["family"] for cls in ("mlp1", "mlp2")}
                       for c in ALL_MAPPINGS},
        "candidate_order": CANDIDATE_ORDER, "params": PARAMS, "d1_reference": D1_REFERENCE,
        "diagnosis": [{"row_id": r["row_id"], "variant": r["variant"],
                       "mean_sym_kl": r["divergence_vs_parent"]["mean_sym_kl"],
                       "argmax_agreement": r["divergence_vs_parent"]["next_token_argmax_agreement"],
                       "verdict": r.get("verdict")} for r in diag if r.get("divergence_vs_parent")],
        "least_divergent_candidates": [{"row_id": r["row_id"], "variant": r["variant"],
                                        "mean_sym_kl": r["divergence_vs_parent"]["mean_sym_kl"],
                                        "argmax_agreement": r["divergence_vs_parent"]["next_token_argmax_agreement"],
                                        "per_expert_whole_bpw": (r.get("budget") or {}).get("per_expert_whole_bpw"),
                                        "verdict": r.get("verdict")} for r in best],
        "capability_candidates": [r["row_id"] for r in cand if r.get("capability_candidate")],
        "eta_seconds_remaining": round(elapsed / max(1, done) * (total - done), 0) if not final else 0,
        "honest_note": "tensor-class + doctor real-forward evidence for the adaptive G5 allocator; "
                       "sub-bit expected NEGATIVE; this seals WHICH treatment loses least per class, "
                       "the per-expert byte budget realized, and (via diagnosis) which organ carries the loss.",
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
