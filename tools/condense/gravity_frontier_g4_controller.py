#!/usr/bin/env python3.12
"""G4 - short end-to-end REAL-forward gate for GPT-OSS-120B (Gravity General Frontier).

The first campaign gate that leaves proxy fidelity. G0-G3 measured only RELATIVE orig-vs-packed
divergence on synthetic tokens with a non-parity activation. G4 runs the validated real
full-model forward (gptoss_real_forward.RealForward; coherent next-token proven, e.g.
"The capital of France is" -> " Paris") and measures REAL capability metrics on a sealed holdout
of REAL tokenizer prompts:

  * ORIGINAL parent (source-native experts): real logits + self-NLL/perplexity. This parent PPL
    is the reference the quality contract requires ("quality = output-space ppl vs the f16 parent")
    and has NEVER been measured in this campaign.
  * PACKED (sub-bit RVQ, gptoss_subbit_packer) at a rate ladder: real logits + PPL, and the real
    original-vs-packed divergence (logit cosine, symmetric softmax KL, top-5 overlap, next-token
    argmax agreement, PPL delta).

Durable: singleton lease, heartbeat, per-row sealed checkpoints, resume, caffeinate detach.
Bounded-memory: streams one block + a per-block expert cache; ~65 GB read per forward, no offload.

HONESTY. Still a from-config forward (parity validated empirically by coherence, not bit-exact vs
HF). The packed challenger here is naive sub-bit RVQ - a LOWER bound; the stronger G3-winner
families (pq_doctor_lowrank / pq_protected_islands) are the intended next rows. No capability pass
is claimed unless the sealed thresholds are met. Do not lower a threshold after seeing results.
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
import gptoss_subbit_packer as pk
import bounded_cache as bc

ROOT = Path(_HERE).resolve().parents[1]
CAMPAIGN = ROOT / "reports/condense/general_frontier/G4"
LEASES = CAMPAIGN / "leases"
HEARTBEAT = CAMPAIGN / "heartbeat"
CHECKPOINTS = CAMPAIGN / "checkpoints"
CONTROLLER = CAMPAIGN / "controller"
STATE_PATH = CAMPAIGN / "G4_REAL_FORWARD_STATE.json"
LEASE_PATH = LEASES / "frontier_g4.lease"
HB_PATH = HEARTBEAT / "frontier_g4.heartbeat.json"
LABEL = "com.hawking.frontier_g4"
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"

# Sealed holdout: real prompts spanning general/factual/code/math/reasoning/instruction. Short so a
# full 36-block forward stays ~3-4 min. Frozen; the token ids are the sealed measurement surface.
HOLDOUT: list[dict[str, str]] = [
    {"id": "gen_paris", "domain": "factual", "text": "The capital of France is"},
    {"id": "gen_science", "domain": "general", "text": "Water is made of two hydrogen atoms and one"},
    {"id": "code_py", "domain": "code", "text": "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n -"},
    {"id": "math_add", "domain": "math", "text": "If a train travels 60 miles in 2 hours, its average speed is 30"},
    {"id": "reason_syllogism", "domain": "reasoning", "text": "All humans are mortal. Socrates is a human. Therefore, Socrates is"},
    {"id": "instr_list", "domain": "instruction", "text": "Here are three primary colors: red, green, and"},
]
RATE_LADDER = [1.0]                   # bpw; sub-bit thesis rate (RAM-bounded shared cache ~30GB).
                                      # 0.5 and 3.0 (and G3-winner families) are the intended next rows.
PROMOTE_KL = 0.10                      # sealed: mean symmetric KL below this = candidate capability
PROMOTE_ARGMAX_AGREE = 0.95           # sealed: next-token argmax agreement at/above this


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
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return isinstance(sys.exc_info()[1], PermissionError)
    except Exception:
        return False


def _tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(ROOT / "models/gpt-oss-120b/tokenizer.json"))


# ── metrics ───────────────────────────────────────────────────────────────────────────────
def _logsoftmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _self_nll(logits: np.ndarray, token_ids: list[int]) -> dict[str, float]:
    """logits:[seq,vocab] teacher-forced. NLL of tokens[i+1] under logits[i], i=0..seq-2."""
    ls = _logsoftmax(logits.astype(np.float64))
    nlls = [-ls[i, token_ids[i + 1]] for i in range(len(token_ids) - 1)]
    m = float(np.mean(nlls)) if nlls else float("nan")
    return {"nll": round(m, 5), "perplexity": round(float(np.exp(m)), 4), "n_positions": len(nlls)}


def _divergence(orig: np.ndarray, packed: np.ndarray) -> dict[str, float]:
    """Per-position original-vs-packed real-logit divergence, averaged over positions."""
    o = orig.astype(np.float64); p = packed.astype(np.float64)
    lo, lp = _logsoftmax(o), _logsoftmax(p)
    po, pp = np.exp(lo), np.exp(lp)
    kl_op = (po * (lo - lp)).sum(axis=-1)
    kl_po = (pp * (lp - lo)).sum(axis=-1)
    sym_kl = 0.5 * (kl_op + kl_po)
    cos = (o * p).sum(-1) / (np.linalg.norm(o, axis=-1) * np.linalg.norm(p, axis=-1) + 1e-12)
    k = 5
    topo = np.argsort(-o, axis=-1)[:, :k]; topp = np.argsort(-p, axis=-1)[:, :k]
    overlap = [len(set(topo[i]) & set(topp[i])) / k for i in range(o.shape[0])]
    argmax_agree = float(np.mean(o.argmax(-1) == p.argmax(-1)))
    return {"mean_sym_kl": round(float(sym_kl.mean()), 5),
            "mean_logit_cosine": round(float(cos.mean()), 5),
            "mean_top5_overlap": round(float(np.mean(overlap)), 4),
            "next_token_argmax_agreement": round(argmax_agree, 4)}


# ── packed expert hook ──────────────────────────────────────────────────────────────────────
def _make_hook(target_bpw: float, cache):
    """Return expert_hook(block, expert, ex) that RVQ-packs mlp1/mlp2 at target_bpw and decodes back
    (real sub-bit roundtrip). Decoded experts live in a PressureAwareCache keyed by
    (block,expert,rate) and are reused across forwards (each distinct expert packed once). The cache
    grows to fill RAM + swap and evicts LRU only under genuine memory/disk pressure - use as much as
    the box gives, never OOM-crash (the earlier fixed-cap over-evicted and re-packed; the earlier
    unbounded cache OOM-killed code_py)."""
    def hook(block: int, expert: int, ex: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ck = (block, expert, target_bpw)
        hit = cache.get(ck)
        if hit is not None:
            out = dict(ex); out["mlp1"], out["mlp2"] = hit
            return out
        out = dict(ex)
        for key in ("mlp1", "mlp2"):
            w = ex[key]
            dim, k, stages = pk.pick_config(w.shape[1], target_bpw)
            code = pk.rvq_encode(w, dim=dim, k=k, stages=stages, iters=8)
            out[key] = pk.rvq_decode(code).cpu().numpy().astype(np.float32)
        cache.put(ck, (out["mlp1"], out["mlp2"]))
        return out
    return hook


# ── durability ────────────────────────────────────────────────────────────────────────────
def _acquire_lease() -> None:
    LEASES.mkdir(parents=True, exist_ok=True)
    if LEASE_PATH.exists():
        try:
            held = json.loads(LEASE_PATH.read_text())
            if _pid_alive(int(held.get("pid", -1))):
                raise SystemExit(f"live lease held by pid {held['pid']} ({held.get('owner')}); refusing second controller")
        except (json.JSONDecodeError, ValueError):
            pass
    _atomic(LEASE_PATH, {"acquired_at": _now(), "owner": LABEL, "pid": os.getpid(),
                         "schema": "hawking.successor.watchdog_lease.v1"})


def _beat(row_id: str, phase: str, done: int, total: int) -> None:
    import shutil
    du = shutil.disk_usage(str(ROOT))
    _atomic(HB_PATH, {"beat_at": _now(), "label": LABEL, "campaign": "frontier_g4", "pid": os.getpid(),
                      "active_generation": "M", "row_id": row_id, "phase": phase,
                      "rows_done": done, "rows_total": total, "free_disk_gb": round(du.free / 1e9, 1),
                      "schema": "hawking.successor.watchdog.v1", "status": "RUNNING"})


def _rows() -> list[dict[str, Any]]:
    # Originals FIRST (fast ~3.2 min each; the never-measured parent real-PPL reference), then the
    # packed sub-bit rows (which reuse the warmed expert cache).
    rows = [{"row_id": f"{h['id']}__original", "prompt": h, "variant": "original", "rate": None}
            for h in HOLDOUT]
    for h in HOLDOUT:
        for r in RATE_LADDER:
            rows.append({"row_id": f"{h['id']}__rvq{r}", "prompt": h, "variant": "packed_rvq", "rate": r})
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
    # cache original logits per prompt for divergence + reuse; shared decoded-expert cache per rate
    orig_logits_cache: dict[str, np.ndarray] = {}
    packed_expert_cache = bc.PressureAwareCache("g4_packed", disk_path=str(ROOT))
    done = 0
    processed = 0
    t_start = time.time()
    for row in rows:
        rid = row["row_id"]
        cp = CHECKPOINTS / f"{rid}.json"
        if cp.exists():
            done += 1
            # reload original logits if a later packed row needs them and file has them
            continue
        if max_rows is not None and processed >= max_rows:
            break
        h = row["prompt"]
        ids = tk.encode(h["text"]).ids
        _beat(rid, "forward", done, total)
        t0 = time.time()
        hook = None if row["variant"] == "original" else _make_hook(row["rate"], packed_expert_cache)
        logits = fwd.logits_for(ids, positions="all", expert_hook=hook)
        secs = round(time.time() - t0, 1)
        nll = _self_nll(logits, ids)
        rec: dict[str, Any] = {
            "row_id": rid, "prompt_id": h["id"], "domain": h["domain"], "variant": row["variant"],
            "rate_bpw": row["rate"], "n_tokens": len(ids), "forward_seconds": secs,
            "quality": nll, "logits_finite": bool(np.isfinite(logits).all()),
            "sealed_at": _now(), "active_generation": "M",
            "honesty": "real full-model from-config forward (coherence-validated, not bit-exact HF); "
                       "packed=naive sub-bit RVQ lower bound; capability tier requires meeting sealed thresholds",
        }
        if row["variant"] == "original":
            orig_logits_cache[h["id"]] = logits.astype(np.float32)
        else:
            orig = orig_logits_cache.get(h["id"])
            if orig is None:
                oc = CHECKPOINTS / f"{h['id']}__original.json"
                # recompute original if not cached (resume path)
                orig = fwd.logits_for(ids, positions="all").astype(np.float32)
                orig_logits_cache[h["id"]] = orig
            div = _divergence(orig, logits)
            rec["divergence_vs_original"] = div
            rec["capability_candidate"] = bool(
                div["mean_sym_kl"] <= PROMOTE_KL and div["next_token_argmax_agreement"] >= PROMOTE_ARGMAX_AGREE)
            rec["verdict"] = ("capability_candidate" if rec["capability_candidate"]
                              else ("degraded" if div["mean_sym_kl"] < 2.0 else "collapse"))
        rec["sha256"] = _sha({k: v for k, v in rec.items() if k != "sha256"})
        _atomic(cp, rec)
        done += 1
        processed += 1
        _beat(rid, "row_done", done, total)
        _write_state(rows, done, total, t_start)
        print(f"[{done}/{total}] {rid}  {secs}s  ppl={nll['perplexity']}"
              + (f"  symKL={rec.get('divergence_vs_original',{}).get('mean_sym_kl')}"
                 f"  agree={rec.get('divergence_vs_original',{}).get('next_token_argmax_agreement')}"
                 f"  -> {rec.get('verdict')}" if row["variant"] != "original" else ""), flush=True)
    _write_state(rows, done, total, t_start, final=True)
    if LEASE_PATH.exists():
        LEASE_PATH.unlink()
    return 0


def _write_state(rows, done, total, t_start, final: bool = False) -> None:
    seals = sorted(CHECKPOINTS.glob("*.json"))
    results = [json.loads(p.read_text()) for p in seals]
    orig = {r["prompt_id"]: r["quality"]["perplexity"] for r in results if r["variant"] == "original"}
    packed = [r for r in results if r["variant"] != "original"]
    candidates = [r["row_id"] for r in packed if r.get("capability_candidate")]
    elapsed = time.time() - t_start
    per_row = elapsed / max(1, done)
    _atomic(STATE_PATH, {
        "schema": "hawking.frontier_g4.real_forward_state.v1", "generated_at": _now(),
        "gate": "G4_short_end_to_end_real_forward", "active_generation": "M",
        "rows_done": done, "rows_total": total, "final": final,
        "parent_real_perplexity_by_prompt": orig,
        "packed_rows": [{"row_id": r["row_id"], "rate_bpw": r["rate_bpw"], "verdict": r.get("verdict"),
                         "mean_sym_kl": r.get("divergence_vs_original", {}).get("mean_sym_kl"),
                         "argmax_agreement": r.get("divergence_vs_original", {}).get("next_token_argmax_agreement"),
                         "perplexity": r["quality"]["perplexity"]} for r in packed],
        "capability_candidates": candidates,
        "eta_seconds_remaining": round(per_row * (total - done), 0) if not final else 0,
        "honest_note": ("first real-forward gate; parent PPL is the never-before-measured reference; "
                        "sub-bit RVQ challenger expected NEGATIVE (consistent with G0-G3 proxy). "
                        "No capability pass unless a packed row meets sealed thresholds "
                        f"(mean_sym_kl<={PROMOTE_KL} AND argmax_agreement>={PROMOTE_ARGMAX_AGREE})."),
    })


def detach() -> int:
    for d in (CONTROLLER,):
        d.mkdir(parents=True, exist_ok=True)
    log = CONTROLLER / "detached.log"
    argv = ["caffeinate", "-i", "-s", PY, os.path.abspath(__file__), "run"]
    with open(log, "ab") as lh:
        proc = subprocess.Popen(argv, stdout=lh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(ROOT), start_new_session=True)
    print(json.dumps({"detached": True, "pid": proc.pid, "log": str(log), "cmd": " ".join(argv)}, indent=2))
    return 0


def status() -> int:
    st = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"status": "no state yet"}
    hb = json.loads(HB_PATH.read_text()) if HB_PATH.exists() else {}
    lease = json.loads(LEASE_PATH.read_text()) if LEASE_PATH.exists() else {}
    alive = _pid_alive(int(lease["pid"])) if lease.get("pid") else False
    print(json.dumps({"lease": lease, "lease_pid_alive": alive, "heartbeat": hb, "state": st},
                     indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G4 real-forward gate controller.")
    ap.add_argument("cmd", choices=["run", "detach", "status"])
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(max_rows=args.max_rows)
    if args.cmd == "detach":
        return detach()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
