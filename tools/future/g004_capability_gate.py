"""G004 capability conjunction gate: D vs E vs a deliberately broken arm.

Structure and matched-EBPW bytes are already measured (c60caeac2). What was
OWED is the gate itself on Kimi-VL-A3B: perplexity AND n-gram diversity, not
reconstruction cosine. Cosine of the broken arm is ~1.0; if the gate cannot
fail that arm, the gate is decorative.

Arms (matched pair D/E from receipts/future/G004_LOWRANK_MATCHED.json):
  E  affine q2-g128 on every expert tensor (the incumbent)
  D  within-tensor low-rank r=828, A/B at 2-bit g128, residual k=306
  broken  0.01 * W on every expert tensor (magnitude destroyed, direction kept)

Non-expert organs stay 4-bit g64, same as gravity_outlier_eval.evaluate.
Runtime is pinned. One body at a time. Model is loaded per arm so peak RSS
is one copy of O003, not three.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import importlib.metadata

import numpy as np

# lowrank_nr lives on main, landed after this worktree's base.
_MAIN_FUTURE = "/Users/scammermike/Downloads/hawking/tools/future"
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_MAIN_FUTURE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import mlx.core as mx
import mlx.nn as nn

import campaign_memory_guard as cmg
import gravity_outlier_eval as G
import lowrank_nr as L

RECEIPT = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "receipts", "future", "G004_CAPABILITY_GATE.json",
)
RANK = 828
RESIDUAL_K = 306
BITS = 2
GROUP = 128


def _runtime() -> dict:
    def ver(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {
        "python": sys.version.split()[0],
        "mlx": ver("mlx"),
        "mlx_lm": ver("mlx_lm"),
        "numpy": np.__version__,
        "default_device": str(mx.default_device()),
        "snap": G.SNAP,
        "loader": "gravity_outlier_eval._load (kimi_vl kv_b_proj split)",
    }


def _score(model, tok) -> dict:
    from mlx_lm import generate
    ids = mx.array([tok.encode(G.NLL_TEXT)[:512]])
    lg = model(ids[:, :-1]).astype(mx.float32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    nll = float(-mx.take_along_axis(lp, ids[:, 1:, None], axis=-1).squeeze(-1).mean())
    ppl = 2.718281828459045 ** nll
    r4s, dis, worst = [], [], 1
    for pr in G.PROMPTS:
        tx = generate(model, tok, prompt=pr, max_tokens=96, verbose=False)
        tt = tok.encode(tx)
        run = cur = 1
        for i in range(1, len(tt)):
            cur = cur + 1 if tt[i] == tt[i - 1] else 1
            run = max(run, cur)
        worst = max(worst, run)
        gr = [tuple(tt[i:i + 4]) for i in range(max(0, len(tt) - 3))]
        r4s.append(1 - len(set(gr)) / max(1, len(gr)))
        dis.append(len(set(tt)) / max(1, len(tt)))
    r4s.sort()
    dis.sort()
    med_r4 = r4s[len(r4s) // 2]
    med_dis = dis[len(dis) // 2]
    ok = bool(med_r4 <= G.R4_MAX and ppl <= G.PPL_MAX)
    return {
        "capability_ok": ok,
        "capability_status": "CANDIDATE_PASS" if ok else "CAPABILITY_LOSS",
        "capability": {
            "ppl": round(ppl, 4),
            "nll": round(nll, 5),
            "median_4gram_repeat": round(med_r4, 4),
            "median_distinct_ratio": round(med_dis, 4),
            "worst_max_repeat_run": worst,
            "n_prompts": len(G.PROMPTS),
            "gate_r4_max": round(G.R4_MAX, 4),
            "gate_ppl_max": round(G.PPL_MAX, 4),
            "reference_bf16_r4": G.BF16_R4,
            "reference_bf16_ppl": G.BF16_PPL,
            "conjunction": "perplexity AND n-gram diversity",
        },
    }


def _experts(model):
    sw = [(p, m) for p, m in model.named_modules()
          if "switch_mlp" in p and isinstance(getattr(m, "weight", None), mx.array)]
    if len(sw) != 78:
        raise RuntimeError(f"expected 78 expert tensors, found {len(sw)}")
    return sw


def _quantize_non_experts(model) -> None:
    def pred(path, mm):
        if "switch_mlp" in path or not hasattr(mm, "to_quantized"):
            return False
        w = getattr(mm, "weight", None)
        return ({"bits": 4, "group_size": 64}
                if w is not None and w.shape[-1] % 64 == 0 else False)
    nn.quantize(model, group_size=64, bits=4, class_predicate=pred)


def _apply_affine_q2g128(sw) -> dict:
    s_w2 = s_h2 = s_dot = 0.0
    for _name, m in sw:
        W = m.weight.astype(mx.float32)
        q, s, b = mx.quantize(m.weight.astype(mx.bfloat16), group_size=GROUP, bits=BITS)
        rec = mx.dequantize(q, s, b, group_size=GROUP, bits=BITS).astype(mx.float32)
        s_w2 += float(mx.sum(W * W))
        s_h2 += float(mx.sum(rec * rec))
        s_dot += float(mx.sum(W * rec))
        m.weight = rec.astype(mx.bfloat16)
        mx.eval(m.weight)
        del W, rec
    return {
        "magnitude_ratio": (s_h2 / s_w2) ** 0.5 if s_w2 else None,
        "direction_similarity": (s_dot / (s_w2 * s_h2) ** 0.5) if s_w2 and s_h2 else None,
    }


def _lowrank_one(W: np.ndarray) -> np.ndarray:
    m, n = int(W.shape[0]), int(W.shape[1])
    r = min(RANK, max(1, min(m, n) - 1))
    k = min(RESIDUAL_K, int(W.size))
    arm = L.encode_lowrank(W, r=r, bits_a=BITS, group_a=GROUP,
                           bits_b=BITS, group_b=GROUP, residual_k=k)
    return arm.reconstruction


def _apply_lowrank_D(sw) -> dict:
    s_w2 = s_h2 = s_dot = 0.0
    n_slices = 0
    t0 = time.perf_counter()
    for i, (_name, m) in enumerate(sw):
        W = np.array(m.weight.astype(mx.float32))
        if W.ndim == 3:
            recs = [_lowrank_one(W[e]) for e in range(W.shape[0])]
            rec = np.stack(recs, axis=0)
            n_slices += W.shape[0]
        elif W.ndim == 2:
            rec = _lowrank_one(W)
            n_slices += 1
        else:
            raise RuntimeError(f"expert weight ndim {W.ndim} at {_name}")
        rec_m = mx.array(rec)
        W_m = mx.array(W)
        s_w2 += float(mx.sum(W_m * W_m))
        s_h2 += float(mx.sum(rec_m * rec_m))
        s_dot += float(mx.sum(W_m * rec_m))
        m.weight = rec_m.astype(mx.bfloat16)
        mx.eval(m.weight)
        del W, rec, rec_m, W_m
        if i % 10 == 0:
            sys.stderr.write(f"  D encoded {i + 1}/{len(sw)} tensors, {n_slices} slices\n")
            sys.stderr.flush()
    return {
        "magnitude_ratio": (s_h2 / s_w2) ** 0.5 if s_w2 else None,
        "direction_similarity": (s_dot / (s_w2 * s_h2) ** 0.5) if s_w2 and s_h2 else None,
        "n_slices": n_slices,
        "encode_wall_s": round(time.perf_counter() - t0, 2),
    }


def _apply_broken(sw, scale: float = 0.01) -> dict:
    s_w2 = s_h2 = s_dot = 0.0
    for _name, m in sw:
        W = m.weight.astype(mx.float32)
        rec = W * scale
        s_w2 += float(mx.sum(W * W))
        s_h2 += float(mx.sum(rec * rec))
        s_dot += float(mx.sum(W * rec))
        m.weight = rec.astype(mx.bfloat16)
        mx.eval(m.weight)
        del W, rec
    return {
        "magnitude_ratio": (s_h2 / s_w2) ** 0.5 if s_w2 else None,
        "direction_similarity": (s_dot / (s_w2 * s_h2) ** 0.5) if s_w2 and s_h2 else None,
    }


def _run_arm(letter: str, spec: str, apply_fn) -> dict:
    snap = cmg.sample(expected_gb=32.0)
    if snap.state == "STOP":
        return {
            "arm": letter, "spec": spec, "status": "GUARD_STOP",
            "refusal": (f"campaign guard STOP before {letter}: free {snap.free_gb} GB, "
                        f"headroom {snap.headroom_gb} GB, {'; '.join(snap.reasons)}"),
            "guard": snap.as_dict(),
        }
    sys.stderr.write(f"\n== arm {letter} {spec}  guard={snap.state} free={snap.free_gb} ==\n")
    sys.stderr.flush()
    t0 = time.perf_counter()
    with cmg.resource_cost(interval_s=2.0, label=f"g004-{letter}") as cost:
        model, tok = G._load()
        sw = _experts(model)
        mag = apply_fn(sw)
        _quantize_non_experts(model)
        mx.eval(model.parameters())
        mx.synchronize()
        scored = _score(model, tok)
        del model, tok, sw
        gc.collect()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    peak = None
    try:
        import resource as _r
        peak = _r.getrusage(_r.RUSAGE_SELF).ru_maxrss / (1 << 30)
    except Exception:
        peak = None
    row = {
        "arm": letter,
        "spec": spec,
        "status": "MEASURED",
        "wall_s": round(time.perf_counter() - t0, 2),
        "peak_rss_gib": None if peak is None else round(peak, 3),
        "swapfile_delta": cost.get("swapfiles_delta"),
        "resource": {k: cost[k] for k in cost if k != "label"},
        "guard": snap.as_dict(),
        **mag,
        **scored,
        "execution_complete": False if letter == "D" else True,
        "execution_note": (
            "no low-rank kernel; W_hat rematerialized dense for scoring; NR not NX"
            if letter == "D" else
            "experts affine-dequantized then scored; non-experts 4b/g64"
            if letter == "E" else
            "deliberate 0.01x expert magnitude; cosine must not excuse this"
        ),
    }
    sys.stderr.write(
        f"  {letter}  ppl={row['capability']['ppl']}  r4={row['capability']['median_4gram_repeat']}  "
        f"ok={row['capability_ok']}  mag={row.get('magnitude_ratio')}\n"
    )
    sys.stderr.flush()
    return row


def _one_arm(letter: str) -> dict:
    mx.set_default_device(mx.gpu)
    if letter == "E":
        return _run_arm("E", "q2-g128-experts", _apply_affine_q2g128)
    if letter == "D":
        return _run_arm("D", "lowrank-r828-A2g128-B2g128-k306", _apply_lowrank_D)
    if letter == "broken":
        return _run_arm("broken", "broken-magnitude-0.01", _apply_broken)
    raise SystemExit(f"unknown arm {letter}")


def main() -> int:
    if "--one-arm" in sys.argv:
        letter = sys.argv[sys.argv.index("--one-arm") + 1]
        out = sys.argv[sys.argv.index("--row-out") + 1] if "--row-out" in sys.argv else None
        row = _one_arm(letter)
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
            with open(out, "w") as fh:
                json.dump(row, fh, indent=1)
                fh.write("\n")
        else:
            print(json.dumps(row, indent=1))
        return 0 if row.get("status") == "MEASURED" else 1

    mx.set_default_device(mx.gpu)
    runtime = _runtime()
    sys.stderr.write(f"runtime {json.dumps(runtime)}\n")
    # Child processes in this sandbox lose Metal (Seatbelt). Score in-process.
    rows = []
    for letter in ("E", "D", "broken"):
        try:
            rows.append(_one_arm(letter))
        except Exception as exc:
            rows.append({
                "arm": letter, "status": "REFUSED",
                "refusal": f"{type(exc).__name__}: {exc}",
                "capability_ok": None,
            })
            sys.stderr.write(f"  {letter} FAILED {type(exc).__name__}: {exc}\n")

    def ok(letter: str) -> bool | None:
        r = next(x for x in rows if x["arm"] == letter)
        return None if r.get("status") != "MEASURED" else bool(r["capability_ok"])

    broken_failed = ok("broken") is False
    e_ok, d_ok = ok("E"), ok("D")
    if d_ok is True and e_ok is False:
        decision = "D_WINS_GATE"
    elif e_ok is True and d_ok is False:
        decision = "E_WINS_GATE"
    elif e_ok is True and d_ok is True:
        decision = "BOTH_PASS"
    elif e_ok is False and d_ok is False:
        decision = "BOTH_FAIL"
    else:
        decision = "INCOMPLETE"
    doc = {
        "obligation": "G004",
        "what": ("capability conjunction gate on the matched D/E pair plus a "
                 "deliberately broken magnitude arm, Kimi-VL-A3B"),
        "method": (
            "Same scorer as gravity_outlier_eval.evaluate: NLL on a fixed passage "
            "and 8 greedy 96-token prompts. Gate is ppl<=1.25*bf16 AND "
            "median_4gram_repeat<=3*bf16. Reconstruction cosine is recorded and "
            "MUST NOT decide. Broken arm is 0.01*W; if it passes, the gate is dead."
        ),
        "runtime": runtime,
        "matched_bytes_organ": {
            "D": 811028, "E": 811032,
            "source": "receipts/future/G004_LOWRANK_MATCHED.json one 2048x1408 down_proj",
        },
        "arms": rows,
        "decision": decision,
        "broken_arm_failed_the_gate": broken_failed,
        "reconstruction_is_not_capability": True,
        "nr_not_nx": True,
    }
    if not broken_failed and ok("broken") is True:
        doc["gate_defect"] = (
            "the 0.01x magnitude arm PASSED the conjunction gate -- the gate "
            "cannot see a representation that gravity_gauntlet already rejects "
            "on magnitude_ratio"
        )
    os.makedirs(os.path.dirname(RECEIPT), exist_ok=True)
    with open(RECEIPT, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(json.dumps({
        "receipt": RECEIPT,
        "decision": decision,
        "broken_arm_failed_the_gate": broken_failed,
        "arms": [{k: r.get(k) for k in
                  ("arm", "spec", "status", "capability_ok", "capability",
                   "magnitude_ratio", "direction_similarity", "wall_s")}
                 for r in rows],
    }, indent=1))
    return 0 if all(r.get("status") == "MEASURED" for r in rows) and broken_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
