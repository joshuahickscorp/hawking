"""The one canonical contract-compliant physical emitter (G026, S009 20/21).

tps_contract over the live receipts: 79 physical, 1 COMPARABLE, 78 incomparable.
Filling the gpu / cpu / tps axes from those would fail the campaign's own
contract, and loosening the contract to make them pass is the move this
forbids. So: emit correctly instead.

The controlled execution is one already happening. Every HCLI round writes
`.hcli/receipts/<goal_id>.json` carrying, per model call, the prefill step
count, the per-bucket GPU / encode / wait / wall nanoseconds, the dispatch
count, the GPU share of wall, the completion tokens and the total call wall --
against a resident whose artifact bytes and mix identity are recorded in the
same file. That is a real workload on real hardware with a pinned runtime; it
is not a synthetic benchmark, and G007 already used it as authority.

What was missing was never the measurement. It was the six contract fields
sitting beside it: specimen, nr, runtime, context, path, concurrency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
ROUNDS = REPO / ".hcli" / "receipts"
OUT = REPO / "receipts" / "future"


def _prefill_wall_ns(call: Dict[str, Any]) -> Optional[float]:
    prof = call.get("prefill_profile") or {}
    buckets = prof.get("buckets") or []
    if not buckets:
        return None
    return sum(b.get("wall_ns_mean", 0.0) * b.get("steps", 0) for b in buckets)


def _gpu_share(call: Dict[str, Any]) -> Optional[float]:
    attr = call.get("prefill_attribution") or {}
    return attr.get("gpu_share_of_wall")


def emit(round_receipt: Path) -> Dict[str, Any]:
    round_receipt = round_receipt.resolve()
    d = json.loads(round_receipt.read_text())
    calls: List[Dict[str, Any]] = d.get("model_calls") or []
    if not calls:
        raise ValueError(f"{round_receipt.name}: no model_calls -- nothing physical happened")

    prov = (d.get("runtime_provenance") or [{}])[0]
    ident = prov.get("identity") or {}
    inv = ident.get("artifact_inventory") or {}

    per_call, prefill_tok, prefill_s, decode_tok, decode_s = [], 0, 0.0, 0, 0.0
    for i, c in enumerate(calls, 1):
        pw_ns = _prefill_wall_ns(c)
        steps = c.get("prefill_tokens_stepped") or c.get("prefill_step_count") or 0
        wall = c.get("wall_s")
        pw_s = (pw_ns / 1e9) if pw_ns else None
        comp = c.get("completion_tokens") or 0
        dec_s = (wall - pw_s) if (wall is not None and pw_s is not None) else None
        row = {
            "call": i,
            "prompt_tokens": c.get("prompt_tokens"),
            "prefill_steps": steps,
            "prefill_wall_s": round(pw_s, 4) if pw_s else None,
            "prefill_tok_per_s": round(steps / pw_s, 4) if (pw_s and steps) else None,
            "completion_tokens": comp,
            "decode_wall_s": round(dec_s, 4) if dec_s and dec_s > 0 else None,
            "decode_tok_per_s": round(comp / dec_s, 4) if (dec_s and dec_s > 0 and comp) else None,
            "call_wall_s": wall,
            "gpu_share_of_prefill_wall": _gpu_share(c),
            "dispatches_per_step": (c.get("prefill_attribution") or {}).get("dispatches_per_step"),
            "prefix_source": c.get("prefix_source"),
            "prefix_reused_tokens": c.get("prefix_reused_tokens"),
        }
        per_call.append(row)
        if pw_s and steps:
            prefill_tok += steps
            prefill_s += pw_s
        if dec_s and dec_s > 0 and comp:
            decode_tok += comp
            decode_s += dec_s

    gpu_shares = [r["gpu_share_of_prefill_wall"] for r in per_call
                  if r["gpu_share_of_prefill_wall"] is not None]

    out: Dict[str, Any] = {
        "schema": "hawking.future.physical.v1",
        "obligation": "G026",
        "emitted_by": "tools/future/physical_emitter.py",
        "source_receipt": str(round_receipt.relative_to(REPO)),
        "goal_id": d.get("goal_id"),

        # THE SIX CONTRACT FIELDS. Present at the TOP LEVEL, because
        # tps_contract walks for the first occurrence of each alias and a field
        # buried under a nested block competes with every other dict in the file.
        "specimen": ident.get("mix_id") or ident.get("catalog") or "sealed-3.14 resident",
        "nr": (f"native-packed, artifact_bytes {inv.get('artifact_bytes')} "
               f"({inv.get('artifact_bytes_source')})" if inv.get("artifact_bytes")
               else "unknown -- artifact inventory absent from the round receipt"),
        "runtime": (f"{prov.get('provider')} endpoint {calls[0].get('endpoint')}, "
                    f"{calls[0].get('layers')} layers"),
        "context": max((c.get("prompt_tokens") or 0) for c in calls),
        "path": ("the engine's agentic tool loop: the round's own production model calls, "
                 "prefill stepped one token at a time, decode sampled"),
        "concurrency": d.get("max_model_in_flight", 1),

        "measurement_contract": {
            "not_a_benchmark": ("these are the round's OWN production calls against the live "
                                "resident, not a synthetic workload"),
            "runtime_count": len(d.get("runtime_provenance") or []),
            "n_model_calls": len(calls),
            "started_at": (d.get("timestamps") or {}).get("started_at"),
            "finished_at": (d.get("timestamps") or {}).get("finished_at"),
        },

        "prefill_tps": round(prefill_tok / prefill_s, 4) if prefill_s else None,
        "decode_tps": round(decode_tok / decode_s, 4) if decode_s else None,
        "gpu_share_of_prefill_wall_mean": (round(sum(gpu_shares) / len(gpu_shares), 4)
                                           if gpu_shares else None),
        "totals": {"prefill_tokens": prefill_tok, "prefill_wall_s": round(prefill_s, 4),
                   "decode_tokens": decode_tok, "decode_wall_s": round(decode_s, 4)},
        "per_call": per_call,

        "evidence_tier": "PRODUCTION_WORKLOAD",
        "gpu_authority": True,
        "claim_boundary": (
            "Real hardware, real resident, pinned runtime, concurrency 1. It is NOT a controlled "
            "sweep: prompt length varies call to call because the round chose what to ask, and "
            "prefix reuse differs between calls. Comparable to another receipt from the same "
            "emitter; NOT comparable to a synthetic fixed-length benchmark."),
    }
    return out


def main(argv: List[str]) -> int:
    if len(argv) > 1:
        targets = [Path(a) for a in argv[1:]]
    else:
        targets = sorted(ROUNDS.glob("*.json"), key=lambda p: p.stat().st_mtime)[-1:]
    if not targets:
        print("no round receipts found under .hcli/receipts")
        return 1
    for t in targets:
        try:
            rec = emit(t)
        except Exception as exc:
            print(f"SKIP {t.name}: {type(exc).__name__}: {exc}")
            continue
        dest = OUT / f"G026_PHYSICAL_{rec['goal_id'][:8]}.json"
        dest.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"wrote {dest.relative_to(REPO)}  prefill_tps={rec['prefill_tps']} "
              f"decode_tps={rec['decode_tps']} gpu_share={rec['gpu_share_of_prefill_wall_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


# ---------------------------------------------------------------------------
# Second FRONT-END to the same contract. [S010 37]
#
# emit() reads a round receipt the resident already produced. That covers the
# resident and nothing else, which is why 28 ModelLake bodies still owe
# gpu/cpu/tps: there was no way to measure a body that has never been a
# resident. This measures one directly and fills in the SAME six contract
# fields, with the same schema and the same top-level placement.
#
# It is emphatically NOT a second contract. Anything measured here is
# comparable to another emit_direct receipt; comparing it to an emit() receipt
# means comparing a controlled fixed-length sweep against production traffic,
# and the path field is what says so.
# ---------------------------------------------------------------------------

DIRECT_PATH = ("controlled fixed-length sweep: one fresh prefill of the whole prompt "
               "followed by greedy single-token decode, no prefix reuse, "
               "concurrency 1")




def _stat(xs: List[float], n: int) -> Dict[str, Any]:
    ordered = [round(n / x, 4) for x in xs if x > 0]
    rates = sorted(ordered)
    if not rates:
        return {"min": None, "median": None, "max": None, "spread_pct": None,
                "in_run_order": [], "monotone_drift": None}
    med = rates[len(rates) // 2]
    # THE DISCRIMINATOR. A sorted min/median/max cannot tell a clock ramp
    # from noise, and the first cells of a sweep on this machine sit on a
    # DVFS ramp. Report the series IN RUN ORDER and say whether it is
    # monotone: a spread that only ever increases is drift, and comparing
    # cells from one ordering is then invalid.
    rising = all(b >= a for a, b in zip(ordered, ordered[1:]))
    falling = all(b <= a for a, b in zip(ordered, ordered[1:]))
    out = {
        "min": rates[0],
        "median": med,
        "max": rates[-1],
        # Spread first-class: a bare median is not a measurement.
        "spread_pct": round(100.0 * (rates[-1] - rates[0]) / rates[0], 2),
        "in_run_order": ordered,
        "monotone_drift": ("RISING" if rising and len(ordered) > 1 and
                           ordered[0] != ordered[-1] else
                           "FALLING" if falling and len(ordered) > 1 and
                           ordered[0] != ordered[-1] else "NONE"),
    }
    if out["monotone_drift"] != "NONE":
        # Only say it when it is true. A drift warning printed beside
        # "monotone_drift: NONE" trains the reader to ignore the field.
        out["drift_note"] = ("the series only moves one way: this is drift, not "
                             "scatter, and a single-order comparison against it "
                             "is invalid -- reverse the order or pair the arms")
    elif len(ordered) > 2 and ordered[0] == rates[0]:
        out["first_cell_note"] = ("the slowest repeat is the FIRST one: a cold "
                                  "cell, not a property of the body; the median "
                                  "excludes it but the spread does not")
    return out


def _direct_receipt(*, specimen: str, nr: str, snapshot: str,
                    prefills: List[float], decodes: List[float],
                    n_prompt: int, n_decoded: int, load_s: float,
                    device: str, runtime: str, repeats: int,
                    warmup_discarded: int) -> Dict[str, Any]:
    """ONE receipt shape for every direct backend. [S010 37]

    torch-on-CPU and mlx-on-Metal are two FRONT-ENDS, not two contracts. They
    fill the same six comparability fields with the same names at the same top
    level; `runtime` and `device` are what say which produced a given number,
    and the claim boundary forbids comparing across them.
    """
    pre = _stat(prefills, n_prompt)
    dec = _stat(decodes, n_decoded)
    gpu = device.startswith("gpu")
    return {
        "schema": "hawking.future.physical.v1",
        "obligation": "G026",
        "emitted_by": "tools/future/physical_emitter.py::emit_direct",

        # THE SIX CONTRACT FIELDS, same names and placement as emit().
        "specimen": specimen,
        "nr": nr,
        "runtime": runtime,
        "context": n_prompt,
        "path": DIRECT_PATH,
        "concurrency": 1,

        "measurement_contract": {
            "fresh_prefill": True,
            "prefix_reuse": False,
            "repeats": int(repeats),
            "warmup_repeats_discarded": warmup_discarded,
            "prompt_tokens": n_prompt,
            "decode_tokens": n_decoded,
            "model_load_s": round(load_s, 3),
            "snapshot": str(snapshot),
            "backend_device": device,
            "not_comparable_to": ("emit() receipts, which measure the resident's own "
                                  "production traffic with prefix reuse; nor to a "
                                  "receipt from the other direct backend -- CPU float32 "
                                  "and Metal are different machines for this purpose. "
                                  "`runtime` and `device` are what distinguish them."),
        },
        "prefill_tps": pre["median"],
        "prefill_tps_spread": pre,
        "decode_tps": dec["median"],
        "decode_tps_spread": dec,
        "gpu_share_of_prefill_wall_mean": None,
        "gpu_share_unavailable_because": (
            "mlx does not expose a per-op GPU/CPU split; the whole forward ran on "
            "Metal, which is what `device` records"
            if gpu else
            f"executed on {device}; no GPU work was dispatched"),
        "device": device,
        "totals": {
            "prefill_tokens": n_prompt * len(prefills),
            "prefill_wall_s": round(sum(prefills), 4),
            "decode_tokens": n_decoded * len(decodes),
            "decode_wall_s": round(sum(decodes), 4),
        },
        "evidence_tier": "CONTROLLED_SWEEP",
        "gpu_authority": bool(gpu),
        "claim_boundary": (
            "Real hardware, real forward passes, fixed prompt length, concurrency 1, "
            "fresh prefill every repeat. "
            + ("Metal via mlx, with every timed region forced through mx.eval() -- mlx "
               "is lazy and an unevaluated forward would time graph construction, not "
               "execution. The first repeat is discarded as kernel-compile warm-up."
               if gpu else
               "CPU float32 -- this is NOT the body's best achievable speed and must "
               "not be read as one.")
            + " Comparable to another receipt from the SAME backend on this machine."),
    }


def _emit_direct_mlx(
    snapshot: str, specimen: str, nr: str,
    prompt_tokens: int, decode_tokens: int, repeats: int,
) -> Dict[str, Any]:
    """Same contract, executed on METAL through mlx_lm. [S004]

    The gpu axis was 0 MEASURED / 29 REFUSED campaign-wide, refused as "torch
    MPS unavailable" -- which names the wrong runtime. MLX runs on Metal, is a
    campaign_suite dependency, and supports qwen3. So the refusal was about
    torch, not about the machine.

    THE TRAP THIS CODE EXISTS TO AVOID: mlx is LAZY. Timing a forward pass
    without mx.eval() measures graph construction, not execution, and would
    report a spectacular and entirely fake tok/s. Every timed region below ends
    in an explicit eval, and the first repeat is discarded as a warm-up because
    Metal compiles kernels on first use.
    """
    import time as _t

    import mlx.core as mx  # type: ignore
    import mlx_lm  # type: ignore
    from mlx_lm.models import cache as _cache  # type: ignore

    t0 = _t.perf_counter()
    model, tok = mlx_lm.load(snapshot)
    load_s = _t.perf_counter() - t0
    # Record the PRECISION. mlx loads the checkpoint's native dtype (bfloat16
    # here) while the torch path forces float32, so a cross-backend comparison
    # mixes device with precision. Putting the dtype in the receipt is what stops
    # someone reading a 20x as "Metal is 20x faster than CPU".
    dtype = None
    for _n, _mod in model.named_modules():
        if getattr(_mod, "weight", None) is not None:
            dtype = str(_mod.weight.dtype)
            break

    base = ("the mixture of experts architecture routes each token to a small "
            "subset of feed forward networks so the number of parameters "
            "activated per token is much smaller than the total ")
    ids = tok.encode(base * 64)[:prompt_tokens]
    n_prompt = len(ids)

    prefills: List[float] = []
    decodes: List[float] = []
    n_decoded = 0
    # repeats + 1: the extra pass is a discarded warm-up, because Metal compiles
    # its kernels on first use and that cost is not the body's decode rate.
    for rep in range(int(repeats) + 1):
        prompt = mx.array([ids])
        kv = _cache.make_prompt_cache(model)
        mx.eval(prompt)
        t1 = _t.perf_counter()
        logits = model(prompt, cache=kv)
        nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(nxt)                       # force the graph to actually run
        pre = _t.perf_counter() - t1

        t2 = _t.perf_counter()
        k = 0
        for _ in range(int(decode_tokens)):
            logits = model(nxt, cache=kv)
            nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(nxt)                   # per step: no lazy accumulation
            k += 1
        dec = _t.perf_counter() - t2
        if rep == 0:
            continue                       # warm-up discarded, and said so
        prefills.append(pre)
        decodes.append(dec)
        n_decoded = k
    del model
    return {
        "_prefills": prefills, "_decodes": decodes, "_n_prompt": n_prompt,
        "_n_decoded": n_decoded, "_load_s": load_s, "_device": "gpu (Metal via mlx)",
        "_runtime": (f"mlx_lm {getattr(mx, '__version__', 'unknown')}, device "
                     f"{mx.default_device()}, weights {dtype}"),
        "_warmup_discarded": 1,
    }


def emit_direct(
    snapshot: str,
    *,
    specimen: str,
    nr: str = "source body as stored on disk, unmodified",
    prompt_tokens: int = 512,
    decode_tokens: int = 64,
    repeats: int = 3,
    backend: str = "torch",
) -> Dict[str, Any]:
    """Measure one body's fresh prefill and decode under the canonical contract.

    Fresh means fresh: the prompt is prefilled from an empty cache every repeat,
    so this is not a warm prefix-reuse number wearing a prefill label. Repeats
    are reported as min/median/max, never as a bare median -- the campaign has
    already shipped one wrong answer off a median that hid a monotone drift.
    """
    import time as _t

    if str(backend).lower() == "mlx":
        raw = _emit_direct_mlx(snapshot, specimen, nr,
                               prompt_tokens, decode_tokens, repeats)
        return _direct_receipt(
            specimen=specimen, nr=nr, snapshot=snapshot,
            prefills=raw["_prefills"], decodes=raw["_decodes"],
            n_prompt=raw["_n_prompt"], n_decoded=raw["_n_decoded"],
            load_s=raw["_load_s"], device=raw["_device"], runtime=raw["_runtime"],
            repeats=repeats, warmup_discarded=raw["_warmup_discarded"])

    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    t_load0 = _t.perf_counter()
    tok = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    load_s = _t.perf_counter() - t_load0

    # A deterministic prompt of the requested length. Content is irrelevant to
    # the timing; the LENGTH is the workload contract.
    base = ("the mixture of experts architecture routes each token to a small "
            "subset of feed forward networks so the number of parameters "
            "activated per token is much smaller than the total ")
    ids = tok(base * 64, return_tensors="pt").input_ids[:, :prompt_tokens]
    n_prompt = int(ids.shape[1])

    prefills: List[float] = []
    decodes: List[float] = []
    n_decoded = 0
    with torch.no_grad():
        for _ in range(int(repeats)):
            t0 = _t.perf_counter()
            out = model(ids, use_cache=True)   # FRESH prefill, empty cache
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            prefills.append(_t.perf_counter() - t0)

            t1 = _t.perf_counter()
            k = 0
            for _ in range(int(decode_tokens)):
                step = model(nxt, past_key_values=past, use_cache=True)
                past = step.past_key_values
                nxt = step.logits[:, -1, :].argmax(-1, keepdim=True)
                k += 1
            decodes.append(_t.perf_counter() - t1)
            n_decoded = k
    del model

    return _direct_receipt(
        specimen=specimen, nr=nr, snapshot=snapshot,
        prefills=prefills, decodes=decodes, n_prompt=n_prompt,
        n_decoded=n_decoded, load_s=load_s, device="cpu",
        runtime="transformers AutoModelForCausalLM, torch float32, device cpu",
        repeats=repeats, warmup_discarded=0)
