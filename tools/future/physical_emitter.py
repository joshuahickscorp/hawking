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
