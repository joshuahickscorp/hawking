#!/usr/bin/env python3
"""Full unfused G0 Q4 dequant-vs-BF16 cosine. Streams. No GPU."""
from __future__ import annotations

import json
import os
import time

# reuse helpers
import importlib.util

spec = importlib.util.spec_from_file_location("m", "/tmp/g1_capability_gate_measure.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

G0 = m.G0
man = json.load(open(os.path.join(G0, "manifest.json")))
weight_map = m.load_safetensors_index()
rows = [t for t in man["tensors"] if t["kind"] == "q4"]
print(f"q4_rows={len(rows)} weight_map={len(weight_map)}", flush=True)

measured = []
skipped = []
t0 = time.perf_counter()
for i, t in enumerate(rows):
    name = t["name"]
    if name not in weight_map:
        skipped.append({"name": name, "reason": "not_in_bf16_map_probably_fused", "shape": t["shape"]})
        continue
    q4_path = os.path.join(G0, "tensors", t["artifact"])
    rec = m.streamed_q4_cosine(q4_path, weight_map[name], name, chunk_groups=8192)
    rec["manifest_cosine"] = t.get("cosine")
    measured.append(rec)
    if (i + 1) % 25 == 0 or rec["cosine"] < 0.993:
        print(
            f"[{len(measured):3}/{len(rows)}] {rec['cosine']:.8f} {name} {rec['wall_s']:.2f}s",
            flush=True,
        )

cos = sorted(r["cosine"] for r in measured)
out = {
    "n_q4_catalog": len(rows),
    "n_measured": len(measured),
    "n_skipped_fused_or_missing": len(skipped),
    "skipped": skipped,
    "min": cos[0] if cos else None,
    "p01": cos[max(0, len(cos) // 100)] if cos else None,
    "p10": cos[max(0, len(cos) // 10)] if cos else None,
    "median": cos[len(cos) // 2] if cos else None,
    "max": cos[-1] if cos else None,
    "argmin": min(measured, key=lambda r: r["cosine"])["name"] if measured else None,
    "argmax": max(measured, key=lambda r: r["cosine"])["name"] if measured else None,
    "n_below_0_993": sum(1 for c in cos if c < 0.993),
    "n_below_0_990": sum(1 for c in cos if c < 0.990),
    "n_below_0_98948": sum(1 for c in cos if c < 0.98948),
    "n_none": sum(1 for r in measured if r["cosine"] is None),
    "n_nonfinite_any": sum(1 for r in measured if r["n_nonfinite"]),
    "wall_s": time.perf_counter() - t0,
    "lowest5": sorted(
        [{"name": r["name"], "cosine": r["cosine"], "rel_l2": r["rel_l2"], "elements": r["elements"]} for r in measured],
        key=lambda x: x["cosine"],
    )[:5],
    "highest3": sorted(
        [{"name": r["name"], "cosine": r["cosine"]} for r in measured],
        key=lambda x: -x["cosine"],
    )[:3],
}
path = "/tmp/g1_capability_gate_full_cosine.json"
json.dump(out, open(path, "w"), indent=2)
print("WROTE", path)
print(json.dumps({k: out[k] for k in out if k not in ("skipped", "lowest5", "highest3")}, indent=2))
print("lowest5", json.dumps(out["lowest5"], indent=2))
print("skipped_n", len(skipped), "sample", skipped[:3])
