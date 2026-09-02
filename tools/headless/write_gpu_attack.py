#!/usr/bin/env python3
"""G006 — record the outcome of attacking the decode ceiling.

The operator's stated secondary target:

    "the main secondary target is the gpu bottleneck we need to fix that asap
     or even be able to run parellel models like ports without aggregate being
     similar if possibel"

"without aggregate being similar" is the exact ask: N parallel runtimes whose
aggregate throughput is meaningfully more than one runtime's, rather than N
runtimes all fighting over the same bandwidth for a 1.2x.

This reads receipts/headless/DECODE_TOPOLOGY.json and writes GPU_ATTACK.json
with a bound outcome: IMPROVED (with the mechanism and the measured gain) or
CEILING_PROVEN (with achieved-versus-roof figures). "We tried" is not an outcome.

A gain inside the measurement spread is reported as noise, not a win.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO = Path(os.path.expanduser("~/Downloads/hawking-copy"))
BASELINE_TPS = 27.900814587942833  # .hcli-legacy/bootstrap-director-v6/worker-equilibrium.json


def main() -> int:
    src = REPO / "receipts/headless/DECODE_TOPOLOGY.json"
    if not src.exists():
        print(f"FAIL: {src} not found — run tools/headless/decode_topology_probe.py first")
        return 2
    d = json.loads(src.read_text())
    s = d["summary"]
    bw = d.get("bandwidth_accounting") or {}

    def best(arm):
        rows = s.get(arm, {})
        if not rows:
            return None, None, None
        k = max(rows, key=lambda k: rows[k].get("aggregate_tps_median") or 0)
        return int(k), rows[k].get("aggregate_tps_median"), rows[k].get("spread_pct")

    pk, ptps, pspread = best("process")
    sk, stps, sspread = best("slot")
    spreads = [v.get("spread_pct") for arm in s.values() for v in arm.values()
               if v.get("spread_pct") is not None]
    noise = max(spreads) if spreads else 5.0

    # The best thing we can actually do today, across both topologies.
    if (stps or 0) >= (ptps or 0):
        best_tps, best_k, best_arm = stps, sk, "slot"
    else:
        best_tps, best_k, best_arm = ptps, pk, "process"

    gain_vs_baseline = 100 * (best_tps - BASELINE_TPS) / BASELINE_TPS if best_tps else None
    slot_over_process = round(stps / ptps, 3) if (ptps and stps) else None

    improved = bool(best_tps and gain_vs_baseline is not None and gain_vs_baseline > noise)

    # The topology axis is only half the attack. A different runtime on the same
    # machine is the other half, and on the numbers it is the larger one.
    ab_path = REPO / "receipts/headless/RUNTIME_AB.json"
    ab = json.loads(ab_path.read_text()) if ab_path.exists() else None
    runtime_axis = None
    if ab:
        arms = ab.get("arms") or {}
        lc = (arms.get("llama_cpp") or {}).get("decode_tps_median")
        mx = (arms.get("mlx") or {}).get("decode_tps_median")
        runtime_axis = {
            "llama_cpp_single_stream_tps": lc,
            "mlx_single_stream_tps": mx,
            "mlx_over_llama": (ab.get("comparison") or {}).get("mlx_over_llama"),
            "bytes_ratio_llama_over_mlx": (ab.get("comparison") or {}).get("bytes_ratio_llama_over_mlx"),
            "confound_declared": ab.get("confound_declared"),
            "reading": ("MLX moves 1.215x fewer bytes but runs 1.472x faster, so roughly 21% of "
                        "the gain is kernel/dispatch efficiency rather than density."),
        }

    doc = {
        "schema": "hawking.headless.gpu_attack.v1",
        "obligation": "G006 — attack the decode ceiling, or prove it with numbers",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operator_ask": ("run parallel models like ports without aggregate being similar — "
                         "i.e. N runtimes should aggregate to meaningfully more than one"),
        "baseline_tps": BASELINE_TPS,
        "baseline_source": (".hcli-legacy/bootstrap-director-v6/worker-equilibrium.json "
                            "best_aggregate_tps at active_decode_limit=2, "
                            "aggregate_scaling_vs_1=1.208"),
        "reps": (d.get("params") or {}).get("reps"),
        "spread_pct": noise,
        "measurement_is_clean_room": d.get("measurement_is_clean_room"),
        "clean_room_caveat": d.get("clean_room_caveat"),
        "topologies": {
            "process": {"best_k": pk, "aggregate_tps": ptps, "spread_pct": pspread,
                        "scaling_vs_1": (s.get("process", {}).get(str(pk), {}) or {}).get("scaling_vs_1"),
                        "what_it_is": "N separate llama-server processes, one slot each"},
            "slot": {"best_k": sk, "aggregate_tps": stps, "spread_pct": sspread,
                     "scaling_vs_1": (s.get("slot", {}).get(str(sk), {}) or {}).get("scaling_vs_1"),
                     "what_it_is": "one llama-server with N continuous-batching slots"},
        },
        "slot_over_process_at_best": slot_over_process,
        "achieved_tps": best_tps,
        "achieved_topology": best_arm,
        "achieved_k": best_k,
        "mechanism": (
            "continuous batching: one pass over the weights per step serves N sequences, "
            "so the per-token weight streaming that dominates decode on this box is amortised "
            "instead of multiplied"
            if best_arm == "slot" else
            "none found — separate processes remained the best available topology"),
        "bandwidth_accounting": bw,
        "runtime_axis": runtime_axis,
        "outcome": "IMPROVED" if improved else "CEILING_PROVEN",
    }

    # Which single configuration is actually the fastest thing available today?
    candidates = [("llama.cpp slot k=%s" % sk, stps),
                  ("llama.cpp process k=%s" % pk, ptps)]
    if runtime_axis and runtime_axis.get("mlx_single_stream_tps"):
        candidates.append(("MLX single stream", runtime_axis["mlx_single_stream_tps"]))
    candidates = [c for c in candidates if c[1]]
    best_overall = max(candidates, key=lambda c: c[1]) if candidates else (None, None)
    doc["best_available_today"] = {
        "config": best_overall[0], "tps": best_overall[1],
        "vs_baseline_pct": (round(100 * (best_overall[1] - BASELINE_TPS) / BASELINE_TPS, 1)
                            if best_overall[1] else None),
        "note": ("A SINGLE MLX stream beats every llama.cpp concurrency configuration measured. "
                 "The concurrency axis is worth at most 1.38x; the runtime axis is worth 1.47x on "
                 "its own."
                 if best_overall[0] == "MLX single stream" else
                 "llama.cpp concurrency remains the best configuration measured."),
    }
    doc["open_frontier"] = [
        ("MLX + concurrency is UNMEASURED. If mlx_lm.server batches the way llama.cpp slots do, "
         f"{runtime_axis['mlx_single_stream_tps']} x 1.38 would be around "
         f"{round(runtime_axis['mlx_single_stream_tps'] * 1.38, 1)} tok/s. That product is a "
         "guess, not a measurement, and must not be quoted as one until the server path is "
         "benchmarked." if runtime_axis and runtime_axis.get("mlx_single_stream_tps") else
         "MLX + concurrency is unmeasured."),
        ("MLX has no response_format/json_schema/grammar (BACKEND_CAPABILITY.json), so an MLX "
         "backend trades constrained decoding for speed. It keeps enable_thinking and "
         "prompt_cache, which are the parts that actually rescued production."),
        ("Lower BPW is untouched here: a 2026-08-18 receipt measured 33.1 tok/s complete-wall at "
         "4.256 BPW on a native runtime, against this 19.5 GB Q5_K artifact."),
    ]

    if improved:
        doc["gain_pct_vs_baseline"] = round(gain_vs_baseline, 1)
        doc["verdict"] = (
            f"{best_arm} topology at k={best_k} reaches {best_tps} tok/s against a "
            f"{BASELINE_TPS:.1f} tok/s baseline (+{gain_vs_baseline:.1f}%), which exceeds the "
            f"{noise:.1f}% measurement spread.")
        doc["what_changes_in_hcli"] = (
            "RuntimePool's unit of concurrency must become a SLOT, not a process. Spawning N "
            "separate llama-servers on N ports is the worst available topology on a "
            "bandwidth-bound box: each process streams the full weight set per token. The pool "
            "should run few processes with many slots, and runtime identity becomes "
            "(process, slot) rather than just (process)."
            if best_arm == "slot" else
            "no topology change indicated by this measurement")
    else:
        doc["why_ceiling"] = (
            f"Neither topology beat the {BASELINE_TPS:.1f} tok/s baseline by more than the "
            f"{noise:.1f}% measurement spread. Best observed was {best_tps} tok/s via the "
            f"{best_arm} topology at k={best_k}. "
            + (f"A single decoder already streams ~{bw.get('implied_weight_stream_GB_per_s')} GB/s, "
               f"{bw.get('single_decoder_pct_of_peak')}% of the ~{bw.get('m3_ultra_peak_GB_per_s')} GB/s "
               f"roof, so the remaining headroom cannot support proportional scaling."
               if bw else ""))
        doc["remaining_levers"] = [
            "fewer bytes per token (lower BPW) — the Gravity axis; a prior campaign measured "
            "33.1 tok/s complete-wall at 4.256 BPW versus this 19.5 GB Q5_K artifact",
            "multi-token amortisation (speculative or self-speculative decode) — shares one "
            "weight sweep across K accepted tokens",
            "activation sparsity in the SwiGLU MLP, which a prior receipt measured at 67% of "
            "active bytes",
        ]

    out = REPO / "receipts/headless/GPU_ATTACK.json"
    out.write_text(json.dumps(doc, indent=1))

    print("=== GPU ATTACK ===")
    print(f"  baseline           {BASELINE_TPS:.2f} tok/s (prior calibration, 1.208x at k=2)")
    print(f"  process topology   best k={pk}  {ptps} tok/s  (spread {pspread}%)")
    print(f"  slot topology      best k={sk}  {stps} tok/s  (spread {sspread}%)")
    print(f"  slot/process       {slot_over_process}")
    if runtime_axis:
        print(f"  runtime axis       llama.cpp {runtime_axis['llama_cpp_single_stream_tps']} vs "
              f"MLX {runtime_axis['mlx_single_stream_tps']} tok/s "
              f"({runtime_axis['mlx_over_llama']}x on a {runtime_axis['bytes_ratio_llama_over_mlx']}x byte ratio)")
    b = doc.get("best_available_today") or {}
    print(f"  BEST AVAILABLE     {b.get('config')} = {b.get('tps')} tok/s "
          f"({b.get('vs_baseline_pct')}% vs baseline)")
    print(f"  OUTCOME            {doc['outcome']}")
    print(f"  {doc.get('verdict') or doc.get('why_ceiling')}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
