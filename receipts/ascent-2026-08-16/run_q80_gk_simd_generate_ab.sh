#!/bin/bash
# 3 alternating BASE (recon-fuse occupancy tiles) vs OURS (HAWKING_Q80_GK_SIMD=1)
# pairs. New process each arm. GPU lock is the caller's job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${CARGO_TARGET_DIR:-/Users/scammermike/Downloads/hawking/workspace/ops/build/rust}/release-fast/examples/ascension_qwen80_mixed_hybrid_greedy"
OUT="$ROOT/receipts/ascent-2026-08-16/q80-gk-simd"
ARTIFACT="/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/mixed-1p5-v1"
mkdir -p "$OUT"
EXPECTED="8420,748,264,729,429,17431,288,264,914,320,72,1734"

run_arm() {
  local arm="$1" pair="$2" flag="$3"
  local dest="$OUT/${arm}_p${pair}.json"
  echo "=== pair $pair arm=$arm HAWKING_Q80_GK_SIMD=$flag ==="
  env -u HAWKING_Q80_RECON_FUSE HAWKING_Q80_GK_SIMD="$flag" \
    "$BIN" \
      --artifact-root "$ARTIFACT" \
      --prompt "Write a function that reverses a string." \
      --max-new-tokens 12 \
      --reps 1 \
      --out "$dest"
}

for pair in 1 2 3; do
  run_arm base "$pair" 0
  run_arm ours "$pair" 1
done

python3 - <<'PY'
import json, pathlib
out = pathlib.Path("receipts/ascent-2026-08-16/q80-gk-simd")
expected = [8420, 748, 264, 729, 429, 17431, 288, 264, 914, 320, 72, 1734]
rows = []
for pair in (1, 2, 3):
    row = {"pair": pair}
    for arm in ("base", "ours"):
        p = out / f"{arm}_p{pair}.json"
        d = json.loads(p.read_text())
        ids = d.get("generated_token_ids")
        silent = d["correctness"]["silent_fallback_count"]
        gpu_missing = d["timing"]["gpu_timestamps_missing"]
        gpu = d["timing"]["gpu_matvec_ns_per_token_reps"][0]
        wall = d["timing"]["wall_ns_per_token_reps"][0]
        bind = d["timing"]["host_expert_bind_ns_per_token_reps"][0]
        wmg = d["timing"]["wait_minus_gpu_ns_per_token_reps"][0]
        cbs = d["timing"]["command_buffers_per_token_reps"][0]
        disp = d["timing"]["dispatches_per_token_reps"][0]
        stages = d["execution"]["stages"]
        row[arm] = {
            "ids": ids,
            "ids_match": ids == expected,
            "silent": silent,
            "gpu_missing": gpu_missing,
            "gpu_matvec_ns": gpu,
            "wall_ns": wall,
            "bind_ns": bind,
            "wait_minus_gpu_ns": wmg,
            "cbs": cbs,
            "dispatches": disp,
            "gk_simd_enabled": d["timing"].get("gk_simd_enabled"),
            "recon_fuse_enabled": d["timing"].get("recon_fuse_enabled"),
            "stages": {
                "deltanet_secs": stages.get("deltanet_secs"),
                "gqa_secs": stages.get("gqa_secs"),
                "moe_shared_secs": stages.get("moe_shared_secs"),
                "moe_routed_secs": stages.get("moe_routed_secs"),
                "terminal_secs": stages.get("terminal_secs"),
                "gpu_matvec_ns": stages.get("gpu_matvec_ns"),
            },
            "dense_w": d["correctness"]["dense_w_materialized"],
        }
    rows.append(row)

def stats(key):
    base = [r["base"][key] for r in rows]
    ours = [r["ours"][key] for r in rows]
    def med(xs):
        ys = sorted(xs)
        return ys[len(ys)//2]
    return {
        "base": {"all": base, "min": min(base), "median": med(base), "max": max(base), "spread": max(base)-min(base)},
        "ours": {"all": ours, "min": min(ours), "median": med(ours), "max": max(ours), "spread": max(ours)-min(ours)},
        "speedup_x": med(base)/med(ours) if med(ours) else None,
    }

summary = {
    "schema": "hawking.ascension.q80_gk_simd_generate_ab.v1",
    "lane": "q80-wire-gk-simd",
    "vehicle": "mixed-1p5-v1",
    "alternating": "BASE(HAWKING_Q80_GK_SIMD=0, recon-fuse default on) then OURS(HAWKING_Q80_GK_SIMD=1) per pair, new process each arm",
    "expected_ids": expected,
    "ids_identical_all_six": all(r["base"]["ids_match"] and r["ours"]["ids_match"] for r in rows),
    "silent_fallbacks_all_zero": all(r["base"]["silent"]==0 and r["ours"]["silent"]==0 for r in rows),
    "gpu_timestamps_missing_all_zero": all(r["base"]["gpu_missing"]==0 and r["ours"]["gpu_missing"]==0 for r in rows),
    "gpu_matvec_ns_per_token": stats("gpu_matvec_ns"),
    "wall_ns_per_token": stats("wall_ns"),
    "host_expert_bind_ns_per_token": stats("bind_ns"),
    "wait_minus_gpu_ns_per_token": stats("wait_minus_gpu_ns"),
    "pair_table": rows,
}
path = out / "SUMMARY.json"
path.write_text(json.dumps(summary, indent=2))
print(json.dumps({
    "ids_ok": summary["ids_identical_all_six"],
    "silent_ok": summary["silent_fallbacks_all_zero"],
    "gpu_missing_ok": summary["gpu_timestamps_missing_all_zero"],
    "gpu_base_median_ms": summary["gpu_matvec_ns_per_token"]["base"]["median"]/1e6,
    "gpu_ours_median_ms": summary["gpu_matvec_ns_per_token"]["ours"]["median"]/1e6,
    "gpu_speedup_x": summary["gpu_matvec_ns_per_token"]["speedup_x"],
    "wall_base_median_ms": summary["wall_ns_per_token"]["base"]["median"]/1e6,
    "wall_ours_median_ms": summary["wall_ns_per_token"]["ours"]["median"]/1e6,
}, indent=2))
print("wrote", path)
PY
