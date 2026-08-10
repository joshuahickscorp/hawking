#!/usr/bin/env bash
# Contaminated A/B for Q30 device-expert-table routing.
#
# Measurement ranking (revision-1):
#   1) command_buffers/token, dispatches/token, host_route_id_readbacks/token
#   2) HAWKING_TCB_TRACE=gpu_prod per-dispatch GPU us (sum + top kernels)
#   3) Paired A/B ratio (control DEVICE_EXPERT_TABLE=0 vs candidate =1)
#   4) Wall-clock tok/s — LABEL CONTAMINATED, never decide on it alone
#
# Requires Metal (gate profile / unsandboxed). Bit-identity gate:
# completion_token_ids control == candidate.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/target-parallel/release/examples/ascension_qwen30_complete_native_runtime}"
OUT="${OUT:-/tmp/q30-device-route-meas}"
TRIALS="${TRIALS:-5}"
MANIFEST="${MANIFEST:-/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity/QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json}"

if [[ ! -x "$BIN" ]]; then
  echo "missing binary: $BIN (cargo build --release -p hawking-core --example ascension_qwen30_complete_native_runtime)" >&2
  exit 2
fi

mkdir -p "$OUT"

COMMON=(
  --manifest "$MANIFEST"
  --expected-manifest-seal-sha256 3321a99d719e70499663b7bfebe14dd6c732bfc533bb05b9277eb398e44d6357
  --expected-source-audit-seal-sha256 00ed3e495416c2cbafbcdb7800528e15f243b1a13f5f4af13240109c8fc69f7b
  --expected-source-revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120
  --mode generate-greedy
  --prompt "Reply with the single word native."
  --prompt-template source-user-chat
  --packed-matvec-kernel control
  --gate-up-swiglu-kernel paired-scalar-order-production-no-parity
  --max-new-tokens 2
  --max-seq-len 256
  --trace-dispatch
)

echo "contamination note: other processes may be running; wall tok/s is CONTAMINATED"
echo "structural counts and gpu_prod GPU us are the decision instruments"
echo "trials=$TRIALS out=$OUT bin=$BIN"

for i in $(seq 1 "$TRIALS"); do
  echo "=== trial $i control DEVICE_EXPERT_TABLE=0 ==="
  HAWKING_TCB_TRACE=gpu_prod HAWKING_QWEN30_DEVICE_EXPERT_TABLE=0 \
    "$BIN" "${COMMON[@]}" \
    >"$OUT/control_t${i}.json" 2>"$OUT/control_t${i}.err" || {
      echo "control trial $i failed; stderr:" >&2
      tail -50 "$OUT/control_t${i}.err" >&2
      exit 1
    }

  echo "=== trial $i candidate DEVICE_EXPERT_TABLE=1 ==="
  HAWKING_TCB_TRACE=gpu_prod HAWKING_QWEN30_DEVICE_EXPERT_TABLE=1 \
    "$BIN" "${COMMON[@]}" \
    >"$OUT/candidate_t${i}.json" 2>"$OUT/candidate_t${i}.err" || {
      echo "candidate trial $i failed; stderr:" >&2
      tail -50 "$OUT/candidate_t${i}.err" >&2
      exit 1
    }
done

python3 - <<'PY' "$OUT" "$TRIALS"
import json, sys
from pathlib import Path
from collections import defaultdict
from statistics import median

out = Path(sys.argv[1])
trials = int(sys.argv[2])

def load(tag, i):
    return json.loads((out / f"{tag}_t{i}.json").read_text())

def step_metrics(doc):
    steps = doc.get("execution", {}).get("steps") or []
    # Prefer first completion step (warm after prefill) if present; else prefill not in steps.
    # generate-greedy puts completion steps only in steps[].
    if not steps:
        return None
    # Use the first completion step; also report max over steps for stability.
    s = steps[0]
    return {
        "command_buffers": s.get("command_buffers"),
        "metal_dispatches": s.get("metal_dispatches"),
        "host_route_id_readbacks": s.get("host_route_id_readbacks"),
        "host_sample_id_readbacks": s.get("host_sample_id_readbacks"),
        "elapsed_us": s.get("elapsed_us_diagnostic_not_tps"),
        "token_id": s.get("sampled_token_id"),
    }

def gpu_sum(doc):
    prof = doc.get("profiler") or {}
    samples = prof.get("ordered_dispatch_samples") or prof.get("dispatch_samples") or []
    by = defaultdict(float)
    total = 0.0
    n = 0
    for s in samples:
        g = s.get("gpu_us")
        if g is None:
            continue
        by[s.get("kernel_name", "?")] += float(g)
        total += float(g)
        n += 1
    top = sorted(by.items(), key=lambda kv: -kv[1])[:12]
    return total, n, top

control_ids = []
cand_ids = []
ctrl_rows = []
cand_rows = []
ctrl_gpu = []
cand_gpu = []

for i in range(1, trials + 1):
    c = load("control", i)
    k = load("candidate", i)
    control_ids.append(c.get("execution", {}).get("completion_token_ids"))
    cand_ids.append(k.get("execution", {}).get("completion_token_ids"))
    cm = step_metrics(c)
    km = step_metrics(k)
    ctrl_rows.append(cm)
    cand_rows.append(km)
    ctrl_gpu.append(gpu_sum(c))
    cand_gpu.append(gpu_sum(k))

print("\n=== BIT-IDENTITY (completion_token_ids) ===")
identical = all(a == b for a, b in zip(control_ids, cand_ids)) and len(set(map(tuple, control_ids))) == 1
print("control sequences:", control_ids)
print("candidate sequences:", cand_ids)
print("bit_identical_all_trials:", identical and control_ids == cand_ids)

print("\n=== STRUCTURAL (completion step 0) — contamination-proof ===")
def show(label, rows):
    cbs = [r["command_buffers"] for r in rows if r]
    disp = [r["metal_dispatches"] for r in rows if r]
    routes = [r["host_route_id_readbacks"] for r in rows if r]
    print(f"{label}:")
    print(f"  command_buffers/token: {cbs}  (median {median(cbs) if cbs else None})")
    print(f"  metal_dispatches/token: {disp}  (median {median(disp) if disp else None})")
    print(f"  host_route_id_readbacks/token: {routes}  (median {median(routes) if routes else None})")

show("CONTROL (host-bind)", ctrl_rows)
show("CANDIDATE (device table)", cand_rows)

print("\n=== GPU us (gpu_prod) — headline ===")
def show_gpu(label, rows):
    totals = [t for t, n, _ in rows]
    print(f"{label}: total_gpu_us samples={ [n for _,n,_ in rows] } totals={totals} median={median(totals) if totals else None}")
    # last trial top kernels
    if rows:
        print("  top kernels (last trial):")
        for k, v in rows[-1][2]:
            print(f"    {k}: {v:.1f} us")

show_gpu("CONTROL", ctrl_gpu)
show_gpu("CANDIDATE", cand_gpu)
if ctrl_gpu and cand_gpu and median([t for t,_,_ in ctrl_gpu]) > 0:
    ratio = median([t for t,_,_ in cand_gpu]) / median([t for t,_,_ in ctrl_gpu])
    print(f"paired A/B GPU us ratio candidate/control (median): {ratio:.4f}")

print("\n=== WALL (CONTAMINATED) ===")
ctrl_us = [r["elapsed_us"] for r in ctrl_rows if r and r["elapsed_us"]]
cand_us = [r["elapsed_us"] for r in cand_rows if r and r["elapsed_us"]]
def tps(us):
    return 1e6 / us if us else None
print("control elapsed_us:", ctrl_us, "median_tok_s_CONTAMINATED:", tps(median(ctrl_us)) if ctrl_us else None)
print("candidate elapsed_us:", cand_us, "median_tok_s_CONTAMINATED:", tps(median(cand_us)) if cand_us else None)

# Success rule
c_routes = median([r["host_route_id_readbacks"] for r in cand_rows if r])
c_cbs = median([r["command_buffers"] for r in cand_rows if r])
print("\n=== SUCCESS RULE ===")
print(f"candidate host_route_id_readbacks median={c_routes} (target 0)")
print(f"candidate command_buffers median={c_cbs} (target 1-2)")
print(f"bit_identity={identical and control_ids==cand_ids}")
win = (c_routes == 0 and c_cbs <= 2 and identical and control_ids == cand_ids)
print("WIN:", win)
PY
