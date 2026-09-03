#!/usr/bin/env bash
# Owner/gate-profile verification for all-layer Q30 capture restore + sharding.
# Requires Metal (refuses under seatbelt sandboxes with nil MTL device).
#
# Runs:
#   1) single-process sealed 32-probe capture (~3929 tokens)
#   2) 4 parallel shards of the same corpus
#   3) merge of the 4 shards
#   4) comparison of totals / probe ids / hidden counts (and optional route digests)
#
# Usage:
#   lab/operators/q30_all_layer_capture_shard_verify.sh [BIN]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Prefer caller-supplied binary; else worktree/repo release example.
BIN="${1:-$ROOT/target/release/examples/ascension_qwen30_broad_activation_all_layer_route_capture}"
HAWK_ROOT="${HAWK_ROOT:-/Users/scammermike/Downloads/hawking}"
MANIFEST="${MANIFEST:-$HAWK_ROOT/workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity/QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json}"
INPUT="${INPUT:-$HAWK_ROOT/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/broad-activation-v1/requests/QWEN30_BROAD_ACTIVATION_L0_ROUTE_CAPTURE_INPUT_901a24bdcfc6c1d2.json}"
EXISTING="${EXISTING:-$HAWK_ROOT/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/all-layer-activation-v1/runs/main_20260809T204844Z}"
OUT_BASE="${OUT_BASE:-$ROOT/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-diagnostics/all-layer-activation-v1/runs}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
METRICS="$OUT_BASE/verify_metrics_${TS}"
SINGLE_OUT="$OUT_BASE/verify_single_${TS}"
MERGE_OUT="$OUT_BASE/verify_merged_${TS}"
SHARD_COUNT=4

if [[ ! -x "$BIN" ]]; then
  echo "missing executable: $BIN" >&2
  echo "build with: cargo build -p hawking-core --release --example ascension_qwen30_broad_activation_all_layer_route_capture" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" || ! -f "$INPUT" ]]; then
  echo "missing manifest or input" >&2
  exit 2
fi

mkdir -p "$METRICS" "$OUT_BASE"
COMMON=(
  --manifest "$MANIFEST"
  --expected-manifest-seal-sha256 3321a99d719e70499663b7bfebe14dd6c732bfc533bb05b9277eb398e44d6357
  --expected-source-audit-seal-sha256 00ed3e495416c2cbafbcdb7800528e15f243b1a13f5f4af13240109c8fc69f7b
  --expected-source-revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120
  --input-json "$INPUT"
)

echo "== single-process =="
echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$METRICS/single.log"
/usr/bin/time -l "$BIN" "${COMMON[@]}" --output-dir "$SINGLE_OUT" \
  >"$METRICS/single.stdout" 2>"$METRICS/single.time"
echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$METRICS/single.log"
cat "$METRICS/single.time" | tee -a "$METRICS/single.log"
cat "$METRICS/single.stdout" | tee -a "$METRICS/single.log"

echo "== 4-shard parallel =="
echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$METRICS/shards.log"
SHARD_PIDS=()
for i in $(seq 0 $((SHARD_COUNT - 1))); do
  SHARD_OUT="$OUT_BASE/verify_shard${i}_${TS}"
  (
    /usr/bin/time -l "$BIN" "${COMMON[@]}" \
      --output-dir "$SHARD_OUT" \
      --probe-shard "$i" \
      --probe-shard-count "$SHARD_COUNT" \
      >"$METRICS/shard${i}.stdout" 2>"$METRICS/shard${i}.time"
  ) &
  SHARD_PIDS+=($!)
done
FAIL=0
for pid in "${SHARD_PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done
echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ) fail=$FAIL" | tee -a "$METRICS/shards.log"
if [[ "$FAIL" -ne 0 ]]; then
  echo "one or more shards failed" >&2
  exit 2
fi
for i in $(seq 0 $((SHARD_COUNT - 1))); do
  echo "--- shard $i ---" | tee -a "$METRICS/shards.log"
  cat "$METRICS/shard${i}.time" | tee -a "$METRICS/shards.log"
  cat "$METRICS/shard${i}.stdout" | tee -a "$METRICS/shards.log"
done

echo "== merge =="
SHARD_ARGS=()
for i in $(seq 0 $((SHARD_COUNT - 1))); do
  SHARD_ARGS+=(--shard-dir "$OUT_BASE/verify_shard${i}_${TS}")
done
python3 "$ROOT/lab/operators/q30_all_layer_capture_merge_shards.py" \
  "${SHARD_ARGS[@]}" \
  --output-dir "$MERGE_OUT" \
  --input-json "$INPUT" | tee "$METRICS/merge.stdout"

echo "== compare single vs merged vs existing =="
python3 - "$SINGLE_OUT" "$MERGE_OUT" "$EXISTING" <<'PY'
import json, sys
from pathlib import Path

def load_summary(run: Path):
    doc = json.loads((run / "capture-result.json").read_text())
    summary = doc["capture_summary"]
    probes = [p["probe_id"] for p in doc["probes"]]
    hidden_files = sum(1 for _ in (run / "hidden").rglob("*.f32le")) if (run / "hidden").is_dir() else 0
    return {
        "probe_count": summary["probe_count"],
        "total_tokens": summary["total_tokens"],
        "layers_executed": summary["layers_executed"],
        "hidden_tokens_retained": summary["hidden_tokens_retained"],
        "probe_ids": probes,
        "hidden_files": hidden_files,
        "status": doc.get("status"),
    }

single_path, merge_path, existing_path = map(Path, sys.argv[1:4])
single = load_summary(single_path)
merged = load_summary(merge_path)
print("single", {k: single[k] for k in single if k != "probe_ids"})
print("merged", {k: merged[k] for k in merged if k != "probe_ids"})
diffs = []
for key in ("probe_count", "total_tokens", "layers_executed", "hidden_tokens_retained", "probe_ids", "hidden_files"):
    if single[key] != merged[key]:
        diffs.append((key, single[key], merged[key]))
if diffs:
    print("SINGLE_VS_MERGED_DIFFS", diffs)
    raise SystemExit(2)
print("SINGLE_VS_MERGED_MATCH")

if existing_path.is_dir() and (existing_path / "capture-result.json").is_file():
    # Stream existing summary without loading full probes body if possible.
    # File is pretty-printed; capture_summary is near the top after bounded_storage.
    text = existing_path.joinpath("capture-result.json").read_text(encoding="utf-8")
    # cheap extract via json if memory allows (~155MB)
    existing = json.loads(text)
    es = existing["capture_summary"]
    print("existing", {
        "probe_count": es["probe_count"],
        "total_tokens": es["total_tokens"],
        "layers_executed": es["layers_executed"],
        "hidden_tokens_retained": es["hidden_tokens_retained"],
    })
    for key, expected in (
        ("probe_count", 32),
        ("total_tokens", 3929),
        ("layers_executed", 48),
        ("hidden_tokens_retained", 1024),
    ):
        if single[key] != expected:
            print(f"SINGLE_VS_EXISTING_MISMATCH {key}: got {single[key]} expected {expected}")
            raise SystemExit(2)
    print("SINGLE_VS_EXISTING_TOTALS_MATCH")
else:
    print("EXISTING_RUN_ABSENT_SKIPPED")
print("OK")
PY

echo "METRICS=$METRICS"
echo "SINGLE_OUT=$SINGLE_OUT"
echo "MERGE_OUT=$MERGE_OUT"
echo "done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
