#!/usr/bin/env bash
# Pack the huihui Qwen3.8-27B bf16 P0 parent into a language-only uniform-Q4
# .gravity artifact and measure native Metal decode three times, matching the
# historical GROUND_TRUTH_TPS methodology (complete-wall, 128 new tokens, pairs 3).
#
# The three-part bar is evaluated honestly in the receipt:
#   1. zero fallbacks
#   2. zero dense_w_materialized if the binary exposes a live counter
#   3. generated text is coherent on-topic prose about compilers
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

SOURCE="${SOURCE:-$HOME/models/qwen3.8-27b-abliterated-bf16}"
ROOT="${ROOT:-$HOME/models/qwen38-gravity-uniform-q4-v1}"
TOKENIZER="${TOKENIZER:-$SOURCE/tokenizer.json}"
PACK_BIN="${PACK_BIN:-$REPO/workspace/ops/build/rust/release-fast/examples/ascension_qwen38_pack}"
DECODE_BIN="${DECODE_BIN:-$REPO/workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy}"
RECEIPT_DIR="$REPO/receipts/headless"
RECEIPT="$RECEIPT_DIR/QWEN38_GRAVITY_NATIVE.json"
RAW="$RECEIPT_DIR/QWEN38_GRAVITY_NATIVE_RAW.json"
PACK_REPORT="$RECEIPT_DIR/QWEN38_GRAVITY_NATIVE_PACK.json"
PROMPT='Explain, in ordinary prose and at length, how a compiler turns a for-loop into basic blocks and then into machine code.'
# Default max_seq_len in the binary is 128. Historical runs used 25 prompt tokens
# + 128 new tokens, which does not fit. 256 is the smallest power-of-two that
# holds the chat-templated prompt plus 128 generated tokens.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PAIRS="${PAIRS:-3}"
N_RUNS="${N_RUNS:-3}"

mkdir -p "$RECEIPT_DIR"

die() { echo "qwen38_gravity_native_bench: $*" >&2; exit 1; }

disk_avail_bytes() {
  df -k / | awk 'NR==2 {print $4 * 1024}'
}

gpu_conflicts() {
  /bin/ps -eo pid,command 2>/dev/null | /usr/bin/grep -E 'llama-server|mlx_lm|ascension_qwen38' | /usr/bin/grep -v grep | /usr/bin/grep -v qwen38_gravity_native_bench || true
}

echo "== GPU occupancy =="
conflicts="$(gpu_conflicts)"
if [[ -n "$conflicts" ]]; then
  echo "conflicting decode process(es) resident; waiting up to 600s" >&2
  echo "$conflicts" >&2
  deadline=$((SECONDS + 600))
  while [[ -n "$(gpu_conflicts)" && $SECONDS -lt $deadline ]]; do
    sleep 15
  done
  conflicts="$(gpu_conflicts)"
  if [[ -n "$conflicts" ]]; then
    die "GPU lane blocked after 600s:
$conflicts"
  fi
fi
echo "NO_CONFLICTING_PROCESSES"

[[ -x "$PACK_BIN" ]] || die "missing pack binary $PACK_BIN"
[[ -x "$DECODE_BIN" ]] || die "missing decode binary $DECODE_BIN"
[[ -f "$SOURCE/config.json" ]] || die "missing $SOURCE/config.json"
[[ -f "$SOURCE/model.safetensors.index.json" ]] || die "missing source index"
[[ -f "$TOKENIZER" ]] || die "missing tokenizer $TOKENIZER"

GIT_HEAD="$(git -C "$REPO" rev-parse HEAD)"
GIT_STATUS="$(git -C "$REPO" status --porcelain -- crates/hawking-core/src/model/qwen38_pack.rs crates/hawking-core/src/model/qwen38_hybrid_decode.rs crates/hawking-core/examples/ascension_qwen38_pack.rs crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs || true)"
DISK_BEFORE="$(disk_avail_bytes)"
echo "git HEAD $GIT_HEAD"
echo "disk_free_before_bytes $DISK_BEFORE"

echo "== pack =="
PACK_STARTED="$(date +%s)"
set +e
"$PACK_BIN" --source "$SOURCE" --root "$ROOT" >"$PACK_REPORT.stdout" 2>"$PACK_REPORT.stderr"
PACK_EC=$?
set -e
PACK_ENDED="$(date +%s)"
PACK_WALL=$((PACK_ENDED - PACK_STARTED))
if [[ $PACK_EC -eq 0 ]]; then
  cp "$PACK_REPORT.stdout" "$PACK_REPORT"
else
  {
    echo '{"status":"failed","exit_code":'"$PACK_EC"',"wall_s":'"$PACK_WALL"'}'
  } >"$PACK_REPORT"
  echo "pack failed exit=$PACK_EC wall=${PACK_WALL}s" >&2
  echo "----- pack stdout -----" >&2
  cat "$PACK_REPORT.stdout" >&2 || true
  echo "----- pack stderr -----" >&2
  cat "$PACK_REPORT.stderr" >&2 || true
  die "pack failed (exit $PACK_EC). See $PACK_REPORT.stderr"
fi
ARTIFACT_BYTES="$(python3 - "$ROOT" <<'PY'
import os, sys
root = sys.argv[1]
total = 0
for dirpath, _, files in os.walk(root):
    for name in files:
        p = os.path.join(dirpath, name)
        if os.path.islink(p):
            continue
        total += os.path.getsize(p)
print(total)
PY
)"
ARTIFACT_HUMAN="$(du -sh "$ROOT" | awk '{print $1}')"
echo "pack ok wall=${PACK_WALL}s size=$ARTIFACT_HUMAN bytes=$ARTIFACT_BYTES"

echo "== decode x${N_RUNS} =="
RUN_PATHS=()
for i in $(seq 1 "$N_RUNS"); do
  conflicts="$(gpu_conflicts)"
  if [[ -n "$conflicts" ]]; then
    die "GPU lane occupied before decode run $i:
$conflicts"
  fi
  out="$RECEIPT_DIR/QWEN38_GRAVITY_NATIVE_RAW.run${i}.json"
  log="$RECEIPT_DIR/QWEN38_GRAVITY_NATIVE_RAW.run${i}.log"
  echo "-- run $i -> $out --"
  set +e
  "$DECODE_BIN" \
    --artifact-root "$ROOT" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --complete-wall \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --pairs "$PAIRS" \
    --out "$out" >"$log" 2>&1
  ec=$?
  set -e
  echo "run $i exit=$ec"
  if [[ $ec -ne 0 ]]; then
    echo "----- run $i log (tail) -----" >&2
    tail -n 80 "$log" >&2 || true
    die "decode run $i failed (exit $ec)"
  fi
  RUN_PATHS+=("$out")
done

python3 - "$REPO" "$RECEIPT" "$RAW" "$PACK_REPORT" "$ROOT" "$SOURCE" "$GIT_HEAD" \
  "$DISK_BEFORE" "$PACK_EC" "$PACK_WALL" "$ARTIFACT_BYTES" "$MAX_SEQ_LEN" \
  "$MAX_NEW_TOKENS" "$PAIRS" "$PACK_BIN" "$DECODE_BIN" "$PROMPT" "${RUN_PATHS[@]}" <<'PY'
import json, os, statistics, subprocess, sys, time, hashlib, re

repo = sys.argv[1]
receipt_path = sys.argv[2]
raw_path = sys.argv[3]
pack_report_path = sys.argv[4]
root = sys.argv[5]
source = sys.argv[6]
git_head = sys.argv[7]
disk_before = int(sys.argv[8])
pack_ec = int(sys.argv[9])
pack_wall = int(sys.argv[10])
artifact_bytes = int(sys.argv[11])
max_seq_len = int(sys.argv[12])
max_new_tokens = int(sys.argv[13])
pairs = int(sys.argv[14])
pack_bin = sys.argv[15]
decode_bin = sys.argv[16]
prompt = sys.argv[17]
run_paths = sys.argv[18:]

def sh(cmd):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True).stdout.strip()

def load(p):
    with open(p) as f:
        return json.load(f)

pack = load(pack_report_path)
runs = []
for i, p in enumerate(run_paths, 1):
    body = load(p)
    auth = body.get("authority") or {}
    ident = body.get("identity") or {}
    cold = body.get("cold_generate") or {}
    tps = auth.get("headline_complete_tps")
    wall_ns = auth.get("headline_complete_wall_ns_per_token")
    gpu_ns = auth.get("headline_gpu_ns_per_token")
    text = ident.get("generated_text") or cold.get("generated_text") or ""
    # complete-wall JSON puts generated_text on each warm summary and cold.
    if not text:
        warm = body.get("warm_reps") or []
        if warm:
            text = ((warm[0].get("summary") or {}).get("generated_text")) or ""
    fallbacks = ident.get("fallbacks")
    if fallbacks is None:
        fallbacks = cold.get("fallbacks")
    dense = body.get("dense_w_materialized")
    dense_exposed = "dense_w_materialized" in body or "dense_w_materialized" in ident
    runs.append({
        "run": i,
        "path": p,
        "headline_complete_tps": tps,
        "headline_complete_wall_ns_per_token": wall_ns,
        "headline_gpu_ns_per_token": gpu_ns,
        "fallbacks": fallbacks,
        "dense_w_materialized": dense,
        "dense_w_materialized_exposed": dense_exposed,
        "generated_text": text,
        "prompt_len": ident.get("prompt_len") or cold.get("prompt_len"),
        "greedy_new_token_ids": ident.get("greedy_new_token_ids"),
        "n_steady_decode_steps": (cold.get("n_steady_decode_steps")
                                  if isinstance(cold, dict) else None),
        "timing_label": body.get("timing_label"),
    })

tps_vals = [r["headline_complete_tps"] for r in runs if r["headline_complete_tps"]]
wall_vals = [r["headline_complete_wall_ns_per_token"] for r in runs
             if r["headline_complete_wall_ns_per_token"]]

def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]

def spread_pct(xs):
    v = sorted(x for x in xs if x)
    if not v or v[0] == 0:
        return None
    return round(100 * (v[-1] - v[0]) / v[0], 1)

med_tps = median(tps_vals)
med_wall = median(wall_vals)
spread = spread_pct(tps_vals)

# Coherence: mechanical gate plus the quoted excerpt for a human reader.
COMPILER_HINTS = (
    "compiler", "basic block", "machine code", "instruction", "assembly",
    "intermediate", "ir ", "ssa", "control flow", "loop", "for-loop",
    "for loop", "backend", "frontend", "register", "codegen", "code generation",
    "ast", "parse", "lexer", "token",
)

def too_repetitive(text, n=12):
    words = re.findall(r"\S+", text)
    if len(words) < n * 3:
        return False
    # flag if any n-gram occupies > 40% of the stream
    grams = [" ".join(words[i:i+n]) for i in range(0, max(0, len(words)-n+1))]
    if not grams:
        return False
    from collections import Counter
    c = Counter(grams)
    top, k = c.most_common(1)[0]
    return (k / len(grams)) > 0.40

texts = [r["generated_text"] or "" for r in runs]
chosen = max(texts, key=len) if texts else ""
lower = chosen.lower()
hint_hits = [h for h in COMPILER_HINTS if h in lower]
coherence_reasons = []
if len(chosen.strip()) < 120:
    coherence_reasons.append("generated text shorter than 120 chars")
if too_repetitive(chosen):
    coherence_reasons.append("high n-gram repetition")
if not hint_hits:
    coherence_reasons.append("no compiler-related vocabulary")
# garbled: high ratio of replacement chars / empty tokens / same token spam
if chosen.count("\ufffd") > 3:
    coherence_reasons.append("replacement characters present")
coherence_pass = not coherence_reasons

fallback_total = 0
fallback_ok = True
for r in runs:
    f = r["fallbacks"]
    if f is None:
        fallback_ok = False
        coherence_reasons.append("fallbacks field missing")
        break
    fallback_total += int(f)
    if int(f) != 0:
        fallback_ok = False

dense_notes = (
    "complete-wall JSON does not expose a live dense_w_materialized counter; "
    "the default generate path hardcodes dense_w_materialized=0. "
    "The session never increments fallbacks on this path: a missing codec "
    "fails the run instead of reconstructing."
)
dense_exposed = any(r["dense_w_materialized_exposed"] for r in runs)
if dense_exposed:
    dense_vals = [r["dense_w_materialized"] for r in runs]
    dense_ok = all(v in (0, None) for v in dense_vals) and all(
        v == 0 for v in dense_vals if v is not None)
else:
    dense_ok = True  # not exposed; do not fail the bar on a missing counter

bar_fail = []
if not fallback_ok or fallback_total != 0:
    bar_fail.append(f"fallbacks={fallback_total} (must be 0)")
if dense_exposed and not dense_ok:
    bar_fail.append("dense_w_materialized is non-zero")
if not coherence_pass:
    bar_fail.append("coherence: " + "; ".join(coherence_reasons))
bar = "PASS" if not bar_fail else "FAIL"

excerpt = chosen[:1200]

df = subprocess.run(["df", "-k", "/"], capture_output=True, text=True, check=True)
disk_after = int(df.stdout.splitlines()[1].split()[3]) * 1024

bpw = pack.get("complete_physical_bpw")
receipt = {
    "schema": "hawking.headless.qwen38_gravity_native.v1",
    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "bar": bar,
    "bar_reasons": bar_fail,
    "three_part_bar": {
        "zero_fallbacks": fallback_ok and fallback_total == 0,
        "fallbacks_total": fallback_total,
        "zero_dense_w_materialized": None if not dense_exposed else dense_ok,
        "dense_w_materialized_exposed": dense_exposed,
        "dense_w_materialized_note": dense_notes,
        "coherent_compiler_prose": coherence_pass,
        "coherence_reasons": coherence_reasons,
        "compiler_hint_hits": hint_hits,
        "generated_text_excerpt": excerpt,
        "generated_text_chars": len(chosen),
    },
    "git": {
        "commit": git_head,
        "working_tree_note": (
            "built from working tree at this commit; qwen38_hybrid_decode.rs "
            "and qwen38_device_activations.metal were already dirty before this campaign"
        ),
    },
    "commands": {
        "build_pack": "cargo build --profile release-fast -p hawking-core --example ascension_qwen38_pack",
        "build_decode": "cargo build --profile release-fast -p hawking-core --example ascension_qwen38_hybrid_greedy",
        "pack": f"{pack_bin} --source {source} --root {root}",
        "decode": (
            f"{decode_bin} --artifact-root {root} --tokenizer {source}/tokenizer.json "
            f"--prompt {prompt!r} --complete-wall --max-new-tokens {max_new_tokens} "
            f"--max-seq-len {max_seq_len} --pairs {pairs} --out {raw_path}"
        ),
        "contract_mismatch_max_seq_len": (
            "the contract command omitted --max-seq-len; the binary default is 128, "
            "which cannot hold historical 25 prompt tokens + 128 new tokens. "
            f"This run used --max-seq-len {max_seq_len}."
        ),
    },
    "binaries": {
        "pack": pack_bin,
        "decode": decode_bin,
        "profile": "release-fast",
    },
    "source": source,
    "artifact_root": root,
    "pack": {
        "exit_code": pack_ec,
        "wall_s": pack_wall,
        "report": pack,
        "artifact_bytes": artifact_bytes,
        "complete_physical_bpw": bpw,
    },
    "decode": {
        "n_process_runs": len(runs),
        "pairs_per_process": pairs,
        "max_new_tokens": max_new_tokens,
        "max_seq_len": max_seq_len,
        "per_run_complete_wall_tps": tps_vals,
        "per_run_complete_wall_ns": wall_vals,
        "median_complete_wall_tps": med_tps,
        "median_complete_wall_ns": med_wall,
        "measurement_spread_pct": spread,
        "runs": [
            dict({k: v for k, v in r.items() if k != "generated_text"},
                 generated_text_excerpt=(r.get("generated_text") or "")[:400])
            for r in runs
        ],
    },
    "disk": {
        "free_bytes_before": disk_before,
        "free_bytes_after": disk_after,
        "free_gib_before": round(disk_before / 1024**3, 2),
        "free_gib_after": round(disk_after / 1024**3, 2),
    },
    "historical_anchor": {
        "receipt": "receipts/ascent-2026-08-18/GROUND_TRUTH_TPS.json",
        "uniform_q4_complete_wall_tps": 33.10,
        "complete_bpw": 4.2560,
        "fallbacks": 0,
        "dense_w_materialized": 0,
    },
}

os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
with open(receipt_path, "w") as f:
    json.dump(receipt, f, indent=2)
    f.write("\n")

# Combined RAW index (per-run files stay authoritative for the binary schema).
with open(raw_path, "w") as f:
    json.dump({
        "schema": "hawking.headless.qwen38_gravity_native_raw.v1",
        "runs": run_paths,
        "generated_text": chosen,
        "per_run_complete_wall_tps": tps_vals,
    }, f, indent=2)
    f.write("\n")

print(f"bar={bar}")
print(f"median_tps={med_tps}")
print(f"spread_pct={spread}")
print(f"fallbacks={fallback_total}")
print(f"wrote {receipt_path}")

# Ledger + registry via the existing CLIs only.
ledger_row = {
    "generation": "qwen38-gravity-uniform-q4-v1",
    "measured_by": "tools/headless/qwen38_gravity_native_bench.sh",
    "model_identity": root,
    "runtime_build": f"{git_head}+release-fast native gravity",
    "source_sha": git_head,
    "context": max_seq_len,
    "decode_tps": med_tps,
    "complete_token_wall_ns": med_wall,
    "workers_resident": 1,
    "workers_decoding": 1,
    "measurement_reps": len(runs),
    "measurement_spread_pct": spread,
    "bpw": bpw,
    "memory_bytes": artifact_bytes,
    "fallbacks": fallback_total,
    "bar": bar,
}
ledger_json = json.dumps(ledger_row)
r = subprocess.run(
    ["python3", os.path.join(repo, "tools/headless/performance_ledger.py"),
     "record", "--json", ledger_json],
    cwd=repo, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    sys.stderr.write(r.stderr)
    sys.exit(r.returncode)

notes = (
    "native Hawking .gravity runtime, uniform Q4, rebuilt from the huihui bf16 P0 "
    "parent (steer S001) after the historical artifact was reclaimed"
)
if bar != "PASS":
    notes += f"; BAR {bar}: " + "; ".join(bar_fail)

recipe = {
    "derived_from": "qwen38-huihui-bf16-P0",
    "quant": "uniform_q4",
    "runtime": "hawking-native-gravity",
    "codec": "uniform-q4-v1",
    "pack_command": f"ascension_qwen38_pack --source {source} --root {root}",
    "bar": bar,
}
r = subprocess.run(
    ["python3", os.path.join(repo, "tools/headless/model_registry.py"),
     "register",
     "--name", "qwen38-gravity-uniform-q4-v1",
     "--path", root,
     "--role", "candidate",
     "--notes", notes,
     "--recipe", json.dumps(recipe)],
    cwd=repo, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    sys.stderr.write(r.stderr)
    sys.exit(r.returncode)
PY

echo "== done =="
echo "receipt $RECEIPT"
df -h / | tail -1
