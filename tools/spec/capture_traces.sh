#!/usr/bin/env bash
# =============================================================================
# tools/spec/capture_traces.sh — Track 6.2 offline replay oracle trace capture.
#
# Generates "verifier traces" — the ground-truth token sequences needed to
# evaluate candidate draft policies OFFLINE without running the full model.
# Each trace records:
#   - prompt_tokens   : tokenized prompt context (byte-level proxy or real IDs)
#   - generated_tokens: the greedy oracle sequence (true next tokens)
#   - positions       : context-length at each generation step
#
# Output (JSONL, one object per prompt):
#   {"prompt_tokens": [1,2,...], "generated_tokens": [4,5,...], "positions": [N,...]}
#
# USAGE:
#   tools/spec/capture_traces.sh
#   tools/spec/capture_traces.sh --out-dir traces/ --tokens 200
#   SAMPLE_FILE=prompts.txt OUT_DIR=traces tools/spec/capture_traces.sh
#
# ENVIRONMENT:
#   SAMPLE_FILE   path to prompt list (default: tools/spec/sample_prompts.txt)
#   TOKENS        generation length per prompt (default: 200)
#   OUT_DIR       directory for output traces.jsonl + per-prompt logs
#                 (default: traces)
#   BIN           dismantle binary (default: ./target/release/hawking)
#   WEIGHTS       GGUF model file (default: models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   PROFILE       kernel profile JSON (default: profiles/qwen3b-instruct-q4k.m3pro18.json)
#   PY            python3 binary (default: .venv/bin/python or /usr/bin/python3)
#
# NOTE ON TOKENS:
#   dismantle generate does not yet expose a --dump-tokens flag. This script
#   tries to parse a [tokens: ...] line from generate output; if absent, falls
#   back to treating the generated text as UTF-8 bytes (a conservative lower
#   bound on real token-level prediction rates).
#
#   prompt_tokens are similarly byte-encoded from the prompt string. Both
#   encodings are consistent within a trace, so n-gram analysis is valid even
#   if the IDs don't match the real tokenizer's output.
#
# COEXISTENCE:
#   All dismantle subprocesses run under `nice -n 19 taskpolicy -b` so a
#   concurrent foreground GPU job retains priority.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_FILE="${SAMPLE_FILE:-tools/spec/sample_prompts.txt}"
TOKENS="${TOKENS:-200}"
OUT_DIR="${OUT_DIR:-traces}"
BIN="${BIN:-./target/release/hawking}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"

# Python — prefer venv if present
if [[ -x ".venv/bin/python" ]]; then
  PY="${PY:-.venv/bin/python}"
else
  PY="${PY:-/usr/bin/python3}"
fi

# Locked fast-path env (same as ngram_baseline.sh)
BASE_ENV="HAWKING_QWEN_TCB=1 HAWKING_QWEN_VOCAB_PRUNE=32000 \
HAWKING_QWEN_Q4K_LMHEAD=1 HAWKING_QWEN_FFN_DOWN_Q4K=1 \
HAWKING_QWEN_Q4K_PREDEC=1"

die()  { printf 'error: %s\n' "$*" >&2; exit 64; }
warn() { printf 'warn: %s\n'  "$*" >&2; }

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)     OUT_DIR="$2"; shift 2 ;;
    --tokens)      TOKENS="$2"; shift 2 ;;
    --sample-file) SAMPLE_FILE="$2"; shift 2 ;;
    --weights)     WEIGHTS="$2"; shift 2 ;;
    --profile)     PROFILE="$2"; shift 2 ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

# ── Pre-flight ─────────────────────────────────────────────────────────────────
[[ -f "$SAMPLE_FILE" ]] || die "SAMPLE_FILE not found: $SAMPLE_FILE"

SKIP_GENERATE=0
if [[ ! -x "$BIN" ]]; then
  warn "binary not found/executable: $BIN"
  warn "  → run: cargo build --release"
  SKIP_GENERATE=1
fi
if [[ ! -f "$WEIGHTS" ]]; then
  warn "weights not found: $WEIGHTS"
  warn "  → set WEIGHTS= pointing to your GGUF file."
  SKIP_GENERATE=1
fi

mkdir -p "$OUT_DIR"
TRACES_JSONL="$OUT_DIR/traces.jsonl"
: > "$TRACES_JSONL"

# ── Read prompts ───────────────────────────────────────────────────────────────
mapfile -t PROMPTS < <(grep -v '^\s*#' "$SAMPLE_FILE" | grep -v '^\s*$')
N_PROMPTS="${#PROMPTS[@]}"
[[ "$N_PROMPTS" -gt 0 ]] || die "no non-comment lines in $SAMPLE_FILE"

printf '=== capture_traces — Track 6.2 offline replay oracle ===\n'
printf 'sample file : %s (%d prompts)\n' "$SAMPLE_FILE" "$N_PROMPTS"
printf 'tokens/gen  : %s\n' "$TOKENS"
printf 'output      : %s\n' "$TRACES_JSONL"
printf 'model       : %s\n' "$WEIGHTS"
printf '\n'

TMPD="$(mktemp -d /tmp/capture_traces.XXXXXX)"
cleanup() { rm -rf "$TMPD"; }
trap cleanup EXIT

N_OK=0
N_FAIL=0

# ── Per-prompt loop ────────────────────────────────────────────────────────────
for i in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$i]}"
  prompt_short="${prompt:0:55}"
  log_f="$TMPD/gen_${i}.log"
  printf '[%d/%d] %.55s...\n' "$((i+1))" "$N_PROMPTS" "$prompt"

  if [[ "$SKIP_GENERATE" -eq 1 ]]; then
    printf '  SKIP: no binary or weights\n'
    (( N_FAIL++ )) || true
    continue
  fi

  # Run greedy generation with stable seed
  if ! env $BASE_ENV nice -n 19 taskpolicy -b "$BIN" generate \
       --weights "$WEIGHTS" \
       --kernel-profile "$PROFILE" \
       --prompt "$prompt" \
       --max-new-tokens "$TOKENS" \
       --temperature 0 --seed 0 \
       > "$log_f" 2>&1; then
    printf '  FAIL: generate exited non-zero\n'
    (( N_FAIL++ )) || true
    cp "$log_f" "$OUT_DIR/prompt_${i}_fail.log" 2>/dev/null || true
    continue
  fi

  # ── Extract generated tokens ─────────────────────────────────────────────────
  # Priority 1: real [tokens: ...] line if the CLI emits one
  # Priority 2: byte-encode the response text (conservative lower bound)
  "$PY" - "$log_f" "$prompt" "$TOKENS" "$i" "$TRACES_JSONL" <<'PYEOF'
import sys, json, re

log_path  = sys.argv[1]
prompt    = sys.argv[2]
tokens    = int(sys.argv[3])
idx       = int(sys.argv[4])
out_jsonl = sys.argv[5]

with open(log_path, "r", errors="replace") as fh:
    raw = fh.read()

lines = raw.splitlines()

# --- Try to extract real token IDs -------------------------------------------
# Format expected from a future --dump-tokens flag:
#   [tokens: 1234 5678 9012 ...]
tok_line = None
for line in lines:
    if line.startswith("[tokens:"):
        tok_line = line
        break

if tok_line:
    ids_str = re.sub(r"^\[tokens:\s*", "", tok_line).rstrip("]")
    try:
        generated_tokens = [int(x) for x in ids_str.split() if x.strip()]
    except ValueError:
        generated_tokens = None
else:
    generated_tokens = None

# --- Fallback: byte-encode the response text ---------------------------------
if not generated_tokens:
    # Strip [stats] and other bracket lines from stdout to isolate response
    response_lines = [l for l in lines if not l.startswith("[") and l.strip()]
    response_text  = "\n".join(response_lines)
    if not response_text.strip():
        # Last resort: use entire raw output
        response_text = raw
    generated_tokens = list(response_text.encode("utf-8", errors="replace"))

if len(generated_tokens) < 2:
    print(f"  WARN: idx={idx} too few generated tokens ({len(generated_tokens)})", file=sys.stderr)
    sys.exit(1)

# --- Encode prompt as byte-level token IDs (consistent with fallback path) ---
prompt_tokens = list(prompt.encode("utf-8", errors="replace"))

# --- Build positions: context length grows from len(prompt_tokens) onward ----
base_pos = len(prompt_tokens)
positions = list(range(base_pos, base_pos + len(generated_tokens)))

# --- Emit JSONL record -------------------------------------------------------
record = {
    "prompt_idx":       idx,
    "prompt_tokens":    prompt_tokens,
    "generated_tokens": generated_tokens,
    "positions":        positions,
    "n_generated":      len(generated_tokens),
    "token_encoding":   "real_ids" if tok_line else "utf8_bytes",
}
with open(out_jsonl, "a") as fh:
    fh.write(json.dumps(record, separators=(",", ":")) + "\n")

print(f"  ok  n_gen={len(generated_tokens)}  encoding={'real_ids' if tok_line else 'utf8_bytes'}  pos_range=[{positions[0]},{positions[-1]}]")
PYEOF
  py_exit=$?
  if [[ "$py_exit" -eq 0 ]]; then
    (( N_OK++ )) || true
  else
    (( N_FAIL++ )) || true
  fi
done

# ── Summary ────────────────────────────────────────────────────────────────────
printf '\n'
printf '=== capture_traces summary ===\n'
printf '  captured : %d\n' "$N_OK"
printf '  failed   : %d\n' "$N_FAIL"
printf '  output   : %s  (%d lines)\n' "$TRACES_JSONL" "$(wc -l < "$TRACES_JSONL" 2>/dev/null || echo 0)"

if [[ "$N_OK" -gt 0 ]]; then
  printf '\nTo run the replay oracle:\n'
  printf '  python3 tools/spec/replay_oracle.py --traces %s --policy ngram,last-repeat\n' "$TRACES_JSONL"
fi

[[ "$N_OK" -gt 0 ]] || { printf '\nerror: no traces captured\n' >&2; exit 1; }
