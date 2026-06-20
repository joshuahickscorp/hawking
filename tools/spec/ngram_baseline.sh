#!/usr/bin/env bash
# =============================================================================
# tools/spec/ngram_baseline.sh — Track 6.1 n-gram spec oracle.
#
# Oracle analysis: measures what acceptance rate a simple n-gram draft would
# achieve on a given workload WITHOUT implementing spec-decode.  This is the
# cheap feasibility gate before investing in a trained draft head.
#
# METHODOLOGY
#   For each prompt in SAMPLE_FILE:
#     1. Generate a full greedy response (temperature=0, seed=0) via
#        `dismantle generate`.
#     2. Extract the token sequence from the output (raw text → byte tokens
#        as a conservative proxy; see NOTE ON TOKENS below).
#     3. Run ngram_analysis.py to compute oracle depth-1 acceptance rates for
#        n-gram orders in N_GRAMS.
#   Print per-prompt stats + aggregate summary.
#
#   VERDICT: if mean oracle_accept_rate ≥ 55%, a trained n-gram draft is
#   likely to clear the spec-decode payoff threshold.
#
# NOTE ON TOKENS
#   dismantle generate does not (yet) expose a --dump-tokens flag in the
#   public CLI.  This script uses two approaches in order:
#     a. Look for a [tokens: ...] line in the generate output (if added in
#        future).
#     b. Fall back to treating the raw response text as a byte sequence.
#        Byte-level n-gram rates are a LOWER BOUND on real token-level rates
#        (byte trigrams are easier to predict than token bigrams).
#   This is sufficient for the "kill or go" oracle decision.
#
# USAGE
#   SAMPLE_FILE=prompts.txt tools/spec/ngram_baseline.sh
#   SAMPLE_FILE=prompts.txt TOKENS=100 N_GRAMS="2 3" tools/spec/ngram_baseline.sh
#
#   Provide one prompt per line in SAMPLE_FILE.  Lines starting with # are
#   treated as comments and skipped.
#
# ENVIRONMENT
#   SAMPLE_FILE      path to prompt list (default: tools/spec/sample_prompts.txt)
#   TOKENS           generation length per prompt (default: 200)
#   N_GRAMS          space-separated n-gram orders (default: "2 3 4")
#   MIN_FREQ         minimum n-gram frequency to count as a prediction (default: 1)
#   BIN              dismantle binary (default: ./target/release/hawking)
#   WEIGHTS          GGUF model file (default: models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   PROFILE          kernel profile JSON (default: profiles/qwen3b-instruct-q4k.m3pro18.json)
#   OUT_JSON         write aggregate JSON to this path (default: reports/ngram_oracle.json)
#   OUT_SEQ_DIR      directory to save per-prompt token sequence files (default: none)
#   PY               python3 binary (default: auto-detect .venv/bin/python or python3)
#
# COEXISTENCE
#   All dismantle subprocesses run under `nice -n 19 taskpolicy -b` so a
#   concurrent foreground GPU job retains priority.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_FILE="${SAMPLE_FILE:-tools/spec/sample_prompts.txt}"
TOKENS="${TOKENS:-200}"
N_GRAMS="${N_GRAMS:-2 3 4}"
MIN_FREQ="${MIN_FREQ:-1}"
BIN="${BIN:-./target/release/hawking}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
OUT_JSON="${OUT_JSON:-reports/ngram_oracle.json}"
OUT_SEQ_DIR="${OUT_SEQ_DIR:-}"

# Python — prefer venv if present
if [[ -x ".venv/bin/python" ]]; then
  PY="${PY:-.venv/bin/python}"
else
  PY="${PY:-/usr/bin/python3}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS_PY="$SCRIPT_DIR/ngram_analysis.py"

die() { printf 'error: %s\n' "$*" >&2; exit 64; }
warn() { printf 'warn: %s\n' "$*" >&2; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
[[ -f "$ANALYSIS_PY" ]] || die "ngram_analysis.py not found at $ANALYSIS_PY"
[[ -f "$SAMPLE_FILE" ]] || die "SAMPLE_FILE not found: $SAMPLE_FILE (create it or set SAMPLE_FILE=...)"

if [[ ! -x "$BIN" ]]; then
  warn "binary not found/executable: $BIN"
  warn "  → set BIN= or run: cargo build --release"
  warn "  Generation step will be SKIPPED; existing .txt files in OUT_SEQ_DIR will be used."
  SKIP_GENERATE=1
else
  SKIP_GENERATE=0
fi

if [[ ! -f "$WEIGHTS" ]]; then
  warn "weights not found: $WEIGHTS"
  warn "  → set WEIGHTS= pointing to your GGUF file."
  warn "  Generation step will be SKIPPED."
  SKIP_GENERATE=1
fi

# ── Temp workspace ────────────────────────────────────────────────────────────
TMPD="$(mktemp -d /tmp/ngram_baseline.XXXXXX)"
cleanup() { rm -rf "$TMPD"; }
trap cleanup EXIT

mkdir -p "$(dirname "$OUT_JSON")"
[[ -n "$OUT_SEQ_DIR" ]] && mkdir -p "$OUT_SEQ_DIR"

# Locked fast-path env (same as paired_lever.sh baseline)
BASE_ENV="HAWKING_QWEN_TCB=1 HAWKING_QWEN_VOCAB_PRUNE=32000 \
HAWKING_QWEN_Q4K_LMHEAD=1 HAWKING_QWEN_FFN_DOWN_Q4K=1 \
HAWKING_QWEN_Q4K_PREDEC=1"

# ── Read prompts ──────────────────────────────────────────────────────────────
mapfile -t PROMPTS < <(grep -v '^\s*#' "$SAMPLE_FILE" | grep -v '^\s*$')
N_PROMPTS="${#PROMPTS[@]}"
[[ "$N_PROMPTS" -gt 0 ]] || die "no non-comment lines in $SAMPLE_FILE"

printf '=== ngram_baseline — Track 6.1 n-gram oracle ===\n'
printf 'sample file  : %s (%d prompts)\n' "$SAMPLE_FILE" "$N_PROMPTS"
printf 'tokens/prompt: %s\n' "$TOKENS"
printf 'n-gram orders: %s\n' "$N_GRAMS"
printf 'min freq     : %s\n' "$MIN_FREQ"
printf 'model        : %s\n' "$WEIGHTS"
printf '\n'

# ── Per-prompt column header ──────────────────────────────────────────────────
printf '%-4s %-50s %10s %10s\n' "idx" "prompt (truncated)" "tok_len" "status"
printf '%s\n' "$(printf '%0.s-' {1..78})"

# ── All-sequences accumulator file ───────────────────────────────────────────
ALL_SEQS_F="$TMPD/all_seqs.txt"
: > "$ALL_SEQS_F"
N_GENERATED=0
N_SKIPPED=0

# ── Per-prompt loop ───────────────────────────────────────────────────────────
for i in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$i]}"
  prompt_short="${prompt:0:48}"
  seq_f="$TMPD/seq_${i}.txt"
  log_f="$TMPD/gen_${i}.log"

  if [[ "$SKIP_GENERATE" -eq 1 ]]; then
    # If OUT_SEQ_DIR is set and has a file for this index, use it
    if [[ -n "$OUT_SEQ_DIR" && -f "$OUT_SEQ_DIR/seq_${i}.txt" ]]; then
      cp "$OUT_SEQ_DIR/seq_${i}.txt" "$seq_f"
    else
      printf '%-4d %-50s %10s %10s\n' "$i" "$prompt_short" "?" "SKIPPED (no binary)"
      (( N_SKIPPED++ ))
      continue
    fi
  else
    # Run generation (nice + taskpolicy for coexistence)
    if env $BASE_ENV nice -n 19 taskpolicy -b "$BIN" generate \
         --weights "$WEIGHTS" \
         --kernel-profile "$PROFILE" \
         --prompt "$prompt" \
         --max-new-tokens "$TOKENS" \
         --temperature 0 --seed 0 \
         > "$log_f" 2>&1; then

      # Try to extract [tokens: ...] line first (future CLI feature)
      tok_line=$(grep -E '^\[tokens:' "$log_f" 2>/dev/null | tail -1 || true)
      if [[ -n "$tok_line" ]]; then
        # Format expected: [tokens: 1234 5678 9012 ...]
        printf '%s\n' "$tok_line" | sed 's/^\[tokens: //;s/\]$//' > "$seq_f"
      else
        # Fallback: extract the generated response text (lines after the
        # [prompt] / [response] / first blank-line boundary).
        # We strip the [stats] line and convert the text to byte values.
        response_text=$(grep -v '^\[' "$log_f" | grep -v '^$' || true)
        if [[ -z "$response_text" ]]; then
          # Last resort: use entire stdout as text
          response_text=$(cat "$log_f")
        fi
        # Convert to space-separated byte values via python3
        printf '%s' "$response_text" | \
          /usr/bin/python3 -c "
import sys
data = sys.stdin.buffer.read()
ids = list(data)
print(' '.join(map(str, ids)))
" > "$seq_f"
      fi
    else
      printf '%-4d %-50s %10s %10s\n' "$i" "$prompt_short" "?" "FAIL (generate exited nonzero)"
      (( N_SKIPPED++ ))
      continue
    fi
  fi

  # Count tokens in sequence file
  tok_len=$(/usr/bin/python3 -c "
import sys
line = open('$seq_f').read().strip()
n = len(line.split()) if line else 0
print(n)
" 2>/dev/null || echo "0")

  if [[ "${tok_len:-0}" -lt 2 ]]; then
    printf '%-4d %-50s %10s %10s\n' "$i" "$prompt_short" "$tok_len" "SKIP (too short)"
    (( N_SKIPPED++ ))
    continue
  fi

  # Append sequence to all-sequences accumulator
  cat "$seq_f" >> "$ALL_SEQS_F"

  # Optionally save per-prompt seq file
  if [[ -n "$OUT_SEQ_DIR" ]]; then
    cp "$seq_f" "$OUT_SEQ_DIR/seq_${i}.txt"
  fi

  printf '%-4d %-50s %10s %10s\n' "$i" "$prompt_short" "$tok_len" "ok"
  (( N_GENERATED++ ))
done

printf '\n'
printf 'generated: %d  skipped: %d\n' "$N_GENERATED" "$N_SKIPPED"

if [[ "$N_GENERATED" -lt 1 ]]; then
  printf 'error: no sequences collected — cannot run oracle analysis.\n' >&2
  exit 1
fi

# ── Run n-gram oracle analysis ────────────────────────────────────────────────
printf '\n--- Running n-gram oracle analysis ---\n'
printf 'n-gram orders: %s  |  min-freq: %s\n\n' "$N_GRAMS" "$MIN_FREQ"

# Build the --ngrams argument list
NGRAM_ARGS=()
for n in $N_GRAMS; do
  NGRAM_ARGS+=(--ngrams "$n")
done

"$PY" "$ANALYSIS_PY" \
  --seqs "$ALL_SEQS_F" \
  "${NGRAM_ARGS[@]}" \
  --min-freq "$MIN_FREQ" \
  --json "$OUT_JSON"

EXIT_CODE=$?

# ── Final verdict banner ──────────────────────────────────────────────────────
if [[ -f "$OUT_JSON" ]]; then
  VERDICT=$(/usr/bin/python3 -c "
import json, sys
d = json.load(open('$OUT_JSON'))
agg = d.get('aggregate', {})
rate = agg.get('mean_oracle_accept_rate', 0)
ok   = agg.get('threshold_55_pct', False)
print(f'{rate:.1%}', 'GO' if ok else 'NO-GO')
" 2>/dev/null || echo "? ?")
  RATE=$(printf '%s' "$VERDICT" | awk '{print $1}')
  DECISION=$(printf '%s' "$VERDICT" | awk '{print $2}')
  printf '\n'
  printf '┌──────────────────────────────────────────────────────────┐\n'
  printf '│  n-gram depth-1 oracle accept rate: %-6s              │\n' "$RATE"
  if [[ "$DECISION" == "GO" ]]; then
    printf '│  >> spec worthwhile if >55%% ── ORACLE SAYS: GO ✓       │\n'
  else
    printf '│  >> spec worthwhile if >55%% ── ORACLE SAYS: NO-GO      │\n'
  fi
  printf '└──────────────────────────────────────────────────────────┘\n'
fi

printf '\njson written to: %s\n' "$OUT_JSON"
exit "$EXIT_CODE"
