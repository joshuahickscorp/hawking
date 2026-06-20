#!/usr/bin/env bash
# 3.1 — re-run the n-gram/PLD speculation oracle on REAL product transcripts.
#
# Why: Oracle A gave τ=1.43 (NO-GO) on a *repo-source* corpus, but a real
# code-completion workload (high copy-rate: the model re-emits chunks of the
# prompt/context) may clear the GO threshold τ≥2.5. This decides GO/NO-GO
# BEFORE investing in any speculator wiring. The lossless PLD/n-gram runtime
# itself ALREADY EXISTS (opt-in HAWKING_LOOKAHEAD=N, see memory
# lookahead_resurrected) — this only re-measures the acceptance ceiling on the
# right distribution.
#
# Pipeline: llama-tokenize -f <transcript> -> id dump -> oracle_spec_accept.py.
# τ = mean accepted tokens per verify forward (the speedup ceiling, since the
# draft is a ~free CPU automaton). Verdict GO≥2.5 / MARGINAL≥1.6 / NO-GO.
#
# Usage:
#   tools/bench/spec_oracle_on_transcripts.sh <transcript.txt> [more.txt ...]
#   MODEL=models/qwen2.5-3b-instruct-q4_k_m.gguf tools/bench/spec_oracle_on_transcripts.sh t1.txt
#
# Transcripts = real served sessions (prompt+completion concatenated), one
# workload per file. Get them from serve logs / saved completions.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
TOKENIZE_BIN="${TOKENIZE_BIN:-llama-tokenize}"
ORACLE="tools/bench/oracle_spec_accept.py"
PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"
OUTDIR="${OUTDIR:-reports/oracle}"

[[ $# -ge 1 ]] || { echo "usage: $0 <transcript.txt> [more.txt ...]"; exit 64; }
command -v "$TOKENIZE_BIN" >/dev/null 2>&1 || {
  echo "error: $TOKENIZE_BIN not found. Build llama.cpp (llama-tokenize) or set TOKENIZE_BIN." >&2
  echo "  fallback: any 'id -> piece' dump works; oracle reads the leading integer per line." >&2
  exit 1; }
[[ -f "$MODEL" ]] || { echo "error: tokenizer model not found: $MODEL" >&2; exit 1; }
mkdir -p "$OUTDIR"

for transcript in "$@"; do
  [[ -f "$transcript" ]] || { echo "skip (missing): $transcript"; continue; }
  base="$(basename "${transcript%.*}")"
  toks="/tmp/spec_oracle_${base}.toks"
  "$TOKENIZE_BIN" -m "$MODEL" -f "$transcript" > "$toks" 2>/dev/null \
    || { echo "tokenize failed: $transcript"; continue; }
  echo "=== $transcript ($(wc -l < "$toks") tokens) ==="
  "$PY" "$ORACLE" "$toks" --out "$OUTDIR/${base}.json"
done

echo
echo "GO (τ≥2.5) on real transcripts ⇒ wire the existing HAWKING_LOOKAHEAD runtime"
echo "into the default serve path; NO-GO ⇒ spec stays the trained-EAGLE-head path only."
