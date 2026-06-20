#!/usr/bin/env bash
# =============================================================================
# small_draft_accept_export.sh — GPU-LANE export + measured-tau for the
#   small DENSE-draft speculative-decoding oracle (axis-3 spec).
# =============================================================================
#
# WHAT THIS DECIDES
#   reports/oracle_small_draft_design.md §4(A): the *measured* mean-accepted-
#   length tau of a small dense draft (0.5B / 1.5B Q4_K_M) verifying the 3B
#   Q4_K_M target on real code. tau is the only number the proxy cannot
#   produce (it needs two model forwards), so this script is the decisive
#   accept-rate gate. Quality risk is ZERO (lossless spec decode emits the
#   target's exact distribution); the only question is whether tau clears the
#   breakeven for the draft's own forward cost.
#
# GO / NO-GO THRESHOLD  (design §3.3, at point-estimate cost-ratio c, k=4, v=0)
#   0.5B draft (c=0.25):  tau > 2.00  to beat plain AR   (GO).
#   1.5B draft (c=0.58):  tau > 3.32  to beat plain AR   (GO).
#   The oracle (draft_accept_oracle.py) reports the full per-k table + the
#   ngram-bonus-first comparison and the best-k verdict; these two numbers are
#   the headline gate. A measured tau BELOW the k=2 plain-AR floor (1.50 for
#   0.5B, 2.16 for 1.5B) is a Type-1 kill — record it in reports/dead_levers.md;
#   do NOT kill on the byte-ratio proxy.
#   The cost-ratio c here is the byte-ratio ESTIMATE; the decisive c comes from
#   the §4(B) paired forward-cost bench (NOT this script) and is passed via
#   ORACLE_C=<measured> to collapse the [lo,hi] bracket.
#
# THE EXPORT MECHANISM  (read this — it is the crux)
#   `dismantle generate` has NO logit-export path (confirmed: crates/hawking/
#   src/main.rs Generate has no --dump-logits / --save-logits flag, and no .npy
#   writer exists anywhere in crates/). The only full-vocab logit exporter on
#   this machine is llama.cpp's `llama-perplexity --save-all-logits FNAME`,
#   which writes the next-token logits for EVERY position of an input file in
#   one teacher-forced pass — exactly the (T, V) stream the oracle's --logits
#   contract wants (design §2). All three models are Qwen2.5-family GGUFs with
#   the SAME tokenizer/vocab, so target and draft logits align position-for-
#   position with no vocab remap (design §1 / §4 pre-flight).
#
#   Pipeline per model:
#     1. llama-perplexity -m <gguf> -f <corpus.txt> --save-all-logits <bin>
#        (greedy/teacher-forced over the corpus tokens; -c CTX window).
#     2. tools/bench/llama_logits_to_npy.py <bin> <out.npy>  -> (T, V) float32.
#   Then:
#     3. draft_accept_oracle.py --logits T.npy D.npy --draft 0.5B [--c ..]
#        and again --draft 1.5B.
#
#   NB on alignment: --save-all-logits emits logits for the positions llama.cpp
#   scores under its -c stride. Target and BOTH drafts are run with the SAME
#   -c / corpus, so the three logit dumps cover the SAME positions and the
#   oracle's shape-equality assert (T,V)==(T,V) holds. The converter trims all
#   three to the common min-T as a safety net (see --no-trim to disable).
#
#   The design's purest form teacher-forces the drafts on the TARGET's own
#   greedy continuation. llama-perplexity teacher-forces on the corpus tokens
#   themselves, not a target-generated continuation; for a high-copy CODE
#   corpus these are close, and this is the cheapest faithful export available
#   without a bespoke dismantle logit dumper. If a dismantle-side exporter is
#   ever added (followup below), prefer it and feed the target's continuation.
#
# MODELS  (all on disk — confirmed `ls models/`, design §1)
#   target : models/qwen2.5-3b-instruct-q4_k_m.gguf     (1,929,903,264 B)
#   0.5B   : models/qwen2.5-0.5b-instruct-q4_k_m.gguf   (  491,400,032 B)
#   1.5B   : models/qwen2.5-1.5b-instruct-q4_k_m.gguf   (1,117,320,736 B)
#
# CORPUS  (design §4(A): "reuse the corpus behind reports/oracle/spec_accept.json,
#          ~40k code tokens")
#   That corpus is the git-history coding-session corpus from
#   tools/bench/make_git_session_corpus.py (the JSONL sessions concatenated to
#   text), tokenized with llama-tokenize (spec_oracle_on_transcripts.sh). It is
#   NOT checked in. This script regenerates it from THIS repo's git history when
#   absent (CORPUS unset), or uses a caller-supplied text file (CORPUS=path).
#   reports/a4_code_prompt.txt is a single 210-byte prompt, NOT the 40k corpus —
#   do not pass it; it would give a meaningless 1-window tau.
#
# RUN-WHERE
#   GPU lane (this is a real two-model forward job). Paired-delta contamination
#   rules do NOT apply — these are absolute logit dumps; their ARGMAX agreement
#   is load-independent, so it is safe to run with Claude open, but heavy. The
#   oracle math at the end is pure CPU numpy.
#
# USAGE
#   tools/bench/small_draft_accept_export.sh                 # both drafts, auto corpus
#   DRAFTS="0.5B" tools/bench/small_draft_accept_export.sh   # just the favorable bet
#   CORPUS=/path/code.txt tools/bench/small_draft_accept_export.sh
#   CTX=2048 CHUNKS=20 ORACLE_C=0.27 tools/bench/small_draft_accept_export.sh
#   KEEP_BINS=1 ...                                          # keep the big .bin dumps
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

# ---- config (override via env) ---------------------------------------------
TARGET="${TARGET:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
DRAFT_05="${DRAFT_05:-models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
DRAFT_15="${DRAFT_15:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
DRAFTS="${DRAFTS:-0.5B 1.5B}"          # which drafts to score
PERP_BIN="${PERP_BIN:-llama-perplexity}"
TOKENIZE_BIN="${TOKENIZE_BIN:-llama-tokenize}"
ORACLE="${ORACLE:-tools/bench/draft_accept_oracle.py}"
CONVERTER="${CONVERTER:-tools/bench/llama_logits_to_npy.py}"
CORPUS="${CORPUS:-}"                   # caller text corpus; empty -> auto git-session
CTX="${CTX:-2048}"                     # llama-perplexity -c window
CHUNKS="${CHUNKS:--1}"                 # -1 = all chunks (whole corpus)
ORACLE_C="${ORACLE_C:-}"              # measured cost-ratio (design §4(B)); empty -> byte-ratio default
KMAX="${KMAX:-}"                       # optional --k override (space-separated)
WORKDIR="${WORKDIR:-reports/oracle/small_draft_export}"
KEEP_BINS="${KEEP_BINS:-0}"            # 1 -> keep the multi-GB --save-all-logits bins
PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"

die() { echo "error: $*" >&2; exit 64; }
hr()  { printf '%s\n' "--------------------------------------------------------------------------------"; }

# ---- hard pre-flight (fail LOUDLY on any missing asset) --------------------
command -v "$PERP_BIN"     >/dev/null 2>&1 || die "$PERP_BIN not found (llama.cpp). brew install llama.cpp or set PERP_BIN. This is the ONLY full-vocab logit exporter; dismantle has none."
command -v "$TOKENIZE_BIN" >/dev/null 2>&1 || die "$TOKENIZE_BIN not found (needed only for the auto git-session corpus token count; set TOKENIZE_BIN or pass CORPUS=)."
[[ -f "$TARGET" ]]    || die "target GGUF missing: $TARGET"
[[ -f "$ORACLE" ]]    || die "oracle missing: $ORACLE"
"$PY" -c 'import numpy' 2>/dev/null || die "numpy missing in $PY — pip install numpy (the converter + oracle need it)."

for d in $DRAFTS; do
  case "$d" in
    0.5B) [[ -f "$DRAFT_05" ]] || die "0.5B draft GGUF missing: $DRAFT_05" ;;
    1.5B) [[ -f "$DRAFT_15" ]] || die "1.5B draft GGUF missing: $DRAFT_15" ;;
    *)    die "unknown draft '$d' (allowed: 0.5B 1.5B — match draft_accept_oracle.py RATIO_DEFAULTS)" ;;
  esac
done

mkdir -p "$WORKDIR"

# ---- corpus: caller-supplied, else regenerate from git history -------------
if [[ -n "$CORPUS" ]]; then
  [[ -f "$CORPUS" ]] || die "CORPUS file not found: $CORPUS"
  CORPUS_TXT="$CORPUS"
  echo "  corpus: caller-supplied $CORPUS_TXT"
else
  CORPUS_TXT="$WORKDIR/git_session_corpus.txt"
  if [[ ! -s "$CORPUS_TXT" ]]; then
    echo "  corpus: regenerating the git-session code corpus (design §4(A): the"
    echo "          ~40k-token corpus behind reports/oracle/spec_accept.json)."
    GEN="tools/bench/make_git_session_corpus.py"
    [[ -f "$GEN" ]] || die "auto-corpus generator missing: $GEN (pass CORPUS=<text file> instead)."
    SESS_DIR="$WORKDIR/git_sessions"
    rm -rf "$SESS_DIR"; mkdir -p "$SESS_DIR"
    "$PY" "$GEN" --repo . --out-dir "$SESS_DIR" --sessions 5 --turns 10 \
      || die "git-session corpus generation failed (need a git repo with history)."
    # Concatenate every session's text payload into one corpus stream. The
    # generator writes JSONL; pull the text field(s) out into plain text.
    : > "$CORPUS_TXT"
    shopt -s nullglob
    for f in "$SESS_DIR"/*.jsonl; do
      "$PY" - "$f" >> "$CORPUS_TXT" <<'PYJL'
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    # Emit any string-valued field that looks like code/text (prompt, target,
    # content, text, completion). Concatenate so the stream is one long corpus.
    for k in ("prompt", "context", "content", "text", "target", "completion", "body"):
        v = obj.get(k)
        if isinstance(v, str) and v:
            print(v)
PYJL
    done
    shopt -u nullglob
    [[ -s "$CORPUS_TXT" ]] || die "regenerated corpus is empty ($CORPUS_TXT) — inspect $SESS_DIR/*.jsonl field names."
  fi
  echo "  corpus: $CORPUS_TXT"
fi

# Report corpus token count (sanity vs the ~40k design target).
if command -v "$TOKENIZE_BIN" >/dev/null 2>&1; then
  CORPUS_TOKS="$("$TOKENIZE_BIN" -m "$TARGET" -f "$CORPUS_TXT" 2>/dev/null | wc -l | tr -d ' ')"
  echo "  corpus tokens (llama-tokenize, target vocab): ${CORPUS_TOKS:-?}"
  if [[ -n "${CORPUS_TOKS:-}" && "$CORPUS_TOKS" -lt 4000 ]]; then
    echo "  WARN: corpus is < 4k tokens — tau will be noisy. Design wants ~40k."  >&2
    echo "        (reports/a4_code_prompt.txt is a single prompt; do not use it.)" >&2
  fi
fi

# ---- export helper: model GGUF -> (T,V) .npy via save-all-logits -----------
# Writes $WORKDIR/<tag>.bin (raw llama logits) then converts to <tag>.npy.
export_logits() {  # $1=gguf  $2=tag
  local gguf="$1" tag="$2"
  local bin="$WORKDIR/${tag}.bin" npy="$WORKDIR/${tag}.npy"
  echo "  [$tag] llama-perplexity --save-all-logits over the corpus (teacher-forced)..."
  echo "        $PERP_BIN -m $gguf -f $CORPUS_TXT -c $CTX --chunks $CHUNKS --save-all-logits $bin"
  # nice/taskpolicy: co-existence (CLAUDE.md memory-coexist rule) — yield to any
  # foreground GPU job. -ngl 999 keeps all layers on the GPU (Metal) lane.
  nice -n 19 taskpolicy -b "$PERP_BIN" \
      -m "$gguf" -f "$CORPUS_TXT" -c "$CTX" --chunks "$CHUNKS" \
      -ngl 999 --save-all-logits "$bin" \
      > "$WORKDIR/${tag}.perplexity.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 || ! -s "$bin" ]]; then
    echo "  [$tag] FAILED (rc=$rc). Tail of $WORKDIR/${tag}.perplexity.log:" >&2
    tail -25 "$WORKDIR/${tag}.perplexity.log" >&2
    die "logit export failed for $tag — see log above (likely OOM at -c $CTX; lower CTX or CHUNKS)."
  fi
  echo "  [$tag] converting $bin -> $npy ((T,V) float32)..."
  [[ -f "$CONVERTER" ]] || die "converter missing: $CONVERTER (companion to this script — author it; see header)."
  "$PY" "$CONVERTER" "$bin" "$npy" || die "bin->npy conversion failed for $tag."
  [[ -s "$npy" ]] || die "converter produced no $npy."
  if [[ "$KEEP_BINS" != 1 ]]; then rm -f "$bin"; fi
  echo "$npy"
}

hr
echo "  EXPORT — target + draft next-token logits over the SAME code stream"
hr
TARGET_NPY="$(export_logits "$TARGET" target)" || exit 1

# ---- per-draft: export + oracle measured-tau -------------------------------
run_oracle() {  # $1=draft tag (0.5B|1.5B)  $2=draft gguf
  local dtag="$1" dgguf="$2"
  local safe="${dtag/./p}"            # 0.5B -> 0p5B for filenames
  local DRAFT_NPY
  DRAFT_NPY="$(export_logits "$dgguf" "draft_${safe}")" || return 1

  hr
  echo "  ORACLE — measured tau, draft=$dtag (design §4(A)/(C))"
  hr
  local out="reports/oracle/small_draft_accept_${safe}.json"
  local cargs=()
  [[ -n "$ORACLE_C" ]] && cargs+=(--c "$ORACLE_C")
  [[ -n "$KMAX"     ]] && cargs+=(--k $KMAX)
  echo "  running: $PY $ORACLE --logits $TARGET_NPY $DRAFT_NPY --draft $dtag ${cargs[*]} --out $out"
  echo "  (c = ${ORACLE_C:-byte-ratio default for $dtag}; GO: 0.5B tau>2.00, 1.5B tau>3.32 at k=4)"
  echo ""
  "$PY" "$ORACLE" --logits "$TARGET_NPY" "$DRAFT_NPY" --draft "$dtag" "${cargs[@]}" --out "$out"
  local rc=$?
  echo ""
  if [[ $rc -ne 0 ]]; then
    echo "  >> draft=$dtag: oracle exited non-zero ($rc) — read its stderr above." >&2
  else
    echo "  >> draft=$dtag: measured-tau verdict written to $out."
    echo "     GO if S_ar>1 at some k (oracle reports best-k). 0.5B needs tau>2.00,"
    echo "     1.5B needs tau>3.32 (k=4) to beat plain AR. NO-GO only on this MEASURED"
    echo "     tau -> then a Type-1 kill in reports/dead_levers.md (design §4(C))."
  fi
}

for d in $DRAFTS; do
  case "$d" in
    0.5B) run_oracle 0.5B "$DRAFT_05" ;;
    1.5B) run_oracle 1.5B "$DRAFT_15" ;;
  esac
done

hr
echo "  DONE. Reports: reports/oracle/small_draft_accept_*.json"
echo "  Next (design §4(B)): collapse the cost-ratio bracket with a paired"
echo "  draft-vs-target forward-cost bench, then re-run with ORACLE_C=<measured>."
hr
