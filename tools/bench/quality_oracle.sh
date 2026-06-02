#!/usr/bin/env bash
# =============================================================================
# quality_oracle.sh — token-identity quality gate for NON-bit-identical levers.
# =============================================================================
#
# WHY THIS EXISTS (closes the Phase-1.2 gap):
#   The bit-identical levers gate on PARITY in paired_lever.sh (greedy temp=0,
#   diff -q). But f16-rounding levers (f16-scales, f16-KV, flash-attn) change
#   the output *slightly* — PARITY just says "DIFF" with no magnitude. The old
#   Phase-1.2 method (reports/bench/p11_default.txt vs p12_f16s.txt) compared
#   only the b3sum HASH column: all-or-nothing per prompt (one rounding flip
#   flips the whole hash), 5 prompts, one length. It answers WHETHER a prompt
#   diverged, never HOW MUCH.
#
#   This oracle measures HOW MUCH, for any DISMANTLE_QWEN_* flag, at short AND
#   long ctx, over N=24 diverse prompts:
#     - token_identical_fraction  (# prompts with identical OFF/ON text / N)
#     - mean first-divergence position (frac of OFF text before the first diff)
#     - corpus drift%  (tokens downstream of the first divergence / total)
#     - per-category breakdown (code/math/prose/... — drift is category-corr.)
#
# METHOD — PAIRED BY CONSTRUCTION (Claude-open OK):
#   Runs `dismantle bench-server` TWICE against the same prompt set: lever env
#   UNSET (OFF) then =1 (ON), locked base env constant. The metric is the
#   OFF-vs-ON text comparison (a RELATIVE delta), so machine contamination
#   cancels — same discipline as paired_lever.sh / f16s_drift_sweep.py.
#   bench-server is a deterministic greedy decoder (temp=0, seed=42, fixed
#   tokenizer) so decoded-text equality <=> token-ID-sequence equality.
#
#   bench-server (not batch-hash) is used because it emits the full
#   completion_text per request (one model load, KV reset between prompts) —
#   batch-hash emits only a b3sum, which is all-or-nothing and cannot yield a
#   divergence POSITION. (batch-hash remains the cheap %-identical-only path:
#   `dismantle batch-hash --prompts P.txt` OFF vs ON, diff the hash columns.)
#
# LOGIT COSINE — the deeper metric, NOT measured here (documented gap):
#   `dismantle` has no logit-export path (see tools/bench/llama_logits_to_npy.py
#   header: "dismantle generate cannot export logits"). The only full-vocab
#   logit dumper on this machine is llama.cpp's --save-all-logits, which runs
#   llama.cpp NOT the dismantle lever, so it cannot measure dismantle's f16
#   logit shift. The plan-1.2 logit-cosine>=~0.999 gate therefore needs a
#   logit-dump flag built into `dismantle generate` first. This oracle prints
#   that gate as REQUIRED-BUT-UNMEASURED rather than faking it.
#
# F16-KV INCOMPATIBILITY (qwen_dense.rs:3539): DISMANTLE_QWEN_F16_KV=1 HARD-
#   ERRORS if combined with W4A8=1 or FLASH_ATTN=1. The locked base below holds
#   neither, and we assert the chosen lever isn't a conflicting pair.
#
# USAGE (the orchestrator/user runs this — never an agent, never GPU here):
#   tools/bench/quality_oracle.sh --lever DISMANTLE_QWEN_PREDEC_F16SCALES --label f16scales
#   tools/bench/quality_oracle.sh --lever DISMANTLE_QWEN_F16_KV          --label f16kv --long
#   tools/bench/quality_oracle.sh --lever DISMANTLE_QWEN_FLASH_ATTN      --label flashattn --long
#
#   --long           also run the long-ctx tier (longctx_prompt.txt prefix);
#                    f16-KV/flash-attn only change KV/attention so long ctx is
#                    their meaningful tier — for those, default to --long.
#   --short-only     skip the long tier (default if neither flag given).
#   --tokens N       decode tokens per prompt (default 48).
#   Gate overrides:  PASS_IDENT_MIN=0.90 PASS_DRIFT_MAX=5.0 (env; per-lever).
#
# Output: human table + JSON at reports/quality/oracle_<label>.json.
# READ-ONLY perf: only measures; safe to re-run; locked shipped decode config.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

# ---- config (override via env) ---------------------------------------------
BIN="${BIN:-./target/release/dismantle}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
TOKENS="${TOKENS:-48}"
SHORT_PROMPTS="${SHORT_PROMPTS:-reports/bench/f16s_drift_prompts.txt}"
LONGCTX_PREFIX="${LONGCTX_PREFIX:-reports/bench/longctx_prompt.txt}"

# Locked Qwen fast-path — constant across OFF/ON (matches paired_lever.sh base,
# minus the lever under test). Contains NO W4A8 and NO FLASH_ATTN so f16-KV is
# legal to toggle on top of it (qwen_dense.rs:3539 guard).
LOCKED_ENV=(
  DISMANTLE_QWEN_TCB=1
  DISMANTLE_QWEN_VOCAB_PRUNE=32000
  DISMANTLE_QWEN_Q4K_LMHEAD=1
  DISMANTLE_QWEN_FFN_DOWN_Q4K=1
  DISMANTLE_QWEN_Q4K_PREDEC=1
)

# Gate thresholds (plan 1.2 + the f16s KEEP-OPT-IN finding). Overridable.
PASS_IDENT_MIN="${PASS_IDENT_MIN:-0.90}"   # PASS if identical-fraction >= this
WARN_IDENT_MIN="${WARN_IDENT_MIN:-0.75}"   # WARN (opt-in only) down to this
PASS_DRIFT_MAX="${PASS_DRIFT_MAX:-5.0}"    # PASS if corpus drift% <= this
FAIL_DRIFT_MAX="${FAIL_DRIFT_MAX:-10.0}"   # FAIL if corpus drift% > this
LOGIT_COSINE_GATE="${LOGIT_COSINE_GATE:-0.999}"  # deeper gate (unmeasured here)

LEVER=""; LABEL=""; DO_LONG=0; LONG_SET=0
die() { echo "error: $*" >&2; exit 64; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lever)      LEVER="$2"; shift 2;;
    --label)      LABEL="$2"; shift 2;;
    --tokens)     TOKENS="$2"; shift 2;;
    --long)       DO_LONG=1; LONG_SET=1; shift;;
    --short-only) DO_LONG=0; LONG_SET=1; shift;;
    -h|--help)    sed -n '2,70p' "$0"; exit 0;;
    *) die "unknown arg: $1";;
  esac
done
[[ -n "$LEVER" ]] || die "--lever DISMANTLE_QWEN_* required (the flag to A/B)"
[[ -n "$LABEL" ]] || LABEL="$(echo "$LEVER" | tr 'A-Z' 'a-z' | sed 's/^dismantle_qwen_//')"
[[ -x "$BIN" ]]   || die "binary not found/executable: $BIN (cargo build --release?)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"
[[ -f "$SHORT_PROMPTS" ]] || die "short prompt corpus not found: $SHORT_PROMPTS"

# f16-KV is the meaningful KV/attention tier at long ctx — default --long for it
# and for flash-attn unless the caller forced --short-only.
if [[ "$LONG_SET" == 0 ]]; then
  case "$LEVER" in
    DISMANTLE_QWEN_F16_KV|DISMANTLE_QWEN_FLASH_ATTN) DO_LONG=1;;
  esac
fi

# Guard: never co-set an incompatible pair (would HARD-ERROR in the ON run).
for kv in "${LOCKED_ENV[@]}"; do :; done
if [[ "$LEVER" == DISMANTLE_QWEN_F16_KV ]]; then
  for kv in "${LOCKED_ENV[@]}"; do
    case "$kv" in
      DISMANTLE_QWEN_W4A8=1|DISMANTLE_QWEN_FLASH_ATTN=1)
        die "f16-KV is incompatible with $kv (qwen_dense.rs:3539); fix LOCKED_ENV";;
    esac
  done
fi
[[ "$DO_LONG" == 1 && ! -f "$LONGCTX_PREFIX" ]] && \
  die "--long needs the long-ctx prefix file: $LONGCTX_PREFIX"

mkdir -p reports/quality
OUT="reports/quality/oracle_${LABEL}.json"
PY="$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)"
hr() { printf '%s\n' "================================================================================"; }

hr
echo "  quality_oracle — lever=$LEVER  label=$LABEL  tokens=$TOKENS  long=$DO_LONG"
echo "  base: ${LOCKED_ENV[*]}"
echo "  gates: identical>=${PASS_IDENT_MIN} (warn>=${WARN_IDENT_MIN}), drift%<=${PASS_DRIFT_MAX} (fail>${FAIL_DRIFT_MAX})"
hr

# Hand everything to one python driver. It spawns bench-server OFF then ON per
# tier (one model load each), feeds JSON-line requests, parses completion_text,
# computes the metrics, prints the tables, writes JSON, and exits 0/1/2 on the
# PASS/WARN/FAIL verdict so the orchestrator can branch.
"$PY" - "$BIN" "$WEIGHTS" "$PROFILE" "$TOKENS" "$LEVER" "$LABEL" "$DO_LONG" \
  "$SHORT_PROMPTS" "$LONGCTX_PREFIX" "$OUT" \
  "$PASS_IDENT_MIN" "$WARN_IDENT_MIN" "$PASS_DRIFT_MAX" "$FAIL_DRIFT_MAX" \
  "$LOGIT_COSINE_GATE" "${LOCKED_ENV[@]}" <<'PYEOF'
import json, os, subprocess, sys

(bin_, weights, profile, tokens, lever, label, do_long,
 short_path, long_prefix_path, out_path,
 pass_ident, warn_ident, pass_drift, fail_drift, logit_gate, *locked) = sys.argv[1:]
tokens = int(tokens); do_long = do_long == "1"
pass_ident = float(pass_ident); warn_ident = float(warn_ident)
pass_drift = float(pass_drift); fail_drift = float(fail_drift)

LOCKED = {}
for kv in locked:
    k, _, v = kv.partition("="); LOCKED[k] = v

# Categories aligned to reports/bench/f16s_drift_prompts.txt (24 lines, in order).
# If the corpus length differs, fall back to a single 'all' bucket.
CATEGORIES = [
    "code", "code", "factual", "code-sql", "prose-edu",
    "prose", "math", "math", "math", "math",
    "dialogue", "dialogue", "lists", "lists", "nonenglish",
    "nonenglish", "nonenglish", "factual", "factual", "factual",
    "prose", "math", "code", "lists",
]

def load_prompts(path):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                # tolerate optional 'pNNN:' prefix (parity_prompts.txt style)
                if len(s) > 5 and s[0] == "p" and s[1:4].isdigit() and s[4] == ":":
                    s = s[5:]
                out.append(s)
    return out

def run_server(prompts, lever_on):
    env = dict(os.environ); env.update(LOCKED)
    if lever_on:
        env[lever] = "1"
    else:
        env.pop(lever, None)            # env_on() is == "1"; UNSET == off
    cmd = ["nice", "-n", "19", "taskpolicy", "-b",
           bin_, "bench-server", "--weights", weights,
           "--kernel-profile", profile, "--stdin"]
    reqs = "".join(
        json.dumps({"id": f"p{i:03d}", "prompt": p, "max_tokens": tokens}) + "\n"
        for i, p in enumerate(prompts))
    proc = subprocess.run(cmd, input=reqs, env=env,
                          capture_output=True, text=True, timeout=3600)
    texts, ntok, errs = {}, {}, {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error"):
            errs[r.get("id", "?")] = r["error"]
        if "id" in r and "completion_text" in r:
            texts[r["id"]] = r["completion_text"]
            ntok[r["id"]] = r.get("completion_tokens", 0)
    if not texts:
        sys.stderr.write("=== bench-server stderr (no responses) ===\n")
        sys.stderr.write(proc.stderr[-2000:] + "\n")
    if errs:
        sys.stderr.write(f"=== bench-server reported errors: {errs} ===\n")
    return texts, ntok, errs

def first_div_char(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1   # -1 == identical

def tier(name, prompts):
    sys.stderr.write(f"[oracle:{name}] {len(prompts)} prompts x {tokens}tok — OFF then ON\n")
    off_t, off_n, off_e = run_server(prompts, lever_on=False)
    on_t,  on_n,  on_e  = run_server(prompts, lever_on=True)
    if on_e:
        # ON run hard-errored (e.g. an incompatible lever combo): no verdict.
        return {"tier": name, "error": on_e, "rows": [], "n_prompts": len(prompts)}
    rows, cat = [], {}
    tot_drift = tot_tok = n_div = 0
    div_fracs = []
    use_cat = len(prompts) == len(CATEGORIES)
    for i, p in enumerate(prompts):
        pid = f"p{i:03d}"
        c = CATEGORIES[i] if use_cat else "all"
        a = off_t.get(pid, "<MISSING>"); b = on_t.get(pid, "<MISSING>")
        nt = off_n.get(pid, tokens) or tokens
        d = first_div_char(a, b)
        ident = (d == -1)
        if ident:
            drift = 0
        else:
            L = max(len(a), 1)
            div_fracs.append(d / L)
            drift = round(max(0.0, (L - d) / L) * nt)
            n_div += 1
        bucket = cat.setdefault(c, [0, 0, 0, 0])  # drift_tok,tot_tok,n,n_div
        bucket[0] += drift; bucket[1] += nt; bucket[2] += 1; bucket[3] += (0 if ident else 1)
        tot_drift += drift; tot_tok += nt
        rows.append({"id": pid, "cat": c, "identical": ident,
                     "first_div_char": d, "off_len": len(a), "on_len": len(b),
                     "n_tok": nt, "drift_tok_est": drift})
    n = len(prompts)
    ident_frac = (n - n_div) / n if n else 0.0
    drift_pct = 100.0 * tot_drift / tot_tok if tot_tok else 0.0
    mean_div_frac = sum(div_fracs) / len(div_fracs) if div_fracs else 1.0
    return {"tier": name, "n_prompts": n, "tokens": tokens,
            "identical_fraction": ident_frac, "n_diverged": n_div,
            "corpus_drift_pct": drift_pct,
            "mean_first_div_frac": mean_div_frac,
            "per_category": {k: {"drift_tok": v[0], "total_tok": v[1],
                                 "n": v[2], "n_diverged": v[3]} for k, v in cat.items()},
            "rows": rows}

def verdict(t):
    if t.get("error"):
        return "ERROR"
    if t["identical_fraction"] >= pass_ident and t["corpus_drift_pct"] <= pass_drift:
        return "PASS"
    if t["corpus_drift_pct"] > fail_drift or t["identical_fraction"] < warn_ident:
        return "FAIL"
    return "WARN"  # opt-in-only band (where f16-scales landed)

short = load_prompts(short_path)
tiers = [tier("short", short)]
if do_long:
    with open(long_prefix_path) as f:
        prefix = f.read()
    long_prompts = [prefix.rstrip() + "\n\n" + p for p in short]
    tiers.append(tier("long", long_prompts))

# ---- print ----
for t in tiers:
    print()
    print(f"== TIER {t['tier']}  ({t.get('n_prompts','?')} prompts x {tokens} tok) ==")
    if t.get("error"):
        print(f"  ON run ERRORED (no verdict): {t['error']}")
        continue
    print(f"  token_identical_fraction = {t['identical_fraction']:.3f}  "
          f"({t['n_prompts']-t['n_diverged']}/{t['n_prompts']} identical)")
    print(f"  corpus_drift%            = {t['corpus_drift_pct']:.2f}%")
    print(f"  mean_first_div_position  = {t['mean_first_div_frac']:.3f}  "
          f"(0=immediate, 1=never; over {t['n_diverged']} diverged)")
    if len(t["per_category"]) > 1:
        print("  per-category (diverged/n, drift%):")
        for c in sorted(t["per_category"]):
            v = t["per_category"][c]
            pct = 100.0 * v["drift_tok"] / v["total_tok"] if v["total_tok"] else 0.0
            print(f"    {c:>11}: {v['n_diverged']}/{v['n']}  {pct:5.2f}%")
    print(f"  >> TIER VERDICT: {verdict(t)}")

# overall verdict = worst tier
order = {"ERROR": 4, "FAIL": 3, "WARN": 2, "PASS": 1}
verdicts = [verdict(t) for t in tiers]
overall = max(verdicts, key=lambda v: order[v])
print()
print("== logit-cosine gate (plan 1.2) ==")
print(f"  REQUIRED >= {logit_gate} but UNMEASURED: `dismantle` has no logit-export")
print("  path (tools/bench/llama_logits_to_npy.py header). Measuring it needs a")
print("  logit-dump flag in `dismantle generate`; llama.cpp --save-all-logits")
print("  runs llama.cpp not the dismantle lever, so it cannot gate this lever.")
print()
print(f"==> OVERALL: {overall}  (lever={lever})")
if overall == "PASS":
    print("    -> token-safe enough to consider default-on (still confirm tps via paired_lever.sh).")
elif overall == "WARN":
    print("    -> KEEP OPT-IN: drift is category-correlated (the f16-scales precedent). Ship as opt-in speed lever only.")
elif overall == "FAIL":
    print("    -> quality-unsafe at these thresholds; do not ship without the logit-cosine check.")
else:
    print("    -> could not evaluate (ON run errored); check the lever/base-env combo.")

json.dump({"lever": lever, "label": label, "tokens": tokens,
           "locked_env": LOCKED, "thresholds": {
               "pass_ident_min": pass_ident, "warn_ident_min": warn_ident,
               "pass_drift_max": pass_drift, "fail_drift_max": fail_drift,
               "logit_cosine_gate": logit_gate, "logit_cosine_measured": False},
           "tiers": tiers, "overall_verdict": overall},
          open(out_path, "w"), indent=2)
print(f"\nwrote {out_path}")
sys.exit({"PASS": 0, "WARN": 1, "FAIL": 2, "ERROR": 3}[overall])
PYEOF
rc=$?
hr
echo "  quality_oracle DONE (exit $rc: 0=PASS 1=WARN/opt-in 2=FAIL 3=ERROR). JSON: $OUT"
hr
exit $rc

