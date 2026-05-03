#!/usr/bin/env bash
# tools/haul/token-regression.sh — token-id-perfect regression test
# against a `_phase1_token_baseline_*.hashes` file.
#
# As of haul 3, regenerates fresh hashes via the in-process
# `dismantle batch-hash` subcommand (one model load amortized across
# all prompts) and diffs per-id against the captured baseline. First
# mismatch ends the run.
#
# Usage: token-regression.sh <baseline-file>
# Optional env:
#   DISMANTLE_KERNEL_PROFILE=/path/profile.json
#   DISMANTLE_SPECULATE=exact-shared
#   DISMANTLE_VERIFY_WINDOW=4
# Exits:
#   0  — every entry matched
#   91 — at least one mismatch (first_fail recorded on stdout)
#   92 — vacuous (zero entries hashed)
#   93 — generate / batch-hash process failed
#   94 — bad CLI usage / missing file

set -uo pipefail

BASELINE="${1:-}"
if [[ -z "$BASELINE" ]]; then
    echo "[error] usage: $0 <baseline-file>" >&2
    exit 94
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DISMANTLE="$REPO_ROOT/target/release/dismantle"
MODEL="$REPO_ROOT/models/deepseek-v2-lite-q4.gguf"

if [[ ! -f "$BASELINE" && -f "$REPO_ROOT/$BASELINE" ]]; then
    BASELINE="$REPO_ROOT/$BASELINE"
fi

[[ -f "$BASELINE" ]] || { echo "[error] baseline not found: $BASELINE" >&2; exit 94; }
[[ -x "$DISMANTLE" ]] || { echo "[error] missing $DISMANTLE — run cargo build --release" >&2; exit 93; }
[[ -f "$MODEL" ]] || { echo "[error] missing $MODEL" >&2; exit 93; }

# Build a temp `<id>:<prompt>` prompts file from the baseline. The
# baseline format is `<id> <N> <hash> <prompt>`; preserve `<id>` and
# `<prompt>` only.
tmp_prompts="$(mktemp -t token-regression-prompts.XXXXXX)"
tmp_fresh="$(mktemp -t token-regression-fresh.XXXXXX)"
trap 'rm -f "$tmp_prompts" "$tmp_fresh"' EXIT

n_tokens_first=""
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    id="${line%% *}"
    rest="${line#* }"
    n="${rest%% *}"
    rest="${rest#* }"
    rest="${rest#* }"           # drop hash
    prompt="$rest"
    printf '%s:%s\n' "$id" "$prompt" >> "$tmp_prompts"
    if [[ -z "$n_tokens_first" ]]; then n_tokens_first="$n"; fi
done < "$BASELINE"

n_prompts=$(grep -c '^p[0-9]' "$tmp_prompts" 2>/dev/null || echo 0)
if [[ "$n_prompts" -lt 1 ]]; then
    echo "[error] no prompts parsed from baseline" >&2
    exit 92
fi

echo "[regression] $n_prompts prompts, $n_tokens_first tokens — running batch-hash..."
extra_args=()
if [[ -n "${DISMANTLE_KERNEL_PROFILE:-}" ]]; then
    extra_args+=(--kernel-profile "$DISMANTLE_KERNEL_PROFILE")
fi
if [[ -n "${DISMANTLE_SPECULATE:-}" ]]; then
    extra_args+=(--speculate "$DISMANTLE_SPECULATE")
fi
if [[ -n "${DISMANTLE_VERIFY_WINDOW:-}" ]]; then
    extra_args+=(--verify-window "$DISMANTLE_VERIFY_WINDOW")
fi
"$DISMANTLE" batch-hash \
    --weights "$MODEL" \
    --prompts "$tmp_prompts" \
    --tokens "$n_tokens_first" \
    --out "$tmp_fresh" \
    "${extra_args[@]}" || { echo "[error] batch-hash failed" >&2; exit 93; }

# Compare hashes per id, walking the baseline in order. Plain
# grep-based lookups; bash 3.2 (macOS default) has no `declare -A`,
# so we don't attempt associative arrays.
total=0; ok=0; fail=0; first_fail=""
while IFS= read -r b_line; do
    [[ -z "$b_line" || "$b_line" == \#* ]] && continue
    id="${b_line%% *}"
    rest="${b_line#* }"
    rest="${rest#* }"           # drop n
    expected="${rest%% *}"
    total=$((total + 1))

    f_line=$(grep -m1 "^${id} " "$tmp_fresh" 2>/dev/null || true)
    if [[ -z "$f_line" ]]; then
        echo "$id: MISSING from fresh capture"
        fail=$((fail + 1))
        first_fail="${first_fail:-$id}"
        break
    fi
    rest="${f_line#* }"
    rest="${rest#* }"           # drop n
    actual="${rest%% *}"

    if [[ "$actual" == "$expected" ]]; then
        ok=$((ok + 1))
        echo "$id: OK ($actual)"
    else
        fail=$((fail + 1))
        first_fail="${first_fail:-$id}"
        echo "$id: MISMATCH expected=$expected actual=$actual"
        break
    fi
done < "$BASELINE"

echo "[summary] total=$total ok=$ok fail=$fail first_fail=${first_fail:-none}"

if [[ $fail -gt 0 ]]; then exit 91; fi
if [[ $ok -lt 1 ]]; then exit 92; fi
exit 0
