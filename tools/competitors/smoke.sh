#!/usr/bin/env bash
# Single-prompt smoke bench across dismantle + llama.cpp.
# Streams each backend's output as it runs, captures the tok/s line,
# prints a summary table at the end.
#
# Usage:
#   ./tools/competitors/smoke.sh                           # default prompt
#   ./tools/competitors/smoke.sh "Your prompt here"        # custom prompt
#   ./tools/competitors/smoke.sh "prompt" 64               # 64 tokens
#
#   SKIP_DISMANTLE=1 ./tools/competitors/smoke.sh          # only competitors
#   SKIP_LLAMACPP=1  ./tools/competitors/smoke.sh          # only dismantle
#
# Compatible with macOS stock bash 3.2 and BSD awk/sed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="$REPO_ROOT/models/deepseek-v2-lite-q4.gguf"
DISMANTLE_BIN="$REPO_ROOT/target/release/dismantle"
PROMPT="${1:-Once upon a time}"
MAX_TOKENS="${2:-256}"
TEMP=0.0

# Per-backend captured stats. Plain variables (parallel by name) so
# this works on macOS stock bash 3.2.
DEC_DISMANTLE=""; PRE_DISMANTLE=""; WALL_DISMANTLE=""; STAT_DISMANTLE=""
DEC_LLAMACPP=""; PRE_LLAMACPP=""; WALL_LLAMACPP=""; STAT_LLAMACPP=""

bar() { printf '%.0s─' $(seq 1 60); echo; }
section() { echo; bar; echo "▶ $1"; bar; }

# Run a command with a hard timeout. Tries gtimeout (brew coreutils),
# then BSD-flavored timeout, then a perl-based wrapper as a portable
# fallback. Exit 124 means the timeout fired.
#
# IMPORTANT: the perl path uses `perl -e '...'` (NOT heredoc). A
# heredoc redirects perl's stdin to the heredoc text, which steals
# any pipe the caller is feeding to the wrapped command.
runwait() {
    local secs="$1"; shift
    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout --kill-after=5 "$secs" "$@"
    elif command -v timeout >/dev/null 2>&1; then
        timeout --kill-after=5 "$secs" "$@" 2>/dev/null \
            || timeout "$secs" "$@"
    else
        perl -e '
            my $t = shift @ARGV;
            my $pid = fork; defined $pid or die "fork: $!";
            if ($pid == 0) { exec { $ARGV[0] } @ARGV; die "exec: $!"; }
            local $SIG{ALRM} = sub {
                kill "TERM", $pid; sleep 2; kill "KILL", $pid; exit 124;
            };
            alarm $t; waitpid $pid, 0; alarm 0;
            exit($? >> 8);
        ' "$secs" "$@"
    fi
}

# ---- 1. dismantle (Phase 0 CPU) -------------------------------------

if [[ "${SKIP_DISMANTLE:-0}" == "1" ]]; then
    STAT_DISMANTLE=skip
    echo "[smoke] skipping dismantle (SKIP_DISMANTLE=1)"
elif [[ -x "$DISMANTLE_BIN" && -f "$MODEL" ]]; then
    section "dismantle (phase 0, CPU reference path)"
    log="$(mktemp -t dismantle.XXXXXX.log)"
    start=$(date +%s)
    set +e
    "$DISMANTLE_BIN" generate \
        --weights "$MODEL" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_TOKENS" \
        --temperature "$TEMP" \
        --max-stall-ms 120000 \
        2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    end=$(date +%s)
    WALL_DISMANTLE=$((end - start))
    STAT_DISMANTLE=$rc
    DEC_DISMANTLE=$(sed -nE 's/.*dec_tps=([0-9.]+).*/\1/p' "$log" | head -1)
    PRE_DISMANTLE=$(sed -nE 's/.*prefill_ms=([0-9.]+).*/\1/p' "$log" | head -1)
else
    STAT_DISMANTLE=skip
    echo "[smoke] skipping dismantle (binary or model missing)"
fi

# ---- 2. llama.cpp ---------------------------------------------------

if [[ "${SKIP_LLAMACPP:-0}" == "1" ]]; then
    STAT_LLAMACPP=skip
    echo "[smoke] skipping llama.cpp (SKIP_LLAMACPP=1)"
elif command -v llama-cli >/dev/null 2>&1 && [[ -f "$MODEL" ]]; then
    section "llama.cpp (Metal backend)"
    log="$(mktemp -t llamacpp.XXXXXX.log)"
    start=$(date +%s)
    # Llama-cli b8870+ defaults to conversation mode and re-prompts
    # on EOF instead of exiting. Three layers of belt-and-suspenders:
    #   1. -no-cnv (or --single-turn on newer builds) → one-shot
    #   2. feed `/exit` through stdin so any chat-mode prompt obeys it
    #   3. hard 120-sec timeout so the worst case is bounded; once we
    #      see the stats footer we have everything we need anyway.
    llama_timeout="${LLAMA_TIMEOUT:-120}"
    set +e
    printf '%s\n/exit\n' "$PROMPT" \
        | runwait "$llama_timeout" llama-cli \
            --model "$MODEL" \
            --prompt "$PROMPT" \
            --predict "$MAX_TOKENS" \
            --temp "$TEMP" \
            -no-cnv \
            --no-display-prompt --no-warmup \
        2>&1 | tee "$log"
    rc=${PIPESTATUS[1]}
    set -e
    end=$(date +%s)
    WALL_LLAMACPP=$((end - start))
    STAT_LLAMACPP=$rc
    # Try new-style first.
    DEC_LLAMACPP=$(sed -nE 's/.*Generation:[[:space:]]+([0-9.]+)[[:space:]]+t.*/\1/p' "$log" | tail -1)
    PRE_LLAMACPP=$(sed -nE 's/.*Prompt:[[:space:]]+([0-9.]+)[[:space:]]+t.*/\1/p' "$log" | tail -1)
    # Fall back to old-style "eval time = ... (... tokens per second)".
    if [[ -z "$DEC_LLAMACPP" ]]; then
        DEC_LLAMACPP=$(grep -E '^[[:space:]]*eval time' "$log" \
            | sed -nE 's/.*\(([0-9.]+)[[:space:]]+tokens per second\).*/\1/p' \
            | tail -1)
    fi
    if [[ -z "$PRE_LLAMACPP" ]]; then
        PRE_LLAMACPP=$(grep 'prompt eval time' "$log" \
            | sed -nE 's/.*\(([0-9.]+)[[:space:]]+tokens per second\).*/\1/p' \
            | tail -1)
    fi
else
    STAT_LLAMACPP=skip
    echo "[smoke] skipping llama.cpp (llama-cli not on PATH)"
fi

# ---- Summary --------------------------------------------------------

canonicalize() { local v="$1"; if [[ "$v" == "0" ]]; then echo "ok"; elif [[ -z "$v" ]]; then echo "—"; else echo "$v"; fi; }

section "SUMMARY"
printf '%-14s  %-14s  %-14s  %-12s  %-8s\n' "BACKEND" "DECODE tok/s" "PREFILL tok/s" "WALL secs" "STATUS"
printf '%-14s  %-14s  %-14s  %-12s  %-8s\n' "──────" "────────────" "─────────────" "─────────" "──────"
printf '%-14s  %-14s  %-14s  %-12s  %-8s\n' "dismantle" "$(canonicalize "$DEC_DISMANTLE")" "$(canonicalize "$PRE_DISMANTLE")" "$(canonicalize "$WALL_DISMANTLE")" "$(canonicalize "$STAT_DISMANTLE")"
printf '%-14s  %-14s  %-14s  %-12s  %-8s\n' "llamacpp"  "$(canonicalize "$DEC_LLAMACPP")"  "$(canonicalize "$PRE_LLAMACPP")"  "$(canonicalize "$WALL_LLAMACPP")"  "$(canonicalize "$STAT_LLAMACPP")"

echo
echo "[smoke] prompt: $(printf '%q' "$PROMPT")"
echo "[smoke] max_tokens: $MAX_TOKENS, temp: $TEMP"
echo "[smoke] hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown), $(sysctl -n hw.memsize 2>/dev/null || echo 0) bytes RAM"
echo "[smoke] note: dismantle's PREFILL field is in milliseconds (raw)."
echo "[smoke]       Convert to tok/s: prompt_tokens / (prefill_ms / 1000)"
echo "[smoke] note: dismantle is in phase 0 (CPU only). The wedges in"
echo "[smoke]       ROADMAP.md are what move it to compete with llama.cpp."
echo "[smoke] paste this summary into docs/competitive_audit.md."
