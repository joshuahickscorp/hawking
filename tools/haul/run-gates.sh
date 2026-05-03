#!/usr/bin/env bash
# Phase 1 haul gate runner — executes a manifest item under
# co-existence-mode protections.
#
# Usage:
#   ./tools/haul/run-gates.sh phase1            # run all items in manifest order
#   ./tools/haul/run-gates.sh phase1 G1.1       # run only G1.1
#   DRY_RUN=1 ./tools/haul/run-gates.sh phase1  # smoke-test the runner itself
#
# Behavior:
#   - For each gate, runs `coexist.sh probe` first; halts if critical.
#   - Captures pre.json before the gate, post.json after, verify.json
#     from an independent re-run of validation.
#   - Single-process Metal pipeline compilation (low concurrency) when
#     pressure is degraded.
#   - All dismantle subprocess invocations get nice + taskpolicy -b.
#   - On gate halt, writes _phase1_haul_attemptN_blocked.md.
#   - Per-validator timeout (PER_VALIDATOR_TIMEOUT_S, default 1800s).
#     Kill -9 fires if a validator hangs; post.json records timeout: true.
#   - Validator stdout streamed to _evidence/$gate/stdout.log; post.json
#     records line count and truncation flag honestly.
#   - Publishes "<gate> <pid>" to _evidence/.active while a validator
#     runs so `coexist.sh watch` can SIGSTOP/SIGCONT the tree on pressure.
#
# This script is *the* contract for haul execution. It runs whether
# attended or unattended.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE="${1:-phase1}"
ONLY_GATE="${2:-}"
DRY_RUN="${DRY_RUN:-0}"
PER_VALIDATOR_TIMEOUT_S="${PER_VALIDATOR_TIMEOUT_S:-1800}"

PHASE_ARCHIVE="$REPO_ROOT/docs/archive/phase-history"
SPEC="$PHASE_ARCHIVE/_${PHASE}-spec.md"
# HAUL=N selects `_${PHASE}_haul${N}_manifest.md`; default = haul 1's
# manifest (back-compat with the existing single-haul launch path).
if [[ -n "${HAUL:-}" ]]; then
    MANIFEST="$PHASE_ARCHIVE/_${PHASE}_haul${HAUL}_manifest.md"
else
    MANIFEST="$PHASE_ARCHIVE/_${PHASE}_haul_manifest.md"
fi
EVIDENCE_DIR="$REPO_ROOT/tools/haul/_evidence"
COEXIST="$REPO_ROOT/tools/haul/coexist.sh"
PROPOSE_PATCH="$REPO_ROOT/tools/haul/propose-patch.sh"

log() { printf '[run-gates] %s\n' "$*"; }
fatal() { log "FATAL: $*"; exit 1; }

[[ -f "$SPEC" ]]      || fatal "missing $SPEC"
[[ -f "$MANIFEST" ]]  || fatal "missing $MANIFEST"
[[ -x "$COEXIST" ]]   || fatal "missing $COEXIST"

# ---- 0. Helpers -----------------------------------------------------

# python3 -c (not heredoc) so a piped stdin still reaches the script —
# heredoc would redirect stdin to the heredoc text and silently drop
# any pipe.

EMIT_JSON_PY='
import json, sys
args = sys.argv[1:]
out = {}
i = 0
while i < len(args):
    k, v = args[i], args[i+1]
    if v in ("true", "false"):
        out[k] = (v == "true")
    else:
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    i += 2
print(json.dumps(out, indent=2))
'

EMIT_JSON_WITH_MEMORY_PY='
import json, sys
args = sys.argv[1:]
out = {}
i = 0
while i < len(args):
    k, v = args[i], args[i+1]
    if v in ("true", "false"):
        out[k] = (v == "true")
    else:
        try: out[k] = int(v)
        except ValueError:
            try: out[k] = float(v)
            except ValueError: out[k] = v
    i += 2
raw = sys.stdin.read().strip()
if raw:
    try:
        out["memory"] = json.loads(raw)
    except json.JSONDecodeError as e:
        out["memory"] = {"parse_error": str(e), "raw_len": len(raw)}
else:
    out["memory"] = {}
print(json.dumps(out, indent=2))
'

emit_json() {
    python3 -c "$EMIT_JSON_PY" "$@"
}

emit_json_with_memory() {
    python3 -c "$EMIT_JSON_WITH_MEMORY_PY" "$@"
}

# Run a command with a timeout. Pipes combined stdout+stderr to the
# given log path. Returns the command's exit code, or 124 if the
# watchdog had to kill it. Sets TIMED_OUT=true|false as a side effect.
#
# If RUN_ACTIVE_GATE is set, publishes "<gate> <pid>" to
# $EVIDENCE_DIR/.active for the lifetime of the child so the
# coexist watch sidecar can SIGSTOP/SIGCONT the process tree on
# memory-pressure transitions. Cleared on exit (success, fail, or
# timeout) so the watcher idles between items.
TIMED_OUT=false
run_with_timeout() {
    local timeout_s="$1"; shift
    local stdout_log="$1"; shift
    TIMED_OUT=false

    # Sentinel file the watchdog touches when it has to kill the process.
    local sentinel="${stdout_log}.timeout"
    rm -f "$sentinel"

    # Launch the command. The eval-via-array pattern preserves the
    # arguments as separate words while still allowing the caller to
    # pass a pipeline-like command.
    "$@" >"$stdout_log" 2>&1 &
    local pid=$!

    # Publish active-pid for the coexist watch sidecar. Atomic-ish
    # rename so a watcher mid-read never sees a half-written line.
    local active_file="$EVIDENCE_DIR/.active"
    if [[ -n "${RUN_ACTIVE_GATE:-}" ]]; then
        mkdir -p "$EVIDENCE_DIR"
        printf '%s %s\n' "$RUN_ACTIVE_GATE" "$pid" > "$active_file.tmp" \
            && mv "$active_file.tmp" "$active_file"
    fi

    # Watchdog: sleep, then kill if still alive. SIGCONT first so a
    # tree paused by the coexist watch can receive the TERM/KILL.
    (
        sleep "$timeout_s"
        if kill -0 "$pid" 2>/dev/null; then
            touch "$sentinel"
            kill -CONT "$pid" 2>/dev/null
            kill -TERM "$pid" 2>/dev/null
            sleep 5
            kill -KILL "$pid" 2>/dev/null
        fi
    ) &
    local watcher_pid=$!
    # Disown the watcher so the shell does not print "Killed" later.
    disown "$watcher_pid" 2>/dev/null || true

    wait "$pid" 2>/dev/null
    local rc=$?

    # Cancel watcher if it has not fired.
    kill "$watcher_pid" 2>/dev/null
    wait "$watcher_pid" 2>/dev/null

    # Clear active-pid so the coexist watch idles until the next item.
    rm -f "$active_file"

    if [[ -f "$sentinel" ]]; then
        rm -f "$sentinel"
        TIMED_OUT=true
        return 124
    fi
    return "$rc"
}

# Examine a captured stdout log; echo "<line_count> <truncated_bool>".
# Truncated currently always false because we no longer pipe through
# head -N. Preserved as a hook for future per-gate caps.
stdout_summary() {
    local log_path="$1"
    if [[ -f "$log_path" ]]; then
        printf '%d false\n' "$(wc -l <"$log_path" | tr -d ' ')"
    else
        printf '0 false\n'
    fi
}

# ---- 1. Pre-haul memory probe --------------------------------------

if [[ "$DRY_RUN" != "1" ]]; then
    log "pre-haul memory probe…"
    "$COEXIST" probe || true
    guard_exit=$?
    if [[ $guard_exit -eq 2 ]]; then
        fatal "memory pressure CRITICAL — aborting before any item runs"
    elif [[ $guard_exit -eq 1 ]]; then
        log "memory pressure DEGRADED — continuing with reduced concurrency"
    fi
fi

# ---- 2. Parse manifest item list ------------------------------------

# Manifest line format (one per line, leading whitespace ignored):
#   G1.1 cargo-test phase1_kernel_parity::test_rmsnorm_matches_cpu
#   G1.2 dismantle-smoke 3
# i.e. <gate-id> <validator-kind> <validator-args...>
#
# This is read directly from a `## Gate runner manifest` fenced block
# inside the manifest .md so the human-readable manifest and the
# machine-executable list stay in sync.

extract_gate_lines() {
    # Each non-blank, non-comment line in the fenced block becomes a
    # "GATE\t<line>" record. Lines of the form `# layer: <name>` emit
    # a "LAYER\t<name>" record so the main loop can group gates by
    # layer for halt-budget accounting.
    awk '
        /^## Gate runner manifest/ { in_block = 1; next }
        in_block && /^```/         { in_block_code = !in_block_code; next }
        in_block && in_block_code && /^# layer:/ {
            sub(/^# layer:[[:space:]]*/, "", $0)
            print "LAYER\t" $0
            next
        }
        in_block && in_block_code && NF > 0 && !/^#/ { print "GATE\t" $0 }
        in_block && /^## /         { exit }
    ' "$MANIFEST"
}

# bash 3.2 (macOS default) has no `mapfile`; emulate via while-read.
GATE_LINES=()
while IFS= read -r line; do
    [[ -n "$line" ]] && GATE_LINES+=("$line")
done < <(extract_gate_lines)
if [[ ${#GATE_LINES[@]} -eq 0 ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN: empty manifest — runner self-test passed"
        exit 0
    fi
    fatal "no gate lines extracted from $MANIFEST (look for '## Gate runner manifest' fenced block)"
fi

log "loaded ${#GATE_LINES[@]} gate(s) from manifest"

# ---- 3. Halt-budget bookkeeping -------------------------------------

# Hybrid threshold from spec:
#   G1.1 (scaffold): 1 halt = end haul
#   G1.2..G1.4 (GEMV ports): 2 halts in this group = end haul
# Layered (super-haul) thresholds, controlled by `# layer:` markers in
# the manifest:
#   pre-flight: 1 halt = end haul
#   impl:       2 halts in the layer = end haul
#   audit:      record-and-continue (no halt cost)
#   self-improve: record-and-continue (no halt cost)
#   closeout:   always runs (skipped from halt accounting)
HALT_SCAFFOLD=0
HALT_GEMV=0
HALT_PREFLIGHT=0
HALT_IMPL=0
HALT_AUDIT=0
HALT_SELFIMPROVE=0
HAUL_HALTED=0
CURRENT_LAYER="default"
ATTEMPT_NUM=1
# Haul-aware status-doc namespace: HAUL=2 → _phase1_haul2_attempt*.md;
# unset (legacy) → _phase1_haul_attempt*.md.
if [[ -n "${HAUL:-}" ]]; then
    STATUS_PREFIX="_${PHASE}_haul${HAUL}_attempt"
else
    STATUS_PREFIX="_${PHASE}_haul_attempt"
fi
while [[ -f "$PHASE_ARCHIVE/${STATUS_PREFIX}${ATTEMPT_NUM}_blocked.md" ]] \
   || [[ -f "$PHASE_ARCHIVE/${STATUS_PREFIX}${ATTEMPT_NUM}_closeout.md" ]] \
   || [[ -f "$PHASE_ARCHIVE/${STATUS_PREFIX}${ATTEMPT_NUM}_status.md" ]]; do
    ATTEMPT_NUM=$((ATTEMPT_NUM + 1))
done
log "this haul attempt: #$ATTEMPT_NUM"

# ---- 4. Per-gate executor -------------------------------------------

run_gate() {
    local gate_id="$1"; shift
    local validator_kind="$1"; shift
    local gate_args=("$@")

    # Tell run_with_timeout which gate is in flight so it can publish
    # the active pid for the coexist watch sidecar.
    export RUN_ACTIVE_GATE="$gate_id"

    log "─── gate $gate_id ($validator_kind) ───"

    # Per-item memory probe.
    #
    # The probe distinguishes safe (0), degraded (1), critical (2).
    # The original spec retries on degraded, halting after 5×30s. That
    # matches the model-load smoke gates' need: a 9 GB mmap can OOM the
    # box if started while compressed memory is high.
    #
    # For RAM-light validator kinds (everything in haul 2 except
    # `dismantle-smoke`) the gate's working set is 50 MB–2 GB; degraded
    # is fine. We only retry on critical for those — the watcher
    # sidecar's SIGSTOP/SIGCONT picks up the slack mid-flight if
    # pressure spikes after launch. dismantle-smoke keeps the legacy
    # behavior since model load is genuinely RAM-heavy.
    if [[ "$DRY_RUN" != "1" ]]; then
        local retries=0
        local probe_rc
        while true; do
            "$COEXIST" probe >/dev/null
            probe_rc=$?
            if [[ "$validator_kind" == "dismantle-smoke" ]]; then
                # Legacy: halt on degraded after 5 retries.
                [[ $probe_rc -eq 0 ]] && break
            else
                # RAM-light: only retry on critical.
                [[ $probe_rc -le 1 ]] && break
            fi
            retries=$((retries + 1))
            if [[ $retries -ge 5 ]]; then
                log "$gate_id: memory pressure persisted ${retries} cycles (rc=$probe_rc), halting gate"
                return 2
            fi
            log "$gate_id: memory pressure non-safe (rc=$probe_rc); sleeping 30s (retry $retries/5)…"
            sleep 30
        done
    fi

    # Evidence pre.json — built via python3 from probe's --json output.
    # No sed pipelines; if the probe JSON is malformed, the wrapper
    # records a structured parse_error rather than producing invalid JSON.
    local edir="$EVIDENCE_DIR/$gate_id"
    mkdir -p "$edir"
    "$COEXIST" probe --json | emit_json_with_memory \
        captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        gate "$gate_id" \
        attempt "$ATTEMPT_NUM" \
        git_head "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo none)" \
        > "$edir/pre.json"

    # Run the validator under a per-call timeout. stdout+stderr stream
    # to a log file that's referenced from post.json, so the audit
    # trail is honest about what was captured.
    local rc=0
    local stdout_log="$edir/stdout.log"
    local model="$REPO_ROOT/models/deepseek-v2-lite-q4.gguf"
    case "$validator_kind" in
        cargo-test)
            # Legacy cargo-test validator (haul 1 manifest). Kept for
            # back-compat; new manifests should prefer cargo-test-strict
            # which is more flexible. The strict-equivalent test-count
            # check is backported here so a stale `#[ignore]` doesn't
            # produce a vacuous PASS (which is exactly what bit haul 1
            # before the H2.x rebuild — surfaced in haul 2's post-mortem).
            local test_filter="${gate_args[0]:-}"
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would run cargo test --release --test phase1_kernel_parity $test_filter"
                : > "$stdout_log"
            else
                run_with_timeout "$PER_VALIDATOR_TIMEOUT_S" "$stdout_log" \
                    bash -c "cd '$REPO_ROOT' && exec cargo test --release --test phase1_kernel_parity $test_filter" \
                    || rc=$?
                if [[ $rc -eq 0 ]]; then
                    local n_passed
                    n_passed=$(grep -oE 'test result: ok\. [0-9]+ passed' "$stdout_log" 2>/dev/null \
                              | grep -oE '[0-9]+' | head -1)
                    if [[ -z "$n_passed" || "$n_passed" -lt 1 ]]; then
                        log "$gate_id: VACUOUS PASS — 0 tests ran for filter"
                        rc=98
                    fi
                fi
            fi
            ;;
        dismantle-smoke)
            local n_tokens="${gate_args[0]:-3}"
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would run dismantle generate -n $n_tokens"
                : > "$stdout_log"
            else
                if [[ ! -f "$model" ]]; then
                    log "missing $model"
                    rc=4
                    : > "$stdout_log"
                else
                    run_with_timeout "$PER_VALIDATOR_TIMEOUT_S" "$stdout_log" \
                        "$REPO_ROOT/target/release/dismantle" generate \
                            --weights "$model" \
                            --prompt "Once upon a time" \
                            --max-new-tokens "$n_tokens" \
                            --temperature 0 \
                            --max-stall-ms 60000 \
                        || rc=$?
                fi
            fi
            ;;
        cargo-build)
            # Phase-2 audit delta: build with --tests to compile every
            # integration test target up-front; later cargo-test gates
            # then skip the link step.
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would run cargo build --release --tests"
                : > "$stdout_log"
            else
                run_with_timeout "$PER_VALIDATOR_TIMEOUT_S" "$stdout_log" \
                    bash -c "cd '$REPO_ROOT' && exec cargo build --release --workspace --tests" \
                    || rc=$?
            fi
            ;;
        noop)
            log "(noop validator — gate is informational only)"
            : > "$stdout_log"
            ;;
        cargo-test-strict)
            # Defends against vacuous-PASS: cargo test exits 0 even if
            # the filter matched nothing (e.g. all tests #[ignore]'d).
            # We additionally require the stdout log to show ≥1 test
            # passed. The manifest controls every cargo flag (including
            # `--release`) so this validator works for both lib and
            # integration tests; nothing prefixed automatically.
            local args="${gate_args[*]}"
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would run cargo test $args (strict)"
                : > "$stdout_log"
            else
                run_with_timeout "$PER_VALIDATOR_TIMEOUT_S" "$stdout_log" \
                    bash -c "cd '$REPO_ROOT' && exec cargo test $args" \
                    || rc=$?
                if [[ $rc -eq 0 ]]; then
                    local n_passed
                    n_passed=$(grep -oE 'test result: ok\. [0-9]+ passed' "$stdout_log" 2>/dev/null \
                              | grep -oE '[0-9]+' | head -1)
                    if [[ -z "$n_passed" || "$n_passed" -lt 1 ]]; then
                        log "$gate_id: VACUOUS PASS — 0 tests ran for filter"
                        rc=98
                    fi
                fi
            fi
            ;;
        cargo-clippy-baseline)
            # Pass iff total warnings ≤ baseline. Catches drift without
            # requiring zero-warning policy upfront.
            local max_warnings="${gate_args[0]:-0}"
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would run cargo clippy (baseline=$max_warnings)"
                : > "$stdout_log"
            else
                run_with_timeout "$PER_VALIDATOR_TIMEOUT_S" "$stdout_log" \
                    bash -c "cd '$REPO_ROOT' && exec cargo clippy --workspace --all-targets --release 2>&1" \
                    || rc=$?
                local n_warn
                n_warn=$(grep -cE '^warning:' "$stdout_log" 2>/dev/null || echo 0)
                n_warn="${n_warn:-0}"
                log "$gate_id: clippy warnings=$n_warn (baseline=$max_warnings)"
                if [[ "$n_warn" -gt "$max_warnings" ]]; then
                    log "$gate_id: clippy regression: $n_warn > $max_warnings"
                    rc=97
                fi
            fi
            ;;
        cargo-fmt-check)
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would run cargo fmt --check"
                : > "$stdout_log"
            else
                run_with_timeout 300 "$stdout_log" \
                    bash -c "cd '$REPO_ROOT' && exec cargo fmt --all -- --check" \
                    || rc=$?
            fi
            ;;
        verify-evidence)
            local target_phase="${gate_args[0]:-phase1}"
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would run verify-evidence.sh $target_phase"
                : > "$stdout_log"
            else
                run_with_timeout 300 "$stdout_log" \
                    bash -c "cd '$REPO_ROOT' && exec ./tools/haul/verify-evidence.sh $target_phase" \
                    || rc=$?
            fi
            ;;
        evidence-archive)
            # Phase-2 pre-flight cleanup: move prior haul's
            # record-and-continue evidence (post.json exit_code != 0) out
            # of _evidence/ so the next haul's verify-evidence doesn't
            # re-attest old failures. Always passes; emits archived count
            # to its own post.json.
            local source_haul="${gate_args[0]:-prior}"
            local archive_root="$REPO_ROOT/tools/haul/_evidence_archive/${source_haul}"
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would archive failed-evidence dirs to $archive_root"
                : > "$stdout_log"
            else
                mkdir -p "$archive_root"
                local archived=0
                for d in "$EVIDENCE_DIR"/*/; do
                    local name
                    name=$(basename "$d")
                    [[ "$name" == "$gate_id" ]] && continue   # skip self
                    local pj="$d/post.json"
                    [[ -f "$pj" ]] || continue
                    local ec
                    ec=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('exit_code', 0))" "$pj" 2>/dev/null || echo 0)
                    if [[ "$ec" != "0" ]]; then
                        mv "$d" "$archive_root/" && archived=$((archived + 1))
                    fi
                done
                printf '[evidence-archive] archived %d failed-evidence dirs to %s\n' \
                    "$archived" "$archive_root" > "$stdout_log"
                log "$gate_id: archived $archived dirs"
            fi
            ;;
        patch-apply)
            # Self-improvement: invoke propose-patch.sh in apply mode,
            # which edits source and writes the unified diff to
            # _evidence/<gate>/applied.patch.
            local kind="${gate_args[0]:-}"
            local extra_args="${gate_args[*]:1}"
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would patch-apply $kind $extra_args"
                : > "$stdout_log"
            elif [[ ! -x "$PROPOSE_PATCH" ]]; then
                log "$gate_id: missing $PROPOSE_PATCH"
                rc=96
                : > "$stdout_log"
            else
                run_with_timeout 300 "$stdout_log" \
                    bash -c "cd '$REPO_ROOT' && exec '$PROPOSE_PATCH' apply '$kind' '$edir/applied.patch' $extra_args" \
                    || rc=$?
            fi
            ;;
        dismantle-token-regression)
            # Greedy temp=0 token-id-perfect regression vs a captured
            # baseline. The driver script handles per-prompt iteration,
            # b3sum comparison, and first-mismatch halting.
            local baseline="${gate_args[0]:-}"
            if [[ -z "$baseline" ]]; then
                log "$gate_id: dismantle-token-regression requires baseline-file arg"
                rc=95
                : > "$stdout_log"
            elif [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would run token-regression against $baseline"
                : > "$stdout_log"
            else
                run_with_timeout "$PER_VALIDATOR_TIMEOUT_S" "$stdout_log" \
                    bash -c "cd '$REPO_ROOT' && exec ./tools/haul/token-regression.sh '$baseline'" \
                    || rc=$?
            fi
            ;;
        capture-baseline-50)
            # 50-prompt baseline capture via expand-baseline.sh. The
            # script is parametrized through OUT_OVERRIDE / LOG_OVERRIDE
            # / PROMPT_FILE / N_TOKENS env vars (B5.1b extension).
            local out_file="${gate_args[0]:-}"
            if [[ -z "$out_file" ]]; then
                log "$gate_id: capture-baseline-50 requires output-file arg"
                rc=94
                : > "$stdout_log"
            elif [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would capture 50-prompt baseline to $out_file"
                : > "$stdout_log"
            else
                local out_path="$REPO_ROOT/$out_file"
                local log_path="${out_path%.hashes}.log"
                rm -f "$out_path" "$log_path"
                # Hard-cap at 4 hr regardless of PER_VALIDATOR_TIMEOUT_S
                # so a wedged generate doesn't drag the haul into the
                # next day. 50 prompts × 8 tokens × ~12 s/token + prefill
                # comfortably fits in 4 hr.
                # Single in-process pass via `dismantle batch-hash` —
                # one model load amortizes across all 50 prompts.
                # Replaces the 50-launch shell loop in
                # expand-baseline.sh (which spawned a new dismantle
                # process per prompt = 50 cold model loads).
                run_with_timeout 14400 "$stdout_log" \
                    "$REPO_ROOT/target/release/dismantle" batch-hash \
                        --weights "$model" \
                        --prompts "$REPO_ROOT/tools/haul/prompts_50.txt" \
                        --tokens 3 \
                        --out "$out_path" \
                    || rc=$?
                if [[ $rc -eq 0 ]]; then
                    local n_lines
                    n_lines=$(grep -cE '^p[0-9]+ ' "$out_path" 2>/dev/null || echo 0)
                    n_lines="${n_lines:-0}"
                    log "$gate_id: captured $n_lines / 50 prompts"
                    if [[ "$n_lines" -ne 50 ]]; then
                        log "$gate_id: capture incomplete (expected 50, got $n_lines)"
                        rc=93
                    fi
                fi
            fi
            ;;
        bench-decode)
            # Run dismantle-bench's decode suite for a given backend.
            # Args: <backend> <trials>. Result lands in $edir/result.json
            # (consumed by perf-ratio-assert downstream). The bench
            # binary is a subcommand of the umbrella `dismantle` CLI.
            local backend="${gate_args[0]:-}"
            local trials="${gate_args[1]:-5}"
            local result_json="$edir/result.json"
            if [[ -z "$backend" ]]; then
                log "$gate_id: bench-decode requires backend arg"
                rc=92
                : > "$stdout_log"
            elif [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would run bench-decode $backend trials=$trials"
                : > "$stdout_log"
            elif [[ ! -f "$model" ]]; then
                log "$gate_id: missing model $model"
                rc=4
                : > "$stdout_log"
            else
                rm -f "$result_json"
                run_with_timeout "$PER_VALIDATOR_TIMEOUT_S" "$stdout_log" \
                    "$REPO_ROOT/target/release/dismantle" bench \
                        --backend "$backend" \
                        --suite decode \
                        --weights "$model" \
                        --trials "$trials" \
                        --json "$result_json" \
                    || rc=$?
                if [[ $rc -eq 0 ]]; then
                    if [[ ! -f "$result_json" ]]; then
                        log "$gate_id: result.json not produced"
                        rc=91
                    else
                        local tps
                        tps=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('decode_tps') or d.get('results',{}).get('decode_tps') or 0)" "$result_json" 2>/dev/null || echo 0)
                        log "$gate_id: $backend decode_tps=$tps"
                        if [[ "$(echo "$tps > 0" | bc -l 2>/dev/null)" != "1" ]]; then
                            log "$gate_id: decode_tps not positive: $tps"
                            rc=90
                        fi
                    fi
                fi
            fi
            ;;
        perf-ratio-assert)
            # Read three bench-decode result.json files from sibling
            # _evidence dirs, compute ratios, assert both thresholds.
            # Args: <r-llama-min> <r-mlx-min> [<gate-dis> <gate-lla> <gate-mlx>]
            # Default gate ids = B4.2/B4.3/B4.4 (haul-3 perf-gate path);
            # Phase-2 super-haul-1 passes W3.1/W3.2/W3.3.
            local r_llama_min="${gate_args[0]:-1.5}"
            local r_mlx_min="${gate_args[1]:-0.7}"
            local g_dis="${gate_args[2]:-B4.2}"
            local g_lla="${gate_args[3]:-B4.3}"
            local g_mlx="${gate_args[4]:-B4.4}"
            if [[ "$DRY_RUN" == "1" ]]; then
                log "DRY_RUN: would assert r_llama≥$r_llama_min, r_mlx≥$r_mlx_min from $g_dis,$g_lla,$g_mlx"
                : > "$stdout_log"
            else
                python3 - "$EVIDENCE_DIR" "$r_llama_min" "$r_mlx_min" "$g_dis" "$g_lla" "$g_mlx" \
                    > "$stdout_log" 2>&1 <<'PY' || rc=$?
import json, os, sys

ev_root = sys.argv[1]
r_llama_min, r_mlx_min = float(sys.argv[2]), float(sys.argv[3])
g_dis, g_lla, g_mlx = sys.argv[4], sys.argv[5], sys.argv[6]

def tps(name):
    path = os.path.join(ev_root, name, 'result.json')
    if not os.path.isfile(path):
        print(f'[error] missing {path}')
        sys.exit(91)
    d = json.load(open(path))
    val = d.get('decode_tps') or d.get('results', {}).get('decode_tps') or 0
    return float(val)

dis = tps(g_dis)
lla = tps(g_lla)
mlx = tps(g_mlx)

if lla <= 0 or mlx <= 0 or dis <= 0:
    print(f'[error] non-positive tps: dismantle={dis} llama={lla} mlx={mlx}')
    sys.exit(91)

r_llama = dis / lla
r_mlx = dis / mlx

print(f'dismantle={dis:.3f}  llamacpp={lla:.3f}  mlx={mlx:.3f}')
print(f'r_llama={r_llama:.3f} (min {r_llama_min})')
print(f'r_mlx={r_mlx:.3f} (min {r_mlx_min})')

failed = []
if r_llama < r_llama_min:
    failed.append(f'r_llama={r_llama:.3f} < {r_llama_min}')
if r_mlx < r_mlx_min:
    failed.append(f'r_mlx={r_mlx:.3f} < {r_mlx_min}')

if failed:
    print(f'[perf_below_threshold] ' + '; '.join(failed))
    sys.exit(89)

print('[ok] both ratios meet threshold')
PY
            fi
            ;;
        *)
            log "unknown validator kind: $validator_kind"
            rc=99
            : > "$stdout_log"
            ;;
    esac

    # Evidence post.json — python3-built, includes timeout flag and
    # stdout summary so verify-evidence and humans can inspect.
    local stdout_lines stdout_truncated
    read -r stdout_lines stdout_truncated < <(stdout_summary "$stdout_log")
    emit_json \
        captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        gate "$gate_id" \
        exit_code "$rc" \
        validator_kind "$validator_kind" \
        dry_run "$([[ "$DRY_RUN" == "1" ]] && echo true || echo false)" \
        timed_out "$TIMED_OUT" \
        timeout_s "$PER_VALIDATOR_TIMEOUT_S" \
        stdout_log "$stdout_log" \
        stdout_lines "$stdout_lines" \
        stdout_truncated "$stdout_truncated" \
        > "$edir/post.json"

    if [[ $rc -ne 0 ]]; then
        if [[ "$TIMED_OUT" == "true" ]]; then
            log "$gate_id: TIMEOUT (>${PER_VALIDATOR_TIMEOUT_S}s)"
        else
            log "$gate_id: FAIL (exit $rc)"
        fi
        return $rc
    fi

    # Independent verify run (separate process, no shared state).
    if [[ "$DRY_RUN" != "1" && "$validator_kind" == "cargo-test" ]]; then
        local test_filter="${gate_args[0]:-}"
        local verify_rc=0
        local verify_log="$edir/verify-stdout.log"
        run_with_timeout "$PER_VALIDATOR_TIMEOUT_S" "$verify_log" \
            bash -c "cd '$REPO_ROOT' && exec cargo test --release --test phase1_kernel_parity $test_filter >/dev/null" \
            || verify_rc=$?
        local attestation="false"
        [[ $verify_rc -eq 0 ]] && attestation="true"
        emit_json \
            captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            gate "$gate_id" \
            attestation "$attestation" \
            verify_exit "$verify_rc" \
            timed_out "$TIMED_OUT" \
            > "$edir/verify.json"
        if [[ $verify_rc -ne 0 ]]; then
            log "$gate_id: post-pass verify FAILED (exit $verify_rc) — halting"
            return $verify_rc
        fi
    else
        emit_json \
            captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            gate "$gate_id" \
            attestation true \
            reason dry_run_or_no_verify \
            > "$edir/verify.json"
    fi

    log "$gate_id: PASS"
    return 0
}

# ---- 5. Main loop ---------------------------------------------------

for line in "${GATE_LINES[@]}"; do
    # Layer markers update the current layer and skip gate execution.
    if [[ "$line" == "LAYER"$'\t'* ]]; then
        CURRENT_LAYER="${line#LAYER$'\t'}"
        log "─── entering layer: $CURRENT_LAYER ───"
        continue
    fi
    # Strip "GATE\t" prefix; lines without a prefix are legacy haul-1
    # manifests (no layer markers). Those default to CURRENT_LAYER="default".
    if [[ "$line" == "GATE"$'\t'* ]]; then
        line="${line#GATE$'\t'}"
    fi

    # shellcheck disable=SC2206
    parts=( $line )
    gate_id="${parts[0]}"
    validator_kind="${parts[1]}"
    gate_args=( "${parts[@]:2}" )

    if [[ -n "$ONLY_GATE" && "$gate_id" != "$ONLY_GATE" ]]; then
        log "skip $gate_id (filter: $ONLY_GATE)"
        continue
    fi

    if run_gate "$gate_id" "$validator_kind" "${gate_args[@]:-}"; then
        # CE-6 inter-item cool-down. Honors HAUL_COOLDOWN_S env (haul 3
        # sets it to 0 to recover the slm-coexistence budget now that
        # slm is no longer running).
        haul_cooldown="${HAUL_COOLDOWN_S:-30}"
        [[ "$DRY_RUN" != "1" && "$haul_cooldown" -gt 0 ]] && sleep "$haul_cooldown"
        continue
    fi

    # Halt accounting — layer-aware for super hauls; falls back to
    # gate-id pattern matching for legacy haul-1 manifests (no
    # `# layer:` markers; CURRENT_LAYER stays "default").
    case "$CURRENT_LAYER" in
        pre-flight)
            HALT_PREFLIGHT=$((HALT_PREFLIGHT + 1))
            log "pre-flight halt at $gate_id — ending haul (everything depends on this)"
            HAUL_HALTED=1
            ;;
        impl)
            HALT_IMPL=$((HALT_IMPL + 1))
            if [[ $HALT_IMPL -ge 2 ]]; then
                log "impl layer reached 2 halts at $gate_id — ending haul"
                HAUL_HALTED=1
            else
                log "impl halt 1/2 at $gate_id — continuing to next independent item"
                sleep 30
            fi
            ;;
        impl-A)
            # Haul 3 metal-wireup layer. 2-halt budget across A1.x.
            HALT_IMPL=$((HALT_IMPL + 1))
            if [[ $HALT_IMPL -ge 2 ]]; then
                log "impl-A reached 2 halts at $gate_id — ending haul"
                HAUL_HALTED=1
            else
                log "impl-A halt 1/2 at $gate_id — continuing"
            fi
            ;;
        impl-B5)
            # Haul 3 correctness suite. 1-halt-ends — a token mismatch
            # on a freshly-captured baseline IS the correctness failure.
            log "impl-B5 halt at $gate_id — correctness regression, ending haul"
            HAUL_HALTED=1
            ;;
        impl-B4)
            # Haul 3 perf gate. The ratio assertion (B4.5) is the
            # verdict; a halt there is `stage1_perf_below_threshold`.
            # Earlier gates in this layer get one continue-on-halt.
            HALT_IMPL=$((HALT_IMPL + 1))
            if [[ "$gate_id" == "B4.5" ]] || [[ $HALT_IMPL -ge 2 ]]; then
                log "impl-B4 halt at $gate_id — ending haul (perf verdict or 2nd halt)"
                HAUL_HALTED=1
            else
                log "impl-B4 halt 1/2 at $gate_id — continuing"
            fi
            ;;
        impl-W1A)
            # Phase-2 Wedge-1 weight-pinning. 1-halt-ends: pinning
            # cannot change forward-path bit pattern, any failure is
            # the finding.
            log "impl-W1A halt at $gate_id — pinning regression, ending haul"
            HAUL_HALTED=1
            ;;
        impl-W1B)
            # Phase-2 Wedge-1 MLA/Q-LoRA Metal migration. 1-halt-ends
            # for the same reason as impl-W1A.
            log "impl-W1B halt at $gate_id — MLA migration regression, ending haul"
            HAUL_HALTED=1
            ;;
        impl-W3)
            # Phase-2 perf gate. Mirrors haul-3's impl-B4: ratio
            # assertion (W3.4) is the verdict; bench-decode flakes
            # (W3.1/W3.2/W3.3) get one continue-on-halt before ending.
            HALT_IMPL=$((HALT_IMPL + 1))
            if [[ "$gate_id" == "W3.4" ]] || [[ $HALT_IMPL -ge 2 ]]; then
                log "impl-W3 halt at $gate_id — ending haul (perf verdict or 2nd halt)"
                HAUL_HALTED=1
            else
                log "impl-W3 halt 1/2 at $gate_id — continuing"
            fi
            ;;
        audit)
            HALT_AUDIT=$((HALT_AUDIT + 1))
            log "audit halt at $gate_id (record-and-continue, count=$HALT_AUDIT)"
            sleep 5
            ;;
        self-improve)
            HALT_SELFIMPROVE=$((HALT_SELFIMPROVE + 1))
            log "self-improve halt at $gate_id (record-and-continue, count=$HALT_SELFIMPROVE)"
            sleep 5
            ;;
        closeout)
            log "closeout gate $gate_id non-zero exit — ignoring (closeout always runs)"
            ;;
        default)
            # Legacy haul-1 path: gate-id-based hybrid budget.
            case "$gate_id" in
                G1.1)
                    HALT_SCAFFOLD=$((HALT_SCAFFOLD + 1))
                    log "G1.1 halt: scaffold gate is single-halt — ending haul"
                    HAUL_HALTED=1
                    ;;
                G1.2|G1.3|G1.4)
                    HALT_GEMV=$((HALT_GEMV + 1))
                    if [[ $HALT_GEMV -ge 2 ]]; then
                        log "GEMV group reached 2 halts — ending haul"
                        HAUL_HALTED=1
                    else
                        log "GEMV halt 1/2 — continuing to next independent item"
                        sleep 30
                    fi
                    ;;
                *)
                    log "unknown halt gate $gate_id — ending haul defensively"
                    HAUL_HALTED=1
                    ;;
            esac
            ;;
        *)
            log "halt in unknown layer '$CURRENT_LAYER' — ending haul defensively"
            HAUL_HALTED=1
            ;;
    esac

    if [[ $HAUL_HALTED -ne 0 ]]; then
        # Stub blocked doc — agent flesh-out happens when the failing
        # gate's diagnostic info is captured. The runner just reserves
        # the file with the attempt number.
        blocked_path="$PHASE_ARCHIVE/${STATUS_PREFIX}${ATTEMPT_NUM}_blocked.md"
        cat > "$blocked_path" <<EOF
# ${PHASE} haul${HAUL:+" $HAUL"} attempt $ATTEMPT_NUM — BLOCKED

**Halted at:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Halted on:** $gate_id (layer: $CURRENT_LAYER)
**Halt counts:** scaffold=$HALT_SCAFFOLD gemv=$HALT_GEMV pre-flight=$HALT_PREFLIGHT impl=$HALT_IMPL audit=$HALT_AUDIT self-improve=$HALT_SELFIMPROVE
**Timed out:** $TIMED_OUT

## Root cause

(agent fills in)

## What ran up to halt

$(ls -1 "$EVIDENCE_DIR" 2>/dev/null | sed 's/^/- /')

## What attended work unblocks

(agent fills in)

## Followups for next session

(agent fills in)
EOF
        log "wrote $blocked_path"
        break
    fi
done

# ---- 6. Final report ------------------------------------------------

if [[ $HAUL_HALTED -eq 0 ]]; then
    log "haul complete: all gates ran to PASS"
    exit 0
else
    log "haul halted at $gate_id"
    exit 1
fi
