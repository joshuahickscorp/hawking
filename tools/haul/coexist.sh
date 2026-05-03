#!/usr/bin/env bash
# coexist.sh — autonomous-haul memory safety + launch umbrella
#
# Single entry point for everything that lets a dismantle haul run
# alongside slm training without thrashing the box. Three subcommands:
#
#   probe [--json]
#       single-shot pressure check. Exit codes:
#         0 = safe       (≥ FREE_SAFE_GB free, joint footprint ok)
#         1 = degraded   (warn pressure / low free / joint > pct)
#         2 = critical   (critical pressure / vm_stat parse failed)
#       Stdout: human one-liner, or full JSON with --json.
#       Used by run-gates.sh's per-item probe and by capture-baseline.sh.
#
#   watch [phase]
#       long-running sidecar. Polls `probe` every POLL_INTERVAL_S; on
#       degraded → SIGSTOPs the active gate's process tree (read from
#       _evidence/.active, written by run-gates.sh while a validator
#       runs); on safe → SIGCONTs it; on critical → releases the pause
#       so run-gates' next probe halts cleanly. Resumes any paused tree
#       on Ctrl-C / TERM. Logs PAUSE/RESUME timeline to
#       _evidence/<gate>/throttle.log.
#
#   launch [phase]
#       end-to-end one-command entrypoint. Auto-detects SLM_PID, starts
#       `watch` in the background, runs `tools/haul/run-gates.sh` in
#       the foreground, cleans up the watcher on exit (resumes any
#       held pause on the way out).
#
# Tunables (env, all optional):
#   probe:
#     FREE_SAFE_GB=4
#     FREE_DEGRADED_GB=2
#     SLM_PID=<pid>             includes slm RSS in joint-footprint check
#     JOINT_DEGRADE_PCT=80      degrade if slm+dismantle > this % of RAM
#     DISMANTLE_PID=<pid>       use real dismantle RSS instead of estimate
#     DISMANTLE_RSS_EST_GB=2
#   watch / launch:
#     POLL_INTERVAL_S=10
#     COEXIST_LOG_LEVEL=info    info|debug
#
# RAM caveat: SIGSTOP frees CPU/GPU contention, not RSS. Smoke gates
# that mmap the GGUF still need physical headroom; this sidecar lets
# the parity (cargo-test) gates run cleanly alongside slm.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVIDENCE_DIR="$REPO_ROOT/tools/haul/_evidence"
SELF="$0"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ─── probe ────────────────────────────────────────────────────────────

cmd_probe() {
    local emit_json=0
    [[ "${1:-}" == "--json" ]] && emit_json=1

    local FREE_SAFE_GB="${FREE_SAFE_GB:-4}"
    local FREE_DEGRADED_GB="${FREE_DEGRADED_GB:-2}"
    local JOINT_DEGRADE_PCT="${JOINT_DEGRADE_PCT:-80}"
    local DISMANTLE_RSS_EST_GB="${DISMANTLE_RSS_EST_GB:-2}"

    local VM_STAT_OUT
    VM_STAT_OUT=$(vm_stat 2>/dev/null || true)

    local parse_failed=false
    local pressure_query_ok=true

    local PAGE_BYTES
    PAGE_BYTES=$(printf '%s\n' "$VM_STAT_OUT" | head -1 | sed -nE 's/.*page size of ([0-9]+).*/\1/p')
    if [[ -z "$PAGE_BYTES" ]]; then
        PAGE_BYTES=16384
        parse_failed=true
    fi

    vm_stat_field() {
        local field="$1"
        printf '%s\n' "$VM_STAT_OUT" \
            | awk -F'[: ]+' -v f="$field" '$0 ~ f {gsub(/\./,"",$NF); print $NF; exit}'
    }

    local free_pages spec_pages inactive_pages wired_pages compressed_pages
    free_pages=$(vm_stat_field 'Pages free')
    spec_pages=$(vm_stat_field 'Pages speculative')
    inactive_pages=$(vm_stat_field 'Pages inactive')
    wired_pages=$(vm_stat_field 'Pages wired down')
    compressed_pages=$(vm_stat_field 'Pages occupied by compressor')

    if [[ -z "$free_pages" || -z "$wired_pages" ]]; then
        parse_failed=true
    fi

    free_pages="${free_pages:-0}"
    spec_pages="${spec_pages:-0}"
    inactive_pages="${inactive_pages:-0}"
    wired_pages="${wired_pages:-0}"
    compressed_pages="${compressed_pages:-0}"

    local avail_pages=$((free_pages + spec_pages + inactive_pages))
    local avail_bytes=$((avail_pages * PAGE_BYTES))
    local avail_gb
    avail_gb=$(awk -v b="$avail_bytes" 'BEGIN { printf "%.2f", b/1024/1024/1024 }')

    local wired_bytes=$((wired_pages * PAGE_BYTES))
    local wired_gb
    wired_gb=$(awk -v b="$wired_bytes" 'BEGIN { printf "%.2f", b/1024/1024/1024 }')

    local compressed_bytes=$((compressed_pages * PAGE_BYTES))
    local compressed_gb
    compressed_gb=$(awk -v b="$compressed_bytes" 'BEGIN { printf "%.2f", b/1024/1024/1024 }')

    local total_bytes total_gb
    total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    total_gb=$(awk -v b="$total_bytes" 'BEGIN { printf "%.2f", b/1024/1024/1024 }')

    local pressure_level="normal"
    local mp_out
    mp_out=$(memory_pressure -Q 2>/dev/null || true)
    if [[ -z "$mp_out" ]]; then
        pressure_query_ok=false
    elif [[ "$mp_out" == *"critically low"* ]]; then
        pressure_level="critical"
    elif [[ "$mp_out" == *"detected memory pressure"* || "$mp_out" == *"warn"* ]]; then
        pressure_level="warn"
    fi

    local slm_rss_gb=0
    local slm_pid_alive=false
    local joint_gb=0
    local joint_over_threshold=false
    if [[ -n "${SLM_PID:-}" ]]; then
        if kill -0 "$SLM_PID" 2>/dev/null; then
            slm_pid_alive=true
            local slm_rss_kb
            slm_rss_kb=$(ps -o rss= -p "$SLM_PID" 2>/dev/null | tr -d ' ' || echo 0)
            slm_rss_kb="${slm_rss_kb:-0}"
            slm_rss_gb=$(awk -v k="$slm_rss_kb" 'BEGIN { printf "%.2f", k/1024/1024 }')
        fi
    fi

    local dismantle_rss_gb="$DISMANTLE_RSS_EST_GB"
    if [[ -n "${DISMANTLE_PID:-}" ]] && kill -0 "$DISMANTLE_PID" 2>/dev/null; then
        local d_rss_kb
        d_rss_kb=$(ps -o rss= -p "$DISMANTLE_PID" 2>/dev/null | tr -d ' ' || echo 0)
        d_rss_kb="${d_rss_kb:-0}"
        dismantle_rss_gb=$(awk -v k="$d_rss_kb" 'BEGIN { printf "%.2f", k/1024/1024 }')
    fi

    if awk -v t="$total_gb" 'BEGIN { exit !(t > 0) }'; then
        joint_gb=$(awk -v a="$slm_rss_gb" -v b="$dismantle_rss_gb" 'BEGIN { printf "%.2f", a+b }')
        if awk -v j="$joint_gb" -v t="$total_gb" -v p="$JOINT_DEGRADE_PCT" \
               'BEGIN { exit !(j > t * p / 100) }'; then
            joint_over_threshold=true
        fi
    fi

    local state="safe"
    local exit_code=0
    if [[ "$parse_failed" == "true" ]]; then
        state="critical"; exit_code=2
    elif [[ "$pressure_level" == "critical" ]] \
      || awk -v a="$avail_gb" -v t="$FREE_DEGRADED_GB" 'BEGIN { exit !(a < t) }'; then
        state="critical"; exit_code=2
    elif [[ "$pressure_level" == "warn" ]] \
      || [[ "$joint_over_threshold" == "true" ]] \
      || awk -v a="$avail_gb" -v t="$FREE_SAFE_GB" 'BEGIN { exit !(a < t) }'; then
        state="degraded"; exit_code=1
    fi

    if [[ "$emit_json" == "1" ]]; then
        cat <<JSON
{
  "captured_at_utc": "$(ts)",
  "state": "$state",
  "exit_code": $exit_code,
  "available_gb": $avail_gb,
  "wired_gb": $wired_gb,
  "compressed_gb": $compressed_gb,
  "total_gb": $total_gb,
  "pressure_level": "$pressure_level",
  "page_bytes": $PAGE_BYTES,
  "parse_failed": $parse_failed,
  "pressure_query_ok": $pressure_query_ok,
  "slm": {
    "pid": "${SLM_PID:-}",
    "alive": $slm_pid_alive,
    "rss_gb": $slm_rss_gb
  },
  "dismantle": {
    "pid": "${DISMANTLE_PID:-}",
    "rss_gb": $dismantle_rss_gb
  },
  "joint": {
    "gb": $joint_gb,
    "threshold_pct": $JOINT_DEGRADE_PCT,
    "over_threshold": $joint_over_threshold
  },
  "thresholds": { "safe_gb": $FREE_SAFE_GB, "degraded_gb": $FREE_DEGRADED_GB }
}
JSON
    else
        printf 'guard: %s | free=%sGB | pressure=%s | wired=%sGB | compressed=%sGB' \
            "$state" "$avail_gb" "$pressure_level" "$wired_gb" "$compressed_gb"
        if [[ -n "${SLM_PID:-}" ]]; then
            printf ' | slm=%sGB joint=%sGB' "$slm_rss_gb" "$joint_gb"
        fi
        if [[ "$parse_failed" == "true" ]]; then printf ' | parse_failed=true'; fi
        if [[ "$pressure_query_ok" == "false" ]]; then printf ' | pressure_query_failed=true'; fi
        printf '\n'
    fi
    return $exit_code
}

# ─── watch ────────────────────────────────────────────────────────────

cmd_watch() {
    local phase="${1:-phase1}"
    local POLL_INTERVAL_S="${POLL_INTERVAL_S:-10}"
    local COEXIST_LOG_LEVEL="${COEXIST_LOG_LEVEL:-info}"
    local ACTIVE_FILE="$EVIDENCE_DIR/.active"

    mkdir -p "$EVIDENCE_DIR"

    log() { printf '[coexist-watch %s] %s\n' "$(ts)" "$*"; }
    debug() { [[ "$COEXIST_LOG_LEVEL" == "debug" ]] && log "$@" || true; }

    if [[ -z "${SLM_PID:-}" ]]; then
        local detected
        detected=$(pgrep -f 'mamba_byte_train' 2>/dev/null | head -1)
        if [[ -n "$detected" ]]; then
            export SLM_PID="$detected"
            log "auto-detected SLM_PID=$SLM_PID (mamba_byte_train)"
        else
            log "no slm process found; relying on system-wide pressure only"
        fi
    fi

    walk_tree() {
        local root="$1"
        [[ -n "$root" ]] || return 0
        kill -0 "$root" 2>/dev/null || return 0
        printf '%s\n' "$root"
        local kids
        kids=$(pgrep -P "$root" 2>/dev/null || true)
        for k in $kids; do walk_tree "$k"; done
    }

    signal_tree() {
        local sig="$1" root="$2"
        while IFS= read -r p; do
            [[ -n "$p" ]] || continue
            kill "-$sig" "$p" 2>/dev/null || true
        done < <(walk_tree "$root")
    }

    PAUSED_PID=""
    PAUSED_GATE=""

    cleanup() {
        if [[ -n "$PAUSED_PID" ]]; then
            log "cleanup: resuming pid $PAUSED_PID before exit"
            signal_tree CONT "$PAUSED_PID"
        fi
        exit 0
    }
    trap cleanup INT TERM

    log "watch up — phase=$phase poll=${POLL_INTERVAL_S}s evidence=$EVIDENCE_DIR"

    while :; do
        sleep "$POLL_INTERVAL_S"

        if [[ ! -f "$ACTIVE_FILE" ]]; then
            if [[ -n "$PAUSED_PID" ]]; then
                log "active file gone; resuming stale pause on pid $PAUSED_PID"
                signal_tree CONT "$PAUSED_PID"
                PAUSED_PID=""; PAUSED_GATE=""
            fi
            debug "idle (no .active file)"
            continue
        fi

        local gate pid
        read -r gate pid <"$ACTIVE_FILE" 2>/dev/null || continue
        [[ -n "${gate:-}" && -n "${pid:-}" ]] || continue

        if ! kill -0 "$pid" 2>/dev/null; then
            debug "pid $pid in .active is gone; skipping tick"
            continue
        fi

        local state
        state=$("$SELF" probe --json 2>/dev/null \
            | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin); print(d.get("state","unknown"))
except Exception:
    print("unknown")' 2>/dev/null || echo unknown)

        local log_path="$EVIDENCE_DIR/$gate/throttle.log"
        mkdir -p "$EVIDENCE_DIR/$gate"

        case "$state" in
            safe)
                if [[ -n "$PAUSED_PID" && "$PAUSED_GATE" == "$gate" ]]; then
                    log "$gate: pressure → safe; resuming pid $PAUSED_PID (tree)"
                    signal_tree CONT "$PAUSED_PID"
                    printf '%s RESUME pid=%s gate=%s\n' \
                        "$(ts)" "$PAUSED_PID" "$gate" >> "$log_path"
                    PAUSED_PID=""; PAUSED_GATE=""
                else
                    debug "$gate: safe (no pause held)"
                fi
                ;;
            degraded|warn)
                if [[ -z "$PAUSED_PID" ]]; then
                    log "$gate: pressure → $state; pausing pid $pid (tree)"
                    signal_tree STOP "$pid"
                    printf '%s PAUSE  pid=%s gate=%s state=%s\n' \
                        "$(ts)" "$pid" "$gate" "$state" >> "$log_path"
                    PAUSED_PID="$pid"; PAUSED_GATE="$gate"
                else
                    debug "$gate: $state, already paused"
                fi
                ;;
            critical)
                if [[ -n "$PAUSED_PID" ]]; then
                    log "$gate: pressure → critical; releasing pause so run-gates can act"
                    signal_tree CONT "$PAUSED_PID"
                    printf '%s CRITRESUME pid=%s gate=%s\n' \
                        "$(ts)" "$PAUSED_PID" "$gate" >> "$log_path"
                    PAUSED_PID=""; PAUSED_GATE=""
                fi
                log "$gate: critical pressure observed; run-gates will halt at next item probe"
                ;;
            unknown)
                debug "$gate: probe returned unknown state; skip tick"
                ;;
        esac
    done
}

# ─── launch ───────────────────────────────────────────────────────────

cmd_launch() {
    local phase="${1:-phase1}"

    if [[ -z "${SLM_PID:-}" ]]; then
        local detected
        detected=$(pgrep -f 'mamba_byte_train' 2>/dev/null | head -1)
        if [[ -n "$detected" ]]; then
            export SLM_PID="$detected"
            printf '[coexist-launch %s] auto-detected SLM_PID=%s\n' "$(ts)" "$SLM_PID"
        fi
    fi

    "$SELF" watch "$phase" &
    local watcher_pid=$!
    printf '[coexist-launch %s] watcher up (pid=%s) — running %s gates\n' \
        "$(ts)" "$watcher_pid" "$phase"

    # Cleanup: TERM the watcher (its handler resumes any paused tree).
    cleanup_launch() {
        if kill -0 "$watcher_pid" 2>/dev/null; then
            kill -TERM "$watcher_pid" 2>/dev/null
            wait "$watcher_pid" 2>/dev/null
        fi
    }
    trap cleanup_launch INT TERM EXIT

    "$REPO_ROOT/tools/haul/run-gates.sh" "$phase"
    local rc=$?

    cleanup_launch
    trap - INT TERM EXIT
    return $rc
}

# ─── dispatch ─────────────────────────────────────────────────────────

usage() {
    cat <<USAGE
coexist.sh — autonomous-haul memory safety + launcher

Usage:
  $0 probe [--json]            # single-shot pressure check (exit 0/1/2)
  $0 watch [phase]             # background sidecar; pauses gates on pressure
  $0 launch [phase]            # one-command: watcher + run-gates together

See header comment for tunables.
USAGE
}

case "${1:-}" in
    probe)  shift; cmd_probe "$@" ;;
    watch)  shift; cmd_watch "$@" ;;
    launch) shift; cmd_launch "$@" ;;
    -h|--help|help|"") usage ;;
    *) printf 'unknown subcommand: %s\n\n' "$1"; usage; exit 2 ;;
esac
