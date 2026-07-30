#!/usr/bin/env bash
# Shared laboratory shell helpers for bench/training scripts.
# Prefer `python3.12 -m lab.bench_harness run <spec>` for orchestration.
set -euo pipefail

lab_stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

lab_log() {
  local log="${LAB_LOG:-/dev/stderr}"
  echo "[$(lab_stamp)] $*" | tee -a "$log" 2>/dev/null || echo "[$(lab_stamp)] $*"
}

lab_write_status() {
  local status_path="$1" module="$2" state="$3" note="${4:-}"
  local uptime=0 free_gb=0
  if [[ -n "${LAB_START_EPOCH:-}" ]]; then
    uptime=$(( $(date +%s) - LAB_START_EPOCH ))
  fi
  free_gb=$(df -g . 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)
  cat > "${status_path}.tmp" <<EOF
{"ts":"$(lab_stamp)","pid":$$,"current_module":"$module","state":"$state","note":"$note","uptime_seconds":$uptime,"free_disk_gb":$free_gb}
EOF
  mv "${status_path}.tmp" "$status_path"
}

lab_wait_if_paused() {
  local module="${1:-work}"
  local pause="${LAB_PAUSE_FLAG:-artifacts/runs/PAUSE}"
  local resume="${LAB_RESUME_FLAG:-artifacts/runs/RESUME}"
  if [[ -f "$pause" ]]; then
    lab_log "PAUSED at $module — touch $resume to continue"
    while [[ -f "$pause" ]]; do
      sleep "${LAB_PAUSE_POLL:-15}"
      if [[ -f "$resume" ]]; then
        rm -f "$pause" "$resume"
        lab_log "resumed"
        break
      fi
    done
  fi
}

lab_detect_power_source() {
  local prefer="${1:-auto}"
  case "$prefer" in
    macmon|powermetrics) echo "$prefer"; return;;
  esac
  if command -v macmon >/dev/null 2>&1; then echo macmon; return; fi
  if command -v powermetrics >/dev/null 2>&1; then echo powermetrics; return; fi
  echo none
}

lab_die() { echo "error: $*" >&2; exit 64; }
