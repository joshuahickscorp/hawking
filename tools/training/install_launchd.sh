#!/usr/bin/env bash
# install_launchd.sh — install the pipeline launchd agent so it auto-runs
# every 60s while the laptop is awake. Survives reboots.
#
# Usage:
#   tools/training/install_launchd.sh           # install
#   tools/training/install_launchd.sh --uninstall  # remove
#
# Why launchd: bash background loops die when the laptop closes. launchd
# launches your job as soon as the OS wakes back up. The pipeline scripts
# are idempotent, so re-running picks up exactly where it left off.

set -eu

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LABEL="com.user.dismantle.pipeline"
PLIST_SRC="$PROJECT_ROOT/tools/training/$LABEL.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "${1:-}" = "--uninstall" ]; then
  if [ -f "$PLIST_DST" ]; then
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "uninstalled $PLIST_DST"
  else
    echo "not installed"
  fi
  exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents"
# Substitute project root into the template.
sed "s|REPLACE_WITH_PROJECT_ROOT|$PROJECT_ROOT|g" "$PLIST_SRC" > "$PLIST_DST"
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"
echo "installed: $PLIST_DST"
echo "verify:    launchctl list | grep $LABEL"
echo "logs:      $PROJECT_ROOT/training_data/c2_hidden/eagle3_v0/pipeline/launchd.{out,err}.log"
echo "uninstall: $0 --uninstall"
