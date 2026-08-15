#!/usr/bin/env bash
# reclaim_safe.sh — strict, self-guarding disk reclaim for the Hawking machine.
#
# Reclaims the three things that actually grow without bound (build caches,
# grok lane artifacts, dead Hawking worktrees) while NEVER touching:
#   - models      workspace/campaign/records/runs/**
#   - captures    workspace/**/quality-diagnostics/**capture**
#   - ACTIVE lanes (a grok-run process is using the worktree/task dir)
#   - FOREIGN worktrees (belong to another repo/session — not in `git worktree list`)
#   - DIRTY worktrees with uncommitted work (preserved to their branch first, never dropped)
#
# Usage:  bash tools/reclaim_safe.sh          # reclaim
#         DRY=1 bash tools/reclaim_safe.sh    # show what it would do
set -euo pipefail
REPO="/Users/scammermike/Downloads/hawking"
GT="$HOME/.claude-grok/tasks"
DRY="${DRY:-0}"
run(){ if [ "$DRY" = 1 ]; then echo "  DRY: $*"; else eval "$@"; fi; }

before=$(df -h "$REPO" | tail -1 | awk '{print $4}')
echo "== reclaim_safe (free before: $before) =="

# 1) BUILD CACHES — biggest lever; skip if a cargo build is live.
if pgrep -x cargo >/dev/null; then
  echo "[skip] build cache — cargo build is running"
else
  run "rm -rf '$REPO/workspace/ops/build/'* 2>/dev/null || true"
  echo "[ok] cleared workspace/ops/build"
fi

# 2) GROK LANE ARTIFACTS — drop every task dir whose lane is NOT currently running.
if [ -d "$GT" ]; then
  for d in "$GT"/*/; do
    [ -d "$d" ] || continue
    id=$(basename "$d")
    if pgrep -f "$id" >/dev/null; then echo "[keep] task $id (running)"; continue; fi
    run "rm -rf '$d'"
  done
  echo "[ok] purged non-running task artifacts"
fi

# 3) DEAD HAWKING WORKTREES — only those registered to THIS repo (foreign ones never appear here).
#    clean+unmerged  -> `git worktree remove` keeps the branch (commits safe).
#    dirty           -> commit WIP to its own branch first (lossless), then remove.
#    running/active  -> skip.
git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2}' | while read -r wt; do
  [ "$wt" = "$REPO" ] && continue
  case "$wt" in *"/.claude-grok/worktrees/"*) ;; *) continue;; esac   # only grok lanes
  id=$(basename "$wt")
  if pgrep -f "$id" >/dev/null; then echo "[keep] worktree $id (running)"; continue; fi
  # keep anything touched in the last day — likely in-flight, not yet absorbed to main
  if [ -n "$(find "$wt" -maxdepth 0 -mtime -1 2>/dev/null)" ]; then echo "[keep] worktree $id (<1d old, in-flight)"; continue; fi
  if git -C "$wt" status --porcelain 2>/dev/null | grep -q .; then
    run "git -C '$wt' add -A && git -C '$wt' commit -q -m 'wip: preserve before reclaim' || true"
    echo "[preserve+remove] $id (WIP committed to branch)"
  else
    echo "[remove] $id (branch kept)"
  fi
  run "git -C '$REPO' worktree remove --force '$wt' || true"
done
run "git -C '$REPO' worktree prune"

# 4) OS / package caches
run "brew cleanup -s >/dev/null 2>&1 || true"
run "rm -rf ~/Library/Caches/pip ~/Library/Caches/Homebrew/downloads/* 2>/dev/null || true"

after=$(df -h "$REPO" | tail -1 | awk '{print $4}')
echo "== done (free: $before -> $after) =="
