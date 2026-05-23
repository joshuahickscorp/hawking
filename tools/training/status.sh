#!/usr/bin/env bash
# Live corpus build status. Run from anywhere:
#   bash /Users/scammermike/Downloads/dismantle/tools/training/status.sh
# Defaults to refreshing every 60s. Override:
#   INTERVAL=30 bash .../status.sh
# Press Ctrl-C to stop. Doesn't affect the build.

set -u
cd "$(dirname "$0")/../.." || exit 1
INTERVAL="${INTERVAL:-60}"

while true; do
    clear
    echo "=== dismantle corpus build — $(date '+%Y-%m-%d %H:%M:%S') ==="
    python3 - <<'PY'
import json, glob, os, re, subprocess, sys
try:
    d = json.load(open('artifacts/calibration/heartbeat.json'))
except FileNotFoundError:
    print("no heartbeat.json — watchdog may not be running")
    sys.exit(0)
shards = sorted(glob.glob('artifacts/calibration/v2_lite_corpus/shard_*.parquet'))
mtimes = [os.path.getmtime(s) for s in shards[-6:]]
rate = (len(mtimes)-1)/(mtimes[-1]-mtimes[0])*3600 if len(mtimes) > 1 else 0
remaining = d['shards_target'] - d['shards_complete']
eta_h = remaining/rate if rate > 0 else float('inf')
log_tail = subprocess.run(['tail','-c','6000','artifacts/calibration/overnight.log'],
                          capture_output=True, text=True).stdout
tps = re.findall(r'(\d+\.\d+)s/it', log_tail)[-10:]
sec_per_batch = sum(float(x) for x in tps)/len(tps) if tps else 0
# batch_size=4, max_tokens_per_seq=256 → 1 it = 1024 tokens
tok_per_s = 1024/sec_per_batch if sec_per_batch else 0
free_gb = subprocess.run(['df','-g','.'], capture_output=True, text=True).stdout.splitlines()[1].split()[3]
print(f"  status:     {d['status']}")
print(f"  shards:     {d['shards_complete']}/{d['shards_target']} ({100*d['shards_complete']/d['shards_target']:.1f}%)")
print(f"  throughput: ~{tok_per_s:.0f} tok/s   |   {rate:.1f} shards/hr")
print(f"  eta:        {eta_h:.1f} h remaining")
print(f"  disk free:  {free_gb} GB")
print(f"  build pid:  {d['build_pid']}   watchdog pid: {d['watchdog_pid']}")
print(f"  last hb:    {d['ts']}")
PY
    echo ""
    echo "  refreshing every ${INTERVAL}s (Ctrl-C to stop, this doesn't affect the build)"
    sleep "$INTERVAL"
done
