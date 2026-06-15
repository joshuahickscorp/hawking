#!/usr/bin/env bash
# pod-watch.sh — 10-min RunPod ladder poll. Launched as a harness-tracked background
# task from the orchestrator session: EXITS on milestone/failure (which re-invokes the
# assistant), heartbeat-exits after 6h so the watcher gets re-armed fresh.
# Samples append to scratch/pod-watch.log. Read-only on the pod, always.
cd "$(dirname "$0")/.."
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519 -p 40078 root@213.192.2.110"
PW=scratch/pod-watch.log
REMOTE='echo "jsons=$(ls /workspace/strand-results/ 2>/dev/null | grep -c "^ppl_.*\.json$") ladder=$(pgrep -f strand-ladder.sh >/dev/null && echo up || echo DOWN) cfg=$(grep -oE "== (START|DONE) [^ ]+" /workspace/strand-ladder.log 2>/dev/null | tail -1 | sed "s/== //;s/ /:/") shards=$(ls /workspace/strand/scratch/*/reopen/*/.tmp-*.json 2>/dev/null | wc -l)"'

fails=0
end=$((SECONDS+21600))
base=$($SSH 'ls /workspace/strand-results/ 2>/dev/null | grep -c "^ppl_.*\.json$"' 2>/dev/null)
[ -z "$base" ] && base=0
echo "[pw $(date '+%d %H:%M:%S')] armed; baseline ladder-jsons=$base" >> "$PW"

while [ $SECONDS -lt $end ]; do
    sleep 600
    S=$($SSH "$REMOTE" 2>/dev/null)
    if [ -z "$S" ]; then
        fails=$((fails+1)); echo "[pw $(date '+%d %H:%M:%S')] ssh FAIL #$fails" >> "$PW"
        if [ "$fails" -ge 3 ]; then
            echo "POD UNREACHABLE 3 consecutive polls (30 min) — pod stopped or IP:port changed (ask owner for: echo \"TCP=\$RUNPOD_PUBLIC_IP:\$RUNPOD_TCP_PORT_22\")"
            exit 0
        fi
        continue
    fi
    fails=0
    echo "[pw $(date '+%d %H:%M:%S')] $S" >> "$PW"
    # continuous local mirror: every poll pulls all result jsons + key logs (cheap, KBs)
    scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P 40078         "root@213.192.2.110:/workspace/strand-results/*.json" scratch/pod-results/ 2>/dev/null
    scp -q -o BatchMode=yes -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -P 40078         "root@213.192.2.110:/workspace/strand-{chain,gate,ladder}.log" scratch/pod-results/ 2>/dev/null
    n=$(echo "$S" | sed -n 's/.*jsons=\([0-9]*\).*/\1/p'); [ -z "$n" ] && n=$base
    lad=$(echo "$S" | sed -n 's/.*ladder=\([A-Za-z]*\).*/\1/p')
    if [ "$n" -gt "$base" ]; then
        echo "MILESTONE: new 7B result(s) banked on pod ($base -> $n) | $S"
        exit 0
    fi
    if [ "$lad" = "DOWN" ]; then
        echo "LADDER EXITED (complete or died) | $S"
        exit 0
    fi
done
echo "HEARTBEAT: 6h elapsed, pod-watch needs re-arm | last: $(tail -1 "$PW")"
