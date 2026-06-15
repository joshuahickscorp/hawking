#!/usr/bin/env bash
# podctl — THE pod operations console (2026-06-11). Every pod intervention goes
# through here: patterns are constructed (never literal), kills take process
# TREES (wrapper + heredoc children), every action echoes what it did.
# Born from 5 self-match incidents and one 134GB ghost download.
# Usage: podctl.sh <status|stop-chain|start-chain|stop-downloads|purge-dir <name>|logs|mem
#                   |runs|launch <name> '<command>'|stop-run <name|pgid>>
# RUN REGISTRY (audit measurement/orchestration): every launch goes through setsid
# and registers its PGID in /workspace/strand-run-registry.tsv (pgid TAB epoch TAB
# name TAB cmd). `runs` lists registered PGIDs + liveness; `stop-run` kills the
# whole process TREE by PGID — no pattern matching, so self-match and unmatchable
# bare-heredoc children ("python3 -") are structurally impossible.
SSHOPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -p 40078)
HOST=root@213.192.2.110
run(){ ssh "${SSHOPTS[@]}" "$HOST" "$1" 2>&1 | grep -v 'Warning:\|post-quantum\|store now\|upgraded'; }

case "${1:-status}" in
status)
  run 'C=pod-chai; G=pod-governo; D=download-mode
echo "chain:    $(pgrep -fc "${C}[n]") proc(s)"
echo "governor: $(pgrep -fc "${G}[r]") proc(s)"
echo "dl:       $(pgrep -fc "${D}[l]") wrapper(s) + $(pgrep -fc "^python3 -$") heredoc python(s)"
echo "quant:    $(pgrep -c quantize-model) proc(s) $(q=$(pgrep quantize-model | head -1); [ -n "$q" ] && ps -o etime= -p "$q")"
echo "mem: $(awk "{printf \"%d\",\$1/1e9}" /sys/fs/cgroup/memory.current)/125GB  vol: $(du -s /workspace 2>/dev/null | awk "{printf \"%d\",\$1/1048576}")/200GB"
grep -a "\[chain" /workspace/strand-chain.log | tail -3' ;;
stop-chain)
  run 'pkill -f "pod-chai[n]" && echo "chain stopped" || echo "chain was not running"' ;;
start-chain)
  run 'C=/workspace/pod-chai; C="${C}n.sh"
pgrep -f "pod-chai[n]" >/dev/null && { echo "already running"; exit 0; }
setsid nohup bash "$C" > /dev/null 2>&1 &
pid=$!; sleep 2
pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d " ")
[ -n "$pgid" ] || pgid=$pid   # setsid execs in place here: new PGID == $!
printf "%s\t%s\t%s\t%s\n" "$pgid" "$(date +%s)" "chain" "$C" >> /workspace/strand-run-registry.tsv
echo "chain: pid=$pid pgid=${pgid:-?} (registered)"' ;;
runs)
  run 'R=/workspace/strand-run-registry.tsv
[ -s "$R" ] || { echo "registry empty ($R)"; exit 0; }
printf "%-8s %-5s %-16s %-12s %s\n" PGID LIVE STARTED NAME CMD
while IFS="$(printf "\t")" read -r pgid ts name cmd; do
  [ -n "$pgid" ] || continue
  if pgrep -g "$pgid" >/dev/null 2>&1; then live=yes; else live=dead; fi
  when=$(date -d "@$ts" "+%m-%d %H:%M" 2>/dev/null || echo "$ts")
  printf "%-8s %-5s %-16s %-12s %.60s\n" "$pgid" "$live" "$when" "$name" "$cmd"
done < "$R"' ;;
launch)
  { [ -z "${2:-}" ] || [ -z "${3:-}" ]; } && { echo "usage: podctl.sh launch <name> '<command>'"; exit 1; }
  Nq=$(printf '%q' "$2"); Cq=$(printf '%q' "$3")
  run "N=$Nq; CMDSTR=$Cq
setsid nohup bash -c \"\$CMDSTR\" >> \"/workspace/run-\$N.log\" 2>&1 &
pid=\$!; sleep 1
pgid=\$(ps -o pgid= -p \"\$pid\" 2>/dev/null | tr -d ' ')
[ -n \"\$pgid\" ] || pgid=\$pid   # setsid execs in place here: new PGID == \$!
printf '%s\t%s\t%s\t%s\n' \"\$pgid\" \"\$(date +%s)\" \"\$N\" \"\$CMDSTR\" >> /workspace/strand-run-registry.tsv
echo \"launched \$N pid=\$pid pgid=\${pgid:-?} log=/workspace/run-\$N.log (registered)\"" ;;
stop-run)
  [ -z "${2:-}" ] && { echo "usage: podctl.sh stop-run <name|pgid>"; exit 1; }
  Aq=$(printf '%q' "$2")
  run "A=$Aq; R=/workspace/strand-run-registry.tsv
case \"\$A\" in
  *[!0-9]*) pgid=\$(awk -F'\t' -v n=\"\$A\" '\$3==n{p=\$1} END{print p}' \"\$R\" 2>/dev/null) ;;
  *) pgid=\$A ;;
esac
[ -n \"\$pgid\" ] || { echo \"no registered run named \$A\"; exit 1; }
if kill -TERM -- \"-\$pgid\" 2>/dev/null; then echo \"TERM sent to tree pgid=\$pgid\"
else echo \"pgid \$pgid not running (or already dead)\"; fi" ;;
stop-downloads)
  run 'pkill -f "download-mode[l]"; pkill -f "^python3 -$"; sleep 1
echo "remaining: $(pgrep -fc "download-mode[l]") wrappers, $(pgrep -fc "^python3 -$") pythons"' ;;
purge-dir)
  [ -z "$2" ] && { echo "purge-dir needs a scratch dirname (e.g. llama2-70b)"; exit 1; }
  run "du -sh /workspace/strand/scratch/$2 2>/dev/null; rm -rf /workspace/strand/scratch/$2 && echo purged; du -s /workspace | awk '{printf \"vol now: %d GB\n\", \$1/1048576}'" ;;
logs)
  run 'echo "=== chain:"; grep -a "\[chain" /workspace/strand-chain.log | tail -8
echo "=== governor:"; tail -4 /workspace/pod-governor.log' ;;
mem)
  run 'echo "cgroup: $(awk "{printf \"%.1f\",\$1/1e9}" /sys/fs/cgroup/memory.current)/125GB"
ps -eo rss,comm --sort=-rss | head -4' ;;
*) echo "usage: podctl.sh <status|stop-chain|start-chain|stop-downloads|purge-dir <name>|logs|mem|runs|launch <name> '<cmd>'|stop-run <name|pgid>>" ;;
esac
