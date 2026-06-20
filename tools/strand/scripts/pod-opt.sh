#!/usr/bin/env bash
# pod-opt.sh — one-shot speed-optimization deploy (2026-06-10 ~20:10 local).
# 1) Defer q4_l12 (skip-sentinel; ~21h saved; delete the json to re-enable).
# 2) Ladder threads 24 -> 27 (the cgroup quota is 27.2): edit applies via a
#    pipe-safe babysitter that relaunches the ladder at the mp_light boundary
#    (killing it mid-config would SIGPIPE the running quant through `| tail`).
# 3) Report state. (The chain restart is done separately via scp'd v3.2.)
set -u
cd /workspace/strand

# ── 1) q4_l12 deferral sentinel ──
d=scratch/qwen-7b/reopen/q4_l12
mkdir -p "$d"
[ -f "$d/ppl_q4_l12.json" ] || cat > "$d/ppl_q4_l12.json" << 'J'
{"tag":"q4_l12","ppl":null,"skipped":"deferred 2026-06-10 for time (Viterbi cost ~4x at k=4,l=12 ~= 21h); delete this file to re-run"}
J
cp "$d/ppl_q4_l12.json" /workspace/strand-results/ 2>/dev/null
echo "q4_l12 sentinel placed"

# ── 2) ladder thread bump (on-disk edit + boundary babysitter) ──
sed -i 's/--threads 24/--threads 27/' /workspace/strand-ladder.sh
echo "ladder script now: $(grep -o -- '--threads [0-9]*' /workspace/strand-ladder.sh | head -1)"
cat > /workspace/ladder-bump.sh << 'B'
#!/usr/bin/env bash
# Wait for the mp_light boundary, then bounce the ladder so the 27-thread edit
# (and the q4 sentinel skip) take effect. Kills config-3's first seconds at
# worst — per-shard resume makes that free.
while [ ! -f /workspace/strand-results/ppl_mp_light_l12_out1.json ]; do sleep 5; done
sleep 5
pkill -f "strand-ladder[.]sh"
pkill -f "strand-7b-ppl[.]sh"
for p in $(pgrep quantize-model); do kill "$p"; done
sleep 5
cd /workspace/strand
nohup /workspace/strand-ladder.sh >> /workspace/strand-ladder.log 2>&1 &
date > /workspace/.ladder-bumped
B
chmod +x /workspace/ladder-bump.sh
pgrep -f "ladder-bump[.]sh" >/dev/null || { nohup bash /workspace/ladder-bump.sh > /dev/null 2>&1 & }
echo "babysitter armed: $(pgrep -f "ladder-bump[.]sh" | head -1)"

# ── 3) state report ──
echo "mp_light shards done: $(ls scratch/qwen-7b/reopen/mp_light_l12_out1/.tmp-*.json 2>/dev/null | wc -l)/4"
echo "32B index-complete: $(python3 -c "import json,glob;w=len(set(json.load(open('scratch/qwen-32b/model.safetensors.index.json'))['weight_map'].values()));h=len(glob.glob('scratch/qwen-32b/*.safetensors'));print(f'{h}/{w}')" 2>/dev/null || echo 'index missing')"
echo "70B shards: $(ls scratch/llama2-70b/*.safetensors 2>/dev/null | wc -l)/15"
