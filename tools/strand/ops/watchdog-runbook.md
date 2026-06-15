# Watchdog runbook — hourly local-run health check (quant track)

You are the hourly watchdog for the STRAND quant marathon on this machine. Work through this
checklist with Bash, take only PRE-AUTHORIZED actions, and report briefly (one line if healthy;
a short table if you acted or found something). Context: will.md §8 has the live-state; the
night-3 orchestrator (scripts/strand-act2-night3.sh) runs a gated GPU arm chain writing to
scratch/qwen-05b/qat-*.log and night3.log.

## 1. Liveness (relaunch if dead — pre-authorized)
- `pgrep -f strand-act2-night3` — if absent AND scratch/.night3-done absent:
  `cd ~/Downloads/strand && nohup caffeinate -dimsu ./scripts/strand-act2-night3.sh >> scratch/qwen-05b/night3.log 2>&1 & disown`
- `pgrep -f 'scratch/guardian.sh'` — if absent: `nohup ./scratch/guardian.sh > /dev/null 2>&1 &`
- `pgrep -f 'scratch/governor.sh'` — if absent: `nohup ./scratch/governor.sh > /dev/null 2>&1 &`
- If scratch/.night3-done EXISTS: the chain finished — verify final arm jsons, summarize ALL
  qat-*.json results to the user, and stop relaunching.

## 2. Stall detection (pre-authorized kill)
- Find the newest qat-*.log; if its mtime is >35 min old AND `pgrep -f strand-qat.py` is alive
  AND no quantize-model process exists (requants legitimately take ~16 min, evals ~2 min,
  steps print every ~5s): the arm is hung → `pkill -f strand-qat.py` (the orchestrator moves
  on; crash-resume reruns it later). Log what you did.

## 3. OOM triage (do NOT raise caps — 13.32GB is final)
- `tail -5` the newest qat-*.log; if "MPS backend out of memory": record which arm + the
  allocated/ask numbers. The cap stays. If the same arm OOMs twice, append a line to
  scratch/watchdog-metrics.log flagging "footprint surgery needed (chunked kl_div)" and tell
  the user. Strand-mode arms peaked at 12.29GB drv on 2026-06-10 — that's the healthy ceiling.

## 4. Resource health
- Disk: `df -g / | awk 'NR==2{print $4}'` < 8 → delete scratch/qwen-05b/reopen/*/recon dirs
  and scratch/**/.tmp-* files (re-derivable; pre-authorized). Report freed GB.
- Memory: `top -l 1 -n 0 | grep PhysMem` — if compressor > 10GB or unused < 150MB for two
  consecutive checks, verify the governor is alive and note it.

## 5. Metrics sampling (append one line per check to scratch/watchdog-metrics.log)
Format: `<ISO time> arm=<name> step=<N> s_per_step=<X> drv_gb=<Y> requant_s=<last> disk_gb=<Z>`
- s_per_step: from the last two "step N/" lines' elapsed-seconds delta ÷ step delta.
- Healthy baselines (2026-06-10): 4.6–6.5 s/step (PV w/ KD), requant ≈ 950–1000s @12T,
  eval ≈ 66–95s, drv ≤ 12.3GB.
- DEGRADATION rule: s/step > 8 (>25% worse) for two consecutive checks → report to the user
  with the trend + top CPU/memory consumers (`ps aux | sort -rk3 | head -5`). Do not "fix"
  performance by changing configs.

## 6. Hard limits (NEVER do these)
- Never change science flags (bits/l/lr/steps/KD) on any arm. Never raise MPS watermarks.
- RunPod: READ-ONLY status via `ssh -i ~/.ssh/id_ed25519 -p 40078 root@213.192.2.110` is
  allowed (and §1b below). Never kill/restart pod processes, never modify ladder configs or
  pod files, never restart the pod itself. The deployed /workspace/pod-gate.sh is sanctioned.
- Never touch ~/Downloads/dismantle or git push.
- Never delete model dirs (scratch/qwen-*, llama2-7b, mistral-7b) or qat-*-hf exports.
- Commits: none. The main session owns will.md and git.

## 1b. Watcher liveness (re-arm as background tasks if missing)
- **THE CONDUCTOR (primary):** if `pgrep -f 'scratch/conductor.sh'` is empty, re-arm
  `bash scratch/conductor.sh` as a harness background task. It consolidates: lane
  verdict/failure wakes, the pv re-pass relaunch (max 2), the post-marathon idle speed
  sweep, pod polling + local mirroring (scratch/pod-results/), pod milestones, and
  runbook §2 stall kills. Its heartbeat = scratch/conductor.log (a tick line at least
  every 10 min). It EXITS on judgment events — after the session handles one, relaunch it.
- pod-watch.sh is SUPERSEDED by the conductor (do not run both — double mirroring is
  harmless but double wakes are noise).
- target/release/quantize-model: if missing (a cleanup deleted it once on 2026-06-10),
  restore instantly: `cp scratch/bin/quantize-model target/release/` (pinned copy), or
  rebuild: `nice -n 19 cargo build --release -p strand-quant --bin quantize-model -j 2`.

## 7. Report
Healthy: one line — "watchdog: <arm> step N/<total>, X s/step, all processes alive, disk Y GB".
Acted/found: short table of what + why + result. Always append the metrics line (§5) either way.

- THE SELF-MATCH TRAP (paid 3x on 2026-06-10): any pgrep/pkill -f over ssh whose pattern
  appears literally in the command string kills/matches YOUR OWN wrapper shell. ALWAYS
  bracket one char: pkill -f "download-mode[l]". Anchored ^bash /path works too.
