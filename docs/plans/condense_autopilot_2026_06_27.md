# Condense Autopilot - 2026-06-27

Goal: keep the 7B frontier moving toward the physical quality-per-bit floor with
deterministic next experiments, not manual overnight babysitting.

## Current automated loop

1. `run_7b_frontier.sh` launches/adopts the ladder, health monitor, caffeinate,
   and keepalive supervisor.
2. `frontier_keepalive.sh` relaunches if the ladder exits before the run is
   genuinely exhausted.
3. `frontier_conductor.py` watches the JSONL, promotes good doctor results and
   Pareto frontier points into `7b_frontier_promotions.*`, plants an autopilot
   inject if one is missing, and asks the launcher to adopt missing supervisors.
4. `frontier_autopilot.py` reads `reports/cron/7b_frontier.jsonl`, appends the
   next justified configs via `{OUTP}_inject.py`, writes
   `{OUTP}_autopilot_state.json`, and re-arms itself while work remains.
5. `doctor_lora.py` saves LoRA adapters, not fused full-model checkpoints, and
   can target a module subset via `DOCTOR_TARGET_REGEX`.

## Deterministic policy

The policy spends more bits only when the previous smaller spend bought quality:

- Seed the Pareto doctor targets: `mp-4a3f`, `3-AWQ`, `4-AWQ`.
- If rank 16 helps by at least `AUTOPILOT_MIN_GAIN_PCT` percentage points,
  explore nearby AWQ alpha and outlier retention.
- If rank 32 improves over rank 16 by at least
  `AUTOPILOT_MIN_RANK_GAIN_PCT`, queue rank 64 and a longer lower-LR run.
- If a doctor hurts quality, retry once with lower LR.
- Stop appending only when no pending configs remain and no completed result
  justifies another deterministic candidate.

## Research-to-code mapping

- AWQ/SmoothQuant family: continue alpha and outlier sweeps around successful
  basins.
- SpQR/SqueezeLLM family: use sparse outlier retention as a first-class axis,
  especially at 3-bit and mixed 4-attn/3-ffn.
- QLoRA/BitDistiller/LLM-QAT family: adapter-only doctor with KD and restartable
  checkpoints.
- QuaRot/SpinQuant/QuIP#/QTIP family: add rotation/incoherence and codebook lanes
  only after an artifact/runtime format exists; these are not safe one-line
  injects because serving must apply the transformed representation correctly.
- KIVI/TurboQuant KV-cache family: separate serving frontier lane for Apple
  Silicon advantage; quality-per-bit for weights is not the whole user-visible
  speed/fit story.

## Next implementation lanes

1. Add targeted doctor injects: attention-only, FFN-only, and FFN-heavy module
   regexes to measure recovered PPL per adapter bit.
2. Add an offline rotation oracle from f16 weights, not requantized Q4, then
   gate any real format work on PPL/logit KL.
3. Promote per-channel int4 KV cache behind a long-context PPL and argmax gate.
4. Add a final scorecard that ranks configs by Pareto frontier, PPL recovery per
   added bpw, artifact size, and Apple Silicon serving throughput.
