# DELEGATION — ANTI-COMPLACENCY v1 (deterministic enforcement; ctl.py + runner)

Steer S004: a Doctor-valid conventional q3/q4 is a CONVENTIONAL_ANCHOR, never
automatically a frontier. Patients must not retire without at least one credible
NONCONVENTIONAL/aggressive probe. Encode this DETERMINISTICALLY (the orchestrator
enforces it; no model reasoning per candidate). Repo `/Users/scammermike/Downloads/hawking`,
python `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`. Branch odyssey-i.
The driver is PAUSED for this integration — no concurrent control-plane writer.

## Read first
- `workspace/campaign/odyssey/ODYSSEY_POLICY.json` — the machine-readable policy you ENFORCE (candidate_classes, conventionality_gate, aggressive_ladder, retirement_gate, arch_objective, accounting_gates). Consume it; do not hardcode duplicate thresholds.
- `tools/odyssey_ctl.py` (cycling + retirement + templates + completions) and `tools/odyssey_patient_runner.py` (has --gravity <spec> + --sensitivity + per-organ sensitivity).

## BUILD 1 — runner aggressive `--gravity` specs (bounded, complete accounting)
Extend `--gravity` to accept AGGRESSIVE specs and emit `odyssey.patient.gravity.v1` receipts with COMPLETE bpw accounting (payload+scales+biases+metadata; record `nominal_bits` AND `complete_bpw`), Doctor battery + delta, and `candidate_class` + `conventionality`:
- `q2-g32-experts` (experts 2-bit group32, attn/router 4-bit) — AGGRESSIVE_QUANT.
- `mixed-q2q3-experts` — SENSITIVITY-DRIVEN: q2 base, promote to q3 the organs/experts with the worst per-organ sensitivity delta (reuse the patient's `representation.per_organ_sensitivity`); record which components were protected — STRUCTURAL_GRAVITY.
- On any FAILURE (delta_hits worse than a small threshold): compute cheap FAILURE LOCALIZATION — rank organs by (sensitivity delta) to name the most-likely responsible component; write it in the receipt `failure_localization`. Do NOT globally retreat; the receipt records the targeted-repair suggestion (protect that component).
- dense/hybrid analogues: `q2-g64` and `q2-g64-attn-mlp` (protect ssm/norm), same accounting.
Keep existing modes working. Runner-only for these.

## BUILD 2 — ctl.py: aggressive obligations + conventionality + retirement gate
- New templates `gravity-aggressive-moe` / `gravity-aggressive-dense` / `gravity-aggressive-hybrid` that render contracts running the aggressive spec above (write_set = runner NOT included since the modes now exist — data-only: patient packet + receipts, so they auto-complete + parallelize).
- Add exactly ONE `gravity-aggressive-<class>` to each patient class's REQUIRED retirement set.
- CONVENTIONALITY tagging: when harvesting/recording a gravity completion, set `candidate_class`/`conventionality` from the receipt into the completion entry + packet.
- RETIREMENT GATE (per ODYSSEY_POLICY.retirement_gate): DEFAULT REFUSE retirement when `conventional_anchor_exists AND aggressive_probe_attempted==false AND cheap_credible_mechanisms_remain`; the ONLY exception is an explicit `LOW_INFORMATION_VALUE` receipt for that patient. `aggressive_probe_attempted` = a `gravity-aggressive-*` completion exists (VERIFIED or REFUTED — a failed aggressive probe still counts as attempted + is valuable data).
- Cycle summary: add `patients_retired_without_nonconventional_probe` counter (desired 0).

## BUILD 3 — self-checks (zero-Opus, §43-46)
Extend `--self-check`:
- anti-complacency: a patient with conventional gravity VERIFIED but NO aggressive probe is NOT retire-eligible; after an aggressive probe is attempted it becomes eligible.
- conventionality: a q3-affine gravity receipt classifies CONVENTIONAL_ANCHOR; a mixed/sensitivity-driven one classifies STRUCTURAL_GRAVITY.
- failure-localization: a failing aggressive receipt yields a non-empty `failure_localization` naming an organ.
Keep all existing self-check assertions green.

## Constraints
Deterministic only — no per-candidate model reasoning. Reuse existing code + doctor_seal + sensitivity. Honor worker_gate/4bit. No commit/push/launchd. Complete accounting always (no fake density).

## ACCEPTANCE
- `python3 tools/odyssey_patient_runner.py --oxx O006 --weights <O006 4bit or snapshot> --gravity q2-g32-experts --out receipts/odyssey-i/O006_GRAVITY_q2-g32-experts.json` produces a valid receipt with complete_bpw, candidate_class=AGGRESSIVE_QUANT, Doctor delta, exit 0.
- `python3 tools/odyssey_ctl.py --self-check` passes including the 3 new checks.
- `python3 tools/odyssey_ctl.py cycle --dry-run` shows no MoE patient retire-eligible on conventional gravity alone (each needs a gravity-aggressive probe); O005 (already retired) is untouched.

## SCOPE
WRITE tools/odyssey_ctl.py
WRITE tools/odyssey_patient_runner.py
WRITE workspace/campaign/odyssey/
WRITE receipts/odyssey-i/
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json, workspace/campaign/odyssey/patients/, tools/worker_gate.py, tools/doctor_seal.py
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py --self-check` and `python3 tools/odyssey_ctl.py cycle --dry-run` — must pass, exit 0.
