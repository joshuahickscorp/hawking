# DELEGATION — RUNNER GRAVITY/NX MODES (gate profile: MLX/Metal, runner ONLY)

Add MODEST Gravity + NX-attempt modes to `tools/odyssey_patient_runner.py` so each
patient can produce ≥1 real Gravity artifact + ≥1 NX/execution attempt (steer S002
§19/§20 — bounded, SPECIMEN-labeled, defer polishing to Odyssey-II). Edit ONLY
`tools/odyssey_patient_runner.py` (+ receipts + the target packet). Do NOT touch
`tools/odyssey_ctl.py` (a parallel lane owns it). Repo
`/Users/scammermike/Downloads/hawking`. Branch odyssey-i.

## Read first
- `tools/odyssey_patient_runner.py` — has external-science / sensitivity / --skip-route already. Extend it; keep those working.
- `receipts/odyssey-i/<OXX>_EXTERNAL.json` — baseline (battery, tps_specimen) to compare against.
- `workspace/campaign/odyssey/patients/<OXX>/census.json` — params/organ bytes.

## SETUP
mlx + mlx_lm already installed on Framework 3.12; the runner re-execs there. Honor the worker_gate + 4-bit fallback already in the runner.

## BUILD — new flags (exact names; the cycling lane references these)
1. `--gravity <spec>` — build a MODEST candidate representation with mlx quantization and grade it:
   - specs: `q3-g32-experts` (MoE: experts→3-bit group32, attention/router→4-bit group64, norms full), `q4-g64` (uniform 4-bit group64), `q4-g64-attn-mlp` (hybrid: attn+mlp→4-bit, protect SSM/conv/norm full). Use mlx_lm quantization with a per-module quant_predicate to realize the mix.
   - reload the quantized model, run the SAME fast-Doctor battery + refusal controls, measure: `stored_bytes`, `stored_bpw` (bytes*8/params), `active_bytes_per_token` + `active_bpw` (using census active-param split for MoE), and battery/refusal DELTA vs `<OXX>_EXTERNAL.json` baseline.
   - emit `receipts/odyssey-i/<OXX>_GRAVITY_<spec>.json` (schema `odyssey.patient.gravity.v1`): spec, stored/active bpw, battery, delta_hits, seal via doctor_seal, SPECIMEN + quant caveat, verdict {CANDIDATE_PASS if delta_hits>=-1 else DEGRADED}. Never call it a Hawking NX win — mlx SPECIMEN only (§15).
2. `--nx-gather` (MoE only) — active-expert-gather accounting (§13): from the router over N tokens, compute THEORETICAL selected-expert bytes/token = topk/n_experts × expert_body_bytes; contrast with full-expert-body bytes and the dense-MLP-equivalent bytes; report the ratio (the NX opportunity). Note whether mlx actually gathers or densely computes (inspect the switch_mlp path). Emit `receipts/odyssey-i/<OXX>_NX_gather.json` (schema `odyssey.patient.nx.v1`, DERIVED/MEASURED labels). This is an ACCOUNTING + minimal-primitive-design attempt, not a full Rust runtime (§14).
3. `--nx-state` (hybrid) — reuse the SSM-vs-KV accounting already computed; emit `<OXX>_NX_state.json` framing the fixed-state residency as the NX lever.
4. `--nx-dense` (dense) — report full-weight-sweep bytes/token as the dense NX floor + note no sparsity lever; emit `<OXX>_NX_dense.json`.

## Constraints
Bounded: one candidate spec per gravity call; do not sweep. Keep external-science/sensitivity modes unchanged. Do not delete canonical weights. Every number §18-labeled + SPECIMEN where quantized. No commit/push. Runner file only.

## ACCEPTANCE
- `python3 tools/odyssey_patient_runner.py --oxx O005 --weights <O005 snapshot> --gravity q3-g32-experts --out receipts/odyssey-i/O005_GRAVITY_q3-g32-experts.json` produces a valid gravity receipt with stored_bpw < 16, active_bpw, battery, delta vs baseline, exit 0.
- `python3 tools/odyssey_patient_runner.py --oxx O005 --weights <snapshot> --nx-gather --out receipts/odyssey-i/O005_NX_gather.json` produces the theoretical-vs-full selected-byte ratio, exit 0.
- existing `--route-tokens`/`--sensitivity`/`--skip-route` paths still work (spot-run one).

## SCOPE
WRITE tools/odyssey_patient_runner.py
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/patients/
READ receipts/odyssey-i/, workspace/campaign/odyssey/patients/, tools/worker_gate.py, tools/doctor_seal.py
VERIFY tools/odyssey_patient_runner.py by running the two ACCEPTANCE commands on O005 — must produce valid receipts, exit 0.
