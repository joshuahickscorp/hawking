# DELEGATION — O006 FRONTIER NOVELTY / arch

Patient O006 (moe; Qwen3VLMoeForConditionalGeneration).
Repo: `/Users/scammermike/Downloads/hawking`. Branch odyssey-i.
Grok novelty lane. Hypotheses only; scripts measure (bible §55 / §13).
Opus is not involved. Do not launch further Grok from this lane.

This patient stalled on conventional / aggressive-quant families with a
LARGE remaining target delta (primary=1.145, pressure=2.5 bpw,
stored_bpw=3.9555, active_bpw=3.645). Best conventional
anchor: spec=q3-g32-experts class=CONVENTIONAL_ANCHOR
complete_bpw=3.9556.
Aggressive failures: q2-g32-experts, q3-g32-experts. Localization: [{"failed": true, "delta_hits": -2, "threshold": -1, "ranked_organs": [{"organ": "attn", "sensitivity_delta": 0, "treatment": "round8"}, {"organ": "embed", "sensitivity_delta": 0, "treatment": "round8"}, {"organ": "expert", "sensitivity_delta": 0, "treatment": "round8"}, {"organ": "lm_head", "sensitivity_delta": 0, "treatment": "round8"}, {"organ": "mlp_dense", "sensitivity_delta": 0, "treatment":.
Native / NX so far: unspecified.

## Read first
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json
READ workspace/campaign/odyssey/GRAVITY_RULEBASE.json
READ workspace/campaign/odyssey/NEGATIVE_SCIENCE.json
READ workspace/campaign/odyssey/TRANSFER_MATRIX.json

Query NEGATIVE_SCIENCE before every expensive hypothesis (bible §19).
A kill stays dead unless this architecture invalidates the old premise.

## QUESTION
What architecture-conditioned lever (stored vs active vs state residency vs per-modality organs) is the real remaining delta, and which NONCONVENTIONAL mechanism attacks that axis?

## SEARCH SPACE (nonconventional only)
MoE: selected/full is a SELECTION opportunity, not 1/16 cost; native expert-gather; route-conditioned repr; no cold-expert assumption until routing is measured. Hybrid: SSM state vs KV. Multimodal: per-modality organs. MTP: tokens/expensive-traversal.

Live negatives that already constrain this lane:
NS-inter-expert-redundancy, NS-expert-merging-omitted-from-survivors, NS-cross-expert-and-cross-layer-tying, NS-global-dense-lowrank-qwen38 (does NOT auto-kill MoE shared structure)

Presumptively CONVENTIONAL (do not re-propose as a frontier):
uniform-quant, affine-quant, symmetric-quant, per-group-scale-bias,
ordinary-mixed-precision, gguf-mlx-like-per-weight, global integer
bit-width retreat. CONVENTIONAL_ANCHOR != REPRESENTATION_FRONTIER.

## BUILD
Write `receipts/odyssey-i/O006_NOVELTY_arch.json` (schema hawking.odyssey.novelty_lane_report.v1) with one or more
proposals. Each proposal REQUIRES these keys: mechanism, complete_byte_accounting, cheapest_falsifier, execution_path, kernel_implications, applicability_class, doctor_risk.

Also required on each proposal:
- family_addition: a candidate_families row (mechanism, conventionality=STRUCTURAL,
  cheapest_falsifier, expected_win, doctor_risk, applicability)
- runner_spec if the hypothesis can be expressed in the runner grammar
  (q<bits>-g<group>-experts, mixed-q2q3-experts, +correction, tiers)
- info_gain (0-10) and cost (wall/gpu relative 1-10)
- reopen_if, evidence class (HYPOTHESIS), NEXT_BOTTLENECK

conventionality of a novelty proposal is STRUCTURAL (or ACTIVE_NX), never
CONVENTIONAL_ANCHOR. Complete byte accounting must name payload+scales+
biases+tables+offsets+correction+tier/router-metadata+alignment+headers+
repr-attributable-state (policy.accounting_gates.no_fake_density).

Do not edit tools/odyssey_ctl.py or tools/odyssey_patient_runner.py.
Do not claim a Hawking NX win. Do not measure TPS (physics is programmatic).

## ACCEPTANCE
- `receipts/odyssey-i/O006_NOVELTY_arch.json` exists, schema hawking.odyssey.novelty_lane_report.v1, each proposal has
  mechanism, complete_byte_accounting, cheapest_falsifier, execution_path, kernel_implications, applicability_class, doctor_risk. Must pass, exit 0.
- Zero conventional-family re-proposals tagged as FRONTIER.

## SCOPE
WRITE workspace/campaign/odyssey/contracts/auto/
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/candidate_families.json
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json, workspace/campaign/odyssey/GRAVITY_RULEBASE.json, workspace/campaign/odyssey/NEGATIVE_SCIENCE.json, workspace/campaign/odyssey/TRANSFER_MATRIX.json
VERIFY receipts/odyssey-i/ by running the unfenced command below; must pass, exit 0.
python3 -c "import json,pathlib,sys; p=pathlib.Path('receipts/odyssey-i/O006_NOVELTY_arch.json'); d=json.loads(p.read_text()); ps=d.get('proposals') if isinstance(d.get('proposals'), list) else [d]; req=('mechanism','complete_byte_accounting','cheapest_falsifier','execution_path','kernel_implications','applicability_class','doctor_risk'); missing=[k for pr in ps if isinstance(pr, dict) for k in req if k not in pr]; sys.exit(0 if p.is_file() and ps and not missing else 1)"
Do not modify tools/odyssey_ctl.py
Do not modify tools/odyssey_patient_runner.py
Do not touch Genesis state or tools/odyssey/.
