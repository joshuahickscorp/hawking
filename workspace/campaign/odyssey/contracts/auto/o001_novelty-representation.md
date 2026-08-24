# DELEGATION — O001 FRONTIER NOVELTY / representation

Patient O001 (hybrid; FalconH1ForCausalLM).
Repo: `/Users/scammermike/Downloads/hawking`. Branch odyssey-i.
Grok novelty lane. Hypotheses only; scripts measure (bible §55 / §13).
Opus is not involved. Do not launch further Grok from this lane.

This patient stalled on conventional / aggressive-quant families with a
LARGE remaining target delta (primary=2.7611, pressure=2.5 bpw,
stored_bpw=5.2611, active_bpw=5.2611). Best conventional
anchor: spec=q4-g64-attn-mlp class=CONVENTIONAL_ANCHOR
complete_bpw=6.5002.
Aggressive failures: q2-g32-attn-mlp. Localization: [{"failed": true, "delta_hits": -10, "threshold": -1, "ranked_organs": [{"organ": "embed", "sensitivity_delta": -2, "treatment": "round8"}, {"organ": "lm_head", "sensitivity_delta": -1, "treatment": "round8"}, {"organ": "mlp_dense", "sensitivity_delta": -1, "treatment": "round8"}, {"organ": "attn", "sensitivity_delta": 0, "treatment": "round8"}, {"organ": "norm", "sensitivity_delta": 0, "treatment.
Native / NX so far: fixed-state residency (SSM vs KV; accounting only).

## Read first
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json
READ workspace/campaign/odyssey/GRAVITY_RULEBASE.json
READ workspace/campaign/odyssey/NEGATIVE_SCIENCE.json
READ workspace/campaign/odyssey/TRANSFER_MATRIX.json

Query NEGATIVE_SCIENCE before every expensive hypothesis (bible §19).
A kill stays dead unless this architecture invalidates the old premise.

## QUESTION
Which NONCONVENTIONAL representation (not uniform/affine/symmetric/per-group-scale/ordinary mixed-precision) can close the remaining stored/active target delta on this patient?

## SEARCH SPACE (nonconventional only)
base+correction (sparse/residual/selected-hi-prec-channels/expert- or route-conditioned/procedural), Matryoshka T0..T3, per-organ or per-expert codecs, layer-0-as-different-source, source-changing methods (not raw-weight PQ/VQ at ~1 bit).

Live negatives that already constrain this lane:
NS-raw-weight-pq-vq-at-one-bit, NS-uniform-subbit-allocation, NS-kronecker-factorisation (depth; layer-0 is the named exception), NS-post-hoc-coding-of-frozen-weights

Presumptively CONVENTIONAL (do not re-propose as a frontier):
uniform-quant, affine-quant, symmetric-quant, per-group-scale-bias,
ordinary-mixed-precision, gguf-mlx-like-per-weight, global integer
bit-width retreat. CONVENTIONAL_ANCHOR != REPRESENTATION_FRONTIER.

## BUILD
Write `receipts/odyssey-i/O001_NOVELTY_representation.json` (schema hawking.odyssey.novelty_lane_report.v1) with one or more
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
- `receipts/odyssey-i/O001_NOVELTY_representation.json` exists, schema hawking.odyssey.novelty_lane_report.v1, each proposal has
  mechanism, complete_byte_accounting, cheapest_falsifier, execution_path, kernel_implications, applicability_class, doctor_risk. Must pass, exit 0.
- Zero conventional-family re-proposals tagged as FRONTIER.

## SCOPE
WRITE workspace/campaign/odyssey/contracts/auto/
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/candidate_families.json
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json, workspace/campaign/odyssey/GRAVITY_RULEBASE.json, workspace/campaign/odyssey/NEGATIVE_SCIENCE.json, workspace/campaign/odyssey/TRANSFER_MATRIX.json
VERIFY receipts/odyssey-i/ by running the unfenced command below; must pass, exit 0.
python3 -c "import json,pathlib,sys; p=pathlib.Path('receipts/odyssey-i/O001_NOVELTY_representation.json'); d=json.loads(p.read_text()); ps=d.get('proposals') if isinstance(d.get('proposals'), list) else [d]; req=('mechanism','complete_byte_accounting','cheapest_falsifier','execution_path','kernel_implications','applicability_class','doctor_risk'); missing=[k for pr in ps if isinstance(pr, dict) for k in req if k not in pr]; sys.exit(0 if p.is_file() and ps and not missing else 1)"
Do not modify tools/odyssey_ctl.py
Do not modify tools/odyssey_patient_runner.py
Do not touch Genesis state or tools/odyssey/.
