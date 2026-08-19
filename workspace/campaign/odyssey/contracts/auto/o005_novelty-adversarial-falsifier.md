# DELEGATION — O005 FRONTIER NOVELTY / adversarial-falsifier

Patient O005 (moe; Qwen3MoeForCausalLM).
Repo: `/Users/scammermike/Downloads/hawking`. Branch odyssey-i.
Grok novelty lane. Hypotheses only; scripts measure (bible §55 / §13).
Opus is not involved. Do not launch further Grok from this lane.

This patient stalled on conventional / aggressive-quant families with a
LARGE remaining target delta (primary=1.7305, pressure=2.5 bpw,
stored_bpw=4.0253, active_bpw=4.2305). Best conventional
anchor: spec=q3-g32-experts class=CONVENTIONAL_ANCHOR
complete_bpw=4.0253.
Aggressive failures: q2-g32-experts. Localization: [{"organ": "gate", "repair": "protect gate/up"}, {"organ": "router", "repair": "protect router precision"}].
Native / NX so far: active-expert-gather.

## Read first
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json
READ workspace/campaign/odyssey/GRAVITY_RULEBASE.json
READ workspace/campaign/odyssey/NEGATIVE_SCIENCE.json
READ workspace/campaign/odyssey/TRANSFER_MATRIX.json

Query NEGATIVE_SCIENCE before every expensive hypothesis (bible §19).
A kill stays dead unless this architecture invalidates the old premise.

## QUESTION
For every live conventional/aggressive point and every novelty hypothesis on this packet: what is the cheapest falsifier, which negative-science entry already kills it, and where is the Goodhart hole?

## SEARCH SPACE (nonconventional only)
Kill first. Scope + evidence + arch_assumptions + reopen_if on every kill (policy.negative_science). Predicates over blacklists. Anti-Goodhart: complete_bpw, held-out Doctor pattern, info-count = novelty of evidence not receipt count.

Live negatives that already constrain this lane:
entire NEGATIVE_SCIENCE.json; also policy.accounting_gates (no_fake_density / no_fake_active_density / no_fake_tps)

Presumptively CONVENTIONAL (do not re-propose as a frontier):
uniform-quant, affine-quant, symmetric-quant, per-group-scale-bias,
ordinary-mixed-precision, gguf-mlx-like-per-weight, global integer
bit-width retreat. CONVENTIONAL_ANCHOR != REPRESENTATION_FRONTIER.

## BUILD
Write `receipts/odyssey-i/O005_NOVELTY_adversarial-falsifier.json` (schema hawking.odyssey.novelty_lane_report.v1) with one or more
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
- `receipts/odyssey-i/O005_NOVELTY_adversarial-falsifier.json` exists, schema hawking.odyssey.novelty_lane_report.v1, each proposal has
  mechanism, complete_byte_accounting, cheapest_falsifier, execution_path, kernel_implications, applicability_class, doctor_risk. Must pass, exit 0.
- Zero conventional-family re-proposals tagged as FRONTIER.

## SCOPE
WRITE workspace/campaign/odyssey/contracts/auto/
WRITE receipts/odyssey-i/
WRITE workspace/campaign/odyssey/candidate_families.json
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json, workspace/campaign/odyssey/GRAVITY_RULEBASE.json, workspace/campaign/odyssey/NEGATIVE_SCIENCE.json, workspace/campaign/odyssey/TRANSFER_MATRIX.json
VERIFY receipts/odyssey-i/ by running the unfenced command below; must pass, exit 0.
python3 -c "import json,pathlib,sys; p=pathlib.Path('receipts/odyssey-i/O005_NOVELTY_adversarial-falsifier.json'); d=json.loads(p.read_text()); ps=d.get('proposals') if isinstance(d.get('proposals'), list) else [d]; req=('mechanism','complete_byte_accounting','cheapest_falsifier','execution_path','kernel_implications','applicability_class','doctor_risk'); missing=[k for pr in ps if isinstance(pr, dict) for k in req if k not in pr]; sys.exit(0 if p.is_file() and ps and not missing else 1)"
Do not modify tools/odyssey_ctl.py
Do not modify tools/odyssey_patient_runner.py
Do not touch Genesis state or tools/odyssey/.
