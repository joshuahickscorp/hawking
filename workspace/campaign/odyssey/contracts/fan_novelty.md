# DELEGATION — GROK NOVELTY ENGINE (NEW tools/odyssey_novelty.py, module-only)
When the deterministic engine converges to conventional families with a large target delta, emit a
FRONTIER_NOVELTY_PACKET and fan out independent Grok lanes for NONCONVENTIONAL mechanisms (steer S004 §19/§54/§35/§36).
Repo /Users/scammermike/Downloads/hawking. Branch odyssey-i. Do NOT edit ctl.py/runner.

## BUILD `tools/odyssey_novelty.py`
- `should_escalate(patient, best_class, target_delta, families_tried) -> bool`: true when all tried families are CONVENTIONAL/AGGRESSIVE_QUANT (no STRUCTURAL survivor) AND target_delta large AND deterministic search exhausted (per ODYSSEY_POLICY.escalation_order_on_stall: deterministic -> rule-transfer -> grok-novelty).
- `build_packet(patient, packet, receipts, rulebase, transfer, negatives) -> dict` FRONTIER_NOVELTY_PACKET: arch, best conventional anchor, aggressive failures + localization, stored/active/physical decomposition, native primitives, negative rules, remaining target delta.
- `render_lane_contracts(packet) -> [contract_path...]`: writes SG-valid grok contracts under workspace/campaign/odyssey/contracts/auto/ for lanes {representation, numerical, arch, kernel, adversarial-falsifier, compression}; each REQUIRES proposals specify mechanism + complete byte accounting + cheapest falsifier + execution path + kernel implications + applicability class + Doctor risk.
- `harvest_proposals(task_ids) -> [proposal...]` dedup + rank by info-gain/cost; turn each into a deterministic experiment contract (a candidate_families addition or a runner spec). Does NOT itself launch grok (the driver/ctl launches); it prepares contracts + parses reports.
Deterministic; opus not involved. Grok does hypotheses, scripts measure (§55).

## Self-check (exit 0): should_escalate true/false on synthetic inputs; build_packet returns all required keys; render writes >=3 SG-valid contracts (each has WRITE + VERIFY unfenced lines).
## SCOPE
WRITE tools/odyssey_novelty.py
WRITE workspace/campaign/odyssey/contracts/auto/
READ workspace/campaign/odyssey/ODYSSEY_POLICY.json, workspace/campaign/odyssey/GRAVITY_RULEBASE.json, workspace/campaign/odyssey/NEGATIVE_SCIENCE.json
VERIFY tools/odyssey_novelty.py by running `python3 tools/odyssey_novelty.py --self-check` — exit 0.
