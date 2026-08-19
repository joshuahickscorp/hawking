# DELEGATION — G2 INTEGRATION: wire modules into the loop (ctl.py + runner grammar)

The super-detachment modules exist standalone + self-check green: `tools/odyssey_memgate.py`
(multi-model swap<=30), `tools/odyssey_candgen.py` + `candidate_families.json` (search-as-data),
`ODYSSEY_MANIFEST.json` (predeclared patients), `tools/odyssey_costmodel.py`, `tools/odyssey_novelty.py`.
Wire them into `tools/odyssey_ctl.py` so the autonomous loop USES them. Repo
`/Users/scammermike/Downloads/hawking`, py `/Library/Frameworks/.../3.12/bin/python3`. Branch odyssey-i.
Driver is PAUSED; no concurrent control-plane writer.

## Read first
`tools/odyssey_ctl.py` (cycle/run/evaluate_gates/select/retire/acquire-next/harvest + templates +
write_set), `ODYSSEY_POLICY.json` (detachment.memory: swap_max_gib 30, multi_model, clean_room_only_for;
gravity_specs table), the 5 modules' public APIs, `tools/odyssey_patient_runner.py` `--gravity` grammar.

## WIRE
1. **Multi-model admission (user directive + memgate)**: replace the model-lane admission in the cycle
   with `odyssey_memgate.admit(est_gib, in_flight_gib, clean_room)`. Launch MULTIPLE model lanes
   concurrently when projected swap <= 30 GiB AND their write_sets are disjoint. `clean_room=True` ONLY
   for protected-timing (TPS/TOKEN_NS) lanes -> exclusive. Track in-flight model memory across the
   admitted set. Remove the old one-at-a-time model serialization. Keep the ODYSSEY_HEADROOM_ADMIT opt-in
   as a fallback but prefer memgate.
2. **write_set fix (parallelize data lanes)**: data-producing templates (external-science-*, route-map,
   sensitivity-map, gravity-*, gravity-aggressive-*, nx-*, transfer-control) write ONLY
   {patient packet, that patient's receipts} — DROP the runner from their write_set (the modes exist;
   harvest already ignores incidental runner tweaks). Different patients -> disjoint -> parallel. Only a
   genuinely-runner-building obligation (rare) claims the runner.
3. **candgen grammar reconcile**: make `classify_gravity_spec` GRAMMAR-BASED — parse `q<bits>-g<group>-experts`
   / `mixed-q2q3...` / `...+correction` / tier specs to derive candidate_class (q>=3 affine=CONVENTIONAL_ANCHOR,
   q<=2=AGGRESSIVE_QUANT, mixed/correction/tier=STRUCTURAL_GRAVITY) with the policy.gravity_specs table as
   override. The `gravity-aggressive-<class>` obligation calls `odyssey_candgen.generate(...)` to pick the
   next-highest-EV aggressive spec; ensure the runner `--gravity` accepts any generated spec (add a generic
   `q<b>-g<g>-experts` parse path if absent). Do NOT require every grid spec pre-listed.
4. **manifest drives acquire + targets**: `acquire-next` reads `ODYSSEY_MANIFEST.json` for the next patient
   (canonical_source/est size/gated/reference_sibling); per-patient targets + search_class + info_budget +
   arch_objective come from the manifest, not hardcode.
5. **costmodel record**: `cycle`/`harvest` call `odyssey_costmodel.record(...)` per event (wall, grok lane,
   opus flag); expose `python3 tools/odyssey_ctl.py economics` printing derive()/detachment_metrics().
6. **novelty escalate**: after a patient's aggressive families converge to conventional/aggressive with a
   large target delta and deterministic search is exhausted, `cycle` calls `odyssey_novelty.should_escalate`
   -> `build_packet` -> `render_lane_contracts` and admits those grok lanes (gated like any lane).

## Constraints
Reuse modules (import; don't reimplement). Keep all existing self-check assertions + the anti-complacency
gate green. Deterministic; no per-candidate model reasoning. No commit/push/launchd.

## ACCEPTANCE
- `python3 tools/odyssey_ctl.py --self-check` passes, incl new checks: memgate admits >=2 disjoint model
  lanes under low swap; a data-producing template's write_set excludes the runner; a candgen spec classifies
  correctly via grammar; acquire-next reads the manifest.
- `python3 tools/odyssey_ctl.py cycle --dry-run` (ODYSSEY_HEADROOM_ADMIT=1) plans MULTIPLE concurrent model
  lanes (disjoint patients) instead of serializing them, and shows gravity-aggressive obligations pending
  for MoE patients; no sealed obligation reappears.
- `python3 tools/odyssey_ctl.py economics` runs, exit 0.

## SCOPE
WRITE tools/odyssey_ctl.py
WRITE tools/odyssey_patient_runner.py
WRITE workspace/campaign/odyssey/
READ tools/odyssey_memgate.py, tools/odyssey_candgen.py, tools/odyssey_costmodel.py, tools/odyssey_novelty.py, workspace/campaign/odyssey/ODYSSEY_POLICY.json, workspace/campaign/odyssey/ODYSSEY_MANIFEST.json, workspace/campaign/odyssey/candidate_families.json
VERIFY tools/odyssey_ctl.py by running `python3 tools/odyssey_ctl.py --self-check` and `python3 tools/odyssey_ctl.py cycle --dry-run` — must pass, exit 0.
