# Hawking Ship Goal Prompts

These are intentionally thin. The real prompt is
`docs/campaign/hawking_ship_finalization_prompt.md`; each goal should make the
agent read that MD first instead of duplicating stale state inline.

## Overall Finalization

````text
/goal Hawking ship finalization in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md` in full first, then
execute the highest-leverage open lane toward shipping Hawking. Use
`docs/plans/hawking_shippability_masterplan_2026_06_22.md` as the strategic
source of truth, keep all work gated, update the campaign standing docs with
evidence, and stop only for owner decisions, destructive changes, external
spending, or risky output/default changes that lack gates.
````

## Phase A - Runtime GA

````text
/goal Hawking Phase A Runtime GA in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md` in full first, then
execute Lane A. Prioritize RWKV serve throughput, active-slot multiseq sizing,
request isolation, valid chat-templated quality eval, and automated
speed/quality/footprint gates. Preserve RWKV parity and serve correctness.
````

## Phase B - Model Press

````text
/goal Hawking Phase B Model Press in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md` in full first, then
execute Lane B. Prioritize un-stubbing bake/press, finishing TQ GPU parity,
wiring only gated default-off compression levers, adding the memory-budgeted
out-of-core press path, recording full recipes in the artifact, and producing
quality-carded reproducible compressed outputs across the 4/3/2/1-bit ladder.
````

## Phase C - Distillation Product

````text
/goal Hawking Phase C distillation product in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md` in full first, then
execute Lane C. Build toward a reproducible compress-then-distill pipeline with
eval-in-loop, tracked recipes, quant-aware KD, and a quality-carded flagship
student. Treat 2-bit, 1-bit, and ternary quality recovery as QAT/KD problems,
not merely post-hoc quantization. Do not spend cloud credits or start long paid
training without owner approval.
````

## Phase D - Hawking Lab

````text
/goal Hawking Phase D Hugging Face Lab launch in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md` in full first, then
execute Lane D. Prepare the HF release system: naming/versioning, model-card
templates, reproducible recipes, quality/compression leaderboard, demo plan, and
release automation. Do not publish publicly or pick final launch SKUs without
owner approval.
````

## Format And Headless App

````text
/goal Hawking format and headless app finalization in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md` in full first, then
execute Lane E. Work toward content-based artifact detection, a versioned format
spec, `hawkingd`, model registry/config, and HF pull/install flow. Stop for owner
approval before finalizing the artifact identity.
````

## Spec Decode Resolution

````text
/goal Hawking spec-decode resolution in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md` in full first, then
execute Lane F. Either prove a named regime where spec-decode wins with warm
measurements and lossless gates, or retire the speed claim and park/prune only
with attended review. Do not ship speculative decode as a single-stream speed
claim.
````

## Condense Frontier

````text
/goal Hawking general Condense pass in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md`,
`docs/plans/condense_frontier_2026_06_22.md`, and
`docs/plans/condense_naming_migration_2026_06_22.md` in full first, then execute
Lane G only after checking whether the core finalization gates are safe to
leave.

This is not just a naming cleanup. Treat Condense as a general program-wide
compression and recovery push: make Hawking able to create verified low-bit
artifacts from parent models that ordinary users could not previously quantize
on the same machine. Prioritize the sector wedge: memory-budgeted out-of-core
artifact creation, stronger 4/3/2/1-bit recipes, output-damage-ranked bit
allocation, QAT/KD/distillation recovery, activation correction, selective
protection, and quality-carded comparisons against normal quantization
baselines at iso-bpw.

The goal is a Model Press that can truthfully claim one or more of these gains:
lower peak creation memory than common tools, lower bpw at comparable quality,
higher retained quality at the same bpw, or a successful artifact from a parent
that could not fit fully resident during quantization. Use Condense as the
public name for the legacy STRAND low-bit line; prefer plain names like Press
Plan, Artifact Manifest, Quality Card, and Condense Ladder over forced
black-hole terminology. Start with dry-run planning and small-model proofs; do
not download frontier-scale weights, spend cloud credits, or publish derivatives
without owner approval.
````

## Condense Naming Cleanup

````text
/goal Hawking Condense naming cleanup in /Users/scammermike/Downloads/hawking.

Read `docs/plans/condense_naming_migration_2026_06_22.md`,
`docs/campaign/hawking_ship_finalization_prompt.md`, and
`docs/plans/condense_frontier_2026_06_22.md` first. Normalize the
public compression naming around Hawking, Model Press, and Condense. Prefer
plain names: Condense Planner, Press Plan, Artifact Manifest, Quality Card, and
Condense Ladder. Treat STRAND/strand as legacy/internal and Dismantle as
historical provenance only. Start with docs and user-facing prompts; if editing
code, add compatibility aliases instead of breaking existing `strand_*`
functions, env vars, crates, scripts, or `.strand`/STR2 artifacts. Keep changes
narrow, run `git diff --check`, and do not rename wire formats or public
artifacts without an explicit compatibility gate.
````

## Apple Fit Frontier

````text
/goal Hawking Apple Fit frontier in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md`,
`docs/plans/apple_fit_frontier_2026_06_22.md`,
`docs/plans/hawking_shippability_masterplan_2026_06_22.md`,
`docs/env_flags.md`, and `docs/architecture.md` first. Execute Lane H only
after checking whether the active finalization and Condense gates are safe to
leave.

Design and implement the Apple-Silicon-specific moat: Hawking should inspect
the current Mac, predict fit, avoid memory-pressure failure, choose the best
model/quant/context/KV policy, and produce measured speed, quality, footprint,
and energy guidance. This must be a capability amplifier, not a throttle:
`hawking fit`, `hawking doctor`, and `hawking serve --auto` must expose the
usable envelope, choose the strongest stable configuration for the declared
intent, show stronger and safer alternatives, and allow expert override.

Do not silently downgrade tps, context, precision, batch, model choice, or
quality to make auto mode look stable. Any downgrade must be explicit,
reversible when pressure clears, and justified by user intent or hard resource
pressure. Add anti-throttle gates so auto-selected runs cannot materially lose
speed, quality, or context versus the best known manual profile without a
stated constraint.
````

## Final Standing Review

````text
/goal Hawking final standing review in /Users/scammermike/Downloads/hawking.

Read `docs/campaign/hawking_ship_finalization_prompt.md` in full first, then
produce a concise current standing review from the campaign docs and command
evidence. Do not make code changes unless the review uncovers a tiny doc
correction needed to prevent stale state.
````
