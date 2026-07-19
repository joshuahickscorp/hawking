# HIDE Claude Code Parity and Supremacy - Research Package

Run date: 2026-07-19
Research cutoff: 2026-07-19
Repository truth pinned at: `4fbca8bc` (branch `main`), verified live at run start
Research branch: `research/hide-claude-parity-2026-07-19` (isolated worktree; no merge to main)
Authoritative spec: `HIDE_CLAUDE_CODE_PARITY_SUPREMACY_BIBLE.md`
Status: sealed and adversarially verified. Start with `HIDE_RESEARCH_FINAL_REPORT.md`.

Input dossiers challenged (not merely summarized):
- `docs/plans/hawking_ide_frontier_2026_07_19.md` (first pass)
- `docs/plans/hawking_ide_claude_research_handoff_2026_07_19.md` (handoff)

## Purpose

Clean-room replication of the Claude Code workflow developers love, fused with a
local Hawking-native IDE and agent operating system. Research first. No product
features implemented in this pass.

The package answers three questions with evidence:

1. What is the current, verified live truth of HIDE and the Hawking runtime?
2. What exactly makes Claude Code loved, and what is the behavioral contract HIDE
   must reproduce to feel immediately familiar and at least as polished?
3. Which Hawking-native mechanisms make HIDE structurally better, and by what
   measurable path?

## Non-interference

This pass is read-only against all live sources, controllers, checkpoints,
providers, and leases. Writes occur only on the research branch inside an isolated
git worktree. The former `hawking-packs` runtime (absorbed at `0adcab57`, retired
at `4fbca8bc`) is not recreated. The Frontier (Generation M / G3 line at
`c9280be3` or a later successor) is not touched.

## Evidence legend

| Label | Meaning |
|---|---|
| DOCUMENTED | Stated in official product documentation, spec, or changelog. |
| OBSERVED | Seen in a credible trace, demo, or recording. |
| MEASURED | A number reported with a stated method / harness. |
| INFERRED | Reasoned from evidence; not directly stated. |
| ANECDOTAL | Community report (forum, blog, video); a lead, not proof. |
| UNKNOWN | Not yet determined; flagged for a probe. |

Repository claims additionally use VERIFIED REPO (confirmed in the active tree or
git history), with component status ACTIVE / ACTIVE_UNEXPOSED / PACKED / UI_ONLY /
PARTIAL / STUB / PROPOSED / MISSING.

## Method

- Two background research fleets: a read-only live-code audit (6 agents) and a
  Claude Code clean-room study (11 scouts, each independently adversarially
  verified). A spec-drafting fleet (13 agents) then produced the self-contained
  specs grounded in those artifacts.
- A final adversarial pass (5 verifiers) falsified the parity spec, the supremacy
  thesis, the gap matrix, and the handoff falsification targets, and re-verified
  the archaeology against live code before anything was sealed. Findings were
  reconciled; see `HIDE_RESEARCH_FINAL_REPORT.md` section 6.
- Clean-room boundary: behavioral contracts, workflow patterns, configuration
  shapes, and short functional labels only. No proprietary source, assets, or
  branding. House rule enforced: no em or en dashes (0 across the package).

## Artifact manifest (Bible section 93) - all sealed

- [x] HIDE_LIVE_ARCHAEOLOGY.md / .json - verified map of the live implementation
- [x] HIDE_CLAUDE_CODE_UX_GENOME.md - the loved-experience genome
- [x] HIDE_CLAUDE_CODE_WORKFLOW_TRACES.jsonl - step-by-step lived-sequence traces
- [x] HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json - the parity contract (32 entries)
- [x] HIDE_CLAUDE_CODE_CONFIGURATION_COMPATIBILITY.md - migration compatibility
- [x] HIDE_USER_LOVE_PAIN_MAP.md - love, tolerance, captivity, and the wedge
- [x] HIDE_2026_COMPETITOR_MATRIX.json - full competitor matrix, Claude Code deepest
- [x] HIDE_TWO_SURFACE_ARCHITECTURE.md - Chat + IDE as two views of one session
- [x] HIDE_CHAT_SPEC.md / HIDE_IDE_SPEC.md - the two surfaces
- [x] HIDE_AGENT_KERNEL_OPTIONS.md - inner loop + outer scheduler options
- [x] HIDE_CONTEXT_OS_SPEC.md / HIDE_MEMORY_SPEC.md - context OS + outcome memory
- [x] HIDE_TOOL_SKILL_PLUGIN_MCP_ABI.md - tool/skill/plugin/MCP ABI
- [x] HIDE_PERMISSION_AND_EFFECT_SYSTEM.md / HIDE_SECURITY_CONSTITUTION.md
- [x] HIDE_DURABLE_AGENT_SPEC.md - durable, proactive agents
- [x] HIDE_STATE_CAPSULE_ABI.md / HIDE_LOCAL_MODEL_TOPOLOGY.md - Hawking-native core
- [x] HIDE_SPEED_FRONTIER.md - latency budget + prefill elimination
- [x] HIDE_CAPABILITY_DENSITY_EVAL.md / HIDE_BLIND_PREFERENCE_STUDY.md
- [x] HIDE_PARITY_GAP_MATRIX.json - current-vs-target, classified
- [x] HIDE_SUPREMACY_THESIS.md - evidence-backed supremacy path
- [x] HIDE_PRIORITIZED_BUILD_LADDER.md - priority-formula-ranked ladder
- [x] HIDE_MINIMUM_LOVABLE_VERTICAL.md / HIDE_SIGNATURE_DEMO.md
- [x] HIDE_EXPERIMENT_MENU.md - isolated bets with kill criteria
- [x] HIDE_RESEARCH_RECEIPT.json / HIDE_RESEARCH_FINAL_REPORT.md - provenance + verdict

## Provenance

- Live-audit fleet: `wf_3a62e82a-5c8` (6 agents, 0 errors)
- Clean-room fleet: `wf_5c04451d-beb` (22 agents, 0 errors)
- Spec-drafting fleet: `wf_98a2bdf4-216` (13 agents, 0 errors)
- Adversarial-verify fleet: `wf_74fdbb5b-b03` (5 agents, 0 errors)
