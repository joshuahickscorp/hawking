# Hawking — Pruning / Condensation Inventory (2026-06-22)

Candidates for deletion, condensation, or archival. **No code is deleted in this pass.** Classes:
**SAFE-DOC-CLEANUP** (low-risk doc move/mark) · **NEEDS-ATTENDED-REVIEW** (judgment + owner) · **DO-NOT-DELETE** (proof shows it's load-bearing).

## SAFE-DOC-CLEANUP (do-able now or trivially)
| Item | Action | Evidence |
|---|---|---|
| `docs/campaign/claude_goal_prompt.md`, `claude_moonshot_goal_prompt.md`, `claude_final_hardening_condensation_goal_prompt.md` | Mark **SUPERSEDED** by `hawking_ship_finalization_prompt.md` + `hawking_ship_goal_prompts.md`; keep for provenance | The new prompt pair is the canonical ship-finalization surface. Don't delete the old prompts (cheap, useful history). |
| Three `claude_*goal_prompt.md` files | Optionally archive under `docs/campaign/archive/` after the new prompt pair is adopted | They're handoff prompts, not standing docs. Condensation only. |
| `docs/campaign/change_manifest.md` | Fold into `commit_plan.md` (now the canonical lane split) | `commit_plan.md` supersedes it; mark change_manifest superseded. |
| `docs/plans/throughput_pivot_campaign.md` | Mark **SUPERSEDED** by `docs/campaign/` artifacts (findings/roadmap/kill_ledger) | The campaign/ set is the consolidated source of truth. |

## NEEDS-ATTENDED-REVIEW (owner decision; do not act unattended)
| Item | Question | Evidence |
|---|---|---|
| `docs/plans/ratios_roadmap_2026_06_21.md` vs `docs/campaign/roadmap.md` | Collapse to one roadmap? | Overlap; the plans/ version has the red-team detail, campaign/ is the summary. Keep both or merge attended. |
| `docs/plans/q6k_predec_design.md` | Keep? Q6_K predec is **confirmed null** (kill_ledger) | Design doc for a dead lever; archival candidate, but documents the analysis. |
| `crates/hawking-core/src/model/qwen_dense.rs` `mod bsize_verify_diag` (+114, `#[cfg(test)]`) | Still wanted? | Diagnostic test from the spec-decode investigation (non-hot-path). User/earlier-authored — confirm before removing. |
| `docs/plans/rwkv7_*_2026_06_20.md` (several) + training scripts | Training-lane docs — current? | Belong to the KD/training lane (R8); owner = training campaign. Out of inference-runtime scope. |

## DO-NOT-DELETE (proof: load-bearing or entangled)
| Item | Why kept |
|---|---|
| `eagle5*` / `speculate/*` (~1.6k LoC) | The trained-EAGLE spec path is NO-GO for speed, BUT it is **entangled with the committed lossless proposal-market layer** (~140 `eagle` references in `qwen_dense.rs`, committed `73fc5b4`). Removal is a higher-care attended untangling with parity — NOT a prune. |
| ~30 dead-called `*_tcb` optimized kernel wrappers | **Intentional** — parity-tested A/B variants + the per-channel int4-KV trio (a LIVE lever being wired). The architecture scan confirmed they are kept-on-purpose, not dead. |
| mamba2 long-ctx path (buggy @8k) | A model path with a fixable kernel bug (R4), not dead code. |
| `predec`, Q6_K 2r, F16_KV, per-channel int4-KV kernels | Live or ready levers (snapshot). |

## Method note
The prior dead-code audit (memory) found "~nothing safely removable — kernels name-string-referenced, modules held." This
inventory reconfirms that: the codebase is curated, and the apparent dead code is either intentional A/B scaffolding or
entangled with shipped features. **Pruning value here is in DOCS (condensation), not code.**
