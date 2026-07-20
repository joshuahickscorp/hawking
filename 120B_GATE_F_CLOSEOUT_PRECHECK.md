# 120B GATE-F CLOSEOUT PRECHECK

Generated 2026-07-19. Campaign: final GPT-OSS-120B Doctor closure (D0-D6), then Qwen3-235B handoff.

## Live truth

- Git: branch `campaign/adaptive-transfer-ladder`, HEAD `ba09c384`, origin/main `4fbca8bc` (5 commits
  ahead, not yet pushed). This /goal authorizes commits/pushes/PRs/merges when green.
- G4 COMPLETE and verified: 12/12 rows sealed, integrity OK (unique ids, valid seals, finite
  metrics), controller exited, lease released, no live controller. Result committed (`ba09c384`).
- Forward validation: 58 cheap adversarial tests pass; 4/5 heavy tests pass (coherence anchor,
  different-prompts, corrupt-tensor, missing-shard); the 5th (wrong-activation-ordering) is
  re-running to complete the audit. Parent perplexity is domain-ordered, which corroborates
  correctness.
- Resources: M3 Ultra 96 GB, 76.8 GB available, 540 GiB disk, network reachable, one heavy lease free.
  Memory policy is the pressure-aware cache (fill RAM + swap, evict only near the danger line).

## G4 = untreated negative control

Uniform untreated RVQ near 1 BPW: 3 collapse, 3 degraded, 0 capability pass (next-token agreement
0.11 to 0.63 vs the 0.95 gate). Sealed as `GPT_OSS_120B_G4_UNTREATED_CONTROL.json` +
`GPT_OSS_120B_G4_UNTREATED_REPORT.md`. This is a real-forward negative control, not Hawking's
strongest treated candidate.

## Final Doctor campaign (D0-D6)

The correction wave already implements the two decisive treated candidates (D2 tensor-class PQ, D4
tensor-class + Doctor). Expanding to add D6 (global byte allocation, asymmetric per organ) and a
diagnosis phase (organ isolation: mlp1-only vs mlp2-only) to localise the failure before treatment.
D1 (uniform RVQ) is already sealed as the G4 control; D3/D5 are variants available if D2/D4/D6 leave
the conclusion open.

Expected science: NEGATIVE at sub-bit (Outcome B honest boundary most likely). The G3 winners needed
2 to 3 BPW for even partial fidelity, and the uniform control collapsed. The purpose is a
transferable scientific result, positive or negative.

## Honest gating

The 120B conclusion requires the D-campaign to COMPLETE (hours, detached, resumable). The Qwen heavy
transfer is gated behind that conclusion plus source release (438 GiB does not coexist with the
61 GB 120B source on 540 GiB). Non-blocking Qwen prep is already done: admission at immutable
revision `ac9c66cc`, Q0 source feasibility proven on real bytes, adapter built and tested.

## Rollback

`git reset --hard 4fbca8bc`; branch holds all work. No source mutated (byte-range read-only). Qwen:
metadata only. Tag on conclusion: `hawking-gptoss-120b-frontier`.
