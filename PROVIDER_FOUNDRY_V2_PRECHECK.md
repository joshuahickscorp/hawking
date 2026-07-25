# PROVIDER FOUNDRY V2 PRECHECK

Generated 2026-07-20T19:16:18Z.

## Non-interference
Live parent Qwen3-235B controller pid 86590 is RUNNING at layer 48/94
in `/Users/scammermike/hawking-qwen-recovery-20260720`. The foundry does not touch that checkout, does not
acquire the heavy lease, starts no download, and launches no saturating process.

## TCC isolation
Prior chain freezes were caused by launchd being unable to exec under `~/Downloads` (protected folder,
EPERM). The foundry worktree is `/Users/scammermike/HawkingWorktrees/deep-architecture-foundry` on `codex/hawking-deep-architecture-foundry` from origin/main
4fbca8bc.

## Provider universe
`crates/hawking-seed-c/src/providers/` already exists (2199 LOC, 13 files) with a declarative
`ArchAdapter` pattern and builtins llama/gemma2/phi3/olmoe/mixtral/gpt_oss/mamba2. The foundry EXTENDS
this ABI. No second provider framework, registry, execution IR, or source authority is created.

## Verification of reported findings
Most reported findings verified from live evidence. Two corrections:

- **Cache policy 64 GB / floor 12 GB is FALSIFIED for the single-pass workload.** A lockstep pass has zero
  cross-layer expert reuse, so a 64 GB cache accumulated 57.7 GB at 0 evictions and drove available RAM
  70 -> 18 GB with swap down to 906 MB free, reproducing the prior jetsam signature. Correct cap is ~20 GB
  (working set is ~8 GB per layer). The aggressive-RAM policy is valid only where real reuse exists.
- **Routing-frequency allocation is not yet calibratable.** At 88 holdout tokens the median hot/cold split
  is only 63.6 percent stable and 26.1 percent of (layer, expert) cells are never routed. ~1000 tokens are
  needed. The live ladder therefore uses the coldest quartile, not the median.
