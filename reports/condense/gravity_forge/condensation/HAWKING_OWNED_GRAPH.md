# Hawking owned graph (deduplicated, honest)

**One repository** (`joshuahickscorp/hawking`) — no sibling repos, no submodules, no worktrees beyond `main`.

| Node | Path | LOC | Status |
|---|---|--:|:--|
| **Seed (gravitational core)** | crates/hawking-seed-c | 2,204 | ACTIVE / SELECTED (merged, tag hawking-seed-event-horizon) |
| legacy multi-model engine | crates/hawking-core | 60,118 | ACTIVE (superseded for Llama by the Seed; sole provider for RWKV/Qwen/DeepSeek/MoE/serving/kernels) |
| CLI | crates/hawking | 4,089 | ACTIVE |
| serving | crates/hawking-serve | 4,404 | ACTIVE |
| speculation pack | crates/hawking-speculate | 4,157 | ACTIVE (extracted, decode-parity green) |
| bench lab | crates/hawking-bench | 1,010 | ACTIVE lab |
| Metal shaders | crates/hawking-core/shaders | 10,215 | ACTIVE performance |
| control/lab tooling (Python) | tools | 51,098 | ACTIVE lab/control |
| HIDE client | app (excl dist) | 6,359 | ACTIVE client |
| strand quant (absorbed) | vendor/strand-* | 47,490 | OWNED-INACTIVE (audit-only, workspace-excluded) |
| built bundles | app/dist | 64,860 | GENERATED (excluded) |
| stale agent worktrees ×3 | .claude/worktrees | ~360k | RECLAIMABLE DUPLICATION (excluded) |

**GLOBAL_ACTIVE_OWNED_LOC ≈ 142,812.** Seed core hits the ≤3,000 target; the remainder is the accretion
disk. Relocation is not reduction and duplication is not implementation — both are labeled and excluded,
never counted as eliminated. See HAWKING_REUNIFIED_ARCHITECTURE.md for the topology and the honest
distinction between what is done and the ongoing capability-preserving collapse.
