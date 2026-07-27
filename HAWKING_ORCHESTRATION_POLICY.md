# Orchestration policy — standing delegation zones

**Status: CANONICAL.** Applies for the remainder of the Continuum, across agent context
boundaries. A successor reads this instead of re-deriving who does what.

Until now delegation was ad-hoc: each lane scoped, launched and reviewed one at a time,
which made Claude the bottleneck on every handoff. This assigns standing ownership instead.

## Zones Grok owns standing — delegate without asking

Claude writes the contract and reviews the diff. It does not need to be asked first.

| Zone | Scope |
|---|---|
| **HIDE implementation** | every remaining wire, Context OS, tools/effects, durable sessions, agents, fleets, worktrees, verification |
| **Fabric / Bridge** | Fabric agent, placement, pipeline, Bridge surfaces, SDKs, headless CLI |
| **Adapters** | one ABI, one registry, per-family coverage raised honestly |
| **Acceleration** | provider ABI, drafting families, tree/multi-candidate verification, lookup, caching |
| **Rust reclamation** | fused kernels, ports, benchmark harnesses |
| **Recon / archaeology** | any "what exists versus what is claimed" survey, always read-only |
| **Test authoring** | reachability tests, benches, fixtures |

Standing rule for every zone: **a report is a claim.** Nothing is accepted until Claude has
read the diff and re-run the tests. That has already caught a wrong recommendation, an
unverifiable assertion, and a survey framing error tonight.

## Zones Claude keeps — never delegated

- **Artifact semantics and codec determinism.** Which rung, which kernel, what a shard
  header asserts. A wrong call here silently corrupts an artifact that takes a day to rebuild.
- **Anything touching a running pack or a sealed shard.** Non-negotiable while a job is live.
- **Security and permission boundaries.** Grok may implement; Claude specifies the invariant
  and verifies it. The `step_id` blanket-approval hole was found by Claude reviewing Grok's
  own passing tests.
- **Scientific claims and evidence tiers.** What counts as measured versus modelled versus
  proxy.
- **Branch topology and merges.**
- **The goal set.**

## Sequencing constraints that bind every zone

1. **No merge into the campaign working tree while PASS3 runs.** Spawned pack workers
   re-import from disk; a merge mid-pack can corrupt a shard.
2. **The HIDE line and the campaign line must meet before Fabric work begins.** They are
   currently separate trees, which is why MCP is simultaneously `REAL_WIRED` and absent.
3. **Recon before build, always.** HIDE's archaeology found zero missing subsystems and
   twelve unreachable ones. Building before surveying adds to an unreachable pile.
4. **No new surface before consolidating authorities.** Six event models exist because
   surface was added faster than authorities were merged.

## Concurrency

Multiple Grok lanes run in parallel, each in its own worktree. Never two agents in one
worktree. Lanes that will both touch a file must be sequenced, not parallelised — the HIDE
increments were chained for exactly this reason.

Every lane is listed in `HAWKING_CONTINUUM_STATUS.md` under delegated lanes with its branch
and whether it finished, so an interrupted session cannot silently duplicate one.
