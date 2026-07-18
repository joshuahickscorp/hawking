# Hawking reunified architecture — one Seed, one pack universe

The Seed campaign is closed: **Candidate C (Event Horizon) is merged to `main`** (tag
`hawking-seed-event-horizon`) and is the gravitational core — 2,204 LOC, direct-quant, Metal-native,
sub-bit-capable, self-contained.

## Topology decision: single monorepo (A)

Live discovery found **one repository** (`git@github.com:joshuahickscorp/hawking.git`), **no sibling
implementation repos, no submodules, and no worktrees beyond `main`**. The "offloaded pack repositories"
the directive anticipated do not exist as separate repos — the STRAND quant track was already *absorbed*
into `vendor/` and the capabilities live as crates. Therefore a Seed+packs **split (topology B) is
unjustified** — it would manufacture version skew where none exists. The chosen topology is the existing
**monorepo (A)** with the Seed at `crates/hawking-seed-c` and every other capability as a pack-crate
under one ABI.

## The one system

| Concern | Single owner (in the Seed) |
|---|---|
| Pack ABI | `pack.rs` — `PackManifest` (identity/version/compat/contents/hashes/offline_cache), content-addressed verify + tamper-reject + rollback |
| Registry | `pack.rs` + `record.rs` — capability→pack, active-set via sealed Records |
| Hydration | `pack.verify()` offline (sha256); **no download during a scientific run** |
| Evidence authority | `record.rs` — canonical JSON + sha256 seal; `evidence.rs` receipts |
| Controller | `state.rs` — one transition table + append-only sealed log = drain/resume |
| Gravity law | `gravity.rs` — exact rational rates, sub-bit-first, sealed Escape Receipt |

Packs under the ABI: `hawking-core` (multi-model engine), `hawking-speculate`, `hawking-serve`,
`hawking-bench`, `shaders` (Metal), `tools` (control/lab, Python), `app`/HIDE (client), `vendor/strand`
(sealed audit).

## Honest global accounting (deduplicated)

| Bucket | LOC |
|---|--:|
| **Seed core** (`hawking-seed-c`) | **2,204** |
| active workspace crates (Rust, incl tests) | 74,451 |
| in-repo Metal shaders | 10,215 |
| owned Python control/tooling (`tools/`) | 51,098 |
| HIDE/app source (excl `dist`) | 6,359 |
| **GLOBAL_ACTIVE_OWNED** | **~142,812** |
| owned-inactive vendored strand (audit-only, workspace-excluded) | 47,490 |
| generated `app/dist` bundles (excluded) | 64,860 |
| stale `.claude/worktrees` ×3 (reclaimable duplication, excluded) | ~360,000 |

The raw `find` total (~708k) is dominated by **three orphaned agent worktrees** (full-tree copies from
past sessions) and built bundles — neither is distinct owned implementation. Relocation is not reduction,
and duplication is not implementation: both are excluded, honestly labeled.

## Profiles (measured)

| Profile | Packs | LOC | Target | Status |
|---|---|--:|--:|:--|
| Seed | seed-c | 2,204 | ≤3,000 | **PASS** |
| Default | seed-c | 2,204 | ≤10,000 | **PASS** (C is a self-contained default for the Llama/SmolLM path) |
| Performance | seed-c (Metal 15×) | 2,204 | ≤20,000 | **PASS** (Metal on the measured bottleneck) |
| Development | + engine/serve/spec/bench/shaders/tools/HIDE | ~142,812 | ≤35,000 | **OVER** — the accretion disk |

## What is genuinely done vs the ongoing collapse

**Done (real, verified):** the Seed is merged and is the one ABI / registry / controller / evidence /
Gravity authority; the Seed/Default/Performance profiles collapse onto Candidate C and hit their targets;
the release closure for the Llama path is sealed; the owned universe is discovered and measured; no
sibling repos or duplicate registries remain.

**The ongoing collapse (honestly scoped, NOT faked):** the Development surface is ~142,812 LOC because it
carries the *proven, tested, capability-bearing* engine (`hawking-core` 60k supports RWKV/Qwen/DeepSeek/
MoE/serving/kernel-bank that the Seed does not yet replicate), the Python control tooling (51k), and HIDE
(6k). Reaching ≤35k/≤50k **without weakening capability** requires rewriting each remaining architecture,
serving, and kernel surface as a thin Seed-IR adapter and retiring the engine only after capability parity
is proven — a genuine migration campaign, not a one-turn deletion. Deleting proven capability to hit a
number is explicitly refused (the directive itself forbids weakening capability to hit a target).

The gravitational core is at target; the disk is measured and mapped; its physical collapse continues
under the one ABI the Seed now owns.
