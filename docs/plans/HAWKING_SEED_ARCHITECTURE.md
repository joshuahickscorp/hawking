# Hawking Seed — architecture specification + 3-candidate design (Phase 1 oracle)

> Seed is the smallest stable object from which Gravity, Forge, Doctor, runtime, evidence, and
> giant-model capability can be summoned. Compress the strings until only the laws remain.

Status: **Phase 0 complete** (main = `hawking-pre-seed-final`, the reference oracle). This document is
the **Phase 1 specification** — the behavioral contract Seed must satisfy — plus the three candidate
architectures (§25). It is the blueprint, not a built Seed; building A/B/C is the multi-session
campaign that follows, gated by the invariants below.

## The oracle (what Seed must reproduce, already executable)

These are the invariants any Seed candidate is gated against — they exist and are green on the
sealed reference:

- **Decode parity** — `decode_parity_harness.py` golden `2d1559cf` (SmolLM-135M, exact profile,
  greedy): baseline / exact-shared / suffix-automaton bit-identical. Seed runtime must reproduce it.
- **Gravity law** — 34 `test_succ_gravity` invariants: whole-artifact BPW < 1.0 default region,
  representation-before-BPW escalation, Doctor bytes inside the physical budget, sealed Escape Receipt
  to leave sub-bit, exact rational rates, one heavy controller, `gravity_enabled=false` default.
- **Pack ABI** — 4 sealed manifests (`packs/*.json`): capsule, hide-desktop, lab-cli, adapters-extra;
  content-addressed, offline-hydratable, source in git history.
- **Runtime contract** — `hawking generate` (load → prefill → decode → sample) on real GGUF fixtures.
- **Evidence** — canonical JSON + sha256 seal + receipts (`eco_common`, the FORGE_* artifacts).

## Seed microkernel target (§9)

```
hawking-seed/
    core        identity, config, Record envelope, seal/hash        (evidence density §23)
    protocol    one typed Record + transition table                 (§10, §11)
    controller  one state machine + policies (Gravity/Forge/Doctor)  (§11, §12)
    runtime     tiny contract (inspect/load/prefill/decode/state/unload) + execution IR  (§18, §19)
    interface   registry CLI (<=1k) + one local API (<=1k)           (§21, §22)
    pack        Pack ABI verify/select/hydrate                       (§13)
```

Seed owns **authority + contracts**; packs own **optional implementation mass**.

### One Record envelope (§10) — replaces the many serializers/validators

```
Record { kind, version, identity, parent, state, payload, evidence, seal }
```
Covers queue rows, parent rows, programs, experiments, events, receipts, readiness, artifacts,
sources, capability results, resource status. Canonical serialization + identity + seal + migration
in one engine. Payloads are declarative/generated typed structs (not stringly-typed).

### One state machine (§11) — replaces procedural controllers

`idle → prepared → admitted → running → {draining|paused|blocked|failed} → sealed`, driven by one
transition table `(from, event, guard, action, receipt, to)`. Queue, acquisition, source streaming,
Forge fitting, Doctor treatment, packaging, evaluation, GC, rollback, resume are all transitions —
not separate engines. Parent/experiment/pack specifics are payload substates.

### Policies as data (§12)

Gravity = a small pure policy `(state, evidence, envelope, impls) -> (decision, reason, receipt,
next)`. Forge family selection = registry lookup. Doctor mechanism selection = data table. No
controller-inside-a-policy.

### Execution IR (§19) — replaces per-model forward duplication

`Load, Norm, Linear, Attention, Route, Experts, Activate, Residual, Sample`. Adapters translate
architecture metadata → IR; the runtime executes it. Gated by logit/KV/routing parity + performance.
Architecture-specific ops use specialization; do not force incompatible archs into a weak abstraction.

## The three candidates (§25) — to be built, compiled, and measured in isolated worktrees

| | Candidate A — Rust microkernel | Candidate B — Functional Rust | Candidate C — Mixed |
|---|---|---|---|
| authority/controller/evidence/runtime | Rust | Rust, aggressively pure transition engine | tiny Rust shell |
| schemas | handwritten typed | generated from one schema authority | generated |
| policies (Gravity/Forge/Doctor) | Rust functions | pure functions + state vectors | Python/WASM policy packs |
| test model | state-machine + parity vectors | property tests over transition vectors | Rust vectors + Python fitting |
| hypothesis | densest authority, direct | smallest test/control surface (§17) | smallest Rust core, policy flexibility |
| risk | Rust boilerplate floor | generation must reduce, not relocate, LOC (§16/§522) | cross-language boundary cost |

Each must do the Phase-3 migration (§29): `status → artifact load → one model → decode → Gravity
fixture → Forge fixture → Doctor fixture → evidence`, all parity-green, before it counts as real.

## Measurement plan (§7) — reported per candidate

`seed_core_LOC, seed_ship_LOC, minimal/performance/development_profile_LOC, Rust/Python/Metal/
generated_LOC, binary_bytes, startup, decode tps, memory, build_time, test_count`. Relocated ≠
eliminated; generated ≠ free; owned pack ≠ external.

## Honest bands (§8)

10k gravitational · 15k extreme · 20k strong · 30k only-with-proven-lower-failures · 40k+ continue.
A final Escape Receipt requires all three candidates + ≥2 lower redesign attempts (§34/§847). Per
§70/§815 an honest larger Seed beats a dishonest smaller one.

## Where the reference stands (the starting mass to compress)

owned product src **88,418** (hawking Rust 69,148 + Python 19,270); hawking-core src 52,413; CLI
4,851; speculate already a 6,312-LOC pack. The compression path: Record+state-machine collapse of the
Python controller (19k) + the Rust controller; execution-IR collapse of the model forwards; kernel-bank
ABI (the tuned Metal to a pack); minimal default packs. Building + measuring this across A/B/C is the
campaign the durable ledger (`NUCLEAR_PASTA_STATE.json`) resumes.
