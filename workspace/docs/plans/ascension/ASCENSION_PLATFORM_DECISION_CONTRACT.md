# Ascension Platform Decision Contract (Bible §1)

**Status:** CONTRACT SEALED (planning only) — implementation **NOT_STARTED**  
**Bible:** `HAWKING_ASCENSION_BIBLE.md` §1 (Platform decision); CUDA future §34  
**Machine-readable:** [`ASCENSION_PLATFORM_DECISION.json`](./ASCENSION_PLATFORM_DECISION.json)  
**Schedule / overview:** [`ASCENSION_MASTER_SCHEDULE.json`](./ASCENSION_MASTER_SCHEDULE.json), [`ASCENSION_PROGRAM_OVERVIEW.md`](./ASCENSION_PROGRAM_OVERVIEW.md)

---

## 1. Purpose

Freeze the Apple-first production stance so planning and later implementation
cannot silently slip into:

- a CUDA-blocking Apple schedule, or
- a lowest-common-denominator kernel mandate that weakens Metal.

This contract does **not** certify Apple production release, CUDA readiness, or
any schedule step. Those remain `NOT_STARTED` / `CANDIDATE` until controller or
human certification (bible §2, §36).

---

## 2. Stance (bible §1)

```text
Apple-first
Metal-dominant
architecture-portable
CUDA-deferred
```

| Flag | Value | Meaning |
|------|-------|---------|
| `apple_first` | true | Apple silicon is the primary production path for this slice |
| `metal_dominant` | true | Tier-1 kernels and graphs are Metal-optimized |
| `architecture_portable` | true | Model / Gravity / IR / harness contracts stay backend-neutral |
| `cuda_deferred` | true | CUDA is future Tier 1B, not rejected |
| `cuda_rejected` | false | Do not treat deferral as a kill |

---

## 3. Tier 1 — production (Metal)

```text
Apple silicon
Metal
fully optimized
parity-gated
performance-gated
production-supported
```

- Production claims for this programme slice are **Apple/Metal claims**.
- Tier-1 kernels may (and should) use Metal-native command topology, tile
  geometry, cache policy, and autotuning without waiting for a CUDA twin.

---

## 4. Tier 2 — portable contracts

These are the **shared surfaces**, not shared kernel source:

```text
model semantics
Gravity representation contract
kernel grammar / IR
scheduler contract
parity/capability harness
receipt schema
backend-neutral runtime interfaces
```

Portable interfaces exist so later CUDA (or other backends) can re-admit the
same model and Gravity semantics. They are **not** a requirement that every
Metal optimization be expressible as a lowest-common-denominator kernel.

---

## 5. Future Tier 1B — CUDA (deferred, not rejected)

CUDA begins only after Apple production release and only with funded hardware
access (bible §34; master schedule step 33).

Rules:

1. **Do not delay Apple release for CUDA.**
2. **Do not force Metal through lowest-common-denominator abstractions.**
3. **Apple claims remain Apple-specific.**

### Metal and CUDA may share

```text
architecture semantics
Gravity tensor semantics
benchmark contracts
parity/capability suites
scheduler interfaces
receipt formats
```

### Metal and CUDA must not be required to share

```text
kernel source
tile geometry
memory layout
graph implementation
cache policy
command topology
autotuning rules
```

---

## 6. Forbidden under this contract

| Forbidden | Why |
|-----------|-----|
| Lowest-common-denominator kernel mandate | Bible §1: Metal must not be diluted for hypothetical CUDA |
| Delay Apple release for CUDA | Bible §1 / §34 |
| Claim CUDA parity from Apple evidence alone | Independent CUDA hardware + evidence required |
| Claim Apple production from a CUDA path | Wrong backend for this slice’s terminal state |

---

## 7. Integration (without marking stages ready)

| Tracker | Integration |
|---------|-------------|
| Master schedule | `cuda_policy` already encodes deferred stance; steps 12, 32, 33 consume this contract as companion doc |
| Completion states | Terminal Apple state is `HAWKING_APPLE_PRODUCTION_RELEASE_READY` — still `CANDIDATE` |
| Gravity research | Mechanism classification may `DEFER` CUDA-only paths without rejecting them |
| Overview §CUDA | Narrative twin of this contract; machine authority is the JSON |

**No step status flips** are authorized by landing this document.

---

## 8. Acceptance for this planning contract

- [x] Stance flags match bible §1.
- [x] Tier-1 Metal production surface enumerated.
- [x] Tier-2 portable interfaces enumerated.
- [x] CUDA deferred with share / must-not-require-share lists.
- [x] Explicit ban on LCD kernel mandate.
- [x] Machine-readable JSON + structural validation tests.
- [ ] Apple production certification (post schedule step 32).
- [ ] CUDA Tier 1B funding + hardware + evidence (post step 33).

---

## 9. Non-goals

- No Metal kernel implementation changes in this wave.
- No CUDA graphs, CUTLASS, or Triton work.
- No reclassification of schedule steps or completion states to ready.
- No performance claims for Qwen or any live model.
