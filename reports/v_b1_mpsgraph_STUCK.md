# Phase B Wedge B1 — STUCK: MPSGraph Rust FFI bindings

**Wedge:** B1 (and B2, B3, C1–C4 by dependency)  
**Status:** STUCK — version incompatibility, hard-fail signal  
**Date:** 2026-05-04

## What failed

`objc2-metal-performance-shaders-graph = "0.3.2"` (the only viable MPSGraph
crate on crates.io) requires `objc2 v0.6.4` and `objc2-foundation v0.3.2`.
The dismantle workspace pins `objc2 = "0.5"`, `objc2-foundation = "0.2"`,
`objc2-metal = "0.2"`. These are incompatible type families — Cargo resolves
them as separate crates, so the public API types of `objc2-metal-performance-shaders-graph`
(NSArray, NSData, NSDictionary, MPSDataType, MPSShape, …) are invisible
from code that imports the workspace versions.

Test compile error (attempt 1 / 1):
```
error[E0432]: unresolved imports `objc2_foundation::NSArray`,
  `objc2_foundation::NSData`, `objc2_foundation::NSDictionary`,
  `objc2_foundation::NSNumber`
```

## Self-healing assessment (hard-fail, not recoverable in 3 iterations)

- **Option A — Upgrade workspace to objc2 0.6.x / foundation 0.3.x / metal 0.3.x:**
  Would require rewriting every `extern_methods!`, `define_class!`, and
  `retain`/`release` call in `src/metal/mod.rs` and all Metal shader dispatchers
  (~2000 lines of safe objc2 code). Estimated: 3–5 days of attended work.

- **Option B — Raw ObjC runtime FFI:**
  `extern "C"` bindings to `objc_msgSend`, class lookup, `NSData` allocation,
  `NSDictionary` construction, MPSGraph `run:...` selector calls — all without
  type safety. ~500 LoC, extremely error-prone, 1–2 weeks.

- **Option C — Separate MPSGraph sub-crate:**
  Create `crates/dismantle-mpsgraph/` with its own dep tree pinning objc2 0.6.x.
  Feasible, but adds cross-crate FFI boundary, Cargo workspace complexity, and
  still requires rewriting all interop types (MTLBuffer, MTLDevice, MTLCommandQueue)
  at the boundary. 1–3 weeks.

None of these are haul-scope work.

## Impact

B2 (LM head via MPSGraph), B3 (Phase B close), C1–C4 (attention projections)
are all blocked by B1. The entire MPSGraph block (Phases B + C) is deferred.

## What attended work unblocks

**Attended Option A (recommended):** In a dedicated session, upgrade workspace:
```toml
objc2 = "0.6"
objc2-foundation = "0.3"
objc2-metal = "0.3"
objc2-metal-performance-shaders-graph = "0.3.2"
```
Then fix all compilation errors iteratively. The existing Metal code largely
just needs `retain` → `Retained<T>` and method-family annotation updates. Once
compiling, B1 hello-world (1×4 @ 4×1 matmul constant graph) should work
as shown in the partially-written test stub.

## Followups

- Re-attempt B1 after objc2 0.6.x workspace migration.
- Consider starting with a minimal `dismantle-mpsgraph` crate if workspace
  migration risk is too high.
- Alternative GPU matmul path: `MPSMatrixMultiplication` from `objc2-metal-performance-shaders`
  has same version conflict but may be addressable with same migration.
