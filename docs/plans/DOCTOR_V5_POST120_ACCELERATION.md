# Doctor V5 120B and post-120B acceleration handoff

This is a separate, default-off qualification layer. It is not imported by the
live Doctor queue, worker, adapter registry, reporter, or runtime-spec loader.
It does not authorize a model launch or a scientific claim.

## Bound 120B graph

The sealed handoff binds, by file and semantic hash:

- the reviewed GPT-OSS 120B inventory-derived work plan: 615 bounded source
  units and 6,150 source/rate outputs;
- the exact pending 10-rate x 4-branch campaign matrix: 40 cells and zero live
  runtime specs or registry entries;
- the source/rate/branch reuse fanout: 24,600 isolated output jobs with one
  bounded source traversal per source unit and no evidence sharing;
- the shared-preprocess consumer manifest and bounded cache plan;
- the tokenizer binding and dual-path/round-trip gate;
- the generic >120B source-manifest and admission contract; and
- the acceleration requirements, GPT-OSS acceleration plan, and three named
  post-120B horizon templates.

Its physical A/B qualification parent is regenerated under
`reports/condense/doctor_v5_unbound/post120_acceleration/`. It is deliberately
not the staged plan inside the live Doctor tree. Verification rebuilds it from
the current controller source, while live issuance remains deferred to the
final-ready, zero-owner release boundary.

The 40-cell matrix and all 24,600 job identities are recomputed during
verification. Re-sealing a substituted cell, parent, profile, source, or
artifact does not make it valid.

## Aggressive single-device facets

Every model/rate has 8/12/16/20-thread calibration candidates. Selection is
allowed only from physical, exact-output receipts for the same source, rate,
branch, executable, host, and generation. The same gate covers:

1. per-rate and per-phase thread profiles;
2. deterministic block parallelism and canonical merge;
3. bounded read/RHT/encode/write/attest overlap across different source units;
4. bounded source/preprocess reuse with isolated evidence and artifacts;
5. measured per-phase RAM packing and behavior-bound resource claims;
6. sealed-baseline, non-ratcheting controlled swap admission and recovery;
7. phase-aware disk admission and hash/fsync/seal/successor-bound ephemeral GC;
8. native I/O and PGO with exact-output and per-rate regression gates;
9. Metal preprocessing only after same-artifact CPU parity and physical
   counter receipts;
10. tokenizer/corpus/model-bound, zero-skip exact quality receipts for all 40
    cells; and
11. immutable-generation start/completion CAS and a sealed rollback point.

Estimates, synthetic fixtures, and structural checks cannot qualify a facet.
Source or parent deletion is never permitted. Runtime defaults remain
unchanged.

## Higher-tier wiring boundary

DeepSeek-V4-Flash, Kimi-K2.6, and DeepSeek-V4-Pro each have an exact 40-cell
template and the complete acceleration profile lattice. Their advertised sizes
are display labels, not parameter authorities. Their adapter IDs, exact logical
parameter counts, source manifests, and admission plans intentionally remain
null until immutable architecture and source evidence exists. Once a valid
manifest and admission plan are supplied, the same tool deterministically seals
the source-unit x 10-rate x 4-branch job space.

This is the maximum honest wiring possible without inventing a model payload,
architecture adapter, or physical result.

## Cheap commands

```sh
python3.12 tools/condense/doctor_v5_post120_acceleration_scaffold.py requirements
python3.12 tools/condense/doctor_v5_post120_acceleration_scaffold.py horizons
python3.12 tools/condense/doctor_v5_post120_acceleration_scaffold.py build-gptoss
python3.12 tools/condense/doctor_v5_post120_acceleration_scaffold.py handoff
python3.12 tools/condense/doctor_v5_post120_acceleration_scaffold.py verify-handoff
```

For a future exact higher-tier manifest:

```sh
python3.12 tools/condense/doctor_v5_post120_acceleration_scaffold.py \
  build-higher --manifest MANIFEST.json --admission-plan ADMISSION.json \
  --output ACCELERATION.json
```

Promotion remains blocked until all eleven physical facet receipts, reviewed
architecture adapters, exact source bindings, owner-free baselines, disk and
lifecycle admission, rollback point, and observer structural-readiness gate
pass at one quiescent generation.
