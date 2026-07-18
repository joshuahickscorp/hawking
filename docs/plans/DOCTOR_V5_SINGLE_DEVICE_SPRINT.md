# Doctor V5 single-device sprint handoff

Status: implemented as an additive, hash-bound, default-off stack. The live
Doctor queue does not import these modules. No completed result, runtime spec,
registry entry, campaign plan, parent source, or runtime default is rewritten by
this work.

## Implemented stack

The single-device path now has all of the following independently gated pieces:

1. Exact 8/12/16/20 thread-profile selection, process-tree RAM reservations,
   and a bounded swap shock absorber.
2. Ordered read/RHT/encode/write execution and a two-slot shard window that may
   overlap one prepare with one serial finalizer only after a measured aggregate
   envelope proves it fits.
3. Bounded source/preprocessing reuse across rates and branches. A whole model
   is never cached; the default is one tensor or expert-batch cache unit, an
   8 GB unit ceiling, and a 64 GB disk reserve. Rate- or branch-dependent work
   is never shared.
4. An elastic phase state machine with one heavy prepare, one primary encoder,
   one serial finalizer, and at most one measured-idle companion. A 20-thread
   encoder is exclusive. Encoder return closes lending and forces a companion
   checkpoint before finalization.
5. A native CPU candidate, receipt-bound PGO generate/merge/use workflow,
   preallocated and mmap I/O, exclusive `.partial` output creation, and source
   identity checks before and after execution.
6. A real Metal RHT adapter and bit-exact CPU/Metal probe. It is compiled but
   intentionally has not dispatched on the occupied machine.
7. A reversible host-sprint proposal set. Caffeination and owned-process
   priority/QoS are gated; fan/SMC writes are forbidden. Spotlight folder
   control is marked unsupported because `mdutil` is volume-oriented.
8. One full-stack paired A/B contract. Component benchmarks are diagnostic only.
   Before execution, a separate external authority must freeze the current
   production-ETA hash, complete tier/rate/branch matrix, workload manifest,
   baseline and candidate binaries, runner, invocations, execution order, and
   per-run owner inventories. The frozen workload contains one source-bound
   real-artifact slice for every 5-tier x 10-rate x 4-branch combination, so a
   convenient unrelated baseline cannot authorize an ETA change. Only at least
   three randomized, owner-free, real-artifact pairs matching that authority,
   with exact outputs, exact scientific receipts, and physical counters, are
   necessary but not currently sufficient to change ETA. Production promotion
   remains hard-disabled until a trusted physical runner and independently
   rooted, non-caller-declared campaign attestation are installed.
9. A phase-aware remaining-scratch ledger. It credits resident scratch only
   after the same ordinal has durable encode, attest, and decode units, keeps
   projected packed output separate, preserves the exact 150 GB reserve, and
   uses descriptor-relative no-follow identity checks without re-reading
   multi-gigabyte payload bodies. This component is mandatory in the full-stack
   promotion receipt; its current output remains read-only and non-activating.

The implementation audit entry point is
`tools/condense/doctor_v5_single_device_sprint_audit.py`. Its packet is written
only below
`reports/condense/doctor_v5_ultra/staged_acceleration/single_device_v1/`.

## Current evidence boundary

The native mmap and preallocated synthetic gates are exact for dense tensors,
sidecars, STR2 archives, and canonical tensor ordering. These receipts are
non-promotable and carry no ETA credit. The Metal contract is compiled but not
dispatched.

The GPT-OSS cache plan covers 615 bounded source units and 24,600 unique
10-rate x 4-branch consumers. At present it permits immutable source-range reuse
only. Decoded, ranked, zeroed-bulk, and RHT reuse remain blocked until the exact
GPT-OSS decoder and preprocessing recipes pass numerical parity. Qwen has a
separate unbound inventory-to-10x4 builder; its recipes must be supplied as an
exact hash-bound matrix rather than inferred from branch names.

Cache receipts now carry and recompute the complete plan/unit/resource sample,
swap-controller transition, and admission decision; an arbitrary resource hash
cannot complete a cache unit. These receipts remain explicitly
synthetic/non-promotable. Production cache execution is hard-disabled until a
lock-scoped trusted local observer replaces caller-supplied resource samples.

Elastic and host plans are staged. Elastic activation remains blocked until the
real tier/rate 8/12/16/20 matrix is qualified. Production phase invocations are
also deliberately structural-only until a reviewed closed target runtime, a
positive RAM ceiling for every heavy phase, and a machine-enforced measured
thread-count proof exist. The two-phase claim handshake, process-group cleanup,
target identity, authoritative thread environment, RSS guard, completion chain,
and semantic validator are fixture-tested scaffolding; they do not themselves
qualify a Python target or grant ETA credit.

## Promotion sequence

At a terminal, quiescent Doctor checkpoint:

1. Prove zero heavy owners, normal pressure, stable thermal state, sufficient
   disk reserve, and a sealed swap baseline.
2. Re-run the exact 8/12/16/20 matrix on representative real artifacts and bind
   the winning thread count for every tier/rate. No nearest-profile fallback is
   allowed.
3. Train PGO only on the sealed representative corpus, merge with the exact
   hash-bound `llvm-profdata`, build a new candidate directory, and re-run exact
   output/receipt parity.
4. Run CPU and Metal preprocessing separately first. Admit a stacked CPU+GPU run
   only if isolated receipts show shared-power, bandwidth, RAM, and thermal
   headroom.
5. Run the bounded Qwen and GPT-OSS reuse paths against independent serial
   oracles. Require complete scheduled/executed/validated coverage, zero skips,
   unique output/evidence instances, and exact cache refcounts before ephemeral
   cache deletion.
6. Run at least three randomized/interleaved full-stack baseline/candidate pairs
   on the same machine and real artifact. The authority file must predate every
   run, remain separate from the receipt, and bind the exact current ETA
   snapshot. Feed only the conservative minimum paired speedup into the ETA
   projection only after the independent trust root is available. A
   self-contained or caller-invented receipt is rejected; a structurally valid
   receipt can be retained for diagnostics but cannot promote an ETA.
7. Create a new pending-only source-bound runtime generation, run rollback and
   crash-resume gates, then promote atomically. Never modify terminal evidence.

The 120B and Appendix segments require their own representative receipts. A
sub-120B speedup is not applied to those segments merely because the code is
shared.

## Commands

All commands below are inert or read-only unless their explicit production gate
has already been satisfied:

```sh
python3.12 tools/condense/doctor_v5_host_sprint_plan.py stage
python3.12 tools/condense/doctor_v5_elastic_phase_scheduler.py stage
python3.12 tools/condense/doctor_v5_shared_preprocess_cache.py requirements
python3.12 tools/condense/doctor_v5_shared_preprocess_cache.py build-gptoss-plan
python3.12 tools/condense/doctor_v5_single_device_sprint_audit.py stage
python3.12 tools/condense/doctor_v5_single_device_sprint_audit.py verify
python3.12 tools/condense/doctor_v5_single_device_benchmark.py threshold \
  --production-eta reports/condense/doctor_v5_ultra/staged_acceleration/production_calibrated_eta.json \
  --target-days 7
```

`benchmark.py validate --require-production` and `benchmark.py project` also
require `--authority <frozen-authority.json>`. No authority is staged while the
machine has heavy owners, so the current implementation intentionally cannot
grant production ETA credit. The code additionally hard-disables production
promotion until the trusted runner/attestation boundary exists; a caller-built
hash chain, even one claiming an arbitrarily large speedup, is insufficient.

Native build and exact I/O details are in
`vendor/strand-quant/NATIVE_EXECUTION.md`. The earlier campaign and higher-tier
map remains in `docs/plans/DOCTOR_V5_PARALLEL_ACCELERATION_HANDOFF.md`.
