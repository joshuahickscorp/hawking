# Doctor V5 14B/3bpw resource-stop recovery staging

Status: implemented as an inert, default-off, incident-specific staging path.
It does not unblock production by itself and it cannot apply or resume a queue.
The incident is now historical and closed: the target completed successfully on
2026-07-16. The expected resource-stop receipt is absent from the completed
generation, so the retained staging tool correctly fails closed against current
state.

The reviewed tool is
[`doctor_v5_resource_stop_recovery_stage.py`](../../tools/condense/doctor_v5_resource_stop_recovery_stage.py).
Its CLI contains only `status`, `stage`, and `verify`. It never imports the live
queue, acquires either live campaign lock, sends a signal, changes control or
state, deletes a source, or writes outside
`staged_acceleration/resource_stop_recovery_3bpw_v1`.

## Incident boundary

The boundary below describes the stopped generation captured during the
incident, not the current live target row.

The path is pinned to `qwen2-5-14b__3bpw__codec-control`, attempt 10, exit 75,
the exact resource-stop reason, plan and cell identity, adapter request and
registry, runtime spec, adapter and worker checkpoints, worker request,
resource-stop receipt, target row, and quantizer/attestor/decoder hashes. The
checkpoint must still contain this exact prefix:

```text
preflight, metadata,
passthrough/encode/attest/decode shard 00000,
passthrough/encode/attest/decode shard 00001,
passthrough shard 00002
```

At the incident snapshot, shards 0 and 1 were durable complete shard units and
shard 2 was only passthrough-complete; no recovery proposal could discard or
invent progress. The captured phase-aware ledger counted only the two exact
encode+attest+decode chains and kept the packed-output projection separate.

## Incident-generation disk incompatibility

Resetting the row would not have been enough. The activated queue generation
charged
the runtime spec's complete 48,000,000,000-byte scratch budget. It subtracted
the 1,855,140,432 bytes of packed archives already present, but it had no
source-bound consumer for the ledger's 7,880,058,234 bytes of completed dense
reconstructions. Its exact checkpoint restart requirement was therefore:

```text
150,000,000,000 reserve
+ 48,000,000,000 full scratch
+  5,237,498,455 remaining packed output
=203,237,498,455 bytes free
```

The phase-aware result was 195,357,440,221 bytes. With free space near 201.7 GB
at the incident snapshot, a CAS-only recovery would have immediately re-blocked
on the old gate.

The separate default-off
[`doctor_v5_remaining_scratch_gate_adapter.py`](../../tools/condense/doctor_v5_remaining_scratch_gate_adapter.py)
implements the future source-bound consumption contract without importing or
patching the live queue. Its only production-facing function fixes the exact
plan/cell/request/checkpoint and uses direct time/disk probes. It accepts a
fresh ledger receipt, independently recomputes it from the frozen request and
checkpoint, and compares every source identity and byte equation. Absent,
stale, malformed, path/ordinal/plan/request/checkpoint-drifted, or resealed
reduction evidence falls back to 150 GB reserve + full 48 GB scratch + full
packed projection. There is no caller scratch-reduction or free-byte input.

The historical recovery packet binds that adapter's source and hash, but
explicitly sets
`live_queue_remaining_scratch_consumption_absent: true` and
`separate_adapter_wired_into_live_queue: false`. Adapter existence or a valid
receipt cannot erase the mandatory re-block blocker. A separately reviewed
queue generation must wire the exact adapter contract before any 3bpw CAS;
this staging path contains no bypass.

The incident design treated the then-blocked
`qwen2-5-14b__4bpw__codec-control` row as a separate prerequisite. That row is
now complete. The tool still binds and structurally audits the reviewed 4bpw
recovery tool; it does not broaden its own CAS to cover that row. The intended
owner-free order was:

1. execute the separately authorized exact-checkpoint 4bpw recovery;
2. re-observe and re-bind every state, owner, lock, disk, RAM, and swap gate;
3. only then consider a separately authorized 3bpw CAS executor.

## Historical swap-promotion design

The original incident baseline of 0.25 MB is retained as history, not made an
eternal recovery ceiling. A new baseline can be sealed only by an explicit
`stage --seal-stable-swap-generation` invocation at an owner-free, child-free,
supervisor-free, live-lock-holder-free checkpoint. The tool itself takes three
direct trusted samples over 10–45 seconds. Every sample must show:

- normal memory pressure, nominal thermal state, and AC power;
- at least the 78 decimal GB process budget plus 8 decimal GB reserve available;
- finite, non-rising swap below the absolute 4096 MB emergency ceiling;
- the exact trusted `sysctl`, `pmset`, `memory_pressure`, and `statvfs` probes.

The final direct observation becomes the generation's immutable sealed
baseline. The existing growth/rate controller is initialized against that
value and all later decisions remain relative to it. Receipts are
content-addressed and create-only; no caller-supplied proof path is accepted
and no generation can be overwritten in place. Invalid, rising, high, stale,
future-dated, owner-contaminated, low-RAM, or ambiguous generations fail
closed.

## CAS and rollback boundary

The staged packet describes, but cannot execute, a future compare-and-swap.
Its only proposed target-row changes are `status`, `blockers`, `error`, and
`last_exit_code`; `attempts` stays 10. It binds the exact state file and
semantic generation, target and non-target row hashes, completed checkpoints,
resource-stop receipt, scratch ledger, swap promotion, and binary hashes. A
future reviewed executor would still have to obtain both exclusive live locks,
revalidate every binding after the expensive checks, and atomically persist
intent and rollback receipts before changing state. Other rows, completed
evidence, results, control, runtime defaults, and sources are outside scope.

Read-only historical-status inspection:

```sh
python3.12 tools/condense/doctor_v5_resource_stop_recovery_stage.py status
```

The following staging surfaces are retained for audit and should not be invoked
for the completed target:

```sh
python3.12 tools/condense/doctor_v5_resource_stop_recovery_stage.py stage
python3.12 tools/condense/doctor_v5_resource_stop_recovery_stage.py verify
```

The historical owner-free sealing surface was:

```sh
python3.12 tools/condense/doctor_v5_resource_stop_recovery_stage.py \
  stage --seal-stable-swap-generation
```

Even a fully green staged packet reports `activation_permitted: false` and
`apply_implementation_present: false`. `verify` separately reports historical
packet integrity and fresh-current commit readiness; an internally intact
packet is never described as commit-ready while the supervisor, owners, locks,
state generation, swap promotion, or live queue-consumer gate is unresolved.
