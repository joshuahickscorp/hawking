# Doctor V5 14B/4bpw blocked-cell recovery

Status: historical and closed. The target completed successfully on 2026-07-15.
The retained tool now rejects the advanced runtime, checkpoint, and terminal
target row, so this document is an incident record rather than a live recovery
instruction.

This is a narrow incident-recovery scaffold for
`qwen2-5-14b__4bpw__codec-control`. It does not change queue policy and must not
be generalized to other cells.

Incident durable boundary:

- all eight shard encodes are complete;
- shards `00000`–`00006` are attested and decoded;
- `attest:00007` is the next worker unit;
- the restored attestor and decoder match the hashes in the frozen runtime;
- the target queue row is `blocked-execution` at attempt 14 after the missing
  attestor error;
- the older outer adapter checkpoint is retained as historical context, while
  the newer worker checkpoint is the resume authority.

The implementation is
[`doctor_v5_blocked_cell_recovery.py`](../../tools/condense/doctor_v5_blocked_cell_recovery.py).
Its normal surfaces are:

```bash
python3.12 tools/condense/doctor_v5_blocked_cell_recovery.py status
python3.12 tools/condense/doctor_v5_blocked_cell_recovery.py stage
python3.12 tools/condense/doctor_v5_blocked_cell_recovery.py verify
```

`status` and `verify` are read-only. `stage` writes only beneath
`reports/condense/doctor_v5_ultra/staged_acceleration/blocked_cell_recovery_v1`.
An unkeyed or owner-active stage is deliberately non-activatable. No staging
command pauses, drains, resumes, or edits the campaign.

## Historical conditional activation contract

For the captured incident generation, activation would have been permitted only
after a fresh owner-free stage committed two independent keys. The inert
`apply` implementation required, under one critical section:

1. the exact plan, target cell, state generation, target row, runtime, request,
   registry, restored binaries, checkpoints, active acceleration marker, queue
   entrypoint, and accelerated autoresume entrypoint;
2. no detached supervisor, no recorded child/cell, and no external heavy owner;
   the persisted queue status had to already be `drained` or
   `waiting-prerequisites`, with control already `run`;
3. the singleton queue lock, campaign-wide heavy lock, and recovery lock;
4. a full hash verification of every completed checkpoint artifact and every
   runtime input, performed only after the owner-free gate;
5. exactly one target-row patch: `blocked-execution` to `pending`, clearing only
   its exit code, blocker list, and error while preserving attempt 14 and all
   source/evidence bindings;
6. semantic equality of every other cell and every forbidden top-level field;
7. durable before-intent and after receipts; and
8. resume through the then-active, hash-verified accelerated-autoresume
   marker, followed by a detached-supervisor ownership verification.

The retained apply command never mutates control, completed evidence, results,
runtime specs, registries, adapters, parent sources, or the acceleration
marker. A stale packet, changed binary/checkpoint, duplicate intent/receipt,
symlink/path escape, malformed JSON, unavailable lock, fake owner, terminal
target, or failed detached handshake is a hard refusal.

Do not run `status --full`, `verify --full`, or `apply` while any campaign or
corpus owner exists. The code also enforces this boundary before any
high-bandwidth read.
