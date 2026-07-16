# Doctor V5 phase-aware remaining-scratch ledger

Status: implemented as a read-only, unbound, default-off observer. It is not
imported by the live Doctor queue and has no activation, write, repair, resume,
or mutation command.

The ledger closes one conservative disk-admission gap for a future reviewed
queue generation. A resident-evaluation request initially declares its complete
scratch envelope. As exact shard reconstructions become durable, charging that
same materialized space again against current free disk double-counts it. The
observer may subtract a reconstruction only when all of the following are true
for the same five-digit source ordinal:

1. `encode:NNNNN` is in the checkpoint's exact completed-plan prefix and binds
   `bundle/shards/NNNNN.strand`.
2. `attest:NNNNN` is completed and binds the identical archive hash and bytes.
3. `decode:NNNNN` is completed and binds the finalized
   `evaluation/reconstruction/NNNNN.safetensors` file.
4. The reconstruction is a non-symlink regular file with the exact checkpoint
   byte count and stable `lstat`/`fstat` identity. No `.partial` is present.

An existing but uncheckpointed reconstruction counts as zero. Deferred mode
counts every reconstruction as zero. The equations are fixed:

```text
remaining_scratch = max(0, declared_total_scratch - durable_materialized)
remaining_packed = max(0, projected_whole_packed - durable_attested_packed)
required_free = 150,000,000,000 + remaining_scratch + remaining_packed
```

Packed output remains a separate term; it is never hidden inside or subtracted
from scratch. The whole packed-output projection is a required frozen input and
is never inferred from a partial run.

## Integrity and I/O boundary

The request must be exactly `output_root/request.json`; the checkpoint path and
all ordinal artifact paths are derived from that frozen output root. The tool
rejects path traversal, workspace escapes, symlink components, partials,
duplicate source ordinals or plan units, conflicting artifact aliases, unknown
evaluation modes, a reserve other than exactly 150 decimal GB, and either
durable total exceeding its declared budget. Hard-linked aliases and files that
become unlinked during observation are also refused so one inode cannot be
credited twice or survive only through another process's open descriptor.

Request and checkpoint JSON are read through descriptor-relative `O_NOFOLLOW`
directory chains with matching pre/open/post `lstat` and `fstat` identities.
Duplicate JSON keys and
non-finite values are refused. Multi-GB archives and reconstructions are not
content-rehashed: the receipt binds the exact small checkpoint-file SHA-256,
syntax-checks every checkpoint artifact SHA-256/byte pair, and verifies each
payload's stable regular-file identity and size.

The emitted receipt is self-hashed and explicitly records that activation is
forbidden, the queue was not imported or mutated, and requests, checkpoints,
results, and runtime defaults were not changed.

## Read-only use

The command prints the receipt to stdout and does not accept an output path:

```sh
python3.12 tools/condense/doctor_v5_remaining_scratch_ledger.py \
  /absolute/workspace/path/to/strand_ladder/request.json \
  --projected-packed-output-bytes FROZEN_WHOLE_OUTPUT_BYTES
```

This receipt grants no production speed or admission authority. A later
owner-free, reviewed queue generation would need to bind this exact source and
receipt validator before using `remaining_scratch_bytes` in an admission gate.
