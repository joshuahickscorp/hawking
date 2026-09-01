# NR/NX artifact contract

Gravity is a search/research process, not an artifact class or file format.
The active Hawking storage boundary is:

```text
SourceSpecimen -> Doctor -> Gravity search -> NR (.nr) -> Noetic compiler -> NX (.nx)
```

NR is the transient, portable representation space. It owns representation
identity, shard maps, codec families, semantic provenance, and portable kernel
requirements. It must not contain a machine, shader, dispatch geometry,
occupancy, residency, TPS, or token timing claim. The existing
`tools/nr_container.py` validator enforces this boundary.

NX is the final machine-bound Noetic executable derived from an NR. It owns the
compiled kernel bindings, layout, scheduling, residency, machine genome, and
loadability checks. `tools/nx_genome.py` binds an NX to the exact NR content
hash and rejects a mismatched machine genome.

The `.gravity` suffix is retained only as a compatibility view for sealed
historical artifacts. New NR shards discovered by Gravity should be emitted as `.nr`; a promoted
executable should be emitted as `.nx`. Extension recognition is backward
compatible, but no historical file is renamed in place.
