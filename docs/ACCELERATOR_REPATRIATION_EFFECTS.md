# Accelerator repatriation effects

`receipts/headless/ACCELERATOR_REPATRIATION_EFFECTS.json` is the canonical
ledger for architectural ideas imported into Hawking Accelerator. It binds
each source school and behavior to a physical invariant, Hawking primitive,
implementation, target model/backend, scoped evidence, measured-result
boundary, and falsifier.

The ledger distinguishes three levels:

- `SPECIMEN_IMPLEMENTATION`: exact model- or shape-specific code.
- `ACCELERATOR_PRIMITIVE`: a reusable physical behavior with explicit scope.
- `PHYSICAL_LAW`: reserved for repeated protected transfer evidence; none is
  currently claimed.

Validate or regenerate it with:

```console
python3 tools/accelerator/repatriation_effects.py \
  --emit receipts/headless/ACCELERATOR_REPATRIATION_EFFECTS.json

python3 tools/accelerator/repatriation_effects.py \
  --validate receipts/headless/ACCELERATOR_REPATRIATION_EFFECTS.json
```

The physical qualification queue carries the per-candidate scope tags and
transfer evidence. A shared runtime seam is `GENERIC_CANDIDATE` until an
integrated protected cross-model/backend A/B proves transfer; genericity is
never inferred from a clean abstraction alone. The effects ledger also emits
an HWIR-facing path through the existing atlas/HWIR pipeline, while keeping
FPGA and ASIC timing claims explicitly absent.
