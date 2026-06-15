# Preserved STRAND measured results (from strand/research)

These are the **small measured-result files** rescued from `~/Downloads/strand/research/` before the
regenerable bulk was deleted during the quant-track absorption (2026-06-15). They are the primary
PPL/eval data behind the scorecard in [`../../ABSORB-STATUS.md`](../../ABSORB-STATUS.md) §5.

Each `ppl_*.json` carries its full harness provenance (`harness_key`: device, dtype, `ctx 2048`,
`chunks 64`, `dataset wikitext`, reproducible `harness_key8`), so a number can be traced to its run.

## What was deleted (regenerable — NOT needed for dismantle's future)

~17 GB of model-derived blobs that are reproducible from public models + the absorbed scripts
(`tools/strand/`):

- `isobpw/` (9.7G) — dequant copies of **public** Qwen-0.5B GGUF quants (Q4_K_M/Q3_K_M/Q2_K/IQ3_S) +
  the public GGUFs + STRAND reconstruction `.safetensors`.
- `mp-frontier/` (5.6G) + `down-protect/` (1.9G) — STRAND reconstruction `.safetensors` for the
  mixed-precision / down-projection-protect sweeps.

The **conclusions** of these studies live in the absorbed markdown (`../bit-ledger-*.md`,
`../debias-results.md`, `../STRAND-supercondenser-sprint.md`, etc.); only the raw model dumps were
dropped. dismantle will re-bake fresh `.strand` artifacts from its own product models, so the 0.5B
research dumps have no forward use here.

## What was kept in strand (NOT absorbed — your call)

`strand/research/pv-deep/pv-cooldown.pt` (3.7G) — a **trained** PyTorch checkpoint from the deferred
2-bit selective-PV frontier. Held rather than deleted because it is the one non-regenerable artifact;
it is not needed for dismantle's near-term work (3-bit / hybrid kernel).
