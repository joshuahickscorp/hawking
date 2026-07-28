# Small formal system training

Local CPU/MPS training of five components on the frozen Ramanujan corpora
(D1–D4, D6, D7). **Not production.** `RAMANUJAN_RESEARCH_AUTHORIZED` stays false.
Math-Preserve is never read.

## Prerequisites

1. Frozen memberships:

   ```bash
   nice -n 15 python3.12 -m ramanujan.data.freeze_memberships
   ```

   Seals `ramanujan/data/corpora/MEMBERSHIP_MANIFEST.json` and re-checks the
   Odyssey contamination barrier (negative control: support-halo exact match).

2. Torch with optional MPS: `~/.grok-vision/bin/python`

## Train + held-out eval

```bash
export PATH="$HOME/.elan/bin:$PATH"
nice -n 15 ~/.grok-vision/bin/python -m ramanujan.train.train_components \
  --epochs 8 --lean-eval-limit 40 --lean-workers 4
```

Max workers for Lean compile checks: **8**. Never touch
`~/Library/Application Support/Hawking/GLM52Gravity/`.

## Artifacts

| Path | Role |
|------|------|
| `../data/corpora/MEMBERSHIP_MANIFEST.json` | sealed train/dev/test by `content_hash` |
| `../data/corpora/FREEZE_RECEIPT.json` | freeze + contamination negative control |
| `HELD_OUT_METRICS.json` | held-out metrics only (test/dev) |
| `TRAINING_RECEIPT.json` | full receipt including histories |
| `checkpoints/*.pt` | small torch weights |

## Components

| Component | Data | Held-out metric |
|-----------|------|-----------------|
| retriever | D3 | recall@k, MRR vs token-overlap baseline |
| formalizer | D1 | first-tactic exact match vs majority |
| prover | D2 | next-tactic exact match vs majority |
| repair | D4 | fix exact match + Lean compile rate |
| value | D2 | closed accuracy + remaining-steps MAE |

A component is **converged** only if the held-out metric **strictly beats** its
baseline. Weights can update without convergence; that is reported honestly.
