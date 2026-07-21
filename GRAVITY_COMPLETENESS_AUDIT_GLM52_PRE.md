# Gravity completeness audit — GLM-5.2 pre-campaign

Frozen boundary: `2026-07-21T20:29:51.121903Z` at `8d634af9702aa965728f057661b5d1fad1883f45`.
Later GLM work is deliberately excluded, so the post-campaign delta cannot rewrite the baseline.

A 5 denotes reproducible evaluation maturity on an axis; it does not imply a capability pass.

| Axis | GPT_OSS_120B | QWEN3_235B | KIMI_K26 | GLM52_PRE |
|---|---:|---:|---:|---:|
| source authority | 5 | 5 | 5 | 1 |
| source precision | 3 | 5 | 3 | 1 |
| logical weight accounting | 5 | 5 | 4 | 1 |
| physical artifact accounting | 5 | 4 | 2 | 1 |
| adapter fidelity | 4 | 4 | 4 | 1 |
| teacher forward fidelity | 4 | 4 | 4 | 0 |
| streaming completeness | 4 | 4 | 4 | 1 |
| resume recovery | 4 | 3 | 4 | 1 |
| data integrity | 3 | 4 | 3 | 0 |
| causal diagnosis | 4 | 5 | 3 | 1 |
| doctor breadth | 4 | 4 | 3 | 1 |
| native studentization | 0 | 0 | 2 | 1 |
| rate exploration | 4 | 5 | 3 | 1 |
| full model artifact | 4 | 4 | 1 | 0 |
| capability evaluation | 5 | 5 | 3 | 0 |
| direct runtime | 4 | 4 | 3 | 0 |
| metal execution | 2 | 2 | 3 | 0 |
| speed efficiency | 4 | 4 | 4 | 1 |
| resource utilization | 4 | 4 | 4 | 1 |
| scientific transfer | 5 | 5 | 4 | 2 |
| reproducibility | 4 | 4 | 3 | 1 |
| **Total / 105** | **81** | **84** | **69** | **16** |

## Honest boundary

- **GPT_OSS_120B:** Official mixed MXFP4-expert/BF16-control parent; complete 0.76976-BPW physical baseline and real full-forward negative, not a BF16-teacher success.
- **QWEN3_235B:** Official BF16 parent and real 94-layer parent-versus-packed negative; Python reference runtime and no retained standalone native complete payload.
- **KIMI_K26:** Official packed-INT4 parent; strongest retained evidence is local F1/F2, with no complete compact capability artifact.
- **GLM52_PRE:** Handoff and planning only at the frozen boundary; no teacher forward, corpus, artifact, Metal execution, or capability result.

Seal: `d298c265b3ecede35fcb3f0809aa004ea2a41bd2c11c1da51b741949e60648bd`.
