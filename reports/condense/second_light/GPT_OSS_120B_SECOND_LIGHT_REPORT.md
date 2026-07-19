# GPT-OSS-120B :: SECOND-LIGHT PQ BASELINE ARTIFACT

> Second Light proves complete sub-bit artifact construction and accounting. It does not pass the
> functional-quality contract.

## Scientific role

This is the **complete negative-quality baseline** for the Gravity Frontier: a full-scope,
complete-declared candidate artifact of GPT-OSS-120B with exact whole-artifact byte accounting.

It is **not** a successful sub-bit model, **not** a capability-preserving solution, and **not** the
final Event Horizon.

## Result (verified, fail-closed)

| field | value |
| --- | --- |
| rows | 183 / 183 sealed, 0 failed |
| complete physical bits | 89,930,626,766 |
| logical weights | 116,829,156,672 |
| realized whole-artifact BPW | **0.76976** (sub-bit) |
| output | 10.47 GiB (vs 60.77 GiB source, 5.8x) |
| within budget | every row |
| all 183 seals valid | yes (authoritative `eco_common.sealed`) |
| controller replay | accepts 183/183 |

## Quality (honest, negative)

| metric | value |
| --- | --- |
| weight relative error (mean) | ~0.554 (weight-space proxy) |
| true-residual functional output divergence (mean) | ~0.688 |
| capability pass | **NO** |

Sub-bit PQ builds the artifact but does not preserve enough function. This is exactly why the next
campaign must search **representation geometry**, not repeat this artifact.

## Bindings

- source: `openai/gpt-oss-120b @ b5c939de`; tokenizer vocab 201088 + Harmony `chat_template.jinja`
- seed commit: `7f237ed3`; program hash `3a4061f2…`; quality contract `6edc2121…`
- Forge: `gravity_forge.py` (PQ 7-verb + protected islands + `doctor_pq` + CPU/Metal parity)
- controller: `second_light_controller.py` (singleton lease `com.hawking.second_light`)
- Doctor + island reserves budgeted but unspent in this base pass (realized 0.770 < budget 0.928)

## Reproduction

See `GPT_OSS_120B_SECOND_LIGHT_REPRODUCTION.json`. Deterministic per-row seeds; exact assignment;
byte accounting exactly deterministic (CPU authoritative for any MPS recon drift).
