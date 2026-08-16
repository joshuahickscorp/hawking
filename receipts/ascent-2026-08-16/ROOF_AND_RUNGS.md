# Roof and rungs — S004 §4

Instrument: `python3 tools/ascent/roof_rungs.py --bytes <B> --ns <ns>`.
Today's table: `python3 tools/ascent/roof_rungs.py --table-today`.

Rungs (complete-token wall): **A** ≤20 ms / ≥50 TPS · **B** ≤10 ms / ≥100 TPS · **C** ≤5 ms / ≥200 TPS · **D** continue toward the measured roof.

Honest decode roof = **411.51 GB/s** (unique-once 512 MiB). Unique-once 1024 MiB = 301.6 GB/s. Reuse band 535.9–637.5 GB/s is cache-resident and is **not** a decode ceiling. Published 819 GB/s was not achieved.

| model | bytes/token | ms/token | TPS | wall GB/s | GPU GB/s | AI flop/B | occ vs roof | dispatch floor (serial CB host, ms) | recon excess (ms) | sync floor (ms) | roof tok/s | frac roof (wall) | fs/weight served | current rung | highest rung at these bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q80_mixed | 2.217 GB | 1170.68 | 0.85 | 1.89 | 2.57 | 3.21 | 0.62% | 69.1 | 858.00 | 123.5 | 185.59 | 0.46% | 328632.4 | below_A | B |
| qwen38 | 13.622 GB | 33.90 | 29.50 | 401.87 | 406.20 | 3.76 | 98.71% | — | 0.43 | 0.4 | 30.21 | 97.66% | 1322.8 | below_A | none (roof < 50 TPS) |
| dsv4f | 5.857 GB | 1037.76 | 0.96 | 5.64 | 14.68 | 4.35 | 3.57% | 28.1 | 384.79 | 28.1 | 70.26 | 1.37% | 81402.3 | below_A | A |

fs_per_weight_served (amortized throughput-derived, NOT latency). fs_per_weight_served is an amortized throughput metric under concurrency, NOT physical femtosecond latency. A single weight's DRAM round trip is ~100 ns. Femtoseconds appear only because thousands of weights are in flight at once.

## Physical-limit audit

| verdict | receipt |
| --- | --- |
| PASS | receipts/ascent-2026-08-16/QWEN38_AT_CEILING_RESOLVED.json |
| PASS_NO_LIMIT_CLAIMED | receipts/ascent-2026-08-16/Q80_MIXED_RECONSTRUCTION_WALL.json |
| PASS_NO_LIMIT_CLAIMED | receipts/ascent-2026-08-16/G001_KERNEL_GAP.json |
| FAIL | receipts/ascent-2026-08-16/TERMINAL_TARGET.json THE_SINGLE_SHARED_BLOCKER |
| FAIL | receipts/ascent-2026-08-16/TERMINAL_TARGET.json machine_reference |
| FAIL | receipts/ascent-2026-08-16/PHYSICAL_FLOOR.json floors.q80_mixed |
| FAIL | receipts/ascent-2026-08-16/PHYSICAL_FLOOR.json answer_per_operation |
| PASS | receipts/ascent-2026-08-16/PHYSICAL_FLOOR.json answer_per_token |
| FAIL | receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json HARD_CONSEQUENCE_1 |
| FAIL | GOAL.md G012 evidence (hawking-femtosecond-ascent) |
| FAIL | SUPERWAVE_STATE.md header (later corrected in-file) |
| SUPERSEDED | QWEN38_ACTIVE_BUDGET_MEASURED.json CORRECTION_TO_MY_OWN_CLAIM (superseded) |

A physical-limit claim requires naming the hardware resource actually at saturation with evidence. 'No further optimization is obvious' is not evidence.

## How to read the rungs

- **qwen38**: GPU is 98.7% of the honest decode ceiling (may claim that kernel is memory-roofed). The *token* is still below rung A, and at 13.622 GB/token the roof is 30.2 TPS, so A is unreachable until bytes drop.
- **q80_mixed**: 0.62% of the roof on GPU matvec. Reconstruction, not bandwidth, is the wall. Current bytes still physically allow rungs A and B (roof 185.6 TPS).
- **dsv4f**: 3.57% of the roof on GPU; token wall is host I/O. Current bytes allow A only (roof 70.3 TPS).
