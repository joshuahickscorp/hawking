# GLM-5.2 preliminary streaming schedule

Status: **PRELIMINARY_DEPENDENCY_COMPLETE_PENDING_XET_AUTOTUNE**

- Immutable source: `zai-org/GLM-5.2@b4734de4facf877f85769a911abafc5283eab3d9`
- Windows: `20`
- Preliminary shard target: `26`
- Maximum resident shards in one dependency window: `26`
- Every official shard scheduled exactly once: `282/282`
- Planned refetches: `0`

This is a dependency-correct disk admission plan, not an Xet throughput result. The Xet autotuner may resize windows after measuring APFS allocation, reconstruction scratch, swap, thermals, and heavy-lane regression.

| Window | Organs | Resident shards | New shards | Carry out | New bytes |
|---|---:|---:|---:|---:|---:|
| `W000` | 8 | 23 | 23 | 11 | 123281520256 |
| `W001` | 3 | 25 | 14 | 15 | 69960060600 |
| `W002` | 3 | 26 | 11 | 15 | 58975577336 |
| `W003` | 3 | 26 | 11 | 15 | 58975577320 |
| `W004` | 3 | 26 | 11 | 15 | 58975577336 |
| `W005` | 3 | 25 | 10 | 14 | 53615592008 |
| `W006` | 3 | 25 | 11 | 14 | 58849747592 |
| `W007` | 3 | 25 | 11 | 14 | 58975577320 |
| `W008` | 3 | 24 | 10 | 12 | 53609170480 |
| `W009` | 4 | 26 | 14 | 12 | 75062678728 |
| `W010` | 3 | 23 | 11 | 12 | 58981998848 |
| `W011` | 4 | 26 | 14 | 10 | 75056619104 |
| `W012` | 4 | 25 | 15 | 10 | 80423026056 |
| `W013` | 4 | 23 | 13 | 8 | 69696271896 |
| `W014` | 5 | 26 | 18 | 8 | 96503705904 |
| `W015` | 5 | 26 | 18 | 6 | 96504067824 |
| `W016` | 5 | 24 | 18 | 6 | 96510127464 |
| `W017` | 5 | 23 | 17 | 4 | 91143720616 |
| `W018` | 6 | 26 | 22 | 4 | 117951154616 |
| `W019` | 4 | 14 | 10 | 0 | 53615616104 |

Schedule seal: `58eda0a6e6c750e9fcdd4850457a443417ee883c0256a9eea43067a2d00ea9d3`.
