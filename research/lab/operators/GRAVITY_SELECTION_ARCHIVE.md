# Gravity selection apparatus archive

The following research-only operators were removed from active HEAD after a
repository-wide caller audit found no production or HCLI entry point. They
were a parallel GLM52 kernel-selection/benchmark/fixture cluster whose only
remaining consumers were registry labels and dedicated tests.

| retired surface | current boundary |
| --- | --- |
| `gravity_bench_lab.py` | matched CPU harness; no production timing claim |
| `gravity_metal_lab_b.py` | Track B research kernel; current decode authority remains `gravity_metal.py` |
| `gravity_real_fixtures.py` | fixture-only support for the retired Track B cluster |
| `gravity_kernel_select.py` | selection matrix with no active caller |
| `gravity_moe_layer.py` | research MoE executor with no active caller |
| `repack_score_index.py` | test-only score-index helper with no live caller |

The dedicated tests and registry-only labels were removed with the cluster.
The implementation history remains recoverable in Git; current packers,
decoders, receipts, and HCLI/AgentOS ownership were not changed.
