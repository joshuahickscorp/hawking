# bit-ledger q2 L=12

weights=357826560 blocks=1397760, total encoded 2.66530 bpw, total deploy 3.16530 bpw

| component | raw_bpw | ent_bpw | recoverable_bpw | H bits/sym | note |
|---|---:|---:|---:|---:|---|
| scale | 0.12500 | 0.04099 | 0.08401 | 10.493 | predictor=prev-block delta, H0=10.55, Hdelta=10.49 |
| sub_scale | 0.18750 | 0.16554 | 0.02196 | 5.297 | predictor=ctx-by-position, H_pool=5.30, H_ctx=5.30, H_super=5.45 |
| init_state | 0.04688 | 0.03899 | 0.00788 | 9.982 | L=12 |
| outl_pos | 0.22584 | 0.07825 | 0.14760 | 7.825 | predictor=gap, H_abs=21.00, H_gap=7.82 |
| outl_val | 0.08000 | 0.05963 | 0.02037 | 5.963 |  |

GATE: scale+sub recoverable = 0.10598 B/w; incl stream-table = 0.60598 B/w; verdict = CLEARS GATE
