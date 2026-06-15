# bit-ledger q3 L=12

weights=357826560 blocks=1397760, total encoded 3.66530 bpw, total deploy 4.16530 bpw

| component | raw_bpw | ent_bpw | recoverable_bpw | H bits/sym | note |
|---|---:|---:|---:|---:|---|
| scale | 0.12500 | 0.04166 | 0.08334 | 10.665 | predictor=prev-block delta, H0=10.71, Hdelta=10.67 |
| sub_scale | 0.18750 | 0.15535 | 0.03215 | 4.971 | predictor=ctx-by-position, H_pool=4.97, H_ctx=4.97, H_super=5.22 |
| init_state | 0.04688 | 0.03511 | 0.01177 | 8.987 | L=12 |
| outl_pos | 0.22584 | 0.07825 | 0.14760 | 7.825 | predictor=gap, H_abs=21.00, H_gap=7.82 |
| outl_val | 0.08000 | 0.05963 | 0.02037 | 5.963 |  |

GATE: scale+sub recoverable = 0.11549 B/w; incl stream-table = 0.61549 B/w; verdict = CLEARS GATE
