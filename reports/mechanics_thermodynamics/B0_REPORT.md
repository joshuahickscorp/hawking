# B0 - direct_compact_baseline (Generation-F path)

Candidate: gen-M.B0B1M1.block0.expert0  |  generated 2026-07-19T04:46:54Z

Energy: UNAVAILABLE (no sudo powermetrics). Wall time MEASURED but contaminated_by_concurrent_cpu_load (MoP ~24 procs). Mechanics ANALYTICAL. CPU authoritative.

## mlp1 d8  shape=[5760, 2880] base_bpw=0.50099
- quality rel_error vs dense: mean 0.777251 (min 0.770731 max 0.783768)
- mech (b0_compact_cpu): F32=3.318e+07 Llookup=2.074e+06 Mread=6.741e+07 Klaunch=0 Ttemporary=8.294e+06 bytes
- wall b0_compact_cpu median 41.47883 ms; b0_dense_cpu 1.89158 ms; b0_dense_metal 1.00946 ms
- no_dense_shadow: True (peak temp b0=8.294e+06 < full_dense 6.636e+07)

## mlp1 d16  shape=[5760, 2880] base_bpw=0.5158
- quality rel_error vs dense: mean 0.752467 (min 0.743499 max 0.762822)
- mech (b0_compact_cpu): F32=3.318e+07 Llookup=1.037e+06 Mread=6.744e+07 Klaunch=0 Ttemporary=1.659e+07 bytes
- wall b0_compact_cpu median 21.32767 ms; b0_dense_cpu 1.899 ms; b0_dense_metal 0.88742 ms
- no_dense_shadow: True (peak temp b0=1.659e+07 < full_dense 6.636e+07)

## mlp2 d8  shape=[2880, 2880] base_bpw=0.50198
- quality rel_error vs dense: mean 0.822678 (min 0.82192 max 0.823474)
- mech (b0_compact_cpu): F32=1.659e+07 Llookup=1.037e+06 Mread=3.371e+07 Klaunch=0 Ttemporary=4.147e+06 bytes
- wall b0_compact_cpu median 21.22075 ms; b0_dense_cpu 0.93742 ms; b0_dense_metal 0.97483 ms
- no_dense_shadow: True (peak temp b0=4.147e+06 < full_dense 3.318e+07)

## mlp2 d16  shape=[2880, 2880] base_bpw=0.5316
- quality rel_error vs dense: mean 0.491945 (min 0.491349 max 0.492154)
- mech (b0_compact_cpu): F32=1.659e+07 Llookup=5.184e+05 Mread=3.374e+07 Klaunch=0 Ttemporary=8.294e+06 bytes
- wall b0_compact_cpu median 13.28704 ms; b0_dense_cpu 0.95554 ms; b0_dense_metal 0.99721 ms
- no_dense_shadow: True (peak temp b0=8.294e+06 < full_dense 3.318e+07)

