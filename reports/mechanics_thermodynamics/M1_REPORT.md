# M1 - lookup_linear_pq (activation->codeword tables, index accumulation)

Candidate: gen-M.B0B1M1.block0.expert0
Table build T[c,s,q]=<C_{s,q}, x_{c,s}> (2*cols*k mults, rows-independent); accumulate y_i=sum_{c,s} T[c,s,q_{i,s}] via index lookups (no dense reconstruction, no accumulation multiplies).
Energy UNAVAILABLE. Wall MEASURED (contaminated). CPU numpy authoritative; MPS variant parity-checked.

## mlp1 d8  (sub=8, S=8, k=16)
- mech (m1_cpu): F32=2.166e+06 Llookup=2.074e+06 Mread=9.345e+06 Klaunch=0 Ttemporary=1.037e+06 bytes
- wall m1_cpu median 16.71904 ms; m1_metal 12.31071 ms; b1_cpu 58.34029 ms; b0_compact_cpu 41.47883 ms
- relative: m1_cpu/b1_cpu = 0.2866, m1_cpu/b0_compact = 0.4031
- M1 vs B1 agreement: rel 1.83e-07 within_tol True
- M1 CPU vs Metal: rel 1.4e-07 within_tol True
- verdict: M1 arithmetic (F32=2.166e+06) << B0/B1 (F32=3.318e+07); wall result reported above (honest, positive or negative)

## mlp1 d16  (sub=16, S=4, k=256)
- mech (m1_cpu): F32=2.511e+06 Llookup=1.037e+06 Mread=5.228e+06 Klaunch=0 Ttemporary=1.037e+06 bytes
- wall m1_cpu median 7.42925 ms; m1_metal 6.51779 ms; b1_cpu 35.95379 ms; b0_compact_cpu 21.32767 ms
- relative: m1_cpu/b1_cpu = 0.2066, m1_cpu/b0_compact = 0.3483
- M1 vs B1 agreement: rel 1.81e-07 within_tol True
- M1 CPU vs Metal: rel 1.2e-07 within_tol True
- verdict: M1 arithmetic (F32=2.511e+06) << B0/B1 (F32=3.318e+07); wall result reported above (honest, positive or negative)

## mlp2 d8  (sub=8, S=8, k=16)
- mech (m1_cpu): F32=1.129e+06 Llookup=1.037e+06 Mread=4.679e+06 Klaunch=0 Ttemporary=5.184e+05 bytes
- wall m1_cpu median 8.34917 ms; m1_metal 12.66892 ms; b1_cpu 29.20337 ms; b0_compact_cpu 21.22075 ms
- relative: m1_cpu/b1_cpu = 0.2859, m1_cpu/b0_compact = 0.3934
- M1 vs B1 agreement: rel 1.1e-07 within_tol True
- M1 CPU vs Metal: rel 9e-08 within_tol True
- verdict: M1 arithmetic (F32=1.129e+06) << B0/B1 (F32=1.659e+07); wall result reported above (honest, positive or negative)

## mlp2 d16  (sub=16, S=4, k=256)
- mech (m1_cpu): F32=1.993e+06 Llookup=5.184e+05 Mread=2.636e+06 Klaunch=0 Ttemporary=5.184e+05 bytes
- wall m1_cpu median 4.0025 ms; m1_metal 6.85304 ms; b1_cpu 18.93846 ms; b0_compact_cpu 13.28704 ms
- relative: m1_cpu/b1_cpu = 0.2113, m1_cpu/b0_compact = 0.3012
- M1 vs B1 agreement: rel 8.4e-08 within_tol True
- M1 CPU vs Metal: rel 5e-08 within_tol True
- verdict: M1 arithmetic (F32=1.993e+06) << B0/B1 (F32=1.659e+07); wall result reported above (honest, positive or negative)

