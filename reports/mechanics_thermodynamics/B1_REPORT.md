# B1 - bounded_reconstruction (tile decode + tile matmul)

Candidate: gen-M.B0B1M1.block0.expert0  |  tile_rows=512
Energy UNAVAILABLE. Wall MEASURED (contaminated). Mechanics ANALYTICAL.

## mlp1 d8
- quality parity: b0_vs_b1_rel 1.75e-07 (same recon)
- mech (b1_cpu): F32=3.318e+07 Llookup=2.074e+06 Mread=6.741e+07 Klaunch=0 Ttemporary=5.898e+06 bytes
- wall b1_cpu median 58.34029 ms; b1_metal 14.79421 ms
- reconstruction temporary bounded to one [512,cols] tile = 5.898e+06 bytes, lifetime = one tile

## mlp1 d16
- quality parity: b0_vs_b1_rel 1.72e-07 (same recon)
- mech (b1_cpu): F32=3.318e+07 Llookup=1.037e+06 Mread=6.744e+07 Klaunch=0 Ttemporary=5.898e+06 bytes
- wall b1_cpu median 35.95379 ms; b1_metal 9.49838 ms
- reconstruction temporary bounded to one [512,cols] tile = 5.898e+06 bytes, lifetime = one tile

## mlp2 d8
- quality parity: b0_vs_b1_rel 9.9e-08 (same recon)
- mech (b1_cpu): F32=1.659e+07 Llookup=1.037e+06 Mread=3.371e+07 Klaunch=0 Ttemporary=5.898e+06 bytes
- wall b1_cpu median 29.20337 ms; b1_metal 10.43483 ms
- reconstruction temporary bounded to one [512,cols] tile = 5.898e+06 bytes, lifetime = one tile

## mlp2 d16
- quality parity: b0_vs_b1_rel 9.2e-08 (same recon)
- mech (b1_cpu): F32=1.659e+07 Llookup=5.184e+05 Mread=3.374e+07 Klaunch=0 Ttemporary=5.898e+06 bytes
- wall b1_cpu median 18.93846 ms; b1_metal 7.30633 ms
- reconstruction temporary bounded to one [512,cols] tile = 5.898e+06 bytes, lifetime = one tile

