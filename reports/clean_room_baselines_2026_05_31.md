# Clean-room baselines — 2026-05-31 (MEASURED, contamination-free)

Run via `tools/bench/clean_room_batch.sh` with **Claude fully quit** (pre-flight
gates passed: no Claude.app/CLI/slm, quiet CPU). These are **absolute measured**
numbers (not in-session estimates) — the contamination rule does not apply.
Two consecutive clean runs; Section A reproduced across both.

## A — Q3_K sub-Q4 byte-cut: **NO-GO** (re-confirmed)
- f32-predec-Q3 best-shape: **33.3 then 34.7 GB/s = 22–23% of 150 GB/s peak** (vs the ~50% GO bar).
- Decisive tell: Q3_K is **slower in absolute µs than Q4_predec on the FFN shapes** that dominate
  decode (ffn-up −37%, ffn-down −35%); it only marginally *holds* on the tiny attn-square shape
  (+5.7%, 7–9 GB/s). So Q3_K GEMV is **compute/residual-bound** (inline 6-bit scale + hmask), not
  bandwidth-bound — fewer bytes buy no tps.
- **Verdict:** byte-cut axis routes through **QTIP** only (`plans/qtip_bytecut_design_2026_05_31.md`).
  Q3_K stays footprint-only. Kill: `reports/dead_levers.md` "Q3_K sub-Q4 decode byte-cut" (Type-1).

## B — decode-tps anchor: **RESOLVED to ~31** (bible §3.0 Correction 2)
- Clean **dec_tps = 29.12** (greedy temp=0, 256 tok, locked fast-path, `nice taskpolicy`).
- vs **~31** recent anchor: **−6.1%**; vs **~39** old anchor: **−25.3%**.
- **The ~31 anchor is real; the ~39 envelope was optimistic.** Consistent with A1 (paired 30.94→31.55)
  and A4 (31.0 median). **Caveat:** single clean run — the canonical figure is a thermal-protocol
  median (`clean_bench.sh` ×N); expect **~29–31**.
- **Consequence:** every "% of the way to ~50" in bible §3's envelope is drawn off the superseded ~39
  anchor and must be **re-projected from ~31** at an attended pass. The *anchor* is no longer ambiguous;
  the *envelope re-projection* is the remaining attended task.

## C — energy baseline: **0.17 J/token** (§8 L4.2)
- **J/token = 0.1702** at 28.63 tps: decode_wall 8.94 s, avg package power **4.87 W** (CPU+GPU+ANE),
  avg **GPU power 3.73 W**, decode energy 43.57 J (macmon, sudo-free).
- First clean measurement of the "sips power" axis — the floor future energy levers measure against.

## Forward read
- **Byte-cut = QTIP or nothing** (Q3 dead). QTIP goes through its offline quality oracle (QTIP-from-f16
  recon-RMSE vs Q4_K at 3 bits) before any kernel.
- **Decode is ~29–31 tps at the kernel ceiling** (§3.0); the real gap to a ~50 dense target is *larger*
  than the ~39 anchor implied. Forward tps = **spec** (draft tuning, once the pruned-Q4K batched-verify
  fast-path unblocks it) or **QTIP** (quality-gated) — not decode kernels.
- **Energy** now has a measured floor (0.17 J/tok) for the §8 L4.2 axis.
