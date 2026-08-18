#!/usr/bin/env python3
"""Can a PARALLEL-DECODABLE code keep the entropy win, or does the kernel eat it?

The 0.802 recode ratio on the coherent artifact is real and unshippable: Huffman decode is
serial per symbol, and the geo_tpr64 class needs roughly twenty symbols per nanosecond per
lane. So the question is not "how small" but "how small while a SIMD lane can decode it".

Three candidate forms, all measured on the real q3 payloads, all counting every byte.

  A  FIXED-BITS-PER-TILE. Each tile stores ceil(log2(range+1)) bits per code plus its min.
     Trivially parallel, one shift per lane, no state. Killed analytically before it is
     measured and then measured anyway: the packer scales each group of 64 by that group's
     ABSMAX, so every group contains an element at the extreme by construction and the range
     is always full. Measured here to show the predicted null rather than assert it.

  B  INTERLEAVED rANS, L lanes per tile. Each lane decodes its own stream serially, L lanes
     advance together, so throughput is L symbols per serial step. This is the form the
     kernel can actually run. Costs L final states flushed per tile.

  C  INTERLEAVED rANS WITH ONE SHARED FREQUENCY TABLE for the whole tensor rather than a
     table per tile. Removes the per-tile table entirely and lets the kernel hold the LUT in
     registers. Costs whatever the per-tile distributions differ from the global one, which
     is measured here rather than assumed small.

rANS within a fraction of a percent of the entropy of its quantized frequency model is the
standard result; what is computed here is that model's exact code length, sum over symbols of
count * log2(total/freq), with 12-bit quantized frequencies -- so the number includes the
real loss from quantizing the model, not an idealised entropy.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_entropy_recode import read_container, unpack_codes  # noqa: E402

RUNS = "workspace/campaign/records/runs/qwen38-27b"
PREC = 12          # rANS frequency precision, 2^12 = 4096


def quantized_freqs(hist, prec=PREC):
    """Frequencies scaled to sum to 2^prec, every present symbol at least 1."""
    tot = hist.sum()
    f = np.maximum((hist.astype(np.float64) * (1 << prec) / tot).astype(np.int64), (hist > 0))
    # fix the sum by adjusting the largest bucket, which is what a real encoder does
    f[f.argmax()] += (1 << prec) - f.sum()
    return f


def rans_bits(hist, freqs, prec=PREC):
    """Exact code length of this frequency model over these counts."""
    m = hist > 0
    return float((hist[m] * (prec - np.log2(freqs[m]))).sum())


def fixed_bits(codes, per_tile):
    n_tiles = -(-len(codes) // per_tile)
    total = 0
    for t in range(n_tiles):
        c = codes[t * per_tile:(t + 1) * per_tile]
        rng = int(c.max()) - int(c.min())
        b = max(1, int(np.ceil(np.log2(rng + 1))))
        total += b * len(c)
    return total, n_tiles


def analyse(path, tile_groups, lanes_list):
    hdr, body, disk = read_container(path)
    codes = unpack_codes(body, hdr)
    per_tile = tile_groups * hdr["group_size"]
    n_tiles = -(-len(codes) // per_tile)
    nsym = int(codes.max()) + 1
    fixed_hdr = disk - len(body)
    scales = hdr["scale_bytes"]

    glob = np.bincount(codes, minlength=nsym)
    gfreq = quantized_freqs(glob)

    per_tile_bits = 0.0
    shared_bits = 0.0
    table_bytes = 0
    for t in range(n_tiles):
        c = codes[t * per_tile:(t + 1) * per_tile]
        h = np.bincount(c, minlength=nsym)
        per_tile_bits += rans_bits(h, quantized_freqs(h))
        shared_bits += rans_bits(h, gfreq)
        table_bytes += 2 * int((h > 0).sum())      # 16-bit quantized freq per present symbol

    fb, _ = fixed_bits(codes, per_tile)
    out = {"disk": disk, "elements": hdr["elements"], "tiles": n_tiles,
           "A_fixed": fixed_hdr + scales + n_tiles * 4 + -(-fb // 8)}
    for L in lanes_list:
        flush = n_tiles * L * 4                     # one 32-bit state per lane per tile
        out[f"B_rans_pertile_L{L}"] = (fixed_hdr + scales + n_tiles * 4 + table_bytes
                                       + flush + -(-int(per_tile_bits) // 8))
        out[f"C_rans_shared_L{L}"] = (fixed_hdr + scales + n_tiles * 4 + 2 * nsym
                                      + flush + -(-int(shared_bits) // 8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="mixed-q3mlp-q3attn-v1")
    ap.add_argument("--tile-groups", type=int, default=512)
    ap.add_argument("--lanes", default="8,16,32,64")
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    lanes = [int(x) for x in a.lanes.split(",")]
    root = os.path.join(RUNS, a.artifact)
    segs = sorted(f for f in os.listdir(os.path.join(root, "segments"))
                  if f.startswith("replace_") and f.endswith(".hq38seg"))
    pick = segs[:: max(1, len(segs) // a.sample)][:a.sample]
    agg = {}
    elems = 0
    for f in pick:
        r = analyse(os.path.join(root, "segments", f), a.tile_groups, lanes)
        elems += r["elements"]
        for k, v in r.items():
            if k != "elements":
                agg[k] = agg.get(k, 0) + v
    disk = agg.pop("disk")
    print(f"{a.artifact}, {len(pick)} segments, tile {a.tile_groups} groups, "
          f"{agg.pop('tiles'):,} tiles, {elems:,} elements")
    print(f"incumbent packed bytes {disk:,} = {8*disk/elems:.4f} bits/element\n")
    print(f"{'form':<34}{'bytes':>16}{'ratio':>9}{'b/elem':>9}  parallel decode?")
    rows = sorted(agg.items(), key=lambda kv: kv[1])
    for k, v in rows:
        par = "NO, serial per symbol" if k.startswith("Huff") else (
            "YES, 1 shift per lane" if k == "A_fixed" else
            f"YES, {k.split('_L')[-1]} symbols per serial step")
        print(f"{k:<34}{v:>16,}{v/disk:>9.4f}{8*v/elems:>9.4f}  {par}")
    best = rows[0]
    print(f"\nbest parallel-decodable form: {best[0]} at ratio {best[1]/disk:.4f}")
    print("compare against the SERIAL Huffman recode measured at ratio 0.8020")
    if a.json:
        json.dump({k: v for k, v in agg.items()}, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
