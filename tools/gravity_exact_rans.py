#!/usr/bin/env python3
"""EXACT complete BPW of the interleaved-rANS form, over every segment. No sampling.

The 2.508 figure was an estimate over 8 of 496 segments, and sampling was its main caveat.
For the SHARED-TABLE form that caveat is removable rather than merely reducible: with one
frequency model for the whole artifact, the coded length is

    sum over symbols of count(sym) * (prec - log2 freq(sym))

which needs only a histogram, not a per-tile encode. So the exact number is one bincount per
segment. Only the per-tile flush states and the tile offset table depend on tile geometry, and
those are arithmetic.

Two table scopes are measured, because they differ in what the KERNEL has to hold:
    per-tensor   one table per segment; the kernel loads a table per tensor
    per-artifact one table for all 496 segments; the kernel holds seven entries in registers
                 for the entire decode and never loads a table again

Everything is counted: coded stream, table(s), per-lane flushed states, tile offset table, the
unchanged f16 group scales, the unchanged container header, and every byte of the copied
segments that this recode does not touch -- the endpoint tables and norms are read off disk
and added at full size, not assumed.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_entropy_recode import read_container, unpack_codes  # noqa: E402
from gravity_parallel_code import quantized_freqs               # noqa: E402

RUNS = "workspace/campaign/records/runs/qwen38-27b"
SOURCE_PARAM_COUNT = 26_895_998_464
PREC = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="mixed-q3mlp-q3attn-r1p2-v1")
    ap.add_argument("--tile-groups", type=int, default=512)
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    root = os.path.join(RUNS, a.artifact)
    segdir = os.path.join(root, "segments")
    files = sorted(os.listdir(segdir))
    enc = [f for f in files if f.startswith("replace_") and f.endswith(".hq38seg")]
    other = [f for f in files if f not in enc]
    print(f"{a.artifact}: {len(enc)} re-encoded segments, {len(other)} copied segments")

    per_seg = []
    hist_all = None
    for i, f in enumerate(enc):
        hdr, body, disk = read_container(os.path.join(segdir, f))
        codes = unpack_codes(body, hdr)
        nsym = (1 << hdr["bits"]) - 1
        h = np.bincount(codes, minlength=nsym).astype(np.int64)
        hist_all = h if hist_all is None else hist_all + h
        per_tile = a.tile_groups * hdr["group_size"]
        per_seg.append({"f": f, "disk": disk, "hist": h, "elements": hdr["elements"],
                        "scale_bytes": hdr["scale_bytes"],
                        "fixed": disk - len(body),
                        "tiles": -(-len(codes) // per_tile)})
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(enc)}")

    def coded(h, freq):
        m = h > 0
        return float((h[m] * (PREC - np.log2(freq[m]))).sum())

    gfreq = quantized_freqs(hist_all, PREC)
    res = {}
    for scope in ("per-artifact", "per-tensor"):
        bits = 0.0
        overhead = 0
        for s in per_seg:
            fq = gfreq if scope == "per-artifact" else quantized_freqs(s["hist"], PREC)
            bits += coded(s["hist"], fq)
            overhead += (s["fixed"] + s["scale_bytes"] + s["tiles"] * 4
                         + s["tiles"] * a.lanes * 4)
            if scope == "per-tensor":
                overhead += 2 * int((s["hist"] > 0).sum())
        if scope == "per-artifact":
            overhead += 2 * int((hist_all > 0).sum())
        res[scope] = -(-int(bits) // 8) + overhead

    enc_disk = sum(s["disk"] for s in per_seg)
    other_disk = sum(os.path.getsize(os.path.join(segdir, f)) for f in other)
    extra = sum(os.path.getsize(os.path.join(root, f)) for f in os.listdir(root)
                if os.path.isfile(os.path.join(root, f)))
    cur_total = enc_disk + other_disk + extra
    print(f"\non-disk now: re-encoded {enc_disk:,} + copied {other_disk:,} + catalog/report "
          f"{extra:,} = {cur_total:,}")
    print(f"complete BPW now: {8*cur_total/SOURCE_PARAM_COUNT:.12f}")
    print(f"\nglobal symbol histogram: {hist_all.tolist()}")
    p = hist_all / hist_all.sum()
    print(f"global entropy: {float(-(p[p>0]*np.log2(p[p>0])).sum()):.4f} bits\n")
    print(f"{'table scope':<16}{'recoded bytes':>16}{'total bytes':>16}{'complete BPW':>16}{'ratio':>8}")
    for scope, v in res.items():
        tot = v + other_disk + extra
        print(f"{scope:<16}{v:>16,}{tot:>16,}{8*tot/SOURCE_PARAM_COUNT:>16.12f}"
              f"{v/enc_disk:>8.4f}")
    best = min(res, key=res.get)
    tot = res[best] + other_disk + extra
    print(f"\nEXACT over all {len(enc)} re-encoded segments, {a.lanes} lanes, tile "
          f"{a.tile_groups} groups.")
    print(f"copied endpoint and norm segments counted at full size, not assumed.")
    print(f"BEST: {best} table -> {8*tot/SOURCE_PARAM_COUNT:.12f} complete BPW")
    if a.json:
        json.dump({k: int(v) for k, v in res.items()} |
                  {"other_disk": other_disk, "extra": extra, "enc_disk": enc_disk},
                  open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
