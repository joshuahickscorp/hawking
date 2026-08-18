#!/usr/bin/env python3
"""How much does the COHERENT artifact shrink under a random-access entropy recode?

This is the one move on the sub-1.0 path that costs nothing in quality. Entropy coding the
q3 codes of mixed-q3mlp-q3attn-v1 is LOSSLESS with respect to those codes: the decoder
reproduces the identical integers, so the model is bit-identical to the artifact measured
COHERENT 10/10. Only the storage layout moves. No Doctor re-verification is owed for
correctness -- only a kernel is owed for execution.

Measured on the REAL payloads, not on a distribution model. For each tensor the packed
uniform-group payload is unpacked back to its integer codes, then re-coded as fixed-size
random-access TILES so a GEMV can still address any group directly:

    per tile: an entropy-coded bitstream of that tile's codes
    plus a tile offset table, counted in full
    plus the f16 group scales, unchanged and counted
    plus the existing container header, unchanged and counted

Random access is preserved at TILE granularity, which is what a tiled GEMV needs; it is not
preserved per element, and a scheme that gave up random access entirely would not be
comparable to the incumbent and is not measured here.

The entropy figure is the exact per-tile Huffman cost (a real code built per tile, not the
Shannon bound), because a bound is not an artifact size.
"""
from __future__ import annotations
import argparse, heapq, json, os, struct, sys
from collections import Counter
import numpy as np

RUNS = "workspace/campaign/records/runs/qwen38-27b"
SOURCE_PARAM_COUNT = 26_895_998_464
GROUP = 64


def read_container(path):
    b = open(path, "rb").read()
    magic = b[:8]
    hlen = struct.unpack("<I", b[8:12])[0]
    hdr = json.loads(b[12:12 + hlen])
    return hdr, b[12 + hlen:], len(b)


def unpack_codes(body, hdr):
    """Mirror of lab/operators/ascension_dual_gravity_worker._unpack_unsigned.

    The packer uses bitorder="little" and LSB-first weights. Reading it MSB-first with
    numpy's default bitorder produces plausible-looking but shifted symbols: a q3 stream,
    which can only hold 0..6, came back with symbol 7 at 5.26% and an entropy of 2.9253 bits
    against a nominal 3. That near-uniform histogram made a real entropy win look like 0.87%.
    The symbol-range check below is what caught it and it stays as an assert.
    """
    groups, bits = hdr["groups"], hdr["bits"]
    sb = hdr["scale_bytes"]
    code = np.frombuffer(body[sb:sb + hdr["code_bytes"]], dtype=np.uint8)
    n = groups * hdr["group_size"]
    raw = np.unpackbits(code, bitorder="little")[: n * bits].reshape(n, bits)
    weights = (1 << np.arange(bits, dtype=np.uint8)).astype(np.uint16)
    out = (raw.astype(np.uint16) * weights).sum(1).astype(np.uint16)
    hi = (1 << bits) - 2          # bound = 2^(bits-1)-1, so unsigned tops out at 2*bound
    assert out.max() <= hi, (f"unpacked symbol {out.max()} exceeds the {hi} a q{bits} "
                             f"group-absmax code can produce -- bit order is wrong")
    return out


def huffman_bits(sym):
    """Exact bits of a real Huffman code over these symbols, plus the code table."""
    cnt = Counter(sym.tolist())
    if len(cnt) == 1:
        return 0, 1
    h = [[c, i, [s]] for i, (s, c) in enumerate(cnt.items())]
    heapq.heapify(h)
    depth = {s: 0 for s in cnt}
    while len(h) > 1:
        c1, _, s1 = heapq.heappop(h)
        c2, _, s2 = heapq.heappop(h)
        for s in s1 + s2:
            depth[s] += 1
        heapq.heappush(h, [c1 + c2, id(s1), s1 + s2])
    return sum(cnt[s] * depth[s] for s in cnt), len(cnt)


def recode(path, tile_groups, offset_bytes=4):
    hdr, body, disk = read_container(path)
    codes = unpack_codes(body, hdr)
    g, gs = hdr["groups"], hdr["group_size"]
    per_tile = tile_groups * gs
    n_tiles = -(-len(codes) // per_tile)
    total_bits, tbl_syms = 0, 0
    for t in range(n_tiles):
        chunk = codes[t * per_tile:(t + 1) * per_tile]
        b, k = huffman_bits(chunk)
        total_bits += b
        tbl_syms += k
    coded = -(-total_bits // 8)
    # a per-tile code table is required to decode that tile independently. 1 byte of length
    # per symbol present is the cheapest canonical-Huffman form.
    table = tbl_syms
    index = n_tiles * offset_bytes
    new = coded + table + index + hdr["scale_bytes"] + (disk - len(body))
    return {"disk": disk, "elements": hdr["elements"], "bits": hdr["bits"],
            "code_bytes": hdr["code_bytes"], "scale_bytes": hdr["scale_bytes"],
            "coded": coded, "table": table, "index": index, "tiles": n_tiles,
            "new": new, "ratio": new / disk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="mixed-q3mlp-q3attn-v1")
    ap.add_argument("--tiles", default="8,32,128")
    ap.add_argument("--sample", type=int, default=6)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    root = os.path.join(RUNS, a.artifact)
    segs = sorted(f for f in os.listdir(os.path.join(root, "segments"))
                  if f.startswith("replace_") and f.endswith(".hq38seg"))
    print(f"{a.artifact}: {len(segs)} re-encoded segments on disk")
    pick = segs[:: max(1, len(segs) // a.sample)][:a.sample]
    out = {}
    print(f"\n{'segment':<44}{'tile':>6}{'disk B':>12}{'new B':>12}{'ratio':>8}"
          f"{'b/elem':>9}{'index%':>8}")
    for tg in (int(x) for x in a.tiles.split(",")):
        agg = [0, 0, 0]
        for f in pick:
            r = recode(os.path.join(root, "segments", f), tg)
            agg[0] += r["disk"]; agg[1] += r["new"]; agg[2] += r["elements"]
            nm = f.replace("replace_language_model_model_layers_", "L").replace("_weight.hq38seg", "")
            print(f"{nm:<44}{tg:>6}{r['disk']:>12,}{r['new']:>12,}{r['ratio']:>8.4f}"
                  f"{8*r['new']/r['elements']:>9.4f}{100*r['index']/r['new']:>8.2f}")
        out[tg] = {"disk": agg[0], "new": agg[1], "elements": agg[2],
                   "ratio": agg[1] / agg[0], "bits_per_elem": 8 * agg[1] / agg[2]}
        print(f"{'  SAMPLE TOTAL':<44}{tg:>6}{agg[0]:>12,}{agg[1]:>12,}"
              f"{agg[1]/agg[0]:>8.4f}{8*agg[1]/agg[2]:>9.4f}\n")

    best = min(out, key=lambda k: out[k]["ratio"])
    r = out[best]["ratio"]
    pr = json.load(open(os.path.join(root, "PACK_REPORT.json")))
    cur = pr["complete_physical_bpw"]
    # only the re-encoded segments shrink; copied endpoint/norm segments do not
    enc_bytes = sum(recode(os.path.join(root, "segments", f), best)["disk"] for f in pick)
    print(f"best tile size {best} groups, ratio {r:.4f} on the sampled segments")
    print(f"current complete BPW {cur:.12f}")
    print(f"if that ratio holds over ALL re-encoded segments, and the copied endpoint and")
    print(f"norm segments are unchanged, the recoded artifact lands near "
          f"{cur * r + (1 - r) * 0.4018:.6f} complete BPW")
    print("  (endpoints are 0.4018 BPW and do not shrink; this is an ESTIMATE over the")
    print("   sampled segments, not a packed artifact)")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
