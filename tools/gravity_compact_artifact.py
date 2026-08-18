#!/usr/bin/env python3
"""Compact an artifact so its on-disk bytes equal the bytes its records address.

The mixed-* pack strategy hard-links every segment of its parent and adds replacements for the
organs it re-encodes. Every segment stays REFERENCED, because the tensors that were not
replaced still live in those parent blobs -- so nothing is prunable at segment granularity.
But the replaced tensors' old bytes are still inside those blobs, addressed by nobody:

    mixed-q3mlp-q3attn-r1p2-v1   records address 11,244,907,853
                                 segments on disk 16,904,130,714
                                 dead bytes inside live blobs 5,659,222,861 = 1.683 BPW

Under this goal's own rule -- count every byte under the artifact root -- the honest complete
BPW of that tree is 5.028102, which is WORSE than G0's 4.255955. The declared 3.344708 is what
the records address, and a declared number is not a measurement.

Compaction rewrites each record's bytes into its own segment, so declared and on-disk become
the same number by construction. Then the capability gate decides whether the compacted
artifact is still the model. Nothing is claimed until it does.
"""
from __future__ import annotations
import argparse, os, shutil, struct, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.operators.qwen38_mlp_not_r160_pack import write_catalog, sha256_hex  # noqa: E402

RUNS = "workspace/campaign/records/runs/qwen38-27b"
SOURCE_PARAM_COUNT = 26_895_998_464
RECORD_SIZE = 128


def parse(path):
    b = open(path, "rb").read()
    assert b[:8] == b"HQ38M20\0"
    ver, n_rec, n_seg, _a, name_len, _c = struct.unpack("<IIIIII", b[8:32])
    off = 32
    segs = {}
    for _ in range(n_seg):
        sid, nlen, nbytes, dg = struct.unpack("<HHQ32s", b[off:off + 44]); off += 44
        segs[sid] = {"id": sid, "filename": b[off:off + nlen].decode(),
                     "bytes": nbytes, "sha256": dg.hex()}
        off += nlen
    tbl = b[off:off + n_rec * RECORD_SIZE]
    names = b[off + n_rec * RECORD_SIZE:]
    recs = []
    for i in range(n_rec):
        r = tbl[i * RECORD_SIZE:(i + 1) * RECORD_SIZE]
        noff, nlen, codec, organ, rank, _p = struct.unpack("<IHBBBB", r[:10])
        d0, d1, d2, d3, elems, sid, arank, boff, nb, dg, flags, nfit, bpw = struct.unpack(
            "<IIIIQHHQQ32sIIf", r[12:12 + struct.calcsize("<IIIIQHHQQ32sIIf")])
        recs.append({"name": names[noff:noff + nlen].decode(),
                     "shape": [d for d in (d0, d1, d2, d3)[:rank]],
                     "codec": codec, "organ": organ, "elements": elems,
                     "segment_id": sid, "offset": boff, "nbytes": nb,
                     "sha256": dg.hex(), "flags": flags, "n_fit_rows": nfit,
                     "achieved_rank": arank, "codec_bpw": bpw})
    return segs, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, dst = os.path.join(RUNS, a.artifact), os.path.join(RUNS, a.out)
    segs, recs = parse(os.path.join(src, "catalog.hq38m20"))
    addressed = sum(r["nbytes"] for r in recs)
    on_disk = sum(os.path.getsize(os.path.join(src, "segments", s["filename"]))
                  for s in segs.values())
    print(f"{a.artifact}: {len(recs)} records addressing {addressed:,} bytes")
    print(f"  {len(segs)} segments holding {on_disk:,} bytes on disk")
    print(f"  dead inside live blobs: {on_disk-addressed:,} = "
          f"{8*(on_disk-addressed)/SOURCE_PARAM_COUNT:.6f} BPW\n")

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "segments"))
    cache, out_segs, out_recs, nxt = {}, [], [], 0
    for i, r in enumerate(recs):
        s = segs[r["segment_id"]]
        p = os.path.join(src, "segments", s["filename"])
        if r["offset"] == 0 and r["nbytes"] == os.path.getsize(p):
            fn = s["filename"]                      # already exactly one record's bytes
            if fn not in cache:
                os.link(p, os.path.join(dst, "segments", fn))
                cache[fn] = nxt
                out_segs.append({"id": nxt, "filename": fn, "bytes": r["nbytes"],
                                 "sha256": s["sha256"]})
                nxt += 1
            sid = cache[fn]
        else:
            with open(p, "rb") as fh:
                fh.seek(r["offset"]); blob = fh.read(r["nbytes"])
            fn = "c_" + r["name"].replace(".", "_") + ".hq38seg"
            with open(os.path.join(dst, "segments", fn), "wb") as fh:
                fh.write(blob)
            out_segs.append({"id": nxt, "filename": fn, "bytes": len(blob),
                             "sha256": sha256_hex(blob)})
            sid = nxt; nxt += 1
        out_recs.append({**r, "segment_id": sid, "offset": 0})
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(recs)}")

    write_catalog(Path(dst) / "catalog.hq38m20", out_recs, out_segs)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p) and f != "catalog.hq38m20":
            os.link(p, os.path.join(dst, f))
    tot = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fs in os.walk(dst) for f in fs)
    print(f"\ncompacted {a.out}: {tot:,} bytes = {8*tot/SOURCE_PARAM_COUNT:.12f} complete BPW")
    print(f"  addressed {addressed:,}; on-disk now equals addressed plus catalog and reports")
    print("  NOT a claim until the capability gate runs on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
