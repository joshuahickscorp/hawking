#!/usr/bin/env python3
"""Prune an artifact to the segments its catalog actually references, then prove it still runs.

gravity_bpw counts every byte under the artifact root, which is the correct rule. Under it the
mixed-* artifacts are far larger than their pack reports declare:

    uniform-q4-v1               declared 4.252735  honest 4.255955   gap 0.003219
    mixed-q3mlp-v1              declared 3.613811  honest 4.153458   gap 0.539647
    mixed-q3mlp-q3attn-v1       declared 3.344708  honest 5.028102   gap 1.683393

The cause is the pack strategy, not a leak: each pack hard-links EVERY segment of its parent
and then adds replacements for the organs it re-encodes. The superseded parent segments stay
in the tree. They cost no extra disk, being hardlinks, but they are bytes under the root and
the rule counts them.

A declared number is not a measurement. Either those bytes are required, in which case the
honest BPW is 5.03 and the artifact is worse than G0 -- or they are dead, in which case
deleting them must leave an artifact that still runs. This builds the pruned tree and the
capability gate decides. Nothing here is asserted about which.

The pruned copy is built with hardlinks, so it costs no disk, and the source artifact is never
modified.
"""
from __future__ import annotations
import argparse, json, os, shutil, struct, sys

RUNS = "workspace/campaign/records/runs/qwen38-27b"
SOURCE_PARAM_COUNT = 26_895_998_464
RECORD_SIZE = 128


def parse_catalog(path):
    b = open(path, "rb").read()
    assert b[:8] == b"HQ38M20\0", f"bad magic {b[:8]!r}"
    ver, n_rec, n_seg, _a, name_len, _b = struct.unpack("<IIIIII", b[8:32])
    off = 32
    segs, used = {}, set()
    for _ in range(n_seg):
        sid, nlen, nbytes, _dg = struct.unpack("<HHQ32s", b[off:off + 44])
        off += 44
        segs[sid] = {"name": b[off:off + nlen].decode(), "bytes": nbytes}
        off += nlen
    tbl = b[off:off + n_rec * RECORD_SIZE]
    for i in range(n_rec):
        r = tbl[i * RECORD_SIZE:(i + 1) * RECORD_SIZE]
        sid = struct.unpack("<H", r[36:38])[0]
        used.add(sid)
    return segs, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = os.path.join(RUNS, a.artifact)
    dst = os.path.join(RUNS, a.out)
    segs, used = parse_catalog(os.path.join(src, "catalog.hq38m20"))
    print(f"{a.artifact}: catalog lists {len(segs)} segments, records reference {len(used)}")
    keep = {segs[i]["name"] for i in used if i in segs}
    sd = os.path.join(src, "segments")
    present = set(os.listdir(sd))
    dead = present - keep
    kb = sum(os.path.getsize(os.path.join(sd, f)) for f in keep if f in present)
    db = sum(os.path.getsize(os.path.join(sd, f)) for f in dead)
    print(f"  present {len(present)}  referenced {len(keep & present)}  UNREFERENCED {len(dead)}")
    print(f"  referenced bytes {kb:,}   unreferenced bytes {db:,} "
          f"= {8*db/SOURCE_PARAM_COUNT:.6f} BPW")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "segments"))
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            os.link(p, os.path.join(dst, f))
    for f in keep & present:
        os.link(os.path.join(sd, f), os.path.join(dst, "segments", f))
    tot = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fs in os.walk(dst) for f in fs)
    print(f"\npruned tree {a.out}: {tot:,} bytes = {8*tot/SOURCE_PARAM_COUNT:.12f} complete BPW")
    print("hardlinked, so it costs no disk and the source artifact is untouched.")
    print("It is not a measurement until the capability gate says it still runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
