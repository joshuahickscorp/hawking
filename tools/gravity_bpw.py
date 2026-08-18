#!/usr/bin/env python3
"""Complete effective BPW accounting for a Gravity artifact.

Counts EVERY byte the artifact needs, not the payload a pack report declares.
A prior Qwen3.8 pack reported 3.6138 BPW while carrying a 1.814 GB leftover
directory it never counted; the honest figure was ~4.15.

complete_effective_bpw = 8 * total_bytes / SOURCE_PARAM_COUNT

SOURCE_PARAM_COUNT is the ORIGINAL language parameter count, always -- never the
candidate's own degrees of freedom. A representation with fewer explicit
parameters must still normalise against the source or the number is not
comparable across candidates.
"""
from __future__ import annotations
import argparse, json, os, sys

# language_model.* elems, re-derived from bf16 shard headers 2026-08-17
SOURCE_PARAM_COUNT = 26_895_998_464


def walk(root):
    """Every regular file under root, deduped by (dev, inode).

    Hardlinked shards are counted once: two names for the same bytes are one
    physical cost. Symlinks pointing outside root are followed and counted,
    because a representation that lives behind a symlink is still required.
    """
    seen, files = set(), []
    for dirpath, _, names in os.walk(root, followlinks=True):
        for n in names:
            p = os.path.join(dirpath, n)
            try:
                st = os.stat(p)
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            files.append((p, st.st_size))
    return files


def account(root, declared=None, extra_roots=()):
    files = walk(root)
    for e in extra_roots:
        files += walk(e)
    total = sum(sz for _, sz in files)

    by_ext = {}
    for p, sz in files:
        by_ext[os.path.splitext(p)[1] or "(none)"] = by_ext.get(os.path.splitext(p)[1] or "(none)", 0) + sz

    out = {
        "root": os.path.abspath(root),
        "extra_roots": [os.path.abspath(e) for e in extra_roots],
        "file_count": len(files),
        "total_bytes": total,
        "source_param_count": SOURCE_PARAM_COUNT,
        "complete_effective_bpw": 8 * total / SOURCE_PARAM_COUNT,
        "bytes_by_ext": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])),
        "largest": sorted(((sz, p) for p, sz in files), reverse=True)[:10],
    }
    if declared is not None:
        out["declared_payload_bytes"] = declared
        out["declared_bpw"] = 8 * declared / SOURCE_PARAM_COUNT
        out["uncounted_bytes"] = total - declared
        out["uncounted_bpw"] = 8 * (total - declared) / SOURCE_PARAM_COUNT
    return out


def declared_from_pack_report(root):
    """Payload a pack report claims, for the uncounted-bytes comparison."""
    for name in ("PACK_REPORT.json", "manifest.json"):
        p = os.path.join(root, name)
        if not os.path.exists(p):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        for k in ("tensor_payload_bytes", "payload_bytes", "total_size"):
            if isinstance(j.get(k), int):
                return j[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--extra", action="append", default=[],
                    help="additional required directory (e.g. a segments dir in another artifact)")
    ap.add_argument("--declared", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    declared = a.declared if a.declared is not None else declared_from_pack_report(a.root)
    r = account(a.root, declared, a.extra)

    if a.json:
        print(json.dumps(r, indent=2))
        return 0

    print(f"root                    {r['root']}")
    for e in r["extra_roots"]:
        print(f"extra root              {e}")
    print(f"files (inode-deduped)   {r['file_count']}")
    print(f"total bytes             {r['total_bytes']}")
    print(f"source param count      {r['source_param_count']}")
    print(f"COMPLETE EFFECTIVE BPW  {r['complete_effective_bpw']:.12f}")
    if declared is not None:
        print(f"declared payload        {r['declared_payload_bytes']} -> {r['declared_bpw']:.12f} BPW")
        print(f"UNCOUNTED               {r['uncounted_bytes']} -> {r['uncounted_bpw']:.12f} BPW")
        if r["uncounted_bpw"] > 0.001:
            print(f"WARNING: {r['uncounted_bpw']:.6f} BPW is not in the declared payload")
    print("largest files:")
    for sz, p in r["largest"][:5]:
        print(f"  {sz:>14}  {p}")
    return 0


def demo():
    """Self-check: a sidecar outside the declared payload must be counted."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "tensors"))
        with open(os.path.join(d, "tensors", "w.bin"), "wb") as f:
            f.write(b"\0" * 1000)
        # a leftover the pack report does not declare
        os.makedirs(os.path.join(d, "leftover"))
        with open(os.path.join(d, "leftover", "extra.bin"), "wb") as f:
            f.write(b"\0" * 500)
        r = account(d, declared=1000)
        assert r["total_bytes"] == 1500, r["total_bytes"]
        assert r["uncounted_bytes"] == 500, r["uncounted_bytes"]
        # hardlink must not double count
        os.link(os.path.join(d, "tensors", "w.bin"), os.path.join(d, "alias.bin"))
        r2 = account(d, declared=1000)
        assert r2["total_bytes"] == 1500, r2["total_bytes"]
        print("gravity_bpw demo: PASS (sidecar counted, hardlink deduped)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        sys.exit(main())
