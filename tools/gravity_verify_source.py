#!/usr/bin/env python3
"""Re-verify the GRAVITY-1 raw source pin against the bytes on disk.

A pinned manifest nobody can recheck is a claim, not a pin. This recomputes every
hash and re-derives the parameter census from real safetensors headers, then diffs
against the pin. Exit 0 only if everything matches.

Usage:  python3 tools/gravity_verify_source.py [--quick]
        --quick skips shard content hashing (sizes only); use for a fast smoke test,
        never as the evidence for source authority.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, struct, sys

PIN = "workspace/superwave/g1/GRAVITY1_SOURCE_PIN.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def census(root):
    lang = other = 0
    nl = no = 0
    for f in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            e = 1
            for d in meta["shape"]:
                e *= d
            if name.startswith("language_model."):
                lang += e
                nl += 1
            else:
                other += e
                no += 1
    return lang, nl, other, no


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pin", default=PIN)
    a = ap.parse_args()

    pin = json.load(open(a.pin))
    root = pin["source_root"]
    fails = []

    if not os.path.isdir(root):
        print(f"FAIL source_root missing: {root}")
        return 1

    for name, rec in sorted(pin["shards"].items()):
        p = os.path.join(root, name)
        if not os.path.exists(p):
            fails.append(f"missing shard {name}")
            continue
        sz = os.path.getsize(p)
        if sz != rec["bytes"]:
            fails.append(f"{name} size {sz} != pinned {rec['bytes']}")
            continue
        if not a.quick:
            d = sha256(p)
            if d != rec["sha256"]:
                fails.append(f"{name} sha256 {d[:16]} != pinned {rec['sha256'][:16]}")

    for name, rec in sorted(pin["metadata_files"].items()):
        p = os.path.join(root, name)
        if not os.path.exists(p):
            fails.append(f"missing metadata {name}")
            continue
        d = sha256(p)
        if d != rec["sha256"]:
            fails.append(f"{name} sha256 {d[:16]} != pinned {rec['sha256'][:16]}")

    lang, nl, other, no = census(root)
    pc = pin["parameter_census"]
    for got, want, what in (
        (lang, pc["language_elems"], "language_elems"),
        (nl, pc["language_tensors"], "language_tensors"),
        (other, pc["other_elems"], "other_elems"),
        (no, pc["other_tensors"], "other_tensors"),
    ):
        if got != want:
            fails.append(f"{what} {got} != pinned {want}")

    if pc["bpw_denominator"] != pc["language_elems"]:
        fails.append("bpw_denominator is not the language parameter count")

    if fails:
        print(f"SOURCE PIN FAIL ({len(fails)})")
        for f in fails:
            print("  " + f)
        return 1

    mode = "sizes only" if a.quick else "full content hashes"
    print(f"SOURCE PIN VERIFIED ({mode})")
    print(f"  shards            {len(pin['shards'])}")
    print(f"  metadata files    {len(pin['metadata_files'])}")
    print(f"  language elems    {lang} across {nl} tensors")
    print(f"  other elems       {other} across {no} tensors")
    print(f"  bpw denominator   {pc['bpw_denominator']}")
    print(f"  upstream          {pin['upstream']['base_model']} @ {pin['upstream']['base_revision']}")
    print(f"  variant           {pin['upstream']['variant']} / {pin['upstream']['abliteration_method']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
