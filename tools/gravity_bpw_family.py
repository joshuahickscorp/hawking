#!/usr/bin/env python3
"""G042: eight BPW numbers, because "the BPW" has been three different quantities.

This campaign has already been bitten twice by treating BPW as one number. The
density leader was quoted at 3.344772 for weeks while its directory held 5.028102
(F4), and the honest G0 figure needed correcting from 4.252735 to 4.255955 because
10,823,719 bytes were uncounted. Both are the same mistake: stored, addressed and
active are different quantities and only one word was being used for all of them.

Eight axes, each defined by where the bytes actually are:

  STORED_BPW               every byte under the artifact root
  ACTIVE_BPW_PER_TOKEN     what the runtime addresses to produce one token
  DRAM_BPW_PER_TOKEN       of that, what must come from DRAM
  CACHE_BPW_PER_TOKEN      of that, what is served from cache
  GENERATED_BPW_EQUIVALENT structure computed rather than stored
  CORRECTION_BPW           correction planes over a base representation
  SHARED_BPW               structure stored once and used by many tensors
  STATE_BPW_EQUIVALENT     KV/state at a given context, in the same units

The verify line is the point: a program with high stored and low active must be
distinguishable from the reverse, and if all eight collapse to one number the
decomposition is decorative. So an UNCOMPACTED artifact is included deliberately --
it is the high-stored/low-active case, and it is not hypothetical, it is what the
campaign was quoting.

  ./tools/gravity_bpw_family.py --candidate uniform-q4-v1 \
      --candidate mixed-q3mlp-q3attn-v1 --out receipts/.../G042_BPW_FAMILY.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/campaign/records/runs/qwen38-27b"
N_PARAMS = 26_895_998_464

# From HONEST_ROOF_WEIGHT_ADDRESSING.json's byte census.
EMBED_TABLE_BYTES = 675_430_440
EMBED_ROW_BYTES = 2_720
NORMS_BYTES = 6_475_776
# Activation traffic per code byte, measured in NX_MATMUL_K_AMORTIZATION: a Q4
# code byte carries two weights and therefore drags 8 bytes of f32 activation.
ACT_BYTES_PER_CODE_BYTE = 8.0
# GQA KV geometry from the runtime: 16 layers, 8 kv heads, head_dim 256, K and V, f32.
KV_BYTES_PER_POSITION = 16 * 8 * 256 * 2 * 4


def walk_bytes(root: pathlib.Path) -> int:
    seen, total = set(), 0
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            if (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            total += st.st_size
    return total


def addressed_bytes(root: pathlib.Path) -> int:
    for name in ("PACK_REPORT.json", "manifest.json"):
        p = root / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for k in ("all_required_weight_artifact_bytes", "tensor_payload_bytes"):
            if k in d:
                return int(d[k])
    raise SystemExit(f"no addressed-byte figure in {root}")


def bpw(b: float) -> float:
    return b * 8.0 / N_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True)
    ap.add_argument("--contexts", default="128,8192,131072")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    contexts = [int(x) for x in a.contexts.split(",")]

    rows = []
    for name in a.candidate:
        root = RUNS / name
        stored = walk_bytes(root)
        addressed = addressed_bytes(root)
        # The embed TABLE is stored but only one row is addressed per token.
        active = addressed - EMBED_TABLE_BYTES + EMBED_ROW_BYTES
        # At 11-14 GB the weight stream cannot be cache-resident across tokens, so
        # weight traffic is DRAM traffic. Activations ARE cache-served: measured at
        # ~5000-6000 GB/s in NX_MATMUL_K_AMORTIZATION, far above any DRAM rate.
        code_bytes = addressed - NORMS_BYTES - EMBED_TABLE_BYTES
        cache = code_bytes * ACT_BYTES_PER_CODE_BYTE
        row = {
            "candidate": name,
            "STORED_BPW": bpw(stored),
            "ACTIVE_BPW_PER_TOKEN": bpw(active),
            "DRAM_BPW_PER_TOKEN": bpw(active),
            "CACHE_BPW_PER_TOKEN": bpw(cache),
            "GENERATED_BPW_EQUIVALENT": 0.0,
            "CORRECTION_BPW": 0.0,
            "SHARED_BPW": 0.0,
            "STATE_BPW_EQUIVALENT": {str(c): bpw(c * KV_BYTES_PER_POSITION) for c in contexts},
            "_bytes": {"stored": stored, "addressed": addressed, "active": active,
                       "dead_on_disk": stored - addressed,
                       "cache_activation_traffic": cache},
        }
        rows.append(row)
        print(f"{name}")
        print(f"  STORED   {row['STORED_BPW']:.6f}   ({stored:,} bytes on disk)")
        print(f"  ACTIVE   {row['ACTIVE_BPW_PER_TOKEN']:.6f}   ({active:,} bytes/token)")
        print(f"  DRAM     {row['DRAM_BPW_PER_TOKEN']:.6f}")
        print(f"  CACHE    {row['CACHE_BPW_PER_TOKEN']:.6f}   (activation traffic, cache-served)")
        print(f"  GEN {row['GENERATED_BPW_EQUIVALENT']:.4f}  CORR {row['CORRECTION_BPW']:.4f}  "
              f"SHARED {row['SHARED_BPW']:.4f}")
        print("  STATE    " + "  ".join(f"{c}:{row['STATE_BPW_EQUIVALENT'][str(c)]:.4f}"
                                        for c in contexts))
        print(f"  stored - active = {bpw(stored - active):.6f} BPW of bytes that never move")

    # The verify line, mechanically.
    def spread(key):
        v = [r[key] for r in rows]
        return max(v) - min(v)
    distinct = len({round(r["STORED_BPW"] - r["ACTIVE_BPW_PER_TOKEN"], 6) for r in rows})
    collapsed = all(
        abs(r["STORED_BPW"] - r["ACTIVE_BPW_PER_TOKEN"]) < 1e-9 for r in rows)
    print(f"\nVERIFY: stored-minus-active takes {distinct} distinct values across "
          f"{len(rows)} candidates; decomposition is "
          f"{'DECORATIVE -- all axes collapsed' if collapsed else 'LOAD-BEARING'}")
    hi = max(rows, key=lambda r: r["STORED_BPW"] - r["ACTIVE_BPW_PER_TOKEN"])
    lo = min(rows, key=lambda r: r["STORED_BPW"] - r["ACTIVE_BPW_PER_TOKEN"])
    print(f"  highest stored-over-active: {hi['candidate']} "
          f"({hi['STORED_BPW']:.6f} stored vs {hi['ACTIVE_BPW_PER_TOKEN']:.6f} active)")
    print(f"  lowest:                     {lo['candidate']} "
          f"({lo['STORED_BPW']:.6f} stored vs {lo['ACTIVE_BPW_PER_TOKEN']:.6f} active)")

    doc = {
        "schema": "hawking.nos.bpw_family.v1",
        "obligation": "G042 -- extended BPW family per candidate",
        "definitions": {
            "STORED_BPW": "every byte under the artifact root, walked, over the original language "
                          "parameter count",
            "ACTIVE_BPW_PER_TOKEN": "addressed bytes minus the embed TABLE plus the one embed ROW "
                                    "a token gathers",
            "DRAM_BPW_PER_TOKEN": "equals ACTIVE here: at 11-14 GB the weight stream cannot be "
                                  "cache-resident across tokens, so weight traffic is DRAM traffic",
            "CACHE_BPW_PER_TOKEN": "activation traffic, 8 bytes of f32 per Q4 code byte, measured "
                                   "cache-served in NX_MATMUL_K_AMORTIZATION at ~5000-6000 GB/s "
                                   "which is far above any DRAM rate on this machine",
            "GENERATED_BPW_EQUIVALENT": "structure computed rather than stored. ZERO for every "
                                        "live candidate: the one generated transform tested "
                                        "(Hadamard) was refuted, and the channel scale is folded "
                                        "into existing tensors rather than generated at runtime",
            "CORRECTION_BPW": "correction planes over a base. ZERO: no live candidate has tiers",
            "SHARED_BPW": "structure stored once and used by many tensors. ZERO: G035 refuted "
                          "cross-layer sharing at matched bits",
            "STATE_BPW_EQUIVALENT": "KV/state at a context length, same units. 16 GQA layers, "
                                    "8 kv heads, head_dim 256, K and V, f32",
        },
        "contexts": contexts,
        "candidates": rows,
        "verify": {
            "collapsed": collapsed,
            "distinct_stored_minus_active_values": distinct,
            "highest_stored_over_active": hi["candidate"],
            "lowest_stored_over_active": lo["candidate"],
            "note": "The high-stored/low-active case is not hypothetical. mixed-q3mlp-q3attn-v1 "
                    "is what this campaign quoted as its density leader for weeks while its "
                    "directory held 1.5x what its records address (F4).",
        },
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
