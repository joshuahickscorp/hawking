#!/usr/bin/env python3.12
"""Run the numpy oracle on the real assembled GLM-5.2 flagship artifact.

The Rust adapter already agreed with this oracle to 3.8e-6 on a tiny fixture carrying the
flagship's exact semantics. This is the other half: the SAME oracle, reading the SAME real
77 GB artifact the Rust runtime read, on the SAME tokens, so the two can be diffed directly
rather than trusted by extrapolation from a synthetic model.

    python3.12 tools/condense/glm52_flagship_oracle.py --tokens 7 1234 9 --dump logits.f32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

import glm52_reference as ref  # noqa: E402
from glm52_gravity_source import GravityGlmSource  # noqa: E402

DEFAULT_MODEL_DIR = (
    Path.home() / "Library/Application Support/Hawking/Models/GLM-5.2"
    "/b4734de4facf877f85769a911abafc5283eab3d9/General-R0"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokens", type=int, nargs="+", default=[7, 1234, 9])
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--no-verify-hash", action="store_true")
    args = ap.parse_args()

    index_json = args.dir / "model.gravity.index.json"
    if not index_json.is_file():
        raise SystemExit(f"no model.gravity.index.json in {args.dir}")
    arch = json.loads(index_json.read_text())["architecture"]

    print(f"opening (indexing via {index_json.name}, decoding nothing)...", file=sys.stderr)
    t0 = time.time()
    source = GravityGlmSource(args.dir, index_json=index_json,
                              verify_hash=not args.no_verify_hash)
    print(f"opened in {time.time()-t0:.1f}s | layers={arch['num_hidden_layers']} "
          f"hidden={arch['hidden_size']} experts={arch['n_routed_experts']} "
          f"vocab={arch['vocab_size']}", file=sys.stderr)

    print(f"forward over {len(args.tokens)} tokens: {args.tokens}", file=sys.stderr)
    t0 = time.time()
    ids = np.array([args.tokens], dtype=np.int64)
    logits, _cache, trace = ref.main_forward(ids, source, arch)
    elapsed = time.time() - t0
    print(f"forward done in {elapsed:.1f}s ({elapsed/len(args.tokens):.2f} s/token)",
          file=sys.stderr)

    flat = np.asarray(logits[0, -1], dtype=np.float32)
    order = np.argsort(-flat, kind="stable")
    topk = np.asarray(trace["final_main_topk"]).reshape(-1).tolist()
    result = {
        "tokens": args.tokens,
        "argmax": int(order[0]),
        "top5": [int(i) for i in order[:5]],
        "final_topk_indices": topk,
        "forward_seconds": elapsed,
    }
    print(json.dumps(result, indent=1))
    if args.dump:
        args.dump.write_bytes(np.ascontiguousarray(flat).tobytes())
        print(f"wrote {flat.size} logits to {args.dump}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
