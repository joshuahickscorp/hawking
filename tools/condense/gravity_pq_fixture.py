#!/usr/bin/env python3.12
"""Emit byte-exact PQ parity fixtures so a non-Python decoder can be proven against
``gravity_forge.pq_execute`` without a served model or a resident source shard.

Each fixture is the real production artifact: the same ``glm52_pack.serialize`` blob that
lands inside a ``.gravity`` tensor payload, plus one input vector and the reference output
that ``pq_execute`` -- the authority for what the codec means -- produces from it.  A
decoder that reproduces ``y`` from ``blob`` and ``x`` decodes the shipped format, and one
that does not is wrong regardless of how plausible its arithmetic looks.

    python3.12 tools/condense/gravity_pq_fixture.py [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_pack as pack  # noqa: E402
import gravity_forge as forge  # noqa: E402

DEFAULT_OUT = HERE.parents[1] / "crates/hawking-core/tests/fixtures/gravity_pq"


def emit(out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rung in pack.LADDER:
        # Rows/cols must divide the rung geometry the way real expert tensors do, and be
        # large enough that the fixed 64-byte header amortizes under the BPW ceiling.
        rows, cols = 512, rung["dim"] * 64
        rng = np.random.default_rng(20260724)
        weights = rng.standard_normal((rows, cols), dtype=np.float32) * 0.02
        artifact = forge.pack_product_quant(
            weights, dim=rung["dim"], subspaces=1, k=rung["k"], seed=0)
        blob = pack.serialize(artifact)
        x = rng.standard_normal(cols, dtype=np.float32)
        # The authority is what is ON DISK.  The fitted artifact keeps fp32 codebooks; the
        # container bills and stores fp16, so the executable truth is the rehydrated one.
        # Referencing the fitted artifact would hand a decoder a target the format cannot
        # represent, and every correct decoder would then "fail" parity.
        stored = pack.load_artifact(blob)
        y = forge.pq_execute(stored, x)
        y_fit = forge.pq_execute(artifact, x)

        stem = f"pq_{rung['rung']}"
        (out_dir / f"{stem}.bin").write_bytes(blob)
        (out_dir / f"{stem}.x.f32").write_bytes(np.ascontiguousarray(x, np.float32).tobytes())
        (out_dir / f"{stem}.y.f32").write_bytes(np.ascontiguousarray(y, np.float32).tobytes())
        codes = artifact.config["pq_codes"]
        row = {
            "fixture": stem, "rung": rung["rung"], "rows": rows, "cols": cols,
            "dim": rung["dim"], "k": rung["k"], "subspaces": int(codes["S"]),
            "sub": int(codes["sub"]), "nchunk": int(codes["nchunk"]),
            "index_bits": pack.index_bits(rung["k"]), "rotate": bool(codes["rotate"]),
            "seed": int(codes["seed"]), "blob_bytes": len(blob),
            "bpw": round(float(artifact.whole_artifact_bpw), 6),
            "y_abs_max": float(np.abs(y).max()),
            # Measured cost of the fp16 codebook billing, reported rather than hidden.
            "fp16_codebook_max_abs_delta": float(np.abs(y - y_fit).max()),
        }
        manifest.append(row)

        # A fixture nobody can round-trip is a liability, so prove the inverse here:
        # deserializing twice must land on the same geometry and the same output.
        again = forge.pq_execute(pack.load_artifact(blob), x)
        if not np.array_equal(again, y):
            raise SystemExit(f"{stem}: deserialize is not deterministic")

    (out_dir / "manifest.json").write_text(
        json.dumps({"schema": "hawking.gravity.pq_fixture.v1", "fixtures": manifest},
                   indent=1, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    for entry in emit(target):
        print(json.dumps(entry, sort_keys=True))
    print(f"wrote {target}")
