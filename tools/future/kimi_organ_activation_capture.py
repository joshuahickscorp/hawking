#!/usr/bin/env python3
"""Capture a bounded real KIMI MoE-organ activation corpus.

The corpus is collected from the immutable KIMI_BASE forward path, not from a
Gaussian proxy.  It is used only as a representation discriminator input
distribution; it does not alter weights or create a candidate model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools" / "future") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "future"))

import gravity_outlier_eval as G  # noqa: E402

SCHEMA = "hawking.future.kimi_organ_activation_capture.v1"
TARGET_LAYER = 10
TARGET_SUFFIX = ".mlp.switch_mlp.up_proj"
MAX_ROWS_PER_EXPERT = 256
TEXTS = (
    G.NLL_TEXT,
    " ".join(G.PROMPTS),
    "Analyze a bounded systems problem and state the evidence needed before acting. "
    * 8,
)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capture(output: Path, max_rows: int = MAX_ROWS_PER_EXPERT,
            target_suffix: str = TARGET_SUFFIX) -> dict[str, Any]:
    from tools.future.campaign_memory_guard import require_ok, resource_cost, sample, watch
    import mlx.core as mx
    from mlx_lm.models import switch_layers as SL

    entry = require_ok("KIMI real organ activation capture", expected_gb=32.0).as_dict()
    load_started = time.perf_counter()
    model, tok = G._load()
    load_s = time.perf_counter() - load_started
    loaded = sample(expected_gb=32.0).as_dict()

    module_paths = {
        id(module): name for name, module in model.named_modules()
        if name.endswith(target_suffix)
        and f"model.layers.{TARGET_LAYER}." in name
    }
    if len(module_paths) != 1:
        raise RuntimeError(f"expected one target up_proj, found {sorted(module_paths.values())}")
    target_path = next(iter(module_paths.values()))
    captures: list[list[Any]] = []
    expert_ids: list[list[int]] = []
    original = SL.SwitchLinear.__call__

    def capture_call(self, x, indices, sorted_indices=False):
        if id(self) == next(iter(module_paths)):
            xf = x.astype(mx.float32).reshape(-1, x.shape[-1])
            ids = indices.reshape(-1)
            if xf.shape[0] != ids.shape[0]:
                if xf.shape[0] == 0 or ids.shape[0] % xf.shape[0] != 0:
                    raise RuntimeError(
                        f"target hook shape mismatch: inputs={xf.shape} indices={ids.shape}")
                xf = mx.repeat(xf, ids.shape[0] // xf.shape[0], axis=0)
            mx.eval(xf, ids)
            captures.append(np.asarray(xf))
            expert_ids.append(np.asarray(ids).astype(np.int32))
        return original(self, x, indices, sorted_indices=sorted_indices)

    started = time.perf_counter()
    with (
        resource_cost(interval_s=2.0, label="real KIMI organ activation capture") as bracket,
        watch(interval_s=2.0) as live_watch,
    ):
        SL.SwitchLinear.__call__ = capture_call
        try:
            for text in TEXTS:
                ids = mx.array([tok.encode(text)[:512]])
                mx.eval(model(ids))
        finally:
            SL.SwitchLinear.__call__ = original
    if not captures:
        raise RuntimeError("target activation hook captured nothing")

    all_x = np.concatenate(captures, axis=0)
    all_ids = np.concatenate(expert_ids, axis=0)
    if all_x.shape[0] != all_ids.shape[0]:
        raise RuntimeError(f"capture accounting mismatch: x={all_x.shape} ids={all_ids.shape}")
    experts = sorted(int(x) for x in np.unique(all_ids))
    arrays: dict[str, np.ndarray] = {}
    rows_by_expert: dict[str, int] = {}
    for expert in experts:
        rows = all_x[all_ids == expert]
        if rows.shape[0] > max_rows:
            choose = np.linspace(0, rows.shape[0] - 1, max_rows, dtype=np.int64)
            rows = rows[choose]
        rows = np.asarray(rows, dtype=np.float32)
        arrays[f"expert_{expert}"] = rows
        rows_by_expert[str(expert)] = int(rows.shape[0])

    digest = _atomic_npz(output, arrays)
    ended = sample().as_dict()
    return {
        "schema": SCHEMA,
        "status": "REAL_KIMI_ORGAN_ACTIVATION_CAPTURE_COMPLETE",
        "source_model": "KIMI_BASE",
        "immutable_base": True,
        "model_snapshot": G.SNAP,
        "target": {"layer": TARGET_LAYER, "module": target_path,
                    "tensor_family": target_suffix.rsplit(".", 1)[-1],
                    "expert_ids_observed": experts, "max_rows_per_expert": max_rows},
        "corpus": {"path": str(output), "sha256": digest, "dtype": "float32",
                   "rows_by_expert": rows_by_expert,
                   "total_retained_rows": int(sum(rows_by_expert.values())),
                   "input_width": int(all_x.shape[-1])},
        "measurement": {"text_count": len(TEXTS), "truncated_tokens_per_text": 512,
                         "load_s": round(load_s, 3),
                         "capture_wall_s": round(time.perf_counter() - started, 3),
                         "resource_bracket": bracket,
                         "watch_worst_state": live_watch.worst,
                         "watch_samples": live_watch.samples,
                         "entry": entry, "loaded": loaded, "end": ended},
        "claim_boundary": (
            "Real forward-captured KIMI organ inputs and routing IDs for a bounded "
            "representation discriminator. This is not learned training, full-model "
            "capability, TPS, a candidate weight artifact, or promotion evidence."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script": str(Path(__file__)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS_PER_EXPERT)
    ap.add_argument("--target-suffix", choices=(".mlp.switch_mlp.up_proj",
                                                   ".mlp.switch_mlp.gate_proj",
                                                   ".mlp.switch_mlp.down_proj"),
                    default=TARGET_SUFFIX)
    args = ap.parse_args()
    if args.max_rows <= 0:
        raise SystemExit("max-rows must be positive")
    receipt = capture(args.output, args.max_rows, args.target_suffix)
    receipt_path = args.output.with_suffix(".json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, receipt_path)
    print(json.dumps({"receipt": str(receipt_path), "corpus": receipt["corpus"],
                      "status": receipt["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
