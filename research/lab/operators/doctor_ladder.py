"""Live Doctor ladder entry point — compose a chain and run L2 rungs on Q30 organs.

Invocable by operators and the CLI. Does not dump the retired condense campaign
surface; it wires registry + mixed_precision_alloc + expert_alloc against real
Qwen3-Coder-30B-A3B expert organs loaded via positioned safetensors reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lab.operators import doctor_registry as registry
from lab.operators import expert_alloc
from lab.operators import mixed_precision_alloc as mp
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
DEFAULT_MODEL_DIR = MAIN_HAWKING / (
    "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
)

# Qwen3-Coder-30B-A3B MoE expert projections.
COMPONENTS = ("gate_proj", "up_proj", "down_proj")


def _tensor_name(layer: int, expert: int, component: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{component}.weight"


def load_q30_expert_organs(
    model_dir: Path,
    *,
    layer: int = 0,
    expert: int = 0,
    components: tuple[str, ...] = COMPONENTS,
) -> dict[str, np.ndarray]:
    """Load one real Q30 expert's gate/up/down weights (float32)."""
    model_dir = model_dir.expanduser().resolve()
    weight_map = load_weight_map(model_dir)
    out: dict[str, np.ndarray] = {}
    for comp in components:
        name = _tensor_name(layer, expert, comp)
        if name not in weight_map:
            raise FileNotFoundError(f"organ not in weight map: {name}")
        W = load_tensor(model_dir, weight_map, name).astype(np.float32, copy=False)
        out[name] = W
    return out


def run_mixed_prec_rung(
    model_dir: Path,
    *,
    layer: int = 0,
    expert: int = 0,
    target_bpw: float = 1.0,
    bits_set: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, Any]:
    organs = load_q30_expert_organs(model_dir, layer=layer, expert=expert)
    result = mp.run_on_organs(organs, bits_set=bits_set, target_bpw=target_bpw)
    result["organ"] = {
        "model_dir": str(model_dir),
        "layer": layer,
        "expert": expert,
        "components": list(COMPONENTS),
        "shapes": {k: list(v.shape) for k, v in organs.items()},
        "elements": {k: int(v.size) for k, v in organs.items()},
    }
    result["status"] = "RUNG_COMPLETE"
    return result


def run_expert_alloc_rung(
    model_dir: Path,
    *,
    layer: int = 0,
    experts: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    bits_set: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    """Measure several real experts on one layer and run decide+propose."""
    weight_map = load_weight_map(model_dir.expanduser().resolve())
    organ_weights: dict[str, np.ndarray] = {}
    for e in experts:
        for comp in COMPONENTS:
            name = _tensor_name(layer, e, comp)
            if name not in weight_map:
                continue
            organ_weights[name] = load_tensor(model_dir, weight_map, name).astype(
                np.float32, copy=False
            )
    if not organ_weights:
        raise FileNotFoundError(
            f"no expert organs found for layer={layer} experts={experts} under {model_dir}"
        )
    rows = expert_alloc.measure_organs_from_weights(organ_weights, bits_set=bits_set)
    result = expert_alloc.run_rung(rows, bits_set=bits_set)
    result["organ"] = {
        "model_dir": str(model_dir),
        "layer": layer,
        "experts_requested": list(experts),
        "n_weight_tensors": len(organ_weights),
        "n_expert_rows": len(rows),
    }
    result["status"] = "RUNG_COMPLETE"
    return result


def run_select(
    params_b: float = 30.0,
    target_bpw: float | None = 1.0,
    *,
    is_moe: bool = True,
) -> dict[str, Any]:
    return registry.plan(params_b, target_bpw, is_moe=is_moe)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Doctor ladder live entry (registry + L2 rungs on Q30 organs)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sel = sub.add_parser("select", help="compose recovery chain")
    p_sel.add_argument("--params-b", type=float, default=30.0)
    p_sel.add_argument("--target-bpw", type=float, default=1.0)
    p_sel.add_argument("--dense", action="store_true", help="disable MoE expert_alloc")

    p_mp = sub.add_parser("mixed-prec", help="run L2 mixed_prec on one Q30 expert organ")
    p_mp.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p_mp.add_argument("--layer", type=int, default=0)
    p_mp.add_argument("--expert", type=int, default=0)
    p_mp.add_argument("--target-bpw", type=float, default=1.0)
    p_mp.add_argument("--bits", default="1,2,3,4")

    p_ex = sub.add_parser("expert-alloc", help="run L2 expert_alloc on real Q30 experts")
    p_ex.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p_ex.add_argument("--layer", type=int, default=0)
    p_ex.add_argument("--experts", default="0,1,2,3,4,5,6,7")
    p_ex.add_argument("--bits", default="1,2")

    p_list = sub.add_parser("list", help="list registry methods")

    args = ap.parse_args(argv)

    if args.cmd == "list":
        return registry.main(["list"])
    if args.cmd == "select":
        out = run_select(args.params_b, args.target_bpw, is_moe=not args.dense)
        print(json.dumps(out, indent=2))
        return 0
    if args.cmd == "mixed-prec":
        bits = tuple(int(x) for x in args.bits.split(",") if x.strip())
        out = run_mixed_prec_rung(
            args.model_dir,
            layer=args.layer,
            expert=args.expert,
            target_bpw=args.target_bpw,
            bits_set=bits,
        )
        print(json.dumps(out, indent=2))
        return 0 if out.get("status") == "RUNG_COMPLETE" else 1
    if args.cmd == "expert-alloc":
        bits = tuple(int(x) for x in args.bits.split(",") if x.strip())
        ex = tuple(int(x) for x in args.experts.split(",") if x.strip())
        out = run_expert_alloc_rung(
            args.model_dir, layer=args.layer, experts=ex, bits_set=bits
        )
        print(json.dumps(out, indent=2))
        return 0 if out.get("status") == "RUNG_COMPLETE" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
