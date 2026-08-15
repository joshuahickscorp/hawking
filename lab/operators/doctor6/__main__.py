"""CLI: python -m lab.operators.doctor6 {prescribe,treat,verify,selftest}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="doctor6",
        description="Doctor v6 — objective-driven, per-organ adaptive, predictive",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("prescribe", help="diagnose + compose + predict + emit/refuse")
    p_pre.add_argument("--model", default="qwen3-coder-30b-a3b", help="model id")
    p_pre.add_argument("--model-dir", type=Path, default=None)
    p_pre.add_argument("--capture", type=Path, default=None)
    p_pre.add_argument("--target-bpw", type=float, default=1.0)
    p_pre.add_argument("--target-cos", type=float, default=None)
    p_pre.add_argument("--le-per-band", type=int, default=4)
    p_pre.add_argument("--min-rows", type=int, default=64)
    p_pre.add_argument("--device", default="auto")
    p_pre.add_argument("--qat-steps", type=int, default=200)
    p_pre.add_argument("--qat-lr", type=float, default=1e-3)
    p_pre.add_argument(
        "--out",
        type=Path,
        default=Path(
            "workspace/campaign/records/lane-doctor-v6/PRESCRIPTION.json"
        ),
    )
    p_pre.add_argument(
        "--force-over-ceiling",
        action="store_true",
        help="inflate bill to prove ceiling fail-closed (acceptance)",
    )
    p_pre.add_argument(
        "--max-rows-per-expert",
        type=int,
        default=2048,
        help="per-expert row cap after sampling (0 disables; default 2048)",
    )
    p_pre.add_argument(
        "--unbounded",
        action="store_true",
        help="old loader: materialize every (layer, expert) pair (OOM on dense)",
    )

    p_tr = sub.add_parser("treat", help="execute a prescription and seal")
    p_tr.add_argument("--prescription", type=Path, required=True)
    p_tr.add_argument("--model-dir", type=Path, default=None)
    p_tr.add_argument("--capture", type=Path, default=None)
    p_tr.add_argument("--device", default="auto")
    p_tr.add_argument("--qat-steps", type=int, default=200)
    p_tr.add_argument("--qat-lr", type=float, default=None)
    p_tr.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/campaign/records/lane-doctor-v6/TREATMENT.json"),
    )

    p_vf = sub.add_parser("verify", help="c^48 coherence screen (+ optional generation)")
    p_vf.add_argument("--artifact", type=Path, default=None)
    p_vf.add_argument(
        "--cosines",
        default=None,
        help="comma-separated organ cosines, or a single mean repeated",
    )
    p_vf.add_argument("--mean-cos", type=float, default=None, help="homogeneous mean cosine")
    p_vf.add_argument("--n-layers", type=int, default=48)
    p_vf.add_argument("--run-generation", action="store_true")
    p_vf.add_argument(
        "--ground-truth",
        action="store_true",
        help="run source-gated-v2 (0.685) refuse + uniform-q4 (0.987) pass checks",
    )

    sub.add_parser("selftest", help="coherence ground truth + ceiling fail-closed smoke")

    args = ap.parse_args(argv)

    def _device(d: str) -> str:
        if d != "auto":
            return d
        try:
            import torch

            return "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            return "cpu"

    if args.cmd == "selftest":
        from lab.operators.doctor6.selftest import run_selftest

        out = run_selftest()
        print(json.dumps(out, indent=2, default=str))
        return 0 if out["all_pass"] else 1

    if args.cmd == "verify":
        from lab.operators.doctor6.verify import verify, verify_ground_truth_pair

        if args.ground_truth:
            out = verify_ground_truth_pair()
            print(json.dumps(out, indent=2))
            return 0 if out["all_pass"] else 1
        cosines = None
        if args.mean_cos is not None:
            cosines = [float(args.mean_cos)] * int(args.n_layers)
        elif args.cosines:
            parts = [float(x) for x in args.cosines.split(",") if x.strip()]
            cosines = parts
        out = verify(
            artifact=args.artifact,
            organ_cosines=cosines,
            n_layers=args.n_layers,
            run_generation=args.run_generation,
        )
        print(json.dumps(out, indent=2))
        return 0 if out["verdict"] == "COHERENT" else 2

    if args.cmd == "prescribe":
        from lab.operators.doctor6.coherence import LAYER_TARGET_COS
        from lab.operators.doctor6.prescribe import prescribe

        target_cos = float(args.target_cos) if args.target_cos is not None else LAYER_TARGET_COS
        out = prescribe(
            model_id=args.model,
            model_dir=args.model_dir,
            capture=args.capture,
            target_bpw=args.target_bpw,
            target_cos=target_cos,
            le_per_band=args.le_per_band,
            min_rows=args.min_rows,
            device=_device(args.device),
            qat_steps=args.qat_steps,
            qat_lr=args.qat_lr,
            out_path=args.out,
            force_over_ceiling=args.force_over_ceiling,
            memory_bounded=not args.unbounded,
            max_rows_per_expert=(
                None if int(args.max_rows_per_expert) <= 0 else int(args.max_rows_per_expert)
            ),
        )
        # Print abridged summary to stdout; full JSON is on disk.
        summary = {
            "status": out.get("status"),
            "elapsed_seconds": out.get("elapsed_seconds"),
            "diagnose": out.get("diagnose"),
            "compose": {
                k: out.get("compose", {}).get(k)
                for k in (
                    "n_organs",
                    "n_clear_target",
                    "mean_prescribed_cosine",
                    "mean_incumbent_cosine",
                    "distinct_chains_by_component",
                    "chains_differ_across_components",
                )
            },
            "coherence_screen": {
                k: out.get("coherence_screen", {}).get(k)
                for k in (
                    "pass_gate",
                    "predicted_composition",
                    "mean_organ_cos",
                    "per_band_deficit",
                    "refusal_reason",
                )
            },
            "billing": {
                k: out.get("billing", {}).get(k)
                for k in ("complete_physical_bpw", "under_1_0", "expert_local_bpw")
            },
            "allocator": {
                "allocator_invoked": out.get("allocator", {}).get("allocator_invoked"),
                "achieved_avg_eff_bpw": out.get("allocator", {}).get(
                    "achieved_avg_eff_bpw"
                ),
                "within_budget": out.get("allocator", {}).get("within_budget"),
                "histogram": out.get("allocator", {}).get("histogram"),
                "objective": out.get("allocator", {}).get("objective"),
                "sensitivity": out.get("allocator", {}).get("sensitivity"),
            },
            "ceiling": {
                "enforcer_called": out.get("ceiling", {}).get("enforcer_called"),
                "legal": out.get("ceiling", {}).get("legal")
                or out.get("ceiling", {}).get("receipt", {}).get("legal"),
                "error": out.get("ceiling", {}).get("error"),
                "escape_applied": out.get("ceiling", {}).get("escape_applied")
                or out.get("ceiling", {}).get("receipt", {}).get("escape_applied"),
                "escape_sealed": bool(
                    (out.get("ceiling") or {}).get("escape_receipt")
                    and (out.get("ceiling") or {})
                    .get("escape_receipt", {})
                    .get("sha256")
                ),
            },
            "deficits": out.get("deficits"),
            "out": str(args.out),
        }
        print(json.dumps(summary, indent=2, default=str))
        if out.get("status") == "REFUSED":
            return 2
        return 0

    if args.cmd == "treat":
        from lab.operators.doctor6.treat import treat

        out = treat(
            prescription_path=args.prescription,
            model_dir=args.model_dir,
            capture=args.capture,
            device=_device(args.device),
            qat_steps=args.qat_steps,
            qat_lr=args.qat_lr,
            out_path=args.out,
        )
        print(
            json.dumps(
                {
                    "status": out.get("status"),
                    "elapsed_seconds": out.get("elapsed_seconds"),
                    "coherence": out.get("coherence_screen", {}).get("pass_gate"),
                    "ceiling": out.get("ceiling"),
                    "n_organs": len(out.get("organs") or []),
                    "out": str(args.out),
                },
                indent=2,
                default=str,
            )
        )
        return 0 if out.get("status") == "SEALED" else 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
