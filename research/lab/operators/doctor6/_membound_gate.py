"""One-shot v2 old-vs-new regression + dense CP1 runner.

Not imported by prescribe. Invoked as:

    PYTHONPATH=<repo> ~/.grok-vision/bin/python -m lab.operators.doctor6._membound_gate
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RECORDS = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records"
)
V2 = (
    RECORDS
    / "ascension-sandbox/physical/qwen30/quality-diagnostics"
    / "source-bf16-capture-v2-perexpert64"
)
DENSE = (
    RECORDS
    / "ascension-sandbox/physical/qwen30/quality-diagnostics"
    / "source-bf16-capture-v3-dense/full-run"
)
OUT_DIR = REPO / "workspace/campaign/records/lane-doctor-v6"
MODEL = RECORDS / "runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"


def _rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _rss_gib() -> float:
    return _rss_bytes() / (1024**3)


def _summarize(rx: dict) -> dict:
    organs = []
    for row in rx.get("predicted_vs_actual") or []:
        organs.append(
            {
                "tensor_name": row["tensor_name"],
                "layer": row["layer"],
                "component": row["component"],
                "band": row.get("band"),
                "incumbent_cosine": row.get("incumbent_cosine"),
                "predicted_cosine": row.get("predicted_cosine"),
                "chain": row.get("chain"),
            }
        )
    return {
        "status": rx.get("status"),
        "refusal": rx.get("refusal"),
        "deficits": [d.get("kind") for d in rx.get("deficits") or []],
        "coherence": {
            k: (rx.get("coherence_screen") or {}).get(k)
            for k in (
                "pass_gate",
                "predicted_composition",
                "mean_organ_cos",
                "refusal_reason",
            )
        },
        "diagnose": rx.get("diagnose"),
        "billing": {
            k: (rx.get("billing") or {}).get(k)
            for k in ("complete_physical_bpw", "expert_local_bpw", "under_1_0")
        },
        "allocator": {
            k: (rx.get("allocator") or {}).get(k)
            for k in (
                "achieved_avg_eff_bpw",
                "within_budget",
                "allocator_invoked",
                "histogram",
            )
        },
        "ceiling": {
            "legal": (rx.get("ceiling") or {}).get("legal")
            or ((rx.get("ceiling") or {}).get("receipt") or {}).get("legal"),
            "escape_applied": (rx.get("ceiling") or {}).get("escape_applied")
            or ((rx.get("ceiling") or {}).get("receipt") or {}).get("escape_applied"),
        },
        "sample_organs": [
            o["tensor_name"] for o in (rx.get("sample") or {}).get("organs") or []
        ],
        "n_routed": {
            o["tensor_name"]: o.get("n_routed")
            for o in (rx.get("sample") or {}).get("organs") or []
        },
        "n_materialized": {
            o["tensor_name"]: o.get("n_materialized")
            for o in (rx.get("sample") or {}).get("organs") or []
        },
        "subsampled": (rx.get("sample") or {}).get("subsampled_layer_experts"),
        "peak_rss_bytes": rx.get("peak_rss_bytes"),
        "elapsed_seconds": rx.get("elapsed_seconds"),
        "organs": organs,
    }


def _compare_rx(old: dict, new: dict) -> dict:
    old_names = [o["tensor_name"] for o in (old.get("sample") or {}).get("organs") or []]
    new_names = [o["tensor_name"] for o in (new.get("sample") or {}).get("organs") or []]
    old_cos = {
        r["tensor_name"]: float(r["predicted_cosine"])
        for r in old.get("predicted_vs_actual") or []
    }
    new_cos = {
        r["tensor_name"]: float(r["predicted_cosine"])
        for r in new.get("predicted_vs_actual") or []
    }
    cos_deltas = []
    for name in old_names:
        if name not in new_cos:
            cos_deltas.append({"tensor_name": name, "missing_in_new": True})
            continue
        d = abs(old_cos[name] - new_cos[name])
        cos_deltas.append(
            {
                "tensor_name": name,
                "old": old_cos[name],
                "new": new_cos[name],
                "abs_delta": d,
            }
        )
    max_delta = max((c.get("abs_delta") or 0.0) for c in cos_deltas) if cos_deltas else None
    return {
        "organ_set_identical": old_names == new_names,
        "n_old": len(old_names),
        "n_new": len(new_names),
        "status_identical": old.get("status") == new.get("status"),
        "old_status": old.get("status"),
        "new_status": new.get("status"),
        "refusal_kinds_identical": (old.get("refusal") or {}).get("kind")
        == (new.get("refusal") or {}).get("kind")
        and (old.get("refusal") or {}).get("reasons")
        == (new.get("refusal") or {}).get("reasons"),
        "max_cosine_abs_delta": max_delta,
        "cosines_within_1e9": max_delta is not None and max_delta <= 1e-9,
        "cosine_deltas": cos_deltas,
    }


def regression_v2() -> dict:
    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        collect_expert_activations,
        count_expert_activations,
    )
    from lab.operators.doctor6.prescribe import deterministic_sample, prescribe
    from lab.operators.q30_activation_aware_family_probe import load_capture

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"capture": str(V2), "rss_start_gib": _rss_gib()}

    print(f"[gate] v2 mmap count + wanted materialize  rss={_rss_gib():.3f} GiB", flush=True)
    t0 = time.perf_counter()
    counts_new, prov_new = count_expert_activations(V2)
    organs_new = deterministic_sample(counts_new)
    wanted = {(o.layer, o.expert) for o in organs_new}
    by_new, _ = collect_expert_activations(
        V2, wanted_keys=wanted, max_rows_per_expert=2048
    )
    report["new_loader"] = {
        "seconds": time.perf_counter() - t0,
        "n_le": len(counts_new),
        "compact_mmap": bool(prov_new.get("compact_mmap")),
        "rss_gib": _rss_gib(),
        "organs": [o.name for o in organs_new],
        "n_routed": {o.name: o.n_routed for o in organs_new},
    }
    print(
        f"[gate] v2 new loader {report['new_loader']['seconds']:.1f}s "
        f"rss={_rss_gib():.3f} GiB compact={prov_new.get('compact_mmap')}",
        flush=True,
    )

    print(f"[gate] v2 in-memory count + wanted collect  rss={_rss_gib():.3f} GiB", flush=True)
    t1 = time.perf_counter()
    cap = load_capture(V2)
    counts_old, _ = count_expert_activations(V2, cap)
    organs_old = deterministic_sample(counts_old)
    by_old, _ = collect_expert_activations(V2, cap, wanted_keys=wanted)
    report["old_loader"] = {
        "seconds": time.perf_counter() - t1,
        "n_le": len(counts_old),
        "rss_gib": _rss_gib(),
        "organs": [o.name for o in organs_old],
    }
    counts_equal = counts_old == counts_new
    organs_equal = [
        (o.layer, o.expert, o.component, o.n_routed) for o in organs_old
    ] == [(o.layer, o.expert, o.component, o.n_routed) for o in organs_new]
    x_ok = True
    x_max = 0.0
    for key in wanted:
        if key not in by_old or key not in by_new:
            x_ok = False
            break
        if by_old[key].shape != by_new[key].shape:
            x_ok = False
            break
        d = float(np.max(np.abs(by_old[key].astype(np.float64) - by_new[key].astype(np.float64))))
        x_max = max(x_max, d)
        if d != 0.0:
            x_ok = False
    report["loader_identity"] = {
        "counts_equal": counts_equal,
        "organs_equal": organs_equal,
        "x_bit_identical": x_ok,
        "x_max_abs_delta": x_max,
        "n_counts_old": len(counts_old),
        "n_counts_new": len(counts_new),
    }
    print(f"[gate] loader identity {report['loader_identity']}", flush=True)

    del by_old, by_new, cap
    import gc

    gc.collect()

    print(f"[gate] v2 NEW prescribe  rss={_rss_gib():.3f} GiB", flush=True)
    new_path = OUT_DIR / "PRESCRIPTION_v2_membound.json"
    rx_new = prescribe(
        model_dir=MODEL,
        capture=V2,
        target_bpw=1.5,
        device="cpu",
        memory_bounded=True,
        max_rows_per_expert=2048,
        out_path=new_path,
    )
    print(
        f"[gate] v2 NEW status={rx_new.get('status')} "
        f"elapsed={rx_new.get('elapsed_seconds'):.1f}s rss={_rss_gib():.3f}",
        flush=True,
    )

    print(f"[gate] v2 OLD prescribe  rss={_rss_gib():.3f} GiB", flush=True)
    old_path = OUT_DIR / "PRESCRIPTION_v2_unbounded.json"
    rx_old = prescribe(
        model_dir=MODEL,
        capture=V2,
        target_bpw=1.5,
        device="cpu",
        memory_bounded=False,
        max_rows_per_expert=None,
        out_path=old_path,
    )
    print(
        f"[gate] v2 OLD status={rx_old.get('status')} "
        f"elapsed={rx_old.get('elapsed_seconds'):.1f}s rss={_rss_gib():.3f}",
        flush=True,
    )

    report["new_rx"] = _summarize(rx_new)
    report["old_rx"] = _summarize(rx_old)
    report["compare"] = _compare_rx(rx_old, rx_new)
    report["rss_end_gib"] = _rss_gib()
    (OUT_DIR / "MEMBOUND_V2_REGRESSION.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print(f"[gate] v2 compare {report['compare']}", flush=True)
    return report


def cp1_dense() -> dict:
    from lab.operators.doctor6.prescribe import prescribe

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "PRESCRIPTION_dense_cp1.json"
    print(f"[gate] CP1 dense prescribe  rss={_rss_gib():.3f} GiB", flush=True)
    t0 = time.perf_counter()
    rx = prescribe(
        model_dir=MODEL,
        capture=DENSE,
        target_bpw=1.5,
        device="cpu",
        memory_bounded=True,
        max_rows_per_expert=2048,
        out_path=out_path,
    )
    summary = _summarize(rx)
    summary["rss_end_gib"] = _rss_gib()
    summary["wall_seconds"] = time.perf_counter() - t0
    summary["out"] = str(out_path)
    (OUT_DIR / "MEMBOUND_CP1_DENSE.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(
        f"[gate] CP1 status={rx.get('status')} elapsed={rx.get('elapsed_seconds'):.1f}s "
        f"rss={_rss_gib():.3f} GiB peak_rx={rx.get('peak_rss_bytes')}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else "all"
    if mode in ("v2", "all"):
        regression_v2()
    if mode in ("dense", "all", "cp1"):
        cp1_dense()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
