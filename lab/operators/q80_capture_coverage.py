"""Q80 capture coverage audit from the compact mmap index.

Does not parse capture-result.json. A (layer, expert) pair is underdetermined
for an organ when retained-hidden rows (the quantity the fitter sees) are
strictly below that organ's fitted dimension.

Fitted dimensions (Qwen3-Coder-Next expert geometry):
  gate_proj / up_proj  X is 2048-dim router input; Gram X'X is 2048 x 2048
  down_proj            X is 512-dim post-SwiGLU; Gram is 512 x 512
                       mixed-representation rank target is 160 (hgravs01_r160)

doctor6 holdout is 25 percent, so n_fit_rows = n - n_hold with the same
rule as holdout_split (n < 4 keeps every row on both sides).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import HOLD_FRAC
from lab.operators.q80_capture_index import inspect_index

SCHEMA = "hawking.ascent.q80_capture_coverage.v1"
N_LAYERS = 48
N_EXPERTS = 512
N_PAIRS = N_LAYERS * N_EXPERTS
HIDDEN = 2048
SWIGLU = 512
DOWN_RANK_TARGET = 160
DEFAULT_CAPTURE = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "ascension-sandbox/physical/qwen80/quality-diagnostics/"
    "source-bf16-capture-n192-scale64"
)
DEFAULT_MODEL = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/qwen-80b/Qwen3-Coder-Next"
)
DEFAULT_IDENTITY = Path("receipts/QWEN80_SOURCE_IDENTITY.json")

ORGANS: tuple[dict[str, Any], ...] = (
    {
        "name": "gate_proj",
        "x_kind": "router_input",
        "x_dim": HIDDEN,
        "w_shape": [512, 2048],
        "fitted_dim": HIDDEN,
        "note": "binary_group is weight-space; activation-weighted / AWQ Gram is 2048-dim",
    },
    {
        "name": "up_proj",
        "x_kind": "router_input",
        "x_dim": HIDDEN,
        "w_shape": [512, 2048],
        "fitted_dim": HIDDEN,
        "note": "binary + residual is weight-space; activation-weighted Gram is 2048-dim",
    },
    {
        "name": "down_proj",
        "x_kind": "swiglu_hidden_routed",
        "x_dim": SWIGLU,
        "w_shape": [2048, 512],
        "fitted_dim": SWIGLU,
        "rank_target": DOWN_RANK_TARGET,
        "note": "hgravs01_r160 needs n_fit >= 160; 512-dim Gram needs n_fit >= 512",
    },
)


def _percentile_sorted(sorted_vals: np.ndarray, p: float) -> int:
    n = int(sorted_vals.size)
    if n == 0:
        return 0
    rank = int(round((p / 100.0) * (n - 1)))
    return int(sorted_vals[min(max(rank, 0), n - 1)])


def summarize(counts: np.ndarray, unit: str) -> dict[str, Any]:
    flat = np.asarray(counts, dtype=np.int32).reshape(-1)
    s = np.sort(flat)
    n = int(s.size)
    return {
        "unit": unit,
        "n": n,
        "min": int(s[0]) if n else 0,
        "p10": _percentile_sorted(s, 10),
        "p25": _percentile_sorted(s, 25),
        "p50": _percentile_sorted(s, 50),
        "p75": _percentile_sorted(s, 75),
        "p90": _percentile_sorted(s, 90),
        "p99": _percentile_sorted(s, 99),
        "max": int(s[-1]) if n else 0,
        "mean": float(s.mean()) if n else 0.0,
        "zero": int((s == 0).sum()),
        "below_8": int((s < 8).sum()),
        "below_16": int((s < 16).sum()),
        "below_32": int((s < 32).sum()),
        "below_64": int((s < 64).sum()),
        "below_160": int((s < 160).sum()),
        "below_192": int((s < 192).sum()),
        "below_512": int((s < 512).sum()),
        "below_2048": int((s < 2048).sum()),
        "at_or_above_64": int((s >= 64).sum()),
        "at_or_above_160": int((s >= 160).sum()),
        "at_or_above_192": int((s >= 192).sum()),
        "at_or_above_512": int((s >= 512).sum()),
        "at_or_above_2048": int((s >= 2048).sum()),
    }


def occupancy_histogram(counts: np.ndarray) -> list[dict[str, Any]]:
    s = np.asarray(counts, dtype=np.int32).reshape(-1)
    n = max(int(s.size), 1)
    bands = [
        (0, 1, "0"),
        (1, 2, "1"),
        (2, 4, "2-3"),
        (4, 8, "4-7"),
        (8, 16, "8-15"),
        (16, 32, "16-31"),
        (32, 64, "32-63"),
        (64, 128, "64-127"),
        (128, 160, "128-159"),
        (160, 192, "160-191"),
        (192, 256, "192-255"),
        (256, 512, "256-511"),
        (512, 768, "512-767"),
        (768, 1024, "768-1023"),
        (1024, 1536, "1024-1535"),
        (1536, 2048, "1536-2047"),
        (2048, 4096, "2048-4095"),
        (4096, 8192, "4096-8191"),
        (8192, 16384, "8192-16383"),
    ]
    out: list[dict[str, Any]] = []
    for lo, hi, label in bands:
        if lo == 0:
            c = int((s == 0).sum())
        else:
            c = int(((s >= lo) & (s < hi)).sum())
        out.append({"lo": lo, "hi": hi, "label": label, "count": c, "frac": c / n})
    c = int((s >= 16384).sum())
    out.append({"lo": 16384, "hi": None, "label": ">=16384", "count": c, "frac": c / n})
    return out


def n_fit_after_holdout(n_rows: np.ndarray, hold_frac: float = HOLD_FRAC) -> np.ndarray:
    """Match holdout_split: n < 4 keeps all rows on the fit side."""
    n = np.asarray(n_rows, dtype=np.int32)
    out = n.copy()
    big = n >= 4
    n_hold = np.minimum(n // 2, np.rint(n * hold_frac).astype(np.int32))
    n_hold = np.maximum(n_hold, 1)
    out[big] = n[big] - n_hold[big]
    return out


def census_from_index(run_dir: Path) -> dict[str, Any]:
    """Route occupancy + retained-hidden occupancy from capture-index.v1."""
    status, root, header = inspect_index(run_dir)
    if status != "ok" or root is None or header is None:
        raise RuntimeError(f"capture-index.v1 not usable under {run_dir}: {status}")
    t0 = time.perf_counter()
    layer = np.load(root / "layer.npy", mmap_mode="r")
    hidden = np.load(root / "hidden_retained.npy", mmap_mode="r")
    expert_ids = np.load(root / "expert_ids.npy", mmap_mode="r")
    expert_off = np.load(root / "expert_offsets.npy", mmap_mode="r")
    route = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.int32)
    fit = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.int32)
    n_rows = int(layer.size)
    layer_a = np.asarray(layer)
    hidden_a = np.asarray(hidden)
    eids = np.asarray(expert_ids)
    eoff = np.asarray(expert_off)
    for rid in range(n_rows):
        L = int(layer_a[rid])
        lo = int(eoff[rid])
        hi = int(eoff[rid + 1])
        es = eids[lo:hi]
        if L < 0 or L >= N_LAYERS:
            continue
        valid = (es >= 0) & (es < N_EXPERTS)
        es = es[valid]
        route[L, es] += 1
        if hidden_a[rid]:
            fit[L, es] += 1
    never = [
        {"layer": int(L), "expert": int(e)}
        for L in range(N_LAYERS)
        for e in range(N_EXPERTS)
        if int(route[L, e]) == 0
    ]
    return {
        "index_dir": str(root),
        "index_status": status,
        "n_rows": n_rows,
        "n_tokens": int(header.get("n_tokens") or 0),
        "n_keys": int(header.get("n_keys") or 0),
        "n_hidden_rows": int(header.get("n_hidden_rows") or 0),
        "header": header,
        "route": route,
        "fit": fit,
        "never_routed": never,
        "census_wall_s": time.perf_counter() - t0,
    }


def swiglu_presence(run_dir: Path) -> dict[str, Any]:
    packed = run_dir / "x" / "swiglu_hidden_routed"
    packed_exists = packed.is_dir()
    n_packed = 0
    if packed_exists:
        n_packed = sum(1 for _ in packed.glob("L*/E*.f32le"))
    # Sample the JSON only if it is small. A 1.38 GB body is not opened.
    result = run_dir / "capture-result.json"
    json_has_swiglu_field = False
    json_sampled = False
    if result.is_file() and result.stat().st_size < 32 * 1024 * 1024:
        json_sampled = True
        head = result.read_text(encoding="utf-8", errors="replace")[:200_000]
        json_has_swiglu_field = "swiglu_hidden_routed" in head
    return {
        "packed_dir": str(packed),
        "packed_dir_present": packed_exists,
        "n_packed_organ_files": int(n_packed),
        "json_sampled_for_swiglu_field": json_sampled,
        "json_head_mentions_swiglu": json_has_swiglu_field,
        "down_proj_x_on_disk": packed_exists and n_packed > 0,
        "note": (
            "down_proj's known-correct X is post-SwiGLU (512). "
            "Absent packed files mean the 25258-token capture stored only "
            "router-input hiddens (2048). organ_activations() can recompute "
            "silu(X @ Wg.T) * (X @ Wu.T) from BF16 gate/up + those hiddens."
        ),
    }


def verify_bf16_source(
    model_dir: Path,
    identity_receipt: Path | None,
    capture_header: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = capture_header.get("runtime_binding") or {}
    claimed = str(runtime.get("source_model_dir") or "")
    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    # Safetensors dtype of one expert organ — do not hash 4 GiB shards.
    shard = model_dir / "model-00001-of-00040.safetensors"
    sample_dtype = None
    sample_shape = None
    sample_name = None
    if shard.is_file():
        import struct

        with shard.open("rb") as handle:
            n = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(n))
        for key, meta in header.items():
            if key == "__metadata__" or not isinstance(meta, Mapping):
                continue
            if key.endswith("mlp.experts.0.gate_proj.weight"):
                sample_dtype = meta.get("dtype")
                sample_shape = meta.get("shape")
                sample_name = key
                break
    shard_sizes: dict[str, Any] = {"checked": 0, "mismatches": []}
    identity_doc: dict[str, Any] | None = None
    if identity_receipt is not None and identity_receipt.is_file():
        identity_doc = json.loads(identity_receipt.read_text(encoding="utf-8"))
        files = identity_doc.get("files") or {}
        for name, rec in files.items():
            if not str(name).endswith(".safetensors"):
                continue
            path = model_dir / name
            want = int(rec.get("bytes") or -1)
            got = int(path.stat().st_size) if path.is_file() else -1
            shard_sizes["checked"] = int(shard_sizes["checked"]) + 1
            if got != want:
                shard_sizes["mismatches"].append(
                    {"name": name, "observed_bytes": got, "identity_bytes": want}
                )
    quantization = "quantization_config" in cfg
    ok = (
        cfg.get("torch_dtype") == "bfloat16"
        and sample_dtype == "BF16"
        and not quantization
        and str(runtime.get("weight_backend") or "") == "source_bf16_safetensors_range_read"
        and bool(runtime.get("packed_complete_binary_not_opened"))
        and len(shard_sizes["mismatches"]) == 0
    )
    return {
        "ok": ok,
        "claimed_source_model_dir": claimed,
        "verified_model_dir": str(model_dir),
        "paths_agree": claimed == str(model_dir) or claimed.endswith("Qwen3-Coder-Next"),
        "config_architectures": cfg.get("architectures"),
        "config_torch_dtype": cfg.get("torch_dtype"),
        "config_has_quantization_config": quantization,
        "sample_tensor": sample_name,
        "sample_dtype": sample_dtype,
        "sample_shape": sample_shape,
        "identity_receipt": str(identity_receipt) if identity_receipt else None,
        "identity_full_content_verification": (
            None if identity_doc is None else identity_doc.get("full_content_verification")
        ),
        "identity_hf_commit": None if identity_doc is None else identity_doc.get("hf_commit_hash"),
        "identity_mismatches_field": None if identity_doc is None else identity_doc.get("mismatches"),
        "shard_size_check": shard_sizes,
        "capture_weight_backend": runtime.get("weight_backend"),
        "capture_packed_binary_opened": not bool(
            runtime.get("packed_complete_binary_not_opened")
        ),
        "capture_metal_used": not bool(runtime.get("metal_not_used")),
        "did_not_rehash_4gib_shards": True,
        "why_not_rehash": (
            "Immutable-identity recomputation of 40 x ~4 GiB shards has been "
            "the real latency more than once. Size+dtype+identity receipt is "
            "the load-time bind; QWEN80_SOURCE_IDENTITY.json already recorded "
            "observed_sha256 == hf_etag for every shard."
        ),
    }


def organ_verdict(fit: np.ndarray) -> list[dict[str, Any]]:
    n_fit = n_fit_after_holdout(fit)
    rows = []
    for spec in ORGANS:
        dim = int(spec["fitted_dim"])
        under = int((fit < dim).sum())
        under_fit = int((n_fit < dim).sum())
        row: dict[str, Any] = {
            "organ": spec["name"],
            "x_kind": spec["x_kind"],
            "x_dim": spec["x_dim"],
            "w_shape": spec["w_shape"],
            "fitted_dim": dim,
            "note": spec["note"],
            "rows_distribution": summarize(fit, f"retained_rows_{spec['name']}"),
            "n_fit_after_holdout_25pct": summarize(n_fit, f"n_fit_{spec['name']}"),
            "pairs_rows_lt_fitted_dim": under,
            "pairs_rows_ge_fitted_dim": int((fit >= dim).sum()),
            "frac_underdetermined_vs_rows": under / N_PAIRS,
            "pairs_n_fit_lt_fitted_dim": under_fit,
            "frac_underdetermined_vs_n_fit": under_fit / N_PAIRS,
            "underdetermined": under > 0,
        }
        if "rank_target" in spec:
            rank = int(spec["rank_target"])
            row["rank_target"] = rank
            row["pairs_rows_lt_rank"] = int((fit < rank).sum())
            row["pairs_n_fit_lt_rank"] = int((n_fit < rank).sum())
            row["frac_underdetermined_vs_rank"] = int((fit < rank).sum()) / N_PAIRS
        rows.append(row)
    return rows


def audit(
    capture: Path,
    *,
    model_dir: Path,
    identity_receipt: Path | None,
) -> dict[str, Any]:
    cap = Path(capture)
    raw = census_from_index(cap)
    header = raw["header"]
    route = raw["route"]
    fit = raw["fit"]
    organs = organ_verdict(fit)
    swiglu = swiglu_presence(cap)
    provenance = verify_bf16_source(Path(model_dir), identity_receipt, header)
    never = raw["never_routed"]
    routed_no_fit = int(((route > 0) & (fit == 0)).sum())
    any_under = any(o["underdetermined"] for o in organs)
    down = next(o for o in organs if o["organ"] == "down_proj")
    gate = next(o for o in organs if o["organ"] == "gate_proj")
    verdict = {
        "capture_sufficient_for_wellposed_fits": False,
        "reason": (
            f"gate_proj/up_proj: {gate['pairs_rows_lt_fitted_dim']}/{N_PAIRS} "
            f"pairs have retained rows < 2048 "
            f"(median {gate['rows_distribution']['p50']}). "
            f"down_proj: {down['pairs_rows_lt_fitted_dim']}/{N_PAIRS} < 512 "
            f"and {down['pairs_rows_lt_rank']}/{N_PAIRS} < rank 160. "
            f"{len(never)} never-routed pairs cannot be fit at all. "
            f"down_proj post-SwiGLU X is "
            f"{'ON DISK' if swiglu['down_proj_x_on_disk'] else 'ABSENT'} "
            f"from this capture."
        ),
        "underdetermined_organs": [o["organ"] for o in organs if o["underdetermined"]],
        "never_routed_pairs": len(never),
        "never_routed_frac": len(never) / N_PAIRS,
        "swiglu_on_disk": bool(swiglu["down_proj_x_on_disk"]),
        "source_is_bf16": bool(provenance["ok"]),
    }
    if not any_under and swiglu["down_proj_x_on_disk"] and provenance["ok"] and not never:
        verdict["capture_sufficient_for_wellposed_fits"] = True
        verdict["reason"] = "every organ has rows >= fitted dim; swiglu present; BF16 source"
    return {
        "schema": SCHEMA,
        "status": "AUDITED",
        "measurement_label": "DIRTY_ENGINEERING",
        "capture": str(cap),
        "n_tokens": raw["n_tokens"],
        "n_layer_expert_pairs": N_PAIRS,
        "census_wall_s": raw["census_wall_s"],
        "index_dir": raw["index_dir"],
        "routing": {
            "distribution": summarize(route, "route_hits_per_layer_expert"),
            "histogram": occupancy_histogram(route),
            "never_routed_count": len(never),
            "never_routed_pairs": never,
            "observed_pairs": N_PAIRS - len(never),
            "observed_frac": (N_PAIRS - len(never)) / N_PAIRS,
            "routed_but_zero_retained": routed_no_fit,
            "pairs_route_ge_192_and_fit_lt_route": int(((route >= 192) & (fit < route)).sum()),
            "pairs_route_ge_2048": int((route >= 2048).sum()),
            "pairs_route_ge_512": int((route >= 512).sum()),
            "pairs_route_ge_160": int((route >= 160).sum()),
        },
        "retained_hidden": {
            "distribution": summarize(fit, "retained_hidden_rows_per_layer_expert"),
            "histogram": occupancy_histogram(fit),
            "pairs_fit_ge_2048": int((fit >= 2048).sum()),
            "pairs_fit_ge_512": int((fit >= 512).sum()),
            "pairs_fit_ge_160": int((fit >= 160).sum()),
            "matches_header_n_fit": (
                int((header.get("bounded_storage") or {}).get("n_fit_distribution", {}).get("p50") or -1)
                == int(summarize(fit, "x")["p50"])
            ),
        },
        "organs": organs,
        "swiglu": swiglu,
        "provenance": provenance,
        "verdict": verdict,
        "holdout": {"HOLD_FRAC": HOLD_FRAC, "rule": "holdout_split; n<4 keeps all rows"},
        "claim_boundary": {
            "does_not_refit_codecs": True,
            "does_not_change_representation": True,
            "did_not_parse_giant_json": True,
            "rows_are_retained_hidden_credits_including_coroute": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--identity-receipt", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    identity = args.identity_receipt if args.identity_receipt.is_file() else None
    report = audit(args.capture, model_dir=args.model_dir, identity_receipt=identity)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    else:
        print(text, end="")
    v = report["verdict"]
    print(
        f"verdict sufficient={v['capture_sufficient_for_wellposed_fits']} "
        f"underdetermined={v['underdetermined_organs']} "
        f"never_routed={v['never_routed_pairs']} "
        f"swiglu={v['swiglu_on_disk']} bf16={v['source_is_bf16']}",
        flush=True,
    )
    return 0 if v["source_is_bf16"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
