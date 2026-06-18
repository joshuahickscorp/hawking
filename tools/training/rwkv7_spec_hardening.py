"""Spec-decode hardening and shrink-frontier report for custom RWKV-7 drafts.

The goal is to turn the draft sweep into a physics gate:
  * accept rate must clear the useful floor,
  * K-wide verification must be cheap enough,
  * predicted effective TPS must beat the llama.cpp reference with margin,
  * if the smallest passing draft wins with margin, the next action is to shrink.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rwkv7_custom_configs import (  # noqa: E402
    CUSTOM_VARIANTS,
    VARIANT_ORDER,
    estimated_params,
    q4k_bytes_per_forward,
)


@dataclass(frozen=True)
class VariantMetrics:
    variant: str
    params_m: float
    bytes_mb: float
    draft_tps: float
    accept_rate: float | None
    ppl: float | None
    step: int | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def latest_eval(runs_dir: Path, variant: str) -> dict[str, Any] | None:
    rows = read_jsonl(runs_dir / f"custom_{variant}" / "eval_log.jsonl")
    return rows[-1] if rows else None


def estimated_draft_tps(variant: str, cap_tps: float, base_tps: float) -> float:
    """Estimate draft decode speed from Q4_K bytes, capped to avoid fantasy."""
    cfg = CUSTOM_VARIANTS[variant]
    base_cfg = type(cfg)(
        n_embd=1024,
        n_layer=24,
        n_ff=4096,
        head_dim=64,
        n_head=16,
        vocab_size=65536,
        decay_lora=64,
        iclr_lora=64,
        value_res_lora=32,
        gate_lora=128,
    )
    raw = base_tps * q4k_bytes_per_forward(base_cfg) / q4k_bytes_per_forward(cfg)
    return min(cap_tps, raw)


def expected_tokens_per_verify(accept: float, k: int) -> float:
    # One target token is always emitted; accepted draft chain adds p + p^2 + ...
    return sum(accept ** i for i in range(k + 1))


def effective_tps(target_tps: float, draft_tps: float, accept: float, k: int, verify_equiv: float) -> float:
    denom_s = verify_equiv / target_tps + k / draft_tps
    return expected_tokens_per_verify(accept, k) / denom_s


def break_even_accept(target_tps: float, draft_tps: float, llama_tps: float, k: int, verify_equiv: float) -> float | None:
    lo, hi = 0.0, 0.999
    if effective_tps(target_tps, draft_tps, hi, k, verify_equiv) < llama_tps:
        return None
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if effective_tps(target_tps, draft_tps, mid, k, verify_equiv) >= llama_tps:
            hi = mid
        else:
            lo = mid
    return hi


def load_metrics(runs_dir: Path, cap_tps: float, base_tps: float) -> list[VariantMetrics]:
    out: list[VariantMetrics] = []
    for variant in VARIANT_ORDER:
        cfg = CUSTOM_VARIANTS[variant]
        latest = latest_eval(runs_dir, variant) or {}
        accept = latest.get("draft_accept_rate")
        ppl = latest.get("wikitext2_ppl")
        out.append(
            VariantMetrics(
                variant=variant,
                params_m=estimated_params(cfg) / 1e6,
                bytes_mb=q4k_bytes_per_forward(cfg) / 1e6,
                draft_tps=estimated_draft_tps(variant, cap_tps=cap_tps, base_tps=base_tps),
                accept_rate=float(accept) if accept is not None else None,
                ppl=float(ppl) if ppl is not None else None,
                step=int(latest["step"]) if latest.get("step") is not None else None,
            )
        )
    return out


def status_for_variant(
    m: VariantMetrics,
    target_tps: float,
    llama_tps: float,
    k: int,
    verify_equiv: float,
    margin: float,
    accept_floor: float,
) -> dict[str, Any]:
    accept = m.accept_rate
    if accept is None:
        return {
            "variant": m.variant,
            "status": "PENDING",
            "reason": "no accept-rate eval yet",
            "effective_tps": None,
            "speedup_vs_llama": None,
        }
    eff = effective_tps(target_tps, m.draft_tps, accept, k, verify_equiv)
    ratio = eff / llama_tps
    if accept < accept_floor:
        status = "FAIL"
        reason = f"accept {accept:.2%} below floor {accept_floor:.0%}"
    elif ratio >= margin:
        status = "PASS"
        reason = f"predicted {eff:.1f} tok/s >= {margin:.2f}x llama"
    elif ratio >= 1.0:
        status = "WARN"
        reason = f"beats llama at {ratio:.2f}x but lacks {margin:.2f}x margin"
    else:
        status = "FAIL"
        reason = f"predicted {eff:.1f} tok/s trails llama {llama_tps:.1f}"
    return {
        "variant": m.variant,
        "status": status,
        "reason": reason,
        "effective_tps": eff,
        "speedup_vs_llama": ratio,
    }


def shrink_recommendation(rows: list[dict[str, Any]], metrics: list[VariantMetrics]) -> list[str]:
    evaluated = [r for r in rows if r.get("status") in {"PASS", "WARN"}]
    by_variant = {str(r["variant"]): r for r in rows}
    by_size = sorted(metrics, key=lambda m: m.params_m)
    if not evaluated:
        return [
            "Do not shrink yet: no evaluated draft clears the spec physics gate.",
            "First improve accept rate with target-logit KD, then re-run this hardening pass.",
            "Once a draft passes, launch the nearest smaller configured probe with `DRAFT_VARIANTS=\"draft_75m_probe draft_50m_probe\"` before scaling anything up.",
        ]
    size_by_variant = {m.variant: m.params_m for m in metrics}
    smallest = min(evaluated, key=lambda r: size_by_variant.get(str(r["variant"]), float("inf")))
    best = max(evaluated, key=lambda r: float(r.get("effective_tps") or 0.0))
    smallest_name = str(smallest["variant"])
    smaller = [m for m in by_size if m.params_m < size_by_variant.get(smallest_name, 0.0)]
    next_smaller = list(reversed(smaller))
    pending_smaller = [
        m for m in next_smaller
        if by_variant.get(m.variant, {}).get("status") == "PENDING"
    ]
    failed_smaller = next(
        (m for m in next_smaller if by_variant.get(m.variant, {}).get("status") == "FAIL"),
        None,
    )

    recs = [
        f"Extend `{best['variant']}` if the immediate goal is max TPS from the current sweep.",
    ]
    if smallest["status"] == "PASS" and failed_smaller:
        recs.append(f"Do not shrink below `{failed_smaller.variant}` yet; the nearest smaller evaluated point already failed.")
        recs.append(f"Use KD or architecture repair on `{failed_smaller.variant}` before probing below it.")
    elif smallest["status"] == "PASS" and pending_smaller:
        probes = " ".join(m.variant for m in pending_smaller[:2])
        recs.append(f"Shrink next: train `{probes}` with `DRAFT_VARIANTS=\"{probes}\"`; stop when accept <60% or predicted TPS loses the llama margin.")
        recs.append("Prefer reducing depth before width; only cut width when head geometry, vocab projection cost, and Metal alignment stay clean.")
    elif smallest["status"] == "PASS":
        recs.append("Lowest configured probe already passes; the next reduction needs an architectural move such as vocab pruning, tied embeddings, or a narrower 128-aligned experimental head.")
    elif smallest["status"] == "WARN":
        recs.append(f"Do one KD extension on `{smallest_name}` before shrinking; it is close but does not have enough margin.")
    else:
        recs.append("Do not shrink yet; the smallest evaluated point is still below the configured margin.")
    if pending_smaller:
        recs.append(f"Pending smaller probes already exist in config: `{', '.join(m.variant for m in pending_smaller)}`.")
    recs.append("Always measure `verify_equiv`: if K-wide verify costs more than ~1.5 target forwards, the draft win collapses.")
    return recs


def fmt(v: float | None, digits: int = 2, suffix: str = "") -> str:
    if v is None or math.isnan(v) or math.isinf(v):
        return "pending"
    return f"{v:.{digits}f}{suffix}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    args = payload["assumptions"]
    lines.append("# RWKV-7 Spec-Decode Hardening")
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Assumptions")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| target_tps | {args['target_tps']:.2f} |")
    lines.append(f"| llama_tps | {args['llama_tps']:.2f} |")
    lines.append(f"| K | {args['k']} |")
    lines.append(f"| verify_equiv | {args['verify_equiv']:.2f} target-forwards |")
    lines.append(f"| accept_floor | {args['accept_floor']:.0%} |")
    lines.append(f"| required margin | {args['margin']:.2f}x llama |")
    lines.append("")
    lines.append("## Variant Physics")
    lines.append("")
    lines.append("| Variant | Params M | Draft TPS est | Accept | PPL | Effective TPS | vs llama | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["rows"]:
        lines.append(
            f"| {row['variant']} | {fmt(row['params_m'], 1)} | {fmt(row['draft_tps'], 0)} | "
            f"{fmt(row.get('accept_rate_pct'), 2, '%')} | {fmt(row.get('ppl'), 2)} | "
            f"{fmt(row.get('effective_tps'), 1)} | {fmt(row.get('speedup_vs_llama'), 2, 'x')} | "
            f"**{row['status']}**: {row['reason']} |"
        )
    lines.append("")
    lines.append("## Break-Even Accept")
    lines.append("")
    lines.append("| Variant | Accept needed to match llama |")
    lines.append("|---|---:|")
    for row in payload["rows"]:
        lines.append(f"| {row['variant']} | {fmt(row.get('break_even_accept_pct'), 2, '%')} |")
    lines.append("")
    lines.append("## Compression Doctrine")
    lines.append("")
    lines.append("1. Promote the smallest draft that passes accept floor, predicted TPS margin, and measured verify cost.")
    lines.append("2. If the smallest passing draft is not the smallest configured probe, immediately train the next smaller probe before extending larger models.")
    lines.append("3. With untied 65k input/output embeddings and 256-aligned width, the practical configured floor is about 35M parameters; going below that requires changing the architecture, not just the layer count.")
    lines.append("")
    lines.append("## Shrink Rule")
    lines.append("")
    for i, rec in enumerate(payload["recommendations"], 1):
        lines.append(f"{i}. {rec}")
    lines.append("")
    lines.append("## Hard Rule")
    lines.append("")
    lines.append("No runtime spec-decode promotion unless an evaluated draft clears accept floor, predicted TPS margin, and measured K-wide verify cost.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs-dir", default=str(ROOT / "artifacts/lowbit_rwkv7/runs"))
    ap.add_argument("--target-tps", type=float, default=31.0)
    ap.add_argument("--llama-tps", type=float, default=50.0)
    ap.add_argument("--base-draft-tps", type=float, default=76.1, help="Measured 0.4B draft decode speed")
    ap.add_argument("--draft-tps-cap", type=float, default=400.0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--verify-equiv", type=float, default=1.15, help="K-wide verify cost in target-forward equivalents")
    ap.add_argument("--accept-floor", type=float, default=0.60)
    ap.add_argument("--margin", type=float, default=1.10, help="Required effective_tps / llama_tps")
    ap.add_argument("--out-md", default=str(ROOT / f"docs/plans/rwkv7_spec_hardening_{datetime.now(timezone.utc).strftime('%Y_%m_%d')}.md"))
    ap.add_argument("--out-json", default=str(ROOT / f"artifacts/lowbit_rwkv7/rwkv7_spec_hardening_{datetime.now(timezone.utc).strftime('%Y_%m_%d')}.json"))
    args = ap.parse_args()

    metrics = load_metrics(Path(args.runs_dir), cap_tps=args.draft_tps_cap, base_tps=args.base_draft_tps)
    rows: list[dict[str, Any]] = []
    for m in metrics:
        verdict = status_for_variant(
            m,
            target_tps=args.target_tps,
            llama_tps=args.llama_tps,
            k=args.k,
            verify_equiv=args.verify_equiv,
            margin=args.margin,
            accept_floor=args.accept_floor,
        )
        be = break_even_accept(args.target_tps, m.draft_tps, args.llama_tps, args.k, args.verify_equiv)
        rows.append({
            "variant": m.variant,
            "params_m": m.params_m,
            "bytes_mb": m.bytes_mb,
            "draft_tps": m.draft_tps,
            "accept_rate": m.accept_rate,
            "accept_rate_pct": 100.0 * m.accept_rate if m.accept_rate is not None else None,
            "ppl": m.ppl,
            "step": m.step,
            "break_even_accept": be,
            "break_even_accept_pct": 100.0 * be if be is not None else None,
            **verdict,
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assumptions": {
            "target_tps": args.target_tps,
            "llama_tps": args.llama_tps,
            "base_draft_tps": args.base_draft_tps,
            "draft_tps_cap": args.draft_tps_cap,
            "k": args.k,
            "verify_equiv": args.verify_equiv,
            "accept_floor": args.accept_floor,
            "margin": args.margin,
        },
        "rows": rows,
        "recommendations": shrink_recommendation(rows, metrics),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = Path(args.out_md)
    write_markdown(out_md, payload)
    print(f"[hardening] wrote {out_md}")
    print(f"[hardening] wrote {out_json}")


if __name__ == "__main__":
    main()
