"""Build the final RWKV-7 competitive scorecard.

This is intentionally a report generator, not a benchmark runner. The chain may
or may not have a clean-room llama.cpp log, TQ artifact, or finished draft sweep;
the scorecard turns those facts into an explicit frontier verdict and backlog.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def fmt_num(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "pending"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v) or math.isinf(v):
        return "pending"
    return f"{v:.{digits}f}{suffix}"


def latest_match_float(path: Path, pattern: str) -> float | None:
    if not path.exists():
        return None
    rx = re.compile(pattern)
    found: float | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = rx.search(line)
        if m:
            try:
                found = float(m.group(1))
            except ValueError:
                pass
    return found


def g1a_summary(events_path: Path, watcher_log: Path, target_steps: int) -> dict[str, Any]:
    events = [r for r in read_jsonl(events_path) if isinstance(r.get("step"), int)]
    if not events:
        return {"status": "pending", "step": None, "target_steps": target_steps}

    events.sort(key=lambda r: r["step"])
    latest = events[-1]
    step = int(latest["step"])
    timestamps = [(int(r["step"]), parse_iso(r.get("timestamp"))) for r in events]
    timestamps = [(s, t) for s, t in timestamps if t is not None]

    deltas_min: list[float] = []
    for (_, prev), (_, cur) in zip(timestamps, timestamps[1:]):
        assert prev is not None and cur is not None
        delta = (cur - prev).total_seconds() / 60.0
        if delta > 0:
            deltas_min.append(delta)
    recent = deltas_min[-8:] or deltas_min
    min_per_step = mean(recent) if recent else None
    remaining_steps = max(0, target_steps - step)
    eta_hours = (remaining_steps * min_per_step / 60.0) if min_per_step else None

    watcher_ppl = latest_match_float(watcher_log, r"\bPPL:\s*([0-9]+(?:\.[0-9]+)?)")
    return {
        "status": "complete" if step >= target_steps else "running",
        "step": step,
        "target_steps": target_steps,
        "loss": latest.get("loss"),
        "loss_ema": latest.get("loss_ema"),
        "tok_s": latest.get("tok_s"),
        "recent_min_per_step": min_per_step,
        "remaining_steps": remaining_steps,
        "eta_hours": eta_hours,
        "latest_watcher_ppl": watcher_ppl,
    }


def draft_summaries(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(runs_dir.glob("custom_draft_*")):
        eval_log = run / "eval_log.jsonl"
        records = read_jsonl(eval_log)
        if not records:
            rows.append({
                "variant": run.name.removeprefix("custom_"),
                "status": "pending",
                "eval_log": str(eval_log),
            })
            continue
        latest = records[-1]
        latest = dict(latest)
        latest["status"] = "evaluated"
        latest["eval_log"] = str(eval_log)
        rows.append(latest)
    order = {
        "draft_35m_probe": 0,
        "draft_50m_probe": 1,
        "draft_75m_probe": 2,
        "draft_100m": 3,
        "draft_150m": 4,
        "draft_200m": 5,
        "draft_300m": 6,
    }
    rows.sort(key=lambda r: order.get(str(r.get("variant")), 99))
    return rows


def parse_llama_head_to_head(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"status": "missing", "log": str(log_path)}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {"status": "present", "log": str(log_path)}

    for engine, key in (("dismantle", "dismantle_dec_tps"), ("llama\\.cpp", "llama_dec_tps")):
        m = re.search(rf"^\s*{engine}\s+dec_tps:\s*([0-9]+(?:\.[0-9]+)?)", text, re.M)
        if m:
            out[key] = float(m.group(1))

    ratio = None
    if out.get("dismantle_dec_tps") and out.get("llama_dec_tps"):
        ratio = float(out["dismantle_dec_tps"]) / float(out["llama_dec_tps"])
    else:
        m = re.search(r"ratio\(d[÷/]l\)\s+dec_tps:\s*([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            ratio = float(m.group(1))
    out["speed_ratio_d_over_llama"] = ratio

    aggregate: list[dict[str, Any]] = []
    row_rx = re.compile(r"^\s*(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)", re.M)
    for b, dis, llama, ratio_s in row_rx.findall(text):
        aggregate.append({
            "batch": int(b),
            "dismantle_tps": float(dis),
            "llama_tps": float(llama),
            "ratio": float(ratio_s),
        })
    out["aggregate"] = aggregate
    return out


def readme_baseline(readme: Path) -> dict[str, Any]:
    if not readme.exists():
        return {}
    text = readme.read_text(encoding="utf-8", errors="ignore")
    approx = "\u2248"
    m = re.search(
        rf"llama\.cpp Metal reaches {approx}([0-9]+).*?dismantle reaches {approx}([0-9]+)",
        text,
        re.S,
    )
    if not m:
        return {}
    return {"llama_dec_tps": float(m.group(1)), "dismantle_dec_tps": float(m.group(2))}


def tq_summary(export_dir: Path, v2_report_dir: Path) -> dict[str, Any]:
    artifacts = sorted(str(p) for p in export_dir.glob("**/*") if p.is_file())
    reports = sorted(v2_report_dir.glob("g1a_v2_expansion_results_*.md"))
    latest_report = reports[-1] if reports else None
    report_text = latest_report.read_text(encoding="utf-8", errors="ignore") if latest_report else ""
    return {
        "artifact_count": len(artifacts),
        "artifacts": artifacts[:12],
        "latest_v2_report": str(latest_report) if latest_report else None,
        "tq_loader_pass": "| rwkv7 tq loader | PASS |" in report_text,
        "tq_bench_pass": "| rwkv7 tq bench | PASS |" in report_text,
    }


def latest_hardening(path: Path | None, artifact_dir: Path) -> dict[str, Any]:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"status": "unreadable", "path": str(path)}
    candidates = sorted(artifact_dir.glob("rwkv7_spec_hardening_*.json"))
    if not candidates:
        return {"status": "missing"}
    latest = candidates[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        data["path"] = str(latest)
        return data
    except json.JSONDecodeError:
        return {"status": "unreadable", "path": str(latest)}


def hardening_verdict(hardening: dict[str, Any]) -> tuple[str, str, list[str]]:
    rows = hardening.get("rows") or []
    recs = [str(x) for x in hardening.get("recommendations") or []]
    if not rows:
        return "PENDING", "No spec hardening report yet.", recs
    statuses = [str(r.get("status")) for r in rows]
    passes = [r for r in rows if r.get("status") == "PASS"]
    warns = [r for r in rows if r.get("status") == "WARN"]
    if passes:
        best = max(passes, key=lambda r: float(r.get("effective_tps") or 0.0))
        return "PASS", f"{best.get('variant')} clears spec physics at {float(best.get('effective_tps')):.1f} tok/s.", recs
    if warns:
        best = max(warns, key=lambda r: float(r.get("effective_tps") or 0.0))
        return "WARN", f"{best.get('variant')} is close but lacks configured speed margin.", recs
    if all(s == "PENDING" for s in statuses):
        return "PENDING", "Draft eval logs are not available to the hardening model yet.", recs
    return "FAIL", "No evaluated draft clears accept, speed, and verify-cost physics.", recs


def verdicts(
    g1a: dict[str, Any],
    drafts: list[dict[str, Any]],
    llama: dict[str, Any],
    tq: dict[str, Any],
    hardening: dict[str, Any],
) -> tuple[list[tuple[str, str, str]], list[str]]:
    gates: list[tuple[str, str, str]] = []
    backlog: list[str] = []

    ppl = g1a.get("latest_watcher_ppl")
    if ppl is None:
        gates.append(("Quality", "PENDING", "No final/step watcher PPL parsed yet."))
        backlog.append("Finish G1a and keep the watcher PPL gate as the quality source of truth.")
    elif g1a.get("status") != "complete":
        gates.append(("Quality", "PENDING", f"Current checkpoint PPL {ppl:.2f}; final G1a checkpoint not reached yet."))
        backlog.append("Finish G1a and keep the watcher PPL gate as the quality source of truth.")
    elif ppl <= 13.56:
        gates.append(("Quality", "PASS", f"Latest watcher PPL {ppl:.2f} is inside the G1b gate (<=13.56)."))
    elif ppl <= 16.95:
        gates.append(("Quality", "WARN", f"Latest watcher PPL {ppl:.2f} is usable but needs more QAT/training."))
        backlog.append("Run/extend the next QAT stage before claiming quality parity.")
    else:
        gates.append(("Quality", "FAIL", f"Latest watcher PPL {ppl:.2f} misses all gates."))
        backlog.append("Stop promotion and debug the QAT regression before speed work.")

    if tq.get("tq_loader_pass") and tq.get("tq_bench_pass"):
        gates.append(("Low-bit quant", "PASS", "TQ loader and bench passed in the v2 report."))
    elif tq.get("artifact_count"):
        gates.append(("Low-bit quant", "WARN", "TQ artifacts exist, but loader/bench pass was not found."))
        backlog.append("Make TQ parity and speed gates mandatory before serving the low-bit artifact.")
    else:
        gates.append(("Low-bit quant", "PENDING", "No exported TQ artifact yet."))
        backlog.append("Finish TQ export/loader/dispatch after G1a final if the quality gate passes.")

    evaluated = [d for d in drafts if d.get("draft_accept_rate") is not None]
    if evaluated:
        best = max(evaluated, key=lambda d: float(d.get("draft_accept_rate", 0.0)))
        accept = float(best["draft_accept_rate"])
        if accept >= 0.70:
            gates.append(("Draft accept", "PASS", f"{best.get('variant')} accept={accept:.2%}; strong enough for spec-decode."))
        elif accept >= 0.60:
            gates.append(("Draft accept", "WARN", f"{best.get('variant')} accept={accept:.2%}; break-even likely but tune further."))
            backlog.append("Extend the winning draft with KD and re-check accept rate against the real 3B target.")
        else:
            gates.append(("Draft accept", "FAIL", f"Best draft accept={accept:.2%}; below the useful spec-decode floor."))
            backlog.append("Distill from the target logits; do not wire runtime spec-decode until accept >=60%.")
    else:
        gates.append(("Draft accept", "PENDING", "No custom draft eval logs yet."))
        backlog.append("Let the 100/150/200/300M sweep finish, then extend the winner instead of all four.")

    spec_status, spec_evidence, spec_recs = hardening_verdict(hardening)
    gates.append(("Spec physics", spec_status, spec_evidence))
    if spec_status != "PASS":
        backlog.extend(spec_recs[:3])

    ratio = llama.get("speed_ratio_d_over_llama")
    if ratio is None:
        gates.append(("llama.cpp comparison", "PENDING", "No clean-room llama head-to-head log parsed."))
        backlog.append("Run `G1A_V2_LLAMA_BASELINE=1` in a clean room; this is the claim gate.")
    elif ratio >= 1.10:
        gates.append(("llama.cpp comparison", "PASS", f"dismantle/llama single-stream ratio={ratio:.3f}x."))
    elif ratio >= 1.0:
        gates.append(("llama.cpp comparison", "WARN", f"dismantle is barely ahead at {ratio:.3f}x; needs margin."))
        backlog.append("Re-run clean-room with more trials and batch sweep; do not market a tiny/noisy win.")
    else:
        gates.append(("llama.cpp comparison", "FAIL", f"dismantle trails llama.cpp at {ratio:.3f}x."))
        backlog.append("Use Metal System Trace to isolate llama's remaining GEMV/kernel advantage before more micro-opts.")

    # Keep backlog unique and in priority order.
    seen: set[str] = set()
    uniq = []
    for item in backlog:
        if item not in seen:
            uniq.append(item)
            seen.add(item)
    return gates, uniq


def write_report(path: Path, payload: dict[str, Any]) -> None:
    g1a = payload["g1a"]
    drafts = payload["drafts"]
    llama = payload["llama"]
    tq = payload["tq"]
    hardening = payload["hardening"]
    gates = payload["gates"]
    backlog = payload["backlog"]
    readme = payload["readme_baseline"]

    lines: list[str] = []
    lines.append("# RWKV-7 Competitive Scorecard")
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Verdict Gates")
    lines.append("")
    lines.append("| Lane | Verdict | Evidence |")
    lines.append("|---|---|---|")
    for lane, verdict, evidence in gates:
        lines.append(f"| {lane} | **{verdict}** | {evidence} |")
    lines.append("")
    lines.append("## G1a State")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| step | {g1a.get('step') or 'pending'} / {g1a.get('target_steps')} |")
    lines.append(f"| loss | {fmt_num(g1a.get('loss'), 4)} |")
    lines.append(f"| loss_ema | {fmt_num(g1a.get('loss_ema'), 4)} |")
    lines.append(f"| watcher PPL | {fmt_num(g1a.get('latest_watcher_ppl'), 2)} |")
    lines.append(f"| recent min/step | {fmt_num(g1a.get('recent_min_per_step'), 1)} |")
    lines.append(f"| ETA remaining | {fmt_num(g1a.get('eta_hours'), 1, 'h')} |")
    lines.append("")
    lines.append("## Draft Sweep")
    lines.append("")
    lines.append("| Variant | Step | PPL | Accept rate | Params M |")
    lines.append("|---|---:|---:|---:|---:|")
    if drafts:
        for d in drafts:
            lines.append(
                f"| {d.get('variant')} | {d.get('step', 'pending')} | "
                f"{fmt_num(d.get('wikitext2_ppl'), 2)} | "
                f"{fmt_num(100.0 * float(d['draft_accept_rate']), 2, '%') if d.get('draft_accept_rate') is not None else 'pending'} | "
                f"{fmt_num(d.get('params_M'), 1)} |"
            )
    else:
        lines.append("| pending | pending | pending | pending | pending |")
    lines.append("")
    lines.append("## llama.cpp Head-to-Head")
    lines.append("")
    if llama.get("status") == "present":
        lines.append(f"- Log: `{llama.get('log')}`")
        lines.append(f"- Single-stream dismantle: {fmt_num(llama.get('dismantle_dec_tps'), 2)} tok/s")
        lines.append(f"- Single-stream llama.cpp: {fmt_num(llama.get('llama_dec_tps'), 2)} tok/s")
        lines.append(f"- Ratio: {fmt_num(llama.get('speed_ratio_d_over_llama'), 3)}x")
    else:
        lines.append("- Clean-room llama.cpp log: pending.")
        if readme:
            lines.append(
                f"- README baseline: dismantle ~{readme.get('dismantle_dec_tps'):.0f} tok/s, "
                f"llama.cpp ~{readme.get('llama_dec_tps'):.0f} tok/s on Qwen Q4."
            )
    lines.append("")
    lines.append("## Low-Bit Quant")
    lines.append("")
    lines.append(f"- Artifact count: {tq.get('artifact_count', 0)}")
    lines.append(f"- Latest v2 report: `{tq.get('latest_v2_report') or 'pending'}`")
    lines.append(f"- TQ loader pass: {tq.get('tq_loader_pass')}")
    lines.append(f"- TQ bench pass: {tq.get('tq_bench_pass')}")
    lines.append("")
    lines.append("## Spec Physics")
    lines.append("")
    if hardening.get("rows"):
        lines.append("| Variant | Effective TPS | vs llama | Status |")
        lines.append("|---|---:|---:|---|")
        for row in hardening.get("rows", []):
            lines.append(
                f"| {row.get('variant')} | {fmt_num(row.get('effective_tps'), 1)} | "
                f"{fmt_num(row.get('speedup_vs_llama'), 2, 'x')} | "
                f"**{row.get('status')}**: {row.get('reason')} |"
            )
    else:
        lines.append("- Spec hardening report: pending.")
    lines.append("")
    lines.append("## Development Backlog")
    lines.append("")
    if backlog:
        for i, item in enumerate(backlog, 1):
            lines.append(f"{i}. {item}")
    else:
        lines.append("1. All evidence gates are green; lock a clean-room release report and freeze the winning config.")
    lines.append("")
    lines.append("## Claim Rule")
    lines.append("")
    lines.append(
        "Do not claim a llama.cpp win until quality, low-bit quant, draft accept, "
        "spec physics, and clean-room llama head-to-head are all green in this report."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--events", default=str(ROOT / "artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/events.jsonl"))
    ap.add_argument("--watcher-log", default=str(ROOT / "artifacts/lowbit_rwkv7/g1a_watcher.log"))
    ap.add_argument("--runs-dir", default=str(ROOT / "artifacts/lowbit_rwkv7/runs"))
    ap.add_argument("--llama-log", default=str(ROOT / "artifacts/lowbit_rwkv7/v2_expansion/llama_qwen_head_to_head.log"))
    ap.add_argument("--export-dir", default=str(ROOT / "artifacts/lowbit_rwkv7/export/g1a"))
    ap.add_argument("--v2-report-dir", default=str(ROOT / "docs/plans"))
    ap.add_argument("--hardening-json", default=None)
    ap.add_argument("--target-steps", type=int, default=150)
    ap.add_argument("--out-md", default=str(ROOT / f"docs/plans/rwkv7_competitive_scorecard_{datetime.now(timezone.utc).strftime('%Y_%m_%d')}.md"))
    ap.add_argument("--out-json", default=str(ROOT / f"artifacts/lowbit_rwkv7/rwkv7_competitive_scorecard_{datetime.now(timezone.utc).strftime('%Y_%m_%d')}.json"))
    args = ap.parse_args()

    g1a = g1a_summary(Path(args.events), Path(args.watcher_log), args.target_steps)
    drafts = draft_summaries(Path(args.runs_dir))
    llama = parse_llama_head_to_head(Path(args.llama_log))
    tq = tq_summary(Path(args.export_dir), Path(args.v2_report_dir))
    hardening = latest_hardening(
        Path(args.hardening_json) if args.hardening_json else None,
        ROOT / "artifacts/lowbit_rwkv7",
    )
    readme = readme_baseline(ROOT / "README.md")
    gates, backlog = verdicts(g1a, drafts, llama, tq, hardening)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "g1a": g1a,
        "drafts": drafts,
        "llama": llama,
        "tq": tq,
        "hardening": hardening,
        "readme_baseline": readme,
        "gates": gates,
        "backlog": backlog,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = Path(args.out_md)
    write_report(out_md, payload)
    print(f"[scorecard] wrote {out_md}")
    print(f"[scorecard] wrote {out_json}")


if __name__ == "__main__":
    main()
