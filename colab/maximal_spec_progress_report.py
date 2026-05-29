#!/usr/bin/env python3
"""Summarize and optionally export maximal_spec_decode_500u progress.

Run this in Colab against the Drive lab root, or locally against a copied
Drive folder. The report is intentionally small and text-first so a training
run can be interpreted without scrolling through notebook output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_LAB_ROOT = Path("/content/drive/MyDrive/dismantle/maximal_spec_500u")


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        return {"_error": f"json decode failed: {exc}"}


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "bytes": int(st.st_size),
        "mtime_unix": int(st.st_mtime),
    }


def corpus_inventory(lab_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus_dir in sorted((lab_root / "corpora").glob("*_corpus")):
        target = corpus_dir.name.removesuffix("_corpus")
        shards = sorted(corpus_dir.glob("shard_*.npz"))
        if not shards:
            shards = sorted(
                p
                for p in corpus_dir.glob("*.npz")
                if p.name != "per_site_activation_stats.npz"
            )
        rows.append(
            {
                "target": target,
                "path": str(corpus_dir),
                "shards": len(shards),
                "stats": file_info(corpus_dir / "per_site_activation_stats.npz"),
            }
        )
    return rows


def checkpoint_inventory(lab_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    artifacts = lab_root / "artifacts"
    for target_dir in sorted(p for p in artifacts.glob("*") if p.is_dir()):
        target = target_dir.name
        ckpt_root = target_dir / "checkpoints"
        for ckpt_dir in sorted(p for p in ckpt_root.glob("*") if p.is_dir()):
            tag = ckpt_dir.name
            eval_dir = target_dir / "eval" / tag
            tau = load_json(eval_dir / "tau.json", {})
            frontier = load_json(eval_dir / "frontier.json", {})
            best = frontier.get("policies", {}).get("best_deployable", {})
            rows.append(
                {
                    "target": target,
                    "tag": tag,
                    "checkpoint_dir": str(ckpt_dir),
                    "head": file_info(ckpt_dir / "head_final.safetensors"),
                    "latest": file_info(ckpt_dir / "latest.npz"),
                    "tau_path": str(eval_dir / "tau.json"),
                    "frontier_path": str(eval_dir / "frontier.json"),
                    "tau": tau.get("tau"),
                    "depth1_accept_rate": tau.get("depth1_accept_rate"),
                    "accepted_draft_tokens_per_verify": best.get(
                        "accepted_draft_tokens_per_verify"
                    ),
                    "offline_projected_tps": best.get("projected_dec_tps"),
                    "policy_kind": best.get("kind"),
                }
            )
    return rows


def leaderboard_rows(lab_root: Path) -> list[dict[str, Any]]:
    data = load_json(lab_root / "leaderboard.json", {})
    rows = data.get("rows", []) if isinstance(data, dict) else []
    return rows if isinstance(rows, list) else []


def progress_inventory(lab_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(lab_root.glob("progress*.json")):
        data = load_json(path, {})
        keys = sorted(data.keys()) if isinstance(data, dict) else []
        rows.append(
            {
                "path": str(path),
                "stages": keys,
                "stage_count": len(keys),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def build_summary(lab_root: Path) -> dict[str, Any]:
    checkpoints = checkpoint_inventory(lab_root)
    finals = [r for r in checkpoints if r["head"]["exists"]]
    partials = [r for r in checkpoints if r["latest"]["exists"] and not r["head"]["exists"]]
    leaderboard = leaderboard_rows(lab_root)
    return {
        "schema": "dismantle-maximal-spec-progress-report-v1",
        "created_at_unix": int(time.time()),
        "lab_root": str(lab_root),
        "progress_files": progress_inventory(lab_root),
        "corpora": corpus_inventory(lab_root),
        "checkpoint_count": len(checkpoints),
        "final_head_count": len(finals),
        "partial_head_count": len(partials),
        "checkpoints": checkpoints,
        "leaderboard": leaderboard,
    }


def fmt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(summary: dict[str, Any], top: int) -> str:
    lines: list[str] = []
    lines.append("# Maximal Spec Progress")
    lines.append("")
    lines.append(f"Lab root: `{summary['lab_root']}`")
    lines.append(
        f"Final heads: {summary['final_head_count']} | "
        f"partial heads: {summary['partial_head_count']} | "
        f"checkpoints: {summary['checkpoint_count']}"
    )
    lines.append("")

    lines.append("## Progress Files")
    if summary["progress_files"]:
        lines.append("| file | stages |")
        lines.append("| --- | ---: |")
        for row in summary["progress_files"]:
            lines.append(f"| `{Path(row['path']).name}` | {row['stage_count']} |")
    else:
        lines.append("No progress files found.")
    lines.append("")

    lines.append("## Corpora")
    if summary["corpora"]:
        lines.append("| target | shards | stats |")
        lines.append("| --- | ---: | --- |")
        for row in summary["corpora"]:
            stats = "yes" if row["stats"]["exists"] else "no"
            lines.append(f"| {row['target']} | {row['shards']} | {stats} |")
    else:
        lines.append("No corpora found.")
    lines.append("")

    rows = summary["leaderboard"][:top]
    lines.append("## Leaderboard")
    if rows:
        lines.append("| target | tag | tau | acc1 | accepted/verify | projected tps | policy |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt_value(row.get("target")),
                        f"`{fmt_value(row.get('tag'))}`",
                        fmt_value(row.get("tau")),
                        fmt_value(row.get("depth1_accept_rate")),
                        fmt_value(row.get("accepted_draft_tokens_per_verify")),
                        fmt_value(row.get("offline_projected_tps")),
                        fmt_value(row.get("policy_kind")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No leaderboard rows yet. Run Cell 5 after training heads finish.")
    lines.append("")

    lines.append("## Final Heads")
    finals = [r for r in summary["checkpoints"] if r["head"]["exists"]]
    if finals:
        lines.append("| target | tag | tau | projected tps | head |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for row in finals[:top]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["target"],
                        f"`{row['tag']}`",
                        fmt_value(row.get("tau")),
                        fmt_value(row.get("offline_projected_tps")),
                        f"`{rel(Path(row['head']['path']), Path(summary['lab_root']))}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No final heads found.")
    lines.append("")

    partials = [r for r in summary["checkpoints"] if r["latest"]["exists"] and not r["head"]["exists"]]
    if partials:
        lines.append("## Partial Heads")
        for row in partials[:top]:
            lines.append(f"- {row['target']} `{row['tag']}` has `latest.npz` but no final head.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def copy_if_exists(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return {"source": str(src), "exported": str(dst), "bytes": int(dst.stat().st_size)}


def export_bundle(
    summary: dict[str, Any],
    report_text: str,
    export_dir: Path,
    include_heads: bool,
    top: int,
) -> dict[str, Any]:
    lab_root = Path(summary["lab_root"])
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "dismantle-maximal-spec-progress-export-v1",
        "created_at_unix": int(time.time()),
        "lab_root": str(lab_root),
        "export_dir": str(export_dir),
        "include_heads": include_heads,
        "files": {},
        "missing": {},
    }
    write_text_atomic(export_dir / "maximal_spec_progress_report.md", report_text)
    write_json_atomic(export_dir / "maximal_spec_progress_report.json", summary)

    candidates: list[tuple[str, Path, Path]] = []
    for path in sorted(lab_root.glob("progress*.json")):
        candidates.append((path.name, path, Path("metadata") / path.name))
    for name in (
        "leaderboard.json",
        "maximal_spec_summary.md",
        "maximal_spec_summary.json",
        "export_manifest.json",
    ):
        candidates.append((name, lab_root / name, Path("metadata") / name))

    selected = summary["leaderboard"][:top]
    if not selected:
        selected = [r for r in summary["checkpoints"] if r["head"]["exists"]][:top]
    for row in selected:
        target = row.get("target", "unknown")
        tag = row.get("tag", Path(str(row.get("head", ""))).parent.name)
        for key in ("tau_path", "frontier_path"):
            value = row.get(key)
            if value:
                candidates.append(
                    (
                        f"{target}_{tag}_{Path(value).name}",
                        Path(value),
                        Path("eval") / target / tag / Path(value).name,
                    )
                )
        head_value = row.get("head")
        if include_heads and isinstance(head_value, str):
            candidates.append(
                (
                    f"{target}_{tag}_head",
                    Path(head_value),
                    Path("heads") / target / f"{tag}.safetensors",
                )
            )
        elif include_heads and isinstance(head_value, dict) and head_value.get("path"):
            candidates.append(
                (
                    f"{target}_{tag}_head",
                    Path(head_value["path"]),
                    Path("heads") / target / f"{tag}.safetensors",
                )
            )

    for name, src, rel_dst in candidates:
        copied = copy_if_exists(src, export_dir / rel_dst)
        if copied is None:
            manifest["missing"][name] = str(src)
        else:
            manifest["files"][name] = copied

    write_json_atomic(export_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-root", type=Path, default=DEFAULT_LAB_ROOT)
    parser.add_argument("--out", type=Path, help="Write markdown report here.")
    parser.add_argument("--json-out", type=Path, help="Write machine-readable summary here.")
    parser.add_argument("--export-dir", type=Path, help="Copy metadata and selected eval files here.")
    parser.add_argument("--include-heads", action="store_true", help="Also copy selected head files.")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    summary = build_summary(args.lab_root)
    report_text = render_markdown(summary, args.top)

    if args.out:
        write_text_atomic(args.out, report_text)
        print(f"wrote {args.out}")
    else:
        print(report_text)

    if args.json_out:
        write_json_atomic(args.json_out, summary)
        print(f"wrote {args.json_out}")

    if args.export_dir:
        manifest = export_bundle(
            summary,
            report_text,
            args.export_dir,
            include_heads=args.include_heads,
            top=args.top,
        )
        copied = len(manifest["files"])
        missing = len(manifest["missing"])
        total = sum(item["bytes"] for item in manifest["files"].values())
        print(f"exported {copied} files ({total / 1e9:.2f} GB), missing={missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
