#!/usr/bin/env python3.12
"""Repository census (CLEAN SLATE Section 13) - the honest measurement before condensation.

Counts ONLY git-tracked files (the real ship / maintain / review / build surface; gitignored reports
and build outputs are excluded), classifies every tracked file, and estimates the irreducible
kernel. This is heuristic (directory + extension + light import signal), labelled as such; a deeper
AST/call-graph pass is a refinement. Writes CODEBASE_CENSUS.{json,md}, PACK_CANDIDATES.json, and
IRREDUCIBLE_KERNEL_ESTIMATE.json.

No file is moved or deleted here. Census is read-only. Relocated != eliminated; generated != free;
archived != deleted (Section 12).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

OUT = Path("reports/condense/gravity_forge/condensation")

# classification buckets (Section 13)
CLASSES = ("kernel", "active_product", "default_validation", "optional_adapter", "laboratory",
           "compatibility", "documentation", "generated", "fixture", "example", "asset",
           "third_party", "dead", "unknown")

CODE_EXT = {".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".metal", ".sh"}
DOC_EXT = {".md", ".rst", ".txt"}
CFG_EXT = {".toml", ".yaml", ".yml", ".json", ".lock", ".cfg", ".ini"}
ASSET_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".icns", ".woff", ".woff2", ".ttf",
             ".mp4", ".webp", ".pdf", ".bin", ".gguf", ".safetensors"}


def _loc(path: str) -> int:
    try:
        return sum(1 for _ in open(path, encoding="utf-8", errors="ignore"))
    except Exception:
        return 0


def classify(f: str) -> str:
    p = f.lower()
    ext = os.path.splitext(f)[1].lower()
    top = f.split("/")[0]
    if p.startswith("vendor/") or p.startswith("tools/strand") or "third_party" in p or "node_modules" in p:
        return "third_party"
    if ext in ASSET_EXT:
        return "asset"
    if ext in DOC_EXT or top == "docs":
        return "documentation"
    if "/tests/" in p or p.startswith("tests/") or "test_" in os.path.basename(p) or p.endswith("_test.rs"):
        return "default_validation"
    if "/fixtures/" in p or "/fixture" in p or "/goldens/" in p or "/snapshots/" in p:
        return "fixture"
    if "/examples/" in p or p.startswith("examples/") or "/example" in p:
        return "example"
    if any(k in p for k in ("/lab", "experiment", "scratch", "sketch", "_poc", "prototype")):
        return "laboratory"
    if any(k in p for k in ("legacy", "compat", "deprecated", "archive")):
        return "compatibility"
    if ext in CFG_EXT:
        return "generated" if any(k in p for k in (".lock", "generated", "_gen.")) else "kernel"
    # code files: split kernel vs product vs adapter vs lab by location
    if ext in CODE_EXT:
        if top == "app" or "frontend" in p or "hide" in p or "desktop" in p or "src-tauri" in p:
            return "active_product"
        if "adapter" in p:
            return "optional_adapter"
        if top in ("crates", "src"):
            return "kernel"
        if top in ("tools", "scaffolding", "profiles", "prompts"):
            # the condense/controller/forge/doctor pipeline; kernel-ish but heavy with labs
            return "laboratory" if any(k in p for k in ("experiment", "sweep", "_scaffold",
                                                        "frontier_autopilot", "preview", "_probe")) else "kernel"
        return "unknown"
    return "unknown"


def build() -> dict[str, Any]:
    files = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n")
             if f and os.path.isfile(f)]
    by_class_loc: Counter = Counter()
    by_class_files: Counter = Counter()
    by_dir_loc: Counter = Counter()
    by_ext_loc: Counter = Counter()
    per_file = []
    for f in files:
        loc = _loc(f)
        cls = classify(f)
        ext = os.path.splitext(f)[1] or "(none)"
        top = f.split("/")[0] if "/" in f else "(root)"
        by_class_loc[cls] += loc
        by_class_files[cls] += 1
        by_dir_loc[top] += loc
        by_ext_loc[ext] += loc
        per_file.append({"f": f, "loc": loc, "class": cls})
    total = sum(by_class_loc.values())

    # kernel estimate = code that is kernel + optional_adapter core; product adds active_product
    kernel = by_class_loc["kernel"]
    active_product = kernel + by_class_loc["active_product"] + by_class_loc["optional_adapter"]
    packable = (by_class_loc["laboratory"] + by_class_loc["documentation"] + by_class_loc["fixture"]
                + by_class_loc["example"] + by_class_loc["compatibility"] + by_class_loc["asset"])

    census = {
        "schema": "hawking.codebase_census.v1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "git-tracked files only; heuristic directory+extension+light-import classification",
        "tracked_files": len(files), "total_tracked_loc": total,
        "loc_by_class": dict(by_class_loc.most_common()),
        "files_by_class": dict(by_class_files.most_common()),
        "loc_by_top_dir": dict(by_dir_loc.most_common()),
        "loc_by_ext": dict(by_ext_loc.most_common(20)),
        "kernel_loc_estimate": kernel, "active_product_loc_estimate": active_product,
        "packable_loc": packable,
        "targets": {"kernel": "50k-75k", "product": "65k-90k"},
        "note": "Relocated != eliminated, generated != free, archived != deleted. Heuristic pass; "
                "a deeper AST/call/import-graph classification will refine kernel vs laboratory.",
    }
    census["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in census.items() if k != "sha256"},
                                                 sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CODEBASE_CENSUS.json").write_text(json.dumps(census, indent=2, sort_keys=True, default=str))

    # PACK_CANDIDATES: largest non-kernel surfaces to seal into offline packs (Section 21)
    packs = {"hawking-docs-archive": by_class_loc["documentation"], "hawking-lab": by_class_loc["laboratory"],
             "hawking-fixtures": by_class_loc["fixture"], "hawking-compat": by_class_loc["compatibility"],
             "hawking-hide-desktop": by_class_loc["active_product"], "hawking-adapters-extra": by_class_loc["optional_adapter"],
             "assets": by_class_loc["asset"]}
    (OUT / "PACK_CANDIDATES.json").write_text(json.dumps(
        {"schema": "hawking.pack_candidates.v1", "candidates": packs,
         "total_packable_loc": packable, "note": "sealed packs, offline-hydratable; not deleted"},
        indent=2, sort_keys=True))

    (OUT / "IRREDUCIBLE_KERNEL_ESTIMATE.json").write_text(json.dumps(
        {"schema": "hawking.irreducible_kernel_estimate.v1",
         "kernel_loc_estimate": kernel, "active_product_loc_estimate": active_product,
         "target_kernel": "50k-75k", "gap_to_75k": max(0, kernel - 75000),
         "method": "heuristic; refine with AST/call-graph before committing to a floor"},
        indent=2, sort_keys=True))

    md = ["# Codebase Census (CLEAN SLATE Section 13)", "",
          f"Generated {census['generated_at']} - git-tracked files only (heuristic classification).", "",
          f"- tracked files: **{len(files)}**", f"- total tracked LOC: **{total:,}**",
          f"- kernel estimate: **{kernel:,}** (target 50k-75k)",
          f"- active product estimate: **{active_product:,}** (target 65k-90k)",
          f"- packable (labs/docs/fixtures/compat/assets): **{packable:,}**", "",
          "## LOC by class", "", "| class | LOC | files |", "|---|---|---|"]
    for c, n in by_class_loc.most_common():
        md.append(f"| {c} | {n:,} | {by_class_files[c]} |")
    md += ["", "## LOC by top directory", "", "| dir | LOC |", "|---|---|"]
    for d, n in by_dir_loc.most_common(16):
        md.append(f"| {d} | {n:,} |")
    (OUT / "CODEBASE_CENSUS.md").write_text("\n".join(md) + "\n")
    return census


def main(argv=None) -> int:
    c = build()
    print(f"tracked files: {c['tracked_files']}  total tracked LOC: {c['total_tracked_loc']:,}")
    print(f"kernel estimate: {c['kernel_loc_estimate']:,}  (target 50k-75k)")
    print(f"active product estimate: {c['active_product_loc_estimate']:,}  (target 65k-90k)")
    print(f"packable: {c['packable_loc']:,}")
    print("LOC by class:")
    for cls, n in c["loc_by_class"].items():
        print(f"  {cls:20s} {n:>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
