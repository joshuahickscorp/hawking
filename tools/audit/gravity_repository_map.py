"""Build the before-state inventory for the Gravity unification.

This is an inventory tool, not a cleanup tool. It never deletes, moves, or
rewrites repository content. The output is a generated receipt so a later
migration can compare source surface and importer state against this baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {
    ".git", ".aider.tags.cache.v4", ".pytest_cache", ".ruff_cache", ".serena",
    "__pycache__", "node_modules", ".venv", "dist", "target", ".worktrees",
    ".preserved", ".hide", ".hcli", ".hcli-legacy", ".claude", ".cargo",
    "workspace", "artifacts", "reports", "logs", "scratchpad", "visionmcp",
}
SOURCE_EXTENSIONS = {".py", ".rs", ".md", ".toml", ".yaml", ".yml", ".json", ".sh"}
ADJACENT_TOKENS = (
    "gravity", "doctor", "ablit", "operator", "tabula", "pq", "quant",
    "ebpw", "noetic", "nx", "physical", "machinegenome", "pareto",
    "singularity", "odyssey", "anatom", "routing", "locality", "refusal",
    "representation", "codebook", "latent", "sparse", "trellis", "binary",
    "benchmark", "bench", "tps", "metal", "mlx", "accelerator", "kernel",
    "verify", "receipt", "provenance", "law", "scar", "transfer",
)


GROUPS = (
    {
        "id": "discover",
        "owner": "Gravity [DISCOVER]",
        "purpose": "Organism identity, architecture, anatomy, provenance, state, modality, and routing facts.",
        "destination": "gravity/core",
        "patterns": ("anatom", "architecture", "config", "token", "model", "organ", "odyssey_census"),
    },
    {
        "id": "diagnose",
        "owner": "Gravity [DIAGNOSE]",
        "purpose": "Behavior, refusal, routing, loader, runtime, representation, and capability pathology.",
        "destination": "gravity/core",
        "patterns": ("refusal", "locality", "pathology", "runtime", "diagnos", "sensitivity", "loader"),
    },
    {
        "id": "shape",
        "owner": "Gravity [SHAPE]; Nova when learned cognition changes",
        "purpose": "Behavior/operator transformation, abliteration, refusal surgery, and candidate shaping.",
        "destination": "gravity/nova",
        "patterns": ("ablit", "operator", "tabula", "steer", "surgery", "deltanet", "teacher"),
    },
    {
        "id": "reduce",
        "owner": "Gravity [REDUCE]",
        "purpose": "Information reduction, representations, quantization, codecs, and complete accounting.",
        "destination": "gravity/representations",
        "patterns": ("pq", "quant", "ebpw", "lowrank", "low_rank", "codebook", "trellis", "binary", "sparse", "compress", "byte_census"),
    },
    {
        "id": "form",
        "owner": "Gravity [REPRESENT]; Noetic Representation substrate",
        "purpose": "The canonical representation graph and Noetic representation language/IR.",
        "destination": "gravity/representations",
        "patterns": ("noetic", "representation", "latent", "ir", "graph", "generated"),
    },
    {
        "id": "run",
        "owner": "Gravity [EXECUTE]",
        "purpose": "Materialization, execution, caching, state/KV, NX artifact production, and runtime plans.",
        "destination": "gravity/execution",
        "patterns": ("nx", "nvm", "execution", "executor", "material", "cache", "state", "kv"),
    },
    {
        "id": "physical",
        "owner": "Gravity [PHYSICAL]",
        "purpose": "Machine fit: CPU/GPU/Metal/MLX, memory, bandwidth, kernels, placement, and throughput.",
        "destination": "gravity/execution",
        "patterns": ("physical", "machinegenome", "metal", "mlx", "gpu", "cpu", "tps", "kernel", "accelerator", "resident", "bandwidth", "bench"),
    },
    {
        "id": "verify",
        "owner": "Gravity [VERIFY]",
        "purpose": "Capability, epistemic, authorization, HCLI, destructive-control, accounting, and parity verification.",
        "destination": "gravity/verification",
        "patterns": ("verify", "verification", "doctor", "audit", "acceptance", "gate", "proof"),
    },
    {
        "id": "attack",
        "owner": "Gravity [ATTACK]",
        "purpose": "Adversarial probes, falsification, destructive controls, and transformation-specific attacks.",
        "destination": "gravity/verification",
        "patterns": ("attack", "advers", "falsif", "negative_science", "adversarial"),
    },
    {
        "id": "learn",
        "owner": "Gravity [LEARN]",
        "purpose": "Laws, Scars, priors, transfer, frontier history, and search policy.",
        "destination": "gravity/learning",
        "patterns": ("law", "scar", "transfer", "prior", "pareto", "singularity", "roadmap", "frontier", "learn"),
    },
)

CANONICAL_CAPABILITIES = {
    "model_anatomy": {
        "canonical_owner": "Gravity [DISCOVER] anatomy (target; not yet migrated)",
        "current_owner_candidate": "hcli/architecture.py",
        "duplicate_candidates": [
            "hcli/architecture.py", "tools/future/anatomy_from_headers.py",
            "tools/future/organ_byte_anatomy.py", "tools/odyssey_patient_runner.py",
        ],
    },
    "complete_ebpw_accounting": {
        "canonical_owner": "Gravity accounting/verification (target; preserve current API)",
        "current_owner_candidate": "tools/future/complete_ebpw.py",
        "duplicate_candidates": [
            "tools/future/complete_ebpw.py", "tools/future/ebpw_categories.py",
            "tools/gravity_allocator.py", "hcli/gravity_gauntlet.py",
            "hcli/flash_next.py", "hcli/odyssey_census.py",
        ],
    },
    "representation": {
        "canonical_owner": "Gravity representation contract (target; Noetic is the substrate)",
        "current_owner_candidate": "hcli/gravity/ plus tools/condense/",
        "duplicate_candidates": [
            "hcli/gravity/", "tools/condense/", "tools/odyssey/noetic_compiler.py",
            "tools/future/flash_meta_representation.py", "tools/future/complete_nr.py",
        ],
    },
    "verification": {
        "canonical_owner": "Gravity verification (target; HCLI authorization remains outside model behavior)",
        "current_owner_candidate": "tools/verify/",
        "duplicate_candidates": ["tools/verify/", "tools/doctor/", "hcli/doctor/", "tools/theia/"],
    },
    "experiment_receipts": {
        "canonical_owner": "Gravity experiment receipt protocol (target)",
        "current_owner_candidate": "tools/theia/intake.py plus tools/future/repro_science.py",
        "duplicate_candidates": ["tools/theia/", "tools/future/repro_science.py", "research/lab/provenance.py"],
    },
    "model_loader": {
        "canonical_owner": "Gravity model-loader abstraction (target)",
        "current_owner_candidate": "tools/future/dsv3_native_loader.py",
        "duplicate_candidates": ["tools/future/dsv3_native_loader.py", "tools/future/kimi_vl_loader.py", "tools/condense/"],
    },
    "routing_statistics": {
        "canonical_owner": "Gravity [DIAGNOSE] routing (target; KIMI is the active compatibility implementation)",
        "current_owner_candidate": "tools/future/moe_refusal_locality.py",
        "duplicate_candidates": ["tools/future/moe_refusal_locality.py", "tools/odyssey_patient_runner.py", "tools/flash_router_sensitivity_map.py"],
    },
    "physical_benchmarking": {
        "canonical_owner": "Gravity physical measurement interface (target)",
        "current_owner_candidate": "tools/bench/ plus tools/accelerator/",
        "duplicate_candidates": ["tools/bench/", "tools/accelerator/", "tools/future/*tps*", "tools/future/*profile*"],
    },
    "operator_provenance": {
        "canonical_owner": "Gravity operator provenance (target; KIMI candidate schema is the bridge)",
        "current_owner_candidate": "tools/future/kimi_operator_candidates.py",
        "duplicate_candidates": ["tools/future/kimi_operator_candidates.py", "tools/future/abliteration.py (historical)", "research/lab/operators/"],
    },
}


def _all_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
            files.append(path)
    return sorted(files)


def _adjacent(path: Path) -> bool:
    text = path.as_posix().lower()
    # The receipt directory itself is not a capability name. Inventory only
    # receipt artifacts whose filenames identify this migration, otherwise a
    # large historical ledger overwhelms the source-owner map.
    if "receipts" in path.parts:
        return any(token in path.name.lower() for token in ADJACENT_TOKENS)
    if any(token in text for token in ADJACENT_TOKENS):
        return True
    # Do not classify every file in the source tree as Gravity-adjacent. The
    # full source/test counts are recorded separately; entries are the paths
    # whose names or locations actually participate in this migration.
    return False


def _git_last_used(root: Path) -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%H%x09%cs", "--name-only", "--all"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    result: dict[str, str] = {}
    commit = date = None
    for line in proc.stdout.splitlines():
        if "\t" in line and len(line.split("\t", 1)[0]) >= 7:
            commit, date = line.split("\t", 1)
        elif line and commit and not line.startswith(" "):
            result.setdefault(line, f"{date} ({commit[:12]})")
    return result


def _tracked_paths(root: Path) -> set[str]:
    try:
        proc = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                              check=True, capture_output=True)
        return {item.decode() for item in proc.stdout.split(b"\0") if item}
    except (OSError, subprocess.CalledProcessError):
        return set()


def _group_for(path: Path) -> dict[str, Any]:
    text = path.as_posix().lower()
    for group in GROUPS:
        if any(pattern in text for pattern in group["patterns"]):
            return group
    return {
        "id": "unclassified",
        "owner": "UNREVIEWED",
        "purpose": "Relevant source/configuration requiring owner review before migration.",
        "destination": "UNDECIDED",
        "patterns": (),
    }


def _lifecycle(path: Path) -> str:
    text = path.as_posix().lower()
    if any(part in {"receipts", "reports", "logs", "artifacts", "workspace", ".hcli"} for part in path.parts):
        return "GENERATED"
    if any(token in text for token in ("/archive/", "/legacy/", "/retired/", ".bak", "_old", "deprecated")):
        return "OBSOLETE_CANDIDATE"
    if any(token in text for token in ("/future/", "/experiment", "/probe", "/bench", "/campaign", "/sweep", "smoke")):
        return "EXPERIMENTAL"
    return "DURABLE"


def _module_name(path: Path) -> str:
    value = path.with_suffix("").as_posix().replace("/", ".")
    return value[2:] if value.startswith("./") else value


def build(root: Path) -> dict[str, Any]:
    files = _all_files(root)
    source_files = [path for path in files if path.suffix == ".py"]
    relevant = [path for path in files if _adjacent(path)]
    contents = {}
    for path in source_files:
        try:
            contents[path] = path.read_text(errors="replace")
        except OSError:
            contents[path] = ""

    importers: dict[str, set[str]] = defaultdict(set)
    import_pattern = re.compile(r"(?:from|import)\s+([A-Za-z_][\w.]*)")
    for importer, content in contents.items():
        for module in import_pattern.findall(content):
            importers[module].add(importer.relative_to(root).as_posix())

    tests = {
        path: contents.get(path, "")
        for path in source_files
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    }
    last_used = _git_last_used(root)
    tracked_paths = _tracked_paths(root)
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in relevant:
        grouped[_group_for(path)["id"]].append(path.relative_to(root).as_posix())

    entries = []
    for path in relevant:
        relative = path.relative_to(root).as_posix()
        group = _group_for(path)
        stem = path.stem
        module = _module_name(path)
        found_importers = set(importers.get(module, set())) | set(importers.get(stem, set()))
        found_tests = {
            test_path.relative_to(root).as_posix()
            for test_path, body in tests.items()
            if relative in body or module in body
            or (len(stem) >= 5 and re.search(
                rf"(?<![A-Za-z0-9]){re.escape(stem)}(?![A-Za-z0-9])", body))
        }
        if relative in tracked_paths:
            used = last_used.get(relative, "NO_GIT_HISTORY")
        else:
            used = "UNCOMMITTED_OR_UNTRACKED"
        siblings = [item for item in grouped[group["id"]] if item != relative]
        entries.append({
            "path": relative,
            "owner": group["owner"],
            "purpose": group["purpose"],
            "importers": sorted(found_importers),
            "tests": sorted(found_tests),
            "last_used": used,
            "duplicates": {
                "group": group["id"],
                "count": len(siblings),
                "examples": siblings[:12],
            },
            "lifecycle": _lifecycle(path),
            "canonical_destination": group["destination"],
        })

    top_level = sorted({path.relative_to(root).parts[0] for path in files if path.parts})
    ext_counts: dict[str, int] = defaultdict(int)
    for path in files:
        ext_counts[path.suffix or "<none>"] += 1
    return {
        "schema": "hawking.gravity.repository_map.v1",
        "status": "BEFORE_STATE_INVENTORY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(root),
        "head": subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                check=False, capture_output=True, text=True).stdout.strip() or "UNKNOWN",
        "scope": {
            "source_extensions": sorted(SOURCE_EXTENSIONS),
            "excluded_directories": sorted(SKIP_DIRS),
            "adjacent_matching": "all relevant source/config/receipt paths plus hcli/tools/research/docs/tests source surface",
        },
        "before_counts": {
            "inventoried_files": len(files),
            "gravity_adjacent_files": len(relevant),
            "python_modules": len(source_files),
            "test_files": len(tests),
            "top_level_paths": len(top_level),
            "extensions": dict(sorted(ext_counts.items())),
        },
        "top_level_paths": top_level,
        "canonical_capability_owners": CANONICAL_CAPABILITIES,
        "entries": entries,
        "migration_rule": "Inventory, migrate unique capability, preserve receipt/regression, verify, update importers, then delete only after importer count is zero.",
        "claim_boundary": "This receipt is an inventory and heuristic duplicate map. It does not authorize deletion and does not prove semantic equivalence of listed paths.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "receipts" / "future" / "GRAVITY_REPOSITORY_MAP.json")
    args = parser.parse_args()
    document = build(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "before_counts": document["before_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
