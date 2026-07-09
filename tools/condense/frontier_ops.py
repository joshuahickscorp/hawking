#!/usr/bin/env python3.12
"""frontier_ops.py - frontier ledger, status, refresh, and source-release guards.

This is the operator layer above `procure.py`. It is intentionally conservative:

  frontier_ops.py status                         # compact human status
  frontier_ops.py ledger [--out PATH]            # machine-readable run ledger, no network by default
  frontier_ops.py ledger --refresh-hf            # add HF model metadata
  frontier_ops.py ledger --dry-run-sizes         # add `hf download --dry-run` size receipts
  frontier_ops.py worktree-plan                  # group dirty tree by subsystem for stack splitting
  frontier_ops.py refresh                        # query HF for candidates to review before launch
  frontier_ops.py review-plan                    # write candidate-review command queue
  frontier_ops.py review-decisions draft         # signed batch candidate-review workbook
  frontier_ops.py storage-plan                   # storage-aware download waves with checkpoints
  frontier_ops.py lifecycle                      # per-model DAG state + next safe command
  frontier_ops.py run-next                       # dry-run/apply one lifecycle-safe command
  frontier_ops.py artifact-inventory LABEL       # hash durable .tq output before source release
  frontier_ops.py record-event LABEL --stage bake --status pass --duration-s N
  frontier_ops.py review-candidate HF_ID --decision watch --by NAME
  frontier_ops.py license-plan                   # accepted-terms commands before procurement
  frontier_ops.py license-decisions draft        # signed batch license/gating workbook
  frontier_ops.py parity-receipt draft LABEL     # signed architecture parity receipt envelopes
  frontier_ops.py coverage-receipt draft LABEL   # signed baseline/eval receipt envelopes
  frontier_ops.py source-provenance draft LABEL  # signed source checkpoint provenance envelopes
  frontier_ops.py receipt-record draft LABEL     # signed native serve/RAM-cliff receipt envelopes
  frontier_ops.py serve-capture LABEL --artifact ARTIFACT.tq --bench-json REPORT.json
  frontier_ops.py doctor-recovery-receipt draft LABEL # signed Doctor recovery receipt envelopes
  frontier_ops.py experiment-receipt draft LABEL # signed expensive-mode matrix envelopes
  frontier_ops.py proof-pack LABEL               # draft all signed envelopes + blocked local bundle
  frontier_ops.py release-source LABEL --dry-run # prove whether source deletion would be safe
  frontier_ops.py release-source LABEL --yes     # delete source only after output + record evidence
  frontier_ops.py launch-gate --phase procure    # go/no-go for large downloads
  frontier_ops.py launch-gate --phase claim      # go/no-go for quality/tok/s claims
  frontier_ops.py claim-bundle build LABEL       # sign public-claim evidence by hash
  frontier_ops.py record-license LABEL --status accepted --by NAME --license ID --terms-url URL ...
  frontier_ops.py selftest                       # synthetic guard tests, no network

The goal is to make the Studio run auditable. Giant sources are transient; receipts and `.tq` outputs
are the durable experiment products. This tool records the evidence that makes source release safe and
summarizes `procure.py` download telemetry when present.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

from studio_manifest import (  # noqa: E402
    DEFAULT_HARDWARE,
    FRONTIER_MODELS,
    FrontierModel,
    eta_hours,
    fmt_hours,
    frontier_by_label,
    storage_wave_plan as manifest_storage_wave_plan,
    total_artifact_gb,
    total_download_gb,
)
import frontier_parity  # noqa: E402
import frontier_parity_runner  # noqa: E402
import frontier_coverage  # noqa: E402
import frontier_coverage_runner  # noqa: E402
import frontier_provenance  # noqa: E402
import frontier_serve_capture  # noqa: E402
import frontier_receipt_runner  # noqa: E402
import frontier_receipts  # noqa: E402
import frontier_experiments  # noqa: E402
import frontier_experiment_runner  # noqa: E402
import frontier_licenses  # noqa: E402
import frontier_claims  # noqa: E402
import frontier_doctor_recovery  # noqa: E402

COND_DIR = pathlib.Path("reports/condense")
LEDGER_PATH = COND_DIR / "frontier_ledger.json"
LICENSE_PATH = COND_DIR / "frontier_license_acceptance.json"
LICENSE_DECISIONS_PATH = COND_DIR / "frontier_license_decisions.draft.json"
RELEASE_LOG = COND_DIR / "frontier_releases.jsonl"
DOWNLOAD_LOG = COND_DIR / "frontier_downloads.jsonl"
EVENT_LOG = COND_DIR / "frontier_events.jsonl"
REFRESH_PATH = COND_DIR / "frontier_refresh.json"
REFRESH_REVIEW_PATH = COND_DIR / "frontier_refresh_reviews.json"
REFRESH_REVIEW_DECISIONS_PATH = COND_DIR / "frontier_refresh_review_decisions.draft.json"
PREFLIGHT_SUMMARY_PATH = COND_DIR / "studio_preflight_summary.json"
STUDIO_ENVIRONMENT_PATH = COND_DIR / "studio_environment.json"
WORKTREE_PLAN_PATH = COND_DIR / "worktree_split_plan.local.json"
PROOF_PACK_PATH = COND_DIR / "frontier_proof_pack.local.json"
WAVE0_PACKET_PATH = COND_DIR / "studio_wave0_launch_packet.json"
AUDIT_GRADE_PATH = COND_DIR / "studio_audit_grade.local.json"
RUNTIME_CONTRACT_PATH = COND_DIR / "studio_runtime_contract.local.json"
COMPLETION_AUDIT_PATH = COND_DIR / "studio_completion_audit.local.json"
DEFAULT_EXTERNAL_AUDIT_PATH = pathlib.Path("/Users/scammermike/Downloads/project_audits/hawking_deep_audit_2026_07_08.md")
DEFAULT_STUDIO_AUDIT_PATH = pathlib.Path("docs/plans/STUDIO_DEEP_AUDIT_2026_07_08.md")
SIGN_ALG = "sha256-json-v1"
SIZE_RE = re.compile(r"\s([0-9]+(?:\.[0-9]+)?)([KMGT])$")
SIZE_SCALE = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
CACHE_RESERVE_GB = 128.0
MANIFEST_CONSUMERS = (
    "tools/condense/ladder.py",
    "tools/condense/subbit.py",
    "tools/condense/procure.py",
    "tools/condense/studio_run.py",
    "tools/condense/ramcliff_bench.py",
    "tools/condense/scorecard.py",
    "tools/condense/frontier_parity.py",
    "tools/condense/frontier_parity_runner.py",
    "tools/condense/frontier_coverage_runner.py",
    "tools/condense/frontier_provenance.py",
    "tools/condense/frontier_serve_capture.py",
    "tools/condense/frontier_receipt_runner.py",
    "tools/condense/frontier_experiment_runner.py",
    "tools/condense/frontier_claims.py",
    "tools/condense/frontier_doctor_recovery.py",
)
RETIRED_FRONTIER_MARKERS = ("GLM-4.5", '"744B"', "'744B'")
WORKTREE_SUBSYSTEMS = (
    {
        "name": "ci-config",
        "prefixes": (".github/", ".cargo/", "Cargo.toml", "Cargo.lock", "rust-toolchain"),
        "branch": "codex/ci-config",
        "risk": "medium",
        "action": "Keep CI/toolchain changes isolated from runtime and proof work.",
    },
    {
        "name": "hide-ui-tauri-assets",
        "prefixes": ("app/", "logo/", "package.json", "pnpm-lock.yaml"),
        "branch": "codex/hide-ui-tauri-assets",
        "risk": "medium",
        "action": "Review as the HIDE desktop/Tauri/UI asset stack.",
    },
    {
        "name": "hide-backend-kernel",
        "prefixes": ("crates/hide-",),
        "branch": "codex/hide-backend-kernel",
        "risk": "medium",
        "action": "Review as HIDE host/replay/serve/kernel behavior.",
    },
    {
        "name": "hawking-core-runtime",
        "prefixes": ("crates/hawking/", "crates/hawking-core/", "crates/hawking-serve/"),
        "branch": "codex/hawking-core-runtime",
        "risk": "high",
        "action": "Review as shippable Hawking CLI/core/serve runtime changes.",
    },
    {
        "name": "condense-frontier-proof",
        "prefixes": ("tools/condense/", "reports/condense/"),
        "branch": "codex/condense-frontier-proof",
        "risk": "high",
        "action": "Review as Studio proof gates, receipts, storage, and frontier operator work.",
    },
    {
        "name": "studio-docs-audits",
        "prefixes": ("docs/plans/", "BASELINES.md"),
        "branch": "codex/studio-docs-audits",
        "risk": "low",
        "action": "Review as scorecard/runbook/audit evidence, paired with command output.",
    },
    {
        "name": "local-deps-generated",
        "prefixes": ("node_modules/", "target/", "scratch/", ".pnpm-store/"),
        "branch": "",
        "risk": "high",
        "action": "Do not commit; clean or ignore intentionally after confirming nothing unique lives here.",
    },
)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 127, "", f"{type(e).__name__}: {e}"


def _git_commit(root: pathlib.Path = ROOT) -> str:
    rc, out, _ = _run(["git", "rev-parse", "--short", "HEAD"], timeout=10)
    return out.strip() if rc == 0 and out.strip() else "unknown"


def _gb_from_bytes(n: int) -> float:
    return n / 1e9


def _path_size_gb(path: pathlib.Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return _gb_from_bytes(path.stat().st_size)
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return _gb_from_bytes(total)


def _hf_cache_paths(root: pathlib.Path = ROOT) -> dict[str, pathlib.Path]:
    return {
        "hf_home": pathlib.Path(os.environ.get("HF_HOME", root / "scratch" / "hf-home")),
        "hf_hub_cache": pathlib.Path(os.environ.get("HF_HUB_CACHE", root / "scratch" / "hf-cache" / "hub")),
        "hf_xet_cache": pathlib.Path(os.environ.get("HF_XET_CACHE", root / "scratch" / "hf-cache" / "xet")),
    }


def _under_root(path: pathlib.Path, root: pathlib.Path = ROOT) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _cache_snapshot(root: pathlib.Path = ROOT) -> dict:
    paths = _hf_cache_paths(root)
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "sizes_gb": {k: round(_path_size_gb(v), 3) for k, v in paths.items()},
        "project_local": {k: _under_root(v, root) for k, v in paths.items()},
        "reserve_gb": CACHE_RESERVE_GB,
        "prune_dry_run_cmd": [
            "python3.12", "tools/condense/procure.py", "--cache-prune",
        ],
    }


def _git_branch(root: pathlib.Path = ROOT) -> str:
    rc, out, _ = _run(["git", "-C", str(root), "branch", "--show-current"], timeout=10)
    if rc == 0 and out.strip():
        return out.strip()
    rc, out, _ = _run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], timeout=10)
    return f"detached:{out.strip()}" if rc == 0 and out.strip() else "unknown"


def _git_status_entries(root: pathlib.Path = ROOT) -> tuple[list[dict], str]:
    rc, out, err = _run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=normal"],
        timeout=20,
    )
    if rc != 0:
        return [], (err or out or "git status failed").strip()
    entries = []
    for line in out.splitlines():
        if not line:
            continue
        xy = line[:2]
        path = line[3:] if len(line) > 3 else ""
        original_path = ""
        if " -> " in path:
            original_path, path = path.split(" -> ", 1)
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        subsystem = _worktree_subsystem(path)
        entries.append({
            "status": xy,
            "path": path,
            "original_path": original_path,
            "subsystem": subsystem["name"],
            "staged": xy[0] not in (" ", "?"),
            "unstaged": xy[1] not in (" ", "?"),
            "untracked": xy == "??",
            "deleted": "D" in xy,
            "renamed": "R" in xy or bool(original_path),
        })
    return entries, ""


def _worktree_subsystem(path: str) -> dict:
    for spec in WORKTREE_SUBSYSTEMS:
        if any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in spec["prefixes"]):
            return spec
    return {
        "name": "uncategorized",
        "prefixes": (),
        "branch": "codex/misc-cleanup",
        "risk": "medium",
        "action": "Inspect manually before pairing with a stack.",
    }


def build_worktree_plan(root: pathlib.Path = ROOT) -> dict:
    entries, error = _git_status_entries(root)
    specs = {spec["name"]: spec for spec in WORKTREE_SUBSYSTEMS}
    specs["uncategorized"] = _worktree_subsystem("__uncategorized__")
    groups = {
        name: {
            "name": name,
            "branch": spec.get("branch", ""),
            "risk": spec.get("risk", "medium"),
            "action": spec.get("action", ""),
            "entries": [],
            "counts": {
                "total": 0,
                "staged": 0,
                "unstaged": 0,
                "untracked": 0,
                "deleted": 0,
                "renamed": 0,
            },
        }
        for name, spec in specs.items()
    }
    for entry in entries:
        group = groups.setdefault(entry["subsystem"], {
            "name": entry["subsystem"],
            "branch": "codex/misc-cleanup",
            "risk": "medium",
            "action": "Inspect manually before pairing with a stack.",
            "entries": [],
            "counts": {"total": 0, "staged": 0, "unstaged": 0, "untracked": 0, "deleted": 0, "renamed": 0},
        })
        group["entries"].append(entry)
        group["counts"]["total"] += 1
        for key in ("staged", "unstaged", "untracked", "deleted", "renamed"):
            if entry[key]:
                group["counts"][key] += 1
    active_groups = [
        group for group in groups.values()
        if group["counts"]["total"] > 0
    ]
    active_groups.sort(key=lambda g: (
        g["name"] == "local-deps-generated",
        g["risk"] != "high",
        -g["counts"]["total"],
        g["name"],
    ))
    local_noise = groups.get("local-deps-generated", {}).get("counts", {}).get("total", 0)
    dirty_subsystems = len(active_groups)
    risk = "clean"
    if error:
        risk = "unknown"
    elif dirty_subsystems >= 5 or local_noise:
        risk = "high"
    elif dirty_subsystems >= 3:
        risk = "medium"
    elif dirty_subsystems:
        risk = "low"
    totals = {
        "entries": len(entries),
        "staged": sum(1 for e in entries if e["staged"]),
        "unstaged": sum(1 for e in entries if e["unstaged"]),
        "untracked": sum(1 for e in entries if e["untracked"]),
        "deleted": sum(1 for e in entries if e["deleted"]),
        "subsystems": dirty_subsystems,
    }
    plan = {
        "schema": "hawking.worktree_split_plan.v1",
        "generated_at": _now(),
        "root": str(root),
        "git_commit": _git_commit(root),
        "branch": _git_branch(root),
        "ok": not error,
        "error": error,
        "risk": risk,
        "totals": totals,
        "groups": active_groups,
        "recommended_stack_order": [
            {
                "subsystem": group["name"],
                "branch": group["branch"],
                "count": group["counts"]["total"],
                "risk": group["risk"],
                "action": group["action"],
            }
            for group in active_groups
            if group["name"] != "local-deps-generated"
        ],
        "local_cleanup": groups.get("local-deps-generated"),
        "note": (
            "Read-only dirty-tree classification. It is evidence for review splitting, not approval to "
            "revert or stage unrelated user changes."
        ),
    }
    return _sign_doc(plan)


def _worktree_plan_status(doc: dict) -> dict:
    problems = []
    if doc.get("schema") != "hawking.worktree_split_plan.v1":
        problems.append("schema mismatch")
    signature_ok = _signature_ok(doc)
    if not signature_ok:
        problems.append("signature digest mismatch")
    if not isinstance(doc.get("groups"), list):
        problems.append("groups must be a list")
    if not isinstance(doc.get("totals"), dict):
        problems.append("totals must be an object")
    return {
        "schema_ok": doc.get("schema") == "hawking.worktree_split_plan.v1",
        "signature_ok": signature_ok,
        "plan_ok": bool(doc.get("ok")),
        "risk": doc.get("risk"),
        "entries": (doc.get("totals") or {}).get("entries"),
        "subsystems": (doc.get("totals") or {}).get("subsystems"),
        "ok": not problems,
        "problems": problems,
    }


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _markdown_row_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = []
    cur = []
    in_code = False
    for ch in stripped:
        if ch == "`":
            in_code = not in_code
            cur.append(ch)
        elif ch == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def _parse_facet_grades(path: pathlib.Path) -> list[dict]:
    text = _read_text(path)
    rows = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Facet | Grade |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = _markdown_row_cells(line)
        if len(cells) < 4 or cells[0].strip("- ") == "":
            continue
        if cells[0].lower() == "facet" or set(cells[1].replace(":", "").strip()) <= {"-"}:
            continue
        try:
            grade = float(cells[1])
        except ValueError:
            continue
        rows.append({
            "facet": cells[0],
            "grade": grade,
            "why_not_10": cells[2],
            "condition_10_plus": cells[3],
        })
    return rows


def _parse_first_grade(text: str, label: str) -> float | None:
    pattern = re.compile(rf"{re.escape(label)}\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", re.I)
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def _parse_operator_plan_grade(text: str) -> float | None:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10\s+as an operator-proof Studio run plan", text)
    return float(m.group(1)) if m else None


def _artifact_doc(path: pathlib.Path, root: pathlib.Path = ROOT) -> dict:
    full = path if path.is_absolute() else root / path
    data = _read_json(full, {})
    return data if isinstance(data, dict) else {}


def build_audit_grade(root: pathlib.Path, args) -> dict:
    target = float(args.target_grade)
    external_audit = pathlib.Path(args.external_audit)
    studio_audit = pathlib.Path(args.studio_audit)
    external_full = external_audit if external_audit.is_absolute() else root / external_audit
    studio_full = studio_audit if studio_audit.is_absolute() else root / studio_audit
    external_text = _read_text(external_full)
    studio_text = _read_text(studio_full)
    facets = _parse_facet_grades(studio_full)
    grades = [row["grade"] for row in facets]
    below = [row for row in facets if row["grade"] < target]
    lowest = sorted(facets, key=lambda r: (r["grade"], r["facet"]))[:8]

    launch_packet_path = pathlib.Path(args.launch_packet)
    proof_pack_path = pathlib.Path(args.proof_pack)
    worktree_plan_path = pathlib.Path(args.worktree_plan)
    runtime_contract_path = pathlib.Path(getattr(args, "runtime_contract", RUNTIME_CONTRACT_PATH))
    claim_gate_path = pathlib.Path(args.claim_gate)
    procurement_gate_path = pathlib.Path(args.procurement_gate)

    launch_packet = _artifact_doc(launch_packet_path, root)
    proof_pack = _artifact_doc(proof_pack_path, root)
    worktree_plan = _artifact_doc(worktree_plan_path, root)
    runtime_contract_artifact = _json_artifact(runtime_contract_path, root)
    claim_gate = _artifact_doc(claim_gate_path, root)
    procurement_gate = _artifact_doc(procurement_gate_path, root)

    claim_bundles = proof_pack.get("claim_bundles") if isinstance(proof_pack.get("claim_bundles"), list) else []
    frontier_claims_walled = bool(
        proof_pack.get("ok") is True
        and claim_bundles
        and all(row.get("claim_admissible") is False for row in claim_bundles)
        and claim_gate.get("ok") is False
    )
    target_reached = bool(
        _parse_first_grade(external_text, "Current overall grade") is not None
        and _parse_first_grade(external_text, "Current overall grade") >= target
        and claim_gate.get("ok") is True
        and launch_packet.get("procurement_permitted") is True
        and runtime_contract_artifact.get("ok") is True
        and runtime_contract_artifact.get("signature_ok") is True
    )
    blockers = []
    if not target_reached:
        blockers.append("harsher audit target is not proven by current external-audit grade plus green gates")
    if not launch_packet.get("procurement_permitted"):
        blockers.append("wave-0 launch packet does not permit procurement")
    if claim_gate.get("ok") is not True:
        blockers.append("claim launch gate is red")
    if worktree_plan.get("risk") == "high":
        blockers.append("signed worktree split reports high dirty-tree risk")
    if runtime_contract_artifact.get("ok") is not True or runtime_contract_artifact.get("signature_ok") is not True:
        blockers.append("native runtime/TQ proof-mode contract is missing, unsigned, or invalid")
    if below:
        blockers.append(f"{len(below)} audit facets remain below {target}/10")

    doc = {
        "schema": "hawking.studio_audit_grade.v1",
        "generated_at": _now(),
        "root": str(root),
        "git_commit": _git_commit(root),
        "ok": True,
        "target_grade": target,
        "target_reached": target_reached,
        "frontier_claims_walled": frontier_claims_walled,
        "external_audit": {
            "path": str(external_audit),
            "exists": external_full.exists(),
            "sha256": _sha256_file(external_full) if external_full.exists() else None,
            "current_overall_grade": _parse_first_grade(external_text, "Current overall grade"),
            "local_laptop_potential": _parse_first_grade(external_text, "Local laptop potential after disk cleanup"),
            "studio_potential": _parse_first_grade(external_text, "Studio potential"),
        },
        "studio_audit": {
            "path": str(studio_audit),
            "exists": studio_full.exists(),
            "sha256": _sha256_file(studio_full) if studio_full.exists() else None,
            "operator_plan_grade": _parse_operator_plan_grade(studio_text),
        },
        "facet_count": len(facets),
        "facet_average": round(sum(grades) / len(grades), 3) if grades else None,
        "facet_min": min(grades) if grades else None,
        "below_target_count": len(below),
        "lowest_facets": lowest,
        "below_target_facets": below,
        "artifacts": {
            "launch_packet": _json_artifact(launch_packet_path, root),
            "proof_pack": _json_artifact(proof_pack_path, root),
            "worktree_plan": _json_artifact(worktree_plan_path, root),
            "runtime_contract": runtime_contract_artifact,
            "claim_gate": _json_artifact(claim_gate_path, root),
            "procurement_gate": _json_artifact(procurement_gate_path, root),
            "scorecard": _json_artifact(pathlib.Path(args.scorecard), root),
        },
        "current_state": {
            "procurement_permitted": bool(launch_packet.get("procurement_permitted")),
            "launch_packet_ok": bool(launch_packet.get("ok")),
            "claim_gate_ok": bool(claim_gate.get("ok")),
            "procurement_gate_ok": bool(procurement_gate.get("ok")),
            "worktree_risk": worktree_plan.get("risk"),
            "runtime_contract_ok": bool(runtime_contract_artifact.get("ok")),
            "proof_pack_blocked_claims": proof_pack.get("blocked_claim_count"),
            "proof_pack_models": proof_pack.get("model_count"),
        },
        "blockers": blockers,
        "note": (
            "This receipt grades audit readiness, not model quality. It can prove that frontier claims are "
            "walled, but it cannot prove the 8.4/10 target until the current external audit and launch/claim "
            "gates are green on real evidence."
        ),
    }
    return _sign_doc(doc)


def _audit_grade_status(doc: dict) -> dict:
    problems = []
    if doc.get("schema") != "hawking.studio_audit_grade.v1":
        problems.append("schema mismatch")
    signature_ok = _signature_ok(doc)
    if not signature_ok:
        problems.append("signature digest mismatch")
    if not isinstance(doc.get("lowest_facets"), list):
        problems.append("lowest_facets must be a list")
    if doc.get("target_grade") is None:
        problems.append("target_grade missing")
    return {
        "schema_ok": doc.get("schema") == "hawking.studio_audit_grade.v1",
        "signature_ok": signature_ok,
        "target_reached": bool(doc.get("target_reached")),
        "frontier_claims_walled": bool(doc.get("frontier_claims_walled")),
        "facet_count": doc.get("facet_count"),
        "below_target_count": doc.get("below_target_count"),
        "ok": not problems,
        "problems": problems,
    }


def _completion_models(labels: list[str]) -> list[FrontierModel]:
    if not labels:
        return list(FRONTIER_MODELS)
    out = []
    for label in labels:
        model = frontier_by_label(label)
        if not model:
            raise ValueError(f"unknown frontier label: {label}")
        out.append(model)
    return out


def _completion_record(path: pathlib.Path, root: pathlib.Path) -> dict | None:
    full = path if path.is_absolute() else root / path
    data = _read_json(full, None)
    return data if isinstance(data, dict) else None


def _signed_status_rollup(kind: str, rows: list[dict]) -> dict:
    blocked = [row.get("label") or row.get("model") for row in rows if not row.get("ok")]
    return {
        "schema": "hawking.studio_completion_signed_rollup.v1",
        "kind": kind,
        "model_count": len(rows),
        "passed_count": len(rows) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_labels": blocked,
        "rows": rows,
        "ok": not blocked,
    }


def _signed_parity_rollup(root: pathlib.Path, models: list[FrontierModel]) -> dict:
    rows = []
    for model in models:
        path = frontier_parity_runner.parity_path(root, model.label)
        record = _completion_record(path, root)
        status = frontier_parity_runner.record_status(record, model=model, require_signature=True)
        status["label"] = model.label
        status["path"] = str(path)
        rows.append(status)
    return _signed_status_rollup("architecture-parity", rows)


def _signed_coverage_rollup(root: pathlib.Path, labels: list[str], kind: str) -> dict:
    rows = []
    for label in labels:
        path = frontier_coverage_runner.default_path(root, label, kind)
        record = _completion_record(path, root)
        status = frontier_coverage_runner.record_status(record, kind=kind, require_signature=True)
        status["label"] = label
        status["path"] = str(path)
        rows.append(status)
    return _signed_status_rollup(kind, rows)


def _signed_native_rollup(root: pathlib.Path, labels: list[str], kind: str) -> dict:
    rows = []
    for label in labels:
        path = frontier_receipt_runner.default_path(root, label, kind)
        record = _completion_record(path, root)
        status = frontier_receipt_runner.record_status(record, kind=kind, require_signature=True)
        status["label"] = label
        status["path"] = str(path)
        rows.append(status)
    return _signed_status_rollup(kind, rows)


def _signed_experiment_rollup(root: pathlib.Path, labels: list[str]) -> dict:
    rows = []
    for label in labels:
        path = frontier_experiments.matrix_path(root, label)
        record = _completion_record(path, root)
        status = frontier_experiment_runner.record_status(record, label=label, require_signature=True)
        status["label"] = label
        status["path"] = str(path)
        rows.append(status)
    return _signed_status_rollup("experiment-depth", rows)


def _completion_requirement(req_id: str, title: str, ok: bool, detail: str,
                            evidence=None, required: bool = True) -> dict:
    return {
        "id": req_id,
        "title": title,
        "required": bool(required),
        "ok": bool(ok),
        "status": "pass" if ok else "block",
        "detail": detail,
        "evidence": evidence,
    }


def build_completion_audit(root: pathlib.Path, args) -> dict:
    models = _completion_models(getattr(args, "label", []) or [])
    labels = [model.label for model in models]
    all_labels = [m.label for m in FRONTIER_MODELS]

    refresh_path = pathlib.Path(args.require_refresh)
    artifacts = {
        "preflight_summary": _json_artifact(pathlib.Path(args.preflight_summary), root),
        "environment": _json_artifact(pathlib.Path(args.environment), root),
        "launch_packet": _json_artifact(pathlib.Path(args.launch_packet), root),
        "worktree_plan": _json_artifact(pathlib.Path(args.worktree_plan), root),
        "runtime_contract": _json_artifact(pathlib.Path(args.runtime_contract), root),
        "proof_pack": _json_artifact(pathlib.Path(args.proof_pack), root),
        "audit_grade": _json_artifact(pathlib.Path(args.audit_grade), root),
        "refresh": _json_artifact(refresh_path, root),
    }
    preflight = artifacts["preflight_summary"]
    environment = artifacts["environment"]
    launch_packet = artifacts["launch_packet"]
    worktree = artifacts["worktree_plan"]
    runtime_contract = artifacts["runtime_contract"]
    audit_grade = artifacts["audit_grade"]

    procurement_gate = build_launch_gate(
        root,
        phase="procure",
        allow_unreviewed=False,
        link_mb_s=args.link_mbs,
        efficiency=args.efficiency,
        scratch_gb=args.scratch_gb,
        cache_reserve_gb=args.cache_reserve_gb,
        storage_budget_gb=args.storage_budget_gb,
        max_wave_hours=args.max_wave_hours,
        require_refresh=str(refresh_path),
    )
    claim_gate = build_launch_gate(
        root,
        phase="claim",
        allow_unreviewed=False,
        link_mb_s=args.link_mbs,
        efficiency=args.efficiency,
        scratch_gb=args.scratch_gb,
        cache_reserve_gb=args.cache_reserve_gb,
        storage_budget_gb=args.storage_budget_gb,
        max_wave_hours=args.max_wave_hours,
        require_refresh=str(refresh_path),
    )
    refresh = _completion_record(refresh_path, root) or {}
    refresh_review = (
        refresh_review_status(refresh, root)
        if refresh.get("schema") == "hawking.frontier_refresh.v1"
        else {"review_worthy_count": 0, "reviewed": {}, "missing": ["refresh ledger missing or invalid"]}
    )
    license_rollup = frontier_licenses.license_rollup(_license_ledger(root), all_labels)
    source_provenance = frontier_provenance.provenance_rollup(root, labels)
    parity = _signed_parity_rollup(root, models)
    baseline = _signed_coverage_rollup(root, labels, "baseline")
    eval_cov = _signed_coverage_rollup(root, labels, "eval")
    serve = _signed_native_rollup(root, labels, "serve")
    ramcliff = _signed_native_rollup(root, labels, "ramcliff")
    experiments = _signed_experiment_rollup(root, labels)
    doctor = frontier_doctor_recovery.recovery_rollup(root, labels)
    claim_bundles = frontier_claims.claim_rollup(root, labels)

    requirements = [
        _completion_requirement(
            "split_clean_worktree",
            "Split dirty branch and keep local worktree clean",
            worktree.get("signature_ok") is True
            and worktree.get("ok") is True
            and worktree.get("risk") == "clean"
            and worktree.get("entries") == 0,
            f"risk={worktree.get('risk')} dirty_entries={worktree.get('entries')}",
            worktree,
        ),
        _completion_requirement(
            "studio_preflight_green",
            "Studio preflight and environment are green",
            preflight.get("signature_ok") is True
            and preflight.get("ok") is True
            and environment.get("signature_ok") is True
            and environment.get("ok") is True,
            (
                f"preflight_ok={preflight.get('ok')} environment_ok={environment.get('ok')} "
                f"environment_failures={environment.get('failure_count')}"
            ),
            {"preflight": preflight, "environment": environment},
        ),
        _completion_requirement(
            "license_and_review_gates",
            "License and refresh-review gates are closed by humans",
            license_rollup["ok"] and not refresh_review["missing"],
            (
                f"licenses={license_rollup['passed_count']}/{license_rollup['model_count']} "
                f"refresh_missing={len(refresh_review['missing'])}/"
                f"{refresh_review['review_worthy_count']}"
            ),
            {"licenses": license_rollup, "refresh_review": refresh_review},
        ),
        _completion_requirement(
            "procurement_gate_green",
            "Procurement launch gate and signed wave-0 packet are green",
            procurement_gate["ok"]
            and launch_packet.get("signature_ok") is True
            and launch_packet.get("procurement_permitted") is True,
            (
                f"procurement_gate={procurement_gate['ok']} "
                f"launch_packet_procurement_permitted={launch_packet.get('procurement_permitted')}"
            ),
            {"procurement_gate": procurement_gate, "launch_packet": launch_packet},
        ),
        _completion_requirement(
            "native_tq_runtime_contract",
            "Native .tq proof-mode runtime contract verifies",
            runtime_contract.get("signature_ok") is True
            and runtime_contract.get("ok") is True
            and (runtime_contract.get("proof_mode_required") or 0) >= 4,
            (
                f"runtime_ok={runtime_contract.get('ok')} "
                f"proof_env={runtime_contract.get('proof_mode_required')}"
            ),
            runtime_contract,
        ),
        _completion_requirement(
            "native_tq_serve",
            "Native .tq serve is measured and signed",
            serve["ok"],
            f"{serve['passed_count']}/{serve['model_count']} signed native .tq serve receipts verify",
            serve,
        ),
        _completion_requirement(
            "architecture_parity",
            "Architecture parity is measured and signed",
            parity["ok"],
            f"{parity['passed_count']}/{parity['model_count']} signed parity receipts verify",
            parity,
        ),
        _completion_requirement(
            "doctor_recovery_7b_plus",
            "Doctor recovery is measured at 7B+ without task collapse",
            doctor["ok"],
            f"{doctor['passed_count']}/{doctor['model_count']} signed Doctor recovery receipts verify",
            doctor,
        ),
        _completion_requirement(
            "ramcliff_energy",
            "RAM-cliff and energy are measured and signed",
            ramcliff["ok"],
            f"{ramcliff['passed_count']}/{ramcliff['model_count']} signed RAM-cliff/energy receipts verify",
            ramcliff,
        ),
        _completion_requirement(
            "same_box_baselines",
            "Same-box baselines are measured and signed",
            baseline["ok"],
            f"{baseline['passed_count']}/{baseline['model_count']} signed same-box baseline receipts verify",
            baseline,
        ),
        _completion_requirement(
            "frozen_eval_coverage",
            "Frozen eval coverage is measured and signed",
            eval_cov["ok"],
            f"{eval_cov['passed_count']}/{eval_cov['model_count']} signed eval coverage receipts verify",
            eval_cov,
        ),
        _completion_requirement(
            "source_provenance",
            "Source provenance is signed",
            source_provenance["ok"],
            (
                f"{source_provenance['passed_count']}/{source_provenance['model_count']} "
                "source-provenance receipts verify"
            ),
            source_provenance,
        ),
        _completion_requirement(
            "experiment_depth",
            "Same-box expensive-mode matrix and nulls are signed",
            experiments["ok"],
            f"{experiments['passed_count']}/{experiments['model_count']} signed experiment matrices verify",
            experiments,
        ),
        _completion_requirement(
            "claim_gate_green",
            "Claim launch gate is green",
            claim_gate["ok"],
            f"claim gate failure_count={claim_gate['failure_count']}",
            claim_gate,
        ),
        _completion_requirement(
            "signed_claim_bundles",
            "Final public-claim bundles verify",
            claim_bundles["ok"],
            f"{claim_bundles['passed_count']}/{claim_bundles['model_count']} signed claim bundles verify",
            claim_bundles,
        ),
        _completion_requirement(
            "audit_grade_target",
            "Studio audit target is reached on signed evidence",
            audit_grade.get("signature_ok") is True and audit_grade.get("target_reached") is True,
            f"target_reached={audit_grade.get('target_reached')} below_target={audit_grade.get('below_target_count')}",
            audit_grade,
        ),
    ]
    required_rows = [row for row in requirements if row["required"]]
    blocked = [row for row in required_rows if not row["ok"]]
    completion_ok = not blocked
    doc = {
        "schema": "hawking.studio_completion_audit.v1",
        "generated_at": _now(),
        "root": str(root),
        "git_commit": _git_commit(root),
        "labels": labels,
        "frontier_label_count": len(labels),
        "ok": completion_ok,
        "completion_ok": completion_ok,
        "required_count": len(required_rows),
        "passed_count": len(required_rows) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_requirements": [row["id"] for row in blocked],
        "requirements": requirements,
        "parameters": {
            "storage_budget_gb": args.storage_budget_gb,
            "link_mbs": args.link_mbs,
            "efficiency": args.efficiency,
            "scratch_gb": args.scratch_gb,
            "cache_reserve_gb": args.cache_reserve_gb,
            "max_wave_hours": args.max_wave_hours,
            "require_refresh": str(refresh_path),
        },
        "artifacts": artifacts,
        "note": (
            "This is the signed Hawking Studio 10/10 completion audit. A valid red receipt is expected on "
            "local hardware; it becomes green only after Studio preflight, native .tq serve, parity, "
            "Doctor recovery, RAM-cliff/energy, same-box baselines, experiment depth, and final signed "
            "claim bundles all verify."
        ),
    }
    return _sign_doc(doc)


def _completion_audit_status(doc: dict) -> dict:
    problems = []
    if doc.get("schema") != "hawking.studio_completion_audit.v1":
        problems.append("schema mismatch")
    signature_ok = _signature_ok(doc)
    if not signature_ok:
        problems.append("signature digest mismatch")
    requirements = doc.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        problems.append("requirements must be a non-empty list")
        requirements = []
    blocked = [row for row in requirements if row.get("required", True) and not row.get("ok")]
    if doc.get("blocked_count") != len(blocked):
        problems.append("blocked_count does not match requirements")
    required_count = len([row for row in requirements if row.get("required", True)])
    if doc.get("required_count") != required_count:
        problems.append("required_count does not match requirements")
    if doc.get("passed_count") != required_count - len(blocked):
        problems.append("passed_count does not match requirements")
    expected_ok = not blocked
    if doc.get("completion_ok") is not expected_ok or doc.get("ok") is not expected_ok:
        problems.append("completion_ok/ok do not match required requirement states")
    return {
        "schema_ok": doc.get("schema") == "hawking.studio_completion_audit.v1",
        "signature_ok": signature_ok,
        "completion_ok": bool(doc.get("completion_ok")),
        "required_count": doc.get("required_count"),
        "passed_count": doc.get("passed_count"),
        "blocked_count": doc.get("blocked_count"),
        "blocked_requirements": doc.get("blocked_requirements") or [],
        "ok": not problems,
        "problems": problems,
    }


def _print_worktree_plan(data: dict, max_paths: int = 8) -> None:
    print(f"# worktree split plan {data['generated_at']}  branch={data['branch']}  risk={data['risk']}")
    if not data.get("ok", False):
        print(f"# error: {data.get('error')}")
        return
    totals = data["totals"]
    print("# dirty entries={entries} staged={staged} unstaged={unstaged} untracked={untracked} "
          "subsystems={subsystems}".format(**totals))
    print("subsystem                 entries  staged  unstaged  untracked  risk    suggested branch")
    for group in data["groups"]:
        counts = group["counts"]
        branch = group["branch"] or "(do not commit)"
        print(f"{group['name']:<25s} {counts['total']:>7d} {counts['staged']:>7d} "
              f"{counts['unstaged']:>9d} {counts['untracked']:>10d} {group['risk']:<7s} {branch}")
        for entry in group["entries"][:max_paths]:
            print(f"  {entry['status']} {entry['path']}")
        remaining = counts["total"] - min(max_paths, counts["total"])
        if remaining > 0:
            print(f"  ... {remaining} more")
    if data["recommended_stack_order"]:
        print("\n# recommended stack order")
        for i, row in enumerate(data["recommended_stack_order"], start=1):
            print(f"{i}. {row['subsystem']} -> {row['branch']} ({row['count']} paths): {row['action']}")
    cleanup = data.get("local_cleanup") or {}
    if cleanup.get("counts", {}).get("total", 0):
        print("\n# local cleanup")
        print(cleanup["action"])


def _nonempty_dir(path: pathlib.Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _read_json(path: pathlib.Path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _write_json(path: pathlib.Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label)


def _hardware_snapshot(root: pathlib.Path = ROOT) -> dict:
    mem_gb = None
    rc, out, _ = _run(["sysctl", "-n", "hw.memsize"], timeout=10)
    if rc == 0:
        try:
            mem_gb = int(out.strip()) / 1e9
        except ValueError:
            pass
    rc_cpu, cpu_brand, _ = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=10)
    rc_ncpu, ncpu, _ = _run(["sysctl", "-n", "hw.ncpu"], timeout=10)
    rc_batt, batt, batt_err = _run(["pmset", "-g", "batt"], timeout=10)
    rc_therm, therm, therm_err = _run(["pmset", "-g", "therm"], timeout=10)
    usage = shutil.disk_usage(root)
    rc_hf, hf_out, hf_err = _run(["hf", "--version"], timeout=10)
    return {
        "profile": DEFAULT_HARDWARE.name,
        "target_ram_gb": DEFAULT_HARDWARE.ram_gb,
        "target_weight_budget_gb": DEFAULT_HARDWARE.weight_budget_gb,
        "target_ssd_tb": DEFAULT_HARDWARE.ssd_tb,
        "actual_ram_gb": round(mem_gb, 2) if mem_gb else None,
        "actual_cpu_brand": cpu_brand.strip() if rc_cpu == 0 and cpu_brand.strip() else None,
        "actual_cpu_count": int(ncpu.strip()) if rc_ncpu == 0 and ncpu.strip().isdigit() else None,
        "disk_total_gb": round(_gb_from_bytes(usage.total), 1),
        "disk_free_gb": round(_gb_from_bytes(usage.free), 1),
        "power_source": (batt or batt_err).splitlines()[0].strip()
        if rc_batt == 0 and (batt or batt_err).strip() else None,
        "thermal_status": (therm or therm_err).strip()[:500]
        if rc_therm == 0 and (therm or therm_err).strip() else None,
        "hf_version": (hf_out or hf_err).strip() if rc_hf == 0 else "missing",
        "git_commit": _git_commit(root),
    }


def _artifact_candidates(model: FrontierModel, root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    slug = _safe_label(model.label)
    paths = [
        root / f"{model.local_dir}.tq",
        root / "scratch" / f"{slug}.tq",
        root / "reports" / "condense" / f"{slug}.tq",
        root / "reports" / "condense" / f"{model.label}.tq",
    ]
    found = []
    for p in paths:
        if p.exists() and p not in found:
            found.append(p)
    for pat in (f"scratch/*{slug}*.tq", f"reports/condense/*{slug}*.tq"):
        for name in glob.glob(str(root / pat)):
            p = pathlib.Path(name)
            if p.exists() and p not in found:
                found.append(p)
    return found


def _frontier_record_path(model: FrontierModel, root: pathlib.Path = ROOT) -> pathlib.Path:
    return root / "reports" / "condense" / f"{model.label}_frontier.json"


def _artifact_inventory_path(model: FrontierModel, root: pathlib.Path = ROOT) -> pathlib.Path:
    return root / "reports" / "condense" / f"{model.label}_artifact_inventory.json"


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_digest(data: dict) -> str:
    unsigned = dict(data)
    unsigned.pop("signature", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sign_doc(data: dict) -> dict:
    data["signature"] = {"algorithm": SIGN_ALG, "digest": _canonical_digest(data)}
    return data


def _signature_ok(data: dict) -> bool:
    signature = data.get("signature") if isinstance(data.get("signature"), dict) else {}
    return signature.get("algorithm") == SIGN_ALG and signature.get("digest") == _canonical_digest(data)


def _json_artifact(path: pathlib.Path, root: pathlib.Path = ROOT) -> dict:
    full = path if path.is_absolute() else root / path
    out = {
        "path": str(path),
        "exists": full.exists(),
    }
    if not full.exists():
        return out
    out["bytes"] = full.stat().st_size
    out["sha256"] = _sha256_file(full)
    data = _read_json(full, None)
    if not isinstance(data, dict):
        out["json_ok"] = False
        return out
    out["json_ok"] = True
    out["schema"] = data.get("schema")
    out["signature_ok"] = _signature_ok(data) if isinstance(data.get("signature"), dict) else None
    if data.get("schema") == "hawking.studio_preflight_summary.v1":
        out["ok"] = bool(data.get("preflight_ok_before_summary"))
        out["generated_at"] = data.get("generated_at")
    elif data.get("schema") == "hawking.studio_environment.v1":
        out["ok"] = bool(data.get("ok"))
        out["failure_count"] = data.get("failure_count")
        out["warning_count"] = data.get("warning_count")
    elif data.get("schema") in (
        "hawking.frontier_license_decisions.v1",
        "hawking.frontier_review_decisions.v1",
    ):
        out["ok"] = bool(data.get("applyable"))
        out["applyable"] = bool(data.get("applyable"))
        out["decision_count"] = data.get("decision_count")
        out["operator_required_count"] = data.get("operator_required_count")
    elif data.get("schema") == "hawking.frontier_proof_pack.v1":
        out["ok"] = bool(data.get("ok"))
        out["model_count"] = data.get("model_count")
        out["blocked_claim_count"] = data.get("blocked_claim_count")
        rows = data.get("evidence_rows") if isinstance(data.get("evidence_rows"), list) else []
        out["evidence_count"] = len(rows)
    elif data.get("schema") == "hawking.frontier_refresh.v1":
        out["ok"] = True
        out["candidate_count"] = len(data.get("candidates") or [])
        review = refresh_review_status(data, root)
        out["review_worthy_count"] = review["review_worthy_count"]
        out["review_missing_count"] = len(review["missing"])
    elif data.get("schema") == "hawking.worktree_split_plan.v1":
        out["ok"] = bool(data.get("ok"))
        out["risk"] = data.get("risk")
        totals = data.get("totals") if isinstance(data.get("totals"), dict) else {}
        out["entries"] = totals.get("entries")
        out["subsystems"] = totals.get("subsystems")
        out["staged"] = totals.get("staged")
        out["unstaged"] = totals.get("unstaged")
        out["untracked"] = totals.get("untracked")
    elif data.get("schema") == "hawking.studio_runtime_contract.v1":
        out["ok"] = bool(data.get("ok"))
        out["profile_count"] = len(data.get("profiles") or [])
        out["workload_count"] = len(data.get("workloads") or [])
        proof = data.get("native_tq_proof_mode") if isinstance(data.get("native_tq_proof_mode"), dict) else {}
        out["proof_mode_required"] = len(proof.get("required_env") or [])
        out["default_unset_policy"] = data.get("default_unset_policy")
    elif data.get("schema") == "hawking.studio_wave0_launch_packet.v1":
        out["ok"] = bool(data.get("ok"))
        out["procurement_permitted"] = bool(data.get("procurement_permitted"))
        out["failure_count"] = data.get("failure_count")
        out["warning_count"] = data.get("warning_count")
    elif data.get("schema") == "hawking.studio_audit_grade.v1":
        out["ok"] = bool(data.get("ok"))
        out["target_reached"] = bool(data.get("target_reached"))
        out["frontier_claims_walled"] = bool(data.get("frontier_claims_walled"))
        out["facet_count"] = data.get("facet_count")
        out["below_target_count"] = data.get("below_target_count")
    return out


def _candidate_metadata(row: dict) -> dict:
    return {
        "model_id": row.get("modelId"),
        "source": row.get("source"),
        "url": row.get("url"),
        "last_modified": row.get("lastModified"),
        "downloads": row.get("downloads"),
        "likes": row.get("likes"),
        "pipeline_tag": row.get("pipeline_tag"),
        "tags": row.get("tags") or [],
    }


def _candidate_digest(row: dict) -> str:
    metadata = _candidate_metadata(row)
    return hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _candidate_review_reason(row: dict) -> str:
    model_id = row.get("modelId") or ""
    tags = [str(t) for t in (row.get("tags") or [])]
    hay = f"{model_id.lower()} {' '.join(t.lower() for t in tags)}"
    hits = [
        token for token in (
            "glm", "kimi", "deepseek", "qwen3", "qwen-3", "llama-3.1",
            "moe", "mixture-of-experts", "text-generation", "causal-lm",
            "v4", "k2", "fp8", "fp4",
        )
        if token in hay
    ]
    if hits:
        return f"review-worthy refresh candidate; matched {', '.join(sorted(set(hits)))}"
    return "review-worthy refresh candidate"


def _official_receipts(model: FrontierModel, root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    receipt_dir = root / "receipts" / "official"
    if not receipt_dir.exists():
        return []
    needles = {model.label.lower(), _safe_label(model.label).lower()}
    out = []
    for p in receipt_dir.glob("*.json"):
        name = p.name.lower()
        if any(n in name for n in needles):
            out.append(p)
    return sorted(out)


def _release_events(root: pathlib.Path = ROOT) -> dict[str, list[dict]]:
    path = root / RELEASE_LOG
    rows: dict[str, list[dict]] = {}
    if not path.exists():
        return rows
    for line in open(path):
        try:
            row = json.loads(line)
        except Exception:
            continue
        rows.setdefault(row.get("label", ""), []).append(row)
    return rows


def _download_events(root: pathlib.Path = ROOT) -> dict[str, list[dict]]:
    path = root / DOWNLOAD_LOG
    rows: dict[str, list[dict]] = {}
    if not path.exists():
        return rows
    for line in open(path):
        try:
            row = json.loads(line)
        except Exception:
            continue
        rows.setdefault(row.get("label", ""), []).append(row)
    return rows


def _operator_events(root: pathlib.Path = ROOT) -> dict[str, list[dict]]:
    path = root / EVENT_LOG
    rows: dict[str, list[dict]] = {}
    if not path.exists():
        return rows
    for line in open(path):
        try:
            row = json.loads(line)
        except Exception:
            continue
        rows.setdefault(row.get("label", ""), []).append(row)
    return rows


def _event_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "stages": {}}
    stages = {}
    for row in rows:
        stage = row.get("stage", "unknown")
        stages.setdefault(stage, {"count": 0, "last_status": None, "total_duration_s": 0.0})
        stages[stage]["count"] += 1
        stages[stage]["last_status"] = row.get("status")
        stages[stage]["last_ts"] = row.get("ts")
        stages[stage]["last_artifact"] = row.get("artifact")
        try:
            stages[stage]["total_duration_s"] += float(row.get("duration_s") or 0.0)
        except (TypeError, ValueError):
            pass
    for stage in stages.values():
        stage["total_duration_s"] = round(stage["total_duration_s"], 3)
    return {"count": len(rows), "last": rows[-1], "stages": stages}


def _download_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    last = rows[-1]
    progress = last.get("progress") if isinstance(last.get("progress"), dict) else {}
    diagnostics = last.get("diagnostics") if isinstance(last.get("diagnostics"), dict) else {}
    return {
        "count": len(rows),
        "last_started_at": last.get("started_at"),
        "last_ended_at": last.get("ended_at"),
        "last_returncode": last.get("returncode"),
        "last_hf_download_returncode": last.get("hf_download_returncode"),
        "last_attempt": last.get("attempt"),
        "last_attempt_count": last.get("attempt_count"),
        "last_retry_reason": last.get("retry_reason"),
        "last_will_retry": last.get("will_retry"),
        "last_duration_s": last.get("duration_s"),
        "last_delta_gb": last.get("delta_local_dir_gb"),
        "last_observed_mb_s": last.get("observed_mb_s_from_delta"),
        "last_eta_hours_at_observed_rate": last.get("eta_hours_at_observed_rate"),
        "last_already_populated_at_start": last.get("already_populated_at_start"),
        "last_progress_sample_count": progress.get("sample_count"),
        "last_progress_avg_mb_s": progress.get("average_tracked_mb_s"),
        "last_progress_window_mb_s": progress.get("last_window_mb_s"),
        "last_longest_no_progress_s": progress.get("longest_no_progress_s"),
        "last_stalled": progress.get("stalled"),
        "last_terminated_for_stall": progress.get("terminated_for_stall"),
        "last_stall_reason": progress.get("stall_reason"),
        "last_diagnostics_present": bool(diagnostics),
        "last_diagnostic_recommendations": diagnostics.get("recommendations", []),
        "last_network_probe_ok": ((diagnostics.get("network_probe") or {}).get("ok")
                                  if isinstance(diagnostics.get("network_probe"), dict) else None),
        "last_verify_returncode": ((last.get("verify") or {}).get("returncode")
                                   if isinstance(last.get("verify"), dict) else None),
    }


def _license_ledger(root: pathlib.Path = ROOT) -> dict:
    return _read_json(root / LICENSE_PATH, {})


def _refresh_review_ledger(root: pathlib.Path = ROOT) -> dict:
    return _read_json(root / REFRESH_REVIEW_PATH, {})


def _is_review_worthy_candidate(row: dict) -> bool:
    """Whether an unknown refresh row is plausible enough to require a human frontier decision."""
    if not row.get("ok") or row.get("known"):
        return False
    model_id = (row.get("modelId") or "").lower()
    tags = " ".join(str(t).lower() for t in (row.get("tags") or []))
    hay = f"{model_id} {tags}"
    needles = (
        "glm", "kimi", "deepseek", "qwen3", "qwen-3", "llama-3.1",
        "moe", "mixture-of-experts", "text-generation", "causal-lm",
        "v4", "k2", "fp8", "fp4",
    )
    return any(n in hay for n in needles)


def refresh_review_status(refresh: dict, root: pathlib.Path = ROOT) -> dict:
    reviews = _refresh_review_ledger(root)
    rows = refresh.get("candidates") or []
    worthy = [r for r in rows if _is_review_worthy_candidate(r)]
    reviewed = {}
    missing = []
    for row in worthy:
        mid = row.get("modelId")
        review = reviews.get(mid)
        if review and review.get("decision") in ("accept", "reject", "watch"):
            reviewed[mid] = review
        else:
            missing.append(mid)
    return {
        "review_path": str(root / REFRESH_REVIEW_PATH),
        "review_worthy_count": len(worthy),
        "reviewed": reviewed,
        "missing": missing,
    }


def artifact_inventory_status(model: FrontierModel, artifacts: list[pathlib.Path],
                              root: pathlib.Path = ROOT) -> dict:
    path = _artifact_inventory_path(model, root)
    rec = _read_json(path, {})
    problems = []
    if not rec:
        problems.append("artifact inventory is missing")
    if rec and rec.get("schema") != "hawking.frontier_artifact_inventory.v1":
        problems.append("artifact inventory schema mismatch")
    if rec and rec.get("label") != model.label:
        problems.append("artifact inventory label mismatch")
    rows = rec.get("artifacts") if isinstance(rec.get("artifacts"), list) else []
    indexed = {r.get("path"): r for r in rows}
    for artifact in artifacts:
        key = str(artifact)
        row = indexed.get(key) or indexed.get(str(artifact.resolve()))
        if not row:
            problems.append(f"artifact {key} missing from inventory")
            continue
        try:
            size = artifact.stat().st_size
        except OSError:
            problems.append(f"artifact {key} cannot be stat()ed")
            continue
        if row.get("bytes") != size:
            problems.append(f"artifact {key} size changed since inventory")
        if not row.get("sha256"):
            problems.append(f"artifact {key} missing sha256")
    return {
        "path": str(path),
        "exists": bool(rec),
        "ok": bool(rec) and not problems and bool(artifacts),
        "problems": problems,
        "artifact_count": len(rows),
    }


def _release_evidence(model: FrontierModel, root: pathlib.Path = ROOT) -> dict:
    source_dir = root / model.local_dir
    artifacts = _artifact_candidates(model, root)
    record = _frontier_record_path(model, root)
    receipts = _official_receipts(model, root)
    inventory = artifact_inventory_status(model, artifacts, root)
    return {
        "source_dir": str(source_dir),
        "source_exists": _nonempty_dir(source_dir),
        "source_gb": round(_path_size_gb(source_dir), 3),
        "artifact_paths": [str(p) for p in artifacts],
        "artifact_exists": bool(artifacts),
        "artifact_gb": round(sum(_path_size_gb(p) for p in artifacts), 3),
        "frontier_record": str(record),
        "frontier_record_exists": record.exists(),
        "artifact_inventory": inventory,
        "official_receipts": [str(p) for p in receipts],
        "official_receipt_exists": bool(receipts),
    }


def release_guard(model: FrontierModel, root: pathlib.Path = ROOT) -> tuple[bool, list[str], dict]:
    evidence = _release_evidence(model, root)
    missing = []
    if not evidence["source_exists"]:
        missing.append("source directory is absent or empty")
    if not evidence["artifact_exists"]:
        missing.append("no .tq artifact candidate exists")
    if not evidence["artifact_inventory"]["ok"]:
        missing.extend(evidence["artifact_inventory"]["problems"])
    if not (evidence["frontier_record_exists"] or evidence["official_receipt_exists"]):
        missing.append("no frontier record or official receipt exists")
    return not missing, missing, evidence


def _parse_dry_run_sizes(text: str) -> dict:
    total = 0.0
    files = 0
    for line in text.splitlines():
        m = SIZE_RE.search(line.strip())
        if not m:
            continue
        total += float(m.group(1)) * SIZE_SCALE[m.group(2)]
        files += 1
    return {"file_count": files, "size_gb": round(total / 1e9, 3)}


def _hf_dry_run(model: FrontierModel, include: str, timeout: int) -> dict:
    cmd = ["hf", "download", model.hf_id, "--dry-run", "--include", include, "--max-workers", "1"]
    rc, out, err = _run(cmd, timeout=timeout)
    parsed = _parse_dry_run_sizes(out + "\n" + err)
    return {
        "cmd": cmd,
        "returncode": rc,
        "include": include,
        "file_count": parsed["file_count"],
        "size_gb": parsed["size_gb"],
        "stderr_tail": err[-800:],
    }


def _hf_model_info(model: FrontierModel, timeout: int) -> dict:
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(model.hf_id, safe="/")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        return {"url": url, "ok": False, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"url": url, "ok": False, "error": f"{type(e).__name__}: {e}"}
    card = data.get("cardData") if isinstance(data.get("cardData"), dict) else {}
    tags = data.get("tags") or []
    license_tag = next((t.split("license:", 1)[1] for t in tags if t.startswith("license:")), None)
    return {
        "url": url,
        "ok": True,
        "modelId": data.get("modelId"),
        "sha": data.get("sha"),
        "lastModified": data.get("lastModified"),
        "gated": data.get("gated"),
        "private": data.get("private"),
        "disabled": data.get("disabled"),
        "downloads": data.get("downloads"),
        "likes": data.get("likes"),
        "pipeline_tag": data.get("pipeline_tag"),
        "license": card.get("license") or license_tag,
        "tags": tags[:40],
    }


def cycle_plan(root: pathlib.Path = ROOT, link_mb_s: float = 300.0, efficiency: float = 0.7,
               scratch_gb: float = 200.0, cache_reserve_gb: float = CACHE_RESERVE_GB,
               keep_outputs: bool = True) -> dict:
    kept_outputs = 0.0
    peak = 0.0
    rows = []
    for model in sorted(FRONTIER_MODELS, key=lambda m: (m.download_gb, m.artifact_gb())):
        output = model.artifact_gb()
        before = kept_outputs if keep_outputs else 0.0
        step_peak = before + model.download_gb + output + scratch_gb + cache_reserve_gb
        peak = max(peak, step_peak)
        rows.append({
            "label": model.label,
            "hf_id": model.hf_id,
            "source_gb": model.download_gb,
            "artifact_gb": round(output, 3),
            "step_peak_gb": round(step_peak, 3),
            "download_eta_hours": eta_hours(model.download_gb, link_mb_s, efficiency),
        })
        if keep_outputs:
            kept_outputs += output
    return {
        "link_mb_s": link_mb_s,
        "efficiency": efficiency,
        "scratch_gb": scratch_gb,
        "cache_reserve_gb": cache_reserve_gb,
        "keep_outputs": keep_outputs,
        "total_source_gb": round(total_download_gb(), 3),
        "total_artifact_gb": round(total_artifact_gb(), 3),
        "full_fit_gb": round(total_download_gb() + total_artifact_gb(), 3),
        "cycle_peak_gb": round(peak, 3),
        "download_eta_hours": eta_hours(total_download_gb(), link_mb_s, efficiency),
        "rows": rows,
    }


def build_ledger(root: pathlib.Path = ROOT, refresh_hf: bool = False, dry_run_sizes: bool = False,
                 include: str = "*.safetensors", timeout: int = 180,
                 link_mb_s: float = 300.0, efficiency: float = 0.7,
                 scratch_gb: float = 200.0, cache_reserve_gb: float = CACHE_RESERVE_GB,
                 storage_budget_gb: float | None = None,
                 max_wave_hours: float = 6.0) -> dict:
    licenses = _license_ledger(root)
    license_rollup = frontier_licenses.license_rollup(licenses, [m.label for m in FRONTIER_MODELS])
    releases = _release_events(root)
    downloads = _download_events(root)
    events = _operator_events(root)
    models = []
    for model in FRONTIER_MODELS:
        evidence = _release_evidence(model, root)
        source_dir = root / model.local_dir
        state = "not-staged"
        if evidence["source_exists"]:
            state = "staged"
        elif releases.get(model.label):
            state = "source-released"
        ok_release, missing_release, _ = release_guard(model, root)
        row = {
            "label": model.label,
            "hf_id": model.hf_id,
            "local_dir": model.local_dir,
            "state": state,
            "role": model.role,
            "source_kind": model.source_kind,
            "total_b": model.total_b,
            "active_b": model.active_b,
            "serve_bpw": model.serve_bpw,
            "manifest_source_gb": model.download_gb,
            "manifest_artifact_gb": round(model.artifact_gb(), 3),
            "q4k_gb": round(model.q4k_gb(), 3),
            "resident_target": model.fits_resident(DEFAULT_HARDWARE),
            "source_dir_exists": evidence["source_exists"],
            "source_dir_gb": evidence["source_gb"],
            "artifact_exists": evidence["artifact_exists"],
            "artifact_gb": evidence["artifact_gb"],
            "artifact_paths": evidence["artifact_paths"],
            "artifact_inventory": evidence["artifact_inventory"],
            "frontier_record_exists": evidence["frontier_record_exists"],
            "frontier_record": evidence["frontier_record"],
            "official_receipts": evidence["official_receipts"],
            "release_safe": ok_release,
            "release_blockers": missing_release,
            "license": licenses.get(model.label, {"status": "unreviewed"}),
            "license_gate": next(r for r in license_rollup["rows"] if r["label"] == model.label),
            "release_events": releases.get(model.label, []),
            "download": _download_summary(downloads.get(model.label, [])),
            "events": _event_summary(events.get(model.label, [])),
            "baseline_coverage": frontier_coverage.baseline_status(root, model.label),
            "eval_coverage": frontier_coverage.eval_status(root, model.label),
            "source_provenance": frontier_provenance.provenance_status(root, model.label),
            "serve_receipt": frontier_receipts.serve_status(root, model.label),
            "ramcliff_receipt": frontier_receipts.ramcliff_status(root, model.label),
            "doctor_recovery": frontier_doctor_recovery.recovery_status(root, model.label),
            "experiment_matrix": frontier_experiments.experiment_status(root, model.label),
            "note": model.note,
        }
        if refresh_hf:
            row["hf_info"] = _hf_model_info(model, timeout=min(timeout, 60))
        if dry_run_sizes:
            row["hf_dry_run"] = _hf_dry_run(model, include=include, timeout=timeout)
        models.append(row)
    usage = shutil.disk_usage(root)
    storage_budget = storage_budget_gb if storage_budget_gb is not None else _gb_from_bytes(usage.free)
    return {
        "schema": "hawking.frontier_ledger.v1",
        "generated_at": _now(),
        "root": str(root),
        "hardware": _hardware_snapshot(root),
        "cache": _cache_snapshot(root),
        "manifest": {
            "model_count": len(FRONTIER_MODELS),
            "total_source_gb": round(total_download_gb(), 3),
            "total_artifact_gb": round(total_artifact_gb(), 3),
            "full_fit_gb": round(total_download_gb() + total_artifact_gb(), 3),
        },
        "cycle_plan": cycle_plan(root, link_mb_s, efficiency, scratch_gb, cache_reserve_gb),
        "storage_wave_plan": manifest_storage_wave_plan(
            storage_budget_gb=storage_budget,
            link_mb_s=link_mb_s,
            efficiency=efficiency,
            scratch_gb=scratch_gb,
            cache_reserve_gb=cache_reserve_gb,
            max_wave_hours=max_wave_hours,
        ),
        "models": models,
    }


def manifest_findings() -> list[dict]:
    findings = []
    seen: dict[str, str] = {}
    fields = {
        "label": [m.label for m in FRONTIER_MODELS],
        "hf_id": [m.hf_id for m in FRONTIER_MODELS],
        "local_dir": [m.local_dir for m in FRONTIER_MODELS],
    }
    for field, values in fields.items():
        seen.clear()
        for value in values:
            key = value.lower()
            if key in seen:
                findings.append({"severity": "fail", "field": field,
                                 "message": f"duplicate {field}: {value} also used by {seen[key]}"})
            seen[key] = value
    alias_owner: dict[str, str] = {}
    for model in FRONTIER_MODELS:
        if model.total_b <= 0 or model.serve_bpw <= 0 or model.download_gb <= 0:
            findings.append({"severity": "fail", "field": model.label,
                             "message": "params, serve_bpw, and download_gb must be positive"})
        if model.artifact_gb() > DEFAULT_HARDWARE.weight_budget_gb:
            findings.append({"severity": "fail", "field": model.label,
                             "message": f"resident target overflows {DEFAULT_HARDWARE.weight_budget_gb:.0f} GB"})
        for alias in model.aliases:
            key = alias.lower()
            if key in alias_owner:
                findings.append({"severity": "fail", "field": "alias",
                                 "message": f"alias {alias} shared by {model.label} and {alias_owner[key]}"})
            alias_owner[key] = model.label
    return findings


def manifest_drift_findings(root: pathlib.Path = ROOT) -> list[dict]:
    findings = []
    for rel in MANIFEST_CONSUMERS:
        path = root / rel
        try:
            text = path.read_text()
        except Exception as e:
            findings.append({"severity": "fail", "file": rel, "message": f"cannot read consumer: {e}"})
            continue
        if "studio_manifest" not in text:
            findings.append({"severity": "fail", "file": rel,
                             "message": "frontier consumer does not import studio_manifest"})
        if rel != "tools/condense/studio_manifest.py" and "FRONTIER_MODELS" not in text:
            findings.append({"severity": "fail", "file": rel,
                             "message": "frontier consumer does not reference FRONTIER_MODELS"})
        retired = [m for m in RETIRED_FRONTIER_MARKERS if m in text]
        if retired:
            findings.append({"severity": "fail", "file": rel,
                             "message": f"retired hard-coded frontier marker(s): {', '.join(retired)}"})
    return findings


def build_launch_gate(root: pathlib.Path = ROOT, phase: str = "procure", allow_unreviewed: bool = False,
                      link_mb_s: float = 300.0, efficiency: float = 0.7,
                      scratch_gb: float = 200.0, cache_reserve_gb: float = CACHE_RESERVE_GB,
                      storage_budget_gb: float | None = None, max_wave_hours: float = 6.0,
                      require_refresh: str | None = None) -> dict:
    ledger = build_ledger(root, link_mb_s=link_mb_s, efficiency=efficiency, scratch_gb=scratch_gb,
                          cache_reserve_gb=cache_reserve_gb,
                          storage_budget_gb=storage_budget_gb,
                          max_wave_hours=max_wave_hours)
    parity = frontier_parity.build_plan(root)
    checks = []

    def add(name: str, ok: bool, severity: str, detail: str, evidence=None) -> None:
        checks.append({"name": name, "ok": bool(ok), "severity": severity,
                       "detail": detail, "evidence": evidence})

    findings = manifest_findings()
    add("manifest-consistency", not any(f["severity"] == "fail" for f in findings), "fail",
        "unique labels/hf ids/local dirs, positive sizes, resident targets fit", findings)
    drift = manifest_drift_findings(root)
    add("manifest-consumer-drift", not any(f["severity"] == "fail" for f in drift), "fail",
        "frontier consumers derive model rows from studio_manifest.py", drift)

    free = ledger["hardware"]["disk_free_gb"]
    peak = ledger["cycle_plan"]["cycle_peak_gb"]
    add("cycle-disk-free", free >= peak, "fail",
        f"free {free:.0f} GB vs cycle peak {peak:.0f} GB", {"free_gb": free, "cycle_peak_gb": peak})

    wave_peak = ledger["storage_wave_plan"]["planned_peak_gb"]
    impossible = ledger["storage_wave_plan"]["impossible_labels"]
    add("storage-wave-plan", not impossible and wave_peak <= free, "fail",
        f"planned wave peak {wave_peak:.0f} GB vs free {free:.0f} GB; "
        f"{ledger['storage_wave_plan']['wave_count']} wave(s), checkpoint <= {max_wave_hours:.1f}h where possible",
        {"planned_peak_gb": wave_peak, "free_gb": free, "impossible_labels": impossible})

    cache_local = ledger["cache"]["project_local"]
    add("hf-cache-project-local", all(cache_local.values()), "warn",
        "HF_HOME/HF_HUB_CACHE/HF_XET_CACHE are project-local or intentionally overridden",
        {"paths": ledger["cache"]["paths"], "project_local": cache_local})

    full_fit = ledger["manifest"]["full_fit_gb"]
    add("full-fit-optional", full_fit <= DEFAULT_HARDWARE.ssd_gb, "warn",
        f"full-fit {full_fit:.0f} GB vs target SSD {DEFAULT_HARDWARE.ssd_gb:.0f} GB; cycle mode may still pass",
        {"full_fit_gb": full_fit, "target_ssd_gb": DEFAULT_HARDWARE.ssd_gb})

    license_rollup = frontier_licenses.license_rollup(_license_ledger(root), [m.label for m in FRONTIER_MODELS])
    add("license-approval", allow_unreviewed or license_rollup["ok"], "fail",
        f"{license_rollup['passed_count']}/{license_rollup['model_count']} model licenses have accepted terms",
        license_rollup)

    if require_refresh:
        path = pathlib.Path(require_refresh)
        refresh = _read_json(path, {})
        ok = refresh.get("schema") == "hawking.frontier_refresh.v1"
        add("refresh-ledger", ok, "fail", f"refresh ledger exists and has expected schema: {path}",
            {"path": str(path), "schema": refresh.get("schema")})
        if ok:
            review = refresh_review_status(refresh, root)
            add("refresh-candidate-review", not review["missing"], "fail",
                f"{len(review['missing'])}/{review['review_worthy_count']} review-worthy refresh candidates need accept/reject/watch",
                review)
    else:
        add("refresh-ledger", False, "warn",
            "no refresh ledger required; run frontier_ops.py refresh before a real launch", None)

    if phase == "claim":
        blocked = parity["blocked_claims"]
        add("frontier-parity", blocked == 0, "fail",
            f"{blocked}/{parity['model_count']} frontier parity rows block claims", parity)
        labels = [m.label for m in FRONTIER_MODELS]
        provenance = frontier_provenance.provenance_rollup(root, labels)
        baseline = frontier_coverage.baseline_rollup(root, labels)
        eval_cov = frontier_coverage.eval_rollup(root, labels)
        serve_receipts = frontier_receipts.serve_rollup(root, labels)
        ramcliff_receipts = frontier_receipts.ramcliff_rollup(root, labels)
        doctor_recovery = frontier_doctor_recovery.recovery_rollup(root, labels)
        experiments = frontier_experiments.experiment_rollup(root, labels)
        claim_bundles = frontier_claims.claim_rollup(root, labels)
        add("frontier-source-provenance", provenance["ok"], "fail",
            f"{provenance['passed_count']}/{provenance['model_count']} frontier source-provenance receipts verify",
            provenance)
        add("frontier-baseline-coverage", baseline["ok"], "fail",
            f"{baseline['passed_count']}/{baseline['model_count']} frontier models have same-box baselines or explicit N/A",
            baseline)
        add("frontier-eval-coverage", eval_cov["ok"], "fail",
            f"{eval_cov['passed_count']}/{eval_cov['model_count']} frontier models have frozen eval coverage",
            eval_cov)
        add("frontier-native-serve-receipts", serve_receipts["ok"], "fail",
            f"{serve_receipts['passed_count']}/{serve_receipts['model_count']} native .tq serve receipts are claim-admissible",
            serve_receipts)
        add("frontier-ramcliff-receipts", ramcliff_receipts["ok"], "fail",
            f"{ramcliff_receipts['passed_count']}/{ramcliff_receipts['model_count']} RAM-cliff receipts are claim-admissible",
            ramcliff_receipts)
        add("frontier-doctor-recovery", doctor_recovery["ok"], "fail",
            f"{doctor_recovery['passed_count']}/{doctor_recovery['model_count']} Doctor recovery receipts verify",
            doctor_recovery)
        add("frontier-experiment-depth", experiments["ok"], "fail",
            f"{experiments['passed_count']}/{experiments['model_count']} frontier models have expensive-mode experiment depth",
            experiments)
        add("frontier-signed-claim-bundles", claim_bundles["ok"], "fail",
            f"{claim_bundles['passed_count']}/{claim_bundles['model_count']} signed claim bundles verify",
            claim_bundles)
    else:
        add("frontier-parity", parity["blocked_claims"] == 0, "warn",
            f"{parity['blocked_claims']}/{parity['model_count']} parity rows still block public claims",
            {"blocked_claims": parity["blocked_claims"], "model_count": parity["model_count"]})

    failures = [c for c in checks if not c["ok"] and c["severity"] == "fail"]
    warnings = [c for c in checks if not c["ok"] and c["severity"] == "warn"]
    return {
        "schema": "hawking.frontier_launch_gate.v1",
        "generated_at": _now(),
        "phase": phase,
        "allow_unreviewed": allow_unreviewed,
        "ok": not failures,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "checks": checks,
        "summary": {
            "model_count": ledger["manifest"]["model_count"],
            "cycle_peak_gb": ledger["cycle_plan"]["cycle_peak_gb"],
            "storage_wave_peak_gb": ledger["storage_wave_plan"]["planned_peak_gb"],
            "storage_wave_count": ledger["storage_wave_plan"]["wave_count"],
            "disk_free_gb": free,
            "download_eta_hours": ledger["cycle_plan"]["download_eta_hours"],
            "parity_blocked": parity["blocked_claims"],
            "source_provenance_blocked": (
                frontier_provenance.provenance_rollup(root, [m.label for m in FRONTIER_MODELS])["blocked_count"]
                if phase == "claim" else None
            ),
            "baseline_coverage_blocked": (
                frontier_coverage.baseline_rollup(root, [m.label for m in FRONTIER_MODELS])["blocked_count"]
                if phase == "claim" else None
            ),
            "eval_coverage_blocked": (
                frontier_coverage.eval_rollup(root, [m.label for m in FRONTIER_MODELS])["blocked_count"]
                if phase == "claim" else None
            ),
            "serve_receipts_blocked": (
                frontier_receipts.serve_rollup(root, [m.label for m in FRONTIER_MODELS])["blocked_count"]
                if phase == "claim" else None
            ),
            "ramcliff_receipts_blocked": (
                frontier_receipts.ramcliff_rollup(root, [m.label for m in FRONTIER_MODELS])["blocked_count"]
                if phase == "claim" else None
            ),
            "doctor_recovery_blocked": (
                frontier_doctor_recovery.recovery_rollup(root, [m.label for m in FRONTIER_MODELS])["blocked_count"]
                if phase == "claim" else None
            ),
            "experiment_depth_blocked": (
                frontier_experiments.experiment_rollup(root, [m.label for m in FRONTIER_MODELS])["blocked_count"]
                if phase == "claim" else None
            ),
            "signed_claim_bundles_blocked": (
                frontier_claims.claim_rollup(root, [m.label for m in FRONTIER_MODELS])["blocked_count"]
                if phase == "claim" else None
            ),
        },
    }


def _parity_row_by_label(root: pathlib.Path = ROOT) -> dict[str, dict]:
    plan = frontier_parity.build_plan(root)
    return {row["label"]: row for row in plan["rows"]}


def _download_cmd(model: FrontierModel) -> str:
    return (
        "python3.12 tools/condense/procure.py "
        f"{model.label} --retries 2 --min-observed-mbs 80 --verify "
        "--progress-interval-s 60 --stall-timeout-s 900"
    )


def _lifecycle_node(model: FrontierModel, ledger_row: dict, parity_row: dict | None,
                    root: pathlib.Path = ROOT) -> dict:
    license_status = ledger_row.get("license", {}).get("status", "unreviewed")
    license_gate = ledger_row.get("license_gate", {})
    source = ledger_row["source_dir_exists"]
    artifact = ledger_row["artifact_exists"]
    inventory_ok = ledger_row.get("artifact_inventory", {}).get("ok", False)
    receiptish = ledger_row["frontier_record_exists"] or bool(ledger_row["official_receipts"])
    released = bool(ledger_row.get("release_events")) and not source
    parity_pass = bool(parity_row and parity_row.get("claim_gate") == "ALLOW")
    source_provenance_ok = bool(ledger_row.get("source_provenance", {}).get("ok"))
    baseline_ok = bool(ledger_row.get("baseline_coverage", {}).get("ok"))
    eval_ok = bool(ledger_row.get("eval_coverage", {}).get("ok"))
    serve_ok = bool(ledger_row.get("serve_receipt", {}).get("ok"))
    ramcliff_ok = bool(ledger_row.get("ramcliff_receipt", {}).get("ok"))
    doctor_ok = bool(ledger_row.get("doctor_recovery", {}).get("ok"))
    experiment_ok = bool(ledger_row.get("experiment_matrix", {}).get("ok"))
    claim_bundle = frontier_claims.bundle_status(root, model.label)
    claim_bundle_ok = bool(claim_bundle.get("ok"))
    download = ledger_row.get("download", {})
    events = ledger_row.get("events", {})
    blockers = []
    commands = []
    state = "planned"

    if license_status == "rejected":
        state = "blocked-license"
        blockers.append("license review rejected")
    elif not license_gate.get("ok"):
        state = "needs-license-review"
        blockers.extend(license_gate.get("problems", ["license/gating approval not accepted"]))
        commands.append(
            f"hawking studio record-license {model.label} "
            "--status accepted --by <name> --license <id> --terms-url <url> "
            "--allowed-use research --redistribution none "
            "--source-policy local-only-delete-after-bake --note <decision>"
        )
    elif download.get("last_will_retry"):
        state = "download-retry-pending"
        blockers.append(download.get("last_retry_reason") or "download scheduled retry")
        blockers.extend(download.get("last_diagnostic_recommendations", [])[:1])
        commands.append(_download_cmd(model))
    elif download.get("last_terminated_for_stall"):
        state = "download-stalled"
        blockers.append(download.get("last_stall_reason") or "last download was terminated for stalled progress")
        blockers.extend(download.get("last_diagnostic_recommendations", [])[:1])
        commands.append(_download_cmd(model))
    elif download.get("count") and download.get("last_returncode") not in (None, 0):
        state = "download-failed"
        blockers.append(f"last download/verify returncode {download.get('last_returncode')}")
        blockers.extend(download.get("last_diagnostic_recommendations", [])[:1])
        commands.append(_download_cmd(model))
    elif not source and not artifact:
        state = "ready-download"
        commands.append(_download_cmd(model))
    elif source and not artifact:
        state = "ready-bake"
        commands.append(f"python3.12 tools/condense/studio_run.py --frontier {model.label}")
        commands.append(
            f"python3.12 tools/condense/frontier_ops.py record-event {model.label} "
            "--stage bake --status pass --duration-s <seconds> --artifact <path>"
        )
    elif artifact and not inventory_ok:
        state = "needs-artifact-inventory"
        blockers.extend(ledger_row.get("artifact_inventory", {}).get("problems", []))
        commands.append(f"python3.12 tools/condense/frontier_ops.py artifact-inventory {model.label}")
    elif artifact and not receiptish:
        state = "needs-receipt"
        blockers.append("artifact exists without frontier record or official receipt")
        commands.append(
            f"write reports/condense/{model.label}_frontier.json or receipts/official/*{model.label}*.json"
        )
    elif ledger_row["release_safe"]:
        state = "ready-release-source"
        commands.append(f"python3.12 tools/condense/frontier_ops.py release-source {model.label} --dry-run")
    elif released or (artifact and not source):
        if parity_pass:
            if not source_provenance_ok:
                state = "claim-blocked-source-provenance"
                blockers.extend(ledger_row.get("source_provenance", {}).get(
                    "problems", ["source provenance missing"]
                ))
                commands.append("hawking studio source-provenance-plan")
            elif not eval_ok:
                state = "claim-blocked-eval"
                blockers.extend(ledger_row.get("eval_coverage", {}).get("problems", ["eval coverage missing"]))
                commands.append("hawking studio coverage-plan")
            elif not baseline_ok:
                state = "claim-blocked-baseline"
                blockers.extend(ledger_row.get("baseline_coverage", {}).get("problems", ["baseline coverage missing"]))
                commands.append("hawking studio coverage-plan")
            elif not serve_ok:
                state = "claim-blocked-serve"
                blockers.extend(ledger_row.get("serve_receipt", {}).get("problems", ["serve receipt missing"]))
                commands.append("hawking studio receipt-plan")
            elif not ramcliff_ok:
                state = "claim-blocked-ramcliff"
                blockers.extend(ledger_row.get("ramcliff_receipt", {}).get("problems", ["RAM-cliff receipt missing"]))
                commands.append("hawking studio receipt-plan")
            elif not doctor_ok:
                state = "claim-blocked-doctor"
                blockers.extend(ledger_row.get("doctor_recovery", {}).get("problems", ["Doctor recovery receipt missing"]))
                commands.append("hawking studio doctor-recovery-plan")
            elif not experiment_ok:
                state = "claim-blocked-experiment"
                blockers.extend(ledger_row.get("experiment_matrix", {}).get("problems", ["experiment matrix missing"]))
                commands.append("hawking studio experiment-plan")
            elif not claim_bundle_ok:
                state = "claim-blocked-bundle"
                blockers.extend(claim_bundle.get("problems", ["signed claim bundle missing"]))
                commands.append(f"hawking studio claim-bundle-build {model.label}")
            else:
                state = "claim-ready"
        else:
            state = "claim-blocked-parity"
            blockers.extend((parity_row or {}).get("parity", {}).get("problems", ["parity missing"]))
            commands.append("python3.12 tools/condense/frontier_parity.py status")
    else:
        state = "needs-operator-review"
        blockers.append("state combination is unusual; inspect ledger evidence")

    if artifact and receiptish and not parity_pass:
        if "parity" not in " ".join(blockers).lower():
            blockers.append("frontier parity is not passing")

    return {
        "label": model.label,
        "hf_id": model.hf_id,
        "state": state,
        "source_exists": source,
        "artifact_exists": artifact,
        "artifact_inventory_ok": inventory_ok,
        "receipt_or_record_exists": receiptish,
        "release_safe": ledger_row["release_safe"],
        "license_status": license_status,
        "download": download,
        "events": events,
        "parity_gate": (parity_row or {}).get("claim_gate", "BLOCK"),
        "parity_family": (parity_row or {}).get("family"),
        "source_provenance_gate": "ALLOW" if source_provenance_ok else "BLOCK",
        "baseline_coverage_gate": "ALLOW" if baseline_ok else "BLOCK",
        "eval_coverage_gate": "ALLOW" if eval_ok else "BLOCK",
        "serve_receipt_gate": "ALLOW" if serve_ok else "BLOCK",
        "ramcliff_receipt_gate": "ALLOW" if ramcliff_ok else "BLOCK",
        "doctor_recovery_gate": "ALLOW" if doctor_ok else "BLOCK",
        "experiment_gate": "ALLOW" if experiment_ok else "BLOCK",
        "claim_bundle_gate": "ALLOW" if claim_bundle_ok else "BLOCK",
        "blockers": blockers,
        "next_commands": commands,
    }


def build_lifecycle(root: pathlib.Path = ROOT, link_mb_s: float = 300.0, efficiency: float = 0.7,
                    scratch_gb: float = 200.0,
                    cache_reserve_gb: float = CACHE_RESERVE_GB,
                    storage_budget_gb: float | None = None,
                    max_wave_hours: float = 6.0) -> dict:
    ledger = build_ledger(root, link_mb_s=link_mb_s, efficiency=efficiency,
                          scratch_gb=scratch_gb, cache_reserve_gb=cache_reserve_gb,
                          storage_budget_gb=storage_budget_gb,
                          max_wave_hours=max_wave_hours)
    parity = _parity_row_by_label(root)
    by_label = {row["label"]: row for row in ledger["models"]}
    nodes = [
        _lifecycle_node(model, by_label[model.label], parity.get(model.label), root)
        for model in FRONTIER_MODELS
    ]
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node["state"]] = counts.get(node["state"], 0) + 1
    return {
        "schema": "hawking.frontier_lifecycle.v1",
        "generated_at": _now(),
        "root": str(root),
        "counts": counts,
        "storage_wave_plan": ledger["storage_wave_plan"],
        "nodes": nodes,
        "next_frontier_command": next((cmd for n in nodes for cmd in n["next_commands"]), None),
    }


def _select_lifecycle_node(lifecycle: dict, label: str | None = None) -> dict | None:
    nodes = lifecycle["nodes"]
    if label:
        needle = label.lower()
        return next((n for n in nodes if n["label"].lower() == needle or n["hf_id"].lower() == needle), None)
    actionable = [
        "ready-download",
        "download-stalled",
        "download-failed",
        "download-retry-pending",
        "ready-bake",
        "needs-artifact-inventory",
        "ready-release-source",
        "needs-license-review",
        "needs-receipt",
        "claim-blocked-parity",
        "claim-blocked-source-provenance",
        "claim-blocked-eval",
        "claim-blocked-baseline",
        "claim-blocked-serve",
        "claim-blocked-ramcliff",
        "claim-blocked-doctor",
        "claim-blocked-experiment",
        "claim-blocked-bundle",
    ]
    for state in actionable:
        found = next((n for n in nodes if n["state"] == state and n["next_commands"]), None)
        if found:
            return found
    return None


def _command_has_placeholder(cmd: str) -> bool:
    return "<" in cmd or ">" in cmd or cmd.startswith("write ")


def _run_frontier_command(cmd: str) -> int:
    argv = shlex.split(cmd)
    if not argv:
        return 2
    print(f"[frontier-ops] exec: {' '.join(argv)}", file=sys.stderr)
    return subprocess.call(argv)


def cmd_ledger(args) -> int:
    out = pathlib.Path(args.out) if args.out else ROOT / LEDGER_PATH
    data = build_ledger(ROOT, refresh_hf=args.refresh_hf, dry_run_sizes=args.dry_run_sizes,
                        include=args.include, timeout=args.timeout, link_mb_s=args.link_mbs,
                        efficiency=args.efficiency, scratch_gb=args.scratch_gb,
                        cache_reserve_gb=args.cache_reserve_gb,
                        storage_budget_gb=args.storage_budget_gb,
                        max_wave_hours=args.max_wave_hours)
    _write_json(out, data)
    print(f"[frontier-ops] wrote {out}", file=sys.stderr)
    print(f"# models={len(data['models'])} full-fit={data['manifest']['full_fit_gb']/1000:.1f} TB "
          f"cycle-peak={data['cycle_plan']['cycle_peak_gb']/1000:.1f} TB", file=sys.stderr)
    return 0


def cmd_status(args) -> int:
    data = build_ledger(ROOT, link_mb_s=args.link_mbs, efficiency=args.efficiency,
                        scratch_gb=args.scratch_gb,
                        cache_reserve_gb=args.cache_reserve_gb,
                        storage_budget_gb=args.storage_budget_gb,
                        max_wave_hours=args.max_wave_hours)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    hw = data["hardware"]
    cp = data["cycle_plan"]
    print(f"# frontier status {data['generated_at']}  root={data['root']}")
    cpu = hw.get("actual_cpu_brand") or "unknown CPU"
    ncpu = hw.get("actual_cpu_count") or "?"
    power = hw.get("power_source") or "unknown power"
    print(f"# host {cpu}  cores={ncpu}  ram={hw.get('actual_ram_gb')} GB  {power}")
    if hw.get("thermal_status"):
        thermal_line = hw["thermal_status"].splitlines()[0]
        print(f"# thermal {thermal_line}")
    print(f"# disk free {hw['disk_free_gb']:.0f} GB / {hw['disk_total_gb']:.0f} GB  "
          f"full-fit {data['manifest']['full_fit_gb']/1000:.1f} TB  "
          f"cycle-peak {cp['cycle_peak_gb']/1000:.1f} TB  "
          f"download ETA {fmt_hours(cp['download_eta_hours'])}")
    cache = data["cache"]
    swp = data["storage_wave_plan"]
    wave_block = f" blocked={len(swp['impossible_labels'])}" if swp["impossible_labels"] else ""
    print(f"# cache reserve {cache['reserve_gb']:.0f} GB  "
          f"HF cache sizes={cache['sizes_gb']}  "
          f"storage waves={swp['wave_count']} peak={swp['planned_peak_gb']/1000:.1f} TB{wave_block}")
    print("label              state            sourceGB  artifact  record  receipts  download-state     release  next")
    for m in data["models"]:
        receipt_count = len(m["official_receipts"])
        dl = m.get("download", {})
        if dl.get("count"):
            rate = dl.get("last_observed_mb_s")
            rc = dl.get("last_returncode")
            dl_txt = f"rc{rc}/{rate:.0f}MBs" if isinstance(rate, (int, float)) else f"rc{rc}"
            if dl.get("last_verify_returncode") is not None:
                dl_txt += f"/v{dl['last_verify_returncode']}"
            if dl.get("last_terminated_for_stall"):
                dl_txt += "/stall"
            if dl.get("last_will_retry"):
                dl_txt += "/retry"
            if dl.get("last_diagnostics_present"):
                dl_txt += "/diag"
        else:
            dl_txt = "none"
        next_step = "download"
        if m["source_dir_exists"] and not m["artifact_exists"]:
            next_step = "bake .tq"
        elif m["artifact_exists"] and not m.get("artifact_inventory", {}).get("ok"):
            next_step = "hash artifact"
        elif m["release_safe"]:
            next_step = "release source"
        elif m["artifact_exists"]:
            next_step = "write/verify receipt"
        elif m["state"] == "source-released":
            next_step = "kept output"
        print(f"{m['label'][:18]:18s} {m['state'][:15]:15s} {m['source_dir_gb']:8.1f}  "
              f"{'yes' if m['artifact_exists'] else 'no ':8s} "
              f"{'yes' if m['frontier_record_exists'] else 'no ':6s} "
              f"{receipt_count:8d}  "
              f"{dl_txt[:18]:18s} "
              f"{'safe' if m['release_safe'] else 'hold':7s} {next_step}")
    return 0


def cmd_worktree_plan(args) -> int:
    if args.verify:
        path = pathlib.Path(args.verify)
        doc = _read_json(path, {})
        status = _worktree_plan_status(doc)
        if args.json:
            print(json.dumps({"path": str(path), **status}, indent=2, sort_keys=True))
        else:
            verdict = "valid" if status["ok"] else "INVALID"
            print(f"[frontier-ops] worktree split plan {verdict}: {path}", file=sys.stderr)
            if status["ok"] and status["risk"] != "clean":
                print(f"[frontier-ops] signed worktree risk={status['risk']} "
                      f"entries={status['entries']} subsystems={status['subsystems']}", file=sys.stderr)
            for problem in status["problems"]:
                print(f"  - {problem}", file=sys.stderr)
        return 0 if status["ok"] else 1
    data = build_worktree_plan(ROOT)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
        print(f"[frontier-ops] wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0 if data["ok"] else 1
    _print_worktree_plan(data, max_paths=args.max_paths)
    return 0 if data["ok"] else 1


def cmd_storage_plan(args) -> int:
    budget = args.storage_budget_gb
    if budget is None:
        budget = _gb_from_bytes(shutil.disk_usage(ROOT).free)
    plan = manifest_storage_wave_plan(
        storage_budget_gb=budget,
        link_mb_s=args.link_mbs,
        efficiency=args.efficiency,
        scratch_gb=args.scratch_gb,
        cache_reserve_gb=args.cache_reserve_gb,
        max_wave_hours=args.max_wave_hours,
    )
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    print(f"# storage wave plan budget={plan['storage_budget_gb']:.0f} GB  "
          f"scratch={plan['scratch_gb']:.0f} GB  cache-reserve={plan['cache_reserve_gb']:.0f} GB")
    print(f"# waves={plan['wave_count']}  planned-peak={plan['planned_peak_gb']/1000:.1f} TB  "
          f"total-download={plan['total_source_gb']/1000:.1f} TB  "
          f"download ETA={fmt_hours(plan['download_eta_hours'])}")
    if plan["impossible_labels"]:
        print(f"# IMPOSSIBLE under this budget: {', '.join(plan['impossible_labels'])}")
    for wave in plan["waves"]:
        checkpoint = " CHECKPOINT+" if wave["exceeds_checkpoint_hours"] else ""
        print(f"wave {wave['wave']:02d}{checkpoint}: source={wave['source_gb']:.0f} GB  "
              f"artifacts={wave['artifact_gb']:.0f} GB  peak={wave['peak_gb']/1000:.1f} TB  "
              f"ETA={fmt_hours(wave['download_eta_hours'])}  "
              f"labels={', '.join(wave['labels'])}")
    return 0 if not plan["impossible_labels"] else 1


def cmd_lifecycle(args) -> int:
    data = build_lifecycle(ROOT, link_mb_s=args.link_mbs, efficiency=args.efficiency,
                           scratch_gb=args.scratch_gb,
                           cache_reserve_gb=args.cache_reserve_gb,
                           storage_budget_gb=args.storage_budget_gb,
                           max_wave_hours=args.max_wave_hours)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
        print(f"[frontier-ops] wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    counts = ", ".join(f"{k}={v}" for k, v in sorted(data["counts"].items()))
    print(f"# frontier lifecycle {data['generated_at']}  {counts}")
    if data.get("next_frontier_command"):
        print(f"# next: {data['next_frontier_command']}")
    print("label              state                  license   src  art  rec  parity  srcpv eval  base  serve ram   doc   exp   bund  next")
    for node in data["nodes"]:
        nxt = node["next_commands"][0] if node["next_commands"] else "-"
        print(f"{node['label'][:18]:18s} {node['state'][:22]:22s} "
              f"{node['license_status'][:8]:8s} "
              f"{'Y' if node['source_exists'] else 'n':3s} "
              f"{'Y' if node['artifact_exists'] else 'n':3s} "
              f"{'Y' if node['receipt_or_record_exists'] else 'n':3s} "
              f"{node['parity_gate'][:6]:6s} "
              f"{node['source_provenance_gate'][:5]:5s} "
              f"{node['eval_coverage_gate'][:5]:5s} "
              f"{node['baseline_coverage_gate'][:5]:5s} "
              f"{node['serve_receipt_gate'][:5]:5s} "
              f"{node['ramcliff_receipt_gate'][:5]:5s} "
              f"{node['doctor_recovery_gate'][:5]:5s} "
              f"{node['experiment_gate'][:5]:5s} "
              f"{node['claim_bundle_gate'][:5]:5s} {nxt[:72]}")
    return 0


def cmd_coverage_plan(args) -> int:
    labels = [m.label for m in FRONTIER_MODELS]
    if args.label:
        selected = []
        for label in args.label:
            model = frontier_by_label(label)
            if not model:
                print(f"[frontier-ops] unknown label {label}", file=sys.stderr)
                return 2
            selected.append(model.label)
        labels = selected
    data = frontier_coverage.coverage_plan(ROOT, labels)
    data["baseline_rollup"] = frontier_coverage.baseline_rollup(ROOT, labels)
    data["eval_rollup"] = frontier_coverage.eval_rollup(ROOT, labels)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
        print(f"[frontier-ops] wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    print("# frontier coverage plan")
    print("# baseline requirements: " + ", ".join(data["baseline_requirements"]))
    print("# eval requirements: " + ", ".join(data["eval_requirements"]))
    print(f"# baseline coverage {data['baseline_rollup']['passed_count']}/{data['baseline_rollup']['model_count']}  "
          f"eval coverage {data['eval_rollup']['passed_count']}/{data['eval_rollup']['model_count']}")
    for row in data["labels"]:
        print(f"{row['label']}:")
        print(f"  baseline: {row['baseline_path']}")
        print(f"  eval:     {row['eval_path']}")
    return 0


def cmd_coverage_receipt(args) -> int:
    return frontier_coverage_runner.dispatch(args, ROOT)


def cmd_source_provenance(args) -> int:
    return frontier_provenance.dispatch(args, ROOT)


def cmd_parity_receipt(args) -> int:
    return frontier_parity_runner.dispatch(args, ROOT)


def cmd_receipt_record(args) -> int:
    return frontier_receipt_runner.dispatch(args, ROOT)


def cmd_serve_capture(args) -> int:
    return frontier_serve_capture.capture(args, ROOT)


def cmd_experiment_receipt(args) -> int:
    return frontier_experiment_runner.dispatch(args, ROOT)


def cmd_doctor_recovery_receipt(args) -> int:
    return frontier_doctor_recovery.dispatch(args, ROOT)


def cmd_doctor_recovery_plan(args) -> int:
    labels = [m.label for m in FRONTIER_MODELS]
    if args.label:
        selected = []
        for label in args.label:
            model = frontier_by_label(label)
            if not model:
                print(f"[frontier-ops] unknown label {label}", file=sys.stderr)
                return 2
            selected.append(model.label)
        labels = selected
    data = frontier_doctor_recovery.recovery_plan(ROOT, labels)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
        print(f"[frontier-ops] wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0 if data["rollup"]["ok"] else 1
    print("# frontier Doctor recovery plan")
    print(f"# receipts {data['rollup']['passed_count']}/{data['rollup']['model_count']}")
    for row in data["labels"]:
        print(f"{row['label']}: {row['path']}")
    return 0 if data["rollup"]["ok"] else 1


def cmd_receipt_plan(args) -> int:
    labels = [m.label for m in FRONTIER_MODELS]
    if args.label:
        selected = []
        for label in args.label:
            model = frontier_by_label(label)
            if not model:
                print(f"[frontier-ops] unknown label {label}", file=sys.stderr)
                return 2
            selected.append(model.label)
        labels = selected
    data = frontier_receipts.receipt_plan(ROOT, labels)
    data["serve_rollup"] = frontier_receipts.serve_rollup(ROOT, labels)
    data["ramcliff_rollup"] = frontier_receipts.ramcliff_rollup(ROOT, labels)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
        print(f"[frontier-ops] wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    print("# frontier receipt plan")
    print(f"# serve receipts {data['serve_rollup']['passed_count']}/{data['serve_rollup']['model_count']}  "
          f"RAM-cliff receipts {data['ramcliff_rollup']['passed_count']}/{data['ramcliff_rollup']['model_count']}")
    for row in data["labels"]:
        print(f"{row['label']}:")
        print(f"  serve:    {row['serve_path']}")
        print(f"  ramcliff: {row['ramcliff_path']}")
    return 0


def cmd_experiment_plan(args) -> int:
    labels = [m.label for m in FRONTIER_MODELS]
    if args.label:
        selected = []
        for label in args.label:
            model = frontier_by_label(label)
            if not model:
                print(f"[frontier-ops] unknown label {label}", file=sys.stderr)
                return 2
            selected.append(model.label)
        labels = selected
    data = frontier_experiments.experiment_plan(ROOT, labels)
    data["experiment_rollup"] = frontier_experiments.experiment_rollup(ROOT, labels)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
        print(f"[frontier-ops] wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    print("# frontier experiment plan")
    print("# requirements: " + ", ".join(r["name"] for r in data["requirements"]))
    roll = data["experiment_rollup"]
    print(f"# experiment depth {roll['passed_count']}/{roll['model_count']}")
    for row in data["labels"]:
        print(f"{row['label']}: {row['matrix_path']}")
    return 0


def cmd_license_plan(args) -> int:
    labels = [m.label for m in FRONTIER_MODELS]
    if args.label:
        selected = []
        for label in args.label:
            model = frontier_by_label(label)
            if not model:
                print(f"[frontier-ops] unknown label {label}", file=sys.stderr)
                return 2
            selected.append(model.label)
        labels = selected
    data = frontier_licenses.license_plan(labels)
    data["license_rollup"] = frontier_licenses.license_rollup(_license_ledger(ROOT), labels)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
        print(f"[frontier-ops] wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    roll = data["license_rollup"]
    print("# frontier license plan")
    print(f"# accepted terms {roll['passed_count']}/{roll['model_count']}")
    for row in data["labels"]:
        print(f"{row['label']}: {row['command']}")
    return 0


def _selected_license_models(labels: list[str]) -> list[FrontierModel] | None:
    if not labels:
        return list(FRONTIER_MODELS)
    models = []
    for label in labels:
        model = frontier_by_label(label)
        if not model:
            print(f"[frontier-ops] unknown label {label}", file=sys.stderr)
            return None
        models.append(model)
    return models


def _license_record_from_decision(row: dict, model: FrontierModel) -> dict:
    return {
        "status": row.get("status", "unreviewed"),
        "by": row.get("by", ""),
        "note": row.get("note", ""),
        "ts": _now(),
        "hf_id": model.hf_id,
        "source_kind": model.source_kind,
        "license": row.get("license", ""),
        "terms_url": row.get("terms_url", ""),
        "terms_snapshot": row.get("terms_snapshot", ""),
        "allowed_use": row.get("allowed_use", ""),
        "redistribution": row.get("redistribution", ""),
        "source_policy": row.get("source_policy", ""),
        "source": "license-decisions",
    }


def _license_decision_row_status(row: dict) -> dict:
    label = row.get("label")
    model = frontier_by_label(label) if label else None
    structural = []
    apply_problems = []
    if not model:
        structural.append(f"{label or '<missing>'}: unknown frontier label")
        return {"structural": structural, "apply_problems": structural, "operator_required": True}
    if row.get("hf_id") and row.get("hf_id") != model.hf_id:
        structural.append(f"{label}: hf_id mismatch")
    if row.get("source_kind") and row.get("source_kind") != model.source_kind:
        structural.append(f"{label}: source_kind mismatch")
    status = row.get("status", "unreviewed")
    if status not in ("unreviewed", "reviewed", "accepted", "rejected"):
        structural.append(f"{label}: invalid status {status!r}")
    final = bool(row.get("final"))
    if not final:
        apply_problems.append(f"{label}: final operator confirmation missing")
    if status == "accepted":
        record = _license_record_from_decision(row, model)
        check = frontier_licenses.license_status(record, label)
        apply_problems.extend(f"{label}: {problem}" for problem in check.get("problems", []))
    elif status == "rejected":
        if not row.get("by"):
            apply_problems.append(f"{label}: rejected row needs by")
        if not row.get("note"):
            apply_problems.append(f"{label}: rejected row needs note")
    elif status == "reviewed":
        if not row.get("by"):
            apply_problems.append(f"{label}: reviewed row needs by")
        if not row.get("note"):
            apply_problems.append(f"{label}: reviewed row needs note")
    else:
        apply_problems.append(f"{label}: status must be accepted/rejected/reviewed before apply")
    operator_required = bool(apply_problems)
    return {
        "structural": structural,
        "apply_problems": structural + apply_problems,
        "operator_required": operator_required,
    }


def _normalize_license_decisions(doc: dict) -> dict:
    decisions = doc.get("decisions") if isinstance(doc.get("decisions"), list) else []
    apply_problems = []
    structural = []
    for row in decisions:
        status = _license_decision_row_status(row)
        row["operator_required"] = status["operator_required"]
        row["problems"] = status["apply_problems"]
        structural.extend(status["structural"])
        apply_problems.extend(status["apply_problems"])
    doc["decision_count"] = len(decisions)
    doc["operator_required_count"] = sum(1 for row in decisions if row.get("operator_required"))
    doc["applyable"] = not apply_problems
    doc["structural_problem_count"] = len(structural)
    return doc


def _build_license_decisions(args) -> dict:
    models = _selected_license_models(args.label)
    if models is None:
        raise SystemExit(2)
    ledger = _license_ledger(ROOT)
    rows = []
    for model in models:
        existing = ledger.get(model.label) if isinstance(ledger.get(model.label), dict) else {}
        row = {
            "label": model.label,
            "hf_id": model.hf_id,
            "source_kind": model.source_kind,
            "source_gb": model.download_gb,
            "status": existing.get("status") or args.status,
            "by": existing.get("by") or args.by,
            "license": existing.get("license") or args.license,
            "terms_url": existing.get("terms_url") or args.terms_url,
            "terms_snapshot": existing.get("terms_snapshot") or args.terms_snapshot,
            "allowed_use": existing.get("allowed_use") or args.allowed_use,
            "redistribution": existing.get("redistribution") or args.redistribution,
            "source_policy": existing.get("source_policy") or args.source_policy,
            "note": existing.get("note") or args.note,
            "final": bool(existing.get("status") == "accepted" and frontier_licenses.license_status(existing, model.label)["ok"])
            or bool(args.final),
            "existing_record": bool(existing),
            "command_template": (
                f"hawking studio record-license {model.label} --status accepted --by <name> "
                "--license <id> --terms-url <url> --allowed-use research --redistribution none "
                "--source-policy local-only-delete-after-bake --note <decision>"
            ),
        }
        rows.append(row)
    doc = {
        "schema": "hawking.frontier_license_decisions.v1",
        "generated_at": _now(),
        "root": str(ROOT),
        "git_commit": _git_commit(ROOT),
        "license_path": str(ROOT / LICENSE_PATH),
        "allowed_use": sorted(frontier_licenses.ALLOWED_USE),
        "redistribution": sorted(frontier_licenses.REDISTRIBUTION),
        "source_policy": sorted(frontier_licenses.SOURCE_POLICY),
        "decisions": rows,
        "note": (
            "Drafting/signing this workbook does not satisfy the launch gate. Only "
            "`license-decisions apply --confirm` writes frontier_license_acceptance.json, and accepted "
            "rows must include by, license, terms_url, allowed_use, redistribution, source_policy, and note."
        ),
    }
    return _sign_doc(_normalize_license_decisions(doc))


def _license_decisions_status(doc: dict) -> dict:
    problems = []
    if doc.get("schema") != "hawking.frontier_license_decisions.v1":
        problems.append("schema mismatch")
    signature_ok = _signature_ok(doc)
    if not signature_ok:
        problems.append("signature digest mismatch")
    decisions = doc.get("decisions")
    if not isinstance(decisions, list):
        problems.append("decisions must be a list")
        decisions = []
    actual_required = 0
    apply_problems = []
    structural = []
    for row in decisions:
        status = _license_decision_row_status(row)
        actual_required += 1 if status["operator_required"] else 0
        structural.extend(status["structural"])
        apply_problems.extend(status["apply_problems"])
    problems.extend(structural)
    if doc.get("decision_count") != len(decisions):
        problems.append("decision_count does not match decisions length")
    if doc.get("operator_required_count") != actual_required:
        problems.append("operator_required_count does not match decisions")
    if doc.get("applyable") != (not apply_problems):
        problems.append("applyable does not match decisions")
    return {
        "signature_ok": signature_ok,
        "schema_ok": doc.get("schema") == "hawking.frontier_license_decisions.v1",
        "ok": not problems,
        "applyable": not problems and not apply_problems,
        "problems": problems,
        "apply_problems": apply_problems,
        "decision_count": len(decisions),
        "operator_required_count": actual_required,
    }


def cmd_license_decisions(args) -> int:
    if args.mode == "draft":
        doc = _build_license_decisions(args)
        out = pathlib.Path(args.out)
        _write_json(out, doc)
        payload = {
            "ok": True,
            "path": str(out),
            "decision_count": doc["decision_count"],
            "operator_required_count": doc["operator_required_count"],
            "applyable": doc["applyable"],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"[frontier-ops] wrote signed license decisions draft {out} "
                f"({doc['decision_count']} rows, {doc['operator_required_count']} need operator)",
                file=sys.stderr,
            )
        return 0

    path = pathlib.Path(args.path)
    doc = _read_json(path, {})
    if args.mode == "sign":
        if doc.get("schema") != "hawking.frontier_license_decisions.v1":
            print(f"[frontier-ops] invalid license decisions workbook: {path}", file=sys.stderr)
            return 2
        doc.pop("signature", None)
        doc["signed_at"] = _now()
        doc["git_commit"] = _git_commit(ROOT)
        doc = _sign_doc(_normalize_license_decisions(doc))
        out = pathlib.Path(args.out or args.path)
        _write_json(out, doc)
        if args.json:
            print(json.dumps({
                "ok": True,
                "path": str(out),
                "decision_count": doc["decision_count"],
                "operator_required_count": doc["operator_required_count"],
                "applyable": doc["applyable"],
            }, indent=2, sort_keys=True))
        else:
            print(f"[frontier-ops] signed license decisions workbook {out}", file=sys.stderr)
        return 0

    status = _license_decisions_status(doc)
    if args.mode == "verify":
        if args.json:
            print(json.dumps({"path": str(path), **status}, indent=2, sort_keys=True))
        else:
            verdict = "valid" if status["ok"] else "INVALID"
            print(f"[frontier-ops] license decisions {verdict}: {path}", file=sys.stderr)
            for problem in status["problems"]:
                print(f"  - {problem}", file=sys.stderr)
        return 0 if status["ok"] else 1

    if args.mode != "apply":
        print(f"[frontier-ops] unknown license-decisions mode: {args.mode}", file=sys.stderr)
        return 2
    if not args.confirm:
        print("[frontier-ops] refusing to apply without --confirm", file=sys.stderr)
        return 2
    if not status["applyable"]:
        print("[frontier-ops] license decisions are not applyable:", file=sys.stderr)
        for problem in status["apply_problems"][:20]:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    ledger = _license_ledger(ROOT)
    decision_sha = _sha256_file(path) if path.exists() else None
    applied = 0
    for row in doc.get("decisions") or []:
        model = frontier_by_label(row["label"])
        record = _license_record_from_decision(row, model)
        record["license_decisions_path"] = str(path)
        record["license_decisions_sha256"] = decision_sha
        ledger[model.label] = record
        applied += 1
    _write_json(ROOT / LICENSE_PATH, ledger)
    if args.json:
        print(json.dumps({
            "ok": True,
            "applied_count": applied,
            "license_path": str(ROOT / LICENSE_PATH),
        }, indent=2, sort_keys=True))
    else:
        print(f"[frontier-ops] applied {applied} license decisions -> {ROOT / LICENSE_PATH}",
              file=sys.stderr)
    return 0


def _claim_models(labels: list[str]) -> list[FrontierModel]:
    if not labels:
        return list(FRONTIER_MODELS)
    models = []
    for label in labels:
        model = frontier_by_label(label)
        if not model:
            raise SystemExit(f"unknown frontier label: {label}")
        models.append(model)
    return models


def cmd_claim_bundle(args) -> int:
    if args.bundle_mode == "build":
        models = _claim_models(args.label)
        ok = True
        rows = []
        for model in models:
            bundle = frontier_claims.build_bundle(
                ROOT,
                model,
                require_ramcliff=not args.no_require_ramcliff,
            )
            out = pathlib.Path(args.out) if args.out and len(models) == 1 else frontier_claims.claim_bundle_path(ROOT, model.label)
            frontier_claims.write_bundle(out, bundle)
            rows.append({
                "label": model.label,
                "path": str(out),
                "claim_admissible": bundle["claim_admissible"],
                "blocker_count": len(bundle["blockers"]),
            })
            ok = ok and bundle["claim_admissible"]
            print(f"[frontier-ops] claim bundle {model.label} -> {out} "
                  f"admissible={bundle['claim_admissible']} blockers={len(bundle['blockers'])}",
                  file=sys.stderr)
        if args.json:
            print(json.dumps({"schema": "hawking.frontier_claim_bundle_build.v1",
                              "ok": ok, "rows": rows}, indent=2, sort_keys=True))
        return 0 if ok else 1

    paths = [pathlib.Path(p) for p in args.path]
    if not paths:
        paths = [frontier_claims.claim_bundle_path(ROOT, model.label) for model in FRONTIER_MODELS]
    rows = [frontier_claims.verify_bundle(path, ROOT) for path in paths]
    ok = all(row["ok"] for row in rows)
    if args.json:
        print(json.dumps({"schema": "hawking.frontier_claim_bundle_verify.v1",
                          "ok": ok, "rows": rows}, indent=2, sort_keys=True))
        return 0 if ok else 1
    print(f"# frontier claim bundles: {'OK' if ok else 'BLOCKED'}")
    for row in rows:
        label = row.get("label") or row["path"]
        print(f"{str(label)[:18]:18s} {'OK' if row['ok'] else 'BLOCK':6s} {row['path']}")
        for problem in row["problems"][:5]:
            print(f"  - {problem}")
    return 0 if ok else 1


def _claim_bundle_path_with_suffix(model: FrontierModel, suffix: str, root: pathlib.Path = ROOT) -> pathlib.Path:
    if suffix:
        safe = _safe_label(model.label)
        return root / COND_DIR / f"{safe}_claim_bundle{suffix}.json"
    return frontier_claims.claim_bundle_path(root, model.label)


def _preserve_final(path: pathlib.Path, force_final: bool) -> bool:
    record = _read_json(path, {})
    return bool(record and record.get("receipt_state") == "final" and not force_final)


def _write_draft_record(path: pathlib.Path, record: dict, *, force: bool,
                        force_final: bool) -> tuple[dict, bool]:
    existing = _read_json(path, {})
    if existing and existing.get("receipt_state") == "final" and not force_final:
        return {"path": str(path), "written": False, "ok": False,
                "problems": ["final receipt exists; preserved unless --force-final is set"]}, False
    if path.exists() and not force:
        return {"path": str(path), "written": False, "ok": False,
                "problems": ["path exists; use --force to overwrite draft/local evidence"]}, False
    _write_json(path, record)
    return {"path": str(path), "written": True, "ok": True, "problems": []}, True


def _proof_pack_models(labels: list[str]) -> list[FrontierModel]:
    if not labels:
        return list(FRONTIER_MODELS)
    models = []
    for label in labels:
        model = frontier_by_label(label)
        if not model:
            raise SystemExit(f"unknown frontier label: {label}")
        models.append(model)
    return models


def _draft_parity(model: FrontierModel, root: pathlib.Path, args) -> dict:
    path = frontier_parity_runner.parity_path(root, model.label)
    record = frontier_parity_runner.draft_record(model, machine_class=args.machine_class)
    record, status = frontier_parity_runner.sign_record(record, model=model, allow_blocked_draft=True)
    row, _ = _write_draft_record(path, record, force=args.force, force_final=args.force_final)
    row.update({"kind": "parity", "label": model.label, "receipt_ok": status["ok"],
                "receipt_problems": status["problems"]})
    return row


def _draft_coverage(model: FrontierModel, root: pathlib.Path, args, kind: str) -> dict:
    path = frontier_coverage_runner.default_path(root, model.label, kind)
    record = frontier_coverage_runner.draft_record(model.label, kind, machine_class=args.machine_class)
    record, status = frontier_coverage_runner.sign_record(record, kind=kind, allow_blocked_draft=True)
    row, _ = _write_draft_record(path, record, force=args.force, force_final=args.force_final)
    row.update({"kind": kind, "label": model.label, "receipt_ok": status["ok"],
                "receipt_problems": status["problems"]})
    return row


def _draft_source_provenance(model: FrontierModel, root: pathlib.Path, args) -> dict:
    path = frontier_provenance.provenance_path(root, model.label)
    record = frontier_provenance.draft_record(model, machine_class=args.machine_class)
    record, status = frontier_provenance.sign_record(record, model=model, allow_blocked_draft=True)
    row, _ = _write_draft_record(path, record, force=args.force, force_final=args.force_final)
    row.update({"kind": "source-provenance", "label": model.label, "receipt_ok": status["ok"],
                "receipt_problems": status["problems"]})
    return row


def _draft_native(model: FrontierModel, root: pathlib.Path, args, kind: str) -> dict:
    path = frontier_receipt_runner.default_path(root, model.label, kind)
    record = frontier_receipt_runner.draft_record(model.label, kind, machine_class=args.machine_class)
    record, status = frontier_receipt_runner.sign_record(record, kind=kind, allow_blocked_draft=True)
    row, _ = _write_draft_record(path, record, force=args.force, force_final=args.force_final)
    row.update({"kind": kind, "label": model.label, "receipt_ok": status["ok"],
                "receipt_problems": status["problems"]})
    return row


def _draft_doctor_recovery(model: FrontierModel, root: pathlib.Path, args) -> dict:
    path = frontier_doctor_recovery.recovery_path(root, model.label)
    record = frontier_doctor_recovery.draft_record(model, machine_class=args.machine_class)
    record, status = frontier_doctor_recovery.sign_record(record, model=model, allow_blocked_draft=True)
    row, _ = _write_draft_record(path, record, force=args.force, force_final=args.force_final)
    row.update({"kind": "doctor-recovery", "label": model.label, "receipt_ok": status["ok"],
                "receipt_problems": status["problems"]})
    return row


def _draft_experiment(model: FrontierModel, root: pathlib.Path, args) -> dict:
    path = frontier_experiments.matrix_path(root, model.label)
    record = frontier_experiment_runner.draft_record(model.label, machine_class=args.machine_class)
    record, status = frontier_experiment_runner.sign_record(record, label=model.label, allow_blocked_draft=True)
    row, _ = _write_draft_record(path, record, force=args.force, force_final=args.force_final)
    row.update({"kind": "experiment", "label": model.label, "receipt_ok": status["ok"],
                "receipt_problems": status["problems"]})
    return row


def build_proof_pack(root: pathlib.Path, args) -> dict:
    rows = []
    bundle_rows = []
    models = _proof_pack_models(args.label)
    for model in models:
        rows.append(_draft_source_provenance(model, root, args))
        rows.append(_draft_parity(model, root, args))
        for kind in ("baseline", "eval"):
            rows.append(_draft_coverage(model, root, args, kind))
        for kind in ("serve", "ramcliff"):
            rows.append(_draft_native(model, root, args, kind))
        rows.append(_draft_doctor_recovery(model, root, args))
        rows.append(_draft_experiment(model, root, args))
        bundle = frontier_claims.build_bundle(root, model, require_ramcliff=not args.no_require_ramcliff)
        bundle_path = _claim_bundle_path_with_suffix(model, args.claim_suffix, root)
        if bundle_path.exists() and not args.force:
            bundle_rows.append({
                "label": model.label,
                "path": str(bundle_path),
                "written": False,
                "claim_admissible": False,
                "blocker_count": len(bundle["blockers"]),
                "problems": ["claim bundle exists; use --force to overwrite local bundle"],
            })
        elif _read_json(bundle_path, {}).get("claim_admissible") is True and not args.force_final:
            bundle_rows.append({
                "label": model.label,
                "path": str(bundle_path),
                "written": False,
                "claim_admissible": False,
                "blocker_count": len(bundle["blockers"]),
                "problems": ["admissible claim bundle exists; preserved unless --force-final is set"],
            })
        else:
            frontier_claims.write_bundle(bundle_path, bundle)
            bundle_rows.append({
                "label": model.label,
                "path": str(bundle_path),
                "written": True,
                "claim_admissible": bundle["claim_admissible"],
                "blocker_count": len(bundle["blockers"]),
                "problems": [] if not bundle["claim_admissible"] else ["unexpected admissible local proof pack"],
            })
    blocked_bundles = [row["label"] for row in bundle_rows if not row["claim_admissible"]]
    result = {
        "schema": "hawking.frontier_proof_pack.v1",
        "generated_at": _now(),
        "root": str(root),
        "model_count": len(models),
        "evidence_rows": rows,
        "claim_bundles": bundle_rows,
        "ok": len(blocked_bundles) == len(models),
        "blocked_claim_count": len(blocked_bundles),
        "blocked_claim_labels": blocked_bundles,
        "note": "Proof-pack drafts are signed but intentionally claim-blocking until final measured evidence replaces TODO rows.",
    }
    if args.out:
        _write_json(pathlib.Path(args.out), result)
    return result


def cmd_proof_pack(args) -> int:
    data = build_proof_pack(ROOT, args)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"# frontier proof pack: blocked claims {data['blocked_claim_count']}/{data['model_count']}")
        for row in data["claim_bundles"]:
            print(f"{row['label'][:18]:18s} {'BLOCK' if not row['claim_admissible'] else 'ALLOW':6s} "
                  f"blockers={row['blocker_count']} {row['path']}")
            for problem in row["problems"][:3]:
                print(f"  - {problem}")
        if args.out:
            print(f"# wrote {args.out}")
    return 0 if data["ok"] else 1


def cmd_run_next(args) -> int:
    lifecycle = build_lifecycle(ROOT, link_mb_s=args.link_mbs, efficiency=args.efficiency,
                                scratch_gb=args.scratch_gb,
                                cache_reserve_gb=args.cache_reserve_gb,
                                storage_budget_gb=args.storage_budget_gb,
                                max_wave_hours=args.max_wave_hours)
    node = _select_lifecycle_node(lifecycle, args.label)
    if not node:
        print("[frontier-ops] no lifecycle node has a next command", file=sys.stderr)
        return 1
    if not node["next_commands"]:
        print(f"[frontier-ops] {node['label']} state={node['state']} has no executable next command",
              file=sys.stderr)
        return 1
    cmd = node["next_commands"][0]
    row = {"label": node["label"], "state": node["state"], "command": cmd, "blockers": node["blockers"]}
    if args.json:
        print(json.dumps(row, indent=2, sort_keys=True))
    else:
        print(f"# run-next label={node['label']} state={node['state']}")
        if node["blockers"]:
            print("# blockers: " + "; ".join(node["blockers"]))
        print(cmd)
    if not args.yes:
        return 0
    if _command_has_placeholder(cmd):
        print("[frontier-ops] REFUSE: next command contains a placeholder and needs a human edit",
              file=sys.stderr)
        return 2
    if node["state"] in ("needs-license-review", "needs-receipt", "claim-blocked-parity"):
        print(f"[frontier-ops] REFUSE: state {node['state']} is a human-proof gate, not an executor step",
              file=sys.stderr)
        return 2
    if node["state"] in ("claim-blocked-source-provenance",
                         "claim-blocked-eval", "claim-blocked-baseline",
                         "claim-blocked-serve", "claim-blocked-ramcliff",
                         "claim-blocked-doctor",
                         "claim-blocked-experiment"):
        print(f"[frontier-ops] REFUSE: state {node['state']} needs receipt work, not an executor step",
              file=sys.stderr)
        return 2
    if node["state"] in ("ready-download", "download-failed", "download-retry-pending"):
        if not args.allow_download:
            print("[frontier-ops] REFUSE: downloads require --allow-download", file=sys.stderr)
            return 2
        if not args.require_refresh:
            print("[frontier-ops] REFUSE: downloads require --require-refresh PATH", file=sys.stderr)
            return 2
        gate = build_launch_gate(ROOT, phase="procure", allow_unreviewed=False,
                                 link_mb_s=args.link_mbs, efficiency=args.efficiency,
                                 scratch_gb=args.scratch_gb,
                                 cache_reserve_gb=args.cache_reserve_gb,
                                 storage_budget_gb=args.storage_budget_gb,
                                 max_wave_hours=args.max_wave_hours,
                                 require_refresh=args.require_refresh)
        if not gate["ok"]:
            print("[frontier-ops] REFUSE: procurement launch gate is red", file=sys.stderr)
            for check in gate["checks"]:
                if not check["ok"] and check["severity"] == "fail":
                    print(f"  - {check['name']}: {check['detail']}", file=sys.stderr)
            return 1
    if node["state"] == "ready-bake" and not args.allow_heavy:
        print("[frontier-ops] REFUSE: bake/serve work requires --allow-heavy", file=sys.stderr)
        return 2
    if node["state"] == "ready-release-source" and not args.allow_release_dry_run:
        print("[frontier-ops] REFUSE: release checks require --allow-release-dry-run", file=sys.stderr)
        return 2
    return _run_frontier_command(cmd)


def cmd_artifact_inventory(args) -> int:
    model = frontier_by_label(args.label)
    if not model:
        print(f"[frontier-ops] unknown label {args.label}", file=sys.stderr)
        return 2
    artifacts = _artifact_candidates(model, ROOT)
    if not artifacts:
        print(f"[frontier-ops] no artifact candidates for {model.label}", file=sys.stderr)
        return 1
    rows = []
    for artifact in artifacts:
        try:
            st = artifact.stat()
        except OSError as e:
            print(f"[frontier-ops] cannot stat {artifact}: {e}", file=sys.stderr)
            return 1
        print(f"[frontier-ops] hashing {artifact} ({st.st_size / 1e9:.3f} GB)", file=sys.stderr)
        rows.append({
            "path": str(artifact),
            "bytes": st.st_size,
            "gb": round(_gb_from_bytes(st.st_size), 6),
            "sha256": _sha256_file(artifact),
        })
    out = {
        "schema": "hawking.frontier_artifact_inventory.v1",
        "generated_at": _now(),
        "label": model.label,
        "hf_id": model.hf_id,
        "git_commit": _git_commit(ROOT),
        "artifacts": rows,
    }
    path = _artifact_inventory_path(model, ROOT)
    _write_json(path, out)
    print(f"[frontier-ops] wrote artifact inventory -> {path}", file=sys.stderr)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_launch_gate(args) -> int:
    data = build_launch_gate(ROOT, phase=args.phase, allow_unreviewed=args.allow_unreviewed,
                             link_mb_s=args.link_mbs, efficiency=args.efficiency,
                             scratch_gb=args.scratch_gb,
                             cache_reserve_gb=args.cache_reserve_gb,
                             storage_budget_gb=args.storage_budget_gb,
                             max_wave_hours=args.max_wave_hours,
                             require_refresh=args.require_refresh)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
        print(f"[frontier-ops] wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"# launch gate phase={data['phase']} ok={data['ok']} "
              f"failures={data['failure_count']} warnings={data['warning_count']}")
        for check in data["checks"]:
            mark = "OK" if check["ok"] else check["severity"].upper()
            print(f"[{mark:4s}] {check['name']}: {check['detail']}")
    return 0 if data["ok"] else 1


def cmd_record_event(args) -> int:
    model = frontier_by_label(args.label)
    if not model:
        print(f"[frontier-ops] unknown label {args.label}", file=sys.stderr)
        return 2
    if args.duration_s is not None and args.duration_s < 0:
        print("[frontier-ops] --duration-s must be non-negative", file=sys.stderr)
        return 2
    row = {
        "schema": "hawking.frontier_event.v1",
        "ts": _now(),
        "label": model.label,
        "hf_id": model.hf_id,
        "stage": args.stage,
        "status": args.status,
        "duration_s": args.duration_s,
        "artifact": args.artifact,
        "note": args.note,
        "git_commit": _git_commit(ROOT),
    }
    _append_jsonl(ROOT / EVENT_LOG, row)
    print(f"[frontier-ops] event {model.label} stage={args.stage} status={args.status} -> {ROOT / EVENT_LOG}",
          file=sys.stderr)
    return 0


def cmd_release_source(args) -> int:
    model = frontier_by_label(args.label)
    if not model:
        print(f"[frontier-ops] unknown label {args.label}", file=sys.stderr)
        return 2
    ok, missing, evidence = release_guard(model, ROOT)
    source = pathlib.Path(evidence["source_dir"]).resolve()
    root = ROOT.resolve()
    if root not in source.parents:
        print(f"[frontier-ops] REFUSE: source {source} is outside root {root}", file=sys.stderr)
        return 2
    if "scratch" not in source.parts:
        print(f"[frontier-ops] REFUSE: source {source} is not under a scratch path", file=sys.stderr)
        return 2
    print(f"[frontier-ops] release-source {model.label}: {source} ({evidence['source_gb']:.3f} GB)",
          file=sys.stderr)
    if not ok:
        print("[frontier-ops] HOLD: release guard failed:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1
    event = {
        "ts": _now(),
        "label": model.label,
        "path": str(source),
        "gb": evidence["source_gb"],
        "dry_run": not args.yes,
        "git_commit": _git_commit(ROOT),
        "evidence": evidence,
    }
    if args.yes:
        shutil.rmtree(source)
        event["deleted"] = True
        _append_jsonl(ROOT / RELEASE_LOG, event)
        print(f"[frontier-ops] deleted {source}; event logged to {ROOT / RELEASE_LOG}", file=sys.stderr)
    else:
        event["deleted"] = False
        print("[frontier-ops] DRY RUN: release would be allowed. Re-run with --yes to delete.", file=sys.stderr)
        print(json.dumps(event, indent=2, sort_keys=True))
    return 0


def build_wave0_packet(root: pathlib.Path, args) -> dict:
    refresh_path = pathlib.Path(args.require_refresh)
    worktree_plan_path = pathlib.Path(getattr(args, "worktree_plan", WORKTREE_PLAN_PATH))
    runtime_contract_path = pathlib.Path(getattr(args, "runtime_contract", RUNTIME_CONTRACT_PATH))
    storage_plan = manifest_storage_wave_plan(
        storage_budget_gb=args.storage_budget_gb,
        link_mb_s=args.link_mbs,
        efficiency=args.efficiency,
        scratch_gb=args.scratch_gb,
        cache_reserve_gb=args.cache_reserve_gb,
        max_wave_hours=args.max_wave_hours,
    )
    lifecycle = build_lifecycle(
        root,
        link_mb_s=args.link_mbs,
        efficiency=args.efficiency,
        scratch_gb=args.scratch_gb,
        cache_reserve_gb=args.cache_reserve_gb,
        storage_budget_gb=args.storage_budget_gb,
        max_wave_hours=args.max_wave_hours,
    )
    selected = _select_lifecycle_node(lifecycle, args.label)
    run_next = None
    if selected:
        command = selected["next_commands"][0] if selected.get("next_commands") else None
        run_next = {
            "label": selected["label"],
            "state": selected["state"],
            "command": command,
            "blockers": selected.get("blockers", []),
            "contains_placeholder": _command_has_placeholder(command or ""),
            "download_candidate": selected["state"] in (
                "ready-download", "download-failed", "download-retry-pending"
            ),
        }
    gate = build_launch_gate(
        root,
        phase="procure",
        allow_unreviewed=False,
        link_mb_s=args.link_mbs,
        efficiency=args.efficiency,
        scratch_gb=args.scratch_gb,
        cache_reserve_gb=args.cache_reserve_gb,
        storage_budget_gb=args.storage_budget_gb,
        max_wave_hours=args.max_wave_hours,
        require_refresh=str(refresh_path),
    )
    artifacts = {
        "preflight_summary": _json_artifact(PREFLIGHT_SUMMARY_PATH, root),
        "environment": _json_artifact(STUDIO_ENVIRONMENT_PATH, root),
        "refresh": _json_artifact(refresh_path, root),
        "license_decisions": _json_artifact(pathlib.Path(args.license_decisions), root),
        "review_decisions": _json_artifact(pathlib.Path(args.review_decisions), root),
        "worktree_plan": _json_artifact(worktree_plan_path, root),
        "runtime_contract": _json_artifact(runtime_contract_path, root),
        "proof_pack": _json_artifact(pathlib.Path(args.proof_pack), root),
    }
    license_rollup = frontier_licenses.license_rollup(_license_ledger(root), [m.label for m in FRONTIER_MODELS])
    refresh = _read_json(refresh_path if refresh_path.is_absolute() else root / refresh_path, {})
    review = refresh_review_status(refresh, root) if refresh.get("schema") == "hawking.frontier_refresh.v1" else {
        "review_worthy_count": 0,
        "reviewed": {},
        "missing": [],
    }

    checks = []

    def add(name: str, ok: bool, detail: str, evidence=None, severity: str = "fail") -> None:
        checks.append({
            "name": name,
            "ok": bool(ok),
            "severity": severity,
            "detail": detail,
            "evidence": evidence,
        })

    pre = artifacts["preflight_summary"]
    add("preflight-summary", pre.get("signature_ok") is True and pre.get("ok") is True,
        "signed Studio preflight summary is green", pre)
    env = artifacts["environment"]
    add("environment", env.get("signature_ok") is True and env.get("ok") is True,
        "signed Studio environment receipt is green", env)
    add("license-decisions-workbook", artifacts["license_decisions"].get("signature_ok") is True,
        "signed license decisions workbook verifies", artifacts["license_decisions"], "warn")
    add("review-decisions-workbook", artifacts["review_decisions"].get("signature_ok") is True,
        "signed refresh-review decisions workbook verifies", artifacts["review_decisions"], "warn")
    worktree_artifact = artifacts["worktree_plan"]
    add("worktree-split-plan",
        worktree_artifact.get("signature_ok") is True and worktree_artifact.get("ok") is True,
        "signed worktree split plan exists and verifies", worktree_artifact)
    add("worktree-risk",
        worktree_artifact.get("risk") in ("clean", "low", "medium"),
        f"worktree risk={worktree_artifact.get('risk')} entries={worktree_artifact.get('entries')}",
        worktree_artifact,
        "warn")
    runtime_artifact = artifacts["runtime_contract"]
    add("runtime-contract",
        runtime_artifact.get("signature_ok") is True
        and runtime_artifact.get("ok") is True
        and (runtime_artifact.get("profile_count") or 0) >= 5
        and (runtime_artifact.get("proof_mode_required") or 0) >= 4,
        (
            f"runtime contract profiles={runtime_artifact.get('profile_count')} "
            f"proof_env={runtime_artifact.get('proof_mode_required')}"
        ),
        runtime_artifact)
    add("license-approval", license_rollup["ok"],
        f"{license_rollup['passed_count']}/{license_rollup['model_count']} model licenses have accepted terms",
        license_rollup)
    add("refresh-candidate-review", not review["missing"],
        f"{len(review['missing'])}/{review['review_worthy_count']} refresh candidates need decisions",
        review)
    add("storage-wave-plan", not storage_plan.get("impossible_labels"),
        f"{storage_plan['wave_count']} wave(s), planned peak {storage_plan['planned_peak_gb']:.0f} GB",
        storage_plan)
    add("proof-pack", artifacts["proof_pack"].get("ok") is True,
        "local proof pack exists and summarizes signed draft wall", artifacts["proof_pack"], "warn")
    add("procurement-launch-gate", gate["ok"],
        f"procurement gate failure_count={gate['failure_count']}", gate)
    add("run-next-dry-run", bool(run_next and run_next.get("command")),
        "run-next selected a dry-run command", run_next, "warn")

    failures = [c for c in checks if not c["ok"] and c["severity"] == "fail"]
    warnings = [c for c in checks if not c["ok"] and c["severity"] == "warn"]
    packet = {
        "schema": "hawking.studio_wave0_launch_packet.v1",
        "generated_at": _now(),
        "root": str(root),
        "git_commit": _git_commit(root),
        "parameters": {
            "storage_budget_gb": args.storage_budget_gb,
            "link_mbs": args.link_mbs,
            "efficiency": args.efficiency,
            "scratch_gb": args.scratch_gb,
            "cache_reserve_gb": args.cache_reserve_gb,
            "max_wave_hours": args.max_wave_hours,
            "require_refresh": str(refresh_path),
            "worktree_plan": str(worktree_plan_path),
            "runtime_contract": str(runtime_contract_path),
        },
        "ok": not failures,
        "procurement_permitted": bool(gate["ok"] and not failures),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "checks": checks,
        "artifacts": artifacts,
        "storage_plan_summary": {
            "wave_count": storage_plan.get("wave_count"),
            "planned_peak_gb": storage_plan.get("planned_peak_gb"),
            "total_source_gb": storage_plan.get("total_source_gb"),
            "download_eta_hours": storage_plan.get("download_eta_hours"),
            "impossible_labels": storage_plan.get("impossible_labels"),
        },
        "lifecycle_summary": {
            "counts": lifecycle.get("counts"),
            "next_frontier_command": lifecycle.get("next_frontier_command"),
        },
        "run_next_dry_run": run_next,
        "procurement_gate": gate,
    }
    return _sign_doc(packet)


def _wave0_packet_status(doc: dict) -> dict:
    problems = []
    if doc.get("schema") != "hawking.studio_wave0_launch_packet.v1":
        problems.append("schema mismatch")
    signature_ok = _signature_ok(doc)
    if not signature_ok:
        problems.append("signature digest mismatch")
    if not isinstance(doc.get("checks"), list):
        problems.append("checks must be a list")
    return {
        "schema_ok": doc.get("schema") == "hawking.studio_wave0_launch_packet.v1",
        "signature_ok": signature_ok,
        "packet_ok": bool(doc.get("ok")),
        "procurement_permitted": bool(doc.get("procurement_permitted")),
        "failure_count": doc.get("failure_count"),
        "warning_count": doc.get("warning_count"),
        "ok": not problems,
        "problems": problems,
    }


def cmd_launch_packet(args) -> int:
    if args.mode == "build":
        packet = build_wave0_packet(ROOT, args)
        out = pathlib.Path(args.out)
        _write_json(out, packet)
        if args.json:
            print(json.dumps({
                "ok": packet["ok"],
                "procurement_permitted": packet["procurement_permitted"],
                "path": str(out),
                "failure_count": packet["failure_count"],
                "warning_count": packet["warning_count"],
            }, indent=2, sort_keys=True))
        else:
            verdict = "GREEN" if packet["ok"] else "RED"
            print(f"[frontier-ops] wrote Studio wave-0 launch packet {out} ({verdict})", file=sys.stderr)
        return 0 if packet["ok"] else 1
    if args.mode != "verify":
        print(f"[frontier-ops] unknown launch-packet mode: {args.mode}", file=sys.stderr)
        return 2
    path = pathlib.Path(args.path)
    doc = _read_json(path, {})
    status = _wave0_packet_status(doc)
    if args.json:
        print(json.dumps({"path": str(path), **status}, indent=2, sort_keys=True))
    else:
        verdict = "valid" if status["ok"] else "INVALID"
        print(f"[frontier-ops] Studio wave-0 launch packet {verdict}: {path}", file=sys.stderr)
        if status["ok"] and not status["packet_ok"]:
            print("[frontier-ops] packet signature is valid but launch readiness is RED", file=sys.stderr)
        for problem in status["problems"]:
            print(f"  - {problem}", file=sys.stderr)
    return 0 if status["ok"] else 1


def cmd_audit_grade(args) -> int:
    if args.mode == "build":
        doc = build_audit_grade(ROOT, args)
        out = pathlib.Path(args.out)
        _write_json(out, doc)
        if args.json:
            print(json.dumps({
                "ok": doc["ok"],
                "path": str(out),
                "target_grade": doc["target_grade"],
                "target_reached": doc["target_reached"],
                "frontier_claims_walled": doc["frontier_claims_walled"],
                "facet_count": doc["facet_count"],
                "below_target_count": doc["below_target_count"],
            }, indent=2, sort_keys=True))
        else:
            verdict = "TARGET" if doc["target_reached"] else "NOT-YET"
            print(f"[frontier-ops] wrote Studio audit-grade receipt {out} ({verdict})", file=sys.stderr)
        return 0
    if args.mode != "verify":
        print(f"[frontier-ops] unknown audit-grade mode: {args.mode}", file=sys.stderr)
        return 2
    path = pathlib.Path(args.path)
    doc = _read_json(path, {})
    status = _audit_grade_status(doc)
    if args.json:
        print(json.dumps({"path": str(path), **status}, indent=2, sort_keys=True))
    else:
        verdict = "valid" if status["ok"] else "INVALID"
        print(f"[frontier-ops] Studio audit-grade receipt {verdict}: {path}", file=sys.stderr)
        if status["ok"] and not status["target_reached"]:
            print("[frontier-ops] receipt signature is valid but audit target is not proven", file=sys.stderr)
        for problem in status["problems"]:
            print(f"  - {problem}", file=sys.stderr)
    return 0 if status["ok"] else 1


def cmd_completion_audit(args) -> int:
    if args.mode == "build":
        try:
            doc = build_completion_audit(ROOT, args)
        except ValueError as e:
            print(f"[frontier-ops] {e}", file=sys.stderr)
            return 2
        out = pathlib.Path(args.out)
        _write_json(out, doc)
        if args.json:
            print(json.dumps({
                "ok": doc["ok"],
                "completion_ok": doc["completion_ok"],
                "path": str(out),
                "required_count": doc["required_count"],
                "passed_count": doc["passed_count"],
                "blocked_count": doc["blocked_count"],
                "blocked_requirements": doc["blocked_requirements"],
            }, indent=2, sort_keys=True))
        else:
            verdict = "GREEN" if doc["completion_ok"] else "RED"
            print(
                f"[frontier-ops] wrote Studio completion audit {out} "
                f"({verdict}; blocked={doc['blocked_count']})",
                file=sys.stderr,
            )
        return 0
    if args.mode != "verify":
        print(f"[frontier-ops] unknown completion-audit mode: {args.mode}", file=sys.stderr)
        return 2
    path = pathlib.Path(args.path)
    doc = _read_json(path, {})
    status = _completion_audit_status(doc)
    if args.json:
        print(json.dumps({"path": str(path), **status}, indent=2, sort_keys=True))
    else:
        verdict = "valid" if status["ok"] else "INVALID"
        print(f"[frontier-ops] Studio completion audit {verdict}: {path}", file=sys.stderr)
        if status["ok"] and not status["completion_ok"]:
            print("[frontier-ops] completion audit signature is valid but Studio 10/10 is not proven",
                  file=sys.stderr)
        for problem in status["problems"]:
            print(f"  - {problem}", file=sys.stderr)
    return 0 if status["ok"] else 1


def cmd_record_license(args) -> int:
    model = frontier_by_label(args.label)
    if not model:
        print(f"[frontier-ops] unknown label {args.label}", file=sys.stderr)
        return 2
    if args.status == "accepted":
        missing = []
        for attr, flag in (
            ("by", "--by"),
            ("license", "--license"),
            ("terms_url", "--terms-url"),
            ("allowed_use", "--allowed-use"),
            ("redistribution", "--redistribution"),
            ("source_policy", "--source-policy"),
            ("note", "--note"),
        ):
            if not getattr(args, attr):
                missing.append(flag)
        if missing:
            print("[frontier-ops] accepted license terms require " + ", ".join(missing), file=sys.stderr)
            return 2
    path = ROOT / LICENSE_PATH
    data = _license_ledger(ROOT)
    row = {
        "status": args.status,
        "by": args.by,
        "note": args.note,
        "ts": _now(),
        "hf_id": model.hf_id,
        "source_kind": model.source_kind,
        "license": args.license,
        "terms_url": args.terms_url,
        "terms_snapshot": args.terms_snapshot,
        "allowed_use": args.allowed_use,
        "redistribution": args.redistribution,
        "source_policy": args.source_policy,
    }
    check = frontier_licenses.license_status(row, model.label)
    if args.status == "accepted" and not check["ok"]:
        print("[frontier-ops] accepted license record is incomplete:", file=sys.stderr)
        for problem in check["problems"]:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    data[model.label] = row
    _write_json(path, data)
    print(f"[frontier-ops] recorded {model.label} license status={args.status} -> {path}",
          file=sys.stderr)
    return 0


def cmd_review_candidate(args) -> int:
    path = ROOT / REFRESH_REVIEW_PATH
    data = _refresh_review_ledger(ROOT)
    data[args.model_id] = {
        "decision": args.decision,
        "by": args.by,
        "note": args.note,
        "ts": _now(),
        "git_commit": _git_commit(ROOT),
    }
    _write_json(path, data)
    print(f"[frontier-ops] reviewed refresh candidate {args.model_id} decision={args.decision} -> {path}",
          file=sys.stderr)
    return 0


def cmd_review_plan(args) -> int:
    path = pathlib.Path(args.refresh)
    refresh = _read_json(path, {})
    if refresh.get("schema") != "hawking.frontier_refresh.v1":
        print(f"[frontier-ops] invalid refresh ledger: {path}", file=sys.stderr)
        return 2
    status = refresh_review_status(refresh, ROOT)
    reviews = _refresh_review_ledger(ROOT)
    rows = []
    for row in refresh.get("candidates") or []:
        if not _is_review_worthy_candidate(row):
            continue
        mid = row.get("modelId")
        review = reviews.get(mid, {})
        rows.append({
            "model_id": mid,
            "source": row.get("source"),
            "url": row.get("url"),
            "last_modified": row.get("lastModified"),
            "downloads": row.get("downloads"),
            "likes": row.get("likes"),
            "pipeline_tag": row.get("pipeline_tag"),
            "tags": row.get("tags") or [],
            "decision": review.get("decision"),
            "reviewed": bool(review.get("decision") in ("accept", "reject", "watch")),
            "command_template": (
                f"hawking studio review-candidate {mid} "
                "--decision watch --by <name> --note <why>"
            ),
        })
    data = {
        "schema": "hawking.frontier_review_plan.v1",
        "generated_at": _now(),
        "refresh_path": str(path),
        "review_path": status["review_path"],
        "review_worthy_count": status["review_worthy_count"],
        "reviewed_count": len(status["reviewed"]),
        "missing_count": len(status["missing"]),
        "rows": rows,
        "note": "Human decisions only: accept enrolls a candidate for a future manifest change, reject excludes it, watch records awareness without procurement.",
    }
    if args.out:
        _write_json(pathlib.Path(args.out), data)
        print(f"[frontier-ops] wrote {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    print("# frontier candidate review plan")
    print(f"# refresh={path}")
    print(f"# review-worthy {data['review_worthy_count']}  reviewed {data['reviewed_count']}  missing {data['missing_count']}")
    for row in rows:
        tag = "OK" if row["reviewed"] else "NEEDS-DECISION"
        print(f"{tag:14s} {row['model_id']}  source={row['source']}  {row['url']}")
        if not row["reviewed"]:
            print(f"  {row['command_template']}")
    return 0 if data["missing_count"] == 0 else 1


def _build_review_decisions(refresh_path: pathlib.Path, refresh: dict, args) -> dict:
    reviews = _refresh_review_ledger(ROOT)
    refresh_sha = _sha256_file(refresh_path) if refresh_path.exists() else None
    decisions = []
    for row in refresh.get("candidates") or []:
        if not _is_review_worthy_candidate(row):
            continue
        mid = row.get("modelId")
        existing = reviews.get(mid) if isinstance(reviews.get(mid), dict) else {}
        decision = existing.get("decision") or args.decision
        by = existing.get("by") or args.by
        note = existing.get("note") or args.note
        final = bool(existing.get("decision") in ("accept", "reject", "watch")) or bool(args.final)
        decision_row = {
            **_candidate_metadata(row),
            "candidate_sha256": _candidate_digest(row),
            "reason": _candidate_review_reason(row),
            "decision": decision,
            "by": by,
            "note": note,
            "final": final,
            "existing_review": bool(existing),
        }
        decision_row["operator_required"] = not (
            final and decision in ("accept", "reject", "watch") and bool(by) and bool(note)
        )
        decisions.append(decision_row)
    operator_required = [row for row in decisions if row["operator_required"]]
    doc = {
        "schema": "hawking.frontier_review_decisions.v1",
        "generated_at": _now(),
        "root": str(ROOT),
        "git_commit": _git_commit(ROOT),
        "refresh_path": str(refresh_path),
        "refresh_sha256": refresh_sha,
        "review_path": str(ROOT / REFRESH_REVIEW_PATH),
        "default_decision": args.decision,
        "decision_count": len(decisions),
        "operator_required_count": len(operator_required),
        "applyable": len(operator_required) == 0,
        "decisions": decisions,
        "note": (
            "Drafting this workbook does not satisfy the launch gate. Only `review-decisions apply "
            "--confirm` writes accepted/rejected/watched rows to frontier_refresh_reviews.json."
        ),
    }
    return _sign_doc(doc)


def _review_decisions_status(doc: dict) -> dict:
    problems = []
    if doc.get("schema") != "hawking.frontier_review_decisions.v1":
        problems.append("schema mismatch")
    if not _signature_ok(doc):
        problems.append("signature digest mismatch")
    decisions = doc.get("decisions")
    if not isinstance(decisions, list):
        problems.append("decisions must be a list")
        decisions = []
    actual_operator_required = sum(1 for row in decisions if row.get("operator_required"))
    if doc.get("decision_count") != len(decisions):
        problems.append("decision_count does not match decisions length")
    if doc.get("operator_required_count") != actual_operator_required:
        problems.append("operator_required_count does not match decisions")
    if doc.get("applyable") != (actual_operator_required == 0):
        problems.append("applyable does not match decisions")
    for idx, row in enumerate(decisions):
        prefix = f"decisions[{idx}]"
        if row.get("decision") not in ("accept", "reject", "watch"):
            problems.append(f"{prefix} has invalid decision")
        if not row.get("model_id"):
            problems.append(f"{prefix} is missing model_id")
        if not row.get("candidate_sha256"):
            problems.append(f"{prefix} is missing candidate_sha256")
    apply_problems = list(problems)
    for row in decisions:
        if row.get("operator_required"):
            apply_problems.append(f"{row.get('model_id')}: operator confirmation missing")
    return {
        "signature_ok": not any("signature" in p for p in problems) and _signature_ok(doc),
        "schema_ok": doc.get("schema") == "hawking.frontier_review_decisions.v1",
        "ok": not problems,
        "applyable": not apply_problems,
        "problems": problems,
        "apply_problems": apply_problems,
        "decision_count": len(decisions),
        "operator_required_count": doc.get("operator_required_count"),
    }


def cmd_review_decisions(args) -> int:
    if args.mode == "draft":
        refresh_path = pathlib.Path(args.refresh)
        refresh = _read_json(refresh_path, {})
        if refresh.get("schema") != "hawking.frontier_refresh.v1":
            print(f"[frontier-ops] invalid refresh ledger: {refresh_path}", file=sys.stderr)
            return 2
        doc = _build_review_decisions(refresh_path, refresh, args)
        out = pathlib.Path(args.out)
        _write_json(out, doc)
        if args.json:
            print(json.dumps({
                "ok": True,
                "path": str(out),
                "decision_count": doc["decision_count"],
                "operator_required_count": doc["operator_required_count"],
                "applyable": doc["applyable"],
            }, indent=2, sort_keys=True))
        else:
            print(
                f"[frontier-ops] wrote signed review decisions draft {out} "
                f"({doc['decision_count']} rows, {doc['operator_required_count']} need operator)",
                file=sys.stderr,
            )
        return 0

    path = pathlib.Path(args.path)
    doc = _read_json(path, {})
    status = _review_decisions_status(doc)
    if args.mode == "verify":
        if args.json:
            print(json.dumps({"path": str(path), **status}, indent=2, sort_keys=True))
        else:
            verdict = "valid" if status["ok"] else "INVALID"
            print(f"[frontier-ops] review decisions {verdict}: {path}", file=sys.stderr)
            for problem in status["problems"]:
                print(f"  - {problem}", file=sys.stderr)
        return 0 if status["ok"] else 1

    if args.mode != "apply":
        print(f"[frontier-ops] unknown review-decisions mode: {args.mode}", file=sys.stderr)
        return 2
    if not args.confirm:
        print("[frontier-ops] refusing to apply without --confirm", file=sys.stderr)
        return 2
    if not status["applyable"]:
        print("[frontier-ops] review decisions are not applyable:", file=sys.stderr)
        for problem in status["apply_problems"][:20]:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    reviews = _refresh_review_ledger(ROOT)
    applied = 0
    decision_sha = _sha256_file(path) if path.exists() else None
    for row in doc.get("decisions") or []:
        mid = row["model_id"]
        reviews[mid] = {
            "decision": row["decision"],
            "by": row["by"],
            "note": row["note"],
            "ts": _now(),
            "git_commit": _git_commit(ROOT),
            "source": "review-decisions",
            "review_decisions_path": str(path),
            "review_decisions_sha256": decision_sha,
            "refresh_path": doc.get("refresh_path"),
            "refresh_sha256": doc.get("refresh_sha256"),
            "candidate_sha256": row.get("candidate_sha256"),
            "url": row.get("url"),
            "last_modified": row.get("last_modified"),
        }
        applied += 1
    _write_json(ROOT / REFRESH_REVIEW_PATH, reviews)
    if args.json:
        print(json.dumps({
            "ok": True,
            "applied_count": applied,
            "review_path": str(ROOT / REFRESH_REVIEW_PATH),
        }, indent=2, sort_keys=True))
    else:
        print(f"[frontier-ops] applied {applied} review decisions -> {ROOT / REFRESH_REVIEW_PATH}",
              file=sys.stderr)
    return 0


def _hf_model_rows(url: str, source: str, timeout: int) -> list[dict]:
    known = {m.hf_id for m in FRONTIER_MODELS}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        return [{"source": source, "url": url, "ok": False, "error": f"{type(e).__name__}: {e}"}]
    rows = []
    for item in data:
        mid = item.get("modelId") or item.get("id")
        rows.append({
            "source": source,
            "ok": True,
            "modelId": mid,
            "known": mid in known,
            "lastModified": item.get("lastModified"),
            "downloads": item.get("downloads"),
            "likes": item.get("likes"),
            "pipeline_tag": item.get("pipeline_tag"),
            "tags": (item.get("tags") or [])[:30],
            "url": f"https://huggingface.co/{mid}" if mid else None,
        })
    return rows


def _hf_search(term: str, limit: int, timeout: int) -> list[dict]:
    qs = urllib.parse.urlencode({"search": term, "sort": "lastModified", "direction": "-1", "limit": limit})
    url = f"https://huggingface.co/api/models?{qs}"
    return _hf_model_rows(url, f"search:{term}", timeout)


def _hf_author(author: str, limit: int, timeout: int) -> list[dict]:
    qs = urllib.parse.urlencode({"author": author, "sort": "lastModified", "direction": "-1", "limit": limit})
    url = f"https://huggingface.co/api/models?{qs}"
    return _hf_model_rows(url, f"author:{author}", timeout)


def cmd_refresh(args) -> int:
    rows = []
    for author in args.author:
        rows.extend(_hf_author(author, args.limit, args.timeout))
    for term in args.search:
        rows.extend(_hf_search(term, args.limit, args.timeout))
    out = {
        "schema": "hawking.frontier_refresh.v1",
        "generated_at": _now(),
        "author": args.author,
        "search": args.search,
        "known_manifest": [m.hf_id for m in FRONTIER_MODELS],
        "candidates": rows,
        "note": "Review unknown high-parameter/MoE candidates before a Studio launch; author results are higher signal than broad search. This does not auto-enroll models.",
    }
    path = pathlib.Path(args.out) if args.out else ROOT / REFRESH_PATH
    _write_json(path, out)
    print(f"[frontier-ops] wrote {path}", file=sys.stderr)
    review = refresh_review_status(out, ROOT)
    for row in rows:
        if row.get("ok") and not row.get("known"):
            tag = "REVIEW" if _is_review_worthy_candidate(row) else "seen"
            print(f"  review {row.get('modelId')}  source={row.get('source')}  "
                  f"lastModified={row.get('lastModified')}  {row.get('url')}  [{tag}]",
                  file=sys.stderr)
    if review["missing"]:
        print("[frontier-ops] review-worthy unknown candidates need decisions:", file=sys.stderr)
        for mid in review["missing"]:
            print(f"  hawking studio review-candidate {mid} "
                  "--decision watch --by <name> --note <why>", file=sys.stderr)
    return 0


def cmd_selftest(args) -> int:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        model = FRONTIER_MODELS[0]
        src = root / model.local_dir
        src.mkdir(parents=True)
        (src / "shard.safetensors").write_bytes(b"x" * 1024)
        tq = root / f"{model.local_dir}.tq"
        tq.write_bytes(b"tq" * 1024)
        rec = root / "reports" / "condense" / f"{model.label}_frontier.json"
        rec.parent.mkdir(parents=True)
        rec.write_text(json.dumps({"model": model.label, "artifact_gb": model.artifact_gb()}))
        _write_json(root / LICENSE_PATH, {
            model.label: {
                "status": "accepted",
                "by": "selftest",
                "note": "synthetic accepted license",
                "license": "selftest-license",
                "terms_url": "https://example.invalid/terms",
                "allowed_use": "research",
                "redistribution": "none",
                "source_policy": "local-only-delete-after-bake",
            }
        })
        _append_jsonl(root / DOWNLOAD_LOG, {
            "schema": "hawking.frontier_download.v1",
            "label": model.label,
            "started_at": "2026-07-08T00:00:00+00:00",
            "ended_at": "2026-07-08T00:10:00+00:00",
            "returncode": 0,
            "duration_s": 600.0,
            "delta_local_dir_gb": 126.0,
            "observed_mb_s_from_delta": 210.0,
            "progress": {
                "sample_count": 3,
                "average_tracked_mb_s": 208.0,
                "last_window_mb_s": 205.0,
                "longest_no_progress_s": 0.0,
                "stalled": False,
                "terminated_for_stall": False,
            },
        })
        ledger = build_ledger(root)
        row = next(r for r in ledger["models"] if r["label"] == model.label)
        check("ledger sees staged source", row["source_dir_exists"])
        check("ledger sees artifact", row["artifact_exists"])
        check("release waits for artifact inventory", not row["release_safe"])
        lc_wait = build_lifecycle(root, storage_budget_gb=8000.0)
        wait_node = next(n for n in lc_wait["nodes"] if n["label"] == model.label)
        check("lifecycle requests artifact inventory before release",
              wait_node["state"] == "needs-artifact-inventory")
        _write_json(_artifact_inventory_path(model, root), {
            "schema": "hawking.frontier_artifact_inventory.v1",
            "generated_at": "2026-07-08T00:05:00+00:00",
            "label": model.label,
            "hf_id": model.hf_id,
            "artifacts": [{
                "path": str(tq),
                "bytes": tq.stat().st_size,
                "gb": round(_gb_from_bytes(tq.stat().st_size), 6),
                "sha256": _sha256_file(tq),
            }],
        })
        ledger = build_ledger(root)
        row = next(r for r in ledger["models"] if r["label"] == model.label)
        check("ledger validates artifact inventory", row["artifact_inventory"]["ok"])
        check("ledger marks release safe", row["release_safe"])
        check("ledger summarizes download telemetry", row["download"]["last_observed_mb_s"] == 210.0)
        check("ledger summarizes download progress telemetry",
              row["download"]["last_progress_sample_count"] == 3
              and row["download"]["last_longest_no_progress_s"] == 0.0)
        check("strict license gate accepts complete accepted record", row["license_gate"]["ok"])
        reviewed_only = frontier_licenses.license_status({"status": "reviewed", "by": "selftest"}, model.label)
        check("strict license gate rejects reviewed-only records", not reviewed_only["ok"])
        license_decisions_path = root / "reports" / "condense" / "frontier_license_decisions.draft.json"
        check("license decisions draft writes signed missing-operator workbook",
              cmd_license_decisions(argparse.Namespace(
                  mode="draft",
                  label=[model.label],
                  out=str(license_decisions_path),
                  path=str(license_decisions_path),
                  status="accepted",
                  by="",
                  license="",
                  terms_url="",
                  terms_snapshot="",
                  allowed_use="",
                  redistribution="",
                  source_policy="",
                  note="",
                  final=False,
                  confirm=False,
                  json=False,
              )) == 0
              and _read_json(license_decisions_path, {}).get("operator_required_count") == 1)
        check("license decisions draft verifies",
              cmd_license_decisions(argparse.Namespace(
                  mode="verify",
                  label=[],
                  out=str(license_decisions_path),
                  path=str(license_decisions_path),
                  status="accepted",
                  by="",
                  license="",
                  terms_url="",
                  terms_snapshot="",
                  allowed_use="",
                  redistribution="",
                  source_policy="",
                  note="",
                  final=False,
                  confirm=False,
                  json=False,
              )) == 0)
        check("license decisions draft is not applyable without required fields",
              not _license_decisions_status(_read_json(license_decisions_path, {}))["applyable"])
        license_decisions_final_path = root / "reports" / "condense" / "frontier_license_decisions.final.json"
        check("license decisions final workbook is applyable",
              cmd_license_decisions(argparse.Namespace(
                  mode="draft",
                  label=[model.label],
                  out=str(license_decisions_final_path),
                  path=str(license_decisions_final_path),
                  status="accepted",
                  by="selftest",
                  license="selftest-license",
                  terms_url="https://example.invalid/terms",
                  terms_snapshot="",
                  allowed_use="research",
                  redistribution="none",
                  source_policy="local-only-delete-after-bake",
                  note="synthetic accepted license",
                  final=True,
                  confirm=False,
                  json=False,
              )) == 0
              and _license_decisions_status(_read_json(license_decisions_final_path, {}))["applyable"])
        final_doc = _read_json(license_decisions_final_path, {})
        final_row = final_doc["decisions"][0]
        final_record = _license_record_from_decision(final_row, model)
        check("license decisions final row passes strict license status",
              frontier_licenses.license_status(final_record, model.label)["ok"])
        hold_model = FRONTIER_MODELS[1]
        ok_release, missing, _ = release_guard(hold_model, root)
        check("missing model is not release safe", not ok_release)
        check("missing model names blockers", bool(missing))
        cp = cycle_plan(root)
        check("cycle plan includes every frontier model", len(cp["rows"]) == len(FRONTIER_MODELS))
        check("cycle peak exceeds largest source", cp["cycle_peak_gb"] > max(m.download_gb for m in FRONTIER_MODELS))
        swp = manifest_storage_wave_plan(storage_budget_gb=8000.0, link_mb_s=300.0,
                                         efficiency=0.7, scratch_gb=200.0,
                                         cache_reserve_gb=CACHE_RESERVE_GB)
        check("storage waves fit the target SSD by cycling", not swp["impossible_labels"])
        check("storage waves add operator checkpoints", swp["wave_count"] >= 2)
        check("storage waves include every frontier model",
              sum(len(w["labels"]) for w in swp["waves"]) == len(FRONTIER_MODELS))
        gate = build_launch_gate(root, phase="claim", allow_unreviewed=True)
        check("launch gate sees missing disk as failure on tiny root", not gate["ok"])
        check("launch gate includes storage wave failure on tiny root", any(
            c["name"] == "storage-wave-plan" and not c["ok"] for c in gate["checks"]))
        check("claim launch gate includes parity failure", any(c["name"] == "frontier-parity" and not c["ok"]
                                                             for c in gate["checks"]))
        check("claim launch gate includes source provenance failure", any(
            c["name"] == "frontier-source-provenance" and not c["ok"] for c in gate["checks"]))
        check("claim launch gate includes baseline coverage failure", any(
            c["name"] == "frontier-baseline-coverage" and not c["ok"] for c in gate["checks"]))
        check("claim launch gate includes eval coverage failure", any(
            c["name"] == "frontier-eval-coverage" and not c["ok"] for c in gate["checks"]))
        check("claim launch gate includes native serve receipt failure", any(
            c["name"] == "frontier-native-serve-receipts" and not c["ok"] for c in gate["checks"]))
        check("claim launch gate includes RAM-cliff receipt failure", any(
            c["name"] == "frontier-ramcliff-receipts" and not c["ok"] for c in gate["checks"]))
        check("claim launch gate includes Doctor recovery failure", any(
            c["name"] == "frontier-doctor-recovery" and not c["ok"] for c in gate["checks"]))
        check("claim launch gate includes experiment depth failure", any(
            c["name"] == "frontier-experiment-depth" and not c["ok"] for c in gate["checks"]))
        check("claim launch gate includes signed claim bundle failure", any(
            c["name"] == "frontier-signed-claim-bundles" and not c["ok"] for c in gate["checks"]))
        check("manifest consistency passes", not manifest_findings())
        check("manifest drift check passes for active consumers", not manifest_drift_findings(ROOT))
        lc = build_lifecycle(root, storage_budget_gb=8000.0)
        node = next(n for n in lc["nodes"] if n["label"] == model.label)
        check("lifecycle marks releasable synthetic model", node["state"] == "ready-release-source")
        hold = next(n for n in lc["nodes"] if n["label"] == hold_model.label)
        check("lifecycle blocks missing license before download", hold["state"] == "needs-license-review")
        licenses = _license_ledger(root)
        licenses[hold_model.label] = {
            "status": "accepted",
            "by": "selftest",
            "note": "synthetic accepted license",
            "license": "selftest-license",
            "terms_url": "https://example.invalid/terms",
            "allowed_use": "research",
            "redistribution": "none",
            "source_policy": "local-only-delete-after-bake",
        }
        _write_json(root / LICENSE_PATH, licenses)
        _append_jsonl(root / DOWNLOAD_LOG, {
            "schema": "hawking.frontier_download.v1",
            "label": hold_model.label,
            "started_at": "2026-07-08T00:00:00+00:00",
            "ended_at": "2026-07-08T00:20:00+00:00",
            "returncode": 124,
            "duration_s": 1200.0,
            "retry_reason": "no tracked local/cache growth >= 64.0 MB for 900s",
            "will_retry": False,
            "progress": {
                "sample_count": 16,
                "longest_no_progress_s": 900.0,
                "stalled": True,
                "terminated_for_stall": True,
                "stall_reason": "no tracked local/cache growth >= 64.0 MB for 900s",
            },
            "diagnostics": {
                "network_probe": {"ok": False, "error": "selftest"},
                "recommendations": [
                    "Hugging Face reachability probe failed; check VPN, DNS, router, and ISP path before retrying."
                ],
            },
        })
        stalled_lifecycle = build_lifecycle(root, storage_budget_gb=8000.0)
        stalled_node = next(n for n in stalled_lifecycle["nodes"] if n["label"] == hold_model.label)
        check("lifecycle distinguishes stalled downloads", stalled_node["state"] == "download-stalled")
        check("ledger summarizes download diagnostics",
              stalled_node["download"]["last_diagnostics_present"]
              and stalled_node["download"]["last_network_probe_ok"] is False
              and bool(stalled_node["download"]["last_diagnostic_recommendations"]))
        check("lifecycle carries first diagnostic recommendation",
              any("Hugging Face reachability probe failed" in b for b in stalled_node["blockers"]))
        check("run-next selector chooses first actionable node", _select_lifecycle_node(lc)["label"] == model.label)
        check("placeholder detector catches human commands",
              _command_has_placeholder("python tool --by <name>"))
        worktree = root / "worktree"
        worktree.mkdir()
        _run(["git", "-C", str(worktree), "init"], timeout=10)
        for rel in (
            "app/src/App.tsx",
            "tools/condense/frontier_ops.py",
            "crates/hawking/src/studio.rs",
            "node_modules/.bin/vite",
        ):
            path = worktree / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
        _run([
            "git", "-C", str(worktree), "add", "-f",
            "app/src/App.tsx",
            "tools/condense/frontier_ops.py",
            "crates/hawking/src/studio.rs",
            "node_modules/.bin/vite",
        ], timeout=10)
        wtp = build_worktree_plan(worktree)
        wtp_counts = {g["name"]: g["counts"]["total"] for g in wtp["groups"]}
        check("worktree plan classifies dirty tree by subsystem",
              wtp["ok"]
              and wtp_counts.get("hide-ui-tauri-assets") == 1
              and wtp_counts.get("condense-frontier-proof") == 1
              and wtp_counts.get("hawking-core-runtime") == 1
              and wtp_counts.get("local-deps-generated") == 1)
        check("worktree plan is signed and verifiable",
              _worktree_plan_status(wtp)["ok"]
              and _worktree_plan_status(wtp)["signature_ok"]
              and _worktree_plan_status(wtp)["risk"] == "high")
        _append_jsonl(root / EVENT_LOG, {
            "schema": "hawking.frontier_event.v1",
            "ts": "2026-07-08T00:20:00+00:00",
            "label": model.label,
            "hf_id": model.hf_id,
            "stage": "bake",
            "status": "pass",
            "duration_s": 12.5,
        })
        ledger2 = build_ledger(root)
        row2 = next(r for r in ledger2["models"] if r["label"] == model.label)
        check("ledger summarizes operator events", row2["events"]["stages"]["bake"]["total_duration_s"] == 12.5)
        refresh_path = root / "refresh.json"
        _write_json(refresh_path, {
            "schema": "hawking.frontier_refresh.v1",
            "candidates": [
                {"ok": True, "known": False, "modelId": "selftest-org/GLM-Review-MoE",
                 "tags": ["text-generation"], "source": "author:zai-org"},
                {"ok": True, "known": False, "modelId": "someone/tiny-embed",
                 "tags": ["sentence-similarity"], "source": "search:test"},
            ],
        })
        gate_review = build_launch_gate(root, phase="procure", allow_unreviewed=True,
                                        storage_budget_gb=8000.0,
                                        require_refresh=str(refresh_path))
        check("refresh review gate blocks review-worthy unknown candidates", any(
            c["name"] == "refresh-candidate-review" and not c["ok"]
            for c in gate_review["checks"]))
        review_plan_path = root / "reports" / "condense" / "frontier_review_plan.json"
        check("review plan blocks missing candidate decisions",
              cmd_review_plan(argparse.Namespace(
                  refresh=str(refresh_path),
                  out=str(review_plan_path),
                  json=False,
              )) == 1
              and _read_json(review_plan_path, {}).get("missing_count") == 1)
        review_decisions_path = root / "reports" / "condense" / "frontier_review_decisions.draft.json"
        check("review decisions draft writes signed missing-operator workbook",
              cmd_review_decisions(argparse.Namespace(
                  mode="draft",
                  refresh=str(refresh_path),
                  out=str(review_decisions_path),
                  decision="watch",
                  by="",
                  note="",
                  final=False,
                  json=False,
              )) == 0
              and _read_json(review_decisions_path, {}).get("operator_required_count") == 1)
        check("review decisions draft verifies",
              cmd_review_decisions(argparse.Namespace(
                  mode="verify",
                  path=str(review_decisions_path),
                  json=False,
              )) == 0)
        check("review decisions draft is not applyable without operator",
              not _review_decisions_status(_read_json(review_decisions_path, {}))["applyable"])
        review_decisions_final_path = root / "reports" / "condense" / "frontier_review_decisions.final.json"
        check("review decisions final workbook is applyable",
              cmd_review_decisions(argparse.Namespace(
                  mode="draft",
                  refresh=str(refresh_path),
                  out=str(review_decisions_final_path),
                  decision="watch",
                  by="selftest",
                  note="synthetic watch decision",
                  final=True,
                  json=False,
              )) == 0
              and _review_decisions_status(_read_json(review_decisions_final_path, {}))["applyable"])
        _write_json(root / REFRESH_REVIEW_PATH, {
            "selftest-org/GLM-Review-MoE": {"decision": "watch", "by": "selftest", "note": "candidate"}
        })
        gate_review2 = build_launch_gate(root, phase="procure", allow_unreviewed=True,
                                         storage_budget_gb=8000.0,
                                         require_refresh=str(refresh_path))
        check("refresh review gate passes after watch decision", any(
            c["name"] == "refresh-candidate-review" and c["ok"]
            for c in gate_review2["checks"]))
        wave0_packet = build_wave0_packet(root, argparse.Namespace(
            require_refresh=str(refresh_path),
            license_decisions=str(license_decisions_path),
            review_decisions=str(review_decisions_path),
            proof_pack=str(root / "reports" / "condense" / "frontier_proof_pack.local.json"),
            label=None,
            link_mbs=300.0,
            efficiency=0.7,
            scratch_gb=200.0,
            cache_reserve_gb=CACHE_RESERVE_GB,
            storage_budget_gb=8000.0,
            max_wave_hours=6.0,
        ))
        check("wave0 launch packet signs red readiness",
              _wave0_packet_status(wave0_packet)["ok"]
              and not wave0_packet["ok"]
              and not wave0_packet["procurement_permitted"])
        audit_dir = root / "docs" / "plans"
        audit_dir.mkdir(parents=True, exist_ok=True)
        studio_audit_path = audit_dir / "STUDIO_DEEP_AUDIT_2026_07_08.md"
        studio_audit_path.write_text(
            "\n".join([
                "# Studio deep audit",
                "Current overall grade after receipts: 9.8 / 10 as an operator-proof Studio run plan.",
                "",
                "| Facet | Grade | Why it is not 10 yet | 10+ condition |",
                "|---|---:|---|---|",
                "| Native `.tq` serving | 6.9 | actual receipts missing | serve receipt |",
                "| Observability / operator UX | 10.0 | signed `a|b` commands | maintain |",
                "",
            ]),
            encoding="utf-8",
        )
        external_audit_path = root / "external_audit.md"
        external_audit_path.write_text(
            "\n".join([
                "Current overall grade: 6.4 / 10.",
                "Local laptop potential after disk cleanup: 5.4 / 10.",
                "Studio potential: 8.4 / 10.",
            ]),
            encoding="utf-8",
        )
        _write_json(root / WORKTREE_PLAN_PATH, wtp)
        _write_json(root / WAVE0_PACKET_PATH, wave0_packet)
        _write_json(root / PROOF_PACK_PATH, {
            "schema": "hawking.frontier_proof_pack.v1",
            "ok": True,
            "model_count": 2,
            "blocked_claim_count": 2,
            "evidence_rows": [],
            "claim_bundles": [
                {"label": "a", "claim_admissible": False},
                {"label": "b", "claim_admissible": False},
            ],
        })
        claim_gate_path = root / "reports" / "condense" / "claim_gate.json"
        procure_gate_path = root / "reports" / "condense" / "procure_gate.json"
        _write_json(claim_gate_path, {"schema": "hawking.frontier_launch_gate.v1", "ok": False})
        _write_json(procure_gate_path, {"schema": "hawking.frontier_launch_gate.v1", "ok": False})
        audit_grade = build_audit_grade(root, argparse.Namespace(
            target_grade=8.4,
            external_audit=str(external_audit_path),
            studio_audit=str(studio_audit_path),
            launch_packet=str(root / WAVE0_PACKET_PATH),
            proof_pack=str(root / PROOF_PACK_PATH),
            worktree_plan=str(root / WORKTREE_PLAN_PATH),
            claim_gate=str(claim_gate_path),
            procurement_gate=str(procure_gate_path),
            scorecard=str(root / "reports" / "condense" / "scorecard.json"),
        ))
        _write_json(root / AUDIT_GRADE_PATH, audit_grade)
        check("audit-grade receipt signs target-not-proven state",
              _audit_grade_status(audit_grade)["ok"]
              and audit_grade["facet_count"] == 2
              and audit_grade["below_target_count"] == 1
              and not audit_grade["target_reached"]
              and audit_grade["frontier_claims_walled"])
        completion_audit = build_completion_audit(root, argparse.Namespace(
            label=[model.label],
            preflight_summary=str(PREFLIGHT_SUMMARY_PATH),
            environment=str(STUDIO_ENVIRONMENT_PATH),
            launch_packet=str(WAVE0_PACKET_PATH),
            worktree_plan=str(WORKTREE_PLAN_PATH),
            runtime_contract=str(RUNTIME_CONTRACT_PATH),
            proof_pack=str(PROOF_PACK_PATH),
            audit_grade=str(AUDIT_GRADE_PATH),
            require_refresh=str(refresh_path),
            storage_budget_gb=8000.0,
            link_mbs=300.0,
            efficiency=0.7,
            scratch_gb=200.0,
            cache_reserve_gb=CACHE_RESERVE_GB,
            max_wave_hours=6.0,
        ))
        completion_ids = {row["id"] for row in completion_audit["requirements"]}
        check("completion audit signs red Studio 10/10 state",
              _completion_audit_status(completion_audit)["ok"]
              and not completion_audit["completion_ok"]
              and "doctor_recovery_7b_plus" in completion_ids
              and "native_tq_serve" in completion_ids)
        for fm in FRONTIER_MODELS:
            provenance_record, _ = frontier_provenance.sign_record(
                frontier_provenance.complete_record(fm),
                model=fm,
            )
            _write_json(root / "reports" / "condense" / f"{fm.label}_source_provenance.json",
                        provenance_record)
            _write_json(root / "reports" / "condense" / f"{fm.label}_baselines.json", {
                "schema": "hawking.frontier_baselines.v1",
                "model": fm.label,
                "source": "real",
                "machine_class": "Studio-M1Ultra-128",
                "same_box": True,
                "baselines": [
                    {
                        "name": req["name"],
                        "status": "measured" if i < 2 else "na",
                        "same_box": True,
                        "reason": "selftest N/A for non-runnable baseline" if i >= 2 else "",
                    }
                    for i, req in enumerate(frontier_coverage.BASELINE_REQUIREMENTS)
                ],
            })
            _write_json(root / "reports" / "condense" / f"{fm.label}_eval.json", {
                "schema": "hawking.frontier_eval_coverage.v1",
                "model": fm.label,
                "mode": "real",
                "machine_class": "Studio-M1Ultra-128",
                "domains": [
                    {"domain": req["name"], "status": "pass", "command": "selftest"}
                    for req in frontier_coverage.EVAL_REQUIREMENTS
                ],
            })
            _write_json(root / "reports" / "condense" / f"{fm.label}_serve.json", {
                "schema": "hawking.frontier_serve.v1",
                "model": fm.label,
                "source": "measured",
                "machine_class": "Studio-M1Ultra-128",
                "status": "pass",
                "native_tq": True,
                "rehydrate_f16": False,
                "tq_strict": True,
                "all_linear": True,
                "gpu_bitslice": True,
                "served_forward_pass": True,
                "parity_pass": True,
                "tok_s": 12.5,
                "artifact_sha256": "a" * 64,
                "commands": ["selftest serve"],
                "git_commit": "deadbeef",
            })
            _write_json(root / "reports" / "condense" / f"{fm.label}_ramcliff.json", {
                "schema": "hawking.frontier_ramcliff.v1",
                "model": fm.label,
                "source": "measured",
                "machine_class": "Studio-M1Ultra-128",
                "verdict": "CLIFF-WIN",
                "served_native_tq": True,
                "tok_s_resident": 20.0,
                "tok_s_swapping": 1.0,
                "j_per_tok_resident": 0.1,
                "j_per_tok_swapping": 1.0,
                "cliff_x": 20.0,
                "gate": {
                    "condensed_resident": True,
                    "served_native_tq": True,
                    "q4k_overflows_box": True,
                    "cliff_x_over_gate": True,
                    "resident_lower_energy": True,
                },
                "artifact_sha256": "b" * 64,
                "commands": ["selftest ramcliff"],
                "git_commit": "deadbeef",
            })
            doctor_record, _ = frontier_doctor_recovery.sign_record(
                frontier_doctor_recovery.complete_record(fm),
                model=fm,
            )
            _write_json(root / "reports" / "condense" / f"{fm.label}_doctor_recovery.json",
                        doctor_record)
            _write_json(root / "reports" / "condense" / f"{fm.label}_experiment_matrix.json", {
                "schema": "hawking.frontier_experiment_matrix.v1",
                "model": fm.label,
                "mode": "real",
                "machine_class": "Studio-M1Ultra-128",
                "experiments": {
                    "floor_seeds": [
                        {"category": "floor_seed", "seed": seed, "status": "pass"}
                        for seed in (1, 2, 3)
                    ],
                    "calibration_ablations": [
                        {"category": "calibration_ablations", "name": name, "status": "pass"}
                        for name in (
                            "domain_matched_calib",
                            "mixed_domain_calib",
                            "awq_alpha_sweep",
                            "residual_depth_sweep",
                        )
                    ],
                    "bpw_ladder": [
                        {"category": "bpw_ladder", "bpw": bpw, "status": "pass"}
                        for bpw in (1.50, 1.25, 1.00, 0.75)
                    ],
                    "moe_expert_ablation": [
                        {"category": "moe_expert_ablation", "status": "pass", "name": "expert_sensitivity"}
                    ],
                    "ramcliff_repeats": [
                        {"category": "ramcliff_repeats", "run_type": run_type, "status": "pass"}
                        for run_type in ("cold", "cold", "cold", "warm", "warm", "warm")
                    ],
                    "baseline_variants": [
                        {"category": "baseline_variants", "name": name, "status": "pass"}
                        for name in ("llama_q4", "llama_iq2", "mlx_4bit", "unsloth_or_exl3")
                    ],
                    "null_certification": [
                        {"category": "null_certification", "name": name, "status": "certified", "reason": "selftest"}
                        for name in ("failed_recipe", "baseline_or_quality_loss")
                    ],
                    "rebake_or_hash_verify": [
                        {"category": "rebake_or_hash_verify", "status": "verified"}
                    ],
                },
            })
        coverage_plan = frontier_coverage.coverage_plan(root, [model.label])
        check("coverage plan exposes baseline skeleton path",
              coverage_plan["labels"][0]["baseline_path"].endswith(f"{model.label}_baselines.json"))
        gate_covered = build_launch_gate(root, phase="claim", allow_unreviewed=True,
                                         storage_budget_gb=8000.0)
        check("baseline coverage gate passes after measured/N/A receipts", any(
            c["name"] == "frontier-baseline-coverage" and c["ok"] for c in gate_covered["checks"]))
        check("eval coverage gate passes after domain receipts", any(
            c["name"] == "frontier-eval-coverage" and c["ok"] for c in gate_covered["checks"]))
        receipt_plan = frontier_receipts.receipt_plan(root, [model.label])
        check("receipt plan exposes serve path",
              receipt_plan["labels"][0]["serve_path"].endswith(f"{model.label}_serve.json"))
        check("native serve receipt gate passes after strict receipts", any(
            c["name"] == "frontier-native-serve-receipts" and c["ok"] for c in gate_covered["checks"]))
        check("RAM-cliff receipt gate passes after strict receipts", any(
            c["name"] == "frontier-ramcliff-receipts" and c["ok"] for c in gate_covered["checks"]))
        check("Doctor recovery gate passes after strict receipts", any(
            c["name"] == "frontier-doctor-recovery" and c["ok"] for c in gate_covered["checks"]))
        experiment_plan = frontier_experiments.experiment_plan(root, [model.label])
        check("experiment plan exposes matrix path",
              experiment_plan["labels"][0]["matrix_path"].endswith(f"{model.label}_experiment_matrix.json"))
        check("experiment depth gate passes after expensive-mode matrix", any(
            c["name"] == "frontier-experiment-depth" and c["ok"] for c in gate_covered["checks"]))
        check("signed claim bundle gate remains explicit after evidence gates pass", any(
            c["name"] == "frontier-signed-claim-bundles" and not c["ok"] for c in gate_covered["checks"]))
        claim_node = _lifecycle_node(model, {
            "license": {"status": "accepted"},
            "license_gate": {"ok": True},
            "source_dir_exists": False,
            "artifact_exists": True,
            "artifact_inventory": {"ok": True},
            "frontier_record_exists": True,
            "official_receipts": [],
            "release_events": [],
            "release_safe": False,
            "source_provenance": {"ok": True},
            "baseline_coverage": {"ok": True},
            "eval_coverage": {"ok": True},
            "serve_receipt": {"ok": True},
            "ramcliff_receipt": {"ok": True},
            "doctor_recovery": {"ok": True},
            "experiment_matrix": {"ok": True},
            "download": {},
            "events": {},
        }, {"claim_gate": "ALLOW", "parity": {"problems": []}}, root)
        check("lifecycle suggests product claim-bundle build command",
              claim_node["state"] == "claim-blocked-bundle"
              and any(cmd.startswith("hawking studio claim-bundle-build")
                      for cmd in claim_node.get("next_commands", [])))
        check("native serve capture harness selftest passes", frontier_serve_capture.selftest())
        proof_pack_path = root / "reports" / "condense" / "frontier_proof_pack.local.json"
        proof = build_proof_pack(root, argparse.Namespace(
            label=[model.label],
            force=True,
            force_final=False,
            claim_suffix=".local",
            machine_class="Studio-M1Ultra-128",
            no_require_ramcliff=False,
            out=str(proof_pack_path),
            json=False,
        ))
        proof_bundle = proof["claim_bundles"][0]
        check("proof pack writes signed draft wall",
              proof["ok"]
              and proof["blocked_claim_count"] == 1
              and proof_bundle["written"]
              and proof_bundle["blocker_count"] > 0
              and proof_pack_path.exists())
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Frontier operator ledger/status/lifecycle guards.")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("ledger", help="write machine-readable frontier ledger")
    p.add_argument("--out", default=str(LEDGER_PATH))
    p.add_argument("--refresh-hf", action="store_true", help="query HF model metadata")
    p.add_argument("--dry-run-sizes", action="store_true", help="run `hf download --dry-run` per model")
    p.add_argument("--include", default="*.safetensors")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--link-mbs", type=float, default=300.0)
    p.add_argument("--efficiency", type=float, default=0.7)
    p.add_argument("--scratch-gb", type=float, default=200.0)
    p.add_argument("--cache-reserve-gb", type=float, default=CACHE_RESERVE_GB)
    p.add_argument("--storage-budget-gb", type=float, default=None,
                   help="storage budget for wave planning; default=current free disk")
    p.add_argument("--max-wave-hours", type=float, default=6.0,
                   help="operator checkpoint target for storage waves; <=0 disables")
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("status", help="print compact frontier status")
    p.add_argument("--json", action="store_true")
    p.add_argument("--link-mbs", type=float, default=300.0)
    p.add_argument("--efficiency", type=float, default=0.7)
    p.add_argument("--scratch-gb", type=float, default=200.0)
    p.add_argument("--cache-reserve-gb", type=float, default=CACHE_RESERVE_GB)
    p.add_argument("--storage-budget-gb", type=float, default=None,
                   help="storage budget for wave planning; default=current free disk")
    p.add_argument("--max-wave-hours", type=float, default=6.0,
                   help="operator checkpoint target for storage waves; <=0 disables")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("worktree-plan", help="group dirty tree by subsystem for stack splitting")
    p.add_argument("--out", default=None)
    p.add_argument("--verify", default=None,
                   help="verify a signed worktree split plan instead of writing a new one")
    p.add_argument("--json", action="store_true")
    p.add_argument("--max-paths", type=int, default=8,
                   help="max paths to print per subsystem in human output")
    p.set_defaults(func=cmd_worktree_plan)

    p = sub.add_parser("storage-plan", help="print storage-aware frontier download waves")
    p.add_argument("--json", action="store_true")
    p.add_argument("--link-mbs", type=float, default=300.0)
    p.add_argument("--efficiency", type=float, default=0.7)
    p.add_argument("--scratch-gb", type=float, default=200.0)
    p.add_argument("--cache-reserve-gb", type=float, default=CACHE_RESERVE_GB)
    p.add_argument("--storage-budget-gb", type=float, default=None,
                   help="storage budget; default=current free disk")
    p.add_argument("--max-wave-hours", type=float, default=6.0,
                   help="operator checkpoint target; <=0 disables")
    p.set_defaults(func=cmd_storage_plan)

    p = sub.add_parser("lifecycle", help="print per-model DAG state and next safe command")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--link-mbs", type=float, default=300.0)
    p.add_argument("--efficiency", type=float, default=0.7)
    p.add_argument("--scratch-gb", type=float, default=200.0)
    p.add_argument("--cache-reserve-gb", type=float, default=CACHE_RESERVE_GB)
    p.add_argument("--storage-budget-gb", type=float, default=None,
                   help="storage budget; default=current free disk")
    p.add_argument("--max-wave-hours", type=float, default=6.0,
                   help="operator checkpoint target; <=0 disables")
    p.set_defaults(func=cmd_lifecycle)

    p = sub.add_parser("coverage-plan", help="print baseline/eval coverage requirements and skeleton paths")
    p.add_argument("label", nargs="*", help="optional frontier label(s); default all")
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_coverage_plan)

    p = sub.add_parser("parity-receipt", help="draft, sign, or verify architecture parity receipts")
    parity_sub = p.add_subparsers(dest="parity_mode", required=True)
    for mode in ("draft", "sign", "verify"):
        c = parity_sub.add_parser(mode, help=f"{mode} signed architecture parity receipts")
        c.add_argument("label", nargs="*", help="frontier label(s); default all")
        c.add_argument("--out-dir", default="")
        c.add_argument("--json", action="store_true")
        if mode == "draft":
            c.add_argument("--force", action="store_true")
            c.add_argument("--sign-draft", action="store_true")
            c.add_argument("--machine-class", default="Studio-M1Ultra-128")
            c.set_defaults(allow_blocked_draft=False)
        else:
            c.set_defaults(force=False, sign_draft=False, machine_class="Studio-M1Ultra-128")
        if mode == "sign":
            c.add_argument("--allow-blocked-draft", action="store_true")
        else:
            c.set_defaults(allow_blocked_draft=False)
        c.set_defaults(func=cmd_parity_receipt)

    p = sub.add_parser("coverage-receipt", help="draft, sign, or verify baseline/eval coverage receipts")
    coverage_sub = p.add_subparsers(dest="receipt_mode", required=True)
    for mode in ("draft", "sign", "verify"):
        c = coverage_sub.add_parser(mode, help=f"{mode} signed baseline/eval coverage receipts")
        c.add_argument("label", nargs="*", help="frontier label(s); default all")
        c.add_argument("--kind", choices=("baseline", "eval", "both"), default="both")
        c.add_argument("--out-dir", default="")
        c.add_argument("--json", action="store_true")
        if mode == "draft":
            c.add_argument("--force", action="store_true")
            c.add_argument("--sign-draft", action="store_true")
            c.add_argument("--machine-class", default="Studio-M1Ultra-128")
            c.set_defaults(allow_blocked_draft=False)
        else:
            c.set_defaults(force=False, sign_draft=False, machine_class="Studio-M1Ultra-128")
        if mode == "sign":
            c.add_argument("--allow-blocked-draft", action="store_true")
        else:
            c.set_defaults(allow_blocked_draft=False)
        c.set_defaults(func=cmd_coverage_receipt)

    p = sub.add_parser("source-provenance", help="plan, draft, sign, or verify source-provenance receipts")
    provenance_sub = p.add_subparsers(dest="provenance_mode", required=True)
    c = provenance_sub.add_parser("plan", help="print source-provenance receipt paths")
    c.add_argument("label", nargs="*", help="frontier label(s); default all")
    c.add_argument("--out", default="")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_source_provenance)
    for mode in ("draft", "sign", "verify"):
        c = provenance_sub.add_parser(mode, help=f"{mode} signed source-provenance receipts")
        c.add_argument("label", nargs="*", help="frontier label(s); default all")
        c.add_argument("--out-dir", default="")
        c.add_argument("--json", action="store_true")
        if mode == "draft":
            c.add_argument("--force", action="store_true")
            c.add_argument("--sign-draft", action="store_true")
            c.add_argument("--machine-class", default="Studio-M1Ultra-128")
            c.set_defaults(allow_blocked_draft=False)
        else:
            c.set_defaults(force=False, sign_draft=False, machine_class="Studio-M1Ultra-128")
        if mode == "sign":
            c.add_argument("--allow-blocked-draft", action="store_true")
        else:
            c.set_defaults(allow_blocked_draft=False)
        c.set_defaults(func=cmd_source_provenance)

    p = sub.add_parser("receipt-record", help="draft, sign, or verify native serve/RAM-cliff receipts")
    native_sub = p.add_subparsers(dest="receipt_mode", required=True)
    for mode in ("draft", "sign", "verify"):
        c = native_sub.add_parser(mode, help=f"{mode} signed native serve/RAM-cliff receipts")
        c.add_argument("label", nargs="*", help="frontier label(s); default all")
        c.add_argument("--kind", choices=("serve", "ramcliff", "both"), default="both")
        c.add_argument("--out-dir", default="")
        c.add_argument("--json", action="store_true")
        if mode == "draft":
            c.add_argument("--force", action="store_true")
            c.add_argument("--sign-draft", action="store_true")
            c.add_argument("--machine-class", default="Studio-M1Ultra-128")
            c.set_defaults(allow_blocked_draft=False)
        else:
            c.set_defaults(force=False, sign_draft=False, machine_class="Studio-M1Ultra-128")
        if mode == "sign":
            c.add_argument("--allow-blocked-draft", action="store_true")
        else:
            c.set_defaults(allow_blocked_draft=False)
        c.set_defaults(func=cmd_receipt_record)

    p = sub.add_parser("serve-capture", help="capture native .tq serve bench JSON as a signed serve receipt")
    p.add_argument("label")
    p.add_argument("--artifact", required=True, help="existing .tq artifact path")
    p.add_argument("--bench-json", required=True, dest="bench_json", help="JSON report emitted by native serve")
    p.add_argument("--command", required=True, help="exact serve command that produced the report")
    p.add_argument("--served-forward-receipt", required=True)
    p.add_argument("--parity-receipt", required=True)
    p.add_argument("--machine-class", default="Studio-M1Ultra-128")
    p.add_argument("--out", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_serve_capture)

    p = sub.add_parser("receipt-plan", help="print strict serve/RAM-cliff receipt requirements")
    p.add_argument("label", nargs="*", help="optional frontier label(s); default all")
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_receipt_plan)

    p = sub.add_parser("experiment-plan", help="print expensive-mode experiment matrix requirements")
    p.add_argument("label", nargs="*", help="optional frontier label(s); default all")
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_experiment_plan)

    p = sub.add_parser("experiment-receipt", help="draft, sign, or verify expensive-mode experiment matrices")
    experiment_sub = p.add_subparsers(dest="experiment_mode", required=True)
    for mode in ("draft", "sign", "verify"):
        c = experiment_sub.add_parser(mode, help=f"{mode} signed experiment matrices")
        c.add_argument("label", nargs="*", help="frontier label(s); default all")
        c.add_argument("--out-dir", default="")
        c.add_argument("--json", action="store_true")
        if mode == "draft":
            c.add_argument("--force", action="store_true")
            c.add_argument("--sign-draft", action="store_true")
            c.add_argument("--machine-class", default="Studio-M1Ultra-128")
            c.set_defaults(allow_blocked_draft=False)
        else:
            c.set_defaults(force=False, sign_draft=False, machine_class="Studio-M1Ultra-128")
        if mode == "sign":
            c.add_argument("--allow-blocked-draft", action="store_true")
        else:
            c.set_defaults(allow_blocked_draft=False)
        c.set_defaults(func=cmd_experiment_receipt)

    p = sub.add_parser("doctor-recovery-plan", help="print Doctor recovery receipt requirements")
    p.add_argument("label", nargs="*", help="optional frontier label(s); default all")
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor_recovery_plan)

    p = sub.add_parser("doctor-recovery-receipt", help="draft, sign, or verify Doctor recovery receipts")
    recovery_sub = p.add_subparsers(dest="recovery_mode", required=True)
    for mode in ("draft", "sign", "verify"):
        c = recovery_sub.add_parser(mode, help=f"{mode} signed Doctor recovery receipts")
        c.add_argument("label", nargs="*", help="frontier label(s); default all")
        c.add_argument("--out-dir", default="")
        c.add_argument("--json", action="store_true")
        if mode == "draft":
            c.add_argument("--force", action="store_true")
            c.add_argument("--sign-draft", action="store_true")
            c.add_argument("--machine-class", default="Studio-M1Ultra-128")
            c.set_defaults(allow_blocked_draft=False)
        else:
            c.set_defaults(force=False, sign_draft=False, machine_class="Studio-M1Ultra-128")
        if mode == "sign":
            c.add_argument("--allow-blocked-draft", action="store_true")
        else:
            c.set_defaults(allow_blocked_draft=False)
        c.set_defaults(func=cmd_doctor_recovery_receipt)

    p = sub.add_parser("proof-pack", help="draft all signed evidence envelopes and blocked local claim bundles")
    p.add_argument("label", nargs="*", help="frontier label(s); default all")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing draft/local evidence; final receipts are still preserved")
    p.add_argument("--force-final", action="store_true",
                   help="dangerous: allow overwriting final evidence or admissible bundles")
    p.add_argument("--claim-suffix", default=".local",
                   help="suffix for generated claim bundles; default writes <LABEL>_claim_bundle.local.json")
    p.add_argument("--machine-class", default="Studio-M1Ultra-128")
    p.add_argument("--no-require-ramcliff", action="store_true",
                   help="build serve-only local bundles; RAM-cliff proof packs should not use this")
    p.add_argument("--out", default=str(COND_DIR / "frontier_proof_pack.local.json"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_proof_pack)

    p = sub.add_parser("license-plan", help="print strict license/gating approval commands")
    p.add_argument("label", nargs="*", help="optional frontier label(s); default all")
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_license_plan)

    p = sub.add_parser("license-decisions", help="draft/sign/verify/apply signed batch license decisions")
    p.add_argument("mode", choices=("draft", "sign", "verify", "apply"))
    p.add_argument("label", nargs="*", help="optional frontier label(s); default all")
    p.add_argument("--out", default=str(LICENSE_DECISIONS_PATH),
                   help="signed license workbook written by draft/sign modes")
    p.add_argument("--path", default=str(LICENSE_DECISIONS_PATH),
                   help="license workbook read by sign/verify/apply modes")
    p.add_argument("--status", choices=("unreviewed", "reviewed", "accepted", "rejected"),
                   default="accepted")
    p.add_argument("--by", default="")
    p.add_argument("--license", default="")
    p.add_argument("--terms-url", default="", dest="terms_url")
    p.add_argument("--terms-snapshot", default="", dest="terms_snapshot")
    p.add_argument("--allowed-use", choices=sorted(frontier_licenses.ALLOWED_USE), default="")
    p.add_argument("--redistribution", choices=sorted(frontier_licenses.REDISTRIBUTION), default="")
    p.add_argument("--source-policy", choices=sorted(frontier_licenses.SOURCE_POLICY), default="")
    p.add_argument("--note", default="")
    p.add_argument("--final", action="store_true",
                   help="mark draft rows final; accepted rows still need all required fields")
    p.add_argument("--confirm", action="store_true",
                   help="required by apply mode before frontier_license_acceptance.json is written")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_license_decisions)

    p = sub.add_parser("run-next", help="dry-run or apply the next lifecycle-safe command")
    p.add_argument("--label", default=None, help="optional frontier label/hf id to target")
    p.add_argument("--json", action="store_true")
    p.add_argument("--yes", action="store_true", help="actually execute if the state is executable")
    p.add_argument("--allow-download", action="store_true",
                   help="permit network/download commands after launch gate passes")
    p.add_argument("--allow-heavy", action="store_true",
                   help="permit heavy bake/serve commands")
    p.add_argument("--allow-release-dry-run", action="store_true",
                   help="permit release-source dry-run commands; never deletes source")
    p.add_argument("--require-refresh", default=None,
                   help="refresh ledger required before downloads")
    p.add_argument("--link-mbs", type=float, default=300.0)
    p.add_argument("--efficiency", type=float, default=0.7)
    p.add_argument("--scratch-gb", type=float, default=200.0)
    p.add_argument("--cache-reserve-gb", type=float, default=CACHE_RESERVE_GB)
    p.add_argument("--storage-budget-gb", type=float, default=None,
                   help="storage budget; default=current free disk")
    p.add_argument("--max-wave-hours", type=float, default=6.0,
                   help="operator checkpoint target; <=0 disables")
    p.set_defaults(func=cmd_run_next)

    p = sub.add_parser("artifact-inventory", help="hash .tq artifacts before guarded source release")
    p.add_argument("label")
    p.set_defaults(func=cmd_artifact_inventory)

    p = sub.add_parser("release-source", help="guarded source deletion after artifact + receipt evidence")
    p.add_argument("label")
    p.add_argument("--dry-run", action="store_true", help="default behavior; print evidence only")
    p.add_argument("--yes", action="store_true", help="actually delete after all guards pass")
    p.set_defaults(func=cmd_release_source)

    p = sub.add_parser("launch-gate", help="go/no-go for procure or claim phases")
    p.add_argument("--phase", choices=("procure", "claim"), default="procure")
    p.add_argument("--allow-unreviewed", action="store_true",
                   help="development escape hatch; real launches should review licenses")
    p.add_argument("--require-refresh", default=None,
                   help="path to a frontier_refresh.json ledger to require")
    p.add_argument("--link-mbs", type=float, default=300.0)
    p.add_argument("--efficiency", type=float, default=0.7)
    p.add_argument("--scratch-gb", type=float, default=200.0)
    p.add_argument("--cache-reserve-gb", type=float, default=CACHE_RESERVE_GB)
    p.add_argument("--storage-budget-gb", type=float, default=None,
                   help="storage budget for wave planning; default=current free disk")
    p.add_argument("--max-wave-hours", type=float, default=6.0,
                   help="operator checkpoint target for storage waves; <=0 disables")
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_launch_gate)

    p = sub.add_parser("launch-packet", help="build or verify signed Studio wave-0 launch packet")
    p.add_argument("mode", choices=("build", "verify"))
    p.add_argument("--out", default=str(WAVE0_PACKET_PATH),
                   help="signed packet written by build mode")
    p.add_argument("--path", default=str(WAVE0_PACKET_PATH),
                   help="signed packet read by verify mode")
    p.add_argument("--require-refresh", default=str(COND_DIR / "frontier_refresh.preflight.json"))
    p.add_argument("--license-decisions", default=str(LICENSE_DECISIONS_PATH))
    p.add_argument("--review-decisions", default=str(REFRESH_REVIEW_DECISIONS_PATH))
    p.add_argument("--worktree-plan", default=str(WORKTREE_PLAN_PATH))
    p.add_argument("--runtime-contract", default=str(RUNTIME_CONTRACT_PATH))
    p.add_argument("--proof-pack", default=str(PROOF_PACK_PATH))
    p.add_argument("--label", default=None, help="optional run-next label/hf id to target")
    p.add_argument("--link-mbs", type=float, default=300.0)
    p.add_argument("--efficiency", type=float, default=0.7)
    p.add_argument("--scratch-gb", type=float, default=200.0)
    p.add_argument("--cache-reserve-gb", type=float, default=CACHE_RESERVE_GB)
    p.add_argument("--storage-budget-gb", type=float, default=8000.0)
    p.add_argument("--max-wave-hours", type=float, default=6.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_launch_packet)

    p = sub.add_parser("audit-grade", help="build or verify signed Studio audit-grade receipt")
    p.add_argument("mode", choices=("build", "verify"))
    p.add_argument("--out", default=str(AUDIT_GRADE_PATH),
                   help="signed audit-grade receipt written by build mode")
    p.add_argument("--path", default=str(AUDIT_GRADE_PATH),
                   help="signed audit-grade receipt read by verify mode")
    p.add_argument("--target-grade", type=float, default=8.4)
    p.add_argument("--external-audit", default=str(DEFAULT_EXTERNAL_AUDIT_PATH))
    p.add_argument("--studio-audit", default=str(DEFAULT_STUDIO_AUDIT_PATH))
    p.add_argument("--launch-packet", default=str(WAVE0_PACKET_PATH))
    p.add_argument("--proof-pack", default=str(PROOF_PACK_PATH))
    p.add_argument("--worktree-plan", default=str(WORKTREE_PLAN_PATH))
    p.add_argument("--runtime-contract", default=str(RUNTIME_CONTRACT_PATH))
    p.add_argument("--claim-gate", default=str(COND_DIR / "frontier_claim_launch_gate.local.json"))
    p.add_argument("--procurement-gate", default=str(COND_DIR / "frontier_launch_gate.preflight.json"))
    p.add_argument("--scorecard", default=str(COND_DIR / "scorecard.json"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_audit_grade)

    p = sub.add_parser("completion-audit", help="build or verify signed Hawking Studio 10/10 completion audit")
    p.add_argument("mode", choices=("build", "verify"))
    p.add_argument("--out", default=str(COMPLETION_AUDIT_PATH),
                   help="signed completion audit written by build mode")
    p.add_argument("--path", default=str(COMPLETION_AUDIT_PATH),
                   help="signed completion audit read by verify mode")
    p.add_argument("--label", action="append", default=[],
                   help="frontier label to audit; may be repeated, default all")
    p.add_argument("--preflight-summary", default=str(PREFLIGHT_SUMMARY_PATH))
    p.add_argument("--environment", default=str(STUDIO_ENVIRONMENT_PATH))
    p.add_argument("--launch-packet", default=str(WAVE0_PACKET_PATH))
    p.add_argument("--worktree-plan", default=str(WORKTREE_PLAN_PATH))
    p.add_argument("--runtime-contract", default=str(RUNTIME_CONTRACT_PATH))
    p.add_argument("--proof-pack", default=str(PROOF_PACK_PATH))
    p.add_argument("--audit-grade", default=str(AUDIT_GRADE_PATH))
    p.add_argument("--require-refresh", default=str(COND_DIR / "frontier_refresh.preflight.json"))
    p.add_argument("--storage-budget-gb", type=float, default=8000.0)
    p.add_argument("--link-mbs", type=float, default=300.0)
    p.add_argument("--efficiency", type=float, default=0.7)
    p.add_argument("--scratch-gb", type=float, default=200.0)
    p.add_argument("--cache-reserve-gb", type=float, default=CACHE_RESERVE_GB)
    p.add_argument("--max-wave-hours", type=float, default=6.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_completion_audit)

    p = sub.add_parser("claim-bundle", help="build or verify signed public-claim bundles")
    claim_sub = p.add_subparsers(dest="bundle_mode", required=True)
    b = claim_sub.add_parser("build", help="build signed claim bundle(s)")
    b.add_argument("label", nargs="*", help="frontier label(s); default all")
    b.add_argument("--out", default="", help="output path for a single label")
    b.add_argument("--no-require-ramcliff", action="store_true",
                   help="build a serve-only claim bundle; RAM-cliff public wins should not use this")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_claim_bundle)
    v = claim_sub.add_parser("verify", help="verify signed claim bundle(s)")
    v.add_argument("path", nargs="*", help="bundle path(s); default all manifest bundle paths")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_claim_bundle)

    p = sub.add_parser("record-event", help="append bake/serve/eval/archive duration evidence")
    p.add_argument("label")
    p.add_argument("--stage", choices=("download", "bake", "verify", "serve", "eval", "archive", "baseline"),
                   required=True)
    p.add_argument("--status", choices=("start", "pass", "warn", "fail", "skip"), required=True)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--artifact", default="")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_record_event)

    p = sub.add_parser("record-license", help="record license/gating review status")
    p.add_argument("label")
    p.add_argument("--status", choices=("unreviewed", "reviewed", "accepted", "rejected"), required=True)
    p.add_argument("--by", default="")
    p.add_argument("--note", default="")
    p.add_argument("--license", default="")
    p.add_argument("--terms-url", default="", dest="terms_url")
    p.add_argument("--terms-snapshot", default="", dest="terms_snapshot")
    p.add_argument("--allowed-use", choices=sorted(frontier_licenses.ALLOWED_USE), default="")
    p.add_argument("--redistribution", choices=sorted(frontier_licenses.REDISTRIBUTION), default="")
    p.add_argument("--source-policy", choices=sorted(frontier_licenses.SOURCE_POLICY), default="")
    p.set_defaults(func=cmd_record_license)

    p = sub.add_parser("review-candidate", help="record accept/reject/watch decision for refresh candidates")
    p.add_argument("model_id")
    p.add_argument("--decision", choices=("accept", "reject", "watch"), required=True)
    p.add_argument("--by", required=True)
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_review_candidate)

    p = sub.add_parser("review-plan", help="write candidate-review command queue from a refresh ledger")
    p.add_argument("--refresh", default=str(REFRESH_PATH),
                   help="path to a frontier_refresh.json ledger")
    p.add_argument("--out", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_review_plan)

    p = sub.add_parser("review-decisions", help="draft/verify/apply signed batch candidate decisions")
    p.add_argument("mode", choices=("draft", "verify", "apply"))
    p.add_argument("--refresh", default=str(REFRESH_PATH),
                   help="refresh ledger used by draft mode")
    p.add_argument("--out", default=str(REFRESH_REVIEW_DECISIONS_PATH),
                   help="signed decision workbook written by draft mode")
    p.add_argument("--path", default=str(REFRESH_REVIEW_DECISIONS_PATH),
                   help="signed decision workbook read by verify/apply modes")
    p.add_argument("--decision", choices=("accept", "reject", "watch"), default="watch",
                   help="default draft decision for missing rows")
    p.add_argument("--by", default="", help="operator name for final draft decisions")
    p.add_argument("--note", default="", help="operator note for final draft decisions")
    p.add_argument("--final", action="store_true",
                   help="mark draft decisions final; requires by/note before apply will pass")
    p.add_argument("--confirm", action="store_true",
                   help="required by apply mode before frontier_refresh_reviews.json is written")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_review_decisions)

    p = sub.add_parser("refresh", help="query HF for review candidates before launch")
    p.add_argument("--author", action="append",
                   default=["moonshotai", "zai-org", "deepseek-ai", "Qwen", "meta-llama"])
    p.add_argument("--search", action="append", default=[],
                   help="optional broad search term; noisier than --author")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--out", default=str(REFRESH_PATH))
    p.set_defaults(func=cmd_refresh)

    p = sub.add_parser("selftest", help="synthetic tests, no network")
    p.set_defaults(func=cmd_selftest)
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.cmd:
        args = ap.parse_args(["status"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
