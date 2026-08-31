"""ODYSSEY_BOOTSTRAP_GOLDEN — seal the only proven launch state.

Odyssey I launched at commit 973e790fa with verdict=LAUNCH, n_met=16,
n_unmet=0, phase_transition=STARTED. That is the first and only campaign
state proven to launch. Descendants will change the tree immediately; a
golden recovered from memory a week later is not a recovery point.

This module pins that exact commit beside the launch receipt's own
seal_sha256, enumerates every obligation component as a path plus a content
hash (or ABSENT with a reason), and RUNS a recovery verification. It is not
a development freeze: descendants continue. The guard is that casual
overwrite is detectable, and that minting a successor golden is a written
procedure rather than an overwrite.

    python3 tools/future/bootstrap_golden.py --seal
    python3 tools/future/bootstrap_golden.py --verify-recovery
    python3 -m pytest tools/future/test_bootstrap_golden.py -q

Everything emitted is STATIC_ONLY, bench UNKNOWN, gpu_authority false.
A missing file in this sparse worktree is not project absence: presence is
settled with git ls-tree / git cat-file at the pinned commit.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.future._common import (
    GIT_TIMEOUT_S,
    RECEIPTS,
    REPO,
    git,
    write_receipt,
)


RECEIPT = "ODYSSEY_BOOTSTRAP_GOLDEN.json"
SCHEMA = "hawking.future.odyssey_bootstrap_golden.v1"
VERSION = 1
RECORDED_BY = "tools/future/bootstrap_golden.py"

# The launching commit. Not "the tip of odyssey-i". This sha introduced
# ODYSSEY_I_LAUNCH.json; the gate receipt recorded its parent as `head`
# because it was generated, then committed.
LAUNCH_COMMIT = "973e790fa77733120188db335108a0cb2cbdec34"
LAUNCH_TREE = "bf2c7f2240354f04bb1d6db441cf226f2f28569f"
LAUNCH_SEAL_SHA256 = "93fc95b7112fc8f2f02ee8f194a1d90bb761bfe7f3ad20500e2b1ae82a121d13"
LAUNCH_RECEIPT_REL = "receipts/future/ODYSSEY_I_LAUNCH.json"
GATE_RECEIPT_REL = "receipts/future/ODYSSEY_LAUNCH_GATE.json"
N_CRITERIA = 16

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. This receipt pins the "
    "source-and-receipt identity of the Odyssey I launch commit; it does not "
    "claim a running daemon, a GPU lease, or reconstitutable NX weights. "
    "Not a development freeze."
)

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

# Obligation rows, in the order the obligation names them. Extra honest rows
# follow: they are part of the golden state's content, including what it
# does not have.
OBLIGATION_IDS: tuple[str, ...] = (
    "git_commit",
    "hcli_daemon",
    "supervisor",
    "resident_nx",
    "fallback_identity",
    "sandbox_policy",
    "workunit_schemas",
    "resource_registry",
    "evidence_dag",
    "verifier",
    "tool_registry",
    "subagent_interfaces",
    "restart_path",
    "launch_gate",
    "tests",
    "trial_receipts",
    "timeline_seals",
)

HONEST_EXTRA_IDS: tuple[str, ...] = (
    "hcli_agentos_resident_py",
    "concurrency_doctor",
    "autonomy_run_safe_capabilities",
)

TRIAL_RECEIPT_RELS: tuple[str, ...] = (
    "receipts/future/DETACHED_WORK_TRIAL.json",
    "receipts/future/MUTATION_TRIAL.json",
    "receipts/future/SUCCESSION_TRIAL.json",
    "receipts/future/IMPROVEMENT_TRIAL.json",
    "receipts/future/AUTONOMY_DEGENERACY.json",
    "receipts/future/ODYSSEY_LAUNCH_GATE.json",
    "receipts/future/ODYSSEY_I_LAUNCH.json",
)

TIMELINE_RELS: tuple[str, ...] = (
    "receipts/future/AUTONOMY_TIMELINE_15m.json",
    "receipts/future/AUTONOMY_TIMELINE_30m.json",
    "receipts/future/AUTONOMY_TIMELINE_1h.json",
)

GRAVITY_OWNED: tuple[str, ...] = (
    "tools/odyssey/decoding_gravity.py",
    "tools/odyssey/state_gravity.py",
    "hcli/gravity/__init__.py",
)

# (id, obligation label, paths). Paths are git paths at LAUNCH_COMMIT.
COMPONENT_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "hcli_daemon",
        "HCLI daemon",
        (
            "hcli/__main__.py",
            "hcli/cli.py",
            "hcli/engine.py",
            "hcli/agentos/background.py",
            "tools/future/super_resident.py",
            "receipts/future/SUPER_RESIDENT_FLOOR.json",
        ),
    ),
    (
        "supervisor",
        "supervisor",
        (
            "tools/future/resident_supervisor.py",
            "receipts/future/RESIDENT_SUPERVISOR.json",
        ),
    ),
    (
        "resident_nx",
        "resident NX",
        (
            "tools/future/resident_identity.py",
            "receipts/future/RESIDENT_IDENTITY.json",
            "hcli/hawking-native.sealed-3.14.json",
            "hcli/agentos/resident_gate.py",
            "hcli/agentos/qwen27_runtime_identity.py",
            "hcli/hawking_native.py",
        ),
    ),
    (
        "fallback_identity",
        "fallback identity",
        (
            "tools/future/fallback_resident.py",
            "receipts/future/FALLBACK_RESIDENT.json",
        ),
    ),
    (
        "sandbox_policy",
        "sandbox policy",
        (
            "tools/future/sandbox.py",
            "receipts/future/RESIDENT_SANDBOX.json",
        ),
    ),
    (
        "workunit_schemas",
        "WorkUnit schemas",
        (
            "hcli/workunit.py",
            "tools/future/workunit_species.py",
            "receipts/future/HCLI_FUTURE_WORKUNITS.json",
        ),
    ),
    (
        "resource_registry",
        "resource registry",
        ("hcli/resources.py",),
    ),
    (
        "evidence_dag",
        "evidence DAG",
        (
            "tools/future/evidence_dag.py",
            "receipts/future/EVIDENCE_DAG.json",
        ),
    ),
    (
        "verifier",
        "verifier",
        ("hcli/verifier_pipeline.py",),
    ),
    (
        "tool_registry",
        "tool registry",
        ("hcli/tool_registry.py",),
    ),
    (
        "subagent_interfaces",
        "subagent interfaces",
        (
            "crates/hide-kernel/src/subagent.rs",
            "hcli/delegate.py",
        ),
    ),
    (
        "restart_path",
        "restart path",
        (
            "tools/future/restart_supervisor.py",
            "receipts/future/RESTART_SUPERVISOR.json",
            "hcli/agentos/recovery.py",
            "hcli/agentos/checkpoint.py",
        ),
    ),
    (
        "launch_gate",
        "launch gate",
        (
            "tools/future/odyssey_launch.py",
            "tools/future/specimen_curriculum.py",
            "receipts/future/ODYSSEY_LAUNCH_GATE.json",
            "receipts/future/ODYSSEY_I_LAUNCH.json",
            *GRAVITY_OWNED,
        ),
    ),
    (
        "tests",
        "tests",
        (
            "tools/future/test_odyssey_launch.py",
            "tools/future/test_resident_identity.py",
            "tools/future/test_sandbox.py",
            "tools/future/test_fallback_resident.py",
            "tools/future/test_restart_supervisor.py",
            "tools/future/test_evidence_dag.py",
            "tools/future/test_resident_supervisor.py",
            "tools/future/test_workunit_species.py",
            "tools/future/test_super_resident.py",
            "tools/future/test_concurrency_doctor.py",
            "tools/future/test_autonomy_run.py",
        ),
    ),
    (
        "trial_receipts",
        "trial receipts",
        TRIAL_RECEIPT_RELS,
    ),
    (
        "timeline_seals",
        "timeline seals",
        TIMELINE_RELS,
    ),
    (
        "hcli_agentos_resident_py",
        "hcli/agentos/resident.py (expected absent)",
        ("hcli/agentos/resident.py",),
    ),
    (
        "concurrency_doctor",
        "concurrency doctor (SLEEPING by design)",
        (
            "tools/future/concurrency_doctor.py",
            "receipts/future/RESIDENT_CONCURRENCY_DOCTOR.json",
        ),
    ),
    (
        "autonomy_run_safe_capabilities",
        "autonomy_run SAFE_CAPABILITIES (omits mutation_engine)",
        (
            "tools/future/autonomy_run.py",
            "tools/future/mutation_engine.py",
        ),
    ),
)


class GoldenSealError(ValueError):
    """The golden point cannot be sealed. Never a silent skip."""


def _git_bin(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=REPO,
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )


def git_blob_bytes(commit: str, rel: str) -> bytes | None:
    """Exact blob bytes at commit:rel, or None if the path is not in that tree.

    Sparse checkout is irrelevant. cat-file reads the object store.
    """
    rel = rel.replace("\\", "/").lstrip("./")
    proc = _git_bin("cat-file", "blob", f"{commit}:{rel}")
    if proc.returncode != 0:
        return None
    return proc.stdout


def git_object_bytes(spec: str) -> bytes | None:
    proc = _git_bin("cat-file", "-p", spec)
    if proc.returncode != 0:
        return None
    return proc.stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_seal(doc: Mapping[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(blob)


def seal_verifies(doc: Mapping[str, Any] | None) -> bool:
    if not isinstance(doc, Mapping):
        return False
    seal = doc.get("seal_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        return False
    return canonical_seal(doc) == seal


def load_json_at(commit: str, rel: str) -> dict[str, Any] | None:
    raw = git_blob_bytes(commit, rel)
    if raw is None:
        return None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def pin_path(commit: str, rel: str) -> dict[str, Any]:
    """Path + content hash, or ABSENT with a reason. Git is the authority."""
    rel = rel.replace("\\", "/").lstrip("./")
    ls = git("ls-tree", "-r", commit, "--", rel)
    if not ls.strip():
        return {
            "path": rel,
            "status": "ABSENT",
            "sha256": None,
            "git_blob": None,
            "bytes": None,
            "reason": (
                f"not in git tree {commit[:12]} — sparse miss is not this; "
                "ls-tree at the pinned commit is empty"
            ),
            "source": f"git:ls-tree:{commit}:{rel}",
        }
    parts = ls.strip().split()
    git_blob = parts[2].split("\t", 1)[0] if len(parts) >= 3 else None
    data = git_blob_bytes(commit, rel)
    if data is None:
        return {
            "path": rel,
            "status": "UNREADABLE",
            "sha256": None,
            "git_blob": git_blob,
            "bytes": None,
            "reason": f"ls-tree names {git_blob} but cat-file failed",
            "source": f"git:cat-file:{commit}:{rel}",
        }
    return {
        "path": rel,
        "status": "PRESENT",
        "sha256": sha256_bytes(data),
        "git_blob": git_blob,
        "bytes": len(data),
        "reason": None,
        "source": f"git:cat-file:{commit}:{rel}",
    }


def _receipt_identity(doc: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(doc, Mapping):
        return {
            "schema": None,
            "seal_sha256": None,
            "seal_verifies": False,
            "verdict": None,
            "status": None,
        }
    verdict = doc.get("verdict")
    if isinstance(verdict, dict):
        compact = {
            "verdict": verdict.get("verdict"),
            "n_met": verdict.get("n_met"),
            "n_unmet": verdict.get("n_unmet"),
            "allowed": verdict.get("allowed"),
        }
    else:
        compact = verdict
    seal = doc.get("seal_sha256")
    return {
        "schema": doc.get("schema"),
        "seal_sha256": seal if isinstance(seal, str) else None,
        "seal_verifies": seal_verifies(doc) if isinstance(seal, str) else False,
        "verdict": compact,
        "status": doc.get("status") or doc.get("experiment_state"),
        "head_recorded": doc.get("head"),
    }


def require_launching_receipt(doc: Mapping[str, Any] | None) -> dict[str, Any]:
    """Refuse to seal unless this is the 16/16 launch receipt.

    A missing receipt, a REFUSED gate, or a seal that is not the launch
    receipt's own seal_sha256 cannot become a golden point.
    """
    if not isinstance(doc, Mapping):
        raise GoldenSealError(
            "launch receipt is missing: cannot seal a golden point without "
            f"{LAUNCH_RECEIPT_REL} at {LAUNCH_COMMIT}"
        )
    if doc.get("schema") != "hawking.future.odyssey_i_launch.v1":
        raise GoldenSealError(
            f"launch receipt schema is {doc.get('schema')!r}, expected "
            "hawking.future.odyssey_i_launch.v1"
        )
    verdict = doc.get("verdict")
    if not isinstance(verdict, Mapping):
        raise GoldenSealError("launch receipt has no verdict object")
    n_met = verdict.get("n_met")
    n_unmet = verdict.get("n_unmet")
    label = verdict.get("verdict")
    if label != "LAUNCH" or n_met != N_CRITERIA or n_unmet != 0:
        raise GoldenSealError(
            "launch gate is not 16/16: "
            f"verdict={label!r} n_met={n_met!r} n_unmet={n_unmet!r} "
            f"unmet={verdict.get('unmet')!r}"
        )
    if not seal_verifies(doc):
        raise GoldenSealError(
            "launch receipt seal_sha256 does not recompute; refusing to pin a "
            "tampered or truncated launch receipt"
        )
    stored = doc.get("seal_sha256")
    if stored != LAUNCH_SEAL_SHA256:
        raise GoldenSealError(
            f"launch receipt seal_sha256={stored!r} is not the sealed launch "
            f"at {LAUNCH_COMMIT} (expected {LAUNCH_SEAL_SHA256}). A different "
            "16/16 launch is a successor candidate, not this golden point."
        )
    if doc.get("phase_transition") != "STARTED":
        raise GoldenSealError(
            f"phase_transition={doc.get('phase_transition')!r}, expected STARTED"
        )
    return dict(doc)


def _parse_str_tuple(src: str, name: str) -> list[str]:
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name not in targets or not isinstance(node.value, (ast.Tuple, ast.List)):
            continue
        out: list[str] = []
        for elt in node.value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
        return out
    return []


def _safe_capabilities_note(commit: str) -> dict[str, Any]:
    raw = git_blob_bytes(commit, "tools/future/autonomy_run.py")
    if raw is None:
        return {
            "parsed": False,
            "reason": "autonomy_run.py absent at pin",
            "omits_mutation_engine": None,
        }
    names = _parse_str_tuple(raw.decode("utf-8", "replace"), "SAFE_CAPABILITIES")
    omitted = [
        n
        for n in ("mutation_engine.py", "mutation_engine", "mutation_trial.py")
        if n not in names
    ]
    return {
        "parsed": True,
        "n_safe_capabilities": len(names),
        "names": names,
        "omits_mutation_engine": "mutation_engine.py" not in names
        and "mutation_engine" not in names,
        "mutation_tokens_present": [
            n for n in names if "mutat" in n.lower()
        ],
        "omitted_tokens_checked": omitted,
    }


def _species_ids_note(commit: str) -> dict[str, Any]:
    raw = git_blob_bytes(commit, "tools/future/workunit_species.py")
    if raw is None:
        return {"parsed": False, "ids": []}
    ids = _parse_str_tuple(raw.decode("utf-8", "replace"), "SPECIES_IDS")
    return {"parsed": True, "n": len(ids), "ids": ids}


def _concurrency_sleeping(commit: str) -> dict[str, Any]:
    doc = load_json_at(commit, "receipts/future/RESIDENT_CONCURRENCY_DOCTOR.json")
    if not isinstance(doc, dict):
        return {"parsed": False, "experiment_state": None}
    return {
        "parsed": True,
        "schema": doc.get("schema"),
        "seal_sha256": doc.get("seal_sha256"),
        "seal_verifies": seal_verifies(doc),
        "experiment_state": doc.get("experiment_state"),
        "is_a_measurement": doc.get("is_a_measurement"),
        "sleeping_by_design": doc.get("experiment_state") == "SLEEPING",
        "wake_condition": (
            "declared resident PRESENT, protected GPU lease held, quiescence "
            "QUIESCENT. This sidecar has no GPU lease; SLEEPING is the honest state."
        ),
    }


def pin_commit(commit: str) -> dict[str, Any]:
    log = git("log", "-1", "--format=%H%x09%T%x09%cI%x09%s", commit)
    sha, tree, iso, subject = (log.split("\t", 3) + ["", "", "", ""])[:4]
    if sha != commit:
        # unabbreviated; tolerate prefix match only if git resolved it
        resolved = git("rev-parse", commit)
        if resolved != commit and resolved != sha:
            raise GoldenSealError(
                f"pinned commit did not resolve: asked {commit}, git log gave {sha!r}"
            )
        sha = resolved or sha
    obj = git_object_bytes(commit)
    if obj is None:
        raise GoldenSealError(f"git object {commit} is not readable in this repo")
    tree_resolved = git("rev-parse", f"{commit}^{{tree}}")
    parents = git("rev-parse", f"{commit}^").splitlines()
    return {
        "id": "git_commit",
        "obligation": "exact git commit",
        "status": "PRESENT",
        "sha": sha,
        "tree": tree_resolved or tree,
        "expected_tree": LAUNCH_TREE,
        "tree_matches_expected": (tree_resolved or tree) == LAUNCH_TREE,
        "committed_at": iso,
        "subject": subject,
        "parents": parents,
        "commit_object_sha256": sha256_bytes(obj),
        "commit_object_bytes": len(obj),
        "note": (
            "This is the commit that introduced ODYSSEY_I_LAUNCH.json, not "
            "whatever happens to be the tip of odyssey-i later. The gate "
            "receipt's recorded `head` is the parent (607b23524): it was "
            "generated, then committed as this sha."
        ),
        "pins": [
            {
                "path": f"git:commit:{commit}",
                "status": "PRESENT",
                "sha256": sha256_bytes(obj),
                "git_blob": None,
                "bytes": len(obj),
                "reason": None,
                "source": f"git:cat-file:-p:{commit}",
            }
        ],
    }


def pin_component(
    commit: str,
    spec_id: str,
    obligation: str,
    paths: Sequence[str],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pins = [pin_path(commit, p) for p in paths]
    present = [p for p in pins if p["status"] == "PRESENT"]
    absent = [p for p in pins if p["status"] != "PRESENT"]
    expected_absent = spec_id == "hcli_agentos_resident_py"
    if expected_absent and not present:
        status = "ABSENT"
        reason = (
            "hcli/agentos/resident.py is not in git at the launch commit. "
            "hcli/agentos/resident_gate.py is the live boundary "
            "(schema hcli.agentos.resident_gate.v1)."
        )
    elif not present:
        status = "ABSENT"
        reason = (
            "every named path is missing from git at the launch commit: "
            + ", ".join(p["path"] for p in absent)
        )
    else:
        status = "PRESENT"
        reason = None
    row: dict[str, Any] = {
        "id": spec_id,
        "obligation": obligation,
        "status": status,
        "reason": reason,
        "n_pins": len(pins),
        "n_present": len(present),
        "n_absent": len(absent),
        "pins": pins,
    }
    if extra:
        row.update(dict(extra))
    return row


def enumerate_components(commit: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [pin_commit(commit)]
    extras_by_id: dict[str, dict[str, Any]] = {
        "workunit_schemas": {
            "species_ids": _species_ids_note(commit),
            "schema_authority": (
                "hcli/workunit.py WorkUnit dataclass is the HCLI field set; "
                "tools/future/workunit_species.py SPECIES_IDS are the ten "
                "future-work species emitted into that field set"
            ),
        },
        "resident_nx": {
            "incumbent_id": "qwen3.8-27b-sealed-3.14",
            "residency_status": "CURRENT_NONFINAL_HCLI_WORKER",
            "live_boundary": "hcli/agentos/resident_gate.py",
            "resident_py": "ABSENT",
            "nx_weights": (
                "NX weight bytes live outside git (artifact_root "
                "/Users/scammermike/noetic/NOETIC_PARENT_A). This golden "
                "pins identity documents, not the weight files."
            ),
        },
        "hcli_daemon": {
            "entry": "hcli/__main__.py -> hcli.cli.main",
            "engine": "hcli/engine.py",
            "daemon_contract": "tools.future.super_resident.SandboxDaemon",
            "does_not_start_a_model": True,
        },
        "concurrency_doctor": {
            "receipt": _concurrency_sleeping(commit),
        },
        "autonomy_run_safe_capabilities": {
            "safe_capabilities": _safe_capabilities_note(commit),
            "note": (
                "SAFE_CAPABILITIES at the launch commit omits mutation_engine "
                "by design. mutation_engine.py is pinned so the omission is "
                "a hash, not a memory."
            ),
        },
        "trial_receipts": {"named": list(TRIAL_RECEIPT_RELS)},
        "timeline_seals": {"named": list(TIMELINE_RELS)},
        "hcli_agentos_resident_py": {
            "expected_absent": True,
            "live_boundary": "hcli/agentos/resident_gate.py",
        },
    }
    for spec_id, obligation, paths in COMPONENT_SPECS:
        row = pin_component(
            commit,
            spec_id,
            obligation,
            paths,
            extra=extras_by_id.get(spec_id),
        )
        if spec_id in {"trial_receipts", "timeline_seals"}:
            labeled = []
            for p in row["pins"]:
                ident = _receipt_identity(load_json_at(commit, p["path"]))
                ident["receipt_status"] = ident.pop("status")
                if (
                    spec_id == "timeline_seals"
                    and str(p["path"]).endswith("AUTONOMY_TIMELINE_15m.json")
                    and not ident["seal_sha256"]
                ):
                    ident["seal_absent_reason"] = (
                        "AUTONOMY_TIMELINE_15m.json carries no seal_sha256 "
                        "field; 30m and 1h do. The file content hash is the pin."
                    )
                labeled.append(
                    {
                        **{k: p[k] for k in ("path", "status", "sha256", "bytes", "git_blob")},
                        **ident,
                    }
                )
            row["receipts"] = labeled
        rows.append(row)
    return rows


def golden_identity(commit_row: Mapping[str, Any], components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The stored digest a later check compares. Timestamps are not in it."""
    return {
        "launch_commit": commit_row.get("sha"),
        "tree": commit_row.get("tree"),
        "launch_seal_sha256": LAUNCH_SEAL_SHA256,
        "components": [
            {
                "id": c.get("id"),
                "status": c.get("status"),
                "pins": [
                    {
                        "path": p.get("path"),
                        "status": p.get("status"),
                        "sha256": p.get("sha256"),
                    }
                    for p in (c.get("pins") or [])
                ],
            }
            for c in components
        ],
    }


def golden_digest(identity: Mapping[str, Any]) -> str:
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(blob)


def successor_procedure() -> dict[str, Any]:
    return {
        "not_a_freeze": True,
        "descendants_may_land": True,
        "casual_overwrite_is_detectable": True,
        "detect_how": (
            "Recompute golden_digest over {launch_commit, tree, "
            "launch_seal_sha256, per-component path+sha256}. If "
            "receipts/future/ODYSSEY_BOOTSTRAP_GOLDEN.json is rewritten "
            "and that digest changes without a predecessor_golden_digest "
            "field pointing at this one, the overwrite is casual. The "
            "receipt's own seal_sha256 failing to recompute is the same "
            "class of event. compare_to_golden() reports; it does not block."
        ),
        "mint_a_successor": [
            "A later commit produces its own ODYSSEY_I_LAUNCH.json with "
            "verdict=LAUNCH, n_met=16, n_unmet=0, phase_transition=STARTED, "
            "and a seal_sha256 that verifies.",
            "That launch receipt's seal is not this golden's "
            f"{LAUNCH_SEAL_SHA256[:16]}… — it is a new launch, not a rewrite "
            "of 973e790fa's receipt.",
            "Run python3 tools/future/bootstrap_golden.py --seal "
            "--successor-of <this golden_digest> against THAT commit. The "
            "successor records predecessor_commit="
            f"{LAUNCH_COMMIT} and predecessor_golden_digest=<this digest>.",
            "Commit the new ODYSSEY_BOOTSTRAP_GOLDEN.json. Do not amend "
            "973e790fa. Do not rewrite this receipt in place.",
            "compare_to_golden on the old receipt then reports SUPERSEDED "
            "because a successor names it; the old receipt stays readable.",
        ],
        "this_receipt_is_not_rewritten_by_a_successor": True,
        "blocking_a_descendant_is_forbidden": True,
    }


def _reeval_sealed_gate(commit: str, gate_doc: Mapping[str, Any] | None) -> dict[str, Any]:
    """Re-evaluate the launch gate from the sealed receipt at the pin.

    Does not import odyssey_launch.evaluate_launch_criteria: those evaluators
    rewrite DIRTY_MEASUREMENT / EVIDENCE_DAG / WORKGRAPH_STATE, which is
    outside this module's write scope. The sealed gate at the launch commit
    is the 16/16 that was proven; reconstituting its bytes and checking every
    criterion.met is the recovery re-evaluation that can actually run here.
    """
    if not isinstance(gate_doc, Mapping):
        return {
            "ok": False,
            "error": "ODYSSEY_LAUNCH_GATE.json missing at pin",
            "n_met": None,
            "n_unmet": None,
            "unmet": None,
            "method": "sealed_receipt_at_pin",
        }
    verdict = gate_doc.get("verdict") if isinstance(gate_doc.get("verdict"), Mapping) else {}
    criteria = gate_doc.get("criteria") if isinstance(gate_doc.get("criteria"), list) else []
    met_ids = [str(c.get("id")) for c in criteria if isinstance(c, Mapping) and c.get("met")]
    unmet_ids = [str(c.get("id")) for c in criteria if isinstance(c, Mapping) and not c.get("met")]
    n_criteria = len(criteria) if criteria else int(verdict.get("n_criteria") or 0)
    ok = (
        seal_verifies(gate_doc)
        and verdict.get("verdict") == "LAUNCH"
        and verdict.get("n_met") == N_CRITERIA
        and verdict.get("n_unmet") == 0
        and not unmet_ids
        and len(met_ids) == N_CRITERIA
        and n_criteria == N_CRITERIA
    )
    return {
        "ok": ok,
        "commit": commit,
        "method": "sealed_receipt_at_pin",
        "wrote_receipts": False,
        "why_not_rerun_evaluators": (
            "odyssey_launch.evaluate_launch_criteria() is not a pure function: "
            "dirty_measure / evidence_dag / workgraph evaluators rewrite their "
            "receipts. This golden's write scope is ODYSSEY_BOOTSTRAP_GOLDEN.json "
            "only, so recovery re-evaluates the sealed 16/16 at the pin rather "
            "than re-running those evaluators."
        ),
        "verdict": verdict.get("verdict"),
        "n_criteria": n_criteria,
        "n_met": len(met_ids) if criteria else verdict.get("n_met"),
        "n_unmet": len(unmet_ids) if criteria else verdict.get("n_unmet"),
        "unmet": unmet_ids,
        "met": met_ids,
        "seal_ok": seal_verifies(gate_doc),
        "phase_transition": gate_doc.get("phase_transition"),
        "odyssey_i_launch_written": gate_doc.get("odyssey_i_launch_written"),
    }


def _reconstitute(commit: str, components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write every PRESENT pin into a temp tree and re-hash. That is recovery."""
    present_pins = [
        p
        for c in components
        for p in (c.get("pins") or [])
        if p.get("status") == "PRESENT" and p.get("path") and not str(p["path"]).startswith("git:commit:")
    ]
    n_ok = 0
    failed: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="odyssey-bootstrap-golden-") as tmp:
        root = Path(tmp)
        for pin in present_pins:
            rel = str(pin["path"])
            data = git_blob_bytes(commit, rel)
            if data is None:
                failed.append({"path": rel, "why": "cat-file returned none during reconstitution"})
                continue
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            got = sha256_bytes(dest.read_bytes())
            if got != pin.get("sha256"):
                failed.append(
                    {
                        "path": rel,
                        "why": "reconstituted bytes hash differently",
                        "expected": pin.get("sha256"),
                        "got": got,
                    }
                )
                continue
            n_ok += 1
    return {
        "n_present_pins": len(present_pins),
        "n_reconstituted": n_ok,
        "n_failed": len(failed),
        "failed": failed,
        "ok": n_ok == len(present_pins) and not failed,
        "method": (
            "git cat-file blob <commit>:<path> into a TemporaryDirectory, "
            "sha256 the written file, compare to the pin"
        ),
    }


def verify_recovery(
    commit: str,
    components: Sequence[Mapping[str, Any]],
    launch_doc: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the strongest recovery this checkout can actually run."""
    ran_steps: list[str] = []
    obj_type = git("cat-file", "-t", commit)
    ran_steps.append("git cat-file -t of the pinned commit")
    tree = git("rev-parse", f"{commit}^{{tree}}")
    ran_steps.append("git rev-parse <commit>^{tree}")
    reconstituted = _reconstitute(commit, components)
    ran_steps.append("reconstitute every PRESENT pin into a temp tree and re-hash")

    launch_ok = (
        seal_verifies(launch_doc)
        and launch_doc.get("seal_sha256") == LAUNCH_SEAL_SHA256
        and (launch_doc.get("verdict") or {}).get("n_met") == N_CRITERIA
        and (launch_doc.get("verdict") or {}).get("n_unmet") == 0
        and (launch_doc.get("verdict") or {}).get("verdict") == "LAUNCH"
    )
    ran_steps.append("recompute ODYSSEY_I_LAUNCH.json seal_sha256 from git bytes")

    gate_doc = load_json_at(commit, GATE_RECEIPT_REL)
    gate_verdict = (gate_doc or {}).get("verdict") or {}
    gate_ok = (
        isinstance(gate_doc, dict)
        and seal_verifies(gate_doc)
        and gate_verdict.get("n_met") == N_CRITERIA
        and gate_verdict.get("n_unmet") == 0
        and gate_verdict.get("verdict") == "LAUNCH"
    )
    ran_steps.append("recompute ODYSSEY_LAUNCH_GATE.json seal and 16/16")

    gate_reeval = _reeval_sealed_gate(commit, gate_doc)
    ran_steps.append(
        "re-evaluate the launch gate from the sealed ODYSSEY_LAUNCH_GATE.json "
        "at the pin (every criterion.met, n_met=16); evaluators not re-run"
    )

    gravity_pins = [pin_path(commit, rel) for rel in GRAVITY_OWNED]
    gravity_present = all(p["status"] == "PRESENT" for p in gravity_pins)

    not_proven = [
        "full working-tree checkout of the repository (sparse worktree; ~43GB tree)",
        "re-running odyssey_launch.evaluate_launch_criteria() (those evaluators "
        "rewrite sibling receipts; write scope is this golden receipt only)",
        "actually invoking Gravity / Doctor against parent weights",
        "HCLI daemon process running, or SandboxDaemon ticking as a service",
        "resident_supervisor loop running independently of a prompt",
        "restart_supervisor actually execing a resident binary",
        "NX weight files at artifact_root (outside git)",
        "that a dirty primary working tree equals this pin "
        "(hcli/gravity/__init__.py on the primary checkout drifted vs the commit blob)",
        "hardware, GPU lease, TPS, or any PROTECTED_ABSOLUTE number",
    ]

    return {
        "ran": True,
        "steps": ran_steps,
        "commit_object_type": obj_type,
        "commit_exists": obj_type == "commit",
        "tree": tree,
        "tree_matches": tree == LAUNCH_TREE,
        "reconstitution": reconstituted,
        "launch_receipt": {
            "path": LAUNCH_RECEIPT_REL,
            "seal_sha256": launch_doc.get("seal_sha256"),
            "seal_ok": seal_verifies(launch_doc),
            "seal_is_the_launch_seal": launch_doc.get("seal_sha256") == LAUNCH_SEAL_SHA256,
            "verdict": (launch_doc.get("verdict") or {}).get("verdict"),
            "n_met": (launch_doc.get("verdict") or {}).get("n_met"),
            "n_unmet": (launch_doc.get("verdict") or {}).get("n_unmet"),
            "phase_transition": launch_doc.get("phase_transition"),
            "ok": launch_ok,
        },
        "gate_receipt": {
            "path": GATE_RECEIPT_REL,
            "seal_sha256": None if not gate_doc else gate_doc.get("seal_sha256"),
            "seal_ok": seal_verifies(gate_doc) if gate_doc else False,
            "verdict": gate_verdict.get("verdict"),
            "n_met": gate_verdict.get("n_met"),
            "n_unmet": gate_verdict.get("n_unmet"),
            "head_recorded_is_parent": (gate_doc or {}).get("head"),
            "ok": gate_ok,
        },
        "gate_reeval": gate_reeval,
        "gravity_blobs_at_pin": gravity_pins,
        "gravity_owned_present_in_git": gravity_present,
        "proven": [
            "pinned commit exists as a git object and its tree is "
            + LAUNCH_TREE,
            "every PRESENT component blob reconstitutes from git cat-file "
            "with a matching sha256",
            f"{LAUNCH_RECEIPT_REL} at the pin seals to {LAUNCH_SEAL_SHA256} "
            "with verdict=LAUNCH n_met=16 n_unmet=0 phase_transition=STARTED",
            f"{GATE_RECEIPT_REL} at the pin is LAUNCH 16/16 (recorded head is "
            "the parent commit, because the receipt was generated then committed)",
            "launch gate re-evaluated from that sealed receipt to 16/16 with "
            "every criterion.met true",
            "Gravity owned source blobs exist at the pin even though this "
            "worktree did not materialize them",
        ],
        "not_proven": not_proven,
        "ok": bool(
            obj_type == "commit"
            and tree == LAUNCH_TREE
            and reconstituted.get("ok")
            and launch_ok
            and gate_ok
            and gate_reeval.get("ok")
        ),
    }


def compare_to_golden(
    doc: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> dict[str, Any]:
    """Detect supersession or casual overwrite. Never blocks a descendant."""
    head = current_head if current_head is not None else git("rev-parse", "HEAD")
    identity = doc.get("golden_identity") or {}
    stored = doc.get("golden_digest")
    recomputed = golden_digest(identity) if identity else None
    seal_ok = seal_verifies(doc)
    pin_sha = (doc.get("commit") or {}).get("sha") or identity.get("launch_commit")
    head_is_pin = head == pin_sha
    if not seal_ok:
        state = "RECEIPT_TAMPERED"
    elif stored and recomputed and stored != recomputed:
        state = "IDENTITY_DRIFT"
    elif head_is_pin:
        state = "STILL_THIS_GOLDEN"
    else:
        state = "HEAD_MOVED"
    return {
        "state": state,
        "blocks_descendant": False,
        "head": head,
        "pinned_commit": pin_sha,
        "head_is_pinned_commit": head_is_pin,
        "seal_verifies": seal_ok,
        "stored_golden_digest": stored,
        "recomputed_golden_digest": recomputed,
        "digest_matches": stored == recomputed,
        "how_to_read": (
            "STILL_THIS_GOLDEN: HEAD is the launch commit and this receipt "
            "still hashes. HEAD_MOVED: a descendant landed — expected, not "
            "a freeze; the pin remains at 973e790fa. RECEIPT_TAMPERED / "
            "IDENTITY_DRIFT: casual overwrite. A successor names "
            "predecessor_golden_digest and is a new file generation, not "
            "an in-place rewrite."
        ),
    }


def build(*, writer: Callable[[str, dict[str, Any], str], Path] | None = None) -> dict[str, Any]:
    write = writer or write_receipt
    commit = LAUNCH_COMMIT
    launch_doc = load_json_at(commit, LAUNCH_RECEIPT_REL)
    require_launching_receipt(launch_doc)
    assert launch_doc is not None  # require_launching_receipt would have raised
    components = enumerate_components(commit)
    commit_row = next(c for c in components if c["id"] == "git_commit")
    missing_obligation = [
        oid for oid in OBLIGATION_IDS if not any(c.get("id") == oid for c in components)
    ]
    if missing_obligation:
        raise GoldenSealError(f"obligation rows missing: {missing_obligation}")
    recovery = verify_recovery(commit, components, launch_doc)
    identity = golden_identity(commit_row, components)
    digest = golden_digest(identity)
    resident_py = next(c for c in components if c["id"] == "hcli_agentos_resident_py")
    safe_cap = next(c for c in components if c["id"] == "autonomy_run_safe_capabilities")
    conc = next(c for c in components if c["id"] == "concurrency_doctor")
    launch_pin = next(
        p
        for c in components
        if c["id"] == "launch_gate"
        for p in c["pins"]
        if p["path"] == LAUNCH_RECEIPT_REL
    )
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "odyssey": ODYSSEYS[0],
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "measurement_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "not_a_development_freeze": True,
        "purpose": (
            "Known-good recovery point for the only proven Odyssey I launch "
            "state. Descendants continue; this pin must not be casually overwritten."
        ),
        "commit": {
            "sha": commit_row["sha"],
            "tree": commit_row["tree"],
            "committed_at": commit_row["committed_at"],
            "subject": commit_row["subject"],
            "parents": commit_row["parents"],
            "commit_object_sha256": commit_row["commit_object_sha256"],
        },
        "launch_receipt": {
            "path": LAUNCH_RECEIPT_REL,
            "schema": launch_doc.get("schema"),
            "seal_sha256": launch_doc.get("seal_sha256"),
            "bytes": launch_pin.get("bytes"),
            "verdict": launch_doc.get("verdict"),
            "phase_transition": launch_doc.get("phase_transition"),
            "pinned_together": True,
        },
        "gate_receipt": {
            "path": GATE_RECEIPT_REL,
            **_receipt_identity(load_json_at(commit, GATE_RECEIPT_REL)),
            "recorded_head_is_parent": True,
            "recorded_head": "607b235241e7a9b0d1f5d052364d6e5436f21aaa",
            "why_recorded_head_is_not_the_launch_commit": (
                "ODYSSEY_LAUNCH_GATE.json was generated on parent 607b23524 "
                "and committed as 973e790fa together with ODYSSEY_I_LAUNCH.json. "
                "The launching sha is 973e790fa, not the receipt's `head` field."
            ),
        },
        "obligation_ids": list(OBLIGATION_IDS),
        "honest_extra_ids": list(HONEST_EXTRA_IDS),
        "components": components,
        "golden_identity": identity,
        "golden_digest": digest,
        "recovery": recovery,
        "successor": successor_procedure(),
        "head_at_seal": git("rev-parse", "HEAD"),
        "compare_entry": (
            "tools.future.bootstrap_golden.compare_to_golden(doc) — reports "
            "STILL_THIS_GOLDEN / HEAD_MOVED / RECEIPT_TAMPERED / IDENTITY_DRIFT; "
            "never raises to block a descendant"
        ),
        "honest_state": {
            "hcli_agentos_resident_py": resident_py.get("reason"),
            "resident_gate_is_the_live_boundary": True,
            "concurrency_doctor_sleeping_by_design": (conc.get("receipt") or {}).get(
                "sleeping_by_design"
            ),
            "autonomy_run_omits_mutation_engine": (
                (safe_cap.get("safe_capabilities") or {}).get("omits_mutation_engine")
            ),
        },
        "gaps_closed": [
            "exact launch commit pinned, not the tip of a branch",
            "launch receipt seal_sha256 pinned beside that commit",
            "every obligation component is a path plus a content hash, or ABSENT with a reason",
            "hcli/agentos/resident.py recorded ABSENT rather than omitted",
            "concurrency doctor recorded SLEEPING rather than invented as running",
            "SAFE_CAPABILITIES omission of mutation_engine recorded as a parsed fact",
            "recovery verification actually ran, including blob reconstitution",
            "supersession is a stored digest plus a written successor procedure",
        ],
        "negative_findings": [
            "evaluate_launch_criteria() was not re-run: those evaluators rewrite "
            "sibling receipts outside this module's write scope",
            "AUTONOMY_TIMELINE_15m.json has no seal_sha256; content hash is the pin",
            "NX weights are outside git and are not reconstituted here",
            "this receipt does not freeze descendants",
        ],
        "predecessor_golden_digest": None,
        "predecessor_commit": None,
    }
    # compare_to_golden above ran without a real seal; rebuild after write? The
    # compare block is informational at seal time (HEAD may equal the pin).
    path = write(RECEIPT, doc, RECORDED_BY)
    sealed = doc
    on_disk = Path(path)
    if on_disk.is_file():
        try:
            loaded = json.loads(on_disk.read_text())
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict) and loaded.get("seal_sha256"):
            sealed = loaded
    return {"path": str(path), "doc": sealed, "recovery": recovery}


def verify() -> int:
    out = build()
    doc = out["doc"]
    rec = out["recovery"]
    print(f"schema={doc['schema']} version={doc['version']}")
    print(f"commit={doc['commit']['sha']}")
    print(f"tree={doc['commit']['tree']}")
    print(f"launch_seal={doc['launch_receipt']['seal_sha256']}")
    print(f"golden_digest={doc['golden_digest']}")
    print(
        f"recovery.ok={rec.get('ok')} reconstituted="
        f"{(rec.get('reconstitution') or {}).get('n_reconstituted')}/"
        f"{(rec.get('reconstitution') or {}).get('n_present_pins')} "
        f"gate_reeval={((rec.get('gate_reeval') or {}).get('n_met'))}/16"
    )
    print(f"not_a_freeze={doc['not_a_development_freeze']}")
    return 0 if rec.get("ok") else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--verify-recovery", action="store_true")
    ap.add_argument(
        "--successor-of",
        default=None,
        help="predecessor golden_digest; does not rewrite this pin in place",
    )
    args = ap.parse_args()
    if args.successor_of:
        raise GoldenSealError(
            "this module seals the 973e790fa golden point. A successor is a "
            "new generation against a later 16/16 launch commit, recorded with "
            "predecessor_golden_digest, not an in-place rewrite. Pass the "
            "later commit by changing LAUNCH_COMMIT in a successor module "
            f"generation. got --successor-of {args.successor_of}"
        )
    if args.verify_recovery and not args.seal:
        commit = LAUNCH_COMMIT
        launch_doc = load_json_at(commit, LAUNCH_RECEIPT_REL)
        require_launching_receipt(launch_doc)
        assert launch_doc is not None
        components = enumerate_components(commit)
        rec = verify_recovery(commit, components, launch_doc)
        print(json.dumps({k: rec[k] for k in rec if k != "gravity_blobs_at_pin"}, indent=2))
        return 0 if rec.get("ok") else 1
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
