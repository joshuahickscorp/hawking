#!/usr/bin/env python3.12
"""Reconcile the live Kimi closure into the GLM-5.2 campaign handoff.

This tool is deliberately append-only in meaning: it does not rewrite the
historical Kimi closure, claim a second deletion, stop a process, or touch a
model cache.  It verifies the sealed science, records the already-completed
source release, and makes any degraded rollback/security boundary explicit.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from glm52_common import (  # noqa: E402
    Glm52Error,
    REPO_ROOT,
    atomic_json,
    atomic_text,
    canonical,
    read_json,
    seal,
    sha256_file,
    utc_now,
    verify_sealed,
)


KIMI_REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
KIMI_CLOSURE_COMMIT = "0210e5aa05f0e3c69d6f2022c539c9dc90cce322"
KIMI_CLEANSE_COMMIT = "8d634af9702aa965728f057661b5d1fad1883f45"
OLD_ARCHIVE_SHA256 = "7e48036324cf2cfa307eb839f542e87ba2998f4777c2e39aa589d2b71c4fda7d"
SOURCE_ROOT = Path.home() / ".cache/huggingface/hub/models--moonshotai--Kimi-K2.6"
RUNTIME_ROOT = Path.home() / "Library/Application Support/Hawking/KimiK26"
PLIST = Path.home() / "Library/LaunchAgents/com.hawking.kimi-k26-doctor-prime.plist"
ARCHIVE = REPO_ROOT / "reports/condense/kimi_k26/KIMI_K26_RUNTIME_PROGRESS_ARCHIVE.zip"
OFFICIAL_MANIFEST = REPO_ROOT / "reports/condense/kimi_k26/KIMI_K26_OFFICIAL_MANIFEST.json"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _seal_ok(path: Path) -> dict[str, Any]:
    value = read_json(path)
    verify_sealed(value, label=str(path))
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "file_sha256": sha256_file(path),
        "seal_sha256": value["seal_sha256"],
        "seal_verified": True,
    }


def _verify_evidence_chain(final: dict[str, Any]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    chain = final.get("evidence_chain")
    if not isinstance(chain, dict) or len(chain) != 9:
        raise Glm52Error(f"expected nine Kimi evidence-chain rows, found {chain!r}")
    for evidence_id, row in sorted(chain.items()):
        if not isinstance(row, dict) or not row.get("path"):
            raise Glm52Error(f"invalid evidence-chain row: {row!r}")
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = REPO_ROOT / path
        value = read_json(path)
        verify_sealed(value, label=str(path))
        if value["seal_sha256"] != row.get("seal_sha256"):
            raise Glm52Error(f"evidence-chain seal pointer mismatch: {path}")
        verified.append({
            "evidence_id": evidence_id,
            "path": str(path.relative_to(REPO_ROOT)),
            "seal_sha256": value["seal_sha256"],
        })
    return verified


def _archive_state(rotation_receipt: Path | None) -> dict[str, Any]:
    if not ARCHIVE.is_file():
        raise Glm52Error(f"sanitized Kimi evidence archive is absent: {ARCHIVE}")
    with zipfile.ZipFile(ARCHIVE) as handle:
        names = handle.namelist()
        integrity_error = handle.testzip()
    credential_entry_present = ".telegram_creds.json" in names
    if credential_entry_present:
        raise Glm52Error("credential-bearing .telegram_creds.json remains in current archive")
    rotation: dict[str, Any] = {
        "status": "REQUIRED_EXTERNAL_ACTION",
        "confirmed": False,
        "receipt": None,
        "reason": "The credential was pushed in Git history; file sanitization cannot revoke it.",
    }
    if rotation_receipt is not None:
        receipt = read_json(rotation_receipt)
        if not bool(receipt.get("telegram_bot_token_rotated")):
            raise Glm52Error("rotation receipt does not confirm telegram_bot_token_rotated=true")
        rotation = {
            "status": "CONFIRMED_BY_OPERATOR_RECEIPT",
            "confirmed": True,
            "receipt": str(rotation_receipt),
            "receipt_sha256": sha256_file(rotation_receipt),
        }
    remote_contains = [
        line.strip()
        for line in _git("branch", "-r", "--contains", KIMI_CLEANSE_COMMIT).splitlines()
        if line.strip()
    ]
    return {
        "path": str(ARCHIVE.relative_to(REPO_ROOT)),
        "old_pushed_sha256": OLD_ARCHIVE_SHA256,
        "sanitized_sha256": sha256_file(ARCHIVE),
        "sanitized_bytes": ARCHIVE.stat().st_size,
        "entry_count": len(names),
        "removed_exact_entry": ".telegram_creds.json",
        "credential_entry_present_now": credential_entry_present,
        "zip_integrity_error": integrity_error,
        "remote_branches_containing_exposed_commit": remote_contains,
        "history_purge_performed": False,
        "history_purge_requires_explicit_coordination": True,
        "rotation": rotation,
    }


def _process_absence() -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    matches = []
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if "kimi-k2.6" in lowered or "kimi_k26" in lowered or "kimi-k26" in lowered:
            if "glm52_precheck.py" not in lowered:
                matches.append(line.strip())
    return {
        "matching_processes": matches,
        "matching_process_count": len(matches),
        "former_pids": {
            "73361_alive": _pid_alive(73361),
            "73362_alive": _pid_alive(73362),
        },
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def build(rotation_receipt: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    final_path = REPO_ROOT / "KIMI_K26_GRAVITY_FINAL.json"
    auction_path = REPO_ROOT / "KIMI_K26_FINAL_BYTE_AUCTION.json"
    final = read_json(final_path)
    auction = read_json(auction_path)
    verify_sealed(final, label=str(final_path))
    verify_sealed(auction, label=str(auction_path))
    chain = _verify_evidence_chain(final)

    if final.get("terminal_outcome") != "OUTCOME_C" or final.get("status") != "CLOSED":
        raise Glm52Error("Kimi final outcome is not sealed CLOSED/OUTCOME_C")
    if auction.get("status") not in {"PASS", "COMPLETE", "CLOSED"}:
        raise Glm52Error(f"Kimi byte auction is not complete: {auction.get('status')}")

    cleanse_path = REPO_ROOT / "KIMI_K26_DEVICE_CLEANSE_FINAL.json"
    cleanse = read_json(cleanse_path)
    verify_sealed(cleanse, label=str(cleanse_path))
    manifest = read_json(OFFICIAL_MANIFEST)
    verify_sealed(manifest, label=str(OFFICIAL_MANIFEST))
    archive = _archive_state(rotation_receipt)
    processes = _process_absence()

    if SOURCE_ROOT.exists() or RUNTIME_ROOT.exists() or PLIST.exists():
        raise Glm52Error("Kimi source/runtime/plist unexpectedly exists after sealed cleanse")
    if processes["matching_process_count"]:
        raise Glm52Error(f"live Kimi processes remain: {processes['matching_processes']}")

    exact_source_allocated = int(cleanse["deleted"]["kimi_k2_6_source_allocated_bytes"])
    all_kimi_roots_allocated = 595_205_173_248
    other_skeletons = all_kimi_roots_allocated - exact_source_allocated
    if other_skeletons != 28_672:
        raise Glm52Error("Kimi allocated-byte reconciliation changed unexpectedly")

    remediation = seal({
        "schema": "hawking.kimi_k26.credential_remediation_for_glm52.v1",
        "status": "SANITIZED_CURRENT_TREE_ROTATION_PENDING" if not archive["rotation"]["confirmed"] else "SANITIZED_AND_ROTATED",
        "created_at": utc_now(),
        "incident": {
            "finding": "The pushed Kimi runtime evidence ZIP contained a nonempty Telegram bot token and chat ID despite a no-credentials claim.",
            "credential_values_read_or_recorded_by_this_artifact": False,
            "affected_commit": KIMI_CLEANSE_COMMIT,
        },
        "archive": archive,
        "historical_cleanup_receipt": {
            "path": str(cleanse_path.relative_to(REPO_ROOT)),
            "seal_sha256": cleanse["seal_sha256"],
            "archive_hash_claim_now_superseded": OLD_ARCHIVE_SHA256,
            "historical_receipt_not_rewritten": True,
        },
        "publication_gate": {
            "telegram_delivery_allowed": bool(archive["rotation"]["confirmed"]),
            "git_history_purge": "NOT_PERFORMED_REQUIRES_EXPLICIT_COORDINATION",
        },
    })

    release = seal({
        "schema": "hawking.kimi_k26.source_release_for_glm52.v1",
        "status": "RECONCILED_ALREADY_RELEASED",
        "created_at": utc_now(),
        "release_timing": "COMPLETED_BEFORE_THIS_GLM52_GOAL",
        "new_deletion_performed_by_this_receipt": False,
        "authorization": "The GLM-5.2 campaign authorizes Kimi raw-source release after closure gates; live evidence proves the release had already occurred.",
        "source": {
            "repo": "moonshotai/Kimi-K2.6",
            "revision": KIMI_REVISION,
            "former_root": str(SOURCE_ROOT),
            "former_snapshot": str(SOURCE_ROOT / "snapshots" / KIMI_REVISION),
            "manifest_path": str(OFFICIAL_MANIFEST.relative_to(REPO_ROOT)),
            "manifest_seal_sha256": manifest["seal_sha256"],
            "manifest_files": int(manifest["file_count"]),
            "weight_shards": int(manifest["weight_shards"]),
            "logical_bytes": int(manifest["total_bytes"]),
            "weight_bytes": int(manifest["weight_bytes"]),
            "exists_now": SOURCE_ROOT.exists(),
        },
        "allocated_byte_reconciliation": {
            "exact_kimi_k2_6_source_root_allocated_bytes": exact_source_allocated,
            "all_four_kimi_named_hf_roots_allocated_bytes": all_kimi_roots_allocated,
            "other_three_cache_skeletons_allocated_bytes": other_skeletons,
            "total_cleanup_free_space_delta_bytes": int(cleanse["execution"]["immediate_free_bytes_delta"]),
            "total_delta_includes_runtime_and_other_targets": True,
            "filesystem_free_delta_is_not_relabelled_as_source_only": True,
        },
        "closure": {
            "closure_commit": KIMI_CLOSURE_COMMIT,
            "cleanse_commit": KIMI_CLEANSE_COMMIT,
            "terminal_outcome": final["terminal_outcome"],
            "final_seal_sha256": final["seal_sha256"],
            "byte_auction_seal_sha256": auction["seal_sha256"],
            "evidence_chain": chain,
        },
        "live_absence": {
            "source_absent": not SOURCE_ROOT.exists(),
            "runtime_absent": not RUNTIME_ROOT.exists(),
            "installed_plist_absent": not PLIST.exists(),
            "process_audit": processes,
            "queue_or_outbox_possible": False,
        },
        "rehydration": {
            "automatic": False,
            "command": (
                "HF_HUB_DISABLE_XET=0 HF_XET_HIGH_PERFORMANCE=1 "
                "/Library/Frameworks/Python.framework/Versions/3.12/bin/hf download "
                f"moonshotai/Kimi-K2.6 --revision {KIMI_REVISION} "
                "--cache-dir /Users/scammermike/.cache/huggingface/hub "
                "--max-workers 8 --format agent"
            ),
        },
        "rollback_boundary": {
            "best_local_payload_preserved": False,
            "runtime_preserved": False,
            "implementation_recoverable_from_git_history": True,
            "scientific_reports_preserved": True,
            "reproducibility_capsule_status": "DEGRADED_BY_PRIOR_BROAD_CLEANSE",
        },
        "credential_remediation_seal_sha256": remediation["seal_sha256"],
    })

    head = _git("rev-parse", "HEAD")
    handoff = seal({
        "schema": "hawking.glm52.handoff_precheck.v1",
        "status": "PASS_WITH_SECURITY_AND_ROLLBACK_EXCEPTIONS",
        "created_at": utc_now(),
        "repository": {
            "path": str(REPO_ROOT),
            "head": head,
            "branch": _git("branch", "--show-current"),
            "origin_contains_head": bool(_git("branch", "-r", "--contains", head)),
        },
        "kimi_science": {
            "terminal_outcome": final["terminal_outcome"],
            "final": _seal_ok(final_path),
            "final_markdown": {
                "path": "KIMI_K26_GRAVITY_FINAL.md",
                "sha256": sha256_file(REPO_ROOT / "KIMI_K26_GRAVITY_FINAL.md"),
                "integrity_authority": KIMI_CLOSURE_COMMIT,
            },
            "byte_auction": _seal_ok(auction_path),
            "next_parent_transfer": {
                "path": "KIMI_K26_NEXT_PARENT_TRANSFER.md",
                "sha256": sha256_file(REPO_ROOT / "KIMI_K26_NEXT_PARENT_TRANSFER.md"),
                "integrity_authority": KIMI_CLOSURE_COMMIT,
            },
            "evidence_chain_verified": chain,
            "primary_diagnosis": final["causal_diagnosis"]["diagnosis"],
            "next_parent_action": final["required_final_summary"]["recommended next parent action"],
        },
        "kimi_source_release": {
            "state": "ALREADY_RELEASED_AND_RECONCILED",
            "receipt_seal_sha256": release["seal_sha256"],
            "new_deletion_performed": False,
        },
        "security_gate": {
            "current_archive_sanitized": True,
            "telegram_token_rotation_confirmed": bool(archive["rotation"]["confirmed"]),
            "telegram_reporting": "ENABLED" if archive["rotation"]["confirmed"] else "QUARANTINED_PENDING_ROTATION",
            "remote_history_exposure": True,
            "history_purge": "NOT_AUTHORIZED_OR_PERFORMED",
            "remediation_seal_sha256": remediation["seal_sha256"],
        },
        "rollback_exception": release["rollback_boundary"],
        "process_state": processes,
        "admission_decision": {
            "safe_to_begin_local_glm_metadata_and_implementation": True,
            "safe_to_send_telegram": bool(archive["rotation"]["confirmed"]),
            "safe_to_start_bf16_stream": False,
            "bf16_stream_blockers": [
                "GLM adapter/twin/reference parity not yet green",
                "real Xet dependency-window fetch/resume/evict path not yet green",
                "physical compact writer/reader/direct execution not yet green",
                "Xet current-version/autotune gate not yet green",
            ],
        },
    })
    return remediation, release, handoff


def render_markdown(remediation: dict[str, Any], release: dict[str, Any], handoff: dict[str, Any]) -> str:
    rotation = remediation["archive"]["rotation"]
    return "\n".join([
        "# GLM-5.2 handoff precheck",
        "",
        f"Status: **{handoff['status']}**",
        "",
        "## Kimi closure",
        "",
        f"- Science: `{handoff['kimi_science']['terminal_outcome']}`; all nine JSON evidence seals verified.",
        f"- Primary diagnosis: `{handoff['kimi_science']['primary_diagnosis']}`.",
        f"- Source: already released before this goal; no new deletion was performed.",
        f"- Exact former K2.6 source allocation: `{release['allocated_byte_reconciliation']['exact_kimi_k2_6_source_root_allocated_bytes']}` bytes.",
        f"- Release receipt seal: `{release['seal_sha256']}`.",
        "",
        "## Exceptions",
        "",
        "- The prior device cleanse was broader than the GLM handoff contract and removed the retained local payload/runtime; reproducibility is degraded but the scientific reports and Git history remain.",
        f"- The current evidence ZIP is sanitized (`{remediation['archive']['sanitized_sha256']}`), but the old credential-bearing ZIP remains reachable from pushed history.",
        f"- Telegram token rotation: `{rotation['status']}`. Telegram delivery stays quarantined until an operator rotation receipt is supplied.",
        "- Git-history rewriting was not authorized or performed.",
        "",
        "## GLM admission boundary",
        "",
        "Local metadata/header/accounting and implementation work may proceed. BF16 shard streaming remains blocked until the adapter/reference, real Xet window engine, physical codec/direct runtime, and current-Xet autotune gates are green.",
        "",
        f"Handoff seal: `{handoff['seal_sha256']}`.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram-rotation-receipt", type=Path)
    args = parser.parse_args()
    remediation, release, handoff = build(args.telegram_rotation_receipt)
    atomic_json(REPO_ROOT / "KIMI_K26_CREDENTIAL_REMEDIATION_FOR_GLM52.json", remediation)
    atomic_json(REPO_ROOT / "KIMI_K26_SOURCE_RELEASE_FOR_GLM52.json", release)
    atomic_json(REPO_ROOT / "GLM52_HANDOFF_PRECHECK.json", handoff)
    atomic_text(
        REPO_ROOT / "GLM52_HANDOFF_PRECHECK.md",
        render_markdown(remediation, release, handoff),
    )
    print(json.dumps({
        "status": handoff["status"],
        "handoff_seal_sha256": handoff["seal_sha256"],
        "release_seal_sha256": release["seal_sha256"],
        "remediation_seal_sha256": remediation["seal_sha256"],
        "telegram_rotation": remediation["archive"]["rotation"]["status"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
