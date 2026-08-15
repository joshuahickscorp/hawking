"""Detached, resumable acquisition and verification of the pinned Qwen80 source.

This worker exists solely to materialize the official raw BF16 source authority
that the Ascension campaign has already admitted.  It does *not* load the
model, produce a Gravity artifact, or change any tournament/manager state.
Every listed file is downloaded at the immutable commit, size checked, and
SHA-256 hashed before a sealed source-body audit is emitted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
DEFAULT_METADATA = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/lifecycle/source-admission/QWEN80_SOURCE_METADATA_CANDIDATE.json"
DEFAULT_TARGET = REPO_ROOT / "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
DEFAULT_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80-acquisition"
POST_DOWNLOAD_RESERVE_BYTES = 160 * 1024**3
DOWNLOAD_WORKERS = 8
SCHEMA = "hawking.ascension.qwen80_acquisition.v1"


class Qwen80AcquisitionError(RuntimeError):
    """A safety check or immutable-source verification failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk_bytes):
            digest.update(data)
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _safe_error(error: BaseException) -> dict[str, str]:
    """Keep useful failure evidence without ever persisting request credentials."""

    return {"type": type(error).__name__, "message": str(error).replace("Bearer ", "[redacted] ")[:600]}


class Qwen80Acquisition:
    def __init__(self, *, metadata_path: Path, target: Path, root: Path, workers: int) -> None:
        self.metadata_path = metadata_path.expanduser().resolve()
        self.target = target.expanduser().resolve()
        self.root = root.expanduser().resolve()
        self.workers = workers
        self.status_path = self.root / "QWEN80_ACQUISITION_STATUS.json"
        self.audit_path = self.root / "QWEN80_SOURCE_BODY_AUDIT_CANDIDATE.json"
        self._stop = False

    def _publish(self, phase: str, **fields: Any) -> None:
        previous = _read_json(self.status_path) or {}
        _atomic_json(
            self.status_path,
            {
                "schema": SCHEMA,
                "recorded_at": _utc_now(),
                "pid": os.getpid(),
                "heartbeat": int(previous.get("heartbeat", 0)) + 1,
                "phase": phase,
                "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
                "target_directory": str(self.target),
                "claim_boundary": {
                    "raw_qwen80_is_source_authority_and_teacher_only": True,
                    "raw_qwen80_is_not_a_tournament_participant": True,
                    "no_model_loaded_or_executed": True,
                    "no_gravity_artifact_or_bpw_claim": True,
                    "no_manager_or_tournament_state_changed": True,
                    "credential_material_never_recorded": True,
                },
                **fields,
            },
        )

    def _metadata(self) -> dict[str, Any]:
        document = _read_json(self.metadata_path)
        if document is None:
            raise Qwen80AcquisitionError(f"missing source admission metadata: {self.metadata_path}")
        checked = verify(document, label=str(self.metadata_path))
        source = checked.get("source") if isinstance(checked.get("source"), Mapping) else {}
        inventory = checked.get("inventory") if isinstance(checked.get("inventory"), Mapping) else {}
        if source.get("repository") != SOURCE_REPOSITORY or source.get("revision") != SOURCE_REVISION:
            raise Qwen80AcquisitionError("source admission does not match the immutable Qwen80 authority")
        files = inventory.get("files")
        if not isinstance(files, list) or not files:
            raise Qwen80AcquisitionError("source admission contains no immutable file inventory")
        return checked

    @staticmethod
    def _file_rows(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
        inventory = metadata["inventory"]
        rows = [dict(row) for row in inventory["files"] if isinstance(row, Mapping)]
        if any(not isinstance(row.get("path"), str) or int(row.get("bytes", -1)) < 0 for row in rows):
            raise Qwen80AcquisitionError("source admission has an invalid inventory row")
        if len({row["path"] for row in rows}) != len(rows):
            raise Qwen80AcquisitionError("source admission inventory contains duplicate paths")
        return sorted(rows, key=lambda row: row["path"])

    def _reserve_check(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        # The acquisition target is deliberately not created until the reserve
        # is proven.  Resolve its nearest existing ancestor for the filesystem
        # measurement instead of weakening that ordering.
        filesystem_root = self.target.parent
        while not filesystem_root.exists() and filesystem_root != filesystem_root.parent:
            filesystem_root = filesystem_root.parent
        disk = shutil.disk_usage(filesystem_root)
        existing = sum(
            min(int(row["bytes"]), (self.target / str(row["path"])).stat().st_size)
            for row in rows
            if (self.target / str(row["path"])).is_file()
        )
        total = sum(int(row["bytes"]) for row in rows)
        missing = total - existing
        qwen30 = _tree_bytes(REPO_ROOT / "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct")
        reserve_components = {
            "scratch_bytes": 32 * 1024**3,
            "rollback_bytes": 16 * 1024**3,
            "build_cache_bytes": 16 * 1024**3,
            "system_reserve_bytes": 96 * 1024**3,
        }
        reserve = sum(reserve_components.values())
        result = {
            "filesystem": str(filesystem_root),
            "free_before_bytes": disk.free,
            "total_filesystem_bytes": disk.total,
            "qwen30_source_already_materialized_bytes": qwen30,
            "qwen80_inventory_total_bytes": total,
            "qwen80_existing_expected_bytes": existing,
            "qwen80_missing_expected_bytes": missing,
            "post_download_reserve_bytes": reserve,
            "post_download_reserve_components": reserve_components,
            "free_after_missing_download_bytes": disk.free - missing,
            "cache_policy": "hf_hub_download local_dir only; no duplicate central weight cache authorized",
            "status": "SAFE" if disk.free >= missing + reserve else "INSUFFICIENT_RESERVE",
        }
        if result["status"] != "SAFE":
            raise Qwen80AcquisitionError(
                f"insufficient local reserve: free={disk.free}, missing={missing}, required_reserve={reserve}"
            )
        return result

    @staticmethod
    def _is_complete(row: Mapping[str, Any], target: Path) -> bool:
        path = target / str(row["path"])
        return path.is_file() and path.stat().st_size == int(row["bytes"])

    def _download_one(self, path: str) -> str:
        from huggingface_hub import hf_hub_download

        # The library resolves the credential from the local Hugging Face store.
        # ``local_dir`` retains only per-directory metadata, avoiding a second
        # 148 GiB central cache.  Revision is the immutable admission commit.
        return str(
            hf_hub_download(
                repo_id=SOURCE_REPOSITORY,
                filename=path,
                revision=SOURCE_REVISION,
                local_dir=self.target,
                token=True,
                force_download=False,
                local_files_only=False,
            )
        )

    def _download(self, rows: Sequence[Mapping[str, Any]], reserve: Mapping[str, Any]) -> None:
        pending = [row for row in rows if not self._is_complete(row, self.target)]
        completed = len(rows) - len(pending)
        completed_bytes = sum(int(row["bytes"]) for row in rows if self._is_complete(row, self.target))
        self._publish(
            "DOWNLOADING_PINNED_SOURCE",
            reserve=reserve,
            progress={
                "expected_files": len(rows), "complete_files": completed,
                "pending_files": len(pending), "expected_bytes": sum(int(row["bytes"]) for row in rows),
                "complete_expected_bytes": completed_bytes, "concurrency": self.workers,
            },
        )
        if not pending:
            return
        self.target.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="qwen80-hf") as executor:
            futures = {executor.submit(self._download_one, str(row["path"])): row for row in pending}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    returned_path = future.result()
                except BaseException as exc:
                    self._publish(
                        "DOWNLOAD_FAILED_WITH_EVIDENCE",
                        reserve=reserve,
                        failed_file=str(row["path"]),
                        failure=_safe_error(exc),
                        progress={"expected_files": len(rows), "complete_files": completed, "pending_files": len(rows) - completed},
                    )
                    raise
                local = self.target / str(row["path"])
                if not self._is_complete(row, self.target):
                    raise Qwen80AcquisitionError(
                        f"downloaded file does not match admitted byte count: {row['path']} ({returned_path})"
                    )
                completed += 1
                completed_bytes += int(row["bytes"])
                elapsed = max(time.monotonic() - started, 0.001)
                self._publish(
                    "DOWNLOADING_PINNED_SOURCE",
                    reserve=reserve,
                    last_completed_file=str(row["path"]),
                    progress={
                        "expected_files": len(rows), "complete_files": completed,
                        "pending_files": len(rows) - completed,
                        "expected_bytes": sum(int(item["bytes"]) for item in rows),
                        "complete_expected_bytes": completed_bytes,
                        "observed_completed_bytes_per_second": completed_bytes / elapsed,
                        "concurrency": self.workers,
                    },
                )

    def _verify_and_audit(self, metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], reserve: Mapping[str, Any]) -> None:
        if not all(self._is_complete(row, self.target) for row in rows):
            raise Qwen80AcquisitionError("full source verification requested before every admitted file is complete")
        self._publish("VERIFYING_SOURCE_BODY_SHA256", reserve=reserve, progress={"expected_files": len(rows), "hashed_files": 0})
        audited: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            local = self.target / str(row["path"])
            digest = _sha256_file(local)
            audited.append({
                "path": str(row["path"]), "kind": str(row.get("kind", "unknown")),
                "bytes": local.stat().st_size, "sha256": digest,
            })
            self._publish(
                "VERIFYING_SOURCE_BODY_SHA256", reserve=reserve, last_hashed_file=str(row["path"]),
                progress={"expected_files": len(rows), "hashed_files": index, "verified_bytes": sum(item["bytes"] for item in audited)},
            )
        weights = [item for item in audited if item["kind"] == "weight"]
        totals = {
            "file_count": len(audited),
            "bytes": sum(item["bytes"] for item in audited),
            "weight_file_count": len(weights),
            "weight_bytes": sum(item["bytes"] for item in weights),
        }
        # A verified source audit is an authority artifact, not a periodic
        # heartbeat.  The detached acquisition watcher may verify the same
        # 52 files again, but it must not reseal the audit merely because a
        # timestamp or free-space observation changed.  Downstream complete
        # artifact/revalidation receipts bind this file byte-for-byte.
        existing = _read_json(self.audit_path)
        if self._existing_audit_matches(
            existing,
            metadata=metadata,
            audited=audited,
            totals=totals,
        ):
            self._publish(
                "EARNED_FULL_PINNED_SOURCE_BODY_VERIFIED",
                reserve=reserve,
                audit_path=str(self.audit_path),
                audit_seal_sha256=existing["seal_sha256"],
                audit_reused_without_reseal=True,
                progress={
                    "expected_files": len(rows),
                    "hashed_files": len(rows),
                    "verified_bytes": sum(item["bytes"] for item in audited),
                },
            )
            return
        audit = seal({
            "schema": "hawking.ascension.qwen80_source_body_audit_candidate.v1",
            "status": "CANDIDATE_FULL_PINNED_SOURCE_BODY_VERIFIED",
            "recorded_at": _utc_now(),
            "source_admission_seal_sha256": metadata["seal_sha256"],
            "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
            "target_directory": str(self.target),
            "files": audited,
            "totals": totals,
            "reserve_at_start": reserve,
            "claim_boundary": {
                "full_pinned_raw_bf16_source_is_verified": True,
                "raw_source_is_authority_teacher_only_not_tournament_participant": True,
                "no_model_loaded_or_executed": True,
                "not_a_gravity_artifact": True,
                "does_not_earn_tps_tg_or_manager_qualification": True,
            },
        })
        _atomic_json(self.audit_path, audit)
        self._publish(
            "EARNED_FULL_PINNED_SOURCE_BODY_VERIFIED",
            reserve=reserve,
            audit_path=str(self.audit_path), audit_seal_sha256=audit["seal_sha256"],
            progress={"expected_files": len(rows), "hashed_files": len(rows), "verified_bytes": sum(item["bytes"] for item in audited)},
        )

    def _existing_audit_matches(
        self,
        existing: Mapping[str, Any] | None,
        *,
        metadata: Mapping[str, Any],
        audited: Sequence[Mapping[str, Any]],
        totals: Mapping[str, int],
    ) -> bool:
        """Return whether a prior sealed audit proves these exact source bytes.

        ``recorded_at`` and ``reserve_at_start`` are operational observations,
        not source identity.  They intentionally do not participate here so a
        completed acquisition watcher cannot invalidate a later native
        artifact solely by refreshing its heartbeat.  A file, source revision,
        target, or upstream source-admission change still forces a new sealed
        audit after the just-completed full hash pass.
        """

        if existing is None:
            return False
        try:
            checked = verify(existing, label=str(self.audit_path))
        except Exception:
            return False
        return (
            checked.get("schema") == "hawking.ascension.qwen80_source_body_audit_candidate.v1"
            and checked.get("status") == "CANDIDATE_FULL_PINNED_SOURCE_BODY_VERIFIED"
            and checked.get("source_admission_seal_sha256") == metadata.get("seal_sha256")
            and checked.get("source")
            == {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION}
            and checked.get("target_directory") == str(self.target)
            and checked.get("files") == [dict(row) for row in audited]
            and checked.get("totals") == dict(totals)
        )

    def run(self) -> int:
        metadata = self._metadata()
        rows = self._file_rows(metadata)
        reserve = self._reserve_check(rows)
        self._download(rows, reserve)
        self._verify_and_audit(metadata, rows, reserve)
        return 0

    def watch(self, *, idle_seconds: float) -> int:
        def stop(_signal: int, _frame: Any) -> None:
            self._stop = True
        old_term = signal.signal(signal.SIGTERM, stop)
        old_int = signal.signal(signal.SIGINT, stop)
        try:
            while not self._stop:
                try:
                    self.run()
                except BaseException as exc:
                    self._publish("FAILED_WITH_EVIDENCE", failure=_safe_error(exc))
                    raise
                if not self._stop:
                    time.sleep(idle_seconds)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("once")
    watch = commands.add_parser("watch")
    watch.add_argument("--idle-seconds", type=float, default=900.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise Qwen80AcquisitionError("workers must be positive")
    worker = Qwen80Acquisition(metadata_path=args.metadata, target=args.target, root=args.root, workers=args.workers)
    return worker.run() if args.command == "once" else worker.watch(idle_seconds=args.idle_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
