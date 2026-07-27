#!/usr/bin/env python3.12
"""Content-addressed checkpoint store for the Odyssey toy trainer apparatus.

A checkpoint's id IS the sha256 of its canonical state payload. Metadata such as
wall_clock is stored beside the payload and is not part of the identity: two
identical training states therefore share one id; any state difference yields a
different id.

Supports multi-shard layout, atomic write (tmp + rename), integrity verification,
and failure detection (corrupt / missing shard / mid-write).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

SCHEMA = "hawking.odyssey.checkpoint.content_addressed.v1"
REQUIRED_STATE_FIELDS = (
    "schema",
    "stage",
    "step",
    "objective",
    "model_state",
    "fixture_label",
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt, incomplete, or hash-mismatched."""


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_id(state_payload: dict[str, Any]) -> str:
    """Checkpoint id IS the content hash of the state payload."""
    return hashlib.sha256(_canonical_bytes(state_payload)).hexdigest()


def _shard_bytes(payload: bytes, shard_size: int) -> list[bytes]:
    if shard_size <= 0:
        return [payload]
    return [payload[i : i + shard_size] for i in range(0, len(payload), shard_size)] or [b""]


class CheckpointStore:
    """Filesystem store under a root directory (usually a temp dir in tests)."""

    def __init__(self, root: Path, *, shard_size: int = 0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.events_path = self.root / "rollback_events.jsonl"
        self.current_path = self.root / "CURRENT"

    def _dir_for(self, cid: str) -> Path:
        return self.root / cid[:2] / cid

    def save(
        self,
        *,
        stage: str,
        step: int,
        objective: str,
        model_state: dict[str, Any],
        parent_id: str | None = None,
        extra: dict[str, Any] | None = None,
        fixture_label: str = "FIXTURE",
    ) -> dict[str, Any]:
        """Write a content-addressed checkpoint. Returns metadata including id."""
        state_payload: dict[str, Any] = {
            "schema": SCHEMA,
            "stage": stage,
            "step": int(step),
            "objective": objective,
            "model_state": model_state,
            "parent_id": parent_id or "genesis",
            "fixture_label": fixture_label,
        }
        if extra:
            # Extra goes into payload only if callers want it part of identity.
            state_payload["extra"] = extra
        for f in REQUIRED_STATE_FIELDS:
            if f not in state_payload:
                raise ValueError(f"checkpoint missing field {f}")

        cid = content_id(state_payload)
        raw = _canonical_bytes(state_payload)
        shards = _shard_bytes(raw, self.shard_size)
        shard_hashes = [hashlib.sha256(s).hexdigest() for s in shards]

        # Atomic write protocol: write into .partial, then rename.
        dest = self._dir_for(cid)
        if dest.is_dir() and (dest / "COMPLETE").is_file():
            # Identical state already stored — id reuse is the point.
            self.current_path.write_text(cid + "\n")
            return self.load_meta(cid)

        partial = self.root / f".partial-{cid}"
        if partial.exists():
            # Clean stale partials from prior killed writes.
            for p in partial.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(partial.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            partial.rmdir()
        partial.mkdir(parents=True)

        try:
            for i, blob in enumerate(shards):
                (partial / f"shard-{i:04d}.bin").write_bytes(blob)
            meta = {
                "schema": SCHEMA,
                "checkpoint_id": cid,
                "stage": stage,
                "step": int(step),
                "objective": objective,
                "parent_id": parent_id or "genesis",
                "n_shards": len(shards),
                "shard_hashes": shard_hashes,
                "payload_sha256": cid,
                "payload_bytes": len(raw),
                "fixture_label": fixture_label,
                "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                # wall_clock is metadata only — NOT part of checkpoint_id.
            }
            (partial / "META.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n"
            )
            (partial / "COMPLETE").write_text("ok\n")
            # Publish
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                # Race: another writer finished; drop our partial.
                for p in partial.rglob("*"):
                    if p.is_file():
                        p.unlink()
                for p in sorted(partial.rglob("*"), reverse=True):
                    if p.is_dir():
                        p.rmdir()
                partial.rmdir()
            else:
                os.rename(partial, dest)
        except Exception:
            # Leave partial for detection tests; re-raise.
            raise

        self.current_path.write_text(cid + "\n")
        return meta

    def load_meta(self, cid: str) -> dict[str, Any]:
        d = self._dir_for(cid)
        if not d.is_dir():
            raise CheckpointError(f"MISSING_CHECKPOINT: {cid}")
        if not (d / "COMPLETE").is_file():
            raise CheckpointError(f"INCOMPLETE_WRITE: {cid} (no COMPLETE marker)")
        meta_path = d / "META.json"
        if not meta_path.is_file():
            raise CheckpointError(f"MISSING_META: {cid}")
        return json.loads(meta_path.read_text())

    def load_state(self, cid: str) -> dict[str, Any]:
        """Load and verify a checkpoint. Raises CheckpointError on any integrity failure."""
        meta = self.load_meta(cid)
        d = self._dir_for(cid)
        n = int(meta["n_shards"])
        expected_hashes = meta["shard_hashes"]
        if len(expected_hashes) != n:
            raise CheckpointError(f"META_INCONSISTENT: shard_hashes length != n_shards for {cid}")

        parts: list[bytes] = []
        for i in range(n):
            sp = d / f"shard-{i:04d}.bin"
            if not sp.is_file():
                raise CheckpointError(f"MISSING_SHARD: {cid} shard {i}")
            blob = sp.read_bytes()
            got = hashlib.sha256(blob).hexdigest()
            if got != expected_hashes[i]:
                raise CheckpointError(
                    f"CORRUPT_SHARD: {cid} shard {i} hash {got} != {expected_hashes[i]}"
                )
            parts.append(blob)
        raw = b"".join(parts)
        got_id = hashlib.sha256(raw).hexdigest()
        if got_id != cid:
            raise CheckpointError(f"HASH_MISMATCH: payload {got_id} != id {cid}")
        state = json.loads(raw.decode("utf-8"))
        # Re-derive content id from parsed state to catch JSON reordering tricks.
        if content_id(state) != cid:
            raise CheckpointError(f"CONTENT_ID_MISMATCH: recomputed id != {cid}")
        return state

    def current_id(self) -> str | None:
        if not self.current_path.is_file():
            return None
        return self.current_path.read_text().strip() or None

    def set_current(self, cid: str) -> None:
        # Verify it loads before pointing CURRENT at it.
        self.load_state(cid)
        self.current_path.write_text(cid + "\n")

    def verify(self, cid: str) -> dict[str, Any]:
        try:
            state = self.load_state(cid)
            return {"status": "OK", "checkpoint_id": cid, "step": state["step"]}
        except CheckpointError as e:
            return {"status": "FAIL", "checkpoint_id": cid, "error": str(e)}

    # --- failure-injection helpers (explicit; tests only) -------------------

    def inject_corrupt_shard(self, cid: str, shard_index: int = 0) -> None:
        d = self._dir_for(cid)
        sp = d / f"shard-{shard_index:04d}.bin"
        if not sp.is_file():
            raise CheckpointError(f"cannot corrupt missing shard {shard_index}")
        sp.write_bytes(sp.read_bytes() + b"\x00CORRUPT")

    def inject_lose_shard(self, cid: str, shard_index: int = 0) -> None:
        d = self._dir_for(cid)
        sp = d / f"shard-{shard_index:04d}.bin"
        if sp.is_file():
            sp.unlink()

    def inject_kill_mid_write(
        self,
        *,
        stage: str,
        step: int,
        objective: str,
        model_state: dict[str, Any],
    ) -> str:
        """Simulate a killed write: partial dir with no COMPLETE marker. Returns intended id."""
        state_payload = {
            "schema": SCHEMA,
            "stage": stage,
            "step": int(step),
            "objective": objective,
            "model_state": model_state,
            "parent_id": "genesis",
            "fixture_label": "FIXTURE: mid-write kill injection",
        }
        cid = content_id(state_payload)
        raw = _canonical_bytes(state_payload)
        partial = self.root / f".partial-{cid}"
        partial.mkdir(parents=True, exist_ok=True)
        # Write only first half of payload, no COMPLETE.
        (partial / "shard-0000.bin").write_bytes(raw[: max(1, len(raw) // 2)])
        (partial / "META.json").write_text(
            json.dumps({"checkpoint_id": cid, "n_shards": 1, "incomplete": True})
        )
        # Also place an incomplete published dir to exercise load detection.
        dest = self._dir_for(cid)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "shard-0000.bin").write_bytes(raw[: max(1, len(raw) // 2)])
        (dest / "META.json").write_text(
            json.dumps(
                {
                    "checkpoint_id": cid,
                    "n_shards": 1,
                    "shard_hashes": ["deadbeef"],
                    "incomplete": True,
                }
            )
        )
        # deliberately no COMPLETE marker
        return cid
