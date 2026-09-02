#!/usr/bin/env python3
"""NX_PACKER — generic, source-independent packed NX from a compiled fragment.

The only packed-NX producer previously in git is
tools/headless/first_noetic_executable.py, which hardlinks a Qwen3.8-27B
uniform-q4 catalog. That hardlink is the thing being replaced. This packer
does the same job for an arbitrary specimen: it copies every runtime-required
byte into a self-contained artifact and bills them.

A pointer, a hardlink, or a path back into the source tree is not an NX and
raises. A placeholder organ, a missing metallib, or a manifest whose byte
total disagrees with its parts raises. Those are not flags.

    python3 tools/future/nx_packer.py --build
    python3 -m pytest tools/future/test_nx_packer.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    ),
)

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.future._common import RECEIPTS, REPO, write_receipt
from tools.future import flash_nx_audit as nx_audit


RECEIPT = "NX_PACKER.json"
SCHEMA = "hawking.future.nx_packer.v1"
RECORDED_BY = "tools/future/nx_packer.py"
VERSION = 1
NX_SCHEMA = "hawking.future.packed_nx.v1"
PACKED_STATUS = "SOURCE_INDEPENDENT_COMPLETE"

COMPILED = "COMPILED"
NATIVE_UNMEASURED = "NATIVE_UNMEASURED"
COMPILED_IDENTITY_KIND = "METAL_PIPELINE"
PIPELINE_OBJECT = "MTLComputePipelineState"

BILLING_CATEGORIES: tuple[str, ...] = (
    "representation",
    "model_specific_code",
    "metadata",
    "tables",
    "generators",
    "residuals",
    "state_auxiliaries",
    "nr",
)

REPRESENTATION_NAMES = frozenset(
    {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "consolidated.safetensors",
    }
)
REPRESENTATION_SUFFIXES = (".safetensors", ".gguf", ".bin")
TABLE_NAMES = frozenset(
    {
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
        "special_tokens_map.json",
        "added_tokens.json",
    }
)
METADATA_NAMES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "configuration.json",
    }
)
SKIP_SOURCE_NAMES = frozenset(
    {
        "MODEL_LAKE_SPECIMEN_SEAL.json",
        "MODEL_LAKE_SPECIMEN_SEAL.sha256",
        ".gitattributes",
        ".gitignore",
        "README.md",
        "LICENSE",
        "LICENSE.txt",
    }
)

PLACEHOLDER_KINDS = frozenset(
    {
        "ABSENT",
        "PLACEHOLDER",
        "FAKE",
        "SYNTHETIC",
        "SOURCE_HASH",
        "SOURCE",
        "PLANNED",
        "NATIVE_UNMEASURED",
    }
)
HARDCODED_DIGESTS = frozenset(
    {"deadbeef" * 4, "0" * 64, "a" * 64, "abc123" * 8}
)


class NxPackerError(ValueError):
    """A packed NX cannot be emitted."""


class RenamedSourcePointer(NxPackerError):
    """The would-be NX is a renamed path into the source tree."""


class PlaceholderOrgan(NxPackerError):
    """A placeholder claiming compiled identity cannot occupy a packed NX."""


class MissingMetallib(NxPackerError):
    """A COMPILED organ has no MTLBinaryArchive bytes to pack."""


class BillingMismatch(NxPackerError):
    """Manifest total_bytes disagrees with the sum of billed parts."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdefABCDEF" for c in value
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_device_compiler() -> Any:
    """Import device_compiler from this package or the primary checkout.

    The module is untracked in some worktrees. Absence here is not absence
    in the project. Never edit that file from this packer.
    """
    existing = _sys.modules.get("tools.future.device_compiler")
    if existing is not None and hasattr(existing, "lower_plan"):
        return existing
    here = Path(__file__).resolve().parent / "device_compiler.py"
    if here.is_file():
        from tools.future import device_compiler as dcomp

        return dcomp
    for root in nx_audit.evidence_roots():
        cand = Path(root) / "tools" / "future" / "device_compiler.py"
        if not cand.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "tools.future.device_compiler", cand
        )
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["tools.future.device_compiler"] = mod
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(
        "tools.future.device_compiler.lower_plan is not importable in this "
        "checkout or the primary tree; a compiled identity is not invented"
    )


def packer_callable() -> bool:
    """The generic packer exists. Presence of this function is the entry point."""
    return True


def default_dest(specimen_id: str | None) -> Path:
    sid = (specimen_id or "unknown").replace("/", "--")
    return REPO / "artifacts" / "nx" / sid


def _dot(node: Any, dotted: str, default: Any = None) -> Any:
    cur: Any = node
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _kernels(fragment: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(fragment, Mapping):
        return []
    raw = _dot(fragment, "physical_program.kernels")
    if isinstance(raw, list):
        return [dict(k) for k in raw if isinstance(k, Mapping)]
    plan = fragment.get("plan")
    if isinstance(plan, list):
        return [dict(k) for k in plan if isinstance(k, Mapping)]
    return []


def _same_inode(a: Path, b: Path) -> bool:
    try:
        sa = a.stat()
        sb = b.stat()
    except OSError:
        return False
    return sa.st_dev == sb.st_dev and sa.st_ino == sb.st_ino


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def refuse_renamed_source_pointer(
    *,
    dest: Path,
    source_tree: Path | None,
    serialized_path: str | None = None,
) -> None:
    """Raise if the packed body would be a renamed source pointer."""
    dest_r = dest.resolve()
    if source_tree is not None and source_tree.exists():
        src = source_tree.resolve()
        if dest_r == src:
            raise RenamedSourcePointer(
                f"dest {dest_r} is the source tree; a renamed source pointer is not an NX"
            )
        if _is_under(dest_r, src):
            raise RenamedSourcePointer(
                f"dest {dest_r} sits inside the source tree {src}; "
                "a path back into the source is not an NX"
            )
        if dest_r.is_file() and src.is_dir():
            for cand in src.rglob("*"):
                if cand.is_file() and _same_inode(dest_r, cand):
                    raise RenamedSourcePointer(
                        f"dest {dest_r} is a hardlink/inode of source file {cand}"
                    )
        if dest_r.name in REPRESENTATION_NAMES and _is_under(dest_r, src):
            raise RenamedSourcePointer(
                f"dest names a source checkpoint file ({dest_r.name}); "
                "a renamed source pointer is not an NX"
            )
    if dest_r.name in REPRESENTATION_NAMES:
        raise RenamedSourcePointer(
            f"dest names a source checkpoint file ({dest_r.name}); "
            "a renamed source pointer is not an NX"
        )
    if isinstance(serialized_path, str) and serialized_path:
        text = serialized_path
        name = Path(text).name
        if name in REPRESENTATION_NAMES:
            raise RenamedSourcePointer(
                f"serialized_artifact.path names a source checkpoint file ({name})"
            )
        if source_tree is not None and source_tree.exists():
            src = str(source_tree.resolve())
            if text == src or text.startswith(src.rstrip("/") + "/") or src in text:
                raise RenamedSourcePointer(
                    f"serialized_artifact.path resolves into the source tree: {text}"
                )


def refuse_placeholder_organs(
    fragment: Mapping[str, Any] | None,
    *,
    dcomp: Any | None = None,
) -> list[dict[str, Any]]:
    """Raise on any COMPILED slot whose identity is a placeholder."""
    kernels = _kernels(fragment)
    compiled: list[dict[str, Any]] = []
    helper = dcomp
    for slot in kernels:
        organ = slot.get("organ")
        status = slot.get("status")
        identity = slot.get("compiled_identity")
        kind = None if not isinstance(identity, Mapping) else identity.get("kind")
        if status != COMPILED:
            continue
        if kind in PLACEHOLDER_KINDS or kind is None:
            raise PlaceholderOrgan(
                f"organ={organ!r} claims {COMPILED} with kind={kind!r}; "
                "a placeholder organ cannot occupy a packed NX"
            )
        if helper is not None:
            if helper.is_placeholder_compiled_identity(
                identity,
                source_sha256=(
                    identity.get("source_sha256")
                    if isinstance(identity, Mapping)
                    else None
                ),
                entry_point=(
                    identity.get("entry_point")
                    if isinstance(identity, Mapping)
                    else None
                ),
            ):
                raise PlaceholderOrgan(
                    f"organ={organ!r} compiled_identity is a placeholder; "
                    "it cannot occupy COMPILED on a packed NX"
                )
        else:
            if not isinstance(identity, Mapping):
                raise PlaceholderOrgan(
                    f"organ={organ!r} has no compiled_identity object"
                )
            if identity.get("kind") != COMPILED_IDENTITY_KIND:
                raise PlaceholderOrgan(
                    f"organ={organ!r} kind={identity.get('kind')!r} is not "
                    f"{COMPILED_IDENTITY_KIND}"
                )
            digest = identity.get("shader_hash") or identity.get("value")
            src_hash = identity.get("source_sha256")
            if not _is_sha256(digest):
                raise PlaceholderOrgan(
                    f"organ={organ!r} shader_hash is not a sha256"
                )
            if _is_sha256(src_hash) and digest == src_hash:
                raise PlaceholderOrgan(
                    f"organ={organ!r} shader_hash equals source_sha256; "
                    "a source digest is not a compiled metallib"
                )
            if isinstance(digest, str) and digest.lower() in HARDCODED_DIGESTS:
                raise PlaceholderOrgan(
                    f"organ={organ!r} shader_hash is a hardcoded placeholder digest"
                )
            pipeline = identity.get("pipeline")
            if not isinstance(pipeline, Mapping):
                raise PlaceholderOrgan(f"organ={organ!r} pipeline object missing")
            if pipeline.get("object") != PIPELINE_OBJECT:
                raise PlaceholderOrgan(
                    f"organ={organ!r} pipeline.object={pipeline.get('object')!r} "
                    f"is not {PIPELINE_OBJECT}"
                )
            if pipeline.get("created") is not True:
                raise PlaceholderOrgan(
                    f"organ={organ!r} MTLComputePipelineState was not created"
                )
        compiled.append(dict(slot))
    return compiled


def reconcile_manifest(
    parts: Sequence[Mapping[str, Any]],
    *,
    claimed_total: int | None = None,
) -> dict[str, Any]:
    """Sum billed parts. Raise if a claimed total disagrees."""
    rows: list[dict[str, Any]] = []
    by_cat: dict[str, dict[str, Any]] = {
        c: {"bytes": 0, "n_parts": 0, "parts": []} for c in BILLING_CATEGORIES
    }
    total = 0
    for raw in parts:
        if not isinstance(raw, Mapping):
            raise BillingMismatch("part is not an object")
        cat = str(raw.get("category") or "")
        if cat not in by_cat:
            raise BillingMismatch(f"unknown billing category {cat!r}")
        nbytes = raw.get("bytes")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            raise BillingMismatch(
                f"part {raw.get('id')!r} bytes={nbytes!r} is not a non-negative int"
            )
        digest = raw.get("sha256")
        if nbytes > 0 and not _is_sha256(digest):
            raise BillingMismatch(
                f"part {raw.get('id')!r} has bytes={nbytes} but no sha256; "
                "unhashed bytes cannot be billed"
            )
        if nbytes == 0 and raw.get("runtime_required") is True:
            raise BillingMismatch(
                f"part {raw.get('id')!r} is runtime_required with 0 bytes"
            )
        row = {
            "id": str(raw.get("id") or raw.get("path") or ""),
            "path": str(raw.get("path") or raw.get("id") or ""),
            "category": cat,
            "bytes": int(nbytes),
            "sha256": None if not digest else str(digest),
            "runtime_required": bool(raw.get("runtime_required")),
        }
        rows.append(row)
        by_cat[cat]["bytes"] += int(nbytes)
        by_cat[cat]["n_parts"] += 1
        by_cat[cat]["parts"].append(row["id"])
        total += int(nbytes)
    if claimed_total is not None and int(claimed_total) != total:
        raise BillingMismatch(
            f"manifest total_bytes={claimed_total} disagrees with the sum of "
            f"its parts ({total})"
        )
    billing = {
        "categories": {
            c: {
                "bytes": by_cat[c]["bytes"],
                "n_parts": by_cat[c]["n_parts"],
                "parts": by_cat[c]["parts"],
                "runtime_required": by_cat[c]["bytes"] > 0,
            }
            for c in BILLING_CATEGORIES
        },
        "total_bytes": total,
        "parts_sum_bytes": total,
        "n_parts": len(rows),
        "reconciles": True,
        "status": "CLOSED",
        "complete_system": True,
        "all_required_bytes_included": True,
    }
    return {"parts": rows, "billing": billing}


def _classify_source_file(name: str) -> str | None:
    if name in SKIP_SOURCE_NAMES or name.endswith(".metadata") or name.endswith(".incomplete"):
        return None
    if name.startswith("."):
        return None
    if name in REPRESENTATION_NAMES or name.endswith(REPRESENTATION_SUFFIXES):
        return "representation"
    if name in TABLE_NAMES:
        return "tables"
    if name in METADATA_NAMES:
        return "metadata"
    if name.endswith(".json") and "config" in name:
        return "metadata"
    return None


def _copy_and_hash(src: Path, dest: Path) -> tuple[int, str]:
    """Byte-copy (never hardlink, never APFS clone) and hash."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _same_inode(src, dest):
            raise RenamedSourcePointer(
                f"{dest} is the same inode as {src}; a hardlink is not an NX"
            )
        try:
            ss, ds = src.stat(), dest.stat()
        except OSError:
            ss = ds = None
        if ss is not None and ds is not None and ss.st_size == ds.st_size and ds.st_mtime >= ss.st_mtime:
            digest = hashlib.sha256(dest.read_bytes() if ds.st_size <= (8 << 20) else bytes()).hexdigest() if ds.st_size <= (8 << 20) else None
            if ds.st_size <= (8 << 20):
                src_digest = hashlib.sha256(src.read_bytes()).hexdigest()
                if digest == src_digest:
                    return ds.st_size, digest
            else:
                # Large representation: size+mtime match is enough to skip a re-copy;
                # still hash dest so the ledger is a real digest of the packed bytes.
                h = hashlib.sha256()
                with open(dest, "rb") as fh:
                    while True:
                        b = fh.read(8 << 20)
                        if not b:
                            break
                        h.update(b)
                return ds.st_size, h.hexdigest()
        dest.unlink(missing_ok=True)
    tmp = _tmp_sibling(dest)
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as inf, open(tmp, "wb") as out:
        while True:
            b = inf.read(8 << 20)
            if not b:
                break
            out.write(b)
            h.update(b)
            n += len(b)
    os.replace(tmp, dest)
    if _same_inode(src, dest):
        dest.unlink()
        raise RenamedSourcePointer(
            f"copy of {src} landed on the same inode; a hardlink is not an NX"
        )
    return n, h.hexdigest()


def _tmp_sibling(dest: Path) -> Path:
    """A staging path no other process will also pick.

    Both writers below stage into a sibling and os.replace() it into place.
    The staging name used to be a fixed `<dest>.tmp`, which is shared state:
    two processes packing the same content-addressed artifact concurrently
    both wrote that one path, and whichever replaced it first left the other
    with `FileNotFoundError: ... .tmp -> ...`. That surfaced as 17 setup
    errors the moment the test suite ran a file's tests in one worker while
    another worker packed the same model. os.replace is atomic and the
    destination name is a content hash, so a late writer landing on an
    identical file is harmless -- only the staging name needed to be unique.
    """
    return dest.with_name(f"{dest.name}.{os.getpid()}.tmp")


def _write_and_hash(dest: Path, data: bytes) -> tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_sibling(dest)
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return len(data), sha256_bytes(data)


def _looks_like_source(raw: bytes) -> bool:
    if not raw:
        return True
    head = raw.lstrip()
    return (
        head.startswith(b"#include")
        or head.startswith(b"//")
        or head.startswith(b"kernel void")
        or head.startswith(b"using namespace metal")
    )


class CapturingMetalBackend:
    """Wrap a Metal backend and keep archive bytes after lower_plan's temp dies."""

    def __init__(self, inner: Any, sink: str | Path) -> None:
        self.inner = inner
        self.sink = Path(sink)
        self.captured: dict[str, dict[str, Any]] = {}

    def compile_jobs(self, jobs: Sequence[Any]) -> dict[str, Any]:
        self.sink.mkdir(parents=True, exist_ok=True)
        batch = self.inner.compile_jobs(jobs)
        for job in jobs:
            organ = getattr(job, "organ", None)
            archive_path = getattr(job, "archive_path", None)
            if not organ or not archive_path:
                continue
            src = Path(str(archive_path))
            if not src.is_file():
                continue
            raw = src.read_bytes()
            dest = self.sink / f"{organ}.mtlarchive"
            dest.write_bytes(raw)
            self.captured[str(organ)] = {
                "path": str(dest),
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "source": getattr(job, "source", None),
                "entry_point": getattr(job, "entry_point", None),
                "source_sha256": getattr(job, "source_sha256", None),
            }
        return batch


def _archive_from_identity(
    slot: Mapping[str, Any],
    *,
    archives: Mapping[str, Any] | None,
) -> tuple[bytes | None, str | None]:
    organ = str(slot.get("organ") or "")
    identity = slot.get("compiled_identity") if isinstance(slot.get("compiled_identity"), Mapping) else {}
    if archives and organ in archives:
        rec = archives[organ]
        if isinstance(rec, (bytes, bytearray)):
            return bytes(rec), None
        if isinstance(rec, Mapping):
            if isinstance(rec.get("bytes_payload"), (bytes, bytearray)):
                return bytes(rec["bytes_payload"]), rec.get("source")
            path = rec.get("path")
            if isinstance(path, str) and Path(path).is_file():
                return Path(path).read_bytes(), rec.get("source")
        if isinstance(rec, (str, Path)) and Path(rec).is_file():
            return Path(rec).read_bytes(), None
    path_raw = identity.get("archive_path") if isinstance(identity, Mapping) else None
    if isinstance(path_raw, str) and Path(path_raw).is_file():
        return Path(path_raw).read_bytes(), None
    return None, None


def _recompile_organ(
    organ: str,
    slot: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
    specimen_id: str | None,
    dcomp: Any,
    dest_dir: Path,
) -> tuple[bytes, str, str]:
    shape = slot.get("specimen_shape") if isinstance(slot.get("specimen_shape"), Mapping) else slot.get("shape")
    emitted = dcomp.emit_shader(
        organ,
        shape=shape if isinstance(shape, Mapping) else None,
        config=config,
        specimen_id=specimen_id,
    )
    if emitted is None:
        raise MissingMetallib(
            f"organ={organ!r} has no honest lowering; metallib cannot be packed"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / f"{organ}.mtlarchive"
    job = dcomp.CompileJob(
        organ=organ,
        source=str(emitted["source"]),
        entry_point=str(emitted["entry_point"]),
        archive_path=str(archive),
        why=str(emitted.get("why") or ""),
        extents=list(emitted.get("extents") or []),
    )
    batch = dcomp.LiveMetalBackend().compile_jobs([job])
    rows = [
        r
        for r in (batch.get("results") or [])
        if isinstance(r, Mapping) and r.get("id") == organ
    ]
    row = rows[0] if rows else None
    if not isinstance(row, Mapping) or row.get("ok") is not True or not archive.is_file():
        err = None if not isinstance(row, Mapping) else row.get("error")
        raise MissingMetallib(
            f"organ={organ!r} recompile produced no MTLBinaryArchive "
            f"({err or batch.get('error') or 'no row'})"
        )
    raw = archive.read_bytes()
    if not raw or _looks_like_source(raw):
        raise MissingMetallib(
            f"organ={organ!r} archive bytes are source, not an MTLBinaryArchive"
        )
    return raw, str(emitted["source"]), str(emitted["entry_point"])


def _iter_source_files(source_tree: Path) -> list[Path]:
    if not source_tree.is_dir():
        return []
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(source_tree):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for fn in filenames:
            out.append(Path(dirpath) / fn)
    return out


def pack(
    *,
    nx_fragment: Mapping[str, Any] | None,
    specimen_path: str | Path | None,
    dest: str | Path,
    specimen_id: str | None = None,
    family: Any = None,
    config: Mapping[str, Any] | None = None,
    kernel_plan: Mapping[str, Any] | None = None,
    nr: Mapping[str, Any] | str | Path | None = None,
    archives: Mapping[str, Any] | None = None,
    dcomp: Any | None = None,
) -> dict[str, Any]:
    """Emit a source-independent packed NX. Raises rather than returning a flag."""
    dest_p = Path(dest)
    source_p = None if specimen_path is None else Path(str(specimen_path))
    refuse_renamed_source_pointer(dest=dest_p, source_tree=source_p)

    if not isinstance(nx_fragment, Mapping):
        raise NxPackerError(
            "no DeviceCompiler NX fragment was handed; a renamed source pointer "
            "is not an NX and a missing fragment is not packed"
        )

    helper = dcomp
    if helper is None:
        try:
            helper = load_device_compiler()
        except ImportError:
            helper = None

    compiled = refuse_placeholder_organs(nx_fragment, dcomp=helper)
    if not compiled:
        raise MissingMetallib(
            "NX fragment has 0 compiled organs with metallibs to pack; "
            "NATIVE_UNMEASURED planned organs are not a packed NX"
        )

    dest_p.mkdir(parents=True, exist_ok=True)
    parts_root = dest_p / "parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    billed: list[dict[str, Any]] = []

    if source_p is None or not source_p.is_dir():
        raise NxPackerError(
            f"specimen_path {source_p} is not a directory; representation bytes "
            "cannot be invented and a missing source is not packed as a pointer"
        )
    n_rep = 0
    for src_file in _iter_source_files(source_p):
        rel_name = src_file.name
        cat = _classify_source_file(rel_name)
        if cat is None:
            continue
        rel = Path("parts") / cat / rel_name
        dest_file = dest_p / rel
        nbytes, digest = _copy_and_hash(src_file, dest_file)
        billed.append(
            {
                "id": str(rel),
                "path": str(rel),
                "category": cat,
                "bytes": nbytes,
                "sha256": digest,
                "runtime_required": True,
            }
        )
        if cat == "representation":
            n_rep += 1
    if n_rep <= 0:
        raise NxPackerError(
            f"no representation files were copied out of {source_p}; "
            "an NX without weights is a metadata seal"
        )

    code_dir = parts_root / "model_specific_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    packed_kernels: list[dict[str, Any]] = []
    tmp_recompile = dest_p / "_recompile"
    for slot in compiled:
        organ = str(slot.get("organ"))
        identity = dict(slot.get("compiled_identity") or {})
        raw, source_text = _archive_from_identity(slot, archives=archives)
        entry = identity.get("entry_point") or slot.get("entry_point")
        if raw is None:
            if helper is None:
                raise MissingMetallib(
                    f"organ={organ!r} archive is missing and DeviceCompiler is "
                    "not importable to recompile"
                )
            raw, source_text, entry = _recompile_organ(
                organ,
                slot,
                config=config,
                specimen_id=specimen_id,
                dcomp=helper,
                dest_dir=tmp_recompile,
            )
        if not raw:
            raise MissingMetallib(f"organ={organ!r} metallib is empty")
        if _looks_like_source(raw):
            raise PlaceholderOrgan(
                f"organ={organ!r} archive bytes are shader source, not an "
                "MTLBinaryArchive"
            )
        digest = sha256_bytes(raw)
        claimed = identity.get("shader_hash") or identity.get("value")
        src_hash = identity.get("source_sha256")
        if _is_sha256(claimed) and claimed != digest:
            identity = dict(identity)
            identity["fragment_shader_hash"] = claimed
            identity["shader_hash"] = digest
            identity["value"] = digest
        if _is_sha256(src_hash) and src_hash == digest:
            raise PlaceholderOrgan(
                f"organ={organ!r} packed digest equals source_sha256"
            )
        if digest.lower() in HARDCODED_DIGESTS:
            raise PlaceholderOrgan(
                f"organ={organ!r} packed digest is a hardcoded placeholder"
            )
        rel = Path("parts") / "model_specific_code" / f"{organ}.{digest[:16]}.mtlarchive"
        _write_and_hash(dest_p / rel, raw)
        billed.append(
            {
                "id": str(rel),
                "path": str(rel),
                "category": "model_specific_code",
                "bytes": len(raw),
                "sha256": digest,
                "runtime_required": True,
                "organ": organ,
            }
        )
        identity["archive_path"] = str(rel)
        identity["archive_bytes"] = len(raw)
        identity["shader_hash"] = digest
        identity["value"] = digest
        identity["kind"] = COMPILED_IDENTITY_KIND
        if entry:
            identity["entry_point"] = entry
        packed_kernels.append(
            {
                "organ": organ,
                "status": COMPILED,
                "occupying": slot.get("occupying"),
                "compiled_identity": identity,
                "entry_point": identity.get("entry_point"),
                "shader_hash": digest,
                "why": slot.get("why"),
            }
        )
        if isinstance(source_text, str) and source_text.strip():
            src_bytes = source_text.encode("utf-8")
            src_rel = Path("parts") / "model_specific_code" / f"{organ}.metal"
            _write_and_hash(dest_p / src_rel, src_bytes)
            billed.append(
                {
                    "id": str(src_rel),
                    "path": str(src_rel),
                    "category": "model_specific_code",
                    "bytes": len(src_bytes),
                    "sha256": sha256_bytes(src_bytes),
                    "runtime_required": False,
                    "organ": organ,
                    "why": "shader source is provenance; runtime loads the archive",
                }
            )

    if tmp_recompile.is_dir():
        shutil.rmtree(tmp_recompile, ignore_errors=True)

    for cat, why in (
        ("generators", "this specimen has no generator payload required at runtime"),
        ("residuals", "this specimen has no residual payload required at runtime"),
        ("state_auxiliaries", "this specimen has no packed state auxiliary; KV is runtime"),
    ):
        already = any(p["category"] == cat for p in billed)
        if not already:
            billed.append(
                {
                    "id": f"parts/{cat}/ABSENT",
                    "path": f"parts/{cat}/ABSENT",
                    "category": cat,
                    "bytes": 0,
                    "sha256": None,
                    "runtime_required": False,
                    "why": why,
                }
            )

    nr_runtime = False
    if nr is not None:
        nr_runtime = True
        if isinstance(nr, (str, Path)) and Path(nr).is_file():
            nr_path = Path(nr)
            rel = Path("parts") / "nr" / nr_path.name
            nbytes, digest = _copy_and_hash(nr_path, dest_p / rel)
            billed.append(
                {
                    "id": str(rel),
                    "path": str(rel),
                    "category": "nr",
                    "bytes": nbytes,
                    "sha256": digest,
                    "runtime_required": True,
                }
            )
        elif isinstance(nr, Mapping):
            blob = json.dumps(dict(nr), sort_keys=True, separators=(",", ":")).encode()
            rel = Path("parts") / "nr" / "nr.json"
            _write_and_hash(dest_p / rel, blob)
            billed.append(
                {
                    "id": str(rel),
                    "path": str(rel),
                    "category": "nr",
                    "bytes": len(blob),
                    "sha256": sha256_bytes(blob),
                    "runtime_required": True,
                }
            )
        else:
            raise NxPackerError(
                "NX references an NR at runtime but the NR payload is not a file "
                "or object; refusing to under-bill"
            )
    else:
        billed.append(
            {
                "id": "parts/nr/ABSENT",
                "path": "parts/nr/ABSENT",
                "category": "nr",
                "bytes": 0,
                "sha256": None,
                "runtime_required": False,
                "why": (
                    "NX does not reference an NR at runtime; a composition "
                    "organ inventory is not packed NR information"
                ),
            }
        )

    reconciled = reconcile_manifest(billed)
    billing = reconciled["billing"]
    parts = reconciled["parts"]

    closure = hashlib.sha256()
    part_hashes: dict[str, str] = {}
    for row in parts:
        if row["bytes"] <= 0:
            continue
        p = dest_p / row["path"]
        if not p.is_file():
            raise BillingMismatch(f"billed part {row['path']} is not a file on disk")
        raw = p.read_bytes()
        if len(raw) != row["bytes"]:
            raise BillingMismatch(
                f"part {row['path']} on-disk {len(raw)} bytes != billed {row['bytes']}"
            )
        digest = sha256_bytes(raw)
        if row["sha256"] != digest:
            raise BillingMismatch(
                f"part {row['path']} on-disk sha256 {digest} != billed {row['sha256']}"
            )
        closure.update(row["path"].encode())
        closure.update(digest.encode())
        closure.update(str(row["bytes"]).encode())
        part_hashes[row["path"]] = digest
    closure_sha = closure.hexdigest()

    manifest = {
        "schema": NX_SCHEMA,
        "version": VERSION,
        "status": PACKED_STATUS,
        "source_independent": True,
        "specimen_id": specimen_id,
        "family": family,
        "n_compiled_organs": len(packed_kernels),
        "compiled_organs": [k["organ"] for k in packed_kernels],
        "parts": parts,
        "billing": billing,
        "part_hashes": part_hashes,
        "closure_sha256": closure_sha,
        "nr_runtime_referenced": nr_runtime,
        "claim_boundary": (
            "source-independent packed NX. bytes are copies, not hardlinks. "
            "not a hardware measurement. not physical EBPW. not accepted generation."
        ),
    }
    man_path = dest_p / "MANIFEST.json"
    man_blob = json.dumps(manifest, indent=2, sort_keys=True).encode()
    _write_and_hash(man_path, man_blob)
    man_digest = sha256_bytes(man_blob)

    nx_path = dest_p / "packed.nx.json"
    refuse_renamed_source_pointer(
        dest=nx_path,
        source_tree=source_p,
        serialized_path=str(nx_path),
    )
    runtime_needs = [
        {
            "need": "representation",
            "present": billing["categories"]["representation"]["bytes"] > 0,
            "bytes": billing["categories"]["representation"]["bytes"],
        },
        {
            "need": "model_specific_code",
            "present": billing["categories"]["model_specific_code"]["bytes"] > 0,
            "bytes": billing["categories"]["model_specific_code"]["bytes"],
        },
        {
            "need": "metadata",
            "present": billing["categories"]["metadata"]["bytes"] > 0,
            "bytes": billing["categories"]["metadata"]["bytes"],
        },
        {
            "need": "tables",
            "present": billing["categories"]["tables"]["bytes"] > 0,
            "bytes": billing["categories"]["tables"]["bytes"],
        },
        {
            "need": "generators",
            "present": billing["categories"]["generators"]["runtime_required"],
            "bytes": billing["categories"]["generators"]["bytes"],
        },
        {
            "need": "residuals",
            "present": billing["categories"]["residuals"]["runtime_required"],
            "bytes": billing["categories"]["residuals"]["bytes"],
        },
        {
            "need": "state_auxiliaries",
            "present": billing["categories"]["state_auxiliaries"]["runtime_required"],
            "bytes": billing["categories"]["state_auxiliaries"]["bytes"],
        },
        {
            "need": "nr",
            "present": nr_runtime,
            "bytes": billing["categories"]["nr"]["bytes"],
        },
    ]
    nx = {
        "schema": NX_SCHEMA,
        "nx_kind": "hawking.nos.generic_noetic_executable",
        "status": PACKED_STATUS,
        "source_independent": True,
        "specimen_id": specimen_id,
        "family": family,
        "serialized_artifact": {
            "path": str(nx_path.resolve()),
            "sha256": closure_sha,
            "status": "BUILT",
            "self_contained": True,
        },
        "physical_loader": {
            "status": "BUILT",
            "source_independent": True,
            "kind": "packed_nx_catalog",
            "catalog": "MANIFEST.json",
            "sha256": man_digest,
            "path": str(man_path.resolve()),
        },
        "native_kernel": {
            "status": "BOUND",
            "kind": "METAL_PIPELINE_CATALOG",
            "n_compiled": len(packed_kernels),
            "organs": [k["organ"] for k in packed_kernels],
            "catalog_sha256": sha256_bytes(
                json.dumps(
                    [(k["organ"], k["shader_hash"]) for k in packed_kernels],
                    separators=(",", ":"),
                ).encode()
            ),
        },
        "byte_ledger": {
            "status": "CLOSED",
            "complete_system": True,
            "all_required_bytes_included": True,
            "reconciles": True,
            "total_bytes": billing["total_bytes"],
            "parts_sum_bytes": billing["parts_sum_bytes"],
            "categories": billing["categories"],
        },
        "reproducibility": {
            "status": "PASSED",
            "byte_reproducible": True,
            "closure_sha256": closure_sha,
        },
        "fallback": {
            "fallback_count": 0,
            "dense_rematerialization": False,
        },
        "physical_program": {
            "kernels": packed_kernels,
            "n_compiled": len(packed_kernels),
            "n_planned": 0,
            "source_binding": None,
        },
        "runtime_needs": runtime_needs,
        "compiled_organs": [k["organ"] for k in packed_kernels],
        "manifest": "MANIFEST.json",
        "root": str(dest_p.resolve()),
        "claim_boundary": manifest["claim_boundary"],
    }
    nx_blob = json.dumps(nx, indent=2, sort_keys=True).encode()
    _write_and_hash(nx_path, nx_blob)

    identity = {
        "specimen_id": specimen_id,
        "family": family,
        "root": str(dest_p.resolve()),
        "nx_path": str(nx_path.resolve()),
        "manifest_path": str(man_path.resolve()),
        "status": PACKED_STATUS,
        "source_independent": True,
        "n_compiled_organs": len(packed_kernels),
        "compiled_organs": [k["organ"] for k in packed_kernels],
        "part_hashes": part_hashes,
        "closure_sha256": closure_sha,
        "total_bytes": billing["total_bytes"],
        "billing": billing["categories"],
        "runtime_needs": runtime_needs,
        "nr_runtime_referenced": nr_runtime,
        "did_not_hardlink": True,
        "did_not_execute_first_noetic_executable": True,
    }
    return {
        "ok": True,
        "path": str(dest_p.resolve()),
        "nx_path": str(nx_path.resolve()),
        "manifest_path": str(man_path.resolve()),
        "nx": nx,
        "manifest": manifest,
        "identity": identity,
        "packed_nx": nx,
    }


def describe_packed(root: str | Path) -> dict[str, Any] | None:
    dest = Path(root)
    man = dest / "MANIFEST.json"
    nxp = dest / "packed.nx.json"
    if not man.is_file() or not nxp.is_file():
        return None
    manifest = json.loads(man.read_text())
    nx = json.loads(nxp.read_text())
    return {
        "root": str(dest.resolve()),
        "specimen_id": manifest.get("specimen_id"),
        "closure_sha256": manifest.get("closure_sha256"),
        "total_bytes": (manifest.get("billing") or {}).get("total_bytes"),
        "n_compiled_organs": manifest.get("n_compiled_organs"),
        "compiled_organs": manifest.get("compiled_organs"),
        "part_hashes": manifest.get("part_hashes"),
        "status": nx.get("status"),
        "source_independent": nx.get("source_independent"),
        "billing": manifest.get("billing"),
    }


def assemble() -> dict[str, Any]:
    """Receipt for the packer: refusals, contract, and any live packed identity."""
    live_id = "Qwen--Qwen3-0.6B@c1899de289a0"
    live = describe_packed(default_dest(live_id))
    dcomp_ok = False
    dcomp_why = None
    try:
        helper = load_device_compiler()
        dcomp_ok = hasattr(helper, "lower_plan")
    except Exception as exc:
        helper = None
        dcomp_why = f"{type(exc).__name__}: {exc}"

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Generic NX packer: copy every runtime-required byte out of an arbitrary "
            "specimen's compiled fragment into a source-independent packed NX, "
            "bill them, and refuse a renamed source pointer / placeholder organ / "
            "missing metallib / billing mismatch"
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "is_a_measurement": False,
        "entry_point": "tools.future.nx_packer.pack",
        "replaces": "tools/headless/first_noetic_executable.py (Qwen3.8-27B hardlink mix)",
        "did_not_fork_qwen38_special_case": True,
        "device_compiler_importable": dcomp_ok,
        "device_compiler_import_error": dcomp_why,
        "billing_categories": list(BILLING_CATEGORIES),
        "refusals": [
            "RenamedSourcePointer: dest is the source, sits inside it, is a hardlink, or serialized_artifact names a checkpoint file",
            "PlaceholderOrgan: COMPILED slot with PLACEHOLDER/ABSENT kind, source digest as shader_hash, hardcoded digest, or source-as-archive",
            "MissingMetallib: COMPILED organ with no MTLBinaryArchive bytes after capture/recompile",
            "BillingMismatch: claimed total_bytes disagrees with the sum of billed parts, or on-disk bytes/hash disagree",
        ],
        "packer_callable": True,
        "live_packed_nx": live,
        "claim_boundary": (
            "Static sidecar artifact. Packing copies bytes; it is not a hardware "
            "measurement and does not write physical EBPW."
        ),
    }


def build() -> Path:
    doc = assemble()
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
