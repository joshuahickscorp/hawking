"""SEAL THE EXTERNAL QWEN PARENT — give the non-lake specimen a recomputable identity.

ModelLake remains the canonical acquisition system. The Qwen3.8-27B parent the
Doctor and Gravity tools actually read lives outside the lake, has already been
whole-tree verified (31/31, 55.6 GB recomputed, zero mismatches), and still
refuses curriculum readiness because a local directory has no repository
revision to seal against. Location is not authority. This module is the seal
that gives it an identity: a tree digest over a sorted per-file sha256
manifest, bound to that verification, labelled authorized-external rather than
lake stock.

It refuses rather than guesses:

* no WHOLE_TREE_VERIFIED row, no seal — a partial, corrupt, or absent
  verification is not rounded into an identity;
* a lake specimen cannot take this path, even if someone stamps a tree digest
  onto it;
* config.json is read for architecture; a parameter count that is not in
  config.json is not invented from layer arithmetic or from the directory name;
* the specimen directory is opened read-only, never moved, never renamed, never
  written. Safetensors are not re-hashed here; their sha256 is recovered from
  the verification that already recomputed them.

It cannot establish that a same-size substitution in a weight file happened
after verification. That detection still belongs to specimen_verify. This seal
binds to that receipt and to the file set and sizes on disk.

    python3 tools/future/external_specimen_seal.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import RECEIPTS, REPO, git, load_json, sha256_file, write_receipt
from tools.future import specimen_verify as sv

RECEIPT = "EXTERNAL_SPECIMEN_SEAL.json"
SCHEMA = "hawking.future.external_specimen_seal.v1"
SPECIMEN_NAME = "qwen3.8-27b-abliterated-bf16@local"
VERIFICATION_REL = "receipts/future/SPECIMEN_VERIFICATION.json"
# tokenizer.json is 12 MB; anything at safetensor scale is recovered, never rehashed.
MAX_REHASH_BYTES = 64 << 20
SAFETENSOR_SUFFIX = ".safetensors"
LAKE_PATH_MARK = "hawking-modellake"


class SealError(Exception):
    """A seal was requested that this module must refuse."""


def authorized_path() -> Path | None:
    return sv.EXTRA_SPECIMENS.get(SPECIMEN_NAME)


def _norm(path: str | Path) -> str:
    return str(path).rstrip("/")


def _search_json(rel: str) -> dict[str, Any] | None:
    """Worktree first, then the primary checkout, then git HEAD. A miss is None."""
    rel = rel.replace("\\", "/").lstrip("./")
    candidates = [REPO / rel]
    common = git("rev-parse", "--git-common-dir")
    if common:
        path = Path(common)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        else:
            path = path.resolve()
        parent = path.parent if path.name == ".git" else path.parent
        candidates.append(parent / rel)
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            try:
                return load_json(p)
            except (OSError, json.JSONDecodeError):
                continue
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None
    return None


def load_verification_doc() -> dict[str, Any] | None:
    return _search_json(VERIFICATION_REL)


def verification_row(doc: Mapping[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not isinstance(doc, Mapping):
        return None
    for row in doc.get("results") or []:
        if isinstance(row, Mapping) and row.get("specimen") == name:
            return dict(row)
    return None


def is_whole_tree_row(row: Mapping[str, Any] | None) -> bool:
    """Same strictness as odyssey_launch._independently_verified. Status is a hypothesis."""
    if not isinstance(row, Mapping):
        return False
    if row.get("status") != "WHOLE_TREE_VERIFIED":
        return False
    if not (isinstance(row.get("bytes_hashed"), int) and row["bytes_hashed"] > 0):
        return False
    if row.get("mismatched") or row.get("no_remote_digest"):
        return False
    if row.get("unrecognized_digest"):
        return False
    if row.get("skipped_time_budget"):
        return False
    if row.get("verified") != row.get("n_files"):
        return False
    n = row.get("n_files")
    return isinstance(n, int) and n > 0


def canonicalize_manifest(manifest: Sequence[Mapping[str, Any]]) -> bytes:
    """Sort first. Reordering the input must not change the digest."""
    rows = []
    for item in manifest:
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("bytes")
        if not isinstance(path, str) or not path:
            raise SealError("manifest entry missing path")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise SealError(f"manifest entry {path!r} has no sha256; refusing a partial tree")
        if not isinstance(size, int) or size < 0:
            raise SealError(f"manifest entry {path!r} has no byte size")
        rows.append((path, digest, size))
    rows.sort(key=lambda r: r[0])
    return "".join(f"{path}\t{digest}\t{size}\n" for path, digest, size in rows).encode()


def tree_digest(manifest: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonicalize_manifest(manifest)).hexdigest()


def manifest_from_directory(root: Path) -> list[dict[str, Any]]:
    """Fixture-scale trees only. A digest of an empty directory is not an identity."""
    if not root.is_dir():
        raise SealError(f"directory absent: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.iterdir() if p.is_file()):
        rows.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "sha256_source": "recomputed_here",
            }
        )
    if not rows:
        raise SealError("empty tree; a digest of nothing is not a specimen identity")
    return rows


def _file_sha256_from_verification(entry: Mapping[str, Any]) -> str | None:
    if entry.get("verdict") != "VERIFIED":
        return None
    if entry.get("digest_kind") != "sha256":
        return None
    actual = entry.get("actual")
    expected = entry.get("expected")
    if not isinstance(actual, str) or len(actual) != 64:
        return None
    if isinstance(expected, str) and expected != actual:
        return None
    if any(c not in "0123456789abcdef" for c in actual):
        return None
    return actual


def _rehash_allowed(entry: Mapping[str, Any], spec_dir: Path | None) -> Path | None:
    """Safetensors are never rehashed here. Small files may be, if present and small."""
    name = entry.get("file")
    if not isinstance(name, str) or not name:
        return None
    if name.endswith(SAFETENSOR_SUFFIX):
        return None
    if spec_dir is None:
        return None
    path = spec_dir / name
    if not path.is_file():
        return None
    size = path.stat().st_size
    recorded = entry.get("bytes")
    if isinstance(recorded, int) and size != recorded:
        raise SealError(
            f"{name}: on-disk size {size} != verified size {recorded}; tree changed"
        )
    if size > MAX_REHASH_BYTES:
        raise SealError(
            f"{name}: {size} bytes exceeds rehash cap {MAX_REHASH_BYTES}; "
            "refusing to hash a weight-scale file in the seal path"
        )
    return path


def manifest_from_verification(
    row: Mapping[str, Any],
    *,
    spec_dir: Path | None = None,
    hash_small: bool = True,
) -> list[dict[str, Any]]:
    """Recover sha256 for files the verifier already hashed as sha256.

    git-blob sidecars carry a sha1, not a sha256. Those files are small; if
    hash_small and the directory is present they are hashed here, read-only.
    A missing sha256 is a refusal, never an omitted row.
    """
    files = row.get("files") or []
    if not isinstance(files, list) or not files:
        raise SealError("verification row carries no file list")
    manifest: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, Mapping):
            raise SealError("verification file list is malformed")
        name = entry.get("file")
        size = entry.get("bytes")
        if not isinstance(name, str) or not isinstance(size, int):
            raise SealError(f"verification file entry missing name or size: {entry!r}")
        recovered = _file_sha256_from_verification(entry)
        source = "recovered_from_verification_actual"
        digest = recovered
        if digest is None:
            if not hash_small:
                raise SealError(
                    f"{name}: no sha256 in the verification row and rehash disabled"
                )
            path = _rehash_allowed(entry, spec_dir)
            if path is None:
                raise SealError(
                    f"{name}: no recovered sha256 and cannot rehash "
                    f"(digest_kind={entry.get('digest_kind')!r})"
                )
            digest = sha256_file(path)
            source = "recomputed_here"
        manifest.append(
            {
                "path": name,
                "sha256": digest,
                "bytes": size,
                "sha256_source": source,
                "verification_verdict": entry.get("verdict"),
                "verification_digest_kind": entry.get("digest_kind"),
            }
        )
    return manifest


def _read_json_readonly(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode())


def read_model_identity(spec_dir: Path) -> dict[str, Any]:
    """Architecture from config.json. A missing parameter count is a refusal, not a guess."""
    cfg_path = spec_dir / "config.json"
    if not cfg_path.is_file():
        return {
            "ok": False,
            "why": "config.json absent; refusing to invent architecture",
            "architecture": None,
            "parameter_count": None,
        }
    cfg = _read_json_readonly(cfg_path)
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    architecture = {
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "text_model_type": text.get("model_type"),
        "hidden_size": text.get("hidden_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "vocab_size": text.get("vocab_size"),
        "dtype": text.get("dtype"),
        "max_position_embeddings": text.get("max_position_embeddings"),
    }
    parameter_count = None
    parameter_count_key = None
    for key in ("num_parameters", "n_params", "params", "parameter_count"):
        value = cfg.get(key)
        if isinstance(value, int) and value > 0:
            parameter_count = value
            parameter_count_key = key
            break
    return {
        "ok": True,
        "architecture": architecture,
        "parameter_count": parameter_count,
        "parameter_count_key": parameter_count_key,
        "parameter_count_source": (
            f"config.json:{parameter_count_key}" if parameter_count_key else None
        ),
        "parameter_count_refused": (
            None
            if parameter_count is not None
            else (
                "config.json does not name a parameter count; refusing to invent "
                "one from layer arithmetic or from the directory name"
            )
        ),
    }


def read_tokenizer_identity(
    spec_dir: Path, manifest: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    cfg_path = spec_dir / "tokenizer_config.json"
    if not cfg_path.is_file():
        return {"ok": False, "why": "tokenizer_config.json absent"}
    cfg = _read_json_readonly(cfg_path)
    by_name = {row["path"]: row for row in manifest if isinstance(row.get("path"), str)}
    return {
        "ok": True,
        "tokenizer_class": cfg.get("tokenizer_class"),
        "model_max_length": cfg.get("model_max_length"),
        "bos_token": cfg.get("bos_token"),
        "eos_token": cfg.get("eos_token"),
        "pad_token": cfg.get("pad_token"),
        "tokenizer_json_sha256": (by_name.get("tokenizer.json") or {}).get("sha256"),
        "tokenizer_config_sha256": (by_name.get("tokenizer_config.json") or {}).get("sha256"),
    }


def _assert_file_set_matches(spec_dir: Path, row: Mapping[str, Any]) -> None:
    """A file added or removed since verification is a different tree."""
    try:
        on_disk = {p.name for p in sv.specimen_files(SPECIMEN_NAME)}
    except sv.SpecimenError as exc:
        raise SealError(str(exc)) from exc
    recorded = {e.get("file") for e in (row.get("files") or []) if isinstance(e, Mapping)}
    if on_disk != recorded:
        raise SealError(
            "file set changed since verification: "
            f"added={sorted(on_disk - recorded)} removed={sorted(recorded - on_disk)}"
        )
    for entry in row.get("files") or []:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("file")
        size = entry.get("bytes")
        if not isinstance(name, str) or not isinstance(size, int):
            continue
        path = spec_dir / name
        if not path.is_file():
            raise SealError(f"{name}: named by verification but absent on disk")
        disk_size = path.stat().st_size
        if disk_size != size:
            raise SealError(
                f"{name}: on-disk size {disk_size} != verified size {size}; tree changed"
            )


def load_seal() -> dict[str, Any] | None:
    path = RECEIPTS / RECEIPT
    if not path.is_file():
        return None
    try:
        doc = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _is_lake_identity(identity: Mapping[str, Any]) -> bool:
    owner = identity.get("specimen_owner")
    if owner == "modellake":
        return True
    path = str(identity.get("specimen_path") or "")
    if LAKE_PATH_MARK in path.replace("\\", "/"):
        return True
    return False


def _is_this_external(identity: Mapping[str, Any]) -> tuple[bool, str]:
    if _is_lake_identity(identity):
        return False, "lake specimens cannot use an external tree digest as identity"
    name = identity.get("specimen")
    path = identity.get("specimen_path")
    expected = authorized_path()
    name_ok = name == SPECIMEN_NAME
    path_ok = bool(expected) and bool(path) and _norm(path) == _norm(expected)
    if not (name_ok or path_ok):
        return False, "not the named authorized extra specimen"
    owner = identity.get("specimen_owner")
    if owner not in {None, "local_directory"} and not identity.get("authorized_external"):
        return False, f"specimen_owner={owner!r} is not local_directory"
    return True, "authorized external specimen"


def accept_as_sealed_identity(
    identity: Mapping[str, Any],
    *,
    seal: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Lake specimens never pass. External specimens pass only with a matching seal.

    Called from odyssey_launch._ready when a specimen has no repository
    revision. Returning False leaves the original refusal in place.
    """
    ok, why = _is_this_external(identity)
    if not ok:
        return False, why
    if not identity.get("whole_tree_verified"):
        return False, "external specimen is not whole-tree verified; no identity to seal against"
    doc = seal if seal is not None else load_seal()
    if not isinstance(doc, Mapping) or doc.get("status") != "SEALED":
        return False, "no external specimen seal"
    sealed_digest = doc.get("tree_digest")
    if not isinstance(sealed_digest, str) or len(sealed_digest) != 64:
        return False, "external seal carries no tree digest"
    offered = identity.get("tree_digest")
    if isinstance(offered, str) and offered and offered != sealed_digest:
        return False, "offered tree digest does not match the sealed identity"
    sealed_path = doc.get("specimen_path")
    identity_path = identity.get("specimen_path")
    if (
        isinstance(sealed_path, str)
        and isinstance(identity_path, str)
        and _norm(sealed_path) != _norm(identity_path)
    ):
        return False, "identity path does not match the sealed external specimen"
    sealed_name = doc.get("specimen")
    identity_name = identity.get("specimen")
    if (
        isinstance(sealed_name, str)
        and isinstance(identity_name, str)
        and sealed_name != identity_name
    ):
        return False, "identity name does not match the sealed external specimen"
    return True, (
        "authorized external specimen; tree digest is sealed identity "
        f"({sealed_digest[:12]}…)"
    )


def seal_from_verification(
    row: Mapping[str, Any] | None,
    *,
    spec_dir: Path | None = None,
    hash_small: bool = True,
    verification_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if row is None:
        raise SealError("no verification row; an unverified external specimen is not sealable")
    if row.get("owner") not in {None, "local_directory"} and row.get("specimen") == SPECIMEN_NAME:
        # The live extra specimen is labelled local_directory. A lake-owned row
        # of the same name would be a disguise; refuse it.
        if row.get("owner") == "modellake":
            raise SealError("refusing to seal a lake-owned row as an external specimen")
    if row.get("specimen") != SPECIMEN_NAME:
        raise SealError(
            f"refusing to seal {row.get('specimen')!r}; "
            f"this module authorizes only {SPECIMEN_NAME}"
        )
    if not is_whole_tree_row(row):
        raise SealError(
            "external specimen has no whole-tree verification; not sealable "
            f"(status={row.get('status')!r} verified={row.get('verified')!r}/"
            f"{row.get('n_files')!r} mismatched={row.get('mismatched')!r} "
            f"no_remote_digest={row.get('no_remote_digest')!r})"
        )
    if spec_dir is not None:
        if not spec_dir.is_dir():
            raise SealError(f"specimen directory absent: {spec_dir}")
        expected = authorized_path()
        if expected is not None and _norm(spec_dir) != _norm(expected):
            raise SealError(
                f"refusing to seal {spec_dir}; authorized path is {expected}"
            )
        _assert_file_set_matches(spec_dir, row)

    manifest = manifest_from_verification(row, spec_dir=spec_dir, hash_small=hash_small)
    digest = tree_digest(manifest)
    # In-process recomputation must be byte-identical even if the caller shuffled.
    if tree_digest(list(reversed(manifest))) != digest:
        raise SealError("tree digest is order-dependent; sort is broken")

    identity: dict[str, Any]
    tokenizer: dict[str, Any]
    if spec_dir is not None:
        identity = read_model_identity(spec_dir)
        tokenizer = read_tokenizer_identity(spec_dir, manifest)
        if not identity.get("ok"):
            raise SealError(identity.get("why") or "model identity unreadable")
        if not tokenizer.get("ok"):
            raise SealError(tokenizer.get("why") or "tokenizer identity unreadable")
    else:
        identity = {
            "ok": False,
            "why": "specimen directory not opened; architecture not read",
            "architecture": None,
            "parameter_count": None,
        }
        tokenizer = {"ok": False, "why": "specimen directory not opened"}

    recovered_hashes = sum(
        1 for m in manifest if m.get("sha256_source") == "recovered_from_verification_actual"
    )
    recomputed_hashes = sum(
        1 for m in manifest if m.get("sha256_source") == "recomputed_here"
    )
    total_bytes = sum(int(m["bytes"]) for m in manifest)

    recorded_at = None
    if isinstance(verification_doc, Mapping):
        bench = verification_doc.get("bench")
        if isinstance(bench, Mapping):
            recorded_at = bench.get("recorded_at")

    return {
        "status": "SEALED",
        "kind": "authorized_external_specimen",
        "not_lake_stock": True,
        "modellake_authority": (
            "ModelLake remains the canonical acquisition system. This is an "
            "authorized EXTERNAL specimen, not lake stock."
        ),
        "read_only_expectation": (
            "The specimen directory is read-only. This module never moves it, "
            "never renames it, and never writes inside it."
        ),
        "location_is_not_authority": True,
        "specimen": SPECIMEN_NAME,
        "specimen_owner": "local_directory",
        "specimen_path": str(spec_dir) if spec_dir is not None else row.get("specimen_path"),
        "n_files": len(manifest),
        "total_bytes": total_bytes,
        "tree_digest": digest,
        "manifest": manifest,
        "model_identity": identity,
        "tokenizer_identity": tokenizer,
        "verification": {
            "status": row.get("status"),
            "whole_tree_verified": True,
            "n_files": row.get("n_files"),
            "verified": row.get("verified"),
            "bytes_hashed": row.get("bytes_hashed"),
            "wall_seconds": row.get("wall_seconds"),
            "recorded_at": recorded_at,
            "source_receipt": VERIFICATION_REL,
        },
        "sha256_sources": {
            "recovered_from_verification_actual": recovered_hashes,
            "recomputed_here": recomputed_hashes,
            "safetensors_rehashed": False,
        },
        "specimen_mutated": False,
        "modellake_mutated": False,
    }


def seal_authorized_external() -> dict[str, Any]:
    """Live path: recover verification, confirm the directory, write a seal or a refusal."""
    if SPECIMEN_NAME not in sv.EXTRA_SPECIMENS:
        return {
            "status": "REFUSED",
            "why": (
                f"{SPECIMEN_NAME} is not in specimen_verify.EXTRA_SPECIMENS; "
                "this module does not invent extra specimens"
            ),
        }
    doc = load_verification_doc()
    if doc is None:
        return {
            "status": "REFUSED",
            "why": (
                "SPECIMEN_VERIFICATION.json not found in this worktree, the "
                "primary checkout, or git HEAD; an unverified external specimen "
                "is not sealable"
            ),
        }
    row = verification_row(doc, SPECIMEN_NAME)
    spec_dir = authorized_path()
    if spec_dir is None or not spec_dir.is_dir():
        return {
            "status": "REFUSED",
            "why": (
                f"authorized specimen directory is not present at {spec_dir}; "
                "refusing to seal against a missing tree"
            ),
            "verification_row_present": row is not None,
        }
    try:
        return seal_from_verification(
            row, spec_dir=spec_dir, hash_small=True, verification_doc=doc
        )
    except SealError as exc:
        return {"status": "REFUSED", "why": str(exc)}


def build() -> Path:
    sealed = seal_authorized_external()
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Give the non-lake Qwen3.8-27B parent a recomputable sealed identity "
            "(sorted-manifest tree digest) so Odyssey I curriculum can accept it "
            "without weakening any lake rule."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        **sealed,
        "recovered_implementation": [
            "tools/future/specimen_verify.py — EXTRA_SPECIMENS, whole-tree verification, owner=local_directory",
            "tools/future/odyssey_launch.py — _ready, _independently_verified, propose_specimen_curriculum",
            "receipts/future/SPECIMEN_VERIFICATION.json — qwen3.8-27b-abliterated-bf16@local WHOLE_TREE_VERIFIED 31/31",
        ],
        "gaps_closed": [
            "the verified local parent had no sealed identity (no repository revision, no patient seal)",
            "odyssey_launch._ready now accepts a matching external tree digest without opening a lake hole",
        ],
        "negative_findings": [
            "config.json does not name a parameter count; this seal does not invent one",
            "safetensors were not re-hashed; a same-size substitution after verification is out of scope here",
            "a local directory still has no repository revision; the tree digest is the identity, not a git commit",
            "this does not make the specimen lake stock",
        ],
        "resident_callable": {
            "entry_point": "tools.future.external_specimen_seal.seal_authorized_external()",
            "workunit": (
                "one CPU_ANALYSIS unit; recover verification, hash only small "
                "non-weight files, emit the tree digest; never take a GPU lease"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_CAPABILITY.hard-gates",
            "fails_closed": (
                "SealError / REFUSED when verification is missing, partial, or "
                "corrupt; lake identities cannot use the tree-digest path; "
                "absent config.json is a refusal, not a default architecture"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/external_specimen_seal.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
