"""SPECIMEN VERIFY — earn whole-tree verification offline, or say precisely why not.

`odyssey_launch` refuses on `specimen_curriculum_ready`, and it is right to:
every ModelLake manifest reports `n_sha256_verified < n_files` with the rest
"size only", and each specimen seal says `final_status: MANIFEST_ONLY` with an
explicit `remote_digest_limitation`. A specimen verified by SIZE is not a sealed
specimen, and Odyssey I must not start on one.

But the digests are not actually missing. HuggingFace writes a `.metadata`
sidecar per downloaded file:

    <commit_sha>
    <etag>          <- 64 hex = sha256 (LFS); 40 hex = git blob sha1 (small file)
    <timestamp>

So the large safetensors — nearly all the bytes, and the part that matters — DO
carry a published sha256 and can be verified here with no network. Small files
carry a git blob sha1, verifiable by the same rule git uses. What genuinely
cannot be verified is a file with no `.metadata` at all, and this reports those
by name rather than rounding them into a pass.

Three rules this module will not bend:

* **ModelLake is never mutated.** Files are opened read-only. The verdict lands
  in the sidecar partition; ModelLake's own manifests and seals are untouched.
* **Verified means recomputed.** A file counts only when its hash was computed
  here and matched a published digest. Size agreement is not verification and is
  reported as its own class.
* **A specimen is WHOLE_TREE_VERIFIED only when every file carries a digest and
  every one matched.** One undigested file means `PARTIAL_NO_REMOTE_DIGEST`, and
  that is a refusal, not a rounding error.

    python3 tools/future/specimen_verify.py --list
    python3 tools/future/specimen_verify.py --verify tiiuae--Falcon-H1-7B-Instruct@41e72f27effb
    python3 tools/future/specimen_verify.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tools.future._common import RECEIPTS, REPO, write_receipt

RECEIPT = "SPECIMEN_VERIFICATION.json"
SCHEMA = "hawking.future.specimen_verify.v1"

LAKE = Path("/Volumes/corpdrive/hawking-modellake")
SPECIMENS = LAKE / "specimens"
MANIFESTS = LAKE / "manifests"

# ModelLake writes these itself; they have no upstream digest by construction and
# are not part of the specimen's source identity.
LAKE_OWN_FILES = {"MODEL_LAKE_SPECIMEN_SEAL.json", "MODEL_LAKE_SPECIMEN_SEAL.sha256"}

# Specimens that live outside ModelLake. ModelLake still owns the lake; these are
# named explicitly, verified by exactly the same rule, and labelled with a
# different owner so nothing downstream can mistake one for a sealed lake
# specimen. The Qwen3.8-27B parent is here because it is the model the Doctor and
# Gravity tools read, it carries the same HuggingFace .metadata sidecars, and it
# is not in the lake.
EXTRA_SPECIMENS: dict[str, Path] = {
    "qwen3.8-27b-abliterated-bf16@local": Path(
        "/Volumes/corpdrive/personalmodel/correspondent/qwen3.8-27b-abliterated-bf16"
    ),
    # Inside ModelLake, but under partial/ rather than specimens/. The gate read
    # that as "identity known but the specimen is not in the ModelLake specimens
    # listing", which is true and was taken to mean the model is not here. It is
    # here: nine files, nine .metadata sidecars, zero .incomplete markers. Only
    # its LOCATION is partial. Verified read-only by the same rule as everything
    # else; ModelLake owns it and nothing here moves, renames, or writes to it.
    "Qwen--Qwen3-0.6B@c1899de289a0#partial": Path(
        "/Volumes/corpdrive/hawking-modellake/partial/Qwen--Qwen3-0.6B@c1899de289a0"
    ),
}

# Written by the local mirror, not published by the source repo. crc32.txt covers
# only the small files here and never the weights, so it is not a digest source.
LOCAL_OWN_FILES = {"crc32.txt"}

CHUNK = 8 << 20


class SpecimenError(Exception):
    pass


def available() -> dict[str, Any]:
    """ModelLake may not be mounted. Cope, and say so, rather than raising."""
    return {
        "lake": str(LAKE),
        "mounted": LAKE.is_dir(),
        "specimens_dir": SPECIMENS.is_dir(),
        "n_specimens": len(list(SPECIMENS.iterdir())) if SPECIMENS.is_dir() else 0,
    }


def specimen_dir(name: str) -> Path:
    """Where this specimen lives. The lake first, then named local directories."""
    if name in EXTRA_SPECIMENS:
        return EXTRA_SPECIMENS[name]
    return SPECIMENS / name


# Where a non-specimens-listing specimen actually lives. "outside the lake" and
# "inside the lake but not under specimens/" are different provenances and must
# not share a label: one is an authorized external directory, the other is
# ModelLake's own storage in a staging location.
EXTRA_OWNER: dict[str, str] = {
    "qwen3.8-27b-abliterated-bf16@local": "local_directory",
    "Qwen--Qwen3-0.6B@c1899de289a0#partial": "modellake_partial",
}


def specimen_owner(name: str) -> str:
    if name in EXTRA_SPECIMENS:
        return EXTRA_OWNER.get(name, "local_directory")
    return "modellake"


def list_specimens() -> list[str]:
    names = []
    if SPECIMENS.is_dir():
        names.extend(p.name for p in SPECIMENS.iterdir() if p.is_dir())
    names.extend(n for n, d in EXTRA_SPECIMENS.items() if d.is_dir())
    return sorted(set(names))


def _read_metadata(spec_dir: Path, rel: str) -> dict[str, Any] | None:
    meta = spec_dir / ".cache" / "huggingface" / "download" / f"{rel}.metadata"
    if not meta.is_file():
        return None
    lines = [l.strip() for l in meta.read_text(errors="replace").splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    etag = lines[1]
    if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag):
        kind = "sha256"
    elif len(etag) == 40 and all(c in "0123456789abcdef" for c in etag):
        kind = "git_blob_sha1"
    else:
        kind = "unrecognized"
    return {"commit": lines[0], "etag": etag, "digest_kind": kind}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    """git's own object id: sha1 of "blob <len>\\0" + content."""
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def specimen_files(name: str) -> list[Path]:
    d = specimen_dir(name)
    if not d.is_dir():
        raise SpecimenError(f"specimen not present: {name}")
    skip = LAKE_OWN_FILES | LOCAL_OWN_FILES
    return sorted(p for p in d.iterdir() if p.is_file() and p.name not in skip)


def verify_specimen(name: str, *, max_seconds: float | None = None) -> dict[str, Any]:
    """Recompute and compare. Read-only against ModelLake, always."""
    started = time.time()
    files = specimen_files(name)
    rows: list[dict[str, Any]] = []
    matched = mismatched = no_digest = unrecognized = skipped = 0
    bytes_hashed = 0

    for path in files:
        if max_seconds is not None and (time.time() - started) > max_seconds:
            skipped += 1
            rows.append({"file": path.name, "verdict": "SKIPPED_TIME_BUDGET"})
            continue
        meta = _read_metadata(specimen_dir(name), path.name)
        size = path.stat().st_size
        if meta is None:
            no_digest += 1
            rows.append({"file": path.name, "bytes": size, "verdict": "NO_REMOTE_DIGEST",
                         "why": "no HuggingFace .metadata sidecar; nothing to compare against"})
            continue
        kind = meta["digest_kind"]
        if kind == "sha256":
            actual = _sha256(path)
        elif kind == "git_blob_sha1":
            actual = _git_blob_sha1(path)
        else:
            unrecognized += 1
            rows.append({"file": path.name, "bytes": size, "verdict": "UNRECOGNIZED_DIGEST",
                         "etag": meta["etag"]})
            continue
        bytes_hashed += size
        ok = actual == meta["etag"]
        (matched := matched + 1) if ok else (mismatched := mismatched + 1)
        rows.append({
            "file": path.name, "bytes": size, "digest_kind": kind,
            "expected": meta["etag"], "actual": actual,
            "verdict": "VERIFIED" if ok else "MISMATCH",
        })

    n = len(files)
    whole_tree = (matched == n and n > 0)
    if whole_tree:
        status = "WHOLE_TREE_VERIFIED"
    elif mismatched:
        status = "CORRUPT_MISMATCH"
    elif skipped:
        status = "INCOMPLETE_TIME_BUDGET"
    else:
        status = "PARTIAL_NO_REMOTE_DIGEST"

    return {
        "specimen": name,
        "owner": specimen_owner(name),
        "specimen_path": str(specimen_dir(name)),
        "n_files": n,
        "verified": matched,
        "mismatched": mismatched,
        "no_remote_digest": no_digest,
        "unrecognized_digest": unrecognized,
        "skipped_time_budget": skipped,
        "bytes_hashed": bytes_hashed,
        "wall_seconds": round(time.time() - started, 2),
        "status": status,
        "whole_tree_verified": whole_tree,
        "modellake_mutated": False,
        "source_mutated": False,
        "files": rows,
    }


def record(result: dict[str, Any]) -> Path:
    """Merge one specimen's verdict into the receipt, keyed by specimen name.

    Whole-tree verification of the large specimens takes hours, and a single
    --build that must finish them all before anything is written loses every
    completed specimen when the window closes. Verifying one specimen is
    complete work and is persisted as such; the receipt is the union of what has
    actually been recomputed, never a claim about what has not.
    """
    name = str(result.get("specimen") or "")
    known = list_specimens()
    if known and name not in known:
        # The Odyssey gate reads this receipt to decide specimen readiness, so a
        # row naming something that is not a specimen is a readiness claim about
        # a model that does not exist. A test fixture leaked exactly one such row
        # in here once; it fails closed now.
        raise SpecimenError(
            f"refusing to record {name!r}: not a ModelLake specimen directory"
        )
    prior: list[dict[str, Any]] = []
    path = RECEIPTS / RECEIPT
    if path.is_file():
        try:
            prior = list(json.loads(path.read_text()).get("results") or [])
        except (json.JSONDecodeError, OSError):
            prior = []
    merged = [r for r in prior if r.get("specimen") != result.get("specimen")]
    merged.append(result)
    merged.sort(key=lambda r: str(r.get("specimen")))
    return _receipt(merged)


# A whole-lake pass is not a bounded unit. ModelLake went from 7 specimens to 43
# during one autonomy trial as the download workers promoted a batch, and
# build() -- iterating every specimen at 900s each -- became an eleven-hour call
# that ran 37 minutes past the end of a 1-hour trial with no way to stop it.
# A capability the resident invokes must finish, or report honestly what it did
# not reach.
BUILD_TOTAL_BUDGET_S = 480.0


def build(
    names: list[str] | None = None,
    *,
    max_seconds_each: float | None = 900.0,
    max_total_seconds: float | None = BUILD_TOTAL_BUDGET_S,
) -> Path:
    started = time.time()
    avail = available()
    results: list[dict[str, Any]] = []
    not_reached: list[str] = []
    if avail["mounted"]:
        # Cheapest first, and prefer specimens with no verdict yet: a rerun that
        # spends its whole budget rehashing what is already verified learns
        # nothing.
        prior = {}
        path = RECEIPTS / RECEIPT
        if path.is_file():
            try:
                prior = {r.get("specimen"): r for r in
                         (json.loads(path.read_text()).get("results") or [])}
            except (json.JSONDecodeError, OSError):
                prior = {}

        def _size(name: str) -> int:
            try:
                return sum(f.stat().st_size for f in specimen_files(name))
            except Exception:
                return 0

        queue = list(names or list_specimens())
        queue.sort(key=lambda n: (n in prior, _size(n)))
        for name in queue:
            if max_total_seconds is not None and (time.time() - started) > max_total_seconds:
                not_reached.append(name)
                continue
            try:
                results.append(verify_specimen(name, max_seconds=max_seconds_each))
            except SpecimenError as exc:
                results.append({"specimen": name, "status": "ABSENT", "why": str(exc)})
        # Anything verified in an earlier pass and not revisited stays on the
        # record; dropping it would make a bounded run look like a regression.
        seen = {r.get("specimen") for r in results}
        for name, row in prior.items():
            if name not in seen:
                results.append(row)
                if name not in not_reached:
                    not_reached.append(str(name))

    return _receipt(results, not_reached=not_reached)


def _receipt(results: list[dict[str, Any]], *, not_reached: list[str] | None = None) -> Path:
    avail = available()
    sealed = [r["specimen"] for r in results if r.get("whole_tree_verified")]
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Earn whole-tree specimen verification offline from the HuggingFace "
            ".metadata digests, or state precisely which files cannot be verified."
        ),
        "modellake_authority": (
            "ModelLake owns specimens. This module opens them read-only, never writes "
            "into the lake, and never edits a manifest or a seal. The verdict lives in "
            "the sidecar partition."
        ),
        "verification_rule": (
            "VERIFIED means the hash was recomputed here and matched a published "
            "digest. Size agreement is not verification. A specimen is "
            "WHOLE_TREE_VERIFIED only when every file carried a digest and every one "
            "matched; a single undigested file yields PARTIAL_NO_REMOTE_DIGEST."
        ),
        "modellake": avail,
        "counts": {
            "specimens_examined": len(results),
            "whole_tree_verified": len(sealed),
            "partial": sum(1 for r in results if r.get("status") == "PARTIAL_NO_REMOTE_DIGEST"),
            "corrupt": sum(1 for r in results if r.get("status") == "CORRUPT_MISMATCH"),
            "incomplete": sum(1 for r in results if r.get("status") == "INCOMPLETE_TIME_BUDGET"),
        },
        "whole_tree_verified_specimens": sealed,
        "not_reached_this_pass": list(not_reached or []),
        "budget_rule": (
            "build() is bounded so a resident can invoke it as a unit. Specimens "
            "it did not reach are named rather than silently omitted, and a prior "
            "verdict is carried forward rather than dropped."
        ),
        "results": results,
        "recovered_implementation": [
            "ModelLake manifests report n_sha256_verified vs n_size_only_verified",
            "each specimen carries MODEL_LAKE_SPECIMEN_SEAL.json, currently MANIFEST_ONLY",
            "HuggingFace .cache/huggingface/download/<file>.metadata carries the etag",
        ],
        "gaps_closed": [
            "nothing recomputed specimen hashes offline; readiness rested on size checks",
        ],
        "negative_findings": [
            "a file with no .metadata sidecar cannot be verified offline at any effort",
            "this does not make ModelLake wrong: its seals already say MANIFEST_ONLY",
        ],
        "resident_callable": {
            "entry_point": "tools.future.specimen_verify.verify_specimen(name)",
            "workunit": "one unit per specimen; CPU_VERIFY lane; genuinely long-running",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_CAPABILITY.hard-gates (specimen curriculum readiness)",
            "fails_closed": "SpecimenError on an absent specimen; MISMATCH is CORRUPT, never rounded",
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/specimen_verify.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", metavar="SPECIMEN")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=900.0)
    a = ap.parse_args()
    if a.list:
        print(json.dumps({"available": available(), "specimens": list_specimens()}, indent=1))
        return 0
    if a.verify:
        res = verify_specimen(a.verify, max_seconds=a.max_seconds)
        out = record(res)
        res.pop("files", None)
        print(json.dumps(res, indent=1, sort_keys=True))
        print(out)
        return 0
    out = build(max_seconds_each=a.max_seconds)
    doc = json.loads(out.read_text())
    print(out)
    print(json.dumps(doc["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
