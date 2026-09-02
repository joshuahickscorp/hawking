"""The specimen registry HCLI reasons over: every sealed specimen, from disk.

47 sealed specimens sit under ``/Volumes/corpdrive/hawking-modellake/specimens``
today (override: ``HCLI_MODEL_LAKE_ROOT``), one directory per
``<org>--<repo>@<revision>``. This module enumerates them fresh on every call
-- never a hand-maintained list -- and reports, per specimen, what a consumer
needs to reason about it: identity, revision, size, architecture where
derivable, whether a manifest exists, and whether that manifest verifies the
specimen complete. Manifests live in
``workspace/campaign/odyssey/watch-manifests/`` (override:
``HCLI_SPECIMEN_MANIFEST_DIR``), not on the ModelLake volume itself.

SEALED DOES NOT MEAN LOAD NOW. A 135 GB source is scientifically useful
without fitting as a live resident on a 96 GB machine. This module reports
facts -- size, architecture, completeness -- and makes no residency,
extraction, or scheduling judgement. That decision belongs to whatever reads
this registry, never to the registry itself.

Because nothing here is cached, a specimen that lands on disk between two
calls is present in the very next call -- no restart, no registration step.
That is the whole mechanism for joining an in-flight campaign mid-flight.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
SCHEMA = "hcli.specimens.registry.v1"


def _lake_root() -> Path:
    configured = os.environ.get("HCLI_MODEL_LAKE_ROOT")
    return Path(configured).expanduser() if configured else Path("/Volumes/corpdrive/hawking-modellake")


def _manifest_dir() -> Path:
    configured = os.environ.get("HCLI_SPECIMEN_MANIFEST_DIR")
    if configured:
        return Path(configured).expanduser()
    return REPO / "workspace" / "campaign" / "odyssey" / "watch-manifests"


def _split_id(name: str) -> tuple[str, Optional[str]]:
    repo, _, rev = name.partition("@")
    return repo.replace("--", "/"), (rev or None)


def _manifest_for(name: str, manifest_dir: Path) -> Optional[dict]:
    path = manifest_dir / f"{name}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _architecture(config_path: Path) -> dict[str, Any]:
    """model_type/hidden_size/num_hidden_layers/architectures, or all-None.

    A missing or unparseable config.json is not an error here -- three of
    the 47 real specimens (Wan2.2, boltz-2, moshika) ship no HF-style config
    at all, and that absence is itself a fact worth reporting, not a reason
    to fail the whole specimen.
    """
    empty = {"model_type": None, "hidden_size": None, "num_hidden_layers": None, "architectures": None}
    if not config_path.is_file():
        return dict(empty)
    try:
        cfg = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return dict(empty)
    # VL/multimodal configs (e.g. Qwen3-VL) nest the language-model shape
    # under text_config; fall back to it when the top level omits a field.
    text = cfg.get("text_config") or {}
    def pick(key: str) -> Any:
        value = cfg.get(key)
        return value if value is not None else text.get(key)
    return {
        "model_type": cfg.get("model_type"),
        "hidden_size": pick("hidden_size"),
        "num_hidden_layers": pick("num_hidden_layers"),
        "architectures": cfg.get("architectures"),
    }


def _dir_bytes(d: Path) -> int:
    """Stat-only recursive size -- no file content is ever read."""
    total = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _verify_complete(d: Path, manifest: Optional[dict]) -> dict[str, Any]:
    """Compare on-disk file sizes against the manifest's recorded sizes.

    A stat() per manifest file, nothing else -- cheap even for a 135 GB
    specimen, because the manifest lists dozens of files, not bytes. No
    manifest means no verification is possible, and that is reported as
    unknown (None), never guessed as either complete or incomplete.
    """
    if manifest is None:
        return {"verified": None, "reason": "no manifest to verify against", "mismatches": []}
    sizes = manifest.get("sizes") or {}
    mismatches = []
    for fname, expected in sizes.items():
        try:
            actual = (d / fname).stat().st_size
        except OSError:
            mismatches.append({"file": fname, "expected": expected, "actual": None})
            continue
        if actual != expected:
            mismatches.append({"file": fname, "expected": expected, "actual": actual})
    if mismatches:
        return {"verified": False, "reason": f"{len(mismatches)} file(s) differ from the manifest", "mismatches": mismatches}
    return {"verified": True, "reason": "every manifest file matches its recorded size", "mismatches": []}


def _specimen(name: str, specimens_dir: Path, manifest_dir: Path) -> dict[str, Any]:
    d = specimens_dir / name
    repo, revision = _split_id(name)
    manifest = _manifest_for(name, manifest_dir)
    verify = _verify_complete(d, manifest)
    if manifest is not None and manifest.get("expected") is not None:
        size_bytes = manifest["expected"]
    else:
        size_bytes = _dir_bytes(d)
    return {
        "id": name,
        "repo": repo,
        "revision": revision,
        "path": str(d),
        "size_bytes": size_bytes,
        "architecture": _architecture(d / "config.json"),
        "manifest_present": manifest is not None,
        "manifest_path": str(manifest_dir / f"{name}.json") if manifest is not None else None,
        "verified_complete": verify["verified"],
        "verified_complete_reason": verify["reason"],
        "verified_complete_mismatches": verify["mismatches"],
    }


def registry() -> dict[str, Any]:
    """Every specimen under the ModelLake specimens/ directory, right now.

    Recomputed from disk on every call -- no cache, so a specimen that
    appears between two calls is simply in the next one.
    """
    specimens_dir = _lake_root() / "specimens"
    manifest_dir = _manifest_dir()
    if not specimens_dir.is_dir():
        return {
            "schema": SCHEMA,
            "lake": str(_lake_root()),
            "specimens_dir": str(specimens_dir),
            "mounted": False,
            "n_specimens": None,
            "specimens": [],
            "reason": f"{specimens_dir} is not present -- the ModelLake volume may not be "
                      "mounted; this is not evidence that zero specimens exist",
        }
    names = sorted(n for n in os.listdir(specimens_dir) if (specimens_dir / n).is_dir())
    rows = [_specimen(n, specimens_dir, manifest_dir) for n in names]
    return {
        "schema": SCHEMA,
        "lake": str(_lake_root()),
        "specimens_dir": str(specimens_dir),
        "mounted": True,
        "n_specimens": len(rows),
        "specimens": rows,
        "sealed_does_not_mean_resident": (
            "these are sealed sources on disk, not a load recommendation -- "
            "whether an experiment needs full residency, organ extraction, a "
            "partial/native artifact, or deferral is a judgement this "
            "registry does not make"
        ),
    }


def get(name: str) -> Optional[dict[str, Any]]:
    """One specimen by id, or None if it is not on disk right now."""
    specimens_dir = _lake_root() / "specimens"
    if not (specimens_dir / name).is_dir():
        return None
    return _specimen(name, specimens_dir, _manifest_dir())
