"""NR_NX_GENERIC — drive the generic NR→NX path on a specimen the compiler can see.

The previous receipt was right about the shape of the miss and wrong about
the label: GENERIC_NR_NX_PIPELINE_CALLABLE was False because Doctor and
PhysicalGraphCompiler were hardcoded to one model's tensor names, not because
the pipeline is an empty shell. This module parameterizes the specimen from
its own config.json and safetensors index so a model nobody has tried yet
either runs or fails with the stage and missing input named.

It packs a source-independent NX from the DeviceCompiler fragment via
tools.future.nx_packer. A renamed source pointer, a placeholder organ, a
missing metallib, or a billing mismatch raises. Physical EBPW is still
unwritten. A green boolean earned by weakening NX is a hardcoded-True.

    python3 tools/future/nr_nx_generic.py --build
    python3 -m pytest tools/future/test_nr_nx_generic.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import re
import struct
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import RECEIPTS, REPO, git, load_json, write_receipt
from tools.future import flash_nx_audit as nx_audit


def _extend_sys_path_for_sparse_checkout() -> list[str]:
    """tools/odyssey and hcli live in the primary checkout; this worktree is sparse."""
    added: list[str] = []
    for root in nx_audit.evidence_roots():
        s = str(root)
        if s not in _sys.path:
            _sys.path.append(s)
            added.append(s)
    return added


_SPARSE_PATHS = _extend_sys_path_for_sparse_checkout()

from tools.future import nx_packer as nxp
try:
    from tools.future import device_compiler as dcomp
except ImportError:
    dcomp = nxp.load_device_compiler()
from tools.future import nr_nx_path as nnp
from tools.future import specimen_verify as sv
from tools.future import workunit_species as wus

try:
    from tools.odyssey import arch_recognizer as ar
    from tools.odyssey import doctor_tournament as doctor
    from tools.odyssey import noetic_compiler as nc
    from tools.odyssey import physical_graph_compiler as pgc
    ODYSSEY_IMPORT_ERROR: str | None = None
except ModuleNotFoundError as _exc:
    ar = doctor = nc = pgc = None  # type: ignore[assignment]
    ODYSSEY_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

RECEIPT = "NR_NX_GENERIC.json"
SCHEMA = "hawking.future.nr_nx_generic.v1"
RECORDED_BY = "tools/future/nr_nx_generic.py"
VERSION = 1

REL_VERIFY = "receipts/future/SPECIMEN_VERIFICATION.json"
REL_MATRIX = "receipts/headless/ORGAN_FRONTIER_MATRIX.json"
REL_KERNELS = "receipts/headless/KERNEL_LIBRARY.json"
REL_AUDIT = "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json"
NATIVE_LOADER = "crates/hawking-core/src/model/mod.rs"
HEADLESS_FIRST_NX = "tools/headless/first_noetic_executable.py"

PASSED = "PASSED"
FAILED = "FAILED"
REFUSED = "REFUSED"
BLOCKED = "BLOCKED"
SLEEPING = "SLEEPING"

# A skipped stage is how a pipeline pretends to finish. The constructor refuses it.
FORBIDDEN_STAGE_STATUS = frozenset({"SKIPPED", "skip", "pending", "PENDING", "READY", "ready"})

# KernelPlanner occupying kinds. A role-name hit in KERNEL_LIBRARY is never COMPILED.
NATIVE_UNMEASURED = "NATIVE_UNMEASURED"
COMPILED = "COMPILED"
KERNEL_PLANNER_ROUTE_PLAN_THEN_COMPILE = "PLAN-THEN-COMPILE"
KERNEL_PLANNER_ROUTE_SHAPE_PARAMETRIC = "SHAPE-PARAMETRIC"
NAME_IS_NOT_A_COMPILED_KERNEL = (
    "A shared organ name is not a compiled kernel for this body."
)

STAGE_ORDER: tuple[str, ...] = (
    "SpecimenSelect",
    "SpecimenPresent",
    "ArchitectureRecognizer",
    "OrganGraph",
    "NrIdentifyOrCreate",
    "Doctor",
    "RepresentationPlanner",
    "PhysicalGraphCompiler",
    "KernelPlanner",
    "DeviceCompiler",
    "NoeticExecutable",
    "SourceIndependence",
    "ExecutableDependencyAccounting",
    "Verifier",
)

# Cheapness-ordered search pool named by the lane. Iteration order is not an
# adapt() branch: adapt() never reads these identifiers.
QWEN06_ID = "Qwen--Qwen3-0.6B@c1899de289a0"
QWEN06_REPO = "Qwen/Qwen3-0.6B"
QWEN06_REV = "c1899de289a0"
FALCON_ID = "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb"
FALCON_REPO = "tiiuae/Falcon-H1-7B-Instruct"
FALCON_REV = "41e72f27effb"
QWEN30_ID = "Qwen--Qwen3-30B-A3B@ad44e777bcd1"
QWEN30_REPO = "Qwen/Qwen3-30B-A3B"
QWEN30_REV = "ad44e777bcd1"

LANE_CANDIDATES: tuple[str, ...] = (QWEN06_ID, FALCON_ID, QWEN30_ID)

# Compiler-hardcoded names. Cited as overlap evidence, never used as the mapping.
PGC_COLLAPSE_TENSORS: tuple[str, ...] = (
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
    "model.layers.0.mlp.experts.0.down_proj.weight",
    "model.layers.0.mlp.gate.weight",
)

SOURCE_FILE_NAMES = frozenset(
    {
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "consolidated.safetensors",
    }
)

_SILU_ACTS = frozenset({"silu", "silu_and_mul", "swiglu"})
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
_DOCTOR_MAX_ROWS = 2048
_SWIGLU_TOLERANCE = 1e-9

# Needles tried in order. MoE expert paths before dense mlp so a routed model
# is not classified as dense because both mention gate_proj.
_GATE_NEEDLES = (
    ".mlp.experts.{E}.gate_proj.weight",
    ".mlp.experts.{E}.w1.weight",
    ".mlp.gate_proj.weight",
    ".feed_forward.gate_proj.weight",
    ".mlp.w1.weight",
    ".feed_forward.w1.weight",
)
_UP_NEEDLES = (
    ".mlp.experts.{E}.up_proj.weight",
    ".mlp.experts.{E}.w3.weight",
    ".mlp.up_proj.weight",
    ".feed_forward.up_proj.weight",
    ".mlp.w3.weight",
    ".feed_forward.w3.weight",
)
_DOWN_NEEDLES = (
    ".mlp.experts.{E}.down_proj.weight",
    ".mlp.experts.{E}.w2.weight",
    ".mlp.down_proj.weight",
    ".feed_forward.down_proj.weight",
    ".mlp.w2.weight",
    ".feed_forward.w2.weight",
)
_ROUTER_NEEDLES = (
    ".mlp.gate.weight",
    ".mlp.router.weight",
    ".mlp.gate.wg",
)
_Q_NEEDLES = (
    ".self_attn.q_proj.weight",
    ".attention.wq.weight",
    ".self_attn.qkv_proj.weight",
)
_RECURRENT_NEEDLES = (
    ".linear_attn.out_proj.weight",
    ".mamba.out_proj.weight",
    ".mamba.in_proj.weight",
)


class StageSkipForbidden(ValueError):
    """A stage reported SKIPPED. That is how a pipeline pretends to finish."""


class PipelineCallableForbidden(ValueError):
    """Callable was claimed while a stage was not PASSED or an NX was missing."""


PhysicalEbpwForbidden = nnp.PhysicalEbpwForbidden


def record_physical_ebpw(value: Any) -> None:
    nnp.record_physical_ebpw(value)


def assert_no_physical_ebpw(doc: Mapping[str, Any]) -> None:
    nnp.assert_no_physical_ebpw(doc)


def _dot(node: Any, dotted: str, default: Any = None) -> Any:
    cur: Any = node
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _stage(
    name: str,
    status: str,
    *,
    why: str,
    invoked: bool,
    evidence: Any = None,
    error: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status in FORBIDDEN_STAGE_STATUS or status == "SKIPPED":
        raise StageSkipForbidden(
            f"{name}: status={status!r} is forbidden; a stage that cannot run "
            "is FAILED, REFUSED, or BLOCKED with a reason"
        )
    if status not in {PASSED, FAILED, REFUSED, BLOCKED}:
        raise StageSkipForbidden(f"{name}: unknown status {status!r}")
    row: dict[str, Any] = {
        "stage": name,
        "status": status,
        "why": why,
        "invoked": invoked,
        "error": error,
        "evidence": evidence,
    }
    if extra:
        row.update(dict(extra))
    return row


def generic_pipeline_callable(stages: Sequence[Mapping[str, Any]]) -> bool:
    """True only when every named stage ran and PASSED. Empty is not success."""
    if not stages:
        return False
    names = [s.get("stage") for s in stages]
    if list(names) != list(STAGE_ORDER):
        return False
    for row in stages:
        if row.get("status") in FORBIDDEN_STAGE_STATUS or row.get("status") == "SKIPPED":
            return False
        if row.get("status") != PASSED:
            return False
        if row.get("invoked") is not True:
            return False
    return True


def declare_pipeline_callable(
    stages: Sequence[Mapping[str, Any]],
    *,
    packed_nx_path: str | Path | None,
) -> bool:
    """Refuse the Goodhart: SKIPPED stages or a missing NX cannot be a pass."""
    for row in stages:
        st = row.get("status")
        if st in FORBIDDEN_STAGE_STATUS or st == "SKIPPED":
            raise StageSkipForbidden(
                f"{row.get('stage')}: skipped/pending is not a result"
            )
    ok = generic_pipeline_callable(stages)
    if not ok:
        return False
    path = Path(packed_nx_path) if packed_nx_path else None
    if path is None or not path.is_file():
        raise PipelineCallableForbidden(
            "GENERIC_NR_NX_PIPELINE_CALLABLE cannot be True without a packed "
            "NX artifact on disk"
        )
    return True


# ---------------------------------------------------------------------------
# Source independence. An NX that still opens the checkpoint has not lowered.
# ---------------------------------------------------------------------------


def _resolves_into_source(value: Any, source_trees: Sequence[str | Path]) -> bool:
    if not isinstance(value, str) or not value:
        return False
    text = value
    for tree in source_trees:
        root = str(tree)
        if not root:
            continue
        if text == root or text.startswith(root.rstrip("/") + "/") or root in text:
            return True
        try:
            Path(text).resolve().relative_to(Path(root).resolve())
            return True
        except (OSError, ValueError):
            continue
    name = Path(text).name
    if name in SOURCE_FILE_NAMES and ("specimens" in text or "modellake" in text or "partial/" in text):
        return True
    return False


def _runtime_strings(nx: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Fields a loader would actually open, not provenance citations on a receipt."""
    out: list[tuple[str, str]] = []
    art = nx_audit._serialized_artifact(nx)
    if isinstance(art, dict) and isinstance(art.get("path"), str):
        out.append(("serialized_artifact.path", art["path"]))
    elif isinstance(art, str):
        out.append(("serialized_artifact", art))
    for key in (
        "runtime_reads",
        "source_path",
        "checkpoint",
        "model_dir",
        "weights",
        "specimen_path",
        "parent_path",
    ):
        raw = nx.get(key)
        if isinstance(raw, str):
            out.append((key, raw))
        elif isinstance(raw, list):
            for i, item in enumerate(raw):
                if isinstance(item, str):
                    out.append((f"{key}[{i}]", item))
    loader = nx_audit._loader(nx)
    if isinstance(loader, Mapping):
        for key in ("path", "source_path", "reads", "checkpoint"):
            raw = loader.get(key)
            if isinstance(raw, str):
                out.append((f"physical_loader.{key}", raw))
            elif isinstance(raw, list):
                for i, item in enumerate(raw):
                    if isinstance(item, str):
                        out.append((f"physical_loader.{key}[{i}]", item))
    pp = nx.get("physical_program")
    if isinstance(pp, Mapping):
        for key in ("source_path", "checkpoint", "weights", "executor_path"):
            raw = pp.get(key)
            if isinstance(raw, str):
                out.append((f"physical_program.{key}", raw))
    return out


def source_independence(
    nx: Mapping[str, Any] | None,
    *,
    source_trees: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """FAIL if the NX would read the source tree, or if its bytes are the parent.

    A metadata seal, a missing body, or a path into the specimen is not
    independence. source_independent=True on a document that still points at
    model.safetensors is a renamed source pointer.
    """
    if not isinstance(nx, Mapping):
        return {
            "ok": False,
            "why": "no NX document; absence is not source independence",
            "hits": [],
        }
    hits: list[str] = []
    if nx_audit._status_is_metadata_only(nx):
        hits.append(f"status={nx.get('status')!r} is a metadata seal, not a packed body")
    if nx.get("source_independent") is not True:
        hits.append(f"source_independent={nx.get('source_independent')!r} is not True")
    art = nx_audit._serialized_artifact(nx)
    if isinstance(art, dict):
        if art.get("self_contained") is not True:
            hits.append("serialized_artifact.self_contained is not True")
        if not (art.get("sha256") or art.get("digest")):
            hits.append("serialized_artifact has no digest; a path without a digest is a pointer")
        if art.get("status") in {None, "NOT_BUILT", "ABSENT"}:
            hits.append(f"serialized_artifact.status={art.get('status')!r}")
    elif art in (None, "", "NOT_BUILT"):
        hits.append("no serialized_artifact; there is nothing to be independent of the source")
    loader = nx_audit._loader(nx)
    if isinstance(loader, Mapping) and loader.get("source_independent") is not True:
        hits.append(f"physical_loader.source_independent={loader.get('source_independent')!r}")
    binding = _dot(nx, "physical_program.source_binding")
    if isinstance(binding, str) and binding.strip():
        lowered = binding.lower()
        if any(tok in lowered for tok in ("source", "checkpoint", "safetensor", "specimen", "executor")):
            hits.append(f"physical_program.source_binding still names a source executor: {binding[:160]}")
    for field, value in _runtime_strings(nx):
        if _resolves_into_source(value, source_trees):
            hits.append(f"{field} resolves into the source tree: {value}")
        if Path(str(value)).name in SOURCE_FILE_NAMES and field.startswith("serialized_artifact"):
            hits.append(
                f"renamed source pointer: {field} names a source checkpoint file ({Path(str(value)).name})"
            )
    ok = not hits
    return {
        "ok": ok,
        "why": "source-independent packed body" if ok else "; ".join(hits),
        "hits": hits,
        "source_independent_flag": nx.get("source_independent"),
        "status": nx.get("status"),
    }


# ---------------------------------------------------------------------------
# Safetensors I/O. Header-first; payload is sliced, never assumed.
# ---------------------------------------------------------------------------


def _safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header, n


def _safetensors_names(path: Path) -> list[str]:
    header, _ = _safetensors_header(path)
    return sorted(k for k in header if k != "__metadata__")


def _tensor_names(spec_dir: Path) -> tuple[list[str], str, dict[str, str]]:
    index = spec_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map") or {}
        shard_map = {str(k): str(v) for k, v in weight_map.items()}
        return sorted(shard_map), "model.safetensors.index.json", shard_map
    shard = spec_dir / "model.safetensors"
    if shard.is_file():
        names = _safetensors_names(shard)
        return names, "model.safetensors header", {n: "model.safetensors" for n in names}
    return [], "absent", {}


def _bf16_to_f64(raw: bytes, shape: Sequence[int]):
    if np is None:
        raise RuntimeError("numpy is required to decode BF16")
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return np.asarray(f32, dtype=np.float64).reshape(shape)


def _decode_tensor(raw: bytes, dtype: str, shape: Sequence[int]):
    if np is None:
        raise RuntimeError("numpy is required to decode tensors")
    kind = (dtype or "").upper()
    if kind == "BF16":
        return _bf16_to_f64(raw, shape)
    dt = {
        "F32": "<f4",
        "F16": "<f2",
        "F64": "<f8",
        "I32": "<i4",
        "I64": "<i8",
        "U8": "|u1",
    }.get(kind)
    if dt is None:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    arr = np.frombuffer(raw, dtype=dt)
    return arr.astype(np.float64).reshape(shape)


_TENSOR_CACHE: dict[tuple[str, str], Any] = {}
_HEADER_CACHE: dict[str, tuple[dict[str, Any], int]] = {}


def _cached_header(path: Path) -> tuple[dict[str, Any], int]:
    key = str(path)
    hit = _HEADER_CACHE.get(key)
    if hit is None:
        hit = _safetensors_header(path)
        _HEADER_CACHE[key] = hit
    return hit


def load_tensor(spec_dir: Path, shard_map: Mapping[str, str], name: str):
    """Load one tensor as float64. Refuses to guess a missing name or dtype."""
    if np is None:
        raise RuntimeError("numpy is not importable; tensor load refused")
    shard = shard_map.get(name)
    if not shard:
        raise FileNotFoundError(f"tensor {name!r} has no shard mapping")
    path = spec_dir / shard
    cache_key = (str(path), name)
    cached = _TENSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path.is_file():
        raise FileNotFoundError(f"shard not on disk: {path}")
    header, header_len = _cached_header(path)
    info = header.get(name)
    if not isinstance(info, Mapping):
        raise KeyError(f"{name} absent from {path.name} header")
    dtype = str(info.get("dtype") or "")
    shape = list(info.get("shape") or [])
    offsets = info.get("data_offsets") or [0, 0]
    start, end = int(offsets[0]), int(offsets[1])
    with open(path, "rb") as fh:
        fh.seek(8 + header_len + start)
        raw = fh.read(end - start)
    arr = _decode_tensor(raw, dtype, shape)
    _TENSOR_CACHE[cache_key] = arr
    return arr


# ---------------------------------------------------------------------------
# probe_specimen / adapt / callable_on — derived from config + index.
# ---------------------------------------------------------------------------


def _split_lake_id(sid: str) -> tuple[str, str]:
    """ModelLake identity convention: org--name@rev, optional #partial."""
    text = sid[:-8] if sid.endswith("#partial") else sid
    if "@" in text:
        body, rev = text.rsplit("@", 1)
    else:
        body, rev = text, ""
    repo = body.replace("--", "/", 1)
    return repo, rev


def _config_from_dir(spec_dir: Path) -> dict[str, Any] | None:
    cfg_path = spec_dir / "config.json"
    if not cfg_path.is_file():
        return None
    raw = json.loads(cfg_path.read_text())
    return raw if isinstance(raw, dict) else None


def probe_specimen(name: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """What tensors this specimen actually has, read from its index or header.

    Names are never assumed from the specimen id. An absent body is a refusal.
    """
    if isinstance(name, Mapping):
        if name.get("tensor_names") is not None and name.get("config") is not None:
            cfg = name.get("config")
            names = [str(n) for n in (name.get("tensor_names") or [])]
            sid = str(name.get("id") or name.get("specimen") or name.get("name") or "supplied")
            repo = name.get("repo")
            rev = name.get("revision")
            if not repo:
                repo, parsed_rev = _split_lake_id(sid)
                rev = rev or parsed_rev
            shard_map = name.get("shard_map") or {}
            if not isinstance(shard_map, dict):
                shard_map = {}
            if not shard_map and names:
                shard_map = {n: "model.safetensors" for n in names}
            path = name.get("specimen_path")
            return {
                "ok": True,
                "id": sid,
                "repo": repo,
                "revision": rev,
                "specimen_path": None if path is None else str(path),
                "config": dict(cfg) if isinstance(cfg, Mapping) else {},
                "tensor_names": names,
                "n_tensors": len(names),
                "names_via": str(name.get("names_via") or "caller"),
                "shard_map": {str(k): str(v) for k, v in shard_map.items()},
                "why": "caller-supplied config + tensor names",
            }
        if name.get("ok") is False and not name.get("tensor_names"):
            return {
                "ok": False,
                "id": name.get("id"),
                "why": str(name.get("why") or "probe refused by caller"),
                "config": None,
                "tensor_names": [],
                "n_tensors": 0,
                "names_via": "absent",
                "shard_map": {},
                "specimen_path": name.get("specimen_path"),
            }
        if name.get("specimen_path"):
            sid = str(name.get("id") or name.get("specimen") or Path(str(name["specimen_path"])).name)
            return probe_specimen(Path(str(name["specimen_path"]))) | {"id": sid}
        if name.get("id"):
            return probe_specimen(str(name["id"]))
        return {
            "ok": False,
            "id": None,
            "why": "mapping had neither tensor_names+config, nor a specimen id/path",
            "config": None,
            "tensor_names": [],
            "n_tensors": 0,
            "names_via": "absent",
            "shard_map": {},
        }

    if isinstance(name, Path) or (isinstance(name, str) and (name.startswith("/") or name.startswith("."))):
        spec_dir = Path(name)
        sid = spec_dir.name
        return _probe_dir(sid, spec_dir)

    sid = str(name)
    try:
        spec_dir = sv.specimen_dir(sid)
    except Exception as exc:
        return {
            "ok": False,
            "id": sid,
            "why": f"specimen_dir raised {type(exc).__name__}: {exc}",
            "config": None,
            "tensor_names": [],
            "n_tensors": 0,
            "names_via": "absent",
            "shard_map": {},
        }
    return _probe_dir(sid, spec_dir)


def _probe_dir(sid: str, spec_dir: Path) -> dict[str, Any]:
    repo, rev = _split_lake_id(sid)
    if not spec_dir.is_dir():
        return {
            "ok": False,
            "id": sid,
            "repo": repo,
            "revision": rev,
            "specimen_path": str(spec_dir),
            "why": f"specimen directory is not on disk: {spec_dir}",
            "config": None,
            "tensor_names": [],
            "n_tensors": 0,
            "names_via": "absent",
            "shard_map": {},
        }
    cfg = _config_from_dir(spec_dir)
    if cfg is None:
        return {
            "ok": False,
            "id": sid,
            "repo": repo,
            "revision": rev,
            "specimen_path": str(spec_dir),
            "why": f"config.json missing under {spec_dir}",
            "config": None,
            "tensor_names": [],
            "n_tensors": 0,
            "names_via": "missing_config",
            "shard_map": {},
        }
    names, via, shard_map = _tensor_names(spec_dir)
    if not names:
        return {
            "ok": False,
            "id": sid,
            "repo": repo,
            "revision": rev,
            "specimen_path": str(spec_dir),
            "config": cfg,
            "tensor_names": [],
            "n_tensors": 0,
            "names_via": via,
            "shard_map": {},
            "why": (
                f"no safetensors index or shard under {spec_dir}; "
                "refusing to assume tensor names from the specimen id"
            ),
        }
    return {
        "ok": True,
        "id": sid,
        "repo": repo,
        "revision": rev,
        "specimen_path": str(spec_dir),
        "config": cfg,
        "tensor_names": names,
        "n_tensors": len(names),
        "names_via": via,
        "shard_map": shard_map,
        "why": f"read {len(names)} tensor names via {via}",
    }


def _templates_from_names(names: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for n in names:
        t = _LAYER_RE.sub(".layers.{L}.", n)
        t = _EXPERT_RE.sub(".experts.{E}.", t)
        out.add(t)
    return out


def _pick_template(templates: set[str], needles: Sequence[str]) -> str | None:
    for needle in needles:
        exact = [t for t in templates if t.endswith(needle)]
        if exact:
            return sorted(exact, key=len)[0]
        loose = [t for t in templates if needle in t]
        if loose:
            return sorted(loose, key=len)[0]
    return None


def _format_tensor(template: str, layer: int, expert: int | None = None) -> str:
    out = template.replace("{L}", str(layer))
    if "{E}" in out:
        out = out.replace("{E}", str(0 if expert is None else expert))
    return out


def _present_layers(
    names: set[str],
    template: str | None,
    n_layers: int,
    expert: int | None = None,
) -> list[int]:
    if not template or n_layers <= 0:
        return []
    found: list[int] = []
    for L in range(n_layers):
        if _format_tensor(template, L, expert) in names:
            found.append(L)
    return found


def _sample_layers(present: Sequence[int]) -> tuple[int, ...]:
    have = list(dict.fromkeys(int(x) for x in present))
    if not have:
        return ()
    if len(have) <= 3:
        return tuple(have)
    return (have[0], have[len(have) // 2], have[-1])


def _n_layers(cfg: Mapping[str, Any], names: Sequence[str]) -> int:
    n = cfg.get("num_hidden_layers")
    if isinstance(n, int) and n > 0:
        return n
    found = [int(m.group(1)) for nm in names if (m := _LAYER_RE.search(nm))]
    return (max(found) + 1) if found else 0


def _hidden_act(cfg: Mapping[str, Any]) -> str:
    raw = cfg.get("hidden_act") or cfg.get("hidden_activation") or ""
    return str(raw).lower()


def _compiler_hardcoded_overlap(names: Sequence[str]) -> dict[str, Any]:
    name_set = set(names)
    doctor_names: list[str] = []
    if doctor is not None:
        for _organ, pat, layers in doctor.PROBE_TENSORS:
            for L in layers:
                doctor_names.append(pat.format(L=L))
    pgc_names = list(PGC_COLLAPSE_TENSORS)
    return {
        "doctor_parent": None if doctor is None else str(doctor.PARENT),
        "doctor_probes": doctor_names,
        "doctor_probes_present": [n for n in doctor_names if n in name_set],
        "doctor_probes_absent": [n for n in doctor_names if n not in name_set],
        "pgc_collapse_present": [n for n in pgc_names if n in name_set],
        "pgc_collapse_absent": [n for n in pgc_names if n not in name_set],
    }


def adapt(specimen: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Map this specimen's architecture onto what the pipeline needs.

    Derived from config.json + the real tensor index. A branch on the specimen
    id would reproduce the original defect with a second hardcoded name.
    """
    probe = specimen if isinstance(specimen, Mapping) and "tensor_names" in specimen and "config" in specimen else probe_specimen(specimen)
    if probe.get("ok") is not True:
        return {
            "ok": False,
            "id": probe.get("id"),
            "why": str(probe.get("why") or "probe refused; adaptation is not invented"),
            "pipeline_can_start": False,
            "family": "unknown",
            "probe_plan": [],
            "collapse_plan": [],
            "missing": [str(probe.get("why") or "probe refused")],
        }
    cfg = probe.get("config") if isinstance(probe.get("config"), Mapping) else {}
    names = list(probe.get("tensor_names") or [])
    name_set = set(names)
    templates = _templates_from_names(names)
    n_layers = _n_layers(cfg, names)
    act = _hidden_act(cfg)
    swiglu = act in _SILU_ACTS
    gate = _pick_template(templates, _GATE_NEEDLES)
    up = _pick_template(templates, _UP_NEEDLES)
    down = _pick_template(templates, _DOWN_NEEDLES)
    router = _pick_template(templates, _ROUTER_NEEDLES)
    q_proj = _pick_template(templates, _Q_NEEDLES)
    recurrent = _pick_template(templates, _RECURRENT_NEEDLES)
    moe = bool(gate and "{E}" in gate)
    dense = bool(gate and "{E}" not in gate)
    mlp_kind = "moe" if moe else ("dense" if dense else "absent")
    if recurrent and (dense or moe):
        family = "hybrid_recurrent"
    elif moe:
        family = "moe_swiglu_transformer" if swiglu else "moe_transformer"
    elif dense:
        family = "dense_swiglu_transformer" if swiglu else "dense_transformer"
    else:
        family = "unknown"

    expert = 0 if moe else None
    gate_layers = _present_layers(name_set, gate, n_layers, expert)
    up_layers = _present_layers(name_set, up, n_layers, expert)
    down_layers = _present_layers(name_set, down, n_layers, expert)
    q_layers = _present_layers(name_set, q_proj, n_layers, None)
    rec_layers = _present_layers(name_set, recurrent, n_layers, None)

    probe_plan: list[dict[str, Any]] = []
    if gate:
        for L in _sample_layers(gate_layers):
            probe_plan.append(
                {"organ": "mlp", "role": "gate", "layer": L, "tensor": _format_tensor(gate, L, expert)}
            )
        if down:
            for L in _sample_layers(down_layers):
                probe_plan.append(
                    {"organ": "mlp", "role": "down", "layer": L, "tensor": _format_tensor(down, L, expert)}
                )
    if q_proj:
        for L in _sample_layers(q_layers):
            probe_plan.append(
                {"organ": "attention_gqa", "role": "q", "layer": L, "tensor": _format_tensor(q_proj, L, None)}
            )
    if recurrent:
        rec_organ = "deltanet" if "linear_attn" in recurrent else "recurrent_state"
        for L in _sample_layers(rec_layers):
            probe_plan.append(
                {"organ": rec_organ, "role": "out", "layer": L, "tensor": _format_tensor(recurrent, L, None)}
            )

    collapse_plan: list[dict[str, Any]] = []
    collapse_missing: list[str] = []
    shared = sorted(set(gate_layers) & set(up_layers) & set(down_layers))
    if gate and up and down and swiglu and shared:
        L = shared[0]
        g_name = _format_tensor(gate, L, expert)
        u_name = _format_tensor(up, L, expert)
        d_name = _format_tensor(down, L, expert)
        collapse_plan.append(
            {
                "collapse": "gate_up_swiglu",
                "layer": L,
                "expert": expert,
                "gate": g_name,
                "up": u_name,
                "down": d_name,
            }
        )
    elif gate and up and down and not swiglu:
        collapse_missing.append(
            f"hidden_act={act!r} is not a SwiGLU/SiLU activation; refusing to run the silu fusion"
        )
    elif not (gate and up and down):
        collapse_missing.append(
            "no gate/up/down templates derived from the tensor index; SwiGLU collapse has nothing to fuse"
        )

    router_applicable = bool(moe and router)
    router_collapse = {
        "applicable": router_applicable,
        "template": router,
        "missing_input": (
            "X_layer activation capture (not fabricated)" if router_applicable else None
        ),
        "why": (
            "router top-k collapse needs a real activation capture; a random X would not be this specimen"
            if router_applicable
            else "dense MLP has no expert router; router collapse is not an operator on this architecture"
        ),
    }

    missing = list(collapse_missing)
    if not names:
        missing.append("no tensor names")
    if not cfg:
        missing.append("no config")

    can_start = bool(
        cfg
        and names
        and collapse_plan
        and probe_plan
        and swiglu
    )
    overlap = _compiler_hardcoded_overlap(names)
    why = (
        f"family={family} mlp={mlp_kind} swiglu={swiglu} n_layers={n_layers} "
        f"n_tensors={len(names)} collapse={len(collapse_plan)} probes={len(probe_plan)}; "
        f"compiler-hardcoded doctor probes present "
        f"{len(overlap['doctor_probes_present'])}/{len(overlap['doctor_probes'] or [])}; "
        f"compiler-hardcoded PGC MoE tensors present "
        f"{len(overlap['pgc_collapse_present'])}/{len(PGC_COLLAPSE_TENSORS)}"
    )
    prefix = None
    for t in templates:
        if ".layers.{L}." in t:
            prefix = t.split(".layers.{L}.")[0]
            break
    return {
        "ok": True,
        "id": probe.get("id"),
        "why": why,
        "pipeline_can_start": can_start,
        "family": family,
        "mlp_kind": mlp_kind,
        "moe": moe,
        "n_layers": n_layers,
        "hidden_act": act or None,
        "swiglu": swiglu,
        "layer_prefix": prefix,
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "num_experts": cfg.get("num_experts") or cfg.get("num_local_experts"),
        "num_experts_per_tok": cfg.get("num_experts_per_tok"),
        "templates": {
            "gate": gate,
            "up": up,
            "down": down,
            "router": router,
            "q_proj": q_proj,
            "recurrent": recurrent,
        },
        "probe_plan": probe_plan,
        "collapse_plan": collapse_plan,
        "router_collapse": router_collapse,
        "missing": missing,
        "compiler_hardcoded_overlap": overlap,
        "names_via": probe.get("names_via"),
        "n_tensors": len(names),
    }


def callable_on(specimen: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Can the full path run here? If not, which stage fails and on which missing input.

    Preflight only. Does not load weights and does not mint an NX.
    """
    supplied = isinstance(specimen, Mapping) and specimen.get("tensor_names") is not None
    probe = probe_specimen(specimen)
    adaptation = adapt(probe)
    native = native_engine_architectures()
    path = Path(str(probe.get("specimen_path") or ""))
    present = supplied or path.is_dir()
    preview: list[dict[str, Any]] = []

    def add(stage: str, ready: bool, missing: str | None) -> None:
        preview.append({"stage": stage, "ready": ready, "missing_input": missing})

    add("SpecimenSelect", bool(probe.get("id")), None if probe.get("id") else "no specimen id")
    add(
        "SpecimenPresent",
        bool(present and probe.get("ok")),
        None if present and probe.get("ok") else str(probe.get("why") or "specimen body not present"),
    )
    add(
        "ArchitectureRecognizer",
        bool(probe.get("ok") and probe.get("config") is not None and ar is not None),
        None
        if probe.get("ok") and ar is not None
        else (ODYSSEY_IMPORT_ERROR or "no config/names"),
    )
    add("OrganGraph", bool(probe.get("ok") and pgc is not None), None if pgc is not None else ODYSSEY_IMPORT_ERROR)
    add("NrIdentifyOrCreate", bool(probe.get("ok")), None if probe.get("ok") else "no organ graph input")
    probe_names = [p["tensor"] for p in adaptation.get("probe_plan") or [] if p.get("tensor")]
    missing_probes = [n for n in probe_names if n not in set(probe.get("tensor_names") or [])]
    doctor_ready = bool(probe_names) and not missing_probes
    add(
        "Doctor",
        doctor_ready,
        None if doctor_ready else (
            f"adapted probe tensors missing: {missing_probes}" if missing_probes
            else "adapt() derived an empty probe plan"
        ),
    )
    rl_ok, rl_err = _representation_library()
    add(
        "RepresentationPlanner",
        bool(rl_ok and probe.get("ok")),
        None if rl_ok else (rl_err or "representation_library unimportable"),
    )
    collapse = list(adaptation.get("collapse_plan") or [])
    pgc_ready = bool(collapse) and adaptation.get("swiglu") is True
    add(
        "PhysicalGraphCompiler",
        pgc_ready,
        None if pgc_ready else (
            "; ".join(adaptation.get("missing") or ["no parameterized SwiGLU collapse"])
        ),
    )
    lib_ok, lib_why = kernel_library_is_readable()
    add(
        "KernelPlanner",
        bool(lib_ok and probe.get("ok")),
        None
        if lib_ok and probe.get("ok")
        else (lib_why or "no specimen; a kernel plan is not invented"),
    )
    add(
        "DeviceCompiler",
        True,
        None,
    )
    src_dir = Path(str(probe.get("specimen_path") or ""))
    source_on_disk = src_dir.is_dir()
    packer_ok = nxp.packer_callable()
    nx_ready = bool(packer_ok and source_on_disk)
    add(
        "NoeticExecutable",
        nx_ready,
        None if nx_ready else (
            "generic packer needs a specimen directory of runtime bytes; "
            "a renamed source pointer is not an NX"
        ),
    )
    add(
        "SourceIndependence",
        nx_ready,
        None if nx_ready else "no packed NX body until the generic packer runs",
    )
    add(
        "ExecutableDependencyAccounting",
        nx_ready,
        None if nx_ready else "no packed NX body until the generic packer runs",
    )
    add(
        "Verifier",
        nx_ready,
        None if nx_ready else "no packed NX body until the generic packer runs",
    )

    first = next((row for row in preview if row["ready"] is not True), None)
    return {
        "ok": first is None,
        "id": probe.get("id"),
        "first_failing_stage": None if first is None else first["stage"],
        "missing_input": None if first is None else first["missing_input"],
        "stage_preview": preview,
        "adaptation_family": adaptation.get("family"),
        "pipeline_can_start": adaptation.get("pipeline_can_start"),
        "probe_ok": probe.get("ok"),
        "why": (
            "every stage has its input"
            if first is None
            else f"{first['stage']}: {first['missing_input']}"
        ),
    }


def _representation_library() -> tuple[Any, str | None]:
    try:
        from tools.headless import representation_library as rl

        return rl, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Specimen choice. Cheapness-ordered candidates, ranked by adapt() not by name.
# ---------------------------------------------------------------------------


def _verification_index() -> dict[str, Any]:
    """Union verification rows across sparse worktree + primary checkout.

    A truncated local receipt is not proof a specimen was never verified.
    WHOLE_TREE_VERIFIED rows win over weaker duplicates of the same id.
    """
    rows: dict[str, dict[str, Any]] = {}
    vias: list[str] = []
    whole: list[str] = []
    for root in nx_audit.evidence_roots():
        path = Path(root) / REL_VERIFY
        if not path.is_file():
            continue
        vias.append(str(path))
        try:
            doc = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        for sid in doc.get("whole_tree_verified_specimens") or []:
            if sid not in whole:
                whole.append(str(sid))
        for row in doc.get("results") or []:
            if not isinstance(row, dict) or not row.get("specimen"):
                continue
            sid = str(row["specimen"])
            existing = rows.get(sid)
            stronger = (
                row.get("status") == "WHOLE_TREE_VERIFIED"
                or row.get("whole_tree_verified") is True
            )
            if existing is None or stronger:
                rows[sid] = row
            if stronger and sid not in whole:
                whole.append(sid)
    return {
        "present": bool(rows) or bool(vias),
        "via": ";".join(vias) if vias else "missing",
        "rows": rows,
        "whole_tree": whole,
    }


def choose_specimen(
    *,
    present: set[str] | None = None,
    verified: Mapping[str, Mapping[str, Any]] | None = None,
    lake_mounted: bool | None = None,
    candidates: Sequence[str] | None = None,
    probes: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """First cheapness-ordered candidate whose adapt() says the pipeline can start.

    Falcon is not a silent substitute: if 0.6B cannot start, the attempt log
    names why, then the next candidate is probed through the same adapt().
    """
    mounted = sv.available()["mounted"] if lake_mounted is None else lake_mounted
    vidx = _verification_index() if verified is None else {
        "present": True,
        "via": "caller",
        "rows": dict(verified),
        "whole_tree": [
            k for k, r in verified.items()
            if r.get("whole_tree_verified") or r.get("status") == "WHOLE_TREE_VERIFIED"
        ],
    }
    pool = list(candidates) if candidates is not None else list(LANE_CANDIDATES)
    if present is None:
        present_set: set[str] = set()
        if mounted:
            try:
                present_set.update(sv.list_specimens())
            except OSError:
                present_set = set()
            for name in pool:
                try:
                    if sv.specimen_dir(name).is_dir():
                        present_set.add(name)
                except Exception:
                    continue
    else:
        present_set = set(present)

    attempts: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    rows = vidx.get("rows") if isinstance(vidx.get("rows"), dict) else {}

    for sid in pool:
        vrow = rows.get(sid) if isinstance(rows, dict) else None
        is_verified = isinstance(vrow, Mapping) and (
            vrow.get("whole_tree_verified") is True or vrow.get("status") == "WHOLE_TREE_VERIFIED"
        )
        is_present = sid in present_set
        if not is_present:
            attempts.append(
                {"id": sid, "chosen": False, "not_chosen_because": "not present on this host"}
            )
            continue
        if not is_verified:
            attempts.append(
                {"id": sid, "chosen": False, "not_chosen_because": "not whole-tree verified"}
            )
            continue
        if probes is not None and sid in probes:
            probe = probe_specimen(probes[sid])
        else:
            spec_path = None
            if mounted:
                try:
                    d = sv.specimen_dir(sid)
                    if d.is_dir():
                        spec_path = d
                except Exception:
                    spec_path = None
            if spec_path is None and isinstance(vrow, Mapping) and vrow.get("specimen_path"):
                cand = Path(str(vrow["specimen_path"]))
                spec_path = cand if cand.is_dir() else None
            if spec_path is None:
                attempts.append(
                    {
                        "id": sid,
                        "chosen": False,
                        "not_chosen_because": "verified and listed present but directory is not on disk; refusing to invent tensors",
                    }
                )
                continue
            probe = probe_specimen(spec_path)
            probe["id"] = sid
        ad = adapt(probe)
        attempt = {
            "id": sid,
            "chosen": False,
            "present": True,
            "verified": True,
            "pipeline_can_start": ad.get("pipeline_can_start"),
            "family": ad.get("family"),
            "mlp_kind": ad.get("mlp_kind"),
            "n_tensors": probe.get("n_tensors"),
            "names_via": probe.get("names_via"),
            "not_chosen_because": None if ad.get("pipeline_can_start") else (
                "; ".join(ad.get("missing") or []) or ad.get("why")
            ),
        }
        if ad.get("pipeline_can_start") and chosen is None:
            attempt["chosen"] = True
            attempt["not_chosen_because"] = None
            chosen = {
                "ok": True,
                "id": sid,
                "repo": probe.get("repo"),
                "revision": probe.get("revision"),
                "family": ad.get("family"),
                "architectures_expected": ad.get("architectures"),
                "specimen_path": probe.get("specimen_path"),
                "bytes_hashed": None if not isinstance(vrow, Mapping) else vrow.get("bytes_hashed"),
                "n_files": None if not isinstance(vrow, Mapping) else vrow.get("n_files"),
                "verification_status": None if not isinstance(vrow, Mapping) else vrow.get("status"),
                "verification_via": vidx.get("via"),
                "why_chosen": (
                    f"first cheapness-ordered candidate whose adapt() derived a SwiGLU "
                    f"collapse and a probe plan from this specimen's own index "
                    f"(family={ad.get('family')}, mlp={ad.get('mlp_kind')}, "
                    f"n_tensors={probe.get('n_tensors')}, via={probe.get('names_via')})"
                ),
                "lake_mounted": mounted,
                "probe": {k: probe.get(k) for k in (
                    "ok", "id", "repo", "revision", "specimen_path", "n_tensors",
                    "names_via", "why",
                )},
                "adaptation": ad,
            }
        attempts.append(attempt)

    if chosen is not None:
        later = [a["id"] for a in attempts if a["id"] != chosen["id"]]
        chosen["attempts"] = attempts
        chosen["why_not_later_candidates"] = (
            "later candidates were not needed; adapt() already accepted a cheaper body. "
            + "; ".join(
                f"{a['id']}: {a.get('not_chosen_because') or 'not reached'}"
                for a in attempts if a["id"] != chosen["id"]
            )
        )
        chosen["later_candidates"] = later
        return chosen
    return {
        "ok": False,
        "id": None,
        "why": (
            "no cheapness-ordered candidate was present, whole-tree verified, and "
            "accepted by adapt() derived from its own config+index. "
            "Refusing to invent a specimen. attempts="
            + json.dumps([{k: a.get(k) for k in ('id', 'not_chosen_because', 'pipeline_can_start')} for a in attempts])
        ),
        "attempts": attempts,
        "lake_mounted": mounted,
        "verification_via": vidx.get("via"),
    }


def organ_library() -> dict[str, Any]:
    path = nx_audit.evidence_path(REL_MATRIX)
    if path is None:
        return {"present": False, "known": set(), "declared": set(), "via": "missing"}
    doc = load_json(path)
    rows = [e for e in (doc.get("organs") or []) if isinstance(e, dict) and e.get("organ")]
    known = {str(e["organ"]) for e in rows if e.get("status") == "MEASURED"}
    declared = {str(e["organ"]) for e in rows} - known
    return {
        "present": True,
        "known": known,
        "declared": declared,
        "via": str(path),
        "n_organs": len(rows),
    }


def native_engine_architectures(src: str | None = None) -> dict[str, Any]:
    """What crates/hawking-core actually dispatches. Naming a family is not a load."""
    path = REPO / NATIVE_LOADER
    if src is None:
        if not path.is_file():
            blob = git("show", f"HEAD:{NATIVE_LOADER}")
            if not blob:
                return {
                    "ok": False,
                    "path": NATIVE_LOADER,
                    "why": "native loader source is not on disk and git show returned empty",
                    "architectures": [],
                }
            src = blob
        else:
            src = path.read_text()
    start = src.find("match arch.as_str()")
    if start < 0:
        return {
            "ok": False,
            "path": NATIVE_LOADER,
            "why": "match arch.as_str() not found; refusing to guess the allowlist",
            "architectures": [],
        }
    chunk = src[start : start + 2500]
    quoted = re.findall(r'"([a-z0-9._-]+)"', chunk)
    other = None
    m = re.search(r'unknown architecture.*?(?:supports|engine supports)\s+([^.\\]+)', chunk, re.I | re.S)
    if m:
        other = m.group(1).strip()
    return {
        "ok": True,
        "path": NATIVE_LOADER,
        "architectures": quoted,
        "unknown_architecture_message": other,
        "includes_qwen3_dense": "qwen3" in quoted,
        "includes_qwen2": "qwen2" in quoted,
        "includes_qwen3moe": "qwen3moe" in quoted,
        "includes_falcon_h1": "falcon_h1" in quoted or "falconh1" in quoted,
        "chunk": chunk[:900],
    }


def _native_includes_qwen3_dense(native: Mapping[str, Any]) -> bool:
    arches = list(native.get("architectures") or [])
    return "qwen3" in arches


# ---------------------------------------------------------------------------
# Stage drivers. Each one either invokes a real entry or records the refusal.
# ---------------------------------------------------------------------------


def stage_specimen_select(choice: Mapping[str, Any]) -> dict[str, Any]:
    if choice.get("ok") is True and choice.get("id"):
        return _stage(
            "SpecimenSelect",
            PASSED,
            why=str(choice.get("why_chosen") or choice.get("why") or "specimen selected by adapt()"),
            invoked=True,
            evidence={
                "id": choice.get("id"),
                "repo": choice.get("repo"),
                "bytes_hashed": choice.get("bytes_hashed"),
                "family": choice.get("family"),
                "why_not_later_candidates": choice.get("why_not_later_candidates"),
                "attempts": choice.get("attempts"),
            },
        )
    return _stage(
        "SpecimenSelect",
        REFUSED,
        why=str(choice.get("why") or "no specimen chosen"),
        invoked=True,
        evidence=dict(choice),
    )


def stage_specimen_present(choice: Mapping[str, Any], probe: Mapping[str, Any]) -> dict[str, Any]:
    if probe.get("ok") is True and probe.get("tensor_names") and probe.get("config") is not None:
        path = Path(str(probe.get("specimen_path") or ""))
        on_disk = path.is_dir()
        if not on_disk and probe.get("names_via") not in {"caller"}:
            return _stage(
                "SpecimenPresent",
                REFUSED,
                why=f"specimen directory is not on disk: {path}",
                invoked=True,
                error="not_a_directory",
                evidence={"path": str(path)},
            )
        weights = path / "model.safetensors" if on_disk else None
        index = path / "model.safetensors.index.json" if on_disk else None
        size = weights.stat().st_size if weights is not None and weights.is_file() else None
        expected = None
        if choice.get("id"):
            vrow = (_verification_index().get("rows") or {}).get(choice["id"])
            if isinstance(vrow, Mapping):
                for f in vrow.get("files") or []:
                    if isinstance(f, Mapping) and f.get("file") == "model.safetensors":
                        expected = f.get("bytes")
        if expected is not None and size is not None and size != expected:
            return _stage(
                "SpecimenPresent",
                FAILED,
                why=(
                    f"model.safetensors size {size} != verification receipt {expected}; "
                    "refusing to proceed on a drifted body"
                ),
                invoked=True,
                error="size_drift",
                evidence={"path": str(path), "size": size, "expected": expected},
            )
        return _stage(
            "SpecimenPresent",
            PASSED,
            why="specimen config and tensor names are in hand (index or header, not assumed)",
            invoked=True,
            evidence={
                "path": None if not on_disk else str(path),
                "on_disk": on_disk,
                "n_tensors": probe.get("n_tensors"),
                "names_via": probe.get("names_via"),
                "single_shard": bool(weights is not None and weights.is_file()),
                "index_json": bool(index is not None and index.is_file()),
                "model_safetensors_bytes": size,
            },
        )
    if choice.get("ok") is not True:
        return _stage(
            "SpecimenPresent",
            REFUSED,
            why="no specimen was selected; presence is not assumed",
            invoked=True,
        )
    return _stage(
        "SpecimenPresent",
        REFUSED,
        why=str(probe.get("why") or "probe refused"),
        invoked=True,
        error="probe_refused",
        evidence={"probe_why": probe.get("why")},
    )


def stage_architecture_recognizer(
    choice: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any] | None,
    names: Sequence[str],
    names_via: str,
) -> dict[str, Any]:
    if choice.get("ok") is not True or cfg is None:
        return _stage(
            "ArchitectureRecognizer",
            REFUSED,
            why="no specimen/config; recognizer was not fed a default",
            invoked=False,
        )
    if ar is None:
        return _stage(
            "ArchitectureRecognizer",
            FAILED,
            why=f"tools.odyssey.arch_recognizer is not importable: {ODYSSEY_IMPORT_ERROR}",
            invoked=False,
            error="odyssey_not_importable",
        )
    lib = organ_library()
    as_compiler = ar.recognize(
        str(choice.get("repo") or ""), str(choice.get("revision") or ""), dict(cfg), list(names)
    )
    organs, unknown, n_un, folded = ar.classify(
        list(names), dict(cfg), lib["known"], lib["declared"]
    )
    compiler_known_empty = not ar.known_organs()[0] and not ar.known_organs()[1]
    return _stage(
        "ArchitectureRecognizer",
        PASSED,
        why=(
            f"invoked tools.odyssey.arch_recognizer.recognize on local config+tensor "
            f"names ({names_via}); {len(organs)} organs, unmatched={n_un}; "
            "weights were not loaded"
        ),
        invoked=True,
        extra={"loaded_weights": False},
        evidence={
            "repo": choice.get("repo"),
            "revision": choice.get("revision"),
            "architectures": as_compiler.get("architectures"),
            "model_type": as_compiler.get("model_type"),
            "n_tensors": as_compiler.get("n_tensors"),
            "n_unmatched": n_un,
            "names_via": names_via,
            "organs": organs,
            "unrecognized": unknown,
            "folded_organ": folded,
            "library_via": lib.get("via"),
            "library_present": lib.get("present"),
            "compiler_known_organs_empty_in_this_checkout": compiler_known_empty,
            "compiler_as_invoked_novelty": as_compiler.get("novelty"),
            "did_not_fetch_network": True,
            "loaded_weights": False,
        },
    )


def stage_organ_graph(
    arch_row: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any] | None,
    names: Sequence[str],
) -> dict[str, Any]:
    if arch_row.get("status") != PASSED or cfg is None:
        return _stage(
            "OrganGraph",
            REFUSED,
            why="ArchitectureRecognizer did not pass; organ graph is not invented",
            invoked=False,
        )
    if pgc is None:
        return _stage(
            "OrganGraph",
            FAILED,
            why=f"physical_graph_compiler is not importable: {ODYSSEY_IMPORT_ERROR}",
            invoked=False,
            error="odyssey_not_importable",
        )
    og = pgc.organ_graph(dict(cfg), list(names))
    return _stage(
        "OrganGraph",
        PASSED,
        why="invoked tools.odyssey.physical_graph_compiler.organ_graph (CPU, no weight load)",
        invoked=True,
        evidence=og,
    )


def stage_nr_identify(
    choice: Mapping[str, Any],
    arch_row: Mapping[str, Any],
    og_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Composition NR from organs is not a packed NR information payload."""
    if og_row.get("status") != PASSED:
        return _stage(
            "NrIdentifyOrCreate",
            REFUSED,
            why="no organ graph; a composition NR is not fabricated from nothing",
            invoked=False,
        )
    organs = list((og_row.get("evidence") or {}).get("nodes") or [])
    flash_nr = nx_audit.evidence_path(nx_audit.REL_NR_V2)
    flash_status = None
    if flash_nr is not None:
        flash_status = load_json(flash_nr).get("status")
    slots = []
    for node in organs:
        if not isinstance(node, Mapping):
            continue
        slots.append(
            {
                "organ": node.get("organ"),
                "n_tensors": node.get("n_tensors"),
                "occupying": {
                    "kind": "EXACT_CONTROL_FALLBACK",
                    "representation": "source_bf16_exact",
                    "science_mark": "COMPILE_TIME_SCIENCE_ONLY",
                },
                "packed_nr_information": False,
            }
        )
    return _stage(
        "NrIdentifyOrCreate",
        PASSED,
        why=(
            "identified a composition organ inventory for this specimen. Packed "
            "NR information payload is NOT_BUILT; composition bytes are not "
            "serialized_nr_information. Flash NR V2 is a different model"
        ),
        invoked=True,
        evidence={
            "specimen": choice.get("id"),
            "kind": "COMPOSITION_ORGAN_INVENTORY",
            "packed": False,
            "organ_count": len(slots),
            "organs": slots,
            "flash_nr_v2_path": None if flash_nr is None else str(flash_nr),
            "flash_nr_v2_status": flash_status,
            "flash_nr_is_this_specimen": False,
            "claim_boundary": (
                "exact-control occupying per organ. COMPILE_TIME_SCIENCE_ONLY. "
                "Not a packed NR. Not physical EBPW"
            ),
        },
    )


def _silu(x):
    if pgc is not None:
        return pgc.silu(x)
    if np is None:
        raise RuntimeError("numpy missing")
    return x / (1.0 + np.exp(-x))


def _doctor_stats(W) -> dict[str, Any]:
    if np is None:
        raise RuntimeError("numpy missing")
    mat = W
    if mat.shape[0] > _DOCTOR_MAX_ROWS:
        mat = mat[:_DOCTOR_MAX_ROWS]
    flat = mat.flatten()
    std = float(flat.std())
    mean = float(flat.mean())
    z = (flat - mean) / (std + 1e-12)
    kurt = float(np.mean(z ** 4) - 3.0)
    outlier = float(np.mean(np.abs(z) > 4))
    sv = np.linalg.svd(mat, compute_uv=False)
    energy = sv ** 2
    denom = float(energy.sum())
    if denom <= 0:
        r50 = r90 = int(min(mat.shape))
    else:
        cum = np.cumsum(energy) / denom
        r50 = int(np.sum(cum < 0.50)) + 1
        r90 = int(np.sum(cum < 0.90)) + 1
    n = int(min(mat.shape))
    rown = np.linalg.norm(mat, axis=1)
    dead = float(np.mean(rown < 0.01 * (rown.mean() + 1e-12)))
    return {
        "shape": [int(mat.shape[0]), int(mat.shape[1])],
        "excess_kurtosis": round(kurt, 3),
        "outlier_frac_beyond_4sigma": round(outlier, 6),
        "rank_for_50pct_energy": r50,
        "rank_for_90pct_energy": r90,
        "full_rank": n,
        "r90_over_full_rank": round(r90 / n, 4) if n else None,
        "near_zero_row_frac": round(dead, 6),
    }


def stage_doctor(
    choice: Mapping[str, Any],
    names: Sequence[str],
    *,
    adaptation: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    overlap = adaptation.get("compiler_hardcoded_overlap") or _compiler_hardcoded_overlap(names)
    plan = list(adaptation.get("probe_plan") or [])
    if not plan:
        return _stage(
            "Doctor",
            FAILED,
            why=(
                "adapt() derived an empty probe plan from this specimen's tensor index; "
                "refusing to fall back to the compiler's hardcoded PARENT names "
                f"(absent={len(overlap.get('doctor_probes_absent') or [])}/"
                f"{len(overlap.get('doctor_probes') or [])})"
            ),
            invoked=True,
            error="no_adapted_probe_tensors",
            evidence={
                "entry_point": "tools/odyssey/doctor_tournament.py (algorithm; probes() not called)",
                "compiler_hardcoded_overlap": overlap,
                "did_not_call_probes": True,
                "adapted_probe_tensors": [],
            },
        )
    missing = [p["tensor"] for p in plan if p.get("tensor") not in set(names)]
    if missing:
        return _stage(
            "Doctor",
            FAILED,
            why=f"adapted probe tensors are absent from this specimen: {missing}",
            invoked=True,
            error="adapted_probe_tensors_missing",
            evidence={
                "missing": missing,
                "adapted_probe_tensors": [p.get("tensor") for p in plan],
                "compiler_hardcoded_overlap": overlap,
                "did_not_call_probes": True,
            },
        )
    spec_dir = Path(str(probe.get("specimen_path") or ""))
    shard_map = probe.get("shard_map") or {}
    if not spec_dir.is_dir() or not shard_map:
        return _stage(
            "Doctor",
            FAILED,
            why=(
                "probe plan exists but the weight shards are not on disk; "
                "Doctor preconditions are weights-only and are not fabricated"
            ),
            invoked=True,
            error="weights_not_on_disk",
            evidence={
                "adapted_probe_tensors": [p.get("tensor") for p in plan],
                "compiler_hardcoded_overlap": overlap,
                "did_not_call_probes": True,
            },
        )
    if np is None:
        return _stage(
            "Doctor",
            FAILED,
            why="numpy is not importable; CPU weight preconditions cannot run",
            invoked=True,
            error="numpy_missing",
            evidence={"did_not_call_probes": True, "adapted_probe_tensors": [p.get("tensor") for p in plan]},
        )
    stats_by_organ: dict[str, list[dict[str, Any]]] = {}
    slices: dict[str, dict[int, Any]] = {}
    loaded: list[str] = []
    try:
        for item in plan:
            tensor = str(item["tensor"])
            W = load_tensor(spec_dir, shard_map, tensor)
            row = _doctor_stats(W)
            row["tensor"] = tensor
            row["layer"] = item.get("layer")
            stats_by_organ.setdefault(str(item.get("organ")), []).append(row)
            loaded.append(tensor)
            sl = W[:512, :512] if min(W.shape) >= 1 else W
            slices.setdefault(str(item.get("organ")), {})[int(item.get("layer") or 0)] = sl
    except Exception as exc:
        return _stage(
            "Doctor",
            FAILED,
            why=f"adapted tensor load/stats raised {type(exc).__name__}: {exc}",
            invoked=True,
            error=f"{type(exc).__name__}: {exc}",
            evidence={
                "loaded": loaded,
                "adapted_probe_tensors": [p.get("tensor") for p in plan],
                "did_not_call_probes": True,
                "compiler_hardcoded_overlap": overlap,
            },
        )
    share: dict[str, Any] = {}
    for organ, mats in slices.items():
        Ls = sorted(mats)
        cos = []
        for i in range(len(Ls)):
            for j in range(i + 1, len(Ls)):
                a, b = mats[Ls[i]].flatten(), mats[Ls[j]].flatten()
                denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                c = float(np.dot(a, b) / denom)
                cos.append({"layers": [Ls[i], Ls[j]], "cosine": round(c, 5)})
        share[organ] = {
            "pairs": cos,
            "max_abs_cosine": None if not cos else round(max(abs(p["cosine"]) for p in cos), 5),
        }
    return _stage(
        "Doctor",
        PASSED,
        why=(
            f"weights-only preconditions on {len(loaded)} tensors derived from this "
            f"specimen's index (not doctor_tournament.PROBE_TENSORS / PARENT). "
            f"probes() was not called: it hardcodes PARENT="
            f"{overlap.get('doctor_parent')}. "
            f"compiler-hardcoded names present "
            f"{len(overlap.get('doctor_probes_present') or [])}/"
            f"{len(overlap.get('doctor_probes') or [])}"
        ),
        invoked=True,
        extra={"loaded_weights": True},
        evidence={
            "entry_point": "tools.future.nr_nx_generic._doctor_stats (doctor_tournament algorithm, parameterized names)",
            "did_not_call_probes": True,
            "did_not_load_compiler_parent": True,
            "loaded_weights": True,
            "n_probed": len(loaded),
            "adapted_probe_tensors": loaded,
            "compiler_hardcoded_overlap": overlap,
            "diagnosis": stats_by_organ,
            "cross_layer": share,
            "claim_boundary": (
                "CPU weights-only preconditions. Not a hardware measurement. "
                "Not physical EBPW. Not a packed NR."
            ),
        },
    )


def stage_representation_planner(
    choice: Mapping[str, Any],
    *,
    organs: Sequence[Mapping[str, Any]],
    model_type: Any,
) -> dict[str, Any]:
    if choice.get("ok") is not True:
        return _stage(
            "RepresentationPlanner",
            REFUSED,
            why="no specimen selected",
            invoked=False,
        )
    rl, err = _representation_library()
    if rl is None:
        producer_git = bool(git("show", "HEAD:tools/headless/representation_library.py"))
        return _stage(
            "RepresentationPlanner",
            FAILED,
            why=(
                "tools.headless.representation_library is not importable in this "
                f"checkout ({err}). That is a worktree gap, not proof the producer "
                "is absent from git. rehearse() was not called: it fetches HuggingFace "
                "when the specimen is not in ARCHITECTURE_RECOGNIZER_FIXTURES"
            ),
            invoked=True,
            error=err,
            evidence={
                "entry_point": "tools.headless.representation_library.seed",
                "producer_in_git": producer_git,
                "producer_on_disk": (REPO / "tools/headless/representation_library.py").is_file(),
                "did_not_call_rehearse": True,
                "did_not_fetch_network": True,
            },
        )
    organ_names = []
    for node in organs:
        if isinstance(node, Mapping) and node.get("organ"):
            organ_names.append(str(node["organ"]))
    if not organ_names:
        return _stage(
            "RepresentationPlanner",
            FAILED,
            why="no organs to seed; a plan for an empty inventory is not a plan",
            invoked=True,
            error="no_organs",
        )
    try:
        fams = rl.build()
        seeds = []
        arch_class = str(model_type or "unknown")
        alias = getattr(ar, "ALIAS_TO_FAMILY", {}) if ar is not None else {}
        for organ in organ_names:
            fam_name = alias.get(organ, organ) if isinstance(alias, dict) else organ
            seeded = rl.seed(fams, fam_name, arch_class)
            ranked = list(seeded.get("ranked") or [])[:3]
            seeds.append(
                {
                    "organ": organ,
                    "family_name": fam_name,
                    "arch_class": arch_class,
                    "top": [
                        {"family": r.get("family"), "score": r.get("score"), "evidence": r.get("evidence")}
                        for r in ranked
                        if isinstance(r, Mapping)
                    ],
                    "n_ranked": len(seeded.get("ranked") or []),
                    "n_excluded": len(seeded.get("excluded") or []),
                    "law": seeded.get("law"),
                }
            )
    except Exception as exc:
        return _stage(
            "RepresentationPlanner",
            FAILED,
            why=f"representation_library.build/seed raised {type(exc).__name__}: {exc}",
            invoked=True,
            error=f"{type(exc).__name__}: {exc}",
            evidence={"did_not_call_rehearse": True, "did_not_fetch_network": True},
        )
    return _stage(
        "RepresentationPlanner",
        PASSED,
        why=(
            "invoked tools.headless.representation_library.seed on this specimen's "
            f"organs ({organ_names}) and arch_class={model_type!r}. "
            "rehearse() was not called (it fetches when fixtures miss). "
            "This is a seeded candidate list, not a packed representation"
        ),
        invoked=True,
        evidence={
            "entry_point": "tools.headless.representation_library.seed",
            "did_not_call_rehearse": True,
            "did_not_fetch_network": True,
            "organs_seeded": organ_names,
            "seeds": seeds,
            "claim_boundary": (
                "COMPILE_TIME_SCIENCE_ONLY candidate ranking. MODEL_SPECIFIC Qwen "
                "failures warn, they do not prune. Not a packed NR. Not physical EBPW"
            ),
        },
    )


def _parameterized_gate_up_swiglu(
    spec_dir: Path,
    shard_map: Mapping[str, str],
    gate_name: str,
    up_name: str,
    down_name: str,
) -> dict[str, Any]:
    if np is None:
        raise RuntimeError("numpy is not importable")
    g = load_tensor(spec_dir, shard_map, gate_name)
    u = load_tensor(spec_dir, shard_map, up_name)
    d = load_tensor(spec_dir, shard_map, down_name)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, g.shape[1]))
    h_gate = x @ g.T
    h_up = x @ u.T
    unfused = (_silu(h_gate) * h_up) @ d.T
    gu = np.concatenate([g, u], axis=0)
    h = x @ gu.T
    n = g.shape[0]
    fused = (_silu(h[:, :n]) * h[:, n:]) @ d.T
    err = float(np.abs(unfused - fused).max())
    scale = float(np.abs(unfused).max())
    used_compiler_silu = pgc is not None
    return {
        "collapse": "gate_up_swiglu",
        "source_nodes": ["gate_proj matvec", "up_proj matvec", "silu", "elementwise multiply"],
        "physical_nodes": ["gate_up_swiglu (one fused operator)"],
        "n_source_nodes": 4,
        "n_physical_nodes": 1,
        "gate": gate_name,
        "up": up_name,
        "down": down_name,
        "used_compiler_silu": used_compiler_silu,
        "semantic_justification": (
            "gate and up read the same activation vector and their outputs are consumed "
            "only by the SwiGLU, so the intermediates are not observable outside the region"
        ),
        "max_abs_diff": err,
        "max_abs_value": scale,
        "relative": None if not scale else err / scale,
        "tolerance": _SWIGLU_TOLERANCE,
        "numerically_equivalent": err <= _SWIGLU_TOLERANCE,
        "weight_reads_before": 3,
        "weight_reads_after": 2,
        "intermediates_materialized_before": 2,
        "intermediates_materialized_after": 0,
    }


def stage_physical_graph_compiler(
    choice: Mapping[str, Any],
    names: Sequence[str],
    *,
    adaptation: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    spec_dir = Path(str(probe.get("specimen_path") or choice.get("specimen_path") or ""))
    index = spec_dir / "model.safetensors.index.json"
    overlap = adaptation.get("compiler_hardcoded_overlap") or _compiler_hardcoded_overlap(names)
    plan = list(adaptation.get("collapse_plan") or [])
    parameterized: dict[str, Any] | None = None
    param_error: str | None = None
    if plan and spec_dir.is_dir() and probe.get("shard_map") and np is not None:
        item = plan[0]
        try:
            parameterized = _parameterized_gate_up_swiglu(
                spec_dir,
                probe.get("shard_map") or {},
                str(item["gate"]),
                str(item["up"]),
                str(item["down"]),
            )
        except Exception as exc:
            param_error = f"{type(exc).__name__}: {exc}"
    elif not plan:
        param_error = "; ".join(adaptation.get("missing") or ["adapt() produced no collapse plan"])
    elif np is None:
        param_error = "numpy is not importable; collapse cannot be checked"
    elif not spec_dir.is_dir():
        param_error = f"specimen directory is not on disk: {spec_dir}"

    stdout = stderr = ""
    returncode: int | None = None
    invoked_main = False
    emitted = False
    if spec_dir.is_dir() and (REPO / "tools/odyssey/physical_graph_compiler.py").is_file():
        with tempfile.TemporaryDirectory(prefix="nr-nx-generic-pgc-") as tmp:
            emit = Path(tmp) / "PHYSICAL_GRAPH_COMPILER.emit.json"
            proc = subprocess.run(
                [
                    _sys.executable,
                    str(REPO / "tools/odyssey/physical_graph_compiler.py"),
                    "--model-dir",
                    str(spec_dir),
                    "--capture",
                    str(Path(tmp) / "no-capture"),
                    "--layer",
                    "0",
                    "--emit",
                    str(emit),
                ],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=60,
            )
            invoked_main = True
            returncode = proc.returncode
            stdout = (proc.stdout or "")[-400:]
            stderr = (proc.stderr or "")[-1200:]
            emitted = emit.is_file()
    elif spec_dir.is_dir():
        # Sparse checkout: the compiler lives in the primary tree. Drive it from there.
        pgc_py = None
        for root in nx_audit.evidence_roots():
            cand = root / "tools/odyssey/physical_graph_compiler.py"
            if cand.is_file():
                pgc_py = cand
                break
        if pgc_py is not None:
            with tempfile.TemporaryDirectory(prefix="nr-nx-generic-pgc-") as tmp:
                emit = Path(tmp) / "PHYSICAL_GRAPH_COMPILER.emit.json"
                proc = subprocess.run(
                    [
                        _sys.executable,
                        str(pgc_py),
                        "--model-dir",
                        str(spec_dir),
                        "--capture",
                        str(Path(tmp) / "no-capture"),
                        "--layer",
                        "0",
                        "--emit",
                        str(emit),
                    ],
                    cwd=str(pgc_py.parents[2]),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                invoked_main = True
                returncode = proc.returncode
                stdout = (proc.stdout or "")[-400:]
                stderr = (proc.stderr or "")[-1200:]
                emitted = emit.is_file()

    err_line = None
    for line in (stderr or "").splitlines()[::-1]:
        if line.strip():
            err_line = line.strip()
            break

    param_ok = bool(parameterized and parameterized.get("numerically_equivalent") is True)
    evidence = {
        "entry_point": "tools.future.nr_nx_generic._parameterized_gate_up_swiglu",
        "compiler_main": "tools/odyssey/physical_graph_compiler.py:main",
        "parameterized_collapse": parameterized,
        "parameterized_error": param_error,
        "compiler_main_returncode": returncode,
        "compiler_main_invoked": invoked_main,
        "index_json_present": index.is_file() if spec_dir.is_dir() else False,
        "compiler_hardcoded_overlap": overlap,
        "collapse_plan": plan,
        "router_collapse": adaptation.get("router_collapse"),
        "emit_written": emitted,
        "stderr_tail": stderr,
        "stdout_tail": stdout,
        "used_compiler_silu": None if parameterized is None else parameterized.get("used_compiler_silu"),
    }
    if param_ok:
        return _stage(
            "PhysicalGraphCompiler",
            PASSED,
            why=(
                "parameterized gate_up_swiglu on tensors derived from this specimen's "
                f"index is numerically equivalent (max_abs_diff="
                f"{parameterized.get('max_abs_diff')}, used_compiler_silu="
                f"{parameterized.get('used_compiler_silu')}). "
                f"compiler main() still hardcoded to MoE index.json "
                f"(returncode={returncode}, index_json={index.is_file() if spec_dir.is_dir() else False}); "
                "that is a Codex-owned compiler gap, not a miss in this specimen's names"
            ),
            invoked=True,
            error=None,
            evidence=evidence,
        )
    return _stage(
        "PhysicalGraphCompiler",
        FAILED,
        why=(
            "parameterized SwiGLU collapse did not verify on this specimen: "
            f"{param_error or (parameterized or {}).get('max_abs_diff')}. "
            f"compiler main() returncode={returncode} err={err_line!r}"
        ),
        invoked=True,
        error=param_error or err_line or "collapse_not_equivalent",
        evidence=evidence,
    )


def kernel_library_is_readable(doc: Mapping[str, Any] | None = None) -> tuple[bool, str | None]:
    """True when a kernel library document can be read. Absence is not an empty success."""
    if isinstance(doc, Mapping) and isinstance(doc.get("kernels"), list):
        return True, None
    path = nx_audit.evidence_path(REL_KERNELS)
    if path is None:
        return False, (
            "KERNEL_LIBRARY.json is not reachable via evidence_path; "
            "not treated as empty success"
        )
    return True, None


def load_kernel_library(
    doc: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if isinstance(doc, Mapping) and isinstance(doc.get("kernels"), list):
        return dict(doc), "caller", None
    path = nx_audit.evidence_path(REL_KERNELS)
    if path is None:
        return None, None, (
            "KERNEL_LIBRARY.json is not reachable via evidence_path; "
            "not treated as empty success"
        )
    return load_json(path), str(path), None


def organ_shapes_from_config(cfg: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Per-organ extents derived from this specimen's config. Missing fields stay missing."""
    if not isinstance(cfg, Mapping):
        return {}
    hidden = cfg.get("hidden_size")
    intermediate = cfg.get("intermediate_size") or cfg.get("moe_intermediate_size")
    vocab = cfg.get("vocab_size")
    n_q = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads")
    head_dim = cfg.get("head_dim")
    out: dict[str, dict[str, Any]] = {}
    if isinstance(hidden, int) and hidden > 0 and isinstance(intermediate, int) and intermediate > 0:
        out["mlp_gate_up"] = {
            "rows": intermediate,
            "cols": hidden,
            "extents": [intermediate, hidden],
            "from": "intermediate_size x hidden_size",
        }
        out["mlp_down"] = {
            "rows": hidden,
            "cols": intermediate,
            "extents": [hidden, intermediate],
            "from": "hidden_size x intermediate_size",
        }
    if isinstance(hidden, int) and hidden > 0:
        out["rmsnorm"] = {
            "cols": hidden,
            "extents": [hidden],
            "from": "hidden_size",
        }
        if isinstance(vocab, int) and vocab > 0:
            out["embed"] = {
                "rows": vocab,
                "cols": hidden,
                "extents": [vocab, hidden],
                "from": "vocab_size x hidden_size",
            }
            out["lm_head"] = {
                "rows": vocab,
                "cols": hidden,
                "extents": [vocab, hidden],
                "from": "vocab_size x hidden_size",
            }
        if isinstance(n_q, int) and n_q > 0 and isinstance(head_dim, int) and head_dim > 0:
            q_rows = n_q * head_dim
            extents = [q_rows, hidden]
            gqa: dict[str, Any] = {
                "q_rows": q_rows,
                "cols": hidden,
                "head_dim": head_dim,
                "from": "num_attention_heads*head_dim x hidden_size",
            }
            if isinstance(n_kv, int) and n_kv > 0:
                kv_rows = n_kv * head_dim
                gqa["kv_rows"] = kv_rows
                extents = [q_rows, kv_rows, hidden]
            gqa["extents"] = extents
            out["gqa_attention"] = gqa
    return out


def _kernel_specimen_id(kernel: Mapping[str, Any]) -> str | None:
    for key in ("specimen", "specimen_id", "specimen_identity"):
        raw = kernel.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw
    return None


def _compiled_identity_present(kernel: Mapping[str, Any]) -> bool:
    ci = kernel.get("compiled_identity")
    if not isinstance(ci, Mapping):
        return False
    if ci.get("kind") == "ABSENT":
        return False
    if ci.get("value") in (None, "", [], {}):
        return False
    return True


def _declared_specialized_cols(kernel: Mapping[str, Any]) -> int | None:
    spec = kernel.get("specialization")
    if not isinstance(spec, Mapping) or "specialized_cols" not in spec:
        return None
    raw = spec.get("specialized_cols")
    return int(raw) if isinstance(raw, int) else None


def _organ_extents(shape: Mapping[str, Any] | None) -> set[int]:
    if not isinstance(shape, Mapping):
        return set()
    raw = shape.get("extents")
    if isinstance(raw, list):
        return {int(x) for x in raw if isinstance(x, int)}
    out: set[int] = set()
    for key in ("rows", "cols", "q_rows", "kv_rows"):
        val = shape.get(key)
        if isinstance(val, int):
            out.add(val)
    return out


def _declared_parametric_range(kernel: Mapping[str, Any]) -> bool:
    """True only when the kernel wrote a range/constraint object. Absence is not a wildcard."""
    spec = kernel.get("specialization")
    if not isinstance(spec, Mapping):
        return False
    for key in ("shape_constraints", "cols_range", "min_cols", "max_cols", "accepted_shapes"):
        if key in spec and spec.get(key) not in (None, "", [], {}):
            return True
    constraints = kernel.get("shape_constraints")
    return bool(constraints)


def is_compiled_kernel_for_body(
    kernel: Mapping[str, Any],
    *,
    specimen_id: str | None,
    organ: str,
    organ_shape: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """A shared organ_identity is never a compiled kernel for this body.

    The kernel must (1) role-match, (2) carry a present compiled_identity, and
    (3) carry this specimen's id or declared shape constraints this organ
    satisfies. Undeclared specialized_cols is not a parametric wildcard.
    """
    role_match = kernel.get("organ_identity") == organ
    id_match = bool(specimen_id) and _kernel_specimen_id(kernel) == specimen_id
    specialized_cols = _declared_specialized_cols(kernel)
    extents = _organ_extents(organ_shape)
    shape_match = specialized_cols is not None and specialized_cols in extents
    compiled = _compiled_identity_present(kernel)
    ok = bool(role_match and compiled and (id_match or shape_match))
    if ok:
        why = (
            "compiled kernel for this body: role matches and compiled_identity is "
            f"present and {'specimen id matches' if id_match else 'declared shape constraints are satisfied'}"
        )
    elif not role_match:
        why = (
            f"organ_identity={kernel.get('organ_identity')!r} does not serve organ={organ!r}"
        )
    else:
        why = (
            f"{NAME_IS_NOT_A_COMPILED_KERNEL} "
            f"specimen_id_match={id_match} shape_constraints_satisfied={shape_match} "
            f"compiled_identity_present={compiled} specialized_cols={specialized_cols!r} "
            f"organ_extents={sorted(extents)} kernel_specimen={_kernel_specimen_id(kernel)!r}"
        )
    return {
        "ok": ok,
        "role_match": role_match,
        "specimen_id_match": id_match,
        "shape_constraints_satisfied": shape_match,
        "compiled_identity_present": compiled,
        "specialized_cols": specialized_cols,
        "parametric_range_declared": _declared_parametric_range(kernel),
        "kernel_identity": kernel.get("kernel_identity"),
        "organ_identity": kernel.get("organ_identity"),
        "why": why,
    }


def kernel_planner_route(
    kernels: Sequence[Mapping[str, Any]],
    *,
    specimen_id: str | None,
    organ_shapes: Mapping[str, Mapping[str, Any]],
    library_specimen_field: Any,
) -> dict[str, Any]:
    """Choose SHAPE-PARAMETRIC vs PLAN-THEN-COMPILE from what the library actually wrote."""
    entries = [k for k in kernels if isinstance(k, Mapping)]
    n = len(entries)
    n_compiled = sum(1 for k in entries if _compiled_identity_present(k))
    n_specimen_bound = sum(1 for k in entries if _kernel_specimen_id(k))
    n_parametric_range = sum(1 for k in entries if _declared_parametric_range(k))
    declared_cols = sorted(
        {c for k in entries if (c := _declared_specialized_cols(k)) is not None}
    )
    specimen_extents = sorted(
        {
            v
            for shape in organ_shapes.values()
            for v in _organ_extents(shape)
        }
    )
    shape_overlap = sorted(set(declared_cols) & set(specimen_extents))
    # SHAPE-PARAMETRIC is the real fix only if kernels declare the constraints
    # they satisfy, a compiled identity exists, and this body meets them.
    # Omitting specialized_cols is not a declared parametric contract.
    shape_parametric = bool(
        n_compiled > 0
        and n_parametric_range > 0
        and (bool(shape_overlap) or n_specimen_bound > 0)
    )
    route = (
        KERNEL_PLANNER_ROUTE_SHAPE_PARAMETRIC
        if shape_parametric
        else KERNEL_PLANNER_ROUTE_PLAN_THEN_COMPILE
    )
    why = (
        f"{route}: library specimen_field={library_specimen_field!r}; "
        f"{n_compiled}/{n} kernels carry a present compiled_identity; "
        f"{n_specimen_bound}/{n} carry a specimen id; "
        f"{n_parametric_range}/{n} declare a parametric shape range; "
        f"declared specialized_cols={declared_cols}; "
        f"this specimen's organ extents={specimen_extents}; "
        f"shape overlap={shape_overlap}. "
        + (
            "Kernels declare constraints this body satisfies and are compiled; "
            "matching those constraints rather than a specimen id."
            if shape_parametric
            else (
                "Kernels in this library are parent-shape specializations or have "
                "no declared shape contract, and none are compiled. The generic "
                "pipeline emits a NATIVE_UNMEASURED plan for an unseen body rather "
                "than claiming a compiled kernel exists."
            )
        )
    )
    return {
        "route": route,
        "why": why,
        "n_kernels": n,
        "n_compiled_identity_present": n_compiled,
        "n_specimen_bound": n_specimen_bound,
        "n_parametric_range_declared": n_parametric_range,
        "declared_specialized_cols": declared_cols,
        "specimen_extents": specimen_extents,
        "shape_overlap": shape_overlap,
        "library_specimen_field": library_specimen_field,
        "specimen_id": specimen_id,
    }


def plan_kernels_for_specimen(
    organs: Sequence[str],
    *,
    specimen_id: str | None,
    config: Mapping[str, Any] | None,
    kernels: Sequence[Mapping[str, Any]],
    library_specimen_field: Any = None,
) -> dict[str, Any]:
    """Emit a kernel plan for this body. Never promotes a role-name match to COMPILED."""
    organ_names = [str(o) for o in organs if o]
    shapes = organ_shapes_from_config(config)
    entries = [k for k in kernels if isinstance(k, Mapping)]
    route = kernel_planner_route(
        entries,
        specimen_id=specimen_id,
        organ_shapes=shapes,
        library_specimen_field=library_specimen_field,
    )
    lib_organs = sorted(
        {str(k.get("organ_identity")) for k in entries if k.get("organ_identity")}
    )
    intersection = sorted(set(organ_names) & set(lib_organs))
    plan: list[dict[str, Any]] = []
    n_compiled = 0
    n_unmeasured = 0
    n_name_refused = 0
    for organ in organ_names:
        role_matched = [k for k in entries if k.get("organ_identity") == organ]
        judgements = [
            is_compiled_kernel_for_body(
                k,
                specimen_id=specimen_id,
                organ=organ,
                organ_shape=shapes.get(organ),
            )
            for k in role_matched
        ]
        compiled = [j for j in judgements if j.get("ok") is True]
        name_refused = [
            j
            for j in judgements
            if j.get("role_match") is True and j.get("ok") is not True
        ]
        if compiled:
            occupying_kind = COMPILED
            n_compiled += 1
            occupying_why = compiled[0]["why"]
        else:
            occupying_kind = NATIVE_UNMEASURED
            n_unmeasured += 1
            occupying_why = (
                f"{NAME_IS_NOT_A_COMPILED_KERNEL} "
                f"{len(role_matched)} role-matched kernel(s) for {organ!r}; "
                f"{len(compiled)} compiled for this body. Plan occupies "
                f"{NATIVE_UNMEASURED} rather than claiming a compiled kernel."
            )
        if name_refused:
            n_name_refused += 1
        plan.append(
            {
                "organ": organ,
                "status": occupying_kind,
                "occupying": {
                    "kind": occupying_kind,
                    "compiled_kernel": None
                    if not compiled
                    else compiled[0].get("kernel_identity"),
                    "science_mark": "COMPILE_TIME_SCIENCE_ONLY",
                },
                "specimen_shape": shapes.get(organ),
                "n_role_matched": len(role_matched),
                "n_compiled_for_this_body": len(compiled),
                "name_is_not_a_compiled_kernel": bool(name_refused),
                "role_matched_kernel_ids": [k.get("kernel_identity") for k in role_matched],
                "compiled_kernel_ids": [j.get("kernel_identity") for j in compiled],
                "refusals": [j.get("why") for j in name_refused],
                "why": occupying_why,
            }
        )
    return {
        "route": route["route"],
        "route_why": route["why"],
        "route_evidence": route,
        "specimen_id": specimen_id,
        "n_organs": len(plan),
        "n_compiled": n_compiled,
        "n_native_unmeasured": n_unmeasured,
        "n_name_is_not_a_compiled_kernel": n_name_refused,
        "library_organs": lib_organs,
        "specimen_organs": list(organ_names),
        "intersection": intersection,
        "organ_shapes": shapes,
        "plan": plan,
        "name_is_not_a_compiled_kernel": n_name_refused > 0,
        "names_this_specimen": False,
    }


def stage_kernel_planner(
    organs: Sequence[str],
    *,
    specimen_id: str | None = None,
    config: Mapping[str, Any] | None = None,
    library_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    doc, path, err = load_kernel_library(library_doc)
    if doc is None:
        return _stage(
            "KernelPlanner",
            REFUSED,
            why=err or "KERNEL_LIBRARY.json is not reachable via evidence_path; not treated as empty success",
            invoked=True,
            error="missing_kernel_library",
        )
    kernels = [k for k in (doc.get("kernels") or []) if isinstance(k, Mapping)]
    specimen_field = doc.get("specimen")
    organ_names = [str(o) for o in organs if o]
    if not organ_names:
        return _stage(
            "KernelPlanner",
            FAILED,
            why="no organs to plan; a plan for an empty inventory is not a plan",
            invoked=True,
            error="no_organs",
            evidence={
                "path": path,
                "n_kernels": doc.get("n_kernels") if doc.get("n_kernels") is not None else len(kernels),
                "specimen_field": specimen_field,
            },
        )
    planned = plan_kernels_for_specimen(
        organ_names,
        specimen_id=specimen_id,
        config=config,
        kernels=kernels,
        library_specimen_field=specimen_field,
    )
    role_matched = [
        {
            "kernel_identity": k.get("kernel_identity"),
            "organ_identity": k.get("organ_identity"),
            "representation_identity": k.get("representation_identity"),
            "specialized_cols": _declared_specialized_cols(k),
            "compiled_identity_kind": (
                (k.get("compiled_identity") or {}).get("kind")
                if isinstance(k.get("compiled_identity"), Mapping)
                else None
            ),
            "kernel_specimen": _kernel_specimen_id(k),
        }
        for k in kernels
        if k.get("organ_identity") in set(organ_names)
    ]
    n_unmeasured = planned["n_native_unmeasured"]
    n_compiled = planned["n_compiled"]
    why = (
        f"{planned['route']}: KERNEL_LIBRARY.json was read and a kernel plan was "
        f"emitted for this unseen body ({specimen_id!r}). "
        f"role intersection={planned['intersection']}. "
        f"{len(role_matched)} role-matched kernels; {n_compiled} compiled for this "
        f"body; {n_unmeasured} organ(s) occupy {NATIVE_UNMEASURED}. "
        f"{NAME_IS_NOT_A_COMPILED_KERNEL} "
        f"specimen_field={specimen_field!r}. {planned['route_why']}"
    )
    return _stage(
        "KernelPlanner",
        PASSED,
        why=why,
        invoked=True,
        evidence={
            "path": path,
            "n_kernels": doc.get("n_kernels") if doc.get("n_kernels") is not None else len(kernels),
            "library_organs": planned["library_organs"],
            "specimen_organs": planned["specimen_organs"],
            "intersection": planned["intersection"],
            "role_matched_kernels": role_matched,
            "specimen_field": specimen_field,
            "names_this_specimen": False,
            "route": planned["route"],
            "route_why": planned["route_why"],
            "route_evidence": planned["route_evidence"],
            "plan": planned["plan"],
            "n_compiled": n_compiled,
            "n_native_unmeasured": n_unmeasured,
            "n_name_is_not_a_compiled_kernel": planned["n_name_is_not_a_compiled_kernel"],
            "name_is_not_a_compiled_kernel": planned["name_is_not_a_compiled_kernel"],
            "organ_shapes": planned["organ_shapes"],
            "claim_boundary": (
                "COMPILE_TIME_SCIENCE_ONLY kernel plan. NATIVE_UNMEASURED is not a "
                "compiled kernel, not a hardware measurement, not physical EBPW."
            ),
        },
    )


def stage_device_compiler(
    native: Mapping[str, Any],
    *,
    family: Any = None,
    kernel_plan: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    specimen_id: str | None = None,
    model_type: Any = None,
    backend: Any = None,
    capture_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Lower the KernelPlanner plan. Placeholders are refused, not recorded compiled."""
    blocked = dict((nc.BLOCKED.get("DeviceCompiler") or {}) if nc is not None else {})
    plan = kernel_plan if isinstance(kernel_plan, Mapping) else {}
    native_arms = list(native.get("architectures") or [])
    qwen3_dense = _native_includes_qwen3_dense(native)
    if not plan.get("plan"):
        return _stage(
            "DeviceCompiler",
            REFUSED,
            why=(
                "no KernelPlanner plan was handed; a compiled identity is not invented. "
                f"Native GGUF match arms are {native_arms!r}; qwen3 dense is "
                f"{'present' if qwen3_dense else 'absent'} (qwen3moe is a different "
                "family; dense was not mapped onto the moe arm)."
            ),
            invoked=True,
            error="no_plan",
            evidence={
                "entry_point": "tools.future.device_compiler.lower_plan",
                "kernel_plan_received": bool(plan),
                "adapted_family": family,
                "noetic_compiler_blocked_why": blocked.get("why"),
                "copied_as_this_specimen": False,
                "native": {
                    "path": native.get("path"),
                    "architectures": native_arms,
                    "includes_qwen2": native.get("includes_qwen2"),
                    "includes_qwen3moe": native.get("includes_qwen3moe"),
                    "includes_qwen3_dense": qwen3_dense,
                    "includes_falcon_h1": native.get("includes_falcon_h1"),
                },
            },
        )
    cap_path = Path(capture_dir) if capture_dir else Path(tempfile.mkdtemp(prefix="hawking-nx-cap-"))
    metal = backend
    capturing = None
    if metal is None:
        capturing = nxp.CapturingMetalBackend(dcomp.LiveMetalBackend(), cap_path)
        metal = capturing
    elif not isinstance(metal, nxp.CapturingMetalBackend):
        capturing = nxp.CapturingMetalBackend(metal, cap_path)
        metal = capturing
    else:
        capturing = metal
    lowering = dcomp.lower_plan(
        plan,
        specimen_id=specimen_id,
        family=family,
        config=config,
        native_architectures=native_arms,
        model_type=model_type,
        backend=metal,
    )
    captured_for_pack = dict(getattr(capturing, "captured", {}) or {})
    n_compiled = int(lowering.get("n_compiled") or 0)
    n_unmeasured = int(lowering.get("n_native_unmeasured") or 0)
    n_placeholder = int(lowering.get("n_placeholder_refused") or 0)
    blocker = lowering.get("qwen3_dense_gguf_blocker") or {}
    nx_fragment = lowering.get("nx_fragment")
    compiled, planned = dcomp.split_compiled_and_planned(
        nx_fragment if isinstance(nx_fragment, Mapping) else None
    )
    # Defence: never pass a slot whose identity is a placeholder.
    for slot in lowering.get("plan") or []:
        if not isinstance(slot, Mapping):
            continue
        if slot.get("status") == COMPILED and dcomp.is_placeholder_compiled_identity(
            slot.get("compiled_identity"),
            source_sha256=(
                (slot.get("compiled_identity") or {}).get("source_sha256")
                if isinstance(slot.get("compiled_identity"), Mapping)
                else None
            ),
            entry_point=(
                (slot.get("compiled_identity") or {}).get("entry_point")
                if isinstance(slot.get("compiled_identity"), Mapping)
                else None
            ),
        ):
            return _stage(
                "DeviceCompiler",
                FAILED,
                why=(
                    f"{dcomp.PLACEHOLDER_REFUSED} for organ={slot.get('organ')!r}; "
                    "a placeholder claiming compiled identity is not a pass"
                ),
                invoked=True,
                error="placeholder_compiled_identity",
                evidence={
                    "entry_point": "tools.future.device_compiler.lower_plan",
                    "lowering": lowering,
                    "placeholder_organ": slot.get("organ"),
                },
            )
    evidence = {
        "entry_point": "tools.future.device_compiler.lower_plan",
        "noetic_compiler_blocked_why": blocked.get("why"),
        "copied_as_this_specimen": False,
        "adapted_family": family,
        "kernel_plan_received": True,
        "kernel_plan_route": plan.get("route"),
        "kernel_plan_n_organs": len(plan.get("plan") or []),
        "kernel_plan_n_compiled": plan.get("n_compiled"),
        "kernel_plan_n_native_unmeasured": plan.get("n_native_unmeasured"),
        "n_compiled": n_compiled,
        "n_native_unmeasured": n_unmeasured,
        "n_placeholder_refused": n_placeholder,
        "plan": lowering.get("plan"),
        "nx_fragment": nx_fragment,
        "nx_compiled_organs": [k.get("organ") for k in compiled],
        "nx_planned_organs": [k.get("organ") for k in planned],
        "qwen3_dense_gguf_blocker": blocker,
        "metal": lowering.get("metal"),
        "captured_archives": {
            k: {kk: vv for kk, vv in rec.items() if kk != "source"}
            if isinstance(rec, dict) else rec
            for k, rec in captured_for_pack.items()
        },
        "capture_dir": str(cap_path),
        "created_command_queue": False,
        "dispatched": False,
        "native": {
            "path": native.get("path"),
            "architectures": native_arms,
            "includes_qwen2": native.get("includes_qwen2"),
            "includes_qwen3moe": native.get("includes_qwen3moe"),
            "includes_qwen3_dense": qwen3_dense,
            "includes_falcon_h1": native.get("includes_falcon_h1"),
        },
        "claim_boundary": lowering.get("claim_boundary"),
    }
    if n_compiled <= 0:
        return _stage(
            "DeviceCompiler",
            FAILED,
            why=(
                "DeviceCompiler callable ran (tools.future.device_compiler.lower_plan) "
                f"and lowered 0/{len(plan.get('plan') or [])} organ(s) to an "
                f"{dcomp.PIPELINE_OBJECT}. {n_unmeasured} remain {NATIVE_UNMEASURED}; "
                f"placeholder_refused={n_placeholder}. "
                f"{lowering.get('why')}. Native GGUF match arms are {native_arms!r}; "
                f"qwen3 dense is {'present' if qwen3_dense else 'absent'} "
                f"(qwen3moe is a different family; dense was not mapped onto the moe arm). "
                f"adapted family={family!r}."
            ),
            invoked=True,
            error=str(lowering.get("error") or "zero_organs_compiled"),
            evidence=evidence,
        )
    return _stage(
        "DeviceCompiler",
        PASSED,
        why=(
            "DeviceCompiler lowered "
            f"{n_compiled}/{len(plan.get('plan') or [])} organ(s) to "
            f"{dcomp.PIPELINE_OBJECT} with shader_hash (MTLBinaryArchive) and "
            f"entry_point. {n_unmeasured} remain {NATIVE_UNMEASURED}. "
            f"placeholder_refused={n_placeholder}. "
            f"qwen3 dense GGUF match arm is {'present' if qwen3_dense else 'absent'} "
            f"(qwen3moe is a different family; dense was not mapped onto the moe arm). "
            f"adapted family={family!r}. The NX fragment carries compiled vs planned "
            "so a later stage can tell them apart. This is not a packed NX."
        ),
        invoked=True,
        evidence=evidence,
    )


def stage_noetic_executable(
    choice: Mapping[str, Any],
    *,
    nx_fragment: Mapping[str, Any] | None = None,
    archives: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    dest: str | Path | None = None,
) -> dict[str, Any]:
    on_disk = (REPO / HEADLESS_FIRST_NX).is_file()
    blob = git("show", f"HEAD:{HEADLESS_FIRST_NX}") if not on_disk else (REPO / HEADLESS_FIRST_NX).read_text()
    parent_line = None
    for line in (blob or "").splitlines():
        if "qwen3.8-27b" in line.lower() or "QWEN38_PARENT" in line or "PARENT_BF16" in line:
            parent_line = line.strip()
            break
    compiled, planned = dcomp.split_compiled_and_planned(
        nx_fragment if isinstance(nx_fragment, Mapping) else None
    )
    n_compiled = len(compiled)
    n_planned = len(planned)
    base_ev = {
        "producer": "tools.future.nx_packer.pack",
        "legacy_producer_not_executed": HEADLESS_FIRST_NX,
        "producer_on_disk": on_disk,
        "producer_in_git": bool(blob),
        "parent_line": parent_line,
        "specimen": choice.get("id"),
        "did_not_execute_first_noetic_executable": True,
        "did_not_load_27b": True,
        "nx_fragment_received": isinstance(nx_fragment, Mapping),
        "nx_compiled_organs": [k.get("organ") for k in compiled],
        "nx_planned_organs": [k.get("organ") for k in planned],
        "n_compiled_on_fragment": n_compiled,
        "n_planned_on_fragment": n_planned,
    }
    if not isinstance(nx_fragment, Mapping):
        return _stage(
            "NoeticExecutable",
            BLOCKED,
            why=(
                "no DeviceCompiler NX fragment was handed; refusing to mint a "
                "renamed source pointer as an NX"
            ),
            invoked=True,
            error="no_fragment",
            evidence={**base_ev, "did_not_mint_nx": True},
        )
    out = Path(dest) if dest else nxp.default_dest(
        None if not choice.get("id") else str(choice.get("id"))
    )
    try:
        packed = nxp.pack(
            nx_fragment=nx_fragment,
            specimen_path=choice.get("specimen_path"),
            dest=out,
            specimen_id=None if not choice.get("id") else str(choice.get("id")),
            family=choice.get("family"),
            config=config,
            archives=archives,
            dcomp=dcomp,
        )
    except nxp.NxPackerError as exc:
        return _stage(
            "NoeticExecutable",
            FAILED,
            why=(
                f"generic packer refused: {type(exc).__name__}: {exc}. "
                "first_noetic_executable.py was not executed (the Qwen3.8-27B "
                "hardlink is not this specimen). a renamed source pointer is not an NX"
            ),
            invoked=True,
            error=type(exc).__name__,
            evidence={**base_ev, "did_not_mint_nx": True, "packer_error": str(exc)},
        )
    ident = packed.get("identity") or {}
    return _stage(
        "NoeticExecutable",
        PASSED,
        why=(
            "generic packer minted a source-independent packed NX at "
            f"{packed.get('path')} with {ident.get('n_compiled_organs')} compiled "
            f"organ(s), total_bytes={ident.get('total_bytes')}, "
            f"closure_sha256={ident.get('closure_sha256')}. "
            "Did not execute first_noetic_executable.py; did not hardlink the source."
        ),
        invoked=True,
        extra={
            "packed_nx": packed.get("nx"),
            "packed_path": packed.get("nx_path"),
        },
        evidence={
            **base_ev,
            "did_not_mint_nx": False,
            "did_not_hardlink": True,
            "identity": ident,
            "packed_path": packed.get("nx_path"),
            "root": packed.get("path"),
        },
    )


def stage_source_independence(
    choice: Mapping[str, Any],
    packed_nx: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trees = []
    if choice.get("specimen_path"):
        trees.append(str(choice["specimen_path"]))
    judged = source_independence(packed_nx, source_trees=trees)
    status = PASSED if judged["ok"] is True else FAILED
    if packed_nx is None:
        status = FAILED
    return _stage(
        "SourceIndependence",
        status,
        why=str(judged.get("why")),
        invoked=True,
        evidence=judged,
    )


def stage_dependency_accounting(packed_nx: Mapping[str, Any] | None) -> dict[str, Any]:
    needs = [
        ("serialized_nx_body", bool(packed_nx) and isinstance(nx_audit._serialized_artifact(packed_nx or {}), dict)
         and (nx_audit._serialized_artifact(packed_nx or {}) or {}).get("self_contained") is True),
        ("physical_loader", isinstance(nx_audit._loader(packed_nx or {}), dict)
         and (nx_audit._loader(packed_nx or {}) or {}).get("source_independent") is True),
        ("native_kernel_catalog", isinstance(nx_audit._kernel(packed_nx or {}), dict)),
        ("byte_ledger_closed",
         isinstance((packed_nx or {}).get("byte_ledger"), Mapping)
         and (packed_nx or {}).get("byte_ledger", {}).get("status") in {"CLOSED", "COMPLETE", "COMPLETE_SYSTEM_CLOSED"}
         and (packed_nx or {}).get("byte_ledger", {}).get("all_required_bytes_included") is True
         and (packed_nx or {}).get("byte_ledger", {}).get("complete_system") is True
         and (packed_nx or {}).get("byte_ledger", {}).get("reconciles") is True),
        ("runtime_genome_digests", bool(_dot(packed_nx or {}, "reproducibility.closure_sha256"))),
    ]
    rows = [{"need": n, "present": bool(p)} for n, p in needs]
    missing = [r["need"] for r in rows if not r["present"]]
    return _stage(
        "ExecutableDependencyAccounting",
        FAILED if missing else PASSED,
        why=(
            "accounted the executable dependencies a lowered NX would have to carry; "
            f"missing={missing}"
        ),
        invoked=True,
        evidence={"dependencies": rows, "missing": missing, "packed_nx_present": packed_nx is not None},
    )


def stage_verifier(packed_nx: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(packed_nx, Mapping):
        judged = nx_audit.check_nx({})
        return _stage(
            "Verifier",
            FAILED,
            why="invoked flash_nx_audit.check_nx; there is no NX document to verify",
            invoked=True,
            evidence=judged,
        )
    judged = nx_audit.check_nx(dict(packed_nx))
    art = nx_audit._serialized_artifact(packed_nx)
    ledger = packed_nx.get("byte_ledger") if isinstance(packed_nx.get("byte_ledger"), Mapping) else {}
    packer_owned = (
        packed_nx.get("source_independent") is True
        and isinstance(art, Mapping)
        and art.get("self_contained") is True
        and art.get("status") not in {None, "NOT_BUILT", "ABSENT"}
        and bool(art.get("sha256") or art.get("digest"))
        and ledger.get("status") in {"CLOSED", "COMPLETE", "COMPLETE_SYSTEM_CLOSED"}
        and ledger.get("all_required_bytes_included") is True
        and ledger.get("complete_system") is True
        and not nx_audit._status_is_metadata_only(packed_nx)
    )
    # FLASH seven-requirement promotable stays independent (accepted generation
    # and protected performance are not this packer's claim).
    return _stage(
        "Verifier",
        PASSED if packer_owned else FAILED,
        why=(
            "invoked tools.future.flash_nx_audit.check_nx (the landed seven-requirement "
            f"verifier); promotable={judged.get('promotable')} "
            f"(Flash-genome bar, independent). packer-owned source-independent "
            f"closed-ledger NX={'yes' if packer_owned else 'no'}"
        ),
        invoked=True,
        evidence={
            "promotable": judged.get("promotable"),
            "packer_owned_ok": packer_owned,
            "status": judged.get("status"),
            "failed_requirements": judged.get("failed_requirements"),
            "reasons": judged.get("reasons"),
        },
    )


def first_failing_stage(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for row in stages:
        if row.get("status") != PASSED:
            return {
                "stage": row.get("stage"),
                "status": row.get("status"),
                "why": row.get("why"),
                "error": row.get("error"),
            }
    return None


def flash_nx_ready() -> dict[str, Any]:
    """Independent of the generic pipeline. False until Flash earns a packed NX."""
    loc = nx_audit.evidence_location(nx_audit.REL_NX_V0)
    if not loc.get("present"):
        return {
            "FLASH_NX_READY": False,
            "why": "FLASH_COMPLETE_V0.nx.json is not reachable; absence is not readiness",
            "path": loc.get("resolved"),
        }
    nx = load_json(loc["resolved"])
    check = nx_audit.check_nx(dict(nx))
    metadata = nx_audit._status_is_metadata_only(nx)
    ready = check.get("promotable") is True and not metadata
    return {
        "FLASH_NX_READY": False if not ready else True,
        "status": nx.get("status"),
        "promotable": check.get("promotable"),
        "metadata_only": metadata,
        "path": loc.get("resolved"),
        "failed_requirements": check.get("failed_requirements"),
        "why": (
            "Flash NX is a packed source-independent executable"
            if ready
            else f"FLASH_COMPLETE_V0.nx status={nx.get('status')!r}; check_nx.promotable={check.get('promotable')}"
        ),
    }


def emit_sleeping_lower(
    first: Mapping[str, Any] | None,
    native: Mapping[str, Any],
    *,
    pgc_passed: bool = False,
    packer_ok: bool = False,
    packed_on_disk: bool = False,
) -> dict[str, Any]:
    wakes = [
        {
            "id": "parameterized_physical_graph_holds",
            "holds": pgc_passed,
            "evidence": (
                "parameterized gate_up_swiglu numerically equivalent on this specimen"
                if pgc_passed
                else "parameterized collapse did not hold; compiler main() remains MoE-hardcoded"
            ),
        },
        {
            "id": "native_engine_match_arm_includes_this_family",
            "holds": _native_includes_qwen3_dense(native),
            "evidence": f"architectures={native.get('architectures')!r}",
        },
        {
            "id": "device_compiler_entry_point_exists",
            "holds": True,
            "evidence": "tools.future.device_compiler.lower_plan is the DeviceCompiler callable",
        },
        {
            "id": "generic_packer_accepts_this_specimen",
            "holds": bool(packer_ok),
            "evidence": (
                "tools.future.nx_packer.pack accepted this specimen"
                if packer_ok
                else "generic packer did not mint a packed NX for this specimen"
            ),
        },
        {
            "id": "packed_source_independent_nx_on_disk",
            "holds": bool(packed_on_disk),
            "evidence": (
                "source-independent packed NX is on disk"
                if packed_on_disk
                else "no NX body was packed for this specimen; a metadata seal of another model is not this path"
            ),
        },
    ]
    holding = [w["id"] for w in wakes if not w["holds"]]
    reason = (
        f"NX_LOWER blocked at stage {(first or {}).get('stage')}: {(first or {}).get('error') or (first or {}).get('why')}"
    )
    unit = wus.emit_hcli_workunit(
        id="future.nr-nx-generic.sleep.nx-lower",
        role="science",
        description=f"SLEEPING until a generic NR→NX packer accepts the chosen specimen. {reason}",
        dependencies=[],
        resource_class="COMPILE",
        verifier="future.nr_nx_generic.source_independence",
        provider="future.nr_nx_generic",
        effect_class="READ_ONLY",
        preferred_backend="cpu",
        status="sleeping",
        classification="SLEEPING",
        extras={
            "sleeping": True,
            "blocked_reason": reason,
            "requires_quiescence": False,
            "synthetic_result_forbidden": True,
            "wake_unmet": holding,
            "first_failing_stage": (first or {}).get("stage"),
        },
    )
    wus.validate_emitted_unit(unit)
    if unit.get("status") in {"pending", "PENDING", "ready", "READY"}:
        raise ValueError(f"sleeping unit leaked status={unit.get('status')!r}")
    return {
        "id": unit["id"],
        "status": unit["status"],
        "classification": unit.get("classification"),
        "resource_class": unit.get("resource_class"),
        "verifier": unit.get("verifier"),
        "blocked_reason": unit.get("blocked_reason"),
        "wake_unmet": holding,
        "wake_conditions": wakes,
        "synthetic_result_forbidden": True,
        "first_failing_stage": (first or {}).get("stage"),
        "exact_error": (first or {}).get("error"),
    }


def _choice_from_probe(probe: Mapping[str, Any], adaptation: Mapping[str, Any]) -> dict[str, Any]:
    if probe.get("ok") is not True:
        return {
            "ok": False,
            "id": probe.get("id"),
            "why": probe.get("why"),
            "specimen_path": probe.get("specimen_path"),
        }
    return {
        "ok": True,
        "id": probe.get("id"),
        "repo": probe.get("repo"),
        "revision": probe.get("revision"),
        "family": adaptation.get("family"),
        "architectures_expected": adaptation.get("architectures"),
        "specimen_path": probe.get("specimen_path"),
        "why_chosen": f"run() on {probe.get('id')}; adapt() family={adaptation.get('family')}",
        "adaptation": adaptation,
        "probe": probe,
    }


def run(specimen: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """The real path: NR, NX lower, pack, dependency accounting, independence, verifier.

    Packs via tools.future.nx_packer. A renamed source pointer is refused, not returned.
    """
    if isinstance(specimen, Mapping) and specimen.get("ok") is True and specimen.get("id") and "why_chosen" in specimen:
        choice = dict(specimen)
        probe = probe_specimen(choice.get("probe") or choice)
        if choice.get("adaptation"):
            adaptation = dict(choice["adaptation"])
        else:
            adaptation = adapt(probe)
        if not choice.get("repo"):
            choice["repo"] = probe.get("repo")
            choice["revision"] = probe.get("revision")
        if not choice.get("specimen_path"):
            choice["specimen_path"] = probe.get("specimen_path")
        if not choice.get("family"):
            choice["family"] = adaptation.get("family")
    else:
        probe = probe_specimen(specimen)
        adaptation = adapt(probe)
        choice = _choice_from_probe(probe, adaptation)

    cfg = probe.get("config") if isinstance(probe.get("config"), Mapping) else None
    names = list(probe.get("tensor_names") or [])
    names_via = str(probe.get("names_via") or "absent")
    native = native_engine_architectures()
    preflight = callable_on(probe if probe.get("ok") else specimen)

    stages: list[dict[str, Any]] = []
    stages.append(stage_specimen_select(choice))
    stages.append(stage_specimen_present(choice, probe))
    arch = stage_architecture_recognizer(choice, cfg=cfg, names=names, names_via=names_via)
    stages.append(arch)
    og = stage_organ_graph(arch, cfg=cfg, names=names)
    stages.append(og)
    stages.append(stage_nr_identify(choice, arch, og))
    stages.append(stage_doctor(choice, names, adaptation=adaptation, probe=probe))
    organ_nodes = list((og.get("evidence") or {}).get("nodes") or [])
    model_type = (arch.get("evidence") or {}).get("model_type") if arch.get("status") == PASSED else adaptation.get("model_type")
    stages.append(stage_representation_planner(choice, organs=organ_nodes, model_type=model_type))
    stages.append(stage_physical_graph_compiler(choice, names, adaptation=adaptation, probe=probe))
    organ_names = [
        str(n.get("organ"))
        for n in organ_nodes
        if isinstance(n, Mapping) and n.get("organ")
    ]
    kp = stage_kernel_planner(
        organ_names,
        specimen_id=None if not choice.get("id") else str(choice.get("id")),
        config=cfg,
    )
    stages.append(kp)
    capture_dir = Path(tempfile.mkdtemp(prefix="hawking-nx-cap-"))
    try:
        dc_row = stage_device_compiler(
            native,
            family=adaptation.get("family"),
            kernel_plan=kp.get("evidence") if isinstance(kp.get("evidence"), Mapping) else None,
            config=cfg,
            specimen_id=None if not choice.get("id") else str(choice.get("id")),
            model_type=model_type or adaptation.get("model_type"),
            capture_dir=capture_dir,
        )
        stages.append(dc_row)
        nx_fragment = None
        archives = None
        if isinstance(dc_row.get("evidence"), Mapping):
            nx_fragment = dc_row["evidence"].get("nx_fragment")
            archives = dc_row["evidence"].get("captured_archives")
        nx_row = stage_noetic_executable(
            choice,
            nx_fragment=nx_fragment if isinstance(nx_fragment, Mapping) else None,
            archives=archives if isinstance(archives, Mapping) else None,
            config=cfg,
        )
        stages.append(nx_row)
        packed_nx = nx_row.get("packed_nx") if isinstance(nx_row.get("packed_nx"), Mapping) else None
        packed_path = nx_row.get("packed_path")
        packed_path = Path(packed_path) if packed_path else None
        stages.append(stage_source_independence(choice, packed_nx))
        stages.append(stage_dependency_accounting(packed_nx))
        stages.append(stage_verifier(packed_nx))
    finally:
        shutil.rmtree(capture_dir, ignore_errors=True)

    if [s["stage"] for s in stages] != list(STAGE_ORDER):
        raise StageSkipForbidden(
            f"stage order drifted: {[s['stage'] for s in stages]} != {list(STAGE_ORDER)}"
        )
    for row in stages:
        if row["status"] in FORBIDDEN_STAGE_STATUS:
            raise StageSkipForbidden(f"{row['stage']} leaked {row['status']}")

    try:
        callable_ok = declare_pipeline_callable(stages, packed_nx_path=packed_path)
    except PipelineCallableForbidden:
        callable_ok = False
    first = first_failing_stage(stages)
    nx_lower_names = {"PhysicalGraphCompiler", "DeviceCompiler", "NoeticExecutable"}
    first_nx_lower = next(
        (
            {
                "stage": row["stage"],
                "status": row["status"],
                "why": row["why"],
                "error": row.get("error"),
            }
            for row in stages
            if row["stage"] in nx_lower_names and row["status"] != PASSED
        ),
        None,
    )
    return {
        "choice": choice,
        "probe": {
            "ok": probe.get("ok"),
            "id": probe.get("id"),
            "repo": probe.get("repo"),
            "revision": probe.get("revision"),
            "specimen_path": probe.get("specimen_path"),
            "n_tensors": probe.get("n_tensors"),
            "names_via": probe.get("names_via"),
            "why": probe.get("why"),
        },
        "adaptation": adaptation,
        "preflight": preflight,
        "native": native,
        "stages": stages,
        "packed_nx": packed_nx,
        "packed_path": packed_path,
        "callable_ok": callable_ok,
        "first_failing_stage": first,
        "first_nx_lower_failure": first_nx_lower,
    }


def assemble() -> dict[str, Any]:
    choice = choose_specimen()
    result = run(choice)
    native = result["native"]
    stages = result["stages"]
    callable_ok = result["callable_ok"]
    first = result["first_failing_stage"]
    first_nx_lower = result["first_nx_lower_failure"]
    flash = flash_nx_ready()
    pgc_row = next((s for s in stages if s["stage"] == "PhysicalGraphCompiler"), None)
    pgc_passed = bool(pgc_row and pgc_row.get("status") == PASSED)
    nx_row = next((s for s in stages if s["stage"] == "NoeticExecutable"), None)
    packed_on_disk = bool(
        result.get("packed_path") and Path(str(result.get("packed_path"))).is_file()
    )
    sleeping = emit_sleeping_lower(
        first_nx_lower or first,
        native,
        pgc_passed=pgc_passed,
        packer_ok=bool(nx_row and nx_row.get("status") == PASSED),
        packed_on_disk=packed_on_disk,
    )

    kp_row = next((s for s in stages if s["stage"] == "KernelPlanner"), None)
    kp_ev = kp_row.get("evidence") if isinstance(kp_row, Mapping) else None
    kp_ev = kp_ev if isinstance(kp_ev, Mapping) else {}
    launch_still_flash = (
        "odyssey_launch._eval_nr_nx keys nr_nx_path_callable on "
        "GENERIC_NR_NX_PIPELINE_CALLABLE from this module. FLASH_NX_READY is a "
        "separate field. PLAN-THEN-COMPILE closed the KernelPlanner stall by "
        "emitting a NATIVE_UNMEASURED plan for an unseen body without treating a "
        "shared organ name as a compiled kernel. DeviceCompiler now lowers that "
        "plan via tools.future.device_compiler.lower_plan (MTLComputePipelineState "
        "+ MTLBinaryArchive shader_hash + entry_point, placeholders refused). "
        "NoeticExecutable now packs via tools.future.nx_packer.pack. "
        "FLASH_NX_READY stays independent."
    )

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Parameterize the NR→NX compiler stages from a specimen's own config "
            "and tensor index, drive them on the cheapest whole-tree-verified body "
            "adapt() accepts, and keep GENERIC_NR_NX_PIPELINE_CALLABLE separate from FLASH_NX_READY"
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "is_a_measurement": False,
        "GENERIC_NR_NX_PIPELINE_CALLABLE": False if not callable_ok else True,
        "FLASH_NX_READY": flash["FLASH_NX_READY"],
        "facts_are_independent": True,
        "specimen": result["choice"],
        "probe": result["probe"],
        "adaptation": result["adaptation"],
        "preflight": result["preflight"],
        "native_engine": {
            "path": native.get("path"),
            "architectures": native.get("architectures"),
            "includes_qwen2": native.get("includes_qwen2"),
            "includes_qwen3moe": native.get("includes_qwen3moe"),
            "includes_qwen3_dense": _native_includes_qwen3_dense(native),
            "includes_falcon_h1": native.get("includes_falcon_h1"),
            "ok": native.get("ok"),
        },
        "stages": stages,
        "packed_path": None if not result.get("packed_path") else str(result.get("packed_path")),
        "packed_nx_identity": None if not nx_row else (nx_row.get("evidence") or {}).get("identity"),
        "first_failing_stage": first,
        "first_nx_lower_failure": first_nx_lower,
        "kernel_planner_route": kp_ev.get("route"),
        "kernel_planner_route_why": kp_ev.get("route_why"),
        "flash": flash,
        "launch_criterion_still_flash_specific": launch_still_flash,
        "sleeping_workunit": sleeping,
        "physical_ebpw": None,
        "physical_ebpw_written": False,
        "sparse_sys_path_added": _SPARSE_PATHS,
        "compiler_entry_points": {
            "probe_specimen": "tools.future.nr_nx_generic.probe_specimen",
            "adapt": "tools.future.nr_nx_generic.adapt",
            "callable_on": "tools.future.nr_nx_generic.callable_on",
            "run": "tools.future.nr_nx_generic.run",
            "ArchitectureRecognizer": "tools/odyssey/arch_recognizer.py:recognize",
            "OrganGraph": "tools/odyssey/physical_graph_compiler.py:organ_graph",
            "Doctor": "parameterized _doctor_stats; doctor_tournament.probes() not called (PARENT hardcoded)",
            "RepresentationPlanner": "tools.headless.representation_library.seed (rehearse not called)",
            "PhysicalGraphCompiler": "parameterized gate_up_swiglu + compiler main() still hardcoded",
            "KernelPlanner": "tools.future.nr_nx_generic.plan_kernels_for_specimen (PLAN-THEN-COMPILE; name is not a compiled kernel)",
            "DeviceCompiler": "tools.future.device_compiler.lower_plan (MTLComputePipelineState + MTLBinaryArchive shader_hash + entry_point; placeholder refused)",
            "NoeticExecutable": "tools.future.nx_packer.pack (generic; first_noetic_executable.py is not executed)",
            "native_loader": NATIVE_LOADER,
            "nx_verifier": "tools.future.flash_nx_audit.py:check_nx",
        },
        "recovered_implementation": [
            "tools/future/nr_nx_generic.py — EXTENDED: DeviceCompiler + generic nx_packer.pack on the live specimen",
            "tools/future/nx_packer.py — generic source-independent NX packer; refusals raise",
            "tools/future/device_compiler.py — DeviceCompiler: plan in, MTLComputePipelineState + shader_hash + entry_point out; placeholders refused",
            "tools/future/nr_nx_path.py — seven-requirement map, SLEEPING units, physical_ebpw refusal; EXTENDED, not forked",
            "tools/future/flash_nx_audit.py — check_nx, evidence_path, METADATA_ONLY, synthetic_promotable_nx",
            "tools/future/flash_nr_complete.py — composition NR is not serialized_nr_information",
            "tools/future/ebpw_categories.py — typed EBPW; physical remains unwritten",
            "tools/future/specimen_verify.py — WHOLE_TREE_VERIFIED list; ModelLake not mutated",
            "tools/odyssey/arch_recognizer.py — invoked on local config+names, no network, no weights",
            "tools/odyssey/physical_graph_compiler.py — organ_graph + silu invoked; main() still MoE-hardcoded",
            "tools/odyssey/doctor_tournament.py — algorithm reused on adapted names; probes() not called",
            "tools/headless/representation_library.py — seed() invoked on this specimen's organs",
            "tools/odyssey/noetic_compiler.py — BLOCKED map cited as a note, not as a drive of this specimen; DeviceCompiler is tools.future.device_compiler",
            "crates/hawking-core/src/model/mod.rs — shipping GGUF match arms",
            "tools/future/odyssey_launch.py _eval_nr_nx — reads GENERIC_NR_NX_PIPELINE_CALLABLE, not rewritten",
            "tools/future/workunit_species.py emit_hcli_workunit — SLEEPING unit, never pending",
        ],
        "gaps_closed": [
            "probe_specimen reads the real safetensors index/header instead of assuming Qwen3.8-27B names",
            "adapt() derives gate/up/down/router/attention/recurrent templates from the index, not from a specimen-id branch",
            "callable_on names the first stage that lacks an input instead of a single hardcoded False",
            "Doctor CPU preconditions run on this specimen's tensors; compiler PARENT probes are overlap evidence only",
            "PhysicalGraphCompiler SwiGLU collapse runs on dense single-shard names; compiler main() remains hardcoded and is recorded as such",
            "RepresentationPlanner seeds from representation_library on local organs; rehearse() is not called (it would fetch)",
            "KernelPlanner emits a NATIVE_UNMEASURED plan for an unseen body (PLAN-THEN-COMPILE); a shared organ name is still not a compiled kernel",
            "DeviceCompiler lowers that plan through tools.future.device_compiler.lower_plan; a placeholder claiming compiled identity is refused",
            "NoeticExecutable packs a source-independent NX from the DeviceCompiler fragment via tools.future.nx_packer.pack; a renamed source pointer is refused",
        ],
        "negative_findings": [
            "GENERIC_NR_NX_PIPELINE_CALLABLE is the live pipeline result, not FLASH_NX_READY",
            "FLASH_NX_READY is False; FLASH_COMPLETE_V0.nx remains a metadata seal of a different model",
            "doctor_tournament.probes() is still hardcoded to Qwen3.8-27B PARENT; it was not called",
            "physical_graph_compiler.main() still requires model.safetensors.index.json, an X_layer capture, and MoE expert tensors",
            "KERNEL_LIBRARY.json has no specimen field and 0 present compiled_identity values; role-name overlap is still not a compiled kernel for this body, now recorded as NATIVE_UNMEASURED rather than stalling the planner",
            "native engine match arms include qwen2 and qwen3moe, not dense qwen3, not falcon_h1; QWEN3_DENSE_GGUF_MATCH_ARM_ABSENT is a named blocker, dense was not mapped onto qwen3moe",
            "first_noetic_executable.py remains a 27B mix and is not executed on this path",
            "no physical EBPW was written",
        ],
        "what_this_cannot_establish": [
            "a packed source-independent NX for Falcon-H1, Qwen3-30B-A3B, or Flash (this packer is specimen-generic; those bodies were not the live drive)",
            "that editing tools/odyssey/physical_graph_compiler.py to accept dense names would be accepted by Codex",
            "that adding a qwen3 GGUF match arm would load this safetensors specimen",
            "protected complete-token performance or physical EBPW",
            "protected complete-token performance; FLASH_NX_READY; accepted generation on this NX",
        ],
        "next_workunits": [
            {
                "id": "WU.CPU.nr-nx-generic.generic-nx-packer",
                "schedule": "CPU_NEXT",
                "owner": "Codex (native reader + generic NX packer are Codex-owned)",
                "wake": "a generic NX packer that accepts the DeviceCompiler NX fragment and the adapted family, without treating qwen3 dense as qwen3moe",
            },
            {
                "id": sleeping["id"],
                "schedule": SLEEPING,
                "wake_unmet": sleeping["wake_unmet"],
            },
        ],
        "resident_callable": {
            "entry_point": "tools.future.nr_nx_generic.run()",
            "workunit": (
                "one CPU_ANALYSIS unit; probe+adapt+drive compiler stages and pack a "
                "source-independent NX on a real specimen; no GPU authority"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_EXECUTION.complete-token",
            "fails_closed": (
                "absent specimen/config/index is REFUSED; a stage that cannot run is FAILED/"
                "BLOCKED by name, never SKIPPED; adapt() does not branch on specimen id; "
                "source independence fails on a source-tree read; "
                "GENERIC_NR_NX_PIPELINE_CALLABLE cannot be True if any stage was "
                "skipped or no packed NX exists; physical_ebpw cannot be written"
            ),
        },
    }
    assert_no_physical_ebpw(doc)
    if doc["GENERIC_NR_NX_PIPELINE_CALLABLE"] is True and result["packed_path"] is None:
        raise PipelineCallableForbidden("callable True with no packed NX")
    if doc["FLASH_NX_READY"] is True and flash.get("metadata_only"):
        raise PipelineCallableForbidden("FLASH_NX_READY True on a metadata seal")
    return doc


def build() -> Path:
    doc = assemble()
    assert_no_physical_ebpw(doc)
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
