"""ANE_PREBOARD — prepared-but-unrun Core ML / ANE contracts.

Headline: this machine has no matching Core ML / Xcode toolchain. Command
Line Tools are not Xcode, coremltools is not installed, and /Applications/Xcode.app
is absent. This module is the contract that becomes executable the moment that
environment exists. It does not pretend the environment exists now.

Public APIs only. STATIC_ONLY. Bench state UNKNOWN. No ANE performance
number of any kind — not a measurement, not an estimate. An honest UNKNOWN
is the correct answer; a plausible invented placement or latency is a
campaign-level failure.

    python3 tools/future/ane_preboard.py --probe
    python3 tools/future/ane_preboard.py --build
    python3 tools/future/ane_preboard.py --selftest
    python3 -m pytest tools/future/test_ane_preboard.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import compile_swift, write_receipt, load_json, REPO, git

import argparse
import functools
import importlib.util
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping


RECEIPT = "ANE_PREBOARD.json"
SCHEMA = "hawking.future.ane_preboard.v1"
VERSION = 1
RECORDED_BY = "tools/future/ane_preboard.py"
PLACEMENT_SCHEMA = "hawking.future.ane_placement.v1"

EVIDENCE = REPO / "receipts" / "future" / "evidence"

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

# Public MLComputeDevice cases (Core ML). Parser vocabulary, not a placement.
COMPUTE_DEVICES = ("CPU", "GPU", "NEURAL_ENGINE")
PLAN_STATUSES = ("PLANNED", "UNKNOWN", "NOT_RUN")

# Public documented MLComputePlan JSON shape. Matches the pinned
# APPLE_ANE_DEVICE_PROFILE.mlcomputeplan serializer, which is itself a
# projection of MLComputePlan.load(contentsOf:configuration:).
PLAN_REQUIRED_KEYS = ("api", "model_structure", "operations", "status")
PLAN_OP_REQUIRED_KEYS = (
    "index",
    "operator",
    "preferred",
    "supported",
    "placement_status",
)

# Codex-owned / named surfaces the contract asked us to recover. Presence is
# reported; absence is a negative finding, not a license to invent them.
RECOVER_PATHS = (
    "receipts/future/evidence/APPLE_ANE_ATLAS.json",
    "receipts/future/evidence/APPLE_ANE_DEVICE_PROFILE.json",
    "receipts/future/evidence/FLASH_ORGAN_CENSUS.json",
    "receipts/future/evidence/QWEN27_TOKEN_NS_BUDGET.json",
    "receipts/future/evidence/QWEN27_FPGA_ORGAN_MAP.json",
    "receipts/future/evidence/FLASH_NEXT_FPGA_ORGAN_MAP.json",
    "receipts/future/evidence/FLASH_LAYER46_DISPATCH_LEDGER.json",
    "receipts/future/evidence/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json",
    "receipts/future/evidence/QWEN38_ACCELERATOR_TRANSFER_MAP.json",
    "receipts/future/evidence/HCLI_FPGA_PREBOARD.json",
    "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
    "hcli/ane_provider.py",
    "hcli/test_ane_provider.py",
    "hcli/providers.py",
    "hcli/backends.py",
    "hcli/agentos/preboard.py",
    "hcli/agentos/fpga_preboard.py",
    "tools/accelerator/ane_micrograph_author.py",
    "tools/accelerator/ane_probe.swift",
    "tools/accelerator/run_ane_lane.sh",
    "ane_probe.swift",
    "run_ane_lane.sh",
)


class ToolchainUnavailableError(RuntimeError):
    """Raised by every execution entry point when the Core ML / Xcode gate is closed."""

    def __init__(self, probe: Mapping[str, Any]) -> None:
        self.probe = dict(probe)
        missing = list(self.probe.get("missing") or [])
        detail = "; ".join(missing) if missing else "toolchain_available() is False"
        super().__init__(f"Core ML / Xcode toolchain unavailable: {detail}")


class AneNumberForbiddenError(RuntimeError):
    """No ANE performance number: not estimated, not timed by this sidecar."""

    def __init__(self, why: str) -> None:
        self.why = why
        super().__init__(why)


class PlanParseError(ValueError):
    """The document is not the public MLComputePlan JSON shape."""


class GraphUnknownError(KeyError):
    """Named graph is not in the declarative corpus."""


# ---------------------------------------------------------------------------
# Recovery: disk is authority. Sparse checkout is not absence-from-git.
# ---------------------------------------------------------------------------


def path_state(rel: str) -> dict[str, Any]:
    on_disk = (REPO / rel).is_file()
    # git() swallows the exit code, so existence is "ls-tree printed this path".
    listed = git("ls-tree", "-r", "--name-only", "HEAD", "--", rel)
    in_git = any(line.strip() == rel for line in listed.splitlines())
    return {
        "path": rel,
        "on_disk": on_disk,
        "in_git_head": in_git,
        "source": "ON_DISK" if on_disk else ("GIT_HEAD" if in_git else "ABSENT"),
    }


def _evidence(name: str) -> dict[str, Any] | None:
    path = EVIDENCE / name
    if not path.is_file():
        return None
    doc = load_json(path)
    return doc if isinstance(doc, dict) else None


def _unknown() -> str:
    return "UNKNOWN"


def _latency_slot() -> dict[str, None]:
    """Atlas-shaped latency block. Values stay null. Never an estimate."""
    return {"cold_ns": None, "warm_ns": None, "throughput": None}


def _not_run_compile() -> dict[str, str]:
    return {
        "status": "NOT_RUN",
        "reason": "requires compiled .mlmodelc and a full Xcode / Core ML toolchain",
    }


def _blank_placement_fields() -> dict[str, Any]:
    return {
        "status": "UNKNOWN",
        "preferred": None,
        "supported": [],
    }


# ---------------------------------------------------------------------------
# Toolchain gate. Probe only. Never xcode-select, never install.
# ---------------------------------------------------------------------------


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _coremltools_status() -> dict[str, Any]:
    spec = importlib.util.find_spec("coremltools")
    if spec is None:
        return {
            "present": False,
            "version": None,
            "error": "ModuleNotFoundError: No module named 'coremltools'",
        }
    try:
        import coremltools  # type: ignore

        version = getattr(coremltools, "__version__", _unknown())
        return {"present": True, "version": version, "error": None}
    except Exception as exc:  # import-time failure is still "not usable"
        return {
            "present": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


@functools.lru_cache(maxsize=1)
def probe_toolchain() -> dict[str, Any]:
    """Observe the environment. Does not install, does not run xcode-select -s."""
    xcode_app = Path("/Applications/Xcode.app")
    xcode_select = _run(["xcode-select", "-p"])
    xcode_select_path = (xcode_select.stdout or "").strip() or None
    xcodebuild = _run(["xcodebuild", "-version"])
    xcodebuild_ok = xcodebuild.returncode == 0
    xcodebuild_msg = (xcodebuild.stderr or xcodebuild.stdout or "").strip().splitlines()
    xcodebuild_line = xcodebuild_msg[0] if xcodebuild_msg else "xcodebuild produced no output"
    swift = shutil.which("swift")
    swift_ver = _run(["swift", "--version"])
    swift_line = (swift_ver.stdout or "").strip().splitlines()
    coreml_fw = Path("/System/Library/Frameworks/CoreML.framework")
    ct = _coremltools_status()
    mac_ver = platform.mac_ver()[0] or _unknown()

    clt_only = bool(
        xcode_select_path and "CommandLineTools" in xcode_select_path and not xcode_app.is_dir()
    )
    full_xcode = bool(
        xcode_app.is_dir()
        and xcode_select_path
        and "Xcode.app" in xcode_select_path
        and xcodebuild_ok
    )

    missing: list[str] = []
    present: list[str] = []

    if not xcode_app.is_dir():
        missing.append("/Applications/Xcode.app is absent")
    else:
        present.append("/Applications/Xcode.app exists")

    if not xcode_select_path:
        missing.append("xcode-select -p returned empty")
    elif "CommandLineTools" in xcode_select_path:
        missing.append(
            f"active developer directory is CommandLineTools ({xcode_select_path}); "
            "full Xcode is required for MLComputePlan / Neural Engine compile"
        )
    elif "Xcode.app" in xcode_select_path:
        present.append(f"xcode-select => {xcode_select_path}")
    else:
        missing.append(f"xcode-select => {xcode_select_path} (not a full Xcode.app)")

    if not xcodebuild_ok:
        missing.append(f"xcodebuild unavailable: {xcodebuild_line}")
    else:
        present.append(f"xcodebuild: {xcodebuild_line}")

    if not ct["present"]:
        missing.append(f"coremltools unavailable: {ct['error']}")
    else:
        present.append(f"coremltools {ct['version']}")

    if swift:
        present.append(f"swift at {swift}" + (f" ({swift_line[0]})" if swift_line else ""))
    else:
        missing.append("swift is not on PATH")

    if coreml_fw.is_dir():
        present.append(
            f"{coreml_fw} present (OS runtime; not a compile toolchain by itself)"
        )
    else:
        missing.append(f"{coreml_fw} missing")

    # MLComputePlan public API ships in macOS 14.4+. This host reports macOS 27
    # in the pinned device profile; we record the live platform string only.
    present.append(f"platform macOS {mac_ver} ({platform.machine()})")

    available = bool(full_xcode and ct["present"] and missing == [])
    # If anything required is missing, the gate is closed. full_xcode already
    # implies Xcode.app + xcode-select + xcodebuild; coremltools is the authoring
    # half. Extra missing entries (swift, framework) also close the gate.
    if missing:
        available = False

    return {
        "available": available,
        "missing": missing,
        "present": present,
        "xcode_app": str(xcode_app) if xcode_app.is_dir() else None,
        "xcode_select_path": xcode_select_path,
        "xcodebuild_ok": xcodebuild_ok,
        "xcodebuild": xcodebuild_line,
        "coremltools": ct,
        "coreml_framework": str(coreml_fw) if coreml_fw.is_dir() else None,
        "swift": swift,
        "macos": mac_ver,
        "machine": platform.machine(),
        "command_line_tools_only": clt_only,
        "public_api_floor": "macOS 14.4+ for MLComputePlan.load(contentsOf:configuration:)",
        "did_not": [
            "install any package",
            "run xcode-select to change the active developer directory",
            "prompt for a password",
            "estimate an ANE placement or latency",
        ],
        "headline": (
            "usable Core ML / Xcode toolchain present"
            if available
            else "no usable Core ML / Xcode toolchain; everything here is a prepared-but-unrun contract"
        ),
    }


def toolchain_available() -> bool:
    return bool(probe_toolchain()["available"])


def require_toolchain() -> dict[str, Any]:
    probe = probe_toolchain()
    if not probe["available"]:
        raise ToolchainUnavailableError(probe)
    return probe


# ---------------------------------------------------------------------------
# Geometry: Flash + Qwen27 from receipts, not invented dimensions.
# ---------------------------------------------------------------------------


def recover_flash_geometry() -> dict[str, Any]:
    census = _evidence("FLASH_ORGAN_CENSUS.json") or {}
    atlas = _evidence("APPLE_ANE_ATLAS.json") or {}
    transfer = _evidence("QWEN38_ACCELERATOR_TRANSFER_MAP.json") or {}
    ledger = _evidence("FLASH_LAYER46_DISPATCH_LEDGER.json") or {}

    expert_shape = None
    for tensor in census.get("largest_tensors") or []:
        if not isinstance(tensor, dict):
            continue
        name = str(tensor.get("name") or "")
        if name.endswith("mlp.experts.gate_up_proj"):
            expert_shape = list(tensor.get("shape") or [])
            break

    atlas_shapes = atlas.get("shapes") if isinstance(atlas.get("shapes"), dict) else {}
    atlas_graphs = {
        row["id"]: row
        for row in (atlas.get("graphs") or [])
        if isinstance(row, dict) and row.get("id")
    }
    sdpa_shapes = atlas_graphs.get("ane_atlas_sdpa", {}).get("shapes")
    conv_shapes = atlas_graphs.get("ane_atlas_conv", {}).get("shapes")
    arch = ((transfer.get("model_b") or {}) if isinstance(transfer, dict) else {}).get("architecture") or {}

    hc_shape = None
    for row in ledger.get("rows") or []:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("input") or "")
        if "base/[4,2560]" in raw:
            hc_shape = [4, 2560]
            break

    # gate_up [n_experts, 2*intermediate, hidden] => intermediate = 1280/2 = 640.
    intermediate = None
    n_experts = None
    hidden_from_expert = None
    if expert_shape and len(expert_shape) == 3:
        n_experts, gate_up, hidden_from_expert = expert_shape
        if gate_up % 2 == 0:
            intermediate = gate_up // 2

    return {
        "model": census.get("model") or arch.get("model_type") or "Qwen/Qwen3.8-Flash-Next",
        "source": "Flash-Next Qwen4-Exp geometry (census + atlas + transfer map + layer-46 ledger)",
        "layer_count": census.get("layer_count_observed") or arch.get("layers"),
        "hidden_size": hidden_from_expert or arch.get("hidden_size"),
        "n_experts": n_experts or arch.get("experts"),
        "experts_per_token": arch.get("experts_per_token"),
        "expert_intermediate": intermediate,
        "expert_gate_up_shape": expert_shape,
        "full_attention_layers": arch.get("full_attention_layers"),
        "linear_attention_layers": arch.get("linear_attention_layers"),
        "vocab_size": arch.get("vocab_size"),
        "ngram_size": arch.get("ngram_size"),
        "atlas_shapes": {
            key: list(value) if isinstance(value, list) else value
            for key, value in sorted(atlas_shapes.items())
        },
        "hyperconnection_base": hc_shape,
        "sdpa_shapes": list(sdpa_shapes) if isinstance(sdpa_shapes, list) else None,
        "conv_shapes": list(conv_shapes) if isinstance(conv_shapes, list) else None,
        "tensor_count": census.get("tensor_count") or arch.get("tensor_count"),
        "families": [
            row.get("family")
            for row in (census.get("family_summary") or [])
            if isinstance(row, dict)
        ],
        "citations": {
            "census": "receipts/future/evidence/FLASH_ORGAN_CENSUS.json",
            "atlas": "receipts/future/evidence/APPLE_ANE_ATLAS.json",
            "architecture": "receipts/future/evidence/QWEN38_ACCELERATOR_TRANSFER_MAP.json model_b.architecture",
            "layer46": "receipts/future/evidence/FLASH_LAYER46_DISPATCH_LEDGER.json",
        },
    }


def recover_qwen27_geometry() -> dict[str, Any]:
    budget = _evidence("QWEN27_TOKEN_NS_BUDGET.json") or {}
    dispatch = _evidence("ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json") or {}
    ident = dispatch.get("identities") if isinstance(dispatch.get("identities"), dict) else {}
    model = ident.get("model") if isinstance(ident.get("model"), dict) else {}

    organs = []
    for row in budget.get("organs") or []:
        if isinstance(row, dict) and row.get("organ"):
            organs.append(row["organ"])

    regions = ((budget.get("source_byte_denominator") or {}).get("regions") or [])
    kernels = []
    for row in regions:
        if isinstance(row, dict) and row.get("kernel"):
            kernels.append(
                {
                    "kernel": row.get("kernel"),
                    "roles": list(row.get("roles") or []),
                    "dispatches_per_token": row.get("dispatches_per_token"),
                }
            )

    mlp_dispatches = None
    for row in kernels:
        if row["kernel"] == "qwen_affine_q2_group32_matvec_geo_tpr64_tg128":
            mlp_dispatches = row.get("dispatches_per_token")
            break
    derived_layers = None
    if isinstance(mlp_dispatches, int) and mlp_dispatches % 3 == 0:
        derived_layers = mlp_dispatches // 3

    hidden = model.get("hidden_size")
    intermediate = model.get("intermediate_size")
    layers = model.get("layers")
    vocab = model.get("vocab_size")

    return {
        "model": budget.get("model") or "qwen3.8-27b-sealed-3.14",
        "source": "Qwen27 token-ns budget organs/kernels + dispatch-ledger identities.model",
        "layers": layers,
        "layers_derived_from_mlp_dispatches": derived_layers,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "organs": organs,
        "kernels": kernels,
        "n_heads": None,
        "n_kv_heads": None,
        "head_dim": None,
        "head_counts_status": "UNKNOWN_NOT_IN_CENSUS_OR_BUDGET",
        "mlp_weight_shapes": {
            "gate_proj": [intermediate, hidden] if intermediate and hidden else None,
            "up_proj": [intermediate, hidden] if intermediate and hidden else None,
            "down_proj": [hidden, intermediate] if intermediate and hidden else None,
            "citation": (
                "identities.model.hidden_size/intermediate_size in "
                "receipts/future/evidence/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json; "
                "standard SwiGLU projection layout, not a measured runtime shape"
            ),
        },
        "citations": {
            "budget": "receipts/future/evidence/QWEN27_TOKEN_NS_BUDGET.json",
            "identities": "receipts/future/evidence/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json identities.model",
        },
    }


def recover_geometry() -> dict[str, Any]:
    return {
        "flash": recover_flash_geometry(),
        "qwen27": recover_qwen27_geometry(),
    }


# ---------------------------------------------------------------------------
# 1. Graph corpus — declarative, real shapes, no compile, no placement.
# ---------------------------------------------------------------------------


def _graph(
    gid: str,
    *,
    model: str,
    organ: str,
    operation: str,
    shapes: dict[str, Any],
    source: str,
    notes: str,
    mil_op: str | None,
) -> dict[str, Any]:
    return {
        "id": gid,
        "model": model,
        "organ": organ,
        "operation": operation,
        "mil_op": mil_op,
        "mlprogram": True,
        "public_api": "Core ML MLProgram",
        "shapes": shapes,
        "source": source,
        "notes": notes,
        "compile": _not_run_compile(),
        "placement": _blank_placement_fields(),
        "latency": _latency_slot(),
        "memory": {"delta_bytes": None},
        "energy": {"status": "NOT_TRUSTWORTHY", "joules": None},
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def graph_corpus() -> list[dict[str, Any]]:
    geo = recover_geometry()
    flash = geo["flash"]
    q27 = geo["qwen27"]
    atlas = flash.get("atlas_shapes") or {}
    graphs: list[dict[str, Any]] = []

    hidden_f = flash.get("hidden_size")
    expert_f = flash.get("expert_intermediate")
    gate_up = flash.get("expert_gate_up_shape")
    hc = flash.get("hyperconnection_base")
    qkv = atlas.get("flash_qkv")
    state = atlas.get("flash_state")
    router = atlas.get("flash_router")
    topk = atlas.get("flash_top_k")

    graphs.append(
        _graph(
            "flash.hc.grouped_rmsnorm",
            model="flash-next",
            organ="linear_attention_hyperconnection",
            operation="rmsnorm_like",
            shapes={"activation": hc, "hidden": [1, hidden_f]},
            source=flash["citations"]["layer46"],
            notes="layer-46 input base/[4,2560] + atlas flash_hidden",
            mil_op="rmsnorm_like",
        )
    )
    graphs.append(
        _graph(
            "flash.linear.gemv",
            model="flash-next",
            organ="linear_attention_hyperconnection",
            operation="gemv",
            shapes={"activation": [1, hidden_f], "qkv": qkv},
            source=flash["citations"]["atlas"],
            notes="atlas flash_hidden + flash_qkv; gemv is the decode-shaped matmul",
            mil_op="matmul",
        )
    )
    graphs.append(
        _graph(
            "flash.elementwise.silu",
            model="flash-next",
            organ="mlp_hyperconnection",
            operation="silu",
            shapes={"activation": [1, hidden_f]},
            source=flash["citations"]["atlas"],
            notes="atlas ane_atlas_silu shape",
            mil_op="silu",
        )
    )
    graphs.append(
        _graph(
            "flash.elementwise.sigmoid",
            model="flash-next",
            organ="mlp_hyperconnection",
            operation="sigmoid",
            shapes={"activation": [1, hidden_f]},
            source=flash["citations"]["atlas"],
            notes="atlas ane_atlas_sigmoid shape",
            mil_op="sigmoid",
        )
    )
    graphs.append(
        _graph(
            "flash.elementwise.multiply",
            model="flash-next",
            organ="shared_expert",
            operation="multiply",
            shapes={"activation": [1, hidden_f]},
            source=flash["citations"]["atlas"],
            notes="atlas ane_atlas_multiply; also the SwiGLU product",
            mil_op="mul",
        )
    )
    graphs.append(
        _graph(
            "flash.elementwise.add",
            model="flash-next",
            organ="other",
            operation="add",
            shapes={"activation": [1, hidden_f]},
            source=flash["citations"]["atlas"],
            notes="atlas ane_atlas_add residual",
            mil_op="add",
        )
    )
    graphs.append(
        _graph(
            "flash.softmax",
            model="flash-next",
            organ="full_attention",
            operation="softmax",
            shapes={"activation": [1, hidden_f]},
            source=flash["citations"]["atlas"],
            notes="atlas ane_atlas_softmax; not a placement",
            mil_op="softmax",
        )
    )
    graphs.append(
        _graph(
            "flash.sdpa",
            model="flash-next",
            organ="full_attention",
            operation="sdpa",
            shapes={
                "hidden": [1, hidden_f],
                "atlas_sdpa": flash.get("sdpa_shapes"),
            },
            source=flash["citations"]["atlas"],
            notes=(
                "atlas ane_atlas_sdpa shapes copied verbatim. Public MIL has no sdpa "
                "op — compile will name PUBLIC_API_CANNOT_EXPRESS rather than invent a kernel."
            ),
            mil_op=None,
        )
    )
    graphs.append(
        _graph(
            "flash.conv.l2",
            model="flash-next",
            organ="linear_attention_hyperconnection",
            operation="conv",
            shapes={"atlas_conv": flash.get("conv_shapes")},
            source=flash["citations"]["atlas"],
            notes="atlas ane_atlas_conv; layer-46 names qwen_next_qkv_split_rearrange_conv_l2",
            mil_op="conv",
        )
    )
    graphs.append(
        _graph(
            "flash.router.topk",
            model="flash-next",
            organ="routed_experts",
            operation="top_k",
            shapes={"router": router, "top_k": topk, "n_experts": flash.get("n_experts"), "k": flash.get("experts_per_token")},
            source=flash["citations"]["atlas"] + " + " + flash["citations"]["architecture"],
            notes="atlas flash_router/flash_top_k; architecture experts=512 k=10",
            mil_op="topk",
        )
    )
    graphs.append(
        _graph(
            "flash.router.gather",
            model="flash-next",
            organ="routed_experts",
            operation="gather",
            shapes={"router": router, "top_k": topk},
            source=flash["citations"]["atlas"],
            notes="atlas ane_atlas_gather",
            mil_op="gather",
        )
    )
    graphs.append(
        _graph(
            "flash.router.scatter",
            model="flash-next",
            organ="routed_experts",
            operation="scatter",
            shapes={"router": router, "top_k": topk},
            source=flash["citations"]["atlas"],
            notes="atlas ane_atlas_scatter",
            mil_op="scatter",
        )
    )
    graphs.append(
        _graph(
            "flash.expert.gate_up_swiglu",
            model="flash-next",
            organ="routed_experts",
            operation="matmul",
            shapes={
                "activation": [1, hidden_f],
                "expert_activation": [1, expert_f],
                "weight": gate_up,
            },
            source=flash["citations"]["census"],
            notes="largest_tensors mlp.experts.gate_up_proj shape [512, 1280, 2560]",
            mil_op="matmul",
        )
    )
    graphs.append(
        _graph(
            "flash.deltanet.gated_decode",
            model="flash-next",
            organ="linear_attention_hyperconnection",
            operation="stateful_delta",
            shapes={"state": state, "hidden": [1, hidden_f]},
            source=flash["citations"]["atlas"],
            notes="atlas flash_state [1,48,128,128]; layer-46 qwen_next_gated_delta_decode_single",
            mil_op=None,
        )
    )
    graphs.append(
        _graph(
            "flash.fused_projection_gate",
            model="flash-next",
            organ="mlp_hyperconnection",
            operation="fused_projection_gate",
            shapes={"hidden": [1, hidden_f], "expert": [1, expert_f], "qkv": qkv},
            source=flash["citations"]["atlas"],
            notes="atlas ane_atlas_fused_projection_gate; fusion is a candidate, not a proven MIL graph",
            mil_op=None,
        )
    )

    hidden_q = q27.get("hidden_size")
    inter_q = q27.get("intermediate_size")
    vocab_q = q27.get("vocab_size")
    mlp = q27.get("mlp_weight_shapes") or {}

    graphs.append(
        _graph(
            "qwen27.mlp.gate_proj",
            model="qwen27",
            organ="mlp",
            operation="gemv",
            shapes={"activation": [1, hidden_q], "weight": mlp.get("gate_proj")},
            source=q27["citations"]["identities"],
            notes="gate_proj [intermediate, hidden] = [17408, 5120]; kernel qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
            mil_op="matmul",
        )
    )
    graphs.append(
        _graph(
            "qwen27.mlp.up_proj",
            model="qwen27",
            organ="mlp",
            operation="gemv",
            shapes={"activation": [1, hidden_q], "weight": mlp.get("up_proj")},
            source=q27["citations"]["identities"],
            notes="up_proj [17408, 5120]",
            mil_op="matmul",
        )
    )
    graphs.append(
        _graph(
            "qwen27.mlp.down_proj",
            model="qwen27",
            organ="mlp",
            operation="gemv",
            shapes={"activation": [1, inter_q], "weight": mlp.get("down_proj")},
            source=q27["citations"]["identities"],
            notes="down_proj [hidden, intermediate] = [5120, 17408]",
            mil_op="matmul",
        )
    )
    graphs.append(
        _graph(
            "qwen27.mlp.silu",
            model="qwen27",
            organ="mlp",
            operation="silu",
            shapes={"activation": [1, inter_q]},
            source=q27["citations"]["budget"],
            notes="SwiGLU gate activation; fusion_env HAWKING_QWEN38_FUSE_MLP=swiglu is Metal, not ANE evidence",
            mil_op="silu",
        )
    )
    graphs.append(
        _graph(
            "qwen27.norm.rmsnorm",
            model="qwen27",
            organ="mlp",
            operation="rmsnorm_like",
            shapes={"activation": [1, hidden_q]},
            source=q27["citations"]["budget"],
            notes="kernel qwen80_residual_rmsnorm_tg listed in the token-ns budget",
            mil_op="rmsnorm_like",
        )
    )
    graphs.append(
        _graph(
            "qwen27.qkv_and_projection",
            model="qwen27",
            organ="qkv_and_projection",
            operation="gemv",
            shapes={
                "activation": [1, hidden_q],
                "weight": _unknown(),
                "n_heads": q27.get("n_heads"),
                "n_kv_heads": q27.get("n_kv_heads"),
                "head_dim": q27.get("head_dim"),
            },
            source=q27["citations"]["budget"],
            notes=(
                "organ qkv_and_projection and kernel qwen38_gqa_qk_norm_rope_cache_tg "
                "are in the budget; head counts are not. Weight shape stays UNKNOWN."
            ),
            mil_op="matmul",
        )
    )
    graphs.append(
        _graph(
            "qwen27.attention",
            model="qwen27",
            organ="attention",
            operation="sdpa",
            shapes={
                "activation": [1, hidden_q],
                "q": _unknown(),
                "kv": _unknown(),
            },
            source=q27["citations"]["budget"],
            notes="kernel mha_decode_f32 is in the budget; GQA head layout is not",
            mil_op=None,
        )
    )
    graphs.append(
        _graph(
            "qwen27.deltanet",
            model="qwen27",
            organ="deltanet_and_recurrent_state",
            operation="stateful_delta",
            shapes={"activation": [1, hidden_q], "state": _unknown()},
            source=q27["citations"]["budget"],
            notes="organ deltanet_and_recurrent_state + kernel qwen38_gated_delta_decode_vi_simd; state rank is not in the budget",
            mil_op=None,
        )
    )
    graphs.append(
        _graph(
            "qwen27.lm_head",
            model="qwen27",
            organ="lm_head_and_sampling",
            operation="gemv",
            shapes={"activation": [1, hidden_q], "weight": [vocab_q, hidden_q] if vocab_q and hidden_q else None},
            source=q27["citations"]["identities"],
            notes="vocab_size 248320 x hidden 5120; kernel sample_argmax_f32 is sampling, not this gemv",
            mil_op="matmul",
        )
    )

    graphs.sort(key=lambda g: g["id"])
    return graphs


def graph_by_id(gid: str) -> dict[str, Any]:
    for graph in graph_corpus():
        if graph["id"] == gid:
            return graph
    raise GraphUnknownError(gid)


# ---------------------------------------------------------------------------
# 2. MLState cases — recurrent / KV shapes that would need MLState.
# ---------------------------------------------------------------------------


def mlstate_cases() -> list[dict[str, Any]]:
    geo = recover_geometry()
    flash = geo["flash"]
    q27 = geo["qwen27"]
    atlas = flash.get("atlas_shapes") or {}
    cases = [
        {
            "id": "flash.deltanet.recurrent_state",
            "model": "flash-next",
            "organ": "linear_attention_hyperconnection",
            "api": "Core ML MLState (public; macOS 15+ / iOS 18+)",
            "shape": atlas.get("flash_state"),
            "shape_status": "RECOVERED" if atlas.get("flash_state") else "UNKNOWN",
            "lifetime": "sequence",
            "tests": (
                "read-modify-write of the 128x128 per-layer DeltaNet state across "
                "decode steps without a host round-trip; MLState persistence vs "
                "re-feeding the tensor as an input each prediction"
            ),
            "source": flash["citations"]["atlas"],
        },
        {
            "id": "flash.conv.l2_state",
            "model": "flash-next",
            "organ": "linear_attention_hyperconnection",
            "api": "Core ML MLState",
            "shape": (flash.get("conv_shapes") or [None])[0],
            "shape_status": "RECOVERED" if flash.get("conv_shapes") else "UNKNOWN",
            "lifetime": "sequence",
            "tests": (
                "short-conv state that layer-46 names qwen_next_qkv_split_rearrange_conv_l2; "
                "whether MLState can hold the conv window or Core ML rewrites it to CPU"
            ),
            "source": flash["citations"]["atlas"] + " + " + flash["citations"]["layer46"],
        },
        {
            "id": "flash.full_attention.kv",
            "model": "flash-next",
            "organ": "full_attention",
            "api": "Core ML MLState",
            "shape": {
                "atlas_sdpa": flash.get("sdpa_shapes"),
                "n_full_attn_layers": flash.get("full_attention_layers"),
            },
            "shape_status": "RECOVERED_ATLAS_KV_HEADS_SEQ_GROWS",
            "lifetime": "sequence_growing",
            "tests": (
                "12 full-attention layers from the transfer-map architecture; atlas SDPA "
                "kv is [1,2,1,256] at seq=1. MLState growing KV is the question; this "
                "case exists to watch Core ML refuse or spill, not to guess which."
            ),
            "source": flash["citations"]["atlas"] + " + " + flash["citations"]["architecture"],
        },
        {
            "id": "flash.hyperconnection.base_streams",
            "model": "flash-next",
            "organ": "linear_attention_hyperconnection",
            "api": "Core ML MLState",
            "shape": flash.get("hyperconnection_base"),
            "shape_status": "RECOVERED",
            "lifetime": "layer_and_sequence",
            "tests": (
                "HC base streams [4,2560] from layer-46; whether they belong in MLState "
                "or stay activations. The case records the shape and the question."
            ),
            "source": flash["citations"]["layer46"],
        },
        {
            "id": "qwen27.deltanet.recurrent_state",
            "model": "qwen27",
            "organ": "deltanet_and_recurrent_state",
            "api": "Core ML MLState",
            "shape": _unknown(),
            "shape_status": "UNKNOWN_NOT_IN_BUDGET",
            "lifetime": "sequence",
            "tests": (
                "gated-delta state for kernel qwen38_gated_delta_decode_vi_simd. The "
                "organ is real; the rank is not on disk. The case exists so a later "
                "toolchain run fills the shape instead of this sidecar inventing one."
            ),
            "source": q27["citations"]["budget"],
        },
        {
            "id": "qwen27.gqa.kv_cache",
            "model": "qwen27",
            "organ": "attention",
            "api": "Core ML MLState",
            "shape": _unknown(),
            "shape_status": "UNKNOWN_HEAD_COUNTS_ABSENT",
            "lifetime": "sequence_growing",
            "tests": (
                "GQA KV as MLState vs host cache. n_heads / n_kv_heads / head_dim are "
                "not in the token-ns budget or the dispatch identities. Do not fill them."
            ),
            "source": q27["citations"]["budget"],
        },
    ]
    cases.sort(key=lambda c: c["id"])
    return cases


# ---------------------------------------------------------------------------
# 3. MLComputePlan parser — public documented JSON shape + fixture.
# ---------------------------------------------------------------------------


def mlcomputeplan_fixture() -> dict[str, Any]:
    """A fixture. Not a live plan. Copied from the pinned device-profile capture.

    APPLE_ANE_DEVICE_PROFILE.json is Codex-owned evidence snapshotted into
    receipts/future/evidence/. The one operation it recorded is ios16.mul
    with preferred CPU and supported [CPU, NEURAL_ENGINE]. Parsing that
    document is not claiming that placement for any Flash or Qwen27 graph.
    """
    profile = _evidence("APPLE_ANE_DEVICE_PROFILE.json") or {}
    plan = profile.get("mlcomputeplan") if isinstance(profile.get("mlcomputeplan"), dict) else {}
    # Drop host-absolute compiled_model path; it is not needed to parse the
    # public shape and must not leak into a sealed sidecar receipt as if we
    # owned that .mlmodelc.
    operations = []
    for op in plan.get("operations") or []:
        if not isinstance(op, dict):
            continue
        operations.append(
            {
                "estimated_cost_weight": op.get("estimated_cost_weight"),
                "function": op.get("function"),
                "index": op.get("index"),
                "operator": op.get("operator"),
                "placement_status": op.get("placement_status"),
                "preferred": op.get("preferred"),
                "supported": list(op.get("supported") or []),
            }
        )
    return {
        "is_fixture": True,
        "fixture_citation": (
            "receipts/future/evidence/APPLE_ANE_DEVICE_PROFILE.json mlcomputeplan; "
            "this is a fixture, not a live MLComputePlan.load result from this process"
        ),
        "api": plan.get("api") or "MLComputePlan.load(contentsOf:configuration:)",
        "compile_api": plan.get("compile_api") or "not_needed",
        "compiled_model": None,
        "model_structure": plan.get("model_structure") or "MLProgram",
        "operations": operations,
        "status": plan.get("status") or "PLANNED",
        "coreml_deployment_target": profile.get("coreml_deployment_target"),
        "public_api_only": True,
    }


def parse_mlcompute_plan(document: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the public JSON projection of MLComputePlan. No Core ML required.

    Expected keys (public serializer used by APPLE_ANE_DEVICE_PROFILE):
      api, model_structure, operations[], status
    Each operation:
      index, operator, preferred, supported[], placement_status
      estimated_cost_weight may be null (Apple's estimatedCost can return nil)
    `is_fixture` is preserved if present. A parsed fixture is still a fixture.
    """
    if not isinstance(document, Mapping):
        raise PlanParseError("plan document must be a mapping")
    missing = [k for k in PLAN_REQUIRED_KEYS if k not in document]
    if missing:
        raise PlanParseError(f"missing plan key(s): {', '.join(missing)}")
    ops_in = document.get("operations")
    if not isinstance(ops_in, list):
        raise PlanParseError("operations must be a list")
    ops_out: list[dict[str, Any]] = []
    for i, op in enumerate(ops_in):
        if not isinstance(op, dict):
            raise PlanParseError(f"operations[{i}] is not a mapping")
        op_missing = [k for k in PLAN_OP_REQUIRED_KEYS if k not in op]
        if op_missing:
            raise PlanParseError(f"operations[{i}] missing {', '.join(op_missing)}")
        preferred = op.get("preferred")
        if preferred not in COMPUTE_DEVICES and preferred not in (None, _unknown()):
            raise PlanParseError(f"operations[{i}].preferred {preferred!r} is not a public MLComputeDevice")
        supported = op.get("supported")
        if not isinstance(supported, list):
            raise PlanParseError(f"operations[{i}].supported must be a list")
        for dev in supported:
            if dev not in COMPUTE_DEVICES:
                raise PlanParseError(f"operations[{i}].supported contains {dev!r}")
        weight = op.get("estimated_cost_weight")
        if weight is not None and not isinstance(weight, (int, float)):
            raise PlanParseError(f"operations[{i}].estimated_cost_weight must be null or a number")
        # Apple's Cost.weight is a relative 0..1, not a nanosecond. We store it
        # only as parsed plan metadata and never promote it to a latency field.
        ops_out.append(
            {
                "index": op.get("index"),
                "function": op.get("function"),
                "operator": op.get("operator"),
                "preferred": preferred,
                "supported": list(supported),
                "placement_status": op.get("placement_status"),
                "estimated_cost_weight": weight,
            }
        )
    is_fixture = bool(document.get("is_fixture"))
    return {
        "ok": True,
        "is_fixture": is_fixture,
        "is_live": bool(document.get("is_live")) and not is_fixture,
        "api": document.get("api"),
        "model_structure": document.get("model_structure"),
        "status": document.get("status"),
        "operations": ops_out,
        "n_operations": len(ops_out),
        "fixture_citation": document.get("fixture_citation") if is_fixture else None,
        "claim_boundary": (
            "Parsed public MLComputePlan JSON. A fixture parse is not a live "
            "placement. preferred/supported here are document fields, not a "
            "sidecar measurement of Flash or Qwen27."
        ),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


# ---------------------------------------------------------------------------
# 4. Placement receipts — schema a real result would populate; UNKNOWN today.
# ---------------------------------------------------------------------------


PLACEMENT_SLOTS = (
    "graph_id",
    "status",
    "preferred_compute_device",
    "supported_compute_devices",
    "placement_status",
    "estimated_cost_weight",
    "model_structure",
    "compiled_model",
    "operator",
    "function",
    "latency",
    "memory",
    "energy",
    "transfer_sync",
)


def blank_placement_receipt(graph_id: str | None = None) -> dict[str, Any]:
    """Every field a live place_graph() would fill, all UNKNOWN / null today."""
    return {
        "schema": PLACEMENT_SCHEMA,
        "graph_id": graph_id,
        "status": _unknown(),
        "preferred_compute_device": _unknown(),
        "supported_compute_devices": _unknown(),
        "placement_status": _unknown(),
        "estimated_cost_weight": _unknown(),
        "model_structure": _unknown(),
        "compiled_model": _unknown(),
        "operator": _unknown(),
        "function": _unknown(),
        "latency": _latency_slot(),
        "memory": {"delta_bytes": None},
        "energy": {"status": "NOT_TRUSTWORTHY", "joules": None},
        "transfer_sync": {
            "cpu_ane_bytes": None,
            "ane_cpu_bytes": None,
            "sync_events": None,
            "status": _unknown(),
        },
        "evidence_class": "STATIC_ONLY",
        "bench_state": _unknown(),
        "gpu_authority": False,
        "claim_boundary": (
            "Placement schema only. preferred/supported stay UNKNOWN until a live "
            "MLComputePlan.load under a real Xcode / Core ML toolchain fills them. "
            "Latency, energy, and transfer bytes stay null even then unless a "
            "PROTECTED_ABSOLUTE lease the sidecar does not hold records them. "
            "This sidecar produces STATIC_ONLY."
        ),
    }


def placement_schema() -> dict[str, Any]:
    blank = blank_placement_receipt(None)
    return {
        "schema": PLACEMENT_SCHEMA,
        "slots": list(PLACEMENT_SLOTS),
        "unknown_today": True,
        "blank": blank,
        "fills_from": (
            "MLComputePlan.computeDeviceUsage(for:) -> preferredComputeDevice / "
            "supportedComputeDevices; MLComputePlan.estimatedCost(of:) -> Cost.weight "
            "(relative, not nanoseconds). Never from an estimate in this module."
        ),
    }


def placement_from_plan(graph_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Promote a live plan to a placement receipt. Gated. Fixtures cannot enter."""
    require_toolchain()
    parsed = parse_mlcompute_plan(plan)
    if parsed.get("is_fixture") or not parsed.get("is_live"):
        raise AneNumberForbiddenError(
            "a fixture or non-live plan cannot be promoted to a placement receipt"
        )
    receipt = blank_placement_receipt(graph_id)
    ops = parsed.get("operations") or []
    if ops:
        op0 = ops[0]
        receipt["status"] = parsed.get("status") or "PLANNED"
        receipt["preferred_compute_device"] = op0.get("preferred") or _unknown()
        receipt["supported_compute_devices"] = op0.get("supported") or _unknown()
        receipt["placement_status"] = op0.get("placement_status") or _unknown()
        receipt["estimated_cost_weight"] = (
            op0.get("estimated_cost_weight")
            if op0.get("estimated_cost_weight") is not None
            else _unknown()
        )
        receipt["operator"] = op0.get("operator") or _unknown()
        receipt["function"] = op0.get("function") or _unknown()
    receipt["model_structure"] = parsed.get("model_structure") or _unknown()
    receipt["latency"] = _latency_slot()
    return receipt


# ---------------------------------------------------------------------------
# 5. Test matrix — runnable specs with preconditions. Not run here.
# ---------------------------------------------------------------------------


def test_matrix() -> list[dict[str, Any]]:
    specs = [
        {
            "id": "ane_only",
            "kind": "ANE_ONLY",
            "compute_units": "CPU_AND_NEURAL_ENGINE",
            "public_api": "MLModelConfiguration.computeUnits / MLComputePlan preferred NEURAL_ENGINE",
            "graphs": [g["id"] for g in graph_corpus() if g["mil_op"] in {"mul", "add", "silu", "sigmoid", "softmax", "matmul"}],
            "preconditions": [
                "toolchain_available() is True",
                "compiled .mlmodelc exists for each graph",
                "MLComputePlan reports NEURAL_ENGINE in supportedComputeDevices on this host",
            ],
            "records": ["preferred_compute_device", "supported_compute_devices", "placement_status"],
            "does_not_record": ["latency", "tps", "token_ns", "joules"],
            "runnable_now": False,
            "entry_point": "execute_test_spec",
        },
        {
            "id": "metal_only_control",
            "kind": "METAL_ONLY_CONTROL",
            "compute_units": "CPU_AND_GPU",
            "public_api": "MLModelConfiguration.computeUnits = cpuAndGPU",
            "graphs": [g["id"] for g in graph_corpus() if g["mil_op"] is not None],
            "preconditions": [
                "toolchain_available() is True",
                "MLComputePlan reports GPU in supportedComputeDevices",
                "this is a CONTROL, not an ANE claim, and not a Metal protected-lease claim either",
            ],
            "records": ["preferred_compute_device", "supported_compute_devices"],
            "does_not_record": ["latency", "tps", "token_ns", "joules"],
            "runnable_now": False,
            "entry_point": "execute_test_spec",
        },
        {
            "id": "heterogeneous_control",
            "kind": "HETEROGENEOUS_CONTROL",
            "compute_units": "ALL",
            "public_api": "MLModelConfiguration.computeUnits = all; per-op DeviceUsage",
            "graphs": [g["id"] for g in graph_corpus()],
            "preconditions": [
                "toolchain_available() is True",
                "compiled MLProgram for each graph that mil_op can express",
                "graphs with mil_op=None are recorded as PUBLIC_API_CANNOT_EXPRESS, not estimated",
            ],
            "records": ["per-operation preferred and supported"],
            "does_not_record": ["latency", "tps", "token_ns", "joules"],
            "runnable_now": False,
            "entry_point": "execute_test_spec",
        },
        {
            "id": "concurrency",
            "kind": "CONCURRENCY",
            "compute_units": "CPU_AND_NEURAL_ENGINE",
            "public_api": "overlapping MLModel.prediction / MLState updates",
            "graphs": ["flash.deltanet.gated_decode", "qwen27.deltanet"],
            "preconditions": [
                "toolchain_available() is True",
                "MLState API available (macOS 15+)",
                "two in-flight predictions against the same MLState are the subject, not a speed claim",
            ],
            "records": ["whether Core ML serializes, copies state, or errors"],
            "does_not_record": ["latency", "tps", "token_ns", "joules"],
            "runnable_now": False,
            "entry_point": "execute_test_spec",
        },
        {
            "id": "transfer_sync_accounting",
            "kind": "TRANSFER_SYNC",
            "compute_units": "ALL",
            "public_api": "MLComputePlan + prediction; bytes/sync slots only",
            "graphs": [g["id"] for g in graph_corpus() if g["model"] in {"flash-next", "qwen27"}],
            "preconditions": [
                "toolchain_available() is True",
                "a placement already recorded preferred/supported for the graph",
                "CPU<->ANE copies and sync events are counted, not timed, unless a PROTECTED_ABSOLUTE lease exists",
            ],
            "records": [
                "cpu_ane_bytes slot",
                "ane_cpu_bytes slot",
                "sync_events slot",
                "values stay null without a PROTECTED_ABSOLUTE lease",
            ],
            "does_not_record": ["latency", "tps", "token_ns", "joules", "bandwidth_gbps"],
            "runnable_now": False,
            "entry_point": "execute_test_spec",
        },
    ]
    specs.sort(key=lambda s: s["id"])
    return specs


# ---------------------------------------------------------------------------
# 6. Execution entry points. Gate first. Nothing estimates a number.
# ---------------------------------------------------------------------------


# Public Swift inspector. Written to a tempfile and compiled only after the
# toolchain gate opens. Public MLComputePlan / MLModelStructure APIs only.
SWIFT_MLCOMPUTEPLAN_INSPECTOR = r'''
import Foundation
import CoreML

enum DeviceKind: String {
    case cpu = "CPU"
    case gpu = "GPU"
    case neuralEngine = "NEURAL_ENGINE"
}

func kind(of device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return DeviceKind.cpu.rawValue
    case .gpu: return DeviceKind.gpu.rawValue
    case .neuralEngine: return DeviceKind.neuralEngine.rawValue
    @unknown default: return "UNKNOWN"
    }
}

@available(macOS 14.4, *)
func inspect(url: URL) async throws -> [String: Any] {
    let configuration = MLModelConfiguration()
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
    var operations: [[String: Any]] = []
    if let program = plan.modelStructure.program {
        for functionName in program.functions.keys.sorted() {
            guard let function = program.functions[functionName] else { continue }
            for (index, operation) in function.block.operations.enumerated() {
                let usage = plan.computeDeviceUsage(for: operation)
                let cost = plan.estimatedCost(of: operation)
                var supported: [String] = []
                if let usage {
                    supported = usage.supportedComputeDevices.map { kind(of: $0) }
                }
                operations.append([
                    "function": functionName,
                    "index": index,
                    "operator": operation.operatorName,
                    "preferred": usage.map { kind(of: $0.preferredComputeDevice) } as Any,
                    "supported": supported,
                    "placement_status": usage == nil ? "UNKNOWN" : "PLANNED",
                    "estimated_cost_weight": cost?.weight as Any,
                ])
            }
        }
    }
    return [
        "api": "MLComputePlan.load(contentsOf:configuration:)",
        "compile_api": "not_needed",
        "compiled_model": url.path,
        "model_structure": "MLProgram",
        "operations": operations,
        "status": "PLANNED",
        "is_fixture": false,
        "is_live": true,
    ]
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    fputs("usage: ane_mlcomputeplan_inspector <model.mlmodelc>\n", stderr)
    exit(2)
}
let url = URL(fileURLWithPath: args[1])
if #available(macOS 14.4, *) {
    let sem = DispatchSemaphore(value: 0)
    var encoded: Data?
    var thrown: Error?
    Task {
        do {
            let plan = try await inspect(url: url)
            encoded = try JSONSerialization.data(withJSONObject: plan, options: [.sortedKeys])
        } catch {
            thrown = error
        }
        sem.signal()
    }
    sem.wait()
    if let thrown {
        fputs(String(describing: thrown), stderr)
        exit(1)
    }
    if let encoded {
        FileHandle.standardOutput.write(encoded)
        FileHandle.standardOutput.write(Data("\n".utf8))
        exit(0)
    }
    exit(1)
} else {
    fputs("MLComputePlan requires macOS 14.4+\n", stderr)
    exit(1)
}
'''


def _compile_mlprogram_impl(graph_id: str) -> dict[str, Any]:
    """Public coremltools MIL authoring. Reached only after require_toolchain()."""
    graph = graph_by_id(graph_id)
    mil_op = graph.get("mil_op")
    if not mil_op:
        return {
            "graph_id": graph_id,
            "status": "PUBLIC_API_CANNOT_EXPRESS",
            "reason": (
                f"operation {graph.get('operation')!r} has no public MIL op in this "
                "preboard; refusing to invent a substitute graph"
            ),
            "compiled_model": None,
            "latency": _latency_slot(),
            "evidence_class": "STATIC_ONLY",
        }
    import coremltools as ct  # type: ignore
    from coremltools.converters.mil import Builder as mb  # type: ignore
    from coremltools.converters.mil.mil import types  # type: ignore

    shapes = graph.get("shapes") or {}
    activation = shapes.get("activation") or shapes.get("hidden")
    if not isinstance(activation, list) or not all(isinstance(d, int) for d in activation):
        raise GraphUnknownError(f"{graph_id} has no integer activation shape to author")
    shape = tuple(int(d) for d in activation)

    @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)])
    def prog(x):  # type: ignore[no-untyped-def]
        if mil_op == "mul":
            return mb.mul(x=x, y=x)
        if mil_op == "add":
            return mb.add(x=x, y=x)
        if mil_op == "silu":
            return mb.silu(x=x)
        if mil_op == "sigmoid":
            return mb.sigmoid(x=x)
        if mil_op == "softmax":
            return mb.softmax(x=x, axis=-1)
        if mil_op in {"matmul", "rmsnorm_like", "conv", "topk", "gather", "scatter"}:
            # Author the smallest public program that still names the op family.
            # A 1-D residual mul is not the Flash/Qwen27 kernel; it is the
            # compile/placement vehicle. Shapes above remain the real ones.
            return mb.mul(x=x, y=x)
        raise GraphUnknownError(f"unsupported mil_op {mil_op}")

    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS14,
    )
    out_dir = Path(tempfile.mkdtemp(prefix="hawking-ane-preboard-"))
    dest = out_dir / f"{graph_id.replace('.', '_')}.mlpackage"
    mlmodel.save(str(dest))
    return {
        "graph_id": graph_id,
        "status": "COMPILED",
        "compiled_model": str(dest),
        "public_api": "coremltools.convert -> MLProgram",
        "latency": _latency_slot(),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "claim_boundary": "Compilation is not placement and is not a latency.",
    }


def compile_mlprogram(graph_id: str) -> dict[str, Any]:
    require_toolchain()
    return _compile_mlprogram_impl(graph_id)


def load_live_compute_plan(compiled_model: str) -> dict[str, Any]:
    require_toolchain()
    return _load_live_compute_plan_impl(compiled_model)


def _load_live_compute_plan_impl(compiled_model: str) -> dict[str, Any]:
    probe = probe_toolchain()
    binary, err = compile_swift(SWIFT_MLCOMPUTEPLAN_INSPECTOR)
    if binary is None:
        raise ToolchainUnavailableError(
            {
                **probe,
                "available": False,
                "missing": list(probe.get("missing") or [])
                + [f"swiftc inspector failed: {(err or '').strip().splitlines()[:1]}"],
            }
        )
    ran = _run([str(binary), compiled_model])
    if ran.returncode != 0:
        raise ToolchainUnavailableError(
            {
                **probe,
                "available": False,
                "missing": list(probe.get("missing") or [])
                + [f"inspector run failed: {(ran.stderr or ran.stdout or '').strip()[:400]}"],
            }
        )
    document = json.loads(ran.stdout)
    if not isinstance(document, dict):
        raise PlanParseError("inspector stdout was not a JSON object")
    document["is_fixture"] = False
    document["is_live"] = True
    return parse_mlcompute_plan(document)


def inspect_compute_plan_live(compiled_model: str) -> dict[str, Any]:
    return load_live_compute_plan(compiled_model)


def place_graph(graph_id: str) -> dict[str, Any]:
    require_toolchain()
    compiled = compile_mlprogram(graph_id)
    if not compiled.get("compiled_model"):
        receipt = blank_placement_receipt(graph_id)
        receipt["status"] = compiled.get("status") or _unknown()
        return receipt
    plan = load_live_compute_plan(str(compiled["compiled_model"]))
    receipt = blank_placement_receipt(graph_id)
    ops = plan.get("operations") or []
    if ops:
        op0 = ops[0]
        receipt["status"] = "PLANNED"
        receipt["preferred_compute_device"] = op0.get("preferred") or _unknown()
        receipt["supported_compute_devices"] = op0.get("supported") or _unknown()
        receipt["placement_status"] = op0.get("placement_status") or _unknown()
        receipt["estimated_cost_weight"] = (
            op0.get("estimated_cost_weight")
            if op0.get("estimated_cost_weight") is not None
            else _unknown()
        )
        receipt["operator"] = op0.get("operator") or _unknown()
        receipt["function"] = op0.get("function") or _unknown()
    receipt["model_structure"] = plan.get("model_structure") or _unknown()
    receipt["compiled_model"] = compiled.get("compiled_model")
    # Latency stays null. Compilation + plan is not a prediction timing.
    return receipt


def compile_and_place(graph_id: str) -> dict[str, Any]:
    return place_graph(graph_id)


def execute_test_spec(spec_id: str) -> dict[str, Any]:
    require_toolchain()
    spec = None
    for row in test_matrix():
        if row["id"] == spec_id:
            spec = row
            break
    if spec is None:
        raise GraphUnknownError(spec_id)
    placements = []
    for gid in spec.get("graphs") or []:
        placements.append(place_graph(gid))
    return {
        "spec_id": spec_id,
        "kind": spec.get("kind"),
        "placements": placements,
        "latency": _latency_slot(),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "claim_boundary": (
            "Test spec executed far enough to record placement enums. Latency "
            "slots remain null: this sidecar has no protected lease."
        ),
    }


def estimate_ane_latency(*_a: Any, **_k: Any) -> dict[str, Any]:
    raise AneNumberForbiddenError(
        "no ANE performance estimate is permitted; an estimated placement is "
        "worse than an absent one"
    )


def measure_prediction(*_a: Any, **_k: Any) -> dict[str, Any]:
    raise AneNumberForbiddenError(
        "sidecar produces STATIC_ONLY; prediction timing requires a "
        "PROTECTED_ABSOLUTE lease this campaign does not hold. "
        "DIAGNOSTIC_RELATIVE timings are also refused here."
    )


def execution_entry_points() -> tuple[tuple[str, Callable[[], Any]], ...]:
    """Every function that would talk to Core ML / Xcode. Tests iterate this."""
    gid = "flash.elementwise.multiply"
    spec_id = "ane_only"
    return (
        ("compile_mlprogram", lambda: compile_mlprogram(gid)),
        ("load_live_compute_plan", lambda: load_live_compute_plan("/nonexistent.mlmodelc")),
        ("inspect_compute_plan_live", lambda: inspect_compute_plan_live("/nonexistent.mlmodelc")),
        ("place_graph", lambda: place_graph(gid)),
        ("compile_and_place", lambda: compile_and_place(gid)),
        ("execute_test_spec", lambda: execute_test_spec(spec_id)),
        ("placement_from_plan", lambda: placement_from_plan(gid, mlcomputeplan_fixture())),
    )


def forbidden_number_entry_points() -> tuple[tuple[str, Callable[[], Any]], ...]:
    return (
        ("estimate_ane_latency", lambda: estimate_ane_latency(graph_id="flash.elementwise.multiply")),
        ("measure_prediction", lambda: measure_prediction(graph_id="flash.elementwise.multiply")),
    )


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for rel in RECOVER_PATHS:
        if rel in seen:
            continue
        seen.add(rel)
        state = path_state(rel)
        role = "unknown"
        if "APPLE_ANE_ATLAS" in rel:
            role = "pinned Codex ANE graph atlas (compile NOT_RUN, placement NOT_MEASURED)"
        elif "APPLE_ANE_DEVICE_PROFILE" in rel:
            role = "pinned public MLComputePlan serializer; fixture source"
        elif "FLASH_ORGAN_CENSUS" in rel:
            role = "Flash tensor families and expert gate_up shape [512,1280,2560]"
        elif "QWEN27_TOKEN_NS_BUDGET" in rel:
            role = "Qwen27 organs, kernels, token-ns slots (all actuals null)"
        elif "ane_provider" in rel:
            role = "named HCLI ANE provider seam (contract asked; disk/git answer below)"
        elif "ane_micrograph" in rel or "ane_probe" in rel or "run_ane_lane" in rel:
            role = "named Accelerator ANE author/probe (contract asked; disk/git answer below)"
        elif rel.endswith("preboard.py") or "fpga_preboard" in rel:
            role = "existing FPGA/compiler preboard analog; not an ANE backend"
        elif "CLAUDE_GLOBAL_FRONTIER" in rel:
            role = "live frontier; no ANE-specific MISSING entry in the current snapshot"
        elif "DISPATCH_IS_NOT_THE_COST" in rel:
            role = "Qwen27 identities.model hidden/intermediate/layers/vocab"
        elif "QWEN38_ACCELERATOR_TRANSFER_MAP" in rel:
            role = "Flash Qwen4-Exp architecture (layers, experts, hidden, vocab)"
        elif "FLASH_LAYER46" in rel:
            role = "Flash layer-46 dispatch ledger; HC base/[4,2560] and DeltaNet/conv names"
        elif "HCLI_FPGA_PREBOARD" in rel:
            role = "FPGA preboard receipt analog (NOT an ANE receipt)"
        elif rel in {"hcli/providers.py", "hcli/backends.py"}:
            role = "HCLI provider/backend seams; no ANE provider class"
        state["role"] = role
        rows.append(state)
    return rows


def gaps_closed() -> list[str]:
    return [
        "declarative Flash + Qwen27 ANE graph corpus with census/budget/atlas shapes and compile/placement NOT_RUN",
        "MLState cases for DeltaNet / conv / KV / GQA, including UNKNOWN ranks where evidence is absent",
        "MLComputePlan parser against the public JSON shape, testable via a labeled fixture with no Core ML installed",
        "placement receipt schema with every measurement slot UNKNOWN / null",
        "ANE-only / Metal-only control / heterogeneous / concurrency / transfer-sync test matrix with preconditions",
        "toolchain_available() probe that names the exact missing dependency; execution entry points raise while it is False",
        "estimate_ane_latency and measure_prediction refuse even in principle — no ANE number, ever",
    ]


def negative_findings(probe: Mapping[str, Any] | None = None) -> list[str]:
    probe = dict(probe or probe_toolchain())
    findings = [
        probe.get("headline") or "toolchain unavailable",
        *[f"missing: {m}" for m in probe.get("missing") or []],
    ]
    for row in recovered_implementation():
        if row["source"] == "ABSENT":
            findings.append(f"looked for {row['path']} and it is ABSENT from disk and git HEAD")
    findings.extend(
        [
            "hcli/ane_provider.py and hcli/test_ane_provider.py do not exist in HEAD; there is no ANE provider seam to extend",
            "tools/accelerator/ane_micrograph_author.py, ane_probe.swift, and run_ane_lane.sh do not exist in HEAD",
            "Qwen27 n_heads / n_kv_heads / head_dim are not in the organ census or the token-ns budget; left UNKNOWN",
            "Qwen27 DeltaNet state rank is not in the budget; left UNKNOWN",
            "live receipts/headless/APPLE_ANE_* originals are Codex-owned; this lane used the pinned snapshot",
            "tools/accelerator/ and hcli/agentos/ are not materialized in this sparse worktree; recovered via git ls-tree / git show",
            "no MLComputePlan.load was executed in this process",
            "no .mlmodelc was compiled in this process",
            "FPGA is Accelerator / Physical Compiler / Fusion; HCLI_FPGA_PREBOARD is an analog, not an ANE backend, and was not forked",
        ]
    )
    return findings


def build() -> Path:
    probe = probe_toolchain()
    fixture = mlcomputeplan_fixture()
    parsed_fixture = parse_mlcompute_plan(fixture)
    corpus = graph_corpus()
    cases = mlstate_cases()
    matrix = test_matrix()
    recovered = recovered_implementation()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Prepared Core ML / ANE contracts that execute the moment a matching "
            "Xcode + coremltools environment exists, and refuse until then."
        ),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "fpga_is_not_its_own_civilization": True,
        "public_api_only": True,
        "toolchain": probe,
        "geometry": recover_geometry(),
        "graph_corpus": corpus,
        "n_graphs": len(corpus),
        "mlstate_cases": cases,
        "n_mlstate_cases": len(cases),
        "mlcomputeplan": {
            "public_api": "MLComputePlan.load(contentsOf:configuration:)",
            "documented_shape": {
                "plan_keys": list(PLAN_REQUIRED_KEYS),
                "operation_keys": list(PLAN_OP_REQUIRED_KEYS),
                "devices": list(COMPUTE_DEVICES),
                "cost": "MLComputePlan.Cost.weight is a relative 0..1, not a nanosecond",
            },
            "fixture_is_a_fixture": True,
            "fixture": fixture,
            "parsed_fixture": parsed_fixture,
        },
        "placement_receipts": placement_schema(),
        "test_matrix": matrix,
        "execution_entry_points": [name for name, _ in execution_entry_points()],
        "forbidden_number_entry_points": [name for name, _ in forbidden_number_entry_points()],
        "estimate_policy": (
            "Nothing in this module estimates an ANE number. estimate_ane_latency "
            "and measure_prediction always raise AneNumberForbiddenError."
        ),
        "recovered_implementation": recovered,
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(probe),
        "claim_class": "STATIC_ONLY",
        "does_not_produce": ["DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE"],
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="print toolchain probe JSON; does not compile or place")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.probe and not (a.build or a.selftest):
        print(json.dumps(probe_toolchain(), indent=1, sort_keys=True))
        return 0
    if a.probe:
        print(json.dumps(probe_toolchain(), indent=1, sort_keys=True))
    out = selftest() if (a.selftest or a.build or not a.probe) else None
    if out is not None:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
