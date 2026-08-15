"""Bounded physical Qwen80 architecture and Gravity research for Ascension.

The raw Qwen3-Coder-Next body remains a source authority and never a tournament
participant.  A separately controlled acquisition lane may retain the full
pinned source body under its own verification and disk-reserve contract.  This
worker itself uses only small authenticated safetensors windows for its two
research lanes:

* exact hybrid architecture extraction (DeltaNet, attention, MoE, state); and
* representative-source Gravity representation research.

All output is candidate research evidence.  It cannot qualify Qwen80 or clear
a manager gate, irrespective of which source-acquisition policy is active.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import struct
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

import numpy as np

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
DEFAULT_METADATA = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/lifecycle/source-admission/QWEN80_SOURCE_METADATA_CANDIDATE.json"
DEFAULT_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80"
QWEN_NEXT_DELTANET_PROBE = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/kernel/QWEN_NEXT_GATED_DELTANET_METAL_COMPONENT_PROBE.json"
SCHEMA = "hawking.ascension.qwen80_physical_campaign.v1"
WINDOW_BYTES = 1 << 20
MAX_HEADER_BYTES = 32 << 20
TARGET_TENSORS = (
    "model.layers.0.linear_attn.in_proj_qkvz.weight",
    "model.layers.11.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.shared_expert.gate_proj.weight",
    "model.layers.0.mlp.experts.0.gate_proj.weight",
)


class Qwen80PhysicalError(RuntimeError):
    """A bounded remote-source operation failed safely."""


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_url(filename: str) -> str:
    return f"https://huggingface.co/{SOURCE_REPOSITORY}/resolve/{SOURCE_REVISION}/{filename}"


def _credential_headers() -> dict[str, str]:
    try:
        from huggingface_hub import get_token
    except ImportError as exc:  # pragma: no cover - installation is environmental
        raise Qwen80PhysicalError("huggingface_hub is required for authenticated bounded streaming") from exc
    token = get_token()
    if not token:
        raise Qwen80PhysicalError("no local Hugging Face credential is available")
    # The header is passed straight to urllib and is never persisted or logged.
    return {"Authorization": f"Bearer {token}"}


def _fetch(filename: str, *, start: int | None = None, length: int | None = None) -> bytes:
    headers = _credential_headers()
    if start is not None:
        if length is None or length <= 0:
            raise Qwen80PhysicalError("bounded range fetch requires a positive length")
        headers["Range"] = f"bytes={start}-{start + length - 1}"
    request = Request(_source_url(filename), headers=headers)
    with urlopen(request, timeout=90) as response:
        payload = response.read()
    if length is not None and len(payload) != length:
        raise Qwen80PhysicalError(
            f"{filename}: requested {length} source bytes, received {len(payload)}"
        )
    return payload


def _safetensors_header(filename: str) -> dict[str, Any]:
    prefix = _fetch(filename, start=0, length=8)
    header_bytes = struct.unpack("<Q", prefix)[0]
    if header_bytes == 0 or header_bytes > MAX_HEADER_BYTES:
        raise Qwen80PhysicalError(f"{filename}: unsafe safetensors header length {header_bytes}")
    raw = _fetch(filename, start=8, length=int(header_bytes))
    decoded = json.loads(raw)
    if not isinstance(decoded, Mapping):
        raise Qwen80PhysicalError(f"{filename}: safetensors header is not an object")
    return {"header_bytes": int(header_bytes), "tensors": dict(decoded)}


def _bf16_values(payload: bytes) -> np.ndarray:
    usable = len(payload) - (len(payload) % 2)
    bits = np.frombuffer(payload[:usable], dtype="<u2").astype(np.uint32)
    return (bits << 16).view(np.float32)


def _quant_research(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise Qwen80PhysicalError("representative BF16 window has no finite values")
    scale_base = max(float(np.max(np.abs(finite))), 1e-12)
    rows = []
    for bits in (2, 3, 4):
        code = (1 << (bits - 1)) - 1
        scale = scale_base / code
        restored = np.rint(finite / scale).clip(-code, code) * scale
        relative_l2 = float(np.linalg.norm(restored - finite) / max(float(np.linalg.norm(finite)), 1e-12))
        rows.append(
            {
                "representation": f"symmetric_uniform_q{bits}_representative_window",
                "weight_bits": bits,
                "billed_bpw_with_fp16_scale_per_64": bits + 16.0 / 64.0,
                "relative_l2": relative_l2,
                "rmse": float(np.sqrt(np.mean(np.square(restored - finite)))),
            }
        )
    return {
        "samples": int(finite.size),
        "mean": float(np.mean(finite, dtype=np.float64)),
        "std": float(np.std(finite, dtype=np.float64)),
        "absmax": scale_base,
        "representation_auction": rows,
    }


def _architecture(index: Mapping[str, str], implementation_path: Path) -> dict[str, Any]:
    names = sorted(index)
    layer_pattern = re.compile(r"^model\.layers\.(\d+)\.")
    linear_layers = sorted({int(match.group(1)) for name in names if ".linear_attn." in name for match in [layer_pattern.match(name)] if match})
    attention_layers = sorted({int(match.group(1)) for name in names if ".self_attn." in name for match in [layer_pattern.match(name)] if match})
    layers = sorted(set(linear_layers) | set(attention_layers))
    implementation = implementation_path.read_bytes()
    return {
        "model_type": "qwen3_next",
        "architecture": "Qwen3NextForCausalLM",
        "layers_observed": layers,
        "layer_count": len(layers),
        "linear_attention_layers": linear_layers,
        "gated_attention_layers": attention_layers,
        "hybrid_layer_schedule_derived_from_weight_names": [
            "gated_deltanet" if layer in linear_layers else "gated_attention" for layer in layers
        ],
        "operators_proven_by_official_weight_names": {
            "gated_deltanet": any(".linear_attn.in_proj_qkvz.weight" in name for name in names),
            "deltanet_state": any(".linear_attn.A_log" in name for name in names),
            "hybrid_attention": bool(attention_layers),
            "moe_router": any(".mlp.gate.weight" in name for name in names),
            "shared_expert": any(".mlp.shared_expert." in name for name in names),
            "routed_experts": any(".mlp.experts." in name for name in names),
        },
        "qwen3next_reference_implementation": {
            "path": str(implementation_path),
            "sha256": _sha256(implementation),
            "classes_required_for_native_port": [
                "Qwen3NextGatedDeltaNet",
                "Qwen3NextAttention",
                "Qwen3NextSparseMoeBlock",
                "Qwen3NextDecoderLayer",
            ],
        },
        "claim_boundary": "architecture extraction and port obligations only; not native runtime execution",
    }


class Qwen80PhysicalCampaign:
    def __init__(self, *, metadata_path: Path, root: Path) -> None:
        self.metadata_path = metadata_path.expanduser().resolve()
        self.root = root.expanduser().resolve()
        self.status_path = self.root / "QWEN80_PHYSICAL_CAMPAIGN_STATUS.json"
        self.architecture_path = self.root / "QWEN80_ARCHITECTURE_SOURCE_WINDOWS_CANDIDATE.json"
        self.gravity_path = self.root / "QWEN80_GRAVITY_REPRESENTATIVE_RESEARCH.json"
        self.index_path = self.root / "source_windows" / "model.safetensors.index.json"
        self.window_dir = self.root / "source_windows" / "tensors"
        self._stopping = False

    def _publish(self, **fields: Any) -> None:
        prior = _read_json(self.status_path) or {}
        _atomic_json(
            self.status_path,
            {
                "schema": SCHEMA,
                "recorded_at": _utc_now(),
                "pid": os.getpid(),
                "heartbeat": int(prior.get("heartbeat", 0)) + 1,
                "phase": fields.pop("phase", "RUNNING"),
                **fields,
                "claim_boundary": {
                    "raw_qwen80_is_source_authority_not_tournament_participant": True,
                    "this_research_worker_materializes_representative_ranges_only": True,
                    "does_not_qualify_80b_manager_or_tournament": True,
                },
            },
        )

    def _metadata(self) -> dict[str, Any]:
        document = _read_json(self.metadata_path)
        if document is None:
            raise Qwen80PhysicalError(f"missing Qwen80 source metadata: {self.metadata_path}")
        checked = verify(document, label=str(self.metadata_path))
        architecture = checked.get("architecture") if isinstance(checked.get("architecture"), Mapping) else {}
        if (
            checked.get("source", {}).get("repository") != SOURCE_REPOSITORY
            or checked.get("source", {}).get("revision") != SOURCE_REVISION
            or architecture.get("num_experts") != 512
            or architecture.get("num_experts_per_tok") != 10
        ):
            raise Qwen80PhysicalError("pinned Qwen80 metadata does not match the required Next topology")
        return checked

    def run_cycle(self) -> None:
        metadata = self._metadata()
        self._publish(phase="ARCHITECTURE_INDEX_STREAMING", lanes={"D": "RUNNING", "E": "WAITING_FOR_WINDOWS"})
        if self.index_path.exists():
            index_bytes = self.index_path.read_bytes()
        else:
            index_bytes = _fetch("model.safetensors.index.json")
            _atomic_bytes(self.index_path, index_bytes)
        index_document = json.loads(index_bytes)
        weight_map = index_document.get("weight_map") if isinstance(index_document, Mapping) else None
        if not isinstance(weight_map, Mapping):
            raise Qwen80PhysicalError("official weight index has no weight_map")
        weight_map = {str(name): str(shard) for name, shard in weight_map.items()}
        missing = [name for name in TARGET_TENSORS if name not in weight_map]
        if missing:
            raise Qwen80PhysicalError(f"pinned official index misses target source windows: {missing}")
        implementation = Path(__import__("transformers").__file__).resolve().parent / "models/qwen3_next/modeling_qwen3_next.py"
        if not implementation.is_file():
            raise Qwen80PhysicalError("local Transformers lacks the Qwen3Next reference implementation")
        architecture = _architecture(weight_map, implementation)
        _atomic_json(
            self.architecture_path,
            seal(
                {
                    "schema": "hawking.ascension.qwen80_architecture_source_windows.v1",
                    "status": "CANDIDATE_ARCHITECTURE_EXTRACTED_FROM_PINNED_REMOTE_INDEX",
                    "recorded_at": _utc_now(),
                    "metadata_seal_sha256": metadata["seal_sha256"],
                    "weight_index_sha256": _sha256(index_bytes),
                    "weight_index_bytes": len(index_bytes),
                    "architecture": architecture,
                    "claim_boundary": {
                        "this_worker_materialized_no_full_weight_body": True,
                        "not_native_runtime_execution": True,
                        "not_qwen80_source_or_manager_certification": True,
                    },
                }
            ),
        )
        self._publish(phase="REPRESENTATIVE_BF16_STREAMING", lanes={"D": "COMPLETE_CANDIDATE", "E": "RUNNING"})
        headers: dict[str, dict[str, Any]] = {}
        windows: list[dict[str, Any]] = []
        for tensor_name in TARGET_TENSORS:
            shard = weight_map[tensor_name]
            header = headers.setdefault(shard, _safetensors_header(shard))
            tensor = header["tensors"].get(tensor_name)
            if not isinstance(tensor, Mapping) or not isinstance(tensor.get("data_offsets"), list):
                raise Qwen80PhysicalError(f"{tensor_name}: absent from advertised shard header")
            begin, end = (int(value) for value in tensor["data_offsets"])
            size = min(WINDOW_BYTES, end - begin)
            absolute_start = 8 + int(header["header_bytes"]) + begin
            destination = self.window_dir / f"{tensor_name.replace('.', '_')}.bf16.window"
            if destination.exists() and destination.stat().st_size == size:
                payload = destination.read_bytes()
            else:
                payload = _fetch(shard, start=absolute_start, length=size)
                _atomic_bytes(destination, payload)
            windows.append(
                {
                    "tensor_name": tensor_name,
                    "source_shard": shard,
                    "shape": tensor.get("shape"),
                    "dtype": tensor.get("dtype"),
                    "source_absolute_byte_range": [absolute_start, absolute_start + size],
                    "materialized_bytes": size,
                    "window_path": str(destination),
                    "window_sha256": _sha256(payload),
                    "research": _quant_research(_bf16_values(payload)),
                }
            )
        _atomic_json(
            self.gravity_path,
            seal(
                {
                    "schema": "hawking.ascension.qwen80_representative_gravity_research.v1",
                    "status": "CANDIDATE_REPRESENTATIVE_BF16_GRAVITY_RESEARCH_COMPLETE",
                    "recorded_at": _utc_now(),
                    "architecture_source_windows_seal_sha256": _read_json(self.architecture_path)["seal_sha256"],
                    "windows": windows,
                    "materialized_weight_bytes": sum(row["materialized_bytes"] for row in windows),
                    "qwen30_seed_priors": self._qwen30_gene_pool(),
                    "qwen30_kernel_priors": self._qwen30_kernel_gene_pool(),
                    "native_qwen_next_kernel_component": self._native_deltanet_kernel_component(),
                    "research_axes": [
                        "gated_deltanet", "hybrid_attention", "512_expert_top10_router",
                        "shared_expert", "q2_q3_q4_representation", "scale_grouping",
                        "native_gated_deltanet_recurrent_state_component",
                    ],
                    "claim_boundary": {
                        "representative_source_windows_only": True,
                        "not_complete_bpw": True,
                        "not_full_gravity_artifact": True,
                        "not_native_full_model_execution": True,
                        "full_source_acquisition_does_not_authorize_manager_promotion": True,
                    },
                }
            ),
        )
        self._publish(
            phase="QWEN80_ARCHITECTURE_AND_GRAVITY_RESEARCH_IDLE",
            lanes={"D": "COMPLETE_CANDIDATE", "E": "COMPLETE_CANDIDATE"},
            artifact_paths={"architecture": str(self.architecture_path), "gravity": str(self.gravity_path)},
            resource_owner={"network": "RELEASED", "disk": "REPRESENTATIVE_WINDOWS_RETAINED"},
        )

    def _qwen30_gene_pool(self) -> dict[str, Any]:
        """Import Qwen30 lessons as priors, never as Qwen80 qualification."""

        path = self.root.parent / "qwen-family" / "QWEN_GRAVITY_GENE_POOL.json"
        document = _read_json(path)
        if document is None:
            return {"status": "NOT_YET_AVAILABLE", "path": str(path)}
        try:
            checked = verify(document, label=str(path))
        except Exception as exc:
            return {"status": "INVALID", "path": str(path), "reason": type(exc).__name__}
        priors = checked.get("priors") if isinstance(checked.get("priors"), list) else []
        return {
            "status": "IMPORTED_AS_RESEARCH_PRIORS",
            "path": str(path),
            "seal_sha256": checked.get("seal_sha256"),
            "prior_count": len(priors),
            "transfer_scope": checked.get("transfer_scope"),
            "claim_boundary": "Qwen80 DeltaNet, hybrid attention, shared expert, and 512/top-10 differences remain independently measured",
        }

    def _native_deltanet_kernel_component(self) -> dict[str, Any]:
        """Bind the actual local Metal recurrence probe into the sealed research record.

        This is a byte-hashed component input, not a promotion-worthy runtime
        receipt: the full source body, convolution, projections, gated norm,
        MoE, and full-token timing remain separate dependencies.
        """

        if not QWEN_NEXT_DELTANET_PROBE.is_file():
            return {"status": "ABSENT", "path": str(QWEN_NEXT_DELTANET_PROBE)}
        raw = QWEN_NEXT_DELTANET_PROBE.read_bytes()
        try:
            document = json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "INVALID_JSON", "path": str(QWEN_NEXT_DELTANET_PROBE), "sha256": _sha256(raw)}
        geometry = document.get("official_qwen_next_geometry") if isinstance(document, Mapping) else {}
        if (
            document.get("status")
            != "PASS_DIRECT_METAL_QWEN_NEXT_GATED_DELTANET_RECURRENCE_COMPONENT_NOT_FULL_MODEL_NOT_TPS_GATE"
            or geometry.get("heads") != 32
            or geometry.get("key_head_dim") != 128
            or geometry.get("value_head_dim") != 128
        ):
            return {"status": "INVALID_OR_UNEXPECTED", "path": str(QWEN_NEXT_DELTANET_PROBE), "sha256": _sha256(raw)}
        return {
            "status": "BYTE_HASH_BOUND_DIRECT_METAL_COMPONENT",
            "path": str(QWEN_NEXT_DELTANET_PROBE),
            "sha256": _sha256(raw),
            "device": document.get("device"),
            "geometry": geometry,
            "max_abs_state_error": document.get("max_abs_state_error"),
            "max_abs_output_error": document.get("max_abs_output_error"),
            "claim_boundary": "exact recurrent-state component only; not a full Qwen80 decoder, TPS gate, or manager qualification",
        }

    def _qwen30_kernel_gene_pool(self) -> dict[str, Any]:
        """Import Qwen30 kernel mechanisms only as non-qualifying priors."""

        path = self.root.parent / "qwen-family" / "QWEN_KERNEL_GENE_POOL.json"
        document = _read_json(path)
        if document is None:
            return {"status": "NOT_YET_AVAILABLE", "path": str(path)}
        try:
            checked = verify(document, label=str(path))
        except Exception as exc:
            return {"status": "INVALID", "path": str(path), "reason": type(exc).__name__}
        return {
            "status": "IMPORTED_AS_KERNEL_RESEARCH_PRIORS",
            "path": str(path),
            "seal_sha256": checked.get("seal_sha256"),
            "transfer_scope": checked.get("transfer_scope"),
            "direct_router_probe_present": isinstance(checked.get("direct_router_probe"), Mapping),
            "direct_gqa_attention_probe_present": isinstance(checked.get("direct_gqa_attention_probe"), Mapping),
            "claim_boundary": "Qwen80 must independently establish its hybrid attention, DeltaNet, shared-expert, and 512/top-10 component evidence",
        }

    def watch(self, *, idle_seconds: float) -> int:
        if idle_seconds <= 0:
            raise Qwen80PhysicalError("idle_seconds must be positive")
        def stop(_signal: int, _frame: Any) -> None:
            self._stopping = True
        old_term = signal.signal(signal.SIGTERM, stop)
        old_int = signal.signal(signal.SIGINT, stop)
        try:
            while not self._stopping:
                self.run_cycle()
                if not self._stopping:
                    time.sleep(idle_seconds)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("once", help="run one bounded architecture and representative-source cycle")
    watch = commands.add_parser("watch", help="rerun safely while retaining bounded source windows")
    watch.add_argument("--idle-seconds", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = Qwen80PhysicalCampaign(metadata_path=args.metadata, root=args.root)
    if args.command == "once":
        campaign.run_cycle()
        return 0
    return campaign.watch(idle_seconds=args.idle_seconds)


__all__ = ["Qwen80PhysicalCampaign", "Qwen80PhysicalError", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
