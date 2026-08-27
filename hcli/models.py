from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelInfo:
    path: str
    name: str
    size_bytes: int
    family: str
    param_class: str
    quantization: str
    is_projector: bool = False
    provider: str = "local"

    @property
    def display_name(self) -> str:
        if self.provider == "remote":
            return f"remote {self.name}"
        return f"{self.family} {self.param_class} {self.quantization}"

    @property
    def selectable(self) -> bool:
        return not self.is_projector


def _infer_family(filename: str) -> str:
    lower = filename.lower()
    if "qwen" in lower:
        return "Qwen"
    if "llama" in lower:
        return "Llama"
    if "mistral" in lower:
        return "Mistral"
    if "phi" in lower:
        return "Phi"
    if "gemma" in lower:
        return "Gemma"
    if "deepseek" in lower:
        return "DeepSeek"
    return "Unknown"


def _infer_param_class(filename: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", filename, re.IGNORECASE)
    if match:
        return f"{match.group(1)}B"
    return "?B"


def _infer_quantization(filename: str) -> str:
    match = re.search(r"(q\d+(?:_\w+)?|f16|f32)", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "?"


# Filename heuristic used only when GGUF metadata cannot be read (tiny/fake
# files, truncated headers). A sidecar is a multimodal projector or
# LoRA/adapter, not a standalone language model. Tokens are matched as
# whole path components (start / separator / extension), not a raw
# substring, so "mmproj-model-bf16.gguf" is a projector and a genuine
# "Qwen3.8-27B-...gguf" is not.
_SIDECAR_NAME_RE = re.compile(
    r"(?:^|[._-])(mmproj|mm-proj|projector|visionproj|vision-proj|adapter|lora)(?:[._-]|$)",
    re.IGNORECASE,
)

_SIDECAR_TYPES = {"mmproj", "projector", "clip", "adapter", "lora"}
_SIDECAR_ARCH = {"clip", "llava"}
_GGUF_MAGIC = b"GGUF"
_GGUF_TYPE_SIZE = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
    10: 8, 11: 8, 12: 8,
}
_GGUF_MAX_HEADER = 256 * 1024
_GGUF_MAX_STRING = 1_000_000
_GGUF_MAX_KV = 64


def _filename_is_sidecar(filename: str) -> bool:
    return _SIDECAR_NAME_RE.search(filename) is not None


def _read_gguf_identity(path: str) -> Optional[Dict[str, object]]:
    """Read just enough GGUF metadata to decide model vs sidecar.

    Stops at the first decisive key (general.type / general.architecture /
    tokenizer / clip projector) so discovery never walks tokenizer arrays.
    Returns None if the file is not a readable GGUF header.
    """
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != _GGUF_MAGIC:
                return None
            version_raw = fh.read(4)
            counts = fh.read(16)
            if len(version_raw) < 4 or len(counts) < 16:
                return None
            _version = struct.unpack("<I", version_raw)[0]
            _n_tensors, n_kv = struct.unpack("<QQ", counts)
            if n_kv > 10_000:
                return None

            def read_u32() -> int:
                raw = fh.read(4)
                if len(raw) < 4:
                    raise EOFError
                return struct.unpack("<I", raw)[0]

            def read_u64() -> int:
                raw = fh.read(8)
                if len(raw) < 8:
                    raise EOFError
                return struct.unpack("<Q", raw)[0]

            def read_str() -> str:
                n = read_u64()
                if n > _GGUF_MAX_STRING:
                    raise ValueError("gguf string too long")
                data = fh.read(int(n))
                if len(data) < n:
                    raise EOFError
                return data.decode("utf-8", "replace")

            identity: Dict[str, object] = {
                "architecture": "",
                "type": "",
                "has_tokenizer": False,
                "has_lm_blocks": False,
                "has_clip": False,
                "got_metadata": False,
            }

            for _ in range(min(int(n_kv), _GGUF_MAX_KV)):
                if fh.tell() > _GGUF_MAX_HEADER:
                    break
                key = read_str()
                vtype = read_u32()
                identity["got_metadata"] = True
                key_l = key.lower()

                if key_l.startswith("tokenizer.ggml") or key_l.startswith("tokenizer.chat"):
                    identity["has_tokenizer"] = True
                if key_l.startswith("clip."):
                    identity["has_clip"] = True
                if key_l.endswith(".block_count") and not key_l.startswith("clip."):
                    identity["has_lm_blocks"] = True

                value: object = None
                if vtype == 8:
                    value = read_str()
                elif vtype == 9:
                    arr_t = read_u32()
                    arr_n = read_u64()
                    if arr_n > 256 or fh.tell() > _GGUF_MAX_HEADER:
                        break
                    elem = _GGUF_TYPE_SIZE.get(arr_t)
                    if arr_t == 8:
                        for _i in range(int(arr_n)):
                            read_str()
                            if fh.tell() > _GGUF_MAX_HEADER:
                                break
                    elif elem is not None:
                        fh.read(elem * int(arr_n))
                    else:
                        break
                elif vtype in _GGUF_TYPE_SIZE:
                    fh.read(_GGUF_TYPE_SIZE[vtype])
                else:
                    break

                if key_l == "general.architecture" and isinstance(value, str):
                    identity["architecture"] = value
                elif key_l == "general.type" and isinstance(value, str):
                    identity["type"] = value

                typ = str(identity["type"]).lower()
                arch = str(identity["architecture"]).lower()
                if typ in _SIDECAR_TYPES or arch in _SIDECAR_ARCH or identity["has_clip"]:
                    identity["sidecar"] = True
                    return identity
                if typ == "model" or identity["has_tokenizer"] or identity["has_lm_blocks"]:
                    identity["sidecar"] = False
                    return identity

            typ = str(identity["type"]).lower()
            arch = str(identity["architecture"]).lower()
            if typ in _SIDECAR_TYPES or arch in _SIDECAR_ARCH or identity["has_clip"]:
                identity["sidecar"] = True
            elif typ == "model" or identity["has_tokenizer"] or identity["has_lm_blocks"]:
                identity["sidecar"] = False
            elif identity["got_metadata"]:
                # Parsed metadata and found no language-model blocks.
                identity["sidecar"] = True
            return identity
    except (OSError, EOFError, ValueError, struct.error):
        return None


def _is_projector(path: str, filename: str) -> bool:
    identity = _read_gguf_identity(path)
    if identity is not None and "sidecar" in identity:
        return bool(identity["sidecar"])
    return _filename_is_sidecar(filename)


def _make_info(full: str, filename: str, size: int) -> ModelInfo:
    return ModelInfo(
        path=full,
        name=filename,
        size_bytes=size,
        family=_infer_family(filename),
        param_class=_infer_param_class(filename),
        quantization=_infer_quantization(filename),
        is_projector=_is_projector(full, filename),
        provider="llamacpp",
    )


def selectable_models(models: List[ModelInfo]) -> List[ModelInfo]:
    return [m for m in models if not m.is_projector]


def discover_models(roots: Optional[List[str]] = None) -> List[ModelInfo]:
    if roots is None:
        roots = [os.path.expanduser("~/models")]
    models: List[ModelInfo] = []
    seen: set = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        # Native profiles/artifacts are model selections too.  Discovery only
        # reads their identity metadata; it never starts a runtime or opens
        # the weight catalog.
        from .hawking_native import is_hawking_native_path

        if is_hawking_native_path(root):
            full = os.path.realpath(root)
            if full not in seen:
                info = _info_from_path(full)
                if info is not None:
                    seen.add(full)
                    models.append(info)
            continue
        if _is_mlx_dir(root):
            full = os.path.realpath(root)
            if full not in seen:
                seen.add(full)
                models.append(_info_from_mlx(full))
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if is_hawking_native_path(dirpath):
                full = os.path.realpath(dirpath)
                if full not in seen:
                    info = _info_from_path(full)
                    if info is not None:
                        seen.add(full)
                        models.append(info)
                dirnames[:] = []
                continue
            if _is_mlx_dir(dirpath):
                full = os.path.realpath(dirpath)
                if full not in seen:
                    seen.add(full)
                    models.append(_info_from_mlx(full))
                dirnames[:] = []
                continue
            for fn in filenames:
                low = fn.lower()
                full_candidate = os.path.realpath(os.path.join(dirpath, fn))
                # A provider profile is a first-class model selection even if
                # its filename is vendor-neutral.  The predicate parses the
                # JSON shape before admitting it, so arbitrary config JSON is
                # not exposed as a selectable model.
                if low.endswith((".noetic", ".hawking", ".hawking.json")) or (
                    low.endswith(".json") and is_hawking_native_path(full_candidate)
                ):
                    full = full_candidate
                    if full in seen:
                        continue
                    info = _info_from_path(full)
                    if info is not None:
                        seen.add(full)
                        models.append(info)
                    continue
                if low.endswith(".gguf"):
                    full = os.path.realpath(os.path.join(dirpath, fn))
                    if full in seen:
                        continue
                    seen.add(full)
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        size = 0
                    models.append(_make_info(full, fn, size))
    return models


def _is_mlx_dir(path: str) -> bool:
    """Same predicate as backends.is_mlx_model_dir; lazy to keep import light."""
    from .backends import is_mlx_model_dir

    return is_mlx_model_dir(path)


def _info_from_mlx(path: str) -> ModelInfo:
    from .backends import mlx_quantisation_label, model_bytes_at

    path = os.path.realpath(os.path.expanduser(path))
    fn = os.path.basename(path.rstrip(os.sep))
    size = model_bytes_at(path) or 0
    return ModelInfo(
        path=path,
        name=fn,
        size_bytes=int(size),
        family=_infer_family(path),
        param_class=(
        _infer_param_class(path)
        if _infer_param_class(path) != "?B"
        else _infer_param_class(fn)
    ),
        quantization=mlx_quantisation_label(path),
        is_projector=False,
        provider="mlx",
    )


def _info_from_path(path: str) -> Optional[ModelInfo]:
    from .backends import OpenAICompatibleBackend, is_remote_endpoint
    from .hawking_native import config_for_model_path, is_hawking_native_path

    if is_remote_endpoint(path):
        # URLs are valid explicit selections even though they cannot be found
        # by local discovery.  RuntimePool will health-check the endpoint.
        remote = OpenAICompatibleBackend(path)
        identity = remote.identity()
        parsed = remote.selection()
        return ModelInfo(
            path=parsed,
            name=str(identity.get("model_id") or "remote"),
            size_bytes=0,
            family="Remote",
            param_class="?B",
            quantization="provider-managed",
            provider="remote",
        )

    path = os.path.realpath(os.path.expanduser(path))

    if is_hawking_native_path(path):
        try:
            config = config_for_model_path(path)
            identity = config.identity()
            size = int(
                (identity.get("artifact_inventory") or {}).get("artifact_bytes") or 0
            )
            name = config.resident_identity
            family = str(config.family or "Unknown")
            param_class = str(config.param_class or "?B")
            quantization = str(config.quantisation or "unknown")
            provider = str(config.provider or "native")
        except Exception:
            size = 0
            name = os.path.basename(path.rstrip(os.sep))
            family = "Unknown"
            param_class = "?B"
            quantization = "unknown"
            provider = "native"
        return ModelInfo(
            path=path,
            name=name,
            size_bytes=size,
            family=family,
            param_class=param_class,
            quantization=quantization,
            is_projector=False,
            provider=provider,
        )
    if os.path.isdir(path) and _is_mlx_dir(path):
        return _info_from_mlx(path)
    if not os.path.isfile(path):
        return None
    fn = os.path.basename(path)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return _make_info(path, fn, size)


def resolve_model(explicit: Optional[str] = None,
                  project_config: Optional[str] = None,
                  global_config: Optional[str] = None,
                  env: Optional[str] = None,
                  discovered: Optional[List[ModelInfo]] = None) -> Optional[ModelInfo]:
    if explicit:
        return _info_from_path(explicit)
    if project_config:
        info = _info_from_path(project_config)
        if info is not None:
            return info
    if global_config:
        info = _info_from_path(global_config)
        if info is not None:
            return info
    if env:
        info = _info_from_path(env)
        if info is not None:
            return info
    if discovered is None:
        discovered = discover_models()
    selectable = selectable_models(discovered)
    if len(selectable) == 1:
        return selectable[0]
    return None


class ModelRegistry:
    """Deterministic local GGUF + MLX-dir inventory and selection facade.

    Discovery never loads a model. Ambiguous discovery remains ambiguous;
    HCLI does not silently choose one of several installed models.
    Projector/adapter sidecar GGUFs are inventoried (is_projector=True) but
    are not selectable and do not count toward auto-select ambiguity.
    MLX weight directories (config.json + safetensors) are first-class.
    The deleted Q5_K GGUF is not required; a missing file is simply absent.
    """

    def __init__(self, roots: Optional[List[str]] = None) -> None:
        self.roots = list(
            roots
            if roots is not None
            else [os.path.expanduser("~/models")]
        )
        self._models: Optional[List[ModelInfo]] = None
        self.selected: Optional[ModelInfo] = None

    def discover(self, refresh: bool = False, include_sidecars: bool = False) -> List[ModelInfo]:
        if self._models is None or refresh:
            self._models = discover_models(self.roots)

        models = list(self._models)
        if not include_sidecars:
            models = selectable_models(models)
        return models

    @property
    def models(self) -> List[ModelInfo]:
        return self.discover()

    def resolve(
        self,
        explicit: Optional[str] = None,
        project_config: Optional[str] = None,
        global_config: Optional[str] = None,
        env: Optional[str] = None,
    ) -> Optional[ModelInfo]:
        info = resolve_model(
            explicit=explicit,
            project_config=project_config,
            global_config=global_config,
            env=env,
            discovered=self.discover(include_sidecars=True),
        )

        if info is not None:
            self.selected = info

        return info

    def select(self, selector: str) -> Optional[ModelInfo]:
        """Select by 1-based index, exact path, filename, or display name."""

        selector = (selector or "").strip()

        if not selector:
            return self.selected

        models = self.discover()

        try:
            index = int(selector)
        except ValueError:
            index = None

        if index is not None:
            if 1 <= index <= len(models):
                self.selected = models[index - 1]
                return self.selected

            return None

        expanded = os.path.realpath(
            os.path.expanduser(selector)
        )

        inventory = self.discover(include_sidecars=True)
        for info in inventory:
            if (
                os.path.realpath(info.path) == expanded
                or info.name == selector
                or info.display_name == selector
            ):
                if info.is_projector:
                    return None
                self.selected = info
                return info

        # Explicit path not under a discovery root is still a valid explicit
        # model selection if it is a real GGUF file or an MLX weight directory.
        info = resolve_model(explicit=selector)

        if info is not None and not info.is_projector:
            self.selected = info
            return info

        return None

    def status(self) -> dict:
        models = self.discover()

        return {
            "roots": list(self.roots),
            "count": len(models),
            "selected": (
                self.selected.path
                if self.selected is not None
                else None
            ),
        }
