#!/usr/bin/env python3
"""Checksum-pinned launcher for the upstream Hugging Face -> GGUF converter.

The converter is a generated, fast-moving 7,879-line upstream tool. Keeping a
second source copy in Hawking made the repository larger without making the
conversion path more maintainable. This launcher preserves the exact historical
CLI and output implementation by materializing the imported blob from Git once,
verifying its SHA-256, then executing it with the caller's Python and arguments.

For source archives or shallow clones that do not contain the pinned commit, set
``STRAND_HF_TO_GGUF`` to a compatible ``convert_hf_to_gguf.py`` checkout.
``HAWKING_TOOL_CACHE`` overrides the default temporary cache directory.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PIN_COMMIT = "5cca10ffc3be133f7fd325f452e90e5301b1628c"
PIN_PATH = "tools/strand/tools/gguf/convert_hf_to_gguf.py"
PIN_SHA256 = "2e73750c607b61ff4cfcade65005c30b8248decc05fbcb52c09cc8c2c0b55552"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _external_converter() -> Path | None:
    configured = os.environ.get("STRAND_HF_TO_GGUF")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path == Path(__file__).resolve():
            raise SystemExit("STRAND_HF_TO_GGUF points back to the launcher")
        if not path.is_file():
            raise SystemExit(f"STRAND_HF_TO_GGUF is not a file: {path}")
        return path

    llama_root = os.environ.get("LLAMA_CPP_ROOT")
    if llama_root:
        candidate = Path(llama_root).expanduser() / "convert_hf_to_gguf.py"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode or not result.stdout.strip():
        raise SystemExit(
            "cannot locate Hawking Git history; set STRAND_HF_TO_GGUF to an "
            "official llama.cpp convert_hf_to_gguf.py"
        )
    return Path(result.stdout.strip())


def _cached_converter() -> Path:
    cache_root = Path(
        os.environ.get(
            "HAWKING_TOOL_CACHE",
            str(Path(tempfile.gettempdir()) / "hawking-tool-cache"),
        )
    )
    target = cache_root / f"convert_hf_to_gguf-{PIN_SHA256[:16]}.py"
    if target.is_file():
        data = target.read_bytes()
        if _sha256(data) == PIN_SHA256:
            return target
        target.unlink()

    repo = _repo_root()
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{PIN_COMMIT}:{PIN_PATH}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise SystemExit(
            "pinned GGUF converter blob is unavailable"
            + (f": {detail}" if detail else "")
            + "\nSet STRAND_HF_TO_GGUF to an official converter checkout."
        )
    actual = _sha256(result.stdout)
    if actual != PIN_SHA256:
        raise SystemExit(
            "pinned GGUF converter checksum mismatch\n"
            f"expected {PIN_SHA256}\nactual   {actual}"
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(result.stdout)
    temporary.chmod(0o755)
    temporary.replace(target)
    return target


def main() -> None:
    converter = _external_converter() or _cached_converter()
    os.execv(sys.executable, [sys.executable, str(converter), *sys.argv[1:]])


if __name__ == "__main__":
    main()
