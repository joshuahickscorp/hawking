#!/usr/bin/env python3
"""Materialize the predeclared Flash E13 Wikinews text corpus.

This is a narrow acquisition adapter, not an E13 evidence producer.  It verifies
the upstream dump identities, streams namespace-0 non-redirect revision text in
dump order, and emits one bounded UTF-8 file for the canonical Rust owner.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import stat
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, BinaryIO


SCHEMAS = {
    "hawking.flash_e13.wikinews_corpus_predeclaration.v1",
    "hawking.flash_e13.wikimedia_corpus_predeclaration.v2",
}
MAX_OUTPUT_BYTES = 1 << 30


def _direct_regular_file(path: Path, label: str) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a direct regular file: {path}")
    return info


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bounded(stream: BinaryIO, payload: bytes, state: dict[str, int]) -> None:
    next_size = state["bytes"] + len(payload)
    if next_size > MAX_OUTPUT_BYTES:
        raise ValueError(
            f"materialized corpus exceeds the Rust producer cap of {MAX_OUTPUT_BYTES} bytes"
        )
    stream.write(payload)
    state["bytes"] = next_size


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _page_child(page: ET.Element, name: str) -> ET.Element | None:
    for child in page:
        if _local_name(child.tag) == name:
            return child
    return None


def _revision_text(page: ET.Element) -> str | None:
    revision = _page_child(page, "revision")
    if revision is None:
        return None
    text = _page_child(revision, "text")
    return None if text is None else text.text


def _append_dump(
    path: Path,
    destination: BinaryIO,
    output: dict[str, int],
    *,
    maximum_emitted_utf8_bytes: int | None,
) -> dict[str, int | bool]:
    source_start = output["bytes"]
    stats = {
        "pages_seen": 0,
        "namespace_zero_pages": 0,
        "redirects_excluded": 0,
        "empty_pages_excluded": 0,
        "pages_emitted": 0,
        "emitted_utf8_bytes": 0,
        "emitted_utf8_byte_limit_reached": False,
    }
    with bz2.open(path, "rb") as source:
        for event, element in ET.iterparse(source, events=("end",)):
            if _local_name(element.tag) != "page":
                continue
            stats["pages_seen"] += 1
            namespace = _page_child(element, "ns")
            if namespace is None or namespace.text != "0":
                element.clear()
                continue
            stats["namespace_zero_pages"] += 1
            if _page_child(element, "redirect") is not None:
                stats["redirects_excluded"] += 1
                element.clear()
                continue
            title_element = _page_child(element, "title")
            text = _revision_text(element)
            title = "" if title_element is None or title_element.text is None else title_element.text
            if not title or not text:
                stats["empty_pages_excluded"] += 1
                element.clear()
                continue
            # ElementTree has already resolved XML entities and normalized XML
            # line endings.  No language cleanup, markup stripping, sampling,
            # normalization, or deduplication occurs here.
            payload = f"{title}\n{text}\n".encode("utf-8", errors="strict")
            source_bytes = output["bytes"] - source_start
            if (
                maximum_emitted_utf8_bytes is not None
                and source_bytes + len(payload) > maximum_emitted_utf8_bytes
            ):
                stats["emitted_utf8_byte_limit_reached"] = True
                element.clear()
                break
            _write_bounded(destination, payload, output)
            stats["pages_emitted"] += 1
            element.clear()
    stats["emitted_utf8_bytes"] = output["bytes"] - source_start
    return stats


def materialize(predeclaration: Path, source_dir: Path, output_path: Path) -> dict[str, Any]:
    _direct_regular_file(predeclaration, "predeclaration")
    document = json.loads(predeclaration.read_text(encoding="utf-8"))
    if document.get("schema") not in SCHEMAS:
        raise ValueError(f"predeclaration schema must be one of {sorted(SCHEMAS)}")
    if document.get("transform", {}).get("implementation") not in {
        "this_file.v1",
        "this_file.v2",
    }:
        raise ValueError("predeclaration does not name this transform implementation")
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("predeclaration must contain at least one source")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to replace existing corpus: {output_path}")
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_receipts: list[dict[str, Any]] = []
    page_receipts: list[dict[str, int]] = []
    output_state = {"bytes": 0}
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{output_path.name}.", dir=output_path.parent, delete=False
        ) as destination:
            temporary = Path(destination.name)
            for source in sources:
                if not isinstance(source, dict):
                    raise ValueError("every predeclared source must be an object")
                name = source.get("dump_name")
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError("dump_name must be a single path component")
                path = source_dir / name
                info = _direct_regular_file(path, f"source {name}")
                expected_bytes = source.get("content_length")
                if info.st_size != expected_bytes:
                    raise ValueError(
                        f"source {name} byte mismatch: {info.st_size} != {expected_bytes}"
                    )
                observed_sha1 = _digest(path, "sha1")
                if observed_sha1 != source.get("upstream_sha1"):
                    raise ValueError(f"source {name} SHA-1 does not match the upstream manifest")
                source_receipts.append(
                    {
                        "dump_name": name,
                        "bytes": info.st_size,
                        "sha1": observed_sha1,
                        "sha256": _digest(path, "sha256"),
                    }
                )
                maximum_emitted = source.get("maximum_emitted_utf8_bytes")
                if maximum_emitted is not None and (
                    not isinstance(maximum_emitted, int)
                    or isinstance(maximum_emitted, bool)
                    or maximum_emitted <= 0
                    or maximum_emitted > MAX_OUTPUT_BYTES
                ):
                    raise ValueError(
                        "maximum_emitted_utf8_bytes must be a positive integer within the corpus cap"
                    )
                page_receipts.append(
                    _append_dump(
                        path,
                        destination,
                        output_state,
                        maximum_emitted_utf8_bytes=maximum_emitted,
                    )
                )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {
        "schema": "hawking.flash_e13.wikinews_corpus_materialization.v1",
        "predeclaration_path": str(predeclaration.resolve()),
        "predeclaration_sha256": _digest(predeclaration, "sha256"),
        "source_receipts": source_receipts,
        "page_receipts": page_receipts,
        "corpus": {
            "path": str(output_path.resolve()),
            "bytes": output_state["bytes"],
            "sha256": _digest(output_path, "sha256"),
            "utf8": True,
        },
        "model_loaded": False,
        "gpu_or_ane_execution": False,
        "tensor_payload_bytes_read": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predeclaration", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(args.predeclaration, args.source_dir, args.output),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
