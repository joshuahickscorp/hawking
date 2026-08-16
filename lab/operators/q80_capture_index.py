"""Memory-mappable sidecar for a multi-GB capture-result.json.

Every Q80 density experiment currently begins by parsing
``capture-result.json`` (~1.38 GiB pretty-printed).  The ``wanted_keys``
filter already skips hidden-vector loads; the remaining wall is the index
parse itself.

This module writes ``capture-index.v1/`` next to the JSON, once:

  header.json          scalars + validity binding + probe ids
  layer.npy            int16   [n_rows]
  token_index.npy      int32   [n_rows]   global step 0..n_tokens-1
  probe_index.npy      int16   [n_rows]
  step_index.npy       int32   [n_rows]   position within the probe
  input_token_id.npy   int32   [n_rows]
  hidden_retained.npy  uint8   [n_rows]
  elements.npy         int32   [n_rows]
  hidden_offset.npy    int64   [n_rows]   byte offset in the hidden file (0 today)
  path_id.npy          int32   [n_rows]   -1 when hidden is null
  path_offsets.npy     int32   [n_unique_paths+1]
  path_blob.npy        uint8   concatenated utf-8 paths
  expert_offsets.npy   int32   [n_rows+1]          CSR
  expert_ids.npy       int32   [n_expert_entries]  CSR
  key_layer.npy        int16   [n_keys]
  key_expert.npy       int32   [n_keys]
  key_offsets.npy      int32   [n_keys+1]          CSR into key_row_ids
  key_row_ids.npy      int32   [n_key_hits]        hidden-retained rows only

Load-time validity is ``size + mtime_ns``.  sha256 of the 1.38 GiB JSON is
recorded at build (hashed while scanning) but is *not* re-checked on every
query — hashing the JSON again would recreate the wall this index exists
to remove.  If size or mtime_ns disagree, the sidecar is ignored and the
caller falls back to today's JSON path.

A directory of ``.npy`` arrays is used instead of ``capture-index.npz``
because npz is a zip archive and is not memory-mappable.

CLI::

    python -m lab.operators.q80_capture_index --capture <dir>
    python -m lab.operators.q80_capture_index --capture <dir> --measure \\
        --receipt receipts/QWEN80_CAPTURE_INDEX.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "hawking.capture.q80_capture_index.v1"
INDEX_DIRNAME = "capture-index.v1"
HEADER_NAME = "header.json"
CAPTURE_RESULT_NAME = "capture-result.json"
DEFAULT_LAYER = 0

# Load-time binding. sha256 is stored for humans / receipts but is too
# expensive to recompute on the query path (see module docstring).
VALIDITY_BINDING = "size_and_mtime_ns"

ARRAY_NAMES: tuple[str, ...] = (
    "layer.npy",
    "token_index.npy",
    "probe_index.npy",
    "step_index.npy",
    "input_token_id.npy",
    "hidden_retained.npy",
    "elements.npy",
    "hidden_offset.npy",
    "path_id.npy",
    "path_offsets.npy",
    "path_blob.npy",
    "expert_offsets.npy",
    "expert_ids.npy",
    "key_layer.npy",
    "key_expert.npy",
    "key_offsets.npy",
    "key_row_ids.npy",
)

_RE_LAYER = re.compile(rb'"layer"\s*:\s*(\d+)')
_RE_EXPERTS = re.compile(rb'"selected_expert_ids"\s*:\s*\[([^\]]*)\]')
_RE_HIDDEN = re.compile(rb'"router_input_hidden_f32le"\s*:\s*(null|\{[^}]*\})')
_RE_PATH = re.compile(rb'"relative_path"\s*:\s*"([^"]+)"')
_RE_ELEMENTS = re.compile(rb'"elements"\s*:\s*(\d+)')
_RE_OFFSET = re.compile(rb'"(?:byte_)?offset"\s*:\s*(\d+)')
_RE_PROBE = re.compile(rb'"probe_id"\s*:\s*"([^"]+)"')
_RE_TOKEN = re.compile(rb'"input_token_id"\s*:\s*(-?\d+)')
_RE_LAYERS_ARR = re.compile(rb'"layers"\s*:\s*\[')

_HASH_CHUNK = 8 * 1024 * 1024
_TAIL_READ = 512 * 1024


class CaptureIndexError(RuntimeError):
    """Sidecar is missing, stale, corrupt, or cannot be built."""


def index_dir(run_dir: Path) -> Path:
    return Path(run_dir) / INDEX_DIRNAME


def candidate_index_dirs(run_dir: Path, dest: Path | None = None) -> list[Path]:
    """Places a valid sidecar may live.

    1. ``<run>/capture-index.v1`` (production default)
    2. next to the resolved capture-result.json (symlink overlays)
    3. ``HAWKING_CAPTURE_INDEX`` if set (read-only capture dirs)
    """
    if dest is not None:
        return [Path(dest)]
    seen: list[Path] = []
    candidates = [
        Path(run_dir) / INDEX_DIRNAME,
        capture_result_path(run_dir).resolve().parent / INDEX_DIRNAME,
    ]
    env = os.environ.get("HAWKING_CAPTURE_INDEX")
    if env:
        candidates.append(Path(env))
    for cand in candidates:
        if cand not in seen:
            seen.append(cand)
    return seen


def capture_result_path(run_dir: Path) -> Path:
    return Path(run_dir) / CAPTURE_RESULT_NAME


def sidecar_nbytes(run_dir: Path) -> int:
    root = index_dir(run_dir)
    if not root.is_dir():
        return 0
    return int(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_list(blob: bytes) -> list[int]:
    if not blob:
        return []
    out: list[int] = []
    for part in blob.split(b","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _read_trailing_scalars(path: Path) -> dict[str, Any]:
    """Parse top-level keys after the probes array (schema, status, ...)."""
    size = path.stat().st_size
    take = min(size, _TAIL_READ)
    with path.open("rb") as handle:
        handle.seek(size - take)
        tail = handle.read()
    text = tail.decode("utf-8", errors="replace")
    pretty = text.rfind('\n  ],\n  "')
    if pretty >= 0:
        extra = "{" + text[pretty + len("\n  ],\n") :]
        try:
            doc = json.loads(extra)
        except json.JSONDecodeError:
            return {}
        return doc if isinstance(doc, dict) else {}
    for marker in (
        '],"runtime_binding":',
        '],"schema":',
        '],"status":',
        '],"claim_boundary":',
        '],"stream_telemetry":',
        '],"wall_clock_secs":',
    ):
        idx = text.rfind(marker)
        if idx >= 0:
            extra = "{" + text[idx + 2 :]
            try:
                doc = json.loads(extra)
            except json.JSONDecodeError:
                continue
            return doc if isinstance(doc, dict) else {}
    return {}


def read_capture_scalars(path: Path) -> dict[str, Any]:
    """Prefix + trailing scalars. Does not parse the probes array."""
    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        _read_capture_header,
    )

    header = dict(_read_capture_header(path))
    header.update(_read_trailing_scalars(path))
    return header


def _source_stat(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "relative_path": CAPTURE_RESULT_NAME,
        "resolved_path": str(path.resolve()),
        "bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def inspect_index(
    run_dir: Path, *, dest: Path | None = None
) -> tuple[str, Path | None, dict[str, Any] | None]:
    """Return ``(ok|missing|stale|corrupt, index_dir, header)``."""
    run_dir = Path(run_dir)
    last: tuple[str, Path | None, dict[str, Any] | None] = ("missing", None, None)
    for root in candidate_index_dirs(run_dir, dest):
        header_path = root / HEADER_NAME
        if not header_path.is_file():
            continue
        try:
            header = json.loads(header_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            last = ("corrupt", root, None)
            continue
        if not isinstance(header, dict) or header.get("schema") != SCHEMA:
            last = ("corrupt", root, header if isinstance(header, dict) else None)
            continue
        missing_arr = next((name for name in ARRAY_NAMES if not (root / name).is_file()), None)
        if missing_arr is not None:
            last = ("corrupt", root, header)
            continue
        src = capture_result_path(run_dir)
        if not src.is_file():
            last = ("stale", root, header)
            continue
        st = src.stat()
        bind = header.get("source") or {}
        try:
            bound_bytes = int(bind.get("bytes", -1))
            bound_mtime = int(bind.get("mtime_ns", -1))
        except (TypeError, ValueError):
            last = ("corrupt", root, header)
            continue
        if bound_bytes != int(st.st_size) or bound_mtime != int(st.st_mtime_ns):
            last = ("stale", root, header)
            continue
        return "ok", root, header
    return last


def index_is_valid(run_dir: Path) -> bool:
    return inspect_index(run_dir)[0] == "ok"


def _last_before(
    starts: np.ndarray, values: list[Any], pos: int, cursor: list[int]
) -> Any | None:
    """Advance ``cursor`` to the last record with start < pos. Return its value."""
    i = cursor[0]
    n = int(starts.size)
    last: Any | None = None
    while i < n and int(starts[i]) < pos:
        last = values[i]
        i += 1
    cursor[0] = i
    return last


def _last_in_range(
    starts: np.ndarray, values: list[Any], lo: int, hi: int
) -> Any | None:
    i = int(np.searchsorted(starts, lo, side="left"))
    j = int(np.searchsorted(starts, hi, side="left"))
    if j <= i:
        return None
    return values[j - 1]


def _first_in_range(
    starts: np.ndarray, values: list[Any], lo: int, hi: int
) -> Any | None:
    i = int(np.searchsorted(starts, lo, side="left"))
    j = int(np.searchsorted(starts, hi, side="left"))
    if j <= i:
        return None
    return values[i]


def _scan_mmap(data: memoryview | bytes) -> dict[str, Any]:
    """One linear scan of the JSON bytes into parallel Python lists."""

    # Six independent regex passes. They only read; run them together.
    with ThreadPoolExecutor(max_workers=6) as pool:
        f_experts = pool.submit(lambda: list(_RE_EXPERTS.finditer(data)))  # type: ignore[arg-type]
        f_layer = pool.submit(lambda: list(_RE_LAYER.finditer(data)))  # type: ignore[arg-type]
        f_hidden = pool.submit(lambda: list(_RE_HIDDEN.finditer(data)))  # type: ignore[arg-type]
        f_probe = pool.submit(lambda: list(_RE_PROBE.finditer(data)))  # type: ignore[arg-type]
        f_token = pool.submit(lambda: list(_RE_TOKEN.finditer(data)))  # type: ignore[arg-type]
        f_steps = pool.submit(lambda: list(_RE_LAYERS_ARR.finditer(data)))  # type: ignore[arg-type]
        expert_ms = f_experts.result()
        layer_ms = f_layer.result()
        hidden_ms = f_hidden.result()
        probe_ms = f_probe.result()
        token_ms = f_token.result()
        step_ms = f_steps.result()

    layer_starts = np.asarray([m.start() for m in layer_ms], dtype=np.int64)
    layer_vals = [int(m.group(1)) for m in layer_ms]
    hidden_starts = np.asarray([m.start() for m in hidden_ms], dtype=np.int64)
    hidden_vals = [m.group(1) for m in hidden_ms]
    probe_starts = np.asarray([m.start() for m in probe_ms], dtype=np.int64)
    probe_vals = [m.group(1).decode("utf-8") for m in probe_ms]
    token_starts = np.asarray([m.start() for m in token_ms], dtype=np.int64)
    token_vals = [int(m.group(1)) for m in token_ms]
    step_starts = np.asarray([m.start() for m in step_ms], dtype=np.int64)

    all_layer = bool(step_starts.size > 0)

    layers: list[int] = []
    token_index: list[int] = []
    probe_index: list[int] = []
    step_index: list[int] = []
    input_token_ids: list[int] = []
    hidden_retained: list[int] = []
    elements: list[int] = []
    hidden_offsets: list[int] = []
    paths: list[str | None] = []
    expert_ids: list[int] = []
    expert_offsets: list[int] = [0]

    probe_ids: list[str] = []
    probe_lookup: dict[str, int] = {}

    cur_probe = [0]
    cur_token = [0]
    cur_step = [0]
    n_expert_matches = len(expert_ms)

    last_probe_name: str | None = None
    last_probe_i = -1
    last_token_id = 0
    last_step_in_probe = -1
    last_global_token = -1
    last_step_marker_i = -1
    hidden_steps = 0
    route_only_steps = 0
    step_has_hidden = False
    steps_seen = 0

    def close_step() -> None:
        nonlocal steps_seen, hidden_steps, route_only_steps, step_has_hidden
        if steps_seen <= 0:
            return
        if step_has_hidden:
            hidden_steps += 1
        else:
            route_only_steps += 1

    zip_layers = len(layer_vals) == n_expert_matches
    zip_hidden = len(hidden_vals) == n_expert_matches

    for ei, match in enumerate(expert_ms):
        pos = match.start()
        prev_end = expert_ms[ei - 1].end() if ei else 0
        next_start = expert_ms[ei + 1].start() if ei + 1 < n_expert_matches else 1 << 62
        if zip_layers:
            layer_hit = layer_vals[ei]
        else:
            layer_hit = _last_in_range(layer_starts, layer_vals, prev_end, pos)
        if zip_hidden:
            hidden_blob = hidden_vals[ei]
        else:
            hidden_blob = _last_in_range(hidden_starts, hidden_vals, prev_end, pos)
            if hidden_blob is None:
                hidden_blob = _first_in_range(
                    hidden_starts, hidden_vals, match.end(), next_start
                )
        probe_name = _last_before(probe_starts, probe_vals, pos, cur_probe)
        token_hit = _last_before(token_starts, token_vals, pos, cur_token)

        # How many "layers": [ markers sit before this row?
        step_cursor = cur_step[0]
        n_step_marks = int(step_starts.size)
        while step_cursor < n_step_marks and int(step_starts[step_cursor]) < pos:
            step_cursor += 1
        cur_step[0] = step_cursor
        step_mark_i = step_cursor - 1

        if probe_name is not None and probe_name != last_probe_name:
            close_step()
            if probe_name not in probe_lookup:
                probe_lookup[probe_name] = len(probe_ids)
                probe_ids.append(probe_name)
            last_probe_name = probe_name
            last_probe_i = probe_lookup[probe_name]
            last_step_in_probe = -1
            last_step_marker_i = -1
            step_has_hidden = False
            steps_seen = 0

        if all_layer:
            if step_mark_i != last_step_marker_i and step_mark_i >= 0:
                close_step()
                last_step_marker_i = step_mark_i
                last_step_in_probe += 1
                last_global_token += 1
                step_has_hidden = False
                steps_seen += 1
        else:
            close_step()
            last_step_in_probe += 1
            last_global_token += 1
            step_has_hidden = False
            steps_seen += 1

        if token_hit is not None:
            last_token_id = int(token_hit)

        ids = _parse_int_list(match.group(1))
        retained = 0
        rel: str | None = None
        n_elem = 0
        off = 0
        if hidden_blob is not None and hidden_blob != b"null":
            retained = 1
            path_m = _RE_PATH.search(hidden_blob)
            el_m = _RE_ELEMENTS.search(hidden_blob)
            off_m = _RE_OFFSET.search(hidden_blob)
            if path_m is not None:
                rel = path_m.group(1).decode("utf-8")
            if el_m is not None:
                n_elem = int(el_m.group(1))
            if off_m is not None:
                off = int(off_m.group(1))
            if rel is not None:
                step_has_hidden = True

        layers.append(int(layer_hit) if layer_hit is not None else DEFAULT_LAYER)
        token_index.append(max(0, last_global_token))
        probe_index.append(max(0, last_probe_i))
        step_index.append(max(0, last_step_in_probe))
        input_token_ids.append(int(last_token_id))
        hidden_retained.append(retained)
        elements.append(n_elem)
        hidden_offsets.append(off)
        paths.append(rel)
        expert_ids.extend(ids)
        expert_offsets.append(len(expert_ids))

    close_step()

    return {
        "layer": layers,
        "token_index": token_index,
        "probe_index": probe_index,
        "step_index": step_index,
        "input_token_id": input_token_ids,
        "hidden_retained": hidden_retained,
        "elements": elements,
        "hidden_offset": hidden_offsets,
        "paths": paths,
        "expert_ids": expert_ids,
        "expert_offsets": expert_offsets,
        "probe_ids": probe_ids,
        "all_layer": all_layer,
        "n_tokens": max(0, last_global_token + 1),
        "hidden_retained_steps": hidden_steps,
        "route_only_steps": route_only_steps,
    }


def _invert_keys(
    layer: np.ndarray,
    hidden_retained: np.ndarray,
    expert_ids: np.ndarray,
    expert_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CSR ``(layer, expert) -> [row_id, ...]`` in first-seen key order."""
    lists: dict[tuple[int, int], list[int]] = {}
    order: list[tuple[int, int]] = []
    n_rows = int(layer.size)
    for rid in range(n_rows):
        if int(hidden_retained[rid]) == 0:
            continue
        lo = int(expert_offsets[rid])
        hi = int(expert_offsets[rid + 1])
        L = int(layer[rid])
        for e in expert_ids[lo:hi]:
            key = (L, int(e))
            bucket = lists.get(key)
            if bucket is None:
                lists[key] = [rid]
                order.append(key)
            else:
                bucket.append(rid)
    n_keys = len(order)
    key_layer = np.empty(n_keys, dtype=np.int16)
    key_expert = np.empty(n_keys, dtype=np.int32)
    key_offsets = np.empty(n_keys + 1, dtype=np.int32)
    key_offsets[0] = 0
    chunks: list[np.ndarray] = []
    for i, key in enumerate(order):
        key_layer[i] = key[0]
        key_expert[i] = key[1]
        ids = np.asarray(lists[key], dtype=np.int32)
        chunks.append(ids)
        key_offsets[i + 1] = int(key_offsets[i]) + int(ids.size)
    if chunks:
        key_row_ids = np.concatenate(chunks).astype(np.int32, copy=False)
    else:
        key_row_ids = np.zeros(0, dtype=np.int32)
    return key_layer, key_expert, key_offsets, key_row_ids


def _intern_paths(paths: list[str | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table: dict[str, int] = {}
    blob_parts: list[bytes] = []
    offsets = [0]
    path_id = np.empty(len(paths), dtype=np.int32)
    for i, rel in enumerate(paths):
        if not rel:
            path_id[i] = -1
            continue
        existing = table.get(rel)
        if existing is None:
            existing = len(table)
            table[rel] = existing
            raw = rel.encode("utf-8")
            blob_parts.append(raw)
            offsets.append(offsets[-1] + len(raw))
        path_id[i] = existing
    if blob_parts:
        blob = np.frombuffer(b"".join(blob_parts), dtype=np.uint8).copy()
    else:
        blob = np.zeros(0, dtype=np.uint8)
    return path_id, np.asarray(offsets, dtype=np.int32), blob


def _save_npy(path: Path, arr: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, arr)


def _atomic_write_index(dest: Path, arrays: Mapping[str, np.ndarray], header: Mapping[str, Any]) -> Path:
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for name, arr in arrays.items():
            _save_npy(tmp / name, arr)
        (tmp / HEADER_NAME).write_text(
            json.dumps(header, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if dest.exists():
            bak = dest.with_name(dest.name + ".bak")
            if bak.exists():
                shutil.rmtree(bak)
            dest.replace(bak)
            try:
                tmp.replace(dest)
            except OSError:
                bak.replace(dest)
                raise
            shutil.rmtree(bak, ignore_errors=True)
        else:
            tmp.replace(dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return dest


def _hidden_width(scalars: Mapping[str, Any], elements: np.ndarray) -> int:
    bounded = scalars.get("bounded_storage") or {}
    if isinstance(bounded, Mapping) and bounded.get("hidden_width"):
        return int(bounded["hidden_width"])
    runtime = scalars.get("runtime_binding") or {}
    if isinstance(runtime, Mapping) and runtime.get("hidden"):
        return int(runtime["hidden"])
    nonzero = elements[elements > 0]
    if nonzero.size:
        return int(nonzero[0])
    return 0


def _layers_with_hidden(layer: np.ndarray, hidden_retained: np.ndarray) -> list[int]:
    if layer.size == 0:
        return []
    mask = hidden_retained.astype(bool, copy=False)
    if not bool(mask.any()):
        return []
    return [int(x) for x in sorted(set(int(v) for v in layer[mask].tolist()))]


def build_capture_index(
    run_dir: Path,
    *,
    force: bool = False,
    workers: int | None = None,
    dest: Path | None = None,
) -> dict[str, Any]:
    """Parse capture-result.json once and write ``capture-index.v1/`` beside it."""
    del workers  # reserved: regex passes already fan out internally
    run_dir = Path(run_dir).expanduser().resolve()
    src = capture_result_path(run_dir)
    if not src.is_file():
        raise CaptureIndexError(f"missing {CAPTURE_RESULT_NAME} under {run_dir}")
    dest_root = Path(dest).expanduser().resolve() if dest is not None else index_dir(run_dir)

    status, _, existing = inspect_index(run_dir, dest=dest_root)
    if status == "ok" and not force:
        header = existing or {}
        return {
            "status": "ALREADY_PRESENT",
            "index_dir": str(dest_root),
            "sidecar_bytes": int(
                sum(p.stat().st_size for p in dest_root.rglob("*") if p.is_file())
            ),
            "n_rows": header.get("n_rows"),
            "n_keys": header.get("n_keys"),
            "source_sha256": (header.get("source") or {}).get("sha256"),
            "validity_binding": VALIDITY_BINDING,
        }

    started = time.perf_counter()
    src_stat = _source_stat(src)
    scalars = read_capture_scalars(src)

    # Hash the JSON concurrently with the mmap scan. Load-time queries do
    # not redo this; it is recorded so a human can audit the binding.
    hash_pool = ThreadPoolExecutor(max_workers=1)
    hash_fut = hash_pool.submit(_sha256_file, src)
    try:
        with src.open("rb") as handle:
            import mmap as _mmap

            mm = _mmap.mmap(handle.fileno(), 0, access=_mmap.ACCESS_READ)
            try:
                scanned = _scan_mmap(mm)
            finally:
                mm.close()
        source_sha = hash_fut.result()
    finally:
        hash_pool.shutdown(wait=True)

    n_rows = len(scanned["layer"])
    if n_rows == 0:
        raise CaptureIndexError(f"scanner found no selected_expert_ids in {src}")

    layer = np.asarray(scanned["layer"], dtype=np.int16)
    token_index = np.asarray(scanned["token_index"], dtype=np.int32)
    probe_index = np.asarray(scanned["probe_index"], dtype=np.int16)
    step_index = np.asarray(scanned["step_index"], dtype=np.int32)
    input_token_id = np.asarray(scanned["input_token_id"], dtype=np.int32)
    hidden_retained = np.asarray(scanned["hidden_retained"], dtype=np.uint8)
    elements = np.asarray(scanned["elements"], dtype=np.int32)
    hidden_offset = np.asarray(scanned["hidden_offset"], dtype=np.int64)
    expert_ids = np.asarray(scanned["expert_ids"], dtype=np.int32)
    expert_offsets = np.asarray(scanned["expert_offsets"], dtype=np.int32)
    path_id, path_offsets, path_blob = _intern_paths(scanned["paths"])
    key_layer, key_expert, key_offsets, key_row_ids = _invert_keys(
        layer, hidden_retained, expert_ids, expert_offsets
    )

    n_hidden = int(hidden_retained.sum())
    n_tokens = int(scanned["n_tokens"])
    expected_hidden = (scalars.get("bounded_storage") or {}).get("hidden_rows_retained_total")
    expected_tokens = (scalars.get("bounded_storage") or {}).get("total_tokens_executed")
    if expected_hidden is None:
        expected_hidden = (scalars.get("capture_summary") or {}).get("hidden_rows_retained_total")
    if expected_tokens is None:
        expected_tokens = (scalars.get("capture_summary") or {}).get("total_tokens")
    if expected_hidden is not None and int(expected_hidden) != n_hidden:
        raise CaptureIndexError(
            f"index hidden-row count {n_hidden} != header hidden_rows_retained_total "
            f"{expected_hidden}; refusing to write a wrong sidecar"
        )
    if expected_tokens is not None and int(expected_tokens) != n_tokens:
        raise CaptureIndexError(
            f"index token count {n_tokens} != header total_tokens {expected_tokens}; "
            "refusing to write a wrong sidecar"
        )

    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        capture_is_all_layer,
    )

    all_layer = bool(scanned["all_layer"] or capture_is_all_layer(scalars))
    hidden_width = _hidden_width(scalars, elements)
    layers_hit = _layers_with_hidden(layer, hidden_retained)
    n_key_hits = int(key_row_ids.size)

    header: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "EARNED_CAPTURE_INDEX_V1",
        "validity_binding": VALIDITY_BINDING,
        "validity_note": (
            "Load-time check is size + mtime_ns. sha256 is recorded at build "
            "(hashed while scanning) but is not re-verified on every query "
            "because hashing a multi-GB JSON would recreate the wall this "
            "index exists to remove. A size or mtime mismatch ignores the "
            "sidecar and falls back to the JSON path. Never served stale."
        ),
        "source": {
            **src_stat,
            "sha256": source_sha,
            "validity_binding": VALIDITY_BINDING,
        },
        "n_rows": n_rows,
        "n_hidden_rows": n_hidden,
        "n_tokens": n_tokens,
        "n_keys": int(key_layer.size),
        "n_expert_entries": int(expert_ids.size),
        "n_key_hits": n_key_hits,
        "n_unique_paths": int(path_offsets.size - 1),
        "hidden_width": hidden_width,
        "all_layer_capture": all_layer,
        "total_steps": n_tokens,
        "hidden_retained_steps": int(scanned["hidden_retained_steps"]),
        "route_only_steps": int(scanned["route_only_steps"]),
        "layers_with_hidden_hits": layers_hit,
        "n_layers_with_hidden_hits": len(layers_hit),
        "probe_ids": list(scanned["probe_ids"]),
        "capture_schema": scalars.get("schema"),
        "capture_status": scalars.get("status"),
        "claim_boundary": scalars.get("claim_boundary"),
        "bounded_storage": scalars.get("bounded_storage"),
        "capture_summary": scalars.get("capture_summary"),
        "runtime_binding": scalars.get("runtime_binding"),
        "capture_protocol_revision": scalars.get("capture_protocol_revision"),
        "input": scalars.get("input"),
        "stream_telemetry": scalars.get("stream_telemetry"),
        "arrays": {name: name for name in ARRAY_NAMES},
        "claim_index": {
            "pack_is_a_lossless_reindex_of_the_json_route_table": True,
            "row_order_matches_json_collect_expert_activations": True,
            "expert_ids_are_csr_int32": True,
            "json_capture_result_remains_the_binding_identity": True,
            "does_not_change_retention_or_sampling": True,
        },
    }

    arrays = {
        "layer.npy": layer,
        "token_index.npy": token_index,
        "probe_index.npy": probe_index,
        "step_index.npy": step_index,
        "input_token_id.npy": input_token_id,
        "hidden_retained.npy": hidden_retained,
        "elements.npy": elements,
        "hidden_offset.npy": hidden_offset,
        "path_id.npy": path_id,
        "path_offsets.npy": path_offsets,
        "path_blob.npy": path_blob,
        "expert_offsets.npy": expert_offsets,
        "expert_ids.npy": expert_ids,
        "key_layer.npy": key_layer,
        "key_expert.npy": key_expert,
        "key_offsets.npy": key_offsets,
        "key_row_ids.npy": key_row_ids,
    }
    written = _atomic_write_index(dest_root, arrays, header)
    wall = time.perf_counter() - started
    header_on_disk = json.loads((written / HEADER_NAME).read_text(encoding="utf-8"))
    header_on_disk["build_wall_secs"] = wall
    (written / HEADER_NAME).write_text(
        json.dumps(header_on_disk, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    nbytes = int(sum(p.stat().st_size for p in written.rglob("*") if p.is_file()))
    return {
        "status": "WRITTEN",
        "index_dir": str(written),
        "sidecar_bytes": nbytes,
        "build_wall_secs": wall,
        "n_rows": n_rows,
        "n_hidden_rows": n_hidden,
        "n_tokens": n_tokens,
        "n_keys": int(key_layer.size),
        "n_expert_entries": int(expert_ids.size),
        "n_key_hits": n_key_hits,
        "hidden_width": hidden_width,
        "source_bytes": src_stat["bytes"],
        "source_sha256": source_sha,
        "source_mtime_ns": src_stat["mtime_ns"],
        "validity_binding": VALIDITY_BINDING,
        "all_layer_capture": all_layer,
        "capture_schema": scalars.get("schema"),
        "capture_status": scalars.get("status"),
    }


def _load_npy(root: Path, name: str, mmap: bool = True) -> np.ndarray:
    path = root / name
    if mmap:
        return np.load(path, mmap_mode="r")
    return np.load(path)


def _decode_path(path_blob: np.ndarray, path_offsets: np.ndarray, path_id: int) -> str:
    if path_id < 0:
        raise CaptureIndexError("row has no hidden path")
    lo = int(path_offsets[path_id])
    hi = int(path_offsets[path_id + 1])
    return bytes(path_blob[lo:hi]).decode("utf-8")


def _index_provenance(
    header: Mapping[str, Any],
    *,
    wanted_keys: set[tuple[int, int]] | None,
    load_vectors: bool,
    by_key: Mapping[tuple[int, int], Any],
) -> dict[str, Any]:
    token_expert_pairs = 0
    for value in by_key.values():
        token_expert_pairs += len(value) if isinstance(value, list) else int(value)
    layers = list(header.get("layers_with_hidden_hits") or [])
    return {
        "total_steps": int(header.get("total_steps") or header.get("n_tokens") or 0),
        "hidden_retained_steps": int(header.get("hidden_retained_steps") or 0),
        "route_only_steps": int(header.get("route_only_steps") or 0),
        "token_expert_pairs": int(token_expert_pairs),
        "layer_expert_pairs_with_hits": len(by_key),
        "experts_with_hits": len({e for (_, e) in by_key}),
        "layers_with_hidden_hits": layers,
        "n_layers_with_hidden_hits": len(layers),
        "all_layer_capture": bool(header.get("all_layer_capture")),
        "capture_schema": header.get("capture_schema"),
        "bounded_storage": header.get("bounded_storage"),
        "n_tokens": int(header.get("n_tokens") or header.get("total_steps") or 0),
        "streamed": False,
        "compact_mmap": False,
        "capture_index": True,
        "capture_index_schema": SCHEMA,
        "capture_index_validity_binding": VALIDITY_BINDING,
        "load_path": "capture_index_v1",
        "wanted_keys": (
            None
            if wanted_keys is None
            else [f"L{L}.E{e}" for L, e in sorted(wanted_keys)]
        ),
        "counts_only": not load_vectors,
    }


def try_walk_from_index(
    run_dir: Path,
    *,
    wanted_keys: set[tuple[int, int]] | None,
    load_vectors: bool,
) -> tuple[dict[tuple[int, int], list[np.ndarray] | int], dict[str, Any]] | None:
    """Answer a collect/count walk from the sidecar, or None to fall back."""
    status, root, header = inspect_index(run_dir)
    if status != "ok" or root is None or header is None:
        return None
    run_dir = Path(run_dir)

    key_layer = _load_npy(root, "key_layer.npy")
    key_expert = _load_npy(root, "key_expert.npy")
    key_offsets = _load_npy(root, "key_offsets.npy")
    n_keys = int(key_layer.size)

    if not load_vectors:
        by_key: dict[tuple[int, int], list[np.ndarray] | int] = {}
        for i in range(n_keys):
            key = (int(key_layer[i]), int(key_expert[i]))
            if wanted_keys is not None and key not in wanted_keys:
                continue
            by_key[key] = int(key_offsets[i + 1]) - int(key_offsets[i])
        return by_key, _index_provenance(
            header, wanted_keys=wanted_keys, load_vectors=False, by_key=by_key
        )

    # Materialize in JSON encounter order: walk rows, load each hidden once,
    # append the same vector to every selected (layer, expert) that is wanted.
    layer = _load_npy(root, "layer.npy")
    hidden_retained = _load_npy(root, "hidden_retained.npy")
    elements = _load_npy(root, "elements.npy")
    hidden_offset = _load_npy(root, "hidden_offset.npy")
    path_id = _load_npy(root, "path_id.npy")
    path_offsets = _load_npy(root, "path_offsets.npy")
    path_blob = _load_npy(root, "path_blob.npy")
    expert_ids = _load_npy(root, "expert_ids.npy")
    expert_offsets = _load_npy(root, "expert_offsets.npy")

    n_rows = int(layer.size)
    by_key = {}
    for rid in range(n_rows):
        if int(hidden_retained[rid]) == 0:
            continue
        lo = int(expert_offsets[rid])
        hi = int(expert_offsets[rid + 1])
        L = int(layer[rid])
        experts = expert_ids[lo:hi]
        if wanted_keys is not None and not any((L, int(e)) in wanted_keys for e in experts):
            continue
        pid = int(path_id[rid])
        rel = _decode_path(path_blob, path_offsets, pid)
        fpath = run_dir / rel
        off = int(hidden_offset[rid])
        n_elem = int(elements[rid])
        if off == 0:
            x = np.fromfile(fpath, dtype="<f4")
        else:
            x = np.memmap(fpath, dtype="<f4", mode="r", offset=off, shape=(n_elem,)).copy()
        if x.size != n_elem:
            raise CaptureIndexError(f"hidden size mismatch at {fpath}: {x.size} != {n_elem}")
        for e in experts:
            key = (L, int(e))
            if wanted_keys is not None and key not in wanted_keys:
                continue
            rows = by_key.setdefault(key, [])
            assert isinstance(rows, list)
            rows.append(x)
    return by_key, _index_provenance(
        header, wanted_keys=wanted_keys, load_vectors=True, by_key=by_key
    )


def _parse_key(text: str) -> tuple[int, int]:
    raw = text.strip().replace("L", "").replace("E", "")
    if "," in raw:
        a, b = raw.split(",", 1)
    elif "." in raw:
        a, b = raw.split(".", 1)
    elif ":" in raw:
        a, b = raw.split(":", 1)
    else:
        raise CaptureIndexError(f"wanted key must be L,E or L.E; got {text!r}")
    return int(a), int(b)


def _arrays_byte_identical(
    left: Mapping[tuple[int, int], np.ndarray],
    right: Mapping[tuple[int, int], np.ndarray],
) -> tuple[bool, str]:
    if set(left) != set(right):
        only_l = sorted(set(left) - set(right))
        only_r = sorted(set(right) - set(left))
        return False, f"key sets differ only_json={only_l[:8]!r} only_index={only_r[:8]!r}"
    for key in sorted(left):
        a = np.ascontiguousarray(left[key], dtype=np.float32)
        b = np.ascontiguousarray(right[key], dtype=np.float32)
        if a.shape != b.shape:
            return False, f"{key} shape {a.shape} != {b.shape}"
        if a.tobytes() != b.tobytes():
            return False, f"{key} bytes differ"
    return True, "byte_identical"


def measure_equivalence(
    run_dir: Path,
    *,
    wanted_keys: set[tuple[int, int]] | None = None,
    max_rows_per_expert: int | None = 2048,
    row_sample_seed: int | None = None,
) -> dict[str, Any]:
    """JSON-path vs index-path: full census + wanted collect + row-cap."""
    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        ROW_CAP_SEED,
        collect_expert_activations,
        count_expert_activations,
    )

    run_dir = Path(run_dir).expanduser().resolve()
    if wanted_keys is None:
        wanted_keys = {(10, 453), (3, 494)}
    if row_sample_seed is None:
        row_sample_seed = ROW_CAP_SEED

    print("[q80-capture-index] census JSON path...", flush=True)
    t0 = time.perf_counter()
    json_counts, json_count_prov = count_expert_activations(run_dir, use_index=False)
    json_census_s = time.perf_counter() - t0
    print(f"[q80-capture-index] census JSON {json_census_s:.3f}s keys={len(json_counts)}", flush=True)

    print("[q80-capture-index] census index path...", flush=True)
    t0 = time.perf_counter()
    idx_counts, idx_count_prov = count_expert_activations(run_dir, use_index=True)
    idx_census_s = time.perf_counter() - t0
    print(f"[q80-capture-index] census index {idx_census_s:.3f}s keys={len(idx_counts)}", flush=True)

    counts_identical = json_counts == idx_counts
    if not counts_identical:
        mismatches = []
        for key in sorted(set(json_counts) | set(idx_counts)):
            if json_counts.get(key) != idx_counts.get(key):
                mismatches.append(
                    {
                        "key": [int(key[0]), int(key[1])],
                        "json": json_counts.get(key),
                        "index": idx_counts.get(key),
                    }
                )
                if len(mismatches) >= 16:
                    break
    else:
        mismatches = []

    print("[q80-capture-index] collect JSON path...", flush=True)
    t0 = time.perf_counter()
    json_stacked, _ = collect_expert_activations(
        run_dir, wanted_keys=wanted_keys, use_index=False
    )
    json_collect_s = time.perf_counter() - t0
    print(f"[q80-capture-index] collect JSON {json_collect_s:.3f}s", flush=True)

    print("[q80-capture-index] collect index path...", flush=True)
    t0 = time.perf_counter()
    idx_stacked, _ = collect_expert_activations(
        run_dir, wanted_keys=wanted_keys, use_index=True
    )
    idx_collect_s = time.perf_counter() - t0
    print(f"[q80-capture-index] collect index {idx_collect_s:.3f}s", flush=True)

    collect_ok, collect_note = _arrays_byte_identical(json_stacked, idx_stacked)

    # Cap lives in collect_expert_activations after the walk. Re-walking the
    # 1.38 GiB JSON a third time does not test anything the first collect did
    # not already prove; apply the same function the loader uses.
    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        subsample_expert_rows,
    )

    print("[q80-capture-index] row-cap on both full matrices + index collect...", flush=True)
    t0 = time.perf_counter()
    json_cap = {}
    for key, arr in json_stacked.items():
        json_cap[key], _ = subsample_expert_rows(
            arr,
            max_rows=max_rows_per_expert,
            seed=int(row_sample_seed),
            layer=key[0],
            expert=key[1],
        )
    json_cap_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx_cap, _ = collect_expert_activations(
        run_dir,
        wanted_keys=wanted_keys,
        max_rows_per_expert=max_rows_per_expert,
        row_sample_seed=row_sample_seed,
        use_index=True,
    )
    idx_cap_s = time.perf_counter() - t0
    cap_ok, cap_note = _arrays_byte_identical(json_cap, idx_cap)
    print(
        f"[q80-capture-index] row-cap json_apply={json_cap_s:.3f}s "
        f"index_collect={idx_cap_s:.3f}s identical={cap_ok}",
        flush=True,
    )

    def _spd(slow: float, fast: float) -> float | None:
        if fast <= 0:
            return None
        return float(slow / fast)

    shapes = {
        f"L{L}.E{e}": [int(arr.shape[0]), int(arr.shape[1])]
        for (L, e), arr in sorted(idx_stacked.items())
    }
    return {
        "census": {
            "json_wall_s": json_census_s,
            "index_wall_s": idx_census_s,
            "speedup": _spd(json_census_s, idx_census_s),
            "n_keys_json": len(json_counts),
            "n_keys_index": len(idx_counts),
            "identical": counts_identical,
            "mismatches_head": mismatches,
            "json_n_tokens": json_count_prov.get("n_tokens"),
            "index_n_tokens": idx_count_prov.get("n_tokens"),
            "json_token_expert_pairs": json_count_prov.get("token_expert_pairs"),
            "index_token_expert_pairs": idx_count_prov.get("token_expert_pairs"),
        },
        "collect_wanted": {
            "keys": [[int(L), int(e)] for L, e in sorted(wanted_keys)],
            "json_wall_s": json_collect_s,
            "index_wall_s": idx_collect_s,
            "speedup": _spd(json_collect_s, idx_collect_s),
            "byte_identical": collect_ok,
            "note": collect_note,
            "shapes": shapes,
        },
        "row_cap": {
            "max_rows_per_expert": max_rows_per_expert,
            "row_sample_seed": int(row_sample_seed),
            "method": (
                "subsample_expert_rows applied to the JSON collect matrices, "
                "compared to collect_expert_activations(..., max_rows_per_expert) "
                "on the index path. json_apply_wall_s is not a JSON parse."
            ),
            "json_apply_wall_s": json_cap_s,
            "index_collect_wall_s": idx_cap_s,
            "bit_identical": cap_ok,
            "note": cap_note,
        },
        "all_equivalent": bool(counts_identical and collect_ok and cap_ok),
    }


def _machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": os.uname().sysname if hasattr(os, "uname") else os.name,
        "release": os.uname().release if hasattr(os, "uname") else "",
        "machine": os.uname().machine if hasattr(os, "uname") else "",
        "cpu_count": os.cpu_count(),
    }
    try:
        import subprocess

        brand = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            timeout=2,
        ).strip()
        if brand:
            info["cpu_brand"] = brand
        mem = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True, timeout=2
        ).strip()
        if mem.isdigit():
            info["unified_memory_bytes"] = int(mem)
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Capture run directory (contains capture-result.json)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if valid")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Index directory (default: <capture>/capture-index.v1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Reserved; regex passes already fan out internally",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Time JSON vs index census + 2-key collect and check equivalence",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["10,453", "3,494"],
        help="wanted_keys for the collect measurement (L,E)",
    )
    parser.add_argument(
        "--max-rows-per-expert",
        type=int,
        default=2048,
        help="Row-cap used for the sampling equivalence check",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Optional measurement JSON path (written only with --measure)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    capture = args.capture.expanduser().resolve()
    if capture.is_file() and capture.name == CAPTURE_RESULT_NAME:
        capture = capture.parent
    print(f"[q80-capture-index] capture {capture}", flush=True)
    t0 = time.perf_counter()
    result = build_capture_index(
        capture, force=args.force, workers=args.workers, dest=args.out
    )
    wall = time.perf_counter() - t0
    result.setdefault("build_wall_secs", wall)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(
        f"[q80-capture-index] {result['status']} wall={result.get('build_wall_secs'):.3f}s "
        f"sidecar_bytes={result.get('sidecar_bytes')}",
        flush=True,
    )
    if not args.measure:
        return 0

    wanted = {_parse_key(k) for k in args.keys}
    print(
        f"[q80-capture-index] measuring JSON vs index; keys={sorted(wanted)}",
        flush=True,
    )
    measured = measure_equivalence(
        capture,
        wanted_keys=wanted,
        max_rows_per_expert=args.max_rows_per_expert,
    )
    receipt = {
        "schema": "hawking.q80.capture_index.measurement.v1",
        "status": (
            "MEASURED_EQUIVALENT" if measured["all_equivalent"] else "MEASURED_DIVERGED"
        ),
        "machine": _machine(),
        "capture_run": str(capture),
        "index_build": result,
        "validity_binding": {
            "load_time": VALIDITY_BINDING,
            "sha256_recorded_at_build": True,
            "sha256_rechecked_on_query": False,
            "stale_sidecar_is_ignored": True,
        },
        "equivalence": measured,
        "claim_boundary": {
            "does_not_change_retention_or_sampling": True,
            "does_not_touch_dsv4f": True,
            "json_path_unchanged_when_sidecar_absent_or_stale": True,
            "speedup_is_measured_not_asserted": True,
        },
    }
    print(json.dumps(measured, indent=2, sort_keys=True), flush=True)
    if args.receipt is not None:
        out = args.receipt.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[q80-capture-index] wrote receipt {out}", flush=True)
    if not measured["all_equivalent"]:
        print("[q80-capture-index] EQUIVALENCE FAILED", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
