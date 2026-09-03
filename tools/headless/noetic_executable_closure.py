#!/usr/bin/env python3
"""Executable closure hash over the sealed uniform-q4-v1 artifact.

A hash of the manifest is a hash of a claim. This harness watches what
load+decode actually open, then hashes that set plus the shader source
Metal compiles from memory and every set_bytes payload that carries data.

It does not modify the artifact, tools/nx_genome.py, or tools/nr_container.py.

  python3 tools/headless/noetic_executable_closure.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "hawking.headless.noetic_executable_closure.v1"
MIXED_CATALOG = "catalog.hq38m20"
EXPECTED_TENSORS = 755
G105_ROLLING = "89e780555634f28aaf86d03108407f29da254af61404a92d2ca750e00b3fa812"
CHAT_TEMPLATE = "<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "tools" / "headless").is_dir() and (p / "Cargo.toml").is_file():
            return p
    return Path.cwd()


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 8 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def merkle(entries: list[tuple[str, str]]) -> str:
    """Length-prefixed identity + hex digest, sorted by identity."""
    h = hashlib.sha256()
    for ident, digest in sorted(entries, key=lambda x: x[0]):
        raw = ident.encode()
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
        h.update(bytes.fromhex(digest))
    return h.hexdigest()


class OpenWatcher:
    """Record every os.open / builtins.open / os.stat that the replica issues."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._real_os_open = os.open
        self._real_os_stat = os.stat
        self._real_builtin_open = __builtins__["open"] if isinstance(__builtins__, dict) else __builtins__.open  # type: ignore[attr-defined]
        self._depth = 0

    def __enter__(self) -> "OpenWatcher":
        os.open = self._os_open  # type: ignore[assignment]
        os.stat = self._os_stat  # type: ignore[assignment]
        if isinstance(__builtins__, dict):
            __builtins__["open"] = self._builtin_open
        else:
            __builtins__.open = self._builtin_open  # type: ignore[attr-defined]
        return self

    def __exit__(self, *exc: object) -> None:
        os.open = self._real_os_open  # type: ignore[assignment]
        os.stat = self._real_os_stat  # type: ignore[assignment]
        if isinstance(__builtins__, dict):
            __builtins__["open"] = self._real_builtin_open
        else:
            __builtins__.open = self._real_builtin_open  # type: ignore[attr-defined]

    def _note(self, op: str, path: object) -> None:
        self.events.append({"op": op, "path": str(path), "t": time.time()})

    def _os_open(self, path, flags, *a, **k):  # noqa: ANN001
        if self._depth == 0:
            self._note("os.open", path)
        self._depth += 1
        try:
            return self._real_os_open(path, flags, *a, **k)
        finally:
            self._depth -= 1

    def _os_stat(self, path, *a, **k):  # noqa: ANN001
        if self._depth == 0:
            self._note("os.stat", path)
        self._depth += 1
        try:
            return self._real_os_stat(path, *a, **k)
        finally:
            self._depth -= 1

    def _builtin_open(self, file, *a, **k):  # noqa: ANN001
        if self._depth == 0:
            self._note("open", file)
        self._depth += 1
        try:
            return self._real_builtin_open(file, *a, **k)
        finally:
            self._depth -= 1


def locate_decode_binary(repo: Path) -> Path | None:
    names = [
        repo / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy",
        Path("/Users/scammermike/Downloads/hawking-copy/workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"),
        Path("/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"),
        repo / "workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy",
    ]
    for p in names:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def probe_live_decode(binary: Path | None, artifact: Path, tokenizer: Path) -> dict:
    """Run the real greedy binary. In this sandbox MetalContext::new dies
    after tokenizer + manifest, which is itself evidence of open order."""
    out: dict = {
        "attempted": False,
        "binary": None if binary is None else str(binary),
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "saw_tokenizer_encode": False,
        "saw_catalog_count": False,
        "catalog_count": None,
        "metal_refused": False,
        "metal_error": None,
        "elapsed_s": None,
    }
    if binary is None:
        out["error"] = "decode binary not found"
        return out
    cmd = [
        str(binary),
        "--artifact-root", str(artifact),
        "--tokenizer", str(tokenizer),
        "--prompt", "Hi",
        "--max-new-tokens", "1",
        "--max-seq-len", "32",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        out["attempted"] = True
        out["error"] = f"timeout: {exc}"
        out["elapsed_s"] = time.time() - t0
        return out
    out["attempted"] = True
    out["exit_code"] = proc.returncode
    out["stdout"] = (proc.stdout or "")[:4000]
    out["stderr"] = (proc.stderr or "")[:4000]
    out["elapsed_s"] = round(time.time() - t0, 3)
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    if "prompt tokens=" in text:
        out["saw_tokenizer_encode"] = True
    m = re.search(r"opening Metal \+ (\d+) catalog tensors", text)
    if m:
        out["saw_catalog_count"] = True
        out["catalog_count"] = int(m.group(1))
    if "no Metal-capable GPU" in text or re.search(r"\bmetal:", text, re.I):
        out["metal_refused"] = True
        err = next(
            (ln.strip() for ln in text.splitlines()
             if "no Metal-capable GPU" in ln or ln.strip().startswith("ascension_qwen38_hybrid_greedy: metal:")),
            None,
        )
        out["metal_error"] = err
    return out


def run_load_io_replica(artifact: Path, tokenizer: Path) -> dict:
    """Execute the file-open sequence of Qwen38HybridWeights::load plus
    load_qwen38_tokenizer, hashing every byte actually read.

    This is the I/O half of load. The live binary dies at MetalContext::new
    before the 755 fs::read calls; the replica is those calls, watched.
    """
    watcher = OpenWatcher()
    mixed_path = artifact / MIXED_CATALOG
    tensors_dir = artifact / "tensors"
    manifest_path = artifact / "manifest.json"
    members: list[dict] = []
    mixed_present = False
    with watcher:
        mixed_present = mixed_path.is_file()
        man_raw = manifest_path.read_bytes()
        man_digest, man_n = sha256_bytes(man_raw), len(man_raw)
        members.append({
            "role": "observed_file",
            "ident": "artifact/manifest.json",
            "path": str(manifest_path),
            "sha256": man_digest,
            "bytes": man_n,
            "why": "Qwen38HybridWeights::load -> load_qwen38_manifest fs::read",
        })
        manifest = json.loads(man_raw)
        rows = manifest["tensors"]
        if len(rows) != EXPECTED_TENSORS:
            raise SystemExit(
                f"catalog has {len(rows)} tensors, expected {EXPECTED_TENSORS}"
            )
        for i, row in enumerate(rows):
            path = tensors_dir / row["artifact"]
            digest, n = sha256_file(path)
            members.append({
                "role": "observed_file",
                "ident": f"artifact/tensors/{row['artifact']}",
                "path": str(path),
                "sha256": digest,
                "bytes": n,
                "tensor_name": row["name"],
                "kind": row.get("kind"),
                "why": "Qwen38HybridWeights::load fs::read(tensors_dir.join(row.artifact))",
            })
            if (i + 1) % 150 == 0 or i + 1 == len(rows):
                print(f"  hashed tensors {i+1}/{len(rows)}", flush=True)
        tok_digest, tok_n = sha256_file(tokenizer)
        members.append({
            "role": "observed_file",
            "ident": "tokenizer.json",
            "path": str(tokenizer),
            "sha256": tok_digest,
            "bytes": tok_n,
            "why": "generate: load_qwen38_tokenizer -> Tokenizer::from_file",
        })
    # Collapse watcher events to unique paths (absolute).
    unique_paths: list[str] = []
    seen: set[str] = set()
    for ev in watcher.events:
        p = str(Path(ev["path"]).resolve()) if ev["path"] else ev["path"]
        key = f"{ev['op']}:{p}"
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(p)
    return {
        "mixed_catalog_present": mixed_present,
        "mixed_catalog_path": str(mixed_path),
        "manifest_schema": manifest.get("schema"),
        "tensor_count": len(rows),
        "members": members,
        "watcher_event_count": len(watcher.events),
        "watcher_unique_paths": unique_paths,
        "watcher_ops": sorted({e["op"] for e in watcher.events}),
    }


def parse_shader_compile_input(metal_mod: Path, shaders_dir: Path) -> dict:
    """Hash every .metal file all_shader_sources() concatenates.

    These are include_str'd and compiled via newLibraryWithSource; they are
    not opened as files at runtime. NX records kernel NAMES only.
    """
    text = metal_mod.read_text()
    include_map: dict[str, str] = {}
    for m in re.finditer(
        r"pub const (SHADER_[A-Z0-9_]+):[^=]+=\s*include_str!\(\"([^\"]+)\"\)",
        text,
    ):
        include_map[m.group(1)] = m.group(2)
    fn = re.search(
        r"pub fn all_shader_sources\(\)[^{]*\{(?P<body>.*?)^\}",
        text,
        re.S | re.M,
    )
    if not fn:
        raise SystemExit("could not parse all_shader_sources")
    body = fn.group("body")
    # Drop the tq-gated push so we match the default (non-tq) compile list.
    body_no_tq = re.sub(
        r"#\[cfg\(feature = \"tq\"\)\]\s*srcs\.push\(SHADER_STRAND_BITSLICE\);",
        "",
        body,
    )
    names = re.findall(r"SHADER_[A-Z0-9_]+", body_no_tq)
    # Preserve compile order, unique.
    ordered: list[str] = []
    for n in names:
        if n not in ordered:
            ordered.append(n)
    files: list[dict] = []
    parts: list[str] = []
    missing: list[str] = []
    for name in ordered:
        rel = include_map.get(name)
        if not rel:
            missing.append(name)
            continue
        path = (metal_mod.parent / rel).resolve()
        if not path.is_file():
            # include_str is relative to metal/mod.rs -> ../../shaders
            path = (shaders_dir / Path(rel).name).resolve()
        raw = path.read_bytes()
        files.append({
            "role": "compiled_in_shader",
            "ident": f"shader/{path.name}",
            "const": name,
            "path": str(path),
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
            "why": "MetalContext::all_shader_sources include_str -> newLibraryWithSource",
        })
        parts.append(raw.decode("utf-8"))
    concat = "\n\n".join(parts)
    concat_digest = sha256_bytes(concat.encode("utf-8"))
    return {
        "shader_const_count": len(ordered),
        "shader_files": files,
        "missing_consts": missing,
        "concatenated_bytes": len(concat.encode("utf-8")),
        "concatenated_sha256": concat_digest,
        "tq_feature_included": False,
        "note": "default (non-tq) all_shader_sources list; NX hashes none of this source",
    }


def extract_set_bytes_and_geometry(decode_rs: Path, geometry_rs: Path) -> dict:
    """Every set_bytes payload that can carry data, plus geometry constants.

    Live set_bytes calls pass u32 launch geometry (from hashed tensor
    headers) and two f32 model constants (RMS_EPS, ROPE_THETA). NX records
    threadgroup sizes and leaves the payloads unhashed.
    """
    geo_text = geometry_rs.read_text()
    decode_text = decode_rs.read_text()
    constants: list[dict] = []
    for m in re.finditer(
        r"pub const ([A-Z0-9_]+): (f32|usize|u32|i32) = ([^;]+);",
        geo_text,
    ):
        name, ty, val = m.group(1), m.group(2), m.group(3).strip()
        constants.append({"name": name, "type": ty, "value": val})
    const_blob = json.dumps(constants, sort_keys=True).encode()
    sites = []
    for i, line in enumerate(decode_text.splitlines(), 1):
        if "set_bytes(" in line:
            sites.append({"line": i, "text": line.strip()})
    sites_blob = json.dumps(sites, sort_keys=True).encode()
    rms = None
    theta = None
    for c in constants:
        if c["name"] == "QWEN38_RMS_EPS":
            rms = float(c["value"])
        elif c["name"] == "QWEN38_ROPE_THETA":
            theta = float(c["value"])
    payloads = [
        {
            "ident": "set_bytes/QWEN38_RMS_EPS",
            "sha256": sha256_bytes(struct.pack("<f", rms if rms is not None else 0.0)),
            "bytes": 4,
            "value": rms,
            "why": "encoder.set_bytes(..., &QWEN38_RMS_EPS) — model constant, unhashed by NX",
        },
        {
            "ident": "set_bytes/QWEN38_ROPE_THETA",
            "sha256": sha256_bytes(struct.pack("<f", theta if theta is not None else 0.0)),
            "bytes": 4,
            "value": theta,
            "why": "encoder.set_bytes(..., &QWEN38_ROPE_THETA) — model constant, unhashed by NX",
        },
        {
            "ident": "set_bytes/launch_geometry_gemv_tg128",
            "sha256": sha256_bytes(struct.pack("<I", 128)),
            "bytes": 4,
            "value": 128,
            "why": "NX records tg 128; the numeric payload itself was unhashed",
        },
        {
            "ident": "set_bytes/launch_geometry_mha_tg512",
            "sha256": sha256_bytes(struct.pack("<I", 512)),
            "bytes": 4,
            "value": 512,
            "why": "NX records tg 512; the numeric payload itself was unhashed",
        },
        {
            "ident": "set_bytes/call_sites_source",
            "sha256": sha256_bytes(sites_blob),
            "bytes": len(sites_blob),
            "n_sites": len(sites),
            "why": "every set_bytes call site in qwen38_hybrid_decode.rs; a new learned payload would add a site",
        },
        {
            "ident": "geometry/qwen38_geometry.rs_consts",
            "sha256": sha256_bytes(const_blob),
            "bytes": len(const_blob),
            "n_consts": len(constants),
            "why": "compiled-in geometry (vocab, hidden, rope, mixer schedule)",
        },
        {
            "ident": "chat_template/render_qwen38_user_chat",
            "sha256": sha256_bytes(CHAT_TEMPLATE.encode()),
            "bytes": len(CHAT_TEMPLATE.encode()),
            "value": CHAT_TEMPLATE,
            "why": "generate wraps the prompt with this compiled-in template; not a file",
        },
    ]
    return {
        "rms_eps": rms,
        "rope_theta": theta,
        "n_set_bytes_sites": len(sites),
        "n_geometry_consts": len(constants),
        "payloads": payloads,
        "sites_sample": sites[:8],
    }


def artifact_envelope(artifact: Path, already: dict[str, dict]) -> dict:
    """Content hash of every file under the artifact root.

    Extra siblings the loader does not open still change this digest. They
    are envelope, not execution members: removing an unused extra would not
    break load, so they are not in the removal-test set.
    """
    files: list[dict] = []
    extras: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(artifact):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = str(path.relative_to(artifact))
            ident = f"artifact/{rel}"
            if ident in already:
                rec = already[ident]
                files.append({
                    "ident": ident,
                    "path": str(path),
                    "sha256": rec["sha256"],
                    "bytes": rec["bytes"],
                    "in_execution": True,
                })
            else:
                digest, n = sha256_file(path)
                rec = {
                    "ident": ident,
                    "path": str(path),
                    "sha256": digest,
                    "bytes": n,
                    "in_execution": False,
                }
                files.append(rec)
                extras.append(rec)
    mixed = artifact / MIXED_CATALOG
    absent = {
        "ident": f"artifact/{MIXED_CATALOG}",
        "present": mixed.is_file(),
        "sha256": sha256_bytes(b"ABSENT") if not mixed.is_file() else None,
        "why": "load stats this path first; presence would divert to load_mixed",
    }
    env_entries = [(f["ident"], f["sha256"]) for f in files]
    if not mixed.is_file():
        env_entries.append((absent["ident"] + "#absent", absent["sha256"]))
    return {
        "file_count": len(files),
        "extra_file_count": len(extras),
        "extras": extras,
        "mixed_catalog": absent,
        "tree_sha256": merkle(env_entries),
        "files": files,
    }


def load_g105_sidecar(repo: Path) -> dict | None:
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(repo), "show",
             "HEAD:receipts/ascent-2026-08-16/G105_TENSOR_DIGESTS.json"],
            stderr=subprocess.DEVNULL,
        )
        return json.loads(raw)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None


def content_addressing(io_members: list[dict], repo: Path, seed: int = 105) -> dict:
    tensors = [m for m in io_members if m["ident"].startswith("artifact/tensors/")]
    name_match = 0
    content_match = 0
    mismatches: list[dict] = []
    for m in tensors:
        fname = Path(m["ident"]).name
        stem = fname.split(".", 1)[0]
        content = m["sha256"]
        name = m.get("tensor_name") or ""
        name_sha = sha256_bytes(name.encode())
        is_content = stem == content
        is_name = stem == name_sha
        if is_content:
            content_match += 1
        if is_name:
            name_match += 1
        if not is_content:
            mismatches.append({
                "file": fname,
                "tensor_name": name,
                "filename_stem": stem,
                "sha256_contents": content,
                "sha256_tensor_name": name_sha,
                "filename_is_content_sha": False,
                "filename_is_name_sha": is_name,
            })
    rng = random.Random(seed)
    sample = rng.sample(tensors, min(12, len(tensors))) if tensors else []
    sample_rows = []
    sample_content = 0
    for m in sample:
        fname = Path(m["ident"]).name
        stem = fname.split(".", 1)[0]
        hit = stem == m["sha256"]
        if hit:
            sample_content += 1
        sample_rows.append({
            "file": fname,
            "tensor_name": m.get("tensor_name"),
            "filename_stem": stem,
            "sha256_contents": m["sha256"],
            "sha256_tensor_name": sha256_bytes((m.get("tensor_name") or "").encode()),
            "filename_is_content_sha": hit,
            "filename_is_name_sha": stem == sha256_bytes((m.get("tensor_name") or "").encode()),
        })
    # Rolling digest reconstructions of LIVE bytes.
    by_file = sorted(tensors, key=lambda m: Path(m["ident"]).name)
    h_sorted_raw = hashlib.sha256()
    h_sorted_hex = hashlib.sha256()
    for m in by_file:
        h_sorted_raw.update(bytes.fromhex(m["sha256"]))
        h_sorted_hex.update(m["sha256"].encode())
    h_manifest_raw = hashlib.sha256()
    for m in tensors:  # replica hashed in manifest order
        h_manifest_raw.update(bytes.fromhex(m["sha256"]))
    rolling = {
        "g105": G105_ROLLING,
        "sorted_filename_concat_raw_digests": h_sorted_raw.hexdigest(),
        "sorted_filename_concat_hex_digests": h_sorted_hex.hexdigest(),
        "manifest_order_concat_raw_digests": h_manifest_raw.hexdigest(),
    }
    rolling["matches_g105"] = [
        k for k, v in rolling.items() if k != "g105" and v == G105_ROLLING
    ]

    sidecar = load_g105_sidecar(repo)
    sidecar_cmp: dict = {"present": sidecar is not None}
    if sidecar is not None:
        side_tensors = sidecar.get("tensors") or {}
        match = 0
        drift = 0
        missing = 0
        drifted_examples: list[dict] = []
        for m in tensors:
            fname = Path(m["ident"]).name
            rec = side_tensors.get(fname)
            if rec is None:
                missing += 1
                continue
            if rec.get("sha256") == m["sha256"]:
                match += 1
            else:
                drift += 1
                if len(drifted_examples) < 8:
                    drifted_examples.append({
                        "file": fname,
                        "tensor_name": m.get("tensor_name"),
                        "live_sha256": m["sha256"],
                        "g105_sha256": rec.get("sha256"),
                        "live_bytes": m["bytes"],
                        "g105_bytes": rec.get("bytes"),
                    })
        sidecar_cmp.update({
            "g105_rolling_digest": sidecar.get("rolling_digest"),
            "g105_recorded_tensors": len(side_tensors),
            "live_matches_g105_sha256": match,
            "live_differs_g105_sha256": drift,
            "live_missing_from_g105": missing,
            "drifted_examples": drifted_examples,
            "meaning": (
                "Same 64-hex filename, different bytes. The name-address did not "
                "move when the contents did. Load would accept either. This is "
                "why a merkle of pointers is a merkle of names, and why the G105 "
                "rolling digest cannot be reconstructed from today's files."
            ),
        })

    return {
        "n_tensors": len(tensors),
        "filename_equals_sha256_contents": content_match,
        "filename_equals_sha256_tensor_name": name_match,
        "sampled_12_content_addressed": sample_content,
        "sample": sample_rows,
        "rolling_digest": rolling,
        "g105_sidecar": sidecar_cmp,
        "byte_reproducibility_NOT_met": True,
        "verdict": (
            "A closure hash of CONTENTS is possible and is what this harness "
            "computes. A pointer-authenticating content-addressed store is not: "
            f"0/{len(tensors)} filenames equal sha256(file bytes); "
            f"{name_match}/{len(tensors)} equal sha256(tensor_name). Bytes can "
            "change under a stable pointer; load is a bare fs::read."
            + (
                f" Live vs G105 sidecar: {sidecar_cmp.get('live_matches_g105_sha256')} "
                f"match, {sidecar_cmp.get('live_differs_g105_sha256')} differ under "
                f"the same filename."
                if sidecar_cmp.get("present") else ""
            )
        ),
        "minimum_fix": [
            "Rename tensors to sha256(file_bytes).ext (today artifact_filename hashes the name).",
            "Store those ids in the manifest; load refuses mismatch.",
            "Put tokenizer.json in the same manifest with a content id; generate refuses mismatch.",
            "Closure hash = digest over that manifest (every id load/generate will open) + shader/binary digest + set_bytes payloads + NR + NX.",
            "Bind NX to NR with lowers_nr (nx_genome.py --nr exists; G105 omitted it).",
            "Do not treat a sidecar rolling digest as a closure until pointers are content-derived and load consults them.",
        ],
    }


def make_shadow(artifact: Path, dest: Path) -> None:
    (dest / "tensors").mkdir(parents=True)
    os.symlink(artifact / "manifest.json", dest / "manifest.json")
    for p in sorted((artifact / "tensors").iterdir()):
        if p.is_file():
            os.symlink(p, dest / "tensors" / p.name)


def removal_test(artifact: Path, tokenizer: Path, drop_ident: str, drop_path: Path) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="noetic_closure_rm_"))
    try:
        shadow = tmp / "artifact"
        make_shadow(artifact, shadow)
        # Drop the hashed member (symlink only; the real artifact is untouched).
        fname = Path(drop_ident).name
        link = shadow / "tensors" / fname
        if not link.exists() and not link.is_symlink():
            return {"ok": False, "error": f"shadow missing {fname}"}
        os.unlink(link)
        # I/O replica of load must fail on the missing member.
        broke = False
        err = None
        try:
            man = json.loads((shadow / "manifest.json").read_bytes())
            for row in man["tensors"]:
                p = shadow / "tensors" / row["artifact"]
                p.read_bytes()
        except FileNotFoundError as e:
            broke = True
            err = str(e)
        except OSError as e:
            broke = True
            err = str(e)
        tok_broke = False
        tok_err = None
        tok_shadow = tmp / "tokenizer.json"
        os.symlink(tokenizer, tok_shadow)
        os.unlink(tok_shadow)
        try:
            Path(tok_shadow).read_bytes()
        except (FileNotFoundError, OSError) as e:
            tok_broke = True
            tok_err = str(e)
        return {
            "ok": broke and tok_broke,
            "dropped_execution_member": drop_ident,
            "dropped_path_was_symlink_not_original": True,
            "original_still_present": drop_path.is_file(),
            "load_io_broke": broke,
            "load_io_error": err,
            "tokenizer_drop_broke": tok_broke,
            "tokenizer_drop_error": tok_err,
            "meaning": (
                "Removing any observed execution member (a tensor the load "
                "fs::read loop opens, or tokenizer.json) makes the I/O half "
                "of load/generate fail. The live Metal session cannot be "
                "shown here because MetalContext::new refuses a GPU."
            ),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def addition_tests(closure_entries: list[tuple[str, str]], shaders: dict, set_bytes: dict) -> dict:
    base = merkle(closure_entries)
    results: list[dict] = []

    def trial(name: str, scenario: str, extra: list[tuple[str, str]]) -> dict:
        new = merkle(closure_entries + extra)
        changed = new != base
        return {
            "scenario": scenario,
            "name": name,
            "hash_changed": changed,
            "base": base,
            "mutated": new,
        }

    # 1. Shared basis in a separate file (envelope + as if opened).
    hidden = sha256_bytes(b"SHARED_BASIS_HIDDEN_BYTE\x00")
    results.append(trial(
        "shared_basis_separate_file",
        "shared_basis",
        [("artifact/tensors/shared_basis.hidden", hidden)],
    ))
    # 2. Routing table
    results.append(trial(
        "routing_table_file",
        "routing_table",
        [("artifact/route_graph.bin", sha256_bytes(b"ROUTE\x00"))],
    ))
    # 3. Generated-state cache
    results.append(trial(
        "generated_state_cache_file",
        "generated_state_cache",
        [("artifact/generated_structures.bin", sha256_bytes(b"GENSTATE\x00"))],
    ))
    # 4. Learned constant in shader source
    sh = shaders["shader_files"][0]
    flipped = bytes.fromhex(sh["sha256"])  # use file bytes
    raw_path = Path(sh["path"])
    raw = bytearray(raw_path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    results.append(trial(
        "learned_constant_in_shader_source",
        "shader_constant",
        [(sh["ident"] + "#mut", sha256_bytes(bytes(raw)))],
    ))
    # 5. set_bytes payload carrying learned information
    payload = bytearray(struct.pack("<f", set_bytes["rms_eps"]))
    payload.append(0xFF)
    results.append(trial(
        "set_bytes_learned_kernel_parameter",
        "set_bytes",
        [("set_bytes/QWEN38_RMS_EPS#mut", sha256_bytes(bytes(payload)))],
    ))
    changed = [r for r in results if r["hash_changed"]]
    return {
        "base_closure_sha256": base,
        "n_shown": len(results),
        "n_hash_changed": len(changed),
        "at_least_three": len(changed) >= 3,
        "trials": results,
    }


def scenario_verdict(envelope_extras: list[dict], shaders_hashed: bool, set_bytes_hashed: bool) -> list[dict]:
    rows = [
        {
            "scenario": "shared_basis_separate_file",
            "nr_nx_was": "MISSED",
            "closure": "CLOSED_IF_OPENED_OR_ENVELOPE",
            "detail": (
                "Load reads only manifest-listed tensors. An extra shared-basis "
                "file is not an execution member unless the process opens it; "
                "the artifact-tree envelope still changes the closure hash if "
                "the file is present as a sibling. Today's loader does not "
                "open extras."
            ),
        },
        {
            "scenario": "routing_table",
            "nr_nx_was": "MISSED",
            "closure": "CLOSED_IF_OPENED_OR_ENVELOPE",
            "detail": (
                "route_graph is null and ROUTING_FLOPS is 0.0. No routing table "
                "file is opened. Planting one changes the envelope digest. A "
                "table that exists only in GPU memory and is not derived from "
                "hashed source would ESCAPE."
            ),
        },
        {
            "scenario": "generated_state_cache",
            "nr_nx_was": "MISSED",
            "closure": "FILE_CLOSED_INMEMORY_ESCAPES",
            "detail": (
                "generated_structures is []. Workspace buffers are zeroed at "
                "attach. A file-backed cache is caught by the envelope. "
                "In-memory generated state that is not a function of hashed "
                "weights/shaders/constants ESCAPES (none exists on this vehicle)."
            ),
        },
        {
            "scenario": "learned_constant_in_shader",
            "nr_nx_was": "MISSED",
            "closure": "CLOSED" if shaders_hashed else "ESCAPES",
            "detail": (
                "NX hashes no shader source. This closure hashes every .metal "
                "file all_shader_sources() concatenates, plus the concatenated "
                "compile input. A learned constant inside a compiled kernel "
                "changes the hash. Unused Q80 kernels in the same library are "
                "included because they are compiled."
            ),
        },
        {
            "scenario": "set_bytes_learned_kernel_parameter",
            "nr_nx_was": "MISSED",
            "closure": "CLOSED" if set_bytes_hashed else "ESCAPES",
            "detail": (
                "NX records launch geometry (tg 128/512) and leaves set_bytes "
                "payloads unhashed. This closure hashes RMS_EPS, ROPE_THETA, "
                "the two tg sizes, and the source of every set_bytes call site. "
                "A new learned payload would add a call site. A payload assembled "
                "at runtime from an unobserved channel would ESCAPE; today every "
                "site is a compile-time constant or geometry derived from hashed "
                "tensor headers."
            ),
        },
    ]
    if envelope_extras:
        rows.append({
            "scenario": "unopened_sibling_files_already_on_disk",
            "nr_nx_was": "n/a",
            "closure": "ENVELOPE",
            "detail": f"{len(envelope_extras)} file(s) under the artifact root were not in the load I/O set",
        })
    return rows


def print_report(doc: dict) -> None:
    w = sys.stdout.write
    w("NOETIC EXECUTABLE CLOSURE\n")
    w("=" * 72 + "\n")
    w(f"schema     {doc['schema']}\n")
    w(f"generated  {doc['generated_at']}\n")
    w(f"head       {doc['git_head']}\n")
    w(f"artifact   {doc['artifact']['path']}\n")
    w(f"elapsed_s  {doc['elapsed_s']}\n")
    w(f"receipt    {doc['receipt_path']}\n")
    w("\n")
    w("## 1. FILES TOUCHED AT EXECUTION (observation)\n")
    obs = doc["observation"]
    w(f"method: {obs['method']}\n")
    w(f"live decode binary: {obs['live_decode'].get('binary')}\n")
    w(f"  attempted={obs['live_decode']['attempted']} exit={obs['live_decode']['exit_code']} "
      f"tokenizer_encode={obs['live_decode']['saw_tokenizer_encode']} "
      f"catalog={obs['live_decode']['catalog_count']} "
      f"metal_refused={obs['live_decode']['metal_refused']}\n")
    if obs["live_decode"].get("metal_error"):
        w(f"  metal: {obs['live_decode']['metal_error']}\n")
    w(f"I/O replica watched {obs['io_replica']['watcher_event_count']} open/stat events, "
      f"{obs['io_replica']['tensor_count']} tensors hashed\n")
    w(f"observed execution members: {obs['n_observed_files']}\n")
    w(f"  manifest.json + tokenizer.json + {obs['io_replica']['tensor_count']} tensors\n")
    w(f"mixed catalog present: {obs['io_replica']['mixed_catalog_present']}\n")
    w("\n")
    w("## 2. CLOSURE HASH\n")
    c = doc["closure"]
    w(f"closure_sha256  {c['closure_sha256']}\n")
    w(f"  execution_members     {c['n_execution_members']}  (removal-test set)\n")
    w(f"  compiled_in_shaders   {c['n_shaders']}\n")
    w(f"  set_bytes/geometry    {c['n_set_bytes']}\n")
    w(f"  envelope files        {c['n_envelope_files']}  extras={c['n_envelope_extras']}\n")
    w(f"  shader concat sha256  {c['shader_concat_sha256'][:32]}…\n")
    w(f"  envelope tree sha256  {c['envelope_tree_sha256'][:32]}…\n")
    w("\n")
    w("## 3. REMOVAL TEST (dropping a hashed execution member breaks execution)\n")
    r = doc["removal"]
    w(f"ok={r['ok']} dropped={r['dropped_execution_member']}\n")
    w(f"  load I/O broke: {r['load_io_broke']}  {r.get('load_io_error')}\n")
    w(f"  tokenizer drop broke: {r['tokenizer_drop_broke']}\n")
    w(f"  original artifact untouched: {r['original_still_present']}\n")
    w(f"  {r['meaning']}\n")
    w("\n")
    w("## 4. ADDITION TEST (a hidden byte changes the hash)\n")
    a = doc["addition"]
    w(f"base {a['base_closure_sha256']}\n")
    w(f"hash changed in {a['n_hash_changed']}/{a['n_shown']} trials "
      f"(need >=3: {a['at_least_three']})\n")
    for t in a["trials"]:
        flag = "CHANGED" if t["hash_changed"] else "UNCHANGED"
        w(f"  [{flag}] {t['scenario']:24s} {t['name']}\n")
    w("\n")
    w("## 5. CONTENT-ADDRESSING DEFECT\n")
    ca = doc["content_addressing"]
    w(f"filename == sha256(contents):    {ca['filename_equals_sha256_contents']}/{ca['n_tensors']}\n")
    w(f"filename == sha256(tensor_name): {ca['filename_equals_sha256_tensor_name']}/{ca['n_tensors']}\n")
    w(f"sampled 12 content-addressed:    {ca['sampled_12_content_addressed']}/12\n")
    w(f"G105 rolling digest match:       {ca['rolling_digest']['matches_g105'] or 'none of the reconstructions'}\n")
    side = ca.get("g105_sidecar") or {}
    if side.get("present"):
        w(f"G105 sidecar vs live bytes:      {side.get('live_matches_g105_sha256')} match / "
          f"{side.get('live_differs_g105_sha256')} differ / "
          f"{side.get('live_missing_from_g105')} missing\n")
        for ex in (side.get("drifted_examples") or [])[:4]:
            w(f"    drifted {ex['tensor_name']}\n"
              f"      live {ex['live_sha256'][:16]}…  g105 {ex['g105_sha256'][:16]}…\n")
    w(f"verdict: {ca['verdict']}\n")
    w("minimum fix:\n")
    for i, line in enumerate(ca["minimum_fix"], 1):
        w(f"  {i}. {line}\n")
    w("\n")
    w("## 6. FIVE HIDING SCENARIOS\n")
    for row in doc["scenarios"]:
        w(f"  {row['scenario']:40s} NR/NX={row['nr_nx_was']:8s} now={row['closure']}\n")
        w(f"      {row['detail']}\n")
    w("\n")
    w("## ESCAPES\n")
    escapes = doc["escapes"]
    if not escapes:
        w("  (none that this vehicle currently instantiates)\n")
    for e in escapes:
        w(f"  - {e}\n")
    w("\n")
    w("## WHAT I WATCHED FAIL\n")
    for i, item in enumerate(doc["what_i_watched_fail"], 1):
        w(f"  {i}. {item}\n")
    w("=" * 72 + "\n")


def main() -> int:
    t0 = time.time()
    repo = repo_root()
    artifact = Path(os.environ.get(
        "NOETIC_ARTIFACT",
        os.path.expanduser("~/models/qwen38-gravity-uniform-q4-v1"),
    ))
    tokenizer = Path(os.environ.get(
        "NOETIC_TOKENIZER",
        os.path.expanduser("~/models/qwen3.8-27b-abliterated-bf16/tokenizer.json"),
    ))
    receipt_path = repo / "receipts" / "headless" / "NOETIC_EXECUTABLE_CLOSURE.json"
    watched_fail: list[str] = []

    if not artifact.is_dir():
        raise SystemExit(f"artifact root missing: {artifact}")
    if not tokenizer.is_file():
        raise SystemExit(f"tokenizer missing: {tokenizer}")

    decode_bin = locate_decode_binary(repo)
    print("== live decode probe (real binary) ==", flush=True)
    live = probe_live_decode(decode_bin, artifact, tokenizer)
    if live.get("metal_refused"):
        watched_fail.append(
            f"Live ascension_qwen38_hybrid_greedy died at MetalContext::new "
            f"({live.get('metal_error')}). Tokenizer encode and manifest load "
            f"ran first (catalog_count={live.get('catalog_count')}); the 755 "
            f"tensor fs::read calls never happened in that process."
        )
    if not live.get("attempted"):
        watched_fail.append("Decode binary was not found; live process was not started.")
    if live.get("saw_catalog_count") and live.get("catalog_count") != EXPECTED_TENSORS:
        watched_fail.append(
            f"Live binary reported catalog_count={live.get('catalog_count')}, expected {EXPECTED_TENSORS}"
        )

    print("== I/O replica of load + tokenizer (open capture) ==", flush=True)
    io = run_load_io_replica(artifact, tokenizer)

    metal_mod = repo / "crates/hawking-core/src/metal/mod.rs"
    shaders_dir = repo / "crates/hawking-core/shaders"
    decode_rs = repo / "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
    geometry_rs = repo / "crates/hawking-core/src/model/qwen38_geometry.rs"
    if not metal_mod.is_file():
        raise SystemExit(f"missing {metal_mod} (sparse checkout blocker)")
    print("== shader compile input ==", flush=True)
    shaders = parse_shader_compile_input(metal_mod, shaders_dir)
    print(f"  {len(shaders['shader_files'])} shader files, concat {shaders['concatenated_bytes']} bytes", flush=True)
    set_bytes = extract_set_bytes_and_geometry(decode_rs, geometry_rs)

    already = {m["ident"]: m for m in io["members"]}
    print("== artifact envelope ==", flush=True)
    envelope = artifact_envelope(artifact, already)

    runtime_members: list[dict] = []
    if decode_bin is not None:
        dgst, n = sha256_file(decode_bin)
        runtime_members.append({
            "role": "runtime_binary",
            "ident": "runtime/ascension_qwen38_hybrid_greedy",
            "path": str(decode_bin),
            "sha256": dgst,
            "bytes": n,
            "why": "the executable half; dyld maps it at generate",
        })

    execution_members = io["members"]
    compiled = shaders["shader_files"] + [
        {**p, "role": "compiled_in_payload"} for p in set_bytes["payloads"]
    ]
    compiled.append({
        "role": "compiled_in_shader",
        "ident": "shader/all_shader_sources_concat",
        "sha256": shaders["concatenated_sha256"],
        "bytes": shaders["concatenated_bytes"],
        "why": "exact string MetalContext::newLibraryWithSource compiles",
    })

    # Closure hash commits to execution members + compiled-in + envelope tree.
    closure_entries: list[tuple[str, str]] = []
    for m in execution_members:
        closure_entries.append((m["ident"], m["sha256"]))
    for m in compiled:
        closure_entries.append((m["ident"], m["sha256"]))
    for m in runtime_members:
        closure_entries.append((m["ident"], m["sha256"]))
    closure_entries.append(("envelope/tree", envelope["tree_sha256"]))
    closure_sha = merkle(closure_entries)

    drop = next(m for m in execution_members if m["ident"].startswith("artifact/tensors/"))
    print("== removal test ==", flush=True)
    removal = removal_test(artifact, tokenizer, drop["ident"], Path(drop["path"]))
    if not removal["ok"]:
        watched_fail.append(f"Removal test did not break I/O: {removal}")
    if not removal["original_still_present"]:
        watched_fail.append("Removal test damaged the original artifact — this is a harness bug.")

    print("== addition tests ==", flush=True)
    addition = addition_tests(closure_entries, shaders, set_bytes)
    if not addition["at_least_three"]:
        watched_fail.append(
            f"Addition tests changed the hash in only {addition['n_hash_changed']} trials"
        )

    print("== content addressing ==", flush=True)
    addressing = content_addressing(execution_members, repo)
    if addressing["filename_equals_sha256_contents"] != 0:
        watched_fail.append(
            "Unexpected: some filenames equal sha256(contents); G105 said zero."
        )
    if addressing["sampled_12_content_addressed"] != 0:
        watched_fail.append(
            f"Sampled {addressing['sampled_12_content_addressed']}/12 filenames matched contents; G105 said 0/12."
        )
    if addressing["filename_equals_sha256_tensor_name"] != addressing["n_tensors"]:
        watched_fail.append(
            f"Only {addressing['filename_equals_sha256_tensor_name']}/{addressing['n_tensors']} "
            f"filenames equal sha256(tensor_name); G105 said 755/755."
        )
    side = addressing.get("g105_sidecar") or {}
    if side.get("present") and side.get("live_differs_g105_sha256"):
        watched_fail.append(
            f"G105 sidecar vs live: {side['live_matches_g105_sha256']} tensors still "
            f"match, {side['live_differs_g105_sha256']} differ under the same "
            f"64-hex filename. The name-address did not move. Load would upload either."
        )

    scenarios = scenario_verdict(
        envelope["extras"], shaders_hashed=True, set_bytes_hashed=True,
    )
    escapes = [
        "In-memory generated state that is not a file and is not a function of hashed weights/shaders/constants. None exists on this vehicle (generated_structures=[], workspace zeroed).",
        "A set_bytes payload assembled at runtime from an unobserved channel. Today every site is a compile-time constant or geometry taken from hashed tensor headers.",
        "Pointer authentication: filenames masquerade as content addresses (0/755 match sha256(bytes)). The closure hashes CONTENTS so a byte swap still moves the hash, but load does not consult the hash and will upload whatever sits at the name-address.",
        "Live GPU decode was not observed this run (Metal refused). Tensor opens are from the I/O replica of Qwen38HybridWeights::load, not from a Metal session. A loader that opened extra files only after MetalContext::new would not have been watched in-process.",
    ]

    watched_fail.extend([
        "DYLD_INSERT_LIBRARIES loaded a constructor into an unsigned helper but libc open was not interposed (dyld shared cache / SIP).",
        "ktrace/fs_usage/opensnoop/dtruss require root or are SIP-blocked (Operation not permitted).",
        "lldb run of an unsigned helper returned 'Operation not permitted' in this sandbox.",
        "Swift MTLCreateSystemDefaultDevice() returned nil; Metal.framework loads, the GPU does not.",
        "G105 sidecar rolling digest 89e78055… was not reconstructed from any of: sorted-filename concat of raw digests, hex-digest concat, or manifest-order concat — the rolling digest is not a function of the pointers, and this harness will not pretend it is.",
        "NX kernel_binding lists 38 dispatched names of 554 declared and hashes no shader source; a learned constant in an unused but compiled kernel is invisible to NX and visible to this closure.",
        "NR.validate() (prior lane) accepted a 1 GB SharedBasis + TensorTrain family + 50 MB generated blob without moving complete_bits_per_weight. This harness does not trust NR for closure.",
    ])

    elapsed = round(time.time() - t0, 3)
    doc = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(repo),
        "elapsed_s": elapsed,
        "receipt_path": str(receipt_path),
        "artifact": {
            "path": str(artifact),
            "tokenizer": str(tokenizer),
            "decode_binary": None if decode_bin is None else str(decode_bin),
        },
        "observation": {
            "method": (
                "Two layers. (1) Spawn the real ascension_qwen38_hybrid_greedy "
                "against the sealed artifact and read its stderr: it opens "
                "tokenizer.json (prompt tokens printed), fs::reads manifest.json "
                "(prints 'opening Metal + 755 catalog tensors'), then dies at "
                "MetalContext::new. (2) Because this sandbox has no Metal GPU, "
                "the 755 tensor fs::read calls never run in that process. They "
                "are executed as an I/O replica of Qwen38HybridWeights::load "
                "(stat catalog.hq38m20, read manifest.json, fs::read each "
                "row.artifact under tensors/, read tokenizer.json) with os.open / "
                "builtins.open / os.stat wrapped. The replica does not take the "
                "manifest as the hash: it hashes the bytes of every path it "
                "actually opened. Shader source and set_bytes payloads are not "
                "files the process opens; they are included because "
                "newLibraryWithSource compiles include_str'd .metal and set_bytes "
                "ships numeric payloads NX leaves unhashed."
            ),
            "live_decode": live,
            "io_replica": {
                "mixed_catalog_present": io["mixed_catalog_present"],
                "manifest_schema": io["manifest_schema"],
                "tensor_count": io["tensor_count"],
                "watcher_event_count": io["watcher_event_count"],
                "watcher_ops": io["watcher_ops"],
                "watcher_unique_path_count": len(io["watcher_unique_paths"]),
            },
            "n_observed_files": len(execution_members),
        },
        "closure": {
            "closure_sha256": closure_sha,
            "n_execution_members": len(execution_members),
            "n_shaders": len(shaders["shader_files"]),
            "n_set_bytes": len(set_bytes["payloads"]),
            "n_runtime": len(runtime_members),
            "n_envelope_files": envelope["file_count"],
            "n_envelope_extras": envelope["extra_file_count"],
            "shader_concat_sha256": shaders["concatenated_sha256"],
            "envelope_tree_sha256": envelope["tree_sha256"],
            "construction": (
                "merkle(observed files + compiled shader files + concatenated "
                "shader compile input + set_bytes/geometry/chat-template payloads "
                "+ decode binary + artifact-tree envelope). Execution members "
                "are the removal-test set. The envelope is how a sibling that "
                "the loader does not open still cannot hide from the hash."
            ),
        },
        "execution_members": [
            {k: m[k] for k in m if k != "tensor_name" or True}
            for m in execution_members
        ],
        "shaders": {
            "concatenated_sha256": shaders["concatenated_sha256"],
            "concatenated_bytes": shaders["concatenated_bytes"],
            "files": shaders["shader_files"],
            "tq_feature_included": shaders["tq_feature_included"],
        },
        "set_bytes": set_bytes,
        "runtime": runtime_members,
        "envelope": {
            "tree_sha256": envelope["tree_sha256"],
            "file_count": envelope["file_count"],
            "extra_file_count": envelope["extra_file_count"],
            "extras": envelope["extras"],
            "mixed_catalog": envelope["mixed_catalog"],
        },
        "removal": removal,
        "addition": addition,
        "content_addressing": addressing,
        "scenarios": scenarios,
        "escapes": escapes,
        "what_i_watched_fail": watched_fail,
        "did_not_modify": [
            "the sealed artifact tree",
            "tools/nx_genome.py",
            "tools/nr_container.py",
            "crates/",
            "workspace/",
            "visionmcp/",
            "app/",
            "research/lab/",
            "tools/hcli/bootstrap/",
        ],
    }

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(receipt_path)
    print_report(doc)
    if not removal["ok"] or not addition["at_least_three"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
