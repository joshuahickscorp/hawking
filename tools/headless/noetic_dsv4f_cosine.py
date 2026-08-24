#!/usr/bin/env python3
"""NNS-004 cheap check: DSV4F routed-expert pairwise cosine, scale-aware.

Q80/F0/F1 shared-basis is dead (Q80 L10 n=96 gate mean cosine 0.00414). That
number was incorrectly listed as a DSV4F fact. This tool measures DSV4F, or
refuses to measure if the weight body is missing/truncated.

Does not spawn a model server. Does not score synthetic activations. Does not
treat the layer-4 diagnostic or the activation-X capture as the artifact.

Run from the repository root:

    python3 tools/headless/noetic_dsv4f_cosine.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "hawking.headless.noetic_dsv4f_cosine.v1"
NNS_ID = "NNS-004"
NS_ID = "NS-010"

# NS-010 reopen: a shared-basis pack is a new premise only if mean pairwise
# cosine is materially above ~0.05. Below that, DSV4F looks like Q80.
COSINE_WORTH_TESTING = 0.05
Q80_GATE_PAIRWISE_MEAN = 0.004142791032791138
Q80_UP_PAIRWISE_MEAN = -5.968913319520652e-05
ACTIVATION_CONSTANT_MEAN_NULL = 0.898  # GLM/family activation null; NOT a weight null

PINNED_REPO = "deepseek-ai/DeepSeek-V4-Flash"
PINNED_REV = "60d8d70770c6776ff598c94bb586a859a38244f1"
EXPECTED_MANIFEST_SEAL = "ba9039bfe71328e2e47ced782bd1f931e2d412055382da0ea669092c1d90bfed"
EXPECTED_CHUNK_TREE_SHA = "15e00fb1b91ac074b7f24686de4e289f76d66eb1c3fb4ad643de027adc78ca13"
EXPECTED_SEAL_SHA = "f9b928f1cbb96e8f7238a3983a3f68ef3dc2946cca66fdf3e098781a533a5b0d"
EXPECTED_CHUNK_COUNT = 69837
EXPECTED_TENSOR_COUNT = 69187
EXPECTED_TENSOR_BYTES = 159609485896
EXPECTED_N_LAYERS = 43
EXPECTED_N_ROUTED = 256
EXPECTED_ORGANS = ("w1", "w2", "w3")
W1_ROWS, W1_K = 2048, 4096
W2_ROWS, W2_K = 4096, 2048
FP4_BLOCK = 32
E2M1FN = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

WEIGHT_RE = re.compile(
    r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$"
)
SCALE_RE = re.compile(
    r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.scale$"
)

# Probe chunk: layers.0.ffn.experts.0.w1.weight (from the sealed range journal).
PROBE_CHUNK_REL = "chunks/17/177ac1285aea3d72fd7e83bbd103d701499001c8a2ecd8950cb51cf74c53b507"
PROBE_CHUNK_SHA = "177ac1285aea3d72fd7e83bbd103d701499001c8a2ecd8950cb51cf74c53b507"
PROBE_CHUNK_BYTES = 4194304

DELETION_RECEIPT = "receipts/ascent-2026-08-16/DSV4F_SEALED_SCIENCE_WEIGHTS_DELETED.json"
DELETION_BYTES_FREED_GIB = 149

REPO = Path(__file__).resolve().parents[2]
RECEIPT_PATH = REPO / "receipts" / "headless" / "NOETIC_DSV4F_COSINE.json"

DEFAULT_ARTIFACTS = [
    Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"),
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"),
    Path.home() / "models" / "DeepSeek-V4-Flash",
    Path.home() / "models" / "deepseek-v4-flash",
    Path.home() / ".cache" / "huggingface" / "hub" / "models--deepseek-ai--DeepSeek-V4-Flash",
]

REFUSE_PARTIAL = [
    Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/streamed-layer4-diagnostic.gravity"),
    Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/activation-x-capture-3k"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def sha256_head_tail(path: Path, span: int = 8 << 20) -> str:
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(span))
        if size > span:
            f.seek(max(0, size - span))
            h.update(f.read(span))
    return h.hexdigest()


def file_record(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    rec = {
        "path": str(path),
        "exists": True,
        "is_dir": path.is_dir(),
        "bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    if path.is_file() and st.st_size > 0:
        rec["identity_kind"] = "sha256_size_head_tail_8MiB"
        rec["identity"] = sha256_head_tail(path)
    return rec


def decode_e8m0fnu(bits: np.ndarray) -> np.ndarray:
    """UE8M0 (E8M0FNU): exponent-only float, NaN at 0xFF, 2^-127 at 0."""
    bits_u = np.asarray(bits, dtype=np.uint8)
    if np.any(bits_u == 0xFF):
        raise ValueError("E8M0FNU contains its NaN encoding")
    out = np.empty(bits_u.shape, dtype=np.float32)
    zero = bits_u == 0
    out[zero] = np.float32(2.0 ** -127)
    nz = ~zero
    out[nz] = np.frombuffer(
        (bits_u[nz].astype(np.uint32) << 23).astype(np.uint32).tobytes(),
        dtype=np.float32,
    ).reshape(out[nz].shape)
    return out


def decode_fp4_e2m1fn_x2_ue8m0(
    packed: np.ndarray, scale: np.ndarray, rows: int, logical_k: int
) -> np.ndarray:
    """Decode native DSV4F expert weights: two E2M1FN nibbles/byte, UE8M0 per 32-K."""
    packed = np.asarray(packed, dtype=np.uint8)
    scale = np.asarray(scale, dtype=np.uint8)
    packed_k = logical_k // 2
    scale_cols = logical_k // FP4_BLOCK
    if packed.size != rows * packed_k:
        raise ValueError(f"packed size {packed.size} != {rows}*{packed_k}")
    if scale.size != rows * scale_cols:
        raise ValueError(f"scale size {scale.size} != {rows}*{scale_cols}")
    packed = packed.reshape(rows, packed_k)
    lo = packed & np.uint8(0x0F)
    hi = packed >> np.uint8(4)
    vals = np.empty((rows, logical_k), dtype=np.float32)
    vals[:, 0::2] = E2M1FN[lo]
    vals[:, 1::2] = E2M1FN[hi]
    scales = decode_e8m0fnu(scale).reshape(rows, scale_cols)
    vals *= np.repeat(scales, FP4_BLOCK, axis=1)
    return vals


def cosine_and_scale(a: np.ndarray, b: np.ndarray) -> dict:
    """Direction cosine, scale ratio, and scale-aware product.

    cosine is scale-invariant: 0.01*W scores 1.0.
    scale_ratio = min(||a||,||b||)/max(||a||,||b||) is 0.01 for that pair.
    scale_aware_cosine = cosine * scale_ratio equals 1 iff the tensors match.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    cosine = float(np.dot(a, b) / ((na * nb) + 1e-30))
    scale_ratio = float(min(na, nb) / (max(na, nb) + 1e-30))
    return {
        "cosine": cosine,
        "scale_ratio": scale_ratio,
        "scale_aware_cosine": cosine * scale_ratio,
        "norm_a": na,
        "norm_b": nb,
        "rel_l2": float(np.linalg.norm(a - b) / (max(na, nb) + 1e-30)),
    }


def offdiag_distribution(mat: np.ndarray) -> dict:
    n = int(mat.shape[0])
    if n < 2:
        return {"n_pairs": 0}
    mask = ~np.eye(n, dtype=bool)
    x = np.asarray(mat[mask], dtype=np.float64)
    return {
        "n": n,
        "n_pairs": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        "frac_gt_0.05": float((x > 0.05).mean()),
        "frac_gt_0.10": float((x > 0.10).mean()),
        "frac_gt_0.30": float((x > 0.30).mean()),
        "frac_gt_0.50": float((x > 0.50).mean()),
        "frac_gt_0.70": float((x > 0.70).mean()),
    }


def pairwise_mats(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """F: (n, d). Returns cosine, scale_ratio, scale_aware, norms."""
    F64 = np.asarray(F, dtype=np.float64)
    norms = np.linalg.norm(F64, axis=1)
    unit = F64 / (norms[:, None] + 1e-30)
    cosine = unit @ unit.T
    sr = np.minimum.outer(norms, norms) / (np.maximum.outer(norms, norms) + 1e-30)
    return cosine, sr, cosine * sr, norms


def method_calibration() -> dict:
    """Show the metric rejecting 0.01*W, and the Gaussian null for these shapes.

    This is a method check, not a codec ranking on synthetic activations.
    """
    rng = np.random.default_rng(20260823)
    # Small explicit trap (the campaign that scored 1.000000).
    W = rng.standard_normal((64, 128)).astype(np.float32)
    scaled = np.float32(0.01) * W
    trap = cosine_and_scale(W, scaled)
    identical = cosine_and_scale(W, W)
    orthogonal = cosine_and_scale(W, rng.standard_normal(W.shape).astype(np.float32))

    # Same-shape Gaussian null for one expert matrix, flattened.
    # Analytic: unit Gaussians in R^D have pairwise cosine ~ N(0, 1/D).
    shapes = {
        "w1": (W1_ROWS, W1_K),
        "w2": (W2_ROWS, W2_K),
        "w3": (W1_ROWS, W1_K),
    }
    nulls = {}
    for organ, (rows, k) in shapes.items():
        d = rows * k
        analytic_std = float(d ** -0.5)
        # Empirical on a cheap subsample of random unit vectors of dim D.
        # Full 256 x D would be 8 GiB; n=12 is enough to confirm the 1/sqrt(D) law.
        n_emp = 12
        G = rng.standard_normal((n_emp, d)).astype(np.float32)
        cos, sr, sa, _ = pairwise_mats(G)
        nulls[organ] = {
            "shape": [rows, k],
            "D": d,
            "analytic_pairwise_cosine_mean": 0.0,
            "analytic_pairwise_cosine_std": analytic_std,
            "analytic_p95_abs": float(1.6448536269514722 * analytic_std),
            "empirical_n_vectors": n_emp,
            "empirical_cosine": offdiag_distribution(cos),
            "empirical_scale_aware": offdiag_distribution(sa),
            "note": (
                "Weight pairwise cosine null is ~0 with std 1/sqrt(D), D=8.39e6. "
                "The raw-activation constant-mean null 0.898 does not apply here."
            ),
        }

    # Full-size 0.01*W trap on one expert-shaped matrix (32 MiB), not a proxy X.
    W_full = rng.standard_normal((W1_ROWS, W1_K)).astype(np.float32)
    trap_full = cosine_and_scale(W_full, np.float32(0.01) * W_full)

    rejects_trap = (
        trap["cosine"] > 0.999
        and trap["scale_aware_cosine"] < 0.02
        and trap_full["cosine"] > 0.999
        and trap_full["scale_aware_cosine"] < 0.02
    )
    return {
        "trap_0.01_W_small": trap,
        "trap_0.01_W_expert_shaped_2048x4096": trap_full,
        "identical": identical,
        "orthogonal_draw": {k: orthogonal[k] for k in ("cosine", "scale_aware_cosine", "rel_l2")},
        "rejects_deliberately_scaled_artifact": bool(rejects_trap),
        "law": (
            "Cosine is scale-invariant. scale_aware_cosine = cosine * min(n,N)/max(n,N) "
            "scores ~0.01 on 0.01*W while cosine scores ~1.0. A fidelity campaign that "
            "used cosine alone called 0.01*W HEALTHY (gravity_doctor_gate._gain)."
        ),
        "activation_null_does_not_apply": {
            "constant_mean_null_cosine": ACTIVATION_CONSTANT_MEAN_NULL,
            "applies_to": "raw activation cosine on this family, not weight pairwise cosine",
        },
        "null_by_organ": nulls,
        "decision_threshold": {
            "source": "NS-010 retry_when",
            "cosine_mean_materially_above": COSINE_WORTH_TESTING,
            "q80_gate_mean_must_not_transfer": Q80_GATE_PAIRWISE_MEAN,
        },
    }


def candidate_chunk_roots(artifact: Path) -> list[Path]:
    env = os.environ.get("HAWKING_DSV4F_CHUNKS")
    roots = []
    if env:
        roots.append(Path(env).expanduser())
    roots.extend(
        [
            artifact / "chunks",
            artifact.parent / "chunks",
            Path.home() / ".cache" / "hawking" / "dsv4f-chunks",
            Path.home() / ".cache" / "hawking" / "dsv4f-admission" / "chunks",
            Path("/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity/chunks"),
        ]
    )
    # unique, preserve order
    out = []
    seen = set()
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s)
            out.append(r)
    return out


def count_regular_files(root: Path, limit: int | None = None) -> tuple[int, int]:
    n = 0
    nbytes = 0
    if not root.is_dir():
        return 0, 0
    for dirpath, dirnames, filenames in os.walk(root):
        # do not follow symlinks out of the tree
        dirnames[:] = [
            d for d in dirnames
            if not os.path.islink(os.path.join(dirpath, d))
        ]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                n += 1
                nbytes += int(st.st_size)
                if limit is not None and n >= limit:
                    return n, nbytes
    return n, nbytes


def locate_artifact() -> dict:
    env = os.environ.get("HAWKING_DSV4F_ARTIFACT")
    searched = []
    chosen = None
    if env:
        p = Path(env).expanduser()
        searched.append({"path": str(p), "exists": p.exists(), "via": "HAWKING_DSV4F_ARTIFACT"})
        if p.exists():
            chosen = p
    if chosen is None:
        for p in DEFAULT_ARTIFACTS:
            rec = {"path": str(p), "exists": p.exists(), "via": "default_search"}
            searched.append(rec)
            if chosen is None and p.exists() and (p / "manifest.json").is_file():
                chosen = p
    return {"searched": searched, "chosen": str(chosen) if chosen else None, "path": chosen}


def refuse_partial_artifacts() -> list[dict]:
    rows = []
    for p in REFUSE_PARTIAL:
        rec = file_record(p)
        rec["refused_as_measurement_subject"] = True
        if "layer4" in str(p):
            chunks = p / "chunks"
            n, nbytes = count_regular_files(chunks)
            rec["chunks_regular_files"] = n
            rec["chunks_bytes"] = nbytes
            rec["why_refused"] = (
                "Layer-4 diagnostic, not the 43-layer family. Chunk tree has "
                f"{n} regular files / {nbytes} bytes (empty prefixes are not a body). "
                "Acceptance forbids measuring a partial artifact."
            )
        else:
            rec["why_refused"] = (
                "Activation-X capture (hidden states), not expert weights. "
                "NS-009: never evaluate compression on (even real-captured) X as a "
                "stand-in for a weight pairwise-cosine measurement."
            )
        rows.append(rec)
    return rows


def verify_artifact(artifact: Path) -> dict:
    manifest = artifact / "manifest.json"
    ranges = artifact / "stream-ranges.jsonl"
    journal = artifact / "stream-journal.json"
    restart = artifact / "restart-receipt.json"
    chunks_dir = artifact / "chunks"
    metadata = artifact / "metadata"

    files = {
        "manifest.json": file_record(manifest),
        "stream-ranges.jsonl": file_record(ranges),
        "stream-journal.json": file_record(journal),
        "restart-receipt.json": file_record(restart),
        "metadata": file_record(metadata),
        "chunks": file_record(chunks_dir),
    }

    journal_obj = None
    if journal.is_file():
        try:
            journal_obj = json.loads(journal.read_text())
        except json.JSONDecodeError as exc:
            files["stream-journal.json"]["parse_error"] = str(exc)

    restart_obj = None
    if restart.is_file():
        try:
            restart_obj = json.loads(restart.read_text())
        except json.JSONDecodeError as exc:
            files["restart-receipt.json"]["parse_error"] = str(exc)

    # Cheap manifest header check without decoding 151 MB of tensor table:
    # confirm schema/seal/chunk_count via a bounded scan of the first 8 KiB
    # plus a targeted Python parse of the small sidecar files. Full tensor
    # table is in stream-ranges.jsonl, which we census separately.
    manifest_head = {}
    if manifest.is_file():
        # The file is one JSON object. Loading it is ~151 MB / a few seconds.
        # We only need artifact/architecture/seal — load once.
        with manifest.open() as f:
            man = json.load(f)
        art = man.get("artifact") or {}
        arch = man.get("architecture") or {}
        inf = (arch.get("inference_config") or {})
        manifest_head = {
            "schema": man.get("schema"),
            "status": man.get("status"),
            "seal_sha256": man.get("seal_sha256"),
            "artifact": {
                "content_addressed_chunk_count": art.get("content_addressed_chunk_count"),
                "content_addressed_chunk_sha256": art.get("content_addressed_chunk_sha256"),
                "total_tensor_bytes": art.get("total_tensor_bytes"),
                "format": art.get("format"),
            },
            "architecture": {
                "model_type": arch.get("model_type"),
                "layer_count": arch.get("layer_count"),
                "tensor_count": arch.get("tensor_count"),
                "n_routed_experts": inf.get("n_routed_experts"),
                "n_activated_experts": inf.get("n_activated_experts"),
                "n_shared_experts": inf.get("n_shared_experts"),
                "expert_dtype": inf.get("expert_dtype"),
                "dtype": inf.get("dtype"),
                "scale_fmt": inf.get("scale_fmt"),
                "moe_inter_dim": inf.get("moe_inter_dim"),
                "dim": inf.get("dim"),
            },
            "source_repository": (man.get("source") or {}).get("repository")
            or ((journal_obj or {}).get("source") or {}).get("repository"),
            "source_revision": (man.get("source") or {}).get("revision")
            or ((journal_obj or {}).get("source") or {}).get("revision"),
        }
        # Drop the 151 MB object.
        del man

    # Chunk body.
    chunk_search = []
    probe_hits = []
    present_files = 0
    present_bytes = 0
    body_root = None
    for root in candidate_chunk_roots(artifact):
        exists = root.is_dir()
        n, nbytes = count_regular_files(root) if exists else (0, 0)
        probe = root.parent / PROBE_CHUNK_REL if root.name == "chunks" else root / PROBE_CHUNK_REL
        # candidate_chunk_roots already includes .../chunks, so probe is root/17/<sha>
        probe = root / "17" / PROBE_CHUNK_SHA
        probe_exists = probe.is_file()
        rec = {
            "root": str(root),
            "exists": exists,
            "regular_files": n,
            "bytes": nbytes,
            "probe_chunk": str(probe),
            "probe_exists": probe_exists,
        }
        if probe_exists:
            st = probe.stat()
            rec["probe_bytes"] = int(st.st_size)
            rec["probe_bytes_match"] = int(st.st_size) == PROBE_CHUNK_BYTES
            if st.st_size <= 64 * 1024 * 1024:
                h = hashlib.sha256()
                with probe.open("rb") as f:
                    for block in iter(lambda: f.read(8 << 20), b""):
                        h.update(block)
                rec["probe_sha256"] = h.hexdigest()
                rec["probe_sha_match"] = h.hexdigest() == PROBE_CHUNK_SHA
            probe_hits.append(rec)
        chunk_search.append(rec)
        if exists and n > present_files:
            present_files, present_bytes, body_root = n, nbytes, root

    intact = (
        files["manifest.json"].get("exists")
        and files["stream-ranges.jsonl"].get("exists")
        and present_files == EXPECTED_CHUNK_COUNT
        and present_bytes > 0
        and bool(probe_hits)
        and all(h.get("probe_bytes_match") and h.get("probe_sha_match") for h in probe_hits[:1])
    )
    truncated = present_files > 0 and present_files < EXPECTED_CHUNK_COUNT
    missing_body = present_files == 0

    reasons = []
    if missing_body:
        reasons.append(
            f"chunk body absent: 0 regular files under any search root "
            f"(expected {EXPECTED_CHUNK_COUNT} content-addressed chunks, "
            f"{EXPECTED_TENSOR_BYTES} tensor bytes, ~{DELETION_BYTES_FREED_GIB} GiB). "
            f"Cited: {DELETION_RECEIPT} status=SEALED_SCIENCE_RETAINED_WEIGHTS_DELETED, "
            f"deleted path=.../full-43-layer-stream.gravity/chunks/, "
            f"source_parent_retained=false so this was the only local copy."
        )
    if truncated:
        reasons.append(
            f"chunk tree truncated: {present_files} files / {present_bytes} bytes "
            f"vs expected {EXPECTED_CHUNK_COUNT} / {EXPECTED_TENSOR_BYTES}. "
            "STOP — do not measure a partial body."
        )
    if manifest_head:
        seal = manifest_head.get("seal_sha256")
        if seal and seal != EXPECTED_MANIFEST_SEAL:
            reasons.append(f"manifest seal mismatch: {seal} != {EXPECTED_MANIFEST_SEAL}")
        art = manifest_head.get("artifact") or {}
        if art.get("content_addressed_chunk_count") not in (None, EXPECTED_CHUNK_COUNT):
            reasons.append(
                f"manifest chunk_count {art.get('content_addressed_chunk_count')} "
                f"!= {EXPECTED_CHUNK_COUNT}"
            )
        src_rev = manifest_head.get("source_revision") or (
            (journal_obj or {}).get("source") or {}
        ).get("revision")
        if src_rev and src_rev != PINNED_REV:
            reasons.append(f"revision {src_rev} != pinned {PINNED_REV}")

    # Journal append is a known defect: 2x sealed prefix. Record, do not
    # treat line count as chunk count.
    range_lines = 0
    if ranges.is_file():
        with ranges.open("rb") as f:
            range_lines = sum(1 for _ in f)

    status = "INTACT" if intact else ("TRUNCATED" if truncated else "BODY_ABSENT")
    return {
        "artifact_path": str(artifact),
        "status": status,
        "intact": bool(intact),
        "truncated": bool(truncated),
        "missing_body": bool(missing_body),
        "stop_measurement": not intact,
        "reasons": reasons,
        "files": files,
        "manifest_head": manifest_head,
        "journal_source": (journal_obj or {}).get("source"),
        "journal_status": (journal_obj or {}).get("status"),
        "journal_storage_policy": (journal_obj or {}).get("storage_policy"),
        "restart_source_parent_retained": (restart_obj or {}).get("source_parent_retained"),
        "restart_range_count": (restart_obj or {}).get("range_count"),
        "stream_ranges_lines_on_disk": range_lines,
        "stream_ranges_line_count_is_not_chunk_count": {
            "lines": range_lines,
            "sealed_prefix": EXPECTED_CHUNK_COUNT,
            "note": (
                "Journal was appended (≈2x). Admission uses a clone-view of the "
                "sealed prefix. Counting jsonl lines as chunks would double the body."
            ),
        },
        "chunk_search": chunk_search,
        "body_root": str(body_root) if body_root else None,
        "chunks_regular_files": present_files,
        "chunks_bytes": present_bytes,
        "probe_hits": probe_hits,
        "expected": {
            "repository": PINNED_REPO,
            "revision": PINNED_REV,
            "manifest_seal_sha256": EXPECTED_MANIFEST_SEAL,
            "content_addressed_chunk_sha256": EXPECTED_CHUNK_TREE_SHA,
            "seal_sha256": EXPECTED_SEAL_SHA,
            "chunk_count": EXPECTED_CHUNK_COUNT,
            "tensor_count": EXPECTED_TENSOR_COUNT,
            "total_tensor_bytes": EXPECTED_TENSOR_BYTES,
            "n_layers": EXPECTED_N_LAYERS,
            "n_routed_experts": EXPECTED_N_ROUTED,
        },
        "deletion_record": {
            "receipt": DELETION_RECEIPT,
            "date": "2026-08-16",
            "authority": "user steer S003 section 1 - PRESERVE SCIENCE, DELETE WEIGHTS",
            "deleted": "workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity/chunks/",
            "bytes_freed_gib": DELETION_BYTES_FREED_GIB,
            "rebuild": (
                "tools/condense/.venv/bin/python tools/condense/deepseek_v4_gravity.py "
                "build-full --artifact-dir <ARTIFACT_DIR> "
                "--workspace-root /Users/scammermike/Downloads/hawking "
                "--xet-root <NEW_EMPTY_XET_ROOT>"
            ),
        },
    }


def census_family(ranges_path: Path) -> dict:
    """Unique routed-expert weight/scale tensors from the range journal."""
    seen_weight = {}
    seen_scale = {}
    n_lines = 0
    n_parse_fail = 0
    with ranges_path.open() as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_parse_fail += 1
                continue
            name = row.get("tensor") or ""
            mw = WEIGHT_RE.match(name)
            if mw:
                key = name
                if key not in seen_weight:
                    seen_weight[key] = {
                        "scope": mw.group(1),
                        "layer": int(mw.group(2)),
                        "expert": int(mw.group(3)),
                        "organ": mw.group(4),
                        "bytes": int(row.get("bytes") or 0),
                        "row_count": int(row.get("row_count") or 0),
                        "chunk_relpath": row.get("chunk_relpath"),
                        "sha256": row.get("sha256"),
                    }
                continue
            ms = SCALE_RE.match(name)
            if ms:
                key = name
                if key not in seen_scale:
                    seen_scale[key] = {
                        "scope": ms.group(1),
                        "layer": int(ms.group(2)),
                        "expert": int(ms.group(3)),
                        "organ": ms.group(4),
                        "bytes": int(row.get("bytes") or 0),
                        "chunk_relpath": row.get("chunk_relpath"),
                        "sha256": row.get("sha256"),
                    }

    def summarize(table: dict) -> dict:
        layers = sorted({v["layer"] for v in table.values() if v["scope"] == "layers"})
        mtp_layers = sorted({v["layer"] for v in table.values() if v["scope"] == "mtp"})
        experts = sorted({v["expert"] for v in table.values() if v["scope"] == "layers"})
        organs = {}
        for v in table.values():
            if v["scope"] != "layers":
                continue
            organs.setdefault(v["organ"], {"n": 0, "bytes": 0, "row_count": v.get("row_count")})
            organs[v["organ"]]["n"] += 1
            organs[v["organ"]]["bytes"] += v["bytes"]
        return {
            "n_unique": len(table),
            "n_layers_base": len(layers),
            "layers_base": layers,
            "n_mtp_layers": len(mtp_layers),
            "n_expert_ids_base": len(experts),
            "organs_base": organs,
        }

    wsum = summarize(seen_weight)
    ssum = summarize(seen_scale)
    expected_weights = EXPECTED_N_LAYERS * EXPECTED_N_ROUTED * len(EXPECTED_ORGANS)
    return {
        "stream_ranges_path": str(ranges_path),
        "n_lines": n_lines,
        "n_parse_fail": n_parse_fail,
        "weights": wsum,
        "scales": ssum,
        "expected_base_weight_tensors": expected_weights,
        "base_weight_count_matches": wsum["n_unique"]
        >= expected_weights  # mtp extras allowed
        and wsum["n_layers_base"] == EXPECTED_N_LAYERS
        and wsum["n_expert_ids_base"] == EXPECTED_N_ROUTED,
        "shared_basis_target": {
            "what": (
                "Within each layer, the 256 routed experts' same organ "
                "(w1 / w2 / w3) as flattened decoded-fp32 tensors. "
                "w1=SwiGLU gate 2048x4096, w3=up 2048x4096, w2=down 4096x2048. "
                "94.35% of stored mass is routed experts."
            ),
            "not": (
                "shared_experts (n=1 per layer — no pairwise), MLA, indexer, "
                "activations, packed FP4 nibbles without UE8M0 decode."
            ),
        },
        "index": {
            # Keep a compact index of (layer, organ) -> expert chunk pointers
            # for the measurement path. Base layers only.
            "base": _index_base(seen_weight, seen_scale),
        },
    }


def _index_base(weights: dict, scales: dict) -> dict:
    idx = {}
    for name, w in weights.items():
        if w["scope"] != "layers":
            continue
        key = f"L{w['layer']}.{w['organ']}"
        slot = idx.setdefault(
            key,
            {
                "layer": w["layer"],
                "organ": w["organ"],
                "rows": w["row_count"],
                "experts": {},
            },
        )
        slot["experts"][str(w["expert"])] = {
            "weight_rel": w["chunk_relpath"],
            "weight_sha": w["sha256"],
            "weight_bytes": w["bytes"],
            "rows": w["row_count"],
        }
    for name, s in scales.items():
        if s["scope"] != "layers":
            continue
        key = f"L{s['layer']}.{s['organ']}"
        if key not in idx:
            continue
        e = idx[key]["experts"].get(str(s["expert"]))
        if e is not None:
            e["scale_rel"] = s["chunk_relpath"]
            e["scale_sha"] = s["sha256"]
            e["scale_bytes"] = s["bytes"]
    return idx


def load_chunk(artifact: Path, rel: str, expected_sha: str, expected_bytes: int) -> bytes:
    path = artifact / rel
    if not path.is_file():
        raise FileNotFoundError(rel)
    data = path.read_bytes()
    if expected_bytes and len(data) != expected_bytes:
        raise ValueError(f"{rel} size {len(data)} != {expected_bytes} (truncated chunk)")
    got = hashlib.sha256(data).hexdigest()
    if expected_sha and got != expected_sha:
        raise ValueError(f"{rel} sha256 {got} != {expected_sha}")
    return data


def organ_geometry(organ: str, rows_from_journal: int) -> tuple[int, int]:
    if organ == "w2":
        return W2_ROWS, W2_K
    return W1_ROWS, W1_K


def measure_family(artifact: Path, family: dict, body_root: Path) -> dict:
    """Decode FP4 experts and compute pairwise scale-aware cosine.

    One (layer, organ) at a time so peak RAM stays ~8 GiB (256 * 32 MiB).
    """
    idx = family["index"]["base"]
    groups = []
    all_cos = []
    all_sa = []
    aligned_subsets = []
    t0 = time.time()

    # Deterministic order: layer then organ.
    keys = sorted(idx, key=lambda k: (idx[k]["layer"], idx[k]["organ"]))
    for key in keys:
        g = idx[key]
        organ = g["organ"]
        layer = g["layer"]
        rows, logical_k = organ_geometry(organ, g["rows"])
        experts = sorted(int(e) for e in g["experts"])
        if len(experts) != EXPECTED_N_ROUTED:
            groups.append({
                "key": key,
                "layer": layer,
                "organ": organ,
                "n_experts_present": len(experts),
                "skipped": "expert count != 256; refusing a partial group",
            })
            continue
        flats = np.empty((len(experts), rows * logical_k), dtype=np.float32)
        for i, eid in enumerate(experts):
            rec = g["experts"][str(eid)]
            packed = np.frombuffer(
                load_chunk(
                    artifact if (artifact / rec["weight_rel"]).is_file() else body_root.parent,
                    rec["weight_rel"],
                    rec["weight_sha"],
                    rec["weight_bytes"],
                ),
                dtype=np.uint8,
            )
            scale = np.frombuffer(
                load_chunk(
                    artifact if (artifact / rec["scale_rel"]).is_file() else body_root.parent,
                    rec["scale_rel"],
                    rec["scale_sha"],
                    rec["scale_bytes"],
                ),
                dtype=np.uint8,
            )
            decoded = decode_fp4_e2m1fn_x2_ue8m0(packed, scale, rows, logical_k)
            flats[i] = decoded.reshape(-1)
        cos, sr, sa, norms = pairwise_mats(flats)
        # In-group 0.01*W trap on expert 0, using the real tensor.
        trap = cosine_and_scale(flats[0], np.float32(0.01) * flats[0])
        cd = offdiag_distribution(cos)
        sad = offdiag_distribution(sa)
        srd = offdiag_distribution(sr)
        all_cos.append(cd)
        all_sa.append(sad)
        row = {
            "key": key,
            "layer": layer,
            "organ": organ,
            "n_experts": len(experts),
            "shape": [rows, logical_k],
            "D": rows * logical_k,
            "pairwise_cosine": cd,
            "pairwise_scale_ratio": srd,
            "pairwise_scale_aware_cosine": sad,
            "norm_mean": float(np.mean(norms)),
            "norm_std": float(np.std(norms)),
            "real_tensor_0.01_trap": trap,
            "null_std": float((rows * logical_k) ** -0.5),
        }
        if cd.get("mean", 0) >= COSINE_WORTH_TESTING or cd.get("frac_gt_0.30", 0) > 0:
            aligned_subsets.append({
                "key": key,
                "cosine_mean": cd.get("mean"),
                "cosine_p95": cd.get("p95"),
                "cosine_max": cd.get("max"),
                "frac_gt_0.05": cd.get("frac_gt_0.05"),
                "frac_gt_0.30": cd.get("frac_gt_0.30"),
                "scale_aware_mean": sad.get("mean"),
                "scale_ratio_mean": srd.get("mean"),
            })
        groups.append(row)
        del flats

    def _pool(dists, field="mean"):
        vals = [d.get(field) for d in dists if d.get("n_pairs")]
        return vals

    cos_means = _pool(all_cos, "mean")
    sa_means = _pool(all_sa, "mean")
    # Pool pair fractions by weighting equally per group (each group has the
    # same n_pairs = 256*255).
    pooled = {
        "n_groups_measured": len(all_cos),
        "cosine_mean_of_group_means": float(np.mean(cos_means)) if cos_means else None,
        "cosine_max_of_group_means": float(np.max(cos_means)) if cos_means else None,
        "cosine_min_of_group_means": float(np.min(cos_means)) if cos_means else None,
        "scale_aware_mean_of_group_means": float(np.mean(sa_means)) if sa_means else None,
        "group_cosine_means": [
            {"key": groups[i]["key"], "mean": all_cos[i]["mean"],
             "p95": all_cos[i]["p95"], "max": all_cos[i]["max"]}
            for i in range(len(all_cos))
        ],
    }
    return {
        "elapsed_s": time.time() - t0,
        "groups": groups,
        "pooled": pooled,
        "aligned_subsets": aligned_subsets,
        "n_aligned_subsets": len(aligned_subsets),
    }


def decide(integrity: dict, calibration: dict, measurement: dict | None) -> dict:
    if not integrity.get("intact"):
        return {
            "verdict": "BODY_ABSENT_UNMEASURED" if integrity.get("missing_body") else "TRUNCATED_UNMEASURED",
            "worth_testing_or_dead": None,
            "why_not_dead": (
                "Issuing DEAD without a DSV4F cosine would transfer Q80's 0.00414. "
                "That transfer is the NNS-004 fallacy this lane exists to refuse. "
                "Shared-basis on DSV4F is unmeasured, not refuted."
            ),
            "deciding_number": {
                "chunks_regular_files": integrity.get("chunks_regular_files"),
                "expected_chunk_count": EXPECTED_CHUNK_COUNT,
                "chunks_bytes": integrity.get("chunks_bytes"),
                "expected_tensor_bytes": EXPECTED_TENSOR_BYTES,
            },
            "threshold": COSINE_WORTH_TESTING,
            "q80_number_must_not_transfer": Q80_GATE_PAIRWISE_MEAN,
        }

    pooled = (measurement or {}).get("pooled") or {}
    subsets = (measurement or {}).get("aligned_subsets") or []
    mean = pooled.get("cosine_mean_of_group_means")
    max_group = pooled.get("cosine_max_of_group_means")
    worth = (
        mean is not None
        and (
            mean >= COSINE_WORTH_TESTING
            or (max_group is not None and max_group >= COSINE_WORTH_TESTING)
            or bool(subsets)
        )
    )
    return {
        "verdict": "WORTH_TESTING" if worth else "DEAD",
        "worth_testing_or_dead": "WORTH_TESTING" if worth else "DEAD",
        "deciding_number": {
            "cosine_mean_of_group_means": mean,
            "cosine_max_of_group_means": max_group,
            "n_aligned_subsets": len(subsets),
            "threshold": COSINE_WORTH_TESTING,
            "null_std_w1": calibration["null_by_organ"]["w1"]["analytic_pairwise_cosine_std"],
        },
        "positive_would_mean": (
            "Shared basis is worth TESTING on DSV4F, not that it works. "
            "Storage-vs-active still applies: Q80 measured 0.6462 stored vs 2.518 active."
        ),
        "scale_aware_companion": pooled.get("scale_aware_mean_of_group_means"),
        "calibration_rejects_0.01_W": calibration.get("rejects_deliberately_scaled_artifact"),
    }


def watched_fail(integrity: dict, calibration: dict) -> list[dict]:
    return [
        {
            "what": "Transfer Q80 pairwise cosine 0.00414 onto DSV4F and call shared-basis dead",
            "why": (
                "NS-010 listed wording was 'DSV4F experts are mutually orthogonal (cos 0.004)'. "
                "The number is Q80 L10, 96 of 512 experts, receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json. "
                "Independently dead on foundry F0 (1e-4) and F1 (0.00166). DSV4F was not_measured_on."
            ),
        },
        {
            "what": "Measure the gravity directory without checking the chunk body",
            "why": (
                f"Metadata is present (manifest 150,990,326 B, ranges "
                f"{integrity.get('stream_ranges_lines_on_disk')} lines) and looks like a sealed artifact. "
                f"chunks/ is gone. {DELETION_RECEIPT} deleted the only local copy (source_parent_retained=false)."
            ),
        },
        {
            "what": "Treat streamed-layer4-diagnostic.gravity as a stand-in",
            "why": (
                "It has a chunks/ directory with 256 prefix folders and 0 regular files / 0 bytes. "
                "Even a full layer-4 body would be a partial family. Acceptance: STOP."
            ),
        },
        {
            "what": "Score pairwise cosine on packed FP4 nibbles, skipping UE8M0 scales",
            "why": "Direction in codebook space is not direction in the decoded GEMM.",
        },
        {
            "what": "Use cosine alone as a fidelity number (0.01*W → 1.000000)",
            "why": (
                f"Calibration: small trap cosine={calibration['trap_0.01_W_small']['cosine']:.6f} "
                f"scale_aware={calibration['trap_0.01_W_small']['scale_aware_cosine']:.6f}; "
                f"expert-shaped trap cosine={calibration['trap_0.01_W_expert_shaped_2048x4096']['cosine']:.6f} "
                f"scale_aware={calibration['trap_0.01_W_expert_shaped_2048x4096']['scale_aware_cosine']:.6f}. "
                "The scale-aware product is what rejects the scaled artifact."
            ),
        },
        {
            "what": "Quote a mean without a null, or against the activation null 0.898",
            "why": (
                f"Weight pairwise Gaussian null std is 1/sqrt(D)="
                f"{calibration['null_by_organ']['w1']['analytic_pairwise_cosine_std']:.6e} "
                "on 2048x4096. 0.898 is the constant-mean null for raw activations."
            ),
        },
        {
            "what": "Report only a global mean and miss a strongly-aligned subset",
            "why": "A subset of (layer, organ) with mean > 0.05 is enough to justify a scoped shared structure.",
        },
        {
            "what": "Count stream-ranges.jsonl lines as the chunk count",
            "why": (
                f"{integrity.get('stream_ranges_lines_on_disk')} lines vs sealed prefix "
                f"{EXPECTED_CHUNK_COUNT}. Appended journal. Double-count is an artifact of method."
            ),
        },
        {
            "what": "Evaluate compression on the 930 MiB activation-X capture",
            "why": "NS-009 / standing discipline: every prior sub-bit negative here was a Gaussian-proxy artifact. This check is weight pairwise cosine, not X.",
        },
        {
            "what": "Jump from a positive cosine to a storage-BPW win",
            "why": "Q80 mixed pack: 0.6462 stored vs 2.518 active (~3.9x). Shared basis that saves storage and costs reconstruction per token can lose.",
        },
    ]


def emit(report: dict) -> None:
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(RECEIPT_PATH)


def banner(report: dict) -> str:
    integ = report["integrity"]
    cal = report["calibration"]
    fam = report["family"]
    verdict = report["verdict"]
    lines = []
    a = lines.append
    a("NOETIC DSV4F PAIRWISE COSINE")
    a("=" * 72)
    a(f"schema     {SCHEMA}")
    a(f"generated  {report['generated_at']}")
    a(f"git_head   {report['git_head']}")
    a(f"repo       {report['repo']}")
    a(f"obligation {NNS_ID} / {NS_ID}")
    a(f"elapsed_s  {report['elapsed_s']:.3f}")
    a("")
    a("## ARTIFACT INTEGRITY")
    a(f"path       {integ.get('artifact_path')}")
    a(f"status     {integ['status']}")
    a(f"intact     {integ['intact']}")
    a(f"chunks     {integ.get('chunks_regular_files')} files / {integ.get('chunks_bytes')} bytes")
    a(f"expected   {EXPECTED_CHUNK_COUNT} files / {EXPECTED_TENSOR_BYTES} bytes")
    a(f"ranges     {integ.get('stream_ranges_lines_on_disk')} jsonl lines "
      f"(sealed prefix {EXPECTED_CHUNK_COUNT}; do not use lines as chunk count)")
    man = integ.get("manifest_head") or {}
    a(f"manifest   schema={man.get('schema')} status={man.get('status')}")
    a(f"seal       {man.get('seal_sha256')}")
    a(f"source     {PINNED_REPO} @{PINNED_REV}")
    arch = man.get("architecture") or {}
    a(f"arch       layers={arch.get('layer_count')} routed={arch.get('n_routed_experts')} "
      f"activated={arch.get('n_activated_experts')} expert_dtype={arch.get('expert_dtype')} "
      f"scale_fmt={arch.get('scale_fmt')}")
    a("chunk search:")
    for row in integ.get("chunk_search") or []:
        a(f"  exists={row['exists']} files={row['regular_files']} bytes={row['bytes']}  {row['root']}")
    a("probe (layers.0.ffn.experts.0.w1.weight):")
    if integ.get("probe_hits"):
        for h in integ["probe_hits"]:
            a(f"  HIT {h['probe_chunk']} bytes={h.get('probe_bytes')} "
              f"sha_match={h.get('probe_sha_match')}")
    else:
        a(f"  MISS {PROBE_CHUNK_REL} under every search root")
    for r in integ.get("reasons") or []:
        a(f"reason     {r}")
    a("")
    a("## REFUSED STAND-INS")
    for row in report["refused_standins"]:
        a(f"  {row['path']}")
        a(f"    exists={row.get('exists')}  {row.get('why_refused')}")
    a("")
    a("## FAMILY (from stream-ranges, unique tensor names)")
    w = fam.get("weights") or {}
    a(f"unique weight tensors {w.get('n_unique')}  base layers={w.get('n_layers_base')} "
      f"expert_ids={w.get('n_expert_ids_base')}  organs={w.get('organs_base')}")
    a(f"expected base weights {fam.get('expected_base_weight_tensors')}  "
      f"match={fam.get('base_weight_count_matches')}")
    a(f"target     {fam.get('shared_basis_target', {}).get('what')}")
    a("")
    a("## METHOD CALIBRATION (scale-aware cosine + null)")
    a(f"rejects 0.01*W          {cal['rejects_deliberately_scaled_artifact']}")
    t = cal["trap_0.01_W_small"]
    a(f"0.01*W small            cosine={t['cosine']:.6f}  scale_ratio={t['scale_ratio']:.6f}  "
      f"scale_aware={t['scale_aware_cosine']:.6f}  rel_l2={t['rel_l2']:.4f}")
    t = cal["trap_0.01_W_expert_shaped_2048x4096"]
    a(f"0.01*W 2048x4096        cosine={t['cosine']:.6f}  scale_ratio={t['scale_ratio']:.6f}  "
      f"scale_aware={t['scale_aware_cosine']:.6f}  rel_l2={t['rel_l2']:.4f}")
    n1 = cal["null_by_organ"]["w1"]
    a(f"null w1 D={n1['D']}     analytic mean=0  std={n1['analytic_pairwise_cosine_std']:.6e}  "
      f"p95≈{n1['analytic_p95_abs']:.6e}")
    a(f"activation null 0.898   DOES NOT APPLY (weight pairwise, not raw activation cosine)")
    a(f"WORTH_TESTING bar       cosine mean materially above {COSINE_WORTH_TESTING} (NS-010)")
    a(f"Q80 number              gate mean {Q80_GATE_PAIRWISE_MEAN} — do not transfer")
    a("")
    a("## MEASUREMENT")
    if report.get("measurement") is None:
        a("STOP — artifact not intact. No pairwise cosine was computed on any weight.")
        a("No layer-4 diagnostic, no activation-X, no packed-nibble proxy, no Q80 transfer.")
    else:
        m = report["measurement"]
        p = m["pooled"]
        a(f"groups measured {p.get('n_groups_measured')}  elapsed_s={m.get('elapsed_s'):.1f}")
        a(f"cosine mean-of-means {p.get('cosine_mean_of_group_means')}  "
          f"min {p.get('cosine_min_of_group_means')}  max {p.get('cosine_max_of_group_means')}")
        a(f"scale-aware mean-of-means {p.get('scale_aware_mean_of_group_means')}")
        a(f"aligned subsets (group mean>=0.05 or frac>0.30): {m.get('n_aligned_subsets')}")
        for s in m.get("aligned_subsets") or []:
            a(f"  {s}")
        a("per-group cosine distribution (mean p50 p95 max frac>0.05):")
        for g in m.get("groups") or []:
            cd = g.get("pairwise_cosine") or {}
            if not cd:
                a(f"  {g.get('key')}  SKIP {g.get('skipped')}")
                continue
            a(
                f"  {g['key']:10s}  mean={cd['mean']:.6f}  p50={cd['p50']:.6f}  "
                f"p95={cd['p95']:.6f}  max={cd['max']:.6f}  "
                f"gt0.05={cd['frac_gt_0.05']:.4f}  gt0.30={cd.get('frac_gt_0.30', 0):.4f}  "
                f"vs_null_std={cd['mean'] / (g['null_std'] + 1e-30):.1f}σ"
            )
    a("")
    a("## VERDICT")
    a(f"verdict                 {verdict['verdict']}")
    a(f"WORTH_TESTING or DEAD   {verdict.get('worth_testing_or_dead')}")
    a(f"deciding_number         {json.dumps(verdict.get('deciding_number'))}")
    if verdict.get("why_not_dead"):
        a(f"why_not_DEAD            {verdict['why_not_dead']}")
    a("")
    a("## WHAT I WATCHED FAIL")
    for i, row in enumerate(report["watched_fail"], 1):
        a(f"  {i}. {row['what']}")
        a(f"     {row['why']}")
    a("")
    a(f"wrote {report['receipt_path']}")
    a("=" * 72)
    return "\n".join(lines) + "\n"


def main() -> int:
    t0 = time.time()
    if not method_calibration.__doc__:
        pass
    loc = locate_artifact()
    refused = refuse_partial_artifacts()
    calibration = method_calibration()
    if not calibration["rejects_deliberately_scaled_artifact"]:
        print("FATAL: scale-aware metric did not reject 0.01*W", file=sys.stderr)
        return 2

    integrity = None
    family = {
        "weights": {},
        "scales": {},
        "shared_basis_target": {},
        "index": {"base": {}},
        "n_lines": 0,
        "base_weight_count_matches": False,
        "expected_base_weight_tensors": EXPECTED_N_LAYERS * EXPECTED_N_ROUTED * 3,
    }
    measurement = None
    artifact = loc.get("path")
    if artifact is None:
        integrity = {
            "artifact_path": None,
            "status": "BODY_ABSENT",
            "intact": False,
            "truncated": False,
            "missing_body": True,
            "stop_measurement": True,
            "reasons": ["no DSV4F artifact directory found in search paths"],
            "files": {},
            "chunk_search": [],
            "chunks_regular_files": 0,
            "chunks_bytes": 0,
            "probe_hits": [],
            "stream_ranges_lines_on_disk": 0,
            "expected": {"chunk_count": EXPECTED_CHUNK_COUNT},
        }
    else:
        integrity = verify_artifact(artifact)
        ranges = artifact / "stream-ranges.jsonl"
        if ranges.is_file():
            family = census_family(ranges)
            # Drop the bulky per-expert index from the receipt if we are not
            # measuring; keep counts only.
            if integrity["stop_measurement"]:
                family = {k: v for k, v in family.items() if k != "index"}
                family["index_omitted"] = (
                    "per-expert chunk index omitted because measurement did not run"
                )
        if not integrity["stop_measurement"]:
            body_root = Path(integrity["body_root"]) if integrity.get("body_root") else artifact / "chunks"
            measurement = measure_family(artifact, family, body_root)
            family = {k: v for k, v in family.items() if k != "index"}
            family["index_omitted"] = "per-expert chunk index omitted from receipt (too large)"

    verdict = decide(integrity, calibration, measurement)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "git_head": git_head(),
        "repo": str(REPO),
        "receipt_path": str(RECEIPT_PATH),
        "elapsed_s": time.time() - t0,
        "obligation": {
            "id": NNS_ID,
            "register": NS_ID,
            "kind": "idea (scoped)",
            "text": (
                "Q80/F0/F1 shared-basis is dead. DSV4F pairwise cosine has never "
                "been measured — cheap check, not a Q80 retry."
            ),
        },
        "did_not": [
            "spawn a 27B model server",
            "evaluate compression on synthetic activations",
            "measure packed FP4 without UE8M0 decode",
            "measure streamed-layer4-diagnostic.gravity",
            "measure activation-x-capture-3k",
            "transfer Q80 cosine 0.00414 onto DSV4F",
            "treat stream-ranges line count as chunk count",
        ],
        "locate": {k: v for k, v in loc.items() if k != "path"},
        "refused_standins": refused,
        "integrity": integrity,
        "family": family,
        "calibration": calibration,
        "measurement": measurement,
        "verdict": verdict,
        "storage_vs_active_reminder": {
            "q80_stored_bpw": 0.6462,
            "q80_active_bpw": 2.518,
            "factor": 3.9,
            "applies_even_if_cosine_is_high": True,
        },
        "watched_fail": watched_fail(integrity, calibration),
    }
    emit(report)
    sys.stdout.write(banner(report))
    return 0 if integrity.get("intact") or integrity.get("missing_body") else 1


if __name__ == "__main__":
    sys.exit(main())
