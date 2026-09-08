"""MLP TEACHER CORPUS — real (X, F(X)) for a small-program fit of the organ.

The 2-bit MLP code body is at its entropy floor (1.87018 bits of 2; 93.5%
independent information; neighbours add 0.00195 bits). Lossless compression of
q is dead. The remaining question is whether a SMALL PROGRAM can reproduce the
organ's FUNCTION F(x) = down(silu(gate(x)) * up(x)) even though it cannot
store q. That question is only well-posed against real input/output
activations. No such corpus existed for sealed-3.14.

    python3 tools/future/mlp_teacher_corpus.py --build
    python3 tools/future/mlp_teacher_corpus.py --capture
    python3 -m pytest tools/future/test_mlp_teacher_corpus.py -q

X is the post-attention RMSNorm hidden (the real gate/up input; see
CORRECTION_MLP_INPUT_TENSOR — post_input_norm is the wrong tensor). Y is
F(X) under the sealed-3.14 affine-Q2 packing the resident fused-SwiGLU
kernel consumes. No Gaussian / synthetic X (NNS-001). Held-out is by
PROMPT, not by row. Sizing refuses the NNS-007 underdetermined-fit scar
(median 92 rows against 2048 dims).

evidence_class DIAGNOSTIC_RELATIVE for capture elapsed; the activations
are exact f32 values of F, not a noisy estimate.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.future._common import (
    REPO,
    git,
    load_json,
    sha256_file,
    write_receipt,
)
from tools.future.mlp_auxiliary_information import (
    _read_f16,
    _read_u8,
    _unpack_q,
    mlp_records,
    parse_hgrafv01_header,
)
from tools.future.mlp_byte_census import (
    CatalogAbsent,
    load_geometry,
    load_sealed,
    resolve_artifact_root,
)
from tools.future.mlp_code_information import RECEIPT as CODE_INFO_RECEIPT


RECEIPT = "MLP_TEACHER_CORPUS.json"
SCHEMA = "hawking.future.mlp_teacher_corpus.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_teacher_corpus.py"
CODE_INFO_REL = f"receipts/future/{CODE_INFO_RECEIPT}"
SEALED_REL = "hcli/hawking-native.sealed-3.14.json"

HIDDEN = 5120
INTERMEDIATE = 17408
N_LAYERS = 64
RANK = 32
# Rank-32 map X ↦ XABᵀ, A,B ∈ R^{5120×32}. Gauge-fixed parameter count
# is RANK * (2 * HIDDEN - RANK). Each row supplies HIDDEN equations, so
# n * HIDDEN >= that count  ⇒  n >= RANK * (2*HIDDEN - RANK) / HIDDEN = 64.
RANK32_PARAMS = RANK * (2 * HIDDEN - RANK)
MIN_TRAIN_ROWS_DETERMINED = int(math.ceil(RANK32_PARAMS / HIDDEN))  # 64
# NNS-007: n_fit >= claimed rank or the score is not the codec's score.
MIN_TRAIN_ROWS_RANK = RANK
# Scar: median 92 rows against 2048 dims. rows-per-dim = 92/2048.
NNS007_SCAR_ROWS = 92
NNS007_SCAR_DIM = 2048
NNS007_SCAR_ROWS_PER_DIM = NNS007_SCAR_ROWS / NNS007_SCAR_DIM
DUP_RATE_MAX = 0.05
HOLD_FRAC = 0.25

CAPABILITY_DOMAINS: tuple[str, ...] = (
    "reasoning",
    "code",
    "tool-calling",
    "long-horizon-coherence",
    "plain-prose",
)

# capture_diverse2 families → the five domains this corpus is required to span.
# instruction is the capture's operator-style task family (the closest real
# held-out-by-prompt set to tool-calling; JSON tool-call templates live in
# CAPTURE_PROMPTS for a live recapture). multilingual joins plain-prose.
FAMILY_TO_DOMAIN: dict[str, str] = {
    "math": "reasoning",
    "code": "code",
    "instruction": "tool-calling",
    "dialogue": "long-horizon-coherence",
    "prose": "plain-prose",
    "multilingual": "plain-prose",
}

SYNTHETIC_KINDS = frozenset(
    {
        "synthesised",
        "synthesized",
        "gaussian",
        "gaussian_proxy",
        "proxy",
        "random",
        "synthetic",
    }
)
POSITION_BANDS = ("early", "middle", "late")

# Named in the receipt. Gitignored via /workspace/ops/local/scratch/.
PAYLOAD_DIR = REPO / "workspace" / "ops" / "local" / "scratch" / "mlp_teacher_corpus"

CAPTURE_X_CANDIDATES: tuple[Path, ...] = (
    Path("/Users/scammermike/Downloads/hawking/workspace/campaign/phaseB/capture_diverse2"),
    REPO / "workspace" / "campaign" / "phaseB" / "capture_diverse2",
    Path("/Users/scammermike/Downloads/hawking-copy/workspace/campaign/phaseB/capture_diverse2"),
)

RESIDENT_CANDIDATES: tuple[Path, ...] = (
    REPO / "workspace" / "ops" / "build" / "rust" / "release" / "examples" / "ascension_qwen38_resident",
    Path(
        "/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/"
        "release/examples/ascension_qwen38_resident"
    ),
    Path(
        "/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/"
        "release-fast/examples/ascension_qwen38_resident"
    ),
)

FUSION_ENV = {
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
    "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
    "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
    "HAWKING_QWEN38_FUSE_MLP": "swiglu",
}

# Fixture prompts so the split/synthetic guards have a real catalog to refuse.
# Live recapture should use these (or a superset) against the sealed residual.
CAPTURE_PROMPTS: tuple[tuple[str, str], ...] = (
    ("reasoning", "Prove that sqrt(2) is irrational by infinite descent."),
    ("reasoning", "A fair die is rolled twice. What is P(sum = 7)? Show the sample space."),
    ("reasoning", "Differentiate x^3 - 4x and find the critical points."),
    ("reasoning", "Why does 0.1 + 0.2 fail to equal 0.3 in IEEE-754? One paragraph."),
    ("reasoning", "The sum of the first n integers is n(n+1)/2. Prove it by induction."),
    ("reasoning", "Solve 3x + 7 = 22 and check the result in the original equation."),
    ("reasoning", "A right triangle has legs 9 and 12. What is the hypotenuse, and why?"),
    ("reasoning", "State Bayes' rule and apply it to a 1% prevalence, 99% sensitive test."),
    ("code", "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    p = a[len(a)//2]\n"),
    ("code", "impl<T: Clone> Stack<T> { fn push(&mut self, v: T) { self.items.push(v); } }"),
    ("code", "SELECT name, COUNT(*) AS n FROM orders GROUP BY name HAVING COUNT(*) > 3;"),
    ("code", "async function fetchJson(url) { const r = await fetch(url); return r.json(); }"),
    ("code", "pub fn gcd(mut a: u64, mut b: u64) -> u64 { while b != 0 { let t = b; b = a % b; a = t; } a }"),
    ("code", "for i, row in enumerate(grid):\n    for j, v in enumerate(row):\n        if v: dfs(i, j)"),
    ("code", "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.left = self.right = None"),
    ("code", "const memo = new Map(); function fib(n) { if (n < 2) return n; }"),
    (
        "tool-calling",
        '{"tool": "search", "arguments": {"query": "unified memory bandwidth", "limit": 5}}',
    ),
    (
        "tool-calling",
        '{"tool": "read_file", "arguments": {"path": "tools/future/mlp_byte_census.py", "limit": 80}}',
    ),
    (
        "tool-calling",
        '{"tool": "run_terminal", "arguments": {"command": "git rev-parse HEAD"}}',
    ),
    (
        "tool-calling",
        '{"name": "catalog_lookup", "args": {"organ": "mlp.down", "layer": 31}}',
    ),
    (
        "tool-calling",
        "Call the `list_dir` tool on workspace/ops/local/scratch and return the names.",
    ),
    (
        "tool-calling",
        '{"tool": "write", "arguments": {"path": "/tmp/out.json", "content": {"ok": true}}}',
    ),
    (
        "tool-calling",
        "Use get_command_or_subagent_output on task_id abc, timeout_ms 5000.",
    ),
    (
        "tool-calling",
        '{"tool": "open_page", "arguments": {"url": "https://docs.python.org/3/library/hashlib.html"}}',
    ),
    (
        "long-horizon-coherence",
        "Write a four-paragraph history of the Library of Alexandria that keeps the "
        "same narrator and does not drop the thread on funding cuts.",
    ),
    (
        "long-horizon-coherence",
        "User: My laptop will not turn on. Assistant: Is the charger light on? Continue "
        "the diagnosis for three more turns without restarting from scratch.",
    ),
    (
        "long-horizon-coherence",
        "Tell the story of a single golden spike across four scenes: crews, granite, "
        "desert, and the last hammer blow, keeping one protagonist.",
    ),
    (
        "long-horizon-coherence",
        "Explain sleep's housekeeping, then connect it to immunity, then to judgment, "
        "without abandoning the original claim that rest is physiological.",
    ),
    (
        "long-horizon-coherence",
        "A dialogue about TCP vs UDP that lasts four exchanges and ends on when to "
        "pick UDP, referring back to the handshake mentioned in turn one.",
    ),
    (
        "long-horizon-coherence",
        "Track one coffee bean from Ethiopian highlands through a 17th-century "
        "coffeehouse to a modern morning, same bean, same causal chain.",
    ),
    (
        "long-horizon-coherence",
        "Describe building a suspension bridge in three phases (cables, towers, "
        "deck) and close by recalling the Tacoma Narrows warning from phase one.",
    ),
    (
        "long-horizon-coherence",
        "Narrate the smallpox eradication village by village and finish on the "
        "freezer, having named the same campaign throughout.",
    ),
    (
        "plain-prose",
        "The Amazon rainforest produces roughly twenty percent of the planet's oxygen.",
    ),
    (
        "plain-prose",
        "Medieval cathedrals took generations to build; the masons rarely saw the spires.",
    ),
    (
        "plain-prose",
        "Ocean currents redistribute heat from the equator toward the poles.",
    ),
    (
        "plain-prose",
        "Photosynthesis splits water, releases oxygen, and fixes carbon into sugars.",
    ),
    (
        "plain-prose",
        "Glass is neither fully solid nor liquid; it carries the internet as fiber.",
    ),
    (
        "plain-prose",
        "Auroras form when solar particles collide with oxygen and nitrogen overhead.",
    ),
    (
        "plain-prose",
        "The printing press standardized spelling and made private silent reading ordinary.",
    ),
    (
        "plain-prose",
        "Volcanic soil is fertile because eruptions grind minerals and spread them.",
    ),
)

CLAIM_BOUNDARY = (
    "DIAGNOSTIC_RELATIVE sidecar. X is real post_attn_norm from teacher-forced "
    "prefill (capture_diverse2; named Qwen3.8 teacher under NNS-006). Y is "
    "F(X)=down(silu(gate(X))*up(X)) reconstructed from the sealed-3.14 HGRAVF01 "
    "affine-Q2 tensors the resident fused-SwiGLU kernel consumes "
    "(w = float(q)*scale+bias). Y is exact for that arithmetic. X is not claimed "
    "to be the sealed residual stream: the JSONL resident does not dump hidden "
    "states, and this module refuses to substitute Gaussian X (NNS-001) or the "
    "wrong norm (CORRECTION_MLP_INPUT_TENSOR). capture_elapsed_s is process "
    "elapsed, not a GPU-lease measurement. bench.gpu_authority is false."
)


class CorpusRefused(ValueError):
    """Loud refusal: leak, synthetic row, or a pad that only meets a count."""

    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result
        self.codes = list(result.get("refusals") or [])


class CorpusInadequate(ValueError):
    """Honest corpus that is too thin for a rank-32 fit."""

    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result
        self.codes = list(result.get("inadequacy") or [])


class CaptureUnavailable(RuntimeError):
    """Artifact, X source, or packed organ is not readable. Not a default."""


# ---------------------------------------------------------------------------
# Identity / paths
# ---------------------------------------------------------------------------


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_canonical(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(blob.encode("utf-8"))


def load_code_information() -> dict[str, Any]:
    path = REPO / CODE_INFO_REL
    if path.is_file():
        return load_json(path)
    raw = git("show", f"HEAD:{CODE_INFO_REL}")
    if not raw:
        raise CaptureUnavailable(f"REFUSED: {CODE_INFO_REL} not visible on disk or git HEAD")
    return json.loads(raw)


def specimen_identity() -> dict[str, Any]:
    sealed = load_sealed()
    try:
        artifact = str(resolve_artifact_root(sealed))
    except CatalogAbsent:
        artifact = str(sealed.get("artifact_root") or "")
    binary = resolve_resident_binary(sealed)
    tok = sealed.get("tokenizer")
    return {
        "resident_identity": sealed.get("resident_identity"),
        "model_id": sealed.get("model_id"),
        "artifact_root": artifact,
        "tokenizer": tok,
        "resident_binary": str(binary) if binary is not None else None,
        "resident_binary_sha256": (
            sha256_file(binary) if binary is not None and binary.is_file() else None
        ),
        "fusion_env": dict(sealed.get("fusion_env") or FUSION_ENV),
        "protocol": sealed.get("protocol") or "hawking.qwen38.resident.v1",
        "require_fusion_env": bool(sealed.get("require_fusion_env", True)),
        "seal_source": SEALED_REL,
        "pinned_revision": git("rev-parse", "HEAD") or None,
        "geometry": {"hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE, "num_hidden_layers": N_LAYERS},
    }


def resolve_resident_binary(sealed: Mapping[str, Any] | None = None) -> Path | None:
    # Prefer the sealed-fusion release example named in this campaign, then
    # the sealed-3.14 JSON path (release-fast), then remaining candidates.
    candidates: list[Path] = list(RESIDENT_CANDIDATES)
    if sealed:
        for key in ("resident_binary", "compiler"):
            raw = sealed.get(key)
            if isinstance(raw, str) and raw.strip():
                candidates.append(Path(raw))
            elif isinstance(raw, dict) and isinstance(raw.get("source"), str):
                candidates.append(REPO / raw["source"])
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def resolve_x_capture_dir() -> Path:
    for path in CAPTURE_X_CANDIDATES:
        if (path / "manifest.json").is_file() and (path / "L00.f16").is_file():
            return path
    raise CaptureUnavailable(
        "REFUSED: real post_attn_norm capture_diverse2 is not readable; "
        "refusing to synthesise X (NNS-001)"
    )


# ---------------------------------------------------------------------------
# Fingerprint all 64 MLP layers. Evidence, not an L0 default.
# ---------------------------------------------------------------------------


def _layer_types(artifact: Path | None = None) -> list[str]:
    try:
        root = artifact if artifact is not None else resolve_artifact_root()
        geo = load_geometry(root)
        types = geo.get("layer_types")
        if isinstance(types, list) and len(types) == N_LAYERS:
            return [str(x) for x in types]
    except (CatalogAbsent, Exception):
        pass
    # Fallback from the measured Qwen3.5 interval-4 pattern (config.json).
    return [
        "full_attention" if (i % 4 == 3) else "linear_attention" for i in range(N_LAYERS)
    ]


def fingerprint_layers(
    *,
    code_info: Mapping[str, Any] | None = None,
    artifact: Path | None = None,
) -> list[dict[str, Any]]:
    """One row per layer: H(q) of gate/up/down plus mixer class.

    H(q) is the already-measured Shannon entropy of the real 2-bit codes
    (MLP_CODE_INFORMATION). Layer 0 is slightly *higher* entropy than the
    later mean — it is not typical.
    """
    doc = code_info if code_info is not None else load_code_information()
    per = (doc.get("measurements") or {}).get("per_tensor") or []
    by: dict[int, dict[str, float]] = defaultdict(dict)
    for row in per:
        layer = int(row["layer"])
        by[layer][str(row["organ"])] = float(row["H_q"])
    if len(by) != N_LAYERS:
        raise CaptureUnavailable(
            f"REFUSED: code-information fingerprint covers {len(by)} layers, want {N_LAYERS}"
        )
    types = _layer_types(artifact)
    global_mean = float(
        np.mean([(by[L]["mlp.gate"] + by[L]["mlp.up"] + by[L]["mlp.down"]) / 3.0 for L in range(N_LAYERS)])
    )
    out: list[dict[str, Any]] = []
    for layer in range(N_LAYERS):
        organs = by[layer]
        h_mean = (organs["mlp.gate"] + organs["mlp.up"] + organs["mlp.down"]) / 3.0
        mixer = types[layer] if layer < len(types) else "unknown"
        out.append(
            {
                "layer": layer,
                "mixer": mixer,
                "H_q_gate": organs["mlp.gate"],
                "H_q_up": organs["mlp.up"],
                "H_q_down": organs["mlp.down"],
                "H_q_mean": h_mean,
                "delta_from_global_mean": h_mean - global_mean,
                "is_layer0": layer == 0,
            }
        )
    return out


def pick_representatives(fingerprints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose organs with evidence. Layer 0 is never the 'typical' pick."""
    if len(fingerprints) != N_LAYERS:
        raise CaptureUnavailable("REFUSED: fingerprint does not cover 64 layers")
    rows = [dict(r) for r in fingerprints]
    global_mean = float(np.mean([r["H_q_mean"] for r in rows]))
    ranked_close = sorted(rows, key=lambda r: abs(r["H_q_mean"] - global_mean))
    typical = next(r for r in ranked_close if r["layer"] != 0)
    entropy_max = max(rows, key=lambda r: r["H_q_mean"])
    entropy_min = min(rows, key=lambda r: r["H_q_mean"])
    full = [r for r in rows if r["mixer"] == "full_attention"]
    linear = [r for r in rows if r["mixer"] == "linear_attention"]
    if not full or not linear:
        raise CaptureUnavailable("REFUSED: mixer classes missing from fingerprint")
    first_full = min(full, key=lambda r: r["layer"])
    last = rows[-1]
    nns015 = rows[31]
    layer0 = rows[0]

    picks: list[dict[str, Any]] = []

    def _add(row: Mapping[str, Any], role: str, why: str) -> None:
        if any(p["layer"] == row["layer"] for p in picks):
            return
        picks.append(
            {
                "layer": int(row["layer"]),
                "role": role,
                "mixer": row["mixer"],
                "H_q_mean": float(row["H_q_mean"]),
                "H_q_gate": float(row["H_q_gate"]),
                "H_q_up": float(row["H_q_up"]),
                "H_q_down": float(row["H_q_down"]),
                "delta_from_global_mean": float(row["delta_from_global_mean"]),
                "why": why,
            }
        )

    _add(
        typical,
        "typical",
        (
            f"Closest H(q) to the 64-layer mean ({global_mean:.6f} bits) among "
            f"layers other than 0. Layer 0 is a high-entropy outlier "
            f"(H_q_mean={layer0['H_q_mean']:.6f}, delta="
            f"{layer0['delta_from_global_mean']:+.6f}) and is not typical."
        ),
    )
    _add(
        first_full,
        "first_full_attention",
        (
            "First full-attention (GQA) layer. Mixer-class contrast against the "
            "48 linear-attention MLPs; independently the stack entropy maximum "
            f"(H_q_mean={first_full['H_q_mean']:.6f})."
            if first_full["layer"] == entropy_max["layer"]
            else "First full-attention (GQA) layer; mixer-class contrast against linear-attention MLPs."
        ),
    )
    _add(
        nns015,
        "nns015_mid_full",
        (
            "Layer 31 is the NNS-015 named surface (down_proj L31, real "
            "post_swiglu X). Mid-depth full-attention; H(q) sits below the "
            f"global mean (delta={nns015['delta_from_global_mean']:+.6f})."
        ),
    )
    _add(
        last,
        "last_layer_entropy_min",
        (
            "Last layer. Depth-axis pole. Independently the stack entropy "
            f"minimum (H_q_mean={entropy_min['H_q_mean']:.6f}, driven by "
            f"mlp.down H_q={last['H_q_down']:.6f})."
        ),
    )
    if entropy_max["layer"] not in {p["layer"] for p in picks}:
        _add(entropy_max, "entropy_max", "Highest mean H(q) of the 64 MLPs.")
    if entropy_min["layer"] not in {p["layer"] for p in picks}:
        _add(entropy_min, "entropy_min", "Lowest mean H(q) of the 64 MLPs.")

    picks.sort(key=lambda p: p["layer"])
    return {
        "global_H_q_mean": global_mean,
        "layer0": {
            "layer": 0,
            "H_q_mean": float(layer0["H_q_mean"]),
            "delta_from_global_mean": float(layer0["delta_from_global_mean"]),
            "mixer": layer0["mixer"],
            "typical": False,
            "why_not_typical": (
                "Layer 0 mean H(q) is above the later-layer mean (MLP_CODE_INFORMATION "
                "answers.are_some_rows_or_layers_lower_entropy). Using it as 'the MLP' "
                "is the default this corpus exists to refuse."
            ),
        },
        "n_linear_attention": len(linear),
        "n_full_attention": len(full),
        "chosen": picks,
        "chosen_layers": [p["layer"] for p in picks],
    }


def rows_per_dimension(n_rows: int, dim: int = HIDDEN) -> float:
    if dim <= 0:
        raise ValueError("dim must be positive")
    return n_rows / dim


def min_train_rows_for_rank32() -> int:
    return MIN_TRAIN_ROWS_DETERMINED


# ---------------------------------------------------------------------------
# Packed SwiGLU = the resident organ function
# ---------------------------------------------------------------------------


def reconstruct_w(segment_path: str | Path) -> np.ndarray:
    """(out, in) float32 from a real HGRAVF01 affine-Q2 tensor."""
    path = Path(segment_path)
    parsed = parse_hgrafv01_header(path)
    out_f, in_f = (int(x) for x in parsed["shape"])
    n_groups = int(parsed["groups"])
    scale = _read_f16(path, parsed["payload_off"], n_groups).astype(np.float32, copy=False)
    bias = _read_f16(
        path, parsed["payload_off"] + parsed["scale_bytes"], n_groups
    ).astype(np.float32, copy=False)
    codes = _read_u8(
        path,
        parsed["payload_off"] + parsed["scale_bytes"] + parsed["bias_bytes"],
        int(parsed["code_bytes"]),
    )
    q = _unpack_q(codes.reshape(n_groups, 16)).astype(np.float32, copy=False)
    weight = (q * scale[:, None] + bias[:, None]).reshape(out_f, in_f)
    return weight


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """f32 GEMM. Uses MPS when this interpreter has torch; else numpy/Accelerate."""
    try:
        import torch

        if torch.backends.mps.is_available():
            ta = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to("mps")
            tb = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to("mps")
            out = ta.matmul(tb).cpu().numpy()
            del ta, tb
            return out
    except Exception:
        pass
    return np.matmul(a, b)


def organ_records(layer: int) -> dict[str, dict[str, Any]]:
    recs = mlp_records()
    out = {r["organ"]: r for r in recs if int(r["layer"]) == int(layer)}
    for name in ("mlp.gate", "mlp.up", "mlp.down"):
        if name not in out:
            raise CaptureUnavailable(f"REFUSED: layer {layer} missing {name}")
    return out


def swiglu_f(x: np.ndarray, layer: int) -> np.ndarray:
    """Y = F_l(X) on the sealed packed organ. X is (n, 5120) post_attn_norm."""
    if x.ndim != 2 or x.shape[1] != HIDDEN:
        raise ValueError(f"X must be (n, {HIDDEN}), got {x.shape}")
    recs = organ_records(layer)
    w_gate = reconstruct_w(recs["mlp.gate"]["segment_path"])
    gate = _matmul(x, w_gate.T)
    del w_gate
    w_up = reconstruct_w(recs["mlp.up"]["segment_path"])
    up = _matmul(x, w_up.T)
    del w_up
    hidden = silu(gate) * up
    del gate, up
    w_down = reconstruct_w(recs["mlp.down"]["segment_path"])
    y = _matmul(hidden, w_down.T)
    del hidden, w_down
    return y


# ---------------------------------------------------------------------------
# Positions, prompts, hashes, provenance
# ---------------------------------------------------------------------------


def position_band(token_position: int, seq_len: int) -> str:
    if seq_len <= 1:
        return "early"
    frac = token_position / float(seq_len - 1)
    if frac <= 1.0 / 3.0:
        return "early"
    if frac <= 2.0 / 3.0:
        return "middle"
    return "late"


def content_sha256_of(row: Mapping[str, Any]) -> str:
    """Identity of captured content. row_id / provenance excluded so copies collide."""
    identity = {
        "layer": int(row.get("layer", -1)),
        "prompt_id": str(row.get("prompt_id") or ""),
        "token_position": int(row.get("token_position", -1)),
        "x_sha256": row.get("x_sha256"),
        "y_sha256": row.get("y_sha256"),
    }
    return _sha256_canonical(identity)


def envelope_sha256_of(row: Mapping[str, Any]) -> str:
    body = {k: v for k, v in row.items() if k not in {"content_sha256", "envelope_sha256"}}
    return _sha256_canonical(body)


def vector_sha256(vec: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(vec, dtype="<f4").tobytes())


def is_synthetic_row(row: Mapping[str, Any]) -> bool:
    if bool(row.get("synthetic")):
        return True
    prov = row.get("provenance") or {}
    kind = str(prov.get("kind") or "").lower()
    if kind in SYNTHETIC_KINDS:
        return True
    gen = str(row.get("x_generator") or prov.get("generator") or "").lower()
    if gen in SYNTHETIC_KINDS or "gaussian" in gen or "np.random" in gen:
        return True
    if str(row.get("x_source_kind") or "").lower() in SYNTHETIC_KINDS:
        return True
    return False


def annotate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["synthetic"] = bool(is_synthetic_row(out))
    out["content_sha256"] = content_sha256_of(out)
    out["envelope_sha256"] = envelope_sha256_of(out)
    return out


def provenance_captured(*, source_path: str, source_sha256: str | None, layer: int) -> dict[str, Any]:
    return {
        "kind": "captured",
        "authority": "DIAGNOSTIC_RELATIVE",
        "source_path": source_path,
        "source_sha256": source_sha256,
        "capture_tool": RECORDED_BY,
        "layer": int(layer),
        "x_kind": "post_attn_norm",
        "y_kind": "sealed_affine_q2_swiglu",
        "not": ["gaussian", "gaussian_proxy", "synthetic", "position_0_only"],
        "scars": ["NNS-001", "NNS-007", "CORRECTION_MLP_INPUT_TENSOR"],
    }


# ---------------------------------------------------------------------------
# Split by prompt. A fit that memorizes a prompt must not score on its tokens.
# ---------------------------------------------------------------------------


def unique_prompt_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = str(row["prompt_id"])
        if pid not in seen:
            seen[pid] = {
                "prompt_id": pid,
                "capability_domain": str(row.get("capability_domain") or ""),
            }
    return [seen[k] for k in sorted(seen)]


def split_by_prompt(
    rows: Sequence[Mapping[str, Any]],
    *,
    hold_frac: float = HOLD_FRAC,
) -> dict[str, Any]:
    """Held-out is a set of prompt ids. Every token of a prompt stays on one side."""
    if hold_frac <= 0.0 or hold_frac >= 1.0:
        raise ValueError("hold_frac must be in (0, 1)")
    prompts = unique_prompt_records(rows)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for rec in prompts:
        by_domain[rec["capability_domain"]].append(rec["prompt_id"])
    train: set[str] = set()
    hold: set[str] = set()
    for domain in CAPABILITY_DOMAINS:
        ids = list(by_domain.get(domain) or [])
        # Include any extra domains (should not happen) after the required five.
        if not ids:
            continue
        ids.sort()
        n_hold = max(1, int(round(len(ids) * hold_frac))) if len(ids) >= 2 else 0
        if n_hold >= len(ids):
            n_hold = max(1, len(ids) // 4) if len(ids) >= 4 else 1
            if n_hold >= len(ids):
                n_hold = 0
        if n_hold:
            hold.update(ids[-n_hold:])
            train.update(ids[:-n_hold])
        else:
            train.update(ids)
    # Domains not in the required tuple still split so they cannot leak.
    for domain, ids in by_domain.items():
        if domain in CAPABILITY_DOMAINS:
            continue
        ids = sorted(ids)
        n_hold = max(1, int(round(len(ids) * hold_frac))) if len(ids) >= 2 else 0
        if n_hold >= len(ids):
            n_hold = 0
        if n_hold:
            hold.update(ids[-n_hold:])
            train.update(ids[:-n_hold])
        else:
            train.update(ids)
    leak = train & hold
    if leak:
        raise CorpusRefused(
            f"REFUSED: split constructor leaked prompt ids {sorted(leak)[:8]}",
            {"accepted": False, "refusals": ["HELD_OUT_PROMPT_LEAK"], "leaked": sorted(leak)},
        )
    return {
        "train_prompt_ids": sorted(train),
        "hold_prompt_ids": sorted(hold),
        "hold_frac": hold_frac,
        "split_unit": "prompt_id",
        "n_train_prompts": len(train),
        "n_hold_prompts": len(hold),
    }


def assign_split(rows: Sequence[Mapping[str, Any]], split: Mapping[str, Any]) -> list[dict[str, Any]]:
    train = set(split["train_prompt_ids"])
    hold = set(split["hold_prompt_ids"])
    leak = train & hold
    if leak:
        raise CorpusRefused(
            "REFUSED: held-out split shares prompt ids with train "
            f"({sorted(leak)[:8]})",
            {
                "accepted": False,
                "refusals": ["HELD_OUT_PROMPT_LEAK"],
                "leaked_prompt_ids": sorted(leak),
            },
        )
    unknown = []
    out: list[dict[str, Any]] = []
    for row in rows:
        pid = str(row["prompt_id"])
        if pid in train:
            side = "train"
        elif pid in hold:
            side = "hold"
        else:
            unknown.append(pid)
            side = "unassigned"
        item = dict(row)
        item["split"] = side
        out.append(item)
    if unknown:
        raise CorpusRefused(
            "REFUSED: rows whose prompt_id is on neither side of the split",
            {"accepted": False, "refusals": ["PROMPT_NOT_IN_SPLIT"], "prompt_ids": sorted(set(unknown))[:12]},
        )
    return out


# ---------------------------------------------------------------------------
# Dedup + emit. Guards nobody has watched fail are not guards.
# ---------------------------------------------------------------------------


def exact_dedup(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    annotated = [annotate_row(r) for r in rows]
    keep: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in annotated:
        h = row["content_sha256"]
        groups.setdefault(h, []).append(str(row.get("row_id")))
        if h not in seen:
            seen.add(h)
            keep.append(row)
    return keep, groups


def _refuse(codes: list[str], message: str, **extra: Any) -> None:
    result = {"accepted": False, "refusals": codes, **extra}
    raise CorpusRefused(f"REFUSED: {message} (codes={codes})", result)


def emit_manifest(
    rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    *,
    min_train_rows: int | None = None,
    allow_fixture: bool = False,
    require_sizing: bool = False,
) -> dict[str, Any]:
    """Build a corpus manifest or refuse. Held-out leak and synthetic rows are loud."""
    leak = set(split.get("train_prompt_ids") or []) & set(split.get("hold_prompt_ids") or [])
    if leak:
        _refuse(
            ["HELD_OUT_PROMPT_LEAK"],
            "held-out split shares a prompt id with train",
            leaked_prompt_ids=sorted(leak),
        )

    annotated = [annotate_row(r) for r in rows]
    synthetic = [r for r in annotated if is_synthetic_row(r)]
    if not allow_fixture:
        fixtures = [
            r
            for r in annotated
            if str((r.get("provenance") or {}).get("kind") or "").lower() == "fixture"
        ]
        if fixtures:
            _refuse(
                ["SYNTHETIC_ROW"],
                "fixture rows are not a promotion path on the real corpus",
                n_fixture=len(fixtures),
            )
    if synthetic:
        _refuse(
            ["SYNTHETIC_ROW"],
            "a synthetic / Gaussian / proxy row is in the corpus (NNS-001)",
            n_synthetic=len(synthetic),
            sample_ids=[str(r.get("row_id")) for r in synthetic[:8]],
        )

    assigned = assign_split(annotated, split)
    n = len(assigned)
    unique = len({r["content_sha256"] for r in assigned})
    collision_rate = (1.0 - unique / n) if n else 1.0
    if n and collision_rate > DUP_RATE_MAX:
        _refuse(
            ["DUPLICATE_ROWS_ABOVE_THRESHOLD"],
            f"duplicate rows {collision_rate:.4f} exceed threshold {DUP_RATE_MAX}",
            n_rows=n,
            n_unique=unique,
            collision_rate=collision_rate,
            dup_rate_max=DUP_RATE_MAX,
        )

    train_rows = [r for r in assigned if r["split"] == "train"]
    hold_rows = [r for r in assigned if r["split"] == "hold"]
    domains = sorted({str(r.get("capability_domain")) for r in assigned})
    missing_domains = [d for d in CAPABILITY_DOMAINS if d not in domains]
    bands = sorted({str(r.get("position_band")) for r in assigned})
    positions = {int(r["token_position"]) for r in assigned}

    inadequacy: list[str] = []
    floor = MIN_TRAIN_ROWS_DETERMINED if min_train_rows is None else int(min_train_rows)
    train_per_layer_counts = list(Counter(int(r["layer"]) for r in train_rows).values())
    min_train_layer = min(train_per_layer_counts) if train_per_layer_counts else 0
    if require_sizing and min_train_layer < floor:
        inadequacy.append("TRAIN_ROWS_UNDERDETERMINED_FOR_RANK32")
    if missing_domains:
        inadequacy.append("MISSING_CAPABILITY_DOMAIN")
    if require_sizing and not POSITION_BANDS[0] in bands:
        inadequacy.append("MISSING_POSITION_BAND")
    if require_sizing and len(positions) < 3:
        inadequacy.append("POSITION_DEGENERACY")

    if require_sizing and inadequacy:
        raise CorpusInadequate(
            f"INADEQUATE: {inadequacy}",
            {
                "accepted": False,
                "refusals": [],
                "inadequacy": inadequacy,
                "n_train_rows": len(train_rows),
                "min_train_rows": floor,
            },
        )

    keep, groups = exact_dedup(assigned)
    per_layer: dict[int, int] = Counter(int(r["layer"]) for r in assigned)
    per_layer_train: dict[int, int] = Counter(int(r["layer"]) for r in train_rows)
    per_domain_prompts: dict[str, int] = Counter(
        rec["capability_domain"] for rec in unique_prompt_records(assigned)
    )
    per_domain_rows: dict[str, int] = Counter(str(r.get("capability_domain")) for r in assigned)
    n_train = len(train_rows)
    rpd_train_per_layer = {
        str(layer): rows_per_dimension(count) for layer, count in sorted(per_layer_train.items())
    }
    min_train_per_layer = min(per_layer_train.values()) if per_layer_train else 0
    min_rpd_train = rows_per_dimension(min_train_per_layer) if min_train_per_layer else 0.0
    return {
        "schema": "hawking.future.mlp_teacher_corpus.manifest.v1",
        "accepted": True,
        "n_rows": n,
        "n_unique_content": unique,
        "collision_rate": collision_rate,
        "dup_rate_max": DUP_RATE_MAX,
        "n_train_rows": n_train,
        "n_hold_rows": len(hold_rows),
        "n_train_rows_per_layer": {str(k): int(v) for k, v in sorted(per_layer_train.items())},
        "rows_per_dimension_train": min_rpd_train,
        "rows_per_dimension_train_per_layer": rpd_train_per_layer,
        "rows_per_dimension_train_pooled": rows_per_dimension(n_train) if n_train else 0.0,
        "rows_per_dimension_all": rows_per_dimension(n) if n else 0.0,
        "rank": RANK,
        "hidden": HIDDEN,
        "min_train_rows_determined": MIN_TRAIN_ROWS_DETERMINED,
        "nns007_scar_rows_per_dim": NNS007_SCAR_ROWS_PER_DIM,
        "beats_nns007_scar": min_rpd_train > NNS007_SCAR_ROWS_PER_DIM,
        "rank32_overdetermined": min_train_per_layer >= MIN_TRAIN_ROWS_DETERMINED,
        "split": {
            "unit": "prompt_id",
            "hold_frac": split.get("hold_frac", HOLD_FRAC),
            "train_prompt_ids": list(split["train_prompt_ids"]),
            "hold_prompt_ids": list(split["hold_prompt_ids"]),
            "n_train_prompts": len(split["train_prompt_ids"]),
            "n_hold_prompts": len(split["hold_prompt_ids"]),
            "disjoint": True,
        },
        "capability_domains": list(CAPABILITY_DOMAINS),
        "n_prompts_per_domain": dict(sorted(per_domain_prompts.items())),
        "n_rows_per_domain": dict(sorted(per_domain_rows.items())),
        "n_rows_per_layer": {str(k): int(v) for k, v in sorted(per_layer.items())},
        "position_bands": {b: sum(1 for r in assigned if r.get("position_band") == b) for b in POSITION_BANDS},
        "n_unique_positions": len(positions),
        "inadequacy": inadequacy,
        "n_dedup_kept": len(keep),
        "n_dedup_collision_groups": sum(1 for ids in groups.values() if len(ids) > 1),
        "rows": assigned,
    }


# ---------------------------------------------------------------------------
# Fixtures (tests). kind=fixture is refused on the real emit path.
# ---------------------------------------------------------------------------


def _fixture_provenance() -> dict[str, Any]:
    return {
        "kind": "fixture",
        "authority": "STATIC_ONLY",
        "source_path": "tools/future/mlp_teacher_corpus.py::fixture",
        "source_sha256": _sha256_bytes(b"mlp-teacher-corpus-fixture-v1"),
        "capture_tool": RECORDED_BY,
        "note": "deterministic fixture; not a promotion and not a Gaussian proxy",
    }


def make_fixture_row(
    *,
    row_id: str,
    layer: int,
    prompt_id: str,
    prompt_text: str,
    token_position: int,
    seq_len: int,
    capability_domain: str,
    payload_seed: bytes,
    provenance: Mapping[str, Any] | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    # Hash-derived vectors so copies collide and distinct seeds do not. Not N(0,1).
    x = np.frombuffer(
        hashlib.sha256(payload_seed + b"|x").digest()
        + hashlib.sha256(payload_seed + b"|x2").digest(),
        dtype=np.uint8,
    ).astype(np.float32)
    y = np.frombuffer(
        hashlib.sha256(payload_seed + b"|y").digest()
        + hashlib.sha256(payload_seed + b"|y2").digest(),
        dtype=np.uint8,
    ).astype(np.float32)
    # Repeat to HIDDEN without drawing Gaussian noise.
    reps = int(math.ceil(HIDDEN / x.size))
    x = np.tile(x, reps)[:HIDDEN]
    y = np.tile(y, reps)[:HIDDEN]
    row = {
        "row_id": row_id,
        "layer": int(layer),
        "prompt_id": str(prompt_id),
        "prompt_text": str(prompt_text),
        "token_position": int(token_position),
        "seq_len": int(seq_len),
        "position_band": position_band(token_position, seq_len),
        "capability_domain": str(capability_domain),
        "x_sha256": vector_sha256(x),
        "y_sha256": vector_sha256(y),
        "synthetic": bool(synthetic),
        "provenance": dict(provenance or _fixture_provenance()),
    }
    return annotate_row(row)


def make_diverse_fixture_corpus(n_prompts_per_domain: int = 4, positions: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    layers = (2, 3, 31, 63)
    for d_i, domain in enumerate(CAPABILITY_DOMAINS):
        domain_prompts = [p for p in CAPTURE_PROMPTS if p[0] == domain]
        for p_i in range(n_prompts_per_domain):
            text = domain_prompts[p_i % len(domain_prompts)][1]
            prompt_id = f"{domain}:p{p_i:02d}"
            seq_len = max(positions, 6)
            pos_list = [0, seq_len // 2, seq_len - 1][:positions]
            for pos in pos_list:
                layer = layers[idx % len(layers)]
                rows.append(
                    make_fixture_row(
                        row_id=f"fx-{idx:04d}",
                        layer=layer,
                        prompt_id=prompt_id,
                        prompt_text=text,
                        token_position=pos,
                        seq_len=seq_len,
                        capability_domain=domain,
                        payload_seed=f"{prompt_id}|{pos}|{layer}".encode(),
                    )
                )
                idx += 1
    return rows


def make_gaussian_row(template: Mapping[str, Any]) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(0))
    x = rng.normal(size=HIDDEN).astype(np.float32)
    y = rng.normal(size=HIDDEN).astype(np.float32)
    row = dict(template)
    row["row_id"] = str(template.get("row_id")) + "-gaussian"
    row["x_sha256"] = vector_sha256(x)
    row["y_sha256"] = vector_sha256(y)
    row["synthetic"] = True
    row["x_generator"] = "gaussian"
    row["provenance"] = {
        "kind": "gaussian",
        "authority": "STATIC_ONLY",
        "source_path": "np.random.Generator.normal",
        "source_sha256": None,
        "capture_tool": RECORDED_BY,
        "note": "NNS-001 negative control; must be refused",
    }
    return annotate_row(row)


# ---------------------------------------------------------------------------
# Real capture
# ---------------------------------------------------------------------------


def load_x_f16(capture_dir: Path, layer: int) -> np.ndarray:
    path = capture_dir / f"L{layer:02d}.f16"
    if not path.is_file():
        raise CaptureUnavailable(f"REFUSED: missing {path}")
    raw = np.fromfile(path, dtype="<f2")
    if raw.size % HIDDEN:
        raise CaptureUnavailable(f"REFUSED: {path} size {raw.size} is not a multiple of {HIDDEN}")
    return raw.reshape(-1, HIDDEN)


def load_x_manifest(capture_dir: Path) -> dict[str, Any]:
    return load_json(capture_dir / "manifest.json")


def expand_capture_rows(
    *,
    layer: int,
    x_manifest: Mapping[str, Any],
    n_tokens: int,
) -> list[dict[str, Any]]:
    """One metadata row per token of capture_diverse2 at this layer."""
    rows: list[dict[str, Any]] = []
    covered = 0
    for prompt in x_manifest["manifest"]:
        family = str(prompt["family"])
        domain = FAMILY_TO_DOMAIN.get(family)
        if domain is None:
            raise CaptureUnavailable(f"REFUSED: unknown capture family {family!r}")
        prompt_id = f"{family}:{int(prompt['prompt_idx']):02d}"
        n = int(prompt["n_tokens"])
        start = int(prompt["row_start"])
        if start < 0 or start + n > n_tokens:
            raise CaptureUnavailable(
                f"REFUSED: prompt {prompt_id} overruns X rows ({start}+{n} > {n_tokens})"
            )
        for pos in range(n):
            rows.append(
                {
                    "layer": int(layer),
                    "prompt_id": prompt_id,
                    "prompt_family": family,
                    "prompt_idx": int(prompt["prompt_idx"]),
                    "token_position": pos,
                    "seq_len": n,
                    "position_band": position_band(pos, n),
                    "capability_domain": domain,
                    "x_row_index": start + pos,
                }
            )
        covered += n
    if covered != n_tokens:
        raise CaptureUnavailable(
            f"REFUSED: manifest covers {covered} tokens, X has {n_tokens}"
        )
    return rows


def capture_dir_complete(payload_dir: Path | None = None) -> bool:
    root = payload_dir if payload_dir is not None else PAYLOAD_DIR
    marker = root / "CAPTURE.json"
    return marker.is_file()


def write_f32(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(arr, dtype="<f4").tofile(path)
    return sha256_file(path)


def run_capture(
    *,
    payload_dir: Path | None = None,
    layers: Sequence[int] | None = None,
    x_dir: Path | None = None,
) -> dict[str, Any]:
    """F_l(X) on real post_attn_norm X for the representative layers."""
    started = time.perf_counter()
    root = payload_dir if payload_dir is not None else PAYLOAD_DIR
    root.mkdir(parents=True, exist_ok=True)
    x_capture = x_dir if x_dir is not None else resolve_x_capture_dir()
    x_man = load_x_manifest(x_capture)
    if int(x_man.get("hidden") or 0) != HIDDEN:
        raise CaptureUnavailable(f"REFUSED: X hidden {x_man.get('hidden')} != {HIDDEN}")
    if str(x_man.get("input") or "") != "post_attn_norm":
        raise CaptureUnavailable(
            f"REFUSED: X input is {x_man.get('input')!r}, want post_attn_norm "
            "(CORRECTION_MLP_INPUT_TENSOR)"
        )
    fps = fingerprint_layers()
    reps = pick_representatives(fps)
    chosen = list(layers) if layers is not None else list(reps["chosen_layers"])
    spec = specimen_identity()
    all_rows: list[dict[str, Any]] = []
    layer_files: list[dict[str, Any]] = []

    x_man_sha = sha256_file(x_capture / "manifest.json")
    for layer in chosen:
        x_f16 = load_x_f16(x_capture, layer)
        n_tokens = int(x_f16.shape[0])
        meta = expand_capture_rows(layer=layer, x_manifest=x_man, n_tokens=n_tokens)
        x = np.ascontiguousarray(x_f16, dtype=np.float32)
        del x_f16
        y = swiglu_f(x, layer)
        if y.shape != x.shape:
            raise CaptureUnavailable(f"REFUSED: Y shape {y.shape} != X shape {x.shape}")
        x_path = root / f"L{layer:02d}_x.f32"
        y_path = root / f"L{layer:02d}_y.f32"
        x_file_sha = write_f32(x_path, x)
        y_file_sha = write_f32(y_path, y)
        x_src = x_capture / f"L{layer:02d}.f16"
        prov = provenance_captured(
            source_path=str(x_src),
            source_sha256=sha256_file(x_src),
            layer=layer,
        )
        for rec in meta:
            i = int(rec["x_row_index"])
            row = {
                "row_id": f"L{layer:02d}-{i:06d}",
                "layer": layer,
                "prompt_id": rec["prompt_id"],
                "prompt_family": rec["prompt_family"],
                "prompt_idx": rec["prompt_idx"],
                "token_position": rec["token_position"],
                "seq_len": rec["seq_len"],
                "position_band": rec["position_band"],
                "capability_domain": rec["capability_domain"],
                "x_row_index": i,
                "x_path": str(x_path.relative_to(REPO)),
                "y_path": str(y_path.relative_to(REPO)),
                "x_sha256": vector_sha256(x[i]),
                "y_sha256": vector_sha256(y[i]),
                "synthetic": False,
                "provenance": prov,
            }
            all_rows.append(annotate_row(row))
        layer_files.append(
            {
                "layer": layer,
                "n_rows": n_tokens,
                "x_path": str(x_path.relative_to(REPO)),
                "y_path": str(y_path.relative_to(REPO)),
                "x_sha256": x_file_sha,
                "y_sha256": y_file_sha,
                "x_source_f16": str(x_src),
                "x_source_f16_sha256": sha256_file(x_src),
            }
        )
        del x, y

    split = split_by_prompt(all_rows, hold_frac=HOLD_FRAC)
    manifest = emit_manifest(
        all_rows,
        split,
        min_train_rows=MIN_TRAIN_ROWS_DETERMINED,
        allow_fixture=False,
        require_sizing=True,
    )
    # Row vectors stay in the .f32 files, not in git JSON.
    row_table = [{k: v for k, v in r.items() if k != "prompt_text"} for r in manifest["rows"]]
    rows_path = root / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for rec in row_table:
            handle.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    elapsed = time.perf_counter() - started
    capture_doc = {
        "schema": "hawking.future.mlp_teacher_corpus.capture.v1",
        "status": "captured",
        "payload_dir": str(root.relative_to(REPO)),
        "x_capture_dir": str(x_capture),
        "x_manifest_sha256": x_man_sha,
        "x_input": "post_attn_norm",
        "y_function": "down(silu(gate(x))*up(x)) affine-Q2 HGRAVF01",
        "specimen": spec,
        "fusion_env": FUSION_ENV,
        "layers": layer_files,
        "n_rows": manifest["n_rows"],
        "n_train_rows": manifest["n_train_rows"],
        "n_hold_rows": manifest["n_hold_rows"],
        "n_train_rows_per_layer": manifest["n_train_rows_per_layer"],
        "rows_per_dimension_train": manifest["rows_per_dimension_train"],
        "rows_per_dimension_train_per_layer": manifest["rows_per_dimension_train_per_layer"],
        "rows_per_dimension_train_pooled": manifest["rows_per_dimension_train_pooled"],
        "rows_per_dimension_all": manifest["rows_per_dimension_all"],
        "beats_nns007_scar": manifest["beats_nns007_scar"],
        "rank32_overdetermined": manifest["rank32_overdetermined"],
        "n_prompts_per_domain": manifest["n_prompts_per_domain"],
        "n_rows_per_domain": manifest["n_rows_per_domain"],
        "position_bands": manifest["position_bands"],
        "split": manifest["split"],
        "rows_jsonl": str(rows_path.relative_to(REPO)),
        "rows_jsonl_sha256": sha256_file(rows_path),
        "representatives": reps,
        "capture_elapsed_s": elapsed,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (root / "CAPTURE.json").write_text(json.dumps(capture_doc, indent=1, sort_keys=True) + "\n")
    # Drop the in-memory vectors from the returned manifest; keep counts.
    slim = dict(manifest)
    slim["rows"] = row_table
    slim["payload"] = capture_doc
    return slim


def load_existing_capture(payload_dir: Path | None = None) -> dict[str, Any] | None:
    root = payload_dir if payload_dir is not None else PAYLOAD_DIR
    marker = root / "CAPTURE.json"
    if not marker.is_file():
        return None
    return load_json(marker)


def refresh_capture_doc(payload_dir: Path | None = None) -> dict[str, Any] | None:
    """Recompute sizing/split fields from rows.jsonl without re-running F."""
    root = payload_dir if payload_dir is not None else PAYLOAD_DIR
    rows_path = root / "rows.jsonl"
    existing = load_existing_capture(root)
    if existing is None or not rows_path.is_file():
        return existing
    rows = [json.loads(line) for line in rows_path.open(encoding="utf-8")]
    train_ids = sorted({str(r["prompt_id"]) for r in rows if r.get("split") == "train"})
    hold_ids = sorted({str(r["prompt_id"]) for r in rows if r.get("split") == "hold"})
    split = {
        "train_prompt_ids": train_ids,
        "hold_prompt_ids": hold_ids,
        "hold_frac": HOLD_FRAC,
        "split_unit": "prompt_id",
        "n_train_prompts": len(train_ids),
        "n_hold_prompts": len(hold_ids),
    }
    manifest = emit_manifest(
        rows,
        split,
        min_train_rows=MIN_TRAIN_ROWS_DETERMINED,
        allow_fixture=False,
        require_sizing=True,
    )
    existing.update(
        {
            "specimen": specimen_identity(),
            "fusion_env": FUSION_ENV,
            "n_rows": manifest["n_rows"],
            "n_train_rows": manifest["n_train_rows"],
            "n_hold_rows": manifest["n_hold_rows"],
            "n_train_rows_per_layer": manifest["n_train_rows_per_layer"],
            "rows_per_dimension_train": manifest["rows_per_dimension_train"],
            "rows_per_dimension_train_per_layer": manifest["rows_per_dimension_train_per_layer"],
            "rows_per_dimension_train_pooled": manifest["rows_per_dimension_train_pooled"],
            "rows_per_dimension_all": manifest["rows_per_dimension_all"],
            "beats_nns007_scar": manifest["beats_nns007_scar"],
            "rank32_overdetermined": manifest["rank32_overdetermined"],
            "n_prompts_per_domain": manifest["n_prompts_per_domain"],
            "n_rows_per_domain": manifest["n_rows_per_domain"],
            "position_bands": manifest["position_bands"],
            "split": manifest["split"],
        }
    )
    (root / "CAPTURE.json").write_text(json.dumps(existing, indent=1, sort_keys=True) + "\n")
    return existing


# ---------------------------------------------------------------------------
# Selftest + receipt
# ---------------------------------------------------------------------------


def selftest() -> dict[str, Any]:
    diverse = make_diverse_fixture_corpus(4, 3)
    split = split_by_prompt(diverse)
    ok = emit_manifest(diverse, split, allow_fixture=True, require_sizing=False)
    if not ok["accepted"]:
        raise SystemExit(f"selftest: diverse fixture must emit, got {ok}")
    if set(ok["split"]["train_prompt_ids"]) & set(ok["split"]["hold_prompt_ids"]):
        raise SystemExit("selftest: constructor leaked prompt ids")

    leaked = {
        "train_prompt_ids": list(split["train_prompt_ids"]) + list(split["hold_prompt_ids"][:1]),
        "hold_prompt_ids": list(split["hold_prompt_ids"]),
        "hold_frac": split["hold_frac"],
    }
    leak_refused = False
    leak_codes: list[str] = []
    try:
        emit_manifest(diverse, leaked, allow_fixture=True)
    except CorpusRefused as exc:
        leak_refused = True
        leak_codes = list(exc.codes)
    else:
        raise SystemExit("selftest: leaked split was NOT refused — the guard is dead")
    if "HELD_OUT_PROMPT_LEAK" not in leak_codes:
        raise SystemExit(f"selftest: expected HELD_OUT_PROMPT_LEAK, got {leak_codes}")

    gauss = list(diverse)
    gauss[0] = make_gaussian_row(diverse[0])
    syn_refused = False
    syn_codes: list[str] = []
    try:
        emit_manifest(gauss, split, allow_fixture=True)
    except CorpusRefused as exc:
        syn_refused = True
        syn_codes = list(exc.codes)
    else:
        raise SystemExit("selftest: gaussian row was NOT refused — NNS-001 guard is dead")
    if "SYNTHETIC_ROW" not in syn_codes:
        raise SystemExit(f"selftest: expected SYNTHETIC_ROW, got {syn_codes}")

    fps = fingerprint_layers()
    reps = pick_representatives(fps)
    if 0 in reps["chosen_layers"] and reps["chosen"][0]["role"] == "typical":
        raise SystemExit("selftest: layer 0 was picked as typical")
    if reps["layer0"]["typical"] is not False:
        raise SystemExit("selftest: layer 0 must be marked not typical")

    return {
        "diverse_accepted": True,
        "diverse_n": ok["n_rows"],
        "diverse_unique": ok["n_unique_content"],
        "held_out_leak_refused": leak_refused,
        "held_out_leak_codes": leak_codes,
        "synthetic_refused": syn_refused,
        "synthetic_codes": syn_codes,
        "fingerprint_n_layers": len(fps),
        "chosen_layers": reps["chosen_layers"],
        "layer0_typical": reps["layer0"]["typical"],
        "min_train_rows_determined": MIN_TRAIN_ROWS_DETERMINED,
        "nns007_scar_rows_per_dim": NNS007_SCAR_ROWS_PER_DIM,
    }


def _fingerprint_block() -> dict[str, Any]:
    fps = fingerprint_layers()
    reps = pick_representatives(fps)
    return {
        "n_layers": len(fps),
        "layers": fps,
        "representatives": reps,
    }


def build(*, capture: bool = False) -> Path:
    test = selftest()
    fp = _fingerprint_block()
    spec = specimen_identity()
    existing = refresh_capture_doc() or load_existing_capture()
    if capture:
        existing = run_capture(
            layers=fp["representatives"]["chosen_layers"],
        )["payload"]

    domain_plan = {
        domain: sum(1 for d, _ in CAPTURE_PROMPTS if d == domain) for domain in CAPABILITY_DOMAINS
    }
    capture_block: dict[str, Any]
    if existing:
        capture_block = {
            k: v
            for k, v in existing.items()
            if k != "rows"
        }
        capture_block["status"] = existing.get("status") or "captured"
    else:
        capture_block = {
            "status": "not_run",
            "payload_dir": str(PAYLOAD_DIR.relative_to(REPO)),
            "note": (
                "Fingerprint and guards are in this receipt. Activation payloads "
                "are written by --capture into the gitignored payload_dir. "
                "pytest does not run the GEMM."
            ),
        }

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Functional MLP teacher corpus for sealed-3.14: real post_attn_norm X "
            "and exact F(X) under the affine-Q2 packing, held out by prompt, "
            "sized so a rank-32 fit over 5120 dims is not underdetermined."
        ),
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "scars": {
            "NNS-001": "Gaussian-proxy method is dead; never evaluate F on synthetic X.",
            "NNS-007": (
                f"Median {NNS007_SCAR_ROWS} rows against {NNS007_SCAR_DIM} dims "
                f"({NNS007_SCAR_ROWS_PER_DIM:.5f} rows/dim) is not a codec score. "
                f"Rank-32 needs n_train >= {MIN_TRAIN_ROWS_DETERMINED}."
            ),
            "CORRECTION_MLP_INPUT_TENSOR": (
                "MLP input is post_attn_norm, not post_input_norm (121% relative)."
            ),
            "NNS-015": "Layer 31 down_proj is the named distillation surface; included.",
        },
        "specimen": spec,
        "fusion_env": FUSION_ENV,
        "organ": {
            "function": "F(x)=down(silu(gate(x))*up(x))",
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "n_layers": N_LAYERS,
            "packing": "affine_q2_group64_fp16_scale_bias",
            "reconstruction": "w = float(q) * scale + bias, q unsigned in {0,1,2,3}",
        },
        "capability_domains": list(CAPABILITY_DOMAINS),
        "n_catalog_prompts_per_domain": domain_plan,
        "family_to_domain": dict(FAMILY_TO_DOMAIN),
        "fingerprint": fp,
        "sizing": {
            "rank": RANK,
            "hidden": HIDDEN,
            "rank32_params": RANK32_PARAMS,
            "min_train_rows_determined": MIN_TRAIN_ROWS_DETERMINED,
            "min_train_rows_rank": MIN_TRAIN_ROWS_RANK,
            "nns007_scar_rows": NNS007_SCAR_ROWS,
            "nns007_scar_dim": NNS007_SCAR_DIM,
            "nns007_scar_rows_per_dim": NNS007_SCAR_ROWS_PER_DIM,
            "dup_rate_max": DUP_RATE_MAX,
            "hold_frac": HOLD_FRAC,
            "split_unit": "prompt_id",
        },
        "payload_dir": str(PAYLOAD_DIR.relative_to(REPO)),
        "payload_gitignored": True,
        "capture": capture_block,
        "selftest": test,
        "anti_fabrication": {
            "detectors": [
                "HELD_OUT_PROMPT_LEAK",
                "SYNTHETIC_ROW",
                "DUPLICATE_ROWS_ABOVE_THRESHOLD",
                "PROMPT_NOT_IN_SPLIT",
            ],
            "dup_rate_max": DUP_RATE_MAX,
            "synthetic_kinds": sorted(SYNTHETIC_KINDS),
            "loud_exception": "CorpusRefused",
            "rule": (
                "emit_manifest raises CorpusRefused if the hold set shares a "
                "prompt_id with train, or if any row is synthetic/Gaussian. A "
                "return-flag nobody checks is not a guard."
            ),
        },
        "gaps_closed": [
            "64 MLP layers fingerprinted from real H(q); layer 0 is not treated as typical.",
            "Representatives chosen with mixer-class, depth, and entropy-pole evidence.",
            "Held-out split is by prompt_id; emit_manifest refuses a leak.",
            "Synthetic / Gaussian rows are refused (NNS-001).",
            "Duplicate rows above 5% refuse the emit.",
            "Rank-32 sizing floor is 64 train rows (n * 5120 >= 32*(10240-32)).",
            "X is post_attn_norm, Y is sealed affine-Q2 SwiGLU; payloads are not in git.",
        ],
        "what_this_does_not_prove": [
            "That a small program actually fits F at q3 quality (that is the next experiment).",
            "That X equals the sealed residual stream (the JSONL resident does not dump it).",
            "A protected complete-token or TPS number.",
        ],
        "era_vocabulary": {
            "evidence_class": "DIAGNOSTIC_RELATIVE",
            "bench_state": "UNKNOWN",
        },
    }
    # Timing of a capture is diagnostic; do not use hardware field names.
    if capture_block.get("status") == "captured":
        doc["bench"] = {
            "state": "UNKNOWN",
            "measurement_state": "DIAGNOSTIC_RELATIVE",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recorded_by": RECORDED_BY,
            "machine": "Apple host; CPU/MPS affine-Q2 F(X), no GPU lease",
            "gpu_authority": False,
            "rule": "no hardware measurement claim without hardware",
        }
    out = write_receipt(RECEIPT, doc, RECORDED_BY)
    written = load_json(out)
    if written.get("schema") != SCHEMA or not written.get("seal_sha256"):
        raise SystemExit(f"receipt {out} failed round-trip")
    return out


selftest_alias = selftest


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else _sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--fingerprint", action="store_true")
    args = parser.parse_args(argv_list)
    if args.fingerprint:
        json.dump(_fingerprint_block(), _sys.stdout, indent=2, sort_keys=True)
        _sys.stdout.write("\n")
        return 0
    if args.selftest:
        json.dump(selftest(), _sys.stdout, indent=2, sort_keys=True)
        _sys.stdout.write("\n")
        return 0
    if args.capture:
        out = build(capture=True)
        print(out)
        return 0
    if args.build or not argv_list:
        out = build(capture=False)
        print(out)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(_sys.argv[1:]))

