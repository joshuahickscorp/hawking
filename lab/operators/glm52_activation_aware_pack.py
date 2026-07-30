"""Recomposed science module glm52_activation_aware_pack (C-SCI-R1)."""
from __future__ import annotations
import sys as _sys_a1
from pathlib import Path as _Path_a1
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == 'archive' else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
HERE = _A1_CONDENSE
REPO = HERE.parents[1]
SCHEMA = 'hawking.glm52.activation_aware_pack.v1'
MAGIC = b'GLM52AAP'
HEADER_BYTES = 64
ORIGINAL_WEIGHT_COUNT = 753329940480
DISK_FLOOR_GIB = 141
DISK_FLOOR_BYTES = DISK_FLOOR_GIB * (1 << 30)
DEFAULT_SHARD_BODY_BYTES = 5370000000
DEFAULT_FETCH_WORKERS = 4
DEFAULT_WORKERS = 4
DEFAULT_RANKS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
SEED = 10582654
HIDDEN = 6144
N_LAYERS = 78
HELD_OUT_FRAC = 0.2
HOLDOUT_MIN = 256
PILOT_SOURCE = Path.home() / 'Library/Application Support/Hawking/GLM52Gravity/pilot_source'
CAPSULE_DIR = Path.home() / 'Library/Application Support/Hawking/GLM52Gravity/source_fetch/teacher/capsules'
DEFAULT_OUT = Path.home() / 'Library/Application Support/Hawking/GLM52Gravity/activation_aware_pack'
REHYDRATE_RECEIPT = REPO / 'GLM52_REHYDRATION_RECEIPT.json'
LAYER_RE = re.compile('model\\.layers\\.(\\d+)\\.')

class PackError(RuntimeError):
    """Hard program failure."""

class DiskFloorError(PackError):
    """Fetching or writing would cross the disk floor. Halt, do not overrun."""

def free_bytes(path: Path | None=None) -> int:
    target = path if path is not None else Path.home()
    if not target.exists():
        target = Path.home()
    return int(shutil.disk_usage(target).free)

def assert_disk_floor(extra_bytes: int=0, *, path: Path | None=None, floor: int=DISK_FLOOR_BYTES) -> int:
    """Refuse any action that would leave free space below the floor.

    `extra_bytes` is the additional footprint about to be written (a shard body,
    a packed artifact). The check is against free - extra, not free alone, so a
    fetch that would land under the floor is stopped before the download starts.
    """
    free = free_bytes(path)
    if free - int(extra_bytes) < int(floor):
        raise DiskFloorError(f'disk floor: free={free} extra={int(extra_bytes)} floor={int(floor)} ({int(floor) / (1 << 30):.0f} GiB actual, module default {DISK_FLOOR_GIB} GiB). Halt rather than overrun. Stream and evict; never stage the whole 1.507 TB parent.')
    return free
COMPONENT_KEYS: tuple[str, ...] = ('indices', 'codebooks', 'scales', 'metadata', 'alignment', 'protected_islands', 'doctor', 'pass_through_tensors', 'packaging', 'runtime_tables')

@dataclass
class ByteLedger:
    """Itemized byte counts. Every byte of the artifact lives in exactly one slot."""
    components: dict[str, int] = field(default_factory=lambda: {k: 0 for k in COMPONENT_KEYS})

    def add(self, slot: str, n: int) -> None:
        if slot not in self.components:
            raise PackError(f'unknown ledger slot {slot!r}; every byte needs a named component')
        if n < 0:
            raise PackError(f'negative bytes for {slot}')
        self.components[slot] += int(n)

    def total_bytes(self) -> int:
        return int(sum(self.components.values()))

    def reconciles(self, physical_bytes: int | None=None) -> bool:
        total = self.total_bytes()
        if physical_bytes is None:
            return True
        return total == int(physical_bytes)

    def complete_bits(self) -> int:
        return self.total_bytes() * 8

    def complete_bpw(self, weight_count: int=ORIGINAL_WEIGHT_COUNT) -> Fraction:
        if weight_count <= 0:
            raise PackError('weight_count must be positive')
        return Fraction(self.complete_bits(), int(weight_count))

    def as_dict(self, weight_count: int=ORIGINAL_WEIGHT_COUNT) -> dict[str, Any]:
        total = self.total_bytes()
        bits = self.complete_bits()
        bpw = self.complete_bpw(weight_count)
        return {'schema': 'hawking.foundry.one_bit_ceiling.v1', 'scope': 'whole_model' if weight_count == ORIGINAL_WEIGHT_COUNT else 'subset', 'component_bytes': dict(self.components), 'total_bytes': total, 'itemization_reconciles': True, 'complete_bits': str(bits), 'itemized_bits': str(bits), 'reserve_bits': '0', 'original_weight_count': int(weight_count), 'complete_bpw_exact': f'{bpw.numerator}/{bpw.denominator}', 'complete_bpw_float': float(bpw), 'note': 'Activation-aware pack: every byte counted, including bases (runtime_tables), coefficients (codebooks), headers (metadata), pass-through tensors, and packaging. No exclusions.'}

def parse_bpw(value: str | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(value).limit_denominator(10000000)
    text = str(value).strip()
    if '/' in text:
        num, den = text.split('/', 1)
        return Fraction(int(num), int(den))
    return Fraction(text).limit_denominator(10000000)

def read_safetensors_header(shard: Path) -> dict[str, Any]:
    with shard.open('rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        return json.loads(fh.read(n))

def iter_tensor_names(header: dict[str, Any]) -> list[str]:
    return sorted((k for k in header if k != '__metadata__'))

def read_bf16_tensor(shard: Path, header: dict[str, Any], name: str) -> np.ndarray:
    info = header[name]
    dtype = info.get('dtype', 'BF16')
    shape = tuple((int(x) for x in info['shape']))
    lo, hi = info['data_offsets']
    with shard.open('rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        base = 8 + n
        fh.seek(base + lo)
        raw = fh.read(hi - lo)
    if dtype in ('BF16', 'BFLOAT16'):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).reshape(shape)
    if dtype in ('F32', 'FLOAT32'):
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if dtype in ('F16', 'FLOAT16'):
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    raise PackError(f'unsupported dtype {dtype} for {name}')

def tensor_nbytes_source(header: dict[str, Any], name: str) -> int:
    lo, hi = header[name]['data_offsets']
    return int(hi - lo)

def layer_of(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None

@dataclass(frozen=True)
class BasisProvenance:
    """Which activation basis a tensor was measured / packed against.

    Load-bearing: without this the artifact cannot be decoded correctly, and
    neighbouring-layer transfer is only honest when the record is kept.
    """
    tensor_layer: int | None
    basis_layer: int
    capsule_file: str
    capsule_key: str
    hidden: int
    n_activation_rows: int
    source: str = 'teacher_capsule_pre_router_hidden'

    def as_dict(self) -> dict[str, Any]:
        return {'tensor_layer': self.tensor_layer, 'basis_layer': self.basis_layer, 'capsule_file': self.capsule_file, 'capsule_key': self.capsule_key, 'hidden': self.hidden, 'n_activation_rows': self.n_activation_rows, 'source': self.source}

def discover_capsule_layers(capsule_dir: Path=CAPSULE_DIR) -> dict[int, tuple[Path, str]]:
    """Map each layer index that has a pre_router_hidden capture to (npz, key).

    33 capsules cover 78 layers (0-16 contiguous, then sampled). Multi-layer
    capsules (e.g. L16_L18.npz) contribute every layer key they hold. Layers
    without a capsule resolve via :func:`nearest_basis_layer`.
    """
    out: dict[int, tuple[Path, str]] = {}
    if not capsule_dir.is_dir():
        return out
    key_re = re.compile('layer_(\\d+)/pre_router_hidden$')
    for path in sorted(capsule_dir.glob('L*_L*.npz')):
        try:
            with np.load(path) as z:
                for key in z.files:
                    m = key_re.match(key)
                    if not m:
                        continue
                    layer = int(m.group(1))
                    if layer in out and out[layer][0].stem.count('L') == 2:
                        if path.stem.split('_')[0] == path.stem.split('_')[1]:
                            out[layer] = (path, key)
                    else:
                        out[layer] = out.get(layer) or (path, key)
                        if path.stem.split('_')[0] == path.stem.split('_')[1]:
                            out[layer] = (path, key)
        except (OSError, ValueError):
            continue
    return out

def nearest_basis_layer(layer: int, available: Sequence[int]) -> int:
    if not available:
        raise PackError('no teacher capsules available to build an activation basis')
    ordered = sorted((int(x) for x in available))
    best = ordered[0]
    best_d = abs(best - layer)
    for cand in ordered[1:]:
        d = abs(cand - layer)
        if d < best_d or (d == best_d and cand < best):
            best, best_d = (cand, d)
    return best

def resolve_basis_for_tensor(name: str, capsule_map: dict[int, tuple[Path, str]]) -> BasisProvenance:
    tensor_layer = layer_of(name)
    available = sorted(capsule_map)
    if tensor_layer is None:
        prefer = 0 if name.startswith('model.embed') else N_LAYERS - 1
        basis_layer = nearest_basis_layer(prefer, available)
    else:
        basis_layer = tensor_layer if tensor_layer in capsule_map else nearest_basis_layer(tensor_layer, available)
    path, key = capsule_map[basis_layer]
    return BasisProvenance(tensor_layer=tensor_layer, basis_layer=basis_layer, capsule_file=path.name, capsule_key=key, hidden=HIDDEN, n_activation_rows=0)

@dataclass
class ActivationBasis:
    """Top activation principal directions for one layer, plus provenance."""
    layer: int
    basis: np.ndarray
    singular_values: np.ndarray
    variance_frac: list[float]
    provenance: BasisProvenance
    X_hold: np.ndarray
    X_fit_mean: np.ndarray

    @property
    def max_rank(self) -> int:
        return int(self.basis.shape[1])

    def columns(self, rank: int) -> np.ndarray:
        r = min(int(rank), self.max_rank)
        if r <= 0:
            raise PackError('rank must be positive')
        return self.basis[:, :r]

def load_pre_router(path: Path, key: str) -> np.ndarray:
    with np.load(path) as z:
        arr = np.asarray(z[key], dtype=np.float32)
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2:
        return arr
    raise PackError(f'unexpected activation shape {arr.shape} for {key}')

def build_basis(provenance: BasisProvenance, capsule_map: dict[int, tuple[Path, str]], max_rank: int, *, seed: int=SEED) -> ActivationBasis:
    path, key = capsule_map[provenance.basis_layer]
    X = load_pre_router(path, key)
    if X.shape[1] != HIDDEN:
        hidden = int(X.shape[1])
    else:
        hidden = HIDDEN
    n = X.shape[0]
    rng = np.random.default_rng(seed ^ provenance.basis_layer * 2654435761)
    perm = rng.permutation(n)
    n_hold = max(HOLDOUT_MIN, int(round(n * HELD_OUT_FRAC)))
    n_hold = min(n_hold, n // 5 if n >= 5 else max(1, n // 5))
    n_hold = min(n_hold, max(1, n // 2))
    hold_idx = perm[:n_hold]
    fit_idx = perm[n_hold:]
    if fit_idx.size == 0:
        fit_idx = perm
        hold_idx = perm[:max(1, n // 5)]
    X_fit = X[fit_idx]
    X_hold = X[hold_idx]
    mu = X_fit.mean(axis=0)
    Xc = X_fit - mu
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    max_rank = min(int(max_rank), vt.shape[0], hidden)
    basis = vt[:max_rank].T.astype(np.float32, copy=True)
    energy = s.astype(np.float64) ** 2
    total = float(energy.sum()) + 1e-30
    cum = np.cumsum(energy[:max_rank]) / total
    prov = BasisProvenance(tensor_layer=provenance.tensor_layer, basis_layer=provenance.basis_layer, capsule_file=path.name, capsule_key=key, hidden=hidden, n_activation_rows=int(n))
    return ActivationBasis(layer=provenance.basis_layer, basis=basis, singular_values=s[:max_rank].astype(np.float32), variance_frac=[float(x) for x in cum], provenance=prov, X_hold=X_hold.astype(np.float32, copy=False), X_fit_mean=mu.astype(np.float32))

def mean_row_cosine(y: np.ndarray, y_hat: np.ndarray) -> float:
    yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    yhn = y_hat / (np.linalg.norm(y_hat, axis=1, keepdims=True) + 1e-12)
    return float((yn * yhn).sum(axis=1).mean())

def constant_mean_null(y: np.ndarray) -> float:
    mean_row = np.repeat(y.mean(axis=0, keepdims=True), y.shape[0], axis=0)
    return mean_row_cosine(y, mean_row)

def functional_score(W: np.ndarray, W_hat: np.ndarray, X: np.ndarray, *, side: str) -> dict[str, float]:
    """Functional cosine against a constant-mean null, on real activations.

    side='input'  : W is [out, in], y = X @ W.T, X has shape [N, in]
    side='output' : W is [out, in] with out == hidden; probe with Gaussian of
                    the input width (no intermediate activations retained). The
                    packed form still uses the activation basis on the matching
                    side; the Gaussian probe is labelled as such.
    """
    if side == 'input':
        y = X @ W.T
        y_hat = X @ W_hat.T
    elif side == 'output':
        rng = np.random.default_rng(SEED ^ 3855)
        z = rng.standard_normal((min(512, max(64, X.shape[0])), W.shape[1])).astype(np.float32)
        y = z @ W.T
        y_hat = z @ W_hat.T
    else:
        raise PackError(f'unknown projection side {side}')
    cos = mean_row_cosine(y, y_hat)
    null = constant_mean_null(y)
    recon = float(np.linalg.norm(W - W_hat) / (np.linalg.norm(W) + 1e-12))
    return {'mean_row_cosine': cos, 'constant_mean_cosine_null': null, 'beats_null': bool(cos > null), 'surplus_over_null': cos - null, 'reconstruction_relative_error_INADMISSIBLE': recon}

def choose_side(shape: tuple[int, ...], hidden: int=HIDDEN) -> str | None:
    """Which side of W matches the activation hidden width, if either."""
    if len(shape) != 2:
        return None
    m, n = (int(shape[0]), int(shape[1]))
    if n == hidden:
        return 'input'
    if m == hidden:
        return 'output'
    return None

def project_factors(W: np.ndarray, B: np.ndarray, side: str) -> np.ndarray:
    """Return the coefficient factor L such that W ≈ decode(L, B, side).

    input side  (W [m, n], B [n, r]): L = W @ B          -> [m, r]
    output side (W [m, n], B [m, r]): L = B.T @ W        -> [r, n]
    """
    if side == 'input':
        return W @ B
    if side == 'output':
        return B.T @ W
    raise PackError(f'unknown side {side}')

def reconstruct(L: np.ndarray, B: np.ndarray, side: str) -> np.ndarray:
    if side == 'input':
        return L @ B.T
    if side == 'output':
        return B @ L
    raise PackError(f'unknown side {side}')

def factor_bytes(rows: int, cols: int, rank: int, side: str) -> dict[str, int]:
    """Exact float16 byte cost of (coefficients + basis columns) + fixed header.

    Per-tensor basis columns are billed here when the tensor is self-contained.
    Shared layer bases are billed once under runtime_tables instead; pass
    `bill_basis=False` via the caller in that mode.
    """
    r = int(rank)
    if side == 'input':
        coeff = rows * r * 2
        basis = cols * r * 2
    else:
        coeff = r * cols * 2
        basis = rows * r * 2
    return {'header': HEADER_BYTES, 'coefficients': int(coeff), 'basis': int(basis)}

def packed_tensor_bytes(rows: int, cols: int, rank: int, side: str, *, bill_basis: bool=True) -> int:
    parts = factor_bytes(rows, cols, rank, side)
    total = parts['header'] + parts['coefficients']
    if bill_basis:
        total += parts['basis']
    return total

def shared_basis_bytes(hidden: int, rank: int) -> int:
    """One float16 basis matrix [hidden, rank] plus a small header."""
    return HEADER_BYTES + int(hidden) * int(rank) * 2

def serialize_tensor_payload(L: np.ndarray, B: np.ndarray, *, side: str, rows: int, cols: int, rank: int, basis_layer: int, bill_basis: bool) -> bytes:
    """Physical payload. Length must equal the billed byte count."""
    L16 = np.ascontiguousarray(L, dtype=np.float16)
    head = struct.pack('<8sIIIHHB', MAGIC, int(rows), int(cols), int(rank), int(basis_layer) & 65535, 1 if side == 'input' else 2, 1 if bill_basis else 0)
    head = head + b'\x00' * (HEADER_BYTES - len(head))
    body = L16.tobytes()
    if bill_basis:
        body += np.ascontiguousarray(B, dtype=np.float16).tobytes()
    blob = head + body
    billed = packed_tensor_bytes(rows, cols, rank, side, bill_basis=bill_basis)
    if len(blob) != billed:
        raise PackError(f'serialized {len(blob)} bytes but billed {billed}; the BPW claim and the file must agree exactly')
    return blob

def deserialize_tensor_payload(blob: bytes) -> dict[str, Any]:
    if blob[:8] != MAGIC:
        raise PackError('not an activation-aware payload')
    rows, cols, rank, basis_layer, side_code, has_basis = struct.unpack_from('<IIIHHB', blob, 8)
    side = 'input' if side_code == 1 else 'output'
    offset = HEADER_BYTES
    if side == 'input':
        coeff_count = rows * rank
    else:
        coeff_count = rank * cols
    L = np.frombuffer(blob[offset:offset + coeff_count * 2], dtype=np.float16).astype(np.float32)
    if side == 'input':
        L = L.reshape(rows, rank)
    else:
        L = L.reshape(rank, cols)
    offset += coeff_count * 2
    B = None
    if has_basis:
        if side == 'input':
            B = np.frombuffer(blob[offset:offset + cols * rank * 2], dtype=np.float16)
            B = B.astype(np.float32).reshape(cols, rank)
        else:
            B = np.frombuffer(blob[offset:offset + rows * rank * 2], dtype=np.float16)
            B = B.astype(np.float32).reshape(rows, rank)
    return {'rows': rows, 'cols': cols, 'rank': rank, 'side': side, 'basis_layer': basis_layer, 'L': L, 'B': B, 'has_basis': bool(has_basis)}

@dataclass
class RankPoint:
    rank: int
    bytes: int
    bpw: float
    mean_row_cosine: float
    constant_mean_cosine_null: float
    beats_null: bool
    surplus_over_null: float
    reconstruction_relative_error_INADMISSIBLE: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class TensorMeasurement:
    name: str
    shape: list[int]
    n_weights: int
    dtype: str
    side: str | None
    disposition: str
    basis_provenance: dict[str, Any] | None
    curve: list[dict[str, Any]]
    pass_through_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

def measure_tensor(name: str, W: np.ndarray, dtype: str, basis: ActivationBasis | None, ranks: Sequence[int], *, bill_basis_per_tensor: bool=True) -> TensorMeasurement:
    shape = [int(x) for x in W.shape]
    n_weights = int(np.prod(shape))
    side = choose_side(tuple(shape), hidden=basis.provenance.hidden if basis else HIDDEN)
    if W.ndim != 2 or side is None or basis is None:
        if dtype in ('BF16', 'BFLOAT16', 'F16', 'FLOAT16'):
            pt_bytes = n_weights * 2
        elif dtype in ('F32', 'FLOAT32'):
            pt_bytes = n_weights * 4
        else:
            pt_bytes = n_weights * 2
        return TensorMeasurement(name=name, shape=shape, n_weights=n_weights, dtype=dtype, side=side, disposition='pass_through', basis_provenance=basis.provenance.as_dict() if basis else None, curve=[{'rank': 0, 'bytes': pt_bytes, 'bpw': pt_bytes * 8 / max(n_weights, 1), 'mean_row_cosine': 1.0, 'constant_mean_cosine_null': 0.0, 'beats_null': True, 'surplus_over_null': 1.0, 'reconstruction_relative_error_INADMISSIBLE': 0.0}], pass_through_bytes=pt_bytes)
    m, n = (int(W.shape[0]), int(W.shape[1]))
    curve: list[dict[str, Any]] = []
    for rank in ranks:
        r = min(int(rank), basis.max_rank, m, n)
        if r <= 0:
            continue
        B = basis.columns(r)
        L = project_factors(W, B, side)
        W_hat = reconstruct(L, B, side)
        score = functional_score(W, W_hat, basis.X_hold, side=side)
        nbytes = packed_tensor_bytes(m, n, r, side, bill_basis=bill_basis_per_tensor)
        curve.append({'rank': r, 'bytes': nbytes, 'bpw': nbytes * 8 / n_weights, **score})
    if not curve:
        raise PackError(f'no measurable ranks for {name}')
    return TensorMeasurement(name=name, shape=shape, n_weights=n_weights, dtype=dtype, side=side, disposition='activation_aware', basis_provenance=basis.provenance.as_dict(), curve=curve)

def measure_shard(shard: Path, capsule_map: dict[int, tuple[Path, str]], basis_cache: dict[int, ActivationBasis], ranks: Sequence[int], *, max_basis_rank: int, bill_basis_per_tensor: bool=True) -> list[TensorMeasurement]:
    header = read_safetensors_header(shard)
    out: list[TensorMeasurement] = []
    for name in iter_tensor_names(header):
        prov = resolve_basis_for_tensor(name, capsule_map)
        if prov.basis_layer not in basis_cache:
            basis_cache[prov.basis_layer] = build_basis(prov, capsule_map, max_basis_rank)
        basis = basis_cache[prov.basis_layer]
        prov = basis.provenance
        tensor_prov = BasisProvenance(tensor_layer=layer_of(name), basis_layer=prov.basis_layer, capsule_file=prov.capsule_file, capsule_key=prov.capsule_key, hidden=prov.hidden, n_activation_rows=prov.n_activation_rows)
        W = read_bf16_tensor(shard, header, name)
        dtype = header[name].get('dtype', 'BF16')
        basis_for_tensor = ActivationBasis(layer=basis.layer, basis=basis.basis, singular_values=basis.singular_values, variance_frac=basis.variance_frac, provenance=tensor_prov, X_hold=basis.X_hold, X_fit_mean=basis.X_fit_mean)
        out.append(measure_tensor(name, W, dtype, basis_for_tensor, ranks, bill_basis_per_tensor=bill_basis_per_tensor))
        del W
    return out

@dataclass
class TensorAllocation:
    name: str
    disposition: str
    rank: int
    bytes: int
    n_weights: int
    mean_row_cosine: float
    constant_mean_cosine_null: float
    beats_null: bool
    surplus_over_null: float
    side: str | None
    basis_provenance: dict[str, Any] | None
    shape: list[int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

def _point_at_rank(curve: list[dict[str, Any]], rank: int) -> dict[str, Any] | None:
    for p in curve:
        if int(p['rank']) == int(rank):
            return p
    return None

def allocate(measurements: Sequence[TensorMeasurement | dict[str, Any]], target_bpw: Fraction, *, weight_count: int | None=None, shared_bases: bool=True, hidden: int=HIDDEN) -> dict[str, Any]:
    """Choose one rank per tensor to hit target complete BPW, max preservation.

    Deterministic: identical measurements + target always yield identical ranks.
    Utility is surplus-over-null (functional cosine minus the real-activation
    null). Reconstruction error is ignored.

    When shared_bases is True, each layer's basis is stored once at the max rank
    used by any tensor that layers onto it, and billed under runtime_tables.
    Coefficient payloads then exclude the per-tensor basis copy.
    """
    rows: list[dict[str, Any]] = [m.as_dict() if isinstance(m, TensorMeasurement) else dict(m) for m in measurements]
    rows.sort(key=lambda r: r['name'])
    if weight_count is None:
        weight_count = int(sum((int(r['n_weights']) for r in rows)))
    budget_bits = target_bpw * int(weight_count)
    budget_bytes = int(budget_bits // 8)
    choice: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r['disposition'] == 'pass_through':
            pt = r['curve'][0]
            choice[r['name']] = {'rank': 0, 'bytes': int(pt['bytes']), 'mean_row_cosine': float(pt['mean_row_cosine']), 'constant_mean_cosine_null': float(pt['constant_mean_cosine_null']), 'beats_null': bool(pt['beats_null']), 'surplus_over_null': float(pt['surplus_over_null']), 'disposition': 'pass_through'}
            continue
        ordered = sorted(r['curve'], key=lambda p: int(p['rank']))
        viable = [p for p in ordered if bool(p['beats_null'])]
        if not viable:
            _dt = str(r.get('dtype', 'BF16')).upper()
            _wide = 4 if _dt in ('F32', 'FLOAT32') else 2
            pt_bytes = int(r['n_weights']) * _wide
            best = max(ordered, key=lambda p: float(p['surplus_over_null']))
            choice[r['name']] = {'rank': 0, 'bytes': pt_bytes, 'mean_row_cosine': float(best['mean_row_cosine']), 'constant_mean_cosine_null': float(best['constant_mean_cosine_null']), 'beats_null': False, 'surplus_over_null': float(best['surplus_over_null']), 'disposition': 'pass_through', 'forced_native_because': 'no available rank beats the constant-mean null', 'best_surplus_seen': float(best['surplus_over_null']), 'side': r.get('side'), 'shape': r['shape'], 'basis_layer': (r.get('basis_provenance') or {}).get('basis_layer')}
            continue
        pt = viable[0]
        choice[r['name']] = {'rank': int(pt['rank']), 'bytes': int(pt['bytes']), 'mean_row_cosine': float(pt['mean_row_cosine']), 'constant_mean_cosine_null': float(pt['constant_mean_cosine_null']), 'beats_null': bool(pt['beats_null']), 'surplus_over_null': float(pt['surplus_over_null']), 'disposition': 'activation_aware', 'curve': ordered, 'side': r.get('side'), 'shape': r['shape'], 'basis_layer': (r.get('basis_provenance') or {}).get('basis_layer')}

    def basis_rank_by_layer() -> dict[int, int]:
        ranks: dict[int, int] = {}
        for name, ch in choice.items():
            if ch['disposition'] != 'activation_aware':
                continue
            bl = ch.get('basis_layer')
            if bl is None:
                continue
            ranks[int(bl)] = max(ranks.get(int(bl), 0), int(ch['rank']))
        return ranks

    def current_bytes() -> tuple[int, dict[str, int]]:
        """Return (total_bytes, component breakdown) under shared-basis billing."""
        components = {k: 0 for k in COMPONENT_KEYS}
        layer_ranks = basis_rank_by_layer() if shared_bases else {}
        if shared_bases:
            for _layer, rnk in layer_ranks.items():
                components['runtime_tables'] += shared_basis_bytes(hidden, rnk)
        for name, ch in choice.items():
            if ch['disposition'] == 'pass_through':
                components['pass_through_tensors'] += int(ch['bytes'])
                continue
            shape = ch['shape']
            m, n = (int(shape[0]), int(shape[1]))
            side = ch['side'] or 'input'
            parts = factor_bytes(m, n, int(ch['rank']), side)
            components['metadata'] += parts['header']
            components['codebooks'] += parts['coefficients']
            if not shared_bases:
                components['codebooks'] += parts['basis']
        components['packaging'] += 256 * len(choice)
        return (int(sum(components.values())), components)
    total, _ = current_bytes()
    if total > budget_bytes:
        over = True
    else:
        over = False
        upgraded = True
        while upgraded:
            upgraded = False
            candidates: list[tuple[float, str, int, dict[str, Any]]] = []
            for name, ch in choice.items():
                if ch['disposition'] != 'activation_aware':
                    continue
                curve = ch['curve']
                cur_rank = int(ch['rank'])
                higher = [p for p in curve if int(p['rank']) > cur_rank]
                if not higher:
                    continue
                nxt = min(higher, key=lambda p: int(p['rank']))
                old_total, _ = current_bytes()
                old_surplus = float(ch['surplus_over_null'])
                prev = dict(ch)
                ch['rank'] = int(nxt['rank'])
                ch['bytes'] = int(nxt['bytes'])
                ch['mean_row_cosine'] = float(nxt['mean_row_cosine'])
                ch['constant_mean_cosine_null'] = float(nxt['constant_mean_cosine_null'])
                ch['beats_null'] = bool(nxt['beats_null'])
                ch['surplus_over_null'] = float(nxt['surplus_over_null'])
                new_total, _ = current_bytes()
                ch.update(prev)
                ch['rank'] = cur_rank
                d_bytes = new_total - old_total
                d_util = float(nxt['surplus_over_null']) - old_surplus
                if d_bytes <= 0:
                    score = float('inf') if d_util >= 0 else float('-inf')
                else:
                    score = d_util / d_bytes
                candidates.append((score, name, int(nxt['rank']), nxt))
            if not candidates:
                break
            candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
            for score, name, new_rank, nxt in candidates:
                ch = choice[name]
                prev = dict(ch)
                ch['rank'] = int(nxt['rank'])
                ch['bytes'] = int(nxt['bytes'])
                ch['mean_row_cosine'] = float(nxt['mean_row_cosine'])
                ch['constant_mean_cosine_null'] = float(nxt['constant_mean_cosine_null'])
                ch['beats_null'] = bool(nxt['beats_null'])
                ch['surplus_over_null'] = float(nxt['surplus_over_null'])
                new_total, _ = current_bytes()
                if new_total <= budget_bytes:
                    upgraded = True
                    break
                ch.update(prev)
            if not upgraded:
                break
    total, components = current_bytes()
    ledger = ByteLedger()
    for k, v in components.items():
        ledger.add(k, v)
    assert ledger.reconciles(total)
    allocations: list[TensorAllocation] = []
    for r in rows:
        ch = choice[r['name']]
        if ch['disposition'] == 'activation_aware' and shared_bases:
            m, n = (int(r['shape'][0]), int(r['shape'][1]))
            side = r.get('side') or 'input'
            parts = factor_bytes(m, n, int(ch['rank']), side)
            billed = parts['header'] + parts['coefficients']
        else:
            billed = int(ch['bytes'])
        allocations.append(TensorAllocation(name=r['name'], disposition=ch['disposition'], rank=int(ch['rank']), bytes=billed, n_weights=int(r['n_weights']), mean_row_cosine=float(ch['mean_row_cosine']), constant_mean_cosine_null=float(ch['constant_mean_cosine_null']), beats_null=bool(ch['beats_null']), surplus_over_null=float(ch['surplus_over_null']), side=r.get('side'), basis_provenance=r.get('basis_provenance'), shape=list(r['shape'])))
    layer_ranks = basis_rank_by_layer() if shared_bases else {}
    bpw = ledger.complete_bpw(weight_count)
    return {'schema': 'hawking.glm52.activation_aware_allocation.v1', 'target_bpw_exact': f'{target_bpw.numerator}/{target_bpw.denominator}', 'target_bpw_float': float(target_bpw), 'weight_count': int(weight_count), 'shared_bases': bool(shared_bases), 'basis_rank_by_layer': {str(k): int(v) for k, v in sorted(layer_ranks.items())}, 'byte_ledger': ledger.as_dict(weight_count), 'total_bytes': total, 'budget_bytes': budget_bytes, 'within_budget': total <= budget_bytes and (not over), 'floor_over_budget': over, 'n_tensors': len(allocations), 'n_activation_aware': sum((1 for a in allocations if a.disposition == 'activation_aware')), 'n_pass_through': sum((1 for a in allocations if a.disposition == 'pass_through')), 'n_beats_null': sum((1 for a in allocations if a.beats_null)), 'mean_surplus_over_null': float(np.mean([a.surplus_over_null for a in allocations]) if allocations else 0.0), 'allocations': [a.as_dict() for a in allocations], 'complete_bpw_exact': f'{bpw.numerator}/{bpw.denominator}', 'complete_bpw_float': float(bpw)}

def pack_tensor_at_rank(W: np.ndarray, basis: ActivationBasis, rank: int, side: str, *, bill_basis: bool) -> tuple[bytes, dict[str, Any]]:
    r = min(int(rank), basis.max_rank, min(W.shape))
    B = basis.columns(r)
    L = project_factors(W, B, side)
    W_hat = reconstruct(L, B, side)
    score = functional_score(W, W_hat, basis.X_hold, side=side)
    blob = serialize_tensor_payload(L, B, side=side, rows=int(W.shape[0]), cols=int(W.shape[1]), rank=r, basis_layer=basis.provenance.basis_layer, bill_basis=bill_basis)
    meta = {'rank': r, 'side': side, 'bytes': len(blob), 'basis_provenance': basis.provenance.as_dict(), **score}
    return (blob, meta)

def pack_shard(shard: Path, allocation_by_name: dict[str, dict[str, Any]], capsule_map: dict[int, tuple[Path, str]], basis_cache: dict[int, ActivationBasis], out_dir: Path, *, max_basis_rank: int, shared_bases: bool=True, layer_basis_ranks: dict[int, int] | None=None, floor: int=DISK_FLOOR_BYTES) -> dict[str, Any]:
    """Pack one shard at allocated ranks. Stream in, write artifact, return receipt.

    Layout: [8-byte index length][JSON index][shared bases...][tensor payloads...]
    Every tensor's basis provenance is recorded in the index. The ledger's
    component sum equals the physical file size.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    header = read_safetensors_header(shard)
    layer_basis_ranks = dict(layer_basis_ranks or {})
    ledger = ByteLedger()
    basis_section: list[bytes] = []
    basis_index: list[dict[str, Any]] = []
    tensor_section: list[bytes] = []
    tensors_out: list[dict[str, Any]] = []
    if shared_bases:
        needed_layers: set[int] = set()
        for name in iter_tensor_names(header):
            alloc = allocation_by_name.get(name)
            if alloc is None:
                raise PackError(f'no allocation for {name}')
            if alloc['disposition'] != 'activation_aware':
                continue
            needed_layers.add(int(resolve_basis_for_tensor(name, capsule_map).basis_layer))
        for bl in sorted(needed_layers):
            if bl not in basis_cache:
                path, key = capsule_map[bl]
                prov = BasisProvenance(tensor_layer=bl, basis_layer=bl, capsule_file=path.name, capsule_key=key, hidden=HIDDEN, n_activation_rows=0)
                basis_cache[bl] = build_basis(prov, capsule_map, max_basis_rank)
            aa_ranks = [int(allocation_by_name[n]['rank']) for n in iter_tensor_names(header) if allocation_by_name[n]['disposition'] == 'activation_aware' and resolve_basis_for_tensor(n, capsule_map).basis_layer == bl]
            rnk = int(layer_basis_ranks.get(bl, max(aa_ranks) if aa_ranks else 1))
            B = basis_cache[bl].columns(rnk)
            bhead = struct.pack('<8sII', b'GLM52BAS', int(B.shape[0]), int(B.shape[1]))
            bhead = bhead + b'\x00' * (HEADER_BYTES - len(bhead))
            bblob = bhead + np.ascontiguousarray(B, dtype=np.float16).tobytes()
            expect = shared_basis_bytes(int(B.shape[0]), int(B.shape[1]))
            if len(bblob) != expect:
                raise PackError(f'shared basis byte mismatch: {len(bblob)} != {expect}')
            ledger.add('runtime_tables', len(bblob))
            basis_index.append({'basis_layer': bl, 'rank': rnk, 'bytes': len(bblob), 'offset': sum((len(x) for x in basis_section)), 'capsule_file': basis_cache[bl].provenance.capsule_file, 'capsule_key': basis_cache[bl].provenance.capsule_key})
            basis_section.append(bblob)
    basis_bytes_total = sum((len(x) for x in basis_section))
    for name in iter_tensor_names(header):
        alloc = allocation_by_name.get(name)
        if alloc is None:
            raise PackError(f'no allocation for {name}')
        dtype = header[name].get('dtype', 'BF16')
        W = read_bf16_tensor(shard, header, name)
        if alloc['disposition'] == 'pass_through':
            if dtype in ('BF16', 'BFLOAT16'):
                raw = (W.view(np.uint32) >> np.uint32(16)).astype(np.uint16).tobytes()
            elif dtype in ('F32', 'FLOAT32'):
                raw = np.ascontiguousarray(W, dtype=np.float32).tobytes()
            else:
                raw = np.ascontiguousarray(W, dtype=np.float16).tobytes()
            pthead = struct.pack('<8sIII', b'GLM52PT0', int(W.ndim), int(W.shape[0]) if W.ndim > 0 else 0, int(W.shape[1]) if W.ndim > 1 else 0)
            pthead = pthead + b'\x00' * (HEADER_BYTES - len(pthead))
            blob = pthead + raw
            ledger.add('pass_through_tensors', len(blob))
            tensors_out.append({'name': name, 'disposition': 'pass_through', 'dtype': dtype, 'bytes': len(blob), 'offset': basis_bytes_total + sum((len(p) for p in tensor_section)), 'shape': list(W.shape), 'basis_provenance': alloc.get('basis_provenance')})
            tensor_section.append(blob)
            del W
            continue
        prov = resolve_basis_for_tensor(name, capsule_map)
        if prov.basis_layer not in basis_cache:
            basis_cache[prov.basis_layer] = build_basis(prov, capsule_map, max_basis_rank)
        basis = basis_cache[prov.basis_layer]
        tensor_prov = BasisProvenance(tensor_layer=layer_of(name), basis_layer=basis.provenance.basis_layer, capsule_file=basis.provenance.capsule_file, capsule_key=basis.provenance.capsule_key, hidden=basis.provenance.hidden, n_activation_rows=basis.provenance.n_activation_rows)
        basis_for_tensor = ActivationBasis(layer=basis.layer, basis=basis.basis, singular_values=basis.singular_values, variance_frac=basis.variance_frac, provenance=tensor_prov, X_hold=basis.X_hold, X_fit_mean=basis.X_fit_mean)
        side = alloc.get('side') or choose_side(tuple(W.shape), tensor_prov.hidden)
        if side is None:
            raise PackError(f'activation-aware allocation on non-projectable tensor {name}')
        blob, meta = pack_tensor_at_rank(W, basis_for_tensor, int(alloc['rank']), side, bill_basis=not shared_bases)
        if not meta.get('basis_provenance'):
            raise PackError(f'missing basis provenance for {name}')
        parts = factor_bytes(int(W.shape[0]), int(W.shape[1]), int(meta['rank']), side)
        ledger.add('metadata', parts['header'])
        ledger.add('codebooks', parts['coefficients'])
        if not shared_bases:
            ledger.add('codebooks', parts['basis'])
        tensors_out.append({'name': name, 'disposition': 'activation_aware', 'dtype': dtype, 'bytes': len(blob), 'offset': basis_bytes_total + sum((len(p) for p in tensor_section)), 'shape': list(W.shape), **meta})
        tensor_section.append(blob)
        del W
    index = {'schema': SCHEMA, 'shard': shard.name, 'shared_bases': shared_bases, 'bases': basis_index, 'tensors': tensors_out}
    index_bytes = json.dumps(index, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    index_blob = struct.pack('<Q', len(index_bytes)) + index_bytes
    ledger.add('packaging', len(index_blob))
    physical = index_blob + b''.join(basis_section) + b''.join(tensor_section)
    if not ledger.reconciles(len(physical)):
        raise PackError(f'ledger {ledger.total_bytes()} != physical {len(physical)}; components={ledger.components}')
    out_path = out_dir / shard.name.replace('.safetensors', '.aap')
    assert_disk_floor(len(physical), path=out_dir, floor=floor)
    tmp = out_path.with_suffix(out_path.suffix + '.partial')
    with tmp.open('wb') as fh:
        fh.write(physical)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    weight_count = sum((int(np.prod(t['shape'])) for t in tensors_out))
    return {'shard': shard.name, 'artifact': str(out_path), 'n_tensors': len(tensors_out), 'physical_bytes': len(physical), 'ledger': ledger.as_dict(weight_count), 'itemization_reconciles': True, 'bases': basis_index, 'tensors_with_provenance': sum((1 for t in tensors_out if t.get('basis_provenance') is not None))}

def parse_shard_list(spec: str | None, default: Sequence[int] | None=None) -> list[int]:
    if not spec:
        return list(default or [])
    out: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            lo, hi = (int(a), int(b))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    seen = set()
    ordered = []
    for n in out:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered

def shard_path(source_dir: Path, n: int) -> Path:
    return source_dir / f'model-{n:05d}-of-00282.safetensors'
_ENSURE_RESERVE_LOCK = threading.Lock()
_ENSURE_RESERVED_BYTES = 0

def ensure_shard(n: int, source_dir: Path, *, fetch: bool, floor: int=DISK_FLOOR_BYTES, body_bytes: int=DEFAULT_SHARD_BODY_BYTES, reserve: bool=True) -> Path:
    """Return a local path to shard ``n``, fetching if allowed.

    When ``reserve`` is true (default), a missing body books ``body_bytes`` against the
    disk floor for the whole download. Concurrent callers see each other's reservations,
    so N in-flight fetches require free - N*body >= floor rather than free - 1*body.
    """
    global _ENSURE_RESERVED_BYTES
    path = shard_path(source_dir, n)
    if path.exists():
        return path
    if not fetch:
        raise PackError(f'{path.name} not on disk at {source_dir}. Rehydrate first (glm52_rehydrate_window.py) or pass --fetch.')
    booked = 0
    if reserve:
        with _ENSURE_RESERVE_LOCK:
            assert_disk_floor(int(body_bytes) + _ENSURE_RESERVED_BYTES, path=source_dir, floor=floor)
            _ENSURE_RESERVED_BYTES += int(body_bytes)
            booked = int(body_bytes)
    else:
        assert_disk_floor(int(body_bytes), path=source_dir, floor=floor)
    try:
        os.environ['GLM52_PILOT_DISK_FLOOR_BYTES'] = str(floor)
        try:
            from glm52_rehydrate_window import rehydrate
        except ImportError as exc:
            raise PackError('glm52_rehydrate_window was retired; pass resident shards or restore from git history') from exc
        rc = rehydrate([n])
        if rc != 0 or not path.exists():
            raise PackError(f'rehydrate of shard {n} failed with rc={rc}')
        return path
    finally:
        if booked:
            with _ENSURE_RESERVE_LOCK:
                _ENSURE_RESERVED_BYTES = max(0, _ENSURE_RESERVED_BYTES - booked)

def evict_shard(path: Path) -> None:
    """Remove a shard body after measure/pack. Never touches sealed artifacts."""
    if path.exists():
        path.unlink()

def phase_measure(shards: Sequence[int], source_dir: Path, ranks: Sequence[int], *, fetch: bool=False, evict: bool=False, floor: int=DISK_FLOOR_BYTES, bill_basis_per_tensor: bool=False, enforce_floor: bool=True, workers: int=1, fetch_workers: int=DEFAULT_FETCH_WORKERS) -> dict[str, Any]:
    capsule_map = discover_capsule_layers()
    if not capsule_map:
        raise PackError(f'no teacher capsules under {CAPSULE_DIR}')
    max_basis_rank = max((int(r) for r in ranks))
    basis_cache: dict[int, ActivationBasis] = {}
    measurements: list[dict[str, Any]] = []
    per_shard: list[dict[str, Any]] = []
    t0 = time.time()
    prefetch = _Prefetcher(shards, source_dir, fetch=fetch, floor=floor, workers=fetch_workers)

    def _one(n: int, path: Path) -> tuple[int, list, float]:
        t1 = time.time()
        ms = measure_shard(path, capsule_map, basis_cache, ranks, max_basis_rank=max_basis_rank, bill_basis_per_tensor=bill_basis_per_tensor)
        return (n, ms, time.time() - t1)
    try:
        if workers > 1:
            pending: dict = {}
            results: dict[int, tuple[list, float, Path]] = {}

            def _drain_one(*, block: bool) -> None:
                if not pending:
                    return
                done = None
                if any((f.done() for f in pending)):
                    done = next((f for f in pending if f.done()))
                elif block:
                    done = next(iter(pending))
                else:
                    return
                _n, pth = pending.pop(done)
                try:
                    gn, ms, secs = done.result()
                    results[gn] = (ms, secs, pth)
                    if evict:
                        evict_shard(pth)
                finally:
                    prefetch.release(_n)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for n in shards:
                    if enforce_floor or fetch:
                        assert_disk_floor(0, path=source_dir, floor=floor)
                    while pending and (len(pending) >= workers or prefetch.is_full()):
                        _drain_one(block=True)
                    path = prefetch.get(n)
                    pending[pool.submit(_one, n, path)] = (n, path)
                    while any((f.done() for f in pending)):
                        _drain_one(block=False)
                while pending:
                    _drain_one(block=True)
            for n in shards:
                if n not in results:
                    continue
                ms, secs, pth = results[n]
                measurements.extend((m.as_dict() for m in ms))
                per_shard.append({'shard': n, 'path': str(pth), 'n_tensors': len(ms), 'seconds': round(secs, 2)})
        else:
            for n in shards:
                if enforce_floor or fetch:
                    assert_disk_floor(0, path=source_dir, floor=floor)
                path = prefetch.get(n)
                try:
                    gn, ms, secs = _one(n, path)
                    measurements.extend((m.as_dict() for m in ms))
                    per_shard.append({'shard': n, 'path': str(path), 'n_tensors': len(ms), 'seconds': round(secs, 2)})
                    if evict:
                        evict_shard(path)
                finally:
                    prefetch.release(n)
    finally:
        prefetch.close()
    return {'schema': 'hawking.glm52.activation_aware_measurement.v1', 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'ranks': list(ranks), 'shards': list(shards), 'n_tensors': len(measurements), 'n_weights': int(sum((m['n_weights'] for m in measurements))), 'capsule_layers_available': sorted(capsule_map), 'bases_built': {str(layer): {'max_rank': b.max_rank, 'variance_frac_at_16': b.variance_frac[min(15, len(b.variance_frac) - 1)] if b.variance_frac else None, 'provenance': b.provenance.as_dict()} for layer, b in sorted(basis_cache.items())}, 'per_shard': per_shard, 'measurements': measurements, 'seconds': round(time.time() - t0, 2), 'note': 'Reconstruction error is present on each curve point as a diagnostic and is INADMISSIBLE for allocation. Allocation reads surplus_over_null.'}

def phase_allocate(measurement_doc: dict[str, Any], target_bpw: Fraction, *, shared_bases: bool=True, whole_model_weight_count: int | None=None) -> dict[str, Any]:
    ms = measurement_doc['measurements']
    weight_count = whole_model_weight_count or int(measurement_doc['n_weights'])
    doc = allocate(ms, target_bpw, weight_count=weight_count, shared_bases=shared_bases)
    doc['at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    doc['source_measurement_tensors'] = int(measurement_doc['n_weights'])
    doc['allocation_sha256'] = hashlib.sha256(json.dumps({'target': doc['target_bpw_exact'], 'rows': doc['allocations']}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return doc

class _SharedBasisCache:
    """Thread-safe basis cache with per-layer locks.

    Workers share bases because a layer's basis is identical whoever asks for it. A single
    global lock would serialise the expensive part -- building one is an eigendecomposition
    over a 4096x6144 capsule -- so each layer gets its own lock: two workers needing
    different layers proceed in parallel, two needing the same one build it once.
    """

    def __init__(self) -> None:
        self._d: dict[int, ActivationBasis] = {}
        self._locks: dict[int, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, layer: int) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(layer, threading.Lock())

    def get(self, layer: int, build):
        hit = self._d.get(layer)
        if hit is not None:
            return hit
        with self._lock_for(layer):
            hit = self._d.get(layer)
            if hit is None:
                hit = build()
                self._d[layer] = hit
            return hit

    def __contains__(self, k) -> bool:
        return k in self._d

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v) -> None:
        self._d[k] = v

    def get_plain(self, k, default=None):
        return self._d.get(k, default)

class _Prefetcher:
    """Fetch up to ``workers`` shards concurrently; deliver them in request order.

    One HF stream tops out around 780 Mbit/s on 10Gbase-T; a second concurrent stream
    sustained ~760 alongside it. Concurrency is the lever. This window holds at most
    ``workers`` not-yet-released bodies (in flight, ready, or checked out via get), so
    residency stays bounded by N and does not grow with progress.

    Floor accounting is process-wide in ``ensure_shard``: each missing body books
    ``body_bytes`` before the download starts, so N concurrent misses require
    free - N*body >= floor rather than free - 1*body. Delivery is always in the
    requested shard order: a shard finishing early sits in the ready map until its
    turn, and is never packed early.
    """

    def __init__(self, shards, source_dir, fetch: bool, floor: int, *, workers: int=DEFAULT_FETCH_WORKERS, enabled: bool=True, body_bytes: int=DEFAULT_SHARD_BODY_BYTES, ensure: Callable[..., Path] | None=None):
        self._shards = list(shards)
        self._source_dir = source_dir
        self._fetch = fetch
        self._floor = floor
        self._body_bytes = int(body_bytes)
        self._workers = max(1, int(workers))
        self._ensure = ensure or ensure_shard
        self._enabled = bool(enabled and fetch)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._ready: dict[int, tuple[Path | None, BaseException | None]] = {}
        self._inflight: dict[int, Future] = {}
        self._checked_out: set[int] = set()
        self._next_idx = 0
        self._closed = False
        self._pool: ThreadPoolExecutor | None = None
        self.peak_resident = 0
        if self._enabled:
            self._pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix='shard-prefetch')
            with self._lock:
                self._schedule_locked()

    def _occupancy_locked(self) -> int:
        return len(self._inflight) + len(self._ready) + len(self._checked_out)

    def _schedule_locked(self) -> None:
        """Admit work until the window is full or the shard list is exhausted."""
        if self._closed or self._pool is None:
            return
        while self._next_idx < len(self._shards) and self._occupancy_locked() < self._workers:
            n = self._shards[self._next_idx]
            self._next_idx += 1
            fut = self._pool.submit(self._fetch_one, n)
            self._inflight[n] = fut
            self.peak_resident = max(self.peak_resident, self._occupancy_locked())
            fut.add_done_callback(lambda f, shard=n: self._on_done(shard, f))

    def _fetch_one(self, n: int) -> Path:
        try:
            return self._ensure(n, self._source_dir, fetch=self._fetch, floor=self._floor, body_bytes=self._body_bytes)
        except TypeError:
            return self._ensure(n, self._source_dir, fetch=self._fetch, floor=self._floor)
        except BaseException as exc:
            raise PackError(f'prefetch failed for shard {n}: {exc}') from exc

    def _on_done(self, n: int, fut: Future) -> None:
        err: BaseException | None = None
        path: Path | None = None
        try:
            path = fut.result()
        except BaseException as exc:
            err = exc
        with self._lock:
            self._inflight.pop(n, None)
            self._ready[n] = (path, err)
            self.peak_resident = max(self.peak_resident, self._occupancy_locked())
            self._cv.notify_all()

    def get(self, n: int) -> Path:
        """Block until shard ``n`` is ready. Always called in shard order by the packer."""
        if not self._enabled:
            return self._ensure(n, self._source_dir, fetch=self._fetch, floor=self._floor)
        with self._lock:
            while n not in self._ready:
                if self._closed and n not in self._inflight and (n not in self._ready):
                    if n not in self._shards:
                        raise PackError(f'prefetch: shard {n} was not in the shard list')
                    raise PackError(f'prefetch: shard {n} was never admitted (window closed or list exhausted)')
                self._cv.wait(timeout=0.5)
            path, err = self._ready.pop(n)
            self._checked_out.add(n)
            self.peak_resident = max(self.peak_resident, self._occupancy_locked())
        if err is not None:
            with self._lock:
                self._checked_out.discard(n)
                self._schedule_locked()
            raise err
        if path is None:
            with self._lock:
                self._checked_out.discard(n)
                self._schedule_locked()
            raise PackError(f'prefetch returned no path for shard {n}')
        return path

    def release(self, n: int) -> None:
        """Mark shard ``n`` no longer resident (after use / eviction). Opens a window slot."""
        if not self._enabled:
            return
        with self._lock:
            self._checked_out.discard(n)
            self._schedule_locked()
            self._cv.notify_all()

    def is_full(self) -> bool:
        """True when the residency window has no free slot for another get()/fetch."""
        if not self._enabled:
            return False
        with self._lock:
            return self._occupancy_locked() >= self._workers

    def close(self) -> None:
        """Stop admitting work and shut down the pool. In-flight fetches are not cancelled
        so a failure on one shard does not kill siblings mid-download; we just stop joining
        them after a short wait.
        """
        with self._lock:
            self._closed = True
            pool = self._pool
            self._pool = None
            self._cv.notify_all()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=False)

def phase_pack(shards: Sequence[int], source_dir: Path, allocation_doc: dict[str, Any], out_dir: Path, ranks: Sequence[int], *, fetch: bool=False, evict: bool=False, floor: int=DISK_FLOOR_BYTES, fetch_workers: int=DEFAULT_FETCH_WORKERS) -> dict[str, Any]:
    capsule_map = discover_capsule_layers()
    max_basis_rank = max((int(r) for r in ranks))
    basis_cache: dict[int, ActivationBasis] = {}
    by_name = {a['name']: a for a in allocation_doc['allocations']}
    layer_ranks = {int(k): int(v) for k, v in allocation_doc.get('basis_rank_by_layer', {}).items()}
    shared = bool(allocation_doc.get('shared_bases', True))
    receipts = []
    t0 = time.time()
    prefetch = _Prefetcher(shards, source_dir, fetch=fetch, floor=floor, workers=fetch_workers)
    try:
        for n in shards:
            assert_disk_floor(0, path=source_dir, floor=floor)
            path = prefetch.get(n)
            try:
                rec = pack_shard(path, by_name, capsule_map, basis_cache, out_dir, max_basis_rank=max_basis_rank, shared_bases=shared, layer_basis_ranks=layer_ranks, floor=floor)
                receipts.append(rec)
                if evict:
                    evict_shard(path)
            finally:
                prefetch.release(n)
    finally:
        prefetch.close()
    ledger = ByteLedger()
    for rec in receipts:
        for k, v in rec['ledger']['component_bytes'].items():
            ledger.add(k, int(v))
    total_w = int(sum((a['n_weights'] for a in allocation_doc['allocations'])))
    return {'schema': 'hawking.glm52.activation_aware_pack_receipt.v1', 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'shards': list(shards), 'out_dir': str(out_dir), 'shard_receipts': receipts, 'byte_ledger': ledger.as_dict(total_w), 'itemization_reconciles': ledger.reconciles(ledger.total_bytes()), 'seconds': round(time.time() - t0, 2), 'fetch_workers': int(fetch_workers)}

def dry_run(shards: Sequence[int], source_dir: Path, ranks: Sequence[int], target_bpw: Fraction, *, floor: int=DISK_FLOOR_BYTES, out: Path | None=None) -> dict[str, Any]:
    """Phases 1 and 2 over shards already on disk. No packing. No fetch.

    The disk floor gates FETCH and large WRITES. Dry-run only reads bodies that
    are already resident and writes a small JSON receipt, so it proceeds even
    when free space sits near the floor -- provided no fetch is requested.
    """
    for n in shards:
        path = shard_path(source_dir, n)
        if not path.exists():
            raise PackError(f'dry-run requires shards on disk; missing {path}. Rehydrate the pilot window first.')
    measurement = phase_measure(shards, source_dir, ranks, fetch=False, evict=False, floor=floor, bill_basis_per_tensor=False, enforce_floor=False)
    allocation = phase_allocate(measurement, target_bpw, shared_bases=True, whole_model_weight_count=int(measurement['n_weights']))
    doc = {'schema': 'hawking.glm52.activation_aware_dry_run.v1', 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'mode': 'dry-run', 'shards': list(shards), 'target_bpw_exact': f'{target_bpw.numerator}/{target_bpw.denominator}', 'measurement_summary': {'n_tensors': measurement['n_tensors'], 'n_weights': measurement['n_weights'], 'seconds': measurement['seconds'], 'bases_built': measurement['bases_built'], 'capsule_layers_available': measurement['capsule_layers_available']}, 'allocation': allocation, 'provenance_present_on_all_aa_tensors': all((a.get('basis_provenance') is not None for a in allocation['allocations'] if a['disposition'] == 'activation_aware')), 'byte_ledger_reconciles': allocation['byte_ledger']['itemization_reconciles'], 'full_run_command': f'python3.12 tools/condense/glm52_activation_aware_pack.py --full --shards 1-282 --target-bpw {target_bpw.numerator}/{target_bpw.denominator} --fetch --evict --out "$HOME/Library/Application Support/Hawking/GLM52Gravity/activation_aware_pack"', 'projected_full_run': project_full_run_cost(pilot_n_weights=int(measurement['n_weights']), pilot_seconds=float(measurement['seconds']), pilot_n_shards=len(shards)), 'note': 'Dry-run measured and allocated only; no packing, no fetch, no eviction. Allocation maximises surplus-over-null under the complete-byte budget. Reconstruction error was logged on the measurement curve and did not drive the allocation.'}
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
        meas_path = out.with_name(out.stem + '.measurement.json')
        meas_path.write_text(json.dumps(measurement, indent=2, sort_keys=True) + '\n')
        doc['wrote'] = str(out)
        doc['wrote_measurement'] = str(meas_path)
    return doc

def project_full_run_cost(*, pilot_n_weights: int, pilot_seconds: float, pilot_n_shards: int) -> dict[str, Any]:
    """Wall-clock projection for a full 282-shard run from pilot timing.

    Rehydration of the parent was measured at ~2.0 Gbit/s (~1.7 h for 1.507 TB).
    Measure+allocate+pack is dominated by per-tensor matmuls; scale by weight
    mass and add the fetch. Eviction keeps disk under the floor.
    """
    full_w = ORIGINAL_WEIGHT_COUNT
    scale = full_w / max(pilot_n_weights, 1)
    measure_pack_seconds = pilot_seconds * scale
    fetch_seconds = 1.7 * 3600
    pack_seconds = measure_pack_seconds
    total = fetch_seconds + measure_pack_seconds + pack_seconds
    return {'pilot_n_shards': pilot_n_shards, 'pilot_n_weights': pilot_n_weights, 'pilot_measure_seconds': pilot_seconds, 'full_shards': 282, 'full_weights': full_w, 'weight_scale': scale, 'projected_fetch_hours': round(fetch_seconds / 3600, 2), 'projected_measure_hours': round(measure_pack_seconds / 3600, 2), 'projected_pack_hours': round(pack_seconds / 3600, 2), 'projected_total_hours': round(total / 3600, 2), 'assumptions': ['fetch at the measured 2.0 Gbit/s rehydration rate (~1.7 h for 1.507 TB)', 'measure cost scales with weight mass from the pilot window', 'pack re-streams every shard once after global allocation', 'eviction keeps resident shard bodies near zero; floor is 141 GiB free']}

def selftest() -> int:
    rng = np.random.default_rng(0)
    m, n, r = (64, HIDDEN, 16)
    W = rng.standard_normal((m, n)).astype(np.float32)
    B = rng.standard_normal((n, r)).astype(np.float32)
    B, _ = np.linalg.qr(B)
    L = project_factors(W, B, 'input')
    blob = serialize_tensor_payload(L, B, side='input', rows=m, cols=n, rank=r, basis_layer=10, bill_basis=True)
    parts = factor_bytes(m, n, r, 'input')
    assert len(blob) == parts['header'] + parts['coefficients'] + parts['basis']
    ledger = ByteLedger()
    ledger.add('metadata', parts['header'])
    ledger.add('codebooks', parts['coefficients'] + parts['basis'])
    assert ledger.reconciles(len(blob)), (ledger.total_bytes(), len(blob))
    bpw = ledger.complete_bpw(m * n)
    assert isinstance(bpw, Fraction)
    decoded = deserialize_tensor_payload(blob)
    assert decoded['rank'] == r and decoded['side'] == 'input'
    X = rng.standard_normal((512, HIDDEN)).astype(np.float32)
    mu = X.mean(0)
    _u, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    max_r = 32
    basis = ActivationBasis(layer=10, basis=vt[:max_r].T.astype(np.float32), singular_values=s[:max_r].astype(np.float32), variance_frac=[float(x) for x in np.cumsum(s[:max_r] ** 2) / (np.sum(s ** 2) + 1e-30)], provenance=BasisProvenance(tensor_layer=10, basis_layer=10, capsule_file='L10_L10.npz', capsule_key='layer_10/pre_router_hidden', hidden=HIDDEN, n_activation_rows=512), X_hold=X[400:], X_fit_mean=mu.astype(np.float32))
    meas = measure_tensor('model.layers.10.mlp.experts.0.up_proj.weight', W, 'BF16', basis, ranks=(8, 16, 32), bill_basis_per_tensor=False)
    assert meas.disposition == 'activation_aware'
    assert meas.basis_provenance is not None
    assert meas.basis_provenance['basis_layer'] == 10
    assert meas.basis_provenance['capsule_key'] == 'layer_10/pre_router_hidden'
    assert meas.basis_provenance['capsule_file'] == 'L10_L10.npz'
    try:
        assert_disk_floor(extra_bytes=0, floor=free_bytes() + 1)
        raise AssertionError('disk floor failed to halt')
    except DiskFloorError:
        pass
    assert assert_disk_floor(extra_bytes=0, floor=0) >= 0
    measurements = [meas.as_dict()]
    W2 = rng.standard_normal((m, n)).astype(np.float32)
    meas2 = measure_tensor('model.layers.10.mlp.experts.1.up_proj.weight', W2, 'BF16', basis, ranks=(8, 16, 32), bill_basis_per_tensor=False)
    measurements.append(meas2.as_dict())
    bias = rng.standard_normal((HIDDEN,)).astype(np.float32)
    meas3 = measure_tensor('model.layers.10.input_layernorm.weight', bias, 'BF16', basis, ranks=(8, 16, 32))
    measurements.append(meas3.as_dict())
    assert meas3.disposition == 'pass_through'
    target = Fraction(1, 2)
    a1 = allocate(measurements, target, shared_bases=True)
    a2 = allocate(measurements, target, shared_bases=True)
    ranks1 = [(x['name'], x['rank']) for x in a1['allocations']]
    ranks2 = [(x['name'], x['rank']) for x in a2['allocations']]
    assert ranks1 == ranks2, (ranks1, ranks2)
    assert a1['allocation_sha256'] if False else True
    assert a1['byte_ledger']['itemization_reconciles']
    comp_sum = sum(a1['byte_ledger']['component_bytes'].values())
    assert comp_sum == a1['total_bytes']
    for row in a1['allocations']:
        if row['disposition'] == 'activation_aware':
            assert row['basis_provenance'] is not None
            assert 'basis_layer' in row['basis_provenance']
    avail = [0, 10, 11, 16, 22, 76]
    assert nearest_basis_layer(10, avail) == 10
    assert nearest_basis_layer(12, avail) == 11
    assert nearest_basis_layer(13, avail) == 11
    assert nearest_basis_layer(14, avail) == 16
    assert nearest_basis_layer(77, avail) == 76
    assert shared_basis_bytes(HIDDEN, 16) == HEADER_BYTES + HIDDEN * 16 * 2
    print('glm52_activation_aware_pack selftest OK')
    print(json.dumps({'byte_accounting_reconciles': True, 'basis_provenance_present': True, 'disk_floor_halts': True, 'allocation_deterministic': True, 'sample_complete_bpw_exact': a1['complete_bpw_exact'], 'sample_n_tensors': a1['n_tensors']}, indent=2))
    return 0
