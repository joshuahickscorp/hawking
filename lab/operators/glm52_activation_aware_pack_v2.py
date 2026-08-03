"""Recomposed science module glm52_activation_aware_pack_v2 (C-SCI-R1)."""
from __future__ import annotations
from lab.operators import glm52_activation_aware_pack as aap
from lab.layout import REPORTS_ROOT, evidence_dir
import sys as _sys_a1
from pathlib import Path as _Path_a1
import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == 'archive' else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
HERE = _A1_CONDENSE
REPO = HERE.parents[1]
SCHEMA = 'hawking.glm52.activation_aware_pack.v2'
FORMAT_VERSION = 2
MAGIC = b'GLM52AA2'
HEADER_BYTES = 256
SEED = 169322466
HIDDEN = aap.HIDDEN
INTERMEDIATE = 2048
ORIGINAL_WEIGHT_COUNT = aap.ORIGINAL_WEIGHT_COUNT
SOURCE_PAYLOAD_BYTES = 1506659919872
EXPECTED_TENSOR_COUNT = 59585
TARGET_BPW = Fraction(49, 50)
SOURCE_HEADERS = REPORTS_ROOT / 'condense/glm52_generation_b/GLM52_SOURCE_SHARD_HEADERS.json'
PILOT_RECEIPT = evidence_dir('glm52') / 'GLM52_BASIS_PILOT_RECEIPT.json'
CONTROLLER_RESEAL = evidence_dir('glm52') / 'GLM52_BASIS_PILOT_CONTROLLER_RESEAL.json'
GEN_B_VERDICT = evidence_dir('glm52') / 'GLM52_GENERATION_B_CAPABILITY_VERDICT.json'
DEFAULT_FEASIBILITY_JSON = evidence_dir('glm52') / 'GLM52_V2_PROGRAM_FEASIBILITY.json'
DEFAULT_FEASIBILITY_MD = evidence_dir('glm52') / 'GLM52_V2_PROGRAM_FEASIBILITY.md'
MIN_ROUTE_ROWS = 32
SAFETY_FENCES: dict[str, bool] = {'RAMANUJAN_RESEARCH_AUTHORIZED': False, 'HIDE_KERNEL_TURN': False, 'ODYSSEY_LAUNCH_AUTHORIZED': False, 'full_parent_traversal_started': False, 'full_traversal_authorized': False, 'capable_artifact_claimed': False, 'MOP_touched': False}
ROUTED_RANK_LOWER_BOUND = 64
ROUTED_RANK_UNCERTAINTY_BOUND = 128
SENSITIVITY_FRACTIONS: tuple[Fraction, ...] = (Fraction(0, 1), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1, 1))
PREREGISTERED_PROGRAM: dict[str, dict[str, Any]] = {'routed_experts': {'organ_classes': ('routed_gate', 'routed_up', 'routed_down'), 'basis_mode': 'uncentered', 'rank': ROUTED_RANK_LOWER_BOUND, 'rank_uncertainty_bound': ROUTED_RANK_UNCERTAINTY_BOUND, 'panel_floor_min_cosine': 0.85, 'panel_floor_median_cosine': 0.96, 'per_tensor_floor_cosine': None, 'route_conditioned': True, 'note': 'Neutral static census class for all routed experts. Rank is assigned by ledger scenario (64 lower-bound vs 128 uncertainty-bound), not by traffic. Traffic is not present in sealed source headers.'}, 'high_traffic_routed_gate_up_down': {'organ_classes': ('high_traffic_routed_gate', 'high_traffic_routed_up', 'high_traffic_routed_down'), 'basis_mode': 'uncentered', 'rank': 64, 'panel_floor_min_cosine': 0.85, 'panel_floor_median_cosine': 0.96, 'per_tensor_floor_cosine': None, 'route_conditioned': True, 'note': 'Pilot evidence only (sealed five-shard high-traffic panel). Not used as a whole-model traffic classification in the static census.'}, 'low_traffic_routed_diagnostics': {'organ_classes': ('low_traffic_routed_gate', 'low_traffic_routed_up', 'low_traffic_routed_down'), 'basis_mode': 'uncentered', 'rank': 128, 'panel_floor_min_cosine': None, 'panel_floor_median_cosine': None, 'per_tensor_floor_cosine': 0.91, 'route_conditioned': True, 'note': 'Pilot diagnostic only; sealed low-traffic panel needed rank 128 to clear the 0.91 per-tensor floor. Not a population traffic map.'}, 'shared_mlp_gate_up_down': {'organ_classes': ('shared_mlp_gate', 'shared_mlp_up', 'shared_mlp_down'), 'basis_mode': 'uncentered', 'rank': 256, 'panel_floor_min_cosine': None, 'panel_floor_median_cosine': 0.93, 'per_tensor_floor_cosine': 0.91, 'route_conditioned': False, 'note': 'Shared expert uses all real capsule rows + actual gate/up for down.'}, 'router_control': {'organ_classes': ('router_control',), 'basis_mode': 'uncentered', 'rank': 128, 'panel_floor_min_cosine': None, 'panel_floor_median_cosine': None, 'per_tensor_floor_cosine': 0.99, 'route_conditioned': False, 'note': 'Router weight control; absolute floor 0.99.'}, 'attention_input_q_a_proj': {'organ_classes': ('attention_q_a_proj',), 'basis_mode': 'uncentered', 'rank': 128, 'panel_floor_min_cosine': None, 'panel_floor_median_cosine': None, 'per_tensor_floor_cosine': 0.91, 'route_conditioned': False, 'note': 'Only supported attention input projection under current capsules.'}}
NATIVE_UNVALIDATED_REASONS: dict[str, str] = {'attention_o_proj': 'Capsules lack the 16384-wide attention intermediate; Gaussian input forbidden.', 'attention_other': 'No bounded real-input pilot for this attention projection.', 'global_embed_tokens': 'embed_tokens not in five-shard pilot; remains native.', 'global_lm_head': 'lm_head not in five-shard pilot; remains native.', 'norm': 'Norms stay native (vector pass-through).', 'router_bias': 'Router bias/e_score stay native.', 'dense_mlp': 'Dense early layers (0-2) unvalidated under MoE pilot program.', 'other': 'No real-input pilot; billed native at source payload width.'}

class V2Error(RuntimeError):
    """Hard v2 failure (fail closed)."""

class RouteUndersampledError(V2Error):
    """Empty or undersampled expert route; never falls back to all rows."""

class BudgetFailure(V2Error):
    """Byte budget exceeded; floors and ranks are NOT reduced."""

class FloorFailure(V2Error):
    """Absolute organ floor not met; beats_null does not override."""

def sha256_file(path: Path, chunk: int=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return sha256_bytes(blob.encode('utf-8'))

@dataclass
class UncenteredBasis:
    """Orthonormal columns from uncentered SVD of real activation rows."""
    basis: np.ndarray
    singular_values: np.ndarray
    rank: int
    X_fit_mean: np.ndarray
    mean_direction_unit: np.ndarray
    mode: str = 'uncentered'

    def columns(self, rank: int | None=None) -> np.ndarray:
        r = self.rank if rank is None else min(int(rank), self.rank)
        if r <= 0:
            raise V2Error('rank must be positive')
        return self.basis[:, :r]

def build_uncentered_basis(X_fit: np.ndarray, rank: int) -> UncenteredBasis:
    """Uncentered SVD. Retains the mean direction in the leading subspace."""
    X_fit = np.asarray(X_fit, dtype=np.float32)
    if X_fit.ndim != 2 or X_fit.shape[0] < 1:
        raise V2Error(f'X_fit must be non-empty 2-D, got {getattr(X_fit, 'shape', None)}')
    n, h = X_fit.shape
    max_r = min(int(rank), h, n)
    if max_r < 1:
        raise V2Error('rank must be positive')
    mu = X_fit.mean(axis=0).astype(np.float32)
    mu_norm = float(np.linalg.norm(mu))
    if mu_norm < 1e-12:
        mdir = np.zeros(h, dtype=np.float32)
    else:
        mdir = (mu / mu_norm).astype(np.float32)
    _u, s, vt = np.linalg.svd(X_fit, full_matrices=False)
    r = min(max_r, vt.shape[0])
    B = vt[:r].T.astype(np.float32, copy=True)
    return UncenteredBasis(basis=B, singular_values=s[:r].astype(np.float32), rank=r, X_fit_mean=mu, mean_direction_unit=mdir, mode='uncentered')

def build_centered_basis_diagnostic(X_fit: np.ndarray, rank: int) -> np.ndarray:
    """Centered residual SVD — diagnostic / contrast only. Not a v2 promotion path."""
    X_fit = np.asarray(X_fit, dtype=np.float32)
    mu = X_fit.mean(axis=0)
    _u, s, vt = np.linalg.svd(X_fit - mu, full_matrices=False)
    r = min(int(rank), vt.shape[0], X_fit.shape[1])
    return vt[:r].T.astype(np.float32)

def mean_direction_retained(basis: UncenteredBasis, atol: float=0.5) -> bool:
    """Leading uncentered column must align with the empirical mean when mean dominates."""
    if float(np.linalg.norm(basis.mean_direction_unit)) < 1e-08:
        return False
    c0 = basis.columns(1)[:, 0]
    align = abs(float(np.dot(c0, basis.mean_direction_unit)))
    return align >= atol

def route_row_indices(topk: np.ndarray, expert_id: int) -> np.ndarray:
    """Row indices where expert_id appears in the top-k route list."""
    topk = np.asarray(topk)
    if topk.ndim != 2:
        raise V2Error(f'topk must be [N,K], got {topk.shape}')
    mask = (topk == int(expert_id)).any(axis=1)
    return np.flatnonzero(mask).astype(np.int64)

def select_route_rows(X: np.ndarray, topk: np.ndarray, expert_id: int, *, min_rows: int=MIN_ROUTE_ROWS) -> np.ndarray:
    """Select real pre_router_hidden rows for expert E. Never falls back to all rows."""
    idx = route_row_indices(topk, expert_id)
    if idx.size == 0:
        raise RouteUndersampledError(f'expert {expert_id}: empty route selection; refuse all-row fallback')
    if idx.size < int(min_rows):
        raise RouteUndersampledError(f'expert {expert_id}: only {idx.size} route rows < min_rows={min_rows}; fail closed (no silent fallback)')
    return np.asarray(X, dtype=np.float32)[idx]

def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))

def swiglu_intermediate(X: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray) -> np.ndarray:
    """Z = silu(X @ W_gate.T) * (X @ W_up.T). Real weights only; no Gaussian."""
    X = np.asarray(X, dtype=np.float32)
    W_gate = np.asarray(W_gate, dtype=np.float32)
    W_up = np.asarray(W_up, dtype=np.float32)
    if X.ndim != 2:
        raise V2Error(f'X must be 2-D, got {X.shape}')
    if W_gate.shape != W_up.shape:
        raise V2Error(f'gate/up shape mismatch {W_gate.shape} vs {W_up.shape}')
    if X.shape[1] != W_gate.shape[1]:
        raise V2Error(f'X width {X.shape[1]} != W_gate.in {W_gate.shape[1]}')
    g = silu(X @ W_gate.T)
    u = X @ W_up.T
    return g * u

def assert_no_gaussian_promotion_path(source_text: str | None=None) -> None:
    """Guard: v2 module source must not define a Gaussian promotion activation builder.

    Looks for real ``def`` callables that would construct proxy activations for
    promotional scoring. Explanatory strings and this guard itself are ignored by
    matching only at line starts (optional indent) for the banned def names.
    """
    import re as _re
    text = source_text if source_text is not None else Path(__file__).read_text(encoding='utf-8')
    banned = _re.compile('(?m)^\\s*def\\s+(gaussian_proxy|build_gaussian_activations|proxy_activations|gaussian_probe_activations)\\s*\\(')
    m = banned.search(text)
    if m:
        raise V2Error(f'forbidden promotion path present: def {m.group(1)}')

def project_factors(W: np.ndarray, B: np.ndarray, side: str='input') -> np.ndarray:
    if side != 'input':
        raise V2Error('v2 promotional path is input-side only (output-side down is a forbidden production negative control)')
    return np.asarray(W, dtype=np.float32) @ np.asarray(B, dtype=np.float32)

def reconstruct(L: np.ndarray, B: np.ndarray, side: str='input') -> np.ndarray:
    if side != 'input':
        raise V2Error('v2 promotional reconstruct is input-side only')
    return np.asarray(L, dtype=np.float32) @ np.asarray(B, dtype=np.float32).T

def mean_row_cosine(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(aap.mean_row_cosine(y, y_hat))

def constant_mean_null(y: np.ndarray) -> float:
    return float(aap.constant_mean_null(y))

def score_input_side(W: np.ndarray, W_hat: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    """Score on real inputs. beats_null is diagnostic only — never promotional."""
    X = np.asarray(X, dtype=np.float32)
    y = X @ np.asarray(W, dtype=np.float32).T
    y_hat = X @ np.asarray(W_hat, dtype=np.float32).T
    cos = mean_row_cosine(y, y_hat)
    null = constant_mean_null(y)
    return {'mean_row_cosine': cos, 'constant_mean_cosine_null': null, 'beats_null': bool(cos > null), 'surplus_over_null': cos - null, 'promotional': False}

@dataclass
class FloorSpec:
    per_tensor_min: float | None = None
    panel_min: float | None = None
    panel_median: float | None = None

def check_absolute_floors(cosines: Sequence[float], floors: FloorSpec, *, beats_null_flags: Sequence[bool] | None=None) -> dict[str, Any]:
    """Reject any point below its absolute floor even if it beats null."""
    cos = [float(c) for c in cosines]
    if not cos:
        raise FloorFailure('no cosines to evaluate')
    failures: list[str] = []
    per_tensor_ok = True
    if floors.per_tensor_min is not None:
        for i, c in enumerate(cos):
            if c < floors.per_tensor_min:
                per_tensor_ok = False
                bn = beats_null_flags[i] if beats_null_flags is not None and i < len(beats_null_flags) else None
                failures.append(f'tensor[{i}] cosine={c:.6f} < floor={floors.per_tensor_min}' + (f' (beats_null={bn} ignored)' if bn is not None else ''))
    panel_min = min(cos)
    panel_med = float(np.median(np.asarray(cos, dtype=np.float64)))
    panel_ok = True
    if floors.panel_min is not None and panel_min < floors.panel_min:
        panel_ok = False
        failures.append(f'panel min {panel_min:.6f} < floor {floors.panel_min}')
    if floors.panel_median is not None and panel_med < floors.panel_median:
        panel_ok = False
        failures.append(f'panel median {panel_med:.6f} < floor {floors.panel_median}')
    return {'ok': per_tensor_ok and panel_ok, 'panel_min': panel_min, 'panel_median': panel_med, 'failures': failures, 'beats_null_is_diagnostic_only': True}

def select_program_or_native(*, cosine: float, beats_null: bool, floor: float, source_payload_bytes: int, encoded_bytes: int, byte_budget_remaining: int | None) -> dict[str, Any]:
    """Allocator unit: absolute floor overrides beats_null; budget never lowers floor.

    If floor fails -> native at exact source width (allowed).
    If floor passes but encoded_bytes > budget_remaining -> BudgetFailure
    (do not reduce rank or floor).
    """
    if cosine < float(floor):
        return {'disposition': 'native', 'reason': 'absolute_floor_failed', 'cosine': cosine, 'floor': float(floor), 'beats_null': bool(beats_null), 'beats_null_overrode_floor': False, 'billed_bytes': int(source_payload_bytes), 'billing': 'source_payload_width'}
    if byte_budget_remaining is not None and encoded_bytes > byte_budget_remaining:
        raise BudgetFailure(f'encoded_bytes={encoded_bytes} exceeds remaining budget {byte_budget_remaining}; refuse to lower rank or floor (cosine={cosine} cleared floor={floor}; beats_null={beats_null} diagnostic only)')
    return {'disposition': 'activation_aware_v2', 'reason': 'absolute_floor_cleared', 'cosine': cosine, 'floor': float(floor), 'beats_null': bool(beats_null), 'billed_bytes': int(encoded_bytes), 'billing': 'coefficients_plus_shared_basis_share'}

def basis_identity(*, kind: str, layer: int, expert_id: int | None, rank: int, scope: str='target_local') -> str:
    """Serializable basis identity.

    target_local routed: one hidden basis and one swiglu basis per (layer, expert).
    transfer_layer: one basis per layer shared across experts (non-authorizing scenario).
    """
    if kind not in ('uncentered_hidden', 'real_swiglu_input', 'uncentered_attention_input', 'uncentered_router_input'):
        raise V2Error(f'unknown basis kind {kind!r}')
    if scope == 'target_local':
        if expert_id is None:
            return f'{scope}|{kind}|L{int(layer)}|r{int(rank)}'
        return f'{scope}|{kind}|L{int(layer)}|E{int(expert_id)}|r{int(rank)}'
    if scope == 'transfer_layer':
        return f'{scope}|{kind}|L{int(layer)}|r{int(rank)}|UNVALIDATED_TRANSFER'
    raise V2Error(f'unknown scope {scope!r}')

def basis_matrix_bytes(width: int, rank: int) -> int:
    """float16 basis matrix + basis header."""
    return HEADER_BYTES + int(width) * int(rank) * 2

def coefficient_bytes(rows: int, rank: int) -> int:
    """float16 coefficient matrix for input-side projection [rows, rank]."""
    return int(rows) * int(rank) * 2

def tensor_header_bytes() -> int:
    return HEADER_BYTES

@dataclass
class BasisLedger:
    """Bill each physical basis object exactly once; track refcounts."""
    bases: dict[str, dict[str, Any]] = field(default_factory=dict)
    coefficient_bytes_total: int = 0
    tensor_header_bytes_total: int = 0
    native_bytes_total: int = 0
    packaging_bytes_total: int = 0
    n_encoded_tensors: int = 0
    n_native_tensors: int = 0

    def add_basis(self, identity: str, *, width: int, rank: int, kind: str, authorizing: bool=True) -> int:
        if identity in self.bases:
            self.bases[identity]['refcount'] += 1
            return 0
        nbytes = basis_matrix_bytes(width, rank)
        self.bases[identity] = {'identity': identity, 'kind': kind, 'width': int(width), 'rank': int(rank), 'bytes': int(nbytes), 'refcount': 1, 'authorizing': bool(authorizing)}
        return int(nbytes)

    def add_coefficients(self, rows: int, rank: int) -> int:
        n = coefficient_bytes(rows, rank) + tensor_header_bytes()
        self.coefficient_bytes_total += coefficient_bytes(rows, rank)
        self.tensor_header_bytes_total += tensor_header_bytes()
        self.n_encoded_tensors += 1
        return n

    def add_native(self, payload_bytes: int) -> int:
        n = int(payload_bytes)
        self.native_bytes_total += n
        self.n_native_tensors += 1
        return n

    def basis_bytes_total(self) -> int:
        return int(sum((b['bytes'] for b in self.bases.values())))

    def total_bytes(self) -> int:
        return self.basis_bytes_total() + self.coefficient_bytes_total + self.tensor_header_bytes_total + self.native_bytes_total + self.packaging_bytes_total

    def component_totals(self) -> dict[str, int]:
        return {'float16_basis_matrices': self.basis_bytes_total(), 'float16_coefficient_matrices': int(self.coefficient_bytes_total), 'tensor_headers_metadata': int(self.tensor_header_bytes_total), 'native_source_payload': int(self.native_bytes_total), 'packaging_alignment': int(self.packaging_bytes_total)}

    def reconciles(self) -> bool:
        return sum(self.component_totals().values()) == self.total_bytes()

    def complete_bpw(self, weight_count: int=ORIGINAL_WEIGHT_COUNT) -> Fraction:
        if weight_count <= 0:
            raise V2Error('weight_count must be positive')
        return Fraction(self.total_bytes() * 8, int(weight_count))

    def as_dict(self, *, include_all_identities: bool=False) -> dict[str, Any]:
        comps = self.component_totals()
        total = self.total_bytes()
        bpw = self.complete_bpw()
        ordered = sorted(self.bases.values(), key=lambda b: b['identity'])
        id_list = [b['identity'] for b in ordered]
        by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {'n_unique': 0, 'refcount_sum': 0, 'bytes': 0})
        for b in ordered:
            k = str(b['kind'])
            by_kind[k]['n_unique'] += 1
            by_kind[k]['refcount_sum'] += int(b['refcount'])
            by_kind[k]['bytes'] += int(b['bytes'])
        out: dict[str, Any] = {'n_unique_bases': len(self.bases), 'basis_identities_sorted_sha256': sha256_json(id_list), 'basis_identity_examples': id_list[:12], 'basis_by_kind': {k: dict(v) for k, v in sorted(by_kind.items())}, 'basis_refcount_sum': int(sum((b['refcount'] for b in self.bases.values()))), 'component_totals': comps, 'total_bytes': total, 'itemization_reconciles': sum(comps.values()) == total, 'complete_bpw_exact': f'{bpw.numerator}/{bpw.denominator}', 'complete_bpw_float': float(bpw), 'n_encoded_tensors': self.n_encoded_tensors, 'n_native_tensors': self.n_native_tensors, 'note': 'Each physical basis is billed exactly once. Full identity strings are hashed (basis_identities_sorted_sha256); examples and by-kind counts are listed. Set include_all_identities for the full list.'}
        if include_all_identities:
            out['basis_identities'] = ordered
        return out
LAYER_RE = re.compile('model\\.layers\\.(\\d+)\\.')
EXPERT_RE = re.compile('model\\.layers\\.(\\d+)\\.mlp\\.experts\\.(\\d+)\\.(gate_proj|up_proj|down_proj)\\.weight$')
SHARED_RE = re.compile('model\\.layers\\.(\\d+)\\.mlp\\.shared_experts\\.(gate_proj|up_proj|down_proj)\\.weight$')
ROUTER_RE = re.compile('model\\.layers\\.(\\d+)\\.mlp\\.gate\\.weight$')
Q_A_RE = re.compile('model\\.layers\\.(\\d+)\\.self_attn\\.q_a_proj\\.weight$')
O_PROJ_RE = re.compile('model\\.layers\\.(\\d+)\\.self_attn\\.o_proj\\.weight$')
DENSE_MLP_RE = re.compile('model\\.layers\\.(\\d+)\\.mlp\\.(gate_proj|up_proj|down_proj)\\.weight$')

@dataclass
class TensorClass:
    name: str
    organ_class: str
    layer: int | None
    expert_id: int | None
    projection: str | None
    program_group: str | None
    shape: list[int]
    payload_bytes: int
    n_weights: int

def n_weights_of(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)

def classify_tensor(name: str, shape: Sequence[int], payload_bytes: int) -> TensorClass:
    """Map a sealed header entry to an organ class / program group."""
    shape_l = [int(x) for x in shape]
    nw = n_weights_of(shape_l)
    pb = int(payload_bytes)
    m = EXPERT_RE.match(name)
    if m:
        layer, expert, proj = (int(m.group(1)), int(m.group(2)), m.group(3))
        if proj == 'gate_proj':
            organ = 'routed_gate'
        elif proj == 'up_proj':
            organ = 'routed_up'
        else:
            organ = 'routed_down'
        return TensorClass(name=name, organ_class=organ, layer=layer, expert_id=expert, projection=proj, program_group='routed_experts', shape=shape_l, payload_bytes=pb, n_weights=nw)
    m = SHARED_RE.match(name)
    if m:
        layer, proj = (int(m.group(1)), m.group(2))
        if proj == 'gate_proj':
            organ = 'shared_mlp_gate'
        elif proj == 'up_proj':
            organ = 'shared_mlp_up'
        else:
            organ = 'shared_mlp_down'
        return TensorClass(name=name, organ_class=organ, layer=layer, expert_id=None, projection=proj, program_group='shared_mlp_gate_up_down', shape=shape_l, payload_bytes=pb, n_weights=nw)
    m = ROUTER_RE.match(name)
    if m:
        return TensorClass(name=name, organ_class='router_control', layer=int(m.group(1)), expert_id=None, projection='gate', program_group='router_control', shape=shape_l, payload_bytes=pb, n_weights=nw)
    m = Q_A_RE.match(name)
    if m:
        return TensorClass(name=name, organ_class='attention_q_a_proj', layer=int(m.group(1)), expert_id=None, projection='q_a_proj', program_group='attention_input_q_a_proj', shape=shape_l, payload_bytes=pb, n_weights=nw)
    m = O_PROJ_RE.match(name)
    if m:
        return TensorClass(name=name, organ_class='attention_o_proj', layer=int(m.group(1)), expert_id=None, projection='o_proj', program_group=None, shape=shape_l, payload_bytes=pb, n_weights=nw)
    m = DENSE_MLP_RE.match(name)
    if m:
        return TensorClass(name=name, organ_class='dense_mlp', layer=int(m.group(1)), expert_id=None, projection=m.group(2), program_group=None, shape=shape_l, payload_bytes=pb, n_weights=nw)
    if name == 'model.embed_tokens.weight' or name.endswith('embed_tokens.weight'):
        return TensorClass(name=name, organ_class='global_embed_tokens', layer=None, expert_id=None, projection=None, program_group=None, shape=shape_l, payload_bytes=pb, n_weights=nw)
    if name == 'lm_head.weight':
        return TensorClass(name=name, organ_class='global_lm_head', layer=None, expert_id=None, projection=None, program_group=None, shape=shape_l, payload_bytes=pb, n_weights=nw)
    if 'layernorm' in name or (name.endswith('.weight') and 'norm' in name):
        if any((s in name for s in ('norm', 'layernorm'))):
            return TensorClass(name=name, organ_class='norm', layer=_layer_or_none(name), expert_id=None, projection=None, program_group=None, shape=shape_l, payload_bytes=pb, n_weights=nw)
    if 'e_score_correction_bias' in name or name.endswith('.mlp.gate.bias'):
        return TensorClass(name=name, organ_class='router_bias', layer=_layer_or_none(name), expert_id=None, projection=None, program_group=None, shape=shape_l, payload_bytes=pb, n_weights=nw)
    if 'self_attn' in name:
        return TensorClass(name=name, organ_class='attention_other', layer=_layer_or_none(name), expert_id=None, projection=name.split('.')[-2] if name.endswith('.weight') else None, program_group=None, shape=shape_l, payload_bytes=pb, n_weights=nw)
    return TensorClass(name=name, organ_class='other', layer=_layer_or_none(name), expert_id=None, projection=None, program_group=None, shape=shape_l, payload_bytes=pb, n_weights=nw)

def _layer_or_none(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None

def load_source_headers(path: Path=SOURCE_HEADERS) -> list[dict[str, Any]]:
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    if 'headers' not in doc:
        raise V2Error(f'missing headers in {path}')
    entries: list[dict[str, Any]] = []
    for shard in doc['headers']:
        for t in shard['tensors']:
            entries.append(t)
    return entries

def build_census(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    classes: list[TensorClass] = []
    names: set[str] = set()
    total_w = 0
    total_b = 0
    organ_counts: Counter[str] = Counter()
    organ_weights: Counter[str] = Counter()
    organ_bytes: Counter[str] = Counter()
    for t in entries:
        name = t['name']
        if name in names:
            raise V2Error(f'duplicate tensor name {name}')
        names.add(name)
        tc = classify_tensor(name, t['shape'], t['payload_bytes'])
        classes.append(tc)
        total_w += tc.n_weights
        total_b += tc.payload_bytes
        organ_counts[tc.organ_class] += 1
        organ_weights[tc.organ_class] += tc.n_weights
        organ_bytes[tc.organ_class] += tc.payload_bytes
    reconcile = {'unique_tensor_names': len(names), 'expected_unique_tensor_names': EXPECTED_TENSOR_COUNT, 'unique_tensor_names_ok': len(names) == EXPECTED_TENSOR_COUNT, 'original_weights': total_w, 'expected_original_weights': ORIGINAL_WEIGHT_COUNT, 'original_weights_ok': total_w == ORIGINAL_WEIGHT_COUNT, 'source_payload_bytes': total_b, 'expected_source_payload_bytes': SOURCE_PAYLOAD_BYTES, 'source_payload_bytes_ok': total_b == SOURCE_PAYLOAD_BYTES}
    if not all([reconcile['unique_tensor_names_ok'], reconcile['original_weights_ok'], reconcile['source_payload_bytes_ok']]):
        raise V2Error(f'census reconciliation failed: {reconcile}')
    return {'tensors': classes, 'reconcile': reconcile, 'organ_counts': dict(organ_counts), 'organ_weights': dict(organ_weights), 'organ_bytes': dict(organ_bytes), 'n_tensors': len(classes)}

def _encode_tensor_into_ledger(tc: TensorClass, ledger: BasisLedger, *, scope: str, rank_override: int | None=None, basis_authorizing: bool=True) -> None:
    """Bill one tensor under the preregistered program into the ledger."""
    if tc.program_group is None:
        ledger.add_native(tc.payload_bytes)
        return
    prog = PREREGISTERED_PROGRAM[tc.program_group]
    rank = int(rank_override if rank_override is not None else prog['rank'])
    rows = int(tc.shape[0])
    if len(tc.shape) != 2:
        ledger.add_native(tc.payload_bytes)
        return
    ledger.add_coefficients(rows, rank)
    if tc.program_group == 'routed_experts':
        assert tc.layer is not None and tc.expert_id is not None
        if tc.projection in ('gate_proj', 'up_proj'):
            if scope == 'transfer_layer':
                bid = basis_identity(kind='uncentered_hidden', layer=tc.layer, expert_id=None, rank=rank, scope=scope)
            else:
                bid = basis_identity(kind='uncentered_hidden', layer=tc.layer, expert_id=tc.expert_id, rank=rank, scope=scope)
            ledger.add_basis(bid, width=HIDDEN, rank=rank, kind='uncentered_hidden', authorizing=bool(basis_authorizing) and scope == 'target_local')
        elif tc.projection == 'down_proj':
            if scope == 'target_local':
                bid = basis_identity(kind='real_swiglu_input', layer=tc.layer, expert_id=tc.expert_id, rank=rank, scope=scope)
            else:
                bid = basis_identity(kind='real_swiglu_input', layer=tc.layer, expert_id=None, rank=rank, scope=scope)
            ledger.add_basis(bid, width=INTERMEDIATE, rank=rank, kind='real_swiglu_input', authorizing=bool(basis_authorizing) and scope == 'target_local')
        else:
            raise V2Error(f'unexpected routed projection {tc.projection}')
        return
    if tc.program_group == 'shared_mlp_gate_up_down':
        assert tc.layer is not None
        if tc.projection in ('gate_proj', 'up_proj'):
            bid = f'target_local|uncentered_hidden|shared|L{tc.layer}|r{rank}'
            ledger.add_basis(bid, width=HIDDEN, rank=rank, kind='uncentered_hidden', authorizing=True)
        elif tc.projection == 'down_proj':
            bid = f'target_local|real_swiglu_input|shared|L{tc.layer}|r{rank}'
            ledger.add_basis(bid, width=INTERMEDIATE, rank=rank, kind='real_swiglu_input', authorizing=True)
        return
    if tc.program_group == 'router_control':
        assert tc.layer is not None
        bid = f'target_local|uncentered_router_input|L{tc.layer}|r{rank}'
        ledger.add_basis(bid, width=HIDDEN, rank=rank, kind='uncentered_router_input', authorizing=True)
        return
    if tc.program_group == 'attention_input_q_a_proj':
        assert tc.layer is not None
        bid = f'target_local|uncentered_attention_input|q_a|L{tc.layer}|r{rank}'
        ledger.add_basis(bid, width=HIDDEN, rank=rank, kind='uncentered_attention_input', authorizing=True)
        return
    ledger.add_native(tc.payload_bytes)

def list_routed_experts(tensors: Sequence[TensorClass]) -> list[tuple[int, int]]:
    """Sorted unique (layer, expert) identities for arithmetic sensitivity."""
    seen: set[tuple[int, int]] = set()
    for tc in tensors:
        if tc.program_group == 'routed_experts' and tc.layer is not None and (tc.expert_id is not None):
            seen.add((int(tc.layer), int(tc.expert_id)))
    return sorted(seen)

def build_routed_rank_ledger(tensors: Sequence[TensorClass], *, routed_rank: int, scope: str='target_local', basis_authorizing: bool=True) -> BasisLedger:
    """Target-local billing with a uniform rank for all routed experts.

    Shared MLP, router, and q_a retain their preregistered ranks. One expert is
    gate/up/down with a shared hidden basis and a separate real-SwiGLU basis.
    """
    ledger = BasisLedger()
    for tc in tensors:
        if tc.program_group == 'routed_experts':
            _encode_tensor_into_ledger(tc, ledger, scope=scope, rank_override=int(routed_rank), basis_authorizing=basis_authorizing)
        else:
            _encode_tensor_into_ledger(tc, ledger, scope=scope, basis_authorizing=basis_authorizing)
    return ledger

def build_routed_mixture_ledger(tensors: Sequence[TensorClass], *, rank128_experts: set[tuple[int, int]], rank64: int=ROUTED_RANK_LOWER_BOUND, rank128: int=ROUTED_RANK_UNCERTAINTY_BOUND) -> BasisLedger:
    """Target-local billing with a deterministic mixture of ranks 64 and 128.

    Experts in ``rank128_experts`` (layer, expert) use rank 128; all other
    routed experts use rank 64. Non-routed organs keep preregistered ranks.
    """
    ledger = BasisLedger()
    for tc in tensors:
        if tc.program_group == 'routed_experts':
            assert tc.layer is not None and tc.expert_id is not None
            key = (int(tc.layer), int(tc.expert_id))
            r = int(rank128) if key in rank128_experts else int(rank64)
            _encode_tensor_into_ledger(tc, ledger, scope='target_local', rank_override=r, basis_authorizing=False)
        else:
            _encode_tensor_into_ledger(tc, ledger, scope='target_local', basis_authorizing=False)
    return ledger

def build_all_routed_rank64_lower_bound_ledger(tensors: Sequence[TensorClass]) -> BasisLedger:
    """All routed experts at rank 64 — optimistic/lower-bound only; non-authorizing."""
    return build_routed_rank_ledger(tensors, routed_rank=ROUTED_RANK_LOWER_BOUND, scope='target_local', basis_authorizing=False)

def build_all_routed_rank128_uncertainty_bound_ledger(tensors: Sequence[TensorClass]) -> BasisLedger:
    """All routed experts at rank 128 — sole authorization-deciding BPW ledger.

    This is a byte-feasibility uncertainty bound. It does not prove that rank 128
    is quality-sufficient for every routed expert.
    """
    return build_routed_rank_ledger(tensors, routed_rank=ROUTED_RANK_UNCERTAINTY_BOUND, scope='target_local', basis_authorizing=True)

def build_conservative_ledger(tensors: Sequence[TensorClass]) -> BasisLedger:
    """Deprecated alias: rank-64 lower-bound ledger (non-authorizing).

    Kept for callers that still use the revision-0 name. Does not decide
    ``within_target_bpw``; use the rank-128 uncertainty-bound ledger instead.
    """
    return build_all_routed_rank64_lower_bound_ledger(tensors)

def build_transfer_scenario_ledger(tensors: Sequence[TensorClass]) -> BasisLedger:
    """Non-authorizing scenario: share one hidden/swiglu basis per layer across experts.

    Labelled UNVALIDATED_TRANSFER. Must not affect any top-level decision.
    Uses rank-64 for routed experts (informational only).
    """
    return build_routed_rank_ledger(tensors, routed_rank=ROUTED_RANK_LOWER_BOUND, scope='transfer_layer', basis_authorizing=False)

def route_population_sensitivity(tensors: Sequence[TensorClass]) -> dict[str, Any]:
    """Exact arithmetic sensitivity of BPW to rank-128 vs rank-64 routed experts.

    Selection is deterministic: sorted ``(layer, expert)`` identities, prefix of
    size k upgraded to rank 128. This is arithmetic sensitivity, not a traffic
    classification.
    """
    experts = list_routed_experts(tensors)
    n = len(experts)
    if n == 0:
        raise V2Error('no routed experts in census')
    sweep: list[dict[str, Any]] = []
    prev_total: int | None = None
    for frac in SENSITIVITY_FRACTIONS:
        k = int(frac * n)
        rank128_set = set(experts[:k])
        led = build_routed_mixture_ledger(tensors, rank128_experts=rank128_set)
        total = led.total_bytes()
        bpw = led.complete_bpw()
        if prev_total is not None and total < prev_total:
            raise V2Error(f'sensitivity non-monotonic at frac={frac}: {total} < {prev_total}')
        prev_total = total
        sweep.append({'fraction_rank128': f'{frac.numerator}/{frac.denominator}', 'fraction_rank128_float': float(frac), 'n_rank128_experts': k, 'n_rank64_experts': n - k, 'total_bytes': total, 'complete_bpw_exact': f'{bpw.numerator}/{bpw.denominator}', 'complete_bpw_float': float(bpw), 'within_target_bpw': bool(bpw <= TARGET_BPW), 'itemization_reconciles': led.reconciles()})
    lo, hi = (0, n)
    best = 0
    best_bytes = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        led = build_routed_mixture_ledger(tensors, rank128_experts=set(experts[:mid]))
        if led.complete_bpw() <= TARGET_BPW:
            best = mid
            best_bytes = led.total_bytes()
            lo = mid + 1
        else:
            hi = mid - 1
    best_frac = Fraction(best, n)
    return {'kind': 'arithmetic_sensitivity', 'not_traffic_classification': True, 'selection_rule': 'sorted_(layer, expert)_prefix', 'expert_count_unit': 'one expert = gate/up/down triplet with shared hidden basis and separate real-SwiGLU-input basis', 'n_routed_experts': n, 'rank64': ROUTED_RANK_LOWER_BOUND, 'rank128': ROUTED_RANK_UNCERTAINTY_BOUND, 'target_bpw': f'{TARGET_BPW.numerator}/{TARGET_BPW.denominator}', 'sweep': sweep, 'max_rank128_experts_under_target_bpw': best, 'max_rank128_fraction_under_target_bpw_exact': f'{best_frac.numerator}/{best_frac.denominator}', 'max_rank128_fraction_under_target_bpw_float': float(best_frac), 'max_rank128_total_bytes': best_bytes, 'monotonic_total_bytes': True, 'note': 'Deterministic arithmetic sensitivity only. Does not classify experts by traffic and does not prove representation quality at either rank.'}

@dataclass
class V2TensorMeta:
    format_version: int
    organ_class: str
    layer: int
    expert_id: int | None
    projection_side: str
    basis_kind: str
    basis_identity: str
    rank: int
    activation_provenance: str
    route_conditioned: bool
    rows: int
    cols: int

def _pack_meta(meta: V2TensorMeta) -> bytes:
    """Fixed HEADER_BYTES metadata prefix for the fake ABI."""
    kind_codes = {'uncentered_hidden': 1, 'real_swiglu_input': 2, 'uncentered_attention_input': 3, 'uncentered_router_input': 4}
    kind_code = kind_codes.get(meta.basis_kind, 0)
    expert = -1 if meta.expert_id is None else int(meta.expert_id)
    head = struct.pack('<8sIIIIiibbB', MAGIC, int(meta.format_version), int(meta.rows), int(meta.cols), int(meta.rank), int(meta.layer), int(expert), 1, 1 if meta.route_conditioned else 0, kind_code)

    def _lp(s: str) -> bytes:
        b = s.encode('utf-8')
        if len(b) > 255:
            b = b[:255]
        return bytes([len(b)]) + b
    body = _lp(meta.basis_identity) + _lp(meta.organ_class) + _lp(meta.activation_provenance) + _lp(meta.basis_kind) + _lp(meta.projection_side)
    raw = head + body
    if len(raw) > HEADER_BYTES:
        raise V2Error(f'metadata overflow {len(raw)} > {HEADER_BYTES}')
    return raw + b'\x00' * (HEADER_BYTES - len(raw))

def _unpack_meta(blob: bytes) -> dict[str, Any]:
    if blob[:8] != MAGIC:
        raise V2Error('not a v2 activation-aware payload')
    ver, rows, cols, rank, layer, expert, side_code, route, kind_code = struct.unpack_from('<IIIIiibbB', blob, 8)
    off = 8 + struct.calcsize('<IIIIiibbB')
    fields = []
    for _ in range(5):
        n = blob[off]
        off += 1
        fields.append(blob[off:off + n].decode('utf-8'))
        off += n
    identity, organ, provenance, basis_kind, proj_side = fields
    kind_names = {1: 'uncentered_hidden', 2: 'real_swiglu_input', 3: 'uncentered_attention_input', 4: 'uncentered_router_input'}
    return {'format_version': ver, 'rows': rows, 'cols': cols, 'rank': rank, 'layer': layer, 'expert_id': None if expert < 0 else expert, 'projection_side': proj_side or ('input' if side_code == 1 else 'unknown'), 'route_conditioned': bool(route), 'basis_kind': basis_kind or kind_names.get(kind_code, 'unknown'), 'basis_identity': identity, 'organ_class': organ, 'activation_provenance': provenance}

def encode_tensor_v2(W: np.ndarray, B: np.ndarray, meta: V2TensorMeta) -> bytes:
    """Encode W on input-side shared basis B. Deterministic float16 payload."""
    W = np.asarray(W, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if W.shape != (meta.rows, meta.cols):
        raise V2Error(f'W shape {W.shape} != meta {(meta.rows, meta.cols)}')
    if B.shape != (meta.cols, meta.rank):
        raise V2Error(f'B shape {B.shape} != expected {(meta.cols, meta.rank)}')
    L = project_factors(W, B, 'input')
    head = _pack_meta(meta)
    body = np.ascontiguousarray(L, dtype=np.float16).tobytes()
    blob = head + body
    return blob

def encode_self_contained(W: np.ndarray, B: np.ndarray, meta: V2TensorMeta) -> bytes:
    """Self-contained blob: header + float16 L + float16 B (for round-trip tests)."""
    base = encode_tensor_v2(W, B, meta)
    return base + np.ascontiguousarray(B, dtype=np.float16).tobytes()

def decode_self_contained(blob: bytes) -> dict[str, Any]:
    meta = _unpack_meta(blob[:HEADER_BYTES])
    rows, cols, rank = (meta['rows'], meta['cols'], meta['rank'])
    coeff_nbytes = rows * rank * 2
    off = HEADER_BYTES
    L = np.frombuffer(blob[off:off + coeff_nbytes], dtype=np.float16).reshape(rows, rank)
    off += coeff_nbytes
    basis_nbytes = cols * rank * 2
    B = np.frombuffer(blob[off:off + basis_nbytes], dtype=np.float16).reshape(cols, rank)
    W_hat = reconstruct(L.astype(np.float32), B.astype(np.float32), 'input')
    return {'meta': meta, 'L': L, 'B': B, 'W_hat': W_hat}

def _array_sha256(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return sha256_bytes(a.tobytes())

def build_down_basis_from_swiglu(X_route: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, rank: int) -> dict[str, Any]:
    """Form Z from real matching gate/up weights and build uncentered B_z on Z.

    Random rows may not substitute for Z while claiming real_swiglu_input.
    """
    Z = swiglu_intermediate(X_route, W_gate, W_up)
    basis = build_uncentered_basis(Z, rank)
    B_z = basis.columns(rank)
    return {'Z': Z, 'B_z': B_z, 'X_route_sha256': _array_sha256(X_route), 'Z_sha256': _array_sha256(Z), 'B_z_sha256': _array_sha256(B_z), 'route_row_count': int(X_route.shape[0]), 'rank': int(rank)}

def verify_down_basis_witness(*, X_route: np.ndarray, W_gate: np.ndarray, W_up: np.ndarray, B_z: np.ndarray, rank: int, expected: dict[str, Any], atol: float=1e-05) -> dict[str, Any]:
    """Recompute Z and B_z; bind hashes. Unrelated bases fail."""
    rebuilt = build_down_basis_from_swiglu(X_route, W_gate, W_up, rank)
    shape_ok = tuple(B_z.shape) == tuple(rebuilt['B_z'].shape)
    close = shape_ok and bool(np.allclose(B_z, rebuilt['B_z'], atol=atol, rtol=0.0))
    hash_x = rebuilt['X_route_sha256'] == expected.get('X_route_sha256')
    hash_z = rebuilt['Z_sha256'] == expected.get('Z_sha256')
    hash_b = rebuilt['B_z_sha256'] == expected.get('B_z_sha256')
    hash_b_direct = _array_sha256(np.asarray(B_z, dtype=np.float32)) == expected.get('B_z_sha256')
    ok = bool(close and hash_x and hash_z and hash_b and hash_b_direct)
    return {'ok': ok, 'shape_ok': shape_ok, 'numeric_close': close, 'X_route_hash_match': hash_x, 'Z_hash_match': hash_z, 'B_z_hash_match': hash_b and hash_b_direct, 'rebuilt_B_z_sha256': rebuilt['B_z_sha256'], 'candidate_B_z_sha256': _array_sha256(np.asarray(B_z, dtype=np.float32))}

def fake_gate_up_down_roundtrip(*, seed: int=SEED, layer: int=5, expert_a: int=11, expert_b: int=165, rank: int=16, n_tokens: int=256, min_route_rows: int=48, hidden: int=64, intermediate: int=32) -> dict[str, Any]:
    """Deterministic fake proof with truthful route rows and SwiGLU down basis.

    Program:
      1. Fake pre-router rows X and synthetic topk_indices.
      2. select_route_rows -> expert-specific X_route for hidden bases.
      3. Matching fake W_gate, W_up form Z = swiglu_intermediate(X_route, ...).
      4. B_z = build_uncentered_basis(Z, rank) for down.
      5. Serialized down metadata still says real_swiglu_input.
      6. Witnesses bind X_route, Z, B_z plus route counts.
      7. Unrelated random basis fails the witness (returned for negative tests).
    """
    rng = np.random.default_rng(int(seed))
    X = (rng.standard_normal((n_tokens, hidden)) + 2.0).astype(np.float32)
    topk = np.zeros((n_tokens, 3), dtype=np.int32)
    half = n_tokens // 2
    topk[:half, 0] = expert_a
    topk[half:, 0] = expert_b
    topk[:, 1] = (topk[:, 0] + 3) % max(expert_b + 10, 200)
    topk[:, 2] = (topk[:, 0] + 7) % max(expert_b + 10, 200)
    if half < min_route_rows:
        raise V2Error(f'fake fixture n_tokens={n_tokens} yields {half} rows/expert < min_route_rows={min_route_rows}')
    X_route_a = select_route_rows(X, topk, expert_a, min_rows=min_route_rows)
    X_route_b = select_route_rows(X, topk, expert_b, min_rows=min_route_rows)
    B_h_a = build_uncentered_basis(X_route_a, rank).columns(rank)
    B_h_b = build_uncentered_basis(X_route_b, rank).columns(rank)
    W_gate_a = rng.standard_normal((intermediate, hidden)).astype(np.float32)
    W_up_a = rng.standard_normal((intermediate, hidden)).astype(np.float32)
    W_down_a = rng.standard_normal((hidden, intermediate)).astype(np.float32)
    W_gate_b = rng.standard_normal((intermediate, hidden)).astype(np.float32)
    W_up_b = rng.standard_normal((intermediate, hidden)).astype(np.float32)
    down_a = build_down_basis_from_swiglu(X_route_a, W_gate_a, W_up_a, rank)
    B_z_a = down_a['B_z']
    down_b = build_down_basis_from_swiglu(X_route_b, W_gate_b, W_up_b, rank)
    B_z_b = down_b['B_z']
    id_h_a = basis_identity(kind='uncentered_hidden', layer=layer, expert_id=expert_a, rank=rank)
    id_h_b = basis_identity(kind='uncentered_hidden', layer=layer, expert_id=expert_b, rank=rank)
    id_z_a = basis_identity(kind='real_swiglu_input', layer=layer, expert_id=expert_a, rank=rank)
    id_z_b = basis_identity(kind='real_swiglu_input', layer=layer, expert_id=expert_b, rank=rank)
    assert id_h_a != id_h_b, 'experts must not alias basis identities'
    assert id_h_a != id_z_a, 'down SwiGLU basis must differ from hidden basis'

    def _meta(organ: str, eid: int, kind: str, bid: str, rows: int, cols: int) -> V2TensorMeta:
        return V2TensorMeta(format_version=FORMAT_VERSION, organ_class=organ, layer=layer, expert_id=eid, projection_side='input', basis_kind=kind, basis_identity=bid, rank=rank, activation_provenance='fake_route_selected_swiglu', route_conditioned=True, rows=rows, cols=cols)
    blobs = {'gate_a': encode_self_contained(W_gate_a, B_h_a, _meta('routed_gate', expert_a, 'uncentered_hidden', id_h_a, intermediate, hidden)), 'up_a': encode_self_contained(W_up_a, B_h_a, _meta('routed_up', expert_a, 'uncentered_hidden', id_h_a, intermediate, hidden)), 'down_a': encode_self_contained(W_down_a, B_z_a, _meta('routed_down', expert_a, 'real_swiglu_input', id_z_a, hidden, intermediate)), 'gate_b': encode_self_contained(W_gate_b, B_h_b, _meta('routed_gate', expert_b, 'uncentered_hidden', id_h_b, intermediate, hidden))}
    dec = {k: decode_self_contained(v) for k, v in blobs.items()}
    assert dec['gate_a']['meta']['basis_identity'] == dec['up_a']['meta']['basis_identity']
    assert dec['gate_a']['meta']['basis_identity'] != dec['gate_b']['meta']['basis_identity']
    assert dec['down_a']['meta']['basis_kind'] == 'real_swiglu_input'
    assert dec['gate_a']['meta']['basis_kind'] == 'uncentered_hidden'
    assert dec['down_a']['meta']['organ_class'] == 'routed_down'
    for key, W, B_orig in (('gate_a', W_gate_a, B_h_a), ('up_a', W_up_a, B_h_a), ('down_a', W_down_a, B_z_a), ('gate_b', W_gate_b, B_h_b)):
        L_dec = dec[key]['L'].astype(np.float32)
        L_ref = project_factors(W, B_orig, 'input')
        err = float(np.max(np.abs(L_dec - L_ref.astype(np.float16).astype(np.float32))))
        if err > 0.01 * (float(np.max(np.abs(L_ref))) + 1e-06):
            raise V2Error(f'{key} coeff roundtrip err {err}')
    if not np.allclose(dec['gate_a']['B'], dec['up_a']['B']):
        raise V2Error('gate/up must share identical basis matrix for expert A')
    if np.allclose(dec['gate_a']['B'], dec['gate_b']['B']):
        raise V2Error('different experts must not alias bases')
    if dec['down_a']['B'].shape[0] != intermediate:
        raise V2Error('down basis width must be intermediate (SwiGLU input)')
    witness_expected = {'X_route_sha256': down_a['X_route_sha256'], 'Z_sha256': down_a['Z_sha256'], 'B_z_sha256': down_a['B_z_sha256']}
    positive = verify_down_basis_witness(X_route=X_route_a, W_gate=W_gate_a, W_up=W_up_a, B_z=B_z_a, rank=rank, expected=witness_expected)
    if not positive['ok']:
        raise V2Error(f'positive SwiGLU down witness failed: {positive}')
    B_unrelated = build_uncentered_basis((rng.standard_normal((min_route_rows, intermediate)) + 0.5).astype(np.float32), rank).columns(rank)
    negative = verify_down_basis_witness(X_route=X_route_a, W_gate=W_gate_a, W_up=W_up_a, B_z=B_unrelated, rank=rank, expected=witness_expected)
    if negative['ok']:
        raise V2Error('unrelated random basis incorrectly passed SwiGLU witness')
    B_h_a_re = build_uncentered_basis(X_route_a, rank).columns(rank)
    if not np.allclose(B_h_a, B_h_a_re, atol=1e-05):
        raise V2Error('hidden basis must be rebuildable from selected route rows')
    return {'ok': True, 'seed': int(seed), 'basis_identities': {'gate_up_a': id_h_a, 'gate_b': id_h_b, 'down_a': id_z_a, 'down_b': id_z_b}, 'gate_up_share': True, 'experts_non_aliasing': True, 'down_separate_swiglu_basis': True, 'down_basis_from_real_swiglu': True, 'hidden_basis_from_route_rows': True, 'format_version': FORMAT_VERSION, 'route_counts': {'expert_a': int(X_route_a.shape[0]), 'expert_b': int(X_route_b.shape[0]), 'n_tokens': int(n_tokens), 'min_route_rows': int(min_route_rows)}, 'witnesses': {'expert_a': {'X_route_sha256': down_a['X_route_sha256'], 'Z_sha256': down_a['Z_sha256'], 'B_z_sha256': down_a['B_z_sha256'], 'route_row_count': down_a['route_row_count']}, 'expert_b': {'X_route_sha256': down_b['X_route_sha256'], 'Z_sha256': down_b['Z_sha256'], 'B_z_sha256': down_b['B_z_sha256'], 'route_row_count': down_b['route_row_count']}, 'positive_verify_ok': True, 'unrelated_basis_fails_witness': True, 'unrelated_B_z_sha256': _array_sha256(B_unrelated)}, 'fixture_handles': {'layer': layer, 'expert_a': expert_a, 'expert_b': expert_b, 'rank': rank, 'hidden': hidden, 'intermediate': intermediate, 'X_route_a_sha256': _array_sha256(X_route_a), 'B_h_a_sha256': _array_sha256(B_h_a), 'B_z_a_sha256': down_a['B_z_sha256'], 'B_z_b_sha256': down_b['B_z_sha256']}}

def pilot_checks_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Summarise sealed pilot evidence against the preregistered v2 candidate."""
    uncentered_panels = receipt.get('verdict', {}).get('panel_summaries', {}).get('uncentered', {})
    checks: list[dict[str, Any]] = []

    def _panel_at(panel: str, rank: int) -> dict[str, Any] | None:
        by = uncentered_panels.get(panel, {}).get('by_rank', {})
        return by.get(str(rank)) or by.get(rank)
    ht = _panel_at('promotion_grade_high_traffic_routed', 64)
    if ht and ht.get('min') is not None:
        floors = FloorSpec(panel_min=0.85, panel_median=0.96)
        cos = [ht['min'], ht['median']]
        ok = ht['min'] >= 0.85 and ht['median'] >= 0.96
        checks.append({'program': 'high_traffic_routed_gate_up_down', 'rank': 64, 'panel': 'promotion_grade_high_traffic_routed', 'measured_min': ht['min'], 'measured_median': ht['median'], 'floor_min': 0.85, 'floor_median': 0.96, 'clears': ok})
    lt = _panel_at('low_traffic_diagnostics', 128)
    if lt and lt.get('min') is not None:
        checks.append({'program': 'low_traffic_routed_diagnostics', 'rank': 128, 'panel': 'low_traffic_diagnostics', 'measured_min': lt['min'], 'measured_median': lt['median'], 'per_tensor_floor': 0.91, 'clears': lt['min'] >= 0.91})
    sh = _panel_at('shared_mlp', 256)
    if sh and sh.get('min') is not None:
        checks.append({'program': 'shared_mlp_gate_up_down', 'rank': 256, 'panel': 'shared_mlp', 'measured_min': sh['min'], 'measured_median': sh['median'], 'per_tensor_floor': 0.91, 'panel_median_floor': 0.93, 'clears': sh['min'] >= 0.91 and sh['median'] >= 0.93})
    for t in receipt.get('tensor_results', []):
        organ = t.get('organ_class')
        arms = t.get('arms', {}).get('uncentered', [])
        if organ == 'router_control':
            pt = next((p for p in arms if p.get('requested_rank') == 128), None)
            if pt:
                checks.append({'program': 'router_control', 'rank': 128, 'name': t['name'], 'measured_cosine': pt['mean_row_cosine'], 'per_tensor_floor': 0.99, 'clears': pt['mean_row_cosine'] >= 0.99})
        if organ == 'attention_q':
            pt = next((p for p in arms if p.get('requested_rank') == 128), None)
            if pt:
                checks.append({'program': 'attention_input_q_a_proj', 'rank': 128, 'name': t['name'], 'measured_cosine': pt['mean_row_cosine'], 'per_tensor_floor': 0.91, 'clears': pt['mean_row_cosine'] >= 0.91})
    uncentered_tied = receipt.get('verdict', {}).get('uncentered_explicit_mean_numerically_tied', True)
    return {'pilot_schema': receipt.get('schema'), 'pilot_revision': receipt.get('revision'), 'basis_mode_chosen': 'uncentered', 'uncentered_explicit_mean_numerically_tied': uncentered_tied, 'numerical_equivalence_tolerance': receipt.get('numerical_equivalence_tolerance', 0.0001), 'centered_only_forbidden': True, 'checks': checks, 'all_candidate_checks_clear': all((c.get('clears', False) for c in checks)), 'gaussian_proxy_used_for_selection': receipt.get('safety', {}).get('gaussian_proxy_used_for_selection', None)}

def build_feasibility_receipt(*, headers_path: Path=SOURCE_HEADERS, pilot_path: Path=PILOT_RECEIPT) -> dict[str, Any]:
    entries = load_source_headers(headers_path)
    census = build_census(entries)
    tensors: list[TensorClass] = census['tensors']
    lb64 = build_all_routed_rank64_lower_bound_ledger(tensors)
    ub128 = build_all_routed_rank128_uncertainty_bound_ledger(tensors)
    xfer = build_transfer_scenario_ledger(tensors)
    sensitivity = route_population_sensitivity(tensors)
    lb64_d = lb64.as_dict()
    ub128_d = ub128.as_dict()
    xfer_d = xfer.as_dict()
    ub128_bpw = ub128.complete_bpw()
    lb64_bpw = lb64.complete_bpw()
    within = bool(ub128_bpw <= TARGET_BPW)
    ranks_in_ub = {int(b['rank']) for b in ub128.bases.values()}
    routed_ranks_ub = {int(b['rank']) for b in ub128.bases.values() if '|E' in str(b['identity'])}
    if routed_ranks_ub != {ROUTED_RANK_UNCERTAINTY_BOUND}:
        raise V2Error(f'uncertainty ledger routed ranks must be {{{ROUTED_RANK_UNCERTAINTY_BOUND}}}, got {routed_ranks_ub}; refuse rank reduction to force budget')
    pilot = {}
    if pilot_path.exists():
        with open(pilot_path, encoding='utf-8') as f:
            pilot_doc = json.load(f)
        pilot = pilot_checks_from_receipt(pilot_doc)
    unsupported = []
    for organ, reason in NATIVE_UNVALIDATED_REASONS.items():
        unsupported.append({'organ_class': organ, 'n_tensors': census['organ_counts'].get(organ, 0), 'n_weights': census['organ_weights'].get(organ, 0), 'source_payload_bytes': census['organ_bytes'].get(organ, 0), 'why': reason, 'billing': 'native_source_payload_width'})
    code_hashes = {'glm52_activation_aware_pack_v2_py_sha256': sha256_file(Path(__file__)), 'glm52_activation_aware_pack_py_sha256': sha256_file(HERE / 'glm52_activation_aware_pack.py')}
    test_path = HERE / 'tests' / 'test_glm52_activation_aware_pack_v2.py'
    if test_path.exists():
        code_hashes['test_glm52_activation_aware_pack_v2_py_sha256'] = sha256_file(test_path)
    source_hashes = {'GLM52_SOURCE_SHARD_HEADERS_sha256': sha256_file(headers_path)}
    if pilot_path.exists():
        source_hashes['GLM52_BASIS_PILOT_RECEIPT_sha256'] = sha256_file(pilot_path)
    if CONTROLLER_RESEAL.exists():
        source_hashes['GLM52_BASIS_PILOT_CONTROLLER_RESEAL_sha256'] = sha256_file(CONTROLLER_RESEAL)
    if GEN_B_VERDICT.exists():
        source_hashes['GLM52_GENERATION_B_CAPABILITY_VERDICT_sha256'] = sha256_file(GEN_B_VERDICT)
    fake = fake_gate_up_down_roundtrip(seed=SEED)
    n_routed = len(list_routed_experts(tensors))
    traffic_keys = [k for k in census['organ_counts'] if k.startswith('high_traffic_') or k.startswith('low_traffic_')]
    if traffic_keys:
        raise V2Error(f'static census must not traffic-label routed organs; found {traffic_keys}')
    receipt = {'schema': SCHEMA, 'format_version': FORMAT_VERSION, 'revision': 1, 'purpose': 'Opt-in feasibility + fake-data ABI for the next GLM-5.2 representation program. Corrects Generation B scientific defects without traversal or capability claims. Revision 1: route-population uncertainty + truthful fake SwiGLU down basis.', 'seed': SEED, 'target_bpw': f'{TARGET_BPW.numerator}/{TARGET_BPW.denominator}', 'preregistered_program': {k: {**{kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()}} for k, v in PREREGISTERED_PROGRAM.items()}, 'scientific_laws': {'basis_mode': 'uncentered', 'centered_only_fitting_forbidden': True, 'route_conditioned_routed_experts': True, 'empty_route_fails_closed': True, 'real_swiglu_inputs_for_down': True, 'gaussian_proxy_forbidden_for_promotion': True, 'beats_null_diagnostic_only': True, 'absolute_floors_required': True, 'budget_failure_never_reduces_floor': True, 'native_fallback_at_source_payload_width_only': True, 'transfer_sharing_non_authorizing': True, 'rank64_population_fit_is_lower_bound_only': True, 'within_target_bpw_decided_by_rank128_uncertainty_bound_only': True}, 'source_hashes': source_hashes, 'code_hashes': code_hashes, 'pilot_checks': pilot, 'census': {'reconcile': census['reconcile'], 'organ_counts': census['organ_counts'], 'organ_weights': census['organ_weights'], 'organ_bytes': census['organ_bytes'], 'n_tensors': census['n_tensors'], 'n_routed_experts': n_routed, 'static_routed_classification': 'routed_gate/up/down', 'traffic_labels_from_headers': False}, 'all_routed_rank64_lower_bound_ledger': {**lb64_d, 'scope': 'target_local', 'routed_rank': ROUTED_RANK_LOWER_BOUND, 'authorizing': False, 'is_conservative': False, 'is_lower_bound_only': True, 'description': 'Target-local basis identities with all routed experts at rank 64. Optimistic/lower-bound byte scenario only — not conservative and not authorization-deciding. No full-model traffic map exists; the sealed low-traffic diagnostic needed rank 128 to clear its 0.91 floor.', 'within_target_bpw': bool(lb64_bpw <= TARGET_BPW)}, 'all_routed_rank128_uncertainty_bound_ledger': {**ub128_d, 'scope': 'target_local', 'routed_rank': ROUTED_RANK_UNCERTAINTY_BOUND, 'authorizing': True, 'is_conservative': False, 'is_uncertainty_bound': True, 'description': 'Target-local basis identities with all routed experts at rank 128 while shared MLP, router, and q_a retain preregistered ranks. This is the only ledger that may decide top-level within_target_bpw. It is a byte-feasibility uncertainty bound, not proof that rank 128 is quality-sufficient for every routed expert. Ranks are never reduced to force this total under budget.', 'within_target_bpw': within, 'routed_ranks_present': sorted(routed_ranks_ub), 'all_basis_ranks_present': sorted(ranks_in_ub)}, 'route_population_sensitivity': sensitivity, 'transfer_sharing_scenario_ledger': {**xfer_d, 'scope': 'transfer_layer', 'authorizing': False, 'description': 'Non-authorizing scenario: one hidden basis and one SwiGLU basis per layer shared across experts. Cross-layer transfer remains unvalidated. Must not affect any top-level decision.', 'within_target_bpw_not_applicable': True, 'note': 'Do not use this total to authorize a traversal or claim capability.'}, 'within_target_bpw': within, 'full_route_population_classified': False, 'route_population_evidence_sufficient_for_rank_assignment': False, 'rank64_population_fit_is_lower_bound_only': True, 'full_traversal_authorized': False, 'unsupported_classes': unsupported, 'fake_codec_proof': fake, 'safety': dict(SAFETY_FENCES), 'remaining_uncertainties': ['Teacher capsules cover only a subset of layers; uncovered layers unvalidated.', 'No full-model traffic map: route population is not classified; rank assignment for the whole population is unproven.', 'Rank-64 whole-population fit is a lower bound only; sealed low-traffic diagnostics required rank 128 to clear the 0.91 per-tensor floor.', 'All-rank-128 uncertainty bound is a byte-feasibility envelope, not quality proof for every routed expert.', 'attention.o_proj lacks real intermediate in current capsules.', 'global embed_tokens and lm_head have no bounded real-input pilot.', 'Dense MLP layers 0-2 unvalidated under the MoE program.', 'Cross-layer basis transfer is unvalidated and non-authorizing.', 'Feasibility is not whole-model capability; Generation B proved relative admission fails.'], 'non_claims': ['Does not prove representation quality on uncovered layers, unmeasured route traffic, globals, or attention output projections.', 'Does not authorize full parent traversal, capability gate, HIDE kernel turn, Odyssey launch, Math-Frozen, or Ramanujan research.', 'Does not claim a capable artifact.', 'Does not treat a passing rank-mixture budget as proof of representation capability.', 'Does not call the rank-64 whole-population total conservative.'], 'next_safe_action': 'Route-population measurement is required before any full traversal. A passing rank-mixture budget alone would still not prove representation capability. Keep v2 opt-in; never lower absolute floors or ranks to force the uncertainty bound under target BPW. Do not start a full traversal from this receipt alone.'}
    receipt['receipt_sha256'] = sha256_json(receipt)
    return receipt

def feasibility_markdown(receipt: dict[str, Any]) -> str:
    lb64 = receipt['all_routed_rank64_lower_bound_ledger']
    ub128 = receipt['all_routed_rank128_uncertainty_bound_ledger']
    xfer = receipt['transfer_sharing_scenario_ledger']
    sens = receipt['route_population_sensitivity']
    rec = receipt['census']['reconcile']
    lines = ['# GLM-5.2 activation-aware pack v2 — program feasibility (revision 1)', '', 'Opt-in, source-body-free feasibility for the next representation program.', 'Corrects Generation B defects (relative admission, centered-mean loss,', 'output-side down, Gaussian proxies) without authorizing a traversal.', '', 'Revision 1: the rank-64 whole-population total is a **lower bound only** (not conservative). Top-level `within_target_bpw` is decided solely by the all-routed rank-128 **uncertainty bound** (byte feasibility, not quality proof).', '', '## Safety fences (all false)', '']
    for k, v in receipt['safety'].items():
        lines.append(f'- `{k}` = `{v}`')
    lines += ['', '## Route-population status', '', f'- `full_route_population_classified`: **{receipt['full_route_population_classified']}**', f'- `route_population_evidence_sufficient_for_rank_assignment`: **{receipt['route_population_evidence_sufficient_for_rank_assignment']}**', f'- `rank64_population_fit_is_lower_bound_only`: **{receipt['rank64_population_fit_is_lower_bound_only']}**', f'- `full_traversal_authorized`: **{receipt['full_traversal_authorized']}**', f'- Routed experts (static census): **{receipt['census'].get('n_routed_experts', 'n/a'):,}**', f'- Static routed classification: `{receipt['census'].get('static_routed_classification')}` (not traffic labels)', '', '## Census (sealed headers)', '', f'- Unique tensors: **{rec['unique_tensor_names']}** (expected {rec['expected_unique_tensor_names']}) {('OK' if rec['unique_tensor_names_ok'] else 'FAIL')}', f'- Original weights: **{rec['original_weights']:,}** {('OK' if rec['original_weights_ok'] else 'FAIL')}', f'- Source payload bytes: **{rec['source_payload_bytes']:,}** {('OK' if rec['source_payload_bytes_ok'] else 'FAIL')}', '', '## Preregistered candidate (not whole-model capability)', '', '| Program | Rank | Floors | Role |', '|---|---:|---|---|', '| Neutral routed gate/up/down (census) | 64 or 128 by scenario | pilot floors retained | ledger-scenario rank only |', '| High-traffic panel (pilot evidence) | 64 | panel min 0.85, median 0.96 | not whole-model traffic |', '| Low-traffic diagnostics (pilot) | 128 | per-tensor 0.91 | not population map |', '| Shared MLP gate/up/down | 256 | per-tensor 0.91, panel median 0.93 | preregistered |', '| Router control | 128 | per-tensor 0.99 | preregistered |', '| Attention `q_a_proj` only | 128 | per-tensor 0.91 | preregistered |', '| All other classes | native | source payload width | unvalidated |', '', '## All-routed rank-64 lower-bound ledger (NON-AUTHORIZING)', '', 'Not conservative. Optimistic whole-population scenario with every routed expert at rank 64.', '', f'- Unique bases: **{lb64['n_unique_bases']:,}**', f'- Total bytes: **{lb64['total_bytes']:,}**', f'- Complete BPW: **{lb64['complete_bpw_exact']}** ({lb64['complete_bpw_float']:.6f})', f'- `authorizing`: **{lb64['authorizing']}**', f'- Scenario within target (informational): **{lb64['within_target_bpw']}**', f'- Itemization reconciles: **{lb64['itemization_reconciles']}**', '', 'Component totals:', '']
    for k, v in lb64['component_totals'].items():
        lines.append(f'- `{k}`: {v:,}')
    lines += ['', '## All-routed rank-128 uncertainty-bound ledger (AUTHORIZING for BPW only)', '', 'Byte-feasibility uncertainty bound for the whole routed population at rank 128. **Not** proof that rank 128 is quality-sufficient for every expert. This total alone decides top-level `within_target_bpw`.', '', f'- Unique bases: **{ub128['n_unique_bases']:,}**', f'- Total bytes: **{ub128['total_bytes']:,}**', f'- Complete BPW: **{ub128['complete_bpw_exact']}** ({ub128['complete_bpw_float']:.6f})', f'- Target: **{receipt['target_bpw']}**', f'- `within_target_bpw` (top-level): **{receipt['within_target_bpw']}**', f'- `authorizing`: **{ub128['authorizing']}**', f'- Itemization reconciles: **{ub128['itemization_reconciles']}**', '', 'Component totals:', '']
    for k, v in ub128['component_totals'].items():
        lines.append(f'- `{k}`: {v:,}')
    lines += ['', '## Route-population sensitivity (arithmetic, not traffic)', '', f'- Selection: `{sens['selection_rule']}`', f'- Expert unit: {sens['expert_count_unit']}', f'- N routed experts: **{sens['n_routed_experts']:,}**', f'- Max rank-128 experts under target: **{sens['max_rank128_experts_under_target_bpw']:,}** ({sens['max_rank128_fraction_under_target_bpw_exact']})', '', '| Fraction @128 | N@128 | Total bytes | BPW | Within target |', '|---:|---:|---:|---:|:---:|']
    for p in sens['sweep']:
        lines.append(f'| {p['fraction_rank128']} | {p['n_rank128_experts']:,} | {p['total_bytes']:,} | {p['complete_bpw_float']:.6f} | {p['within_target_bpw']} |')
    lines += ['', '## Transfer-sharing scenario (NON-AUTHORIZING)', '', f'- Unique bases: **{xfer['n_unique_bases']:,}**', f'- Total bytes: **{xfer['total_bytes']:,}**', f'- Complete BPW (informational): **{xfer['complete_bpw_exact']}**', '- Cross-layer / cross-expert transfer remains **unvalidated**.', '- This total must **not** affect any top-level decision.', '', '## Scientific laws', '']
    for k, v in receipt['scientific_laws'].items():
        lines.append(f'- `{k}`: `{v}`')
    lines += ['', '## Pilot checks (sealed receipt)', '']
    pc = receipt.get('pilot_checks') or {}
    for c in pc.get('checks', []):
        lines.append(f'- `{c.get('program')}` rank {c.get('rank')}: clears={c.get('clears')} detail={{{', '.join((f'{k}={v}' for k, v in c.items() if k not in ('program', 'rank')))}}}')
    lines += ['', '## Unsupported / native islands', '']
    for u in receipt['unsupported_classes']:
        if u['n_tensors']:
            lines.append(f'- `{u['organ_class']}`: n={u['n_tensors']}, bytes={u['source_payload_bytes']:,} — {u['why']}')
    lines += ['', '## Remaining uncertainties', '']
    for u in receipt['remaining_uncertainties']:
        lines.append(f'- {u}')
    lines += ['', '## Non-claims', '']
    for u in receipt['non_claims']:
        lines.append(f'- {u}')
    lines += ['', '## Next safe action', '', receipt['next_safe_action'], '', f'Receipt sha256: `{receipt.get('receipt_sha256', '')}`', '']
    return '\n'.join(lines)

def write_feasibility(*, out_json: Path=DEFAULT_FEASIBILITY_JSON, out_md: Path=DEFAULT_FEASIBILITY_MD) -> dict[str, Any]:
    receipt = build_feasibility_receipt()
    out_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    out_md.write_text(feasibility_markdown(receipt), encoding='utf-8')
    return receipt

def selftest() -> int:
    rng = np.random.default_rng(SEED)
    h, n, r = (48, 200, 8)
    mean = rng.standard_normal(h).astype(np.float32)
    mean /= np.linalg.norm(mean) + 1e-12
    X = (0.2 * rng.standard_normal((n, h)) + 3.0 * mean).astype(np.float32)
    bu = build_uncentered_basis(X, r)
    bc = build_centered_basis_diagnostic(X, r)
    assert mean_direction_retained(bu, atol=0.9)
    align_u = abs(float(np.dot(bu.columns(1)[:, 0], mean)))
    align_c = abs(float(np.dot(bc[:, 0], mean)))
    assert align_u > align_c
    topk = np.array([[1, 2], [3, 4], [1, 5], [6, 7]], dtype=np.int32)
    assert route_row_indices(topk, 1).tolist() == [0, 2]
    Xr = rng.standard_normal((4, 8)).astype(np.float32)
    try:
        select_route_rows(Xr, topk, 99, min_rows=1)
        raise AssertionError('empty route must fail')
    except RouteUndersampledError:
        pass
    try:
        select_route_rows(Xr, topk, 1, min_rows=10)
        raise AssertionError('undersampled route must fail')
    except RouteUndersampledError:
        pass
    got = select_route_rows(Xr, topk, 1, min_rows=2)
    assert got.shape == (2, 8)
    Xh = rng.standard_normal((16, 32)).astype(np.float32)
    Wg = rng.standard_normal((8, 32)).astype(np.float32)
    Wu = rng.standard_normal((8, 32)).astype(np.float32)
    Z = swiglu_intermediate(Xh, Wg, Wu)
    Z_ref = silu(Xh @ Wg.T) * (Xh @ Wu.T)
    assert np.allclose(Z, Z_ref, atol=1e-05)
    floors = FloorSpec(per_tensor_min=0.9)
    res = check_absolute_floors([0.5], floors, beats_null_flags=[True])
    assert res['ok'] is False
    sel = select_program_or_native(cosine=0.5, beats_null=True, floor=0.9, source_payload_bytes=1000, encoded_bytes=100, byte_budget_remaining=10000)
    assert sel['disposition'] == 'native'
    assert sel['beats_null_overrode_floor'] is False
    try:
        select_program_or_native(cosine=0.95, beats_null=True, floor=0.9, source_payload_bytes=1000, encoded_bytes=500, byte_budget_remaining=100)
        raise AssertionError('budget must fail closed')
    except BudgetFailure:
        pass
    proof = fake_gate_up_down_roundtrip(seed=SEED)
    assert proof['ok'] and proof['gate_up_share'] and proof['experts_non_aliasing']
    assert proof['down_basis_from_real_swiglu'] is True
    assert proof['hidden_basis_from_route_rows'] is True
    assert proof['witnesses']['positive_verify_ok'] is True
    assert proof['witnesses']['unrelated_basis_fails_witness'] is True
    led = BasisLedger()
    bid = basis_identity(kind='uncentered_hidden', layer=5, expert_id=11, rank=64)
    b1 = led.add_basis(bid, width=HIDDEN, rank=64, kind='uncentered_hidden')
    b2 = led.add_basis(bid, width=HIDDEN, rank=64, kind='uncentered_hidden')
    assert b1 > 0 and b2 == 0
    assert led.bases[bid]['refcount'] == 2
    led.add_coefficients(2048, 64)
    led.add_coefficients(2048, 64)
    assert led.reconciles()
    assert_no_gaussian_promotion_path()
    assert all((v is False for v in SAFETY_FENCES.values()))
    if SOURCE_HEADERS.exists():
        receipt = build_feasibility_receipt()
        assert receipt['census']['reconcile']['unique_tensor_names_ok']
        assert receipt['all_routed_rank64_lower_bound_ledger']['itemization_reconciles']
        assert receipt['all_routed_rank128_uncertainty_bound_ledger']['itemization_reconciles']
        assert receipt['safety']['full_traversal_authorized'] is False
        assert receipt['safety']['capable_artifact_claimed'] is False
        assert receipt['transfer_sharing_scenario_ledger']['authorizing'] is False
        assert receipt['all_routed_rank64_lower_bound_ledger']['authorizing'] is False
        assert receipt['all_routed_rank128_uncertainty_bound_ledger']['authorizing'] is True
        assert receipt['within_target_bpw'] == receipt['all_routed_rank128_uncertainty_bound_ledger']['within_target_bpw']
        assert receipt['full_route_population_classified'] is False
        assert receipt['rank64_population_fit_is_lower_bound_only'] is True
        assert receipt['full_traversal_authorized'] is False
        r2 = build_feasibility_receipt()
        assert r2['receipt_sha256'] == receipt['receipt_sha256']
    print('glm52_activation_aware_pack_v2 selftest OK')
    return 0
