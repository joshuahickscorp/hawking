#!/usr/bin/env python3
"""MLX routed direct consumer for a shared-core/local-residual Nova organ.

The consumer accepts the two ``SwitchLinear`` routing layouts used by MLX-LM:

* sorted ``x=(N,1,D)``, ``indices=(N,)``;
* unsorted ``x=(B,T,1,D)``, ``indices=(B,T,K)``.

It returns ``indices.shape + (1, O)`` and evaluates
``U(C_e(V^T x)) + A_e(B_e^T x)`` directly.  The dense reconstruction helper is
test/oracle-only and is intentionally absent from the consumer call path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _bf16_bits_to_mlx(array: np.ndarray):
    import torch
    import mlx.core as mx

    if array.dtype != np.uint16:
        raise ValueError(f"expected uint16 BF16 bit pattern, got {array.dtype}")
    float32 = torch.from_numpy(np.array(array, copy=True)).view(torch.bfloat16).float().numpy()
    return mx.array(float32).astype(mx.bfloat16)


def _int8_rows_to_mlx(array: np.ndarray):
    import mlx.core as mx

    if array.dtype != np.int8:
        raise ValueError(f"expected signed INT8 payload, got {array.dtype}")
    return mx.array(np.array(array, copy=True)).astype(mx.bfloat16)


def _fp32_scales_to_mlx(array: np.ndarray):
    import mlx.core as mx

    if array.dtype != np.float32:
        raise ValueError(f"expected FP32 row scales, got {array.dtype}")
    return mx.array(np.array(array, copy=True)).astype(mx.bfloat16)


class NovaLocalResidualRoutedLinear:
    """Direct MLX factor consumer with SwitchLinear-compatible routing shapes."""

    def __init__(self, u, v, cores, residual_a, residual_b):
        import mlx.core as mx

        self.u = u
        self.v = v
        self.cores = cores
        self.residual_a = residual_a
        self.residual_b = residual_b
        self._expert_count = int(cores.shape[0])
        self._output_dim = int(u.shape[0])
        self._input_dim = int(v.shape[0])
        if tuple(v.shape) != (self._input_dim, int(v.shape[1])):
            raise ValueError("invalid shared input basis shape")
        if tuple(cores.shape[1:]) != (int(u.shape[1]), int(v.shape[1])):
            raise ValueError(
                f"core shape {cores.shape} incompatible with U/V {u.shape}/{v.shape}")
        if tuple(residual_a.shape[:2]) != (self._expert_count, self._output_dim):
            raise ValueError("residual A shape is incompatible with expert/output dimensions")
        if tuple(residual_b.shape[:2]) != (self._expert_count, self._input_dim):
            raise ValueError("residual B shape is incompatible with expert/input dimensions")
        if residual_a.shape[2] != residual_b.shape[2]:
            raise ValueError("residual rank mismatch")
        self._mx = mx

    @classmethod
    def from_bf16_artifact(cls, path: str | Path, expert_count: int):
        import mlx.core as mx

        with np.load(path, allow_pickle=False) as payload:
            u = _bf16_bits_to_mlx(payload["u"])
            v = _bf16_bits_to_mlx(payload["v"])
            cores = mx.stack([_bf16_bits_to_mlx(payload[f"core_{i}"])
                              for i in range(expert_count)], axis=0)
            aa = mx.stack([_bf16_bits_to_mlx(payload[f"residual_a_{i}"])
                           for i in range(expert_count)], axis=0)
            bb = mx.stack([_bf16_bits_to_mlx(payload[f"residual_b_{i}"])
                           for i in range(expert_count)], axis=0)
        return cls(u, v, cores, aa, bb)

    def __call__(self, x, indices, sorted_indices: bool = False):
        import mlx.core as mx

        if x.shape[-1] != self._input_dim:
            raise ValueError(f"input width {x.shape[-1]} != {self._input_dim}")
        if x.shape[-2] != 1:
            raise ValueError("Nova routed consumer currently requires matrix dimension M=1")
        ids = indices.reshape(-1)
        rows = x.reshape(-1, self._input_dim)
        if ids.size != rows.shape[0]:
            if rows.shape[0] == 0 or ids.size % rows.shape[0] != 0:
                raise ValueError(f"routing shape mismatch: x={x.shape}, indices={indices.shape}")
            rows = mx.repeat(rows, ids.size // rows.shape[0], axis=0)
        if bool(mx.any(ids >= self._expert_count).item()):
            raise ValueError("routing index exceeds factor expert count")

        # Gather only the compact factor cores for the selected experts.  No
        # W_e = U C_e V^T + A_e B_e^T tensor is formed anywhere in this path.
        core_rows = mx.take(self.cores, ids, axis=0)
        a_rows = mx.take(self.residual_a, ids, axis=0)
        b_rows = mx.take(self.residual_b, ids, axis=0)
        z = rows @ self.v
        hidden = mx.einsum("nc,nrc->nr", z, core_rows)
        out = hidden @ self.u.T
        residual_hidden = mx.einsum("ni,nir->nr", rows, b_rows)
        out = out + mx.einsum("nr,nor->no", residual_hidden, a_rows)
        mx.eval(out)
        return out.reshape(tuple(indices.shape) + (1, self._output_dim))


class ExpertLowRankRoutedLinear:
    """Direct routed consumer for independent ``A_e B_e^T`` factors."""

    def __init__(self, factor_a, factor_b):
        import mlx.core as mx

        self.factor_a = factor_a
        self.factor_b = factor_b
        self._expert_count = int(factor_a.shape[0])
        self._output_dim = int(factor_a.shape[1])
        self._input_dim = int(factor_b.shape[1])
        if tuple(factor_a.shape[:2]) != (self._expert_count, self._output_dim):
            raise ValueError("invalid independent output-factor shape")
        if tuple(factor_b.shape[:2]) != (self._expert_count, self._input_dim):
            raise ValueError("invalid independent input-factor shape")
        if factor_a.shape[2] != factor_b.shape[2]:
            raise ValueError("independent factor rank mismatch")
        self._mx = mx

    @classmethod
    def from_bf16_artifact(cls, path: str | Path, expert_count: int):
        import mlx.core as mx

        with np.load(path, allow_pickle=False) as payload:
            factor_a = mx.stack([
                _bf16_bits_to_mlx(payload[f"a_{i}"])
                for i in range(expert_count)
            ], axis=0)
            factor_b = mx.stack([
                _bf16_bits_to_mlx(payload[f"b_{i}"])
                for i in range(expert_count)
            ], axis=0)
        return cls(factor_a, factor_b)

    def __call__(self, x, indices, sorted_indices: bool = False):
        import mlx.core as mx

        if x.shape[-1] != self._input_dim:
            raise ValueError(f"input width {x.shape[-1]} != {self._input_dim}")
        if x.shape[-2] != 1:
            raise ValueError("independent routed consumer requires matrix dimension M=1")
        ids = indices.reshape(-1)
        rows = x.reshape(-1, self._input_dim)
        if ids.size != rows.shape[0]:
            if rows.shape[0] == 0 or ids.size % rows.shape[0] != 0:
                raise ValueError(f"routing shape mismatch: x={x.shape}, indices={indices.shape}")
            rows = mx.repeat(rows, ids.size // rows.shape[0], axis=0)
        if bool(mx.any(ids >= self._expert_count).item()):
            raise ValueError("routing index exceeds factor expert count")
        a_rows = mx.take(self.factor_a, ids, axis=0)
        b_rows = mx.take(self.factor_b, ids, axis=0)
        hidden = mx.einsum("ni,nir->nr", rows, b_rows)
        out = mx.einsum("nr,nor->no", hidden, a_rows)
        mx.eval(out)
        return out.reshape(tuple(indices.shape) + (1, self._output_dim))


class Int8ExpertLowRankRoutedLinear:
    """Direct routed consumer for row-wise-INT8 ``A_e B_e^T`` factors."""

    def __init__(self, factor_a_q, factor_a_scale, factor_b_q, factor_b_scale):
        import mlx.core as mx

        self.factor_a_q = factor_a_q
        self.factor_a_scale = factor_a_scale
        self.factor_b_q = factor_b_q
        self.factor_b_scale = factor_b_scale
        self._expert_count = int(factor_a_q.shape[0])
        self._output_dim = int(factor_a_q.shape[1])
        self._input_dim = int(factor_b_q.shape[1])
        if tuple(factor_a_q.shape[:2]) != (self._expert_count, self._output_dim):
            raise ValueError("invalid INT8 independent output-factor shape")
        if tuple(factor_b_q.shape[:2]) != (self._expert_count, self._input_dim):
            raise ValueError("invalid INT8 independent input-factor shape")
        if factor_a_q.shape[2] != factor_b_q.shape[2]:
            raise ValueError("INT8 independent factor rank mismatch")
        if tuple(factor_a_scale.shape) != (self._expert_count, self._output_dim):
            raise ValueError("invalid output row-scale shape")
        if tuple(factor_b_scale.shape) != (self._expert_count, self._input_dim):
            raise ValueError("invalid input row-scale shape")
        self._mx = mx

    @classmethod
    def from_int8_artifact(cls, path: str | Path, expert_count: int):
        import mlx.core as mx

        with np.load(path, allow_pickle=False) as payload:
            aq = mx.stack([_int8_rows_to_mlx(payload[f"a_q_{i}"])
                           for i in range(expert_count)], axis=0)
            ass = mx.stack([_fp32_scales_to_mlx(payload[f"a_scale_{i}"])
                            for i in range(expert_count)], axis=0)
            bq = mx.stack([_int8_rows_to_mlx(payload[f"b_q_{i}"])
                           for i in range(expert_count)], axis=0)
            bss = mx.stack([_fp32_scales_to_mlx(payload[f"b_scale_{i}"])
                            for i in range(expert_count)], axis=0)
        return cls(aq, ass, bq, bss)

    def dequantized_factors(self):
        """Oracle-only dequantization; the consumer call path stays factored."""
        return (
            self.factor_a_q * self.factor_a_scale[..., None],
            self.factor_b_q * self.factor_b_scale[..., None],
        )

    def __call__(self, x, indices, sorted_indices: bool = False):
        import mlx.core as mx

        if x.shape[-1] != self._input_dim:
            raise ValueError(f"input width {x.shape[-1]} != {self._input_dim}")
        if x.shape[-2] != 1:
            raise ValueError("INT8 independent routed consumer requires M=1")
        ids = indices.reshape(-1)
        rows = x.reshape(-1, self._input_dim)
        if ids.size != rows.shape[0]:
            if rows.shape[0] == 0 or ids.size % rows.shape[0] != 0:
                raise ValueError(f"routing shape mismatch: x={x.shape}, indices={indices.shape}")
            rows = mx.repeat(rows, ids.size // rows.shape[0], axis=0)
        if bool(mx.any(ids >= self._expert_count).item()):
            raise ValueError("routing index exceeds INT8 factor expert count")
        aq = mx.take(self.factor_a_q, ids, axis=0)
        ass = mx.take(self.factor_a_scale, ids, axis=0)
        bq = mx.take(self.factor_b_q, ids, axis=0)
        bss = mx.take(self.factor_b_scale, ids, axis=0)
        # Use MLX's fused routed matmul twice.  B's row scale belongs to the
        # input coordinate, so it is applied to the selected activation rows;
        # A's row scale belongs to the output coordinate, so it is applied
        # after the second matmul.  No dense W_e parent is formed.
        scaled_rows = rows * bss
        hidden = mx.gather_mm(
            scaled_rows[:, None, :], self.factor_b_q,
            rhs_indices=ids, sorted_indices=sorted_indices,
        ).reshape(-1, self.factor_b_q.shape[-1])
        out = mx.gather_mm(
            hidden[:, None, :], self.factor_a_q.swapaxes(-1, -2),
            rhs_indices=ids, sorted_indices=sorted_indices,
        ).reshape(-1, self._output_dim)
        out = out * ass
        mx.eval(out)
        return out.reshape(tuple(indices.shape) + (1, self._output_dim))


class VariableExpertLowRankRoutedLinear:
    """Direct routed consumer for per-expert, variable-rank ``A_e B_e^T``.

    Factors are grouped by rank.  The implementation keeps the compact
    variable-rank storage and selects the matching factor group for each
    routed row; it does not pad the persistent representation to the maximum
    rank.  Group masks are intentionally explicit so the experimental kernel
    exposes its extra dispatch work in timing rather than hiding it.
    """

    def __init__(self, factor_groups, expert_ranks):
        import mlx.core as mx

        if not expert_ranks or len(factor_groups) == 0:
            raise ValueError("variable routed consumer needs experts and factor groups")
        self.factor_groups = factor_groups
        self.expert_ranks = tuple(int(rank) for rank in expert_ranks)
        self._expert_count = len(self.expert_ranks)
        first_a, first_b = next(iter(factor_groups.values()))
        self._output_dim = int(first_a.shape[1])
        self._input_dim = int(first_b.shape[1])
        self._group_maps = {}
        for rank, (a, b) in factor_groups.items():
            if tuple(a.shape[1:]) != (self._output_dim, int(rank)):
                raise ValueError(f"rank {rank} output factors have shape {a.shape}")
            if tuple(b.shape[1:]) != (self._input_dim, int(rank)):
                raise ValueError(f"rank {rank} input factors have shape {b.shape}")
            members = [i for i, expert_rank in enumerate(self.expert_ranks)
                       if expert_rank == int(rank)]
            if len(members) != int(a.shape[0]):
                raise ValueError(f"rank {rank} has {a.shape[0]} factors for {len(members)} experts")
            mapping = np.zeros(self._expert_count, dtype=np.int32)
            for local, expert in enumerate(members):
                mapping[expert] = local
            self._group_maps[int(rank)] = (mx.array(mapping), tuple(members))
        self._mx = mx

    @classmethod
    def from_bf16_artifact(cls, path: str | Path, expert_ranks):
        import mlx.core as mx

        ranks = tuple(int(rank) for rank in expert_ranks)
        groups = {}
        with np.load(path, allow_pickle=False) as payload:
            for rank in sorted(set(ranks)):
                members = [i for i, expert_rank in enumerate(ranks) if expert_rank == rank]
                a = mx.stack([_bf16_bits_to_mlx(payload[f"a_{i}"]) for i in members], axis=0)
                b = mx.stack([_bf16_bits_to_mlx(payload[f"b_{i}"]) for i in members], axis=0)
                groups[rank] = (a, b)
        return cls(groups, ranks)

    def __call__(self, x, indices, sorted_indices: bool = False):
        import mlx.core as mx

        if x.shape[-1] != self._input_dim:
            raise ValueError(f"input width {x.shape[-1]} != {self._input_dim}")
        if x.shape[-2] != 1:
            raise ValueError("variable routed consumer requires matrix dimension M=1")
        ids = indices.reshape(-1)
        rows = x.reshape(-1, self._input_dim)
        if ids.size != rows.shape[0]:
            if rows.shape[0] == 0 or ids.size % rows.shape[0] != 0:
                raise ValueError(f"routing shape mismatch: x={x.shape}, indices={indices.shape}")
            rows = mx.repeat(rows, ids.size // rows.shape[0], axis=0)
        if bool(mx.any(ids >= self._expert_count).item()):
            raise ValueError("routing index exceeds variable factor expert count")
        out = mx.zeros((rows.shape[0], self._output_dim), dtype=rows.dtype)
        for rank, (factor_a, factor_b) in self.factor_groups.items():
            mapping, members = self._group_maps[rank]
            group_mask = mx.zeros(ids.shape, dtype=mx.bool_)
            for expert in members:
                group_mask = mx.logical_or(group_mask, ids == expert)
            group_count = int(mx.sum(group_mask).item())
            if group_count == 0:
                continue
            # Compact selected rows before matmul.  This avoids evaluating
            # every rank group over the complete routed batch, which would
            # erase the byte benefit with dispatch-only arithmetic.
            order = mx.argsort(mx.where(group_mask, 0, 1))[:group_count]
            selected_ids = mx.take(ids, order, axis=0)
            selected_rows = mx.take(rows, order, axis=0)
            local_ids = mx.take(mapping, selected_ids, axis=0)
            a_rows = mx.take(factor_a, local_ids, axis=0)
            b_rows = mx.take(factor_b, local_ids, axis=0)
            hidden = mx.einsum("ni,nir->nr", selected_rows, b_rows)
            group_out = mx.einsum("nr,nor->no", hidden, a_rows)
            out = mx.put_along_axis(out, order[:, None], group_out, axis=0)
        mx.eval(out)
        return out.reshape(tuple(indices.shape) + (1, self._output_dim))


def dense_oracle_from_factors(u, v, cores, residual_a, residual_b):
    """Test/oracle-only reconstruction; never called by the consumer."""
    import mlx.core as mx

    return mx.einsum("or,erc,ic->eoi", u, cores, v) + mx.einsum(
        "eor,eir->eoi", residual_a, residual_b)


def direct_vs_gather_mm(x, indices, factors) -> dict[str, Any]:
    """Compare the direct consumer with MLX's dense gather oracle."""
    import mlx.core as mx

    u, v, cores, aa, bb = factors
    dense = dense_oracle_from_factors(u, v, cores, aa, bb)
    direct = NovaLocalResidualRoutedLinear(u, v, cores, aa, bb)(x, indices)
    # SwitchGLU presents unsorted top-k routing as x=(B,T,1,D) and
    # indices=(B,T,K), while gather_mm consumes one routed row per id.
    # Canonicalize that public shape so the comparison exercises the same
    # row replication and output shape as the production SwitchGLU path.
    base_rows = x.reshape(-1, 1, x.shape[-1])
    flat_indices = indices.reshape(-1)
    if flat_indices.size != base_rows.shape[0]:
        if flat_indices.size % base_rows.shape[0] != 0:
            raise ValueError(
                "routing indices must be one id per row or an integer top-k multiple"
            )
        top_k = flat_indices.size // base_rows.shape[0]
        base_rows = mx.repeat(base_rows, top_k, axis=0)
    oracle_flat = mx.gather_mm(
        base_rows, dense.swapaxes(-1, -2), rhs_indices=flat_indices
    )
    oracle = oracle_flat.reshape(tuple(indices.shape) + (1, dense.shape[-2]))
    mx.eval(direct, oracle)
    d = np.asarray(direct.astype(mx.float32))
    o = np.asarray(oracle.astype(mx.float32))
    return {
        "direct_shape": list(direct.shape),
        "oracle_shape": list(oracle.shape),
        "max_abs_error": float(np.max(np.abs(d - o))),
        "mean_abs_error": float(np.mean(np.abs(d - o))),
        "exact_shape": tuple(direct.shape) == tuple(oracle.shape),
        "dense_oracle_used": True,
    }
