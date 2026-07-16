"""CPU correctness gates for RWKV-7 recurrence and batched training.

The chunked-scan gate asserts, for random RWKV-7-shaped inputs (including T not
divisible by chunk_size), that
``wkv7_chunked`` matches the sequential reference to:
    forward   max abs diff <= 1e-4
    gradients max abs diff <= 1e-3  (autograd through both, all 6 inputs)

It also checks that the extracted reference (:func:`wkv7_sequential_ref`)
reproduces the *actual* ``RWKV7TimeMix`` recurrence loop bit-for-bit (so the
reference is a faithful stand-in for the validated model), and that right-padded
batched forwards preserve every real-position logit.

CPU-only by construction (the GPU is reserved for a training run).
"""

from __future__ import annotations

import os
import sys

import torch

try:
    import pytest
except ImportError:  # CPU research venv may not have pytest; we provide a runner.
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rwkv7_chunked import wkv7_chunked, wkv7_sequential_ref  # noqa: E402
from rwkv7_torch_model import RWKV7Config, RWKV7Model  # noqa: E402


def _parametrize(argnames, argvalues):
    """No-op stand-in for pytest.mark.parametrize when pytest is absent.

    The manual runner in ``__main__`` drives the parametrization itself, so the
    decorator only needs to not crash at import time.
    """
    if pytest is not None:
        return pytest.mark.parametrize(argnames, argvalues)

    def deco(fn):
        return fn

    return deco

FWD_TOL = 1e-4
GRAD_TOL = 1e-3

# RWKV-7 0.4B head geometry: 16 heads x 64 head_dim.
SHAPES = [
    # (B, T, H, D, chunk_size)
    (2, 37, 16, 64, 8),    # T not divisible by chunk_size (primary case from spec)
    (2, 32, 16, 64, 16),   # T divisible by chunk_size
    (2, 64, 16, 64, 16),   # multiple full chunks
    (1, 1, 16, 64, 16),    # single token (degenerate / decode-shaped)
    (2, 13, 4, 8, 8),      # small head geometry, T < chunk_size in last chunk
    (3, 100, 8, 32, 32),   # larger chunk, T not divisible (100 = 3*32 + 4)
]


def _make_inputs(B, T, H, D, *, requires_grad=False, seed=0, device="cpu"):
    """Build (r, w, k, v, a_op, b_op) with realistic RWKV-7 statistics:
    w (decay) in (0.5, 1) per the spec; a_op = -kk (kk l2-normed per head);
    b_op = kk * a with a = sigmoid(.) in (0, 1)."""
    g = torch.Generator(device=device).manual_seed(seed)
    f = torch.float32

    def rn(*shape):
        return torch.randn(*shape, generator=g, dtype=f, device=device)

    r = rn(B, T, H, D)
    k = rn(B, T, H, D)
    v = rn(B, T, H, D)

    # kk: l2-normalized per head (as in the time-mix), then a_op = -kk.
    kk = rn(B, T, H, D)
    kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a = torch.sigmoid(rn(B, T, H, D))  # in (0, 1)
    a_op = -kk
    b_op = kk * a

    # w (multiplicative decay) in (0.5, 1): exp(-0.606531 * sigmoid(.)) lands in
    # (exp(-0.606531), 1) = (~0.545, 1), matching the time-mix decay nonlinearity.
    w = torch.exp(-0.606531 * torch.sigmoid(rn(B, T, H, D)))

    tensors = [r, w, k, v, a_op, b_op]
    if requires_grad:
        tensors = [t.detach().clone().requires_grad_(True) for t in tensors]
    return tensors


@_parametrize("B,T,H,D,chunk_size", SHAPES)
def test_forward_parity(B, T, H, D, chunk_size):
    r, w, k, v, a_op, b_op = _make_inputs(B, T, H, D, seed=B * 1000 + T)
    out_ref = wkv7_sequential_ref(r, w, k, v, a_op, b_op)
    out_chunk = wkv7_chunked(r, w, k, v, a_op, b_op, chunk_size=chunk_size)
    assert out_chunk.shape == out_ref.shape
    max_abs = (out_chunk - out_ref).abs().max().item()
    assert max_abs <= FWD_TOL, (
        f"forward parity failed for {(B,T,H,D,chunk_size)}: max abs diff "
        f"{max_abs:.3e} > {FWD_TOL:.0e}"
    )


@_parametrize("B,T,H,D,chunk_size", SHAPES)
def test_grad_parity(B, T, H, D, chunk_size):
    names = ["r", "w", "k", "v", "a_op", "b_op"]

    # Shared upstream gradient so the two backward passes see identical d(out).
    g = torch.Generator().manual_seed(7)
    seed = B * 1000 + T + 1

    ins_ref = _make_inputs(B, T, H, D, requires_grad=True, seed=seed)
    out_ref = wkv7_sequential_ref(*ins_ref)
    gout = torch.randn(out_ref.shape, generator=g, dtype=out_ref.dtype)
    out_ref.backward(gout)
    grads_ref = [t.grad.detach().clone() for t in ins_ref]

    ins_chunk = _make_inputs(B, T, H, D, requires_grad=True, seed=seed)
    out_chunk = wkv7_chunked(*ins_chunk, chunk_size=chunk_size)
    out_chunk.backward(gout)
    grads_chunk = [t.grad.detach().clone() for t in ins_chunk]

    for name, gr, gc in zip(names, grads_ref, grads_chunk):
        max_abs = (gr - gc).abs().max().item()
        assert max_abs <= GRAD_TOL, (
            f"grad parity failed for d/d{name} {(B,T,H,D,chunk_size)}: "
            f"max abs diff {max_abs:.3e} > {GRAD_TOL:.0e}"
        )


def test_reference_matches_timemix_loop():
    """The extracted reference must reproduce RWKV7TimeMix's recurrence loop
    exactly (it is the same code path, so this should be bit-identical)."""
    B, T, H, D = 2, 19, 16, 64
    r, w, k, v, a_op, b_op = _make_inputs(B, T, H, D, seed=123)

    # Re-run the *literal* loop from RWKV7TimeMix.forward over per-head tensors.
    rh, kh, vh = r, k, v
    wh, ah, bh = w, a_op, b_op
    S = torch.zeros(B, H, D, D, dtype=r.dtype)
    out = torch.empty(B, T, H, D, dtype=r.dtype)
    for t in range(T):
        w_t, k_t, v_t = wh[:, t], kh[:, t], vh[:, t]
        a_t, b_t, r_t = ah[:, t], bh[:, t], rh[:, t]
        sa = torch.einsum("bhij,bhj->bhi", S, a_t)
        S = (
            S * w_t.unsqueeze(2)
            + v_t.unsqueeze(3) * k_t.unsqueeze(2)
            + sa.unsqueeze(3) * b_t.unsqueeze(2)
        )
        out[:, t] = torch.einsum("bhij,bhj->bhi", S, r_t)

    out_ref = wkv7_sequential_ref(r, w, k, v, a_op, b_op)
    assert torch.equal(out, out_ref)


BATCH_SEQS = [
    [3, 7, 1, 9, 200, 13, 42],
    [5, 5, 5, 1, 2, 3, 4, 8, 99, 17, 6],
    [10, 20, 30, 40, 50],
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
]


def _batch_cfg(use_chunked: bool, chunk_size: int) -> RWKV7Config:
    return RWKV7Config(
        n_embd=128,
        n_layer=3,
        n_ff=256,
        head_dim=64,
        n_head=2,
        vocab_size=256,
        use_chunked=use_chunked,
        chunk_size=chunk_size,
    )


def _individual_logits(model, ids):
    hidden = model(torch.tensor([ids], dtype=torch.long), return_final_hidden=True)
    return model.lm_head(hidden[0]).float()


def _batched_logits(model, sequences, pad_id=0):
    width = max(len(sequence) for sequence in sequences)
    inputs = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        inputs[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    hidden = model(inputs, return_final_hidden=True)
    return [
        model.lm_head(hidden[index, : len(sequence)]).float()
        for index, sequence in enumerate(sequences)
    ]


def _batch_case(use_chunked: bool, chunk_size: int, tolerance: float = 2e-4) -> float:
    torch.manual_seed(0)
    model = RWKV7Model(_batch_cfg(use_chunked, chunk_size))
    model.eval()
    with torch.no_grad():
        individual = [_individual_logits(model, sequence) for sequence in BATCH_SEQS]
        batched = _batched_logits(model, BATCH_SEQS)
    worst = max((left - right).abs().max().item() for left, right in zip(individual, batched))
    assert worst <= tolerance, (
        f"batched parity failed for chunked={use_chunked} chunk_size={chunk_size}: "
        f"{worst:.3e} > {tolerance:.0e}"
    )
    return worst


def test_batch_equivalence():
    for use_chunked, chunk_size in ((False, 32), (True, 4), (True, 32)):
        _batch_case(use_chunked, chunk_size)


def _run_manually() -> int:
    """pytest-free runner: drive the parametrized tests and report max diffs."""
    failures = 0
    print("== forward parity ==")
    for shp in SHAPES:
        try:
            test_forward_parity(*shp)
            B, T, H, D, c = shp
            r, w, k, v, a_op, b_op = _make_inputs(B, T, H, D, seed=B * 1000 + T)
            md = (wkv7_chunked(r, w, k, v, a_op, b_op, chunk_size=c)
                  - wkv7_sequential_ref(r, w, k, v, a_op, b_op)).abs().max().item()
            print(f"  PASS {shp}  max_abs={md:.3e}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {shp}: {e}")
        except NotImplementedError as e:
            failures += 1
            print(f"  FAIL {shp}: NotImplementedError({e})")

    print("== gradient parity ==")
    for shp in SHAPES:
        try:
            test_grad_parity(*shp)
            print(f"  PASS {shp}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {shp}: {e}")
        except NotImplementedError as e:
            failures += 1
            print(f"  FAIL {shp}: NotImplementedError({e})")

    print("== reference matches RWKV7TimeMix loop ==")
    try:
        test_reference_matches_timemix_loop()
        print("  PASS")
    except AssertionError as e:
        failures += 1
        print(f"  FAIL: {e}")

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


def _run_batch_manually() -> int:
    failures = 0
    print("== batched-vs-individual forward parity ==")
    for use_chunked, chunk_size in ((False, 32), (True, 4), (True, 32)):
        try:
            worst = _batch_case(use_chunked, chunk_size)
            print(
                f"  PASS chunked={use_chunked} chunk_size={chunk_size} "
                f"max_abs={worst:.3e}"
            )
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    if pytest is not None:
        raise SystemExit(pytest.main([__file__, "-v"]))
    raise SystemExit(max(_run_manually(), _run_batch_manually()))
