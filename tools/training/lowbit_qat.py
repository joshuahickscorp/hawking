"""lowbit_qat.py — Shared QAT (quantization-aware training) library for RWKV-7 low-bit quantization.

Pure PyTorch. No Hugging Face imports.
"""

import re
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "quant_binary",
    "quant_ternary",
    "quant_uniform_symmetric",
    "QuantLinear",
    "wrap_linears",
    "list_wrapped_modules",
    "freeze_by_module_policy",
    "RWKV7_TIME_SUFFIXES",
    "RWKV7_FFN_SUFFIXES",
    "RWKV7_ALL_PROJ_SUFFIXES",
    "QUANT_FNS",
]

# ---------------------------------------------------------------------------
# Quantizers
# ---------------------------------------------------------------------------

def quant_binary(w: torch.Tensor, bits=None) -> torch.Tensor:
    """Binary STE: per-row scale = mean(|w|), alphabet {-s, +s}."""
    s = w.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    wq = q * s
    return w + (wq - w).detach()


def quant_ternary(w: torch.Tensor, bits=None) -> torch.Tensor:
    """Ternary STE: per-row scale = mean(|w|), alphabet {-s, 0, +s}."""
    s = w.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.clamp(torch.round(w / s), -1, 1)
    wq = q * s
    return w + (wq - w).detach()


def quant_uniform_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric uniform STE. bits must be >= 2."""
    assert bits >= 2, "Use quant_binary for 1-bit"
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    s = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(w / s), qmin, qmax)
    wq = q * s
    return w + (wq - w).detach()


# ---------------------------------------------------------------------------
# QuantLinear
# ---------------------------------------------------------------------------

class QuantLinear(nn.Module):
    """STE wrapper around a linear layer.

    The shadow weight stays fp32; forward fake-quantizes via quant_fn.
    When quant_fn is None (strand-pv mode), the reconstruction stored in the
    ``base`` buffer is combined with the residual weight.
    """

    def __init__(self, lin: nn.Linear, quant_fn: Optional[Callable], bits: Optional[int]):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.data.clone().float())
        self.bias = (
            nn.Parameter(lin.bias.data.clone().float())
            if lin.bias is not None
            else None
        )
        self.quant_fn = quant_fn
        self.bits = bits
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        if quant_fn is None:
            # strand-pv mode: base holds the reconstruction from quantize-model
            self.register_buffer("base", torch.zeros_like(self.weight.data))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_fn is None:
            wq = self.base + self.weight
        else:
            wq = self.quant_fn(self.weight, self.bits)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, wq.to(x.dtype), bias)

    def extra_repr(self) -> str:
        fn_name = self.quant_fn.__name__ if self.quant_fn is not None else "strand-pv"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"quant={fn_name}, bits={self.bits}"
        )


# ---------------------------------------------------------------------------
# Module wrapping
# ---------------------------------------------------------------------------

def wrap_linears(
    model: nn.Module,
    suffixes: Tuple[str, ...],
    quant_fn: Optional[Callable],
    bits: Optional[int],
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
) -> int:
    """Replace every nn.Linear whose attribute name ends with a string in ``suffixes``
    with QuantLinear.

    include_regex: if set, the full module path must match.
    exclude_regex: if set, the full module path must NOT match.
    Returns number of wrapped modules.
    """
    inc_pat = re.compile(include_regex) if include_regex is not None else None
    exc_pat = re.compile(exclude_regex) if exclude_regex is not None else None

    count = 0
    # Collect replacements first to avoid mutating the tree mid-traversal.
    replacements: List[Tuple[nn.Module, str, str]] = []

    for parent_path, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if not any(child_name.endswith(s) for s in suffixes):
                continue
            full_path = f"{parent_path}.{child_name}" if parent_path else child_name
            if inc_pat is not None and not inc_pat.search(full_path):
                continue
            if exc_pat is not None and exc_pat.search(full_path):
                continue
            replacements.append((parent, child_name, full_path))

    for parent, child_name, _full_path in replacements:
        old_lin: nn.Linear = getattr(parent, child_name)
        new_mod = QuantLinear(old_lin, quant_fn, bits)
        setattr(parent, child_name, new_mod)
        count += 1

    return count


def list_wrapped_modules(model: nn.Module) -> List[Tuple[str, "QuantLinear"]]:
    """Return list of (full_name, module) for all QuantLinear instances in model."""
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, QuantLinear)
    ]


def freeze_by_module_policy(model: nn.Module, policy: Dict[str, bool]) -> None:
    """Set requires_grad according to prefix policy.

    policy: dict mapping module-path-prefix -> bool (True = trainable).
    Any parameter whose full name does not match any prefix gets requires_grad=False.

    Example:
        freeze_by_module_policy(model, {"layers": True, "lm_head": True})
        # freezes embeddings, norms, and any other prefix not listed.
    """
    for param_name, param in model.named_parameters():
        trainable = False
        for prefix, flag in policy.items():
            if param_name.startswith(prefix):
                trainable = flag
                break
        param.requires_grad_(trainable)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RWKV7_TIME_SUFFIXES: Tuple[str, ...] = ("r_proj", "k_proj", "v_proj", "o_proj")
RWKV7_FFN_SUFFIXES: Tuple[str, ...] = ("key", "value")
RWKV7_ALL_PROJ_SUFFIXES: Tuple[str, ...] = RWKV7_TIME_SUFFIXES + RWKV7_FFN_SUFFIXES

QUANT_FNS: Dict[str, Callable] = {
    "binary": quant_binary,
    "ternary": quant_ternary,
    "uniform": quant_uniform_symmetric,
}
