#!/usr/bin/env python3.12
"""Versioned, fail-closed metadata contract for Hawking LoRA adapter safetensors.

The factor orientation is load-bearing. Hawking stores ``A`` as ``[out, rank]`` and ``B`` as
``[rank, in]`` and executes ``xW^T + (xB^T)A^T``. A generic LoRA file may use the opposite naming
or layout, so suffixes and matching shapes alone are not sufficient proof of compatibility.
"""
from __future__ import annotations

import os
import re
import sys


ADAPTER_SCHEMA = "hawking.lora_adapter"
ADAPTER_VERSION = "1"
FACTOR_ORIENTATION = "A[out,rank];B[rank,in]"
FACTOR_DTYPE = "F16"
A_SUFFIX = ".lora_A"
B_SUFFIX = ".lora_B"


class AdapterContractError(ValueError):
    pass


def build_metadata(*, model, wbase, rank, adapter_count, target_regex=None):
    return {
        "artifact_type": "hawking_lora_adapter",
        "adapter_schema": ADAPTER_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "factor_orientation": FACTOR_ORIENTATION,
        "factor_dtype": FACTOR_DTYPE,
        "factor_a_suffix": A_SUFFIX,
        "factor_b_suffix": B_SUFFIX,
        "model": str(model),
        "wbase": str(wbase),
        "rank": str(int(rank)),
        "adapter_count": str(int(adapter_count)),
        "target_regex": str(target_regex) if target_regex else "all",
    }


def _same_identity(actual, expected):
    if expected is None:
        return True
    if not isinstance(actual, str) or not actual:
        return False
    expected = str(expected)
    if actual == expected:
        return True
    # Local artifacts may be passed once as relative and once as absolute paths. HF ids normally
    # match exactly above; resolving two identical non-path ids also remains deterministic.
    return os.path.normcase(os.path.realpath(actual)) == os.path.normcase(os.path.realpath(expected))


def _positive_int(value, field, problems):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        problems.append(f"metadata {field} must be a positive integer, got {value!r}")
        return None
    if parsed <= 0 or str(parsed) != str(value):
        problems.append(f"metadata {field} must be a canonical positive integer, got {value!r}")
        return None
    return parsed


def validate_handle(handle, *, expected_model=None, expected_wbase=None, expected_rank=None,
                    expected_names=None, expected_target_regex=None):
    """Validate metadata plus every factor pair before a caller mutates a model."""
    metadata = dict(handle.metadata() or {})
    problems = []
    required_exact = {
        "artifact_type": "hawking_lora_adapter",
        "adapter_schema": ADAPTER_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "factor_orientation": FACTOR_ORIENTATION,
        "factor_dtype": FACTOR_DTYPE,
        "factor_a_suffix": A_SUFFIX,
        "factor_b_suffix": B_SUFFIX,
    }
    for field, expected in required_exact.items():
        if metadata.get(field) != expected:
            problems.append(f"metadata {field}={metadata.get(field)!r}, expected {expected!r}")

    rank = _positive_int(metadata.get("rank"), "rank", problems)
    adapter_count = _positive_int(metadata.get("adapter_count"), "adapter_count", problems)
    if expected_rank is not None and rank is not None and rank != int(expected_rank):
        problems.append(f"metadata rank={rank}, expected {int(expected_rank)}")
    if not _same_identity(metadata.get("model"), expected_model):
        problems.append(f"metadata model={metadata.get('model')!r} does not match {expected_model!r}")
    if not _same_identity(metadata.get("wbase"), expected_wbase):
        problems.append(f"metadata wbase={metadata.get('wbase')!r} does not match {expected_wbase!r}")
    target_regex = metadata.get("target_regex")
    if not isinstance(target_regex, str) or not target_regex:
        problems.append(f"metadata target_regex must be a non-empty string, got {target_regex!r}")
        target_regex = None
    elif target_regex != "all":
        try:
            re.compile(target_regex)
        except re.error as exc:
            problems.append(f"metadata target_regex is invalid: {exc}")
    expected_target = expected_target_regex or "all"
    if expected_target_regex is not None and target_regex != expected_target:
        problems.append(f"metadata target_regex={target_regex!r}, expected {expected_target!r}")

    keys = set(handle.keys())
    unexpected = sorted(k for k in keys if not (k.endswith(A_SUFFIX) or k.endswith(B_SUFFIX)))
    if unexpected:
        problems.append(f"unexpected non-adapter tensors: {unexpected[:5]}")
    names = sorted(k[:-len(A_SUFFIX)] for k in keys if k.endswith(A_SUFFIX))
    b_names = {k[:-len(B_SUFFIX)] for k in keys if k.endswith(B_SUFFIX)}
    if set(names) != b_names:
        missing_b = sorted(set(names) - b_names)
        missing_a = sorted(b_names - set(names))
        problems.append(f"unpaired factors: missing_B={missing_b[:5]} missing_A={missing_a[:5]}")
    if not names:
        problems.append("adapter contains no factor pairs")
    if adapter_count is not None and adapter_count != len(names):
        problems.append(f"metadata adapter_count={adapter_count}, file contains {len(names)} pairs")
    if expected_names is not None and set(names) != set(expected_names):
        missing = sorted(set(expected_names) - set(names))
        extra = sorted(set(names) - set(expected_names))
        problems.append(f"adapter module set mismatch: missing={missing[:5]} extra={extra[:5]}")

    entries = []
    for name in names:
        ak, bk = name + A_SUFFIX, name + B_SUFFIX
        if bk not in keys:
            continue
        ashape = tuple(handle.get_slice(ak).get_shape())
        bshape = tuple(handle.get_slice(bk).get_shape())
        adtype = str(handle.get_slice(ak).get_dtype())
        bdtype = str(handle.get_slice(bk).get_dtype())
        if len(ashape) != 2 or len(bshape) != 2:
            problems.append(f"{name}: factors must be rank-2, got A{ashape} B{bshape}")
            continue
        if rank is not None and (ashape[1] != rank or bshape[0] != rank):
            problems.append(
                f"{name}: orientation/rank mismatch A{ashape} B{bshape}, expected "
                f"A[out,{rank}] B[{rank},in]"
            )
        if adtype != FACTOR_DTYPE or bdtype != FACTOR_DTYPE:
            problems.append(f"{name}: factor dtype A={adtype} B={bdtype}, expected {FACTOR_DTYPE}")
        entries.append({"name": name, "a_key": ak, "b_key": bk,
                        "a_shape": ashape, "b_shape": bshape})

    if problems:
        raise AdapterContractError("invalid Hawking adapter: " + "; ".join(problems))
    return {"metadata": metadata, "rank": rank, "adapter_count": adapter_count,
            "target_regex": target_regex, "orientation": FACTOR_ORIENTATION, "entries": entries}


def expected_module_names(model, wbase, target_regex):
    """Reproduce Doctor's exact adapter selection from the model and decoded base keys."""
    import torch.nn as nn
    from safetensors import safe_open

    matcher = None if target_regex == "all" else re.compile(target_regex)
    with safe_open(wbase, framework="pt") as base:
        base_keys = set(base.keys())
    return {
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name + ".weight" in base_keys
        and (matcher is None or matcher.search(name))
    }


def validate_for_model(handle, model, *, expected_model, expected_wbase, expected_rank=None,
                       expected_target_regex="all"):
    """Bind metadata to an independently supplied targeting policy and exact module set.

    Callers must never derive authorization from the adapter's own regex. Studio evaluation passes
    ``all``; a future targeted lane must supply its regex from a separately bound run identity.
    """
    preliminary = validate_handle(
        handle, expected_model=expected_model, expected_wbase=expected_wbase,
        expected_rank=expected_rank, expected_target_regex=expected_target_regex,
    )
    expected_names = expected_module_names(model, expected_wbase, expected_target_regex)
    return validate_handle(
        handle, expected_model=expected_model, expected_wbase=expected_wbase,
        expected_rank=expected_rank, expected_names=expected_names,
        expected_target_regex=expected_target_regex,
    )


def selftest():
    class Slice:
        def __init__(self, shape, dtype="F16"):
            self.shape, self.dtype = shape, dtype
        def get_shape(self): return self.shape
        def get_dtype(self): return self.dtype

    class Handle:
        def __init__(self, metadata, tensors):
            self._metadata, self.tensors = metadata, tensors
        def metadata(self): return self._metadata
        def keys(self): return list(self.tensors)
        def get_slice(self, key): return self.tensors[key]

    meta = build_metadata(model="model", wbase="base", rank=8, adapter_count=1)
    valid = Handle(meta, {"layer.lora_A": Slice((16, 8)), "layer.lora_B": Slice((8, 32))})
    result = validate_handle(valid, expected_model="model", expected_wbase="base",
                             expected_rank=8, expected_names={"layer"})
    assert result["orientation"] == FACTOR_ORIENTATION and result["adapter_count"] == 1

    for mutation in (
        {**meta, "factor_orientation": "A[rank,out];B[in,rank]"},
        {key: value for key, value in meta.items() if key != "adapter_schema"},
        {key: value for key, value in meta.items() if key != "adapter_version"},
    ):
        try:
            validate_handle(Handle(mutation, valid.tensors))
            raise AssertionError("invalid metadata was accepted")
        except AdapterContractError:
            pass
    try:
        validate_handle(Handle(meta, {"layer.lora_A": Slice((8, 16)),
                                      "layer.lora_B": Slice((32, 8))}))
        raise AssertionError("transposed factors were accepted")
    except AdapterContractError:
        pass
    try:
        validate_handle(valid, expected_names={"layer", "missing"})
        raise AssertionError("subset adapter was accepted as a complete module set")
    except AdapterContractError:
        pass
    targeted_meta = build_metadata(
        model="model", wbase="base", rank=8, adapter_count=1,
        target_regex="^layer$",
    )
    try:
        validate_handle(
            Handle(targeted_meta, valid.tensors), expected_target_regex="all",
        )
        raise AssertionError("adapter-declared narrow target policy was accepted")
    except AdapterContractError:
        pass
    print("adapter_contract.py selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else 0)
