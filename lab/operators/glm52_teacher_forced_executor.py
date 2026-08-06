#!/usr/bin/env python3.12
"""GLM-5.2 teacher-forced executor — thin config-bound wrapper.

All implementation lives in ``frankenstein_teacher_forced_executor`` with a
``DonorArchitecture`` object.  This module binds the GLM-5.2 architecture so
existing 78-layer capture callers, CLIs, and tests keep working unchanged.
"""
from __future__ import annotations

from lab.operators import frankenstein_teacher_forced_executor as _core
from lab.operators.frankenstein_teacher_forced_executor import (  # noqa: F401
    CORPUS_LEVELS,
    DEFAULT_MAX_SEQ,
    DEFAULT_MICROBATCH,
    DEFAULT_OUT,
    DEFAULT_SAMPLE_HIDDEN,
    GLM52_ARCHITECTURE,
    KIMI_K3_ARCHITECTURE,
    MIN_FREE_FLOOR_BYTES,
    SAMPLE_TOKEN_SLOTS,
    SCHEMA_CORPUS,
    SCHEMA_EVICTION,
    SCHEMA_LAYER_SHARD,
    SCHEMA_RECEIPT,
    DonorArchitecture,
    DoubleBufferState,
    ExecutorConfig,
    FrozenCorpus,
    FrozenSequence,
    LayerScopedSource,
    LayerWeightPlan,
    TeacherForcedError,
    _array_sha256,
    _expert_cartography_arrays,
    _glm_side_from_layers,
    _hidden_sample,
    _layer_metrics,
    _router_margin,
    _sample_positions,
    _write_layer_shard,
    atomic_seal_state,
    build_weight_plan,
    capture_layer_bounded,
    evict_shards,
    freeze_corpus,
    main,
    run_teacher_forced,
    shards_evictable_after,
)

# Explicit aliases kept for any external importers that expect GLM-only names.
ARCHITECTURE = GLM52_ARCHITECTURE

# free_bytes / assert_floor are re-bound in *this* module so monkeypatch on
# glm52_teacher_forced_executor.free_bytes still works (assert_floor looks up
# free_bytes in the defining module unless we wrap it here).
free_bytes = _core.free_bytes


def assert_floor(path, *, label: str = "workspace"):
    free = free_bytes(path)
    ok = free >= MIN_FREE_FLOOR_BYTES
    record = {
        "label": label,
        "path": str(path),
        "free_bytes": free,
        "floor_bytes": MIN_FREE_FLOOR_BYTES,
        "floor_preserved": ok,
        "headroom_bytes": free - MIN_FREE_FLOOR_BYTES,
    }
    if not ok:
        raise TeacherForcedError(
            f"25 GiB floor breached under {label}: free={free} floor={MIN_FREE_FLOOR_BYTES}"
        )
    return record

__all__ = [
    "ARCHITECTURE",
    "CORPUS_LEVELS",
    "DEFAULT_MAX_SEQ",
    "DEFAULT_MICROBATCH",
    "DEFAULT_OUT",
    "DEFAULT_SAMPLE_HIDDEN",
    "DonorArchitecture",
    "DoubleBufferState",
    "ExecutorConfig",
    "FrozenCorpus",
    "FrozenSequence",
    "GLM52_ARCHITECTURE",
    "KIMI_K3_ARCHITECTURE",
    "LayerScopedSource",
    "LayerWeightPlan",
    "MIN_FREE_FLOOR_BYTES",
    "SAMPLE_TOKEN_SLOTS",
    "SCHEMA_CORPUS",
    "SCHEMA_EVICTION",
    "SCHEMA_LAYER_SHARD",
    "SCHEMA_RECEIPT",
    "TeacherForcedError",
    "_array_sha256",
    "assert_floor",
    "atomic_seal_state",
    "build_weight_plan",
    "capture_layer_bounded",
    "evict_shards",
    "free_bytes",
    "freeze_corpus",
    "main",
    "run_teacher_forced",
    "shards_evictable_after",
]


if __name__ == "__main__":
    raise SystemExit(main())
