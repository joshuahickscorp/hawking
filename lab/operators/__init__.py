"""Lab science operators — single registry-backed authority (C-SCI-R1)."""
from __future__ import annotations
from typing import Any
import importlib
__all__ = ['acquire', 'acquisition', 'auth', 'bounded_cache', 'condense_controller', 'eco_common', 'evaluate', 'forge', 'glm52_activation_aware_pack', 'glm52_activation_aware_pack_v2', 'glm52_adapter', 'glm52_assemble', 'glm52_capture_program', 'glm52_common', 'glm52_contract', 'glm52_corpus', 'glm52_evidence_auth', 'glm52_framed_window_operator', 'glm52_functional_gauntlet', 'glm52_grounding', 'glm52_grounding_auth', 'glm52_moe_student', 'glm52_pack', 'glm52_parity', 'glm52_range_stream_executor', 'glm52_reference', 'glm52_restream_contract', 'glm52_shard_probe', 'glm52_source_fetch', 'glm52_state', 'glm52_synthetic', 'glm52_teacher_capture', 'glm52_telegram', 'glm52_terminal_proofs', 'glm52_xet_autotune', 'glm52_xet_live', 'gptoss_live_probe', 'gptoss_subbit_packer', 'gravity_bench_lab', 'gravity_exec', 'gravity_flop_ledger', 'gravity_forge', 'gravity_functional_codec', 'gravity_kernel_select', 'gravity_math', 'gravity_metal', 'gravity_metal_lab_b', 'gravity_moe_layer', 'gravity_potency', 'gravity_range_scheduler', 'gravity_real_fixtures', 'hawking_null_metric', 'notify', 'one_bit_ceiling', 'pack', 'quality_contract', 'storage_modes', 'subbit_closure']
_CACHE: dict[str, Any] = {}
def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    if name not in _CACHE:
        _CACHE[name] = importlib.import_module(f"lab.operators.{name}")
    return _CACHE[name]
def __dir__() -> list[str]:
    return list(__all__)
