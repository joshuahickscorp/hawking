"""Lab science operators — single registry-backed authority (C-SCI-R1)."""
from __future__ import annotations
from typing import Any
import importlib
from lab.layout import ensure_experiment_imports
ensure_experiment_imports()
__all__ = ['acquire', 'acquisition', 'auth', 'bounded_cache', 'condense_controller', 'deepseek_v4_stream_executor', 'doctor_ladder', 'doctor_registry', 'eco_common', 'evaluate', 'expert_alloc', 'forge', 'glm52_activation_aware_pack', 'glm52_adapter', 'glm52_assemble', 'glm52_capture_program', 'glm52_common', 'glm52_contract', 'glm52_corpus', 'glm52_evidence_auth', 'glm52_framed_window_operator', 'glm52_functional_gauntlet', 'glm52_grounding', 'glm52_grounding_auth', 'glm52_moe_student', 'glm52_pack', 'glm52_parity', 'glm52_range_stream_executor', 'glm52_reference', 'glm52_restream_contract', 'glm52_shard_probe', 'glm52_source_fetch', 'glm52_state', 'glm52_synthetic', 'glm52_teacher_capture', 'glm52_telegram', 'glm52_terminal_proofs', 'glm52_xet_autotune', 'glm52_xet_live', 'gptoss_live_probe', 'gptoss_subbit_packer', 'gravity_exec', 'gravity_flop_ledger', 'gravity_forge', 'gravity_functional_codec', 'gravity_math', 'gravity_metal', 'gravity_potency', 'gravity_range_scheduler', 'kimi_k3_source_admission', 'lowbit_qat', 'mixed_precision_alloc', 'notify', 'one_bit_ceiling', 'pack', 'quality_contract', 'storage_modes', 'subbit_closure']
_CACHE: dict[str, Any] = {}
def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    if name not in _CACHE:
        _CACHE[name] = importlib.import_module(f"lab.operators.{name}")
    return _CACHE[name]
def __dir__() -> list[str]:
    return list(__all__)
