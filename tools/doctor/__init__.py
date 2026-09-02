"""I-B Doctor: diagnose real receipts and sealed specimen metadata.

Public gate symbols (imports are not call sites; callers must invoke these):

    check_three_zeros(organ) -> dict[str, ZeroResult]
    ordinary_quantization(results) -> bool
    diagnose(target) -> dict
    walk_order() -> tuple[Technique, ...]
"""
from tools.doctor.access import AccessLog, WeightBytesForbidden, is_weight_file
from tools.doctor.engine import build, diagnose, selftest, zeros_controls
from tools.doctor.order import TECHNIQUE_ORDER, walk_order
from tools.doctor.zeros import (
    FAIL,
    PASS,
    THREE_ZEROS,
    UNKNOWN,
    ZERO_EXECUTION,
    ZERO_INDEPENDENT_INFORMATION,
    ZERO_STORAGE,
    BROKEN_ORGAN,
    ROUTED_EXPERT_ORGAN,
    TIED_EMBED_ORGAN,
    ZeroResult,
    check_three_zeros,
    check_zero_execution,
    check_zero_independent_information,
    check_zero_storage,
    ordinary_quantization,
    organ_from_mapping,
    organs_from_doc,
)

__all__ = [
    "AccessLog",
    "BROKEN_ORGAN",
    "FAIL",
    "PASS",
    "ROUTED_EXPERT_ORGAN",
    "TECHNIQUE_ORDER",
    "THREE_ZEROS",
    "TIED_EMBED_ORGAN",
    "UNKNOWN",
    "WeightBytesForbidden",
    "ZERO_EXECUTION",
    "ZERO_INDEPENDENT_INFORMATION",
    "ZERO_STORAGE",
    "ZeroResult",
    "build",
    "check_three_zeros",
    "check_zero_execution",
    "check_zero_independent_information",
    "check_zero_storage",
    "diagnose",
    "is_weight_file",
    "ordinary_quantization",
    "organ_from_mapping",
    "organs_from_doc",
    "selftest",
    "walk_order",
    "zeros_controls",
]
