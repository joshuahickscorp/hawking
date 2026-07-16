#!/usr/bin/env python3.12
"""Project-suite entry point for the shared-preprocessing adversarial tests."""
from importlib import util
from pathlib import Path
import sys


_CORE = Path(__file__).resolve().parents[1] / "test_doctor_v5_shared_preprocess_cache.py"
_SPEC = util.spec_from_file_location("_doctor_v5_shared_preprocess_cache_core_tests", _CORE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load canonical shared-preprocessing tests: {_CORE}")
_MODULE = util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

SharedPreprocessCacheTests = _MODULE.SharedPreprocessCacheTests
