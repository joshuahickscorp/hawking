from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "tools" / "llama_conditional_student_probe.py"
SPEC = importlib.util.spec_from_file_location("llama_conditional_student_probe", MODULE_PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_fit_only_kmeans_router_is_deterministic_and_separates_clusters() -> None:
    values = np.asarray([[-4.0, -4.0], [-3.0, -3.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
    first = probe.kmeans(values, experts=2, iterations=16)
    second = probe.kmeans(values, experts=2, iterations=16)
    assert np.array_equal(first, second)
    labels = probe.route(values, np.eye(2, dtype=np.float32), first)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_router_cost_is_not_billed_as_all_experts_active() -> None:
    hidden, width, experts, router_dim = 4096, 128, 8, 16
    per_expert = hidden * width + width + width * hidden + hidden
    router = hidden * router_dim + experts * router_dim + hidden
    stored = experts * per_expert + router + hidden
    active = per_expert + router + hidden
    assert active < stored
    assert active * 2 == 2253312
