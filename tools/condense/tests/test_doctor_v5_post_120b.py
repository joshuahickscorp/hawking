#!/usr/bin/env python3.12
"""Durability tests for the unattended post-120B handoff."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


CONDENSE = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CONDENSE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


post = _load("doctor_v5_post_120b", "doctor_v5_post_120b.py")
autoresume = _load("doctor_v5_ultra_autoresume", "doctor_v5_ultra_autoresume.py")


class Post120BDurabilityTests(unittest.TestCase):
    def test_autoresume_never_restarts_complete_queue(self) -> None:
        self.assertTrue(autoresume._should_resume(
            {"mode": "run"}, {"status": "running"}
        ))
        self.assertFalse(autoresume._should_resume(
            {"mode": "run"}, {"status": "complete"}
        ))
        self.assertFalse(autoresume._should_resume(
            {"mode": "drain"}, {"status": "running"}
        ))

    def test_negative_dependency_does_not_create_finite_eta(self) -> None:
        plan = {"cells": [
            {"cell_id": "parent", "model_label": "0.5B", "priority": 0,
             "dependencies": []},
            {"cell_id": "child", "model_label": "0.5B", "priority": 1,
             "dependencies": ["parent"]},
        ]}
        campaign = {"cells": [
            {"cell_id": "parent", "status": "negative"},
            {"cell_id": "child", "status": "blocked-dependency"},
        ]}
        boundary, _, blocker = post._simulate_schedule(
            plan, campaign, {"parent": 0.0, "child": 1.0},
            {"0.5B": 4_000_000_000}, False,
        )
        self.assertIsNone(boundary)
        self.assertIn("no runnable cell", blocker)

    def test_live_runtime_spec_uses_queue_program_hash_contract(self) -> None:
        plan = post._read_json(post.PLAN)
        cell = plan["cells"][0]
        spec_path = post.ROOT / cell["runtime_spec_path"]
        spec = post._read_json(spec_path)
        runtime, _, errors = post.queue_contract._validate_runtime_spec(
            cell, spec, spec_path, verify_inputs=False
        )
        self.assertEqual([], errors)
        self.assertIsNotNone(runtime)


if __name__ == "__main__":
    unittest.main()
