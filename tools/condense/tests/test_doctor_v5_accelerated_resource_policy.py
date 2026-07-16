#!/usr/bin/env python3.12
"""Adversarial no-model gates for accelerated resource-stop truthfulness."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))
import doctor_v5_accelerated_resource_policy as policy
import doctor_v5_ultra_accelerated_queue as accelerated


class AcceleratedResourcePolicyTests(unittest.TestCase):
    BUDGET = 78_000_000_000
    MAX_STOPS = 5

    def decide(self, *, reason: str, measured: int, previous: int = 0) -> dict:
        return policy.resource_stop_decision(
            reason=reason, measured_cell_rss_bytes=measured,
            process_budget_bytes=self.BUDGET,
            previous_consecutive_stops=previous,
            max_resource_stops=self.MAX_STOPS,
        )

    def test_global_pressure_clears_contention_without_escalating_sole_lane(self) -> None:
        decision = self.decide(
            reason=policy.GLOBAL_PRESSURE_REASON,
            measured=self.BUDGET - 1,
            previous=self.MAX_STOPS - 1,
        )
        self.assertEqual("global-pressure-or-swap", decision["classification"])
        self.assertFalse(decision["escalate"])
        self.assertFalse(decision["cell_at_or_over_budget"])
        self.assertEqual(0, decision["next_consecutive_stops"])

    def test_exact_measured_cell_budget_escalates_immediately(self) -> None:
        decision = self.decide(
            reason=policy.POOL_BUDGET_REASON, measured=self.BUDGET,
        )
        self.assertEqual("measured-cell-budget", decision["classification"])
        self.assertTrue(decision["escalate"])
        self.assertTrue(decision["cell_at_or_over_budget"])
        self.assertIn(str(self.BUDGET), decision["detail"])

    def test_measured_budget_dominates_even_when_global_pressure_coexists(self) -> None:
        decision = self.decide(
            reason=policy.GLOBAL_PRESSURE_REASON, measured=self.BUDGET + 1,
        )
        self.assertEqual("measured-cell-budget", decision["classification"])
        self.assertTrue(decision["escalate"])

    def test_aggregate_contention_retains_bounded_retry_safety(self) -> None:
        for previous in range(self.MAX_STOPS - 1):
            decision = self.decide(
                reason=policy.POOL_BUDGET_REASON,
                measured=self.BUDGET - 1, previous=previous,
            )
            self.assertFalse(decision["escalate"])
            self.assertEqual(previous + 1, decision["next_consecutive_stops"])
        final = self.decide(
            reason=policy.POOL_BUDGET_REASON,
            measured=self.BUDGET - 1, previous=self.MAX_STOPS - 1,
        )
        self.assertTrue(final["escalate"])
        self.assertEqual("aggregate-pool-contention", final["classification"])
        self.assertFalse(final["cell_at_or_over_budget"])

    def test_untyped_or_unclassified_inputs_fail_closed(self) -> None:
        cases = [
            {"reason": "unknown", "measured_cell_rss_bytes": 1,
             "process_budget_bytes": self.BUDGET,
             "previous_consecutive_stops": 0,
             "max_resource_stops": self.MAX_STOPS},
            {"reason": policy.POOL_BUDGET_REASON,
             "measured_cell_rss_bytes": True,
             "process_budget_bytes": self.BUDGET,
             "previous_consecutive_stops": 0,
             "max_resource_stops": self.MAX_STOPS},
            {"reason": policy.POOL_BUDGET_REASON,
             "measured_cell_rss_bytes": 1,
             "process_budget_bytes": self.BUDGET,
             "previous_consecutive_stops": -1,
             "max_resource_stops": self.MAX_STOPS},
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(policy.ResourcePolicyError):
                    policy.resource_stop_decision(**case)

    def test_fixed_twenty_launch_charges_total_host_cpu(self) -> None:
        blocked = policy.fixed_thread_cpu_launch_decision(
            [24.0], logical_cores=28, guard_cores=2.0, launch_threads=20.0
        )
        self.assertFalse(blocked["ok"])
        self.assertEqual(0, blocked["launch_tokens"])
        self.assertEqual(24.0, blocked["charged_global_cpu_cores"])
        exact = policy.fixed_thread_cpu_launch_decision(
            [6.0], logical_cores=28, guard_cores=2.0, launch_threads=20.0
        )
        self.assertTrue(exact["ok"])
        self.assertEqual(1, exact["launch_tokens"])

    def test_cpu_window_is_immediate_on_saturation_and_hysteretic_on_recovery(self) -> None:
        held = policy.fixed_thread_cpu_launch_decision(
            [24.0, 0.0, 0.0], logical_cores=28, guard_cores=2.0,
            launch_threads=20.0,
        )
        recovered = policy.fixed_thread_cpu_launch_decision(
            [0.0, 0.0, 0.0], logical_cores=28, guard_cores=2.0,
            launch_threads=20.0,
        )
        self.assertFalse(held["ok"])
        self.assertTrue(recovered["ok"])

    def test_invalid_global_cpu_evidence_fails_closed(self) -> None:
        for samples in ([], [True], [math.nan], [-1.0]):
            with self.subTest(samples=samples):
                with self.assertRaises(policy.ResourcePolicyError):
                    policy.fixed_thread_cpu_launch_decision(
                        samples, logical_cores=28, guard_cores=2.0,
                        launch_threads=20.0,
                    )

    def test_accelerated_hook_ignores_sole_live_proxy_under_global_pressure(self) -> None:
        cell_id = "global-pressure-cell"
        state = self._state(cell_id, reason=policy.GLOBAL_PRESSURE_REASON)
        state["resource_stop_counts"][cell_id] = self.MAX_STOPS - 1
        saved_append = accelerated._BASE._append_event
        events: list[tuple[tuple, dict]] = []
        accelerated._RESOURCE_STOP_SAMPLES.clear()
        accelerated._RESOURCE_STOP_SAMPLES[cell_id] = self.BUDGET - 1
        try:
            accelerated._BASE._append_event = (
                lambda *args, **kwargs: events.append((args, kwargs))
            )
            blocker = accelerated._truthful_record_resource_stop(
                state, cell_id, sole_live=True
            )
        finally:
            accelerated._BASE._append_event = saved_append
            accelerated._RESOURCE_STOP_SAMPLES.clear()
        self.assertIsNone(blocker)
        self.assertEqual("running", state["cells"][cell_id]["status"])
        self.assertNotIn(cell_id, state["resource_stop_counts"])
        self.assertEqual([], events)

    def test_accelerated_hook_uses_measured_rss_even_without_sole_lane(self) -> None:
        cell_id = "measured-budget-cell"
        state = self._state(cell_id, reason=policy.POOL_BUDGET_REASON)
        saved_append = accelerated._BASE._append_event
        events: list[tuple[tuple, dict]] = []
        accelerated._RESOURCE_STOP_SAMPLES.clear()
        accelerated._RESOURCE_STOP_SAMPLES[cell_id] = self.BUDGET
        try:
            accelerated._BASE._append_event = (
                lambda *args, **kwargs: events.append((args, kwargs))
            )
            blocker = accelerated._truthful_record_resource_stop(
                state, cell_id, sole_live=False
            )
        finally:
            accelerated._BASE._append_event = saved_append
            accelerated._RESOURCE_STOP_SAMPLES.clear()
        self.assertIsNotNone(blocker)
        self.assertEqual("blocked-execution", state["cells"][cell_id]["status"])
        self.assertEqual("measured-cell-budget", events[0][1]["classification"])
        self.assertTrue(events[0][1]["cell_at_or_over_budget"])

    def test_enforcement_wrapper_scopes_exact_sample_and_clears_it(self) -> None:
        cell_id = "context-cell"
        state = self._state(cell_id, reason=policy.GLOBAL_PRESSURE_REASON)
        saved_enforce = accelerated._ORIGINAL_ENFORCE_POOL_BUDGET
        seen: dict[str, int] = {}

        def frozen_guard(plan: dict, current: dict, live: dict,
                         samples: dict, aggregate: int) -> list[str]:
            seen.update(accelerated._RESOURCE_STOP_SAMPLES)
            blocker = accelerated._truthful_record_resource_stop(
                current, cell_id, sole_live=True
            )
            self.assertIsNone(blocker)
            current["cells"][cell_id]["status"] = "pending"
            return [cell_id]

        accelerated._RESOURCE_STOP_SAMPLES.clear()
        try:
            accelerated._ORIGINAL_ENFORCE_POOL_BUDGET = frozen_guard
            stopped = accelerated._truthful_enforce_pool_budget(
                {}, state, {}, {cell_id: {"tree_rss_bytes": 1234}}, 1234
            )
        finally:
            accelerated._ORIGINAL_ENFORCE_POOL_BUDGET = saved_enforce
            accelerated._RESOURCE_STOP_SAMPLES.clear()
        self.assertEqual([cell_id], stopped)
        self.assertEqual({cell_id: 1234}, seen)
        self.assertEqual("pending", state["cells"][cell_id]["status"])
        self.assertEqual({}, accelerated._RESOURCE_STOP_SAMPLES)

    @staticmethod
    def _state(cell_id: str, *, reason: str) -> dict:
        return {
            "last_resource_stop": {"reason": reason},
            "resource_stop_counts": {},
            "cells": {cell_id: {"status": "running", "error": None,
                                "blockers": []}},
        }


if __name__ == "__main__":
    unittest.main()
