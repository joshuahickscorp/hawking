"""Focused regression: the execution_plan task class is admitted and routable.

P4_GOAL_WORKUNIT_AUTO_EXECUTIONPLAN requires that Auto's declared task-class
vocabulary include the execution_plan label so that WorkUnit packets carrying
that class are accepted by the deterministic admission path instead of being
rejected as an unknown class.
"""
from __future__ import annotations

from hawking.auto_orchestration import AUTO_TASK_CLASSES


def test_execution_plan_is_declared_task_class() -> None:
    assert "execution_plan" in AUTO_TASK_CLASSES


def test_execution_plan_task_class_is_unique() -> None:
    assert AUTO_TASK_CLASSES.count("execution_plan") == 1