"""P4 auto-implementation contract: task-class alias routing regression.

This exercises the production behavior added to ``hawking.auto_orchestration``:
legacy task-class labels must fold onto routable canonical classes, and every
alias target must itself be a declared task class.
"""
from __future__ import annotations

from hawking.auto_orchestration import (
    AUTO_TASK_CLASSES,
    TASK_CLASS_ALIASES,
    canonical_task_class,
)


def test_alias_targets_are_routable_task_classes() -> None:
    unroutable = sorted(
        {target for target in TASK_CLASS_ALIASES.values() if target not in AUTO_TASK_CLASSES}
    )
    assert unroutable == [], f"alias targets not routable: {unroutable}"


def test_legacy_labels_fold_onto_canonical_classes() -> None:
    assert canonical_task_class("research") == "source_mapping"
    assert canonical_task_class("repo_engineering") == "routine_implementation"
    assert canonical_task_class("debugging") == "bug_repair"
    assert canonical_task_class("planning") == "architecture_planning"
    assert canonical_task_class("architecture") == "architecture_planning"
    assert canonical_task_class("benchmark") == "test_design"
    assert canonical_task_class("physical_engineering") == "hardware_reasoning"
    assert canonical_task_class("synthesis") == "long_context_synthesis"
    assert canonical_task_class("model_science") == "long_context_synthesis"


def test_canonical_and_unknown_labels_are_preserved() -> None:
    assert canonical_task_class("routine_implementation") == "routine_implementation"
    assert canonical_task_class("source_mapping") == "source_mapping"
    assert canonical_task_class("totally_unknown_class") == "totally_unknown_class"


def test_non_string_labels_pass_through_unchanged() -> None:
    sentinel = object()
    assert canonical_task_class(sentinel) is sentinel