"""§19.6 student ladder as declared stages.

THEIA_MICRO / THEIA_LAB / THEIA_WORKER / THEIA_RESEARCH need a training
campaign. They are BLOCKED_EXTERNAL, never ABSENT and never SCAFFOLDED.
Each carries a machine-readable wake condition. This module does not
train, download, or stub a weight file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"

WAKE_THEIA_MICRO: dict[str, Any] = {
    "id": "WAKE.THEIA_MICRO",
    "kind": "TRAINING_CAMPAIGN",
    "status_if_unmet": BLOCKED_EXTERNAL,
    "all_of": [
        "T0_TEACHER_REGISTRY_LIVE",
        "T0_DATA_LICENSE_REGISTRY_LIVE",
        "T0_TRACE_SCHEMA_SEALED",
        "T0_CURRICULUM_DAG_SEALED",
        "T0_CHECKPOINT_AUTHORITY_LIVE",
        "T0_ADAPTERS_LIVE",
        "T0_RETENTION_EVALUATOR_LIVE",
        "T0_TRAINING_RECEIPT_AUTHORITY_LIVE",
        "T0_ROLLBACK_EXISTS",
        "D0_SOURCE_AND_TEACHER_QUALIFIED",
    ],
    "produces": (
        "THEIA_MICRO (~1B-3B) checkpoint with measured training loss, "
        "tool plumbing, retention and compression-aware recovery"
    ),
    "not_sufficient": [
        "tools/theia directory exists",
        "import tools.theia succeeds",
        "empty weight file",
        "SCAFFOLDED status",
        "ABSENT flipped to a label",
    ],
    "roadmap": "§19.6 THEIA_MICRO; APPENDIX G.6 T0→T1",
}

WAKE_THEIA_LAB: dict[str, Any] = {
    "id": "WAKE.THEIA_LAB",
    "kind": "TRAINING_CAMPAIGN",
    "status_if_unmet": BLOCKED_EXTERNAL,
    "all_of": [
        "THEIA_MICRO_PROMOTED",
        "D1_SUPERVISED_FUNCTIONAL_DISTILLATION_RECEIPTED",
        "D2_ON_POLICY_DISTILLATION_RECEIPTED",
        "GENERALIZED_METHOD_TOURNAMENT_COMPLETE",
        "COST_SAMPLE_EFFICIENCY_COMPARISON_RECEIPTED",
    ],
    "produces": (
        "THEIA_LAB (~7B-14B) with distillation/curriculum tournament evidence"
    ),
    "not_sufficient": [
        "THEIA_MICRO directory stub",
        "parameter count in range",
        "SCAFFOLDED status",
    ],
    "roadmap": "§19.6 THEIA_LAB; APPENDIX G.6 T2",
}

WAKE_THEIA_WORKER: dict[str, Any] = {
    "id": "WAKE.THEIA_WORKER",
    "kind": "TRAINING_CAMPAIGN",
    "status_if_unmet": BLOCKED_EXTERNAL,
    "all_of": [
        "THEIA_LAB_PROMOTED",
        "MULTI_LAB_BOUNTY_WORK_RECEIPTED",
        "HCLI_INTEGRATION_RECEIPTED",
        "T3_SERIOUS_LOCAL_BOUNTY_AGENT_QUALIFIED",
    ],
    "produces": (
        "THEIA_WORKER (~20B-40B) serious local multi-lab/bounty work with "
        "HCLI integration"
    ),
    "not_sufficient": [
        "HCLI import succeeds",
        "lab enum exists",
        "SCAFFOLDED status",
    ],
    "roadmap": "§19.6 THEIA_WORKER; APPENDIX G.6 T3",
}

WAKE_THEIA_RESEARCH: dict[str, Any] = {
    "id": "WAKE.THEIA_RESEARCH",
    "kind": "TRAINING_CAMPAIGN",
    "status_if_unmet": BLOCKED_EXTERNAL,
    "all_of": [
        "THEIA_WORKER_PROMOTED",
        "SIZE_ARCHITECTURE_CHOSEN_FROM_EVIDENCE",
        "GRAVITY_READY_BASELINE_FROZEN",
        "RETENTION_AND_BOUNTY_EVALUATION_PASSED",
        "INDEPENDENT_CHALLENGE_PASSED",
        "ROLLBACK_EXISTS",
    ],
    "produces": (
        "THEIA_RESEARCH (~30B-100B+) flagship candidate frozen as "
        "Gravity-ready baseline"
    ),
    "not_sufficient": [
        "largest checkpoint on disk",
        "name contains RESEARCH",
        "SCAFFOLDED status",
    ],
    "roadmap": (
        "§19.6 THEIA_RESEARCH; APPENDIX G.6 T4; APPENDIX G.7 promotion gate"
    ),
}


@dataclass(frozen=True)
class ModelStage:
    name: str
    size_hint: str
    purpose: str
    status: str
    blocker: str
    wake_condition: dict[str, Any]

    def __post_init__(self) -> None:
        if self.status != BLOCKED_EXTERNAL:
            raise ValueError(
                f"{self.name} status must be {BLOCKED_EXTERNAL}, got {self.status!r}"
            )
        if self.status in {"ABSENT", "SCAFFOLDED"}:
            raise ValueError(f"{self.name} must not be ABSENT or SCAFFOLDED")
        wc = self.wake_condition
        for key in ("id", "kind", "all_of", "status_if_unmet"):
            if key not in wc:
                raise ValueError(f"{self.name} wake_condition missing {key}")
        if wc["status_if_unmet"] != BLOCKED_EXTERNAL:
            raise ValueError(f"{self.name} wake status_if_unmet must be BLOCKED_EXTERNAL")
        if not wc["all_of"]:
            raise ValueError(f"{self.name} wake_condition.all_of is empty")


STAGES: tuple[ModelStage, ...] = (
    ModelStage(
        name="THEIA_MICRO",
        size_hint="~1B–3B",
        purpose="prove data/loss/tool/retention plumbing",
        status=BLOCKED_EXTERNAL,
        blocker=(
            "Hawking Train T0 substrate is not a live campaign in this "
            "checkout: no teacher/data/trace/curriculum/checkpoint authority "
            "is running, and this engine must not train or stub a model"
        ),
        wake_condition=WAKE_THEIA_MICRO,
    ),
    ModelStage(
        name="THEIA_LAB",
        size_hint="~7B–14B",
        purpose="distillation/curriculum tournament",
        status=BLOCKED_EXTERNAL,
        blocker=(
            "THEIA_MICRO has not been promoted; D1/D2 distillation receipts "
            "and a generalized-method tournament do not exist"
        ),
        wake_condition=WAKE_THEIA_LAB,
    ),
    ModelStage(
        name="THEIA_WORKER",
        size_hint="~20B–40B",
        purpose="serious local bounty/research agent",
        status=BLOCKED_EXTERNAL,
        blocker=(
            "THEIA_LAB has not been promoted; multi-lab bounty work with "
            "HCLI integration has not been receipted"
        ),
        wake_condition=WAKE_THEIA_WORKER,
    ),
    ModelStage(
        name="THEIA_RESEARCH",
        size_hint="~30B–100B+",
        purpose=(
            "flagship candidate, preferably sparse/low-active-compute if large"
        ),
        status=BLOCKED_EXTERNAL,
        blocker=(
            "THEIA_WORKER has not been promoted; size/architecture has not "
            "been chosen from evidence and no Gravity-ready baseline is frozen"
        ),
        wake_condition=WAKE_THEIA_RESEARCH,
    ),
)


@dataclass(frozen=True)
class WakeEvaluation:
    stage: str
    satisfied: bool
    missing: tuple[str, ...]
    status: str


def evaluate_wake(
    stage: ModelStage, evidence: Mapping[str, bool] | None = None
) -> WakeEvaluation:
    evidence = evidence or {}
    missing = tuple(p for p in stage.wake_condition["all_of"] if not evidence.get(p))
    return WakeEvaluation(
        stage=stage.name,
        satisfied=not missing,
        missing=missing,
        status="AWAKE" if not missing else BLOCKED_EXTERNAL,
    )


def stages() -> tuple[ModelStage, ...]:
    return STAGES


def stage_by_name(name: str) -> ModelStage:
    for s in STAGES:
        if s.name == name:
            return s
    raise KeyError(name)
