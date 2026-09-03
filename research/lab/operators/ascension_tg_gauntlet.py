"""Self-TG gauntlet + temporal Gravity ladder measurement harness (scaffold).

Generalizes tonight's DSV4F complete-token profiler + separated TPS scoreboards:

- ``lab/operators/deepseek_v4_gravity.py`` (``_DiagnosticTokenProfiler``,
  ``profile_prompt``, ``_aggregate_complete_token_profile``)
- ``tools/condense/tests/test_deepseek_v4_complete_token_profile.py``
- ``crates/hawking-speculate/src/metrics_sep.rs``
  (``BaseTrueTps`` / ``AcceleratedAcceptedTps`` — never averaged)
- bible §10 Self-TG loop + TG32..TG1 ladder + TG3 stop-for-review

Metrics kept **structurally separate** (never blended into a mean TPS):

- BASE_TRUE_TPS
- BLOCK_EXECUTED_TPS
- ACCELERATED_ACCEPTED_TPS
- PREFILL_TPS
- TTFT
- HCLI_TOOL_AUGMENTED_THROUGHPUT

Scaffold only: no live Qwen benchmarks, no BASE_TRUE_TPS claims without an
eligible full-runtime receipt (mirrors ``BASE_TRUE_TPS_WITHHELD``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from lab.operators.ascension_parity_ladder import (
    ModelFamily,
    ParityLadderHarness,
    RungStatus,
    default_claim_boundary,
)
from lab.receipts import seal, verify

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

TG_GAUNTLET_RECEIPT_SCHEMA = "hawking.ascension.tg_gauntlet_receipt.v1"
TG_RUNG_RECEIPT_SCHEMA = "hawking.ascension.tg_rung_receipt.v1"
COMPLETE_TOKEN_PROFILE_SCHEMA = "hawking.ascension.complete_token_profile.v1"
SCOREBOARD_SCHEMA = "hawking.ascension.separated_tps_scoreboard.v1"
SELF_TG_LOOP_SCHEMA = "hawking.ascension.self_tg_loop.v1"

# ---------------------------------------------------------------------------
# Temporal Gravity ladder (bible §10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TgRung:
    id: str
    target_tps: float
    note: str = ""


# Ordered from easiest (lowest TPS) to hardest (highest TPS).
TG_LADDER: tuple[TgRung, ...] = (
    TgRung("TG32", 31.25),
    TgRung("TG20", 50.0),
    TgRung("TG16", 62.5),
    TgRung("TG12", 83.3),
    TgRung("TG10", 100.0),
    TgRung("TG8", 125.0),
    TgRung("TG5", 200.0),
    TgRung("TG4", 250.0),
    TgRung("TG3", 333.0, note="mandatory stop-for-human-review threshold"),
    TgRung("TG2", 500.0),
    TgRung("TG1", 1000.0, note="final scientific target class, not auto-promoted"),
)

TG_LADDER_BY_ID: dict[str, TgRung] = {r.id: r for r in TG_LADDER}
TG3_REVIEW_THRESHOLD = "TG3"
TG3_TARGET_TPS = 333.0


class SeparatedMetric(str, Enum):
    """Scoreboard names that must never be averaged together."""

    BASE_TRUE_TPS = "BASE_TRUE_TPS"
    BLOCK_EXECUTED_TPS = "BLOCK_EXECUTED_TPS"
    ACCELERATED_ACCEPTED_TPS = "ACCELERATED_ACCEPTED_TPS"
    PREFILL_TPS = "PREFILL_TPS"
    TTFT = "TTFT"
    HCLI_TOOL_AUGMENTED_THROUGHPUT = "HCLI_TOOL_AUGMENTED_THROUGHPUT"


SEPARATED_METRIC_NAMES: tuple[str, ...] = tuple(m.value for m in SeparatedMetric)


# ---------------------------------------------------------------------------
# Self-TG loop steps (bible §10)
# ---------------------------------------------------------------------------

SELF_TG_LOOP_STEPS: tuple[str, ...] = (
    "profile_own_complete_token",
    "rank_bottleneck",
    "retrieve_prior_failures",
    "propose_three_materially_different_mechanisms",
    "select_cheapest_distinguishing_experiment",
    "implement_in_isolated_worktree",
    "protected_parity",
    "protected_capability",
    "clean_benchmark",
    "adversarial_review",
    "report",
    "continue",
)


# Every TG rung requires (bible §10):
TG_RUNG_REQUIREMENTS: tuple[str, ...] = (
    "same_model",
    "same_capability_tier",
    "complete_token_timing",
    "batch_1_base_runtime",
    "clean_benchmark",
    "fallback_eq_0",
    "real_gpu_dispatch",
    "stable_p99",
    "prompt_dependent_coherent_generation",
)


# ---------------------------------------------------------------------------
# Separated metric cells (generalizes metrics_sep.rs newtypes in Python)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricCell:
    """One scoreboard cell. Value is None until an eligible measurement exists."""

    name: str
    value: float | None
    status: str
    unit: str
    eligibility: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "unit": self.unit,
            "eligibility": self.eligibility,
            "note": self.note,
        }


def withheld_cell(name: str, *, reason: str) -> MetricCell:
    unit = "ms" if name == SeparatedMetric.TTFT.value else "tokens_per_second"
    return MetricCell(
        name=name,
        value=None,
        status=RungStatus.METRIC_WITHHELD.value,
        unit=unit,
        eligibility="NOT_ELIGIBLE_SCAFFOLD_OR_NO_RUNTIME",
        note=reason,
    )


def empty_separated_scoreboard(*, reason: str) -> dict[str, Any]:
    """Structurally separated scoreboard — no average / blend field."""
    cells = {name: withheld_cell(name, reason=reason).as_dict() for name in SEPARATED_METRIC_NAMES}
    body = {
        "schema": SCOREBOARD_SCHEMA,
        "status": RungStatus.BASE_TRUE_TPS_WITHHELD.value,
        "metrics": cells,
        "blended_tps_forbidden": True,
        "average_of_scoreboards_forbidden": True,
        "reference": {
            "rust_metrics_sep": "crates/hawking-speculate/src/metrics_sep.rs",
            "dsv4f_profile": "lab/operators/deepseek_v4_gravity.py",
            "dsv4f_profile_tests": "tools/condense/tests/test_deepseek_v4_complete_token_profile.py",
            "dsv4f_scoreboard": "tools/condense/tests/test_deepseek_v4_child_baseline.py",
        },
        "claim_boundary": {
            "base_true_tps": False,
            "accelerated_accepted_tps": False,
            "block_executed_tps": False,
            "prefill_tps": False,
            "ttft": False,
            "hcli_tool_augmented_throughput": False,
            "live_benchmark": False,
        },
    }
    return seal(body)


def assert_no_blended_tps(scoreboard: Mapping[str, Any]) -> None:
    """Fail closed if a caller tries to store a blended mean TPS."""
    forbidden_keys = (
        "mean_tps",
        "avg_tps",
        "average_tps",
        "blended_tps",
        "combined_tps",
        "overall_tps",
    )
    for key in forbidden_keys:
        if key in scoreboard:
            raise ValueError(f"forbidden blended TPS field present: {key}")
        metrics = scoreboard.get("metrics")
        if isinstance(metrics, Mapping) and key in metrics:
            raise ValueError(f"forbidden blended TPS metric present: {key}")


# ---------------------------------------------------------------------------
# Complete-token profiler scaffold (generalizes _DiagnosticTokenProfiler shape)
# ---------------------------------------------------------------------------


@dataclass
class StageMetric:
    stage: str
    cpu_duration_ms: float = 0.0
    cpu_wall_elapsed_ms: float = 0.0
    gpu_duration_ms: float = 0.0
    dispatches: int = 0
    bytes_read_estimate: int = 0
    bytes_written_estimate: int = 0
    execution_status: str = "not_executed_scaffold"
    occupancy_status: str = "not_available_scaffold"
    effective_bandwidth_status: str = "not_available_scaffold"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "cpu_duration_ms": self.cpu_duration_ms,
            "cpu_wall_elapsed_ms": self.cpu_wall_elapsed_ms,
            "gpu_duration_ms": self.gpu_duration_ms,
            "dispatches": self.dispatches,
            "bytes_read_estimate": self.bytes_read_estimate,
            "bytes_written_estimate": self.bytes_written_estimate,
            "execution_status": self.execution_status,
            "occupancy_status": self.occupancy_status,
            "effective_bandwidth_status": self.effective_bandwidth_status,
        }


@dataclass
class CompleteTokenProfiler:
    """Family-parameterized complete-token stage ledger.

    Shape mirrors DSV4F ``_DiagnosticTokenProfiler`` / ``finish()`` output so
    live Qwen profilers can drop in the same aggregation path later.
    """

    family: str
    phase: str  # "prefill" | "decode"
    token_ordinal: int
    position: int
    stages: Sequence[str]
    _rows: dict[str, StageMetric] = field(init=False)

    def __post_init__(self) -> None:
        self._rows = {name: StageMetric(stage=name) for name in self.stages}

    def mark_executed(
        self,
        stage: str,
        *,
        cpu_wall_elapsed_ms: float,
        cpu_duration_ms: float | None = None,
        gpu_duration_ms: float = 0.0,
        dispatches: int = 0,
        bytes_read_estimate: int = 0,
        bytes_written_estimate: int = 0,
    ) -> None:
        if stage not in self._rows:
            raise KeyError(f"stage {stage!r} not in family inventory")
        row = self._rows[stage]
        row.cpu_wall_elapsed_ms = cpu_wall_elapsed_ms
        row.cpu_duration_ms = (
            cpu_duration_ms if cpu_duration_ms is not None else cpu_wall_elapsed_ms
        )
        row.gpu_duration_ms = gpu_duration_ms
        row.dispatches = dispatches
        row.bytes_read_estimate = bytes_read_estimate
        row.bytes_written_estimate = bytes_written_estimate
        row.execution_status = "executed" if (cpu_wall_elapsed_ms > 0 or dispatches > 0) else "measured_zero"

    def finish(
        self,
        *,
        forward_wall_elapsed_ms: float,
        forward_cpu_duration_ms: float,
    ) -> dict[str, Any]:
        stage_metrics = [self._rows[name].as_dict() for name in self.stages]
        named_wall = sum(r["cpu_wall_elapsed_ms"] for r in stage_metrics)
        unexplained = max(0.0, forward_wall_elapsed_ms - named_wall)
        other_share = (
            0.0
            if forward_wall_elapsed_ms <= 0
            else 100.0 * unexplained / forward_wall_elapsed_ms
        )
        timing_status = (
            "PASS_ALL_TIME_EXPLICITLY_NAMED"
            if unexplained == 0.0
            else "FAIL_UNEXPLAINED_OTHER_WALL"
        )
        body = {
            "schema": COMPLETE_TOKEN_PROFILE_SCHEMA,
            "family": self.family,
            "phase": self.phase,
            "token_ordinal": self.token_ordinal,
            "position": self.position,
            "stage_metrics": stage_metrics,
            "timing_accounting": {
                "observed_complete_token_wall_elapsed_ms": forward_wall_elapsed_ms,
                "observed_complete_token_cpu_duration_ms": forward_cpu_duration_ms,
                "named_stage_wall_elapsed_ms": named_wall,
                "unexplained_other_wall_elapsed_ms": unexplained,
                "other_share_percent": other_share,
                "status": timing_status,
                "target_explained_percent": 98.0,
            },
            "gpu_dispatch_accounting": {
                "dispatches_total": sum(r["dispatches"] for r in stage_metrics),
                "command_buffers_total": 0,
                "waits_total": 0,
            },
            "claim_boundary": {
                "base_true_tps": False,
                "live_gpu_profile": False,
                "scaffold_profile_shape_only": True,
            },
        }
        return seal(body)


def scaffold_complete_token_profile(family: ModelFamily | str) -> dict[str, Any]:
    """Zero-filled profile with the correct stage inventory for the family."""
    harness = ParityLadderHarness(family=family)
    profiler = CompleteTokenProfiler(
        family=harness.family_key,
        phase="decode",
        token_ordinal=0,
        position=0,
        stages=harness.stages(),
    )
    # Bookkeeping stage absorbs the synthetic observed wall so "other" stays 0
    # on the scaffold shape test (same honesty pattern as DSV4F profile tests).
    profiler.mark_executed("runtime_bookkeeping", cpu_wall_elapsed_ms=1.0)
    return profiler.finish(forward_wall_elapsed_ms=1.0, forward_cpu_duration_ms=1.0)


# ---------------------------------------------------------------------------
# TG rung evaluation
# ---------------------------------------------------------------------------


@dataclass
class TgRungObservation:
    """Inputs required to decide whether a TG rung is earned."""

    base_true_tps: float | None
    fallback_count: int | None
    gpu_dispatches: int | None
    p99_stable: bool | None
    same_model: bool
    same_capability_tier: bool
    clean_benchmark: bool
    prompt_dependent_coherent: bool | None
    complete_token_timing: bool
    batch_1_base_runtime: bool


def evaluate_tg_rung(rung: TgRung, obs: TgRungObservation) -> RungStatus:
    """Honest TG-rung gate. Mirrors DSV4F fallback=0 + real GPU + withheld TPS."""
    if obs.base_true_tps is None:
        return RungStatus.BASE_TRUE_TPS_WITHHELD
    if obs.fallback_count is None or obs.fallback_count != 0:
        return RungStatus.REJECT_FALLBACK_NONZERO
    if obs.gpu_dispatches is None or obs.gpu_dispatches <= 0:
        return RungStatus.REJECT_NO_REAL_GPU_DISPATCH
    if not obs.same_model or not obs.same_capability_tier:
        return RungStatus.REJECT_CLAIM_BOUNDARY
    if not obs.clean_benchmark or not obs.complete_token_timing or not obs.batch_1_base_runtime:
        return RungStatus.REJECT_CLAIM_BOUNDARY
    if obs.p99_stable is not True:
        return RungStatus.REJECT_CAPABILITY
    if obs.prompt_dependent_coherent is not True:
        return RungStatus.REJECT_CAPABILITY
    if obs.base_true_tps + 1e-9 < rung.target_tps:
        return RungStatus.REJECT_CAPABILITY
    if rung.id == TG3_REVIEW_THRESHOLD:
        return RungStatus.TG3_REVIEW_REQUIRED
    return RungStatus.PASS_FULL_STACK


def tg_rung_receipt(
    rung: TgRung,
    *,
    family: str,
    status: RungStatus,
    scoreboard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "schema": TG_RUNG_RECEIPT_SCHEMA,
        "status": status.value,
        "family": family,
        "rung": {
            "id": rung.id,
            "target_tps": rung.target_tps,
            "note": rung.note,
            "is_tg3_review_threshold": rung.id == TG3_REVIEW_THRESHOLD,
            "is_final_scientific_target": rung.id == "TG1",
        },
        "requirements": list(TG_RUNG_REQUIREMENTS),
        "scoreboard": scoreboard or empty_separated_scoreboard(
            reason="scaffold — no eligible Qwen BASE_TRUE_TPS run"
        ),
        "claim_boundary": default_claim_boundary(family=family),
        "implementation": {"body_filled": False, "test_stub": True},
    }
    assert_no_blended_tps(body["scoreboard"])
    return seal(body)


# ---------------------------------------------------------------------------
# Gauntlet harness
# ---------------------------------------------------------------------------


@dataclass
class TgGauntletHarness:
    """Self-TG gauntlet scaffold for one model family."""

    family: ModelFamily
    weights_present: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.family, str):
            self.family = ModelFamily(self.family)

    @property
    def family_key(self) -> str:
        return self.family.value

    def loop_steps(self) -> tuple[str, ...]:
        return SELF_TG_LOOP_STEPS

    def ladder(self) -> tuple[TgRung, ...]:
        return TG_LADDER

    def scoreboard(self) -> dict[str, Any]:
        return empty_separated_scoreboard(
            reason=(
                "Qwen weights not local; Proto-Frankenstein offload gate; "
                "no BASE_TRUE_TPS eligibility"
            )
        )

    def stub_rung_receipts(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        board = self.scoreboard()
        for rung in self.ladder():
            status = (
                RungStatus.REJECT_WEIGHTS_ABSENT
                if not self.weights_present
                else RungStatus.BASE_TRUE_TPS_WITHHELD
            )
            out.append(
                tg_rung_receipt(
                    rung,
                    family=self.family_key,
                    status=status,
                    scoreboard=board,
                )
            )
        return out

    def self_tg_loop_receipt(self) -> dict[str, Any]:
        steps = []
        for name in self.loop_steps():
            steps.append(
                {
                    "step": name,
                    "status": RungStatus.SCAFFOLD_PENDING.value,
                    "body_filled": False,
                    "models_cannot_self_promote": True,
                }
            )
        return seal(
            {
                "schema": SELF_TG_LOOP_SCHEMA,
                "status": RungStatus.PASS_SCAFFOLD_CONTRACT.value,
                "family": self.family_key,
                "steps": steps,
                "promotion_authority": "human_or_protected_controller_only",
                "models_cannot_promote_themselves": True,
            }
        )

    def gauntlet_receipt(self) -> dict[str, Any]:
        body = {
            "schema": TG_GAUNTLET_RECEIPT_SCHEMA,
            "status": RungStatus.PASS_SCAFFOLD_CONTRACT.value,
            "family": self.family_key,
            "self_tg_loop": self.self_tg_loop_receipt(),
            "tg_ladder": [
                {"id": r.id, "target_tps": r.target_tps, "note": r.note} for r in self.ladder()
            ],
            "tg3_rule": {
                "threshold_id": TG3_REVIEW_THRESHOLD,
                "target_tps": TG3_TARGET_TPS,
                "action": [
                    "stop_autonomous_promotion",
                    "checkpoint",
                    "seal_complete_evidence",
                    "emit_TG3_REVIEW_REQUIRED",
                    "notify_human",
                ],
                "is_final_target": False,
                "note": "TG3 is a review threshold, not the final scientific target",
            },
            "rung_receipts": self.stub_rung_receipts(),
            "separated_scoreboard": self.scoreboard(),
            "rung_requirements": list(TG_RUNG_REQUIREMENTS),
            "complete_token_profile_scaffold": scaffold_complete_token_profile(self.family),
            "claim_boundary": default_claim_boundary(
                family=self.family_key, weights_present=self.weights_present
            ),
            "honesty": {
                "live_benchmark": False,
                "qwen_download": False,
                "base_true_tps_claimed": False,
                "reason": "harness scaffold only; Qwen not local",
            },
        }
        assert_no_blended_tps(body["separated_scoreboard"])
        return seal(body)


def verify_gauntlet_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return verify(receipt, label="ascension TG gauntlet receipt")


__all__ = [
    "COMPLETE_TOKEN_PROFILE_SCHEMA",
    "CompleteTokenProfiler",
    "MetricCell",
    "SCOREBOARD_SCHEMA",
    "SELF_TG_LOOP_SCHEMA",
    "SELF_TG_LOOP_STEPS",
    "SEPARATED_METRIC_NAMES",
    "SeparatedMetric",
    "StageMetric",
    "TG3_REVIEW_THRESHOLD",
    "TG3_TARGET_TPS",
    "TG_GAUNTLET_RECEIPT_SCHEMA",
    "TG_LADDER",
    "TG_LADDER_BY_ID",
    "TG_RUNG_RECEIPT_SCHEMA",
    "TG_RUNG_REQUIREMENTS",
    "TgGauntletHarness",
    "TgRung",
    "TgRungObservation",
    "assert_no_blended_tps",
    "empty_separated_scoreboard",
    "evaluate_tg_rung",
    "scaffold_complete_token_profile",
    "tg_rung_receipt",
    "verify_gauntlet_receipt",
    "withheld_cell",
]
