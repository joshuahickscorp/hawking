#!/usr/bin/env python3.12
"""Unified command dispatcher for Hawking condensation tooling."""

from __future__ import annotations

import importlib
import inspect
import pathlib
import sys


COMMANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "appendix.runtime": ("appendix_runtime", ("profile",)),
    "appendix.profile": ("appendix_runtime", ("profile",)),
    "appendix.requirements": ("appendix_runtime", ("requirements",)),
    "appendix.gate": ("appendix_physical_evidence_gate", ()),
    "appendix.authority": ("appendix_physical_counter_authority", ()),
    "frontier.runtime": ("frontier_runtime", ("profile",)),
    "frontier.profile": ("frontier_runtime", ("profile",)),
    "frontier.status": ("frontier_runtime", ("status",)),
    "frontier.conductor": ("frontier_runtime", ("conductor",)),
    "frontier.autopilot": ("frontier_runtime", ("autopilot",)),
    "frontier.verifier": ("frontier_runtime", ("verifier",)),
    "frontier.launcher": ("frontier_runtime", ("launcher",)),
    "frontier.ops": ("frontier_ops", ()),
    "core.profile": ("condense_profiles", ("list",)),
    "core.preflight": ("preflight", ()),
    "core.environment": ("studio_environment", ()),
    "core.sweep": ("sweep", ()),
    "doctor.profiles": ("doctor_v5_profiles", ()),
    "doctor.report": ("doctor_v5_campaign_report", ()),
    "doctor.telegram": ("doctor_v5_telegram_rung_notifier", ()),
    "doctor.physical-controller": ("doctor_v5_physical_ab_controller", ()),
    "doctor.physical-executor": ("doctor_v5_physical_ab_executor", ()),
}

for legacy, alias in {
    "appendix_catalog": "appendix.catalog",
    "appendix_contract": "appendix.contract",
    "appendix_corpus": "appendix.corpus",
    "appendix_handoff": "appendix.handoff",
    "appendix_ledger": "appendix.ledger",
    "appendix_postrun": "appendix.postrun",
    "appendix_scaffold": "appendix.scaffold",
}.items():
    COMMANDS[alias] = ("condense_profiles", ("run", legacy))

for profile in (
    "supervisor", "admission", "recovery", "reporting", "methodology",
    "block_parallel", "post120", "physical", "notification",
):
    COMMANDS[f"doctor.{profile.replace('_', '-')}"] = (
        "doctor_v5_profiles", ("show", profile),
    )

for legacy in """
doctor_v5_acceleration_eta doctor_v5_acceleration_reentry
doctor_v5_aggressive_admission_policy doctor_v5_audit
doctor_v5_block_parallel_config_matrix doctor_v5_block_parallel_real_canary
doctor_v5_blocked_cell_recovery doctor_v5_condenser_mountain_methodology
doctor_v5_distributed_transport doctor_v5_elastic_phase_scheduler
doctor_v5_fixture_phase_validator doctor_v5_forward_recovery
doctor_v5_gptoss_execution_thread_contract doctor_v5_gptoss_parallel_scaffold
doctor_v5_gptoss_reuse_fanout doctor_v5_gptoss_tokenizer_gate
doctor_v5_higher_tier_authority doctor_v5_higher_tier_scaffold
doctor_v5_host_sprint_plan doctor_v5_inert_phase_launcher
doctor_v5_mountain_ladder doctor_v5_pass_b_bootstrap doctor_v5_pass_b_queue
doctor_v5_post120_acceleration_scaffold doctor_v5_production_eta
doctor_v5_queue doctor_v5_qwen_shard_window
doctor_v5_qwen_thread_profile_runner doctor_v5_remaining_scratch_gate_adapter
doctor_v5_resource_stop_recovery_stage doctor_v5_root
doctor_v5_shared_preprocess_cache doctor_v5_single_device_benchmark
doctor_v5_single_device_sprint_audit doctor_v5_strand_control_adapter
doctor_v5_streaming_source doctor_v5_ultra_aggressive_autoresume
doctor_v5_ultra_aggressive_queue doctor_v5_ultra_autoresume
""".split():
    target = ("doctor_v5_profiles", ("compat", legacy))
    COMMANDS[legacy] = target
    COMMANDS[f"doctor.{legacy.removeprefix('doctor_v5_').replace('_', '-')}"] = target


def _usage() -> str:
    commands = "\n".join(f"  {name}" for name in sorted(COMMANDS))
    return (
        "usage: python -m tools.condense COMMAND [ARGS...]\n"
        "       python -m tools.condense legacy MODULE [ARGS...]\n\n"
        f"commands:\n{commands}"
    )


def _invoke(module_name: str, prefix: tuple[str, ...], args: list[str]) -> int:
    module = importlib.import_module(f"tools.condense.{module_name}")
    entry = module.main
    values = [*prefix, *args]
    parameters = inspect.signature(entry).parameters
    if parameters:
        return int(entry(values) or 0)
    prior = sys.argv
    sys.argv = [str(pathlib.Path(module.__file__).resolve()), *values]
    try:
        return int(entry() or 0)
    finally:
        sys.argv = prior


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    if args[0] == "--list":
        print("\n".join(sorted(COMMANDS)))
        return 0
    command = args.pop(0)
    local = pathlib.Path(__file__).resolve().parent
    if str(local) not in sys.path:
        sys.path.insert(0, str(local))
    if command == "legacy":
        if not args:
            print("legacy requires MODULE", file=sys.stderr)
            return 64
        return _invoke("condense_profiles", ("run", args.pop(0)), args)
    command_spec = COMMANDS.get(command)
    if command_spec is None:
        print(f"unknown command: {command}\n\n{_usage()}", file=sys.stderr)
        return 64
    module_name, prefix = command_spec
    return _invoke(module_name, prefix, args)


if __name__ == "__main__":
    raise SystemExit(main())
