"""Command-line surfaces for the provider-neutral AgentOS control plane."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


class _ControlPlaneEngine:
    """Metadata-only engine used by inspection commands; it never generates."""

    def identity(self) -> dict[str, Any]:
        return {"provider": "control-plane", "model_id": "none", "runtime": "inspection"}

    def supports(self, feature: str) -> None:
        del feature
        return None


def _agent(workspace: str, repo_root: Optional[str]) -> Any:
    from hcli.agentos import AgentOS

    return AgentOS(
        Path(workspace).expanduser().resolve(),
        engine=_ControlPlaneEngine(),
        repo_root=Path(repo_root).expanduser().resolve() if repo_root else None,
    )


def _default_repo_root(workspace: str) -> Path:
    for base in (Path.cwd(), Path(__file__).resolve().parents[1]):
        for candidate in (base, *base.parents):
            if (candidate / "hcli").is_dir() and (candidate / "pyproject.toml").is_file():
                return candidate.resolve()
    return Path(workspace).expanduser().resolve()


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _parse_env_assignment(value: str) -> tuple[str, str]:
    key, separator, item = str(value).partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(
            f"expected KEY=VALUE, got {value!r}"
        )
    return key, item


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=os.getcwd(), help="AgentOS workspace root")
    parser.add_argument("--repo-root", default=None, help="repository root for read/evidence surfaces")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcli agentos",
        description="Inspect and operate the durable provider-neutral AgentOS control plane.",
    )
    sub = parser.add_subparsers(dest="command")

    tools = sub.add_parser("tools", help="list typed tools and their permission contracts")
    _add_paths(tools)
    tools.add_argument("--role", default=None)

    status = sub.add_parser("status", help="show mission, provider, tool, and recovery state")
    _add_paths(status)

    checkpoint = sub.add_parser("checkpoint", help="persist an evidence-backed program checkpoint")
    _add_paths(checkpoint)
    checkpoint.add_argument("--network", action="store_true", help="perform bounded public connectivity probes")
    checkpoint.add_argument("--emit", default=None, help="explicit checkpoint output path")

    recovery = sub.add_parser("recovery-gate", help="run the bounded physical fixture recovery gate")
    recovery.add_argument("--workspace", default=None, help="disposable gate workspace; omitted creates one")
    recovery.add_argument("--emit", default=None)
    recovery.add_argument("--timeout-s", type=float, default=30.0)

    research = sub.add_parser("research-gate", help="run the bounded public research gate")
    research.add_argument("--workspace", default=None)
    research.add_argument("--repo-root", default=None)
    research.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    research.add_argument("--search-query", default="Cloudflare Agents SDK official documentation")
    research.add_argument("--emit", default=None)
    research.add_argument("--timeout-s", type=float, default=12.0)

    vmcp = sub.add_parser("vmcp-gate", help="run the callable VMCP evidence-boundary gate")
    vmcp.add_argument("--workspace", default=None)
    vmcp.add_argument("--repo-root", default=None)
    vmcp.add_argument("--search-query", default="VisionMCP public API evidence tools")
    vmcp.add_argument("--emit", default=None)
    vmcp.add_argument("--timeout-s", type=float, default=12.0)

    native = sub.add_parser("native-gate", help="run the live native HCLI A1-A6 reproduction ladder")
    native.add_argument("--workspace", default=None)
    native.add_argument("--repo-root", default=None)
    native.add_argument("--profile", default=None)
    native.add_argument("--prompt", default="Return exactly: HAWKING_OK")
    native.add_argument("--emit", default=None)
    native.add_argument("--timeout-s", type=float, default=180.0)
    native.add_argument("--model-tokens", type=int, default=64)

    resident = sub.add_parser("resident-gate", help="prove one native resident serves sequential requests")
    resident.add_argument("--workspace", default=None)
    resident.add_argument("--repo-root", default=None)
    resident.add_argument("--profile", default=None)
    resident.add_argument("--count", type=int, default=20)
    resident.add_argument("--timeout-s", type=float, default=180.0)
    resident.add_argument("--model-tokens", type=int, default=32)
    resident.add_argument("--emit", default=None)

    mission_gate = sub.add_parser("native-mission-gate", help="run one live native tool/verifier mission")
    mission_gate.add_argument("--repo-root", default=None)
    mission_gate.add_argument("--profile", default=None)
    mission_gate.add_argument("--emit", default=None)

    autonomy = sub.add_parser("autonomy-gate", help="run live AgentOS WorkUnit and crash-recovery qualification")
    autonomy.add_argument("--repo-root", default=None)
    autonomy.add_argument("--profile", default=None)
    autonomy.add_argument("--stage", default="all", help="all, a1, a2, a3/resident-kill, a4/process-kill, or a5/idempotency")
    autonomy.add_argument("--count", type=int, default=10)
    autonomy.add_argument("--emit", default=None)

    unattended = sub.add_parser("unattended-window", help="run bounded real native WorkUnits without human intervention")
    unattended.add_argument("--repo-root", default=None)
    unattended.add_argument("--profile", default=None)
    unattended.add_argument("--workspace", default=None)
    unattended.add_argument("--duration-s", type=float, default=3600.0)
    unattended.add_argument("--interval-s", type=float, default=30.0)
    unattended.add_argument("--emit", default=None)

    accelerator_regression = sub.add_parser(
        "accelerator-regression",
        help="audit the current resident regression with one bounded request",
    )
    accelerator_regression.add_argument("--repo-root", default=None)
    accelerator_regression.add_argument("--profile", default=None)
    accelerator_regression.add_argument("--timeout-s", type=float, default=180.0)
    accelerator_regression.add_argument("--emit", default=None)

    qwen27_runtime_archaeology = sub.add_parser(
        "qwen27-runtime-archaeology",
        help="reconstruct and diff the current Qwen27 runtime identity against the pinned historical receipt",
    )
    qwen27_runtime_archaeology.add_argument("--repo-root", default=None)
    qwen27_runtime_archaeology.add_argument("--profile", default=None)
    qwen27_runtime_archaeology.add_argument("--identity-emit", default=None)
    qwen27_runtime_archaeology.add_argument("--diff-emit", default=None)

    qwen27_mlp_ab = sub.add_parser(
        "qwen27-mlp-ab",
        help="run the bounded source-approved Qwen27 MLP selector diagnostic A/B",
    )
    qwen27_mlp_ab.add_argument("--repo-root", default=None)
    qwen27_mlp_ab.add_argument("--profile", default=None)
    qwen27_mlp_ab.add_argument("--resident-binary", default=None)
    qwen27_mlp_ab.add_argument("--timeout-s", type=float, default=180.0)
    qwen27_mlp_ab.add_argument("--emit", default=None)

    fusion_audit = sub.add_parser(
        "qwen38-fusion-audit",
        help="resolve Qwen3.8 fusion values and source-derived dispatch consequences without GPU work",
    )
    fusion_audit.add_argument("--repo-root", default=None)
    fusion_audit.add_argument("--profile", default=None)
    fusion_audit.add_argument("--emit", default=None)

    modellake = sub.add_parser(
        "modellake-census",
        help="census ModelLake and capture the pinned Flash-Next manifest without downloading",
    )
    modellake.add_argument("--repo-root", default=None)
    modellake.add_argument("--timeout-s", type=float, default=30.0)
    modellake.add_argument("--emit", default=None)

    modellake_supervise = sub.add_parser(
        "modellake-supervise",
        help="observe a supervised pinned ModelLake acquisition without mutating it",
    )
    modellake_supervise.add_argument("--repo-root", default=None)
    modellake_supervise.add_argument("--job-id", required=True)
    modellake_supervise.add_argument("--emit", default=None)

    flash_science = sub.add_parser(
        "flash-science",
        help="inspect pinned Flash-Next metadata and build a pre-runtime organ/Gravity plan",
    )
    flash_science.add_argument("--repo-root", default=None)
    flash_science.add_argument("--timeout-s", type=float, default=30.0)
    flash_science.add_argument("--emit", default=None)

    flash_executable = sub.add_parser(
        "flash-executable",
        help="emit the Flash-Next native executable, EBPW, and complete-token budget scaffolds",
    )
    flash_executable.add_argument("--repo-root", default=None)
    flash_executable.add_argument("--science-receipt", default=None)
    flash_executable.add_argument("--tensor-probe-receipt", default=None)
    flash_executable.add_argument("--representation-experiment-receipt", default=None)
    flash_executable.add_argument("--transform-parity-receipt", default=None)
    flash_executable.add_argument("--loader-roundtrip-receipt", default=None)
    flash_executable.add_argument("--kernel-parity-receipt", default=None)
    flash_executable.add_argument("--shared-expert-kernel-parity-receipt", default=None)
    flash_executable.add_argument("--deltanet-kernel-parity-receipt", default=None)
    flash_executable.add_argument("--sparse-attention-kernel-parity-receipt", default=None)
    flash_executable.add_argument("--mtp-gate-kernel-parity-receipt", default=None)
    flash_executable.add_argument("--graph-component-receipt", default=None)
    flash_executable.add_argument("--component-campaign-receipt", default=None)
    flash_executable.add_argument("--router-graph-receipt", default=None)
    flash_executable.add_argument("--router-selection-receipt", default=None)
    flash_executable.add_argument("--native-router-selection-receipt", default=None)
    flash_executable.add_argument("--native-routed-expert-dispatch-receipt", default=None)
    flash_executable.add_argument("--native-gate-up-swiglu-receipt", default=None)
    flash_executable.add_argument("--native-expert-composition-receipt", default=None)
    flash_executable.add_argument("--native-shared-expert-composition-receipt", default=None)
    flash_executable.add_argument("--native-shared-residual-hyperconnection-receipt", default=None)
    flash_executable.add_argument("--native-exact-hyperconnection-receipt", default=None)
    flash_executable.add_argument("--router-representation-ab-receipt", default=None)
    flash_executable.add_argument("--emit", default=None)
    flash_executable.add_argument("--ebpw-emit", default=None)
    flash_executable.add_argument("--token-ns-emit", default=None)

    flash_tensor_probe = sub.add_parser(
        "flash-tensor-probe",
        help="read a bounded tensor slice from the pinned Flash-Next specimen and compare a derived packed candidate",
    )
    flash_tensor_probe.add_argument("--root", default=None, help="final specimen root; defaults to the canonical ModelLake specimen")
    flash_tensor_probe.add_argument("--tensor-name", default=None)
    flash_tensor_probe.add_argument("--sample-bytes", type=int, default=1 * 1024 * 1024)
    flash_tensor_probe.add_argument("--emit", default=None)

    flash_representation = sub.add_parser(
        "flash-representation-experiment",
        help="run a bounded source-layout-aware routed-expert representation/reference-vector experiment",
    )
    flash_representation.add_argument("--root", default=None, help="final specimen root; defaults to the canonical ModelLake specimen")
    flash_representation.add_argument("--tensor-name", default=None)
    flash_representation.add_argument("--expert-indices", default=None, help="comma-separated expert indices")
    flash_representation.add_argument("--row-start", type=int, default=0)
    flash_representation.add_argument("--row-count", type=int, default=16)
    flash_representation.add_argument("--emit", default=None)

    flash_transform = sub.add_parser(
        "flash-transform-parity",
        help="stream the full pinned routed-expert tensor through derived representations and verify pack/unpack parity",
    )
    flash_transform.add_argument("--root", default=None, help="final specimen root; defaults to the canonical ModelLake specimen")
    flash_transform.add_argument("--tensor-name", default=None)
    flash_transform.add_argument("--chunk-rows", type=int, default=128)
    flash_transform.add_argument("--emit", default=None)

    flash_loader = sub.add_parser(
        "flash-loader-roundtrip",
        help="validate a bounded noetic Flash representation descriptor and source-block loader round-trip",
    )
    flash_loader.add_argument("--root", default=None, help="final specimen root; defaults to the canonical ModelLake specimen")
    flash_loader.add_argument("--repo-root", default=None)
    flash_loader.add_argument("--transform-receipt", default=None)
    flash_loader.add_argument("--tensor-name", default=None)
    flash_loader.add_argument("--candidate", default="independent_q4_g64")
    flash_loader.add_argument("--expert-index", type=int, default=0)
    flash_loader.add_argument("--row-start", type=int, default=0)
    flash_loader.add_argument("--row-count", type=int, default=2)
    flash_loader.add_argument("--emit", default=None)

    flash_body = sub.add_parser(
        "flash-component-body",
        help="persist one bounded source-independent Flash Noetic Q4/G64 component body",
    )
    flash_body.add_argument("--root", default=None, help="final specimen root; defaults to the canonical ModelLake specimen")
    flash_body.add_argument("--repo-root", default=None)
    flash_body.add_argument("--transform-receipt", default=None)
    flash_body.add_argument("--loader-receipt", default=None)
    flash_body.add_argument("--tensor-name", default=None)
    flash_body.add_argument("--candidate", default="independent_q4_g64")
    flash_body.add_argument("--expert-index", type=int, default=0)
    flash_body.add_argument("--row-start", type=int, default=0)
    flash_body.add_argument("--row-count", type=int, default=128)
    flash_body.add_argument("--body", default=None)
    flash_body.add_argument("--emit", default=None)

    flash_matrix = sub.add_parser(
        "flash-matrix-body",
        help="persist one bounded source-independent Noetic Q4/G64 body for a pinned BF16 matrix",
    )
    flash_matrix.add_argument("--root", default=None, help="final specimen root; defaults to the canonical ModelLake specimen")
    flash_matrix.add_argument("--repo-root", default=None)
    flash_matrix.add_argument("--tensor-name", default="model.language_model.layers.0.mlp.gate.weight")
    flash_matrix.add_argument("--candidate", default="independent_q4_g64")
    flash_matrix.add_argument("--component-kind", default="router")
    flash_matrix.add_argument("--row-start", type=int, default=0)
    flash_matrix.add_argument("--row-count", type=int, default=128)
    flash_matrix.add_argument("--body", default=None)
    flash_matrix.add_argument("--emit", default=None)

    flash_vector = sub.add_parser(
        "flash-vector-body",
        help="persist one exact source BF16 vector for a Flash Noetic boundary",
    )
    flash_vector.add_argument("--root", default=None, help="final specimen root; defaults to the canonical ModelLake specimen")
    flash_vector.add_argument("--repo-root", default=None)
    flash_vector.add_argument("--tensor-name", default="model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight")
    flash_vector.add_argument("--candidate", default="source_bf16_exact")
    flash_vector.add_argument("--component-kind", default="mlp_hyperconnection_hc_norm")
    flash_vector.add_argument("--body", default=None)
    flash_vector.add_argument("--emit", default=None)

    flash_router_graph = sub.add_parser(
        "flash-router-graph",
        help="compile the bounded source-independent Flash router matrix into a Noetic graph",
    )
    flash_router_graph.add_argument("--repo-root", default=None)
    flash_router_graph.add_argument("--body-receipt", default=None)
    flash_router_graph.add_argument("--kernel-receipt", default=None)
    flash_router_graph.add_argument("--emit", default=None)

    flash_router_selection = sub.add_parser(
        "flash-router-selection",
        help="execute bounded Flash router FP32 softmax/top-k semantics over a persisted full Noetic body",
    )
    flash_router_selection.add_argument("--repo-root", default=None)
    flash_router_selection.add_argument("--root", default=None, help="pinned ModelLake specimen root; defaults to the body receipt root")
    flash_router_selection.add_argument("--body-receipt", default=None)
    flash_router_selection.add_argument("--kernel-receipt", default=None)
    flash_router_selection.add_argument("--emit", default=None)

    flash_router_representation_ab = sub.add_parser(
        "flash-router-representation-ab",
        help="compare bounded Flash router representations against pinned source top-k routing",
    )
    flash_router_representation_ab.add_argument("--repo-root", default=None)
    flash_router_representation_ab.add_argument("--root", default=None)
    flash_router_representation_ab.add_argument("--tensor-name", default="model.language_model.layers.0.mlp.gate.weight")
    flash_router_representation_ab.add_argument("--emit", default=None)

    flash_campaign = sub.add_parser(
        "flash-component-campaign",
        help="compose multiple persisted source-independent Flash Noetic components into one bounded campaign graph",
    )
    flash_campaign.add_argument("--repo-root", default=None)
    flash_campaign.add_argument("--loader-receipt", default=None)
    flash_campaign.add_argument("--transform-receipt", default=None)
    flash_campaign.add_argument("--component", action="append", default=None, help="BODY_RECEIPT,KERNEL_RECEIPT; repeat for additional bounded components")
    flash_campaign.add_argument("--emit", default=None)

    flash_graph = sub.add_parser(
        "flash-graph-component",
        help="compile the bounded Flash Noetic routed-expert graph component from validated receipts",
    )
    flash_graph.add_argument("--repo-root", default=None)
    flash_graph.add_argument("--loader-receipt", default=None)
    flash_graph.add_argument("--kernel-receipt", default=None)
    flash_graph.add_argument("--transform-receipt", default=None)
    flash_graph.add_argument("--emit", default=None)

    preboard = sub.add_parser(
        "preboard",
        help="census negative science and define the FPGA/compiler preboard boundary",
    )
    preboard.add_argument("--repo-root", default=None)
    preboard.add_argument("--emit", default=None)

    charge = sub.add_parser(
        "initial-charge",
        help="create or inspect the durable provider-neutral Hawking initial charge",
    )
    charge.add_argument("--repo-root", default=None)
    charge.add_argument("--workspace", default=None)
    charge.add_argument("--emit", default=None)
    charge.add_argument("--force", action="store_true")

    science_maps = sub.add_parser(
        "science-maps",
        help="build the receipt-first two-Qwen transfer and Flash precedent maps",
    )
    science_maps.add_argument("--repo-root", default=None)
    science_maps.add_argument("--transfer-emit", default=None)
    science_maps.add_argument("--precedent-emit", default=None)

    ab_scaffold = sub.add_parser(
        "ab-scaffold",
        help="emit the dense-versus-NF organ/full-model A/B protocol scaffold",
    )
    ab_scaffold.add_argument("--repo-root", default=None)
    ab_scaffold.add_argument("--emit", default=None)

    fpga_preboard = sub.add_parser(
        "fpga-preboard",
        help="build Qwen27 and Flash-Next FPGA/HWIR pre-board maps",
    )
    fpga_preboard.add_argument("--repo-root", default=None)
    fpga_preboard.add_argument("--emit", default=None)

    protected_bench_watch = sub.add_parser(
        "protected-bench-watch",
        help="wait for a safe protected Qwen window and run the bounded diagnostic",
    )
    protected_bench_watch.add_argument("--repo-root", default=None)
    protected_bench_watch.add_argument("--profile", default=None)
    protected_bench_watch.add_argument("--resident-binary", default=None)
    protected_bench_watch.add_argument("--emit", default=None)
    protected_bench_watch.add_argument("--result-emit", default=None)
    protected_bench_watch.add_argument("--duration-s", type=float, default=6 * 3600.0)
    protected_bench_watch.add_argument("--interval-s", type=float, default=60.0)
    protected_bench_watch.add_argument("--timeout-s", type=float, default=180.0)
    protected_bench_watch.add_argument("--once", action="store_true")
    protected_bench_watch.add_argument("--pause-known-jobs", action="store_true")

    protected_accelerator = sub.add_parser(
        "protected-accelerator-bench",
        help="run one provider-neutral protected persistent-resident physical benchmark",
    )
    protected_accelerator.add_argument("--repo-root", default=None)
    protected_accelerator.add_argument("--profile", default=None)
    protected_accelerator.add_argument("--resident-binary", default=None)
    protected_accelerator.add_argument(
        "--fusion-env",
        action="append",
        default=[],
        type=_parse_env_assignment,
        metavar="KEY=VALUE",
        help="child-only fusion environment override; repeat for multiple controls",
    )
    protected_accelerator.add_argument("--prompt", default="Return exactly: HAWKING_OK")
    protected_accelerator.add_argument("--warmup-requests", type=int, default=1)
    protected_accelerator.add_argument("--measure-requests", type=int, default=5)
    protected_accelerator.add_argument("--max-new-tokens", type=int, default=32)
    protected_accelerator.add_argument("--ready-timeout-s", type=float, default=6 * 3600.0)
    protected_accelerator.add_argument("--interval-s", type=float, default=30.0)
    protected_accelerator.add_argument("--timeout-s", type=float, default=180.0)
    protected_accelerator.add_argument("--emit", default=None)

    handoff = sub.add_parser(
        "handoff",
        help="write the resumable overnight Hawking status and continuation handoff",
    )
    handoff.add_argument("--repo-root", default=None)
    handoff.add_argument("--emit", default=None)

    background = sub.add_parser("background", help="inspect or manage durable shell-free background jobs")
    bgsub = background.add_subparsers(dest="background_command")
    bg_list = bgsub.add_parser("list", help="list persisted jobs")
    _add_paths(bg_list)
    bg_status = bgsub.add_parser("status", help="inspect one persisted job")
    _add_paths(bg_status)
    bg_status.add_argument("job_id")
    bg_start = bgsub.add_parser("start", help="start argv directly without a shell")
    _add_paths(bg_start)
    bg_start.add_argument("--cwd", default=None)
    bg_start.add_argument("--label", default=None)
    bg_start.add_argument("--timeout-s", type=float, default=None)
    bg_start.add_argument("--non-resumable", action="store_true")
    bg_start.add_argument("argv", nargs=argparse.REMAINDER, help="argv after --")
    bg_resume = bgsub.add_parser("resume", help="rerun an interrupted resumable job")
    _add_paths(bg_resume)
    bg_resume.add_argument("job_id")
    bg_cancel = bgsub.add_parser("cancel", help="cancel one running job")
    _add_paths(bg_cancel)
    bg_cancel.add_argument("job_id")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command is None:
        build_parser().print_help()
        return 0
    try:
        if args.command == "recovery-gate":
            from hcli.agentos.recovery import run_recovery_gate

            report = run_recovery_gate(
                args.workspace,
                emit=args.emit,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "research-gate":
            from hcli.agentos.research import run_research_gate

            report = run_research_gate(
                args.workspace,
                repo_root=args.repo_root,
                repo=args.repo,
                search_query=args.search_query,
                emit=args.emit,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "vmcp-gate":
            from hcli.agentos.vmcp_gate import run_vmcp_gate

            report = run_vmcp_gate(
                args.workspace,
                repo_root=args.repo_root,
                emit=args.emit,
                search_query=args.search_query,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "native-gate":
            from hcli.agentos.native_gate import run_native_gate

            report = run_native_gate(
                args.workspace,
                repo_root=args.repo_root,
                profile=args.profile,
                prompt=args.prompt,
                emit=args.emit,
                timeout_s=args.timeout_s,
                model_tokens=args.model_tokens,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "resident-gate":
            from hcli.agentos.resident_gate import run_resident_gate

            report = run_resident_gate(
                args.workspace,
                repo_root=args.repo_root,
                profile=args.profile,
                count=args.count,
                timeout_s=args.timeout_s,
                model_tokens=args.model_tokens,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "native-mission-gate":
            from hcli.agentos.native_mission_gate import run_native_mission_gate

            report = run_native_mission_gate(
                repo_root=args.repo_root,
                profile=args.profile,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "autonomy-gate":
            from hcli.agentos.autonomy_gate import run_autonomy_gate

            report = run_autonomy_gate(
                repo_root=args.repo_root,
                profile=args.profile,
                emit=args.emit,
                stage=args.stage,
                count=args.count,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "unattended-window":
            from hcli.agentos.autonomy_gate import run_unattended_window

            report = run_unattended_window(
                repo_root=args.repo_root,
                profile=args.profile,
                workspace=args.workspace,
                emit=args.emit,
                duration_s=args.duration_s,
                interval_s=args.interval_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "accelerator-regression":
            from hcli.agentos.accelerator_regression import run_accelerator_regression

            report = run_accelerator_regression(
                repo_root=args.repo_root,
                profile=args.profile,
                emit=args.emit,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "qwen27-runtime-archaeology":
            from hcli.agentos.qwen27_runtime_identity import run_runtime_archaeology

            report = run_runtime_archaeology(
                repo_root=args.repo_root,
                profile=args.profile,
                identity_emit=args.identity_emit,
                diff_emit=args.diff_emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "qwen27-mlp-ab":
            from hcli.agentos.qwen27_mlp_diagnostic import run_qwen27_mlp_diagnostic_ab

            report = run_qwen27_mlp_diagnostic_ab(
                repo_root=args.repo_root,
                profile=args.profile,
                resident_binary=args.resident_binary,
                fusion_env_overrides=dict(args.fusion_env),
                emit=args.emit,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "qwen38-fusion-audit":
            from hcli.agentos.qwen38_fusion_audit import run_qwen38_fusion_source_audit

            report = run_qwen38_fusion_source_audit(
                repo_root=args.repo_root,
                profile=args.profile,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "modellake-census":
            from hcli.agentos.modellake_gate import run_modellake_census

            report = run_modellake_census(
                repo_root=args.repo_root,
                emit=args.emit,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "modellake-supervise":
            from hcli.agentos.modellake_supervisor import run_model_lake_supervision

            report = run_model_lake_supervision(
                repo_root=args.repo_root,
                job_id=args.job_id,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") in {"PASSED", "RUNNING_SAFE", "WAITING_OR_NOT_OBSERVED"} else 1

        if args.command == "flash-science":
            from hcli.agentos.flash_science import run_flash_science_gate

            report = run_flash_science_gate(
                repo_root=args.repo_root,
                emit=args.emit,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-executable":
            from hcli.agentos.flash_executable import run_flash_executable_scaffold

            report = run_flash_executable_scaffold(
                repo_root=args.repo_root,
                science_receipt=args.science_receipt,
                tensor_probe_receipt=args.tensor_probe_receipt,
                representation_experiment_receipt=args.representation_experiment_receipt,
                transform_parity_receipt=args.transform_parity_receipt,
                loader_roundtrip_receipt=args.loader_roundtrip_receipt,
                kernel_parity_receipt=args.kernel_parity_receipt,
                shared_expert_kernel_parity_receipt=args.shared_expert_kernel_parity_receipt,
                deltanet_kernel_parity_receipt=args.deltanet_kernel_parity_receipt,
                sparse_attention_kernel_parity_receipt=args.sparse_attention_kernel_parity_receipt,
                mtp_gate_kernel_parity_receipt=args.mtp_gate_kernel_parity_receipt,
                graph_component_receipt=args.graph_component_receipt,
                component_campaign_receipt=args.component_campaign_receipt,
                router_graph_receipt=args.router_graph_receipt,
                router_selection_receipt=args.router_selection_receipt,
                native_router_selection_receipt=args.native_router_selection_receipt,
                native_routed_expert_dispatch_receipt=args.native_routed_expert_dispatch_receipt,
                native_gate_up_swiglu_receipt=args.native_gate_up_swiglu_receipt,
                native_expert_composition_receipt=args.native_expert_composition_receipt,
                native_shared_expert_composition_receipt=args.native_shared_expert_composition_receipt,
                native_shared_residual_hyperconnection_receipt=args.native_shared_residual_hyperconnection_receipt,
                native_exact_hyperconnection_receipt=args.native_exact_hyperconnection_receipt,
                router_representation_ab_receipt=args.router_representation_ab_receipt,
                emit=args.emit,
                ebpw_emit=args.ebpw_emit,
                token_ns_emit=args.token_ns_emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-tensor-probe":
            from hcli.agentos.flash_tensor_probe import DEFAULT_TENSOR, run_flash_tensor_probe

            report = run_flash_tensor_probe(
                root=args.root,
                tensor_name=args.tensor_name or DEFAULT_TENSOR,
                sample_bytes=args.sample_bytes,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-representation-experiment":
            from hcli.agentos.flash_representation_experiment import (
                DEFAULT_EXPERT_INDICES,
                DEFAULT_TENSOR,
                run_flash_representation_experiment,
            )

            report = run_flash_representation_experiment(
                root=args.root,
                tensor_name=args.tensor_name or DEFAULT_TENSOR,
                expert_indices=args.expert_indices if args.expert_indices is not None else DEFAULT_EXPERT_INDICES,
                row_start=args.row_start,
                row_count=args.row_count,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-transform-parity":
            from hcli.agentos.flash_transform_parity import DEFAULT_TENSOR, run_flash_transform_parity

            report = run_flash_transform_parity(
                root=args.root,
                tensor_name=args.tensor_name or DEFAULT_TENSOR,
                chunk_rows=args.chunk_rows,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-loader-roundtrip":
            from hcli.agentos.flash_loader_roundtrip import DEFAULT_TENSOR, run_flash_loader_roundtrip

            report = run_flash_loader_roundtrip(
                root=args.root,
                repo_root=args.repo_root,
                transform_receipt=args.transform_receipt,
                tensor_name=args.tensor_name or DEFAULT_TENSOR,
                candidate_id=args.candidate,
                expert_index=args.expert_index,
                row_start=args.row_start,
                row_count=args.row_count,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-component-body":
            from hcli.agentos.flash_component_body import DEFAULT_TENSOR, run_flash_component_body

            report = run_flash_component_body(
                root=args.root,
                repo_root=args.repo_root,
                transform_receipt=args.transform_receipt,
                loader_receipt=args.loader_receipt,
                tensor_name=args.tensor_name or DEFAULT_TENSOR,
                candidate_id=args.candidate,
                expert_index=args.expert_index,
                row_start=args.row_start,
                row_count=args.row_count,
                body=args.body,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-matrix-body":
            from hcli.agentos.flash_matrix_component_body import run_flash_matrix_component_body

            report = run_flash_matrix_component_body(
                root=args.root,
                repo_root=args.repo_root,
                tensor_name=args.tensor_name,
                candidate_id=args.candidate,
                component_kind=args.component_kind,
                row_start=args.row_start,
                row_count=args.row_count,
                body=args.body,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-vector-body":
            from hcli.agentos.flash_vector_component_body import run_flash_vector_body

            report = run_flash_vector_body(
                root=args.root,
                repo_root=args.repo_root,
                tensor_name=args.tensor_name,
                candidate_id=args.candidate,
                component_kind=args.component_kind,
                body=args.body,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-router-graph":
            from hcli.agentos.flash_router_graph import run_flash_router_graph

            report = run_flash_router_graph(
                repo_root=args.repo_root,
                body_receipt=args.body_receipt,
                kernel_receipt=args.kernel_receipt,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-router-selection":
            from hcli.agentos.flash_router_selection import run_flash_router_selection

            report = run_flash_router_selection(
                repo_root=args.repo_root,
                root=args.root,
                body_receipt=args.body_receipt,
                kernel_receipt=args.kernel_receipt,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-router-representation-ab":
            from hcli.agentos.flash_router_representation_ab import run_flash_router_representation_ab

            report = run_flash_router_representation_ab(
                repo_root=args.repo_root,
                root=args.root,
                tensor_name=args.tensor_name,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-component-campaign":
            from hcli.agentos.flash_component_campaign import run_flash_component_campaign

            report = run_flash_component_campaign(
                repo_root=args.repo_root,
                loader_receipt=args.loader_receipt,
                transform_receipt=args.transform_receipt,
                component_specs=args.component,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "flash-graph-component":
            from hcli.agentos.flash_graph_component import run_flash_graph_component

            report = run_flash_graph_component(
                repo_root=args.repo_root,
                loader_receipt=args.loader_receipt,
                kernel_receipt=args.kernel_receipt,
                transform_receipt=args.transform_receipt,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "preboard":
            from hcli.agentos.preboard import run_preboard

            report = run_preboard(
                repo_root=args.repo_root,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "initial-charge":
            from hcli.agentos.charge import create_initial_charge

            report = create_initial_charge(
                repo_root=args.repo_root,
                workspace=args.workspace,
                emit=args.emit,
                force=args.force,
            )
            _emit(report)
            return 0

        if args.command == "science-maps":
            from hcli.agentos.science_maps import write_science_maps

            report = write_science_maps(
                repo_root=args.repo_root,
                transfer_emit=args.transfer_emit,
                precedent_emit=args.precedent_emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "ab-scaffold":
            from hcli.agentos.representation_ab import run_ab_scaffold

            report = run_ab_scaffold(repo_root=args.repo_root, emit=args.emit)
            _emit(report)
            return 0 if report.get("status") == "READY_SCAFFOLD" else 1

        if args.command == "fpga-preboard":
            from hcli.agentos.fpga_preboard import run_fpga_preboard

            report = run_fpga_preboard(repo_root=args.repo_root, emit=args.emit)
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "protected-bench-watch":
            from hcli.agentos.protected_benchmark_watcher import run_protected_benchmark_watcher

            report = run_protected_benchmark_watcher(
                repo_root=args.repo_root,
                profile=args.profile,
                resident_binary=args.resident_binary,
                emit=args.emit,
                result_emit=args.result_emit,
                duration_s=args.duration_s,
                interval_s=args.interval_s,
                once=args.once,
                pause_known_jobs=args.pause_known_jobs,
                timeout_s=args.timeout_s,
            )
            _emit(report)
            return 0 if report.get("status") in {"COMPLETED", "WAITING_FOR_QUIESCENCE"} else 1

        if args.command == "protected-accelerator-bench":
            from hcli.agentos.protected_accelerator_benchmark import run_protected_accelerator_benchmark

            report = run_protected_accelerator_benchmark(
                repo_root=args.repo_root,
                profile=args.profile,
                resident_binary=args.resident_binary,
                fusion_env_overrides=dict(args.fusion_env),
                prompt=args.prompt,
                warmup_requests=args.warmup_requests,
                measure_requests=args.measure_requests,
                max_new_tokens=args.max_new_tokens,
                ready_timeout_s=args.ready_timeout_s,
                interval_s=args.interval_s,
                timeout_s=args.timeout_s,
                emit=args.emit,
            )
            _emit(report)
            return 0 if report.get("status") == "PASSED" else 1

        if args.command == "handoff":
            from hcli.agentos.handoff import build_handoff

            report = build_handoff(args.repo_root, emit=args.emit)
            _emit(report)
            return 0

        if args.command == "checkpoint":
            from hcli.agentos.checkpoint import write_program_checkpoint

            report = write_program_checkpoint(
                args.repo_root or _default_repo_root(args.workspace),
                workspace=args.workspace,
                emit=args.emit,
                network=args.network,
            )
            _emit(report)
            return 0

        agent = _agent(args.workspace, args.repo_root)
        if args.command == "tools":
            _emit({"schema": "hcli.agentos.tool_catalog.v1", "tools": agent.tools.discover(role=args.role)})
            return 0
        if args.command == "status":
            _emit(agent.status())
            return 0
        if args.command == "background":
            command = args.background_command
            if command == "list":
                _emit({"schema": "hcli.agentos.background_catalog.v1", "jobs": agent.background_jobs()})
                return 0
            if command == "status":
                _emit(agent.background_status(args.job_id))
                return 0
            if command == "start":
                argv_value = list(args.argv)
                if argv_value and argv_value[0] == "--":
                    argv_value = argv_value[1:]
                if not argv_value:
                    raise ValueError("background start requires argv after --")
                _emit(agent.start_background(
                    argv_value,
                    cwd=args.cwd,
                    label=args.label,
                    resumable=not args.non_resumable,
                    timeout_s=args.timeout_s,
                ))
                return 0
            if command == "resume":
                _emit(agent.resume_background(args.job_id))
                return 0
            if command == "cancel":
                _emit(agent.cancel_background(args.job_id))
                return 0
            build_parser().parse_args(["background", "--help"])
            return 0
    except Exception as exc:  # CLI surfaces must leave a machine-readable failure.
        _emit({
            "schema": "hcli.agentos.cli_error.v1",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        return 1
    return 0


__all__ = ["build_parser", "main"]
