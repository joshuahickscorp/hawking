#!/usr/bin/env python3.12
"""Shared Studio hardware + frontier model manifest.

The frontier scripts used to duplicate the same Studio and model-size facts in
several files. This module is deliberately tiny and stdlib-only so every tool can
import the same hardware envelope, model labels, download sizes, and serve-fit
math without creating a dependency cycle.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    ram_gb: float
    weight_budget_gb: float
    process_budget_gb: float
    ssd_tb: float
    ssd_read_gbps: float
    ram_gbps: float
    disk_reserve_gb: float
    scratch_reserve_gb: float
    cache_reserve_gb: float

    @property
    def ssd_gb(self) -> float:
        return self.ssd_tb * 1000.0

    @property
    def storage_budget_gb(self) -> float:
        """Maximum live payload on the internal SSD after the hard safety reserve."""
        return max(0.0, self.ssd_gb - self.disk_reserve_gb)


M3_ULTRA_96GB = HardwareProfile(
    name="Studio-M3Ultra-96",
    ram_gb=96.0,                 # GiB, as sold/reported by macOS
    weight_budget_gb=78.0,       # interactive resident-weight envelope
    process_budget_gb=78.0,      # aggregate scheduled-job estimate envelope
    ssd_tb=1.0,
    ssd_read_gbps=6.0,
    ram_gbps=819.0,
    disk_reserve_gb=150.0,
    scratch_reserve_gb=64.0,
    cache_reserve_gb=32.0,
)
DEFAULT_HARDWARE = M3_ULTRA_96GB


@dataclass(frozen=True)
class FrontierModel:
    label: str
    hf_id: str
    local_dir: str
    total_b: float
    active_b: float | None
    serve_bpw: float
    moe: bool
    role: str
    download_gb: float
    source_kind: str
    note: str = ""
    aliases: tuple[str, ...] = ()

    def artifact_gb(self) -> float:
        return self.total_b * self.serve_bpw / 8.0

    def q4k_gb(self, q4k_bpw: float = 4.5) -> float:
        return self.total_b * q4k_bpw / 8.0

    def fits_resident(self, hw: HardwareProfile = DEFAULT_HARDWARE) -> bool:
        return self.artifact_gb() <= hw.weight_budget_gb

    def fits_storage(self, hw: HardwareProfile = DEFAULT_HARDWARE) -> bool:
        return self.download_gb + self.artifact_gb() <= hw.storage_budget_gb


# Exact labels are stable CLI labels. Aliases are accepted by procure/studio_run.
# Sizes are decimal GB, matching Hugging Face dry-run output and the repo's existing
# estimate convention. DeepSeek-V4, GLM-5.2, and the Kimi-K2 series were checked on 2026-07-08.
FRONTIER_MODELS: tuple[FrontierModel, ...] = (
    FrontierModel(
        label="235B-A22B",
        hf_id="Qwen/Qwen3-235B-A22B",
        local_dir="scratch/qwen3-235b-a22b",
        total_b=235.0,
        active_b=22.0,
        serve_bpw=1.34,
        moe=True,
        role="moe-resident",
        download_gb=470.0,
        source_kind="bf16 parent",
        note="comfy resident MoE reference",
    ),
    FrontierModel(
        label="405B",
        hf_id="meta-llama/Llama-3.1-405B-Instruct",
        local_dir="scratch/llama31-405b",
        total_b=405.0,
        active_b=None,
        serve_bpw=1.34,
        moe=False,
        role="dense-resident",
        download_gb=810.0,
        source_kind="bf16 parent (gated)",
        note="dense resident cliff reference",
    ),
    FrontierModel(
        label="671B",
        hf_id="deepseek-ai/DeepSeek-V3",
        local_dir="scratch/deepseek-v3",
        total_b=671.0,
        active_b=37.0,
        serve_bpw=1.00,
        moe=True,
        role="moe-capstone",
        download_gb=1342.0,
        source_kind="bf16 parent",
        note="resident 1.0-bpw prize target",
    ),
    FrontierModel(
        label="DeepSeek-V4-Flash",
        hf_id="deepseek-ai/DeepSeek-V4-Flash-DSpark",
        local_dir="scratch/deepseek-v4-flash-dspark",
        total_b=284.0,
        active_b=13.0,
        serve_bpw=1.34,
        moe=True,
        role="moe-longctx-flash",
        download_gb=168.0,
        source_kind="fp4+fp8 mixed DSpark checkpoint",
        note="DeepSeek V4 Flash; 1M context, 284B total / 13B active, DSpark module attached",
        aliases=("DS-V4-Flash", "DeepSeek-V4-Flash-DSpark", "V4-Flash", "DSV4-Flash"),
    ),
    FrontierModel(
        label="DeepSeek-V4-Pro",
        hf_id="deepseek-ai/DeepSeek-V4-Pro-DSpark",
        local_dir="scratch/deepseek-v4-pro-dspark",
        total_b=1600.0,
        active_b=49.0,
        serve_bpw=0.50,
        moe=True,
        role="moe-mega-frontier",
        download_gb=894.0,
        source_kind="fp4+fp8 mixed DSpark checkpoint",
        note="DeepSeek V4 Pro; 1.6T total / 49B active, resident only at a 0.50-bpw research rung",
        aliases=("DS-V4-Pro", "DeepSeek-V4-Pro-DSpark", "V4-Pro", "DSV4-Pro"),
    ),
    FrontierModel(
        label="GLM-5.2",
        hf_id="zai-org/GLM-5.2",
        local_dir="scratch/glm-5.2",
        total_b=753.0,
        active_b=39.0,
        serve_bpw=1.00,
        moe=True,
        role="moe-capstone",
        download_gb=1517.0,
        source_kind="bf16 parent",
        note="latest GLM-5 representative; 1M-context model-card lane",
        aliases=("GLM5.2", "GLM-5", "glm5", "glm"),
    ),
    FrontierModel(
        label="Kimi-K2.6",
        hf_id="moonshotai/Kimi-K2.6",
        local_dir="scratch/kimi-k2.6",
        total_b=1100.0,
        active_b=32.0,
        serve_bpw=0.75,
        moe=True,
        role="moe-mega-stretch",
        download_gb=595.0,
        source_kind="compressed-tensors checkpoint",
        note="Moonshot K2-series representative; 0.75 bpw is the resident stretch rung",
        aliases=("K2", "Kimi-K2", "KimiK2", "Kimi-K2.6"),
    ),
    FrontierModel(
        label="Kimi-K2.7-Code",
        hf_id="moonshotai/Kimi-K2.7-Code",
        local_dir="scratch/kimi-k2.7-code",
        total_b=1100.0,
        active_b=32.0,
        serve_bpw=0.75,
        moe=True,
        role="moe-mega-code",
        download_gb=595.0,
        source_kind="compressed-tensors checkpoint",
        note="coding-focused K2-series control built on K2.6; native INT4/compressed source",
        aliases=("Kimi-K2.7", "K2.7", "K2-Code", "K2.7-Code", "Kimi-Code"),
    ),
    FrontierModel(
        label="Kimi-K2-Instruct",
        hf_id="moonshotai/Kimi-K2-Instruct",
        local_dir="scratch/kimi-k2-instruct",
        total_b=1000.0,
        active_b=32.0,
        serve_bpw=0.75,
        moe=True,
        role="moe-mega-text",
        download_gb=1031.0,
        source_kind="public text MoE checkpoint",
        note="text-only K2 control if K2.6 multimodal/custom-code path is inconvenient",
        aliases=("Kimi-K2-Text", "K2-Instruct"),
    ),
)


def tq_gb(params_b: float, bpw: float) -> float:
    return params_b * bpw / 8.0


def frontier_by_label(label_or_id: str) -> FrontierModel | None:
    needle = label_or_id.lower()
    for model in FRONTIER_MODELS:
        names = (model.label, model.hf_id, *model.aliases)
        if any(needle == n.lower() for n in names):
            return model
    return None


def frontier_labels() -> list[str]:
    return [m.label for m in FRONTIER_MODELS]


def total_download_gb(models: tuple[FrontierModel, ...] = FRONTIER_MODELS) -> float:
    return sum(m.download_gb for m in models)


def total_artifact_gb(models: tuple[FrontierModel, ...] = FRONTIER_MODELS) -> float:
    return sum(m.artifact_gb() for m in models)


def eta_hours(gb: float, link_mb_s: float, efficiency: float = 1.0) -> float | None:
    effective = link_mb_s * efficiency
    if effective <= 0:
        return None
    return gb * 1000.0 / effective / 3600.0


def fmt_hours(hours: float | None) -> str:
    if hours is None:
        return "unknown"
    if hours < 1.0:
        return f"{hours * 60:.0f}m"
    return f"{hours:.1f}h"


def storage_wave_plan(
    models: tuple[FrontierModel, ...] = FRONTIER_MODELS,
    storage_budget_gb: float = DEFAULT_HARDWARE.storage_budget_gb,
    link_mb_s: float = 300.0,
    efficiency: float = 0.7,
    scratch_gb: float = DEFAULT_HARDWARE.scratch_reserve_gb,
    cache_reserve_gb: float = DEFAULT_HARDWARE.cache_reserve_gb,
    keep_outputs: bool = True,
    max_wave_hours: float = 6.0,
) -> dict:
    """Greedy storage waves for the source-download -> bake -> release lifecycle.

    `cycle_plan` in frontier_ops models the safest one-source-at-a-time path. This planner
    uses the same manifest but batches sources when the target SSD has enough room, while
    keeping operator checkpoints under `max_wave_hours` where possible.
    """
    pending = sorted(models, key=lambda m: (m.download_gb, m.artifact_gb()))
    kept_outputs = 0.0
    peak = 0.0
    waves = []
    impossible = []
    wave_no = 1

    while pending:
        wave = []
        wave_source = 0.0
        wave_artifact = 0.0
        i = 0
        while i < len(pending):
            model = pending[i]
            next_source = wave_source + model.download_gb
            next_artifact = wave_artifact + model.artifact_gb()
            next_peak = (
                (kept_outputs if keep_outputs else 0.0)
                + next_source
                + next_artifact
                + scratch_gb
                + cache_reserve_gb
            )
            next_hours = eta_hours(next_source, link_mb_s, efficiency)
            fits_disk = next_peak <= storage_budget_gb
            fits_checkpoint = (
                max_wave_hours <= 0
                or next_hours is None
                or next_hours <= max_wave_hours
                or not wave
            )
            if fits_disk and fits_checkpoint:
                wave.append(model)
                wave_source = next_source
                wave_artifact = next_artifact
                pending.pop(i)
                continue
            i += 1

        if not wave:
            model = pending.pop(0)
            wave = [model]
            wave_source = model.download_gb
            wave_artifact = model.artifact_gb()
            impossible.append(model.label)

        wave_peak = (
            (kept_outputs if keep_outputs else 0.0)
            + wave_source
            + wave_artifact
            + scratch_gb
            + cache_reserve_gb
        )
        peak = max(peak, wave_peak)
        waves.append({
            "wave": wave_no,
            "labels": [m.label for m in wave],
            "source_gb": round(wave_source, 3),
            "artifact_gb": round(wave_artifact, 3),
            "kept_outputs_before_gb": round(kept_outputs if keep_outputs else 0.0, 3),
            "peak_gb": round(wave_peak, 3),
            "download_eta_hours": eta_hours(wave_source, link_mb_s, efficiency),
            "exceeds_checkpoint_hours": bool(
                max_wave_hours > 0
                and eta_hours(wave_source, link_mb_s, efficiency) is not None
                and eta_hours(wave_source, link_mb_s, efficiency) > max_wave_hours
            ),
            "models": [
                {
                    "label": m.label,
                    "hf_id": m.hf_id,
                    "source_gb": m.download_gb,
                    "artifact_gb": round(m.artifact_gb(), 3),
                    "download_eta_hours": eta_hours(m.download_gb, link_mb_s, efficiency),
                }
                for m in wave
            ],
        })
        if keep_outputs:
            kept_outputs += wave_artifact
        wave_no += 1

    return {
        "storage_budget_gb": storage_budget_gb,
        "link_mb_s": link_mb_s,
        "efficiency": efficiency,
        "scratch_gb": scratch_gb,
        "cache_reserve_gb": cache_reserve_gb,
        "keep_outputs": keep_outputs,
        "max_wave_hours": max_wave_hours,
        "model_count": len(models),
        "wave_count": len(waves),
        "total_source_gb": round(total_download_gb(models), 3),
        "total_artifact_gb": round(total_artifact_gb(models), 3),
        "full_fit_gb": round(total_download_gb(models) + total_artifact_gb(models), 3),
        "planned_peak_gb": round(peak, 3),
        "download_eta_hours": eta_hours(total_download_gb(models), link_mb_s, efficiency),
        "impossible_labels": impossible,
        "waves": waves,
    }
