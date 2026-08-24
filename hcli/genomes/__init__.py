"""Genomes: learned persistent machine / model / runtime science.

``MachineGenome`` is a compatibility bag in ``hcli.machine``. Admission
numbers come from ``resolve_runtime_limits``; the producer is
``tools/headless/machine_probe.py``. ``RuntimeGenome`` records per-backend
performance (MLX live from CONVENTIONAL_CONTROL_SET, llama.cpp Q5_K
archived). This package is the ownership surface, not a second genome
authority and not a second scheduler.
"""
from hcli.machine import (
    GenomeFreshness,
    GenomeStale,
    MachineGenome,
    host_snapshot,
)
from hcli.genomes.runtime_genome import RuntimeGenome, load_runtime_genome

__all__ = [
    "GenomeFreshness",
    "GenomeStale",
    "MachineGenome",
    "RuntimeGenome",
    "host_snapshot",
    "load_runtime_genome",
]
