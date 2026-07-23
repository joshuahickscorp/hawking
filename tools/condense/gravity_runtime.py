#!/usr/bin/env python3.12
"""Lane G: the persistent causal engine, built before the kernel that will fill it.

The submission floor is the reason this exists.  A command buffer costs 215.8 microseconds
on this machine no matter what it carries, so a runtime that submits one per layer spends
16.83 ms per token on submission alone and cannot exceed 59.41 tok/s however good its
kernels become.  Lane A can therefore make every projection arbitrarily faster and the token
will not move.  The floor has to go first, and removing it is a runtime shape change rather
than an optimization.

Three Metal mechanisms make that possible and all three were probed present on this device:
a placement MTLHeap so every resident tensor is allocated once from one suballocated block,
an MTLIndirectCommandBuffer so a whole token's dispatches are ENCODED once and thereafter
only replayed, and a timestamp counter set so per-stage GPU time is sampled without a CPU
round trip that would reintroduce the cost being measured.

This module is deliberately representation-independent.  It knows nothing about product
quantization, codebooks or lookup tables; it knows that a stage binds some resident buffers,
runs some pipeline over some grid, and depends on the stages before it.  Whatever kernel the
selection matrix picks drops in behind :class:`StageSpec` without the graph changing, which
is the whole point of building this while the GPU is busy measuring kernels.

What it does not do, stated plainly so the ceiling is not mistaken for a floor: replaying an
indirect command buffer still costs ONE command buffer submit per token.  That is 215.8
microseconds, 21.6 percent of a 1,000 tok/s budget and 108 percent of a 5,000 tok/s budget.
So this design reaches the first milestone class and structurally cannot reach the last one.
Four-digit-plus throughput needs the CPU out of the per-token path entirely, which is a
persistent GPU loop and a different program than this file.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

RUNTIME_SCHEMA = "hawking.glm52.persistent_causal_engine.v1"

# Measured on this box; the budget arithmetic in hawking_tps_budget.py uses the same figures.
COMMAND_BUFFER_SECONDS = 215.8e-6
DISPATCH_SECONDS = 0.71e-6

DEFAULT_ARENA_BYTES = 8 << 30          # 8 GiB, well inside the 77 GiB working-set recommendation
DEFAULT_MAX_COMMANDS = 4096


class RuntimeError_(RuntimeError):
    """A runtime invariant failed: identity, residency, graph shape or backend contract."""


# ---------------------------------------------------------------------------------------
# Stable identity.  The decoder cache already learned this lesson the hard way: an id()
# based key collides after a garbage collection and silently serves the wrong weights, and
# a literal key reused across tensors does the same thing under a friendlier name.  A
# resident arena that lives for the whole process makes both failures permanent, so identity
# here is content plus provenance and there is no fallback that guesses.
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class TensorIdentity:
    """Where a resident tensor came from and what it is, sufficient to never confuse two."""

    shard: str
    name: str
    sha256: str
    role: str = "weight"

    @property
    def key(self) -> str:
        return f"{self.shard}|{self.name}|{self.sha256}|{self.role}"

    @classmethod
    def for_content(cls, payload: bytes, *, shard: str, name: str, role: str = "weight") -> "TensorIdentity":
        return cls(shard=shard, name=name, sha256=hashlib.sha256(payload).hexdigest(), role=role)


@dataclass
class Residency:
    """One allocation inside the arena."""

    identity: TensorIdentity
    offset: int
    length: int
    buffer: Any = None          # the MTLBuffer, or a stand-in under test

    @property
    def end(self) -> int:
        return self.offset + self.length


class ResidentArena:
    """One placement heap, allocated once, holding every tensor a token touches.

    Per-token allocation is one of the costs Lane G exists to delete, so this arena never
    frees during a run.  It is a bump allocator with alignment and a hard ceiling rather than
    a general allocator: a run either fits its working set or it is told, at load time, that
    it does not.  Discovering that mid-token is exactly the failure a persistent engine must
    not have.
    """

    ALIGNMENT = 256

    def __init__(self, capacity_bytes: int = DEFAULT_ARENA_BYTES, *, device: Any = None,
                 allocator: Callable[[int, int], Any] | None = None) -> None:
        self.capacity = int(capacity_bytes)
        self.device = device
        self._allocator = allocator
        self._cursor = 0
        self._by_key: dict[str, Residency] = {}
        self._order: list[str] = []

    # -- accounting -----------------------------------------------------------------
    @property
    def used_bytes(self) -> int:
        return self._cursor

    @property
    def free_bytes(self) -> int:
        return self.capacity - self._cursor

    @property
    def tensor_count(self) -> int:
        return len(self._by_key)

    def stats(self) -> dict:
        return {"capacity": self.capacity, "used": self.used_bytes, "free": self.free_bytes,
                "tensors": self.tensor_count,
                "utilization": round(self.used_bytes / self.capacity, 6) if self.capacity else 0.0}

    # -- allocation -----------------------------------------------------------------
    def _align(self, value: int) -> int:
        return (value + self.ALIGNMENT - 1) // self.ALIGNMENT * self.ALIGNMENT

    def place(self, identity: TensorIdentity, payload: bytes | memoryview) -> Residency:
        """Suballocate and record a tensor.  Re-placing identical content is a no-op hit."""
        existing = self._by_key.get(identity.key)
        length = len(payload)
        if existing is not None:
            if existing.length != length:
                raise RuntimeError_(
                    f"{identity.name}: already resident at {existing.length} bytes, "
                    f"re-placed at {length}; identity does not cover the content")
            return existing

        offset = self._align(self._cursor)
        if offset + length > self.capacity:
            raise RuntimeError_(
                f"arena full: {identity.name} needs {length} bytes, {self.free_bytes} free "
                f"of {self.capacity}; raise the arena or reduce the resident set")
        buffer = self._allocator(offset, length) if self._allocator else None
        residency = Residency(identity=identity, offset=offset, length=length, buffer=buffer)
        self._cursor = offset + length
        self._by_key[identity.key] = residency
        self._order.append(identity.key)
        return residency

    def get(self, identity: TensorIdentity) -> Residency:
        found = self._by_key.get(identity.key)
        if found is None:
            raise RuntimeError_(f"{identity.name} is not resident; place it before the graph is built")
        return found

    def resident(self) -> list[Residency]:
        return [self._by_key[k] for k in self._order]


# ---------------------------------------------------------------------------------------
# Mutable per-token state.  Weights are placed once and never change; token and KV state
# change every step, so they are separated to make it obvious which buffers a replayable
# graph is allowed to alias.
# ---------------------------------------------------------------------------------------

@dataclass
class StateBuffers:
    """The buffers whose CONTENTS change per token while their ADDRESSES do not.

    Address stability is what makes an encoded graph replayable: an indirect command buffer
    captures buffer bindings at encode time, so any state the graph reads must live at a
    fixed offset for the life of the run and be updated in place.
    """

    hidden: Residency | None = None          # the token's activation, hidden_size floats
    residual: Residency | None = None
    router_logits: Residency | None = None
    expert_ids: Residency | None = None      # GPU-side top-k output, never a CPU decision
    expert_weights: Residency | None = None
    logits: Residency | None = None
    sampled_token: Residency | None = None   # sampling stays on GPU so the loop never returns
    kv_cache: Residency | None = None
    kv_position: Residency | None = None

    def declared(self) -> dict[str, int]:
        return {name: res.length for name, res in vars(self).items() if res is not None}


# ---------------------------------------------------------------------------------------
# The graph.
# ---------------------------------------------------------------------------------------

class Backend(Protocol):
    """What a kernel must supply to be schedulable.  Deliberately minimal.

    The selection matrix picks a grammar per geometry; this is the only surface it has to
    satisfy, so a lookup-linear kernel and a 2D split decode-FMA kernel are interchangeable
    to the graph.
    """

    def pipeline_for(self, stage: "StageSpec") -> Any: ...

    def grid_for(self, stage: "StageSpec") -> tuple[tuple[int, int, int], tuple[int, int, int]]: ...

    def threadgroup_memory_for(self, stage: "StageSpec") -> int: ...


@dataclass
class StageSpec:
    """One dispatch in the token graph, described without naming a representation."""

    name: str
    kind: str                                   # projection, activation, reduce, router, sample
    reads: tuple[TensorIdentity, ...] = ()
    state_reads: tuple[str, ...] = ()
    state_writes: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    geometry: dict = field(default_factory=dict)
    layer: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise RuntimeError_("a stage needs a name; the graph is keyed on it")
        if self.kind not in {"projection", "activation", "reduce", "router", "sample", "norm", "attention"}:
            raise RuntimeError_(f"{self.name}: unknown stage kind {self.kind!r}")


class TokenGraph:
    """The dispatches for one complete token, ordered once and thereafter replayed.

    Two properties matter and both are checked rather than assumed.  The order must be a
    valid topological order of the declared dependencies, because an indirect command buffer
    executes in encode order and there is no scheduler to rescue a bad one.  And a stage may
    not read state that no earlier stage has written, because on a replayable graph that
    reads last token's value and produces a plausible wrong answer rather than an error.
    """

    def __init__(self, stages: Sequence[StageSpec]) -> None:
        self.stages = list(stages)
        names = [s.name for s in self.stages]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise RuntimeError_(f"duplicate stage names: {sorted(duplicates)}")
        self._validate_order()

    def _validate_order(self) -> None:
        seen: set[str] = set()
        written: set[str] = set()
        for stage in self.stages:
            missing = [d for d in stage.depends_on if d not in seen]
            if missing:
                raise RuntimeError_(
                    f"{stage.name} depends on {missing}, which do not precede it; "
                    "an indirect command buffer runs in encode order")
            unwritten = [s for s in stage.state_reads if s not in written]
            if unwritten:
                raise RuntimeError_(
                    f"{stage.name} reads state {unwritten} that nothing has written this token; "
                    "on a replayed graph that silently reads the previous token's value")
            seen.add(stage.name)
            written.update(stage.state_writes)

    # -- the numbers this whole lane is judged on ------------------------------------
    @property
    def dispatch_count(self) -> int:
        return len(self.stages)

    def submission_plan(self, *, commands_per_icb: int = DEFAULT_MAX_COMMANDS) -> dict:
        """How many command buffers a token costs, and what that implies for throughput.

        The honest part is the last field: an encoded graph still needs one submit, so the
        215.8 microsecond floor is divided by the whole token rather than by each layer, and
        that is the entire win.  It does not go to zero.
        """
        icbs = max(1, (self.dispatch_count + commands_per_icb - 1) // commands_per_icb)
        submission = icbs * COMMAND_BUFFER_SECONDS
        naive = self.dispatch_count * COMMAND_BUFFER_SECONDS
        return {
            "dispatches_per_token": self.dispatch_count,
            "indirect_command_buffers": icbs,
            "command_buffers_per_token": icbs,
            "submission_seconds_per_token": submission,
            "submission_tps_ceiling": round(1.0 / submission, 2),
            "naive_per_dispatch_submission_seconds": naive,
            "naive_tps_ceiling": round(1.0 / naive, 2),
            "submission_collapse_factor": round(naive / submission, 2),
            "share_of_1000tps_budget": round(submission / 1e-3, 4),
            "share_of_5000tps_budget": round(submission / 2e-4, 4),
            "reaches_1000tps_on_submission_alone": submission < 1e-3,
            "reaches_5000tps_on_submission_alone": submission < 2e-4,
        }

    def stage_kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for stage in self.stages:
            counts[stage.kind] = counts.get(stage.kind, 0) + 1
        return counts

    def describe(self) -> dict:
        return {
            "schema": RUNTIME_SCHEMA,
            "stages": self.dispatch_count,
            "stage_kinds": self.stage_kinds(),
            "layers_touched": sorted({s.layer for s in self.stages if s.layer is not None}),
            "submission": self.submission_plan(),
            "cpu_in_hot_loop": False,
            "router_on_gpu": any(s.kind == "router" for s in self.stages),
            "sampling_on_gpu": any(s.kind == "sample" for s in self.stages),
        }


def moe_layer_stages(layer: int, *, experts: Sequence[int], weights: dict[str, TensorIdentity],
                     shared_expert: bool = True) -> list[StageSpec]:
    """The real MoE dependency graph for one layer, as stages rather than as 24 matvecs.

    Wave A's gate and up projections are independent of each other and of every other
    expert, so they carry no mutual dependency and the graph is free to run all sixteen
    concurrently.  Wave B depends on the activation, and the combine depends on all of
    Wave B.  Expressing that here rather than in a Python loop is what lets the whole layer
    become encode-once work.
    """
    stages: list[StageSpec] = []
    router = f"L{layer}.router"
    stages.append(StageSpec(name=router, kind="router", layer=layer,
                            state_reads=("hidden",),
                            state_writes=("router_logits", "expert_ids", "expert_weights")))

    wave_a: list[str] = []
    for expert in experts:
        for proj in ("gate", "up"):
            key = f"layers.{layer}.experts.{expert}.{proj}"
            name = f"L{layer}.e{expert}.{proj}"
            stages.append(StageSpec(name=name, kind="projection", layer=layer,
                                    reads=(weights[key],) if key in weights else (),
                                    state_reads=("hidden",), depends_on=(router,)))
            wave_a.append(name)

    act = f"L{layer}.swiglu"
    stages.append(StageSpec(name=act, kind="activation", layer=layer,
                            depends_on=tuple(wave_a)))

    wave_b: list[str] = []
    for expert in experts:
        key = f"layers.{layer}.experts.{expert}.down"
        name = f"L{layer}.e{expert}.down"
        stages.append(StageSpec(name=name, kind="projection", layer=layer,
                                reads=(weights[key],) if key in weights else (),
                                depends_on=(act,)))
        wave_b.append(name)

    if shared_expert:
        shared = f"L{layer}.shared"
        stages.append(StageSpec(name=shared, kind="projection", layer=layer,
                                state_reads=("hidden",), depends_on=(router,)))
        wave_b.append(shared)

    stages.append(StageSpec(name=f"L{layer}.combine", kind="reduce", layer=layer,
                            state_reads=("expert_weights",), state_writes=("hidden", "residual"),
                            depends_on=tuple(wave_b)))
    return stages


def probe_device_support(device: Any = None) -> dict:
    """Which of Lane G's three mechanisms this device actually provides.

    pyobjc's respondsToSelector_ answers no for MTLDevice protocol methods that in fact
    work, so this probes by construction rather than by introspection.  A capability that
    cannot be exercised is reported absent regardless of what the API claims.
    """
    report: dict[str, Any] = {"schema": RUNTIME_SCHEMA + ".device_probe"}
    try:
        import Metal
    except ImportError as exc:
        return {**report, "available": False, "reason": f"Metal unavailable: {exc}"}

    device = device or Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        return {**report, "available": False, "reason": "no default Metal device"}
    report["device"] = str(device.name())

    try:
        desc = Metal.MTLIndirectCommandBufferDescriptor.alloc().init()
        desc.setCommandTypes_(Metal.MTLIndirectCommandTypeConcurrentDispatch)
        desc.setInheritBuffers_(False)
        desc.setMaxKernelBufferBindCount_(8)
        icb = device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(
            desc, DEFAULT_MAX_COMMANDS, 0)
        report["indirect_command_buffer"] = bool(icb)
        report["max_command_count"] = DEFAULT_MAX_COMMANDS if icb else 0
    except Exception as exc:  # noqa: BLE001
        report["indirect_command_buffer"] = False
        report["indirect_command_buffer_error"] = f"{type(exc).__name__}: {exc}"

    try:
        hd = Metal.MTLHeapDescriptor.alloc().init()
        hd.setSize_(1 << 20)
        hd.setStorageMode_(Metal.MTLStorageModeShared)
        hd.setType_(Metal.MTLHeapTypePlacement)
        report["placement_heap"] = bool(device.newHeapWithDescriptor_(hd))
    except Exception as exc:  # noqa: BLE001
        report["placement_heap"] = False
        report["placement_heap_error"] = f"{type(exc).__name__}: {exc}"

    try:
        report["counter_sets"] = [str(s.name()) for s in (device.counterSets() or [])]
        report["gpu_timestamps"] = any("timestamp" in s for s in report["counter_sets"])
    except Exception as exc:  # noqa: BLE001
        report["counter_sets"] = []
        report["gpu_timestamps"] = False
        report["counter_error"] = f"{type(exc).__name__}: {exc}"

    report["available"] = bool(report.get("indirect_command_buffer")
                               and report.get("placement_heap"))
    report["note"] = ("an encoded graph still costs one command-buffer submit per token, "
                      f"{COMMAND_BUFFER_SECONDS*1e6:.1f} us, which is a floor this design "
                      "lowers but does not remove")
    return report


def selftest() -> int:
    """Every invariant that does not need a GPU, which is all of them at G0."""
    ident = TensorIdentity.for_content(b"abc", shard="model-00002", name="w")
    assert ident.sha256 == hashlib.sha256(b"abc").hexdigest()
    assert TensorIdentity.for_content(b"abc", shard="model-00002", name="w") == ident
    assert TensorIdentity.for_content(b"abd", shard="model-00002", name="w") != ident

    # arena: alignment, accounting, idempotent placement, honest exhaustion
    arena = ResidentArena(4096)
    a = arena.place(TensorIdentity("s", "a", "h1"), b"\x00" * 100)
    b = arena.place(TensorIdentity("s", "b", "h2"), b"\x00" * 100)
    assert a.offset == 0 and b.offset == 256, (a.offset, b.offset)
    assert arena.place(TensorIdentity("s", "a", "h1"), b"\x00" * 100) is a, "identical content re-hits"
    assert arena.tensor_count == 2 and arena.used_bytes == 356
    try:
        arena.place(TensorIdentity("s", "big", "h3"), b"\x00" * 8192)
        raise AssertionError("arena overflow went undetected")
    except RuntimeError_ as exc:
        assert "arena full" in str(exc)
    try:
        arena.place(TensorIdentity("s", "a", "h1"), b"\x00" * 50)
        raise AssertionError("same identity, different length went undetected")
    except RuntimeError_ as exc:
        assert "identity does not cover" in str(exc)

    # graph: dependency order and unwritten-state detection
    try:
        TokenGraph([StageSpec("b", "reduce", depends_on=("a",)), StageSpec("a", "projection")])
        raise AssertionError("out-of-order dependency went undetected")
    except RuntimeError_ as exc:
        assert "do not precede it" in str(exc)
    try:
        TokenGraph([StageSpec("a", "projection", state_reads=("hidden",))])
        raise AssertionError("read of never-written state went undetected")
    except RuntimeError_ as exc:
        assert "silently reads the previous token" in str(exc)

    # a real MoE layer's shape
    weights = {f"layers.5.experts.{e}.{p}": TensorIdentity("s", f"e{e}.{p}", f"h{e}{p}")
               for e in range(8) for p in ("gate", "up", "down")}
    stages = moe_layer_stages(5, experts=range(8), weights=weights)
    seed = [StageSpec("seed", "norm", state_writes=("hidden",))]
    graph = TokenGraph(seed + stages)
    kinds = graph.stage_kinds()
    assert kinds["projection"] == 8 * 2 + 8 + 1, kinds       # wave A, wave B, shared
    assert kinds["router"] == 1 and kinds["reduce"] == 1 and kinds["activation"] == 1

    # the submission arithmetic, which is the point of the lane
    plan = graph.submission_plan()
    assert plan["command_buffers_per_token"] == 1
    assert plan["dispatches_per_token"] == 29, plan["dispatches_per_token"]
    assert plan["submission_collapse_factor"] == 29.0, plan["submission_collapse_factor"]
    assert abs(plan["share_of_1000tps_budget"] - 0.2158) < 1e-3
    assert plan["reaches_1000tps_on_submission_alone"]
    assert not plan["reaches_5000tps_on_submission_alone"], "5,000 tok/s must stay out of reach"

    # a whole-token graph: 75 sparse layers under one encoded buffer
    big = seed[:]
    for layer in range(3, 78):
        big.extend(moe_layer_stages(layer, experts=range(8), weights={}))
    whole = TokenGraph(big)
    wp = whole.submission_plan()
    assert wp["dispatches_per_token"] == 1 + 75 * 28, wp["dispatches_per_token"]
    assert wp["indirect_command_buffers"] == 1, "2101 dispatches fit one 4096-command buffer"
    assert wp["command_buffers_per_token"] == 1
    assert wp["naive_tps_ceiling"] < 3.0, wp["naive_tps_ceiling"]
    assert wp["submission_tps_ceiling"] > 4600, wp["submission_tps_ceiling"]

    print(json.dumps({
        "selftest": "PASS", "schema": RUNTIME_SCHEMA,
        "whole_token_dispatches": wp["dispatches_per_token"],
        "command_buffers_per_token": wp["command_buffers_per_token"],
        "submission_collapse_factor": wp["submission_collapse_factor"],
        "submission_tps_ceiling": wp["submission_tps_ceiling"],
        "naive_per_dispatch_tps_ceiling": wp["naive_tps_ceiling"],
        "reaches_5000tps_on_submission_alone": wp["reaches_5000tps_on_submission_alone"],
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        print(json.dumps(probe_device_support(), indent=2))
        return 0
    sys.stderr.write("usage: gravity_runtime.py [selftest|probe]\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
