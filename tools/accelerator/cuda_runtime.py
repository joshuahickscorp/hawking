"""A CUDA runtime and memory-API subset, executing on Metal. C2M-T1 (G045).

C2M-T0 translated KERNELS. Every receipt since has recorded the same blocker for
T1: "no runtime, no memory API, no streams". This is that layer -- cudaMalloc,
cudaMemcpy in both directions, a <<<blocks, threads>>> launch, cudaDeviceSynchronize
and cudaFree -- parsed from CUDA HOST source and executed through AIR.

NOT A C PARSER. Like c2m.py, this is a line-oriented pattern matcher over a named
set of statement forms, and anything else is REFUSED with the statement quoted. A
host language has far more surface than a kernel body does, so the refusal
discipline matters more here, not less.

TWO MODES, AND THE DIFFERENCE BETWEEN THEM IS THE POINT (S015 §9, §125):

  FAITHFUL   cudaMalloc really allocates a second buffer and cudaMemcpy really
             copies bytes. This is what a faithful port does and what the CUDA
             semantics say.

  UNIFIED    there is no separate device memory on this machine, so cudaMalloc
             returns an ALIAS and both cudaMemcpy directions become no-ops. This
             is the steer's instruction to delete a concept that is unnecessary
             here rather than emulate it.

THE DELETION IS NOT UNCONDITIONALLY SAFE, which is the finding this module exists
to make checkable. In CUDA, a host-to-device copy is a SNAPSHOT: whatever the host
buffer held at that instant. Under aliasing it becomes a LIVE REFERENCE. A program
that writes to the host buffer after the copy and before the launch gets a
DIFFERENT ANSWER under the two modes -- so `may_delete_copies` refuses the
optimization for exactly those programs, and `unsafe_reason` names the statement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from c2m import C2MRefusal, translate

# Host statement forms this subset understands. Everything else is refused.
_MALLOC = re.compile(r"cudaMalloc\s*\(\s*\(?\s*void\s*\*\*\s*\)?\s*&\s*(\w+)\s*,")
_MEMCPY = re.compile(r"cudaMemcpy\s*\(\s*(\w+)\s*,\s*(\w+)\s*,[^,]+,\s*cudaMemcpy(\w+)\s*\)")
_LAUNCH = re.compile(r"(\w+)\s*<<<\s*([^,]+),\s*([^>]+)\s*>>>\s*\(([^)]*)\)")
_FREE = re.compile(r"cudaFree\s*\(\s*(\w+)\s*\)")
_SYNC = re.compile(r"cudaDeviceSynchronize\s*\(\s*\)")
_HOST_WRITE = re.compile(r"^\s*(\w+)\s*\[\s*(\d+)\s*\]\s*=\s*([-\d.eE+]+)f?\s*;")

# Named so a refusal says what the program used, not "unsupported".
_UNSUPPORTED = {
    "cudaMallocManaged": "managed memory (its semantics differ from both modes here)",
    "cudaMemcpyAsync": "asynchronous copies",
    "cudaStreamCreate": "streams",
    "cudaStreamSynchronize": "streams",
    "cudaEventCreate": "events",
    "cudaEventRecord": "events",
    "cudaMemset": "device memset",
    "cudaHostAlloc": "pinned host memory",
    "cudaMallocPitch": "pitched allocation",
    "cudaMemcpy2D": "2-D copies",
    "cudaGraphLaunch": "CUDA graphs",
    "cudaSetDevice": "multi-device selection",
    "cudaDeviceToDevice": "device-to-device copies",
}


@dataclass
class HostStatement:
    kind: str                       # malloc | memcpy | launch | sync | free | host_write
    text: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class HostProgram:
    statements: list[HostStatement]

    def may_delete_copies(self) -> tuple[bool, str | None]:
        """Is the UNIFIED alias observationally equivalent to a real copy?

        Only when no host buffer that was copied to the device is written again
        before the launch that reads it. A host-to-device copy is a SNAPSHOT and an
        alias is a LIVE REFERENCE; the two agree only while nobody writes.
        """
        copied_from: dict[str, str] = {}          # host name -> device name
        for st in self.statements:
            if st.kind == "memcpy" and st.data["direction"] == "HostToDevice":
                copied_from[st.data["src"]] = st.data["dst"]
            elif st.kind == "host_write" and st.data["buffer"] in copied_from:
                return False, (
                    f"host buffer {st.data['buffer']!r} is written AFTER being copied "
                    f"to the device, at {st.text.strip()!r}. In CUDA that copy is a "
                    f"snapshot and the write does not reach the device; aliasing would "
                    f"make it a live reference and change the answer. Refusing to "
                    f"delete the copy for this program.")
            elif st.kind == "launch":
                copied_from.clear()               # the snapshot has been consumed
        return True, None


def parse_host(src: str) -> HostProgram:
    for token, human in _UNSUPPORTED.items():
        if token in src:
            raise C2MRefusal(f"{human} ({token}) is not in the C2M-T1 runtime subset; "
                             f"refusing rather than pretending it is a no-op")
    stmts: list[HostStatement] = []
    for raw in src.splitlines():
        line = raw.split("//")[0]
        if not line.strip():
            continue
        if m := _MALLOC.search(line):
            stmts.append(HostStatement("malloc", raw, {"name": m.group(1)}))
        elif m := _MEMCPY.search(line):
            stmts.append(HostStatement("memcpy", raw, {
                "dst": m.group(1), "src": m.group(2), "direction": m.group(3)}))
        elif m := _LAUNCH.search(line):
            stmts.append(HostStatement("launch", raw, {
                "kernel": m.group(1), "grid": m.group(2).strip(),
                "block": m.group(3).strip(),
                "args": [a.strip() for a in m.group(4).split(",") if a.strip()]}))
        elif _SYNC.search(line):
            stmts.append(HostStatement("sync", raw))
        elif m := _FREE.search(line):
            stmts.append(HostStatement("free", raw, {"name": m.group(1)}))
        elif m := _HOST_WRITE.match(line):
            stmts.append(HostStatement("host_write", raw, {
                "buffer": m.group(1), "index": int(m.group(2)),
                "value": float(m.group(3))}))
        else:
            raise C2MRefusal(
                f"host statement outside the C2M-T1 subset: {raw.strip()!r}. "
                f"Supported: cudaMalloc, cudaMemcpy (H2D/D2H), kernel<<<g,b>>>(...), "
                f"cudaDeviceSynchronize, cudaFree, and a scalar host-array store.")
    return HostProgram(stmts)


def _memcpy_direction_ok(d: str) -> None:
    if d not in ("HostToDevice", "DeviceToHost"):
        raise C2MRefusal(f"cudaMemcpy{d} is not in the C2M-T1 subset (H2D and D2H only)")


def execute_host(host_src: str, kernels: dict[str, str], host_arrays: dict[str, Any],
                 *, elements: int, mode: str = "FAITHFUL",
                 _unsafe_ok: bool = False) -> dict[str, Any]:
    """Run the host program. Returns the host arrays as the program left them.

    mode="FAITHFUL" performs the allocations and copies the source asks for.
    mode="UNIFIED" deletes them -- and REFUSES to when may_delete_copies() says the
    two are not observationally equivalent, because a faster wrong answer is the
    one failure mode this whole program is built to avoid.

    _unsafe_ok bypasses that guard. It exists ONLY so the hazard can be
    DEMONSTRATED rather than asserted: a refusal nobody has watched be necessary is
    indistinguishable from a refusal that is merely cautious. No caller outside the
    hazard demonstration and its test may pass it.
    """
    import numpy as np
    import mlx.core as mx
    import air

    if mode not in ("FAITHFUL", "UNIFIED"):
        raise ValueError(f"unknown mode {mode!r}")
    prog = parse_host(host_src)
    if mode == "UNIFIED" and not _unsafe_ok:
        ok, why = prog.may_delete_copies()
        if not ok:
            raise C2MRefusal(why)

    # Host buffers live as device-visible arrays in BOTH modes -- that is what
    # unified memory means. What differs is whether cudaMemcpy takes a SNAPSHOT of
    # one (a real byte copy through host memory) or hands back the SAME OBJECT.
    host: dict[str, Any] = {k: mx.array(np.asarray(v, dtype=np.float32))
                            for k, v in host_arrays.items()}
    dev: dict[str, Any] = {}
    copies = 0
    for st in prog.statements:
        if st.kind == "malloc":
            dev[st.data["name"]] = None
        elif st.kind == "memcpy":
            _memcpy_direction_ok(st.data["direction"])
            d, s = st.data["dst"], st.data["src"]
            if st.data["direction"] == "HostToDevice":
                if mode == "FAITHFUL":
                    # through numpy: a genuine byte copy, and therefore a snapshot
                    dev[d] = mx.array(np.array(host[s], copy=True))
                    copies += 1
                else:
                    dev[d] = host[s]                       # alias: a live reference
            else:
                if mode == "FAITHFUL":
                    host[d] = mx.array(np.array(dev[s], copy=True))
                    copies += 1
                else:
                    host[d] = dev[s]
        elif st.kind == "launch":
            name = st.data["kernel"]
            if name not in kernels:
                raise C2MRefusal(f"launch of {name!r} but no such kernel source was given")
            tk = translate(kernels[name], elements=elements)
            ptr_args = [a for a in st.data["args"] if a in dev]
            if len(ptr_args) != len(tk.inputs) + 1:
                raise C2MRefusal(
                    f"launch of {name!r} passes {len(ptr_args)} device pointers but the "
                    f"kernel takes {len(tk.inputs)} inputs and one output")
            *ins, out = ptr_args
            missing = [p for p in ins if dev.get(p) is None]
            if missing:
                raise C2MRefusal(
                    f"kernel {name!r} reads device buffer(s) {missing} that were "
                    f"allocated but never written. CUDA leaves that memory "
                    f"uninitialised; refusing rather than silently reading zeros.")
            dev[out] = air.execute(tk.program, dict(zip(tk.inputs, [dev[p] for p in ins])))
        elif st.kind == "sync":
            mx.eval(*[v for v in dev.values() if v is not None])
        elif st.kind == "free":
            dev.pop(st.data["name"], None)
        elif st.kind == "host_write":
            host[st.data["buffer"]][st.data["index"]] = st.data["value"]
    mx.eval(*[v for v in host.values()])
    return {"host": {k: np.array(v) for k, v in host.items()},
            "copies_performed": copies, "mode": mode,
            "statements": len(prog.statements)}


def conformance_t1(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """T1 is claimed only if a host program with the runtime subset actually ran
    and matched, in BOTH modes, with identical answers."""
    ok = [r for r in runs if r.get("matches_oracle") and r.get("both_modes_agree")]
    return {
        "tier_claimed": "C2M-T1" if ok else "C2M-T0",
        "tier_definition": "T1 = kernels plus a runtime and memory API subset",
        "runtime_calls_supported": ["cudaMalloc", "cudaMemcpy(HostToDevice)",
                                    "cudaMemcpy(DeviceToHost)", "kernel<<<g,b>>>",
                                    "cudaDeviceSynchronize", "cudaFree"],
        "host_programs_run": len(runs), "host_programs_matching": len(ok),
        "higher_tiers": {
            "C2M-T2": "NOT CLAIMED: AIR now has GEMM, reduction, scan, softmax and "
                      "attention, but NONE of them is reachable through the C2M "
                      "frontend -- the T0 kernel parser handles 4 elementwise "
                      "expression forms and cannot express any of them",
            "C2M-T3": "NOT CLAIMED: no real open CUDA project has been run",
            "C2M-T4": "NOT CLAIMED: no AI workload",
            "C2M-T5": "NOT CLAIMED: nothing is production-supported",
        },
        "streams_and_events": "NOT SUPPORTED: refused by name, not silently ignored",
        "oracle": "numpy on CPU",
        "is_a_cuda_differential": False,
        "why_not": "no NVIDIA hardware, so nothing was compared against CUDA itself",
    }
