#!/usr/bin/env python3.12
"""Metal execution grammars for ``glm52.functional.moe.v1``, parity-gated against the CPU
authority.

Three materially different ways to run the same function, because the interesting question
is not "can Metal do a matvec" but which grammar wins when the payload is a seed:

* **FRT-A, explicit feature map.** Hold the generated projection resident and do two
  matvecs.  Fastest arithmetic, but it turns a 12 MB artifact into a 37 MB resident organ.
* **FRT-B, procedural feature generation.** Generate each projection element inside the
  kernel from the seed.  Nothing but the readout is resident, at the cost of six million
  generator calls per token per layer.  This is why the generator is a stateless hash
  rather than NumPy's sequential PCG64.
* **FRT-D, direct linear.** The dense upper control, and the runtime baseline: one matvec
  against a 75 MB map.

The teacher is the number all three are trying to beat: eight active experts of three
[2048, 6144] BF16 matrices is 604 MB of weight traffic per token per layer.

    bench [HIDDEN]
    selftest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import gravity_functional_codec as codec  # noqa: E402

HIDDEN = 6144
TEACHER_ACTIVE_BYTES_PER_LAYER = 8 * 3 * 2048 * 6144 * 2

METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// The generator, transliterated from gravity_functional_codec._splitmix64.  It is here in
// ten lines precisely because a sequential library RNG could not be.
inline ulong splitmix64(ulong state) {
    state += 0x9E3779B97F4A7C15UL;
    ulong z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

inline float uniform_open(ulong seed, ulong stream, ulong index) {
    ulong key = seed * 0xD1342543DE82EF95UL + stream * 0xA24BAED4963EE407UL;
    ulong bits = splitmix64(key ^ index);
    return ((float)(bits >> 11) + 0.5f) * (1.0f / 9007199254740992.0f);
}

inline float projection_element(ulong seed, ulong index, float inv_sqrt_width) {
    float u1 = uniform_open(seed, 1UL, index);
    float u2 = uniform_open(seed, 2UL, index);
    return sqrt(-2.0f * log(u1)) * cos(6.28318530717958647692f * u2) * inv_sqrt_width;
}

inline float silu(float v) { return v / (1.0f + exp(-v)); }

// FRT-A stage 1: features from a resident projection.  One thread per feature.
kernel void features_explicit(
    device const half*  projection [[buffer(0)]],   // [width * hidden], row major
    device const float* x          [[buffer(1)]],   // [width]
    device       float* phi        [[buffer(2)]],   // [hidden]
    constant     uint&  width      [[buffer(3)]],
    constant     uint&  hidden     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hidden) return;
    float acc = 0.0f;
    for (uint i = 0; i < width; ++i) {
        acc = fma(x[i], (float)projection[(ulong)i * hidden + gid], acc);
    }
    phi[gid] = silu(acc);
}

// FRT-B stage 1: the same features, generated rather than read.
kernel void features_procedural(
    device const float* x       [[buffer(0)]],
    device       float* phi     [[buffer(1)]],
    constant     uint&  width   [[buffer(2)]],
    constant     uint&  hidden  [[buffer(3)]],
    constant     ulong& seed    [[buffer(4)]],
    constant     float& inv_sqrt_width [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hidden) return;
    float acc = 0.0f;
    for (uint i = 0; i < width; ++i) {
        acc = fma(x[i], projection_element(seed, (ulong)i * hidden + gid, inv_sqrt_width),
                  acc);
    }
    phi[gid] = silu(acc);
}

// Shared readout, and the whole of FRT-D: one thread per output element.
kernel void readout(
    device const half*  weight [[buffer(0)]],   // [rows * cols], row major
    device const float* input  [[buffer(1)]],   // [rows]
    device       float* out    [[buffer(2)]],   // [cols]
    constant     uint&  rows   [[buffer(3)]],
    constant     uint&  cols   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cols) return;
    float acc = 0.0f;
    for (uint r = 0; r < rows; ++r) {
        acc = fma(input[r], (float)weight[(ulong)r * cols + gid], acc);
    }
    out[gid] = acc;
}
"""


class MetalUnavailable(RuntimeError):
    pass


class FunctionalMetal:
    def __init__(self):
        try:
            import Metal
        except ImportError as error:
            raise MetalUnavailable(f"pyobjc Metal not importable: {error}") from error
        self.metal = Metal
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise MetalUnavailable("no default Metal device")
        library, error = self.device.newLibraryWithSource_options_error_(
            METAL_SOURCE, None, None)
        if library is None:
            raise MetalUnavailable(f"kernel compile failed: {error}")
        self.queue = self.device.newCommandQueue()
        self.pipelines = {}
        for name in ("features_explicit", "features_procedural", "readout"):
            pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
                library.newFunctionWithName_(name), None)
            if pipeline is None:
                raise MetalUnavailable(f"{name}: {error}")
            self.pipelines[name] = pipeline

    def _buffer(self, array: np.ndarray):
        data = np.ascontiguousarray(array)
        return self.device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, 0)

    def _empty(self, count: int, itemsize: int = 4):
        return self.device.newBufferWithLength_options_(count * itemsize, 0)

    def _scalar(self, value, dtype):
        return self._buffer(np.array([value], dtype=dtype))

    def _dispatch(self, encoder, pipeline, threads: int):
        encoder.setComputePipelineState_(pipeline)
        width = min(pipeline.maxTotalThreadsPerThreadgroup(), 256)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.metal.MTLSizeMake(threads, 1, 1),
            self.metal.MTLSizeMake(width, 1, 1))

    def _read(self, buffer, count: int) -> np.ndarray:
        return np.frombuffer(buffer.contents().as_buffer(count * 4),
                             dtype=np.float32).copy()

    def prepare(self, grammar: str, payload: dict, x: np.ndarray, *,
                projection: np.ndarray | None = None) -> dict:
        """Allocate every device buffer once.

        Buffer creation copies the payload through the host, and at 75 MB that dwarfs the
        kernel it is meant to feed.  Timing an organ that is re-uploaded every token would
        measure the harness, not the grammar: a real runtime uploads once and decodes
        many.
        """
        resident = {
            "x": self._buffer(np.asarray(x, dtype=np.float32).reshape(-1)),
            "readout": self._buffer(np.ascontiguousarray(payload["left"], dtype=np.float16)),
            "out": self._empty(payload["out_width"]),
            "width": self._scalar(payload["width"], np.uint32),
            "hidden": self._scalar(payload["hidden"], np.uint32),
            "out_width": self._scalar(payload["out_width"], np.uint32),
            "seed": self._scalar(payload["seed"], np.uint64),
            "inv_sqrt_width": self._scalar(1.0 / np.sqrt(payload["width"]), np.float32),
        }
        if grammar != "FRT_D":
            resident["phi"] = self._empty(payload["hidden"])
        if grammar == "FRT_A":
            resident["projection"] = self._buffer(
                np.ascontiguousarray(projection, dtype=np.float16))
        return resident

    def run(self, grammar: str, payload: dict, x: np.ndarray, *,
            projection: np.ndarray | None = None,
            resident: dict | None = None) -> tuple[np.ndarray, int]:
        """Returns the output and the command buffers it took."""
        width, hidden = payload["width"], payload["hidden"]
        out_width = payload["out_width"]
        resident = resident if resident is not None else self.prepare(
            grammar, payload, x, projection=projection)
        x_buffer = resident["x"]
        readout_buffer = resident["readout"]

        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        out = resident["out"]
        if grammar == "FRT_D":
            encoder.setBuffer_offset_atIndex_(readout_buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(x_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(out, 0, 2)
            encoder.setBuffer_offset_atIndex_(resident["width"], 0, 3)
            encoder.setBuffer_offset_atIndex_(resident["out_width"], 0, 4)
            self._dispatch(encoder, self.pipelines["readout"], out_width)
        else:
            phi = resident["phi"]
            if grammar == "FRT_A":
                encoder.setBuffer_offset_atIndex_(resident["projection"], 0, 0)
                encoder.setBuffer_offset_atIndex_(x_buffer, 0, 1)
                encoder.setBuffer_offset_atIndex_(phi, 0, 2)
                encoder.setBuffer_offset_atIndex_(resident["width"], 0, 3)
                encoder.setBuffer_offset_atIndex_(resident["hidden"], 0, 4)
                self._dispatch(encoder, self.pipelines["features_explicit"], hidden)
            else:
                encoder.setBuffer_offset_atIndex_(x_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(phi, 0, 1)
                encoder.setBuffer_offset_atIndex_(resident["width"], 0, 2)
                encoder.setBuffer_offset_atIndex_(resident["hidden"], 0, 3)
                encoder.setBuffer_offset_atIndex_(resident["seed"], 0, 4)
                encoder.setBuffer_offset_atIndex_(resident["inv_sqrt_width"], 0, 5)
                self._dispatch(encoder, self.pipelines["features_procedural"], hidden)
            encoder.setBuffer_offset_atIndex_(readout_buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(phi, 0, 1)
            encoder.setBuffer_offset_atIndex_(out, 0, 2)
            encoder.setBuffer_offset_atIndex_(resident["hidden"], 0, 3)
            encoder.setBuffer_offset_atIndex_(resident["out_width"], 0, 4)
            self._dispatch(encoder, self.pipelines["readout"], out_width)
        final, count = out, out_width

        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        return self._read(final, count), 1

    def _encode_layer(self, encoder, payload, resident) -> None:
        """Encode one FRT-B functional MoE (procedural features + readout) into an existing
        encoder, without committing. This is the unit the submission-collapse benchmark
        chains: many of these into a single command buffer."""
        hidden, out_width = payload["hidden"], payload["out_width"]
        phi, out = resident["phi"], resident["out"]
        encoder.setBuffer_offset_atIndex_(resident["x"], 0, 0)
        encoder.setBuffer_offset_atIndex_(phi, 0, 1)
        encoder.setBuffer_offset_atIndex_(resident["width"], 0, 2)
        encoder.setBuffer_offset_atIndex_(resident["hidden"], 0, 3)
        encoder.setBuffer_offset_atIndex_(resident["seed"], 0, 4)
        encoder.setBuffer_offset_atIndex_(resident["inv_sqrt_width"], 0, 5)
        self._dispatch(encoder, self.pipelines["features_procedural"], hidden)
        encoder.setBuffer_offset_atIndex_(resident["readout"], 0, 0)
        encoder.setBuffer_offset_atIndex_(phi, 0, 1)
        encoder.setBuffer_offset_atIndex_(out, 0, 2)
        encoder.setBuffer_offset_atIndex_(resident["hidden"], 0, 3)
        encoder.setBuffer_offset_atIndex_(resident["out_width"], 0, 4)
        self._dispatch(encoder, self.pipelines["readout"], out_width)

    def stack_buffers(self, payload, layers: int) -> list:
        """Pre-allocate one resident buffer set per layer, once, outside any timed path."""
        return [self.prepare("FRT_B", payload,
                             np.random.default_rng(i).standard_normal(HIDDEN)
                             .astype(np.float32))
                for i in range(layers)]

    def run_stack(self, payload, residents: list, *, collapsed: bool) -> int:
        """Run the functional MoE across pre-allocated layers; return buffers submitted.

        collapsed=False submits one command buffer per layer, the current per-layer cost:
        76 layers is 76 submissions per token. collapsed=True encodes every layer into a
        single command buffer and submits once. Each layer owns its output buffer, so the
        collapsed path has no false serialisation the split path would not also have.
        """
        if collapsed:
            command = self.queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            for resident in residents:
                self._encode_layer(encoder, payload, resident)
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            return 1
        for resident in residents:
            command = self.queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            self._encode_layer(encoder, payload, resident)
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
        return len(residents)


def bench_submission_collapse(layers: int = 76, hidden: int = 1024,
                              repeats: int = 10) -> dict:
    """The token-level submission cost: 76 command buffers versus one.

    The functional MoE at each layer is one FRT-B dispatch pair. A whole token runs it once
    per sparse layer, so the per-layer submission model costs one command buffer per layer.
    Encoding the stack into a single command buffer is the collapse the roofline said was
    the runtime work, since the kernels were already at a few percent of bandwidth.
    """
    metal = FunctionalMetal()
    payload = _payload(hidden)
    residents = metal.stack_buffers(payload, layers)  # allocated once, outside timing

    metal.run_stack(payload, residents, collapsed=False)  # warm
    split_started = time.perf_counter()
    for _ in range(repeats):
        split_buffers = metal.run_stack(payload, residents, collapsed=False)
    split = (time.perf_counter() - split_started) / repeats

    metal.run_stack(payload, residents, collapsed=True)  # warm
    collapsed_started = time.perf_counter()
    for _ in range(repeats):
        collapsed_buffers = metal.run_stack(payload, residents, collapsed=True)
    collapsed = (time.perf_counter() - collapsed_started) / repeats

    return {
        "schema": "hawking.glm52.submission_collapse.v1",
        "device": str(metal.device.name()),
        "layers": layers,
        "repeats": repeats,
        "per_layer_command_buffers": {
            "command_buffers_per_token": split_buffers,
            "seconds_per_token_moe_path": split,
            "moe_tps": 1.0 / split},
        "single_command_buffer": {
            "command_buffers_per_token": collapsed_buffers,
            "seconds_per_token_moe_path": collapsed,
            "moe_tps": 1.0 / collapsed},
        "command_buffer_reduction": f"{split_buffers} -> {collapsed_buffers}",
        "speedup": split / collapsed,
        "submission_overhead_seconds_per_token": split - collapsed,
        "submission_overhead_per_layer_us": (split - collapsed) / layers * 1e6,
        "reading": "the gap between the two is the per-token command-submission cost the "
                   "collapse removes; the kernels themselves are unchanged",
    }


def _payload(hidden: int, seed: int = 17, linear: bool = False) -> dict:
    generator = np.random.default_rng(0)
    if linear:
        # hidden = 0 is the codec's "no feature map" form: the payload maps the hidden
        # state directly, which is exactly what FRT-D executes.
        weight = (generator.standard_normal((HIDDEN, HIDDEN)) / 78.0).astype(np.float16)
        return {"width": HIDDEN, "hidden": 0, "out_width": HIDDEN, "rank": 0,
                "activation": codec.ACTIVATION_IDENTITY, "seed": seed, "scale": 1.0,
                "layer": 38, "left": weight, "right": None}
    readout = (generator.standard_normal((hidden, HIDDEN)) / 32.0).astype(np.float16)
    return {"width": HIDDEN, "hidden": hidden, "out_width": HIDDEN, "rank": 0,
            "activation": codec.ACTIVATION_SILU, "seed": seed, "scale": 1.0,
            "layer": 38, "left": readout, "right": None}


def bench(hidden: int = 1024, *, repeats: int = 20) -> dict:
    metal = FunctionalMetal()
    x = np.random.default_rng(1).standard_normal(HIDDEN).astype(np.float32)
    rows = []

    for grammar in ("FRT_A", "FRT_B", "FRT_D"):
        payload = _payload(hidden, linear=(grammar == "FRT_D"))
        projection = (codec.projection(HIDDEN, hidden, payload["seed"])
                      if grammar == "FRT_A" else None)
        reference = codec.execute(payload, x[None])[0]

        buffers_in = metal.prepare(grammar, payload, x, projection=projection)
        upload_started = time.perf_counter()
        metal.prepare(grammar, payload, x, projection=projection)
        upload = time.perf_counter() - upload_started

        produced, buffers = metal.run(grammar, payload, x, resident=buffers_in)
        metal.run(grammar, payload, x, resident=buffers_in)  # warm
        started = time.perf_counter()
        for _ in range(repeats):
            metal.run(grammar, payload, x, resident=buffers_in)
        elapsed = (time.perf_counter() - started) / repeats

        relative = float(np.linalg.norm(produced - reference)
                         / max(np.linalg.norm(reference), 1e-12))
        cosine = float(np.dot(produced, reference)
                       / max(np.linalg.norm(produced) * np.linalg.norm(reference), 1e-12))
        readout_bytes = payload["left"].nbytes
        resident = readout_bytes + (projection.nbytes // 2 if grammar == "FRT_A" else 0)
        rows.append({
            "grammar": grammar,
            "hidden": payload["hidden"],
            "cpu_authority_relative_l2": relative,
            "cpu_authority_cosine": cosine,
            "parity": bool(relative < 5e-2 and cosine > 0.999),
            "seconds_per_layer_call": elapsed,
            "one_time_upload_seconds": upload,
            "command_buffers_per_call": buffers,
            "resident_bytes_per_layer": int(resident),
            "active_bytes_per_token_per_layer": int(resident),
            "traffic_versus_teacher": TEACHER_ACTIVE_BYTES_PER_LAYER / max(resident, 1),
            "multiply_accumulate_per_token": int(
                HIDDEN * payload["out_width"] if payload["hidden"] == 0
                else HIDDEN * payload["hidden"] + payload["hidden"] * payload["out_width"]),
            "generator_calls_per_token":
                int(HIDDEN * payload["hidden"]) if grammar == "FRT_B" else 0,
        })

    return {
        "schema": "hawking.glm52.functional_metal_benchmark.v1",
        "bandwidth_probe": bandwidth_probe(),
        "device": str(metal.device.name()),
        "unified_memory": bool(metal.device.hasUnifiedMemory()),
        "teacher_active_bytes_per_layer": TEACHER_ACTIVE_BYTES_PER_LAYER,
        "teacher_active_experts": 8,
        "repeats": repeats,
        "grammars": rows,
        "parity_gate": "relative L2 below 5e-2 and cosine above 0.999 against the CPU "
                       "authority; fp16 readouts and a float32 accumulator will not agree "
                       "to bit level and are not asked to",
        "all_parity": all(row["parity"] for row in rows),
        "honest_scope": "one layer, batch 1, one command buffer per call, no residual, no "
                        "attention, no KV. This is the MoE replacement in isolation.",
    }


def bandwidth_probe(rows: int = 6144, cols: int = 262144, repeats: int = 5) -> dict:
    """What this machine actually sustains through the same kernel, at real occupancy.

    A 6144-wide dispatch is far too small to saturate an M3 Ultra, so the per-layer
    latencies above are occupancy-bound rather than bandwidth-bound.  Widening the same
    readout kernel until occupancy stops being the limit gives the roofline number the
    token budget should be divided by, measured on this box rather than quoted from a
    specification.
    """
    metal = FunctionalMetal()
    payload = {"width": rows, "hidden": 0, "out_width": cols, "rank": 0,
               "activation": codec.ACTIVATION_IDENTITY, "seed": 0, "scale": 1.0,
               "layer": 0,
               "left": np.zeros((rows, cols), dtype=np.float16), "right": None}
    x = np.ones(rows, dtype=np.float32)
    resident = metal.prepare("FRT_D", payload, x)
    metal.run("FRT_D", payload, x, resident=resident)
    started = time.perf_counter()
    for _ in range(repeats):
        metal.run("FRT_D", payload, x, resident=resident)
    elapsed = (time.perf_counter() - started) / repeats
    read_bytes = rows * cols * 2
    return {
        "kernel": "readout",
        "threads": cols,
        "bytes_read_per_call": read_bytes,
        "seconds_per_call": elapsed,
        "achieved_bytes_per_second": read_bytes / elapsed,
        "achieved_gigabytes_per_second": read_bytes / elapsed / 1e9,
        "note": "same kernel as FRT-D, widened until occupancy stops binding; this is a "
                "measured achievable rate for a strided fp16 read, not a peak figure",
    }


def selftest() -> int:
    # The kernel's generator must reproduce the CPU authority's, or FRT-B is a different
    # function wearing the same codec id.  Checked through parity in bench; here the CPU
    # side is checked for the properties the kernel relies on.
    index = np.arange(16, dtype=np.uint64)
    assert np.all(codec._uniform(17, 1, index) > 0.0)
    assert np.all(codec._uniform(17, 1, index) < 1.0)

    try:
        metal = FunctionalMetal()
    except MetalUnavailable as error:
        print(json.dumps({"selftest": "SKIPPED_NO_METAL", "reason": str(error)}))
        return 0

    x = np.random.default_rng(2).standard_normal(HIDDEN).astype(np.float32)
    for grammar in ("FRT_A", "FRT_B", "FRT_D"):
        payload = _payload(64, linear=(grammar == "FRT_D"))
        projection = (codec.projection(HIDDEN, 64, payload["seed"])
                      if grammar == "FRT_A" else None)
        produced, buffers = metal.run(grammar, payload, x, projection=projection)
        reference = codec.execute(payload, x[None])[0]
        cosine = float(np.dot(produced, reference)
                       / max(np.linalg.norm(produced) * np.linalg.norm(reference), 1e-12))
        assert produced.shape == reference.shape, (grammar, produced.shape)
        assert cosine > 0.999, (grammar, cosine)
        assert buffers == 1, (grammar, buffers)
    print(json.dumps({"selftest": "PASS", "grammars": ["FRT_A", "FRT_B", "FRT_D"],
                      "device": str(metal.device.name())}))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "selftest":
        raise SystemExit(selftest())
    if command == "bench":
        payload = bench(int(sys.argv[2]) if len(sys.argv) > 2 else 1024)
        out = Path(__file__).resolve().parents[2] / "reports" / "condense" / \
            "glm52_generation_b" / "GLM52_FUNCTIONAL_METAL_BENCHMARK.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
        print(json.dumps(payload, indent=2, default=float))
    elif command == "collapse":
        payload = bench_submission_collapse(
            int(sys.argv[2]) if len(sys.argv) > 2 else 76)
        out = Path(__file__).resolve().parents[2] / "reports" / "condense" / \
            "glm52_generation_b" / "GLM52_FUNCTIONAL_SUBMISSION_COLLAPSE.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
        print(json.dumps(payload, indent=2, default=float))
    else:
        raise SystemExit(f"unknown command: {command}")
