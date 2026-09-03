"""Recomposed science module gravity_metal (C-SCI-R1)."""
from __future__ import annotations
from lab.operators import gravity_forge as forge
import sys as _sys_a1
from pathlib import Path as _Path_a1
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == 'archive' else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
HERE = _A1_CONDENSE
KERNEL_VERSION = 2
THREADS = 256
DEFAULT_CACHE_BUDGET_BYTES = 2 * 1024 ** 3
CACHE_KEY_VERSION = 1
DEFAULT_THREADGROUP_MEMORY = 32768
METAL_SOURCE = '\n#include <metal_stdlib>\nusing namespace metal;\n\nstruct Dims { uint rows; uint nchunk; uint D; uint k; uint stage_x; uint pad0; uint pad1; uint pad2; };\n\n// One THREAD per output row.  Accumulator stays in a register; no reduction, no barriers\n// past the cooperative staging.  Indices arrive transposed so this reads them coalesced.\nkernel void gravity_pq_matvec_rows(\n    device   const uchar*  indices   [[buffer(0)]],   // [nchunk][rows], transposed\n    device   const half*   codebook  [[buffer(1)]],   // [k * D]\n    device   const float*  x         [[buffer(2)]],   // [nchunk * D]\n    device         float*  y         [[buffer(3)]],   // [rows]\n    constant       Dims&   dims      [[buffer(4)]],\n    threadgroup    float*  scratch   [[threadgroup(0)]],\n    uint gid [[thread_position_in_grid]],\n    uint tid [[thread_position_in_threadgroup]],\n    uint tsz [[threads_per_threadgroup]])\n{\n    const uint D = dims.D, k = dims.k, nchunk = dims.nchunk, rows = dims.rows;\n\n    threadgroup float* book = scratch;\n    threadgroup float* xs   = scratch + k * D;\n\n    for (uint i = tid; i < k * D; i += tsz) book[i] = float(codebook[i]);\n    if (dims.stage_x) {\n        for (uint i = tid; i < nchunk * D; i += tsz) xs[i] = x[i];\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    const uint row = gid;\n    if (row >= rows) return;\n\n    float acc = 0.0f;\n    if (dims.stage_x) {\n        for (uint c = 0; c < nchunk; ++c) {\n            threadgroup const float* w  = book + (uint)indices[c * rows + row] * D;\n            threadgroup const float* xv = xs + c * D;\n            for (uint j = 0; j < D; ++j) acc = fma(w[j], xv[j], acc);\n        }\n    } else {\n        for (uint c = 0; c < nchunk; ++c) {\n            threadgroup const float* w = book + (uint)indices[c * rows + row] * D;\n            device const float* xv = x + c * D;\n            for (uint j = 0; j < D; ++j) acc = fma(w[j], xv[j], acc);\n        }\n    }\n    y[row] = acc;\n}\n\nstruct DimsV1 { uint rows; uint nchunk; uint D; uint k; };\n\n// One threadgroup per output row.  The codebook is staged in threadgroup memory once and\n// then every codeword lookup is on-chip; decoded weights live only in registers.\nkernel void gravity_pq_matvec(\n    device   const uchar*  indices   [[buffer(0)]],   // [rows * nchunk], one index per chunk\n    device   const half*   codebook  [[buffer(1)]],   // [k * D]\n    device   const float*  x         [[buffer(2)]],   // [nchunk * D]\n    device         float*  y         [[buffer(3)]],   // [rows]\n    constant       DimsV1& dims      [[buffer(4)]],\n    threadgroup    float*  scratch   [[threadgroup(0)]],\n    uint tgid [[threadgroup_position_in_grid]],\n    uint tid  [[thread_position_in_threadgroup]],\n    uint tsz  [[threads_per_threadgroup]])\n{\n    const uint D = dims.D;\n    const uint k = dims.k;\n    const uint nchunk = dims.nchunk;\n\n    threadgroup float* book = scratch;              // k * D floats\n    threadgroup float* partial = scratch + k * D;   // tsz floats, for the reduction\n\n    for (uint i = tid; i < k * D; i += tsz) {\n        book[i] = float(codebook[i]);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    const uint row = tgid;\n    float acc = 0.0f;\n    if (row < dims.rows) {\n        device const uchar* row_idx = indices + (ulong)row * nchunk;\n        for (uint c = tid; c < nchunk; c += tsz) {\n            threadgroup const float* w = book + (uint)row_idx[c] * D;\n            device const float* xv = x + c * D;\n            float s = 0.0f;\n            for (uint j = 0; j < D; ++j) {\n                s = fma(w[j], xv[j], s);            // decoded weight stays in a register\n            }\n            acc += s;\n        }\n    }\n\n    partial[tid] = acc;\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint stride = tsz >> 1; stride > 0; stride >>= 1) {\n        if (tid < stride) partial[tid] += partial[tid + stride];\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n    if (tid == 0 && row < dims.rows) y[row] = partial[0];\n}\n'

class MetalUnavailable(RuntimeError):
    """The Metal framework or a suitable device is not usable from this process."""

class GravityMetalInputError(ValueError):
    """A caller handed the kernel arguments that do not match the uploaded geometry.

    Separate from :class:`MetalUnavailable` on purpose: that one means "this machine cannot
    run the kernel", this one means "this call is wrong", and the two want different
    handling.  Every check that guards a raw device pointer raises this.
    """

def _index_bits(k: int) -> int:
    """Bits the packer bills per index.  k=128 bills 7; the kernel still uploads 8."""
    return max(1, int(k - 1).bit_length())

def _stage_plan(k: int, D: int, nchunk: int, limit: int) -> tuple[bool, int]:
    """x is staged next to the codebook only when both fit the threadgroup allotment."""
    stage_x = (k * D + nchunk * D) * 4 <= limit
    return (stage_x, (k * D + (nchunk * D if stage_x else 0)) * 4)

def _validate_x(x, *, nchunk: int, D: int, allocated_bytes: int) -> np.ndarray:
    """Everything that stands between a caller's x and a bare device pointer.

    ``MTLBuffer.contents()`` returns a void* with no length attached, so the ``as_buffer``
    view is sized by whatever number we hand it -- a longer x writes straight past the end
    of an allocation fixed at nchunk*D*4 and corrupts whatever the heap put next.  There is
    no bounds check downstream of this function, so there has to be one here, and it runs
    before the pointer is ever obtained.  ``allocated_bytes`` is the size the buffer was
    really created with, carried in the cache entry, not a recomputation of it.
    """
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        raise GravityMetalInputError(f'x must be float32 (the kernel reads device const float*), got {arr.dtype}')
    expected_elems = nchunk * D
    expected_bytes = expected_elems * 4
    if arr.size != expected_elems:
        raise GravityMetalInputError(f"x has {arr.size} elements, this tensor's geometry needs nchunk*D = {nchunk}*{D} = {expected_elems}")
    xv = np.ascontiguousarray(arr.ravel())
    if xv.nbytes != expected_bytes:
        raise GravityMetalInputError(f'x is {xv.nbytes} bytes, expected {expected_bytes}')
    if allocated_bytes != expected_bytes:
        raise GravityMetalInputError(f'staged x buffer was allocated with {allocated_bytes} bytes but the geometry needs {expected_bytes}; the cache entry does not match this tensor')
    return xv

def content_key(codes: dict) -> str:
    """Opt-in content address for a codes dict: geometry, indices and codebooks.

    Offered, never applied silently.  Two tensors with byte-identical indices and codebooks
    genuinely are the same upload, so sharing a cache slot is correct; two tensors that
    merely happen to reuse a freed ``id()`` are not, which is the bug this replaces.  It
    costs a full pass over the index array, so the caller decides when that is worth paying.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(f'v{CACHE_KEY_VERSION}|{int(codes['D'])}|{int(codes['S'])}|{int(codes['sub'])}|{int(codes['rows'])}|{int(codes['cols'])}|{int(codes['nchunk'])}|{int(bool(codes['rotate']))}|{int(codes['seed'])}'.encode())
    idx = np.ascontiguousarray(codes['indices'])
    h.update(f'|idx:{idx.dtype.str}:{idx.shape}|'.encode())
    h.update(idx.tobytes())
    for cb in codes['codebooks']:
        book = np.ascontiguousarray(cb, dtype=np.float32)
        h.update(f'|cb:{book.shape}|'.encode())
        h.update(book.tobytes())
    return h.hexdigest()

def _slot_fingerprint(codes: dict) -> str:
    """Cheap witness that a cache slot still holds the tensor the caller is asking about.

    An explicit key moves the uniqueness obligation to the caller, and a caller that reuses
    one literal across tensors gets the old bug back under a new name.  Hashing the indices
    on every call would close that completely but costs a pass over 1.5 MB per matvec, which
    is the per-call overhead this cache exists to remove.  The codebook is 2 KB and is
    provably distinct per tensor -- 60 of 60 codebooks on a real shard hash differently, and
    two same-layer projections differ by more than the magnitude of their own values -- so
    geometry plus the codebook separates every tensor the production ladder actually emits.

    Residual, stated rather than hidden: two tensors sharing a geometry AND a byte-identical
    codebook but differing in indices would still collide.  That is the shared-grammar
    packing the ladder does not use; if it ever does, this must become the full content key.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(f'v{CACHE_KEY_VERSION}|{int(codes['D'])}|{int(codes['rows'])}|{int(codes['nchunk'])}|{int(codes['cols'])}'.encode())
    for cb in codes['codebooks']:
        book = np.ascontiguousarray(cb, dtype=np.float32)
        h.update(f'|cb:{book.shape}|'.encode())
        h.update(book.tobytes())
    return h.hexdigest()

def matvec_bytes(codes: dict, *, threadgroup_memory_limit: int=DEFAULT_THREADGROUP_MEMORY, threads: int=THREADS) -> dict:
    """The traffic one matvec really moves, split from the traffic the artifact claims.

    Two numbers were being conflated.  The *logical* artifact is what the 0.87633 BPW claim
    describes: 7-bit packed indices plus one fp16 codebook, which is what the file on disk
    holds.  The *executed* stream is fatter for two reasons that are properties of this
    kernel and not of the representation: indices go up one byte each because there is no
    in-kernel bit unpacking (exactly 8/7 of the billed index stream), and both the codebook
    and, when staged, x are re-read once per threadgroup because threadgroup memory is not
    shared between groups.  Reporting a single integer hid all of that, and the integer it
    reported was the one that flattered the kernel.  Pure function of the geometry, so it
    answers on a machine with no Metal device.
    """
    if int(codes['S']) != 1:
        raise GravityMetalInputError('accounting covers the subspaces == 1 kernel only')
    rows, nchunk = (int(codes['rows']), int(codes['nchunk']))
    cols, D = (int(codes['cols']), int(codes['D']))
    book = np.asarray(codes['codebooks'][0])
    k, sub = (int(book.shape[0]), int(book.shape[1]))
    bits = _index_bits(k)
    stage_x, _ = _stage_plan(k, D, nchunk, threadgroup_memory_limit)
    groups = (rows + threads - 1) // threads
    logical_index = (rows * nchunk * bits + 7) // 8
    logical_book = k * sub * 2
    executed_index = rows * nchunk
    executed_book = groups * k * sub * 2
    executed_x = (groups if stage_x else 1) * nchunk * D * 4
    executed_out = rows * 4
    read = executed_index + executed_book + executed_x
    return {'index_bits_billed': bits, 'threadgroups': groups, 'stage_x': bool(stage_x), 'logical_index_bytes': logical_index, 'logical_codebook_bytes': logical_book, 'logical_artifact_bytes': logical_index + logical_book, 'executed_index_bytes': executed_index, 'executed_codebook_bytes': executed_book, 'executed_activation_bytes': executed_x, 'executed_output_bytes': executed_out, 'executed_read_bytes': read, 'executed_total_bytes': read + executed_out, 'dense_bf16_bytes': rows * cols * 2, 'logical_bpw': (logical_index + logical_book) * 8 / (rows * cols), 'executed_read_bpw': read * 8 / (rows * cols)}

class _ByteBudgetCache:
    """LRU over uploaded tensors, bounded by the device bytes each entry pins.

    The buffers are shared-storage allocations that stay resident until released, so an
    unbounded dict is an unbounded allocation.  Entries carry their own ``pinned_bytes`` so
    the accounting is the real allocation size rather than a re-derivation of it.
    """

    def __init__(self, budget_bytes: int=DEFAULT_CACHE_BUDGET_BYTES) -> None:
        self.budget_bytes = int(budget_bytes)
        self.bytes_pinned = 0
        self.evictions = 0
        self._entries: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: str):
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(self, key: str, entry: dict) -> dict:
        old = self._entries.pop(key, None)
        if old is not None:
            self.bytes_pinned -= old['pinned_bytes']
        self._entries[key] = entry
        self.bytes_pinned += entry['pinned_bytes']
        while self.bytes_pinned > self.budget_bytes and len(self._entries) > 1:
            _, dropped = self._entries.popitem(last=False)
            self.bytes_pinned -= dropped['pinned_bytes']
            self.evictions += 1
        return entry

    def stats(self) -> dict:
        return {'entries': len(self._entries), 'bytes_pinned': self.bytes_pinned, 'budget_bytes': self.budget_bytes, 'evictions': self.evictions, 'keys': list(self._entries)}

    def __len__(self) -> int:
        return len(self._entries)

class GravityMetalDecoder:
    """Compiles the kernel once and reuses the pipeline across calls."""

    def __init__(self, *, cache_budget_bytes: int=DEFAULT_CACHE_BUDGET_BYTES) -> None:
        try:
            import Metal
        except ImportError as exc:
            raise MetalUnavailable('pyobjc-framework-Metal is not installed') from exc
        self._Metal = Metal
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise MetalUnavailable('no default Metal device')
        library, error = self.device.newLibraryWithSource_options_error_(METAL_SOURCE, None, None)
        if library is None:
            raise MetalUnavailable(f'kernel failed to compile: {error}')
        function = library.newFunctionWithName_('gravity_pq_matvec_rows')
        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            raise MetalUnavailable(f'pipeline failed: {error}')
        self.pipeline = pipeline
        self.queue = self.device.newCommandQueue()
        self._cache = _ByteBudgetCache(cache_budget_bytes)
        self.threadgroup_memory_limit = int(self.device.maxThreadgroupMemoryLength())

    @property
    def cache_stats(self) -> dict:
        """Entries, device bytes pinned, the budget and how many entries it has evicted."""
        return self._cache.stats()

    def _buffer(self, array: np.ndarray):
        data = np.ascontiguousarray(array)
        return self.device.newBufferWithBytes_length_options_(data.tobytes(), data.nbytes, self._Metal.MTLResourceStorageModeShared)

    def _cache_tensor(self, codes: dict, key: str):
        """Upload the immutable parts once and keep them.  Per-call allocation was pure overhead."""
        fingerprint = _slot_fingerprint(codes)
        entry = self._cache.get(key)
        if entry is not None:
            if entry['fingerprint'] != fingerprint:
                raise GravityMetalInputError(f'cache key {key!r} already holds a different tensor; keys must be unique per tensor (use content_key(codes) if you want that decided for you)')
            return entry
        D = int(codes['D'])
        rows, nchunk = (int(codes['rows']), int(codes['nchunk']))
        book = np.ascontiguousarray(codes['codebooks'][0], dtype=np.float16)
        k = int(book.shape[0])
        if int(book.shape[1]) != D:
            raise GravityMetalInputError(f'codebook subvector is {int(book.shape[1])} wide but geometry says D = {D}')
        raw = np.ascontiguousarray(codes['indices'])
        if raw.size and int(raw.max()) >= k:
            raise GravityMetalInputError(f'index {int(raw.max())} is out of range for a {k}-entry codebook')
        flat = np.ascontiguousarray(raw, dtype=np.uint8).ravel()
        if flat.size != rows * nchunk:
            raise GravityMetalInputError(f'codes hold {flat.size} indices, geometry says rows*nchunk = {rows * nchunk}')
        indices = np.ascontiguousarray(flat.reshape(rows, nchunk).T)
        stage_x, scratch = _stage_plan(k, D, nchunk, self.threadgroup_memory_limit)
        x_bytes = nchunk * D * 4
        dims = np.array([rows, nchunk, D, k, int(stage_x), 0, 0, 0], dtype=np.uint32)
        entry = {'idx': self._buffer(indices), 'book': self._buffer(book), 'dims': self._buffer(dims), 'y': self.device.newBufferWithLength_options_(rows * 4, self._Metal.MTLResourceStorageModeShared), 'x': self.device.newBufferWithLength_options_(x_bytes, self._Metal.MTLResourceStorageModeShared), 'rows': rows, 'nchunk': nchunk, 'D': D, 'scratch': scratch, 'stage_x': stage_x, 'fingerprint': fingerprint, 'x_bytes': x_bytes, 'pinned_bytes': int(indices.nbytes + book.nbytes + dims.nbytes + rows * 4 + x_bytes)}
        return self._cache.put(key, entry)

    def matvec(self, codes: dict, x: np.ndarray, *, key: str | None=None) -> np.ndarray:
        """y = W_gravity @ x, decoding in registers.  One thread per row, no reduction."""
        if int(codes['S']) != 1:
            raise MetalUnavailable('this kernel handles subspaces == 1 only')
        if codes['rotate']:
            raise MetalUnavailable('rotated geometry is not wired into this kernel yet')
        if int(codes['codebooks'][0].shape[0]) > 256:
            raise MetalUnavailable('k > 256 exceeds the one-byte index this kernel uses')
        if key is None:
            raise GravityMetalInputError('matvec needs an explicit cache key; pass key=<stable id> or key=content_key(codes) to opt into content addressing')
        entry = self._cache_tensor(codes, key)
        if entry['scratch'] > self.threadgroup_memory_limit:
            raise MetalUnavailable('codebook exceeds threadgroup memory')
        rows = entry['rows']
        xv = _validate_x(x, nchunk=entry['nchunk'], D=entry['D'], allocated_bytes=entry['x_bytes'])
        entry['x'].contents().as_buffer(entry['x_bytes'])[:] = xv.tobytes()
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        for slot, buffer in enumerate((entry['idx'], entry['book'], entry['x'], entry['y'], entry['dims'])):
            encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
        encoder.setThreadgroupMemoryLength_atIndex_(entry['scratch'], 0)
        groups = (rows + THREADS - 1) // THREADS
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(self._Metal.MTLSizeMake(groups, 1, 1), self._Metal.MTLSizeMake(THREADS, 1, 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if command.error() is not None:
            raise MetalUnavailable(f'dispatch failed: {command.error()}')
        return np.frombuffer(entry['y'].contents().as_buffer(rows * 4), dtype=np.float32).copy()

    def bytes_read_per_matvec(self, codes: dict) -> dict:
        """Full split of logical artifact bytes versus what this kernel really streams.

        Returns the dict from :func:`matvec_bytes`, not an integer.  The old scalar was
        wrong (it ignored the per-threadgroup re-reads of the codebook and of x) and the
        grep says nothing outside this module ever called it, so there is no shim to keep.
        """
        return matvec_bytes(codes, threadgroup_memory_limit=self.threadgroup_memory_limit)
_DECODER: GravityMetalDecoder | None = None

def decoder() -> GravityMetalDecoder:
    global _DECODER
    if _DECODER is None:
        _DECODER = GravityMetalDecoder()
    return _DECODER
