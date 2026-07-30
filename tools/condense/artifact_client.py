#!/usr/bin/env python3.12
"""Official Core A client: one-shot owned FFI (ABI v1). No probe/replay.
`performance-smoke` matches old gravity_format selftest; default adds one-shot proof.
"""
import ctypes, json, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORMAT_VERSION, HEADER_SCHEMA = 1, "hawking.gravity.shard_header.v1"
ABI_VERSION = 1
_LIB = _LIB_LOCK = None

class GravityFormatError(Exception):
    pass

def _core_lib():
    global _LIB, _LIB_LOCK
    if _LIB is not None:
        return _LIB
    if _LIB_LOCK is None:
        import threading
        _LIB_LOCK = threading.Lock()
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        names = ("libhawking_core.dylib", "libhawking_core.so", "hawking_core.dll")
        cands = [os.environ["HAWKING_CORE_LIB"]] if "HAWKING_CORE_LIB" in os.environ else []
        roots = ([os.environ["CARGO_TARGET_DIR"]] if "CARGO_TARGET_DIR" in os.environ else []) + [_REPO, os.getcwd()]
        cands += [os.path.join(r, s, n) for r in roots for s in ("release", "debug", "target/release", "target/debug") for n in names]
        owned = [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t)]
        cu, cs, cp = ctypes.c_uint, ctypes.c_size_t, ctypes.c_char_p
        p8, p32 = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32)
        specs = [
            ("hawking_pack_indices_owned", [p32, cs, cu]),
            ("hawking_write_shard_owned", [cp, p8, cs, p8, cs]),
            ("hawking_verify_owned", [cp]),
            ("hawking_read_header_owned", [cp]),
            ("hawking_read_tensor_owned", [cp, cp, ctypes.c_int]),
            ("hawking_open_shard_owned", [cp]),
        ]
        last, seen = None, set()
        for path in cands:
            if path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            try:
                lib = ctypes.CDLL(path)
                if not hasattr(lib, "hawking_artifact_abi_version"):
                    raise GravityFormatError(f"{path}: no ABI")
                lib.hawking_artifact_abi_version.argtypes = []
                lib.hawking_artifact_abi_version.restype = cu
                if int(lib.hawking_artifact_abi_version()) != ABI_VERSION:
                    raise GravityFormatError("ABI mismatch")
                for name, extra in specs:
                    fn = getattr(lib, name)
                    fn.argtypes = list(extra) + owned
                    fn.restype = ctypes.c_int
                free = lib.hawking_artifact_free
                free.argtypes = [ctypes.POINTER(ctypes.c_uint8), cs]
                free.restype = None
                _LIB = lib
                return _LIB
            except (OSError, GravityFormatError, AttributeError) as e:
                last = e
        raise GravityFormatError(f"libhawking_core not loadable: {last}")

def _call_owned(name, *args, as_json=True):
    """One FFI call; copy Rust-owned bytes; free in finally (no leak on JSON errors)."""
    fn = getattr(_core_lib(), name)
    out_ptr = ctypes.POINTER(ctypes.c_uint8)()
    out_len = ctypes.c_size_t(0)
    rc = fn(*args, ctypes.byref(out_ptr), ctypes.byref(out_len))
    lib = _core_lib()
    try:
        if rc == 7:
            raise GravityFormatError("unknown tensor")
        if rc != 0:
            raise GravityFormatError(f"rc={rc}")
        n = int(out_len.value)
        raw = b"" if n == 0 or not out_ptr else ctypes.string_at(out_ptr, n)
        return json.loads(raw) if as_json else raw
    finally:
        lib.hawking_artifact_free(out_ptr, int(out_len.value))

def read_header(path):
    return _call_owned("hawking_read_header_owned", os.fsencode(os.fspath(path)))

def verify(path):
    return _call_owned("hawking_verify_owned", os.fsencode(os.fspath(path)))

def open_shard(path):
    p = _call_owned("hawking_open_shard_owned", os.fsencode(os.fspath(path)))
    return p["header"], int(p["body_offset"])

def read_tensor(path, name, *, verify_hash=True):
    return _call_owned(
        "hawking_read_tensor_owned", os.fsencode(os.fspath(path)), name.encode(),
        1 if verify_hash else 0, as_json=False,
    )

def iter_tensors(path):
    path = os.fspath(path)
    for e in sorted(read_header(path)["tensors"], key=lambda t: int(t["offset"])):
        yield e, read_tensor(path, e["name"], verify_hash=False)

def write_shard(path, payloads, *, model, compression, tokenizer=None,
                architecture=None, shard=None, telemetry=None):
    path = os.fspath(path)
    tensors, lengths, blobs = [], [], []
    for d, blob in payloads:
        tensors.append({k: v for k, v in dict(d).items()
                        if k not in ("offset", "bytes", "sha256")})
        lengths.append(len(blob))
        blobs.append(blob)
    body = b"".join(blobs)
    meta = json.dumps({"model": model, "compression": compression,
                       "tokenizer": tokenizer or {}, "architecture": architecture or {},
                       "shard": shard or {}, "tensors": tensors, "payload_lengths": lengths},
                      separators=(",", ":")).encode()
    mb = (ctypes.c_uint8 * len(meta)).from_buffer_copy(meta) if meta else (ctypes.c_uint8 * 0)()
    bb = (ctypes.c_uint8 * len(body)).from_buffer_copy(body) if body else (ctypes.c_uint8 * 0)()
    bp = (ctypes.cast(bb, ctypes.POINTER(ctypes.c_uint8))
          if body else ctypes.POINTER(ctypes.c_uint8)())
    header = _call_owned(
        "hawking_write_shard_owned", os.fsencode(path), mb, len(meta), bp, len(body),
    )
    if telemetry is not None:
        telemetry["body_bytes"] = len(body)
    return header

def pack_indices(indices, bits):
    import numpy as np
    flat = np.ascontiguousarray(indices, dtype=np.uint32).ravel()
    ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    return _call_owned("hawking_pack_indices_owned", ptr, int(flat.size), int(bits), as_json=False)

def selftest(performance_smoke: bool = False):
    payloads = [({"name": f"model.layers.{i}.weight", "elements": 1024, "shape": [32, 32],
                  "category": "routed_expert", "codec": "pq", "bpw": (128 + i) * 8 / 1024},
                 bytes([(i * 37 + b) % 256 for b in range(128 + i)])) for i in range(4)]
    c_bits = sum(len(b) for _, b in payloads) * 8
    c_el = sum(d["elements"] for d, _ in payloads)
    bpw = c_bits / c_el
    base = dict(model={"repo": "zai-org/GLM-5.2", "revision": "b" * 40},
                architecture={"type": "GlmMoeDsaForCausalLM", "hidden_layers": 78},
                tokenizer={"kind": "reference", "source": "zai-org/GLM-5.2"},
                compression={"codec": "gravity-pq", "packed_bpw": bpw},
                shard={"index": 1, "count": 1})
    organ = ({"name": "model.layers.0.mlp.gate.weight", "elements": 64, "shape": [2, 32],
              "category": "router", "codec": "native.bf16",
              "terminal_state": "PROTECTED_SOURCE_NATIVE", "bpw": 16.0}, bytes([0, 0x11]) * 64)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"hawking-ac-{os.getpid()}-{id(payloads):x}")
    os.mkdir(tmp)
    try:
        p = os.path.join(tmp, "model-00001-of-00001.gravity")
        write_shard(p, payloads, **base)
        h = read_header(p)
        assert h["schema"] == HEADER_SCHEMA and h["model"]["repo"] == "zai-org/GLM-5.2"
        assert h["architecture"]["hidden_layers"] == 78 and len(h["tensors"]) == 4
        for d, blob in payloads:
            assert read_tensor(p, d["name"]) == blob, d["name"]
        st = list(iter_tensors(p))
        assert [d["name"] for d, _ in st] == [d["name"] for d, _ in payloads]
        assert [b for _, b in st] == [b for _, b in payloads]
        report = verify(p)
        assert report["ok"] and report["rate_self_consistent"], report
        with open(p, "rb") as fh:
            raw = bytearray(fh.read())
        raw[-1] ^= 0xFF
        tp = os.path.join(tmp, "tampered.gravity")
        with open(tp, "wb") as fh:
            fh.write(raw)
        dmg = verify(tp)
        assert not dmg["ok"] and dmg["bad_tensors"] == [payloads[-1][0]["name"]]
        mxy = {"repo": "x", "revision": "y"}
        lp = os.path.join(tmp, "lying.gravity")
        write_shard(lp, payloads, model=mxy, compression={"codec": "gravity-pq", "packed_bpw": 0.001})
        assert not verify(lp)["rate_self_consistent"]
        ab, ae = c_bits + len(organ[1]) * 8, c_el + organ[0]["elements"]
        mp = os.path.join(tmp, "mixed.gravity")
        write_shard(mp, payloads + [organ], model=mxy,
                    compression={"codec": "gravity-pq", "packed_bpw": c_bits / c_el, "complete_bpw": ab / ae})
        r = verify(mp)
        assert r["ok"] and r["observed_complete_bpw"] > r["observed_packed_bpw"]
        assert r["tensors_without_payload"] == [] and read_tensor(mp, organ[0]["name"]) == organ[1]
        hollow = ({"name": "model.layers.0.input_layernorm.weight", "elements": 6144,
                   "codec": "native.bf16", "bpw": 16.0}, b"")
        hp = os.path.join(tmp, "hollow.gravity")
        write_shard(hp, payloads + [hollow], model={"repo": "x", "revision": "y"},
                    compression={"codec": "gravity-pq", "packed_bpw": bpw})
        hr = verify(hp)
        assert not hr["ok"] and hr["tensors_without_payload"] == [hollow[0]["name"]]
        if not performance_smoke:
            big = [({"name": f"t{i}", "elements": 4096, "shape": [64, 64],
                     "category": "routed_expert", "codec": "pq", "bpw": 8.0},
                    bytes([(i * 13 + b) % 256 for b in range(4096)])) for i in range(32)]
            bp, counts, real = os.path.join(tmp, "oneshot.gravity"), {}, _call_owned
            g = globals()
            g["_call_owned"] = (lambda name, *a, _r=real, _c=counts, **k: (
                _c.__setitem__(name, _c.get(name, 0) + 1), _r(name, *a, **k))[1])
            try:
                steps = [
                    ("hawking_write_shard_owned", lambda: write_shard(
                        bp, big, model={"repo": "x", "revision": "y"},
                        compression={"codec": "gravity-pq", "packed_bpw": 8.0})),
                    ("hawking_verify_owned", lambda: verify(bp)),
                    ("hawking_read_header_owned", lambda: read_header(bp)),
                    ("hawking_open_shard_owned", lambda: open_shard(bp)),
                    ("hawking_read_tensor_owned", lambda: read_tensor(bp, "t0", verify_hash=True)),
                    ("hawking_pack_indices_owned", lambda: pack_indices([1, 2, 3, 4], 4)),
                ]
                results = []
                for sym, fn in steps:
                    counts.clear()
                    results.append(fn())
                    assert counts == {sym: 1}, counts
                assert results[1]["ok"] and len(results[4]) == 4096
            finally:
                g["_call_owned"] = real
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    out = {"selftest": "PASS", "format": "gravity", "version": FORMAT_VERSION,
           "prefix_bytes": 20, "boundary": "ctypes", "abi_version": ABI_VERSION}
    out.update(
        {"seek_by_name": True, "integrity_two_level": True, "false_rate_claim_rejected": True,
         "native_organs_carried": True, "empty_payload_rejected": True, "mode": "performance-smoke"}
        if performance_smoke else {"native_organs": True, "one_shot_owned": True}
    )
    print(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "verify" and len(sys.argv) > 2:
        print(json.dumps(verify(sys.argv[2]), indent=2))
        raise SystemExit(0)
    if cmd == "header" and len(sys.argv) > 2:
        print(json.dumps(read_header(sys.argv[2]), indent=2)[:4000])
        raise SystemExit(0)
    if cmd == "selftest":
        raise SystemExit(selftest())
    if cmd == "performance-smoke":
        raise SystemExit(selftest(performance_smoke=True))
    sys.stderr.write("usage: artifact_client.py [selftest|performance-smoke|header PATH|verify PATH]\n")
    raise SystemExit(2)
