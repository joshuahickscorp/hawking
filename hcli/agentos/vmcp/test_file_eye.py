"""File/binary eye: real classification of real bytes, never empty success."""
from __future__ import annotations

import gzip
import hashlib
import io
import zipfile
from pathlib import Path

from hcli.agentos.vmcp.file_eye import classify_bytes, observe


def _png() -> bytes:
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def test_observe_classifies_real_temp_files(tmp_path: Path):
    png = tmp_path / "x.png"
    png.write_bytes(_png())
    seen = observe(png)
    assert seen["present"] is True
    assert seen["kind"] == "png"
    assert seen["classification"]["width"] == 1
    assert seen["empty_success"] is False
    assert seen["execution"] == "REAL"
    assert seen["sha256"] == hashlib.sha256(png.read_bytes()).hexdigest()

    zpath = tmp_path / "x.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi\n")
    zpath.write_bytes(buf.getvalue())
    zseen = observe(zpath)
    assert zseen["kind"] == "zip"
    assert "hello.txt" in zseen["classification"]["entries"]

    gz = tmp_path / "x.gz"
    gz.write_bytes(gzip.compress(b"hello-gzip\n"))
    gseen = observe(gz)
    assert gseen["kind"] == "gzip"

    wasm = tmp_path / "x.wasm"
    wasm.write_bytes(b"\x00asm" + b"\x01\x00\x00\x00")
    wseen = observe(wasm)
    assert wseen["kind"] == "wasm"


def test_observe_classifies_bin_echo_on_this_disk():
    echo = Path("/bin/echo")
    if not echo.is_file():
        return
    seen = observe(echo)
    assert seen["present"] is True
    assert seen["kind"] in {"mach-o", "mach-o-fat", "elf", "pe"}
    assert seen["looked"] is True
    assert seen["empty_success"] is False


def test_absent_is_target_absent(tmp_path: Path):
    missing = observe(tmp_path / "nope.bin")
    assert missing["present"] is False
    assert "TARGET_ABSENT" in missing["limitations"]
    assert missing["empty_success"] is False
    assert missing["looked"] is True


def test_unrecognized_magic_is_binary_not_empty():
    got = classify_bytes(b"\x00\x01\x02\x03XXXX")
    assert got["kind"] in {"binary", "unknown"}
    assert got.get("limitations")


def test_png_magic_mutation_drops_png_kind():
    data = _png()
    assert classify_bytes(data)["kind"] == "png"
    mutated = b"\x00" + data[1:]
    assert classify_bytes(mutated)["kind"] != "png"
