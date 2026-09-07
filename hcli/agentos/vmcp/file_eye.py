"""Generic file/binary metadata eye (roadmap E.5 / §8.4 CLASSIFY).

This is target-independent format awareness on a real local file: magic,
hash, size, container inventory, section/import inventory when the bytes
support it. It does not decompile, does not call visionmcp, and does not
treat an absent target as an empty success.

    from hcli.agentos.vmcp.file_eye import observe
"""
from __future__ import annotations

import re

import io
import json
import os
import struct
import tarfile
import zipfile
import zlib
from pathlib import Path
from typing import Any, Mapping

from hcli.agentos.vmcp.receipt import content_digest, sha256_bytes, utc_now


NAME = "file.eye"
VERSION = "2"
MAX_STRINGS = 32
MIN_STRING_LEN = 6
MAX_STRING_CHARS = 80
MAX_INVENTORY = 64
MAX_SECTIONS = 64

# Mach-O
MH_MAGIC = 0xFEEDFACE
MH_CIGAM = 0xCEFAEDFE
MH_MAGIC_64 = 0xFEEDFACF
MH_CIGAM_64 = 0xCFFAEDFE
FAT_MAGIC = 0xCAFEBABE
FAT_CIGAM = 0xBEBAFECA
FAT_MAGIC_64 = 0xCAFEBABF
FAT_CIGAM_64 = 0xBFBAFECA
LC_REQ_DYLD = 0x80000000
LC_SEGMENT = 0x01
LC_SYMTAB = 0x02
LC_LOAD_DYLIB = 0x0C
LC_ID_DYLIB = 0x0D
LC_LOAD_WEAK_DYLIB = 0x18
LC_SEGMENT_64 = 0x19
LC_UUID = 0x1B
LC_CODE_SIGNATURE = 0x1D
LC_RPATH = 0x1C | LC_REQ_DYLD
LC_LOAD_DYLINKER = 0x0E
LC_MAIN = 0x28 | LC_REQ_DYLD
MH_FILETYPES = {
    1: "MH_OBJECT",
    2: "MH_EXECUTE",
    3: "MH_FVMLIB",
    4: "MH_CORE",
    5: "MH_PRELOAD",
    6: "MH_DYLIB",
    7: "MH_DYLINKER",
    8: "MH_BUNDLE",
    9: "MH_DYLIB_STUB",
    10: "MH_DSYM",
    11: "MH_KEXT_BUNDLE",
}
CPU_TYPES = {
    7: "x86",
    0x01000007: "x86_64",
    12: "arm",
    0x0100000C: "arm64",
    18: "powerpc",
}

# ELF
ELFMAG = b"\x7fELF"
ET_NAMES = {1: "REL", 2: "EXEC", 3: "DYN", 4: "CORE"}
EM_NAMES = {3: "386", 62: "X86_64", 40: "ARM", 183: "AARCH64", 243: "RISCV"}

# WASM section ids
WASM_SECTIONS = {
    0: "custom",
    1: "type",
    2: "import",
    3: "function",
    4: "table",
    5: "memory",
    6: "global",
    7: "export",
    8: "start",
    9: "element",
    10: "code",
    11: "data",
    12: "datacount",
}


def _u8(buf: bytes, off: int) -> int | None:
    if off >= len(buf):
        return None
    return buf[off]


def _u16(buf: bytes, off: int, endian: str) -> int | None:
    if off + 2 > len(buf):
        return None
    return struct.unpack_from(endian + "H", buf, off)[0]


def _u32(buf: bytes, off: int, endian: str) -> int | None:
    if off + 4 > len(buf):
        return None
    return struct.unpack_from(endian + "I", buf, off)[0]


def _i32(buf: bytes, off: int, endian: str) -> int | None:
    if off + 4 > len(buf):
        return None
    return struct.unpack_from(endian + "i", buf, off)[0]


def _u64(buf: bytes, off: int, endian: str) -> int | None:
    if off + 8 > len(buf):
        return None
    return struct.unpack_from(endian + "Q", buf, off)[0]


def _cstr(buf: bytes, off: int, limit: int = 256) -> str:
    end = buf.find(b"\x00", off, min(len(buf), off + limit))
    if end < 0:
        end = min(len(buf), off + limit)
    return buf[off:end].decode("utf-8", errors="replace")


def _limitations_for(path: Path, *, truncated: bool) -> list[str]:
    if not path.exists():
        return ["TARGET_ABSENT"]
    if path.is_dir():
        return ["TARGET_IS_DIRECTORY"]
    if not path.is_file():
        return ["TARGET_NOT_A_FILE"]
    out: list[str] = []
    if truncated:
        out.append("TRUNCATED_TO_MAX_BYTES")
    return out


def strings_inventory(data: bytes, *, limit: int = MAX_STRINGS) -> list[str]:
    out: list[str] = []
    buf = bytearray()
    for byte in data:
        if 32 <= byte < 127:
            buf.append(byte)
            continue
        if len(buf) >= MIN_STRING_LEN:
            out.append(bytes(buf)[:MAX_STRING_CHARS].decode("ascii"))
            if len(out) >= limit:
                return out
        buf.clear()
    if len(buf) >= MIN_STRING_LEN and len(out) < limit:
        out.append(bytes(buf)[:MAX_STRING_CHARS].decode("ascii"))
    return out


def _png(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    info: dict[str, Any] = {"kind": "png", "container": "png"}
    if len(data) >= 33 and data[12:16] == b"IHDR":
        w, h, bit_depth, color_type = struct.unpack_from(">IIBB", data, 16)
        info["width"] = int(w)
        info["height"] = int(h)
        info["bit_depth"] = int(bit_depth)
        info["color_type"] = int(color_type)
    chunks: list[str] = []
    off = 8
    while off + 8 <= len(data) and len(chunks) < MAX_INVENTORY:
        length = _u32(data, off, ">")
        tag = data[off + 4 : off + 8]
        if length is None or off + 12 + length > len(data):
            break
        try:
            chunks.append(tag.decode("ascii"))
        except UnicodeDecodeError:
            chunks.append(tag.hex())
        off += 12 + length
        if tag == b"IEND":
            break
    if chunks:
        info["chunks"] = chunks
    return info


# SOF markers that carry frame geometry. DHT/DAC/DRI and the RST range are NOT
# frame headers and must not be read as one; SOF4 (0xC4) is DHT and SOF12 (0xCC)
# is DAC, which is why the set is enumerated rather than expressed as a range.
_JPEG_SOF = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _jpeg_geometry(data: bytes) -> tuple[int, int] | None:
    """Width/height from the first SOF frame header, or None.

    Walks the marker chain rather than scanning for a byte pattern: a scan can
    match compressed entropy data and report a confident wrong answer.
    """
    i = 2  # past SOI
    n = len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA or marker == 0xD9:  # start of scan / end of image
            return None
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if seg_len < 2 or i + 2 + seg_len > n:
            return None
        if marker in _JPEG_SOF:
            if i + 9 > n:
                return None
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            if width > 0 and height > 0:
                return width, height
            return None
        i += 2 + seg_len
    return None


def _jpeg(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(b"\xff\xd8\xff"):
        return None
    info: dict[str, Any] = {"kind": "jpeg", "container": "jpeg"}
    geom = _jpeg_geometry(data)
    if geom is not None:
        info["width"], info["height"] = geom
    return info


def verify_image_geometry(path: str | Path, *, timeout: float = 20.0) -> dict[str, Any]:
    """Cross-check this eye's image geometry against an INDEPENDENT local tool.

    The directive's VMCP priority is "exact tool-grounded verification". Parsing
    a header and believing yourself is not verification; agreeing with a second
    implementation that read the same bytes is.

    Ground truth here is `sips`, which is native and not derived from this code.
    Disagreement is reported as DISAGREE and the geometry is NOT claimed --
    a confident wrong answer about what is on screen is the failure mode this
    whole subsystem exists to avoid.
    """
    import subprocess

    target = Path(path)
    out: dict[str, Any] = {
        "path": str(target),
        "eye": None,
        "ground_truth": None,
        "ground_truth_tool": "sips",
        "agree": None,
        "verdict": "UNKNOWN",
    }
    if not target.is_file():
        out["verdict"] = "NO_TARGET"
        return out

    data = target.read_bytes()
    classified = _jpeg(data) or _png(data)
    if classified is None:
        out["verdict"] = "NOT_AN_IMAGE_BY_MAGIC"
        return out
    if "width" not in classified or "height" not in classified:
        out["eye"] = classified
        out["verdict"] = "EYE_HAS_NO_GEOMETRY"
        return out
    out["eye"] = {"kind": classified["kind"],
                  "width": classified["width"], "height": classified["height"]}

    try:
        proc = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(target)],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        out["verdict"] = "GROUND_TRUTH_UNAVAILABLE"
        out["ground_truth_error"] = f"{type(exc).__name__}: {exc}"
        return out

    w = re.search(r"pixelWidth:\s*(\d+)", proc.stdout or "")
    h = re.search(r"pixelHeight:\s*(\d+)", proc.stdout or "")
    if not (w and h):
        out["verdict"] = "GROUND_TRUTH_UNAVAILABLE"
        return out

    truth = {"width": int(w.group(1)), "height": int(h.group(1))}
    out["ground_truth"] = truth
    out["agree"] = (truth["width"] == classified["width"]
                    and truth["height"] == classified["height"])
    out["verdict"] = "AGREE" if out["agree"] else "DISAGREE"
    return out


def _gif(data: bytes) -> dict[str, Any] | None:
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return {"kind": "gif", "container": "gif", "version": data[3:6].decode("ascii")}
    return None


def _pdf(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(b"%PDF-"):
        return None
    ver = data[5:8].decode("ascii", errors="replace")
    return {"kind": "pdf", "container": "pdf", "version": ver}


def _sqlite(data: bytes) -> dict[str, Any] | None:
    if data.startswith(b"SQLite format 3\x00"):
        return {"kind": "sqlite", "container": "sqlite"}
    return None


def _wasm(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(b"\x00asm"):
        return None
    version = _u32(data, 4, "<") or 0
    sections: list[dict[str, Any]] = []
    off = 8
    while off + 2 <= len(data) and len(sections) < MAX_SECTIONS:
        sid = data[off]
        off += 1
        # LEB128 size
        size = 0
        shift = 0
        while off < len(data):
            byte = data[off]
            off += 1
            size |= (byte & 0x7F) << shift
            if byte & 0x80 == 0:
                break
            shift += 7
            if shift > 28:
                break
        sections.append(
            {
                "id": int(sid),
                "name": WASM_SECTIONS.get(sid, f"unknown-{sid}"),
                "size": int(size),
            }
        )
        off += size
        if off > len(data):
            break
    return {
        "kind": "wasm",
        "container": "wasm",
        "version": int(version),
        "sections": sections,
    }


def _zip_inventory(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(b"PK\x03\x04") and not data.startswith(b"PK\x05\x06"):
        return None
    info: dict[str, Any] = {"kind": "zip", "container": "zip"}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()[:MAX_INVENTORY]
            info["entries"] = names
            info["n_entries"] = len(zf.namelist())
            if any(n.endswith("/") or n.endswith("\\") for n in names):
                info["has_directories"] = True
            if any(n.endswith(".wasm") for n in names):
                info["embedded"] = ["wasm"]
            if "[Content_Types].xml" in zf.namelist():
                info["kind"] = "office-openxml"
            if "META-INF/MANIFEST.MF" in zf.namelist():
                info["kind"] = "jar"
    except zipfile.BadZipFile:
        info["limitations"] = ["ZIP_PARSE_INCOMPLETE"]
    return info


def _gzip_inventory(data: bytes, *, _depth: int = 0) -> dict[str, Any] | None:
    if not data.startswith(b"\x1f\x8b"):
        return None
    info: dict[str, Any] = {"kind": "gzip", "container": "gzip"}
    try:
        dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
        inner = dec.decompress(data, 1_000_000)
        info["inner_size"] = len(inner)
        info["inner_sha256"] = sha256_bytes(inner)
        inner_kind = classify_bytes(inner[:4096], _depth=_depth + 1).get("kind")
        if inner_kind and inner_kind != "unknown":
            info["inner_kind"] = inner_kind
    except (OSError, EOFError, OverflowError, zlib.error):
        info["limitations"] = ["GZIP_PARSE_INCOMPLETE"]
    return info


def _tar_inventory(data: bytes) -> dict[str, Any] | None:
    if len(data) >= 265 and data[257:262] == b"ustar":
        pass
    else:
        return None
    info: dict[str, Any] = {"kind": "tar", "container": "tar"}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            names = [m.name for m in tf.getmembers()[:MAX_INVENTORY]]
            info["entries"] = names
            info["n_entries"] = len(tf.getmembers())
    except tarfile.TarError:
        info["limitations"] = ["TAR_PARSE_INCOMPLETE"]
    return info


def _macho_cpu(value: int) -> str:
    return CPU_TYPES.get(value, f"cpu-{value:#x}")


def _parse_macho(data: bytes, off: int, endian: str, header64: bool) -> dict[str, Any]:
    hdr_size = 32 if header64 else 28
    if off + hdr_size > len(data):
        return {"kind": "mach-o", "limitations": ["MACHO_HEADER_TRUNCATED"]}
    cputype = _i32(data, off + 4, endian) or 0
    filetype = _u32(data, off + 12, endian) or 0
    ncmds = _u32(data, off + 16, endian) or 0
    sizeofcmds = _u32(data, off + 20, endian) or 0
    cursor = off + hdr_size
    sections: list[str] = []
    dylibs: list[str] = []
    uuid = None
    has_code_sig = False
    ncmds = min(int(ncmds), 256)
    end_cmds = min(len(data), cursor + int(sizeofcmds))
    for _ in range(ncmds):
        if cursor + 8 > end_cmds:
            break
        cmd = _u32(data, cursor, endian) or 0
        cmdsize = _u32(data, cursor + 4, endian) or 0
        if cmdsize < 8:
            break
        body = cursor
        if cmd in {LC_SEGMENT, LC_SEGMENT_64}:
            name = data[body + 8 : body + 24].split(b"\x00", 1)[0].decode("ascii", errors="replace")
            nsects_off = 56 if cmd == LC_SEGMENT else 64
            nsects = _u32(data, body + nsects_off, endian) or 0
            sect_off = (body + 56) if cmd == LC_SEGMENT else (body + 72)
            sect_size = 68 if cmd == LC_SEGMENT else 80
            for i in range(min(int(nsects), MAX_SECTIONS)):
                s_at = sect_off + i * sect_size
                if s_at + 16 > len(data):
                    break
                sname = data[s_at : s_at + 16].split(b"\x00", 1)[0].decode("ascii", errors="replace")
                if sname:
                    sections.append(f"{name}.{sname}" if name else sname)
        elif cmd in {LC_LOAD_DYLIB, LC_ID_DYLIB, LC_LOAD_WEAK_DYLIB}:
            name_off = _u32(data, body + 8, endian)
            if name_off:
                dylibs.append(_cstr(data, body + int(name_off)))
        elif cmd == LC_UUID and cmdsize >= 24:
            uuid = data[body + 8 : body + 24].hex()
        elif cmd == LC_CODE_SIGNATURE:
            has_code_sig = True
        cursor += int(cmdsize)
        if len(sections) >= MAX_SECTIONS:
            sections = sections[:MAX_SECTIONS]
    return {
        "kind": "mach-o",
        "container": "mach-o",
        "cpu": _macho_cpu(int(cputype) & 0xFFFFFFFF),
        "filetype": MH_FILETYPES.get(int(filetype), f"type-{filetype}"),
        "ncmds": int(ncmds),
        "sections": sections[:MAX_SECTIONS],
        "dylibs": dylibs[:MAX_INVENTORY],
        "uuid": uuid,
        "code_signature": has_code_sig,
        "header64": header64,
    }


def _macho(data: bytes) -> dict[str, Any] | None:
    if len(data) < 4:
        return None
    magic = _u32(data, 0, "<") or 0
    if magic in {MH_MAGIC, MH_MAGIC_64}:
        return _parse_macho(data, 0, "<", magic == MH_MAGIC_64)
    if magic in {MH_CIGAM, MH_CIGAM_64}:
        return _parse_macho(data, 0, ">", magic == MH_CIGAM_64)
    be = _u32(data, 0, ">") or 0
    if be in {FAT_MAGIC, FAT_MAGIC_64, FAT_CIGAM, FAT_CIGAM_64}:
        endian = ">" if be in {FAT_MAGIC, FAT_MAGIC_64} else "<"
        nfat = _u32(data, 4, endian) or 0
        if nfat > 32:
            # Java class files also start with CAFEBABE.
            return {
                "kind": "java-class",
                "container": "jvm",
                "note": "cafebabe with nfat_arch>32 treated as Java, not fat Mach-O",
            }
        slices: list[dict[str, Any]] = []
        arch_size = 20 if be in {FAT_MAGIC, FAT_CIGAM} else 32
        for i in range(min(int(nfat), 8)):
            aoff = 8 + i * arch_size
            if be in {FAT_MAGIC, FAT_CIGAM}:
                cpu = _i32(data, aoff, endian) or 0
                offset = _u32(data, aoff + 8, endian) or 0
            else:
                cpu = _i32(data, aoff, endian) or 0
                offset = _u64(data, aoff + 8, endian) or 0
            slice_info = {"cpu": _macho_cpu(int(cpu) & 0xFFFFFFFF), "offset": int(offset)}
            if 0 < int(offset) < len(data):
                inner = _macho(data[int(offset) :])
                if inner:
                    slice_info["inner"] = {
                        k: inner[k] for k in ("kind", "cpu", "filetype", "dylibs") if k in inner
                    }
            slices.append(slice_info)
        return {
            "kind": "mach-o-fat",
            "container": "mach-o",
            "nfat_arch": int(nfat),
            "slices": slices,
        }
    return None


def _elf(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(ELFMAG) or len(data) < 16:
        return None
    ei_class = data[4]
    ei_data = data[5]
    endian = "<" if ei_data == 1 else ">"
    info: dict[str, Any] = {
        "kind": "elf",
        "container": "elf",
        "class": {1: "32", 2: "64"}.get(ei_class, str(ei_class)),
        "endian": "le" if endian == "<" else "be",
    }
    if ei_class == 1 and len(data) >= 52:
        e_type = _u16(data, 16, endian) or 0
        e_machine = _u16(data, 18, endian) or 0
        e_shoff = _u32(data, 32, endian) or 0
        e_shentsize = _u16(data, 46, endian) or 0
        e_shnum = _u16(data, 48, endian) or 0
        e_shstrndx = _u16(data, 50, endian) or 0
    elif ei_class == 2 and len(data) >= 64:
        e_type = _u16(data, 16, endian) or 0
        e_machine = _u16(data, 18, endian) or 0
        e_shoff = _u64(data, 40, endian) or 0
        e_shentsize = _u16(data, 58, endian) or 0
        e_shnum = _u16(data, 60, endian) or 0
        e_shstrndx = _u16(data, 62, endian) or 0
    else:
        return info
    info["type"] = ET_NAMES.get(int(e_type), f"ET_{e_type}")
    info["machine"] = EM_NAMES.get(int(e_machine), f"EM_{e_machine}")
    sections: list[str] = []
    imports: list[str] = []
    shoff = int(e_shoff)
    shentsize = int(e_shentsize)
    shnum = min(int(e_shnum), 128)
    if shoff and shentsize >= 40 and shoff + shentsize <= len(data):
        str_off = None
        str_size = None
        shstr_off = shoff + int(e_shstrndx) * shentsize
        if ei_class == 2:
            str_off = _u64(data, shstr_off + 24, endian)
            str_size = _u64(data, shstr_off + 32, endian)
        else:
            str_off = _u32(data, shstr_off + 16, endian)
            str_size = _u32(data, shstr_off + 20, endian)
        strtab = b""
        if str_off and str_size:
            start = int(str_off)
            end = min(len(data), start + int(str_size))
            strtab = data[start:end]
        for i in range(shnum):
            at = shoff + i * shentsize
            if at + 4 > len(data):
                break
            name_off = _u32(data, at, endian) or 0
            if strtab and name_off < len(strtab):
                sections.append(_cstr(strtab, int(name_off), 64))
        # DT_NEEDED lives in dynamic + dynstr; best-effort via .dynstr printable
        if b"\x00" in strtab:
            pass
    if sections:
        info["sections"] = [s for s in sections if s][:MAX_SECTIONS]
    if imports:
        info["imports"] = imports[:MAX_INVENTORY]
    return info


def _pe(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(b"MZ") or len(data) < 64:
        return None
    e_lfanew = _u32(data, 60, "<") or 0
    if e_lfanew + 24 > len(data) or data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
        return {"kind": "mz", "container": "dos-mz"}
    machine = _u16(data, e_lfanew + 4, "<") or 0
    n_sections = _u16(data, e_lfanew + 6, "<") or 0
    opt_size = _u16(data, e_lfanew + 20, "<") or 0
    magic = _u16(data, e_lfanew + 24, "<") or 0
    sections: list[str] = []
    sec_off = e_lfanew + 24 + int(opt_size)
    for i in range(min(int(n_sections), MAX_SECTIONS)):
        at = sec_off + i * 40
        if at + 8 > len(data):
            break
        name = data[at : at + 8].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        if name:
            sections.append(name)
    machine_name = {0x14C: "i386", 0x8664: "x64", 0xAA64: "arm64"}.get(int(machine), hex(machine))
    pe32plus = int(magic) == 0x20B
    return {
        "kind": "pe",
        "container": "pe",
        "machine": machine_name,
        "pe32plus": pe32plus,
        "sections": sections,
        "n_sections": int(n_sections),
    }


def _shebang(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(b"#!"):
        return None
    line = data.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
    return {"kind": "script", "container": "text", "shebang": line[:200]}


def _textish(data: bytes) -> dict[str, Any] | None:
    if not data:
        return {"kind": "empty", "container": "empty"}
    if b"\x00" in data[:4096]:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    stripped = text.lstrip()
    info: dict[str, Any] = {"kind": "text", "container": "text", "encoding": "utf-8"}
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(text)
            info["kind"] = "json"
        except json.JSONDecodeError:
            pass
    elif stripped.startswith("<?xml") or stripped.startswith("<"):
        info["kind"] = "xml"
    elif stripped.startswith("---") or stripped.startswith("%YAML"):
        info["kind"] = "yaml"
    return info


def classify_bytes(data: bytes, *, _depth: int = 0) -> dict[str, Any]:
    """Classify a byte buffer. Never returns an empty success."""
    if _depth > 2:
        return {
            "kind": "binary",
            "container": "unknown",
            "magic": data[:8].hex() if data else "",
            "limitations": ["CLASSIFY_DEPTH"],
        }
    for parser in (
        _png,
        _jpeg,
        _gif,
        _pdf,
        _sqlite,
        _wasm,
        _zip_inventory,
        _gzip_inventory,
        _tar_inventory,
        _macho,
        _elf,
        _pe,
        _shebang,
        _textish,
    ):
        got = parser(data, _depth=_depth) if parser is _gzip_inventory else parser(data)
        if got:
            got.setdefault("magic", data[:8].hex() if data else "")
            return got
    kind = "unknown"
    if data:
        kind = "binary"
    return {
        "kind": kind,
        "container": "unknown",
        "magic": data[:8].hex() if data else "",
        "limitations": ["UNRECOGNIZED_MAGIC"] if data else ["EMPTY"],
    }


def observe(
    path: str | os.PathLike[str] | None = None,
    *,
    max_bytes: int = 8_000_000,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Observe + classify a real local file. TARGET_ABSENT is a classification."""
    args = dict(arguments or {})
    raw = path if path is not None else args.get("path")
    started = utc_now()
    if raw is None or str(raw) == "":
        return {
            "act": "see",
            "organ": "file",
            "status": "CONNECTED",
            "present": False,
            "limitations": ["PATH_REQUIRED"],
            "empty_success": False,
            "looked": False,
            "execution": "REAL",
            "evidence_tier": "FUNCTIONAL_SIM",
            "note": "see requires a path; refusing to invent a target",
        }
    max_bytes = int(args.get("max_bytes") or max_bytes)
    target = Path(str(raw))
    try:
        resolved = target.resolve()
    except OSError:
        resolved = target
    if not resolved.is_file():
        limitations = _limitations_for(resolved, truncated=False)
        return {
            "act": "see",
            "organ": "file",
            "status": "CONNECTED",
            "present": False,
            "path": str(resolved),
            "size": None,
            "mode": None,
            "sha256": None,
            "classification": {"kind": "absent" if "TARGET_ABSENT" in limitations else "not-a-file"},
            "limitations": limitations,
            "empty_success": False,
            "looked": True,
            "execution": "REAL",
            "evidence_tier": "FUNCTIONAL_SIM",
        }
    data = resolved.read_bytes()
    full_size = len(data)
    truncated = full_size > max_bytes
    hashed = data if not truncated else data[:max_bytes]
    digest = sha256_bytes(hashed)
    st = resolved.stat()
    classification = classify_bytes(hashed)
    strings = []
    if classification.get("kind") not in {"png", "jpeg", "gif", "gzip"}:
        strings = strings_inventory(hashed[: min(len(hashed), 256_000)])
    limitations = _limitations_for(resolved, truncated=truncated)
    extra_lim = classification.pop("limitations", None)
    if extra_lim:
        for item in extra_lim:
            if item not in limitations:
                limitations.append(item)
    evidence = {
        "present": True,
        "path": str(resolved),
        "size": int(st.st_size),
        "mode": oct(st.st_mode),
        "sha256": digest,
        "hashed_bytes": len(hashed),
        "classification": classification,
    }
    result = {
        "act": "see",
        "organ": "file",
        "status": "CONNECTED",
        "present": True,
        "path": str(resolved),
        "size": int(st.st_size),
        "mode": oct(st.st_mode),
        "sha256": digest,
        "hashed_bytes": len(hashed),
        "classification": classification,
        "kind": classification.get("kind"),
        "container": classification.get("container"),
        "strings": strings,
        "limitations": limitations,
        "empty_success": False,
        "looked": True,
        "execution": "REAL",
        "evidence_tier": "FUNCTIONAL_SIM",
        "gpu_authority": False,
        "network_used": False,
        "matches_file_eye_fields": ["present", "path", "size", "mode", "sha256"],
        "started_at": started,
        "deep_digest": content_digest(evidence),
        "artifacts": [str(resolved)],
        "evidence": [evidence],
        "residuals": limitations,
        "next_actions": [],
        "note": (
            "stdlib magic/header classification of a real local file; not "
            "CaptureBus, not Ghidra, not a GPU measurement"
        ),
    }
    return result
