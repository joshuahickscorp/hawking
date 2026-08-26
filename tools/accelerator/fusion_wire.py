"""Fusion wire format -- the binary transport encoding for FRONT G054's protocol
half. HUMF (tools/accelerator/humf.py) owns the memory-fabric half: domains,
object identity, coherence state, trust. This module owns none of that logic --
it only carries HUMF's vocabulary (an object, a byte range within it, a
representation, a version, a dependency ordering) as bytes on a wire, and
carries fusion_isa.py's commands the same way. humf.py is READ, never imported
or modified, by this module.

NO TRANSPORT IS REAL. There is no external GPU and no Spark on this machine.
Nothing in this module measures bandwidth or latency -- encode()/decode() are
pure byte transcoding with no I/O at all.

PACKET LAYOUT (42 bytes, fixed size, network byte order / big-endian):

    offset  size  field               notes
    0       1     protocol_version    u8; decoder refuses anything != 1
    1       1     command_id          u8; a fusion_isa.FusionOp value, opaque here
    2       2     flags               u16 bitmask; see FusionFlag
    4       2     representation_id   u16; a small dictionary id, not a string
    6       8     object_id           u64; numeric handle, not HUMF's string identity
    14      8     byte_offset         u64; offset into the object's byte range
    22      8     length              u64; length of that range, capped by MAX_LENGTH
    30      4     object_version      u32
    34      4     dependency_epoch    u32; the ordering/epoch counter a command depends on
    38      4     checksum            u32; crc32 over bytes [0:38)

Every field is packed and unpacked BY NAME, one at a time -- never
`struct.pack(fmt, *dataclasses.astuple(pkt))` and never pickle. A single
combined struct call over an object's `__dict__` would make the wire layout an
accident of Python attribute order; here the layout is the thing being
specified, and the field list above is the spec, not a side effect of the
dataclass.

NOT IMPLEMENTED, named rather than left silent:
  - No variable-length payload trailer. A packet describes an operation on a
    byte range of an object; it does not carry the object's bytes. Moving
    actual bytes is a provider's job (see humf.py's copy_in/copy_out) and is
    out of scope for a wire format that has no real transport under it.
  - No encryption, no authentication. checksum is crc32 -- an accident
    detector (see humf.py's own _digest for the measured reasoning), not
    protection against a hostile peer.
  - No compression, no fragmentation/reassembly across multiple packets.
  - Only protocol_version 1 exists. A packet from a newer version is REFUSED,
    not best-effort parsed -- there is nothing to best-effort parse it INTO,
    since no version 2 layout has been designed.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from enum import IntFlag

PROTOCOL_VERSION = 1
HEADER_SIZE = 42
_CHECKSUM_SPAN = HEADER_SIZE - 4  # bytes [0:38) are what the checksum covers

# A SANITY CEILING, not a measured transport or object-size limit. Nothing on
# this machine has moved a 1 TiB object; this exists purely to give the
# decoder something to refuse absurd length fields against, the way a real
# parser refuses a length field bigger than the address space it will ever see.
MAX_LENGTH = 1 << 40


class FusionFlag(IntFlag):
    NONE = 0
    REQUIRES_ACK = 1 << 0
    IS_RESPONSE = 1 << 1
    BATCH_END = 1 << 2


class FusionWireError(RuntimeError):
    """Base for every error this module raises. decode() and encode() raise
    ONLY subclasses of this (see test_fusion_wire.py's fuzz test) -- never a
    bare exception, never a silent wrong packet."""


class MalformedPacketError(FusionWireError):
    """Input was not even bytes, or some other shape error decode() cannot
    otherwise name."""


class TruncatedPacketError(FusionWireError):
    """Fewer bytes than the layout requires."""


class UnsupportedProtocolVersionError(FusionWireError):
    """protocol_version names a layout this decoder has never seen. Refused,
    not best-effort parsed -- see the module docstring."""


class ChecksumMismatchError(FusionWireError):
    """The trailing crc32 does not match the header bytes it claims to cover."""


class AbsurdLengthError(FusionWireError):
    """length exceeds MAX_LENGTH. Catches a fuzzed or corrupted length field
    before it is handed to any caller as if it were trustworthy."""


class FieldOutOfRangeError(FusionWireError):
    """encode() refuses to pack a field value that would not round-trip
    through its wire width."""


_FIELD_BOUNDS = {
    "protocol_version": (0, (1 << 8) - 1),
    "command_id": (0, (1 << 8) - 1),
    "flags": (0, (1 << 16) - 1),
    "representation_id": (0, (1 << 16) - 1),
    "object_id": (0, (1 << 64) - 1),
    "byte_offset": (0, (1 << 64) - 1),
    "length": (0, MAX_LENGTH),
    "object_version": (0, (1 << 32) - 1),
    "dependency_epoch": (0, (1 << 32) - 1),
}


def _check_bounds(name: str, value: int) -> None:
    lo, hi = _FIELD_BOUNDS[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise FieldOutOfRangeError(f"{name}={value!r} is not an int")
    if not (lo <= value <= hi):
        raise FieldOutOfRangeError(f"{name}={value} is outside its wire range [{lo}, {hi}]")


@dataclass
class FusionPacket:
    """One decoded/encodable packet. `checksum` is DERIVED -- encode() always
    recomputes it and decode() always verifies it before returning; it is
    excluded from equality so two packets built with the same logical fields
    compare equal regardless of how checksum was set."""
    command_id: int
    object_id: int
    byte_offset: int = 0
    length: int = 0
    object_version: int = 0
    representation_id: int = 0
    flags: int = 0
    dependency_epoch: int = 0
    protocol_version: int = PROTOCOL_VERSION
    checksum: int = field(default=0, compare=False)

    def encode(self) -> bytes:
        for name in ("protocol_version", "command_id", "flags", "representation_id",
                     "object_id", "byte_offset", "length", "object_version",
                     "dependency_epoch"):
            _check_bounds(name, getattr(self, name))
        out = bytearray()
        out += struct.pack(">B", self.protocol_version)
        out += struct.pack(">B", self.command_id)
        out += struct.pack(">H", self.flags)
        out += struct.pack(">H", self.representation_id)
        out += struct.pack(">Q", self.object_id)
        out += struct.pack(">Q", self.byte_offset)
        out += struct.pack(">Q", self.length)
        out += struct.pack(">I", self.object_version)
        out += struct.pack(">I", self.dependency_epoch)
        assert len(out) == _CHECKSUM_SPAN
        checksum = zlib.crc32(bytes(out)) & 0xFFFFFFFF
        out += struct.pack(">I", checksum)
        self.checksum = checksum
        return bytes(out)


def decode(data: bytes) -> FusionPacket:
    """The inverse of FusionPacket.encode(). Returns a valid packet or raises
    a FusionWireError subclass naming the reason -- never anything else,
    never a silently wrong packet. See test_fusion_wire.py::test_fuzzed_*."""
    if not isinstance(data, (bytes, bytearray)):
        raise MalformedPacketError(f"expected bytes, got {type(data).__name__}")
    if len(data) < 1:
        raise TruncatedPacketError("packet is 0 bytes; cannot even read protocol_version")
    protocol_version = data[0]
    if protocol_version != PROTOCOL_VERSION:
        raise UnsupportedProtocolVersionError(
            f"packet declares protocol_version={protocol_version}; this decoder "
            f"only understands protocol_version={PROTOCOL_VERSION} and refuses to "
            f"best-effort parse a layout it has never seen")
    if len(data) != HEADER_SIZE:
        raise TruncatedPacketError(
            f"packet is {len(data)} bytes; protocol_version={PROTOCOL_VERSION} "
            f"packets are exactly {HEADER_SIZE} bytes")
    (command_id,) = struct.unpack(">B", data[1:2])
    (flags,) = struct.unpack(">H", data[2:4])
    (representation_id,) = struct.unpack(">H", data[4:6])
    (object_id,) = struct.unpack(">Q", data[6:14])
    (byte_offset,) = struct.unpack(">Q", data[14:22])
    (length,) = struct.unpack(">Q", data[22:30])
    (object_version,) = struct.unpack(">I", data[30:34])
    (dependency_epoch,) = struct.unpack(">I", data[34:38])
    (checksum_stored,) = struct.unpack(">I", data[38:42])
    checksum_actual = zlib.crc32(data[:_CHECKSUM_SPAN]) & 0xFFFFFFFF
    if checksum_stored != checksum_actual:
        raise ChecksumMismatchError(
            f"checksum mismatch: packet claims {checksum_stored:#010x}, computed "
            f"{checksum_actual:#010x} over its own header bytes; refusing to trust "
            f"a packet whose own integrity check fails")
    if length > MAX_LENGTH:
        raise AbsurdLengthError(
            f"length field is {length} bytes, over the {MAX_LENGTH}-byte sanity "
            f"ceiling (a KNOB, not a measured transport or object-size limit)")
    pkt = FusionPacket(command_id=command_id, object_id=object_id,
                       byte_offset=byte_offset, length=length,
                       object_version=object_version,
                       representation_id=representation_id, flags=flags,
                       dependency_epoch=dependency_epoch,
                       protocol_version=protocol_version)
    pkt.checksum = checksum_stored
    return pkt
