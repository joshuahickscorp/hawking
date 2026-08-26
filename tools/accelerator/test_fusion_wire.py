"""fusion_wire pins. FRONT G054 (transport-protocol half; humf.py owns the
memory-fabric half and is not touched here).

GOLDEN VECTORS ARE THE POINT: they were derived by hand from the documented
field layout (see fusion_wire.py's module docstring table), not by calling
FusionPacket.encode() and pasting its output -- pasting the encoder's own
output back as its "expected" value would prove nothing about the layout,
only that the function returns what it returns. Each vector's field values
and their byte positions are laid out in comments here so a change to the
layout (a reordered field, a resized field, a different checksum span) shows
up as a byte-for-byte diff against a value that was fixed independently of
the code under test.
"""
import random
import struct
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fusion_wire  # noqa: E402
from fusion_wire import (  # noqa: E402
    HEADER_SIZE,
    MAX_LENGTH,
    PROTOCOL_VERSION,
    AbsurdLengthError,
    ChecksumMismatchError,
    FieldOutOfRangeError,
    FusionPacket,
    FusionWireError,
    MalformedPacketError,
    TruncatedPacketError,
    UnsupportedProtocolVersionError,
    decode,
)

# --------------------------------------------------------------------- golden vectors
#
# Layout recap (offset, size, field), all big-endian:
#   0 1 protocol_version | 1 1 command_id | 2 2 flags | 4 2 representation_id
#   6 8 object_id | 14 8 byte_offset | 22 8 length | 30 4 object_version
#   34 4 dependency_epoch | 38 4 checksum(crc32 of bytes[0:38))
#
# GOLDEN_1: every field zero except protocol_version=1, command_id=0 (the
# ACQUIRE_READ opcode by fusion_isa convention), object_id=1. Byte-by-byte:
#   01                                    protocol_version=1
#   00                                    command_id=0
#   0000                                  flags=0
#   0000                                  representation_id=0
#   0000000000000001                      object_id=1
#   0000000000000000                      byte_offset=0
#   0000000000000000                      length=0
#   00000000                              object_version=0
#   00000000                              dependency_epoch=0
#   b0faf65f                              crc32 of the 38 bytes above
GOLDEN_1_HEX = (
    "0100000000000000000000000001000000000000000000000000000000000000000000"
    "000000b0faf65f"
)
GOLDEN_1_FIELDS = dict(protocol_version=1, command_id=0, flags=0, representation_id=0,
                       object_id=1, byte_offset=0, length=0, object_version=0,
                       dependency_epoch=0)

# GOLDEN_2: mid-range values, command_id=4 (COPY by convention), exercises every
# field being distinct so a field-swap bug cannot hide behind a repeated value.
#   01                                    protocol_version=1
#   04                                    command_id=4
#   0005                                  flags=5
#   0003                                  representation_id=3
#   00000000deadbeef                      object_id=0xDEADBEEF
#   0000000000001000                      byte_offset=4096
#   0000000000010000                      length=65536
#   00000007                              object_version=7
#   0000002a                              dependency_epoch=42
#   f31d553c                              crc32 of the 38 bytes above
GOLDEN_2_HEX = (
    "01040005000300000000deadbeef00000000000010000000000000010000000000070000"
    "002af31d553c"
)
GOLDEN_2_FIELDS = dict(protocol_version=1, command_id=4, flags=5, representation_id=3,
                       object_id=0xDEADBEEF, byte_offset=4096, length=65536,
                       object_version=7, dependency_epoch=42)

# GOLDEN_3: every field at (or near) its maximum representable value,
# command_id=2 (RELEASE by convention). Catches a field boundary that is one
# byte too narrow or too wide.
#   01                                    protocol_version=1
#   02                                    command_id=2
#   ffff                                  flags=0xFFFF
#   ffff                                  representation_id=0xFFFF
#   ffffffffffffffff                      object_id=2**64-1
#   0000000000000000                      byte_offset=0
#   0000000040000000                      length=2**30 (1 GiB, well under MAX_LENGTH)
#   ffffffff                              object_version=2**32-1
#   ffffffff                              dependency_epoch=2**32-1
#   b7d696c5                              crc32 of the 38 bytes above
GOLDEN_3_HEX = (
    "0102ffffffffffffffffffffffff0000000000000000"
    "0000000040000000ffffffffffffffffb7d696c5"
)
GOLDEN_3_FIELDS = dict(protocol_version=1, command_id=2, flags=0xFFFF,
                       representation_id=0xFFFF, object_id=(1 << 64) - 1,
                       byte_offset=0, length=1 << 30, object_version=(1 << 32) - 1,
                       dependency_epoch=(1 << 32) - 1)

GOLDEN_VECTORS = [
    ("GOLDEN_1", GOLDEN_1_HEX, GOLDEN_1_FIELDS),
    ("GOLDEN_2", GOLDEN_2_HEX, GOLDEN_2_FIELDS),
    ("GOLDEN_3", GOLDEN_3_HEX, GOLDEN_3_FIELDS),
]


def test_golden_vector_lengths():
    """Every golden hex string must be exactly HEADER_SIZE bytes -- a typo in
    the hand-written hex would otherwise silently shrink the test's power."""
    for name, hexstr, _ in GOLDEN_VECTORS:
        assert len(hexstr) == HEADER_SIZE * 2, f"{name} hex is not {HEADER_SIZE} bytes"


@pytest.mark.parametrize("name,hexstr,fields", GOLDEN_VECTORS)
def test_golden_vector_encode_matches_exactly(name, hexstr, fields):
    pkt = FusionPacket(**fields)
    assert pkt.encode().hex() == hexstr, f"{name}: layout changed under encode()"


@pytest.mark.parametrize("name,hexstr,fields", GOLDEN_VECTORS)
def test_golden_vector_decode_matches_fields(name, hexstr, fields):
    pkt = decode(bytes.fromhex(hexstr))
    for k, v in fields.items():
        assert getattr(pkt, k) == v, f"{name}: decoded {k} mismatch"


# --------------------------------------------------------------------------- round trip

def _random_packet(rng: random.Random) -> FusionPacket:
    return FusionPacket(
        command_id=rng.randint(0, 255),
        object_id=rng.randint(0, (1 << 64) - 1),
        byte_offset=rng.randint(0, (1 << 64) - 1),
        length=rng.randint(0, MAX_LENGTH),
        object_version=rng.randint(0, (1 << 32) - 1),
        representation_id=rng.randint(0, (1 << 16) - 1),
        flags=rng.randint(0, (1 << 16) - 1),
        dependency_epoch=rng.randint(0, (1 << 32) - 1),
    )


def test_round_trip_many_field_combinations():
    rng = random.Random(20260825)
    for _ in range(2000):
        pkt = _random_packet(rng)
        wire = pkt.encode()
        assert len(wire) == HEADER_SIZE
        back = decode(wire)
        assert back.command_id == pkt.command_id
        assert back.object_id == pkt.object_id
        assert back.byte_offset == pkt.byte_offset
        assert back.length == pkt.length
        assert back.object_version == pkt.object_version
        assert back.representation_id == pkt.representation_id
        assert back.flags == pkt.flags
        assert back.dependency_epoch == pkt.dependency_epoch
        assert back.protocol_version == pkt.protocol_version
        assert back == pkt          # dataclass equality (checksum excluded)
        assert back.checksum == pkt.checksum


def test_round_trip_boundary_values():
    """Zero and max-width values at every field, not just interior randoms --
    off-by-one field widths hide at the edges, not the middle."""
    edge = [0, 1, (1 << 8) - 1, (1 << 16) - 1, (1 << 32) - 1, (1 << 64) - 1]
    rng = random.Random(7)
    for _ in range(300):
        pkt = FusionPacket(
            command_id=rng.choice(edge) & 0xFF,
            object_id=rng.choice(edge),
            byte_offset=rng.choice(edge),
            length=min(rng.choice(edge), MAX_LENGTH),
            object_version=rng.choice(edge) & 0xFFFFFFFF,
            representation_id=rng.choice(edge) & 0xFFFF,
            flags=rng.choice(edge) & 0xFFFF,
            dependency_epoch=rng.choice(edge) & 0xFFFFFFFF,
        )
        assert decode(pkt.encode()) == pkt


# ---------------------------------------------------- encode() refuses out-of-range

def test_encode_refuses_length_over_MAX_LENGTH():
    pkt = FusionPacket(command_id=0, object_id=0, length=MAX_LENGTH + 1)
    with pytest.raises(FieldOutOfRangeError, match="length"):
        pkt.encode()


def test_encode_refuses_negative_field():
    pkt = FusionPacket(command_id=0, object_id=-1)
    with pytest.raises(FieldOutOfRangeError):
        pkt.encode()


def test_encode_refuses_field_too_wide_for_its_wire_width():
    pkt = FusionPacket(command_id=256, object_id=0)   # command_id is u8
    with pytest.raises(FieldOutOfRangeError, match="command_id"):
        pkt.encode()


def test_encode_accepts_the_boundary_MAX_LENGTH_itself():
    """The control for test_encode_refuses_length_over_MAX_LENGTH: the ceiling
    itself must still be a legal length, only ceiling+1 is refused."""
    pkt = FusionPacket(command_id=0, object_id=0, length=MAX_LENGTH)
    decode(pkt.encode())   # must not raise


# ------------------------------------------------------------- named decode failures

def test_decode_refuses_empty_input():
    with pytest.raises(TruncatedPacketError):
        decode(b"")


def test_decode_refuses_short_input():
    with pytest.raises(TruncatedPacketError):
        decode(bytes([PROTOCOL_VERSION, 0, 0, 0]))


def test_decode_refuses_long_input():
    good = FusionPacket(command_id=0, object_id=0).encode()
    with pytest.raises(TruncatedPacketError):
        decode(good + b"\x00")


def test_decode_refuses_unknown_protocol_version():
    good = bytearray(FusionPacket(command_id=0, object_id=0).encode())
    good[0] = PROTOCOL_VERSION + 1
    with pytest.raises(UnsupportedProtocolVersionError, match=str(PROTOCOL_VERSION + 1)):
        decode(bytes(good))


def test_decode_refuses_a_future_protocol_version_by_name_not_best_effort():
    """A packet claiming a version this decoder has never seen must be
    REFUSED with the version named, not parsed as if it were the current
    layout."""
    good = bytearray(FusionPacket(command_id=0, object_id=0).encode())
    good[0] = 200
    with pytest.raises(UnsupportedProtocolVersionError) as exc:
        decode(bytes(good))
    assert "200" in str(exc.value)


def test_decode_refuses_bad_checksum():
    good = bytearray(FusionPacket(command_id=1, object_id=42, length=100).encode())
    good[-1] ^= 0xFF   # flip a bit in the checksum itself
    with pytest.raises(ChecksumMismatchError):
        decode(bytes(good))


def test_decode_refuses_payload_corruption_via_checksum():
    """A bit flipped in the BODY, not the checksum, must be caught the same
    way -- the checksum protects the whole header, not just itself."""
    good = bytearray(FusionPacket(command_id=1, object_id=42, length=100).encode())
    good[10] ^= 0x01
    with pytest.raises(ChecksumMismatchError):
        decode(bytes(good))


def test_decode_refuses_absurd_length_even_with_a_correct_checksum():
    """Constructed by hand below FusionPacket.encode()'s own guard, because a
    hostile or corrupted packet will not have gone through that guard --
    decode() must defend itself independently of encode()'s validation."""
    fields = bytearray()
    fields += struct.pack(">B", PROTOCOL_VERSION)
    fields += struct.pack(">B", 0)
    fields += struct.pack(">H", 0)
    fields += struct.pack(">H", 0)
    fields += struct.pack(">Q", 0)
    fields += struct.pack(">Q", 0)
    fields += struct.pack(">Q", (1 << 64) - 1)   # absurd length, correctly checksummed
    fields += struct.pack(">I", 0)
    fields += struct.pack(">I", 0)
    checksum = zlib.crc32(bytes(fields)) & 0xFFFFFFFF
    raw = bytes(fields) + struct.pack(">I", checksum)
    assert len(raw) == HEADER_SIZE
    with pytest.raises(AbsurdLengthError):
        decode(raw)


def test_decode_refuses_non_bytes_input():
    with pytest.raises(MalformedPacketError):
        decode("not bytes")               # type: ignore[arg-type]


def test_decode_refuses_none_input():
    with pytest.raises(MalformedPacketError):
        decode(None)                      # type: ignore[arg-type]


# ------------------------------------------------------------------------- fuzzing
#
# What the fuzzer is FOR: proving decode() never crashes uncaught and never
# returns a packet built from bytes that failed their own integrity check.
# Reported honestly below (see test_fuzz_random_bytes_result_breakdown) --
# almost every purely random buffer is caught by the protocol_version check
# alone, because only 1/256 random first bytes equal PROTOCOL_VERSION. That
# is a true fact about this fuzzer's power, not a gap: the mutated-valid-
# packet fuzz below is what reaches the checksum and length checks instead.

def test_fuzzed_random_bytes_never_crash_uncaught():
    rng = random.Random(1)
    for _ in range(5000):
        n = rng.randint(0, 200)
        data = bytes(rng.randrange(256) for _ in range(n))
        try:
            pkt = decode(data)
        except FusionWireError:
            continue
        except Exception as e:   # pragma: no cover -- this is the failure this test exists to catch
            pytest.fail(f"decode() raised a non-FusionWireError on random input "
                       f"{data.hex()!r}: {type(e).__name__}: {e}")
        else:
            # decode() claims success: its own checksum must actually agree,
            # i.e. it must not be a silently-wrong packet.
            assert pkt.encode()[:HEADER_SIZE - 4] == bytes(data)[:HEADER_SIZE - 4]


def test_fuzz_random_bytes_result_breakdown():
    """Names what the random-bytes fuzzer actually found, honestly. If this
    ever reports zero decode successes and zero of each error kind, that is a
    result about the fuzzer (or a regression in decode()), not evidence the
    decoder is correct -- read the printed breakdown, don't just trust green."""
    rng = random.Random(2)
    outcomes: dict[str, int] = {}
    for _ in range(5000):
        n = rng.randint(0, 200)
        data = bytes(rng.randrange(256) for _ in range(n))
        try:
            decode(data)
            outcomes["OK"] = outcomes.get("OK", 0) + 1
        except FusionWireError as e:
            key = type(e).__name__
            outcomes[key] = outcomes.get(key, 0) + 1
    print("random-bytes fuzz outcome breakdown:", outcomes)
    # TruncatedPacketError and UnsupportedProtocolVersionError dominate by
    # construction (see comment above); this assertion pins that the fuzzer
    # is actually exercising more than one failure path, not degenerating to
    # a single always-taken branch.
    assert len([k for k, v in outcomes.items() if v > 0]) >= 2


def test_fuzzed_mutated_valid_packets_never_crash_uncaught():
    """Starts from packets that DID pass encode(), then mutates single bytes
    -- this is what actually reaches ChecksumMismatchError and
    AbsurdLengthError, since those checks only run once protocol_version and
    length-of-input already look plausible."""
    rng = random.Random(3)
    base_packets = [
        FusionPacket(command_id=c, object_id=o, byte_offset=b, length=l,
                    object_version=v, representation_id=r, flags=f,
                    dependency_epoch=d).encode()
        for c, o, b, l, v, r, f, d in [
            (0, 1, 0, 0, 0, 0, 0, 0),
            (4, 0xDEADBEEF, 4096, 65536, 7, 3, 5, 42),
            (13, 999, 10, 20, 1, 1, 1, 1),
        ]
    ]
    seen_kinds: set[str] = set()
    for _ in range(5000):
        base = bytearray(rng.choice(base_packets))
        mutation = rng.choice(["flip_byte", "truncate", "extend", "zero_all"])
        if mutation == "flip_byte":
            i = rng.randrange(len(base))
            base[i] ^= 1 << rng.randrange(8)
        elif mutation == "truncate":
            base = base[: rng.randrange(0, len(base))]
        elif mutation == "extend":
            base += bytes(rng.randrange(256) for _ in range(rng.randint(1, 20)))
        else:
            base = bytearray(len(base))
        try:
            pkt = decode(bytes(base))
            seen_kinds.add("OK")
            assert pkt.encode()[:HEADER_SIZE - 4] == bytes(base)[:HEADER_SIZE - 4]
        except FusionWireError as e:
            seen_kinds.add(type(e).__name__)
        except Exception as e:   # pragma: no cover
            pytest.fail(f"decode() raised a non-FusionWireError on mutated input "
                       f"{bytes(base).hex()!r}: {type(e).__name__}: {e}")
    print("mutated-packet fuzz outcome kinds:", sorted(seen_kinds))
    # The mutated-input fuzz is the one that should actually reach a bad
    # checksum, since it starts from otherwise-valid bytes.
    assert "ChecksumMismatchError" in seen_kinds


def test_fuzz_never_returns_a_packet_that_fails_its_own_checksum():
    """The strongest fuzz property: whenever decode() returns successfully,
    re-encoding must reproduce the exact input bytes (protocol_version==1
    packets are canonical -- there is only one way to encode a given field
    set), proving decode() never silently accepted bytes it should have
    rejected."""
    rng = random.Random(4)
    hits = 0
    for _ in range(20000):
        n = rng.choice([HEADER_SIZE] * 10 + [rng.randint(0, HEADER_SIZE * 2)])
        data = bytes(rng.randrange(256) for _ in range(n))
        try:
            pkt = decode(data)
        except FusionWireError:
            continue
        hits += 1
        assert pkt.encode() == data
    print(f"random HEADER_SIZE-biased fuzz: {hits} packets decoded successfully "
         f"and round-tripped byte-exact")
