# IANA media-type registration (draft) — `application/vnd.strand.archive`

Draft registration per RFC 6838 §5.6, intended for the IANA
"Application for a Media Type" form (https://www.iana.org/form/media-types),
vendor tree. Not yet submitted.

---

**Type name:** application

**Subtype name:** vnd.strand.archive

**Required parameters:** N/A

**Optional parameters:** N/A

**Encoding considerations:** binary

This media type consists of arbitrary binary data (a framed container of
entropy-coded blocks) and is not representable in 7-bit or 8-bit text
encodings. Transports that cannot carry binary data must apply a suitable
content transfer encoding (e.g. base64).

**Security considerations:**

The format contains arbitrary compressed data. The principal risks are
those common to all compression containers:

- *Decompression bombs.* Small archives can decode to vastly larger
  output. The container records per-block uncompressed sizes in its
  header, so implementations can bound or refuse expansion before
  allocating; consumers SHOULD enforce output-size limits.
- *Malformed input.* The reference decoder is written entirely in
  memory-safe Rust (no `unsafe` on the decode path) and validates magic
  bytes, format version, declared lengths, and per-block checksums before
  and during decode; structurally invalid or truncated input yields a
  typed error, not undefined behavior.
- *Integrity.* Block payloads carry checksums that are verified on
  decode; the encoder is deterministic (identical input produces
  byte-identical archives on every platform), which makes archives
  independently re-verifiable by hash.
- *Path handling.* Archive entries carry relative file names. Extracting
  implementations MUST reject absolute paths and path components that
  escape the destination directory (`..`), as with any archive format.
- The format carries no executable or active content, no scripts, and no
  external references; decoding causes no I/O beyond reading the input
  and writing the requested output. It provides no confidentiality:
  contents are compressed, not encrypted.

**Interoperability considerations:**

All multi-byte integers are little-endian. The format is frozen at
version 1; wire-incompatible revisions change both the format-version
field and the magic number. The encoder is fully deterministic — no
floating-point operations exist on any codec path — so independent
implementations can be validated by byte-comparison against the
reference vectors. A solid `.sa` archive wraps one or more STRAND
containers (inner magic `STRND1`) behind an archive table of contents.

**Published specification:**

"STRAND Container Format Specification", `SPEC.md` in the source
repository: https://github.com/joshuahickscorp/strand (normative
implementation: `crates/strand-container`). The specification is
sufficient to write an independent reference decoder.

**Applications that use this media type:**

The `strand` command-line tool (`strand pack` / `unpack` / `list`) and
STRAND.app, the macOS document handler for `.sa` archives.

**Fragment identifier considerations:** N/A

**Additional information:**

- *Deprecated alias names for this type:* N/A
- *Magic number(s):* first 6 bytes `53 54 52 41 31 00` ("STRA1" followed
  by NUL) for solid archives; embedded containers begin `53 54 52 4E 44 31`
  ("STRND1").
- *File extension(s):* `.sa`
- *Macintosh file type code(s):* N/A (UTI used instead)
- *macOS Uniform Type Identifier:* `com.strand.archive`, conforming to
  `public.data` and `public.archive` (exported by STRAND.app,
  `packaging/macos/Info.plist`)

**Person & email address to contact for further information:**

Joshua Hicks <joshuahicksboba@gmail.com>

**Intended usage:** COMMON

Interchange of losslessly compressed file archives produced by the STRAND
deterministic compression toolchain.

**Restrictions on usage:** none

**Author:** Joshua Hicks <joshuahicksboba@gmail.com>

**Change controller:** Joshua Hicks <joshuahicksboba@gmail.com>

**Provisional registration? (standards tree only):** N/A (vendor tree)
