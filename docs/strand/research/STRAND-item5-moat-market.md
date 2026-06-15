# STRAND determinism moat — as a market, not a feature

_Created 2026-06-13 (Item 5). Scope: position the determinism property — bit-identical,
float-free integer decode plus the SPRV/SPV3 attestation layer — as a set of concrete
buyable markets, with the value prop, the structural reason it requires bit-exactness, the
exact thing competitors cannot claim, and the honest places STRAND is not yet the choice.
Every claim here is traceable to the shipped property; no claim rests on quality numbers
the quant track has not yet earned._

Source of truth for the property: `docs/STRAND-quality-density-frontier.md` (the moat
framing, §9/§17), `docs/STRAND-v3-provenance-spec.md` (the attestation layer, SHIPPED
hashing), `docs/STRAND-proofs.md` (what is PROVEN), `crates/strand-quant/src/provenance.rs`
(reference impl, KAT-pinned), `crates/strand-decode-kernel/src/bin/attest-strand.rs` (the
verify tool).

---

## 0. The one-sentence frame

STRAND is not selling "a smaller model." Speed and density are threshold goods that
several formats clear. **The scarce, un-substitutable good STRAND sells is a model whose
weights are a verifiable constant — one 32-byte hash any stranger on any hardware can
recompute and match.** That is a property no float-codebook format can offer, and it maps
to real budgets in regulated, scientific, legal, and supply-chain settings.

---

## 1. The property the whole market rests on (kept honest)

Two facts, both grounded in shipped code, not aspiration:

**(P1) The decode is integer-only and byte-identical everywhere.** STRAND weights are
recovered by an integer Q12 lookup-table trellis decode — no float dequantization in the
decode path. This is proven, not asserted: `docs/STRAND-proofs.md` records exhaustive
enumeration (486,720 tensors over the (L,k) corners), a 52M-case golden byte-stability
vector reproducing `LUT_GOLDEN_HASH`, and Kani bounded model checking on the arithmetic
primitives (`reconstruct_q`, `eff_scale_q`, `eff_min_q`, the bit reader) — all
VERIFICATION SUCCESSFUL. GPU==CPU decode is pinned on the M3. The decoded Q12 integers are
the same bits on an Apple M3, an x86 server, and an ARM phone.

**(P2) Therefore "the SHA-256 of the decoded model" is a well-defined constant**, and the
SPV3/SPRV provenance layer makes it canonical. The hashing layer is SHIPPED
(`provenance.rs`, 8/8 tests, KATs cross-pinned against an independent hashlib impl). It
defines a 3-level Merkle commitment over the **decoded Q12 integers** (never the wire
bytes): per-block leaves → per-tensor roots → one 32-byte `model_root`. The `attest-strand`
tool recomputes roots and runs `verify_archive`/`--recon-check`. An SPRV section can be
appended to a v2 `.strand` file (append-only, header-frozen) so the artifact self-verifies
at load.

The honest boundary, stated up front so nothing downstream overclaims: the root binds the
**weights** (the values the model computes with), not a file-at-rest checksum and not any
particular inference *response*. Proving a given output token came from these weights needs
attestation/zkML and is explicitly **out of scope** (`provenance-spec.md` §5.2). What v3
proves is possession-and-decode of the committed weights — the challenge protocol forces a
provider to actually hold the real model to answer. That is the line every value prop below
respects.

---

## 2. Why competitors structurally cannot match this (the load-bearing argument)

This is not "STRAND does it better." It is "the others **cannot** do it at all," and the
reason is structural, not effort.

GGUF (k-quants), AQLM, and QuIP# all reconstruct weights through a **float dequantization**:
scales and codebook entries are floats, and the dequantized weight is a float computed at
load/inference time. Floating-point results are **not portable**: the same dequant produces
different bits across hardware, compiler, SIMD width, FMA-contraction, and thread/reduction
order. So the value the model actually computes with **differs per machine**.

The consequence is exact and unavoidable:

| capability | STRAND (integer decode) | GGUF / AQLM / QuIP# (float dequant) |
|---|---|---|
| hash the **file bytes** | yes | yes (everyone can) |
| hash the **weights** (decoded values) | **yes — a platform constant** | **no — there is no canonical value to hash; it drifts per machine** |
| recompute one root on untrusted hardware and match the publisher | **yes** | **no — your recompute differs from theirs by float drift** |
| sign/attest "these are the weights" | **yes** | **no — nothing canonical to sign** |
| two parties prove they hold the *same* model | **yes (compare roots)** | **no — same file can decode to different weight bits** |

The crisp version for a buyer: **everyone can hash the box the model came in; only STRAND
can hash the model.** A float-codebook format's "weights" are a per-machine artifact, so
they are not hashable, not signable, and not attestable as a constant. This is the entire
moat in one line, and it follows directly from P1/P2 — no benchmark required.

(Caveat kept honest: the *decode* is bit-identical everywhere; the *encode-side* RHT has
one documented ~1e-6 crack for odd-power block widths, e.g. 896-wide tensors, where f32
1/√h is non-dyadic — `quality-density-frontier.md` §17. This does not touch the moat: the
artifact ships its decoded-weight roots, and the verifier recomputes over the **decode**,
which is exact. Encode parity is a publisher-side concern, sealed by the published root.)

---

## 3. The segments that NEED a bit-reproducible / attestable model

Five segments where bit-exact weights are not a nice-to-have but a requirement that maps to
a budget. For each: the value prop, and the structural reason it requires bit-exactness.

### 3.1 Regulated AI — finance, healthcare, government

**Value prop:** "Deploy a quantized model and prove, to an auditor or regulator, that the
exact weights running in production are byte-identical to the validated, approved artifact —
recomputable by the regulator on their own hardware, with no trust in your mirror or your
build."

**Why it requires bit-exactness:** model-risk regimes (e.g. SR 11-7 in banking, GxP/FDA
SaMD validation in healthcare, FedRAMP/agency change-control in gov) turn on a validated
artifact that does not silently change. With a float-codebook model, the validated artifact
and the deployed artifact can be the same file yet compute with **different weight bits** on
the validation box vs the prod box — so "we validated this model" is not a checkable claim.
STRAND collapses validation-artifact and deploy-artifact to one 32-byte root the regulator
can recompute. The auditor does not have to trust the vendor's environment; they recompute
`model_root` and compare. This is the difference between an attestable control and a
paperwork assertion.

**Why competitors cannot:** their weights are not a constant, so there is no root to hand
the regulator; the best they can offer is a file checksum, which does not bind the values
the model computes with.

### 3.2 Scientific reproducibility

**Value prop:** "Publish a quantized model with one root in the paper. Any reviewer or
replicator, on any hardware they happen to have, decodes the weights, recomputes the root,
and gets a bit-identical match — reproducibility of the *model*, not just of a download
link."

**Why it requires bit-exactness:** reproducibility means a third party regenerates the same
object. A float-codebook artifact is reproducible only as a file; the weights a replicator's
machine computes differ from the authors' by float drift, so "same model" is unverifiable
and downstream numeric divergence is unattributable (was it the method, or the hardware?).
STRAND makes the published model a citable constant: the root in the paper is the model, and
a mismatch is a hard signal that something diverged. This is the same value as a dataset
hash, extended to the weights themselves.

**Why competitors cannot:** "the SHA-256 of the model" is undefined for them — there is no
canonical decoded value. They can pin a file hash, but a file hash does not certify the
numbers the network ran.

### 3.3 Legal / audit & model provenance

**Value prop:** "Establish a tamper-evident, court-defensible chain of custody for model
weights: a signed root at training/release, recomputable at any later date to prove the
weights have not been altered — and a cheap challenge protocol to prove a provider actually
holds the committed model."

**Why it requires bit-exactness:** provenance and chain-of-custody require an immutable
fingerprint of the *content*. STRAND's root is a fingerprint of the decoded weights, so any
alteration — a swapped tensor, a poisoned block, a quietly fine-tuned layer — changes the
root and is detectable by anyone. The block-level Merkle leaves let an auditor spot-check a
specific tensor without a full decode (`m/B` cost, milliseconds). The §5.2 challenge
protocol lets a verifier holding only the root force a provider to prove possession of the
real model (random block challenges; answering correctly requires holding it). Honest scope:
this proves possession and integrity of weights, not that a specific output was produced by
them — response-binding is zkML/attestation territory and out of scope. Stated that way, it
is exactly the provenance primitive a legal/audit workflow needs and can defend.

**Why competitors cannot:** a fingerprint that drifts per machine is not tamper-evidence —
a verifier cannot tell "tampered" from "different SIMD width." Float-codebook formats have
no stable content fingerprint to anchor a chain of custody.

### 3.4 Supply-chain integrity (model distribution)

**Value prop:** "Ship the model through any untrusted channel — public mirror, torrent,
CDN, a partner's USB stick — and let the recipient verify byte-identical weights with zero
trust in the channel. The artifact self-verifies at load."

**Why it requires bit-exactness:** supply-chain integrity for software rests on
reproducible-build hashes; models have lacked the equivalent because float dequant makes the
deployed weights non-reproducible. STRAND restores it: the recipient recomputes `model_root`
over their decode and matches the publisher — no trusted mirror, no per-platform
"expected-outputs" table (`provenance-spec.md` §5.1). The SPRV self-test vectors let a
loader catch bit-rot, truncation, a corrupted mirror, or a tampered payload **at mmap time,
before serving**, with a cost knob (`m`) the deployment chooses — instead of discovering it
as silent garbage during inference. This is SLSA/sigstore-grade integrity, finally
applicable to the weights and not just the wrapper.

**Why competitors cannot:** with non-portable weights there is nothing canonical for the
recipient to recompute, so "verify the model you received" reduces to "trust the file hash
the mirror gave you" — which a compromised mirror controls.

### 3.5 On-device / edge verification

**Value prop:** "A phone, car, medical device, or sensor verifies that the model it is about
to run is the exact approved artifact — at load, cheaply, with no network round-trip to a
trust server."

**Why it requires bit-exactness:** an edge device cannot phone home to validate a model
against a per-platform expected-output table, and it must not run a silently-corrupted or
substituted model. Because STRAND's decode is the *same integers on the device's own
hardware*, the device recomputes the embedded roots (or just the `m` self-test vectors —
milliseconds) locally and refuses to serve on mismatch. Verification is offline, cheap, and
self-contained in the artifact. The integer-only decode is also a fit for fixed-function /
no-FPU targets independent of the attestation story.

**Why competitors cannot:** local verification needs a local recompute that matches the
publisher; a float-codebook model's local decode does not match the publisher's, so the
device has nothing to check against without a trusted external oracle.

---

## 4. What GGUF and AQLM/QuIP# structurally CANNOT claim (the negative list)

Stated as flat capability denials, each a direct corollary of "their dequant is non-portable
→ their weights differ per machine → not hashable/attestable":

1. **A canonical model hash.** "The SHA-256 of the weights" is undefined for them. They can
   hash the file; they cannot hash the values the model computes with.
2. **Cross-hardware reproducibility of the weights.** Two correct implementations on
   different hardware produce different weight bits. They are reproducible as a download, not
   as a model.
3. **Untrusted-channel verification.** A recipient cannot recompute-and-match, because their
   recompute differs from the publisher's by float drift. Integrity collapses to trusting
   the channel's file hash.
4. **Tamper-evidence of content.** No stable content fingerprint, so "altered" is
   indistinguishable from "different SIMD/FMA/threads." A chain of custody cannot be anchored.
5. **A two-party "same model" proof.** Two parties cannot prove they hold identical weights
   by exchanging a hash — the same file can decode to different bits for each of them.
6. **Multi-node serving agreement by construction.** Distributed serving of one large model
   needs every node to compute with the same weights; float dequant gives each node subtly
   different weights, so agreement is best-effort, not guaranteed. (This matters *more* at
   405B-scale distributed serving, where the moat is strongest — `quality-density-frontier.md`
   §10.)

None of these is a benchmark gap they can close with engineering; each follows from the
floating-point reconstruction being in the contract. STRAND removed the float from the
decode contract, which is why it can make all six claims.

---

## 5. Where STRAND is NOT yet the choice (honest, defensible)

The moat is real and shipped; these are the places to **not** lead with it, stated plainly
so the positioning stays credible.

1. **Raw 2-bit quality, before the PV run.** Today's untrained 2-bit STRAND carries a real
   loss tax — honestly ~0.22–0.25 nats at true 2-bit (de-bias adopted, before selective-PV),
   per the RED-TEAMED targets in `quality-density-frontier.md` §12/§12.1. The ≤0.15 figure
   is *conditional* on selective-PV landing at scale (the cloud run is the decider), not
   banked. So a buyer optimizing purely for "best quality at 2-bit, determinism irrelevant"
   is not yet choosing STRAND on quality alone. The honest pitch is "verifiable AND
   competitive," never "verifiable AND best." 3-bit is stronger (loss tax 0.056 proven on
   llama2-7b mp_light), so the 3-bit-class lane is where determinism + quality currently
   co-sell cleanly.

2. **Ecosystem reach.** llama.cpp/GGUF has the runtimes, the tooling, the quant zoo, the
   community, and one-command deployment on every platform. STRAND's decode kernel and
   loader exist and are fast, but the surrounding ecosystem (server integrations, framework
   bindings, hub presence) is early. For a buyer whose first requirement is "drop into my
   existing llama.cpp stack today," STRAND is not the path of least resistance yet.

3. **Attestation depth — STRAND proves weights, not responses.** The provenance layer
   proves possession and integrity of the committed weights (§5.2). It does **not** prove a
   specific inference output was produced by them — that needs hardware attestation or zkML,
   which STRAND does not provide. A buyer whose requirement is "cryptographically bind every
   API response to the model" needs more than v3 offers. State the boundary; do not let the
   strength of the weight-attestation claim imply response-attestation.

4. **The encode-side RHT crack for odd-power block widths.** Cross-device bit-identity of
   the *encoder's* RHT is exact only for even-power-of-two block sizes; odd-power widths
   (e.g. 896-wide Qwen-0.5B) are approximate at ~1e-6, resting on IEEE-754 f32 no-FMA rather
   than a proof (§17). This does not break the moat (the published root is over the exact
   decode, which the verifier recomputes), but it is a real, documented limit for anyone who
   needs *encoder* determinism across hardware, and it should be disclosed rather than
   papered over.

5. **GPU-vs-CPU *encoder* parity is not yet machine-proven.** Decode parity is proven; the
   encode path's cross-hardware parity is hardware-gated and separate (§17). For workflows
   that re-encode on heterogeneous hardware and demand bit-identical *artifacts* (not just
   bit-identical decode of one published artifact), this gap is open.

---

## 6. The positioning in one paragraph (for reuse)

STRAND is the only quantized-model format whose **weights are a verifiable constant**:
because the decode is integer-only and byte-identical on every machine (proven exhaustively
+ Kani), the SHA-256 of the decoded model is well-defined, and the shipped SPRV layer turns
it into one 32-byte root any party can recompute on untrusted hardware and match. That single
property — not speed, not density, which others also clear — is what regulated AI, scientific
reproducibility, legal/audit provenance, supply-chain integrity, and on-device verification
actually need, and it is something GGUF, AQLM, and QuIP# **structurally cannot offer**,
because their float dequantization makes the weights a per-machine artifact with no canonical
value to hash, sign, or attest. STRAND is "verifiable AND competitive" today and the place to
sell on the moat is wherever a stranger must be able to check the model — while being honest
that raw 2-bit quality (pre-PV), ecosystem reach, and response-level attestation are not yet
where STRAND leads.
