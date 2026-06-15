//! strand_bake — PLANNED, NOT IMPLEMENTED (a deliberately empty seam).
//!
//! Intended future tool: convert GGUF/safetensors weights into a `.strand` v2
//! trellis-coded deploy artifact by absorbing the `strand-quant` QTIP backend
//! (sub-4-bit, integer float-free decode). That backend is developed in a
//! separate project and is NOT wired in here — so there is no encoder, no
//! writer, and no baking logic in this binary.
//!
//! dismantle has NO quantization or baking capability of its own today; it
//! consumes pre-quantized GGUF files. Do not read this as a working tool.

fn main() {
    eprintln!(
        "strand_bake: not implemented.\n\
         \n\
         dismantle does not bake quantized weights — it loads pre-quantized GGUF \
         files. Absorbing the strand-quant (QTIP) sub-4-bit backend is a planned, \
         deferred feature; see the Roadmap in README.md."
    );
    std::process::exit(2);
}
