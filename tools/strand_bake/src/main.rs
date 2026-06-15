//! strand_bake — offline STRAND (QTIP) weight baker. SEAM — pipeline not yet wired.
//!
//! Converts GGUF/safetensors weights into a `.strand` v2 trellis-coded deploy
//! artifact using the `strand-quant` QTIP backend (sub-4-bit, integer float-free
//! decode). That backend is now ABSORBED into this repo at `vendor/strand-quant`
//! (from ~/Downloads/strand, tag `quant-handoff`) — the encoder, the STR2 v2
//! writer, and the load-bearing i64 reconstruct all live there.
//!
//! What remains here is the baker's own glue: read f32 weights out of a GGUF,
//! select the projection tensors, call `strand_quant::encode::encode_tensor`, and
//! emit the file with `strand_quant::format::write_strand_v2`. See
//! `docs/strand/STRAND-dismantle-wiring.md` Step 4 for the exact pipeline.
//!
//! Until that glue lands, dismantle still consumes pre-quantized GGUF files; this
//! binary is a documented seam, not a working tool.

fn main() {
    eprintln!(
        "strand_bake: baker pipeline not yet wired.\n\
         \n\
         The strand-quant (QTIP) backend is absorbed at vendor/strand-quant, but the\n\
         GGUF f32 read -> encode_tensor -> write_strand_v2 glue in this binary is not\n\
         implemented yet. See docs/strand/STRAND-dismantle-wiring.md Step 4.\n\
         dismantle currently loads pre-quantized GGUF files."
    );
    std::process::exit(2);
}
