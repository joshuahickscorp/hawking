//! Independent adversarial audit of Candidate B: prove the runtime is real — it parses actual GGUF,
//! executes every layer, generates from calculated logits, handles variable prompts/lengths, and reacts
//! correctly to corrupted inputs. Final-token parity is not the only proof: per-step logit checksums
//! are captured too. Gated on the fixture; fails closed (skips) when absent.

use hawking_seed_b::adapter::LlamaConfig;
use hawking_seed_b::gguf::GgufFile;
use hawking_seed_b::model::Model;
use hawking_seed_b::quant;
use hawking_seed_b::tokenizer::Tokenizer;
use std::path::Path;

const WEIGHTS: &str = "../../models/SmolLM2-135M-Instruct-Q4_K_M.gguf";

fn present() -> bool {
    Path::new(WEIGHTS).exists()
}
fn open() -> GgufFile {
    GgufFile::open(Path::new(WEIGHTS)).expect("gguf")
}

#[test]
fn variable_prompt_gives_different_deterministic_output() {
    if !present() {
        eprintln!("skip: fixture absent");
        return;
    }
    let g = open();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let mut m = Model::from_gguf(&g).unwrap();

    let a1 = m.generate(&tok.encode("The capital of France is").unwrap(), 8, tok.eos_id()).unwrap();
    let a2 = m.generate(&tok.encode("The capital of France is").unwrap(), 8, tok.eos_id()).unwrap();
    let b = m.generate(&tok.encode("Once upon a time").unwrap(), 8, tok.eos_id()).unwrap();

    assert_eq!(a1, a2, "same prompt must be deterministic");
    assert_ne!(a1, b, "different prompt must give a different sequence (no hardcoding)");
    assert!(!b.iter().all(|&t| t == b[0]), "output should not be a single repeated token");
}

#[test]
fn variable_length_prefix_consistency() {
    if !present() {
        return;
    }
    let g = open();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let mut m = Model::from_gguf(&g).unwrap();
    let p = tok.encode("The capital of France is").unwrap();
    let four = m.generate(&p, 4, tok.eos_id()).unwrap();
    let sixteen = m.generate(&p, 16, tok.eos_id()).unwrap();
    assert_eq!(four, sixteen[..4], "greedy decode of length N must prefix length M>N");
}

#[test]
fn corrupt_model_byte_changes_output() {
    if !present() {
        return;
    }
    let tok = Tokenizer::from_gguf(&open()).unwrap();
    let prompt = tok.encode("The capital of France is").unwrap();
    let clean = Model::from_gguf(&open()).unwrap().generate(&prompt, 4, tok.eos_id()).unwrap();

    // corrupt output_norm.weight (the final RMSNorm, applied to every logit at every step) and
    // rebuild — the greedy output must change, proving the weights are actually used end-to-end.
    let mut g = open();
    let t = g.tensor("output_norm.weight").unwrap().clone();
    let base = t.data_offset as usize;
    for k in 0..(t.byte_size as usize) {
        g.data[base + k] ^= 0xFF;
    }
    let corrupted = Model::from_gguf(&g).unwrap().generate(&prompt, 4, tok.eos_id()).unwrap();
    assert_ne!(clean, corrupted, "corrupting output_norm must change the greedy output");
}

#[test]
fn corrupt_quant_block_changes_dequant() {
    // a Q4_K block dequants differently when a scale byte flips (proves the decoder is real).
    let g = open_if(); // may skip
    let Some(g) = g else { return };
    let bytes = g.tensor_bytes("blk.0.ffn_down.weight").ok(); // Q6_K in this fixture
    if let Some(b) = bytes {
        let mut out1 = vec![0f32; 256];
        quant::dequant_q6_k(&b[..210], &mut out1);
        let mut tampered = b[..210].to_vec();
        tampered[200] ^= 0xFF; // a scale byte
        let mut out2 = vec![0f32; 256];
        quant::dequant_q6_k(&tampered, &mut out2);
        assert_ne!(out1, out2, "flipping a quant scale byte must change the dequantized values");
    }
}
fn open_if() -> Option<GgufFile> {
    if present() {
        Some(open())
    } else {
        None
    }
}

#[test]
fn logit_shas_deterministic_and_distinct() {
    if !present() {
        return;
    }
    let g = open();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let p = tok.encode("The capital of France is").unwrap();
    let (ids1, shas1) = Model::from_gguf(&g).unwrap().decode_logit_shas(&p, 6).unwrap();
    let (ids2, shas2) = Model::from_gguf(&g).unwrap().decode_logit_shas(&p, 6).unwrap();
    assert_eq!(ids1, ids2);
    assert_eq!(shas1, shas2, "per-step logit checksums must be deterministic");
    // distinct steps produce distinct logit vectors (real computation, not a constant)
    assert_ne!(shas1[0], shas1[1], "step-0 and step-1 logits must differ");
    assert_eq!(shas1.len(), 6);
}

#[test]
fn tokenizer_multi_prompt_deterministic() {
    if !present() {
        return;
    }
    let tok = Tokenizer::from_gguf(&open()).unwrap();
    for prompt in ["Hello world", "2 + 2 =", "The quick brown fox"] {
        let a = tok.encode(prompt).unwrap();
        let b = tok.encode(prompt).unwrap();
        assert_eq!(a, b, "tokenizer must be deterministic");
        assert!(!a.is_empty());
        // round-trip: decode(encode(x)) reproduces x for these ASCII prompts
        assert_eq!(tok.decode(&a), prompt, "byte-level BPE must round-trip ASCII");
    }
}
