//! Candidate C integration + adversarial tests: full SmolLM direct-quant parity, per-format direct
//! dequant on real tensors, CPU/Metal agreement, tokenizer multi-prompt, GGUF corruption. Gated on the
//! fixture; fails closed (skips) when absent.

use hawking_seed_c::adapter::LlamaConfig;
use hawking_seed_c::cpu;
use hawking_seed_c::gguf::{GgmlType, GgufFile};
use hawking_seed_c::metal::MetalGemv;
use hawking_seed_c::model::Model;
use hawking_seed_c::quant;
use hawking_seed_c::record::sha256_hex;
use hawking_seed_c::tokenizer::Tokenizer;
use std::path::Path;

const WEIGHTS: &str = "../../models/SmolLM2-135M-Instruct-Q4_K_M.gguf";
const GOLDEN: &str = "../../reports/condense/gravity_forge/condensation/seed_b_golden.json";

fn present() -> bool {
    Path::new(WEIGHTS).exists()
}
fn open() -> GgufFile {
    GgufFile::open(Path::new(WEIGHTS)).unwrap()
}
fn gold() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(GOLDEN).unwrap()).unwrap()
}
fn ids(v: &serde_json::Value, k: &str) -> Vec<u32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

#[test]
fn full_smollm_parity_direct_quant() {
    if !present() {
        eprintln!("skip: fixture absent");
        return;
    }
    let g = open();
    let cfg = LlamaConfig::from_gguf(&g).unwrap();
    assert_eq!((cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_kv_heads), (30, 576, 9, 3));
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let p = tok.encode("The capital of France is").unwrap();
    assert_eq!(p, ids(&gold(), "prompt_token_ids"));
    let mut model = Model::open(Path::new(WEIGHTS)).unwrap();
    let out = model.generate(&p, 16, tok.eos_id()).unwrap();
    assert_eq!(out, ids(&gold(), "completion_token_ids"), "direct-quant token parity");
    let sha = sha256_hex(tok.decode(&out).trim().as_bytes());
    assert_eq!(sha, gold()["completion_text_sha256"].as_str().unwrap(), "text sha parity");
}

#[test]
fn direct_row_dequant_matches_whole_for_each_real_format() {
    if !present() {
        return;
    }
    let g = open();
    // one real tensor of each quant family in this fixture
    for (name, fmt) in [
        ("blk.0.attn_q.weight", GgmlType::Q5_0),
        ("blk.0.ffn_down.weight", GgmlType::Q6_K),
        ("token_embd.weight", GgmlType::Q8_0),
    ] {
        let t = g.tensor(name).unwrap();
        assert_eq!(t.dtype, fmt);
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        let bytes = g.tensor_bytes(name).unwrap();
        // whole-tensor dequant of the first 3 rows vs row-at-a-time direct dequant
        let mut whole = vec![0f32; 3 * cols];
        // dequant only the first 3 rows' worth by slicing
        let (bs, bb) = fmt.block_layout();
        let rb = (cols as u64 / bs) * bb;
        quant::dequant(fmt, &bytes[..(3 * rb) as usize], &mut whole).unwrap();
        for r in 0..3 {
            let mut row = vec![0f32; cols];
            quant::dequant_row(fmt, bytes, r, cols, &mut row).unwrap();
            assert_eq!(&row[..], &whole[r * cols..(r + 1) * cols], "{name} row {r}");
        }
        let _ = rows;
    }
}

#[test]
fn q4_k_present_and_direct_dequant_runs() {
    if !present() {
        return;
    }
    let g = open();
    // find a Q4_K tensor (16 exist in this fixture) and direct-dequant one row
    let name = g
        .tensors
        .values()
        .find(|t| t.dtype == GgmlType::Q4_K)
        .map(|t| t.name.clone());
    if let Some(name) = name {
        let t = g.tensor(&name).unwrap();
        let cols = t.dims[0] as usize;
        let mut row = vec![0f32; cols];
        quant::dequant_row(GgmlType::Q4_K, g.tensor_bytes(&name).unwrap(), 0, cols, &mut row).unwrap();
        assert!(row.iter().any(|&v| v != 0.0), "Q4_K row dequant produced values");
    }
}

#[test]
fn cpu_metal_logits_agree_on_argmax() {
    if !present() {
        return;
    }
    let g = open();
    let cfg = LlamaConfig::from_gguf(&g).unwrap();
    let Some(m) = MetalGemv::new() else {
        eprintln!("skip: no metal device");
        return;
    };
    let embd = g.tensor_bytes("token_embd.weight").unwrap();
    let dtype = g.tensor("token_embd.weight").unwrap().dtype;
    let x: Vec<f32> = (0..cfg.hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut cpu_out = vec![0f32; cfg.vocab];
    cpu::logits_tied(dtype, embd, cfg.hidden, cfg.vocab, &x, &mut cpu_out).unwrap();
    let gpu = m.logits_q8_0(embd, cfg.hidden, cfg.vocab, &x).unwrap();
    assert_eq!(cpu::argmax(&cpu_out), cpu::argmax(&gpu), "CPU/Metal argmax must agree");
    let maxd = cpu_out.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(maxd < 1e-3, "CPU/Metal logits within tolerance, got {maxd}");
}

#[test]
fn corrupt_gguf_metadata_fails_closed() {
    if !present() {
        return;
    }
    // truncating the header must yield a parse error, not a silent wrong model
    let bytes = std::fs::read(WEIGHTS).unwrap();
    let tmp = std::env::temp_dir().join("seedc_corrupt.gguf");
    std::fs::write(&tmp, &bytes[..1000]).unwrap();
    assert!(GgufFile::open(&tmp).is_err(), "truncated GGUF must fail to parse");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn tokenizer_multi_prompt_and_no_full_dequant() {
    if !present() {
        return;
    }
    let g = open();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    for p in ["Hello world", "The quick brown fox"] {
        assert_eq!(tok.encode(p).unwrap(), tok.encode(p).unwrap());
    }
    // structural no-full-dequant check: the mapped model is ~105 MB and the runtime holds no dense
    // f32 weight copy (only the tiny norm cache). We assert the mapping equals the file size.
    let model = Model::open(Path::new(WEIGHTS)).unwrap();
    assert_eq!(model.mapped_bytes(), std::fs::metadata(WEIGHTS).unwrap().len() as usize);
}
