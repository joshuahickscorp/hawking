//! hawking-seed-b CLI: one registry-driven command surface over Candidate B's self-contained vertical
//! path. Unlike Candidate A, decode runs on THIS crate's own runtime (GGUF reader + dequant + tokenizer
//! + IR + forward), never the predecessor engine.
//!   run     execute the complete real vertical path
//!   status | inspect | verify | drain | resume

use hawking_seed_b::adapter::LlamaConfig;
use hawking_seed_b::evidence::receipt;
use hawking_seed_b::gguf::GgufFile;
use hawking_seed_b::gravity::{self, Ask, Evidence, Rate};
use hawking_seed_b::model::Model;
use hawking_seed_b::pack::{PackEntry, PackManifest};
use hawking_seed_b::record::sha256_hex;
use hawking_seed_b::state::{Event, Machine, State};
use hawking_seed_b::tokenizer::Tokenizer;
use hawking_seed_b::{forge, ops};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SEED_COMPAT: &str = "seed-b-1";
const WEIGHTS: &str = "models/SmolLM2-135M-Instruct-Q4_K_M.gguf";
const GOLDEN: &str = "reports/condense/gravity_forge/condensation/seed_b_golden.json";
const PROMPT: &str = "The capital of France is";
const MAX_TOK: usize = 16;

fn state_root() -> PathBuf {
    PathBuf::from("reports/condense/gravity_forge/condensation/seed_b_state")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let code = match cmd {
        "run" => cmd_run(),
        "status" => cmd_status(),
        "inspect" => cmd_inspect(),
        "verify" => cmd_verify(),
        "drain" => cmd_simple(Event::Drain, "drained"),
        "resume" => cmd_simple(Event::Resume, "resumed"),
        _ => {
            eprintln!("usage: hawking-seed-b [run|status|inspect|verify|drain|resume]");
            2
        }
    };
    std::process::exit(code);
}

fn err(code: i32, msg: impl std::fmt::Display) -> i32 {
    eprintln!("hawking-seed-b: {msg}");
    code
}

fn ensure_default_pack(root: &Path) -> Result<PackManifest, String> {
    std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let content = b"# hawking-seed-b default runtime pack: self-contained GGUF+dequant+tokenizer+IR forward\n";
    std::fs::write(root.join("runtime.txt"), content).map_err(|e| e.to_string())?;
    let man = PackManifest {
        pack: "hawking-seed-b-runtime".into(),
        version: "1.0.0".into(),
        compatibility: SEED_COMPAT.into(),
        source_commit: "seed-b".into(),
        offline_cache: root.to_string_lossy().into(),
        contents: vec![PackEntry { path: "runtime.txt".into(), sha256: sha256_hex(content) }],
    };
    std::fs::write(root.join("manifest.json"), serde_json::to_string_pretty(&man).unwrap())
        .map_err(|e| e.to_string())?;
    Ok(man)
}

fn golden() -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(GOLDEN).ok()?).ok()
}

fn ids_of(v: &serde_json::Value, key: &str) -> Vec<u32> {
    v[key]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect())
        .unwrap_or_default()
}

fn cmd_run() -> i32 {
    let root = state_root();
    let _ = std::fs::remove_dir_all(&root);
    let mut m = match Machine::open(&root) {
        Ok(m) => m,
        Err(e) => return err(1, e),
    };

    // 1. verified default (self-contained runtime) pack -----------------------------------
    let pack_dir = root.join("packs/hawking-seed-b-runtime");
    let pack = match ensure_default_pack(&pack_dir) {
        Ok(p) => p,
        Err(e) => return err(1, format!("pack build: {e}")),
    };
    if !pack.compatible_with(SEED_COMPAT) {
        return err(1, "pack incompatible");
    }
    match pack.verify(pack_dir.join("manifest.json")) {
        Ok(n) => println!("[1] pack verified: {} ({} content entries, offline)", pack.pack, n),
        Err(e) => return err(1, e),
    }

    for ev in [Event::Prepare, Event::Admit, Event::Run] {
        if let Err(e) = m.apply(ev, serde_json::json!({"pack": pack.pack})) {
            return err(1, e);
        }
    }

    let weights = PathBuf::from(WEIGHTS);
    if !weights.exists() {
        return err(2, format!("FAIL-CLOSED: model fixture absent at {}", weights.display()));
    }

    // 2. inspect real GGUF ----------------------------------------------------------------
    let g = match GgufFile::open(&weights) {
        Ok(g) => g,
        Err(e) => return err(2, e),
    };
    let cfg = match LlamaConfig::from_gguf(&g) {
        Ok(c) => c,
        Err(e) => return err(2, e),
    };
    let artifact_id = sha256_hex(&g.data);
    println!(
        "[2] GGUF: arch=llama layers={} hidden={} heads={}/{} head_dim={} vocab={} | artifact {}",
        cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.vocab, &artifact_id[..16]
    );

    // 3-4. build execution IR + load tensors (self-contained runtime) ----------------------
    let t_load = Instant::now();
    let mut model = match Model::from_gguf(&g) {
        Ok(m) => m,
        Err(e) => return err(2, e),
    };
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[3] IR built: {} | [4] tensors loaded ({:.2}M f32 weights + tied f16 embed) in {:.0} ms",
        model.plan.summary(),
        model.weight_f32_elems as f64 / 1e6,
        load_ms
    );

    // 5. tokenize prompt (own tokenizer) --------------------------------------------------
    let tok = match Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => return err(2, e),
    };
    let prompt_ids = match tok.encode(PROMPT) {
        Ok(ids) => ids,
        Err(e) => return err(2, e),
    };
    let gold = match golden() {
        Some(v) => v,
        None => return err(2, "FAIL-CLOSED: golden fixture absent"),
    };
    let gold_prompt = ids_of(&gold, "prompt_token_ids");
    let tok_parity = prompt_ids == gold_prompt;
    println!(
        "[5] tokenize: {:?} | tokenizer parity vs golden: {}",
        prompt_ids,
        if tok_parity { "GREEN" } else { "MISMATCH" }
    );
    if !tok_parity {
        return err(3, format!("tokenizer mismatch: got {prompt_ids:?} want {gold_prompt:?}"));
    }

    // 6-7. execute prefill + decode 16 (own forward pass) ---------------------------------
    let t_gen = Instant::now();
    let out_ids = match model.generate(&prompt_ids, MAX_TOK, tok.eos_id()) {
        Ok(v) => v,
        Err(e) => return err(2, e),
    };
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

    // 8. match golden (token ids + text sha) ----------------------------------------------
    let gold_ids = ids_of(&gold, "completion_token_ids");
    let ids_match = out_ids == gold_ids;
    let text = tok.decode(&out_ids);
    let text_sha = sha256_hex(text.trim().as_bytes());
    let gold_sha = gold["completion_text_sha256"].as_str().unwrap_or("");
    let sha_match = text_sha == gold_sha;
    println!("[6-7] decode {} tok in {:.0} ms ({:.2} tok/s)", out_ids.len(), gen_ms, out_ids.len() as f64 / (gen_ms / 1000.0));
    println!("[8] ids match golden: {} | text sha {} match: {}", ids_match, &text_sha[..12], sha_match);
    if !ids_match || !sha_match {
        eprintln!("  got ids {out_ids:?}");
        eprintln!("  want ids {gold_ids:?}");
        eprintln!("  got text {:?}", text.trim());
        return err(3, "DECODE PARITY FAILED");
    }
    println!("     PARITY GREEN (bit-identical greedy tokens, no predecessor engine)");

    // 9. Gravity vector -------------------------------------------------------------------
    let g_sub = gravity::decide(Rate::new(4, 5), &Ask::RepresentationEscalation, &Evidence::default());
    let g_esc = gravity::decide(
        Rate::new(4, 5),
        &Ask::EscapeAboveSubbit { to: Rate::new(5, 4), sealed_receipt: false },
        &Evidence::default(),
    );
    let gravity_ok = g_sub.allow && !g_esc.allow && Rate::new(4, 5).is_subbit();
    println!("[9] Gravity: sub-bit default + escape-without-receipt denied -> {}", if gravity_ok { "law intact" } else { "VIOLATION" });
    if !gravity_ok {
        return err(3, "Gravity law violated");
    }

    // 10. Forge fixture, executed through B's own gemv ------------------------------------
    let (rows, cols) = (16usize, 16usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i % 11) as f32 - 5.0) * 0.07).collect();
    let packed = match forge::pack(&w, rows, cols) {
        Ok(p) => p,
        Err(e) => return err(3, e),
    };
    let recon = forge::decode(&packed);
    let f_err = forge::rel_error(&w, &recon);
    // execute the reconstructed weight through Candidate B's compact-linear op
    let x = vec![1.0f32; cols];
    let mut gemv_out = vec![0f32; rows];
    ops::gemv(&recon, &x, rows, cols, &mut gemv_out);
    println!(
        "[10] Forge int8: {:.3} BPW, {} bytes, round-trip rel_err {:.4}, executed through B gemv (out[0]={:.3})",
        packed.whole_artifact_bpw(),
        packed.physical_bytes(),
        f_err,
        gemv_out[0]
    );

    // 11. Doctor treatment on a B fixture -------------------------------------------------
    let mut wd = w.clone();
    for r in 0..rows {
        wd[r * cols + 7] = if r % 2 == 0 { 8.0 } else { -8.0 };
    }
    let pd = forge::pack(&wd, rows, cols).unwrap();
    let (treat, rep) = match hawking_seed_b::doctor::treat(&wd, &pd, 2, 20.0) {
        Ok(x) => x,
        Err(e) => return err(3, e),
    };
    println!(
        "[11] Doctor: {} | rel_err {:.4}->{:.4}, total {:.3} BPW (base {:.3}+doctor {:.3}), within_budget={}",
        treat.diagnosis, rep.before_rel_error, rep.after_rel_error, rep.total_bpw, rep.base_bpw, rep.doctor_bpw, rep.within_budget
    );
    if !(rep.improved && rep.within_budget) {
        return err(3, "Doctor treatment did not improve within budget");
    }

    // 12. sealed evidence -----------------------------------------------------------------
    let ev = receipt(
        "evaluation",
        serde_json::json!({
            "candidate": "B (self-contained runtime)",
            "artifact": artifact_id,
            "decode_parity": ids_match && sha_match,
            "decode_sha": text_sha,
            "delegates_model_math_to_predecessor": false,
            "gravity_law": gravity_ok,
            "forge_bpw": packed.whole_artifact_bpw(),
            "doctor_after": rep.after_rel_error,
            "authorizes_escape": false,
        }),
    );
    if let Err(e) = m.record(ev.clone()) {
        return err(1, e);
    }
    println!("[12] evidence sealed: receipt {} ({})", ev.kind, &ev.seal[..12]);

    // 13. drain -> resume -> verify -------------------------------------------------------
    m.apply(Event::Pause, serde_json::json!({})).ok();
    m.apply(Event::Resume, serde_json::json!({})).ok();
    m.apply(Event::Drain, serde_json::json!({})).ok();
    m.apply(Event::Seal, serde_json::json!({})).ok();
    match Machine::open(&root) {
        Ok(m2) => {
            let ok = m2.log.iter().all(|r| r.verify().is_ok());
            println!("[13] pause->resume->drain->seal -> {:?} | {} sealed records, all seals valid: {}", m2.state, m2.log.len(), ok);
            if !(ok && m2.state == State::Sealed) {
                return err(3, "verify failed");
            }
        }
        Err(e) => return err(1, e),
    }

    println!("\nVERTICAL PATH GREEN: pack->GGUF->IR->tensors->tokenize->prefill->decode->parity->Gravity->Forge->Doctor->evidence->drain->resume->verify");
    0
}

fn cmd_status() -> i32 {
    match Machine::open(state_root()) {
        Ok(m) => {
            println!("state: {:?} | records: {}", m.state, m.log.len());
            0
        }
        Err(e) => err(1, e),
    }
}

fn cmd_inspect() -> i32 {
    let weights = PathBuf::from(WEIGHTS);
    if let Ok(g) = GgufFile::open(&weights) {
        if let Ok(cfg) = LlamaConfig::from_gguf(&g) {
            println!(
                "model: llama {} layers, hidden {}, {}q/{}kv heads, head_dim {}, vocab {}, rope_base {}",
                cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.vocab, cfg.rope_base
            );
        }
    } else {
        println!("model fixture not present at {}", weights.display());
    }
    match Machine::open(state_root()) {
        Ok(m) => {
            for r in m.log.iter().rev().take(6) {
                println!("{:>12} {} {}", r.kind, r.state, &r.identity[..12]);
            }
            0
        }
        Err(e) => err(1, e),
    }
}

fn cmd_verify() -> i32 {
    match Machine::open(state_root()) {
        Ok(m) => {
            let ok = m.log.iter().all(|r| r.verify().is_ok());
            println!("verify: {} records, all seals valid: {}", m.log.len(), ok);
            if ok {
                0
            } else {
                3
            }
        }
        Err(e) => err(1, e),
    }
}

fn cmd_simple(ev: Event, verb: &str) -> i32 {
    match Machine::open(state_root()) {
        Ok(mut m) => match m.apply(ev, serde_json::json!({})) {
            Ok(s) => {
                println!("{verb} -> {:?}", s);
                0
            }
            Err(e) => err(1, e),
        },
        Err(e) => err(1, e),
    }
}
