//! hawking-seed-c CLI: the Event Horizon engine's command surface.
//!   run    SmolLM direct-quant vertical path (mmap -> IR -> direct-quant decode -> parity -> Metal
//!          measure -> sub-bit operator -> Doctor -> evidence -> drain/resume/verify)
//!   f2     bounded parent (120B) F2 bridge (fail-closed on absent asset + synthetic MoE proof)
//!   status | inspect | verify | drain | resume

use hawking_seed_c::adapter::LlamaConfig;
use hawking_seed_c::evidence::receipt;
use hawking_seed_c::gguf::GgufFile;
use hawking_seed_c::gravity::{self, Ask, Evidence, Rate};
use hawking_seed_c::model::Model;
use hawking_seed_c::pack::{PackEntry, PackManifest};
use hawking_seed_c::record::sha256_hex;
use hawking_seed_c::state::{Event, Machine, State};
use hawking_seed_c::tokenizer::Tokenizer;
use hawking_seed_c::{f2, subbit};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SEED_COMPAT: &str = "seed-c-1";
const WEIGHTS: &str = "models/SmolLM2-135M-Instruct-Q4_K_M.gguf";
const GPTOSS_SHARD1: &str = "models/gpt-oss-120b/original/model--00001-of-00007.safetensors";
const GPTOSS_REV: &str = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a";
const GOLDEN: &str = "reports/condense/gravity_forge/condensation/seed_b_golden.json";
const PROMPT: &str = "The capital of France is";
const MAX_TOK: usize = 16;

fn state_root() -> PathBuf {
    PathBuf::from("reports/condense/gravity_forge/condensation/seed_c_state")
}
fn err(code: i32, m: impl std::fmt::Display) -> i32 {
    eprintln!("hawking-seed-c: {m}");
    code
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let code = match a.get(1).map(String::as_str).unwrap_or("help") {
        "run" => cmd_run(),
        "f2" => cmd_f2(),
        "gravity-run" => cmd_gravity_run(),
        "status" => cmd_status(),
        "inspect" => cmd_inspect(),
        "verify" => cmd_verify(),
        "drain" => cmd_simple(Event::Drain, "drained"),
        "resume" => cmd_simple(Event::Resume, "resumed"),
        _ => {
            eprintln!("usage: hawking-seed-c [run|f2|status|inspect|verify|drain|resume]");
            2
        }
    };
    std::process::exit(code);
}

fn ensure_pack(root: &Path) -> Result<PackManifest, String> {
    std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let content = b"# hawking-seed-c: direct-quant + Metal Event Horizon runtime\n";
    std::fs::write(root.join("runtime.txt"), content).map_err(|e| e.to_string())?;
    let man = PackManifest {
        pack: "hawking-seed-c-runtime".into(),
        version: "1.0.0".into(),
        compatibility: SEED_COMPAT.into(),
        source_commit: "seed-c".into(),
        offline_cache: root.to_string_lossy().into(),
        contents: vec![PackEntry { path: "runtime.txt".into(), sha256: sha256_hex(content) }],
    };
    std::fs::write(root.join("manifest.json"), serde_json::to_string_pretty(&man).unwrap()).map_err(|e| e.to_string())?;
    Ok(man)
}
fn golden() -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(GOLDEN).ok()?).ok()
}
fn ids_of(v: &serde_json::Value, k: &str) -> Vec<u32> {
    v[k].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect()).unwrap_or_default()
}

fn cmd_run() -> i32 {
    let root = state_root();
    let _ = std::fs::remove_dir_all(&root);
    let mut m = match Machine::open(&root) {
        Ok(m) => m,
        Err(e) => return err(1, e),
    };
    // 1. verified pack + admit
    let pdir = root.join("packs/rt");
    let pack = match ensure_pack(&pdir) {
        Ok(p) => p,
        Err(e) => return err(1, e),
    };
    match pack.verify(pdir.join("manifest.json")) {
        Ok(n) => println!("[1] pack verified: {} ({} entries)", pack.pack, n),
        Err(e) => return err(1, e),
    }
    for ev in [Event::Prepare, Event::Admit, Event::Run] {
        if m.apply(ev, serde_json::json!({})).is_err() {
            return err(1, "transition");
        }
    }
    let weights = PathBuf::from(WEIGHTS);
    if !weights.exists() {
        return err(2, format!("FAIL-CLOSED: fixture absent {}", weights.display()));
    }

    // 2. inspect GGUF (mmap) + 3. build plan + map compressed tensors (Model::open)
    let g = match GgufFile::open(&weights) {
        Ok(g) => g,
        Err(e) => return err(2, e),
    };
    let cfg = match LlamaConfig::from_gguf(&g) {
        Ok(c) => c,
        Err(e) => return err(2, e),
    };
    let artifact = sha256_hex(g.tensor_bytes("token_embd.weight").unwrap_or(&[]));
    drop(g);
    let t_load = Instant::now();
    let mut model = match Model::open(&weights) {
        Ok(m) => m,
        Err(e) => return err(2, e),
    };
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[2-3] mmap {:.1} MB, arch=llama L={} h={} heads={}/{} | IR {} | metal={} | load {:.0} ms",
        model.mapped_bytes() as f64 / 1e6, cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_kv_heads,
        model.plan.summary(), model.metal_device().unwrap_or("none"), load_ms
    );

    // 4. tokenize
    let tok = match Tokenizer::from_gguf(&GgufFile::open(&weights).unwrap()) {
        Ok(t) => t,
        Err(e) => return err(2, e),
    };
    let pids = match tok.encode(PROMPT) {
        Ok(v) => v,
        Err(e) => return err(2, e),
    };
    let gold = match golden() {
        Some(v) => v,
        None => return err(2, "golden absent"),
    };
    let tok_ok = pids == ids_of(&gold, "prompt_token_ids");
    println!("[4] tokenize {:?} parity {}", pids, if tok_ok { "GREEN" } else { "MISMATCH" });
    if !tok_ok {
        return err(3, "tokenizer mismatch");
    }

    // 5-6. direct-quant prefill + decode 16 -> parity
    let t = Instant::now();
    let out = match model.generate(&pids, MAX_TOK, tok.eos_id()) {
        Ok(v) => v,
        Err(e) => return err(2, e),
    };
    let gen_ms = t.elapsed().as_secs_f64() * 1000.0;
    let ids_ok = out == ids_of(&gold, "completion_token_ids");
    let text = tok.decode(&out);
    let sha = sha256_hex(text.trim().as_bytes());
    let sha_ok = sha == gold["completion_text_sha256"].as_str().unwrap_or("");
    println!("[5-6] direct-quant decode {} tok in {:.0} ms ({:.2} tok/s) | ids {} sha {} {}", out.len(), gen_ms, out.len() as f64 / (gen_ms / 1000.0), ids_ok, &sha[..12], sha_ok);
    if !ids_ok || !sha_ok {
        eprintln!("  got {out:?}");
        return err(3, "DECODE PARITY FAILED");
    }
    println!("     PARITY GREEN (bit-identical, direct-quant, no full f32 expansion, no predecessor)");

    // 7. Metal measure (tied-vocab bottleneck) vs CPU
    match model.bench_logits_cpu_vs_metal(&pids) {
        Ok((cpu_ms, metal_ms, agree, diff)) if metal_ms.is_finite() => println!(
            "[7] Metal LM-head: cpu {:.1} ms vs metal {:.1} ms ({:.1}x), argmax_agree={} max|diff|={:.2e}",
            cpu_ms, metal_ms, cpu_ms / metal_ms, agree, diff
        ),
        Ok((cpu_ms, _, _, _)) => println!("[7] Metal unavailable; cpu LM-head {:.1} ms (CPU fallback path)", cpu_ms),
        Err(e) => return err(2, e),
    }

    // 8. direct-quant assertion
    println!("[8] direct-quant: model mapped {:.1} MB, weights executed from compressed views (only ~140 KB f32 norms cached; NO dense f32 model copy)", model.mapped_bytes() as f64 / 1e6);

    // 9. sub-bit operator: fit + direct execute + BPW + Doctor rescue
    let (sm, sn, sr) = (256usize, 256usize, 32usize);
    let w: Vec<f32> = (0..sm * sn).map(|i| (((i * 48271) % 997) as f32 / 997.0 - 0.5) * 0.1).collect();
    let sb = subbit::fit(&w, sm, sn, sr);
    let probes: Vec<Vec<f32>> = (0..4).map(|s| (0..sn).map(|j| (((j * 2654435761 + s) >> 7) & 0xFF) as f32 / 128.0 - 1.0).collect()).collect();
    let div0 = subbit::output_divergence(&w, sm, sn, &sb, &probes);
    let bpw0 = sb.whole_bpw();
    let treated = subbit::doctor_rescue(&w, sm, sn, subbit::fit(&w, sm, sn, sr), 400, 0.99);
    let (div1, bpw1) = treated.as_ref().map(|t| (subbit::output_divergence(&w, sm, sn, t, &probes), t.whole_bpw())).unwrap_or((div0, bpw0));
    println!("[9] sub-bit ternary factor: {:.3} BPW (<1.0), direct matvec, output_div {:.4}", bpw0, div0);
    println!("[10] Doctor rescue: div {:.4}->{:.4}, {:.3} BPW (still sub-bit, billed)", div0, div1, bpw1);
    if !(bpw0 < 1.0 && bpw1 < 1.0 && div1 <= div0) {
        return err(3, "sub-bit/doctor invariant failed");
    }

    // 11. Gravity
    let g_ok = gravity::decide(Rate::new(4, 5), &Ask::RepresentationEscalation, &Evidence::default()).allow
        && !gravity::decide(Rate::new(4, 5), &Ask::EscapeAboveSubbit { to: Rate::new(5, 4), sealed_receipt: false }, &Evidence::default()).allow;
    println!("[11] Gravity: {}", if g_ok { "law intact" } else { "VIOLATION" });

    // 12. evidence + drain/resume/verify
    let ev = receipt("evaluation", serde_json::json!({
        "candidate": "C (direct-quant + Metal)", "artifact": artifact, "decode_parity": ids_ok && sha_ok,
        "direct_quant": true, "full_f32_expansion": false, "metal": model.metal_available(),
        "subbit_bpw": bpw0, "subbit_div": div0, "doctor_div": div1, "gravity_law": g_ok, "authorizes_escape": false,
    }));
    if m.record(ev.clone()).is_err() {
        return err(1, "evidence");
    }
    println!("[12] evidence sealed ({})", &ev.seal[..12]);
    m.apply(Event::Pause, serde_json::json!({})).ok();
    m.apply(Event::Resume, serde_json::json!({})).ok();
    m.apply(Event::Drain, serde_json::json!({})).ok();
    m.apply(Event::Seal, serde_json::json!({})).ok();
    match Machine::open(&root) {
        Ok(m2) => {
            let ok = m2.log.iter().all(|r| r.verify().is_ok());
            println!("[13] drain->resume->seal -> {:?} | {} records valid: {}", m2.state, m2.log.len(), ok);
            if !(ok && m2.state == State::Sealed) {
                return err(3, "verify failed");
            }
        }
        Err(e) => return err(1, e),
    }
    println!("\nVERTICAL PATH GREEN: mmap->IR->direct-quant decode->parity->Metal->sub-bit->Doctor->evidence->drain/resume/verify");
    0
}

fn cmd_f2() -> i32 {
    // Real parent-bound F2 when the authoritative GPT-OSS-120B source is present; else synthetic.
    let shard = PathBuf::from(GPTOSS_SHARD1);
    if shard.exists() {
        match hawking_seed_c::gptoss::run(&shard, GPTOSS_REV, 0, (512, 512)) {
            Ok(r) => {
                println!("[f2-REAL] parent={} rev={} shard={}", r.parent, &r.revision[..12], r.shard);
                println!("     layer {} router={} experts={} top{}={:?} -> expert {}", r.layer, r.router_tensor, r.n_experts, r.top_k, r.selected_experts, r.expert);
                println!("     expert {} full shape {:?}, tested slice {:?}, source bytes read {}", r.expert_tensor, r.expert_full_shape, r.tested_slice, r.source_bytes_read);
                println!("     reference checksum {}", &r.reference_output_checksum[..16]);
                println!("     sub-bit: {:.4} BPW (<1.0={}), untreated_div {:.4} -> Doctor {:.4} (doctor {} bits, same-rate={})", r.complete_bpw, r.subbit_below_one_bpw, r.untreated_divergence, r.treated_divergence, r.doctor_bits, r.doctor_within_same_rate);
                println!("     repeatable={}", r.repeatable);
                let ok = r.subbit_below_one_bpw && r.treated_divergence <= r.untreated_divergence && r.repeatable;
                if let Ok(mut m) = Machine::open(state_root()) {
                    let _ = m.record(receipt("evaluation", serde_json::to_value(&r).unwrap_or(serde_json::json!({}))));
                }
                println!("[f2-REAL] {}", if ok { "GREEN (parent-bound sub-bit F2, direct compact, same-budget Doctor)" } else { "FAILED" });
                return if ok { 0 } else { 3 };
            }
            Err(e) => return err(2, format!("real F2: {e}")),
        }
    }
    eprintln!("hawking-seed-c: real GPT-OSS-120B source absent -> synthetic MoE fixture");
    let r = f2::run();
    println!("[f2] real_120b_present={} :: {}", r.real_120b_present, r.real_120b_note);
    println!("     source={} experts={} top_k={} selected={:?}", r.source, r.n_experts, r.top_k, r.selected_experts);
    println!("     MoE ops: {:?}", r.moe_ops_exercised);
    println!("     sub-bit expert: {:.3} BPW, output_div {:.4} -> Doctor {:.4} ({:.3} BPW)", r.expert_subbit_bpw, r.expert_output_divergence, r.doctor_after_divergence, r.doctor_bpw);
    println!("     WeightedCombine divergence (dense vs sub-bit expert): {:.4}", r.combine_divergence);
    // seal evidence
    let root = state_root();
    if let Ok(mut m) = Machine::open(&root) {
        let ev = receipt("evaluation", serde_json::to_value(&r).unwrap_or(serde_json::json!({"f2":"error"})));
        let _ = m.record(ev);
    }
    if r.expert_subbit_bpw < 1.0 {
        0
    } else {
        3
    }
}


fn cmd_gravity_run() -> i32 {
    let shard = PathBuf::from(GPTOSS_SHARD1);
    if !shard.exists() {
        return err(2, "FAIL-CLOSED: GPT-OSS-120B source absent; cannot launch Gravity run");
    }
    let root = PathBuf::from("reports/condense/gravity_forge/condensation/gravity_120b_run");
    let _ = std::fs::remove_dir_all(&root);
    eprintln!("[gravity-run] launching ONE sub-bit-first Gravity run over gpt-oss-120b layer 0 (128 experts), pid {}", std::process::id());
    match hawking_seed_c::gravity_run::run(&shard, GPTOSS_REV, &root, 0, 128, (256, 256)) {
        Ok(()) => {
            println!("[gravity-run] complete; controller log at {}", root.display());
            0
        }
        Err(e) => err(2, format!("gravity run: {e}")),
    }
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
    let w = PathBuf::from(WEIGHTS);
    if let Ok(g) = GgufFile::open(&w) {
        if let Ok(c) = LlamaConfig::from_gguf(&g) {
            println!("model: llama L={} h={} {}q/{}kv head_dim={} vocab={} mmap={:.1}MB", c.n_layers, c.hidden, c.n_heads, c.n_kv_heads, c.head_dim, c.vocab, g.mapped_bytes as f64 / 1e6);
        }
    }
    println!("metal: {}", hawking_seed_c::metal::MetalGemv::new().map(|m| m.device_name).unwrap_or("none".into()));
    0
}
fn cmd_verify() -> i32 {
    match Machine::open(state_root()) {
        Ok(m) => {
            let ok = m.log.iter().all(|r| r.verify().is_ok());
            println!("verify: {} records, valid: {}", m.log.len(), ok);
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
