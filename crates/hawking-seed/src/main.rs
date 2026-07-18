//! hawking-seed CLI: one registry-driven command surface over the Seed vertical path.
//!   run     execute the complete real vertical path (pack -> record -> state -> artifact -> decode
//!           -> parity -> Gravity -> Forge -> Doctor -> evidence -> drain -> resume -> verify)
//!   status | inspect | verify | drain | resume

use hawking_seed::evidence::receipt;
use hawking_seed::forge;
use hawking_seed::gravity::{self, Ask, Evidence, Rate};
use hawking_seed::pack::{PackEntry, PackManifest};
use hawking_seed::record::sha256_hex;
use hawking_seed::runtime::{DecodeSpec, DefaultRuntimePack, Runtime};
use hawking_seed::state::{Event, Machine, State};
use std::path::{Path, PathBuf};

const SEED_COMPAT: &str = "seed-1";
const GOLDEN: &str = "reports/condense/gravity_forge/condensation/decode_parity_golden.json";
const WEIGHTS: &str = "models/SmolLM2-135M-Instruct-Q4_K_M.gguf";
const PROMPT: &str = "The capital of France is";
const MAX_TOK: usize = 16;

fn state_root() -> PathBuf {
    PathBuf::from("reports/condense/gravity_forge/condensation/seed_a_state")
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
            eprintln!("usage: hawking-seed [run|status|inspect|verify|drain|resume]");
            2
        }
    };
    std::process::exit(code);
}

fn err(code: i32, msg: impl std::fmt::Display) -> i32 {
    eprintln!("hawking-seed: {msg}");
    code
}

/// Build + verify a tiny default pack (content-addressed), proving pack identity/hash/tamper/offline.
fn ensure_default_pack(root: &Path) -> Result<PackManifest, String> {
    std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let content = b"# hawking-forge-default: int8 forge + sparse Doctor (see crate forge/doctor)\n";
    let content_path = root.join("forge_default.txt");
    std::fs::write(&content_path, content).map_err(|e| e.to_string())?;
    let man = PackManifest {
        pack: "hawking-forge-default".into(),
        version: "1.0.0".into(),
        compatibility: SEED_COMPAT.into(),
        source_commit: "seed-a".into(),
        offline_cache: root.to_string_lossy().into(),
        contents: vec![PackEntry { path: "forge_default.txt".into(), sha256: sha256_hex(content) }],
    };
    let man_path = root.join("manifest.json");
    std::fs::write(&man_path, serde_json::to_string_pretty(&man).unwrap()).map_err(|e| e.to_string())?;
    Ok(man)
}

fn golden_sha() -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(GOLDEN).ok()?).ok()?;
    v["modes"]["baseline_no_spec"]["text_sha256"].as_str().map(String::from)
}

fn cmd_run() -> i32 {
    let root = state_root();
    let _ = std::fs::remove_dir_all(&root);
    let mut m = match Machine::open(&root) {
        Ok(m) => m,
        Err(e) => return err(1, e),
    };

    // 1. verified pack ------------------------------------------------------------------
    let pack_dir = root.join("packs/hawking-forge-default");
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

    // 2-4. record + state transitions + artifact identity --------------------------------
    for ev in [Event::Prepare, Event::Admit, Event::Run] {
        if let Err(e) = m.apply(ev, serde_json::json!({"pack": pack.pack})) {
            return err(1, e);
        }
    }
    println!("[2-3] state -> {:?} (transitions persisted + sealed)", m.state);
    let weights = PathBuf::from(WEIGHTS);
    if !weights.exists() {
        return err(2, format!("FAIL-CLOSED: model fixture absent at {}", weights.display()));
    }
    let artifact_id = match std::fs::read(&weights) {
        Ok(b) => sha256_hex(&b),
        Err(e) => return err(2, e),
    };
    println!("[4] artifact identity: sha256 {}", &artifact_id[..16]);

    // 5-7. real fixture load + deterministic greedy decode + parity ----------------------
    let rt = match DefaultRuntimePack::locate() {
        Ok(r) => r,
        Err(e) => return err(2, e),
    };
    let spec = DecodeSpec {
        weights: weights.clone(),
        prompt: PROMPT.into(),
        max_tokens: MAX_TOK,
        profile: "exact".into(),
        seed: 0,
    };
    let dec = match rt.decode_greedy(&spec) {
        Ok(d) => d,
        Err(e) => return err(2, e),
    };
    let golden = golden_sha();
    let parity = golden.as_deref() == Some(dec.text_sha256.as_str());
    println!(
        "[5-7] decode ({} tok) sha {} | parity vs golden: {}",
        dec.tokens,
        &dec.text_sha256[..12],
        if parity { "GREEN (bit-identical)" } else { "MISMATCH" }
    );
    if golden.is_none() {
        return err(2, "FAIL-CLOSED: golden reference fixture absent");
    }
    if !parity {
        return err(3, "decode parity FAILED");
    }

    // 8. Gravity sub-bit decision fixture -----------------------------------------------
    let g_subbit = gravity::decide(Rate::new(4, 5), &Ask::RepresentationEscalation, &Evidence::default());
    let g_escape_denied = gravity::decide(
        Rate::new(4, 5),
        &Ask::EscapeAboveSubbit { to: Rate::new(5, 4), sealed_receipt: false },
        &Evidence::default(),
    );
    let gravity_ok = g_subbit.allow && !g_escape_denied.allow && Rate::new(4, 5).is_subbit();
    println!(
        "[8] Gravity: sub-bit default={} escape-without-receipt denied={} -> {}",
        Rate::new(4, 5).is_subbit(),
        !g_escape_denied.allow,
        if gravity_ok { "law intact" } else { "VIOLATION" }
    );
    if !gravity_ok {
        return err(3, "Gravity law violated");
    }

    // 9. Forge pack + round-trip fixture -------------------------------------------------
    let (rows, cols) = (16usize, 16usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i % 11) as f32 - 5.0) * 0.07).collect();
    let packed = match forge::pack(&w, rows, cols) {
        Ok(p) => p,
        Err(e) => return err(3, e),
    };
    let recon = forge::decode(&packed);
    let f_err = forge::rel_error(&w, &recon);
    println!(
        "[9] Forge int8: whole {:.3} BPW, {} bytes, round-trip rel_err {:.4}",
        packed.whole_artifact_bpw(),
        packed.physical_bytes(),
        f_err
    );

    // 10. Doctor same-budget treatment fixture ------------------------------------------
    let mut wd = w.clone();
    for r in 0..rows {
        wd[r * cols + 7] = if r % 2 == 0 { 8.0 } else { -8.0 }; // an outlier column int8 mangles
    }
    let pd = forge::pack(&wd, rows, cols).unwrap();
    let (treat, rep) = match hawking_seed::doctor::treat(&wd, &pd, 2, 20.0) {
        Ok(x) => x,
        Err(e) => return err(3, e),
    };
    println!(
        "[10] Doctor: {} | rel_err {:.4}->{:.4}, total {:.3} BPW (base {:.3}+doctor {:.3}), within_budget={}",
        treat.diagnosis, rep.before_rel_error, rep.after_rel_error, rep.total_bpw, rep.base_bpw, rep.doctor_bpw, rep.within_budget
    );
    if !(rep.improved && rep.within_budget) {
        return err(3, "Doctor treatment did not improve within budget");
    }

    // 11. sealed evidence receipts -------------------------------------------------------
    let ev_rec = receipt(
        "evaluation",
        serde_json::json!({
            "artifact": artifact_id, "parity": parity, "decode_sha": dec.text_sha256,
            "gravity_law": gravity_ok, "forge_bpw": packed.whole_artifact_bpw(),
            "doctor_before": rep.before_rel_error, "doctor_after": rep.after_rel_error,
            "doctor_total_bpw": rep.total_bpw, "authorizes_escape": false,
        }),
    );
    if let Err(e) = m.record(ev_rec.clone()) {
        return err(1, e);
    }
    println!("[11] evidence sealed: receipt {} ({})", ev_rec.kind, &ev_rec.seal[..12]);

    // 12. drain -> resume -> verify ------------------------------------------------------
    m.apply(Event::Pause, serde_json::json!({})).ok();
    m.apply(Event::Resume, serde_json::json!({})).ok();
    println!("[12a] pause->resume -> {:?}", m.state);
    m.apply(Event::Drain, serde_json::json!({})).ok();
    m.apply(Event::Seal, serde_json::json!({})).ok();
    println!("[12b] drain->seal -> {:?}", m.state);

    // verify: re-open the machine and re-verify every sealed record.
    match Machine::open(&root) {
        Ok(m2) => {
            let n = m2.log.len();
            let all_ok = m2.log.iter().all(|r| r.verify().is_ok());
            println!("[verify] resumed {} sealed records, all seals valid: {}", n, all_ok);
            if !(all_ok && m2.state == State::Sealed) {
                return err(3, "verify failed");
            }
        }
        Err(e) => return err(1, e),
    }

    println!("\nVERTICAL PATH GREEN: pack->record->state->artifact->decode->parity->Gravity->Forge->Doctor->evidence->drain->resume->verify");
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
    match Machine::open(state_root()) {
        Ok(m) => {
            for r in m.log.iter().rev().take(6) {
                println!("{:>12} {} {}", r.kind, r.state, &r.identity[..12]);
            }
            match DefaultRuntimePack::locate() {
                Ok(rt) => println!("runtime: {}", rt.inspect()),
                Err(_) => println!("runtime: default pack not built"),
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
