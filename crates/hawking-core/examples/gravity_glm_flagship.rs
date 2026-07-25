//! Run the GLM-5.2 adapter against the real assembled flagship artifact.
//!
//! This is gate M04 at scale: the fixture already proved the algorithm right
//! (3.8e-6 against the oracle, argmax/top5/DSA exact); this proves the same
//! adapter reads the real 282-shard, 77 GB artifact without falling over, and
//! reports what a real forward costs. Grading against the Python oracle's
//! frozen logits (if given one) happens by diffing this program's `--dump`
//! output against `tools/condense/glm52_flagship_oracle.py`'s.
//!
//! Lazy multi-shard loading is what makes this possible on a 96 GB machine at
//! all: only 8 of 256 experts activate per layer, so this touches roughly
//! 3.5% of the artifact's bytes for one token, not all of it.
//!
//!     cargo run --release -p hawking-core --example gravity_glm_flagship -- \
//!         --tokens 7 1234 9 --dump logits.f32

use std::path::PathBuf;
use std::time::Instant;

use hawking_core::gravity_glm::GravityGlm;

const DEFAULT_MODEL_DIR: &str = "Library/Application Support/Hawking/Models/GLM-5.2/\
    b4734de4facf877f85769a911abafc5283eab3d9/General-R0";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dir: Option<PathBuf> = None;
    let mut tokens: Vec<u32> = vec![7, 1234, 9];
    let mut dump: Option<PathBuf> = None;
    let mut verify_hash = true;
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--dir" => {
                i += 1;
                dir = Some(PathBuf::from(&argv[i]));
            }
            "--tokens" => {
                tokens.clear();
                i += 1;
                while i < argv.len() {
                    match argv[i].parse::<u32>() {
                        Ok(v) => tokens.push(v),
                        Err(_) => break, // next flag; leave it for the outer loop
                    }
                    i += 1;
                }
                continue;
            }
            "--dump" => {
                i += 1;
                dump = Some(PathBuf::from(&argv[i]));
            }
            "--no-verify-hash" => verify_hash = false,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
        i += 1;
    }
    let dir = dir.unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(DEFAULT_MODEL_DIR)
    });
    if !dir.is_dir() {
        return Err(format!("no model directory at {dir:?}").into());
    }

    eprintln!("opening (indexing 282 shard headers, decoding nothing)...");
    let t_open = Instant::now();
    let model = GravityGlm::open_dir(&dir, verify_hash)?;
    eprintln!(
        "opened in {:.1} s | layers={} hidden={} experts={} vocab={}",
        t_open.elapsed().as_secs_f64(),
        model.arch.n_layers,
        model.arch.hidden,
        model.arch.n_routed_experts,
        model.arch.vocab_size
    );

    eprintln!("forward over {} tokens: {:?}", tokens.len(), tokens);
    let t_fwd = Instant::now();
    let (logits, trace) = model.forward(&tokens)?;
    let elapsed = t_fwd.elapsed().as_secs_f64();
    eprintln!(
        "forward done in {elapsed:.1} s ({:.2} s/token, CPU/lazy — no acceleration)",
        elapsed / tokens.len() as f64
    );

    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap()
            .then(a.cmp(&b))
    });
    println!(
        "argmax={} top5={:?}",
        order[0],
        &order[..5.min(order.len())]
    );
    println!("final_topk_selected={:?}", trace.final_topk);
    println!(
        "expert_choices_per_sparse_layer={:?}",
        trace
            .expert_choices
            .iter()
            .map(|v| v.len())
            .collect::<Vec<_>>()
    );

    if let Some(path) = dump {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, &bytes)?;
        eprintln!("wrote {} logits ({} bytes) to {path:?}", logits.len(), bytes.len());
    }
    Ok(())
}
