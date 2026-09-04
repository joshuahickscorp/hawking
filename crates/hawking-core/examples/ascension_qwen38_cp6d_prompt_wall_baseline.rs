//! CP6d — the prompt wall itself, which is what G019 actually asks for.
//!
//! CP3 through CP6c measured ORGANS. G019's acceptance is a complete prompt wall
//! against the retained sequential baseline, and that baseline had never been
//! measured in this campaign — every ratio so far divides one organ by another.
//! This establishes the number CP6 stage 1 has to beat, on the current build.
//!
//! It also produces the directive's own headline metric: fresh-compute prompt
//! tok/s, which the standing record puts at ~36 and the milestone ladder wants
//! at 50, 75, 100, 150, 200+.
//!
//! Scaling matters and is measured rather than extrapolated: the runtime's own
//! `prefill_profile` module records that prefill was SUPERLINEAR (2500 tokens in
//! 170.7 s against 3099 in 385.4 s) and lists five mechanisms with different
//! shapes against position. A single length cannot tell them apart, so this
//! sweeps lengths and reports per-position cost at each.
//!
//! The bridge to the organ work is stated as ARITHMETIC, not measurement: given
//! the measured gate_up share of a step and its measured 1.903x, what would the
//! prompt wall be? That names exactly what CP6 stage 1 must deliver and makes it
//! falsifiable, without pretending the substitution has been performed.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh cp6d \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_cp6d_prompt_wall_baseline \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --tokenizer ~/noetic/NOETIC_PARENT_A/tokenizer.json \
//!   --out receipts/runtime/CP6D_PROMPT_WALL.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen38_hybrid_decode::{
        generate_greedy, load_qwen38_tokenizer, Qwen38HybridDecodeSession, Qwen38MlpFusion,
    };
    use serde_json::json;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::process;

    /// Measured share of a step's GPU ns, from
    /// receipts/headless/_ORGAN_BANDWIDTH_raw.json isolated_organs.
    const GATE_UP_SHARE: f64 = 0.359;
    /// Measured on real weights across 64 layers: CP6a 1.890x, CP6b 1.890x,
    /// CP6c 1.903x on separate lanes.
    const GATE_UP_BATCHED_SPEEDUP: f64 = 1.903;

    fn fail(m: impl std::fmt::Display) -> ! {
        eprintln!("cp6d: {m}");
        process::exit(2);
    }

    pub fn run() {
        let home = env::var("HOME").unwrap_or_default();
        let mut artifact_root = PathBuf::from(&home).join("noetic").join("NOETIC_PARENT_A");
        let mut tokenizer_path = artifact_root.join("tokenizer.json");
        let mut out: Option<PathBuf> = None;
        let mut lengths = vec![128usize, 256, 512, 1024];
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--artifact-root" => {
                    artifact_root = PathBuf::from(it.next().unwrap_or_default());
                    tokenizer_path = artifact_root.join("tokenizer.json");
                }
                "--tokenizer" => tokenizer_path = PathBuf::from(it.next().unwrap_or_default()),
                "--out" => out = it.next().map(PathBuf::from),
                "--lengths" => {
                    lengths = it
                        .next()
                        .unwrap_or_default()
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect()
                }
                o => fail(format!("unknown flag {o}")),
            }
        }

        let tok = load_qwen38_tokenizer(&tokenizer_path).unwrap_or_else(|e| fail(e));
        let seed = "Explain, in ordinary prose and at length, how a compiler turns a for-loop \
into basic blocks and then into machine code, starting from the abstract syntax tree and \
ending at register allocation. ";
        let seed_ids = tok
            .encode(seed, false)
            .unwrap_or_else(|e| fail(format!("encode: {e}")));
        if seed_ids.is_empty() {
            fail("tokenizer produced no ids");
        }

        let max_len = *lengths.iter().max().unwrap_or(&1024);
        let mut session = Qwen38HybridDecodeSession::open(&artifact_root, max_len + 64)
            .unwrap_or_else(|e| fail(e));
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);

        let mut rows = Vec::new();
        for &n in &lengths {
            // Real token ids, repeated to length. Content does not change the
            // physics of a dense prefill; length does, and length is the variable.
            let prompt: Vec<u32> = seed_ids.iter().cloned().cycle().take(n).collect();
            // ONE new token: this measures the PROMPT wall, and decode would
            // otherwise dominate the total and hide it.
            let r = generate_greedy(&mut session, &prompt, 1).unwrap_or_else(|e| fail(e));

            let prefill_ms = r.prefill_wall_ns as f64 / 1e6;
            let per_pos_ns = r.prefill_wall_ns as f64 / n as f64;
            let tok_s = n as f64 / (r.prefill_wall_ns as f64 / 1e9);
            let gpu: u64 = r.gpu_ns.iter().take(n).filter_map(|v| *v).sum();
            let disp: u64 = r.dispatches.iter().take(n).sum();
            let host_ns = r.prefill_wall_ns.saturating_sub(gpu);

            // ARITHMETIC, not measurement: substitute the measured gate_up share
            // at its measured batched speedup and see what the wall becomes.
            let projected = 1.0 - GATE_UP_SHARE + GATE_UP_SHARE / GATE_UP_BATCHED_SPEEDUP;

            eprintln!(
                "{n:>5} tok  prefill {prefill_ms:>9.1} ms  {per_pos_ns:>10.0} ns/pos  \
{tok_s:>7.2} tok/s  gpu {:>5.1}%  {:>6.1} disp/pos",
                100.0 * gpu as f64 / r.prefill_wall_ns as f64,
                disp as f64 / n as f64
            );
            rows.push(json!({
                "prompt_tokens": n,
                "prefill_wall_ns": r.prefill_wall_ns,
                "prefill_wall_ms": prefill_ms,
                "ns_per_position": per_pos_ns,
                "fresh_compute_prompt_tok_s": tok_s,
                "gpu_ns_sum_over_prefill_steps": gpu,
                "gpu_fraction_of_prefill_wall": gpu as f64 / r.prefill_wall_ns as f64,
                "host_ns_outside_gpu": host_ns,
                "dispatches_total": disp,
                "dispatches_per_position": disp as f64 / n as f64,
                "first_step_wall_ns": r.first_step_wall_ns,
                "resident_weight_bytes": r.resident_weight_bytes,
                "workspace_resident_bytes": r.workspace_resident_bytes,
                "fallbacks": r.fallbacks,
                "dense_w_materialized": r.dense_w_materialized,
                "projected_wall_if_gate_up_batched_ARITHMETIC_NOT_MEASURED": {
                    "multiplier": projected,
                    "prefill_wall_ms": prefill_ms * projected,
                    "fresh_compute_prompt_tok_s": tok_s / projected,
                },
            }));
        }

        let doc = json!({
            "schema": "hawking.cp6d.prompt_wall_baseline.v1",
            "what_this_is": "the complete PROMPT WALL on the current build -- the sequential \
baseline G019 requires a chunked path to beat. Every prior CP receipt divides one organ by \
another; none of them measured this.",
            "one_new_token_on_purpose": "decode would otherwise dominate the total and hide the \
prompt wall, which is the quantity under test.",
            "artifact_root": artifact_root.to_string_lossy(),
            "gpu_timestamp_authority": "per-step gpu_ns from completed command buffers, summed over \
prefill steps only; prefill_wall_ns is host wall around the same steps",
            "projection_basis": {
                "gate_up_share_of_step_gpu_ns": GATE_UP_SHARE,
                "source": "receipts/headless/_ORGAN_BANDWIDTH_raw.json isolated_organs",
                "gate_up_batched_speedup": GATE_UP_BATCHED_SPEEDUP,
                "speedup_source": "CP6a 1.890x, CP6b 1.890x, CP6c 1.903x on real weights, 64 layers, separate lanes",
                "caveat": "ARITHMETIC over a measured organ share and a measured organ speedup. It \
is NOT a measured prompt wall, and this campaign has already retracted two projections of that \
shape. It states what CP6 stage 1 must deliver, so that stage can falsify it."
            },
            "rows": rows,
        });
        match out {
            Some(p) => {
                if let Some(d) = p.parent() { fs::create_dir_all(d).ok(); }
                fs::write(&p, serde_json::to_string_pretty(&doc).unwrap() + "\n")
                    .unwrap_or_else(|e| fail(e));
                eprintln!("wrote {}", p.display());
            }
            None => println!("{}", serde_json::to_string_pretty(&doc).unwrap()),
        }
    }
}
