//! Uniform-Q4 Q30 kernel levers + dispatch folds.
//!
//! Loads the sealed group-64 candidate once, then for each mode:
//!   1. France 8-token identity vs serial and vs simdgroup8-nofold
//!   2. first-token logit/hidden delta vs those references
//!   3. three untraced coherence probes (Paris / 4 / Jupiter)
//!   4. optional >=32-decode-token generate (off by default; WAVE-0
//!      does not claim TPS — set HAWKING_QWEN30_Q4_GATES_SKIP_TPS=0)
//!
//! HAWKING_TCB_TRACE must stay off — it changes tokens.
//!
//! Modes: `HAWKING_QWEN30_Q4_GATES_MODES=serial,simdgroup8,simdgroup8-paired,...`
//! Fold suffixes: `-paired`, `-qk`, `-lmhead`, `-folds` (all three).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("uniform-q4 tps gates require macOS Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen30_complete_runtime::{
        Qwen30CompleteNativeRuntime, Qwen30CompleteRuntimeOptions, Qwen30GateUpSwiGluKernel,
        Qwen30NativeGeneration, Qwen30PackedMatvecKernel,
    };
    use hawking_core::model::qwen_complete_binary::Qwen30UniformQ4Admission;
    use serde_json::json;
    use std::env;
    use std::path::PathBuf;
    use std::time::Instant;

    const MANIFEST: &str = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/uniform-q4-group64-v1/QWEN30_UNIFORM_Q4_GROUP64_V1_COMPLETE_BINARY_GRAVITY_CANDIDATE.json";
    const REVALIDATION: &str = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity/QWEN30_CURRENT_SOURCE_SHARD_REVALIDATION.json";
    const TERMINAL: &str = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/uniform-q4-group64-v1/QWEN30_UNIFORM_Q4_GROUP64_V1_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json";

    const SEALED_FRANCE_IDS: [u32; 8] = [785, 6722, 315, 9625, 374, 12095, 13, 151645];

    fn fail(msg: impl AsRef<str>) -> ! {
        eprintln!("uniform-q4 tps gates refused: {}", msg.as_ref());
        std::process::exit(2);
    }

    fn summarize(gen: &Qwen30NativeGeneration) -> serde_json::Value {
        let step_us: Vec<u64> = gen
            .steps
            .iter()
            .map(|s| s.elapsed.as_micros() as u64)
            .collect();
        let n = gen.steps.len();
        let sum_us: u64 = step_us.iter().copied().sum();
        let mean_us = if n == 0 {
            0.0
        } else {
            sum_us as f64 / n as f64
        };
        let tps = if mean_us > 0.0 { 1.0e6 / mean_us } else { 0.0 };
        let cb: Vec<usize> = gen.steps.iter().map(|s| s.command_buffers).collect();
        let disp: Vec<usize> = gen.steps.iter().map(|s| s.metal_dispatches).collect();
        let routes: Vec<usize> = gen.steps.iter().map(|s| s.host_route_id_readbacks).collect();
        json!({
            "completion_text": gen.completion_text,
            "completion_token_ids": gen.completion_token_ids,
            "ended_on_eog": gen.ended_on_eog,
            "n_decode_steps": n,
            "sum_step_us": sum_us,
            "mean_step_us": mean_us,
            "tps_sum_over_n": tps,
            "command_buffers_per_step": cb,
            "metal_dispatches_per_step": disp,
            "host_route_id_readbacks_per_step": routes,
            "first_cb": cb.first().copied().unwrap_or(0),
            "first_disp": disp.first().copied().unwrap_or(0),
        })
    }

    fn max_abs_delta(a: &[f32], b: &[f32]) -> (f32, usize, f64) {
        let n = a.len().min(b.len());
        let mut max_abs = 0.0f32;
        let mut argmax = 0usize;
        let mut l2 = 0.0f64;
        for i in 0..n {
            let d = (a[i] - b[i]).abs();
            if d > max_abs {
                max_abs = d;
                argmax = i;
            }
            l2 += (d as f64) * (d as f64);
        }
        (max_abs, argmax, (l2 / n.max(1) as f64).sqrt())
    }

    fn ids_diverged(a: &[u32], b: &[u32]) -> usize {
        let n = a.len().max(b.len());
        (0..n)
            .filter(|&i| a.get(i) != b.get(i))
            .count()
    }

    /// Returns (kernel, folds_master, paired, qk, lmhead).
    fn parse_gate_mode(mode: &str) -> (&'static str, bool, bool, bool, bool) {
        match mode {
            "default" => ("default", false, false, false, false),
            "serial" => ("serial", false, false, false, false),
            "fused" | "serial-fused" | "serial_fused" => ("fused", false, false, false, false),
            "rowblock" | "rowblock4" => ("rowblock", false, false, false, false),
            "simdgroup" | "simd" => ("simdgroup", false, false, false, false),
            "simdgroup4" | "simd4" => ("simdgroup4", false, false, false, false),
            "simdgroup8" | "simd8" => ("simdgroup8", false, false, false, false),
            "simd64" | "x64" => ("simd64", false, false, false, false),
            "simdgroup8-paired" | "simd8-paired" => ("simdgroup8", true, true, false, false),
            "simdgroup8-qk" | "simd8-qk" => ("simdgroup8", true, false, true, false),
            "simdgroup8-lmhead" | "simd8-lmhead" => ("simdgroup8", true, false, false, true),
            "simdgroup8-folds" | "simd8-folds" | "folds" => ("simdgroup8", true, true, true, true),
            "serial-paired" => ("serial", true, true, false, false),
            "serial-qk" => ("serial", true, false, true, false),
            "serial-lmhead" => ("serial", true, false, false, true),
            "serial-folds" => ("serial", true, true, true, true),
            other => {
                eprintln!("uniform-q4 tps gates refused: unknown mode {other}");
                std::process::exit(2);
            }
        }
    }

    fn apply_fold_env(folds: bool, paired: bool, qk: bool, lmhead: bool) {
        env::set_var("HAWKING_QWEN30_Q4_FOLDS", if folds { "1" } else { "0" });
        env::set_var("HAWKING_QWEN30_Q4_PAIRED", if paired { "1" } else { "0" });
        env::set_var("HAWKING_QWEN30_Q4_FUSED_QK_ROPE", if qk { "1" } else { "0" });
        env::set_var("HAWKING_QWEN30_Q4_FUSED_LM_HEAD", if lmhead { "1" } else { "0" });
    }

    pub fn run() {
        env::remove_var("HAWKING_TCB_TRACE");
        env::set_var("HAWKING_QWEN30_DEVICE_EXPERT_TABLE", "1");
        env::set_var("HAWKING_QWEN30_DEVICE_RESIDENT_AR", "1");

        let skip_tps = env::var("HAWKING_QWEN30_Q4_GATES_SKIP_TPS")
            .map(|s| {
                let t = s.trim();
                !(t == "0" || t.eq_ignore_ascii_case("false") || t.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(true);

        let modes: Vec<String> = env::var("HAWKING_QWEN30_Q4_GATES_MODES")
            .ok()
            .map(|s| {
                s.split(',')
                    .map(|p| p.trim().to_ascii_lowercase())
                    .filter(|p| !p.is_empty())
                    .collect()
            })
            .unwrap_or_else(|| {
                vec![
                    "serial".into(),
                    "simdgroup8".into(),
                    "simdgroup8-paired".into(),
                    "simdgroup8-qk".into(),
                    "simdgroup8-lmhead".into(),
                    "simdgroup8-folds".into(),
                ]
            });

        let admission = Qwen30UniformQ4Admission {
            expected_manifest_seal_sha256:
                "6d8468dd88890ba3d980249b464c9b19b87d6c186d2c2a1c1021410a997a89b4".into(),
            expected_source_audit_seal_sha256:
                "00ed3e495416c2cbafbcdb7800528e15f243b1a13f5f4af13240109c8fc69f7b".into(),
            expected_source_revision: "b2cff646eb4bb1d68355c01b18ae02e7cf42d120".into(),
            expected_revalidation_path: PathBuf::from(REVALIDATION),
            expected_revalidation_seal_sha256:
                "ac7208e11c31bbd035bd87fd62a80020b9d1d05970867576f4649f6bebe68123".into(),
            expected_terminal_path: PathBuf::from(TERMINAL),
            expected_terminal_seal_sha256:
                "6b54d9f54887d31cd657f9de4282f72d16c0824df2666733e8e4bf687d4906a8".into(),
        };
        let options = Qwen30CompleteRuntimeOptions {
            max_seq_len: 256,
            trace_dispatch: false,
            packed_matvec_kernel: Qwen30PackedMatvecKernel::SerialControl,
            gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel::ThreeDispatchControl,
        };

        eprintln!("[gates] admitting uniform-q4-group64-v1 …");
        let load_t = Instant::now();
        let mut runtime = Qwen30CompleteNativeRuntime::load_uniform_q4(MANIFEST, &admission, options)
            .unwrap_or_else(|e| fail(e.to_string()));
        eprintln!(
            "[gates] admission+construct {:.1}s",
            load_t.elapsed().as_secs_f64()
        );

        let identity_prompt = "What is the capital of France?";
        let mut serial_ids: Option<Vec<u32>> = None;
        let mut serial_logits: Option<Vec<f32>> = None;
        let mut serial_hidden: Option<Vec<f32>> = None;
        let mut simd8_ids: Option<Vec<u32>> = None;
        let mut simd8_logits: Option<Vec<f32>> = None;
        let mut simd8_hidden: Option<Vec<f32>> = None;
        let mut mode_reports = Vec::new();

        for mode in &modes {
            let (kernel, folds, paired, qk, lmhead) = parse_gate_mode(mode);
            apply_fold_env(folds, paired, qk, lmhead);
            if kernel != "default" {
                env::set_var("HAWKING_QWEN30_Q4_KERNEL", kernel);
            } else {
                env::remove_var("HAWKING_QWEN30_Q4_KERNEL");
            }
            eprintln!(
                "[gates] ===== mode={mode} kernel={kernel} folds={folds} paired={paired} qk={qk} lmhead={lmhead} ====="
            );

            eprintln!("[gates] {kernel} first-token numeric …");
            let first = runtime
                .generate_source_user_chat_greedy(identity_prompt, 1)
                .unwrap_or_else(|e| fail(e.to_string()));
            let logits = runtime
                .diagnostic_final_logits_f32()
                .unwrap_or_else(|e| fail(e.to_string()));
            let hidden = runtime
                .diagnostic_residual_hidden_f32()
                .unwrap_or_else(|e| fail(e.to_string()));
            if kernel == "serial" && !folds {
                serial_logits = Some(logits.clone());
                serial_hidden = Some(hidden.clone());
            }
            if kernel == "simdgroup8" && !folds {
                simd8_logits = Some(logits.clone());
                simd8_hidden = Some(hidden.clone());
            }
            let (logit_max, logit_arg, logit_rms) = match &serial_logits {
                Some(s) => {
                    let (m, a, r) = max_abs_delta(s, &logits);
                    (m, a, r)
                }
                None => (0.0, 0, 0.0),
            };
            let (hidden_max, hidden_arg, hidden_rms) = match &serial_hidden {
                Some(s) => {
                    let (m, a, r) = max_abs_delta(s, &hidden);
                    (m, a, r)
                }
                None => (0.0, 0, 0.0),
            };
            let (logit_max_s8, logit_arg_s8, logit_rms_s8) = match &simd8_logits {
                Some(s) => {
                    let (m, a, r) = max_abs_delta(s, &logits);
                    (m, a, r)
                }
                None => (0.0, 0, 0.0),
            };
            let (hidden_max_s8, hidden_arg_s8, hidden_rms_s8) = match &simd8_hidden {
                Some(s) => {
                    let (m, a, r) = max_abs_delta(s, &hidden);
                    (m, a, r)
                }
                None => (0.0, 0, 0.0),
            };

            eprintln!("[gates] {mode} France 8-token …");
            let france = runtime
                .generate_source_user_chat_greedy(identity_prompt, 8)
                .unwrap_or_else(|e| fail(e.to_string()));
            let france_sum = summarize(&france);
            if kernel == "serial" && !folds {
                serial_ids = Some(france.completion_token_ids.clone());
            }
            if kernel == "simdgroup8" && !folds {
                simd8_ids = Some(france.completion_token_ids.clone());
            }
            let ref_ids = serial_ids
                .as_deref()
                .unwrap_or(SEALED_FRANCE_IDS.as_slice());
            let n_div = ids_diverged(&france.completion_token_ids, ref_ids);
            let ids_match = n_div == 0
                && france.completion_token_ids.len() == ref_ids.len();
            let n_div_s8 = simd8_ids
                .as_deref()
                .map(|s| ids_diverged(&france.completion_token_ids, s))
                .unwrap_or(0);
            let ids_match_s8 = simd8_ids
                .as_deref()
                .map(|s| {
                    n_div_s8 == 0 && france.completion_token_ids.len() == s.len()
                })
                .unwrap_or(kernel == "simdgroup8" && !folds);
            let first_cb = france.steps.first().map(|s| s.command_buffers).unwrap_or(0);
            let first_disp = france.steps.first().map(|s| s.metal_dispatches).unwrap_or(0);
            eprintln!(
                "[gates] {mode} text={:?} ids={:?} vs_serial={ids_match} div={n_div}/{} vs_simd8={ids_match_s8} div_s8={n_div_s8} cb={} disp={}",
                france.completion_text,
                france.completion_token_ids,
                ref_ids.len(),
                first_cb,
                first_disp,
            );

            let mut probe_out = vec![json!({
                "name": "france",
                "prompt": identity_prompt,
                "needle": "Paris",
                "hit": france.completion_text.contains("Paris"),
                "completion_text": france.completion_text,
                "completion_token_ids": france.completion_token_ids,
            })];
            for (name, prompt, needle) in [
                ("two_plus_two", "What is 2 + 2?", "4"),
                (
                    "planet",
                    "What is the largest planet in the solar system?",
                    "Jupiter",
                ),
            ] {
                eprintln!("[gates] {kernel} coherence probe {name} …");
                let gen = runtime
                    .generate_source_user_chat_greedy(prompt, 16)
                    .unwrap_or_else(|e| fail(e.to_string()));
                let hit = gen.completion_text.contains(needle);
                eprintln!("[gates] {kernel} {name} hit={hit} text={:?}", gen.completion_text);
                probe_out.push(json!({
                    "name": name,
                    "prompt": prompt,
                    "needle": needle,
                    "hit": hit,
                    "completion_text": gen.completion_text,
                    "completion_token_ids": gen.completion_token_ids,
                }));
            }

            let tps_device_resident = if skip_tps {
                eprintln!("[gates] {mode} skip 32-token generate (HAWKING_QWEN30_Q4_GATES_SKIP_TPS)");
                json!({ "skipped": true })
            } else {
                eprintln!("[gates] {mode} 32-token generate …");
                let tps_wall = Instant::now();
                let tps_gen = runtime
                    .generate_source_user_chat_greedy(
                        "Write a numbered list of forty world capitals, one per line.",
                        32,
                    )
                    .unwrap_or_else(|e| fail(e.to_string()));
                let tps_wall_s = tps_wall.elapsed().as_secs_f64();
                let tps_sum = summarize(&tps_gen);
                let decode_n = tps_gen.steps.len().max(1);
                let wall_tps = decode_n as f64 / tps_wall_s;
                eprintln!(
                    "[gates] {mode} generate steps={} sum/n={:.3} wall/n={:.3} text_prefix={:?}",
                    decode_n,
                    tps_sum["tps_sum_over_n"].as_f64().unwrap_or(0.0),
                    wall_tps,
                    tps_gen.completion_text.chars().take(80).collect::<String>(),
                );
                json!({
                    "summarize": tps_sum,
                    "wall_s": tps_wall_s,
                    "wall_tps_decode_steps_over_wall": wall_tps,
                })
            };

            mode_reports.push(json!({
                "mode": mode,
                "kernel": kernel,
                "folds": { "master": folds, "paired": paired, "qk_rope": qk, "lm_head": lmhead },
                "identity": {
                    "ids_match_serial_or_sealed": ids_match,
                    "ids_diverged_vs_serial": n_div,
                    "ids_match_simdgroup8": ids_match_s8,
                    "ids_diverged_vs_simdgroup8": n_div_s8,
                    "ids_compared_n": ref_ids.len(),
                    "first_token": first.completion_token_ids,
                    "first_cb": first_cb,
                    "first_disp": first_disp,
                    "france": france_sum,
                },
                "numeric": {
                    "first_token_id": first.completion_token_ids,
                    "logit_max_abs_vs_serial": logit_max,
                    "logit_argmax_vs_serial": logit_arg,
                    "logit_rms_vs_serial": logit_rms,
                    "hidden_max_abs_vs_serial": hidden_max,
                    "hidden_argmax_vs_serial": hidden_arg,
                    "hidden_rms_vs_serial": hidden_rms,
                    "logit_max_abs_vs_simdgroup8": logit_max_s8,
                    "logit_argmax_vs_simdgroup8": logit_arg_s8,
                    "logit_rms_vs_simdgroup8": logit_rms_s8,
                    "hidden_max_abs_vs_simdgroup8": hidden_max_s8,
                    "hidden_argmax_vs_simdgroup8": hidden_arg_s8,
                    "hidden_rms_vs_simdgroup8": hidden_rms_s8,
                    "logit_n": logits.len(),
                    "hidden_n": hidden.len(),
                },
                "coherence": probe_out,
                "tps_device_resident": tps_device_resident,
            }));
        }

        let report = json!({
            "schema": "hawking.uniform_q4.q4_kernel_levers.gates.v1",
            "tcb_trace": env::var("HAWKING_TCB_TRACE").ok(),
            "modes": modes,
            "sealed_france_ids": SEALED_FRANCE_IDS,
            "results": mode_reports,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}
