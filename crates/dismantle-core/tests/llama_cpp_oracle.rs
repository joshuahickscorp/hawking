//! Independent-oracle parity: dismantle's greedy decode agrees with llama.cpp's
//! on the same GGUF.
//!
//! Every other parity gate checks Metal against dismantle's *own* CPU reference
//! port — self-consistency. This one checks against an EXTERNAL implementation
//! (`llama-cli`), which is the only thing that can catch a bug both of our paths
//! share. Greedy on identical weights must agree on the leading token; tiny
//! numerical differences (kernel order, f16 accumulation) eventually diverge, so
//! we assert the first generated word, not the whole continuation.
//!
//! Skips cleanly when `llama-cli` or the model is absent, so it is CI-safe: the
//! gate runs wherever llama.cpp + a model fixture exist (set `DISMANTLE_LLAMA_CLI`
//! to point at a specific binary).
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::process::{Command, Stdio};

const PROMPT: &str = "The capital of France is";
const N: usize = 8;

fn model() -> Option<PathBuf> {
    let p = PathBuf::from("../../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf");
    p.exists().then_some(p)
}

fn llama_cli_bin() -> Option<String> {
    let bin = std::env::var("DISMANTLE_LLAMA_CLI").unwrap_or_else(|_| "llama-cli".into());
    Command::new(&bin)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()
        .filter(|s| s.success())
        .map(|_| bin)
}

/// Extract the model's continuation from `llama-cli`'s noisy stdout (banner,
/// echoed prompt, timing line, interactive `>` prompts). It is the text right
/// after the echoed prompt, up to the `[ Prompt: … t/s ]` timing line.
fn continuation(raw: &str) -> String {
    raw.rsplit_once(PROMPT)
        .map(|(_, after)| after)
        .unwrap_or(raw)
        .split('[')
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// First alphanumeric word, lowercased — the comparison unit (robust to leading
/// whitespace and trailing punctuation).
fn first_word(s: &str) -> String {
    s.trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// llama.cpp greedy continuation, hang-proof. `llama-cli` prints the generated
/// tokens but may not exit on its own (it can drop into an interactive prompt
/// after `-n` tokens), so we read stdout on a side thread, poll for a clean exit
/// up to a wall-clock budget, then kill it regardless and take what it wrote.
fn llama_greedy(bin: &str, model: &PathBuf) -> String {
    use std::io::Read;
    use std::time::{Duration, Instant};

    let mut child = Command::new(bin)
        .args([
            "-m",
            model.to_str().unwrap(),
            "-p",
            PROMPT,
            "-n",
            &N.to_string(),
            "--temp",
            "0",
            "-ngl",
            "99",
            "--log-disable",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn llama-cli");

    let mut pipe = child.stdout.take().expect("llama-cli stdout");
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = pipe.read_to_string(&mut s);
        s
    });

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                break;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(_) => {
                let _ = child.kill();
                break;
            }
        }
    }
    let _ = child.wait();
    reader.join().unwrap_or_default().replace('\0', "")
}

/// dismantle greedy continuation, in-process.
fn dismantle_greedy(model: &PathBuf) -> String {
    let cfg = dismantle_core::EngineConfig::default();
    let mut engine = dismantle_core::model::load_engine(model, cfg).expect("load engine");
    let req = dismantle_core::GenerateRequest {
        prompt: PROMPT.into(),
        max_new_tokens: N,
        sampling: dismantle_core::SamplingParams {
            temperature: 0.0,
            seed: Some(0),
            ..Default::default()
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    };
    let mut text = String::new();
    engine
        .generate(req, &mut |ev| {
            if let dismantle_core::StreamEvent::Token { text: t, .. } = ev {
                text.push_str(&t);
            }
        })
        .expect("generate");
    text
}

#[test]
fn dismantle_greedy_matches_llama_cpp() {
    let (Some(model), Some(bin)) = (model(), llama_cli_bin()) else {
        eprintln!("skipping llama.cpp oracle: needs llama-cli + Qwen-3B on disk");
        return;
    };
    let llama = continuation(&llama_greedy(&bin, &model));
    let dism = dismantle_greedy(&model);
    let (lw, dw) = (first_word(&llama), first_word(&dism));
    eprintln!("[oracle] llama.cpp -> {llama:?}  (first word {lw:?})");
    eprintln!("[oracle] dismantle -> {dism:?}  (first word {dw:?})");
    assert!(!lw.is_empty(), "llama.cpp produced no word: {llama:?}");
    assert!(!dw.is_empty(), "dismantle produced no word: {dism:?}");
    assert_eq!(
        dw, lw,
        "dismantle and llama.cpp greedy disagree on the first token — an \
         independent-oracle divergence a self-consistency gate cannot catch"
    );
}
