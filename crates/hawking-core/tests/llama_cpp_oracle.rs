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
    let bin = std::env::var("HAWKING_LLAMA_CLI").unwrap_or_else(|_| "llama-cli".into());
    Command::new(&bin)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()
        .filter(|s| s.success())
        .map(|_| bin)
}
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
fn first_word(s: &str) -> String {
    s.trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}
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
fn hawking_greedy(model: &PathBuf) -> String {
    let cfg = hawking_core::EngineConfig::default();
    let mut engine = hawking_core::model::load_engine(model, cfg).expect("load engine");
    let req = hawking_core::GenerateRequest {
        prompt: PROMPT.into(),
        max_new_tokens: N,
        sampling: hawking_core::SamplingParams {
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
            if let hawking_core::StreamEvent::Token { text: t, .. } = ev {
                text.push_str(&t);
            }
        })
        .expect("generate");
    text
}
#[test]
fn hawking_greedy_matches_llama_cpp() {
    let (Some(model), Some(bin)) = (model(), llama_cli_bin()) else {
        eprintln!("skipping llama.cpp oracle: needs llama-cli + Qwen-3B on disk");
        return;
    };
    let llama = continuation(&llama_greedy(&bin, &model));
    let dism = hawking_greedy(&model);
    let (lw, dw) = (first_word(&llama), first_word(&dism));
    assert!(!lw.is_empty(), "llama.cpp produced no word: {llama:?}");
    assert!(!dw.is_empty(), "hawking produced no word: {dism:?}");
    assert_eq!(
        dw, lw,
        "hawking and llama.cpp greedy disagree on the first token — an \
         independent-oracle divergence a self-consistency gate cannot catch"
    );
}
