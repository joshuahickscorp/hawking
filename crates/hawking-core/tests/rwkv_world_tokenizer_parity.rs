use hawking_core::gguf::GgufFile;
use hawking_core::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::process::Command;
fn f32_gguf_path() -> PathBuf {
    std::env::var("HAWKING_RWKV7_F32_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rwkv_ref/rwkv7-191M-f32.gguf"))
}
fn oracle_bin() -> PathBuf {
    std::env::var("LLAMA_TOKENIZE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/private/tmp/llamacpp/build/bin/llama-tokenize"))
}
fn parse_ids(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if let Ok(v) = s[start..i].parse::<u32>() {
                out.push(v);
            }
        } else {
            i += 1;
        }
    }
    out
}
fn oracle_ids(bin: &PathBuf, gguf: &PathBuf, text: &str) -> Vec<u32> {
    let output = Command::new(bin)
        .arg("-m")
        .arg(gguf)
        .arg("-p")
        .arg(text)
        .arg("--ids")
        .arg("--no-bos")
        .arg("--no-escape")
        .arg("--log-disable")
        .output()
        .expect("spawn llama-tokenize");
    assert!(
        output.status.success(),
        "llama-tokenize failed on {text:?}: status={:?}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    parse_ids(&String::from_utf8_lossy(&output.stdout))
}
fn corpus() -> Vec<&'static str> {
    vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def foo(x): return x * 2  # doubles the input",
        "let v = vec![1, 2, 3]; v.iter().map(|x| x + 1).sum::<i32>();",
        "{\"key\": [1, 2.5, true, null], \"nested\": {\"a\": \"b\"}}",
        "你好世界，再见！",
        "東京は日本の首都です。",
        "안녕하세요 세계",
        "emoji 😀🚀🔥 test 🎉",
        "Mixed: café résumé naïve façade — Москва 北京 \t tab end",
        "  leading and    interior   spaces  ",
        "tabs\tand\t\ttabs\nand\n\nnewlines\r\ncarriage",
        "line1\nline2\nline3\n\n\nlots of blank\n\n\nlines",
        "ALLCAPS lowercase MixedCase 1234567890",
        "punctuation!?.,;:'\"()[]{}<>/\\|@#$%^&*-_=+~`",
        "a backslash \\ and an escape-looking \\n \\t \\x41 sequence",
        "supercalifragilisticexpialidocious antidisestablishmentarianism",
        "🇺🇸🇯🇵 flags and ZWJ 👨‍👩‍👧‍👦 family",
        "the the the the repeated words words words",
        "x",
        " ",
        "\n",
    ]
}
#[test]
fn rwkv_world_tokenizer_parity_vs_llamacpp() {
    let gguf_path = f32_gguf_path();
    if !gguf_path.exists() {
        eprintln!(
            "skipping rwkv_world_tokenizer_parity: no RWKV World GGUF at {gguf_path:?} \
             (set HAWKING_RWKV7_F32_GGUF; the vocab is weight-independent so any World GGUF works)"
        );
        return;
    }
    let bin = oracle_bin();
    let have_oracle = bin.exists();
    if !have_oracle {
        eprintln!(
            "note: oracle binary absent at {bin:?} (set LLAMA_TOKENIZE) — running round-trip only, \
             skipping the bit-exact encode-vs-llama.cpp gate"
        );
    }
    let gguf = GgufFile::open(&gguf_path).expect("open rwkv world gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("rwkv world tokenizer must build from gguf");
    let with_special = tok.encode("Hello", true).expect("encode add_special");
    let without_special = tok.encode("Hello", false).expect("encode no special");
    assert_eq!(
        with_special, without_special,
        "RWKV must not add BOS/EOS: encode(add_special=true) should equal encode(false)"
    );
    let cases = corpus();
    let total = cases.len();
    let mut encode_match = 0usize;
    let mut roundtrip_ok = 0usize;
    for case in &cases {
        let mine = tok.encode(case, false).expect("hawking encode");
        let decoded = tok.decode(&mine, false).expect("hawking decode");
        let rt = decoded == *case;
        if rt {
            roundtrip_ok += 1;
        } else {
        }
        assert!(rt, "round-trip must be exact for {case:?}");
        if have_oracle {
            let oracle = oracle_ids(&bin, &gguf_path, case);
            if mine == oracle {
                encode_match += 1;
            } else {
            }
            assert_eq!(
                mine, oracle,
                "encode ids must be bit-identical to llama.cpp for {case:?}"
            );
        }
    }
    if have_oracle {
        assert_eq!(
            encode_match, total,
            "all cases must match the oracle exactly"
        );
    } else {
        eprintln!(
            "rwkv world tokenizer parity (round-trip only): {roundtrip_ok}/{total} \
             (oracle skipped — no llama-tokenize)"
        );
    }
    assert_eq!(roundtrip_ok, total, "all cases must round-trip exactly");
}
