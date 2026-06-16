//! RWKV "World" tokenizer parity gate vs llama.cpp.
//!
//! The World tokenizer is a greedy longest-match trie over raw bytes (not
//! BPE/SPM): the GGUF stores each token id as an *escaped* byte string, encode
//! walks the input bytes choosing the longest vocab entry at each position, and
//! decode concatenates the raw bytes. dismantle's `RwkvWorldTokenizer` mirrors
//! llama.cpp's `llm_tokenizer_rwkv` / `llama_unescape_rwkv_token` / `naive_trie`
//! exactly, so for any input the produced ids must be **bit-identical** to
//! `llama-tokenize` run on the same GGUF.
//!
//! ## Two gates (per corpus case)
//!
//! 1. **Encode parity** — shell out to `llama-tokenize --ids --no-bos
//!    --no-escape` on the same GGUF, parse its ids, and assert they are
//!    identical to `Tokenizer::encode`. `--no-escape` is essential: without it
//!    `llama-tokenize` interprets `\n`/`\t`/`\xNN` in the *command-line string*
//!    before tokenizing, which would mangle the corpus and is unrelated to the
//!    vocab-level un-escape under test. `--no-bos` matches RWKV's behavior
//!    (llama.cpp adds no BOS for this vocab; see below).
//! 2. **Round-trip** — `decode(encode(x)) == x` for every (valid-UTF8) case.
//!
//! ## Skips (mirrors `rwkv7_parity.rs`)
//! Needs the f32 reference GGUF (env `DISMANTLE_RWKV7_F32_GGUF`, else the
//! conventional `/tmp/rwkv_ref/rwkv7-191M-f32.gguf`) and the oracle binary (env
//! `LLAMA_TOKENIZE`, else `/private/tmp/llamacpp/build/bin/llama-tokenize`). The
//! tokenizer vocab is weight-independent and identical across all RWKV World
//! models, so any World GGUF works. Both are absent in CI, so this gate
//! `eprintln!`s and returns there, exactly like the other rwkv parity gates.
//!
//! ## BOS
//! RWKV adds **no BOS** by default. Although the GGUF declares
//! `tokenizer.ggml.bos_token_id = 1`, llama.cpp's RWKV tokenize path never
//! auto-inserts it (verified: `llama-tokenize` output is identical with and
//! without `--no-bos`). dismantle's `encode` likewise emits no BOS.

use dismantle_core::gguf::GgufFile;
use dismantle_core::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::process::Command;

fn f32_gguf_path() -> PathBuf {
    std::env::var("DISMANTLE_RWKV7_F32_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rwkv_ref/rwkv7-191M-f32.gguf"))
}

fn oracle_bin() -> PathBuf {
    std::env::var("LLAMA_TOKENIZE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/private/tmp/llamacpp/build/bin/llama-tokenize"))
}

/// Parse every signed integer out of `llama-tokenize --ids` output. The binary
/// prints a single bracketed list `[1097, 39934]`; scanning for integer runs is
/// robust to that and to the `id -> 'piece'` line form.
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

/// Run the oracle on `text` with literal (un-escaped) input and no BOS.
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

/// Diverse corpus: plain ASCII, source code with symbols, CJK, emoji /
/// multibyte unicode, runs of whitespace and newlines, accented latin, and
/// mixed scripts. Every case is valid UTF-8 so round-trip must be exact.
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
             (set DISMANTLE_RWKV7_F32_GGUF; the vocab is weight-independent so any World GGUF works)"
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

    // Sanity: RWKV adds no BOS — confirm `encode` is identical with and without
    // the add_special flag (and, when the oracle is present, that the oracle is
    // BOS-free too: the comparison below uses --no-bos and would diverge by a
    // leading id if dismantle prepended one).
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
        let mine = tok.encode(case, false).expect("dismantle encode");

        // Round-trip: decode(encode(x)) == x for valid UTF-8.
        let decoded = tok.decode(&mine, false).expect("dismantle decode");
        let rt = decoded == *case;
        if rt {
            roundtrip_ok += 1;
        } else {
            eprintln!("  ROUND-TRIP MISMATCH {case:?}\n    decoded={decoded:?}");
        }
        assert!(rt, "round-trip must be exact for {case:?}");

        if have_oracle {
            let oracle = oracle_ids(&bin, &gguf_path, case);
            if mine == oracle {
                encode_match += 1;
            } else {
                eprintln!("  ENCODE MISMATCH {case:?}\n    mine  ={mine:?}\n    oracle={oracle:?}");
            }
            assert_eq!(
                mine, oracle,
                "encode ids must be bit-identical to llama.cpp for {case:?}"
            );
        }
    }

    if have_oracle {
        eprintln!(
            "rwkv world tokenizer parity: encode ids match llama.cpp {encode_match}/{total}; \
             round-trip {roundtrip_ok}/{total}"
        );
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
