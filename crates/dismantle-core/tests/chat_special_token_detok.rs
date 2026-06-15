//! Regression for the `/v1/chat/completions` special-token detokenization leak.
//!
//! Streaming generation decodes one token at a time via `Tokenizer::decode_one`,
//! which (before the fix) rendered control tokens like `<|im_end|>` literally.
//! The chat path, whose template emits `<|im_start|>`/`<|im_end|>`, therefore
//! leaked `"<|im_end|>"` / `"|>"` fragments into the response, while
//! `/v1/completions` (no template) stayed clean.
//!
//! Pure-CPU: loads only the GGUF tokenizer — no Metal, no model forward. Skips
//! cleanly when the Qwen weights are absent so it is CI-safe without a fixture.

use dismantle_core::gguf::GgufFile;
use dismantle_core::tokenizer::Tokenizer;
use std::path::Path;

fn qwen_tokenizer() -> Option<Tokenizer> {
    let p = Path::new("../../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf");
    if !p.exists() {
        eprintln!("skipping chat_special_token_detok: weights missing at {p:?}");
        return None;
    }
    let gguf = GgufFile::open(p).expect("open gguf");
    Some(Tokenizer::from_gguf(&gguf).expect("build tokenizer"))
}

/// A control token must never render as visible text in the per-token streaming
/// path. This is the exact mechanism behind the chat-endpoint garbage.
#[test]
fn streamed_decode_suppresses_special_tokens() {
    let Some(tok) = qwen_tokenizer() else { return };

    // The declared eos is always a control token; it must never render as text.
    let eos = tok.eos_id().expect("qwen gguf declares an eos");
    eprintln!(
        "[diag] eos_id={eos} is_eog={} old_decode={:?}",
        tok.is_eog(eos),
        tok.decode_one(eos).unwrap()
    );
    assert!(tok.is_special(eos), "eos must be flagged special");
    assert!(tok.is_eog(eos), "eos must be end-of-generation");
    assert_eq!(
        tok.decode_one(eos).unwrap(),
        "",
        "eos must not leak into output"
    );

    // Qwen2.5 control ids: <|endoftext|>=151643, <|im_start|>=151644, <|im_end|>=151645.
    for id in [151643u32, 151644, 151645] {
        assert!((id as usize) < tok.vocab_size(), "id {id} in vocab");
        assert!(
            tok.is_special(id),
            "control id {id} must be flagged special"
        );
        assert_eq!(
            tok.decode_one(id).unwrap(),
            "",
            "control id {id} must be suppressed in streamed output"
        );
    }

    // The chat turn terminator must terminate generation even when the GGUF
    // sets eos to <|endoftext|> instead of <|im_end|>.
    assert!(tok.is_eog(151645), "<|im_end|> must be end-of-generation");

    // Normal tokens are unaffected — real text still decodes, with no markup leak.
    let ids = tok
        .encode("The capital of France is Paris.", false)
        .expect("encode");
    let text: String = ids.iter().map(|&id| tok.decode_one(id).unwrap()).collect();
    assert!(
        text.contains("Paris") && text.contains("capital"),
        "normal tokens must still decode to text, got: {text:?}"
    );
    assert!(
        !text.contains("<|"),
        "plain text must contain no control markup, got: {text:?}"
    );
}
