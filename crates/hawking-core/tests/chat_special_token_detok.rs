use hawking_core::gguf::GgufFile;
use hawking_core::tokenizer::Tokenizer;
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
#[test]
fn streamed_decode_suppresses_special_tokens() {
    let Some(tok) = qwen_tokenizer() else { return };
    let eos = tok.eos_id().expect("qwen gguf declares an eos");
    assert!(tok.is_special(eos), "eos must be flagged special");
    assert!(tok.is_eog(eos), "eos must be end-of-generation");
    assert_eq!(
        tok.decode_one(eos).unwrap(),
        "",
        "eos must not leak into output"
    );
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
    assert!(tok.is_eog(151645), "<|im_end|> must be end-of-generation");
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
#[test]
fn chat_template_encodes_control_tokens_atomically() {
    let Some(tok) = qwen_tokenizer() else { return };
    let template = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
    let ids = tok.encode(template, false).expect("encode template");
    assert!(
        ids.contains(&151644),
        "<|im_start|> must encode as atomic id 151644, got {ids:?}"
    );
    assert!(
        ids.contains(&151645),
        "<|im_end|> must encode as atomic id 151645, got {ids:?}"
    );
    assert!(
        ids.len() < 16,
        "template shattered into {} tokens (control tokens not atomic): {ids:?}",
        ids.len()
    );
}
