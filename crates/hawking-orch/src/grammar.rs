use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarRequest {
    pub name: String,
    pub schema_json: String,
    pub tokenizer_signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledGrammar {
    pub name: String,
    pub grammar_hash: String,
    pub tokenizer_signature: String,
    pub mask_cache_key: String,
}

pub trait GrammarCompiler: Send + Sync {
    fn compile(&self, request: GrammarRequest) -> hide_core::Result<CompiledGrammar>;
}

#[derive(Default)]
pub struct StubGrammarCompiler;

impl GrammarCompiler for StubGrammarCompiler {
    fn compile(&self, request: GrammarRequest) -> hide_core::Result<CompiledGrammar> {
        Ok(CompiledGrammar {
            grammar_hash: format!("stub:{}:{}", request.name, request.schema_json.len()),
            mask_cache_key: format!("{}:{}", request.name, request.tokenizer_signature),
            name: request.name,
            tokenizer_signature: request.tokenizer_signature,
        })
    }
}
