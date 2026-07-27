//! One module per architecture family. No giant adapter file.

pub mod deepseek;
pub mod gemma;
pub mod glm;
pub mod kimi;
pub mod llama;
pub mod minimax;
pub mod mistral;
pub mod phi;
pub mod qwen;
pub mod state_space;

use crate::abi::FamilyAdapter;

/// All built-in family adapters in stable id order.
pub fn all_families() -> Vec<Box<dyn FamilyAdapter>> {
    vec![
        Box::new(llama::LlamaFamily),
        Box::new(mistral::MistralFamily),
        Box::new(qwen::QwenFamily),
        Box::new(glm::GlmFamily),
        Box::new(deepseek::DeepSeekFamily),
        Box::new(kimi::KimiFamily),
        Box::new(minimax::MiniMaxFamily),
        Box::new(gemma::GemmaFamily),
        Box::new(phi::PhiFamily),
        Box::new(state_space::StateSpaceFamily),
    ]
}
