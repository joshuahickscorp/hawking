//! GGUF v3 reader. mmap-backed, expert-tensor-aware tensor index.
//!
//! Tested against canonical TheBloke and official GGUF exports.
//! Non-canonical exporter layouts are out of scope, not bugs.

pub mod reader;

pub use reader::{GgmlType, GgufFile, MetaValue, TensorInfo};
