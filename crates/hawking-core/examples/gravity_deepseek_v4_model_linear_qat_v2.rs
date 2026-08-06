//! QAT-integrated successor to the sealed source-linear V1 component checkpoint.
//!
//! This wrapper intentionally reuses the audited reader, CPU-oracle binding,
//! source tensor contract, and receipt sealing helpers from the V1 component
//! probe while invoking only its additive V2 QAT-chain entry point.  V1's
//! receipt and original CLI behavior remain unchanged.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_model_linear_qat_v2 -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --cpu-oracle /absolute/path/to/DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json \
//!   --predecessor-v1 /absolute/path/to/DSV4F_MODEL_LINEAR_METAL_COMPONENT_PARITY-v1.json \
//!   --out /absolute/path/to/DSV4F_MODEL_LINEAR_METAL_COMPONENT_PARITY-v2.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("gravity_deepseek_v4_model_linear_qat_v2 requires macOS Metal")
            .into(),
    )
}

#[cfg(target_os = "macos")]
#[path = "gravity_deepseek_v4_model_linear_metal_checkpoint.rs"]
mod checkpoint_v1;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    checkpoint_v1::macos::run_qat_v2()
}
