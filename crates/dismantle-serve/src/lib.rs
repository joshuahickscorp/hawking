//! dismantle-serve: OpenAI-compatible HTTP server.
//!
//! Drives a `dismantle_core::Engine` through axum. Continuous
//! batching lives in [`batch`]; the HTTP surface in [`http`].

pub mod batch;
pub mod http;

use anyhow::Result;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub weights: PathBuf,
    pub addr: SocketAddr,
    pub max_batch_size: usize,
    pub speculate: Option<String>,
    pub verify_window: usize,
    pub kernel_profile: Option<PathBuf>,
    pub prefill_cache_dir: Option<PathBuf>,
    pub max_routed_expert_ram_mb: Option<usize>,
}

pub async fn run(opts: ServeOptions) -> Result<()> {
    use dismantle_core::{profile::KernelProfile, EngineConfig, SpeculateMode};

    let speculate_mode = SpeculateMode::from_cli(opts.speculate.as_deref(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let kernel_profile = match opts.kernel_profile.as_ref() {
        Some(path) => Some(KernelProfile::load(path)?),
        None => None,
    };
    let cfg = EngineConfig {
        max_seq_len: 4096,
        max_batch_size: opts.max_batch_size,
        speculate: speculate_mode != SpeculateMode::Off,
        speculate_mode,
        verify_window: opts.verify_window,
        prefill_cache_dir: opts.prefill_cache_dir,
        kernel_profile,
        trace_dispatch: false,
        activation_dtype: Default::default(),
        max_routed_expert_ram_mb: opts.max_routed_expert_ram_mb,
        ..Default::default()
    };

    let engine = dismantle_core::model::load_engine(&opts.weights, cfg)
        .map_err(|e| anyhow::anyhow!("load engine: {e}"))?;
    let model_arch = engine.model_arch().to_string();
    let state = http::AppState {
        engine: Arc::new(parking_lot::Mutex::new(engine)),
        model_arch,
    };
    let app = http::router(state);
    tracing::info!(addr = %opts.addr, "dismantle-serve listening");
    let listener = tokio::net::TcpListener::bind(opts.addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
