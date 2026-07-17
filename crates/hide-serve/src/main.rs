//! hide-serve bin: open the workspace, build the router, serve on localhost.
//!
//! Mirrors `hawking-serve`'s boot-and-serve shape: construct the host
//! (`BackendHost::open_workspace`), wrap it in the [`hide_serve::router`], bind
//! a loopback `TcpListener`, and `axum::serve`. Nothing else: the contract
//! lives in the host, this is pure transport.
//!
//! Usage: `hide-serve [WORKSPACE_ROOT] [--port N]` (defaults: cwd, port 8744).
//! Env overrides: `HIDE_SERVE_ADDR` (full `ip:port`) takes precedence over both.

use std::sync::Arc;

use anyhow::Context;
use hide_backend::BackendHost;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let (workspace_root, addr) = parse_args()?;

    let host = BackendHost::open_workspace(&workspace_root)
        .map_err(|e| anyhow::anyhow!("open workspace {}: {e}", workspace_root.display()))?;
    let app = hide_serve::router(Arc::new(host));

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    tracing::info!(addr = %addr, workspace = %workspace_root.display(), "hide-serve listening");
    axum::serve(listener, app).await.context("axum serve")?;
    Ok(())
}

/// Resolve (workspace_root, bind addr) from argv + env. The FE defaults to
/// `127.0.0.1:8744` (see `app/src/ipc.ts`), so that is the default here.
fn parse_args() -> anyhow::Result<(std::path::PathBuf, String)> {
    let mut args = std::env::args().skip(1);
    let mut workspace_root: Option<std::path::PathBuf> = None;
    let mut port: u16 = 8744;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" | "-p" => {
                let value = args.next().context("--port requires a value")?;
                port = value
                    .parse()
                    .with_context(|| format!("invalid port {value}"))?;
            }
            other if other.starts_with('-') => {
                anyhow::bail!("unknown flag {other}");
            }
            other => {
                if workspace_root.is_some() {
                    anyhow::bail!("unexpected extra argument {other}");
                }
                workspace_root = Some(std::path::PathBuf::from(other));
            }
        }
    }

    let workspace_root = match workspace_root {
        Some(root) => root,
        None => std::env::current_dir().context("resolve current dir")?,
    };
    // Loopback only: this is a localhost transport, never bound to 0.0.0.0.
    let addr = std::env::var("HIDE_SERVE_ADDR").unwrap_or_else(|_| format!("127.0.0.1:{port}"));
    Ok((workspace_root, addr))
}
