//! hide-acp-server: the ACP agent entrypoint.
//!
//! Wires the newline-delimited stdio [`LineTransport`] to an [`AcpServer`] so
//! HIDE can be launched as an agent by an ACP-speaking editor. The server framed
//! here negotiates capabilities, binds sessions through the host registry, and
//! projects a turn's HIDE items into ACP session updates.
//!
//! The turn handler is [`BackendTurnHandler`]: each `session/prompt` posts
//! `Intent::SubmitTurn` onto a live [`BackendHost`] via `handle_intent` (the
//! same path as `/v1/hide/intent`). Without `HIDE_MODEL_WEIGHTS` the host still
//! accepts the turn and surfaces model-offline honestly.

use std::io::{stdin, stdout, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use hide_acp::backend_host::{BackendTurnHandler, HostSessionBinder};
use hide_acp::server::AcpServer;
use hide_acp::transport::LineTransport;
use hide_acp::{HideExposure, Result};
use hide_backend::BackendHost;

fn main() -> Result<()> {
    let workspace = std::env::var_os("HIDE_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let host = Arc::new(BackendHost::open_workspace(&workspace).map_err(|e| {
        std::io::Error::other(format!("open workspace {}: {e}", workspace.display()))
    })?);

    let reader = BufReader::new(stdin().lock());
    let writer = stdout().lock();
    let transport = LineTransport::new(reader, writer);

    let handler = BackendTurnHandler::new(host.clone());
    let binder = HostSessionBinder::new(host);

    let mut server = AcpServer::new(transport, handler, binder, HideExposure::full_local());
    server.run()
}
