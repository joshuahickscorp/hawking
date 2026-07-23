//! hide-acp-server: the ACP agent entrypoint.
//!
//! Wires the newline-delimited stdio [`LineTransport`] to an [`AcpServer`] so
//! HIDE can be launched as an agent by an ACP-speaking editor. The server framed
//! here negotiates capabilities, binds sessions, and projects a turn's HIDE items
//! into ACP session updates.
//!
//! The turn handler bound below is the [`DeferredTurnHandler`]: it runs NO model
//! and yields an honest blocker so an editor sees exactly why no turn executed.
//! Binding the real HIDE backend as the [`TurnHandler`] is
//! DEFERRED_MODEL_REQUIRED (RIP doctrine: this crate stays model-free).

use std::io::{stdin, stdout, BufReader};

use hide_acp::server::{AcpServer, CountingBinder, DeferredTurnHandler};
use hide_acp::transport::LineTransport;
use hide_acp::{HideExposure, Result};

fn main() -> Result<()> {
    let reader = BufReader::new(stdin().lock());
    let writer = stdout().lock();
    let transport = LineTransport::new(reader, writer);

    // DEFERRED_MODEL_REQUIRED: replace DeferredTurnHandler with the real
    // backend-bound handler that executes a turn with a live model.
    let handler = DeferredTurnHandler;
    let binder = CountingBinder::default();

    let mut server = AcpServer::new(transport, handler, binder, HideExposure::full_local());
    server.run()
}
