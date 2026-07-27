//! A thin async client over a [`Transport`].
//!
//! The client builds `hide-protocol` [`Request`] envelopes around typed
//! [`Method`] values, sends them through a [`Transport`], and decodes the typed
//! result. The typed helpers ([`Client::session_start`], [`Client::turn_start`],
//! [`Client::item_subscribe`], [`Client::thread_fork`]) are the ergonomic
//! surface a frontend or an integration would call.
//!
//! # DEFERRED_MODEL_REQUIRED
//!
//! The [`Transport`] trait is the seam to a live agent server. The real
//! loopback / HTTP transport that carries these requests to a running
//! `hide-serve` and streams notifications back needs a model-bearing server to
//! answer them, so it is out of scope here and not implemented. Everything in
//! this crate exercises the client over [`MockTransport`], a deterministic
//! in-memory stand-in with a fixed routing table and a preloaded notification
//! queue. No socket is opened and no model is run.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

use hide_protocol::ids::RequestId;
use hide_protocol::protocol::{Method, Notification, Request, Response, RpcError};

/// Errors the SDK surfaces to a caller.
#[derive(Debug, Error)]
pub enum SdkError {
    /// The transport itself failed (connection, encode, timeout). The real
    /// transport is DEFERRED_MODEL_REQUIRED; the mock never returns this.
    #[error("transport error: {0}")]
    Transport(String),

    /// The server answered with a protocol-level error envelope.
    #[error("server returned an error for {method}: {code} {message}")]
    Rpc {
        method: String,
        code: i32,
        message: String,
    },

    /// The response envelope carried neither a result nor an error.
    #[error("no result in the response to {method}")]
    MissingResult { method: String },

    /// The result JSON did not decode into the expected typed result.
    #[error("could not decode the result for {method}: {source}")]
    Decode {
        method: String,
        #[source]
        source: serde_json::Error,
    },

    /// The mock transport had no canned response registered for the method.
    #[error("no mock handler registered for method {0}")]
    Unhandled(String),
}

/// The transport seam: send one [`Request`] and await its [`Response`], and
/// drain any [`Notification`]s the server has pushed.
///
/// DEFERRED_MODEL_REQUIRED: the production implementation talks to a live agent
/// server. Only [`MockTransport`] is provided here.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a request and await its response envelope.
    async fn request(&self, request: Request) -> Result<Response, SdkError>;

    /// Drain and return any notifications buffered for the client. A real
    /// streaming transport delivers these as they arrive; the mock returns
    /// whatever was preloaded.
    async fn notifications(&self) -> Vec<Notification>;
}

/// The typed client. Generic over the [`Transport`] so tests use
/// [`MockTransport`] and production would use the deferred real transport.
pub struct Client<T: Transport> {
    transport: T,
    next_id: AtomicU64,
}

impl<T: Transport> Client<T> {
    /// Wrap a transport. Request ids are minted deterministically starting at
    /// `req_1`, so a fixed sequence of calls produces a fixed set of ids.
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            next_id: AtomicU64::new(1),
        }
    }

    /// Borrow the underlying transport (handy for asserting what the mock saw).
    pub fn transport(&self) -> &T {
        &self.transport
    }

    fn mint_id(&self) -> RequestId {
        let n = self.next_id.fetch_add(1, Ordering::SeqCst);
        RequestId::from(format!("req_{n}"))
    }

    /// The core call: build a [`Request`] around a [`Method`] and params, send
    /// it, and decode the typed result `R`. Every typed helper routes through
    /// here.
    pub async fn call<R: DeserializeOwned>(
        &self,
        method: Method,
        params: Value,
    ) -> Result<R, SdkError> {
        let request = Request {
            id: self.mint_id(),
            method,
            params,
        };
        let response = self.transport.request(request).await?;

        if let Some(err) = response.error {
            return Err(SdkError::Rpc {
                method: method.as_str().to_string(),
                code: err.code,
                message: err.message,
            });
        }
        let result = response.result.ok_or_else(|| SdkError::MissingResult {
            method: method.as_str().to_string(),
        })?;
        serde_json::from_value(result).map_err(|source| SdkError::Decode {
            method: method.as_str().to_string(),
            source,
        })
    }

    // -- typed helpers -----------------------------------------------------

    /// Start a session in a workspace (`session/new`). Returns the created
    /// [`Session`](hide_protocol::model::Session).
    pub async fn session_start<R: DeserializeOwned>(
        &self,
        workspace: &str,
        title: Option<&str>,
    ) -> Result<R, SdkError> {
        self.call(
            Method::SessionNew,
            serde_json::json!({ "workspace": workspace, "title": title }),
        )
        .await
    }

    /// Start a turn on a thread (`turn/create`). Returns the created
    /// [`Turn`](hide_protocol::model::Turn).
    pub async fn turn_start<R: DeserializeOwned>(
        &self,
        thread: &str,
        text: &str,
    ) -> Result<R, SdkError> {
        self.call(
            Method::TurnCreate,
            serde_json::json!({ "thread": thread, "text": text }),
        )
        .await
    }

    /// Fork a thread (`thread/fork`). Returns the new
    /// [`Thread`](hide_protocol::model::Thread).
    pub async fn thread_fork<R: DeserializeOwned>(&self, thread: &str) -> Result<R, SdkError> {
        self.call(Method::ThreadFork, serde_json::json!({ "thread": thread }))
            .await
    }

    /// Subscribe to a thread's item stream (`item/subscribe`). The subscribe
    /// call returns an ack (decoded and discarded); the items themselves arrive
    /// as [`Notification`]s, which this drains from the transport. A live
    /// streaming subscription is DEFERRED_MODEL_REQUIRED.
    pub async fn item_subscribe(&self, thread: &str) -> Result<Vec<Notification>, SdkError> {
        let _ack: Value = self
            .call(Method::ItemSubscribe, serde_json::json!({ "thread": thread }))
            .await?;
        Ok(self.transport.notifications().await)
    }
}

/// A deterministic in-memory transport for tests. Register a canned result (or
/// error) per method, preload notifications, and inspect the requests it
/// received. No network, no model.
#[derive(Default)]
pub struct MockTransport {
    results: Mutex<HashMap<String, Value>>,
    errors: Mutex<HashMap<String, RpcError>>,
    notifications: Mutex<Vec<Notification>>,
    received: Mutex<Vec<Request>>,
}

impl MockTransport {
    /// A transport with no handlers. Add them with [`MockTransport::on`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the canned result a method returns. Builder-style.
    pub fn on(self, method: Method, result: Value) -> Self {
        self.results
            .lock()
            .unwrap()
            .insert(method.as_str().to_string(), result);
        self
    }

    /// Register a protocol error a method returns instead of a result.
    pub fn on_error(self, method: Method, error: RpcError) -> Self {
        self.errors
            .lock()
            .unwrap()
            .insert(method.as_str().to_string(), error);
        self
    }

    /// Preload a notification the client will drain on the next
    /// [`Transport::notifications`] call.
    pub fn push_notification(self, notification: Notification) -> Self {
        self.notifications.lock().unwrap().push(notification);
        self
    }

    /// Every request the transport has received so far, in order.
    pub fn received(&self) -> Vec<Request> {
        self.received.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn request(&self, request: Request) -> Result<Response, SdkError> {
        let key = request.method.as_str().to_string();
        self.received.lock().unwrap().push(request.clone());

        if let Some(err) = self.errors.lock().unwrap().get(&key).cloned() {
            return Ok(Response {
                id: request.id,
                result: None,
                error: Some(err),
            });
        }
        match self.results.lock().unwrap().get(&key).cloned() {
            Some(result) => Ok(Response {
                id: request.id,
                result: Some(result),
                error: None,
            }),
            None => Err(SdkError::Unhandled(key)),
        }
    }

    async fn notifications(&self) -> Vec<Notification> {
        std::mem::take(&mut *self.notifications.lock().unwrap())
    }
}
