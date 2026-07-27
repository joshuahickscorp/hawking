//! `AcpToHide`: map an inbound ACP prompt into a HIDE turn intent.
//!
//! The reverse of [`crate::project`]. An ACP `session/prompt` names a session
//! and carries a list of content blocks. This module resolves the session to its
//! HIDE (session, thread) binding and folds the content blocks into a
//! `hide_protocol::item::UserMessage`: text blocks concatenate into the message
//! text, and resource/link/image blocks become message attachments. The result
//! is a [`HideTurnIntent`] the HIDE runtime can turn into a `turn/create`.
//! Nothing here runs a model.

use hide_protocol::ids::{SessionId, ThreadId};
use hide_protocol::item::{Attachment, UserMessage};
use hide_protocol::protocol::Method;

use crate::content::ContentBlock;
use crate::error::{AcpError, Result};
use crate::map::SessionThreadMap;
use crate::session::AcpPromptRequest;

/// The HIDE-side intent an ACP prompt maps to: which turn method to invoke, the
/// resolved session/thread, and the folded user message.
#[derive(Debug, Clone, PartialEq)]
pub struct HideTurnIntent {
    /// Always [`Method::TurnCreate`]: a prompt opens a new turn.
    pub method: Method,
    pub session: SessionId,
    pub thread: ThreadId,
    pub message: UserMessage,
}

/// Maps ACP prompts into HIDE turn intents against a session binding registry.
#[derive(Debug)]
pub struct AcpToHide<'a> {
    map: &'a SessionThreadMap,
}

impl<'a> AcpToHide<'a> {
    pub fn new(map: &'a SessionThreadMap) -> Self {
        Self { map }
    }

    /// Map a `session/prompt` request into a HIDE turn intent.
    ///
    /// Errors with [`AcpError::UnknownSession`] if the ACP session is not bound,
    /// and [`AcpError::EmptyPrompt`] if the prompt carries no text or resource
    /// content.
    pub fn map_prompt(&self, req: &AcpPromptRequest) -> Result<HideTurnIntent> {
        let binding = self
            .map
            .hide_for(&req.session_id)
            .ok_or_else(|| AcpError::UnknownSession(req.session_id.to_string()))?;

        let message = fold_prompt(&req.prompt)?;

        Ok(HideTurnIntent {
            method: Method::TurnCreate,
            session: binding.session.clone(),
            thread: binding.thread.clone(),
            message,
        })
    }
}

/// Fold ACP content blocks into a HIDE user message. Text blocks join with a
/// single newline; resource, link, and image blocks become attachments.
///
/// DEFERRED_MODEL_REQUIRED: attachment content hashing and sizing need the
/// actual bytes (a live fetch of the referenced resource). The model-free
/// boundary records the address and media type and leaves the hash empty and the
/// size zero; a runtime fills them when it fetches the resource.
fn fold_prompt(blocks: &[ContentBlock]) -> Result<UserMessage> {
    let mut texts: Vec<String> = Vec::new();
    let mut attachments: Vec<Attachment> = Vec::new();

    for (i, block) in blocks.iter().enumerate() {
        match block {
            ContentBlock::Text { text } => texts.push(text.clone()),
            ContentBlock::ResourceLink {
                uri,
                name,
                mime_type,
            } => {
                attachments.push(Attachment {
                    id: name.clone().unwrap_or_else(|| uri.clone()),
                    hash: String::new(),
                    size_bytes: 0,
                    media_type: mime_type.clone(),
                });
            }
            ContentBlock::Resource { resource } => {
                // An embedded resource carries its text inline; fold that text
                // into the message and record the address as an attachment.
                if let Some(t) = &resource.text {
                    texts.push(t.clone());
                }
                attachments.push(Attachment {
                    id: resource.uri.clone(),
                    hash: String::new(),
                    size_bytes: 0,
                    media_type: resource.mime_type.clone(),
                });
            }
            ContentBlock::Image { mime_type, .. } => {
                attachments.push(Attachment {
                    id: format!("image_{i}"),
                    hash: String::new(),
                    size_bytes: 0,
                    media_type: Some(mime_type.clone()),
                });
            }
        }
    }

    if texts.is_empty() && attachments.is_empty() {
        return Err(AcpError::EmptyPrompt);
    }

    Ok(UserMessage {
        text: texts.join("\n"),
        attachments,
    })
}
