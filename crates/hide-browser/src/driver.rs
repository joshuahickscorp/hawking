//! The browser driver trait and a deterministic replay driver.
//!
//! [`BrowserDriver`] is the seam every browser backend implements: the action
//! verbs (navigate, click, fill) and the observer verbs (screenshot, dom,
//! accessibility, console, network). Backends return owned evidence so the trait
//! stays object-safe and implementable by both a replay driver and (later) a
//! live one.
//!
//! [`ReplayDriver`] is the only backend in this crate. It plays a recorded
//! [`BrowserSession`] back, step by step, verifying that each driver call
//! matches the next recorded step. The action verbs advance a cursor and return
//! the played step; the observer verbs read the evidence captured at the current
//! step. This makes a recorded trace a strict, deterministic contract that a
//! test can drive without a browser.
//!
//! DEFERRED_MODEL_REQUIRED-adjacent: a real chromium/CDP backend that produces
//! these records from a live page is out of scope for this crate and is NOT
//! implemented or claimed here. It would implement this same trait, most likely
//! async, wrapping the transport; nothing about the evidence model depends on
//! it.

use crate::a11y::AccessibilityTree;
use crate::dom::DomSnapshot;
use crate::error::{BrowserError, Result};
use crate::evidence::{
    BrowserAction, BrowserSession, BrowserStep, ConsoleEvent, ElementSelector, NetworkEvent,
};
use crate::ids::ArtifactRef;

/// The seam a browser backend implements. Action verbs mutate the page and
/// return the resulting evidence step; observer verbs read evidence about the
/// current step without changing it.
pub trait BrowserDriver {
    /// Navigate to a URL. Returns the resulting evidence step.
    fn navigate(&mut self, url: &str) -> Result<BrowserStep>;

    /// Click the element the selector locates. Returns the resulting step.
    fn click(&mut self, selector: &ElementSelector) -> Result<BrowserStep>;

    /// Fill the located element with `value`. Returns the resulting step.
    fn fill(&mut self, selector: &ElementSelector, value: &str) -> Result<BrowserStep>;

    /// The screenshot reference for the current step.
    fn screenshot(&self) -> Result<ArtifactRef>;

    /// The DOM snapshot captured at the current step.
    fn dom(&self) -> Result<DomSnapshot>;

    /// The accessibility tree captured at the current step.
    fn accessibility(&self) -> Result<AccessibilityTree>;

    /// The console events captured at the current step.
    fn console(&self) -> Result<Vec<ConsoleEvent>>;

    /// The network events captured at the current step.
    fn network(&self) -> Result<Vec<NetworkEvent>>;
}

/// A deterministic driver that replays a recorded [`BrowserSession`].
///
/// The driver holds the recorded steps and a cursor at the last-played step
/// (`None` before the first call). Each action verb consumes the next recorded
/// step, checks that the call matches what was recorded, advances the cursor,
/// and returns the played step. Observer verbs read the current step.
#[derive(Debug, Clone)]
pub struct ReplayDriver {
    steps: Vec<BrowserStep>,
    /// Index of the last-played step; `None` before anything is played.
    cursor: Option<usize>,
}

impl ReplayDriver {
    pub fn new(session: BrowserSession) -> Self {
        Self {
            steps: session.steps,
            cursor: None,
        }
    }

    /// The index of the step a subsequent action verb would play.
    fn next_index(&self) -> usize {
        match self.cursor {
            None => 0,
            Some(i) => i + 1,
        }
    }

    /// The step most recently played, for the observer verbs.
    pub fn current_step(&self) -> Option<&BrowserStep> {
        self.cursor.and_then(|i| self.steps.get(i))
    }

    fn require_current(&self) -> Result<&BrowserStep> {
        self.current_step().ok_or(BrowserError::NoCurrentStep)
    }

    /// Whether every recorded step has been played.
    pub fn is_exhausted(&self) -> bool {
        self.next_index() >= self.steps.len()
    }

    /// The number of recorded steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Take the next step, enforcing that `requested` is the verb the recorded
    /// action expects. `matches` decides whether the recorded action is the one
    /// the caller asked for; `describe_expected` renders the recorded action for
    /// a mismatch error.
    fn play_next(
        &mut self,
        requested: &'static str,
        matches: impl FnOnce(&BrowserStep) -> bool,
    ) -> Result<BrowserStep> {
        let idx = self.next_index();
        let step = self
            .steps
            .get(idx)
            .ok_or(BrowserError::ReplayExhausted { requested })?;
        if !matches(step) {
            return Err(BrowserError::ReplayMismatch {
                index: idx,
                requested,
                expected: step.action.label().to_string(),
                detail: format!("recorded url={:?} action={}", step.url, step.action.label()),
            });
        }
        self.cursor = Some(idx);
        Ok(self.steps[idx].clone())
    }
}

impl BrowserDriver for ReplayDriver {
    fn navigate(&mut self, url: &str) -> Result<BrowserStep> {
        self.play_next("navigate", |step| {
            matches!(step.action, BrowserAction::Navigate) && step.url == url
        })
    }

    fn click(&mut self, selector: &ElementSelector) -> Result<BrowserStep> {
        self.play_next("click", |step| {
            matches!(step.action, BrowserAction::Click)
                && step
                    .selected_element
                    .as_ref()
                    .map(|e| e.same_target(selector))
                    .unwrap_or(false)
        })
    }

    fn fill(&mut self, selector: &ElementSelector, value: &str) -> Result<BrowserStep> {
        self.play_next("fill", |step| {
            matches!(&step.action, BrowserAction::Fill { value: v } if v == value)
                && step
                    .selected_element
                    .as_ref()
                    .map(|e| e.same_target(selector))
                    .unwrap_or(false)
        })
    }

    fn screenshot(&self) -> Result<ArtifactRef> {
        let step = self.require_current()?;
        step.screenshot_ref
            .clone()
            .ok_or(BrowserError::MissingEvidence { what: "screenshot" })
    }

    fn dom(&self) -> Result<DomSnapshot> {
        Ok(self.require_current()?.dom_snapshot.clone())
    }

    fn accessibility(&self) -> Result<AccessibilityTree> {
        Ok(self.require_current()?.accessibility_tree.clone())
    }

    fn console(&self) -> Result<Vec<ConsoleEvent>> {
        Ok(self.require_current()?.console_events.clone())
    }

    fn network(&self) -> Result<Vec<NetworkEvent>> {
        Ok(self.require_current()?.network_events.clone())
    }
}
