//! Adapters from deprecated / projection event models into the canonical
//! [`hide_core::event::Event`] envelope.
//!
//! These modules do **not** claim the source models are gone. They make the
//! mapping explicit so two live authorities cannot silently diverge.

pub mod item;
pub mod seed;
pub mod stream;
pub mod ui;

pub use item::item_to_canonical;
pub use seed::{seed_event_to_canonical, SeedFsmEvent};
pub use stream::{
    stream_done_to_canonical, stream_event_to_canonical, stream_token_to_canonical, StreamEventView,
};
pub use ui::ui_event_to_canonical;
