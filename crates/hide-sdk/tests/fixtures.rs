//! Event fixtures round-trip through hide-protocol serde (test group d).
//!
//! Each canonical fixture is serialized and parsed back through the ONE schema
//! authority, proving the fixtures are faithful protocol shapes. The committed
//! `fixtures/events.json` golden is also regenerated-and-compared, and every
//! entry in it reparses into its protocol type. No model, no network.

use hide_protocol::item::Item;
use hide_protocol::protocol::Notification;

#[test]
fn item_fixtures_round_trip_through_serde() {
    for (name, item) in hide_sdk::fixtures::item_fixtures() {
        let json = serde_json::to_value(&item).expect("serialize item fixture");
        let back: Item = serde_json::from_value(json).expect("deserialize item fixture");
        assert_eq!(back, item, "item fixture `{name}` must round-trip");
    }
}

#[test]
fn notification_fixtures_round_trip_through_serde() {
    for (name, notification) in hide_sdk::fixtures::notification_fixtures() {
        let json = serde_json::to_value(&notification).expect("serialize notification fixture");
        let back: Notification =
            serde_json::from_value(json).expect("deserialize notification fixture");
        assert_eq!(back, notification, "notification fixture `{name}` must round-trip");
    }
}

#[test]
fn events_golden_is_stable_and_every_entry_reparses() {
    let generated = hide_sdk::fixtures::events_json();
    let golden = include_str!("../fixtures/events.json");
    assert_eq!(
        generated, golden,
        "event fixtures drifted; regenerate with `cargo run -p hide-sdk --bin hide-sdk-codegen`"
    );

    // Every entry in the committed golden reparses into its protocol type,
    // so the file cannot hold a shape hide-protocol would reject.
    let bundle: serde_json::Value = serde_json::from_str(golden).expect("golden is valid JSON");
    for (name, value) in bundle["items"].as_object().expect("items object") {
        serde_json::from_value::<Item>(value.clone())
            .unwrap_or_else(|e| panic!("item golden `{name}` must parse: {e}"));
    }
    for (name, value) in bundle["notifications"].as_object().expect("notifications object") {
        serde_json::from_value::<Notification>(value.clone())
            .unwrap_or_else(|e| panic!("notification golden `{name}` must parse: {e}"));
    }
}
