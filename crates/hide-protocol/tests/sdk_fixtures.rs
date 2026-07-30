use hide_protocol::item::Item;
use hide_protocol::protocol::Notification;
#[test]
fn item_fixtures_round_trip_through_serde() {
    for (name, item) in hide_protocol::sdk::fixtures::item_fixtures() {
        let json = serde_json::to_value(&item).expect("serialize item fixture");
        let back: Item = serde_json::from_value(json).expect("deserialize item fixture");
        assert_eq!(back, item, "item fixture `{name}` must round-trip");
    }
}
#[test]
fn notification_fixtures_round_trip_through_serde() {
    for (name, notification) in hide_protocol::sdk::fixtures::notification_fixtures() {
        let json = serde_json::to_value(&notification).expect("serialize notification fixture");
        let back: Notification =
            serde_json::from_value(json).expect("deserialize notification fixture");
        assert_eq!(
            back, notification,
            "notification fixture `{name}` must round-trip"
        );
    }
}
#[test]
fn events_golden_is_stable_and_every_entry_reparses() {
    let generated = hide_protocol::sdk::fixtures::events_json();
    let golden = include_str!("../fixtures/events.json");
    assert_eq!(generated, golden);
    let bundle: serde_json::Value = serde_json::from_str(golden).expect("golden is valid JSON");
    for (name, value) in bundle["items"].as_object().expect("items object") {
        serde_json::from_value::<Item>(value.clone())
            .unwrap_or_else(|e| panic!("item golden `{name}` must parse: {e}"));
    }
    for (name, value) in bundle["notifications"]
        .as_object()
        .expect("notifications object")
    {
        serde_json::from_value::<Notification>(value.clone())
            .unwrap_or_else(|e| panic!("notification golden `{name}` must parse: {e}"));
    }
}
