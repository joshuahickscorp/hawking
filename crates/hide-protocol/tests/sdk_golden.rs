const REGEN: &str =
    "regenerate with `cargo run -p hide-sdk --bin hide-sdk-codegen` after an intended protocol change";
#[test]
fn json_schema_golden_is_stable() {
    let generated = hide_protocol::sdk::schema::protocol_schema_json();
    let golden = include_str!("../generated/protocol.schema.json");
    assert_eq!(generated, golden, "protocol JSON Schema drifted; {REGEN}");
}
#[test]
fn typescript_golden_is_stable() {
    let generated = hide_protocol::sdk::ts::protocol_typescript();
    let golden = include_str!("../goldens/protocol.d.ts");
    assert_eq!(generated, golden, "generated TypeScript drifted; {REGEN}");
}
#[test]
fn command_catalog_json_golden_is_stable() {
    let generated = hide_protocol::sdk::command::command_catalog_json();
    let golden = include_str!("../generated/command_catalog.json");
    assert_eq!(generated, golden, "command catalog JSON drifted; {REGEN}");
}
#[test]
fn command_typescript_golden_is_stable() {
    let generated = hide_protocol::sdk::command::command_typescript();
    let golden = include_str!("../goldens/commands.d.ts");
    assert_eq!(generated, golden, "command TypeScript drifted; {REGEN}");
}
#[test]
fn generation_is_deterministic_across_runs() {
    assert_eq!(
        hide_protocol::sdk::schema::protocol_schema_json(),
        hide_protocol::sdk::schema::protocol_schema_json()
    );
    assert_eq!(
        hide_protocol::sdk::ts::protocol_typescript(),
        hide_protocol::sdk::ts::protocol_typescript()
    );
    assert_eq!(
        hide_protocol::sdk::command::command_catalog_json(),
        hide_protocol::sdk::command::command_catalog_json()
    );
    assert_eq!(
        hide_protocol::sdk::command::command_typescript(),
        hide_protocol::sdk::command::command_typescript()
    );
}
#[test]
fn generated_artifacts_carry_no_en_or_em_dashes() {
    for artifact in [
        hide_protocol::sdk::schema::protocol_schema_json(),
        hide_protocol::sdk::ts::protocol_typescript(),
        hide_protocol::sdk::command::command_catalog_json(),
        hide_protocol::sdk::command::command_typescript(),
        hide_protocol::sdk::fixtures::events_json(),
    ] {
        assert!(!artifact.contains('\u{2013}') && !artifact.contains('\u{2014}'));
    }
}
#[test]
fn typescript_covers_the_core_interfaces() {
    let ts = hide_protocol::sdk::ts::protocol_typescript();
    for iface in [
        "export interface Session {",
        "export interface Thread {",
        "export interface Turn {",
        "export interface Plan {",
        "export interface Agent {",
        "export interface InitializeRequest {",
        "export interface InitializeResult {",
    ] {
        assert!(ts.contains(iface), "generated TS is missing `{iface}`");
    }
}
#[test]
fn typescript_covers_a_string_literal_union() {
    let ts = hide_protocol::sdk::ts::protocol_typescript();
    assert!(ts.contains("export type Method = \"workspace/create\""));
    assert!(ts.contains("\"thread/fork\""), "a Method member is missing");
    assert!(ts.contains("export type SessionStatus = \"active\" | \"idle\" | \"closed\";"));
}
#[test]
fn typescript_covers_a_tagged_union() {
    let ts = hide_protocol::sdk::ts::protocol_typescript();
    assert!(ts.contains("export type Item = { kind: \"user_message\"; payload: UserMessage; }"));
    assert!(
        ts.contains("export type Notification = { method: \"session/updated\";"),
        "Notification tagged union is wrong"
    );
}
#[test]
fn typescript_covers_optionals_refs_maps_and_value() {
    let ts = hide_protocol::sdk::ts::protocol_typescript();
    assert!(
        ts.contains("capsule?: StateCapsuleRef | null;"),
        "nullable ref field is wrong"
    );
    assert!(
        ts.contains("output: unknown;"),
        "Value should map to unknown"
    );
    assert!(ts.contains("experimental?: Record<string, unknown>;"));
    assert!(ts.contains("methods?: Method[];"), "ref array is wrong");
}
