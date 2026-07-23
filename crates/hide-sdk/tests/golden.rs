//! Golden-file stability + TypeScript coverage (test groups a and b).
//!
//! The schema bundle and the generated TypeScript are regenerated in memory and
//! compared byte-for-byte against the committed goldens. A protocol change that
//! would break the frontend fails here until the goldens are refreshed with
//! `cargo run -p hide-sdk --bin hide-sdk-codegen`. No model, no network.

const REGEN: &str =
    "regenerate with `cargo run -p hide-sdk --bin hide-sdk-codegen` after an intended protocol change";

// -- (a) golden-file stability ---------------------------------------------

#[test]
fn json_schema_golden_is_stable() {
    let generated = hide_sdk::schema::protocol_schema_json();
    let golden = include_str!("../goldens/protocol.schema.json");
    assert_eq!(generated, golden, "protocol JSON Schema drifted; {REGEN}");
}

#[test]
fn typescript_golden_is_stable() {
    let generated = hide_sdk::ts::protocol_typescript();
    let golden = include_str!("../goldens/protocol.d.ts");
    assert_eq!(generated, golden, "generated TypeScript drifted; {REGEN}");
}

#[test]
fn command_catalog_json_golden_is_stable() {
    let generated = hide_sdk::command::command_catalog_json();
    let golden = include_str!("../goldens/command_catalog.json");
    assert_eq!(generated, golden, "command catalog JSON drifted; {REGEN}");
}

#[test]
fn command_typescript_golden_is_stable() {
    let generated = hide_sdk::command::command_typescript();
    let golden = include_str!("../goldens/commands.d.ts");
    assert_eq!(generated, golden, "command TypeScript drifted; {REGEN}");
}

#[test]
fn generation_is_deterministic_across_runs() {
    assert_eq!(
        hide_sdk::schema::protocol_schema_json(),
        hide_sdk::schema::protocol_schema_json(),
        "schema generation must be deterministic"
    );
    assert_eq!(
        hide_sdk::ts::protocol_typescript(),
        hide_sdk::ts::protocol_typescript(),
        "TypeScript generation must be deterministic"
    );
    assert_eq!(
        hide_sdk::command::command_catalog_json(),
        hide_sdk::command::command_catalog_json(),
        "command catalog generation must be deterministic"
    );
    assert_eq!(
        hide_sdk::command::command_typescript(),
        hide_sdk::command::command_typescript(),
        "command TypeScript generation must be deterministic"
    );
}

#[test]
fn generated_artifacts_carry_no_en_or_em_dashes() {
    for artifact in [
        hide_sdk::schema::protocol_schema_json(),
        hide_sdk::ts::protocol_typescript(),
        hide_sdk::command::command_catalog_json(),
        hide_sdk::command::command_typescript(),
        hide_sdk::fixtures::events_json(),
    ] {
        assert!(
            !artifact.contains('\u{2013}') && !artifact.contains('\u{2014}'),
            "generated artifact must use plain hyphens only"
        );
    }
}

// -- (b) the generated TypeScript covers the core types --------------------

#[test]
fn typescript_covers_the_core_interfaces() {
    let ts = hide_sdk::ts::protocol_typescript();
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
    let ts = hide_sdk::ts::protocol_typescript();
    // Method is a unit enum -> a string-literal union.
    assert!(
        ts.contains("export type Method = \"workspace/create\""),
        "Method should render as a string-literal union"
    );
    assert!(ts.contains("\"thread/fork\""), "a Method member is missing");
    // SessionStatus is a small snake_case enum.
    assert!(
        ts.contains("export type SessionStatus = \"active\" | \"idle\" | \"closed\";"),
        "SessionStatus enum union is wrong"
    );
}

#[test]
fn typescript_covers_a_tagged_union() {
    let ts = hide_sdk::ts::protocol_typescript();
    // Item is an adjacently tagged union of { kind, payload }.
    assert!(
        ts.contains("export type Item = { kind: \"user_message\"; payload: UserMessage; }"),
        "Item tagged union is wrong"
    );
    // Notification too, with an inline params object.
    assert!(
        ts.contains("export type Notification = { method: \"session/updated\";"),
        "Notification tagged union is wrong"
    );
}

#[test]
fn typescript_covers_optionals_refs_maps_and_value() {
    let ts = hide_sdk::ts::protocol_typescript();
    // Option<Ref> -> `Ref | null`, and the field is optional.
    assert!(
        ts.contains("capsule?: StateCapsuleRef | null;"),
        "nullable ref field is wrong"
    );
    // serde_json::Value -> `unknown`.
    assert!(ts.contains("output: unknown;"), "Value should map to unknown");
    // BTreeMap<String, Value> -> Record<string, unknown>.
    assert!(
        ts.contains("experimental?: Record<string, unknown>;"),
        "map field should map to Record"
    );
    // Vec<Ref> -> `Ref[]`.
    assert!(ts.contains("methods?: Method[];"), "ref array is wrong");
}
