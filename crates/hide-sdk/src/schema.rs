//! Schema export: emit the protocol JSON Schema from the ONE source.
//!
//! hide-protocol derives schemars on every wire type. This module collects the
//! top protocol types into a single, deterministic JSON Schema bundle (an
//! OpenAPI-`components`-style document with a shared `definitions` map), so the
//! whole protocol is one artifact that codegen, contract tests, and a published
//! schema bundle all render from. Because the schemas come from the Rust types,
//! the bundle can never silently drift from the code.
//!
//! Determinism: schemars' `definitions` and each object's `properties` /
//! `required` are backed by `BTreeMap`/`BTreeSet`, and serde_json here is built
//! without `preserve_order`, so the serialized bundle is byte-stable run to run.
//! That stability is what makes the golden-file test meaningful.

use schemars::gen::SchemaGenerator;
use schemars::JsonSchema;
use serde_json::{Map, Value};

use hide_protocol::{
    Agent, InitializeRequest, InitializeResult, Item, Method, Notification, Plan, Session, Thread,
    Turn, PROTOCOL_VERSION,
};

/// The top protocol types the bundle roots at, in a fixed order. These are the
/// entry points a frontend or an external client cares about; every other type
/// they reference is pulled into `definitions` transitively.
///
/// "Initialize" from the Bible list expands to both halves of the handshake:
/// [`InitializeRequest`] and [`InitializeResult`].
pub const ROOT_TYPE_NAMES: &[&str] = &[
    "Session",
    "Thread",
    "Turn",
    "Item",
    "Method",
    "Notification",
    "InitializeRequest",
    "InitializeResult",
    "Plan",
    "Agent",
];

/// Register one root type into the generator and record its schema name.
fn register<T: JsonSchema>(gen: &mut SchemaGenerator, roots: &mut Vec<String>) {
    // subschema_for adds T (and everything it references) to the generator's
    // definitions and returns a $ref we do not need to keep here.
    let _ = gen.subschema_for::<T>();
    roots.push(T::schema_name());
}

/// Collect every root type's schema into one shared generator and return its
/// definitions map (type name -> JSON Schema), plus the ordered root names.
///
/// The definitions map is a `BTreeMap`, so iteration and serialization are
/// deterministic.
pub fn collect_definitions() -> (Vec<String>, Map<String, Value>) {
    let mut gen = SchemaGenerator::default();
    let mut roots = Vec::new();

    register::<Session>(&mut gen, &mut roots);
    register::<Thread>(&mut gen, &mut roots);
    register::<Turn>(&mut gen, &mut roots);
    register::<Item>(&mut gen, &mut roots);
    register::<Method>(&mut gen, &mut roots);
    register::<Notification>(&mut gen, &mut roots);
    register::<InitializeRequest>(&mut gen, &mut roots);
    register::<InitializeResult>(&mut gen, &mut roots);
    register::<Plan>(&mut gen, &mut roots);
    register::<Agent>(&mut gen, &mut roots);

    let mut definitions = Map::new();
    for (name, schema) in gen.definitions().iter() {
        let value = serde_json::to_value(schema)
            .expect("a schemars Schema always serializes to serde_json::Value");
        definitions.insert(name.clone(), value);
    }

    (roots, definitions)
}

/// Collect a single root type's schema (plus everything it references) into a
/// deterministic definitions map, for a codegen surface that roots on one type
/// (the command catalog roots on [`CommandSpec`](hide_protocol::CommandSpec)).
pub fn collect_root<T: JsonSchema>() -> (Vec<String>, Map<String, Value>) {
    let mut gen = SchemaGenerator::default();
    let mut roots = Vec::new();
    register::<T>(&mut gen, &mut roots);

    let mut definitions = Map::new();
    for (name, schema) in gen.definitions().iter() {
        let value = serde_json::to_value(schema)
            .expect("a schemars Schema always serializes to serde_json::Value");
        definitions.insert(name.clone(), value);
    }
    (roots, definitions)
}

/// The full protocol JSON Schema bundle as a [`serde_json::Value`].
///
/// Shape:
///
/// ```json
/// {
///   "$schema": "http://json-schema.org/draft-07/schema#",
///   "title": "HIDE Agent Protocol",
///   "protocolVersion": "hide.agent.v1",
///   "roots": ["Session", "Thread", ...],
///   "definitions": { "Session": { ... }, ... }
/// }
/// ```
pub fn protocol_schema_bundle() -> Value {
    let (roots, definitions) = collect_definitions();

    let mut bundle = Map::new();
    bundle.insert(
        "$schema".to_string(),
        Value::String("http://json-schema.org/draft-07/schema#".to_string()),
    );
    bundle.insert(
        "title".to_string(),
        Value::String("HIDE Agent Protocol".to_string()),
    );
    bundle.insert(
        "protocolVersion".to_string(),
        Value::String(PROTOCOL_VERSION.to_string()),
    );
    bundle.insert(
        "roots".to_string(),
        Value::Array(roots.into_iter().map(Value::String).collect()),
    );
    bundle.insert("definitions".to_string(), Value::Object(definitions));

    Value::Object(bundle)
}

/// The protocol JSON Schema bundle as a stable, pretty-printed string. This is
/// the exact artifact the golden-file test pins.
pub fn protocol_schema_json() -> String {
    let mut s = serde_json::to_string_pretty(&protocol_schema_bundle())
        .expect("the bundle is plain JSON and always serializes");
    s.push('\n');
    s
}
