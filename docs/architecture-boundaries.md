# Architecture boundaries

## Hawking ↔ HIDE

Sixteen cross-edges, measured from the Cargo manifests.

### hawking → HIDE (6 edges) — the inversion

| edge | what it consumes | classification |
| --- | --- | --- |
| `hawking-context → hide-core` | `error::Result`, `ids::ModelId` | misplaced shared primitive |
| `hawking-index → hide-core` | `HideError`, `Result` | misplaced shared primitive |
| `hawking-orch → hide-core` | `error::HideError::Config`, `Result` | misplaced shared primitive |
| `hawking-research → hide-core` | `BlobStore`, `FileBlobStore`, `HideError` | misplaced shared primitive |
| `hawking-events → hide-core` | `api::UiEvent`, `event::Event` | **UI concern** |
| `hawking-events → hide-protocol` | protocol types | UI concern |

Four of the six consume only an error type, an id type and a blob store. None
of those is a product concern: a `ModelId` and a `Result` are lower-level than
the IDE that happens to own them today. They are named `Hide*` for historical
reasons, not architectural ones.

The sixth is the real inversion. `hawking-events` uses `hide_core::api::UiEvent`
in 31 places, and that is what drags HIDE into the MVP triangle:

```
hawking-serve → hawking-adapters → hawking-events → hide-core + hide-protocol
```

`hawking-adapters` genuinely uses `hawking_events` (12 call sites), so the chain
is load-bearing rather than accidental. A serving runtime depending on a UI
event vocabulary is wrong, but it is 31 uses of a shared vocabulary, not a
stray import — cutting it is a deliberate change, not a surgical one.

**Smallest real cut**, when it is worth doing: move `error`, `ids` and
`BlobStore` out of `hide-core` into a neutral low-level crate. That removes four
of the six edges and leaves exactly one inversion — the event vocabulary — which
can then be judged on its own.

### HIDE → hawking (10 edges) — legitimate

`hide-backend`, `hide-kernel` and `hide-fleet` depend on `hawking-context`,
`hawking-events`, `hawking-index`, `hawking-orch`, `hawking-research` and
`hawking-speculate`. A product built on the runtime is the right direction.

## MVP closure

`{hawking-core, hawking, hawking-serve}` closes over nine crates:

```
hawking · hawking-adapters · hawking-bench · hawking-core
hawking-events · hawking-serve · hawking-speculate
hide-core · hide-protocol
```

The last two are present only through the chain above.
