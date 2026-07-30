Live HIDE opens SqliteMemoryStore at .hide/memory/memory.db and ClassedMemorySystem at
classes.db + user.db. This fixture records write+forget as JSONL so the rebuild can see
the semantic content without requiring a live sqlite write from this repair lane.
COULD_NOT_PRODUCE: memory.db / classes.db binary via hide-backend open_workspace in this
lane without adding a throwaway binary to the crate graph; JSONL mirrors the record shape.
